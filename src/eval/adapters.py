"""Detector adapters -> the common (x, y) detection interface.

Every method, ours or external, is reduced to a function that maps one image array
to a list of ``(x, y, score)`` LOCATIONS. That is the only thing the benchmark
compares; the shared photometry + fit + GT-eval layer (``benchmark_core``) does the
rest identically for everyone. A per-image detection CSV
(``image_id, x, y, score``) is the on-disk interchange format, so external tools
(cme-analysis [MATLAB], SpotMAX) that run on the user's machine can drop their
coordinates in and flow through the exact same evaluation.

Adapters implemented now:
  * ``detect_ours``      — hrnet_v1 (models/hrnet_v1/best.pt) -> decode -> (x,y,score).
  * ``detect_classical`` — scikit-image ``blob_log`` (Laplacian-of-Gaussian) on the
                           lipid channel: a standard-method baseline that runs on the
                           instance now.

External tools use ``read_detection_csv`` (the ``external_csv`` path): the user runs
cme-analysis / SpotMAX on the EXPORTED synthetic (or the real) images elsewhere,
emits ``image_id, x, y[, score]`` CSVs, and those are read back here -> identical
downstream. See ``export_for_external.py`` and EXPERIMENT.md.

``image`` for every adapter is a ``(2, H, W)`` array, channel 0 = protein, channel
1 = lipid, RAW ADU (offset included) — the same convention the detector trained on
(see ``src/eval/real_data.py``).
"""

import csv
from pathlib import Path

import numpy as np

from src.eval.matching import decode_image_array

#: CSV columns of the common detection interchange format.
CSV_COLUMNS = ('image_id', 'x', 'y', 'score')


# --------------------------------------------------------------------------- #
# ours — the trained detector                                                 #
# --------------------------------------------------------------------------- #
def detect_ours(model, cfg, device, image):
    """Run hrnet_v1 -> decode -> ``(N,3)`` array of ``(x, y, detection_score)``.

    For the CONTROLLED comparison only the (x, y) are used (fed through the shared
    photometry like every method). Our NATIVE intensities + logvar are available
    separately via ``detect_ours_native`` for the end-to-end table.
    """
    dets = decode_image_array(model, cfg, device, image)
    if not dets:
        return np.zeros((0, 3), np.float64)
    return np.array([[d['x'], d['y'], d['detection_score']] for d in dets],
                    np.float64)


def detect_ours_native(model, cfg, device, image):
    """Full native detections (the ``decode.SCHEMA_KEYS`` dicts) for the end-to-end
    table: our own (x,y) AND our own predicted lipid/protein intensity (+logvar)."""
    return decode_image_array(model, cfg, device, image)


# --------------------------------------------------------------------------- #
# classical — LoG/DoG blob detection (scikit-image)                           #
# --------------------------------------------------------------------------- #
def detect_classical(image, min_sigma=1.0, max_sigma=4.0, num_sigma=8,
                     threshold_rel=0.1, overlap=0.5):
    """``blob_log`` on the lipid channel -> ``(N,3)`` array of ``(x, y, score)``.

    A standard Laplacian-of-Gaussian blob detector. The lipid channel (index 1) is
    the size/structure channel (spots ~ d^2), so detection runs there, mirroring the
    detector's lipid-led detection. The image is per-image min-max normalized first
    (the classical method's own preprocessing); ``score`` is the LoG response
    strength (``blob_log``'s third column). ``threshold_rel`` is relative to the max
    LoG response, so it adapts across images without a hand-tuned absolute level.

    sigma range covers the calibrated PSF (sigma ~1.9 px) plus a margin; tune via
    the kwargs if needed. Requires scikit-image (a project dependency).
    """
    from skimage.feature import blob_log

    lip = np.asarray(image[1], np.float64)
    lo, hi = float(lip.min()), float(lip.max())
    norm = (lip - lo) / (hi - lo) if hi > lo else np.zeros_like(lip)
    blobs = blob_log(norm, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=None,
                     threshold_rel=threshold_rel, overlap=overlap)
    if blobs.size == 0:
        return np.zeros((0, 3), np.float64)
    # blob_log returns (row=y, col=x, sigma-or-response). With threshold_rel the
    # 3rd column is the response; use it as the score, and (x,y) = (col,row).
    ys, xs, resp = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    return np.stack([xs, ys, resp], axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# external_csv — ingest coordinates produced elsewhere                        #
# --------------------------------------------------------------------------- #
def write_detection_csv(path, per_image_xy, image_ids=None):
    """Write a per-image ``(x, y, score)`` list to the common CSV.

    ``per_image_xy`` is a list (per image) of ``(M, >=2)`` arrays; ``image_ids`` are
    matching ids (default ``0..n-1``). The CSV is the interchange any tool emits.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_ids is None:
        image_ids = list(range(len(per_image_xy)))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for img_id, arr in zip(image_ids, per_image_xy):
            arr = np.asarray(arr, np.float64).reshape(-1, arr.shape[-1] if np.ndim(arr) == 2 else 1)
            for row in np.atleast_2d(arr):
                score = float(row[2]) if row.shape[0] > 2 else float('nan')
                w.writerow([img_id, float(row[0]), float(row[1]), score])


def read_detection_csv(path):
    """Read a common detection CSV -> ``{image_id: (M,3) array of (x,y,score)}``.

    Accepts CSVs from ``write_detection_csv`` AND from external tools (cme-analysis,
    SpotMAX) as long as they have ``image_id, x, y`` columns (``score`` optional).
    ``image_id`` is kept as a string key so any naming scheme round-trips.
    """
    by_image = {}
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        cols = {c.lower(): c for c in (r.fieldnames or [])}
        if not {'image_id', 'x', 'y'} <= set(cols):
            raise ValueError(f"{path}: need columns image_id,x,y; got {r.fieldnames}")
        for row in r:
            img = str(row[cols['image_id']])
            x, y = float(row[cols['x']]), float(row[cols['y']])
            sc = float(row[cols['score']]) if 'score' in cols and row[cols['score']] not in (None, '') else float('nan')
            by_image.setdefault(img, []).append((x, y, sc))
    return {k: np.array(v, np.float64).reshape(-1, 3) for k, v in by_image.items()}
