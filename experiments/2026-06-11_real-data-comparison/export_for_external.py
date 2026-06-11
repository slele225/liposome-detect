"""Export synthetic benchmark images as plain TIFFs + a GT table for external tools.

The enabler for putting cme-analysis (MATLAB) / SpotMAX on the SAME ground-truth
footing as ours/classical: they run on the user's machine, so they need the EXACT
SAME synthetic images as standard TIFFs plus the ground truth. They emit a per-image
(x, y) CSV that comes back through ``adapters.read_detection_csv`` ->
``benchmark_core`` -> identical evaluation.

Per dataset it writes, under ``datasets/external_export/<dataset>/`` (gitignored):
  * ``tiffs/img_NNNNNN.tif`` — one 2-channel uint16 ImageJ TIFF per image, axes
    ``CYX``: **channel 0 = protein (488), channel 1 = lipid (561)**. Values are the
    raw-ADU synthetic intensities (offset included, 12-bit range) — same scale as the
    real microscope data.
  * ``ground_truth.csv`` — one row per GT spot:
    ``image_id, x, y, diameter_nm, lipid_flux, protein_flux, alpha`` where x/y are
    full-res px (origin top-left, matching the TIFF), the fluxes are TOTAL integrated
    flux in ADU, and alpha is the spot's true curvature exponent.
  * ``README_export.txt`` — the layout + the coordinate/back-import contract.

``image_id`` is the integer image index (``img_NNNNNN``); an external tool's returned
CSV must use the SAME ``image_id`` so detections map back to the right image.

    uv run python experiments/2026-06-11_real-data-comparison/export_for_external.py
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import tifffile
import yaml

import firm_calibration as fc
from _common import EXP_DIR, REPO_ROOT

EXPORT_ROOT = REPO_ROOT / 'datasets' / 'external_export'   # gitignored (datasets/*)

README = """External-tool export — liposome-detect synthetic benchmark
=========================================================
Layout
  tiffs/img_NNNNNN.tif : 2-channel uint16 ImageJ TIFF, axes CYX.
                         channel 0 = protein (488 nm)
                         channel 1 = lipid   (561 nm)
                         raw-ADU intensities (dark offset included, 12-bit range),
                         same scale/format as the real microscope data.
  ground_truth.csv     : columns image_id,x,y,diameter_nm,lipid_flux,protein_flux,alpha
                         x,y are full-resolution pixels (origin top-left, matching
                         the TIFF). *_flux are TOTAL integrated flux in ADU.

Run your detector (cme-analysis / SpotMAX) on tiffs/*.tif and emit ONE CSV with
columns:  image_id,x,y[,score]
where image_id is the integer NNNNNN of the source TIFF and x,y are the detected
spot centre in the same full-resolution pixel frame. Bring that CSV back via the
external_csv adapter (src/eval/adapters.read_detection_csv) to run the IDENTICAL
shared photometry + EIV+calibration fit + GT evaluation as ours/classical.
"""


def _dataset_dirs(which):
    """Resolve the requested datasets to (name, dataset_dir) pairs."""
    out = []
    if 'emphasis' in which:
        out.append(('bench_emphasis', REPO_ROOT / 'datasets' / 'bench_emphasis'))
    if 'dls' in which:
        out.append(('bench_dls', REPO_ROOT / 'datasets' / 'bench_dls'))
    if 'sweep' in which:
        for a in fc.ALPHAS:
            out.append((f"real_cmp_alpha_{fc._tag(a)}", fc._dataset_dir(a)))
    return out


def export_dataset(name, dataset_dir):
    img_dir = Path(dataset_dir) / 'images'
    lbl_dir = Path(dataset_dir) / 'labels'
    if not img_dir.exists():
        print(f"[export] SKIP {name}: {img_dir} missing")
        return 0
    out = EXPORT_ROOT / name
    (out / 'tiffs').mkdir(parents=True, exist_ok=True)
    (out / 'README_export.txt').write_text(README)

    rows = [['image_id', 'x', 'y', 'diameter_nm', 'lipid_flux', 'protein_flux', 'alpha']]
    n = 0
    for ip in sorted(img_dir.glob('img_*.npy')):
        image_id = ip.stem.split('_')[-1]          # NNNNNN
        arr = np.load(ip).astype(np.float32)        # (2,H,W) [protein, lipid], raw ADU
        u16 = np.clip(np.round(arr), 0, 65535).astype(np.uint16)
        tifffile.imwrite(out / 'tiffs' / f'{ip.stem}.tif', u16,
                         imagej=True, metadata={'axes': 'CYX'})
        lbl = json.loads((lbl_dir / f'{ip.stem}.json').read_text())
        for s in lbl['spots']:
            rows.append([image_id, s['x'], s['y'], s['diameter_nm'],
                         s['lipid_intensity'], s['protein_intensity'],
                         s.get('alpha_used', '')])
        n += 1
    with open(out / 'ground_truth.csv', 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"[export] {name}: {n} TIFFs + ground_truth.csv -> {out}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+',
                    default=['emphasis', 'dls', 'sweep'],
                    help='which to export: emphasis, dls, sweep (the alpha sweep).')
    args = ap.parse_args()
    pairs = _dataset_dirs(args.datasets)
    total = sum(export_dataset(n, d) for n, d in pairs)
    print(f"\n[export] DONE. {total} images exported under {EXPORT_ROOT}")
    print(f"[export] Feed datasets/external_export/<set>/tiffs/*.tif to "
          f"cme-analysis / SpotMAX; bring back image_id,x,y CSVs.")


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    main()
