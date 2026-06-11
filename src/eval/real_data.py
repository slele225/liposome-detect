"""Real-image loading + detection for the trained Stage-2 detector.

The detector was trained on SYNTHETIC images. This module is the bridge that
feeds REAL microscope frames through the exact same path, so the predicted
fluxes are on the scale the model expects. It is the single place the real-data
experiments (``experiments/2026-06-11_real-data-comparison/``) get their spots
from; the orchestration scripts there import from here.

Scaling convention (THE #1 risk — pinned here, documented in the experiment)
---------------------------------------------------------------------------
The synthetic generator's images INCLUDE the detector dark offset: the forward
model adds ``offset`` (the dark-frame DC level, ~138-152 ADU) and clips to the
12-bit range (``src/simulator/forward_model.py`` ``_apply_pmt_noise`` +
``np.clip(..., 0, 4095)``), and ``generator.core.serialize_image`` saves that
raw-ADU-like array straight to ``.npy``. The training dataloader
(``src/train/dataset.py``) only applies ``(img - norm_mean) / norm_std``. So the
network was trained on RAW-ADU-like inputs (offset included), centered by
``norm_mean`` / ``norm_std``.

=> Real images must be fed RAW (NO dark subtraction) so the offset is present
   exactly as in training; ``norm_mean`` / ``norm_std`` then center them.

``subtract_dark`` is exposed only as an escape hatch / robustness check. The
correct setting that matches the generator is OFF (raw); ``'auto'`` resolves to
that. Turning it ON would offset-subtract the real images, which would NOT match
training and is provided only for diagnostics.

Channel order matches end to end: the synthetic ``.npy`` is ``[protein, lipid]``
(``generator.core.generate_one_image``) with
``norm_mean = [protein, lipid] = [207.43, 326.64]``; the real 3-channel TIFF is
``[protein(488), lipid(561), transmitted]`` (``simulator.io.load_tiff_stack``),
so taking channels ``[protein, lipid]`` lines up directly. Transmitted is
dropped.

Cropping matches the rest of the pipeline: ``io.load_tiff_stack`` center-crops
to 256x256 (``io.CROP_SIZE``), the same domain the simulator generates on and
the calibration was fit on. The detector is fully convolutional, but we keep the
256 center crop so the input distribution matches training/calibration exactly.
"""

import glob
import os

import numpy as np

from src.simulator.io import analyze_dark_frames, load_tiff_stack
from src.eval.matching import decode_image_array

#: ``subtract_dark`` mode that matches the synthetic generator (offset included).
SUBTRACT_DARK_DEFAULT = 'off'


def find_dls_xlsx(sample_dir):
    """Return the single DLS ``*.xlsx`` under ``sample_dir`` (glob, never hardcode).

    DLS filenames differ per sample (EGFP: ``batch1_dls_corrected.xlsx``;
    endophilin: ``Ternary 0.5% PEGB.xlsx``), so we glob rather than assume a name.
    Raises if zero or more than one match.
    """
    hits = sorted(glob.glob(os.path.join(sample_dir, '*.xlsx')))
    if not hits:
        raise FileNotFoundError(f"no *.xlsx (DLS) found in {sample_dir}")
    if len(hits) > 1:
        raise ValueError(f"ambiguous DLS xlsx in {sample_dir}: {hits}")
    return hits[0]


def dark_offsets(sample_dir):
    """Per-channel dark-frame offsets ``[protein, lipid]`` from ``dark_frames/``.

    Used ONLY when ``subtract_dark`` is on (a diagnostic mode — the convention
    that matches training is raw/no-subtraction; see the module docstring).
    """
    res = analyze_dark_frames(os.path.join(sample_dir, 'dark_frames'))
    return np.array([res['protein']['offset'], res['lipid']['offset']], np.float32)


def _resolve_subtract(subtract_dark):
    """Map the ``{auto,on,off}`` switch to a bool. ``auto`` -> matches generator."""
    if subtract_dark in ('auto', SUBTRACT_DARK_DEFAULT):
        return False
    if subtract_dark == 'on':
        return True
    if subtract_dark == 'off':
        return False
    raise ValueError(f"subtract_dark must be auto|on|off, got {subtract_dark!r}")


def load_real_image(tif_path, subtract_dark='auto', offsets=None):
    """Load one real 3-ch TIFF -> ``(2, 256, 256)`` float32 ``[protein, lipid]``.

    Center-cropped to 256 (via ``io.load_tiff_stack``), transmitted dropped, and
    RAW ADU by default so the dark offset is present exactly as in training. If
    ``subtract_dark`` resolves to True, ``offsets`` (``[protein, lipid]``, e.g.
    from ``dark_offsets``) is subtracted per channel BEFORE normalization (a
    diagnostic that does NOT match the generator).
    """
    d = load_tiff_stack(tif_path)
    arr = np.stack([d['protein'], d['lipid']], axis=0).astype(np.float32)
    if _resolve_subtract(subtract_dark):
        if offsets is None:
            raise ValueError("subtract_dark on requires offsets ([protein, lipid])")
        arr = arr - np.asarray(offsets, np.float32)[:, None, None]
    return arr


def list_sample_images(sample_dir):
    """Sorted ``images/*.tif`` paths under a sample folder."""
    paths = sorted(glob.glob(os.path.join(sample_dir, 'images', '*.tif')) +
                   glob.glob(os.path.join(sample_dir, 'images', '*.tiff')))
    if not paths:
        raise FileNotFoundError(f"no images/*.tif under {sample_dir}")
    return paths


def detect_sample(model, cfg, device, sample_dir, subtract_dark='auto',
                  offsets=None):
    """Run the detector over every image of a sample.

    Returns a list (per image) of detection lists (the ``decode.SCHEMA_KEYS``
    dicts). Each detection is one liposome (number-weighted), so pooling across
    images gives a number distribution of detected spots.
    """
    if _resolve_subtract(subtract_dark) and offsets is None:
        offsets = dark_offsets(sample_dir)
    per_image = []
    for p in list_sample_images(sample_dir):
        arr = load_real_image(p, subtract_dark=subtract_dark, offsets=offsets)
        per_image.append(decode_image_array(model, cfg, device, arr))
    return per_image


def spots_to_logxy(dets):
    """Stack a flat detection list -> ``(log_lipid, log_protein, vlip, vpro)``.

    ``log_*`` are ``log(intensity)`` (clipped > 0); ``v* = exp(logvar)`` are the
    per-spot log-space variances consumed DIRECTLY (no /intensity**2 — the NLL
    residual is already in log space; see ``src/eval/alpha_fit.py``).
    """
    if not dets:
        z = np.zeros(0, np.float64)
        return z, z, z, z
    lip = np.array([d['lipid_intensity'] for d in dets], np.float64)
    pro = np.array([d['protein_intensity'] for d in dets], np.float64)
    llv = np.array([d['lipid_intensity_logvar'] for d in dets], np.float64)
    plv = np.array([d['protein_intensity_logvar'] for d in dets], np.float64)
    Llip = np.log(np.clip(lip, 1e-6, None))
    Lpro = np.log(np.clip(pro, 1e-6, None))
    return Llip, Lpro, np.exp(llv), np.exp(plv)


def lipid_to_diameter(lipid_intensity, lipid_brightness):
    """Size proxy: invert ``lipid_amp = lipid_brightness * (d/100)**2``.

    The simulator renders a lipid spot's total flux as ``lipid_brightness *
    (d/100)**2`` (``forward_model.simulate_image``), so the detector's predicted
    lipid flux maps back to a diameter proxy
    ``d = 100 * sqrt(lipid_intensity / lipid_brightness)``. This is a PROXY (the
    network never outputs diameter); it assumes the calibrated ``lipid_brightness``
    and the d^2 area law.
    """
    li = np.clip(np.asarray(lipid_intensity, np.float64), 0.0, None)
    return 100.0 * np.sqrt(li / float(lipid_brightness))
