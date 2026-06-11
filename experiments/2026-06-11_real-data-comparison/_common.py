"""Shared constants + arg plumbing for the real-data-comparison scripts.

Kept tiny and experiment-local: only the sample roster, default model paths, and
the synthetic-scale reference numbers the smoke check eyeballs against. All
reusable detector/loader/fit logic lives in ``src/eval`` (imported, not copied).
"""

import argparse
import sys
from pathlib import Path

# Repo root = three levels up from this file (experiments/<exp>/_common.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

# These scripts are run as file paths (not ``-m``), so the script directory — not
# the repo root — is on sys.path[0]. ``_common`` is imported FIRST by every script
# here, so inserting the repo root now makes the later ``import src.*`` lines work
# whether the script is run directly or via run.sh.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = str(REPO_ROOT / 'configs' / 'train' / 'hrnet_v1.yaml')
DEFAULT_CKPT = str(REPO_ROOT / 'models' / 'hrnet_v1' / 'best.pt')
DEFAULT_DATA_ROOT = str(REPO_ROOT / 'data')

# EGFP is the negative control: alpha = 2.0 BY CONSTRUCTION (protein binds in
# proportion to membrane area, the d^2 / curvature-insensitive limit). This is
# the real-data ground-truth anchor.
EGFP_SAMPLES = ['20nM_EGFP', '50nM_EGFP', '100nM_EGFP', '300nM_EGFP']
# Endophilin is the curvature sensor: alpha < 2 expected.
ENDO_SAMPLES = ['25nM_endophilin', '300nM_endophilin']
ALL_SAMPLES = EGFP_SAMPLES + ENDO_SAMPLES
EGFP_TRUE_ALPHA = 2.0

# Synthetic training scale, for the smoke-check sanity band (from
# configs/train/hrnet_v1.yaml): per-channel norm_mean [protein, lipid] and the
# log-space eps floors (~ dimmest real flux). Predicted *fluxes* (total ADU) are
# larger than the pixel-mean norm; the head bias is log(init_flux) with
# lipid_init_flux=5000, protein_init_flux=3000.
SYNTH_NORM_MEAN = {'protein': 207.43, 'lipid': 326.64}
SYNTH_EPS = {'protein': 62.47, 'lipid': 80.15}
# A predicted total-flux median outside this (very wide) band signals an
# order-of-magnitude normalization/offset mismatch -> WARN.
PLAUSIBLE_FLUX_BAND = (30.0, 50000.0)

# Calibration with the shared (microscope) lipid_brightness used as the
# lipid-intensity -> diameter size proxy in dls_consistency.py.
LIPID_BRIGHTNESS_CALIB = str(
    REPO_ROOT / 'experiments' / '2026-06-03_per-sample-calibration'
    / 'runs' / '20nM_EGFP' / 'calibration_results.json')


def add_model_args(ap):
    """Attach the shared --config/--ckpt/--data-root/--subtract_dark args."""
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--subtract_dark', choices=['auto', 'on', 'off'],
                    default='auto',
                    help="auto==off==matches the synthetic generator (raw, "
                         "offset included). 'on' is a diagnostic only.")
    return ap


def sample_dir(data_root, sample):
    return str(Path(data_root) / sample)
