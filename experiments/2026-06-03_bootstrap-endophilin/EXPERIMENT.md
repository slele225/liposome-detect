# EXP 3 — Bootstrap calibration stability (25nM_endophilin)

- **Date:** 2026-06-03
- **Mode:** `bootstrap` (see `src/calibration/study.py`)
- **Status:** scaffolded; **not yet run** (runs on the 32-core VM)

## Question

How **stable** is the calibration to the particular subset of images it sees? If
we calibrate on many different random 25-image subsets of `25nM_endophilin`
(which has 55 images total), how much do the fitted parameters vary? A small
coefficient of variation (CV = std/mean) across repeats means the parameter is
well-determined by any 25-image subset; a large CV means the fit is sensitive to
which images happen to be included.

## Method

- One sample: `25nM_endophilin` (55 images available).
- **100 repeats.** Each repeat draws `d = 25` images at random **without
  replacement**, with a **distinct seed per repeat** (`base_seed + i`,
  `base_seed = 20260603`), then runs the standard lipid-only, detection-free
  calibration on that 25-image subset. The pipeline internally splits each
  subset into train/val at `val_fraction = 0.2` (≈20 train / 5 val).
- `n_trials = 200`, `n_sim_per_trial = 30`, default discrepancy weights.
- Fitted params per repeat: `lipid_brightness, psf_sigma_x, psf_sigma_y,
  psf_theta, gain, enf, optical_bg_lipid, spot_density`.
- Config: [config_snapshot/study.yaml](config_snapshot/study.yaml).

> **This is 100 independent 200-trial calibrations.** Set `n_workers` to the VM
> core count (e.g. 32) so they run in parallel; otherwise it will take a long
> time. Each worker runs single-threaded numpy (the runner pins
> OMP/MKL/OpenBLAS to 1) so `n_workers` ≈ cores is the right setting and will
> not oversubscribe.

### Analysis (`analyze.py`, runs after the calibrations)

- `figures/bootstrap_params_per_repeat.csv` — one row per repeat.
- `figures/bootstrap_summary.csv` — per-parameter mean / std / **CV** / min / max.
- `figures/bootstrap_distributions.png` — a histogram per fitted parameter, mean
  marked, with mean/std/CV annotated.
- `results.json` / `aggregated_params.csv` — written by the runner.

## Reproduce

```bash
# From the repo root, with the environment synced (uv sync).
N_WORKERS=32 bash experiments/2026-06-03_bootstrap-endophilin/run.sh
# equivalently:
bash experiments/2026-06-03_bootstrap-endophilin/run.sh 32
```

Underlying commands the wrapper runs:

```bash
python -m src.calibration.study \
  --config experiments/2026-06-03_bootstrap-endophilin/config_snapshot/study.yaml \
  --n-workers 32
python experiments/2026-06-03_bootstrap-endophilin/analyze.py
```

Outputs land in `runs/repeat_000/` … `runs/repeat_099/` (per-repeat calibration
results, plots, `trials.csv`, `convergence.png`), plus `results.json`,
`aggregated_params.csv`, `run_manifest.json`, and `figures/`.

**Re-run only the analysis** on existing `runs/` (no recalibration):
`python experiments/2026-06-03_bootstrap-endophilin/analyze.py` — it rebuilds
`results.json` from the 100 per-repeat `calibration_results.json` if missing,
then redraws the figures. Force a rebuild with
`python -m src.calibration.study --aggregate-only experiments/2026-06-03_bootstrap-endophilin`.

## Findings

All 100/100 repeats succeeded (d=25 of 55 images, a distinct subset per repeat,
200 trials each). Per-parameter coefficient of variation across repeats
(`figures/bootstrap_summary.csv`, `figures/bootstrap_distributions.png`):

| parameter         | CV     | verdict                              |
|-------------------|--------|--------------------------------------|
| psf_sigma_y       | ~11%   | well-determined                      |
| psf_sigma_x       | ~12%   | well-determined                      |
| lipid_brightness  | ~17%   | well-determined                      |
| spot_density      | ~17%   | well-determined                      |
| enf               | ~21%   | moderate                             |
| gain              | ~35%   | moderate / loose                     |
| optical_bg_lipid  | ~83%   | poorly determined                    |
| psf_theta         | ~560%  | degenerate (CV inflated; mean ≈ 0)   |

**Conclusion:** the parameters governing detection and intensity — PSF widths,
`lipid_brightness`, `spot_density` — are robust to which 25-image subset is used
(CV ≲ 17%). The noise-decomposition parameters (`gain`, `enf`) and `psf_theta`
are subset-sensitive or degenerate: `gain`/`enf` trade off (only their product is
well constrained) and `psf_theta` is ill-defined for an essentially circular PSF
(its values scatter around 0, inflating the CV). No repeats failed or stalled.
