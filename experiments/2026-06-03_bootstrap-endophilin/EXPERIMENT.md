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

## Findings

> **TODO: fill after the VM run.** (Studies are not run during scaffolding.)
> Report the per-parameter CV table and which parameters are tight (low CV,
> robust to subset) vs loose (high CV, subset-sensitive). Note any repeats that
> failed or converged poorly (see `run_manifest.json` and each repeat's
> `convergence.png`).
