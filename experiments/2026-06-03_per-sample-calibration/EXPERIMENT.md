# EXP 2 — Per-sample independent calibration

- **Date:** 2026-06-03
- **Mode:** `per_sample` (see `src/calibration/study.py`)
- **Status:** scaffolded; **not yet run** (runs on the 32-core VM)

## Question

When each biological sample is calibrated **independently** (its own
gain/PSF measurement and its own Optuna search, **not** a joint fit):

1. Which lipid-channel parameters are **setting-dependent** (move from sample to
   sample) versus **stable** (a shared microscope property)?
2. Does the fitted **gain** track the known **561 nm detector voltage**? PMT gain
   is roughly exponential in control voltage, and the voltage was turned *down*
   for brighter (higher-concentration) samples, so a physically meaningful fitted
   gain should rise with voltage.

This is the key reason the study is per-sample and not joint: the 561 detector
voltage differs per sample, so a single shared gain cannot be correct. See
[docs/decisions/2026-06-03_calibration-study-design.md](../../docs/decisions/2026-06-03_calibration-study-design.md).

## Method

- Six samples: `20nM_EGFP`, `50nM_EGFP`, `100nM_EGFP`, `300nM_EGFP`,
  `25nM_endophilin`, `300nM_endophilin`.
- Each calibration: lipid-only, detection-free, **default** discrepancy weights,
  `n_trials=200`, `n_sim_per_trial=30`, `val_fraction=0.2`, `seed=0`.
- Fitted shared params per sample: `lipid_brightness, psf_sigma_x, psf_sigma_y,
  psf_theta, gain, enf, optical_bg_lipid`; per-sample free: `spot_density`.
- Known 561 detector voltages (V): 20nM_EGFP=750, 50nM_EGFP=640,
  100nM_EGFP=630, 300nM_EGFP=580, 25nM_endophilin=670, 300nM_endophilin=600.
- Config: [config_snapshot/study.yaml](config_snapshot/study.yaml).

### Analysis (`analyze.py`, runs after the calibrations)

- `figures/cross_sample_params.csv` — fitted-parameter table, one row per sample
  (gain, psf_sigma_x, psf_sigma_y, psf_theta, enf, optical_bg_lipid,
  lipid_brightness, spot_density).
- `figures/gain_vs_voltage.png` — fitted gain vs the known 561 detector voltage.
- `figures/params_across_samples.png` — each non-gain fitted parameter across
  samples (stable vs setting-dependent).
- `results.json` / `aggregated_params.csv` — written by the runner.

## Reproduce

```bash
# From the repo root, with the environment synced (uv sync).
# n_workers: positional arg OR $N_WORKERS env var; default os.cpu_count().
N_WORKERS=32 bash experiments/2026-06-03_per-sample-calibration/run.sh
# equivalently:
bash experiments/2026-06-03_per-sample-calibration/run.sh 32
```

Underlying commands the wrapper runs:

```bash
python -m src.calibration.study \
  --config experiments/2026-06-03_per-sample-calibration/config_snapshot/study.yaml \
  --n-workers 32
python experiments/2026-06-03_per-sample-calibration/analyze.py
```

Outputs land in `runs/<sample_name>/` (per-sample calibration results, plots,
`trials.csv`, `convergence.png`), plus `results.json`, `aggregated_params.csv`,
`run_manifest.json`, and `figures/`.

> **VM note (case-sensitive filesystem):** the config uses standardized
> `images/` and `dark_frames/` subfolders. If the endophilin datasets are on
> disk as `Images/` / `Dark_frames/`, rename them to lowercase on the VM (Linux
> is case-sensitive) or the loader will not find them.

## Findings

> **TODO: fill after the VM run.** (Studies are not run during scaffolding.)
> Summarize: which params are stable vs setting-dependent; whether fitted gain
> tracks 561 voltage (and the sign/strength of the relationship); any sample
> that calibrated poorly (check its `convergence.png` and validation
> discrepancy).
