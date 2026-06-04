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

**Re-run only the analysis** on existing `runs/` (no recalibration):
`python experiments/2026-06-03_per-sample-calibration/analyze.py` — it rebuilds
`results.json`/`aggregated_params.csv` from the per-run
`runs/<id>/calibration_results.json` if missing, then redraws the figures. To
force a rebuild without plotting:
`python -m src.calibration.study --aggregate-only experiments/2026-06-03_per-sample-calibration`.

> **Folder names:** the config uses standardized `images/` and `dark_frames/`
> subfolders, and `src/simulator/io.py` now resolves case- and
> space/underscore variants (`Images`, `Dark_frames`, `dark frames`) tolerantly,
> so a stray variant on a case-sensitive filesystem is auto-resolved (and logged)
> rather than failing.

## Findings

Each sample was calibrated independently (200 trials, default weights). See
`figures/cross_sample_params.csv`, `figures/gain_vs_voltage.png` and
`figures/params_across_samples.png`.

- **Fitted gain does NOT cleanly track the 561 detector voltage.** It sits in a
  narrow band ~11–22 ADU/photon (50nM_EGFP 10.7, 300nM_EGFP 13.5, 100nM_EGFP
  14.5, 300nM_endophilin 18.5, 25nM_endophilin 19.2, 20nM_EGFP 21.7) with no
  monotonic relationship to voltage — e.g. 50nM_EGFP at 640 V has the *lowest*
  fitted gain. The whole band is ~15–40× below the photon-transfer **measured**
  gain (~270–450). **Conclusion: gain is not independently identifiable** from
  the lipid moment-matching objective (it trades off against enf/brightness) and
  must not be read as a physical gain measurement.
- **Setting-dependent, as expected:** `lipid_brightness` (EGFP ~4.8–8.4k vs
  endophilin ~12–17k), `spot_density` (~490–834) and the PSF widths
  (`psf_sigma_x` ~1.6–2.3, `psf_sigma_y` ~1.5–2.6) vary sample to sample.
- **Weakly constrained:** `psf_theta` spans nearly the full ±45° prior
  (near-circular PSFs make it ill-defined); `optical_bg_lipid` is small for every
  sample (~0.3–2.2 photons).
- All six samples converged with low training/validation discrepancy and no
  failures (see `run_manifest.json` and each run's `convergence.png`).
