# CLAUDE.md — liposome-detect project entry point

## Project summary

A physics-calibrated forward simulator of a confocal fluorescence microscope,
plus a detector for liposome puncta in two-channel images, used to measure
protein curvature sensing on liposomes (the SLiC assay). Real samples are an
EGFP negative control and an endophilin curvature sensor. The simulator is
calibrated to real microscope data so that a detector can later be trained on
synthetic images with known ground truth.

This repository is a clean rebuild. **This stage ports only the simulator and
the calibration pipeline.** Training/detector code (U-Net) and downstream
analysis are intentionally not yet ported.

## Directory structure

- `calibrate.py` — entry point: `python calibrate.py --config <yaml>`
- `src/simulator/` — the forward model
  - `io.py` — loading/parsing: TIFF stacks (center-cropped to 256x256), DLS
    xlsx, dark frames
  - `estimation.py` — microscope parameter estimation: gain, PSF
  - `forward_model.py` — generative simulator (`simulate_image`/`simulate_batch`)
    + rotated-covariance PSF + PMT noise with photon-stage optical background
- `src/calibration/` — joint multi-sample calibration
  - `statistics.py` — summary statistics for moment matching
  - `discrepancy.py` — config-driven real-vs-sim discrepancy (per-term weights)
  - `optimize.py` — Optuna joint optimization (shared + per-sample params);
    also logs per-trial objectives (`trials.csv` + `convergence.png`)
  - `run.py` — full pipeline orchestration + comparison plots
  - `study.py` — reusable parallel study runner (many calibrations →
    aggregated results); three modes: `per_sample`, `bootstrap`, `weight_sweep`
  - `study_analysis.py` — reusable tables/plots over a study's `results.json`
- `src/provenance.py` — writes `provenance.json` for each artifact
- `configs/calibration/` — YAML calibration configs (single-calibration runs)
- `experiments/` — self-contained calibration **studies** (one folder per
  experiment, `<date>_<name>/`); see "Calibration studies" below
- `scripts/` — cross-experiment orchestration (e.g.
  `run_all_calibration_studies.sh`)
- `docs/decisions/` — design decision records
- `data/` — raw biological images (LARGE — gitignored)
- `calibrations/` — calibration outputs: results JSON, plots, provenance (gitignored)
- `datasets/` — generated synthetic datasets (LARGE — gitignored)
- `tests/` — unit tests

## Key conventions

- YAML configs drive everything; no hardcoded paths in code.
- Calibration output goes to `calibrations/<name>/` (set via `output_dir` in
  the config). Each run writes `calibration_results.json`, comparison plots, a
  `provenance.json` (git commit hash + config used), and a per-trial objective
  log (`trials.csv` + `convergence.png`).
- Sample naming: `<conc>nM_<protein>` (e.g. `20nM_EGFP`, `25nM_endophilin`).
- Channels in a 3-channel TIFF: 0 = protein (488 nm), 1 = lipid (561 nm),
  2 = transmitted light.

## Calibration model (what is fitted)

Joint multi-sample moment matching via Optuna, **lipid-channel only and
detection-free**, on the center-256x256 crop. Real and simulated lipid images
are reduced to summary statistics; the optimizer minimizes a weighted
discrepancy. The simulator renders spots with a per-channel rotated-Gaussian
PSF (covariance matrix, normalized to sum to 1 → amplitude is total flux); the
lipid optical background is a scalar in photons injected at the photon stage of
the PMT noise step.

- Shared (microscope) parameters, fitted jointly across all samples:
  `lipid_brightness, psf_sigma_x, psf_sigma_y, psf_theta, gain, enf,
  optical_bg_lipid` (the three PSF terms define the rotated lipid PSF).
- Per-sample free parameter: `spot_density`.
- Per-sample pinned (measured from dark frames): `offset, read_noise_var`
  (lipid + protein; protein values recorded for downstream generation).
- Not fitted (generation-time knobs): `protein_brightness, curvature_alpha`,
  and the protein PSF (initialized from the measured per-channel PSF).

### Config-driven discrepancy

`src/calibration/discrepancy.py` computes the lipid-image loss from up to five
terms, each gated by a `{enabled, weight}` entry. An enabled term contributes
`weight * raw_term`; a disabled term is skipped.

| term       | default weight | raw term                                      |
|------------|----------------|-----------------------------------------------|
| pixel_hist | 0.01           | pixel-intensity Wasserstein                   |
| psd        | 1.0            | log radial PSD MSE                            |
| mean_pixel | 1.0            | relative squared error of mean pixel          |
| quantiles  | 1.0            | sum of rel. sq. errors of 99th & 99.9th pctl  |
| skewness   | 1.0            | relative squared error of pixel skewness      |

`mean_pixel`, `quantiles`, and `skewness` are all relative squared errors, so a
shared weight of 1.0 keeps them on the same scale. A `discrepancy:` block in a
calibration YAML overrides any term's `enabled`/`weight`; omitting the block
uses the defaults. See the commented example in
[configs/calibration/joint_all_samples.yaml](configs/calibration/joint_all_samples.yaml).

## Calibration studies (experiments)

A single calibration writes one `output_dir` (via `calibrate.py`). A **study**
runs many independent calibrations and aggregates them = an **experiment**.

- **Runner:** `src/calibration/study.py` — a `multiprocessing.Pool` runner with
  `n_workers` configurable (`--n-workers`; default `os.cpu_count()`). Each worker
  pins numpy/BLAS to a single thread (OMP/MKL/OpenBLAS=1) via a worker
  initializer that runs before numpy is imported, so `n_workers ≈ cores` without
  oversubscription. Three modes:
  - `per_sample` — calibrate each of several samples independently.
  - `bootstrap` — one sample, N repeats, each on a random d-image subset
    (distinct seed per repeat, without replacement).
  - `weight_sweep` — one sample, calibrated once per named discrepancy-weight
    config.
  Run: `python -m src.calibration.study --config <study.yaml> --n-workers N`.
  Outputs into the study's `output_dir`: `runs/<run_id>/` (one calibration
  each), `results.json` (list of `{run_id, fitted_params}`),
  `aggregated_params.csv`, `run_manifest.json`.
- **Experiments:** each study lives in a self-contained folder
  `experiments/<date>_<name>/` with `EXPERIMENT.md` (question / method /
  reproduce / findings), `run.sh` (thin wrapper: runner + analysis),
  `config_snapshot/` (the exact study config), `analyze.py` (experiment-specific
  plots/tables, using `src/calibration/study_analysis.py`), and at run time
  `runs/`, `figures/`, `results.json`.
- **Orchestration:** `scripts/run_all_calibration_studies.sh` runs every
  experiment's `run.sh` sequentially, passing `n_workers` through (positional
  arg or `N_WORKERS` env var). Designed to run on a 32-core VM
  (`N_WORKERS=32 ./scripts/run_all_calibration_studies.sh`).
- Design rationale (per-sample not joint, bootstrap d/repeats, weight-sweep
  compared by parameter spread not loss, single-threaded-per-worker): see
  [docs/decisions/2026-06-03_calibration-study-design.md](docs/decisions/2026-06-03_calibration-study-design.md).

## Running

```bash
uv sync
python calibrate.py --config configs/calibration/smoke_single.yaml
```

`joint_smoke.yaml` (5 trials) / `smoke_single.yaml` (30 trials) are smoke tests,
not real calibration results. `joint_all_samples.yaml` is the real 6-sample run.

## Not yet ported

- Training data generation, U-Net architecture/training, inference.
- Downstream analysis (slope/sorting-curve computation, diameter
  stratification). The `src.provenance` helper is the only piece carried over
  from the archive's `analysis/`, because the calibration pipeline depends on it.
