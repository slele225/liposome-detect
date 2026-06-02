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
  - `io.py` — loading/parsing: TIFF stacks, DLS xlsx, dark frames
  - `estimation.py` — microscope parameter estimation: gain, PSF, backgrounds
  - `forward_model.py` — generative simulator (`simulate_*`) + PMT noise +
    non-puncta protein pixel extraction
- `src/calibration/` — joint multi-sample calibration
  - `statistics.py` — summary statistics for moment matching
  - `discrepancy.py` — config-driven real-vs-sim discrepancy (per-term weights)
  - `optimize.py` — Optuna joint optimization (shared + per-sample params)
  - `run.py` — full pipeline orchestration + comparison plots
- `src/provenance.py` — writes `provenance.json` for each artifact
- `configs/calibration/` — YAML calibration configs
- `data/` — raw biological images (LARGE — gitignored)
- `calibrations/` — calibration outputs: results JSON, plots, provenance (gitignored)
- `datasets/` — generated synthetic datasets (LARGE — gitignored)
- `tests/` — unit tests

## Key conventions

- YAML configs drive everything; no hardcoded paths in code.
- Calibration output goes to `calibrations/<name>/` (set via `output_dir` in
  the config). Each run writes `calibration_results.json`, comparison plots,
  and a `provenance.json` (git commit hash + config used).
- Sample naming: `<conc>nM_<protein>` (e.g. `20nM_EGFP`, `25nM_endophilin`).
- Channels in a 3-channel TIFF: 0 = protein (488 nm), 1 = lipid (561 nm),
  2 = transmitted light.

## Calibration model (what is fitted)

Joint multi-sample moment matching via Optuna. Real and simulated images are
reduced to summary statistics; the optimizer minimizes a weighted discrepancy.

- Shared (microscope) parameters, fitted jointly across all samples:
  `gain, enf, psf_sigma_x, psf_sigma_y, bg_amplitude, haze_level, labeling_eff`.
- Per-sample free parameters:
  `spot_density, bg_amplitude_protein, autofl_protein, voltage_scale_protein`.
- Per-sample pinned (measured from dark frames): `offset, read_noise_var`.

### Config-driven discrepancy

`src/calibration/discrepancy.py` computes the loss from up to six terms, each
gated by a `{enabled, weight}` entry. An enabled term contributes
`weight * raw_term`; a disabled term is skipped. Defaults reproduce the
archive's original hardcoded weighting exactly:

| term              | default weight | raw term                              |
|-------------------|----------------|---------------------------------------|
| pixel_hist        | 0.01           | pixel-intensity Wasserstein           |
| spot_intensity    | 0.005          | spot peak-intensity Wasserstein       |
| psd               | 1.0            | log radial PSD MSE                     |
| spot_density      | 1.0            | normalized spot-count squared error   |
| mean_pixel        | 1.0            | normalized mean-pixel squared error   |
| protein_nonpuncta | 0.005          | non-puncta protein Wasserstein        |

A `discrepancy:` block in a calibration YAML overrides any term's
`enabled`/`weight`. If the block is omitted, defaults apply and behavior is
identical to the original pipeline. See the commented example in
[configs/calibration/joint_all_samples.yaml](configs/calibration/joint_all_samples.yaml).

## Running

```bash
uv sync
python calibrate.py --config configs/calibration/smoke_single.yaml
```

`joint_smoke.yaml` / `smoke_single.yaml` are fast smoke tests (5 trials), not
real calibration results. `joint_all_samples.yaml` is the real 6-sample run.

## Not yet ported

- Training data generation, U-Net architecture/training, inference.
- Downstream analysis (slope/sorting-curve computation, diameter
  stratification). The `src.provenance` helper is the only piece carried over
  from the archive's `analysis/`, because the calibration pipeline depends on it.
