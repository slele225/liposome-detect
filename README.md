# liposome-detect

Physics-calibrated forward simulator of a confocal fluorescence microscope, plus
a detector for liposome puncta in two-channel images, for measuring protein
curvature sensing (the SLiC assay).

This stage ships the **forward simulator** and the **joint multi-sample
calibration** pipeline. Training/detector code comes later.

## Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Calibrate

Calibration fits the simulator's microscope parameters to real images by
moment matching (Optuna). Configs live in `configs/calibration/`.

```bash
# Fast single-sample smoke test against the bundled data/20nM_EGFP dataset
python calibrate.py --config configs/calibration/smoke_single.yaml

# Real joint calibration across all 6 samples (needs all datasets in data/)
python calibrate.py --config configs/calibration/joint_all_samples.yaml
```

Each run writes to `calibrations/<config-name>/`:
- `calibration_results.json` — fitted parameters + discrepancies
- `plots/<sample>/comparison_*.png` — real-vs-simulated comparisons
- `provenance.json` — git commit + config used

The real-vs-sim discrepancy loss is config-driven: a `discrepancy:` block in
the YAML re-weights or disables individual terms (defaults reproduce the
original hardcoded weighting). See
`configs/calibration/joint_all_samples.yaml` for the documented schema.

## Test

```bash
uv run pytest
```

## Layout

```
calibrate.py              entry point
src/simulator/            forward model (io, estimation, forward_model)
src/calibration/          calibration (statistics, discrepancy, optimize, run)
configs/calibration/      YAML configs
data/                     raw images (gitignored)
calibrations/             outputs (gitignored)
```
