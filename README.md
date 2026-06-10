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

## Diagnostic training run (the gate before a full H100 job)

Before committing a multi-hour H100 training run, do a SHORT real-scale diagnostic
and read one report. The diagnostic verifies two things smoke scale cannot: that
the four WEIGHTED loss terms stay in balance AFTER the focal heatmap term drops,
and whether the static `val_total` steps at the MSE→NLL boundary (a metric
artifact) or genuinely regresses. It is the GATE, not the final run.

On the H100, `scripts/run_diagnostic.sh` walks the six steps in order (no new code
written on the paid instance):

```bash
N_WORKERS=32 ./scripts/run_diagnostic.sh     # or: ./scripts/run_diagnostic.sh 32
```

1. `uv sync` (install the CUDA torch wheel matching the GPU — see the script note).
2. Regenerate the per-sample calibrations if absent (they are gitignored).
3. Generate a REAL train set + a SEPARATE real-sized val set (different seed)
   from `configs/generator/diag_train.yaml` / `diag_val.yaml`.
4. `python -m src.train.compute_stats --dataset datasets/diag_train` → paste the
   `norm_mean`/`norm_std` + per-channel `eps_*` into
   `configs/train/hrnet_diagnostic.yaml` (its placeholders are `null`).
5. `python -m src.train.train --config configs/train/hrnet_diagnostic.yaml`.
6. `python -m src.train.diagnostic --run runs/hrnet_diagnostic` → read the VERDICT.

The verdict's three lines say whether term balance is healthy, whether the
`val_total` boundary step is a real regression or an artifact (judged against the
boundary-consistent `val_detection_f1` / `val_intensity_logmse`), and which
`early_stop_metric` the FULL run should use. Configure that separate full run from
what the diagnostic shows.

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
