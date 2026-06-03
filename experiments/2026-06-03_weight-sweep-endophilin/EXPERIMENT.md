# EXP 1 — Discrepancy weight-sweep (25nM_endophilin)

- **Date:** 2026-06-03
- **Mode:** `weight_sweep` (see `src/calibration/study.py`)
- **Status:** scaffolded; **not yet run** (runs on the 32-core VM)

## Question

Does the choice of **discrepancy-term weighting** change the fitted calibration?
The objective is a weighted sum of up to five terms (pixel_hist, psd, mean_pixel,
quantiles, skewness). If we re-weight the terms, do the fitted **parameters**
move? If they stay put across very different weightings, the weighting doesn't
matter — the data, not the loss weighting, determines the calibration. If a
parameter moves a lot when a particular term is up-weighted, that term genuinely
steers the fit.

## Method

- One sample: `25nM_endophilin`, calibrated once per weight config.
- Five named configs:
  1. `equal` — all five terms at weight 1.0 (baseline).
  2. `up_pixel_hist` — pixel_hist 10×, others 1.0.
  3. `up_psd` — psd 10×, others 1.0.
  4. `up_quantiles` — quantiles 10×, others 1.0.
  5. `up_skewness` — skewness 10×, others 1.0.
- `n_trials = 200`, `n_sim_per_trial = 30`, `val_fraction = 0.2`, `seed = 0`.
  Same sample, same seed across configs, so the *only* thing that changes is the
  objective weighting.
- Config: [config_snapshot/study.yaml](config_snapshot/study.yaml).

> **Do NOT compare configs by loss value.** Each config optimizes a differently
> weighted objective, so the loss is on a different scale and the numbers are not
> comparable. The analysis compares configs **only** by whether the fitted
> parameters move. See
> [docs/decisions/2026-06-03_calibration-study-design.md](../../docs/decisions/2026-06-03_calibration-study-design.md).

### Analysis (`analyze.py`, runs after the calibrations)

- `figures/params_across_configs.csv` — fitted params, one row per weight config.
- `figures/param_spread_across_configs.csv` — per-parameter mean / std / **CV**
  / min / max **across the configs** (the comparison metric; low CV = stable).
- `figures/params_across_configs.png` — each fitted parameter across the configs.
- `results.json` / `aggregated_params.csv` — written by the runner. (The
  per-config loss values are also recorded but are intentionally not used to
  rank configs.)

## Reproduce

```bash
# From the repo root, with the environment synced (uv sync).
N_WORKERS=5 bash experiments/2026-06-03_weight-sweep-endophilin/run.sh
# (only 5 calibrations here, so n_workers > 5 gives no extra speedup)
```

Underlying commands the wrapper runs:

```bash
python -m src.calibration.study \
  --config experiments/2026-06-03_weight-sweep-endophilin/config_snapshot/study.yaml \
  --n-workers 5
python experiments/2026-06-03_weight-sweep-endophilin/analyze.py
```

Outputs land in `runs/equal/`, `runs/up_pixel_hist/`, `runs/up_psd/`,
`runs/up_quantiles/`, `runs/up_skewness/` (per-config calibration results, plots,
`trials.csv`, `convergence.png`), plus `results.json`, `aggregated_params.csv`,
`run_manifest.json`, and `figures/`.

## Findings

> **TODO: fill after the VM run.** (Studies are not run during scaffolding.)
> Report the across-config CV for each parameter. State the conclusion in terms
> of parameter spread only (NOT loss): which parameters are stable across
> weightings (weighting irrelevant) and which move (that term's weight matters).
