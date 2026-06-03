# Calibration study design — decisions (2026-06-03)

This records the design decisions behind the calibration-study machinery
(`src/calibration/study.py`) and the three experiments under `experiments/`.
These are **decisions**, made up front; results go in each experiment's
`EXPERIMENT.md` after the VM run.

## 1. Calibration is detection-free and lipid-only

The calibration objective uses **only the lipid (561 nm) channel** and does
**no spot detection**. Real and simulated lipid 256×256 crops are reduced to
summary statistics (pixel histogram, radial PSD, mean, high quantiles,
skewness) and the optimizer minimizes a weighted discrepancy between them. The
protein channel is never simulated or scored during calibration.

**Why:** the goal of calibration is to match the *image-formation physics*
(PSF, gain, noise, optical background, spot density) of the microscope. That
signal lives in the lipid channel's pixel statistics. Detection introduces a
thresholding/matching step with its own hyperparameters and failure modes;
moment-matching on the raw pixel statistics is more robust and is sufficient to
pin the forward model. The protein channel is a downstream generation-time knob,
not a calibration target.

## 2. Per-sample calibration, NOT joint (EXP 2)

EXP 2 calibrates each of the six samples **independently** — its own gain/PSF
measurement and its own Optuna search — rather than one joint fit with shared
microscope parameters.

**Why:** the **561 nm detector (PMT) voltage was set per sample** (750, 640,
630, 580 V for 20/50/100/300 nM EGFP; 670, 600 V for 25/300 nM endophilin). PMT
gain is roughly exponential in control voltage, so the effective gain genuinely
**differs per sample**. A joint fit with a single shared `gain` would be
physically wrong. Per-sample fits let gain (and the other shared-microscope
params) float per setting, and EXP 2 then checks whether the fitted gain tracks
the known voltage and which parameters are stable vs setting-dependent.

(The existing `configs/calibration/joint_all_samples.yaml` joint fit remains for
the separate purpose of estimating a single best-compromise microscope model; it
is not what these studies do.)

## 3. Bootstrap: d = 25, without replacement, 100 repeats (EXP 3)

EXP 3 measures calibration **stability to the image subset** on
`25nM_endophilin` (55 images). Each of **100 repeats** draws **d = 25** images
**without replacement** (distinct seed per repeat) and calibrates that subset;
we then look at the distribution / coefficient of variation of each fitted
parameter across repeats.

**Why these choices:**
- **Without replacement** — each repeat is a genuine distinct subset of the real
  data, the honest question being "would a different 25-image acquisition give
  the same calibration?" Drawing with replacement would duplicate frames within
  a subset and muddy that interpretation.
- **d = 25** of 55 — large enough for a stable per-subset calibration (≈20
  train / 5 val after the pipeline's internal split) yet small enough that
  subsets differ substantially, so the spread across repeats is informative.
- **100 repeats** — enough to estimate a per-parameter mean / std / CV and see
  the shape of each distribution.

## 4. Weight-sweep compared by parameter spread, NOT loss (EXP 1)

EXP 1 calibrates `25nM_endophilin` once per discrepancy-weight config (an
all-equal baseline plus one config per term up-weighted 10×) and asks whether the
fitted **parameters** move.

**Why compare by parameter spread and not loss:** each config optimizes a
**differently weighted objective**, so its loss is on a different scale — the
loss numbers are simply **not comparable across configs**. The only meaningful
question is whether the *fit* changes: if a parameter barely moves as the
weighting changes (low CV across configs), the weighting is irrelevant and the
data drives the calibration; if it moves a lot, that term's weight steers the
fit. The analysis therefore reports across-config parameter spread (CV) and
never ranks configs by loss.

## 5. Parallelism: single-threaded numpy per worker

Studies run many independent calibrations in a `multiprocessing.Pool`. Each
calibration is CPU-bound numpy (FFTs, PSF rendering, curve fits). Left alone,
each worker's BLAS/OpenMP backend would spin up its own thread pool, and
`n_workers × cores` threads would fight over the cores (oversubscription),
making everything slower.

**Decision:** every worker pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` (plus numexpr/veclib) to **1**, in a Pool **worker
initializer** that runs **before numpy is imported in that worker**. To make
that ordering hold, `src/calibration/study.py` imports nothing heavy at module
top (numpy / `run_full_pipeline` are imported lazily inside the per-run task
function), and the pool uses the **`spawn`** start method so each worker is a
fresh interpreter that runs the initializer before importing numpy — identically
on Windows and the Linux VM. Net effect: `n_workers` single-threaded
calibrations, so `n_workers ≈ cores` is the right setting.

## 6. n_workers is configurable; default = os.cpu_count()

The worker count is a CLI arg (`--n-workers`) / function arg, defaulting to
`os.cpu_count()`. Each experiment's `run.sh` forwards a positional arg or the
`N_WORKERS` env var, and `scripts/run_all_calibration_studies.sh` passes it
through to all three. **Why:** development happens on a small machine but the
real studies run on a 32-core VM; nothing about the core count is hardcoded, so
the same scripts scale by setting `N_WORKERS` (e.g. `N_WORKERS=32`).

## 7. Self-contained experiment folders

Each study is an **experiment**: a self-contained folder
`experiments/<date>_<name>/` holding its `EXPERIMENT.md`, its `run.sh`, a
`config_snapshot/` of the exact study config, and (at run time) `runs/`,
`figures/`, `results.json`. **Why:** an experiment's provenance — what was asked,
how it was run, and what came out — lives in one place, reproducible with a
single `run.sh`. Reusable machinery stays in `src/` (the runner and analysis
helpers); only the thin per-experiment wrapper and bespoke analysis live in the
experiment folder.
