# EXP — Stage-2 full training run + uncertainty-head investigation

- **Date:** 2026-06-10
- **Model:** two-channel HRNet detector with per-spot intensity + log-variance heads
  (see [docs/decisions/2026-06-10_detector-loss-design.md](../../docs/decisions/2026-06-10_detector-loss-design.md)).
- **Status:** run complete; findings established. The trained `best.pt`,
  `runs/hrnet_v1/metrics.jsonl`, and `provenance.json` are saved OFF-REPO
  (gitignored). Datasets regenerate from the committed generator configs.

## Question

Two questions, answered on one full run + a chain of post-hoc analyses on the best
checkpoint:

1. Does the detector train to a usable intensity-recovery accuracy at full scale,
   and which validation metric is trustworthy across the MSE→NLL loss boundary?
2. How should the curvature-sensing parameter **alpha** be recovered from the
   detector's outputs — specifically, is ordinary least squares adequate, and do the
   per-spot uncertainty heads earn a place as **regression weights**?

## Method

- **Training:** 20k synthetic images (train_v1 distribution), single H100, ~5.6 h
  wall (input-bound — data loading, not GPU, was the bottleneck). Early stopping on
  `val_intensity_logmse` (boundary-consistent). Loss: penalty-reduced focal heatmap +
  L1 offset + log-space intensity NLL (lipid + protein), with an MSE→NLL warmup.
- **Detection sampling analysis:** recall and within-size-bin detected-vs-missed
  protein comparison over a held-out diameter-stratified val set.
- **Alpha-estimator analysis:** fixed-alpha synthetic test sets at true
  α = 0.5 / 1.0 / 1.5 / 2.0 (`global_coherent` alpha, single-point `alpha_range`,
  `size_mode=emphasis`). Recover alpha from the slope of log(protein) vs log(lipid)
  (α = 2·slope) using OLS, total-least-squares / constant-λ Deming, and a correct
  per-point errors-in-variables (York) fit. Run on both TRUE intensities (isolates
  the estimator) and PREDICTED intensities (the full pipeline).
- **Uncertainty-head analysis:** predicted `exp(logvar)` vs actual log-error², binned
  by predicted variance; and a per-spot-weighting test (York) vs constant-λ Deming on
  synthetic known-noise and on the assay data.

The diagnostic/test sets use `size_mode=emphasis` (uniform-ish over the diameter
range — valid for METHOD VALIDATION because it stresses the small tail). The final
benchmark uses the DLS-realistic size distribution, not emphasis sizing.

## Findings

### Training & metric behavior

- **Training is healthy.** Detection F1 plateaus early at **~0.64** and stays there
  (detection converges fast). `val_intensity_logmse` keeps dropping for much longer —
  the intensity heads are the slow, important part of the fit. The full run reached
  **logmse 0.0835** (best @ epoch 14; early-stopped @ epoch 24) versus the short
  14-epoch diagnostic's ~0.10–0.12: the extra training buys deeper INTENSITY
  convergence, not better detection.
- **The NLL phase boundary makes `val_total` discontinuous.** At the MSE→NLL switch
  (`nll_warmup`) the NLL terms go NEGATIVE once the model is confident, so the
  statically-weighted `val_total` STEPS in character at that epoch. The
  boundary-consistent metrics (`val_detection_f1`, `val_intensity_logmse`) move
  smoothly across it. **Early stopping must use a boundary-consistent metric
  (`val_intensity_logmse`), never `val_total`.**
- **The late plateau is a generalization floor, not an optimization problem.** Train
  loss kept dropping while val plateaued. Do NOT tune AdamW/LR in response — this is
  real convergence on this data. A better model would need a smaller calibration
  correction (below), but the plateau itself is not an optimizer-tuning target.

### Detection sampling

- **Recall rises with diameter:** ~0.33 at 40–55 nm → ~0.67 at 220–300 nm.
- **But within each size bin, detection is REPRESENTATIVE.** The median true protein
  intensity of *detected* vs *missed* spots has ratio **~1.00–1.02 for every bin
  ≥70 nm**, **~1.05 at 55–70 nm**, and **~1.16 only at the smallest 40–55 nm bin**.
  So low recall does NOT bias the recovered sorting curve except mildly at the
  smallest bin: within a size bin the detector is not cherry-picking the bright spots.
- **The operative property is REPRESENTATIVENESS, not raw recall.** A detector can
  miss many small spots and still recover alpha unbiased, provided the ones it keeps
  are representative of their size bin. Raw recall feeds the detection-efficiency
  correction; it is not itself the bias source.

### Alpha-measurement method: OLS is biased, EIV corrects it (the core methodological finding)

- **OLS is biased LOW** (regression dilution / errors-in-variables). BOTH axes are
  noisy — lipid carries PMT noise, protein carries PMT noise PLUS lognormal(0, 0.1)
  η heterogeneity — and noise on the x-axis (log-lipid) attenuates the slope toward
  zero. On fixed-alpha synthetic sets, OLS on TRUE intensities recovered
  **~0.42 / 0.85 / 1.28 / 1.70** for true 0.5 / 1.0 / 1.5 / 2.0.
- **Constant-λ Deming (total least squares / errors-in-variables) recovers the
  diagonal.** On the same TRUE intensities it recovered **~0.46 / 0.93 / 1.42 / 1.92**
  — essentially y = x. The METHOD is sound; the estimator is constant-λ Deming, not
  OLS.
- **This applies identically to every benchmarked method** (ours and all baselines).
  It is a correct estimator the whole field should use, not a moat — the benchmark
  runs every method's slope through the same EIV fit.

### Instrument calibration curve (a deliverable)

- On the full-run best checkpoint, **PREDICTED-intensity** alpha recovery under
  constant-λ Deming was **monotonic and well-ordered:** recovered
  **0.644 / 0.957 / 1.321 / 1.677** for true **0.5 / 1.0 / 1.5 / 2.0**. The
  compression is mild, smooth, and roughly proportional — i.e. **correctable** via
  the recovered→true mapping. This mapping is the instrument's **calibration curve**:
  the deliverable that corrects residual bias on real data.
- **Paper caveat:** this is a standard instrument-response calibration (monotonic +
  stable + characterized on knowns). Its validity on REAL data must be confirmed via
  the **DLS-consistency anchor** (not yet done). Fitting/validating the curve on
  synthetic data alone would be circular.
- **The PRED↔TRUE gap is residual intensity-estimation compression** (model
  undertraining / the generalization floor), NOT a method flaw. Crucially the
  **ordering is preserved** — compression with preserved ordering is the benign,
  calibratable kind. A better-trained model would need a SMALLER calibration
  correction (smaller correction ⇒ more robust), so model quality still matters even
  though calibration rescues the current bias.

### Per-spot uncertainty heads: calibrated & informative, but weighting is contraindicated

- **The uncertainty is informative and roughly calibrated.** Predicted `exp(logvar)`
  vs actual log-error², binned by predicted variance: actual_mse rises MONOTONICALLY
  across deciles for both channels, with a stable ratio **~1.4** (mild, consistent
  overconfidence). So the head correctly RANKS spots by reliability and is calibrated
  up to a constant factor.
  - **Caveat:** predicted variance is positively correlated with intensity (see
    below), so part of what the head encodes is SIZE. Do not overclaim it as pure
    "measurement reliability."
- **Units lesson (recorded to prevent recurrence):** the intensity NLL uses a
  LOG-space residual `r = log(pred+eps) − log(true+eps)` with `sigma2 = exp(logvar)`.
  So `logvar` is ALREADY the variance of the log-residual — consume `exp(logvar)`
  **directly**; do NOT apply a delta-method `/intensity²` conversion. An early
  calibration script did the double conversion and produced spurious ~1e8 ratios.
- **Per-spot weighting was validated as an ESTIMATOR but rejected for THIS assay.**
  A correct per-point errors-in-variables fit (York 1968/2004; no global λ) was
  validated on synthetic known heteroscedastic noise — it recovered the true slope
  with **~1/3 the variance** of constant-λ Deming (sd_ratio **~0.35**). This confirms
  both that the York estimator is correct AND that an earlier λ-mixing "weighted
  Deming" script was an invalid chimera.
- **On the assay data, per-spot weighting biases alpha LOW** (worse than constant-λ),
  even on TRUE intensities. **Mechanism:** predicted per-spot variance is CORRELATED
  with the regression x-axis (log-lipid), **r ≈ +0.58**. When measurement variance is
  confounded with position ALONG the line, weighting down-weights one end and tilts
  the slope. This is structural to the assay (bigger spots genuinely differ in noise),
  NOT a head bug — no amount of "fixing the head" makes weighting unbiased here.
- **Conclusion:** the alpha estimator is **UNWEIGHTED constant-λ Deming**. Per-spot
  uncertainty is used for **QC/filtering** and for **error propagation** (honest
  confidence intervals on alpha), NOT as regression weights. This is cleaner and more
  defensible than "weighting didn't help": weighting is **contraindicated** for a
  principled, demonstrated reason.

## Reproduce

Datasets are gitignored; regenerate the fixed-alpha test sets and the diagnostic val
set from the committed generator configs, then run the eval scripts against the
trained checkpoint (saved off-repo as `runs/hrnet_v1/best.pt`).

```bash
uv sync

# Fixed-alpha test sets (global_coherent alpha, single-point alpha_range, emphasis sizing)
for a in 0p50 1p00 1p50 2p00; do
  uv run python -m src.generator.generate --config configs/generator/alpha_${a}.yaml
done
# Diagnostic / diameter-stratified val set
uv run python -m src.generator.generate --config configs/generator/diag_val.yaml
```

Eval scripts (now under `src/eval/`; each takes `--config` / `--ckpt` / `--datasets`):

- `python -m src.eval.alpha_recovery` — OLS vs constant-λ Deming on TRUE and
  PREDICTED intensities (the OLS-bias / EIV-correction + the calibration-curve points).
- `python -m src.eval.detection_bias` — within-size-bin detected-vs-missed protein
  ratio (representativeness), plus slopes with/without the smallest bin.
- `python -m src.eval.recall_vs_diameter` — recall + protein-error per diameter bin.
- `python -m src.eval.uncertainty_calibration` — predicted `exp(logvar)` vs actual
  log-error² across deciles (the ~1.4 ratio, monotone).
- `python -m src.eval.york_test` — correct per-point EIV (York) vs constant-λ Deming;
  synthetic Test A (`--skip-model`) confirms the variance reduction, B/C the assay
  bias. **`york_test.py` SUPERSEDES the deleted `weighted_deming_boot.py`** (whose
  λ-mixing weighted fit was an invalid estimator).

The line fits all come from the canonical `src/eval/alpha_fit.py` (single source of
truth: `ols_slope`, `deming_slope`, `york_slope`, `recover_alpha`, `CalibrationCurve`).

> The diagnostic/test sets used `size_mode=emphasis` (valid for method validation —
> it stresses the small-liposome tail). The final cross-method benchmark uses the
> DLS-realistic size distribution, not emphasis sizing.
