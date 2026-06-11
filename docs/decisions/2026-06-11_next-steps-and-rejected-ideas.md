# Next steps & rejected ideas (2026-06-11)

Forward-plan record so future work (and future chats) doesn't re-derive settled
questions or re-propose dead ends. Builds on the Stage-2 findings in
[experiments/2026-06-10_diagnostic-run/EXPERIMENT.md](../../experiments/2026-06-10_diagnostic-run/EXPERIMENT.md)
and the benchmark plan in
[2026-06-10_benchmark-design.md](2026-06-10_benchmark-design.md).

## Headline next experiment

**Real-data corrected-vs-standard comparison.** On the real EGFP (negative control)
+ endophilin (curvature sensor) data, run the detector → EIV (constant-λ Deming) →
calibration-curve correction, and ask: **does the EIV + calibration correction change
the recovered alpha enough to alter a BIOLOGICAL conclusion** — sensor vs non-sensor
classification, or the sensing-strength ranking — versus the naive OLS / uncorrected
pipeline?

This is the experiment that decides whether the contribution is **strong** (the
correction flips or sharpens a biological call) or **modest** (same call, tighter
numbers). It ALSO doubles as the off-simulator validation of the calibration curve:
its real-data validity is anchored by **DLS consistency** (the recovered lipid size
distribution must match measured DLS), which is the only non-circular check available
without hand annotation.

## Keep (planned, in priority order)

1. **Firm up the calibration curve** — more fixed-alpha points (beyond
   0.5/1.0/1.5/2.0) + bootstrap error bars on each recovered→true anchor, so the
   mapping has uncertainty. `src/eval/alpha_fit.py::CalibrationCurve` already loads a
   fitted curve from file for exactly this refresh.
2. **Alpha error-propagation** — a proper confidence interval on alpha that folds in
   detection (efficiency/representativeness), per-spot intensity variance (the
   uncertainty heads), and the EIV fit uncertainty. This is where the per-spot σ earns
   its place (NOT as a regression weight — see rejected (c)).
3. **Cross-method benchmark** — Spotiflow / DECODE / SpotMAX / classical run through
   the SAME EIV + calibration pipeline (per 2026-06-10_benchmark-design.md), with the
   headline metric binned by liposome size.
4. **Detection-efficiency correction** — recall-vs-diameter curve anchored by DLS, to
   correct the size distribution for missed small spots.

## Rejected — do NOT re-propose

- **(a) DLS as a detection PRIOR / conditioning.** Feeding the measured DLS size
  distribution into detection (as a prior or conditioning input) is CIRCULAR: it
  biases the recovered size distribution toward DLS and therefore biases alpha. DLS is
  a VALIDATION ANCHOR only — it checks the recovered size distribution, it never
  shapes it.
- **(b) RL / KL fine-tuning toward curvature recovery.** Fine-tuning the detector
  against a curvature-recovery objective is CIRCULAR and contaminates the measurement:
  it leaks the curvature law into the model and violates the alpha-agnostic invariant
  (see [2026-06-04_protein-channel-parameterization.md](2026-06-04_protein-channel-parameterization.md)
  and the loss-design doc). The detector must stay alpha-agnostic.
- **(c) Per-spot uncertainty-WEIGHTED EIV.** Contraindicated by the variance/size-axis
  confound (predicted variance correlates with log-lipid, r ≈ +0.58), which biases the
  slope. Use UNWEIGHTED constant-λ Deming for the slope and reserve the per-spot
  uncertainty for QC and error-propagation (keep (2) above). The York per-point
  estimator is correct in general (validated on synthetic known noise) but is for
  diagnostics/QC here, not the production alpha.
