# Synthetic-data generation strategy (2026-06-04)

Decision record for HOW synthetic training/test data is generated for Stage 2, and
why. This is the design that every downstream experiment (architecture work, the
DECODE/cme-analysis benchmark, Phase 3) inherits. Companion records:
`2026-06-04_protein-channel-parameterization.md` (protein channel specifics) and
`2026-06-10_detector-loss-design.md` (how the detector is trained on this data).
Numeric knobs (exact ranges, sizes, seeds) live in each run's
`config_snapshot/`; this doc fixes the SHAPE of the strategy, not the numbers.

## What the forward model already provides

`src/simulator/forward_model.py` (`simulate_image`/`simulate_batch`) already
samples spot counts (Poisson), samples diameters from a supplied PMF, computes
lipid flux ∝ d² and protein flux ∝ d^alpha with per-spot lognormal heterogeneity,
renders rotated-Gaussian PSF spots, applies the calibrated PMT noise model, and
RETURNS per-spot ground truth (x, y, diameter, lipid/protein intensity). The
generator is an orchestration + parameter-sampling + serialization layer over it,
NOT a reimplementation of the physics.

## One detector for all imaging settings (Option A)

The six real samples were imaged at different settings (561 detector voltage, 488
intensity), so their calibrations differ. We generate ONE training set that
randomizes over the FULL range spanning all samples, so a SINGLE detector
generalizes across all imaging conditions — rather than training per-setting.
Rationale: the cross-setting variation (noise scale, contrast, density) is exactly
what a detector should be invariant to; per-setting retraining would pay cost to
make the detector WORSE at generalizing, and any architecture comparison on one
cherry-picked setting would just reward overfitting that setting's noise floor.

## Structured parameter sampling (not flat-independent, not whole-sets)

Cross-sample correlations from only six samples are noisy artifacts of which
settings we happened to image at, NOT physics. Encoding them (whole-calibrated-set
sampling) would make the detector brittle to a seventh setting. But two couplings
ARE physical and are preserved:

1. **gain/enf — sample as PRODUCT × arbitrary SPLIT.** Calibration found only the
   product (effective noise scale) is constrained; the individual values are
   degenerate. Sample `noise_scale` over its range, then split into a (gain, enf)
   pair via a random ratio. The split should never matter to any result; if it
   ever does, that is a bug worth seeing. Do NOT sample gain and enf independently.
2. **PSF near-circular.** Calibration found the PSF essentially circular (theta
   ill-defined). Sample one width + a small eccentricity (~0.9–1.1) and a free
   rotation, rather than independent sigma_x/sigma_y, which would produce elongated
   PSFs the optics never make.

Everything else (`lipid_brightness`, `spot_density`) is sampled INDEPENDENTLY,
uniform over the UNION of per-sample fitted ranges, WIDENED ±30% past the observed
extremes so the detector is hardened beyond the six samples rather than tuned to
them. `optical_bg_lipid` kept small. The protein channel follows
`2026-06-04_protein-channel-parameterization.md`.

Use the FITTED gain (effective noise param), never the photon-transfer
`measured_params.gain` (~270–450), which is a separate physical estimate and ~15–
40× larger — see `2026-06-03_calibration-findings.md`.

## Alpha: per-spot-random by default (alpha-agnostic detector)

Three selectable modes; default `per_spot_random`. Per-spot-random draws an
independent curvature alpha PER SPOT, decorrelating protein intensity from diameter
WITHIN an image so no detector can exploit a global enrichment trend as a shortcut.
This is the conservative, provably-safe training distribution: by construction it
cannot encode a global alpha–diameter relationship. `global_coherent` (one alpha
per image) and `mixed` exist for a later study testing whether coherence helps
small-spot detection — but that is a validation experiment, not the default.
Alpha is NEVER fitted; it is randomized in generation and MEASURED downstream from
the sorting curve. See the protein-channel record for the full argument.

Implementation note: the simulator takes one alpha per call. Per-spot alpha is
produced WITHOUT editing the simulator where possible (render lipid + ground truth
from one call, build protein per-spot with each spot's own alpha, reusing the
simulator's flux formula and noise path). Any need to change the simulator is
surfaced to the human, not done unilaterally.

## Spot-size distribution: emphasis for training, DLS for testing

- **Training:** a wide, SMALL-SIZE-EMPHASIS diameter distribution (default uniform
  in CURVATURE, i.e. 1/d), deliberately flatter than — and tilted small relative
  to — the real DLS distribution. Justification: curvature ∝ 1/r, so the
  small/high-curvature tail is the most important AND hardest regime; training flat
  or DLS-shaped would under-represent it and let the detector learn a size prior
  that biases exactly where the assay is most sensitive. Implemented by passing a
  custom (diameters, probs) PMF into the simulator in place of the DLS arrays.
- **Testing / Phase 3:** the REAL DLS distribution (via `io.parse_dls`), because
  size CONDITIONING is the point there and we specifically want detection
  performance in the small-diameter tail DLS actually populates.

So: wide-small-emphasis for train, DLS-realistic for test — the same max-entropy
"don't give the detector a prior it can exploit" logic as alpha.

## Spot count

Driven by `spot_density` (Poisson mean), itself sampled wide/flat over the widened
union range — no separate count prior. For controlled test SWEEPS, a
`fixed_spot_density` override lets test batches step density deterministically
(disperse → saturated) to probe crowding.

## Ground truth: centroids + properties (no masks)

Per spot: x, y, diameter_nm, lipid_intensity, protein_intensity, alpha_used,
sample_regime_id. Both intensities are emitted because downstream needs both
(lipid = size readout / diameter axis; protein = binding readout / sorting-curve
y-axis); diameter is emitted because Phase 3's sorting curve needs per-spot size.
Detection target is a SINGLE lipid-located heatmap (one liposome = one location,
present in lipid regardless of protein binding); intensity is regressed for BOTH
channels separately (different noise regimes). See the loss-design record.

## Provenance & reproducibility

Every dataset writes `provenance.json` (git commit + config + seed via
`src.provenance.write_provenance`) and a `dataset_manifest.json`. A dataset is
fully reproducible from (config + seed + commit). Datasets are large and
gitignored; the config + provenance are what get committed/tracked.
