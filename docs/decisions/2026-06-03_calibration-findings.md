# Calibration findings (2026-06-03)

Results of the three calibration studies run on the 32-core VM
(`experiments/2026-06-03_per-sample-calibration`, `_bootstrap-endophilin`,
`_weight-sweep-endophilin`). The study *design* decisions are in
[2026-06-03_calibration-study-design.md](2026-06-03_calibration-study-design.md);
this records what the runs actually showed and what it means for the next phase
(synthetic data generation). The per-experiment numbers live in each
`EXPERIMENT.md`.

## 1. The detection-free, lipid-only objective is validated

Moment-matching on lipid-channel summary statistics (pixel histogram, radial
PSD, mean, high quantiles, skewness) reproduces the real images well:

- The bright-spot **tail quantiles** (p99 / p99.9) match to within ~6%.
- The **radial PSD** of real and simulated images overlaps across spatial
  frequencies (the structural term).
- Real vs simulated lipid crops are **visually indistinguishable**
  (see each run's `plots/<sample>/comparison_images_lipid.png`).

Conclusion: detection-free lipid moment-matching is sufficient to fit the forward
model; we keep it (no per-spot detection in the calibration loop).

## 2. Which parameters are identifiable vs degenerate

Pooling the per-sample fit, the bootstrap stability study (100 × d=25 subsets of
25nM_endophilin), and the weight-sweep:

**Well-determined / identifiable** (bootstrap CV ≲ 17%, and move ≲ 9% across
objective weightings):
- `psf_sigma_x`, `psf_sigma_y` — PSF widths (bootstrap CV ~11–12%).
- `lipid_brightness` (bootstrap CV ~17%).
- `spot_density` (bootstrap CV ~17%).

**Moderate:**
- `enf` — excess-noise factor (bootstrap CV ~21%).

**Degenerate / weakly constrained:**
- `gain` — bootstrap CV ~35%, wanders ~85% across weightings, and the per-sample
  fits cluster in a narrow ~11–22 band that is ~15–40× below the photon-transfer
  *measured* gain (~270–450) and does **not** track the 561 detector voltage.
  Gain trades off against `enf` (and `lipid_brightness`): only the gain/enf
  *product* (the effective noise scale) is constrained, not gain alone.
- `optical_bg_lipid` — bootstrap CV ~83%; small in absolute terms (~0–7 photons).
- `psf_theta` — bootstrap CV ~560% (values scatter around 0); the PSF is
  essentially circular so the rotation angle is ill-defined.

## 3. Consequences for synthetic-data generation

- **Trust and use directly:** PSF widths (`psf_sigma_x`, `psf_sigma_y`),
  `lipid_brightness`, and `spot_density`. These are robust to image subset and to
  objective weighting.
- **Treat `gain` and `enf` as a degenerate pair.** Do not fix them at their
  point estimates and do not present the fitted `gain` as a physical gain
  measurement. For generation, **randomize over their fitted ranges** (jointly,
  since their product is what matters) so the trained detector sees the full
  plausible noise scale rather than one arbitrary split of it.
- **`psf_theta` and `optical_bg_lipid` are weakly constrained.** Sample
  `psf_theta` freely (the PSF is near-circular anyway) and keep
  `optical_bg_lipid` small; neither should be treated as a precise measurement.
- **Do NOT claim to physically measure the PMT gain** from this calibration. The
  photon-transfer `measured_params.gain` (~270–450) is a separate estimate; the
  fitted gain is an effective noise-model parameter, not a calibrated detector
  gain.

## 4. Per-sample calibration, not joint

Confirmed by the data: the fitted lipid parameters (`lipid_brightness`,
`spot_density`, PSF widths) genuinely differ per sample, and the 561 detector
voltage was set per sample. A single joint fit would force one gain/brightness
across settings that do not share them. We therefore calibrate **each sample
independently** (the `per_sample` study), as decided in the design doc — and the
results bear out that the shared-microscope assumption of a joint fit would be
wrong for `gain` in particular.
