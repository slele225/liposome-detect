# Protein-channel parameterization (2026-06-04)

Decision record for how the protein (488 nm, channel 0) channel is parameterized
for synthetic-data generation in Stage 2. The lipid channel was calibrated by
joint moment-matching (see `2026-06-03_calibration-findings.md`); this records
why the protein channel is **not** calibrated the same way, and what is done
instead.

## Why the protein channel is not moment-matched

The lipid channel was moment-matchable because, to first order, every liposome
puncture is the same kind of object: brightness scales with membrane, the spatial
pattern is scattered spots, and a liposome's *size* does not change the per-spot
lipid physics beyond brightness. Global summary statistics (pixel histogram,
radial PSD, quantiles, skewness) therefore capture the lipid channel fully.

The protein channel is fundamentally different, and that difference is the entire
point of the assay. Protein signal on a liposome depends on membrane curvature,
i.e. on liposome size: the curvature-sensing law (`curvature_alpha`) is precisely
the relationship between per-puncta protein intensity and liposome diameter.

**The protein intensity-vs-size pairing IS the curvature signal — and global
summary statistics discard exactly that pairing.** A histogram of all protein
puncta intensities does not record which intensity came from which size liposome.
Moment-matching the protein channel would therefore either be uninformative about
alpha or, worse, absorb the curvature effect into a wrong `protein_brightness`,
yielding a generator that is confidently wrong. So we do not moment-match the
protein channel for brightness or alpha.

## What is measured (done, not fitted by Optuna)

- **Protein PSF:** measured directly from isolated protein spots, ~1.85–1.90.
  Not an Optuna parameter. Done.
- **Protein dark floor:** measured from dark frames. Done.
- **Protein optical background:** ~0 (samples washed). Done.

## What is randomized during generation (not fitted)

- **`protein_brightness`:** randomized per image over a wide, realistic range.
  Randomizing absolute scale makes the detector robust to absolute brightness
  rather than tuned to one value. An EGFP-control sanity check (protein intensity
  should be flat vs liposome size, since EGFP has no curvature sensing) is
  **optional insurance**, not a required calibration step.
- **`curvature_alpha`:** randomized per image over a plausible range (e.g.
  0.5–2.0). Alpha is **not fitted anywhere** in calibration or generation.

## Where alpha is actually measured: downstream, from real data

Alpha is a *measurement output* of the whole pipeline, not an input to it. It is
obtained by running the trained detector on **real** images and fitting the
log–log sorting curve (protein enrichment vs diameter). The diameter information
needed to form that curve comes from DLS injection (Stage 2, Phase 3), which is
the only stage where per-puncta size is available — confirming that alpha could
not have been fit earlier even in principle.

## Critical constraint: the detector must stay agnostic to alpha

The detector is trained to find puncta and report their properties; it must never
be allowed to assume or learn a particular alpha. Because alpha is randomized
per training image, the network sees the full plausible range and cannot lock
onto one value — this is by design.

**If a sorting-curve loss is used later, it must not let the network learn a
particular alpha.** Baking a fixed (or learnable-toward-fixed) alpha into a
training loss would contaminate the downstream measurement: the detector would
report curvature sensing partly because it was trained to expect it, not because
the real data shows it. The sorting curve is a *measurement* made on the
detector's outputs over real data, and it must remain so. Any training-time use
of sorting information must be constructed to be alpha-agnostic (e.g. it must not
supply or reward a specific enrichment-vs-size slope).

## Summary

| Param                | Status            | How obtained                          |
|----------------------|-------------------|---------------------------------------|
| protein PSF          | measured, done    | isolated-spot fit (~1.85–1.90)        |
| protein dark floor   | measured, done    | dark frames                           |
| protein optical bg   | measured, ~0      | washed samples                        |
| protein_brightness   | randomized        | wide realistic range per image        |
| curvature_alpha      | randomized        | plausible range per image (e.g. 0.5–2)|
| alpha (the result)   | measured downstream | log–log sorting curve on real data, post-detection |
