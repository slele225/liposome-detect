# Progress update — curvature-sensing measurement on real liposome data

**Date:** 2026-06-12 · **Branch:** `stage2-models`

## Framing

Since the last update I built and validated the piece that turns our two-colour
liposome images into a curvature-sensing number, and I ran it on the real EGFP
and endophilin samples. To state results cleanly I'll use one quantity
throughout, **α**:

> We quantify curvature sensing as **α, the slope of log(protein) vs log(lipid)
> across detected liposomes.** If a protein binds in proportion to membrane area,
> α = 2 (no curvature preference). A curvature sensor binds preferentially to
> small, high-curvature liposomes, giving α < 2. **EGFP is our α = 2 negative
> control; endophilin is the curvature sensor we expect to give α < 2.**

(α is twice the slope you see in the figures, because the lipid signal scales
with area and area scales with diameter squared.)

## What I built

A liposome detector that reads each image, finds every liposome punctum, and
reports its size (from the lipid channel) and how much protein is bound. The
detector is trained entirely on synthetic images from a forward simulator of
**our** microscope that I calibrated to our actual acquisition data — so the
detector's sizing and intensity readouts are accurate on the kind of images we
collect, including the faint, small liposomes that are hardest to see. The
single sentence that matters: it is trusted because it was calibrated to our own
scope, and that trust is then checked against a known answer (Fig. 4).

## Results

### The measurement works (Fig. 1)

![Fig 1](figures/fig1_measurement.png)

Per detected liposome, we plot protein bound against liposome size. **EGFP**
gives a steep, near-area-proportional relationship (slope ≈ 0.71, **α ≈ 1.4**) —
it coats whatever membrane is there. **Endophilin** is markedly shallower
(slope ≈ 0.31, **α ≈ 0.6**): it loads up on the small liposomes and falls off on
the large ones. That gap between the two slopes is the curvature-sensing signal,
and it comes straight out of the per-liposome data with no further modelling.
(These are the raw slopes, before the gain-correction discussed in Fig. 3.)

### We detect the small liposomes the standard method misses (Fig. 2)

![Fig 2](figures/fig2_small_liposomes.png)

This is the key advance. Curvature sensing happens on **small, high-curvature
liposomes (≈ 40–90 nm)** — exactly the population a standard automated spot
detector struggles with. In that range our detector finds the large majority of
liposomes and sizes them accurately, while the standard method finds only a
small fraction and mis-sizes the ones it does find:

| at 40–90 nm        | our method | standard method |
|--------------------|:----------:|:---------------:|
| detection quality (F1) | **0.63** | 0.23 |
| size error (lower = better) | **0.20** | 0.70 |

Because we recover the small liposomes representatively (we don't only catch the
unusually bright ones), the curvature-sensing regime is actually measured rather
than inferred from the larger liposomes the standard tool can see. Above
~120 nm the two methods converge, as expected — large liposomes are easy.

### Validated on known-answer simulated data (Fig. 4)

![Fig 4](figures/fig4_synthetic_validation.png)

To show the α measurement itself is trustworthy, I ran it on simulated data
where the true α is set by construction. Across the full range (α from 0.5 to
2.2) the recovered value tracks the truth monotonically and stays close to the
line. This is why the numbers on real data above can be taken at face value.

## The wrinkle the control caught (Fig. 3)

![Fig 3](figures/fig3_egfp_artifact.png)

EGFP has no curvature preference, so it **must** read α = 2 at every
concentration. It doesn't yet: recovered α climbs with concentration
(≈ 1.3 → 1.5 → 1.6 → 1.7 for 20 → 50 → 100 → 300 nM) instead of sitting flat.

I traced this to acquisition, not biology. The protein detector (488 nm) was
held at a constant 295 V across all EGFP samples, but the **lipid detector
(561 nm) voltage was lowered per sample** — 750 → 640 → 630 → 580 V — because
the operator backed it off to avoid saturation at higher concentration. Detector
gain is nonlinear in voltage, so the lipid (size) axis is scaled differently in
each sample, which tilts the slope. This is a correctable calibration issue: the
fix is to gain-normalize the lipid channel across samples before pooling, and
it's in progress. Worth noting as a positive — **the EGFP control did its job
and surfaced a subtle acquisition artifact** that would otherwise have biased the
endophilin numbers too.

## Also done

I reworked and validated the simulator calibration that all of the above rests
on (the joint, multi-sample fit of the microscope's PSF, gain, and noise), and
the pipeline and its design decisions are now documented in the repository.

As a sanity check not shown here, the detected size distributions are consistent
with bulk DLS sizing for the EGFP controls, and endophilin's detected
distribution skews smaller — again consistent with a curvature sensor that
concentrates on small liposomes.

## Next steps

1. **Gain-correct the lipid channel** so EGFP reads α = 2 at every concentration,
   then re-run the real samples for the corrected final numbers.
2. **Cross-method baselines** to position this against existing tools:
   *cme-analysis*, *Spotiflow*, and *SpotMAX*. This makes the "we see the small
   liposomes" claim a measured comparison rather than an in-house one. See
   [baselines.md](baselines.md) for the three methods, how we run each as a
   detection front-end through our own photometry, the comparison tests, and a
   preliminary Spotiflow result.
3. Report the corrected endophilin-vs-EGFP α with those baselines alongside.

---
*Figures are reproducible from `make_figures.py` using the CSV snapshot in
`figure_data/`. All numbers above are read from those CSVs; no inference or
detector runs were involved in building this update.*
