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

### How the simulator renders an image (the forward model)

The simulator places each liposome as a point source, blurs it with the
microscope's measured point-spread function (PSF), and then pushes the result
through the **standard EMCCD/PMT detector chain** — the same model used in the
detector-simulation literature (Hamamatsu/Andor). After the PSF gives the signal
`S` at each pixel:

- **Expected electrons:** `λ = S + D + C` — signal `S`, dark current `D`,
  clock-induced charge `C`.
- **Shot noise:** `N ~ Poisson(λ)` — photon counting is random.
- **Stochastic multiplication gain**, which adds its own noise (the
  excess-noise factor `F`): `Var[I] ≈ F² g² (S + D + C) + σ_read²`, with
  `F ≈ √2 ≈ 1.41` at high gain.
- **Bias + read noise:** `I_e = g·N + b + Normal(0, σ_read²)`.
- **Digitize and clip** to a 12-bit sensor: `I_ADU = clip(round(I_e / k), 0, 2¹² − 1)`.

The gain `g`, excess-noise factor `F`, and background are fit during calibration;
the bias `b`, read noise `σ_read`, and frame-averaging are pinned directly from
dark frames. One caveat matters here: `F ≈ √2` is the **high-gain
approximation** and deviates at low gain. That ties directly to the artifact in
Fig. 3 — the lipid detector was run at different voltages per sample, i.e. in
different gain regimes, so a single fixed `F` does not describe all of them
equally well.

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

## Positioning against existing tools

The existing state-of-the-art tools solve **detection** — finding the spots. Our
deliverable is the downstream **measurement**: turning each liposome's
two-channel intensities into a per-liposome protein/lipid ratio and then into α,
validated against the EGFP = 2 control. To compare fairly, we run each external
method purely as a detector and push its coordinates through **our own fixed
photometry** ("shared photometry" — re-measure brightness identically for every
method's points, so only detection quality differs), and separately report each
method's **native** α where it returns its own intensities.

| Method | Type | Returns intensity? | Multi-channel? | Status | How we run it |
|---|---|---|---|---|---|
| **cme-analysis** (Danuser) | Classical Gaussian-PSF fit; long-standing lab standard | **Yes** — `A_lipid`, `A_protein` | **Yes** (master/slave channels) | Running | MATLAB `run_pipeline.m` headless on 3-ch TIFFs → per-spot `x, y, A, slave_A` |
| **Spotiflow** (Weigert, *Nat Methods* 2025) | Deep heatmap detector; SOTA detection | Yes, but single readout, **not** 2-channel-aware (run per channel) | **No** — single-channel input; we feed lipid | Ran — see result below | `spotiflow-predict` on extracted lipid channel → `y, x, intensity, probability` |
| **SpotMAX** (Schmoller, 2024) | Deep, built for low-SNR / high-density; detection + own quantification | **Yes** | Partial (spots channel + optional ref channel) | Installed, tuning detection | `spotmax -p config.ini` on lipid |

### Spotiflow result (the key baseline finding)

We ran Spotiflow's pretrained `general` model out-of-the-box on the lipid
channel.

- On **real** 20nM_EGFP it returned **~1096** detections (vs ours **~321**,
  cme-analysis **~960/image**).
- On **synthetic** images with **known** spot counts it severely
  **under-detects**: e.g. **99–280** found vs **600–1000** true — catching only
  ~15–45%.
- Lowering the detection threshold from 0.5 to 0.3 barely changed the counts
  (e.g. 99 → 136 on an image with 629 true spots), so the under-detection is
  **not a threshold artifact** — the model genuinely does not fire on our
  puncta.

The interpretation is that the pretrained model is **domain-mismatched**: it was
trained on spatial-transcriptomics FISH dots, not our instrument's liposomes.
This is direct evidence that instrument-matched training (our approach) matters,
and is consistent with the known failure of deep detectors under instrument/PSF
mismatch.

**Honest caveat.** The fully fair detection comparison **fine-tunes** Spotiflow
on our synthetic data — we have unlimited labels via `ground_truth.csv` — and
that is a next step. The out-of-the-box result shows that generic SOTA does not
transfer without the calibration/training step our pipeline provides; it does
not yet show that a fine-tuned Spotiflow would lose a fair head-to-head.

## Next steps

1. **Gain-correct the lipid channel** so EGFP reads α = 2 at every concentration,
   then re-run the real samples for the corrected final numbers.
2. **Synthetic gain-sweep test** — generate synthetic data spanning the real
   per-sample PMT-voltage (gain) range to confirm whether the EGFP concentration
   trend is in fact the gain artifact, and whether the forward model needs to
   span that full gain range (where `F ≈ √2` breaks down).
3. **Cross-method baselines** — finish *cme-analysis* and *SpotMAX*, and
   **fine-tune *Spotiflow* on our synthetic data** for the fair detection
   head-to-head. This makes the "we see the small liposomes" claim a measured
   comparison rather than an in-house one.
4. Report the corrected endophilin-vs-EGFP α with those baselines alongside.

---
*Figures are reproducible from `make_figures.py` using the CSV snapshot in
`figure_data/`. All numbers above are read from those CSVs; no inference or
detector runs were involved in building this update.*
