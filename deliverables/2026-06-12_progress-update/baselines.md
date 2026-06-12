# Baseline methods and how we compare against them

**Date:** 2026-06-12 · **Branch:** `stage2-models`

## Framing

The existing state-of-the-art tools solve **detection** — finding the spots.
Our deliverable is the downstream **measurement**: turning each liposome's
two-channel intensities into a per-liposome protein/lipid ratio and then into
the curvature exponent **α**, validated against the EGFP = 2 negative control.
Detection is a front-end to that, not the product.

So we benchmark fairly in two ways. We run each external method purely as a
detector and push its coordinates through **our own fixed photometry + α fit**
("shared photometry" = re-measure brightness identically for every method's
points, so only detection quality differs). Separately, where a method returns
its own intensities, we also report its **native end-to-end α**.

## The three baselines

| Method | Type | Returns intensity? | Multi-channel? | Status | How we run it |
|---|---|---|---|---|---|
| **cme-analysis** (Danuser) | Classical Gaussian-PSF fit; long-standing lab standard | **Yes** — `A_lipid`, `A_protein` | **Yes** (master/slave channels) | Running | MATLAB `run_pipeline.m` headless on 3-ch TIFFs → per-spot `x, y, A, slave_A` |
| **Spotiflow** (Weigert, *Nat Methods* 2025) | Deep heatmap + stereographic-flow detector; SOTA detection | Yes (`details.intens`), but single readout, **not** 2-channel-aware (run per channel) | **No** — single-channel input; we feed lipid | Running (preliminary result below) | `spotiflow-predict` on extracted lipid channel → `y, x, intensity, probability` (note **y,x** order; single-channel intensity only) |
| **SpotMAX** (Schmoller, 2024) | Deep U-Net, built for low-SNR / high-density; detection + own quantification | **Yes** | Partial (spots channel + optional ref channel) | Installed, tuning detection | `spotmax -p config.ini` on lipid |

## Preliminary Spotiflow result

We ran Spotiflow's pretrained `general` model out-of-the-box on the lipid
channel. On **real** 20nM_EGFP it returned ~1096 detections (vs our ~321,
cme-analysis ~960/image). But on **synthetic** images with **known** spot
counts it under-detects badly: ~99–400 found vs ~600–1000 true (catching only
~15–40%).

Interpretation: the pretrained model is **domain-mismatched** — it was trained
on spatial-transcriptomics FISH dots, not our instrument's liposome puncta.
This is consistent with the known failure mode that deep detectors degrade under
instrument/PSF mismatch, and is direct evidence for why **instrument-matched
training** (our approach) matters.

**Honest caveat.** The fully fair detection comparison fine-tunes Spotiflow on
our synthetic data — we have unlimited labeled synthetic images via
`ground_truth.csv`. The out-of-the-box result shows that generic SOTA does not
transfer without the calibration/training step our pipeline provides; a
**fine-tuned** Spotiflow is the proper detection head-to-head and is a next
step. We also need to verify the under-detection is not merely a
probability-threshold / intensity-scale effect before over-claiming (checked
separately).

## Comparison tests (what we compute for each method)

1. **Detection vs diameter** — precision / recall / F1, binned by true diameter,
   on the synthetic ground-truth export. Emphasis on the small bins (40–90 nm).
2. **Intensity-recovery vs diameter** — same binning, for methods that return
   intensity.
3. **Alpha recovery**, two ways: **(a) shared** — the method's coordinates →
   our fixed photometry + EIV + calibration → α (isolates detection quality);
   **(b) native** — the method's own intensities → α (end-to-end). Run on
   synthetic (vs known α) and anchored to real EGFP = 2.
4. **EGFP = 2 anchor** — does each method recover the negative-control α, and
   does it reproduce the concentration trend (the PMT-gain diagnostic from
   Fig. 3 of the progress update)?
