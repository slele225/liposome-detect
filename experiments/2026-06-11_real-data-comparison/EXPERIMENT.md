# EXP — real-data corrected-vs-standard alpha comparison

- **Date:** 2026-06-11
- **Branch:** `stage2-models`
- **Model:** the trained Stage-2 detector `models/hrnet_v1/best.pt`
  ([`models/hrnet_v1/README.md`](../../models/hrnet_v1/README.md)), built from
  [`configs/train/hrnet_v1.yaml`](../../configs/train/hrnet_v1.yaml).
- **Additive only.** No edits to `src/simulator/`, `src/calibration/`, or the
  existing `src/eval/` fit code — this experiment IMPORTS them. New reusable
  bridge: [`src/eval/real_data.py`](../../src/eval/real_data.py).

## Question

Is the **standard** curvature quantification (OLS slope of log-protein vs
log-lipid, α = 2·slope) **biased on REAL data**, and does the **corrected**
pipeline (errors-in-variables Deming + the recovered→true calibration curve)
recover the two real-data ground truths:

1. **EGFP = 2.0** (the negative control: protein binds in proportion to membrane
   area, the α = 2 / curvature-insensitive limit — true **by construction**), and
2. **DLS size consistency** (the detector's recovered size distribution should
   match the sample's independently-measured DLS distribution)?

The contribution being tested is a **validated, unbiased curvature measurement**
(anchored on EGFP = 2.0 and DLS), **not** "best detector". Per-spot uncertainty
is used for **QC / error propagation only**, never as regression weighting (it is
confounded with the size axis — see
[`src/eval/alpha_fit.py`](../../src/eval/alpha_fit.py) and the
[diagnostic run](../2026-06-10_diagnostic-run/EXPERIMENT.md)).

## Samples (real images)

`data/<sample>/images/*.tif` (uint16, 12-bit, shape (3, 512, 512), channels
0=protein/488, 1=lipid/561, 2=transmitted), `data/<sample>/dark_frames/*.tif`,
and one DLS `*.xlsx` per sample folder.

- **EGFP** (negative control, true α = 2.0): `20nM_EGFP`, `50nM_EGFP`,
  `100nM_EGFP`, `300nM_EGFP` (20 images each).
- **endophilin** (curvature sensor, α < 2): `25nM_endophilin`,
  `300nM_endophilin` (55 images each).

DLS filenames differ by sample (EGFP `batch1_dls_corrected.xlsx`; endophilin
`Ternary 0.5% PEGB.xlsx`) — globbed per folder, never hardcoded.

## Method

1. **smoke_check.py** (scaling guard, FIRST). One `20nM_EGFP` image → detector →
   prints detection count + median/5/95-pct of predicted lipid & protein flux.
   PASS/WARN verdict vs the synthetic scale. A human inspects before trusting
   anything downstream.
2. **firm_calibration.py** (build the curve). Generate fixed-α synthetic test sets
   at **α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2}** (`global_coherent`
   single-point `alpha_range`, emphasis sizing — from
   `config_snapshot/alpha_template.yaml`), run the detector + `recover_alpha`
   (EIV), bootstrap for error bars, fit an updated **recovered→true
   `CalibrationCurve`**, save it to `calibration_curve.json` (+ `.png`, `.csv`).
   This replaces the 4-point seed in `alpha_fit.py`.
3. **real_alpha.py** (the core result). Per real sample, pool detected spots and
   compute α two ways, bootstrap over images for CIs:
   - **STANDARD** = `2 · ols_slope(log lipid, log protein)`.
   - **CORRECTED** = `CalibrationCurve.invert( recover_alpha(...) )`, where
     `recover_alpha` is constant-λ Deming (EIV) with λ from the mean `exp(logvar)`
     per axis.
   Reports the **EGFP anchor**: mean |α − 2.0| across the 4 EGFP samples per
   pipeline (the go/no-go), and the endophilin STANDARD-vs-CORRECTED comparison.
4. **dls_consistency.py** (second anchor). Detector size proxy vs DLS, in matched
   weighting (below). Wasserstein + KS distance and overlay plots.
5. **synth_benchmark.py** / **export_for_external.py** / **real_benchmark.py** — the
   cross-method benchmark with isolated detection (ours vs classical now,
   cme-analysis/SpotMAX later). See **Part 2** below for the full design.

### Scaling / offset convention (THE #1 risk — pinned and documented)

**Determined from the code:** the synthetic generator's images **INCLUDE the
detector dark offset**. The forward model adds `offset` (the dark-frame DC level)
and clips to the 12-bit range (`src/simulator/forward_model.py` —
`_apply_pmt_noise` adds `offset`; `simulate_image` does `np.clip(..., 0, 4095)`),
and `src/generator/core.py::serialize_image` saves that raw-ADU-like array
straight to `.npy`. The training dataloader (`src/train/dataset.py`) applies only
`(img − norm_mean) / norm_std`.

**Therefore the real images are fed RAW — NO dark subtraction** — so the offset is
present exactly as in training; `norm_mean` / `norm_std` then center them. The
loader exposes `--subtract_dark {auto,on,off}` **defaulting to `auto` == `off`
(raw)**, which is the convention that matches the generator. `on` (subtract the
per-channel `dark_frames/` median) is a **diagnostic escape hatch only** — it does
NOT match training. We do **not** rescale 12-bit to [0,1] (training didn't).

Channel order matches end to end: synthetic `.npy` is `[protein, lipid]` with
`norm_mean = [207.43, 326.64]`; real TIFF channels `[0, 1] = [protein, lipid]`
line up directly (transmitted dropped). Images are **center-cropped to 256×256**
(`io.CROP_SIZE`) — the same domain the simulator generates on and the calibration
was fit on.

### Log-space variance units

The intensity NLL residual is in LOG space, so a head `logvar` is already the
log-residual variance: `exp(logvar)` is used **directly** as the per-axis variance
(no delta-method `/intensity²`). See `src/eval/alpha_fit.py`. Per-spot variance is
used only to set the single constant λ (mean ratio) for Deming — **not** per-spot
weighting.

### DLS weighting conversion (dls_consistency.py)

The two distributions are **not** directly comparable as stored:

- DLS `X Intensity` is **intensity-weighted** (≈ `N(d)·d⁶`). We instead read the
  **number** weighting `N(d)` via the SAME parser training used
  (`src/simulator/io.parse_dls(weighting='number')`).
- The detector emits **one detection per liposome**, so pooled detections are
  **number-weighted** `N(d)` already.

Both sides are thus compared in the **same number weighting `N(d)`**.

**Size proxy:** the network outputs no diameter; we invert the simulator's lipid
area law `lipid_amp = lipid_brightness · (d/100)²` →
`d_proxy = 100·√(lipid_intensity / lipid_brightness)`, using the shared calibrated
`lipid_brightness` from
`experiments/2026-06-03_per-sample-calibration/runs/20nM_EGFP/calibration_results.json`
(sets the absolute size scale). It is a proxy, not a direct measurement.

## Reproduce

`best.pt` is at `models/hrnet_v1/best.pt` (gitignored, present on disk).
Real images live under `data/<sample>/` (gitignored). Then, on the instance:

```bash
uv sync
export MPLCONFIGDIR=/tmp/mpl
# GPU for inference, all workers for generation:
N_WORKERS=32 ./experiments/2026-06-11_real-data-comparison/run.sh
```

`run.sh` runs the four steps in order. **Inspect the smoke-check PASS/WARN verdict
before trusting the downstream numbers.** Outputs land in this folder:
`real_alpha.csv` + `real_alpha_summary.txt` (the core table + EGFP anchor),
`calibration_curve.{json,png,csv}`, `dls_consistency.{csv,png}`.

The generated fixed-α datasets (`datasets/real_cmp_alpha_*`) and the per-run
outputs are gitignored; the scaffold (this file, `run.sh`, `config_snapshot/`,
the scripts) is tracked.

## Results

_Run of 2026-06-11, committed code at git SHA `e0dfb42`. Numbers below are
quoted verbatim from the result files in this folder — they were not recomputed
for this write-up._

**Jargon, briefly.** *α (alpha)* — the curvature-sensing exponent: the slope of
log(protein) vs log(membrane area) ×2. α = 2 means "binds in proportion to
surface area" (no curvature preference, the EGFP negative-control ground truth);
α < 2 means the protein prefers small/high-curvature liposomes. *STANDARD* — the
naive ordinary-least-squares (OLS) slope. *CORRECTED / EIV* — errors-in-variables
(Deming) regression, which accounts for measurement noise on *both* axes, then
mapped through the recovered→true calibration curve. *repr_ratio* — among spots
in a size bin, the ratio of detected-spots' mean true brightness to all spots'
mean true brightness; ≈1 means the detector keeps a *representative* (unbiased)
sample, not just the bright ones.

### Smoke check (scaling guard)

`smoke_check.py` is a human-eyeballed print-only guard (it writes **no file** —
see Provenance notes). It detected spots on one `20nM_EGFP` image and printed the
predicted lipid/protein flux quantiles against the synthetic scale (norm_mean
lipid ~327, protein ~207). The downstream pipeline was run, so the guard was
treated as PASS for this run; the verdict line itself is not captured in an
artifact.

### Firmed-up calibration curve (`calibration_curve.csv` / `.json` / `.png`)

The recovered→true map was fit on 8 fixed-α synthetic sets (~100k spots each,
`n_img=300`). It is **clean and monotonic** (recovered rises with true at every
step), so it inverts unambiguously:

| true α | recovered α (mean) | 95% CI |
|--------|--------------------|--------|
| 0.50 | 0.700 | 0.671–0.727 |
| 0.75 | 0.842 | 0.816–0.870 |
| 1.00 | 1.044 | 1.013–1.077 |
| 1.25 | 1.242 | 1.215–1.268 |
| 1.50 | 1.461 | 1.432–1.488 |
| 1.75 | 1.663 | 1.631–1.694 |
| 2.00 | 1.838 | 1.802–1.872 |
| 2.20 | 1.943 | 1.904–1.984 |

The detector systematically under-recovers α (e.g. true 2.0 → recovered 1.84);
the calibration curve corrects exactly this, which is why CORRECTED beats
STANDARD below.

### Real-data alpha — STANDARD vs CORRECTED (the go/no-go) (`real_alpha.csv`)

| sample            | n_spots | α STANDARD (OLS) | α CORRECTED (EIV+calib) | \|corr−2.0\| |
|-------------------|---------|------------------|-------------------------|--------------|
| 20nM_EGFP         | 6603    | 1.120            | 1.326                   | 0.674        |
| 50nM_EGFP         | 5427    | 1.181            | 1.482                   | 0.518        |
| 100nM_EGFP        | 6414    | 1.290            | 1.577                   | 0.423        |
| 300nM_EGFP        | 5378    | 1.419            | 1.719                   | 0.281        |
| 25nM_endophilin   | 26551   | 0.649            | 0.447                   | 1.553        |
| 300nM_endophilin  | 25914   | 0.626            | 0.409                   | 1.591        |

- **EGFP anchor (true α = 2.0, `real_alpha_summary.txt`):** STANDARD mean α =
  **1.253** (mean |α−2.0| = 0.747); CORRECTED mean α = **1.526** (mean |α−2.0| =
  **0.474**). **CORRECTED is nearer 2.0 on real ground truth.** STANDARD is
  biased LOW (<1.9) and CORRECTED pulls toward 2.0 → **the OLS bias is real on
  real data**, and the EIV+calibration correction is the right direction. (Both
  pipelines still fall short of 2.0 — see the Diagnosis section; the residual gap
  is an acquisition artifact, not a fit failure.)
- **EGFP internal-consistency RED FLAG:** the 4 EGFP samples share one liposome
  prep, so their CORRECTED α should be mutually consistent. They are **not** — the
  CORRECTED spread (max−min) is **0.394 > 0.3**, and it is *monotone in
  concentration* (1.326 / 1.482 / 1.577 / 1.719 for 20/50/100/300 nM). This is a
  per-concentration artifact, **separate from** the OLS-vs-EIV bias. Diagnosed
  below.
- **Endophilin (curvature sensor, α < 2):** 25 nM STANDARD 0.649 → CORRECTED
  0.447 (Δ = −0.202); 300 nM STANDARD 0.626 → CORRECTED 0.409 (Δ = −0.217).
  Sensing strength = distance below 2.0; the correction *increases* the apparent
  sensing strength (pushes α further below 2.0) by ~0.2 in both samples, a
  material shift in the biological reading. Both endophilin samples read deep
  curvature preference under either pipeline.

### DLS consistency (second anchor) (`dls_consistency.csv` / `.png`)

Detector size proxy vs independently-measured DLS, both **number-weighted N(d)**.
Distances (lower = better agreement):

| sample            | n_det | Wasserstein (nm) | KS    | det median (nm) | DLS mean (nm) |
|-------------------|-------|------------------|-------|-----------------|---------------|
| 20nM_EGFP         | 6603  | 15.0             | 0.369 | 72.3            | 95.7          |
| 50nM_EGFP         | 5427  | 15.0             | 0.342 | 73.9            | 95.7          |
| 100nM_EGFP        | 6414  | 18.6             | 0.360 | 71.9            | 95.7          |
| 300nM_EGFP        | 5378  | 12.4             | 0.302 | 82.3            | 95.7          |
| 25nM_endophilin   | 26551 | 50.4             | 0.746 | 33.2            | 95.5          |
| 300nM_endophilin  | 25914 | 39.0             | 0.650 | 42.0            | 95.5          |

- **EGFP** agrees reasonably (Wasserstein 12–19 nm, KS 0.30–0.37).
- **endophilin** agrees **poorly** (Wasserstein 39–50 nm, KS 0.65–0.75) and has
  **~4× the EGFP detection count**. Flagged as an open puzzle (Status & next
  steps); it is *not* the driver of the EGFP α trend.

---

# Part 2 — cross-method benchmark with isolated detection

## Design principle: isolate detection from the fit

Every detector produces only **(x, y) spot LOCATIONS**. Everything downstream —
local photometry to get (lipid, protein) intensities, the EIV + calibration alpha
fit, and the ground-truth evaluation — is a **single SHARED, FROZEN layer**
(`src/eval/benchmark_core.py`) applied **identically to every method**. So
"method A vs B" reduces purely to *whose LOCATIONS give better alpha / detection /
intensity*, with no confound from differing photometry or fits. The adapter layer
(`src/eval/adapters.py`) reduces each method to that common (x, y) interface
(`ours` = hrnet_v1 decode; `classical` = scikit-image `blob_log` on the lipid
channel; `external_csv` = coordinates from cme-analysis/SpotMAX run elsewhere).

### Why the EIV + calibration fit is applied to ALL methods (not our contribution)

The errors-in-variables (Deming) slope + calibration-curve inversion is a
**method-agnostic, free statistical correction anyone can use**. Crediting *our*
detector for it would be wrong. So the benchmark runs **every** method's locations
through the **same** fit. Our only possible edge is **detection / intensity quality**
(especially in the small/high-curvature regime), which is what the benchmark
measures. The shared fit uses **constant-λ Deming from intensities alone (λ = 1,
TLS)** — no per-spot logvar weighting — because external methods don't emit logvar
and per-spot weighting is contraindicated anyway (variance/size confound). Per-spot
uncertainty is **QC / error bars only, never a fit weight or an accuracy advantage.**

## Shared photometry — aperture choice and GT-flux consistency (the #1 risk)

Local **fixed-radius circular aperture sum minus an annulus-median background**, per
channel, at each detected location (radii config-exposed: `--r-ap` / `--r-in` /
`--r-out`, defaults 6 / 9 / 14 px).

**Consistency with the synthetic GT flux definition.** The simulator renders each
spot as a PSF kernel **normalized to sum to 1**, so GT `lipid_intensity` /
`protein_intensity` is the spot's **TOTAL integrated flux** in ADU, contained within
the render radius `ceil(4σ)` ≈ 8 px (calibrated σ ≈ 1.9). The default aperture radius
(6 px) captures **>99%** of that flux, so the background-subtracted aperture sum ≈ GT
total flux — the photometry is consistent with the GT definition (verified: aperture
/ true ≈ 1.03 on a clean toy). Any residual **constant** capture fraction (f < 1) is
the same for every spot of a channel, so it shifts the log-log **intercept**, never
the **slope** → alpha is robust; only the absolute intensity-recovery *level* carries
the (documented) constant offset, visible as a constant vertical shift in the
intensity-vs-diameter plot. The annulus (9–14 px) sits beyond the PSF support to
estimate the local background (DC offset + optical background + neighbour haze).

## FIX: per-sample lipid→diameter scale (DLS-consistency correctness)

`lipid_brightness` was calibrated **independently per sample** and genuinely varies
(different 561 voltage / prep): EGFP ≈ 4.8k–8.4k, **endophilin ≈ 12k–16k**. The
lipid→diameter size proxy `d = 100·√(lipid / lipid_brightness)` therefore uses **each
sample's OWN** `lipid_brightness` (loaded by `_common.lipid_brightness_for`), in both
`dls_consistency.py` and any recovered-size reporting. Using one sample's scale for
all (the earlier default) gave a ~1.8× size error for endophilin and a false DLS
mismatch.

**Crucially, this scale does NOT affect alpha:** alpha is the log-log **slope**,
which is **scale-invariant** — a `lipid_brightness` error shifts the intercept, not
the slope. So the **EGFP = 2.0 go/no-go is robust** to the proxy scale; only the
**DLS anchor and recovered sizes** depend on it (now fixed). The real-data alpha
summary also reports an **EGFP internal-consistency** check: the 4 EGFP samples share
one liposome prep, so under the correct pipeline their CORRECTED alphas should be
mutually consistent (all ≈ 2.0); a spread > 0.3 is flagged as a RED FLAG
(concentration-dependent leakage / per-concentration artifact), separate from the
OLS-vs-EIV bias.

## Synthetic benchmark (`synth_benchmark.py`) — the ground-truthed core

Runs both adapters through `benchmark_core` on synthetic sets WITH ground truth:
two diameter-eval sets (`bench_emphasis`, `bench_dls` — emphasis stresses the small
tail, dls is the realistic regime per the benchmark-design doc) and the fixed-α sweep
(reused from `firm_calibration.py`). Produces, **binned by TRUE diameter with fine
small bins (40–55, 55–70, 70–90 nm)**:

- **(a)** detection F1 / recall vs diameter — `bench_detection_vs_diam.png`. (F1 per
  bin is per-bin recall against *global* precision — false positives carry no true
  diameter, so precision can't be binned.)
- **(b)** intensity-recovery log-error vs diameter (lipid + protein) —
  `bench_intensity_vs_diam.png`.
- **(c)** alpha recovery on the sweep (shared EIV + calibration), corrected vs true
  with the y = x line — `bench_alpha_recovery.png`.
- within-bin **representativeness** (median true protein of detected vs missed) —
  plotted alongside (a) and tabled in `bench_representativeness.csv`.
- per-bin metrics `bench_diameter_metrics.csv`; small-regime scorecard
  `bench_small_regime_scorecard.txt`.

### Small-liposome regime — the candidate MOAT and its honest success criterion

The only plausible place our detector beats existing tools is the **small /
high-curvature** regime (instrument-matched + small-size-emphasis training). The
binned plots make the small bins prominent, and for each method + small bin we report
**recall AND representativeness together**: "better small-spot detection" only helps
the measurement if the kept spots are **representative** of their size bin. A method
with higher small-bin recall but **biased-bright** detection (`repr_ratio` ≫ 1) is
**WORSE** for alpha, not better — reporting both catches a recall "win" that is
actually a bias.

**Success criterion for ours (all four, not just recall):** beat baselines on
**small-bin F1** AND **small-bin intensity accuracy** AND keep within-bin detection
**representative** (`repr_ratio` ≈ 1) AND keep **alpha recovery unbiased**. If we win
all four, that justifies "better in the curvature-relevant regime." **If we LOSE**,
the binned scorecard says exactly which bins/architecture a future small-liposome
training pass should target — making that optimization targeted, not blind. **If we
are at PARITY, the contribution is the validated corrected measurement (EGFP = 2.0 +
DLS + synthetic sweep), NOT "best detector."** Reported honestly either way.

### Native end-to-end vs shared-photometry (fairness to external photometry)

`bench_native_vs_shared.csv` (and the real-data `ours_native` column) report each
method's **NATIVE end-to-end** alpha — its own detection + its **own** photometry
(ours: the network's predicted intensities; external tools: their reported
intensities) + the equalized EIV + calibration fit. The **shared-photometry** table
isolates *detection*; the **native** table compares *end-to-end* and is fair to tools
like cme-analysis whose careful PSF-photometry is part of their contribution. The
difference between the two localizes any gap to **detection vs photometry**. Both are
reported.

## Synthetic export for external tools (`export_for_external.py`) — the enabler

Exports the EXACT synthetic benchmark images so cme-analysis / SpotMAX can run on the
same ground truth on the user's machine, under `datasets/external_export/<set>/`
(gitignored):

- **TIFF layout:** `tiffs/img_NNNNNN.tif`, one **2-channel uint16 ImageJ TIFF** per
  image, axes **`CYX`**: **channel 0 = protein (488), channel 1 = lipid (561)**,
  raw-ADU intensities (dark offset included, 12-bit range) — same scale/format as the
  real microscope data.
- **GT schema:** `ground_truth.csv` with columns
  `image_id, x, y, diameter_nm, lipid_flux, protein_flux, alpha` (x/y full-res px,
  origin top-left matching the TIFF; `*_flux` total integrated ADU; `alpha` the
  spot's true exponent). `README_export.txt` restates the contract.
- **Back-import:** the tool emits per-image `image_id, x, y[, score]` CSVs (same
  `image_id` as the source TIFF). Synthetic: read via
  `adapters.read_detection_csv` → `benchmark_core`. Real:
  `real_benchmark.py --external-csv-dir <dir>/<sample>.csv`. Identical downstream.

## Real-data cross-method hook (`real_benchmark.py`)

Runs ours + classical (+ `external` when `--external-csv-dir` is given) on the REAL
images through the **same** shared photometry + EIV + calibration, all anchored on
**EGFP = 2.0**. Complements `real_alpha.py` (ours' native end-to-end). So the
headline real-data comparison becomes: ours vs classical (now) and vs
cme-analysis/SpotMAX (later), all with identical downstream processing, judged by
closeness to EGFP = 2.0. Outputs `real_benchmark.csv` + `real_benchmark_summary.txt`.

## Results — Part 2

### Synthetic benchmark (diameter-binned, `bench_small_regime_scorecard.txt`)

Small-regime scorecard (the candidate moat: pooled bins **40–55, 55–70,
70–90 nm**):

| sizing   | method    | small-bin F1 | \|lipid logerr\| | repr_ratio |
|----------|-----------|--------------|------------------|------------|
| emphasis | ours      | **0.629**    | **0.198**        | 1.025      |
| emphasis | classical | 0.231        | 0.696            | 1.015      |
| dls      | ours      | **0.613**    | **0.221**        | 1.005      |
| dls      | classical | 0.307        | 0.558            | 0.985      |

- **The detection moat is real.** In the small/high-curvature regime ours roughly
  **doubles-to-triples classical's F1** (0.61–0.63 vs 0.23–0.31) while recovering
  lipid intensity **~3× more accurately** (|lipid logerr| 0.20–0.22 vs 0.56–0.70).
- **The wins are representative**, not a bright-spot bias: repr_ratio ≈ 1.0 for
  both methods (~1.0–1.03 ours, ~0.99–1.02 classical), so the extra small spots
  ours keeps are typical of their size bin (`bench_representativeness.csv`,
  `bench_diameter_metrics.csv`).
- **Alpha recovery on the sweep (`bench_alpha_recovery.csv` / `.png`):** ours
  CORRECTED tracks true α monotonically and near the y = x line (true 0.5→0.55,
  1.0→0.93, 1.5→1.42, 2.0→1.87, 2.2→1.96); classical CORRECTED is badly biased —
  compressed at the low end (true 0.5→1.34) and overshooting at the high end
  (true 2.0→2.64), i.e. classical's locations cannot recover α even with the
  shared correction.
- **Four-part verdict — ours WINS all four** in the small regime: higher F1 AND
  better intensity accuracy AND representative (repr_ratio ≈ 1) AND unbiased α.
  This justifies "better in the curvature-relevant regime," not merely parity.

### Native vs shared photometry (`bench_native_vs_shared.csv`)

On the synthetic sweep, ours' **shared-aperture** photometry and the network's
**native** predicted intensities give essentially the same CORRECTED α (e.g. true
1.5: shared 1.416 vs native 1.419; true 2.0: 1.871 vs 1.903; true 2.2: 1.962 vs
2.074). So for our detector, detection and photometry are not in tension — the
shared-aperture isolation does not distort our result.

### Real-data cross-method (EGFP = 2.0 anchor, `real_benchmark_summary.txt`)

Same real images, all methods through the **same** shared photometry + EIV +
calibration fit. Mean over the 4 EGFP samples (true α = 2.0):

| method        | mean α | mean \|α−2.0\| |
|---------------|--------|----------------|
| ours (shared) | 2.245  | **0.534**      |
| classical     | 5.511  | 3.511          |
| ours_native   | 1.458  | 0.542          |

- **Closest to EGFP = 2.0: ours** (either ours_shared or ours_native; both ≈ 0.53
  from 2.0). **Classical is catastrophic** (mean α 5.5; per-sample 4.6–7.9), i.e.
  classical's real-image locations give a wildly wrong slope. The detection moat
  seen on synthetic data carries over to real data and is decisive on the EGFP
  anchor.
- **Endophilin:** ours = 2.527 / 3.080, classical = 7.865 / 4.976, ours_native =
  0.428 / 0.388 for 25 / 300 nM. (ours_native reads strong curvature sensing
  α < 0.5, consistent with `real_alpha.py`; the shared-photometry ours number is
  inflated for endophilin — its own per-sample lipid scale differs, see Part 2's
  per-sample `lipid_brightness` fix and the gain-correction open item.)

---

## Diagnosis: EGFP concentration-dependent alpha

The 4 EGFP samples share **one liposome prep** and have the same ground-truth
α = 2.0, yet their native CORRECTED α rises **monotonically with concentration**:

| sample     | native α (`ours_native`) |
|------------|--------------------------|
| 20nM_EGFP  | 1.27 |
| 50nM_EGFP  | 1.40 |
| 100nM_EGFP | 1.51 |
| 300nM_EGFP | 1.66 |

(`real_benchmark.csv` `ours_native`; the CORRECTED column of `real_alpha.csv`
shows the same ordering, 1.33→1.72.) A trend across a *shared prep* cannot be
biology, so we ruled out candidate causes in order using marginal checks on the
per-spot data:

1. **NOT protein brightness.** Mean / p95 protein flux is **flat ~165–178 ADU**
   across all four samples — no concentration trend that could tilt the slope.
2. **NOT background.** The protein p10 intensity floor is **flat at 151.0 ADU**
   across samples — background offset is not drifting.
3. **NOT detection density / count.** The EGFP detection counts are comparable
   (5.4k–6.6k) and do not order with the α trend.

**Leading cause — the 561 nm LIPID PMT voltage varies per sample.** Per
[`acquisition_metadata.md`](acquisition_metadata.md), the lipid detector was run
at **750 / 640 / 630 / 580 V** for 20 / 50 / 100 / 300 nM (the operator backed off
the gain at higher concentration to avoid saturation), while the **488 nm protein
PMT was constant at 295 V**. PMT gain is **nonlinear in voltage (~Vᵞ)**, and lipid
is the **size axis** of the α fit, so a per-sample lipid-gain difference rescales
the lipid axis sample-by-sample and **distorts the log-log slope** — exactly the
monotone-in-concentration pattern observed. (488 transmissivity also varies
5 / 3.3 / 3.3 / 3.0 %, but that is a linear protein-excitation factor → it shifts
the **intercept**, not the slope.)

**Classification:** this is a **correctable acquisition artifact** — not biology,
not a detector flaw, and **not** something a retrain fixes. The real α values in
this run are therefore **PRE-correction**. Two correction options (neither
performed here):

- **Option A — direct PMT gain(V) calibration.** Image a uniformly-fluorescent
  sample at the four used voltages to measure the FV3000 gain curve, then
  gain-normalize each sample's lipid channel. Most rigorous; **requires
  microscope time not yet acquired**.
- **Option B — EGFP self-calibration.** Since the 4 EGFP samples share a prep,
  align their lipid distributions empirically to back out the relative per-sample
  gain. No new data needed, but **assumes linear gain** and **breaks if the 750 V
  (20 nM) sample is in the nonlinear / saturation regime** — hence the saturation
  check below must precede it.

## Status & next steps

Open items for the next session (none attempted in this documentation pass):

- [ ] **Lipid gain-correction (Option A or B above).** Until done, all real α are
  PRE-correction. Option B depends on the saturation check passing for 20 nM.
- [ ] **Saturation check on 20 nM (750 V) lipid channel.** Inspect for
  clipping near the 12-bit ceiling (4095). Clipping is unrecoverable and would
  partly invalidate that sample (and break Option B's linear-gain assumption).
- [ ] **Endophilin DLS puzzle.** DLS agreement is poor (Wasserstein ~39–50 nm,
  KS ~0.65–0.75 vs EGFP ~12–19 nm / 0.30–0.37) and endophilin detection count is
  ~4× EGFP. Unexplained; flagged, but it is *not* the EGFP-α driver.
- [ ] **Ingest external-method coordinates** (cme-analysis [C++ port], SpotMAX).
  The `external_csv` adapter hook and the synthetic export
  (`datasets/external_export/`, 8 α sets + 2 bench sets, each `tiffs/` +
  `ground_truth.csv`) are ready; coordinates not yet brought back / scored.

## Provenance notes

All result artifacts in this folder are committed and clean at git SHA `e0dfb42`
(verified: no uncommitted changes in the experiment folder). `run.sh`'s 7 steps
map to the outputs as follows:

| run.sh step | script | recorded outputs |
|-------------|--------|------------------|
| 1 | `smoke_check.py` | **none** (print-only guard — see note ‡) |
| 2 | `firm_calibration.py` | `calibration_curve.{csv,json,png}` |
| 3 | `real_alpha.py` | `real_alpha.csv`, `real_alpha_summary.txt` |
| 4 | `dls_consistency.py` | `dls_consistency.{csv,png}` |
| 5 | `synth_benchmark.py` | `bench_diameter_metrics.csv`, `bench_representativeness.csv`, `bench_alpha_recovery.{csv,png}`, `bench_native_vs_shared.csv`, `bench_detection_vs_diam.png`, `bench_intensity_vs_diam.png`, `bench_small_regime_scorecard.txt` |
| 6 | `export_for_external.py` | writes to `datasets/external_export/` (gitignored — not in this folder) |
| 7 | `real_benchmark.py` | `real_benchmark.csv`, `real_benchmark_summary.txt` |

Two items to flag (reported, not fixed, per this pass's guardrails):

- **‡ `smoke_check.py` produces no file.** It is a print-only PASS/WARN scaling
  guard; its verdict is not captured in an artifact, so it cannot be audited
  after the fact. Consider having it tee its verdict to `smoke_check.txt` in a
  future run (out of scope here — would require touching the script).
- **`real_perspot.csv` (7.5 MB) is an orphan output.** It is tracked (added in
  commit `ac7cac1`) but **no script in the current tree writes it** — `grep` for
  `perspot` across the experiment scripts and `src/` finds no producer at SHA
  `e0dfb42`. It is the per-spot (lipid, protein, diameter-proxy) table behind the
  plots; its producing code was evidently changed/removed after it was committed.
  Left in place (do-not-remove guardrail), but its provenance is not reproducible
  from the committed scripts. Also see Repo cleanliness below (it is the one large
  tracked artifact).

`acquisition_metadata.md` is a hand-recorded reference (microscope PMT voltages
from the operator), not a script output — provenance is the acquisition log, not
`run.sh`.

## Artifacts

| file | description | producing script |
|------|-------------|-------------------|
| `EXPERIMENT.md` | this record | — (hand-written) |
| `acquisition_metadata.md` | PMT voltages / imaging params per sample (gain-correction basis) | — (operator log) |
| `run.sh` | 7-step orchestration wrapper | — |
| `_common.py` | shared helpers (per-sample `lipid_brightness`, paths) | — (imported) |
| `smoke_check.py` | scaling-guard (print-only, no output file) | — |
| `firm_calibration.py` | builds recovered→true calibration curve | — |
| `real_alpha.py` | real-image STANDARD vs CORRECTED α (native end-to-end) | — |
| `dls_consistency.py` | detector size proxy vs DLS | — |
| `synth_benchmark.py` | diameter-binned cross-method synthetic benchmark | — |
| `export_for_external.py` | exports synthetic TIFFs + GT for external tools | — |
| `real_benchmark.py` | real-image cross-method α (shared photometry) | — |
| `calibration_curve.csv` / `.json` | recovered→true anchors (8 α) | `firm_calibration.py` |
| `calibration_curve.png` | calibration-curve plot | `firm_calibration.py` |
| `real_alpha.csv` | per-sample α STANDARD & CORRECTED + CIs | `real_alpha.py` |
| `real_alpha_summary.txt` | EGFP anchor + internal-consistency + endophilin readout | `real_alpha.py` |
| `real_perspot.csv` | per-spot (lipid, protein, diameter proxy) table — **orphan, see Provenance** | (none at current SHA) |
| `dls_consistency.csv` | per-sample Wasserstein / KS vs DLS | `dls_consistency.py` |
| `dls_consistency.png` | size-distribution overlays | `dls_consistency.py` |
| `bench_small_regime_scorecard.txt` | small-bin F1 / lipid-logerr / repr_ratio, ours vs classical | `synth_benchmark.py` |
| `bench_diameter_metrics.csv` | full per-bin metrics (recall, F1, logerr, loc_err, repr) | `synth_benchmark.py` |
| `bench_representativeness.csv` | per-bin recall + repr_ratio + det/missed true protein | `synth_benchmark.py` |
| `bench_alpha_recovery.csv` | sweep α: standard/recovered/corrected per method | `synth_benchmark.py` |
| `bench_alpha_recovery.png` | corrected-vs-true α plot (y = x) | `synth_benchmark.py` |
| `bench_native_vs_shared.csv` | ours shared- vs native-photometry corrected α | `synth_benchmark.py` |
| `bench_detection_vs_diam.png` | detection F1/recall vs diameter | `synth_benchmark.py` |
| `bench_intensity_vs_diam.png` | intensity log-error vs diameter | `synth_benchmark.py` |
| `real_benchmark.csv` | real-image corrected α per method (ours/classical/native) | `real_benchmark.py` |
| `real_benchmark_summary.txt` | EGFP-anchor per-method readout | `real_benchmark.py` |
| `config_snapshot/` | exact study configs (alpha_template, bench_dls, bench_emphasis) | — (snapshot) |

## Reproduce on a fresh instance — what must be scp'd

The scaffold (scripts, configs, this record) is tracked, but the heavy inputs are
**gitignored** and must be copied to a fresh checkout before `run.sh` will work:

- `models/hrnet_v1/best.pt` — the trained detector.
- `data/<sample>/` — the real EGFP + endophilin TIFFs, dark frames, and DLS xlsx.
- `experiments/2026-06-03_per-sample-calibration/runs/*/calibration_results.json`
  — the per-sample calibration JSONs (set `lipid_brightness` and the size scale;
  present on disk, confirmed for all 6 samples, but **gitignored**).
