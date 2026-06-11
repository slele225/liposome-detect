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

_(fill in after running on the instance)_

### Smoke check (scaling guard)

- Verdict: `…`
- Predicted lipid/protein flux medians: `…` (vs synthetic norm_mean lipid ~327,
  protein ~207; eps lipid 80, protein 62).

### Firmed-up calibration curve

- Recovered→true anchors (8 α): `…` (see `calibration_curve.png` / `.csv`).

### Real-data alpha — STANDARD vs CORRECTED (the go/no-go)

| sample            | n_spots | alpha_STANDARD(OLS) | alpha_CORRECTED(EIV+calib) | \|corr−2.0\| |
|-------------------|---------|---------------------|----------------------------|-------------|
| 20nM_EGFP         |         |                     |                            |             |
| 50nM_EGFP         |         |                     |                            |             |
| 100nM_EGFP        |         |                     |                            |             |
| 300nM_EGFP        |         |                     |                            |             |
| 25nM_endophilin   |         |                     |                            |             |
| 300nM_endophilin  |         |                     |                            |             |

- **EGFP anchor (true α = 2.0):** STANDARD mean |α−2.0| = `…`; CORRECTED mean
  |α−2.0| = `…`. Nearer 2.0 → `…`.
  - If STANDARD is biased low (<~1.9) and CORRECTED is nearer 2.0 → the bias is
    real on real data. If STANDARD is already ~2.0 → the correction is small in
    this regime (reported honestly).
- **Endophilin (α < 2):** STANDARD vs CORRECTED `…` — does it change the
  biological reading (sensing strength = distance below 2.0)? `…`

### DLS consistency (second anchor)

- Per-sample Wasserstein / KS (number-weighted): `…` (see `dls_consistency.png`).

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

## Results — Part 2 (fill in after running)

### Synthetic benchmark (diameter-binned)

- Global detection F1 (ours / classical), emphasis + dls: `…`
- Small-regime scorecard (bins 40–55, 55–70, 70–90): ours vs classical on F1 +
  |lipid logerr| + repr_ratio: `…` (`bench_small_regime_scorecard.txt`).
- Alpha recovery on the sweep (corrected vs true), per method: `…`
  (`bench_alpha_recovery.png`).
- Four-part verdict (does ours win small-regime F1 + intensity + representativeness +
  unbiased alpha, or is it parity?): `…`

### Native vs shared photometry

- ours_shared vs ours_native corrected alpha on the sweep: `…`
  (`bench_native_vs_shared.csv`).

### Real-data cross-method (EGFP = 2.0 anchor)

- Mean |corrected − 2.0| per method (ours / classical / ours_native [/ external]):
  `…` (`real_benchmark_summary.txt`). Closest to 2.0: `…`.
