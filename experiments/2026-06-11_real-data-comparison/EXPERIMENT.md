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
