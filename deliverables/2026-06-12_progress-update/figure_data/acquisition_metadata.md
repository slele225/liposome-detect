# Acquisition metadata — EGFP + endophilin real samples (Olympus FV3000)

## PMT detector voltages
| Sample           | 488 (protein, det#2) | 561 (lipid, det#3) | 488 transmissivity |
|------------------|----------------------|--------------------|--------------------|
| 20nM_EGFP        | 295 V                | 750 V              | 5.0 %              |
| 50nM_EGFP        | 295 V                | 640 V              | 3.3 %              |
| 100nM_EGFP       | 295 V                | 630 V              | 3.3 %              |
| 300nM_EGFP       | 295 V                | 580 V              | 3.0 %              |

## Key consequence (drives gain-correction)
- 488 protein PMT: CONSTANT 295 V across EGFP samples -> protein directly comparable.
- 561 lipid PMT: VARIES 750->580 V (operator backed off to avoid saturation at higher conc).
  PMT gain is nonlinear in V (~V^gamma). Since lipid is the size axis, this per-sample
  gain difference is the leading explanation for the EGFP concentration-dependent alpha
  trend (native alpha 1.27/1.40/1.51/1.66 for 20/50/100/300nM). FIX: gain-normalize lipid
  per sample (FV3000 gain curve, or empirically align identical-prep EGFP lipid distributions)
  before pooling/fitting. This is an acquisition artifact, NOT biology, NOT a detector flaw.
- 488 transmissivity also varies (5.0/3.3/3.3/3.0%) -> protein excitation power differs;
  linear per-sample factor (intercept, not slope) but normalize for clean cross-sample pooling.

## Imaging
60x/1.2 NA water, pixel 0.0691 um, pinhole 250 um, 12-bit, 512x512, 3-frame integration.
Channel map: det#2 = 488-excited = His-mEGFP (protein, ch0); det#3 = 561-excited = lipid dye (ch1).
(Presets mislabeled Alexa488/Alexa594; confirm ch0=protein before use — smoke-check was sane under this assumption.)
