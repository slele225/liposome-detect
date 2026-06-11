# hrnet_v1 — Stage-2 liposome puncta detector

This folder archives the **first full Stage-2 trained model** (`hrnet_v1`) for the
liposome-detect project: its weights' metadata, per-epoch training metrics, and
run provenance. It is meant to be self-explanatory to a human or LLM reading the
repo cold. The large weight blob (`best.pt`, 54 MB) is **not** in git — see
[Regenerating best.pt](#regenerating-bestpt) — but everything needed to
understand, use, and reproduce it is.

## What it is

A two-channel HRNet detector for liposome puncta in synthetic confocal
fluorescence images.

- **Backbone:** `timm hrnet_w18_small_v2`, output stride 4 (heatmap resolution =
  input / 4).
- **Heads:**
  - a **heatmap** head (puncta detection / centers),
  - an **offset** head (sub-pixel center refinement),
  - two **heteroscedastic intensity** heads — one for the **lipid** channel and
    one for the **protein** channel. Each intensity head emits both a *mean flux*
    and a *log-space variance* (an aleatoric, per-spot uncertainty).
- **Per-detection output schema:**
  `x, y, detection_score, lipid_intensity, lipid_intensity_logvar,
  protein_intensity, protein_intensity_logvar`.
  There is **no diameter output** — diameter is derived downstream from the lipid
  intensity, which serves as the size proxy (lipid signal ~ d²).

## When / how it was trained

- **Hardware / scale:** one full run on a single **H100**, **20,000 synthetic
  training images** ([`configs/generator/train_full.yaml`](../../configs/generator/train_full.yaml))
  plus **3,000 validation images**
  ([`configs/generator/val_full.yaml`](../../configs/generator/val_full.yaml)).
- **Wall-clock:** ~5.6 h, **input-bound** (data loading, not GPU compute, was the
  bottleneck).
- **Early stopping:** stopped at **epoch 24** on `val_intensity_logmse`. The
  **best checkpoint is epoch 14**, `val_intensity_logmse = 0.0835` — this is what
  `best.pt` contains.
- **Normalization:** `norm_mean` / `norm_std` and `eps` were **recomputed on the
  20k training set** (recorded in [`provenance.json`](provenance.json) and
  [`configs/train/hrnet_v1.yaml`](../../configs/train/hrnet_v1.yaml)).

Per-epoch metrics (train/val losses, detection P/R/F1, localization error,
intensity log-MSE, early-stopping state) are in [`metrics.jsonl`](metrics.jsonl),
one JSON object per epoch.

## Why it exists / what it's for

This detector is the **enabling instrument** for measuring the curvature-sensing
parameter **alpha**.

On a liposome of diameter `d`, lipid signal scales as `d²` and protein signal as
`d^alpha`, so

```
alpha = 2 × slope( log(protein_intensity) vs log(lipid_intensity) )
```

Critically, the model is trained **alpha-agnostically**: training images use
**per-spot-random alpha**, and alpha never appears in the loss. The detector
learns only to read out lipid and protein flux per punctum. The downstream alpha
is therefore a **measurement**, not a baked-in assumption.

## Validated behavior

- **Alpha recovery is monotonic and correctable.** Under the canonical
  constant-lambda **Deming** estimator
  ([`src/eval/alpha_fit.py`](../../src/eval/alpha_fit.py)), true alpha
  `0.5 / 1.0 / 1.5 / 2.0` recovers as `0.644 / 0.957 / 1.321 / 1.677`.
- **The uncertainty heads are calibrated and informative**, but they are for
  **QC / error propagation, NOT for regression weighting.** Per-spot variance is
  confounded with the size axis (`r ≈ 0.58`), so weighting the alpha regression by
  it would bias the slope.
- Full validation chain and figures:
  [`experiments/2026-06-10_diagnostic-run/EXPERIMENT.md`](../../experiments/2026-06-10_diagnostic-run/EXPERIMENT.md).

## How to use it

This checkpoint pairs with [`configs/train/hrnet_v1.yaml`](../../configs/train/hrnet_v1.yaml)
(it must be built from the same config it was trained with). `best.pt` is a
checkpoint dict whose `model` key holds the state dict:

```python
import torch, yaml
from src.train.train import build_model

cfg = yaml.safe_load(open("configs/train/hrnet_v1.yaml"))
model = build_model(cfg).eval()
state = torch.load("models/hrnet_v1/best.pt", map_location="cpu")
model.load_state_dict(state["model"])
```

The eval scripts wrap exactly this in `src/eval/matching.py::load_model`, and it
is consumed by [`src/eval/alpha_recovery.py`](../../src/eval/alpha_recovery.py)
and the other `src/eval/` scripts (`detection_bias.py`,
`recall_vs_diameter.py`, `uncertainty_calibration.py`, …).

## Regenerating best.pt

The weights are intentionally **not in git** (`*.pt` is gitignored repo-wide; see
this folder's `.gitignore`). To regenerate:

1. Regenerate the datasets from
   [`configs/generator/train_full.yaml`](../../configs/generator/train_full.yaml)
   and [`configs/generator/val_full.yaml`](../../configs/generator/val_full.yaml)
   (deterministic seeds → reproducible images).
2. Train [`configs/train/hrnet_v1.yaml`](../../configs/train/hrnet_v1.yaml) on
   them. Needs a GPU; ~hours.

[`metrics.jsonl`](metrics.jsonl) and [`provenance.json`](provenance.json) record
the exact run that produced this `best.pt`, so a regenerated checkpoint can be
checked against them.

## Provenance

See [`provenance.json`](provenance.json) for the precise seed
(`seed: 0`), config path, dataset path, code commit
(`code_commit: ab8df9c…`), and Python version of the run that produced `best.pt`.
The training config itself was introduced on the **`stage2-models`** branch (the
same commit that adds this archive).
