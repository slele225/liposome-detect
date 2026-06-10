"""Training harness for the two-channel detector (Stage 2).

Config-driven (`train.py`) over a Prompt-1 generator dataset: rasterizes GT
targets (`targets.py`), assembles the loss-design loss (`losses.py`), trains with
AdamW + linear-warmup/cosine LR and a separate NLL loss-warmup (`engine.py`), and
evaluates matched-F1 + intensity log-error (`metrics.py`).

See src/train/README.md and docs/decisions/2026-06-10_detector-loss-design.md.
"""
