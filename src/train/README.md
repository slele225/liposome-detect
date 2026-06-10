# `src/train/` — detector training harness

Config-driven training of the `src/models/` detector on a Prompt-1 generator
dataset. Single H100; no hyperparameter sweep.

## Why (decision records)
- [Detector loss design](../../docs/decisions/2026-06-10_detector-loss-design.md) — THE loss spec, implemented in `losses.py`.
- [Generation strategy](../../docs/decisions/2026-06-04_synthetic-generation-strategy.md) — the data + alpha-agnostic invariant.
- [Benchmark design](../../docs/decisions/2026-06-10_benchmark-design.md) — eval metrics / output schema.

## Loss (`losses.py`, faithful to the loss doc)
`total = w_hm·heatmap + w_off·offset + w_lip·intensity(lipid) + w_pro·intensity(protein)`
- heatmap: penalty-reduced focal + a BOUNDED per-spot size weight on the POSITIVE
  term only (`clip(d_ref/d, 1, w_max)`). **Invariant:** the size weight is on the
  heatmap (location) ONLY — never on intensity (that would be a backdoor curvature
  prior). `intensity_nll_loss` takes no diameter, by construction.
- offset: L1 on subpixel (dx,dy) at GT centers (added for CenterNet localization;
  see the note in `losses.py`).
- intensity (per channel): log-space heteroscedastic NLL, with a per-channel `eps`
  floor; plain log-space MSE during the NLL loss-warmup.

## Two separate warmups (`engine.py`)
- LR schedule: linear warmup → cosine to ~0 (AdamW).
- Loss warmup: log-MSE for the first `nll_warmup_epochs`, then full NLL.

## CLI
```bash
python -m src.train.train --config configs/train/hrnet_v1.yaml --n-workers 8
python -m src.train.train --config configs/train/smoke.yaml --smoke
```
`--smoke` runs a tiny dataset for a couple epochs and writes `smoke_detections.json`
(schema-validated) — verify before a real run. Outputs (checkpoints, `metrics.jsonl`,
`provenance.json`) go to `runs/<name>/` (gitignored). Per-epoch val: matched-F1
(fixed radius) + per-channel intensity log-error (`metrics.py`).

## Pieces
`targets` (GT rasterization) · `dataset` (loader + collate) · `losses` · `metrics`
· `engine` (loops + schedules) · `train` (CLI/checkpoint/provenance).
