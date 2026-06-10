# `src/models/` — two-channel detector

ONE architecture (HRNet) behind a thin, swappable backbone interface, plus the
CenterNet-style + heteroscedastic heads and the inference decode that emits the
benchmark output schema.

## Why (decision records)
- [Detector loss design](../../docs/decisions/2026-06-10_detector-loss-design.md) — what the heads/decode serve.
- [Benchmark design](../../docs/decisions/2026-06-10_benchmark-design.md) — the output schema is a hard contract; the multi-architecture bake-off is deferred.
- [Generation strategy](../../docs/decisions/2026-06-04_synthetic-generation-strategy.md) — the data + alpha-agnostic invariant.

## Pieces
- `interface.Backbone` — minimal contract: `[B,2,H,W] → [B,C,H/S,W/S]`, declaring
  `out_channels` / `out_stride`. `interface.DetectorModel` composes ANY conforming
  backbone with the four heads, so a new backbone drops in without touching
  heads/loss/decode/harness (proven by `dummy.DummyBackbone` in tests).
- `hrnet.HRNetBackbone` — timm HRNet, 2-channel stem (`in_chans=2`, no pretrained),
  stride-4 high-res branch (`out_index=1`) — chosen for the small-spot tail.
- `heads` — `HeatmapHead` (prob), `OffsetHead` (subpixel dx,dy), `IntensityHead`
  (×2: lipid, protein; channel 0 = mean flux >0 via exp, channel 1 = log-variance).
- `decode` — heatmap peak NMS → offset refine → sample the four intensity maps →
  per-spot dicts with `decode.SCHEMA_KEYS` (the contract). `write_detections` /
  `load_detections` round-trip a per-image list to JSON. The schema omits diameter
  by design (size derived downstream from lipid flux; alpha from log-protein vs
  log-lipid slope).

Numeric knobs live in `configs/train/<name>.yaml`. Backbone runs a CPU forward on
tiny inputs for tests.
