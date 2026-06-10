"""Two-channel liposome-puncta detector (Stage 2).

ONE architecture is built here — an HRNet backbone (`hrnet.HRNetBackbone`) behind a
minimal `Backbone` interface (`interface.Backbone`) so other backbones can drop in
later without touching the heads, loss, or training harness. `DetectorModel`
composes any conforming backbone with the four CenterNet-style + heteroscedastic
heads (heatmap, offset, lipid intensity, protein intensity).

Design / "why":
  - docs/decisions/2026-06-10_detector-loss-design.md   (the loss this serves)
  - docs/decisions/2026-06-10_benchmark-design.md        (the output-schema contract)
  - docs/decisions/2026-06-04_synthetic-generation-strategy.md (the data)

The decode + output schema (`decode.py`) is the hard benchmark contract every
method's adapter must emit. Exact numeric knobs live in configs/train/<name>.yaml.
"""

from src.models.interface import Backbone, DetectorModel
from src.models.heads import HeatmapHead, OffsetHead, IntensityHead

__all__ = [
    'Backbone', 'DetectorModel',
    'HeatmapHead', 'OffsetHead', 'IntensityHead',
]
