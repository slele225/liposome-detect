"""A dummy backbone conforming to the interface plugs into heads + loss unchanged.

Proves the substitutability claim: ANY ``Backbone`` (here a hand-rolled one, not
even the provided DummyBackbone) composes with DetectorModel, and a full
loss+backward step runs — i.e. the heads, loss, and harness need no change to
swap the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.interface import Backbone, DetectorModel
from src.train.losses import compute_losses

WEIGHTS = {'w_hm': 1.0, 'w_off': 1.0, 'w_lip': 1.0, 'w_pro': 1.0}


class _CustomBackbone(Backbone):
    """A different conforming backbone (single stride-4 path)."""

    def __init__(self, in_chans=2, out_channels=8):
        super().__init__()
        self.out_channels = out_channels
        self.out_stride = 4
        self.conv = nn.Conv2d(in_chans, out_channels, 3, stride=4, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))


def _toy_targets(h, w):
    return {
        'heatmap': torch.zeros(1, 1, h, w),
        'pos_weight_map': torch.ones(1, 1, h, w),
        'bidx': torch.zeros(2, dtype=torch.long),
        'iy': torch.tensor([2, 5], dtype=torch.long),
        'ix': torch.tensor([3, 6], dtype=torch.long),
        'offset': torch.tensor([[0.3, 0.4], [0.1, 0.2]]),
        'lipid': torch.tensor([4000.0, 800.0]),
        'protein': torch.tensor([2500.0, 600.0]),
    }


def test_custom_backbone_plugs_into_detector_and_loss():
    model = DetectorModel(_CustomBackbone(in_chans=2, out_channels=8))
    x = torch.randn(1, 2, 64, 64)
    out = model(x)
    h, w = 16, 16
    assert out['heatmap'].shape == (1, 1, h, w)

    targets = _toy_targets(h, w)
    # mark the two centers on the heatmap target
    targets['heatmap'][0, 0, 2, 3] = 1.0
    targets['heatmap'][0, 0, 5, 6] = 1.0

    total, parts = compute_losses(out, targets, WEIGHTS, use_nll=True)
    assert torch.isfinite(total)
    total.backward()
    # gradients flow into the backbone (the substituted component)
    assert model.backbone.conv.weight.grad is not None
    assert set(parts) == {'heatmap', 'offset', 'lipid', 'protein'}
