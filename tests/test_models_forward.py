"""Forward pass: 2-channel input -> all four head maps at the right shape."""

import torch

from src.models.dummy import DummyBackbone
from src.models.hrnet import HRNetBackbone
from src.models.interface import DetectorModel


@torch.no_grad()
def test_dummy_detector_forward_shapes():
    model = DetectorModel(DummyBackbone(in_chans=2, out_channels=16))
    x = torch.randn(2, 2, 64, 64)
    out = model(x)
    s = model.out_stride
    h, w = 64 // s, 64 // s
    assert out['heatmap'].shape == (2, 1, h, w)
    assert out['offset'].shape == (2, 2, h, w)
    assert out['lipid'].shape == (2, 2, h, w)
    assert out['protein'].shape == (2, 2, h, w)
    # heatmap is a probability; intensity channel 0 (mean flux) is strictly > 0.
    assert float(out['heatmap'].min()) > 0.0 and float(out['heatmap'].max()) < 1.0
    assert float(out['lipid'][:, 0].min()) > 0.0
    assert float(out['protein'][:, 0].min()) > 0.0


@torch.no_grad()
def test_hrnet_detector_forward_cpu_tiny():
    model = DetectorModel(HRNetBackbone(variant='hrnet_w18_small_v2',
                                        out_index=1, in_chans=2, pretrained=False))
    assert model.out_stride == 4
    x = torch.randn(1, 2, 64, 64)
    out = model(x)
    h, w = 64 // 4, 64 // 4
    for k, c in [('heatmap', 1), ('offset', 2), ('lipid', 2), ('protein', 2)]:
        assert out[k].shape == (1, c, h, w)
