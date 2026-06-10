"""Heatmap size-weight: bounded, heatmap-only, and ABSENT from intensity.

Guards the alpha-agnostic invariant (docs/decisions/2026-06-10_detector-loss-
design.md): the per-spot size weight up-weights small-spot DETECTABILITY on the
heatmap, and must NEVER appear on the intensity losses (a diameter weight on
intensity is a backdoor curvature prior).
"""

import numpy as np
import torch

from src.models.dummy import DummyBackbone
from src.models.interface import DetectorModel
from src.train.losses import compute_losses
from src.train.targets import build_targets, compute_size_weight

WEIGHTS = {'w_hm': 1.0, 'w_off': 1.0, 'w_lip': 1.0, 'w_pro': 1.0}


def test_size_weight_is_bounded():
    diam = np.array([10.0, 50.0, 100.0, 250.0, 1000.0])
    w = compute_size_weight(diam, d_ref=100.0, w_max=5.0)
    assert w.min() >= 1.0
    assert w.max() <= 5.0
    assert w[0] == 5.0          # 100/10 = 10 -> clamped to w_max
    assert w[1] == 2.0          # 100/50 = 2 (interior)
    assert w[-1] == 1.0         # 100/1000 = 0.1 -> clamped to 1.0


def _collate_one(t):
    """Wrap one build_targets() output into the collated targets dict (B=1)."""
    n = t['iy'].shape[0]
    return {
        'heatmap': torch.from_numpy(t['heatmap'])[None],
        'pos_weight_map': torch.from_numpy(t['pos_weight_map'])[None],
        'bidx': torch.zeros(n, dtype=torch.long),
        'iy': torch.from_numpy(t['iy']),
        'ix': torch.from_numpy(t['ix']),
        'offset': torch.from_numpy(t['offset']),
        'lipid': torch.from_numpy(t['lipid']),
        'protein': torch.from_numpy(t['protein']),
    }


def test_intensity_loss_invariant_to_diameter_heatmap_is_not():
    # Two GT sets identical in EVERYTHING except diameter; fluxes held fixed.
    base = dict(x=120.0, y=80.0, lipid_intensity=4000.0, protein_intensity=2500.0)
    spots_small = [dict(base, diameter_nm=50.0)]    # s_weight = 2.0
    spots_large = [dict(base, diameter_nm=250.0)]   # s_weight = 1.0

    t_small = _collate_one(build_targets(spots_small, (64, 64), 4,
                                         heatmap_sigma=1.0, d_ref=100.0, w_max=5.0))
    t_large = _collate_one(build_targets(spots_large, (64, 64), 4,
                                         heatmap_sigma=1.0, d_ref=100.0, w_max=5.0))

    # the size weight shows up only in the heatmap's positive-weight map
    assert float(t_small['pos_weight_map'].max()) == 2.0
    assert float(t_large['pos_weight_map'].max()) == 1.0

    torch.manual_seed(0)
    model = DetectorModel(DummyBackbone(in_chans=2, out_channels=16))
    out = model(torch.randn(1, 2, 64, 64))

    _, p_small = compute_losses(out, t_small, WEIGHTS, use_nll=True)
    _, p_large = compute_losses(out, t_large, WEIGHTS, use_nll=True)

    # intensity terms are IDENTICAL (diameter never enters them) ...
    assert torch.allclose(p_small['lipid'], p_large['lipid'])
    assert torch.allclose(p_small['protein'], p_large['protein'])
    assert torch.allclose(p_small['offset'], p_large['offset'])
    # ... while the heatmap term DOES change with the size weight.
    assert not torch.allclose(p_small['heatmap'], p_large['heatmap'])
