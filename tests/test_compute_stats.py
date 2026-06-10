"""compute_stats: per-channel norm + intensity eps floors on a fixture dataset.

The eps floor must sit BELOW the dim-flux percentile (so it never swamps a real
dim spot) but stay strictly > 0, per channel. norm/eps must all be finite.
"""

import json
import math

import numpy as np
import pytest

from src.train.compute_stats import (
    compute_dataset_stats,
    compute_eps,
    compute_norm_stats,
)


def _write_dataset(root, n_images=5, hw=16, seed=0):
    """Write a tiny generator-style dataset: (2,H,W) images + label JSON each.

    Channel 0 (protein) and channel 1 (lipid) get DISTINCT scales so the per-channel
    stats are clearly separable; GT spot fluxes span a known range per channel.
    """
    rng = np.random.default_rng(seed)
    (root / 'images').mkdir(parents=True)
    (root / 'labels').mkdir(parents=True)
    # Fluxes drawn from disjoint ranges per channel so percentiles are predictable.
    lipid_fluxes = np.linspace(200.0, 5000.0, 40)
    protein_fluxes = np.linspace(80.0, 3000.0, 40)
    for i in range(n_images):
        protein = rng.normal(200.0, 10.0, size=(hw, hw))
        lipid = rng.normal(800.0, 50.0, size=(hw, hw))
        img = np.stack([protein, lipid], axis=0).astype(np.float32)
        np.save(root / 'images' / f'img_{i:06d}.npy', img)
        spots = [{
            'x': 1.0, 'y': 2.0, 'diameter_nm': 100.0,
            'lipid_intensity': float(lipid_fluxes[(i * 7 + j) % len(lipid_fluxes)]),
            'protein_intensity': float(protein_fluxes[(i * 5 + j) % len(protein_fluxes)]),
            'alpha_used': 1.0, 'sample_regime_id': 'A',
        } for j in range(8)]
        (root / 'labels' / f'img_{i:06d}.json').write_text(
            json.dumps({'index': i, 'n_spots': len(spots), 'spots': spots}))


def test_norm_stats_finite_and_per_channel(tmp_path):
    _write_dataset(tmp_path)
    paths = sorted((tmp_path / 'images').glob('img_*.npy'))
    mean, std = compute_norm_stats(paths)
    assert len(mean) == 2 and len(std) == 2
    assert all(math.isfinite(v) for v in mean + std)
    # protein channel (0) is dimmer than lipid (1) by construction.
    assert mean[0] < mean[1]
    assert all(s > 0 for s in std)


def test_eps_below_dim_flux_pct_and_positive(tmp_path):
    _write_dataset(tmp_path)
    labels = sorted((tmp_path / 'labels').glob('img_*.json'))
    eps = compute_eps(labels, floor_pct=1.0, floor_frac=0.1)
    for ch in ('lipid', 'protein'):
        e = eps[f'eps_{ch}']
        pct = eps[f'{ch}_floor_pct']
        assert math.isfinite(e) and e > 0.0           # strictly positive
        assert e < pct                                 # below the dim-flux percentile
    # floor_frac scales it: eps == floor_frac * percentile.
    assert eps['eps_lipid'] == pytest.approx(0.1 * eps['lipid_floor_pct'])
    assert eps['eps_protein'] == pytest.approx(0.1 * eps['protein_floor_pct'])


def test_compute_dataset_stats_end_to_end(tmp_path):
    _write_dataset(tmp_path, n_images=6)
    stats = compute_dataset_stats(tmp_path, floor_pct=1.0, floor_frac=0.1)
    assert stats['n_images'] == 6
    assert stats['n_spots'] == 6 * 8
    assert len(stats['norm_mean']) == 2 and len(stats['norm_std']) == 2
    assert all(math.isfinite(v) for v in stats['norm_mean'] + stats['norm_std'])
    assert stats['eps_lipid'] > 0 and stats['eps_protein'] > 0
    assert stats['eps_lipid'] < stats['lipid_floor_pct']
    assert stats['eps_protein'] < stats['protein_floor_pct']


def test_norm_stats_matches_numpy_reference(tmp_path):
    _write_dataset(tmp_path, n_images=4, hw=8)
    paths = sorted((tmp_path / 'images').glob('img_*.npy'))
    mean, std = compute_norm_stats(paths)
    stacked = np.stack([np.load(p) for p in paths])     # (N,2,H,W)
    ref_mean = stacked.mean(axis=(0, 2, 3))
    ref_std = stacked.std(axis=(0, 2, 3))
    assert mean == pytest.approx(ref_mean.tolist(), rel=1e-4)
    assert std == pytest.approx(ref_std.tolist(), rel=1e-4)
