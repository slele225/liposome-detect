"""Shared fixtures for the generator tests (hermetic — no real data/calibrations).

``experiments/*/runs/`` is gitignored, so tests must NOT depend on real
calibration outputs. These fixtures write synthetic ``calibration_results.json``
files (matching the real schema) into ``tmp_path`` and build generator specs from
them, so the suite runs on a fresh checkout.
"""

import json

import pytest


def _fake_calibration_dict(name, *, lipid_brightness, spot_density,
                           psf_sigma_x, psf_sigma_y, gain, enf,
                           optical_bg_lipid=1.0, sx_p=1.87, sy_p=1.91,
                           offset_lipid=150.0, rnv_lipid=11.0,
                           offset_protein=152.0, rnv_protein=1.2, n_frame_avg=3):
    """A per-sample calibration_results.json dict in the real schema."""
    return {
        'best_params': {
            'lipid_brightness': lipid_brightness,
            'psf_sigma_x': psf_sigma_x, 'psf_sigma_y': psf_sigma_y,
            'psf_theta': 44.0, 'gain': gain, 'enf': enf,
            'optical_bg_lipid': optical_bg_lipid, 'n_frame_avg': n_frame_avg,
        },
        'per_sample_params': {name: {
            'spot_density': spot_density, 'offset_lipid': offset_lipid,
            'offset_protein': offset_protein, 'read_noise_var_lipid': rnv_lipid,
            'read_noise_var_protein': rnv_protein,
        }},
        # measured_params.gain (270) is intentionally NOT the fitted gain.
        'measured_params': {
            'psf_sigma_x_protein': sx_p, 'psf_sigma_y_protein': sy_p, 'gain': 270.0,
        },
    }


@pytest.fixture
def write_calibration(tmp_path):
    """Factory: write a synthetic calibration JSON, return its path string."""
    def _write(name, **kw):
        p = tmp_path / f'cal_{name}.json'
        p.write_text(json.dumps(_fake_calibration_dict(name, **kw)))
        return str(p)
    return _write


@pytest.fixture
def make_spec(write_calibration):
    """Factory: build a generator spec from two synthetic calibrations.

    ``config_overrides`` shallow-updates the base config (pass a full ``sampling``
    block to override it). Defaults: two regimes, emphasis sizing, per-spot alpha,
    a fixed spot density (so images have plenty of spots for statistics).
    """
    from src.generator.generate import _build_spec

    def _make(config_overrides=None):
        cals = [
            write_calibration('A', lipid_brightness=5000.0, spot_density=500.0,
                              psf_sigma_x=1.90, psf_sigma_y=1.95, gain=21.0, enf=1.6),
            write_calibration('B', lipid_brightness=15000.0, spot_density=560.0,
                              psf_sigma_x=1.88, psf_sigma_y=2.28, gain=20.0, enf=2.3),
        ]
        config = {
            'name': 'test', 'n_images': 4, 'image_size': 256, 'base_seed': 0,
            'calibrations': cals,
            'alpha_mode': 'per_spot_random', 'alpha_range': [0.5, 2.0],
            'size_mode': 'emphasis',
            'size': {'d_min': 40.0, 'd_max': 300.0, 'n_bins': 128, 'size_emphasis': 1.0},
            'sampling': {'protein_brightness_range': [2500.0, 7500.0],
                         'fixed_spot_density': 500},
        }
        if config_overrides:
            config.update(config_overrides)
        return _build_spec(config, '<test>')

    return _make
