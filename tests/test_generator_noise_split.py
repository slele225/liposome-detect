"""gain/enf product x split: noise_scale must round-trip and the split must be
recoverable, per docs/decisions/2026-06-04_synthetic-generation-strategy.md."""

import numpy as np
import pytest

from src.generator.sampling import sample_image_params, sample_noise_split


def test_noise_split_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        noise_scale = float(rng.uniform(5.0, 80.0))
        gain, enf, r = sample_noise_split(rng, noise_scale, 0.2, 5.0)
        # product is the constrained quantity; split r is gain/enf.
        assert gain * enf == pytest.approx(noise_scale, rel=1e-9)
        assert gain / enf == pytest.approx(r, rel=1e-9)
        assert 0.2 <= r <= 5.0
        assert gain > 0 and enf > 0


def test_split_is_log_uniform_symmetric():
    # log-uniform [0.2, 5.0] is symmetric in log around 1 -> median r ~ 1.
    rng = np.random.default_rng(1)
    rs = np.array([sample_noise_split(rng, 30.0, 0.2, 5.0)[2] for _ in range(20000)])
    assert np.median(np.log(rs)) == pytest.approx(0.0, abs=0.05)


def test_sampled_params_recover_noise_scale(make_spec):
    spec = make_spec()
    rng = np.random.default_rng(7)
    lo, hi = spec['ranges']['widened']['noise_scale']
    for _ in range(100):
        params, meta = sample_image_params(
            rng, spec['ranges'], spec['regimes'][0], spec['cfg'])
        assert params['gain'] * params['enf'] == pytest.approx(meta['noise_scale'], rel=1e-9)
        assert params['gain'] / params['enf'] == pytest.approx(meta['noise_split_r'], rel=1e-9)
        assert lo <= meta['noise_scale'] <= hi


def test_psf_is_near_circular(make_spec):
    """PSF sampled as one width x small eccentricity, not independent x/y."""
    spec = make_spec()
    rng = np.random.default_rng(3)
    for _ in range(200):
        params, meta = sample_image_params(
            rng, spec['ranges'], spec['regimes'][0], spec['cfg'])
        ecc = params['psf_sigma_y'] / params['psf_sigma_x']
        assert 0.9 <= ecc <= 1.1
        assert params['psf_sigma_x'] == pytest.approx(meta['psf_sigma'])
        assert 0.0 <= params['psf_theta'] < 180.0
