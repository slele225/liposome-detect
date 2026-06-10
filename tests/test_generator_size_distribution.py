"""Spot-size distribution: the curvature-emphasis sampler must tilt small."""

import numpy as np
import pytest

from src.generator.size_distribution import curvature_emphasis_pmf


def test_pmf_normalized_and_strictly_decreasing_under_full_emphasis():
    d, p = curvature_emphasis_pmf(40.0, 300.0, 256, size_emphasis=1.0)
    assert len(d) == 256
    assert d[0] == pytest.approx(40.0) and d[-1] == pytest.approx(300.0)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)
    # emphasis=1 -> p ∝ 1/d^2, strictly decreasing in diameter.
    assert np.all(np.diff(p) < 0)


def test_emphasis_tilts_mass_to_small_diameters():
    d, p1 = curvature_emphasis_pmf(40.0, 300.0, 256, size_emphasis=1.0)
    _, p0 = curvature_emphasis_pmf(40.0, 300.0, 256, size_emphasis=0.0)
    mean1 = float((d * p1).sum())
    mean0 = float((d * p0).sum())
    # uniform-in-d mean ~ grid midpoint; curvature emphasis is well below it.
    assert mean0 == pytest.approx((40.0 + 300.0) / 2, rel=0.02)
    assert mean1 < mean0
    # majority of the probability mass below 100 nm under full emphasis.
    assert float(p1[d < 100].sum()) > 0.5


def test_emphasis_knob_is_monotone():
    means = []
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        d, p = curvature_emphasis_pmf(40.0, 300.0, 128, size_emphasis=e)
        means.append(float((d * p).sum()))
    assert all(means[i] > means[i + 1] for i in range(len(means) - 1))


def test_sampling_reproduces_weighted_histogram():
    """Drawing diameters with the PMF reproduces the small-tilted histogram."""
    d, p = curvature_emphasis_pmf(40.0, 300.0, 256, size_emphasis=1.0)
    rng = np.random.default_rng(0)
    draws = rng.choice(d, size=50000, p=p)
    # sampled mean tracks the analytic PMF mean and sits well below midpoint.
    assert draws.mean() == pytest.approx(float((d * p).sum()), rel=0.03)
    assert draws.mean() < (40.0 + 300.0) / 2
    assert np.mean(draws < 100) > 0.5


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        curvature_emphasis_pmf(0.0, 300.0, 64, 1.0)
    with pytest.raises(ValueError):
        curvature_emphasis_pmf(300.0, 40.0, 64, 1.0)
