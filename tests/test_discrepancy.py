"""Tests for the config-driven, lipid-only discrepancy.

Verifies that:
  1. ``compute_discrepancy`` with the default config reproduces a value worked
     out by hand from deliberately clean inputs, for the five-term set
     (pixel_hist, psd, mean_pixel, quantiles, skewness).
  2. Disabling a term removes exactly that term's contribution.
  3. Overriding a term's weight scales exactly that term's contribution.
  4. Unknown / removed term names raise KeyError.
"""

import numpy as np
import pytest

from src.calibration.discrepancy import (
    DEFAULT_DISCREPANCY_CONFIG,
    compute_discrepancy,
)


def make_real():
    """Real-image stats with clean values chosen for hand computation."""
    return {
        'pixel_hist_bins': np.array([0.0, 1.0, 2.0]),   # values [0, 1]
        'pixel_hist': np.array([1.0, 1.0]),             # normalized -> p0 = 0.5
        'radial_psd': np.array([10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        'mean_pixel': 200.0,
        'p99': 600.0,
        'p999': 1000.0,
        'skewness': 4.0,
    }


def make_sim():
    """Simulated-image stats offset from the real ones by known amounts."""
    return {
        'pixel_hist_bins': np.array([0.0, 1.0, 2.0]),   # values [0, 1]
        'pixel_hist': np.array([3.0, 1.0]),             # normalized -> q0 = 0.75
        'radial_psd': np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        'mean_pixel': 180.0,
        'p99': 540.0,
        'p999': 900.0,
        'skewness': 3.6,
    }


# Per-term contributions for the clean inputs above (all at default weight):
#   pixel_hist : W1(mass 0.5@0/0.5@1, mass 0.75@0/0.25@1) = 0.25 ; w 0.01 -> 0.0025
#   psd        : mean((log10 100 - log10 10)^2 x6) = 1.0           ; w 1.0  -> 1.0
#   mean_pixel : (200-180)^2 / 200^2 = 0.01                        ; w 1.0  -> 0.01
#   quantiles  : (60/600)^2 + (100/1000)^2 = 0.01 + 0.01 = 0.02    ; w 1.0  -> 0.02
#   skewness   : (0.4/4.0)^2 = 0.01                                ; w 1.0  -> 0.01
#   total      = 0.0025 + 1.0 + 0.01 + 0.02 + 0.01 = 1.0425
EXPECTED_TOTAL = 1.0425


def test_default_config_has_expected_terms():
    assert set(DEFAULT_DISCREPANCY_CONFIG) == {
        'pixel_hist', 'psd', 'mean_pixel', 'quantiles', 'skewness'}


def test_value_computed_by_hand():
    real, sim = make_real(), make_sim()
    got = compute_discrepancy(real, sim)  # default config
    assert got == pytest.approx(EXPECTED_TOTAL, abs=1e-6)


def test_disabling_term_changes_result():
    real, sim = make_real(), make_sim()
    full = compute_discrepancy(real, sim)

    no_psd = compute_discrepancy(real, sim, {'psd': {'enabled': False}})
    assert no_psd != full
    assert no_psd == pytest.approx(full - 1.0, abs=1e-6)        # psd contributes 1.0

    no_quant = compute_discrepancy(real, sim, {'quantiles': {'enabled': False}})
    assert no_quant == pytest.approx(full - 0.02, abs=1e-6)     # quantiles contributes 0.02

    no_skew = compute_discrepancy(real, sim, {'skewness': {'enabled': False}})
    assert no_skew == pytest.approx(full - 0.01, abs=1e-6)      # skewness contributes 0.01


def test_weight_override_scales_term():
    real, sim = make_real(), make_sim()
    full = compute_discrepancy(real, sim)
    # Double quantiles weight (1.0 -> 2.0): adds another 1.0 * 0.02 = 0.02.
    heavier = compute_discrepancy(real, sim, {'quantiles': {'weight': 2.0}})
    assert heavier == pytest.approx(full + 0.02, abs=1e-6)


def test_unknown_term_raises():
    real, sim = make_real(), make_sim()
    with pytest.raises(KeyError):
        compute_discrepancy(real, sim, {'not_a_real_term': {'weight': 1.0}})


@pytest.mark.parametrize('removed_term', ['spot_intensity', 'spot_density', 'protein_nonpuncta'])
def test_removed_terms_raise(removed_term):
    """The old detection-based terms are gone and should be rejected."""
    real, sim = make_real(), make_sim()
    with pytest.raises(KeyError):
        compute_discrepancy(real, sim, {removed_term: {'enabled': False}})
