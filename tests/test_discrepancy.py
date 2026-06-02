"""Tests for the config-driven discrepancy refactor.

Verifies that:
  1. ``compute_discrepancy`` with the default config reproduces the archive's
     old hardcoded formula (pixel_hist/100 + spot_intensity/200 + psd +
     spot_density + mean_pixel + protein_nonpuncta/200), both against an
     explicit re-derivation of that formula and against a value worked out by
     hand from deliberately clean inputs.
  2. Disabling a term removes exactly that term's contribution.
  3. Overriding a term's weight scales exactly that term's contribution.
"""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from src.calibration.discrepancy import compute_discrepancy


def make_real():
    """Real-image stats with clean values chosen for hand computation."""
    return {
        'pixel_hist_bins': np.array([0.0, 1.0, 2.0]),   # values [0, 1]
        'pixel_hist': np.array([1.0, 1.0]),             # normalized -> p0 = 0.5
        'spot_intensities': np.full(12, 100.0),         # point mass at 100
        'radial_psd': np.array([10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        'mean_spot_count': 100.0,
        'mean_pixel': 200.0,
        'protein_nonpuncta': np.full(12, 50.0),         # point mass at 50
    }


def make_sim():
    """Simulated-image stats offset from the real ones by known amounts."""
    return {
        'pixel_hist_bins': np.array([0.0, 1.0, 2.0]),   # values [0, 1]
        'pixel_hist': np.array([3.0, 1.0]),             # normalized -> q0 = 0.75
        'spot_intensities': np.full(12, 110.0),         # point mass at 110
        'radial_psd': np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        'mean_spot_count': 90.0,
        'mean_pixel': 180.0,
        'protein_nonpuncta': np.full(12, 70.0),         # point mass at 70
    }


def old_hardcoded_formula(real_stats, sim_stats):
    """Verbatim re-derivation of the archive's original compute_discrepancy
    (the six terms with their original hardcoded divisors)."""
    loss = 0.0

    w_pixel = wasserstein_distance(
        real_stats['pixel_hist_bins'][:-1], sim_stats['pixel_hist_bins'][:-1],
        u_weights=real_stats['pixel_hist'], v_weights=sim_stats['pixel_hist'])
    loss += w_pixel / 100.0

    if len(real_stats['spot_intensities']) > 10 and len(sim_stats['spot_intensities']) > 10:
        w_spot = wasserstein_distance(real_stats['spot_intensities'], sim_stats['spot_intensities'])
        loss += w_spot / 200.0

    min_len = min(len(real_stats['radial_psd']), len(sim_stats['radial_psd']))
    if min_len > 5:
        real_psd = np.log10(real_stats['radial_psd'][:min_len] + 1e-10)
        sim_psd = np.log10(sim_stats['radial_psd'][:min_len] + 1e-10)
        psd_mse = np.mean((real_psd[1:] - sim_psd[1:])**2)
        loss += psd_mse

    density_err = (real_stats['mean_spot_count'] - sim_stats['mean_spot_count'])**2
    loss += density_err / max(real_stats['mean_spot_count']**2, 1)

    mean_err = (real_stats['mean_pixel'] - sim_stats['mean_pixel'])**2
    loss += mean_err / max(real_stats['mean_pixel']**2, 1)

    real_np = real_stats.get('protein_nonpuncta')
    sim_np = sim_stats.get('protein_nonpuncta')
    if (real_np is not None and sim_np is not None
            and len(real_np) > 10 and len(sim_np) > 10):
        w_protein_np = wasserstein_distance(real_np, sim_np)
        loss += w_protein_np / 200.0

    return loss


def test_default_reproduces_hardcoded_formula():
    real, sim = make_real(), make_sim()
    expected = old_hardcoded_formula(real, sim)
    got = compute_discrepancy(real, sim)  # default config
    assert got == pytest.approx(expected)


def test_value_computed_by_hand():
    real, sim = make_real(), make_sim()
    # Term-by-term, by hand:
    #   pixel_hist : W1 between mass(0.5@0, 0.5@1) and mass(0.75@0, 0.25@1)
    #                = |0.5 - 0.75| * 1 = 0.25  ;  weight 0.01 -> 0.0025
    #   spot_int   : W1(point 100, point 110)   = 10    ;  weight 0.005 -> 0.05
    #   psd        : mean((log10 100 - log10 10)^2 x6) = 1.0 ; weight 1.0 -> 1.0
    #   spot_dens  : (100-90)^2 / 100^2 = 0.01   ;  weight 1.0 -> 0.01
    #   mean_pixel : (200-180)^2 / 200^2 = 0.01  ;  weight 1.0 -> 0.01
    #   protein_np : W1(point 50, point 70) = 20 ;  weight 0.005 -> 0.1
    #   total = 0.0025 + 0.05 + 1.0 + 0.01 + 0.01 + 0.1 = 1.1725
    got = compute_discrepancy(real, sim)
    assert got == pytest.approx(1.1725, abs=1e-6)


def test_disabling_term_changes_result():
    real, sim = make_real(), make_sim()
    full = compute_discrepancy(real, sim)

    no_psd = compute_discrepancy(real, sim, {'psd': {'enabled': False}})
    assert no_psd != full
    assert no_psd == pytest.approx(full - 1.0, abs=1e-6)   # psd contributes 1.0

    no_pixel = compute_discrepancy(real, sim, {'pixel_hist': {'enabled': False}})
    assert no_pixel == pytest.approx(full - 0.0025, abs=1e-6)  # pixel_hist contributes 0.0025


def test_weight_override_scales_term():
    real, sim = make_real(), make_sim()
    full = compute_discrepancy(real, sim)
    # Double protein_nonpuncta weight (0.005 -> 0.01): adds another 0.005 * 20 = 0.1
    heavier = compute_discrepancy(real, sim, {'protein_nonpuncta': {'weight': 0.01}})
    assert heavier == pytest.approx(full + 0.1, abs=1e-6)


def test_unknown_term_raises():
    real, sim = make_real(), make_sim()
    with pytest.raises(KeyError):
        compute_discrepancy(real, sim, {'not_a_real_term': {'weight': 1.0}})
