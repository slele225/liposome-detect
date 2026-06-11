"""The canonical alpha estimator (``src.eval.alpha_fit``).

Covers: (1) on both-axis-noisy synthetic data OLS is biased low while Deming/York
recover the true slope, and York reduces variance vs constant-lambda Deming when the
per-point noise is position-INDEPENDENT; (2) the CalibrationCurve round-trips its
seeded anchor points through ``invert``; (3) ``recover_alpha`` returns 2*slope and is
finite on a simple set.
"""

import numpy as np
import pytest

from src.eval.alpha_fit import (
    SEED_RECOVERED,
    SEED_TRUE,
    CalibrationCurve,
    apply_calibration,
    deming_slope,
    ols_slope,
    recover_alpha,
    york_slope,
)


# --------------------------------------------------------------------------- #
# Estimator bias / variance                                                    #
# --------------------------------------------------------------------------- #
def _synth_both_axis_noise(rng, true_b=0.75, n=4000):
    """True line with slope ``true_b`` + heteroscedastic noise on BOTH axes whose
    per-point scale is assigned INDEPENDENTLY of x (position-independent)."""
    xt = rng.uniform(0.0, 4.0, n)                    # true log-lipid
    yt = 1.0 + true_b * xt                           # true log-protein
    s = rng.uniform(0.05, 0.5, n)                    # per-point noise scale (indep of xt)
    var_x = s ** 2
    var_y = (1.3 * s) ** 2                           # y a touch noisier
    x = xt + rng.normal(0.0, np.sqrt(var_x))
    y = yt + rng.normal(0.0, np.sqrt(var_y))
    return x, y, var_x, var_y


def test_ols_biased_low_eiv_recovers_true_slope():
    rng = np.random.default_rng(0)
    true_b = 0.75
    x, y, var_x, var_y = _synth_both_axis_noise(rng, true_b=true_b)
    a_ols = ols_slope(x, y)
    a_dem = deming_slope(x, y, lam=var_y.mean() / var_x.mean())
    a_york = york_slope(x, y, var_x, var_y)
    # OLS attenuates (regression dilution): clearly below the truth.
    assert a_ols < true_b - 0.03
    # Errors-in-variables fits recover the true slope.
    assert a_dem == pytest.approx(true_b, abs=0.05)
    assert a_york == pytest.approx(true_b, abs=0.05)
    # ...and both are less biased than OLS.
    assert abs(a_dem - true_b) < abs(a_ols - true_b)


def test_york_reduces_variance_vs_deming_position_independent_noise():
    """With position-independent per-point noise, correct per-point weighting (York)
    should give lower bootstrap variance than constant-lambda Deming."""
    rng = np.random.default_rng(1)
    true_b = 0.75
    x, y, var_x, var_y = _synth_both_axis_noise(rng, true_b=true_b, n=3000)
    lam0 = var_y.mean() / var_x.mean()
    n = x.size
    dem, yk = [], []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        dem.append(deming_slope(x[idx], y[idx], lam0))
        yk.append(york_slope(x[idx], y[idx], var_x[idx], var_y[idx]))
    dem, yk = np.array(dem), np.array(yk)
    # Both unbiased around the truth.
    assert dem.mean() == pytest.approx(true_b, abs=0.05)
    assert yk.mean() == pytest.approx(true_b, abs=0.05)
    # York's per-point weighting reduces the spread.
    assert yk.std() < dem.std()


# --------------------------------------------------------------------------- #
# recover_alpha                                                                #
# --------------------------------------------------------------------------- #
def test_recover_alpha_is_twice_deming_and_finite():
    rng = np.random.default_rng(2)
    x, y, var_x, var_y = _synth_both_axis_noise(rng)
    a = recover_alpha(x, y, lam=1.0)
    assert np.isfinite(a)
    assert a == pytest.approx(2.0 * deming_slope(x, y, 1.0))


def test_recover_alpha_default_lam_is_tls():
    rng = np.random.default_rng(3)
    x, y, _, _ = _synth_both_axis_noise(rng)
    # No lam, no variances -> TLS (lam=1).
    assert recover_alpha(x, y) == pytest.approx(2.0 * deming_slope(x, y, 1.0))


def test_recover_alpha_lam_from_variances():
    rng = np.random.default_rng(4)
    x, y, var_x, var_y = _synth_both_axis_noise(rng)
    lam = var_y.mean() / var_x.mean()
    assert recover_alpha(x, y, var_x=var_x, var_y=var_y) == pytest.approx(
        2.0 * deming_slope(x, y, lam))


# --------------------------------------------------------------------------- #
# CalibrationCurve                                                             #
# --------------------------------------------------------------------------- #
def test_calibration_curve_round_trips_seed_points():
    curve = CalibrationCurve.default()                # interp -> exact at anchors
    for rec, tru in zip(SEED_RECOVERED, SEED_TRUE):
        assert curve.invert(rec) == pytest.approx(tru, abs=1e-9)
        assert apply_calibration(rec, curve) == pytest.approx(tru, abs=1e-9)


def test_calibration_curve_interp_midpoint_and_extrapolation():
    curve = CalibrationCurve.default()
    # Midpoint between two anchors interpolates monotonically between their trues.
    mid_rec = 0.5 * (SEED_RECOVERED[0] + SEED_RECOVERED[1])
    mid_true = curve.invert(mid_rec)
    assert SEED_TRUE[0] < mid_true < SEED_TRUE[1]
    # Beyond the top anchor it extrapolates (does not clamp).
    assert curve.invert(SEED_RECOVERED[-1] + 0.5) > SEED_TRUE[-1]


def test_calibration_curve_array_input():
    curve = CalibrationCurve.default()
    out = curve.invert(np.array(SEED_RECOVERED))
    assert np.allclose(out, SEED_TRUE, atol=1e-9)


def test_calibration_curve_save_load_round_trip(tmp_path):
    curve = CalibrationCurve.default()
    p = tmp_path / 'curve.json'
    curve.save(p)
    loaded = CalibrationCurve.load(p)
    assert np.allclose(loaded.recovered, curve.recovered)
    assert np.allclose(loaded.true, curve.true)
    assert loaded.invert(SEED_RECOVERED[1]) == pytest.approx(SEED_TRUE[1], abs=1e-9)


def test_calibration_curve_linear_fit_kind():
    # Linear fit on collinear anchors recovers the exact line.
    curve = CalibrationCurve([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], kind='linear')
    assert curve.invert(4.0) == pytest.approx(8.0, abs=1e-9)
