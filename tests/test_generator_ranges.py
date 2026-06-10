"""Range-building: schema parse, union, and the ±30% widening math."""

import pytest

from src.generator.calibration_io import (
    build_param_ranges,
    build_regimes,
    load_calibration,
)


def test_load_calibration_uses_fitted_gain_and_measured_protein_psf(write_calibration):
    p = write_calibration('20nM_EGFP', lipid_brightness=4845.3, spot_density=596.3,
                          psf_sigma_x=1.926, psf_sigma_y=1.928, gain=21.7, enf=1.58,
                          sx_p=1.868, sy_p=1.913, offset_protein=152.04, rnv_protein=1.2)
    c = load_calibration(p)
    assert c['name'] == '20nM_EGFP'
    assert c['lipid_brightness'] == pytest.approx(4845.3)
    assert c['spot_density'] == pytest.approx(596.3)
    # FITTED gain, NOT measured_params.gain (270).
    assert c['gain'] == pytest.approx(21.7)
    assert c['enf'] == pytest.approx(1.58)
    # measured protein PSF carried through for the regime.
    assert c['psf_sigma_x_protein'] == pytest.approx(1.868)
    assert c['psf_sigma_y_protein'] == pytest.approx(1.913)
    assert c['offset_protein'] == pytest.approx(152.04)
    assert c['read_noise_var_protein'] == pytest.approx(1.2)


def _cal(lb, sd, sx, sy, g, e):
    return {'lipid_brightness': lb, 'spot_density': sd, 'psf_sigma_x': sx,
            'psf_sigma_y': sy, 'gain': g, 'enf': e}


def test_union_and_widen_math():
    cals = [_cal(4000, 500, 1.9, 2.0, 20, 1.5),
            _cal(16000, 560, 1.3, 3.0, 21, 2.0)]
    r = build_param_ranges(cals, widen_frac=0.3)
    raw, wid = r['raw'], r['widened']

    assert raw['lipid_brightness'] == (4000, 16000)
    assert wid['lipid_brightness'] == pytest.approx((4000 * 0.7, 16000 * 1.3))
    assert raw['spot_density'] == (500, 560)
    assert wid['spot_density'] == pytest.approx((500 * 0.7, 560 * 1.3))

    # noise_scale = gain*enf -> {30, 42}; union [30, 42].
    assert raw['noise_scale'] == (30, 42)
    assert wid['noise_scale'] == pytest.approx((30 * 0.7, 42 * 1.3))

    # sigma = union of the sigma_x [1.3,1.9] and sigma_y [2.0,3.0] ranges -> [1.3,3.0].
    assert raw['sigma'] == (1.3, 3.0)
    assert wid['sigma'] == pytest.approx((1.3 * 0.7, 3.0 * 1.3))


def test_single_calibration_widens_a_point():
    r = build_param_ranges([_cal(5000, 500, 1.9, 1.9, 20, 1.5)], widen_frac=0.3)
    # lo==hi==value -> [v*0.7, v*1.3]
    assert r['widened']['lipid_brightness'] == pytest.approx((3500, 6500))
    assert r['raw']['noise_scale'] == (30, 30)


def test_widen_frac_zero_is_identity():
    cals = [_cal(4000, 500, 1.9, 2.0, 20, 1.5), _cal(16000, 560, 1.3, 3.0, 21, 2.0)]
    r = build_param_ranges(cals, widen_frac=0.0)
    assert r['widened']['lipid_brightness'] == pytest.approx((4000, 16000))


def test_build_regimes_carries_pinned_fields(write_calibration):
    from src.generator.calibration_io import load_calibration
    cals = [load_calibration(write_calibration(
        'S', lipid_brightness=5000, spot_density=500, psf_sigma_x=1.9,
        psf_sigma_y=1.95, gain=21, enf=1.6, sx_p=1.87, sy_p=1.91,
        offset_lipid=149.2, rnv_lipid=11.5, offset_protein=152.0, rnv_protein=1.2))]
    reg = build_regimes(cals)[0]
    assert reg['name'] == 'S'
    assert reg['psf_sigma_x_protein'] == pytest.approx(1.87)
    assert reg['offset_lipid'] == pytest.approx(149.2)
    assert reg['read_noise_var_protein'] == pytest.approx(1.2)
    assert reg['n_frame_avg'] == 3
