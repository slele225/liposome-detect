"""Per-spot curvature-alpha protein channel, reusing the simulator's internals.

The simulator applies ONE ``curvature_alpha`` per ``simulate_image`` call (the
diameter exponent of the protein-flux formula). The ``per_spot_random`` alpha
mode needs an independent alpha per spot. We achieve this WITHOUT editing the
simulator (see docs/decisions/2026-06-04_synthetic-generation-strategy.md):

  1. render the LIPID channel + per-spot ground truth from one
     ``simulate_image(..., lipid_only=True)`` call (gives positions + diameters +
     lipid intensity), then
  2. synthesize the PROTEIN channel HERE, drawing each spot's own alpha and
     reusing the simulator's exact flux formula and its spot-renderer + PMT-noise
     functions (``_psf_inv_cov`` / ``_render_spot`` / ``_apply_pmt_noise``) so the
     noise model is byte-for-byte identical to the simulator's protein path.

Reusing those module-level helpers (rather than duplicating the noise code) is the
whole point: the per-spot path differs from the simulator ONLY in that alpha is
per-spot instead of per-image. No simulator change was required.
"""

import numpy as np

from src.simulator.forward_model import (
    _apply_pmt_noise,
    _psf_inv_cov,
    _render_spot,
)

# Per-spot lognormal heterogeneity, matching forward_model.simulate_image's
# ``eta = rng.lognormal(mean=0.0, sigma=0.1, ...)``.
ETA_SIGMA = 0.1


def render_protein_per_spot(ground_truth, params, alpha_range, rng, image_size=256):
    """Build the protein channel with an independent alpha per spot.

    Mutates each ``ground_truth`` dict in place, adding ``protein_intensity`` and
    ``alpha_used``. Returns ``(protein_img, alphas)`` where ``protein_img`` is the
    noisy, clipped (0..4095) protein image and ``alphas`` is the per-spot list.

    The flux per spot is ``protein_brightness * (d/100)**alpha_k * eta_k`` with
    ``eta_k ~ lognormal(0, ETA_SIGMA)`` and ``alpha_k ~ U(alpha_range)``, rendered
    with the measured protein PSF; the protein PMT noise uses the pinned protein
    dark floor and NO optical background (fixed 0), exactly as the simulator does.
    """
    a_lo, a_hi = alpha_range
    sx_p = float(params['psf_sigma_x_protein'])
    sy_p = float(params['psf_sigma_y_protein'])
    theta_p = float(params.get('psf_theta_protein', 0.0))
    inv_protein = _psf_inv_cov(sx_p, sy_p, theta_p)
    r_protein = int(np.ceil(max(sx_p, sy_p) * 4))
    protein_brightness = float(params['protein_brightness'])

    clean = np.zeros((image_size, image_size), dtype=np.float64)
    alphas = []
    for gt in ground_truth:
        d = float(gt['diameter_nm'])
        alpha_k = float(rng.uniform(a_lo, a_hi))
        eta_k = float(rng.lognormal(mean=0.0, sigma=ETA_SIGMA))
        amp = protein_brightness * (d / 100.0) ** alpha_k * eta_k
        _render_spot(clean, gt['x'], gt['y'], amp, inv_protein, r_protein, image_size)
        gt['protein_intensity'] = float(amp)
        gt['alpha_used'] = alpha_k
        alphas.append(alpha_k)

    noisy = _apply_pmt_noise(
        clean,
        float(params['gain']),
        float(params['read_noise_var_protein']),
        float(params['offset_protein']),
        int(params['n_frame_avg']),
        float(params['enf']),
        rng,
        optical_bg_photons=0.0,
    )
    noisy = np.clip(noisy, 0, 4095)
    return noisy, alphas
