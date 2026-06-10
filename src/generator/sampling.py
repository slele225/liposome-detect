"""Per-image structured parameter sampling.

Implements the sampling strategy from
docs/decisions/2026-06-04_synthetic-generation-strategy.md exactly:

  - lipid_brightness, spot_density : independent, uniform over the widened union.
  - PSF near-circular              : one sigma + small eccentricity + free theta
                                     (NOT independent sigma_x/sigma_y).
  - gain/enf as PRODUCT x SPLIT    : noise_scale ~ U(range); split r log-uniform;
                                     gain=sqrt(ns*r), enf=sqrt(ns/r). The split
                                     should never affect any result.
  - optical_bg_lipid               : small, U(0, ~7) photons. Protein optical bg
                                     stays 0 (the simulator hardcodes it).
  - protein channel                : protein_brightness randomized; protein PSF +
                                     dark floor pinned from the chosen regime.

Returns a ``params`` dict whose keys match ``simulate_image`` EXACTLY, plus a
``meta`` dict recording the noise product/split and PSF shape for provenance.
"""

import numpy as np


def sample_noise_split(rng, noise_scale, r_lo=0.2, r_hi=5.0):
    """Split an effective ``noise_scale`` into a (gain, enf) pair.

    ``r`` is drawn log-uniform over [r_lo, r_hi]; ``gain=sqrt(noise_scale*r)`` and
    ``enf=sqrt(noise_scale/r)`` so ``gain*enf == noise_scale`` (the constrained
    product) and ``gain/enf == r`` (the degenerate split). Returns
    ``(gain, enf, r)``.
    """
    r = float(np.exp(rng.uniform(np.log(r_lo), np.log(r_hi))))
    gain = float(np.sqrt(noise_scale * r))
    enf = float(np.sqrt(noise_scale / r))
    return gain, enf, r


def sample_image_params(rng, ranges, regime, cfg):
    """Sample one image's simulator parameters + metadata.

    Args:
        rng: numpy Generator (the image's deterministic stream).
        ranges: output of ``calibration_io.build_param_ranges`` (uses 'widened').
        regime: one entry of ``calibration_io.build_regimes`` (pinned protein PSF
            + dark floors + n_frame_avg for this image's ``sample_regime_id``).
        cfg: flat sampling config (protein_brightness_range [required],
            noise_split_range, optical_bg_lipid_range, eccentricity_range,
            fixed_spot_density [optional override]).

    Returns ``(params, meta)``; ``params`` keys match ``simulate_image`` exactly.
    """
    w = ranges['widened']

    lipid_brightness = float(rng.uniform(*w['lipid_brightness']))

    fixed_density = cfg.get('fixed_spot_density')
    if fixed_density is not None:
        spot_density = float(fixed_density)
    else:
        spot_density = float(rng.uniform(*w['spot_density']))

    # Near-circular PSF: one width, small eccentricity, free rotation (degrees).
    sigma = float(rng.uniform(*w['sigma']))
    ecc_lo, ecc_hi = cfg.get('eccentricity_range', (0.9, 1.1))
    eccentricity = float(rng.uniform(ecc_lo, ecc_hi))
    psf_sigma_x = sigma
    psf_sigma_y = sigma * eccentricity
    psf_theta = float(rng.uniform(0.0, 180.0))

    # gain/enf as product x split.
    noise_scale = float(rng.uniform(*w['noise_scale']))
    r_lo, r_hi = cfg.get('noise_split_range', (0.2, 5.0))
    gain, enf, split_r = sample_noise_split(rng, noise_scale, r_lo, r_hi)

    obg_lo, obg_hi = cfg.get('optical_bg_lipid_range', (0.0, 7.0))
    optical_bg_lipid = float(rng.uniform(obg_lo, obg_hi))

    pb_range = cfg.get('protein_brightness_range')
    if pb_range is None:
        raise ValueError("cfg['protein_brightness_range'] is required")
    protein_brightness = float(rng.uniform(pb_range[0], pb_range[1]))

    params = {
        # --- lipid channel (sampled) ---
        'spot_density': spot_density,
        'lipid_brightness': lipid_brightness,
        'gain': gain,
        'enf': enf,
        'n_frame_avg': int(regime['n_frame_avg']),
        'offset_lipid': float(regime['offset_lipid']),
        'read_noise_var_lipid': float(regime['read_noise_var_lipid']),
        'optical_bg_lipid': optical_bg_lipid,
        'psf_sigma_x': psf_sigma_x,
        'psf_sigma_y': psf_sigma_y,
        'psf_theta': psf_theta,
        # --- protein channel (brightness sampled; PSF + floor pinned) ---
        'protein_brightness': protein_brightness,
        'psf_sigma_x_protein': float(regime['psf_sigma_x_protein']),
        'psf_sigma_y_protein': float(regime['psf_sigma_y_protein']),
        'psf_theta_protein': 0.0,
        'offset_protein': float(regime['offset_protein']),
        'read_noise_var_protein': float(regime['read_noise_var_protein']),
    }
    meta = {
        'noise_scale': noise_scale,           # gain * enf (constrained product)
        'noise_split_r': split_r,             # gain / enf (degenerate split)
        'psf_sigma': sigma,
        'psf_eccentricity': eccentricity,
        'sample_regime_id': regime['name'],
    }
    return params, meta
