"""The forward generative model: synthesize dual-channel microscopy images.

  - _apply_pmt_noise : PMT (ENF) + read noise + frame averaging, with a
                       photon-stage optical-background injection
  - _psf_inv_cov     : inverse covariance of a rotated 2D Gaussian PSF
  - _render_spot     : add one sum-normalized PSF spot to a clean image
  - simulate_image   : one (dual- or lipid-only) image + ground truth
  - simulate_batch   : batch wrapper around simulate_image

Background / noise model
------------------------
Spots are rendered as a per-channel rotated 2D Gaussian PSF (covariance matrix
built from sigma_x, sigma_y, theta) NORMALIZED TO SUM TO 1, so a spot's
amplitude is its TOTAL integrated flux in ADU (not a peak value).

  * LIPID channel:   clean signal = spot flux only. A flat optical background
    ``optical_bg_lipid`` (in PHOTONS) is injected at the photon stage inside
    ``_apply_pmt_noise`` (see the CRITICAL note there) and the dark-frame floor
    (offset_lipid + read_noise_var_lipid, measured/pinned) is the read-noise +
    DC offset added there as well.
  * PROTEIN channel: clean signal = spot flux only, with NO fitted optical
    background (fixed at 0); its baseline is purely the dark-frame floor.
    Protein is rendered only when ``lipid_only`` is False — the calibration
    path is lipid-only and never touches the protein channel.

This replaces the previous bg-patch + haze background model.
"""

import numpy as np


def _apply_pmt_noise(clean, gain, read_noise_var, offset, n_frames, enf, rng,
                     optical_bg_photons=0.0):
    """Apply PMT noise (Gaussian with ENF) + read noise + frame averaging.

    ``clean`` is the noise-free signal in ADU (gain * photons) from spots.

    ``optical_bg_photons`` is a flat optical background expressed in PHOTONS.

    CRITICAL — no double gain conversion: the optical background is in photons,
    so it is added directly to the photon count

        photon_count = clean / gain + optical_bg_photons       [photons]

    NOT to the ADU signal ``clean`` (which would divide it by gain). Its ADU
    mean contribution is therefore ``optical_bg_photons * gain`` and it carries
    ENF shot noise exactly like signal photons. (Equivalently one could add
    ``optical_bg_photons * gain`` to ``clean`` before the /gain step; that is
    the same value and is intentionally NOT what we do here, to keep the
    photon-stage injection explicit and impossible to double-convert.)

    The dark-frame floor (read_noise_var + offset) is the read noise and DC
    offset added AFTER amplification, as before.
    """
    clean = np.maximum(clean, 0)
    optical_bg_photons = max(float(optical_bg_photons), 0.0)
    if n_frames <= 1:
        if gain > 0:
            photon_count = clean / gain + optical_bg_photons
            mean = photon_count * gain
            variance = enf * photon_count * gain**2
            noisy = mean + rng.normal(0, 1, clean.shape) * np.sqrt(np.maximum(variance, 0))
        else:
            # Degenerate (gain<=0) path, not used in calibration; treat the
            # optical background as a flat additive signal.
            noisy = clean + optical_bg_photons
        noisy += rng.normal(0, np.sqrt(read_noise_var), clean.shape)
        noisy += offset
        return noisy
    accumulator = np.zeros_like(clean)
    for _ in range(n_frames):
        if gain > 0:
            photon_count = clean / gain + optical_bg_photons
            mean = photon_count * gain
            variance = enf * photon_count * gain**2
            frame = mean + rng.normal(0, 1, clean.shape) * np.sqrt(np.maximum(variance, 0))
        else:
            frame = clean + optical_bg_photons
        frame += rng.normal(0, np.sqrt(read_noise_var), clean.shape)
        accumulator += frame
    return accumulator / n_frames + offset


def _psf_inv_cov(sigma_x, sigma_y, theta_deg):
    """Inverse covariance matrix of a rotated 2D Gaussian PSF.

    Sigma = R(theta) @ diag(sigma_x^2, sigma_y^2) @ R(theta)^T, with R the 2D
    rotation by ``theta_deg`` degrees. Returns inv(Sigma) (a 2x2 array). The
    PSF value at offset (dx, dy) is then exp(-0.5 * [dx,dy] @ inv(Sigma) @ [dx,dy]^T).
    """
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([sigma_x**2, sigma_y**2])
    sigma = R @ D @ R.T
    return np.linalg.inv(sigma)


def _render_spot(img, cx, cy, amplitude, inv_cov, radius, image_size):
    """Add one spot of total flux ``amplitude`` to ``img`` in place.

    The PSF kernel is the rotated Gaussian defined by ``inv_cov``, normalized to
    sum to 1 over the (possibly edge-clipped) render window, so the spot's
    integrated flux equals ``amplitude`` in ADU.
    """
    x_lo = max(0, int(cx) - radius)
    x_hi = min(image_size, int(cx) + radius + 1)
    y_lo = max(0, int(cy) - radius)
    y_hi = min(image_size, int(cy) + radius + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dx = xx - cx
    dy = yy - cy
    # quadratic form [dx,dy] @ inv_cov @ [dx,dy]^T for the symmetric 2x2 inv_cov
    quad = (inv_cov[0, 0] * dx * dx
            + 2.0 * inv_cov[0, 1] * dx * dy
            + inv_cov[1, 1] * dy * dy)
    kernel = np.exp(-0.5 * quad)
    ksum = kernel.sum()
    if ksum > 0:
        kernel /= ksum
    img[y_lo:y_hi, x_lo:x_hi] += amplitude * kernel


def simulate_image(params, dls_diameters, dls_probs, image_size=256, rng=None,
                   lipid_only=False):
    """
    Generate one simulated microscopy image with ground truth.

    Args:
        params: dict with simulator parameters (see below).
        dls_diameters: array of diameters from DLS (nm).
        dls_probs: probability for each diameter.
        image_size: square image size in pixels (default 256, matching the
            center-crop applied to real images in ``io.load_tiff_stack``).
        rng: numpy random generator.
        lipid_only: if True, skip protein rendering entirely and return None for
            the protein channel — used by the lipid-only calibration path so no
            protein parameters are required.

    Parameters consumed (always):  spot_density, lipid_brightness, gain, enf,
        psf_sigma_x, psf_sigma_y, psf_theta, offset_lipid, read_noise_var_lipid,
        optical_bg_lipid, n_frame_avg.
    Parameters consumed (generation only, when lipid_only is False):
        protein_brightness, curvature_alpha, psf_sigma_x_protein,
        psf_sigma_y_protein, psf_theta_protein, offset_protein,
        read_noise_var_protein (+ optional independent_protein_* knobs).

    Returns:
        protein_img (or None if lipid_only), lipid_img, ground_truth (list of dicts).
    """
    if rng is None:
        rng = np.random.default_rng()

    spot_density = params['spot_density']            # mean spots per FOV
    lipid_brightness = params['lipid_brightness']    # TOTAL flux (ADU) of a d=100nm lipid spot
    gain = params['gain']                            # effective PMT gain
    enf = params.get('enf', 1.3)                     # excess noise factor
    n_frame_avg = params.get('n_frame_avg', 3)       # frame averaging (FV3000: 3)
    offset_lipid = params['offset_lipid']            # dark-frame DC offset (pinned)
    read_noise_var_lipid = params['read_noise_var_lipid']  # dark-frame read noise (pinned)
    optical_bg_lipid = params.get('optical_bg_lipid', 0.0)  # optical background in PHOTONS

    # Lipid PSF (fitted in calibration): rotated 2D Gaussian via covariance.
    psf_sigma_x = params['psf_sigma_x']
    psf_sigma_y = params['psf_sigma_y']
    psf_theta = params.get('psf_theta', 0.0)
    inv_lipid = _psf_inv_cov(psf_sigma_x, psf_sigma_y, psf_theta)
    r_lipid = int(np.ceil(max(psf_sigma_x, psf_sigma_y) * 4))

    # Sample number of spots and their diameters from the DLS distribution.
    n_spots = rng.poisson(spot_density)
    diameters = rng.choice(dls_diameters, size=n_spots, p=dls_probs)

    # Lipid spot amplitude (total flux) ~ surface area (d^2).
    lipid_amp = lipid_brightness * (diameters / 100.0)**2

    # Random positions. Keep the existing edge margin (10 px) for spot placement
    # on the (now 256x256) grid so spot centers stay off the very edge.
    positions_x = rng.uniform(10, image_size - 10, size=n_spots)
    positions_y = rng.uniform(10, image_size - 10, size=n_spots)

    clean_lipid = np.zeros((image_size, image_size), dtype=np.float64)

    if not lipid_only:
        # Protein channel (generation only). alpha controls curvature
        # preference; alpha=2 -> proportional to area, alpha<2 -> small
        # liposomes bind more protein per unit area. eta is per-spot
        # log-normal biological heterogeneity.
        curvature_alpha = params.get('curvature_alpha', 1.0)
        protein_brightness = params.get('protein_brightness', lipid_brightness)
        psf_sigma_x_p = params.get('psf_sigma_x_protein', psf_sigma_x)
        psf_sigma_y_p = params.get('psf_sigma_y_protein', psf_sigma_y)
        psf_theta_p = params.get('psf_theta_protein', 0.0)
        inv_protein = _psf_inv_cov(psf_sigma_x_p, psf_sigma_y_p, psf_theta_p)
        r_protein = int(np.ceil(max(psf_sigma_x_p, psf_sigma_y_p) * 4))
        eta = rng.lognormal(mean=0.0, sigma=0.1, size=n_spots)
        if params.get('independent_protein_intensity', False):
            # A_protein sampled independently of diameter (no power-law prior).
            ip_mu = params.get('independent_protein_mu', np.log(50.0))
            ip_sigma = params.get('independent_protein_sigma', 0.5)
            protein_amp = rng.lognormal(mean=ip_mu, sigma=ip_sigma, size=n_spots)
        else:
            protein_amp = (protein_brightness
                           * (diameters / 100.0)**curvature_alpha * eta)
        clean_protein = np.zeros((image_size, image_size), dtype=np.float64)
    else:
        clean_protein = None

    # Render spots: lipid with the lipid PSF, protein (if any) with the protein PSF.
    ground_truth = []
    for i in range(n_spots):
        cx, cy = positions_x[i], positions_y[i]
        _render_spot(clean_lipid, cx, cy, lipid_amp[i], inv_lipid, r_lipid, image_size)
        gt = {
            'x': float(cx), 'y': float(cy),
            'diameter_nm': float(diameters[i]),
            'lipid_intensity': float(lipid_amp[i]),
        }
        if not lipid_only:
            _render_spot(clean_protein, cx, cy, protein_amp[i], inv_protein,
                         r_protein, image_size)
            gt['protein_intensity'] = float(protein_amp[i])
        ground_truth.append(gt)

    # Lipid noise: optical_bg_lipid is injected in PHOTON units inside the PMT
    # noise step (no double gain conversion); the dark-frame floor is added there.
    noisy_lipid = _apply_pmt_noise(
        clean_lipid, gain, read_noise_var_lipid, offset_lipid,
        n_frame_avg, enf, rng, optical_bg_photons=optical_bg_lipid)
    noisy_lipid = np.clip(noisy_lipid, 0, 4095)  # 12-bit range

    if lipid_only:
        return None, noisy_lipid, ground_truth

    # Protein noise: NO fitted optical background (fixed at 0); baseline is the
    # dark-frame floor only.
    offset_protein = params['offset_protein']
    read_noise_var_protein = params['read_noise_var_protein']
    noisy_protein = _apply_pmt_noise(
        clean_protein, gain, read_noise_var_protein, offset_protein,
        n_frame_avg, enf, rng, optical_bg_photons=0.0)
    noisy_protein = np.clip(noisy_protein, 0, 4095)

    return noisy_protein, noisy_lipid, ground_truth


def simulate_batch(params, dls_diameters, dls_probs, n_images=50,
                   image_size=256, seed=None, lipid_only=False):
    """Generate a batch of simulated images.

    Returns (all_protein, all_lipid, all_gt). When ``lipid_only`` is True,
    ``all_protein`` is a list of ``None`` (the protein channel is not rendered).
    """
    rng = np.random.default_rng(seed)

    all_protein = []
    all_lipid = []
    all_gt = []

    for _ in range(n_images):
        protein, lipid, gt = simulate_image(
            params, dls_diameters, dls_probs,
            image_size=image_size, rng=rng, lipid_only=lipid_only,
        )
        all_protein.append(protein)
        all_lipid.append(lipid)
        all_gt.append(gt)

    return all_protein, all_lipid, all_gt
