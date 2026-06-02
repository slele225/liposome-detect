"""The forward generative model: synthesize dual-channel microscopy images.

  - _apply_pmt_noise                  : PMT (ENF) + read noise + frame averaging
  - simulate_image                    : one dual-channel image + ground truth
  - simulate_batch                    : batch with a single background pool
  - simulate_batch_dual_bg            : batch with separate lipid/protein bg
  - simulate_protein_calibration_batch: no-spot protein channel for calibration
  - extract_protein_nonpuncta_pixels  : protein pixels away from lipid puncta
  - _gather_nonpuncta_protein         : collect non-puncta protein pixels

Ported verbatim from the archive's pipeline.py (Module 8). The non-puncta
helpers live here (rather than in calibration) because they are part of the
simulator's pixel model and operate on both real and simulated arrays.
"""

import numpy as np
from scipy.ndimage import maximum_filter


def _apply_pmt_noise(clean, gain, read_noise_var, offset, n_frames, enf, rng):
    """Apply PMT noise (Gaussian with ENF) + read noise + frame averaging."""
    clean = np.maximum(clean, 0)
    if n_frames <= 1:
        if gain > 0:
            photon_count = clean / gain
            mean = photon_count * gain
            variance = enf * photon_count * gain**2
            noisy = mean + rng.normal(0, 1, clean.shape) * np.sqrt(np.maximum(variance, 0))
        else:
            noisy = clean.copy()
        noisy += rng.normal(0, np.sqrt(read_noise_var), clean.shape)
        noisy += offset
        return noisy
    accumulator = np.zeros_like(clean)
    for _ in range(n_frames):
        if gain > 0:
            photon_count = clean / gain
            mean = photon_count * gain
            variance = enf * photon_count * gain**2
            frame = mean + rng.normal(0, 1, clean.shape) * np.sqrt(np.maximum(variance, 0))
        else:
            frame = clean.copy()
        frame += rng.normal(0, np.sqrt(read_noise_var), clean.shape)
        accumulator += frame
    return accumulator / n_frames + offset


def simulate_image(params, dls_diameters, dls_probs, bg_patches,
                   bg_patches_protein=None, image_size=512, rng=None,
                   brightness_scale=1.0):
    """
    Generate one simulated dual-channel microscopy image with ground truth.

    Args:
        params: dict with simulator parameters
        dls_diameters: array of diameters from DLS (nm)
        dls_probs: probability for each diameter
        bg_patches: list of real background patches to sample from
        image_size: image dimensions (square)
        rng: numpy random generator

    Returns:
        protein_img: simulated protein channel (512x512)
        lipid_img: simulated lipid channel (512x512)
        ground_truth: list of dicts with spot properties
    """
    if rng is None:
        rng = np.random.default_rng()

    # Unpack parameters
    spot_density = params['spot_density']        # mean spots per FOV
    labeling_eff = params['labeling_eff']         # PEAK intensity scale (counts above bg for d=100nm), shared between channels
    curvature_alpha = params['curvature_alpha']   # curvature exponent
    psf_sigma_x = params['psf_sigma_x']           # PSF width x (pixels)
    psf_sigma_y = params['psf_sigma_y']           # PSF width y (pixels)
    gain = params['gain']                         # effective PMT gain
    offset_protein = params['offset_protein']     # baseline offset protein channel
    offset_lipid = params['offset_lipid']         # baseline offset lipid channel
    read_noise_var_protein = params['read_noise_var_protein']
    read_noise_var_lipid = params['read_noise_var_lipid']
    bg_amplitude = params['bg_amplitude']         # background scaling factor
    haze_level = params.get('haze_level', 0)      # out-of-focus haze
    n_frame_avg = params.get('n_frame_avg', 3)    # frame averaging (3 per FV3000 metadata)

    # Sample number of spots
    n_spots = rng.poisson(spot_density)

    # Sample diameters from DLS
    diameters = rng.choice(dls_diameters, size=n_spots, p=dls_probs)

    # Compute intensities
    # Lipid: proportional to surface area (d^2)
    lipid_intensities = labeling_eff * (diameters / 100.0)**2

    # Protein: curvature-dependent binding.
    # alpha=2 means no curvature preference (proportional to area);
    # alpha<2 means small liposomes bind more protein per unit area;
    # alpha=0 means no binding (EGFP-like, intensity ~ labeling_eff * brightness_scale).
    # eta is a small per-spot multiplicative log-normal noise (biological heterogeneity).
    eta = rng.lognormal(mean=0.0, sigma=0.1, size=n_spots)
    if params.get('independent_protein_intensity', False):
        # A_protein sampled independently of diameter (no power-law prior).
        ip_mu = params.get('independent_protein_mu', np.log(50.0))
        ip_sigma = params.get('independent_protein_sigma', 0.5)
        protein_intensities = rng.lognormal(mean=ip_mu, sigma=ip_sigma,
                                            size=n_spots)
    else:
        protein_intensities = (labeling_eff * brightness_scale
                               * (diameters / 100.0)**curvature_alpha * eta)

    # Random positions
    positions_x = rng.uniform(10, image_size - 10, size=n_spots)
    positions_y = rng.uniform(10, image_size - 10, size=n_spots)

    # Sample a background from real data
    # If bg_patches_protein is provided separately, use it; otherwise reuse bg_patches
    if bg_patches_protein is None:
        bg_patches_protein_use = bg_patches
    else:
        bg_patches_protein_use = bg_patches_protein

    bg_idx = rng.integers(0, len(bg_patches))
    bg_lipid = bg_patches[bg_idx].copy() * bg_amplitude
    bg_protein_idx = rng.integers(0, len(bg_patches_protein_use))
    # Protein channel has its own scale + autofluorescence baseline (per-sample
    # in joint calibration, optionally randomized per-image at training-data
    # generation time so the network sees the full EGFP→endophilin range).
    bg_amplitude_protein = params.get('bg_amplitude_protein', bg_amplitude)
    autofl_protein = params.get('autofl_protein', 0.0)
    bg_protein = (bg_patches_protein_use[bg_protein_idx].copy()
                  * bg_amplitude_protein + autofl_protein)

    # Ensure backgrounds are the right size
    if bg_lipid.shape != (image_size, image_size):
        bg_lipid = np.full((image_size, image_size), np.median(bg_lipid))
    if bg_protein.shape != (image_size, image_size):
        bg_protein = np.full((image_size, image_size), np.median(bg_protein))

    # Add haze (smooth out-of-focus fluorescence)
    if haze_level > 0:
        bg_lipid += haze_level
        bg_protein += haze_level * 0.5

    # Start with clean signal on background
    clean_lipid = bg_lipid.copy()
    clean_protein = bg_protein.copy()

    # Render spots as 2D Gaussians
    # Intensities are PEAK values above background (at the spot center)
    # A spot with peak=500 and sigma=2 contributes 500 counts at the center pixel
    ground_truth = []

    # Use truncated rendering for speed (only compute Gaussian within ±4*sigma)
    render_radius = int(np.ceil(max(psf_sigma_x, psf_sigma_y) * 4))

    for i in range(n_spots):
        cx, cy = positions_x[i], positions_y[i]
        li, pi = lipid_intensities[i], protein_intensities[i]

        # Compute Gaussian in a small window
        x_lo = max(0, int(cx) - render_radius)
        x_hi = min(image_size, int(cx) + render_radius + 1)
        y_lo = max(0, int(cy) - render_radius)
        y_hi = min(image_size, int(cy) + render_radius + 1)

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        gauss = np.exp(-((xx - cx)**2 / (2 * psf_sigma_x**2) +
                         (yy - cy)**2 / (2 * psf_sigma_y**2)))

        # Use PEAK intensity parameterization:
        # gauss peaks at 1.0 at the center, so li * gauss peaks at li
        clean_lipid[y_lo:y_hi, x_lo:x_hi] += li * gauss
        clean_protein[y_lo:y_hi, x_lo:x_hi] += pi * gauss

        ground_truth.append({
            'x': float(cx), 'y': float(cy),
            'diameter_nm': float(diameters[i]),
            'lipid_intensity': float(li),
            'protein_intensity': float(pi),
        })

    # Apply noise model
    # Real microscope does frame averaging (3 scans averaged per image per FV3000 metadata)
    # We simulate this by adding noise independently n_frame_avg times and averaging
    # PMTs have an excess noise factor (ENF) that makes per-photon variance higher than
    # pure Poisson. We use a Gaussian approximation: output ~ N(gain*N, ENF*gain^2*N)
    # This produces smooth noise distributions matching real PMT data.

    enf = params.get('enf', 1.3)  # excess noise factor (typical 1.0-2.0 for PMTs)

    noisy_lipid = _apply_pmt_noise(
        clean_lipid, gain, read_noise_var_lipid, offset_lipid, n_frame_avg, enf, rng
    )
    noisy_protein = _apply_pmt_noise(
        clean_protein, gain, read_noise_var_protein, offset_protein, n_frame_avg, enf, rng
    )

    # Clip to 12-bit range
    noisy_lipid = np.clip(noisy_lipid, 0, 4095)
    noisy_protein = np.clip(noisy_protein, 0, 4095)

    return noisy_protein, noisy_lipid, ground_truth


def simulate_batch(params, dls_diameters, dls_probs, bg_patches,
                   n_images=50, image_size=512, seed=None):
    """Generate a batch of simulated images (single background pool)."""
    rng = np.random.default_rng(seed)

    all_protein = []
    all_lipid = []
    all_gt = []

    for i in range(n_images):
        protein, lipid, gt = simulate_image(
            params, dls_diameters, dls_probs, bg_patches,
            image_size=image_size, rng=rng
        )
        all_protein.append(protein)
        all_lipid.append(lipid)
        all_gt.append(gt)

    return all_protein, all_lipid, all_gt


def simulate_batch_dual_bg(params, dls_diameters, dls_probs,
                           bg_patches_lipid, bg_patches_protein,
                           n_images=50, image_size=512, seed=None):
    """Generate a batch of simulated images with separate lipid and protein backgrounds."""
    rng = np.random.default_rng(seed)

    all_protein = []
    all_lipid = []
    all_gt = []

    for i in range(n_images):
        protein, lipid, gt = simulate_image(
            params, dls_diameters, dls_probs,
            bg_patches=bg_patches_lipid,
            bg_patches_protein=bg_patches_protein,
            image_size=image_size, rng=rng
        )
        all_protein.append(protein)
        all_lipid.append(lipid)
        all_gt.append(gt)

    return all_protein, all_lipid, all_gt


def simulate_protein_calibration_batch(shared_params, sample_params, dark_results,
                                       bg_patches_protein, n_images=30,
                                       image_size=512, seed=None):
    """
    Generate protein-channel images for calibration: bg + autofluorescence + noise
    only (no spot-bound protein). Used to match non-puncta protein pixel
    distributions during joint calibration.
    """
    rng = np.random.default_rng(seed)
    gain = shared_params['gain']
    enf = shared_params.get('enf', 1.3)
    haze = shared_params.get('haze_level', 0.0)
    n_frames = shared_params.get('n_frame_avg', 3)
    bg_amp_p = sample_params['bg_amplitude_protein']
    autofl = sample_params['autofl_protein']
    vscale = sample_params['voltage_scale_protein']
    offset_p = dark_results['protein']['offset']
    rn_var_p = dark_results['protein']['read_noise_var']

    out = []
    for _ in range(n_images):
        idx = rng.integers(0, len(bg_patches_protein))
        bg = bg_patches_protein[idx]
        if bg.shape != (image_size, image_size):
            bg = np.full((image_size, image_size), float(np.median(bg)))
        clean = (bg * bg_amp_p + autofl + 0.5 * haze) * vscale
        noisy = _apply_pmt_noise(clean, gain, rn_var_p, offset_p, n_frames, enf, rng)
        out.append(np.clip(noisy, 0, 4095))
    return out


def extract_protein_nonpuncta_pixels(protein_img, lipid_img, mask_radius=2,
                                     snr_threshold=3.0):
    """
    Return the protein-channel pixel values at locations far from lipid puncta.

    Detects local maxima in the lipid channel above a simple background+SNR
    threshold (no trained model), masks a (2*mask_radius+1) square around each
    peak, and returns the protein pixels at all UN-masked positions, flattened.
    """
    bg = float(np.median(lipid_img))
    bg_std = float(np.std(lipid_img[lipid_img < np.percentile(lipid_img, 70)]))
    threshold = bg + snr_threshold * bg_std
    local_max = maximum_filter(lipid_img, size=2 * mask_radius + 1)
    peaks = (lipid_img == local_max) & (lipid_img > threshold)

    mask = np.zeros_like(lipid_img, dtype=bool)
    coords = np.argwhere(peaks)
    H, W = lipid_img.shape
    for py, px in coords:
        y0 = max(0, py - mask_radius); y1 = min(H, py + mask_radius + 1)
        x0 = max(0, px - mask_radius); x1 = min(W, px + mask_radius + 1)
        mask[y0:y1, x0:x1] = True
    return protein_img[~mask].astype(np.float64)


def _gather_nonpuncta_protein(images_dict_or_arrays, sim_protein_arrays=None,
                              sim_lipid_arrays=None, max_pixels=200000,
                              rng=None):
    """
    Collect non-puncta protein pixels from either a list of real image dicts
    (each with 'protein' and 'lipid') or aligned simulated arrays.
    Returns a 1d float array, subsampled to at most `max_pixels`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pixels = []
    if sim_protein_arrays is None:
        for d in images_dict_or_arrays:
            pixels.append(extract_protein_nonpuncta_pixels(d['protein'], d['lipid']))
    else:
        for p_img, l_img in zip(sim_protein_arrays, sim_lipid_arrays):
            pixels.append(extract_protein_nonpuncta_pixels(p_img, l_img))
    if not pixels:
        return np.array([])
    flat = np.concatenate(pixels)
    if flat.size > max_pixels:
        idx = rng.choice(flat.size, size=max_pixels, replace=False)
        flat = flat[idx]
    return flat
