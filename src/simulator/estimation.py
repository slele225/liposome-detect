"""Microscope parameter estimation from real images.

  - estimate_gain      : effective PMT gain via photon-transfer (Var vs Mean)
  - gaussian_2d        : 2D Gaussian model used for PSF fitting
  - estimate_psf       : PSF widths from bright isolated spots
  - extract_backgrounds: spot-masked background structure

Ported verbatim from the archive's pipeline.py (Modules 4-6).
"""

import numpy as np
from scipy.ndimage import maximum_filter, median_filter
from scipy.optimize import curve_fit


def estimate_gain(images, dark_results, channel='lipid', patch_size=16, min_patches=500):
    """
    Estimate effective gain from variance-vs-mean relationship in background regions.
    Uses the photon transfer method: Var = gain * (Mean - offset) + read_noise_var

    Returns:
        gain: effective gain (slope of variance vs mean)
    """
    print(f"=== Gain Estimation ({channel}) ===")

    offset = dark_results[channel]['offset']

    means = []
    variances = []

    for img_dict in images:
        img = img_dict[channel]

        # Divide image into patches, use only low-intensity patches (background)
        for y in range(0, img.shape[0] - patch_size, patch_size):
            for x in range(0, img.shape[1] - patch_size, patch_size):
                patch = img[y:y+patch_size, x:x+patch_size]
                patch_mean = np.mean(patch)
                patch_var = np.var(patch)

                # Only use background-like patches (below median + modest threshold)
                # and above offset (need some signal for the fit)
                if offset + 5 < patch_mean < offset + 500:
                    means.append(patch_mean - offset)
                    variances.append(patch_var)

    means = np.array(means)
    variances = np.array(variances)

    if len(means) < 50:
        print(f"  Warning: only {len(means)} valid patches. Gain estimate may be unreliable.")
        # Fallback: assume gain = 1
        print(f"  Using default gain = 1.0")
        return 1.0

    # Fit: Var = gain * Mean + read_noise_var
    # Use robust fit (ignore outliers)
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(means, variances, 1)  # coeffs[0] = intercept, coeffs[1] = slope
    gain = float(coeffs[1])

    if gain <= 0:
        print(f"  Warning: negative gain estimated ({gain:.3f}). Using gain = 1.0")
        gain = 1.0

    print(f"  Estimated gain: {gain:.3f}")
    print(f"  Used {len(means)} background patches")

    return gain


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian function for fitting."""
    x, y = xy
    return offset + amplitude * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    )


def estimate_psf(images, channel='lipid', crop_radius=7, min_separation=15,
                 snr_threshold=5, max_spots=100):
    """
    Estimate PSF width by fitting 2D Gaussians to bright, isolated spots.

    Returns:
        sigma_x: median PSF width in x (pixels)
        sigma_y: median PSF width in y (pixels)
        all_sigmas: list of (sigma_x, sigma_y) for each fitted spot
    """
    print(f"=== PSF Estimation ({channel}) ===")

    all_sigmas = []

    for img_idx, img_dict in enumerate(images):
        img = img_dict[channel]

        # Find local maxima
        bg = np.median(img)
        bg_std = np.std(img[img < np.percentile(img, 70)])  # noise from lower pixels
        threshold = bg + snr_threshold * bg_std

        local_max = maximum_filter(img, size=2*crop_radius+1)
        peaks = (img == local_max) & (img > threshold)
        peak_coords = np.argwhere(peaks)

        # Filter: must be far from edges
        margin = crop_radius + 2
        peak_coords = peak_coords[
            (peak_coords[:, 0] > margin) & (peak_coords[:, 0] < img.shape[0] - margin) &
            (peak_coords[:, 1] > margin) & (peak_coords[:, 1] < img.shape[1] - margin)
        ]

        if len(peak_coords) == 0:
            continue

        # Filter: must be isolated (no neighbor within min_separation)
        isolated = []
        for i, p in enumerate(peak_coords):
            dists = np.sqrt(np.sum((peak_coords - p)**2, axis=1))
            dists[i] = np.inf  # exclude self
            if np.min(dists) > min_separation:
                isolated.append(p)

        # Sort by brightness, take top spots
        isolated = sorted(isolated, key=lambda p: img[p[0], p[1]], reverse=True)
        isolated = isolated[:max_spots // max(len(images), 1)]

        for py, px in isolated:
            # Crop patch
            patch = img[py-crop_radius:py+crop_radius+1,
                       px-crop_radius:px+crop_radius+1].copy()

            # Local background subtraction
            local_bg = np.median(np.concatenate([
                patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]
            ]))

            # Fit 2D Gaussian
            size = 2 * crop_radius + 1
            y_grid, x_grid = np.mgrid[0:size, 0:size]

            try:
                p0 = [patch.max() - local_bg, crop_radius, crop_radius, 2.0, 2.0, local_bg]
                bounds = ([0, crop_radius-3, crop_radius-3, 0.5, 0.5, 0],
                         [4096, crop_radius+3, crop_radius+3, 8.0, 8.0, 4096])

                popt, _ = curve_fit(
                    gaussian_2d, (x_grid.ravel(), y_grid.ravel()), patch.ravel(),
                    p0=p0, bounds=bounds, maxfev=5000
                )

                sx, sy = abs(popt[3]), abs(popt[4])

                # Sanity check: PSF should be ~1.5-6 pixels wide
                if 1.0 < sx < 7.0 and 1.0 < sy < 7.0:
                    all_sigmas.append((sx, sy))
            except (RuntimeError, ValueError):
                continue

    if len(all_sigmas) < 5:
        print(f"  Warning: only fitted {len(all_sigmas)} spots. Using theoretical PSF estimate.")
        # Theoretical estimate: sigma ≈ 0.21 * lambda / NA / pixel_size
        # For 520nm emission, NA=1.2, pixel=69nm: sigma ≈ 0.21*520/1.2/69 ≈ 1.3 pixels
        # But confocal is slightly better, and real PSF is broader due to aberrations
        # Conservative estimate: ~2 pixels
        return 2.0, 2.0, []

    sigmas = np.array(all_sigmas)
    sigma_x = float(np.median(sigmas[:, 0]))
    sigma_y = float(np.median(sigmas[:, 1]))

    print(f"  Fitted {len(all_sigmas)} spots")
    print(f"  PSF sigma_x={sigma_x:.2f} px ({sigma_x*69:.0f} nm), "
          f"sigma_y={sigma_y:.2f} px ({sigma_y*69:.0f} nm)")
    print(f"  Spread: sigma_x std={np.std(sigmas[:,0]):.2f}, sigma_y std={np.std(sigmas[:,1]):.2f}")

    return sigma_x, sigma_y, all_sigmas


def extract_backgrounds(images, channel='lipid', spot_mask_radius=8):
    """
    Extract background structure by masking out detected spots.

    Returns:
        bg_patches: list of background images (spots masked and interpolated)
        bg_stats: dict with background statistics
    """
    print(f"=== Background Extraction ({channel}) ===")

    bg_patches = []
    bg_means = []
    bg_stds = []

    for img_dict in images:
        img = img_dict[channel]

        # Detect spots (crude: local maxima above threshold)
        bg_level = np.median(img)
        bg_std = np.std(img[img < np.percentile(img, 70)])
        threshold = bg_level + 3 * bg_std

        local_max = maximum_filter(img, size=2*spot_mask_radius+1)
        peaks = (img == local_max) & (img > threshold)

        # Create mask around each peak
        mask = np.zeros_like(img, dtype=bool)
        peak_coords = np.argwhere(peaks)
        for py, px in peak_coords:
            y_lo = max(0, py - spot_mask_radius)
            y_hi = min(img.shape[0], py + spot_mask_radius + 1)
            x_lo = max(0, px - spot_mask_radius)
            x_hi = min(img.shape[1], px + spot_mask_radius + 1)
            mask[y_lo:y_hi, x_lo:x_hi] = True

        # Fill masked regions with median-filtered background estimate
        bg = img.copy()
        bg_smooth = median_filter(img, size=2*spot_mask_radius+1)
        bg[mask] = bg_smooth[mask]

        bg_patches.append(bg)
        bg_means.append(np.mean(bg[~mask]))
        bg_stds.append(np.std(bg[~mask]))

    bg_stats = {
        'mean': float(np.mean(bg_means)),
        'std': float(np.mean(bg_stds)),
        'min_mean': float(np.min(bg_means)),
        'max_mean': float(np.max(bg_means)),
    }

    print(f"  Extracted {len(bg_patches)} background patches")
    print(f"  Background mean={bg_stats['mean']:.1f}, std={bg_stats['std']:.1f}")
    print(f"  Range of means: {bg_stats['min_mean']:.1f} - {bg_stats['max_mean']:.1f}")

    return bg_patches, bg_stats
