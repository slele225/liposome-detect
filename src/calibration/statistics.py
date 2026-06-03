"""Summary statistics for moment matching.

  - compute_image_statistics: pixel histogram, high quantiles, skewness, radial
    PSD, a crude spot count, and a background noise curve.

Works on both real image dicts and simulated numpy arrays.

The calibration objective is detection-free: the per-spot intensity/width
extraction was removed. A crude local-maximum COUNT is still computed because
``optimize.py`` uses ``mean_spot_count`` only to anchor the per-sample
``spot_density`` prior range (ref_spots); it does not enter any discrepancy term.
"""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import skew


def compute_image_statistics(images_or_arrays, channel=None, is_simulated=False):
    """
    Compute summary statistics from a set of images for moment matching.
    Works on both real image dicts and simulated numpy arrays.

    Returns:
        dict of statistics (pixel histogram, high quantiles, skewness, radial
        PSD, mean/std pixel, crude spot count, background noise curve).
    """
    # Extract the right channel
    if is_simulated:
        arrays = images_or_arrays  # already numpy arrays
    else:
        arrays = [img[channel] for img in images_or_arrays]

    all_pixels = []
    spot_counts = []
    all_psd = []
    noise_means = []
    noise_vars = []

    for img in arrays:
        img = img.astype(np.float64)
        all_pixels.append(img.ravel())

        # Power spectral density
        f_img = np.fft.fft2(img)
        psd = np.abs(f_img)**2
        # Radial average
        cy, cx = psd.shape[0]//2, psd.shape[1]//2
        psd_shifted = np.fft.fftshift(psd)
        y_grid, x_grid = np.ogrid[-cy:psd.shape[0]-cy, -cx:psd.shape[1]-cx]
        r = np.sqrt(x_grid**2 + y_grid**2).astype(int)
        max_r = min(cy, cx)
        radial_psd = np.zeros(max_r)
        for ri in range(max_r):
            ring = psd_shifted[r == ri]
            if len(ring) > 0:
                radial_psd[ri] = np.mean(ring)
        all_psd.append(radial_psd)

        # Crude local-max spot COUNT (detection-free objective: used only to
        # anchor the spot_density prior range, not in any discrepancy term).
        bg = np.median(img)
        bg_std = np.std(img[img < np.percentile(img, 70)])
        threshold = bg + 3 * bg_std

        local_max = maximum_filter(img, size=7)
        peaks = (img == local_max) & (img > threshold)
        spot_counts.append(int(peaks.sum()))

        # Noise in background regions (for the noise-vs-signal curve)
        patch_size = 8
        for y in range(0, img.shape[0] - patch_size, patch_size * 2):
            for x in range(0, img.shape[1] - patch_size, patch_size * 2):
                patch = img[y:y+patch_size, x:x+patch_size]
                pmean = np.mean(patch)
                pvar = np.var(patch)
                if pmean < bg + 2 * bg_std:  # background only
                    noise_means.append(pmean)
                    noise_vars.append(pvar)

    # Compile statistics
    all_pixels = np.concatenate(all_pixels)

    # Pixel intensity histogram (use fixed bins for comparability)
    hist_bins = np.linspace(0, 1000, 200)
    pixel_hist, _ = np.histogram(all_pixels, bins=hist_bins, density=True)

    # Average PSD
    min_len = min(len(p) for p in all_psd)
    avg_psd = np.mean([p[:min_len] for p in all_psd], axis=0)

    stats = {
        'pixel_hist': pixel_hist,
        'pixel_hist_bins': hist_bins,
        'radial_psd': avg_psd,
        'noise_means': np.array(noise_means),
        'noise_vars': np.array(noise_vars),
        'mean_pixel': float(np.mean(all_pixels)),
        'std_pixel': float(np.std(all_pixels)),
        # High quantiles + skewness pin the bright-spot tail without detection.
        'p99': float(np.percentile(all_pixels, 99)),
        'p999': float(np.percentile(all_pixels, 99.9)),
        'skewness': float(skew(all_pixels)),
        # Crude spot count: anchors the spot_density prior only (see module doc).
        'mean_spot_count': float(np.mean(spot_counts)) if spot_counts else 0.0,
        'std_spot_count': float(np.std(spot_counts)) if spot_counts else 0.0,
    }

    return stats
