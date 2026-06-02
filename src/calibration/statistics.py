"""Summary statistics for moment matching.

  - compute_image_statistics: histograms, radial PSD, spot stats, noise curve

Works on both real image dicts and simulated numpy arrays. Ported verbatim
from the archive's pipeline.py (Module 7).
"""

import numpy as np
from scipy.ndimage import maximum_filter


def compute_image_statistics(images_or_arrays, channel=None, is_simulated=False):
    """
    Compute summary statistics from a set of images for moment matching.
    Works on both real image dicts and simulated numpy arrays.

    Returns:
        dict of statistics (histograms, spectra, spot stats)
    """
    # Extract the right channel
    if is_simulated:
        arrays = images_or_arrays  # already numpy arrays
    else:
        arrays = [img[channel] for img in images_or_arrays]

    all_pixels = []
    all_spot_intensities = []
    all_spot_widths = []
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

        # Spot detection (crude local max finder)
        bg = np.median(img)
        bg_std = np.std(img[img < np.percentile(img, 70)])
        threshold = bg + 3 * bg_std

        local_max = maximum_filter(img, size=7)
        peaks = (img == local_max) & (img > threshold)
        peak_coords = np.argwhere(peaks)

        spot_counts.append(len(peak_coords))

        for py, px in peak_coords:
            all_spot_intensities.append(float(img[py, px]))

            # Estimate width via second moment
            r_crop = 4
            if (py > r_crop and py < img.shape[0]-r_crop and
                px > r_crop and px < img.shape[1]-r_crop):
                patch = img[py-r_crop:py+r_crop+1, px-r_crop:px+r_crop+1]
                patch_bg = np.min(patch)
                patch_sub = patch - patch_bg
                total = patch_sub.sum()
                if total > 0:
                    yy, xx = np.mgrid[-r_crop:r_crop+1, -r_crop:r_crop+1]
                    sigma_est = np.sqrt((patch_sub * (xx**2 + yy**2)).sum() / total / 2)
                    if 0.5 < sigma_est < 10:
                        all_spot_widths.append(float(sigma_est))

        # Noise in background regions (for noise-vs-signal curve)
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

    # Spot intensity histogram
    spot_int_bins = np.linspace(0, 2000, 100)
    if len(all_spot_intensities) > 0:
        spot_hist, _ = np.histogram(all_spot_intensities, bins=spot_int_bins, density=True)
    else:
        spot_hist = np.zeros(len(spot_int_bins) - 1)

    # Average PSD
    min_len = min(len(p) for p in all_psd)
    avg_psd = np.mean([p[:min_len] for p in all_psd], axis=0)

    stats = {
        'pixel_hist': pixel_hist,
        'pixel_hist_bins': hist_bins,
        'spot_intensity_hist': spot_hist,
        'spot_intensity_bins': spot_int_bins,
        'spot_intensities': np.array(all_spot_intensities) if all_spot_intensities else np.array([]),
        'spot_widths': np.array(all_spot_widths) if all_spot_widths else np.array([]),
        'mean_spot_count': float(np.mean(spot_counts)) if spot_counts else 0,
        'std_spot_count': float(np.std(spot_counts)) if spot_counts else 0,
        'radial_psd': avg_psd,
        'noise_means': np.array(noise_means),
        'noise_vars': np.array(noise_vars),
        'mean_pixel': float(np.mean(all_pixels)),
        'std_pixel': float(np.std(all_pixels)),
    }

    return stats
