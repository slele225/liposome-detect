"""Config-driven real-vs-simulated lipid-image discrepancy.

Calibration is lipid-only and detection-free. ``compute_discrepancy`` sums up
to five terms, each gated by a per-term ``{enabled: bool, weight: float}``
entry. An enabled term contributes ``weight * raw_term``; a disabled term is
skipped entirely.

Terms and default weights
-------------------------
| term       | weight | raw term                                              |
|------------|--------|-------------------------------------------------------|
| pixel_hist | 0.01   | pixel-intensity Wasserstein distance                  |
| psd        | 1.0    | MSE of log10 radial power spectral density            |
| mean_pixel | 1.0    | relative squared error of the mean pixel intensity    |
| quantiles  | 1.0    | sum of relative sq. errors of the 99th & 99.9th pctl  |
| skewness   | 1.0    | relative squared error of the pixel-distribution skew |

``mean_pixel``, ``quantiles`` and ``skewness`` are all RELATIVE squared errors
(same construction, denominator ``max(real**2, 1)``), so a single shared weight
of 1.0 puts them on the same scale by design. On a typical 20nM_EGFP trial
(real lipid 256-crop: mean~201, p99~628, p99.9~1006, skew~4.5) a ~20% miss on a
quantile or skewness contributes ~0.04-0.09 each — the same order of magnitude
as ``mean_pixel`` — while ``pixel_hist`` (Wasserstein in ADU, ~tens) * 0.01 and
``psd`` remain the larger structural terms, as before.

A calibration YAML may supply a ``discrepancy:`` block to override any term's
``enabled``/``weight``; unspecified terms fall back to these defaults.
"""

import numpy as np
from scipy.stats import wasserstein_distance


# Per-term defaults.
DEFAULT_DISCREPANCY_CONFIG = {
    'pixel_hist': {'enabled': True, 'weight': 0.01},
    'psd':        {'enabled': True, 'weight': 1.0},
    'mean_pixel': {'enabled': True, 'weight': 1.0},
    'quantiles':  {'enabled': True, 'weight': 1.0},
    'skewness':   {'enabled': True, 'weight': 1.0},
}


def _rel_sq_err(real, sim):
    """Relative squared error, floored like the original mean_pixel term."""
    return (real - sim)**2 / max(real**2, 1.0)


def resolve_discrepancy_config(discrepancy_config=None):
    """Merge a user-supplied per-term config over the defaults.

    Returns a fresh dict mapping each known term to a ``{enabled, weight}``
    dict. ``discrepancy_config`` may specify any subset of terms and, for each,
    any subset of keys; unspecified values fall back to the defaults. Unknown
    term names raise ``KeyError`` to catch typos in YAML.
    """
    resolved = {term: dict(spec) for term, spec in DEFAULT_DISCREPANCY_CONFIG.items()}
    if discrepancy_config:
        for term, override in discrepancy_config.items():
            if term not in resolved:
                raise KeyError(
                    f"Unknown discrepancy term '{term}'. Valid terms: "
                    f"{sorted(resolved)}")
            if override:
                resolved[term].update(override)
    return resolved


def compute_discrepancy(real_stats, sim_stats, discrepancy_config=None):
    """
    Compute the scalar lipid-image discrepancy between real and simulated stats.

    Args:
        real_stats, sim_stats: dicts from ``compute_image_statistics``.
        discrepancy_config: optional per-term overrides. When ``None`` the
            defaults above are used.
    """
    cfg = resolve_discrepancy_config(discrepancy_config)
    loss = 0.0

    # 1. Pixel intensity histogram (Wasserstein distance)
    term = cfg['pixel_hist']
    if term['enabled']:
        w_pixel = wasserstein_distance(
            real_stats['pixel_hist_bins'][:-1], sim_stats['pixel_hist_bins'][:-1],
            u_weights=real_stats['pixel_hist'], v_weights=sim_stats['pixel_hist']
        )
        loss += term['weight'] * w_pixel

    # 2. Power spectral density (MSE on log scale)
    term = cfg['psd']
    if term['enabled']:
        min_len = min(len(real_stats['radial_psd']), len(sim_stats['radial_psd']))
        if min_len > 5:
            real_psd = np.log10(real_stats['radial_psd'][:min_len] + 1e-10)
            sim_psd = np.log10(sim_stats['radial_psd'][:min_len] + 1e-10)
            psd_mse = np.mean((real_psd[1:] - sim_psd[1:])**2)  # skip DC component
            loss += term['weight'] * psd_mse

    # 3. Mean pixel intensity (relative squared error)
    term = cfg['mean_pixel']
    if term['enabled']:
        loss += term['weight'] * _rel_sq_err(real_stats['mean_pixel'],
                                              sim_stats['mean_pixel'])

    # 4. High quantiles of the pixel distribution (sum of relative squared
    #    errors on the 99th and 99.9th percentiles) — pins the bright tail.
    term = cfg['quantiles']
    if term['enabled']:
        q_err = (_rel_sq_err(real_stats['p99'], sim_stats['p99'])
                 + _rel_sq_err(real_stats['p999'], sim_stats['p999']))
        loss += term['weight'] * q_err

    # 5. Skewness of the pixel distribution (relative squared error)
    term = cfg['skewness']
    if term['enabled']:
        loss += term['weight'] * _rel_sq_err(real_stats['skewness'],
                                             sim_stats['skewness'])

    return loss
