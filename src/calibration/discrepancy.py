"""Config-driven real-vs-simulated image discrepancy.

`compute_discrepancy` sums up to six terms, each gated by a per-term
``{enabled: bool, weight: float}`` entry. An enabled term contributes
``weight * raw_term``; a disabled term is skipped entirely.

The DEFAULT weights below are chosen so that calling ``compute_discrepancy``
with no ``discrepancy_config`` reproduces *exactly* the old hardcoded formula
from the archive's pipeline.py (Module 9), where the raw terms were divided by
fixed constants:

    pixel_hist        -> w_pixel      / 100   == 0.01  * w_pixel
    spot_intensity    -> w_spot       / 200   == 0.005 * w_spot
    psd               -> psd_mse      * 1.0   == 1.0   * psd_mse
    spot_density      -> density_err  * 1.0   == 1.0   * (density_err / norm)
    mean_pixel        -> mean_err     * 1.0   == 1.0   * (mean_err / norm)
    protein_nonpuncta -> w_protein_np / 200   == 0.005 * w_protein_np

A calibration YAML may supply a ``discrepancy:`` block to override any term's
``enabled``/``weight``; unspecified terms fall back to these defaults.
"""

import numpy as np
from scipy.stats import wasserstein_distance


# Per-term defaults. These reproduce the archive's hardcoded normalization.
DEFAULT_DISCREPANCY_CONFIG = {
    'pixel_hist':        {'enabled': True, 'weight': 0.01},
    'spot_intensity':    {'enabled': True, 'weight': 0.005},
    'psd':               {'enabled': True, 'weight': 1.0},
    'spot_density':      {'enabled': True, 'weight': 1.0},
    'mean_pixel':        {'enabled': True, 'weight': 1.0},
    'protein_nonpuncta': {'enabled': True, 'weight': 0.005},
}


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
    Compute scalar discrepancy between real and simulated image statistics.

    Args:
        real_stats, sim_stats: dicts from ``compute_image_statistics`` (plus an
            optional ``protein_nonpuncta`` array).
        discrepancy_config: optional per-term overrides. When ``None`` the
            defaults reproduce the archive's hardcoded weighting exactly.
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

    # 2. Spot intensity distribution (Wasserstein distance)
    term = cfg['spot_intensity']
    if (term['enabled']
            and len(real_stats['spot_intensities']) > 10
            and len(sim_stats['spot_intensities']) > 10):
        w_spot = wasserstein_distance(real_stats['spot_intensities'], sim_stats['spot_intensities'])
        loss += term['weight'] * w_spot

    # 3. Power spectral density (MSE on log scale)
    term = cfg['psd']
    if term['enabled']:
        min_len = min(len(real_stats['radial_psd']), len(sim_stats['radial_psd']))
        if min_len > 5:
            real_psd = np.log10(real_stats['radial_psd'][:min_len] + 1e-10)
            sim_psd = np.log10(sim_stats['radial_psd'][:min_len] + 1e-10)
            psd_mse = np.mean((real_psd[1:] - sim_psd[1:])**2)  # skip DC component
            loss += term['weight'] * psd_mse

    # 4. Spot density (squared error)
    term = cfg['spot_density']
    if term['enabled']:
        density_err = (real_stats['mean_spot_count'] - sim_stats['mean_spot_count'])**2
        loss += term['weight'] * (density_err / max(real_stats['mean_spot_count']**2, 1))

    # 5. Mean pixel intensity (squared error)
    term = cfg['mean_pixel']
    if term['enabled']:
        mean_err = (real_stats['mean_pixel'] - sim_stats['mean_pixel'])**2
        loss += term['weight'] * (mean_err / max(real_stats['mean_pixel']**2, 1))

    # 6. Protein non-puncta pixel distribution (Wasserstein) — only if both
    #    sides supplied a 'protein_nonpuncta' array.
    term = cfg['protein_nonpuncta']
    if term['enabled']:
        real_np = real_stats.get('protein_nonpuncta')
        sim_np = sim_stats.get('protein_nonpuncta')
        if (real_np is not None and sim_np is not None
                and len(real_np) > 10 and len(sim_np) > 10):
            w_protein_np = wasserstein_distance(real_np, sim_np)
            loss += term['weight'] * w_protein_np

    return loss
