"""Read per-sample calibration outputs and build the generator's sampling ranges.

The generator samples microscope/biology parameters over the UNION of the named
per-sample calibrations' fitted ranges, widened ±30% (see
docs/decisions/2026-06-04_synthetic-generation-strategy.md). This module parses
the real ``calibration_results.json`` schema and exposes:

  - ``load_calibration``    : one JSON -> a flat per-sample dict (the fields the
                              generator needs: fitted lipid params + measured
                              protein PSF + pinned dark floors).
  - ``build_param_ranges``  : union [min,max] + ±widen of the sampled quantities
                              across several calibrations.

Schema consumed (confirmed against real outputs):
  best_params              : lipid_brightness, psf_sigma_x, psf_sigma_y, psf_theta,
                             gain, enf, optical_bg_lipid, n_frame_avg (passthrough)
  per_sample_params.<name> : spot_density, offset_lipid, offset_protein,
                             read_noise_var_lipid, read_noise_var_protein
  measured_params          : psf_sigma_x_protein, psf_sigma_y_protein

IMPORTANT: the FITTED ``best_params.gain`` is used (the effective noise-model
gain). ``measured_params.gain`` (the photon-transfer estimate, ~15-40x larger) is
deliberately NOT read here — see 2026-06-03_calibration-findings.md.
"""

import json
from pathlib import Path


def load_calibration(path):
    """Parse one per-sample ``calibration_results.json`` into a flat dict.

    Returns a dict carrying everything the generator needs from this sample: the
    fitted lipid parameters used to build ranges (``lipid_brightness``,
    ``spot_density``, ``psf_sigma_x/y``, ``gain``, ``enf``, ``optical_bg_lipid``),
    plus the per-sample-pinned protein PSF (measured) and dark floors that define
    this sample's "regime".
    """
    data = json.loads(Path(path).read_text())
    bp = data['best_params']
    per_sample = data['per_sample_params']
    if not per_sample:
        raise ValueError(f"{path}: per_sample_params is empty (expected one sample)")
    name, per = next(iter(per_sample.items()))  # per-sample calibration = one sample
    measured = data.get('measured_params', {})

    return {
        'name': name,
        # fitted lipid params (used to build union sampling ranges)
        'lipid_brightness': float(bp['lipid_brightness']),
        'spot_density': float(per['spot_density']),
        'psf_sigma_x': float(bp['psf_sigma_x']),
        'psf_sigma_y': float(bp['psf_sigma_y']),
        'gain': float(bp['gain']),                 # FITTED gain (not measured_params.gain)
        'enf': float(bp['enf']),
        'optical_bg_lipid': float(bp.get('optical_bg_lipid', 0.0)),
        'n_frame_avg': int(bp.get('n_frame_avg', 3)),
        # protein channel: measured PSF (fall back to lipid PSF if absent) + pinned floor
        'psf_sigma_x_protein': float(measured.get('psf_sigma_x_protein', bp['psf_sigma_x'])),
        'psf_sigma_y_protein': float(measured.get('psf_sigma_y_protein', bp['psf_sigma_y'])),
        'offset_lipid': float(per['offset_lipid']),
        'read_noise_var_lipid': float(per['read_noise_var_lipid']),
        'offset_protein': float(per['offset_protein']),
        'read_noise_var_protein': float(per['read_noise_var_protein']),
    }


def _widen(lo, hi, frac):
    """Widen [lo, hi] outward by ``frac`` past each extreme.

    For the default ``frac=0.3`` and positive extremes this is exactly the
    spec's ``(min*0.7, max*1.3)``. Both extremes are assumed positive (true for
    brightness, density, sigma, noise_scale).
    """
    return lo * (1.0 - frac), hi * (1.0 + frac)


def build_param_ranges(calibrations, widen_frac=0.3):
    """Union [min,max] of the sampled quantities across calibrations, ±widened.

    Args:
        calibrations: list of dicts from :func:`load_calibration`.
        widen_frac: outward widening fraction (default 0.3 -> min*0.7, max*1.3).

    Returns a dict ``{'raw': {...}, 'widened': {...}, 'widen_frac': frac}`` where
    each inner dict maps a quantity to a ``(lo, hi)`` tuple. Quantities:
    ``lipid_brightness``, ``spot_density``, ``psf_sigma_x``, ``psf_sigma_y``,
    ``noise_scale`` (= gain*enf, the degenerate product calibration constrains),
    and ``sigma`` (the single near-circular PSF width = union of the sigma_x and
    sigma_y ranges).

    All values are plain Python floats/tuples (no numpy) so the result pickles to
    worker processes without importing numpy before BLAS pinning.
    """
    if not calibrations:
        raise ValueError("build_param_ranges requires at least one calibration")

    def union(key):
        vals = [float(c[key]) for c in calibrations]
        return min(vals), max(vals)

    noise_scales = [float(c['gain']) * float(c['enf']) for c in calibrations]
    raw = {
        'lipid_brightness': union('lipid_brightness'),
        'spot_density': union('spot_density'),
        'psf_sigma_x': union('psf_sigma_x'),
        'psf_sigma_y': union('psf_sigma_y'),
        'noise_scale': (min(noise_scales), max(noise_scales)),
    }
    # Single near-circular PSF width: draw from the union of the sigma_x/sigma_y
    # ranges (one sigma + small eccentricity, not independent x/y).
    raw['sigma'] = (min(raw['psf_sigma_x'][0], raw['psf_sigma_y'][0]),
                    max(raw['psf_sigma_x'][1], raw['psf_sigma_y'][1]))

    widened = {k: tuple(_widen(lo, hi, widen_frac)) for k, (lo, hi) in raw.items()}
    return {'raw': raw, 'widened': widened, 'widen_frac': float(widen_frac)}


def build_regimes(calibrations):
    """Per-sample 'regimes' = the pinned protein PSF + dark floors per sample.

    Each generated image picks one regime (its ``sample_regime_id``); the regime
    supplies the measured protein PSF and the pinned lipid/protein dark floors,
    while the microscope dynamics (brightness, density, sigma, noise scale,
    optical bg) are sampled from the widened union ranges independently.
    Returns plain-Python dicts (picklable without numpy).
    """
    return [{
        'name': c['name'],
        'n_frame_avg': int(c['n_frame_avg']),
        'offset_lipid': float(c['offset_lipid']),
        'read_noise_var_lipid': float(c['read_noise_var_lipid']),
        'offset_protein': float(c['offset_protein']),
        'read_noise_var_protein': float(c['read_noise_var_protein']),
        'psf_sigma_x_protein': float(c['psf_sigma_x_protein']),
        'psf_sigma_y_protein': float(c['psf_sigma_y_protein']),
    } for c in calibrations]
