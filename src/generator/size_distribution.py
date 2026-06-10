"""Spot-diameter PMFs handed to the simulator in place of the DLS arrays.

The simulator samples diameters via ``rng.choice(dls_diameters, p=dls_probs)``;
we exploit that by passing a CUSTOM (diameters, probs) PMF, without editing the
simulator (see docs/decisions/2026-06-04_synthetic-generation-strategy.md):

  - TRAINING (``emphasis``): a wide, small-size-emphasis distribution. Default is
    uniform-in-CURVATURE (pdf ∝ 1/d^2), because curvature ∝ 1/d makes the
    small/high-curvature tail the most important AND hardest regime. A
    ``size_emphasis`` knob blends uniform-in-d (0) <-> uniform-in-1/d (1).
  - TESTING / Phase 3 (``dls``): the REAL DLS distribution via ``io.parse_dls``.
"""

import numpy as np


def curvature_emphasis_pmf(d_min, d_max, n_bins=256, size_emphasis=1.0):
    """Small-size-emphasis diameter PMF over [d_min, d_max].

    Built on a uniform-in-d grid; weighting each grid point by ``1/d^2`` realizes
    the uniform-in-curvature (1/d) density on that grid. ``size_emphasis`` blends
    the two normalized components:

        p(d) ∝ (1 - e) * uniform   +   e * (1/d^2)

    e=0 -> uniform in diameter; e=1 (default) -> uniform in curvature (tilted
    small). Returns ``(diameters, probs)`` with ``probs`` summing to 1.
    """
    if d_min <= 0 or d_max <= d_min:
        raise ValueError(f"need 0 < d_min < d_max, got d_min={d_min}, d_max={d_max}")
    diam = np.linspace(float(d_min), float(d_max), int(n_bins))

    p_uniform = np.full_like(diam, 1.0)
    p_uniform /= p_uniform.sum()

    p_curv = 1.0 / (diam ** 2)
    p_curv /= p_curv.sum()

    e = float(np.clip(size_emphasis, 0.0, 1.0))
    probs = (1.0 - e) * p_uniform + e * p_curv
    probs /= probs.sum()
    return diam, probs


def dls_pmf(dls_path, weighting='number', max_diameter_nm=500):
    """Real DLS diameter PMF (test/Phase-3 mode), via ``io.parse_dls``.

    Returns ``(diameters, probs)`` exactly as the calibration path consumes them.
    """
    from src.simulator.io import parse_dls
    diameters, probs, _ = parse_dls(
        dls_path, weighting=weighting, max_diameter_nm=max_diameter_nm)
    return diameters, probs
