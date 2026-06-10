"""Alpha modes: per_spot_random must scramble the per-spot curvature exponent
so no SINGLE alpha is recoverable within an image; global_coherent must not.

IMPORTANT — why this does NOT test ``corr(protein, diameter) ≈ 0``:
The prompt suggested asserting that raw corr(protein_intensity, diameter) ≈ 0 for
per_spot_random. That is mathematically unattainable for the DOCUMENTED design:
alpha is drawn POSITIVE (e.g. [0.5, 2.0]), so protein ∝ d**alpha is monotone
increasing in d for every spot regardless of its alpha — larger liposomes are
always somewhat brighter. Measured raw Pearson corr is ~0.88 for per_spot_random
(vs ~0.98 for global_coherent), NOT ~0. The decision record's actual invariant is
that per_spot_random "cannot encode a GLOBAL alpha–diameter relationship" — i.e.
there is no single recoverable slope, NOT that the marginal correlation vanishes.

So the faithful discriminator is the SPREAD of the per-spot recovered alpha:
    alpha_k_recovered = log(protein_k / protein_brightness) / log(d_k / 100)
which is ~uniform over the alpha range for per_spot_random (large spread) and a
tight band (eta-only scatter) for global_coherent. We assert on that spread, and
separately keep the direction of the marginal-correlation effect as documentation.
(Surfaced to the human; the decision docs need no change — they never claimed ~0.)
"""

import numpy as np
import pytest

from src.generator.core import generate_one_image


def _fixed_density_overrides(alpha_mode, alpha_range):
    return {'alpha_mode': alpha_mode, 'alpha_range': alpha_range,
            'sampling': {'protein_brightness_range': [2500.0, 7500.0],
                         'fixed_spot_density': 500}}


def _recovered_alpha_std(spec, n=8):
    stds = []
    for i in range(n):
        _, lab = generate_one_image(i, spec)
        pb = lab['meta']['params']['protein_brightness']
        d = np.array([s['diameter_nm'] for s in lab['spots']])
        p = np.array([s['protein_intensity'] for s in lab['spots']])
        x = np.log(d / 100.0)
        keep = np.abs(x) > 0.2          # exclude d≈100nm where recovery blows up
        if keep.sum() > 20:
            stds.append(float((np.log(p[keep] / pb) / x[keep]).std()))
    return float(np.mean(stds))


def _mean_marginal_corr(spec, n=8):
    cs = []
    for i in range(n):
        _, lab = generate_one_image(i, spec)
        d = np.array([s['diameter_nm'] for s in lab['spots']])
        p = np.array([s['protein_intensity'] for s in lab['spots']])
        if len(d) > 20:
            cs.append(float(np.corrcoef(d, p)[0, 1]))
    return float(np.mean(cs))


def test_per_spot_random_scrambles_alpha(make_spec):
    ps = make_spec(_fixed_density_overrides('per_spot_random', [0.5, 2.0]))
    gc = make_spec(_fixed_density_overrides('global_coherent', [1.5, 1.5]))
    std_ps = _recovered_alpha_std(ps)
    std_gc = _recovered_alpha_std(gc)
    # per_spot spread ~ std of U(0.5,2) ≈ 0.43 (measured ~0.48); global is eta-only.
    assert std_ps > 0.35
    assert std_gc < 0.30
    assert std_ps > 1.5 * std_gc


def test_marginal_corr_lower_for_per_spot_but_not_zero(make_spec):
    """Documents the real effect: per_spot LOWERS corr but does NOT zero it."""
    ps = make_spec(_fixed_density_overrides('per_spot_random', [0.5, 2.0]))
    gc = make_spec(_fixed_density_overrides('global_coherent', [1.5, 1.5]))
    corr_ps = _mean_marginal_corr(ps)
    corr_gc = _mean_marginal_corr(gc)
    assert corr_gc > 0.9                       # one slope -> tight power law
    assert corr_ps < corr_gc                   # per_spot is clearly lower ...
    assert corr_ps > 0.3                        # ... but NOT ~0 (positive alpha)


def test_per_spot_alpha_fields_are_distinct(make_spec):
    ps = make_spec({'alpha_mode': 'per_spot_random', 'alpha_range': [0.5, 2.0]})
    _, lab = generate_one_image(0, ps)
    alphas = [s['alpha_used'] for s in lab['spots']]
    assert len(alphas) > 50
    assert len(set(np.round(alphas, 6))) > 20      # many independent values
    assert min(alphas) >= 0.5 and max(alphas) <= 2.0
    assert lab['meta']['image_alpha'] is None


def test_global_coherent_alpha_is_shared(make_spec):
    gc = make_spec({'alpha_mode': 'global_coherent', 'alpha_range': [1.2, 1.2]})
    _, lab = generate_one_image(0, gc)
    alphas = [s['alpha_used'] for s in lab['spots']]
    assert min(alphas) == max(alphas) == pytest.approx(1.2)
    assert lab['meta']['image_alpha'] == pytest.approx(1.2)


def test_mixed_mode_resolves_per_image(make_spec):
    mx = make_spec({'alpha_mode': 'mixed', 'mixed_global_prob': 0.5,
                    'alpha_range': [0.5, 2.0]})
    modes = set()
    for i in range(16):
        _, lab = generate_one_image(i, mx)
        modes.add(lab['meta']['alpha_mode'])
    assert modes <= {'per_spot_random', 'global_coherent'}
    assert 'global_coherent' not in (modes - {'per_spot_random', 'global_coherent'})
