"""Per-image generation + serialization (the heavy, numpy-bearing layer).

Kept OUT of ``generate.py``'s module top so that, in a spawned worker, the BLAS
thread pinning set by the pool initializer runs BEFORE numpy is imported (the same
discipline as ``src.calibration.study``). ``generate.py`` imports this module
lazily, inside the worker, after pinning.

``generate_one_image(index, spec)`` is fully determined by ``(base_seed, index,
config)`` — the seed stream is ``SeedSequence([base_seed, index])`` so images are
reproducible and independent regardless of worker assignment.
"""

import json
from pathlib import Path

import numpy as np

from src.simulator.forward_model import simulate_image
from src.generator.protein_channel import render_protein_per_spot
from src.generator.sampling import sample_image_params


def _resolve_alpha_mode(mode, rng, mixed_global_prob):
    """Resolve the per-image alpha mode; 'mixed' picks one of the two per image."""
    if mode == 'mixed':
        return ('global_coherent' if rng.random() < mixed_global_prob
                else 'per_spot_random')
    if mode not in ('per_spot_random', 'global_coherent'):
        raise ValueError(f"unknown alpha_mode '{mode}'")
    return mode


def generate_one_image(index, spec):
    """Generate one image + label. Returns ``(image, label)``.

    ``image`` is a float32 array of shape (2, H, W): channel 0 = protein,
    channel 1 = lipid (transmitted is not emitted). ``label`` is a JSON-ready dict
    with per-spot ground truth and full per-image metadata.
    """
    cfg = spec['cfg']
    image_size = int(spec['image_size'])
    base_seed = int(spec['base_seed'])
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, int(index)]))

    regimes = spec['regimes']
    regime = regimes[int(rng.integers(len(regimes)))]

    diameters = np.asarray(spec['diameters'], dtype=float)
    probs = np.asarray(spec['probs'], dtype=float)

    params, meta = sample_image_params(rng, spec['ranges'], regime, cfg)

    mode = _resolve_alpha_mode(
        spec['alpha_mode'], rng, float(cfg.get('mixed_global_prob', 0.5)))
    alpha_range = tuple(cfg.get('alpha_range', (0.5, 2.0)))

    if mode == 'per_spot_random':
        # Lipid + ground truth from the simulator; protein synthesized per-spot.
        _, lipid, gt = simulate_image(
            params, diameters, probs, image_size=image_size, rng=rng,
            lipid_only=True)
        protein, _alphas = render_protein_per_spot(
            gt, params, alpha_range, rng, image_size=image_size)
        image_alpha = None
    else:  # global_coherent: one alpha per image -> straight through the simulator
        image_alpha = float(rng.uniform(*alpha_range))
        sim_params = dict(params)
        sim_params['curvature_alpha'] = image_alpha
        protein, lipid, gt = simulate_image(
            sim_params, diameters, probs, image_size=image_size, rng=rng,
            lipid_only=False)
        for g in gt:
            g['alpha_used'] = image_alpha

    for g in gt:
        g['sample_regime_id'] = regime['name']

    # (2, H, W): channel 0 = protein, channel 1 = lipid.
    image = np.stack([protein, lipid], axis=0).astype(np.float32)

    label = {
        'index': int(index),
        'n_spots': int(len(gt)),
        'spots': [{
            'x': float(g['x']),
            'y': float(g['y']),
            'diameter_nm': float(g['diameter_nm']),
            'lipid_intensity': float(g['lipid_intensity']),
            'protein_intensity': float(g['protein_intensity']),
            'alpha_used': float(g['alpha_used']),
            'sample_regime_id': g['sample_regime_id'],
        } for g in gt],
        'meta': {
            'alpha_mode': mode,
            'size_mode': spec['size_mode'],
            'image_alpha': image_alpha,
            'rng_seed': [base_seed, int(index)],
            'config_hash': spec['config_hash'],
            **meta,
            'params': {k: float(v) for k, v in params.items()},
        },
    }
    return image, label


def serialize_image(output_dir, index, image, label):
    """Write ``images/img_<i>.npy`` + ``labels/img_<i>.json``; return a record."""
    out = Path(output_dir)
    img_path = out / 'images' / f'img_{int(index):06d}.npy'
    lbl_path = out / 'labels' / f'img_{int(index):06d}.json'
    np.save(img_path, image)
    lbl_path.write_text(json.dumps(label, indent=2))
    return {
        'index': int(index),
        'n_spots': int(label['n_spots']),
        'image': str(img_path),
        'label': str(lbl_path),
    }
