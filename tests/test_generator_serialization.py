"""Ground-truth serialization round-trips and matches simulator output counts;
generation is deterministic from (config + base_seed + index)."""

import json
from pathlib import Path

import numpy as np

from src.generator.core import generate_one_image, serialize_image

_GT_FIELDS = ['x', 'y', 'diameter_nm', 'lipid_intensity', 'protein_intensity',
              'alpha_used', 'sample_regime_id']
_META_FIELDS = ['alpha_mode', 'size_mode', 'image_alpha', 'rng_seed',
                'config_hash', 'noise_scale', 'noise_split_r', 'sample_regime_id',
                'params']


def _prep(spec, tmp_path):
    spec = dict(spec)
    spec['output_dir'] = str(tmp_path / 'ds')
    (tmp_path / 'ds' / 'images').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'ds' / 'labels').mkdir(parents=True, exist_ok=True)
    return spec


def test_image_shape_and_channel_order(make_spec, tmp_path):
    spec = _prep(make_spec(), tmp_path)
    image, label = generate_one_image(0, spec)
    assert image.shape == (2, spec['image_size'], spec['image_size'])
    assert image.dtype == np.float32      # channel 0 = protein, 1 = lipid


def test_serialize_roundtrip_and_counts(make_spec, tmp_path):
    spec = _prep(make_spec(), tmp_path)
    image, label = generate_one_image(0, spec)
    rec = serialize_image(spec['output_dir'], 0, image, label)

    arr = np.load(rec['image'])
    lbl = json.loads(Path(rec['label']).read_text())

    # counts agree across: simulator GT -> label -> record -> serialized file.
    assert label['n_spots'] == len(label['spots'])
    assert lbl['n_spots'] == len(lbl['spots']) == label['n_spots'] == rec['n_spots']
    assert lbl['n_spots'] > 0
    assert np.array_equal(arr, image)

    for s in lbl['spots']:
        for k in _GT_FIELDS:
            assert k in s
    for k in _META_FIELDS:
        assert k in lbl['meta']
    # params carry the simulator-exact keys.
    for k in ['spot_density', 'lipid_brightness', 'gain', 'enf', 'psf_sigma_x',
              'protein_brightness', 'offset_protein', 'read_noise_var_protein']:
        assert k in lbl['meta']['params']


def test_deterministic_same_seed_same_index(make_spec):
    spec = make_spec()
    img1, lab1 = generate_one_image(2, spec)
    img2, lab2 = generate_one_image(2, spec)
    assert np.array_equal(img1, img2)
    assert lab1 == lab2


def test_different_index_differs(make_spec):
    spec = make_spec()
    img0, _ = generate_one_image(0, spec)
    img1, _ = generate_one_image(1, spec)
    assert not np.array_equal(img0, img1)


def test_centroids_within_bounds(make_spec):
    spec = make_spec()
    _, lab = generate_one_image(0, spec)
    size = spec['image_size']
    for s in lab['spots']:
        assert 0 <= s['x'] <= size
        assert 0 <= s['y'] <= size
        assert s['diameter_nm'] > 0
        assert s['lipid_intensity'] >= 0
        assert s['protein_intensity'] >= 0
