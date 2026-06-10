"""Decode -> the benchmark OUTPUT SCHEMA, and writer/loader round-trip.

The schema is a hard contract (docs/decisions/2026-06-10_benchmark-design.md):
exact keys, all floats, no diameter.
"""

import pytest
import torch

from src.models.decode import (
    SCHEMA_KEYS,
    decode_image,
    load_detections,
    validate_detection,
    write_detections,
)


def _fake_outputs():
    """One-image head outputs (h=w=8, stride 4) with two clear peaks."""
    h = w = 8
    heatmap = torch.zeros(1, h, w)
    heatmap[0, 2, 3] = 0.9
    heatmap[0, 5, 6] = 0.5
    offset = torch.zeros(2, h, w)
    offset[0, 2, 3] = 0.25
    offset[1, 2, 3] = 0.5
    lipid = torch.zeros(2, h, w)
    lipid[0, 2, 3] = 4000.0
    lipid[1, 2, 3] = -1.0
    lipid[0, 5, 6] = 800.0
    protein = torch.zeros(2, h, w)
    protein[0, 2, 3] = 2500.0
    protein[1, 2, 3] = 0.5
    protein[0, 5, 6] = 600.0
    return {'heatmap': heatmap, 'offset': offset, 'lipid': lipid, 'protein': protein}


def test_decode_emits_exact_schema():
    dets = decode_image(_fake_outputs(), out_stride=4,
                        score_threshold=0.3, nms_kernel=3)
    assert len(dets) == 2
    # sorted by score desc
    assert dets[0]['detection_score'] > dets[1]['detection_score']
    for d in dets:
        assert tuple(d.keys()) == SCHEMA_KEYS         # exact keys, in order
        assert all(isinstance(d[k], float) for k in SCHEMA_KEYS)
    # subpixel refine: x=(3+0.25)*4=13, y=(2+0.5)*4=10
    top = dets[0]
    assert top['x'] == pytest.approx(13.0)
    assert top['y'] == pytest.approx(10.0)
    assert top['lipid_intensity'] == pytest.approx(4000.0)
    assert top['lipid_intensity_logvar'] == pytest.approx(-1.0)
    assert top['protein_intensity'] == pytest.approx(2500.0)


def test_score_threshold_filters():
    dets = decode_image(_fake_outputs(), out_stride=4,
                        score_threshold=0.7, nms_kernel=3)
    assert len(dets) == 1
    assert dets[0]['detection_score'] == pytest.approx(0.9)


def test_writer_loader_roundtrip(tmp_path):
    dets = decode_image(_fake_outputs(), out_stride=4, score_threshold=0.3)
    path = tmp_path / 'dets.json'
    write_detections(path, dets)
    loaded = load_detections(path)
    assert loaded == dets


def test_validate_rejects_bad_schema():
    bad = {'x': 1.0, 'y': 2.0}                      # missing keys
    with pytest.raises(ValueError):
        validate_detection(bad)
    extra = {k: 0.0 for k in SCHEMA_KEYS}
    extra['diameter_nm'] = 100.0                    # schema must NOT carry diameter
    with pytest.raises(ValueError):
        validate_detection(extra)
