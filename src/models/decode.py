"""Inference decode -> the benchmark output schema (a hard contract).

Decode: heatmap -> local-max peak extraction (maxpool NMS + score threshold +
optional cap) -> refine each peak by its offset -> sample the four intensity-map
values at the refined cell. Emit, per detected spot, EXACTLY:

    {
      "x": float, "y": float,                 # subpixel centroid (px)
      "detection_score": float,               # heatmap peak prob
      "lipid_intensity": float,               # predicted mean total flux
      "lipid_intensity_logvar": float,        # predicted log-variance
      "protein_intensity": float,
      "protein_intensity_logvar": float
    }

This is the format EVERY benchmarked method's adapter must also produce (see
docs/decisions/2026-06-10_benchmark-design.md). The network does NOT output
diameter — size is derived downstream from lipid intensity (lipid ∝ d²), and
alpha from the log-protein vs log-lipid slope; the schema intentionally omits it.
``write_detections`` / ``load_detections`` round-trip a per-image list to JSON.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

#: Exact, ordered keys of one detection (the benchmark contract).
SCHEMA_KEYS = (
    'x', 'y', 'detection_score',
    'lipid_intensity', 'lipid_intensity_logvar',
    'protein_intensity', 'protein_intensity_logvar',
)


def _peak_mask(heatmap, nms_kernel, score_threshold):
    """Local maxima above threshold (maxpool-NMS). ``heatmap`` is [H, W]."""
    pad = nms_kernel // 2
    pooled = F.max_pool2d(heatmap[None, None], nms_kernel, stride=1, padding=pad)[0, 0]
    return (heatmap >= pooled) & (heatmap > score_threshold)


@torch.no_grad()
def decode_image(outputs, out_stride, score_threshold=0.3, nms_kernel=3,
                 max_detections=None):
    """Decode ONE image's head outputs into a list of schema dicts.

    Args:
        outputs: dict of per-image tensors (no batch dim): 'heatmap' [1,h,w],
            'offset' [2,h,w], 'lipid' [2,h,w], 'protein' [2,h,w] (lipid/protein
            channel 0 is already the positive mean flux).
        out_stride: feature-map -> full-resolution scale.
        score_threshold / nms_kernel / max_detections: decode params.

    Returns a list of dicts with exactly ``SCHEMA_KEYS``, sorted by score desc.
    """
    heatmap = outputs['heatmap'][0]
    h, w = heatmap.shape
    full_h, full_w = h * out_stride, w * out_stride

    mask = _peak_mask(heatmap, nms_kernel, score_threshold)
    ys, xs = mask.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return []
    scores = heatmap[ys, xs]
    order = torch.argsort(scores, descending=True)
    if max_detections is not None:
        order = order[:max_detections]
    ys, xs, scores = ys[order], xs[order], scores[order]

    offset = outputs['offset']
    lipid = outputs['lipid']
    protein = outputs['protein']

    dets = []
    for yy, xx, sc in zip(ys.tolist(), xs.tolist(), scores.tolist()):
        ox = float(offset[0, yy, xx])
        oy = float(offset[1, yy, xx])
        x_full = min(max((xx + ox) * out_stride, 0.0), full_w - 1.0)
        y_full = min(max((yy + oy) * out_stride, 0.0), full_h - 1.0)
        dets.append({
            'x': x_full, 'y': y_full,
            'detection_score': float(sc),
            'lipid_intensity': float(lipid[0, yy, xx]),
            'lipid_intensity_logvar': float(lipid[1, yy, xx]),
            'protein_intensity': float(protein[0, yy, xx]),
            'protein_intensity_logvar': float(protein[1, yy, xx]),
        })
    return dets


@torch.no_grad()
def decode_batch(outputs, out_stride, **kw):
    """Decode a batched head-output dict -> list (per image) of detection lists."""
    b = outputs['heatmap'].shape[0]
    return [decode_image({k: v[i] for k, v in outputs.items()}, out_stride, **kw)
            for i in range(b)]


def validate_detection(det):
    """Raise if ``det`` does not have exactly the schema keys (all floats)."""
    if set(det) != set(SCHEMA_KEYS):
        raise ValueError(f"detection keys {sorted(det)} != schema {sorted(SCHEMA_KEYS)}")
    for k in SCHEMA_KEYS:
        if not isinstance(det[k], float):
            raise TypeError(f"detection['{k}'] must be float, got {type(det[k])}")


def write_detections(path, detections, validate=True):
    """Serialize a per-image list of detection dicts to JSON."""
    if validate:
        for d in detections:
            validate_detection(d)
    Path(path).write_text(json.dumps({'detections': detections}, indent=2))


def load_detections(path):
    """Load a per-image detection list written by ``write_detections``."""
    data = json.loads(Path(path).read_text())
    return data['detections']
