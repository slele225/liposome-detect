"""Shared eval plumbing: load a trained detector + greedy-match GT to decode output.

The ad-hoc analysis scripts in this package all (1) build the model from a training
config + checkpoint and (2) greedily match ground-truth spots to decoded detections
within a radius. That boilerplate lives here so the scripts stay focused on their
specific analysis and there is one decode/match path.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.decode import decode_image
from src.train.train import build_model


def load_model(config_path, ckpt_path, device=None):
    """Build the model from ``config_path`` and load weights from ``ckpt_path``.

    Returns ``(model, cfg, device)``. ``ckpt_path`` is a checkpoint dict with a
    ``"model"`` state-dict (the format ``train.py`` writes for ``best.pt`` /
    ``checkpoint.pt``).
    """
    cfg = yaml.safe_load(open(config_path))
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    return model, cfg, device


def decode_image_array(model, cfg, device, arr):
    """Normalize ``arr`` ([C,H,W]) per ``cfg['data']`` and decode -> detection list."""
    dec = cfg['decode']
    nm = np.array(cfg['data']['norm_mean'], np.float32)
    ns = np.array(cfg['data']['norm_std'], np.float32)
    x = torch.from_numpy((arr - nm[:, None, None]) / ns[:, None, None])[None].to(device)
    with torch.no_grad():
        out = model(x)
    out = {k: v[0] for k, v in out.items()}
    return decode_image(out, model.out_stride,
                        score_threshold=dec['score_threshold'],
                        nms_kernel=dec['nms_kernel'])


def iter_images(val_dir):
    """Yield ``(image_array, spots)`` for every image under ``val_dir``."""
    val_dir = Path(val_dir)
    for ip in sorted((val_dir / 'images').glob('*.npy')):
        arr = np.load(ip).astype(np.float32)
        spots = json.load(open(val_dir / 'labels' / (ip.stem + '.json')))['spots']
        yield arr, spots


def greedy_match(gt_xy, dets, match_radius):
    """Greedy nearest-neighbour match GT -> detections within ``match_radius``.

    Returns an int array ``match`` of length ``len(gt_xy)``: ``match[i]`` is the
    index into ``dets`` matched to GT spot ``i``, or ``-1`` if unmatched. Each
    detection is used at most once.
    """
    n_gt = len(gt_xy)
    match = np.full(n_gt, -1, dtype=int)
    if not dets:
        return match
    dxy = np.array([[d['x'], d['y']] for d in dets], np.float32)
    used = np.zeros(len(dets), bool)
    for i in range(n_gt):
        dd = np.hypot(dxy[:, 0] - gt_xy[i, 0], dxy[:, 1] - gt_xy[i, 1])
        dd[used] = 1e9
        j = int(dd.argmin())
        if dd[j] <= match_radius:
            used[j] = True
            match[i] = j
    return match


def matched_pairs(model, cfg, device, val_dir, match_radius=4.0):
    """Yield ``(gt_spot_dict, det_dict)`` for every GT spot matched within radius."""
    for arr, spots in iter_images(val_dir):
        if not spots:
            continue
        gt_xy = np.array([[s['x'], s['y']] for s in spots], np.float32)
        dets = decode_image_array(model, cfg, device, arr)
        match = greedy_match(gt_xy, dets, match_radius)
        for i, j in enumerate(match):
            if j >= 0:
                yield spots[i], dets[j]
