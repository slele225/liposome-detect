"""Validation metrics: matched-detection F1 + per-channel intensity log-error.

Method-neutral matching (same as the benchmark contract): greedy nearest-neighbor
within a FIXED match radius, predictions consumed in score order, each GT matched
at most once.
"""

import numpy as np


def match_detections(preds, gts, radius):
    """Greedy score-ordered matching within ``radius`` (full-res px).

    ``preds``: list of dicts with 'x','y','detection_score'. ``gts``: list with
    'x','y'. Returns ``(matches, tp, fp, fn)`` where ``matches`` is a list of
    ``(pred_idx, gt_idx)``.
    """
    if not preds or not gts:
        return [], 0, len(preds), len(gts)
    order = sorted(range(len(preds)), key=lambda i: -preds[i]['detection_score'])
    gx = np.array([g['x'] for g in gts], dtype=np.float64)
    gy = np.array([g['y'] for g in gts], dtype=np.float64)
    used = np.zeros(len(gts), dtype=bool)
    r2 = float(radius) ** 2
    matches = []
    for pi in order:
        d2 = (gx - preds[pi]['x']) ** 2 + (gy - preds[pi]['y']) ** 2
        d2[used] = np.inf
        j = int(np.argmin(d2))
        if d2[j] <= r2:
            used[j] = True
            matches.append((pi, j))
    tp = len(matches)
    return matches, tp, len(preds) - tp, len(gts) - tp


def precision_recall_f1(tp, fp, fn):
    """P, R, F1 from aggregate counts (0 when undefined)."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def localization_error(preds, gts, matches):
    """Mean Euclidean centroid error (px) over matched pairs (nan if none)."""
    if not matches:
        return float('nan')
    d = [np.hypot(preds[pi]['x'] - gts[gj]['x'], preds[pi]['y'] - gts[gj]['y'])
         for pi, gj in matches]
    return float(np.mean(d))


def intensity_log_error(preds, gts, matches, eps_lipid=10.0, eps_protein=10.0):
    """Mean |log-residual| of lipid and protein flux over matched pairs."""
    if not matches:
        return float('nan'), float('nan')
    lip, pro = [], []
    for pi, gj in matches:
        lip.append(abs(np.log(preds[pi]['lipid_intensity'] + eps_lipid)
                       - np.log(gts[gj]['lipid_intensity'] + eps_lipid)))
        pro.append(abs(np.log(preds[pi]['protein_intensity'] + eps_protein)
                       - np.log(gts[gj]['protein_intensity'] + eps_protein)))
    return float(np.mean(lip)), float(np.mean(pro))
