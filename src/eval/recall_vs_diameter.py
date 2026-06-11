"""Recall + protein-intensity error vs liposome diameter, from a trained ckpt.

Gates the full run: is the recall uniform, or does it crater on small spots (which
would bias the recovered sorting curve)? Reports per-diameter-bin recall and median
|log protein error| over matched spots.

    python -m src.eval.recall_vs_diameter --config configs/train/hrnet_diagnostic.yaml \
        --ckpt runs/hrnet_diagnostic/checkpoint.pt --val datasets/diag_val
"""

import argparse

import numpy as np

from src.eval.matching import (
    decode_image_array,
    greedy_match,
    iter_images,
    load_model,
)

# diameter bins (nm): heavy emphasis on the small tail
BIN_EDGES = [40, 55, 70, 90, 120, 160, 220, 300]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/train/hrnet_diagnostic.yaml')
    ap.add_argument('--ckpt', default='runs/hrnet_diagnostic/checkpoint.pt')
    ap.add_argument('--val', default='datasets/diag_val')
    ap.add_argument('--match-radius', type=float, default=4.0)
    args = ap.parse_args()

    model, cfg, device = load_model(args.config, args.ckpt)
    nb = len(BIN_EDGES) - 1
    n_gt = np.zeros(nb)
    n_hit = np.zeros(nb)
    log_pro_err = [[] for _ in range(nb)]

    for arr, spots in iter_images(args.val):
        if not spots:
            continue
        gt_xy = np.array([[s['x'], s['y']] for s in spots], np.float32)
        gt_d = np.array([s['diameter_nm'] for s in spots], np.float32)
        gt_pro = np.array([s['protein_intensity'] for s in spots], np.float32)
        dets = decode_image_array(model, cfg, device, arr)
        match = greedy_match(gt_xy, dets, args.match_radius)
        for i in range(len(spots)):
            b = np.digitize(gt_d[i], BIN_EDGES) - 1
            if b < 0 or b >= nb:
                continue
            n_gt[b] += 1
            j = match[i]
            if j >= 0:
                n_hit[b] += 1
                pt, pp = gt_pro[i], dets[j]['protein_intensity']
                if pt > 0 and pp > 0:
                    log_pro_err[b].append(abs(np.log(pp) - np.log(pt)))

    print(f"{'diam bin (nm)':>16} {'n_gt':>7} {'recall':>7} {'med|log protein err|':>22}")
    for b in range(nb):
        rec = n_hit[b] / n_gt[b] if n_gt[b] else float('nan')
        med = float(np.median(log_pro_err[b])) if log_pro_err[b] else float('nan')
        print(f"{BIN_EDGES[b]:>7}-{BIN_EDGES[b + 1]:<7} {int(n_gt[b]):>7} "
              f"{rec:>7.3f} {med:>22.3f}")
    overall = n_hit.sum() / n_gt.sum() if n_gt.sum() else float('nan')
    print(f"\noverall recall (binned spots): {overall:.3f}")


if __name__ == '__main__':
    main()
