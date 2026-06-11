"""Within-size-bin detection bias + slope checks (true vs predicted, with/without
the smallest bin).

A ratio > 1 of detected-spot to missed-spot median protein in a diameter bin means
the detector preferentially catches the BRIGHT spots in that bin = a detection bias
that would steepen the recovered sorting curve. Slopes use the canonical OLS fit
from ``src.eval.alpha_fit`` (reported for comparison; production alpha uses Deming).

    python -m src.eval.detection_bias --config configs/train/hrnet_diagnostic.yaml \
        --ckpt runs/hrnet_diagnostic/checkpoint.pt --val datasets/diag_val
"""

import argparse

import numpy as np

from src.eval.alpha_fit import ols_slope
from src.eval.matching import (
    decode_image_array,
    greedy_match,
    iter_images,
    load_model,
)

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
    det_pro = [[] for _ in range(nb)]
    mis_pro = [[] for _ in range(nb)]
    all_lip, all_pro, all_d, all_plip, all_ppro = [], [], [], [], []

    for arr, spots in iter_images(args.val):
        if not spots:
            continue
        gxy = np.array([[s['x'], s['y']] for s in spots], np.float32)
        gd = np.array([s['diameter_nm'] for s in spots], np.float32)
        glip = np.array([s['lipid_intensity'] for s in spots], np.float32)
        gpro = np.array([s['protein_intensity'] for s in spots], np.float32)
        dets = decode_image_array(model, cfg, device, arr)
        match = greedy_match(gxy, dets, args.match_radius)
        for i in range(len(spots)):
            b = np.digitize(gd[i], BIN_EDGES) - 1
            if b < 0 or b >= nb:
                continue
            j = match[i]
            if j >= 0:
                all_lip.append(glip[i]); all_pro.append(gpro[i]); all_d.append(gd[i])
                all_plip.append(dets[j]['lipid_intensity'])
                all_ppro.append(dets[j]['protein_intensity'])
            (det_pro if j >= 0 else mis_pro)[b].append(gpro[i])

    print(f"{'bin(nm)':>12} {'det_med':>9} {'mis_med':>9} {'ratio':>7}  "
          f"(>1 = catching bright ones = BIAS)")
    for b in range(nb):
        dm = np.median(det_pro[b]) if det_pro[b] else np.nan
        mm = np.median(mis_pro[b]) if mis_pro[b] else np.nan
        r = dm / mm if (mm and not np.isnan(mm)) else np.nan
        print(f"{BIN_EDGES[b]:>5}-{BIN_EDGES[b + 1]:<5} {dm:>9.0f} {mm:>9.0f} {r:>7.3f}")

    L, P, D = (np.log(np.array(all_lip)), np.log(np.array(all_pro)),
               np.array(all_d))
    PL = np.log(np.clip(np.array(all_plip), 1e-6, None))
    PP = np.log(np.clip(np.array(all_ppro), 1e-6, None))

    def slope(lx, ly, mask=None):
        if mask is not None:
            lx, ly = lx[mask], ly[mask]
        return ols_slope(lx, ly)

    s_true_all = slope(L, P)
    s_true_no40 = slope(L, P, D >= 55)
    s_pred_all = slope(PL, PP)
    s_pred_no40 = slope(PL, PP, D >= 55)
    print(f"\n{'':28} slope   alpha(=2*slope)")
    print(f"  TRUE intens, all bins:    {s_true_all:6.3f}   {2 * s_true_all:6.3f}")
    print(f"  TRUE intens, >=55nm:      {s_true_no40:6.3f}   {2 * s_true_no40:6.3f}")
    print(f"  PRED intens, all bins:    {s_pred_all:6.3f}   {2 * s_pred_all:6.3f}")
    print(f"  PRED intens, >=55nm:      {s_pred_no40:6.3f}   {2 * s_pred_no40:6.3f}")


if __name__ == '__main__':
    main()
