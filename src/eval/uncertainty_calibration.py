"""Calibration: predicted log-space variance exp(logvar) vs actual log-error^2.

The NLL residual is ``r = log(pred+eps) - log(true+eps)`` and ``sigma2 = exp(logvar)``
is the variance OF THAT LOG RESIDUAL -- so NO delta-method conversion. Compare
exp(logvar) directly to r^2. Calibrated => ratio ~1 and actual_mse rises across
deciles of predicted variance.

    python -m src.eval.uncertainty_calibration --config configs/train/hrnet_v1.yaml \
        --ckpt runs/hrnet_v1/best.pt --eps-lipid 80.15 --eps-protein 62.47
"""

import argparse
from pathlib import Path

import numpy as np

from src.eval.alpha_recovery import DEFAULT_DATASETS
from src.eval.matching import load_model, matched_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/train/hrnet_v1.yaml')
    ap.add_argument('--ckpt', default='runs/hrnet_v1/best.pt')
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--match-radius', type=float, default=4.0)
    # eps MUST match the loss eps (same log-residual definition).
    ap.add_argument('--eps-lipid', type=float, default=80.15)
    ap.add_argument('--eps-protein', type=float, default=62.47)
    args = ap.parse_args()

    model, cfg, device = load_model(args.config, args.ckpt)
    pv_l, err_l, pv_p, err_p = [], [], [], []
    for d in args.datasets:
        if not Path(d).exists():
            continue
        for gt, det in matched_pairs(model, cfg, device, d, args.match_radius):
            rl = (np.log(det['lipid_intensity'] + args.eps_lipid)
                  - np.log(gt['lipid_intensity'] + args.eps_lipid))
            rp = (np.log(det['protein_intensity'] + args.eps_protein)
                  - np.log(gt['protein_intensity'] + args.eps_protein))
            pv_l.append(np.exp(det['lipid_intensity_logvar'])); err_l.append(rl * rl)
            pv_p.append(np.exp(det['protein_intensity_logvar'])); err_p.append(rp * rp)

    for name, pv, err in [('LIPID', pv_l, err_l), ('PROTEIN', pv_p, err_p)]:
        pv, err = np.array(pv), np.array(err)
        order = np.argsort(pv); pv, err = pv[order], err[order]
        print(f"\n{name}: predicted var exp(logvar) vs actual log-error^2 "
              f"(deciles by predicted var)")
        print(f"{'decile':>7} {'pred_var':>10} {'actual_mse':>11} {'ratio':>7}")
        for q in range(10):
            lo, hi = q * len(pv) // 10, (q + 1) * len(pv) // 10
            pm, am = pv[lo:hi].mean(), err[lo:hi].mean()
            print(f"{q + 1:>7} {pm:>10.4f} {am:>11.4f} "
                  f"{am / pm if pm > 0 else float('nan'):>7.2f}")
        print(f"  overall: mean_pred_var={pv.mean():.4f}  "
              f"mean_actual_mse={err.mean():.4f}  ratio={err.mean() / pv.mean():.2f}")


if __name__ == '__main__':
    main()
