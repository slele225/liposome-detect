"""Fixed-alpha recovery with OLS vs errors-in-variables (Deming) fits.

Both axes are noisy (lipid: PMT noise; protein: PMT + eta heterogeneity), so OLS
attenuates the slope. Deming accounts for noise in both; for a_PRED we set lambda
from the model's mean predicted per-spot log-space variances. The fits come from the
canonical ``src.eval.alpha_fit`` (single source of truth).

The NLL residual is in LOG space, so ``exp(logvar)`` is ALREADY the log-residual
variance — used directly as the per-axis variance with NO delta-method /intensity^2
conversion (see src/eval/alpha_fit.py).

    python -m src.eval.alpha_recovery --config configs/train/hrnet_v1.yaml \
        --ckpt runs/hrnet_v1/best.pt \
        --datasets datasets/alpha_0p50 datasets/alpha_1p00 \
                   datasets/alpha_1p50 datasets/alpha_2p00
"""

import argparse
from pathlib import Path

import numpy as np

from src.eval.alpha_fit import deming_slope, ols_slope
from src.eval.matching import load_model, matched_pairs

DEFAULT_DATASETS = ['datasets/alpha_0p50', 'datasets/alpha_1p00',
                    'datasets/alpha_1p50', 'datasets/alpha_2p00']


def alpha_from_dir(d):
    """``datasets/alpha_0p50`` -> 0.50 (the true alpha encoded in the dir name)."""
    tok = Path(d).name.split('_')[-1].replace('p', '.')
    try:
        return float(tok)
    except ValueError:
        return float('nan')


def recover(model, cfg, device, val_dir, match_radius):
    tl, tp, pl, pp, plv, ppv = [], [], [], [], [], []
    for gt, det in matched_pairs(model, cfg, device, val_dir, match_radius):
        tl.append(gt['lipid_intensity']); tp.append(gt['protein_intensity'])
        pl.append(det['lipid_intensity']); pp.append(det['protein_intensity'])
        plv.append(det['lipid_intensity_logvar']); ppv.append(det['protein_intensity_logvar'])
    tl, tp, pl, pp, plv, ppv = (np.array(a, np.float64)
                                for a in (tl, tp, pl, pp, plv, ppv))
    Lt, Pt = np.log(np.clip(tl, 1e-6, None)), np.log(np.clip(tp, 1e-6, None))
    Lp, Pp = np.log(np.clip(pl, 1e-6, None)), np.log(np.clip(pp, 1e-6, None))
    a_true_ols = 2 * ols_slope(Lt, Pt)
    a_true_tls = 2 * deming_slope(Lt, Pt, 1.0)          # total least squares
    a_true_d2 = 2 * deming_slope(Lt, Pt, 2.0)           # y noisier (eta)
    # PRED: lambda from the model's mean predicted log-space variances (used DIRECTLY).
    vx = np.exp(plv); vy = np.exp(ppv)
    lam_pred = vy.mean() / vx.mean() if vx.mean() > 0 else 1.0
    a_pred_ols = 2 * ols_slope(Lp, Pp)
    a_pred_dem = 2 * deming_slope(Lp, Pp, lam_pred)
    return a_true_ols, a_true_tls, a_true_d2, a_pred_ols, a_pred_dem, lam_pred, len(tl)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/train/hrnet_v1.yaml')
    ap.add_argument('--ckpt', default='runs/hrnet_v1/best.pt')
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--match-radius', type=float, default=4.0)
    args = ap.parse_args()

    model, cfg, device = load_model(args.config, args.ckpt)
    print(f"{'true':>5} | {'TRUE_ols':>8} {'TRUE_tls':>8} {'TRUE_dem2':>9} | "
          f"{'PRED_ols':>8} {'PRED_dem':>8} {'lam_p':>6} | {'n':>7}")
    for d in args.datasets:
        if not Path(d).exists():
            print(f"{alpha_from_dir(d):>5} missing"); continue
        r = recover(model, cfg, device, d, args.match_radius)
        print(f"{alpha_from_dir(d):>5.2f} | {r[0]:>8.3f} {r[1]:>8.3f} {r[2]:>9.3f} | "
              f"{r[3]:>8.3f} {r[4]:>8.3f} {r[5]:>6.2f} | {r[6]:>7}")


if __name__ == '__main__':
    main()
