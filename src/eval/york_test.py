"""Settle whether per-spot weighting CAN beat constant-lambda, using the correct
per-point errors-in-variables fit (York 1968/2004 -- no global lambda).

Three tests:
  (A) synthetic known heteroscedastic noise: correct weighting MUST reduce variance.
  (B) TRUE intensities: clean noise, no model bias -> weighting should help if est ok.
  (C) PREDICTED intensities: real pipeline -> compare to (B) to isolate model bias.

The fits (``deming_slope``, ``york_slope``) come from the canonical
``src.eval.alpha_fit``. Predicted log-space variances are ``exp(logvar)`` used
DIRECTLY (the NLL residual is already in log space; no delta-method conversion).

    python -m src.eval.york_test --config configs/train/hrnet_v1.yaml \
        --ckpt runs/hrnet_v1/best.pt
"""

import argparse
from pathlib import Path

import numpy as np

from src.eval.alpha_fit import deming_slope, york_slope
from src.eval.alpha_recovery import DEFAULT_DATASETS, alpha_from_dir
from src.eval.matching import load_model, matched_pairs

NBOOT = 200


def test_A():
    print("=== TEST A: synthetic known heteroscedastic noise (correct weighting MUST win) ===")
    rng = np.random.default_rng(1)
    true_b = 0.75; n = 4000
    xt = rng.uniform(0, 4, n)                       # true log-lipid
    yt = 1.0 + true_b * xt                          # true log-protein, slope 0.75
    # heteroscedastic: noise grows with a per-point scale s_i (varies 0.02..0.5)
    s = rng.uniform(0.02, 0.5, n)
    sx2 = s ** 2; sy2 = (1.3 * s) ** 2              # y a bit noisier
    x = xt + rng.normal(0, np.sqrt(sx2)); y = yt + rng.normal(0, np.sqrt(sy2))
    ac, aw = [], []
    lam0 = sy2.mean() / sx2.mean()
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        ac.append(deming_slope(x[idx], y[idx], lam0))
        aw.append(york_slope(x[idx], y[idx], sx2[idx], sy2[idx]))
    ac, aw = np.array(ac), np.array(aw)
    print(f"  true slope=0.750")
    print(f"  const-lam: {ac.mean():.4f} +/- {ac.std():.4f}")
    print(f"  york wtls: {aw.mean():.4f} +/- {aw.std():.4f}   "
          f"sd_ratio={aw.std() / ac.std():.2f}")
    print(f"  -> if york sd_ratio < 1 (and mean nearer 0.75), estimator is correct.\n")


def collect(model, cfg, device, val_dir, match_radius):
    tl, tp, pl, pp, plv, ppv = [], [], [], [], [], []
    for gt, det in matched_pairs(model, cfg, device, val_dir, match_radius):
        tl.append(gt['lipid_intensity']); tp.append(gt['protein_intensity'])
        pl.append(det['lipid_intensity']); pp.append(det['protein_intensity'])
        plv.append(det['lipid_intensity_logvar']); ppv.append(det['protein_intensity_logvar'])
    return (np.array(a, np.float64) for a in (tl, tp, pl, pp, plv, ppv))


def test_BC(model, cfg, device, datasets, match_radius):
    print("=== TEST B/C: york WTLS vs const-lambda on TRUE and PRED intensities ===")
    print(f"{'true':>5} | {'TRUE const':>11} {'TRUE york':>11} | "
          f"{'PRED const':>11} {'PRED york':>11}")
    for d in datasets:
        if not Path(d).exists():
            print(f"{alpha_from_dir(d):>5} missing"); continue
        tl, tp, pl, pp, plv, ppv = collect(model, cfg, device, d, match_radius)
        # per-point log-space variances used directly (best available proxy for TRUE)
        vx = np.exp(plv); vy = np.exp(ppv)
        Xt, Yt = np.log(np.clip(tl, 1e-6, None)), np.log(np.clip(tp, 1e-6, None))
        Xp, Yp = np.log(np.clip(pl, 1e-6, None)), np.log(np.clip(pp, 1e-6, None))
        lam0 = vy.mean() / vx.mean()
        a_t_c = 2 * deming_slope(Xt, Yt, lam0); a_t_y = 2 * york_slope(Xt, Yt, vx, vy)
        a_p_c = 2 * deming_slope(Xp, Yp, lam0); a_p_y = 2 * york_slope(Xp, Yp, vx, vy)
        print(f"{alpha_from_dir(d):>5.2f} | {a_t_c:>11.3f} {a_t_y:>11.3f} | "
              f"{a_p_c:>11.3f} {a_p_y:>11.3f}")
    print("  -> TRUE york nearer truth than TRUE const => estimator helps on clean noise.")
    print("  -> if TRUE york helps but PRED york doesn't => model intensity bias defeats it.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/train/hrnet_v1.yaml')
    ap.add_argument('--ckpt', default='runs/hrnet_v1/best.pt')
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--match-radius', type=float, default=4.0)
    ap.add_argument('--skip-model', action='store_true',
                    help='run only the synthetic Test A (no checkpoint needed)')
    args = ap.parse_args()

    test_A()
    if args.skip_model:
        return
    model, cfg, device = load_model(args.config, args.ckpt)
    test_BC(model, cfg, device, args.datasets, args.match_radius)


if __name__ == '__main__':
    main()
