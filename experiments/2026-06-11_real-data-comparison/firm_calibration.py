"""E — firm up the recovered->true calibration curve (feeds real_alpha.py).

The default ``CalibrationCurve`` is seeded with only 4 points (alpha 0.5/1.0/1.5/
2.0). This rebuilds it on a denser, error-barred grid so the curve real_alpha.py
inverts is a real, well-sampled deliverable rather than a 4-point seed.

For each true alpha in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2}:
  1. generate a fixed-alpha synthetic test set (global_coherent single-alpha,
     emphasis sizing) from config_snapshot/alpha_template.yaml — CPU, all workers;
  2. run the detector, greedily match GT, and recover alpha from PREDICTED
     intensities with the canonical EIV estimator (``recover_alpha``: constant-
     lambda Deming, lambda from mean exp(logvar) per axis — used DIRECTLY, no
     per-spot weighting, no /intensity**2);
  3. bootstrap over images for an error bar on the recovered alpha.

Then fit a CalibrationCurve(recovered -> true) and SAVE it to
``calibration_curve.json`` (loaded by real_alpha.py), and plot recovered-vs-true
with the y=x line + error bars.

    uv run python experiments/2026-06-11_real-data-comparison/firm_calibration.py \
        --n-workers 32
"""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from _common import EXP_DIR, REPO_ROOT, add_model_args
from src.eval.alpha_fit import CalibrationCurve, recover_alpha
from src.eval.matching import (decode_image_array, greedy_match, iter_images,
                               load_model)

ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]
TEMPLATE = EXP_DIR / 'config_snapshot' / 'alpha_template.yaml'
CURVE_OUT = EXP_DIR / 'calibration_curve.json'
PLOT_OUT = EXP_DIR / 'calibration_curve.png'
TABLE_OUT = EXP_DIR / 'calibration_curve.csv'


def _tag(alpha):
    return f"{alpha:.2f}".replace('.', 'p')          # 0.5 -> 0p50, 2.2 -> 2p20


def _dataset_dir(alpha):
    return REPO_ROOT / 'datasets' / f"real_cmp_alpha_{_tag(alpha)}"


def generate_all(n_workers):
    """Generate the 8 fixed-alpha synthetic sets in-process (CPU, all workers)."""
    from src.generator.generate import run_generation
    template = yaml.safe_load(open(TEMPLATE))
    base_seed = int(template.get('base_seed', 11000))
    for i, a in enumerate(ALPHAS):
        cfg = dict(template)
        cfg['name'] = f"real_cmp_alpha_{_tag(a)}"
        cfg['alpha_range'] = [float(a), float(a)]
        cfg['base_seed'] = base_seed + i * 1000        # distinct stream per alpha
        print(f"\n[firm_calibration] generating true alpha={a} -> {cfg['name']}")
        run_generation(cfg, str(TEMPLATE), n_workers=n_workers)


def _per_image_pred(model, cfg, device, val_dir, match_radius):
    """List (per image) of (Llip, Lpro, vlip, vpro) for GT-matched predicted spots."""
    out = []
    for arr, spots in iter_images(val_dir):
        if not spots:
            continue
        gt_xy = np.array([[s['x'], s['y']] for s in spots], np.float32)
        dets = decode_image_array(model, cfg, device, arr)
        match = greedy_match(gt_xy, dets, match_radius)
        lip, pro, vl, vp = [], [], [], []
        for j in match:
            if j >= 0:
                d = dets[j]
                lip.append(d['lipid_intensity']); pro.append(d['protein_intensity'])
                vl.append(d['lipid_intensity_logvar'])
                vp.append(d['protein_intensity_logvar'])
        if lip:
            lip = np.array(lip, np.float64); pro = np.array(pro, np.float64)
            out.append((np.log(np.clip(lip, 1e-6, None)),
                        np.log(np.clip(pro, 1e-6, None)),
                        np.exp(np.array(vl, np.float64)),
                        np.exp(np.array(vp, np.float64))))
    return out


def _recover_pooled(groups, idx):
    """recover_alpha (EIV) over the pooled spots of image-groups ``idx``."""
    Ll = np.concatenate([groups[i][0] for i in idx])
    Lp = np.concatenate([groups[i][1] for i in idx])
    vl = np.concatenate([groups[i][2] for i in idx])
    vp = np.concatenate([groups[i][3] for i in idx])
    # x = log lipid, y = log protein; variances used directly (log space).
    return recover_alpha(Ll, Lp, var_x=vl, var_y=vp)


def recover_one(model, cfg, device, alpha, match_radius, n_boot, rng):
    val_dir = _dataset_dir(alpha)
    groups = _per_image_pred(model, cfg, device, str(val_dir), match_radius)
    n_img = len(groups)
    n_spots = int(sum(len(g[0]) for g in groups))
    if n_img == 0:
        return dict(true=alpha, recovered=float('nan'), lo=float('nan'),
                    hi=float('nan'), n_img=0, n_spots=0)
    point = _recover_pooled(groups, range(n_img))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, size=n_img)
        try:
            boots.append(_recover_pooled(groups, idx))
        except Exception:
            pass
    boots = np.array(boots, np.float64)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
        if boots.size else (float('nan'), float('nan'))
    return dict(true=float(alpha), recovered=float(point), lo=lo, hi=hi,
                n_img=n_img, n_spots=n_spots)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_model_args(ap)
    ap.add_argument('--n-workers', type=int, default=None,
                    help='generation worker processes (CPU).')
    ap.add_argument('--match-radius', type=float, default=4.0)
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-generate', action='store_true',
                    help='reuse already-generated datasets/real_cmp_alpha_*.')
    ap.add_argument('--curve-kind', choices=['interp', 'linear'], default='interp')
    args = ap.parse_args()

    if not args.skip_generate:
        generate_all(args.n_workers)

    model, cfg, device = load_model(args.config, args.ckpt)
    rng = np.random.default_rng(args.seed)

    rows = []
    print(f"\n{'true':>5} | {'recovered':>9} {'lo95':>7} {'hi95':>7} | "
          f"{'n_img':>6} {'n_spots':>8}")
    for a in ALPHAS:
        r = recover_one(model, cfg, device, a, args.match_radius, args.n_boot, rng)
        rows.append(r)
        print(f"{r['true']:>5.2f} | {r['recovered']:>9.3f} {r['lo']:>7.3f} "
              f"{r['hi']:>7.3f} | {r['n_img']:>6} {r['n_spots']:>8}")

    good = [r for r in rows if np.isfinite(r['recovered'])]
    rec = [r['recovered'] for r in good]
    tru = [r['true'] for r in good]
    curve = CalibrationCurve(rec, tru, kind=args.curve_kind)
    curve.save(CURVE_OUT)
    print(f"\n[firm_calibration] saved calibration curve -> {CURVE_OUT}")

    # CSV table.
    lines = ["true,recovered,lo95,hi95,n_img,n_spots"]
    for r in rows:
        lines.append(f"{r['true']},{r['recovered']},{r['lo']},{r['hi']},"
                     f"{r['n_img']},{r['n_spots']}")
    Path(TABLE_OUT).write_text('\n'.join(lines) + '\n')
    print(f"[firm_calibration] wrote {TABLE_OUT}")

    _plot(rows, curve)


def _plot(rows, curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    good = [r for r in rows if np.isfinite(r['recovered'])]
    rec = np.array([r['recovered'] for r in good])
    tru = np.array([r['true'] for r in good])
    lo = np.array([r['lo'] for r in good])
    hi = np.array([r['hi'] for r in good])
    err = np.vstack([rec - lo, hi - rec])

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = [min(rec.min(), tru.min()) - 0.1, max(rec.max(), tru.max()) + 0.1]
    ax.plot(lim, lim, 'k--', lw=1, label='y = x (ideal)')
    ax.errorbar(rec, tru, xerr=err, fmt='o-', color='C0', capsize=3,
                label='recovered -> true (EIV)')
    ax.set_xlabel('recovered alpha (EIV, predicted intensities)')
    ax.set_ylabel('true alpha')
    ax.set_title('Firmed-up calibration curve (recovered vs true)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=130)
    print(f"[firm_calibration] wrote {PLOT_OUT}")


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    main()
