"""C — per-sample alpha on REAL images: STANDARD (OLS) vs CORRECTED (EIV+calib).

For each real sample, run the detector over all images, pool the detected spots'
(log lipid, log protein) [+ per-spot log-variances], and compute alpha two ways:

  * STANDARD : alpha = 2 * ols_slope(log lipid, log protein)               (biased)
  * CORRECTED: alpha = CalibrationCurve.invert( recover_alpha(...) )
               recover_alpha = constant-lambda Deming (EIV), lambda from mean
               exp(logvar) per axis, used DIRECTLY (log space; no /intensity**2),
               NO per-spot weighting (contraindicated: variance/size confound).

Bootstrap over images gives a CI on each alpha.

THE ANCHOR: EGFP is the negative control with true alpha = 2.0 (area-proportional
binding). Per pipeline we report mean |alpha - 2.0| across the 4 EGFP samples —
the pipeline nearer 2.0 is more accurate on real-data ground truth (the go/no-go).
We also report endophilin (the curvature sensor, alpha < 2): does STANDARD vs
CORRECTED differ enough to change the biological reading?

    uv run python experiments/2026-06-11_real-data-comparison/real_alpha.py
"""

import argparse
from pathlib import Path

import numpy as np

from _common import (ALL_SAMPLES, EGFP_SAMPLES, EGFP_TRUE_ALPHA, ENDO_SAMPLES,
                     EXP_DIR, add_model_args, sample_dir)
from src.eval.alpha_fit import CalibrationCurve, ols_slope, recover_alpha
from src.eval.real_data import detect_sample, spots_to_logxy

CURVE_IN = EXP_DIR / 'calibration_curve.json'
TABLE_OUT = EXP_DIR / 'real_alpha.csv'
SUMMARY_OUT = EXP_DIR / 'real_alpha_summary.txt'
MIN_SPOTS = 10


def _alphas(groups, idx, curve):
    """(standard_ols, corrected) over the pooled spots of image-groups ``idx``."""
    Ll = np.concatenate([groups[i][0] for i in idx])
    Lp = np.concatenate([groups[i][1] for i in idx])
    vl = np.concatenate([groups[i][2] for i in idx])
    vp = np.concatenate([groups[i][3] for i in idx])
    if Ll.size < MIN_SPOTS or np.unique(Ll).size < 2:
        return float('nan'), float('nan')
    standard = 2.0 * ols_slope(Ll, Lp)
    recovered = recover_alpha(Ll, Lp, var_x=vl, var_y=vp)   # x=lipid, y=protein
    corrected = float(curve.invert(recovered))
    return standard, corrected


def analyze_sample(model, cfg, device, sdir, subtract_dark, curve, n_boot, rng):
    per_image = detect_sample(model, cfg, device, sdir, subtract_dark=subtract_dark)
    groups = []
    for dets in per_image:
        Ll, Lp, vl, vp = spots_to_logxy(dets)
        if Ll.size:
            groups.append((Ll, Lp, vl, vp))
    n_img = len(groups)
    n_spots = int(sum(len(g[0]) for g in groups))
    if n_img == 0:
        return dict(n_img=0, n_spots=0, std=float('nan'), cor=float('nan'),
                    std_ci=(float('nan'),) * 2, cor_ci=(float('nan'),) * 2)
    std, cor = _alphas(groups, range(n_img), curve)
    sb, cb = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, size=n_img)
        s, c = _alphas(groups, idx, curve)
        if np.isfinite(s):
            sb.append(s); cb.append(c)
    sb, cb = np.array(sb), np.array(cb)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
        if a.size else (float('nan'), float('nan'))
    return dict(n_img=n_img, n_spots=n_spots, std=std, cor=cor,
                std_ci=ci(sb), cor_ci=ci(cb))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_model_args(ap)
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--samples', nargs='+', default=ALL_SAMPLES)
    args = ap.parse_args()

    if CURVE_IN.exists():
        curve = CalibrationCurve.load(CURVE_IN)
        curve_src = str(CURVE_IN)
    else:
        curve = CalibrationCurve.default()
        curve_src = 'CalibrationCurve.default() (4-pt seed; run firm_calibration.py)'

    from src.eval.matching import load_model
    model, cfg, device = load_model(args.config, args.ckpt)
    rng = np.random.default_rng(args.seed)

    results = {}
    hdr = (f"{'sample':<17} {'n_spots':>8}  {'alpha_STANDARD(OLS)':>20}  "
           f"{'alpha_CORRECTED(EIV+calib)':>27}  {'|corr-2.0|':>10}")
    print(f"calibration curve: {curve_src}")
    print(hdr)
    print('-' * len(hdr))
    csv = ["sample,n_img,n_spots,alpha_standard,std_lo,std_hi,"
           "alpha_corrected,cor_lo,cor_hi,abs_corr_minus_2"]
    for s in args.samples:
        r = analyze_sample(model, cfg, device, sample_dir(args.data_root, s),
                           args.subtract_dark, curve, args.n_boot, rng)
        results[s] = r
        d2 = abs(r['cor'] - EGFP_TRUE_ALPHA)
        std_str = f"{r['std']:.3f} [{r['std_ci'][0]:.3f},{r['std_ci'][1]:.3f}]"
        cor_str = f"{r['cor']:.3f} [{r['cor_ci'][0]:.3f},{r['cor_ci'][1]:.3f}]"
        print(f"{s:<17} {r['n_spots']:>8}  {std_str:>20}  {cor_str:>27}  {d2:>10.3f}")
        csv.append(f"{s},{r['n_img']},{r['n_spots']},{r['std']},{r['std_ci'][0]},"
                   f"{r['std_ci'][1]},{r['cor']},{r['cor_ci'][0]},{r['cor_ci'][1]},{d2}")
    Path(TABLE_OUT).write_text('\n'.join(csv) + '\n')

    # EGFP anchor: mean |alpha - 2.0| per pipeline across the 4 EGFP samples.
    egfp = [results[s] for s in EGFP_SAMPLES if s in results and np.isfinite(results[s]['std'])]
    lines = []
    if egfp:
        std_bias = float(np.mean([abs(r['std'] - EGFP_TRUE_ALPHA) for r in egfp]))
        cor_bias = float(np.mean([abs(r['cor'] - EGFP_TRUE_ALPHA) for r in egfp]))
        std_mean = float(np.mean([r['std'] for r in egfp]))
        cor_mean = float(np.mean([r['cor'] for r in egfp]))
        winner = 'CORRECTED' if cor_bias < std_bias else 'STANDARD'
        lines += [
            "",
            "=" * 70,
            "EGFP ANCHOR (true alpha = 2.0) — the go/no-go",
            "=" * 70,
            f"  n EGFP samples            : {len(egfp)}",
            f"  STANDARD  mean alpha       : {std_mean:.3f}   "
            f"mean |alpha-2.0| = {std_bias:.3f}",
            f"  CORRECTED mean alpha       : {cor_mean:.3f}   "
            f"mean |alpha-2.0| = {cor_bias:.3f}",
            f"  -> nearer 2.0 on real ground truth: {winner}",
        ]
        if std_mean < 1.9 and cor_bias < std_bias:
            lines.append("  READING: STANDARD is biased LOW (<1.9) and CORRECTED "
                         "is nearer 2.0 -> the bias is REAL on real data.")
        elif std_bias < 0.1:
            lines.append("  READING: STANDARD is already ~2.0 -> the correction "
                         "is small in this regime (report honestly).")
        else:
            lines.append("  READING: see the numbers above; interpret per the "
                         "EXPERIMENT.md go/no-go criteria.")
        # Internal consistency: the 4 EGFP samples share ONE liposome prep, so under
        # the correct pipeline their CORRECTED alphas should be mutually consistent
        # (all ~2.0). Divergence (e.g. concentration-dependent leakage) is a RED FLAG
        # separate from the standard-vs-corrected bias.
        cor_vals = [r['cor'] for r in egfp]
        spread = float(np.max(cor_vals) - np.min(cor_vals))
        lines += [
            f"  EGFP internal consistency : CORRECTED spread (max-min) = {spread:.3f} "
            f"across the 4 samples",
        ]
        if spread > 0.3:
            lines.append("    RED FLAG: EGFP corrected alphas DIVERGE (>0.3) despite "
                         "a shared prep -> suspect concentration-dependent leakage / "
                         "a per-concentration artifact, NOT the OLS-vs-EIV bias.")
        else:
            lines.append("    OK: EGFP corrected alphas are mutually consistent "
                         "(shared prep -> shared alpha, as expected).")

    # Endophilin: does the correction change the biological reading?
    endo = [(s, results[s]) for s in ENDO_SAMPLES
            if s in results and np.isfinite(results[s]['std'])]
    if endo:
        lines += ["", "-" * 70,
                  "ENDOPHILIN (curvature sensor, expect alpha < 2)", "-" * 70]
        for s, r in endo:
            lines.append(f"  {s:<17} STANDARD={r['std']:.3f}  "
                         f"CORRECTED={r['cor']:.3f}  (delta={r['cor']-r['std']:+.3f})")
        lines.append("  Sensing strength = distance below 2.0; compare whether "
                     "STANDARD vs CORRECTED shifts that distance materially.")

    summary = '\n'.join(lines)
    print(summary)
    Path(SUMMARY_OUT).write_text(summary + '\n')
    print(f"\n[real_alpha] wrote {TABLE_OUT} and {SUMMARY_OUT}")


if __name__ == '__main__':
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    main()
