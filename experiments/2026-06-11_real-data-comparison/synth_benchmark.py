"""Synthetic cross-method benchmark — detection isolated from the fit (runs NOW).

With photometry + fit FROZEN (``src/eval/benchmark_core.py``) and applied
identically to every method, this compares detectors purely on whose LOCATIONS give
better detection / intensity / alpha. Two adapters run on the instance now: ``ours``
(hrnet_v1) and ``classical`` (skimage ``blob_log``). External tools join later via
the ``external_csv`` path on the EXACTLY exported images (``export_for_external.py``).

Outputs (all binned by TRUE diameter, fine small bins):
  (a) detection F1 / recall vs diameter            -> bench_detection_vs_diam.png
  (b) intensity-recovery logerr vs diameter        -> bench_intensity_vs_diam.png
      (lipid + protein, on emphasis AND dls sizing)
  (c) alpha recovery on the fixed-alpha sweep       -> bench_alpha_recovery.png
      (shared EIV+calibration fit, corrected vs true, y=x)
  + within-bin representativeness table             -> bench_representativeness.csv
  + per-bin metrics                                 -> bench_diameter_metrics.csv
  + native end-to-end vs shared-photometry alpha    -> bench_native_vs_shared.csv

Headline question: with photometry + fit frozen, does our detector beat classical
on small-diameter F1 AND small-bin intensity accuracy WHILE staying representative
AND keeping alpha unbiased — the four-part success criterion (EXPERIMENT.md)?
Reported honestly; ours may or may not win.

    uv run python experiments/2026-06-11_real-data-comparison/synth_benchmark.py \
        --n-workers 32
"""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

import firm_calibration as fc           # reuse ALPHAS + alpha-sweep dataset dirs
from _common import EXP_DIR, REPO_ROOT, add_model_args
from src.eval.adapters import (detect_classical, detect_ours,
                               detect_ours_native)
from src.eval.alpha_fit import CalibrationCurve
from src.eval.benchmark_core import (DIAM_EDGES, DIAM_LABELS, R_AP_DEFAULT,
                                     R_IN_DEFAULT, R_OUT_DEFAULT,
                                     aperture_photometry, evaluate_synthetic,
                                     fit_alpha_bootstrap)
from src.eval.matching import iter_images, load_model

BENCH_CONFIGS = {'emphasis': EXP_DIR / 'config_snapshot' / 'bench_emphasis.yaml',
                 'dls': EXP_DIR / 'config_snapshot' / 'bench_dls.yaml'}
CURVE_IN = EXP_DIR / 'calibration_curve.json'
SMALL_BINS = ('40-55', '55-70', '70-90')         # the candidate-moat regime
BIN_CENTERS = 0.5 * (DIAM_EDGES[:-1] + DIAM_EDGES[1:])


def _dataset_images_dir(name):
    return REPO_ROOT / 'datasets' / name / 'images'


def generate_if_missing(config_path, n_workers, skip):
    cfg = yaml.safe_load(open(config_path))
    img_dir = _dataset_images_dir(cfg['name'])
    if skip or (img_dir.exists() and any(img_dir.glob('*.npy'))):
        print(f"[bench] reuse dataset {cfg['name']} ({img_dir})")
        return REPO_ROOT / 'datasets' / cfg['name']
    from src.generator.generate import run_generation
    print(f"[bench] generating {cfg['name']} ...")
    run_generation(cfg, str(config_path), n_workers=n_workers)
    return REPO_ROOT / 'datasets' / cfg['name']


# --------------------------------------------------------------------------- #
# Run a method over a dataset                                                  #
# --------------------------------------------------------------------------- #
def _methods(model, cfg, device):
    return {
        'ours': lambda arr: detect_ours(model, cfg, device, arr)[:, :2],
        'classical': lambda arr: detect_classical(arr)[:, :2],
    }


def eval_dataset(method_fn, val_dir, geom):
    """(evaluate_synthetic result, per_image_pairs for the fit) over a GT dataset."""
    per_image, pairs = [], []
    for arr, spots in iter_images(val_dir):
        det_xy = method_fn(arr)
        per_image.append((arr, spots, det_xy))
        lip, pro = aperture_photometry(arr, det_xy, *geom)
        pairs.append((lip, pro))
    return evaluate_synthetic(per_image, r_ap=geom[0], r_in=geom[1],
                              r_out=geom[2]), pairs


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_model_args(ap)
    ap.add_argument('--n-workers', type=int, default=None)
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--n-boot', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--r-ap', type=float, default=R_AP_DEFAULT)
    ap.add_argument('--r-in', type=float, default=R_IN_DEFAULT)
    ap.add_argument('--r-out', type=float, default=R_OUT_DEFAULT)
    args = ap.parse_args()
    geom = (args.r_ap, args.r_in, args.r_out)

    # Datasets: the two diameter-eval sets + the alpha sweep (reused from
    # firm_calibration; generate the diameter sets if missing).
    bench_dirs = {k: generate_if_missing(p, args.n_workers, args.skip_generate)
                  for k, p in BENCH_CONFIGS.items()}

    curve = CalibrationCurve.load(CURVE_IN) if CURVE_IN.exists() \
        else CalibrationCurve.default()
    model, cfg, device = load_model(args.config, args.ckpt)
    methods = _methods(model, cfg, device)
    rng = np.random.default_rng(args.seed)

    # --- diameter-binned detection + intensity + representativeness ---------- #
    diam_eval = {}          # diam_eval[sizing][method] = evaluate_synthetic result
    for sizing, vdir in bench_dirs.items():
        diam_eval[sizing] = {}
        for mname, mfn in methods.items():
            print(f"[bench] diameter eval — sizing={sizing} method={mname}")
            ev, _ = eval_dataset(mfn, str(vdir), geom)
            diam_eval[sizing][mname] = ev
            g = ev['global']
            print(f"        global P={g['precision']:.3f} R={g['recall']:.3f} "
                  f"F1={g['f1']:.3f}  (n_gt={g['n_gt']}, n_det={g['n_det']})")
    _write_diameter_tables(diam_eval)
    _plot_detection_vs_diam(diam_eval)
    _plot_intensity_vs_diam(diam_eval)

    # --- alpha recovery on the fixed-alpha sweep (shared fit) ---------------- #
    sweep = {m: [] for m in methods}
    native = []
    for a in fc.ALPHAS:
        vdir = fc._dataset_dir(a)
        if not (vdir / 'images').exists():
            print(f"[bench] sweep alpha={a}: dataset missing ({vdir}); "
                  f"run firm_calibration.py first. Skipping.")
            continue
        for mname, mfn in methods.items():
            _, pairs = eval_dataset(mfn, str(vdir), geom)
            r = fit_alpha_bootstrap(pairs, curve, n_boot=args.n_boot, rng=rng)
            r['true'] = float(a)
            sweep[mname].append(r)
            print(f"[bench] sweep alpha={a} method={mname}: "
                  f"corrected={r['corrected']:.3f} "
                  f"[{r['cor_ci'][0]:.3f},{r['cor_ci'][1]:.3f}] n={r['n_spots']}")
        # native end-to-end for ours (its own photometry/intensities + same fit).
        native.append(_native_ours(model, cfg, device, str(vdir), curve,
                                   args.n_boot, rng, a))
    _plot_alpha_recovery(sweep)
    _write_alpha_tables(sweep, native)
    _small_regime_verdict(diam_eval)
    print(f"\n[bench] done. Tables + plots in {EXP_DIR}")


def _native_ours(model, cfg, device, vdir, curve, n_boot, rng, alpha):
    """Ours NATIVE end-to-end: own (x,y) + own predicted intensities + shared fit."""
    pairs = []
    for arr, _spots in iter_images(vdir):
        dets = detect_ours_native(model, cfg, device, arr)
        lip = np.array([d['lipid_intensity'] for d in dets], np.float64)
        pro = np.array([d['protein_intensity'] for d in dets], np.float64)
        pairs.append((lip, pro))
    r = fit_alpha_bootstrap(pairs, curve, n_boot=n_boot, rng=rng)
    r['true'] = float(alpha)
    return r


# --------------------------------------------------------------------------- #
# Tables + plots                                                              #
# --------------------------------------------------------------------------- #
def _write_diameter_tables(diam_eval):
    rows = ["sizing,method,bin,n_gt,recall,f1,lipid_logerr,protein_logerr,"
            "loc_err,repr_ratio,det_true_protein,missed_true_protein"]
    rep = ["sizing,method,bin,n_gt,recall,repr_ratio,det_true_protein,"
           "missed_true_protein"]
    for sizing, bym in diam_eval.items():
        for m, ev in bym.items():
            for b in ev['per_bin']:
                rows.append(f"{sizing},{m},{b['label']},{b['n_gt']},{b['recall']},"
                            f"{b['f1']},{b['lipid_logerr']},{b['protein_logerr']},"
                            f"{b['loc_err']},{b['repr_ratio']},"
                            f"{b['det_true_protein']},{b['missed_true_protein']}")
                rep.append(f"{sizing},{m},{b['label']},{b['n_gt']},{b['recall']},"
                           f"{b['repr_ratio']},{b['det_true_protein']},"
                           f"{b['missed_true_protein']}")
    (EXP_DIR / 'bench_diameter_metrics.csv').write_text('\n'.join(rows) + '\n')
    (EXP_DIR / 'bench_representativeness.csv').write_text('\n'.join(rep) + '\n')
    print(f"[bench] wrote bench_diameter_metrics.csv + bench_representativeness.csv")


def _write_alpha_tables(sweep, native):
    rows = ["method,true,standard,recovered,corrected,cor_lo,cor_hi,n_spots"]
    for m, lst in sweep.items():
        for r in lst:
            rows.append(f"{m},{r['true']},{r['standard']},{r['recovered']},"
                        f"{r['corrected']},{r['cor_ci'][0]},{r['cor_ci'][1]},"
                        f"{r['n_spots']}")
    (EXP_DIR / 'bench_alpha_recovery.csv').write_text('\n'.join(rows) + '\n')

    # Native (ours) vs shared (ours) end-to-end, side by side on the sweep.
    shared_ours = {r['true']: r for r in sweep.get('ours', [])}
    nv = ["true,ours_shared_corrected,ours_native_corrected,"
          "ours_shared_n,ours_native_n"]
    for r in native:
        s = shared_ours.get(r['true'])
        nv.append(f"{r['true']},{s['corrected'] if s else float('nan')},"
                  f"{r['corrected']},{s['n_spots'] if s else 0},{r['n_spots']}")
    (EXP_DIR / 'bench_native_vs_shared.csv').write_text('\n'.join(nv) + '\n')
    print(f"[bench] wrote bench_alpha_recovery.csv + bench_native_vs_shared.csv")


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _plot_detection_vs_diam(diam_eval):
    plt = _plt()
    sizings = list(diam_eval)
    fig, axes = plt.subplots(2, len(sizings), figsize=(5.5 * len(sizings), 8),
                             squeeze=False)
    for c, sizing in enumerate(sizings):
        ax_f1, ax_rep = axes[0][c], axes[1][c]
        for m, ev in diam_eval[sizing].items():
            f1 = [b['f1'] for b in ev['per_bin']]
            rep = [b['repr_ratio'] for b in ev['per_bin']]
            ax_f1.plot(BIN_CENTERS, f1, 'o-', label=f"{m} (F1={ev['global']['f1']:.2f})")
            ax_rep.plot(BIN_CENTERS, rep, 'o-', label=m)
        ax_f1.set_title(f"detection F1 vs diameter ({sizing})")
        ax_f1.set_ylabel('F1 (per-bin recall vs global precision)')
        ax_f1.set_ylim(0, 1.02)
        ax_rep.axhline(1.0, color='k', ls='--', lw=1)
        ax_rep.set_title(f"within-bin representativeness ({sizing})")
        ax_rep.set_ylabel('median true protein: detected / missed')
        for ax in (ax_f1, ax_rep):
            ax.set_xlabel('diameter (nm)'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            for e in DIAM_EDGES:
                ax.axvline(e, color='0.9', lw=0.5, zorder=0)
    fig.suptitle('Detection vs diameter — F1 AND representativeness '
                 '(a recall win that is biased-bright is NOT a win)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(EXP_DIR / 'bench_detection_vs_diam.png', dpi=130)
    print(f"[bench] wrote bench_detection_vs_diam.png")


def _plot_intensity_vs_diam(diam_eval):
    plt = _plt()
    sizings = list(diam_eval)
    fig, axes = plt.subplots(2, len(sizings), figsize=(5.5 * len(sizings), 8),
                             squeeze=False)
    for c, sizing in enumerate(sizings):
        for r, key, lab in ((0, 'lipid_logerr', 'lipid'),
                            (1, 'protein_logerr', 'protein')):
            ax = axes[r][c]
            for m, ev in diam_eval[sizing].items():
                vals = [b[key] for b in ev['per_bin']]
                ax.plot(BIN_CENTERS, vals, 'o-', label=m)
            ax.axhline(0.0, color='k', ls='--', lw=1)
            ax.set_title(f"{lab} intensity recovery ({sizing})")
            ax.set_xlabel('diameter (nm)')
            ax.set_ylabel('median log10(measured / true)')
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Intensity recovery vs diameter (0 = exact; constant offset = '
                 'aperture capture fraction, does NOT bias alpha)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(EXP_DIR / 'bench_intensity_vs_diam.png', dpi=130)
    print(f"[bench] wrote bench_intensity_vs_diam.png")


def _plot_alpha_recovery(sweep):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    allv = []
    for m, lst in sweep.items():
        if not lst:
            continue
        tru = np.array([r['true'] for r in lst])
        cor = np.array([r['corrected'] for r in lst])
        lo = np.array([r['cor_ci'][0] for r in lst])
        hi = np.array([r['cor_ci'][1] for r in lst])
        ax.errorbar(tru, cor, yerr=np.vstack([cor - lo, hi - cor]),
                    fmt='o-', capsize=3, label=m)
        allv += [tru.min(), tru.max(), cor.min(), cor.max()]
    if allv:
        lim = [min(allv) - 0.1, max(allv) + 0.1]
        ax.plot(lim, lim, 'k--', lw=1, label='y = x')
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('true alpha'); ax.set_ylabel('corrected alpha (shared EIV+calib)')
    ax.set_title('Alpha recovery — shared photometry + fit, per method')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(EXP_DIR / 'bench_alpha_recovery.png', dpi=130)
    print(f"[bench] wrote bench_alpha_recovery.png")


def _small_regime_verdict(diam_eval):
    """Print the four-part small-regime criterion per method (vs the field)."""
    lines = ["", "=" * 70,
             "SMALL-REGIME SCORECARD (the candidate moat: bins "
             + ", ".join(SMALL_BINS) + ")", "=" * 70]
    for sizing, bym in diam_eval.items():
        lines.append(f"-- sizing={sizing} --")
        for m, ev in bym.items():
            sb = [b for b in ev['per_bin'] if b['label'] in SMALL_BINS]
            f1 = np.nanmean([b['f1'] for b in sb])
            le = np.nanmean([abs(b['lipid_logerr']) for b in sb])
            rep = np.nanmean([b['repr_ratio'] for b in sb])
            lines.append(f"  {m:<10} small-bin F1={f1:.3f}  |lipid logerr|={le:.3f}"
                         f"  repr_ratio={rep:.3f} (1.0=representative)")
    lines += [
        "Success for ours = beat baselines on small-bin F1 AND small-bin intensity",
        "accuracy AND keep repr_ratio ~1 AND keep alpha unbiased (all four). A higher",
        "small-bin recall that is biased-bright (repr_ratio>>1) is WORSE for alpha.",
        "If ours is at parity: the contribution is the validated corrected measurement",
        "(EGFP=2.0 + DLS + sweep), NOT 'best detector'.",
    ]
    txt = '\n'.join(lines)
    print(txt)
    (EXP_DIR / 'bench_small_regime_scorecard.txt').write_text(txt + '\n')


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    main()
