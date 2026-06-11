"""Real-data cross-method alpha, all anchored on EGFP = 2.0 (shared downstream).

The headline real-data comparison: ours vs classical (now) and vs cme-analysis /
SpotMAX (later, via --external-csv-dir), with IDENTICAL downstream processing — the
shared photometry + EIV + calibration of ``benchmark_core`` — so the only thing that
differs between methods is the (x, y) LOCATIONS. Every method is judged by closeness
to the EGFP = 2.0 ground truth.

This complements ``real_alpha.py`` (ours' NATIVE end-to-end alpha): here every method
goes through the SAME shared photometry, isolating detection from the fit.

  * ours      : hrnet_v1 -> (x,y) -> shared photometry + fit.
  * classical : skimage blob_log -> (x,y) -> shared photometry + fit.
  * external  : per-sample (x,y) CSV from cme-analysis/SpotMAX (run elsewhere) via
                --external-csv-dir <dir>/<sample>.csv -> same shared path.

Also reports each method's NATIVE end-to-end alpha where it has its own photometry
(ours: predicted intensities; external: intensities if present in the CSV) for the
fairness table — see EXPERIMENT.md.

    uv run python experiments/2026-06-11_real-data-comparison/real_benchmark.py \
        [--external-csv-dir DIR]
"""

import argparse
import os
from pathlib import Path

import numpy as np

from _common import (ALL_SAMPLES, EGFP_SAMPLES, EGFP_TRUE_ALPHA, ENDO_SAMPLES,
                     EXP_DIR, add_model_args, sample_dir)
from src.eval.adapters import (detect_classical, detect_ours,
                               detect_ours_native, read_detection_csv)
from src.eval.alpha_fit import CalibrationCurve
from src.eval.benchmark_core import (R_AP_DEFAULT, R_IN_DEFAULT, R_OUT_DEFAULT,
                                     aperture_photometry, fit_alpha_bootstrap)
from src.eval.real_data import list_sample_images, load_real_image

CURVE_IN = EXP_DIR / 'calibration_curve.json'
TABLE_OUT = EXP_DIR / 'real_benchmark.csv'
SUMMARY_OUT = EXP_DIR / 'real_benchmark_summary.txt'


def _shared_pairs(images, det_per_image, geom):
    """Per-image (lipid, protein) from shared aperture photometry at det locations."""
    pairs = []
    for arr, xy in zip(images, det_per_image):
        lip, pro = aperture_photometry(arr, xy, *geom)
        pairs.append((lip, pro))
    return pairs


def run_sample(sample, sdir, methods, ext_csv, model, cfg, device, curve,
               subtract_dark, geom, n_boot, rng):
    paths = list_sample_images(sdir)
    images = [load_real_image(p, subtract_dark=subtract_dark) for p in paths]
    out = {}

    # ours / classical: detect -> shared photometry -> shared fit.
    for mname, mfn in methods.items():
        det = [mfn(arr) for arr in images]
        pairs = _shared_pairs(images, det, geom)
        out[mname] = fit_alpha_bootstrap(pairs, curve, n_boot=n_boot, rng=rng)

    # ours NATIVE end-to-end (own predicted intensities + same fit).
    npairs = []
    for arr in images:
        dets = detect_ours_native(model, cfg, device, arr)
        npairs.append((np.array([d['lipid_intensity'] for d in dets], np.float64),
                       np.array([d['protein_intensity'] for d in dets], np.float64)))
    out['ours_native'] = fit_alpha_bootstrap(npairs, curve, n_boot=n_boot, rng=rng)

    # external CSV (optional): map image_id (sorted index NNNNNN or 0..n-1) -> image.
    if ext_csv is not None:
        csv_path = Path(ext_csv) / f"{sample}.csv"
        if csv_path.exists():
            by_id = read_detection_csv(csv_path)
            det = []
            for i, p in enumerate(paths):
                key = next((k for k in (Path(p).stem.split('_')[-1], str(i),
                                        Path(p).stem) if k in by_id), None)
                det.append(by_id[key][:, :2] if key else np.zeros((0, 2)))
            pairs = _shared_pairs(images, det, geom)
            out['external'] = fit_alpha_bootstrap(pairs, curve, n_boot=n_boot, rng=rng)
        else:
            print(f"[real-bench] no external CSV for {sample} at {csv_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_model_args(ap)
    ap.add_argument('--samples', nargs='+', default=ALL_SAMPLES)
    ap.add_argument('--external-csv-dir', default=None,
                    help='dir with <sample>.csv coordinate files (cme-analysis/SpotMAX).')
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--r-ap', type=float, default=R_AP_DEFAULT)
    ap.add_argument('--r-in', type=float, default=R_IN_DEFAULT)
    ap.add_argument('--r-out', type=float, default=R_OUT_DEFAULT)
    args = ap.parse_args()
    geom = (args.r_ap, args.r_in, args.r_out)

    curve = CalibrationCurve.load(CURVE_IN) if CURVE_IN.exists() \
        else CalibrationCurve.default()
    model, cfg, device = load_model_(args)
    methods = {'ours': lambda arr: detect_ours(model, cfg, device, arr)[:, :2],
               'classical': lambda arr: detect_classical(arr)[:, :2]}
    rng = np.random.default_rng(args.seed)

    results = {}
    for s in args.samples:
        results[s] = run_sample(s, sample_dir(args.data_root, s), methods,
                                args.external_csv_dir, model, cfg, device, curve,
                                args.subtract_dark, geom, args.n_boot, rng)

    method_order = ['ours', 'classical', 'ours_native']
    if args.external_csv_dir:
        method_order.append('external')

    # Table: corrected alpha per method per sample.
    hdr = f"{'sample':<17}" + ''.join(f"{m:>20}" for m in method_order)
    print(hdr); print('-' * len(hdr))
    csv = ["sample,method,corrected_alpha,cor_lo,cor_hi,n_spots"]
    for s in args.samples:
        cells = ''
        for m in method_order:
            r = results[s].get(m)
            if r and np.isfinite(r['corrected']):
                cells += f"{r['corrected']:>13.3f}[{r['cor_ci'][0]:.2f},{r['cor_ci'][1]:.2f}]".rjust(20)
                csv.append(f"{s},{m},{r['corrected']},{r['cor_ci'][0]},"
                           f"{r['cor_ci'][1]},{r['n_spots']}")
            else:
                cells += f"{'n/a':>20}"
        print(f"{s:<17}{cells}")
    Path(TABLE_OUT).write_text('\n'.join(csv) + '\n')

    # EGFP anchor per method: mean |corrected - 2.0| across the 4 EGFP samples.
    lines = ["", "=" * 70,
             "EGFP ANCHOR (true alpha = 2.0) per method — shared photometry + fit",
             "=" * 70]
    ranking = []
    for m in method_order:
        vals = [results[s][m]['corrected'] for s in EGFP_SAMPLES
                if m in results.get(s, {}) and np.isfinite(results[s][m]['corrected'])]
        if vals:
            bias = float(np.mean([abs(v - EGFP_TRUE_ALPHA) for v in vals]))
            ranking.append((m, bias, float(np.mean(vals))))
            lines.append(f"  {m:<14} mean alpha={np.mean(vals):.3f}  "
                         f"mean |alpha-2.0|={bias:.3f}  (n={len(vals)} EGFP samples)")
    if ranking:
        best = min(ranking, key=lambda t: t[1])[0]
        lines.append(f"  -> closest to EGFP=2.0 (shared photometry): {best}")
        lines.append("  NOTE: this isolates DETECTION (shared photometry). The native")
        lines.append("  table (ours_native / external own photometry) is the end-to-end")
        lines.append("  comparison; differences localize the gap to detection vs photometry.")

    # Endophilin readout per method.
    lines += ["", "-" * 70, "ENDOPHILIN (curvature sensor, alpha < 2)", "-" * 70]
    for s in ENDO_SAMPLES:
        if s in results:
            cells = '  '.join(f"{m}={results[s][m]['corrected']:.3f}"
                              for m in method_order
                              if m in results[s] and np.isfinite(results[s][m]['corrected']))
            lines.append(f"  {s:<17} {cells}")

    summary = '\n'.join(lines)
    print(summary)
    Path(SUMMARY_OUT).write_text(summary + '\n')
    print(f"\n[real-bench] wrote {TABLE_OUT} and {SUMMARY_OUT}")


def load_model_(args):
    from src.eval.matching import load_model
    return load_model(args.config, args.ckpt)


if __name__ == '__main__':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    main()
