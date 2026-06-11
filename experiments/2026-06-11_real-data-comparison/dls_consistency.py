"""D — DLS consistency: detector size distribution vs the sample's DLS.

Second real-data ground-truth anchor. The detector's recovered lipid-channel size
distribution should resemble the sample's DLS size distribution.

WEIGHTING FIX (critical — the two are NOT directly comparable raw):
  * DLS 'X Intensity' is INTENSITY-weighted (~ N(d) * d^6). We instead read the
    NUMBER weighting N(d) via the SAME parser training used
    (``src/simulator/io.parse_dls(weighting='number')``).
  * The detector emits ONE detection per liposome, so pooled detections are
    NUMBER-weighted N(d) already.
  => Both sides are converted to the SAME number weighting N(d) before comparing.

SIZE PROXY: the network does not output diameter. We invert the simulator's lipid
area law ``lipid_amp = lipid_brightness * (d/100)**2`` (forward_model), i.e.
``d_proxy = 100 * sqrt(lipid_intensity / lipid_brightness)`` with the shared
calibrated ``lipid_brightness`` (sets the absolute size scale). It is a proxy, not
a direct measurement.

Distance: weighted Wasserstein (1-D earth-mover, exact with DLS number weights) +
a KS statistic against DLS samples drawn from N(d). Plots overlay the two number
distributions per sample.

    uv run python experiments/2026-06-11_real-data-comparison/dls_consistency.py
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from _common import (ALL_SAMPLES, EXP_DIR, add_model_args, lipid_brightness_for,
                     sample_dir)
from src.eval.real_data import (detect_sample, find_dls_xlsx, lipid_to_diameter)
from src.simulator.io import parse_dls

PLOT_OUT = EXP_DIR / 'dls_consistency.png'
TABLE_OUT = EXP_DIR / 'dls_consistency.csv'
MAX_D = 500


def detector_diameters(model, cfg, device, sdir, subtract_dark, lipid_brightness):
    per_image = detect_sample(model, cfg, device, sdir, subtract_dark=subtract_dark)
    lip = np.array([dd['lipid_intensity'] for img in per_image for dd in img],
                   np.float64)
    d = lipid_to_diameter(lip, lipid_brightness)
    return d[(d > 0) & (d <= MAX_D)]


def analyze(model, cfg, device, sdir, subtract_dark, lipid_brightness, rng):
    det_d = detector_diameters(model, cfg, device, sdir, subtract_dark,
                               lipid_brightness)
    dls_d, dls_p, _ = parse_dls(find_dls_xlsx(sdir), weighting='number',
                                max_diameter_nm=MAX_D)
    # Weighted Wasserstein: detector samples (uniform weight) vs DLS N(d) weights.
    if det_d.size and dls_d.size:
        wass = float(wasserstein_distance(det_d, dls_d,
                                          v_weights=dls_p / dls_p.sum()))
        dls_samples = rng.choice(dls_d, size=max(det_d.size, 1),
                                 p=dls_p / dls_p.sum())
        ks = float(ks_2samp(det_d, dls_samples).statistic)
    else:
        wass = ks = float('nan')
    return dict(det_d=det_d, dls_d=dls_d, dls_p=dls_p, n=det_d.size,
                wass=wass, ks=ks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_model_args(ap)
    ap.add_argument('--samples', nargs='+', default=ALL_SAMPLES)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from src.eval.matching import load_model
    model, cfg, device = load_model(args.config, args.ckpt)
    rng = np.random.default_rng(args.seed)

    res = {}
    # PER-SAMPLE size-proxy scale (lipid_brightness varies by prep/voltage).
    print("lipid->diameter scale: each sample's OWN fitted lipid_brightness")
    print(f"{'sample':<17} {'lipid_bright':>12} {'n_det':>7} "
          f"{'wasserstein_nm':>15} {'ks':>7}")
    print('-' * 64)
    csv = ["sample,lipid_brightness,n_detections,wasserstein_nm,ks_statistic,"
           "det_median_nm,dls_mean_nm"]
    for s in args.samples:
        lb = lipid_brightness_for(s)
        r = analyze(model, cfg, device, sample_dir(args.data_root, s),
                    args.subtract_dark, lb, rng)
        r['lipid_brightness'] = lb
        res[s] = r
        det_med = float(np.median(r['det_d'])) if r['n'] else float('nan')
        dls_mean = float(np.average(r['dls_d'], weights=r['dls_p'])) \
            if r['dls_d'].size else float('nan')
        print(f"{s:<17} {r['lipid_brightness']:>12.1f} {r['n']:>7} "
              f"{r['wass']:>15.2f} {r['ks']:>7.3f}")
        csv.append(f"{s},{r['lipid_brightness']},{r['n']},{r['wass']},{r['ks']},"
                   f"{det_med},{dls_mean}")
    Path(TABLE_OUT).write_text('\n'.join(csv) + '\n')
    _plot(args.samples, res)
    print(f"\n[dls_consistency] wrote {TABLE_OUT} and {PLOT_OUT}")


def _plot(samples, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(samples)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    bins = np.linspace(0, 350, 40)
    for k, s in enumerate(samples):
        ax = axes[k // ncol][k % ncol]
        r = res[s]
        if r['n']:
            ax.hist(r['det_d'], bins=bins, density=True, alpha=0.5,
                    color='C0', label=f"detector proxy (n={r['n']})")
        if r['dls_d'].size:
            # DLS as a number-weighted density on the same axis.
            ax.hist(r['dls_d'], bins=bins, weights=r['dls_p'], density=True,
                    histtype='step', color='C3', lw=2, label='DLS N(d)')
        ax.set_title(f"{s}\nW={r['wass']:.1f}nm KS={r['ks']:.2f}", fontsize=9)
        ax.set_xlabel('diameter (nm)'); ax.set_ylabel('number density')
        ax.legend(fontsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle('Detector size proxy (number) vs DLS N(d) — same weighting',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_OUT, dpi=130)


if __name__ == '__main__':
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    main()
