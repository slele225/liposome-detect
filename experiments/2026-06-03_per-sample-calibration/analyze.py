"""EXP 2 analysis — per-sample independent calibration.

Runs after the per_sample study (invoked by run.sh). Reads results.json and
produces, into figures/:
  - cross_sample_params.csv       wide table: one row per sample, fitted params
  - gain_vs_voltage.png           fitted gain vs the known 561 detector voltage
  - params_across_samples.png     each non-gain fitted param across samples

Question this addresses: which lipid parameters are setting-dependent vs stable
across samples, and does the fitted gain track the 561 detector voltage?
"""

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import study_analysis as sa

# Known 561 nm detector (PMT) voltage per sample, in volts. The detector is
# turned DOWN for brighter (higher-concentration) samples, so if the fitted gain
# reflects real PMT gain it should rise with voltage.
VOLTAGE_561 = {
    '20nM_EGFP': 750, '50nM_EGFP': 640, '100nM_EGFP': 630,
    '300nM_EGFP': 580, '25nM_endophilin': 670, '300nM_endophilin': 600,
}


def plot_gain_vs_voltage(records, out_png):
    pts = []
    for r in records:
        v = VOLTAGE_561.get(r['run_id'])
        g = (r.get('fitted_params') or {}).get('gain')
        if v is not None and g is not None:
            pts.append((v, float(g), r['run_id']))
    if not pts:
        print('[exp2] no gain/voltage points to plot.')
        return
    pts.sort()
    vs = np.array([p[0] for p in pts])
    gs = np.array([p[1] for p in pts])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['C0' if 'EGFP' in p[2] else 'C1' for p in pts]
    ax.scatter(vs, gs, c=colors, s=70, zorder=3)
    for v, g, name in pts:
        ax.annotate(name, (v, g), textcoords='offset points', xytext=(6, 4),
                    fontsize=8)
    ax.set_xlabel('561 nm detector voltage (V)')
    ax.set_ylabel('Fitted gain (ADU / photon)')
    ax.set_title('Fitted gain vs known 561 detector voltage')
    ax.grid(True, alpha=0.3)
    # Legend proxy for the two protein families.
    ax.scatter([], [], c='C0', label='EGFP')
    ax.scatter([], [], c='C1', label='endophilin')
    ax.legend()
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'[exp2] wrote {out_png}')


def main():
    records = sa.load_results(EXP_DIR)
    fig_dir = EXP_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Cross-sample fitted-parameter table (gain, psf_*, enf, optical_bg_lipid,
    # lipid_brightness, spot_density), one row per sample.
    sa.write_param_table_csv(records, fig_dir / 'cross_sample_params.csv')

    # Fitted gain vs the known 561 detector voltage.
    plot_gain_vs_voltage(records, fig_dir / 'gain_vs_voltage.png')

    # Each non-gain fitted parameter across the samples (is it stable or
    # setting-dependent?).
    other = [p for p in sa.PARAM_ORDER if p != 'gain']
    sa.plot_params_across_runs(
        records, fig_dir / 'params_across_samples.png', params=other,
        title='Fitted parameters across samples', xlabel='sample')

    print('[exp2] analysis complete.')


if __name__ == '__main__':
    main()
