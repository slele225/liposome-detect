"""EXP 3 analysis — bootstrap stability on 25nM_endophilin.

Runs after the bootstrap study (invoked by run.sh). Reads results.json (one
record per repeat) and produces, into figures/:
  - bootstrap_params_per_repeat.csv  wide table: one row per repeat
  - bootstrap_summary.csv            per-param mean / std / CV / min / max
  - bootstrap_distributions.png      a histogram per fitted param (mean marked,
                                     mean/std/CV annotated)

Question this addresses: how stable is the calibration to the particular subset
of d=25 images? Low coefficient of variation (CV = std/mean) across repeats
means the fitted parameter is well-determined by any 25-image subset.
"""

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.calibration import study_analysis as sa


def main():
    records = sa.load_results(EXP_DIR)
    fig_dir = EXP_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print(f'[exp3] {len(records)} successful repeats.')

    # Per-repeat wide table and the per-parameter summary (mean/std/CV/min/max).
    sa.write_param_table_csv(records, fig_dir / 'bootstrap_params_per_repeat.csv')
    sa.write_summary_csv(records, fig_dir / 'bootstrap_summary.csv')

    # Distribution of each fitted parameter across the repeats.
    sa.plot_param_histograms(
        records, fig_dir / 'bootstrap_distributions.png',
        title='Bootstrap parameter distributions — 25nM_endophilin (d=25)')

    # Echo the CV table to stdout for a quick read of stability.
    print('[exp3] coefficient of variation (CV = std/mean) per parameter:')
    for p in sa.PARAM_ORDER:
        s = sa.summary_stats(sa.param_values(records, p))
        print(f'         {p:18s} mean={s["mean"]:.4g}  CV={s["cv"]:.3f}')

    print('[exp3] analysis complete.')


if __name__ == '__main__':
    main()
