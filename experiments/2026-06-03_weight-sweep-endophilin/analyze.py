"""EXP 1 analysis — discrepancy weight-sweep on 25nM_endophilin.

Runs after the weight_sweep study (invoked by run.sh). Reads results.json (one
record per weight config) and produces, into figures/:
  - params_across_configs.csv         wide table: one row per weight config
  - param_spread_across_configs.csv   per-param mean / std / CV / min / max
                                      ACROSS the configs (the spread metric)
  - params_across_configs.png         each fitted param across the configs

IMPORTANT: configs are NOT compared by loss value. Each config optimizes a
DIFFERENTLY WEIGHTED objective, so its loss is on a different scale and the
numbers are not comparable. The only valid comparison is whether the fitted
PARAMETERS move across configs:
  - parameters that barely move (low CV across configs) => the weighting does
    not matter for the calibration; the fit is driven by the data, not the loss
    weighting.
  - parameters that move a lot => that term's weight genuinely steers the fit.
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

    print(f'[exp1] {len(records)} weight configs: '
          f'{[r["run_id"] for r in records]}')

    # Wide table of fitted params across the configs.
    sa.write_param_table_csv(records, fig_dir / 'params_across_configs.csv')

    # Spread of each fitted parameter ACROSS the configs (the comparison metric
    # — NOT loss). Low CV => weighting doesn't change the calibration.
    sa.write_summary_csv(records, fig_dir / 'param_spread_across_configs.csv')

    # Each fitted parameter across the configs (visual spread).
    sa.plot_params_across_runs(
        records, fig_dir / 'params_across_configs.png',
        title='Fitted parameters across discrepancy-weight configs',
        xlabel='weight config')

    # Echo the across-config spread to stdout. We deliberately do NOT print or
    # rank by loss value (not comparable across weightings).
    print('[exp1] parameter spread ACROSS configs (CV = std/mean; low = stable):')
    for p in sa.PARAM_ORDER:
        s = sa.summary_stats(sa.param_values(records, p))
        print(f'         {p:18s} mean={s["mean"]:.4g}  CV={s["cv"]:.3f}')

    print('[exp1] analysis complete.')


if __name__ == '__main__':
    main()
