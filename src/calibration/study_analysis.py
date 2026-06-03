"""Reusable analysis helpers for calibration studies.

These operate on a study's ``results.json`` (the list of
``{run_id, fitted_params, status}`` written by :mod:`src.calibration.study`) and
produce the readable tables and figures that each experiment's ``analyze.py``
assembles. The per-experiment scripts add their own bespoke plots (e.g. the
per-sample gain-vs-voltage plot) on top of these primitives.

All plotting uses the non-interactive Agg backend; these run in the parent
process after the parallel study has finished.
"""

import csv
import json
from pathlib import Path

import numpy as np

# Fitted parameters in a sensible display order. ``spot_density`` is the only
# per-sample free parameter; the rest are the shared microscope parameters.
PARAM_ORDER = (
    'gain', 'psf_sigma_x', 'psf_sigma_y', 'psf_theta', 'enf',
    'optical_bg_lipid', 'lipid_brightness', 'spot_density',
)


def load_results(exp_dir, ok_only=True):
    """Load ``<exp_dir>/results.json`` -> list of run dicts.

    With ``ok_only`` (default), drop runs whose status is not ``'ok'`` and warn
    about how many were dropped.
    """
    path = Path(exp_dir) / 'results.json'
    records = json.loads(path.read_text())
    if ok_only:
        kept = [r for r in records if r.get('status') == 'ok'
                and r.get('fitted_params')]
        dropped = len(records) - len(kept)
        if dropped:
            print(f"[analysis] WARNING: dropping {dropped} non-ok run(s) "
                  f"out of {len(records)}.")
        return kept
    return records


def param_values(records, param):
    """Return a numpy array of one fitted parameter across runs (NaN if absent)."""
    return np.array(
        [(_get(r, param)) for r in records], dtype=float)


def _get(record, param):
    fp = record.get('fitted_params') or {}
    v = fp.get(param)
    return float(v) if v is not None else np.nan


def summary_stats(values):
    """mean / std / cv / min / max / n for a 1D array, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {'n': 0, 'mean': np.nan, 'std': np.nan, 'cv': np.nan,
                'min': np.nan, 'max': np.nan}
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    cv = float(std / abs(mean)) if mean != 0 else np.nan
    return {'n': int(v.size), 'mean': mean, 'std': std, 'cv': cv,
            'min': float(np.min(v)), 'max': float(np.max(v))}


def write_param_table_csv(records, out_csv, params=PARAM_ORDER):
    """Write a wide table: one row per run, one column per fitted parameter."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', *params])
        for r in records:
            fp = r.get('fitted_params') or {}
            w.writerow([r['run_id'], *[fp.get(p, '') for p in params]])
    print(f"[analysis] wrote {out_csv}")


def write_summary_csv(records, out_csv, params=PARAM_ORDER):
    """Write a per-parameter summary: mean / std / cv / min / max / n."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'n', 'mean', 'std', 'cv', 'min', 'max'])
        for p in params:
            s = summary_stats(param_values(records, p))
            w.writerow([p, s['n'], s['mean'], s['std'], s['cv'],
                        s['min'], s['max']])
    print(f"[analysis] wrote {out_csv}")


def _new_axes_grid(n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             squeeze=False)
    return plt, fig, axes, nrows, ncols


def plot_params_across_runs(records, out_png, params=PARAM_ORDER,
                            title='Fitted parameters across runs',
                            xlabel='run'):
    """Small-multiples: one panel per parameter, value vs run_id.

    Used to eyeball whether fitted parameters MOVE across runs (weight-sweep
    configs, or samples). A flat line means the parameter is stable.
    """
    run_ids = [r['run_id'] for r in records]
    x = np.arange(len(run_ids))
    plt, fig, axes, nrows, ncols = _new_axes_grid(len(params))
    for idx, p in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        y = param_values(records, p)
        ax.plot(x, y, 'o-', color='C0', ms=5)
        ax.set_title(p)
        ax.set_xticks(x)
        ax.set_xticklabels(run_ids, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    for j in range(len(params), nrows * ncols):  # hide unused panels
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[analysis] wrote {out_png}")


def plot_param_histograms(records, out_png, params=PARAM_ORDER,
                          title='Bootstrap parameter distributions'):
    """Small-multiples: one histogram per parameter with mean/std/CV annotated.

    Used for the bootstrap study (distribution of each fitted parameter over the
    repeats).
    """
    plt, fig, axes, nrows, ncols = _new_axes_grid(len(params))
    for idx, p in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        y = param_values(records, p)
        y = y[~np.isnan(y)]
        s = summary_stats(y)
        if y.size:
            ax.hist(y, bins=min(20, max(5, y.size // 3)),
                    color='C0', alpha=0.75, edgecolor='white')
            ax.axvline(s['mean'], color='C3', lw=1.5, label='mean')
        ax.set_title(p)
        cv_txt = f"{s['cv']:.3f}" if np.isfinite(s['cv']) else 'n/a'
        ax.text(0.97, 0.95,
                f"mean={s['mean']:.3g}\nstd={s['std']:.3g}\nCV={cv_txt}\nn={s['n']}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.grid(True, alpha=0.3)
    for j in range(len(params), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[analysis] wrote {out_png}")
