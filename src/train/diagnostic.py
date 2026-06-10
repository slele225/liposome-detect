"""Diagnostic check over a completed (short) training run — the GATE before a
full multi-hour H100 job.

    python -m src.train.diagnostic --run <run_dir> [--out figures/diag]

Reads the run's ``metrics.jsonl`` and answers the two questions smoke scale cannot
(see PROMPT 2c):

  1. TERM BALANCE after the focal heatmap term drops. The four WEIGHTED loss terms
     (w_hm*heatmap, w_off*offset, w_lip*lipid, w_pro*protein) start balanced but
     SHIFT as detection improves and the focal term shrinks; intensity terms
     becoming dominant in Phase 2 is desired — but we want it OBSERVED. We report
     the spread (max/min of |weighted term|) at epoch 0 vs the last epoch and flag
     any epoch whose spread exceeds ``--spread-threshold`` (default 10x).

  2. PHASE-BOUNDARY smoothness. The static-weighted ``val_total`` changes character
     at the MSE->NLL boundary (the intensity term's magnitude shifts), so a step in
     ``val_total`` there may be a METRIC ARTIFACT, not real degradation. We compare
     the ``val_total`` step ACROSS the boundary to the median |Δval_total| of the
     surrounding epochs and flag it if it is > ``--boundary-threshold``x (default 3x).
     We ALSO plot the BOUNDARY-CONSISTENT metrics (val_detection_f1,
     val_intensity_logmse): if those are smooth while val_total jumps, the jump is
     an artifact and the full run should early-stop on a boundary-consistent metric.

Prints a 3-line VERDICT and (with ``--out``) writes the two plots.
"""

import argparse
import json
from pathlib import Path

import numpy as np

_TERM_KEYS = ('heatmap', 'offset', 'lipid', 'protein')
_WEIGHT_OF = {'heatmap': 'w_hm', 'offset': 'w_off', 'lipid': 'w_lip',
              'protein': 'w_pro'}
_CONSISTENT = ('val_detection_f1', 'val_intensity_logmse')


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def load_metrics(run_dir):
    """Load ``<run_dir>/metrics.jsonl`` into a list of per-epoch records."""
    path = Path(run_dir) / 'metrics.jsonl'
    if not path.exists():
        raise FileNotFoundError(f"no metrics log at {path}")
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    records.sort(key=lambda r: r['epoch'])
    return records


def boundary_epoch(records):
    """First epoch with NLL on (== the emphasis/loss-warmup boundary), or None."""
    for r in records:
        if r.get('use_nll'):
            return int(r['epoch'])
    return None


def epoch_weighted_terms(rec):
    """The four WEIGHTED per-term losses for an epoch.

    Prefers the logged ``train_weighted`` block; falls back to multiplying the
    unweighted ``train`` parts by the active ``weights`` (so an older log without
    the block still works).
    """
    tw = rec.get('train_weighted')
    if tw is not None:
        return {k: float(tw[k]) for k in _TERM_KEYS}
    w, tr = rec['weights'], rec['train']
    return {k: float(w[_WEIGHT_OF[k]]) * float(tr[k]) for k in _TERM_KEYS}


def term_spread(terms):
    """max/min of the |weighted terms| — the balance ratio (1.0 = perfectly even)."""
    vals = [abs(float(v)) for v in terms.values()]
    lo = max(min(vals), 1e-9)            # floor: NLL terms can be ~0 / negative
    return max(vals) / lo


# --------------------------------------------------------------------------- #
# Boundary-step analysis                                                       #
# --------------------------------------------------------------------------- #
def _val_series(records, key):
    """``key`` from each record's ``val`` block, aligned with the epoch list."""
    return [float(r['val'].get(key, float('nan'))) for r in records]


def boundary_step_ratio(epochs, values, b_epoch):
    """Step in ``values`` across ``b_epoch`` vs the median surrounding |step|.

    Returns ``{boundary_step, surrounding_median, ratio}`` or ``None`` if it cannot
    be computed (no boundary, boundary at epoch 0, or non-finite values at the
    boundary). ``ratio`` is ``inf`` when the surrounding median is 0.
    """
    if b_epoch is None or b_epoch not in epochs:
        return None
    idx = epochs.index(b_epoch)
    if idx == 0:
        return None
    a, b = values[idx - 1], values[idx]
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    boundary_step = abs(b - a)
    deltas = []
    for i in range(1, len(values)):
        if i == idx:
            continue
        x, y = values[i - 1], values[i]
        if np.isfinite(x) and np.isfinite(y):
            deltas.append(abs(y - x))
    med = float(np.median(deltas)) if deltas else float('nan')
    if np.isfinite(med) and med > 0:
        ratio = boundary_step / med
    else:
        ratio = float('inf') if boundary_step > 0 else 0.0
    return {'boundary_step': boundary_step, 'surrounding_median': med,
            'ratio': ratio}


# --------------------------------------------------------------------------- #
# Top-level analysis                                                           #
# --------------------------------------------------------------------------- #
def analyze(records, spread_threshold=10.0, boundary_threshold=3.0):
    """Compute term-balance + boundary-smoothness diagnostics from the records."""
    if not records:
        raise ValueError("no metrics records to analyze")
    epochs = [int(r['epoch']) for r in records]
    weighted = [epoch_weighted_terms(r) for r in records]
    spread = [term_spread(t) for t in weighted]
    spread_flagged = [ep for ep, sp in zip(epochs, spread) if sp > spread_threshold]

    b = boundary_epoch(records)
    vt = _val_series(records, 'val_total')
    vt_boundary = boundary_step_ratio(epochs, vt, b)
    vt_flagged = (vt_boundary is not None
                  and vt_boundary['ratio'] > boundary_threshold)

    consistent = {}
    consistent_ratios = []
    for key in _CONSISTENT:
        info = boundary_step_ratio(epochs, _val_series(records, key), b)
        consistent[key] = info
        if info is not None and np.isfinite(info['ratio']):
            consistent_ratios.append((key, info['ratio']))

    # Consistent metrics are "smooth" if every computable one is within threshold.
    consistent_available = bool(consistent_ratios)
    consistent_smooth = all(r <= boundary_threshold for _, r in consistent_ratios)
    # Artifact = val_total jumps at the boundary while the boundary-consistent
    # metrics stay smooth (the jump is a yardstick artifact, not real regression).
    artifact = bool(vt_flagged and consistent_available and consistent_smooth)

    if not vt_flagged:
        recommended = 'val_total'
    elif artifact:
        # Prefer the intensity-led consistent metric if it was computable.
        lm = consistent.get('val_intensity_logmse')
        if lm is not None and np.isfinite(lm['ratio']):
            recommended = 'val_intensity_logmse'
        else:
            recommended = 'val_detection_f1'
    else:
        # val_total jumps AND consistent metrics also jump (or none available):
        # the step is likely real degradation — keep val_total, investigate.
        recommended = 'val_total'

    return {
        'epochs': epochs,
        'boundary_epoch': b,
        'weighted': weighted,
        'spread': spread,
        'spread_threshold': float(spread_threshold),
        'spread_epoch0': spread[0],
        'spread_last': spread[-1],
        'spread_max': max(spread),
        'spread_flagged_epochs': spread_flagged,
        'boundary_threshold': float(boundary_threshold),
        'val_total_boundary': vt_boundary,
        'val_total_flagged': bool(vt_flagged),
        'consistent': consistent,
        'consistent_available': consistent_available,
        'consistent_smooth': bool(consistent_smooth),
        'artifact': artifact,
        'recommended_metric': recommended,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def format_verdict(a):
    """Build the printed summary + the 3-line VERDICT from an ``analyze`` result."""
    lines = []
    lines.append('=' * 64)
    lines.append(f"DIAGNOSTIC  (boundary epoch = {a['boundary_epoch']}, "
                 f"{len(a['epochs'])} epochs)")
    lines.append('-' * 64)

    # 1. Term balance.
    lines.append("Term balance (weighted: w_hm*hm, w_off*off, w_lip*lip, w_pro*pro)")
    lines.append(f"  spread epoch0 = {a['spread_epoch0']:.2f}x  ->  "
                 f"last = {a['spread_last']:.2f}x   (max {a['spread_max']:.2f}x, "
                 f"threshold {a['spread_threshold']:.0f}x)")
    if a['spread_flagged_epochs']:
        lines.append(f"  FLAG: spread > {a['spread_threshold']:.0f}x at epochs "
                     f"{a['spread_flagged_epochs']}")
    balance_ok = not a['spread_flagged_epochs']

    # 2. Boundary smoothness.
    vt = a['val_total_boundary']
    lines.append("Phase boundary (val_total, static yardstick)")
    if vt is None:
        lines.append("  (not computable — no boundary or boundary at epoch 0)")
    else:
        lines.append(f"  step across boundary = {vt['boundary_step']:.4f}  vs "
                     f"surrounding median {vt['surrounding_median']:.4f}  -> "
                     f"{vt['ratio']:.2f}x (threshold {a['boundary_threshold']:.0f}x)")
    for key in _CONSISTENT:
        info = a['consistent'].get(key)
        if info is None:
            lines.append(f"  {key}: (not computable)")
        else:
            lines.append(f"  {key}: boundary {info['ratio']:.2f}x "
                         f"(step {info['boundary_step']:.4f})")

    # 3-line VERDICT.
    lines.append('-' * 64)
    v1 = ("(a) term balance: HEALTHY - within "
          f"{a['spread_threshold']:.0f}x all epochs" if balance_ok else
          f"(a) term balance: CHECK - spread exceeded {a['spread_threshold']:.0f}x "
          f"at {a['spread_flagged_epochs']}")
    if not a['val_total_flagged']:
        v2 = "(b) boundary: SMOOTH — val_total does not jump at the boundary"
    elif a['artifact']:
        v2 = ("(b) boundary: val_total JUMPS but consistent metrics are smooth "
              "-> ARTIFACT (use a boundary-consistent early-stop metric)")
    else:
        v2 = ("(b) boundary: val_total jumps AND consistent metrics also move "
              "-> likely REAL degradation, investigate")
    v3 = f"(c) recommended early_stop_metric for the full run: {a['recommended_metric']}"
    lines.extend([v1, v2, v3, '=' * 64])
    return '\n'.join(lines)


def make_plots(records, a, out_dir):
    """Write the two diagnostic plots; returns the written paths (or [] if no mpl)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover - env-dependent
        print(f"[diagnostic] matplotlib unavailable ({e}); skipping plots")
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = a['epochs']
    b = a['boundary_epoch']
    written = []

    # Plot 1: weighted-term-vs-epoch (4 lines + total), boundary marked.
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in _TERM_KEYS:
        ax.plot(epochs, [w[k] for w in a['weighted']], marker='o', label=k)
    ax.plot(epochs, [sum(w.values()) for w in a['weighted']], 'k--', label='total')
    if b is not None:
        ax.axvline(b, color='grey', ls=':', label=f'NLL boundary (ep {b})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('weighted train loss term')
    ax.set_title('Weighted term balance over training')
    ax.legend(fontsize=8)
    p1 = out / 'term_balance.png'
    fig.tight_layout()
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    written.append(p1)

    # Plot 2: val_total + the boundary-consistent metrics, boundary marked.
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    series = [('val_total', 'val_total (static yardstick)'),
              ('val_detection_f1', 'val_detection_f1 (consistent)'),
              ('val_intensity_logmse', 'val_intensity_logmse (consistent)')]
    for ax, (key, title) in zip(axes, series):
        ax.plot(epochs, _val_series(records, key), marker='o')
        if b is not None:
            ax.axvline(b, color='grey', ls=':')
        ax.set_ylabel(key)
        ax.set_title(title, fontsize=9)
    axes[-1].set_xlabel('epoch')
    p2 = out / 'boundary_smoothness.png'
    fig.tight_layout()
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    written.append(p2)
    return written


def main():
    ap = argparse.ArgumentParser(
        description='Diagnostic check over a training run (the gate before a full run).')
    ap.add_argument('--run', required=True, help='Run directory (has metrics.jsonl).')
    ap.add_argument('--out', default=None, help='Plot output dir (default: <run>/diag).')
    ap.add_argument('--spread-threshold', type=float, default=10.0)
    ap.add_argument('--boundary-threshold', type=float, default=3.0)
    args = ap.parse_args()

    records = load_metrics(args.run)
    a = analyze(records, spread_threshold=args.spread_threshold,
                boundary_threshold=args.boundary_threshold)
    print(format_verdict(a))

    out_dir = args.out or (Path(args.run) / 'diag')
    written = make_plots(records, a, out_dir)
    for p in written:
        print(f"[diagnostic] wrote {p}")


if __name__ == '__main__':
    main()
