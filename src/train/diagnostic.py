"""Diagnostic check over a completed (short) training run — the GATE before a
full multi-hour H100 job.

    python -m src.train.diagnostic --run <run_dir> [--out figures/diag]

Reads the run's ``metrics.jsonl`` and answers the two questions smoke scale cannot
(see PROMPT 2c):

  1. TERM BALANCE after the focal heatmap term drops. The four WEIGHTED loss terms
     (w_hm*heatmap, w_off*offset, w_lip*lipid, w_pro*protein) start balanced but
     SHIFT as detection improves and the focal term shrinks. Intensity terms
     DOMINATING in Phase 2 is EXPECTED and healthy — and once NLL is on the lipid/
     protein terms go NEGATIVE, so a naive max/min spread ratio is meaningless. We
     instead watch for a term that COLLAPSES toward zero relative to the others
     (i.e. stops contributing to the loss): ``term_collapse`` is the smallest
     |weighted term| as a fraction of the MEDIAN |weighted term|. We flag any epoch
     whose collapse ratio falls below ``--collapse-threshold`` (default 0.05). A
     term merely being LARGE/dominant does NOT lower this ratio and is not flagged.

  2. PHASE-BOUNDARY smoothness. The static-weighted ``val_total`` changes character
     at the MSE->NLL boundary (the intensity term's magnitude shifts), so a step in
     ``val_total`` there may be a METRIC ARTIFACT, not real degradation. We compare
     the ``val_total`` step ACROSS the boundary to the median |Δval_total| of the
     surrounding epochs (a RELATIVE step ratio) and flag it if it is >
     ``--boundary-threshold``x (default 3x). We then check the BOUNDARY-CONSISTENT
     metrics (val_detection_f1, val_intensity_logmse) the SAME relative way. The
     recommendation defaults to ``val_intensity_logmse`` and only falls back to
     ``val_total`` if a consistent metric GENUINELY degrades comparably to val_total
     at the boundary (its relative step is both significant AND a sizable fraction
     of val_total's) — a tiny wiggle amplified by a near-flat baseline does not
     count.

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


def term_collapse(terms):
    """Smallest |weighted term| as a fraction of the MEDIAN |weighted term|.

    Robust to the NEGATIVE / near-zero NLL terms that appear once NLL is on (it
    compares magnitudes, not signed values). ~1.0 = balanced; -> 0 means one term
    has COLLAPSED toward zero relative to the others and stopped contributing to the
    loss. A term being LARGE/dominant (intensity terms leading in Phase 2 is
    EXPECTED) does NOT lower this ratio, so it is not mistaken for a problem.
    """
    vals = sorted(abs(float(v)) for v in terms.values())
    med = float(np.median(vals))
    if med <= 0:                          # all terms ~0 — degenerate, treat as collapsed
        return 0.0
    return vals[0] / med


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
def analyze(records, collapse_threshold=0.05, boundary_threshold=3.0,
            comparable_frac=0.5):
    """Compute term-balance + boundary-smoothness diagnostics from the records.

    ``collapse_threshold`` — flag an epoch if its ``term_collapse`` ratio (smallest
    |weighted term| / median |weighted term|) drops below this (a term has stopped
    contributing). ``boundary_threshold`` — a metric "jumps" at the boundary if its
    step there exceeds this multiple of the surrounding-epoch median step.
    ``comparable_frac`` — a consistent metric "genuinely degrades comparably to
    val_total" only if its boundary step ratio is at least this fraction of
    val_total's (guards against a tiny wiggle on a near-flat baseline reading as a
    huge relative step).
    """
    if not records:
        raise ValueError("no metrics records to analyze")
    epochs = [int(r['epoch']) for r in records]
    weighted = [epoch_weighted_terms(r) for r in records]
    collapse = [term_collapse(t) for t in weighted]
    collapse_flagged = [ep for ep, c in zip(epochs, collapse)
                        if c < collapse_threshold]

    b = boundary_epoch(records)
    vt = _val_series(records, 'val_total')
    vt_boundary = boundary_step_ratio(epochs, vt, b)
    vt_ratio = vt_boundary['ratio'] if vt_boundary is not None else None
    vt_flagged = vt_ratio is not None and vt_ratio > boundary_threshold

    consistent = {}
    for key in _CONSISTENT:
        consistent[key] = boundary_step_ratio(epochs, _val_series(records, key), b)

    # A consistent metric "genuinely degrades comparably to val_total" only if its
    # OWN relative boundary step is significant (> boundary_threshold) AND a sizable
    # fraction of val_total's jump. The comparability guard is what stops a tiny
    # absolute wiggle on a near-flat consistent metric (huge relative ratio, but a
    # negligible step) from being read as real degradation.
    consistent_degrades = []
    for key, info in consistent.items():
        if info is None or not np.isfinite(info['ratio']):
            continue
        r = info['ratio']
        comparable = vt_ratio is not None and r >= comparable_frac * vt_ratio
        if r > boundary_threshold and comparable:
            consistent_degrades.append((key, r))

    # Artifact = val_total jumps at the boundary but no consistent metric degrades
    # comparably -> the jump is a yardstick artifact, not real regression.
    artifact = bool(vt_flagged and not consistent_degrades)

    # Default to the boundary-consistent intensity metric (val_total changes
    # character at the switch). Fall back to val_total ONLY if a consistent metric
    # genuinely degrades comparably — i.e. the boundary step is real. The default
    # keys off whether the metric is LOGGED (a degenerate/infinite boundary-step
    # ratio from a flat baseline must not disqualify it — that is the benign case).
    logmse_available = any(np.isfinite(v)
                           for v in _val_series(records, 'val_intensity_logmse'))
    if consistent_degrades:
        recommended = 'val_total'
    elif logmse_available:
        recommended = 'val_intensity_logmse'
    else:
        recommended = 'val_detection_f1'

    return {
        'epochs': epochs,
        'boundary_epoch': b,
        'weighted': weighted,
        'collapse': collapse,
        'collapse_threshold': float(collapse_threshold),
        'collapse_epoch0': collapse[0],
        'collapse_last': collapse[-1],
        'collapse_min': min(collapse),
        'collapse_flagged_epochs': collapse_flagged,
        'boundary_threshold': float(boundary_threshold),
        'comparable_frac': float(comparable_frac),
        'val_total_boundary': vt_boundary,
        'val_total_flagged': bool(vt_flagged),
        'consistent': consistent,
        'consistent_degrades': consistent_degrades,
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
    lines.append(f"  collapse ratio (min|term|/median|term|) epoch0 = "
                 f"{a['collapse_epoch0']:.3f}  ->  last = {a['collapse_last']:.3f}   "
                 f"(min {a['collapse_min']:.3f}, threshold {a['collapse_threshold']:.2f})")
    lines.append("  (intensity terms dominating / going negative in Phase 2 is "
                 "EXPECTED; we only flag a term collapsing toward 0)")
    if a['collapse_flagged_epochs']:
        lines.append(f"  FLAG: a term collapsed (< {a['collapse_threshold']:.2f}) at "
                     f"epochs {a['collapse_flagged_epochs']}")
    balance_ok = not a['collapse_flagged_epochs']

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
    v1 = ("(a) term balance: HEALTHY - no term collapsed (all epochs >= "
          f"{a['collapse_threshold']:.2f})" if balance_ok else
          f"(a) term balance: CHECK - a term collapsed (< {a['collapse_threshold']:.2f}) "
          f"at {a['collapse_flagged_epochs']}")
    if not a['val_total_flagged']:
        v2 = "(b) boundary: SMOOTH — val_total does not jump at the boundary"
    elif a['artifact']:
        v2 = ("(b) boundary: val_total JUMPS but no consistent metric degrades "
              "comparably -> ARTIFACT (use a boundary-consistent early-stop metric)")
    else:
        degraded = ', '.join(k for k, _ in a['consistent_degrades'])
        v2 = ("(b) boundary: val_total jumps AND consistent metric(s) degrade "
              f"comparably ({degraded}) -> likely REAL degradation, investigate")
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
    ap.add_argument('--collapse-threshold', type=float, default=0.05,
                    help='Flag an epoch if min|term|/median|term| falls below this.')
    ap.add_argument('--boundary-threshold', type=float, default=3.0)
    ap.add_argument('--comparable-frac', type=float, default=0.5)
    args = ap.parse_args()

    records = load_metrics(args.run)
    a = analyze(records, collapse_threshold=args.collapse_threshold,
                boundary_threshold=args.boundary_threshold,
                comparable_frac=args.comparable_frac)
    print(format_verdict(a))

    out_dir = args.out or (Path(args.run) / 'diag')
    written = make_plots(records, a, out_dir)
    for p in written:
        print(f"[diagnostic] wrote {p}")


if __name__ == '__main__':
    main()
