"""diagnostic: term-spread + boundary-step analysis over a synthetic metrics log.

Plants (a) a >10x weighted-term spread at epoch 0 and (b) a val_total jump at the
NLL boundary that is a METRIC ARTIFACT (the boundary-consistent metrics stay
smooth) and checks the analyzer flags both and recommends a consistent metric.
"""

import json

import pytest

from src.train.diagnostic import (
    analyze,
    boundary_epoch,
    boundary_step_ratio,
    epoch_weighted_terms,
    load_metrics,
    term_spread,
)

BOUNDARY = 5
N_EPOCHS = 11
BASE_W = {'w_hm': 0.3, 'w_off': 1.0, 'w_lip': 1.0, 'w_pro': 1.0}


def _rec(epoch, parts, weights, val_total, f1, logmse, use_nll):
    return {
        'epoch': epoch, 'use_nll': use_nll, 'weights': weights,
        'lr': 1e-4, 'train': {'total': sum(parts.values()), **parts},
        'val': {'val_total': val_total, 'val_detection_f1': f1,
                'val_intensity_logmse': logmse},
    }


def _synthetic_records():
    """A run where epoch 0 has a huge term imbalance and val_total jumps at the
    boundary while f1 / logmse stay smooth (an artifact)."""
    recs = []
    for ep in range(N_EPOCHS):
        use_nll = ep >= BOUNDARY
        # Balanced weighted terms (~2 each) EXCEPT epoch 0 where heatmap dominates.
        if ep == 0:
            # weighted heatmap = 0.3 * 100 = 30; others ~2 -> spread 15x (>10x).
            parts = {'heatmap': 100.0, 'offset': 2.0, 'lipid': 2.0, 'protein': 2.0}
        else:
            parts = {'heatmap': 6.67, 'offset': 2.0, 'lipid': 2.0, 'protein': 2.0}
        # val_total drifts slowly (~0.05/epoch) but STEPS by ~2.0 at the boundary.
        val_total = 5.0 - 0.05 * ep + (2.0 if ep >= BOUNDARY else 0.0)
        # Boundary-consistent metrics: smooth monotone, no step at the boundary.
        f1 = 0.30 + 0.03 * ep
        logmse = 1.0 - 0.05 * ep
        recs.append(_rec(ep, parts, dict(BASE_W), val_total, f1, logmse, use_nll))
    return recs


def test_boundary_epoch_detected():
    assert boundary_epoch(_synthetic_records()) == BOUNDARY


def test_term_spread_and_weighted_terms():
    rec0 = _synthetic_records()[0]
    terms = epoch_weighted_terms(rec0)
    assert terms['heatmap'] == pytest.approx(0.3 * 100.0)   # weight * unweighted part
    assert terms['offset'] == pytest.approx(2.0)
    # spread = max(|30|) / min(|2|) = 15x
    assert term_spread(terms) == pytest.approx(30.0 / 2.0)


def test_train_weighted_block_is_preferred_when_present():
    rec = _synthetic_records()[1]
    rec['train_weighted'] = {'heatmap': 99.0, 'offset': 1.0,
                             'lipid': 1.0, 'protein': 1.0}
    terms = epoch_weighted_terms(rec)
    assert terms['heatmap'] == pytest.approx(99.0)          # read, not recomputed


def test_boundary_step_ratio_math():
    epochs = list(range(N_EPOCHS))
    vt = [r['val']['val_total'] for r in _synthetic_records()]
    info = boundary_step_ratio(epochs, vt, BOUNDARY)
    # step across boundary ~ |(-0.05) + 2.0| = 1.95; surrounding |delta| ~ 0.05.
    assert info['boundary_step'] == pytest.approx(1.95, abs=1e-6)
    assert info['surrounding_median'] == pytest.approx(0.05, abs=1e-6)
    assert info['ratio'] > 3.0


def test_analyze_flags_spread_and_boundary_artifact():
    a = analyze(_synthetic_records(), spread_threshold=10.0, boundary_threshold=3.0)
    # (1) planted >10x spread at epoch 0 is flagged.
    assert 0 in a['spread_flagged_epochs']
    assert a['spread_epoch0'] > 10.0
    assert a['spread_last'] < 10.0                          # later epochs balanced
    # (2) val_total jumps at the boundary, consistent metrics smooth -> artifact.
    assert a['val_total_flagged'] is True
    assert a['consistent_smooth'] is True
    assert a['artifact'] is True
    assert a['recommended_metric'] == 'val_intensity_logmse'


def test_analyze_smooth_boundary_keeps_val_total():
    """No planted jump -> boundary smooth -> recommend val_total."""
    recs = _synthetic_records()
    for ep, r in enumerate(recs):                           # remove the boundary step
        r['val']['val_total'] = 5.0 - 0.05 * ep
        if ep == 0:                                          # also remove the spread
            r['train'] = {'total': 12.0, 'heatmap': 6.67,
                          'offset': 2.0, 'lipid': 2.0, 'protein': 2.0}
    a = analyze(recs, spread_threshold=10.0, boundary_threshold=3.0)
    assert a['val_total_flagged'] is False
    assert a['artifact'] is False
    assert a['recommended_metric'] == 'val_total'
    assert a['spread_flagged_epochs'] == []


def test_load_metrics_roundtrip(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    recs = _synthetic_records()
    (run / 'metrics.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in recs) + '\n')
    loaded = load_metrics(run)
    assert [r['epoch'] for r in loaded] == list(range(N_EPOCHS))
