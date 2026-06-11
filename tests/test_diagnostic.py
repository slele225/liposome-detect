"""diagnostic: term-collapse + boundary-step analysis over a synthetic metrics log.

Covers the two hardened flags:
  (a) term balance via ``term_collapse`` (smallest |weighted term| / median): a term
      merely DOMINATING (or NLL terms going negative) does NOT flag — only a term
      collapsing toward zero does;
  (b) the boundary recommendation defaults to ``val_intensity_logmse`` and falls back
      to ``val_total`` only when a consistent metric genuinely degrades COMPARABLY to
      val_total at the boundary (a tiny wiggle amplified by a flat baseline does not).
"""

import json

import pytest

from src.train.diagnostic import (
    analyze,
    boundary_epoch,
    boundary_step_ratio,
    epoch_weighted_terms,
    load_metrics,
    term_collapse,
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


def _wrec(epoch, weighted, val_total, f1, logmse, use_nll):
    """Record carrying an explicit ``train_weighted`` block (controls term signs)."""
    return {
        'epoch': epoch, 'use_nll': use_nll, 'weights': dict(BASE_W),
        'train_weighted': dict(weighted),
        'train': {'total': sum(weighted.values()), **weighted},
        'val': {'val_total': val_total, 'val_detection_f1': f1,
                'val_intensity_logmse': logmse},
    }


def _synthetic_records():
    """A run where val_total jumps at the boundary while f1 / logmse stay smooth (an
    artifact). Term magnitudes are balanced/dominant but NEVER collapse, so the term
    flag should stay clear."""
    recs = []
    for ep in range(N_EPOCHS):
        use_nll = ep >= BOUNDARY
        # epoch 0: heatmap DOMINATES (weighted 30 vs 2) — dominance is not a collapse.
        if ep == 0:
            parts = {'heatmap': 100.0, 'offset': 2.0, 'lipid': 2.0, 'protein': 2.0}
        else:
            parts = {'heatmap': 6.67, 'offset': 2.0, 'lipid': 2.0, 'protein': 2.0}
        val_total = 5.0 - 0.05 * ep + (2.0 if ep >= BOUNDARY else 0.0)
        f1 = 0.30 + 0.03 * ep
        logmse = 1.0 - 0.05 * ep
        recs.append(_rec(ep, parts, dict(BASE_W), val_total, f1, logmse, use_nll))
    return recs


def test_boundary_epoch_detected():
    assert boundary_epoch(_synthetic_records()) == BOUNDARY


# --------------------------------------------------------------------------- #
# (a) term collapse                                                            #
# --------------------------------------------------------------------------- #
def test_term_collapse_balanced_and_dominant_are_healthy():
    # All equal -> 1.0.
    assert term_collapse({'heatmap': 2.0, 'offset': 2.0,
                          'lipid': 2.0, 'protein': 2.0}) == pytest.approx(1.0)
    # One term DOMINATING (others equal) -> still 1.0 (dominance is not collapse).
    assert term_collapse({'heatmap': 30.0, 'offset': 2.0,
                          'lipid': 2.0, 'protein': 2.0}) == pytest.approx(1.0)


def test_term_collapse_robust_to_negative_nll_terms():
    # Phase-2 NLL terms go negative but stay substantial -> not a collapse.
    c = term_collapse({'heatmap': 0.6, 'offset': 0.4, 'lipid': -3.0, 'protein': -2.5})
    assert c > 0.05


def test_term_collapse_detects_genuine_collapse():
    # One term ~0 while others substantial -> collapsed (ratio << threshold).
    c = term_collapse({'heatmap': 1e-4, 'offset': 1.0, 'lipid': -2.0, 'protein': 1.5})
    assert c < 0.05


def test_epoch_weighted_terms_from_parts_and_block():
    rec0 = _synthetic_records()[0]
    terms = epoch_weighted_terms(rec0)
    assert terms['heatmap'] == pytest.approx(0.3 * 100.0)   # weight * unweighted part
    assert terms['offset'] == pytest.approx(2.0)
    rec0['train_weighted'] = {'heatmap': 99.0, 'offset': 1.0,
                              'lipid': 1.0, 'protein': 1.0}
    assert epoch_weighted_terms(rec0)['heatmap'] == pytest.approx(99.0)  # block wins


def test_analyze_does_not_flag_negative_or_dominant_terms():
    """Negative NLL terms + a dominant heatmap epoch -> NO false collapse flag."""
    recs = []
    for ep in range(8):
        use_nll = ep >= 4
        if not use_nll:
            w = {'heatmap': 2.0, 'offset': 2.0, 'lipid': 2.0, 'protein': 2.0}
        else:                                   # intensity terms lead AND go negative
            w = {'heatmap': 0.6, 'offset': 0.4, 'lipid': -3.0, 'protein': -2.5}
        recs.append(_wrec(ep, w, 5.0 - 0.05 * ep, 0.3 + 0.02 * ep, 1.0 - 0.04 * ep, use_nll))
    a = analyze(recs)
    assert a['collapse_flagged_epochs'] == []


def test_analyze_flags_genuine_term_collapse():
    recs = []
    for ep in range(8):
        use_nll = ep >= 4
        if ep == 6:                             # protein head collapses at epoch 6
            w = {'heatmap': 1.0, 'offset': 1.0, 'lipid': 1.0, 'protein': 1e-4}
        else:
            w = {'heatmap': 1.0, 'offset': 1.0, 'lipid': 1.0, 'protein': 1.0}
        recs.append(_wrec(ep, w, 5.0 - 0.05 * ep, 0.3 + 0.02 * ep, 1.0 - 0.04 * ep, use_nll))
    a = analyze(recs)
    assert 6 in a['collapse_flagged_epochs']


# --------------------------------------------------------------------------- #
# (b) boundary recommendation                                                  #
# --------------------------------------------------------------------------- #
def test_boundary_step_ratio_math():
    epochs = list(range(N_EPOCHS))
    vt = [r['val']['val_total'] for r in _synthetic_records()]
    info = boundary_step_ratio(epochs, vt, BOUNDARY)
    # step across boundary ~ |(-0.05) + 2.0| = 1.95; surrounding |delta| ~ 0.05.
    assert info['boundary_step'] == pytest.approx(1.95, abs=1e-6)
    assert info['surrounding_median'] == pytest.approx(0.05, abs=1e-6)
    assert info['ratio'] > 3.0


def test_analyze_boundary_artifact_recommends_logmse():
    a = analyze(_synthetic_records())
    assert a['val_total_flagged'] is True       # val_total jumps at the boundary
    assert a['consistent_degrades'] == []       # f1/logmse stay smooth
    assert a['artifact'] is True
    assert a['recommended_metric'] == 'val_intensity_logmse'


def test_analyze_smooth_boundary_still_recommends_logmse():
    """No val_total jump -> default to the boundary-consistent val_intensity_logmse
    (val_total is no longer the default)."""
    recs = _synthetic_records()
    for ep, r in enumerate(recs):
        r['val']['val_total'] = 5.0 - 0.05 * ep         # remove the boundary step
    a = analyze(recs)
    assert a['val_total_flagged'] is False
    assert a['artifact'] is False
    assert a['recommended_metric'] == 'val_intensity_logmse'


def test_analyze_tiny_consistent_wiggle_is_not_real_degradation():
    """A flat consistent metric with a tiny absolute wiggle at the boundary has a big
    RELATIVE ratio but is not comparable to val_total's jump -> still an artifact."""
    recs = _synthetic_records()
    for ep, r in enumerate(recs):
        # logmse essentially flat, with a negligible 0.002 wiggle exactly at boundary.
        r['val']['val_intensity_logmse'] = 0.500 + (0.002 if ep == BOUNDARY else 0.0)
    a = analyze(recs)
    assert a['recommended_metric'] == 'val_intensity_logmse'
    assert a['artifact'] is True


def test_analyze_consistent_degrades_comparably_recommends_val_total():
    """val_total jumps AND logmse jumps comparably at the boundary -> real
    degradation, keep val_total."""
    recs = _synthetic_records()
    for ep, r in enumerate(recs):
        # logmse drifts 0.05/epoch but STEPS by 1.5 at the boundary (comparable to
        # val_total's ~2.0 step).
        r['val']['val_intensity_logmse'] = (1.0 - 0.05 * ep
                                            + (1.5 if ep >= BOUNDARY else 0.0))
    a = analyze(recs)
    assert a['val_total_flagged'] is True
    assert any(k == 'val_intensity_logmse' for k, _ in a['consistent_degrades'])
    assert a['artifact'] is False
    assert a['recommended_metric'] == 'val_total'


def test_load_metrics_roundtrip(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    recs = _synthetic_records()
    (run / 'metrics.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in recs) + '\n')
    loaded = load_metrics(run)
    assert [r['epoch'] for r in loaded] == list(range(N_EPOCHS))
