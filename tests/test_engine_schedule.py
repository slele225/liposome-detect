"""Emphasis schedule + early stopping (engine.py).

The emphasis schedule and the MSE->NLL loss warmup share ONE phase boundary
(``nll_warmup_epochs``): Phase 1 holds the static config weights; Phase 2 ramps
``w_lip``/``w_pro`` up to their phase-2 targets, with ``w_hm``/``w_off`` fixed.
Early stopping tracks the best (lowest) val metric and keeps the best epoch.
"""

import pytest

from src.train.engine import EarlyStopping, emphasis_weights, schedule_weights

BASE = {'w_hm': 0.3, 'w_off': 1.0, 'w_lip': 1.0, 'w_pro': 1.0}
PHASE2 = {'w_lip': 2.0, 'w_pro': 2.0}
N = 5            # nll_warmup boundary
RAMP = 3


def _w(epoch):
    return emphasis_weights(epoch, N, BASE, PHASE2, RAMP)


def test_phase1_holds_base_weights():
    for epoch in range(N):                       # epochs 0..N-1
        w = _w(epoch)
        assert w == pytest.approx(BASE)


def test_phase2_ramps_linearly_to_targets():
    # frac = (epoch - N + 1) / RAMP, capped at 1.0.
    assert _w(N)['w_lip'] == pytest.approx(1.0 + (1 / RAMP) * 1.0)      # frac 1/3
    assert _w(N)['w_pro'] == pytest.approx(1.0 + (1 / RAMP) * 1.0)
    assert _w(N + 1)['w_lip'] == pytest.approx(1.0 + (2 / RAMP) * 1.0)  # frac 2/3
    # frac saturates at 1.0 -> phase-2 target reached and held.
    assert _w(N + 2)['w_lip'] == pytest.approx(2.0)                     # frac 1.0
    assert _w(N + 10)['w_lip'] == pytest.approx(2.0)
    assert _w(N + 10)['w_pro'] == pytest.approx(2.0)


def test_hm_and_off_held_constant_across_schedule():
    for epoch in (0, N - 1, N, N + 1, N + 5):
        w = _w(epoch)
        assert w['w_hm'] == pytest.approx(BASE['w_hm'])
        assert w['w_off'] == pytest.approx(BASE['w_off'])


def test_static_when_phase2_equals_base():
    # ramp OFF: phase-2 targets == phase-1 values => constant weights everywhere.
    flat = {'w_lip': BASE['w_lip'], 'w_pro': BASE['w_pro']}
    for epoch in (0, N - 1, N, N + 5, N + 100):
        assert emphasis_weights(epoch, N, BASE, flat, RAMP) == pytest.approx(BASE)


def test_schedule_weights_reads_config():
    cfg = {'nll_warmup_epochs': N,
           'loss': {'weights': BASE, 'w_lip_phase2': 2.0, 'w_pro_phase2': 2.0,
                    'emphasis_ramp_epochs': RAMP}}
    assert schedule_weights(cfg, 0) == pytest.approx(BASE)
    assert schedule_weights(cfg, N + 2)['w_lip'] == pytest.approx(2.0)


def test_schedule_weights_defaults_to_static_without_phase2():
    # No phase-2 keys -> targets default to base -> static (ramp OFF).
    cfg = {'nll_warmup_epochs': N, 'loss': {'weights': BASE}}
    assert schedule_weights(cfg, N + 5) == pytest.approx(BASE)


def test_early_stopping_stops_after_patience_non_improving():
    es = EarlyStopping(patience=3)
    improved, stop = es.update(1.0, 0)                  # first is the best
    assert improved and not stop
    # three non-improving epochs in a row -> stop on the third.
    assert es.update(1.5, 1) == (False, False)          # bad 1
    assert es.update(1.5, 2) == (False, False)          # bad 2
    assert es.update(1.5, 3) == (False, True)           # bad 3 == patience


def test_early_stopping_best_is_best_val_not_last():
    es = EarlyStopping(patience=10)
    es.update(1.0, 0)
    es.update(0.4, 1)        # new best at epoch 1
    es.update(0.5, 2)        # worse
    es.update(0.6, 3)        # worse (last epoch)
    assert es.best == pytest.approx(0.4)
    assert es.best_epoch == 1            # the best-val epoch, NOT the last (3)


def test_early_stopping_resets_bad_counter_on_improvement():
    es = EarlyStopping(patience=2)
    es.update(1.0, 0)
    assert es.update(1.1, 1) == (False, False)   # bad 1
    assert es.update(0.9, 2) == (True, False)    # improved -> counter resets
    assert es.num_bad == 0
    assert es.update(1.0, 3) == (False, False)   # bad 1 again, not stop


def test_early_stopping_min_delta_requires_real_improvement():
    es = EarlyStopping(patience=2, min_delta=0.1)
    es.update(1.0, 0)
    # 0.95 is lower but within min_delta -> NOT counted as improvement.
    assert es.update(0.95, 1) == (False, False)
    assert es.best == pytest.approx(1.0)
    # 0.85 clears the 0.1 margin -> improvement.
    assert es.update(0.85, 2) == (True, False)
