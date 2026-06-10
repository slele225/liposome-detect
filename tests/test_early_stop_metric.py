"""EarlyStopping metric selection (min/max) + boundary burn-in.

``early_stop_metric`` chooses the val key AND its improvement direction (via
``EARLY_STOP_METRICS``); ``early_stop_burnin_epochs`` defers bad-epoch counting
until that many epochs PAST the NLL/emphasis boundary so a metric artifact there
cannot trip early stopping.
"""

import pytest

from src.train.engine import (
    EARLY_STOP_METRICS,
    EarlyStopping,
    resolve_early_stopping,
)


def test_metric_directions_mapping():
    # val_total / logmse are lower-better; detection F1 is higher-better.
    assert EARLY_STOP_METRICS['val_total'] == 'min'
    assert EARLY_STOP_METRICS['val_intensity_logmse'] == 'min'
    assert EARLY_STOP_METRICS['val_detection_f1'] == 'max'


def test_resolve_defaults_to_val_total_min():
    es = resolve_early_stopping({}, nll_warmup=5)
    assert es['metric'] == 'val_total' and es['mode'] == 'min'
    assert es['burnin_until'] == -1            # no burn-in by default
    assert es['enabled'] is True


def test_resolve_selects_metric_and_mode():
    es = resolve_early_stopping({'early_stop_metric': 'val_detection_f1'},
                                nll_warmup=5)
    assert es['metric'] == 'val_detection_f1' and es['mode'] == 'max'


def test_resolve_legacy_metric_alias():
    es = resolve_early_stopping({'metric': 'val_intensity_logmse'}, nll_warmup=3)
    assert es['metric'] == 'val_intensity_logmse' and es['mode'] == 'min'


def test_resolve_burnin_is_absolute_epoch():
    es = resolve_early_stopping({'early_stop_burnin_epochs': 4}, nll_warmup=5)
    assert es['burnin_until'] == 9            # boundary(5) + burnin(4)


def test_resolve_rejects_unknown_metric():
    with pytest.raises(ValueError):
        resolve_early_stopping({'early_stop_metric': 'val_nonsense'}, nll_warmup=5)


def test_max_mode_treats_higher_as_better():
    es = EarlyStopping(patience=2, mode='max')
    improved, stop = es.update(0.30, 0)              # first is best
    assert improved and not stop
    assert es.update(0.40, 1) == (True, False)       # higher -> improvement
    assert es.update(0.35, 2) == (False, False)      # lower -> bad 1
    assert es.update(0.32, 3) == (False, True)       # bad 2 == patience
    assert es.best == pytest.approx(0.40)
    assert es.best_epoch == 1


def test_burnin_defers_bad_counting_past_boundary():
    # boundary=5, burn-in 5 epochs past -> count bad only from epoch 5.
    es = EarlyStopping(patience=2, mode='min', burnin_until=5)
    es.update(1.0, 0)                                # best
    # epochs 1..4 are worse but inside burn-in -> NOT counted.
    for ep in (1, 2, 3, 4):
        assert es.update(1.5, ep) == (False, False)
        assert es.num_bad == 0
    # from epoch 5 bad epochs count; stop on the 2nd (patience=2).
    assert es.update(1.5, 5) == (False, False)       # bad 1
    assert es.update(1.5, 6) == (False, True)        # bad 2 -> stop


def test_burnin_still_tracks_improvements_during_window():
    es = EarlyStopping(patience=2, mode='min', burnin_until=5)
    es.update(1.0, 0)
    assert es.update(0.5, 3) == (True, False)        # improvement inside burn-in
    assert es.best == pytest.approx(0.5)
    assert es.best_epoch == 3
