"""Train / eval loops, the LR schedule, and per-epoch metric aggregation.

Two SEPARATE warmups (do not conflate):
  - LR schedule  : linear warmup then cosine decay to ~0 (``build_scheduler``),
    stepped per optimizer step. Stabilizes AdamW's early second moments.
  - Loss warmup  : intensity terms are plain log-space MSE for the first
    ``nll_warmup_epochs`` epochs, then the full heteroscedastic NLL. Decided by
    ``use_nll`` passed into ``train_one_epoch`` / ``compute_losses``. Prevents
    variance collapse in the uncertainty heads.

The MSE->NLL switch is ALSO the emphasis-schedule boundary (``emphasis_weights``):
ONE conceptual phase, not two. Phase 1 (detection-led, MSE) uses the static config
weights; Phase 2 (intensity-led, NLL) ramps ``w_lip``/``w_pro`` up to their phase-2
targets. The TRAIN objective is reweighted by the schedule; the VAL loss keeps the
static config weights as a fixed early-stopping yardstick (``EarlyStopping``).
"""

import math

import torch

from src.models.decode import decode_batch
from src.train.dataset import targets_to_device
from src.train.losses import compute_losses
from src.train.metrics import (
    intensity_log_error,
    localization_error,
    match_detections,
    precision_recall_f1,
)


def build_scheduler(optimizer, total_steps, warmup_frac):
    """Linear warmup over ``warmup_frac`` of steps, then cosine decay to ~0.

    Single warmup->cosine, NO warm restarts. SGDR (cosine warm restarts) is the
    deferred NSF-run lever if HRNet underfits in a way more data can't fix.
    """
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


_WEIGHT_KEYS = ('w_hm', 'w_off', 'w_lip', 'w_pro')


def emphasis_weights(epoch, nll_warmup, base_weights, phase2, ramp_epochs):
    """Per-epoch loss weights under the heatmap-led -> intensity-led schedule.

    Phase 1 (``epoch < nll_warmup``, the MSE loss-warmup): the static ``base_weights``
    (detection-led). Phase 2 (``epoch >= nll_warmup``, NLL on): ``w_lip``/``w_pro``
    ramp LINEARLY from their phase-1 values to ``phase2`` targets over ``ramp_epochs``,
    starting at the NLL switch; ``w_hm``/``w_off`` are held fixed. The schedule shares
    the MSE->NLL boundary so there is ONE phase boundary, not two.

    Setting ``phase2`` targets equal to the phase-1 values => static weights (ramp
    OFF). ``frac`` reaches 1.0 ``ramp_epochs`` epochs after the switch, so the targets
    are held from epoch ``nll_warmup + ramp_epochs - 1`` onward.
    """
    w = {k: float(base_weights[k]) for k in _WEIGHT_KEYS}
    if epoch < nll_warmup:
        return w
    frac = min(1.0, (epoch - nll_warmup + 1) / max(1, int(ramp_epochs)))
    for k in ('w_lip', 'w_pro'):
        target = float(phase2.get(k, base_weights[k]))
        w[k] = float(base_weights[k]) + frac * (target - float(base_weights[k]))
    return w


def schedule_weights(cfg, epoch):
    """Resolve ``emphasis_weights`` for ``epoch`` from a training config."""
    L = cfg['loss']
    base = {k: float(L['weights'][k]) for k in _WEIGHT_KEYS}
    phase2 = {'w_lip': float(L.get('w_lip_phase2', base['w_lip'])),
              'w_pro': float(L.get('w_pro_phase2', base['w_pro']))}
    return emphasis_weights(epoch, int(cfg['nll_warmup_epochs']), base, phase2,
                            int(L.get('emphasis_ramp_epochs', 3)))


class EarlyStopping:
    """Track the best (lowest) validation metric; stop after ``patience`` epochs
    without improvement.

    ``update(metric, epoch)`` returns ``(improved, should_stop)``. ``improved`` is
    True when ``metric`` beats the running best by more than ``min_delta`` (the
    harness then keeps that epoch's checkpoint as the best one); ``best_epoch`` is
    the best-val epoch, NOT the last. Assumes lower-is-better (e.g. ``val_total``).
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.best_epoch = -1
        self.num_bad = 0

    def update(self, metric, epoch):
        if self.best is None or metric < self.best - self.min_delta:
            self.best = float(metric)
            self.best_epoch = int(epoch)
            self.num_bad = 0
            return True, False
        self.num_bad += 1
        return False, self.num_bad >= self.patience


def _loss_kwargs(cfg, weights=None):
    if weights is None:
        weights = {k: float(cfg['loss']['weights'][k]) for k in _WEIGHT_KEYS}
    return dict(
        weights={k: float(weights[k]) for k in _WEIGHT_KEYS},
        focal_alpha=float(cfg['loss']['focal_alpha']),
        focal_beta=float(cfg['loss']['focal_beta']),
        eps_lipid=float(cfg['loss']['eps_lipid']),
        eps_protein=float(cfg['loss']['eps_protein']),
        beta_nll=float(cfg['loss'].get('beta_nll', 0.0)),
    )


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, use_nll,
                    weights=None):
    """One training epoch. Returns mean total + per-term losses.

    ``weights`` (per-epoch, from ``schedule_weights``) overrides the static config
    weights so the emphasis schedule reweights the TRAIN objective; ``None`` falls
    back to the static config weights.
    """
    model.train()
    lk = _loss_kwargs(cfg, weights=weights)
    agg = {'total': 0.0, 'heatmap': 0.0, 'offset': 0.0, 'lipid': 0.0, 'protein': 0.0}
    n = 0
    for images, targets, _meta in loader:
        images = images.to(device)
        targets = targets_to_device(targets, device)
        outputs = model(images)
        total, parts = compute_losses(outputs, targets, use_nll=use_nll, **lk)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if cfg['optim'].get('grad_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg['optim']['grad_clip']))
        optimizer.step()
        scheduler.step()

        agg['total'] += float(total.detach())
        for k, v in parts.items():
            agg[k] += float(v)
        n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Validation: per-term val loss + matched-F1 + intensity log-error."""
    model.eval()
    lk = _loss_kwargs(cfg)
    dec = cfg['decode']
    radius = float(cfg['eval']['match_radius'])

    agg = {'total': 0.0, 'heatmap': 0.0, 'offset': 0.0, 'lipid': 0.0, 'protein': 0.0}
    n = 0
    tp = fp = fn = 0
    loc, lip_e, pro_e = [], [], []
    for images, targets, meta in loader:
        images = images.to(device)
        targets_d = targets_to_device(targets, device)
        outputs = model(images)
        total, parts = compute_losses(outputs, targets_d, use_nll=True, **lk)
        agg['total'] += float(total)
        for k, v in parts.items():
            agg[k] += float(v)
        n += 1

        dets = decode_batch(
            {k: v.cpu() for k, v in outputs.items()}, model.out_stride,
            score_threshold=float(dec['score_threshold']),
            nms_kernel=int(dec['nms_kernel']),
            max_detections=dec.get('max_detections'))
        for preds, gts in zip(dets, meta['spots']):
            matches, b_tp, b_fp, b_fn = match_detections(preds, gts, radius)
            tp += b_tp
            fp += b_fp
            fn += b_fn
            if matches:
                loc.append(localization_error(preds, gts, matches))
                le, pe = intensity_log_error(preds, gts, matches,
                                             lk['eps_lipid'], lk['eps_protein'])
                lip_e.append(le)
                pro_e.append(pe)

    p, r, f1 = precision_recall_f1(tp, fp, fn)
    nan = float('nan')
    return {
        'val_total': agg['total'] / max(1, n),
        'val_heatmap': agg['heatmap'] / max(1, n),
        'val_offset': agg['offset'] / max(1, n),
        'val_lipid': agg['lipid'] / max(1, n),
        'val_protein': agg['protein'] / max(1, n),
        'precision': p, 'recall': r, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'loc_error': float(sum(loc) / len(loc)) if loc else nan,
        'lipid_log_error': float(sum(lip_e) / len(lip_e)) if lip_e else nan,
        'protein_log_error': float(sum(pro_e) / len(pro_e)) if pro_e else nan,
    }
