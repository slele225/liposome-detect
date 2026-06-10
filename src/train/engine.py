"""Train / eval loops, the LR schedule, and per-epoch metric aggregation.

Two SEPARATE warmups (do not conflate):
  - LR schedule  : linear warmup then cosine decay to ~0 (``build_scheduler``),
    stepped per optimizer step. Stabilizes AdamW's early second moments.
  - Loss warmup  : intensity terms are plain log-space MSE for the first
    ``nll_warmup_epochs`` epochs, then the full heteroscedastic NLL. Decided by
    ``use_nll`` passed into ``train_one_epoch`` / ``compute_losses``. Prevents
    variance collapse in the uncertainty heads.
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
    """Linear warmup over ``warmup_frac`` of steps, then cosine decay to ~0."""
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _loss_kwargs(cfg):
    return dict(
        weights={k: float(cfg['loss']['weights'][k])
                 for k in ('w_hm', 'w_off', 'w_lip', 'w_pro')},
        focal_alpha=float(cfg['loss']['focal_alpha']),
        focal_beta=float(cfg['loss']['focal_beta']),
        eps_lipid=float(cfg['loss']['eps_lipid']),
        eps_protein=float(cfg['loss']['eps_protein']),
        beta_nll=float(cfg['loss'].get('beta_nll', 0.0)),
    )


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, use_nll):
    """One training epoch. Returns mean total + per-term losses."""
    model.train()
    lk = _loss_kwargs(cfg)
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
