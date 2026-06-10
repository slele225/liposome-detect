"""The detector training loss — faithful to 2026-06-10_detector-loss-design.md.

    total = w_hm*heatmap + w_off*offset + w_lip*intensity(lipid) + w_pro*intensity(protein)

  1. heatmap : CenterNet penalty-reduced focal loss, with a BOUNDED per-spot size
     weight on the POSITIVE term only (detection of small spots). The size weight
     enters via ``pos_weight_map`` (built in targets.py) and touches ONLY this
     term — never the intensity losses (the alpha-agnostic invariant).
  2. offset  : L1 on subpixel (dx, dy) at GT centers.
  3-4. intensity (per channel) : LOG-SPACE heteroscedastic NLL at GT centers,
     ``r = log(pred_mean+eps) - log(true+eps)``,
     ``nll = 0.5*(r^2/sigma2 + log(sigma2))``, ``sigma2 = exp(pred_logvar)``.
     During the loss warmup it is plain log-space MSE (``r^2``) instead, to avoid
     variance collapse. ``eps`` is a per-channel floor (lipid and protein live on
     different noise regimes). NOTE: intensity takes NO diameter argument — by
     construction it cannot encode the curvature law.

NOTE vs the loss-design doc: the doc enumerates three terms (heatmap, intensity,
folded-in uncertainty). The offset (L1) term is added here because the schema/
decode require subpixel localization (CenterNet's standard offset head); it acts
on LOCATION only and does not reference intensity or diameter, so it preserves the
doc's invariants. Surfaced in the PROMPT 2 report with a proposed one-line doc note.
"""

import torch
import torch.nn.functional as F

_HM_EPS = 1e-4


def penalty_reduced_focal_loss(pred, target, pos_weight_map=None,
                               focal_alpha=2.0, focal_beta=4.0):
    """CenterNet penalty-reduced focal loss.

    Args:
        pred: (B,1,H,W) probability in (0,1).
        target: (B,1,H,W) Gaussian-splatted target, ==1 at centers.
        pos_weight_map: (B,1,H,W) per-pixel multiplier on the POSITIVE term (the
            size weight; 1 elsewhere). None -> uniform.
        focal_alpha, focal_beta: focal exponents.

    Loss is normalized by the number of positive (center) pixels.
    """
    pred = pred.clamp(_HM_EPS, 1.0 - _HM_EPS)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    neg_weights = (1.0 - target).pow(focal_beta)

    pos_loss = torch.log(pred) * (1.0 - pred).pow(focal_alpha) * pos
    if pos_weight_map is not None:
        pos_loss = pos_loss * pos_weight_map
    neg_loss = torch.log(1.0 - pred) * pred.pow(focal_alpha) * neg_weights * neg

    num_pos = pos.sum()
    pos_sum = pos_loss.sum()
    neg_sum = neg_loss.sum()
    if num_pos == 0:
        return -neg_sum
    return -(pos_sum + neg_sum) / num_pos


def offset_l1_loss(pred_offset, target_offset):
    """L1 between predicted and GT subpixel offset at centers. ``[N,2]`` each."""
    if pred_offset.numel() == 0:
        return pred_offset.new_zeros(())
    return F.l1_loss(pred_offset, target_offset)


def intensity_nll_loss(pred_mean, pred_logvar, true_flux, eps,
                       use_nll=True, beta_nll=0.0):
    """Per-channel log-space intensity loss at GT centers (all ``[N]``).

    ``pred_mean`` is already positive (the head exp's it). Warmup (``use_nll=False``)
    returns plain log-space MSE ``mean(r^2)``; otherwise the heteroscedastic NLL
    ``mean(0.5*(r^2/sigma2 + log(sigma2)))``. ``beta_nll>0`` applies the beta-NLL
    weighting ``sigma2^beta`` (detached) for stability.

    Takes NO diameter — keeping the curvature law out of the intensity objective.
    """
    if pred_mean.numel() == 0:
        return pred_mean.new_zeros(())
    r = torch.log(pred_mean + eps) - torch.log(true_flux + eps)
    if not use_nll:
        return (r * r).mean()
    sigma2 = torch.exp(pred_logvar)
    nll = 0.5 * (r * r / sigma2 + pred_logvar)
    if beta_nll > 0.0:
        nll = nll * (sigma2.detach() ** beta_nll)
    return nll.mean()


def _gather_centers(map_bchw, bidx, iy, ix):
    """Gather per-center values: (B,C,H,W)[bidx, :, iy, ix] -> (N, C)."""
    return map_bchw[bidx, :, iy, ix]


def compute_losses(outputs, targets, weights, focal_alpha=2.0, focal_beta=4.0,
                   eps_lipid=10.0, eps_protein=10.0, use_nll=True, beta_nll=0.0):
    """Assemble the full loss from batched head outputs + collated targets.

    Args:
        outputs: model dict 'heatmap'(B,1,h,w),'offset'(B,2,h,w),'lipid'(B,2,h,w),
            'protein'(B,2,h,w) (lipid/protein channel 0 = positive mean flux).
        targets: collated dict with 'heatmap'(B,1,h,w),'pos_weight_map'(B,1,h,w),
            and center tensors 'bidx','iy','ix' (N,), 'offset'(N,2),
            'lipid'(N,),'protein'(N,).
        weights: dict with 'w_hm','w_off','w_lip','w_pro'.

    Returns ``(total, parts)`` where ``parts`` is a dict of unweighted term values
    (floats-as-tensors) for logging.
    """
    bidx, iy, ix = targets['bidx'], targets['iy'], targets['ix']

    hm = penalty_reduced_focal_loss(
        outputs['heatmap'], targets['heatmap'], targets['pos_weight_map'],
        focal_alpha=focal_alpha, focal_beta=focal_beta)

    pred_off = _gather_centers(outputs['offset'], bidx, iy, ix)      # (N,2)
    off = offset_l1_loss(pred_off, targets['offset'])

    lip_pred = _gather_centers(outputs['lipid'], bidx, iy, ix)        # (N,2)
    pro_pred = _gather_centers(outputs['protein'], bidx, iy, ix)      # (N,2)
    lip = intensity_nll_loss(lip_pred[:, 0], lip_pred[:, 1], targets['lipid'],
                             eps_lipid, use_nll=use_nll, beta_nll=beta_nll)
    pro = intensity_nll_loss(pro_pred[:, 0], pro_pred[:, 1], targets['protein'],
                             eps_protein, use_nll=use_nll, beta_nll=beta_nll)

    total = (weights['w_hm'] * hm + weights['w_off'] * off
             + weights['w_lip'] * lip + weights['w_pro'] * pro)
    parts = {'heatmap': hm.detach(), 'offset': off.detach(),
             'lipid': lip.detach(), 'protein': pro.detach()}
    return total, parts
