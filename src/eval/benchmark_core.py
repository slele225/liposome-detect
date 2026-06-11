"""Shared, FROZEN photometry + fit + GT-eval layer for the cross-method benchmark.

The benchmark's design principle: every detector produces only (x, y) spot
LOCATIONS; everything downstream is a single shared layer applied IDENTICALLY to
every method. So "method A vs B" reduces purely to "whose LOCATIONS give better
alpha / detection / intensity", with no confound from differing photometry or fits.
External tools (cme-analysis [MATLAB], SpotMAX) participate by emitting a per-image
(x, y) CSV that flows through this same layer — they never run on this instance.

This module is method-agnostic and has THREE pieces:

1. Shared local photometry (``aperture_photometry``): at each location, a fixed-
   radius circular aperture sum minus a local annulus-median background, per channel.

   Consistency with the synthetic GT flux definition (the #1 photometry risk):
   the simulator renders each spot as a PSF kernel NORMALIZED TO SUM TO 1, so GT
   ``lipid_intensity`` / ``protein_intensity`` is the spot's TOTAL integrated flux
   in ADU, fully contained within the render radius ``ceil(4*sigma)`` (~8 px for the
   calibrated sigma~1.9). The default aperture radius is sized to that support, so a
   background-subtracted aperture sum ~ total flux ~ GT. Any residual CONSTANT
   capture fraction (f<1) is the same for every spot of a channel, so it shifts the
   log-log INTERCEPT, never the SLOPE -> alpha is robust to it; only the absolute
   intensity-recovery level carries the (documented) constant offset.

2. Shared fit (``fit_alpha_bootstrap``): pool (lipid, protein) across a sample's
   images, ``alpha = 2*deming_slope`` (constant-lambda EIV / TLS, lam=1, from
   ``alpha_fit.py``), then ``CalibrationCurve.invert``. From intensities ALONE — no
   per-spot logvar weighting (external methods don't provide it, and it is
   contraindicated here anyway). Bootstrap over images for a CI. Identical for all.

3. Shared GT eval (``evaluate_synthetic``, synthetic only): greedy-match detections
   to GT (reuse ``matching.greedy_match``); F1, per-matched intensity-recovery error
   (lipid + protein) and localization error, plus within-bin detection
   representativeness — ALL BINNED BY TRUE DIAMETER, with fine small-diameter bins
   (the curvature-sensing-relevant regime).

Nothing here is detector-specific; the adapters (``src/eval/adapters.py``) feed it.
"""

import numpy as np

from src.eval.alpha_fit import ols_slope, recover_alpha
from src.eval.matching import greedy_match

# Fine at the small (high-curvature) end where a detector edge would matter, coarser
# above. Edges in nm; bin i is [DIAM_EDGES[i], DIAM_EDGES[i+1]).
DIAM_EDGES = np.array([40, 55, 70, 90, 120, 160, 220, 300], float)
DIAM_LABELS = [f"{int(DIAM_EDGES[i])}-{int(DIAM_EDGES[i+1])}"
               for i in range(len(DIAM_EDGES) - 1)]

# Default photometry geometry (px). r_ap sized to the PSF support so the aperture
# captures ~all of the GT total flux (sigma~1.9 -> 99%+ within r=6); the annulus
# sits beyond the support to estimate local background (offset + optical bg + any
# neighbour haze). All three are config-exposed by the callers.
R_AP_DEFAULT = 6.0
R_IN_DEFAULT = 9.0
R_OUT_DEFAULT = 14.0


# --------------------------------------------------------------------------- #
# 1. Shared local photometry                                                  #
# --------------------------------------------------------------------------- #
def aperture_photometry(image, xy, r_ap=R_AP_DEFAULT, r_in=R_IN_DEFAULT,
                        r_out=R_OUT_DEFAULT):
    """Aperture-sum photometry with local annulus background, per channel.

    Args:
        image: ``(2, H, W)`` array, channel 0 = protein, channel 1 = lipid (the
            project's channel order). RAW ADU (offset included) — the annulus
            background subtraction removes the DC offset + optical background.
        xy: ``(N, 2)`` array of ``(x, y)`` locations (full-res px).
        r_ap / r_in / r_out: aperture radius and background annulus inner/outer radii.

    Returns ``(lipid, protein)``: two length-N float arrays of background-subtracted
    integrated intensities (total flux in ADU, ~ the GT definition). Values can go
    slightly negative for spurious detections on noise; callers mask non-positive
    intensities before the log fit.
    """
    image = np.asarray(image, np.float64)
    _, H, W = image.shape
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    R = int(np.ceil(r_out)) + 1
    lip = np.full(len(xy), np.nan)
    pro = np.full(len(xy), np.nan)
    for i in range(len(xy)):
        x, y = xy[i]
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = max(0, ix - R), min(W, ix + R + 1)
        y0, y1 = max(0, iy - R), min(H, iy + R + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - x, yy - y)
        ap = rr <= r_ap
        ann = (rr >= r_in) & (rr <= r_out)
        n_ap = int(ap.sum())
        if n_ap == 0:
            continue
        for c, dst in ((0, 'pro'), (1, 'lip')):
            sub = image[c, y0:y1, x0:x1]
            bg = np.median(sub[ann]) if ann.any() else 0.0
            val = float(sub[ap].sum() - bg * n_ap)
            if dst == 'lip':
                lip[i] = val
            else:
                pro[i] = val
    return lip, pro


# --------------------------------------------------------------------------- #
# 2. Shared fit (identical for every method)                                  #
# --------------------------------------------------------------------------- #
def _alpha_from_pairs(lipid, protein, curve):
    """(standard_ols, recovered_eiv, corrected) from pooled intensity pairs.

    From intensities alone (lam=1 constant-lambda Deming / TLS — no per-spot
    weighting), so it is identical for methods that do and don't emit logvar.
    """
    lipid = np.asarray(lipid, np.float64)
    protein = np.asarray(protein, np.float64)
    m = np.isfinite(lipid) & np.isfinite(protein) & (lipid > 0) & (protein > 0)
    lipid, protein = lipid[m], protein[m]
    if lipid.size < 10 or np.unique(np.log(lipid)).size < 2:
        return float('nan'), float('nan'), float('nan')
    L, P = np.log(lipid), np.log(protein)
    standard = 2.0 * ols_slope(L, P)
    recovered = recover_alpha(L, P)                 # lam=1 (intensities only)
    corrected = float(curve.invert(recovered)) if curve is not None else recovered
    return standard, recovered, corrected


def fit_alpha_bootstrap(per_image_pairs, curve, n_boot=500, rng=None):
    """Pool + bootstrap the shared alpha fit over images.

    Args:
        per_image_pairs: list (per image) of ``(lipid_arr, protein_arr)``.
        curve: a ``CalibrationCurve`` (recovered->true) or None.
        n_boot / rng: bootstrap repeats over images and a numpy Generator.

    Returns a dict with point ``standard`` / ``recovered`` / ``corrected`` and 95%
    bootstrap CIs ``std_ci`` / ``cor_ci``, plus ``n_img`` / ``n_spots``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    groups = [(np.asarray(l, np.float64), np.asarray(p, np.float64))
              for l, p in per_image_pairs if len(l)]
    n_img = len(groups)
    n_spots = int(sum(len(l) for l, _ in groups))
    if n_img == 0:
        nan2 = (float('nan'), float('nan'))
        return dict(standard=float('nan'), recovered=float('nan'),
                    corrected=float('nan'), std_ci=nan2, cor_ci=nan2,
                    n_img=0, n_spots=0)

    def pooled(idx):
        L = np.concatenate([groups[i][0] for i in idx])
        P = np.concatenate([groups[i][1] for i in idx])
        return _alpha_from_pairs(L, P, curve)

    std, rec, cor = pooled(range(n_img))
    sb, cb = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, size=n_img)
        s, _r, c = pooled(idx)
        if np.isfinite(s):
            sb.append(s); cb.append(c)
    sb, cb = np.array(sb), np.array(cb)

    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
            if a.size else (float('nan'), float('nan'))

    return dict(standard=std, recovered=rec, corrected=cor,
                std_ci=ci(sb), cor_ci=ci(cb), n_img=n_img, n_spots=n_spots)


# --------------------------------------------------------------------------- #
# 3. Shared GT eval (synthetic only) — all binned by TRUE diameter            #
# --------------------------------------------------------------------------- #
def _bin_index(diam):
    """Index into DIAM_LABELS for a diameter (nm), or -1 if outside the range."""
    if diam < DIAM_EDGES[0] or diam >= DIAM_EDGES[-1]:
        return -1
    return int(np.searchsorted(DIAM_EDGES, diam, side='right') - 1)


def evaluate_synthetic(per_image, match_radius=4.0, r_ap=R_AP_DEFAULT,
                       r_in=R_IN_DEFAULT, r_out=R_OUT_DEFAULT):
    """Diameter-binned detection / intensity / localization / representativeness.

    Args:
        per_image: list of ``(image_array, gt_spots, det_xy)`` where ``image_array``
            is ``(2,H,W)`` raw, ``gt_spots`` is the label list (dicts with x, y,
            diameter_nm, lipid_intensity, protein_intensity), ``det_xy`` is the
            method's ``(M,2)`` detected locations.
        match_radius / r_ap / r_in / r_out: matching radius and photometry geometry.

    Returns a dict:
        per_bin: list (one per DIAM_LABELS) of dict(label, n_gt, recall, f1,
            lipid_logerr, protein_logerr, loc_err, repr_ratio, det_true_protein,
            missed_true_protein)
        global: dict(precision, recall, f1, n_gt, n_det, n_tp)
      where ``*_logerr`` is the median ``log10(measured/true)`` over matched spots in
      the bin (the constant aperture-capture offset shows as the level; size-
      dependence as the trend), ``repr_ratio`` is median true-protein(detected) /
      median true-protein(missed) within the bin (>1 => detector keeps the brighter
      spots = NON-representative; ~1 => representative).
    """
    nb = len(DIAM_LABELS)
    g = {'n_gt': np.zeros(nb, int), 'n_tp': np.zeros(nb, int),
         'lip_le': [[] for _ in range(nb)], 'pro_le': [[] for _ in range(nb)],
         'loc': [[] for _ in range(nb)],
         'det_tp': [[] for _ in range(nb)], 'mis_tp': [[] for _ in range(nb)]}
    tot_gt = tot_det = tot_tp = 0

    for image, gt_spots, det_xy in per_image:
        det_xy = np.asarray(det_xy, np.float64).reshape(-1, 2)
        tot_det += len(det_xy)
        if not gt_spots:
            continue
        gt_xy = np.array([[s['x'], s['y']] for s in gt_spots], np.float64)
        tot_gt += len(gt_spots)
        match = greedy_match(gt_xy, [{'x': float(x), 'y': float(y)}
                                     for x, y in det_xy], match_radius)
        # Photometry at matched detections (for intensity recovery).
        matched_det = np.array([det_xy[j] for j in match if j >= 0]).reshape(-1, 2)
        lip_m, pro_m = (aperture_photometry(image, matched_det, r_ap, r_in, r_out)
                        if len(matched_det) else (np.array([]), np.array([])))
        k = 0
        for i, s in enumerate(gt_spots):
            bi = _bin_index(float(s['diameter_nm']))
            if bi < 0:
                if match[i] >= 0:
                    k += 1
                continue
            g['n_gt'][bi] += 1
            tp = match[i] >= 0
            if tp:
                tot_tp += 1
                g['n_tp'][bi] += 1
                g['det_tp'][bi].append(float(s['protein_intensity']))
                j = match[i]
                # localization error
                g['loc'][bi].append(float(np.hypot(det_xy[j, 0] - s['x'],
                                                    det_xy[j, 1] - s['y'])))
                # intensity recovery (measured aperture vs GT total flux)
                ml, mp = lip_m[k], pro_m[k]
                k += 1
                if ml > 0 and s['lipid_intensity'] > 0:
                    g['lip_le'][bi].append(np.log10(ml / s['lipid_intensity']))
                if mp > 0 and s['protein_intensity'] > 0:
                    g['pro_le'][bi].append(np.log10(mp / s['protein_intensity']))
            else:
                g['mis_tp'][bi].append(float(s['protein_intensity']))

    precision = tot_tp / tot_det if tot_det else 0.0
    recall = tot_tp / tot_gt if tot_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    def med(a):
        return float(np.median(a)) if len(a) else float('nan')

    per_bin = []
    for bi in range(nb):
        n_gt = int(g['n_gt'][bi])
        r_bin = g['n_tp'][bi] / n_gt if n_gt else float('nan')
        # diameter-resolved F1 proxy: per-bin recall against GLOBAL precision
        # (false positives carry no true diameter, so precision can't be binned).
        f1_bin = (2 * precision * r_bin / (precision + r_bin)
                  if np.isfinite(r_bin) and (precision + r_bin) > 0 else float('nan'))
        det_tp, mis_tp = med(g['det_tp'][bi]), med(g['mis_tp'][bi])
        repr_ratio = (det_tp / mis_tp
                      if np.isfinite(det_tp) and np.isfinite(mis_tp) and mis_tp > 0
                      else float('nan'))
        per_bin.append(dict(
            label=DIAM_LABELS[bi], n_gt=n_gt, recall=r_bin, f1=f1_bin,
            lipid_logerr=med(g['lip_le'][bi]), protein_logerr=med(g['pro_le'][bi]),
            loc_err=med(g['loc'][bi]), repr_ratio=repr_ratio,
            det_true_protein=det_tp, missed_true_protein=mis_tp))

    return dict(per_bin=per_bin,
                **{'global': dict(precision=precision, recall=recall, f1=f1,
                                  n_gt=tot_gt, n_det=tot_det, n_tp=tot_tp)})
