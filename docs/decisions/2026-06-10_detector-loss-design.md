# Detector loss design (2026-06-10)

Decision record for the detector's training loss. Companion records:
`2026-06-04_synthetic-generation-strategy.md` (the data this trains on) and
`2026-06-04_protein-channel-parameterization.md` (the alpha-agnostic invariant
this loss must preserve). Exact numeric knobs (focal exponents, size-weight bounds,
eps floors, NLL warmup epochs, term weights) live in each run's `config_snapshot/`;
this doc fixes the SHAPE and the reasoning, not the numbers.

Three additive terms, each doing one job. Small-liposome prioritization enters
ONLY in safe places (detectability + fractional intensity accuracy), never in a
way that leaks the curvature law (an intensity-vs-diameter relationship) into the
training objective. This preserves the alpha-agnostic guarantee from
`docs/decisions/2026-06-04_protein-channel-parameterization.md`.

total_loss = w_hm * heatmap_loss
           + w_int * intensity_nll_logspace   # lipid + protein
           + (uncertainty is folded into the NLL term, not separate)

## 1. Heatmap loss — detection (penalty-reduced focal + size weight)

Per-pixel center-detection map. Ground-truth centers splatted as Gaussian bumps
(sigma tied to spot footprint). CornerNet/CenterNet-style penalty-reduced focal
loss so (a) easy background pixels stop dominating, (b) near-center near-misses
are penalized softly.

SAFE size prioritization: weight each ground-truth spot's contribution by a
BOUNDED decreasing function of diameter, e.g.
    s_weight = clip(d_ref / d, 1.0, w_max)        # d_ref ~ 100 nm, w_max ~ 4-6
or  s_weight = 1 + log(clip(d_ref / d, 1.0, .))   # gentler
This up-weights SMALL-spot DETECTABILITY only. It touches location, not intensity,
so it cannot encode the curvature law. MUST be clamped (raw 1/d explodes for the
~40-80 nm tail) and smoke-tested on the real diameter range.

Pseudocode (per positive pixel p belonging to spot k, alpha_f/beta_f focal exps):
    if target == 1:
        L_p = s_weight_k * (1 - pred)^alpha_f * log(pred)
    else:  # near/far negative, (1 - gaussian_bump) is the penalty reduction
        L_p = (1 - target)^beta_f * pred^alpha_f * log(1 - pred)
    heatmap_loss = -sum(L_p) / num_positive_spots

## 2. Intensity loss — LOG SPACE (the validated approach)

Equalizes FRACTIONAL accuracy across brightness, so dim small-liposome spots
matter as much as bright ones — WITHOUT referencing diameter, so no curvature
leakage. Applied per channel (lipid, protein) at matched/true centroids.

    r = log(pred_flux + eps) - log(true_flux + eps)

eps = a small floor relative to the dimmest REAL flux (NOT an arbitrary 1e-8).
Never feed background / zero-flux into the log. Troubleshooting note: prior
instability here is almost always (a) eps too small or unset, or (b) zeros/
background entering the log.

## 3. Uncertainty — heteroscedastic NLL in log space (folds into term 2)

Network predicts per-spot intensity mean AND log-variance. Train the intensity
term as Gaussian NLL on the LOG residual (keeps everything on the fractional
scale, consistent with term 2):

    intensity_nll_logspace = 0.5 * ( r^2 / sigma2 + log(sigma2) )
    # r is the log-residual from term 2; sigma2 = exp(pred_log_var)

Rewards honest large sigma where genuinely uncertain (dim/small spots), punishes
false confidence. The per-spot sigma propagates DOWNSTREAM as the weight in the
log-log sorting-curve fit (small uncertain spots contribute less to alpha) — this
is the principled handling of "small liposomes are measured less precisely."

Variance-collapse guard: warm up with plain log-space MSE (r^2) for the first N
epochs (default 5), THEN switch on the full NLL. Optionally beta-NLL variant.
This is a LOSS schedule, separate from the LR warmup (linear->cosine).

## Why small-liposome priority is SAFE here (the key invariant)

- Detection size-weight (term 1): acts on LOCATION. Up-weighting small-spot
  detectability says nothing about intensity, so it cannot encode alpha.
- Log-space intensity (terms 2-3): acts on FRACTIONAL intensity accuracy via the
  intensity SCALE, not via diameter. No diameter term multiplies the intensity
  magnitude.
- FORBIDDEN: a diameter-dependent weight on the INTENSITY loss. That would couple
  "how much we care about a spot's intensity" to its size, i.e. a backdoor
  curvature prior, biasing the exact measurement (alpha) we need unbiased.

## Validation requirement

If a size-weight is added to detection, CONFIRM on the synthetic alpha-recovery
sweep that it improves small-radius detection fraction WITHOUT biasing recovered
alpha (the two-axis test). If recovered alpha shifts when the size-weight is
turned on, the weight is leaking — reduce w_max or remove it.
