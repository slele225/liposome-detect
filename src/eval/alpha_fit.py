"""THE alpha estimator for the whole project — a single source of truth.

The curvature-sensing exponent ``alpha`` is read off the slope of log-protein vs
log-lipid intensity across detected liposomes: ``protein ~ lipid**(alpha/2)`` (lipid
flux ~ d^2, protein-on-membrane ~ d^alpha), so ``alpha = 2 * slope`` of
log(protein) on log(lipid). Every benchmark adapter's sorting-curve step calls
``recover_alpha`` here rather than re-deriving a fit inline.

Why errors-in-variables (Deming), not OLS
------------------------------------------
BOTH axes are noisy: lipid carries PMT/shot noise and protein carries PMT noise
plus per-spot eta heterogeneity. Ordinary least squares assumes a noiseless x and
so ATTENUATES the slope toward zero (regression dilution) — it reports alpha biased
LOW. ``deming_slope`` is the constant-lambda errors-in-variables fit that accounts
for noise in both axes and is the CANONICAL production estimator.

Why NOT per-spot weighting in production
----------------------------------------
``york_slope`` is the statistically-correct per-POINT errors-in-variables fit (York
1968/2004): it weights each spot by its own x/y variance. On clean,
position-independent noise it is unbiased and lower-variance than constant-lambda
Deming. BUT in this pipeline the model's predicted per-spot variance is CONFOUNDED
with the size axis (small/dim spots are both noisier AND sit at the low-lipid end of
the line), so per-spot weighting tilts the fit and BIASES the recovered slope. It is
therefore included for diagnostics/QC only and is NOT the production alpha. See
``experiments/2026-06-10_diagnostic-run/EXPERIMENT.md``.

Log-space variance note (IMPORTANT)
-----------------------------------
The detector's intensity NLL residual is defined in LOG space
(``r = log(pred + eps) - log(true + eps)``), so a head ``logvar`` output is ALREADY
the log-residual variance. Pass ``exp(logvar)`` DIRECTLY as ``var_x`` / ``var_y`` —
do NOT apply a delta-method ``/intensity**2`` conversion (that would double-count the
log transform).
"""

import json
from pathlib import Path

import numpy as np

__all__ = [
    'ols_slope', 'deming_slope', 'york_slope',
    'recover_alpha', 'CalibrationCurve', 'apply_calibration',
    'SEED_RECOVERED', 'SEED_TRUE',
]

#: Measured Stage-2 calibration points (recovered -> true alpha), from the
#: fixed-alpha synthetic test sets. Used to seed the default ``CalibrationCurve``.
SEED_RECOVERED = (0.644, 0.957, 1.321, 1.677)
SEED_TRUE = (0.5, 1.0, 1.5, 2.0)


def _as_1d(*arrays):
    return tuple(np.asarray(a, dtype=np.float64).ravel() for a in arrays)


def ols_slope(logx, logy):
    """Ordinary least squares slope of ``logy`` on ``logx``.

    For comparison/reporting the OLS bias ONLY — it assumes a noiseless x and so
    attenuates the slope when x is noisy. NOT the recommended estimator; use
    ``deming_slope`` / ``recover_alpha`` for production.
    """
    logx, logy = _as_1d(logx, logy)
    return float(np.polyfit(logx, logy, 1)[0])


def deming_slope(logx, logy, lam=1.0):
    """Constant-lambda Deming (errors-in-variables) slope of ``logy`` on ``logx``.

    ``lam = var(y_noise) / var(x_noise)`` is the (assumed constant) ratio of the
    measurement-error variances. ``lam = 1`` is total least squares (TLS), the right
    default when the noise ratio is unknown. THIS IS THE CANONICAL ESTIMATOR.
    """
    logx, logy = _as_1d(logx, logy)
    mx, my = logx.mean(), logy.mean()
    sxx = np.mean((logx - mx) ** 2)
    syy = np.mean((logy - my) ** 2)
    sxy = np.mean((logx - mx) * (logy - my))
    return float((syy - lam * sxx
                  + np.sqrt((syy - lam * sxx) ** 2 + 4.0 * lam * sxy ** 2))
                 / (2.0 * sxy))


def york_slope(logx, logy, var_x, var_y, iters=100, tol=1e-10):
    """York (1968/2004) best-fit line with per-POINT errors in both axes.

    ``var_x`` / ``var_y`` are the per-point variances of ``logx`` / ``logy`` (pass
    ``exp(logvar)`` directly — see the module docstring; the residual is already in
    log space). Weights ``wx = 1/var_x``, ``wy = 1/var_y`` and per-point
    ``W = wx*wy / (wx + b**2 * wy)`` (zero x-y error correlation), iterated because
    ``W`` depends on the slope ``b``.

    Included for COMPLETENESS / QC diagnostics only. Per-spot weighting is NOT used
    as the production estimator: the predicted per-spot variance is confounded with
    the size axis, which biases the slope (see module docstring + EXPERIMENT.md). Use
    ``recover_alpha`` (constant-lambda Deming) for the production alpha.
    """
    logx, logy, var_x, var_y = _as_1d(logx, logy, var_x, var_y)
    wx = 1.0 / np.maximum(var_x, 1e-12)
    wy = 1.0 / np.maximum(var_y, 1e-12)
    b = float(np.polyfit(logx, logy, 1)[0])          # OLS init
    for _ in range(iters):
        W = wx * wy / (wx + (b ** 2) * wy)
        Wsum = W.sum()
        xbar = (W * logx).sum() / Wsum
        ybar = (W * logy).sum() / Wsum
        U = logx - xbar
        V = logy - ybar
        # York's beta_i with zero x-y error correlation.
        beta = W * (U / wy + (b * V) / wx)
        denom = (W * beta * U).sum()
        if denom == 0:
            break
        b_new = (W * beta * V).sum() / denom
        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new
    return float(b)


def recover_alpha(logx, logy, lam=None, var_x=None, var_y=None):
    """Production entry point: recovered ``alpha = 2 * deming_slope(...)``.

    ``lam`` (var(y_noise)/var(x_noise)) selection, in order of precedence:
      * explicit ``lam`` if given;
      * else, if ``var_x`` and ``var_y`` are given, the data estimate
        ``mean(var_y) / mean(var_x)`` (the AVERAGE noise ratio — still a single
        constant lambda, so it stays in the unbiased constant-lambda regime and does
        NOT do per-spot weighting);
      * else ``lam = 1`` (total least squares), the documented default when the
        noise ratio is unknown.

    Constant-lambda Deming is used deliberately rather than per-spot York weighting
    (see module docstring: per-spot variance is confounded with the size axis).
    """
    if lam is None:
        if var_x is not None and var_y is not None:
            mvx = float(np.mean(np.asarray(var_x, dtype=np.float64)))
            mvy = float(np.mean(np.asarray(var_y, dtype=np.float64)))
            lam = mvy / mvx if mvx > 0 else 1.0
        else:
            lam = 1.0
    return 2.0 * deming_slope(logx, logy, lam)


class CalibrationCurve:
    """Recovered-alpha -> true-alpha mapping measured on fixed-alpha synthetic sets.

    The recovered alpha is biased (detection misses + residual attenuation), so the
    fixed-alpha sets give a set of (recovered, true) anchor points. ``invert`` maps a
    recovered alpha to a corrected (true) alpha. Two interpolation kinds:

      * ``'interp'`` (default) — piecewise-linear through the anchors, with linear
        extrapolation beyond the ends. Passes through every anchor EXACTLY, so it
        round-trips the seed points.
      * ``'linear'`` — a single least-squares line ``true ~ a*recovered + b`` (more
        robust to anchor noise once there are many points + error bars).

    Seeded by default with the measured Stage-2 points; ``CalibrationCurve.load`` /
    ``save`` round-trip a fitted curve to JSON so it can be refreshed with more
    points (and error bars) later.
    """

    def __init__(self, recovered, true, kind='interp'):
        rec, tru = _as_1d(recovered, true)
        if rec.size != tru.size:
            raise ValueError("recovered and true must have the same length")
        if rec.size < 2:
            raise ValueError("need >= 2 anchor points")
        order = np.argsort(rec)
        self.recovered = rec[order]
        self.true = tru[order]
        if kind not in ('interp', 'linear'):
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        if kind == 'linear':
            self._a, self._b = np.polyfit(self.recovered, self.true, 1)

    @classmethod
    def default(cls, kind='interp'):
        """The default curve seeded with the measured Stage-2 anchor points."""
        return cls(SEED_RECOVERED, SEED_TRUE, kind=kind)

    def invert(self, alpha_recovered):
        """Map a recovered alpha (scalar or array) to a corrected (true) alpha."""
        a = np.asarray(alpha_recovered, dtype=np.float64)
        if self.kind == 'linear':
            out = self._a * a + self._b
        else:
            out = self._interp_extrap(a)
        return float(out) if np.isscalar(alpha_recovered) or out.ndim == 0 else out

    def _interp_extrap(self, a):
        """Piecewise-linear interp through anchors + linear extrapolation at ends."""
        x, y = self.recovered, self.true
        out = np.interp(a, x, y)                       # clamps outside [x0, x-1]
        # Replace clamped ends with linear extrapolation off the end segments.
        below = a < x[0]
        above = a > x[-1]
        if np.any(below):
            s = (y[1] - y[0]) / (x[1] - x[0])
            out = np.where(below, y[0] + s * (a - x[0]), out)
        if np.any(above):
            s = (y[-1] - y[-2]) / (x[-1] - x[-2])
            out = np.where(above, y[-1] + s * (a - x[-1]), out)
        return out

    def to_dict(self):
        return {'recovered': self.recovered.tolist(),
                'true': self.true.tolist(), 'kind': self.kind}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d):
        return cls(d['recovered'], d['true'], kind=d.get('kind', 'interp'))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def apply_calibration(alpha_recovered, curve):
    """Correct a recovered alpha through a ``CalibrationCurve`` -> true alpha."""
    return curve.invert(alpha_recovered)
