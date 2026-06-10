"""Dense prediction heads (CenterNet-style + heteroscedastic intensity).

All heads are dense per-pixel maps at the backbone's output resolution. At
training they are supervised at GROUND-TRUTH center cells; at inference they are
SAMPLED at detected peak locations (see decode.py). Four heads:

  HeatmapHead   (1 ch) : per-pixel spot-presence probability (sigmoid, clamped).
  OffsetHead    (2 ch) : raw subpixel (dx, dy) refinement of the integer peak.
  IntensityHead (2 ch) : channel 0 = MEAN total flux (>0, via exp), channel 1 =
                         predicted LOG-VARIANCE (raw). One instance per channel
                         (lipid, protein) — they live on different noise regimes.

Intensity mean is parameterized in log space (``flux = exp(raw_mean)``, clamped)
so it is strictly positive for the ``log(pred_flux + eps)`` residual in the loss
and so the regression target's huge dynamic range is handled on a log scale. The
mean-head output bias is initialized to ``log(init_flux)`` to start near a typical
flux. ``decode`` reads channel 0 as the flux directly (already exp'd).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Clamp on the log-flux before exp, to keep the mean finite and gradients sane.
_LOG_FLUX_MIN = -10.0
_LOG_FLUX_MAX = 20.0


class _HeadConv(nn.Module):
    """3x3 conv -> ReLU -> 1x1 conv to ``out_ch`` (the standard CenterNet head)."""

    def __init__(self, in_ch, out_ch, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, out_ch, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x), inplace=True))


class HeatmapHead(nn.Module):
    """Per-pixel spot-presence probability in (0, 1), clamped off 0/1 for the log.

    The final-layer bias is initialized so the initial probability is ~``prior``
    (focal-loss stability: most pixels are background).
    """

    def __init__(self, in_ch, hidden=64, prior=0.01, clamp_eps=1e-4):
        super().__init__()
        self.head = _HeadConv(in_ch, 1, hidden)
        self.clamp_eps = clamp_eps
        nn.init.constant_(self.head.conv2.bias, -math.log((1.0 - prior) / prior))

    def forward(self, x):
        p = torch.sigmoid(self.head(x))
        return p.clamp(self.clamp_eps, 1.0 - self.clamp_eps)


class OffsetHead(nn.Module):
    """Raw subpixel (dx, dy) offset of the true center from its integer cell."""

    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.head = _HeadConv(in_ch, 2, hidden)

    def forward(self, x):
        return self.head(x)


class IntensityHead(nn.Module):
    """Heteroscedastic total-flux head: channel 0 = mean flux (>0), 1 = log-var.

    ``forward`` returns a (B, 2, H, W) tensor whose channel 0 is the POSITIVE mean
    flux (``exp`` of a clamped raw log-flux) and channel 1 is the raw predicted
    log-variance (any real). ``init_flux`` sets the mean-head bias to
    ``log(init_flux)`` so predictions start near a typical flux.
    """

    def __init__(self, in_ch, hidden=64, init_flux=1000.0):
        super().__init__()
        self.head = _HeadConv(in_ch, 2, hidden)
        nn.init.zeros_(self.head.conv2.weight)
        nn.init.zeros_(self.head.conv2.bias)
        # bias[0] -> log-flux start; bias[1] -> logvar start at 0 (var = 1).
        with torch.no_grad():
            self.head.conv2.bias[0] = math.log(float(init_flux))

    def forward(self, x):
        raw = self.head(x)
        log_flux = raw[:, 0:1].clamp(_LOG_FLUX_MIN, _LOG_FLUX_MAX)
        flux = torch.exp(log_flux)
        logvar = raw[:, 1:2]
        return torch.cat([flux, logvar], dim=1)
