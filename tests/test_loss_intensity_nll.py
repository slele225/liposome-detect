"""Log-space intensity NLL: finite for dim spots, sane variance behavior, and the
MSE->NLL loss warmup switchover."""

import math

import pytest
import torch

from src.train.losses import intensity_nll_loss


def test_finite_for_dim_spots_with_eps_floor():
    # very dim true + pred fluxes: the eps floor keeps the log finite.
    pred_mean = torch.tensor([1.0, 0.0])
    pred_logvar = torch.tensor([0.0, 0.0])
    true = torch.tensor([0.0, 2.0])
    val = intensity_nll_loss(pred_mean, pred_logvar, true, eps=50.0, use_nll=True)
    assert torch.isfinite(val)


def test_warmup_mse_is_twice_nll_at_unit_variance():
    pred_mean = torch.tensor([100.0, 3000.0])
    pred_logvar = torch.zeros(2)                  # sigma2 = 1
    true = torch.tensor([1000.0, 2500.0])
    mse = intensity_nll_loss(pred_mean, pred_logvar, true, eps=10.0, use_nll=False)
    nll = intensity_nll_loss(pred_mean, pred_logvar, true, eps=10.0, use_nll=True)
    # with logvar=0: nll = 0.5*(r^2) = 0.5*mse
    assert float(nll) == pytest.approx(0.5 * float(mse), rel=1e-5)


def test_nll_rewards_large_sigma_on_large_residual():
    pred_mean = torch.tensor([10.0])
    true = torch.tensor([10000.0])
    eps = 1.0
    r = math.log(10.0 + eps) - math.log(10000.0 + eps)
    nll_confident = intensity_nll_loss(pred_mean, torch.tensor([0.0]), true, eps)
    nll_honest = intensity_nll_loss(pred_mean, torch.tensor([math.log(r * r)]), true, eps)
    # honest large sigma (matching the big residual) beats false confidence
    assert float(nll_honest) < float(nll_confident)


def test_nll_punishes_large_sigma_on_zero_residual():
    pred_mean = torch.tensor([1000.0])
    true = torch.tensor([1000.0])               # r = 0
    nll_tight = intensity_nll_loss(pred_mean, torch.tensor([0.0]), true, eps=10.0)
    nll_loose = intensity_nll_loss(pred_mean, torch.tensor([4.0]), true, eps=10.0)
    assert float(nll_loose) > float(nll_tight)
