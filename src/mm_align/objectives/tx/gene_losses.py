"""Gene reconstruction loss menu — direct port of SEAL's `get_gene_loss_fn`.

Methods available
  mse              : nn.MSELoss
  standardized_mse : MSE on z-scored y/ŷ (per-gene mean/std supplied via gene_stats)
  pcc              : 1 − mean per-sample Pearson r
  barlow           : Barlow Twins cross-correlation on row-normalised z1,z2
  barlow_mse       : λ·BarlowTwins + (1−λ)·MSE
  barlow_std_mse   : λ·BarlowTwins + (1−λ)·StandardizedMSE   ← SEAL's default
  l1, huber        : robust regression losses
  negbin           : Negative-binomial likelihood (log1p-space input → expm1)

The Barlow variants double as anti-collapse regularisers: when there's no
explicit negative signal (e.g. JEPA-only), Barlow's off-diagonal term
de-correlates dimensions, preventing rank collapse.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class StandardizedMSELoss(nn.Module):
    def __init__(self, gene_means: Optional[np.ndarray] = None,
                 gene_stds: Optional[np.ndarray] = None):
        super().__init__()
        if gene_means is not None:
            self.register_buffer("means", torch.tensor(gene_means, dtype=torch.float32))
            self.register_buffer("stds",  torch.tensor(gene_stds,  dtype=torch.float32))
            self.has_stats = True
        else:
            self.has_stats = False
        self.eps = 1e-8

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.has_stats:
            m = self.means.to(y_pred.device).to(y_pred.dtype)
            s = self.stds.to(y_pred.device).to(y_pred.dtype) + self.eps
            y_pred = (y_pred - m) / s
            y_true = (y_true - m) / s
        return F.mse_loss(y_pred, y_true)


class PearsonSampleLoss(nn.Module):
    """1 − mean of per-sample Pearson r."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        xm = y_true - y_true.mean(-1, keepdim=True)
        ym = y_pred - y_pred.mean(-1, keepdim=True)
        num = (xm * ym).sum(-1)
        den = torch.sqrt((xm ** 2).sum(-1) * (ym ** 2).sum(-1)) + 1e-8
        return 1 - (num / den).mean()


class BarlowTwinsLoss(nn.Module):
    """Cross-correlation Barlow Twins on (z1, z2). Both shaped (B, D)."""

    def __init__(self, lambd: float = 5e-3, scaling: float = 0.05):
        super().__init__()
        self.lambd = lambd
        self.scaling = scaling

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)
        z1 = z1 / (z1.var(dim=0, unbiased=False, keepdim=True).add(1e-9).sqrt())
        z2 = z2 / (z2.var(dim=0, unbiased=False, keepdim=True).add(1e-9).sqrt())
        B, D = z1.shape
        c = (z1.T @ z2) / B
        diag = torch.eye(D, device=c.device, dtype=c.dtype)
        c_diff = (c - diag).pow(2)
        loss = self.lambd * c_diff.sum() + (1 - self.lambd) * c_diff.diagonal().sum()
        return loss * self.scaling


class BarlowMSELoss(nn.Module):
    def __init__(self, lambd: float = 0.0, mse_norm: bool = True,
                 gene_means: Optional[np.ndarray] = None,
                 gene_stds: Optional[np.ndarray] = None,
                 lambda_barlow: float = 1.0, lambda_mse: float = 5.0):
        super().__init__()
        self.barlow = BarlowTwinsLoss(lambd=lambd)
        if mse_norm:
            self.mse = StandardizedMSELoss(gene_means=gene_means, gene_stds=gene_stds)
        else:
            self.mse = nn.MSELoss()
        self.lambda_barlow = lambda_barlow
        self.lambda_mse = lambda_mse

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.lambda_barlow * self.barlow(z1, z2) + self.lambda_mse * self.mse(z1, z2)


class NegBinomialLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1: prediction in log1p space; z2: target in log1p space
        alpha_t = torch.as_tensor(self.alpha, dtype=z1.dtype, device=z1.device)
        mu = torch.expm1(z1).clamp(min=self.eps)
        z2 = torch.expm1(z2).clamp(min=0.0)
        p = mu / (mu + alpha_t)
        log_prob = (
            torch.lgamma(z2 + alpha_t)
            - torch.lgamma(alpha_t)
            - torch.lgamma(z2 + 1.0)
            + alpha_t * torch.log(1 - p + self.eps)
            + z2 * torch.log(p + self.eps)
        )
        return -log_prob.mean()


# ---------------------------------------------------------------------------
# Factory (matches SEAL's get_gene_loss_fn API)
# ---------------------------------------------------------------------------

_SCALE = {
    "mse": 10.0, "standardized_mse": 10.0,
    "pcc": 5.0,
    "barlow": 0.1, "barlow_mse": 0.1, "barlow_std_mse": 0.1,
    "l1": 10.0, "huber": 10.0,
    "negbin": 10.0,
}


def build_gene_loss(method: str, gene_means: Optional[np.ndarray] = None,
                    gene_stds: Optional[np.ndarray] = None) -> tuple[nn.Module, float]:
    """Returns (loss_fn(pred, target), scale_factor).  Use `scale * loss_fn(...)`
    so all heads land in the same magnitude as MSE @ scale 10."""
    method = method.lower()
    scale = _SCALE.get(method, 1.0)

    if method == "mse":
        return nn.MSELoss(), scale
    if method == "standardized_mse":
        return StandardizedMSELoss(gene_means=gene_means, gene_stds=gene_stds), scale
    if method == "pcc":
        return PearsonSampleLoss(), scale
    if method == "barlow":
        return BarlowTwinsLoss(lambd=5e-3), scale
    if method == "barlow_soft":
        return BarlowTwinsLoss(lambd=0.0), scale
    if method == "barlow_mse":
        return BarlowMSELoss(lambd=0.0, mse_norm=False), scale
    if method == "barlow_std_mse":
        return BarlowMSELoss(lambd=0.0, mse_norm=True,
                             gene_means=gene_means, gene_stds=gene_stds), scale
    if method == "l1":
        return nn.L1Loss(), scale
    if method == "huber":
        return nn.SmoothL1Loss(), scale
    if method == "negbin":
        return NegBinomialLoss(), scale
    raise ValueError(f"Unknown gene loss method: {method}")
