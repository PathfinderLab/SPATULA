from __future__ import annotations
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """SimCLR/VICReg-style projector with optional last BN."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, bias: bool = False, last_bn: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for i in range(n_layers - 1):
            layers += [nn.Linear(cur, hidden_dim, bias=bias),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
            cur = hidden_dim
        layers += [nn.Linear(cur, out_dim, bias=bias)]
        if last_bn:
            layers += [nn.BatchNorm1d(out_dim, affine=False)]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedGeneHead(nn.Module):
    """Linear decoder back to HVG dimension for masked-gene reconstruction."""

    def __init__(self, in_dim: int, hvg_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hvg_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class JEPAPredictor(nn.Module):
    """Predict target latent from context latent."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for i in range(n_layers - 1):
            layers += [nn.Linear(cur, hidden_dim), nn.GELU()]
            cur = hidden_dim
        layers += [nn.Linear(cur, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
