"""SEAL-style gene reconstruction decoders.

  ImgEmbDecoder : image latent → predicted HVG (linear + ReLU final).
  GeneEmbDecoder: tx latent    → predicted HVG (same shape; used as the
                                  "gene auto-encoder decoder" head).

These mirror SEAL's `ImgEmbDecoder` and the `decoder` inside `GeneMLP/GeneVAE`.
ReLU at the end forces non-negative predictions (log1p-normalised HVG).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ImgEmbDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid_dim: int = 512,
                 batch_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hid_dim)]
        if batch_norm:
            layers += [nn.BatchNorm1d(hid_dim)]
        layers += [nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(hid_dim, out_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class GeneEmbDecoder(nn.Module):
    """Single-layer linear decoder back to HVG; matches SEAL's gene auto-encoder head."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
