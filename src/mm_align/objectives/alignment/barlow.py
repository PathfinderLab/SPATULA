"""Barlow Twins cross-correlation alignment (no negatives needed)."""
from __future__ import annotations
import torch

from .base import AlignLoss
from ..tx.gene_losses import BarlowTwinsLoss


class BarlowAlign(AlignLoss):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        ocfg = cfg["experiment"]["align"]
        self.bt = BarlowTwinsLoss(
            lambd=ocfg.get("lambd", 5e-3),
            scaling=ocfg.get("scaling", 0.05),
        )

    def forward(self, model_out, batch):
        loss = self.bt(model_out["z_image"], model_out["z_tx"])
        return loss, {"align/loss": loss.detach(),
                       "align/method": torch.tensor(0.0)}  # dummy for json schema stability
