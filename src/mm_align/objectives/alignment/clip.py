"""CLIP-style InfoNCE alignment (symmetric)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlignLoss


class CLIPAlign(AlignLoss):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        ocfg = cfg["experiment"]["align"]
        init = ocfg.get("temperature_init", 0.07)
        log_scale = math.log(1.0 / max(init, 1e-4))
        if ocfg.get("learn_temperature", True):
            self.log_scale = nn.Parameter(torch.tensor(log_scale))
        else:
            self.register_buffer("log_scale", torch.tensor(log_scale))

    def forward(self, model_out, batch):
        zi = F.normalize(model_out["z_image"], dim=-1)
        zt = F.normalize(model_out["z_tx"], dim=-1)
        scale = self.log_scale.exp().clamp(max=100.0)
        logits = scale * zi @ zt.t()
        tgt = torch.arange(zi.size(0), device=zi.device)
        l_i2t = F.cross_entropy(logits, tgt)
        l_t2i = F.cross_entropy(logits.t(), tgt)
        loss = 0.5 * (l_i2t + l_t2i)
        return loss, {
            "align/loss": loss.detach(),
            "align/i2t": l_i2t.detach(),
            "align/t2i": l_t2i.detach(),
            "align/temp": (1.0 / scale).detach(),
        }
