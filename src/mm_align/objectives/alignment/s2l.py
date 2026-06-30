"""Soft-CLIP / Sample-to-Label (S2L) alignment.

Instead of hard 1-of-N positives, the target distribution for image i is
softmax over the (image-image) tx-tx similarity within the batch.  This
relaxes the "all other spots are negatives" assumption — useful when many
spots in a batch share organ/tissue context and so are *not* fully negative.

Equivalent to CLIP when the off-diagonal target is 0.  When `beta>0`, we
mix in a soft label built from tx-tx cosine similarity (analogous to SEAL's
soft_clip and "S2L" methods).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlignLoss


class S2LAlign(AlignLoss):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        ocfg = cfg["experiment"]["align"]
        init = ocfg.get("temperature_init", 0.07)
        log_scale = math.log(1.0 / max(init, 1e-4))
        if ocfg.get("learn_temperature", True):
            self.log_scale = nn.Parameter(torch.tensor(log_scale))
        else:
            self.register_buffer("log_scale", torch.tensor(log_scale))
        # weight on soft-label (0 → vanilla CLIP, 1 → fully soft).
        self.beta = float(ocfg.get("soft_beta", 0.5))
        # soft-label temperature for the tx-tx similarity matrix.
        self.label_temp = float(ocfg.get("label_temp", 0.1))

    def forward(self, model_out, batch):
        zi = F.normalize(model_out["z_image"], dim=-1)
        zt = F.normalize(model_out["z_tx"], dim=-1)
        scale = self.log_scale.exp().clamp(max=100.0)

        # Hard CLIP logits.
        logits = scale * zi @ zt.t()  # (B, B)

        # Soft label: softmax over zt @ zt^T (tx-tx similarity).
        with torch.no_grad():
            sim = zt @ zt.t() / max(self.label_temp, 1e-4)
            soft = F.softmax(sim, dim=-1)
        eye = torch.eye(logits.size(0), device=logits.device, dtype=soft.dtype)
        target = self.beta * soft + (1 - self.beta) * eye

        logp_i2t = F.log_softmax(logits, dim=-1)
        logp_t2i = F.log_softmax(logits.t(), dim=-1)
        l_i2t = -(target * logp_i2t).sum(-1).mean()
        l_t2i = -(target * logp_t2i).sum(-1).mean()
        loss = 0.5 * (l_i2t + l_t2i)
        return loss, {
            "align/loss": loss.detach(),
            "align/i2t": l_i2t.detach(),
            "align/t2i": l_t2i.detach(),
            "align/temp": (1.0 / scale).detach(),
        }
