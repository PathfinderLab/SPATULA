from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Objective(nn.Module):
    """Owns extra heads + computes loss given the aligner output and batch.

    `step(batch, model_out)` returns (loss, log_dict).
    Sub-classes may register their own buffers/parameters which the trainer
    auto-includes via this module's `parameters()`.
    """

    def __init__(self, cfg: dict, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        # Bypass nn.Module's __setattr__ so we don't double-register the aligner
        # (which would also cause optimizer duplicate-param warnings).
        object.__setattr__(self, "model", model)

    def step(self, batch: dict, model_out: dict) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def forward(self, batch: dict, model_out: dict) -> tuple[torch.Tensor, dict]:
        # Alias so DDP-wrapped objectives are callable via __call__(...).
        # nn.Module.__call__ → forward → subclass.step, with DDP forward hooks
        # installed so gradients on objective-owned params (e.g. CLIP temp, JEPA
        # predictors) still get all-reduced on backward.
        return self.step(batch, model_out)

    # Optional EMA hook (used by JEPA)
    def on_after_step(self) -> None:
        return


def masked_gene_recon_loss(model, batch: dict, model_out: dict,
                           mask_ratio: float) -> tuple[torch.Tensor, dict]:
    """Auxiliary masked HVG-gene reconstruction from the (already encoded) tx latent.

    Uses MULTIPLICATIVE masking (vs. boolean indexing) so the autograd graph
    has the same shape every iteration — required when DDP runs with
    static_graph=True (which we need for gradient checkpointing).
    """
    if "hvg" not in batch or not hasattr(model, "gene_head"):
        return torch.zeros((), device=batch["image"].device), {}
    hvg = batch["hvg"]
    pred = model.gene_head(model_out["h_tx"])              # (B, D_hvg) — always called
    mask = (torch.rand_like(hvg) < mask_ratio).float()
    sq = (pred - hvg).pow(2) * mask
    denom = mask.sum().clamp(min=1.0)
    loss = sq.sum() / denom
    return loss, {"aux/gene_mse": loss.detach()}
