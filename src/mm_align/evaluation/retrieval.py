from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def retrieval_metrics(z_image: torch.Tensor, z_tx: torch.Tensor,
                      ks: tuple[int, ...] = (1, 5, 10, 50)) -> dict[str, float]:
    """Top-K image-to-tx and tx-to-image retrieval on a paired test pool.

    Diagonals are the positives.
    """
    zi = F.normalize(z_image.float(), dim=-1)
    zt = F.normalize(z_tx.float(), dim=-1)
    sim = zi @ zt.t()
    n = sim.size(0)
    out: dict[str, float] = {}
    # image -> tx
    rank_i2t = (sim.argsort(dim=-1, descending=True) == torch.arange(n, device=sim.device).unsqueeze(-1)).float()
    # tx -> image
    rank_t2i = (sim.t().argsort(dim=-1, descending=True) == torch.arange(n, device=sim.device).unsqueeze(-1)).float()
    for k in ks:
        out[f"i2t/R@{k}"] = float(rank_i2t[:, :k].sum(-1).clamp(max=1).mean().item())
        out[f"t2i/R@{k}"] = float(rank_t2i[:, :k].sum(-1).clamp(max=1).mean().item())
    # Mean reciprocal rank
    pos_i2t = (sim.argsort(dim=-1, descending=True) == torch.arange(n, device=sim.device).unsqueeze(-1)).int()
    rr_i2t = 1.0 / (pos_i2t.argmax(dim=-1).float() + 1.0)
    pos_t2i = (sim.t().argsort(dim=-1, descending=True) == torch.arange(n, device=sim.device).unsqueeze(-1)).int()
    rr_t2i = 1.0 / (pos_t2i.argmax(dim=-1).float() + 1.0)
    out["i2t/MRR"] = float(rr_i2t.mean().item())
    out["t2i/MRR"] = float(rr_t2i.mean().item())
    return out
