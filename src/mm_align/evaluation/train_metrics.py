"""Light per-batch metrics — SEAL's `CalcMetrics` adapted.

Computed cheaply *during* training on each batch (no extra forwards).
Aggregate over a batch via mean.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


@torch.no_grad()
def batch_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = _flat(pred); target = _flat(target)
    xm = target - target.mean(-1, keepdim=True)
    ym = pred - pred.mean(-1, keepdim=True)
    num = (xm * ym).sum(-1)
    den = torch.sqrt((xm ** 2).sum(-1) * (ym ** 2).sum(-1)) + 1e-8
    return (num / den).mean()


@torch.no_grad()
def batch_spearman(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Spearman r via rank transform (cheap, differentiable-free)."""
    pred = _flat(pred); target = _flat(target)
    pr = pred.argsort(dim=-1).argsort(dim=-1).float()
    tr = target.argsort(dim=-1).argsort(dim=-1).float()
    return batch_pearson(pr, tr)


@torch.no_grad()
def batch_cosine(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(z1, z2, dim=-1).mean()


@torch.no_grad()
def compute_train_metrics(model_out: dict, batch: dict) -> dict:
    """SEAL CalcMetrics-style batch metrics. Returns a dict of scalar tensors
    (kept as tensors so the trainer can `.detach().item()` consistently)."""
    m: dict[str, torch.Tensor] = {}

    # Contrast: cosine_sim(z_image, z_tx).
    if "z_image" in model_out and "z_tx" in model_out:
        m["metric/cosine_sim"] = batch_cosine(model_out["z_image"], model_out["z_tx"])
        # Collapse diagnostic: paired (diag) vs unpaired (off-diag) cosine.
        # Healthy align: diag >> off-diag.  Collapse: diag ≈ off-diag.
        zi = F.normalize(model_out["z_image"], dim=-1)
        zt = F.normalize(model_out["z_tx"], dim=-1)
        sim = zi @ zt.t()
        B = sim.size(0)
        if B > 1:
            diag = sim.diag().mean()
            off = (sim.sum() - sim.diag().sum()) / (B * (B - 1))
            m["metric/diag_cos"] = diag
            m["metric/offdiag_cos"] = off
            m["metric/diag_minus_off"] = diag - off

    # Gene reconstruction quality (uses tx-side reconstruction when present,
    # falls back to image-side; both predict HVG and target is batch["hvg"]).
    if "hvg" in batch:
        target = batch["hvg"]
        for name, key in [("tx", "gene_recon_from_tx"),
                          ("img", "gene_recon_from_image")]:
            if key in model_out:
                pred = model_out[key]
                m[f"metric/gene_{name}_mse"] = F.mse_loss(pred, target).detach()
                m[f"metric/gene_{name}_pcc"] = batch_pearson(pred, target)
                m[f"metric/gene_{name}_spearman"] = batch_spearman(pred, target)
    return m
