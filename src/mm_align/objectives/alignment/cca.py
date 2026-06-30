"""DeepCCA-style alignment: maximise sum of canonical correlations.

We use the standard deep-CCA gradient surrogate: trace-norm of the
canonical correlation matrix Σ = T1^{-1/2} Σ12 T2^{-1/2} where T_k are
the regularised within-modality covariances.  Equivalent (up to sign) to
the loss in `seal/losses/gene_loss.py::DeepPLSLoss` but using the proper
whitening.

Loss = − ‖Σ‖_*  (negative trace norm; we minimise it).
"""
from __future__ import annotations
import torch

from .base import AlignLoss


def _cca_corr(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    z1 = z1 - z1.mean(0, keepdim=True)
    z2 = z2 - z2.mean(0, keepdim=True)
    B = z1.shape[0]
    D = z1.shape[1]
    I = torch.eye(D, device=z1.device, dtype=z1.dtype) * eps
    S11 = (z1.t() @ z1) / (B - 1) + I
    S22 = (z2.t() @ z2) / (B - 1) + I
    S12 = (z1.t() @ z2) / (B - 1)

    # Symmetric inverse-sqrt via eigendecomposition (numerically stable).
    def _inv_sqrt(M):
        evals, evecs = torch.linalg.eigh(M)
        evals = evals.clamp(min=eps).rsqrt()
        return evecs @ torch.diag(evals) @ evecs.t()

    T1 = _inv_sqrt(S11.float()).to(S11.dtype)
    T2 = _inv_sqrt(S22.float()).to(S22.dtype)
    M = T1 @ S12 @ T2
    # Sum of singular values = trace norm ≈ Σ canonical correlations.
    # Use sqrt(diag(M^T M)) for stability vs full SVD on small batches.
    sv = torch.linalg.svdvals(M.float()).to(M.dtype)
    return sv.sum()


class CCAAlign(AlignLoss):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        ocfg = cfg["experiment"]["align"]
        self.eps = ocfg.get("reg_eps", 1e-4)
        self.scale = ocfg.get("scale", 0.01)

    def forward(self, model_out, batch):
        corr = _cca_corr(model_out["z_image"], model_out["z_tx"], eps=self.eps)
        loss = -self.scale * corr
        return loss, {"align/loss": loss.detach(),
                       "align/cca_corr": corr.detach()}
