"""UnifiedObjective — SEAL-style single training objective with three terms.

  total = λ_align · L_align        ← align method ∈ {clip, barlow, cca, s2l, jepa}
        + λ_recon_gene · L_recon_gene(tx-side gene decoder, hvg)
        + λ_recon_img  · L_recon_img(image-side gene decoder, hvg)

Each gene-recon term uses the same loss menu (mse / standardized_mse /
pcc / barlow_mse / barlow_std_mse / l1 / huber / negbin), independently
configurable.  Per-batch training metrics (cosine_sim, gene_pcc, …) are
added to the log dict so the trainer can stream them without an extra eval.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .alignment import build_align_loss
from .tx.gene_losses import build_gene_loss
from .tx.masked import MaskedTxLosses
from ..evaluation.train_metrics import compute_train_metrics


class UnifiedObjective(nn.Module):
    def __init__(self, cfg: dict, model: nn.Module,
                 gene_means: Optional[np.ndarray] = None,
                 gene_stds: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "model", model)

        exp = cfg["experiment"]
        # ── Align term ──────────────────────────────────────────────
        method = exp["align"]["method"]
        self.align_method = method
        self.align_weight = float(exp["align"].get("weight", 1.0))
        self.align = build_align_loss(method, cfg, model)

        # ── Gene-recon term (tx-side decoder) ───────────────────────
        gcfg = exp.get("gene_recon", {})
        self.gene_recon_weight = float(gcfg.get("weight", 1.0))
        self.gene_recon = None
        self.gene_scale = 1.0
        if self.gene_recon_weight > 0:
            loss_fn, scale = build_gene_loss(gcfg.get("method", "mse"),
                                             gene_means, gene_stds)
            self.gene_recon = loss_fn
            self.gene_scale = scale

        # ── Image-side recon term (image latent → HVG decoder) ──────
        icfg = exp.get("image_recon", {})
        self.image_recon_weight = float(icfg.get("weight", 1.0))
        self.image_recon = None
        self.image_scale = 1.0
        if self.image_recon_weight > 0:
            loss_fn, scale = build_gene_loss(icfg.get("method", "mse"),
                                             gene_means, gene_stds)
            self.image_recon = loss_fn
            self.image_scale = scale

        # ── Tx-side self-supervised losses (learnable gene encoders only) ─
        tx_kind = cfg["model"]["transcriptomics"].get("kind", "novae_adapter")
        tcfg = exp.get("tx_self", {})
        self.tx_self_weight = float(tcfg.get("weight", 0.0))
        self.tx_self = None
        if tx_kind in ("hvg_tokenizer", "top_hvg_gene") and self.tx_self_weight > 0:
            self.tx_self = MaskedTxLosses(
                model=model,
                ema_momentum=tcfg.get("ema_momentum", 0.999),
                enable_jepa=tcfg.get("enable_masked_jepa", True),
                enable_dino_consistency=tcfg.get("enable_dino_consistency", False),
                symbol_weight=tcfg.get("symbol_weight", 1.0),
                value_weight=tcfg.get("value_weight", 0.5),
                jepa_weight=tcfg.get("jepa_weight", 1.0),
                enable_view_jepa=tcfg.get("enable_view_jepa", False),
                view_jepa_weight=tcfg.get("view_jepa_weight", 0.0),
                view_jepa_loss=tcfg.get("view_jepa_loss", "smooth_l1"),
                view_jepa_hidden_dim=tcfg.get("view_jepa_hidden_dim", 1024),
                view_jepa_warmup_epochs=tcfg.get("view_jepa_warmup_epochs", 0),
                view_jepa_ramp_epochs=tcfg.get("view_jepa_ramp_epochs", 0),
                enable_multi_chunk_jepa=tcfg.get("enable_multi_chunk_jepa", False),
                multi_chunk_weight=tcfg.get("multi_chunk_weight", 0.0),
                multi_chunk_n_chunks=tcfg.get("multi_chunk_n_chunks", 4),
                multi_chunk_len=tcfg.get("multi_chunk_len", 256),
                multi_chunk_loss=tcfg.get("multi_chunk_loss", "smooth_l1"),
                multi_chunk_target=tcfg.get("multi_chunk_target", "target_chunk"),
                multi_chunk_dynamic=tcfg.get("multi_chunk_dynamic", True),
                multi_chunk_target_chunks=tcfg.get("multi_chunk_target_chunks", 2),
                multi_chunk_target_scale=tcfg.get("multi_chunk_target_scale", (0.15, 0.25)),
                multi_chunk_context_scale=tcfg.get("multi_chunk_context_scale", (0.45, 0.65)),
                multi_chunk_hidden_dim=tcfg.get("multi_chunk_hidden_dim", 1024),
                multi_chunk_target_id_scale=tcfg.get("multi_chunk_target_id_scale", 0.5),
                multi_chunk_koleo_weight=tcfg.get("multi_chunk_koleo_weight", 0.0),
                multi_chunk_regularizer=tcfg.get("multi_chunk_regularizer", "none"),
                multi_chunk_vicreg_var_weight=tcfg.get("multi_chunk_vicreg_var_weight", 1.0),
                multi_chunk_vicreg_cov_weight=tcfg.get("multi_chunk_vicreg_cov_weight", 1.0),
                multi_chunk_vicreg_gamma=tcfg.get("multi_chunk_vicreg_gamma", 1.0),
                multi_chunk_warmup_epochs=tcfg.get("multi_chunk_warmup_epochs", 0),
                multi_chunk_ramp_epochs=tcfg.get("multi_chunk_ramp_epochs", 0),
                dino_weight=tcfg.get("dino_weight", 0.0),
                dino_loss=tcfg.get("dino_loss", "cosine"),
                dino_student_temp=tcfg.get("dino_student_temp", 0.1),
                dino_teacher_temp=tcfg.get("dino_teacher_temp", 0.04),
                sinkhorn_iterations=tcfg.get("sinkhorn_iterations", 3),
                dino_warmup_epochs=tcfg.get("dino_warmup_epochs", 0),
                dino_ramp_epochs=tcfg.get("dino_ramp_epochs", 0),
                koleo_weight=tcfg.get("koleo_weight", 0.0),
                koleo_eps=tcfg.get("koleo_eps", 1e-6),
                koleo_warmup_epochs=tcfg.get("koleo_warmup_epochs", 0),
                koleo_ramp_epochs=tcfg.get("koleo_ramp_epochs", 0),
                masking_obj=tcfg.get("masking_obj", "both"),
            )

    def _student(self):
        return self.__dict__["model"]

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        if self.tx_self is not None and hasattr(self.tx_self, "set_epoch"):
            self.tx_self.set_epoch(epoch)

    def forward(self, batch: dict, model_out: dict) -> tuple[torch.Tensor, dict]:
        log: dict[str, torch.Tensor] = {}
        loss_align, align_log = self.align(model_out, batch)
        log.update(align_log)
        loss = self.align_weight * loss_align

        # Tx-side gene reconstruction (always available when HVG is present).
        if self.gene_recon is not None and "gene_recon_from_tx" in model_out and "hvg" in batch:
            pred = model_out["gene_recon_from_tx"]
            tgt = batch["hvg"]
            l = self.gene_recon(pred, tgt) * self.gene_scale
            loss = loss + self.gene_recon_weight * l
            log["recon/gene"] = l.detach()

        # Image-side reconstruction — SEAL's lambda_recon_img.
        if self.image_recon is not None and "gene_recon_from_image" in model_out and "hvg" in batch:
            pred = model_out["gene_recon_from_image"]
            tgt = batch["hvg"]
            l = self.image_recon(pred, tgt) * self.image_scale
            loss = loss + self.image_recon_weight * l
            log["recon/image"] = l.detach()

        # Tx-side self-supervised losses (masked-recon + masked-JEPA).
        # Only active when transcriptomics.kind == "hvg_tokenizer".
        if self.tx_self is not None:
            l_tx, tx_log = self.tx_self(model_out, batch)
            loss = loss + self.tx_self_weight * l_tx
            log.update(tx_log)

        # Light per-batch monitoring metrics (no extra forwards).
        log.update(compute_train_metrics(model_out, batch))
        log["loss/total"] = loss.detach()
        return loss, log

    # ------------------------------------------------------------------

    def step(self, batch: dict, model_out: dict) -> tuple[torch.Tensor, dict]:
        # Back-compat: older trainers may call objective.step(...)
        return self.forward(batch, model_out)

    def on_after_step(self) -> None:
        # Forward to the inner align loss (JEPA needs EMA update).
        self.align.on_after_step()
        if self.tx_self is not None:
            self.tx_self.on_after_step()
