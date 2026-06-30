"""Top-level multimodal aligner — SEAL-style with image gene-decoder.

Pipeline:

      ┌──────────────────────────────────┐
      │ image  (raw or pre-extracted)     │
      └────────────────┬─────────────────┘
                       ▼
           UNI / feature image_encoder            ─► h_image (D)
                       │                                 │
        ┌──────────────┴──────────┐                      │
        ▼                         ▼                      ▼
  image_projector            image_decoder       (downstream tasks)
   z_image (D_proj)           gene_recon(B,Ghvg)
        │                         │
        │                         │
  align loss                gene-image-recon loss   (lambda_recon_img)
        ▲
        │
   z_tx (D_proj)
        ▲
  tx_projector
        │
        h_tx
        ▲
  tx_encoder (frozen novae+hvg adapter)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..image.encoder import build_image_encoder, UNIImageEncoder, FeatureImageEncoder
from ..tx.factory import build_tx_encoder, NovaeAdapterTxEncoder, HVGTokenizerTxEncoder
from .heads import MLPProjector
from .decoders import ImgEmbDecoder, GeneEmbDecoder


class MMAligner(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.embed_dim = mcfg["embed_dim"]
        self.image_backbone_kind = mcfg["image"]["backbone"]
        # When True, the masked-modeling mask is sampled even in .eval() mode.
        # The trainer sets this in Stage 1 so the val pass produces a real
        # tx_self/loss to early-stop on.  Inference paths (eval_zero_shot,
        # eval_linearprobe, viz_spatial) leave it False so retrieval embeddings
        # are deterministic.
        self.force_mask_in_eval: bool = False

        # ── Encoders ────────────────────────────────────────────────
        self.image_encoder = build_image_encoder(cfg)

        tcfg = mcfg["transcriptomics"]
        self.tx_encoder = build_tx_encoder(cfg)
        self.tx_kind = self.tx_encoder.kind            # "novae_adapter" | "hvg_tokenizer"

        # ── Projectors (optional — for alignment heads) ─────────────
        pcfg = mcfg["projector"]
        proj_kind = pcfg.get("kind", "mlp")     # "mlp" | "linear" | "none"
        proj_dim = pcfg.get("out_dim", self.embed_dim)
        if proj_kind == "none":
            self.image_proj = nn.Identity()
            self.tx_proj = nn.Identity()
            self.proj_dim = self.embed_dim
        elif proj_kind == "linear":
            self.image_proj = nn.Linear(self.embed_dim, proj_dim, bias=pcfg.get("bias", False))
            self.tx_proj = nn.Linear(self.embed_dim, proj_dim, bias=pcfg.get("bias", False))
            self.proj_dim = proj_dim
        else:
            self.image_proj = MLPProjector(
                self.embed_dim, pcfg["hidden_dim"], proj_dim,
                n_layers=pcfg["n_layers"], bias=pcfg.get("bias", False),
                last_bn=pcfg.get("last_bn", False),
            )
            self.tx_proj = MLPProjector(
                self.embed_dim, pcfg["hidden_dim"], proj_dim,
                n_layers=pcfg["n_layers"], bias=pcfg.get("bias", False),
                last_bn=pcfg.get("last_bn", False),
            )
            self.proj_dim = proj_dim

        # ── Decoders ────────────────────────────────────────────────
        # Both decoders predict the same HVG vector.  Image-side is the
        # SEAL "ImgEmbDecoder" (heavy; predicts gene expression directly
        # from the image latent).  Tx-side is the classic gene auto-encoder
        # head used for masked-gene reconstruction.
        d_cfg = mcfg.get("decoder", {})
        hvg_dim = tcfg["hvg_in_dim"]
        self.use_hvg = tcfg["use_hvg"]
        self.image_gene_decoder = ImgEmbDecoder(
            in_dim=self.embed_dim,
            out_dim=hvg_dim,
            hid_dim=d_cfg.get("img_hid_dim", 512),
            batch_norm=d_cfg.get("img_batch_norm", False),
            dropout=d_cfg.get("img_dropout", 0.0),
        ) if d_cfg.get("enable_image_decoder", True) and self.use_hvg else None
        # Tx-side reconstruction head — kept as `gene_head` for back-compat
        # with masked_gene_recon_loss in objectives/base.py.
        self.gene_head = GeneEmbDecoder(self.embed_dim, hvg_dim) if self.use_hvg else None

    # ------------------------------------------------------------------

    def encode_image(self, batch: dict) -> torch.Tensor:
        return self.image_encoder(
            image_feat=batch.get("image"),
            image_raw=batch.get("image_raw"),
        )

    def encode_tx(self, tx_latent: torch.Tensor | None, hvg: torch.Tensor | None,
                  mask: torch.Tensor | None = None) -> dict:
        """Returns a dict {h_tx, per_token, mask, orig_gene_ids?}.
        All tx encoders accept mask=None — Novae-adapter just ignores it."""
        return self.tx_encoder(tx_latent, hvg, mask=mask)

    def project_image(self, h: torch.Tensor) -> torch.Tensor:
        return self.image_proj(h)

    def project_tx(self, h: torch.Tensor) -> torch.Tensor:
        return self.tx_proj(h)

    def forward(self, batch: dict) -> dict:
        h_img = self.encode_image(batch)

        # For hvg_tokenizer, optionally sample a random mask so masked-recon /
        # masked-JEPA can train. (Mask only affects the *student* tx pass; the
        # objective creates its own unmasked-teacher pass when needed.)
        mask = None
        if self.tx_kind in ("hvg_tokenizer", "top_hvg_gene") \
                and (self.training or self.force_mask_in_eval) \
                and batch.get("hvg") is not None:
            mask = self.tx_encoder.sample_mask(batch["hvg"])
        tx_out = self.encode_tx(batch.get("tx_latent"), batch.get("hvg"), mask=mask)
        h_tx = tx_out["h_tx"]
        z_img = self.project_image(h_img)
        z_tx = self.project_tx(h_tx)
        out = {
            "h_image": h_img, "h_tx": h_tx,
            "z_image": z_img, "z_tx": z_tx,
            "z_shared": 0.5 * (z_img + z_tx),
            "tx_per_token": tx_out.get("per_token"),
            "tx_mask": tx_out.get("mask"),
            "orig_gene_ids": tx_out.get("orig_gene_ids"),     # for masked-symbol CE
            "orig_positions": tx_out.get("orig_positions"),   # to route value preds back to HVG slots
            "tx_attention_mask": tx_out.get("attention_mask"),
        }
        # Predict HVG from image latent (SEAL-style image-side reconstruction).
        if self.image_gene_decoder is not None:
            out["gene_recon_from_image"] = self.image_gene_decoder(h_img)
        # Predict HVG from tx latent (gene auto-encoder head, shared across tx kinds).
        if self.gene_head is not None:
            out["gene_recon_from_tx"] = self.gene_head(h_tx)
        # Masked-recon: per-token scalar prediction (only when hvg_tokenizer).
        if self.tx_kind == "hvg_tokenizer" and out["tx_per_token"] is not None:
            out["masked_recon_pred"] = self.tx_encoder.encoder.recon_head(out["tx_per_token"]).squeeze(-1)

        # top_hvg_gene heads — call INSIDE forward so DDP sees the params used.
        # Otherwise calling them from masked_tx.py (objective) triggers
        # "marked-ready-twice" because the heads belong to model's DDP wrapper
        # but the autograd op lives outside its forward.
        #
        # MEMORY CRITICAL: symbol_head outputs (B, L_max, V) logits.  With
        # V=19,183 and L_max≈5,000 this is ~98 GB fp32 per step — instant OOM.
        # Only the ~15% masked positions are ever used for CE, so we gather
        # those positions BEFORE the head and emit (N_masked, V) instead.
        # This trades a 6-7× memory reduction for one extra index op.
        if self.tx_kind == "top_hvg_gene" and out["tx_per_token"] is not None:
            pt = out["tx_per_token"]                              # (B, L, D)
            mask_t = out.get("tx_mask")                            # (B, L) bool — masked positions
            # When no positions are masked (e.g. Stage 1.25 chunk-JEPA refinement
            # runs with mask_ratio=0), the MSM heads are not used by any loss.
            # Calling them on an empty (N=0) tensor still puts their weights in
            # the autograd graph but produces no gradient at backward — DDP's
            # reducer expects either a grad or that the param be truly absent
            # from the forward graph (find_unused_parameters only catches the
            # latter).  Skip the heads entirely when there's no mask.
            if mask_t is not None and mask_t.any():
                pt_masked = pt[mask_t]                             # (N_masked, D)
                if hasattr(self.tx_encoder, "symbol_head"):
                    out["masked_symbol_logits"] = self.tx_encoder.symbol_head(pt_masked)
                if hasattr(self.tx_encoder, "value_head"):
                    out["masked_value_pred"] = self.tx_encoder.value_head(pt_masked).squeeze(-1)
                if hasattr(self.tx_encoder, "jepa_head"):
                    out["masked_jepa_pred"] = self.tx_encoder.jepa_head(pt_masked)
        return out

    # ------------------------------------------------------------------
    # Differential LR groups (UNI trunk vs. the rest).
    # ------------------------------------------------------------------
    def param_groups(self, base_lr: float, uni_lr_mult: float) -> list[dict]:
        is_uni = isinstance(self.image_encoder, UNIImageEncoder)
        if not is_uni or self.image_encoder.tune_mode in ("none",):
            rest = [p for p in self.parameters() if p.requires_grad]
            return [{"params": rest, "lr": base_lr, "name": "rest"}]
        trunk_params = [p for p in self.image_encoder.trunk.parameters() if p.requires_grad]
        trunk_ids = {id(p) for p in trunk_params}
        rest = [p for p in self.parameters()
                if p.requires_grad and id(p) not in trunk_ids]
        groups = []
        if trunk_params:
            groups.append({"params": trunk_params, "lr": base_lr * uni_lr_mult, "name": "uni_backbone"})
        groups.append({"params": rest, "lr": base_lr, "name": "rest"})
        return groups
