"""Tx-side encoder — two kinds switchable via config:

  `kind: "novae_adapter"`  (default)
    Small MLP over the *pre-extracted, frozen* Novae 64-d latent (and optional
    HVG vector concatenated).  Novae itself is NOT run during training.
    Use this when you want the simplest, fastest tx branch.

  `kind: "hvg_tokenizer"`  (new — SEAL-style learned gene encoder)
    Transformer over per-gene tokens, where each token = gene-symbol embedding
    + Fourier-encoded value.  Supports masked-JEPA + masked-recon objectives
    via UnifiedObjective.  See models/hvg_tokenizer.py.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .hvg_tokenizer import HVGTokenEncoder
from .top_hvg_gene import TopHVGGeneEncoder


def _mlp(in_dim: int, hidden: int, out: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(cur, hidden), nn.GELU(), nn.Dropout(dropout)]
        cur = hidden
    layers += [nn.Linear(cur, out)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------

class NovaeAdapterTxEncoder(nn.Module):
    """MLP adapter on pre-extracted Novae (+ optional HVG) → out_dim."""

    def __init__(self, *, novae_dim: int, hvg_dim: int,
                 hidden_dim: int, out_dim: int,
                 use_novae: bool = True, use_hvg: bool = True,
                 n_layers: int = 2, dropout: float = 0.1,
                 freeze: bool = False):
        super().__init__()
        assert use_novae or use_hvg
        self.use_novae = use_novae
        self.use_hvg = use_hvg
        self.in_dim = (novae_dim if use_novae else 0) + (hvg_dim if use_hvg else 0)
        self.norm = nn.LayerNorm(self.in_dim)
        self.net = _mlp(self.in_dim, hidden_dim, out_dim, n_layers, dropout)
        self.kind = "novae_adapter"
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, novae_latent: Optional[torch.Tensor],
                hvg: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> dict:
        parts = []
        if self.use_novae:
            parts.append(novae_latent)
        if self.use_hvg:
            parts.append(hvg)
        x = torch.cat(parts, dim=-1)
        return {"h_tx": self.net(self.norm(x)), "per_token": None}


# ---------------------------------------------------------------------------

class HVGTokenizerTxEncoder(nn.Module):
    """Wrap HVGTokenEncoder + a final projection to out_dim (=embed_dim).
    Exposes the masked-modeling heads on the underlying HVGTokenEncoder so
    UnifiedObjective can use them."""

    def __init__(self, *, n_genes: int, out_dim: int,
                 dim: int = 256, n_freqs: int = 16, max_freq: float = 32.0,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                 cls_token: bool = True,
                 mask_ratio: float = 0.3):
        super().__init__()
        self.kind = "hvg_tokenizer"
        self.mask_ratio = mask_ratio
        self.token_dim = dim
        self.encoder = HVGTokenEncoder(
            n_genes=n_genes, dim=dim, n_freqs=n_freqs, max_freq=max_freq,
            n_layers=n_layers, n_heads=n_heads, dropout=dropout, cls_token=cls_token,
        )
        # Project cls / per-token features to the shared embed_dim.
        self.cls_proj = nn.Linear(dim, out_dim)

    # ------------------------------------------------------------------

    def sample_mask(self, hvg: torch.Tensor) -> torch.Tensor:
        """Bernoulli mask over the K HVG tokens, shape (B, K) bool."""
        return torch.rand_like(hvg) < self.mask_ratio

    def forward(self, novae_latent: Optional[torch.Tensor],
                hvg: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> dict:
        assert hvg is not None, "HVGTokenizerTxEncoder requires HVG values in batch['hvg']"
        out = self.encoder(hvg, mask=mask)
        return {
            "h_tx": self.cls_proj(out["h_tx"]),
            "per_token": out["per_token"],   # (B, K, dim_token)
            "mask": mask,
        }


# ---------------------------------------------------------------------------
# Public alias / factory
# ---------------------------------------------------------------------------

def build_tx_encoder(cfg: dict) -> nn.Module:
    tcfg = cfg["model"]["transcriptomics"]
    kind = tcfg.get("kind", "novae_adapter")
    embed_dim = cfg["model"]["embed_dim"]

    if kind == "novae_adapter":
        return NovaeAdapterTxEncoder(
            novae_dim=tcfg["novae_in_dim"], hvg_dim=tcfg["hvg_in_dim"],
            hidden_dim=tcfg["hidden_dim"], out_dim=embed_dim,
            use_novae=tcfg["use_novae"], use_hvg=tcfg["use_hvg"],
            n_layers=tcfg["n_layers"], dropout=tcfg["dropout"],
            freeze=bool(tcfg.get("freeze", False)),
        )

    if kind == "top_hvg_gene":
        # SEAL/spatula-style: symbol + Fourier-value + Transformer, with vocab.
        gcfg = tcfg.get("top_hvg_gene", {})
        # Resolve vocab.  Prefer hvg_vocab_dict.json (new prepare_data output).
        from pathlib import Path
        import json
        prepared_dir = Path(cfg["data"]["prepared_dir"])
        vp = gcfg.get("vocab_path") or str(prepared_dir / "hvg_vocab_dict.json")
        try:
            with open(vp) as f:
                vocab = json.load(f)
            vocab_size = len(vocab)
        except FileNotFoundError:
            # Fallback when vocab file not yet emitted (smoke tests etc.)
            vocab_size = gcfg.get("vocab_size", 4 + tcfg["hvg_in_dim"])

        # Value augmentation knobs. New schema separates target [MASK]
        # positions from unmasked context positions. Legacy value_aug maps to
        # masked_value_aug for older configs/scripts.
        _legacy_vacfg = gcfg.get("value_aug") or {}
        _masked_vacfg = gcfg.get("masked_value_aug") or _legacy_vacfg
        _unmasked_vacfg = gcfg.get("unmasked_value_aug") or {}
        return TopHVGGeneEncoder(
            vocab_size=vocab_size,
            n_hvg=tcfg["hvg_in_dim"],
            out_dim=embed_dim,
            dim=gcfg.get("dim", 256),
            n_layers=gcfg.get("n_layers", 4),
            n_heads=gcfg.get("n_heads", 4),
            fourier_dim=gcfg.get("fourier_dim", 64),
            fourier_scale=gcfg.get("fourier_scale", 1.0),
            dropout=gcfg.get("dropout", 0.1),
            mask_ratio=gcfg.get("mask_ratio", 0.30),
            mask_kind=gcfg.get("mask_kind", "both"),
            value_aug_mode=_masked_vacfg.get("mode", "keep"),
            value_aug_noise_std=float(_masked_vacfg.get("noise_std", 1.0)),
            value_aug_keep_p=float(_masked_vacfg.get("keep_p", 0.80)),
            value_aug_noise_p=float(_masked_vacfg.get("noise_p", 0.10)),
            value_aug_drop_p=float(_masked_vacfg.get("drop_p", 0.10)),
            masked_value_aug_mode=_masked_vacfg.get("mode", None),
            masked_value_aug_noise_std=float(_masked_vacfg.get("noise_std", 1.0)),
            masked_value_aug_keep_p=float(_masked_vacfg.get("keep_p", 0.80)),
            masked_value_aug_noise_p=float(_masked_vacfg.get("noise_p", 0.10)),
            masked_value_aug_drop_p=float(_masked_vacfg.get("drop_p", 0.10)),
            unmasked_value_aug_mode=_unmasked_vacfg.get("mode", "keep"),
            unmasked_value_aug_noise_std=float(_unmasked_vacfg.get("noise_std", 0.15)),
            unmasked_value_aug_keep_p=float(_unmasked_vacfg.get("keep_p", 0.90)),
            unmasked_value_aug_noise_p=float(_unmasked_vacfg.get("noise_p", 0.10)),
            unmasked_value_aug_drop_p=float(_unmasked_vacfg.get("drop_p", 0.0)),
            pooling_mode=gcfg.get("pooling_mode", "cls"),
        )

    if kind == "hvg_tokenizer":
        hcfg = tcfg.get("hvg_tokenizer", {})
        return HVGTokenizerTxEncoder(
            n_genes=tcfg["hvg_in_dim"], out_dim=embed_dim,
            dim=hcfg.get("dim", 256),
            n_freqs=hcfg.get("n_freqs", 16),
            max_freq=hcfg.get("max_freq", 32.0),
            n_layers=hcfg.get("n_layers", 4),
            n_heads=hcfg.get("n_heads", 4),
            dropout=hcfg.get("dropout", 0.1),
            cls_token=hcfg.get("cls_token", True),
            mask_ratio=hcfg.get("mask_ratio", 0.3),
        )

    raise ValueError(f"Unknown tx encoder kind: {kind}")


# Back-compat alias — old code imports TranscriptEncoder.
class TranscriptEncoder(NovaeAdapterTxEncoder):
    def __init__(self, novae_dim, hvg_dim, hidden_dim, out_dim,
                 use_novae=True, use_hvg=True, n_layers=2, dropout=0.1, freeze=False):
        super().__init__(novae_dim=novae_dim, hvg_dim=hvg_dim,
                         hidden_dim=hidden_dim, out_dim=out_dim,
                         use_novae=use_novae, use_hvg=use_hvg,
                         n_layers=n_layers, dropout=dropout, freeze=freeze)

    def forward(self, novae_latent, hvg):
        # Old call sig returned the latent tensor directly.
        return super().forward(novae_latent, hvg)["h_tx"]
