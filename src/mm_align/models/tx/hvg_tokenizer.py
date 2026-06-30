"""HVG-token gene encoder.

Design:

  values v ∈ ℝ^K   (top-K HVG log1p-normalized expression)
        │
        ├── Symbol embedding:  E_gene = Embedding(K, d)               ─► (B, K, d)
        │
        └── Value encoding:    Fourier features at multiple log-freqs ─► (B, K, 2F)
                                ↓
                                MLP(2F → d)                            ─► (B, K, d)

  token  = LayerNorm( E_gene + Value_embed )                           ─► (B, K, d)
  prepend [CLS]                                                         ─► (B, K+1, d)

  Transformer (Pre-LN, n_layers, n_heads)                              ─► (B, K+1, d)

  Heads:
    cls_out (B, d)                  ←  pool token #0
    per_token_out (B, K, d)         ←  tokens #1..K   (for masked-recon / masked-JEPA)

  Pretraining objectives (applied when training the gene encoder):
    masked-recon : per-token Linear(d → 1)  on masked positions  vs. true value
    masked-JEPA  : per-token Linear(d → d)  vs. EMA-teacher's *unmasked* per-token latent

Both are added to the existing align/gene-recon stack via UnifiedObjective when
`transcriptomics.kind == "hvg_tokenizer"` is set.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fourier value encoder
# ---------------------------------------------------------------------------

class FourierValueEncoder(nn.Module):
    """Fourier features of a scalar value + 2-layer MLP to dim d.

    For input value v, computes [cos(2π·f_i·v), sin(2π·f_i·v)] for F log-spaced
    frequencies, then projects (2F) → d.
    """

    def __init__(self, n_freqs: int, dim: int, *, max_freq: float = 32.0):
        super().__init__()
        # log-spaced frequencies (matches NeRF / Fourier features convention)
        freqs = 2.0 ** torch.linspace(0.0, math.log2(max_freq), n_freqs)
        self.register_buffer("freqs", freqs.view(1, 1, -1))   # (1, 1, F)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B, K) → (B, K, d)"""
        ang = 2 * math.pi * v.unsqueeze(-1) * self.freqs
        feats = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
        return self.mlp(feats)


# ---------------------------------------------------------------------------
# HVG token encoder
# ---------------------------------------------------------------------------

class HVGTokenEncoder(nn.Module):
    """Top-K HVG → per-gene transformer encoder.

    Outputs:
      cls_emb (B, dim)       — pooled representation (used as h_tx)
      tokens  (B, K, dim)    — per-gene contextualised tokens (for masked heads)
    """

    def __init__(self, *,
                 n_genes: int,
                 dim: int = 256,
                 n_freqs: int = 16,
                 max_freq: float = 32.0,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 cls_token: bool = True):
        super().__init__()
        self.n_genes = n_genes
        self.dim = dim
        self.use_cls = cls_token

        self.symbol_emb = nn.Embedding(n_genes, dim)
        self.value_emb = FourierValueEncoder(n_freqs=n_freqs, dim=dim, max_freq=max_freq)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.cls, std=0.02)

        self.ln_in = nn.LayerNorm(dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4 * dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(dim)

        # Pretraining heads — kept here so EMA teacher can deepcopy them.
        self.recon_head = nn.Linear(dim, 1)                  # masked-recon (per-token scalar)
        self.jepa_head = nn.Linear(dim, dim)                 # masked-JEPA predictor (student → teacher latent)

    # ------------------------------------------------------------------

    def _token_inputs(self, values: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build (B, K, dim) tokens (no CLS prepended). `mask` is (B, K) bool."""
        B, K = values.shape
        gene_ids = torch.arange(K, device=values.device).unsqueeze(0).expand(B, K)
        sym = self.symbol_emb(gene_ids)
        val = self.value_emb(values)
        tok = self.ln_in(sym + val)
        if mask is not None and mask.any():
            tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        return tok

    def forward(self, values: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> dict:
        """values: (B, K) HVG expression. mask: (B, K) bool, True at masked positions."""
        tok = self._token_inputs(values, mask=mask)
        if self.use_cls:
            B = tok.size(0)
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
        x = self.transformer(tok)
        x = self.ln_out(x)
        if self.use_cls:
            cls_emb, per_token = x[:, 0], x[:, 1:]
        else:
            cls_emb = x.mean(dim=1)
            per_token = x
        return {"h_tx": cls_emb, "per_token": per_token}
