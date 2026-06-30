"""Top-HVG gene encoder — SEAL/spatula-style with zero-expressed gene dropping.

Per-spot input is built dynamically:

  values v ∈ ℝ^K   (K = n_hvg HVG genes' log1p-normalized expression)
        │
  drop zero-expressed positions PER SPOT  →  variable-length real-token set
        │
  sort each row so real tokens are first; pad to L_max = max real tokens in batch
        │
  attention_mask[b, t] = 1 if real-expression at t else 0 (padding)
        │
  Symbol embed: nn.Embedding(V, D)   — PAD=0, MASK=1, CLS=2, UNK=3, then HVG genes
  Value  embed: Fourier(2π·B·v) + MLP — B∼N(0,σ²) buffer
  GeneEmbedding = LayerNorm(sym + val)
        │
  prepend [CLS]  (learnable)
        ↓
  Pre-LN TransformerEncoder w/ key_padding_mask  (NO positional embedding →
  permutation-invariant set encoder, as in spatula)
        ↓
  LayerNorm
        ↓
  cls_emb       ← x[:, 0]      → cls_proj → h_tx (B, out_dim)
  per_token     ← x[:, 1:]     → masked-modeling heads

Why drop zeros?
  Most HVG genes are 0-expressed in any given spot (counts data).  Padding the
  whole 2048-vec to the model wastes attention capacity on uninformative 0
  tokens and dilutes the co-expression signal — masked-symbol prediction
  becomes "guess from a sea of zeros".  Dropping zeros gives a compact,
  spot-specific "expressed gene set" which is what biology cares about.

  Downstream gene-expression prediction can still output (B, K) by routing
  per-token predictions back through `gene_ids_in_seq` (kept in model output).

Heads:
  symbol_head : Linear(D, V)   — CE over masked symbols
  value_head  : Linear(D, 1)   — MSE on masked values
  jepa_head   : Linear(D, D)   — cosine-to-EMA teacher (optional)
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


# ───────────────────────────── building blocks ─────────────────────────────

class _SymbolEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        nn.init.xavier_uniform_(self.emb.weight)
        with torch.no_grad():
            self.emb.weight[pad_token_id].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class _ValueEmbedding(nn.Module):
    def __init__(self, dim: int, fourier_dim: int = 64, fourier_scale: float = 1.0):
        super().__init__()
        assert fourier_dim % 2 == 0
        n_freq = fourier_dim // 2
        B = torch.randn(n_freq) * fourier_scale
        self.register_buffer("B_freq", B)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        proj = 2 * math.pi * v.unsqueeze(-1) * self.B_freq.float()
        feats = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return self.mlp(feats)


class _GeneEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int,
                 fourier_dim: int = 64, fourier_scale: float = 1.0,
                 pad_token_id: int = 0):
        super().__init__()
        self.symbol = _SymbolEmbedding(vocab_size, dim, pad_token_id)
        self.value = _ValueEmbedding(dim, fourier_dim, fourier_scale)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.symbol(ids) + self.value(values))


# ─────────────────────────────── encoder ──────────────────────────────────

class TopHVGGeneEncoder(nn.Module):
    PAD_ID = 0
    MASK_ID = 1
    CLS_ID = 2
    UNK_ID = 3
    N_SPECIAL = 4

    def __init__(self, *,
                 vocab_size: int,
                 n_hvg: int,
                 out_dim: int,
                 dim: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 fourier_dim: int = 64,
                 fourier_scale: float = 1.0,
                 dropout: float = 0.1,
                 mask_ratio: float = 0.30,
                 mask_kind: str = "both",
                 min_seq_len: int = 1,
                 value_aug_mode: str = "keep",
                 value_aug_noise_std: float = 1.0,
                 value_aug_keep_p: float = 0.80,
                 value_aug_noise_p: float = 0.10,
                 value_aug_drop_p: float = 0.10,
                 masked_value_aug_mode: str | None = None,
                 masked_value_aug_noise_std: float | None = None,
                 masked_value_aug_keep_p: float | None = None,
                 masked_value_aug_noise_p: float | None = None,
                 masked_value_aug_drop_p: float | None = None,
                 unmasked_value_aug_mode: str = "keep",
                 unmasked_value_aug_noise_std: float = 0.15,
                 unmasked_value_aug_keep_p: float = 0.90,
                 unmasked_value_aug_noise_p: float = 0.10,
                 unmasked_value_aug_drop_p: float = 0.0,
                 pooling_mode: str = "cls"):
        """
        Value augmentation is split by token role when mask_kind == 'symbol'.

        masked_value_aug
            Applied to symbol-masked target positions. It prevents a direct
            value→symbol shortcut while retaining enough expression signal for
            a meaningful MSM task.

        unmasked_value_aug
            Applied to unmasked context positions. It should usually be weak:
            context values are the sentence around the masked gene, so heavy
            corruption makes the task noisy rather than biologically useful.

        Legacy value_aug_* arguments are kept as a fallback and map to
        masked_value_aug for old configs/scripts.
        """
        super().__init__()
        assert mask_kind in ("symbol", "value", "both")
        valid_aug = ("keep", "noise", "dropout", "mixed")
        assert value_aug_mode in valid_aug
        if masked_value_aug_mode is not None:
            assert masked_value_aug_mode in valid_aug
        assert unmasked_value_aug_mode in valid_aug
        self.kind = "top_hvg_gene"
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_hvg = n_hvg
        self.mask_ratio = mask_ratio
        self.mask_kind = mask_kind
        self.min_seq_len = min_seq_len     # safety floor when a spot has all zeros
        pooling_alias = {
            "cls_mean_sum": "cls_token_mean_sum",
            "cls_mean_avg": "cls_token_mean_avg",
            "mean": "token_mean",
        }
        pooling_mode = pooling_alias.get(str(pooling_mode), str(pooling_mode))
        valid_pooling = {"cls", "token_mean", "cls_token_mean_sum", "cls_token_mean_avg"}
        if pooling_mode not in valid_pooling:
            raise ValueError(f"pooling_mode must be one of {sorted(valid_pooling)}, got {pooling_mode!r}")
        self.pooling_mode = pooling_mode
        def _norm_probs(keep_p: float, noise_p: float, drop_p: float) -> tuple[float, float, float]:
            total = max(1e-9, float(keep_p) + float(noise_p) + float(drop_p))
            return float(keep_p) / total, float(noise_p) / total, float(drop_p) / total

        # Legacy alias: train.py's clean-MSM path may temporarily set this to
        # "keep"; forward treats that as disabling masked value corruption.
        self.value_aug_mode = masked_value_aug_mode or value_aug_mode
        self.value_aug_noise_std = float(
            value_aug_noise_std if masked_value_aug_noise_std is None else masked_value_aug_noise_std
        )
        mkp = value_aug_keep_p if masked_value_aug_keep_p is None else masked_value_aug_keep_p
        mnp = value_aug_noise_p if masked_value_aug_noise_p is None else masked_value_aug_noise_p
        mdp = value_aug_drop_p if masked_value_aug_drop_p is None else masked_value_aug_drop_p
        self.value_aug_keep_p, self.value_aug_noise_p, self.value_aug_drop_p = _norm_probs(mkp, mnp, mdp)

        self.unmasked_value_aug_mode = unmasked_value_aug_mode
        self.unmasked_value_aug_noise_std = float(unmasked_value_aug_noise_std)
        (self.unmasked_value_aug_keep_p,
         self.unmasked_value_aug_noise_p,
         self.unmasked_value_aug_drop_p) = _norm_probs(
             unmasked_value_aug_keep_p, unmasked_value_aug_noise_p, unmasked_value_aug_drop_p
         )

        self.gene_emb = _GeneEmbedding(
            vocab_size=vocab_size, dim=dim,
            fourier_dim=fourier_dim, fourier_scale=fourier_scale,
            pad_token_id=self.PAD_ID,
        )

        # Token IDs for each HVG slot (gene #i → token_id N_SPECIAL+i).
        hvg_ids = torch.arange(self.N_SPECIAL, self.N_SPECIAL + n_hvg, dtype=torch.long)
        self.register_buffer("hvg_token_ids", hvg_ids)         # (K,)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4 * dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

        self.symbol_head = nn.Linear(dim, vocab_size)
        self.value_head = nn.Linear(dim, 1)
        self.jepa_head = nn.Linear(dim, dim)

        self.cls_proj = nn.Linear(dim, out_dim)

    # ------------------------------------------------------------------

    def _pool_spot(self, cls_emb: torch.Tensor, per_token: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the spot-level readout before the final projection.

        `cls` is the historical default. The token-mean variants are
        checkpoint-compatible readout ablations: they add no parameters and
        combine the [CLS] summary with the mean of valid gene-token latents.
        """
        if self.pooling_mode == "cls":
            return cls_emb

        weights = attention_mask.to(dtype=per_token.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        token_mean = (per_token * weights).sum(dim=1) / denom

        if self.pooling_mode == "token_mean":
            return token_mean
        if self.pooling_mode == "cls_token_mean_sum":
            return cls_emb + token_mean
        if self.pooling_mode == "cls_token_mean_avg":
            return 0.5 * (cls_emb + token_mean)
        raise RuntimeError(f"unhandled pooling_mode={self.pooling_mode!r}")

    # ------------------------------------------------------------------

    def sample_mask(self, values: torch.Tensor) -> torch.Tensor:
        """No-op kept for back-compat with aligner code.  The encoder samples
        its own mask internally (restricted to real / non-padding tokens)."""
        return None

    # ------------------------------------------------------------------

    def _pack_nonzero(self, hvg: torch.Tensor) -> dict:
        """Drop zero-expressed positions per row; pad to batch max real length.

        Returns
        -------
        gene_ids       : (B, L_max) int64   — token id of each real gene, PAD elsewhere
        values         : (B, L_max) float32 — value of each real gene, 0 elsewhere
        attention_mask : (B, L_max) int64   — 1 for real, 0 for padding
        orig_positions : (B, L_max) int64   — original HVG slot in 0..K-1 (for routing)
        """
        B, K = hvg.shape
        device = hvg.device

        # Real positions per row.
        real = hvg > 0                                   # (B, K) bool
        seq_lens = real.sum(dim=1).clamp(min=self.min_seq_len)   # (B,)
        L_max = int(seq_lens.max().item())
        L_max = max(L_max, self.min_seq_len)

        # Per-spot random gene permutation among the REAL positions.
        #
        # The gene-token sequence is an unordered set in biology; a stable
        # argsort would always place the same lowest-HVG-index gene at
        # sequence position 0, creating an implicit positional bias the
        # transformer can exploit even though no positional embeddings are
        # added.  At training we assign rand keys to real entries (and a +1
        # offset to padding so it always sinks to the tail) and argsort by
        # those keys, which shuffles real genes per spot per step.  Eval
        # keeps the deterministic canonical ordering so downstream caches
        # (h_tx cache, val metrics, shard-level reproducibility) match.
        if self.training or getattr(self, "_force_shuffle", False):
            rand_key = torch.rand((B, K), device=device, dtype=torch.float32)
            sort_key = torch.where(real, rand_key, rand_key + 1.0)
            order = torch.argsort(sort_key, dim=1, stable=False)
        else:
            order = torch.argsort((~real).long(), dim=1, stable=True)
        order = order[:, :L_max]                                       # (B, L_max)

        gene_ids = self.hvg_token_ids[order]                           # (B, L_max)
        values = torch.gather(hvg, 1, order)                           # (B, L_max)
        # Replace fake "real" positions (when a row has fewer than L_max non-zeros)
        # with PAD / 0 — those are at the tail.
        arange = torch.arange(L_max, device=device).unsqueeze(0)       # (1, L_max)
        attn = (arange < seq_lens.unsqueeze(1)).long()                 # (B, L_max)
        gene_ids = torch.where(attn.bool(), gene_ids, torch.full_like(gene_ids, self.PAD_ID))
        values = torch.where(attn.bool(), values, torch.zeros_like(values))

        return {"gene_ids": gene_ids, "values": values,
                "attention_mask": attn, "orig_positions": order}

    # ------------------------------------------------------------------

    def _apply_value_aug(self, values: torch.Tensor, positions: torch.Tensor, *,
                         mode: str, keep_p: float, noise_p: float,
                         noise_std: float) -> torch.Tensor:
        if mode == "keep" or positions is None or not positions.any():
            return values
        rand = torch.rand_like(values)
        if mode == "noise":
            do_noise = positions & (rand >= keep_p)
            noise = torch.randn_like(values) * noise_std
            return torch.where(do_noise, values + noise, values)
        if mode == "dropout":
            do_drop = positions & (rand >= keep_p)
            return torch.where(do_drop, torch.zeros_like(values), values)
        # mixed: keep / noise / dropout partition.
        thresh_noise = keep_p
        thresh_drop = keep_p + noise_p
        do_noise = positions & (rand >= thresh_noise) & (rand < thresh_drop)
        do_drop = positions & (rand >= thresh_drop)
        noise = torch.randn_like(values) * noise_std
        values = torch.where(do_noise, values + noise, values)
        return torch.where(do_drop, torch.zeros_like(values), values)

    # ------------------------------------------------------------------

    def forward(self, novae_latent: Optional[torch.Tensor],
                hvg: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> dict:
        assert hvg is not None
        assert hvg.shape[1] == self.n_hvg, f"expected K={self.n_hvg}, got {hvg.shape[1]}"

        packed = self._pack_nonzero(hvg)
        gene_ids = packed["gene_ids"]              # (B, L_max)
        values = packed["values"]                  # (B, L_max)
        attn = packed["attention_mask"]            # (B, L_max)
        orig_positions = packed["orig_positions"]  # (B, L_max)
        B, L_max = gene_ids.shape
        device = gene_ids.device

        # Internal mask sampling (restricted to real tokens only).  Gated on
        # training to keep inference deterministic; the trainer flips
        # `_force_mask_in_eval` for the Stage-1 val pass so we still get a
        # masked-modeling loss signal.
        do_sample = self.training or getattr(self, "_force_mask_in_eval", False)
        if mask is None and do_sample and self.mask_ratio > 0:
            rand = torch.rand((B, L_max), device=device)
            mask = (rand < self.mask_ratio) & attn.bool()

        # Apply masking — symbol and/or value at masked positions.
        orig_gene_ids = gene_ids.clone()
        if mask is not None and mask.any():
            if self.mask_kind in ("symbol", "both"):
                gene_ids = torch.where(mask, torch.full_like(gene_ids, self.MASK_ID), gene_ids)
            if self.mask_kind in ("value", "both"):
                values = torch.where(mask, torch.zeros_like(values), values)
            elif self.mask_kind == "symbol":
                do_value_aug = self.training or getattr(self, "_force_mask_in_eval", False)
                if do_value_aug:
                    # Target positions: [MASK] symbol plus partly corrupted value.
                    values = self._apply_value_aug(
                        values, mask,
                        mode=self.value_aug_mode,
                        keep_p=self.value_aug_keep_p,
                        noise_p=self.value_aug_noise_p,
                        noise_std=self.value_aug_noise_std,
                    )
                    # Context positions: original symbol plus weak value noise/dropout.
                    context_pos = attn.bool() & ~mask
                    values = self._apply_value_aug(
                        values, context_pos,
                        mode=self.unmasked_value_aug_mode,
                        keep_p=self.unmasked_value_aug_keep_p,
                        noise_p=self.unmasked_value_aug_noise_p,
                        noise_std=self.unmasked_value_aug_noise_std,
                    )

        # Embed + prepend CLS.
        x = self.gene_emb(gene_ids, values)            # (B, L_max, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                  # (B, L_max+1, D)

        # Build key_padding_mask (True = ignore = padding).
        cls_attn = torch.ones(B, 1, dtype=attn.dtype, device=device)
        attn_w_cls = torch.cat([cls_attn, attn], dim=1)
        kpm = (attn_w_cls == 0)
        x = self.transformer(x, src_key_padding_mask=kpm)
        x = self.norm(x)

        cls_emb = x[:, 0]                              # (B, D)
        per_token = x[:, 1:]                            # (B, L_max, D)
        spot_emb = self._pool_spot(cls_emb, per_token, attn)

        return {
            "h_tx": self.cls_proj(spot_emb),
            "per_token": per_token,
            "mask": mask,                              # (B, L_max) bool or None
            "orig_gene_ids": orig_gene_ids,            # (B, L_max) targets for CE
            "attention_mask": attn,                    # (B, L_max) 1=real,0=pad
            "orig_positions": orig_positions,          # (B, L_max) in 0..K-1 — for routing back
        }
