"""Spatial-Foundation encoder (Stage 1.5).

See docs/design/stage15_spatial_jepa.md for the full design.  Takes per-spot
(h_tx, h_img, x, y) tuples for a single sample, fuses them into a per-spot
token, then contextualises each token via its KNN neighborhood.

Three swappable backbones:
  - 'kgnn'      light GAT-like layer with Δxy edge features (default)
  - 'kxformer'  set-transformer with sparse KNN attention + relative pos bias
  - 'smooth'    non-parametric KNN mean (F2 control, no learnable params)
"""
from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Per-spot fuser: (h_tx, h_img, xy) → spot token
# ────────────────────────────────────────────────────────────────────────────

class _PosEnc2D(nn.Module):
    """Fourier-feature 2D positional encoding (translation-invariant after
    per-sample centring + scale)."""

    def __init__(self, dim: int = 32, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("B", torch.randn(2, dim // 2) * scale)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy : (..., 2)
        proj = xy @ self.B                          # (..., dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class SpotFuser(nn.Module):
    """Cell-level fuser: combine up to four modality channels per anchor cell
    into a single spatial-encoder token (the **fused** region-token mode).

        spot_tx     — Stage 1 tx_encoder CLS (frozen)
        spot_img    — UNI feature at the spot
        region_tx   — tx_encoder applied to aggregated neighbor expression
        region_img  — pooled neighbor UNI features
        + 2D positional encoding for the anchor xy

    region_* inputs are optional — when disabled, this reduces to the original
    single-modality fuser.  All present channels are concatenated then
    projected through a small MLP.  The result is one token per anchor spot,
    so the downstream backbone sees the same N as before — block masking
    works at the cell level (an anchor's whole token gets replaced).

    NOTE: in this mode, masking an anchor removes spot AND region context
    simultaneously.  See `SeparateTokenFuser` for the more I-JEPA-faithful
    "spot is target, region is visible context" variant.
    """

    def __init__(self, tx_dim: int, img_dim: int, fuse_dim: int = 256,
                 fuse_image: bool = True, fuse_region: bool = True,
                 region_img_dim: int | None = None,
                 region_tx_dim: int | None = None,
                 posenc_dim: int = 32):
        super().__init__()
        self.fuse_image = fuse_image
        self.fuse_region = fuse_region
        self.posenc = _PosEnc2D(posenc_dim)
        # Default: region tokens share dim with spot tokens (they go through
        # the same Stage-1 encoder / UNI features).
        region_tx_dim = region_tx_dim or tx_dim
        region_img_dim = region_img_dim or img_dim
        in_dim = tx_dim + posenc_dim
        if fuse_image:
            in_dim += img_dim
        if fuse_region:
            in_dim += region_tx_dim
            if fuse_image:
                in_dim += region_img_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, fuse_dim),
            nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim),
        )

    def forward(self,
                h_tx: torch.Tensor,
                h_img: torch.Tensor | None,
                xy: torch.Tensor,
                h_region_tx: torch.Tensor | None = None,
                h_region_img: torch.Tensor | None = None) -> torch.Tensor:
        """All inputs are (N, *).  Returns (N, fuse_dim)."""
        parts = [h_tx]
        if self.fuse_image and h_img is not None:
            parts.append(h_img)
        if self.fuse_region:
            if h_region_tx is None:
                # Caller should have provided region_tx when fuse_region=True;
                # fall back to zeros so smoke tests don't crash, but log a
                # warning by raising in training (Trainer should fill this).
                raise RuntimeError(
                    "SpotFuser.fuse_region=True but h_region_tx is None — "
                    "did you forget to run tx_encoder over region_hvg in the trainer?"
                )
            parts.append(h_region_tx)
            if self.fuse_image:
                if h_region_img is None:
                    raise RuntimeError(
                        "SpotFuser.fuse_region=True and fuse_image=True but "
                        "h_region_img is missing from the batch."
                    )
                parts.append(h_region_img)
        parts.append(self.posenc(xy))
        return self.proj(torch.cat(parts, dim=-1))


class SeparateTokenFuser(nn.Module):
    """Two-stream fuser: emits a spot token and a region token per anchor.

    Layout in the backbone sequence (concatenated, NOT interleaved — easier
    indexing for masking):

        positions 0   .. N-1   → spot tokens   (h_tx, h_img, xy, type=0)
        positions N   .. 2N-1  → region tokens (h_region_tx, h_region_img, xy, type=1)

    Graph edges in this mode (built by the caller, see SpatialEncoder.forward):

        spot↔spot     — original KNN edge_index over anchors
        region↔region — same edges shifted by +N
        spot↔region   — bidirectional self-link (i ↔ N+i) so each cell can
                        attend across token types

    The two-stream layout makes I-JEPA-style "predict masked spot from visible
    region context" a one-line mask: replace `[0..N-1][mask]` with mask_embed
    while leaving `[N..2N-1]` untouched.
    """

    def __init__(self, tx_dim: int, img_dim: int, fuse_dim: int = 256,
                 fuse_image: bool = True,
                 region_tx_dim: int | None = None,
                 region_img_dim: int | None = None,
                 posenc_dim: int = 32):
        super().__init__()
        self.fuse_image = bool(fuse_image)
        self.posenc = _PosEnc2D(posenc_dim)
        region_tx_dim = region_tx_dim or tx_dim
        region_img_dim = region_img_dim or img_dim
        # Spot proj: tx [+ img] + posenc → fuse_dim
        in_spot = tx_dim + posenc_dim + (img_dim if fuse_image else 0)
        self.spot_proj = nn.Sequential(
            nn.LayerNorm(in_spot), nn.Linear(in_spot, fuse_dim),
            nn.GELU(), nn.Linear(fuse_dim, fuse_dim),
        )
        # Region proj: region_tx [+ region_img] + posenc → fuse_dim
        in_reg = region_tx_dim + posenc_dim + (region_img_dim if fuse_image else 0)
        self.region_proj = nn.Sequential(
            nn.LayerNorm(in_reg), nn.Linear(in_reg, fuse_dim),
            nn.GELU(), nn.Linear(fuse_dim, fuse_dim),
        )
        # Learnable type embeddings (HyperST-style "spot vs niche" type bias).
        self.type_emb = nn.Parameter(torch.randn(2, fuse_dim) * 0.02)

    def forward(self,
                h_tx: torch.Tensor,
                h_img: torch.Tensor | None,
                xy: torch.Tensor,
                h_region_tx: torch.Tensor,
                h_region_img: torch.Tensor | None) -> torch.Tensor:
        """Returns (2N, fuse_dim).  positions 0..N-1 spot, N..2N-1 region."""
        if h_region_tx is None:
            raise RuntimeError(
                "SeparateTokenFuser requires h_region_tx (trainer must run "
                "frozen tx_encoder over region_hvg before forward).")
        pe = self.posenc(xy)                                   # (N, P)
        spot_parts = [h_tx, pe]
        if self.fuse_image and h_img is not None:
            spot_parts.insert(1, h_img)                        # tx | img | pe
        reg_parts = [h_region_tx, pe]
        if self.fuse_image:
            if h_region_img is None:
                raise RuntimeError(
                    "SeparateTokenFuser.fuse_image=True but h_region_img is None.")
            reg_parts.insert(1, h_region_img)
        spot_tok = self.spot_proj(torch.cat(spot_parts, dim=-1)) + self.type_emb[0]
        region_tok = self.region_proj(torch.cat(reg_parts, dim=-1)) + self.type_emb[1]
        return torch.cat([spot_tok, region_tok], dim=0)        # (2N, D)


def build_separate_edges(edge_index: torch.Tensor, n_nodes: int,
                            *, link_self: bool = True) -> torch.Tensor:
    """Build the 2N-node edge_index for SeparateTokenFuser.

    Sources of edges (all symmetric):
        spot-spot      : original edge_index               (E_s edges)
        region-region  : original + N                      (E_s edges)
        spot↔region    : (i, N+i) and (N+i, i)             (2N edges)
    """
    device = edge_index.device
    ss = edge_index                                            # spot-spot
    rr = edge_index + n_nodes                                  # region-region
    edges = [ss, rr]
    if link_self and n_nodes > 0:
        idx = torch.arange(n_nodes, device=device)
        sr = torch.stack([idx,             idx + n_nodes], dim=0)
        rs = torch.stack([idx + n_nodes,   idx],          dim=0)
        edges.extend([sr, rs])
    return torch.cat(edges, dim=1)


# ────────────────────────────────────────────────────────────────────────────
# Backbones
# ────────────────────────────────────────────────────────────────────────────

class _KGNNLayer(nn.Module):
    """Light GAT-style layer with edge features = MLP(Δxy)."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim == self.head_dim * n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, n_heads)
        )
        self.out = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, xy: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        # x (N, D), xy (N, 2), edge_index (2, E)  src→dst
        N, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(N, 3, H, Hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                  # (N, H, Hd)

        src, dst = edge_index[0], edge_index[1]
        dxy = xy[src] - xy[dst]
        e_bias = self.edge_mlp(dxy)                                # (E, H)

        # Per-edge attention score: q_dst · k_src + e_bias
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(Hd) + e_bias # (E, H)
        # softmax over each dst's incoming edges (segment softmax)
        score_max = torch.full((N, H), -1e9, device=x.device).index_reduce_(
            0, dst, score, "amax", include_self=True)
        score = (score - score_max[dst]).exp()
        denom = torch.zeros((N, H), device=x.device).index_add_(0, dst, score)
        weight = score / (denom[dst] + 1e-9)                        # (E, H)

        msg = v[src] * weight.unsqueeze(-1)                         # (E, H, Hd)
        out = torch.zeros_like(v).index_add_(0, dst, msg)            # (N, H, Hd)
        out = out.reshape(N, D)
        x = x + self.dropout(self.out(out))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class _SmoothBackbone(nn.Module):
    """Non-parametric KNN mean (F2 control)."""

    def __init__(self, dim: int, n_layers: int = 3, **_):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        # one identity proj so the module has a parameter (DDP requirement).
        self.identity = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.identity.weight)

    def forward(self, x: torch.Tensor, xy: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        for _ in range(self.n_layers):
            agg = torch.zeros_like(x).index_add_(0, dst, x[src])
            deg = torch.zeros(N, device=x.device).index_add_(
                0, dst, torch.ones(dst.shape[0], device=x.device))
            x = agg / (deg.unsqueeze(-1) + 1e-9)
        return self.identity(x)


class _KGNNBackbone(nn.Module):
    def __init__(self, dim: int, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _KGNNLayer(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, xy, edge_index):
        for layer in self.layers:
            x = layer(x, xy, edge_index)
        return self.norm(x)


# ────────────────────────────────────────────────────────────────────────────
# Top-level encoder
# ────────────────────────────────────────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """Stage 1.5 spatial encoder.  Frozen Stage 1 tx/img features → contextualised z.

    Token-mode options:
        `fused`     — SpotFuser collapses spot+region channels into ONE token
                       per anchor (N tokens total).  Masking removes all
                       channels at that anchor.  This is the cheap baseline.
        `separate`  — SeparateTokenFuser emits TWO tokens per anchor: a spot
                       token and a region token (2N total, layout [spots|regions]).
                       Masking can target spot only / region only / both,
                       which mirrors the HyperST + I-JEPA setup ("region as
                       visible context predicts masked spot latent").
    """

    MASK_EMB_INIT_STD = 0.02

    def __init__(self, *, tx_dim: int, img_dim: int,
                 fuse_dim: int = 256, fuse_image: bool = True,
                 fuse_region: bool = True,
                 token_mode: str = "fused",
                 region_tx_dim: int | None = None,
                 region_img_dim: int | None = None,
                 arch: str = "kgnn", n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        assert arch in ("kgnn", "kxformer", "smooth")
        assert token_mode in ("fused", "separate")
        if token_mode == "separate" and not fuse_region:
            raise ValueError("token_mode='separate' requires fuse_region=True "
                             "(region tokens are the second stream).")
        self.arch = arch
        self.fuse_region = fuse_region
        self.token_mode = token_mode
        if token_mode == "fused":
            self.fuse = SpotFuser(
                tx_dim, img_dim, fuse_dim,
                fuse_image=fuse_image, fuse_region=fuse_region,
                region_tx_dim=region_tx_dim, region_img_dim=region_img_dim,
            )
        else:
            self.fuse = SeparateTokenFuser(
                tx_dim, img_dim, fuse_dim,
                fuse_image=fuse_image,
                region_tx_dim=region_tx_dim, region_img_dim=region_img_dim,
            )
        if arch == "kgnn":
            self.backbone = _KGNNBackbone(fuse_dim, n_layers, n_heads, dropout)
        elif arch == "smooth":
            self.backbone = _SmoothBackbone(fuse_dim, n_layers)
        else:
            raise NotImplementedError("kxformer not yet implemented; use 'kgnn' or 'smooth'")
        # Learnable token for masked positions (shared across spot & region
        # streams — they live in the same fuse_dim latent space after MLP).
        self.mask_embed = nn.Parameter(torch.randn(fuse_dim) * self.MASK_EMB_INIT_STD)

    def forward(self, h_tx, h_img, xy, edge_index,
                 mask: torch.Tensor | None = None,
                 h_region_tx: torch.Tensor | None = None,
                 h_region_img: torch.Tensor | None = None):
        """All anchor tensors are (N, *).  Returns:
            fused mode    : (N, fuse_dim)
            separate mode : (2N, fuse_dim) — first N spot, next N region

        `mask` semantics depend on token_mode:
            fused mode    : (N,) bool   — masked anchor's whole token replaced.
            separate mode : (2N,) bool  — masked positions replaced.  The
                            caller (objective) builds the 2N mask by deciding
                            which stream(s) to mask.
        Edge_index passed in is the spot-spot KNN over N anchors; in separate
        mode we extend it to 2N internally.
        """
        if self.token_mode == "fused":
            x = self.fuse(h_tx, h_img, xy,
                           h_region_tx=h_region_tx, h_region_img=h_region_img)
            if mask is not None and mask.any():
                x = torch.where(mask.unsqueeze(-1), self.mask_embed.expand_as(x), x)
            return self.backbone(x, xy, edge_index)

        # separate mode — 2N tokens, expanded edges.
        n = h_tx.shape[0]
        x = self.fuse(h_tx, h_img, xy,
                       h_region_tx=h_region_tx, h_region_img=h_region_img)   # (2N, D)
        if mask is not None and mask.any():
            x = torch.where(mask.unsqueeze(-1), self.mask_embed.expand_as(x), x)
        # 2D coords doubled to (2N, 2) so _KGNNLayer's edge_mlp(Δxy) still works.
        xy2 = torch.cat([xy, xy], dim=0)
        e2 = build_separate_edges(edge_index, n_nodes=n)
        return self.backbone(x, xy2, e2)

    def n_streams(self) -> int:
        """Token multiplier: 1 for fused, 2 for separate."""
        return 2 if self.token_mode == "separate" else 1
