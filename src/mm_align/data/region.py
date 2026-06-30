"""Region aggregation — build region-level tx + image features.

Naming note: we deliberately avoid "niche" (the HyperST term) because it
overloads with biology jargon.  In this codebase a **region** is simply the
ball/set "anchor spot + its k spatial neighbors".  Per anchor spot we emit
two extra tokens beyond the spot itself:

    spot_tx, spot_img, region_tx, region_img

so the Stage-1.5 SpatialEncoder sees 4 tokens per anchor, ST-JEPA style.

Aggregation strategies (configurable via `region.tx_agg` / `region.img_pool`):

    Transcriptomics
    ---------------
    mean        Σ log1p(x_i) / (k + 1)               — average in log space
    sum_log1p   log1p( Σ exp(log1p(x_i)) - 1 )       — sum in raw space, then re-log1p
    weighted    distance-weighted mean (w_i ∝ exp(-d_i/σ))

    Image
    -----
    mean        Σ uni_feat[i] / (k + 1)               — average
    attn        learned attention pool (placeholder; defaults to mean for now)

The raw-patch variant (resize+UNI-LoRA) lives in a separate prepare step
and is consumed via a precomputed `region_uni_feat` shard column when present.
"""
from __future__ import annotations
from typing import Literal

import numpy as np
import torch


TxAggKind = Literal["mean", "sum_log1p", "weighted"]
ImgPoolKind = Literal["mean", "attn"]


# ────────────────────────────────────────────────────────────────────────────
# Transcriptomics aggregation
# ────────────────────────────────────────────────────────────────────────────

def aggregate_region_tx(
    hvg_block: np.ndarray,                # (n_anchor + n_neighbors, n_vocab) log1p values
    neighbor_index: np.ndarray,           # (n_anchor, k) — each row is the k neighbor positions into hvg_block
    *,
    kind: TxAggKind = "mean",
    distances: np.ndarray | None = None,  # (n_anchor, k) — optional, for weighted
    sigma: float = 1.0,
    include_anchor: bool = False,
) -> np.ndarray:
    """Return region-level tx (n_anchor, n_vocab) given block + neighbor index.

    `include_anchor`:
        False (default, JEPA-faithful)  — region = neighbours-only.  Required
            when the SpatialJEPAObjective masks the spot stream and predicts
            its latent from the region context: if the anchor's own RNA leaks
            into the region, the prediction is trivial.
        True                            — region = anchor + neighbours
            (HyperST "niche" definition).  Acceptable in fused-token mode
            since the whole cell-level token is masked together.

    Memory note: this runs on the SUBGRAPH level (a few hundred spots), not
    on the full sample.  `hvg_block` is already clipped to the subgraph's
    spots and (optionally) vocab-clipped.
    """
    n_anchor, k = neighbor_index.shape

    def _valid_rows(i: int) -> np.ndarray:
        """Return the rows to aggregate for anchor i.  Drops -1 sentinels
        (used by `build_neighbor_index` when an anchor has < k real neighbours
        and `pad_with_self=False`)."""
        nbr = neighbor_index[i]
        nbr = nbr[nbr >= 0]
        return (np.concatenate([[i], nbr]) if include_anchor else nbr)

    if kind == "mean":
        out = np.empty((n_anchor, hvg_block.shape[1]), dtype=np.float32)
        for i in range(n_anchor):
            rows = _valid_rows(i)
            out[i] = hvg_block[rows].mean(axis=0) if rows.size else 0.0
        return out

    if kind == "sum_log1p":
        # Sum in RAW space (un-log first), then re-log1p.  Zero positions
        # stay zero (expm1(0)=0) — preserves zero-removal semantics.
        out = np.empty((n_anchor, hvg_block.shape[1]), dtype=np.float32)
        for i in range(n_anchor):
            rows = _valid_rows(i)
            if rows.size:
                raw_sum = np.expm1(hvg_block[rows]).sum(axis=0)
                out[i] = np.log1p(np.maximum(raw_sum, 0.0))
            else:
                out[i] = 0.0
        return out

    if kind == "weighted":
        if distances is None:
            raise ValueError("kind='weighted' requires `distances` argument")
        # `distances` is (n_anchor, k) for neighbours.  Sentinel -1 in
        # neighbor_index → drop that column's weight too.
        weights_nb = np.exp(-distances / max(sigma, 1e-6))      # (n_anchor, k)
        out = np.empty((n_anchor, hvg_block.shape[1]), dtype=np.float32)
        for i in range(n_anchor):
            valid = neighbor_index[i] >= 0
            nbr_rows = neighbor_index[i][valid]
            w_nb = weights_nb[i][valid]
            if include_anchor:
                w = np.concatenate([[1.0], w_nb])
                rows = np.concatenate([[i], nbr_rows])
            else:
                w = w_nb
                rows = nbr_rows
            if not rows.size:
                out[i] = 0.0
                continue
            w = w / max(float(w.sum()), 1e-9)
            out[i] = (hvg_block[rows] * w[:, None]).sum(axis=0)
        return out

    raise ValueError(f"unknown region tx_agg kind: {kind!r}")


# ────────────────────────────────────────────────────────────────────────────
# Image aggregation
# ────────────────────────────────────────────────────────────────────────────

def aggregate_region_img(
    uni_feat_block: np.ndarray,           # (n_spots_in_subgraph, D_img=1536)
    neighbor_index: np.ndarray,           # (n_anchor, k)
    *,
    kind: ImgPoolKind = "mean",
    include_anchor: bool = False,
) -> np.ndarray:
    """Return region-level image features (n_anchor, D_img).

    See `aggregate_region_tx` for the semantics of `include_anchor`.
    """
    n_anchor, k = neighbor_index.shape

    def _valid_rows(i: int) -> np.ndarray:
        nbr = neighbor_index[i]
        nbr = nbr[nbr >= 0]
        return (np.concatenate([[i], nbr]) if include_anchor else nbr)

    if kind == "mean":
        out = np.empty((n_anchor, uni_feat_block.shape[1]), dtype=np.float32)
        for i in range(n_anchor):
            rows = _valid_rows(i)
            out[i] = uni_feat_block[rows].mean(axis=0) if rows.size else 0.0
        return out
    if kind == "attn":
        # TODO: learnable attention pool (would need a small attention head
        # exposed by SpatialEncoder).  For now fall back to mean — the ablation
        # script will report both `mean` and `attn` once that head exists.
        return aggregate_region_img(uni_feat_block, neighbor_index,
                                     kind="mean", include_anchor=include_anchor)
    raise ValueError(f"unknown region img_pool kind: {kind!r}")


# ────────────────────────────────────────────────────────────────────────────
# Helpers used by SpatialSampleDataset / SpatialEncoder
# ────────────────────────────────────────────────────────────────────────────

def build_neighbor_index(edge_index: np.ndarray, n_nodes: int, k: int,
                         pad_with_self: bool = True) -> np.ndarray:
    """From a generic edge_index (2, E) build a dense (n_nodes, k) neighbor
    matrix.  Nodes with fewer than k neighbors get self-padded so downstream
    aggregation always has a well-defined mean.

    Returns int64 indices into the same node ordering as `edge_index`.
    """
    src, dst = edge_index[0], edge_index[1]
    neigh: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for s, d in zip(src, dst):
        s, d = int(s), int(d)
        if s == d:
            continue
        if d not in neigh[s]:
            neigh[s].append(d)
    out = np.full((n_nodes, k), -1, dtype=np.int64)
    for i in range(n_nodes):
        nb = neigh[i][:k]
        if len(nb) < k and pad_with_self:
            nb = nb + [i] * (k - len(nb))
        out[i, : len(nb)] = nb[:k]
    return out
