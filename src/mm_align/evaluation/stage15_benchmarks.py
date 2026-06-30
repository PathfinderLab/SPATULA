"""Stage 1.5 (Spatial Foundation) in-distribution evaluation metrics.

Measures whether the spatial encoder has actually learned spatial structure
— complementary to the JEPA val_loss, which can decrease via trivial EMA
collapse.

Metrics (all label-free, computed per sample then macro-averaged):

  spatial_knn_overlap   Jaccard(embedding kNN, spatial kNN).  For each
                         spot, what fraction of its top-k embedding
                         neighbours are also among its top-k spatial
                         neighbours.  HIGH = the embedding captures
                         the same locality the data has.

  spatial_smoothness    Spearman( embedding cosine vs spatial Δ-distance ).
                         For random spot pairs, do embeddings get less
                         similar as spatial distance grows?  HIGH (positive)
                         = embedding respects spatial structure but does
                         NOT collapse to one cluster.

  boundary_preservation Mean inter-domain / mean intra-domain embedding
                         distance, using Novae niche labels (when present)
                         or sample-level coarse domains otherwise.  > 1
                         means inter-domain pairs are further apart than
                         intra-domain — the encoder respects boundaries.

  augmentation_consistency  For each anchor, cos_sim of its embedding
                         under two independent ego-subgraphs.  HIGH = the
                         encoder is stable under random subgraph sampling.

  effective_rank         exp(entropy) of singular-value distribution of
                         per-sample embedding matrix.  Detects collapse.

All metrics live in [0, 1] except boundary_preservation (> 0, > 1 desired).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# 1. kNN overlap — embedding KNN vs spatial KNN
# ───────────────────────────────────────────────────────────────────────────

def spatial_knn_overlap(emb: np.ndarray, xy: np.ndarray, k: int = 10) -> float:
    """For each spot, Jaccard between its k-NN in embedding space and its
    k-NN in spatial space.  Returns mean Jaccard over spots."""
    n = emb.shape[0]
    if n < k + 2:
        return float("nan")
    from sklearn.neighbors import NearestNeighbors
    nn_e = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    nn_s = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, ie = nn_e.kneighbors(emb)
    _, isp = nn_s.kneighbors(xy)
    ie = ie[:, 1:]; isp = isp[:, 1:]
    inter = np.zeros(n, dtype=np.float32)
    for i in range(n):
        inter[i] = len(set(ie[i].tolist()) & set(isp[i].tolist()))
    union = 2 * k - inter
    jacc = inter / np.maximum(union, 1.0)
    return float(jacc.mean())


# ───────────────────────────────────────────────────────────────────────────
# 2. Smoothness vs spatial distance
# ───────────────────────────────────────────────────────────────────────────

def spatial_smoothness(emb: np.ndarray, xy: np.ndarray,
                        n_pairs: int = 5000, seed: int = 0) -> float:
    """Spearman correlation between:
        embedding pairwise cosine SIMILARITY  (high → close)
        spatial pairwise distance              (high → far)
    Negative correlation expected: closer spatially → higher cosine.
    We flip the sign so HIGHER = better (more spatially-coherent).
    """
    n = emb.shape[0]
    if n < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    e_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cos = (e_norm[i] * e_norm[j]).sum(axis=1)
    dist = np.linalg.norm(xy[i] - xy[j], axis=1)
    # Spearman without scipy
    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x), dtype=np.float64)
        return ranks
    rc = _rank(cos); rd = _rank(dist)
    rc -= rc.mean(); rd -= rd.mean()
    denom = np.sqrt((rc ** 2).sum() * (rd ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    # Negative correlation expected → flip sign for higher-is-better.
    return float(-(rc * rd).sum() / denom)


# ───────────────────────────────────────────────────────────────────────────
# 3. Boundary preservation — inter/intra domain distance ratio
# ───────────────────────────────────────────────────────────────────────────

def boundary_preservation(emb: np.ndarray, labels: np.ndarray,
                           max_pairs: int = 10000, seed: int = 0) -> float:
    """labels: (n,) integer domain ids (Novae niches, or any clustering).
    Returns mean(inter-domain dist) / mean(intra-domain dist).  > 1 desired.
    """
    n = emb.shape[0]
    if len(np.unique(labels)) < 2 or n < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    d = np.linalg.norm(emb[i] - emb[j], axis=1)
    intra = d[labels[i] == labels[j]]
    inter = d[labels[i] != labels[j]]
    if not intra.size or not inter.size:
        return float("nan")
    return float(inter.mean() / max(intra.mean(), 1e-9))


# ───────────────────────────────────────────────────────────────────────────
# 4. Augmentation consistency — two random ego subgraphs of the same anchor
# ───────────────────────────────────────────────────────────────────────────

def augmentation_consistency(emb_a: np.ndarray, emb_b: np.ndarray,
                              anchor_idx_a: np.ndarray,
                              anchor_idx_b: np.ndarray) -> float:
    """Both forward passes are over different subgraphs but contain shared
    anchors (`anchor_idx_a[i] == anchor_idx_b[i]` means same spot id).
    Returns mean cosine of the shared anchors across the two passes.
    """
    common = np.intersect1d(anchor_idx_a, anchor_idx_b)
    if common.size < 5:
        return float("nan")
    map_a = {int(v): k for k, v in enumerate(anchor_idx_a.tolist())}
    map_b = {int(v): k for k, v in enumerate(anchor_idx_b.tolist())}
    a = emb_a[[map_a[int(x)] for x in common]]
    b = emb_b[[map_b[int(x)] for x in common]]
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float((a * b).sum(axis=1).mean())


# ───────────────────────────────────────────────────────────────────────────
# 5. Effective rank — collapse detector
# ───────────────────────────────────────────────────────────────────────────

def effective_rank(emb: np.ndarray) -> float:
    if emb.shape[0] < 2:
        return float("nan")
    x = emb - emb.mean(0, keepdims=True)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except Exception:
        return float("nan")
    p = s ** 2
    p = p / max(float(p.sum()), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


# ───────────────────────────────────────────────────────────────────────────
# Aggregator used by scripts/eval/stage15_indist.py
# ───────────────────────────────────────────────────────────────────────────

def stage15_metrics_for_sample(emb: np.ndarray, xy: np.ndarray, *,
                                  k: int = 10,
                                  domain_labels: Optional[np.ndarray] = None,
                                  prefix: str = "stage15") -> dict:
    """All metrics that need only (embedding, coords [, labels])."""
    out = {
        f"{prefix}/spatial_knn_overlap_k{k}": spatial_knn_overlap(emb, xy, k=k),
        f"{prefix}/spatial_smoothness":        spatial_smoothness(emb, xy),
        f"{prefix}/effective_rank":            effective_rank(emb),
    }
    if domain_labels is not None:
        out[f"{prefix}/boundary_preservation"] = boundary_preservation(
            emb, domain_labels)
    return out
