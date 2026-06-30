"""Spatial metrics for DLPFC / SVG / spot-encoder downstream evaluation.

Pure numpy / scipy / scanpy / sklearn — NO squidpy dependency.  All
implementations follow the SDMBench / SVG_Benchmarking definitions so reported
numbers stay comparable with published benchmarks.

Implemented:
  * cluster_assign        — KMeans | GMM ("mclust"-equivalent) | Leiden
  * cluster_label_metrics — ARI, NMI, HOM, COM (homogeneity / completeness)
  * silhouette_spatial    — silhouette using PRECOMPUTED spatial distances
  * chaos_score           — 1-NN within-cluster spatial distance sum (SDMBench)
  * pas_score             — fraction of spots whose k-NN majority disagrees
  * morans_i              — global Moran's I given KNN adjacency
  * gearys_c              — global Geary's C given KNN adjacency
  * marker_spatial_autocorr — Moran's I / Geary's C aggregated over top-K
                              marker genes per cluster
  * top_marker_genes       — per-cluster top differentially expressed genes
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, completeness_score, homogeneity_score,
    normalized_mutual_info_score, silhouette_score,
)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_assign(emb: np.ndarray, n_clusters: int, *, method: str = "kmeans",
                   seed: int = 0, n_repeats: int = 1) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run a clustering method on an embedding matrix.

    Returns (best_pred, all_preds).  best_pred = run with highest within-cluster
    cohesion (lowest inertia for KMeans / highest BIC for GMM).  all_preds is a
    list of every repeat's labels (length = n_repeats) — useful for stability
    reporting (mean / std across runs, à la SDMBench).
    """
    method = method.lower()
    all_preds = []
    best_pred = None
    best_score = None
    for r in range(int(n_repeats)):
        s = seed + r
        if method == "kmeans":
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=s).fit(emb)
            pred = km.labels_
            score = -km.inertia_                                  # higher = better
        elif method in ("gmm", "mclust"):
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(
                n_components=n_clusters, covariance_type="tied",
                random_state=s, max_iter=200, reg_covar=1e-4,
            ).fit(emb)
            pred = gm.predict(emb)
            score = float(gm.bic(emb)) * -1.0                     # higher = better
        elif method == "leiden":
            pred = _leiden_predict(emb, n_clusters_hint=n_clusters, seed=s)
            score = float(-_within_cluster_dispersion(emb, pred))
        else:
            raise ValueError(f"unknown clustering method: {method}")
        all_preds.append(np.asarray(pred, dtype=int))
        if best_score is None or score > best_score:
            best_score = score
            best_pred = pred
    return np.asarray(best_pred, dtype=int), all_preds


def _within_cluster_dispersion(emb: np.ndarray, pred: np.ndarray) -> float:
    out = 0.0
    for u in np.unique(pred):
        m = pred == u
        if m.sum() <= 1:
            continue
        c = emb[m].mean(0, keepdims=True)
        out += float(((emb[m] - c) ** 2).sum())
    return out


def _leiden_predict(emb: np.ndarray, *, n_clusters_hint: int, seed: int) -> np.ndarray:
    """Try to land near n_clusters_hint by tuning Leiden resolution lightly.

    We do a small bisection on resolution (5 attempts) to find the closest #
    clusters; this matches SDMBench's "Leiden tuned to target k" practice.
    """
    import scanpy as sc
    from anndata import AnnData
    ad = AnnData(X=emb.astype(np.float32))
    sc.pp.neighbors(ad, n_neighbors=min(30, max(2, emb.shape[0] - 1)),
                     use_rep="X", metric="euclidean", random_state=seed)
    lo, hi = 0.1, 2.5
    best_pred = None
    best_diff = None
    for _ in range(5):
        mid = (lo + hi) / 2.0
        sc.tl.leiden(ad, resolution=mid, random_state=seed,
                     flavor="igraph", n_iterations=2, directed=False)
        pred = ad.obs["leiden"].astype(int).to_numpy()
        k_now = int(len(np.unique(pred)))
        diff = abs(k_now - n_clusters_hint)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_pred = pred
        if k_now < n_clusters_hint:
            lo = mid
        else:
            hi = mid
    return best_pred


# ---------------------------------------------------------------------------
# Label-aware cluster metrics (extends ARI/NMI to HOM/COM)
# ---------------------------------------------------------------------------


def cluster_label_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(gt, pred)),
        "nmi": float(normalized_mutual_info_score(gt, pred)),
        "homogeneity": float(homogeneity_score(gt, pred)),
        "completeness": float(completeness_score(gt, pred)),
    }


# ---------------------------------------------------------------------------
# Spatial silhouette / continuity scores (SDMBench style)
# ---------------------------------------------------------------------------


def silhouette_spatial(coords: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette computed on spatial (xy) distances — labels typically come
    from a method's clustering.  Lower = clusters disrespect geography.

    Quote SDMBench.py:147-149: silhouette_score with metric='precomputed'.
    """
    if len(np.unique(labels)) < 2 or len(coords) < 3:
        return float("nan")
    try:
        from sklearn.metrics import pairwise_distances
        d = pairwise_distances(coords, metric="euclidean")
        return float(silhouette_score(d, labels, metric="precomputed"))
    except Exception:
        return float("nan")


def chaos_score(labels: np.ndarray, coords: np.ndarray) -> float:
    """CHAOS — Spatial continuity.  For each spot, distance to its nearest
    neighbour *that shares its cluster label* — summed and normalised by N.
    Lower (= geometrically tight clusters) is better.

    Implementation mirrors SDMBench.py:63-81 (Shao et al. 2023).
    """
    from sklearn.neighbors import NearestNeighbors
    if len(np.unique(labels)) < 2 or len(coords) < 4:
        return float("nan")
    total = 0.0
    n = 0
    for u in np.unique(labels):
        m = labels == u
        if m.sum() < 2:
            continue
        xy = coords[m]
        nn = NearestNeighbors(n_neighbors=2).fit(xy)
        d, _ = nn.kneighbors(xy)
        total += float(d[:, 1].sum())          # nearest within-cluster neighbour
        n += int(m.sum())
    return total / max(1, n)


def pas_score(labels: np.ndarray, coords: np.ndarray, k: int = 10) -> float:
    """PAS — Percent Abnormal Spots.  Fraction of spots whose k spatial
    neighbours don't share the same label as themselves (k=10 in SDMBench).
    Lower is better.

    SDMBench.py:108-114 / Shao et al. 2023.
    """
    from sklearn.neighbors import NearestNeighbors
    if len(coords) < k + 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                                # drop self
    neigh_labels = labels[idx]
    # A spot is "abnormal" if NONE of its k neighbours share its label.
    matches = (neigh_labels == labels[:, None]).any(axis=1)
    return float((~matches).mean())


# ---------------------------------------------------------------------------
# Spatial autocorrelation (pure numpy — no squidpy)
# ---------------------------------------------------------------------------


def _knn_weight_matrix(coords: np.ndarray, k: int = 6,
                        *, row_normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (W_dense, knn_idx).  W is row-normalised KNN adjacency
    (excluding self).  knn_idx is (N, k) for reuse."""
    from sklearn.neighbors import NearestNeighbors
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                                # drop self
    W = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    W[rows, cols] = 1.0
    if row_normalize:
        rsum = W.sum(axis=1, keepdims=True)
        rsum[rsum == 0] = 1.0
        W = W / rsum
    return W, idx


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    """Global Moran's I.  values: (N,), W: (N, N) row-normalised."""
    v = np.asarray(values, dtype=np.float64)
    v = v - v.mean()
    n = len(v)
    s0 = W.sum()
    if s0 == 0:
        return float("nan")
    num = float((W * (v[:, None] * v[None, :])).sum())
    denom = float((v * v).sum())
    if denom == 0:
        return float("nan")
    return (n / s0) * (num / denom)


def gearys_c(values: np.ndarray, W: np.ndarray) -> float:
    """Global Geary's C.  Values in [0, 2]; lower = more positive autocorr."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    s0 = W.sum()
    if s0 == 0:
        return float("nan")
    diff = v[:, None] - v[None, :]
    num = float((W * (diff * diff)).sum())
    vc = v - v.mean()
    denom = float((vc * vc).sum())
    if denom == 0:
        return float("nan")
    return ((n - 1) / (2 * s0)) * (num / denom)


def morans_i_per_gene(hvg: np.ndarray, coords: np.ndarray, *,
                       k: int = 6, gene_subset: np.ndarray | None = None) -> np.ndarray:
    """Vectorised Moran's I for every column of `hvg` over a KNN weight
    matrix built from `coords`.  Returns (n_genes,) array (NaN if a gene's
    variance is 0)."""
    W, _ = _knn_weight_matrix(coords, k=k, row_normalize=True)
    if gene_subset is not None:
        hvg = hvg[:, gene_subset]
    centered = hvg - hvg.mean(axis=0, keepdims=True)
    denom = (centered ** 2).sum(axis=0)
    s0 = W.sum()
    n = hvg.shape[0]
    # numerator per gene: sum_{i,j} W_ij * (x_i - x_bar) * (x_j - x_bar)
    Wx = W @ centered                                   # (N, G)
    num = (centered * Wx).sum(axis=0)
    out = np.full(centered.shape[1], np.nan, dtype=np.float64)
    nz = denom > 0
    out[nz] = (n / s0) * (num[nz] / denom[nz])
    return out


def gearys_c_per_gene(hvg: np.ndarray, coords: np.ndarray, *,
                      k: int = 6, gene_subset: np.ndarray | None = None) -> np.ndarray:
    W, idx = _knn_weight_matrix(coords, k=k, row_normalize=True)
    if gene_subset is not None:
        hvg = hvg[:, gene_subset]
    n = hvg.shape[0]
    s0 = W.sum()
    centered = hvg - hvg.mean(axis=0, keepdims=True)
    denom = (centered ** 2).sum(axis=0)
    # numerator = sum_ij W_ij * (x_i - x_j)^2
    # = sum_i ( sum_j W_ij * (x_i^2 - 2*x_i*x_j + x_j^2) )
    # Vectorised via the per-neighbour expansion:
    num = np.zeros(hvg.shape[1], dtype=np.float64)
    for g in range(hvg.shape[1]):
        x = hvg[:, g]
        diffs = x[:, None] - x[idx]                    # (N, k)
        # W is row-normalised → each row weight = 1/k; collapse
        num[g] = float(((diffs * diffs).sum(axis=1) * (1.0 / idx.shape[1])).sum())
    out = np.full(hvg.shape[1], np.nan, dtype=np.float64)
    nz = denom > 0
    out[nz] = ((n - 1) / (2 * s0)) * (num[nz] / denom[nz])
    return out


# ---------------------------------------------------------------------------
# Marker-gene aware spatial autocorr (SDMBench.py:170-186)
# ---------------------------------------------------------------------------


def top_marker_genes(hvg: np.ndarray, labels: np.ndarray, gene_names: list[str],
                      *, top_per_cluster: int = 5) -> dict[int, list[str]]:
    """Per-cluster top markers — Welch t-statistic of (in-cluster vs rest).

    Returns dict {cluster_label: [gene_name, ...]} with up to top_per_cluster
    genes each.  Filters out genes with zero variance.
    """
    out: dict[int, list[str]] = {}
    eps = 1e-8
    for u in np.unique(labels):
        m = labels == u
        if m.sum() < 3 or (~m).sum() < 3:
            out[int(u)] = []
            continue
        a = hvg[m]
        b = hvg[~m]
        ma, va = a.mean(0), a.var(0) + eps
        mb, vb = b.mean(0), b.var(0) + eps
        t = (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0] + eps)
        order = np.argsort(-t)
        top_idx = [int(i) for i in order[:top_per_cluster] if t[i] > 0]
        out[int(u)] = [gene_names[i] for i in top_idx]
    return out


# ---------------------------------------------------------------------------
# Novae-style domain-continuity metrics (eval.py:12-170 in references/novae)
# ---------------------------------------------------------------------------


def fide_score(labels: np.ndarray, coords: np.ndarray, *, k: int = 6) -> float:
    """F1-style intra-domain edge fraction (novae's FIDE).

    For each spot we compute its k spatial neighbours.  An edge (i, j) is
    "intra-domain" iff labels[i] == labels[j].  FIDE = mean per-class recall
    of intra-domain edges, averaged across all clusters (so a single
    spatially-coherent cluster can't dominate).

    Higher = clusters are tight in space.  Comparable across samples
    regardless of cluster count.  Reference: novae/monitor/eval.py:34-62.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    if n < k + 2 or len(np.unique(labels)) < 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                                     # drop self
    neigh_labels = labels[idx]
    same = (neigh_labels == labels[:, None])             # (n, k) bool
    per_class = []
    for u in np.unique(labels):
        m = labels == u
        if m.sum() < 2:
            continue
        per_class.append(float(same[m].mean()))
    return float(np.mean(per_class)) if per_class else float("nan")


def slide_jsd(labels: np.ndarray, slide_ids: np.ndarray) -> float:
    """Jensen-Shannon divergence of cluster distribution across slides.

    Tests whether each cluster's slide composition is uniform — high JSD
    means a cluster lives mostly in one slide (batch effect / source leakage).
    Returns a single scalar averaged across clusters.  Lower = better mixing.
    Reference: novae/monitor/eval.py:65-109.
    """
    from scipy.stats import entropy
    labels = np.asarray(labels); slide_ids = np.asarray(slide_ids)
    if labels.size != slide_ids.size or labels.size < 2:
        return float("nan")
    unique_clusters = np.unique(labels)
    unique_slides = np.unique(slide_ids)
    if len(unique_slides) < 2:
        return float("nan")
    # Global slide proportions = reference distribution P.
    P = np.array([(slide_ids == s).mean() for s in unique_slides])
    P = P / max(P.sum(), 1e-12)
    jsd_per_cluster = []
    for c in unique_clusters:
        m = labels == c
        if m.sum() < 2:
            continue
        Q = np.array([(slide_ids[m] == s).mean() for s in unique_slides])
        Q = Q / max(Q.sum(), 1e-12)
        M = 0.5 * (P + Q)
        jsd_per_cluster.append(0.5 * entropy(P, M) + 0.5 * entropy(Q, M))
    return float(np.mean(jsd_per_cluster)) if jsd_per_cluster else float("nan")


def normalized_entropy(labels: np.ndarray) -> float:
    """Mean normalised entropy of cluster sizes (novae heuristic component).

    Returns entropy / log2(K) ∈ [0, 1].  1 = balanced clusters; 0 = single
    cluster dominates.  Use as a sanity bound on cluster utilisation.
    """
    labels = np.asarray(labels)
    K = len(np.unique(labels))
    if K < 2:
        return float("nan")
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log2(np.clip(p, 1e-12, 1)))
    return float(H / np.log2(K))


def novae_heuristic(labels: np.ndarray, coords: np.ndarray, *, k: int = 6) -> dict:
    """Bundle FIDE + entropy + heuristic.  Single import handle for callers."""
    fide = fide_score(labels, coords, k=k)
    ent = normalized_entropy(labels)
    if np.isfinite(fide) and np.isfinite(ent):
        K = len(np.unique(labels))
        heur = fide * ent / max(1.0, np.log2(K))
    else:
        heur = float("nan")
    return {"fide": fide, "entropy_norm": ent, "heuristic": heur}


def marker_spatial_autocorr(hvg: np.ndarray, coords: np.ndarray, labels: np.ndarray,
                            gene_names: list[str], *, top_per_cluster: int = 5,
                            k: int = 6) -> dict[str, float]:
    """Aggregate Moran's I / Geary's C across each cluster's top markers.

    Reports median across all markers (matches SDMBench.py:184-186).
    """
    markers = top_marker_genes(hvg, labels, gene_names, top_per_cluster=top_per_cluster)
    flat: list[str] = []
    for v in markers.values():
        flat.extend(v)
    flat = list(dict.fromkeys(flat))                  # dedup, keep order
    if not flat:
        return {"marker_morans_i_median": float("nan"),
                "marker_gearys_c_median": float("nan"),
                "n_markers": 0}
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    idx = np.array([gene_to_idx[g] for g in flat if g in gene_to_idx], dtype=int)
    if idx.size == 0:
        return {"marker_morans_i_median": float("nan"),
                "marker_gearys_c_median": float("nan"),
                "n_markers": 0}
    mi = morans_i_per_gene(hvg, coords, k=k, gene_subset=idx)
    gc = gearys_c_per_gene(hvg, coords, k=k, gene_subset=idx)
    return {"marker_morans_i_median": float(np.nanmedian(mi)),
            "marker_gearys_c_median": float(np.nanmedian(gc)),
            "n_markers": int(idx.size)}
