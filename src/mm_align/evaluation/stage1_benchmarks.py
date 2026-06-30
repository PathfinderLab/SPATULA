"""Stage-1 evaluation benchmarks — spot-level + sample-level.

Mirrors histology evaluation:
  histology slide-level   : survival (MIL) / subtype classification (MIL)
  histology patch-level   : gene expression (linear probe) / patch class (linear probe)

  transcriptomics spot-level  : masked-symbol top-K acc / value PCC / spot clustering
  transcriptomics sample-level: organ classification (linear probe over mean-pooled spots)
                                + dataset-source recognition (sanity: low = good)

Inputs are spot embeddings (h_tx from the Stage-1 tx_encoder).  Pooling to
sample-level is mean (could swap to attention/median if needed).

Run-time: one full pass over val/test (encoded once, multiple metrics).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Intrinsic + short downstream probes for Stage-1 ablation selection
# ─────────────────────────────────────────────────────────────────────

def embedding_health_metrics(h_tx: np.ndarray, prefix: str = "intrinsic") -> dict[str, float]:
    """Cheap representation-health metrics for in-training validation.

    These metrics are label-free. They are meant to catch collapse or overly
    low-dimensional embeddings while ablations are still short-running.
    """
    if h_tx.size == 0:
        return {
            f"{prefix}/norm_mean": float("nan"),
            f"{prefix}/norm_std": float("nan"),
            f"{prefix}/effective_rank": float("nan"),
            f"{prefix}/explained_top10": float("nan"),
        }
    x = np.asarray(h_tx, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(x, compute_uv=False)
        p = s ** 2
        p = p / max(float(p.sum()), 1e-12)
        eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        explained_top10 = float(p[:10].sum())
    except Exception:
        eff_rank = float("nan")
        explained_top10 = float("nan")
    return {
        f"{prefix}/norm_mean": float(norm.mean()),
        f"{prefix}/norm_std": float(norm.std()),
        f"{prefix}/effective_rank": eff_rank,
        f"{prefix}/explained_top10": explained_top10,
    }


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small scipy-free average-rank helper for Spearman correlation."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = _rankdata_average(np.asarray(x)[mask])
    ry = _rankdata_average(np.asarray(y)[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)



def _ridge_alpha(features: int, targets: int, mode: str | float) -> float:
    """`auto` uses the HEST/SEAL formula: `100 / (features · targets)`."""
    if isinstance(mode, (int, float)):
        return float(mode)
    if isinstance(mode, str) and mode.lower() == "auto":
        return 100.0 / max(1, features * targets)
    return float(mode)


def _select_high_var_genes(y: np.ndarray, n_targets: int,
                             *, min_nz_frac: float = 0.01) -> np.ndarray:
    var = y.var(axis=0)
    nz_frac = (y != 0).mean(axis=0)
    valid = np.where((var > 1e-8) & (nz_frac > min_nz_frac))[0]
    if valid.size == 0:
        return np.array([], dtype=np.int64)
    order = valid[np.argsort(var[valid])[::-1]]
    return order[:min(int(n_targets), order.size)]


def hvg_linear_probe(h_tx: np.ndarray,
                     hvg: np.ndarray,
                     n_targets: int = 256,
                     train_frac: float = 0.70,
                     max_spots: int = 5000,
                     seed: int = 0,
                     alpha: float = 1.0,
                     prefix: str = "linear_probe/hvg",
                     pca_n: int | None = None,
                     metric_suite: str = "legacy",
                     return_per_gene: bool = False,
                     ) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Frozen-embedding Ridge probe — predicts log1p HVG expression from h_tx.

    The target is the current input HVG matrix after the checkpoint's
    normalization/vocab-clip pipeline. We select high-variance, non-trivial
    genes and fit a small linear regressor from h_tx -> expression.

    Parameters
    ----------
    pca_n        : if set, prepend StandardScaler+PCA(pca_n) to the input
                   features — matches HEST/SEAL preprocessing.
    alpha        : float, or the literal string ``"auto"`` which uses the
                   HEST/SEAL formula  α = 100 / (features · targets).
    metric_suite : ``legacy`` (Pearson/Spearman/R²/RMSE_norm — backwards-compat)
                   or ``spatial_bench`` which ALSO emits SSIM and JSD per gene
                   plus median / Q1 / Q3 across genes (SpatialBenchmarking
                   GenesMetrics convention).
    return_per_gene: when True, returns (summary_dict, per_gene_dict).
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from .gene_imputation_metrics import (
        pearson_per_gene, spearman_per_gene, rmse_zscore_per_gene,
        ssim_per_gene, jsd_per_gene, summarise,
    )

    def _nan_out(n_now: int) -> dict[str, float]:
        out = {
            f"{prefix}/pearson_mean": float("nan"),
            f"{prefix}/spearman_mean": float("nan"),
            f"{prefix}/r2_mean": float("nan"),
            f"{prefix}/rmse_norm": float("nan"),
            f"{prefix}/n_targets": 0.0,
            f"{prefix}/n_spots": float(n_now),
        }
        return out

    x = np.asarray(h_tx, dtype=np.float32)
    y = np.asarray(hvg, dtype=np.float32)
    n = min(x.shape[0], y.shape[0])
    if n < 20 or x.ndim != 2 or y.ndim != 2:
        return (_nan_out(n), {}) if return_per_gene else _nan_out(n)
    x, y = x[:n], y[:n]
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        x, y = x[sel], y[sel]
        n = max_spots

    genes = _select_high_var_genes(y, n_targets)
    if genes.size == 0:
        return (_nan_out(n), {}) if return_per_gene else _nan_out(n)
    y = y[:, genes]

    perm = rng.permutation(n)
    n_train = max(10, min(n - 5, int(round(n * train_frac))))
    tr, te = perm[:n_train], perm[n_train:]
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x[tr])
    x_te = scaler.transform(x[te])
    if pca_n is not None and pca_n > 0 and pca_n < x_tr.shape[1]:
        p = PCA(n_components=int(pca_n), random_state=int(seed))
        x_tr = p.fit_transform(x_tr)
        x_te = p.transform(x_te)
    alpha_eff = _ridge_alpha(x_tr.shape[1], y.shape[1], alpha)
    model = Ridge(alpha=alpha_eff)
    model.fit(x_tr, y[tr])
    pred = model.predict(x_te)
    target = y[te]

    err = pred - target
    mse = (err ** 2).mean(axis=0)
    target_var = target.var(axis=0)
    r2 = 1.0 - mse / np.maximum(target_var, 1e-12)
    rmse_norm = np.sqrt(mse) / np.maximum(np.sqrt(target_var), 1e-12)
    pearson = pearson_per_gene(pred, target)
    spearman = spearman_per_gene(pred, target)

    out = {
        f"{prefix}/pearson_mean": float(np.nanmean(pearson)),
        f"{prefix}/spearman_mean": float(np.nanmean(spearman)),
        f"{prefix}/r2_mean": float(np.nanmean(r2)),
        f"{prefix}/rmse_norm": float(np.nanmean(rmse_norm)),
        f"{prefix}/n_targets": float(len(genes)),
        f"{prefix}/n_spots": float(n),
        f"{prefix}/alpha_used": float(alpha_eff),
    }
    per_gene = {
        "pearson": pearson,
        "spearman": spearman,
        "r2": r2,
        "rmse_norm": rmse_norm,
    }

    # SpatialBenchmarking-style suite: add SSIM + JSD (+ quantile summaries).
    if metric_suite == "spatial_bench":
        ssim = ssim_per_gene(pred, target)
        jsd = jsd_per_gene(pred, target)
        rmse_z = rmse_zscore_per_gene(pred, target)
        per_gene["ssim"] = ssim
        per_gene["jsd"] = jsd
        per_gene["rmse_zscore"] = rmse_z
        out.update(summarise(
            {k: per_gene[k] for k in ("pearson", "spearman", "ssim", "jsd", "rmse_zscore")},
            prefix=f"{prefix}/spatial_bench/",
        ))

    if return_per_gene:
        return out, per_gene
    return out


def hvg_rank_probe(h_tx: np.ndarray,
                   hvg: np.ndarray,
                   n_targets: int = 256,
                   train_frac: float = 0.70,
                   max_spots: int = 5000,
                   seed: int = 0,
                   alpha: float = 1.0,
                   n_bins: int = 8,
                   prefix: str = "linear_probe/hvg_rank") -> dict[str, float]:
    """Frozen-embedding probe for relative expression salience.

    Geneformer-style inputs care more about within-spot relative salience than
    absolute expression scale. This probe fits h_tx -> selected HVG values, but
    scores whether predicted values recover each spot's gene ranking and coarse
    expression bins. It complements MSE/R2, which can penalize useful rank
    representations under global-median normalization.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(h_tx, dtype=np.float32)
    y = np.asarray(hvg, dtype=np.float32)
    n = min(x.shape[0], y.shape[0])
    if n < 20 or x.ndim != 2 or y.ndim != 2:
        return {
            f"{prefix}/spot_rank_spearman": float("nan"),
            f"{prefix}/top10_overlap": float("nan"),
            f"{prefix}/bin_acc": float("nan"),
            f"{prefix}/n_targets": 0.0,
            f"{prefix}/n_spots": float(n),
        }
    x, y = x[:n], y[:n]
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        x, y = x[sel], y[sel]
        n = max_spots

    genes = _select_probe_genes(y, n_targets, min_nz_frac=0.01)
    if len(genes) < 3:
        return {
            f"{prefix}/spot_rank_spearman": float("nan"),
            f"{prefix}/top10_overlap": float("nan"),
            f"{prefix}/bin_acc": float("nan"),
            f"{prefix}/n_targets": float(len(genes)),
            f"{prefix}/n_spots": float(n),
        }
    y = y[:, genes]
    perm = rng.permutation(n)
    n_train = max(10, min(n - 5, int(round(n * train_frac))))
    tr, te = perm[:n_train], perm[n_train:]
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x[tr])
    x_te = scaler.transform(x[te])
    model = Ridge(alpha=alpha)
    model.fit(x_tr, y[tr])
    pred = np.asarray(model.predict(x_te), dtype=np.float32)
    target = y[te].astype(np.float32, copy=False)

    rank_scores = []
    top_overlaps = []
    k = min(10, target.shape[1])
    for p_row, y_row in zip(pred, target):
        rank_scores.append(_spearman(p_row, y_row))
        true_top = set(np.argpartition(y_row, -k)[-k:].tolist())
        pred_top = set(np.argpartition(p_row, -k)[-k:].tolist())
        top_overlaps.append(len(true_top.intersection(pred_top)) / max(k, 1))

    # Coarse bin accuracy over gene-wise train quantiles. This is an
    # expression-scale-light metric similar in spirit to scGPT-style binning.
    bin_acc = float("nan")
    if n_bins and n_bins > 1:
        edges = []
        q = np.linspace(0, 1, n_bins + 1)[1:-1]
        for j in range(y.shape[1]):
            vals = y[tr, j]
            if np.nanstd(vals) < 1e-8:
                edges.append(None)
            else:
                edges.append(np.unique(np.quantile(vals, q)))
        hits, total = 0, 0
        for j, e in enumerate(edges):
            if e is None or len(e) == 0:
                continue
            yt = np.digitize(target[:, j], e)
            yp = np.digitize(pred[:, j], e)
            hits += int((yt == yp).sum())
            total += int(yt.size)
        if total > 0:
            bin_acc = float(hits / total)

    return {
        f"{prefix}/spot_rank_spearman": float(np.nanmean(rank_scores)),
        f"{prefix}/top10_overlap": float(np.nanmean(top_overlaps)),
        f"{prefix}/bin_acc": bin_acc,
        f"{prefix}/n_targets": float(len(genes)),
        f"{prefix}/n_spots": float(n),
    }


def _select_probe_genes(hvg: np.ndarray, n_targets: int, min_nz_frac: float = 0.01) -> np.ndarray:
    y = np.asarray(hvg, dtype=np.float32)
    var = y.var(axis=0)
    nz_frac = (y != 0).mean(axis=0)
    valid = np.where((var > 1e-8) & (nz_frac > min_nz_frac))[0]
    if valid.size == 0:
        return np.array([], dtype=np.int64)
    order = valid[np.argsort(var[valid])[::-1]]
    return order[:min(int(n_targets), order.size)].astype(np.int64)


def expression_manifold_metrics(h_tx: np.ndarray,
                                hvg: np.ndarray,
                                max_spots: int = 2000,
                                k: int = 20,
                                seed: int = 0,
                                prefix: str = "intrinsic/expression") -> dict[str, float]:
    """Label-free zero-shot check: does h_tx preserve expression neighbors?"""
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(h_tx, dtype=np.float32)
    y = np.asarray(hvg, dtype=np.float32)
    n = min(x.shape[0], y.shape[0])
    if n < k + 2:
        return {f"{prefix}/knn_overlap@{k}": float("nan"),
                f"{prefix}/distance_spearman": float("nan"),
                f"{prefix}/n_spots": float(n)}
    x, y = x[:n], y[:n]
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        x, y = x[sel], y[sel]
        n = max_spots
    kk = min(k, n - 1)
    nn_x = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(x)
    nn_y = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(y)
    idx_x = nn_x.kneighbors(x, return_distance=False)[:, 1:]
    idx_y = nn_y.kneighbors(y, return_distance=False)[:, 1:]
    overlap = np.mean([len(set(a).intersection(set(b))) / kk for a, b in zip(idx_x, idx_y)])

    # Pairwise distance correlation on a bounded random subset.
    m = min(n, 800)
    sel = rng.choice(n, m, replace=False) if n > m else np.arange(n)
    dx = pairwise_distances(x[sel], metric="cosine")
    dy = pairwise_distances(y[sel], metric="cosine")
    tri = np.triu_indices_from(dx, k=1)
    return {f"{prefix}/knn_overlap@{k}": float(overlap),
            f"{prefix}/distance_spearman": _spearman(dx[tri], dy[tri]),
            f"{prefix}/n_spots": float(n)}


def source_knn_leakage_metrics(h_tx: np.ndarray,
                               source_labels: np.ndarray,
                               k: int = 20,
                               max_spots: int = 5000,
                               seed: int = 0,
                               prefix: str = "leakage/source_knn") -> dict[str, float]:
    """Zero-shot leakage check: do nearest neighbors come from the same source?"""
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(h_tx, dtype=np.float32)
    src = np.asarray(source_labels)
    n = min(x.shape[0], src.shape[0])
    if n < k + 2 or len(np.unique(src[:n])) < 2:
        return {f"{prefix}/same_rate@{k}": float("nan"),
                f"{prefix}/entropy@{k}": float("nan"),
                f"{prefix}/n_spots": float(n)}
    x, src = x[:n], src[:n]
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        x, src = x[sel], src[sel]
        n = max_spots
    kk = min(k, n - 1)
    idx = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(x).kneighbors(x, return_distance=False)[:, 1:]
    nbr_src = src[idx]
    same = (nbr_src == src[:, None]).mean()
    entropies = []
    for row in nbr_src:
        _, counts = np.unique(row, return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
    return {f"{prefix}/same_rate@{k}": float(same),
            f"{prefix}/entropy@{k}": float(np.mean(entropies)),
            f"{prefix}/n_spots": float(n)}


def gene_embedding_correlation_alignment(hvg: np.ndarray,
                                         gene_embeddings: np.ndarray,
                                         n_genes: int = 512,
                                         top_pairs: int = 1000,
                                         seed: int = 0,
                                         prefix: str = "intrinsic/gene_embedding") -> dict[str, float]:
    """Compare expression gene-gene correlation with learned gene-token cosine.

    If the symbol embedding understands co-expression structure, highly
    correlated gene pairs in the ground-truth HVG matrix should also have high
    cosine similarity in the learned gene embedding table.
    """
    y = np.asarray(hvg, dtype=np.float32)
    e = np.asarray(gene_embeddings, dtype=np.float32)
    if y.ndim != 2 or e.ndim != 2:
        return {f"{prefix}/corr_spearman": float("nan"),
                f"{prefix}/top_pair_overlap": float("nan"),
                f"{prefix}/n_genes": 0.0}
    genes = _select_probe_genes(y, min(n_genes, y.shape[1], e.shape[0]), min_nz_frac=0.01)
    if len(genes) < 3:
        return {f"{prefix}/corr_spearman": float("nan"),
                f"{prefix}/top_pair_overlap": float("nan"),
                f"{prefix}/n_genes": float(len(genes))}
    y = y[:, genes]
    e = e[genes]
    y = y - y.mean(axis=0, keepdims=True)
    y = y / (y.std(axis=0, keepdims=True) + 1e-8)
    expr_corr = (y.T @ y) / max(1, y.shape[0])
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    emb_cos = e @ e.T
    tri = np.triu_indices(len(genes), k=1)
    expr_v = expr_corr[tri]
    emb_v = emb_cos[tri]
    spearman = _spearman(expr_v, emb_v)
    m = min(int(top_pairs), len(expr_v))
    if m <= 0:
        overlap = float("nan")
    else:
        expr_top = set(np.argpartition(expr_v, -m)[-m:].tolist())
        emb_top = set(np.argpartition(emb_v, -m)[-m:].tolist())
        overlap = len(expr_top.intersection(emb_top)) / m
    return {f"{prefix}/corr_spearman": spearman,
            f"{prefix}/top_pair_overlap": float(overlap),
            f"{prefix}/n_genes": float(len(genes)),
            f"{prefix}/n_pairs": float(len(expr_v))}


def gene_embeddings_from_encoder(tx_encoder) -> np.ndarray | None:
    """Return HVG symbol embeddings, excluding PAD/MASK/CLS/UNK special rows."""
    emb = getattr(getattr(getattr(tx_encoder, "gene_emb", None), "symbol", None), "emb", None)
    if emb is None or not hasattr(tx_encoder, "N_SPECIAL"):
        return None
    weight = emb.weight.detach().float().cpu().numpy()
    n_hvg = int(getattr(tx_encoder, "n_hvg", max(0, weight.shape[0] - tx_encoder.N_SPECIAL)))
    return weight[tx_encoder.N_SPECIAL:tx_encoder.N_SPECIAL + n_hvg]


def chunk_view_embeddings_from_encoder(tx_encoder,
                                      hvg: np.ndarray,
                                      n_chunks: int = 4,
                                      chunk_len: int = 256,
                                      dynamic: bool = True,
                                      batch_size: int = 128,
                                      max_spots: int = 5000,
                                      seed: int = 0,
                                      device: str | None = None) -> dict[str, np.ndarray]:
    """Return JEPA diagnostic chunk and inference spot embeddings.

    z_chunk is the first sampled context-like gene chunk. z_spot follows the
    I-JEPA inference contract: after training on partial->target prediction, we
    encode the whole clean non-zero sequence once and use that full-view output
    as the spot embedding. The legacy mean over sampled chunks is also returned
    as z_spot_pooled for diagnostics, but it is no longer the canonical
    spot_state used by evaluation scripts.
    """
    import math
    import torch

    y = np.asarray(hvg, dtype=np.float32)
    n = y.shape[0]
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {"z_chunk": empty, "z_spot": empty, "z_spot_pooled": empty}
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        y = y[sel]
        n = max_spots
    n_chunks = max(1, int(n_chunks))
    chunk_len = max(1, int(chunk_len))
    batch_size = max(1, int(batch_size))
    was_training = bool(tx_encoder.training)
    old_force = getattr(tx_encoder, "_force_mask_in_eval", False)
    tx_encoder.eval()
    if hasattr(tx_encoder, "_force_mask_in_eval"):
        tx_encoder._force_mask_in_eval = False
    if device is None:
        device = next(tx_encoder.parameters()).device
    cur_batch = max(1, int(batch_size))
    try:
        while True:
            chunk0_outs = []
            spot_full_outs = []
            spot_pooled_outs = []
            try:
                with torch.no_grad():
                    for r0 in range(0, n, cur_batch):
                        xb_np = y[r0:r0 + cur_batch]
                        xb = torch.from_numpy(xb_np).to(device)
                        spot_full_outs.append(
                            tx_encoder(novae_latent=None, hvg=xb)["h_tx"].detach().float().cpu().numpy()
                        )
                        chunk_outs = []
                        for c in range(n_chunks):
                            ch = np.zeros_like(xb_np, dtype=np.float32)
                            for i in range(xb_np.shape[0]):
                                idx = np.flatnonzero(xb_np[i] > 0)
                                if idx.size == 0:
                                    continue
                                if dynamic:
                                    take = min(chunk_len, max(1, math.ceil(idx.size / float(n_chunks))))
                                else:
                                    take = min(chunk_len, idx.size)
                                if idx.size <= 1:
                                    keep = idx
                                else:
                                    perm = rng.permutation(idx)
                                    start = c * take
                                    end = min(start + take, idx.size)
                                    keep = perm[start:end]
                                    if keep.size == 0:
                                        keep = rng.choice(idx, size=min(take, idx.size), replace=False)
                                ch[i, keep] = xb_np[i, keep]
                            xt = torch.from_numpy(ch).to(device)
                            chunk_outs.append(tx_encoder(novae_latent=None, hvg=xt)["h_tx"].detach())
                        z_stack = torch.stack(chunk_outs, dim=0)
                        chunk0_outs.append(z_stack[0].float().cpu().numpy())
                        spot_pooled_outs.append(z_stack.mean(dim=0).float().cpu().numpy())
                return {
                    "z_chunk": np.concatenate(chunk0_outs, axis=0),
                    "z_spot": np.concatenate(spot_full_outs, axis=0),
                    "z_spot_pooled": np.concatenate(spot_pooled_outs, axis=0),
                }
            except torch.cuda.OutOfMemoryError:
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                if cur_batch <= 1:
                    raise
                cur_batch = max(1, cur_batch // 2)
    finally:
        if was_training:
            tx_encoder.train()
        if hasattr(tx_encoder, "_force_mask_in_eval"):
            tx_encoder._force_mask_in_eval = old_force


def chunk_spot_embeddings_from_encoder(tx_encoder,
                                       hvg: np.ndarray,
                                       n_chunks: int = 4,
                                       chunk_len: int = 256,
                                       batch_size: int = 128,
                                       max_spots: int = 5000,
                                       seed: int = 0,
                                       device: str | None = None) -> np.ndarray:
    """Back-compatible wrapper returning pooled z_spot only."""
    return chunk_view_embeddings_from_encoder(
        tx_encoder, hvg,
        n_chunks=n_chunks,
        chunk_len=chunk_len,
        dynamic=True,
        batch_size=batch_size,
        max_spots=max_spots,
        seed=seed,
        device=device,
    )["z_spot"]


def masked_hvg_linear_probe_from_encoder(tx_encoder,
                                         hvg: np.ndarray,
                                         n_targets: int = 256,
                                         mask_ratio: float = 1.0,
                                         batch_size: int = 256,
                                         max_spots: int = 5000,
                                         seed: int = 0,
                                         device: str | None = None,
                                         prefix: str = "linear_probe/masked_hvg") -> dict[str, float]:
    """Encode inputs with selected target genes masked, then probe h_tx -> targets."""
    import torch

    y = np.asarray(hvg, dtype=np.float32)
    n = y.shape[0]
    if n < 20:
        return {f"{prefix}/pearson_mean": float("nan"),
                f"{prefix}/spearman_mean": float("nan"),
                f"{prefix}/r2_mean": float("nan"),
                f"{prefix}/rmse_norm": float("nan"),
                f"{prefix}/n_targets": 0.0,
                f"{prefix}/n_spots": float(n)}
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        y = y[sel]
        n = max_spots
    genes = _select_probe_genes(y, n_targets, min_nz_frac=0.01)
    if len(genes) == 0:
        return {f"{prefix}/pearson_mean": float("nan"),
                f"{prefix}/spearman_mean": float("nan"),
                f"{prefix}/r2_mean": float("nan"),
                f"{prefix}/rmse_norm": float("nan"),
                f"{prefix}/n_targets": 0.0,
                f"{prefix}/n_spots": float(n)}
    x_masked = y.copy()
    if mask_ratio >= 1.0:
        x_masked[:, genes] = 0.0
    else:
        mask = rng.random((n, len(genes))) < mask_ratio
        x_masked[:, genes] = np.where(mask, 0.0, x_masked[:, genes])
    was_training = bool(tx_encoder.training)
    tx_encoder.eval()
    if device is None:
        device = next(tx_encoder.parameters()).device
    outs = []
    with torch.no_grad():
        for r0 in range(0, n, batch_size):
            xb = torch.from_numpy(x_masked[r0:r0 + batch_size]).to(device)
            outs.append(tx_encoder(novae_latent=None, hvg=xb)["h_tx"].detach().cpu().numpy())
    if was_training:
        tx_encoder.train()
    emb = np.concatenate(outs, axis=0)
    return hvg_linear_probe(emb, y[:, genes], n_targets=len(genes), max_spots=max_spots,
                            seed=seed, prefix=prefix)


def gene_held_out_probe(tx_encoder,
                         hvg: np.ndarray,
                         n_targets: int = 256,
                         gene_folds: int = 5,
                         batch_size: int = 256,
                         max_spots: int = 5000,
                         seed: int = 0,
                         device: str | None = None,
                         pca_n: int | None = None,
                         alpha: float | str = "auto",
                         prefix: str = "linear_probe/gene_held_out",
                         metric_suite: str = "spatial_bench",
                         return_per_gene: bool = False,
                         ) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """SpatialBenchmarking-style **gene-fold** CV imputation probe.

    Splits the selected target-gene set into `gene_folds` folds.  For each
    fold:
        1. Mask THAT fold's genes to 0 in the input HVG.
        2. Re-encode with the frozen tx_encoder.
        3. Train Ridge( h_tx_masked_train -> heldout_genes ) on a SPOT-split
           train pool, predict on test spots.
        4. Record per-gene Pearson / SSIM / RMSE_z / JSD on the test spots
           for ONLY the held-out genes of this fold.
    Aggregating across folds yields a per-gene metric for every selected
    gene — directly comparable with SpatialBenchmarking GenesMetrics.

    The single-fold variant (`gene_folds=1`) is roughly equivalent to
    `masked_hvg_linear_probe_from_encoder` but uses the same metric suite.
    """
    import torch
    from .gene_imputation_metrics import (
        pearson_per_gene, spearman_per_gene, ssim_per_gene,
        jsd_per_gene, rmse_zscore_per_gene, summarise,
    )

    y_full = np.asarray(hvg, dtype=np.float32)
    n = y_full.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_spots:
        sel = rng.choice(n, max_spots, replace=False)
        y_full = y_full[sel]
        n = max_spots

    target_genes = _select_probe_genes(y_full, n_targets, min_nz_frac=0.01)
    if len(target_genes) < gene_folds * 2:
        # fall back: too few candidates for a meaningful gene CV.
        nan = {f"{prefix}/pearson_mean": float("nan"),
                f"{prefix}/n_targets": float(len(target_genes)),
                f"{prefix}/n_spots": float(n),
                f"{prefix}/gene_folds": float(gene_folds)}
        return (nan, {}) if return_per_gene else nan

    perm = rng.permutation(len(target_genes))
    gene_perm = target_genes[perm]
    folds = np.array_split(gene_perm, gene_folds)
    if device is None:
        device = next(tx_encoder.parameters()).device

    per_gene_pred = np.full((y_full.shape[0], len(target_genes)), np.nan, dtype=np.float32)
    per_gene_target = np.full((y_full.shape[0], len(target_genes)), np.nan, dtype=np.float32)
    gene_to_col = {int(g): k for k, g in enumerate(target_genes)}

    was_training = bool(tx_encoder.training)
    tx_encoder.eval()
    try:
        for fold_idx, held in enumerate(folds):
            x_masked = y_full.copy()
            x_masked[:, held] = 0.0
            # Encode every spot with held genes masked out.
            outs = []
            with torch.no_grad():
                for r0 in range(0, n, batch_size):
                    xb = torch.from_numpy(x_masked[r0:r0 + batch_size]).to(device)
                    outs.append(tx_encoder(novae_latent=None, hvg=xb)["h_tx"].detach().cpu().numpy())
            emb = np.concatenate(outs, axis=0)
            # Spot-split Ridge probe — predict only the held-out genes.
            from sklearn.decomposition import PCA
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            spot_perm = rng.permutation(n)
            n_tr = max(10, int(round(n * 0.7)))
            tr, te = spot_perm[:n_tr], spot_perm[n_tr:]
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(emb[tr])
            x_te = scaler.transform(emb[te])
            if pca_n is not None and pca_n > 0 and pca_n < x_tr.shape[1]:
                p = PCA(n_components=int(pca_n), random_state=int(seed) + fold_idx)
                x_tr = p.fit_transform(x_tr)
                x_te = p.transform(x_te)
            y_held = y_full[:, held]
            alpha_eff = _ridge_alpha(x_tr.shape[1], y_held.shape[1], alpha)
            model = Ridge(alpha=alpha_eff)
            model.fit(x_tr, y_held[tr])
            pred = model.predict(x_te)
            # Fill predictions / targets for these held genes into the global table.
            for j, g in enumerate(held):
                col = gene_to_col[int(g)]
                per_gene_pred[te, col] = pred[:, j]
                per_gene_target[te, col] = y_held[te, j]
    finally:
        if was_training:
            tx_encoder.train()

    # Aggregate per-gene across all eval spots (NaN rows where the spot was
    # never used as a test row for that gene — np.nanmean handles it).
    valid = np.isfinite(per_gene_pred) & np.isfinite(per_gene_target)
    pred = np.where(valid, per_gene_pred, 0.0)
    target = np.where(valid, per_gene_target, 0.0)
    # Compute per-gene metrics on the valid subset (each column has its own
    # row count — handled by nanmean inside our helpers).
    pearson = pearson_per_gene(pred, target)
    spearman = spearman_per_gene(pred, target)
    rmse_z = rmse_zscore_per_gene(pred, target)
    ssim = ssim_per_gene(pred, target)
    jsd = jsd_per_gene(pred, target)
    per_gene = {"pearson": pearson, "spearman": spearman,
                  "ssim": ssim, "jsd": jsd, "rmse_zscore": rmse_z}
    out = {
        f"{prefix}/pearson_mean": float(np.nanmean(pearson)),
        f"{prefix}/spearman_mean": float(np.nanmean(spearman)),
        f"{prefix}/ssim_mean": float(np.nanmean(ssim)),
        f"{prefix}/rmse_zscore_mean": float(np.nanmean(rmse_z)),
        f"{prefix}/jsd_mean": float(np.nanmean(jsd)),
        f"{prefix}/n_targets": float(len(target_genes)),
        f"{prefix}/n_spots": float(n),
        f"{prefix}/gene_folds": float(gene_folds),
    }
    if metric_suite == "spatial_bench":
        out.update(summarise(per_gene, prefix=f"{prefix}/"))
    if return_per_gene:
        return out, per_gene
    return out


# ─────────────────────────────────────────────────────────────────────
# Spot-level
# ─────────────────────────────────────────────────────────────────────

def spot_clustering_metrics(h_tx: np.ndarray, organ_labels: np.ndarray,
                             n_clusters: int | None = None) -> dict[str, float]:
    """KMeans on spot embeddings → cluster vs organ labels (ARI / NMI / Silhouette).
    organ_labels: per-spot string labels (e.g. ['Breast', 'Breast', 'Lung', ...]).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    if n_clusters is None:
        n_clusters = len(set(organ_labels))
    if h_tx.shape[0] < n_clusters + 1:
        return {"spot/cluster/silhouette": float("nan"),
                "spot/cluster/ari": float("nan"),
                "spot/cluster/nmi": float("nan")}
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(h_tx)
    out = {}
    try:    out["spot/cluster/silhouette"] = float(silhouette_score(h_tx, pred))
    except Exception: out["spot/cluster/silhouette"] = float("nan")
    out["spot/cluster/ari"] = float(adjusted_rand_score(organ_labels, pred))
    out["spot/cluster/nmi"] = float(normalized_mutual_info_score(organ_labels, pred))
    return out


def spot_linear_probe(h_tx: np.ndarray, labels: np.ndarray,
                      label_prefix: str = "organ",
                      n_folds: int = 5, max_train_per_fold: int = 20000) -> dict[str, float]:
    """5-fold cross-validation logistic regression on spot embeddings."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return {f"spot/probe_{label_prefix}/acc": float("nan"),
                f"spot/probe_{label_prefix}/f1_macro": float("nan")}
    # cap to keep CV cheap
    if len(y) > 50_000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(y), 50_000, replace=False)
        h_tx, y = h_tx[sel], y[sel]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    accs, f1s = [], []
    for tr, te in skf.split(h_tx, y):
        # subsample train to bound cost
        if len(tr) > max_train_per_fold:
            tr = np.random.default_rng(0).choice(tr, max_train_per_fold, replace=False)
        clf = LogisticRegression(max_iter=200, n_jobs=-1, solver="lbfgs",
                                 multi_class="auto", C=1.0)
        clf.fit(h_tx[tr], y[tr])
        pred = clf.predict(h_tx[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {
        f"spot/probe_{label_prefix}/acc": float(np.mean(accs)),
        f"spot/probe_{label_prefix}/f1_macro": float(np.mean(f1s)),
        f"spot/probe_{label_prefix}/n_classes": float(n_classes),
    }


# ─────────────────────────────────────────────────────────────────────
# Sample-level (mean-pool spots → 1 vector per sample)
# ─────────────────────────────────────────────────────────────────────

def pool_to_sample(h_tx: np.ndarray, sample_idx: np.ndarray, method: str = "mean"):
    """Aggregate spots into per-sample embeddings.
    Returns (sample_emb, sample_id_array_of_unique_ids)."""
    uniq = np.unique(sample_idx)
    out = np.zeros((len(uniq), h_tx.shape[1]), dtype=np.float32)
    for i, sid in enumerate(uniq):
        sel = sample_idx == sid
        if method == "mean":
            out[i] = h_tx[sel].mean(0)
        elif method == "median":
            out[i] = np.median(h_tx[sel], axis=0)
        else:
            raise ValueError(method)
    return out, uniq


def sample_clustering_metrics(sample_emb: np.ndarray, organ_labels: np.ndarray,
                               n_clusters: int | None = None) -> dict[str, float]:
    """Cluster sample-level embeddings, compare to organ labels."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    if n_clusters is None:
        n_clusters = len(set(organ_labels))
    if sample_emb.shape[0] < n_clusters + 1:
        return {"sample/cluster/silhouette": float("nan"),
                "sample/cluster/ari": float("nan"),
                "sample/cluster/nmi": float("nan")}
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(sample_emb)
    out = {}
    try:    out["sample/cluster/silhouette"] = float(silhouette_score(sample_emb, pred))
    except Exception: out["sample/cluster/silhouette"] = float("nan")
    out["sample/cluster/ari"] = float(adjusted_rand_score(organ_labels, pred))
    out["sample/cluster/nmi"] = float(normalized_mutual_info_score(organ_labels, pred))
    return out


def sample_linear_probe(sample_emb: np.ndarray, labels: np.ndarray,
                        label_prefix: str = "organ", n_folds: int = 5) -> dict[str, float]:
    """LogisticRegression cross-validation at the SAMPLE level (one vector
    per sample, organ label).  This is the transcriptomics analogue of
    slide-level subtype classification in histology."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    # need ≥ n_folds samples per class for stratified KF
    counts = np.bincount(y)
    if n_classes < 2 or counts.min() < n_folds:
        return {f"sample/probe_{label_prefix}/acc": float("nan"),
                f"sample/probe_{label_prefix}/f1_macro": float("nan"),
                f"sample/probe_{label_prefix}/n_classes": float(n_classes),
                f"sample/probe_{label_prefix}/skipped": float(1.0)}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    accs, f1s = [], []
    for tr, te in skf.split(sample_emb, y):
        clf = LogisticRegression(max_iter=500, n_jobs=-1, solver="lbfgs",
                                 multi_class="auto", C=1.0)
        clf.fit(sample_emb[tr], y[tr])
        pred = clf.predict(sample_emb[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {
        f"sample/probe_{label_prefix}/acc": float(np.mean(accs)),
        f"sample/probe_{label_prefix}/f1_macro": float(np.mean(f1s)),
        f"sample/probe_{label_prefix}/n_classes": float(n_classes),
    }


# ─────────────────────────────────────────────────────────────────────
# Combined entry point
# ─────────────────────────────────────────────────────────────────────

def run_stage1_benchmarks(h_tx: np.ndarray,
                           organ_per_spot: np.ndarray,
                           sample_idx_per_spot: np.ndarray,
                           source_per_sample: Optional[dict[int, str]] = None,
                           organ_per_sample: Optional[dict[int, str]] = None,
                           max_spots_for_cluster: int = 20000) -> dict[str, float]:
    """All-in-one wrapper.

    Parameters
    ----------
    h_tx                 : (N_spot, D) embeddings
    organ_per_spot       : (N_spot,)   organ label string per spot
    sample_idx_per_spot  : (N_spot,)   integer sample index per spot
    source_per_sample    : optional {sample_idx: source_str}  ('hest'/'st1k'/'spc')
    organ_per_sample     : optional {sample_idx: organ_str}   used for sample-level
    """
    out = {}

    # ── Spot-level ─────────────────────────────────────────────────
    # Subsample for clustering (KMeans + silhouette on millions is slow)
    if h_tx.shape[0] > max_spots_for_cluster:
        rng = np.random.default_rng(0)
        sel = rng.choice(h_tx.shape[0], max_spots_for_cluster, replace=False)
        h_sub, org_sub = h_tx[sel], organ_per_spot[sel]
    else:
        h_sub, org_sub = h_tx, organ_per_spot
    out.update(spot_clustering_metrics(h_sub, org_sub))
    out.update(spot_linear_probe(h_tx, organ_per_spot, label_prefix="organ"))

    # ── Sample-level ────────────────────────────────────────────────
    sample_emb, uniq = pool_to_sample(h_tx, sample_idx_per_spot, method="mean")

    if organ_per_sample is not None:
        organ_labels = np.array([organ_per_sample.get(int(s), "Unknown") for s in uniq])
        out.update(sample_clustering_metrics(sample_emb, organ_labels))
        out.update(sample_linear_probe(sample_emb, organ_labels, label_prefix="organ"))

    if source_per_sample is not None:
        src_labels = np.array([source_per_sample.get(int(s), "Unknown") for s in uniq])
        # Sample-level dataset-source recognition: LOWER is better.
        # If sample embeddings are dataset-agnostic, a classifier shouldn't
        # be able to recover the source.
        out.update(sample_linear_probe(sample_emb, src_labels, label_prefix="source"))

    out["sample/n_samples"] = float(len(uniq))
    out["spot/n_spots"] = float(h_tx.shape[0])
    return out
