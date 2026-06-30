"""Shared metric helpers for gene-imputation / gene-mapping style evaluations.

All metrics treat each gene as a 1-D signal across spots, mirroring the
canonical SpatialBenchmarking protocol (GenesMetrics.py) — PCC, SSIM-1D,
RMSE (z-score scaled), JSD (scale_plus + symmetric KL).  We add Spearman so
legacy reports keep working and a rank-domain helper for figures.

Designed to be import-friendly from any eval script (stage1_tx, dlpfc_eval,
stage15_gene_map, svg_eval, spot_deconv).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-gene 1-D metrics (vector vs vector across spots)
# ---------------------------------------------------------------------------


def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    ra = _rankdata(a); rb = _rankdata(b)
    return pearson_1d(ra, rb)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Stable average-rank — matches scipy.stats.rankdata(method='average')."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64) + 1.0
    return ranks


def rmse_zscore(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE after independent z-scoring — matches SpatialBenchmarking RMSE."""
    def _z(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-12 else 1.0)
    return float(np.sqrt(np.mean((_z(np.asarray(a, dtype=np.float64))
                                    - _z(np.asarray(b, dtype=np.float64))) ** 2)))


def ssim_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D structural similarity per SpatialBenchmarking cal_ssim — K1=0.01,
    K2=0.03, M (dynamic range) = 1.0 after min-max scaling each array.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    def _scale(x):
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
    a = _scale(a); b = _scale(b); M = 1.0
    mu1, mu2 = a.mean(), b.mean()
    s1 = np.sqrt(((a - mu1) ** 2).mean())
    s2 = np.sqrt(((b - mu2) ** 2).mean())
    s12 = ((a - mu1) * (b - mu2)).mean()
    k1, k2 = 0.01, 0.03
    C1 = (k1 * M) ** 2; C2 = (k2 * M) ** 2; C3 = C2 / 2.0
    l = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c = (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2)
    s = (s12 + C3) / (s1 * s2 + C3) if s1 * s2 > 1e-12 else 0.0
    return float(l * c * s)


def jsd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric KL after scale_plus normalisation — matches
    SpatialBenchmarking GenesMetrics.JS."""
    from scipy.stats import entropy
    def _splus(x):
        x = x - x.min() + 1e-9
        return x / x.sum()
    p = _splus(np.asarray(a, dtype=np.float64).ravel())
    q = _splus(np.asarray(b, dtype=np.float64).ravel())
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))


# ---------------------------------------------------------------------------
# Vectorised across many genes (returns one value per column)
# ---------------------------------------------------------------------------


def pearson_per_gene(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-column Pearson — pred/target both (N, G)."""
    pc = pred - pred.mean(axis=0, keepdims=True)
    tc = target - target.mean(axis=0, keepdims=True)
    denom = np.sqrt((pc ** 2).mean(axis=0) * (tc ** 2).mean(axis=0))
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    nz = denom > 1e-12
    out[nz] = (pc[:, nz] * tc[:, nz]).mean(axis=0) / denom[nz]
    return out


def spearman_per_gene(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorised Spearman — rank pred & target independently per column,
    then per-gene Pearson on ranks."""
    pr = np.apply_along_axis(_rankdata, 0, pred)
    tr = np.apply_along_axis(_rankdata, 0, target)
    return pearson_per_gene(pr, tr)


def rmse_zscore_per_gene(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    def _z(M):
        s = M.std(axis=0, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        return (M - M.mean(axis=0, keepdims=True)) / s
    diff = _z(pred) - _z(target)
    return np.sqrt((diff ** 2).mean(axis=0))


def ssim_per_gene(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([ssim_1d(target[:, j], pred[:, j])
                       for j in range(pred.shape[1])], dtype=np.float64)


def jsd_per_gene(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([jsd_1d(target[:, j], pred[:, j])
                       for j in range(pred.shape[1])], dtype=np.float64)


# ---------------------------------------------------------------------------
# Convenience — full 4-metric suite (SpatialBenchmarking compatible)
# ---------------------------------------------------------------------------


def spatial_bench_suite(pred: np.ndarray, target: np.ndarray,
                         *, include_spearman: bool = True) -> dict[str, np.ndarray]:
    """Return per-gene arrays for pearson, ssim, rmse_zscore, jsd (+spearman).
    All arrays shape (n_genes,).  NaN where a gene has zero variance.
    """
    out = {
        "pearson": pearson_per_gene(pred, target),
        "ssim": ssim_per_gene(pred, target),
        "rmse_zscore": rmse_zscore_per_gene(pred, target),
        "jsd": jsd_per_gene(pred, target),
    }
    if include_spearman:
        out["spearman"] = spearman_per_gene(pred, target)
    return out


def summarise(per_gene: dict[str, np.ndarray], prefix: str = "") -> dict[str, float]:
    """Reduce per-gene arrays to scalar summary stats (mean / median / Q1 / Q3).
    `prefix` is prepended to every output key (e.g. 'imp/' or 'gene_map/').
    """
    out: dict[str, float] = {}
    for k, v in per_gene.items():
        if v.size == 0:
            continue
        out[f"{prefix}{k}_mean"] = float(np.nanmean(v))
        out[f"{prefix}{k}_median"] = float(np.nanmedian(v))
        out[f"{prefix}{k}_q1"] = float(np.nanpercentile(v, 25))
        out[f"{prefix}{k}_q3"] = float(np.nanpercentile(v, 75))
        out[f"{prefix}{k}_n_valid"] = float(np.isfinite(v).sum())
    return out
