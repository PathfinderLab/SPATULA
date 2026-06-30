"""SEAL-style linear probing — 5-fold Ridge with optional PCA, per-gene
Pearson / Spearman / R² / L2, aggregated per-organ.

Mirrors `seal.utils.eval_utils.run_linprobe`.  No MLP head — pure Ridge,
since that's what SEAL uses for fair comparison with their HEST-Bench setup.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import Ridge as _SkRidge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _SkPCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


def _ridge(alpha: float = 1.0):
    """sklearn Ridge — handles multi-output (y ∈ ℝ^G) natively.
    cuML.Ridge is single-output only; we don't use it here."""
    return _SkRidge(alpha=alpha)


def _per_gene_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, list[float]]:
    """Per-gene Pearson, Spearman, R², L2 error.  pred/target shape (N, G)."""
    G = pred.shape[1]
    pccs, spcs, r2s, l2s = [], [], [], []
    for j in range(G):
        y = target[:, j]; p = pred[:, j]
        if y.std() < 1e-8 or p.std() < 1e-8:
            pccs.append(0.0); spcs.append(0.0); r2s.append(0.0)
            l2s.append(float(np.mean((p - y) ** 2))); continue
        try:
            pccs.append(float(pearsonr(p, y)[0]))
        except Exception:
            pccs.append(0.0)
        try:
            spcs.append(float(spearmanr(p, y)[0]))
        except Exception:
            spcs.append(0.0)
        try:
            r2s.append(float(r2_score(y, p)))
        except Exception:
            r2s.append(0.0)
        l2s.append(float(np.mean((p - y) ** 2)))
    return {"pcc": pccs, "spearman": spcs, "r2": r2s, "l2": l2s}


def run_linprobe(Z_train: np.ndarray, Z_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray,
                 *,
                 gene_names: Optional[List[str]] = None,
                 organ_to_genes: Optional[Dict[str, List[str]]] = None,
                 folds: int = 5,
                 pca_reduce: bool = True,
                 pca_n: int = 256,
                 alpha: float = 1.0,
                 random_state: int = 42) -> dict:
    """5-fold Ridge probe on (Z_train, y_train), evaluated on (Z_test, y_test).

    Per fold:
      (optional) StandardScaler + PCA(`pca_n`) on the *training* features.
      Fit Ridge on (Z_train_fold, y_train_fold), predict on Z_test (fixed).
      Score per-gene PCC/Spearman/R²/L2, then average across genes.
    Final report = mean ± std across folds.  When `organ_to_genes` is given,
    we additionally break down per-organ.
    """
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_test = y_test[:, None]
    G = y_test.shape[1]
    gene_names = gene_names or [f"gene_{j}" for j in range(G)]

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_records: list[dict[str, list[float]]] = []

    for fold_i, (tr_idx, _) in enumerate(kf.split(Z_train)):
        Zt = Z_train[tr_idx]; yt = y_train[tr_idx]
        Ze = Z_test
        if pca_reduce and Zt.shape[1] > pca_n:
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("pca", _SkPCA(n_components=pca_n, random_state=random_state + fold_i,
                               svd_solver="auto")),
            ])
            Zt = pipe.fit_transform(Zt)
            Ze = pipe.transform(Ze)
        clf = _ridge(alpha=alpha)
        clf.fit(Zt, yt)
        pred = clf.predict(Ze)
        pred = np.asarray(pred)
        fold_records.append(_per_gene_metrics(pred, y_test))

    # mean / std across folds, per-gene
    out: dict = {}
    out["folds"] = folds
    out["n_genes"] = G
    out["gene_names"] = gene_names
    pcc_mat = np.array([rec["pcc"] for rec in fold_records])
    sp_mat  = np.array([rec["spearman"] for rec in fold_records])
    r2_mat  = np.array([rec["r2"] for rec in fold_records])
    l2_mat  = np.array([rec["l2"] for rec in fold_records])
    out["pcc/mean"] = float(pcc_mat.mean())
    out["pcc/std"] = float(pcc_mat.mean(axis=0).std())
    out["spearman/mean"] = float(sp_mat.mean())
    out["spearman/std"] = float(sp_mat.mean(axis=0).std())
    out["r2/mean"] = float(r2_mat.mean())
    out["r2/std"] = float(r2_mat.mean(axis=0).std())
    out["l2/mean"] = float(l2_mat.mean())
    out["l2/std"] = float(l2_mat.mean(axis=0).std())

    # per-gene mean across folds (for sorting / inspection)
    out["per_gene_pcc_mean"] = dict(zip(gene_names, pcc_mat.mean(axis=0).tolist()))
    out["per_gene_spearman_mean"] = dict(zip(gene_names, sp_mat.mean(axis=0).tolist()))

    # per-organ breakdown
    if organ_to_genes:
        gene_idx = {g: i for i, g in enumerate(gene_names)}
        per_organ = {}
        for organ, genes in organ_to_genes.items():
            cols = [gene_idx[g] for g in genes if g in gene_idx]
            if not cols:
                continue
            per_organ[organ] = {
                "n_genes": len(cols),
                "pcc_mean": float(pcc_mat[:, cols].mean()),
                "pcc_std": float(pcc_mat[:, cols].mean(axis=0).std()),
                "spearman_mean": float(sp_mat[:, cols].mean()),
                "spearman_std": float(sp_mat[:, cols].mean(axis=0).std()),
                "r2_mean": float(r2_mat[:, cols].mean()),
            }
        out["per_organ"] = per_organ
    return out
