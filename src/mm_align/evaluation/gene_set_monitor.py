"""Curated gene-set monitoring for Stage-1 training.

Idea: train.py validation pass usually emits aggregate masked-symbol metrics.
But to understand WHAT the encoder is learning, we want to track gene-gene
co-occurrence and CLS-embedding clustering for a few biologically-meaningful
gene sets (e.g. immune-T-cell, fibroblast, epithelial-ductal).

This module:
  - exposes `CURATED_SETS` with a few small, well-known marker panels
  - `compute_set_metrics(pool, gene_vocab)` returns:
      * per-set: mean & max pairwise PCC over spots (data-side check —
        confirms our HVG-normalized data actually shows the expected biology)
      * per-set: silhouette of CLS-embeddings of "spots highly expressing
        this set" vs "background spots" (representation-quality check)

Called from train.py val loop at low frequency (every K epochs).
"""
from __future__ import annotations

import numpy as np


# ── Curated marker panels (small, well-known) ──
# Source: standard immunology / pathology references; ≤8 genes each
# so a missing-one-gene doesn't kill the panel.
CURATED_SETS = {
    "immune_T":      ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "GZMA", "GZMB", "PRF1"],
    "immune_B":      ["MS4A1", "CD79A", "CD79B", "IGHM", "IGKC", "JCHAIN"],
    "macrophage":    ["CD68", "CD163", "MRC1", "MSR1", "C1QA", "C1QB", "C1QC", "MARCO"],
    "fibroblast":    ["COL1A1", "COL1A2", "COL3A1", "FAP", "PDGFRA", "PDGFRB", "ACTA2"],
    "endothelial":   ["PECAM1", "VWF", "CDH5", "CLDN5", "ENG", "KDR"],
    "epithelial":    ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    "proliferation": ["MKI67", "PCNA", "TOP2A", "MCM2", "MCM5"],
    "muscle_smooth": ["ACTA2", "MYH11", "TAGLN", "CNN1"],
}


def panel_indices(gene_vocab: list[str]) -> dict[str, np.ndarray]:
    """For each curated set, return the indices of its genes that exist in the
    HVG vocab.  Sets with < 2 members in vocab are dropped."""
    vmap = {g: i for i, g in enumerate(gene_vocab)}
    out = {}
    for name, members in CURATED_SETS.items():
        idx = np.array([vmap[g] for g in members if g in vmap], dtype=np.int64)
        if len(idx) >= 2:
            out[name] = idx
    return out


def compute_set_metrics(hvg_batch: np.ndarray,
                        cls_batch: np.ndarray | None,
                        gene_vocab: list[str],
                        *, top_quantile: float = 0.20) -> dict[str, float]:
    """
    Parameters
    ----------
    hvg_batch : (N, n_hvg) float — log1p HVG values for some spot pool
    cls_batch : (N, D) float | None — CLS embeddings for the same spots
                (set None to skip silhouette metrics)
    gene_vocab : list of gene symbols, same order as hvg_batch columns
    top_quantile : float — "high-expressing this panel" cutoff for silhouette

    Returns
    -------
    dict of metric_name → float, suitable for direct logging.
      <set>/pcc_mean      : mean pairwise PCC of the panel's genes across spots
      <set>/coverage_pct  : pct of panel genes that are present in vocab
      <set>/cls_silhouette: silhouette of "high vs low expression" clusters
                              (only when cls_batch is given AND both clusters
                              have ≥10 members)
    """
    panels = panel_indices(gene_vocab)
    metrics: dict[str, float] = {}
    for name, idx in panels.items():
        sub = hvg_batch[:, idx]                     # (N, k)
        # PCC matrix on genes (columns); off-diagonal mean
        sub_c = sub - sub.mean(axis=0, keepdims=True)
        std = sub_c.std(axis=0) + 1e-8
        sub_z = sub_c / std                          # (N, k)
        corr = (sub_z.T @ sub_z) / max(1, sub_z.shape[0])
        k = corr.shape[0]
        off = corr[~np.eye(k, dtype=bool)]
        metrics[f"set/{name}/pcc_mean"] = float(off.mean())
        metrics[f"set/{name}/coverage"] = float(len(idx) / len(CURATED_SETS[name]))

        # silhouette of high vs low expressors of this panel
        if cls_batch is not None:
            score = sub.mean(axis=1)                 # panel-score per spot
            thresh = np.quantile(score, 1 - top_quantile)
            hi = score >= thresh
            lo = ~hi
            if hi.sum() >= 10 and lo.sum() >= 10:
                try:
                    from sklearn.metrics import silhouette_score
                    labels = hi.astype(np.int32)
                    # Subsample for speed if N > 4000
                    if cls_batch.shape[0] > 4000:
                        rng = np.random.default_rng(0)
                        sel = rng.choice(cls_batch.shape[0], 4000, replace=False)
                        sil = silhouette_score(cls_batch[sel], labels[sel])
                    else:
                        sil = silhouette_score(cls_batch, labels)
                    metrics[f"set/{name}/cls_silhouette"] = float(sil)
                except Exception:
                    pass
    return metrics
