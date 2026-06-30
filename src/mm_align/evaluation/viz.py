"""UMAP visualisation of learned representations.

Three plots per call:
  - h_image colored by organ (or sample if no organ)
  - h_tx colored by organ (or sample)
  - joint (h_image, h_tx) embedded together, colored by modality, to see if they mix
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
               seed: int = 0) -> np.ndarray:
    if X.shape[0] < n_neighbors + 1:
        n_neighbors = max(2, X.shape[0] - 1)
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist, random_state=seed, verbose=False)
        return reducer.fit_transform(X)
    except Exception:
        # Fallback: PCA
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)


def _scatter(ax, xy: np.ndarray, labels: np.ndarray | None, title: str,
             max_pts: int = 5000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    if xy.shape[0] > max_pts:
        idx = rng.choice(xy.shape[0], max_pts, replace=False)
        xy = xy[idx]
        if labels is not None:
            labels = labels[idx]
    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6)
    else:
        uniq = np.unique(labels)
        cmap = plt.cm.tab20.colors if len(uniq) <= 20 else plt.cm.viridis(np.linspace(0, 1, len(uniq)))
        for i, u in enumerate(uniq):
            mask = labels == u
            ax.scatter(xy[mask, 0], xy[mask, 1], s=4, alpha=0.7,
                       color=cmap[i % len(cmap)], label=str(u))
        if len(uniq) <= 12:
            ax.legend(fontsize=7, markerscale=2)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def render_umap_panel(pool: dict, out_path: Path, *,
                      organ_labels: np.ndarray | None = None,
                      title: str = "") -> None:
    """`pool` is the output of evaluation.zero_shot.encode_loader on val (or test)."""
    if pool["h_image"] is None:
        return

    labels = organ_labels if organ_labels is not None else pool.get("sample_idx")

    emb_img = _safe_umap(pool["h_image"])
    emb_tx = _safe_umap(pool["h_tx"])
    joint = np.concatenate([pool["h_image"], pool["h_tx"]], axis=0)
    emb_joint = _safe_umap(joint)
    n = pool["h_image"].shape[0]
    modality = np.array(["image"] * n + ["tx"] * n)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    _scatter(axes[0, 0], emb_img, labels, "UMAP of h_image"
             + (" (color: organ)" if organ_labels is not None else " (color: slide)"))
    _scatter(axes[0, 1], emb_tx, labels, "UMAP of h_tx"
             + (" (color: organ)" if organ_labels is not None else " (color: slide)"))
    _scatter(axes[1, 0], emb_joint, modality, "UMAP of h_image+h_tx (color: modality)")
    # Per-arm raw UNI baseline UMAP for direct visual contrast
    emb_uni = _safe_umap(pool["image_feat"])
    _scatter(axes[1, 1], emb_uni, labels, "UMAP of raw UNI (baseline)")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
