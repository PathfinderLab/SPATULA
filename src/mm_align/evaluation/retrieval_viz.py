"""Qualitative retrieval grid — image → tx top-K with hit / miss marks.

For each of `n_queries` randomly-picked test spots:
  • Query patch (the spot's own H&E patch)
  • Top-K spots ranked by cosine similarity in the (paired tx-side) projection
  • Each retrieved patch is shown with its H&E image and a green/red box:
      green = the retrieved spot is the *same* one as the query (rank-1 hit)
      red   = different spot (miss)
  • The rank of the true positive is printed above the query.

Saved as a single PNG.

The function assumes the dataset is in `image_mode="raw"` (i.e. raw 224×224
uint8 patches are present in batches as `image_raw`); if not, we still draw
something but with a placeholder for the patch.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import h5py
import torch
import torch.nn.functional as F


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def _load_patch(shard_h5_path: Path, hest_patch_dir: Path, spot_idx: int) -> np.ndarray | None:
    """Load the raw image patch for one spot from the original HEST patches H5.
    Returns (224, 224, 3) uint8 or None on failure."""
    try:
        sid = shard_h5_path.stem
        with h5py.File(shard_h5_path, "r") as sf:
            pidx = int(sf["patch_idx"][spot_idx])
        with h5py.File(hest_patch_dir / f"{sid}.h5", "r") as pf:
            img = pf["img"][pidx]
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


def render_retrieval_examples(
    z_image: np.ndarray, z_tx: np.ndarray,
    *,
    sample_idx: np.ndarray, spot_idx: np.ndarray,
    sample_ids: list[str],
    prepared_dir: Path, hest_patch_dir: Path,
    out_path: Path,
    n_queries: int = 6, top_k: int = 5,
    seed: int = 0, title: str = "",
) -> None:
    """
    z_image, z_tx       : (N, D) numpy arrays (already-pool-encoded test reps)
    sample_idx, spot_idx: (N,) maps each row to (shard_index, spot_in_shard)
    sample_ids          : list of shard sample-ids (len = #shards)
    """
    if z_image is None or z_tx is None or z_image.shape[1] != z_tx.shape[1]:
        return

    a = _norm(z_image.astype(np.float32))
    b = _norm(z_tx.astype(np.float32))
    sim = a @ b.T                                      # (N, N)
    N = sim.shape[0]

    rng = np.random.default_rng(seed)
    queries = rng.choice(N, size=min(n_queries, N), replace=False)

    fig, axes = plt.subplots(len(queries), top_k + 1,
                             figsize=(2.2 * (top_k + 1), 2.4 * len(queries)))
    if len(queries) == 1:
        axes = axes[None, :]

    for row, qi in enumerate(queries):
        order = np.argsort(-sim[qi])                  # descending
        rank_of_truth = int(np.where(order == qi)[0][0]) + 1
        topk = order[:top_k]

        # Query patch
        ax = axes[row, 0]
        si = int(sample_idx[qi]); pi = int(spot_idx[qi])
        shard = prepared_dir / f"{sample_ids[si]}.h5"
        img = _load_patch(shard, hest_patch_dir, pi)
        if img is not None:
            ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Q {sample_ids[si]}#{pi}\nrank(GT)={rank_of_truth}",
                     fontsize=8)
        for s in ax.spines.values():
            s.set_color("#0064c8"); s.set_linewidth(2)

        # Top-K retrievals
        for k_idx, ri in enumerate(topk):
            ax = axes[row, k_idx + 1]
            si_r = int(sample_idx[ri]); pi_r = int(spot_idx[ri])
            shard_r = prepared_dir / f"{sample_ids[si_r]}.h5"
            img_r = _load_patch(shard_r, hest_patch_dir, pi_r)
            if img_r is not None:
                ax.imshow(img_r)
            ax.set_xticks([]); ax.set_yticks([])
            hit = (ri == qi)
            color = "#1e9d4d" if hit else "#cc2424"
            mark = "✓" if hit else "✗"
            ax.set_title(f"{mark} #{k_idx+1}  {sample_ids[si_r]}#{pi_r}\n"
                         f"sim={sim[qi, ri]:.3f}", fontsize=8, color=color)
            for s in ax.spines.values():
                s.set_color(color); s.set_linewidth(2.5)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
