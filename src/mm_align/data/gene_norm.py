"""Runtime gene normalisation — Stage-1 / Stage-1.5 shared helper.

Stage 1's `PairedSpotDataset` applies a configurable per-gene normalisation
(mode + stats_path + eps + min_scale + clip) to `hvg_log` before handing it
to the tx encoder.  Stage 1.5 must apply the **same** transformation when:

  (1) it pre-encodes anchor `hvg_log` into `h_tx` (cached to <shard>.htx.npy)
  (2) it aggregates a region's `hvg_log` and pushes the result through the
       frozen Stage-1 tx_encoder to get `h_region_tx`

Without this, the Stage-1 encoder sees a distribution shift between training
and inference — e.g. `nonzero_z`-trained encoder receiving raw log1p values.

This module exposes that logic as a standalone, vectorised callable so
PairedSpotDataset and SpatialSampleDataset can share it.  The behaviour is
intentionally identical to the pairs.py implementation (mode names, eps,
min_scale, clip, zero-preservation).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np


_VALID_MODES = ("none", "global_z", "global_robust_z", "nonzero_z", "global_median")


class GeneNormalizer:
    """Per-gene log1p → normalised features, matching PairedSpotDataset.

    Build from the **same** cfg dict that pairs.py / Stage 1 used:
        cfg = {"mode": "nonzero_z",
                "stats_path": "...gene_stats.npz",
                "eps": 1e-6, "min_scale": 0.05, "clip": 8.0}

    The optional `vocab_keep_indices` slices the per-gene stats vectors to
    match Stage-1's clipped vocabulary order — pass the same array (or None)
    that the sampler/encoder uses for column subsetting.
    """

    def __init__(self, cfg: dict | None,
                 *, full_hvg_dim: int, hvg_dim: int,
                 vocab_keep_indices: Optional[np.ndarray] = None):
        self.mode: str = "none"
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.zero_preserve: bool = False
        self.clip: float = 0.0
        self.hvg_dim = int(hvg_dim)
        if not cfg or cfg.get("mode", "none") == "none":
            return
        mode = cfg["mode"]
        if mode not in _VALID_MODES:
            raise ValueError(f"gene_norm.mode={mode!r} unknown; "
                             f"expected one of {_VALID_MODES}")
        stats_path = Path(cfg.get("stats_path",
                                    "results/cache/prepared/gene_stats.npz"))
        if not stats_path.exists():
            raise FileNotFoundError(
                f"gene_norm.mode={mode} but stats_path not found: {stats_path}.")
        stats = np.load(stats_path)
        eps = float(cfg.get("eps", 1e-6))
        min_scale = float(cfg.get("min_scale", 0.05))

        if mode == "global_z":
            center = stats["mean"].astype(np.float32)
            raw_scale = stats["std"].astype(np.float32)
        elif mode == "global_robust_z":
            center = stats["median"].astype(np.float32)
            raw_scale = stats["mad"].astype(np.float32)
        elif mode == "nonzero_z":
            if "nonzero_mean" not in stats.files:
                raise KeyError(
                    f"nonzero_z requires nonzero_mean/nonzero_std in {stats_path}; "
                    "re-run prepare_data.py.")
            center = stats["nonzero_mean"].astype(np.float32)
            raw_scale = stats["nonzero_std"].astype(np.float32)
            self.zero_preserve = True
        else:                                  # global_median
            if "nonzero_median" in stats.files:
                center = np.zeros_like(stats["nonzero_median"]).astype(np.float32)
                raw_scale = stats["nonzero_median"].astype(np.float32)
            else:
                center = np.zeros_like(stats["median"]).astype(np.float32)
                raw_scale = stats["median"].astype(np.float32)
            self.zero_preserve = True

        scale = np.maximum(raw_scale + eps, min_scale).astype(np.float32)

        # Slice stats to match clipped vocab when applicable.
        if vocab_keep_indices is not None and center.shape[0] == full_hvg_dim:
            center = center[vocab_keep_indices]
            scale = scale[vocab_keep_indices]

        if center.shape[0] != self.hvg_dim:
            raise ValueError(
                f"gene_stats has {center.shape[0]} genes but effective "
                f"hvg_dim={self.hvg_dim} (vocab_clip-aware).  Re-run prepare_data.py.")
        self.center = center
        self.scale = scale
        self.clip = float(cfg.get("clip", 8.0))
        self.mode = mode

    def __bool__(self) -> bool:
        return self.mode != "none"

    def apply_np(self, hvg: np.ndarray) -> np.ndarray:
        """Vectorised — accepts (N, D) or (D,).  Returns float32."""
        if self.mode == "none":
            return hvg.astype(np.float32, copy=False)
        z = (hvg - self.center) / self.scale
        if self.zero_preserve:
            z = np.where(hvg > 0, z, 0.0)
        if self.clip > 0:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32, copy=False)

    # Sugar — also callable.
    __call__ = apply_np
