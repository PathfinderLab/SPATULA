"""Paired-spot dataset.

Per-sample HDF5 shard ({prepared_dir}/{sid}.h5) layout:
  /barcode         (S,)  str
  /coords          (S, 2) float32                pixel coords
  /uni_feat        (S, D_img=1536) float32       pre-extracted UNI features (frozen)
  /novae_latent    (S, D_tx=64) float32          pre-extracted novae latent (frozen)
  /hvg_log         (S, n_hvg=2048) float32       log1p normalized expression on shared HVG vocab
  /patch_idx       (S,) int32                    index into /data/hest/patches/{sid}.h5["img"]
                                                  (filled in by prepare_data; lazily backfilled here
                                                   from barcode lookup if absent.)

Items returned:
  image           : float32(D_img)         pre-extracted UNI features (always — cheap)
  image_raw       : uint8(224,224,3)       only if image_mode="raw"
  tx_latent       : float32(D_tx)
  hvg             : float32(D_hvg)         if load_hvg=True
  neighbors       : int64(k_spatial)       within-sample spot indices
  sample_idx, spot_idx
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Split + shard helpers were factored out for readability — keep them
# importable from this module so existing call-sites (scripts/data/prepare.py,
# scripts/data/resplit.py) don't need to change.
from .splits import (
    default_split,
    stratified_split,
    list_prepared_shards,
    decode_bytes_array as _decode_bytes_array,
)
from .shard import Shard as _Shard


class PairedSpotDataset(Dataset):
    def __init__(self, shards: Sequence[str | Path], k_spatial: int = 8,
                 load_hvg: bool = True, image_mode: str = "feature",
                 hest_patch_dir: str | Path | None = None,
                 gene_norm_cfg: dict | None = None,
                 tx_only: bool = False,
                 min_seq_len: int = 0,
                 max_seq_len: int = 0,
                 vocab_keep_indices: np.ndarray | None = None,
                 sampling_strategy: str = "random",
                 sampling_alpha: float = 1.0,
                 must_include_mask: np.ndarray | None = None):
        """
        vocab_keep_indices : optional np.int64 array of column indices into
          hvg_log to KEEP.  When set, every fetched hvg row is sliced to
          hvg[vocab_keep_indices].  Use to runtime-clip a large prepared
          vocab to top-K by priority_rank without re-running prepare.
          Order is preserved — vocab token IDs downstream must use the same
          ordering (read it from vocab.csv['gene'].iloc[keep_indices]).
        max_seq_len : if > 0, cap non-zero positions per spot to this many.
          When a spot expresses more genes than the cap, sample a random
          subset (without replacement) preserving expression values.  Zero
          positions are unaffected.  This bounds the transformer's
          attention O(L²) cost on outlier spots (TENX HD spots reach 13K+).
        """
        self.vocab_keep_indices = (None if vocab_keep_indices is None
                                    else np.asarray(vocab_keep_indices, dtype=np.int64))
        self.max_seq_len = int(max_seq_len)
        if sampling_strategy not in ("random", "top_k", "weighted"):
            raise ValueError(f"sampling_strategy={sampling_strategy!r} unknown; "
                             "expected: random | top_k | weighted")
        self.sampling_strategy = sampling_strategy
        self.sampling_alpha = float(sampling_alpha)
        # Optional boolean mask (post vocab-clip) marking must_include genes.
        # When max_seq_len triggers, those positions are never dropped if they
        # are expressed.  Shape == (effective hvg_dim,).
        self.must_include_mask = (None if must_include_mask is None
                                   else np.asarray(must_include_mask, dtype=bool))
        """
        tx_only: if True, skip loading image/tx_latent/neighbors arrays in
          __getitem__.  Stage 1 doesn't use them (align_weight=0 → image side
          and neighbor-aux are off).  Cuts per-spot I/O ~3.6× because we
          skip uni_feat (1536) + tx_latent (64) + neighbors (k=8).
          Image/tx fields are filled with cheap zero scalars so collate
          (pad_collate) and downstream model.forward still see consistent
          schema.
        """
        self.image_mode = image_mode
        load_raw = (image_mode == "raw")
        if load_raw and hest_patch_dir is None:
            raise ValueError("image_mode='raw' requires hest_patch_dir.")
        self.hest_patch_dir = Path(hest_patch_dir) if hest_patch_dir else None
        self.tx_only = bool(tx_only)
        self.min_seq_len = int(min_seq_len)

        self.shards = [
            _Shard(Path(p), k_spatial=k_spatial,
                   hest_patch_dir=self.hest_patch_dir, load_raw=load_raw)
            for p in shards
        ]
        self.load_hvg = load_hvg
        self.load_raw = load_raw

        starts = np.cumsum([0] + [s.n_spots for s in self.shards])
        self._starts = starts
        self.total = int(starts[-1])

        # Build a spot-level filter mask: drop spots with < min_seq_len expressed HVG.
        # Done as a one-time pass over hvg_log (just (X > 0).sum(1) per shard).
        # Result: self._valid_indices = sorted list of global indices to use.
        #
        # PERFORMANCE: cache the per-shard nnz vector on disk next to the
        # shard (small — 4 bytes × n_spots).  With 8 DDP ranks each invoking
        # this filter independently, the cache turns a ~10 min scan into a
        # ~1 s metadata read on warm runs.  The cache key is the shard mtime,
        # so it auto-invalidates when prepare_data.py rewrites a shard.
        self._valid_indices = None
        if self.min_seq_len > 0 and load_hvg:
            CHUNK = 8192
            keep_lists: list[np.ndarray] = []
            kept_idx = self.vocab_keep_indices
            # Cache key: full vocab → `.nnz.npy`; clipped → keyed by the
            # vocab_keep_indices file content hash (so reused across runs
            # that share the same clip).
            clip_tag = ""
            if kept_idx is not None:
                import hashlib
                clip_tag = "." + hashlib.md5(kept_idx.tobytes()).hexdigest()[:8] + ".nnz.npy"
            for shard_idx, shard in enumerate(self.shards):
                shard_path = Path(shard.path) if hasattr(shard, "path") else None
                if shard_path is None:
                    shard_path = Path(getattr(shard, "_path", ""))
                if shard_path and shard_path.exists():
                    nnz_cache = (shard_path.with_suffix(clip_tag)
                                  if kept_idx is not None
                                  else shard_path.with_suffix(".nnz.npy"))
                else:
                    nnz_cache = None
                f = shard.open()
                if "hvg_log" not in f:
                    keep_lists.append(np.arange(starts[shard_idx], starts[shard_idx+1], dtype=np.int64))
                    continue
                dset = f["hvg_log"]
                n_rows = dset.shape[0]
                # Try cache (vocab-clip aware via clip_tag).
                # Read defensively — under DDP, several ranks may be
                # writing this file concurrently and one of them can read it
                # mid-truncate (empty or partial), which np.load surfaces as
                # EOFError.  Treat any read failure as a cache miss and recompute.
                nnz = None
                if nnz_cache and nnz_cache.exists() \
                        and nnz_cache.stat().st_size > 0 \
                        and nnz_cache.stat().st_mtime >= shard_path.stat().st_mtime:
                    try:
                        cached = np.load(nnz_cache)
                        if cached.shape[0] == n_rows:
                            nnz = cached
                    except (EOFError, ValueError, OSError) as e:
                        # Another rank is writing; fall through to recompute.
                        pass
                if nnz is None:
                    # Compute in chunks (bounded RAM) and cache.
                    parts = []
                    for r0 in range(0, n_rows, CHUNK):
                        r1 = min(r0 + CHUNK, n_rows)
                        block = dset[r0:r1]
                        if kept_idx is not None:
                            block = block[:, kept_idx]
                        parts.append((block > 0).sum(axis=1).astype(np.int32))
                    nnz = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
                    if nnz_cache is not None:
                        # Atomic write — write to a per-process temp file then
                        # rename over the final path.  `os.replace` is atomic
                        # on POSIX, so concurrent readers never see a
                        # half-written file.
                        try:
                            import os, tempfile
                            tmp_fd, tmp_path = tempfile.mkstemp(
                                prefix=nnz_cache.name + ".",
                                suffix=".tmp",
                                dir=str(nnz_cache.parent),
                            )
                            os.close(tmp_fd)
                            np.save(tmp_path, nnz)
                            # np.save may append `.npy` if absent — find the real file.
                            real_tmp = (tmp_path
                                        if Path(tmp_path).exists()
                                        else tmp_path + ".npy")
                            os.replace(real_tmp, nnz_cache)
                        except Exception:
                            pass
                ok = np.nonzero(nnz >= self.min_seq_len)[0]
                keep_lists.append(ok.astype(np.int64) + starts[shard_idx])
            self._valid_indices = (np.concatenate(keep_lists).astype(np.int64)
                                    if keep_lists else np.empty(0, dtype=np.int64))
            self._effective_total = len(self._valid_indices)
        else:
            self._effective_total = self.total
        self.uni_dim = self.shards[0].uni_dim if self.shards else 0
        self.tx_dim = self.shards[0].tx_dim if self.shards else 0
        full_hvg_dim = self.shards[0].hvg_dim if (self.shards and load_hvg) else 0
        # Effective vocab dimension after optional runtime clip.
        if self.vocab_keep_indices is not None and full_hvg_dim:
            if int(self.vocab_keep_indices.max()) >= full_hvg_dim:
                raise ValueError(
                    f"vocab_keep_indices max={int(self.vocab_keep_indices.max())} "
                    f"≥ shard hvg_dim={full_hvg_dim}"
                )
            self.hvg_dim = int(len(self.vocab_keep_indices))
        else:
            self.hvg_dim = full_hvg_dim
        self._full_hvg_dim = full_hvg_dim

        # ── Runtime gene normalization (batch-effect reducer) ──
        # gene_norm_cfg = {"mode": "none" | "global_z" | "global_robust_z" | "nonzero_z",
        #                  "stats_path": "...gene_stats.npz",
        #                  "eps": 1e-6,
        #                  "min_scale": 0.05}
        #
        # Modes (zero-inflation handling):
        #   none            — pass `hvg_log` through unchanged.
        #   global_z        — (x − μ_all[g]) / max(σ_all[g], min_scale).
        #                     μ/σ over ALL values (zeros included) — dominated
        #                     by zeros; gives small per-spot lift.
        #   global_robust_z — same with median/MAD.  ⚠ DANGEROUS — 99% of
        #                     HVG have median=MAD=0 on ST data.
        #   nonzero_z (recommended for masked modeling)
        #                   — at non-zero positions: (x − μ_nz[g]) / max(σ_nz[g], min_scale)
        #                     at zero positions:    leave 0.0
        #                     μ_nz/σ_nz computed over NON-zero values only,
        #                     so values are in a meaningful range (1–4 for log1p).
        #                     Removes per-gene baseline magnitude AND per-dataset
        #                     batch effect (since zero-inflation differences fall
        #                     out of the mask).  Preserves zero-removed token
        #                     semantics — exactly what SPATULA assumes.
        self._norm_mode = "none"
        self._norm_center: np.ndarray | None = None      # subtractive offset (per gene)
        self._norm_scale: np.ndarray | None = None       # divisor (post-floor)
        self._norm_zero_preserve = False                 # if True, keep zeros as zeros
        if gene_norm_cfg and gene_norm_cfg.get("mode", "none") != "none":
            mode = gene_norm_cfg["mode"]
            if mode not in ("global_z", "global_robust_z", "nonzero_z", "global_median"):
                raise ValueError(f"gene_norm.mode={mode!r} unknown; "
                                 f"expected: none | global_z | global_robust_z | nonzero_z | global_median")
            stats_path = Path(gene_norm_cfg.get("stats_path",
                              "results/cache/prepared/gene_stats.npz"))
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"gene_norm.mode={mode} but stats_path not found: {stats_path}.  "
                    "Run prepare_data.py to emit it, or set gene_norm.mode=none."
                )
            stats = np.load(stats_path)
            eps = float(gene_norm_cfg.get("eps", 1e-6))
            min_scale = float(gene_norm_cfg.get("min_scale", 0.05))
            if mode == "global_z":
                self._norm_center = stats["mean"].astype(np.float32)
                raw_scale = stats["std"].astype(np.float32)
            elif mode == "global_robust_z":
                self._norm_center = stats["median"].astype(np.float32)
                raw_scale = stats["mad"].astype(np.float32)
            elif mode == "nonzero_z":
                if "nonzero_mean" not in stats.files:
                    raise KeyError(
                        f"gene_norm.mode='nonzero_z' requires nonzero_mean/nonzero_std "
                        f"in {stats_path}, but they're absent.  Re-run prepare_data.py "
                        f"with the latest compute_global_gene_stats to refresh."
                    )
                self._norm_center = stats["nonzero_mean"].astype(np.float32)
                raw_scale = stats["nonzero_std"].astype(np.float32)
                self._norm_zero_preserve = True
            else:  # global_median — Geneformer-style ratio normalisation
                # Geneformer:  X_norm = (X / n_counts * 1e4) / nonzero_median(g)
                # We've already stored log1p(X / n_counts * 1e4) on disk, so the
                # equivalent operation in log-space is:
                #   z = log1p(x_raw) / log1p(median_raw)
                # We approximate via the precomputed log-space nonzero_median if
                # available, else fall back to the median field (median over ALL
                # values, including zeros — biased on zero-inflated ST data).
                #
                # NOTE: rank-based downstream models (Geneformer) discard
                # absolute value entirely; SPATULA does NOT — we feed value into
                # the Fourier value embedding.  So keep zeros at 0.0 and divide
                # non-zero positions only.
                if "nonzero_median" in stats.files:
                    self._norm_center = np.zeros_like(stats["nonzero_median"]).astype(np.float32)
                    raw_scale = stats["nonzero_median"].astype(np.float32)
                else:
                    log.warning("global_median requested but nonzero_median absent in "
                                f"{stats_path}; falling back to (zero-inflated) median.")
                    self._norm_center = np.zeros_like(stats["median"]).astype(np.float32)
                    raw_scale = stats["median"].astype(np.float32)
                self._norm_zero_preserve = True
            scale = np.maximum(raw_scale + eps, min_scale).astype(np.float32)
            # If runtime vocab_clip is on, gene_stats was computed for the full
            # vocab — slice it to match the clipped column order.
            if self.vocab_keep_indices is not None \
                    and self._norm_center.shape[0] == self._full_hvg_dim:
                self._norm_center = self._norm_center[self.vocab_keep_indices]
                scale = scale[self.vocab_keep_indices]
            self._norm_scale = scale
            self._norm_mode = mode
            # Outlier clamp — rare genes (small nonzero_count in train pool) can
            # produce huge z's when a never-seen-before sample expresses them.
            # Default ±8 keeps the input range to the Fourier value embedding
            # bounded.
            self._norm_clip = float(gene_norm_cfg.get("clip", 8.0))
            if self._norm_center.shape[0] != self.hvg_dim:
                raise ValueError(
                    f"gene_stats has {self._norm_center.shape[0]} genes but effective "
                    f"hvg_dim={self.hvg_dim} (after vocab_clip).  Re-run prepare_data.py."
                )

    def __len__(self):
        return self._effective_total

    def _locate(self, idx: int) -> tuple[int, int]:
        # If min_seq_len filter is active, redirect idx through the valid-index map.
        if self._valid_indices is not None:
            idx = int(self._valid_indices[idx])
        sample_idx = int(np.searchsorted(self._starts, idx, side="right") - 1)
        spot_idx = int(idx - self._starts[sample_idx])
        return sample_idx, spot_idx

    # Module-level zero tensors so workers don't allocate per-spot.
    _ZERO_IMG = torch.zeros(1, dtype=torch.float32)   # placeholder shape (filled at first call)
    _ZERO_TX = torch.zeros(1, dtype=torch.float32)
    _ZERO_NB = torch.zeros(1, dtype=torch.int64)

    def __getitem__(self, idx: int) -> dict:
        sample_idx, spot_idx = self._locate(idx)
        shard = self.shards[sample_idx]
        f = shard.open()

        if self.tx_only:
            # Fast path — Stage 1: skip uni_feat / novae_latent / neighbors I/O.
            # Provide cheap zero placeholders so the model schema stays unchanged.
            if PairedSpotDataset._ZERO_IMG.shape[0] != shard.uni_dim:
                PairedSpotDataset._ZERO_IMG = torch.zeros(shard.uni_dim, dtype=torch.float32)
                PairedSpotDataset._ZERO_TX = torch.zeros(shard.tx_dim, dtype=torch.float32)
                PairedSpotDataset._ZERO_NB = torch.zeros(shard.k_spatial, dtype=torch.int64)
            out = {
                "image": PairedSpotDataset._ZERO_IMG,
                "tx_latent": PairedSpotDataset._ZERO_TX,
                "sample_idx": sample_idx,
                "spot_idx": spot_idx,
                "neighbors": PairedSpotDataset._ZERO_NB,
            }
        else:
            out = {
                "image": torch.from_numpy(f["uni_feat"][spot_idx].astype(np.float32)),
                "tx_latent": torch.from_numpy(f["novae_latent"][spot_idx].astype(np.float32)),
                "sample_idx": sample_idx,
                "spot_idx": spot_idx,
                "neighbors": torch.from_numpy(shard.neighbors(spot_idx).astype(np.int64)),
            }
        if self.load_hvg and shard.hvg_dim > 0:
            hvg = f["hvg_log"][spot_idx].astype(np.float32)
            # Runtime vocab clip — keep only priority-top-K columns.  Order is
            # the caller's (e.g. priority_rank order) so the column index in
            # the returned `hvg` is the token-id-minus-N_SPECIAL when paired
            # with the matching clipped hvg_vocab_dict.json.
            if self.vocab_keep_indices is not None:
                hvg = hvg[self.vocab_keep_indices]
            pre_sampling_nnz = int(np.count_nonzero(hvg))
            if self._norm_mode != "none":
                z = (hvg - self._norm_center) / self._norm_scale
                if self._norm_zero_preserve:
                    # Keep zero positions as zero (zero-removed-token semantics).
                    z = np.where(hvg > 0, z, 0.0).astype(np.float32)
                # Clamp outliers (rare genes hitting unseen samples).
                if self._norm_clip > 0:
                    z = np.clip(z, -self._norm_clip, self._norm_clip)
                hvg = z
            # Token-budget cap — if this spot expresses more non-zero genes
            # than max_seq_len, drop the excess (set to zero) so the
            # downstream zero-removal stays within the budget.  Three
            # strategies:
            #   random  — uniform drop (dropout-like augmentation, low bias)
            #   top_k   — keep highest-value K (deterministic, Geneformer-ish)
            #   weighted — multinomial without replacement, p ∝ |value|^alpha
            #              (alpha=0 → random, alpha=∞ → top_k)
            # Must-include curated markers are ALWAYS kept when expressed
            # (NLP analogy: don't truncate proper nouns).
            if self.max_seq_len > 0:
                nz_idx = np.flatnonzero(hvg)
                if nz_idx.size > self.max_seq_len:
                    keep_budget = self.max_seq_len
                    forced_keep = np.empty(0, dtype=np.int64)
                    if self.must_include_mask is not None:
                        forced = nz_idx[self.must_include_mask[nz_idx]]
                        if forced.size:
                            if forced.size >= keep_budget:
                                # Edge case: more curated markers than budget
                                # → keep only the top-value ones.
                                fv = np.abs(hvg[forced])
                                forced = forced[np.argsort(fv)[::-1][:keep_budget]]
                                forced_keep = forced
                                keep_budget = 0
                            else:
                                forced_keep = forced
                                keep_budget -= forced.size
                    # Candidates for normal sampling = nz minus the forced set.
                    if forced_keep.size:
                        mask = np.ones(nz_idx.size, dtype=bool)
                        mask[np.searchsorted(nz_idx, forced_keep)] = False
                        cand_idx = nz_idx[mask]
                    else:
                        cand_idx = nz_idx
                    if keep_budget > 0 and cand_idx.size > keep_budget:
                        values = np.abs(hvg[cand_idx])
                        if self.sampling_strategy == "top_k":
                            keep_extra = cand_idx[np.argsort(values)[::-1][:keep_budget]]
                        elif self.sampling_strategy == "weighted":
                            p = values ** self.sampling_alpha
                            s = p.sum()
                            if s <= 0 or not np.isfinite(s):
                                keep_extra = np.random.choice(cand_idx, keep_budget, replace=False)
                            else:
                                p = p / s
                                keep_extra = np.random.choice(cand_idx, keep_budget,
                                                               replace=False, p=p)
                        else:   # random
                            keep_extra = np.random.choice(cand_idx, keep_budget, replace=False)
                    else:
                        keep_extra = cand_idx
                    keep_set = np.concatenate([forced_keep, keep_extra])
                    drop_mask = np.ones(nz_idx.size, dtype=bool)
                    drop_mask[np.searchsorted(nz_idx, keep_set)] = False
                    hvg[nz_idx[drop_mask]] = 0.0
            post_sampling_nnz = int(np.count_nonzero(hvg))
            out["hvg_seq_len_pre_sampling"] = pre_sampling_nnz
            out["hvg_seq_len_post_sampling"] = post_sampling_nnz
            out["hvg"] = torch.from_numpy(hvg)
        if self.load_raw:
            pf = shard.open_patch()
            pi = int(shard.patch_idx()[spot_idx])
            img = pf["img"][pi]               # (224,224,3) uint8
            out["image_raw"] = torch.from_numpy(img)  # keep uint8; encoder normalises.
        return out


def build_dataset_from_split(prepared_dir: str | Path, sample_ids: Sequence[str],
                             k_spatial: int = 8, load_hvg: bool = True,
                             image_mode: str = "feature",
                             hest_patch_dir: str | Path | None = None,
                             gene_norm_cfg: dict | None = None,
                             tx_only: bool = False,
                             min_seq_len: int = 0,
                             max_seq_len: int = 0,
                             vocab_keep_indices: np.ndarray | None = None,
                             sampling_strategy: str = "random",
                             sampling_alpha: float = 1.0,
                             must_include_mask: np.ndarray | None = None
                             ) -> PairedSpotDataset:
    """Resolve sample ids to shard paths across all sources.
    A sample id can live at one of:
      - <prepared_dir>/<sid>.h5                       (HEST / generic)
      - <prepared_dir>/<sid>.st1k.h5                  (ST1K)
      - <prepared_dir>/<sid>.spatialcorpus.h5         (SpatialCorpus)
      - <prepared_dir>/<sid>.gse176078.h5             (external annotated validation)
      - <prepared_dir>/<sid>.her2st.h5                (HER2ST external validation)
    The first one found (in this priority order) is used.
    """
    base = Path(prepared_dir)
    paths: list[Path] = []
    missing: list[str] = []
    for sid in sample_ids:
        for suffix in ("", ".st1k", ".spatialcorpus", ".gse176078", ".her2st"):
            p = base / f"{sid}{suffix}.h5"
            if p.exists():
                paths.append(p)
                break
        else:
            missing.append(sid)
    if not paths:
        raise FileNotFoundError(f"No prepared shards under {base} for samples {sample_ids[:5]}...")
    if missing:
        # Best-effort warning — don't crash; common case is partial-prep.
        import logging
        logging.getLogger(__name__).warning(
            f"build_dataset_from_split: {len(missing)} sample(s) had no matching shard "
            f"(first 5: {missing[:5]}); {len(paths)} loaded."
        )
    return PairedSpotDataset(paths, k_spatial=k_spatial, load_hvg=load_hvg,
                             image_mode=image_mode, hest_patch_dir=hest_patch_dir,
                             gene_norm_cfg=gene_norm_cfg,
                             tx_only=tx_only,
                             min_seq_len=min_seq_len,
                             max_seq_len=max_seq_len,
                             vocab_keep_indices=vocab_keep_indices,
                             sampling_strategy=sampling_strategy,
                             sampling_alpha=sampling_alpha,
                             must_include_mask=must_include_mask)


def prebuild_knn_caches(dataset: PairedSpotDataset) -> int:
    """Trigger the KNN cache for every shard once (serial, on the calling process)
    so DataLoader workers never race to build them. Safe to call multiple times.
    """
    built = 0
    for shard in dataset.shards:
        if shard._spatial_nn is None:
            shard._build_knn()
            built += 1
    return built
