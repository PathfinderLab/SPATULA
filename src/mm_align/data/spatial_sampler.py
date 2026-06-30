"""Per-sample KNN subgraph yielder for Stage 1.5 Spatial Foundation.

Each "training step" needs:
  - A subgraph of size N (e.g. 256 spots) sampled from one shard
  - The (h_tx, h_img, xy) features for those spots
  - The KNN edge_index restricted to the subgraph

To keep this independent from the Stage-1 pipeline, the FROZEN Stage-1
tx_encoder is invoked here in eval mode on the raw hvg_log of each spot.
For very large shards we cache `h_tx` to disk next to the shard the first
time it is touched.
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _build_knn(xy: np.ndarray, k: int = 8) -> np.ndarray:
    """Symmetric KNN edge_index (drops self)."""
    n = xy.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    from sklearn.neighbors import NearestNeighbors
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(xy)
    _, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]                                   # drop self
    src = np.repeat(np.arange(n), idx.shape[1])
    dst = idx.reshape(-1)
    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])
    return np.stack([edge_src, edge_dst], axis=0).astype(np.int64)


def _build_radius(xy: np.ndarray, radius: float) -> np.ndarray:
    """Radius graph: all (i, j) pairs with ||xy_i − xy_j|| ≤ radius (excl. self).
    Already symmetric.  For Visium-like spots ~100 µm spacing, radius ≈ 250 µm
    gives an average degree of 6-8 (similar to KNN k=8) but adapts to local
    density (boundary spots get fewer neighbors)."""
    n = xy.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=radius).fit(xy)
    rng = nn.radius_neighbors(xy, return_distance=False)
    src_list, dst_list = [], []
    for i, neigh in enumerate(rng):
        neigh = neigh[neigh != i]                       # drop self
        if neigh.size == 0:
            continue
        src_list.append(np.full(neigh.shape, i))
        dst_list.append(neigh)
    if not src_list:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack(
        [np.concatenate(src_list), np.concatenate(dst_list)], axis=0
    ).astype(np.int64)


def _build_grid(xy: np.ndarray) -> np.ndarray:
    """Visium-lattice adjacency — connect each spot to its ~6 immediate
    hex-grid neighbors.  Inferred from xy by snapping to a regular spacing.
    Approximates the original platform graph without external metadata."""
    n = xy.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    # Use the median nearest-neighbor distance as the lattice spacing.
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=min(2, n)).fit(xy)
    dists, _ = knn.kneighbors(xy)
    spacing = float(np.median(dists[:, 1]))
    if spacing <= 0:
        return _build_knn(xy, k=6)
    # Visium hex-grid: 6 neighbors within ~1.05× spacing.
    return _build_radius(xy, radius=spacing * 1.10)


def build_edges(xy: np.ndarray, *, kind: str = "knn",
                k: int = 8, radius: float = 250.0) -> np.ndarray:
    if kind == "knn":
        return _build_knn(xy, k=k)
    if kind == "radius":
        return _build_radius(xy, radius=radius)
    if kind == "grid":
        return _build_grid(xy)
    raise ValueError(f"unknown graph kind: {kind!r}")


class SpatialSampleDataset(Dataset):
    """One item = ONE subgraph from ONE shard.

    On __init__ we precompute h_tx for every shard by running the frozen
    Stage-1 tx_encoder on its hvg_log.  Cached to <shard>.h_tx.npy.

    Sampling a __getitem__:
      1. pick shard `i`
      2. pick `subgraph_size` random spots (or all if shard is smaller)
      3. fetch h_tx[spots], h_img[spots], xy[spots]
      4. build KNN graph restricted to those spots
      5. return torch tensors
    """

    def __init__(self, shards: list[str | Path], tx_encoder,
                 *, k: int = 8, subgraph_size: int = 256,
                 device: str = "cpu", encode_batch: int = 1024,
                 fuse_image: bool = True,
                 tx_dim: int | None = None,
                 graph_kind: str = "knn",
                 radius_px: float = 600.0,
                 # Stage-1 normalisation reuse — pass the SAME gene_norm cfg
                 # that the Stage-1 tx_encoder was trained with (read it from
                 # the ckpt's `cfg_tx["data"]["gene_norm"]`).  Without this,
                 # the frozen encoder sees a distribution it didn't train on.
                 gene_norm_cfg: dict | None = None,
                 vocab_keep_indices: np.ndarray | None = None,
                 # Subgraph sampling strategy:
                 #   random — random subset within shard, KNN rebuilt over subset
                 #             (cheap; ignores true sample-level neighbourhood
                 #             — two faraway spots in the subset can become
                 #             nearest neighbours).
                 #   ego    — sample-level KNN cache + BFS ego subgraph from a
                 #             random centre.  Reflects true tissue locality;
                 #             matches HyperST / ST-JEPA neighbourhood usage.
                 subgraph_kind: str = "random",
                 # Region aggregation knobs (Stage 1.5 "spot + neighbors" → region)
                 region_enable: bool = True,
                 region_k: int | None = None,
                 region_tx_agg: str = "mean",
                 region_img_pool: str = "mean",
                 region_weighted_sigma: float = 1.0,
                 tx_cache_tag: str | None = None,
                 # JEPA contract: region is VISIBLE context for the masked spot,
                 # so the anchor's own RNA/image must NOT leak into its region.
                 # Stay False unless you're in fused-token mode and don't mind.
                 region_include_anchor: bool = False):
        self.shards = [Path(p) for p in shards]
        self.k = int(k)
        self.subgraph_size = int(subgraph_size)
        self.fuse_image = bool(fuse_image)
        self.graph_kind = str(graph_kind)
        self.radius_px = float(radius_px)
        self.subgraph_kind = str(subgraph_kind)
        assert self.graph_kind in ("knn", "radius", "grid")
        assert self.subgraph_kind in ("random", "ego")
        # Lazy-built per-shard sample-level KNN (only populated for ego mode).
        self._full_knn: dict[int, np.ndarray] = {}
        # Region: by default k_region = k.  region_enable=False reverts to the
        # single-token-per-spot behaviour (spot_tx [+ spot_img] only).
        self.region_enable = bool(region_enable)
        self.region_k = int(region_k if region_k is not None else k)
        self.region_tx_agg = str(region_tx_agg)
        self.region_img_pool = str(region_img_pool)
        self.region_weighted_sigma = float(region_weighted_sigma)
        self.tx_cache_tag = str(tx_cache_tag or "")
        self.region_include_anchor = bool(region_include_anchor)
        assert self.region_tx_agg in ("mean", "sum_log1p", "weighted")
        assert self.region_img_pool in ("mean", "attn")
        # Encoder output dim — try (out_dim, dim, embed_dim) in that order;
        # fall back to caller-provided tx_dim.
        self._tx_dim = (tx_dim
                         or getattr(tx_encoder, "out_dim", None)
                         or getattr(tx_encoder, "dim", None)
                         or getattr(tx_encoder, "embed_dim", None))
        if self._tx_dim is None:
            raise RuntimeError(
                "Cannot infer tx_encoder output dim; pass tx_dim=N explicitly.")
        # Build the gene normaliser ONCE (shared by _encode_shard and the
        # region path).  Stats are sliced to vocab_keep_indices if given.
        # We need a shard's hvg_log width to validate stats; peek at the first.
        from .gene_norm import GeneNormalizer
        with h5py.File(self.shards[0], "r") as _f0:
            _full_dim = int(_f0["hvg_log"].shape[1])
        _hvg_dim = (int(len(vocab_keep_indices)) if vocab_keep_indices is not None
                     else _full_dim)
        self.gene_norm = GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=_full_dim, hvg_dim=_hvg_dim,
            vocab_keep_indices=vocab_keep_indices,
        )
        self.vocab_keep_indices = vocab_keep_indices
        if self.gene_norm:
            print(f"[spatial_sampler] gene_norm.mode={self.gene_norm.mode}, "
                  f"clip={self.gene_norm.clip}, "
                  f"zero_preserve={self.gene_norm.zero_preserve}")

        # h_tx cache lives PER-NORMALIZER and PER-CHECKPOINT — invalidate when
        # normalisation or frozen tx_encoder checkpoint changes.  Stage 1.25
        # refinement can keep the same gene_norm/vocab as Stage 1 while changing
        # h_tx, so a normalizer-only cache name would silently reuse stale
        # embeddings.
        norm_tag = "none" if not self.gene_norm else self.gene_norm.mode
        ckpt_tag = f"_{self.tx_cache_tag}" if self.tx_cache_tag else ""
        cache_suffix = f".htx_{norm_tag}{ckpt_tag}.npy"
        self._h_tx_paths: list[Path] = []
        self._n_spots: list[int] = []
        for sp in self.shards:
            cache = sp.with_suffix(cache_suffix)
            self._h_tx_paths.append(cache)
            with h5py.File(sp, "r") as f:
                self._n_spots.append(int(f["hvg_log"].shape[0]))
            if not cache.exists():
                self._encode_shard(sp, cache, tx_encoder, device, encode_batch)

    @torch.no_grad()
    def _encode_shard(self, shard_path: Path, cache_path: Path,
                      tx_encoder, device: str, batch: int):
        """Pre-encode every spot's hvg_log into h_tx, applying the Stage-1
        gene_norm transform so the frozen tx_encoder sees the same input
        distribution it was trained on."""
        tx_encoder.eval().to(device)
        with h5py.File(shard_path, "r") as f:
            hvg = f["hvg_log"]
            n = hvg.shape[0]
            out = np.zeros((n, self._tx_dim), dtype=np.float32)
            for r0 in range(0, n, batch):
                r1 = min(r0 + batch, n)
                raw = hvg[r0:r1].astype(np.float32)
                if self.vocab_keep_indices is not None:
                    raw = raw[:, self.vocab_keep_indices]
                normed = self.gene_norm.apply_np(raw)
                block = torch.from_numpy(normed).to(device)
                # tx_encoder forward returns {h_tx: (B, D), ...}.
                out_blk = tx_encoder(novae_latent=None, hvg=block)["h_tx"]
                out[r0:r1] = out_blk.detach().cpu().numpy().astype(np.float32)
        np.save(cache_path, out)

    def __len__(self) -> int:
        return len(self.shards)

    def _get_full_knn(self, idx: int) -> np.ndarray:
        """Lazy sample-level KNN cache.  (n_full, k) int64.  On-disk under
        <shard>.knn_k{k}.npz so the cost is one-shot per shard."""
        if idx in self._full_knn:
            return self._full_knn[idx]
        sp = self.shards[idx]
        cache = sp.with_suffix(f".knn_k{self.k}.npz")
        if cache.exists():
            arr = np.load(cache)["indices"]
        else:
            with h5py.File(sp, "r") as f:
                xy_full = f["coords"][:].astype(np.float32)
            n_full = xy_full.shape[0]
            from sklearn.neighbors import NearestNeighbors
            kk = min(self.k + 1, n_full)
            nn = NearestNeighbors(n_neighbors=kk).fit(xy_full)
            _, ix = nn.kneighbors(xy_full)
            arr = ix[:, 1:].astype(np.int64)
            try:
                np.savez(cache, indices=arr)
            except OSError:
                pass         # read-only fs is fine; we'll rebuild next time
        self._full_knn[idx] = arr
        return arr

    def _ego_select(self, idx: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """BFS ego subgraph from a random centre.

        Returns (sel_sorted, edge_index_sub):
            sel_sorted : (M,) original spot ids, ascending
            edge_index_sub : (2, E) symmetric edges, indexed into sel_sorted
        """
        full_knn = self._get_full_knn(idx)              # (n_full, k)
        n_full = full_knn.shape[0]
        if n_full <= self.subgraph_size:
            sel = np.arange(n_full, dtype=np.int64)
        else:
            centre = int(rng.integers(n_full))
            visited = {centre}
            queue = [centre]
            target = self.subgraph_size
            head = 0
            while head < len(queue) and len(visited) < target:
                cur = queue[head]; head += 1
                # Randomise neighbour order so ties get diversified.
                nbs = full_knn[cur].tolist()
                rng.shuffle(nbs)
                for nb in nbs:
                    nb = int(nb)
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
                        if len(visited) >= target:
                            break
            sel = np.sort(np.fromiter(visited, dtype=np.int64))
        # Build edges by restricting full KNN to sel.
        sel_set = {int(s): i for i, s in enumerate(sel)}
        src_list, dst_list = [], []
        for i_new, old in enumerate(sel):
            for nb in full_knn[int(old)]:
                j = sel_set.get(int(nb))
                if j is not None:
                    src_list.append(i_new); dst_list.append(j)
        if src_list:
            src = np.asarray(src_list, dtype=np.int64)
            dst = np.asarray(dst_list, dtype=np.int64)
            edge_index = np.stack(
                [np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        return sel, edge_index

    def __getitem__(self, idx: int) -> dict:
        sp = self.shards[idx]
        cache = self._h_tx_paths[idx]
        n = self._n_spots[idx]
        rng = np.random.default_rng()

        # ── 1. Pick the subgraph spots & their internal edges ──────────────
        if self.subgraph_kind == "ego":
            sel, edge_index_pre = self._ego_select(idx, rng)
        else:
            if n > self.subgraph_size:
                sel = np.sort(rng.choice(n, self.subgraph_size, replace=False))
            else:
                sel = np.arange(n)
            edge_index_pre = None        # build over local xy below

        h_tx = np.load(cache, mmap_mode="r")[sel].astype(np.float32)
        with h5py.File(sp, "r") as f:
            xy = f["coords"][sel].astype(np.float32)
            h_img = f["uni_feat"][sel].astype(np.float32) if self.fuse_image else None
            # We need RAW log1p for region aggregation (we'll normalise AFTER
            # aggregating so the frozen Stage-1 tx_encoder sees its training
            # distribution — same gene_norm pipeline applied to anchors).
            hvg = (f["hvg_log"][sel].astype(np.float32)
                   if (self.region_enable and "hvg_log" in f) else None)
            if hvg is not None and self.vocab_keep_indices is not None:
                hvg = hvg[:, self.vocab_keep_indices]
        # Per-sample centring & scale → translation-invariant pos enc.
        xy = xy - xy.mean(axis=0, keepdims=True)
        scale = max(1e-6, float(np.linalg.norm(xy, axis=1).max()))
        xy = xy / scale
        # Edge index: ego mode already has sample-level KNN restricted to sel;
        # random mode rebuilds over the sampled subset (legacy behaviour).
        if edge_index_pre is None:
            eff_radius = self.radius_px / max(scale, 1e-6)
            edge_index = build_edges(xy, kind=self.graph_kind,
                                      k=self.k, radius=eff_radius)
        else:
            edge_index = edge_index_pre
        out = {
            "h_tx": torch.from_numpy(h_tx),
            "xy": torch.from_numpy(xy),
            "edge_index": torch.from_numpy(edge_index),
        }
        if h_img is not None:
            out["h_img"] = torch.from_numpy(h_img)

        # ── Region-level tokens (Stage 1.5 "anchor + neighbors") ──────────
        if self.region_enable:
            from .region import (
                build_neighbor_index, aggregate_region_tx, aggregate_region_img,
            )
            # pad_with_self mirrors include_anchor: when anchor is excluded
            # from aggregation (JEPA semantics), padding the deficit with the
            # anchor itself would re-introduce the leak.  We instead pad with
            # -1 and aggregate ignores those slots.
            nbr_idx = build_neighbor_index(edge_index, n_nodes=len(sel),
                                            k=self.region_k,
                                            pad_with_self=self.region_include_anchor)
            # region_img: pool on UNI features (cheap, no UNI re-forward).
            if h_img is not None:
                r_img = aggregate_region_img(
                    h_img, nbr_idx, kind=self.region_img_pool,
                    include_anchor=self.region_include_anchor)
                out["h_region_img"] = torch.from_numpy(r_img)
            # region_tx pipeline (matches the anchor pipeline exactly):
            #   1. aggregate neighbours' RAW log1p (mean / sum_log1p / weighted)
            #   2. apply the SAME gene_norm transform Stage-1 saw at train time
            #   3. trainer pushes the result through the frozen Stage-1
            #      tx_encoder to produce `h_region_tx` (in the same latent
            #      space as `h_tx`).
            # Anchor inclusion: gated by region_include_anchor.  In the JEPA
            # main-candidate (mask spot, see region) this MUST be False or
            # anchor RNA/image leaks into the visible context.
            if hvg is not None:
                if self.region_tx_agg == "weighted":
                    # Distances in normalised xy units.
                    d = np.linalg.norm(
                        xy[nbr_idx] - xy[:, None, :], axis=-1
                    ).astype(np.float32)
                    r_tx_agg = aggregate_region_tx(
                        hvg, nbr_idx, kind="weighted",
                        distances=d, sigma=self.region_weighted_sigma,
                        include_anchor=self.region_include_anchor,
                    )
                else:
                    r_tx_agg = aggregate_region_tx(
                        hvg, nbr_idx, kind=self.region_tx_agg,
                        include_anchor=self.region_include_anchor,
                    )
                # Apply Stage-1 gene_norm (no-op when gene_norm.mode='none').
                r_tx_norm = self.gene_norm.apply_np(r_tx_agg)
                out["region_hvg"] = torch.from_numpy(r_tx_norm)
            out["neighbor_index"] = torch.from_numpy(nbr_idx)
        return out


def spatial_collate(items: list[dict]) -> dict:
    """Concatenate subgraphs back-to-back, with proper offsetting of every
    index-bearing tensor (edge_index, neighbor_index).

    The collated batch is one big graph composed of B disconnected
    subgraphs; Spatial JEPA fires per-subgraph so disconnection is correct.
    """
    h_tx = torch.cat([it["h_tx"] for it in items], dim=0)
    xy = torch.cat([it["xy"] for it in items], dim=0)
    has_img = "h_img" in items[0]
    h_img = torch.cat([it["h_img"] for it in items], dim=0) if has_img else None
    offsets, off = [], 0
    for it in items:
        offsets.append(off); off += it["h_tx"].shape[0]
    edges = []
    for it, o in zip(items, offsets):
        edges.append(it["edge_index"] + o)
    edge_index = torch.cat(edges, dim=1)
    out = {"h_tx": h_tx, "h_img": h_img, "xy": xy, "edge_index": edge_index}

    # Region tokens (optional — only present when SpatialSampleDataset has
    # region_enable=True).  neighbor_index also needs offsetting since each
    # entry is a node ID into the per-subgraph node list.
    if "h_region_img" in items[0]:
        out["h_region_img"] = torch.cat([it["h_region_img"] for it in items], dim=0)
    if "region_hvg" in items[0]:
        out["region_hvg"] = torch.cat([it["region_hvg"] for it in items], dim=0)
    if "neighbor_index" in items[0]:
        offset_nbr = []
        for it, o in zip(items, offsets):
            ni = it["neighbor_index"].clone()
            ni[ni >= 0] += o                # -1 entries (no-neighbor pad) stay -1
            offset_nbr.append(ni)
        out["neighbor_index"] = torch.cat(offset_nbr, dim=0)
    # Per-node subgraph id — needed by spatial JEPA for "mask whole anchor cell"
    # within a subgraph (so masking doesn't leak across subgraphs).
    out["subgraph_id"] = torch.cat([
        torch.full((it["h_tx"].shape[0],), i, dtype=torch.long)
        for i, it in enumerate(items)
    ], dim=0)
    return out
