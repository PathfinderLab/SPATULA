"""Per-shard handle used by `PairedSpotDataset`.

A shard is one sample's prepared `<sid>.h5` file.  This class wraps it with:
    - lazy h5py handle (one per DataLoader worker, never crosses fork)
    - cached patch_idx (barcode → image-patch row), built once per shard
    - cached spatial KNN (rebuilt on disk if missing or corrupt)

Concurrent-safe: KNN cache writes are atomic (mkstemp + os.replace) so
multiple ranks running on the same prepared_dir can't see half-written files.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .splits import decode_bytes_array


class Shard:
    """Handle to one prepared shard + (optionally) the raw-patch H5."""

    def __init__(self, path: Path, k_spatial: int = 8,
                 hest_patch_dir: Path | None = None, load_raw: bool = False):
        self.path = Path(path)
        self.sample_id = self.path.stem
        self.k_spatial = k_spatial
        self.load_raw = load_raw
        self.hest_patch_path = (
            (Path(hest_patch_dir) / f"{self.sample_id}.h5") if hest_patch_dir else None
        )

        # Stats — read once, then close to avoid fd leaks under multiprocessing.
        with h5py.File(self.path, "r") as f:
            self.n_spots = int(f["uni_feat"].shape[0])
            self.uni_dim = int(f["uni_feat"].shape[1])
            self.tx_dim = int(f["novae_latent"].shape[1]) if "novae_latent" in f else 0
            self.hvg_dim = int(f["hvg_log"].shape[1]) if "hvg_log" in f else 0
            self._has_patch_idx = "patch_idx" in f

        # Lazy handles — opened on first read per worker.
        self._h5: Optional[h5py.File] = None
        self._patch_h5: Optional[h5py.File] = None
        self._patch_idx_cache: Optional[np.ndarray] = None
        self._spatial_nn: Optional[np.ndarray] = None

    # ── file handles (lazy per worker) ─────────────────────────────────────
    def open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", libver="latest", swmr=True)
        return self._h5

    def open_patch(self):
        if self._patch_h5 is None:
            assert self.hest_patch_path is not None
            self._patch_h5 = h5py.File(self.hest_patch_path, "r", libver="latest", swmr=True)
        return self._patch_h5

    def close(self):
        if self._h5 is not None:
            self._h5.close(); self._h5 = None
        if self._patch_h5 is not None:
            self._patch_h5.close(); self._patch_h5 = None

    def __getstate__(self):
        # Drop fd-bearing fields so the shard can pickle across DataLoader
        # worker fork without inheriting a stale handle.
        d = self.__dict__.copy()
        d["_h5"] = None; d["_patch_h5"] = None
        d["_patch_idx_cache"] = None; d["_spatial_nn"] = None
        return d

    # ── patch_idx (built / loaded once) ────────────────────────────────────
    def patch_idx(self) -> np.ndarray:
        """Spot index → patch row in the raw UNI patches HDF5."""
        if self._patch_idx_cache is not None:
            return self._patch_idx_cache
        f = self.open()
        if self._has_patch_idx:
            self._patch_idx_cache = f["patch_idx"][:].astype(np.int64)
            return self._patch_idx_cache
        # Fallback: barcode-match against the original patches file.
        bc_shard = decode_bytes_array(f["barcode"][:])
        pf = self.open_patch()
        bc_patch = decode_bytes_array(pf["barcode" if "barcode" in pf else "barcodes"][:])
        idx_map = {b: i for i, b in enumerate(bc_patch)}
        self._patch_idx_cache = np.array([idx_map[b] for b in bc_shard], dtype=np.int64)
        return self._patch_idx_cache

    # ── spatial neighbours ────────────────────────────────────────────────
    def neighbors(self, idx: int) -> np.ndarray:
        if self._spatial_nn is None:
            self._build_knn()
        return self._spatial_nn[idx]

    def _build_knn(self):
        """Build / load the per-shard KNN cache (`<shard>.knn.npy`).

        Concurrency: atomic-replace write (mkstemp + os.replace) so multiple
        DDP ranks reading the same prepared_dir never see a half-written file.

        Degenerate input: ST1K / spatialcorpus shards have zero-padded coords
        (no spatial layout).  Building KNN on zeros is meaningless AND slow
        (every pairwise distance is 0, ranking is arbitrary).  We emit a
        zero-padded cache instead — downstream spatial-aux loss is gated
        on align_weight > 0 anyway, so the dummy never feeds into a real loss.
        """
        cache = self.path.with_suffix(".knn.npy")
        if cache.exists() and cache.stat().st_size > 0:
            try:
                self._spatial_nn = np.load(cache)
                return
            except (EOFError, ValueError, OSError):
                # Corrupt / mid-write — fall through to rebuild.
                pass
        f = self.open()
        coords = f["coords"][:]
        n = len(coords)
        k = min(self.k_spatial + 1, n)
        if n >= 2 and (coords ** 2).sum() == 0.0:
            idx = np.zeros((n, max(0, k - 1)), dtype=np.int32)
        else:
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=k).fit(coords)
            _, idx = knn.kneighbors(coords)
            idx = idx[:, 1:].astype(np.int32)
        # Atomic write — see docstring.
        import os, tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=cache.name + ".", dir=str(cache.parent))
        os.close(tmp_fd)
        try:
            np.save(tmp_path, idx)
            os.replace(tmp_path, cache)
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
        self._spatial_nn = idx
