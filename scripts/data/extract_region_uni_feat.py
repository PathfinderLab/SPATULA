"""Extract region-level UNI features (Stage 1.5 Phase-2 image branch).

The default Stage 1.5 region_img uses **mean-pool of neighbour `uni_feat`** —
cheap and modality-faithful, but throws away spatial structure within the
region.  This script computes a stronger alternative: for each anchor spot,
**stitch the anchor + its 8 nearest spot-level patches into a 3×3 grid,
resize to 224×224, and run UNI forward once**.  That preserves the visible
spatial layout (which spot is to the north / southeast / …) inside one
1536-d region embedding.

Layout decision: anchor is fixed at grid position (1, 1); the 8 neighbours
are placed by angle around the anchor:

        NW (0,0)   N (0,1)   NE (0,2)
         W (1,0)  anchor    E (1,2)
        SW (2,0)   S (2,1)   SE (2,2)

For anchors with fewer than 8 neighbours, missing cells stay zero-padded.
Anchors with NO neighbours fall back to the anchor patch repeated (effectively
a 224×224 → 672×672 → 224×224 round-trip), so we still emit a valid feature.

Inputs (per shard):
  shard.h5         (HEST):  /coords, /barcode, /patch_idx
  /data/hest/patches/{sid}.h5:  /img   (n, 224, 224, 3) uint8

Output:
  <shard>.region_uni.npy   (n_spots, 1536) float32

HEST shards only — ST1K / spatialcorpus shards have no `uni_feat` source,
so they keep the mean-pool path.

Usage:
    PYTHONPATH=src python scripts/data/extract_region_uni_feat.py \
        --prepared-dir results/cache/prepared_expanded \
        --hest-patch-dir /data/hest/patches \
        --uni-weights assets/uni2-h.bin \
        [--k 8]  [--limit 50]  [--shards INT1 INT2 ...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger
from mm_align.models.image.uni import build_uni

log = get_logger("extract_region_uni")


# ────────────────────────────────────────────────────────────────────────────
# Grid placement: angle around anchor → 3×3 cell.
# ────────────────────────────────────────────────────────────────────────────
# 8 octants, mapped to 8 grid cells (anchor at (1,1)).  atan2 returns [-π, π];
# we shift so 0=east, then quantise into 8 buckets.
_OCTANT_TO_GRID = [
    (1, 2),  # E    (-22.5°..  22.5°)
    (0, 2),  # NE   ( 22.5°..  67.5°)
    (0, 1),  # N
    (0, 0),  # NW
    (1, 0),  # W
    (2, 0),  # SW
    (2, 1),  # S
    (2, 2),  # SE
]


def _place_neighbors_by_angle(anchor_xy: np.ndarray, neigh_xy: np.ndarray,
                                neigh_idx: np.ndarray) -> dict[tuple[int, int], int]:
    """Map each neighbour to a grid cell (anchor at (1,1)).

    If two neighbours fall into the same octant the closer one wins.  Empty
    cells stay un-assigned (caller zero-pads).
    """
    if neigh_xy.shape[0] == 0:
        return {}
    delta = neigh_xy - anchor_xy
    dist = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(-delta[:, 1], delta[:, 0])         # screen y-flip
    octant = ((angle + np.pi / 8) // (np.pi / 4)).astype(int) % 8
    out: dict[tuple[int, int], tuple[float, int]] = {}
    for i, oc in enumerate(octant):
        cell = _OCTANT_TO_GRID[int(oc)]
        if cell not in out or dist[i] < out[cell][0]:
            out[cell] = (float(dist[i]), int(neigh_idx[i]))
    return {cell: idx for cell, (_, idx) in out.items()}


# ────────────────────────────────────────────────────────────────────────────
# Per-shard processing
# ────────────────────────────────────────────────────────────────────────────

def _knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """Return (n, k) indices of nearest neighbours (excluding self).
    Zero-coord (placeholder) rows get a self-padded row."""
    n = coords.shape[0]
    if n <= 1:
        return np.zeros((n, k), dtype=np.int64)
    from sklearn.neighbors import NearestNeighbors
    if (coords ** 2).sum() == 0.0:
        # ST1K / spatialcorpus stub — we won't touch these shards, but be safe.
        return np.tile(np.arange(n)[:, None], (1, k))
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:].astype(np.int64)


def _stitch_3x3(patches: np.ndarray,
                  grid: dict[tuple[int, int], int],
                  anchor_patch: np.ndarray) -> np.ndarray:
    """Assemble a 672×672 RGB image from 9 cells.  Missing cells = zeros.
    anchor at (1, 1)."""
    big = np.zeros((672, 672, 3), dtype=np.uint8)
    big[224:448, 224:448] = anchor_patch
    for (r, c), patch_idx in grid.items():
        big[r * 224:(r + 1) * 224, c * 224:(c + 1) * 224] = patches[patch_idx]
    return big


@torch.no_grad()
def _process_shard(shard_path: Path, hest_patch_dir: Path, k: int,
                    uni, transform, device, batch_size: int = 32,
                    overwrite: bool = False) -> bool:
    """Returns True if a new cache was written."""
    cache = shard_path.with_suffix(".region_uni.npy")
    if cache.exists() and not overwrite:
        return False
    # Source of raw patches: HEST patches H5 keyed by sample_id.
    sid = shard_path.stem
    raw_path = hest_patch_dir / f"{sid}.h5"
    if not raw_path.exists():
        return False        # ST1K / spatialcorpus — skip silently
    with h5py.File(shard_path, "r") as fs, h5py.File(raw_path, "r") as fp:
        coords = fs["coords"][:]
        patch_idx = fs["patch_idx"][:]               # (n_shard,) row in raw patches
        n = coords.shape[0]
        # If the shard's coords are zero-padded (non-HEST), the KNN is degenerate.
        if (coords ** 2).sum() == 0.0:
            return False
        # NB: raw_patches indexed by patch_idx → spot patch.
        all_patches = fp["img"]                      # h5 dataset, lazy
        nbr = _knn_indices(coords, k)                # (n, k)

        out = np.zeros((n, 1536), dtype=np.float32)
        pbar = tqdm(range(0, n, batch_size), desc=sid, leave=False)
        for r0 in pbar:
            r1 = min(r0 + batch_size, n)
            big_imgs: list[np.ndarray] = []
            for i in range(r0, r1):
                anchor_pix = all_patches[int(patch_idx[i])]      # (224, 224, 3)
                nbr_rows_in_patches = patch_idx[nbr[i]]          # (k,)
                nbr_patches = all_patches[
                    np.asarray(sorted(set(nbr_rows_in_patches.tolist())))
                ] if len(nbr_rows_in_patches) else np.zeros((0, 224, 224, 3), dtype=np.uint8)
                # Map back from sorted unique → original order by lookup table.
                lut = {int(v): j for j, v in enumerate(
                    np.asarray(sorted(set(nbr_rows_in_patches.tolist()))))}
                ordered_nbr = np.stack([nbr_patches[lut[int(v)]] for v in nbr_rows_in_patches]) \
                              if len(nbr_rows_in_patches) else np.zeros((0, 224, 224, 3), dtype=np.uint8)
                grid = _place_neighbors_by_angle(coords[i], coords[nbr[i]],
                                                  np.arange(len(nbr[i])))
                big = _stitch_3x3(ordered_nbr, grid, anchor_pix)
                big_imgs.append(big)

            # 672 → 224 resize via PIL (UNI's transform expects 224 input).
            tensors = [
                transform(Image.fromarray(big).resize((224, 224), Image.BILINEAR))
                for big in big_imgs
            ]
            x = torch.stack(tensors, dim=0).to(device)
            feat = uni(x).detach().cpu().numpy().astype(np.float32)   # (B, 1536)
            out[r0:r1] = feat
    # Atomic write
    import os, tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=cache.name + ".", dir=str(cache.parent))
    os.close(tmp_fd)
    try:
        np.save(tmp_path, out)
        os.replace(tmp_path, cache)
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
    return True


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--hest-patch-dir", default="/data/hest/patches")
    ap.add_argument("--uni-weights", default="assets/uni2-h.bin")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                     help="Process only the first N shards (smoke / partial runs).")
    ap.add_argument("--shards", nargs="*", default=None,
                     help="Optional explicit list of shard stems (e.g. INT1 MEND2).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = Path(args.prepared_dir)
    hest_patch_dir = Path(args.hest_patch_dir)

    log.info(f"loading UNI from {args.uni_weights}")
    uni, transform, *_ = build_uni(args.uni_weights, device=device)
    uni.eval()
    for p in uni.parameters():
        p.requires_grad_(False)

    # Pick HEST shards (no .st1k / .spatialcorpus suffix).
    shards = sorted([s for s in prep.glob("*.h5")
                     if ".st1k." not in s.name and ".spatialcorpus." not in s.name])
    if args.shards:
        wanted = set(args.shards)
        shards = [s for s in shards if s.stem in wanted]
    if args.limit:
        shards = shards[: args.limit]
    log.info(f"processing {len(shards)} HEST shards (k={args.k})")

    n_built, n_skipped = 0, 0
    for s in tqdm(shards, desc="region_uni"):
        try:
            built = _process_shard(s, hest_patch_dir, args.k, uni, transform,
                                     device, batch_size=args.batch_size,
                                     overwrite=args.overwrite)
            if built:
                n_built += 1
            else:
                n_skipped += 1
        except Exception as e:
            log.warning(f"{s.stem}: {type(e).__name__}: {e}")
            n_skipped += 1
    log.info(f"done — built {n_built}, skipped {n_skipped}")


if __name__ == "__main__":
    main()
