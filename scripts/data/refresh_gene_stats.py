"""Refresh `gene_stats.npz` in-place — add nonzero_mean / nonzero_std fields
without rebuilding shards.

Reads existing `hvg_log` from shards (which were already log1p-normalized at
prepare time), computes per-gene statistics over the train split, and writes
a richer `gene_stats.npz` that supports the new `nonzero_z` runtime normalization.

This is the fast path for switching to nonzero_z mode without a full reprepare.

Usage:
  PYTHONPATH=src python scripts/data/refresh_gene_stats.py \\
      --prepared-dir results/cache/prepared_expanded \\
      --max-spots-per-sample 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger
log = get_logger("refresh_stats")


def _resolve_shard(prep: Path, sid: str) -> Path | None:
    for suffix in ("", ".st1k", ".spatialcorpus"):
        p = prep / f"{sid}{suffix}.h5"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--max-spots-per-sample", type=int, default=100,
                    help="Cap per-sample subsample to bound memory.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prep = Path(args.prepared_dir)
    splits = json.loads((prep / "splits.json").read_text())
    train_ids = splits["train"]
    vocab = json.loads((prep / "hvg_vocab.json").read_text())
    n_g = len(vocab)

    log.info(f"refreshing gene_stats over {len(train_ids)} train samples (cap "
             f"{args.max_spots_per_sample} spots/sample, n_hvg={n_g})")

    rng = np.random.default_rng(args.seed)
    pooled = []
    src_pooled = []  # source string per row, for batch-effect diagnostics
    for sid in tqdm(train_ids, desc="scan"):
        p = _resolve_shard(prep, sid)
        if p is None:
            continue
        src = ("st1k" if p.stem.endswith(".st1k")
               else "spatialcorpus" if p.stem.endswith(".spatialcorpus")
               else "hest")
        try:
            with h5py.File(p, "r") as f:
                X = f["hvg_log"][:]                # (S, n_hvg) log1p-normalized
        except Exception as e:
            log.warning(f"skip {p.name}: {e}")
            continue
        if X.shape[1] != n_g:
            log.warning(f"skip {p.name}: hvg_dim {X.shape[1]} != vocab {n_g}")
            continue
        n_take = min(args.max_spots_per_sample, X.shape[0])
        sel = rng.choice(X.shape[0], n_take, replace=False)
        pooled.append(X[sel].astype(np.float32))
        src_pooled.append(np.array([src] * n_take))

    pool = np.concatenate(pooled, axis=0)
    src_arr = np.concatenate(src_pooled)
    log.info(f"pool shape = {pool.shape}  (mem ~{pool.nbytes/2**30:.2f} GiB)")

    # Aggregate stats (all values)
    mean = pool.mean(axis=0).astype(np.float32)
    std = pool.std(axis=0).astype(np.float32)
    median = np.median(pool, axis=0).astype(np.float32)
    mad = (np.median(np.abs(pool - median[None, :]), axis=0) * 1.4826).astype(np.float32)

    # Nonzero-only stats (the new bit)
    nonzero_mean = np.zeros(n_g, dtype=np.float32)
    nonzero_std = np.zeros(n_g, dtype=np.float32)
    nonzero_count = np.zeros(n_g, dtype=np.int64)
    for g in range(n_g):
        col = pool[:, g]
        nz = col[col > 0]
        nonzero_count[g] = nz.size
        if nz.size >= 2:
            nonzero_mean[g] = float(nz.mean())
            nonzero_std[g] = float(nz.std())
        elif nz.size == 1:
            nonzero_mean[g] = float(nz[0])
            nonzero_std[g] = 0.0

    # Per-source stats (for batch-effect diagnostic)
    per_src_lib = {}
    for src in np.unique(src_arr):
        m = (src_arr == src)
        per_src_lib[str(src)] = {
            "n_spots_sampled": int(m.sum()),
            "lib_size_mean": float(pool[m].sum(axis=1).mean()),
            "lib_size_median": float(np.median(pool[m].sum(axis=1))),
            "nonzero_frac_mean": float((pool[m] > 0).mean()),
        }
    log.info(f"per-source library stats: {json.dumps(per_src_lib, indent=2)}")

    out = prep / "gene_stats.npz"
    np.savez(out,
             mean=mean, std=std, median=median, mad=mad,
             nonzero_mean=nonzero_mean, nonzero_std=nonzero_std,
             nonzero_count=nonzero_count,
             hvg_vocab=np.array(vocab))
    log.info(f"wrote {out}")
    log.info(f"  median        : {(median==0).sum()}/{n_g} have value=0  (legacy mode unusable)")
    log.info(f"  nonzero_mean  ∈ [{nonzero_mean[nonzero_count>0].min():.3f}, {nonzero_mean.max():.3f}]  "
             f"({(nonzero_count==0).sum()} genes never observed)")
    log.info(f"  nonzero_std   ∈ [{nonzero_std[nonzero_count>=2].min():.3f}, {nonzero_std.max():.3f}]")


if __name__ == "__main__":
    main()
