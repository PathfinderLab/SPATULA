"""Visualize the two-stage normalization (spot-level vs gene-level) clearly.

Background: SPATULA paper (page 11) prescribes two-stage normalization:
  1. Spot-level   — sc.pp.normalize_total(target_sum=1e4) + log1p
                    (removes per-spot sequencing-depth differences)
  2. Gene-level   — divide by global median expression (per-gene)
                    (removes per-gene baseline magnitude → relative variation)

Our shards store the result of **stage 1 only** (hvg_log).  Stage 2 is applied
at dataloader-time when `data.gene_norm.mode != none`.

This script samples a few spots and renders FOUR informative figures:

  fig_norm_perspot_density.png
    For 6 random spots: distribution of NON-ZERO HVG values
      panel A: raw counts (loaded from h5ad via st1k/hest source)
      panel B: after spot-level normalize_total(1e4) + log1p (= hvg_log)
      panel C: after gene-level ÷ global_median (= what gene_norm does)
    Shows how each stage flattens the per-spot distribution.

  fig_norm_pergene_baseline.png
    For 10 representative genes (high/mid/low baseline expression):
    box-plot of values across spots before vs after gene-level norm.
    Shows that gene-level norm centers each gene at ~1.0.

  fig_norm_lib_size.png
    Per-spot library size (sum of HVG values) before vs after spot-norm.
    Spot-norm should flatten this to ~constant.

  fig_norm_gene_baseline_distribution.png
    Histogram of global per-gene MEDIAN before gene-level norm.
    Shows what magnitudes the gene-level step is normalising away.

Usage:
    PYTHONPATH=src python scripts/viz/normalization.py \\
        --prepared-dir results/cache/prepared_expanded \\
        --n-spots 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for
log = get_logger("viz_norm")


# ---------------------------------------------------------------------- helpers

def pick_diverse_spots(prep: Path, n: int = 6) -> list[tuple[str, int]]:
    """Pick `n` spots from shards spanning HEST / ST1K (if available).
    Returns list of (shard_path, spot_idx)."""
    hest = sorted([p for p in prep.glob("*.h5") if not p.stem.endswith((".st1k", ".spatialcorpus"))])
    st1k = sorted(prep.glob("*.st1k.h5"))
    out = []
    for src_list in (hest, st1k):
        # Take 1 shard, 3 spots from it (to keep figure compact)
        if not src_list:
            continue
        p = src_list[len(src_list) // 2]
        with h5py.File(p, "r") as f:
            S = f["hvg_log"].shape[0]
        rng = np.random.default_rng(0)
        spot_ids = rng.choice(S, min(n // 2, S), replace=False)
        for s in spot_ids:
            out.append((str(p), int(s)))
    return out[:n]


def load_spot(shard: str, spot: int):
    with h5py.File(shard, "r") as f:
        x = f["hvg_log"][spot]
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------- figures

def fig_perspot_density(prep: Path, gene_stats: dict, out_path: Path,
                        n_spots: int = 6):
    """Per-spot density of HVG expression at three normalization stages.
    Note: 'raw counts' here = log1p inverse (expm1) — approximation only,
    since we don't have raw integer counts in the shard."""
    median = gene_stats["median"].astype(np.float32)
    spots = pick_diverse_spots(prep, n_spots)
    fig, axes = plt.subplots(3, len(spots), figsize=(3 * len(spots), 9), squeeze=False)
    for j, (shard, idx) in enumerate(spots):
        x = load_spot(shard, idx)              # stored = log1p of normalize_total(1e4)
        nz = x > 0
        # Approximate "raw count":  invert sc.pp.normalize_total(1e4)+log1p IS
        # ambiguous, but expm1(x) gives the normalized fractional count (so
        # large values are "high relative expression").  We label it
        # appropriately.
        raw_approx = np.expm1(x)               # back to normalized-count scale
        gene_normed = x / (median + 1e-6)

        for row, (data, title, color) in enumerate([
            (raw_approx[nz], "stage-A  expm1(hvg_log)\n(=  normalized-count, pre log)", "#888888"),
            (x[nz],          "stage-B  hvg_log\n(spot-level: sum→1e4 + log1p — stored)", "#3a82e0"),
            (gene_normed[nz],"stage-C  ÷ global_median\n(gene-level baseline removal)", "#e07a3a"),
        ]):
            ax = axes[row, j]
            if data.size > 0:
                ax.hist(data, bins=30, color=color, alpha=0.85)
                ax.axvline(np.median(data), color="black", linestyle="--", alpha=0.4,
                           label=f"median={np.median(data):.2f}")
                ax.legend(fontsize=7)
            ax.set_title(f"{Path(shard).stem}\nspot {idx}\n{title}", fontsize=8)
            ax.set_xlabel("value (non-zero only)", fontsize=8)
            if row == 2:
                ax.set_yscale("log")
    fig.suptitle("Per-spot HVG-value distribution — normalization pipeline (3 stages)\n"
                 "rows: A=normalized-count (pre-log)  B=spot-level (stored)  C=gene-level (÷ median)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_pergene_baseline(prep: Path, gene_stats: dict, vocab: list[str],
                          out_path: Path, n_genes: int = 10, n_spots: int = 5000):
    """For 10 genes spanning the median spectrum, show per-gene distribution
    before and after gene-level normalization."""
    median = gene_stats["median"].astype(np.float32)
    # Pick 10 genes: 3 with highest median, 4 mid, 3 low (but > 0)
    order = np.argsort(median)
    nz_idx = order[median[order] > 0]
    if len(nz_idx) >= n_genes:
        chosen = list(nz_idx[-3:]) + list(nz_idx[len(nz_idx)//2 - 2: len(nz_idx)//2 + 2]) + list(nz_idx[:3])
    else:
        chosen = list(nz_idx)
    chosen = sorted(set(chosen))[:n_genes]
    gene_names = [vocab[i] for i in chosen]
    log.info(f"chosen genes: {gene_names}")

    # Sample n_spots from a HEST shard.
    hest = sorted([p for p in prep.glob("*.h5")
                   if not p.stem.endswith((".st1k", ".spatialcorpus"))])
    pool = []
    for p in hest[:5]:
        with h5py.File(p, "r") as f:
            X = f["hvg_log"][:]
        pool.append(X)
        if sum(p.shape[0] for p in pool) >= n_spots:
            break
    X_pool = np.concatenate(pool, axis=0)[:n_spots]                 # (N, n_hvg)
    sub_raw = X_pool[:, chosen]                                     # (N, K)
    sub_norm = sub_raw / (median[chosen][None, :] + 1e-6)           # (N, K)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Box plots — pre vs post
    axes[0].boxplot([sub_raw[:, k][sub_raw[:, k] > 0] for k in range(len(chosen))],
                    labels=gene_names, showfliers=False)
    axes[0].set_title("Before gene-level norm  (stored hvg_log; non-zero only)")
    axes[0].set_ylabel("log1p value")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].boxplot([sub_norm[:, k][sub_raw[:, k] > 0] for k in range(len(chosen))],
                    labels=gene_names, showfliers=False)
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="ratio=1 (median)")
    axes[1].set_title("After ÷ global_median  (gene-level normalized)")
    axes[1].set_ylabel("normalized ratio")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()
    fig.suptitle("Per-gene distribution before & after gene-level normalization\n"
                 "genes selected to span low / mid / high median expression",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_lib_size(prep: Path, out_path: Path, n_spots: int = 8000):
    """Per-spot library size — sum of HVG values per spot, before normalization
    (raw expm1) and after (stored hvg_log).  This tests whether spot-level
    normalization actually flattened seq-depth differences."""
    # Pick 1 HEST + 1 ST1K + 1 spc shard
    shards = []
    for pat in ("*.h5", "*.st1k.h5", "*.spatialcorpus.h5"):
        cand = sorted(prep.glob(pat))
        # filter out the .st1k.h5 from the "*.h5" glob
        if pat == "*.h5":
            cand = [c for c in cand if not c.stem.endswith((".st1k", ".spatialcorpus"))]
        if cand:
            shards.append(cand[len(cand)//2])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    cmap = {"hest": "#3a82e0", "st1k": "#e07a3a", "spatialcorpus": "#7e3aff"}
    for p in shards:
        src = "spatialcorpus" if p.stem.endswith(".spatialcorpus") else \
              "st1k" if p.stem.endswith(".st1k") else "hest"
        with h5py.File(p, "r") as f:
            X = f["hvg_log"][:n_spots]
        raw_approx = np.expm1(X).sum(axis=1)
        normed = X.sum(axis=1)
        axes[0].hist(raw_approx, bins=80, alpha=0.55, label=f"{src} (n={X.shape[0]})",
                     color=cmap.get(src, "gray"))
        axes[1].hist(normed, bins=80, alpha=0.55, label=f"{src} (n={X.shape[0]})",
                     color=cmap.get(src, "gray"))
    axes[0].set_title("Pre spot-norm (≈ expm1 of stored hvg_log)\nlibrary size = Σ_genes raw value")
    axes[1].set_title("Post spot-norm (= stored hvg_log)\nlibrary size = Σ_genes log1p value")
    axes[0].set_xlabel("Σ value per spot"); axes[0].set_xscale("log")
    axes[1].set_xlabel("Σ value per spot")
    for ax in axes:
        ax.set_ylabel("# spots"); ax.legend()
    fig.suptitle("Per-spot library size — confirms spot-level normalization flattens depth variation",
                 y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_gene_baseline_distribution(gene_stats: dict, vocab: list[str], out_path: Path):
    """Histogram of per-gene global median — what the gene-level step normalises away."""
    median = gene_stats["median"].astype(np.float32)
    mean = gene_stats["mean"].astype(np.float32)
    std = gene_stats["std"].astype(np.float32)
    mad = gene_stats["mad"].astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, vals, label, color in [
        (axes[0,0], median, "median(log1p hvg) per gene", "#3a82e0"),
        (axes[0,1], mean,   "mean(log1p hvg)   per gene", "#e07a3a"),
        (axes[1,0], std,    "std(log1p hvg)    per gene", "#7e3aff"),
        (axes[1,1], mad,    "MAD(log1p hvg)    per gene", "#2a8f3a"),
    ]:
        ax.hist(vals, bins=80, color=color, alpha=0.85)
        ax.set_title(f"{label}\n(over training pool)")
        ax.set_xlabel("value"); ax.set_ylabel("# HVG")
        ax.axvline(np.median(vals), color="black", linestyle="--", alpha=0.5,
                   label=f"global median = {np.median(vals):.3f}")
        ax.legend()
    # Annotate the zero-inflation
    zero_med = int((median == 0).sum())
    axes[0,0].text(0.55, 0.85,
        f"{zero_med}/{len(median)} genes\nhave median=0\n(zero-inflated;\nmin_scale floor needed)",
        transform=axes[0,0].transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    fig.suptitle("Per-gene baseline statistics over the training pool\n"
                 "gene-level normalization = ÷ median (or center by mean)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="/workspace/mm_align/results/cache/prepared_expanded")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-spots", type=int, default=6)
    args = ap.parse_args()

    prep = Path(args.prepared_dir)
    out_dir = Path(args.out_dir or (reports_dir_for(prep) / "eda_v2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"out_dir = {out_dir}")

    vocab = json.loads((prep / "hvg_vocab.json").read_text())
    gs_path = prep / "gene_stats.npz"
    if not gs_path.exists():
        raise SystemExit(f"gene_stats.npz not found at {gs_path}")
    gene_stats = dict(np.load(gs_path))
    log.info(f"loaded gene_stats: keys={list(gene_stats.keys())}, n_hvg={gene_stats['median'].shape[0]}")

    log.info("fig 1/4: per-spot density …")
    fig_perspot_density(prep, gene_stats, out_dir / "fig_norm_perspot_density.png", args.n_spots)
    log.info("fig 2/4: per-gene baseline …")
    fig_pergene_baseline(prep, gene_stats, vocab, out_dir / "fig_norm_pergene_baseline.png")
    log.info("fig 3/4: library size …")
    fig_lib_size(prep, out_dir / "fig_norm_lib_size.png")
    log.info("fig 4/4: gene baseline distribution …")
    fig_gene_baseline_distribution(gene_stats, vocab, out_dir / "fig_norm_gene_baseline_distribution.png")
    log.info(f"4 figures under {out_dir}/")


if __name__ == "__main__":
    main()
