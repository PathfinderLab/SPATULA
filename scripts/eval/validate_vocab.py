"""Validation sanity-checks for a prepared vocab.

Answers two questions:
  1. Does the shard's `hvg_log` actually correspond to the genes named in
     `hvg_vocab.json`?  (regression check on the indexing step in
     `process_sample`.)
  2. What does the post-zero-removal sequence length distribution look like?
     We expect roughly seq_len ≈ vocab_size · mean_spot_prevalence.

Run after a successful prepare:
  PYTHONPATH=src python scripts/eval/validate_vocab.py --prepared-dir <dir>

Writes:
  results/eda/<prepared_name>/validate_vocab/
    seqlen_distribution.png
    vocab_match_audit.csv
    validate_vocab.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for

log = get_logger("validate_vocab")


# ────────────────────────────────────────────────────────────────────────────
# 1. Vocab-match correctness
# ────────────────────────────────────────────────────────────────────────────

def check_vocab_match(prepared: Path, vocab: list[str], n_check: int = 5) -> pd.DataFrame:
    """Re-load `n_check` shards' source AnnData, log-normalize again, and
    compare against the shard's `hvg_log` column-by-column.  Mismatches
    indicate the prepare-time projection step is buggy.

    We can only audit HEST shards (st1k/spatialcorpus readers re-densify, OOM
    risk).  For HEST we have the raw on-disk h5ad at /data/hest/st/{id}.h5ad.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import scanpy as sc
    import anndata as ad
    from mm_align.data.gene_symbols import load_gtf_symbol_map

    # Reuse the exact cleaner / loader the prepare step uses (now proper
    # library code under mm_align.data.gene_cleaning + scripts/data/prepare.py).
    from mm_align.data.gene_cleaning import clean_adata_var_names as _clean_adata_var_names

    hest_shards = sorted([s for s in prepared.glob("*.h5") if ".st1k." not in s.name and ".spatialcorpus." not in s.name])
    if not hest_shards:
        log.warning("No HEST shards to audit.")
        return pd.DataFrame()

    # HEST shards only retain spots whose barcodes match the patch file —
    # re-use the same loader the prepare step used so spot counts line up.
    # `load_paired` still lives in scripts/data/prepare.py — import it
    # there via importlib since scripts/ isn't a package.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_prepare_mod",
        Path(__file__).resolve().parents[1] / "data" / "prepare.py",
    )
    _prep = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_prep)
    load_paired = _prep.load_paired

    sample_paths = hest_shards[:n_check]
    rows = []
    for s in sample_paths:
        sid = s.stem
        h5ad_path = Path("/data/hest/st") / f"{sid}.h5ad"
        if not h5ad_path.exists():
            log.warning(f"raw h5ad missing for {sid}; skip")
            continue
        try:
            adata_p, _uni, _coords, _bc, _idx = load_paired(
                sid, Path("/data/hest/st"), Path("/data/hest/patches"))
        except Exception as e:
            log.warning(f"load_paired failed for {sid}: {e}")
            continue
        if adata_p is None:
            log.warning(f"no barcode match for {sid}; skip")
            continue
        sc.pp.normalize_total(adata_p, target_sum=1e4)
        sc.pp.log1p(adata_p)
        X = adata_p.X.toarray() if hasattr(adata_p.X, "toarray") else np.asarray(adata_p.X)
        X = X.astype(np.float32, copy=False)
        var_idx = {g: i for i, g in enumerate(adata_p.var_names.astype(str))}

        with h5py.File(s, "r") as f:
            shard_hvg = f["hvg_log"][:].astype(np.float32)
            shard_bc = [b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
                        for b in f["barcode"][:]]

        # Reorder source X to match shard barcode order.
        obs_idx = {b: i for i, b in enumerate(adata_p.obs_names.astype(str))}
        order = [obs_idx.get(b, -1) for b in shard_bc]
        if any(i < 0 for i in order):
            n_miss = sum(1 for i in order if i < 0)
            log.warning(f"{sid}: {n_miss} shard barcodes missing in source AnnData; skipping")
            continue
        X = X[np.asarray(order, dtype=np.int64)]

        # Sanity 1: row alignment (now identical by construction).
        spots_ok = shard_hvg.shape[0] == X.shape[0]
        # Sanity 2: for each vocab gene present in the sample, the column
        # should equal the source X column (within float tolerance).
        n_match, n_diff, n_missing = 0, 0, 0
        diff_examples = []
        for j, g in enumerate(vocab):
            i = var_idx.get(g)
            if i is None:
                # gene not in this sample → shard column must be all zeros
                if (shard_hvg[:, j] != 0).any():
                    n_diff += 1
                    if len(diff_examples) < 3:
                        diff_examples.append(f"{g}: missing from sample but shard has values")
                else:
                    n_missing += 1
                continue
            if np.allclose(shard_hvg[:, j], X[:, i], atol=1e-4):
                n_match += 1
            else:
                n_diff += 1
                if len(diff_examples) < 3:
                    diff_examples.append(
                        f"{g}: max diff {np.max(np.abs(shard_hvg[:,j]-X[:,i])):.3g}"
                    )
        rows.append({
            "sample_id": sid,
            "n_spots_shard": shard_hvg.shape[0],
            "n_spots_source": X.shape[0],
            "row_align_ok": spots_ok,
            "n_genes_matched": n_match,
            "n_genes_diff": n_diff,
            "n_genes_zero_filled": n_missing,
            "diff_examples": " | ".join(diff_examples) if diff_examples else "",
        })
        del adata_p, X, shard_hvg
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# 2. seq_len distribution after zero-removal
# ────────────────────────────────────────────────────────────────────────────

def seqlen_distribution(prepared: Path, max_spots_per_shard: int = 2000) -> pd.DataFrame:
    """For every shard, sample `max_spots_per_shard` rows and compute the
    nonzero-position count per spot (= the tokenizer's `seq_len`)."""
    shards = sorted(prepared.glob("*.h5"))
    rng = np.random.default_rng(0)
    rows = []
    for s in shards:
        with h5py.File(s, "r") as f:
            if "hvg_log" not in f: continue
            n = f["hvg_log"].shape[0]
            if n > max_spots_per_shard:
                sel = np.sort(rng.choice(n, max_spots_per_shard, replace=False))
                h = f["hvg_log"][sel]
            else:
                h = f["hvg_log"][:]
            attrs = dict(f.attrs)
        sl = (h > 0).sum(axis=1)
        rows.append({
            "shard": s.name,
            "source": str(attrs.get("source", "hest")),
            "n_spots": int(n),
            "vocab_size": int(h.shape[1]),
            "seq_len_mean": float(sl.mean()),
            "seq_len_median": float(np.median(sl)),
            "seq_len_p5": float(np.percentile(sl, 5)),
            "seq_len_p95": float(np.percentile(sl, 95)),
            "seq_len_min": int(sl.min()),
            "seq_len_max": int(sl.max()),
            "frac_zero_spots": float((sl == 0).mean()),
        })
    return pd.DataFrame(rows)


def fig_seqlen(df: pd.DataFrame, vocab_size: int, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    sources = sorted(df["source"].unique())
    cmap = {"hest": "#4c72b0", "st1k": "#dd8452", "spatialcorpus": "#55a868"}
    for src in sources:
        sub = df[df["source"] == src]
        ax.errorbar(
            np.arange(len(sub)),
            sub["seq_len_mean"],
            yerr=[sub["seq_len_mean"] - sub["seq_len_p5"],
                  sub["seq_len_p95"] - sub["seq_len_mean"]],
            fmt="o", color=cmap.get(src, "gray"), label=src,
            alpha=0.7, capsize=2, markersize=4, elinewidth=0.5,
        )
    ax.set_yscale("symlog", linthresh=10)
    ax.axhline(vocab_size, color="k", linestyle="--", alpha=0.4,
                label=f"vocab size = {vocab_size}")
    ax.set_xlabel("shard index (sorted by source)")
    ax.set_ylabel("seq_len (genes/spot, log-scaled)")
    ax.set_title(f"Per-shard seq_len distribution (mean ± 5–95 percentile)\n"
                  f"Higher = denser tokens; very low = vocab mismatch suspected",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--n-audit", type=int, default=5,
                    help="How many HEST shards to audit for vocab-match correctness.")
    args = ap.parse_args()
    prepared = Path(args.prepared_dir)
    vocab = json.loads((prepared / "hvg_vocab.json").read_text())
    out_dir = reports_dir_for(prepared) / "validate_vocab"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"prepared = {prepared}  vocab = {len(vocab)}  out = {out_dir}")

    # 1) vocab-match audit (HEST only)
    match_df = check_vocab_match(prepared, vocab, n_check=args.n_audit)
    if not match_df.empty:
        match_df.to_csv(out_dir / "vocab_match_audit.csv", index=False)
        log.info(f"vocab-match audit: {len(match_df)} samples")
        if (match_df["n_genes_diff"] > 0).any():
            log.warning("⚠ vocab-match audit found differences — investigate diff_examples column")
        else:
            log.info("✓ vocab-match audit: all sampled shards match their source AnnData")

    # 2) seq_len distribution
    sl_df = seqlen_distribution(prepared, max_spots_per_shard=2000)
    sl_df.to_csv(out_dir / "seqlen_distribution.csv", index=False)
    fig_seqlen(sl_df, vocab_size=len(vocab), out=out_dir / "seqlen_distribution.png")

    # 3) markdown summary
    md = [f"# Vocab validation\n",
          f"prepared: `{prepared.name}`  ·  vocab: **{len(vocab)}**\n",
          "## 1. Vocab-match audit (HEST shards re-projected)\n"]
    if not match_df.empty:
        md.append(match_df.to_markdown(index=False))
        if (match_df["n_genes_diff"] > 0).any():
            md.append("\n**⚠ Differences found** — see `vocab_match_audit.csv`.\n")
        else:
            md.append("\n**✓ All sampled shards match their source AnnData column-by-column.**\n")
    else:
        md.append("_No HEST shards available to audit._\n")

    md.append("\n## 2. seq_len distribution (post zero-removal)\n")
    summary = sl_df.groupby("source").agg(
        n_shards=("shard","count"),
        seq_len_mean=("seq_len_mean","mean"),
        seq_len_median=("seq_len_median","median"),
        seq_len_p5=("seq_len_p5","mean"),
        seq_len_p95=("seq_len_p95","mean"),
        frac_zero_spots=("frac_zero_spots","mean"),
    ).round(2)
    md.append(summary.to_markdown())
    md.append(f"\nVocab size = **{len(vocab)}** → expected `seq_len ≈ vocab_size × mean_spot_prev`.\n")
    md.append("\n![](seqlen_distribution.png)\n")

    (out_dir / "validate_vocab.md").write_text("\n".join(md))
    log.info(f"report → {out_dir / 'validate_vocab.md'}")


if __name__ == "__main__":
    main()
