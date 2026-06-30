"""Create a clipped vocab subset for runtime use without re-running prepare.

Reads results/cache/prepared_expanded/vocab.csv, keeps top-K by priority_rank,
emits:
  - clipped_vocab.json          flat list of N genes (subset)
  - clipped_vocab_dict.json     {gene -> token_id} with specials at front
  - clipped_keep_indices.npy    int64 indices INTO the full hvg_log
                                (use as vocab_keep_indices in PairedSpotDataset)

Usage:
  python scripts/data/make_clipped_vocab.py --top-k 4096
  python scripts/data/make_clipped_vocab.py --top-k 8192 --out-tag clip8k
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--top-k", type=int, required=True,
                    help="How many vocab genes to keep (by priority_rank).")
    ap.add_argument("--out-tag", default=None,
                    help="Suffix for output files; defaults to clip<K>k.")
    args = ap.parse_args()

    prep = Path(args.prepared_dir)
    full_vocab = json.loads((prep / "hvg_vocab.json").read_text())   # 19,183 genes, hvg_log column order
    df = pd.read_csv(prep / "vocab.csv")
    assert df["gene"].nunique() == len(full_vocab), "vocab.csv vs hvg_vocab.json mismatch"

    # gene → its column index in hvg_log (= position in full_vocab list)
    gene_to_col = {g: i for i, g in enumerate(full_vocab)}

    # Keep top-K by priority_rank.  priority_rank already places must_include
    # first, then dispersion-ordered survivors.
    keep_df = df.sort_values("priority_rank").head(args.top_k)
    keep_genes = keep_df["gene"].tolist()

    # Map back to hvg_log column indices (preserved order = priority_rank order).
    keep_idx = np.asarray([gene_to_col[g] for g in keep_genes], dtype=np.int64)

    tag = args.out_tag or f"clip{args.top_k}"
    out_vocab = prep / f"{tag}_vocab.json"
    out_dict  = prep / f"{tag}_vocab_dict.json"
    out_idx   = prep / f"{tag}_keep_indices.npy"

    out_vocab.write_text(json.dumps(keep_genes))
    # token table with special tokens up front (PAD=0/MASK=1/CLS=2/UNK=3)
    SPECIALS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[UNK]": 3}
    full_dict = dict(SPECIALS)
    for i, g in enumerate(keep_genes):
        full_dict[g] = i + len(SPECIALS)
    out_dict.write_text(json.dumps(full_dict))
    np.save(out_idx, keep_idx)

    # Quick stats
    must_kept = int(keep_df["must_include"].sum())
    tier_dist = keep_df.get("review_tier", pd.Series(dtype=str)).value_counts().to_dict()
    print(f"Wrote {out_vocab.name} / {out_dict.name} / {out_idx.name}")
    print(f"  kept: {len(keep_genes)} genes")
    print(f"  must_include in subset: {must_kept} / {df['must_include'].sum()}")
    print(f"  dispersion range: [{keep_df['norm_dispersion'].min():.2f}, "
          f"{keep_df['norm_dispersion'].max():.2f}]")
    print(f"  sample_prev median: {keep_df['sample_prev'].median():.3f}")
    if tier_dist:
        print(f"  tier distribution: {tier_dist}")


if __name__ == "__main__":
    main()
