"""Re-emit `splits.json` for the prepared shards with organ-stratified
train/val/test split (default 8:1:1).

Source of organ labels: /data/hest/HEST_v1_1_0.csv (HEST metadata).

Usage:
    python scripts/data/resplit.py                        # default 8:1:1, seed=42
    python scripts/data/resplit.py --val_frac 0.05 --test_frac 0.05
    python scripts/data/resplit.py --backup                # save splits.json.bak first
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import load_config, get_logger
from mm_align.data.pairs import stratified_split

log = get_logger("resplit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/stage1/data.yaml")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hest_csv", default="/data/hest/HEST_v1_1_0.csv")
    ap.add_argument("--group_col", default="organ",
                    help="Metadata column to stratify by (organ / disease_state / tissue).")
    ap.add_argument("--backup", action="store_true",
                    help="Save existing splits.json as splits.json.bak first.")
    args = ap.parse_args()

    cfg = load_config([args.data])["data"]
    prepared = Path(cfg["prepared_dir"])
    splits_path = prepared / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"[resplit] {splits_path} not found — run prepare_data first")

    cur = json.loads(splits_path.read_text())
    all_ids = sorted(set(cur.get("train", []) + cur.get("val", []) + cur.get("test", [])))
    log.info(f"Found {len(all_ids)} samples in current splits.json "
             f"(train={len(cur.get('train',[]))} val={len(cur.get('val',[]))} test={len(cur.get('test',[]))})")

    # Load HEST organ labels.
    df = pd.read_csv(args.hest_csv)
    id_to_group = dict(zip(df["id"].astype(str), df[args.group_col].astype(str)))
    n_known = sum(1 for sid in all_ids if sid in id_to_group)
    log.info(f"HEST metadata: matched organ for {n_known}/{len(all_ids)} samples "
             f"(stratify column = '{args.group_col}')")

    # Stratified split.
    new_splits = stratified_split(
        all_ids, id_to_group,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    sizes = {k: len(v) for k, v in new_splits.items()}
    log.info(f"New split sizes: train={sizes['train']} val={sizes['val']} test={sizes['test']}  "
             f"(target ratios train={1-args.val_frac-args.test_frac:.2f} "
             f"val={args.val_frac:.2f} test={args.test_frac:.2f})")

    # Distribution table.
    def _dist(ids: list[str]) -> dict[str, int]:
        return dict(Counter(id_to_group.get(sid, "Unknown") for sid in ids))
    for k in ("train", "val", "test"):
        d = _dist(new_splits[k])
        log.info(f"  {k:5s} | "
                 + " ".join(f"{g}={n}" for g, n in sorted(d.items(), key=lambda x: -x[1])[:10])
                 + (f"  (+{len(d)-10} more groups)" if len(d) > 10 else ""))

    # Backup + write.
    if args.backup and splits_path.exists():
        bak = splits_path.with_suffix(".json.bak")
        bak.write_text(splits_path.read_text())
        log.info(f"Backed up old splits → {bak}")
    splits_path.write_text(json.dumps(new_splits, indent=2))
    log.info(f"Wrote new splits → {splits_path}")


if __name__ == "__main__":
    main()
