#!/usr/bin/env python
"""Evaluate Stage-1 tx_encoder on external spot-level annotations.

This script expects shards produced by scripts/data/prepare_external_validation.py.
It reads `/hvg_log` and `/annotation/*` from each shard, encodes spots with a
frozen Stage-1 tx_encoder, and reports simple extrinsic annotation metrics:
linear-probe accuracy/F1 and kNN retrieval purity/recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

from mm_align.data.gene_norm import GeneNormalizer  # noqa: E402
from stage1_tx import (  # noqa: E402
    encode_pool,
    linear_probe,
    load_ablation_meta,
    load_tx_encoder,
    retrieval_recall,
)


def _decode(arr) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def _resolve_shard(prepared: Path, sid: str) -> Path | None:
    for suffix in ("", ".gse176078", ".her2st", ".st1k", ".spatialcorpus"):
        p = prepared / f"{sid}{suffix}.h5"
        if p.exists():
            return p
    return None


def _load_split(prepared: Path, split_file: str | None, split: str) -> list[str]:
    if split_file:
        p = Path(split_file)
    else:
        cands = sorted(prepared.glob("splits_*validation.json")) + [prepared / "splits.json"]
        p = next((x for x in cands if x.exists()), None)
        if p is None:
            raise FileNotFoundError(f"no split json under {prepared}")
    obj = json.loads(p.read_text())
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if split not in obj:
        # External files use external_validation as a friendly alias.
        split = "external_validation" if "external_validation" in obj else next(iter(obj))
    return [str(x) for x in obj[split]]


def load_external_pool(prepared: Path, sample_ids: list[str], field: str | None, max_spots: int, seed: int):
    hvg_parts = []
    labels_parts = []
    sample_parts = []
    fields_seen: set[str] = set()
    rng = np.random.default_rng(seed)
    for sid in sample_ids:
        p = _resolve_shard(prepared, sid)
        if p is None:
            continue
        with h5py.File(p, "r") as f:
            if "hvg_log" not in f or "annotation" not in f:
                continue
            ann = f["annotation"]
            fields = [k for k in ann.keys() if not k.endswith("_proba")]
            fields_seen.update(fields)
            use_field = field
            if use_field is None:
                preferred = [x for x in fields if any(k in x.lower() for k in ["major", "cell", "type", "label", "cluster", "argmax"])]
                use_field = preferred[0] if preferred else (fields[0] if fields else None)
            if not use_field or use_field not in ann:
                continue
            y = _decode(ann[use_field][:])
            keep = np.array([(str(v).strip() not in {"", "nan", "None", "Unknown"}) for v in y], dtype=bool)
            if keep.sum() < 5:
                continue
            X = f["hvg_log"][:]
            idx = np.flatnonzero(keep)
            if max_spots > 0 and idx.size > max_spots:
                idx = rng.choice(idx, max_spots, replace=False)
            hvg_parts.append(X[idx].astype(np.float32))
            labels_parts.append(y[idx])
            sample_parts.append(np.asarray([sid] * len(idx), dtype=object))
    if not hvg_parts:
        raise RuntimeError(f"no annotated spots loaded. fields_seen={sorted(fields_seen)}")
    return np.concatenate(hvg_parts), np.concatenate(labels_parts), np.concatenate(sample_parts), sorted(fields_seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--split", default="external_validation")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--field", default=None, help="Annotation field under /annotation. Auto-select if omitted.")
    ap.add_argument("--max-spots", type=int, default=20000)
    ap.add_argument("--out", default="results/eval/external_annotation_probe.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    sample_ids = _load_split(prepared, args.split_file, args.split)
    hvg, labels, sample_ids_per_spot, fields_seen = load_external_pool(
        prepared, sample_ids, args.field, max_spots=args.max_spots, seed=0)
    label_names, y = np.unique(labels.astype(str), return_inverse=True)

    rows = []
    for ck in args.ckpts:
        ck = Path(ck)
        enc, cfg, vocab_keep, gene_norm_cfg = load_tx_encoder(ck, device=args.device)
        d_eff = int(len(vocab_keep)) if vocab_keep is not None else hvg.shape[1]
        normalizer = GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=hvg.shape[1], hvg_dim=d_eff,
            vocab_keep_indices=vocab_keep,
        ) if gene_norm_cfg else None
        emb = encode_pool(enc, hvg, vocab_keep, normalizer, device=args.device)
        row = {
            "ckpt": ck.parent.name,
            "ckpt_path": str(ck),
            "prepared_dir": str(prepared),
            "field": args.field or "auto",
            "fields_seen": json.dumps(fields_seen, ensure_ascii=False),
            "n_spots": int(emb.shape[0]),
            "n_samples": int(len(np.unique(sample_ids_per_spot))),
            "n_classes": int(len(label_names)),
            "classes": json.dumps(label_names.tolist(), ensure_ascii=False),
            "embed_dim": int(emb.shape[1]),
            "input_dim": int(d_eff),
        }
        row.update(load_ablation_meta(ck))
        row.update({f"annotation_probe_{k}": v for k, v in linear_probe(emb, y).items()})
        row.update({f"annotation_retrieval_{k}": v for k, v in retrieval_recall(emb, y).items()})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[external-eval] saved {out}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
