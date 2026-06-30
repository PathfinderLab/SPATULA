"""Stage-2 slide-level downstream evaluation.

Generic HEST/PathBench-style MIL runner. It freezes a trained Stage-2 model,
extracts per-spot embeddings, groups them into slide bags, and trains only a
small downstream classifier head from train slides to val/test slides.

Example:
    python scripts/eval/slide_mil.py \
        --ckpt results/runs/stage2_best/ckpt_last.pt \
        --experiment configs/sweep/stage2_align.yaml \
        --label-csv /path/to/hest_tasks.csv \
        --sample-id-col sample_id \
        --label-col msi \
        --eval-split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data import build_dataset_from_split, pad_collate
from mm_align.evaluation import encode_loader
from mm_align.evaluation.mil import (
    make_slide_bags,
    run_attention_mil,
    run_pooled_slide_probe,
)
from mm_align.models import MMAligner
from mm_align.utils import get_logger, load_config

log = get_logger("eval_slide_mil")


def _read_labels(path: Path, sample_id_col: str, label_col: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if sample_id_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain {sample_id_col!r} and {label_col!r}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            sid = str(row.get(sample_id_col, "")).strip()
            val = str(row.get(label_col, "")).strip()
            if sid and val and val.lower() not in {"nan", "none", "unknown"}:
                labels[sid] = val
    return labels


def _features_for_arm(arm: str, pool: dict) -> np.ndarray:
    if arm == "image":
        return pool["h_image"]
    if arm == "z_image":
        return pool["z_image"]
    if arm == "multimodal":
        return np.concatenate([pool["h_image"], pool["h_tx"]], axis=-1)
    if arm == "tx":
        return pool["h_tx"]
    if arm == "baseline_uni":
        return pool["image_feat"]
    raise ValueError(f"unknown arm={arm!r}")


def _sample_ids(ds) -> list[str]:
    return [str(getattr(s, "sample_id", s.path.stem.split(".")[0])) for s in ds.shards]


def _filter_labeled(ids: list[str], labels: dict[str, str], prepared: Path) -> list[str]:
    out = []
    for sid in ids:
        # Stage2 is HEST-only by default; non-HEST shards are suffixed and do not
        # usually have H&E labels in HEST/PathBench task CSVs.
        if sid in labels and (prepared / f"{sid}.h5").exists():
            out.append(sid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/stage1/data.yaml")
    ap.add_argument("--model", default="configs/stage1/model.yaml")
    ap.add_argument("--train", default="configs/stage1/train.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--label-csv", required=True)
    ap.add_argument("--sample-id-col", default="sample_id")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--eval-split", default="test", choices=("val", "test"))
    ap.add_argument("--arms", nargs="+", default=["image", "z_image", "baseline_uni"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-spots", type=int, default=8)
    ap.add_argument("--attention-epochs", type=int, default=50)
    ap.add_argument("--skip-attention", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt_state, dict) and "cfg" in ckpt_state:
        cfg = ckpt_state["cfg"]
    else:
        cfg = load_config([args.data, args.model, args.train, args.experiment])
    prepared = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared / "splits.json").read_text())
    labels = _read_labels(Path(args.label_csv), args.sample_id_col, args.label_col)

    train_ids = _filter_labeled(list(splits.get("train", [])), labels, prepared)
    eval_ids = _filter_labeled(list(splits.get(args.eval_split, [])), labels, prepared)
    if not train_ids or not eval_ids:
        raise SystemExit(
            f"No labeled slides for train/eval. train={len(train_ids)} eval={len(eval_ids)} "
            f"label_col={args.label_col!r}"
        )
    log.info(f"task={args.label_col} train_slides={len(train_ids)} {args.eval_split}_slides={len(eval_ids)}")

    ds_kwargs = dict(
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"].get("use_hvg", True),
        image_mode=cfg["data"].get("image_mode", "feature"),
        hest_patch_dir=cfg["data"].get("hest_patch_dir"),
        gene_norm_cfg=cfg["data"].get("gene_norm"),
    )
    train_ds = build_dataset_from_split(prepared, train_ids, **ds_kwargs)
    eval_ds = build_dataset_from_split(prepared, eval_ids, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=pad_collate)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMAligner(cfg).to(device)
    state = ckpt_state.get("model", ckpt_state) if isinstance(ckpt_state, dict) else ckpt_state
    model.load_state_dict(state, strict=False)
    model.eval()

    log.info("encoding train bags...")
    pool_t = encode_loader(model, train_loader, device, desc="mil train")
    log.info(f"encoding {args.eval_split} bags...")
    pool_e = encode_loader(model, eval_loader, device, desc=f"mil {args.eval_split}")

    rows: dict[str, dict] = {}
    sample_ids_t = _sample_ids(train_ds)
    sample_ids_e = _sample_ids(eval_ds)
    for arm in args.arms:
        zt = _features_for_arm(arm, pool_t)
        ze = _features_for_arm(arm, pool_e)
        train_bags = make_slide_bags(zt, pool_t["sample_idx"], sample_ids_t, labels, min_spots=args.min_spots)
        eval_bags = make_slide_bags(
            ze, pool_e["sample_idx"], sample_ids_e, labels,
            min_spots=args.min_spots,
            label_names=train_bags.label_names,
        )
        row = {
            "task": args.label_col,
            "arm": arm,
            "eval_split": args.eval_split,
            "label_names": train_bags.label_names,
            "n_train_bags": len(train_bags.bags),
            "n_eval_bags": len(eval_bags.bags),
        }
        for pooling in ("mean", "max"):
            metrics = run_pooled_slide_probe(train_bags, eval_bags, pooling=pooling)
            row.update({f"{pooling}/{k}": v for k, v in metrics.items()})
        if not args.skip_attention:
            metrics = run_attention_mil(
                train_bags, eval_bags,
                epochs=args.attention_epochs,
                device=str(device),
            )
            row.update({f"attention/{k}": v for k, v in metrics.items()})
        rows[arm] = row

    out = Path(args.out or Path(args.ckpt).parent / f"slide_mil_{args.label_col}_{args.eval_split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    csv_out = out.with_suffix(".csv")
    flat_rows = []
    for row in rows.values():
        flat = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in row.items()}
        flat_rows.append(flat)
    import pandas as pd
    pd.DataFrame(flat_rows).to_csv(csv_out, index=False)
    log.info(f"wrote {out}")
    log.info(f"wrote {csv_out}")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
