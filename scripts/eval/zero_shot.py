"""Pure zero-shot evaluation against a trained ckpt — NO supervised probe.

Tasks:
  - Retrieval (image ↔ tx)
  - Clustering (KMeans silhouette + ARI/NMI vs organ labels from HEST CSV)
  - RankMe (effective rank)
  - Alignment / Uniformity (Wang & Isola)
  - Modality gap
  - UMAP visualisation (PNG)

For supervised "image → HVG" or "image → organ" prediction, use eval_linearprobe.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import load_config, get_logger
from mm_align.data import build_dataset_from_split, pad_collate
from mm_align.models import MMAligner
from mm_align.evaluation import (
    encode_loader, run_zero_shot_eval, render_zero_shot_curves,
    render_umap_panel, hest_metadata, spot_organ_labels,
)

log = get_logger("eval_zs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/stage1/data.yaml")
    ap.add_argument("--model", default="configs/stage1/model.yaml")
    ap.add_argument("--train", default="configs/stage1/train.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Prefer the cfg that the ckpt was trained with (covers backbone/tune
    # combinations the default yamls may no longer match).
    ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt_state, dict) and "cfg" in ckpt_state:
        cfg = ckpt_state["cfg"]
    else:
        cfg = load_config([args.data, args.model, args.train, args.experiment])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared / "splits.json").read_text())
    ids = splits[args.split]
    if args.limit:
        ids = ids[: args.limit]

    ds_kwargs = dict(
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
        image_mode=cfg["data"]["image_mode"],
        hest_patch_dir=cfg["data"]["hest_patch_dir"],
        gene_norm_cfg=cfg["data"].get("gene_norm"),
    )
    ds = build_dataset_from_split(prepared, ids, **ds_kwargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=pad_collate)

    model = MMAligner(cfg).to(device)
    state = ckpt_state  # already loaded above
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata + labels
    try:
        meta = hest_metadata()
    except Exception as e:
        log.warning(f"HEST metadata unavailable ({e})")
        meta = None
    sample_ids_local = [s.sample_id for s in ds.shards]

    # ---- Zero-shot eval ----
    log.info(f"Running zero-shot eval on split={args.split} (samples={len(ids)})")
    ev = run_zero_shot_eval(model, loader, device, cfg, epoch=state.get("epoch", 0),
                            sample_id_for_idx=sample_ids_local,
                            organ_for_sample=meta["organ"] if meta else None)
    (out_dir / f"zero_shot_{args.split}.json").write_text(json.dumps(ev, indent=2))
    log.info(f"Wrote {out_dir/f'zero_shot_{args.split}.json'}")

    # ---- UMAP panel ----
    pool = encode_loader(model, loader, device, desc="umap pool")
    organ = spot_organ_labels(sample_ids_local, pool.get("sample_idx"), meta) if meta else None
    render_umap_panel(pool, out_dir / f"umap_{args.split}.png",
                      organ_labels=organ, title=f"{Path(args.ckpt).parent.name} ({args.split})")
    log.info(f"Wrote {out_dir/f'umap_{args.split}.png'}")

    # ---- Pretty print summary ----
    summary = {k: v for k, v in ev.items() if any(
        s in k for s in ("retr/i2t_R@10", "cluster/silhouette", "cluster/ari",
                          "rankme/image", "modality_gap"))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
