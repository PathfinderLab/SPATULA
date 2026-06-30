"""Compute per-spot z_tx (and h_image) for every spot in the prepared shards
using a trained mm_align checkpoint.

This is the "ours embedding prepare" step — after training the gene encoder,
run this to materialise z_tx/h_tx as float arrays you can use downstream
(retrieval indexes, gene-prediction probes, plotting, etc.) without
re-running the model each time.

Output: one .npz per sample under `--out_dir/<sample_id>.npz`:
    barcodes    (S,)         str
    coords      (S, 2)       float32
    h_image     (S, D)       float32 — model's pre-projector image latent
    h_tx        (S, D)       float32 — model's pre-projector tx latent  ★
    z_image     (S, D_proj)  float32 — model's projected image latent
    z_tx        (S, D_proj)  float32 — model's projected tx latent

Use:
    python scripts/eval/extract_embeddings.py \\
        --ckpt results/runs/<tag>/ckpt_best.pt \\
        --out_dir results/cache/embeddings/<tag> \\
        --splits train val test
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

log = get_logger("extract_emb")


@torch.no_grad()
def encode_sample(model, shard_path: Path, cfg: dict, device, batch_size: int = 256,
                  num_workers: int = 4) -> dict[str, np.ndarray]:
    """Encode one prepared shard end-to-end."""
    sid = shard_path.stem
    ds = build_dataset_from_split(
        shard_path.parent, [sid],
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
        image_mode=cfg["data"]["image_mode"],
        hest_patch_dir=cfg["data"]["hest_patch_dir"],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=pad_collate)

    h_imgs, h_txs, z_imgs, z_txs = [], [], [], []
    barcodes, coords = [], []

    model.eval()
    for batch in loader:
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
        out = model(b)
        h_imgs.append(out["h_image"].float().cpu().numpy())
        h_txs.append(out["h_tx"].float().cpu().numpy())
        z_imgs.append(out["z_image"].float().cpu().numpy())
        z_txs.append(out["z_tx"].float().cpu().numpy())

    # Pull barcodes/coords directly from the shard.
    import h5py
    with h5py.File(shard_path, "r") as f:
        raw_bc = f["barcode"][:]
        bcs = np.array([
            (b[0].item() if hasattr(b, "shape") and b.shape else b)
            if not isinstance(b, (bytes, bytearray)) else b.decode("utf-8")
            for b in raw_bc
        ])
        if bcs.dtype.kind in ("S", "U"):
            bcs = np.array([str(x) if not isinstance(x, str) else x for x in bcs])
        coords_arr = f["coords"][:].astype(np.float32)

    return {
        "barcodes": bcs,
        "coords": coords_arr,
        "h_image": np.concatenate(h_imgs, axis=0),
        "h_tx":    np.concatenate(h_txs,  axis=0),
        "z_image": np.concatenate(z_imgs, axis=0),
        "z_tx":    np.concatenate(z_txs,  axis=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_best.pt or ckpt_last.pt.")
    ap.add_argument("--out_dir", required=True, help="Where to write .npz files.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Which split(s) to embed.")
    ap.add_argument("--data", default="configs/stage1/data.yaml")
    ap.add_argument("--model", default="configs/stage1/model.yaml")
    ap.add_argument("--train", default="configs/stage1/train.yaml")
    ap.add_argument("--experiment", default=None,
                    help="Optional: matched experiment yaml (only needed if "
                         "the ckpt doesn't embed cfg).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "cfg" in state:
        cfg = state["cfg"]
    else:
        cfg_files = [args.data, args.model, args.train]
        if args.experiment:
            cfg_files.append(args.experiment)
        cfg = load_config(cfg_files)
        if not args.experiment:
            log.warning("ckpt has no embedded cfg and --experiment not given; "
                        "you may hit a shape mismatch on load.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    prepared = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared / "splits.json").read_text())

    log.info(f"Loading model from {args.ckpt}")
    model = MMAligner(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    log.info(f"Loaded. tx_kind={getattr(model, 'tx_kind', 'n/a')}")

    summary = []
    for split in args.splits:
        ids = splits.get(split, [])
        log.info(f"== split={split}: {len(ids)} samples ==")
        for sid in ids:
            shard = prepared / f"{sid}.h5"
            if not shard.exists():
                log.warning(f"  missing shard: {sid}, skip"); continue
            out_path = out_dir / f"{sid}.npz"
            if out_path.exists() and not args.overwrite:
                log.info(f"  {sid}: already exists, skip (use --overwrite)")
                continue
            try:
                rec = encode_sample(model, shard, cfg, device,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
                np.savez_compressed(out_path, **rec)
                log.info(f"  {sid}: wrote {out_path.name} "
                         f"(N={rec['h_image'].shape[0]}, D_h={rec['h_image'].shape[1]}, "
                         f"D_z={rec['z_image'].shape[1]})")
                summary.append({"sample": sid, "split": split,
                                 "n_spots": int(rec["h_image"].shape[0])})
            except Exception as e:
                log.exception(f"  {sid}: failed ({e})")

    (out_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Done. {len(summary)} samples written to {out_dir}")


if __name__ == "__main__":
    main()
