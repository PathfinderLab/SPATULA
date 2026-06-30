"""SEAL-style linear-probe evaluation against a trained ckpt.

5-fold Ridge on (Z_train, y_train_HVG), evaluated on a held-out test pool.
Per-gene Pearson / Spearman / R² / L2, mean ± std across folds, plus an
optional per-organ breakdown when organ labels are available.

Runs once at end of training (not per-epoch).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import load_config, get_logger
from mm_align.data import build_dataset_from_split, pad_collate
from mm_align.models import MMAligner
from mm_align.evaluation import (
    encode_loader, run_linprobe, hest_metadata, spot_organ_labels,
)

log = get_logger("eval_probe")


def _img_of(arm: str, pool: dict) -> np.ndarray:
    if arm == "ours_image_only": return pool["h_image"]
    if arm == "ours_multimodal": return np.concatenate([pool["h_image"], pool["h_tx"]], axis=-1)
    if arm == "ours_tx_only":    return pool["h_tx"]
    if arm == "baseline_uni":    return pool["image_feat"]
    if arm == "baseline_novae":  return pool["novae_latent"]
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="configs/stage1/data.yaml")
    ap.add_argument("--model", default="configs/stage1/model.yaml")
    ap.add_argument("--train", default="configs/stage1/train.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--eval_split", default="test", choices=["val", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca_n", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max_train_spots", type=int, default=100_000)
    ap.add_argument("--max_eval_spots", type=int, default=30_000)
    ap.add_argument("--arms", nargs="+",
                    default=["ours_image_only", "ours_multimodal", "baseline_uni"])
    args = ap.parse_args()

    ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt_state, dict) and "cfg" in ckpt_state:
        cfg = ckpt_state["cfg"]
    else:
        cfg = load_config([args.data, args.model, args.train, args.experiment])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared / "splits.json").read_text())
    train_ids = splits["train"]
    eval_ids = splits[args.eval_split]

    ds_kwargs = dict(
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
        image_mode=cfg["data"]["image_mode"],
        hest_patch_dir=cfg["data"]["hest_patch_dir"],
        gene_norm_cfg=cfg["data"].get("gene_norm"),
    )
    train_ds = build_dataset_from_split(prepared, train_ids, **ds_kwargs)
    eval_ds = build_dataset_from_split(prepared, eval_ids, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, collate_fn=pad_collate)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, collate_fn=pad_collate)

    log.info(f"Loading model + ckpt: {args.ckpt}")
    model = MMAligner(cfg).to(device)
    model.load_state_dict(ckpt_state["model"], strict=False)
    model.eval()

    log.info("Encoding train pool ...")
    pool_t = encode_loader(model, train_loader, device, desc="probe train")
    log.info(f"Encoding {args.eval_split} pool ...")
    pool_e = encode_loader(model, eval_loader, device, desc=f"probe {args.eval_split}")

    # Subsample for speed.
    rng = np.random.default_rng(0)
    if pool_t["h_image"].shape[0] > args.max_train_spots:
        idx = rng.choice(pool_t["h_image"].shape[0], args.max_train_spots, replace=False)
        for k, v in list(pool_t.items()):
            if v is None: continue
            try: pool_t[k] = v[idx]
            except Exception: pass
    if pool_e["h_image"].shape[0] > args.max_eval_spots:
        idx = rng.choice(pool_e["h_image"].shape[0], args.max_eval_spots, replace=False)
        for k, v in list(pool_e.items()):
            if v is None: continue
            try: pool_e[k] = v[idx]
            except Exception: pass

    # HVG gene names (for per-gene reporting).
    vocab_path = prepared / "hvg_vocab.json"
    if vocab_path.exists():
        gene_names = json.loads(vocab_path.read_text())
    else:
        gene_names = None

    # Per-organ mapping (optional).
    try:
        meta = hest_metadata()
        organ_per_spot = spot_organ_labels(
            [s.sample_id for s in eval_ds.shards],
            pool_e.get("sample_idx"), meta,
        )
    except Exception:
        organ_per_spot = None

    # We DON'T have per-gene-per-organ annotations from SEAL's `n_test_genes`
    # mapping out of the box; fall back to flat (no per-organ breakdown).
    organ_to_genes = None

    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for arm in args.arms:
        Z_tr = _img_of(arm, pool_t)
        Z_ev = _img_of(arm, pool_e)
        y_tr = pool_t["hvg"]; y_ev = pool_e["hvg"]
        if y_tr is None or y_ev is None:
            log.warning("HVG missing in encode pool — cannot run linear probe.")
            continue
        log.info(f"5-fold Ridge probe — arm={arm}, Z_train={Z_tr.shape}, Z_test={Z_ev.shape}")
        r = run_linprobe(Z_tr, Z_ev, y_tr, y_ev,
                         gene_names=gene_names,
                         organ_to_genes=organ_to_genes,
                         folds=args.folds, pca_reduce=True, pca_n=args.pca_n,
                         alpha=args.alpha)
        # Drop per-gene huge dicts from headline json (keep them in a separate file).
        per_gene = {k: r.pop(k) for k in list(r.keys()) if k.startswith("per_gene_")}
        results[arm] = r
        (out_dir / f"linear_probe_{arm}_per_gene.json").write_text(json.dumps(per_gene, indent=2))
        log.info(f"[{arm}] PCC={r['pcc/mean']:.4f}±{r['pcc/std']:.4f}  "
                 f"Spearman={r['spearman/mean']:.4f}±{r['spearman/std']:.4f}  "
                 f"R²={r['r2/mean']:.4f}  L2={r['l2/mean']:.4f}")

    (out_dir / f"linear_probe_{args.eval_split}.json").write_text(json.dumps(results, indent=2))

    # Bar chart of per-arm PCC mean.
    try:
        arms = list(results.keys())
        fig, ax = plt.subplots(figsize=(6, 4))
        means = [results[a]["pcc/mean"] for a in arms]
        stds  = [results[a]["pcc/std"]  for a in arms]
        ax.bar(arms, means, yerr=stds, capsize=4)
        ax.set_ylabel("Pearson r (mean across HVG)")
        ax.set_title(f"Linear probe (5-fold Ridge, PCA={args.pca_n})")
        plt.xticks(rotation=20, ha="right"); plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / f"linear_probe_{args.eval_split}.png", dpi=120)
        plt.close()
    except Exception as e:
        log.warning(f"plot failed: {e}")

    print(json.dumps({a: {k: v for k, v in r.items() if not isinstance(v, dict) and "per_" not in k}
                      for a, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
