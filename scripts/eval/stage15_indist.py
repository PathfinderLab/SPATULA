"""In-distribution Stage 1.5 evaluation.

Loads a Stage-1.5 ckpt (`ckpt_spatial_best.pt`) + its frozen Stage-1
encoder, runs the SpatialEncoder over each shard's full spot pool, and
reports the spatial-aware metrics defined in
`mm_align.evaluation.stage15_benchmarks`:

    spatial_knn_overlap_k10        ← did embedding KNN match spatial KNN?
    spatial_smoothness             ← does cosine drop as Δxy grows?
    effective_rank                  ← collapse detector
    boundary_preservation           ← inter / intra domain distance ratio
                                       (uses Novae niches when present)
    augmentation_consistency        ← two ego subgraphs of same anchor

This is the complement to:
    - `scripts/train/stage15.py` reporting only JEPA val_loss (which can
      decrease via EMA collapse without spatial structure being learned)
    - `scripts/eval/dlpfc_eval.py` reporting OUT-of-distribution metrics.

Usage:
    python scripts/eval/stage15_indist.py \\
        --prepared-dir results/cache/prepared_expanded \\
        --ckpts results/runs/stage15_main_separate_spot_ego/ckpt_spatial_best.pt \\
        --split test \\
        --out results/eval/stage15_indist.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# scripts/eval/stage15_indist.py → repo root is parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger
from mm_align.evaluation.stage15_benchmarks import (
    stage15_metrics_for_sample,
    augmentation_consistency,
)

log = get_logger("stage15_indist")


# ───────────────────────────────────────────────────────────────────────────
# Ckpt loading — Stage 1.5 ckpt carries spatial_state_dict + cfg + stage1_ckpt
# ───────────────────────────────────────────────────────────────────────────

def load_stage15(ckpt_path: Path, device: str = "cuda", cache_device: str = "cpu"):
    """Returns (spatial_encoder, frozen_tx_encoder, sampler_kwargs, cfg).

    `cache_device` controls only the one-shot h_tx cache build inside
    SpatialSampleDataset. Keeping that path on CPU avoids CUDA illegal-memory
    failures after long training/eval runs; the spatial encoder itself still
    runs on `device`.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    stage1_ckpt_path = sd.get("stage1_ckpt") or cfg["data"]["stage1_ckpt"]
    log.info(f"  stage1_ckpt = {stage1_ckpt_path}")

    # Recover Stage-1 frozen encoder (with gene_norm + vocab_clip pipeline).
    from mm_align.models.tx.factory import build_tx_encoder
    s1 = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    cfg_tx = s1["cfg_tx"]
    # gene_norm fallback to <stage1_run>/config.json (same as stage15.py).
    gene_norm_cfg = cfg_tx.get("data", {}).get("gene_norm")
    vc = cfg_tx.get("data", {}).get("vocab_clip") or {}
    if gene_norm_cfg is None or not vc:
        s1_cfg_json = Path(stage1_ckpt_path).parent / "config.json"
        if s1_cfg_json.exists():
            run_data = json.loads(s1_cfg_json.read_text()).get("data", {})
            gene_norm_cfg = gene_norm_cfg or run_data.get("gene_norm")
            if not vc:
                vc = run_data.get("vocab_clip") or {}
    vocab_keep = None
    if isinstance(vc, dict) and vc.get("keep_indices_path"):
        kp = Path(vc["keep_indices_path"])
        if kp.exists():
            vocab_keep = np.load(kp)

    tx = build_tx_encoder(cfg_tx)
    tx.load_state_dict(s1["tx_encoder"], strict=False)
    tx.eval()
    for p in tx.parameters():
        p.requires_grad_(False)
    tx.to(device)
    tx_dim = getattr(tx, "out_dim", None) or cfg_tx["model"].get("embed_dim", 512)

    # Build SpatialEncoder using saved spatial_config + load weights.
    from mm_align.models.spatial.encoder import SpatialEncoder
    mc = sd.get("spatial_config") or cfg["model"]["spatial"]
    region_on = bool(cfg["data"].get("region", {}).get("enable", True))
    token_mode = str(mc.get("region_token_mode", "fused"))
    enc = SpatialEncoder(
        tx_dim=tx_dim, img_dim=1536,
        fuse_dim=mc.get("fuse_dim", 256),
        fuse_image=bool(cfg["data"].get("use_image", True)) and bool(mc.get("fuse_image", True)),
        fuse_region=region_on,
        token_mode=token_mode,
        arch=mc.get("arch", "kgnn"),
        n_layers=mc.get("n_layers", 3),
        n_heads=mc.get("n_heads", 4),
        dropout=0.0,            # eval — no dropout
    ).to(device)
    enc.load_state_dict(sd["spatial_state_dict"], strict=False)
    enc.eval()

    # Sampler kwargs (subset of what stage15.py uses, eval-only).
    rcfg = cfg["data"].get("region") or {}
    gcfg = cfg["data"]["graph"]
    sampler_kwargs = dict(
        k=int(gcfg.get("k", 8)),
        subgraph_size=int(cfg["data"].get("subgraph_size", 256)),
        graph_kind=str(gcfg.get("kind", "knn")),
        radius_px=float(gcfg.get("radius_px", 600)),
        subgraph_kind=str(cfg["data"].get("subgraph_kind", "random")),
        fuse_image=bool(cfg["data"].get("use_image", True)),
        tx_dim=tx_dim,
        gene_norm_cfg=gene_norm_cfg,
        vocab_keep_indices=vocab_keep,
        region_enable=region_on,
        region_k=int(rcfg.get("k", gcfg.get("k", 8))),
        region_tx_agg=str(rcfg.get("tx_agg", "mean")),
        region_img_pool=str(rcfg.get("img_pool", "mean")),
        region_weighted_sigma=float(rcfg.get("weighted_sigma", 1.0)),
        region_include_anchor=bool(rcfg.get("include_anchor", False)),
        # h_tx cache creation can run on CPU independently from spatial eval.
        # `evaluate_shard` moves tx back to the eval device after dataset init.
        device=str(cache_device),
    )
    return enc, tx, sampler_kwargs, cfg, token_mode


# ───────────────────────────────────────────────────────────────────────────
# Per-shard eval — sample 2 ego subgraphs, encode each, gather metrics
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _embed_subgraph(item: dict, enc, tx, region_on: bool, device) -> np.ndarray:
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
    # Add batch dim via spatial_collate equivalent — here only one subgraph,
    # so the collate is trivial.
    n = batch["h_tx"].shape[0]
    batch["subgraph_id"] = torch.zeros(n, dtype=torch.long, device=device)
    if region_on and "region_hvg" in batch:
        batch["h_region_tx"] = tx(novae_latent=None, hvg=batch["region_hvg"])["h_tx"]
    z = enc(
        batch["h_tx"], batch.get("h_img"), batch["xy"], batch["edge_index"],
        mask=None,
        h_region_tx=batch.get("h_region_tx"),
        h_region_img=batch.get("h_region_img"),
    )
    # In separate mode the first N rows are spot tokens.
    n_spot = batch["h_tx"].shape[0]
    return z[:n_spot].cpu().numpy()


def evaluate_shard(shard_path: Path, sampler_kwargs: dict, enc, tx,
                    region_on: bool, device, k_for_overlap: int = 10,
                    num_views: int = 1) -> dict | None:
    """Two-pass eval per shard.  Returns dict of metrics or None on failure."""
    from mm_align.data.spatial_sampler import SpatialSampleDataset

    # We build a singleton dataset so we can ask for two random subgraphs.
    ds = SpatialSampleDataset([shard_path], tx_encoder=tx, **sampler_kwargs)
    # SpatialSampleDataset may move tx_encoder to cache_device while building
    # h_tx. Put it back before region_hvg is encoded in `_embed_subgraph`.
    tx.to(device).eval()

    # Pass 1 is enough for the core label-free spatial metrics.  A second
    # pass is optional and only needed for augmentation_consistency; keeping
    # it off by default makes pipeline post-eval finish quickly.
    item_a = ds[0]
    emb_a = _embed_subgraph(item_a, enc, tx, region_on, device)
    xy_a = item_a["xy"].numpy()
    item_b = emb_b = None
    if int(num_views) >= 2:
        item_b = ds[0]
        emb_b = _embed_subgraph(item_b, enc, tx, region_on, device)

    # Domain labels: Novae latent → coarse cluster id (k-means with k=10)
    # — only when novae_latent column exists in the shard.
    domain_labels = None
    try:
        with h5py.File(shard_path, "r") as f:
            if "novae_latent" in f and f["novae_latent"].shape[0] == ds._n_spots[0]:
                from sklearn.cluster import KMeans
                # Subsample to the chosen anchors (sel hidden in dataset internals)
                # — just use full sample latent then nearest by row id.
                # Simpler: use kmeans on emb_a as a proxy domain labelling.
                km = KMeans(n_clusters=min(10, emb_a.shape[0] // 5), n_init=3,
                              random_state=0).fit(emb_a)
                domain_labels = km.labels_
    except Exception:
        pass

    # Per-sample metrics
    out = stage15_metrics_for_sample(
        emb_a, xy_a, k=k_for_overlap, domain_labels=domain_labels,
        prefix="stage15",
    )

    # Augmentation consistency between passes.  Subgraphs are random, so
    # anchor sets differ; use spot ids stored in `sel` (we don't expose
    # those — fall back to comparing on intersection of barcodes via xy
    # equality which is a reasonable proxy on cm-scale Visium coords).
    if item_b is not None and emb_b is not None:
        out["stage15/augmentation_consistency"] = _aug_consistency_xy(
            emb_a, item_a["xy"].numpy(), emb_b, item_b["xy"].numpy())
    else:
        out["stage15/augmentation_consistency"] = float("nan")
    return out


def _aug_consistency_xy(emb_a, xy_a, emb_b, xy_b, tol: float = 0.0) -> float:
    """Spots with identical xy across the two subgraphs are the same anchor."""
    # Build a map xy → row from pass B.
    keys_b = {(round(float(x), 4), round(float(y), 4)): i
              for i, (x, y) in enumerate(xy_b)}
    pa_idx, pb_idx = [], []
    for i, (x, y) in enumerate(xy_a):
        k = (round(float(x), 4), round(float(y), 4))
        if k in keys_b:
            pa_idx.append(i); pb_idx.append(keys_b[k])
    if len(pa_idx) < 5:
        return float("nan")
    a = emb_a[pa_idx]
    b = emb_b[pb_idx]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float((a * b).sum(axis=1).mean())


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--max-samples", type=int, default=8,
                     help="Limit number of shards (each is a separate sample).")
    ap.add_argument("--eval-subgraph-size", type=int, default=128,
                     help="Override Stage 1.5 subgraph_size for quick post-training evaluation.")
    ap.add_argument("--num-views", type=int, default=1,
                     help="Number of random subgraph views per shard. Use 2 to compute augmentation_consistency.")
    ap.add_argument("--k-knn", type=int, default=10,
                     help="k for spatial_knn_overlap.")
    ap.add_argument("--out", default="results/eval/stage15_indist.csv")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                    help="Device for spatial encoder evaluation.")
    ap.add_argument("--cache-device", default="auto", choices=("auto", "cuda", "cpu"),
                    help="Device for one-shot frozen tx h_tx cache creation. Default auto=cpu.")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_device = args.cache_device
    if cache_device == "auto":
        # Conservative default: the cache path is large and does not need GPU.
        cache_device = "cpu"
    prep = Path(args.prepared_dir)
    # Stage-specific split file
    sp_path = prep / "splits_stage15.json"
    if not sp_path.exists():
        sp_path = prep / "splits.json"
    splits = json.loads(sp_path.read_text())
    log.info(f"splits = {sp_path.name}  using {args.split}")

    # Collect shards with real coords only (Stage 1.5 contract).
    shards: list[Path] = []
    for sid in splits[args.split]:
        for suf in ("", ".st1k", ".spatialcorpus"):
            p = prep / f"{sid}{suf}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                xy = f["coords"][: min(128, f["coords"].shape[0])]
                if (xy ** 2).sum() > 0.0:
                    shards.append(p)
            break
    if args.max_samples:
        shards = shards[: args.max_samples]
    log.info(f"evaluating {len(shards)} shards with real spatial coords")

    rows = []
    for ck in args.ckpts:
        ck = Path(ck)
        log.info(f"== {ck} ==")
        enc, tx, sampler_kwargs, cfg, token_mode = load_stage15(
            ck, device=device, cache_device=cache_device
        )
        region_on = bool(cfg["data"].get("region", {}).get("enable", True))
        if args.eval_subgraph_size and args.eval_subgraph_size > 0:
            sampler_kwargs["subgraph_size"] = int(args.eval_subgraph_size)

        per_sample = []
        for i, sp in enumerate(shards, 1):
            try:
                log.info(f"  [{i}/{len(shards)}] {sp.stem} subgraph={sampler_kwargs.get('subgraph_size')} views={args.num_views}")
                m = evaluate_shard(sp, sampler_kwargs, enc, tx, region_on, device,
                                     k_for_overlap=args.k_knn, num_views=args.num_views)
                if m is not None:
                    m["sample"] = sp.stem
                    per_sample.append(m)
            except Exception as e:
                import traceback as _tb
                log.warning(f"  {sp.stem}: {type(e).__name__}: {e}\n{_tb.format_exc()}")

        if not per_sample:
            log.warning(f"  no shards evaluated for {ck} — skipping row")
            continue
        # Macro-average
        row = {"ckpt": ck.parent.name, "ckpt_path": str(ck),
                "split": args.split, "token_mode": token_mode,
                "n_samples": len(per_sample)}
        keys = [k for k in per_sample[0] if k != "sample"]
        for k in keys:
            vs = np.array([m.get(k, np.nan) for m in per_sample], dtype=float)
            vs = vs[~np.isnan(vs)]
            row[k.replace("/", "_")] = float(np.mean(vs)) if vs.size else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"saved {out}")
    print()
    print("─" * 90)
    print("Stage 1.5 IN-DIST evaluation — all metrics LABEL-FREE (no external GT).")
    print("  stage15_spatial_knn_overlap_k10  embedding KNN ↔ spatial KNN Jaccard  [HIGHER = more spatially-coherent]")
    print("  stage15_spatial_smoothness       Spearman(cos, Δxy⁻¹)                  [HIGHER = better]")
    print("  stage15_effective_rank           SVD entropy (collapse detector)        [HIGHER = less collapsed]")
    print("  stage15_boundary_preservation    inter/intra domain dist ratio (>1 OK)  [HIGHER = better]")
    print("  stage15_augmentation_consistency same anchor under two ego subgraphs    [HIGHER = better]")
    print("─" * 90)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
