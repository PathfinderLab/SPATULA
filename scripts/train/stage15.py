"""Stage 1.5 (Spatial Foundation) trainer — minimal scaffold.

Loads frozen Stage-1 tx_encoder + the per-shard SpatialSampleDataset, trains
a SpatialEncoder via Spatial Predictive JEPA, saves ckpt_spatial_best.pt.

See docs/design/stage15_spatial_jepa.md for the full design.

⚠ SCAFFOLD: ckpt-reload, AMP, multi-GPU sync, full eval loop will be added
   when this stage is run for real.  Single-GPU works as-is.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _render_curves(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(r["epoch"]) for r in history]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
        ax.plot(epochs, [r["train/loss"] for r in history], label="train/loss", lw=2)
        ax.plot(epochs, [r["val/loss"] for r in history], label="val/loss", lw=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Stage 1.5 Spatial JEPA")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        print(f"[stage1.5] WARNING: failed to render curves: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--tag", default="stage15_spatial_jepa")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-train-shards", type=int, default=None,
                     help="Cap Stage 1.5 train shard list for smoke runs (None = full).")
    ap.add_argument("--limit-val-shards", type=int, default=None,
                     help="Cap Stage 1.5 val shard list for smoke runs (None = full).")
    args = ap.parse_args()

    cfg = {
        "data":       load_yaml(args.data)["data"],
        "model":      load_yaml(args.model)["model"],
        "train":      load_yaml(args.train)["train"],
        "experiment": load_yaml(args.experiment)["experiment"],
    }
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg["train"]["output_dir"]) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", cfg)
    print(f"[stage1.5] run dir = {run_dir}")

    # ── 1. Load frozen Stage-1 tx_encoder ──────────────────────────────────
    # The Stage-1 ckpt format (see _save_tx_encoder_only in train.py):
    #   { "tx_encoder": <state_dict>,
    #     "cfg_tx":  {"model": {"transcriptomics": ..., "embed_dim": ...},
    #                  "data":  {"prepared_dir": ...}},
    #     "ckpt_format": "tx_encoder_only" }
    # build_tx_encoder(cfg) expects the full {"model": {...}, ...} cfg dict.
    import sys
    # scripts/train/stage15.py → repo root is parents[2]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mm_align.models.tx.factory import build_tx_encoder

    stage1_ckpt_path = Path(cfg["data"]["stage1_ckpt"])
    st = stage1_ckpt_path.stat()
    cache_src = f"{stage1_ckpt_path.resolve()}::{st.st_size}::{int(st.st_mtime)}"
    tx_cache_tag = hashlib.sha1(cache_src.encode("utf-8")).hexdigest()[:10]
    print(f"[stage1.5] tx cache tag = {tx_cache_tag} ({stage1_ckpt_path.name})")
    stage1_ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    cfg_tx = stage1_ckpt.get("cfg_tx")
    if cfg_tx is None:
        raise RuntimeError(
            f"Stage-1 ckpt {cfg['data']['stage1_ckpt']} is missing `cfg_tx` — "
            "re-export with the current train.py (_save_tx_encoder_only)."
        )
    tx_encoder = build_tx_encoder(cfg_tx)
    sd = stage1_ckpt.get("tx_encoder")
    if sd is None:
        raise RuntimeError("Stage-1 ckpt missing `tx_encoder` state_dict")
    missing, unexpected = tx_encoder.load_state_dict(sd, strict=False)
    print(f"[stage1.5] tx_encoder loaded — missing={len(missing)} unexpected={len(unexpected)}")
    for p in tx_encoder.parameters():
        p.requires_grad_(False)
    tx_encoder.eval().to(device)
    tx_dim = getattr(tx_encoder, "out_dim", None) or cfg_tx["model"].get("embed_dim", 512)

    # ── Carry Stage-1 input pipeline into Stage 1.5 ────────────────────────
    # gene_norm + vocab_clip: the encoder was trained on (e.g.) nonzero_z-
    # normalised hvg on a clipped vocab, so Stage 1.5 MUST apply the SAME
    # transform here (anchors AND region aggregates) before the frozen
    # tx_encoder forward — otherwise the encoder sees a distribution it
    # never saw at training time.
    #
    # Older Stage-1 ckpts persisted only `prepared_dir` in cfg_tx["data"],
    # so fall back to the run-dir's config.json (always written by train.py).
    gene_norm_cfg = cfg_tx.get("data", {}).get("gene_norm")
    vc = cfg_tx.get("data", {}).get("vocab_clip") or {}
    if gene_norm_cfg is None or not vc:
        from pathlib import Path as _P
        run_cfg = _P(cfg["data"]["stage1_ckpt"]).parent / "config.json"
        if run_cfg.exists():
            run_data = json.loads(run_cfg.read_text()).get("data", {})
            if gene_norm_cfg is None and run_data.get("gene_norm"):
                gene_norm_cfg = run_data["gene_norm"]
                print(f"[stage1.5] gene_norm not in ckpt; recovered from {run_cfg}")
            if (not vc) and run_data.get("vocab_clip"):
                vc = run_data["vocab_clip"]
                print(f"[stage1.5] vocab_clip not in ckpt; recovered from {run_cfg}")
    vocab_keep = None
    keep_path = vc.get("keep_indices_path") if isinstance(vc, dict) else None
    if keep_path:
        from pathlib import Path as _P
        if _P(keep_path).exists():
            vocab_keep = np.load(keep_path)
            print(f"[stage1.5] vocab_clip carried — keep {len(vocab_keep)} genes from {keep_path}")
        else:
            print(f"[stage1.5] WARNING: stage1 cfg points to {keep_path} (missing); "
                  "running without vocab_clip — verify your prepared shards match.")
    if gene_norm_cfg:
        print(f"[stage1.5] carrying Stage-1 gene_norm: mode={gene_norm_cfg.get('mode')} "
              f"stats={gene_norm_cfg.get('stats_path')}")
    else:
        print("[stage1.5] WARNING: NO gene_norm found in Stage-1 ckpt or config.json — "
              "the frozen encoder will see a different input distribution than at training!")

    # ── 2. Build SpatialEncoder + SpatialJEPAObjective ────────────────────
    from mm_align.models.spatial.encoder import SpatialEncoder
    from mm_align.objectives.spatial.jepa import SpatialJEPAObjective
    from mm_align.data.spatial_sampler import SpatialSampleDataset, spatial_collate

    mc = cfg["model"]["spatial"]
    rcfg = cfg["data"].get("region") or {}
    img_dim = 1536    # UNI feature dim (constant)
    region_on = bool(rcfg.get("enable", True))
    # Token mode: fused (single token per anchor) | separate (spot + region tokens).
    token_mode = str(mc.get("region_token_mode", "fused"))
    if token_mode == "separate" and not region_on:
        raise ValueError("model.spatial.region_token_mode='separate' requires "
                          "data.region.enable=true")
    student = SpatialEncoder(
        tx_dim=tx_dim, img_dim=img_dim,
        fuse_dim=mc.get("fuse_dim", 256),
        fuse_image=bool(cfg["data"].get("use_image", True)) and bool(mc.get("fuse_image", True)),
        fuse_region=region_on,
        token_mode=token_mode,
        arch=mc.get("arch", "kgnn"),
        n_layers=mc.get("n_layers", 3),
        n_heads=mc.get("n_heads", 4),
        dropout=mc.get("dropout", 0.1),
    ).to(device)
    ec = cfg["experiment"]["jepa"]
    obj = SpatialJEPAObjective(
        student,
        mask_ratio=ec["mask_ratio"],
        mask_strategy=ec.get("mask_strategy", "block"),
        mask_target=ec.get("mask_target", "spot"),
        block_size=int(ec.get("block_size", 8)),
        loss_kind=ec.get("loss", "smooth_l1"),
        smoothness_weight=ec.get("smoothness_weight", 0.0),
        ema_momentum=ec.get("ema_momentum", 0.999),
    ).to(device)

    # ── 3. Datasets ────────────────────────────────────────────────────────
    prep = Path(cfg["data"]["prepared_dir"])
    # Stage-specific split if present; else fall back to the global splits.json.
    splits_path = prep / "splits_stage15.json"
    if not splits_path.exists():
        splits_path = prep / "splits.json"
    splits = json.loads(splits_path.read_text())
    print(f"[stage1.5] splits file = {splits_path.name}")

    import h5py as _h5_check
    def _has_real_coords(shard: Path) -> bool:
        """Drop placeholder shards (coords identically zero) — spatial JEPA
        learns nothing from them and the ego-KNN degenerates."""
        with _h5_check.File(shard, "r") as f:
            if "coords" not in f:
                return False
            xy = f["coords"][: min(128, f["coords"].shape[0])]
            return bool((xy ** 2).sum() > 0.0)

    def _collect(sid_list: list[str]) -> tuple[list[Path], int]:
        kept, skipped = [], 0
        for sid in sid_list:
            for suf in ("", ".st1k", ".spatialcorpus"):
                p = prep / f"{sid}{suf}.h5"
                if not p.exists():
                    continue
                if _has_real_coords(p):
                    kept.append(p)
                else:
                    skipped += 1
                break
        return kept, skipped
    train_paths, train_skip = _collect(splits["train"])
    val_paths, val_skip = _collect(splits["val"])
    if args.limit_train_shards is not None and args.limit_train_shards > 0:
        print(f"[stage1.5] limit_train_shards={args.limit_train_shards} "
              f"(was {len(train_paths)})")
        train_paths = train_paths[: args.limit_train_shards]
    if args.limit_val_shards is not None and args.limit_val_shards > 0:
        print(f"[stage1.5] limit_val_shards={args.limit_val_shards} "
              f"(was {len(val_paths)})")
        val_paths = val_paths[: args.limit_val_shards]
    print(f"[stage1.5] train shards={len(train_paths)} (skipped {train_skip} zero-coord)  "
          f"val shards={len(val_paths)} (skipped {val_skip} zero-coord)")
    if not train_paths:
        raise RuntimeError("[stage1.5] no train shards with real spatial coords — "
                            "re-run prepare.py with ST1K/spatialcorpus coord support, "
                            "or point this trainer at a HEST-only split.")
    print(f"[stage1.5] caching h_tx per shard (one-time)...")
    sub_n = int(cfg["data"].get("subgraph_size", 256))
    gcfg = cfg["data"]["graph"]
    k = int(gcfg.get("k", 8))
    graph_kind = str(gcfg.get("kind", "knn"))
    radius_px = float(gcfg.get("radius_px", 600))
    use_img = bool(cfg["data"].get("use_image", True))
    tx_encode_batch = int(cfg["data"].get("tx_encode_batch", 256))
    tx_cache_device = str(cfg["data"].get("tx_cache_device", "cpu"))
    if tx_cache_device == "auto":
        tx_cache_device = str(device)
    print(f"[stage1.5] tx_encoder micro-batch={tx_encode_batch} "
          f"(cache_device={tx_cache_device}, train_device={device})")
    ds_kwargs = dict(k=k, subgraph_size=sub_n, device=tx_cache_device,
                      encode_batch=tx_encode_batch,
                      fuse_image=use_img, tx_dim=tx_dim,
                      graph_kind=graph_kind, radius_px=radius_px,
                      subgraph_kind=str(cfg["data"].get("subgraph_kind", "random")),
                      # Stage-1 input pipeline reuse — apply the SAME normalisation
                      # and vocab slice that the frozen encoder trained on.
                      gene_norm_cfg=gene_norm_cfg,
                      vocab_keep_indices=vocab_keep,
                      # Region knobs (Stage 1.5 "anchor + neighbors" tokens)
                      region_enable=region_on,
                      region_k=int(rcfg.get("k", k)),
                      region_tx_agg=str(rcfg.get("tx_agg", "mean")),
                      region_img_pool=str(rcfg.get("img_pool", "mean")),
                      region_weighted_sigma=float(rcfg.get("weighted_sigma", 1.0)),
                      tx_cache_tag=tx_cache_tag,
                      # JEPA semantics: when masking spot tokens, region must
                      # not see the anchor's own RNA/image — keep False unless
                      # explicitly overridden in HyperST-style fused mode.
                      region_include_anchor=bool(rcfg.get("include_anchor", False)),
                      )
    train_ds = SpatialSampleDataset(train_paths, tx_encoder, **ds_kwargs)
    val_ds = SpatialSampleDataset(val_paths, tx_encoder, **ds_kwargs)
    # Dataset construction may move the frozen encoder to the cache device
    # (CPU by default for robust full-shard pre-encoding). Move it back for
    # on-the-fly region_hvg embedding during GPU training.
    tx_encoder.eval().to(device)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=cfg["train"].get("num_workers", 4),
                              collate_fn=spatial_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=2,
                             collate_fn=spatial_collate)

    # ── 4. Region tx forward (frozen) — convert region_hvg to h_region_tx ─
    # The dataset hands us aggregated `region_hvg` (n_anchors, hvg_dim).  The
    # SpatialEncoder fuser wants a (n_anchors, tx_dim) latent in the same
    # space as `h_tx`, so we run it through the SAME frozen Stage-1 tx
    # encoder.  This is cheap because tx_encoder is frozen + we don't store
    # gradients.
    @torch.no_grad()
    def _embed_region(b: dict) -> dict:
        if not region_on or "region_hvg" not in b:
            return b
        rhv = b["region_hvg"]                         # (N, hvg_dim)
        outs = []
        for r0 in range(0, rhv.shape[0], tx_encode_batch):
            block = rhv[r0:r0 + tx_encode_batch]
            outs.append(tx_encoder(novae_latent=None, hvg=block)["h_tx"])
        b["h_region_tx"] = torch.cat(outs, dim=0)
        return b

    # ── 5. Optim + train loop ──────────────────────────────────────────────
    opt = torch.optim.AdamW(student.parameters(),
                             lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"].get("weight_decay", 0.05))
    epochs = int(cfg["train"]["epochs"])
    print(f"[stage1.5] training {epochs} epochs, lr={cfg['train']['lr']}, "
          f"region={'on' if region_on else 'off'} "
          f"(tx_agg={rcfg.get('tx_agg','mean')}, img_pool={rcfg.get('img_pool','mean')})")
    best_val = float("inf")
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        student.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            batch = _embed_region(batch)
            loss, log = obj(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),
                                            cfg["train"].get("max_grad_norm", 1.0))
            opt.step()
            obj.on_after_step()
            tot += float(loss.item()); n += 1
        train_loss = tot / max(n, 1)

        # Val
        student.eval()
        with torch.no_grad():
            vtot, vn = 0.0, 0
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                batch = _embed_region(batch)
                loss, _ = obj(batch)
                vtot += float(loss.item()); vn += 1
            val_loss = vtot / max(vn, 1)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        row = {
            "epoch": epoch,
            "train/loss": float(train_loss),
            "train/n_batches": int(n),
            "val/loss": float(val_loss),
            "val/n_batches": int(vn),
            "stage15/best_val_loss": float(best_val),
            "stage15/is_best": float(is_best),
            "lr": float(cfg["train"]["lr"]),
        }
        history.append(row)
        _write_json(run_dir / "history.json", history)
        _write_json(run_dir / "val_history.json", history)
        _render_curves(history, run_dir / "loss_curve.png")
        _render_curves(history, run_dir / "metric_curves.png")
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        torch.save({
            "spatial_state_dict": student.state_dict(),
            "spatial_config": cfg["model"]["spatial"],
            "cfg": cfg,
            "stage1_ckpt": cfg["data"].get("stage1_ckpt"),
            "epoch": epoch, "val_loss": val_loss,
        }, run_dir / "ckpt_spatial_last.pt")

        if is_best:
            torch.save({
                "spatial_state_dict": student.state_dict(),
                "spatial_config": cfg["model"]["spatial"],
                "cfg": cfg,
                "stage1_ckpt": cfg["data"].get("stage1_ckpt"),
                "epoch": epoch, "val_loss": val_loss,
            }, run_dir / "ckpt_spatial_best.pt")
            print(f"  ↑ saved ckpt_spatial_best.pt")

    print(f"[stage1.5] done.  best val_loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
