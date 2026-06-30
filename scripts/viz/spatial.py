"""Visualise a trained model's per-spot embeddings + gene predictions on the
H&E image of one HEST sample, using `scanpy.pl.spatial`.

Pipeline:
  1. Load a ckpt (train.py / Stage-1 / Stage-2 — any flavor).
  2. Pick one sample (`--sample-id INT1`), load its shard +
     `/data/hest/thumbnails/<id>_downscaled_fullres.jpeg` +
     `/data/hest/metadata/<id>.json` for fullres dims.
  3. Run model.forward per spot, collect:
        h_image / h_tx / z_shared          (per-spot embeddings)
        gene_recon_from_image              (predicted HVG, B × 2048)
  4. Build an AnnData:
        X      = predicted HVG (or observed `hvg_log` if `--use-observed`)
        obsm["spatial"] = fullres pixel coords
        uns["spatial"][sample_id] = {image: thumbnail, scalef: thumbnail/fullres}
        obs["leiden_zshared" / "kmeans_K"] = clusterings on shared embedding
  5. Render with `sc.pl.spatial(..., color=...)`:
        - one PNG with leiden + kmeans overlays (cluster grid).
        - one PNG per gene in `--genes` (image-overlay predicted expression).
        - if `--use-observed`, observed values are plotted alongside predicted.

Usage:
    python scripts/viz/spatial.py \
        --ckpt results/runs/full_human_jepa_uni_lora/ckpt_best.pt \
        --sample-id MEND140 \
        --genes ALB COL1A1 IGFBP1 \
        --out-dir results/viz

Defaults:
    leiden_resolution = 1.0
    kmeans_k          = 8
    embedding         = z_shared   (use h_image | h_tx | z_image | z_tx to switch)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import load_config, get_logger
from mm_align.data import build_dataset_from_split, pad_collate
from mm_align.models import MMAligner

log = get_logger("viz_spatial")


# ────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") if isinstance(state, dict) else None
    if cfg is None:
        raise SystemExit(f"ckpt {ckpt_path} has no 'cfg' — re-train or pass --cfg manually.")
    model = MMAligner(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


def load_thumbnail(sample_id: str, hest_root: Path) -> tuple[np.ndarray, dict]:
    thumb_path = hest_root / "thumbnails" / f"{sample_id}_downscaled_fullres.jpeg"
    meta_path = hest_root / "metadata" / f"{sample_id}.json"
    img = np.asarray(Image.open(thumb_path).convert("RGB"))
    meta = json.loads(meta_path.read_text())
    return img, meta


def encode_one_sample(model, ds, sample_idx_target: int, device,
                      batch_size: int = 256) -> dict:
    """Run model.forward over every spot of one shard.  Returns numpy arrays
    keyed by ('h_image','h_tx','z_image','z_tx','z_shared',
              'gene_recon_from_image','hvg','coords','spot_idx')."""
    shard = ds.shards[sample_idx_target]
    start = ds._starts[sample_idx_target]
    stop = ds._starts[sample_idx_target + 1]
    indices = list(range(start, stop))

    from torch.utils.data import Subset
    sub = Subset(ds, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=pad_collate)

    bufs: dict[str, list] = {k: [] for k in
        ("h_image", "h_tx", "z_image", "z_tx", "gene_recon_from_image", "hvg")}
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
            out = model(b)
            for k in ("h_image", "h_tx", "z_image", "z_tx"):
                bufs[k].append(out[k].float().cpu().numpy())
            if "gene_recon_from_image" in out and out["gene_recon_from_image"] is not None:
                bufs["gene_recon_from_image"].append(out["gene_recon_from_image"].float().cpu().numpy())
            if "hvg" in b:
                bufs["hvg"].append(b["hvg"].float().cpu().numpy())

    res = {k: (np.concatenate(v) if v else None) for k, v in bufs.items()}
    res["z_shared"] = 0.5 * (res["z_image"] + res["z_tx"])
    # Coordinates from the raw shard file.
    with h5py.File(shard.path, "r") as f:
        res["coords"] = f["coords"][:].astype(np.float32)        # (N, 2) fullres px
    return res


# ────────────────────────────────────────────────────────────────────────
# AnnData construction
# ────────────────────────────────────────────────────────────────────────

def build_anndata(pool: dict, gene_names: list[str], sample_id: str,
                   thumb_img: np.ndarray, meta: dict,
                   use_observed: bool,
                   leiden_resolution: float,
                   kmeans_k: int,
                   embedding_key: str = "z_shared"):
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    # X = predicted HVG (default) or observed
    if use_observed and pool.get("hvg") is not None:
        X = pool["hvg"]
    elif pool.get("gene_recon_from_image") is not None:
        X = pool["gene_recon_from_image"]
    elif pool.get("hvg") is not None:
        X = pool["hvg"]
    else:
        raise SystemExit("No HVG predictions or observations available.")

    n_spots, n_genes = X.shape
    assert n_genes == len(gene_names), (
        f"HVG dim mismatch: X has {n_genes} genes but vocab has {len(gene_names)}")

    var = pd.DataFrame(index=gene_names)
    obs = pd.DataFrame(index=[f"spot_{i:06d}" for i in range(n_spots)])
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)

    adata.obsm["spatial"] = pool["coords"]            # fullres pixel
    for k in ("h_image", "h_tx", "z_image", "z_tx", "z_shared"):
        adata.obsm[k] = pool[k]

    # ── Clusterings on the chosen embedding ──
    Z = pool[embedding_key]
    # KMeans
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=0).fit(Z)
    adata.obs[f"kmeans_k{kmeans_k}"] = pd.Categorical(km.labels_.astype(str))
    # Leiden via scanpy (needs nearest-neighbour graph built on the chosen embedding)
    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
    sc.tl.leiden(adata, resolution=leiden_resolution,
                  key_added=f"leiden_r{leiden_resolution:g}", flavor="igraph", n_iterations=2,
                  directed=False)

    # ── Spatial metadata for sc.pl.spatial ──
    fullres_w = float(meta["fullres_width"])
    thumb_h, thumb_w = thumb_img.shape[:2]
    scalef = thumb_w / fullres_w
    adata.uns["spatial"] = {
        sample_id: {
            "images": {"hires": thumb_img, "lowres": thumb_img},
            "scalefactors": {
                "tissue_hires_scalef": scalef,
                "tissue_lowres_scalef": scalef,
                "spot_diameter_fullres": float(meta.get("spot_diameter", 55.0)),
                "fiducial_diameter_fullres": float(meta.get("spot_diameter", 55.0)),
            },
            "metadata": {"source_image_path": meta.get("image_filename", "")},
        }
    }
    return adata


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────

def render_cluster_panel(adata, sample_id: str, leiden_key: str, kmeans_key: str,
                          out_path: Path, spot_size: float = 6.0):
    import scanpy as sc
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.spatial(adata, color=leiden_key, library_id=sample_id, ax=axes[0],
                  show=False, size=spot_size, title=f"{sample_id} — {leiden_key}")
    sc.pl.spatial(adata, color=kmeans_key, library_id=sample_id, ax=axes[1],
                  show=False, size=spot_size, title=f"{sample_id} — {kmeans_key}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_gene_panel(adata, sample_id: str, genes: list[str], out_path: Path,
                       use_observed: bool, spot_size: float = 6.0):
    import scanpy as sc
    import matplotlib.pyplot as plt
    missing = [g for g in genes if g not in adata.var_names]
    genes = [g for g in genes if g in adata.var_names]
    if missing:
        log.warning(f"genes not in HVG vocab (skipping): {missing}")
    if not genes:
        log.warning("no genes to plot.")
        return
    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    title_prefix = "OBS" if use_observed else "PRED"
    for ax, g in zip(axes[0], genes):
        sc.pl.spatial(adata, color=g, library_id=sample_id, ax=ax, show=False,
                      size=spot_size, cmap="viridis", title=f"{sample_id} {title_prefix} {g}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_predicted_vs_observed(adata_pred, adata_obs, sample_id: str,
                                  genes: list[str], out_path: Path,
                                  spot_size: float = 6.0,
                                  scale_ref: str = "union"):
    """Side-by-side OBS (top row) vs PRED (bottom row).  Per-gene colorbar is
    shared between the two panels so colors are directly comparable.
    `scale_ref`:
      - "union" (default) — vmin/vmax = min/max over obs ∪ pred
      - "pred"            — vmin/vmax taken from pred only
      - "obs"             — vmin/vmax taken from obs only
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    genes = [g for g in genes if g in adata_pred.var_names and g in adata_obs.var_names]
    if not genes:
        return
    n = len(genes)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), squeeze=False)
    for j, g in enumerate(genes):
        v_pred = adata_pred[:, g].X
        v_obs = adata_obs[:, g].X
        # densify if sparse
        v_pred = v_pred.toarray() if hasattr(v_pred, "toarray") else np.asarray(v_pred)
        v_obs = v_obs.toarray() if hasattr(v_obs, "toarray") else np.asarray(v_obs)
        if scale_ref == "pred":
            vmin, vmax = float(v_pred.min()), float(v_pred.max())
        elif scale_ref == "obs":
            vmin, vmax = float(v_obs.min()), float(v_obs.max())
        else:  # union
            vmin = float(min(v_pred.min(), v_obs.min()))
            vmax = float(max(v_pred.max(), v_obs.max()))
        sc.pl.spatial(adata_obs, color=g, library_id=sample_id, ax=axes[0, j],
                      show=False, size=spot_size, cmap="viridis",
                      vmin=vmin, vmax=vmax,
                      title=f"OBS {g}  [{vmin:.2f},{vmax:.2f}]")
        sc.pl.spatial(adata_pred, color=g, library_id=sample_id, ax=axes[1, j],
                      show=False, size=spot_size, cmap="viridis",
                      vmin=vmin, vmax=vmax,
                      title=f"PRED {g}  [{vmin:.2f},{vmax:.2f}]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_*.pt produced by train.py")
    ap.add_argument("--sample-id", required=True,
                    help="HEST sample id (e.g. MEND140) — must be present in splits.json")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="Gene symbols to overlay (must be in HVG vocab).  "
                         "Default: top-10 best-predicted genes inferred from per-spot variance.")
    ap.add_argument("--embedding", default="z_shared",
                    choices=["h_image", "h_tx", "z_image", "z_tx", "z_shared"],
                    help="Which per-spot embedding to cluster on.")
    ap.add_argument("--leiden-resolution", type=float, default=1.0)
    ap.add_argument("--kmeans-k", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--hest-root", default="/data/hest",
                    help="Root of HEST raw data (for thumbnails + metadata).")
    ap.add_argument("--spot-size", type=float, default=3.0,
                    help="Scanpy spot-size multiplier (×spot_diameter_fullres). "
                         "Default 3.0 — increase for sparser slides, decrease if spots overlap.")
    ap.add_argument("--scale-ref", default="union", choices=["union", "pred", "obs"],
                    help="Colorbar reference for the pred-vs-obs panel.  "
                         "`union` (default) uses min/max over both; `pred` uses predicted "
                         "value range only; `obs` uses observed only.")
    ap.add_argument("--use-observed", action="store_true",
                    help="Use observed `hvg_log` (instead of `gene_recon_from_image`) for "
                         "the gene overlays.  Useful for predicted-vs-observed comparisons; "
                         "the script will emit BOTH panels when this flag is set together "
                         "with the default predicted overlay.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    out_dir = Path(args.out_dir or (ckpt.parent / "viz" / args.sample_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"out_dir = {out_dir}")

    # ── Model + cfg from ckpt ──
    model, cfg = load_model(ckpt, device)

    # ── Build dataset over just this one sample ──
    prepared = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared / "splits.json").read_text())
    all_ids = splits["train"] + splits["val"] + splits["test"]
    if args.sample_id not in all_ids:
        raise SystemExit(f"sample-id {args.sample_id!r} not in splits.json "
                         f"({len(all_ids)} samples). pick one from train/val/test.")
    ds_kwargs = dict(
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
        image_mode=cfg["data"]["image_mode"],
        hest_patch_dir=cfg["data"]["hest_patch_dir"],
        gene_norm_cfg=cfg["data"].get("gene_norm"),
    )
    ds = build_dataset_from_split(prepared, [args.sample_id], **ds_kwargs)
    log.info(f"loaded sample={args.sample_id} with {ds.total} spots")

    # ── Encode every spot ──
    pool = encode_one_sample(model, ds, sample_idx_target=0, device=device)
    log.info(f"encoded: z_shared={pool['z_shared'].shape}, "
             f"coords={pool['coords'].shape}, "
             f"gene_pred={None if pool['gene_recon_from_image'] is None else pool['gene_recon_from_image'].shape}, "
             f"hvg_obs={None if pool.get('hvg') is None else pool['hvg'].shape}")

    # ── Thumbnail + metadata for sc.pl.spatial ──
    thumb_img, meta = load_thumbnail(args.sample_id, Path(args.hest_root))
    log.info(f"thumbnail {thumb_img.shape}, fullres "
             f"{meta['fullres_width']}x{meta['fullres_height']}")

    # ── Gene name list ──
    gene_names = json.loads((prepared / "hvg_vocab.json").read_text())

    # ── Build the AnnData (predicted X) ──
    adata_pred = build_anndata(
        pool, gene_names, args.sample_id, thumb_img, meta,
        use_observed=False, leiden_resolution=args.leiden_resolution,
        kmeans_k=args.kmeans_k, embedding_key=args.embedding,
    )
    adata_pred.write_h5ad(out_dir / "adata_pred.h5ad")
    log.info(f"wrote adata_pred.h5ad ({adata_pred.shape})")

    # Optional: build a parallel observed AnnData for pred-vs-obs panels.
    adata_obs = None
    if args.use_observed and pool.get("hvg") is not None:
        adata_obs = build_anndata(
            pool, gene_names, args.sample_id, thumb_img, meta,
            use_observed=True, leiden_resolution=args.leiden_resolution,
            kmeans_k=args.kmeans_k, embedding_key=args.embedding,
        )

    # ── Pick genes to plot ──
    if args.genes:
        genes_to_plot = list(args.genes)
    else:
        # Auto: top-10 genes by spot variance in the predicted matrix
        # (proxy for "the model is putting structure on these").
        var = adata_pred.X.var(axis=0)
        top = np.argsort(-var)[:10]
        genes_to_plot = [gene_names[i] for i in top]
    log.info(f"plotting genes: {genes_to_plot}")

    # ── Render ──
    leiden_key = f"leiden_r{args.leiden_resolution:g}"
    kmeans_key = f"kmeans_k{args.kmeans_k}"
    render_cluster_panel(adata_pred, args.sample_id,
                         leiden_key=leiden_key, kmeans_key=kmeans_key,
                         out_path=out_dir / "fig_clusters.png",
                         spot_size=args.spot_size)
    log.info(f"wrote fig_clusters.png")

    render_gene_panel(adata_pred, args.sample_id, genes_to_plot,
                       out_path=out_dir / "fig_genes_predicted.png",
                       use_observed=False, spot_size=args.spot_size)
    log.info("wrote fig_genes_predicted.png")

    if adata_obs is not None:
        render_gene_panel(adata_obs, args.sample_id, genes_to_plot,
                           out_path=out_dir / "fig_genes_observed.png",
                           use_observed=True, spot_size=args.spot_size)
        render_predicted_vs_observed(adata_pred, adata_obs, args.sample_id,
                                      genes_to_plot,
                                      out_path=out_dir / "fig_genes_pred_vs_obs.png",
                                      spot_size=args.spot_size,
                                      scale_ref=args.scale_ref)
        log.info("wrote fig_genes_observed.png + fig_genes_pred_vs_obs.png")

    log.info(f"done — see {out_dir}/")


if __name__ == "__main__":
    main()
