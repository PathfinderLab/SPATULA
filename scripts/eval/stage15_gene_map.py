"""Stage-1.5 spatial gene-map evaluation.

This evaluator freezes a trained Stage-1.5 SpatialEncoder, learns a small
Ridge probe from spatial embeddings to selected gene expression on train
spots, then reports test-sample spatial maps:

    GT expression map vs predicted expression map + Spearman SCC.

It is intentionally a TEST/downstream evaluator, not a validation loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.data.region import build_neighbor_index, aggregate_region_img, aggregate_region_tx
from mm_align.data.spatial_sampler import build_edges
from mm_align.evaluation.stage1_benchmarks import _spearman
from mm_align.evaluation.gene_imputation_metrics import (
    pearson_1d as _pearson,
    rmse_zscore as _rmse_zscore,
    ssim_1d as _ssim_1d,
    jsd_1d as _jsd_1d,
)
from mm_align.models.spatial.encoder import SpatialEncoder
from mm_align.models.tx.factory import build_tx_encoder
from mm_align.utils import get_logger

log = get_logger("eval_stage15_gene_map")


def _rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return order / max(1, len(x) - 1)


def _spatial_pattern_score(coords: np.ndarray, expr: np.ndarray, *, k: int = 8) -> np.ndarray:
    """Cheap GT-only spatial pattern score per gene.

    The score prefers genes that are expressed in a non-trivial fraction of
    spots, have variance, and are locally smooth over tissue coordinates.  It
    is only used to choose qualitative HEST gene-map examples; it does not use
    model predictions.
    """
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(expr, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 8:
        return np.zeros((x.shape[1] if x.ndim == 2 else 0,), dtype=np.float32)
    nz = (x > 0).mean(axis=0)
    var = np.nanvar(x, axis=0)
    kk = min(k + 1, x.shape[0])
    idx = NearestNeighbors(n_neighbors=kk).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    neigh = np.nanmean(x[idx], axis=1)
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    nc = neigh - np.nanmean(neigh, axis=0, keepdims=True)
    denom = np.sqrt(np.nansum(xc * xc, axis=0) * np.nansum(nc * nc, axis=0)) + 1e-8
    local_corr = np.nansum(xc * nc, axis=0) / denom
    # Penalize ubiquitous or ultra-sparse genes; both make maps visually weak.
    prevalence = np.clip((nz - 0.03) / 0.20, 0.0, 1.0) * np.clip((0.95 - nz) / 0.30, 0.0, 1.0)
    return (np.maximum(local_corr, 0.0) * np.log1p(var) * prevalence).astype(np.float32)


def _auto_select_spatial_genes(paths: list[Path], vocab_keep: np.ndarray | None,
                               genes_eff: list[str], n: int, *,
                               exclude: set[str] | None = None) -> list[str]:
    if n <= 0 or not paths:
        return []
    scores = np.zeros(len(genes_eff), dtype=np.float64)
    counts = np.zeros(len(genes_eff), dtype=np.float64)
    exclude = {g.upper() for g in (exclude or set())}
    for p in paths:
        try:
            _full, hvg_eff, coords, _uni = _load_shard_arrays(p, vocab_keep)
            sc = _spatial_pattern_score(coords.astype(np.float32), hvg_eff.astype(np.float32))
            if sc.shape[0] == scores.shape[0]:
                scores += sc
                counts += np.isfinite(sc).astype(np.float64)
        except Exception as e:
            log.warning(f"auto gene scoring skipped {p.name}: {e}")
    scores = scores / np.maximum(counts, 1.0)
    for i, g in enumerate(genes_eff):
        if g.upper() in exclude:
            scores[i] = -np.inf
    order = np.argsort(-scores)
    out = [genes_eff[int(i)] for i in order if np.isfinite(scores[int(i)]) and scores[int(i)] > 0]
    return out[:n]


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_stage1_tx(stage1_ckpt: Path, device: str, tx_pooling_mode: str | None = None):
    sd = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    cfg_tx = sd.get("cfg_tx") or sd.get("cfg")
    if cfg_tx is None:
        cfg_json = stage1_ckpt.parent / "config.json"
        if not cfg_json.exists():
            raise RuntimeError(f"cannot recover cfg_tx from {stage1_ckpt}")
        cfg_tx = json.loads(cfg_json.read_text())
    cfg_tx = copy.deepcopy(cfg_tx)
    if tx_pooling_mode and tx_pooling_mode not in ("", "ckpt"):
        gcfg = (cfg_tx.setdefault("model", {})
                      .setdefault("transcriptomics", {})
                      .setdefault("top_hvg_gene", {}))
        gcfg["pooling_mode"] = tx_pooling_mode
    tx_encoder = build_tx_encoder(cfg_tx)
    state = sd.get("tx_encoder") or sd.get("tx_state_dict") or sd.get("model")
    if state is None:
        raise RuntimeError(f"{stage1_ckpt} has no tx encoder state")
    tx_encoder.load_state_dict(state, strict=False)
    tx_encoder.eval().to(device)
    for p in tx_encoder.parameters():
        p.requires_grad_(False)

    data_cfg = dict(cfg_tx.get("data", {}))
    cfg_json = stage1_ckpt.parent / "config.json"
    if cfg_json.exists():
        fallback = json.loads(cfg_json.read_text()).get("data", {})
        for k, v in fallback.items():
            data_cfg.setdefault(k, v)
    vc = data_cfg.get("vocab_clip") or {}
    keep_path = vc.get("keep_indices_path") if isinstance(vc, dict) else None
    vocab_keep = np.load(keep_path) if keep_path and Path(keep_path).exists() else None
    gene_norm_cfg = data_cfg.get("gene_norm")
    tx_dim = getattr(tx_encoder, "out_dim", None) or cfg_tx.get("model", {}).get("embed_dim", 512)
    return tx_encoder, tx_dim, vocab_keep, gene_norm_cfg


@torch.no_grad()
def _embed_tx(tx_encoder, hvg_norm: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    outs = []
    for r0 in range(0, hvg_norm.shape[0], batch):
        xb = torch.from_numpy(hvg_norm[r0:r0 + batch]).to(device)
        outs.append(tx_encoder(novae_latent=None, hvg=xb)["h_tx"].detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def _load_gene_names(prepared: Path, vocab_keep: np.ndarray | None) -> list[str]:
    genes = json.loads((prepared / "hvg_vocab.json").read_text())
    if vocab_keep is not None:
        genes = [genes[int(i)] for i in vocab_keep]
    return genes


def _resolve_shard(prepared: Path, sid: str) -> Path | None:
    for suf in ("", ".st1k", ".spatialcorpus"):
        p = prepared / f"{sid}{suf}.h5"
        if p.exists():
            return p
    return None


def _split_paths(prepared: Path, split: str, source: str = "hest") -> list[Path]:
    splits = json.loads((prepared / "splits.json").read_text())
    out = []
    for sid in splits.get(split, []):
        if source == "hest":
            p = prepared / f"{sid}.h5"
            if p.exists():
                out.append(p)
        else:
            p = _resolve_shard(prepared, sid)
            if p is not None:
                out.append(p)
    return out


def _load_shard_arrays(path: Path, vocab_keep: np.ndarray | None):
    with h5py.File(path, "r") as f:
        hvg_raw_full = f["hvg_log"][:].astype(np.float32)
        coords = f["coords"][:].astype(np.float32)
        uni = f["uni_feat"][:].astype(np.float32)
    hvg_raw = hvg_raw_full[:, vocab_keep] if vocab_keep is not None else hvg_raw_full
    return hvg_raw_full, hvg_raw.astype(np.float32, copy=False), coords, uni


def _normalise_xy(coords: np.ndarray) -> np.ndarray:
    xy = coords.astype(np.float32, copy=True)
    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = max(1e-6, float(np.linalg.norm(xy, axis=1).max()))
    return xy / scale


@torch.no_grad()
def encode_spatial_sample(path: Path,
                          tx_encoder,
                          spatial: SpatialEncoder,
                          normalizer: GeneNormalizer,
                          vocab_keep: np.ndarray | None,
                          stage15_data: dict,
                          device: str,
                          tx_batch: int = 512) -> dict:
    hvg_raw_full, hvg_raw_eff, coords, uni = _load_shard_arrays(path, vocab_keep)
    hvg_norm = normalizer.apply_np(hvg_raw_eff).astype(np.float32, copy=False)
    h_tx = _embed_tx(tx_encoder, hvg_norm, device=device, batch=tx_batch)
    xy = _normalise_xy(coords)
    gcfg = stage15_data.get("graph", {})
    k = int(gcfg.get("k", 8))
    kind = str(gcfg.get("kind", "knn"))
    radius = float(gcfg.get("radius_px", 600.0))
    # coords are normalized; follow Stage1.5 dataset's effective radius logic.
    coord_scale = max(1e-6, float(np.linalg.norm(coords - coords.mean(axis=0, keepdims=True), axis=1).max()))
    edge_index = build_edges(xy, kind=kind, k=k, radius=radius / coord_scale)

    region_cfg = stage15_data.get("region", {}) or {}
    region_on = bool(region_cfg.get("enable", True))
    h_region_tx = None
    h_region_img = None
    if region_on:
        rk = int(region_cfg.get("k", k))
        nbr = build_neighbor_index(edge_index, n_nodes=xy.shape[0], k=rk, pad_with_self=True)
        h_region_img = aggregate_region_img(uni, nbr, kind=str(region_cfg.get("img_pool", "mean")))
        tx_agg = str(region_cfg.get("tx_agg", "mean"))
        if tx_agg == "weighted":
            d = np.linalg.norm(xy[nbr] - xy[:, None, :], axis=-1).astype(np.float32)
            r_hvg = aggregate_region_tx(
                hvg_raw_eff, nbr, kind="weighted", distances=d,
                sigma=float(region_cfg.get("weighted_sigma", 1.0)),
            )
        else:
            r_hvg = aggregate_region_tx(hvg_raw_eff, nbr, kind=tx_agg)
        r_norm = normalizer.apply_np(r_hvg).astype(np.float32, copy=False)
        h_region_tx = _embed_tx(tx_encoder, r_norm, device=device, batch=tx_batch)

    spatial.eval().to(device)
    z = spatial(
        torch.from_numpy(h_tx).to(device),
        torch.from_numpy(uni).to(device),
        torch.from_numpy(xy).to(device),
        torch.from_numpy(edge_index).long().to(device),
        h_region_tx=(torch.from_numpy(h_region_tx).to(device) if h_region_tx is not None else None),
        h_region_img=(torch.from_numpy(h_region_img).to(device) if h_region_img is not None else None),
    ).detach().cpu().numpy().astype(np.float32)
    if getattr(spatial, "token_mode", "fused") == "separate":
        z = z[: h_tx.shape[0]]
    return {
        "sample_id": path.stem.split(".")[0],
        "z": z,
        "hvg_raw_full": hvg_raw_full,
        "hvg_eff": hvg_raw_eff,
        "coords": coords,
    }


def _plot_gene_map(coords: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                   sample_id: str, gene: str, metrics: dict[str, float], out: Path) -> None:
    """4-panel: GT / Pred / |GT - Pred| (z-score space) / |rank(GT) - rank(Pred)|.

    Z-score on the first two panels uses shared limits so the colours are
    directly comparable.  The Abs-diff panels reveal where the spatial
    encoder mispredicts.  Rank-diff is robust to gain mismatch.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    # GT & Pred share viridis colour limits.
    vmin = float(np.nanpercentile(np.concatenate([gt, pred]), 1))
    vmax = float(np.nanpercentile(np.concatenate([gt, pred]), 99))
    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=gt, s=7,
                           cmap="viridis", vmin=vmin, vmax=vmax)
    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=pred, s=7,
                           cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.02)
    # |z(gt) - z(pred)| — magma highlights worst-error regions.
    def _z(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-12 else 1.0)
    abs_diff = np.abs(_z(gt) - _z(pred))
    sc2 = axes[2].scatter(coords[:, 0], coords[:, 1], c=abs_diff, s=7, cmap="magma")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.02)
    # |rank(GT) - rank(Pred)| — robust to scale.
    rank_diff = np.abs(_rank_norm(gt) - _rank_norm(pred))
    sc3 = axes[3].scatter(coords[:, 0], coords[:, 1], c=rank_diff, s=7, cmap="magma")
    fig.colorbar(sc3, ax=axes[3], fraction=0.046, pad=0.02)
    for ax, title in zip(axes, ["GT", "Pred", "|z(GT) - z(Pred)|", "|rank(GT) - rank(Pred)|"]):
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    parts = [f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float))]
    fig.suptitle(f"{sample_id} | {gene}  ({', '.join(parts)})")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)




def _write_hest_style_summary(spot_tables: list[pd.DataFrame], out_dir: Path) -> None:
    """Write metric summaries with explicit HEST-like semantics.

    Current per-row metrics in gene_map_scc.csv are per_gene_per_sample
    spatial-map correlations across spots.  This summary additionally reports:
      - genewise PCC/SCC: corr over all spots pooled across samples for each gene
      - spotwise SCC/PCC: corr over selected genes within each spot, then mean
      - overall PCC/SCC: corr over all (spot, gene) values flattened
    """
    if not spot_tables:
        return
    df = pd.concat(spot_tables, axis=0, ignore_index=True)
    rows = []
    for gene, g in df.groupby("gene"):
        rows.append({
            "unit": "gene",
            "id": gene,
            "pcc_genewise": _pearson(g["pred"].to_numpy(), g["gt"].to_numpy()),
            "scc_genewise": _spearman(g["pred"].to_numpy(), g["gt"].to_numpy()),
            "n_values": int(len(g)),
        })
    for (sample, spot_id), g in df.groupby(["sample_id", "spot_index"]):
        if len(g) < 2:
            continue
        rows.append({
            "unit": "spot",
            "id": f"{sample}:{spot_id}",
            "pcc_spotwise": _pearson(g["pred"].to_numpy(), g["gt"].to_numpy()),
            "scc_spotwise": _spearman(g["pred"].to_numpy(), g["gt"].to_numpy()),
            "n_values": int(len(g)),
        })
    rows.append({
        "unit": "overall_flat",
        "id": "all_spots_all_genes",
        "pcc_overall_flat": _pearson(df["pred"].to_numpy(), df["gt"].to_numpy()),
        "scc_overall_flat": _spearman(df["pred"].to_numpy(), df["gt"].to_numpy()),
        "n_values": int(len(df)),
    })
    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "gene_map_metric_summary.csv", index=False)


def _plot_scc_barplot(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in ["spearman_scc", "pearson", "ssim"] if c in df.columns]
    err_cols = [c for c in ["rmse_zscore", "jsd"] if c in df.columns]
    n_metric = len(metric_cols) + len(err_cols)
    fig, axes = plt.subplots(1, max(1, n_metric),
                              figsize=(4 * max(1, n_metric), max(3, 0.30 * df["gene"].nunique())),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, col in zip(axes, metric_cols + err_cols):
        agg = df.groupby("gene", as_index=False)[col].mean().sort_values(col)
        color = "#4c78a8" if col in metric_cols else "#e15759"
        ax.barh(agg["gene"], agg[col], color=color)
        if col in metric_cols:
            ax.axvline(0, color="0.4", linewidth=1)
        ax.set_xlabel(f"mean {col} across samples")
    fig.suptitle("Stage1.5 spatial gene-map metric summary")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_spatial_embedding_umap(z: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                                 sample_id: str, genes: list[str], out: Path,
                                 *, max_spots: int = 6000, seed: int = 0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(z.shape[0])
    if z.shape[0] > max_spots:
        idx = np.sort(rng.choice(z.shape[0], max_spots, replace=False))
    X = z[idx].astype(np.float32, copy=False)
    obs = pd.DataFrame({"sample_id": [sample_id] * len(idx)})
    for j, g in enumerate(genes):
        obs[f"gt_{g}"] = gt[idx, j]
        obs[f"pred_{g}"] = pred[idx, j]
    try:
        import scanpy as sc
        import anndata as ad
        adata = ad.AnnData(X=X, obs=obs)
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=min(30, max(2, X.shape[0] - 1)))
        sc.tl.umap(adata, min_dist=0.1, random_state=seed)
        colors = []
        for g in genes[:3]:
            colors.extend([f"gt_{g}", f"pred_{g}"])
        sc.pl.umap(adata, color=colors, show=False, frameon=False, ncols=2, size=12)
        plt.suptitle(f"Stage1.5 spatial embedding UMAP | {sample_id}", y=1.02)
        plt.savefig(out, dpi=170, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.warning(f"scanpy Stage1.5 UMAP failed for {sample_id}: {e}")
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=seed).fit_transform(X)
        n = min(3, len(genes))
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False, constrained_layout=True)
        for j, g in enumerate(genes[:n]):
            for r, vals, title in [(0, gt[idx, j], f"GT {g}"), (1, pred[idx, j], f"Pred {g}")]:
                sca = axes[r, j].scatter(xy[:, 0], xy[:, 1], c=vals, s=6, cmap="viridis")
                axes[r, j].set_title(title)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
                fig.colorbar(sca, ax=axes[r, j], fraction=0.046, pad=0.02)
        fig.suptitle(f"Stage1.5 spatial embedding PCA | {sample_id}")
        fig.savefig(out, dpi=170)
        plt.close(fig)

def _copy_top_gene_maps(df: pd.DataFrame, out_dir: Path, *, top_k: int = 5) -> None:
    if df.empty or "spearman_scc" not in df.columns:
        return
    import shutil
    top = df.sort_values("spearman_scc", ascending=False).head(int(top_k))
    dst = out_dir / "top_scc_gene_maps"
    dst.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(top.itertuples(index=False), 1):
        sample = getattr(row, "sample_id")
        gene = getattr(row, "gene")
        src = out_dir / sample / f"{gene}.png"
        if src.exists():
            score = float(getattr(row, "spearman_scc"))
            shutil.copyfile(src, dst / f"rank{rank:02d}_scc{score:+.3f}_{sample}_{gene}.png")
    top.to_csv(dst / "top_scc_gene_maps.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--spatial-ckpt", required=True)
    ap.add_argument("--data", default="configs/stage15/data.yaml")
    ap.add_argument("--model", default="configs/stage15/model.yaml")
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--source", default="hest", choices=("hest", "all"))
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--genes", nargs="*", default=[],
                    help="Genes to plot/evaluate. Combine with --auto-select-genes for pattern-rich examples.")
    ap.add_argument("--auto-select-genes", type=int, default=0,
                    help="Append top-N GT spatially patterned genes from eval shards for qualitative maps.")
    ap.add_argument("--probe-train-samples", type=int, default=20)
    ap.add_argument("--max-train-spots", type=int, default=20000)
    ap.add_argument("--tx-batch", type=int, default=256)
    ap.add_argument("--tx-pooling-mode", default="ckpt",
                    choices=("ckpt", "cls", "token_mean", "cls_token_mean_sum", "cls_token_mean_avg",
                             "cls_mean_sum", "cls_mean_avg", "mean"),
                    help="Override Stage1 tx_encoder readout. Use with care for existing Stage1.5 ckpts.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-viz", action="store_true",
                     help="Only write CSVs; skip SCC barplot and embedding UMAP artifacts.")
    ap.add_argument("--viz-max-spots", type=int, default=6000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = _load_yaml(args.data)["data"]
    model_cfg = _load_yaml(args.model)["model"]
    prepared = Path(args.prepared_dir or data_cfg["prepared_dir"])

    tx_encoder, tx_dim, vocab_keep, gene_norm_cfg = _load_stage1_tx(
        Path(args.stage1_ckpt), device, tx_pooling_mode=args.tx_pooling_mode
    )
    with h5py.File(_split_paths(prepared, "train", source="hest")[0], "r") as f0:
        full_dim = int(f0["hvg_log"].shape[1])
    hvg_dim = int(len(vocab_keep)) if vocab_keep is not None else full_dim
    normalizer = GeneNormalizer(gene_norm_cfg, full_hvg_dim=full_dim, hvg_dim=hvg_dim, vocab_keep_indices=vocab_keep)
    genes_eff = _load_gene_names(prepared, vocab_keep)
    gene_to_eff = {g.upper(): i for i, g in enumerate(genes_eff)}

    # Resolve eval paths before finalising gene list so HEST examples can pick
    # genes that actually show spatial structure in the selected test shards.
    if args.samples:
        eval_paths = []
        for sid in args.samples:
            p = _resolve_shard(prepared, sid)
            if p is not None:
                eval_paths.append(p)
    else:
        eval_paths = _split_paths(prepared, args.split, source=args.source)[:5]
    if not eval_paths:
        raise SystemExit(f"No eval shards found for split={args.split}")

    requested_genes = list(args.genes or [])
    if args.auto_select_genes > 0:
        auto = _auto_select_spatial_genes(
            eval_paths, vocab_keep, genes_eff, int(args.auto_select_genes),
            exclude={g.upper() for g in requested_genes},
        )
        log.info(f"auto-selected spatial genes: {auto}")
        requested_genes.extend(auto)
    # Deduplicate while preserving order.
    seen = set()
    requested_genes = [g for g in requested_genes if not (g.upper() in seen or seen.add(g.upper()))]
    if not requested_genes:
        raise SystemExit("No genes requested. Use --genes ... or --auto-select-genes N")
    missing = [g for g in requested_genes if g.upper() not in gene_to_eff]
    if missing:
        raise SystemExit(f"Genes not in effective Stage1 vocab: {missing}")
    args.genes = requested_genes
    gene_idx = np.array([gene_to_eff[g.upper()] for g in args.genes], dtype=np.int64)

    sckpt = torch.load(args.spatial_ckpt, map_location="cpu", weights_only=False)
    scfg = sckpt.get("spatial_config") or model_cfg["spatial"]
    region_on = bool(data_cfg.get("region", {}).get("enable", True))
    spatial = SpatialEncoder(
        tx_dim=tx_dim,
        img_dim=1536,
        fuse_dim=scfg.get("fuse_dim", 256),
        fuse_image=bool(data_cfg.get("use_image", True)) and bool(scfg.get("fuse_image", True)),
        fuse_region=region_on,
        token_mode=str(scfg.get("region_token_mode", "fused")),
        arch=scfg.get("arch", "kgnn"),
        n_layers=scfg.get("n_layers", 3),
        n_heads=scfg.get("n_heads", 4),
        dropout=scfg.get("dropout", 0.1),
    )
    state = sckpt.get("spatial_state_dict") or sckpt.get("model") or sckpt
    spatial.load_state_dict(state, strict=False)
    spatial.eval().to(device)

    train_paths = _split_paths(prepared, "train", source="hest")[: args.probe_train_samples]
    if not train_paths:
        raise SystemExit("No HEST train shards found for probe training")
    rng = np.random.default_rng(0)
    x_parts, y_parts = [], []
    per = max(1, args.max_train_spots // max(1, len(train_paths)))
    log.info(f"encoding {len(train_paths)} train shards for Ridge probe...")
    for p in train_paths:
        rec = encode_spatial_sample(p, tx_encoder, spatial, normalizer, vocab_keep, data_cfg, device, tx_batch=args.tx_batch)
        n = rec["z"].shape[0]
        sel = rng.choice(n, min(n, per), replace=False) if n > per else np.arange(n)
        x_parts.append(rec["z"][sel])
        y_parts.append(rec["hvg_eff"][sel][:, gene_idx])
    Xtr = np.concatenate(x_parts, axis=0)
    Ytr = np.concatenate(y_parts, axis=0)
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    probe.fit(Xtr, Ytr)

    out_dir = Path(args.out_dir or Path(args.spatial_ckpt).parent / f"gene_maps_{args.split}")
    rows = []
    spot_tables = []
    for p in eval_paths:
        rec = encode_spatial_sample(p, tx_encoder, spatial, normalizer, vocab_keep, data_cfg, device, tx_batch=args.tx_batch)
        pred = np.asarray(probe.predict(rec["z"]), dtype=np.float32)
        gt = rec["hvg_eff"][:, gene_idx]
        if not args.no_viz:
            try:
                _plot_spatial_embedding_umap(
                    rec["z"], gt, pred, rec["sample_id"], list(args.genes),
                    out_dir / rec["sample_id"] / "spatial_embedding_umap.png",
                    max_spots=args.viz_max_spots,
                )
            except Exception as e:
                log.warning(f"Stage1.5 embedding UMAP failed for {rec['sample_id']}: {e}", exc_info=True)
        # Optional normalised-domain metrics — same caveat as dlpfc_eval:
        # SCC/PCC are mathematically invariant under per-gene positive
        # scaling, but RMSE/SSIM/JSD differ and reflect the loss surface the
        # encoder actually sees.
        gt_norm = pred_norm = None
        if normalizer is not None and normalizer.mode != "none":
            full_gt = np.zeros((gt.shape[0], normalizer.hvg_dim), dtype=np.float32)
            full_gt[:, gene_idx] = gt
            gt_norm = normalizer.apply_np(full_gt)[:, gene_idx]
            full_pred = np.zeros_like(full_gt)
            full_pred[:, gene_idx] = pred
            pred_norm = normalizer.apply_np(full_pred)[:, gene_idx]

        for j, gene in enumerate(args.genes):
            scc = _spearman(pred[:, j], gt[:, j])
            pcc = _pearson(pred[:, j], gt[:, j])
            rmse = _rmse_zscore(gt[:, j], pred[:, j])
            ssim = _ssim_1d(gt[:, j], pred[:, j])
            jsd = _jsd_1d(gt[:, j], pred[:, j])
            row = {
                "sample_id": rec["sample_id"],
                "gene": gene,
                "metric_unit": "per_gene_per_sample",
                "pcc_definition": "Pearson across spots for one gene within one sample",
                "scc_definition": "Spearman across spots for one gene within one sample",
                # raw log1p_cp10k metrics
                "spearman_scc": scc, "spearman_scc_raw": scc,
                "pearson": pcc, "pearson_raw": pcc,
                "rmse_zscore": rmse, "rmse_zscore_raw": rmse,
                "ssim": ssim, "ssim_raw": ssim,
                "jsd": jsd, "jsd_raw": jsd,
                "n_spots": int(gt.shape[0]),
                "gt_nonzero_frac": float((gt[:, j] > 0).mean()),
            }
            metrics_raw = {"SCC": scc, "PCC": pcc, "SSIM": ssim, "JSD": jsd}
            metrics_norm = None
            if gt_norm is not None:
                ns = _spearman(pred_norm[:, j], gt_norm[:, j])
                npc = _pearson(pred_norm[:, j], gt_norm[:, j])
                nrm = _rmse_zscore(gt_norm[:, j], pred_norm[:, j])
                nss = _ssim_1d(gt_norm[:, j], pred_norm[:, j])
                njs = _jsd_1d(gt_norm[:, j], pred_norm[:, j])
                row.update({
                    "spearman_scc_norm": ns, "pearson_norm": npc,
                    "rmse_zscore_norm": nrm, "ssim_norm": nss, "jsd_norm": njs,
                })
                metrics_norm = {"SCC": ns, "PCC": npc, "SSIM": nss, "JSD": njs}
            rows.append(row)
            sample_dir = out_dir / rec["sample_id"]
            if not args.no_viz:
                _plot_gene_map(
                    rec["coords"], gt[:, j], pred[:, j], rec["sample_id"], gene,
                    {**({"SCC": scc, "PCC": pcc, "SSIM": ssim, "JSD": jsd, "RMSE_z": rmse}),
                      **({"SCC_norm": metrics_norm["SCC"],
                          "SSIM_norm": metrics_norm["SSIM"],
                          "JSD_norm": metrics_norm["JSD"]} if metrics_norm else {})},
                    sample_dir / f"{gene}.png",
                )
            sample_dir.mkdir(parents=True, exist_ok=True)
            spot_df = pd.DataFrame({
                "sample_id": rec["sample_id"],
                "spot_index": np.arange(gt.shape[0], dtype=np.int64),
                "gene": gene,
                "x": rec["coords"][:, 0],
                "y": rec["coords"][:, 1],
                "gt": gt[:, j],
                "pred": pred[:, j],
            })
            spot_tables.append(spot_df)
            spot_df.to_csv(sample_dir / f"{gene}_spots.csv", index=False)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gene_map_scc.csv", index=False)
    _write_hest_style_summary(spot_tables, out_dir)
    if not args.no_viz:
        try:
            _plot_scc_barplot(df, out_dir / "gene_map_scc_barplot.png")
            _copy_top_gene_maps(df, out_dir, top_k=5)
        except Exception as e:
            log.warning(f"Stage1.5 SCC/top-map plotting failed: {e}", exc_info=True)
    log.info(f"wrote {out_dir / 'gene_map_scc.csv'}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
