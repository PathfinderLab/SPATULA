"""DLPFC / spatialLIBD downstream evaluation for Stage-1 RNA embeddings.

The evaluator scores multiple transcriptomics representations from a frozen
Stage-1 tx encoder:

  - h_tx        : clean full/sampled encoder output
  - chunk_state : one sampled gene chunk embedding, z_chunk
  - spot_state  : full clean non-zero sequence embedding, z_spot, used for inference

Metrics are grouped by representation:

  - supervised linear probe: embedding -> cortical layer label
  - zero-shot clustering: KMeans(embedding) vs cortical layer label, ARI/NMI
  - kNN purity: local embedding neighbours share layer labels
  - optional gene-map probe: leave-one-sample-out Ridge embedding -> gene
    expression, reported as spatial Spearman SCC and GT/pred heatmaps.
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
import pandas as pd
import torch

# scripts/eval/dlpfc_eval.py -> repo root is parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.dlpfc import load_dlpfc_sample, list_dlpfc_samples
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.evaluation.stage1_benchmarks import chunk_view_embeddings_from_encoder
from mm_align.evaluation.spatial_metrics import (
    cluster_assign, cluster_label_metrics, silhouette_spatial,
    chaos_score, pas_score, marker_spatial_autocorr,
    fide_score, normalized_entropy, novae_heuristic, slide_jsd,
)
from mm_align.utils import get_logger

# Re-use checkpoint/config recovery helpers from stage1_tx.py.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_stage1_tx", str(Path(__file__).parent / "stage1_tx.py"))
_stage1_tx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stage1_tx)
load_tx_encoder = _stage1_tx.load_tx_encoder
_prepare_input = _stage1_tx._prepare_input
linear_probe = _stage1_tx.linear_probe

log = get_logger("dlpfc_eval")


@torch.no_grad()
def _encode_h_tx(enc, x: np.ndarray, batch: int = 256, device: str = "cuda") -> np.ndarray:
    out = []
    for r0 in range(0, x.shape[0], batch):
        xb = torch.from_numpy(x[r0:r0 + batch].astype(np.float32)).to(device)
        out.append(enc(novae_latent=None, hvg=xb)["h_tx"].detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _encode_representations(enc, x_norm: np.ndarray, reps: list[str], *,
                            batch: int, device: str, chunk_n: int,
                            chunk_len: int, seed: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "h_tx" in reps:
        out["h_tx"] = _encode_h_tx(enc, x_norm, batch=batch, device=device)
    if "chunk_state" in reps or "spot_state" in reps:
        views = chunk_view_embeddings_from_encoder(
            enc, x_norm,
            n_chunks=chunk_n,
            chunk_len=chunk_len,
            dynamic=True,
            batch_size=min(batch, 128),
            max_spots=max(1, x_norm.shape[0]),
            seed=seed,
            device=device,
        )
        if "chunk_state" in reps:
            out["chunk_state"] = views["z_chunk"].astype(np.float32)
        if "spot_state" in reps:
            out["spot_state"] = views["z_spot"].astype(np.float32)
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = _rankdata(x[ok])
    ry = _rankdata(y[ok])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den <= 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def _spatial_pattern_score(coords: np.ndarray, expr: np.ndarray, *, k: int = 8) -> np.ndarray:
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
    prevalence = np.clip((nz - 0.03) / 0.20, 0.0, 1.0) * np.clip((0.95 - nz) / 0.30, 0.0, 1.0)
    return (np.maximum(local_corr, 0.0) * np.log1p(var) * prevalence).astype(np.float32)


def _auto_select_spatial_genes(per_sample: list[dict], genes: list[str], n: int,
                               *, exclude: set[str] | None = None) -> list[str]:
    if n <= 0 or not per_sample:
        return []
    scores = np.zeros(len(genes), dtype=np.float64)
    counts = np.zeros(len(genes), dtype=np.float64)
    exclude = {g.upper() for g in (exclude or set())}
    for sample in per_sample:
        try:
            sc = _spatial_pattern_score(sample["coords"], sample["hvg_eff"])
            if sc.shape[0] == scores.shape[0]:
                scores += sc
                counts += np.isfinite(sc).astype(np.float64)
        except Exception as e:
            log.warning(f"auto DLPFC gene scoring skipped {sample.get('sample_id')}: {e}")
    scores = scores / np.maximum(counts, 1.0)
    for i, g in enumerate(genes):
        if g.upper() in exclude:
            scores[i] = -np.inf
    order = np.argsort(-scores)
    out = [genes[int(i)] for i in order if np.isfinite(scores[int(i)]) and scores[int(i)] > 0]
    return out[:n]


def knn_purity(emb: np.ndarray, labels: np.ndarray, ks=(5, 10, 20)) -> dict[str, float]:
    from sklearn.neighbors import NearestNeighbors
    out = {}
    n = emb.shape[0]
    if n < max(ks) + 1:
        return {f"knn_purity_at_{k}": float("nan") for k in ks}
    nn = NearestNeighbors(n_neighbors=max(ks) + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    for k in ks:
        out[f"knn_purity_at_{k}"] = float((labels[idx[:, :k]] == labels[:, None]).mean())
    return out


def _standardize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x - mu) / sd


def _spatial_augmented_embedding(emb: np.ndarray, coords: np.ndarray, *, weight: float) -> np.ndarray:
    """Embedding + weak spatial coordinate cue for GraphST/STAGATE-style clustering.

    This is only a clustering view; supervised probes and gene-map probes still
    use the original embedding.  The small default weight keeps coordinates as
    a locality prior rather than letting xy dominate the result.
    """
    e = _standardize_features(emb)
    xy = _standardize_features(coords) * float(weight)
    return np.concatenate([e, xy], axis=1).astype(np.float32, copy=False)


def clustering_metrics_one_method(emb: np.ndarray, labels: np.ndarray,
                                    coords: np.ndarray,
                                    n_clusters: int,
                                    *, method: str,
                                    seed: int = 0,
                                    repeats: int = 1,
                                    spatial_weight: float = 0.25) -> tuple[dict[str, float], np.ndarray]:
    """Run a clustering method N times, return (metrics_with_prefix, best_pred).

    Metric prefix = "{method}_".  Includes label metrics (ari/nmi/hom/com),
    embedding silhouette, plus spatial-continuity metrics (chaos/pas/asw)
    when coords are provided.  Adds <metric>_std when repeats > 1.
    """
    from sklearn.metrics import silhouette_score
    nan_keys = ["ari", "nmi", "homogeneity", "completeness", "silhouette",
                "chaos", "pas", "asw_spatial"]
    if emb.shape[0] < n_clusters + 2 or n_clusters < 2:
        return ({f"{method}_{k}": float("nan") for k in nan_keys} |
                {f"{method}_k": float(n_clusters)}), np.zeros(emb.shape[0], dtype=int)

    cluster_emb = emb
    cluster_method = method
    method_prefix = method
    if method in ("spatial_leiden", "leiden_spatial"):
        cluster_emb = _spatial_augmented_embedding(emb, coords, weight=spatial_weight)
        cluster_method = "leiden"
        method_prefix = "spatial_leiden"
    best_pred, all_preds = cluster_assign(
        cluster_emb, n_clusters=n_clusters, method=cluster_method, seed=seed, n_repeats=repeats,
    )

    def _per_run(pred: np.ndarray) -> dict[str, float]:
        m = cluster_label_metrics(labels, pred)
        try:
            m["silhouette"] = float(silhouette_score(cluster_emb, pred))
        except Exception:
            m["silhouette"] = float("nan")
        m["chaos"] = chaos_score(pred, coords)
        m["pas"] = pas_score(pred, coords)
        m["asw_spatial"] = silhouette_spatial(coords, pred)
        # Novae-style domain-continuity metrics.
        m["fide"] = fide_score(pred, coords, k=6)
        m["entropy_norm"] = normalized_entropy(pred)
        heur = novae_heuristic(pred, coords, k=6)
        m["heuristic"] = heur["heuristic"]
        return m

    runs = [_per_run(p) for p in all_preds]
    keys = list(runs[0].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[f"{method_prefix}_{k}"] = float(np.nanmean(vals))
        if repeats > 1:
            out[f"{method_prefix}_{k}_std"] = (
                float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
    out[f"{method_prefix}_k"] = float(len(np.unique(best_pred)))
    if method_prefix == "spatial_leiden":
        out[f"{method_prefix}_coord_weight"] = float(spatial_weight)
    return out, best_pred


def gt_layer_continuity(layers: np.ndarray, coords: np.ndarray,
                         hvg: np.ndarray | None, gene_names: list[str] | None,
                         *, marker_top_per_cluster: int, spatial_k: int) -> dict[str, float]:
    """Spatial-continuity metrics of the GROUND-TRUTH layer labels themselves —
    these are a property of the dataset/sample, not of any encoder, but they
    anchor what "good" CHAOS/PAS/ASW look like for that sample (sanity check).
    """
    out: dict[str, float] = {
        "gt_layer_n_layers": float(len(np.unique(layers))),
        "gt_layer_chaos": chaos_score(layers, coords),
        "gt_layer_pas": pas_score(layers, coords),
        "gt_layer_asw_spatial": silhouette_spatial(coords, layers),
        "gt_layer_fide": fide_score(layers, coords, k=6),
        "gt_layer_entropy_norm": normalized_entropy(layers),
    }
    if hvg is not None and gene_names is not None:
        try:
            mm = marker_spatial_autocorr(
                hvg, coords, layers, gene_names,
                top_per_cluster=marker_top_per_cluster, k=spatial_k,
            )
            out["gt_layer_marker_morans_i_median"] = mm["marker_morans_i_median"]
            out["gt_layer_marker_gearys_c_median"] = mm["marker_gearys_c_median"]
            out["gt_layer_n_markers"] = float(mm["n_markers"])
        except Exception:
            out["gt_layer_marker_morans_i_median"] = float("nan")
            out["gt_layer_marker_gearys_c_median"] = float("nan")
            out["gt_layer_n_markers"] = 0.0
    return out


def leiden_cluster_metrics(emb: np.ndarray, labels: np.ndarray,
                           n_neighbors: int = 30,
                           resolution: float = 1.0,
                           seed: int = 0) -> dict[str, float]:
    try:
        import scanpy as sc
        from anndata import AnnData
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError as e:
        log.warning(f"scanpy missing -> skipping leiden: {e}")
        return {"leiden_ari": float("nan"), "leiden_nmi": float("nan"), "leiden_k": float(0)}
    ad = AnnData(X=emb.astype(np.float32))
    sc.pp.neighbors(ad, n_neighbors=min(n_neighbors, max(2, emb.shape[0] - 1)), use_rep="X",
                    metric="euclidean", random_state=seed)
    sc.tl.leiden(ad, resolution=resolution, random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    pred = ad.obs["leiden"].astype(int).to_numpy()
    return {
        "leiden_ari": float(adjusted_rand_score(labels, pred)),
        "leiden_nmi": float(normalized_mutual_info_score(labels, pred)),
        "leiden_k": float(len(np.unique(pred))),
    }


def _load_gene_names(prepared: Path, vocab_keep: np.ndarray | None) -> list[str]:
    genes = json.loads((prepared / "hvg_vocab.json").read_text())
    if vocab_keep is not None:
        genes = [genes[int(i)] for i in vocab_keep]
    return [str(g) for g in genes]


def _plot_cluster_panel(coords: np.ndarray, layer: np.ndarray,
                        preds: dict[str, np.ndarray],
                        sample_id: str, rep: str, out: Path) -> None:
    """GT layer + every requested clustering method side-by-side."""
    out.parent.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[np.ndarray, str]] = [(layer.astype(str), "GT layer")]
    for m, p in preds.items():
        panels.append((p.astype(str), m))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (vals, title) in zip(axes, panels):
        uniq = np.unique(vals)
        colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(uniq))))
        for i, u in enumerate(uniq):
            m = vals == u
            ax.scatter(coords[m, 0], coords[m, 1], s=8,
                       color=colors[i % len(colors)], label=str(u), alpha=0.9)
        if len(uniq) <= 12:
            ax.legend(fontsize=6, markerscale=1.8, loc="best")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DLPFC | {sample_id} | rep={rep}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_method_summary_bar(per_sample_df: pd.DataFrame, methods: list[str],
                              out: Path) -> None:
    """Per-method clustering summary with scale-aware panels.

    Label-agreement metrics (ARI/NMI/HOM/COM) are bounded near [0, 1], while
    CHAOS is a distance-like quantity and can be orders of magnitude larger.
    Plotting them on one axis makes the informative metrics look like zeros, so
    we split the figure into metric families.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = [
        ("Label agreement (higher better)", ["ari", "nmi", "homogeneity", "completeness"]),
        ("Spatial continuity (scale differs)", ["chaos", "pas", "asw_spatial"]),
        ("Domain regularity / Novae-style", ["fide", "entropy_norm", "heuristic"]),
    ]

    def _collect(metric_keys: list[str]) -> pd.DataFrame:
        rows = []
        for m in methods:
            for k in metric_keys:
                col = f"{m}_{k}"
                if col not in per_sample_df.columns:
                    continue
                vals = pd.to_numeric(per_sample_df[col], errors="coerce")
                rows.append({
                    "method": m,
                    "metric": k,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0,
                })
        return pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(groups), figsize=(6.0 * len(groups), 4.4),
                             constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    any_rows = False
    for ax, (title, metric_keys) in zip(axes, groups):
        sdf = _collect(metric_keys)
        if sdf.empty:
            ax.axis("off")
            ax.set_title(title)
            continue
        any_rows = True
        pivot_mean = sdf.pivot(index="metric", columns="method", values="mean")
        pivot_std = sdf.pivot(index="metric", columns="method", values="std")
        pivot_mean = pivot_mean.reindex([m for m in metric_keys if m in pivot_mean.index])
        pivot_std = pivot_std.reindex(pivot_mean.index)
        x = np.arange(len(pivot_mean.index))
        width = 0.8 / max(1, pivot_mean.shape[1])
        for j, m in enumerate(pivot_mean.columns):
            ax.bar(x + j * width - 0.4 + width / 2,
                   pivot_mean[m].values, width=width, yerr=pivot_std[m].values,
                   capsize=3, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_mean.index, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if title.startswith("Label"):
            ax.set_ylim(bottom=min(0.0, float(np.nanmin(pivot_mean.values)) - 0.02),
                        top=max(0.1, float(np.nanmax(pivot_mean.values)) + 0.05))
        ax.set_ylabel("mean ± std")
    if not any_rows:
        plt.close(fig)
        return
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
                   bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("DLPFC clustering metrics by method", y=1.02)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_map(coords: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                   sample_id: str, rep: str, gene: str,
                   metrics_raw: dict, metrics_norm: dict | None,
                   out: Path) -> None:
    """Two-row panel: GT(left) + Pred(right).  Title prints metrics in BOTH
    log1p (raw, post-cp10k log) AND the encoder's normalised space.

    Why both?  Per-gene Spearman/Pearson are mathematically invariant under
    the per-gene positive scaling of `global_median`, but RMSE / SSIM / JSD
    are NOT scale-invariant — the encoder lives in normalised space so its
    "distance" intuition only matches what it saw at train time.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    vals = np.concatenate([gt, pred])
    vmin = float(np.nanpercentile(vals, 1))
    vmax = float(np.nanpercentile(vals, 99))
    for ax, y, title in [(axes[0], gt, "GT (log1p cp10k)"),
                          (axes[1], pred, "Predicted (log1p cp10k)")]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, s=8, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    raw_line = " | ".join(f"{k}={v:.3f}" for k, v in metrics_raw.items()
                            if isinstance(v, (int, float)) and np.isfinite(v))
    if metrics_norm:
        norm_line = " | ".join(f"{k}={v:.3f}" for k, v in metrics_norm.items()
                                if isinstance(v, (int, float)) and np.isfinite(v))
        fig.suptitle(f"{sample_id} | {rep} | {gene}\n"
                      f"raw  ({raw_line})\n"
                      f"norm ({norm_line})", fontsize=9)
    else:
        fig.suptitle(f"{sample_id} | {rep} | {gene} | {raw_line}", fontsize=10)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _ridge_gene_probe_and_maps(per_sample: list[dict], genes: list[str], requested: list[str],
                               *, ckpt_name: str, rep: str, out_dir: Path,
                               max_viz_samples: int = 3, alpha: float = 10.0,
                               normalizer=None) -> pd.DataFrame:
    """Leave-one-sample-out Ridge probe (emb -> selected genes).

    Reports BOTH:
        * raw metrics  — on log1p_cp10k scale (the GT scale, biologically
                          meaningful magnitudes)
        * norm metrics — after applying the encoder's training-time
                          `gene_norm` (global_median etc.) to BOTH gt & pred.
                          The encoder lives in this scale, so RMSE / SSIM /
                          JSD here reflect what the model's loss "feels".
    Note: per-gene Spearman & Pearson are mathematically invariant under
    per-gene positive scaling — they will be identical in raw and norm modes
    unless the `clip` arm of gene_norm fires.  RMSE / SSIM / JSD genuinely
    differ between modes.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from mm_align.evaluation.gene_imputation_metrics import (
        pearson_1d as _p1d, ssim_1d as _ssim, jsd_1d as _jsd, rmse_zscore as _rmse_z,
    )

    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    selected = [g for g in requested if g.upper() in gene_to_idx]
    if not selected:
        log.warning(f"no requested genes found in effective vocab: {requested}")
        return pd.DataFrame()
    gidx = np.array([gene_to_idx[g.upper()] for g in selected], dtype=np.int64)
    rows = []
    for hold_i, sample in enumerate(per_sample):
        train = [x for j, x in enumerate(per_sample) if j != hold_i]
        if not train:
            continue
        Xtr = np.concatenate([x["emb"] for x in train], axis=0)
        Ytr = np.concatenate([x["hvg_eff"][:, gidx] for x in train], axis=0)
        Xte = sample["emb"]
        Yte = sample["hvg_eff"][:, gidx]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(Xtr, Ytr)
        pred = model.predict(Xte).astype(np.float32)
        # Build normalised copies for both gt and pred when a gene_norm is
        # available.  The normaliser was fit on the encoder's full HVG vocab,
        # so we feed the per-gene scalar slice via apply_np over the relevant
        # column subset.  Reuse existing logic by constructing a full-width
        # array with zeros elsewhere and slicing back.
        gt_norm = pred_norm = None
        if normalizer is not None and normalizer.mode != "none":
            full = np.zeros((Yte.shape[0], normalizer.hvg_dim), dtype=np.float32)
            full[:, gidx] = Yte
            gt_norm = normalizer.apply_np(full)[:, gidx]
            full[:, gidx] = pred
            pred_norm = normalizer.apply_np(full)[:, gidx]

        for j, gene in enumerate(selected):
            # raw-space metrics
            raw_scc = _spearman(Yte[:, j], pred[:, j])
            raw_pcc = _p1d(Yte[:, j], pred[:, j])
            raw_ssim = _ssim(Yte[:, j], pred[:, j])
            raw_jsd = _jsd(Yte[:, j], pred[:, j])
            raw_rmse = _rmse_z(Yte[:, j], pred[:, j])
            rec = {
                "ckpt": ckpt_name,
                "representation": rep,
                "sample": sample["sample_id"],
                "gene": gene,
                "metric_unit": "per_gene_per_sample",
                "pcc_definition": "Pearson across spots for one gene within one sample",
                "scc_definition": "Spearman across spots for one gene within one sample",
                "n_spots": int(Xte.shape[0]),
                # raw-space (log1p cp10k) metrics
                "spearman_scc_raw": raw_scc,
                "pearson_raw": raw_pcc,
                "ssim_raw": raw_ssim,
                "jsd_raw": raw_jsd,
                "rmse_zscore_raw": raw_rmse,
                # back-compat column name
                "spearman_scc": raw_scc,
            }
            metrics_raw = {"SCC": raw_scc, "PCC": raw_pcc,
                            "SSIM": raw_ssim, "JSD": raw_jsd}
            metrics_norm = None
            if gt_norm is not None:
                ns = _spearman(gt_norm[:, j], pred_norm[:, j])
                np_ = _p1d(gt_norm[:, j], pred_norm[:, j])
                nss = _ssim(gt_norm[:, j], pred_norm[:, j])
                njs = _jsd(gt_norm[:, j], pred_norm[:, j])
                nrm = _rmse_z(gt_norm[:, j], pred_norm[:, j])
                rec.update({
                    "spearman_scc_norm": ns,
                    "pearson_norm": np_,
                    "ssim_norm": nss,
                    "jsd_norm": njs,
                    "rmse_zscore_norm": nrm,
                })
                metrics_norm = {"SCC": ns, "PCC": np_, "SSIM": nss, "JSD": njs}
            rows.append(rec)
            if hold_i < max_viz_samples:
                _plot_gene_map(
                    sample["coords"], Yte[:, j], pred[:, j],
                    sample["sample_id"], rep, gene,
                    metrics_raw=metrics_raw, metrics_norm=metrics_norm,
                    out=out_dir / ckpt_name / rep / "gene_maps" / sample["sample_id"] / f"{gene}.png",
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        out = out_dir / ckpt_name / rep / "gene_map_scc.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        # 2-column bar: SCC_raw (= identical to norm in theory) and SSIM_norm
        # (where the encoder's loss actually lives).  Two stacks per gene.
        agg = df.groupby("gene", as_index=False).agg(
            scc=("spearman_scc_raw", "mean"),
            ssim_raw=("ssim_raw", "mean"),
            ssim_norm=("ssim_norm", "mean") if "ssim_norm" in df.columns else ("ssim_raw", "mean"),
        ).sort_values("scc")
        fig, axes = plt.subplots(1, 2, figsize=(11, max(3, 0.35 * len(agg))),
                                   constrained_layout=True)
        axes[0].barh(agg["gene"], agg["scc"], color="#4c78a8")
        axes[0].axvline(0, color="0.4", linewidth=1)
        axes[0].set_xlabel("leave-one-sample-out Spearman SCC (raw log1p)")
        axes[0].set_title("SCC")
        w = 0.4
        y = np.arange(len(agg))
        axes[1].barh(y - w/2, agg["ssim_raw"], height=w, color="#4c78a8", label="SSIM (raw log1p)")
        axes[1].barh(y + w/2, agg["ssim_norm"], height=w, color="#e15759", label="SSIM (norm)")
        axes[1].set_yticks(y); axes[1].set_yticklabels(agg["gene"])
        axes[1].set_xlabel("SSIM mean across samples")
        axes[1].set_title("SSIM (raw vs normalised)")
        axes[1].legend()
        fig.suptitle(f"DLPFC gene-map probe | {ckpt_name} | {rep}")
        fig.savefig(out_dir / ckpt_name / rep / "gene_map_scc_barplot.png", dpi=170)
        plt.close(fig)
    return df


def _parse_reps(s: str) -> list[str]:
    if s == "all":
        return ["h_tx", "chunk_state", "spot_state"]
    reps = [x.strip() for x in s.split(",") if x.strip()]
    valid = {"h_tx", "chunk_state", "spot_state"}
    bad = [x for x in reps if x not in valid]
    if bad:
        raise ValueError(f"unknown representations={bad}; use h_tx,chunk_state,spot_state,all")
    return reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlpfc-dir", default="/data/spatiallibd")
    ap.add_argument("--vocab", default="results/cache/prepared_expanded/hvg_vocab_dict.json")
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default="results/eval/dlpfc_compare.csv")
    ap.add_argument("--per-sample-out", default="results/eval/dlpfc_per_sample.csv")
    ap.add_argument("--representations", default="h_tx,spot_state",
                    help="Comma list: h_tx,chunk_state,spot_state or all. Default h_tx,spot_state.")
    ap.add_argument("--stage", choices=("1", "15"), default="1",
                    help="Stage 15 additionally reports Leiden clustering.")
    ap.add_argument("--leiden-resolution", type=float, default=1.0)
    ap.add_argument("--leiden-n-neighbors", type=int, default=30)
    # Clustering options (SDMBench-style).  --cluster-methods takes a comma list
    # so we can run BOTH KMeans and Leiden and report each with its own prefix.
    ap.add_argument("--cluster-methods", default="kmeans,gmm,leiden,spatial_leiden",
                    help="Comma list; any of: kmeans, gmm/mclust, leiden, spatial_leiden. "
                         "spatial_leiden runs Leiden on embedding plus a weak xy prior.")
    ap.add_argument("--spatial-cluster-weight", type=float, default=0.25,
                    help="Weight for standardised xy coordinates in spatial_leiden.")
    ap.add_argument("--cluster-k-mode", default="adaptive",
                    choices=("adaptive", "fixed"),
                    help="adaptive = len(unique(layers)); fixed = --cluster-k.")
    ap.add_argument("--cluster-k", type=int, default=7,
                    help="Number of clusters when --cluster-k-mode fixed (DLPFC papers use 7).")
    ap.add_argument("--cluster-repeats", type=int, default=1,
                    help="Independent clustering runs; std reported across repeats.")
    ap.add_argument("--marker-top-per-cluster", type=int, default=5,
                    help="Per-cluster top markers for Moran's I / Geary's C reporting.")
    ap.add_argument("--spatial-k", type=int, default=6,
                    help="KNN for spatial weight matrix (Moran's I / Geary's C).")
    ap.add_argument("--encode-batch", type=int, default=256)
    ap.add_argument("--neural-linear-probe", action="store_true",
                    help="Also fit a trainable nn.Linear layer probe with early stopping.")
    ap.add_argument("--neural-probe-epochs", type=int, default=50)
    ap.add_argument("--neural-probe-patience", type=int, default=8)
    ap.add_argument("--neural-probe-lr", type=float, default=1e-3)
    ap.add_argument("--neural-probe-weight-decay", type=float, default=1e-4)
    ap.add_argument("--neural-probe-batch", type=int, default=512)
    ap.add_argument("--tx-pooling-mode", default="ckpt",
                    choices=("ckpt", "cls", "token_mean", "cls_token_mean_sum", "cls_token_mean_avg",
                             "cls_mean_sum", "cls_mean_avg", "mean"),
                    help="Override top_hvg_gene spot readout at eval time.")
    ap.add_argument("--chunk-n", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=256)
    ap.add_argument("--samples-cap", type=int, default=None)
    ap.add_argument("--include-no-truth", action="store_true",
                    help="Include samples without a *_truth.txt (e.g. 151675). "
                         "Supervised layer-probe metrics for these samples are NaN; "
                         "unsupervised clustering / gene-map / spatial-continuity still work.")
    ap.add_argument("--viz-out-dir", default="",
                    help="If set, write cluster panels and gene maps here.")
    ap.add_argument("--viz-samples", type=int, default=3)
    ap.add_argument("--genes", nargs="*", default=["MBP", "SNAP25", "PCP4", "GFAP", "MOBP", "CARTPT"],
                    help="Genes for leave-one-sample-out expression maps. Defaults are the "
                         "canonical DLPFC cortical layer markers from Maynard et al. 2021 "
                         "(spatialLIBD): MBP/MOBP = white matter (myelin), SNAP25 = pan-neuronal, "
                         "PCP4 = layer 5, GFAP = astrocyte/WM, CARTPT = layer 4. These are the "
                         "same genes STAGATE / BANKSY / GraphST / BayesSpace papers use for "
                         "qualitative inspection.")
    ap.add_argument("--auto-select-genes", type=int, default=0,
                    help="Append top-N GT spatially patterned DLPFC genes for qualitative gene-map probes.")
    ap.add_argument("--gene-map-representations", default="spot_state",
                    help="Comma list/all subset used for gene-map Ridge probe.")
    args = ap.parse_args()

    reps = _parse_reps(args.representations)
    gene_map_reps = _parse_reps(args.gene_map_representations)
    include_leiden = args.stage == "15"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = Path(args.prepared_dir)

    samples = list_dlpfc_samples(args.dlpfc_dir, require_truth=not args.include_no_truth)
    if args.samples_cap:
        samples = samples[: args.samples_cap]
    log.info(f"DLPFC samples: {len(samples)} — {[s.name for s in samples]}")
    log.info("loading DLPFC samples and aligning to project vocab...")
    data = []
    for s in samples:
        d = load_dlpfc_sample(s, args.vocab)
        log.info(f"  {d['sample_id']}: {d['hvg_log'].shape} layers={len(np.unique(d['layers']))}")
        data.append(d)

    rows = []
    all_per_sample = []
    all_gene_rows = []
    viz_dir = Path(args.viz_out_dir) if args.viz_out_dir else None

    for ck_s in args.ckpts:
        ck = Path(ck_s)
        ckpt_name = ck.parent.name
        log.info(f"== {ck} ==")
        enc, _full_cfg, vocab_keep, gene_norm_cfg = load_tx_encoder(
            ck, device=device, tx_pooling_mode=args.tx_pooling_mode
        )
        d_eff = int(len(vocab_keep)) if vocab_keep is not None else data[0]["hvg_log"].shape[1]
        normalizer = GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=data[0]["hvg_log"].shape[1],
            hvg_dim=d_eff, vocab_keep_indices=vocab_keep,
        ) if gene_norm_cfg else None
        genes_eff = _load_gene_names(prepared, vocab_keep)

        per_rep_samples: dict[str, list[dict]] = {r: [] for r in reps}
        for d_i, d in enumerate(data):
            x_norm = _prepare_input(d["hvg_log"], vocab_keep, normalizer)
            hvg_eff = d["hvg_log"][:, vocab_keep] if vocab_keep is not None else d["hvg_log"]
            emb_by_rep = _encode_representations(
                enc, x_norm, reps,
                batch=args.encode_batch,
                device=device,
                chunk_n=args.chunk_n,
                chunk_len=args.chunk_len,
                seed=d_i,
            )
            layers_str = d["layers"].astype(str)
            _, labels = np.unique(layers_str, return_inverse=True)
            # `has_truth` distinguishes labelled DLPFC samples (e.g. 151507~151674,
            # 151676) from sample 151675 which has h5 + spatial but no truth.
            # Unlabelled samples: every spot's layer is "NA", so np.unique returns
            # a single class — supervised metrics return NaN gracefully.
            has_truth = bool(np.any(layers_str != "NA"))
            for rep, emb in emb_by_rep.items():
                sample_rec = {
                    "sample_id": d["sample_id"],
                    "emb": emb,
                    "hvg_eff": hvg_eff.astype(np.float32, copy=False),
                    "coords": d["coords"].astype(np.float32, copy=False),
                    "layers_str": layers_str,
                    "labels": labels,
                    "has_truth": has_truth,
                }
                per_rep_samples[rep].append(sample_rec)
                metrics = {
                    "ckpt": ckpt_name,
                    "representation": rep,
                    "sample": d["sample_id"],
                    "n_spots": int(emb.shape[0]),
                    "has_truth": int(has_truth),
                }
                if has_truth:
                    metrics.update({f"layer_probe_{k}": v for k, v in linear_probe(emb, labels).items()})
                    if args.neural_linear_probe:
                        from mm_align.evaluation.neural_linear_probe import (
                            NeuralProbeConfig, neural_classification_probe,
                        )
                        ncfg = NeuralProbeConfig(
                            epochs=int(args.neural_probe_epochs),
                            patience=int(args.neural_probe_patience),
                            lr=float(args.neural_probe_lr),
                            weight_decay=float(args.neural_probe_weight_decay),
                            batch_size=int(args.neural_probe_batch),
                            max_spots=20000,
                            seed=d_i,
                            device=device,
                        )
                        metrics.update({
                            k.replace("neural_linear_probe/class", "neural_layer_probe"): v
                            for k, v in neural_classification_probe(
                                emb, labels, config=ncfg, prefix="neural_linear_probe/class"
                            ).items()
                        })
                    metrics.update(knn_purity(emb, labels))
                else:
                    metrics["layer_probe_acc"] = float("nan")
                    metrics["layer_probe_f1_macro"] = float("nan")
                    for k in (5, 10, 20):
                        metrics[f"knn_purity_at_{k}"] = float("nan")
                # Spatial continuity of the GT layers themselves — anchors what
                # "good" CHAOS/PAS/ASW look like on this sample.  Reported only
                # once per (sample, rep) — its rep-axis duplication is harmless
                # but it does NOT depend on `emb`.
                metrics.update(gt_layer_continuity(
                    labels, d["coords"].astype(np.float32),
                    hvg_eff, genes_eff,
                    marker_top_per_cluster=args.marker_top_per_cluster,
                    spatial_k=args.spatial_k,
                ))
                # Resolve clustering target k.
                n_layers = int(len(np.unique(labels)))
                k_for_cluster = n_layers if args.cluster_k_mode == "adaptive" else int(args.cluster_k)
                methods = [m.strip().lower() for m in args.cluster_methods.split(",") if m.strip()]
                pred_by_method: dict[str, np.ndarray] = {}
                for method in methods:
                    try:
                        cls_metrics, best_pred = clustering_metrics_one_method(
                            emb, labels, d["coords"].astype(np.float32),
                            n_clusters=k_for_cluster,
                            method=method,
                            seed=0,
                            repeats=args.cluster_repeats,
                            spatial_weight=args.spatial_cluster_weight,
                        )
                    except Exception as exc:
                        log.warning(f"cluster {method} failed ({d['sample_id']}/{rep}): {exc}")
                        cls_metrics, best_pred = (
                            {f"{method}_failed": 1.0},
                            np.zeros(emb.shape[0], dtype=int),
                        )
                    metrics.update(cls_metrics)
                    pred_by_method[method] = best_pred
                    # Marker Moran's I / Geary's C derived from THIS method's clusters.
                    try:
                        marker_m = marker_spatial_autocorr(
                            hvg_eff, d["coords"].astype(np.float32),
                            best_pred, genes_eff,
                            top_per_cluster=args.marker_top_per_cluster,
                            k=args.spatial_k,
                        )
                        for mk, mv in marker_m.items():
                            metrics[f"{method}_{mk}"] = mv
                    except Exception as exc:
                        log.warning(f"marker autocorr {method} failed ({d['sample_id']}/{rep}): {exc}")
                if include_leiden:
                    # Back-compat: also emit `leiden_ari`/`leiden_nmi` matching the
                    # pre-existing column names if Leiden was requested.
                    if "leiden" in pred_by_method:
                        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                        lp = pred_by_method["leiden"]
                        metrics["leiden_ari"] = float(adjusted_rand_score(labels, lp))
                        metrics["leiden_nmi"] = float(normalized_mutual_info_score(labels, lp))
                        metrics["leiden_k"] = float(len(np.unique(lp)))
                all_per_sample.append(metrics)
                # Save predicted clusters for downstream side-by-side figures.
                sample_rec["pred_by_method"] = pred_by_method

                if viz_dir is not None and rep == "spot_state" and d_i < args.viz_samples:
                    _plot_cluster_panel(
                        d["coords"], layers_str, pred_by_method,
                        d["sample_id"], rep,
                        viz_dir / "zero_shot" / "cluster" / "dlpfc" / ckpt_name / rep / f"{d['sample_id']}_clusters.png",
                    )

        ps = pd.DataFrame([m for m in all_per_sample if m["ckpt"] == ckpt_name])
        for rep in reps:
            sub = ps[ps["representation"] == rep]
            if sub.empty:
                continue
            agg = {
                "ckpt": ckpt_name,
                "ckpt_path": str(ck),
                "representation": rep,
                "embed_dim": int(per_rep_samples[rep][0]["emb"].shape[1]),
                "input_dim": d_eff,
                "gene_norm": (gene_norm_cfg or {}).get("mode", "none"),
                "n_samples": int(sub["sample"].nunique()),
            }
            for col in sub.columns:
                if col in {"ckpt", "representation", "sample"}:
                    continue
                if pd.api.types.is_numeric_dtype(sub[col]):
                    agg[col] = float(sub[col].mean())
            rows.append(agg)

            if viz_dir is not None and rep in gene_map_reps:
                requested_genes = list(args.genes or [])
                if args.auto_select_genes > 0:
                    auto = _auto_select_spatial_genes(
                        per_rep_samples[rep], genes_eff, int(args.auto_select_genes),
                        exclude={g.upper() for g in requested_genes},
                    )
                    log.info(f"auto-selected DLPFC genes for {ckpt_name}/{rep}: {auto}")
                    requested_genes.extend(auto)
                seen_genes = set()
                requested_genes = [g for g in requested_genes if not (g.upper() in seen_genes or seen_genes.add(g.upper()))]
                gdf = _ridge_gene_probe_and_maps(
                    per_rep_samples[rep], genes_eff, requested_genes,
                    ckpt_name=ckpt_name,
                    rep=rep,
                    out_dir=viz_dir / "linear_probe" / "gene_map" / "dlpfc",
                    max_viz_samples=args.viz_samples,
                    normalizer=normalizer,
                )
                if not gdf.empty:
                    all_gene_rows.append(gdf)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"saved {out}")
    if args.per_sample_out:
        psout = Path(args.per_sample_out)
        psout.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_per_sample).to_csv(psout, index=False)
        log.info(f"saved per-sample -> {psout}")
    if viz_dir is not None and all_gene_rows:
        gdf = pd.concat(all_gene_rows, axis=0, ignore_index=True)
        gout = viz_dir / "linear_probe" / "gene_map" / "dlpfc" / "gene_map_scc_all.csv"
        gout.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_csv(gout, index=False)
        log.info(f"saved gene-map SCC -> {gout}")
    if viz_dir is not None and all_per_sample:
        methods_seen = [m.strip().lower() for m in args.cluster_methods.split(",") if m.strip()]
        ps_df = pd.DataFrame(all_per_sample)
        # Per-ckpt method summary bar.
        for (ck_name, rep), grp in ps_df.groupby(["ckpt", "representation"]):
            _plot_method_summary_bar(
                grp, methods_seen,
                viz_dir / "zero_shot" / "cluster" / "dlpfc" / ck_name / rep / "method_summary.png",
            )

    print()
    print("-" * 96)
    print(f"DLPFC / spatialLIBD downstream [stage={args.stage}]")
    print("  representation              h_tx / chunk_state(z_chunk) / spot_state(full clean z_spot)")
    print("  layer_probe_*               supervised linear probe: embedding -> cortical layer")
    print("  zero_shot_cluster_ari/nmi   KMeans without labels vs cortical layer")
    print("  knn_purity_at_*             local layer purity in embedding space")
    if include_leiden:
        print("  leiden_ari/nmi              graph clustering benchmark, mostly Stage1.5-oriented")
    if viz_dir is not None:
        print(f"  visualizations              {viz_dir}")
    print("-" * 96)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
