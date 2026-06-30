"""Spot-level cell-type deconvolution / annotation eval for a frozen spot
encoder.

Inputs are graded — pick whichever you have:

  Mode A.  PROPORTION       : a per-spot cell-type proportion CSV
                              (`spot_id, ctype_1, ctype_2, ...`).
                              We probe Ridge( z -> proportion ) and report
                              per-spot / per-cell-type PCC, RMSE, SSIM, JSD,
                              plus aggregate stats — mirrors SpatialBenchmarking
                              GenesMetrics directly.

  Mode B.  CELLTYPE REFERENCE : a scRNA-seq reference with cell-type centroids
                              (CSV or AnnData).  We build centroid embeddings
                              by running the encoder on each centroid's mean
                              expression vector then deconvolve via cosine
                              similarity → softmax → proportions.  Compare
                              against any GT proportion CSV if provided;
                              otherwise just emit predicted proportions.

  Mode C.  HARD ANNOTATION   : per-spot hard cell-type labels.  We probe
                              LogisticRegression( z -> label ), report
                              accuracy, macro-F1, top-1 acc, plus a confusion
                              matrix figure.

Each mode is independent; pass `--mode {A,B,C}` (default = autodetect from the
provided files).  Designed to gracefully degrade when datasets are missing —
e.g. a smoke-test mode can be invoked with `--mode synthetic` to verify the
pipeline end-to-end without external data.
"""
from __future__ import annotations

import argparse
import importlib.util
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score,
                              confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.evaluation.gene_imputation_metrics import (
    pearson_1d as _pearson,
    rmse_zscore as _rmse_zscore,
    ssim_1d as _ssim_1d,
    jsd_1d as _jsd_1d,
)
from mm_align.utils import get_logger

log = get_logger("spot_deconv")

# Re-use existing ckpt loader.
_spec = importlib.util.spec_from_file_location(
    "_dc_load_helper", Path(__file__).resolve().parent / "stage1_tx.py",
)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
load_tx_encoder = _helper.load_tx_encoder


# ---------------------------------------------------------------------------
# Metric helpers (mirrors SpatialBenchmarking + stage15_gene_map)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_vectors(enc, hvg: np.ndarray, normalizer, vocab_keep,
                     *, device: str, batch: int = 256) -> np.ndarray:
    if vocab_keep is not None:
        hvg = hvg[:, vocab_keep]
    if normalizer is not None:
        hvg = normalizer.apply_np(hvg)
    outs = []
    for r0 in range(0, hvg.shape[0], batch):
        xb = torch.from_numpy(hvg[r0:r0 + batch].astype(np.float32)).to(device)
        outs.append(enc(novae_latent=None, hvg=xb)["h_tx"].detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def _encode_shard(shard: Path, enc, normalizer, vocab_keep, *,
                   device: str, batch: int = 256) -> dict:
    with h5py.File(shard, "r") as f:
        hvg = f["hvg_log"][:].astype(np.float32)
        coords = f["coords"][:].astype(np.float32)
    z = _encode_vectors(enc, hvg.copy(), normalizer, vocab_keep,
                         device=device, batch=batch)
    return {"sample_id": shard.stem, "z": z, "hvg_raw": hvg, "coords": coords}


# ---------------------------------------------------------------------------
# Mode A — supervised regression from z to proportions
# ---------------------------------------------------------------------------


def _proportion_eval(emb: np.ndarray, prop: np.ndarray, *,
                      seed: int = 0, train_frac: float = 0.7,
                      alpha: float = 1.0) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Train Ridge(emb -> prop) on a per-spot split; return pred + metrics."""
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_tr = int(n * train_frac)
    tr, te = np.sort(order[:n_tr]), np.sort(order[n_tr:])
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(emb[tr], prop[tr])
    pred = np.asarray(probe.predict(emb), dtype=np.float32)
    # Ensure valid proportions (clip nonneg + renormalise per spot).
    pred_n = np.clip(pred, 0.0, None)
    rs = pred_n.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    pred_n = pred_n / rs
    gt_te = prop[te]
    pr_te = pred_n[te]
    # Per cell-type metrics
    ct_rows = []
    for j in range(prop.shape[1]):
        ct_rows.append({
            "cell_type_index": j,
            "pcc": _pearson(gt_te[:, j], pr_te[:, j]),
            "rmse": _rmse_zscore(gt_te[:, j], pr_te[:, j]),
            "ssim": _ssim_1d(gt_te[:, j], pr_te[:, j]),
            "jsd": _jsd_1d(gt_te[:, j], pr_te[:, j]),
        })
    ct_df = pd.DataFrame(ct_rows)
    # Per-spot composition-level metrics
    spot_pcc = np.array([_pearson(gt_te[i], pr_te[i]) for i in range(gt_te.shape[0])])
    spot_jsd = np.array([_jsd_1d(gt_te[i], pr_te[i]) for i in range(gt_te.shape[0])])
    summary = {
        "pcc_celltype_mean": float(ct_df["pcc"].mean()),
        "rmse_celltype_mean": float(ct_df["rmse"].mean()),
        "ssim_celltype_mean": float(ct_df["ssim"].mean()),
        "jsd_celltype_mean": float(ct_df["jsd"].mean()),
        "pcc_spot_mean": float(np.nanmean(spot_pcc)),
        "jsd_spot_mean": float(np.nanmean(spot_jsd)),
        "n_spots_train": int(len(tr)),
        "n_spots_test": int(len(te)),
        "n_cell_types": int(prop.shape[1]),
    }
    return pred_n, summary, ct_df


# ---------------------------------------------------------------------------
# Mode B — scRNA reference centroids → cosine softmax → proportions
# ---------------------------------------------------------------------------


def _centroid_softmax_deconv(emb: np.ndarray, centroids: np.ndarray,
                              temperature: float = 1.0) -> np.ndarray:
    """proportions = softmax( cos(emb, centroid) / T ).  Output (N, K)."""
    a = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    sim = a @ c.T                                          # (N, K)
    sim = sim / max(1e-6, temperature)
    sim -= sim.max(axis=1, keepdims=True)
    e = np.exp(sim)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Mode C — supervised hard cell-type annotation probe
# ---------------------------------------------------------------------------


def _hard_label_eval(emb: np.ndarray, labels: np.ndarray, *,
                      seed: int = 0, train_frac: float = 0.7) -> tuple[np.ndarray, dict]:
    n = emb.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_tr = int(n * train_frac)
    tr, te = np.sort(order[:n_tr]), np.sort(order[n_tr:])
    pipe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, n_jobs=1,
                                             random_state=seed))
    pipe.fit(emb[tr], labels[tr])
    pred = pipe.predict(emb)
    acc_te = float(accuracy_score(labels[te], pred[te]))
    f1_te = float(f1_score(labels[te], pred[te], average="macro"))
    return pred, {"accuracy_test": acc_te, "macro_f1_test": f1_te,
                  "n_spots_train": int(len(tr)),
                  "n_spots_test": int(len(te)),
                  "n_classes": int(len(np.unique(labels)))}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_proportion_maps(coords: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                           ct_names: list[str], sample_id: str, out: Path,
                           *, n_show: int = 6) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n_show = min(n_show, gt.shape[1])
    fig, axes = plt.subplots(2, n_show, figsize=(3.0 * n_show, 6.5),
                              constrained_layout=True)
    if n_show == 1:
        axes = axes.reshape(2, 1)
    for j in range(n_show):
        for r, vals, title in [(0, gt[:, j], f"GT {ct_names[j]}"),
                                (1, pred[:, j], f"Pred {ct_names[j]}")]:
            vmin = float(np.nanpercentile(vals, 1))
            vmax = float(np.nanpercentile(vals, 99))
            ax = axes[r, j]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=6,
                             cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Deconv proportion maps  |  {sample_id}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_per_celltype_metrics(ct_df: pd.DataFrame, sample_id: str, out: Path) -> None:
    if ct_df.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, max(3, 0.25 * len(ct_df))),
                              constrained_layout=True)
    for ax, col, color in zip(axes, ["pcc", "rmse", "ssim", "jsd"],
                                ["#4c78a8", "#e15759", "#59a14f", "#f28e2b"]):
        y = np.arange(len(ct_df))
        ax.barh(y, ct_df[col].values, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(ct_df.get("cell_type", ct_df["cell_type_index"]).astype(str))
        ax.set_xlabel(col)
    fig.suptitle(f"Deconv per cell-type | {sample_id}")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_confusion(labels: np.ndarray, pred: np.ndarray, class_names: list[str],
                     sample_id: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, pred, labels=np.arange(len(class_names)))
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(0.5 + 0.4 * len(class_names),
                                       0.5 + 0.4 * len(class_names)),
                            constrained_layout=True)
    im = ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Hard annotation confusion (row-normalised) | {sample_id}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------


def _load_proportion_csv(path: Path, n_spots: int) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "spot_id" in df.columns:
        df = df.set_index("spot_id")
    arr = df.values.astype(np.float32)
    ct_names = [str(c) for c in df.columns]
    if arr.shape[0] != n_spots:
        log.warning(f"proportion CSV has {arr.shape[0]} rows but shard has {n_spots} spots — "
                    "row order assumed to match shard ordering.")
        if arr.shape[0] > n_spots:
            arr = arr[:n_spots]
    return arr, ct_names


def _load_hard_label_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    col = "label" if "label" in df.columns else df.columns[-1]
    raw = df[col].astype(str).values
    classes = sorted(set(raw))
    map_ = {c: i for i, c in enumerate(classes)}
    return np.array([map_[r] for r in raw], dtype=np.int64), classes


def _load_centroids(path: Path, gene_list: list[str]) -> tuple[np.ndarray, list[str]]:
    """Centroid matrix shape (K, n_genes_eff).  CSV with index = cell types,
    columns = gene names.  Restrict to genes that overlap our HVG vocab."""
    df = pd.read_csv(path, index_col=0)
    common = [g for g in df.columns if g in gene_list]
    if not common:
        raise SystemExit(f"No gene overlap between centroid CSV and HVG vocab — "
                          f"centroid has {len(df.columns)} genes, vocab has {len(gene_list)}.")
    log.info(f"centroid genes overlap: {len(common)}/{len(df.columns)} "
              f"(vocab={len(gene_list)})")
    idx = np.array([gene_list.index(g) for g in common], dtype=int)
    # Expand to full vocab size (zeros for missing genes).
    out = np.zeros((df.shape[0], len(gene_list)), dtype=np.float32)
    out[:, idx] = df[common].values.astype(np.float32)
    return out, [str(i) for i in df.index]


def _synthetic_shard(n_spots: int, n_genes: int, n_cell_types: int,
                      seed: int = 0) -> dict:
    """Build a synthetic spot dataset with known cell-type proportions.

    Each spot is a soft mixture of K cell types with location-dependent
    proportions.  Used only when --mode synthetic to smoke-test the pipeline
    without external data.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n_spots, 2)).astype(np.float32)
    centroids = rng.normal(0, 1, size=(n_cell_types, n_genes)).astype(np.float32)
    # Spatially smooth proportions: K Gaussian centers in xy.
    centers = rng.uniform(0, 1, size=(n_cell_types, 2)).astype(np.float32)
    sig = 0.25
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    aff = np.exp(-d2 / (2 * sig ** 2))
    prop = aff / aff.sum(axis=1, keepdims=True)
    # log1p HVG = noisy linear combo of centroids.
    hvg = prop @ centroids + 0.3 * rng.normal(size=(n_spots, n_genes)).astype(np.float32)
    hvg = np.log1p(np.clip(hvg - hvg.min(), 0, None))
    return {"sample_id": "synthetic", "coords": coords, "hvg_raw": hvg,
             "proportions": prop, "centroids": centroids,
             "cell_type_names": [f"CT{i}" for i in range(n_cell_types)]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--shards", nargs="*", default=None,
                     help="Optional explicit shard paths.  Otherwise we use one shard "
                          "from the prepared dir's `splits.json[test]`.")
    ap.add_argument("--mode", default="auto",
                     choices=("auto", "proportion", "reference", "hard", "synthetic"))
    ap.add_argument("--proportion-csv", default="",
                     help="Mode A: per-spot proportion CSV.")
    ap.add_argument("--reference-csv", default="",
                     help="Mode B: scRNA centroid CSV (rows = cell types, cols = genes).")
    ap.add_argument("--label-csv", default="",
                     help="Mode C: per-spot hard label CSV.")
    ap.add_argument("--temperature", type=float, default=0.5,
                     help="Softmax temperature for Mode B centroid deconv.")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--encode-batch", type=int, default=256)
    ap.add_argument("--out-dir", default="results/eval/spot_deconv")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--viz-n-celltypes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Mode autodetect ──
    mode = args.mode
    if mode == "auto":
        if args.proportion_csv:
            mode = "proportion"
        elif args.reference_csv:
            mode = "reference"
        elif args.label_csv:
            mode = "hard"
        else:
            mode = "synthetic"
            log.warning("No input CSV provided — running in synthetic-smoke mode.")
    log.info(f"deconv mode = {mode}")

    summary_rows = []
    for ck_s in args.ckpts:
        ck = Path(ck_s)
        ckpt_name = ck.parent.name
        log.info(f"== {ck} ==")
        if mode == "synthetic":
            # Synthetic: don't use ckpt encoder, just use the centroids as
            # the embedding directly — this validates the metric & figure
            # code path without needing real data.
            syn = _synthetic_shard(n_spots=500, n_genes=128, n_cell_types=6,
                                     seed=args.seed)
            # Use the synthetic HVG itself as a stand-in embedding (PCA to 32).
            from sklearn.decomposition import PCA
            emb = PCA(n_components=32, random_state=args.seed).fit_transform(syn["hvg_raw"])
            pred_prop, summary, ct_df = _proportion_eval(
                emb, syn["proportions"],
                seed=args.seed, train_frac=args.train_frac, alpha=args.ridge_alpha,
            )
            ct_df["cell_type"] = syn["cell_type_names"]
            sample_dir = out_root / ckpt_name / "synthetic"
            sample_dir.mkdir(parents=True, exist_ok=True)
            ct_df.to_csv(sample_dir / "per_celltype.csv", index=False)
            summary_rows.append({"ckpt": ckpt_name, "sample": "synthetic", **summary,
                                 "mode": mode})
            if not args.no_viz:
                _plot_proportion_maps(syn["coords"], syn["proportions"], pred_prop,
                                       syn["cell_type_names"], "synthetic",
                                       sample_dir / "proportion_maps.png",
                                       n_show=args.viz_n_celltypes)
                _plot_per_celltype_metrics(ct_df, "synthetic",
                                            sample_dir / "per_celltype_metrics.png")
            continue

        enc, _cfg, vocab_keep, gene_norm_cfg = load_tx_encoder(ck, device=device)
        prepared = Path(args.prepared_dir)
        with h5py.File((prepared / f"{json.loads((prepared/'splits.json').read_text())['test'][0]}.h5"),
                        "r") as f0:
            full_hvg = int(f0["hvg_log"].shape[1])
        eff = int(len(vocab_keep)) if vocab_keep is not None else full_hvg
        normalizer = (GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=full_hvg, hvg_dim=eff,
            vocab_keep_indices=vocab_keep,
        ) if gene_norm_cfg else None)

        # Resolve shards.
        if args.shards:
            shards = [Path(s) for s in args.shards]
        else:
            splits = json.loads((prepared / "splits.json").read_text())
            shards = [prepared / f"{sid}.h5" for sid in splits.get("test", [])[:1]
                       if (prepared / f"{sid}.h5").exists()]
        if not shards:
            log.warning("no shards available — skipping ckpt"); continue

        # Load HVG gene list (only needed for Mode B).
        try:
            gene_list = json.loads((prepared / "hvg_vocab.json").read_text())
            if vocab_keep is not None:
                gene_list = [gene_list[int(i)] for i in vocab_keep]
        except Exception:
            gene_list = [f"G{i}" for i in range(eff)]

        for sp in shards:
            rec = _encode_shard(sp, enc, normalizer, vocab_keep,
                                 device=device, batch=args.encode_batch)
            sample_dir = out_root / ckpt_name / rec["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)

            if mode == "proportion":
                prop, ct_names = _load_proportion_csv(Path(args.proportion_csv),
                                                       rec["z"].shape[0])
                pred_prop, summary, ct_df = _proportion_eval(
                    rec["z"], prop,
                    seed=args.seed, train_frac=args.train_frac, alpha=args.ridge_alpha,
                )
                ct_df["cell_type"] = ct_names
                ct_df.to_csv(sample_dir / "per_celltype.csv", index=False)
                summary_rows.append({"ckpt": ckpt_name, "sample": rec["sample_id"],
                                     **summary, "mode": mode})
                if not args.no_viz:
                    _plot_proportion_maps(rec["coords"], prop, pred_prop, ct_names,
                                           rec["sample_id"],
                                           sample_dir / "proportion_maps.png",
                                           n_show=args.viz_n_celltypes)
                    _plot_per_celltype_metrics(ct_df, rec["sample_id"],
                                                sample_dir / "per_celltype_metrics.png")

            elif mode == "reference":
                centroid_full, ct_names = _load_centroids(Path(args.reference_csv),
                                                            gene_list)
                z_cent = _encode_vectors(enc, centroid_full, normalizer, vocab_keep,
                                          device=device, batch=args.encode_batch)
                pred_prop = _centroid_softmax_deconv(rec["z"], z_cent,
                                                      temperature=args.temperature)
                pd.DataFrame(pred_prop, columns=ct_names).to_csv(
                    sample_dir / "predicted_proportions.csv", index=False,
                )
                summary = {"n_spots": int(rec["z"].shape[0]),
                            "n_cell_types": int(len(ct_names)),
                            "temperature": float(args.temperature)}
                if args.proportion_csv:
                    gt_prop, gt_ct = _load_proportion_csv(Path(args.proportion_csv),
                                                            rec["z"].shape[0])
                    # Align columns by name where possible.
                    if gt_ct == ct_names:
                        ct_df = pd.DataFrame({
                            "cell_type": ct_names,
                            "pcc": [_pearson(gt_prop[:, j], pred_prop[:, j])
                                     for j in range(len(ct_names))],
                            "rmse": [_rmse_zscore(gt_prop[:, j], pred_prop[:, j])
                                     for j in range(len(ct_names))],
                            "ssim": [_ssim_1d(gt_prop[:, j], pred_prop[:, j])
                                     for j in range(len(ct_names))],
                            "jsd": [_jsd_1d(gt_prop[:, j], pred_prop[:, j])
                                     for j in range(len(ct_names))],
                        })
                        ct_df.to_csv(sample_dir / "per_celltype.csv", index=False)
                        summary.update({
                            "pcc_celltype_mean": float(ct_df["pcc"].mean()),
                            "rmse_celltype_mean": float(ct_df["rmse"].mean()),
                            "ssim_celltype_mean": float(ct_df["ssim"].mean()),
                            "jsd_celltype_mean": float(ct_df["jsd"].mean()),
                        })
                        if not args.no_viz:
                            _plot_proportion_maps(rec["coords"], gt_prop, pred_prop, ct_names,
                                                    rec["sample_id"],
                                                    sample_dir / "proportion_maps.png",
                                                    n_show=args.viz_n_celltypes)
                            _plot_per_celltype_metrics(ct_df, rec["sample_id"],
                                                        sample_dir / "per_celltype_metrics.png")
                    else:
                        log.warning(f"GT cell-types differ from centroid cell-types; "
                                     f"GT={gt_ct} CENT={ct_names} — skipping metric overlay.")
                summary_rows.append({"ckpt": ckpt_name, "sample": rec["sample_id"],
                                     **summary, "mode": mode})

            elif mode == "hard":
                labels, class_names = _load_hard_label_csv(Path(args.label_csv))
                if len(labels) != rec["z"].shape[0]:
                    log.warning(f"label rows ({len(labels)}) != n_spots ({rec['z'].shape[0]}) "
                                 "— truncating to min.")
                    n = min(len(labels), rec["z"].shape[0])
                    labels = labels[:n]
                    rec["z"] = rec["z"][:n]
                    rec["coords"] = rec["coords"][:n]
                pred, summary = _hard_label_eval(
                    rec["z"], labels, seed=args.seed, train_frac=args.train_frac,
                )
                summary_rows.append({"ckpt": ckpt_name, "sample": rec["sample_id"],
                                     **summary, "mode": mode})
                if not args.no_viz:
                    _plot_confusion(labels, pred, class_names, rec["sample_id"],
                                     sample_dir / "confusion.png")

    df = pd.DataFrame(summary_rows)
    csv_path = out_root / f"spot_deconv_{mode}_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"saved {csv_path}")
    print()
    print("-" * 96)
    print(f"Spot deconvolution / annotation eval [mode={mode}]")
    print("-" * 96)
    if not df.empty:
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
