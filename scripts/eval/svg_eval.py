"""Spatially-Variable-Gene (SVG) ranking evaluation for the spot encoder.

The spot encoder is foundation-scale: the question we want to answer is
"does the model preserve genes that are spatially structured?".  We answer
it by comparing two SVG rankings per sample:

  GT ranking      : Moran's I (and Geary's C) on the actual log1p HVG of
                    every gene over the sample's spatial KNN graph.
  Predicted ranking : Moran's I / Geary's C on a Ridge-probe's predicted
                    expression maps  (z  →  ĝ),  where  z  is either
                    h_tx, z_spot, or the Stage 1.5 spatial embedding.

For each sample we report
  - kendall_tau, spearman, pearson      between rank vectors
  - top_k_overlap                       for k ∈ {25, 50, 100, 200}
  - aupr                                — area under PR curve treating
                                          GT top-K as positive class
  - per_gene rows                       — full SVG rankings (Moran's I,
                                          Geary's C, gt vs pred ranks)

Figures (when --viz-dir is set):
  - top SVG gene spatial maps (GT vs Pred)
  - GT vs Pred rank scatter (per sample)
  - top-K overlap barplot

NO squidpy — pure scanpy + numpy.  Inputs: a Stage-1 or Stage-1.5 ckpt and
the prepared shard directory (HEST or DLPFC).
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
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.evaluation.spatial_metrics import (
    morans_i_per_gene, gearys_c_per_gene,
)
from mm_align.utils import get_logger

log = get_logger("svg_eval")

# Re-use stage1_tx's ckpt-recovery helpers (same pattern as dlpfc_eval).
_spec = importlib.util.spec_from_file_location(
    "_svg_load_helper", Path(__file__).resolve().parent / "stage1_tx.py",
)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
load_tx_encoder = _helper.load_tx_encoder


# ---------------------------------------------------------------------------
# Rank-comparison helpers
# ---------------------------------------------------------------------------


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau
    if a.size < 3:
        return float("nan")
    tau, _ = kendalltau(a, b, nan_policy="omit")
    return float(tau) if np.isfinite(tau) else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if a.size < 3:
        return float("nan")
    r, _ = spearmanr(a, b, nan_policy="omit")
    return float(r) if np.isfinite(r) else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _top_k_overlap(gt_rank: np.ndarray, pred_rank: np.ndarray, k: int) -> float:
    n = len(gt_rank)
    if n < k + 1:
        return float("nan")
    top_gt = set(np.argsort(gt_rank)[:k])
    top_pr = set(np.argsort(pred_rank)[:k])
    return float(len(top_gt & top_pr) / k)


def _aupr_top_k(gt_morans: np.ndarray, pred_morans: np.ndarray, top_k: int) -> float:
    """Treat the top-K GT-Moran's I genes as positives; pred Moran's I as score."""
    gt = np.asarray(gt_morans, dtype=np.float64)
    pr = np.asarray(pred_morans, dtype=np.float64)
    valid = np.isfinite(gt) & np.isfinite(pr)
    if valid.sum() < top_k + 1:
        return float("nan")
    gt, pr = gt[valid], pr[valid]
    order = np.argsort(-gt)
    pos = np.zeros(len(gt), dtype=int)
    pos[order[:top_k]] = 1
    if pos.sum() == 0 or pos.sum() == len(pos):
        return float("nan")
    try:
        return float(average_precision_score(pos, pr))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Ridge-probe prediction of every HVG from the encoder embedding
# ---------------------------------------------------------------------------


def _ridge_predict_all_hvg(emb_train: np.ndarray, hvg_train: np.ndarray,
                            emb_eval: np.ndarray,
                            *, alpha: float = 1.0,
                            max_train_spots: int = 8000,
                            seed: int = 0) -> np.ndarray:
    """Train a Ridge from emb -> hvg on train spots, predict on eval spots.

    Heavy-multi-output Ridge — we train once on all genes simultaneously.
    """
    rng = np.random.default_rng(seed)
    n = emb_train.shape[0]
    if n > max_train_spots:
        idx = np.sort(rng.choice(n, max_train_spots, replace=False))
        emb_train = emb_train[idx]
        hvg_train = hvg_train[idx]
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(emb_train, hvg_train)
    return np.asarray(probe.predict(emb_eval), dtype=np.float32)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_rank_scatter(gt_rank: np.ndarray, pred_rank: np.ndarray,
                        sample_id: str, k_marks: tuple[int, ...],
                        out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(gt_rank, pred_rank, s=4, alpha=0.4)
    ax.plot([0, gt_rank.max()], [0, gt_rank.max()], color="0.4", linewidth=1)
    for k in k_marks:
        ax.axhline(k, color="0.7", linestyle=":", linewidth=0.7)
        ax.axvline(k, color="0.7", linestyle=":", linewidth=0.7)
    ax.set_xlabel("GT Moran's I rank (1 = most spatial)")
    ax.set_ylabel("Pred Moran's I rank")
    ax.set_title(f"SVG rank: GT vs Pred | {sample_id}")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_top_overlap_bar(overlap_table: pd.DataFrame, out: Path) -> None:
    if overlap_table.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    pivot = overlap_table.pivot_table(index="sample_id", columns="k",
                                       values="overlap", aggfunc="mean")
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, pivot.shape[1])
    for j, k in enumerate(pivot.columns):
        ax.bar(x + j * width - 0.4 + width / 2, pivot[k].values, width=width, label=f"k={k}")
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("top-K overlap with GT SVG ranking")
    ax.set_title("SVG top-K overlap by sample")
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_top_svg_maps(coords: np.ndarray, hvg: np.ndarray, pred: np.ndarray,
                       gene_names: list[str], gt_top_idx: np.ndarray,
                       sample_id: str, out: Path, *, n: int = 6) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = min(n, len(gt_top_idx))
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.0), constrained_layout=True)
    for j in range(n):
        gi = int(gt_top_idx[j])
        gname = gene_names[gi]
        for r, vals, title in [(0, hvg[:, gi], f"GT {gname}"),
                                (1, pred[:, gi], f"Pred {gname}")]:
            ax = axes[r, j]
            vmin = float(np.nanpercentile(vals, 1))
            vmax = float(np.nanpercentile(vals, 99))
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=6,
                             cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Top SVG by GT Moran's I  |  {sample_id}")
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sample helper — load a shard, build embedding via frozen Stage-1 encoder
# ---------------------------------------------------------------------------


def _encode_one_shard(shard: Path, enc, normalizer, vocab_keep, *,
                       device: str, batch: int = 256) -> dict:
    with h5py.File(shard, "r") as f:
        hvg = f["hvg_log"][:].astype(np.float32)
        coords = f["coords"][:].astype(np.float32)
    if vocab_keep is not None:
        hvg = hvg[:, vocab_keep]
    if normalizer is not None:
        x_norm = normalizer.apply_np(hvg)
    else:
        x_norm = hvg
    outs = []
    with torch.no_grad():
        for r0 in range(0, x_norm.shape[0], batch):
            xb = torch.from_numpy(x_norm[r0:r0 + batch]).to(device)
            outs.append(enc(novae_latent=None, hvg=xb)["h_tx"].detach().cpu().numpy())
    z = np.concatenate(outs, axis=0).astype(np.float32)
    return {"sample_id": shard.stem, "z": z, "hvg_eff": hvg.astype(np.float32),
             "coords": coords}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--ckpts", nargs="+", required=True,
                     help="Stage-1 (tx-only) ckpts — chunk/spot encoders alike.")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--source", default="hest", choices=("hest", "all"))
    ap.add_argument("--samples", nargs="*", default=None,
                     help="Optional sample ids; overrides --split.")
    ap.add_argument("--max-samples", type=int, default=5,
                     help="Cap number of shards used for the eval.")
    ap.add_argument("--spatial-k", type=int, default=6,
                     help="KNN for spatial weight matrix.")
    ap.add_argument("--top-k", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--aupr-top-k", type=int, default=200)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--ridge-train-spots", type=int, default=8000)
    ap.add_argument("--ridge-train-frac", type=float, default=0.7,
                     help="Fraction of spots per shard reserved for fitting Ridge.")
    ap.add_argument("--encode-batch", type=int, default=256)
    ap.add_argument("--out-dir", default="results/eval/svg_eval")
    ap.add_argument("--viz-samples", type=int, default=3)
    ap.add_argument("--viz-top-n", type=int, default=6,
                     help="Number of top GT SVG genes to render maps for.")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = Path(args.prepared_dir)
    splits = json.loads((prepared / "splits.json").read_text())

    if args.samples:
        shard_ids = args.samples
    else:
        shard_ids = splits.get(args.split, [])
    # Drop shards with placeholder (zero / identically degenerate) coords —
    # they collapse the SVG figure to a single dot and produce meaningless
    # Moran's I rankings.  Stage 1.5 does the same filter via _has_real_coords.
    def _has_real_coords(p: Path) -> bool:
        try:
            with h5py.File(p, "r") as f:
                if "coords" not in f:
                    return False
                xy = f["coords"][: min(256, f["coords"].shape[0])]
                if (xy ** 2).sum() <= 0.0:
                    return False
                # Reject if all spots collapse onto one location.
                if xy.std(axis=0).sum() < 1e-6:
                    return False
                return True
        except Exception:
            return False

    shards: list[Path] = []
    skipped_no_coords = 0
    for sid in shard_ids[: max(args.max_samples * 8, args.max_samples)]:
        candidates: list[Path] = []
        if args.source == "hest":
            p = prepared / f"{sid}.h5"
            if p.exists():
                candidates.append(p)
        else:
            for suf in ("", ".st1k", ".spatialcorpus"):
                p = prepared / f"{sid}{suf}.h5"
                if p.exists():
                    candidates.append(p)
                    break
        for c in candidates:
            if _has_real_coords(c):
                shards.append(c)
                break
            else:
                skipped_no_coords += 1
        if len(shards) >= args.max_samples:
            break
    if not shards:
        raise SystemExit(f"No shards with real spatial coords for split={args.split} source={args.source}")
    log.info(f"using {len(shards)} shards for SVG eval (skipped {skipped_no_coords} zero-coord)")

    # Pre-compute GT Moran's I / Geary's C per shard, independent of ckpt.
    gt_per_shard: dict[str, dict] = {}
    with h5py.File(shards[0], "r") as f0:
        full_hvg_dim = int(f0["hvg_log"].shape[1])

    rows = []
    overlap_rows = []
    per_gene_rows = []
    rng = np.random.default_rng(args.seed)

    for ck_s in args.ckpts:
        ck = Path(ck_s)
        ckpt_name = ck.parent.name
        log.info(f"== {ck} ==")
        enc, _cfg, vocab_keep, gene_norm_cfg = load_tx_encoder(ck, device=device)
        eff_dim = int(len(vocab_keep)) if vocab_keep is not None else full_hvg_dim
        normalizer = (GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=full_hvg_dim,
            hvg_dim=eff_dim, vocab_keep_indices=vocab_keep,
        ) if gene_norm_cfg else None)
        # gene names
        try:
            from json import loads as _loads
            gene_names = _loads((prepared / "hvg_vocab.json").read_text())
            if vocab_keep is not None:
                gene_names = [gene_names[int(i)] for i in vocab_keep]
        except Exception:
            gene_names = [f"G{i}" for i in range(eff_dim)]

        for shi, shard in enumerate(shards):
            rec = _encode_one_shard(shard, enc, normalizer, vocab_keep,
                                     device=device, batch=args.encode_batch)
            n = rec["z"].shape[0]
            if n < args.spatial_k + 2:
                log.warning(f"  {rec['sample_id']}: too few spots ({n}), skipping.")
                continue
            # Train/test split for Ridge.
            order = rng.permutation(n)
            n_tr = int(n * args.ridge_train_frac)
            tr_idx = np.sort(order[:n_tr])
            te_idx = np.sort(order[n_tr:])
            if len(te_idx) < args.spatial_k + 2:
                te_idx = order                                # fallback: full
            pred = _ridge_predict_all_hvg(
                emb_train=rec["z"][tr_idx],
                hvg_train=rec["hvg_eff"][tr_idx],
                emb_eval=rec["z"],
                alpha=args.ridge_alpha,
                max_train_spots=args.ridge_train_spots,
                seed=args.seed,
            )
            # Moran's I / Geary's C on the EVAL spots only — both for GT and
            # for predicted maps.  Geary's C is reported but ranking uses
            # Moran's I as primary (matches SVG_Benchmarking convention).
            gt_morans = morans_i_per_gene(
                rec["hvg_eff"][te_idx], rec["coords"][te_idx], k=args.spatial_k,
            )
            gt_gearys = gearys_c_per_gene(
                rec["hvg_eff"][te_idx], rec["coords"][te_idx], k=args.spatial_k,
            )
            pred_morans = morans_i_per_gene(
                pred[te_idx], rec["coords"][te_idx], k=args.spatial_k,
            )
            pred_gearys = gearys_c_per_gene(
                pred[te_idx], rec["coords"][te_idx], k=args.spatial_k,
            )
            # Higher Moran's I = more spatial → rank 1 = most spatial.
            gt_rank = np.argsort(np.argsort(-gt_morans, kind="stable"), kind="stable").astype(np.int64) + 1
            pred_rank = np.argsort(np.argsort(-pred_morans, kind="stable"), kind="stable").astype(np.int64) + 1

            row = {
                "ckpt": ckpt_name,
                "sample": rec["sample_id"],
                "n_spots_eval": int(len(te_idx)),
                "n_genes": int(len(gt_morans)),
                "kendall_tau_morans": _kendall_tau(gt_morans, pred_morans),
                "spearman_morans": _spearman(gt_morans, pred_morans),
                "pearson_morans": _pearson(gt_morans, pred_morans),
                "spearman_gearys": _spearman(gt_gearys, pred_gearys),
                "aupr_top_k": _aupr_top_k(gt_morans, pred_morans, args.aupr_top_k),
                "aupr_top_k_value": int(args.aupr_top_k),
            }
            for k in args.top_k:
                ov = _top_k_overlap(gt_rank, pred_rank, k)
                row[f"top_{k}_overlap"] = ov
                overlap_rows.append({"ckpt": ckpt_name, "sample_id": rec["sample_id"],
                                      "k": int(k), "overlap": float(ov)})
            rows.append(row)

            # Per-gene table — top-200 by GT Moran's I for compactness.
            order_g = np.argsort(-gt_morans)[:200]
            for gi in order_g:
                per_gene_rows.append({
                    "ckpt": ckpt_name,
                    "sample": rec["sample_id"],
                    "gene": gene_names[int(gi)],
                    "gt_morans_i": float(gt_morans[gi]),
                    "pred_morans_i": float(pred_morans[gi]),
                    "gt_gearys_c": float(gt_gearys[gi]),
                    "pred_gearys_c": float(pred_gearys[gi]),
                    "gt_rank": int(gt_rank[gi]),
                    "pred_rank": int(pred_rank[gi]),
                })

            if not args.no_viz and shi < args.viz_samples:
                ck_dir = out_dir / ckpt_name
                _plot_rank_scatter(
                    gt_rank, pred_rank, rec["sample_id"],
                    tuple(args.top_k),
                    ck_dir / "rank_scatter" / f"{rec['sample_id']}.png",
                )
                _plot_top_svg_maps(
                    rec["coords"][te_idx], rec["hvg_eff"][te_idx], pred[te_idx],
                    gene_names, np.argsort(-gt_morans)[: args.viz_top_n],
                    rec["sample_id"],
                    ck_dir / "top_svg_maps" / f"{rec['sample_id']}.png",
                    n=args.viz_top_n,
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "svg_eval.csv", index=False)
    log.info(f"saved {out_dir/'svg_eval.csv'}")
    pd.DataFrame(per_gene_rows).to_csv(out_dir / "svg_per_gene.csv", index=False)
    log.info(f"saved {out_dir/'svg_per_gene.csv'}")
    pd.DataFrame(overlap_rows).to_csv(out_dir / "svg_overlap.csv", index=False)

    if not args.no_viz:
        for ckpt_name, grp in pd.DataFrame(overlap_rows).groupby("ckpt"):
            _plot_top_overlap_bar(grp, out_dir / ckpt_name / "top_k_overlap.png")

    print()
    print("-" * 96)
    print("SVG eval — GT vs Ridge-predicted Moran's I rankings")
    print("  kendall_tau / spearman / pearson over per-gene Moran's I across all HVGs")
    print(f"  top_K_overlap for K = {args.top_k}")
    print(f"  AUPR with GT top-{args.aupr_top_k} as positive")
    print("-" * 96)
    if not df.empty:
        cols = ["ckpt", "sample", "n_genes", "kendall_tau_morans",
                "spearman_morans", "aupr_top_k"] + [f"top_{k}_overlap" for k in args.top_k]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
