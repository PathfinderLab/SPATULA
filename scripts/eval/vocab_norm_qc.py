"""Vocab and normalization QC tables + figures.

This script evaluates two pre-model choices:

1. Vocab construction / clipping
   - Does a clipped vocab keep marker genes and high-priority genes?
   - How much prevalence / dispersion signal is retained?

2. Gene normalization modes
   - How do value distributions change under none / global_median / nonzero_z?
   - How much clipping or scale inflation does each mode introduce?

Outputs:
  results/eval/vocab_quality.csv
  results/eval/normalization_quality.csv
  results/figures/vocab_norm_qc/*.png
  results/figures/vocab_norm_qc/vocab_norm_qc_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.utils import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))
try:
    from vocab_qc import MARKER_PANELS
except Exception:
    MARKER_PANELS = {}

log = get_logger("vocab_norm_qc")


def _load_vocab(prepared: Path) -> list[str]:
    return json.loads((prepared / "hvg_vocab.json").read_text())


def _keep_indices(prepared: Path, vocab_name: str, full_dim: int) -> np.ndarray | None:
    if vocab_name == "full":
        return None
    path = prepared / f"clip{vocab_name}_keep_indices.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing vocab clip indices: {path}")
    idx = np.load(path)
    if idx.ndim != 1 or idx.size <= 0 or int(idx.max()) >= full_dim:
        raise ValueError(f"bad keep indices for {vocab_name}: shape={idx.shape}")
    return idx.astype(np.int64)


def _panel_coverage(genes: set[str]) -> tuple[float, int, int]:
    if not MARKER_PANELS:
        return float("nan"), 0, 0
    all_markers = sorted({g for panel in MARKER_PANELS.values() for g in panel})
    kept = sum(1 for g in all_markers if g in genes)
    return kept / max(len(all_markers), 1), kept, len(all_markers)


def vocab_quality_table(prepared: Path, vocab_names: list[str]) -> pd.DataFrame:
    vocab = _load_vocab(prepared)
    full_dim = len(vocab)
    vdf = pd.read_csv(prepared / "vocab.csv")
    gene_to_row = {g: i for i, g in enumerate(vdf["gene"].tolist())}
    rows = []
    for name in vocab_names:
        keep = _keep_indices(prepared, name, full_dim)
        genes = vocab if keep is None else [vocab[int(i)] for i in keep]
        sub = vdf.iloc[[gene_to_row[g] for g in genes if g in gene_to_row]].copy()
        gset = set(genes)
        marker_cov, marker_kept, marker_total = _panel_coverage(gset)
        n = max(len(sub), 1)
        rows.append({
            "vocab": name,
            "n_genes": len(genes),
            "protein_coding_pct": float((sub["gene_type"] == "protein_coding").mean()),
            "must_include_kept": int(sub["must_include"].sum()) if "must_include" in sub else 0,
            "must_include_pct": float(sub["must_include"].mean()) if "must_include" in sub else float("nan"),
            "marker_panel_coverage": marker_cov,
            "marker_panel_kept": marker_kept,
            "marker_panel_total": marker_total,
            "sample_prev_median": float(sub["sample_prev"].median()),
            "sample_prev_mean": float(sub["sample_prev"].mean()),
            "spot_prev_median": float(sub["spot_prev"].median()),
            "spot_prev_mean": float(sub["spot_prev"].mean()),
            "rare_sample_pct": float((sub["sample_prev"] < 0.10).sum() / n),
            "rare_spot_pct": float((sub["spot_prev"] < 0.01).sum() / n),
            "norm_dispersion_median": float(sub["norm_dispersion"].median()),
            "norm_dispersion_mean": float(sub["norm_dispersion"].mean()),
            "priority_rank_max": int(sub["priority_rank"].max()),
        })
    return pd.DataFrame(rows)


def _source_of(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".spatialcorpus"):
        return "spatialcorpus"
    if stem.endswith(".st1k"):
        return "st1k"
    return "hest"


def _sample_shards(prepared: Path, split: str, max_shards: int, seed: int) -> list[Path]:
    splits = json.loads((prepared / "splits.json").read_text())
    ids = list(splits.get(split, []))
    paths = []
    for sid in ids:
        for suf in ("", ".st1k", ".spatialcorpus"):
            p = prepared / f"{sid}{suf}.h5"
            if p.exists():
                paths.append(p)
                break
    rng = np.random.default_rng(seed)
    if len(paths) > max_shards:
        paths = [paths[i] for i in rng.choice(len(paths), max_shards, replace=False)]
    return sorted(paths)


def _load_sample_matrix(shards: list[Path], max_spots_per_shard: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, srcs = [], []
    for p in shards:
        with h5py.File(p, "r") as f:
            x = np.asarray(f["hvg_log"][:], dtype=np.float32)
        if x.shape[0] > max_spots_per_shard:
            sel = rng.choice(x.shape[0], max_spots_per_shard, replace=False)
            x = x[sel]
        xs.append(x)
        srcs.extend([_source_of(p)] * x.shape[0])
    if not xs:
        raise RuntimeError("no spots sampled")
    return np.concatenate(xs, axis=0), np.asarray(srcs)


def _normalizer_cfg(mode: str, stats_path: Path) -> dict | None:
    if mode == "none":
        return None
    return {
        "mode": mode,
        "stats_path": str(stats_path),
        "eps": 1e-6,
        "min_scale": 0.05,
        "clip": 8.0,
    }


def _value_stats(x: np.ndarray) -> dict[str, float]:
    nz = x[x != 0]
    flat = x.reshape(-1)
    if nz.size == 0:
        nz = flat
    spot_sum = x.sum(axis=1)
    spot_nnz = (x != 0).sum(axis=1)
    gene_nz_frac = (x != 0).mean(axis=0)
    gene_mean = x.mean(axis=0)
    out = {
        "value_mean": float(np.mean(flat)),
        "value_std": float(np.std(flat)),
        "value_p01": float(np.percentile(flat, 1)),
        "value_p50": float(np.percentile(flat, 50)),
        "value_p99": float(np.percentile(flat, 99)),
        "nonzero_value_mean": float(np.mean(nz)),
        "nonzero_value_std": float(np.std(nz)),
        "nonzero_value_p50": float(np.percentile(nz, 50)),
        "nonzero_value_p99": float(np.percentile(nz, 99)),
        "zero_fraction": float((x == 0).mean()),
        "spot_sum_mean": float(np.mean(spot_sum)),
        "spot_sum_cv": float(np.std(spot_sum) / max(abs(np.mean(spot_sum)), 1e-12)),
        "spot_nnz_mean": float(np.mean(spot_nnz)),
        "gene_detected_median": float(np.median(gene_nz_frac)),
        "gene_mean_cv": float(np.std(gene_mean) / max(abs(np.mean(gene_mean)), 1e-12)),
        "clip_abs8_pct": float((np.abs(x) >= 7.999).mean()),
    }
    return out


def normalization_quality_table(
    prepared: Path,
    base_x: np.ndarray,
    vocab_names: list[str],
    modes: list[str],
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    full_dim = base_x.shape[1]
    stats_path = prepared / "gene_stats.npz"
    rows = []
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for vocab_name in vocab_names:
        keep = _keep_indices(prepared, vocab_name, full_dim)
        x_vocab = base_x if keep is None else base_x[:, keep]
        eff_dim = x_vocab.shape[1]
        for mode in modes:
            normalizer = GeneNormalizer(
                _normalizer_cfg(mode, stats_path),
                full_hvg_dim=full_dim,
                hvg_dim=eff_dim,
                vocab_keep_indices=keep,
            )
            x_norm = normalizer.apply_np(x_vocab.copy())
            matrices[(vocab_name, mode)] = x_norm
            row = {
                "vocab": vocab_name,
                "normalization": mode,
                "n_spots": int(x_norm.shape[0]),
                "n_genes": int(x_norm.shape[1]),
            }
            row.update(_value_stats(x_norm))
            rows.append(row)
    return pd.DataFrame(rows), matrices



def _rankdata_ordinal(x: np.ndarray) -> np.ndarray:
    """Fast ordinal ranks for one vector; ties are rare after normalization."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(x.size, dtype=np.float32)
    return ranks


def _spearman_rank_vectors(rx: np.ndarray, ry: np.ndarray) -> float:
    rx = rx.astype(np.float64, copy=False)
    ry = ry.astype(np.float64, copy=False)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / den)


def normalization_rank_shift_table(
    matrices: dict[tuple[str, str], np.ndarray],
    vocab_names: list[str],
    modes: list[str],
    *,
    max_spots: int = 1200,
    seed: int = 0,
    topks: tuple[int, ...] = (10, 50, 100),
) -> pd.DataFrame:
    """How much per-spot gene ranking changes after normalization.

    The prepared shards do not keep raw integer counts. We therefore use
    hvg_log / none as the monotonic proxy for spot-normalized raw abundance.
    Within a spot, log1p and expm1 preserve rank, so rank comparisons remain
    meaningful.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for vocab_name in vocab_names:
        raw = matrices[(vocab_name, "none")]
        n = raw.shape[0]
        spot_idx = np.arange(n)
        if n > max_spots:
            spot_idx = rng.choice(n, max_spots, replace=False)
        for mode in modes:
            x = matrices[(vocab_name, mode)]
            rank_corrs = []
            abs_shift_norm = []
            top_overlap = {k: [] for k in topks}
            n_nonzero = []
            for i in spot_idx:
                mask = raw[i] > 0
                nnz = int(mask.sum())
                if nnz < 5:
                    continue
                r0 = raw[i, mask]
                r1 = x[i, mask]
                rr0 = _rankdata_ordinal(r0)
                rr1 = _rankdata_ordinal(r1)
                rank_corrs.append(_spearman_rank_vectors(rr0, rr1))
                denom = max(nnz - 1, 1)
                abs_shift_norm.append(float(np.mean(np.abs(rr0 - rr1)) / denom))
                for k in topks:
                    kk = min(k, nnz)
                    a = set(np.argpartition(r0, -kk)[-kk:].tolist())
                    b = set(np.argpartition(r1, -kk)[-kk:].tolist())
                    top_overlap[k].append(len(a & b) / kk)
                n_nonzero.append(nnz)
            row = {
                "vocab": vocab_name,
                "normalization": mode,
                "rank_reference": "none/hvg_log",
                "n_spots": int(len(rank_corrs)),
                "nonzero_genes_mean": float(np.mean(n_nonzero)) if n_nonzero else float("nan"),
                "rank_spearman_mean": float(np.nanmean(rank_corrs)) if rank_corrs else float("nan"),
                "rank_spearman_median": float(np.nanmedian(rank_corrs)) if rank_corrs else float("nan"),
                "mean_abs_rank_shift_norm": float(np.nanmean(abs_shift_norm)) if abs_shift_norm else float("nan"),
            }
            for k in topks:
                row[f"top{k}_overlap_mean"] = float(np.nanmean(top_overlap[k])) if top_overlap[k] else float("nan")
                row[f"top{k}_changed_mean"] = 1.0 - row[f"top{k}_overlap_mean"]
            rows.append(row)
    return pd.DataFrame(rows)

def _fig_vocab_quality(vdf: pd.DataFrame, out: Path) -> None:
    order = vdf["vocab"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("n_genes", "#4c78a8", "Vocab size"),
        ("marker_panel_coverage", "#59a14f", "Curated marker coverage"),
        ("protein_coding_pct", "#59a14f", "Protein-coding fraction"),
        ("sample_prev_median", "#f28e2b", "Median sample prevalence"),
        ("spot_prev_median", "#f28e2b", "Median spot prevalence"),
        ("norm_dispersion_median", "#b07aa1", "Median normalized dispersion"),
    ]
    for ax, (col, color, title) in zip(axes.ravel(), panels):
        vals = vdf[col].to_numpy(dtype=float)
        ax.bar(order, vals, color=color, alpha=0.86)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        if "pct" in col or "coverage" in col:
            ax.set_ylim(0, 1.05)
        for i, v in enumerate(vals):
            label = f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
            ax.text(i, v, label, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Vocab quality by clipping option", y=1.02)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig_vocab_scatter(prepared: Path, vocab_names: list[str], out: Path) -> None:
    vocab = _load_vocab(prepared)
    full_dim = len(vocab)
    vdf = pd.read_csv(prepared / "vocab.csv").copy()
    vdf["clip_group"] = "outside"
    colors = {"4096": "#59a14f", "8192": "#f28e2b", "full": "#4c78a8", "outside": "#bbbbbb"}
    # Mark smallest/highest priority clip first.
    for name in reversed(vocab_names):
        if name == "full":
            continue
        keep = set(_keep_indices(prepared, name, full_dim).tolist())
        idx_to_gene = {i: g for i, g in enumerate(vocab)}
        genes = {idx_to_gene[i] for i in keep}
        vdf.loc[vdf["gene"].isin(genes), "clip_group"] = name
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for group, sub in vdf.groupby("clip_group"):
        c = colors.get(group, "#999999")
        alpha = 0.18 if group == "outside" else 0.75
        axes[0].scatter(sub["sample_prev"], sub["norm_dispersion"], s=5, alpha=alpha, label=group, color=c)
        axes[1].scatter(sub["spot_prev"], sub["norm_dispersion"], s=5, alpha=alpha, label=group, color=c)
    axes[0].set_xlabel("sample prevalence")
    axes[1].set_xlabel("spot prevalence")
    for ax in axes:
        ax.set_ylabel("normalized dispersion")
        ax.set_yscale("log")
        ax.legend(markerscale=3)
    axes[0].set_title("Sample prevalence vs dispersion")
    axes[1].set_title("Spot prevalence vs dispersion")
    fig.suptitle("Which genes are retained by vocab clipping?")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig_norm_distributions(matrices: dict[tuple[str, str], np.ndarray], vocab_name: str, modes: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(modes), figsize=(4.5 * len(modes), 4), constrained_layout=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        x = matrices[(vocab_name, mode)]
        nz = x[x != 0]
        if nz.size > 400000:
            rng = np.random.default_rng(0)
            nz = nz[rng.choice(nz.size, 400000, replace=False)]
        ax.hist(nz, bins=100, color="#4c78a8", alpha=0.84)
        ax.axvline(np.median(nz), color="black", ls="--", lw=1, label=f"median={np.median(nz):.2f}")
        ax.set_title(mode)
        ax.set_xlabel("nonzero value")
        ax.set_ylabel("# entries")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(f"Normalization value distributions | vocab={vocab_name}")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig_norm_metrics(ndf: pd.DataFrame, vocab_name: str, out: Path) -> None:
    sub = ndf[ndf["vocab"] == vocab_name].copy()
    order = sub["normalization"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("nonzero_value_p99", "Nonzero p99"),
        ("spot_sum_cv", "Spot-sum CV"),
        ("gene_mean_cv", "Gene-mean CV"),
        ("zero_fraction", "Zero fraction"),
        ("clip_abs8_pct", "Clipped at |8| fraction"),
        ("nonzero_value_std", "Nonzero std"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels):
        vals = sub[col].to_numpy(dtype=float)
        ax.bar(order, vals, color="#e15759", alpha=0.82)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Normalization quantitative effects | vocab={vocab_name}")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig_norm_source_shift(
    matrices: dict[tuple[str, str], np.ndarray],
    sources: np.ndarray,
    vocab_name: str,
    modes: list[str],
    out: Path,
) -> None:
    src_order = [s for s in ("hest", "st1k", "spatialcorpus") if s in set(sources)]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.8 * len(modes), 4), constrained_layout=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        x = matrices[(vocab_name, mode)]
        vals = [x[sources == s].sum(axis=1) for s in src_order]
        ax.boxplot(vals, tick_labels=src_order, showfliers=False)
        ax.set_title(mode)
        ax.set_ylabel("spot sum after normalization")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Source-wise spot-sum shift after normalization | vocab={vocab_name}")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)



def _fig_rank_shift(rdf: pd.DataFrame, vocab_name: str, out: Path) -> None:
    sub = rdf[rdf["vocab"] == vocab_name].copy()
    order = sub["normalization"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        ("rank_spearman_mean", "Rank Spearman vs none", "higher = rank preserved"),
        ("top50_overlap_mean", "Top-50 overlap vs none", "higher = top genes preserved"),
        ("mean_abs_rank_shift_norm", "Mean absolute rank shift", "lower = less reorder"),
    ]
    colors = ["#4c78a8", "#59a14f", "#e15759"]
    for ax, (col, title, subtitle), color in zip(axes, panels, colors):
        vals = sub[col].to_numpy(dtype=float)
        ax.bar(order, vals, color=color, alpha=0.86)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        if "spearman" in col or "overlap" in col:
            ax.set_ylim(0, 1.05)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Within-spot gene-rank changes after normalization | vocab={vocab_name}")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _fig_rank_shift_scatter(
    matrices: dict[tuple[str, str], np.ndarray],
    vocab_name: str,
    mode: str,
    out: Path,
    *,
    n_spots: int = 4,
    seed: int = 0,
) -> None:
    """Qualitative examples: raw proxy rank vs normalized rank for spots."""
    raw = matrices[(vocab_name, "none")]
    x = matrices[(vocab_name, mode)]
    rng = np.random.default_rng(seed)
    nnz = (raw > 0).sum(axis=1)
    candidates = np.where(nnz >= 50)[0]
    if candidates.size == 0:
        return
    chosen = rng.choice(candidates, min(n_spots, candidates.size), replace=False)
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4), constrained_layout=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, i in zip(axes, chosen):
        mask = raw[i] > 0
        rr0 = _rankdata_ordinal(raw[i, mask])
        rr1 = _rankdata_ordinal(x[i, mask])
        ax.scatter(rr0, rr1, s=5, alpha=0.22, color="#4c78a8")
        lim = max(float(rr0.max()), float(rr1.max()))
        ax.plot([0, lim], [0, lim], color="black", lw=1, alpha=0.5)
        rho = _spearman_rank_vectors(rr0, rr1)
        ax.set_title(f"spot {int(i)} | nnz={int(mask.sum())}\nrho={rho:.3f}")
        ax.set_xlabel("rank in none/hvg_log")
        ax.set_ylabel(f"rank in {mode}")
    fig.suptitle(f"Rank scatter examples | vocab={vocab_name} | mode={mode}")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

def _write_report(vdf: pd.DataFrame, ndf: pd.DataFrame, rdf: pd.DataFrame, fig_dir: Path, eval_dir: Path, prepared: Path) -> None:
    md = [
        "# Vocab / Normalization QC",
        "",
        f"prepared_dir: `{prepared}`",
        "",
        "## Tables",
        "",
        f"- Vocab metrics: `{eval_dir / 'vocab_quality.csv'}`",
        f"- Normalization metrics: `{eval_dir / 'normalization_quality.csv'}`",
        f"- Normalization rank-shift metrics: `{eval_dir / 'normalization_rank_shift.csv'}`",
        "",
        "## Vocab Summary",
        "",
        vdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![](fig_vocab_quality.png)",
        "",
        "![](fig_vocab_prevalence_dispersion.png)",
        "",
        "## Normalization Summary",
        "",
        ndf.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![](fig_norm_distributions_vocab4096.png)",
        "",
        "![](fig_norm_metrics_vocab4096.png)",
        "",
        "![](fig_norm_source_shift_vocab4096.png)",
        "",
        "## Rank Shift Summary",
        "",
        "These metrics compare within-spot gene rank after normalization against `none/hvg_log`.",
        "The prepared shards do not store raw integer counts, but `hvg_log` is a monotonic",
        "transform of spot-normalized abundance, so ranks are valid as a raw-abundance proxy.",
        "",
        rdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        "![](fig_norm_rank_shift_vocab4096.png)",
        "",
        "![](fig_norm_rank_scatter_global_median_vocab4096.png)",
        "",
        "## Interpretation Guide",
        "",
        "- `marker_panel_coverage`: curated biology markers retained by the vocab.",
        "- `sample_prev_median` / `spot_prev_median`: whether vocab is broad but not too sparse.",
        "- `norm_dispersion_median`: whether clipped genes are information-rich.",
        "- `spot_sum_cv`: lower means less spot-depth variation remains.",
        "- `gene_mean_cv`: lower means gene-level magnitude imbalance is reduced.",
        "- `clip_abs8_pct`: high values mean normalization is producing extreme values.",
        "- `rank_spearman_mean`: within-spot gene rank preservation vs none/hvg_log.",
        "- `top50_overlap_mean`: how many top-50 genes remain top-50 after normalization.",
        "- `mean_abs_rank_shift_norm`: average rank movement normalized by nonzero gene count.",
    ]
    (fig_dir / "vocab_norm_qc_report.md").write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--vocab", nargs="+", default=["4096", "8192", "full"],
                    help="Vocab options: 4096 8192 full. Uses clip<N>_keep_indices.npy.")
    ap.add_argument("--norm", nargs="+", default=["none", "global_median", "nonzero_z"],
                    help="Normalization modes to compare.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-shards", type=int, default=24)
    ap.add_argument("--max-spots-per-shard", type=int, default=256)
    ap.add_argument("--max-rank-spots", type=int, default=1200,
                    help="Max sampled spots for within-spot rank-shift analysis.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--fig-dir", default="results/figures/vocab_norm_qc")
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    eval_dir = Path(args.eval_dir)
    fig_dir = Path(args.fig_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"prepared={prepared}")
    log.info(f"vocab options={args.vocab}  norm modes={args.norm}")

    vdf = vocab_quality_table(prepared, args.vocab)
    vdf.to_csv(eval_dir / "vocab_quality.csv", index=False)

    shards = _sample_shards(prepared, args.split, args.max_shards, args.seed)
    log.info(f"sampling {len(shards)} shards × <= {args.max_spots_per_shard} spots")
    base_x, sources = _load_sample_matrix(shards, args.max_spots_per_shard, args.seed)
    ndf, matrices = normalization_quality_table(prepared, base_x, args.vocab, args.norm)
    ndf.to_csv(eval_dir / "normalization_quality.csv", index=False)
    rdf = normalization_rank_shift_table(
        matrices, args.vocab, args.norm,
        max_spots=args.max_rank_spots,
        seed=args.seed,
    )
    rdf.to_csv(eval_dir / "normalization_rank_shift.csv", index=False)

    _fig_vocab_quality(vdf, fig_dir / "fig_vocab_quality.png")
    _fig_vocab_scatter(prepared, args.vocab, fig_dir / "fig_vocab_prevalence_dispersion.png")
    preferred = "4096" if "4096" in args.vocab else args.vocab[0]
    _fig_norm_distributions(matrices, preferred, args.norm, fig_dir / f"fig_norm_distributions_vocab{preferred}.png")
    _fig_norm_metrics(ndf, preferred, fig_dir / f"fig_norm_metrics_vocab{preferred}.png")
    _fig_norm_source_shift(matrices, sources, preferred, args.norm, fig_dir / f"fig_norm_source_shift_vocab{preferred}.png")
    _fig_rank_shift(rdf, preferred, fig_dir / f"fig_norm_rank_shift_vocab{preferred}.png")
    if "global_median" in args.norm:
        _fig_rank_shift_scatter(
            matrices, preferred, "global_median",
            fig_dir / f"fig_norm_rank_scatter_global_median_vocab{preferred}.png",
            seed=args.seed,
        )
    if "nonzero_z" in args.norm:
        _fig_rank_shift_scatter(
            matrices, preferred, "nonzero_z",
            fig_dir / f"fig_norm_rank_scatter_nonzero_z_vocab{preferred}.png",
            seed=args.seed + 13,
        )
    _write_report(vdf, ndf, rdf, fig_dir, eval_dir, prepared)

    log.info(f"wrote {eval_dir / 'vocab_quality.csv'}")
    log.info(f"wrote {eval_dir / 'normalization_quality.csv'}")
    log.info(f"wrote {eval_dir / 'normalization_rank_shift.csv'}")
    log.info(f"figures/report under {fig_dir}")


if __name__ == "__main__":
    main()
