"""Build curated report figures for the current mm_align project state.

This script does not try to be a full EDA suite.  It gathers the existing EDA
artifacts and renders a small report-ready set under results/figures/report/:

  - fig_project_overview.png
  - fig_dataset_overview.png
  - fig_vocab_overview.png
  - fig_sequence_qc.png
  - fig_ablation_summary.png (copied from results/figures/ablation1.png if present)
  - figure_manifest.md

Use this for reports/slides; keep scripts/data/dataset_eda.py for deep audits.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


PALETTE = {
    "stage1": "#d9ecff",
    "stage15": "#ddf3df",
    "stage2": "#ffe4c7",
    "eval": "#f2dcff",
    "data": "#eeeeee",
    "edge": "#404040",
    "accent": "#1f5fbf",
}
SRC_COLORS = {"hest": "#2f6fdd", "st1k": "#df7a2e", "spatialcorpus": "#6f50c8"}


def _clean_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".png", ".md"}:
            p.unlink()


def _box(ax, xy, wh, text, color, *, fs=10, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color,
        edgecolor=PALETTE["edge"],
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return (x, y, w, h)


def _arrow(ax, src, dst):
    ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="->", mutation_scale=14, linewidth=1.4, color=PALETTE["edge"]))


def fig_project_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    ax.text(6.75, 6.35, "mm_align project overview", ha="center", fontsize=18, weight="bold")
    ax.text(6.75, 6.02, "RNA foundation -> spatial context -> pathology-transcriptomics alignment", ha="center", fontsize=10, color="#555")

    data = _box(ax, (0.4, 3.9), (2.4, 1.1), "Prepared ST shards\nHVG values + coords\nUNI image features", PALETTE["data"], fs=9, weight="bold")
    s1 = _box(ax, (3.4, 3.65), (2.7, 1.6), "Stage 1\nSpot / RNA encoder\n\nMSM primary\nView-JEPA optional\nGlobal-median + clip4096", PALETTE["stage1"], fs=9, weight="bold")
    s15 = _box(ax, (6.8, 3.65), (2.7, 1.6), "Stage 1.5\nSpatial encoder\n\nspot + neighbors\nSpatial-JEPA\nHyperST-style context", PALETTE["stage15"], fs=9, weight="bold")
    s2 = _box(ax, (10.2, 3.65), (2.7, 1.6), "Stage 2\nImage-RNA alignment\n\nretrieval\nimage-to-expression\nslide-level MIL", PALETTE["stage2"], fs=9, weight="bold")

    _arrow(ax, (2.8, 4.45), (3.4, 4.45))
    _arrow(ax, (6.1, 4.45), (6.8, 4.45))
    _arrow(ax, (9.5, 4.45), (10.2, 4.45))

    _box(ax, (3.35, 1.0), (2.8, 1.65), "Stage 1 eval\nMSM top-k / CE\nHVG & masked-HVG probe\nrank probe\ngene embedding corr", PALETTE["eval"], fs=8.7)
    _box(ax, (6.75, 1.0), (2.8, 1.65), "Stage 1.5 eval\nspatial kNN overlap\nsmoothness\ngene map SCC\nqualitative spatial maps", PALETTE["eval"], fs=8.7)
    _box(ax, (10.15, 1.0), (2.8, 1.65), "Stage 2 eval\ncross-modal retrieval\nimage -> HVG probe\nHEST/SEAL/PathBench\nMIL downstream", PALETTE["eval"], fs=8.7)

    ax.text(0.45, 0.4, "Working hypothesis: strong relative RNA representation first, then spatial context, then multimodal alignment.", fontsize=10, color="#333")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _read_dataset_summary(eda_dir: Path) -> dict:
    p = eda_dir / "eda_v2" / "dataset_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def fig_dataset_overview(eda_dir: Path, out: Path) -> None:
    summary = _read_dataset_summary(eda_dir)
    qc_path = eda_dir / "process_qc" / "per_sample_quality.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()

    sources = sorted(summary.keys()) if summary else sorted(qc["source"].dropna().unique())
    samples = []
    spots = []
    for src in sources:
        if summary:
            samples.append(summary[src].get("n_samples", 0))
            spots.append(summary[src].get("total_spots", 0))
        else:
            q = qc[qc["source"] == src]
            samples.append(len(q))
            spots.append(float(q.get("n_spots_total", pd.Series(dtype=float)).sum()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = [SRC_COLORS.get(s, "#777") for s in sources]
    axes[0].bar(sources, samples, color=colors)
    axes[0].set_title("Samples by source")
    axes[0].set_ylabel("# samples")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(sources, np.array(spots) / 1e6, color=colors)
    axes[1].set_title("Spots by source")
    axes[1].set_ylabel("million spots")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.25, axis="y")

    if not qc.empty and "seq_len_median" in qc:
        vals = [qc.loc[qc["source"] == s, "seq_len_median"].dropna().values for s in sources]
        axes[2].boxplot(vals, tick_labels=sources, patch_artist=True)
        for patch, c in zip(axes[2].artists, colors):
            patch.set_facecolor(c)
        axes[2].set_title("Median sequence length by source")
        axes[2].set_ylabel("non-zero genes / spot")
        axes[2].tick_params(axis="x", rotation=20)
        axes[2].grid(alpha=0.25, axis="y")
    else:
        axes[2].axis("off")
    fig.suptitle("Dataset composition and sequence-length sanity", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_vocab_overview(eda_dir: Path, out: Path) -> None:
    vh_path = eda_dir / "eda_v2" / "vocab_health.json"
    health = json.loads(vh_path.read_text()) if vh_path.exists() else {}
    total = int(health.get("n_hvg", 0))
    rare = int(health.get("n_rare    (prevalence < 10%)", 0))
    common = int(health.get("n_common  (prevalence >= 50%)", health.get("n_common  (prevalence \u2265 50%)", 0)))
    singleton = int(health.get("n_singleton (coverage <= 1 sample)", health.get("n_singleton (coverage \u2264 1 sample)", 0)))
    common_examples = health.get("common_examples", [])[:10]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    labels = ["total vocab", "common\n>=50% samples", "rare\n<10% samples", "singleton"]
    vals = [total, common, rare, singleton]
    axes[0].bar(labels, vals, color=["#555", "#2f8f46", "#c9862b", "#b63b3b"])
    axes[0].set_title("HVG vocabulary health")
    axes[0].set_ylabel("# genes")
    axes[0].grid(alpha=0.25, axis="y")
    for i, v in enumerate(vals):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    if common_examples:
        genes = [g for g, _ in common_examples][::-1]
        counts = [c for _, c in common_examples][::-1]
        axes[1].barh(genes, counts, color="#2f6fdd")
        axes[1].set_title("Most broadly expressed vocab genes")
        axes[1].set_xlabel("# samples with non-zero expression")
        axes[1].grid(alpha=0.25, axis="x")
    else:
        axes[1].axis("off")
    fig.suptitle("Vocab design: dense enough for representation, clipped enough for speed", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_sequence_qc(eda_dir: Path, out: Path) -> None:
    qc_path = eda_dir / "process_qc" / "per_sample_quality.csv"
    if not qc_path.exists():
        return
    qc = pd.read_csv(qc_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sources = sorted(qc["source"].dropna().unique())
    colors = [SRC_COLORS.get(s, "#777") for s in sources]
    vals = [qc.loc[qc["source"] == s, "vocab_hit"].dropna().values for s in sources]
    axes[0].boxplot(vals, tick_labels=sources, patch_artist=True)
    for patch, c in zip(axes[0].artists, colors):
        patch.set_facecolor(c)
    axes[0].set_title("Vocab hit rate by source")
    axes[0].set_ylabel("clean genes covered by vocab")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.25, axis="y")

    for src, c in zip(sources, colors):
        sub = qc[qc["source"] == src]
        axes[1].scatter(sub["seq_len_median"], sub["vocab_hit"], s=16, alpha=0.6, label=src, color=c)
    axes[1].set_xlabel("median non-zero genes / spot")
    axes[1].set_ylabel("vocab hit rate")
    axes[1].set_title("Sequence length vs vocab coverage")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Input sequence quality checks", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def copy_selected(src: Path, dst: Path) -> bool:
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda-dir", default="results/eda/prepared_expanded")
    ap.add_argument("--out-dir", default="results/figures/report")
    args = ap.parse_args()
    eda_dir = Path(args.eda_dir)
    out_dir = Path(args.out_dir)
    _clean_out(out_dir)

    outputs = []
    fig_project_overview(out_dir / "fig_project_overview.png")
    outputs.append(("fig_project_overview.png", "Stage-level project overview and evaluation map."))
    fig_dataset_overview(eda_dir, out_dir / "fig_dataset_overview.png")
    outputs.append(("fig_dataset_overview.png", "Dataset composition and sequence-length sanity."))
    fig_vocab_overview(eda_dir, out_dir / "fig_vocab_overview.png")
    outputs.append(("fig_vocab_overview.png", "HVG vocab health and common-gene coverage."))
    fig_sequence_qc(eda_dir, out_dir / "fig_sequence_qc.png")
    outputs.append(("fig_sequence_qc.png", "Vocab hit rate and non-zero sequence length QC."))

    copies = [
        (Path("results/figures/ablation1.png"), "fig_ablation_summary.png", "Stage1 ablation summary across normalization/vocab/value/objective groups."),
        (eda_dir / "eda_v2" / "fig_norm_before_after.png", "fig_norm_before_after.png", "Raw log1p versus global-median normalized value distribution."),
        (eda_dir / "vocab_qc" / "fig_filter_quality.png", "fig_vocab_filter_quality.png", "Vocab filtering quality checks."),
    ]
    for src, name, desc in copies:
        if copy_selected(src, out_dir / name):
            outputs.append((name, desc))

    manifest = ["# Curated Report Figures", "", f"Source EDA: `{eda_dir}`", "", "| file | intended use |", "|---|---|"]
    for name, desc in outputs:
        manifest.append(f"| `{name}` | {desc} |")
    manifest.append("")
    manifest.append("Note: deep EDA artifacts remain under `results/eda/`; this directory is the report-ready subset.")
    (out_dir / "figure_manifest.md").write_text("\n".join(manifest))
    print(f"Wrote {len(outputs)} report figures to {out_dir}")


if __name__ == "__main__":
    main()
