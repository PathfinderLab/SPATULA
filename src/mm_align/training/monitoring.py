"""Lightweight monitoring helpers for the training loop.

These are *not* the per-stage scientific monitors (gene_set monitor, stage1
benchmarks live in `mm_align.evaluation`); they are the small UI helpers
that the entrypoint needs: log line summariser + matplotlib loss curve.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_loss_curve(history: list[dict], path: Path, tag: str) -> None:
    """Dump the per-step training-loss curve as a single PNG."""
    if not history:
        return
    steps = [h["step"] for h in history]
    losses = [h.get("loss/total", h.get("loss", 0.0)) for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, label="train loss")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.title(tag); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def _series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in history:
        if key not in row:
            continue
        try:
            y = float(row[key])
            x = float(row.get("epoch", row.get("step", len(xs))))
        except Exception:
            continue
        if y == y:  # drop NaN
            xs.append(x); ys.append(y)
    return xs, ys


def _plot_group(ax, history: list[dict], keys: list[str], title: str) -> bool:
    plotted = False
    for key in keys:
        xs, ys = _series(history, key)
        if not xs:
            continue
        label = key.replace("val/", "")
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3, label=label)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=7)
    return plotted


def render_val_metric_curves(history: list[dict], path: Path, tag: str) -> None:
    """Render compact multi-panel validation curves from val_history.json rows."""
    if not history:
        return
    groups = [
        ("Loss", ["val/loss", "val/tx_self/loss", "val/clean_msm/loss"]),
        ("MSM Top-k", [
            "val/tx_self/masked_symbol_top1_acc",
            "val/tx_self/masked_symbol_top5_acc",
            "val/tx_self/masked_symbol_top10_acc",
            "val/clean_msm/tx_self/masked_symbol_top10_acc",
        ]),
        ("MSM CE", [
            "val/tx_self/masked_symbol_ce_norm",
            "val/clean_msm/tx_self/masked_symbol_ce_norm",
            "val/tx_self/masked_symbol_ce_gain",
        ]),
        ("Linear Probes", [
            "val/linear_probe/hvg/spearman_mean",
            "val/linear_probe/masked_hvg/spearman_mean",
            "val/linear_probe/hvg/pearson_mean",
            "val/linear_probe/masked_hvg/pearson_mean",
        ]),
        ("Rank / Manifold", [
            "val/linear_probe/hvg_rank/spot_rank_spearman",
            "val/linear_probe/hvg_rank/top10_overlap",
            "val/intrinsic/expression/distance_spearman",
            "val/intrinsic/expression/knn_overlap@20",
        ]),
        ("Embedding Health", [
            "val/intrinsic/effective_rank",
            "val/intrinsic/explained_top10",
            "val/intrinsic/gene_embedding/corr_spearman",
            "val/intrinsic/gene_embedding/top_pair_overlap",
        ]),
        ("Auxiliary", [
            "val/tx_self/view_jepa_weight_effective",
            "val/tx_self/view_jepa",
            "val/tx_self/view_jepa_cosine_distance",
            "val/tx_self/dino_weight_effective",
            "val/tx_self/dino_cosine_distance",
            "val/tx_self/koleo_weight_effective",
            "val/tx_self/koleo",
        ]),
        ("Sequence", [
            "val/tx_self/seq_len_mean",
            "val/tx_self/seq_len_median",
            "val/tx_self/mask_actual_ratio",
            "val/tx_self/n_masked_mean",
        ]),
    ]
    ncols = 2
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 3.2 * nrows), squeeze=False)
    used = 0
    for ax, (title, keys) in zip(axes.ravel(), groups):
        if _plot_group(ax, history, keys, title):
            used += 1
        else:
            ax.set_axis_off()
    for ax in axes.ravel()[len(groups):]:
        ax.set_axis_off()
    fig.suptitle(f"{tag} validation metrics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_stage1_metric_curves(history: list[dict], path: Path, tag: str) -> None:
    """Focused Stage-1 dashboard: primary selectors only."""
    if not history:
        return
    groups = [
        ("MSM", ["val/tx_self/masked_symbol_top10_acc", "val/clean_msm/tx_self/masked_symbol_top10_acc", "val/tx_self/masked_symbol_ce_norm"]),
        ("Probe", ["val/linear_probe/hvg/spearman_mean", "val/linear_probe/masked_hvg/spearman_mean", "val/linear_probe/hvg_rank/spot_rank_spearman"]),
        ("Intrinsic", ["val/intrinsic/expression/distance_spearman", "val/intrinsic/expression/knn_overlap@20", "val/intrinsic/gene_embedding/corr_spearman"]),
        ("Health/Aux", ["val/intrinsic/effective_rank", "val/intrinsic/explained_top10", "val/tx_self/view_jepa_weight_effective", "val/tx_self/dino_weight_effective"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
    for ax, (title, keys) in zip(axes.ravel(), groups):
        _plot_group(ax, history, keys, title)
    fig.suptitle(f"{tag} Stage-1 validation dashboard", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# Keys we surface in the per-epoch one-line "[VAL]" summary.  Order matters —
# loss/total first so the most important number shows up before the others.
_SUMMARY_KEYS = (
    "loss/total", "align/loss", "recon/gene", "recon/image",
    "metric/cosine_sim", "metric/gene_tx_pcc", "metric/gene_img_pcc",
)


def summarize_log(d: dict) -> str:
    """One-line `k=v k=v ...` string for the per-epoch print."""
    out = []
    for k in _SUMMARY_KEYS:
        if k not in d:
            continue
        try:
            out.append(f"{k.split('/', 1)[-1]}={d[k]:.3f}")
        except Exception:
            pass
    return " ".join(out)
