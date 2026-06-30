"""Render all per-run report figures from the artefacts a training run produces.

Looks for, inside `results/runs/<tag>/`:
  - history.json           (step-level train log)        → train loss curve
  - val_history.json       (epoch-level val log)         → val loss / val component curves
  - zero_shot_test.json    (eval_zero_shot output)       → retrieval R@K bars, silhouette bars
  - linear_probe_test.json (eval_linearprobe output)     → per-arm PCC / Spearman / R² bars
  - linear_probe_*_per_gene.json                          → top-N variable-gene PCC bar
  - ckpt_best.pt / ckpt_last.pt + test split             → retrieval examples grid

Run after training/eval (the sweep runner calls this automatically).

Usage:
    python scripts/viz/figures.py --run results/runs/<tag>
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ---------------------------------------------------------------------------
# Loss / val curves
# ---------------------------------------------------------------------------

def fig_loss_curves(run_dir: Path) -> None:
    train_path = run_dir / "history.json"
    val_path = run_dir / "val_history.json"
    if not train_path.exists():
        return
    train = json.loads(train_path.read_text())
    val = json.loads(val_path.read_text()) if val_path.exists() else []

    # Pick the loss components present in the train log
    comp_keys = sorted({k for h in train for k in h
                        if k.startswith(("loss/", "align/", "recon/", "metric/"))})
    # Choose the most informative subset for the headline figure
    headline = [k for k in comp_keys if k in (
        "loss/total", "align/loss", "recon/gene", "recon/image",
        "metric/cosine_sim", "metric/gene_tx_pcc", "metric/gene_img_pcc",
    )] or comp_keys[:6]

    n = len(headline)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.0 * nrow), squeeze=False)
    steps = [h["step"] for h in train]
    val_epochs = [v.get("epoch") for v in val]
    for i, k in enumerate(headline):
        ax = axes[i // ncol, i % ncol]
        ys = [h.get(k, np.nan) for h in train]
        ax.plot(steps, ys, label=f"train {k.split('/', 1)[-1]}", color="C0")
        if val and f"val/{k}" in val[0]:
            vy = [v.get(f"val/{k}", np.nan) for v in val]
            # Place val points at the step *boundary* of each epoch
            steps_per_epoch = len(train) // max(1, train[-1]["epoch"] - train[0]["epoch"] + 1)
            vx = [ep * steps_per_epoch * 20 for ep in val_epochs]   # approx (log_every≈20)
            ax.plot(vx, vy, marker="o", linestyle="--", color="C3",
                    label=f"val {k.split('/', 1)[-1]}")
        ax.set_title(k); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # turn off empty axes
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "fig_train_curves.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Linear probe bar charts (PCC / Spearman / R² across arms)
# ---------------------------------------------------------------------------

def fig_linear_probe(run_dir: Path) -> None:
    for split in ("test", "val"):
        p = run_dir / f"linear_probe_{split}.json"
        if not p.exists():
            continue
        results = json.loads(p.read_text())  # {arm: {pcc/mean, ...}}
        if not results:
            continue
        arms = list(results.keys())
        metrics = [("pcc/mean", "pcc/std", "Pearson r (all HVG)"),
                   ("spearman/mean", "spearman/std", "Spearman ρ (all HVG)"),
                   ("r2/mean", "r2/std", "R² (all HVG)"),
                   ("l2/mean", "l2/std", "L2 error (lower=better)")]
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        colors = ["#3b6ea5", "#e89c1d", "#9c27b0", "#1e9d4d", "#cc2424"]
        for ax, (m, e, name) in zip(axes, metrics):
            means = [results[a].get(m, np.nan) for a in arms]
            stds  = [results[a].get(e, 0.0)   for a in arms]
            cs = [colors[i % len(colors)] for i in range(len(arms))]
            ax.bar(arms, means, yerr=stds, capsize=4, color=cs)
            ax.set_title(name); ax.tick_params(axis="x", rotation=20)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle(f"Linear probe — {split} split (5-fold Ridge + PCA; "
                     f"mean over all HVG)", fontsize=12)
        fig.tight_layout()
        fig.savefig(run_dir / f"fig_linear_probe_{split}.png", dpi=130)
        plt.close(fig)

        # NEW: compact bar of {Pearson all-HVG / Pearson top-50 / Spearman all-HVG}
        # so the difference between the full-distribution mean and the top genes
        # is immediately visible (the headline numbers we report).
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            x = np.arange(len(arms))
            w = 0.27
            for i, (key, label, color) in enumerate([
                ("pcc/mean", "PCC (all HVG)", "#3b6ea5"),
                ("spearman/mean", "Spearman (all HVG)", "#1e9d4d"),
            ]):
                vals = [results[a].get(key, np.nan) for a in arms]
                ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=color)
            # top-50 PCC from per_gene files when available
            top50_vals = []
            for a in arms:
                pg = run_dir / f"linear_probe_{a}_per_gene.json"
                v = np.nan
                if pg.exists():
                    d = json.loads(pg.read_text())
                    pcc_dict = d.get("per_gene_pcc_mean") or d.get("per_gene_pcc") or {}
                    if pcc_dict:
                        sorted_vals = sorted(pcc_dict.values(), reverse=True)[:50]
                        v = float(np.mean(sorted_vals)) if sorted_vals else np.nan
                top50_vals.append(v)
            ax.bar(x + 1.5 * w, top50_vals, w, label="PCC (top-50 HVG)", color="#cc2424")
            ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
            ax.set_ylabel("score"); ax.set_title(f"Linear probe overview — {split}")
            ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(run_dir / f"fig_linear_probe_overview_{split}.png", dpi=130)
            plt.close(fig)
        except Exception:
            pass

        # Per-arm top-50 most-variable HVG PCC
        for arm in arms:
            pg_path = run_dir / f"linear_probe_{arm}_per_gene.json"
            if not pg_path.exists():
                continue
            pg = json.loads(pg_path.read_text())
            pcc = pg.get("per_gene_pcc_mean") or pg.get("per_gene_pcc")
            if not pcc:
                continue
            items = sorted(pcc.items(), key=lambda x: x[1], reverse=True)[:50]
            names = [k for k, _ in items]; vals = [v for _, v in items]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(range(len(items)), vals, color="#3b6ea5")
            ax.set_xticks(range(len(items)))
            ax.set_xticklabels(names, rotation=80, fontsize=7)
            ax.set_ylabel("Pearson r")
            ax.set_title(f"Top-50 genes by probe PCC — {arm} ({split})")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(run_dir / f"fig_linear_probe_top50_{arm}_{split}.png", dpi=130)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Zero-shot bar charts (retrieval R@K, silhouette, RankMe, modality gap)
# ---------------------------------------------------------------------------

def fig_zero_shot(run_dir: Path) -> None:
    for split in ("test", "val"):
        p = run_dir / f"zero_shot_{split}.json"
        if not p.exists():
            continue
        zs = json.loads(p.read_text())
        # Collect arms
        arms = sorted({k.split("/", 1)[0] for k in zs if "/" in k})
        if not arms:
            continue

        def _ser(suffix: str) -> dict[str, float]:
            return {a: zs.get(f"{a}/{suffix}", np.nan) for a in arms}

        panels = [
            ("Retrieval i2t R@10 (global)", "retr/global/i2t_R@10"),
            ("Retrieval i2t R@10 (per_organ)", "retr/per_organ/i2t_R@10"),
            ("Retrieval i2t R@10 (per_sample)", "retr/per_sample/i2t_R@10"),
            ("Clustering silhouette", "cluster/silhouette"),
            ("RankMe (image side)", "rankme/image"),
            ("Modality gap ‖μ_i−μ_t‖", "modality_gap"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (name, key) in zip(axes.flatten(), panels):
            vals = [zs.get(f"{a}/{key}", np.nan) for a in arms]
            ax.bar(arms, vals)
            ax.set_title(name); ax.tick_params(axis="x", rotation=20)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle(f"Zero-shot evaluation — {split} split  "
                     f"(global / per_organ / per_sample retrieval + clustering)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(run_dir / f"fig_zero_shot_{split}.png", dpi=130)
        plt.close(fig)

        # 3-granularity retrieval table figure (R@1/5/10/50 × granularity).
        try:
            grans = ["global", "per_organ", "per_sample"]
            ks = [1, 5, 10, 50]
            fig, axes = plt.subplots(1, len(grans), figsize=(5.0 * len(grans), 4))
            for ax, g in zip(axes, grans):
                x = np.arange(len(arms))
                w = 0.18
                for i, k in enumerate(ks):
                    vals = [zs.get(f"{a}/retr/{g}/i2t_R@{k}", np.nan) for a in arms]
                    ax.bar(x + (i - 1.5) * w, vals, w, label=f"R@{k}")
                ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
                ax.set_title(f"Retrieval (image→tx) — {g}")
                ax.set_ylabel("R@K"); ax.grid(alpha=0.3, axis="y")
                ax.legend(fontsize=7)
            fig.suptitle(f"Retrieval @ 3 granularities — {split} split", fontsize=12)
            fig.tight_layout()
            fig.savefig(run_dir / f"fig_retrieval_granularity_{split}.png", dpi=130)
            plt.close(fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Retrieval qualitative grid (queries + top-K with hit/miss marks)
# ---------------------------------------------------------------------------

def fig_retrieval_examples(run_dir: Path,
                           n_queries: int = 6, top_k: int = 5) -> None:
    """Run a small inference pool on the test split and render the grid."""
    from mm_align.utils import load_config
    from mm_align.data import build_dataset_from_split, pad_collate
    from mm_align.models import MMAligner
    from mm_align.evaluation import encode_loader, render_retrieval_examples
    import torch
    from torch.utils.data import DataLoader

    ckpt = run_dir / "ckpt_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "ckpt_last.pt"
    if not ckpt.exists():
        return

    state = __import__("torch").load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = state.get("cfg") if isinstance(state, dict) else None
    if cfg is None:
        return

    # Force image_mode=raw so we have the patch pixels to draw.
    cfg["data"]["image_mode"] = "raw"
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")

    prepared = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared / "splits.json").read_text())
    test_ids = splits.get("test", [])[:6]            # keep the pool small for speed
    if not test_ids:
        return
    ds = build_dataset_from_split(prepared, test_ids,
                                  k_spatial=cfg["data"]["k_spatial"],
                                  load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
                                  image_mode="raw",
                                  hest_patch_dir=cfg["data"]["hest_patch_dir"])
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=2, collate_fn=pad_collate)

    model = MMAligner(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    pool = encode_loader(model, loader, device, desc="retrieval pool")
    # Subsample for grid speed (cap at 2000 spots).
    cap = min(2000, pool["z_image"].shape[0])
    if pool["z_image"].shape[0] > cap:
        idx = np.random.default_rng(0).choice(pool["z_image"].shape[0], cap, replace=False)
        for k in list(pool.keys()):
            if pool[k] is None: continue
            try: pool[k] = pool[k][idx]
            except Exception: pass

    sample_ids_local = [s.sample_id for s in ds.shards]
    spot_idx = pool.get("spot_idx")
    if spot_idx is None:
        spot_idx = np.zeros(pool["z_image"].shape[0], dtype=np.int64)
    render_retrieval_examples(
        pool["z_image"], pool["z_tx"],
        sample_idx=pool["sample_idx"], spot_idx=spot_idx,
        sample_ids=sample_ids_local,
        prepared_dir=prepared,
        hest_patch_dir=Path(cfg["data"]["hest_patch_dir"]),
        out_path=run_dir / "fig_retrieval_examples.png",
        n_queries=n_queries, top_k=top_k,
        title=f"{run_dir.name} — image→tx retrieval (✓=true positive)",
    )


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to a results/runs/<tag>/ directory.")
    ap.add_argument("--skip-retrieval-grid", action="store_true",
                    help="Skip the qualitative retrieval examples (saves a few minutes).")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    print(f"[figures] {run_dir}")
    fig_loss_curves(run_dir)
    fig_linear_probe(run_dir)
    fig_zero_shot(run_dir)
    if not args.skip_retrieval_grid:
        try:
            fig_retrieval_examples(run_dir)
        except Exception as e:
            print(f"[figures] retrieval grid failed: {e}")

    pngs = sorted(run_dir.glob("fig_*.png"))
    print(f"[figures] wrote {len(pngs)} figures:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
