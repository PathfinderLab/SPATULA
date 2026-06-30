"""Masked-Symbol-Modeling (MSM) downstream evaluation on the test split.

We re-use the encoder's INTERNAL masking pipeline (the same one used at train
time): with `_force_mask_in_eval = True`, the encoder samples a `mask_ratio`
fraction of real (non-padding) gene tokens, replaces their symbols with
[MASK], runs the transformer, and returns the contextual per-token embedding
plus the original gene-token IDs.  We then probe the trained `symbol_head` on
the masked positions and compare predicted vs. ground-truth token IDs.

Reports
-------
1. Overall accuracies:
     - top_1 / top_5 / top_10 / top_20
     - macro-averaged per gene (so abundant genes don't dominate)
     - cross-entropy NLL (lower = better)
2. Per-gene table (gene name, n_observations, top1/top5/top10 acc, NLL).
   Sorted lists of "best-predicted" and "worst-predicted" genes are written.
3. Per-organ and per-source breakdowns.
4. Figures:
   - top-K accuracy bar plot (overall + macro)
   - per-gene top-1 accuracy histogram
   - top-K best/worst genes bar plot
   - per-organ accuracy bar plot
   - per-source accuracy bar plot
   - confidence histogram (softmax max prob on the GT token)

Pure scanpy/sklearn — no squidpy.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.utils import get_logger

log = get_logger("msm_eval")

# Re-use stage1_tx ckpt loader + eval pool builder.
_spec = importlib.util.spec_from_file_location(
    "_msm_helper", Path(__file__).resolve().parent / "stage1_tx.py",
)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
load_tx_encoder = _helper.load_tx_encoder
build_eval_pool = _helper.build_eval_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_hvg(hvg: np.ndarray, vocab_keep, gene_norm_cfg, full_hvg_dim) -> np.ndarray:
    eff = int(len(vocab_keep)) if vocab_keep is not None else hvg.shape[1]
    if vocab_keep is not None:
        hvg = hvg[:, vocab_keep]
    if gene_norm_cfg:
        normaliser = GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=full_hvg_dim, hvg_dim=eff,
            vocab_keep_indices=vocab_keep,
        )
        hvg = normaliser.apply_np(hvg)
    return hvg


def _hvg_index_from_token(token_id: int, n_special: int) -> int:
    """Inverse of  encoder.hvg_token_ids = arange(N_SPECIAL, N_SPECIAL + n_hvg)."""
    return int(token_id) - int(n_special)


@torch.no_grad()
def _msm_forward(tx_encoder, hvg_norm: np.ndarray, device: str,
                  batch: int = 64, seed: int = 0) -> dict:
    """Run the encoder with internal masking forced ON, gather predictions
    on masked positions across the whole input.

    Returns a dict of CPU numpy arrays:
        gt_token   : (N_masked,)  ground-truth gene token ids
        pred_logits: (N_masked, V) symbol-head logits (float16 to save mem)
        confidence_gt : (N_masked,) softmax prob at GT token
        sample_row : (N_masked,) row index of the spot that produced it
    """
    enc = tx_encoder
    was_training = bool(enc.training)
    old_force = bool(getattr(enc, "_force_mask_in_eval", False))
    enc.eval()
    enc._force_mask_in_eval = True
    gt_list, logit_list, conf_list, row_list = [], [], [], []
    g = torch.Generator(device=device).manual_seed(int(seed))
    try:
        for r0 in range(0, hvg_norm.shape[0], batch):
            xb = torch.from_numpy(hvg_norm[r0:r0 + batch].astype(np.float32)).to(device)
            # Seed each batch deterministically so reruns match.
            torch.manual_seed(seed + r0)
            out = enc(novae_latent=None, hvg=xb)
            per_token = out["per_token"]                      # (B, L_max, D)
            mask = out["mask"]                                # (B, L_max) bool or None
            orig = out["orig_gene_ids"]                       # (B, L_max) long
            if mask is None or not mask.any():
                continue
            pt_m = per_token[mask]                            # (N_m, D)
            logits = enc.symbol_head(pt_m)                    # (N_m, V)
            probs = torch.softmax(logits.float(), dim=-1)
            gt = orig[mask]                                   # (N_m,)
            conf = probs.gather(1, gt.unsqueeze(1)).squeeze(1)
            # Row index = which spot each masked token came from.
            row_idx_grid = (torch.arange(mask.shape[0], device=device)
                             .unsqueeze(1).expand_as(mask))
            row_of_m = row_idx_grid[mask] + r0
            gt_list.append(gt.detach().cpu().numpy())
            logit_list.append(logits.half().detach().cpu().numpy())
            conf_list.append(conf.detach().cpu().numpy())
            row_list.append(row_of_m.detach().cpu().numpy())
    finally:
        enc._force_mask_in_eval = old_force
        if was_training:
            enc.train()
    return {
        "gt_token": np.concatenate(gt_list, axis=0) if gt_list else np.empty((0,), dtype=np.int64),
        "pred_logits": np.concatenate(logit_list, axis=0) if logit_list else np.empty((0, 0), dtype=np.float16),
        "confidence_gt": np.concatenate(conf_list, axis=0) if conf_list else np.empty((0,), dtype=np.float32),
        "sample_row": np.concatenate(row_list, axis=0) if row_list else np.empty((0,), dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _topk_acc(logits: np.ndarray, gt: np.ndarray, k: int) -> float:
    if logits.shape[0] == 0:
        return float("nan")
    # argpartition O(V log k) per row.
    if k >= logits.shape[1]:
        return 1.0
    top = np.argpartition(-logits.astype(np.float32, copy=False), k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(top == gt[:, None], axis=1)))


def _per_gene_table(gt: np.ndarray, logits: np.ndarray,
                     gene_names: list[str], n_special: int,
                     ks: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    rows = []
    # Precompute top-K predictions across all observations to avoid recomputing
    # per gene.
    if logits.shape[0] == 0:
        return pd.DataFrame()
    top_k_max = max(ks)
    top = np.argpartition(-logits.astype(np.float32, copy=False),
                          top_k_max - 1, axis=1)[:, :top_k_max]
    eq = top == gt[:, None]
    for tok in np.unique(gt):
        m = gt == tok
        if not m.any():
            continue
        idx = _hvg_index_from_token(tok, n_special)
        if idx < 0 or idx >= len(gene_names):
            name = f"TOKEN_{tok}"
        else:
            name = gene_names[idx]
        row = {"gene": name, "token_id": int(tok), "n_obs": int(m.sum())}
        for k in ks:
            row[f"top{k}"] = float(eq[m, :k].any(axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_obs", ascending=False).reset_index(drop=True)


def _nll(logits: np.ndarray, gt: np.ndarray) -> float:
    if logits.shape[0] == 0:
        return float("nan")
    # Stable log-softmax over float32 chunk-by-chunk to avoid OOM.
    n = logits.shape[0]
    nll = 0.0
    chunk = 4096
    for r0 in range(0, n, chunk):
        l = logits[r0:r0 + chunk].astype(np.float32)
        l = l - l.max(axis=1, keepdims=True)
        ls = np.log(np.exp(l).sum(axis=1, keepdims=True))
        log_p = l - ls
        gt_b = gt[r0:r0 + chunk]
        nll += float(-log_p[np.arange(len(gt_b)), gt_b].sum())
    return nll / max(1, n)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_topk_bar(topk: dict[int, float], macro: dict[int, float], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(topk.keys())
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(x - 0.2, [topk[k] for k in ks], width=0.4,
            label="micro (per masked position)", color="#4c78a8")
    ax.bar(x + 0.2, [macro[k] for k in ks], width=0.4,
            label="macro (gene-balanced)", color="#e15759")
    # Highlight the K=10 column — primary reporting metric.
    if 10 in topk:
        i10 = ks.index(10)
        ax.axvspan(i10 - 0.45, i10 + 0.45, alpha=0.10, color="#59a14f")
    ax.set_xticks(x); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("MSM accuracy  (top-10 is the primary reporting metric)")
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_per_gene_hist(per_gene: pd.DataFrame, out: Path) -> None:
    """Per-gene top-10 distribution + median.  top-1 also shown for reference."""
    if per_gene.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, col, color in zip(axes, ["top10", "top1"], ["#59a14f", "#4c78a8"]):
        if col not in per_gene.columns:
            ax.set_visible(False); continue
        ax.hist(per_gene[col].values, bins=30, color=color, edgecolor="white")
        med = float(per_gene[col].median())
        ax.axvline(med, color="#e15759", linestyle="--", linewidth=1,
                    label=f"median = {med:.3f}")
        emphasis = " (PRIMARY)" if col == "top10" else ""
        ax.set_xlabel(f"per-gene {col}{emphasis}")
        ax.set_ylabel("# genes")
        ax.legend()
    fig.suptitle("Per-gene MSM accuracy distribution")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_best_worst_bar(per_gene: pd.DataFrame, out: Path, *,
                           top_n: int = 25, min_obs: int = 20,
                           rank_by: str = "top10") -> None:
    """Best / worst genes ranked by `rank_by` (default = top-10, the primary
    reporting metric).  Bars show top-10 accuracy; top-1 printed as small text.
    """
    if per_gene.empty or rank_by not in per_gene.columns:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    pg = per_gene[per_gene["n_obs"] >= min_obs].copy()
    if pg.empty:
        return
    best = pg.nlargest(top_n, rank_by)[::-1]
    worst = pg.nsmallest(top_n, rank_by)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, top_n * 0.25)),
                              constrained_layout=True)
    for ax, df, title, color in [
        (axes[0], best, f"Best {len(best)} genes (top-10)", "#59a14f"),
        (axes[1], worst, f"Worst {len(worst)} genes (top-10)", "#e15759"),
    ]:
        ax.barh(df["gene"], df[rank_by], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"{rank_by} accuracy")
        ax.set_title(title)
        for y, (n, t1) in enumerate(zip(df["n_obs"], df.get("top1", df[rank_by]))):
            ax.text(0.02, y, f"n={n} | top1={t1:.2f}", va="center", fontsize=7)
    fig.suptitle(f"MSM per-gene best vs worst predicted (ranked by {rank_by}, min_obs={min_obs})")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_group_bar(acc_by_group: dict[str, float], counts: dict[str, int],
                     title: str, out: Path,
                     metric_label: str = "top-10 accuracy") -> None:
    if not acc_by_group:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(acc_by_group.items(), key=lambda kv: -kv[1])
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    cnts = [counts.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(keys)), 4),
                            constrained_layout=True)
    bars = ax.bar(range(len(keys)), vals, color="#59a14f")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={cnts[i]}",
                ha="center", fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_confidence_hist(conf: np.ndarray, out: Path) -> None:
    if conf.size == 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(conf, bins=40, color="#4c78a8", edgecolor="white")
    med = float(np.median(conf))
    ax.axvline(med, color="#e15759", linestyle="--", linewidth=1,
                label=f"median p(GT) = {med:.3f}")
    ax.set_xlabel("softmax probability on GT token")
    ax.set_ylabel("# masked positions")
    ax.set_title("Model confidence on the ground-truth gene token")
    ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--val-samples", type=int, default=76)
    ap.add_argument("--pool-spots", type=int, default=8000)
    ap.add_argument("--encode-batch", type=int, default=32)
    ap.add_argument("--mask-ratio", type=float, default=None,
                     help="Override encoder's mask_ratio. Default = the value baked into the ckpt.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/eval/msm_eval")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--per-gene-min-obs", type=int, default=20,
                     help="Minimum #observations a gene must have to enter best/worst tables.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prep = Path(args.prepared_dir)
    log.info(f"building eval pool [{args.split}] ({args.pool_spots} from {args.val_samples} samples)...")
    hvg_full, organ, source, _meta = build_eval_pool(
        prep, args.val_samples, args.pool_spots, split=args.split, return_meta=True,
    )
    if organ is None:
        organ = np.array([""] * hvg_full.shape[0])
    if source is None:
        source = np.array([""] * hvg_full.shape[0])
    log.info(f"pool: {hvg_full.shape} | sources={sorted(set(source.tolist()))}")

    # Gene names for the full HVG vocab (after vocab_clip).
    full_dim = hvg_full.shape[1]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ck_s in args.ckpts:
        ck = Path(ck_s)
        ckpt_name = ck.parent.name
        log.info(f"== {ck} ==")
        enc, _cfg, vocab_keep, gene_norm_cfg = load_tx_encoder(ck, device=device)
        # Optional override of internal mask_ratio.
        if args.mask_ratio is not None and hasattr(enc, "mask_ratio"):
            enc.mask_ratio = float(args.mask_ratio)
            log.info(f"  using mask_ratio={enc.mask_ratio}")
        n_special = int(getattr(enc, "N_SPECIAL", 4))
        # Effective gene names (after vocab_clip).
        gene_names = json.loads((prep / "hvg_vocab.json").read_text())
        if vocab_keep is not None:
            gene_names = [gene_names[int(i)] for i in vocab_keep]

        hvg_norm = _normalise_hvg(hvg_full, vocab_keep, gene_norm_cfg, full_dim)
        log.info(f"running encoder with forced internal masking — encode-batch={args.encode_batch}")
        res = _msm_forward(enc, hvg_norm, device=device,
                            batch=args.encode_batch, seed=args.seed)
        gt = res["gt_token"]
        logits = res["pred_logits"]
        conf = res["confidence_gt"]
        row_of_m = res["sample_row"]
        log.info(f"masked positions: {gt.shape[0]:,} across {hvg_full.shape[0]} spots "
                  f"(mask_ratio≈{gt.shape[0] / max(1, hvg_full.shape[0] * len(gene_names)):.4f})")
        if gt.shape[0] == 0:
            log.warning("no masked positions — abort")
            continue

        # Aggregates
        ks = (1, 5, 10, 20)
        micro = {k: _topk_acc(logits, gt, k) for k in ks}
        per_gene = _per_gene_table(gt, logits, gene_names, n_special, ks=(1, 5, 10))
        macro = {k: float(per_gene[f"top{k}"].mean()) for k in (1, 5, 10) if not per_gene.empty}
        macro[20] = float("nan")  # not in per-gene table; only micro
        nll = _nll(logits, gt)

        # Per-organ and per-source: use the spot of origin to look up the metadata.
        # We report TOP-10 accuracy per group (primary metric — easier to see
        # signal vs. top-1 which is bottlenecked by abundant genes).  top-1
        # remains in the CSVs for reference.
        per_organ_acc, per_source_acc = {}, {}
        per_organ_acc_t1, per_source_acc_t1 = {}, {}
        per_organ_n, per_source_n = {}, {}
        organ_of_m = organ[row_of_m]
        source_of_m = source[row_of_m]
        # Compute top-10 correctness once across all masked positions.
        top10_idx = np.argpartition(-logits.astype(np.float32, copy=False), 9, axis=1)[:, :10]
        correct_top10 = (top10_idx == gt[:, None]).any(axis=1)
        top1_pred = np.argmax(logits.astype(np.float32, copy=False), axis=1)
        correct_top1 = (top1_pred == gt)
        for grp_name, grp_vals, acc_t10, acc_t1, cnt_dict in [
            ("organ", organ_of_m, per_organ_acc, per_organ_acc_t1, per_organ_n),
            ("source", source_of_m, per_source_acc, per_source_acc_t1, per_source_n),
        ]:
            for u in sorted(set(grp_vals.tolist())):
                if not u:
                    continue
                m = grp_vals == u
                if m.sum() < 20:
                    continue
                acc_t10[u] = float(correct_top10[m].mean())
                acc_t1[u] = float(correct_top1[m].mean())
                cnt_dict[u] = int(m.sum())

        ck_dir = out_root / ckpt_name
        ck_dir.mkdir(parents=True, exist_ok=True)
        per_gene.to_csv(ck_dir / "per_gene.csv", index=False)
        # Best/worst tables (min_obs filter) — RANKED BY TOP-10 (primary metric).
        # top-1 / top-5 stay in the CSV columns for inspection.
        pg_filt = per_gene[per_gene["n_obs"] >= args.per_gene_min_obs]
        pg_filt.nlargest(50, "top10").to_csv(ck_dir / "best_genes.csv", index=False)
        pg_filt.nsmallest(50, "top10").to_csv(ck_dir / "worst_genes.csv", index=False)
        pd.DataFrame([{"organ": k,
                        "top10_acc": v,
                        "top1_acc": per_organ_acc_t1[k],
                        "n": per_organ_n[k]}
                       for k, v in per_organ_acc.items()]).to_csv(
            ck_dir / "per_organ.csv", index=False)
        pd.DataFrame([{"source": k,
                        "top10_acc": v,
                        "top1_acc": per_source_acc_t1[k],
                        "n": per_source_n[k]}
                       for k, v in per_source_acc.items()]).to_csv(
            ck_dir / "per_source.csv", index=False)

        if not args.no_viz:
            _plot_topk_bar(micro, macro, ck_dir / "fig_topk_bar.png")
            _plot_per_gene_hist(per_gene, ck_dir / "fig_per_gene_hist.png")
            _plot_best_worst_bar(per_gene, ck_dir / "fig_best_worst.png",
                                   top_n=25, min_obs=args.per_gene_min_obs,
                                   rank_by="top10")
            _plot_group_bar(per_organ_acc, per_organ_n,
                              "MSM top-10 by organ", ck_dir / "fig_per_organ.png")
            _plot_group_bar(per_source_acc, per_source_n,
                              "MSM top-10 by source", ck_dir / "fig_per_source.png")
            _plot_confidence_hist(conf, ck_dir / "fig_confidence.png")

        summary_rows.append({
            "ckpt": ckpt_name,
            "n_masked_positions": int(gt.shape[0]),
            "n_unique_genes": int(per_gene.shape[0]),
            "top1_micro": micro[1],
            "top5_micro": micro[5],
            "top10_micro": micro[10],
            "top20_micro": micro[20],
            "top1_macro": macro.get(1, float("nan")),
            "top5_macro": macro.get(5, float("nan")),
            "top10_macro": macro.get(10, float("nan")),
            "nll": nll,
            "mean_conf_gt": float(np.mean(conf)),
            "median_conf_gt": float(np.median(conf)),
            "split": args.split,
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_root / "msm_summary.csv", index=False)
    log.info(f"saved {out_root/'msm_summary.csv'}")
    print()
    print("-" * 96)
    print("Masked Symbol Modeling eval — internal masking re-used at eval time.")
    print("  micro = over all masked tokens (abundant genes dominate)")
    print("  macro = average of per-gene accuracies (each gene weighted equally)")
    print("-" * 96)
    if not df.empty:
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
