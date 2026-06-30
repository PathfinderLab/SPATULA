"""Per-source / per-organ / per-sample data-processing quality test.

Reads a finished `prepared_*` directory and produces an intuitive report on
how well the vocab + cleaning pipeline matched genes for each subgroup.

For every shard we measure:
  * coverage     = mean(seq_len)  /  vocab_size               # what fraction of vocab is present per spot
  * seq_len      = (hvg_log > 0).sum(axis=1)                  # genes-per-spot
  * vocab_hit    = (hvg_log.sum(axis=0) > 0).sum() / vocab    # genes-in-this-sample-that-survived-vocab-projection
  * vocab_density= mean(seq_len) / vocab_size

Grouped by:
  - source            (hest / st1k / spatialcorpus)
  - organ             (HEST metadata join)
  - cancer/disease    (HEST metadata join)
  - sample_id         (per-shard sorted table)

Outputs to `results/eda/<prepared_name>/process_qc/`:
  process_qc.md                                 — human-readable summary
  per_sample_quality.csv                        — one row per shard
  fig_seqlen_by_source.png                      — box: seq_len distribution by source
  fig_vocab_hit_by_source.png                   — box: per-sample vocab coverage by source
  fig_quality_by_organ.png                      — heatmap: organ × metric
  fig_quality_scatter.png                       — n_genes_clean vs vocab_hit, color=source
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for

log = get_logger("process_qc")


def _shard_metrics(shard_path: Path, vocab_size: int,
                   max_spots: int = 2000, rng: np.random.Generator = None) -> dict:
    """Pull (n_spots, n_in_vocab, seq_len stats) from one .h5 shard.

    Subsamples spots to keep runtime bounded — vocab-matching stats are stable
    even at 2000 spots/shard.
    """
    rng = rng or np.random.default_rng(0)
    with h5py.File(shard_path, "r") as f:
        if "hvg_log" not in f:
            return {}
        n_spots_total = f["hvg_log"].shape[0]
        if n_spots_total > max_spots:
            sel = np.sort(rng.choice(n_spots_total, max_spots, replace=False))
            hvg = f["hvg_log"][sel]
        else:
            hvg = f["hvg_log"][:]
        attrs = dict(f.attrs)
    nz = (hvg > 0).astype(np.int32)
    seq_lens = nz.sum(axis=1)
    n_genes_in_sample = int((nz.sum(axis=0) > 0).sum())  # ≥1 spot expresses
    return {
        "n_spots_total": int(n_spots_total),
        "n_spots_sampled": int(hvg.shape[0]),
        "n_genes_in_vocab_sample": n_genes_in_sample,
        "vocab_hit": n_genes_in_sample / vocab_size,
        "seq_len_mean": float(seq_lens.mean()),
        "seq_len_median": float(np.median(seq_lens)),
        "seq_len_p10": float(np.percentile(seq_lens, 10)),
        "seq_len_p90": float(np.percentile(seq_lens, 90)),
        "frac_zero_spot": float((seq_lens == 0).mean()),
        "vocab_density": float(seq_lens.mean() / vocab_size),
        "sample_id_attr": str(attrs.get("sample_id", "")),
        "source_attr": str(attrs.get("source", "")),
    }


def _infer_source_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith(".st1k.h5"):
        return "st1k"
    if name.endswith(".spatialcorpus.h5"):
        return "spatialcorpus"
    return "hest"


def _load_hest_meta(csv_path: Path | None) -> pd.DataFrame | None:
    if csv_path is None or not csv_path.exists():
        log.warning(f"HEST metadata not found at {csv_path}; organ/disease join skipped.")
        return None
    df = pd.read_csv(csv_path)
    # Normalise columns we want.
    keep = {"id": "sample_id", "organ": "organ", "disease_state": "disease",
            "cancer_type": "cancer_type", "tissue": "tissue", "subseries": "subseries"}
    cols = {c: keep[c] for c in keep if c in df.columns}
    if "id" not in df.columns:
        log.warning("HEST CSV has no `id` column; organ join skipped.")
        return None
    return df.rename(columns=cols)[list(cols.values())]


def _save_box(df: pd.DataFrame, x: str, y: str, title: str, out: Path,
              order: list[str] | None = None, figsize=(10, 5),
              ylabel: str | None = None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, order=order, ax=ax, showfliers=False,
                palette="Set2")
    sns.stripplot(data=df, x=x, y=y, order=order, ax=ax, size=3, color="0.3",
                  alpha=0.5, jitter=0.25)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ylabel: ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def _save_organ_heatmap(df: pd.DataFrame, out: Path):
    if "organ" not in df.columns or df["organ"].isna().all():
        return
    metrics = ["seq_len_mean", "seq_len_median", "vocab_hit", "vocab_density",
                "frac_zero_spot"]
    agg = (df.groupby("organ")[metrics].median()
             .sort_values("seq_len_mean", ascending=False))
    # Z-score columns so the heatmap is interpretable across metrics with
    # different scales.
    z = (agg - agg.mean()) / (agg.std(ddof=0).replace(0, 1))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(agg))))
    sns.heatmap(z, annot=agg.round(3), fmt="", cmap="RdBu_r", center=0,
                cbar_kws={"label": "z (vs other organs)"}, ax=ax)
    ax.set_title("Process quality by organ (annot = raw value; color = z-score across organs)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def _save_quality_scatter(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="n_genes_clean", y="vocab_hit", hue="source",
                    style="source", size="n_spots_total", sizes=(20, 200),
                    alpha=0.7, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("# cleaned genes in sample (pre-vocab-projection)")
    ax.set_ylabel("vocab_hit = fraction of vocab seen in this sample")
    ax.set_title("Per-sample: raw breadth → vocab coverage  (dot size = #spots)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--hest-csv", default="/data/hest/HEST_v1_1_0.csv")
    ap.add_argument("--max-spots-per-shard", type=int, default=2000)
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Cap for quick smoke runs (default = all).")
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    vocab = json.loads((prepared / "hvg_vocab.json").read_text())
    vocab_size = len(vocab)
    reports = reports_dir_for(prepared)
    out_dir = reports / "process_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"prepared = {prepared}  vocab = {vocab_size}  out = {out_dir}")

    # ── 1. Read sample_qc.csv (raw → clean → noise stats per train sample) ──
    sample_qc_path = prepared / "sample_qc.csv"
    if sample_qc_path.exists():
        sample_qc = pd.read_csv(sample_qc_path)
        log.info(f"sample_qc rows: {len(sample_qc)}")
    else:
        sample_qc = pd.DataFrame(columns=["sample_id","source","n_genes_raw",
                                          "n_genes_clean","n_genes_noise","n_spots"])
        log.warning(f"{sample_qc_path} not found — rerun prepare_data.py to emit it.")

    # ── 2. Walk every shard and compute coverage ──
    shards = sorted(prepared.glob("*.h5"))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    log.info(f"scanning {len(shards)} shards (≤{args.max_spots_per_shard} spots each)…")
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for s in shards:
        m = _shard_metrics(s, vocab_size, max_spots=args.max_spots_per_shard, rng=rng)
        if not m: continue
        sid = m.pop("sample_id_attr") or s.stem.split(".")[0]
        src = m.pop("source_attr") or _infer_source_from_filename(s)
        rows.append({"sample_id": sid, "source": src, **m})

    df = pd.DataFrame(rows)

    # ── 3. Join sample_qc (raw/clean gene counts) ──
    if not sample_qc.empty:
        df = df.merge(sample_qc[["sample_id","source","n_genes_raw","n_genes_clean",
                                  "n_genes_noise"]],
                      on=["sample_id","source"], how="left")
    else:
        df["n_genes_raw"] = np.nan; df["n_genes_clean"] = np.nan; df["n_genes_noise"] = np.nan

    # ── 4. Join HEST organ metadata (HEST only) ──
    hest_meta = _load_hest_meta(Path(args.hest_csv))
    if hest_meta is not None:
        df = df.merge(hest_meta, on="sample_id", how="left")

    csv_path = out_dir / "per_sample_quality.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"saved per-sample CSV → {csv_path}")

    # ── 5. Figures ──
    src_order = [s for s in ["hest","st1k","spatialcorpus"] if s in set(df["source"])]
    _save_box(df, "source", "seq_len_mean",
              "Genes-per-spot by data source  (higher = denser, more informative tokens)",
              out_dir / "fig_seqlen_by_source.png", order=src_order,
              ylabel="mean seq_len (genes/spot)")
    _save_box(df, "source", "vocab_hit",
              "Vocab coverage per sample by data source  (fraction of vocab present)",
              out_dir / "fig_vocab_hit_by_source.png", order=src_order,
              ylabel="vocab_hit (0..1)")
    _save_organ_heatmap(df, out_dir / "fig_quality_by_organ.png")
    _save_quality_scatter(df, out_dir / "fig_quality_scatter.png")

    # ── 6. Markdown report ──
    md = [f"# Process Quality QC\n",
          f"prepared: `{prepared.name}`  ·  vocab: **{vocab_size}**  ·  shards: **{len(df)}**\n",
          "## 1. Per-source summary  *(higher seq_len / vocab_hit = cleaner processing)*\n"]
    src_sum = df.groupby("source").agg(
        n_samples=("sample_id","count"),
        n_spots_total=("n_spots_total","sum"),
        seq_len_mean=("seq_len_mean","mean"),
        seq_len_median=("seq_len_median","median"),
        vocab_hit=("vocab_hit","mean"),
        n_genes_clean=("n_genes_clean","mean"),
        n_genes_noise=("n_genes_noise","mean"),
        frac_zero_spot=("frac_zero_spot","mean"),
    ).round(3)
    md.append(src_sum.to_markdown())
    md.append("\n![](fig_seqlen_by_source.png)\n")
    md.append("![](fig_vocab_hit_by_source.png)\n")

    md.append("\n## 2. Per-organ summary (HEST only)\n")
    if "organ" in df.columns and df["organ"].notna().any():
        org_sum = (df[df["organ"].notna()]
                   .groupby("organ")
                   .agg(n=("sample_id","count"),
                        seq_len_mean=("seq_len_mean","mean"),
                        vocab_hit=("vocab_hit","mean"),
                        n_genes_clean=("n_genes_clean","mean"))
                   .sort_values("seq_len_mean", ascending=False).round(3))
        md.append(org_sum.to_markdown())
        md.append("\n![](fig_quality_by_organ.png)\n")
    else:
        md.append("_no HEST metadata; organ join skipped._\n")

    md.append("\n## 3. Raw breadth → vocab coverage (per-sample)\n")
    md.append("![](fig_quality_scatter.png)\n")

    md.append("\n## 4. Worst 15 samples by seq_len_mean  *(processing red-flags)*\n")
    worst = df.nsmallest(15, "seq_len_mean")[
        ["sample_id","source","organ" if "organ" in df.columns else "source",
         "n_genes_raw","n_genes_clean","n_genes_noise",
         "seq_len_mean","vocab_hit","n_spots_total"]
    ]
    md.append(worst.round(3).to_markdown(index=False))
    md.append("\n")

    (out_dir / "process_qc.md").write_text("\n".join(md))
    log.info(f"report → {out_dir / 'process_qc.md'}")


if __name__ == "__main__":
    main()
