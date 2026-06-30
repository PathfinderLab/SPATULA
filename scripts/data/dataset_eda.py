"""Comprehensive dataset EDA + Stage-1 design verification against
docs/archive/contexts/spatula_research_overview.pdf.

Outputs (under results/cache/<prepared_dir>/eda_v2/):
  - dataset_summary.json    : per-source spot/sample counts + sample lists
  - eda_summary.md          : human-readable report (vocab health, design check)
  - fig_n_spots_per_sample.png   : sample-size distribution per source
  - fig_genes_per_spot.png       : zero-removal check (median ~150 expected)
  - fig_hvg_prevalence.png       : per-gene coverage across samples (healthy = uniform)
  - fig_hvg_singleton_share.png  : "특정 샘플에만 등장하는 HVG" 비율 → vocab healthiness
  - fig_norm_before_after.png    : 3 sample spots, raw log1p vs global-median-normalized
  - fig_hvg_vocab_overlap.png    : intersection of HVG-per-sample with the global vocab

Usage:
  PYTHONPATH=src python scripts/data/dataset_eda.py \
      --prepared-dir results/cache/prepared_expanded
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for
log = get_logger("eda")


def _source_of(shard_path: Path) -> str:
    name = shard_path.stem
    if name.endswith(".st1k"): return "st1k"
    if name.endswith(".spatialcorpus"): return "spatialcorpus"
    return "hest"


def _strip_source(shard_path: Path) -> str:
    name = shard_path.stem
    if name.endswith(".st1k"): return name[:-5]
    if name.endswith(".spatialcorpus"): return name[:-len(".spatialcorpus")]
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — per-shard scan: spots, library size, expressed-gene count, presence
# ─────────────────────────────────────────────────────────────────────────────

def scan_shards(prepared: Path, hvg_vocab: list[str]):
    """One pass over every shard.  Returns dict keyed by sample_id."""
    n_hvg = len(hvg_vocab)
    out = {}
    shards = sorted(prepared.glob("*.h5"))
    for p in shards:
        sid = _strip_source(p)
        src = _source_of(p)
        try:
            with h5py.File(p, "r") as f:
                if "hvg_log" not in f:
                    continue
                X = f["hvg_log"][:]                  # (S, n_hvg) log1p-normalized
        except Exception as e:
            log.warning(f"skip {p.name}: {e}")
            continue
        if X.shape[1] != n_hvg:
            log.warning(f"skip {p.name}: hvg_dim {X.shape[1]} != vocab {n_hvg}")
            continue
        nnz_per_spot = (X > 0).sum(axis=1).astype(np.int32)
        lib_size = X.sum(axis=1).astype(np.float32)
        gene_present_in_sample = (X > 0).any(axis=0)  # (n_hvg,) bool
        out[sid] = {
            "source": src,
            "n_spots": int(X.shape[0]),
            "nnz_per_spot": nnz_per_spot,
            "lib_size": lib_size,
            "gene_present": gene_present_in_sample,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vocab health
# ─────────────────────────────────────────────────────────────────────────────

def vocab_health(per_sample: dict, hvg_vocab: list[str]) -> dict:
    n_samples = len(per_sample)
    n_hvg = len(hvg_vocab)
    # for each gene, count how many samples have it expressed in ≥1 spot
    coverage = np.zeros(n_hvg, dtype=np.int64)
    for s in per_sample.values():
        coverage += s["gene_present"].astype(np.int64)
    prevalence = coverage / max(1, n_samples)

    singleton_count = int((coverage <= 1).sum())
    rare_count = int((prevalence < 0.10).sum())
    common_count = int((prevalence >= 0.50).sum())
    universal_count = int((prevalence >= 0.95).sum())

    # Top-10 rarest + top-10 most-common genes
    rare_idx = np.argsort(coverage)[:10]
    common_idx = np.argsort(-coverage)[:10]
    return {
        "n_samples": n_samples,
        "n_hvg": n_hvg,
        "coverage_array": coverage,
        "prevalence_array": prevalence,
        "n_singleton (coverage ≤ 1 sample)": singleton_count,
        "n_rare    (prevalence < 10%)":      rare_count,
        "n_common  (prevalence ≥ 50%)":      common_count,
        "n_universal (prevalence ≥ 95%)":    universal_count,
        "rarest_examples": [(hvg_vocab[i], int(coverage[i])) for i in rare_idx],
        "common_examples": [(hvg_vocab[i], int(coverage[i])) for i in common_idx],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _src_colors():
    return {"hest": "#3a82e0", "st1k": "#e07a3a", "spatialcorpus": "#7e3aff"}


def fig_n_spots_per_sample(per_sample, out_path):
    by_src = defaultdict(list)
    for s in per_sample.values():
        by_src[s["source"]].append(s["n_spots"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for src, vals in by_src.items():
        ax.hist(vals, bins=40, alpha=0.6, label=f"{src} (n={len(vals)})",
                color=_src_colors().get(src, "gray"))
    ax.set_xlabel("# spots / sample"); ax.set_ylabel("# samples")
    ax.set_title("Spots-per-sample distribution by source")
    ax.set_xscale("log"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_genes_per_spot(per_sample, out_path):
    by_src = defaultdict(list)
    for s in per_sample.values():
        by_src[s["source"]].append(s["nnz_per_spot"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for src, lists in by_src.items():
        flat = np.concatenate(lists) if lists else np.array([])
        ax.hist(flat, bins=80, alpha=0.5,
                label=f"{src}  median={int(np.median(flat))}  p25/p75={int(np.percentile(flat,25))}/{int(np.percentile(flat,75))}",
                color=_src_colors().get(src, "gray"), density=True)
    ax.set_xlabel("# expressed HVG / spot (zero-removed token length)")
    ax.set_ylabel("density")
    ax.set_title("Per-spot expressed-HVG count by source — zero-removal sanity check\n"
                 "Short sequences (< 30) → masked-modeling task degenerate")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_hvg_prevalence(prevalence, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(prevalence, bins=50, color="#3a82e0", alpha=0.85)
    ax.set_xlabel("prevalence — fraction of samples in which this HVG has any non-zero spot")
    ax.set_ylabel("# HVG genes")
    healthy_share = float((prevalence >= 0.5).mean())
    ax.set_title(f"HVG prevalence across samples\n"
                 f"healthy (≥50% samples): {healthy_share:.1%}   "
                 f"sample-specific (<10%): {(prevalence<0.10).mean():.1%}")
    ax.axvline(0.10, color="red", linestyle="--", alpha=0.5, label="rare cutoff 10%")
    ax.axvline(0.50, color="green", linestyle="--", alpha=0.5, label="common cutoff 50%")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_singleton_share(per_sample, hvg_vocab, out_path):
    """For each SAMPLE, count how many of its expressed genes are 'sample-specific'
    (prevalence < 10% across all samples).  Vocab health: low share = healthy."""
    n_samples = len(per_sample)
    coverage = np.zeros(len(hvg_vocab), dtype=np.int64)
    for s in per_sample.values():
        coverage += s["gene_present"].astype(np.int64)
    prevalence = coverage / max(1, n_samples)
    rare_mask = prevalence < 0.10        # per-gene flag

    src_to_shares = defaultdict(list)
    for s in per_sample.values():
        present = s["gene_present"]      # (n_hvg,) bool
        n_present = int(present.sum())
        if n_present == 0:
            continue
        n_rare_in_sample = int((present & rare_mask).sum())
        src_to_shares[s["source"]].append(n_rare_in_sample / n_present)

    fig, ax = plt.subplots(figsize=(8, 4))
    for src, vals in src_to_shares.items():
        ax.hist(vals, bins=40, alpha=0.55,
                label=f"{src}  median={np.median(vals):.2%}",
                color=_src_colors().get(src, "gray"))
    ax.set_xlabel("share of this sample's expressed HVG that are 'rare' across the corpus (prevalence < 10%)")
    ax.set_ylabel("# samples")
    ax.set_title("Vocab-health check: how much of each sample is 'sample-specific'?\n"
                 "lower = healthier (the vocab covers what this sample actually expresses)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_norm_before_after(prepared, hvg_vocab, stats_path, out_path, n_samples=3, n_spots=20):
    """Take a few spots, show raw log1p vs gene-level (÷ global median) normalized."""
    if not stats_path.exists():
        log.warning(f"gene_stats.npz not found at {stats_path} — skipping norm figure")
        return
    stats = np.load(stats_path)
    median = stats["median"].astype(np.float32)   # (n_hvg,)
    # Pick a few shards, sample n_spots each
    shards = sorted(prepared.glob("*.h5"))[:n_samples]
    fig, axes = plt.subplots(2, n_samples, figsize=(5*n_samples, 8), squeeze=False)
    for j, p in enumerate(shards):
        src = _source_of(p)
        sid = _strip_source(p)
        with h5py.File(p, "r") as f:
            X = f["hvg_log"][:n_spots]
        # Raw log1p — plot a small box/violin per spot, restricted to non-zero values
        for s in range(min(n_spots, X.shape[0])):
            nz = X[s][X[s] > 0]
            if nz.size:
                axes[0, j].scatter(np.full(nz.size, s), nz, s=4, alpha=0.4, color="#3a82e0")
        axes[0, j].set_title(f"{src}  {sid[:30]}\nraw log1p (nonzero values only)")
        axes[0, j].set_xlabel("spot idx"); axes[0, j].set_ylabel("log1p value")

        # Normalized: x / (median[g] + eps).  Median is mostly 0 → identical for those.
        # Show only the genes WITH a meaningful median (>= 0.01).  Others are passed through.
        eps = 1e-6
        norm = X / (median[None, :] + eps)
        for s in range(min(n_spots, X.shape[0])):
            nz = norm[s][X[s] > 0]
            if nz.size:
                axes[1, j].scatter(np.full(nz.size, s), nz, s=4, alpha=0.4, color="#e07a3a")
        axes[1, j].set_title("after gene-level norm (÷ global_median)")
        axes[1, j].set_xlabel("spot idx"); axes[1, j].set_ylabel("normalized value")
        axes[1, j].set_yscale("symlog", linthresh=1.0)
    fig.suptitle("Normalization sanity — raw log1p vs ÷ global_median (per-gene)", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_hvg_vocab_overlap(per_sample, n_hvg, out_path):
    """Fraction of vocab covered by each sample (intersection size / vocab size)."""
    by_src = defaultdict(list)
    for s in per_sample.values():
        by_src[s["source"]].append(float(s["gene_present"].sum()) / n_hvg)
    fig, ax = plt.subplots(figsize=(8, 4))
    for src, vals in by_src.items():
        ax.hist(vals, bins=40, alpha=0.6,
                label=f"{src}  median={np.median(vals):.1%}",
                color=_src_colors().get(src, "gray"))
    ax.set_xlabel("fraction of the 2048-HVG vocab expressed in this sample (≥1 spot)")
    ax.set_ylabel("# samples")
    ax.set_title("Sample × Vocab coverage")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="/workspace/mm_align/results/cache/prepared_expanded")
    ap.add_argument("--out-dir", default=None,
                    help="Default: results/eda/<prepared-dir-name>/eda_v2/")
    args = ap.parse_args()
    prep = Path(args.prepared_dir)
    out_dir = Path(args.out_dir or (reports_dir_for(prep) / "eda_v2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"out_dir = {out_dir}")

    hvg_vocab = json.loads((prep / "hvg_vocab.json").read_text())
    log.info(f"loaded HVG vocab: {len(hvg_vocab)} genes from {prep}/hvg_vocab.json")

    # Pass 1 — scan all shards
    log.info("scanning shards (this can take a few minutes — 1389 shards)…")
    per_sample = scan_shards(prep, hvg_vocab)
    log.info(f"scanned {len(per_sample)} shards")

    # Dataset summary
    by_src = defaultdict(lambda: {"n_samples": 0, "total_spots": 0, "sample_ids": []})
    for sid, s in per_sample.items():
        by_src[s["source"]]["n_samples"] += 1
        by_src[s["source"]]["total_spots"] += s["n_spots"]
        by_src[s["source"]]["sample_ids"].append(sid)

    summary = {src: {"n_samples": v["n_samples"],
                     "total_spots": v["total_spots"],
                     "sample_ids_preview": sorted(v["sample_ids"])[:15],
                     "sample_ids_count_check": len(v["sample_ids"])}
               for src, v in by_src.items()}
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # Vocab health
    health = vocab_health(per_sample, hvg_vocab)
    # strip arrays for json
    health_json = {k: v for k, v in health.items()
                   if k not in ("coverage_array", "prevalence_array")}
    (out_dir / "vocab_health.json").write_text(json.dumps(health_json, indent=2))

    # Figures
    log.info("rendering figures…")
    fig_n_spots_per_sample(per_sample, out_dir / "fig_n_spots_per_sample.png")
    fig_genes_per_spot(per_sample, out_dir / "fig_genes_per_spot.png")
    fig_hvg_prevalence(health["prevalence_array"], out_dir / "fig_hvg_prevalence.png")
    fig_singleton_share(per_sample, hvg_vocab, out_dir / "fig_hvg_singleton_share.png")
    fig_hvg_vocab_overlap(per_sample, len(hvg_vocab), out_dir / "fig_hvg_vocab_overlap.png")
    fig_norm_before_after(prep, hvg_vocab, prep / "gene_stats.npz",
                          out_dir / "fig_norm_before_after.png")

    # Markdown report
    md = []
    md.append("# Stage-1 Dataset EDA + Design Verification\n")
    md.append(f"_prepared dir_: `{prep}`\n")
    md.append("## 1. Dataset summary\n")
    md.append("| source | # samples | # spots | example IDs |")
    md.append("|---|---:|---:|---|")
    for src, v in by_src.items():
        ex = ", ".join(sorted(v["sample_ids"])[:5])
        md.append(f"| {src} | {v['n_samples']} | {v['total_spots']:,} | {ex}, … |")
    md.append("")

    md.append("## 2. Zero-removal sanity (per-spot expressed HVG count)\n")
    md.append("| source | median | p25 | p75 | min | max |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for src, lists in defaultdict(list, {s["source"]: [] for s in per_sample.values()}).items():
        pass
    # recompute properly
    src_nnz = defaultdict(list)
    for s in per_sample.values():
        src_nnz[s["source"]].extend(s["nnz_per_spot"].tolist())
    for src, vals in src_nnz.items():
        a = np.array(vals)
        md.append(f"| {src} | {int(np.median(a))} | {int(np.percentile(a,25))} | "
                  f"{int(np.percentile(a,75))} | {int(a.min())} | {int(a.max())} |")
    md.append("")
    md.append("**Interpretation**: spot당 expressed HVG가 30 미만이면 masked-symbol modeling "
              "context가 너무 짧아 학습 어려움. SPATULA paper의 'zero-removed token' 가정은 "
              "median ~100+ 일 때 잘 작동.\n")

    md.append("## 3. Vocab health\n")
    md.append(f"- HVG vocab size: **{health['n_hvg']}**")
    md.append(f"- # samples in pool: **{health['n_samples']}**")
    md.append(f"- Singleton genes (only in ≤1 sample): **{health['n_singleton (coverage ≤ 1 sample)']}**")
    md.append(f"- Rare genes (prevalence < 10%): **{health['n_rare    (prevalence < 10%)']}**")
    md.append(f"- Common genes (prevalence ≥ 50%): **{health['n_common  (prevalence ≥ 50%)']}**")
    md.append(f"- Universal genes (prevalence ≥ 95%): **{health['n_universal (prevalence ≥ 95%)']}**")
    md.append("")
    md.append("### Top-10 most-common HVG (likely housekeeping):")
    md.append("| gene | n_samples_with_nonzero |")
    md.append("|---|---:|")
    for g, c in health["common_examples"]:
        md.append(f"| {g} | {c} |")
    md.append("\n### Top-10 rarest HVG (candidates for vocab pruning):")
    md.append("| gene | n_samples_with_nonzero |")
    md.append("|---|---:|")
    for g, c in health["rarest_examples"]:
        md.append(f"| {g} | {c} |")
    md.append("")

    md.append("## 4. Stage-1 design verification vs research_overview.pdf\n")
    md.append("| design choice (PDF) | implementation | status |")
    md.append("|---|---|---|")
    md.append("| Vocab = HVG 2048 | hvg_vocab.json len = " + str(health['n_hvg']) + " | " +
              ("✅" if health['n_hvg'] == 2048 else "⚠️") + " |")
    md.append("| Zero-removed tokens | `_pack_nonzero` in top_hvg_gene.py drops zero-expressed | ✅ |")
    md.append("| Fourier value features | `gene_emb` uses Fourier projection | ✅ |")
    md.append("| Symbol embedding (nn.Embedding) | symbol_embed in `_GeneEmbedding` | ✅ |")
    md.append("| CLS token for spot representation | `cls_emb = x[:, 0]` → `cls_proj` → h_tx | ✅ |")
    md.append("| Special tokens: PAD/MASK/CLS (+UNK) | PAD=0, MASK=1, CLS=2, UNK=3 (4 specials) | ✅ |")
    md.append("| Primary obj: MSM (cross-entropy) | masked_symbol_ce in masked_tx.py | ✅ |")
    md.append("| Aux obj: Predictive JEPA (EMA + smooth-L1) | masked_jepa uses cosine, not smooth-L1 | ⚠️ — JEPA distance metric mismatch (cosine vs smooth-L1 in paper) |")
    md.append("| λ for JEPA: 0.1~0.5 | currently jepa_weight=1.0 | ⚠️ — needs lower |")
    md.append("| Two-stage norm: spot 1e4 + gene÷median | prepare_data does spot 1e4 + log1p; gene÷median is run-time via `gene_norm.mode` | ⚠️ — gene-level mode currently default=`none`; should be `global_median_divide` per paper |")
    md.append("")

    md.append("## 5. Figures\n")
    md.append("- `fig_n_spots_per_sample.png` — sample-size distribution per source")
    md.append("- `fig_genes_per_spot.png`     — zero-removal sanity (sequence length per spot)")
    md.append("- `fig_hvg_prevalence.png`     — per-HVG coverage across all samples")
    md.append("- `fig_hvg_singleton_share.png`— per-sample share of 'rare' HVG (vocab health)")
    md.append("- `fig_hvg_vocab_overlap.png`  — per-sample vocab coverage")
    md.append("- `fig_norm_before_after.png`  — raw log1p vs ÷ global_median normalized")

    (out_dir / "eda_summary.md").write_text("\n".join(md))
    log.info(f"wrote eda_summary.md + dataset_summary.json + vocab_health.json")
    log.info(f"figures under {out_dir}/")


if __name__ == "__main__":
    main()
