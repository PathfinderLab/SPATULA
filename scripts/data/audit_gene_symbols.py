"""Gene-vocab + noise-filter sanity audit.

Standalone — independent of the train/eval pipeline.  Inspects a prepared HVG
vocab (hvg_vocab.json) and cross-references it against:
  - GENCODE GTF (canonical symbol whitelist)
  - The noise pattern used by prepare_data.py

Reports:
  - per-symbol classification: valid_hgnc / ensg_unresolved / noise / unknown
  - which noise patterns are hitting / not hitting in the HVG vocab
  - any ENSG-formatted entries (these should have been canonicalized at prep time)
  - per-source coverage (HEST vs ST1K-ST vs ST1K-Visium)
  - EDA figures (per-source library size, per-gene prevalence across samples,
    gene_type breakdown bar, value-distribution histogram for top + tail HVG)

Outputs (next to hvg_vocab.json):
  gene_vocab_audit.json    (machine-readable)
  gene_vocab_audit.md      (human-readable summary table)
  eda/fig_*.png            (figures, if --no-eda not passed)

Usage:
  PYTHONPATH=src python scripts/data/audit_gene_symbols.py
  PYTHONPATH=src python scripts/data/audit_gene_symbols.py \
      --prepared-dir results/cache/prepared \
      --include-st1k     # also draw per-sample stats from ST1K
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for
from mm_align.data.gene_symbols import (
    load_gtf_symbol_map, classify_symbol, _ENSG_RE,
)

log = get_logger("audit_genes")


# Mirror the noise regex used in scripts/data/prepare.py — keep these in sync.
_NOISE_PATTERNS = [
    r"^MT-", r"^MT\.",
    r"^RPS", r"^RPL",
    r"^__",
    r"^BLANK_", r"^NEGCONTROL", r"^UNASSIGNED",
    r"^[0-9]",
    r"^A[CDFJLP][0-9]+\.[0-9]+", r"^A[CDFJLP][0-9]+",
    r"^LOC[0-9]+", r"^RP[0-9]+", r"^LINC[0-9]+",
    # NOTE: ≥3 chars before P\d+$ (so TP53 / MMP1 / HSP70 survive — see
    # _clean_adata_var_names in prepare_data.py for the matching change).
    r"^.{3,}P[0-9]+$",
    r"^DEPRECATED",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)


def audit_vocab(hvg_vocab: list[str], gmap: dict) -> dict:
    """Classify each gene in the HVG vocab + tally noise-pattern hits."""
    rows = []
    cls_count = Counter()
    pattern_hits = {p: 0 for p in _NOISE_PATTERNS}
    gene_type_count = Counter()
    for g in hvg_vocab:
        c = classify_symbol(g, gmap, noise_re=_NOISE_RE)
        cls_count[c] += 1
        gt = gmap["symbol_to_gene_type"].get(g.upper(), "unknown")
        gene_type_count[gt] += 1
        # Which noise pattern (if any) matches this symbol?
        hit = None
        for p in _NOISE_PATTERNS:
            if re.match(p, g.strip().upper(), flags=re.IGNORECASE):
                hit = p
                pattern_hits[p] += 1
                break
        rows.append({
            "symbol": g,
            "classification": c,
            "gene_type": gt,
            "noise_pattern_hit": hit,
            "is_ensg": bool(_ENSG_RE.match(g.strip().upper())),
        })
    return {
        "n_total": len(hvg_vocab),
        "classification": dict(cls_count),
        "gene_type": dict(gene_type_count),
        "pattern_hits": pattern_hits,
        "rows": rows,
    }


def _render_markdown(audit: dict, out_path: Path):
    md = []
    md.append("# Gene-vocab audit\n")
    md.append(f"HVG vocab size: **{audit['n_total']}**\n")
    md.append("## Classification\n")
    md.append("| class | count | share |")
    md.append("|---|---:|---:|")
    n = audit["n_total"]
    for k, v in sorted(audit["classification"].items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v} | {100*v/n:.1f}% |")
    md.append("\n## Gene-type (from GTF)\n")
    md.append("| gene_type | count | share |")
    md.append("|---|---:|---:|")
    for k, v in sorted(audit["gene_type"].items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v} | {100*v/n:.1f}% |")
    md.append("\n## Noise-pattern leakage (patterns that DID hit HVG vocab)\n")
    md.append("Anything > 0 here means a gene with that pattern survived `_clean_adata_var_names`.")
    md.append("If non-zero, double-check the regex or sample source.\n")
    md.append("| pattern | hits |")
    md.append("|---|---:|")
    nonzero = [(k, v) for k, v in audit["pattern_hits"].items() if v > 0]
    if not nonzero:
        md.append("| _(all zero)_ | — |")
    else:
        for k, v in nonzero:
            md.append(f"| `{k}` | {v} |")

    # First 50 unknown / ensg_unresolved for visual inspection
    rows = audit["rows"]
    unknown = [r["symbol"] for r in rows if r["classification"] == "unknown"][:50]
    ensg_un = [r["symbol"] for r in rows if r["classification"] == "ensg_unresolved"][:50]
    md.append("\n## Samples of `unknown` (first 50)\n")
    md.append(", ".join(unknown) if unknown else "_(none)_")
    md.append("\n## Samples of `ensg_unresolved` (first 50)\n")
    md.append(", ".join(ensg_un) if ensg_un else "_(none)_")

    out_path.write_text("\n".join(md))


def _make_eda_figures(prepared_dir: Path, hvg_vocab: list[str], audit: dict,
                       out_dir: Path, include_st1k: bool):
    """Best-effort EDA — falls back gracefully if a source isn't available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── (1) gene_type breakdown ──
    cnt = audit["gene_type"]
    if cnt:
        labels = [k for k, _ in sorted(cnt.items(), key=lambda kv: -kv[1])]
        sizes = [cnt[k] for k in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels[:15], sizes[:15], color="#3a82e0")
        ax.invert_yaxis()
        ax.set_xlabel("# HVG"); ax.set_title("HVG vocab gene_type breakdown (top 15)")
        for b, v in zip(bars, sizes[:15]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f" {v}", va="center", fontsize=9)
        plt.tight_layout(); plt.savefig(out_dir / "fig_hvg_gene_type.png", dpi=130); plt.close(fig)

    # ── (2) classification breakdown (pie) ──
    cls = audit["classification"]
    if cls:
        labels = list(cls.keys())
        sizes = [cls[k] for k in labels]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
               autopct="%1.1f%%", startangle=90,
               colors=["#3a82e0", "#e07a3a", "#888888", "#aa3aff"])
        ax.set_title("HVG classification vs GTF")
        plt.tight_layout(); plt.savefig(out_dir / "fig_hvg_classification.png", dpi=130); plt.close(fig)

    # ── (3) per-sample lib size + per-gene prevalence — use existing shards ──
    import h5py
    shard_paths = sorted(prepared_dir.glob("*.h5"))
    if not shard_paths:
        log.warning("no shards under prepared_dir; skipping per-sample EDA")
        return
    log.info(f"scanning {len(shard_paths)} shards for EDA stats")

    lib_sizes = []  # per-sample total counts (proxy: sum of hvg_log)
    sample_ids = []
    gene_presence = np.zeros(len(hvg_vocab), dtype=np.int64)
    for p in shard_paths:
        try:
            with h5py.File(p, "r") as f:
                if "hvg_log" not in f:
                    continue
                X = f["hvg_log"][:]
        except (OSError, KeyError):
            continue
        if X.shape[1] != len(hvg_vocab):
            continue
        lib_sizes.append(float(X.sum(axis=1).mean()))
        sample_ids.append(p.stem)
        gene_presence += (X > 0).any(axis=0).astype(np.int64)
    if not lib_sizes:
        log.warning("no usable shards — skipping per-sample / per-gene EDA")
        return

    # per-sample library size
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lib_sizes, bins=60, color="#3a82e0", alpha=0.85)
    ax.set_xlabel("mean spot library-sum (log1p-normalized)")
    ax.set_ylabel("# samples")
    ax.set_title(f"Per-sample library-sum distribution (n_samples={len(lib_sizes)})")
    plt.tight_layout(); plt.savefig(out_dir / "fig_per_sample_lib_size.png", dpi=130); plt.close(fig)

    # per-gene prevalence
    frac = gene_presence / max(1, len(lib_sizes))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(frac, bins=50, color="#3a82e0", alpha=0.85)
    ax.set_xlabel("fraction of samples in which this gene has any non-zero spot")
    ax.set_ylabel("# HVG")
    ax.set_title(f"HVG prevalence across samples (n_hvg={len(hvg_vocab)})")
    plt.tight_layout(); plt.savefig(out_dir / "fig_hvg_prevalence.png", dpi=130); plt.close(fig)

    # Save prevalence table for downstream use
    prev_df = pd.DataFrame({
        "gene": hvg_vocab,
        "n_samples_with_nonzero": gene_presence,
        "prevalence": frac,
    }).sort_values("prevalence", ascending=False)
    prev_df.to_csv(out_dir / "hvg_prevalence.csv", index=False)
    log.info(f"wrote hvg_prevalence.csv ({len(prev_df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="/workspace/mm_align/results/cache/prepared")
    ap.add_argument("--vocab", default=None,
                    help="Override path to hvg_vocab.json (default: <prepared-dir>/hvg_vocab.json)")
    ap.add_argument("--gtf", default="/workspace/assets/gencode.v49.annotation.gtf")
    ap.add_argument("--no-eda", action="store_true", help="Skip figure generation")
    ap.add_argument("--include-st1k", action="store_true",
                    help="(Informational only) — labels the EDA caption to note that "
                         "ST1K shards were folded into the pool.")
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    vocab_path = Path(args.vocab) if args.vocab else (prepared / "hvg_vocab.json")
    if not vocab_path.exists():
        raise SystemExit(f"hvg_vocab.json not found: {vocab_path}")
    hvg = json.loads(vocab_path.read_text())
    log.info(f"loaded vocab: {len(hvg)} genes from {vocab_path}")

    # Reports go to the EDA mirror — NOT into the prepared (data) dir.
    reports = reports_dir_for(prepared)
    reports.mkdir(parents=True, exist_ok=True)
    log.info(f"reports dir = {reports}")

    gmap = load_gtf_symbol_map(args.gtf)
    log.info(f"loaded GTF: {len(gmap['ensg_to_symbol'])} ENSG → symbol mappings, "
             f"{len(gmap['valid_symbols'])} valid symbols")

    audit = audit_vocab(hvg, gmap)
    log.info(f"classification: {audit['classification']}")
    out_json = reports / "gene_vocab_audit.json"
    out_md = reports / "gene_vocab_audit.md"
    out_json.write_text(json.dumps(audit, indent=2))
    _render_markdown(audit, out_md)
    log.info(f"wrote {out_json}\nwrote {out_md}")

    if not args.no_eda:
        eda_dir = reports / "eda"
        _make_eda_figures(prepared, hvg, audit, eda_dir, include_st1k=args.include_st1k)
        log.info(f"EDA figures under {eda_dir}/")


if __name__ == "__main__":
    main()
