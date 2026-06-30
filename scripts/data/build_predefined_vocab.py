"""Build a curated, health-checked HVG vocab → `predefined_vocab.json`.

The raw `hvg_vocab.json` produced by `prepare_data.py` contains all 2048
top-variance HVG candidates, but many of those are:
  - ENSG-IDs that didn't resolve to HGNC symbols
  - SARS-CoV-2 reference fragments (`NC_045512.2_*`)
  - AMBIGUOUS multi-gene calls (`AMBIGUOUS[A+B]`)
  - Pseudogenes / non-coding hits passed our regex filter
  - Sample-specific singletons (only expressed in 1 sample)

This script:
  1. Reads `hvg_vocab.json` + `gene_vocab_audit.json` + `eda_v2/vocab_health.json`
  2. Drops genes by these rules (each rule logged separately):
       a. classification == "ensg_unresolved" | "unknown"
       b. starts with "NC_" / "AMBIGUOUS[" / contains "_ORF"
       c. coverage <= 1 sample (singleton)
       d. (optional) gene_type ∉ {protein_coding, lncRNA, miRNA}
  3. Writes:
       predefined_vocab.json       — flat list, kept HGNC symbols
       predefined_vocab_dict.json  — {token_str: id} with specials front-loaded
       predefined_vocab_report.md  — summary of which/why each gene was kept/dropped
       eda_v2/fig_vocab_filter_funnel.png  — visual funnel diagram

Usage:
  PYTHONPATH=src python scripts/data/build_predefined_vocab.py \
      --prepared-dir results/cache/prepared_expanded \
      --strict-gene-type    # also drop pseudogenes/snRNA/etc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for
log = get_logger("predef_vocab")


# Patterns to drop on top of the noise regex already in prepare_data.py
_DROP_PATTERNS = [
    r"^NC_",
    r"^AMBIGUOUS\[",
    r"_ORF\d",                   # SARS-CoV-2 ORFs
    r"^ENSG\d+",                  # unresolved ENSG
    r"^ENSMUSG\d+",               # mouse Ensembl (leaked from spatialcorpus)
    r"^MTRNR2L\d+",               # MT-pseudogenes that slipped MT regex
    r"^WHR1",                     # noise sequence
    r"^CTD-\d", r"^CTA-\d", r"^CTB-\d", r"^CTC-\d",
    r"^LL\d", r"^LLNL",
    r"^GS1-\d",
    r"^KB-\d",
    r"^LA16C-\d",
    r"^CH\d+-",
    r"^XX[A-Z]+\d",
]
_DROP_RE = re.compile("|".join(_DROP_PATTERNS), re.IGNORECASE)

# Default ALLOWED gene types when --strict-gene-type is on.
_ALLOWED_TYPES = {
    "protein_coding", "lncRNA", "miRNA",
    "IG_C_gene", "IG_V_gene", "IG_J_gene", "IG_D_gene",
    "TR_C_gene", "TR_V_gene", "TR_J_gene", "TR_D_gene",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--out-prefix", default="predefined_vocab",
                    help="Filename prefix under <prepared-dir>/")
    ap.add_argument("--strict-gene-type", action="store_true",
                    help="Drop pseudogenes/snRNA/snoRNA/etc; keep only protein_coding+lncRNA+IG/TR")
    ap.add_argument("--min-coverage", type=int, default=2,
                    help="Drop genes that appear in fewer than this many samples (default 2)")
    args = ap.parse_args()

    prep = Path(args.prepared_dir)
    reports = reports_dir_for(prep); reports.mkdir(parents=True, exist_ok=True)
    hvg = json.loads((prep / "hvg_vocab.json").read_text())
    n_in = len(hvg)
    log.info(f"input vocab: {n_in} genes from {prep}/hvg_vocab.json")
    log.info(f"reports dir = {reports}")

    # Load audit (classification, gene_type) — from EDA mirror.
    audit_path = reports / "gene_vocab_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        per_gene = {r["symbol"]: r for r in audit["rows"]}
    else:
        log.warning(f"no gene_vocab_audit.json at {audit_path} — "
                    f"run scripts/data/audit_gene_symbols.py first")
        per_gene = {}

    # Load coverage (prevalence per gene) from the EDA mirror.
    coverage_map = {}
    prev_csv_candidates = [
        reports / "eda_v2" / "hvg_prevalence.csv",
        reports / "eda" / "hvg_prevalence.csv",
    ]
    for prev_csv in prev_csv_candidates:
        if prev_csv.exists():
            import csv
            with open(prev_csv) as fh:
                r = csv.DictReader(fh)
                for row in r:
                    coverage_map[row["gene"]] = int(row["n_samples_with_nonzero"])
            log.info(f"loaded coverage map from {prev_csv} ({len(coverage_map)} genes)")
            break
    if not coverage_map:
        log.warning("hvg_prevalence.csv not found — skipping coverage filter")

    # ── Apply filters ──
    drops = {"classification": [], "noise_pattern": [], "gene_type": [], "low_coverage": []}
    kept = []
    for g in hvg:
        # 1. classification
        cls = per_gene.get(g, {}).get("classification", "unknown")
        if cls in ("unknown", "ensg_unresolved"):
            drops["classification"].append(g); continue
        # 2. noise pattern
        if _DROP_RE.search(g):
            drops["noise_pattern"].append(g); continue
        # 3. gene_type
        gt = per_gene.get(g, {}).get("gene_type", "unknown")
        if args.strict_gene_type and gt not in _ALLOWED_TYPES:
            drops["gene_type"].append(g); continue
        # 4. coverage
        cov = coverage_map.get(g, None)
        if cov is not None and cov < args.min_coverage:
            drops["low_coverage"].append(g); continue
        kept.append(g)

    n_kept = len(kept)
    log.info(f"kept {n_kept} / {n_in}  ({n_kept/n_in:.1%})")
    log.info(f"dropped — classification: {len(drops['classification'])}")
    log.info(f"        — noise pattern : {len(drops['noise_pattern'])}")
    log.info(f"        — gene_type     : {len(drops['gene_type'])}  (strict={args.strict_gene_type})")
    log.info(f"        — low coverage  : {len(drops['low_coverage'])}")

    # Write flat list + token dict — these are DATA the model reads, so they
    # live in `prep` (the data dir).
    (prep / f"{args.out_prefix}.json").write_text(json.dumps(kept))
    SPECIAL = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[UNK]": 3}
    full = dict(SPECIAL)
    for i, g in enumerate(kept):
        full[g] = i + len(SPECIAL)
    (prep / f"{args.out_prefix}_dict.json").write_text(json.dumps(full))

    # ── Report (markdown) ──
    md = []
    md.append("# Predefined Vocab — Filtering Report\n")
    md.append(f"Input: `hvg_vocab.json` ({n_in} genes)\n")
    md.append("## Filter funnel\n")
    md.append("| stage | dropped | remaining |")
    md.append("|---|---:|---:|")
    remaining = n_in
    for stage, removed in [
        ("classification (unknown / ensg_unresolved)", drops["classification"]),
        ("noise pattern (NC_ / AMBIGUOUS / pseudogene clones)", drops["noise_pattern"]),
        (f"gene_type ∉ {sorted(_ALLOWED_TYPES)}" if args.strict_gene_type
         else "gene_type (skipped — pass --strict-gene-type to enable)", drops["gene_type"]),
        (f"coverage < {args.min_coverage} samples", drops["low_coverage"]),
    ]:
        remaining -= len(removed)
        md.append(f"| {stage} | {len(removed)} | {remaining} |")
    md.append("")
    md.append(f"**Output**: `{args.out_prefix}.json` with **{n_kept} genes** "
              f"+ `{args.out_prefix}_dict.json` ({len(full)} entries = {len(SPECIAL)} specials + {n_kept} HVG).\n")
    md.append("## Sample dropped genes (first 20 per category)\n")
    for cat, vs in drops.items():
        md.append(f"### {cat}  (n={len(vs)})")
        md.append(", ".join(vs[:20]) if vs else "_(none)_")
        md.append("")
    # Report markdown → reports/EDA dir.
    (reports / f"{args.out_prefix}_report.md").write_text("\n".join(md))

    # ── Figure: filter funnel — also reports/EDA ──
    eda_dir = reports / "eda_v2"
    eda_dir.mkdir(parents=True, exist_ok=True)
    stages_disp = [
        ("HVG raw\n(top-2048 variance)", n_in),
        ("after classification\nfilter", n_in - len(drops["classification"])),
        ("after noise-pattern\nfilter", n_in - len(drops["classification"]) - len(drops["noise_pattern"])),
        ("after gene_type\nfilter", n_in - len(drops["classification"]) - len(drops["noise_pattern"]) - len(drops["gene_type"])),
        ("after coverage\nfilter (kept)", n_kept),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [s[0] for s in stages_disp]
    counts = [s[1] for s in stages_disp]
    cols = ["#888888", "#3a82e0", "#e07a3a", "#7e3aff", "#2a8f3a"]
    bars = ax.bar(range(len(stages_disp)), counts, color=cols, alpha=0.85)
    ax.set_xticks(range(len(stages_disp)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("# genes in vocab")
    ax.set_title(f"Predefined vocab filtering funnel — {n_in} → {n_kept} ({n_kept/n_in:.1%} retained)")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + n_in*0.01,
                f"{c}", ha="center", va="bottom", fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(eda_dir / "fig_vocab_filter_funnel.png", dpi=130); plt.close(fig)
    log.info(f"wrote DATA: {prep}/{args.out_prefix}.json + _dict.json")
    log.info(f"wrote REPORTS: {reports}/{args.out_prefix}_report.md + eda_v2/fig_vocab_filter_funnel.png")


if __name__ == "__main__":
    main()
