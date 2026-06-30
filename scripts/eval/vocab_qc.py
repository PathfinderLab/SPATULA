"""Comprehensive HVG-vocab quality check + visualization.

Designed to answer:
  1. What's the gene-type breakdown of the vocab? (protein_coding share)
  2. Did the noise/ambiguous filtering work? (filter quality)
  3. Per-sample AND per-spot prevalence — distributions + correlation
  4. Are well-known marker genes (TP53, SFRP1, MYC, EGFR, …) in the vocab?
     For missing ones — diagnose why (filter? not in any sample? GTF mismatch?)
  5. Disease-stratified expression of select markers (e.g., TP53 cancer vs normal)

Inputs:
  - <prepared_dir>/hvg_vocab.json   (raw HVG output)
  - <prepared_dir>/*.h5 shards       (for per-spot prevalence + expression scan)
  - <prepared_dir>/splits.json       (for source/sample listing)
  - <reports_dir>/gene_vocab_audit.json (gene_type + classification)
  - /data/hest/HEST_v1_1_0.csv       (disease_state, organ)

Outputs (under <reports_dir>/vocab_qc/):
  - vocab_qc.md                       (human-readable report)
  - fig_genetype_pie.png
  - fig_genetype_bar.png
  - fig_prevalence_scatter.png       (sample-prev × spot-prev per HVG)
  - fig_prevalence_hist.png          (marginal distributions)
  - fig_marker_panel_status.png      (which markers in/out + their prevalence)
  - fig_marker_disease_stratified.png (TP53/MMP9/etc cancer vs normal)
  - fig_filter_quality.png           (% protein_coding / lncRNA / ambiguous / etc)
  - marker_panel_status.csv          (full per-marker table)
  - missing_marker_diagnosis.csv     (why a known marker is missing)

Usage:
  PYTHONPATH=src python scripts/eval/vocab_qc.py \
      --prepared-dir results/cache/prepared_4k
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger, reports_dir_for
log = get_logger("vocab_qc")


# ─────────────────────────────────────────────────────────────────────────
# Curated marker gene panels (well-known biology)
# ─────────────────────────────────────────────────────────────────────────
MARKER_PANELS = {
    "Tumor suppressors":        ["TP53","TP63","TP73","RB1","PTEN","BRCA1","BRCA2",
                                  "APC","CDKN2A","CDKN1A","SMAD4","VHL","NF1","NF2",
                                  "MEN1","STK11"],
    "Oncogenes":                ["MYC","MYCN","KRAS","HRAS","NRAS","EGFR","ERBB2",
                                  "ALK","MET","BRAF","PIK3CA","AKT1","CCND1","BCL2",
                                  "MDM2","FOXO1"],
    "Wnt / antagonists":        ["SFRP1","SFRP2","SFRP3","SFRP4","SFRP5","DKK1","DKK2",
                                  "DKK3","WIF1","FRZB","WNT5A","WNT3A","CTNNB1","AXIN2",
                                  "LRP5","LRP6","TCF7"],
    "ECM / fibroblast":         ["COL1A1","COL1A2","COL3A1","COL5A1","FN1","VIM","ACTA2",
                                  "FAP","PDGFRA","PDGFRB","POSTN","S100A4","DCN","LUM"],
    "Epithelial":               ["EPCAM","KRT5","KRT8","KRT14","KRT18","KRT19","CDH1",
                                  "MUC1","CEACAM5"],
    "Immune T-cell":            ["CD3D","CD3E","CD3G","CD4","CD8A","CD8B","TRAC","TRBC1",
                                  "GZMA","GZMB","PRF1","IFNG","IL2RA"],
    "Immune B-cell":            ["MS4A1","CD79A","CD79B","IGHM","IGKC","JCHAIN","IGHG1"],
    "Macrophage":               ["CD68","CD163","MRC1","MSR1","C1QA","C1QB","C1QC",
                                  "MARCO","ITGAM","LYZ"],
    "Endothelial":              ["PECAM1","VWF","CDH5","CLDN5","ENG","KDR","FLT1","PROX1"],
    "Proliferation":            ["MKI67","PCNA","TOP2A","MCM2","MCM4","MCM5","CDK1",
                                  "CCNB1","AURKA"],
    "MMPs (invasion)":          ["MMP1","MMP2","MMP3","MMP7","MMP9","MMP14","TIMP1","TIMP2"],
    "Heat shock":               ["HSP90AA1","HSP90AB1","HSPA1A","HSPA1B","HSPA5","HSPA8",
                                  "DNAJB1","HSPB1"],
    "Stem cell / dev":          ["SOX2","NANOG","POU5F1","KLF4","MYC","OCT4"],
    "Cell death":               ["BAX","BAK1","BAD","CASP3","CASP8","CASP9","FAS","FASLG"],
    "Hypoxia / angiogenesis":   ["HIF1A","VEGFA","VEGFB","VEGFC","ANGPT1","ANGPT2"],
}


# ─────────────────────────────────────────────────────────────────────────
# Per-shard prevalence scan (single pass, two stats per gene)
# ─────────────────────────────────────────────────────────────────────────
def scan_prevalence(prepared: Path, hvg: list[str], train_ids: list[str]) -> pd.DataFrame:
    """One pass over all train shards.  Returns DataFrame per gene with:
        sample_prevalence  — fraction of samples where the gene has ≥1 nonzero spot
        spot_prevalence    — fraction of ALL train spots where the gene is nonzero
        nonzero_mean       — average value at nonzero positions (log1p scale)
        nonzero_max        — global max value at nonzero positions
    """
    n_g = len(hvg)
    sample_present = np.zeros(n_g, dtype=np.int64)
    nz_spots_per_gene = np.zeros(n_g, dtype=np.int64)
    nz_sum_per_gene = np.zeros(n_g, dtype=np.float64)
    nz_max_per_gene = np.zeros(n_g, dtype=np.float64)
    total_spots = 0
    samples_seen = 0

    from tqdm import tqdm
    for sid in tqdm(train_ids, desc="scan prevalence"):
        # find shard
        p = None
        for suf in ("", ".st1k", ".spatialcorpus"):
            cand = prepared / f"{sid}{suf}.h5"
            if cand.exists(): p = cand; break
        if p is None: continue
        try:
            with h5py.File(p, "r") as f:
                if "hvg_log" not in f: continue
                X = f["hvg_log"][:]
        except Exception:
            continue
        if X.shape[1] != n_g: continue
        samples_seen += 1
        total_spots += X.shape[0]
        gene_present_in_sample = (X > 0).any(axis=0)
        sample_present += gene_present_in_sample.astype(np.int64)
        nnz_per_gene = (X > 0).sum(axis=0)
        nz_spots_per_gene += nnz_per_gene
        nz_sum_per_gene += X.sum(axis=0)
        nz_max_per_gene = np.maximum(nz_max_per_gene, X.max(axis=0))

    sample_prev = sample_present / max(1, samples_seen)
    spot_prev = nz_spots_per_gene / max(1, total_spots)
    nz_mean = np.divide(nz_sum_per_gene, np.maximum(nz_spots_per_gene, 1))

    return pd.DataFrame({
        "gene": hvg,
        "sample_prevalence": sample_prev,
        "spot_prevalence": spot_prev,
        "n_samples_with_nonzero": sample_present,
        "n_nonzero_spots": nz_spots_per_gene,
        "nonzero_mean": nz_mean,
        "nonzero_max": nz_max_per_gene,
    })


# ─────────────────────────────────────────────────────────────────────────
# Disease-stratified expression for a few markers
# ─────────────────────────────────────────────────────────────────────────
def disease_stratified_expression(prepared: Path, hvg: list[str], train_ids: list[str],
                                    markers: list[str],
                                    hest_csv: str = "/data/hest/HEST_v1_1_0.csv",
                                    max_spots_per_sample: int = 200) -> pd.DataFrame:
    """For each marker, gather log1p value distribution across (disease_state, organ).
    Restricted to HEST samples (where disease_state is reliable)."""
    gene_idx = {g: i for i, g in enumerate(hvg)}
    miss = [m for m in markers if m not in gene_idx]
    markers = [m for m in markers if m in gene_idx]
    if miss:
        log.info(f"disease-stratified: skipping {len(miss)} markers not in vocab: {miss}")
    if not markers:
        return pd.DataFrame()
    midx = [gene_idx[m] for m in markers]

    meta = pd.read_csv(hest_csv).set_index("id")
    rng = np.random.default_rng(0)
    rows = []
    from tqdm import tqdm
    for sid in tqdm(train_ids, desc="disease scan"):
        if sid not in meta.index: continue
        p = prepared / f"{sid}.h5"
        if not p.exists(): continue
        try:
            with h5py.File(p, "r") as f:
                if "hvg_log" not in f: continue
                X = f["hvg_log"][:]
        except Exception:
            continue
        n_take = min(max_spots_per_sample, X.shape[0])
        sel = rng.choice(X.shape[0], n_take, replace=False)
        sub = X[sel][:, midx]
        ds = str(meta.loc[sid, "disease_state"]) if "disease_state" in meta.columns else "Unknown"
        organ = str(meta.loc[sid, "organ"]) if "organ" in meta.columns else "Unknown"
        for spot_row in sub:
            for m, v in zip(markers, spot_row):
                rows.append({"gene": m, "disease": ds, "organ": organ, "value": float(v)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────
def fig_genetype_breakdown(audit: dict, out_path: Path):
    """Pie + bar of gene_type composition of the vocab."""
    gt = audit.get("gene_type", {})
    if not gt: return
    items = sorted(gt.items(), key=lambda kv: -kv[1])
    labels, sizes = zip(*items)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1, 1.5]})
    # Pie of top-8 + Other
    top = items[:8]
    other = sum(v for _, v in items[8:])
    pie_lbls = [k for k, _ in top] + (["Other"] if other else [])
    pie_szs = [v for _, v in top] + ([other] if other else [])
    ax1.pie(pie_szs, labels=[f"{l} ({s})" for l, s in zip(pie_lbls, pie_szs)],
            autopct="%1.1f%%", startangle=90)
    ax1.set_title("HVG vocab — gene_type breakdown")

    # Bar of all (log-scale)
    cols = ["#3a82e0"] * len(labels)
    # Highlight protein_coding green, pseudogenes red
    for i, l in enumerate(labels):
        if "protein_coding" in l: cols[i] = "#2a8f3a"
        elif "pseudogene" in l: cols[i] = "#e07a3a"
        elif l == "unknown": cols[i] = "#888888"
    ax2.barh(range(len(labels)), sizes, color=cols)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("# HVG"); ax2.set_xscale("log")
    ax2.set_title("Full breakdown (green=protein_coding, orange=pseudogene)")
    for i, s in enumerate(sizes):
        ax2.text(s, i, f"  {s}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_filter_quality(audit: dict, out_path: Path):
    """% protein_coding vs ambiguous/pseudo/unknown — bar."""
    gt = audit.get("gene_type", {})
    cls = audit.get("classification", {})
    n_total = audit.get("n_total", sum(gt.values()))
    if n_total == 0: return

    cats = {
        "protein_coding": gt.get("protein_coding", 0),
        "lncRNA": gt.get("lncRNA", 0),
        "IG/TR (immunoglobulin)": sum(v for k, v in gt.items() if k.startswith(("IG_","TR_"))),
        "pseudogenes (any)": sum(v for k, v in gt.items() if "pseudogene" in k),
        "small RNA (mi/sn/sno/sca)": sum(v for k, v in gt.items() if k in ("miRNA","snRNA","snoRNA","scaRNA","misc_RNA","ribozyme")),
        "unknown (in GTF lookup)": gt.get("unknown", 0),
        "ENSG unresolved": cls.get("ensg_unresolved", 0),
        "classified unknown": cls.get("unknown", 0) - cls.get("noise", 0),
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = list(cats.keys()); vals = list(cats.values())
    colors = ["#2a8f3a","#3a82e0","#7e3aff","#e07a3a","#bfa83a","#888","#aa3aff","#666"]
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        pct = 100*v/n_total
        ax.text(b.get_width(), b.get_y()+b.get_height()/2,
                f"  {v}  ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel(f"# HVG (total = {n_total})")
    ax.set_title("Filter quality — composition of the HVG vocab\n"
                 "(higher protein_coding / lower pseudogene & unknown = healthier)")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_prevalence_scatter(prev_df: pd.DataFrame, out_path: Path):
    """sample-prevalence × spot-prevalence scatter, with marker callouts."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(prev_df["sample_prevalence"], prev_df["spot_prevalence"],
               s=4, alpha=0.3, color="#3a82e0")
    # diagonal reference
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
            label="y=x (equal sample/spot prevalence)")
    # Highlight extreme cases
    # high-sample low-spot = "sample-specific marker" (rare per spot, but specific samples have it)
    odd = prev_df[(prev_df["sample_prevalence"] > 0.5) & (prev_df["spot_prevalence"] < 0.05)]
    ax.scatter(odd["sample_prevalence"], odd["spot_prevalence"],
               s=12, color="#e07a3a", label=f"sample≥50% & spot<5%  (n={len(odd)})",
               edgecolors="black", linewidths=0.4)
    # Annotate a few known biology markers if present
    annot = ["TP53","SFRP1","MMP9","MYC","EGFR","KRT19","CD3D","COL1A1","MKI67","HIF1A"]
    for g in annot:
        sel = prev_df[prev_df["gene"] == g]
        if not sel.empty:
            r = sel.iloc[0]
            ax.annotate(g, (r["sample_prevalence"], r["spot_prevalence"]),
                        fontsize=8, color="black",
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Sample prevalence  (fraction of samples with any nonzero spot)")
    ax.set_ylabel("Spot prevalence  (fraction of all train spots that are nonzero)")
    ax.set_title("HVG prevalence — sample-level vs spot-level\n"
                 "below-diagonal = expressed in many samples but in few spots (focal markers)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_prevalence_hist(prev_df: pd.DataFrame, out_path: Path):
    """Marginal distributions of sample- and spot-prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(prev_df["sample_prevalence"], bins=40, color="#3a82e0", alpha=0.85)
    axes[0].set_title("Sample prevalence histogram")
    axes[0].set_xlabel("fraction of samples"); axes[0].set_ylabel("# HVG")
    axes[0].axvline(0.1, ls="--", color="red", alpha=0.5, label="rare <10%")
    axes[0].axvline(0.5, ls="--", color="green", alpha=0.5, label="common ≥50%")
    axes[0].legend()
    axes[1].hist(prev_df["spot_prevalence"], bins=80, color="#e07a3a", alpha=0.85)
    axes[1].set_title("Spot prevalence histogram  (log y)")
    axes[1].set_xlabel("fraction of spots"); axes[1].set_ylabel("# HVG (log)")
    axes[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def fig_marker_panel_status(prev_df: pd.DataFrame, vocab: set[str], out_path: Path,
                             marker_panels: dict[str, list[str]]):
    """For each curated panel, plot in-vocab vs missing + their prevalence."""
    rows = []
    for panel_name, members in marker_panels.items():
        for g in members:
            in_vocab = g in vocab
            sample_prev = float(prev_df[prev_df["gene"] == g]["sample_prevalence"].iloc[0]) if in_vocab else float("nan")
            spot_prev = float(prev_df[prev_df["gene"] == g]["spot_prevalence"].iloc[0]) if in_vocab else float("nan")
            rows.append({"panel": panel_name, "gene": g, "in_vocab": in_vocab,
                          "sample_prevalence": sample_prev, "spot_prevalence": spot_prev})
    df = pd.DataFrame(rows)
    # Coverage per panel
    panel_cov = df.groupby("panel")["in_vocab"].agg(["sum", "count"]).reset_index()
    panel_cov["coverage"] = panel_cov["sum"] / panel_cov["count"]
    panel_cov = panel_cov.sort_values("coverage")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1, 1.6]})
    # Coverage bar
    cols = ["#2a8f3a" if c >= 0.8 else "#e07a3a" if c >= 0.5 else "#aa3a3a"
            for c in panel_cov["coverage"]]
    axes[0].barh(panel_cov["panel"], panel_cov["coverage"], color=cols)
    for i, (_, r) in enumerate(panel_cov.iterrows()):
        axes[0].text(r["coverage"]+0.005, i, f"  {int(r['sum'])}/{int(r['count'])}", va="center", fontsize=9)
    axes[0].set_xlim(0, 1.1)
    axes[0].set_xlabel("fraction of panel in vocab")
    axes[0].set_title("Marker panel coverage")
    axes[0].axvline(0.8, ls=":", color="green", alpha=0.5)

    # Per-gene scatter (sample_prev × spot_prev, colored by panel)
    palette = plt.cm.tab20.colors
    panels = list(marker_panels.keys())
    for i, panel in enumerate(panels):
        sub = df[(df["panel"] == panel) & (df["in_vocab"])]
        axes[1].scatter(sub["sample_prevalence"], sub["spot_prevalence"],
                        label=panel, color=palette[i % 20], s=30, alpha=0.7,
                        edgecolors="black", linewidths=0.3)
        for _, r in sub.iterrows():
            axes[1].annotate(r["gene"], (r["sample_prevalence"], r["spot_prevalence"]),
                              fontsize=7, color=palette[i % 20], xytext=(3, 3),
                              textcoords="offset points")
    axes[1].set_xlabel("sample prevalence"); axes[1].set_ylabel("spot prevalence")
    axes[1].set_title("Marker genes in vocab — prevalence locations")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return df  # marker_panel_status table


def fig_marker_disease_stratified(strat_df: pd.DataFrame, out_path: Path):
    """Box plots of marker expression across disease_state for selected genes."""
    if strat_df.empty: return
    genes = strat_df["gene"].unique().tolist()
    # Group small categories
    keep_disease = ["Healthy", "Cancer", "Diseased", "Tumor"]
    strat_df = strat_df[strat_df["disease"].isin(keep_disease)].copy()
    if strat_df.empty: return
    n_g = len(genes); ncols = min(4, n_g); nrows = (n_g + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    for i, g in enumerate(genes):
        ax = axes[i // ncols][i % ncols]
        sub = strat_df[strat_df["gene"] == g]
        groups = [sub[sub["disease"] == d]["value"].values for d in keep_disease]
        # Only plot non-empty groups
        labels = [d for d, gr in zip(keep_disease, groups) if len(gr) > 0]
        groups = [gr for gr in groups if len(gr) > 0]
        if groups:
            bp = ax.boxplot(groups, tick_labels=labels, showfliers=False,
                            patch_artist=True)
            for patch, lbl in zip(bp["boxes"], labels):
                patch.set_facecolor({"Healthy": "#2a8f3a", "Cancer": "#aa3a3a",
                                       "Diseased": "#e07a3a", "Tumor": "#7e3aff"}.get(lbl, "gray"))
                patch.set_alpha(0.6)
        ax.set_title(g); ax.set_ylabel("log1p value")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    # Hide unused axes
    for j in range(i+1, nrows*ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Marker expression stratified by HEST disease_state\n"
                 "(per-spot log1p, sampled ≤200 spots/sample)", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Missing-marker diagnosis
# ─────────────────────────────────────────────────────────────────────────
def diagnose_missing_markers(missing: list[str], audit_path: Path,
                              gtf_map: dict, noise_re) -> pd.DataFrame:
    """For each missing marker, classify the reason."""
    rows = []
    for g in missing:
        # 1. caught by noise filter?
        noise_hit = bool(noise_re.match(g))
        # 2. valid HGNC per GTF?
        gt = gtf_map["symbol_to_gene_type"].get(g.upper(), "<not in GTF>")
        ensg = gtf_map["symbol_to_ensg"].get(g.upper(), "-")
        # 3. Conclusion
        if gt == "<not in GTF>":
            reason = "not in GTF (deprecated symbol / wrong name)"
        elif noise_hit and gt != "protein_coding":
            reason = "caught by noise regex (and not protein_coding rescue)"
        elif noise_hit and gt == "protein_coding":
            reason = "noise regex hit BUT protein_coding → should have been rescued; check prepare timing"
        else:
            reason = "passed filter but ranked OUTSIDE top-N HVG (low dispersion)"
        rows.append({"gene": g, "gene_type": gt, "ensg": ensg,
                       "noise_pattern_hit": noise_hit, "reason": reason})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", default="results/cache/prepared_4k")
    ap.add_argument("--out-dir", default=None,
                    help="default: <reports-dir-for-prepared>/vocab_qc/")
    ap.add_argument("--max-disease-spots", type=int, default=200,
                    help="cap spots/sample for disease-stratified scan")
    args = ap.parse_args()

    prep = Path(args.prepared_dir)
    reports = reports_dir_for(prep)
    out_dir = Path(args.out_dir or (reports / "vocab_qc"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"prepared = {prep}\nreports  = {reports}\nout_dir  = {out_dir}")

    # Load vocab + audit
    hvg = json.loads((prep / "hvg_vocab.json").read_text())
    vocab_set = set(hvg)
    log.info(f"vocab: {len(hvg)} genes")

    audit_path = reports / "gene_vocab_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    # Load GTF for missing-marker diagnosis
    from mm_align.data.gene_symbols import load_gtf_symbol_map
    gtf = load_gtf_symbol_map()
    import sys as _sys
    _sys.path.insert(0, "scripts")
    if "prepare_data" in _sys.modules: del _sys.modules["prepare_data"]
    from prepare_data import _NOISE_RE

    # Splits — for prevalence scan
    splits = json.loads((prep / "splits.json").read_text())
    train_ids = splits["train"]

    # ── Prevalence scan ──
    log.info(f"scanning prevalence over {len(train_ids)} train shards…")
    prev_df = scan_prevalence(prep, hvg, train_ids)
    prev_df.to_csv(out_dir / "prevalence_table.csv", index=False)
    log.info(f"saved prevalence_table.csv ({len(prev_df)} rows)")

    # ── Figures ──
    log.info("rendering figures…")
    if audit:
        fig_genetype_breakdown(audit, out_dir / "fig_genetype_pie.png")
        fig_filter_quality(audit, out_dir / "fig_filter_quality.png")
    fig_prevalence_scatter(prev_df, out_dir / "fig_prevalence_scatter.png")
    fig_prevalence_hist(prev_df, out_dir / "fig_prevalence_hist.png")
    marker_df = fig_marker_panel_status(prev_df, vocab_set,
                                          out_dir / "fig_marker_panel_status.png",
                                          MARKER_PANELS)
    marker_df.to_csv(out_dir / "marker_panel_status.csv", index=False)

    # ── Disease-stratified expression ──
    # Pick a few markers known to differ between cancer/normal
    disease_markers = ["TP53","MMP9","MKI67","CD3D","KRT19","SFRP1","SFRP2",
                        "MYC","EGFR","HIF1A","VEGFA","CTNNB1"]
    disease_markers = [m for m in disease_markers if m in vocab_set]
    if disease_markers:
        # Restrict to HEST samples (disease_state reliable there)
        hest_train = [s for s in train_ids if not (
            (prep / f"{s}.st1k.h5").exists()
            or (prep / f"{s}.spatialcorpus.h5").exists())]
        log.info(f"disease-stratified scan on {len(hest_train)} HEST train shards × {len(disease_markers)} markers")
        strat_df = disease_stratified_expression(prep, hvg, hest_train, disease_markers,
                                                  max_spots_per_sample=args.max_disease_spots)
        strat_df.to_csv(out_dir / "disease_stratified.csv", index=False)
        fig_marker_disease_stratified(strat_df, out_dir / "fig_marker_disease_stratified.png")

    # ── Missing-marker diagnosis ──
    all_markers = [g for genes in MARKER_PANELS.values() for g in genes]
    missing = sorted(set(all_markers) - vocab_set)
    diag = diagnose_missing_markers(missing, audit_path, gtf, _NOISE_RE)
    diag.to_csv(out_dir / "missing_marker_diagnosis.csv", index=False)
    log.info(f"missing markers: {len(missing)}; diagnosis → {out_dir/'missing_marker_diagnosis.csv'}")

    # ── Markdown summary ──
    md = []
    md.append(f"# Vocab QC Report\n")
    md.append(f"prepared: `{prep}`  ·  vocab size: **{len(hvg)}**\n")

    # Gene type
    gt = audit.get("gene_type", {}) if audit else {}
    if gt:
        n_pc = gt.get("protein_coding", 0)
        n_pseudo = sum(v for k, v in gt.items() if "pseudogene" in k)
        n_lncrna = gt.get("lncRNA", 0)
        n_unknown = gt.get("unknown", 0)
        md.append("## 1. Gene-type composition\n")
        md.append(f"- protein_coding : **{n_pc}** ({100*n_pc/len(hvg):.1f}%)")
        md.append(f"- lncRNA         : {n_lncrna} ({100*n_lncrna/len(hvg):.1f}%)")
        md.append(f"- pseudogenes    : {n_pseudo} ({100*n_pseudo/len(hvg):.1f}%)")
        md.append(f"- unknown (no GTF match): {n_unknown} ({100*n_unknown/len(hvg):.1f}%)\n")
        md.append("![](fig_genetype_pie.png)\n")
        md.append("![](fig_filter_quality.png)\n")

    # Prevalence summary
    md.append("## 2. Prevalence — sample-level vs spot-level\n")
    md.append(f"- median sample prevalence: {prev_df['sample_prevalence'].median():.3f}")
    md.append(f"- median spot   prevalence: {prev_df['spot_prevalence'].median():.4f}")
    md.append(f"- genes with sample_prev ≥ 0.5: {(prev_df['sample_prevalence']>=0.5).sum()}  "
              f"({100*(prev_df['sample_prevalence']>=0.5).mean():.1f}%)")
    md.append(f"- genes with spot_prev ≥ 0.1: {(prev_df['spot_prevalence']>=0.1).sum()}  "
              f"({100*(prev_df['spot_prevalence']>=0.1).mean():.1f}%)\n")
    md.append("![](fig_prevalence_scatter.png)\n")
    md.append("![](fig_prevalence_hist.png)\n")

    # Marker panel
    md.append("## 3. Marker-panel coverage\n")
    panel_cov = marker_df.groupby("panel")["in_vocab"].agg(["sum", "count"]).reset_index()
    panel_cov["coverage"] = panel_cov["sum"] / panel_cov["count"]
    md.append("| panel | in vocab | total | coverage |")
    md.append("|---|---:|---:|---:|")
    for _, r in panel_cov.sort_values("coverage").iterrows():
        md.append(f"| {r['panel']} | {int(r['sum'])} | {int(r['count'])} | {100*r['coverage']:.0f}% |")
    md.append("\n![](fig_marker_panel_status.png)\n")

    if disease_markers:
        md.append("## 4. Disease-stratified expression (HEST only)\n")
        md.append(f"Markers checked: {disease_markers}\n")
        md.append("![](fig_marker_disease_stratified.png)\n")

    # Missing diagnosis
    md.append("## 5. Missing-marker diagnosis\n")
    md.append(f"Out of {len(all_markers)} curated markers, **{len(missing)} are missing** from vocab.\n")
    if len(missing) > 0:
        md.append("\nFull table in `missing_marker_diagnosis.csv`.  Top reasons:\n")
        reasons = Counter(diag["reason"])
        md.append("| reason | count |")
        md.append("|---|---:|")
        for r, n in reasons.most_common():
            md.append(f"| {r} | {n} |")
        md.append("\nFirst 20 missing genes:\n")
        md.append("| gene | gene_type | noise_hit | reason |")
        md.append("|---|---|---|---|")
        for _, r in diag.head(20).iterrows():
            md.append(f"| {r['gene']} | {r['gene_type']} | {r['noise_pattern_hit']} | {r['reason']} |")

    (out_dir / "vocab_qc.md").write_text("\n".join(md))
    log.info(f"wrote vocab_qc.md")


if __name__ == "__main__":
    main()
