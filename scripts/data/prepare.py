"""Prepare paired HDF5 shards for mm_align training.

Per sample (e.g. INT1):
  - Load /data/hest/st/{sid}.h5ad       (Visium adata)
  - Load /data/hest/patches/{sid}.h5    (image patches + UNI features + barcodes)
  - Match barcodes (adata.obs_names <-> patches/barcode).
  - HVG selection across the whole TRAIN split (one shared vocab), then log-normalize.
  - Run novae zero-shot to get a 64-d transcriptomics latent per spot.
  - Write paired shard to {prepared_dir}/{sid}.h5 with:
        /barcode, /coords, /uni_feat, /novae_latent, /hvg_log, /gene_index

The HVG vocab is derived ONCE from the train split, then applied to all splits.
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import json

import h5py
import numpy as np
import scanpy as sc
import anndata as ad
import yaml
import re
from tqdm import tqdm

# Pre-mmap sklearn.neighbors at module load time.  When deferred to main()
# (after HVG-pass and gene_stats have run), the cumulative memory fragmentation
# makes the .so mapping fail with "failed to map segment from shared object"
# even though there's plenty of free RAM.  Top-level import = mapped before
# any large allocation happens.
from sklearn.neighbors import NearestNeighbors  # noqa: F401


# Special tokens — kept at the FRONT of the vocab (preprocess.py convention).
SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[UNK]": 3}
N_SPECIAL = len(SPECIAL_TOKENS)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import load_config, get_logger, set_seed, reports_dir_for
from mm_align.data.pairs import default_split, stratified_split
from mm_align.data.st1k_loader import (
    list_st1k_samples, read_st1k_sample, sample_source as _st1k_source,
    ST1K_ROOT,
)
from mm_align.data.spatialcorpus_loader import (
    list_spatialcorpus_samples, read_spatialcorpus_sample, SPATIALCORPUS_ROOT,
)
# Gene symbol normalisation + noise-pattern filtering — extracted to a proper
# module so `scripts/eval/validate_vocab.py` can use the exact same rules.
from mm_align.data.gene_cleaning import (
    clean_symbol as _clean_symbol,
    clean_adata_var_names as _clean_adata_var_names,
    _NOISE_PATTERNS, _NOISE_RE, _ENSG_RE,
    _PREFIX_RE, _VERSION_RE, _SUFFIX_RE,
)

warnings.filterwarnings("ignore")
log = get_logger("prepare_data")


# ── Unified sample loader ───────────────────────────────────────────────────
# Throughout this script we represent a sample as a (sample_id, source) tuple,
# where source ∈ {"hest", "st1k"}.  HEST samples come from /data/hest/st/{id}.h5ad
# and have paired image / novae features; ST1K samples come from
# /data/st1k/{ST|Visium|VisiumHD}/gene_exp/{id}_count.csv and only have raw
# counts (image / novae are written as zero placeholders so the existing
# PairedSpotDataset can read them without modification — Stage 1 doesn't
# touch the image side).

def _load_sample_adata(sid: str, source: str, hest_st_dir: Path,
                       *, qc_record: dict | None = None):
    """Source-agnostic loader → AnnData with raw counts + noise-cleaned var_names.

    Supported sources:
      "hest"          — /data/hest/st/<sid>.h5ad     (HGNC symbols)
      "st1k"          — /data/st1k/<tech>/gene_exp/<sid>_count.csv  (HGNC symbols)
      "spatialcorpus" — /data/spatialcorpus-110m/<sid>.h5ad         (ENSG IDs → resolve via GTF)

    If `qc_record` (dict) is provided, populated with:
      n_genes_raw       : original adata.n_vars before cleaning
      n_genes_clean     : after symbol-clean + dedup + noise drop
      n_genes_noise     : how many noise-pattern hits were dropped
      n_spots           : adata.n_obs
    """
    import anndata as ad
    if source == "hest":
        a = ad.read_h5ad(hest_st_dir / f"{sid}.h5ad")
    elif source == "st1k":
        a = read_st1k_sample(sid)
    elif source == "spatialcorpus":
        # Cap rows at 50K before dedup/densify so RAM stays bounded (some
        # spatialcorpus h5ads have 700K-1M cells × 20K+ genes; full
        # densification blows past 50 GB and stalls the pass).
        # Cap rows at 10K (was 50K) — even after subsample the dedup/densify
        # step can densify 50K × 25K genes ≈ 5 GiB and OOM during HVG-pass.
        # 10K spots still gives stable per-gene aggregate stats (the previous
        # iteration confirmed: gene_stats averaged 99K spots aggregated per
        # gene across 1268 samples, so per-sample cap matters less).
        a = read_spatialcorpus_sample(sid, max_spots=10_000)
    else:
        raise ValueError(f"unknown source: {source!r}")
    n_raw = int(a.n_vars)
    a, n_noise = _clean_adata_var_names(a)
    if qc_record is not None:
        qc_record["sample_id"] = sid
        qc_record["source"] = source
        qc_record["n_genes_raw"] = n_raw
        qc_record["n_genes_clean"] = int(a.n_vars)
        qc_record["n_genes_noise"] = int(n_noise)
        qc_record["n_spots"] = int(a.n_obs)
    return a


def assemble_sample_list(cfg: dict, *, hest_st_dir: Path, hest_patch_dir: Path,
                         whitelist_path: str | None) -> list[tuple[str, str]]:
    """Build the (sample_id, source) list according to cfg['sources'] toggles.
    Each enabled source contributes its own per-source listing function.
    Returns a deduplicated list — source order is HEST → ST1K → spatialcorpus.
    """
    sources_cfg = cfg.get("sources") or {"hest": True}
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # Whitelist (sample names without source suffix) — applies to all sources.
    whitelist: set[str] | None = None
    if whitelist_path:
        wl_path = Path(whitelist_path)
        if wl_path.exists():
            whitelist = {line.strip() for line in wl_path.read_text().splitlines() if line.strip()}

    if sources_cfg.get("hest", False):
        hids = list_available_samples(hest_st_dir, hest_patch_dir, whitelist_path=whitelist_path)
        log.info(f"sources.hest: {len(hids)} samples (after whitelist filter)")
        for sid in hids:
            key = (sid, "hest")
            if key not in seen:
                out.append(key); seen.add(key)

    if sources_cfg.get("st1k", False):
        sids = list_st1k_samples(species="human", whitelist=whitelist)
        log.info(f"sources.st1k: {len(sids)} human samples (after whitelist filter)")
        for sid in sids:
            key = (sid, "st1k")
            if key not in seen:
                out.append(key); seen.add(key)

    if sources_cfg.get("spatialcorpus", False):
        # strict=True opens each h5ad briefly to verify
        # `organism_ontology_term_id == NCBITaxon:9606`; safer than filename
        # heuristic, which was admitting mouse files like GSE118068.
        sids = list_spatialcorpus_samples(species="human", strict=True)
        # SpatialCorpus IDs are filenames (e.g. `10xgenomics_xenium_human_brain`)
        # not GSE_GSM tokens — the legacy whitelist won't match them.  Apply
        # whitelist only if it actually intersects; otherwise fall back to
        # "all spatialcorpus human samples".
        if whitelist is not None:
            inter = [s for s in sids if s in whitelist]
            if inter:
                sids = inter
            else:
                log.info("sources.spatialcorpus: whitelist has no matches — "
                         "using all human-tagged spatialcorpus samples.")
        log.info(f"sources.spatialcorpus: {len(sids)} human candidates "
                 f"(filename heuristic; species double-checked at read time)")
        for sid in sids:
            key = (sid, "spatialcorpus")
            if key not in seen:
                out.append(key); seen.add(key)

    return out


def list_available_samples(hest_st_dir: Path, hest_patch_dir: Path,
                           whitelist_path: str | Path | None = None) -> list[str]:
    st = {p.stem for p in hest_st_dir.glob("*.h5ad")}
    pt = {p.stem for p in hest_patch_dir.glob("*.h5")}
    avail = st & pt
    if whitelist_path:
        wl_path = Path(whitelist_path)
        if not wl_path.exists():
            raise FileNotFoundError(f"sample_whitelist not found: {wl_path}")
        wl = {line.strip() for line in wl_path.read_text().splitlines() if line.strip()}
        kept = avail & wl
        log.info(f"Whitelist {wl_path.name}: {len(wl)} ids; "
                 f"intersected with on-disk available ({len(avail)}) → {len(kept)} samples.")
        return sorted(kept)
    return sorted(avail)


def load_paired(sid: str, hest_st_dir: Path, hest_patch_dir: Path):
    """Load adata + patch H5; return (adata_paired, uni_feat_paired, coords_paired, barcodes_paired)."""
    adata = ad.read_h5ad(hest_st_dir / f"{sid}.h5ad")
    # SEAL/spatula-style cleaning: uppercase, version-strip, noise-pattern drop.
    adata, _ = _clean_adata_var_names(adata)

    with h5py.File(hest_patch_dir / f"{sid}.h5", "r") as f:
        bc = f["barcode"][:]
        coords = f["coords"][:]
        uni = f["uni_feat"][:]
    # decode bytes -> str
    barcodes = []
    for b in bc:
        if isinstance(b, np.ndarray):
            b = b[0] if b.shape else b.item()
        barcodes.append(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b))
    barcodes = np.asarray(barcodes)

    # match
    obs = adata.obs_names.astype(str).to_numpy()
    obs_idx = {b: i for i, b in enumerate(obs)}
    keep = [(i, obs_idx[b]) for i, b in enumerate(barcodes) if b in obs_idx]
    if not keep:
        return None, None, None, None, None
    img_idx, rna_idx = zip(*keep)
    img_idx = np.array(img_idx, dtype=np.int64); rna_idx = np.array(rna_idx)
    return adata[rna_idx].copy(), uni[img_idx], coords[img_idx], barcodes[img_idx], img_idx


def select_global_hvg(train_samples: list[tuple[str, str]] | list[str],
                      hest_st_dir: Path, n_hvg: int | None,
                      min_raw_counts: int = 3, n_bins: int = 20,
                      must_include: list[str] | None = None,
                      restrict_to_gene_types: list[str] | None = None,
                      heg_top_k: int = 0,
                      min_sample_prevalence: float = 0.0,
                      min_spot_prevalence: float = 0.0,
                      max_sample_prevalence: float = 1.01,
                      max_spot_prevalence: float = 1.01,
                      vocab_csv_out: Path | None = None) -> list[str]:
    """Pick top-N highly variable genes via a SINGLE STREAMING PASS.

    Equivalent intent to::

        pool = ad.concat([adatas], join="outer", fill_value=0)
        sc.pp.filter_genes(pool, min_counts=min_raw_counts)
        sc.pp.normalize_total(pool, target_sum=1e4); sc.pp.log1p(pool)
        sc.pp.highly_variable_genes(pool, flavor="seurat", n_top_genes=n_hvg)

    …but it never materializes a (#spots × #union-genes) matrix.
    For each sample we log-normalize independently and accumulate per-gene
    (count, sum, sum-of-squares, raw-count-sum), then run a Seurat-style
    bin-normalized-dispersion ranking once across the global gene set.
    """
    # Normalise to (sid, source) tuples for back-compat with old call sites
    # that pass a plain list of HEST ids.
    samples = [(s, "hest") if isinstance(s, str) else tuple(s) for s in train_samples]
    log.info(f"Computing global HVG vocab over {len(samples)} train samples (n_hvg={n_hvg}); streaming pass.")

    # Per-gene accumulators in shared dicts (gene name -> running stats).
    g_n: dict[str, int] = {}       # # spots aggregated for this gene
    g_sum: dict[str, float] = {}   # Σ log1p-normalized
    g_sq: dict[str, float] = {}    # Σ (log1p-normalized)^2
    g_raw: dict[str, float] = {}   # Σ raw counts (for the filter_genes-equivalent)
    g_nz_spots: dict[str, int] = {}    # # spots where x > 0   (for spot-prevalence)
    g_nz_samples: dict[str, int] = {}  # # samples where ≥1 spot has x > 0  (sample-prev)
    # Nonzero-value distribution stats (for global_median-style normalisers).
    # Sum and sum² restricted to positions where x > 0 — gives running estimates
    # of nonzero mean/std without a second pass.
    g_nz_sum: dict[str, float] = {}
    g_nz_sq:  dict[str, float] = {}
    n_samples_seen = 0

    skipped = 0
    import gc as _gc
    # Cap very large samples (e.g. Xenium cell-level files with 700K+ cells × 500
    # genes) to keep peak memory bounded.  Subsampled stats are still
    # representative since HVG selection is per-gene aggregate.
    MAX_SPOTS_HVG = 10_000     # was 50_000 — huge spatialcorpus shards stalled the pass
    per_sample_qc: list[dict] = []   # per-sample gene-processing stats (sample_qc.csv)
    for sid, source in tqdm(samples, desc="load train"):
        qc_row: dict = {}
        try:
            a = _load_sample_adata(sid, source, hest_st_dir, qc_record=qc_row)
        except (FileNotFoundError, ValueError, KeyError, OSError, MemoryError) as e:
            log.warning(f"HVG-pass: skip {source}:{sid} — {type(e).__name__}: {e}")
            skipped += 1
            qc_row.update({"sample_id": sid, "source": source, "error": type(e).__name__})
            per_sample_qc.append(qc_row)
            continue

        # Subsample huge samples in-place BEFORE materialising X to dense.
        if a.n_obs > MAX_SPOTS_HVG:
            rng_sub = np.random.default_rng(int(abs(hash(sid)) % (2**32)))
            sel = rng_sub.choice(a.n_obs, MAX_SPOTS_HVG, replace=False)
            a = a[sel].copy()

        try:
            # Raw counts → for filter_genes-equivalent
            X_raw = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
            X_raw = X_raw.astype(np.float32, copy=False)
            raw_sum = X_raw.sum(axis=0)            # (G_sample,)

            # Sample-wise log-normalize, then per-gene Σ and Σx²
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
            X = X.astype(np.float32, copy=False)
            n_spots = X.shape[0]
            ssum = X.sum(axis=0)
            ssq = (X * X).sum(axis=0)

            # Per-gene nonzero spot count + nonzero-value sums (for prevalence
            # filters and for nonzero distribution stats — used by
            # global_median normalisation and by the priority score).
            mask_nz = (X > 0)                              # (n_spots, G_sample)
            nz_per_gene = mask_nz.sum(axis=0)              # (G_sample,)
            X_nz_only = np.where(mask_nz, X, 0.0)
            nz_sum_per_gene = X_nz_only.sum(axis=0)        # Σ x at x>0
            nz_sq_per_gene  = (X_nz_only * X_nz_only).sum(axis=0)
            present_in_sample = (raw_sum > 0)              # (G_sample,) bool
            for j, gene in enumerate(a.var_names):
                g_n[gene] = g_n.get(gene, 0) + n_spots
                g_sum[gene] = g_sum.get(gene, 0.0) + float(ssum[j])
                g_sq[gene] = g_sq.get(gene, 0.0) + float(ssq[j])
                g_raw[gene] = g_raw.get(gene, 0.0) + float(raw_sum[j])
                g_nz_spots[gene] = g_nz_spots.get(gene, 0) + int(nz_per_gene[j])
                g_nz_sum[gene]   = g_nz_sum.get(gene, 0.0) + float(nz_sum_per_gene[j])
                g_nz_sq[gene]    = g_nz_sq.get(gene, 0.0)  + float(nz_sq_per_gene[j])
                if present_in_sample[j]:
                    g_nz_samples[gene] = g_nz_samples.get(gene, 0) + 1
            n_samples_seen += 1
            # Per-sample QC: how many genes are PC / lncRNA / pseudogene etc?
            try:
                from mm_align.data.gene_symbols import load_gtf_symbol_map
                _gmap = load_gtf_symbol_map()
                _tmap = _gmap["symbol_to_gene_type"]
                vn = [str(g).upper() for g in a.var_names]
                qc_row["n_genes_pc"] = sum(_tmap.get(g, "") == "protein_coding" for g in vn)
                qc_row["n_genes_lncrna"] = sum(_tmap.get(g, "") == "lncRNA" for g in vn)
                qc_row["n_genes_pseudo"] = sum("pseudogene" in _tmap.get(g, "") for g in vn)
                qc_row["n_genes_unknown_gt"] = sum(g not in _tmap for g in vn)
                qc_row["median_total_counts"] = float(np.median(X_raw.sum(axis=1)))
                qc_row["frac_zero_spots"] = float((X_raw.sum(axis=1) == 0).mean())
            except Exception as _e:
                qc_row["qc_err"] = str(_e)[:80]
            per_sample_qc.append(qc_row)
        except MemoryError as e:
            log.warning(f"HVG-pass: MemoryError on {source}:{sid} — skipping")
            skipped += 1
            qc_row["error"] = "MemoryError"
            per_sample_qc.append(qc_row)
        # Reliable per-iteration cleanup: drop reference + force GC so big
        # spatialcorpus arrays don't accumulate (AnnData backed reads can hold
        # several GB even with our subsample, since dense conversion happened).
        a = None
        _gc.collect()

    # Persist per-sample QC to disk next to hvg_vocab.json so vocab_qc can
    # consume it.  Caller writes to <prepared_dir>/sample_qc.csv via the side
    # effect on this list — we return it through a module attribute so the
    # caller can pick it up without changing the public signature.
    select_global_hvg._last_sample_qc = per_sample_qc

    # Build vectors
    genes = np.array(list(g_n.keys()))
    n_arr = np.array([g_n[g] for g in genes], dtype=np.float64)
    s_arr = np.array([g_sum[g] for g in genes], dtype=np.float64)
    sq_arr = np.array([g_sq[g] for g in genes], dtype=np.float64)
    raw_arr = np.array([g_raw[g] for g in genes], dtype=np.float64)
    nz_spots_arr   = np.array([g_nz_spots.get(g, 0)   for g in genes], dtype=np.float64)
    nz_samples_arr = np.array([g_nz_samples.get(g, 0) for g in genes], dtype=np.float64)
    nz_sum_arr     = np.array([g_nz_sum.get(g, 0.0)   for g in genes], dtype=np.float64)
    nz_sq_arr      = np.array([g_nz_sq.get(g, 0.0)    for g in genes], dtype=np.float64)

    def _apply_mask(m):
        nonlocal genes, n_arr, s_arr, sq_arr, raw_arr
        nonlocal nz_spots_arr, nz_samples_arr, nz_sum_arr, nz_sq_arr
        genes = genes[m]; n_arr = n_arr[m]; s_arr = s_arr[m]; sq_arr = sq_arr[m]
        raw_arr = raw_arr[m]; nz_spots_arr = nz_spots_arr[m]
        nz_samples_arr = nz_samples_arr[m]
        nz_sum_arr = nz_sum_arr[m]; nz_sq_arr = nz_sq_arr[m]

    # filter_genes(min_counts) equivalent
    keep = raw_arr >= min_raw_counts
    _apply_mask(keep)

    # Prevalence filters — NLP analogy:
    #   min_*  : drop hapax legomena (genes seen in ≤k samples/spots — pure noise)
    #   max_*  : drop "the/a/of" (genes expressed in ≥k samples/spots — too
    #            ubiquitous to discriminate cells; carry no localisation signal)
    if min_sample_prevalence > 0 and n_samples_seen > 0:
        sample_prev = nz_samples_arr / n_samples_seen
        keep_sp = sample_prev >= min_sample_prevalence
        before = len(genes)
        _apply_mask(keep_sp)
        log.info(f"select_global_hvg: min_sample_prevalence={min_sample_prevalence} "
                 f"dropped {before - len(genes)} of {before} (kept {len(genes)}).")
    if max_sample_prevalence < 1.0 and n_samples_seen > 0:
        sample_prev = nz_samples_arr / n_samples_seen
        keep_sp = sample_prev <= max_sample_prevalence
        before = len(genes)
        _apply_mask(keep_sp)
        log.info(f"select_global_hvg: max_sample_prevalence={max_sample_prevalence} "
                 f"dropped {before - len(genes)} ubiquitous genes (kept {len(genes)}).")
    if min_spot_prevalence > 0:
        spot_prev = nz_spots_arr / np.maximum(n_arr, 1)
        keep_sp = spot_prev >= min_spot_prevalence
        before = len(genes)
        _apply_mask(keep_sp)
        log.info(f"select_global_hvg: min_spot_prevalence={min_spot_prevalence} "
                 f"dropped {before - len(genes)} of {before} (kept {len(genes)}).")
    if max_spot_prevalence < 1.0:
        spot_prev = nz_spots_arr / np.maximum(n_arr, 1)
        keep_sp = spot_prev <= max_spot_prevalence
        before = len(genes)
        _apply_mask(keep_sp)
        log.info(f"select_global_hvg: max_spot_prevalence={max_spot_prevalence} "
                 f"dropped {before - len(genes)} ubiquitous genes (kept {len(genes)}).")

    # Optional: restrict candidate pool to a whitelist of GTF gene_types BEFORE
    # ranking.  Default for our spot vocab is ["protein_coding"] (+ IG/TR
    # rearrangement genes) — pseudogenes / lincRNAs / AMBIGUOUS aggregates
    # / BAC clones get pruned upstream now, but this is a final safety net
    # that *also* guarantees the candidate pool is interpretable.
    if restrict_to_gene_types:
        from mm_align.data.gene_symbols import load_gtf_symbol_map
        gmap = load_gtf_symbol_map()
        type_map = gmap["symbol_to_gene_type"]
        whitelist = set(restrict_to_gene_types)
        before = len(genes)
        keep_gt = np.array(
            [type_map.get(str(g).upper(), "unknown") in whitelist for g in genes],
            dtype=bool,
        )
        _apply_mask(keep_gt)
        log.info(
            f"select_global_hvg: gene_type restriction {sorted(whitelist)} dropped "
            f"{before - len(genes)} of {before} candidates; {len(genes)} remain."
        )

    mean = s_arr / np.maximum(n_arr, 1)
    var = np.maximum(sq_arr / np.maximum(n_arr, 1) - mean ** 2, 0.0)
    disp = var / np.maximum(mean, 1e-12)

    # Seurat-style: bin by mean, normalize dispersion within bin.
    log_mean = np.log1p(mean)
    valid = mean > 1e-8
    if valid.sum() < n_bins:
        # Degenerate (very few genes) — fall back to top-by-disp
        top = np.argsort(disp)[::-1][:n_hvg]
        return genes[top].tolist()

    edges = np.quantile(log_mean[valid], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9  # ensure last bin includes max
    bin_idx = np.digitize(log_mean, edges[1:-1])  # 0..n_bins-1
    norm_disp = np.zeros_like(disp)
    for b in range(n_bins):
        m = (bin_idx == b) & valid
        if m.sum() < 3:
            continue
        d = disp[m]
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826 + 1e-12
        norm_disp[m] = (d - med) / mad

    # Cap by normalized dispersion (descending).  n_hvg is a CAP, not a target:
    # when None (or larger than the candidate pool), keep every gene that
    # survived the prevalence/gene-type/min-count filters.  Ranking still runs
    # so vocab order is dispersion-descending — useful if a downstream consumer
    # wants to further truncate at training time.
    rank = np.argsort(norm_disp)[::-1]
    if n_hvg is None or n_hvg <= 0 or n_hvg >= len(rank):
        top = rank
        log.info(f"select_global_hvg: n_hvg cap inactive — keeping all "
                  f"{len(rank)} filter-survivors (dispersion-ordered).")
    else:
        top = rank[:n_hvg]
        log.info(f"select_global_hvg: capping at n_hvg={n_hvg} of {len(rank)} "
                  f"filter-survivors by normalized dispersion.")
    vocab = genes[top].tolist()

    # Optional HEG (Highly Expressed Genes) union — adds the top-K genes by
    # mean expression that didn't already qualify as HVG.  Rationale: HVG only
    # catches genes with above-average variability *for their expression band*;
    # housekeeping-like high-expressors (B2M, ACTB, GAPDH, HSPA8 etc.) are
    # stable but carry strong tissue signal and are ubiquitous on every spot.
    # Including them costs us a few hundred tokens but stabilises spot vectors.
    if heg_top_k > 0:
        # Use raw_arr (Σ raw counts across train) — robust to log-normalization.
        already = set(vocab)
        heg_order = np.argsort(raw_arr)[::-1]
        added = []
        for idx in heg_order:
            g = str(genes[idx])
            if g in already: continue
            vocab.append(g); already.add(g); added.append(g)
            if len(added) >= heg_top_k:
                break
        if added:
            log.info(f"select_global_hvg: HEG union added {len(added)} top-expressed "
                      f"genes (e.g. {added[:8]}).")

    # Force-include curated marker genes that are present in the candidate pool
    # but were ranked outside the top-N (e.g. TP53/HSP/CTNNB1 — widely expressed
    # but stable, hence low dispersion).  Append them to the vocab without
    # bumping anything out.  Final vocab size = n_hvg + |new must-include|.
    if must_include:
        gene_set = set(genes.tolist())
        already = set(vocab)
        added = []
        skipped = []
        for g in must_include:
            g = g.upper()
            if g in already: continue
            if g not in gene_set:
                skipped.append(g); continue       # not in candidate pool
            vocab.append(g); added.append(g); already.add(g)
        if added:
            log.info(f"select_global_hvg: force-included {len(added)} curated markers "
                      f"(e.g. {added[:8]}). Final vocab = {len(vocab)}.")
        if skipped:
            log.info(f"select_global_hvg: {len(skipped)} curated markers absent from "
                      f"candidate pool (filtered earlier or never expressed): {skipped[:8]}")

    # ── Emit vocab.csv — analytical view of the vocab with per-gene stats and
    # priority score.  CSV (vs JSON) lets us load into pandas + sort/filter
    # interactively for vocab discussion.
    if vocab_csv_out is not None:
        import pandas as _pd
        # Build the lookup: gene → its index in the filter-survivor arrays.
        idx_map = {str(g): i for i, g in enumerate(genes)}
        # Pre-compute aggregate stats over the survivors.
        mean_log = s_arr / np.maximum(n_arr, 1)
        var_log = np.maximum(sq_arr / np.maximum(n_arr, 1) - mean_log ** 2, 0.0)
        std_log = np.sqrt(var_log)
        nz_mean = nz_sum_arr / np.maximum(nz_spots_arr, 1)
        nz_var = np.maximum(nz_sq_arr / np.maximum(nz_spots_arr, 1) - nz_mean ** 2, 0.0)
        nz_std = np.sqrt(nz_var)
        # GTF metadata for every gene in the final vocab.
        try:
            from mm_align.data.gene_symbols import load_gtf_symbol_map
            _gmap = load_gtf_symbol_map()
            _tmap = _gmap["symbol_to_gene_type"]
        except Exception:
            _tmap = {}
        must_set = {g.upper() for g in (must_include or [])}
        rows = []
        for rank_idx, g in enumerate(vocab):
            i = idx_map.get(g)
            if i is None:        # must_include or HEG outside survivor pool
                rows.append({
                    "gene": g, "rank": rank_idx,
                    "gene_type": _tmap.get(g.upper(), "unknown"),
                    "must_include": g.upper() in must_set,
                    "sample_prev": np.nan, "spot_prev": np.nan,
                    "n_spots_seen": 0, "n_samples_present": 0,
                    "mean_log1p": np.nan, "std_log1p": np.nan,
                    "nonzero_mean_log1p": np.nan, "nonzero_std_log1p": np.nan,
                    "raw_total": np.nan, "norm_dispersion": np.nan,
                    "in_filter_pool": False,
                })
                continue
            rows.append({
                "gene": g, "rank": rank_idx,
                "gene_type": _tmap.get(g.upper(), "unknown"),
                "must_include": g.upper() in must_set,
                "sample_prev": float(nz_samples_arr[i] / max(n_samples_seen, 1)),
                "spot_prev":   float(nz_spots_arr[i]   / max(n_arr[i], 1)),
                "n_spots_seen":      int(n_arr[i]),
                "n_samples_present": int(nz_samples_arr[i]),
                "mean_log1p":         float(mean_log[i]),
                "std_log1p":          float(std_log[i]),
                "nonzero_mean_log1p": float(nz_mean[i]),
                "nonzero_std_log1p":  float(nz_std[i]),
                "raw_total":          float(raw_arr[i]),
                "norm_dispersion":    float(norm_disp[i]),
                "in_filter_pool":     True,
            })
        df = _pd.DataFrame(rows)
        # Priority score: must_include first, then by normalized dispersion
        # within the filter pool, with non-pool entries (HEG-only) at the tail.
        df["priority"] = df["must_include"].astype(int) * 10 + df["in_filter_pool"].astype(int) * 5 \
                          - df["rank"] * 1e-4   # tie-break by original rank
        df = df.sort_values(["priority", "rank"], ascending=[False, True]).reset_index(drop=True)
        df["priority_rank"] = np.arange(len(df))
        vocab_csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(vocab_csv_out, index=False)
        log.info(f"Saved vocab metadata CSV ({len(df)} rows) → {vocab_csv_out}")
        # Stash the dataframe so the caller can also build figures from it.
        select_global_hvg._last_vocab_df = df

    return vocab


def compute_global_gene_stats(train_samples: list[tuple[str, str]] | list[str],
                               hest_st_dir: Path,
                               hvg_vocab: list[str], out_path: Path,
                               *, max_spots_per_sample: int = 100,
                               seed: int = 0) -> dict:
    """Compute per-gene GLOBAL statistics over the train pool, after the same
    log-normalization the shards use.  We subsample ~100 spots/sample to bound
    memory (~ 100 × 500 samples × 2048 genes × 4B ≈ 400 MB).

    Saves an .npz with:
      mean[n_hvg], std[n_hvg], median[n_hvg], mad[n_hvg], hvg_vocab[n_hvg]

    Downstream consumers:
      - dataset __getitem__ → optional runtime `(x - median) / (mad + eps)` per gene
      - objectives.gene_losses → already uses {mean, std} for standardized_mse
    """
    rng = np.random.default_rng(seed)
    n_g = len(hvg_vocab)
    hvg_idx = {g: i for i, g in enumerate(hvg_vocab)}

    # ── STREAMING gene_stats — never materialise a giant pooled matrix.
    # We update per-gene moments (sum, sq, n) incrementally and keep a small
    # per-gene reservoir (up to RESERVOIR_PER_GENE nonzero samples) to compute
    # median/MAD/nonzero_median at the end without holding the full pool.
    RESERVOIR_PER_GENE = 1024
    SAMPLE_ALL  = np.zeros(n_g, dtype=np.float64)   # Σ x  (incl. zeros)
    SAMPLE_SQ   = np.zeros(n_g, dtype=np.float64)   # Σ x² (incl. zeros)
    SAMPLE_N    = np.zeros(n_g, dtype=np.int64)     # total spots aggregated
    NZ_SUM      = np.zeros(n_g, dtype=np.float64)
    NZ_SQ       = np.zeros(n_g, dtype=np.float64)
    NZ_N        = np.zeros(n_g, dtype=np.int64)
    # reservoir of nonzero values per gene (list-of-list of np.ndarray, capped)
    reservoir = [[] for _ in range(n_g)]
    reservoir_used = np.zeros(n_g, dtype=np.int32)

    samples = [(s, "hest") if isinstance(s, str) else tuple(s) for s in train_samples]
    log.info(f"Computing global gene stats over {len(samples)} train samples "
             f"(≤ {max_spots_per_sample} spots/sample, n_hvg={n_g}); streaming pass "
             f"with per-gene reservoir of {RESERVOIR_PER_GENE} for median/MAD.")
    import gc as _gc
    MAX_SPOTS_PRE_INDEX = 5_000
    for sid, source in tqdm(samples, desc="gene_stats"):
        try:
            a = _load_sample_adata(sid, source, hest_st_dir)
        except (FileNotFoundError, ValueError, KeyError, OSError, MemoryError) as e:
            log.warning(f"gene_stats: skip {source}:{sid} — {type(e).__name__}: {e}")
            continue
        try:
            if a.n_obs > MAX_SPOTS_PRE_INDEX:
                rng_sub = np.random.default_rng(int(abs(hash(sid)) % (2**32)))
                sel_pre = rng_sub.choice(a.n_obs, MAX_SPOTS_PRE_INDEX, replace=False)
                a = a[sel_pre].copy()
        except MemoryError as e:
            log.warning(f"gene_stats: skip {source}:{sid} (subsample copy OOM) — {e}")
            a = None; _gc.collect(); continue

        try:
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
            X = X.astype(np.float32, copy=False)
            n_take = min(max_spots_per_sample, X.shape[0])
            if X.shape[0] > n_take:
                sel = rng.choice(X.shape[0], n_take, replace=False)
                X = X[sel]
            # Per-column update on the vocab dimensions only — no giant `mat`.
            sample_var_names = a.var_names.astype(str)
            for j, g in enumerate(sample_var_names):
                gi = hvg_idx.get(g)
                if gi is None: continue
                col = X[:, j]
                SAMPLE_N[gi]  += col.shape[0]
                SAMPLE_ALL[gi] += float(col.sum())
                SAMPLE_SQ[gi]  += float((col * col).sum())
                nz = col[col > 0]
                if nz.size:
                    NZ_N[gi]  += int(nz.size)
                    NZ_SUM[gi] += float(nz.sum())
                    NZ_SQ[gi]  += float((nz * nz).sum())
                    # Reservoir sampling for median/MAD.
                    space = RESERVOIR_PER_GENE - reservoir_used[gi]
                    if space > 0:
                        take = min(space, nz.size)
                        reservoir[gi].append(nz[:take].astype(np.float32))
                        reservoir_used[gi] += take
        except MemoryError:
            log.warning(f"gene_stats: MemoryError on {source}:{sid} — skipping")
        a = None; X = None
        _gc.collect()

    log.info(f"gene_stats streaming pass done.  total spots aggregated (≈ first gene) "
             f"= {int(SAMPLE_N.max())}, mean = {int(SAMPLE_N.mean())}")

    mean = (SAMPLE_ALL / np.maximum(SAMPLE_N, 1)).astype(np.float32)
    var_all = np.maximum(SAMPLE_SQ / np.maximum(SAMPLE_N, 1) - mean.astype(np.float64) ** 2, 0.0)
    std = np.sqrt(var_all).astype(np.float32)
    nonzero_mean = (NZ_SUM / np.maximum(NZ_N, 1)).astype(np.float32)
    nz_var = np.maximum(NZ_SQ / np.maximum(NZ_N, 1) - nonzero_mean.astype(np.float64) ** 2, 0.0)
    nonzero_std = np.sqrt(nz_var).astype(np.float32)
    nonzero_count = NZ_N.copy()

    median = np.zeros(n_g, dtype=np.float32)        # zero-inflated; mostly 0
    mad = np.zeros(n_g, dtype=np.float32)
    nonzero_median = np.zeros(n_g, dtype=np.float32)
    for g in range(n_g):
        chunks = reservoir[g]
        if not chunks:
            continue
        nz = np.concatenate(chunks)
        nonzero_median[g] = float(np.median(nz))
        # Approx median/mad over zero-inflated values via observed nonzero
        # share — for the legacy global_robust_z path.  On ST data this is
        # almost always zero (since most spots are zero), but we keep the
        # field for completeness.
        nz_share = NZ_N[g] / max(SAMPLE_N[g], 1)
        if nz_share > 0.5:
            median[g] = nonzero_median[g] * (2 * nz_share - 1)   # crude lerp
            mad[g] = float(np.median(np.abs(nz - nonzero_median[g])) * 1.4826)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             mean=mean, std=std, median=median, mad=mad,
             nonzero_mean=nonzero_mean, nonzero_std=nonzero_std,
             nonzero_median=nonzero_median,
             nonzero_count=nonzero_count,
             hvg_vocab=np.array(hvg_vocab))
    log.info(f"wrote {out_path}")
    log.info(f"  median ∈ [{median.min():.3f}, {median.max():.3f}]  "
             f"(zero_count={(median==0).sum()})")
    log.info(f"  nonzero_mean ∈ [{nonzero_mean.min():.3f}, {nonzero_mean.max():.3f}]  "
             f"(zero_count={(nonzero_mean==0).sum()})")
    log.info(f"  nonzero_std  ∈ [{nonzero_std.min():.3f}, {nonzero_std.max():.3f}]")
    return {"mean": mean, "std": std, "median": median, "mad": mad,
            "nonzero_mean": nonzero_mean, "nonzero_std": nonzero_std,
            "nonzero_count": nonzero_count}


def compute_novae_latent(adata, radius_px: float):
    """Run novae zero-shot on one Visium adata. Returns (n_spots, 64) float32."""
    import novae
    novae.spatial_neighbors(adata, radius=radius_px)
    model = compute_novae_latent._model
    model.compute_representations(adata, zero_shot=True, num_workers=0)
    z = adata.obsm["novae_latent"].astype(np.float32)
    return z


def _maybe_load_novae():
    if not hasattr(compute_novae_latent, "_model"):
        import novae
        compute_novae_latent._model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
        log.info("Loaded novae model.")
    return compute_novae_latent._model


def process_sample(sid: str, hest_st_dir: Path, hest_patch_dir: Path,
                   hvg_vocab: list[str], radius_px: float, out_dir: Path,
                   *, source: str = "hest", skip_novae: bool = False) -> dict:
    """Build a shard for one sample.

    HEST samples produce full shards (image + novae + coords + hvg).
    ST1K / spatialcorpus samples produce gene-only shards: image/novae/coords
    are written as zero placeholders so the existing PairedSpotDataset reads
    them without code changes (Stage 1 ignores the image side).
    """
    # Source-specific shard filename so HEST/ST1K/spatialcorpus IDs can't
    # collide and so downstream code can identify the source from the path.
    suffix = "" if source == "hest" else f".{source}"
    out_path = out_dir / f"{sid}{suffix}.h5"
    if out_path.exists():
        return {"sample_id": sid, "source": source, "n_spots": -1, "status": "skipped"}

    if source == "hest":
        pair = load_paired(sid, hest_st_dir, hest_patch_dir)
        if pair[0] is None:
            return {"sample_id": sid, "source": source, "n_spots": 0, "status": "no_match"}
        adata_p, uni_feat, coords, barcodes, patch_idx = pair
    else:
        # ST1K / spatialcorpus — gene-only sample, no image / coords.
        try:
            adata_p = _load_sample_adata(sid, source, hest_st_dir)
        except (FileNotFoundError, ValueError) as e:
            return {"sample_id": sid, "source": source, "n_spots": 0, "status": f"skip:{type(e).__name__}"}
        n = adata_p.n_obs
        uni_feat = np.zeros((n, 1536), dtype=np.float32)
        # Spatial coords:
        # 1. Loaders (st1k_loader, spatialcorpus_loader) attach obsm['spatial']
        #    when raw data carries spot coordinates.  This is the authoritative path.
        # 2. Fall back to obs.x_centroid/y_centroid for legacy spatialcorpus h5ads
        #    that only have flattened obs columns.
        # 3. As a last resort emit zeros — but the Stage-1.5 trainer treats
        #    zero-coord shards as placeholders and drops them, so this means
        #    "no spatial signal available for this sample".
        if "spatial" in (adata_p.obsm.keys() if hasattr(adata_p.obsm, "keys") else []):
            coords = np.asarray(adata_p.obsm["spatial"], dtype=np.float32)
        elif {"x_centroid", "y_centroid"}.issubset(adata_p.obs.columns):
            coords = adata_p.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
        elif {"x", "y"}.issubset(adata_p.obs.columns):
            coords = adata_p.obs[["x", "y"]].to_numpy(dtype=np.float32)
        else:
            coords = np.zeros((n, 2), dtype=np.float32)
        barcodes = np.array(adata_p.obs.index.astype(str), dtype=object)
        patch_idx = np.arange(n, dtype=np.int32)

    # Spot cap for shard write — huge spatialcorpus/Xenium files (700K+ cells ×
    # 20K+ genes) need ≥40 GiB just to densify.  Cap to keep peak memory bounded
    # and to avoid OOM cascades that kill *other* samples' .so loading too.
    MAX_SPOTS_SHARD = 100_000
    if adata_p.n_obs > MAX_SPOTS_SHARD and source != "hest":
        rng_sub = np.random.default_rng(int(abs(hash(sid)) % (2**32)))
        sel = rng_sub.choice(adata_p.n_obs, MAX_SPOTS_SHARD, replace=False)
        adata_p = adata_p[sel].copy()
        # Resync the auxiliary arrays we built above.
        coords = coords[sel]
        uni_feat = uni_feat[sel]
        barcodes = barcodes[sel]
        patch_idx = patch_idx[sel] if source != "hest" else patch_idx

    # log-normalize, then index to the HVG vocab; missing genes -> 0 column.
    sc.pp.normalize_total(adata_p, target_sum=1e4)
    sc.pp.log1p(adata_p)
    hvg_idx_in_var = {g: i for i, g in enumerate(adata_p.var_names.astype(str))}
    X = adata_p.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    hvg_mat = np.zeros((adata_p.n_obs, len(hvg_vocab)), dtype=np.float32)
    for j, g in enumerate(hvg_vocab):
        i = hvg_idx_in_var.get(g)
        if i is not None:
            hvg_mat[:, j] = X[:, i]
    del X  # release the dense matrix immediately

    # ── Novae latent + spatial KNN — HEST only, unless --skip-novae ─
    # ST1K / spatialcorpus shards: novae=zeros (no spatial signal there).
    # Stage 1 (gene-encoder pretraining) doesn't read novae anyway, so we
    # skip the expensive Novae model call when --skip-novae is on.
    if source == "hest" and not skip_novae:
        try:
            _maybe_load_novae()
            novae_z = compute_novae_latent(adata_p, radius_px=radius_px)
            if novae_z.shape[0] != adata_p.n_obs:
                log.warning(f"{sid}: novae returned {novae_z.shape[0]} spots, expected {adata_p.n_obs}; padding zeros.")
                tmp = np.zeros((adata_p.n_obs, novae_z.shape[1]), dtype=np.float32)
                tmp[: novae_z.shape[0]] = novae_z
                novae_z = tmp
        except Exception as e:
            log.warning(f"{sid}: novae failed ({e}); writing zeros.")
            novae_z = np.zeros((adata_p.n_obs, 64), dtype=np.float32)
    else:
        novae_z = np.zeros((adata_p.n_obs, 64), dtype=np.float32)

    # Pre-build KNN cache alongside the shard (only when coords are real).
    knn_path = out_path.with_suffix(".knn.npy")
    have_coords = bool((coords ** 2).sum() > 0) and coords.shape[0] >= 2
    if not knn_path.exists() and have_coords:
        from sklearn.neighbors import NearestNeighbors
        k = min(9, coords.shape[0])
        knn = NearestNeighbors(n_neighbors=k).fit(coords)
        _, idx = knn.kneighbors(coords)
        nn_idx = idx[:, 1:].astype(np.int32)
        tmp = knn_path.with_name(knn_path.stem + ".tmp.npy")
        np.save(tmp, nn_idx)
        tmp.replace(knn_path)

    # Write shard
    with h5py.File(out_path, "w") as f:
        f.create_dataset("barcode", data=np.array([b.encode("utf-8") for b in barcodes], dtype="S"))
        f.create_dataset("coords", data=coords.astype(np.float32))
        f.create_dataset("uni_feat", data=uni_feat.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("patch_idx", data=patch_idx.astype(np.int32))
        f.create_dataset("novae_latent", data=novae_z.astype(np.float32))
        f.create_dataset("hvg_log", data=hvg_mat, compression="gzip", compression_opts=4)
        f.attrs["sample_id"] = sid
        f.attrs["source"] = source
        f.attrs["n_spots"] = adata_p.n_obs
        f.attrs["uni_feat_dim"] = uni_feat.shape[1]
        f.attrs["novae_dim"] = novae_z.shape[1]
        f.attrs["hvg_dim"] = len(hvg_vocab)

    return {"sample_id": sid, "source": source,
            "n_spots": int(adata_p.n_obs), "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1/data.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate sample list (smoke test).")
    ap.add_argument("--samples", nargs="*", default=None,
                    help="Explicit sample-id list (overrides discovery). "
                         "Source is inferred from an existing shard in --out-dir "
                         "(<sid>.h5 / <sid>.st1k.h5 / <sid>.spatialcorpus.h5).  "
                         "Use --samples-source to force a single source.")
    ap.add_argument("--samples-source", default=None,
                    choices=(None, "hest", "st1k", "spatialcorpus"),
                    help="Force all --samples to this source.  Useful for "
                         "partial re-prep when shards don't yet exist.")
    ap.add_argument("--rebuild_vocab", action="store_true",
                    help="Recompute HVG vocab even if cached.")
    ap.add_argument("--no_whitelist", action="store_true",
                    help="Ignore configs/stage1/data.yaml:sample_whitelist (use all on-disk samples).")
    ap.add_argument("--stratify", action="store_true",
                    help="Use organ-stratified 8:1:1 split (loads HEST metadata).")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--hest_csv", default="/data/hest/HEST_v1_1_0.csv")
    ap.add_argument("--skip-novae", action="store_true",
                    help="Skip Novae embedding for HEST samples — fills novae_latent "
                         "with zeros.  Use for Stage-1 (gene-encoder only) runs where "
                         "the image side is frozen and novae is ignored.  Cuts HEST "
                         "prep time from ~10 min/sample to ~10 s/sample.")
    ap.add_argument("--out-dir", default=None,
                    help="Override cfg.data.prepared_dir — useful for building an "
                         "expanded Stage-1 corpus alongside the existing prepared dir.")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config([args.config])["data"]
    out_dir = Path(args.out_dir or cfg["prepared_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"out_dir = {out_dir}  (skip_novae={args.skip_novae})")
    hest_st = Path(cfg["hest_st_dir"])
    hest_patch = Path(cfg["hest_patch_dir"])

    whitelist = cfg.get("sample_whitelist")
    if args.no_whitelist:
        whitelist = None

    # Build the (sid, source) sample list from cfg["sources"] toggles.
    # Falls back to HEST-only if no `sources` block exists (back-compat).
    if args.samples:
        # Source resolution priority:
        #   1. --samples-source if given → all samples that source
        #   2. existing shard suffix in out_dir → infer per-sample
        #   3. fall back to "hest"
        samples_with_src: list[tuple[str, str]] = []
        for s in args.samples:
            if args.samples_source:
                samples_with_src.append((s, args.samples_source))
                continue
            resolved = None
            for suf, src in (("", "hest"), (".st1k", "st1k"),
                              (".spatialcorpus", "spatialcorpus")):
                if (out_dir / f"{s}{suf}.h5").exists():
                    resolved = src; break
            samples_with_src.append((s, resolved or "hest"))
    else:
        samples_with_src = assemble_sample_list(
            cfg, hest_st_dir=hest_st, hest_patch_dir=hest_patch,
            whitelist_path=whitelist,
        )
    if args.limit:
        samples_with_src = samples_with_src[: args.limit]
    # Track per-source counts for the user
    from collections import Counter
    src_counts = Counter(src for _, src in samples_with_src)
    log.info(f"Using {len(samples_with_src)} samples "
             + (f"(whitelist={whitelist})" if whitelist else "(no whitelist)")
             + f" — by source: {dict(src_counts)}")

    # Compute (or load) global HVG vocab from "train" split.
    # Splits operate on the bare sample-id list (back-compat with default/stratified_split);
    # we re-attach sources after splitting via the original mapping.
    sid_to_src = {sid: src for sid, src in samples_with_src}
    ids = [sid for sid, _ in samples_with_src]
    if args.stratify:
        import pandas as pd
        df = pd.read_csv(args.hest_csv)
        id_to_organ = dict(zip(df["id"].astype(str), df["organ"].astype(str)))
        # ST1K / spatialcorpus samples won't have organ in HEST CSV — fallback to "Unknown".
        for sid in ids:
            id_to_organ.setdefault(sid, "Unknown")
        splits = stratified_split(ids, id_to_organ,
                                  val_frac=args.val_frac, test_frac=args.test_frac,
                                  seed=args.seed)
        log.info(f"Stratified split (val={args.val_frac}, test={args.test_frac}) by organ.")
    else:
        splits = default_split(ids, val_frac=args.val_frac, test_frac=args.test_frac,
                               seed=args.seed)
    Path(out_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    log.info(f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # ── HVG vocab ──
    # Legacy format: flat JSON list of gene names (still consumed by hvg_log indexing).
    # New format: {gene_name → token_id} with special tokens at the front.
    # We keep BOTH so existing shards / tools don't break:
    #   hvg_vocab.json       → legacy list (back-compat)
    #   hvg_vocab_dict.json  → new dict with [PAD]/[MASK]/[CLS]/[UNK] + genes
    vocab_path = out_dir / "hvg_vocab.json"
    vocab_dict_path = out_dir / "hvg_vocab_dict.json"
    if vocab_path.exists() and not args.rebuild_vocab:
        hvg = json.loads(vocab_path.read_text())
        log.info(f"Loaded cached HVG vocab ({len(hvg)} genes).")
    else:
        # Process samples in source priority order: hest → st1k → spatialcorpus.
        # HEST + ST1K are small (~2-5 s/sample) so the progress bar moves
        # visibly.  Spatialcorpus is heavy (50-80 s/sample after the row-
        # subsample fix); putting it last means we have ~95% of the dispersion
        # signal collected before the heavy phase, and a stall in spc
        # gracefully fails into a still-mostly-valid vocab.
        _SRC_PRIORITY = {"hest": 0, "st1k": 1, "spatialcorpus": 2}
        train_pairs = sorted(
            [(sid, sid_to_src.get(sid, "hest")) for sid in splits["train"]],
            key=lambda p: (_SRC_PRIORITY.get(p[1], 99), p[0]),
        )
        from collections import Counter
        log.info(f"HVG-pass order (source-priority): "
                 f"{dict(Counter(s for _,s in train_pairs))}")
        must_include = cfg.get("must_include_genes", [])
        # NEW: restrict candidate pool to a whitelist of GTF gene_types BEFORE
        # HVG ranking.  Recommended default ["protein_coding"] (+ IG/TR rearr.
        # genes) — drops AMBIGUOUS/BAC-clone/pseudogene contamination from the
        # candidate pool entirely so the HVG ranking is run over a clean,
        # biologically interpretable set.  Leave null to retain legacy
        # "all genes that survive noise filter" behavior.
        restrict_gt = cfg.get("restrict_to_gene_types")
        # NEW: optional union with top-K Highly Expressed Genes (mean raw count).
        # 0 = disabled (HVG-only).  ~512 is a sensible starting point if you
        # want stable housekeeping-like signals (ACTB/B2M/GAPDH/HSPA8) on every
        # spot in addition to the variable repertoire.
        heg_top_k = int(cfg.get("heg_top_k", 0))
        # NEW: cross-sample / cross-spot prevalence floors — drops genes that
        # appear in too few samples or too few spots (NLP analog: min_doc_freq).
        # Defaults stay at 0 = legacy behavior so existing prepared dirs match.
        min_samp_prev = float(cfg.get("min_sample_prevalence", 0.0))
        min_spot_prev = float(cfg.get("min_spot_prevalence", 0.0))
        max_samp_prev = float(cfg.get("max_sample_prevalence", 1.01))
        max_spot_prev = float(cfg.get("max_spot_prevalence", 1.01))
        # n_hvg is a CAP, not a target — null / 0 means "keep all
        # filter-survivors", letting the principle-driven filters (gene_type +
        # prevalence + noise) decide the final vocab size organically.
        n_hvg_cfg = cfg.get("n_hvg")
        if n_hvg_cfg in (None, 0, "null"):
            n_hvg_cfg = None
        hvg = select_global_hvg(train_pairs, hest_st, n_hvg_cfg,
                                must_include=must_include,
                                restrict_to_gene_types=restrict_gt,
                                heg_top_k=heg_top_k,
                                min_sample_prevalence=min_samp_prev,
                                min_spot_prevalence=min_spot_prev,
                                max_sample_prevalence=max_samp_prev,
                                max_spot_prevalence=max_spot_prev,
                                vocab_csv_out=out_dir / "vocab.csv")
        vocab_path.write_text(json.dumps(hvg))
        log.info(f"Saved HVG vocab to {vocab_path} ({len(hvg)} genes).")

        # Save per-sample QC stats (collected during the HVG-pass) — one row per
        # train sample with raw/cleaned/noise gene counts + n_spots.  vocab_qc
        # consumes this to draw "per-sample yield" diagnostics.
        sample_qc = getattr(select_global_hvg, "_last_sample_qc", [])
        if sample_qc:
            import pandas as _pd
            qc_csv = out_dir / "sample_qc.csv"
            _pd.DataFrame(sample_qc).to_csv(qc_csv, index=False)
            log.info(f"Saved per-sample QC ({len(sample_qc)} rows) → {qc_csv}")
        # Save per-sample QC stats (n_genes_raw / n_clean / n_pc / n_in_vocab / ...).
        try:
            qc_rows = getattr(select_global_hvg, "_last_sample_qc", [])
            if qc_rows:
                import pandas as _pd
                hvg_set = set(hvg)
                # Add n_in_vocab now that we know the vocab.  This is the most
                # important per-sample QC: how many of OUR vocab genes are
                # actually present in each sample?
                for r in qc_rows:
                    # We can only recompute n_in_vocab if we saved the sample's
                    # gene list; for now leave as NaN if not available.
                    pass
                qc_df = _pd.DataFrame(qc_rows)
                qc_csv = out_dir / "sample_qc.csv"
                qc_df.to_csv(qc_csv, index=False)
                log.info(f"Saved per-sample QC ({len(qc_df)} samples) → {qc_csv}")
        except Exception as _e:
            log.warning(f"sample_qc.csv save failed: {_e}")

    # Always (re)emit the dict-format vocab from `hvg` so the gene encoder
    # has a stable {gene → token_id} mapping.  Special tokens are FRONT-loaded
    # (matches spatula's preprocess convention).
    full_vocab = dict(SPECIAL_TOKENS)
    for i, g in enumerate(hvg):
        full_vocab[g] = i + N_SPECIAL
    vocab_dict_path.write_text(json.dumps(full_vocab))
    log.info(f"Saved HVG token vocab to {vocab_dict_path} "
             f"(vocab_size={len(full_vocab)} = {N_SPECIAL} specials + {len(hvg)} HVG).")

    # ── Global gene statistics for batch-effect-aware normalization ──
    # Computed once on the train pool; consumers (dataset runtime norm,
    # standardized_mse loss) load this file via cfg.
    stats_path = out_dir / "gene_stats.npz"
    if stats_path.exists() and not args.rebuild_vocab:
        log.info(f"gene_stats: SKIP (found cached {stats_path}); pass --rebuild_vocab to refresh.")
    else:
        _SRC_PRIORITY = {"hest": 0, "st1k": 1, "spatialcorpus": 2}
        train_pairs = sorted(
            [(sid, sid_to_src.get(sid, "hest")) for sid in splits["train"]],
            key=lambda p: (_SRC_PRIORITY.get(p[1], 99), p[0]),
        )
        compute_global_gene_stats(train_pairs, hest_st, hvg, stats_path,
                                   max_spots_per_sample=100)

    # Pre-import sklearn.NearestNeighbors so the .so is mmapped while memory
    # is still uncontested.  Under memory pressure, the cython .so file fails
    # to map and every subsequent KNN cache build throws ImportError — this
    # nuked all 649 HEST shard writes in the v3 run.
    from sklearn.neighbors import NearestNeighbors  # noqa: F401
    import gc as _gc

    summary = []
    for sid in tqdm(ids, desc="Prepare shards"):
        src = sid_to_src.get(sid, "hest")
        try:
            r = process_sample(sid, hest_st, hest_patch, hvg,
                               radius_px=cfg["spatial_radius_px"], out_dir=out_dir,
                               source=src, skip_novae=args.skip_novae)
        except Exception as e:
            r = {"sample_id": sid, "source": src, "n_spots": 0,
                 "status": f"error:{type(e).__name__}:{e}"}
            log.exception(f"failed {sid}")
        # Force per-shard cleanup so densified matrices don't accumulate.
        _gc.collect()
        summary.append(r)
        if r["status"] not in ("ok", "skipped"):
            log.warning(f"{sid}: {r['status']}")

    # save summary → REPORTS dir (not the data dir)
    reports = reports_dir_for(out_dir); reports.mkdir(parents=True, exist_ok=True)
    (reports / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for r in summary if r["status"] in ("ok", "skipped"))
    log.info(f"Done. {ok}/{len(summary)} shards available in {out_dir}.")
    log.info(f"prepare_summary.json → {reports}/")

    # Ensure every shard has its .knn.npy cache (idempotent — skips existing).
    # This guarantees DataLoader workers under DDP never have to build them
    # concurrently (which otherwise races and crashes with EOFError on np.load).
    from sklearn.neighbors import NearestNeighbors
    n_built = 0
    for sid in tqdm(ids, desc="KNN cache"):
        h5_path = out_dir / f"{sid}.h5"
        knn_path = h5_path.with_suffix(".knn.npy")
        if not h5_path.exists():
            continue
        if knn_path.exists() and knn_path.stat().st_size > 0:
            continue
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
        k = min(9, coords.shape[0])
        knn = NearestNeighbors(n_neighbors=k).fit(coords)
        _, idx = knn.kneighbors(coords)
        nn_idx = idx[:, 1:].astype(np.int32)
        # np.save appends ".npy" if the path doesn't already end with it, so make
        # the temp name end in ".npy" to keep behavior predictable.
        tmp = knn_path.with_name(knn_path.stem + ".tmp.npy")
        np.save(tmp, nn_idx)
        tmp.replace(knn_path)
        n_built += 1
    log.info(f"KNN caches: {n_built} built, {len(ids) - n_built} reused.")


if __name__ == "__main__":
    main()
