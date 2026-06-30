"""SpatialCorpus-110M loader.

Layout: /data/spatialcorpus-110m/*.h5ad — 252 spatial transcriptomics datasets
spanning Xenium / MERSCOPE / GSE legacy / etc.  Schema (per-sample):
  - X        : (n_spots/cells, n_genes) raw counts (int counts in obs)
  - var_names: ENSG... (HUMAN) or ENSMUSG... (MOUSE) Ensembl IDs — *not* symbols
  - obs has x_centroid / y_centroid (spatial), total_counts, plus various
    organism / assay ontology terms.

Notes for our prepare pipeline:
  - Species filter:  most files name themselves (`*human*`, `*mouse*`, etc.)
    or carry `organism_ontology_term_id` in obs (NCBITaxon:9606 = human).
  - Gene IDs are Ensembl; we resolve them to HGNC symbols via the GTF
    (`mm_align.data.gene_symbols`) BEFORE the noise filter runs.
  - These samples have NO image / NO novae — same Stage-1-only treatment
    as ST1K (zeros pad on the image side at shard write time).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np


SPATIALCORPUS_ROOT = Path("/data/spatialcorpus-110m")

# NCBITaxon:9606 = Homo sapiens (cellxgene convention)
_HUMAN_TAXON = "NCBITaxon:9606"


# Filenames that obviously match a species (cheap pre-filter; final check is on obs)
_HUMAN_NAME_RE = re.compile(r"(human|breast|colon|liver|lung|brain|prostate|skin|kidney|lymph|melanoma|pancrea|ovary|cervic|gastric|stomach|placenta|tonsil|heart|bladder|nerve|bone|cervix)",
                             re.IGNORECASE)
_MOUSE_NAME_RE = re.compile(r"(mouse|murine|mus_musculus)", re.IGNORECASE)

# Non-human / non-Visium-like platforms to *exclude* by filename.
# Per design decision: drop Xenium (cell-level resolution + narrow gene panel
# 200-500 genes — incompatible with our 2048-HVG spot vocabulary).  Also drop
# MERSCOPE / MERFISH (similar issue) and any animal-species names.
_EXCLUDE_NAME_RE = re.compile(
    r"(xenium|merscope|merfish|cosmx|nanostring|"
    r"mouse|murine|rat|rattus|zebrafish|drosophila|"
    r"alzheimer.*mouse|brain_mouse)",
    re.IGNORECASE,
)


def _species_from_obs(adata) -> str | None:
    """Returns 'human' / 'mouse' / None based on obs metadata, if present."""
    if "organism_ontology_term_id" in adata.obs.columns:
        vals = adata.obs["organism_ontology_term_id"].astype(str).unique()
        if len(vals) == 1:
            v = vals[0]
            if v == _HUMAN_TAXON:
                return "human"
            if v == "NCBITaxon:10090":  # Mus musculus
                return "mouse"
    return None


def _species_from_filename(name: str) -> str | None:
    if _MOUSE_NAME_RE.search(name):
        return "mouse"
    if _HUMAN_NAME_RE.search(name):
        # Default to human if the name suggests a human tissue and there's no
        # mouse hint — verified later against obs metadata.
        return "human"
    return None


@lru_cache(maxsize=1)
def list_spatialcorpus_files(root: Path = SPATIALCORPUS_ROOT) -> list[Path]:
    return sorted(Path(root).glob("*.h5ad"))


def list_spatialcorpus_samples(*, species: str = "human",
                                root: Path = SPATIALCORPUS_ROOT,
                                strict: bool = False,
                                exclude_platforms: bool = True) -> list[str]:
    """Enumerate sample ids (= filenames without .h5ad) for the requested species.

    Behavior:
      - First pass: filename heuristic.  Excludes:
          * obvious mouse / animal markers
          * (if `exclude_platforms`) Xenium / MERSCOPE / MERFISH / CosMx (per
            design decision — cell-level platforms with narrow gene panels
            that don't fit our 2048-HVG spot vocabulary).
      - If `strict=True`, also OPENs each remaining file in backed mode to
        confirm via `obs.organism_ontology_term_id` (slow — opens ~200 files).
    """
    files = list_spatialcorpus_files(root)
    out: list[str] = []
    excluded_by_platform = 0
    for p in files:
        # Hard exclusion by filename (Xenium / animal / etc.)
        if exclude_platforms and _EXCLUDE_NAME_RE.search(p.name):
            excluded_by_platform += 1
            continue
        sp = _species_from_filename(p.name)
        if sp is not None and sp != species:
            continue
        # Unknown by filename → tentatively accept; the real loader will filter.
        if strict and sp is None:
            try:
                import anndata as ad
                a = ad.read_h5ad(p, backed="r")
                if _species_from_obs(a) != species:
                    continue
            except Exception:
                continue
        out.append(p.stem)
    if exclude_platforms and excluded_by_platform > 0:
        import logging
        logging.getLogger(__name__).info(
            f"list_spatialcorpus_samples: excluded {excluded_by_platform} files "
            f"by platform/species filter (Xenium/MERSCOPE/animal)."
        )
    return out


def read_spatialcorpus_sample(slide: str, *, root: Path = SPATIALCORPUS_ROOT,
                              species: str = "human",
                              gtf_map: dict | None = None,
                              max_spots: int | None = None):
    """Read one spatialcorpus .h5ad, convert Ensembl IDs → HGNC symbols,
    and return an AnnData with raw counts in X and HGNC symbols in var_names.

    If `gtf_map` is None we load the default GTF map lazily.  Unmapped IDs are
    kept as-is (uppercased); downstream `_clean_adata_var_names` may then drop
    them or they fall under the audit's `ensg_unresolved` bucket.

    Raises:
      FileNotFoundError       — slide file missing
      ValueError              — species mismatch (only when obs says so)
    """
    import anndata as ad
    p = Path(root) / f"{slide}.h5ad"
    if not p.exists():
        raise FileNotFoundError(p)
    a = ad.read_h5ad(p)
    if species:
        sp_obs = _species_from_obs(a)
        if sp_obs is not None and sp_obs != species:
            raise ValueError(f"{slide}: species={sp_obs!r} ≠ requested {species!r}")

    # CRITICAL: subsample rows BEFORE any column dedup / densification.  Some
    # spatialcorpus h5ads have 500K-1M cells × 20K+ genes — densifying first
    # (for groupby-sum of duplicate gene symbols) blows RAM (50+ GB) and stalls
    # the streaming pass for 20+ minutes per sample.  Capping early keeps the
    # dense matrix bounded at max_spots × n_genes (manageable).
    if max_spots is not None and a.n_obs > max_spots:
        rng_pre = np.random.default_rng(int(abs(hash(slide)) % (2**32)))
        sel_pre = rng_pre.choice(a.n_obs, max_spots, replace=False)
        a = a[sel_pre].copy()

    # Attach spatial coords from obs.{x,y} or obs.{x_centroid,y_centroid}.
    # Schema varies across the 252 spatialcorpus h5ads — we try both pairs
    # and store the result in obsm['spatial'] so downstream prepare.py can
    # treat all three sources (HEST / ST1K / SC) uniformly.
    spatial_cols = None
    for cand in (("x", "y"), ("x_centroid", "y_centroid"),
                  ("x_um", "y_um"), ("spatial_x", "spatial_y")):
        if cand[0] in a.obs.columns and cand[1] in a.obs.columns:
            spatial_cols = cand
            break
    if spatial_cols is not None:
        xy = np.stack([
            a.obs[spatial_cols[0]].to_numpy(dtype=np.float32),
            a.obs[spatial_cols[1]].to_numpy(dtype=np.float32),
        ], axis=1)
        has_coord = ~(np.isnan(xy).any(axis=1) | (xy == 0).all(axis=1))
        xy = np.nan_to_num(xy, nan=0.0)
        a.obsm["spatial"] = xy
        a.obs["has_coord"] = has_coord
    # Resolve ENSG → symbol (no-op for entries that don't look like ENSG).
    if gtf_map is None:
        from .gene_symbols import load_gtf_symbol_map
        gtf_map = load_gtf_symbol_map()
    ensg_to_sym = gtf_map["ensg_to_symbol"]

    def _strip_v(s: str) -> str:
        return s.split(".", 1)[0].upper()

    new_names = []
    for n in a.var_names.astype(str):
        u = _strip_v(n)
        new_names.append(ensg_to_sym.get(u, n.upper()))
    a.var_names = np.asarray(new_names, dtype=object)
    # If multiple ENSG mapped to the same symbol, sum their columns to a single
    # column (typical biology: sum is the canonical aggregation).
    if len(set(new_names)) < len(new_names):
        import pandas as pd
        # Use AnnData's built-in `var_names_make_unique` after summing dup cols.
        X = a.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        df = pd.DataFrame(X, columns=new_names)
        df = df.T.groupby(level=0).sum().T  # sum dup-symbol columns
        # Carry obsm forward (esp. spatial coords) through the AnnData rebuild.
        obsm_keep = dict(a.obsm) if hasattr(a.obsm, "keys") else {}
        a = ad.AnnData(X=df.to_numpy(dtype=np.float32, copy=False),
                       obs=a.obs.copy(),
                       var=__import__("pandas").DataFrame(index=df.columns.astype(str)))
        for k, v in obsm_keep.items():
            a.obsm[k] = v
    # (max_spots was applied earlier, before dedup/densify, to avoid OOM.)
    return a
