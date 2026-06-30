"""STimage-1K4M (st1k) dataset loader.

Data layout (/data/st1k/):
  ST/gene_exp/<slide>_count.csv         — raw counts, rows = spots, cols = gene symbols
  Visium/gene_exp/<slide>_count.csv     — same format
  VisiumHD/gene_exp/<slide>_count.csv   — same format
  meta/meta_all_gene02122025.csv        — per-slide metadata (species, tissue, tech, ...)

CSV format:
  - First column "Unnamed: 0" → spot barcode (e.g. "GSE144239_GSM4284316_10x26")
  - Remaining ~17K–35K columns → gene symbols (HGNC for the most part; some
    pseudogene IDs like RP11-*, AC0123*, AL1234.5* etc.)
  - Values: raw transcript counts (float32; many zeros)

Used for Stage-1 (gene-encoder only) where we need expression matrices, not
images.  The dataset returns an `AnnData` so the existing HEST pipeline
helpers (`_clean_adata_var_names`, `sc.pp.normalize_total`, `sc.pp.log1p`,
HVG selection) work unchanged.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ST1K_ROOT = Path("/data/st1k")
META_CSV_DEFAULT = ST1K_ROOT / "meta" / "meta_all_gene02122025.csv"

# tech → gene_exp directory.  We probe each in order.
_TECH_DIRS = ["Visium", "ST", "VisiumHD"]


def _csv_path_for(slide: str, base_dir: Path = ST1K_ROOT) -> Path | None:
    """Locate the count CSV for a slide.  Slide id can be either the bare
    GSE...GSM... form (matching the meta CSV) or with the trailing `_count`
    (matching human_list.txt).  We strip `_count` and search each tech dir.
    """
    s = re.sub(r"_count$", "", slide)
    for tech in _TECH_DIRS:
        p = base_dir / tech / "gene_exp" / f"{s}_count.csv"
        if p.exists():
            return p
    return None


def _coord_path_for(slide: str, base_dir: Path = ST1K_ROOT) -> Path | None:
    """Locate the coord CSV for a slide.  Format (per /data/st1k/{tech}/coord/):
        index = <slide>_<spot_barcode>
        columns = yaxis, xaxis, r        (NB: NOT (x, y, r))
    """
    s = re.sub(r"_count$", "", slide)
    for tech in _TECH_DIRS:
        p = base_dir / tech / "coord" / f"{s}_coord.csv"
        if p.exists():
            return p
    return None


def load_st1k_metadata(meta_csv: Path = META_CSV_DEFAULT) -> pd.DataFrame:
    """Load the slide-level metadata.  Returns the dataframe with `slide`,
    `species`, `tissue`, `tech` columns."""
    if not meta_csv.exists():
        raise FileNotFoundError(f"st1k metadata not found: {meta_csv}")
    df = pd.read_csv(meta_csv)
    return df


def list_st1k_samples(*, species: str = "human", whitelist: set[str] | None = None,
                      base_dir: Path = ST1K_ROOT,
                      meta_csv: Path = META_CSV_DEFAULT) -> list[str]:
    """Enumerate st1k slide ids that satisfy:
       - species filter (default 'human')
       - CSV exists under one of the tech dirs
       - if whitelist is provided, slide_id (with `_count` stripped) ∈ whitelist
         OR `<slide>_count` ∈ whitelist  (so both formats are accepted)

    Returns slide ids in BARE form (no `_count` suffix) — same convention as
    `meta.slide`.
    """
    meta = load_st1k_metadata(meta_csv)
    if species:
        meta = meta[meta["species"].str.lower() == species.lower()]
    slides = meta["slide"].astype(str).tolist()

    def _wl_match(s):
        if whitelist is None:
            return True
        return s in whitelist or f"{s}_count" in whitelist

    return [s for s in slides if _csv_path_for(s, base_dir) is not None and _wl_match(s)]


def read_st1k_sample(slide: str, base_dir: Path = ST1K_ROOT):
    """Read one st1k CSV and return a minimal AnnData with raw counts AND
    spot pixel coordinates (when the coord CSV is present).

    Returns
    -------
    AnnData with X = (n_spots, n_genes) float32 raw counts,
                obs.index = spot barcodes,
                obsm['spatial'] = (n_spots, 2) float32 pixel coords (xaxis,
                                  yaxis from the coord CSV — `obs.r` carries
                                  the radius column when present),
                var.index = gene symbols.
    Spots whose barcode isn't in the coord CSV get `(0, 0)` — but in practice
    the two CSVs are produced by the same pipeline and barcodes match 1-to-1.
    """
    import anndata as ad

    p = _csv_path_for(slide, base_dir)
    if p is None:
        raise FileNotFoundError(f"st1k CSV for slide {slide!r} not found under {base_dir}")
    # The CSVs are typically 600–4000 spots × 17K–34K genes → ~50–400 MB.
    # Index col is the first unnamed column.
    df = pd.read_csv(p, index_col=0, low_memory=False)
    X = df.to_numpy(dtype=np.float32, copy=False)
    var = pd.DataFrame(index=df.columns.astype(str))
    obs = pd.DataFrame(index=df.index.astype(str))
    a = ad.AnnData(X=X, obs=obs, var=var)

    # Attach spatial coords if the coord CSV exists.
    cp = _coord_path_for(slide, base_dir)
    if cp is not None:
        c = pd.read_csv(cp, index_col=0, low_memory=False)
        # Columns are (yaxis, xaxis, r).  We expose obsm['spatial'] as (x, y)
        # in pixel space — this matches the HEST convention (`adata.obsm['spatial']`
        # = column 0 → x_pixel, column 1 → y_pixel).
        if {"xaxis", "yaxis"}.issubset(c.columns):
            c_aligned = c.reindex(a.obs_names)
            xy = np.stack([
                c_aligned["xaxis"].to_numpy(dtype=np.float32),
                c_aligned["yaxis"].to_numpy(dtype=np.float32),
            ], axis=1)
            # Spots missing from coord file → NaN; replace with 0 + flag for
            # the caller via obs['has_coord'].
            has_coord = ~np.isnan(xy).any(axis=1)
            xy = np.nan_to_num(xy, nan=0.0)
            a.obsm["spatial"] = xy.astype(np.float32)
            a.obs["has_coord"] = has_coord
            if "r" in c.columns:
                a.obs["spot_radius"] = c_aligned["r"].to_numpy(dtype=np.float32)
    return a


def sample_source(slide: str, base_dir: Path = ST1K_ROOT) -> str:
    """Return which tech subtree a slide came from ('ST'/'Visium'/'VisiumHD')."""
    s = re.sub(r"_count$", "", slide)
    for tech in _TECH_DIRS:
        if (base_dir / tech / "gene_exp" / f"{s}_count.csv").exists():
            return tech
    return "?"
