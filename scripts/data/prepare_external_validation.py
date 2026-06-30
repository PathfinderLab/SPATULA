#!/usr/bin/env python
"""Prepare external annotated ST validation datasets for mm_align.

Supported targets
-----------------
1) GSE176078-like annotated spot tables
   The GEO series page exposes processed supplementary files, but public file
   layouts vary after extraction.  This importer accepts either AnnData files
   (.h5ad) or loose count/coordinate/annotation tables and writes mm_align HDF5
   shards with spot-level annotation columns.

2) almaan/her2st
   HER2ST data are distributed via Zenodo and encrypted 7z archives.  After the
   user downloads/decrypts/extracts the repository data directory, this importer
   discovers ST count matrices, spot coordinate files, pathologist labels, and
   deconvolution proportion tables when present.

Output shard layout matches the Stage-1/Stage-1.5 loader contract:
    /barcode, /coords, /uni_feat, /novae_latent, /hvg_log, /patch_idx
Additional validation-only datasets are written when available:
    /annotation/<field>           string labels per spot
    /annotation/<field>_proba     numeric matrices such as cell-type proportions

Example
-------
python scripts/data/prepare_external_validation.py \
  --source her2st \
  --raw-dir /data/her2st \
  --prepared-dir results/cache/prepared_external/her2st \
  --vocab results/cache/prepared_expanded/hvg_vocab.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mm_align.data.gene_cleaning import clean_symbol  # noqa: E402

URLS = {
    "gse176078_geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078",
    "gse176078_suppl": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/",
    "her2st_repo": "https://github.com/almaan/her2st",
    "her2st_zenodo": "https://zenodo.org/record/3957257",
}

IMAGE_DIM = 1536
TX_DIM = 64


@dataclass
class ExternalSample:
    sample_id: str
    counts: pd.DataFrame           # rows=spots, cols=genes, raw/count-like values
    coords: np.ndarray             # rows aligned to counts, shape (n, 2)
    barcodes: list[str]
    annotations: dict[str, np.ndarray]
    matrices: dict[str, tuple[np.ndarray, list[str]]]
    source: str


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(path)
    # Spatial transcriptomics tables are usually TSV or whitespace-delimited.
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def _read_count_matrix(path: Path) -> pd.DataFrame:
    """Read a count matrix and return spots x genes.

    The function handles common ST layouts:
      - genes as rows, spots as columns, first column gene name
      - spots as rows, genes as columns, first column barcode
      - MatrixMarket is intentionally not guessed here because it requires
        separate barcode/gene files; use h5ad for those exports.
    """
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"empty count table: {path}")

    # If first column is non-numeric, use it as index.
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        df = df.set_index(first)

    # Keep numeric expression columns only.
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.dropna(axis=0, how="all")
    numeric = numeric.fillna(0)

    # Heuristic orientation: gene axis often has many HGNC-like symbols and fewer
    # rows than spots in small ST sections.  If index looks gene-heavy, transpose.
    idx_gene_like = sum(bool(re.match(r"^[A-Z0-9][A-Z0-9\-\.]+$", str(x).upper())) for x in numeric.index[:200])
    col_gene_like = sum(bool(re.match(r"^[A-Z0-9][A-Z0-9\-\.]+$", str(x).upper())) for x in numeric.columns[:200])
    if idx_gene_like >= col_gene_like and numeric.shape[0] <= max(50000, numeric.shape[1] * 4):
        numeric = numeric.T

    numeric.index = numeric.index.astype(str)
    numeric.columns = [clean_symbol(str(c)) for c in numeric.columns]
    numeric = numeric.loc[:, [bool(c) for c in numeric.columns]]
    numeric = numeric.groupby(level=0, axis=1).sum()
    return numeric.astype(np.float32)


def _find_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return sorted(set(p for p in out if p.is_file()))


def _sample_key(path: Path) -> str:
    stem = path.name
    for suf in [".txt.gz", ".tsv.gz", ".csv.gz", ".txt", ".tsv", ".csv", ".xlsx", ".h5ad"]:
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    stem = re.sub(r"(?:_counts?|[-.]counts?|_matrix|[-.]matrix|_stdata|_spots?)$", "", stem, flags=re.I)
    return stem


def _read_coords(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    cols = {_norm_name(c): c for c in df.columns}
    x_col = None
    y_col = None
    for cand in ["x", "pxlcolinfullres", "pixelx", "imagex", "xcoord", "arraycol", "col"]:
        if cand in cols:
            x_col = cols[cand]; break
    for cand in ["y", "pxlrowinfullres", "pixely", "imagey", "ycoord", "arrayrow", "row"]:
        if cand in cols:
            y_col = cols[cand]; break
    bc_col = None
    for cand in ["barcode", "spot", "spotid", "id", "name"]:
        if cand in cols:
            bc_col = cols[cand]; break
    if x_col is None or y_col is None:
        # Fallback: first two numeric columns.
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
        if len(nums) < 2:
            raise ValueError(f"cannot infer coordinate columns in {path}")
        x_col, y_col = nums[:2]
    out = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"),
                        "y": pd.to_numeric(df[y_col], errors="coerce")})
    out["barcode"] = df[bc_col].astype(str).values if bc_col else df.index.astype(str).values
    return out.dropna(subset=["x", "y"])


def _align_coords(counts: pd.DataFrame, coord_df: pd.DataFrame | None) -> np.ndarray:
    if coord_df is None or coord_df.empty:
        return np.zeros((counts.shape[0], 2), dtype=np.float32)
    lookup = coord_df.set_index("barcode")[["x", "y"]]
    coords = np.zeros((counts.shape[0], 2), dtype=np.float32)
    for i, bc in enumerate(counts.index.astype(str)):
        if bc in lookup.index:
            v = lookup.loc[bc]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[0]
            coords[i] = [float(v["x"]), float(v["y"])]
    if not np.any(coords):
        # Last resort: if same length, align by row order.
        m = min(len(coord_df), counts.shape[0])
        coords[:m] = coord_df[["x", "y"]].to_numpy(dtype=np.float32)[:m]
    return coords


def _read_annotation_table(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    cols = {_norm_name(c): c for c in df.columns}
    bc_col = None
    for cand in ["barcode", "spot", "spotid", "id", "name"]:
        if cand in cols:
            bc_col = cols[cand]; break
    if bc_col is None:
        df = df.copy()
        df["barcode"] = df.index.astype(str)
    else:
        df = df.rename(columns={bc_col: "barcode"})
    df["barcode"] = df["barcode"].astype(str)
    return df


def _annotation_candidates(root: Path, source: str) -> list[Path]:
    pats = ["*.tsv", "*.tsv.gz", "*.csv", "*.csv.gz", "*.txt", "*.txt.gz"]
    files = _find_files(root, pats)
    keep = []
    for p in files:
        low = str(p).lower()
        if source == "her2st":
            if any(k in low for k in ["st-pat", "st-cluster", "st-deconv", "props", "lbl", "label", "annotation"]):
                keep.append(p)
        else:
            if any(k in low for k in ["annot", "label", "cell", "type", "metadata", "meta", "deconv", "prop"]):
                keep.append(p)
    return sorted(set(keep))


def _collect_annotations(root: Path, source: str) -> dict[str, list[Path]]:
    by_key: dict[str, list[Path]] = {}
    for p in _annotation_candidates(root, source):
        by_key.setdefault(_sample_key(p), []).append(p)
    return by_key


def _attach_annotations(sample_id: str, barcodes: list[str], ann_files: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, list[str]]]]:
    annotations: dict[str, np.ndarray] = {}
    matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
    if not ann_files:
        return annotations, matrices
    bc = pd.Index([str(b) for b in barcodes])
    for path in ann_files:
        try:
            df = _read_annotation_table(path)
        except Exception as e:
            print(f"[external] warn: skip annotation {path}: {e}")
            continue
        df = df.drop_duplicates("barcode").set_index("barcode")
        aligned = df.reindex(bc)
        base = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()
        for col in aligned.columns:
            name = re.sub(r"[^a-zA-Z0-9]+", "_", str(col)).strip("_").lower()
            if name in {"", "x", "y", "row", "col"}:
                continue
            vals = aligned[col]
            numeric = pd.to_numeric(vals, errors="coerce")
            if numeric.notna().mean() > 0.9:
                # Store numeric proportion/value columns collectively by source file.
                continue
            arr = vals.fillna("Unknown").astype(str).to_numpy()
            key = f"{base}_{name}"
            annotations[key] = arr
        numeric_cols = []
        for col in aligned.columns:
            vals = pd.to_numeric(aligned[col], errors="coerce")
            if vals.notna().mean() > 0.8:
                numeric_cols.append(col)
        if numeric_cols:
            mat = aligned[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            mat_name = base if base else "numeric_annotation"
            matrices[mat_name] = (mat, [str(c) for c in numeric_cols])
            # For deconvolution proportions, add argmax label as major_cell_type.
            if mat.shape[1] >= 2 and not any("major_cell_type" in k for k in annotations):
                labels = np.array([numeric_cols[int(i)] for i in np.argmax(mat, axis=1)], dtype=object)
                zero = mat.sum(axis=1) <= 0
                labels[zero] = "Unknown"
                annotations[f"{mat_name}_argmax"] = labels.astype(str)
    return annotations, matrices


def _project_to_vocab(counts: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    x = counts.copy()
    x.columns = [clean_symbol(str(c)) for c in x.columns]
    x = x.loc[:, [bool(c) for c in x.columns]]
    x = x.groupby(level=0, axis=1).sum()
    # Spot-level normalize_total + log1p to match prepared hvg_log convention.
    sums = x.sum(axis=1).to_numpy(dtype=np.float64)
    scale = np.divide(1e4, sums, out=np.zeros_like(sums), where=sums > 0)
    x_norm = x.to_numpy(dtype=np.float32) * scale[:, None].astype(np.float32)
    np.log1p(x_norm, out=x_norm)
    mat = np.zeros((x.shape[0], len(vocab)), dtype=np.float32)
    col_lookup = {g: i for i, g in enumerate(x.columns.astype(str))}
    for j, g in enumerate(vocab):
        i = col_lookup.get(g)
        if i is not None:
            mat[:, j] = x_norm[:, i]
    return mat


def _write_shard(sample: ExternalSample, out_dir: Path, vocab: list[str], suffix: str, overwrite: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sample.sample_id}{suffix}.h5" if suffix else f"{sample.sample_id}.h5"
    out = out_dir / fname
    if out.exists() and not overwrite:
        return {"sample_id": sample.sample_id, "source": sample.source, "status": "skipped", "path": str(out)}
    hvg = _project_to_vocab(sample.counts, vocab)
    n = hvg.shape[0]
    coords = sample.coords.astype(np.float32)
    if coords.shape != (n, 2):
        coords = np.zeros((n, 2), dtype=np.float32)
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as f:
        f.create_dataset("barcode", data=np.asarray(sample.barcodes, dtype=object), dtype=dt)
        f.create_dataset("coords", data=coords)
        f.create_dataset("uni_feat", data=np.zeros((n, IMAGE_DIM), dtype=np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("patch_idx", data=np.arange(n, dtype=np.int32))
        f.create_dataset("novae_latent", data=np.zeros((n, TX_DIM), dtype=np.float32))
        f.create_dataset("hvg_log", data=hvg, compression="gzip", compression_opts=4)
        f.attrs["sample_id"] = sample.sample_id
        f.attrs["source"] = sample.source
        f.attrs["hvg_dim"] = len(vocab)
        f.attrs["uni_feat_dim"] = IMAGE_DIM
        if sample.annotations or sample.matrices:
            grp = f.create_group("annotation")
            for k, arr in sample.annotations.items():
                grp.create_dataset(k, data=np.asarray(arr, dtype=object), dtype=dt)
            for k, (mat, cols) in sample.matrices.items():
                ds = grp.create_dataset(f"{k}_proba", data=mat.astype(np.float32), compression="gzip", compression_opts=4)
                ds.attrs["columns"] = json.dumps(cols)
    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "status": "written",
        "path": str(out),
        "n_spots": int(n),
        "n_genes_vocab": int(len(vocab)),
        "n_nonzero_projected": int((hvg > 0).sum()),
        "annotation_fields": sorted(sample.annotations),
        "matrix_fields": sorted(sample.matrices),
    }


def _load_h5ad_samples(root: Path, source: str) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for p in sorted(root.rglob("*.h5ad")):
        a = sc.read_h5ad(p)
        sid = _sample_key(p)
        if a.n_obs == 0 or a.n_vars == 0:
            continue
        x = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
        counts = pd.DataFrame(x, index=a.obs_names.astype(str), columns=[clean_symbol(str(v)) for v in a.var_names])
        coords = np.zeros((a.n_obs, 2), dtype=np.float32)
        if "spatial" in a.obsm:
            coords = np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2]
        annotations = {}
        for col in a.obs.columns:
            if a.obs[col].dtype.name == "category" or a.obs[col].dtype == object:
                low = _norm_name(col)
                if any(k in low for k in ["cell", "type", "annot", "label", "cluster", "major"]):
                    annotations[low] = a.obs[col].astype(str).to_numpy()
        samples.append(ExternalSample(sid, counts, coords, list(counts.index), annotations, {}, source))
    return samples


def discover_table_samples(root: Path, source: str) -> list[ExternalSample]:
    h5ad = _load_h5ad_samples(root, source)
    if h5ad:
        return h5ad

    count_patterns = ["*count*.tsv", "*count*.tsv.gz", "*count*.txt", "*count*.txt.gz", "*count*.csv", "*cnt*.tsv", "*cnt*.tsv.gz"]
    if source == "her2st":
        count_patterns += ["ST-cnts/*.tsv", "ST-cnts/*.txt", "ST-cnts/*.csv", "**/ST-cnts/*"]
    count_files = [p for p in _find_files(root, count_patterns) if not p.name.startswith(".")]
    coord_files = _find_files(root, ["*coord*.tsv", "*coord*.csv", "*spot*.tsv", "*spot*.csv", "**/ST-spotfiles/*"])
    ann_by_key = _collect_annotations(root, source)

    coords_by_key: dict[str, Path] = {}
    for p in coord_files:
        coords_by_key.setdefault(_sample_key(p), p)

    samples: list[ExternalSample] = []
    seen: set[str] = set()
    for cf in count_files:
        key = _sample_key(cf)
        if key in seen:
            continue
        seen.add(key)
        try:
            counts = _read_count_matrix(cf)
        except Exception as e:
            print(f"[external] warn: skip counts {cf}: {e}")
            continue
        coord_path = coords_by_key.get(key)
        coord_df = None
        if coord_path is not None:
            try:
                coord_df = _read_coords(coord_path)
            except Exception as e:
                print(f"[external] warn: coords failed {coord_path}: {e}")
        coords = _align_coords(counts, coord_df)
        ann_files = ann_by_key.get(key, [])
        # If exact key fails, attach files whose key contains/contained by sample key.
        if not ann_files:
            for ak, files in ann_by_key.items():
                if ak in key or key in ak:
                    ann_files.extend(files)
        annotations, matrices = _attach_annotations(key, list(counts.index.astype(str)), ann_files)
        samples.append(ExternalSample(key, counts, coords, list(counts.index.astype(str)), annotations, matrices, source))
    return samples


def write_splits(out_dir: Path, sample_ids: list[str], split_name: str) -> None:
    payload = {"train": [], "val": sample_ids, "test": sample_ids, "external_validation": sample_ids}
    (out_dir / split_name).write_text(json.dumps(payload, indent=2))


def load_vocab(path: Path) -> list[str]:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return [str(x).upper() for x in obj]
    if isinstance(obj, dict):
        return [k for k, v in sorted(obj.items(), key=lambda kv: kv[1]) if not str(k).startswith("[")]
    raise ValueError(f"unsupported vocab JSON: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("gse176078", "her2st", "generic"), required=True)
    ap.add_argument("--raw-dir", required=True, help="Extracted/downloaded dataset root.")
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--vocab", default="results/cache/prepared_expanded/hvg_vocab.json")
    ap.add_argument("--suffix", default=None, help="Shard suffix. Defaults to .<source>.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--splits-name", default=None)
    ap.add_argument("--print-download-info", action="store_true")
    args = ap.parse_args()

    if args.print_download_info:
        print(json.dumps(URLS, indent=2))
        return

    raw = Path(args.raw_dir)
    out = Path(args.prepared_dir)
    vocab = load_vocab(Path(args.vocab))
    suffix = args.suffix if args.suffix is not None else ("" if args.source == "generic" else f".{args.source}")
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix

    samples = discover_table_samples(raw, args.source)
    if not samples:
        raise SystemExit(
            f"No usable samples found under {raw}. For HER2ST, first decrypt/extract the Zenodo archives "
            "so ST-cnts/, ST-spotfiles/, and res/ are visible."
        )

    rows = []
    for sample in samples:
        rows.append(_write_shard(sample, out, vocab, suffix=suffix, overwrite=args.overwrite))
    manifest = Path(args.manifest) if args.manifest else out / f"{args.source}_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        writer.writeheader(); writer.writerows(rows)

    ids = [r["sample_id"] for r in rows if r.get("status") in {"written", "skipped"}]
    split_name = args.splits_name or f"splits_{args.source}_validation.json"
    write_splits(out, ids, split_name)
    print(f"[external] source={args.source} samples={len(ids)}")
    print(f"[external] wrote manifest: {manifest}")
    print(f"[external] wrote splits:   {out / split_name}")
    print(f"[external] shard suffix:   {suffix or '(none)'}")


if __name__ == "__main__":
    main()
