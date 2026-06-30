"""DLPFC (Maynard et al. 2021) loader — external test-set for Stage 1.

Each DLPFC sample is a Visium slide with per-spot **cortical layer labels**
(Layer_1..Layer_6 + WM) — the standard benchmark for spatial-aware
embeddings.  This module turns the raw 10x feature-barcode HDF5 + tissue
positions + truth.txt into the same `(hvg_log, coords, labels)` triple the
Stage-1 evaluation expects.

The HVG matrix is aligned to **the project's own vocabulary**
(`hvg_vocab_dict.json`) so the encoder sees its training-time gene columns:
missing genes → zero, duplicates → sum of counts.

Layout (one sample directory):
    <root>/<sid>/
        <sid>_filtered_feature_bc_matrix.h5        (10x sparse counts)
        <sid>_truth.txt                            (barcode\\tlayer)
        spatial/tissue_positions_list.{csv,txt}    (barcode, in_tissue, row, col, px_y, px_x)
"""
from __future__ import annotations
from pathlib import Path

import h5py
import json
import numpy as np
from scipy.sparse import csc_matrix


def list_dlpfc_samples(root: str | Path, *,
                         require_truth: bool = True) -> list[Path]:
    """All sample subdirs that have h5 + spatial/  (and optionally truth).

    When `require_truth=False` we include samples without `*_truth.txt`.
    Those samples still support unsupervised eval (clustering, gene-map
    probe, SVG, spatial-continuity) — supervised layer-probe metrics return
    NaN gracefully because the loader stamps every spot's layer as "NA".
    """
    root = Path(root)
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / f"{d.name}_filtered_feature_bc_matrix.h5").exists():
            continue
        if not (d / "spatial").is_dir():
            continue
        if require_truth and not (d / f"{d.name}_truth.txt").exists():
            continue
        out.append(d)
    return out


def _read_10x_h5(h5_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Returns (X_dense_TxG_in_h5_order, barcodes, gene_symbols).

    The 10x H5 stores CSC indexed by spot: `data`/`indices`/`indptr` follow
    scipy.sparse.csc_matrix convention with shape stored in `matrix['shape']`
    as (n_genes, n_spots).
    """
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(int(x) for x in m["shape"][:])         # (n_genes, n_spots)
        bcs = [b.decode() if isinstance(b, bytes) else str(b)
                for b in m["barcodes"][:]]
        names = [n.decode() if isinstance(n, bytes) else str(n)
                  for n in m["features"]["name"][:]]
    spm = csc_matrix((data, indices, indptr), shape=shape)
    # 10x stores (gene, spot); transpose to (spot, gene) before densifying.
    dense = spm.T.toarray().astype(np.float32)
    return dense, bcs, names


def _read_tissue_positions(spatial_dir: Path,
                             barcodes_keep: list[str]) -> np.ndarray:
    """Returns (n_keep, 2) pixel coords, ordered to match barcodes_keep."""
    csv = spatial_dir / "tissue_positions_list.csv"
    if not csv.exists():
        csv = spatial_dir / "tissue_positions_list.txt"
    if not csv.exists():
        raise FileNotFoundError(f"tissue_positions in {spatial_dir}")
    rows: dict[str, tuple[float, float]] = {}
    with open(csv) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            bc = parts[0]
            try:
                py, px = float(parts[4]), float(parts[5])
            except ValueError:
                continue
            rows[bc] = (px, py)             # store as (x, y) — px=col_pixel, py=row_pixel
    out = np.zeros((len(barcodes_keep), 2), dtype=np.float32)
    for i, b in enumerate(barcodes_keep):
        out[i] = rows.get(b, (0.0, 0.0))
    return out


def _read_truth(truth_path: Path) -> dict[str, str]:
    """barcode → layer string."""
    out: dict[str, str] = {}
    with open(truth_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def _align_to_vocab(dense: np.ndarray, sample_symbols: list[str],
                     vocab_symbol_to_idx: dict[str, int]) -> np.ndarray:
    """Re-project (n_spots, n_features_sample) onto (n_spots, n_vocab).

    Genes outside the project vocab → dropped; vocab genes absent in sample
    → zero column.  Duplicate symbols in the sample are summed.
    """
    n_spots = dense.shape[0]
    n_vocab = len(vocab_symbol_to_idx)
    out = np.zeros((n_spots, n_vocab), dtype=np.float32)
    for col_in_sample, sym in enumerate(sample_symbols):
        vidx = vocab_symbol_to_idx.get(sym)
        if vidx is None:
            continue
        # Symbol-duplicate genes (rare but real) get summed.
        out[:, vidx] += dense[:, col_in_sample]
    return out


def load_dlpfc_sample(sample_dir: str | Path,
                       vocab_path: str | Path,
                       *, normalize: str = "log1p_cp10k") -> dict:
    """Load one DLPFC sample, aligned to our HVG vocabulary.

    Returns:
        {
          'sample_id': str,
          'hvg_log':   (n_spots, vocab_size) float32 — log1p(X/n_counts*1e4) by default
          'coords':    (n_spots, 2) float32 (pixel)
          'layers':    (n_spots,)   object  — Layer_1..6, WM, or NA for missing
          'barcodes':  (n_spots,)   object
        }
    Spots with no truth label are kept (layers='NA') so the caller can decide
    whether to drop them.
    """
    sample_dir = Path(sample_dir)
    sid = sample_dir.name
    vocab_dict = json.loads(Path(vocab_path).read_text())
    # vocab_dict maps symbol → token id including specials ([PAD] etc.).  The
    # downstream hvg_log matrix is (n_spots, n_real_genes), so we reindex
    # real genes to a contiguous 0..n_real_genes-1 column space, in the same
    # order they appear in the vocab.
    real_pairs = sorted(((k, v) for k, v in vocab_dict.items()
                         if not k.startswith("[")), key=lambda kv: kv[1])
    sym2idx = {sym: i for i, (sym, _) in enumerate(real_pairs)}
    # Reindex to dense 0..vocab_size−1 over real-gene rows (keep original ids
    # so the encoder's vocab_clip mapping still works).
    counts, bcs, sym_in_sample = _read_10x_h5(
        sample_dir / f"{sid}_filtered_feature_bc_matrix.h5")
    truth_path = sample_dir / f"{sid}_truth.txt"
    if truth_path.exists():
        truth = _read_truth(truth_path)
        # Keep only barcodes with a truth label — that's what DLPFC eval uses.
        keep_mask = np.array([b in truth for b in bcs])
        counts = counts[keep_mask]
        bcs = [b for b, k in zip(bcs, keep_mask) if k]
        layers = np.array([truth[b] for b in bcs], dtype=object)
    else:
        # No truth file (e.g. 151675 — present in some spatialLIBD distributions
        # but missing per-spot annotations).  Keep every spot whose barcode has
        # a tissue-position match; stamp every layer as "NA" so downstream
        # supervised metrics return NaN gracefully.
        layers = np.array(["NA"] * len(bcs), dtype=object)
    aligned = _align_to_vocab(counts, sym_in_sample, sym2idx)   # (n_spots, vocab_size)
    coords = _read_tissue_positions(sample_dir / "spatial", bcs)
    # Normalise to log1p(CP10K) — the same convention prepare_data.py uses.
    if normalize == "log1p_cp10k":
        n_counts = aligned.sum(axis=1, keepdims=True)
        n_counts = np.maximum(n_counts, 1.0)
        aligned = np.log1p(aligned / n_counts * 1e4).astype(np.float32)
    elif normalize == "none":
        aligned = aligned.astype(np.float32)
    else:
        raise ValueError(f"unknown normalize={normalize!r}")
    return {
        "sample_id": sid,
        "hvg_log": aligned,                # (n_spots, vocab_size_incl_specials)
        "coords": coords,
        "layers": layers,
        "barcodes": np.array(bcs, dtype=object),
    }
