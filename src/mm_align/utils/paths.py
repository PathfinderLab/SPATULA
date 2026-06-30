"""Conventional paths for the project.

Convention:
  results/cache/<name>/   — DATA: shards (.h5), vocab json, splits.json,
                            gene_stats.npz, knn caches.  Anything the
                            dataloader / model actually reads.
  results/eda/<name>/     — REPORTS: prepare_summary.json, gene_vocab_audit.*,
                            predefined_vocab_report.md, figures/*.png.
                            Useful for inspection but never read by training.

`reports_dir_for(prepared_dir)` returns the eda mirror for a given prepared dir.
"""
from __future__ import annotations

from pathlib import Path


def reports_dir_for(prepared_dir: str | Path) -> Path:
    """Mirror `results/cache/<X>` → `results/eda/<X>`.
    Falls back to `<prepared_dir>/_reports` when the convention doesn't match.
    """
    p = Path(prepared_dir).resolve()
    parts = p.parts
    try:
        i = parts.index("cache")
        mirror = Path(*parts[:i], "eda", *parts[i+1:])
        return mirror
    except ValueError:
        return p / "_reports"
