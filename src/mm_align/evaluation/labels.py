"""Helpers to materialize per-spot biological labels (organ, disease) for eval.

Source of truth: /data/hest/HEST_v1_1_0.csv (sample_id -> organ / disease_state).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def hest_metadata(csv_path: str = "/data/hest/HEST_v1_1_0.csv") -> dict:
    """Returns {'organ': {sid: organ}, 'disease': {sid: disease}, 'species': {sid: species}}."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {
        "organ": dict(zip(df["id"].astype(str), df["organ"].astype(str))),
        "disease": dict(zip(df["id"].astype(str), df["disease_state"].astype(str))),
        "species": dict(zip(df["id"].astype(str), df["species"].astype(str))),
    }


def spot_organ_labels(sample_ids: list[str], sample_idx_per_spot: np.ndarray | None,
                      meta: dict | None = None) -> np.ndarray | None:
    """Map a per-spot sample_idx array to per-spot organ strings."""
    if sample_idx_per_spot is None or not sample_ids:
        return None
    meta = meta or hest_metadata()
    organ_map = meta["organ"]
    labels = np.array([
        organ_map.get(sample_ids[int(si)], "Unknown") for si in sample_idx_per_spot
    ])
    # Cast empty / nan organs to "Unknown" — keep string
    labels = np.where((labels == "") | (labels == "nan"), "Unknown", labels)
    return labels
