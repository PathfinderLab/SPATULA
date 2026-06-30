"""Train / val / test split builders.

Two strategies:
    default_split     — random shuffle, ID-only
    stratified_split  — keeps each group (e.g. organ) proportional in all splits

Both return  {"train": [...], "val": [...], "test": [...]}.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def default_split(sample_ids: list[str],
                   val_frac: float = 0.05, test_frac: float = 0.05,
                   seed: int = 42) -> dict[str, list[str]]:
    """Simple random shuffle into train/val/test fractions."""
    rng = np.random.default_rng(seed)
    ids = sorted(set(sample_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = ids[:n_test]
    val = ids[n_test:n_test + n_val]
    train = ids[n_test + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def stratified_split(sample_ids: list[str],
                      id_to_group: dict[str, str],
                      *,
                      val_frac: float = 0.10, test_frac: float = 0.10,
                      seed: int = 42) -> dict[str, list[str]]:
    """Group-stratified split.

    For each group (e.g. organ), allocate roughly val_frac / test_frac of its
    members to val / test.  Small-group edge cases:
        n == 1 → all goes to train
        n == 2 → 1 train, 1 val, test empty
        n >= 3 → at least 1 sample in each of train/val/test

    Why the edge cases: with PDF-curated tissue panels we have organs with as
    few as 1 sample; dropping them or evenly slicing would either lose them
    or violate the "≥1 in each split" requirement for downstream linear probes.
    """
    assert 0 < val_frac < 1 and 0 < test_frac < 1
    assert val_frac + test_frac < 1
    rng = np.random.default_rng(seed)

    groups: dict[str, list[str]] = {}
    for sid in sorted(set(sample_ids)):
        g = id_to_group.get(sid, "Unknown")
        if g is None or (isinstance(g, float) and np.isnan(g)):
            g = "Unknown"
        groups.setdefault(str(g), []).append(sid)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for members in groups.values():
        members = list(members)
        rng.shuffle(members)
        n = len(members)
        if n == 1:
            train.extend(members); continue
        if n == 2:
            train.append(members[0]); val.append(members[1]); continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(1, n - n_train - n_test)
            n_test = max(1, n - n_train - n_val)
        test.extend(members[:n_test])
        val.extend(members[n_test:n_test + n_val])
        train.extend(members[n_test + n_val:])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def list_prepared_shards(prepared_dir: str | Path) -> list[Path]:
    """Enumerate `<prepared_dir>/*.h5` in sorted order."""
    return sorted(Path(prepared_dir).glob("*.h5"))


def decode_bytes_array(arr) -> np.ndarray:
    """h5py byte-string columns → numpy str array (handles object-array nesting)."""
    out = []
    for b in arr:
        if isinstance(b, np.ndarray):
            b = b[0] if b.shape else b.item()
        out.append(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b))
    return np.asarray(out)
