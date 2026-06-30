"""Custom samplers used by training DataLoaders."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

from ..data import PairedSpotDataset


class SampleBlockSampler(Sampler[int]):
    """Yields indices in shuffled per-sample blocks.

    Why per-sample blocks: the spatial-JEPA aux loss needs each batch to
    contain spatial neighbours from the SAME sample (different samples
    have different coord systems).  Shuffling at the shard level gives us
    high spatial-neighbour hit-ratio without sacrificing randomness across
    epochs.
    """

    def __init__(self, dataset: PairedSpotDataset, batch_size: int, shuffle: bool, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        starts = self.dataset._starts
        valid_idx = getattr(self.dataset, "_valid_indices", None)
        rng = np.random.default_rng(self.seed if not self.shuffle else None)
        shard_order = list(range(len(self.dataset.shards)))
        if self.shuffle:
            rng.shuffle(shard_order)
        out: list[int] = []
        if valid_idx is None:
            # Legacy path: raw global indices into hvg_log rows.
            for si in shard_order:
                spots = np.arange(starts[si], starts[si + 1])
                if self.shuffle:
                    rng.shuffle(spots)
                out.extend(spots.tolist())
        else:
            # min_seq_len filter active.  Yield EFFECTIVE positions (indices
            # into valid_idx), so PairedSpotDataset._locate can do
            # `idx = valid_idx[idx]` without going out of bounds.  Shard
            # blocking preserved for cache locality.
            for si in shard_order:
                lo = int(np.searchsorted(valid_idx, starts[si], side="left"))
                hi = int(np.searchsorted(valid_idx, starts[si + 1], side="left"))
                block = np.arange(lo, hi)
                if self.shuffle:
                    rng.shuffle(block)
                out.extend(block.tolist())
        return iter(out)

    def __len__(self):
        return len(self.dataset)
