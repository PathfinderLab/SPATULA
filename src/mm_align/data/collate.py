from __future__ import annotations
import torch


def pad_collate(batch: list[dict]) -> dict:
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = torch.tensor(vals)
    return out
