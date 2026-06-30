"""LR schedule + gene-stats loader helpers."""
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_schedule(optimizer, total_steps: int, warmup_steps: int,
                            min_lr_ratio: float = 0.1) -> LambdaLR:
    """Linear ramp for `warmup_steps`, then cosine decay to `min_lr_ratio · lr`.

    Used at the top of every training stage.  Behaves correctly when
    Accelerate's prepare() advances the scheduler `num_processes` times per
    optimizer.step() (caller should multiply total/warmup by num_processes).
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return LambdaLR(optimizer, lr_lambda)


def load_gene_stats(path: str | None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load (mean, std) arrays from `gene_stats.npz`; None on missing/error.

    Used by the `standardized_mse` recon path inside UnifiedObjective.
    """
    if not path:
        return None, None
    try:
        data = np.load(path)
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)
    except Exception:
        return None, None
