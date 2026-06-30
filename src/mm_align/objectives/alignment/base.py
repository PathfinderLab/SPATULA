"""Base interface for align losses."""
from __future__ import annotations
import torch
import torch.nn as nn


class AlignLoss(nn.Module):
    """Subclasses implement forward(model_out, batch) → (loss, log_dict).

    A reference to the student model is kept via __dict__ (not as a submodule)
    to avoid double-registration of parameters.
    """

    def __init__(self, cfg: dict, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "model", model)

    def forward(self, model_out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def on_after_step(self) -> None:
        """Hook called after each optimizer.step() (e.g. JEPA EMA update)."""
        return
