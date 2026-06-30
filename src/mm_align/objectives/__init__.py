"""Objective builder. The unified objective handles all alignment methods.

Legacy "kind: clip|vicreg|jepa|cross_attn|mmae" experiment files are
remapped to the new schema (`align.method`) for backwards compatibility.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .unified import UnifiedObjective


_LEGACY_KIND_TO_ALIGN = {
    "clip":       {"method": "clip"},
    "vicreg":     {"method": "barlow"},        # closest no-negative analogue
    "cross_attn": {"method": "clip"},          # treat as InfoNCE on projections
    "mmae":       {"method": "clip"},          # MMAE-style: keep CLIP align,
                                                # rely on image_recon for the masked-AE signal
    "jepa":       {"method": "jepa"},
}


def _coerce_legacy(cfg: dict) -> dict:
    """If experiment.objective.kind is set (old schema), translate to the
    new `experiment.align` / `gene_recon` / `image_recon` schema."""
    exp = cfg.get("experiment", {})
    if "align" in exp:
        return cfg
    legacy = exp.get("objective", {})
    kind = legacy.get("kind")
    if kind in _LEGACY_KIND_TO_ALIGN:
        align = dict(_LEGACY_KIND_TO_ALIGN[kind])
        align.update({k: v for k, v in legacy.items() if k != "kind"})
        exp["align"] = align
        exp.setdefault("gene_recon", {"method": "mse", "weight": 1.0})
        exp.setdefault("image_recon", {"method": "mse", "weight": 1.0})
        cfg["experiment"] = exp
    return cfg


def build_objective(cfg: dict, model,
                    gene_means: Optional[np.ndarray] = None,
                    gene_stds: Optional[np.ndarray] = None) -> UnifiedObjective:
    cfg = _coerce_legacy(cfg)
    return UnifiedObjective(cfg, model, gene_means=gene_means, gene_stds=gene_stds)


__all__ = ["UnifiedObjective", "build_objective"]
