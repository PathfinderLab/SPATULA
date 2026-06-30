"""UNI ViT-G/14 backbone with optional freeze strategies.

Weights default location: /workspace/mm_align/assets/uni2-h.bin
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def build_uni(weights_path: str | Path) -> tuple[nn.Module, dict]:
    """Build UNI ViT-G/14 and load weights. Returns (model, transform_cfg).

    `transform_cfg` is a dict with `mean`, `std`, `input_size` etc. (timm convention)
    so callers can apply normalization either in-dataset or in-model.
    """
    timm_kwargs = dict(
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model = timm.create_model(pretrained=False, **timm_kwargs)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    return model, data_cfg


def apply_freeze(model: nn.Module, mode: str) -> dict:
    """Apply a freeze policy to a UNI ViT-G. Returns a dict with stats.

    Modes (case-insensitive — "None"/"NONE"/None all normalised to "none"):
      "all"        : everything frozen.
      "none"       : everything trainable.
      "partial:N"  : freeze patch_embed/pos/cls/reg + first (24-N) blocks;
                     unfreeze last N blocks + final norm.
    """
    # Normalise input — accept None, "None", "NONE", etc. as "none".
    if mode is None:
        mode = "none"
    mode = str(mode).strip().lower()

    for p in model.parameters():
        p.requires_grad = False

    if mode == "all":
        pass
    elif mode == "none":
        for p in model.parameters():
            p.requires_grad = True
    elif mode.startswith("partial:"):
        n_unfreeze = int(mode.split(":")[1])
        blocks = model.blocks
        for b in blocks[-n_unfreeze:]:
            for p in b.parameters():
                p.requires_grad = True
        # final norm — different timm versions name it differently
        for name in ("norm", "fc_norm"):
            if hasattr(model, name):
                m = getattr(model, name)
                for p in m.parameters():
                    p.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze mode: {mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total,
            "trainable_frac": trainable / max(1, total), "mode": mode}


def uni_trainable_param_groups(model: nn.Module, base_lr: float, uni_lr_mult: float) -> list[dict]:
    """Two parameter groups (UNI backbone vs. everything else, when present)."""
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    return [{"params": backbone_params, "lr": base_lr * uni_lr_mult}]


class UNINormalize(nn.Module):
    """Convert uint8 (B,3,H,W) or (B,H,W,3) -> normalized float per timm config."""

    def __init__(self, data_cfg: dict):
        super().__init__()
        mean = torch.tensor(data_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        std = torch.tensor(data_cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8 or x.is_floating_point() is False:
            x = x.float()
        # Accept (B,H,W,3) → (B,3,H,W)
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        # If already in [0,255] scale, divide.
        if x.max() > 1.5:
            x = x / 255.0
        return (x - self.mean) / self.std
