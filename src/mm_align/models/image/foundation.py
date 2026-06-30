"""Pathology foundation-model registry.

Each entry returns ``(trunk, feat_dim, post_forward, image_size, mean, std)``:
  * trunk: the timm/HF model
  * feat_dim: dimension of the trunk's pre-classifier feature
  * post_forward: function (trunk_out) -> (B, feat_dim) — handles CLS/pool/concat quirks
  * image_size / mean / std: defaults for preprocessing transforms

To add a new backbone:
  - register a `_build_<name>` function returning the tuple above
  - add it to `BUILDERS`
"""
from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Default forward post-processors
# ---------------------------------------------------------------------------

def _cls_only(x: torch.Tensor) -> torch.Tensor:
    """timm ViT with num_classes=0 returns the pre-classifier feature (pooled or CLS)."""
    return x


def _concat_cls_mean(prefix_tokens: int = 1) -> Callable[[torch.Tensor], torch.Tensor]:
    """Some FMs (Virchow, H0Mini) recommend concat(CLS, mean(patch_tokens))."""
    def _f(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            return x
        cls = x[:, 0]
        patch = x[:, prefix_tokens:].mean(1)
        return torch.cat([cls, patch], dim=-1)
    return _f


# ---------------------------------------------------------------------------
# Per-backbone builders
# ---------------------------------------------------------------------------

def _build_uni(weights_path: Optional[str] = None):
    """UNI-2 ViT-G/14 — local weight file (`assets/uni2-h.bin`) preferred."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    timm_kwargs = dict(
        model_name="vit_giant_patch14_224",
        img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536,
        mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
    )
    if weights_path:
        model = timm.create_model(pretrained=False, **timm_kwargs)
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    return {
        "trunk": model, "feat_dim": 1536, "post": _cls_only,
        "image_size": 224, "mean": data_cfg.get("mean", (0.485, 0.456, 0.406)),
        "std": data_cfg.get("std", (0.229, 0.224, 0.225)),
        "n_blocks": 24,
    }


def _build_virchow2(weights_path: Optional[str] = None):
    import timm
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                              mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU)
    # Virchow2 returns (B, 257, 1280) — CLS + 4 reg + patch tokens (per SEAL)
    return {
        "trunk": model, "feat_dim": 2560, "post": _concat_cls_mean(prefix_tokens=5),
        "image_size": 224, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "n_blocks": 32,
    }


def _build_gigapath(weights_path: Optional[str] = None):
    import timm
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return {
        "trunk": model, "feat_dim": 1536, "post": _cls_only,
        "image_size": 224, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "n_blocks": 40,
    }


def _build_hoptimus0(weights_path: Optional[str] = None):
    import timm
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False)
    return {
        "trunk": model, "feat_dim": 1536, "post": _cls_only,
        "image_size": 224,
        "mean": (0.707223, 0.578729, 0.703617),
        "std":  (0.211883, 0.230117, 0.177517),
        "n_blocks": 40,
    }


def _build_hoptimus1(weights_path: Optional[str] = None):
    import timm
    model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False)
    return {
        "trunk": model, "feat_dim": 1536, "post": _cls_only,
        "image_size": 224,
        "mean": (0.707223, 0.578729, 0.703617),
        "std":  (0.211883, 0.230117, 0.177517),
        "n_blocks": 40,
    }


def _build_h0mini(weights_path: Optional[str] = None):
    import timm
    model = timm.create_model("hf-hub:bioptimus/H0-mini", pretrained=True,
                              mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU)
    return {
        "trunk": model, "feat_dim": 1536,  # concat(cls, mean) = 768 + 768 actually
        "post": _concat_cls_mean(prefix_tokens=1),
        "image_size": 224, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "n_blocks": 12,
    }


BUILDERS = {
    "uni": _build_uni,
    "virchow2": _build_virchow2,
    "gigapath": _build_gigapath,
    "hoptimus0": _build_hoptimus0,
    "hoptimus1": _build_hoptimus1,
    "h0mini": _build_h0mini,
}


def build_foundation(name: str, weights_path: Optional[str] = None) -> dict:
    name = name.lower()
    if name not in BUILDERS:
        raise ValueError(
            f"Unknown foundation model: {name!r}. "
            f"Available: {sorted(BUILDERS.keys())}"
        )
    return BUILDERS[name](weights_path=weights_path)


def n_transformer_blocks(name: str) -> int:
    """For partial-unfreeze / LoRA block targeting."""
    return BUILDERS[name](weights_path=None).get("n_blocks", 12) if name in BUILDERS else 12
