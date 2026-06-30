"""Image encoder — feature-mode or foundation-mode with multi-FM registry.

backbone ∈ {feature, foundation}

For backbone="foundation" pick which FM via `foundation.name`:
  uni | virchow2 | gigapath | hoptimus0 | hoptimus1 | h0mini   (see foundation_models.py)

Then choose how to tune it:

  tune ∈ {none, all, partial:N, lora, adapter}
    - none       : everything in trunk frozen.  Only adapter MLP trains.
    - all        : full fine-tune.
    - partial:N  : freeze trunk except last N transformer blocks + final norm.
    - lora       : freeze trunk, wrap PEFT-LoRA on the last `partial_blocks`
                   blocks' attn+mlp Linear modules.
    - adapter    : freeze trunk entirely, attach a SEAL VisionAdapter
                   (Linear→ReLU→LayerNorm→Linear→LayerNorm with bottleneck).

Back-compat: old yaml with `image.uni.*` still works — it's treated as
`foundation.name="uni"` automatically by `build_image_encoder`.
"""
from __future__ import annotations
import re
from typing import Optional

import torch
import torch.nn as nn

from .foundation import build_foundation


# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(cur, hidden), nn.GELU(), nn.Dropout(dropout)]
        cur = hidden
    layers += [nn.Linear(cur, out)]
    return nn.Sequential(*layers)


class _VisionAdapter(nn.Module):
    """SEAL-style adapter (Linear→ReLU→LN→Linear→LN) over a frozen trunk."""

    def __init__(self, dim: int, bottleneck: int | None = None):
        super().__init__()
        if bottleneck is None:
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim),
                nn.Linear(dim, dim), nn.LayerNorm(dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, bottleneck), nn.ReLU(),
                nn.Linear(bottleneck, dim),
            )

    def forward(self, x):
        return self.net(x)


class _FMNormalize(nn.Module):
    """Convert uint8 (B,3,H,W) or (B,H,W,3) -> normalized float per FM config."""

    def __init__(self, mean, std):
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32 and x.dtype != torch.bfloat16 and x.dtype != torch.float16:
            x = x.float()
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.max() > 1.5:
            x = x / 255.0
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Freeze utilities
# ---------------------------------------------------------------------------

def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_partial(trunk: nn.Module, n_blocks: int, block_re: str = r"blocks\.(\d+)") -> None:
    """Unfreeze the last `n_blocks` transformer blocks + final norm."""
    _freeze_all(trunk)
    if n_blocks <= 0:
        return
    pat = re.compile(block_re)
    block_ids: list[int] = []
    for name, _ in trunk.named_modules():
        m = pat.search(name)
        if m:
            block_ids.append(int(m.group(1)))
    total = max(block_ids) + 1 if block_ids else 0
    keep = set(range(total - n_blocks, total))
    incl_blocks = {f"blocks.{i}." for i in keep} | {f".blocks.{i}." for i in keep}
    for n, p in trunk.named_parameters():
        if any(s in (n + ".") for s in incl_blocks) or "norm.weight" in n or "norm.bias" in n:
            p.requires_grad = True


def _wrap_lora(trunk: nn.Module, n_blocks: int, lora_cfg: dict,
               block_re: str = r"blocks\.(\d+)\."):
    from peft import LoraConfig, get_peft_model
    _freeze_all(trunk)
    pat_full = re.compile(rf"^{block_re}")
    pat_search = re.compile(block_re)
    block_ids: list[int] = []
    for name, _ in trunk.named_modules():
        m = pat_search.search(name)
        if m:
            block_ids.append(int(m.group(1)))
    total = max(block_ids) + 1 if block_ids else 0
    keep = set(range(total - n_blocks, total))
    target_modules: set[str] = set()
    for name, mod in trunk.named_modules():
        m = pat_search.search(name)
        if not m or int(m.group(1)) not in keep:
            continue
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if "attn" in name or "mlp" in name:
                target_modules.add(name)
    if not target_modules:
        raise RuntimeError("LoRA target_modules empty — trunk pattern mismatch.")
    config = LoraConfig(
        target_modules=sorted(target_modules), task_type=None,
        **(lora_cfg or {"r": 8, "lora_alpha": 8, "lora_dropout": 0.15, "use_rslora": False}),
    )
    return get_peft_model(trunk, config)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class FeatureImageEncoder(nn.Module):
    """MLP adapter on top of pre-extracted image features (no trunk)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.adapter = _mlp(in_dim, hidden_dim, out_dim, n_layers, dropout)
        self.tune_mode = "none"
        self.has_trunk = False
        self.foundation_name = "feature"

    def forward(self, *, image_feat: torch.Tensor,
                image_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.adapter(self.norm(image_feat))


class FoundationImageEncoder(nn.Module):
    """Generic foundation-model image encoder with tunable trunk."""

    def __init__(self, *, foundation_name: str, weights_path: Optional[str],
                 tune: str, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.1,
                 partial_blocks: int = 4,
                 adapter_bottleneck: int | None = 256,
                 lora_cfg: dict | None = None,
                 apply_transform_in_model: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.has_trunk = True
        self.foundation_name = foundation_name

        spec = build_foundation(foundation_name, weights_path=weights_path)
        trunk = spec["trunk"]
        feat_dim = spec["feat_dim"]
        self._post = spec["post"]
        self.image_size = spec["image_size"]
        self.preproc = _FMNormalize(spec["mean"], spec["std"]) if apply_transform_in_model else None

        tune = (tune or "none").strip().lower()
        self.tune_mode = tune

        if tune == "none":
            _freeze_all(trunk)
            self.trunk = trunk
            self.trunk_adapter = None
        elif tune == "all":
            for p in trunk.parameters():
                p.requires_grad = True
            self.trunk = trunk
            self.trunk_adapter = None
        elif tune.startswith("partial:"):
            n = int(tune.split(":", 1)[1])
            _unfreeze_partial(trunk, n)
            self.trunk = trunk
            self.trunk_adapter = None
        elif tune == "lora":
            self.trunk = _wrap_lora(trunk, partial_blocks, lora_cfg or {})
            self.trunk_adapter = None
        elif tune == "adapter":
            _freeze_all(trunk)
            self.trunk = trunk
            self.trunk_adapter = _VisionAdapter(feat_dim, adapter_bottleneck)
        else:
            raise ValueError(f"Unknown tune mode '{tune}'")

        self.norm = nn.LayerNorm(feat_dim)
        self.adapter = _mlp(feat_dim, hidden_dim, out_dim, n_layers, dropout)

    def _trunk_forward(self, image_raw: torch.Tensor) -> torch.Tensor:
        x = self.preproc(image_raw) if self.preproc is not None else image_raw
        out = self.trunk(x)
        out = self._post(out)
        if self.trunk_adapter is not None:
            out = self.trunk_adapter(out)
        return out

    def forward(self, *, image_feat: Optional[torch.Tensor] = None,
                image_raw: torch.Tensor) -> torch.Tensor:
        feat = self._trunk_forward(image_raw)
        return self.adapter(self.norm(feat))

    @property
    def freeze_stats(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trunk_total = sum(p.numel() for p in self.trunk.parameters()) if hasattr(self, "trunk") else 0
        trunk_trainable = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad) if hasattr(self, "trunk") else 0
        return {
            "mode": self.tune_mode, "foundation": self.foundation_name,
            "trainable_params": trainable, "total_params": total,
            "trainable_frac": trainable / max(1, total),
            "trunk_trainable": trunk_trainable, "trunk_total": trunk_total,
        }


# Back-compat alias — old code/configs still refer to UNIImageEncoder.
UNIImageEncoder = FoundationImageEncoder


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_image_encoder(cfg: dict) -> nn.Module:
    mcfg = cfg["model"]["image"]
    backbone = mcfg["backbone"]
    embed_dim = cfg["model"]["embed_dim"]

    if backbone == "feature":
        return FeatureImageEncoder(
            in_dim=mcfg["in_dim"], hidden_dim=mcfg["hidden_dim"],
            out_dim=embed_dim, n_layers=mcfg["n_layers"], dropout=mcfg["dropout"],
        )

    # backbone == "uni" / "foundation" — back-compat: both map to FoundationImageEncoder.
    if backbone in ("uni", "foundation"):
        # back-compat: read from `image.foundation` if present, else fall back to `image.uni`
        fcfg = mcfg.get("foundation") or mcfg.get("uni") or {}
        name = fcfg.get("name", "uni")  # default to UNI for back-compat
        return FoundationImageEncoder(
            foundation_name=name,
            weights_path=fcfg.get("weights_path"),
            tune=fcfg.get("tune", fcfg.get("freeze", "none")),  # accept legacy "freeze"
            hidden_dim=mcfg["hidden_dim"], out_dim=embed_dim,
            n_layers=mcfg["n_layers"], dropout=mcfg["dropout"],
            partial_blocks=fcfg.get("partial_blocks", 4),
            adapter_bottleneck=fcfg.get("adapter_bottleneck", 256),
            lora_cfg=fcfg.get("lora"),
            apply_transform_in_model=fcfg.get("apply_transform_in_model", True),
        )

    raise ValueError(f"Unknown image backbone: {backbone}")
