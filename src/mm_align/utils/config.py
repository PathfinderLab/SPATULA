from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import copy
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(*cfgs: Mapping[str, Any]) -> dict:
    """Deep-merge dicts; later dicts override earlier ones."""
    out: dict = {}
    for cfg in cfgs:
        out = _deep_update(out, cfg)
    return out


def _deep_update(base: dict, override: Mapping[str, Any]) -> dict:
    base = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(paths: list[str | Path], overrides: dict | None = None) -> dict:
    cfg: dict = {}
    for p in paths:
        cfg = merge_configs(cfg, load_yaml(p))
    if overrides:
        cfg = merge_configs(cfg, overrides)
    return cfg
