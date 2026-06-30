"""Stage-1/2 checkpoint save / prune / tx-encoder-only export.

Pulled out of `scripts/train.py` so the entrypoint stays orchestration-only.
"""
from __future__ import annotations
import re
from pathlib import Path

import torch


def build_ckpt_state(model, objective, cfg: dict, epoch: int, accelerator,
                     *, full: bool = False) -> dict:
    """Build the dict torch.save() will dump.

    full=False (default): only the trainable params + buffers — far smaller
    file, fine for resume since frozen weights come from a known init.
    full=True: every weight in `model` and `objective` (debug / final).
    """
    m = accelerator.unwrap_model(model) if accelerator else model
    o = accelerator.unwrap_model(objective) if accelerator else objective
    if full:
        return {"model": m.state_dict(), "objective": o.state_dict(),
                "cfg": cfg, "epoch": epoch, "ckpt_format": "full"}

    def _filter(mod):
        train_names = {n for n, p in mod.named_parameters() if p.requires_grad}
        buf_names = {n for n, _ in mod.named_buffers()}
        sd = mod.state_dict()
        return {k: v for k, v in sd.items() if k in train_names or k in buf_names}

    return {"model": _filter(m), "objective": _filter(o),
            "cfg": cfg, "epoch": epoch, "ckpt_format": "trainable_only"}


def save_tx_encoder_only(model, cfg: dict, epoch: int, accelerator, run_dir: Path,
                          *, basename: str = "ckpt_tx_encoder.pt") -> None:
    """Stage-1 artifact for Stage-2 reuse.

    Saves only the `tx_encoder` state_dict plus the minimum config slice that
    Stage 1.5 / Stage 2 need to rebuild the encoder (`cfg_tx` block).
    See `scripts/train/stage15.py` for how this is consumed.
    """
    m = accelerator.unwrap_model(model) if accelerator else model
    tx_sd = m.tx_encoder.state_dict()
    # Stage 1.5 / 2 also need the input pipeline (gene_norm + vocab_clip)
    # so the frozen encoder sees the same distribution it trained on.
    data_slice = {"prepared_dir": cfg["data"]["prepared_dir"]}
    for k in ("gene_norm", "vocab_clip"):
        if k in cfg["data"]:
            data_slice[k] = cfg["data"][k]
    cfg_tx_slice = {
        "model": {"transcriptomics": cfg["model"]["transcriptomics"],
                  "embed_dim": cfg["model"]["embed_dim"]},
        "data":  data_slice,
    }
    torch.save(
        {"tx_encoder": tx_sd, "cfg_tx": cfg_tx_slice,
         "epoch": epoch, "ckpt_format": "tx_encoder_only"},
        run_dir / basename,
    )


def prune_old_ckpts(run_dir: Path, keep_n: int) -> None:
    """Keep only the latest `keep_n` `ckpt_epochN.pt` files in `run_dir`."""
    epoch_ckpts = sorted(
        run_dir.glob("ckpt_epoch*.pt"),
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    )
    if len(epoch_ckpts) > keep_n:
        for old in epoch_ckpts[:-keep_n]:
            try:
                old.unlink()
            except OSError:
                pass
