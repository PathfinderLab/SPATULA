"""Training-time helpers — pulled out of `scripts/train.py` so the entrypoint
stays orchestration-only.  Subdivided by role:

    .checkpoints  — ckpt build / save / prune
    .monitoring   — log summariser + loss-curve renderer
    .samplers     — DataLoader samplers (e.g. shard-blocked)
    .schedulers   — cosine-warmup LR schedule + gene-stats loader
"""
from .checkpoints import build_ckpt_state, save_tx_encoder_only, prune_old_ckpts
from .monitoring import render_loss_curve, render_val_metric_curves, render_stage1_metric_curves, summarize_log
from .samplers import SampleBlockSampler
from .schedulers import cosine_warmup_schedule, load_gene_stats

__all__ = [
    "build_ckpt_state", "save_tx_encoder_only", "prune_old_ckpts",
    "render_loss_curve", "render_val_metric_curves", "render_stage1_metric_curves", "summarize_log",
    "SampleBlockSampler",
    "cosine_warmup_schedule", "load_gene_stats",
]
