"""Accelerate-based training entrypoint — SEAL-style objective.

Per-batch metrics (cosine_sim, gene_pcc/spearman/mse) are streamed during
training.  No zero-shot / linear-probe eval is run inside the epoch loop —
those are heavy and only fire ONCE at the end of training via
`scripts/eval/zero_shot.py` + `scripts/eval/linear_probe.py` (or via the
`final_eval` flags in train.yaml).

Validation loss (running the objective on the val loader with no grad)
provides the early-stop signal.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Env hygiene (must be set before importing torch).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_align.utils import load_config, set_seed, get_logger
from mm_align.data import PairedSpotDataset, build_dataset_from_split, pad_collate
from mm_align.data.pairs import prebuild_knn_caches
from mm_align.models import MMAligner
from mm_align.objectives import build_objective
from mm_align.training import (
    SampleBlockSampler,
    cosine_warmup_schedule,
    load_gene_stats,
    build_ckpt_state,
    save_tx_encoder_only,
    prune_old_ckpts,
    render_loss_curve,
    render_val_metric_curves,
    render_stage1_metric_curves,
    summarize_log,
)

# Local aliases keep the original call-sites unchanged.
_build_ckpt_state = build_ckpt_state
_save_tx_encoder_only = save_tx_encoder_only
_prune_old_ckpts = prune_old_ckpts
_render_loss_curve = render_loss_curve
_render_val_metric_curves = render_val_metric_curves
_render_stage1_metric_curves = render_stage1_metric_curves
_summarize_log = summarize_log
_load_gene_stats = load_gene_stats

log = get_logger("train")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/stage1/data.yaml")
    ap.add_argument("--model", default="configs/stage1/model.yaml")
    ap.add_argument("--train", default="configs/stage1/train.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--val-every-epoch", type=int, default=None,
                    help="Override train.val_every_epoch. Useful for long main runs.")
    ap.add_argument("--val-max-batches", type=int, default=None,
                    help="Override train.val_loss.max_batches. Use <=0 for full validation.")
    ap.add_argument("--stage1-quick-every", type=int, default=None,
                    help="Override experiment.monitor.stage1_quick_every.")
    ap.add_argument("--stage1-clean-msm-every", type=int, default=None,
                    help="Override experiment.monitor.stage1_clean_msm_every.")
    ap.add_argument("--gene-set-every", type=int, default=None,
                    help="Override experiment.monitor.gene_set_every.")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--image-backbone", choices=["feature", "uni"], default=None)
    ap.add_argument("--uni-tune", default=None,
                    help="none / all / partial:N / lora / adapter")
    ap.add_argument("--tx-warmup-epochs", type=int, default=0,
                    help="Pre-train tx encoder ONLY (align+recon disabled) for N "
                         "epochs before the joint training phase begins.  Only "
                         "meaningful when transcriptomics.kind is hvg_tokenizer or "
                         "top_hvg_gene.")
    ap.add_argument("--stage1-only", action="store_true",
                    help="STAGE 1: train tx encoder ONLY.  Forces align/gene_recon/"
                         "image_recon weights to 0 for the full run, freezes the "
                         "image side, skips the final zero-shot / linear-probe eval, "
                         "and saves a standalone ckpt_tx_encoder.pt for Stage 2 reuse.")
    ap.add_argument("--tx-ckpt", default=None,
                    help="STAGE 2: load tx-encoder weights from this Stage 1 ckpt "
                         "(either ckpt_tx_encoder.pt or any ckpt_last.pt/ckpt_best.pt) "
                         "and FREEZE the tx encoder.  Disables tx_self loss automatically.")
    ap.add_argument("--init-tx-ckpt", default=None,
                    help="STAGE 1/1.25: initialize tx-encoder weights from a checkpoint "
                         "but keep the tx encoder trainable. Useful for MSM -> "
                         "chunk-JEPA sequential ablations.")
    ap.add_argument("--resume-ckpt", default=None,
                    help="Resume a run from ckpt_last.pt/ckpt_epoch*.pt. Restores "
                         "model/objective weights and, for new checkpoints, optimizer/"
                         "scheduler state. Older checkpoints resume weights only.")
    args = ap.parse_args()
    if args.tx_ckpt and args.init_tx_ckpt:
        raise SystemExit("Use either --tx-ckpt (frozen Stage2) or --init-tx-ckpt (trainable init), not both.")
    if args.resume_ckpt and (args.tx_ckpt or args.init_tx_ckpt):
        raise SystemExit("Use --resume-ckpt by itself; do not combine it with --tx-ckpt or --init-tx-ckpt.")

    cfg = load_config([args.data, args.model, args.train, args.experiment])
    set_seed(cfg["train"]["seed"])

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.val_every_epoch is not None:
        cfg["train"]["val_every_epoch"] = int(args.val_every_epoch)
    if args.val_max_batches is not None:
        vcfg_cli = cfg["train"].setdefault("val_loss", {})
        vcfg_cli["max_batches"] = None if int(args.val_max_batches) <= 0 else int(args.val_max_batches)
    monitor_override = cfg.setdefault("experiment", {}).setdefault("monitor", {})
    if args.stage1_quick_every is not None:
        monitor_override["stage1_quick_every"] = int(args.stage1_quick_every)
    if args.stage1_clean_msm_every is not None:
        monitor_override["stage1_clean_msm_every"] = int(args.stage1_clean_msm_every)
    if args.gene_set_every is not None:
        monitor_override["gene_set_every"] = int(args.gene_set_every)
    if args.image_backbone is not None:
        cfg["model"]["image"]["backbone"] = args.image_backbone
        # The CLI alias "uni" maps to the registry-driven "foundation" backbone.
        if args.image_backbone == "uni":
            cfg["model"]["image"]["backbone"] = "foundation"
    if args.uni_tune is not None:
        # Write tune into whichever block exists.  Prefer the new schema
        # `image.foundation.tune`; fall back to legacy `image.uni.tune`.
        img_cfg = cfg["model"]["image"]
        target = (img_cfg.get("foundation")
                  if "foundation" in img_cfg
                  else img_cfg.setdefault("uni", {}))
        target["tune"] = args.uni_tune
        # Ensure both blocks are kept consistent if both happen to exist.
        if "uni" in img_cfg and "foundation" in img_cfg:
            img_cfg["uni"]["tune"] = args.uni_tune

    # ── Stage-1 short-circuit ─────────────────────────────────────────
    # When --stage1-only is set we only train the tx encoder.  Force the
    # image side to use the cheap precomputed-feature MLP path (no UNI
    # forward), and zero out align + both recon losses inside cfg so the
    # objective constructor doesn't even instantiate them.
    if args.stage1_only:
        cfg["model"]["image"]["backbone"] = "feature"
        cfg["experiment"]["align"]["weight"] = 0.0
        cfg["experiment"].setdefault("gene_recon", {})["weight"] = 0.0
        cfg["experiment"].setdefault("image_recon", {})["weight"] = 0.0
        # Make sure the tx self-supervision actually fires.
        tx_self = cfg["experiment"].setdefault("tx_self", {})
        if float(tx_self.get("weight", 0.0)) <= 0:
            tx_self["weight"] = 1.0  # default Stage-1 weight
        # ── Sync tokenizer masking with experiment-level config ──
        # experiment.tx_self.masking_obj (loss-side) + experiment.tokenizer.mask_ratio
        # (data-side) are the single source of truth for Stage 1.  Propagate them
        # to model.transcriptomics.top_hvg_gene so the encoder and the loss agree.
        exp_tok = cfg["experiment"].get("tokenizer", {})
        masking_obj = tx_self.get("masking_obj", "both")
        thg = cfg["model"]["transcriptomics"].setdefault("top_hvg_gene", {})
        if "mask_ratio" in exp_tok:
            thg["mask_ratio"] = float(exp_tok["mask_ratio"])
        # If the loss only uses one objective, prefer masking only that side in
        # the input too.  "both" keeps the legacy behaviour (mask both sides).
        if masking_obj in ("symbol", "value", "both"):
            thg["mask_kind"] = masking_obj
        log.info(f"[STAGE1] masking_obj={masking_obj}  mask_ratio={thg.get('mask_ratio')}  "
                 f"dino={tx_self.get('enable_dino_consistency', False)} "
                 f"jepa={tx_self.get('enable_masked_jepa', True)}  "
                 f"weights: sym={tx_self.get('symbol_weight', 1.0)} "
                 f"val={tx_self.get('value_weight', 0.5)} "
                 f"view_jepa={tx_self.get('view_jepa_weight', 0.0)} "
                 f"view_jepa_warmup={tx_self.get('view_jepa_warmup_epochs', 0)}+{tx_self.get('view_jepa_ramp_epochs', 0)} "
                 f"multi_chunk={tx_self.get('multi_chunk_weight', 0.0)} "
                 f"multi_chunk_warmup={tx_self.get('multi_chunk_warmup_epochs', 0)}+{tx_self.get('multi_chunk_ramp_epochs', 0)} "
                 f"dino={tx_self.get('dino_weight', 0.0)} "
                 f"dino_loss={tx_self.get('dino_loss', 'cosine')} "
                 f"dino_warmup={tx_self.get('dino_warmup_epochs', 0)}+{tx_self.get('dino_ramp_epochs', 0)} "
                 f"koleo={tx_self.get('koleo_weight', 0.0)} "
                 f"koleo_warmup={tx_self.get('koleo_warmup_epochs', 0)}+{tx_self.get('koleo_ramp_epochs', 0)} "
                 f"jepa={tx_self.get('jepa_weight', 1.0)}")
        # Disable post-training cross-modal eval (no alignment to evaluate).
        cfg["train"].setdefault("final_eval", {})
        cfg["train"]["final_eval"]["run_zero_shot"] = False
        cfg["train"]["final_eval"]["run_linear_probe"] = False
        # Stage-1 may include auxiliary losses such as multi-chunk JEPA/VICReg.
        # Do not force best-checkpoint selection onto the composite tx_self
        # loss; keep an explicit config/script choice, and only upgrade legacy
        # val/loss defaults to the primary clean-MSM diagnostic.
        es_cfg = cfg["train"].get("early_stopping", {})
        if es_cfg.get("enable") and es_cfg.get("metric") in (None, "", "val/loss"):
            es_cfg["metric"] = "val/clean_msm/tx_self/masked_symbol_ce"
            es_cfg["mode"] = "min"
        # In Stage 1 the masked-modeling heads (symbol/value/jepa) only fire
        # on masked positions — if a particular rank's batch has all-empty
        # samples (zero-drop) some heads see no input, causing DDP under
        # static_graph=True to error.  Relax DDP for Stage 1.
        cfg["train"]["ddp_static_graph"] = False
        cfg["train"]["find_unused_parameters"] = True

    # ── Stage-2 wiring ────────────────────────────────────────────────
    # When --tx-ckpt is given, we always run the full image+align stack
    # (so image_backbone stays whatever was requested) but freeze the tx
    # encoder and disable tx_self self-supervision.
    if args.tx_ckpt:
        cfg["experiment"].setdefault("tx_self", {})["weight"] = 0.0
        # Tag the tx config so build_tx_encoder sees freeze=true.
        cfg["model"]["transcriptomics"]["freeze"] = True

    # Sync image_mode with the chosen backbone (raw only when needed).
    if cfg["model"]["image"]["backbone"] in ("uni", "foundation"):
        cfg["data"]["image_mode"] = "raw"
    else:
        # feature backbone doesn't need raw images — saves I/O.
        cfg["data"]["image_mode"] = "feature"

    tag = args.tag or f"{cfg['experiment']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg["train"]["output_dir"]) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log.info(f"Run dir: {run_dir}")

    # ── Free perf: TF32 ────────────────────────────────────────────────
    if cfg["train"].get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── Accelerate ─────────────────────────────────────────────────────
    try:
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
        from datetime import timedelta
        pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=cfg["train"].get("find_unused_parameters", False),
            static_graph=cfg["train"].get("ddp_static_graph", True),
        )
        accelerator = Accelerator(
            mixed_precision=cfg["train"]["mixed_precision"],
            gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
            kwargs_handlers=[pg_kwargs, ddp_kwargs],
        )
    except Exception as e:
        log.warning(f"Accelerator init failed ({e}); running single-process.")
        accelerator = None
    device = (accelerator.device if accelerator
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    is_main = (accelerator is None) or accelerator.is_main_process
    log.info(f"Device: {device} | backbone={cfg['model']['image']['backbone']} | "
             f"tune={cfg['model']['image'].get('uni', {}).get('tune', 'n/a')} | "
             f"image_mode={cfg['data']['image_mode']}")

    def _barrier():
        if accelerator is not None:
            accelerator.wait_for_everyone()

    def _empty_cache():
        if cfg["train"].get("empty_cache_after_eval", True):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Datasets ───────────────────────────────────────────────────────
    prepared_dir = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared_dir / "splits.json").read_text())
    train_ids = splits["train"]; val_ids = splits["val"]
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val:
        val_ids = val_ids[: args.limit_val]

    # ── Runtime vocab clip (no re-prepare needed) ────────────────────────
    # Caller may pass `data.vocab_clip.keep_indices_path` (a .npy of int64
    # column indices into the FULL prepared hvg_log) to clip the vocab at
    # training time.  Make a clipped file with scripts/data/make_clipped_vocab.py.
    vocab_keep_indices = None
    _vc_cfg = cfg["data"].get("vocab_clip") or {}
    if _vc_cfg.get("keep_indices_path"):
        vocab_keep_indices = np.load(_vc_cfg["keep_indices_path"]).astype(np.int64)
        log.info(f"vocab_clip: loaded {len(vocab_keep_indices)} keep-indices from "
                 f"{_vc_cfg['keep_indices_path']}")

    # ── Sequence-length sampling (max_seq_len + strategy) ────────────────
    # Token budget cap with optional sampling strategy for the drop set.
    # config (all optional):
    #   data.max_seq_len            int    0 = off, else cap per-spot nonzero
    #   data.sampling.strategy      str    'random' (default) | 'top_k' | 'weighted'
    #   data.sampling.alpha         float  weighted-only sharpness (1.0)
    #   data.sampling.keep_must_include  bool  protect curated markers (default true)
    _samp = cfg["data"].get("sampling") or {}
    sampling_strategy = _samp.get("strategy", "random")
    sampling_alpha = float(_samp.get("alpha", 1.0))
    keep_must = bool(_samp.get("keep_must_include", True))

    # Build the must_include mask (post vocab-clip), only when needed for the
    # sampling cap.  Reads must_include_genes from data.yaml + the on-disk
    # hvg_vocab.json (clipped if vocab_clip is active).
    must_include_mask = None
    if keep_must and int(cfg["data"].get("max_seq_len", 0)) > 0:
        must_genes = {g.upper() for g in (cfg["data"].get("must_include_genes") or [])}
        if must_genes:
            full_vocab = json.loads((Path(cfg["data"]["prepared_dir"]) / "hvg_vocab.json").read_text())
            if vocab_keep_indices is not None:
                eff_vocab = [full_vocab[i] for i in vocab_keep_indices.tolist()]
            else:
                eff_vocab = full_vocab
            must_include_mask = np.array([g.upper() in must_genes for g in eff_vocab], dtype=bool)
            log.info(f"sampling keep_must_include: {must_include_mask.sum()} curated markers will be "
                     f"protected from drop when max_seq_len triggers")

    ds_kwargs = dict(
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
        image_mode=cfg["data"]["image_mode"],
        hest_patch_dir=cfg["data"]["hest_patch_dir"],
        gene_norm_cfg=cfg["data"].get("gene_norm"),
        # Stage 1 skips uni_feat / novae_latent / neighbors I/O (3.6× speedup).
        tx_only=bool(args.stage1_only),
        # Drop spots with fewer than `min_seq_len` expressed HVG (zero-removal
        # makes the token sequence too short for masked modeling — language-
        # model analogy breaks under length < ~30).  Set to 0 to disable.
        min_seq_len=int(cfg["data"].get("min_seq_len", 0)),
        # Cap non-zero positions per spot to bound attention O(L²) memory on
        # outlier shards (TENX HD spots can reach 13K+ expressed genes).  0 = off.
        max_seq_len=int(cfg["data"].get("max_seq_len", 0)),
        vocab_keep_indices=vocab_keep_indices,
        sampling_strategy=sampling_strategy,
        sampling_alpha=sampling_alpha,
        must_include_mask=must_include_mask,
    )
    train_ds = build_dataset_from_split(prepared_dir, train_ids, **ds_kwargs)
    val_ds = build_dataset_from_split(prepared_dir, val_ids, **ds_kwargs)
    log.info(f"Train samples={len(train_ids)} spots={len(train_ds)} | "
             f"Val samples={len(val_ids)} spots={len(val_ds)}")

    # Stage 1 doesn't use spatial neighbors (align_weight=0 → spatial-aux loss off),
    # so skip the prebuild step entirely.  Saves ~20 min on a 1300-shard pool
    # where rank 0 would otherwise hold 7 ranks at the barrier.
    if is_main and not args.stage1_only:
        n_built = prebuild_knn_caches(train_ds) + prebuild_knn_caches(val_ds)
        if n_built:
            log.info(f"Pre-built {n_built} KNN cache(s).")
    elif is_main:
        log.info("[STAGE1] skipping prebuild_knn_caches (neighbors not used).")
    _barrier()

    # ── DataLoaders ─────────────────────────────────────────────────────
    nw = cfg["train"]["num_workers"]
    persistent = cfg["train"]["persistent_workers"] and nw > 0
    pf = cfg["train"].get("prefetch_factor", 4) if nw > 0 else None
    loader_kwargs = dict(num_workers=nw, pin_memory=cfg["train"]["pin_memory"],
                         persistent_workers=persistent, collate_fn=pad_collate)
    if pf:
        loader_kwargs["prefetch_factor"] = pf

    sampler = SampleBlockSampler(train_ds, cfg["train"]["batch_size"],
                                 shuffle=cfg["train"]["shuffle"], seed=cfg["train"]["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              sampler=sampler, drop_last=True, **loader_kwargs)
    # Eval loaders stay un-prepared — only rank 0 touches them.
    eval_val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                                 shuffle=False, **loader_kwargs)

    # ── Model + objective ──────────────────────────────────────────────
    gene_means, gene_stds = _load_gene_stats(cfg["train"].get("gene_stats_path"))

    # Sync model.transcriptomics.hvg_in_dim to actual vocab size.
    # select_global_hvg can append `must_include` curated markers, so the on-
    # disk hvg_vocab.json may be slightly larger than data.n_hvg (e.g. 4096 →
    # 4175).  Without this sync the encoder asserts hvg.shape[1] == n_hvg and
    # crashes at the first batch.
    _vocab_path = Path(cfg["data"]["prepared_dir"]) / "hvg_vocab.json"
    if _vocab_path.exists():
        _real_n = len(json.loads(_vocab_path.read_text()))
        # vocab_clip overrides — model sees only the clipped subset.
        if vocab_keep_indices is not None:
            _real_n = int(len(vocab_keep_indices))
            keep_path = Path(_vc_cfg.get("keep_indices_path", ""))
            # Keep the symbol-head class count aligned with hvg_in_dim.
            # Without this, the input is clipped to e.g. 4096 genes but the CE
            # head still predicts the full 19k vocabulary, making MSM CE scales
            # incomparable and wasting classes that can never be targets.
            if keep_path.name.endswith("_keep_indices.npy"):
                clip_dict = keep_path.with_name(keep_path.name.replace("_keep_indices.npy", "_vocab_dict.json"))
                if clip_dict.exists():
                    thg = cfg["model"]["transcriptomics"].setdefault("top_hvg_gene", {})
                    old_vp = thg.get("vocab_path")
                    thg["vocab_path"] = str(clip_dict)
                    log.info(f"vocab_clip: model top_hvg_gene.vocab_path {old_vp or '<default>'} → {clip_dict}")
                else:
                    log.warning(f"vocab_clip: expected clipped vocab dict at {clip_dict}; "
                                "symbol head may keep full vocab size")
        _cfg_n = cfg["model"]["transcriptomics"].get("hvg_in_dim", _real_n)
        if _cfg_n != _real_n:
            src = "vocab_clip" if vocab_keep_indices is not None else _vocab_path.name
            log.info(f"vocab size sync: model.hvg_in_dim {_cfg_n} → {_real_n} (from {src})")
            cfg["model"]["transcriptomics"]["hvg_in_dim"] = _real_n

    model = MMAligner(cfg).to(device)

    # ── Optional trainable tx-encoder initialization ─────────────────────
    # This is intentionally separate from --tx-ckpt.  --tx-ckpt is Stage-2
    # frozen loading; --init-tx-ckpt is for sequential Stage-1 ablations such
    # as MSM -> chunk-JEPA where the tx encoder should keep receiving grads.
    if args.init_tx_ckpt:
        init_path = Path(args.init_tx_ckpt)
        if not init_path.exists():
            raise SystemExit(f"--init-tx-ckpt path not found: {init_path}")
        st = torch.load(init_path, map_location="cpu")
        if "tx_encoder" in st:
            tx_sd = st["tx_encoder"]
        else:
            full_sd = st.get("model", st)
            tx_sd = {k.removeprefix("tx_encoder."): v
                     for k, v in full_sd.items() if k.startswith("tx_encoder.")}
        missing, unexpected = model.tx_encoder.load_state_dict(tx_sd, strict=False)
        if is_main:
            log.info(f"[INIT] initialized trainable tx encoder from {init_path} "
                     f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ── Stage-1: freeze everything except tx_encoder ────────────────────
    # DDP with static_graph=True rejects parameters that don't receive
    # gradients (image_proj / tx_proj / decoders feed only into align/recon
    # losses, which have weight=0 in Stage 1).  Explicitly drop them from
    # the optimizer + DDP reducer.
    if args.stage1_only:
        for name, p in model.named_parameters():
            if not name.startswith("tx_encoder."):
                p.requires_grad = False
        if is_main:
            n_keep = sum(p.numel() for p in model.tx_encoder.parameters() if p.requires_grad)
            log.info(f"[STAGE1] froze image side + projectors + decoders; "
                     f"tx_encoder trainable params = {n_keep:,}")

    # ── Stage-2: load pretrained tx encoder + FREEZE ────────────────────
    if args.tx_ckpt:
        tx_ckpt_path = Path(args.tx_ckpt)
        if not tx_ckpt_path.exists():
            raise SystemExit(f"--tx-ckpt path not found: {tx_ckpt_path}")
        st = torch.load(tx_ckpt_path, map_location="cpu")
        # Two supported formats:
        #   (a) {"tx_encoder": <sd>}  ← emitted by Stage 1 _save_tx_encoder_only
        #   (b) {"model": <full sd>}   ← classic trainer ckpt
        if "tx_encoder" in st:
            tx_sd = st["tx_encoder"]
            missing, unexpected = model.tx_encoder.load_state_dict(tx_sd, strict=False)
        else:
            full_sd = st.get("model", st)
            tx_sd = {k.removeprefix("tx_encoder."): v
                     for k, v in full_sd.items() if k.startswith("tx_encoder.")}
            missing, unexpected = model.tx_encoder.load_state_dict(tx_sd, strict=False)
        for p in model.tx_encoder.parameters():
            p.requires_grad = False
        model.tx_encoder.eval()
        if is_main:
            log.info(f"[STAGE2] loaded tx encoder from {tx_ckpt_path} "
                     f"(missing={len(missing)}, unexpected={len(unexpected)}); FROZEN")

    # Optional gradient checkpointing on UNI trunk
    if (cfg["model"]["image"]["backbone"] == "uni"
            and cfg["model"]["image"]["uni"].get("tune", "none") not in ("none",)
            and cfg["train"].get("uni_gradient_checkpointing", False)):
        # PEFT wraps the trunk — set_grad_checkpointing lives on the underlying timm model.
        trunk = model.image_encoder.trunk
        base = getattr(trunk, "base_model", None)
        target = base.model if base is not None and hasattr(base, "model") else trunk
        if hasattr(target, "set_grad_checkpointing"):
            target.set_grad_checkpointing(True)
            log.info("UNI trunk gradient checkpointing: ENABLED")

    objective = build_objective(cfg, model, gene_means=gene_means, gene_stds=gene_stds).to(device)

    # ── Stage-1: also freeze any objective-owned params (align predictor etc.) ─
    # In Stage 1 the objective's forward calls model.tx_encoder.symbol_head /
    # value_head / jepa_head directly.  Those weights belong to `model`'s DDP
    # wrapper.  If `objective` is also DDP-wrapped, both reducers attach grad
    # hooks to the same parameter — backward marks it ready twice and DDP errors.
    # By zero-ing objective.requires_grad and skipping its DDP-prepare we keep
    # everything under model's single DDP wrapper.
    if args.stage1_only:
        for name, p in objective.named_parameters():
            # Keep objective-owned view-JEPA predictor trainable.  Shared model
            # params remain under the model DDP wrapper; teacher copies stay frozen.
            p.requires_grad = (
                name.startswith("tx_self.view_jepa_predictor.")
                or name.startswith("tx_self.multi_chunk_predictor.")
                or name.startswith("tx_self.multi_chunk_target_query")
                or name.startswith("tx_self.multi_chunk_target_id_proj.")
            )
        # Stage 1.25 (multi_chunk_jepa refinement) zeroes MSM weights — the
        # symbol/value/masked-JEPA heads on tx_encoder then produce no grad
        # because no loss term backwards through them.  DDP's reducer still
        # expects either a grad or an explicit "unused" mark; find_unused
        # works for params *outside* the autograd graph, not for heads whose
        # forward output happens to be ignored.  Freeze them explicitly so
        # they leave the reducer's param set entirely.
        ts = cfg["experiment"].get("tx_self", {}) or {}
        tok = cfg["experiment"].get("tokenizer", {}) or {}
        mask_ratio = float(tok.get("mask_ratio", 0.0))
        msm_off = (
            mask_ratio <= 0.0
            or (float(ts.get("symbol_weight", 0.0)) <= 0.0
                and float(ts.get("value_weight", 0.0)) <= 0.0)
        )
        jepa_off = (
            not bool(ts.get("enable_masked_jepa", False))
            or float(ts.get("jepa_weight", 0.0)) <= 0.0
        )
        froze_heads = []
        if msm_off and hasattr(model.tx_encoder, "symbol_head"):
            for p in model.tx_encoder.symbol_head.parameters():
                p.requires_grad = False
            froze_heads.append("symbol_head")
        if msm_off and hasattr(model.tx_encoder, "value_head"):
            for p in model.tx_encoder.value_head.parameters():
                p.requires_grad = False
            froze_heads.append("value_head")
        if jepa_off and hasattr(model.tx_encoder, "jepa_head"):
            for p in model.tx_encoder.jepa_head.parameters():
                p.requires_grad = False
            froze_heads.append("jepa_head")
        if froze_heads and is_main:
            log.info(f"[STAGE1] froze unused tx_encoder heads (no loss): "
                     f"{', '.join(froze_heads)}")

    # ── Optimizer (differential LR groups) ─────────────────────────────
    base_lr = cfg["train"]["lr"]
    groups = model.param_groups(base_lr=base_lr, uni_lr_mult=cfg["train"]["uni_lr_multiplier"])
    obj_params = [p for p in objective.parameters() if p.requires_grad]
    if obj_params:
        groups.append({"params": obj_params, "lr": base_lr, "name": "objective"})
    optim = AdamW(groups, lr=base_lr, weight_decay=cfg["train"]["weight_decay"])

    n_train = sum(p.numel() for grp in groups for p in grp["params"])
    n_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in objective.parameters())
    log.info(f"Trainable params: {n_train:,} / {n_total:,} ({100*n_train/max(1,n_total):.2f}%)")
    if hasattr(model.image_encoder, "freeze_stats"):
        fs = model.image_encoder.freeze_stats
        log.info(f"Image encoder: trainable {fs['trainable_params']:,}/{fs['total_params']:,} "
                 f"({100*fs['trainable_frac']:.2f}%) — mode={fs['mode']}")

    if accelerator is not None:
        train_loader = accelerator.prepare(train_loader)

    steps_per_epoch = max(1, len(train_loader))   # per-RANK batches
    total_steps = steps_per_epoch * cfg["train"]["epochs"]   # per-RANK total
    # Warmup priority order (all in PER-RANK step units):
    #   warmup_steps   — absolute # of optimizer steps
    #   warmup_epochs  — fraction of one epoch (or N epochs)
    #   warmup_ratio   — legacy: fraction of total_steps (LEGACY, can over-warmup)
    if cfg["train"].get("warmup_steps") is not None:
        warmup = int(cfg["train"]["warmup_steps"])
        warmup_src = "warmup_steps"
    elif cfg["train"].get("warmup_epochs") is not None:
        warmup = int(round(float(cfg["train"]["warmup_epochs"]) * steps_per_epoch))
        warmup_src = "warmup_epochs"
    else:
        warmup = int(total_steps * cfg["train"].get("warmup_ratio", 0.05))
        warmup_src = "warmup_ratio"
    # Clamp to ≤ 1 epoch so warmup never eats multiple epochs by accident.
    if warmup > steps_per_epoch:
        log.warning(f"warmup={warmup} ({warmup_src}) exceeds 1 epoch ({steps_per_epoch}); "
                    f"clamping to 1 epoch.")
        warmup = steps_per_epoch
    # ── Accelerate scheduler scaling ──
    # accelerator.prepare(scheduler) wraps step() to advance `num_processes`
    # times per call (preserves single-process semantics).  So the INTERNAL
    # scheduler counter advances faster than our per-rank step count.
    # Multiply total_steps and warmup by num_processes so the internal counter
    # has the right limits.
    n_proc = accelerator.num_processes if accelerator is not None else 1
    total_internal = total_steps * n_proc
    warmup_internal = warmup * n_proc
    log.info(f"scheduler: per-rank total={total_steps} ({cfg['train']['epochs']} ep × "
             f"{steps_per_epoch} step/ep)  warmup={warmup} step ({warmup_src})  "
             f"= {100*warmup/max(1,steps_per_epoch):.0f}% of 1 epoch  | "
             f"× num_processes={n_proc} → internal: total={total_internal}, warmup={warmup_internal}")
    scheduler = cosine_warmup_schedule(
        optim, total_internal, warmup_internal,
        min_lr_ratio=cfg["train"].get("min_lr_ratio", 0.1),
    )

    if accelerator is not None:
        model, objective, optim, scheduler = accelerator.prepare(
            model, objective, optim, scheduler)

    resume_epoch = 0
    resume_step = 0
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise SystemExit(f"--resume-ckpt path not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu")
        model_unwrap = accelerator.unwrap_model(model) if accelerator else model
        obj_unwrap_for_load = accelerator.unwrap_model(objective) if accelerator else objective
        missing_m, unexpected_m = model_unwrap.load_state_dict(
            resume_state.get("model", {}), strict=False)
        missing_o, unexpected_o = obj_unwrap_for_load.load_state_dict(
            resume_state.get("objective", {}), strict=False)
        if "optimizer" in resume_state:
            try:
                optim.load_state_dict(resume_state["optimizer"])
            except Exception as e:
                log.warning(f"[RESUME] optimizer state restore failed; continuing with fresh optimizer: {e}")
        else:
            log.warning("[RESUME] checkpoint has no optimizer state; continuing with fresh optimizer.")
        if "scheduler" in resume_state:
            try:
                scheduler.load_state_dict(resume_state["scheduler"])
            except Exception as e:
                log.warning(f"[RESUME] scheduler state restore failed; continuing with fresh scheduler: {e}")
        else:
            log.warning("[RESUME] checkpoint has no scheduler state; continuing with fresh scheduler.")
        resume_epoch = int(resume_state.get("epoch", 0) or 0)
        resume_step = int(resume_state.get("step", resume_epoch * steps_per_epoch) or 0)
        if is_main:
            log.info(f"[RESUME] loaded {resume_path} at epoch={resume_epoch}, step={resume_step} "
                     f"(model missing={len(missing_m)}, unexpected={len(unexpected_m)}; "
                     f"objective missing={len(missing_o)}, unexpected={len(unexpected_o)})")

    # ── WandB (optional) ────────────────────────────────────────────────
    wb = None
    if cfg["train"]["wandb"]["enable"] and is_main:
        try:
            import wandb
            wb = wandb.init(project=cfg["train"]["wandb"]["project"],
                            entity=cfg["train"]["wandb"]["entity"],
                            name=tag, config=cfg)
        except Exception as e:
            log.warning(f"wandb init failed: {e}")

    # ── Early-stopping tracker (rank 0) ─────────────────────────────────
    es_cfg = cfg["train"].get("early_stopping", {"enable": False})
    es_best: float | None = None
    es_bad = 0
    es_stop = False

    train_history: list[dict] = []
    val_history: list[dict] = []
    if args.resume_ckpt and is_main:
        hist_path = run_dir / "history.json"
        val_hist_path = run_dir / "val_history.json"
        if hist_path.exists():
            try:
                train_history = json.loads(hist_path.read_text())
                train_history = [r for r in train_history if int(r.get("step", -1)) <= resume_step]
            except Exception as e:
                log.warning(f"[RESUME] failed to load existing history.json: {e}")
        if val_hist_path.exists():
            try:
                val_history = json.loads(val_hist_path.read_text())
                val_history = [r for r in val_history if int(float(r.get("epoch", -1))) <= resume_epoch]
            except Exception as e:
                log.warning(f"[RESUME] failed to load existing val_history.json: {e}")
    step = resume_step

    # ── Optional tx-only warmup ──
    # When --tx-warmup-epochs > 0, run the first N epochs with align/recon
    # weights forced to 0 so only tx_self losses (masked-symbol CE, masked-
    # value MSE, masked-JEPA) drive learning.  After warmup the original
    # weights are restored.
    obj_unwrap = accelerator.unwrap_model(objective) if accelerator else objective
    saved_align_w   = getattr(obj_unwrap, "align_weight",        1.0)
    saved_gene_w    = getattr(obj_unwrap, "gene_recon_weight",   0.0)
    saved_image_w   = getattr(obj_unwrap, "image_recon_weight",  0.0)
    saved_tx_self_w = getattr(obj_unwrap, "tx_self_weight",      0.0)
    if args.tx_warmup_epochs > 0 and saved_tx_self_w > 0:
        if is_main:
            log.info(f"[WARMUP] tx encoder pre-training for {args.tx_warmup_epochs} epochs "
                     f"(align/recon weights → 0; tx_self → {saved_tx_self_w})")
    start_epoch = resume_epoch + 1
    if start_epoch > cfg["train"]["epochs"] and is_main:
        log.warning(f"[RESUME] start_epoch={start_epoch} exceeds configured epochs="
                    f"{cfg['train']['epochs']}; no training epochs will run.")
    epoch_pbar = tqdm(range(start_epoch, cfg["train"]["epochs"] + 1),
                      desc=f"[{tag}] epochs", disable=not is_main, position=0)

    for epoch in epoch_pbar:
        if hasattr(obj_unwrap, "set_epoch"):
            obj_unwrap.set_epoch(epoch)
        # Toggle weights according to warmup phase.
        in_warmup = (epoch <= args.tx_warmup_epochs) and (saved_tx_self_w > 0)
        obj_unwrap.align_weight = 0.0 if in_warmup else saved_align_w
        obj_unwrap.gene_recon_weight = 0.0 if in_warmup else saved_gene_w
        obj_unwrap.image_recon_weight = 0.0 if in_warmup else saved_image_w
        obj_unwrap.tx_self_weight = saved_tx_self_w if in_warmup else saved_tx_self_w
        if is_main and (epoch == args.tx_warmup_epochs + 1) and args.tx_warmup_epochs > 0:
            log.info(f"[WARMUP] done — restoring align={saved_align_w} "
                     f"gene_recon={saved_gene_w} image_recon={saved_image_w}")

        model.train()
        step_pbar = tqdm(train_loader,
                         desc=f"epoch {epoch}/{cfg['train']['epochs']}",
                         leave=False, disable=not is_main, position=1)
        for batch in step_pbar:
            with (accelerator.accumulate(model) if accelerator else _NullCtx()):
                b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
                out = model(b)
                loss, log_dict = objective(b, out)
                if accelerator is not None:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and cfg["train"]["max_grad_norm"]:
                        accelerator.clip_grad_norm_(
                            [p for grp in groups for p in grp["params"]],
                            cfg["train"]["max_grad_norm"])
                else:
                    loss.backward()
                    if cfg["train"]["max_grad_norm"]:
                        torch.nn.utils.clip_grad_norm_(
                            [p for grp in groups for p in grp["params"]],
                            cfg["train"]["max_grad_norm"])
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                if accelerator is None or accelerator.sync_gradients:
                    (accelerator.unwrap_model(objective) if accelerator else objective).on_after_step()

            cur_loss = float(loss.detach().item())
            if is_main:
                step_pbar.set_postfix(loss=f"{cur_loss:.3f}",
                                      lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % cfg["train"]["log_every"] == 0 and is_main:
                d = {"step": step, "epoch": epoch,
                     "lr": scheduler.get_last_lr()[0],
                     "loss/total": cur_loss}
                for k, v in log_dict.items():
                    d[k] = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                train_history.append(d)
                if wb is not None:
                    wb.log(d, step=step)
            step += 1
        step_pbar.close()

        # ── Val loss (rank 0 only) ─────────────────────────────────────
        vmetrics: dict[str, float] = {}
        vcfg = cfg["train"].get("val_loss", {"enable": False})
        val_every = int(cfg["train"].get("val_every_epoch", 1))
        val_due = (epoch % val_every == 0) or (epoch == cfg["train"]["epochs"])
        if vcfg.get("enable") and is_main and val_due:
            _model_for_eval = accelerator.unwrap_model(model) if accelerator else model
            _obj_for_eval = accelerator.unwrap_model(objective) if accelerator else objective
            _model_for_eval.eval()
            # Stage 1: force the masked-modeling mask to fire in eval too,
            # otherwise tx_self/loss returns 0 and val_history is useless.
            if args.stage1_only:
                _model_for_eval.force_mask_in_eval = True
                if hasattr(_model_for_eval.tx_encoder, "_force_mask_in_eval"):
                    _model_for_eval.tx_encoder._force_mask_in_eval = True
                else:
                    _model_for_eval.tx_encoder._force_mask_in_eval = True
            sums: dict[str, float] = {}
            n_batches = 0
            mb = vcfg.get("max_batches")
            # ── Gene-set monitor (Stage 1 only) ──
            # Configurable cadence: experiment.monitor.gene_set_every (default = 1
            # → every val epoch). The monitor logs co-occurrence PCCs + CLS
            # silhouette for a curated marker panel.  See
            # src/mm_align/evaluation/gene_set_monitor.py for the panel list.
            monitor_cfg = cfg.get("experiment", {}).get("monitor", {})
            monitor_every = int(monitor_cfg.get("gene_set_every", 1))
            run_set_monitor = (
                bool(args.stage1_only)
                and monitor_every > 0
                and (epoch % monitor_every == 0)
            )
            # ── Stage-1 quick ablation monitors ──
            # Primary short-run selection metrics: label-free intrinsic health
            # + frozen HVG linear probe. Organ/source probes stay optional in
            # stage1_bench because organ labels are often too easy/trivial.
            quick_every = int(monitor_cfg.get("stage1_quick_every", 1))
            run_quick = (
                bool(args.stage1_only)
                and quick_every > 0
                and (epoch % quick_every == 0)
            )
            quick_max_spots = int(monitor_cfg.get("stage1_quick_max_spots", 5000))
            quick_n_genes = int(monitor_cfg.get("stage1_quick_n_genes", 256))
            clean_msm_every = int(monitor_cfg.get("stage1_clean_msm_every", 1))
            run_clean_msm = (
                bool(args.stage1_only)
                and clean_msm_every > 0
                and (epoch % clean_msm_every == 0)
            )
            # ── Stage-1 sample-level + spot-level benchmarks ──
            # Cadence: experiment.monitor.stage1_bench_every. Default is now
            # opt-in in configs because organ probe is not a reliable ablation
            # selector for the foundation tx_encoder.
            bench_every = int(monitor_cfg.get("stage1_bench_every", 0))
            run_bench = (
                bool(args.stage1_only)
                and bench_every > 0
                and (epoch % bench_every == 0)
            )
            if is_main and run_set_monitor:
                log.info(f"[gene_set_monitor] active at epoch={epoch} (every {monitor_every})")
            if is_main and run_quick:
                log.info(f"[stage1_quick_eval] active at epoch={epoch} (every {quick_every})")
            if is_main and run_clean_msm:
                log.info(f"[stage1_clean_msm] active at epoch={epoch} (every {clean_msm_every})")
            if is_main and run_bench:
                log.info(f"[stage1_bench] active at epoch={epoch} (every {bench_every})")
            hvg_pool: list[np.ndarray] = []
            cls_pool: list[np.ndarray] = []
            sidx_pool: list[np.ndarray] = []      # for sample-level pooling
            with torch.no_grad():
                # Keys ending with these suffixes get min/max aggregation
                # across batches (not mean).  Mean is wrong for "_min"/"_max".
                _MIN_SUFFIX = ("_min",)
                _MAX_SUFFIX = ("_max",)
                mins: dict[str, float] = {}
                maxs: dict[str, float] = {}
                for i, batch in enumerate(tqdm(eval_val_loader,
                                               desc=f"val[ep{epoch}]",
                                               leave=False, disable=not is_main, position=2)):
                    if mb is not None and i >= mb:
                        break
                    b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                    out = _model_for_eval(b)
                    loss, log_dict = _obj_for_eval(b, out)
                    sums["val/loss"] = sums.get("val/loss", 0.0) + float(loss.detach().item())
                    for k, v in log_dict.items():
                        kv = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                        key = f"val/{k}"
                        if k.endswith(_MIN_SUFFIX):
                            mins[key] = kv if key not in mins else min(mins[key], kv)
                        elif k.endswith(_MAX_SUFFIX):
                            maxs[key] = kv if key not in maxs else max(maxs[key], kv)
                        else:
                            sums[key] = sums.get(key, 0.0) + kv
                    n_batches += 1
                    # Capture pool for gene-set monitor + Stage-1 quick/bench eval.
                    # Quick probe is the primary ablation selector; full bench is
                    # heavier and opt-in.
                    need_pool = run_set_monitor or run_quick or run_bench
                    pool_cap = 20_000 if run_bench else max(5_000, quick_max_spots)
                    if need_pool and sum(p.shape[0] for p in hvg_pool) < pool_cap:
                        hvg_pool.append(b["hvg"].detach().float().cpu().numpy())
                        # Quick/intrinsic probes should score the clean spot
                        # representation. The validation loss pass above uses
                        # force_mask_in_eval=True so MSM metrics are meaningful;
                        # reusing that corrupted h_tx would make linear_probe/hvg
                        # overlap with masked_hvg and understate normal encoding
                        # quality. masked_hvg still runs its own explicit
                        # held-out-gene corruption below.
                        tx_encoder = getattr(_model_for_eval, "tx_encoder", None)
                        if args.stage1_only and tx_encoder is not None:
                            old_force = getattr(tx_encoder, "_force_mask_in_eval", False)
                            tx_encoder._force_mask_in_eval = False
                            try:
                                clean_out = tx_encoder(novae_latent=None, hvg=b["hvg"])
                                cls_pool.append(clean_out["h_tx"].detach().float().cpu().numpy())
                            finally:
                                tx_encoder._force_mask_in_eval = old_force
                        else:
                            cls_pool.append(out["h_tx"].detach().float().cpu().numpy())
                        if "sample_idx" in b:
                            si = b["sample_idx"]
                            sidx_pool.append(si.cpu().numpy() if torch.is_tensor(si)
                                             else np.asarray(si))
                    # Drop the last validation batch/output before optional clean
                    # passes. They can otherwise keep a large transformer batch
                    # alive when multi-chunk validation starts immediately after.
                    try:
                        del b, out, loss, log_dict
                    except Exception:
                        pass
                _empty_cache()
            # Clean MSM diagnostic: value augmentation can make top-k acc
            # incomparable across ablations. Run an extra eval pass with the
            # tx value augmentation disabled and store it under val/clean_msm/*.
            clean_sums: dict[str, float] = {}
            clean_batches = 0
            if run_clean_msm:
                tx_encoder = getattr(_model_for_eval, "tx_encoder", None)
                tx_self_eval = getattr(_obj_for_eval, "tx_self", None)
                old_value_aug = getattr(tx_encoder, "value_aug_mode", None)
                old_unmasked_value_aug = getattr(tx_encoder, "unmasked_value_aug_mode", None)
                # clean_msm is a pure masked-symbol diagnostic. Temporarily
                # disable auxiliary objectives so this pass does not run the
                # expensive multi-chunk/DINO/JEPA forwards and so loss scale is
                # comparable across objective ablations.
                aux_attrs = ("w_jepa", "w_view_jepa", "w_multi_chunk", "w_dino", "w_koleo")
                old_aux = {}
                if tx_self_eval is not None:
                    for attr in aux_attrs:
                        if hasattr(tx_self_eval, attr):
                            old_aux[attr] = getattr(tx_self_eval, attr)
                            setattr(tx_self_eval, attr, 0.0)
                if tx_encoder is not None and old_value_aug is not None:
                    tx_encoder.value_aug_mode = "keep"
                if tx_encoder is not None and old_unmasked_value_aug is not None:
                    tx_encoder.unmasked_value_aug_mode = "keep"
                _empty_cache()
                try:
                    for i, batch in enumerate(tqdm(eval_val_loader,
                                                   desc=f"val-clean-msm[ep{epoch}]",
                                                   leave=False, disable=not is_main, position=2)):
                        if mb is not None and i >= mb:
                            break
                        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                             for k, v in batch.items()}
                        out = _model_for_eval(b)
                        loss, log_dict = _obj_for_eval(b, out)
                        clean_sums["val/clean_msm/loss"] = (
                            clean_sums.get("val/clean_msm/loss", 0.0)
                            + float(loss.detach().item())
                        )
                        for k, v in log_dict.items():
                            if not ("masked_symbol" in k or k == "tx_self/loss"):
                                continue
                            kv = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                            clean_sums[f"val/clean_msm/{k}"] = clean_sums.get(f"val/clean_msm/{k}", 0.0) + kv
                        clean_batches += 1
                        try:
                            del b, out, loss, log_dict
                        except Exception:
                            pass
                finally:
                    if tx_self_eval is not None:
                        for attr, val in old_aux.items():
                            setattr(tx_self_eval, attr, val)
                    if tx_encoder is not None and old_value_aug is not None:
                        tx_encoder.value_aug_mode = old_value_aug
                    if tx_encoder is not None and old_unmasked_value_aug is not None:
                        tx_encoder.unmasked_value_aug_mode = old_unmasked_value_aug
                    _empty_cache()

            _model_for_eval.train()
            if args.stage1_only:
                _model_for_eval.force_mask_in_eval = False
                _model_for_eval.tx_encoder._force_mask_in_eval = False
            if n_batches > 0:
                vmetrics = {k: v / n_batches for k, v in sums.items()}
                if clean_batches > 0:
                    vmetrics.update({k: v / clean_batches for k, v in clean_sums.items()})
                vmetrics.update(mins)            # true min across batches
                vmetrics.update(maxs)            # true max across batches
                vmetrics["epoch"] = float(epoch)

                # Primary Stage-1 quick eval: intrinsic health + short HVG probe.
                if run_quick and hvg_pool:
                    try:
                        from mm_align.evaluation.stage1_benchmarks import (
                            embedding_health_metrics,
                            expression_manifold_metrics,
                            gene_embedding_correlation_alignment,
                            gene_embeddings_from_encoder,
                            hvg_linear_probe,
                            hvg_rank_probe,
                            masked_hvg_linear_probe_from_encoder,
                            chunk_view_embeddings_from_encoder,
                            source_knn_leakage_metrics,
                        )
                        hvg_arr = np.concatenate(hvg_pool, axis=0)
                        cls_arr = np.concatenate(cls_pool, axis=0)
                        seed = int(cfg.get("seed", 0))
                        quick_m = {}
                        quick_m.update(embedding_health_metrics(cls_arr, prefix="intrinsic"))
                        quick_m.update(expression_manifold_metrics(
                            cls_arr, hvg_arr,
                            max_spots=min(quick_max_spots, 2000),
                            k=20,
                            seed=seed,
                            prefix="intrinsic/expression",
                        ))
                        quick_m.update(hvg_linear_probe(
                            cls_arr, hvg_arr,
                            n_targets=quick_n_genes,
                            max_spots=quick_max_spots,
                            seed=seed,
                            prefix="linear_probe/hvg",
                        ))
                        quick_m.update(hvg_rank_probe(
                            cls_arr, hvg_arr,
                            n_targets=quick_n_genes,
                            max_spots=quick_max_spots,
                            seed=seed,
                            prefix="linear_probe/hvg_rank",
                        ))
                        quick_m.update(masked_hvg_linear_probe_from_encoder(
                            _model_for_eval.tx_encoder, hvg_arr,
                            n_targets=quick_n_genes,
                            max_spots=quick_max_spots,
                            seed=seed,
                            device=str(device),
                            prefix="linear_probe/masked_hvg",
                        ))
                        try:
                            chunk_views = chunk_view_embeddings_from_encoder(
                                _model_for_eval.tx_encoder, hvg_arr,
                                n_chunks=4,
                                chunk_len=min(256, hvg_arr.shape[1]),
                                dynamic=True,
                                batch_size=128,
                                max_spots=quick_max_spots,
                                seed=seed,
                                device=str(device),
                            )
                            for rep_name, rep_prefix in (("z_chunk", "chunk_state"), ("z_spot", "spot_state")):
                                z = chunk_views[rep_name]
                                hvg_z = hvg_arr[:z.shape[0]]
                                quick_m.update(embedding_health_metrics(
                                    z, prefix=f"{rep_prefix}/intrinsic"))
                                quick_m.update(expression_manifold_metrics(
                                    z, hvg_z,
                                    max_spots=min(quick_max_spots, 2000),
                                    k=20,
                                    seed=seed,
                                    prefix=f"{rep_prefix}/intrinsic/expression",
                                ))
                                quick_m.update(hvg_linear_probe(
                                    z, hvg_z,
                                    n_targets=quick_n_genes,
                                    max_spots=quick_max_spots,
                                    seed=seed,
                                    prefix=f"{rep_prefix}/linear_probe/hvg",
                                ))
                                quick_m.update(hvg_rank_probe(
                                    z, hvg_z,
                                    n_targets=quick_n_genes,
                                    max_spots=quick_max_spots,
                                    seed=seed,
                                    prefix=f"{rep_prefix}/linear_probe/hvg_rank",
                                ))
                        except Exception as e:
                            log.warning(f"stage1 chunk/spot-state eval failed: {e}", exc_info=True)
                        gene_emb = gene_embeddings_from_encoder(_model_for_eval.tx_encoder)
                        if gene_emb is not None:
                            quick_m.update(gene_embedding_correlation_alignment(
                                hvg_arr, gene_emb,
                                n_genes=min(512, hvg_arr.shape[1]),
                                prefix="intrinsic/gene_embedding",
                            ))
                        if sidx_pool:
                            sidx_arr_q = np.concatenate(sidx_pool, axis=0).astype(np.int64)
                            source_labels = []
                            for s_idx in sidx_arr_q[:cls_arr.shape[0]]:
                                src_path = val_ds.shards[int(s_idx)].path.stem
                                source_labels.append(
                                    "st1k" if src_path.endswith(".st1k")
                                    else "spatialcorpus" if src_path.endswith(".spatialcorpus")
                                    else "hest"
                                )
                            quick_m.update(source_knn_leakage_metrics(
                                cls_arr[:len(source_labels)], np.asarray(source_labels),
                                k=20,
                                max_spots=quick_max_spots,
                                seed=seed,
                                prefix="leakage/source_knn",
                            ))
                        for k_, v_ in quick_m.items():
                            vmetrics[f"val/{k_}"] = v_
                        log.info(f"[stage1_quick_eval] logged {len(quick_m)} keys "
                                 f"(pool: {hvg_arr.shape[0]} spots, targets={quick_n_genes})")
                    except Exception as e:
                        log.warning(f"stage1_quick_eval failed: {e}", exc_info=True)

                # Curated gene-set monitor.
                if run_set_monitor and hvg_pool:
                    try:
                        from mm_align.evaluation.gene_set_monitor import compute_set_metrics
                        hvg_arr = np.concatenate(hvg_pool, axis=0)
                        cls_arr = np.concatenate(cls_pool, axis=0)
                        vocab_path = Path(cfg["data"]["prepared_dir"]) / "hvg_vocab.json"
                        if not vocab_path.exists():
                            log.warning(f"gene_set_monitor: vocab not at {vocab_path}; skipping")
                        else:
                            vocab = json.loads(vocab_path.read_text())
                            # If vocab_clip is active, the hvg arrays only see
                            # the clipped subset of columns — slice the gene
                            # name list to match so panel_indices line up.
                            if vocab_keep_indices is not None:
                                vocab = [vocab[i] for i in vocab_keep_indices.tolist()]
                            set_m = compute_set_metrics(hvg_arr, cls_arr, vocab)
                            for k_, v_ in set_m.items():
                                vmetrics[f"val/{k_}"] = v_
                            log.info(f"[gene_set_monitor] logged {len(set_m)} keys "
                                     f"(pool: {hvg_arr.shape[0]} spots × {hvg_arr.shape[1]} genes)")
                    except Exception as e:
                        log.warning(f"gene-set monitor failed: {e}", exc_info=True)

                # Stage-1 benchmarks: spot-level + sample-level.
                if run_bench and hvg_pool and sidx_pool:
                    try:
                        from mm_align.evaluation.stage1_benchmarks import run_stage1_benchmarks
                        from mm_align.evaluation.labels import hest_metadata, spot_organ_labels
                        cls_arr = np.concatenate(cls_pool, axis=0)
                        sidx_arr = np.concatenate(sidx_pool, axis=0).astype(np.int64)
                        # Labels — organ per spot (from HEST CSV when available);
                        # for ST1K / spatialcorpus shards, the val_ds.shards[i].sample_id
                        # is the slide id which we can look up too.
                        try:
                            meta = hest_metadata()
                            sample_ids = [s.sample_id for s in val_ds.shards]
                            organ_per_spot = spot_organ_labels(sample_ids, sidx_arr, meta)
                        except Exception:
                            organ_per_spot = np.array(["Unknown"] * len(sidx_arr))
                        # Build sample-level organ map by majority vote
                        organ_per_sample = {}
                        source_per_sample = {}
                        for s_idx in np.unique(sidx_arr):
                            sel = sidx_arr == s_idx
                            os_ = organ_per_spot[sel]
                            organ_per_sample[int(s_idx)] = max(set(os_), key=lambda x: (os_ == x).sum())
                            src_path = val_ds.shards[int(s_idx)].path.stem
                            source_per_sample[int(s_idx)] = (
                                "st1k" if src_path.endswith(".st1k")
                                else "spatialcorpus" if src_path.endswith(".spatialcorpus")
                                else "hest"
                            )
                        bench = run_stage1_benchmarks(
                            cls_arr, organ_per_spot, sidx_arr,
                            source_per_sample=source_per_sample,
                            organ_per_sample=organ_per_sample,
                        )
                        for k_, v_ in bench.items():
                            vmetrics[f"val/{k_}"] = v_
                        log.info(f"[stage1_bench] {len(bench)} keys: "
                                 f"sample_acc={bench.get('sample/probe_organ/acc', float('nan')):.3f} "
                                 f"spot_acc={bench.get('spot/probe_organ/acc', float('nan')):.3f} "
                                 f"spot_ari={bench.get('spot/cluster/ari', float('nan')):.3f}")
                    except Exception as e:
                        log.warning(f"stage1_bench failed: {e}", exc_info=True)
                val_history.append(vmetrics)
                (run_dir / "val_history.json").write_text(json.dumps(val_history, indent=2))
                try:
                    _render_val_metric_curves(val_history, run_dir / "metric_curves.png", tag)
                    if args.stage1_only:
                        _render_stage1_metric_curves(val_history, run_dir / "stage1_metric_curves.png", tag)
                except Exception as e:
                    log.warning(f"metric curve rendering failed: {e}", exc_info=True)
                log.info(f"[VAL] epoch={epoch} loss={vmetrics.get('val/loss', float('nan')):.4f} "
                         + _summarize_log(vmetrics))
                if wb is not None:
                    wb.log(vmetrics, step=step)

            # Early stopping check.
            if es_cfg.get("enable") and es_cfg["metric"] in vmetrics:
                cur = vmetrics[es_cfg["metric"]]
                better = (
                    (es_cfg["mode"] == "max" and (es_best is None or cur > es_best + es_cfg["min_delta"]))
                    or (es_cfg["mode"] == "min" and (es_best is None or cur < es_best - es_cfg["min_delta"]))
                )
                if better:
                    es_best = cur
                    es_bad = 0
                    if es_cfg.get("save_best_ckpt", True):
                        state = _build_ckpt_state(model, objective, cfg, epoch, accelerator,
                                                  full=cfg["train"].get("save_full_ckpt", False))
                        state["optimizer"] = optim.state_dict()
                        state["scheduler"] = scheduler.state_dict()
                        state["step"] = step
                        state["best_metric"] = cur
                        state["best_metric_key"] = es_cfg["metric"]
                        torch.save(state, run_dir / "ckpt_best.pt")
                        log.info(f"[ES] ↑ improved {es_cfg['metric']}={cur:.4f} → saved ckpt_best.pt")
                        if args.stage1_only:
                            _save_tx_encoder_only(model, cfg, epoch, accelerator,
                                                  run_dir, basename="ckpt_tx_encoder_best.pt")
                else:
                    es_bad += 1
                    log.info(f"[ES] no improvement ({es_bad}/{es_cfg['patience']}); "
                             f"best {es_cfg['metric']}={es_best:.4f}")
                    if es_bad >= es_cfg["patience"]:
                        log.info("[ES] PATIENCE EXHAUSTED — stopping training.")
                        es_stop = True
        _empty_cache(); _barrier()

        # ── Broadcast stop flag ────────────────────────────────────────
        if accelerator is not None:
            stop_t = torch.tensor([1 if es_stop else 0], device=device)
            stop_t = accelerator.gather(stop_t).max()
            es_stop = bool(stop_t.item())

        # ── Periodic ckpt save + prune ─────────────────────────────────
        if (epoch % cfg["train"]["save_every_epoch"] == 0) and is_main:
            state = _build_ckpt_state(model, objective, cfg, epoch, accelerator,
                                      full=cfg["train"].get("save_full_ckpt", False))
            state["optimizer"] = optim.state_dict()
            state["scheduler"] = scheduler.state_dict()
            state["step"] = step
            torch.save(state, run_dir / f"ckpt_epoch{epoch}.pt")
            torch.save(state, run_dir / "ckpt_last.pt")
            _prune_old_ckpts(run_dir, cfg["train"].get("keep_last_n_ckpts", 3))
            if args.stage1_only:
                _save_tx_encoder_only(model, cfg, epoch, accelerator, run_dir,
                                       basename="ckpt_tx_encoder_last.pt")
        _empty_cache(); _barrier()

        if es_stop:
            break

    # ── Final loss curve + history dump ────────────────────────────────
    if is_main:
        (run_dir / "history.json").write_text(json.dumps(train_history, indent=2))
        _render_loss_curve(train_history, run_dir / "loss_curve.png", tag)
        _render_val_metric_curves(val_history, run_dir / "metric_curves.png", tag)
        if args.stage1_only:
            _render_stage1_metric_curves(val_history, run_dir / "stage1_metric_curves.png", tag)
        if args.stage1_only:
            # Promote the best-by-val tx encoder as the canonical Stage-2 input.
            best_src = run_dir / "ckpt_tx_encoder_best.pt"
            last_src = run_dir / "ckpt_tx_encoder_last.pt"
            canonical = run_dir / "ckpt_tx_encoder.pt"
            import shutil
            if best_src.exists():
                shutil.copy2(best_src, canonical)
                log.info(f"[STAGE1] canonical tx encoder → {canonical} (from best)")
            elif last_src.exists():
                shutil.copy2(last_src, canonical)
                log.info(f"[STAGE1] canonical tx encoder → {canonical} (from last)")

    # ── End-of-training final eval (single pass, NOT per-epoch) ────────
    fe = cfg["train"].get("final_eval", {})
    if is_main:
        import subprocess
        if args.stage1_only and fe.get("run_stage1_test", True):
            tx_ckpt = run_dir / "ckpt_tx_encoder_best.pt"
            if not tx_ckpt.exists():
                tx_ckpt = run_dir / "ckpt_tx_encoder_last.pt"
            if not tx_ckpt.exists():
                tx_ckpt = run_dir / "ckpt_tx_encoder.pt"
            if tx_ckpt.exists():
                out_global = Path(fe.get("stage1_test_out", "results/eval")) / f"stage1_test_{tag}.csv"
                out_run = run_dir / "test_stage1.csv"
                cmd = [
                    sys.executable, str(Path(__file__).parent / "eval" / "stage1_tx.py"),
                    "--prepared-dir", str(cfg["data"]["prepared_dir"]),
                    "--split", str(fe.get("stage1_test_split", "test")),
                    "--ckpts", str(tx_ckpt),
                    "--val-samples", str(int(fe.get("stage1_test_samples", 76))),
                    "--pool-spots", str(int(fe.get("stage1_test_pool_spots", 8000))),
                    "--linear-probe-genes", str(int(fe.get("stage1_test_linear_probe_genes", 256))),
                    "--out", str(out_global),
                ]
                if bool(fe.get("stage1_test_include_organ_probe", False)):
                    cmd.append("--include-organ-probe")
                if bool(fe.get("stage1_test_skip_source_probe", False)):
                    cmd.append("--skip-source-probe")
                if bool(fe.get("stage1_test_make_viz", False)):
                    cmd.append("--make-viz")
                    cmd.extend(["--viz-out-dir", str(run_dir / "test_stage1_viz")])
                    cmd.extend(["--viz-max-spots", str(int(fe.get("stage1_test_viz_max_spots", 5000)))])
                    viz_genes = fe.get("stage1_test_viz_genes", [])
                    if viz_genes:
                        cmd.append("--viz-genes")
                        cmd.extend([str(g) for g in viz_genes])
                log.info(f"[FINAL][stage1-test] {' '.join(cmd)}")
                rc = subprocess.run(cmd, check=False).returncode
                if rc == 0 and out_global.exists():
                    import shutil
                    shutil.copy2(out_global, out_run)
                    log.info(f"[FINAL][stage1-test] saved {out_global} and {out_run}")
                else:
                    log.warning(f"[FINAL][stage1-test] failed with returncode={rc}")
            else:
                log.warning("[FINAL][stage1-test] no tx encoder checkpoint found; skipping")

        if fe.get("run_zero_shot") or fe.get("run_linear_probe"):
            ckpt = run_dir / "ckpt_best.pt"
            if not ckpt.exists():
                ckpt = run_dir / "ckpt_last.pt"
            if ckpt.exists():
                if fe.get("run_zero_shot"):
                    log.info("[FINAL] zero-shot eval ...")
                    subprocess.run([sys.executable, str(Path(__file__).parent / "eval_zero_shot.py"),
                                    "--experiment", str(args.experiment),
                                    "--ckpt", str(ckpt), "--split", "test"], check=False)
                if fe.get("run_linear_probe"):
                    log.info("[FINAL] linear-probe eval ...")
                    subprocess.run([sys.executable, str(Path(__file__).parent / "eval_linearprobe.py"),
                                    "--experiment", str(args.experiment),
                                    "--ckpt", str(ckpt), "--eval_split", "test"], check=False)
                # Always render report figures from whatever JSONs are present.
                log.info("[FINAL] rendering report figures ...")
                subprocess.run([sys.executable, str(Path(__file__).parent / "make_figures.py"),
                                "--run", str(run_dir)], check=False)

    _empty_cache(); _barrier()
    if is_main:
        log.info(f"Done. Artifacts under {run_dir}")
        if wb is not None:
            wb.finish()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
