#!/usr/bin/env python
"""Probe the largest stable Stage-1 per-rank batch for an ablation profile.

This intentionally runs only one forward/backward/optimizer step per trial.
It is meant to answer "what ABL_BATCH can this profile tolerate?" without
spending a full training run.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mm_align.data import build_dataset_from_split, pad_collate
from mm_align.models import MMAligner
from mm_align.objectives import build_objective
from mm_align.training import SampleBlockSampler, load_gene_stats
from mm_align.utils import load_config


BASE_DATA = "configs/stage1/data.yaml"
BASE_MODEL = "configs/stage1/model.yaml"
BASE_TRAIN = "configs/stage1/train.yaml"
BASE_EXP = "configs/stage1/experiment.yaml"


def setp(d: dict[str, Any], dotted: str, value: Any) -> None:
    cur = d
    keys = dotted.split(".")
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def ensure_clip(prepared_dir: Path, n: int) -> Path:
    keep = prepared_dir / f"clip{n}_keep_indices.npy"
    if not keep.exists():
        raise SystemExit(
            f"Missing {keep}. Build it first with: "
            f"python scripts/data/make_clipped_vocab.py --top-k {n}"
        )
    return keep


def apply_capacity(cfg: dict[str, Any], capacity: str) -> None:
    profiles = {
        "spatula_lite": dict(tx_hidden=1024, tok_dim=256, tok_layers=4, tok_heads=4, proj_hidden=1024),
        "spatula_mid": dict(tx_hidden=1536, tok_dim=384, tok_layers=6, tok_heads=6, proj_hidden=1536),
        "spatula_large": dict(tx_hidden=2048, tok_dim=512, tok_layers=6, tok_heads=8, proj_hidden=2048),
    }
    if capacity not in profiles:
        raise SystemExit(f"Unknown capacity={capacity}; use {sorted(profiles)}")
    p = profiles[capacity]
    setp(cfg, "model.transcriptomics.hidden_dim", p["tx_hidden"])
    setp(cfg, "model.transcriptomics.n_layers", 2)
    setp(cfg, "model.transcriptomics.top_hvg_gene.dim", p["tok_dim"])
    setp(cfg, "model.transcriptomics.top_hvg_gene.n_layers", p["tok_layers"])
    setp(cfg, "model.transcriptomics.top_hvg_gene.n_heads", p["tok_heads"])
    setp(cfg, "model.projector.hidden_dim", p["proj_hidden"])


def apply_value_aug(cfg: dict[str, Any], value_aug: str) -> None:
    if value_aug == "keep":
        masked = dict(mode="keep", keep_p=1.0, noise_p=0.0, drop_p=0.0, noise_std=1.0)
        unmasked = dict(mode="keep", keep_p=1.0, noise_p=0.0, drop_p=0.0, noise_std=0.15)
    elif value_aug == "mixed":
        masked = dict(mode="mixed", keep_p=0.75, noise_p=0.15, drop_p=0.10, noise_std=0.35)
        unmasked = dict(mode="mixed", keep_p=0.90, noise_p=0.10, drop_p=0.00, noise_std=0.15)
    else:
        raise SystemExit("value_aug must be keep or mixed")
    for prefix, values in (
        ("model.transcriptomics.top_hvg_gene.value_aug", masked),
        ("model.transcriptomics.top_hvg_gene.masked_value_aug", masked),
        ("model.transcriptomics.top_hvg_gene.unmasked_value_aug", unmasked),
    ):
        for k, v in values.items():
            setp(cfg, f"{prefix}.{k}", v)


def apply_objective(cfg: dict[str, Any], objective: str) -> None:
    tx = cfg.setdefault("experiment", {}).setdefault("tx_self", {})
    tx.update(
        {
            "masking_obj": "symbol",
            "symbol_weight": 1.0,
            "value_weight": 0.0,
            "enable_masked_jepa": False,
            "jepa_weight": 0.0,
            "enable_dino_consistency": False,
            "dino_weight": 0.0,
            "koleo_weight": 0.0,
            "enable_view_jepa": False,
            "view_jepa_weight": 0.0,
            "enable_multi_chunk_jepa": False,
            "multi_chunk_weight": 0.0,
        }
    )
    if objective == "msm_only":
        return
    if objective == "msm_multi_chunk":
        tx.update(
            {
                "enable_multi_chunk_jepa": True,
                "multi_chunk_weight": 0.10,
                "multi_chunk_n_chunks": 4,
                "multi_chunk_len": 256,
                "multi_chunk_loss": "smooth_l1",
                "multi_chunk_target": "target_chunk",
                "multi_chunk_dynamic": True,
                "multi_chunk_target_chunks": 2,
                "multi_chunk_hidden_dim": 1024,
                "multi_chunk_warmup_epochs": 5,
                "multi_chunk_ramp_epochs": 5,
            }
        )
        return
    raise SystemExit("objective must be msm_only or msm_multi_chunk")


def build_cfg(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    cfg = load_config([BASE_DATA, BASE_MODEL, BASE_TRAIN, BASE_EXP])
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["num_workers"] = int(args.num_workers)
    cfg["train"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg["train"]["prefetch_factor"] = 2
    cfg["train"]["epochs"] = 1
    cfg["model"]["image"]["backbone"] = "feature"
    cfg["data"]["image_mode"] = "feature"
    cfg["experiment"]["align"]["weight"] = 0.0
    cfg["experiment"].setdefault("gene_recon", {})["weight"] = 0.0
    cfg["experiment"].setdefault("image_recon", {})["weight"] = 0.0
    cfg["experiment"].setdefault("tokenizer", {})["mask_ratio"] = float(args.mask_ratio)
    cfg["experiment"].setdefault("monitor", {})["stage1_bench_every"] = 0
    cfg["experiment"]["tx_self"]["weight"] = 1.0
    cfg["train"]["final_eval"]["run_stage1_test"] = False
    cfg["train"]["final_eval"]["run_zero_shot"] = False
    cfg["train"]["final_eval"]["run_linear_probe"] = False
    cfg["train"]["ddp_static_graph"] = False
    cfg["train"]["find_unused_parameters"] = True

    setp(cfg, "data.gene_norm.mode", args.norm)
    if args.vocab == "full":
        setp(cfg, "data.vocab_clip", None)
    else:
        keep = ensure_clip(Path(cfg["data"]["prepared_dir"]), int(args.vocab))
        setp(cfg, "data.vocab_clip.keep_indices_path", str(keep))
    setp(cfg, "data.max_seq_len", int(args.seq_len))
    setp(cfg, "data.sampling.strategy", "random")
    setp(cfg, "data.sampling.keep_must_include", True)
    setp(cfg, "data.sampling.alpha", 1.0)

    apply_capacity(cfg, args.capacity)
    apply_value_aug(cfg, args.value_aug)
    apply_objective(cfg, args.objective)
    return cfg


def stage1_setup(cfg: dict[str, Any], limit_samples: int, device: torch.device):
    prepared_dir = Path(cfg["data"]["prepared_dir"])
    splits = json.loads((prepared_dir / "splits.json").read_text())
    train_ids = splits["train"][:limit_samples]

    vocab_keep_indices = None
    vc = cfg["data"].get("vocab_clip") or {}
    if vc.get("keep_indices_path"):
        vocab_keep_indices = np.load(vc["keep_indices_path"]).astype(np.int64)
        keep_path = Path(vc["keep_indices_path"])
        clip_dict = keep_path.with_name(keep_path.name.replace("_keep_indices.npy", "_vocab_dict.json"))
        if clip_dict.exists():
            cfg["model"]["transcriptomics"].setdefault("top_hvg_gene", {})["vocab_path"] = str(clip_dict)

    must_include_mask = None
    samp = cfg["data"].get("sampling") or {}
    if bool(samp.get("keep_must_include", True)) and int(cfg["data"].get("max_seq_len", 0)) > 0:
        must_genes = {g.upper() for g in (cfg["data"].get("must_include_genes") or [])}
        if must_genes:
            full_vocab = json.loads((prepared_dir / "hvg_vocab.json").read_text())
            eff_vocab = [full_vocab[i] for i in vocab_keep_indices.tolist()] if vocab_keep_indices is not None else full_vocab
            must_include_mask = np.array([g.upper() in must_genes for g in eff_vocab], dtype=bool)

    ds = build_dataset_from_split(
        prepared_dir,
        train_ids,
        k_spatial=cfg["data"]["k_spatial"],
        load_hvg=cfg["model"]["transcriptomics"]["use_hvg"],
        image_mode=cfg["data"]["image_mode"],
        hest_patch_dir=cfg["data"]["hest_patch_dir"],
        gene_norm_cfg=cfg["data"].get("gene_norm"),
        tx_only=True,
        min_seq_len=int(cfg["data"].get("min_seq_len", 0)),
        max_seq_len=int(cfg["data"].get("max_seq_len", 0)),
        vocab_keep_indices=vocab_keep_indices,
        sampling_strategy=(cfg["data"].get("sampling") or {}).get("strategy", "random"),
        sampling_alpha=float((cfg["data"].get("sampling") or {}).get("alpha", 1.0)),
        must_include_mask=must_include_mask,
    )

    real_n = len(json.loads((prepared_dir / "hvg_vocab.json").read_text()))
    if vocab_keep_indices is not None:
        real_n = int(len(vocab_keep_indices))
    cfg["model"]["transcriptomics"]["hvg_in_dim"] = real_n

    model = MMAligner(cfg).to(device)
    for name, p in model.named_parameters():
        if not name.startswith("tx_encoder."):
            p.requires_grad = False

    gene_means, gene_stds = load_gene_stats(cfg["train"].get("gene_stats_path"))
    objective = build_objective(cfg, model, gene_means=gene_means, gene_stds=gene_stds).to(device)
    for name, p in objective.named_parameters():
        p.requires_grad = (
            name.startswith("tx_self.view_jepa_predictor.")
            or name.startswith("tx_self.multi_chunk_predictor.")
        )
    params = [p for p in model.parameters() if p.requires_grad] + [p for p in objective.parameters() if p.requires_grad]
    optim = AdamW(params, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    return ds, model, objective, optim


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def try_batch(args: argparse.Namespace, batch_size: int) -> tuple[bool, str, float]:
    clear_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        cfg = build_cfg(args, batch_size)
        ds, model, objective, optim = stage1_setup(cfg, args.limit_samples, device)
        sampler = SampleBlockSampler(ds, batch_size, shuffle=True, seed=cfg["train"]["seed"])
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.num_workers) > 0),
            collate_fn=pad_collate,
        )
        batch = next(iter(loader))
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model.train()
        objective.train()
        objective.set_epoch(int(args.epoch_for_probe))
        optim.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda" and args.mixed_precision != "no"
        amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(batch)
            loss, _ = objective(batch, out)
        loss.backward()
        optim.step()
        objective.on_after_step()
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        del batch, loader, sampler, optim, objective, model, ds, cfg, out, loss
        clear_cuda()
        return True, "ok", peak_gb
    except torch.cuda.OutOfMemoryError as exc:
        clear_cuda()
        return False, f"cuda_oom: {str(exc).splitlines()[0][:180]}", 0.0
    except RuntimeError as exc:
        msg = str(exc)
        clear_cuda()
        if "out of memory" in msg.lower():
            return False, f"runtime_oom: {msg.splitlines()[0][:180]}", 0.0
        return False, f"runtime_error: {msg.splitlines()[0][:180]}", 0.0


def parse_batches(raw: str) -> list[int]:
    vals = sorted({int(x) for x in raw.replace(",", " ").split() if x.strip()})
    if not vals:
        raise SystemExit("--batches is empty")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["msm_only", "msm_multi_chunk"], default="msm_multi_chunk")
    ap.add_argument("--capacity", choices=["spatula_lite", "spatula_mid", "spatula_large"], default="spatula_mid")
    ap.add_argument("--vocab", default="4096", help="4096, 8192, or full")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--value-aug", choices=["keep", "mixed"], default="mixed")
    ap.add_argument("--norm", default="global_median")
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--batches", default="512 768 1024 1280 1536 1792 2048")
    ap.add_argument("--limit-samples", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    ap.add_argument("--epoch-for-probe", type=int, default=10,
                    help="Use a post-warmup epoch so multi-chunk loss is active.")
    ap.add_argument("--binary", action="store_true",
                    help="After the largest listed passing batch, binary-search up to --max-batch.")
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument("--out", default="results/eval/stage1_batch_probe.csv")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    batches = parse_batches(args.batches)
    rows: list[dict[str, Any]] = []
    best = 0
    best_peak = 0.0
    first_fail = None
    profile = {
        "objective": args.objective,
        "capacity": args.capacity,
        "vocab": args.vocab,
        "seq_len": args.seq_len,
        "value_aug": args.value_aug,
        "norm": args.norm,
        "mask_ratio": args.mask_ratio,
    }
    print(f"[probe] profile={profile}")
    for b in batches:
        ok, reason, peak = try_batch(args, b)
        print(f"[probe] batch={b:<5} ok={ok} peak_gb={peak:.2f} reason={reason}")
        rows.append({**profile, "batch": b, "ok": ok, "peak_gb": round(peak, 4), "reason": reason})
        if ok:
            best = b
            best_peak = peak
        elif first_fail is None:
            first_fail = b
            break

    if args.binary and best > 0:
        lo = best + 1
        hi = min(args.max_batch, (first_fail - 1) if first_fail else args.max_batch)
        while lo <= hi:
            mid = ((lo + hi) // 2) // 8 * 8
            mid = max(mid, lo)
            ok, reason, peak = try_batch(args, mid)
            print(f"[probe] batch={mid:<5} ok={ok} peak_gb={peak:.2f} reason={reason}")
            rows.append({**profile, "batch": mid, "ok": ok, "peak_gb": round(peak, 4), "reason": reason})
            if ok:
                best = mid
                best_peak = peak
                lo = mid + 8
            else:
                hi = mid - 8

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[probe] best_batch={best} peak_gb={best_peak:.2f}")
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
