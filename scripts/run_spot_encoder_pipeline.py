#!/usr/bin/env python3
"""One-command spot-encoder pipeline: Stage 1 -> Stage 1.5.

This is an orchestrator, not a second trainer. It reuses the repository's
existing, tested training/evaluation entrypoints and keeps checkpoint boundaries
explicit:

  joint/default:
    Stage 1    = MSM curriculum + multi-chunk JEPA in one trainer
    Stage 1.5  = spatial JEPA from the Stage 1 best tx checkpoint

The old sequential Stage 1.25 path is retained only as an explicit deprecated
control. The default project pipeline no longer runs it.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import re
import shlex
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, dry_run: bool = False) -> None:
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items() if v is not None})
    printable = " ".join(shlex.quote(x) for x in cmd)
    print("\n" + "=" * 88)
    print(printable)
    print("=" * 88, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd or _repo_root()), env=merged, check=True)


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0




def _slug(value: object) -> str:
    text = str(value).strip().replace(".", "p").replace(",", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "x"


def _set_attr(args: argparse.Namespace, key: str, value: str) -> None:
    attr = key.replace("-", "_")
    if not hasattr(args, attr):
        raise SystemExit(f"Unknown sweep key {key!r}. Use --help to see supported arguments.")
    setattr(args, attr, value)


def _parse_sweep_specs(specs: list[str]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --sweep {spec!r}; expected key=a,b,c")
        key, vals = spec.split("=", 1)
        values = [v.strip() for v in vals.split(",") if v.strip()]
        if not key.strip() or not values:
            raise SystemExit(f"Invalid --sweep {spec!r}; expected key=a,b,c")
        out.append((key.strip().replace("-", "_"), values))
    return out


def _grid_specs(name: str) -> list[tuple[str, list[str]]]:
    if name == "none":
        return []
    if name == "stage1_core":
        return [
            # Primary first, MSM-only as the control.
            ("stage1_objective", ["msm_multi_chunk", "msm_only"]),
            ("capacity", ["spatula_mid"]),
        ]
    if name == "joint_chunk":
        return [("stage1_objective", ["msm_multi_chunk"])]
    if name == "value_aug":
        # Ablate the masked-position value augmentation while keeping
        # everything else fixed:
        #   keep_only   — pure MASK, no value perturbation at all
        #   dropout     — paper-aligned default (Sinha et al. 2021):
        #                 85% keep + 15% token-level dropout, no noise
        #   with_noise  — 85/10/5 + noise_std=0.15  (does Gaussian help?)
        #   aggressive  — legacy 75/15/10 + noise_std=0.35  (control)
        return [("masked_value_aug_profile",
                  ["keep_only", "dropout", "with_noise", "aggressive"])]
    if name == "stage1_vocab":
        return [("stage1_vocab", ["4096", "8192", "full"])]
    if name == "stage1_capacity":
        return [("capacity", ["spatula_lite", "spatula_mid", "spatula_large"])]
    if name == "sequential_core":
        return [
            ("pipeline", ["sequential"]),
            ("stage125_mc_weight", ["0.10", "0.30"]),
            ("stage125_target_chunks", ["auto"]),
        ]
    if name == "stage15_spatial":
        return [
            ("stage15_mask_target", ["spot", "region"]),
            ("stage15_region_tx_agg", ["mean", "weighted"]),
        ]
    raise SystemExit(f"Unknown --grid {name!r}")


def _expanded_runs(base_args: argparse.Namespace) -> list[tuple[argparse.Namespace, str]]:
    specs = _grid_specs(base_args.grid) + _parse_sweep_specs(base_args.sweep or [])
    if not specs:
        return [(base_args, "")]
    keys = [k for k, _ in specs]
    values = [v for _, v in specs]
    runs: list[tuple[argparse.Namespace, str]] = []
    for combo in itertools.product(*values):
        a = copy.deepcopy(base_args)
        parts = []
        for key, val in zip(keys, combo):
            _set_attr(a, key, val)
            parts.append(f"{key}-{_slug(val)}")
        runs.append((a, "__" + "__".join(parts)))
    return runs


def _prepare_tags(args: argparse.Namespace, suffix: str) -> None:
    if args.stage1_ckpt and not args._stage1_tag_user:
        # Existing Stage1 checkpoint is the source for all downstream spatial
        # variants. Keep its tag stable instead of inventing per-spatial Stage1
        # tags that would imply retraining.
        args.stage1_tag = Path(args.stage1_ckpt).parent.name
    elif not args._stage1_tag_user:
        if args.pipeline == "sequential":
            args.stage1_tag = f"stage1_pipe_msm_v{args.stage1_vocab}_{args.capacity}{suffix}"
        else:
            args.stage1_tag = f"stage1_pipe_{args.stage1_objective}_v{args.stage1_vocab}_{args.capacity}{suffix}"
    elif suffix and args.append_sweep_suffix and not args.stage1_ckpt:
        args.stage1_tag = f"{args.stage1_tag}{suffix}"
    if not args._stage125_tag_user:
        args.stage125_tag = f"stage125_pipe_chunk_jepa_{args.capacity}{suffix}"
    elif suffix and args.append_sweep_suffix:
        args.stage125_tag = f"{args.stage125_tag}{suffix}"
    if not args._stage15_tag_user:
        source = args.stage125_tag if args.pipeline == "sequential" else args.stage1_tag
        mt = _slug(args.stage15_mask_target)
        agg = _slug(args.stage15_region_tx_agg)
        args.stage15_tag = f"stage15_pipe_spatial_{mt}_{agg}_from_{source}"
    elif suffix and args.append_sweep_suffix:
        args.stage15_tag = f"{args.stage15_tag}{suffix}"
    if not args._eval_prefix_user:
        args.eval_prefix = f"spot_encoder_pipeline{suffix}" if suffix else "spot_encoder_pipeline"
    elif suffix and args.append_sweep_suffix:
        args.eval_prefix = f"{args.eval_prefix}{suffix}"


def _maybe_train_stage1(args, root: Path) -> Path:
    if args.stage1_ckpt:
        ckpt = Path(args.stage1_ckpt)
        if args.dry_run:
            print(f"[pipeline] Stage 1 ckpt supplied, skip training: {ckpt}")
            return ckpt
        if not _exists(ckpt):
            raise FileNotFoundError(f"--stage1-ckpt was supplied but does not exist: {ckpt}")
        print(f"[pipeline] Stage 1 ckpt supplied, skip training: {ckpt}")
        return ckpt

    ckpt = root / "results" / "runs" / args.stage1_tag / "ckpt_tx_encoder_best.pt"
    if _exists(ckpt) and not args.force_stage1:
        print(f"[pipeline] Stage 1 ckpt exists, skip training: {ckpt}")
        return ckpt
    env = {
        "STAGE1_TAG": args.stage1_tag,
        "STAGE1_OBJECTIVE": args.stage1_objective,
        "STAGE1_CAPACITY": args.capacity,
        "STAGE1_EPOCHS": args.stage1_epochs,
        "NUM_PROC_STAGE1": args.num_proc_stage1,
        "STAGE1_MAX_SEQ_LEN": args.stage1_max_seq_len,
        "STAGE1_VOCAB": args.stage1_vocab,
        "STAGE1_LIMIT_TRAIN": args.stage1_limit_train,
        "STAGE1_RESUME_CKPT": args.stage1_resume_ckpt,
        "SKIP_STAGE15": "1",
        "SKIP_EVAL": "1",
    }
    if args.stage1_batch:
        env["STAGE1_BATCH"] = args.stage1_batch
    # masked_value_aug_profile -> concrete env knobs that run_all.sh reads.
    # 'dropout' (default) follows the paper-aligned recipe: NO noise, mild
    # token-level dropout only.  'keep_only' and 'with_noise' are ablation
    # controls.  The previously named 'mild' alias keeps backward compat.
    prof = getattr(args, "masked_value_aug_profile", "dropout")
    if prof == "keep_only":
        env["STAGE1_MASKED_AUG_MODE"] = "keep"
        env["STAGE1_MASKED_KEEP_P"] = "1.0"
        env["STAGE1_MASKED_NOISE_P"] = "0.0"
        env["STAGE1_MASKED_DROP_P"] = "0.0"
        env["STAGE1_MASKED_NOISE_STD"] = "0.0"
    elif prof == "with_noise":
        # Used to test whether Gaussian noise on values HURTS, as the paper
        # would predict (value noise warps gene-value co-occurrence).
        env["STAGE1_MASKED_AUG_MODE"] = "mixed"
        env["STAGE1_MASKED_KEEP_P"] = "0.85"
        env["STAGE1_MASKED_NOISE_P"] = "0.10"
        env["STAGE1_MASKED_DROP_P"] = "0.05"
        env["STAGE1_MASKED_NOISE_STD"] = "0.15"
    elif prof == "aggressive":
        env["STAGE1_MASKED_AUG_MODE"] = "mixed"
        env["STAGE1_MASKED_KEEP_P"] = "0.75"
        env["STAGE1_MASKED_NOISE_P"] = "0.15"
        env["STAGE1_MASKED_DROP_P"] = "0.10"
        env["STAGE1_MASKED_NOISE_STD"] = "0.35"
    else:  # "dropout" (default) or legacy "mild" alias
        env["STAGE1_MASKED_AUG_MODE"] = "mixed"
        env["STAGE1_MASKED_KEEP_P"] = "0.85"
        env["STAGE1_MASKED_NOISE_P"] = "0.0"
        env["STAGE1_MASKED_DROP_P"] = "0.15"
        env["STAGE1_MASKED_NOISE_STD"] = "0.0"
    _run(["bash", "scripts/run_all.sh"], env=env, cwd=root, dry_run=args.dry_run)
    if args.dry_run:
        return ckpt
    if not _exists(ckpt):
        raise FileNotFoundError(f"Stage 1 best checkpoint not found after training: {ckpt}")
    return ckpt


def _maybe_train_stage125(args, root: Path, stage1_ckpt: Path) -> Path:
    ckpt = root / "results" / "runs" / args.stage125_tag / "ckpt_tx_encoder_best.pt"
    if _exists(ckpt) and not args.force_stage125:
        print(f"[pipeline] Stage 1.25 ckpt exists, skip training: {ckpt}")
        return ckpt
    env = {
        "SKIP_BASE_STAGE1": "1",
        "SKIP_STAGE15": "1",
        "SKIP_EVAL": "1",
        "STAGE1_CKPT": str(stage1_ckpt),
        "STAGE1_CAPACITY": args.capacity,
        "STAGE125_TAG": args.stage125_tag,
        "STAGE125_EPOCHS": args.stage125_epochs,
        "STAGE125_MC_WEIGHT": args.stage125_mc_weight,
        "STAGE125_MC_KOLEO": args.stage125_mc_koleo,
        "STAGE125_MC_REGULARIZER": args.stage125_mc_regularizer,
        "STAGE125_VICREG_VAR_WEIGHT": args.stage125_vicreg_var_weight,
        "STAGE125_VICREG_COV_WEIGHT": args.stage125_vicreg_cov_weight,
        "STAGE125_VICREG_GAMMA": args.stage125_vicreg_gamma,
        "STAGE125_TARGET_ID_SCALE": args.stage125_target_id_scale,
        "STAGE125_TARGET_CHUNKS": args.stage125_target_chunks,
        "STAGE125_TARGET_SCALE": args.stage125_target_scale,
        "STAGE125_CONTEXT_SCALE": args.stage125_context_scale,
        "STAGE125_MASK_RATIO": args.stage125_mask_ratio,
        "STAGE125_WARMUP": args.stage125_warmup,
        "STAGE125_RAMP": args.stage125_ramp,
        "STAGE125_LIMIT_TRAIN": args.stage125_limit_train,
        "NUM_PROC_STAGE1": args.num_proc_stage1,
    }
    if args.stage125_batch:
        env["STAGE125_BATCH"] = args.stage125_batch
    if args.stage125_lr_mult:
        env["STAGE125_LR_MULT"] = args.stage125_lr_mult
    _run(["bash", "scripts/ablation/run_all_stage1_chunk_stage15.sh"], env=env, cwd=root, dry_run=args.dry_run)
    if args.dry_run:
        return ckpt
    if not _exists(ckpt):
        raise FileNotFoundError(f"Stage 1.25 best checkpoint not found after training: {ckpt}")
    return ckpt


def _maybe_train_stage15(args, root: Path, tx_ckpt: Path) -> Path:
    ckpt = root / "results" / "runs" / args.stage15_tag / "ckpt_spatial_best.pt"
    if _exists(ckpt) and not args.force_stage15:
        print(f"[pipeline] Stage 1.5 ckpt exists, skip training: {ckpt}")
        return ckpt
    env = {
        "STAGE1_CKPT": str(tx_ckpt),
        "TAG": args.stage15_tag,
        "EPOCHS": args.stage15_epochs,
        "BATCH_SIZE": args.stage15_batch,
        "SUBGRAPH_SIZE": args.stage15_subgraph_size,
        "TX_ENCODE_BATCH": args.stage15_tx_encode_batch,
        "MP": args.mixed_precision,
        "SUBGRAPH_KIND": args.stage15_subgraph_kind,
        "REGION_TOKEN_MODE": args.stage15_region_token_mode,
        "MASK_TARGET": args.stage15_mask_target,
        "JEPA_MASK_RATIO": args.stage15_jepa_mask_ratio,
        "JEPA_MASK_STRATEGY": args.stage15_jepa_mask_strategy,
        "JEPA_BLOCK_SIZE": args.stage15_jepa_block_size,
        "REGION_TX_AGG": args.stage15_region_tx_agg,
        "REGION_WEIGHTED_SIGMA": args.stage15_region_weighted_sigma,
        "REGION_INCLUDE_ANCHOR": args.stage15_region_include_anchor,
        "LIMIT_TRAIN_SHARDS": args.stage15_limit_train_shards,
        "LIMIT_VAL_SHARDS": args.stage15_limit_val_shards,
        "NUM_PROC": args.num_proc_stage15,
    }
    _run(["bash", "scripts/train/stage15_main.sh"], env=env, cwd=root, dry_run=args.dry_run)
    if args.dry_run:
        return ckpt
    if not _exists(ckpt):
        raise FileNotFoundError(f"Stage 1.5 best checkpoint not found after training: {ckpt}")
    return ckpt


def _eval_stage1(args, root: Path, ckpts: list[Path]) -> None:
    out = root / "results" / "eval" / f"{args.eval_prefix}_stage1_tx.csv"
    fig = root / "results" / "figures" / args.eval_prefix / "representation" / "stage1" / "mixed_test"
    cmd = [
        "python", "scripts/eval/stage1_tx.py",
        "--prepared-dir", args.prepared_dir,
        "--split", "test",
        "--ckpts", *[str(c) for c in ckpts],
        "--val-samples", str(args.stage1_test_samples),
        "--pool-spots", str(args.stage1_pool_spots),
        "--linear-probe-genes", str(args.stage1_linear_probe_genes),
        "--encode-batch", str(args.stage1_encode_batch),
        "--tx-pooling-mode", args.stage1_tx_pooling_mode,
        "--out", str(out),
    ]
    if args.neural_linear_probe:
        cmd += [
            "--neural-linear-probe",
            "--neural-probe-epochs", str(args.neural_probe_epochs),
            "--neural-probe-patience", str(args.neural_probe_patience),
            "--neural-probe-lr", str(args.neural_probe_lr),
            "--neural-probe-weight-decay", str(args.neural_probe_weight_decay),
            "--neural-probe-batch", str(args.neural_probe_batch),
        ]
    if args.make_viz:
        cmd += [
            "--make-viz",
            "--probe-figures",
            "--viz-out-dir", str(fig),
            "--viz-max-spots", str(args.stage1_viz_max_spots),
            "--viz-genes", *args.stage1_viz_genes,
        ]
    _run(cmd, cwd=root, dry_run=args.dry_run)


def _eval_stage1_gene_map(args, root: Path, ckpts: list[Path]) -> None:
    if not args.make_viz:
        return
    base = root / "results" / "figures" / args.eval_prefix / "linear_probe" / "gene_map" / "stage1" / "hest"
    for ckpt in ckpts:
        out_dir = base / ckpt.parent.name
        cmd = [
            "python", "scripts/eval/stage1_gene_map.py",
            "--stage1-ckpt", str(ckpt),
            "--prepared-dir", args.prepared_dir,
            "--split", "test",
            "--representations", args.stage1_gene_map_representations,
            "--genes", *args.stage1_gene_map_genes,
            "--auto-select-genes", str(args.stage1_gene_map_auto_genes),
            "--probe-train-samples", str(args.stage1_gene_map_probe_train_samples),
            "--max-train-spots", str(args.stage1_gene_map_max_train_spots),
            "--tx-batch", str(args.stage1_gene_map_tx_batch),
            "--tx-pooling-mode", args.stage1_tx_pooling_mode,
            "--out-dir", str(out_dir),
        ]
        _run(cmd, cwd=root, dry_run=args.dry_run)


def _eval_dlpfc(args, root: Path, ckpts: list[Path]) -> None:
    if not args.dlpfc_dir or not Path(args.dlpfc_dir).exists():
        print(f"[pipeline] skip DLPFC eval: missing dir {args.dlpfc_dir}")
        return
    out = root / "results" / "eval" / f"{args.eval_prefix}_dlpfc.csv"
    per = root / "results" / "eval" / f"{args.eval_prefix}_dlpfc_per_sample.csv"
    fig = root / "results" / "figures" / args.eval_prefix
    cmd = [
        "python", "scripts/eval/dlpfc_eval.py",
        "--dlpfc-dir", args.dlpfc_dir,
        "--ckpts", *[str(c) for c in ckpts],
        "--representations", args.dlpfc_representations,
        "--gene-map-representations", args.dlpfc_gene_map_representations,
        "--out", str(out),
        "--per-sample-out", str(per),
        "--viz-samples", str(args.dlpfc_viz_samples),
        "--genes", *args.dlpfc_genes,
        "--auto-select-genes", str(args.dlpfc_auto_select_genes),
        "--cluster-methods", args.dlpfc_cluster_methods,
        "--spatial-cluster-weight", str(args.dlpfc_spatial_cluster_weight),
        "--tx-pooling-mode", args.stage1_tx_pooling_mode,
        "--encode-batch", str(args.stage1_encode_batch),
    ]
    if args.neural_linear_probe:
        cmd += [
            "--neural-linear-probe",
            "--neural-probe-epochs", str(args.neural_probe_epochs),
            "--neural-probe-patience", str(args.neural_probe_patience),
            "--neural-probe-lr", str(args.neural_probe_lr),
            "--neural-probe-weight-decay", str(args.neural_probe_weight_decay),
            "--neural-probe-batch", str(args.neural_probe_batch),
        ]
    if args.make_viz:
        cmd += ["--viz-out-dir", str(fig)]
    _run(cmd, cwd=root, dry_run=args.dry_run)


def _copy_eval_to_run_dir(out: Path, run_ckpt: Path, name: str, *, dry_run: bool = False) -> None:
    """Mirror important eval artifacts next to the checkpoint for easy audit."""
    run_out = run_ckpt.parent / name
    if dry_run:
        print(f"[pipeline] would copy {out} -> {run_out}")
        return
    if not _exists(out):
        raise FileNotFoundError(f"Expected eval output was not created: {out}")
    run_out.write_bytes(out.read_bytes())
    print(f"[pipeline] copied eval artifact: {run_out}")


def _eval_stage15(args, root: Path, spatial_ckpt: Path, tx_ckpt: Path) -> None:
    out = root / "results" / "eval" / f"{args.eval_prefix}_stage15_indist.csv"
    cmd = [
        "python", "scripts/eval/stage15_indist.py",
        "--prepared-dir", args.prepared_dir,
        "--split", "test",
        "--ckpts", str(spatial_ckpt),
        "--max-samples", str(args.stage15_eval_max_samples),
        "--eval-subgraph-size", str(args.stage15_eval_subgraph_size),
        "--num-views", str(args.stage15_eval_num_views),
        "--out", str(out),
        "--device", args.stage15_eval_device,
        "--cache-device", args.stage15_eval_cache_device,
    ]
    _run(cmd, cwd=root, dry_run=args.dry_run)
    _copy_eval_to_run_dir(out, spatial_ckpt, "test_stage15_indist.csv", dry_run=args.dry_run)
    if args.stage15_gene_map:
        fig = root / "results" / "figures" / args.eval_prefix / "gene_map" / "stage15" / "hest"
        cmd = [
            "python", "scripts/eval/stage15_gene_map.py",
            "--stage1-ckpt", str(tx_ckpt),
            "--spatial-ckpt", str(spatial_ckpt),
            "--split", "test",
            "--out-dir", str(fig),
            "--genes", *args.stage15_gene_map_genes,
            "--auto-select-genes", str(args.stage15_gene_map_auto_genes),
            "--tx-batch", str(args.stage15_tx_encode_batch),
        ]
        _run(cmd, cwd=root, dry_run=args.dry_run)
        gene_csv = fig / "gene_map_scc.csv"
        if _exists(gene_csv):
            _copy_eval_to_run_dir(gene_csv, spatial_ckpt, "test_stage15_gene_map_scc.csv", dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pipeline", choices=("joint", "sequential"), default="joint",
                    help="joint: Stage1 MSM+chunk-JEPA curriculum -> spatial. sequential is deprecated Stage1.25 control.")
    ap.add_argument("--prepared-dir", default="results/cache/prepared_expanded")
    ap.add_argument("--capacity", choices=("spatula_lite", "spatula_mid", "spatula_large"), default="spatula_mid")
    ap.add_argument("--mixed-precision", default="bf16")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--grid", default="none",
                    choices=("none", "stage1_core", "joint_chunk", "stage1_vocab", "stage1_capacity", "sequential_core", "stage15_spatial", "value_aug"),
                    help="Predefined ablation grid. Combine with --sweep for extra axes.")
    ap.add_argument("--sweep", action="append", default=[],
                    help="Add a cartesian sweep axis, e.g. --sweep capacity=spatula_mid,spatula_large")
    ap.add_argument("--append-sweep-suffix", action=argparse.BooleanOptionalAction, default=True,
                    help="Append sweep suffixes to user-provided tags/eval prefixes.")

    ap.add_argument("--masked-value-aug-profile", default="dropout",
                    choices=("keep_only", "dropout", "mild", "with_noise", "aggressive"),
                    help="Masked-position value augmentation profile. Default 'dropout' is "
                         "paper-aligned (Sinha et al. 2021 EMNLP): NO noise, 85% keep + 15% "
                         "token-level value dropout. Other profiles for ablation: 'keep_only' "
                         "(pure MASK, no value perturbation), 'with_noise' (85/10/5 with mild "
                         "noise), 'aggressive' (legacy 75/15/10 with strong noise). 'mild' is "
                         "kept as a backward-compat alias for 'dropout'.")
    ap.add_argument("--stage1-objective", default="msm_multi_chunk",
                    choices=("msm_only", "msm_multi_chunk", "view_jepa_w005", "view_jepa_w010", "dino_late_no_koleo"))
    ap.add_argument("--stage1-tag", default=None)
    ap.add_argument("--stage1-ckpt", default="", help="Existing Stage 1 tx checkpoint. If set, Stage 1 training is skipped unless --force-stage1.")
    ap.add_argument("--stage1-resume-ckpt", default="", help="Resume Stage 1 training from a train.py checkpoint while keeping the same pipeline config/tag.")
    ap.add_argument("--stage1-epochs", default="50")
    ap.add_argument("--stage1-batch", default="")
    ap.add_argument("--stage1-max-seq-len", default="auto")
    ap.add_argument("--stage1-vocab", default="4096", choices=("4096", "8192", "full"))
    ap.add_argument("--stage1-limit-train", default="0", help="Cap Stage 1 train pool for smoke runs (<=0 = full).")
    ap.add_argument("--stage125-limit-train", default="0", help="Cap Stage 1.25 train pool for smoke runs (<=0 = full).")
    ap.add_argument("--num-proc-stage1", default="8", help="Accelerate processes for Stage 1 and Stage 1.25.")
    ap.add_argument("--num-proc-stage15", default="1", help="Accelerate processes for Stage 1.5. Keep 1 unless the Stage1.5 trainer is made DDP-safe.")
    ap.add_argument("--force-stage1", action="store_true")

    ap.add_argument("--stage125-tag", default=None)
    ap.add_argument("--stage125-epochs", default="30")
    ap.add_argument("--stage125-batch", default="")
    ap.add_argument("--stage125-lr-mult", default="1.0")
    ap.add_argument("--stage125-mask-ratio", default="0.0")
    ap.add_argument("--stage125-mc-weight", default="0.30")
    ap.add_argument("--stage125-mc-koleo", default="0.0")
    ap.add_argument("--stage125-mc-regularizer", default="vicreg", choices=("none", "koleo", "vicreg"))
    ap.add_argument("--stage125-vicreg-var-weight", default="0.05")
    ap.add_argument("--stage125-vicreg-cov-weight", default="0.01")
    ap.add_argument("--stage125-vicreg-gamma", default="1.0")
    ap.add_argument("--stage125-target-id-scale", default="0.25")
    ap.add_argument("--stage125-target-chunks", default="auto")
    ap.add_argument("--stage125-target-scale", default="0.15,0.25")
    ap.add_argument("--stage125-context-scale", default="0.45,0.65")
    ap.add_argument("--stage125-warmup", default="0")
    ap.add_argument("--stage125-ramp", default="5")
    ap.add_argument("--force-stage125", action="store_true")

    ap.add_argument("--stage15-tag", default=None)
    ap.add_argument("--stage15-epochs", default="30")
    ap.add_argument("--stage15-batch", default="4")
    ap.add_argument("--stage15-subgraph-size", default="256")
    ap.add_argument("--stage15-tx-encode-batch", default="64")
    ap.add_argument("--stage15-subgraph-kind", default="ego", choices=("ego", "random"))
    ap.add_argument("--stage15-region-token-mode", default="separate", choices=("separate", "fused"))
    ap.add_argument("--stage15-mask-target", default="spot", choices=("spot", "region", "both"))
    ap.add_argument("--stage15-jepa-mask-ratio", default="")
    ap.add_argument("--stage15-jepa-mask-strategy", default="")
    ap.add_argument("--stage15-jepa-block-size", default="")
    ap.add_argument("--stage15-region-tx-agg", default="mean", choices=("mean", "sum_log1p", "weighted"))
    ap.add_argument("--stage15-region-weighted-sigma", default="1.0")
    ap.add_argument("--stage15-region-include-anchor", default="false", choices=("true", "false"))
    ap.add_argument("--stage15-limit-train-shards", default="0",
                     help="Cap Stage 1.5 train shards for smoke runs (<=0 = full).")
    ap.add_argument("--stage15-limit-val-shards", default="0",
                     help="Cap Stage 1.5 val shards for smoke runs (<=0 = full).")
    ap.add_argument("--force-stage15", action="store_true")

    ap.add_argument("--skip-stage15", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-prefix", default="spot_encoder_pipeline")
    ap.add_argument("--stage1-test-samples", default="76")
    ap.add_argument("--stage1-pool-spots", default="8000")
    ap.add_argument("--stage1-linear-probe-genes", default="256")
    ap.add_argument("--stage1-encode-batch", default="16",
                    help="Micro-batch for Stage1 eval encoding. Keep low for vocab8192/full; OOM fallback halves it further.")
    ap.add_argument("--neural-linear-probe", action="store_true",
                    help="Run slower torch nn.Linear probes with early stopping in Stage1/DLPFC eval.")
    ap.add_argument("--neural-probe-epochs", default="50")
    ap.add_argument("--neural-probe-patience", default="8")
    ap.add_argument("--neural-probe-lr", default="1e-3")
    ap.add_argument("--neural-probe-weight-decay", default="1e-4")
    ap.add_argument("--neural-probe-batch", default="512")
    ap.add_argument("--stage1-tx-pooling-mode", default="ckpt",
                    choices=("ckpt", "cls", "token_mean", "cls_token_mean_sum", "cls_token_mean_avg",
                             "cls_mean_sum", "cls_mean_avg", "mean"),
                    help="Eval-time Stage1 tx readout override. 'ckpt' preserves training config.")
    ap.add_argument("--stage1-viz-max-spots", default="5000")
    ap.add_argument("--stage1-viz-genes", nargs="*", default=["MKI67", "EPCAM", "COL1A1", "CD3D"])
    ap.add_argument("--stage1-gene-map-representations", default="spot_state")
    ap.add_argument("--stage1-gene-map-genes", nargs="*", default=["MKI67", "EPCAM", "COL1A1", "CD3D"])
    ap.add_argument("--stage1-gene-map-auto-genes", default="4")
    ap.add_argument("--stage1-gene-map-probe-train-samples", default="20")
    ap.add_argument("--stage1-gene-map-max-train-spots", default="20000")
    ap.add_argument("--stage1-gene-map-tx-batch", default="16")
    ap.add_argument("--dlpfc-dir", default="/data/spatiallibd")
    ap.add_argument("--dlpfc-representations", default="h_tx,chunk_state,spot_state")
    ap.add_argument("--dlpfc-gene-map-representations", default="spot_state")
    ap.add_argument("--dlpfc-cluster-methods", default="kmeans,gmm,leiden,spatial_leiden",
                    help="Comma list for DLPFC zero-shot clustering. Recommended: kmeans,gmm,leiden,spatial_leiden.")
    ap.add_argument("--dlpfc-spatial-cluster-weight", default="0.25",
                    help="Weak xy-prior weight for spatial_leiden clustering.")
    ap.add_argument("--dlpfc-viz-samples", default="6")
    ap.add_argument("--dlpfc-genes", nargs="*", default=["MBP", "SNAP25", "PCP4", "GFAP", "MOBP", "CARTPT"])
    ap.add_argument("--dlpfc-auto-select-genes", default="4",
                    help="Append N GT spatially patterned DLPFC genes for qualitative gene maps.")
    ap.add_argument("--make-viz", action="store_true")
    ap.add_argument("--stage15-eval-max-samples", default="8")
    ap.add_argument("--stage15-eval-subgraph-size", default="128")
    ap.add_argument("--stage15-eval-num-views", default="1")
    ap.add_argument("--stage15-eval-device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--stage15-eval-cache-device", default="cpu", choices=("auto", "cuda", "cpu"),
                    help="Device for stage15_indist h_tx cache creation. CPU is safer after long CUDA runs.")
    ap.add_argument("--stage15-gene-map", action="store_true")
    ap.add_argument("--stage15-gene-map-genes", nargs="*", default=["MKI67", "EPCAM", "COL1A1", "CD3D"])
    ap.add_argument("--stage15-gene-map-auto-genes", default="4",
                    help="Append N GT spatially patterned HEST genes for qualitative Stage1.5 maps.")
    args = ap.parse_args()
    args._stage1_tag_user = args.stage1_tag is not None
    args._stage125_tag_user = args.stage125_tag is not None
    args._stage15_tag_user = args.stage15_tag is not None
    args._eval_prefix_user = args.eval_prefix != "spot_encoder_pipeline"
    return args


def _run_single(args: argparse.Namespace, root: Path, suffix: str = "") -> None:
    _prepare_tags(args, suffix)
    print(f"[pipeline] root={root}")
    print(f"[pipeline] mode={args.pipeline} suffix={suffix or '<single>'}")
    if args.pipeline == "sequential":
        print(f"[pipeline] tags: stage1={args.stage1_tag} stage125={args.stage125_tag} stage15={args.stage15_tag}")
    else:
        print(f"[pipeline] tags: stage1={args.stage1_tag} stage15={args.stage15_tag}")

    stage1_ckpt = _maybe_train_stage1(args, root)
    tx_for_spatial = stage1_ckpt
    eval_tx_ckpts = [stage1_ckpt]

    if args.pipeline == "sequential":
        stage125_ckpt = _maybe_train_stage125(args, root, stage1_ckpt)
        tx_for_spatial = stage125_ckpt
        eval_tx_ckpts.append(stage125_ckpt)

    spatial_ckpt = None
    if not args.skip_stage15:
        spatial_ckpt = _maybe_train_stage15(args, root, tx_for_spatial)

    if not args.skip_eval:
        (root / "results" / "eval").mkdir(parents=True, exist_ok=True)
        _eval_stage1(args, root, eval_tx_ckpts)
        _eval_stage1_gene_map(args, root, eval_tx_ckpts)
        _eval_dlpfc(args, root, eval_tx_ckpts)
        if spatial_ckpt is not None:
            _eval_stage15(args, root, spatial_ckpt, tx_for_spatial)

    print("\n" + "=" * 88)
    print("[pipeline] run complete")
    print(f"  Stage 1 tx ckpt     : {stage1_ckpt}")
    if args.pipeline == "sequential":
        print(f"  Stage 1.25 tx ckpt  : {tx_for_spatial}")
    if spatial_ckpt is not None:
        print(f"  Stage 1.5 ckpt      : {spatial_ckpt}")
    print(f"  Eval prefix         : {args.eval_prefix}")
    print("=" * 88)


def main() -> None:
    root = _repo_root()
    base_args = parse_args()
    runs = _expanded_runs(base_args)
    print(f"[pipeline] planned runs: {len(runs)}")
    for i, (args, suffix) in enumerate(runs, 1):
        print("\n" + "#" * 88)
        print(f"[pipeline] RUN {i}/{len(runs)}")
        print("#" * 88)
        _run_single(args, root, suffix=suffix)


if __name__ == "__main__":
    main()
