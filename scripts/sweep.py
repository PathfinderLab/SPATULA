"""Sweep runner — YAML drives everything.

Reads configs/sweep/<file>.yaml, then for each experiment listed there:
  1. (once) runs prepare_data.py if shards/vocab are missing or rebuild flag is set
  2. launches scripts/train.py with the merged hyper-parameters
  3. runs scripts/eval/zero_shot.py and scripts/eval/linear_probe.py on ckpt_last

Usage:
    python scripts/sweep.py --sweep configs/sweep/stage1_ours_tx.yaml
    # — or via the thin shell wrapper —
    bash scripts/run_experiments.sh configs/sweep/stage1_ours_tx.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = ROOT / "results" / "cache" / "prepared"


# -------------------------------------------------------------------- helpers

def _normalise_tune(val) -> str:
    if val is None:
        return "none"
    return str(val).strip().lower()


def _validate_tune(val: str) -> None:
    if val in ("none", "all", "lora", "adapter"):
        return
    if val.startswith("partial:"):
        try:
            int(val.split(":", 1)[1])
            return
        except ValueError:
            pass
    raise SystemExit(
        f"[sweep] ERROR: uni_tune='{val}' invalid. "
        f"Expected one of: none | all | partial:N | lora | adapter."
    )


def _validate_backbone(val: str) -> None:
    if val not in ("feature", "uni"):
        raise SystemExit(
            f"[sweep] ERROR: image_backbone='{val}' invalid. Expected feature | uni."
        )


def _run(cmd: list[str]) -> None:
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


# -------------------------------------------------------------------- prepare

def maybe_run_prepare(s: dict) -> None:
    """Run prepare_data.py only if needed."""
    splits = PREPARED_DIR / "splits.json"
    vocab = PREPARED_DIR / "hvg_vocab.json"
    n_shards = len(list(PREPARED_DIR.glob("*.h5")))

    need = False
    args: list[str] = []
    if s.get("rebuild"):
        print("[sweep] rebuild=true — wiping splits.json + hvg_vocab.json")
        if splits.exists(): splits.unlink()
        if vocab.exists(): vocab.unlink()
        args += ["--rebuild_vocab"]
        need = True
    elif s.get("force_prepare"):
        need = True
    elif not splits.exists() or not vocab.exists() or n_shards == 0:
        need = True
    elif (not s.get("smoke")) and n_shards < 100:
        print(f"[sweep] WARNING: only {n_shards} shards on disk — looks like a smoke cache.")
        print(f"[sweep]          Set sweep.rebuild=true in YAML to redo full prep (~4h novae).")
        raise SystemExit(1)

    if s.get("smoke"):
        args += ["--limit", "6"]

    if need:
        print("[sweep] prepare_data ...")
        _run([sys.executable, "scripts/data/prepare.py", *args])
    else:
        print(f"[sweep] prepare_data: SKIP "
              f"(found {n_shards} shards + splits.json + hvg_vocab.json).")


# -------------------------------------------------------------------- train

def build_train_args(s: dict, exp: dict) -> tuple[list[str], str]:
    """Merge sweep defaults with per-experiment overrides into argparse tokens."""
    name = exp["name"]
    backbone = exp.get("image_backbone", s["image_backbone"])
    _validate_backbone(backbone)

    epochs = int(exp.get("epochs", s["epochs"]))
    batch_size = int(exp.get("batch_size", s["batch_size"]))

    tag_prefix = s.get("tag_prefix", "full_human")
    if backbone == "uni":
        # Accept both new key (uni_tune) and legacy key (uni_freeze).
        tune_raw = exp.get("uni_tune",
                            exp.get("uni_freeze",
                                    s.get("uni_tune", s.get("uni_freeze", "none"))))
        tune = _normalise_tune(tune_raw)
        _validate_tune(tune)
        tag = f"{tag_prefix}_{name}_uni_{tune.replace(':', '_')}"
    else:
        tune = None
        tag = f"{tag_prefix}_{name}_feature"

    # Allow sweep-level / per-experiment override of the experiment yaml path.
    # (mirrors data_yaml / model_yaml / train_yaml override semantics)
    experiment_yaml = exp.get("experiment_yaml", s.get("experiment_yaml"))
    if not experiment_yaml:
        experiment_yaml = f"configs/experiments/{name}.yaml"
    cli = [
        "--experiment", str(experiment_yaml),
        "--tag", tag,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    # Optional model/data/train yaml overrides at the sweep level.
    for key, flag in [("model_yaml", "--model"),
                       ("data_yaml", "--data"),
                       ("train_yaml", "--train")]:
        val = exp.get(key, s.get(key))
        if val:
            cli += [flag, str(val)]
    if backbone == "uni":
        cli += ["--image-backbone", "uni", "--uni-tune", tune]
    else:
        cli += ["--image-backbone", "feature"]
    # Forward tx-warmup setting if provided in sweep (or per-experiment).
    warmup = exp.get("tx_warmup_epochs", s.get("tx_warmup_epochs"))
    if warmup is not None and int(warmup) > 0:
        cli += ["--tx-warmup-epochs", str(int(warmup))]
    # Stage-1 / Stage-2 wiring.
    if exp.get("stage1_only", s.get("stage1_only", False)):
        cli += ["--stage1-only"]
    tx_ckpt = exp.get("tx_ckpt", s.get("tx_ckpt"))
    if tx_ckpt:
        cli += ["--tx-ckpt", str(tx_ckpt)]
    cli += list(exp.get("extra_args", []))
    return cli, tag


def launcher_cmd(s: dict) -> list[str]:
    n = int(s.get("num_processes", 1))
    if n > 1:
        return [
            "accelerate", "launch",
            "--num_processes", str(n),
            "--mixed_precision", s.get("mixed_precision", "bf16"),
            "scripts/train.py",
        ]
    return [sys.executable, "scripts/train.py"]


def run_experiment(s: dict, exp: dict) -> None:
    cli, tag = build_train_args(s, exp)
    cmd = launcher_cmd(s) + cli
    print(f"\n[sweep] === {exp['name']}  (tag={tag}) ===")
    _run(cmd)

    # Stage 1 — no align, no cross-modal eval.  Just produce the loss curve.
    if exp.get("stage1_only", s.get("stage1_only", False)):
        print(f"[sweep] stage1_only={exp['name']} done — skipping cross-modal eval.")
        return

    ckpt = ROOT / "results" / "runs" / tag / "ckpt_last.pt"
    exp_cfg = ROOT / f"configs/experiments/{exp['name']}.yaml"
    if not ckpt.exists():
        print(f"[sweep] no ckpt for {exp['name']} — skipping eval.")
        return
    print(f"[sweep] --- zero-shot eval ({tag}) ---")
    _run([sys.executable, "scripts/eval/zero_shot.py",
          "--experiment", str(exp_cfg), "--ckpt", str(ckpt), "--split", "test"])
    print(f"[sweep] --- MLP linear-probe eval ({tag}) ---")
    _run([sys.executable, "scripts/eval/linear_probe.py",
          "--experiment", str(exp_cfg), "--ckpt", str(ckpt), "--eval_split", "test"])

    downstream = s.get("downstream", {}) or {}
    label_csv = downstream.get("label_csv")
    label_col = downstream.get("label_col")
    if label_csv and label_col:
        print(f"[sweep] --- slide-level downstream MIL ({tag}, label={label_col}) ---")
        cmd = [
            sys.executable, "scripts/eval/slide_mil.py",
            "--experiment", str(exp_cfg),
            "--ckpt", str(ckpt),
            "--label-csv", str(label_csv),
            "--sample-id-col", str(downstream.get("sample_id_col", "sample_id")),
            "--label-col", str(label_col),
            "--eval-split", str(downstream.get("eval_split", "test")),
        ]
        arms = downstream.get("arms") or []
        if arms:
            cmd += ["--arms"] + [str(a) for a in arms]
        if downstream.get("skip_attention", False):
            cmd += ["--skip-attention"]
        if "attention_epochs" in downstream:
            cmd += ["--attention-epochs", str(downstream["attention_epochs"])]
        _run(cmd)

    print(f"[sweep] --- render report figures ({tag}) ---")
    run_dir = ckpt.parent
    try:
        _run([sys.executable, "scripts/viz/figures.py", "--run", str(run_dir)])
    except subprocess.CalledProcessError as e:
        print(f"[sweep] make_figures failed (exit={e.returncode}); continuing.")


# -------------------------------------------------------------------- main

def print_plan(s: dict) -> None:
    print("─" * 70)
    print(f"[sweep] num_processes  : {s.get('num_processes')}   mp={s.get('mixed_precision', 'bf16')}")
    print(f"[sweep] smoke          : {s.get('smoke', False)}")
    print(f"[sweep] rebuild        : {s.get('rebuild', False)}    force_prepare={s.get('force_prepare', False)}")
    print(f"[sweep] image_backbone : {s['image_backbone']}")
    if s["image_backbone"] == "uni":
        tune = _normalise_tune(s.get("uni_tune", s.get("uni_freeze", "none")))
        print(f"[sweep] uni_tune       : {tune}")
    print(f"[sweep] epochs         : {s['epochs']}    batch_size={s['batch_size']}")
    exps = [e["name"] for e in s["experiments"]]
    print(f"[sweep] experiments    : {exps}")
    print("─" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep/stage1_ours_tx.yaml",
                    help="Path to sweep YAML (default: configs/sweep/stage1_ours_tx.yaml).")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only run experiments whose name is in this list.")
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.is_absolute():
        sweep_path = ROOT / sweep_path
    if not sweep_path.exists():
        raise SystemExit(f"[sweep] ERROR: sweep file not found: {sweep_path}")

    cfg = yaml.safe_load(sweep_path.read_text())
    if "sweep" not in cfg:
        raise SystemExit(f"[sweep] ERROR: top-level 'sweep:' key missing in {sweep_path}.")
    s = cfg["sweep"]

    # Validate up front so bad configs fail fast (before any GPU work).
    _validate_backbone(s["image_backbone"])
    if s["image_backbone"] == "uni":
        tune = _normalise_tune(s.get("uni_tune", s.get("uni_freeze", "none")))
        _validate_tune(tune)
        s["uni_tune"] = tune
    if args.only:
        s["experiments"] = [e for e in s["experiments"] if e["name"] in args.only]
        if not s["experiments"]:
            raise SystemExit(f"[sweep] no experiments left after --only filter")

    print_plan(s)
    maybe_run_prepare(s)
    for exp in s["experiments"]:
        try:
            run_experiment(s, exp)
        except subprocess.CalledProcessError as e:
            print(f"[sweep] !!! experiment {exp['name']} failed with exit {e.returncode}. "
                  f"Continuing with next.")
    print("[sweep] all done.")


if __name__ == "__main__":
    main()
