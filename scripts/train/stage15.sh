#!/usr/bin/env bash
# STAGE 1.5 — Spatial Foundation pretraining (Spatial Predictive JEPA).
# BASELINE configuration: fused region token + random subgraph sampling.
#
# Loads:
#   * frozen Stage-1 tx_encoder (ckpt_tx_encoder_best.pt) + carries its
#     gene_norm + vocab_clip so the encoder sees its training distribution.
#   * frozen image features (UNI on disk) — optionally LoRA-tuned later
# Trains:
#   * SpatialEncoder (KGNN / smooth / kxformer)
# Saves:
#   * ckpt_spatial_best.pt for Stage 2 reuse
#
# For the methodological MAIN CANDIDATE (separate region token + spot-only
# mask + ego subgraph), use `bash scripts/train/stage15_main.sh` instead.
#
# Usage:
#     bash scripts/train/stage15.sh
#     ABL_PROFILE=fast bash scripts/train/stage15.sh
#     STAGE1_CKPT=results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt bash scripts/train/stage15.sh
#
# See docs/design/stage15_spatial_jepa.md for the full design.

set -euo pipefail
# scripts/train/stage15.sh → repo root is 2 levels up
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Stage 1.5 trainer is NOT DDP-aware (single-process scaffold).  Spawning
# multiple Accelerate processes would have every rank duplicate the work
# (and race on the same checkpoint file).  Keep NUM_PROC=1 unless/until
# the trainer is upgraded with a DistributedSampler + rank-0 save guard.
NUM_PROC="${NUM_PROC:-1}"
MP="${MP:-bf16}"
EPOCHS="${EPOCHS:-30}"
TAG="${TAG:-stage15_spatial_jepa}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
TX_ENCODE_BATCH="${TX_ENCODE_BATCH:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SUBGRAPH_SIZE="${SUBGRAPH_SIZE:-}"

if [[ "$NUM_PROC" != "1" ]]; then
    echo "[stage15] WARNING: NUM_PROC=$NUM_PROC but trainer is single-process."
    echo "[stage15]          Each rank will duplicate the run and race on ckpt writes."
    echo "[stage15]          Set NUM_PROC=1 (default) unless you know what you're doing."
fi

DATA_CFG="configs/stage15/data.yaml"
TRAIN_CFG="configs/stage15/train.yaml"
if [[ -n "$STAGE1_CKPT" || -n "$TX_ENCODE_BATCH" || -n "$SUBGRAPH_SIZE" || -n "$BATCH_SIZE" ]]; then
    source "${ROOT}/scripts/ablation/_common.sh" 1>/dev/null
fi
if [[ -n "$STAGE1_CKPT" || -n "$TX_ENCODE_BATCH" || -n "$SUBGRAPH_SIZE" ]]; then
    DATA_CFG="/tmp/${TAG}_data.yaml"
    DATA_OVERRIDES=()
    if [[ -n "$STAGE1_CKPT" ]]; then DATA_OVERRIDES+=("data.stage1_ckpt" "$(realpath "$STAGE1_CKPT")"); fi
    if [[ -n "$TX_ENCODE_BATCH" ]]; then DATA_OVERRIDES+=("data.tx_encode_batch" "$TX_ENCODE_BATCH"); fi
    if [[ -n "$SUBGRAPH_SIZE" ]]; then DATA_OVERRIDES+=("data.subgraph_size" "$SUBGRAPH_SIZE"); fi
    make_yaml_override "$DATA_CFG" configs/stage15/data.yaml "${DATA_OVERRIDES[@]}"
    [[ -n "$STAGE1_CKPT" ]] && echo "[stage15] stage1_ckpt=$(realpath "$STAGE1_CKPT")"
fi
if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_CFG="/tmp/${TAG}_train.yaml"
    make_yaml_override "$TRAIN_CFG" configs/stage15/train.yaml "train.batch_size" "$BATCH_SIZE"
fi

accelerate launch --num_processes "$NUM_PROC" --mixed_precision "$MP" \
    scripts/train/stage15.py \
        --data "$DATA_CFG" \
        --model configs/stage15/model.yaml \
        --train "$TRAIN_CFG" \
        --experiment configs/stage15/experiment.yaml \
        --tag "$TAG" \
        --epochs "$EPOCHS"
