#!/usr/bin/env bash
# STAGE 1.5 — MAIN candidate entrypoint.
#
# Runs the methodologically preferred configuration:
#   * region_token_mode = separate    (spot token + region token per anchor)
#   * mask_target       = spot        (region is visible context; spot is target)
#   * subgraph_kind     = ego         (sample-level KNN, BFS ego subgraph)
#
# These mirror what scripts/ablation/run_region_token.sh runs as variant R2
# (the I-JEPA-faithful "region predicts masked spot" candidate).
#
# Everything else (data shards, frozen Stage-1 ckpt, gene_norm reuse, fuse_dim,
# JEPA mask_ratio etc.) inherits from the base configs at runtime via YAML
# overrides — so you don't need to maintain a parallel config file.
#
# Usage:
#     bash scripts/train/stage15_main.sh
#     TAG=stage15_main_v2 bash scripts/train/stage15_main.sh
#     EPOCHS=50 bash scripts/train/stage15_main.sh
#     STAGE1_CKPT=results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt \
#       bash scripts/train/stage15_main.sh
#     TX_ENCODE_BATCH=128 BATCH_SIZE=8 bash scripts/train/stage15_main.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_PROC="${NUM_PROC:-1}"        # single-process trainer (see stage15.sh comment)
MP="${MP:-bf16}"
EPOCHS="${EPOCHS:-30}"
TAG="${TAG:-stage15_main_separate_spot_ego}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
TX_ENCODE_BATCH="${TX_ENCODE_BATCH:-}"
TX_CACHE_DEVICE="${TX_CACHE_DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
SUBGRAPH_SIZE="${SUBGRAPH_SIZE:-}"
SUBGRAPH_KIND="${SUBGRAPH_KIND:-ego}"
REGION_TOKEN_MODE="${REGION_TOKEN_MODE:-separate}"
MASK_TARGET="${MASK_TARGET:-spot}"
JEPA_MASK_RATIO="${JEPA_MASK_RATIO:-}"
JEPA_MASK_STRATEGY="${JEPA_MASK_STRATEGY:-}"
JEPA_BLOCK_SIZE="${JEPA_BLOCK_SIZE:-}"
REGION_TX_AGG="${REGION_TX_AGG:-}"
REGION_WEIGHTED_SIGMA="${REGION_WEIGHTED_SIGMA:-}"
REGION_INCLUDE_ANCHOR="${REGION_INCLUDE_ANCHOR:-}"
LIMIT_TRAIN_SHARDS="${LIMIT_TRAIN_SHARDS:-0}"  # 0 = full
LIMIT_VAL_SHARDS="${LIMIT_VAL_SHARDS:-0}"      # 0 = full

if [[ "$NUM_PROC" != "1" ]]; then
    echo "[stage15_main] WARNING: NUM_PROC=$NUM_PROC but trainer is single-process."
fi

# Build override yamls via the same helper used by ablation scripts.
source "${ROOT}/scripts/ablation/_common.sh" 1>/dev/null

D_BASE="configs/stage15/data.yaml"
M_BASE="configs/stage15/model.yaml"
T_BASE="configs/stage15/train.yaml"
E_BASE="configs/stage15/experiment.yaml"

DATA="/tmp/${TAG}_data.yaml"
MODEL="/tmp/${TAG}_model.yaml"
EXP="/tmp/${TAG}_exp.yaml"

DATA_OVERRIDES=("data.subgraph_kind" "$SUBGRAPH_KIND")
if [[ -n "$STAGE1_CKPT" ]]; then
    DATA_OVERRIDES+=("data.stage1_ckpt" "$(realpath "$STAGE1_CKPT")")
fi
if [[ -n "$TX_ENCODE_BATCH" ]]; then
    DATA_OVERRIDES+=("data.tx_encode_batch" "$TX_ENCODE_BATCH")
fi
if [[ -n "$TX_CACHE_DEVICE" ]]; then
    DATA_OVERRIDES+=("data.tx_cache_device" "$TX_CACHE_DEVICE")
fi
if [[ -n "$SUBGRAPH_SIZE" ]]; then
    DATA_OVERRIDES+=("data.subgraph_size" "$SUBGRAPH_SIZE")
fi
if [[ -n "$REGION_TX_AGG" ]]; then
    DATA_OVERRIDES+=("data.region.tx_agg" "$REGION_TX_AGG")
fi
if [[ -n "$REGION_WEIGHTED_SIGMA" ]]; then
    DATA_OVERRIDES+=("data.region.weighted_sigma" "$REGION_WEIGHTED_SIGMA")
fi
if [[ -n "$REGION_INCLUDE_ANCHOR" ]]; then
    DATA_OVERRIDES+=("data.region.include_anchor" "$REGION_INCLUDE_ANCHOR")
fi
make_yaml_override "$DATA"  "$D_BASE"  "${DATA_OVERRIDES[@]}"
make_yaml_override "$MODEL" "$M_BASE"  "model.spatial.region_token_mode" "$REGION_TOKEN_MODE"
EXP_OVERRIDES=("experiment.jepa.mask_target" "$MASK_TARGET")
if [[ -n "$JEPA_MASK_RATIO" ]]; then
    EXP_OVERRIDES+=("experiment.jepa.mask_ratio" "$JEPA_MASK_RATIO")
fi
if [[ -n "$JEPA_MASK_STRATEGY" ]]; then
    EXP_OVERRIDES+=("experiment.jepa.mask_strategy" "$JEPA_MASK_STRATEGY")
fi
if [[ -n "$JEPA_BLOCK_SIZE" ]]; then
    EXP_OVERRIDES+=("experiment.jepa.block_size" "$JEPA_BLOCK_SIZE")
fi
make_yaml_override "$EXP"   "$E_BASE"  "${EXP_OVERRIDES[@]}"
TRAIN_CFG="$T_BASE"
if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_CFG="/tmp/${TAG}_train.yaml"
    make_yaml_override "$TRAIN_CFG" "$T_BASE" "train.batch_size" "$BATCH_SIZE"
fi

echo "=========================================================="
echo "[stage15_main] tag=$TAG  (token_mode=$REGION_TOKEN_MODE mask_target=$MASK_TARGET subgraph=$SUBGRAPH_KIND)"
echo "  data:  $DATA"
if [[ -n "$STAGE1_CKPT" ]]; then echo "  stage1_ckpt: $(realpath "$STAGE1_CKPT")"; fi
echo "  model: $MODEL"
echo "  train: $TRAIN_CFG"
echo "  exp:   $EXP"
echo "=========================================================="

extra_args=()
if [[ "$LIMIT_TRAIN_SHARDS" -gt 0 ]]; then
    extra_args+=(--limit-train-shards "$LIMIT_TRAIN_SHARDS")
fi
if [[ "$LIMIT_VAL_SHARDS" -gt 0 ]]; then
    extra_args+=(--limit-val-shards "$LIMIT_VAL_SHARDS")
fi

accelerate launch --num_processes "$NUM_PROC" --mixed_precision "$MP" \
    scripts/train/stage15.py \
        --data       "$DATA" \
        --model      "$MODEL" \
        --train      "$TRAIN_CFG" \
        --experiment "$EXP" \
        --tag        "$TAG" \
        --epochs     "$EPOCHS" \
        "${extra_args[@]}"
