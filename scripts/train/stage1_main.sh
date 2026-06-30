#!/usr/bin/env bash
# Stage 1 MAIN-CANDIDATE training (PDF B2×C4 + Gene-JEPA E2).
#
#   mask_ratio        0.30          (PDF B2)
#   value_aug         mixed (C4)    (80/10/10 keep/noise/drop)
#   enable_masked_jepa  true        (E2 — Gene-JEPA aux)
#   jepa_weight       0.1
#
# Output: results/runs/stage1_main_candidate/ckpt_tx_encoder_best.pt
# Compare against the conservative baseline (results/runs/stage1_ours_tx_stage1_feature/)
# with scripts/eval/stage1_tx.py.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_PROC="${NUM_PROC:-8}"
MP="${MP:-bf16}"
EPOCHS="${EPOCHS:-50}"
TAG="${TAG:-stage1_main_candidate}"

accelerate launch --num_processes "$NUM_PROC" --mixed_precision "$MP" \
    scripts/train.py \
        --experiment configs/stage1/experiment_main.yaml \
        --model      configs/stage1/model_main.yaml \
        --data       configs/stage1/data.yaml \
        --train      configs/stage1/train.yaml \
        --tag        "$TAG" \
        --epochs     "$EPOCHS" \
        --image-backbone feature \
        --stage1-only
