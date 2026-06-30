#!/usr/bin/env bash
# Ablation: VOCAB SIZE — runtime clip by priority_rank.
#
# What this does:
#   - Builds `clipN_keep_indices.npy` for each requested N via
#     scripts/data/make_clipped_vocab.py (5 sec each).
#   - For each N, writes a /tmp data.yaml that points data.vocab_clip.keep_indices_path
#     at the right file.
#   - Launches Stage 1 training in sequence (one at a time — 8 GPUs shared).
#
# Each run lands at  results/runs/stage1_clipN/
#
# Usage:
#     bash scripts/ablation/run_vocab_clip.sh                   # default sizes
#     bash scripts/ablation/run_vocab_clip.sh 4096 8192         # custom
#     ABL_EPOCHS=20 bash scripts/ablation/run_vocab_clip.sh     # shorter sweep
#
# Notes:
#   - `make_clipped_vocab.py` keeps all 152 must_include markers in every
#     clip (they live at the top of priority_rank), so smaller-N runs
#     do NOT drop curated clinical markers.
#   - The "full" 19183 baseline is included if no args are given.

source "$(dirname "$0")/_common.sh"

# Default ablation sweep: 2048 / 4096 / 8192 / full (19183).
SIZES=("$@")
if [[ ${#SIZES[@]} -eq 0 ]]; then
    SIZES=(2048 4096 8192 full)
fi

PREP="results/cache/prepared_expanded"
BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for N in "${SIZES[@]}"; do
    KEEP=""
    if [[ "$N" == "full" ]]; then
        TAG="stage1_vocab_full"
        DATA="$BASE_DATA"     # data.yaml has vocab_clip null by default
    else
        TAG="stage1_vocab_clip${N}"
        KEEP="$PREP/clip${N}_keep_indices.npy"
        if [[ ! -f "$KEEP" ]]; then
            echo "[ablation] building clip${N} vocab..."
            python scripts/data/make_clipped_vocab.py --top-k "$N"
        fi
        DATA="/tmp/data_${TAG}.yaml"
        make_yaml_override "$DATA" "$BASE_DATA" \
            "data.vocab_clip.keep_indices_path" "$(realpath "$KEEP")"
    fi
    ABL_GROUP="vocab_clip" ABL_VARIANT="${N}" \
    ABL_CHANGED_JSON="{\"data.vocab_clip.keep_indices_path\":\"${KEEP}\",\"vocab_size\":\"${N}\"}" \
    launch_run "$TAG" "$DATA" "$BASE_MODEL" "$BASE_TRAIN" "$BASE_EXP"
done

echo "[ablation] vocab_clip sweep complete.  Runs in results/runs/stage1_vocab_*"
