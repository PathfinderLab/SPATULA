#!/usr/bin/env bash
# Ablation: NORMALIZE MODE — compare runtime gene_norm strategies.
#
# Variants (set in configs/stage1/data.yaml → gene_norm.mode):
#   - none           : raw log1p (baseline / debugging)
#   - nonzero_z      : (x - μ_nz) / σ_nz, zeros preserved  ← current default
#   - global_median  : x / nonzero_median(g)  (Geneformer-style)
#
# Each run lands at  results/runs/stage1_norm_<mode>/
#
# Usage:
#     bash scripts/ablation/run_normalize.sh
#     bash scripts/ablation/run_normalize.sh nonzero_z global_median   # subset
#     ABL_EPOCHS=30 bash scripts/ablation/run_normalize.sh

source "$(dirname "$0")/_common.sh"

MODES=("$@")
if [[ ${#MODES[@]} -eq 0 ]]; then
    MODES=(none nonzero_z global_median)
fi

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for MODE in "${MODES[@]}"; do
    TAG="stage1_norm_${MODE}"
    DATA="/tmp/data_${TAG}.yaml"
    make_yaml_override "$DATA" "$BASE_DATA" \
        "data.gene_norm.mode" "$MODE"
    ABL_GROUP="normalize" ABL_VARIANT="$MODE" \
    ABL_CHANGED_JSON="{\"data.gene_norm.mode\":\"$MODE\"}" \
    launch_run "$TAG" "$DATA" "$BASE_MODEL" "$BASE_TRAIN" "$BASE_EXP"
done

echo "[ablation] normalize sweep complete.  Runs in results/runs/stage1_norm_*"
