#!/usr/bin/env bash
# Ablation: MASK RATIO — fraction of real tokens replaced with [MASK] in MSM.
#
# Variants (set in configs/stage1/experiment.yaml → masking.mask_ratio):
#   - 0.15  ← BERT default, current
#   - 0.30  ← SCGPT/Geneformer use higher ratios on long sequences
#   - 0.50  ← aggressive; tests gradient signal density
#
# Each run lands at  results/runs/stage1_mask_<ratio>/
#
# Usage:
#     bash scripts/ablation/run_mask_ratio.sh
#     bash scripts/ablation/run_mask_ratio.sh 0.15 0.30
#     ABL_EPOCHS=30 bash scripts/ablation/run_mask_ratio.sh

source "$(dirname "$0")/_common.sh"

RATIOS=("$@")
if [[ ${#RATIOS[@]} -eq 0 ]]; then
    RATIOS=(0.15 0.30 0.50)
fi

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for R in "${RATIOS[@]}"; do
    TAG="stage1_mask_$(echo "$R" | tr '.' 'p')"   # e.g. 0.15 → 0p15
    EXP="/tmp/exp_${TAG}.yaml"
    # CORRECT key: experiment.tokenizer.mask_ratio
    # (train.py line ~269 reads exp_tok["mask_ratio"] and propagates to
    #  model.transcriptomics.top_hvg_gene.mask_ratio.)
    make_yaml_override "$EXP" "$BASE_EXP" \
        "experiment.tokenizer.mask_ratio" "$R"
    ABL_GROUP="mask_ratio" ABL_VARIANT="$R" \
    ABL_CHANGED_JSON="{\"experiment.tokenizer.mask_ratio\":$R}" \
    launch_run "$TAG" "$BASE_DATA" "$BASE_MODEL" "$BASE_TRAIN" "$EXP"
done

echo "[ablation] mask_ratio sweep complete.  Runs in results/runs/stage1_mask_*"
