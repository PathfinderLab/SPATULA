#!/usr/bin/env bash
# Ablation: GENE-LEVEL PREDICTIVE JEPA on/off — PDF p.8 (E2/E3).
#
# This is the Stage-1 RNA encoder's auxiliary regulariser: predict masked-
# gene latents from context via an EMA-teacher predictor.  NOT the Stage-2
# image↔RNA align JEPA (that's a different module — experiment.align.*).
#
# Variants (set in configs/stage1/experiment.yaml → tx_self.*):
#   - off    (E1) : enable_masked_jepa=false, jepa_weight=0       — MSM only
#   - lite   (E2) : enable_masked_jepa=true,  jepa_weight=0.1     — weak reg
#   - paper  (E3) : enable_masked_jepa=true,  jepa_weight=0.3     — strong reg
#
# Each run lands at  results/runs/stage1_jepa_<variant>/
#
# Usage:
#     bash scripts/ablation/run_jepa.sh
#     bash scripts/ablation/run_jepa.sh off paper
#     ABL_PROFILE=triage bash scripts/ablation/run_jepa.sh

source "$(dirname "$0")/_common.sh"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=(off lite paper)
fi

declare -A WEIGHT
declare -A ENABLE
WEIGHT[off]="0.0";  ENABLE[off]="false"
WEIGHT[lite]="0.1"; ENABLE[lite]="true"
WEIGHT[paper]="0.3"; ENABLE[paper]="true"

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for V in "${VARIANTS[@]}"; do
    TAG="stage1_jepa_${V}"
    EXP="/tmp/exp_${TAG}.yaml"
    # CORRECT keys: Gene-level JEPA lives under experiment.tx_self.*, not
    # experiment.align.* (that's Stage-2 image↔RNA alignment, weight=0 in
    # Stage 1 anyway).
    make_yaml_override "$EXP" "$BASE_EXP" \
        "experiment.tx_self.enable_masked_jepa" "${ENABLE[$V]}" \
        "experiment.tx_self.jepa_weight"        "${WEIGHT[$V]}"
    ABL_GROUP="jepa" ABL_VARIANT="$V" \
    ABL_CHANGED_JSON="{\"experiment.tx_self.enable_masked_jepa\":${ENABLE[$V]},\"experiment.tx_self.jepa_weight\":${WEIGHT[$V]}}" \
    launch_run "$TAG" "$BASE_DATA" "$BASE_MODEL" "$BASE_TRAIN" "$EXP"
done

echo "[ablation] Gene-JEPA sweep complete.  Runs in results/runs/stage1_jepa_*"
