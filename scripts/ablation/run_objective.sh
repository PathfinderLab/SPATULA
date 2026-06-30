#!/usr/bin/env bash
# Ablation: FOUNDATION OBJECTIVE (PDF p.5, table A1–A5).
#
# Tests whether MSM alone is the right foundation objective, or whether
# value reconstruction (MVM), hybrid (MSM+MVM), or latent regularisation
# (MSM + Gene-JEPA) yields a better spot encoder.
#
# Variants (set in experiment.yaml `experiment.tx_self.*`):
#   - A1 (msm)        : symbol_w=1.0, value_w=0.0, jepa off       ← MSM only baseline
#   - A2 (mvm)        : symbol_w=0.0, value_w=1.0, jepa off       ← MVM only (denoising)
#   - A3 (msm_mvm)    : symbol_w=1.0, value_w=1.0, jepa off       ← hybrid
#   - A4 (msm_jepa)   : symbol_w=1.0, value_w=0.0, jepa λ=0.1     ← main candidate
#   - A5 (full)       : symbol_w=1.0, value_w=1.0, jepa λ=0.1     ← upper bound
#
# Each run lands at  results/runs/stage1_obj_<A_id>/
#
# Usage:
#     bash scripts/ablation/run_objective.sh
#     bash scripts/ablation/run_objective.sh msm msm_jepa       # subset
#     ABL_PROFILE=fast bash scripts/ablation/run_objective.sh
#
# Priority comparison per PDF p.5: A1 vs A2 vs A4.  A5 is last (high-cost).

source "$(dirname "$0")/_common.sh"

VARIANTS=("$@")
# MVM is reserved for downstream imputation, not a foundation objective
# (per project decision).  Default sweep = MSM only vs MSM+Gene-JEPA.
# To still ablate mvm/msm_mvm/full, pass them explicitly: `... msm mvm full`.
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=(msm msm_jepa)
fi

declare -A MASKING_OBJ
declare -A SYM_W
declare -A VAL_W
declare -A JEPA_EN
declare -A JEPA_W

MASKING_OBJ[msm]="symbol";     SYM_W[msm]="1.0";     VAL_W[msm]="0.0";     JEPA_EN[msm]="false";    JEPA_W[msm]="0.0"
MASKING_OBJ[mvm]="value";      SYM_W[mvm]="0.0";     VAL_W[mvm]="1.0";     JEPA_EN[mvm]="false";    JEPA_W[mvm]="0.0"
MASKING_OBJ[msm_mvm]="both";   SYM_W[msm_mvm]="1.0"; VAL_W[msm_mvm]="1.0"; JEPA_EN[msm_mvm]="false"; JEPA_W[msm_mvm]="0.0"
MASKING_OBJ[msm_jepa]="symbol";SYM_W[msm_jepa]="1.0";VAL_W[msm_jepa]="0.0";JEPA_EN[msm_jepa]="true"; JEPA_W[msm_jepa]="0.1"
MASKING_OBJ[full]="both";      SYM_W[full]="1.0";    VAL_W[full]="1.0";    JEPA_EN[full]="true";    JEPA_W[full]="0.1"

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for V in "${VARIANTS[@]}"; do
    if [[ -z "${MASKING_OBJ[$V]:-}" ]]; then
        echo "[abl] unknown objective variant: $V (use: msm mvm msm_mvm msm_jepa full)" >&2
        exit 1
    fi
    TAG="stage1_obj_${V}"
    EXP="/tmp/exp_${TAG}.yaml"
    # CORRECT schema: experiment.tx_self.* (train.py line ~259 reads from here).
    # The legacy experiment.masking.* path is NOT consumed anywhere.
    make_yaml_override "$EXP" "$BASE_EXP" \
        "experiment.tx_self.masking_obj"        "${MASKING_OBJ[$V]}" \
        "experiment.tx_self.symbol_weight"      "${SYM_W[$V]}" \
        "experiment.tx_self.value_weight"       "${VAL_W[$V]}" \
        "experiment.tx_self.enable_masked_jepa" "${JEPA_EN[$V]}" \
        "experiment.tx_self.jepa_weight"        "${JEPA_W[$V]}"
    ABL_GROUP="objective" ABL_VARIANT="$V" \
    ABL_CHANGED_JSON="{\"experiment.tx_self.masking_obj\":\"${MASKING_OBJ[$V]}\",\"experiment.tx_self.symbol_weight\":${SYM_W[$V]},\"experiment.tx_self.value_weight\":${VAL_W[$V]},\"experiment.tx_self.enable_masked_jepa\":${JEPA_EN[$V]},\"experiment.tx_self.jepa_weight\":${JEPA_W[$V]}}" \
    launch_run "$TAG" "$BASE_DATA" "$BASE_MODEL" "$BASE_TRAIN" "$EXP"
done

echo "[ablation] objective sweep complete.  Runs in results/runs/stage1_obj_*"
