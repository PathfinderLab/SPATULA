#!/usr/bin/env bash
# Ablation: REGION TOKEN STRATEGY — how the anchor's region/niche context is
# presented to the spatial backbone, and what gets masked.
#
# Four variants:
#   R0  no-region        data.region.enable=false                              [single-token control]
#   R1  fused            data.region.enable=true, region_token_mode=fused      [current baseline]
#   R2  separate-spot    region_token_mode=separate, mask_target=spot          [I-JEPA-style "region predicts spot" — main candidate]
#   R3  separate-both    region_token_mode=separate, mask_target=both          [block masking on both streams]
#
# Each run lands at  results/runs/stage15_region_token_<variant>/
#
# Usage:
#     bash scripts/ablation/run_region_token.sh                  # full R0..R3
#     bash scripts/ablation/run_region_token.sh R2               # one variant
#     ABL_SUBGRAPH_KIND=ego bash scripts/ablation/run_region_token.sh
#
# Stage 1.5 trainer is single-process; this script does NOT use accelerate.

source "$(dirname "$0")/_common.sh"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=(R0 R1 R2 R3)
fi

BASE_DATA="configs/stage15/data.yaml"
BASE_MODEL="configs/stage15/model.yaml"
BASE_TRAIN="configs/stage15/train.yaml"
BASE_EXP="configs/stage15/experiment.yaml"

# Subgraph kind override (default: keep base) — recommend `ego` for true tissue locality.
SUBGRAPH_KIND="${ABL_SUBGRAPH_KIND:-}"

for V in "${VARIANTS[@]}"; do
    case "$V" in
        R0) REGION_ON=false; TOKEN_MODE=fused;    MASK_TARGET=spot ;;
        R1) REGION_ON=true;  TOKEN_MODE=fused;    MASK_TARGET=spot ;;
        R2) REGION_ON=true;  TOKEN_MODE=separate; MASK_TARGET=spot ;;
        R3) REGION_ON=true;  TOKEN_MODE=separate; MASK_TARGET=both ;;
        *) echo "[ablation] unknown variant $V (use R0..R3)"; exit 1 ;;
    esac

    TAG="stage15_region_token_${V}"
    DATA="/tmp/data_${TAG}.yaml"
    MODEL="/tmp/model_${TAG}.yaml"
    EXP="/tmp/exp_${TAG}.yaml"

    data_overrides=("data.region.enable" "$REGION_ON")
    if [[ -n "$SUBGRAPH_KIND" ]]; then
        data_overrides+=("data.subgraph_kind" "$SUBGRAPH_KIND")
    fi
    make_yaml_override "$DATA"  "$BASE_DATA"  "${data_overrides[@]}"
    make_yaml_override "$MODEL" "$BASE_MODEL" \
        "model.spatial.region_token_mode" "$TOKEN_MODE"
    make_yaml_override "$EXP"   "$BASE_EXP" \
        "experiment.jepa.mask_target" "$MASK_TARGET"

    echo "=========================================================="
    echo "[ablation] $TAG  (region=$REGION_ON, token=$TOKEN_MODE, mask=$MASK_TARGET, subgraph=${SUBGRAPH_KIND:-base})"
    echo "=========================================================="
    # Stage 1.5 trainer is NOT DDP-aware — single process.
    python scripts/train/stage15.py \
        --data       "$DATA" \
        --model      "$MODEL" \
        --train      "$BASE_TRAIN" \
        --experiment "$EXP" \
        --tag        "$TAG" \
        --epochs     "$ABL_EPOCHS"
done

echo "[ablation] region_token sweep complete.  Runs in results/runs/stage15_region_token_*"
