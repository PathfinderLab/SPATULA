#!/usr/bin/env bash
# Ablation: REGION AGGREGATION — how anchor + neighbor tokens are pooled
# into the region_tx / region_img inputs of the Stage 1.5 SpatialEncoder.
#
# Variants (set in configs/stage15/data.yaml → region.*):
#   tx_agg knob:
#     mean       Σ log1p(x_i) / (k+1)             — HyperST default
#     sum_log1p  log1p(Σ expm1(x))                — sum in raw space, then re-log1p
#     weighted   exp(-d/σ) weighted mean          — distance-aware
#   img_pool knob:
#     mean       mean of neighbor uni_feat        — cheap default
#     attn       attention pool (TODO; mean fallback)
#
# Variant grid (default): tx_agg ∈ {mean, sum_log1p, weighted} × img_pool ∈ {mean}.
# To also sweep image pool: set ABL_IMG_POOLS="mean attn".
#
# Each run lands at  results/runs/stage15_region_<tx_agg>__<img_pool>/
#
# Usage:
#     bash scripts/ablation/run_region_agg.sh                   # default 3×1 grid
#     bash scripts/ablation/run_region_agg.sh mean weighted     # tx_agg variants only
#     ABL_IMG_POOLS="mean attn" bash scripts/ablation/run_region_agg.sh
#
# Profile + epoch knobs are inherited from _common.sh (ABL_PROFILE=triage etc.).

source "$(dirname "$0")/_common.sh"

TX_VARIANTS=("$@")
if [[ ${#TX_VARIANTS[@]} -eq 0 ]]; then
    TX_VARIANTS=(mean sum_log1p weighted)
fi
IMG_VARIANTS=( ${ABL_IMG_POOLS:-mean} )

BASE_DATA="configs/stage15/data.yaml"
BASE_MODEL="configs/stage15/model.yaml"
BASE_TRAIN="configs/stage15/train.yaml"
BASE_EXP="configs/stage15/experiment.yaml"

for TX in "${TX_VARIANTS[@]}"; do
    for IMG in "${IMG_VARIANTS[@]}"; do
        TAG="stage15_region_${TX}__${IMG}"
        DATA="/tmp/data_${TAG}.yaml"
        make_yaml_override "$DATA" "$BASE_DATA" \
            "data.region.enable"   "true" \
            "data.region.tx_agg"   "$TX" \
            "data.region.img_pool" "$IMG"

        # Stage 1.5 uses its own scripts/train/stage15.py wrapper instead of
        # the shared scripts/train.py + accelerate launch helper.  Build a
        # bespoke launch that mirrors launch_run()'s env hygiene.
        echo "=========================================================="
        echo "[ablation] $TAG  (tx_agg=$TX, img_pool=$IMG)"
        echo "=========================================================="
        accelerate launch --num_processes "$ABL_NUM_PROC" --mixed_precision "$ABL_MP" \
            scripts/train/stage15.py \
                --data       "$DATA" \
                --model      "$BASE_MODEL" \
                --train      "$BASE_TRAIN" \
                --experiment "$BASE_EXP" \
                --tag        "$TAG" \
                --epochs     "$ABL_EPOCHS"
    done
done

echo "[ablation] region_agg sweep complete.  Runs in results/runs/stage15_region_*"
