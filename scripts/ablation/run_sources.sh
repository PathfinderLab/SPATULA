#!/usr/bin/env bash
# Ablation: TRAINING POOL SOURCES — measure covariate-shift / multi-source
# benefit.
#
# Variants (set in configs/stage1/data.yaml → sources):
#   - hest_only           HEST 649 samples
#   - hest_st1k           HEST + ST1K (1,321 samples)
#   - all                 HEST + ST1K + spatialcorpus (current default, 1,409)
#
# ⚠  IMPORTANT: switching sources changes the train pool, which can change
#    the VOCAB itself if you re-run prepare with restrict to that subset.
#    This script keeps the SAME vocab (the prepared_expanded one built from
#    all sources) and only filters which shards the dataset enumerates.
#    That gives a true "what if we'd trained on subset X" ablation without
#    rebuilding the vocab.  See dataset.sources in data.yaml for the toggle.
#
# Each run lands at  results/runs/stage1_src_<variant>/
#
# Usage:
#     bash scripts/ablation/run_sources.sh
#     bash scripts/ablation/run_sources.sh hest_only all
#     ABL_EPOCHS=30 bash scripts/ablation/run_sources.sh

source "$(dirname "$0")/_common.sh"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=(hest_only hest_st1k all)
fi

declare -A SRC
SRC[hest_only]='{"hest": true, "st1k": false, "spatialcorpus": false}'
SRC[hest_st1k]='{"hest": true, "st1k": true,  "spatialcorpus": false}'
SRC[all]='{"hest": true, "st1k": true,  "spatialcorpus": true}'

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for V in "${VARIANTS[@]}"; do
    TAG="stage1_src_${V}"
    DATA="/tmp/data_${TAG}.yaml"
    make_yaml_override "$DATA" "$BASE_DATA" \
        "data.sources" "${SRC[$V]}"
    ABL_GROUP="sources" ABL_VARIANT="$V" \
    ABL_CHANGED_JSON="{\"data.sources.variant\":\"$V\"}" \
    launch_run "$TAG" "$DATA" "$BASE_MODEL" "$BASE_TRAIN" "$BASE_EXP"
done

echo "[ablation] sources sweep complete.  Runs in results/runs/stage1_src_*"
