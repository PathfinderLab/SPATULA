#!/usr/bin/env bash
# Ablation: SEQUENCE-LENGTH SAMPLING — how to pick K tokens when a spot
# expresses > max_seq_len genes.
#
# Variants (set in configs/stage1/data.yaml → sampling.strategy):
#   - none       no cap (max_seq_len=0)  ← include for control
#   - random     max_seq_len=K, uniform drop
#   - top_k      max_seq_len=K, keep highest-value K (deterministic)
#   - weighted   max_seq_len=K, multinomial p ∝ value^alpha (alpha=1)
#
# Each variant also keeps must_include curated markers when expressed
# (NLP analogy: don't truncate proper nouns).
#
# Default K = 1024 (~ p95 of seq_len before clip; covers >95% of spots).
#
# Usage:
#     bash scripts/ablation/run_seq_sampling.sh
#     bash scripts/ablation/run_seq_sampling.sh random top_k
#     ABL_EPOCHS=15 ABL_MAX_SEQ_LEN=512 bash scripts/ablation/run_seq_sampling.sh
#
# Combine with vocab_clip via env (the clip file must already exist):
#     ABL_CLIP_INDICES=results/cache/prepared_expanded/clip4096_keep_indices.npy \
#       bash scripts/ablation/run_seq_sampling.sh weighted

source "$(dirname "$0")/_common.sh"

ABL_MAX_SEQ_LEN="${ABL_MAX_SEQ_LEN:-1024}"
ABL_SAMPLING_ALPHA="${ABL_SAMPLING_ALPHA:-1.0}"
ABL_CLIP_INDICES="${ABL_CLIP_INDICES:-}"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=(none random top_k weighted)
fi

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

for V in "${VARIANTS[@]}"; do
    TAG="stage1_samp_${V}"
    DATA="/tmp/data_${TAG}.yaml"
    if [[ "$V" == "none" ]]; then
        OVERRIDES=(
            "data.max_seq_len" "0"
            "data.sampling.strategy" "random"
        )
    else
        OVERRIDES=(
            "data.max_seq_len" "$ABL_MAX_SEQ_LEN"
            "data.sampling.strategy" "$V"
            "data.sampling.alpha" "$ABL_SAMPLING_ALPHA"
            "data.sampling.keep_must_include" "true"
        )
    fi
    if [[ -n "$ABL_CLIP_INDICES" ]]; then
        OVERRIDES+=("data.vocab_clip.keep_indices_path" "$(realpath "$ABL_CLIP_INDICES")")
        TAG="${TAG}_clip$(basename "$ABL_CLIP_INDICES" | sed 's/_keep_indices.npy//')"
    fi
    make_yaml_override "$DATA" "$BASE_DATA" "${OVERRIDES[@]}"
    META_MAX_SEQ_LEN="0"
    [[ "$V" != "none" ]] && META_MAX_SEQ_LEN="$ABL_MAX_SEQ_LEN"
    ABL_GROUP="seq_sampling" ABL_VARIANT="$V" \
    ABL_CHANGED_JSON="{\"data.sampling.strategy\":\"$V\",\"data.max_seq_len\":${META_MAX_SEQ_LEN}}" \
    launch_run "$TAG" "$DATA" "$BASE_MODEL" "$BASE_TRAIN" "$BASE_EXP"
done

echo "[ablation] seq_sampling sweep complete.  Runs in results/runs/stage1_samp_*"
