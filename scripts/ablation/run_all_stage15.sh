#!/usr/bin/env bash
# Full Stage-1.5 cascade for already-trained Stage-1 ablation candidates.
#
# Usage:
#   bash scripts/ablation/run_all_stage15.sh
#   bash scripts/ablation/run_all_stage15.sh default obj_msm_multi_chunk
#   ABL_STAGE15_BATCH=8 ABL_STAGE15_EPOCHS=30 bash scripts/ablation/run_all_stage15.sh
#
# It expects Stage-1 checkpoints at:
#   results/runs/stage1_full_<candidate>/ckpt_tx_encoder_best.pt
# and writes Stage-1.5 runs to:
#   results/runs/stage15_full_<candidate>_separate_spot_ego/

set -euo pipefail

export ABL_STAGE15_EPOCHS="${ABL_STAGE15_EPOCHS:-30}"
export ABL_STAGE15_BATCH="${ABL_STAGE15_BATCH:-8}"
export ABL_STAGE15_SUBGRAPH_SIZE="${ABL_STAGE15_SUBGRAPH_SIZE:-256}"
export ABL_STAGE15_TX_ENCODE_BATCH="${ABL_STAGE15_TX_ENCODE_BATCH:-256}"
export ABL_STAGE15_SKIP_EXISTING="${ABL_STAGE15_SKIP_EXISTING:-1}"
export ABL_STAGE15_EVAL_MAX_SAMPLES="${ABL_STAGE15_EVAL_MAX_SAMPLES:-20}"

ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    CANDIDATES=(
        obj_msm_multi_chunk
        default
        vocab_8192 vocab_full
        cap_lite cap_large
        seq_random_256
        va_keep
    )
fi

run_stage15_candidate() {
    local name="$1"
    local stage1_tag="stage1_full_${name}"
    local stage1_ckpt="results/runs/${stage1_tag}/ckpt_tx_encoder_best.pt"
    local stage15_tag="stage15_full_${name}_separate_spot_ego"
    local stage15_ckpt="results/runs/${stage15_tag}/ckpt_spatial_best.pt"

    if [[ ! -f "$stage1_ckpt" ]]; then
        echo "[run_all_stage15] ERROR: missing Stage1 ckpt: $stage1_ckpt"
        echo "[run_all_stage15] Run Stage1 first, e.g. bash scripts/ablation/run_all_stage1.sh $name"
        exit 1
    fi
    if [[ "$ABL_STAGE15_SKIP_EXISTING" == "1" && -f "$stage15_ckpt" ]]; then
        echo "[run_all_stage15] Stage1.5 exists, skipping: $stage15_ckpt"
        return 0
    fi

    echo "=========================================================="
    echo "[run_all_stage15] candidate=$name  tag=$stage15_tag"
    echo "  stage1_ckpt: $stage1_ckpt"
    echo "  epochs: $ABL_STAGE15_EPOCHS  batch=$ABL_STAGE15_BATCH  subgraph=$ABL_STAGE15_SUBGRAPH_SIZE  tx_encode_batch=$ABL_STAGE15_TX_ENCODE_BATCH"
    echo "=========================================================="
    STAGE1_CKPT="$stage1_ckpt"     TAG="$stage15_tag"     EPOCHS="$ABL_STAGE15_EPOCHS"     BATCH_SIZE="$ABL_STAGE15_BATCH"     SUBGRAPH_SIZE="$ABL_STAGE15_SUBGRAPH_SIZE"     TX_ENCODE_BATCH="$ABL_STAGE15_TX_ENCODE_BATCH"         bash scripts/train/stage15_main.sh
}

for c in "${CANDIDATES[@]}"; do
    run_stage15_candidate "$c"
done

mapfile -t STAGE15_CKPTS < <(for c in "${CANDIDATES[@]}"; do echo "results/runs/stage15_full_${c}_separate_spot_ego/ckpt_spatial_best.pt"; done | xargs -r ls 2>/dev/null || true)
if [[ ${#STAGE15_CKPTS[@]} -gt 0 ]]; then
    echo "[run_all_stage15] writing combined Stage1.5 in-dist CSV..."
    PYTHONPATH=src python scripts/eval/stage15_indist.py         --prepared-dir results/cache/prepared_expanded         --split test         --ckpts "${STAGE15_CKPTS[@]}"         --max-samples "$ABL_STAGE15_EVAL_MAX_SAMPLES"         --out results/eval/stage15_core_full_indist_compare.csv
fi

echo "[run_all_stage15] Stage1.5 runs: results/runs/stage15_full_*_separate_spot_ego/"
echo "[run_all_stage15] combined Stage1.5 CSV: results/eval/stage15_core_full_indist_compare.csv"
