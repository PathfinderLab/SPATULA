#!/usr/bin/env bash
# Run focused Stage-1 objective/mask ablations through Stage 1.5.
#
# Default candidates:
#   mask_0p15_msm_only       : MSM only, mask_ratio=0.15
#   mask_0p15_view_jepa_w005     : MSM + view-JEPA predictor, λ=0.05
#   mask_0p15_view_jepa_w010     : MSM + view-JEPA predictor, λ=0.10
#   mask_0p30_msm_dino_warm      : stronger corruption check, mask_ratio=0.30
#
# Usage:
#   bash scripts/ablation/run_stage15_cascade.sh
#   bash scripts/ablation/run_stage15_cascade.sh mask_0p15_dino_late_no_koleo
#   ABL_EPOCHS=50 STAGE15_EPOCHS=30 bash scripts/ablation/run_stage15_cascade.sh
#   SKIP_EXISTING_STAGE1=1 SKIP_EXISTING_STAGE15=1 bash scripts/ablation/run_stage15_cascade.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    CANDIDATES=(mask_0p15_msm_only mask_0p15_view_jepa_w005 mask_0p15_view_jepa_w010 mask_0p30_msm_dino_warm)
fi

export ABL_EPOCHS="${ABL_EPOCHS:-50}"
export ABL_BATCH="${ABL_BATCH:-512}"
export ABL_STAGE1_QUICK_EVERY="${ABL_STAGE1_QUICK_EVERY:-5}"
export ABL_CLEAN_MSM_EVERY="${ABL_CLEAN_MSM_EVERY:-1}"
export ABL_GENE_SET_EVERY="${ABL_GENE_SET_EVERY:-10}"
STAGE15_EPOCHS="${STAGE15_EPOCHS:-30}"
BATCH_SIZE_STAGE15="${BATCH_SIZE_STAGE15:-8}"
SUBGRAPH_SIZE="${SUBGRAPH_SIZE:-256}"
TX_ENCODE_BATCH="${TX_ENCODE_BATCH:-256}"
SKIP_EXISTING_STAGE1="${SKIP_EXISTING_STAGE1:-1}"
SKIP_EXISTING_STAGE15="${SKIP_EXISTING_STAGE15:-0}"

for c in "${CANDIDATES[@]}"; do
    stage1_tag="stage1_full_${c}"
    stage1_ckpt="results/runs/${stage1_tag}/ckpt_tx_encoder_best.pt"
    stage15_tag="stage15_${c}_separate_spot_ego"
    stage15_ckpt="results/runs/${stage15_tag}/ckpt_spatial_best.pt"

    echo "=========================================================="
    echo "[cascade-abl] candidate=$c"
    echo "  stage1_tag:  $stage1_tag"
    echo "  stage15_tag: $stage15_tag"
    echo "=========================================================="

    if [[ "$SKIP_EXISTING_STAGE1" == "1" && -f "$stage1_ckpt" ]]; then
        echo "[cascade-abl] Stage1 exists, skipping: $stage1_ckpt"
    else
        bash scripts/ablation/run_all.sh "$c"
    fi

    if [[ ! -f "$stage1_ckpt" ]]; then
        echo "[cascade-abl] ERROR: missing Stage1 ckpt: $stage1_ckpt"
        exit 1
    fi

    if [[ "$SKIP_EXISTING_STAGE15" == "1" && -f "$stage15_ckpt" ]]; then
        echo "[cascade-abl] Stage1.5 exists, skipping: $stage15_ckpt"
    else
        STAGE1_CKPT="$stage1_ckpt" \
        TAG="$stage15_tag" \
        EPOCHS="$STAGE15_EPOCHS" \
        BATCH_SIZE="$BATCH_SIZE_STAGE15" \
        SUBGRAPH_SIZE="$SUBGRAPH_SIZE" \
        TX_ENCODE_BATCH="$TX_ENCODE_BATCH" \
            bash scripts/train/stage15_main.sh
    fi

done

echo "[cascade-abl] done."
