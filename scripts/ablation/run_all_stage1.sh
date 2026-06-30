#!/usr/bin/env bash
# Stage1-only ablation entrypoint.
#
# This wrapper intentionally does NOT train Stage1.5 spatial-JEPA.  It runs
# Stage1 train + Stage1/DLPFC/gene-map eval/figures through the current unified
# pipeline script.
#
# Default suite:
#   1) primary joint MSM + multi_chunk_JEPA
#   2) vocab ablation: 4096 / 8192 / full
#   3) capacity ablation: spatula_lite / mid / large
#   4) value augmentation ablation
#   5) readout eval: CLS vs CLS + gene-token mean
#
# Usage:
#   bash scripts/ablation/run_all_stage1.sh
#   DRY_RUN=1 bash scripts/ablation/run_all_stage1.sh
#   STAGE1_EPOCHS=100 bash scripts/ablation/run_all_stage1.sh
#   NEURAL_LINEAR_PROBE=1 bash scripts/ablation/run_all_stage1.sh
#   MAKE_VIZ=0 bash scripts/ablation/run_all_stage1.sh
#
# Narrow suites:
#   bash scripts/ablation/run_all_stage1.sh core
#   bash scripts/ablation/run_all_stage1.sh vocab
#   bash scripts/ablation/run_all_stage1.sh capacity
#   bash scripts/ablation/run_all_stage1.sh value_aug
#   bash scripts/ablation/run_all_stage1.sh pooling STAGE1_CKPT=...  (or export STAGE1_CKPT first)
#
# Legacy implementation is preserved at:
#   scripts/ablation/run_all_stage1_legacy.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"

MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$MODE" in
    all|stage1_all|full)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh stage1_all "$@"
        ;;
    core|stage1|stage1_core)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh stage1 "$@"
        ;;
    vocab|stage1_vocab)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh vocab "$@"
        ;;
    capacity|cap|stage1_capacity)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh capacity "$@"
        ;;
    value_aug|augmentation|va)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh value_aug "$@"
        ;;
    pooling|pooling_pair|eval_pooling_pair)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh eval_pooling_pair "$@"
        ;;
    cls_mean|eval_cls_mean)
        exec bash scripts/ablation/run_spot_encoder_pipeline.sh eval_cls_mean "$@"
        ;;
    legacy)
        exec bash scripts/ablation/run_all_stage1_legacy.sh "$@"
        ;;
    *)
        echo "Unknown Stage1 mode: $MODE" >&2
        echo "Use one of: all core vocab capacity value_aug pooling cls_mean legacy" >&2
        exit 1
        ;;
esac
