#!/usr/bin/env bash
# Full Stage-1 -> Stage-1.5 ablation cascade.
#
# Usage:
#   bash scripts/ablation/run_all.sh
#   bash scripts/ablation/run_all.sh default obj_msm_multi_chunk
#   ABL_BATCH=1024 bash scripts/ablation/run_all.sh default
#
# For separate phases:
#   bash scripts/ablation/run_all_stage1.sh   # Stage1 only
#   bash scripts/ablation/run_all_stage15.sh  # Stage1.5 only from existing Stage1 ckpts

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"

bash scripts/ablation/run_all_stage1.sh "$@"
bash scripts/ablation/run_all_stage15.sh "$@"
