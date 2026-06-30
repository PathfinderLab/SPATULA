#!/usr/bin/env bash
# Thin wrapper around scripts/sweep.py.
#
# All sweep knobs (experiments, epochs, batch_size, freeze, GPUs, …) now live
# in YAML — see configs/sweep/stage1_ours_tx.yaml.  This script only handles env
# hygiene (thread caps, NCCL timeout) and dispatch.
#
# Usage:
#   bash scripts/run_experiments.sh                              # configs/sweep/stage1_ours_tx.yaml
#   bash scripts/run_experiments.sh configs/sweep/my_sweep.yaml  # any other file
#   bash scripts/run_experiments.sh configs/sweep/stage1_ours_tx.yaml --only jepa
#
# To experiment with a one-off variant: copy default.yaml, edit, point this
# script at it.  No env vars to remember.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# CPU / NCCL hygiene — needed in the parent shell so accelerate child ranks inherit.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# First positional arg: sweep YAML path (optional).
SWEEP="${1:-configs/sweep/stage1_ours_tx.yaml}"
shift || true

if [[ ! -f "$SWEEP" ]]; then
    echo "ERROR: sweep file not found: $SWEEP" >&2
    echo "Available templates under configs/sweep/:" >&2
    ls configs/sweep/ 2>/dev/null | sed 's/^/  /' >&2
    exit 2
fi

exec python scripts/sweep.py --sweep "$SWEEP" "$@"
