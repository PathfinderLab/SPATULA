#!/usr/bin/env bash
# STAGE 1 — pretrain the OURS gene encoder (top_hvg_gene) ALONE.
#
# What this does:
#   - Loads configs/sweep/stage1_ours_tx.yaml which already points at:
#       data_yaml       = configs/stage1/data.yaml
#       model_yaml      = configs/stage1/model.yaml         (kind=top_hvg_gene)
#       train_yaml      = configs/stage1/train.yaml
#       experiment_yaml = configs/stage1/experiment.yaml    (MSM + JEPA config)
#   - Forwards --stage1-only to train.py:
#       * align/gene_recon/image_recon weights forced to 0
#       * masking_obj / mask_ratio propagated into model.transcriptomics.top_hvg_gene
#       * monitor.gene_set_every honored for the gene-set co-occurrence monitor
#   - Saves ckpt_tx_encoder.pt for Stage-2 reuse.
#
# Usage:
#   bash scripts/train/stage1.sh                                   # 8 GPUs, defaults
#   bash scripts/train/stage1.sh configs/sweep/stage1_ours_tx.yaml --only stage1
#
# Env knobs:
#   STRATIFY=1   — re-emit splits.json with organ-stratified 8:1:1
#
# Stage 2 follow-up:
#   bash scripts/train/stage2.sh <stage1_ckpt_tx_encoder.pt>

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Standard env hygiene.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

SWEEP="${1:-configs/sweep/stage1_ours_tx.yaml}"
shift || true

# ── Optional Stage 0 — re-split with organ stratification ─────────────
if [[ "${STRATIFY:-0}" == "1" ]]; then
    echo "[stage1] re-splitting with organ stratification (8:1:1) ..."
    python scripts/data/resplit.py --val_frac 0.10 --test_frac 0.10 --backup
fi

# ── Inject stage1_only=true into a tmp sweep YAML ────────────────────
TMP_SWEEP=$(mktemp --suffix=.yaml -p /tmp stage1_sweep_XXXX)
python - <<EOF
import yaml
s = yaml.safe_load(open("${SWEEP}"))
s["sweep"]["stage1_only"] = True
open("${TMP_SWEEP}", "w").write(yaml.dump(s))
EOF
echo "[stage1] sweep → ${TMP_SWEEP}  (stage1_only=true injected)"

# ── Dispatch ─────────────────────────────────────────────────────────
exec python scripts/sweep.py --sweep "${TMP_SWEEP}" "$@"
