#!/usr/bin/env bash
# STAGE 2 — VL-JEPA-style multimodal align with a FROZEN Stage-1 tx encoder.
#
# Setup:
#   - Tx encoder  : load from <stage1_ckpt> and freeze (no grads, eval() mode).
#   - Image trunk : UNI ViT-G/14, LoRA-tuned.
#   - Trainable  : LoRA + projectors + decoders + JEPA predictor.
#   - Objective  : align (per experiment) + image_recon + gene_recon (NO tx_self).
#
# Usage:
#   bash scripts/train/stage2.sh <stage1_ckpt> [extra args...]
#   bash scripts/train/stage2.sh \
#       results/runs/stage1_ours_tx_stage1_feature/ckpt_tx_encoder.pt
#   bash scripts/train/stage2.sh <stage1_ckpt> --only jepa
#
# Env knobs:
#   SWEEP=configs/sweep/<file>.yaml   — override default stage2 sweep
#   IMG_BACKBONE=uni|feature           — default uni
#   UNI_TUNE=lora|adapter|partial:4|none|all
#   BATCH_SIZE=64
#   EPOCHS=100

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

if [[ $# -lt 1 ]]; then
    echo "[stage2] usage: $0 <stage1_ckpt> [extra args...]" >&2
    echo "[stage2]        e.g. results/runs/stage1_ours_tx_stage1_feature/ckpt_tx_encoder.pt" >&2
    exit 1
fi
TX_CKPT="$1"; shift
if [[ ! -f "${TX_CKPT}" ]]; then
    echo "[stage2] ERROR: stage1 ckpt not found: ${TX_CKPT}" >&2
    exit 1
fi
echo "[stage2] tx_ckpt = ${TX_CKPT}"

SWEEP="${SWEEP:-configs/sweep/stage2_align.yaml}"

# ── Stage 2a — temp model config that forces kind=top_hvg_gene ─────────
TMP_MODEL=$(mktemp --suffix=.yaml -p /tmp stage2_model_XXXX)
python - <<EOF
import yaml
c = yaml.safe_load(open("configs/stage1/model.yaml"))
c["model"]["transcriptomics"]["kind"] = "top_hvg_gene"
c["model"]["transcriptomics"]["freeze"] = True
open("${TMP_MODEL}", "w").write(yaml.dump(c))
EOF
echo "[stage2] model override → ${TMP_MODEL}"

# ── Stage 2b — patch sweep with tx_ckpt + model yaml ───────────────────
TMP_SWEEP=$(mktemp --suffix=.yaml -p /tmp stage2_sweep_XXXX)
python - <<EOF
import yaml
s = yaml.safe_load(open("${SWEEP}"))
s["sweep"].setdefault("model_yaml", "${TMP_MODEL}")
s["sweep"]["tx_ckpt"] = "${TX_CKPT}"
open("${TMP_SWEEP}", "w").write(yaml.dump(s))
EOF

# ── Stage 2c — dispatch ────────────────────────────────────────────────
exec python scripts/sweep.py --sweep "${TMP_SWEEP}" "$@"
