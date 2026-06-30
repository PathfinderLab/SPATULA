#!/usr/bin/env bash
# Sequential spot-encoder ablation:
#
#   Stage 1      : MSM only
#   Stage 1.25   : trainable multi-chunk JEPA refinement initialized from MSM ckpt
#   Stage 1.5    : spatial JEPA initialized from the Stage 1.25 tx encoder
#
# This script is intentionally separate from run_all_stage1.sh/run_all_stage15.sh
# so the current mainline (Stage1 intra-spot -> Stage1.5 inter-spot) remains
# available while this cleaner MSM -> chunk -> spatial cascade can be compared.
#
# Usage:
#   bash scripts/ablation/run_all_stage1_chunk_stage15.sh
#   STAGE1_CKPT=results/runs/<msm>/ckpt_tx_encoder_best.pt \
#     bash scripts/ablation/run_all_stage1_chunk_stage15.sh
#   SKIP_BASE_STAGE1=1 STAGE1_CKPT=... SKIP_EVAL=1 bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

source "${ROOT}/scripts/ablation/_common.sh" 1>/dev/null
cd "$ROOT"

PREP="results/cache/prepared_expanded"
STAGE1_CAPACITY="${STAGE1_CAPACITY:-spatula_mid}"
STAGE1_TAG="${STAGE1_TAG:-stage1_seq_msm_${STAGE1_CAPACITY}}"
STAGE125_TAG="${STAGE125_TAG:-stage125_seq_chunk_jepa_${STAGE1_CAPACITY}}"
STAGE15_TAG="${STAGE15_TAG:-stage15_seq_spatial_from_${STAGE125_TAG}}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE125_EPOCHS="${STAGE125_EPOCHS:-30}"
STAGE15_EPOCHS="${STAGE15_EPOCHS:-30}"
NUM_PROC_STAGE1="${NUM_PROC_STAGE1:-8}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
STAGE125_BATCH="${STAGE125_BATCH:-auto}"
STAGE125_LR_MULT="${STAGE125_LR_MULT:-1.0}"
STAGE125_MASK_RATIO="${STAGE125_MASK_RATIO:-0.0}"
STAGE125_MC_WEIGHT="${STAGE125_MC_WEIGHT:-0.30}"
STAGE125_MC_KOLEO="${STAGE125_MC_KOLEO:-0.0}"
STAGE125_MC_REGULARIZER="${STAGE125_MC_REGULARIZER:-vicreg}"
STAGE125_VICREG_VAR_WEIGHT="${STAGE125_VICREG_VAR_WEIGHT:-0.05}"
STAGE125_VICREG_COV_WEIGHT="${STAGE125_VICREG_COV_WEIGHT:-0.01}"
STAGE125_VICREG_GAMMA="${STAGE125_VICREG_GAMMA:-1.0}"
STAGE125_TARGET_ID_SCALE="${STAGE125_TARGET_ID_SCALE:-0.25}"
STAGE125_TARGET_CHUNKS="${STAGE125_TARGET_CHUNKS:-auto}"
STAGE125_TARGET_SCALE="${STAGE125_TARGET_SCALE:-0.15,0.25}"
STAGE125_CONTEXT_SCALE="${STAGE125_CONTEXT_SCALE:-0.45,0.65}"
STAGE125_WARMUP="${STAGE125_WARMUP:-0}"
STAGE125_RAMP="${STAGE125_RAMP:-5}"
STAGE15_BATCH="${STAGE15_BATCH:-8}"
STAGE15_SUBGRAPH_SIZE="${STAGE15_SUBGRAPH_SIZE:-256}"
STAGE15_TX_ENCODE_BATCH="${STAGE15_TX_ENCODE_BATCH:-256}"
SKIP_BASE_STAGE1="${SKIP_BASE_STAGE1:-0}"
SKIP_CHUNK_STAGE="${SKIP_CHUNK_STAGE:-0}"
SKIP_STAGE15="${SKIP_STAGE15:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_DLPFC="${RUN_DLPFC:-1}"
DLPFC_DIR="${DLPFC_DIR:-/data/spatiallibd}"

ensure_clip4096() {
    local f="$PREP/clip4096_keep_indices.npy"
    if [[ ! -f "$f" ]]; then
        echo "[seq] building clip4096 vocab indices..."
        python scripts/data/make_clipped_vocab.py --top-k 4096
    fi
    realpath "$f"
}

capacity_overrides() {
    local cap="$1"
    case "$cap" in
        spatula_lite)
            echo "1024 2 256 4 4 1024 512" ;;
        spatula_mid)
            echo "1536 2 384 6 6 1536 384" ;;
        spatula_large)
            echo "2048 2 512 6 8 2048 256" ;;
        *) echo "[seq] unknown STAGE1_CAPACITY=$cap" >&2; exit 1 ;;
    esac
}

if [[ "$SKIP_BASE_STAGE1" == "0" && -z "$STAGE1_CKPT" ]]; then
    echo "=========================================================="
    echo "[seq] Stage 1 MSM baseline"
    echo "  tag=$STAGE1_TAG capacity=$STAGE1_CAPACITY epochs=$STAGE1_EPOCHS"
    echo "=========================================================="
    STAGE1_OBJECTIVE=msm_only \
    STAGE1_CAPACITY="$STAGE1_CAPACITY" \
    STAGE1_TAG="$STAGE1_TAG" \
    STAGE1_EPOCHS="$STAGE1_EPOCHS" \
    NUM_PROC_STAGE1="$NUM_PROC_STAGE1" \
    SKIP_STAGE15=1 SKIP_EVAL=1 \
        bash scripts/run_all.sh
    STAGE1_CKPT="results/runs/${STAGE1_TAG}/ckpt_tx_encoder_best.pt"
fi
if [[ -z "$STAGE1_CKPT" ]]; then
    STAGE1_CKPT="results/runs/${STAGE1_TAG}/ckpt_tx_encoder_best.pt"
fi
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "[seq] ERROR: missing Stage1 MSM checkpoint: $STAGE1_CKPT" >&2
    exit 1
fi

STAGE125_CKPT="results/runs/${STAGE125_TAG}/ckpt_tx_encoder_best.pt"
if [[ "$SKIP_CHUNK_STAGE" == "0" ]]; then
    read -r TX_HID TX_LAY TOK_DIM TOK_LAY TOK_HEAD PROJ_HID AUTO_BATCH < <(capacity_overrides "$STAGE1_CAPACITY")
    if [[ "$STAGE125_BATCH" == "auto" ]]; then
        STAGE125_BATCH="$AUTO_BATCH"
    fi
    KEEP_PATH="$(ensure_clip4096)"
    DATA_YAML="/tmp/${STAGE125_TAG}_data.yaml"
    MODEL_YAML="/tmp/${STAGE125_TAG}_model.yaml"
    EXP_YAML="/tmp/${STAGE125_TAG}_exp.yaml"
    TRAIN_YAML="/tmp/${STAGE125_TAG}_train.yaml"
    LR_VALUE=$(python - "configs/stage1/train.yaml" "$STAGE125_LR_MULT" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(float(cfg.get('train', {}).get('lr', 2.0e-4)) * float(sys.argv[2]))
PY
)
    make_yaml_override "$DATA_YAML" "configs/stage1/data.yaml" \
        "data.vocab_clip.keep_indices_path" "$KEEP_PATH" \
        "data.max_seq_len" "512" \
        "data.sampling.strategy" "random" \
        "data.sampling.keep_must_include" "true" \
        "data.sampling.alpha" "1.0"
    make_yaml_override "$MODEL_YAML" "configs/stage1/model_main.yaml" \
        "model.transcriptomics.hidden_dim" "$TX_HID" \
        "model.transcriptomics.n_layers" "$TX_LAY" \
        "model.transcriptomics.top_hvg_gene.dim" "$TOK_DIM" \
        "model.transcriptomics.top_hvg_gene.n_layers" "$TOK_LAY" \
        "model.transcriptomics.top_hvg_gene.n_heads" "$TOK_HEAD" \
        "model.projector.hidden_dim" "$PROJ_HID"
    make_yaml_override "$TRAIN_YAML" "configs/stage1/train.yaml" \
        "train.batch_size" "$STAGE125_BATCH" \
        "train.lr" "$LR_VALUE" \
        "train.early_stopping.patience" "8"
    make_yaml_override "$EXP_YAML" "configs/stage1/experiment.yaml" \
        "experiment.name" "stage125_chunk_jepa" \
        "experiment.tokenizer.mask_ratio" "$STAGE125_MASK_RATIO" \
        "experiment.tx_self.masking_obj" "symbol" \
        "experiment.tx_self.symbol_weight" "0.0" \
        "experiment.tx_self.value_weight" "0.0" \
        "experiment.tx_self.enable_masked_jepa" "false" \
        "experiment.tx_self.jepa_weight" "0.0" \
        "experiment.tx_self.enable_view_jepa" "false" \
        "experiment.tx_self.view_jepa_weight" "0.0" \
        "experiment.tx_self.enable_dino_consistency" "false" \
        "experiment.tx_self.dino_weight" "0.0" \
        "experiment.tx_self.koleo_weight" "0.0" \
        "experiment.tx_self.enable_multi_chunk_jepa" "true" \
        "experiment.tx_self.multi_chunk_weight" "$STAGE125_MC_WEIGHT" \
        "experiment.tx_self.multi_chunk_n_chunks" "4" \
        "experiment.tx_self.multi_chunk_len" "256" \
        "experiment.tx_self.multi_chunk_target" "target_chunk" \
        "experiment.tx_self.multi_chunk_dynamic" "true" \
        "experiment.tx_self.multi_chunk_target_chunks" "$STAGE125_TARGET_CHUNKS" \
        "experiment.tx_self.multi_chunk_target_scale" "$STAGE125_TARGET_SCALE" \
        "experiment.tx_self.multi_chunk_context_scale" "$STAGE125_CONTEXT_SCALE" \
        "experiment.tx_self.multi_chunk_loss" "smooth_l1" \
        "experiment.tx_self.multi_chunk_target_id_scale" "$STAGE125_TARGET_ID_SCALE" \
        "experiment.tx_self.multi_chunk_regularizer" "$STAGE125_MC_REGULARIZER" \
        "experiment.tx_self.multi_chunk_koleo_weight" "$STAGE125_MC_KOLEO" \
        "experiment.tx_self.multi_chunk_vicreg_var_weight" "$STAGE125_VICREG_VAR_WEIGHT" \
        "experiment.tx_self.multi_chunk_vicreg_cov_weight" "$STAGE125_VICREG_COV_WEIGHT" \
        "experiment.tx_self.multi_chunk_vicreg_gamma" "$STAGE125_VICREG_GAMMA" \
        "experiment.tx_self.multi_chunk_warmup_epochs" "$STAGE125_WARMUP" \
        "experiment.tx_self.multi_chunk_ramp_epochs" "$STAGE125_RAMP" \
        "experiment.monitor.stage1_quick_every" "5" \
        "experiment.monitor.stage1_clean_msm_every" "5" \
        "experiment.monitor.gene_set_every" "10"

    mkdir -p "results/runs/${STAGE125_TAG}"
    python - "$STAGE125_TAG" "$STAGE1_CKPT" "$STAGE1_CAPACITY" "$STAGE125_BATCH" "$STAGE125_MC_WEIGHT" "$STAGE125_TARGET_CHUNKS" "$STAGE125_TARGET_SCALE" "$STAGE125_CONTEXT_SCALE" "$STAGE125_MC_REGULARIZER" "$STAGE125_VICREG_VAR_WEIGHT" "$STAGE125_VICREG_COV_WEIGHT" "$STAGE125_VICREG_GAMMA" > "results/runs/${STAGE125_TAG}/sequential_meta.json" <<'PY'
import json, sys
(tag, stage1_ckpt, cap, batch, mc_w, target_chunks, target_scale, context_scale, regularizer, vicreg_var, vicreg_cov, vicreg_gamma) = sys.argv[1:]
print(json.dumps({
    "stage": "stage1.25_chunk_jepa_refinement",
    "tag": tag,
    "init_tx_ckpt": stage1_ckpt,
    "capacity": cap,
    "batch_size_per_rank": int(batch),
    "objective": "multi_chunk_jepa_only",
    "multi_chunk_weight": float(mc_w),
    "multi_chunk_target_chunks": target_chunks,
    "multi_chunk_target_scale": [float(x) for x in target_scale.split(',')],
    "multi_chunk_context_scale": [float(x) for x in context_scale.split(',')],
    "multi_chunk_regularizer": regularizer,
    "multi_chunk_vicreg_var_weight": float(vicreg_var),
    "multi_chunk_vicreg_cov_weight": float(vicreg_cov),
    "multi_chunk_vicreg_gamma": float(vicreg_gamma),
}, indent=2, sort_keys=True))
PY

    echo "=========================================================="
    echo "[seq] Stage 1.25 chunk-JEPA refinement"
    echo "  init=$STAGE1_CKPT"
    echo "  tag=$STAGE125_TAG epochs=$STAGE125_EPOCHS batch/rank=$STAGE125_BATCH mc_weight=$STAGE125_MC_WEIGHT"
    echo "=========================================================="
    extra_args=()
    if [[ "${STAGE125_LIMIT_TRAIN:-0}" -gt 0 ]]; then
        extra_args+=(--limit-train "$STAGE125_LIMIT_TRAIN")
    fi
    accelerate launch --num_processes "$NUM_PROC_STAGE1" --mixed_precision bf16 \
        scripts/train.py \
            --experiment "$EXP_YAML" \
            --model "$MODEL_YAML" \
            --data "$DATA_YAML" \
            --train "$TRAIN_YAML" \
            --tag "$STAGE125_TAG" \
            --epochs "$STAGE125_EPOCHS" \
            --image-backbone feature \
            --stage1-only \
            --init-tx-ckpt "$STAGE1_CKPT" \
            "${extra_args[@]}"
fi

if [[ ! -f "$STAGE125_CKPT" ]]; then
    echo "[seq] ERROR: missing Stage1.25 chunk checkpoint: $STAGE125_CKPT" >&2
    exit 1
fi

STAGE15_CKPT="results/runs/${STAGE15_TAG}/ckpt_spatial_best.pt"
if [[ "$SKIP_STAGE15" == "0" ]]; then
    echo "=========================================================="
    echo "[seq] Stage 1.5 spatial-JEPA from chunk-refined tx encoder"
    echo "  stage1.25_ckpt=$STAGE125_CKPT"
    echo "  tag=$STAGE15_TAG epochs=$STAGE15_EPOCHS"
    echo "=========================================================="
    STAGE1_CKPT="$STAGE125_CKPT" \
    TAG="$STAGE15_TAG" \
    EPOCHS="$STAGE15_EPOCHS" \
    BATCH_SIZE="$STAGE15_BATCH" \
    SUBGRAPH_SIZE="$STAGE15_SUBGRAPH_SIZE" \
    TX_ENCODE_BATCH="$STAGE15_TX_ENCODE_BATCH" \
        bash scripts/train/stage15_main.sh
fi

if [[ "$SKIP_EVAL" == "0" ]]; then
    mkdir -p results/eval
    echo "── eval: Stage 1.25 tx checkpoint ──"
    python scripts/eval/stage1_tx.py \
        --prepared-dir "$PREP" \
        --split test \
        --ckpts "$STAGE1_CKPT" "$STAGE125_CKPT" \
        --val-samples 76 --pool-spots 8000 \
        --linear-probe-genes 256 \
        --out "results/eval/sequential_stage1_vs_stage125.csv"
    if [[ "$RUN_DLPFC" == "1" && -d "$DLPFC_DIR" ]]; then
        echo "── eval: DLPFC h_tx/chunk_state/spot_state comparison ──"
        python scripts/eval/dlpfc_eval.py \
            --dlpfc-dir "$DLPFC_DIR" \
            --ckpts "$STAGE1_CKPT" "$STAGE125_CKPT" \
            --representations h_tx,chunk_state,spot_state \
            --gene-map-representations spot_state \
            --viz-out-dir results/figures/sequential_dlpfc \
            --out results/eval/sequential_dlpfc_stage1_vs_stage125.csv \
            --per-sample-out results/eval/sequential_dlpfc_stage1_vs_stage125_per_sample.csv
    fi
    if [[ -f "$STAGE15_CKPT" ]]; then
        echo "── eval: Stage 1.5 spatial metrics ──"
        python scripts/eval/stage15_indist.py \
            --prepared-dir "$PREP" \
            --split test \
            --ckpts "$STAGE15_CKPT" \
            --max-samples 30 \
            --out "results/eval/sequential_stage15_indist.csv"
    fi
fi

echo "=========================================================="
echo "[seq] sequential cascade complete"
echo "  Stage 1 MSM ckpt      : $STAGE1_CKPT"
echo "  Stage 1.25 chunk ckpt : $STAGE125_CKPT"
echo "  Stage 1.5 spatial ckpt: $STAGE15_CKPT"
echo "=========================================================="
