#!/usr/bin/env bash
# Bash preset wrapper for the Python spot-encoder pipeline.
#
# This script runs either a single experiment or curated ablation grids while
# delegating the actual orchestration to scripts/run_spot_encoder_pipeline.py.
# Stage 1 uses multi-GPU accelerate through --num-proc-stage1.
# Stage 1.5 remains single-process by default unless NUM_PROC_STAGE15 is set.
#
# Usage:
#   bash scripts/ablation/run_spot_encoder_pipeline.sh single
#   bash scripts/ablation/run_spot_encoder_pipeline.sh stage1        # Stage1-only core objective ablation
#   bash scripts/ablation/run_spot_encoder_pipeline.sh stage1_all    # Stage1-only objective/vocab/capacity/value_aug
#   bash scripts/ablation/run_spot_encoder_pipeline.sh spatial --stage1-ckpt results/runs/.../ckpt_tx_encoder_best.pt
#   bash scripts/ablation/run_spot_encoder_pipeline.sh all
#
# Common env knobs:
#   PIPELINE             joint                         (single preset only; sequential is deprecated)
#   CAPACITY             spatula_lite|spatula_mid|spatula_large
#   STAGE1_CKPT          existing Stage1 tx ckpt
#   STAGE1_EPOCHS        default 50
#   STAGE15_EPOCHS       default 30
#   NUM_PROC_STAGE1      default 8
#   NUM_PROC_STAGE15     default 1
#   SKIP_EVAL            1 to skip post-stage eval
#   MAKE_VIZ             1 to write DLPFC/gene-map figures
#   TX_POOLING_MODE      ckpt|cls|token_mean|cls_token_mean_sum|cls_token_mean_avg
#   DLPFC_CLUSTER_METHODS kmeans,gmm,leiden,spatial_leiden
#   NEURAL_LINEAR_PROBE  1 to add 50-epoch early-stop neural linear probes
#   STAGE1_EVAL_BATCH    Stage1 eval micro-batch, default 16 for vocab8192/full safety
#   DRY_RUN              1 to print commands only
#   EXTRA_ARGS           extra args appended to every python call

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"

PRESET="${1:-single}"
shift || true

PIPELINE="${PIPELINE:-joint}"
CAPACITY="${CAPACITY:-spatula_mid}"
STAGE1_VOCAB="${STAGE1_VOCAB:-4096}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE15_EPOCHS="${STAGE15_EPOCHS:-30}"
NUM_PROC_STAGE1="${NUM_PROC_STAGE1:-8}"
NUM_PROC_STAGE15="${NUM_PROC_STAGE15:-1}"
MAKE_VIZ="${MAKE_VIZ:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
TX_POOLING_MODE="${TX_POOLING_MODE:-ckpt}"
DLPFC_CLUSTER_METHODS="${DLPFC_CLUSTER_METHODS:-kmeans,gmm,leiden,spatial_leiden}"
NEURAL_LINEAR_PROBE="${NEURAL_LINEAR_PROBE:-0}"
STAGE1_EVAL_BATCH="${STAGE1_EVAL_BATCH:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BASE_ARGS=(
    --capacity "$CAPACITY"
    --stage1-epochs "$STAGE1_EPOCHS"
    --stage1-vocab "$STAGE1_VOCAB"
    --stage1-encode-batch "$STAGE1_EVAL_BATCH"
    --stage1-gene-map-tx-batch "$STAGE1_EVAL_BATCH"
    --stage15-epochs "$STAGE15_EPOCHS"
    --num-proc-stage1 "$NUM_PROC_STAGE1"
    --num-proc-stage15 "$NUM_PROC_STAGE15"
    --stage1-tx-pooling-mode "$TX_POOLING_MODE"
    --dlpfc-cluster-methods "$DLPFC_CLUSTER_METHODS"
)

if [[ -n "${STAGE1_CKPT:-}" ]]; then
    BASE_ARGS+=(--stage1-ckpt "$STAGE1_CKPT")
fi
if [[ "$SKIP_EVAL" == "1" ]]; then
    BASE_ARGS+=(--skip-eval)
fi
if [[ "$MAKE_VIZ" == "1" ]]; then
    BASE_ARGS+=(--make-viz --stage15-gene-map)
fi
if [[ "$DRY_RUN" == "1" ]]; then
    BASE_ARGS+=(--dry-run)
fi
if [[ "$NEURAL_LINEAR_PROBE" == "1" ]]; then
    BASE_ARGS+=(--neural-linear-probe)
fi

run_pipeline() {
    echo "=========================================================="
    echo "[spot-pipeline] preset=$PRESET"
    echo "  args: ${BASE_ARGS[*]} $* ${EXTRA_ARGS}"
    echo "=========================================================="
    # shellcheck disable=SC2086
    python scripts/run_spot_encoder_pipeline.py "${BASE_ARGS[@]}" "$@" ${EXTRA_ARGS}
}

case "$PRESET" in
    smoke|smoke_a|smoke_b)
        # End-to-end sanity check with tiny data and 1 epoch per stage.
        # Verifies Stage1 MSM+chunk-JEPA -> Stage1.5 spatial-JEPA checkpoint
        # pass-through, eval CSV writing, and no-OOM.
        SMOKE_LIMIT_TRAIN="${SMOKE_LIMIT_TRAIN:-50}"
        SMOKE_LIMIT_SHARDS="${SMOKE_LIMIT_SHARDS:-4}"
        SMOKE_NUM_PROC1="${SMOKE_NUM_PROC1:-4}"

        _smoke_common_args=(
            --capacity spatula_lite
            --stage1-epochs 1
            --stage15-epochs 1
            --stage1-limit-train "$SMOKE_LIMIT_TRAIN"
            --stage15-limit-train-shards "$SMOKE_LIMIT_SHARDS"
            --stage15-limit-val-shards 2
            --stage15-batch 4
            --stage15-subgraph-size 64
            --stage15-tx-encode-batch 64
            --stage15-eval-max-samples 4
            --num-proc-stage1 "$SMOKE_NUM_PROC1"
            --num-proc-stage15 1
            --stage1-test-samples 8
            --stage1-pool-spots 200
            --stage1-linear-probe-genes 64
            --dlpfc-viz-samples 0
            --force-stage1
            --force-stage15
        )
        if [[ "$DRY_RUN" == "1" ]]; then
            _smoke_common_args+=(--dry-run)
        fi
        if [[ "$SKIP_EVAL" == "1" ]]; then
            _smoke_common_args+=(--skip-eval)
        fi
        if [[ "$MAKE_VIZ" == "1" ]]; then
            _smoke_common_args+=(--make-viz --stage15-gene-map)
        fi

        _run_smoke_joint() {
            echo "##########################################################"
            echo "[smoke] Stage1 MSM+multi_chunk_JEPA -> spatial_JEPA"
            echo "##########################################################"
            BASE_ARGS=("${_smoke_common_args[@]}"
                --pipeline joint
                --stage1-objective msm_multi_chunk
                --eval-prefix smoke_joint
                --stage1-tag smoke_joint_stage1
                --stage15-tag smoke_joint_stage15
            )
            run_pipeline "$@"
        }

        case "$PRESET" in
            smoke|smoke_b) _run_smoke_joint "$@" ;;
            smoke_a) echo "[smoke_a] deprecated sequential smoke removed; running joint smoke."; _run_smoke_joint "$@" ;;
        esac
        ;;

    single)
        run_pipeline --pipeline "$PIPELINE" --stage1-objective "${STAGE1_OBJECTIVE:-msm_multi_chunk}" "$@"
        ;;

    stage1|stage1_core)
        # Stage1-only core objective ablation. No spatial-JEPA training/eval.
        run_pipeline --grid stage1_core --skip-stage15 "$@"
        ;;

    stage1_all|stage1_only|stage1_ablation)
        # Stage1-only train+eval suite.  This is the recommended quick loop
        # before spending compute on Stage1.5 spatial-JEPA.
        # Order: primary objective -> vocab -> capacity -> value augmentation -> pooling readout eval.
        export STAGE1_MC_WARMUP="${STAGE1_MC_WARMUP:-5}"
        export STAGE1_MC_RAMP="${STAGE1_MC_RAMP:-8}"
        run_pipeline --grid joint_chunk --skip-stage15 "$@"
        run_pipeline --grid stage1_vocab --skip-stage15 "$@"
        run_pipeline --grid stage1_capacity --skip-stage15 "$@"
        run_pipeline --grid value_aug --pipeline joint --stage1-objective msm_multi_chunk --skip-stage15 "$@"

        primary_stage1_tag="stage1_pipe_msm_multi_chunk_v${STAGE1_VOCAB}_${CAPACITY}__stage1_objective-msm_multi_chunk"
        primary_stage1_ckpt="results/runs/${primary_stage1_tag}/ckpt_tx_encoder_best.pt"
        if [[ "$SKIP_EVAL" != "1" ]]; then
            echo "[stage1_all] pooling eval will reuse primary Stage1 ckpt: ${primary_stage1_ckpt}"
            run_pipeline --pipeline joint --stage1-objective msm_multi_chunk --stage1-ckpt "$primary_stage1_ckpt" --skip-stage15 --stage1-tx-pooling-mode cls --eval-prefix "${EVAL_PREFIX_POOL_CLS:-spot_encoder_pipeline_stage1only_pooling_cls}" "$@"
            run_pipeline --pipeline joint --stage1-objective msm_multi_chunk --stage1-ckpt "$primary_stage1_ckpt" --skip-stage15 --stage1-tx-pooling-mode cls_token_mean_sum --eval-prefix "${EVAL_PREFIX_POOL_CLS_MEAN:-spot_encoder_pipeline_stage1only_pooling_cls_mean}" "$@"
        fi
        ;;

    vocab|stage1_vocab)
        run_pipeline --grid stage1_vocab --skip-stage15 "$@"
        ;;

    capacity|stage1_capacity)
        run_pipeline --grid stage1_capacity --skip-stage15 "$@"
        ;;

    joint|joint_chunk|msm_chunk|msm_multi_chunk)
        # PRIMARY spot-encoder objective.  Config addresses two known failure
        # modes (see docs/design/2_stage1_objectives_kr.md, §EMA divergence):
        #
        # 1) Loss-budget balance.  MSM cross-entropy lives in [0, log(vocab)]
        #    ≈ [0, 8.3]; multi_chunk_jepa lives in [0, ~2].  Equal weights mean
        #    MSM is ~100× louder, so multi_chunk barely contributes gradients.
        #    Default normalises symbol_weight by 1/log(vocab) ≈ 0.12 so the
        #    two objectives share a similar loss-magnitude budget.
        # 2) EMA teacher-student divergence (Stage 1.25 saw this).  Default
        #    switches multi_chunk_loss to `cosine` (scale-invariant) which
        #    stays stable when the EMA teacher's norm drifts.
        # MSM saturates for STAGE1_MC_WARMUP epochs, then multi_chunk_JEPA
        # ramps in over STAGE1_MC_RAMP epochs to its full weight.
        export STAGE1_SYMBOL_WEIGHT="${STAGE1_SYMBOL_WEIGHT:-0.12}"
        export STAGE1_MC_LOSS="${STAGE1_MC_LOSS:-cosine}"
        export STAGE1_MC_WEIGHT="${STAGE1_MC_WEIGHT:-0.5}"
        export STAGE1_MC_KOLEO="${STAGE1_MC_KOLEO:-0.0}"
        export STAGE1_MC_REGULARIZER="${STAGE1_MC_REGULARIZER:-vicreg}"
        export STAGE1_MC_WARMUP="${STAGE1_MC_WARMUP:-5}"
        export STAGE1_MC_RAMP="${STAGE1_MC_RAMP:-8}"
        export STAGE1_MC_TARGET_CHUNKS="${STAGE1_MC_TARGET_CHUNKS:-auto}"
        echo "[joint] symbol_weight=${STAGE1_SYMBOL_WEIGHT}  mc_loss=${STAGE1_MC_LOSS}"
        echo "[joint] mc_weight=${STAGE1_MC_WEIGHT}  mc_regularizer=${STAGE1_MC_REGULARIZER}  target_chunks=${STAGE1_MC_TARGET_CHUNKS}"
        echo "[joint] MSM warmup=${STAGE1_MC_WARMUP}ep | multi_chunk ramp=${STAGE1_MC_RAMP}ep"
        run_pipeline --grid joint_chunk "$@"
        ;;

    value_aug|noise|augmentation)
        # Ablate masked-position value augmentation under the joint objective.
        # Three profiles:
        #   keep_only   — pure MASK, no value perturbation.  Best test of
        #                 whether the model needs ANY corruption to avoid the
        #                 value→symbol shortcut.
        #   mild        — paper-aligned (Sinha et al. 2021 EMNLP) default:
        #                 85% keep, 10% mild noise (std=0.15), 5% drop.
        #                 Preserves gene-value co-occurrence which is the
        #                 actual signal MLMs learn.
        #   aggressive  — pre-patch legacy 75/15/10, noise_std=0.35.  Mostly
        #                 a control to confirm the paper's claim that strong
        #                 corruption hurts learning.
        run_pipeline --grid value_aug --pipeline joint --stage1-objective msm_multi_chunk --skip-stage15 "$@"
        ;;

    eval_cls_mean|pooling_cls_mean|cls_mean_eval)
        # Evaluate an existing Stage1 checkpoint with Prov-GigaPath-style
        # spot readout: CLS + valid gene-token mean. No retraining.
        # Example:
        #   STAGE1_CKPT=results/runs/.../ckpt_tx_encoder_best.pt \
        #   bash scripts/ablation/run_spot_encoder_pipeline.sh eval_cls_mean
        if [[ -z "${STAGE1_CKPT:-}" ]]; then
            echo "eval_cls_mean requires STAGE1_CKPT=/path/to/ckpt_tx_encoder_best.pt" >&2
            exit 1
        fi
        TX_POOLING_MODE="cls_token_mean_sum"
        BASE_ARGS+=(--stage1-tx-pooling-mode "$TX_POOLING_MODE")
        run_pipeline \
            --pipeline joint \
            --stage1-objective msm_multi_chunk \
            --stage1-ckpt "$STAGE1_CKPT" \
            --skip-stage15 \
            --eval-prefix "${EVAL_PREFIX:-spot_encoder_pipeline_cls_token_mean_sum}" \
            "$@"
        ;;

    eval_pooling_pair|pooling_pair)
        # Evaluate the same ckpt twice: original readout and CLS+gene-token mean.
        if [[ -z "${STAGE1_CKPT:-}" ]]; then
            echo "eval_pooling_pair requires STAGE1_CKPT=/path/to/ckpt_tx_encoder_best.pt" >&2
            exit 1
        fi
        TX_POOLING_MODE="cls"
        BASE_ARGS+=(--stage1-tx-pooling-mode "$TX_POOLING_MODE")
        run_pipeline --pipeline joint --stage1-objective msm_multi_chunk --stage1-ckpt "$STAGE1_CKPT" --skip-stage15 --eval-prefix "${EVAL_PREFIX:-spot_encoder_pipeline_pooling_cls}" "$@"
        TX_POOLING_MODE="cls_token_mean_sum"
        BASE_ARGS=("${BASE_ARGS[@]/--stage1-tx-pooling-mode/--stage1-tx-pooling-mode}")
        run_pipeline --pipeline joint --stage1-objective msm_multi_chunk --stage1-ckpt "$STAGE1_CKPT" --skip-stage15 --stage1-tx-pooling-mode "$TX_POOLING_MODE" --eval-prefix "${EVAL_PREFIX_CLS_MEAN:-spot_encoder_pipeline_pooling_cls_mean}" "$@"
        ;;

    spatial|stage15)
        # Spatial direction/aggregation ablation. Usually provide STAGE1_CKPT.
        run_pipeline --grid stage15_spatial "$@"
        ;;

    all)
        # Practical full set. Deliberate order after smoke tests:
        #   1) joint MSM + multi_chunk_JEPA (primary objective)
        #   2) vocab controls
        #   3) capacity controls
        #   4) spatial-JEPA direction/aggregation using the primary Stage1 ckpt
        # Stage1.25 is no longer part of `all`; chunk-JEPA lives inside Stage1.
        export STAGE1_MC_WARMUP="${STAGE1_MC_WARMUP:-5}"
        export STAGE1_MC_RAMP="${STAGE1_MC_RAMP:-8}"
        run_pipeline --grid joint_chunk "$@"
        run_pipeline --grid stage1_vocab "$@"
        run_pipeline --grid stage1_capacity "$@"

        primary_stage1_tag="stage1_pipe_msm_multi_chunk_v${STAGE1_VOCAB}_${CAPACITY}__stage1_objective-msm_multi_chunk"
        primary_stage1_ckpt="results/runs/${primary_stage1_tag}/ckpt_tx_encoder_best.pt"
        echo "[all] spatial ablation will reuse primary Stage1 ckpt: ${primary_stage1_ckpt}"
        run_pipeline --grid stage15_spatial --stage1-ckpt "$primary_stage1_ckpt" "$@"

        if [[ "$SKIP_EVAL" != "1" ]]; then
            echo "[all] pooling eval will reuse primary Stage1 ckpt: ${primary_stage1_ckpt}"
            echo "[all] pooling eval 1/2: CLS-only readout"
            run_pipeline \
                --pipeline joint \
                --stage1-objective msm_multi_chunk \
                --stage1-ckpt "$primary_stage1_ckpt" \
                --skip-stage15 \
                --stage1-tx-pooling-mode cls \
                --eval-prefix "${EVAL_PREFIX_POOL_CLS:-spot_encoder_pipeline_pooling_cls}" \
                "$@"
            echo "[all] pooling eval 2/2: CLS + gene-token mean readout"
            run_pipeline \
                --pipeline joint \
                --stage1-objective msm_multi_chunk \
                --stage1-ckpt "$primary_stage1_ckpt" \
                --skip-stage15 \
                --stage1-tx-pooling-mode cls_token_mean_sum \
                --eval-prefix "${EVAL_PREFIX_POOL_CLS_MEAN:-spot_encoder_pipeline_pooling_cls_mean}" \
                "$@"
        fi
        ;;

    custom)
        # Use EXTRA_ARGS or trailing args for arbitrary --sweep combinations.
        # Example:
        #   EXTRA_ARGS='--grid stage15_spatial --stage1-ckpt results/runs/.../ckpt_tx_encoder_best.pt' \
        #     bash scripts/ablation/run_spot_encoder_pipeline.sh custom
        run_pipeline "$@"
        ;;

    *)
        echo "Unknown preset: $PRESET" >&2
        echo "Use one of: smoke single stage1 stage1_all joint joint_chunk vocab capacity value_aug eval_cls_mean eval_pooling_pair spatial all custom" >&2
        exit 1
        ;;
esac
