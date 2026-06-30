#!/usr/bin/env bash
# Stage 1 → Stage 1.5 main-candidate CASCADE.
#
# Trains the project's working hypothesis end-to-end:
#
#     Stage 1 (RNA foundation)        global_median normalization
#                                     vocab_clip 4096
#                                     value_aug = balanced (mixed)
#                                     MSM (masked-symbol modelling)  ← primary
#                                     DINO OFF by default            (Stage1 main)
#                                     Optional DINO late warmup      (no KoLeo)
#                                     Gene-JEPA OFF
#                                     mask_ratio 0.15, max_seq_len auto→512
#
#     Stage 1.5 (Spatial foundation)  separate region+spot tokens
#                                     mask_target = spot             (region predicts spot)
#                                     subgraph_kind = ego            (sample-level KNN)
#                                     region.include_anchor = false  (JEPA contract)
#                                     uses the Stage-1 ckpt produced above
#
# After both stages finish we run BOTH evaluation suites on the test split:
#     scripts/eval/stage1_tx.py         → stage1 holdout metrics
#     scripts/eval/dlpfc_eval.py        → DLPFC layer recovery
#     scripts/eval/stage15_indist.py    → Stage 1.5 spatial in-dist
#
# Usage:
#   bash scripts/run_all.sh                      # full cascade, default tags
#   STAGE1_EPOCHS=20 STAGE15_EPOCHS=15 \
#       bash scripts/run_all.sh                  # shorter runs
#   STAGE1_TAG=foo STAGE15_TAG=bar \
#       bash scripts/run_all.sh
#   SKIP_STAGE1=1 STAGE1_CKPT=results/runs/.../ckpt_tx_encoder_best.pt \
#       bash scripts/run_all.sh                  # reuse existing stage1 ckpt
#   SKIP_EVAL=1 bash scripts/run_all.sh          # train only, no eval
#
# Env knobs:
#   STAGE1_TAG, STAGE15_TAG          run dirs under results/runs/
#   STAGE1_EPOCHS, STAGE15_EPOCHS    epoch caps (default 50 / 30)
#   STAGE1_CKPT                      override stage1 ckpt path (skips training)
#   SKIP_STAGE1, SKIP_STAGE15        skip the respective training step
#   SKIP_EVAL                        skip the post-train evaluation block
#   STAGE1_OBJECTIVE                 msm_only | msm_multi_chunk | view_jepa_w005 | view_jepa_w010 | dino_late_no_koleo
#   STAGE1_CAPACITY                  spatula_lite | spatula_mid | spatula_large
#                                    (keeps h_tx=512; scales token transformer capacity)
#   STAGE1_MAX_SEQ_LEN               auto | explicit cap (default auto; clip4096→512)
#   STAGE1_BATCH                     per-rank Stage 1 batch; auto if unset
#   NUM_PROC_STAGE1                  accelerate ranks for Stage 1 (default 8)
#   BATCH_SIZE_STAGE15               Stage 1.5 batch size (default 8)
#   SUBGRAPH_SIZE                    Stage 1.5 subgraph size (default 256)

set -euo pipefail
# Use a distinctive variable name — `_common.sh` (sourced below) defines
# its own `ROOT` and `cd`s to it, which lands at /workspace if we leave
# the variable named `ROOT`.  Capture our root under MM_ROOT so the
# downstream paths in this wrapper stay anchored to /workspace/mm_align.
MM_ROOT="$(cd "$(dirname "$0")/.."; pwd)"
cd "$MM_ROOT"
export PYTHONPATH="${MM_ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# ── defaults ──────────────────────────────────────────────────────────────
STAGE1_OBJECTIVE="${STAGE1_OBJECTIVE:-msm_only}"
STAGE1_CAPACITY="${STAGE1_CAPACITY:-spatula_lite}"
STAGE1_TAG="${STAGE1_TAG:-stage1_main_${STAGE1_OBJECTIVE}_${STAGE1_CAPACITY}}"
STAGE15_TAG="${STAGE15_TAG:-stage15_main_separate_spot_ego}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE15_EPOCHS="${STAGE15_EPOCHS:-30}"
NUM_PROC_STAGE1="${NUM_PROC_STAGE1:-8}"
BATCH_SIZE_STAGE15="${BATCH_SIZE_STAGE15:-8}"
SUBGRAPH_SIZE="${SUBGRAPH_SIZE:-256}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
STAGE1_RESUME_CKPT="${STAGE1_RESUME_CKPT:-}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE15="${SKIP_STAGE15:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"   # <=0 means full validation
STAGE1_LIMIT_TRAIN="${STAGE1_LIMIT_TRAIN:-0}"   # <=0 means full train pool (smoke knob)
STAGE1_BATCH_USER_SET="${STAGE1_BATCH+x}"
STAGE1_BATCH="${STAGE1_BATCH:-auto}"
STAGE1_MAX_SEQ_LEN="${STAGE1_MAX_SEQ_LEN:-auto}"
STAGE1_VOCAB="${STAGE1_VOCAB:-4096}"  # 4096 | 8192 | full

# We reuse the ablation YAML-override helper.  NOTE: _common.sh sources its
# own ROOT via `$0` which, when sourced, points to OUR script — and its
# `cd "$MM_ROOT"` lands one level too high (`/workspace` instead of
# `/workspace/mm_align`).  Force cwd back to our ROOT after sourcing.
source "${MM_ROOT}/scripts/ablation/_common.sh" 1>/dev/null
cd "$MM_ROOT"

PREP="results/cache/prepared_expanded"

# ── STAGE 1 ───────────────────────────────────────────────────────────────
# Override the experiment_main + model_main + data YAMLs to encode the
# Stage-1 objective choice. vocab_clip = 4096 is the project's working
# choice (see scripts/ablation/run_all.sh, the rel4096 hypothesis).
ensure_clip() {
    local n="$1"
    local f="$PREP/clip${n}_keep_indices.npy"
    if [[ ! -f "$f" ]]; then
        echo "[run_all] building clip${n} vocab indices..."
        python scripts/data/make_clipped_vocab.py --top-k "$n"
    fi
    realpath "$f"
}

seq_cap_for_vocab() {
    case "$1" in
        4096|8192|full) echo "512" ;;
        *) echo "512" ;;
    esac
}

train_stage1() {
    local tag="$1"
    local data_yaml="/tmp/${tag}_data.yaml"
    local exp_yaml="/tmp/${tag}_exp.yaml"
    local model_yaml="/tmp/${tag}_model.yaml"
    local train_yaml="configs/stage1/train.yaml"

    local stage1_seq_cap="$STAGE1_MAX_SEQ_LEN"
    if [[ "$stage1_seq_cap" == "auto" ]]; then
        stage1_seq_cap="$(seq_cap_for_vocab "$STAGE1_VOCAB")"
    fi
    local data_overrides=(
        "data.max_seq_len" "$stage1_seq_cap"
        "data.sampling.strategy" "random"
        "data.sampling.keep_must_include" "true"
        "data.sampling.alpha" "1.0"
    )
    if [[ "$STAGE1_VOCAB" == "full" ]]; then
        data_overrides+=("data.vocab_clip" "null")
    else
        data_overrides+=("data.vocab_clip.keep_indices_path" "$(ensure_clip "$STAGE1_VOCAB")")
    fi
    # data overrides — global_median (default) + vocab selection + random sampling.
    make_yaml_override "$data_yaml" "configs/stage1/data.yaml" "${data_overrides[@]}"
    # model overrides — keep output h_tx=512 for Stage1.5 compatibility, but
    # scale the internal token transformer to test the SPATULA legacy
    # hypothesis that higher capacity improves MSM without changing the task.
    local tx_hidden="1024"
    local tx_layers="2"
    local tok_dim="256"
    local tok_layers="4"
    local tok_heads="4"
    local proj_hidden="1024"
    local auto_batch="512"
    case "$STAGE1_CAPACITY" in
        spatula_lite)
            STAGE1_CAPACITY="spatula_lite"; auto_batch="512"
            ;;
        spatula_mid)
            STAGE1_CAPACITY="spatula_mid"
            tx_hidden="1536"; tx_layers="2"; tok_dim="384"; tok_layers="6"; tok_heads="6"; proj_hidden="1536"; auto_batch="384"
            ;;
        spatula_large)
            STAGE1_CAPACITY="spatula_large"
            tx_hidden="2048"; tx_layers="2"; tok_dim="512"; tok_layers="6"; tok_heads="8"; proj_hidden="2048"; auto_batch="256"
            ;;
        *)
            echo "[run_all] unknown STAGE1_CAPACITY=$STAGE1_CAPACITY (use: spatula_lite | spatula_mid | spatula_large)"
            exit 1
            ;;
    esac
    local stage1_batch="$STAGE1_BATCH"
    if [[ "$stage1_batch" == "auto" || -z "$STAGE1_BATCH_USER_SET" ]]; then
        stage1_batch="$auto_batch"
    fi
    # Value-augmentation knobs (env-overridable).  Default = paper-aligned
    # (Sinha et al. 2021 EMNLP):  MLMs succeed via gene-value co-occurrence,
    # so we minimise NOISE (which warps the value distribution) and use
    # token-level DROPOUT (value=0, symbol kept) sparingly to break the
    # value→symbol shortcut at masked positions.  Context (unmasked) is
    # untouched by default.
    #   STAGE1_MASKED_KEEP_P / NOISE_P / DROP_P / NOISE_STD   masked targets
    #   STAGE1_UNMASKED_KEEP_P / NOISE_P / DROP_P / NOISE_STD context
    local m_keep="${STAGE1_MASKED_KEEP_P:-0.85}"
    local m_noise="${STAGE1_MASKED_NOISE_P:-0.0}"
    local m_drop="${STAGE1_MASKED_DROP_P:-0.15}"
    local m_std="${STAGE1_MASKED_NOISE_STD:-0.0}"
    local u_keep="${STAGE1_UNMASKED_KEEP_P:-1.0}"
    local u_noise="${STAGE1_UNMASKED_NOISE_P:-0.0}"
    local u_drop="${STAGE1_UNMASKED_DROP_P:-0.0}"
    local u_std="${STAGE1_UNMASKED_NOISE_STD:-0.0}"
    local u_mode="${STAGE1_UNMASKED_AUG_MODE:-keep}"
    local m_mode="${STAGE1_MASKED_AUG_MODE:-mixed}"
    make_yaml_override "$model_yaml" "configs/stage1/model_main.yaml" \
        "model.transcriptomics.hidden_dim" "$tx_hidden" \
        "model.transcriptomics.n_layers" "$tx_layers" \
        "model.transcriptomics.top_hvg_gene.dim" "$tok_dim" \
        "model.transcriptomics.top_hvg_gene.n_layers" "$tok_layers" \
        "model.transcriptomics.top_hvg_gene.n_heads" "$tok_heads" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.mode" "$m_mode" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.keep_p" "$m_keep" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.noise_p" "$m_noise" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.drop_p" "$m_drop" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.noise_std" "$m_std" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.mode" "$u_mode" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.keep_p" "$u_keep" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.noise_p" "$u_noise" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.drop_p" "$u_drop" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.noise_std" "$u_std" \
        "model.projector.hidden_dim" "$proj_hidden"

    # experiment overrides — Stage1 main is MSM-only.  DINO is kept as an
    # explicit ablation because it can disturb expression-neighbor geometry.
    local dino_on="false"
    local dino_w="0.0"
    local dino_warm="0"
    local dino_ramp="0"
    local koleo_w="0.0"
    local koleo_warm="0"
    local koleo_ramp="0"
    local mc_loss="smooth_l1"               # overridden below for joint objective
    # MSM symbol_weight default: 1.0 (raw CE).  Joint objective normalises it
    # by 1/log(vocab) ≈ 0.12 so MSM and multi_chunk_jepa share the same loss
    # budget after the multi_chunk ramp completes (~ 4:1 instead of 100:1).
    local sym_w="${STAGE1_SYMBOL_WEIGHT:-1.0}"
    local variant="$STAGE1_OBJECTIVE"
    case "$STAGE1_OBJECTIVE" in
        msm_only)
            ;;
        msm_multi_chunk)
            ;;
        view_jepa_w005)
            ;;
        view_jepa_w010)
            ;;
        dino_late_no_koleo)
            dino_on="true"
            dino_w="0.02"
            dino_warm="10"
            dino_ramp="10"
            ;;
        *)
            echo "[run_all] unknown STAGE1_OBJECTIVE=$STAGE1_OBJECTIVE (use: msm_only | msm_multi_chunk | view_jepa_w005 | view_jepa_w010 | dino_late_no_koleo)"
            exit 1
            ;;
    esac
    local view_on="false"
    local view_w="0.0"
    local view_warm="0"
    local view_ramp="0"
    local mc_on="false"
    local mc_w="0.0"
    local mc_chunks="4"
    local mc_len="256"
    local mc_target_id_scale="0.25"
    local mc_koleo="0.0"
    local mc_regularizer="none"
    local mc_vicreg_var="0.05"
    local mc_vicreg_cov="0.01"
    local mc_vicreg_gamma="1.0"
    local mc_warm="0"
    local mc_ramp="0"
    local mc_target_chunks="auto"
    local mc_target_scale="0.15,0.25"
    local mc_context_scale="0.45,0.65"
    if [[ "$STAGE1_OBJECTIVE" == "msm_multi_chunk" ]]; then
        # Normalise MSM CE to ce_norm scale by default (1/log(4100) ≈ 0.12)
        # so MSM and multi_chunk_jepa share the same loss-magnitude budget.
        # Override with STAGE1_SYMBOL_WEIGHT to revert to raw-CE behaviour.
        if [[ "${STAGE1_SYMBOL_WEIGHT:-}" == "" ]]; then
            # vocab_size = special tokens (4) + n_hvg.  For default 4096 vocab
            # the effective symbol vocabulary is 4100; log ≈ 8.32.
            sym_w="0.12"
        fi
        # Joint MSM + multi_chunk_JEPA — primary spot-encoder objective.
        #
        # Default config addresses the two failure modes we observed:
        #   1) Stage 1.25-style EMA divergence: smooth_l1 loss accumulates the
        #      teacher-student magnitude drift.  Cosine loss is scale-invariant
        #      and stabilises the EMA self-distillation.
        #   2) MSM signal dominates multi_chunk_jepa when both are on:
        #      MSM cross-entropy lives in [0, log(vocab)] ≈ [0, 8.3], whereas
        #      cosine/smooth_l1 lives in [0, ~2].  At equal weights MSM is
        #      ~100× louder, so multi_chunk barely contributes to gradients.
        #      We normalise MSM weight by 1/log(vocab) ≈ 0.12 so the two
        #      objectives land in the same loss-magnitude budget after MSM
        #      saturates.  multi_chunk_weight is then bumped to 0.5 so it has
        #      a meaningful share once the warmup ramp completes.
        #
        # MSM is active from epoch 0 (mask_ratio=0.15).  multi_chunk_jepa is
        # gated 0 -> full over STAGE1_MC_WARMUP + STAGE1_MC_RAMP epochs.  MSM
        # has time to saturate first, giving the EMA teacher a stable target.
        mc_on="true"
        mc_w="${STAGE1_MC_WEIGHT:-0.5}"
        mc_chunks="${STAGE1_MC_N_CHUNKS:-4}"
        mc_len="${STAGE1_MC_LEN:-256}"
        mc_target_id_scale="${STAGE1_MC_TARGET_ID_SCALE:-0.25}"
        mc_koleo="${STAGE1_MC_KOLEO:-0.0}"
        mc_regularizer="${STAGE1_MC_REGULARIZER:-vicreg}"
        mc_vicreg_var="${STAGE1_MC_VICREG_VAR_WEIGHT:-0.05}"
        mc_vicreg_cov="${STAGE1_MC_VICREG_COV_WEIGHT:-0.01}"
        mc_vicreg_gamma="${STAGE1_MC_VICREG_GAMMA:-1.0}"
        mc_warm="${STAGE1_MC_WARMUP:-5}"
        mc_ramp="${STAGE1_MC_RAMP:-8}"
        mc_target_chunks="${STAGE1_MC_TARGET_CHUNKS:-auto}"
        mc_target_scale="${STAGE1_MC_TARGET_SCALE:-0.15,0.25}"
        mc_context_scale="${STAGE1_MC_CONTEXT_SCALE:-0.45,0.65}"
        mc_loss="${STAGE1_MC_LOSS:-cosine}"
    elif [[ "$STAGE1_OBJECTIVE" == "view_jepa_w005" ]]; then
        view_on="true"; view_w="0.05"; view_warm="10"; view_ramp="10"
    elif [[ "$STAGE1_OBJECTIVE" == "view_jepa_w010" ]]; then
        view_on="true"; view_w="0.10"; view_warm="10"; view_ramp="10"
    fi
    make_yaml_override "$exp_yaml" "configs/stage1/experiment_main.yaml" \
        "experiment.tokenizer.mask_ratio" "0.15" \
        "experiment.tx_self.symbol_weight" "$sym_w" \
        "experiment.tx_self.enable_view_jepa" "$view_on" \
        "experiment.tx_self.view_jepa_weight" "$view_w" \
        "experiment.tx_self.view_jepa_loss" "smooth_l1" \
        "experiment.tx_self.view_jepa_warmup_epochs" "$view_warm" \
        "experiment.tx_self.view_jepa_ramp_epochs" "$view_ramp" \
        "experiment.tx_self.enable_multi_chunk_jepa" "$mc_on" \
        "experiment.tx_self.multi_chunk_weight" "$mc_w" \
        "experiment.tx_self.multi_chunk_n_chunks" "$mc_chunks" \
        "experiment.tx_self.multi_chunk_len" "$mc_len" \
        "experiment.tx_self.multi_chunk_target" "target_chunk" \
        "experiment.tx_self.multi_chunk_dynamic" "true" \
        "experiment.tx_self.multi_chunk_target_chunks" "$mc_target_chunks" \
        "experiment.tx_self.multi_chunk_target_scale" "$mc_target_scale" \
        "experiment.tx_self.multi_chunk_context_scale" "$mc_context_scale" \
        "experiment.tx_self.multi_chunk_loss" "$mc_loss" \
        "experiment.tx_self.multi_chunk_target_id_scale" "$mc_target_id_scale" \
        "experiment.tx_self.multi_chunk_regularizer" "$mc_regularizer" \
        "experiment.tx_self.multi_chunk_koleo_weight" "$mc_koleo" \
        "experiment.tx_self.multi_chunk_vicreg_var_weight" "$mc_vicreg_var" \
        "experiment.tx_self.multi_chunk_vicreg_cov_weight" "$mc_vicreg_cov" \
        "experiment.tx_self.multi_chunk_vicreg_gamma" "$mc_vicreg_gamma" \
        "experiment.tx_self.multi_chunk_warmup_epochs" "$mc_warm" \
        "experiment.tx_self.multi_chunk_ramp_epochs" "$mc_ramp" \
        "experiment.tx_self.enable_dino_consistency" "$dino_on" \
        "experiment.tx_self.dino_weight" "$dino_w" \
        "experiment.tx_self.dino_loss" "cosine" \
        "experiment.tx_self.dino_warmup_epochs" "$dino_warm" \
        "experiment.tx_self.dino_ramp_epochs" "$dino_ramp" \
        "experiment.tx_self.koleo_weight" "$koleo_w" \
        "experiment.tx_self.koleo_warmup_epochs" "$koleo_warm" \
        "experiment.tx_self.koleo_ramp_epochs" "$koleo_ramp" \
        "experiment.tx_self.enable_masked_jepa" "false" \
        "experiment.tx_self.jepa_weight" "0.0"

    # ablation_meta — same schema as ablation runs use.
    local run_dir="results/runs/${tag}"
    mkdir -p "$run_dir"
    local changed_json
    changed_json=$(python - "$STAGE1_OBJECTIVE" "$STAGE1_CAPACITY" "$stage1_batch" "$stage1_seq_cap" "$tx_hidden" "$tok_dim" "$tok_layers" "$tok_heads" "$view_on" "$view_w" "$view_warm" "$view_ramp" "$mc_on" "$mc_w" "$mc_chunks" "$mc_len" "$mc_target_id_scale" "$mc_koleo" "$mc_regularizer" "$mc_vicreg_var" "$mc_vicreg_cov" "$mc_vicreg_gamma" "$mc_warm" "$mc_ramp" "$mc_target_chunks" "$mc_target_scale" "$mc_context_scale" "$dino_on" "$dino_w" "$dino_warm" "$dino_ramp" "$koleo_w" <<'PY'
import json, sys
(objective, capacity, batch, seq_cap, tx_hidden, tok_dim, tok_layers, tok_heads, view_on, view_w, view_warm, view_ramp, mc_on, mc_w, mc_chunks, mc_len, mc_target_id_scale, mc_koleo, mc_regularizer, mc_vicreg_var, mc_vicreg_cov, mc_vicreg_gamma, mc_warm, mc_ramp, mc_target_chunks, mc_target_scale, mc_context_scale, dino_on, dino_w, dino_warm, dino_ramp, koleo_w) = sys.argv[1:]
print(json.dumps({
    "data.gene_norm.mode": "global_median",
    "data.vocab_clip": "clip" + str(__import__("os").environ.get("STAGE1_VOCAB", "4096")),
    "data.max_seq_len": int(seq_cap),
    "experiment.tokenizer.mask_ratio": 0.15,
    "experiment.tx_self.objective_profile": objective,
    "model.capacity_profile": capacity,
    "train.batch_size_per_rank": int(batch),
    "model.transcriptomics.hidden_dim": int(tx_hidden),
    "model.transcriptomics.top_hvg_gene.dim": int(tok_dim),
    "model.transcriptomics.top_hvg_gene.n_layers": int(tok_layers),
    "model.transcriptomics.top_hvg_gene.n_heads": int(tok_heads),
    "experiment.tx_self.enable_view_jepa": view_on == "true",
    "experiment.tx_self.view_jepa_weight": float(view_w),
    "experiment.tx_self.view_jepa_loss": "smooth_l1",
    "experiment.tx_self.view_jepa_warmup_epochs": int(view_warm),
    "experiment.tx_self.view_jepa_ramp_epochs": int(view_ramp),
    "experiment.tx_self.enable_multi_chunk_jepa": mc_on == "true",
    "experiment.tx_self.multi_chunk_weight": float(mc_w),
    "experiment.tx_self.multi_chunk_n_chunks": int(mc_chunks),
    "experiment.tx_self.multi_chunk_len": int(mc_len),
    "experiment.tx_self.multi_chunk_target": "target_chunk",
    "experiment.tx_self.multi_chunk_dynamic": True,
    "experiment.tx_self.multi_chunk_target_chunks": mc_target_chunks,
    "experiment.tx_self.multi_chunk_target_scale": [float(x) for x in mc_target_scale.split(",")],
    "experiment.tx_self.multi_chunk_context_scale": [float(x) for x in mc_context_scale.split(",")],
    "experiment.tx_self.multi_chunk_target_id_scale": float(mc_target_id_scale),
    "experiment.tx_self.multi_chunk_regularizer": mc_regularizer,
    "experiment.tx_self.multi_chunk_koleo_weight": float(mc_koleo),
    "experiment.tx_self.multi_chunk_vicreg_var_weight": float(mc_vicreg_var),
    "experiment.tx_self.multi_chunk_vicreg_cov_weight": float(mc_vicreg_cov),
    "experiment.tx_self.multi_chunk_vicreg_gamma": float(mc_vicreg_gamma),
    "experiment.tx_self.multi_chunk_warmup_epochs": int(mc_warm),
    "experiment.tx_self.multi_chunk_ramp_epochs": int(mc_ramp),
    "experiment.tx_self.enable_dino_consistency": dino_on == "true",
    "experiment.tx_self.dino_weight": float(dino_w),
    "experiment.tx_self.dino_loss": "cosine",
    "experiment.tx_self.dino_warmup_epochs": int(dino_warm),
    "experiment.tx_self.dino_ramp_epochs": int(dino_ramp),
    "experiment.tx_self.koleo_weight": float(koleo_w),
    "experiment.tx_self.enable_masked_jepa": False,
    "model.transcriptomics.top_hvg_gene.value_aug.mode": "mixed",
}, sort_keys=True))
PY
)
    ABL_GROUP="main" ABL_VARIANT="$variant" ABL_CHANGED_JSON="$changed_json" \
    ABL_PROFILE="cascade" \
    write_ablation_meta "$tag" "$data_yaml" "$model_yaml" "$train_yaml" "$exp_yaml"

    echo "=========================================================="
    echo "[run_all] STAGE 1  tag=$tag  objective=$STAGE1_OBJECTIVE  capacity=$STAGE1_CAPACITY  vocab=$STAGE1_VOCAB  batch/rank=$stage1_batch  epochs=$STAGE1_EPOCHS  ranks=$NUM_PROC_STAGE1"
    echo "  data:  $data_yaml"
    echo "  model: $model_yaml"
    echo "  exp:   $exp_yaml"
    echo "=========================================================="
    local extra_args=()
    if [[ "$STAGE1_LIMIT_TRAIN" -gt 0 ]]; then
        extra_args+=(--limit-train "$STAGE1_LIMIT_TRAIN")
    fi
    if [[ -n "$STAGE1_RESUME_CKPT" ]]; then
        extra_args+=(--resume-ckpt "$STAGE1_RESUME_CKPT")
    fi
    accelerate launch --num_processes "$NUM_PROC_STAGE1" --mixed_precision bf16 \
        scripts/train.py \
            --experiment "$exp_yaml" \
            --model      "$model_yaml" \
            --data       "$data_yaml" \
            --train      "$train_yaml" \
            --tag        "$tag" \
            --epochs     "$STAGE1_EPOCHS" \
            --val-max-batches "$VAL_MAX_BATCHES" \
            --image-backbone feature \
            --stage1-only \
            "${extra_args[@]}"
}

if [[ "$SKIP_STAGE1" == "0" && -z "$STAGE1_CKPT" ]]; then
    train_stage1 "$STAGE1_TAG"
    STAGE1_CKPT="results/runs/${STAGE1_TAG}/ckpt_tx_encoder_best.pt"
fi
if [[ -z "$STAGE1_CKPT" ]]; then
    STAGE1_CKPT="results/runs/${STAGE1_TAG}/ckpt_tx_encoder_best.pt"
fi
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "[run_all] ERROR: Stage-1 ckpt not found at $STAGE1_CKPT"
    exit 1
fi
echo "[run_all] Stage 1 ckpt: $STAGE1_CKPT"

# ── STAGE 1.5 ─────────────────────────────────────────────────────────────
# Delegate to scripts/train/stage15_main.sh (already encodes
# separate + spot mask + ego subgraph).  Pass our Stage-1 ckpt.
if [[ "$SKIP_STAGE15" == "0" ]]; then
    echo "=========================================================="
    echo "[run_all] STAGE 1.5  tag=$STAGE15_TAG  epochs=$STAGE15_EPOCHS"
    echo "  stage1_ckpt: $STAGE1_CKPT"
    echo "  batch_size: $BATCH_SIZE_STAGE15  subgraph_size: $SUBGRAPH_SIZE"
    echo "=========================================================="
    STAGE1_CKPT="$STAGE1_CKPT" \
    TAG="$STAGE15_TAG" \
    EPOCHS="$STAGE15_EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE_STAGE15" \
    SUBGRAPH_SIZE="$SUBGRAPH_SIZE" \
        bash scripts/train/stage15_main.sh
fi

STAGE15_CKPT="results/runs/${STAGE15_TAG}/ckpt_spatial_best.pt"
echo "[run_all] Stage 1.5 ckpt: $STAGE15_CKPT"

# ── POST-TRAIN EVAL ───────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "[run_all] SKIP_EVAL=1 — done."
    exit 0
fi

mkdir -p results/eval

# 1) Stage 1 holdout eval — primary metric: linear_probe_hvg_pearson, intrinsic_*
echo "── eval: Stage 1 holdout (test split) ──"
python scripts/eval/stage1_tx.py \
    --prepared-dir "$PREP" \
    --split test \
    --ckpts "$STAGE1_CKPT" \
    --val-samples 76 --pool-spots 8000 \
    --linear-probe-genes 256 \
    --out "results/eval/cascade_${STAGE1_TAG}_stage1_test.csv"

# 2) DLPFC layer probe (stage=1 — RNA only, no Leiden) on the Stage 1 ckpt.
echo "── eval: DLPFC layer probe (Stage 1 RNA-only) ──"
python scripts/eval/dlpfc_eval.py \
    --dlpfc-dir /data/spatiallibd \
    --vocab "$PREP/hvg_vocab_dict.json" \
    --ckpts "$STAGE1_CKPT" \
    --stage 1 \
    --out "results/eval/cascade_${STAGE1_TAG}_dlpfc_stage1.csv"

# 3) Stage 1.5 in-dist eval (kNN overlap, smoothness, augmentation consistency).
if [[ -f "$STAGE15_CKPT" ]]; then
    echo "── eval: Stage 1.5 in-dist spatial metrics ──"
    python scripts/eval/stage15_indist.py \
        --prepared-dir "$PREP" \
        --ckpts "$STAGE15_CKPT" \
        --split test --max-samples 30 \
        --out "results/eval/cascade_${STAGE15_TAG}_indist.csv"
else
    echo "[run_all] WARNING: $STAGE15_CKPT missing — skipping in-dist eval"
fi

echo "=========================================================="
echo "[run_all] cascade complete."
echo "  Stage 1 ckpt   : $STAGE1_CKPT"
echo "  Stage 1.5 ckpt : $STAGE15_CKPT"
echo "  Eval csvs      : results/eval/cascade_*"
echo "=========================================================="
