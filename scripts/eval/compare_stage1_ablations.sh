#!/usr/bin/env bash
# Compare Stage-1 ablation ckpts on TWO eval suites:
#   1) held-out split pool           → results/eval/stage1_compare.csv
#      Primary metrics: intrinsic_* + linear_probe_hvg_* + clean MVM.
#      HEST organ probe is opt-in via INCLUDE_ORGAN=1.
#   2) DLPFC layer recovery (external) → results/eval/dlpfc_compare.csv
#
# Each ckpt's input pipeline (vocab_clip + gene_norm) is recovered from its
# config.json — so vocab_clip 2k/4k/8k ablations are compared fairly. All
# reported metrics are VOCAB-SCALE INVARIANT.
#
# Usage:
#     bash scripts/eval/compare_stage1_ablations.sh                       # all stage1_*, both suites
#     bash scripts/eval/compare_stage1_ablations.sh stage1_vocab           # group filter (prefix)
#     bash scripts/eval/compare_stage1_ablations.sh stage1_norm stage1_va
#     BY_GROUP=1  bash scripts/eval/compare_stage1_ablations.sh           # SWEEP — one CSV per ablation group
#     SUITE=val     bash scripts/eval/compare_stage1_ablations.sh ...    # val pool only
#     SUITE=dlpfc   bash scripts/eval/compare_stage1_ablations.sh ...    # DLPFC only
#     SPLIT=val ... (use val instead of test — for debugging only, val was used during ES)
#     STAGE=15  bash ...   # Stage-1.5 eval: DLPFC also reports spatial leiden_ari/nmi
#     POOL_SPOTS=8000 VAL_SAMPLES=60 ... (eval-pool knobs)
#     SAMPLES_CAP=4   LEIDEN_RES=1.0 ... (DLPFC knobs)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

PREPARED_DIR="${PREPARED_DIR:-results/cache/prepared_expanded}"
DLPFC_DIR="${DLPFC_DIR:-/data/spatiallibd}"
POOL_SPOTS="${POOL_SPOTS:-5000}"
VAL_SAMPLES="${VAL_SAMPLES:-60}"
LINEAR_PROBE_GENES="${LINEAR_PROBE_GENES:-256}"
INCLUDE_ORGAN="${INCLUDE_ORGAN:-0}"
SKIP_SOURCE_PROBE="${SKIP_SOURCE_PROBE:-0}"
SPLIT="${SPLIT:-test}"                    # test (default, true holdout) | val | train
SAMPLES_CAP="${SAMPLES_CAP:-0}"          # 0 = all DLPFC samples
LEIDEN_RES="${LEIDEN_RES:-1.0}"
# Foundation stage we are scoring.
#   1  : Stage 1 (RNA-only).  DLPFC reports layer_probe + kNN purity only.
#   15 : Stage 1.5 (spatial). Also includes leiden_ari/nmi (the DLPFC spatial
#         clustering benchmark — meaningful once spatial JEPA is trained).
STAGE="${STAGE:-1}"
SUITE="${SUITE:-both}"                    # val | dlpfc | both
OUT_VAL="${OUT_VAL:-results/eval/stage1_compare.csv}"
OUT_DLPFC="${OUT_DLPFC:-results/eval/dlpfc_compare.csv}"

# `BY_GROUP=1` makes the wrapper sweep each ablation group separately and
# write a CSV per group (e.g. stage1_compare.vocab.csv, ...).  Default is
# false — runs all matched ckpts in one shared evaluation.
BY_GROUP="${BY_GROUP:-0}"
# The canonical ablation prefixes — extend as needed.
ABLATION_GROUPS=(vocab norm samp va obj mask)

PREFIXES=("$@")
if [[ ${#PREFIXES[@]} -eq 0 ]]; then
    PREFIXES=(stage1_)
fi

collect_ckpts() {
    # Args: prefixes...   stdout: one ckpt path per line.
    local prefixes=("$@")
    for d in results/runs/*/; do
        local name; name="$(basename "$d")"
        local keep=0
        for pre in "${prefixes[@]}"; do
            if [[ "$name" == "$pre"* ]]; then keep=1; break; fi
        done
        [[ "$keep" == "1" ]] || continue
        if [[ -f "$d/ckpt_tx_encoder_best.pt" ]]; then
            echo "$d/ckpt_tx_encoder_best.pt"
        fi
    done
}

run_eval() {
    # Args: tag (used in csv name)  ckpt1 ckpt2 ...
    local tag="$1"; shift
    local cks=("$@")
    if [[ ${#cks[@]} -eq 0 ]]; then
        echo "[compare] $tag: no ckpts matched — skipping"
        return
    fi
    local out_val="$OUT_VAL"
    local out_dlpfc="$OUT_DLPFC"
    if [[ "$tag" != "all" ]]; then
        out_val="${OUT_VAL%.csv}.${tag}.csv"
        out_dlpfc="${OUT_DLPFC%.csv}.${tag}.csv"
    fi
    echo "─────────── group=$tag  ckpts=${#cks[@]}  split=$SPLIT  suite=$SUITE"
    printf '  %s\n' "${cks[@]}"
    if [[ "$SUITE" == "val" || "$SUITE" == "both" ]]; then
        local stage1_extra=()
        [[ "$INCLUDE_ORGAN" == "1" ]] && stage1_extra+=(--include-organ-probe)
        [[ "$SKIP_SOURCE_PROBE" == "1" ]] && stage1_extra+=(--skip-source-probe)
        python scripts/eval/stage1_tx.py \
            --prepared-dir "$PREPARED_DIR" \
            --split "$SPLIT" \
            --ckpts "${cks[@]}" \
            --val-samples "$VAL_SAMPLES" \
            --pool-spots "$POOL_SPOTS" \
            --linear-probe-genes "$LINEAR_PROBE_GENES" \
            --out "$out_val" \
            "${stage1_extra[@]}"
    fi
    if [[ "$SUITE" == "dlpfc" || "$SUITE" == "both" ]]; then
        local extra=()
        [[ "$SAMPLES_CAP" -gt 0 ]] && extra+=(--samples-cap "$SAMPLES_CAP")
        local out_ps="${out_dlpfc%.csv}.per_sample.csv"
        python scripts/eval/dlpfc_eval.py \
            --dlpfc-dir "$DLPFC_DIR" \
            --vocab "$PREPARED_DIR/hvg_vocab_dict.json" \
            --ckpts "${cks[@]}" \
            --stage "$STAGE" \
            --leiden-resolution "$LEIDEN_RES" \
            --out "$out_dlpfc" \
            --per-sample-out "$out_ps" \
            "${extra[@]}"
    fi
}

if [[ "$BY_GROUP" == "1" ]]; then
    # Sweep each canonical ablation group on its own — separate CSVs so
    # within-group ranking isn't drowned by across-group variance.
    echo "[compare] BY_GROUP=1 — sweeping ${ABLATION_GROUPS[*]}"
    for grp in "${ABLATION_GROUPS[@]}"; do
        mapfile -t CKPTS < <(collect_ckpts "stage1_${grp}")
        run_eval "$grp" "${CKPTS[@]}"
    done
else
    mapfile -t CKPTS < <(collect_ckpts "${PREFIXES[@]}")
    if [[ ${#CKPTS[@]} -eq 0 ]]; then
        echo "[compare] no ckpts matched prefixes: ${PREFIXES[*]}"
        exit 1
    fi
    run_eval "all" "${CKPTS[@]}"
fi
