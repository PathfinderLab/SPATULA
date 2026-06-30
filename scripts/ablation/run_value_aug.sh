#!/usr/bin/env bash
# Ablation: split VALUE AUGMENTATION under symbol masking.
#
# The model sees two token roles:
#   1) masked genes   : symbol=[MASK], value is keep/noise/dropout
#   2) unmasked genes : symbol=original, context value is weakly corrupted
#
# Hypothesis:
#   - masked_value_aug prevents a direct value→symbol shortcut.
#   - unmasked_value_aug should be weak; these values are the context needed
#     to learn a robust spot representation.
#
# Variants:
#   clean_context      masked 75/15/10, unmasked 100/0/0
#   weak_context_noise masked 75/15/10, unmasked 90/10/0   (recommended)
#   context_noise_drop masked 75/15/10, unmasked 85/10/5
#   hard_masked        masked 65/25/10, unmasked 90/10/0
#   keep               masked 100/0/0,  unmasked 100/0/0  (control)
#
# Each run lands at results/runs/stage1_va_<variant>/

source "$(dirname "$0")/_common.sh"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=(keep clean_context weak_context_noise context_noise_drop hard_masked)
fi

BASE_DATA="configs/stage1/data.yaml"
BASE_MODEL="configs/stage1/model.yaml"
BASE_TRAIN="configs/stage1/train.yaml"
BASE_EXP="configs/stage1/experiment.yaml"

profile_values() {
    local v="$1"
    case "$v" in
        keep)
            echo "keep 1.0 0.0 0.0 1.0 keep 1.0 0.0 0.0 0.15" ;;
        clean_context)
            echo "mixed 0.75 0.15 0.10 0.35 keep 1.0 0.0 0.0 0.15" ;;
        weak_context_noise)
            echo "mixed 0.75 0.15 0.10 0.35 mixed 0.90 0.10 0.00 0.15" ;;
        context_noise_drop)
            echo "mixed 0.75 0.15 0.10 0.35 mixed 0.85 0.10 0.05 0.15" ;;
        hard_masked)
            echo "mixed 0.65 0.25 0.10 0.50 mixed 0.90 0.10 0.00 0.15" ;;
        *)
            echo "[abl] unknown value_aug variant: $v" >&2
            echo "      use: keep clean_context weak_context_noise context_noise_drop hard_masked" >&2
            return 1 ;;
    esac
}

for V in "${VARIANTS[@]}"; do
    read -r M_MODE M_KEEP M_NOISE M_DROP M_STD U_MODE U_KEEP U_NOISE U_DROP U_STD < <(profile_values "$V")
    TAG="stage1_va_${V}"
    MODEL="/tmp/model_${TAG}.yaml"
    make_yaml_override "$MODEL" "$BASE_MODEL" \
        "model.transcriptomics.top_hvg_gene.value_aug.mode" "$M_MODE" \
        "model.transcriptomics.top_hvg_gene.value_aug.keep_p" "$M_KEEP" \
        "model.transcriptomics.top_hvg_gene.value_aug.noise_p" "$M_NOISE" \
        "model.transcriptomics.top_hvg_gene.value_aug.drop_p" "$M_DROP" \
        "model.transcriptomics.top_hvg_gene.value_aug.noise_std" "$M_STD" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.mode" "$M_MODE" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.keep_p" "$M_KEEP" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.noise_p" "$M_NOISE" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.drop_p" "$M_DROP" \
        "model.transcriptomics.top_hvg_gene.masked_value_aug.noise_std" "$M_STD" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.mode" "$U_MODE" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.keep_p" "$U_KEEP" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.noise_p" "$U_NOISE" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.drop_p" "$U_DROP" \
        "model.transcriptomics.top_hvg_gene.unmasked_value_aug.noise_std" "$U_STD"

    changed=$(python - "$V" "$M_MODE" "$M_KEEP" "$M_NOISE" "$M_DROP" "$M_STD" \
                       "$U_MODE" "$U_KEEP" "$U_NOISE" "$U_DROP" "$U_STD" <<'PYJSON'
import json, sys
(v, mm, mk, mn, md, ms, um, uk, un, ud, us) = sys.argv[1:]
print(json.dumps({
    "value_aug_profile": v,
    "masked_value_aug": {
        "mode": mm, "keep_p": float(mk), "noise_p": float(mn),
        "drop_p": float(md), "noise_std": float(ms),
    },
    "unmasked_value_aug": {
        "mode": um, "keep_p": float(uk), "noise_p": float(un),
        "drop_p": float(ud), "noise_std": float(us),
    },
}, sort_keys=True))
PYJSON
)

    ABL_GROUP="value_aug" ABL_VARIANT="$V" ABL_CHANGED_JSON="$changed" \
    launch_run "$TAG" "$BASE_DATA" "$MODEL" "$BASE_TRAIN" "$BASE_EXP"
done

echo "[ablation] split value_aug sweep complete. Runs in results/runs/stage1_va_*"
