#!/usr/bin/env bash
# Common helpers for ablation scripts.
#
# Sourced (not executed).  Defines:
#   - ROOT, PYTHONPATH, env hygiene
#   - launch_run <tag> <data_yaml> <experiment_yaml>     # one Stage-1 run
#   - make_data_override <out_yaml> <base_yaml> <key.path> <value> [<key.path> <value> ...]
#   - make_exp_override  <out_yaml> <base_yaml> <key.path> <value> [...]
#
# All ablation .sh files source this and call launch_run for each variant.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Standard env hygiene
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Default training hyperparams (override via env).
#
# Ablation runs use far fewer epochs than the production baseline (the
# baseline ran 50 epochs to find best-val 4.5453; ranking trends are clear
# by epoch ~15).  Three preset profiles via ABL_PROFILE:
#
#   ABL_PROFILE=fast    epochs=10, limit_train=200  (~30 min/variant) — ranking
#   ABL_PROFILE=normal  epochs=20, full data        (~10 h/variant)   — paper-grade
#   ABL_PROFILE=full    epochs=50, full data        (~25 h/variant)   — converged
#
# Custom env overrides take precedence over the profile.
ABL_PROFILE="${ABL_PROFILE:-normal}"
case "$ABL_PROFILE" in
    # Ultra-quick (~10 min/variant) — ranking sanity only, tiny train pool.
    fast)    : "${ABL_EPOCHS:=10}"; : "${ABL_LIMIT_TRAIN:=200}"; ;;
    # Recommended for objective/normalize/value_aug ranking:
    # vocab clip 4096 + ~400 samples, ~1.5 h/variant. Keep every other
    # method/process knob at baseline unless explicitly overridden.
    triage)  : "${ABL_EPOCHS:=20}"; : "${ABL_LIMIT_TRAIN:=400}"; \
             : "${ABL_CLIP_INDICES:=results/cache/prepared_expanded/clip4096_keep_indices.npy}"; ;;
    normal)  : "${ABL_EPOCHS:=20}"; : "${ABL_LIMIT_TRAIN:=0}";   ;;
    full)    : "${ABL_EPOCHS:=50}"; : "${ABL_LIMIT_TRAIN:=0}";   ;;
    *)       echo "[abl] unknown ABL_PROFILE=$ABL_PROFILE (use fast|triage|normal|full)"; exit 1 ;;
esac
export ABL_PROFILE ABL_CLIP_INDICES ABL_MAX_SEQ_LEN ABL_SAMPLING_STRATEGY
ABL_NUM_PROC="${ABL_NUM_PROC:-8}"
ABL_BATCH="${ABL_BATCH:-256}"
ABL_MP="${ABL_MP:-bf16}"
ABL_ES_PATIENCE="${ABL_ES_PATIENCE:-5}"   # tighter early-stop for ablation
ABL_VAL_EVERY="${ABL_VAL_EVERY:-2}"
ABL_VAL_MAX_BATCHES="${ABL_VAL_MAX_BATCHES:-0}"   # <=0 means full validation
ABL_STAGE1_QUICK_EVERY="${ABL_STAGE1_QUICK_EVERY:-$ABL_VAL_EVERY}"
ABL_CLEAN_MSM_EVERY="${ABL_CLEAN_MSM_EVERY:-$ABL_VAL_EVERY}"
ABL_GENE_SET_EVERY="${ABL_GENE_SET_EVERY:-0}"
export ABL_EPOCHS ABL_LIMIT_TRAIN ABL_NUM_PROC ABL_BATCH ABL_MP ABL_ES_PATIENCE
export ABL_VAL_EVERY ABL_VAL_MAX_BATCHES ABL_STAGE1_QUICK_EVERY ABL_CLEAN_MSM_EVERY ABL_GENE_SET_EVERY
echo "[abl] profile=$ABL_PROFILE  epochs=$ABL_EPOCHS  limit_train=$ABL_LIMIT_TRAIN"
echo "[abl] batch/rank=$ABL_BATCH  ranks=$ABL_NUM_PROC  mp=$ABL_MP  ES patience=$ABL_ES_PATIENCE"
echo "[abl] val_every=$ABL_VAL_EVERY  quick_every=$ABL_STAGE1_QUICK_EVERY  clean_msm_every=$ABL_CLEAN_MSM_EVERY  gene_set_every=$ABL_GENE_SET_EVERY"

# ── make_yaml_override <out> <base> <key.path1> <val1> [<key.path2> <val2> ...]
# Produces a YAML at <out> that is <base> with the given dotted keys overridden.
# Uses python-yaml (already required by the project).
make_yaml_override() {
    local out_yaml="$1"; local base_yaml="$2"; shift 2
    python - "$out_yaml" "$base_yaml" "$@" <<'PYEOF'
import sys, yaml, json
out, base, *overrides = sys.argv[1:]
with open(base) as f:
    cfg = yaml.safe_load(f)
# overrides come as alternating key, value pairs.
def setp(d, dotted, val):
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = d.get(k)
        if not isinstance(cur, dict):
            cur = {}
            d[k] = cur
        d = cur
    # Try JSON-decode first (so we can pass [a,b,c], 3.14, true, null).
    try:
        v = json.loads(val)
    except Exception:
        v = val
    d[keys[-1]] = v
for i in range(0, len(overrides), 2):
    setp(cfg, overrides[i], overrides[i+1])
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
}

# ── write_ablation_meta <tag> <data> <model> <train> <experiment>
# Optional per-variant variables consumed here:
#   ABL_GROUP, ABL_VARIANT, ABL_CHANGED_JSON
write_ablation_meta() {
    local tag="$1"; local d="$2"; local m="$3"; local t="$4"; local e="$5"
    local run_dir="results/runs/${tag}"
    mkdir -p "$run_dir"
    local changed_json="${ABL_CHANGED_JSON:-}"
    if [[ -z "$changed_json" ]]; then
        changed_json="{}"
    fi
    ABL_GROUP="${ABL_GROUP:-unknown}" \
    ABL_VARIANT="${ABL_VARIANT:-$tag}" \
    ABL_CHANGED_JSON="$changed_json" \
    python - "$run_dir/ablation_meta.json" "$tag" "$d" "$m" "$t" "$e" <<'PYEOF'
import json, os, sys, time
out, tag, data_yaml, model_yaml, train_yaml, exp_yaml = sys.argv[1:]
def parse_json_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return raw
meta = {
    "tag": tag,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "ablation_group": os.environ.get("ABL_GROUP", "unknown"),
    "ablation_variant": os.environ.get("ABL_VARIANT", tag),
    "changed_args": parse_json_env("ABL_CHANGED_JSON", {}),
    "base_configs": {
        "data": data_yaml,
        "model": model_yaml,
        "train": train_yaml,
        "experiment": exp_yaml,
    },
    "profile": os.environ.get("ABL_PROFILE"),
    "speed_overrides": {
        "limit_train": int(os.environ.get("ABL_LIMIT_TRAIN", "0")),
        "vocab_clip_keep_indices": os.environ.get("ABL_CLIP_INDICES", ""),
        "max_seq_len": os.environ.get("ABL_MAX_SEQ_LEN", ""),
        "sampling_strategy": os.environ.get("ABL_SAMPLING_STRATEGY", ""),
    },
    "train_overrides": {
        "epochs": int(os.environ.get("ABL_EPOCHS", "0")),
        "batch_size": int(os.environ.get("ABL_BATCH", "0")),
        "num_processes": int(os.environ.get("ABL_NUM_PROC", "0")),
        "mixed_precision": os.environ.get("ABL_MP"),
        "val_every_epoch": int(os.environ.get("ABL_VAL_EVERY", "0")),
        "stage1_quick_every": int(os.environ.get("ABL_STAGE1_QUICK_EVERY", "0")),
        "stage1_clean_msm_every": int(os.environ.get("ABL_CLEAN_MSM_EVERY", "0")),
        "gene_set_every": int(os.environ.get("ABL_GENE_SET_EVERY", "0")),
    },
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[abl] wrote {out}")
PYEOF
}

# ── launch_run <tag> <data_yaml> <model_yaml> <train_yaml> <exp_yaml> [extra train.py flags]
launch_run() {
    local tag="$1"; local d="$2"; local m="$3"; local t="$4"; local e="$5"; shift 5
    local extra_args=("$@")
    if [[ "$ABL_LIMIT_TRAIN" -gt 0 ]]; then
        extra_args+=(--limit-train "$ABL_LIMIT_TRAIN")
    fi
    # Triage / explicit env: clip vocab + cap seq_len globally so any
    # downstream ablation script gets the smaller token budget for free.
    if [[ -n "${ABL_CLIP_INDICES:-}" || -n "${ABL_MAX_SEQ_LEN:-}" \
            || -n "${ABL_SAMPLING_STRATEGY:-}" ]]; then
        # Build a per-variant data.yaml that applies the global overrides on
        # top of whatever the variant already chose.
        local d_patched="/tmp/${tag}_patched_data.yaml"
        local overrides=()
        if [[ -n "${ABL_CLIP_INDICES:-}" ]]; then
            overrides+=("data.vocab_clip.keep_indices_path" "$(realpath "$ABL_CLIP_INDICES" 2>/dev/null || echo "$ABL_CLIP_INDICES")")
        fi
        if [[ -n "${ABL_MAX_SEQ_LEN:-}" ]]; then
            overrides+=("data.max_seq_len" "$ABL_MAX_SEQ_LEN")
        fi
        if [[ -n "${ABL_SAMPLING_STRATEGY:-}" ]]; then
            overrides+=("data.sampling.strategy" "$ABL_SAMPLING_STRATEGY")
            overrides+=("data.sampling.keep_must_include" "true")
        fi
        make_yaml_override "$d_patched" "$d" "${overrides[@]}"
        d="$d_patched"
    fi
    write_ablation_meta "$tag" "$d" "$m" "$t" "$e"
    extra_args+=(--val-every-epoch "$ABL_VAL_EVERY")
    extra_args+=(--stage1-quick-every "$ABL_STAGE1_QUICK_EVERY")
    extra_args+=(--stage1-clean-msm-every "$ABL_CLEAN_MSM_EVERY")
    extra_args+=(--gene-set-every "$ABL_GENE_SET_EVERY")
    echo "=========================================================="
    echo "[ablation] $tag"
    echo "  data:  $d"
    echo "  model: $m"
    echo "  train: $t"
    echo "  exp:   $e"
    echo "  extra: ${extra_args[*]}"
    echo "=========================================================="
    accelerate launch --num_processes "$ABL_NUM_PROC" --mixed_precision "$ABL_MP" \
        scripts/train.py \
            --experiment "$e" \
            --tag "$tag" \
            --epochs "$ABL_EPOCHS" \
        --val-max-batches "$ABL_VAL_MAX_BATCHES" \
            --batch-size "$ABL_BATCH" \
            --model "$m" \
            --data  "$d" \
            --train "$t" \
            --image-backbone feature \
            --stage1-only \
            "${extra_args[@]}"
}
