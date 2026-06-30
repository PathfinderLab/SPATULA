# Stage 1 Ablation Scripts

Self-contained launch scripts for each Stage-1 ablation we'd want to compare
in the paper.  All scripts share the same conventions:

- 8 GPUs, bf16, `accelerate launch` under the hood
- Each variant goes to `results/runs/<tag>/` with its own `train.log`,
  `metrics.csv`, `ckpt_tx_encoder.pt`
- Hyper-parameters are overridden via a generated `/tmp/...yaml` file —
  the base configs in `configs/stage1/` are untouched

## Time / cost profiles

Set `ABL_PROFILE` (default `normal`) — three presets balance signal vs. time:

| Profile | Epochs | limit_train | Time / variant | Use for |
|---|---:|---:|---:|---|
| `fast` | 10 | 200 | ~30 min | Quick ranking before committing |
| `normal` *(default)* | 20 | full | ~10 h | Paper-grade comparisons |
| `full` | 50 | full | ~25 h | Converged numbers (rarely needed) |

Examples:
```bash
ABL_PROFILE=fast   bash scripts/ablation/run_vocab_clip.sh    # 4 × 30 min = 2 h
ABL_PROFILE=normal bash scripts/ablation/run_normalize.sh     # 3 × 10 h = 30 h
ABL_EPOCHS=15      bash scripts/ablation/run_mask_ratio.sh    # custom override
```

## Quick reference

| Ablation | Script | What changes | Output tag prefix |
|---|---|---|---|
| Vocab size (runtime clip) | `run_vocab_clip.sh` | `data.vocab_clip.keep_indices_path` | `stage1_vocab_clip{N}` / `stage1_vocab_full` |
| Normalize mode | `run_normalize.sh` | `data.gene_norm.mode` | `stage1_norm_{none,nonzero_z,global_median}` |
| Mask ratio | `run_mask_ratio.sh` | `experiment.tokenizer.mask_ratio` | `stage1_mask_{0p15,0p30,0p50}` |
| Gene-JEPA (Stage-1 aux) | `run_jepa.sh` | `experiment.tx_self.{enable_masked_jepa,jepa_weight}` | `stage1_jepa_{off,lite,paper}` |
| Foundation objective | `run_objective.sh` | `experiment.tx_self.{masking_obj,symbol_weight,value_weight,enable_masked_jepa,jepa_weight}` | `stage1_obj_{msm,msm_jepa}` |
| Value augmentation | `run_value_aug.sh` | `model.transcriptomics.top_hvg_gene.value_aug.*` | `stage1_va_{keep,noise,dropout,mixed}` |
| Source mix | `run_sources.sh` | `data.sources` | `stage1_src_{hest_only,hest_st1k,all}` |
| **Sequence-length sampling** | **`run_seq_sampling.sh`** | **`data.max_seq_len` + `data.sampling.strategy`** | **`stage1_samp_{none,random,top_k,weighted}`** |

### Sampling strategies (when a spot expresses > `max_seq_len` genes)

| Strategy | Selection | Notes |
|---|---|---|
| `none` | no cap | full sequence, attention O(L²) cost |
| `random` | uniform drop | dropout-like augmentation, low bias |
| `top_k` | keep highest-value K | deterministic, Geneformer-style |
| `weighted` | `p ∝ value^alpha` (default α=1) | best of both: informative + stochastic |

Curated `must_include` markers are ALWAYS kept when expressed.

## Usage

Run an entire sweep:

```bash
bash scripts/ablation/run_vocab_clip.sh                  # 2048, 4096, 8192, full
bash scripts/ablation/run_normalize.sh                   # none, nonzero_z, global_median
bash scripts/ablation/run_mask_ratio.sh                  # 0.15, 0.30, 0.50
bash scripts/ablation/run_jepa.sh                        # off, lite, paper
bash scripts/ablation/run_sources.sh                     # hest_only, hest_st1k, all
```

Run a subset of variants:

```bash
bash scripts/ablation/run_vocab_clip.sh 4096 8192        # only two
bash scripts/ablation/run_normalize.sh nonzero_z global_median
bash scripts/ablation/run_mask_ratio.sh 0.30
```

Tighter sweep (e.g. for fast triage — fewer epochs):

```bash
ABL_EPOCHS=20 bash scripts/ablation/run_vocab_clip.sh
ABL_EPOCHS=30 ABL_BATCH=512 bash scripts/ablation/run_normalize.sh
```

## Environment overrides

| Env var | Default | Purpose |
|---|---|---|
| `ABL_PROFILE` | `normal` | `fast` / `normal` / `full` (see profile table above) |
| `ABL_EPOCHS` | from profile | Epochs per run |
| `ABL_LIMIT_TRAIN` | from profile | Subset of train samples (0 = all) |
| `ABL_NUM_PROC` | 8 | Accelerate process count (== GPU count) |
| `ABL_BATCH` | 256 | Batch per rank |
| `ABL_MP` | bf16 | `--mixed_precision` |
| `ABL_ES_PATIENCE` | 5 | Early-stop patience (tighter than baseline 8) |
| `ABL_MAX_SEQ_LEN` | 1024 | seq_sampling ablation default cap |
| `ABL_SAMPLING_ALPHA` | 1.0 | `weighted` strategy sharpness |
| `ABL_CLIP_INDICES` | (none) | path to clip{N}_keep_indices.npy to compose vocab_clip with another ablation |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Memory fragmentation guard |
| `NCCL_TIMEOUT` | 7200 | NCCL barrier seconds |

## Runs are sequential

Each script launches its variants ONE AT A TIME because all 8 GPUs are
held by the current run.  An ablation with 4 variants × 30 epochs runs in
roughly 4 × 30 × (epoch-time) — ~6h for vocab_clip with 4 sizes at our
~30 min/epoch.

To parallelise across machines, copy the script and run on a different host
with `ABL_NUM_PROC=8` set there.

## Comparing results

All metrics get written to `results/runs/<tag>/metrics.csv`.  Compare with:

```python
import pandas as pd, glob
runs = {p.split('/')[-2]: pd.read_csv(p)
        for p in glob.glob('results/runs/stage1_vocab_*/metrics.csv')}
for name, df in runs.items():
    last = df.tail(1).iloc[0]
    print(f"{name:25s} val_loss={last['val/tx_self/loss']:.3f}  "
          f"top1={last['val/tx_self/masked_symbol_top1_acc']:.3f}")
```

## Adding a new ablation

Create `scripts/ablation/run_<name>.sh`:

```bash
#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then VARIANTS=(default custom1 custom2); fi

for V in "${VARIANTS[@]}"; do
    TAG="stage1_my_ablation_${V}"
    DATA="/tmp/data_${TAG}.yaml"
    make_yaml_override "$DATA" "configs/stage1/data.yaml" \
        "data.some.key" "$V"
    launch_run "$TAG" "$DATA" \
        "configs/stage1/model.yaml" "configs/stage1/train.yaml" \
        "configs/stage1/experiment.yaml"
done
```

`make_yaml_override <out> <base> <key.dotted> <value> [<key2> <val2> ...]`
takes JSON-decoded values (so `'[a,b,c]'`, `0.15`, `true` all work) and
writes a fresh YAML.

## See also

- [`../../docs/design/vocab.md`](../../docs/design/vocab.md) — vocab build pipeline + design rationale
- [`../../docs/design/training_and_ablation.md`](../../docs/design/training_and_ablation.md) — full ablation knob matrix
- [`../make_clipped_vocab.py`](../make_clipped_vocab.py) — builds runtime clip files
