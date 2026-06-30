# Stage 1 Training, Vocab Clipping & Ablation Guide

This is the practical companion to [`vocab.md`](vocab.md).  vocab.md explains
**how the vocab is built**; this file explains **how to run training on it,
how to shrink the vocab without re-running prepare, and what knobs exist for
ablation studies**.

---

## 1. How to launch Stage 1 training

### 1.1 The canonical command

```bash
# from /workspace/mm_align
bash scripts/train/stage1.sh
```

That's it.  `train_ours_tx.sh` reads `configs/sweep/stage1_ours_tx.yaml`,
injects `stage1_only=true`, then dispatches the actual launch.  Under the
hood it runs:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
    scripts/train.py \
        --experiment configs/stage1/experiment.yaml \
        --tag stage1_ours_tx_stage1_feature \
        --epochs 100 --batch-size <from sweep> \
        --model configs/stage1/model.yaml \
        --data  configs/stage1/data.yaml \
        --train configs/stage1/train.yaml \
        --image-backbone feature \
        --stage1-only
```

### 1.2 Don't run accelerate directly on the .sh wrapper

`accelerate launch scripts/train/stage1.sh` fails because the wrapper is a
**bash script** — accelerate expects a Python file as its target.  Two valid
launch forms:

| Form | Command |
|---|---|
| **Wrapper script (recommended)** | `bash scripts/train/stage1.sh` |
| **Direct accelerate** | `accelerate launch --num_processes 8 --mixed_precision bf16 scripts/train.py --experiment configs/stage1/experiment.yaml ...` |

### 1.3 Environment knobs (optional)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_TIMEOUT=7200 \
STRATIFY=1 \
bash scripts/train/stage1.sh
```

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces memory
  fragmentation when batch sizes vary (long-tailed seq_len spots).
- `NCCL_TIMEOUT=7200` — 2-hour barrier timeout (default is 10 min, may fire
  on slow shard reads).
- `STRATIFY=1` — re-emit `splits.json` with organ-stratified 8:1:1 before
  training (`scripts/data/resplit.py`).

### 1.4 Output layout

```
results/runs/stage1_ours_tx_stage1_feature/
    train.log
    metrics.csv              # val/tx_self/* per epoch
    ckpt_tx_encoder.pt       # for Stage 2 reuse
    tb_events/               # tensorboard scalars
```

---

## 2. Per-spot sequence length: why it matters

### 2.1 The numbers (vocab = 19,183, post zero-removal)

| Source | mean | median | p5 | p95 | max |
|---|---:|---:|---:|---:|---:|
| HEST (649 shards) | 2,105 | 1,964 | 759 | 3,963 | **13,665** |
| ST1K (672 shards) | 2,487 | 2,387 | 931 | 4,392 | 12,730 |
| spatialcorpus (99 shards) | 1,993 | 1,865 | 551 | 3,873 | 10,652 |

So the average spot uses ~11% of the vocab; p95 ≈ 4.4K; **outlier spots reach
13K+**.

### 2.2 Why we need to bound it

The MSM transformer's attention is O(L²) in sequence length:

| L per spot | (B=1024, heads=8, head_dim=64) per-layer attention tensor |
|---|---|
| 500 | ~16 GB |
| 1,000 | ~67 GB |
| 2,000 | ~268 GB — OOM |
| 4,000 | ~1 TB — OOM |

The previous Stage-1 OOM was a related but separate problem (symbol_head
logits at full L_max).  That's now fixed in `aligner.forward` (gather to
masked positions BEFORE the head — see `vocab.md` §6).  But attention itself
still scales O(L²), so **outlier-long spots can still OOM a batch**.

### 2.3 Two complementary defences

| Mechanism | Where | What it does |
|---|---|---|
| **Vocab clip** (§3) | data layer | Trim the *vocab* — every spot's nonzero count drops proportionally |
| **`max_seq_len` cap** (§4) | data layer | Trim per-spot *tokens* — pick a random subset of nonzero positions when a spot exceeds the cap |

You can use either alone or together.  Recommended baseline: clip vocab to
~4–8K + leave `max_seq_len=0`.  If outlier shards still trip OOM, set
`max_seq_len ≈ p95 of seq_len`.

---

## 3. Runtime vocab clip — without re-running prepare

The on-disk shards (`hvg_log`) keep the full 19,183 columns.  At training
time, the dataset picks a column subset and the model sees a smaller vocab.

### 3.1 Build the clipped lookup

```bash
# 1. Pick top-K by priority_rank (must_include first, then dispersion)
python scripts/data/make_clipped_vocab.py --top-k 4096

# Outputs in results/cache/prepared_expanded/:
#   clip4096_vocab.json           # flat list (subset of 19,183)
#   clip4096_vocab_dict.json      # {gene → token_id} with specials at front
#   clip4096_keep_indices.npy     # int64 column indices into the full hvg_log
```

`make_clipped_vocab.py` reads `vocab.csv` and takes the top-K rows by
`priority_rank` (so all 152 must_include markers are kept, then dispersion
order).  Tested: top-K=4096 keeps **all 152 curated markers** plus the 3,944
best-dispersion genes.

### 3.2 Wire it into training

Add to `configs/stage1/data.yaml`:

```yaml
data:
  vocab_clip:
    keep_indices_path: /workspace/mm_align/results/cache/prepared_expanded/clip4096_keep_indices.npy
```

That's all.  `train.py` autodetects the path, the dataset slices `hvg_log`
on the fly, `gene_stats` is sliced to match, and `model.hvg_in_dim` is
auto-synced to the clipped count.

### 3.3 Effects (estimated for K=4096)

| Metric | Full (19,183) | Clip 4,096 |
|---|---:|---:|
| Vocab size | 19,183 | 4,096 |
| Avg seq_len/spot | ~2,200 | ~470 (proportional) |
| symbol_head logits at masked positions | 0.15 × B × L × 19,187 | 0.15 × B × L × 4,100 |
| Attention O(L²) | × 22× vs L=470 | baseline |

Token budget per step drops ~5× at K=4096 — same effective batch size becomes
feasible.  For deeper transformers / longer training, K=4096 is the safe
default; K=8192 if you have room.

---

## 4. Per-spot token cap (`max_seq_len`)

When a spot's nonzero count exceeds `max_seq_len`, we randomly drop excess
positions (without replacement) so the post zero-removal sequence is bounded.

```yaml
# configs/stage1/data.yaml
data:
  max_seq_len: 1024     # 0 = no cap (default)
```

- Truncation is **random**, not "top-by-value" — preserves the long-tail
  distribution of expressed genes per spot.
- Applies AFTER `vocab_clip` (so the cap is relative to the clipped vocab).
- Only affects outlier spots; median spots are untouched.

Pick `max_seq_len` by looking at the
`results/eda/prepared_expanded/validate_vocab/seqlen_distribution.csv` p95
or p99 column.

---

## 5. Ablation matrix

### 5.1 Vocab design ([configs/stage1/data.yaml](../configs/stage1/data.yaml))

| Option | Values to try | Hypothesis |
|---|---|---|
| `n_hvg` | null vs 2048 vs 4096 vs 8192 | Capacity ↑ ↔ generalisation |
| `restrict_to_gene_types` | [PC+IG/TR] vs [PC] vs null | Non-coding gene helpfulness |
| `min_sample_prevalence` | 0.02 / 0.05 / 0.10 | Stricter floor → cleaner vocab, fewer rare markers |
| `min_spot_prevalence` | 0.0005 / 0.005 | — |
| `max_sample_prevalence` | 1.01 (off) / 0.98 / 0.5 | Explicit house-keeping cut (usually redundant with dispersion) |
| `must_include_genes` | full 154 / 0 / hand-picked | Curation impact |
| `heg_top_k` | 0 / 256 / 1024 | High-expression union |

Vocab-clip ablation (runtime, **no re-prepare**):
```yaml
data:
  vocab_clip:
    keep_indices_path: clip{4096,8192,2048}_keep_indices.npy
```

### 5.2 Normalize ([configs/stage1/data.yaml](../configs/stage1/data.yaml) `gene_norm`)

```yaml
gene_norm:
  mode: nonzero_z     # 'none' / 'global_z' / 'global_robust_z' / 'nonzero_z' / 'global_median'
  clip: 8.0
```

`nonzero_z` is recommended baseline.  `global_median` mirrors Geneformer's
scheme (still in log-space here because we passed log1p through).

### 5.3 Objective ([configs/stage1/experiment.yaml](../configs/stage1/experiment.yaml))

```yaml
masking:
  masking_obj: symbol     # 'symbol' / 'value' / 'both'
  mask_ratio: 0.15        # 0.10 / 0.15 / 0.30 / 0.50
  symbol_weight: 1.0
  value_weight: 0.0
align:
  use_align_jepa: false   # toggle predictive JEPA aux
  weight: 0.0             # 0.0 / 0.1 / 0.3 (per paper)
```

### 5.4 Model ([configs/stage1/model.yaml](../configs/stage1/model.yaml))

| Knob | Path | Defaults |
|---|---|---|
| Encoder dim | `transcriptomics.top_hvg_gene.dim` | 256 / **384** / 512 |
| Encoder depth | `transcriptomics.top_hvg_gene.depth` | 4 / **6** / 12 |
| Attention heads | `transcriptomics.top_hvg_gene.heads` | 4 / **8** |
| Fourier value freqs | `transcriptomics.top_hvg_gene.fourier_value_freqs` | **16** / 32 |
| Min seq_len (model side) | `transcriptomics.top_hvg_gene.min_seq_len` | 0 / 8 / 32 |
| Force mask in eval | `transcriptomics.top_hvg_gene.force_mask_in_eval` | **true** |

### 5.5 Training ([configs/stage1/train.yaml](../configs/stage1/train.yaml))

| Knob | Range |
|---|---|
| `batch_size` | 256 / 512 / **1024** |
| `lr` | 1e-4 / **2e-4** / 5e-4 |
| `weight_decay` | 0.01 / **0.1** |
| `warmup_epochs` | 0.5 / 1 / 5 |
| `epochs` | 50 / **100** / 200 |
| `min_lr_ratio` | 0.05 / 0.1 |
| `max_grad_norm` | 1.0 / 5.0 |

### 5.6 Sources ([configs/stage1/data.yaml](../configs/stage1/data.yaml))

```yaml
sources:
  hest: true            # HEST-only baseline vs HEST+ST1K vs all
  st1k: true
  spatialcorpus: true
```

### 5.7 Suggested ablation sequence

1. **Vocab size** (clip 19183 → 4096 → 8192) — establishes capacity scaling
2. **Normalize** (`nonzero_z` vs `global_median`) — paper-level comparison
3. **Mask ratio** (0.15 / 0.30 / 0.50)
4. **JEPA** on/off (paper baseline)
5. **HEG union** (0 vs 256)
6. **Sources** — HEST-only vs all (covariate shift test)

Each ablation needs only the relevant config file edited; `train.py` rebuilds
everything from scratch.

---

## 6. Common gotchas

| Symptom | Likely cause | Fix |
|---|---|---|
| `accelerate launch scripts/train/stage1.sh` errors out | accelerate expects a `.py`, not `.sh` | `bash scripts/train/stage1.sh` |
| `expected K=4096, got 19183` | model.hvg_in_dim out of sync | already auto-synced — pull latest `train.py` |
| `CUDA out of memory ... 30+ GiB` at first step | symbol_head logits at full L_max | already fixed in `aligner.forward` (gather to masked positions) |
| OOM on outlier shards (TENX HD, 13K seq_len) | attention O(L²) | set `data.max_seq_len: 1024` or use vocab_clip |
| Loss flat at ~ln(vocab_size) | random init — wait for warmup to finish | normal for first ~50 steps |
| `failed to map segment from shared object` | sklearn import during memory-fragmented main() | already fixed (sklearn imported at module top) |

---

## 7. Where to find everything

| Artefact | Path |
|---|---|
| Vocab build pipeline + theory | [guides/vocab.md](vocab.md) |
| **Training + ablation (this file)** | [guides/training_and_ablation.md](training_and_ablation.md) |
| Vocab production files | `results/cache/prepared_expanded/{hvg_vocab.json,vocab.csv,gene_stats.npz,sample_qc.csv}` |
| Peer-review (with tiers) | `results/eda/prepared_expanded/vocab_with_tiers.csv` |
| QC reports | `results/eda/prepared_expanded/{vocab_qc,process_qc,validate_vocab}/*.md` |
| Presentation slides | `results/eda/prepared_expanded/vocab_presentation.pptx` |
| Stage 1 launch script | `scripts/train/stage1.sh` |
| Stage 1 configs | `configs/stage1/{data,model,train,experiment}.yaml` |
| Sweep config | `configs/sweep/stage1_ours_tx.yaml` |
| Make clipped vocab | `scripts/data/make_clipped_vocab.py` |
