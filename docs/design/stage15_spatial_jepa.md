# Spatial Foundation Stage — Design (Stage 1.5)

This document specifies **Stage Spatial Foundation**, the missing layer between
Stage 1 (RNA Foundation — MSM + Gene-JEPA) and Stage 2 (Multimodal VL-JEPA
Alignment).  Implements pages 9-10 and 18 of `RNA_Encoder_Research_Strategy_Finalized.pdf`,
combined with the **region-token design** from HyperST (`docs/archive/contexts/`
+ `references/hyperST/`) and the **block-masking strategy** from ST-JEPA
(`references/jepa/st-jepa.pdf`).

## Naming convention

We call "anchor spot + its k spatial neighbors" a **region** (HyperST calls
it a *niche*, ST-JEPA calls it a *cellular neighborhood* — we use "region"
to avoid biology jargon overlap).  Per anchor spot, the SpatialEncoder sees
up to **four modality channels**:

| Channel        | Source                                                    | Dim    |
|---|---|---:|
| `h_tx`         | Stage-1 frozen tx_encoder CLS at the anchor              | tx_dim |
| `h_img`        | Stage-1 frozen UNI feature at the anchor                 | 1536   |
| `h_region_tx`  | tx_encoder( aggregate(neighbors' `hvg_log`) )            | tx_dim |
| `h_region_img` | pool(neighbors' `uni_feat`)                              | 1536   |

### Region token mode (`model.spatial.region_token_mode`)

These four channels can be presented to the spatial backbone in two ways:

- **`fused`** *(baseline)* — `SpotFuser` concatenates all four channels and
  projects them through a small MLP into ONE `fuse_dim` token per anchor.
  The backbone sees `N_anchors` tokens.  Block masking replaces the whole
  cell-level token — both spot AND region channels disappear together.
  Cheap and stable, but **the I-JEPA story is muddier** because the target
  and its global context are tied.

- **`separate`** *(main candidate, R2)* — `SeparateTokenFuser` emits TWO
  tokens per anchor — a *spot token* (`h_tx`, `h_img`, posenc, type=0) and a
  *region token* (`h_region_tx`, `h_region_img`, posenc, type=1).  The
  backbone sees `2 × N_anchors` tokens with a graph that has:

      spot↔spot      original KNN edges
      region↔region  same edges shifted by +N
      spot↔region    bidirectional self-link  (anchor_i ↔ region_i)

  This unlocks the HyperST + I-JEPA combination: **region is visible
  context, spot is the masked target**.  See `mask_target` below.

### Mask target (`experiment.jepa.mask_target`, separate mode only)

| Target  | What it masks                                          | When to use |
|---|---|---|
| `spot`   | spot tokens only (R2)                                   | I-JEPA-faithful: predict masked spot latent from visible region. Main candidate. |
| `region` | region tokens only                                      | Inverse: predict region latent from visible spot context. |
| `both`   | spot AND region of the same anchor (block-style)        | R3.  Closer to fused-mode masking but each stream stays decodable. |

In fused mode the mask is always cell-level — only `spot`/`both` are valid
(they alias to the same thing), and `region` raises a config error.

### Region aggregation strategies (`data.region.*` config)

`region.tx_agg`:
- `mean` — `Σ log1p(x_i) / (k+1)` (HyperST default)
- `sum_log1p` — `log1p(Σ expm1(log1p(x_i)))` (sum in raw-count space)
- `weighted` — `exp(-d/σ)` weighted mean of `log1p`, neighbour-distance aware

`region.img_pool`:
- `mean` — average of neighbours' precomputed `uni_feat` (cheap)
- `attn` — learned attention pool (TODO; currently falls back to mean)
- (option) **raw-patch resize** — see `scripts/data/extract_region_uni_feat.py`:
  the anchor + 8 nearest patches are stitched into a 3×3 grid, resized to
  224×224 and pushed through UNI once.  Output cached as
  `<shard>.region_uni.npy`.  Preserves *visible* spatial layout inside a
  single 1536-d region embedding.

### Subgraph sampling (`data.subgraph_kind`)

| Kind     | What it does | Trade-off |
|---|---|---|
| `random` *(default)* | Random subset, then KNN rebuilt over the subset | Cheap; "neighbours" inside the subset may not be true tissue neighbours. |
| `ego`              | Sample-level KNN (cached to `<shard>.knn_k{k}.npz`) + BFS ego subgraph from a random centre | Reflects true tissue locality; matches HyperST / ST-JEPA neighbourhood semantics. Recommended. |

### Ablation entry points

```bash
bash scripts/train/stage15.sh                              # baseline (fused + random)
bash scripts/train/stage15_main.sh                         # main candidate (separate + spot mask + ego)
bash scripts/ablation/run_region_agg.sh                     # tx_agg × img_pool
bash scripts/ablation/run_region_token.sh                   # R0 / R1 / R2 / R3
ABL_SUBGRAPH_KIND=ego bash scripts/ablation/run_region_token.sh   # all R*, ego sampler
```

### Stage-1 input pipeline reuse (gene_norm + vocab_clip)

The frozen Stage-1 tx_encoder was trained on `nonzero_z`-normalised hvg
(`configs/stage1/data.yaml`, `gene_norm` block).  Stage 1.5 must apply the
**same normalisation** to:

1. **Anchor `hvg_log`** before pre-encoding into `h_tx` (cached under
   `<shard>.htx_<mode>.npy` — the cache name embeds the normaliser mode
   so switching modes invalidates the cache automatically).
2. **Region aggregates** (`mean` / `sum_log1p` / `weighted` of neighbours)
   before pushing through the same encoder to produce `h_region_tx`.

Without this, the encoder receives raw log1p at inference while it was
trained on z-scores — a large distribution shift.  The pipeline:

```
hvg_log[sel] → vocab_clip → gene_norm (Stage-1 cfg) → tx_encoder → h_tx
region_hvg   = aggregate(neighbours' hvg_log)
             → gene_norm                            → tx_encoder → h_region_tx
```

Source of truth for `gene_norm` / `vocab_clip` at Stage 1.5 launch time:
1. `ckpt['cfg_tx']['data']` (newer ckpts; written by the updated
   `save_tx_encoder_only`).
2. Fallback: `<stage1_run_dir>/config.json` (always present — older ckpts
   only had `prepared_dir` in `cfg_tx['data']`).
3. None → trainer emits a warning and runs without normalisation
   (verify the frozen encoder really doesn't expect any).

> Stage hierarchy (PDF p.18):
> 1. **RNA Foundation** = MSM + Gene-JEPA → Spot Encoder *(Stage 1, done)*
> 2. **Spatial Foundation** = Spatial JEPA → Spatial Embedding *(this stage)*
> 3. **Multimodal Foundation** = VL-JEPA Alignment → Shared Latent *(Stage 2)*
> 4. **Biological Foundation** = Transfer Learning *(downstream)*

---

## 1. Purpose

Stage 1 produces a **spot-level encoder** that knows what a spot looks like
*on its own* (cross-gene context within the spot).  But spatial transcriptomics
spots have **neighbors that carry structural information**: a tumor-stromal
boundary, a glandular cluster, a fibroblast halo.

Stage 1.5's job: **inject neighborhood context into each spot's embedding**
without rewriting the upstream encoders.  Self-supervised — no labels.

---

## 2. Inputs

Frozen, per spot:

| Modality | Source | Dim |
|---|---|---|
| RNA spot embedding `h_tx` | Stage 1 `ckpt_tx_encoder_best.pt` (frozen) | 256 (default) |
| Image patch feature `h_img` | UNI feature, optionally LoRA-tuned in Stage 2 | 1536 |
| Spatial coords `(x, y)` | Shard `/coords` | 2 |
| Sample id | Shard `attrs.sample_id` | (used only to scope graph) |

The Stage 1.5 trainable parameters live **on top of** these — Stage 1 stays
frozen.

---

## 3. Architecture

```
                      ┌─────────────────────────────────────────┐
spot i:               │ Spot Token = MLP_fuse([h_tx_i; h_img_i; │
  h_tx_i  ──┐         │                       pos_enc(x_i,y_i)])│   d_spatial
  h_img_i ──┼────────►│                                         │  ────────►
  (x,y)_i ──┘         └─────────────────────────────────────────┘

         (build KNN graph from (x,y) within sample)

                      ┌─────────────────────────────────────────┐
                      │ Spatial Transformer  /  GNN             │
spot tokens  ───────► │   - depth: 2-4 layers                   │
of one sample         │   - attention restricted to KNN graph   │
                      │   - relative positional bias from Δxy   │
                      └─────────────────────────────────────────┘
                                       │
                                       ▼
                  z_i  = contextualized spot latent (per spot, d_spatial)
```

Three swappable spatial backbones (selected via config):

| `spatial_arch` | Description | When to use |
|---|---|---|
| `kgnn` *(default)* | Light GAT with KNN edges + Δxy edge features | strong baseline |
| `kxformer` | Set-transformer with sparse KNN attention + relative-pos bias | bigger model, slower |
| `smooth` | Non-parametric mean over KNN neighbors (no learnable params) | F2 control from PDF p.10 |

`smooth` is the F2 ablation control — measures whether learned spatial
context beats simple averaging.

---

## 4. Objective — Spatial Predictive JEPA

Per the PDF (p.7-9), this is **latent prediction** at masked spots —
*not* expression reconstruction.  Closely parallels Stage 1's Gene-JEPA, but
the "tokens" are now spots and the "context" is the spatial graph.

### 4.1 Mask construction (ST-JEPA-style block masking)

Given a sample's spots S (typically ~1-5k):

1. Build the spatial graph from `(x,y)` (config: `knn` / `radius` / `grid`).
2. Sample `mask_ratio` × |S| **target cells** in one of two strategies
   (`experiment.jepa.mask_strategy`):
    - **`random`** — i.i.d. Bernoulli per cell (legacy).
    - **`block`** (default) — within each subgraph, grow connected blocks of
      `block_size` cells via random walks on the spatial graph and mask the
      whole block together.  ST-JEPA's recommended default — forces the
      model to predict masked cells from spatially adjacent visible
      context, which is much harder than scattered i.i.d. masking.
3. Replace each masked anchor's whole `SpotFuser` token (h_tx + h_img +
   region_tx + region_img) with the learnable `mask_embed`.
4. Pass the masked graph through the **student** SpatialEncoder.
5. Pass the **unmasked** graph through the **EMA teacher**.
6. Loss: `smooth_l1(z_student[mask], z_teacher[mask].detach())`
   (cosine optional, β = 1).

Masking is **per-subgraph aware** (the collator emits `subgraph_id`), so a
random walk seeded in subgraph A cannot leak into subgraph B even though
the batch is concatenated into one big disconnected graph.

### 4.2 Auxiliary smoothness loss (optional, default off)

`λ_smooth · Σ ||z_i − z_j||²` over KNN edges.  Acts as F4 control vs F3.

### 4.3 Training-time augmentations

| Aug | Purpose |
|---|---|
| Random-walk subgraph (size 200-500) | Stage 1.5 doesn't need the whole sample per step |
| Coord jitter (±5 px) | Bound to translation invariance |
| KNN drop edge (p=0.10) | Prevent over-reliance on dense neighborhoods |

---

## 5. Outputs

`results/runs/stage15_spatial_<tag>/`:

| File | Purpose |
|---|---|
| `ckpt_spatial_best.pt` | spatial encoder weights only |
| `metrics.csv` | val/spatial/{loss, kNN_overlap, smoothness} |
| `embeddings_val.npy` | (optional) cached `z_i` for downstream eval |

For Stage 2 reuse: `ckpt_spatial_best.pt` is loaded as the **Spatial Encoder**
inside VL-JEPA (PDF p.20).

---

## 6. Evaluation (PDF p.10, p.15)

| Metric | Computed how |
|---|---|
| Reconstruction loss (val) | smooth_l1 on held-out masked spots |
| **kNN graph overlap** | for each held-out spot, top-k embedding neighbors vs spatial neighbors — Jaccard |
| **Spatial domain ARI** (zero-shot) | Leiden cluster of `z` vs `organ` per spot |
| **Boundary preservation** | tissue-boundary spots' embedding distance > intra-region (PDF p.14) |
| **Smoothness-balance** | embedding kNN cosine vs spatial Δ-distance correlation |
| **Augmentation consistency** | same spot under 2 random subgraphs → cos sim |

Three F-conditions per PDF p.10 (built-in ablations):

| Cond. | What |
|---|---|
| `F1` | no Stage 1.5 (Stage 1 z_i alone) |
| `F2` | non-parametric KNN smoothing (`spatial_arch=smooth`) |
| `F3` | Spatial JEPA only |
| `F4` | Spatial JEPA + smoothness loss |

---

## 7. Configuration sketch — `configs/stage15/{data,model,train,experiment}.yaml`

```yaml
# configs/stage15/data.yaml
data:
  prepared_dir: results/cache/prepared_expanded   # reuse Stage 1 shards
  stage1_ckpt: results/runs/stage1_ours_tx_stage1_feature/ckpt_tx_encoder_best.pt
  graph:
    kind: knn          # knn / radius / grid
    k: 8
    radius_px: 600    # if kind == radius
  subgraph_size: 256   # spots per training step

# configs/stage15/model.yaml
model:
  spatial:
    arch: kgnn         # kgnn / kxformer / smooth
    d_spatial: 256
    n_layers: 3
    n_heads: 4
    fuse_dim: 256       # MLP_fuse output
    fuse_image: true    # set false to ablate image input

# configs/stage15/experiment.yaml
experiment:
  name: stage15_spatial_jepa
  jepa:
    mask_ratio: 0.30
    loss: smooth_l1     # smooth_l1 / cosine
    smoothness_weight: 0.0
    ema_momentum: 0.999

# configs/stage15/train.yaml — short stage (cheap)
train:
  epochs: 30
  batch_size: 32        # samples per step (each yields one subgraph of subgraph_size)
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 1
```

---

## 8. Launch (when implemented)

```bash
bash scripts/train/stage15.sh
```

Ablation:
```bash
bash scripts/ablation/run_spatial.sh           # F1/F2/F3/F4
bash scripts/ablation/run_spatial_graph.sh     # G1/G2/G3 graph types
```

---

## 9. Code skeleton — files to add

| Path | Purpose | Status |
|---|---|---|
| `src/mm_align/models/spatial_encoder.py` | `SpatialEncoder` (kgnn / kxformer / smooth backbones) | **NEW** |
| `src/mm_align/data/spatial_sampler.py` | Per-sample KNN graph + subgraph yielding | **NEW** |
| `src/mm_align/objectives/spatial_jepa.py` | Mask-and-predict latents + EMA teacher | **NEW** |
| `scripts/train/stage15.sh` | Launch wrapper | **NEW** |
| `scripts/train/stage15.py` | Stage-1.5 train.py (loads frozen Stage 1 ckpt, trains spatial only) | **NEW** |
| `configs/stage15/*.yaml` | data / model / train / experiment configs | **NEW** |
| `src/mm_align/evaluation/spatial_eval.py` | KNN overlap / ARI / boundary metrics | **NEW** |

---

## 10. Dependencies on Stage 1

Hard:
- Frozen `tx_encoder` from `ckpt_tx_encoder_best.pt` — must have a working spot encoder first.
- Shard format (`coords`, `uni_feat`, `barcode`) — unchanged.

Soft:
- Vocab clip choice from Stage 1 doesn't affect Stage 1.5 (we work in
  latent space, not the vocab).
- Stage 2 (VL-JEPA) will consume Stage 1.5's `ckpt_spatial_best.pt` as the
  "Spatial Encoder" of PDF p.20.

---

## 11. Why a separate stage (not bolted on Stage 2)

PDF p.9 puts Spatial JEPA explicitly as "sample-level self-supervised
adaptation" *between* the RNA encoder and downstream / multimodal use.
Two reasons:

1. **Modality-agnostic**: Stage 1.5 can run with image features turned off
   (`fuse_image: false`), giving a pure RNA-spatial encoder.  Useful for
   non-paired ST datasets (e.g. spatialcorpus shards that have no image).
2. **Cheap & swappable**: spatial encoder is small (a few M params) and
   trains in hours, not days.  We can ablate F1/F2/F3/F4 quickly without
   re-running the (expensive) Stage 1.
