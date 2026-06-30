# SPATULA Vocab Design & Data-Prep Pipeline

This document specifies **how the spot-level gene vocabulary is built**, **how
each sample is processed into a shard**, and **how the runtime tokenizer
converts a spot's gene-expression vector into the masked-modeling token
sequence**. It also captures the design principles we follow when deciding
*which genes belong in the vocab in the first place.*

The intended audience is anyone modifying:
- `scripts/data/prepare.py`
- `src/mm_align/data/pairs.py` (runtime normalization)
- `src/mm_align/models/top_hvg_gene.py` (zero-removal + token packing)
- `configs/stage1/data.yaml`

---

## 1. Design principles

A SPATULA vocab serves the same role as a language-model vocab: it lists the
**tokens the model is allowed to talk about**. A spot's gene-expression vector
becomes a sentence of (token, value) pairs, and the model learns to predict
masked tokens from the surrounding context.

The corollary is that **every gene in the vocab consumes capacity** — embedding
parameters, softmax output dimension, attention bandwidth. So selection must
be deliberate.

The principles below are listed in priority order. Higher principles override
lower ones. **Vocab size is a CONSEQUENCE of these principles**, not a target
to be hit. `n_hvg` is a cap for when the principle-driven survivor pool
exceeds the model's capacity budget; default `null` lets the filters
determine the count.

| # | Principle | Mechanism |
|---|---|---|
| 1 | **Clinically meaningful markers MUST be present** | `must_include_genes` (154 curated symbols) |
| 2 | **HVG ranking**: variability is the MSM signal | Seurat-style binned normalized dispersion (orders the vocab, only truncates when `n_hvg` cap hits) |
| 3 | **Protein-coding focus**: interpretable, well-annotated | `restrict_to_gene_types: [protein_coding, IG_*, TR_*]` |
| 4 | **Non-PC genes allowed if clinically informative** | added through `must_include_genes` (lncRNA / linc / pseudogene survives only via curation) |
| 5 | **No NLP "stopwords"** — house-keeping genes carry low MSM signal | `heg_top_k` default `0`; if enabled, expect to push out useful HVG |
| 6 | **No NLP "hapax legomena"** — drop genes that appear in too few samples/spots | `min_sample_prevalence`, `min_spot_prevalence` |
| 7 | **No noise** — multi-mapper aggregators, viral contigs, BAC clones, lab artefacts | strengthened `_NOISE_PATTERNS` + GTF rescue |

### Why HEG is OFF by default

Highly expressed genes that are *also* stable (ACTB, B2M, GAPDH, HSPA8) are
the spatial-transcriptomics analog of NLP stopwords. They occupy every spot at
roughly the same level, so the masked symbol task on them collapses to "pick
the gene whose mean is highest" — the model learns a trivial prior. They also
crowd out useful HVG slots.

Enable `heg_top_k > 0` only when you specifically want stable signals as
positional anchors and have budget to spare.

### Why prevalence filters matter

Without them, the HVG dispersion ranking happily promotes genes that fire in a
single sample at high level — the per-gene variance looks enormous, but the
gene is uninformative for 99% of training spots. NLP min-document-frequency
fixes the same problem. Recommended defaults for spot data:

```yaml
min_sample_prevalence: 0.02    # gene must be expressed in ≥2% of samples
min_spot_prevalence:  0.0005   # and in ≥0.05% of spots
```

---

## 2. Pipeline at a glance

```
┌────────────────────────────────────────────────────────────────────┐
│                    SOURCE ENUMERATION                              │
│   hest  (649) ∪ st1k (672) ∪ spatialcorpus (88) = 1409 samples     │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                  PER-SAMPLE LOAD + CLEAN                           │
│  read raw counts → symbol normalize → noise filter → GTF rescue    │
│         (with per-sample QC dict accumulator)                      │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│              BUILD GLOBAL VOCAB  (select_global_hvg)               │
│   ① per-gene streaming stats   (sum, sum², raw, nz_spots,          │
│                                  nz_samples)                       │
│   ② filter:  raw ≥ min_raw_counts                                  │
│   ③ filter:  sample_prev ≥ min_sample_prevalence                   │
│   ④ filter:  spot_prev   ≥ min_spot_prevalence                     │
│   ⑤ restrict: gene_type ∈ {protein_coding, IG_*, TR_*}             │
│   ⑥ rank:   Seurat binned-dispersion → top-n_hvg                   │
│   ⑦ MUST:   force-include curated markers (TP53, MMP, HSP, ...)    │
│   ⑧ HEG:    (optional, default off) union top-K by mean raw count  │
│   ⑨ save:   hvg_vocab.json + hvg_vocab_dict.json + sample_qc.csv   │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│           COMPUTE GLOBAL GENE STATS  (gene_stats.npz)              │
│   subsample 100 spots / sample → log-norm → per-gene:              │
│     mean, std, median, mad,                                        │
│     nonzero_mean, nonzero_std, nonzero_median, nonzero_count       │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│              WRITE SHARDS  (process_sample, per sample)            │
│   raw counts → normalize_total(1e4) → log1p →                      │
│   project onto hvg_vocab → hvg_log[spots, n_hvg] (float32, gzip)   │
│   {prepared_dir}/{sample}.h5 with                                  │
│     /barcode  /coords  /uni_feat  /novae_latent  /hvg_log          │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                       (training-time only)
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│      RUNTIME NORMALIZE  (PairedSpotDataset.__getitem__)            │
│   hvg = shard["hvg_log"][spot_idx]                                 │
│   apply gene_norm.mode (see §5):                                   │
│     none | global_z | global_robust_z | nonzero_z | global_median  │
│   clip ±gene_norm.clip (default 8)                                 │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│   ZERO_REMOVAL + TOKEN_FROM_VOCAB  (TopHVGGeneEncoder._pack_nonzero)│
│   real = hvg > 0                                                   │
│   order = argsort(~real, stable=True)[:, :L_max]                   │
│   gene_ids = hvg_token_ids[order]      (B, L_max)                  │
│   values   = gather(hvg, 1, order)     (B, L_max)                  │
│   attention_mask = arange(L_max) < seq_lens                        │
│                                                                    │
│   → masked-modeling backbone consumes (gene_ids, values, mask)     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage details

### 3.1 Source loading (`_load_sample_adata`)

Source dispatch:

| source | reader | gene IDs as | size hint |
|---|---|---|---|
| `hest` | `anndata.read_h5ad('/data/hest/st/{id}.h5ad')` | HGNC symbol | typical 1K–5K spots |
| `st1k` | `read_st1k_sample` — CSV | HGNC symbol | typical 1K–20K spots |
| `spatialcorpus` | `read_spatialcorpus_sample` — h5ad | Ensembl ID → HGNC via GTF | up to 1M cells (capped at 50K rows BEFORE dedup) |

The spatialcorpus reader subsamples rows *before* ENSG-to-symbol dedup so the
dense pivot doesn't blow past 50 GB. Without this cap, full-resolution
spatialcorpus h5ads stalled the pipeline for 20+ minutes per sample.

### 3.2 Symbol cleaning (`_clean_adata_var_names`)

Two stages of cleaning:

**A) Symbol normalize** (`_clean_symbol`):
- Strip genome prefix: `GRCH38_TP53` → `TP53`
- Strip version: `ENSG00000141510.5` → `ENSG00000141510`
- Strip suffix: `TP53.AS1` → `TP53-AS1`
- Uppercase

**B) Noise patterns** (`_NOISE_RE`):

```
^MT-, ^MT\.                       (mitochondrial)
^RPS, ^RPL                        (ribosomal — house-keeping noise)
^BLANK_, ^NEGCONTROL, ^UNASSIGNED (negative controls)
^[0-9]                            (numeric-leading)
^A[CDFJLP][0-9]+, ^LOC[0-9]+,
  ^RP[0-9]+, ^LINC[0-9]+          (uncharacterized/non-coding placeholders)
^.{3,}P[0-9]+$                    (pseudogene suffix; ≥3 chars before P\d+
                                   so TP53/MMP1/MAP2 are NOT caught)
^DEPRECATED                       (HGNC-flagged deprecated symbols)
^AMBIGUOUS\[                      (STAR multi-mapper aggregates;
                                   571/1152 of prepared_4k's "unknown")
^NC_[0-9]                         (RefSeq contigs — SARS-CoV-2 ORF probes)
^CT[A-D]-[0-9], ^LL[0-9A-Z]+-,    (BAC/fosmid clone IDs;
  ^GS[12]-, ^LA16C-, ^KB-[0-9],    395/1152 of prepared_4k's "unknown")
  ^CH[0-9]+-[0-9], ^XX[A-Z]+
```

**GTF rescue**: a pattern-hit gene is **kept** if its GTF `gene_type` is
`protein_coding` or an IG/TR rearrangement type. This rescues ARHGAP1, CDC42EP1,
NUP62 etc. that look like pseudogenes by pattern but are real coding genes.

### 3.3 Per-gene streaming stats (`select_global_hvg`)

For each train sample we **subsample to ≤10K spots** (`MAX_SPOTS_HVG`) to bound
memory, log-normalize independently, and accumulate per gene:

```
g_n[gene]          ← Σ n_spots
g_sum[gene]        ← Σ log1p(normalized x)
g_sq[gene]         ← Σ (log1p(normalized x))²
g_raw[gene]        ← Σ raw count
g_nz_spots[gene]   ← Σ (x > 0)              # for min_spot_prevalence
g_nz_samples[gene] ← Σ 1[any spot has x>0]  # for min_sample_prevalence
```

Per sample we also write a row to `sample_qc.csv`:
`{sample_id, source, n_genes_raw, n_genes_clean, n_genes_noise, n_spots,
n_genes_pc, n_genes_lncrna, n_genes_pseudo, n_genes_unknown_gt,
median_total_counts, frac_zero_spots}`.

### 3.4 Filters (in order)

1. `raw ≥ min_raw_counts` (default `3`) — Scanpy `filter_genes(min_counts=)` equivalent
2. `sample_prev ≥ min_sample_prevalence` — kills hapax legomena (NLP "min_doc_freq")
3. `spot_prev ≥ min_spot_prevalence` — kills 1-spot-in-1-sample extreme rares
4. `sample_prev ≤ max_sample_prevalence` *(default off, 1.01)* — optional NLP "the" cut
5. `spot_prev ≤ max_spot_prevalence` *(default off, 1.01)* — optional pan-expression cut
6. `gene_type ∈ restrict_to_gene_types` (if set) — kills lncRNA / pseudogene / unknown

> **Why `max_*_prevalence` is off by default.** A house-keeping gene is
> `high prevalence ∩ low variance`. The dispersion ranking already pushes
> such genes to the *bottom* of the vocab (low variance → low normalized
> dispersion). With `n_hvg` cap or `priority_rank`-based clipping, they
> drop out naturally. The upper-bound knobs are kept as escape hatches for
> the cases where a survey of `vocab.csv` shows a specific class of
> ubiquitous-but-noisy genes that ranking alone didn't suppress.

### 3.5 HVG ranking (Seurat-style) — cap, not target

For each gene compute log-mean `log(1+μ)` and dispersion `var/μ`. Bin by
log-mean (20 bins by quantile), then within each bin normalize dispersion by
the bin's median and MAD: `(d − med) / (1.4826 · mad)`. Sort descending by
normalized dispersion.

**`n_hvg` is a CAP, not a target.** When `n_hvg=null` (recommended), the
vocab keeps every gene that survives §3.4's filters — the principles, not a
fixed number, determine vocab size. The dispersion ranking still runs so
the vocab order is interpretable, but no truncation happens unless the
filter-survivor pool exceeds the desired token budget.

> Rationale: fixed-`n_hvg` (e.g. 4096) is a *capacity* knob disguised as a
> *selection* criterion. If the principle-driven filters return 1800 quality
> genes, padding to 4096 just re-admits the noise we filtered out. If they
> return 3000, capping at 2048 silently drops 952 genes the filters approved.
> Better to let the filters decide size, and tighten/loosen the filter
> thresholds when the resulting vocab is too large/small.

This matches `scanpy.pp.highly_variable_genes(flavor='seurat',
n_top_genes=n_hvg)` semantically (when capped) but never materializes the
`#spots × #union-genes` matrix (which would be ~10M × 30K = 1.2 TB dense).

### 3.6 must_include (curated markers)

After ranking, **force-include 154 curated symbols** (TP53, MMP*, HSP*, SFRP*,
CTNNB1, EGFR, KRT*, immune T/B/M markers, ...) that are *present in the
candidate pool* but ranked outside the top-N. These widely-expressed
clinical markers tend to have low binned dispersion (they're expressed in many
tissues) and get pushed out by hyper-variable single-sample-only genes
otherwise. Final vocab size = `n_hvg + |new must_include|`.

If a curated marker is **absent from the candidate pool** (filtered earlier or
never expressed), it is logged as "skipped" — that's how we discover GTF alias
issues like `OCT4 → POU5F1` and `SFRP3 → FRZB`.

### 3.7 HEG union (optional, default off)

If `heg_top_k > 0`, append the top-K genes by `Σ raw count` (not already in
vocab) onto the end. Stable housekeeping anchors. **Default 0** for the
reason in §1.

### 3.8 Vocab persistence

| file | format | purpose |
|---|---|---|
| `hvg_vocab.json` | flat JSON list: `["TP53","MMP1",...]` | shard `hvg_log` indexing (production) |
| `hvg_vocab_dict.json` | `{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[UNK]":3,"TP53":4,...}` | encoder token table |
| **`vocab.csv`** | **per-gene rich metadata** (see below) | **interactive discussion / QC / priority tracking** |
| `gene_stats.npz` | per-gene `mean, std, median, mad, nonzero_mean, nonzero_std, nonzero_median, nonzero_count` | runtime norm |
| `sample_qc.csv` | per-train-sample processing stats | process_qc |

**`vocab.csv` columns** (sortable in Excel / pandas):

| column | meaning |
|---|---|
| `gene` | HGNC symbol |
| `rank` | original dispersion rank (0 = highest variability) |
| `gene_type` | from GTF (`protein_coding` / `lncRNA` / `unknown` / ...) |
| `must_include` | true if listed in `must_include_genes` |
| `sample_prev` | fraction of train samples where gene has ≥ 1 nz spot |
| `spot_prev` | fraction of spots (across train samples) where gene > 0 |
| `n_spots_seen` | total spots that *saw* this gene's column |
| `n_samples_present` | absolute count for sample_prev |
| `mean_log1p`, `std_log1p` | aggregate over all spots (incl. zeros) |
| `nonzero_mean_log1p`, `nonzero_std_log1p` | aggregate over x > 0 only |
| `raw_total` | Σ raw counts across train pool |
| `norm_dispersion` | Seurat binned dispersion z (vocab ordering key) |
| `in_filter_pool` | true if survived all filters (false = must_include or HEG outside pool) |
| `priority` | composite: must_include × 10 + in_filter_pool × 5 − rank × 1e-4 |
| `priority_rank` | row index after `priority` sort |

---

## 4. Per-sample shard write (`process_sample`)

For every sample (across all splits):

```python
adata = _load_sample_adata(sid, source)            # raw counts, cleaned var_names
sc.pp.normalize_total(adata, target_sum=1e4)       # row sum → 1e4
sc.pp.log1p(adata)                                  # log(1+x)
hvg_mat = np.zeros((n_spots, len(hvg_vocab)), dtype=np.float32)
for j, g in enumerate(hvg_vocab):
    i = adata.var_names.get_loc(g)
    if i is not None:
        hvg_mat[:, j] = X[:, i]                    # missing genes stay 0
h5: /hvg_log  = hvg_mat   (gzip-4)
    /barcode, /coords, /uni_feat, /novae_latent, /patch_idx
```

Spatialcorpus / Xenium files are additionally **row-capped at 100K** before
shard write to prevent OOM cascades.

---

## 5. Runtime normalization (`PairedSpotDataset`, §`gene_norm.mode`)

The shard stores `log1p(normalized)` values. At training time the dataset
applies one of the following transforms per spot:

| `mode` | Formula at `x > 0` | At `x == 0` | When to use |
|---|---|---|---|
| `none` | `x` | `0` | Baseline / debugging |
| `global_z` | `(x − μ_all[g]) / max(σ_all[g], min_scale)` | shifted by `-μ/σ` | μ, σ over all values; biased by zeros |
| `global_robust_z` | `(x − median[g]) / max(mad[g], min_scale)` | shifted | ⚠ on ST data ≥99% genes have median=mad=0 — broken |
| **`nonzero_z`** (recommended) | `(x − μ_nz[g]) / max(σ_nz[g], min_scale)` | **`0` preserved** | μ/σ over NON-zero values; zero-removed-token semantics intact |
| **`global_median`** (Geneformer-style) | `x / max(nonzero_median[g], min_scale)` | **`0` preserved** | Mirrors Geneformer's `X_norm = X_scaled / median_g` |

All modes apply `np.clip(z, −clip, +clip)` (`clip=8` default) to bound the
Fourier value embedding's input range.

### What we share / don't share with Geneformer

| | Geneformer | SPATULA |
|---|---|---|
| Row sum | `normalize_total(1e4)` | `normalize_total(1e4)` ✓ |
| Log step | — (no log) | `log1p` |
| Per-gene baseline | `÷ nonzero_median(g)` (`global_median` here) | `(x − μ_nz) / σ_nz` (`nonzero_z` here) |
| Token value | rank-order only | value passed to Fourier value embedding |
| Token sequence | sorted by rank | sorted by original vocab position; values carry magnitude |

Geneformer can skip `log` because **it discards absolute value** — only the
ranking matters. SPATULA passes the value into the Fourier value embedding, so
we need bounded dynamic range (hence `log1p`) AND per-gene baseline removal
(hence `nonzero_z` or `global_median`).

---

## 6. Zero-removal + tokenization (`_pack_nonzero`, top_hvg_gene.py)

This is the analog of NLP tokenizer: convert a fixed-length value vector into
a *sparse* sequence of `(token_id, value)` pairs.

```python
hvg            : (B, K)   float32   — runtime-normalized
real           = hvg > 0             — non-zero positions are "real words"
seq_lens       = real.sum(dim=1)     — (B,) words per spot
L_max          = seq_lens.max()      — longest sentence in batch
order          = argsort(~real, stable=True)[:, :L_max]   # real positions first
gene_ids       = hvg_token_ids[order]                     # vocab token (≥ N_SPECIAL=4)
values         = gather(hvg, 1, order)                    # paired (normalized) value
attention_mask = arange(L_max) < seq_lens                 # PAD elsewhere
gene_ids[~mask] = PAD_ID;  values[~mask] = 0
```

Properties:
- **Variable length per spot** (`seq_len ∈ [min_seq_len, K]`). Earlier we saw
  `seq_len_mean ≈ 500` on PC-only `prepared_smoketest` vs `≈ 8` on the
  contaminated `prepared_4k` — vocab contamination directly drives sparsity.
- **CLS not packed here** — the model prepends a `[CLS]` learnable token.
- **MASK injection** happens inside the encoder (`_force_mask_in_eval`
  enables masking during validation too).

The downstream Pre-LN transformer attends over this packed sequence; the
masked-symbol head reads the masked positions and predicts gene_id via CE
over `vocab_size = K + N_SPECIAL`.

---

## 7. What to monitor

Three QC scripts share the same `prepared_dir` and report into
`results/eda/<prepared_name>/`:

| Script | Output | Key questions answered |
|---|---|---|
| [scripts/eval/vocab_qc.py](../scripts/eval/vocab_qc.py) | `vocab_qc/` | Is the vocab clean? Are clinical markers in? What's the gene-type breakdown? |
| [scripts/eval/process_quality_qc.py](../scripts/eval/process_quality_qc.py) | `process_qc/` | Does processing work per-source / per-organ / per-sample? Which samples are red flags? |
| **[scripts/eval/validate_vocab.py](../scripts/eval/validate_vocab.py)** | **`validate_vocab/`** | **Does each shard's `hvg_log` actually reflect the source AnnData? What does the post-zero-removal seq_len distribution look like?** |
| [scripts/expand_hvg_vocab.sh](../scripts/expand_hvg_vocab.sh) | `prepared_*` (full pipeline) | Bumps vocab size and re-runs prepare end-to-end |

### Red-flag thresholds

| signal | healthy | red flag |
|---|---|---|
| `unknown` (no GTF match) fraction in vocab | < 5% | > 20% |
| protein_coding fraction | > 80% | < 50% |
| Missing curated markers | 0–3 (alias misses only) | > 10 |
| Median sample prevalence | > 0.05 | < 0.02 |
| Mean `seq_len` per spot | > 100 | < 30 |

The PC-only fix took `prepared_4k` from `seq_len_mean ≈ 8` to
`seq_len_mean ≈ 500` on the smoke test — a 60× density gain, almost entirely
from cutting out AMBIGUOUS / BAC clone / pseudogene noise.

---

## 8. Configuration reference (`configs/stage1/data.yaml`)

```yaml
data:
  prepared_dir: /workspace/mm_align/results/cache/prepared_expanded
  sources:
    hest: true
    st1k: true
    spatialcorpus: true

  # vocab size — CAP, not target.  null = let principles decide.
  n_hvg: null

  # candidate-pool restrictions (BEFORE HVG ranking)
  restrict_to_gene_types:        # null = legacy "all noise-survivors"
    - protein_coding
    - IG_C_gene
    - IG_V_gene
    - IG_J_gene
    - IG_D_gene
    - TR_C_gene
    - TR_V_gene
    - TR_J_gene
    - TR_D_gene
  min_sample_prevalence: 0.02    # primary lever when n_hvg=null (NLP min_doc_freq)
  min_spot_prevalence:   0.0005
  max_sample_prevalence: 1.01    # set < 1.0 to drop NLP "the" (e.g. 0.98)
  max_spot_prevalence:   1.01    # set < 1.0 to drop pan-expressed (e.g. 0.50)

  # vocab augmentation
  must_include_genes: [TP53, MMP1, HSP90AA1, SFRP1, ...]   # 154 markers
  heg_top_k: 0                                              # leave OFF unless intentional

  # runtime normalization
  gene_norm:
    mode: nonzero_z              # or global_median (Geneformer-style)
    stats_path: .../gene_stats.npz
    eps: 1.0e-06
    min_scale: 0.05
    clip: 8.0
```

---

## 9. Operational checklist when changing vocab

1. Edit `configs/stage1/data.yaml` (vocab knobs above).
2. Decide on a fresh `prepared_dir` (compare side-by-side) or overwrite — if
   overwriting, **empty the old dir first** because shards from the old vocab
   are not index-compatible with the new vocab.
3. Run `scripts/data/prepare.py --config configs/stage1/data.yaml --rebuild_vocab`.
4. Run `scripts/eval/vocab_qc.py --prepared-dir <new_dir>` — verify red-flag table.
5. Run `scripts/eval/process_quality_qc.py --prepared-dir <new_dir>` — verify per-source / per-organ coverage.
6. Sync `model.transcriptomics.hvg_in_dim` (auto-handled in `scripts/train.py`).
7. Re-run Stage 1 pretraining.
