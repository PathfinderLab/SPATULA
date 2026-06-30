# Stage 2 Downstream Evaluation & Ablation Plan

This document defines the Stage 2 evaluation layer beyond image-RNA retrieval.
It is intentionally task-oriented: what should be measured, which references
provide the task definitions, and how the ablations should be organized.

## 1. Stage 2 Meaning

Stage 2 trains an image-side encoder to align pathology patch representations
with the Stage 1 RNA encoder latent space.

Stage 2 should therefore be evaluated on three layers:

| Layer | Question | Example metrics |
|---|---|---|
| Alignment | Does image latent retrieve/match RNA latent? | Recall@K, MRR, modality gap |
| Molecular prediction | Does image latent preserve transcriptomic signal? | Pearson, Spearman, R2, MSE |
| Biological downstream | Does the aligned image encoder help pathology tasks? | AUROC, macro-F1, C-index |

The first two already exist in the repository. The third is the missing layer
needed for MSI/subtype/survival claims.

## 2. Reference Sources

| Reference | Use in this project |
|---|---|
| `references/HEST/` | HEST metadata, HEST benchmark task patterns, gene-expression prediction |
| `references/SEAL/` | Image-to-gene decoder, HEST-Bench style linear probing and metrics |
| `references/PathBench/` | Slide-level pathology benchmark task taxonomy and metric conventions |

The implementation should not import reference code directly in the training
path. Instead, references should be used to define task schemas, metrics, and
expected outputs.

## 3. Downstream Task Families

### 3.1 Patch/spot-level molecular task

Purpose:
- Test whether image features contain local molecular information.

Inputs:
- Patch or spot-level image embedding: `h_image` or `z_image`
- Target HVG expression from prepared HEST shard: `hvg`

Protocol:
- Freeze Stage 2 model.
- Extract embeddings for train/val/test spots.
- Train a lightweight probe: Ridge/MLP.
- Evaluate on held-out samples.

Metrics:
- Mean Pearson / Spearman across genes
- Per-gene Pearson table
- R2 / MSE
- Optional marker-panel metrics for clinically relevant genes

Existing support:
- `scripts/eval/linear_probe.py`
- `src/mm_align/evaluation/linear_probe.py`

Needed additions:
- HEST-Bench task wrapper with fixed gene panels and split reporting.
- Report top/bottom genes and marker-panel performance.

### 3.2 Slide-level MIL classification

Purpose:
- Test whether patch/spot embeddings support whole-slide pathology labels.

Candidate tasks:
- MSI prediction
- Tumor subtype prediction
- Lymph-node metastasis
- TNM stage classification
- Tissue/disease subtype classification

Inputs:
- Per-slide bag of patch/spot embeddings
- Slide-level labels from HEST metadata or external task CSV

Protocol:
- Freeze Stage 2 image encoder.
- Aggregate patch embeddings with a MIL head.
- Train only MIL head on train slides.
- Evaluate on held-out slides.

MIL heads:
- Mean pooling + linear classifier: baseline
- Attention MIL: main default
- Gated Attention MIL: stronger default

Metrics:
- Binary: AUROC, AUPRC, accuracy, F1
- Multiclass: macro-F1, balanced accuracy, one-vs-rest AUROC

Needed files:
- `src/mm_align/evaluation/mil.py`
- `scripts/eval/slide_mil.py`
- `configs/eval/stage2_downstream.yaml`

### 3.3 Slide-level survival

Purpose:
- Test whether the image/RNA-aligned representation improves clinical risk
prediction.

Candidate tasks:
- Overall survival
- Disease-free survival
- Disease-specific survival

Inputs:
- Per-slide embedding bag
- Survival time and event indicator

Protocol:
- Freeze Stage 2 image encoder.
- Train MIL Cox or discrete-time survival head.
- Evaluate on held-out slides.

Metrics:
- C-index
- Integrated Brier score if time grid is available

Needed files:
- `src/mm_align/evaluation/survival.py`
- Survival label adapter for PathBench-style tasks

## 4. Task Registry

Downstream tasks should be registered in a small schema rather than hard-coded
inside scripts.

Proposed schema:

```yaml
tasks:
  - name: hest_msi
    source: hest
    level: slide
    problem_type: binary_classification
    label_csv: /path/to/labels.csv
    sample_id_col: sample_id
    label_col: msi
    metric: auroc

  - name: pathbench_overall_survival
    source: pathbench
    level: slide
    problem_type: survival
    label_csv: /path/to/labels.csv
    sample_id_col: sample_id
    time_col: time
    event_col: event
    metric: c_index
```

This lets HEST, SEAL, and PathBench tasks share the same evaluator.

## 5. Stage 2 Ablation Matrix

Stage 2 ablations should be organized along three axes.

### 5.1 Image encoder ablation

Question:
- Does the aligned representation depend on the pathology foundation model?

Backbones already supported by `src/mm_align/models/image/foundation.py`:

| Backbone | Config name | Notes |
|---|---|---|
| UNI / UNI2-h | `uni` | current default |
| H0-mini | `h0mini` | lighter pathology FM |
| GigaPath | `gigapath` | large pathology FM |
| Virchow2 | `virchow2` | strong pathology FM |
| H-optimus-0/1 | `hoptimus0`, `hoptimus1` | optional |

Tuning modes:
- `none`
- `adapter`
- `lora`
- `partial:N`

Recommended first grid:

| ID | Backbone | Tune |
|---|---|---|
| I1 | `uni` | `lora` |
| I2 | `h0mini` | `lora` |
| I3 | `gigapath` | `lora` |
| I4 | `uni` | `adapter` |
| I5 | `uni` | `none` |

### 5.2 Alignment method ablation

Question:
- Which cross-modal objective is best for pathology-RNA alignment?

Current supported methods:
- `jepa`
- `clip`
- `barlow`
- `cca`
- `s2l`

Recommended first grid:

| ID | Method | Purpose |
|---|---|---|
| M1 | JEPA | main latent prediction hypothesis |
| M2 | CLIP | contrastive baseline |
| M3 | Barlow | non-contrastive redundancy reduction |
| M4 | CCA | correlation objective |
| M5 | S2L | soft contrastive baseline |

### 5.3 Tx encoder ablation

Question:
- Does our Stage 1 tx encoder provide better molecular supervision than
  existing transcriptomics embeddings?

Candidates:

| ID | Tx branch | Description |
|---|---|---|
| T1 | `ours_stage1` | frozen `top_hvg_gene` checkpoint |
| T2 | `novae` | frozen Novae latent adapter |
| T3 | `hvg_mlp` | HVG vector MLP baseline |
| T4 | `ours_stage1_main` | main-candidate Stage 1 checkpoint |

Expected config changes:
- `model.transcriptomics.kind`
- `--tx-ckpt` on/off
- `model.transcriptomics.use_novae`
- `model.transcriptomics.use_hvg`

## 6. Recommended Experiment Order

Run in this order to keep compute manageable.

1. **Method triage with UNI-LoRA**
   - JEPA vs CLIP vs Barlow
   - Metrics: retrieval, gene prediction, RankMe, modality gap

2. **Tx encoder triage**
   - ours Stage 1 vs Novae vs HVG MLP
   - Keep image encoder and method fixed

3. **Image encoder triage**
   - UNI vs H0-mini vs GigaPath
   - Keep tx encoder and method fixed

4. **Downstream task probe**
   - Run MSI/subtype MIL on the best 2-3 Stage 2 checkpoints

5. **Slide-level survival**
   - Add after classification task plumbing is stable

## 7. New Scripts To Add

| File | Purpose |
|---|---|
| `scripts/eval/stage2_downstream.py` | common dispatcher for Stage 2 downstream tasks |
| `scripts/eval/slide_mil.py` | slide-level classification MIL evaluator |
| `scripts/eval/slide_survival.py` | slide-level survival evaluator |
| `scripts/ablation/run_stage2_method.sh` | CLIP/JEPA/Barlow/CCA/S2L sweep |
| `scripts/ablation/run_stage2_image_encoder.sh` | UNI/H0-mini/GigaPath/etc. sweep |
| `scripts/ablation/run_stage2_tx_encoder.sh` | ours/ours-main/Novae/HVG baseline sweep |

## 8. Success Criteria

Stage 2 should not be selected by loss alone.

Minimum tracked metrics for checkpoint comparison:
- image-to-RNA Recall@10
- RNA-to-image Recall@10
- image-to-HVG Pearson
- RankMe image / tx
- modality gap
- downstream AUROC or macro-F1 when labels exist

For a paper claim, report:
- alignment metrics independently from RNA encoder metrics
- gene-expression prediction independently from retrieval
- at least one slide-level biological task if claiming clinical utility
