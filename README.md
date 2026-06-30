# SPATULA — Multimodal Spatial Transcriptomics Foundation Model

Two-stage (+ optional spatial foundation) self-supervised pretraining:

1. **Stage 1 — RNA Foundation**: MSM-based spot encoder on HVG vocab
2. **Stage 1.5 — Spatial Foundation**: Spatial Predictive JEPA on neighborhood graphs
3. **Stage 2 — VL Alignment**: image ↔ RNA alignment with frozen tx + UNI LoRA

See **[docs/strategy/roadmap.md](docs/strategy/roadmap.md)** for the canonical pipeline overview.

## Quickstart

```bash
# 1. Prepare shards once (~3 h)
PYTHONPATH=src python scripts/data/prepare.py \
    --config configs/stage1/data.yaml --rebuild_vocab --skip-novae --stratify

# 2. Stage 1 (RNA Foundation) — conservative baseline
bash scripts/train/stage1.sh

#    or the PDF main-candidate (mask 0.30 + value_aug mixed + Gene-JEPA λ=0.1)
bash scripts/train/stage1_main.sh

# 3. Stage 1.5 (Spatial Foundation)
bash scripts/train/stage15.sh

# 4. Stage 2 (VL Alignment)
bash scripts/train/stage2.sh \
    results/runs/stage1_ours_tx_stage1_feature/ckpt_tx_encoder_best.pt
```

## Layout

```
mm_align/
├── README.md                    # ← you are here
├── docs/
│   ├── strategy/roadmap.md      # canonical pipeline overview
│   ├── design/                  # vocab + stages + training + ablation docs
│   └── archive/                 # legacy notes + reference PDFs
├── configs/
│   ├── stage1/                  # RNA Foundation (data/model/train/experiment)
│   ├── stage15/                 # Spatial Foundation
│   ├── experiments/             # Stage-2 alignment variants
│   ├── sweep/                   # entry-point sweep YAMLs
│   └── _archive/                # legacy/replaced configs (do not use)
├── scripts/
│   ├── train/                   # stage1*.sh, stage15.{sh,py}, stage2.sh
│   ├── eval/                    # ckpt comparison + QC scripts
│   ├── data/                    # prepare/refresh/clip/audit
│   ├── viz/                     # figure + ppt builders
│   ├── ablation/                # per-knob ablation scripts (use _common.sh)
│   ├── train.py                 # canonical training entry-point (all stages)
│   ├── sweep.py                 # multi-experiment dispatcher
│   └── run_experiments.sh
├── src/mm_align/                # python package (models / data / objectives / evaluation)
├── references/                  # SEAL repo snapshot + context scripts
├── assets/                      # UNI weights, gene_vocab.csv, etc.
└── results/                     # cache, runs, eda, eval (generated)
```

## Stage entry-points

| Stage | Script | Output |
|---|---|---|
| 1 (baseline) | `scripts/train/stage1.sh` | `results/runs/stage1_ours_tx_stage1_feature/ckpt_tx_encoder_best.pt` |
| 1 (main-cand.) | `scripts/train/stage1_main.sh` | `results/runs/stage1_main_candidate/ckpt_tx_encoder_best.pt` |
| 1.5 | `scripts/train/stage15.sh` | `results/runs/<tag>/ckpt_spatial_best.pt` |
| 2 | `scripts/train/stage2.sh <stage1_ckpt>` | `results/runs/<tag>/ckpt_align_best.pt` |

## Ablation

```bash
# Triage (fast ranking, vocab clip + seq cap)
ABL_PROFILE=triage bash scripts/ablation/run_objective.sh
ABL_PROFILE=triage bash scripts/ablation/run_value_aug.sh
ABL_PROFILE=triage bash scripts/ablation/run_jepa.sh

# Paper-grade (full pool, 20+ epochs)
ABL_PROFILE=normal bash scripts/ablation/run_objective.sh

# Compare ckpts
python scripts/eval/stage1_tx.py \
    --prepared-dir results/cache/prepared_expanded \
    --ckpts results/runs/stage1_obj_*/ckpt_tx_encoder_best.pt
```

See **[scripts/ablation/README.md](scripts/ablation/README.md)** for the full ablation knob list and **[docs/design/training_and_ablation.md](docs/design/training_and_ablation.md)** for design rationale.

## Key design docs

- [`docs/design/vocab.md`](docs/design/vocab.md) — vocab build + normalization (Stage 1)
- [`docs/design/stage15_spatial_jepa.md`](docs/design/stage15_spatial_jepa.md) — Spatial Foundation design
- [`docs/design/training_and_ablation.md`](docs/design/training_and_ablation.md) — ablation matrix
- [`docs/design/stage2_downstream_and_ablation.md`](docs/design/stage2_downstream_and_ablation.md) — Stage 2 downstream benchmarks + encoder/method ablations
