# SPATULA — Research Roadmap

Concise pipeline-level overview of the project's four stages.  Detailed
designs live in `docs/design/`; this file is the single source of truth for
"what runs where".

## Stages

| Stage | Output | Canonical entry-point | Config dir |
|---|---|---|---|
| **Stage 1 — RNA Foundation** | `ckpt_tx_encoder_best.pt` | `bash scripts/train/stage1.sh` | `configs/stage1/` |
| **Stage 1.5 — Spatial Foundation** | `ckpt_spatial_best.pt` | `bash scripts/train/stage15.sh` | `configs/stage15/` |
| **Stage 2 — Image ↔ Tx Alignment** | `ckpt_align_best.pt` | `bash scripts/train/stage2.sh <stage1_ckpt>` | `configs/experiments/*.yaml` + `configs/sweep/stage2_align.yaml` |
| **Evaluation** | reports + ckpt comparison tables | `python scripts/eval/stage1_tx.py --ckpts …` | — |

## Pipeline shape

```
                    ┌──────────────────────────────────┐
                    │  scripts/data/prepare.py          │
prepared shards  ◄──│  → results/cache/prepared_*/      │
                    │     hvg_vocab.json / vocab.csv /  │
                    │     gene_stats.npz / *.h5 / nnz   │
                    └──────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
Stage 1 │  MSM + (optional) Gene-JEPA on top_hvg_gene       │
        │  scripts/train/stage1.sh  /  stage1_main.sh       │
        │  → ckpt_tx_encoder_best.pt (frozen for next)      │
        └──────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
Stage1.5│  Spatial Predictive JEPA over per-sample KNN/     │
        │  radius/grid graph; tx + (optional) image inputs  │
        │  scripts/train/stage15.sh                         │
        │  → ckpt_spatial_best.pt                            │
        └──────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
Stage 2 │  Image ↔ Tx alignment (JEPA / CLIP / Barlow /     │
        │  CCA), frozen tx + UNI-LoRA                       │
        │  scripts/train/stage2.sh <stage1_ckpt>            │
        │  → ckpt_align_best.pt                              │
        └──────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
Eval    │  scripts/eval/*.py  (probes, retrieval, mvm,      │
        │  zero-shot, vocab QC, spatial QC)                  │
        └──────────────────────────────────────────────────┘
```

## Canonical configs

```
configs/
├── stage1/                       # RNA Foundation
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   ├── experiment.yaml           ← conservative baseline (MSM only, mask 0.15)
│   ├── experiment_main.yaml      ← PDF main-candidate (B2×C4 + JEPA λ=0.1)
│   ├── model_main.yaml           ← value_aug=mixed (C4)
│   └── train.yaml
├── stage15/                      # Spatial Foundation
│   ├── data.yaml
│   ├── model.yaml                ← arch: kgnn / smooth / kxformer
│   ├── train.yaml
│   └── experiment.yaml           ← jepa.mask_ratio / smoothness_weight
├── experiments/                  # Stage 2 alignment variants
│   ├── jepa.yaml
│   ├── clip.yaml
│   ├── barlow.yaml
│   ├── cca.yaml
│   └── s2l.yaml
├── sweep/                        # Sweep entry-points
│   ├── stage1_ours_tx.yaml
│   ├── stage2_align.yaml
│   └── smoke.yaml
└── _archive/                     # Legacy / replaced configs (DO NOT use)
```

## Ablation scripts

Live at `scripts/ablation/`.  All share `_common.sh` for env, profile
(`fast` / `triage` / `normal` / `full`), and `make_yaml_override` helper.

| Stage | Knob | Script |
|---|---|---|
| Stage 1 | foundation objective (MSM / MSM+JEPA) | `run_objective.sh` |
| Stage 1 | mask ratio (0.15 / 0.30 / 0.50) | `run_mask_ratio.sh` |
| Stage 1 | value augmentation (C1–C4) | `run_value_aug.sh` |
| Stage 1 | Gene-JEPA on/off + λ | `run_jepa.sh` |
| Stage 1 | vocab size (clip 2048/4096/8192/full) | `run_vocab_clip.sh` |
| Stage 1 | seq sampling (random / top_k / weighted) | `run_seq_sampling.sh` |
| Stage 1 | normalization (none / nonzero_z / global_median) | `run_normalize.sh` |
| Stage 1 | source mix (HEST only / +ST1K / all) | `run_sources.sh` |

Ablation triage profile (fast ranking before paper-grade rerun):
```bash
ABL_PROFILE=triage bash scripts/ablation/run_objective.sh
```

## Design docs

- [`docs/design/vocab.md`](../design/vocab.md) — vocab build + normalization
- [`docs/design/stage15_spatial_jepa.md`](../design/stage15_spatial_jepa.md) — Spatial Foundation
- [`docs/design/training_and_ablation.md`](../design/training_and_ablation.md) — ablation knobs in depth

## References

- `docs/archive/contexts/RNA_Encoder_Research_Strategy_Finalized.pdf` — primary PDF
- `docs/archive/contexts/spatula_research_overview.pdf` — SPATULA paper
- `references/SEAL/` — SEAL repository snapshot
- `references/context_scripts/` — pre-project helper notebooks/scripts
