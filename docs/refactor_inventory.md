# Refactor Inventory

Snapshot of file classification BEFORE the Stage-aligned refactor.
"canonical" = stay or get promoted; "legacy" = archive; "reference" = move
out of the code path; "delete" = no longer referenced.

## Configs

| Path | Status | Destination |
|---|---|---|
| `configs/data.yaml` | legacy (top-level duplicate of stage1) | `configs/_archive/data.yaml` |
| `configs/model.yaml` | legacy | `configs/_archive/model.yaml` |
| `configs/train.yaml` | legacy | `configs/_archive/train.yaml` |
| `configs/experiments/stage1.yaml` | legacy (duplicate of `configs/stage1/experiment.yaml`) | `configs/_archive/experiments_stage1.yaml` |
| `configs/experiments/{barlow,cca,clip,jepa,s2l}.yaml` | **canonical** Stage 2 variants | keep |
| `configs/stage1/data.yaml` | **canonical** | keep |
| `configs/stage1/model.yaml` | **canonical** | keep |
| `configs/stage1/experiment.yaml` | **canonical** | keep |
| `configs/stage1/train.yaml` | **canonical** | keep |
| `configs/stage1/data_4k.yaml` | legacy (4k vocab, superseded by clip-runtime) | `configs/_archive/stage1_data_4k.yaml` |
| `configs/stage1/model_4k.yaml` | legacy | `configs/_archive/stage1_model_4k.yaml` |
| `configs/stage1/experiment_main.yaml` | **canonical** PDF main-candidate | keep |
| `configs/stage1/model_main.yaml` | **canonical** PDF main-candidate | keep |
| `configs/stage_spatial/*.yaml` | **canonical** Stage 1.5 | rename → `configs/stage15/` |
| `configs/sweep/stage1_ours_tx.yaml` | **canonical** Stage 1 sweep | keep |
| `configs/sweep/stage2_align.yaml` | **canonical** Stage 2 sweep | keep |
| `configs/sweep/ours_tx.yaml` | legacy (pre-stage1 sweep) | `configs/_archive/sweep_ours_tx.yaml` |
| `configs/sweep/default.yaml` | legacy default | `configs/_archive/sweep_default.yaml` |
| `configs/sweep/smoke.yaml` | reference smoke | keep (rename `sweep/smoke.yaml`) |

## Scripts

| Path | Status | Destination |
|---|---|---|
| `scripts/prepare_data.py` | canonical | `scripts/data/prepare.py` |
| `scripts/refresh_gene_stats.py` | canonical | `scripts/data/refresh_gene_stats.py` |
| `scripts/audit_gene_symbols.py` | canonical | `scripts/data/audit_gene_symbols.py` |
| `scripts/build_predefined_vocab.py` | canonical | `scripts/data/build_predefined_vocab.py` |
| `scripts/make_clipped_vocab.py` | canonical | `scripts/data/make_clipped_vocab.py` |
| `scripts/dataset_eda.py` | reference | `scripts/data/dataset_eda.py` |
| `scripts/vocab_qc.py` | canonical | `scripts/eval/vocab_qc.py` |
| `scripts/validate_vocab.py` | canonical | `scripts/eval/validate_vocab.py` |
| `scripts/process_quality_qc.py` | canonical | `scripts/eval/process_quality_qc.py` |
| `scripts/eval_tx_encoder.py` | canonical | `scripts/eval/stage1_tx.py` |
| `scripts/eval_linearprobe.py` | canonical | `scripts/eval/linear_probe.py` |
| `scripts/eval_zero_shot.py` | canonical | `scripts/eval/zero_shot.py` |
| `scripts/extract_embeddings.py` | canonical | `scripts/eval/extract_embeddings.py` |
| `scripts/train.py` | canonical (used by all stages) | `scripts/train/_main.py` (re-exported) |
| `scripts/train_ours_tx.sh` | canonical | `scripts/train/stage1.sh` |
| `scripts/train_main_candidate.sh` | canonical | `scripts/train/stage1_main.sh` |
| `scripts/train_stage2_align.sh` | canonical | `scripts/train/stage2.sh` |
| `scripts/train_stage_spatial.sh` | canonical | `scripts/train/stage15.sh` |
| `scripts/stage_spatial.py` | canonical | `scripts/train/stage15.py` |
| `scripts/sweep.py` | canonical | keep at top of scripts/ |
| `scripts/build_vocab_ppt.py` | canonical | `scripts/viz/vocab_ppt.py` |
| `scripts/make_figures.py` | canonical | `scripts/viz/figures.py` |
| `scripts/make_method_figure.py` | canonical | `scripts/viz/method_figure.py` |
| `scripts/viz_normalization.py` | canonical | `scripts/viz/normalization.py` |
| `scripts/viz_spatial.py` | canonical | `scripts/viz/spatial.py` |
| `scripts/resplit.py` | canonical | `scripts/data/resplit.py` |
| `scripts/resplit_stage1.py` | legacy duplicate of `resplit.py` | delete after verifying |
| `scripts/expand_hvg_vocab.sh` | legacy (one-shot run) | `scripts/_archive/expand_hvg_vocab.sh` |
| `scripts/run_experiments.sh` | reference dispatcher | keep at top |
| `scripts/ablations/` | canonical | rename → `scripts/ablation/` |

## guides → docs

| Source | Destination |
|---|---|
| `guides/vocab.md` | `docs/design/vocab.md` |
| `guides/spatial_foundation.md` | `docs/design/stage15_spatial_jepa.md` |
| `guides/training_and_ablation.md` | `docs/design/training_and_ablation.md` |
| `guides/usage.md` | `docs/archive/legacy_notes/usage.md` |
| `guides/description.md` | merge into top-level `README.md` if missing, else archive |
| `guides/context.md` | `docs/archive/legacy_notes/` |
| `guides/harness.md` | `docs/archive/legacy_notes/` |
| `guides/contexts/RNA_Encoder_Research_Strategy_Finalized.pdf` | `docs/archive/contexts/` |
| `guides/contexts/spatula_research_overview.pdf` | `docs/archive/contexts/` |
| `guides/contexts/SEAL/` | `references/SEAL/` |
| `guides/contexts/*.py` | `references/context_scripts/` |
| `guides/contexts/*.ipynb` | `references/context_scripts/` |
| `guides/contexts/*.md` | `references/context_scripts/` |

## src/mm_align — DONE (Phase D)

Subdomain split applied + every import path updated:
- `models/{tx,image,spatial,alignment}/`
- `objectives/{tx,spatial,alignment}/`

Legacy duplicates (top-level `clip.py`, `jepa.py`, `vicreg.py`, `cross_attn.py`,
`mmae.py`, `encoders.py`) are NOT under src/ anymore — moved to
`references/legacy_code/` so import scanners stop tripping on them.

## Results / generated files (not touched)

- `results/cache/prepared_expanded/**` — production prepared shards (DO NOT MOVE)
- `results/runs/**` — trained ckpts
- `results/eda/**`, `results/eval/**` — QC outputs
