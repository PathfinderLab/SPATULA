# SPATULA 프로젝트 가이드

이 문서는 `README.md`보다 실험 실행 관점에 더 가까운 한글 안내서입니다.
프로젝트의 각 stage가 무엇을 학습하고, 어떤 스크립트와 config를 쓰며,
어떤 산출물을 다음 단계로 넘기는지 빠르게 확인하는 용도입니다.

## 전체 흐름

SPATULA는 spatial transcriptomics와 pathology image를 단계적으로 정렬하는
foundation model 연구 프로젝트입니다.

```text
Data preparation
  -> Stage 1 RNA Foundation
  -> Stage 1.5 Spatial Foundation
  -> Stage 2 Image-RNA Alignment
  -> Evaluation / Ablation
```

각 단계의 핵심 산출물은 다음 단계의 입력으로 쓰입니다.

| 단계 | 목적 | 주요 산출물 |
|---|---|---|
| Data preparation | HEST/ST1K/SpatialCorpus를 공통 shard와 vocab으로 변환 | `results/cache/prepared_*/*.h5`, `hvg_vocab.json`, `gene_stats.npz` |
| Stage 1 | RNA spot encoder 사전학습 | `ckpt_tx_encoder_best.pt` |
| Stage 1.5 | spatial graph 기반 spot embedding 적응 | `ckpt_spatial_best.pt` |
| Stage 2 | pathology image와 RNA latent 정렬 | `ckpt_align_best.pt` |
| Evaluation | encoder/alignment 품질 평가 | metrics, tables, figures |

## 디렉터리 역할

```text
configs/       stage별 실험 설정
scripts/       데이터 준비, 학습, 평가, ablation 실행 스크립트
src/mm_align/  Python package 구현
docs/          설계 문서와 연구 로드맵
references/    SEAL 등 외부 참고 코드와 context script
assets/        모델 weight, gene list 등 입력 asset
results/       cache, checkpoint, 평가 결과
```

`src/mm_align/`는 Python package namespace입니다. repo root가
`/workspace/mm_align`이어도 import는 `mm_align.*`를 유지합니다.

`pyproject.toml`이 추가됐기 때문에 다음 두 가지 방식 모두 가능합니다.

```bash
# 1) editable install — 모든 스크립트에서 PYTHONPATH 없이 import 가능
pip install -e .

# 2) PYTHONPATH 명시 — install 없이도 동작
PYTHONPATH=src python scripts/...
```

## Data Preparation

역할:
- 여러 source의 raw spatial transcriptomics 데이터를 읽습니다.
- gene symbol을 정규화하고 noise gene을 제거합니다.
- global HVG/marker vocab을 만듭니다.
- 각 sample을 학습용 HDF5 shard로 저장합니다.
- runtime normalization에 필요한 gene statistics를 저장합니다.

주요 스크립트:

| 스크립트 | 역할 |
|---|---|
| `scripts/data/prepare.py` | 전체 shard/vocab/stat 생성의 canonical entrypoint |
| `scripts/data/refresh_gene_stats.py` | 기존 shard에서 gene normalization 통계 재생성 |
| `scripts/data/make_clipped_vocab.py` | 큰 vocab을 runtime clip하기 위한 index 생성 |
| `scripts/data/audit_gene_symbols.py` | gene symbol mapping/QC 점검 |
| `scripts/data/dataset_eda.py` | prepared dataset EDA |

대표 실행:

```bash
PYTHONPATH=src python scripts/data/prepare.py \
  --config configs/stage1/data.yaml \
  --rebuild_vocab \
  --skip-novae \
  --stratify
```

주요 config:
- `configs/stage1/data.yaml`

## Stage 1: RNA Foundation

목적:
- spot을 `(gene symbol, expression value)` token set으로 보고 RNA encoder를 학습합니다.
- 핵심 objective는 MSM, 즉 masked symbol modeling입니다.
- optional DINO-style consistency는 clean/unmasked teacher view와 masked/noisy student view의 spot embedding을 맞추는 보조 loss입니다.
- Gene-level JEPA는 masked token latent prediction 보조 objective로 별도 ablation합니다.

핵심 구현:

| 모듈 | 역할 |
|---|---|
| `src/mm_align/models/tx/top_hvg_gene.py` | gene symbol + Fourier(value) tokenizer/encoder |
| `src/mm_align/objectives/tx/masked.py` | MSM, optional MVM, DINO-style consistency, Gene-JEPA loss |
| `scripts/train.py` | Stage 1/2 공통 training loop |

실행 스크립트:

| 스크립트 | 역할 |
|---|---|
| `scripts/train/stage1.sh` | conservative baseline, MSM only, mask ratio 0.15 |
| `scripts/train/stage1_main.sh` | PDF main-candidate, mask 0.30 + mixed value aug + Gene-JEPA |

대표 실행:

```bash
# Baseline
bash scripts/train/stage1.sh

# Main candidate
bash scripts/train/stage1_main.sh
```

주요 config:

| 파일 | 역할 |
|---|---|
| `configs/stage1/data.yaml` | data source, vocab, normalization |
| `configs/stage1/model.yaml` | baseline model |
| `configs/stage1/model_main.yaml` | main-candidate value augmentation |
| `configs/stage1/experiment.yaml` | baseline MSM objective |
| `configs/stage1/experiment_main.yaml` | main-candidate objective |
| `configs/stage1/train.yaml` | optimizer, epoch, logging, checkpoint |

산출물:
- `results/runs/<tag>/ckpt_tx_encoder_best.pt`
- Stage 1.5와 Stage 2에서 frozen tx encoder로 사용합니다.

Validation metric 해석:
- `val_history.json`에 기록되는 Stage 1 지표의 의미는 `docs/design/stage1_validation_metrics_kr.md`를 참고합니다.

## Stage 1.5: Spatial Foundation

목적:
- Stage 1 RNA encoder가 만든 spot embedding에 spatial graph context를 추가합니다.
- KNN/radius/grid graph 위에서 masked spot latent를 예측하는 Spatial JEPA를 학습합니다.

핵심 구현:

| 모듈 | 역할 |
|---|---|
| `src/mm_align/models/spatial/encoder.py` | spatial encoder, KNN graph contextualization |
| `src/mm_align/objectives/spatial/jepa.py` | Spatial Predictive JEPA objective |
| `src/mm_align/data/spatial_sampler.py` | sample graph/subgraph loader |

실행 스크립트:

| 스크립트 | 역할 |
|---|---|
| `scripts/train/stage15.sh` | Stage 1.5 wrapper |
| `scripts/train/stage15.py` | Spatial JEPA trainer |

대표 실행:

```bash
bash scripts/train/stage15.sh
```

주요 config:
- `configs/stage15/data.yaml`
- `configs/stage15/model.yaml`
- `configs/stage15/experiment.yaml`
- `configs/stage15/train.yaml`

산출물:
- `results/runs/<tag>/ckpt_spatial_best.pt`

## Stage 2: Image-RNA Alignment

목적:
- Stage 1 RNA encoder를 frozen target으로 두고 pathology image encoder와 RNA latent를 정렬합니다.
- JEPA, CLIP, Barlow, CCA, S2L objective를 비교할 수 있습니다.

핵심 구현:

| 모듈 | 역할 |
|---|---|
| `src/mm_align/models/alignment/aligner.py` | image encoder, tx encoder, projector, decoder를 묶는 top-level model |
| `src/mm_align/objectives/alignment/*.py` | alignment objectives |
| `src/mm_align/objectives/unified.py` | align/reconstruction/tx-self loss 조합 |

실행 스크립트:

| 스크립트 | 역할 |
|---|---|
| `scripts/train/stage2.sh` | frozen Stage 1 checkpoint를 받아 Stage 2 sweep 실행 |
| `scripts/sweep.py` | 여러 experiment config dispatch |

대표 실행:

```bash
bash scripts/train/stage2.sh \
  results/runs/stage1_ours_tx_stage1_feature/ckpt_tx_encoder_best.pt
```

주요 config:

| 파일 | 역할 |
|---|---|
| `configs/sweep/stage2_align.yaml` | Stage 2 sweep entrypoint |
| `configs/experiments/jepa.yaml` | JEPA alignment |
| `configs/experiments/clip.yaml` | CLIP alignment |
| `configs/experiments/barlow.yaml` | Barlow alignment |
| `configs/experiments/cca.yaml` | CCA alignment |
| `configs/experiments/s2l.yaml` | Soft-CLIP/S2L alignment |

산출물:
- `results/runs/<tag>/ckpt_align_best.pt`

## Evaluation

평가는 stage별로 목적을 분리합니다. Stage 1 ablation의 1차 선택 기준은 organ probe가 아니라 `clean_msm/*`, `intrinsic_*`, `linear_probe_hvg_*`, `linear_probe_masked_hvg_*`입니다. `set/*` gene-set 지표는 생물학적 sanity monitor로만 둡니다. Organ probe는 HEST organ label이 쉬운 경우가 많아서 기본값에서는 제외하고, 필요할 때만 `--include-organ-probe`로 확인합니다.

Stage 2에서 image-to-transcriptomics expression prediction을 해야 하므로 value signal 자체는 버리지 않습니다. 다만 Stage 1의 철학이 relative salience/spot representation이면 `mvm_mse`만으로 normalization을 판단하지 않고, `mvm_spearman`, `linear_probe_*_spearman_mean`, `intrinsic_expression_distance_spearman`을 같이 봅니다.

Vocab clip ablation에서 raw `masked_symbol_ce`는 vocab size에 따라 scale이 달라집니다. Random CE baseline은 대략 `log(num_classes)`이므로 full vocab(약 19k classes)은 `~9.86`, clip4096(+specials 4100 classes)은 `~8.32`입니다. 따라서 서로 다른 vocab 크기의 MSM loss를 비교할 때는 `masked_symbol_ce_norm = CE / log(num_classes)`와 top-k/intrinsic/linear-probe 지표를 함께 봅니다.

| Stage | Eval type | 주요 지표/태스크 | 스크립트 |
|---|---|---|---|
| Stage 1 | intrinsic | `masked_symbol_top{k}_acc`, `clean_msm/*`, `intrinsic_effective_rank`, `intrinsic_explained_top10`, `intrinsic_expression_knn_overlap@20`, `intrinsic_expression_distance_spearman`, `intrinsic_gene_embedding_corr_spearman`, gene-set monitor | in-training `scripts/train.py`, `scripts/eval/stage1_tx.py` |
| Stage 1 | linear_probe | frozen `h_tx -> HVG expression`, masked-input `h_tx -> held-out HVG expression` Ridge probe (`linear_probe_hvg_*`, `linear_probe_masked_hvg_*`; Pearson/Spearman/R2/RMSE) | `scripts/eval/stage1_tx.py` |
| Stage 1 | extrinsic | DLPFC layer probe/kNN purity, optional HEST organ probe | `scripts/eval/dlpfc_eval.py`, `scripts/eval/stage1_tx.py --include-organ-probe` |
| Stage 1.5 | intrinsic/extrinsic | spatial JEPA val loss, DLPFC spatial clustering ARI/NMI, layer purity | `scripts/train/stage15.py`, `scripts/eval/dlpfc_eval.py --stage 15` |
| Stage 2 | zero_shot | image-RNA retrieval, alignment/uniformity, modality gap | `scripts/eval/zero_shot.py` |
| Stage 2 | linear_probe/extrinsic | image embedding -> HVG prediction, HEST/SEAL/PathBench/MIL task 확장 | `scripts/eval/linear_probe.py` |
| Stage 2 | slide-level downstream | Loki/HEST/PathBench-style MSI/subtype 등 slide task: mean/max pooling probe + attention MIL | `scripts/eval/slide_mil.py`, `configs/eval/stage2_downstream.yaml` |

주요 스크립트:

| 스크립트 | 역할 |
|---|---|
| `scripts/eval/stage1_tx.py` | Stage 1 tx encoder checkpoint 비교. 기본은 intrinsic + HVG linear probe + source leakage + MVM |
| `scripts/eval/compare_stage1_ablations.sh` | Stage 1 ckpt 비교 wrapper. `INCLUDE_ORGAN=1`일 때만 organ probe 실행 |
| `scripts/eval/dlpfc_eval.py` | Stage 1/1.5 external DLPFC 평가 |
| `scripts/eval/zero_shot.py` | Stage 2 image-RNA retrieval, zero-shot alignment 평가 |
| `scripts/eval/linear_probe.py` | Stage 2 frozen embedding linear probe |
| `scripts/data/write_stage_splits.py` | 전역 `splits.json`에서 stage별 split manifest 생성 |
| `scripts/eval/vocab_qc.py` | vocab 품질 점검 |
| `scripts/eval/validate_vocab.py` | prepared vocab consistency 확인 |
| `scripts/eval/process_quality_qc.py` | sample processing QC |

대표 실행:

```bash
PYTHONPATH=src python scripts/eval/stage1_tx.py \
  --prepared-dir results/cache/prepared_expanded \
  --ckpts results/runs/stage1_obj_*/ckpt_tx_encoder_best.pt

bash scripts/eval/compare_stage1_ablations.sh stage1_norm
INCLUDE_ORGAN=1 bash scripts/eval/compare_stage1_ablations.sh stage1_norm

python scripts/data/write_stage_splits.py \
  --prepared-dir results/cache/prepared_expanded
```

## Ablation

Stage 1 ablation은 `scripts/ablation/` 아래에 모여 있습니다.
각 스크립트는 `_common.sh`를 공유하고, variant별 임시 YAML을 만들어 실행합니다.
각 run은 `results/runs/<tag>/ablation_meta.json`에 `ablation_group`, `ablation_variant`, `changed_args`, `speed_overrides`, `train_overrides`를 남깁니다. `scripts/eval/stage1_tx.py`와 `compare_stage1_ablations.sh`는 이 metadata를 CSV 컬럼으로 자동 병합합니다.

Ablation 원칙: 한 그룹에서는 해당 argument만 바꾸고 나머지는 baseline으로 둡니다. 속도용 공통 override는 기본적으로 `ABL_PROFILE=triage`의 `vocab_clip=4096`만 사용합니다. `ABL_MAX_SEQ_LEN`, `ABL_SAMPLING_STRATEGY`는 명시적으로 줄 때만 적용합니다.

Stage 1 학습 종료 시 test split 평가가 자동 실행되며 `results/eval/stage1_test_<tag>.csv`와 `results/runs/<tag>/test_stage1.csv`에 저장됩니다. Test CSV는 최종 보고용이며 ablation 선택 피드백으로 반복 사용하지 않습니다.

DINO-style consistency ablation은 `configs/stage1/experiment.yaml`의 아래 값을 바꿉니다. MSM only는 모두 false/0으로 두고, 기본 MSM+DINO 후보는 `enable_dino_consistency=true`, `dino_weight=0.1`, `dino_loss=cosine`, `koleo_weight=0.05`를 사용합니다. Cosine 경로는 teacher target batch-centering을 적용하고, KoLeo가 collapse 방지 regularizer 역할을 합니다. `dino_loss=sinkhorn`은 teacher view에 Sinkhorn-Knopp centering을 3 iteration 적용하고 student view에는 softmax normalization을 적용하지만, 현재는 별도 DINO projection head가 아니라 `h_tx` 위에서 동작하므로 진단/비교 후보로 둡니다. 이 objective는 `enable_masked_jepa`와 독립적이므로 `MSM`, `MSM+DINO`, `MSM+Gene-JEPA`를 분리 비교할 수 있습니다.

```yaml
experiment:
  tx_self:
    masking_obj: symbol
    symbol_weight: 1.0
    value_weight: 0.0
    enable_dino_consistency: true
    dino_weight: 0.1
    dino_loss: cosine
    dino_student_temp: 0.1
    dino_teacher_temp: 0.04
    sinkhorn_iterations: 3
    koleo_weight: 0.05
    enable_masked_jepa: false
    jepa_weight: 0.0
```

Value augmentation은 이제 target/context를 분리합니다. 추천 main profile은 `mixed`입니다. masked gene은 shortcut 방지를 위해 keep/noise/dropout을 섞고, unmasked context gene은 약한 noise만 주어 representation 안정성을 유지합니다.

```yaml
model:
  transcriptomics:
    top_hvg_gene:
      masked_value_aug:
        mode: mixed
        keep_p: 0.75
        noise_p: 0.15
        drop_p: 0.10
        noise_std: 0.35
      unmasked_value_aug:
        mode: mixed
        keep_p: 0.90
        noise_p: 0.10
        drop_p: 0.00
        noise_std: 0.15
```

| 스크립트 | 바꾸는 것 |
|---|---|
| `scripts/ablation/run_objective.sh` | MSM, MVM, MSM+JEPA 등 foundation objective |
| `scripts/ablation/run_mask_ratio.sh` | MSM mask ratio |
| `scripts/ablation/run_value_aug.sh` | masked token value augmentation |
| `scripts/ablation/run_jepa.sh` | Gene-level JEPA on/off와 weight |
| `scripts/ablation/run_vocab_clip.sh` | vocab size runtime clipping |
| `scripts/ablation/run_normalize.sh` | gene normalization mode |
| `scripts/ablation/run_seq_sampling.sh` | long sequence token sampling |
| `scripts/ablation/run_sources.sh` | training source mix |

대표 실행:

```bash
# 추천 full grid 실행: default에서 한 축씩만 변경
# default = global_median + mask0.15 + msm_only + spatula_mid + vocab4096 + seq_random_512 + mixed
bash scripts/ablation/run_all.sh
# 기본적으로 각 Stage1 후보를 Stage1.5까지 이어서 실행합니다.
# Stage1만 보고 싶으면 ABL_RUN_STAGE15=0을 붙입니다.

# default만 실행
bash scripts/ablation/run_all.sh default

# vocab_clip ablation
bash scripts/ablation/run_all.sh vocab_4096 vocab_8192 vocab_full

# capacity ablation: batch/rank는 capacity별 자동 선택
# seq_random_512 기준 spatula_lite=768, spatula_mid=512, spatula_large=384
bash scripts/ablation/run_all.sh cap_lite cap_mid cap_large

# objective ablation: msm_only vs msm_multi_chunk(spot-state JEPA)
bash scripts/ablation/run_all.sh obj_msm obj_msm_multi_chunk

# seq length ablation: sampling은 random으로 고정
bash scripts/ablation/run_all.sh seq_random_256 seq_random

# value augmentation ablation
bash scripts/ablation/run_all.sh va_keep va_mixed

# 개별 triage ablation
ABL_PROFILE=triage bash scripts/ablation/run_objective.sh
ABL_PROFILE=triage bash scripts/ablation/run_value_aug.sh
ABL_PROFILE=triage bash scripts/ablation/run_jepa.sh

# 100 epoch main run에서 validation은 매 epoch, expensive quick eval은 10 epoch마다 실행
python scripts/train.py \
  --experiment configs/stage1/experiment_main.yaml \
  --model configs/stage1/model_main.yaml \
  --data configs/stage1/data.yaml \
  --train configs/stage1/train.yaml \
  --stage1-only \
  --epochs 100 \
  --val-every-epoch 10 \
  --stage1-quick-every 10 \
  --stage1-clean-msm-every 10
```

자세한 내용:
- `scripts/ablation/README.md`
- `docs/design/training_and_ablation.md`

## 개발자가 먼저 봐야 할 파일

| 파일 | 이유 |
|---|---|
| `README.md` | 영어 quickstart와 전체 layout |
| `PROJECT_GUIDE_KR.md` | 한글 stage/script 안내 |
| `docs/strategy/roadmap.md` | canonical pipeline |
| `configs/stage1/README.md` | Stage 1 baseline vs main-candidate 차이 |
| `docs/design/stage2_downstream_and_ablation.md` | Stage 2 downstream benchmark와 encoder/method ablation 설계 |
| `docs/design/vocab.md` | vocab과 normalization 설계 |
| `scripts/train.py` | 공통 training entrypoint |
| `src/mm_align/models/__init__.py` | model public API |
| `src/mm_align/objectives/__init__.py` | objective public API |

## 현재 리팩토링 원칙

- repo root는 Python package가 아닙니다.
- Python package namespace는 `mm_align`입니다.
- 도메인별 구현은 `src/mm_align/models/{tx,image,spatial,alignment}`와
  `src/mm_align/objectives/{tx,spatial,alignment}`에 둡니다.
- 오래된 구현은 `references/legacy_code/`(외부에서 가져온 코드) 또는
  `configs/_archive/`(과거 실험 config)로 격리합니다.
- 새 코드의 주석은 영어로 작성하고, "무엇"보다 "왜"를 설명합니다.
  함수·클래스 이름은 역할이 드러나게 둡니다.

## 더 깊이 보려면

| 영역 | 진입점 |
|---|---|
| 전체 layout | `README.md` |
| 4-stage roadmap | `docs/strategy/roadmap.md` |
| 리팩토링 inventory | `docs/refactor_inventory.md` |
| Stage 1 baseline vs main-candidate | `configs/stage1/README.md` |
| Vocab 디자인 | `docs/design/vocab.md` |
| Stage 1.5 디자인 | `docs/design/stage15_spatial_jepa.md` |
| Stage 2 downstream/ablation | `docs/design/stage2_downstream_and_ablation.md` |
| Ablation 설계 | `docs/design/training_and_ablation.md` |


### Stage 2 slide-level downstream 실행 예시

Loki류 최종 목표를 확인하려면 retrieval/linear-probe 뒤에 slide-level task를 붙입니다. label CSV는 `sample_id`와 task label column을 가져야 합니다. 예를 들어 MSI label이 `msi` 컬럼에 있으면 다음처럼 실행합니다.

```bash
python scripts/eval/slide_mil.py \
  --ckpt results/runs/stage2_best/ckpt_last.pt \
  --experiment configs/sweep/stage2_align.yaml \
  --label-csv /path/to/hest_or_pathbench_labels.csv \
  --sample-id-col sample_id \
  --label-col msi \
  --eval-split test
```

출력은 `<run>/slide_mil_<label>_<split>.json/.csv`입니다. `image`, `z_image`, `baseline_uni` arm을 비교해서 Stage 2 alignment가 단순 UNI feature보다 downstream task에 도움이 되는지 봅니다.

### 외부 annotation validation 데이터 준비

GSE176078처럼 spot-level annotation이 있는 ST 데이터나 HER2ST처럼 paired ST/H&E 및 deconvolution annotation이 있는 데이터는 아래 adapter로 mm_align shard로 변환합니다. 출력 shard는 기존 Stage 1/Stage 1.5 loader가 읽는 `/hvg_log`, `/coords`, `/barcode`를 포함하고, annotation은 `/annotation/*`에 함께 저장됩니다.

```bash
# 다운로드/압축해제 후 GSE176078-style annotation 데이터 변환
python scripts/data/prepare_external_validation.py \
  --source gse176078 \
  --raw-dir /data/external/GSE176078 \
  --prepared-dir results/cache/prepared_external/gse176078 \
  --vocab results/cache/prepared_expanded/hvg_vocab.json

# HER2ST: Zenodo archive를 7z password로 먼저 해제한 뒤 변환
python scripts/data/prepare_external_validation.py \
  --source her2st \
  --raw-dir /data/external/her2st \
  --prepared-dir results/cache/prepared_external/her2st \
  --vocab results/cache/prepared_expanded/hvg_vocab.json

# 필요한 public URL/password 확인
python scripts/data/prepare_external_validation.py \
  --source her2st --raw-dir /data/external/her2st \
  --prepared-dir results/cache/prepared_external/her2st \
  --print-download-info
```

HER2ST 원본은 Zenodo에서 제공되며 7z 암호가 필요합니다. count/image archive password는 `zNLXkYk3Q9znUseS`, metadata/spot selection password는 `yUx44SzG6NdB32gY`입니다. 변환 후에는 `splits_<source>_validation.json`과 `<source>_manifest.csv`가 생성됩니다.

변환된 annotation validation set은 Stage 1 checkpoint에 대해 아래처럼 평가합니다. `--field`를 생략하면 `/annotation` 안의 major/cell/type/label/cluster 계열 필드를 자동 선택합니다.

```bash
python scripts/eval/external_annotation_probe.py \
  --prepared-dir results/cache/prepared_external/her2st \
  --ckpts results/runs/stage1_full_default/ckpt_tx_encoder_best.pt \
  --out results/eval/her2st_annotation_probe.csv
```
