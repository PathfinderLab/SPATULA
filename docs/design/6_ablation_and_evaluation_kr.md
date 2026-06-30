# 6. Ablation and Evaluation Plan

작성일: 2026-06-24  
역할: internal ablation 대상, stage별 validation/test evaluation, metric 해석 원칙 정리

## 1. Ablation의 철학

Ablation은 full training 성능만 보는 것이 아니라, 각 설계 가정이 학습에 어떤 영향을 주는지 분리해서 확인하기 위한 실험이다.

원칙:

1. 한 번에 하나의 axis만 바꾼다.
2. default는 가장 철학적으로 일관된 baseline으로 둔다.
3. speed ablation은 빠르게 saturation 경향을 보는 용도다.
4. final comparison은 같은 epoch/batch/eval setting으로 다시 맞춘다.
5. loss만 보지 말고 representation metric과 downstream을 함께 본다.

## 2. 현재 default

```text
norm        = global_median
vocab       = 4096
seq_sampling= random
max_seq_len = 512
mask_ratio  = 0.15
value_aug   = mixed
capacity    = spatula_mid
objective   = msm_only or sequential msm -> chunk_jepa
```

## 3. Internal ablation axes

### 3.1 Vocab

| 후보 | 목적 |
|---|---|
| `4096` | default, 속도/표현력 균형 |
| `8192` | 더 넓은 gene coverage 확인 |
| `full` | vocab clipping이 손해인지 확인 |

주의: vocab size가 다르면 CE scale이 달라지므로 `ce_norm`과 top-k를 함께 본다.

### 3.2 Capacity

| 후보 | 목적 |
|---|---|
| `spatula_lite` | 빠른 baseline, low memory |
| `spatula_mid` | default |
| `spatula_large` | legacy/SPATULA hypothesis: capacity 증가가 MSM saturation을 돕는지 확인 |

### 3.3 Objective

| 후보 | 목적 |
|---|---|
| `msm_only` | 가장 해석 가능한 baseline |
| `msm_multi_chunk` | Stage1에서 intra-spot chunk JEPA를 동시에 학습 |
| `sequential msm -> chunk_jepa` | gene-symbol pretraining과 chunk refinement 분리 |
| `view_jepa` | corrupted view -> clean teacher latent |
| `dino` | consistency regularization 후보 |

현재 우선순위는 sequential이 높다.

### 3.4 Augmentation

| 후보 | 목적 |
|---|---|
| `value_aug=keep` | value shortcut 허용 baseline |
| `value_aug=mixed` | shortcut 방지 default |
| `mask_ratio=0.15` | 안정적 baseline |
| `mask_ratio=0.20` | MSM saturation 가속 후보 |

### 3.5 Sequence / chunk

| 후보 | 목적 |
|---|---|
| `random512` | default |
| `random256` | 속도 개선, chunk 다양성 확인 |
| `top_k512` | high salience gene 중심 비교 |
| target chunks `2` | fixed multi-target |
| target chunks `auto` | I-JEPA-like dynamic multi-target |

### 3.6 Spatial

| 후보 | 목적 |
|---|---|
| `mask_target=spot` | region -> center spot, main |
| `mask_target=region` | inverse ablation |
| `region_agg=mean` | simple smoothing baseline |
| `region_agg=weighted` | distance-aware aggregation |
| `residual target` | future hard spatial objective |
| `edge dropout` | dense neighbor shortcut 방지 |

### 3.7 Multimodal

| 후보 | 목적 |
|---|---|
| image encoder | UNI/UNI2/H0-mini/GigaPath |
| alignment | CLIP/JEPA/Barlow/S2L |
| target | `z_tx`, `z_chunk`, `z_spatial`, expression/rank |
| tuning | frozen/LoRA/partial |

## 4. Stage별 validation metric

### Stage 1 / 1.25

| Category | Metric | 목적 |
|---|---|---|
| MSM | top-k, CE norm | masked symbol 학습 확인 |
| clean MSM | clean top-k/CE | augmentation 영향 분리 |
| linear probe | HVG/masked-HVG Spearman | embedding에 expression signal이 남는지 |
| rank probe | spot-rank Spearman, top10 overlap | relative salience 보존 |
| intrinsic | effective rank, distance Spearman | collapse/geometry 확인 |
| gene embedding | corr Spearman | gene-gene dependency 반영 |
| multi-chunk | context gain, query-only, overlap | JEPA shortcut 여부 |

### Stage 1.5

| Category | Metric | 목적 |
|---|---|---|
| spatial loss | JEPA loss | masked spatial target 예측 |
| smoothing baseline | gain over neighbor mean | trivial smoothing보다 나은지 |
| DLPFC | layer probe, ARI/NMI | spatial domain 분리 |
| gene map | SCC, heatmap | spatial expression pattern 보존 |
| boundary | boundary subset SCC | transition/niche 감지 |

### Stage 2

| Category | Metric | 목적 |
|---|---|---|
| retrieval | Recall@k, MRR | cross-modal alignment |
| image-to-expression | Pearson/Spearman/R2 | molecular predictability |
| image-to-rank | rank Spearman/top-k overlap | relative salience prediction |
| slide MIL | AUROC/accuracy | sample-level downstream |
| visualization | gene heatmap, retrieval panels | qualitative validation |

## 5. Test evaluation 원칙

Validation은 학습 선택용이고, test는 stage별 frozen evaluation이다.

```text
train objective: self-supervised only
validation     : loss + quick probe
hold-out test  : frozen embedding downstream + visualization
```

Test에서는 다음 산출물을 남긴다.

- CSV metrics
- per-sample metrics
- UMAP
- spatial cluster map
- gene GT/pred heatmap
- SCC barplot
- ablation comparison plot

## 6. 실행 순서 권장

현재 권장 ablation 순서:

1. sequential `msm -> chunk_jepa -> spatial_jepa`
2. joint `msm + chunk_jepa -> spatial_jepa`
3. vocab 4096/8192/full
4. capacity lite/mid/large
5. spatial mask/aggregation
6. stage2 target/objective/image encoder

이 순서는 “가장 핵심적인 구조 가정”을 먼저 확인하고, 그 다음 세부 hyperparameter를 보는 방식이다.

## 7. 해석 규칙

### 좋은 신호

- MSM top-k와 clean MSM이 같이 상승
- masked-HVG probe가 유지/상승
- rank probe가 상승
- effective rank가 collapse하지 않음
- multi-chunk context gain이 양수
- spatial encoder가 neighbor mean baseline을 이김
- gene map SCC와 heatmap이 정성적으로 맞음

### 위험 신호

- raw CE만 좋아지고 CE norm/top-k/probe가 개선되지 않음
- top-k는 오르지만 masked-HVG가 떨어짐
- effective rank 급락
- query-only JEPA가 context predictor와 비슷함
- spatial-JEPA가 neighbor mean baseline과 차이가 없음
- image alignment retrieval만 좋고 expression prediction은 나쁨

## 8. 최종 보고 table 구조

최종 report에서는 embedding별로 비교한다.

| Embedding | Stage | Intrinsic | Spot downstream | Spatial downstream | Image alignment |
|---|---|---:|---:|---:|---:|
| `z_spot_tx` | Stage1 | yes | yes | optional | target |
| `z_spot_tx_chunk` | Stage1.25 | yes | yes | optional | target |
| `z_spot_tx_spatial` | Stage1.5 | optional | yes | yes | target |
| `z_patch_img` | PFM | no | image probe | image spatial probe | baseline |
| `z_patch_img_aligned` | Stage2 | no | image-to-RNA | image-to-spatial | main |

이 table이 프로젝트 전체의 성과를 가장 잘 설명한다.
