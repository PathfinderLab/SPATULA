# Stage 1 Validation Metrics Guide

이 문서는 `results/runs/<tag>/val_history.json`에 기록되는 Stage 1 지표를 직관적으로 읽기 위한 안내서입니다.
Stage 1의 목표는 **absolute expression imputation**이 아니라, 이후 Stage 1.5/Stage 2에서 쓸 수 있는 **좋은 spot/RNA representation**을 만드는 것입니다. 따라서 한 지표만 보지 말고, MSM + probe + intrinsic + collapse/leakage 지표를 함께 봐야 합니다.

## 빠른 결론: 먼저 볼 지표

| 우선순위 | 지표 | 직관 | 방향 |
|---:|---|---|---|
| 1 | `val/tx_self/masked_symbol_top10_acc` | 가려진 gene symbol을 top-10 안에 맞추는가 | 높을수록 좋음 |
| 1 | `val/clean_msm/tx_self/masked_symbol_top10_acc` | value augmentation 없이 깨끗한 조건에서도 symbol을 맞추는가 | 높을수록 좋음 |
| 1 | `val/linear_probe/hvg_rank/spot_rank_spearman` | spot 안에서 gene들의 상대적 발현 순서를 보존하는가 | 높을수록 좋음 |
| 1 | `val/linear_probe/masked_hvg/spearman_mean` | 특정 gene 정보를 가린 뒤에도 embedding이 expression을 복원하는가 | 높을수록 좋음 |
| 2 | `val/linear_probe/hvg/spearman_mean` | frozen spot embedding으로 HVG expression rank를 예측할 수 있는가 | 높을수록 좋음 |
| 2 | `val/intrinsic/gene_embedding/corr_spearman` | co-expression이 높은 gene끼리 embedding도 가까운가 | 높을수록 좋음 |
| 2 | `val/intrinsic/effective_rank` | embedding이 한두 방향으로 collapse되지 않았는가 | 너무 낮으면 위험 |
| 보조 | `val/intrinsic/expression/distance_spearman` | spot 간 expression 거리 구조가 embedding 거리에도 남아 있는가 | 높을수록 좋지만 Stage1 핵심은 아님 |

`distance_spearman`은 유용하지만, Stage 1의 MSM 자체가 nearest-neighbor 구조 보존을 직접 학습하는 objective는 아닙니다. spatial neighbor 구조는 Stage 1.5에서 더 중요하게 봐야 합니다.

## 1. 기본 loss 계열

| 지표 | 의미 | 해석 |
|---|---|---|
| `epoch` | validation이 기록된 epoch | x축입니다. |
| `val/loss` | validation 전체 loss | 낮을수록 좋지만, auxiliary loss가 켜지면 run 간 직접 비교가 어려움. |
| `val/loss/total` | total loss alias | `val/loss`와 거의 같은 용도. |
| `val/tx_self/loss` | Stage1 transcriptomics self-supervised loss | MSM, optional DINO/View-JEPA/KoLeo 등이 합쳐진 loss. |
| `val/clean_msm/loss` | clean MSM 평가 loss | value augmentation을 끄고 MSM만 본 보조 loss. |

주의할 점:
- `vocab_clip=4096`과 full vocab은 CE scale이 다릅니다. vocab이 작으면 random CE도 작아지므로 raw CE만 비교하면 안 됩니다.
- vocab이 다른 실험은 `masked_symbol_ce_norm`이나 top-k accuracy를 같이 봐야 합니다.

## 2. MSM: masked symbol modeling

MSM은 “spot 안의 일부 gene symbol을 가리고, 주변 gene/value context로 어떤 gene인지 맞히는 문제”입니다.

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/tx_self/masked_symbol_ce` | masked gene symbol cross entropy | 낮을수록 좋음 | 정답 gene에 확률을 잘 주는가. |
| `val/tx_self/masked_symbol_random_ce` | random guess의 CE 기준선 | 참고값 | 대략 `log(vocab_size)`. |
| `val/tx_self/masked_symbol_ce_norm` | CE / random CE | 낮을수록 좋음 | vocab 크기가 달라도 비교하기 쉬운 normalized CE. 1이면 random 수준. |
| `val/tx_self/masked_symbol_ce_gain` | random 대비 CE 개선량 | 높을수록 좋음 | random보다 얼마나 나아졌는가. |
| `val/tx_self/masked_symbol_top1_acc` | 정답이 top-1인가 | 높을수록 좋음 | 가장 엄격한 symbol accuracy. |
| `val/tx_self/masked_symbol_top5_acc` | 정답이 top-5 안에 있는가 | 높을수록 좋음 | 유사 gene 후보까지 고려. |
| `val/tx_self/masked_symbol_top10_acc` | 정답이 top-10 안에 있는가 | 높을수록 좋음 | Stage1 ablation에서 가장 안정적으로 보기 좋음. |
| `val/tx_self/masked_symbol_acc` | top-1 acc alias | 높을수록 좋음 | `top1_acc`와 같은 의미. |
| `val/tx_self/masked_symbol_vocab_size` | 현재 symbol vocab 크기 | 참고값 | clip4096인지 full vocab인지 확인. |

### `val/tx_self/*` vs `val/clean_msm/*`

| prefix | 의미 |
|---|---|
| `val/tx_self/*` | 실제 validation loss 경로입니다. 학습 설정과 같은 masking/value augmentation이 적용됩니다. |
| `val/clean_msm/*` | value augmentation을 끈 깨끗한 MSM 평가입니다. augmentation 때문에 성능이 낮아 보이는지 분리해서 봅니다. |

직관:
- `tx_self`가 낮고 `clean_msm`이 높으면 augmentation 조건에서는 어렵지만 clean task는 잘하는 것입니다.
- `clean_msm`도 낮으면 symbol context 자체를 잘 못 배우는 것입니다.
- value leakage를 강하게 허용하면 MSM top-k는 올라갈 수 있지만, 좋은 representation이라는 보장은 약합니다.

## 3. Masking / sequence diagnostics

| 지표 | 의미 | 해석 |
|---|---|---|
| `val/tx_self/mask_actual_ratio` | 실제 mask된 token 비율 | 설정한 mask ratio, 예: 0.15 근처인지 확인. |
| `val/tx_self/n_masked_mean` | spot당 평균 masked token 수 | 너무 작으면 MSM signal이 약함. |
| `val/tx_self/seq_len_mean` | spot당 non-zero gene token 평균 길이 | context 길이. 너무 짧으면 문장이 짧은 것과 비슷함. |
| `val/tx_self/seq_len_median` | sequence length 중앙값 | 평균보다 robust. |
| `val/tx_self/seq_len_min` / `max` | 최소/최대 sequence length | 이상치 확인. |
| `val/tx_self/seq_len_p10` / `p90` | sequence length 하위/상위 분위 | 짧은 spot이 너무 많은지 확인. |

Stage1에서는 `mask_ratio=0.15`가 현재 가장 보수적인 기본값입니다. sequence가 너무 짧은 데이터에서는 같은 0.15라도 masked token 수가 부족할 수 있습니다.

## 4. Linear probe: embedding에 expression 정보가 남아 있는가

Linear probe는 encoder를 freeze하고, `h_tx -> expression`을 얕은 Ridge/linear head로 예측하는 평가입니다. 즉 “embedding만 보고 expression을 얼마나 복원할 수 있는가”를 봅니다.

### HVG probe

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/linear_probe/hvg/pearson_mean` | gene별 absolute value 선형 상관 평균 | 높을수록 좋음 | expression scale을 잘 맞추는가. |
| `val/linear_probe/hvg/spearman_mean` | gene별 rank 상관 평균 | 높을수록 좋음 | 값의 순서를 잘 맞추는가. global_median 철학에서는 중요. |
| `val/linear_probe/hvg/r2_mean` | R2 평균 | 높을수록 좋음 | 예측이 평균값 baseline보다 나은가. 음수면 평균보다 못함. |
| `val/linear_probe/hvg/rmse_norm` | normalized RMSE | 낮을수록 좋음 | 예측 오차. |
| `val/linear_probe/hvg/n_targets` | probe 대상 gene 수 | 참고값 | 보통 256. |
| `val/linear_probe/hvg/n_spots` | probe에 사용한 spot 수 | 참고값 | 샘플링 크기 확인. |

### Masked-HVG probe

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/linear_probe/masked_hvg/*` | 일부 target gene 정보를 encoder 입력에서 가린 뒤 같은 probe 수행 | 높을수록 좋음 | target gene value 자체를 본 shortcut이 아니라, 주변 gene context로 expression을 담는지 확인. |

`masked_hvg`가 특히 중요합니다. 그냥 `hvg` probe가 높아도 입력에 target gene value가 직접 들어가 있으면 쉬운 문제가 될 수 있습니다. `masked_hvg`는 그 shortcut을 줄인 평가입니다.

### Rank probe

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/linear_probe/hvg_rank/spot_rank_spearman` | 각 spot 내부에서 gene 발현 순위가 맞는가 | 높을수록 좋음 | “이 spot에서 어떤 gene이 상대적으로 중요한가”를 맞히는 지표. |
| `val/linear_probe/hvg_rank/top10_overlap` | 실제 top-10 high-expression gene과 예측 top-10의 overlap | 높을수록 좋음 | salience gene을 잘 잡는가. |
| `val/linear_probe/hvg_rank/bin_acc` | expression bin 분류 정확도 | 높을수록 좋음 | 매우 낮음/중간/높음 같은 coarse rank를 맞히는가. |
| `val/linear_probe/hvg_rank/n_targets` | rank probe 대상 gene 수 | 참고값 | 보통 256. |

global median normalization과 vocab clip의 의도는 absolute value 복원보다 **relative salience**에 가깝기 때문에, `spot_rank_spearman`과 `top10_overlap`을 중요하게 보는 것이 맞습니다.

## 5. Intrinsic representation metrics

Intrinsic metric은 downstream head 없이 embedding 자체의 geometry를 봅니다.

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/intrinsic/effective_rank` | embedding이 몇 개의 독립 방향을 쓰는지 | 너무 낮으면 위험 | collapse 감지. 모든 spot이 비슷한 벡터가 되면 낮아짐. |
| `val/intrinsic/explained_top10` | 상위 10개 성분이 설명하는 분산 비율 | 너무 높으면 위험 | 소수 차원에만 정보가 몰리면 높아짐. |
| `val/intrinsic/norm_mean` | embedding norm 평균 | 참고값 | 갑자기 폭주/소실하는지 확인. |
| `val/intrinsic/norm_std` | embedding norm 표준편차 | 참고값 | norm 분포가 비정상인지 확인. |
| `val/intrinsic/expression/distance_spearman` | expression 거리와 embedding 거리의 Spearman 상관 | 높을수록 좋음, 보조 | expression이 비슷한 spot이 embedding에서도 비슷한가. |
| `val/intrinsic/expression/knn_overlap@20` | expression kNN과 embedding kNN의 overlap | 높을수록 좋음, 보조 | 이웃 spot 구조가 얼마나 겹치는가. |
| `val/intrinsic/expression/n_spots` | 계산에 사용한 spot 수 | 참고값 | 샘플링 크기. |

중요한 해석:
- `distance_spearman`과 `knn_overlap@20`은 spot-to-spot 구조 보존 지표입니다.
- MSM은 gene symbol prediction objective라서 이 지표를 직접 최적화하지 않습니다.
- spatial neighbor/context 구조는 Stage 1.5에서 더 직접적으로 봐야 합니다.

## 6. Gene embedding correlation

이 지표는 spot embedding이 아니라 **gene symbol embedding table**이 co-expression 구조를 이해하는지 봅니다.

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/intrinsic/gene_embedding/corr_spearman` | 실제 gene-gene co-expression과 gene embedding cosine similarity의 Spearman 상관 | 높을수록 좋음 | 같이 발현되는 gene끼리 embedding도 가까운가. |
| `val/intrinsic/gene_embedding/top_pair_overlap` | 실제 co-expression top pair와 embedding similarity top pair의 overlap | 높을수록 좋음 | 강한 gene-gene 관계를 embedding이 잡는가. |
| `val/intrinsic/gene_embedding/n_genes` | 평가 대상 gene 수 | 참고값 | clip/vocab 상태 확인. |
| `val/intrinsic/gene_embedding/n_pairs` | 평가한 gene pair 수 | 참고값 | 샘플링된 pair 수. |

이 지표는 사용자의 핵심 질문인 “gene-gene correlation이 높은 gene들이 embedding에서도 가까운가?”에 가장 직접적으로 대응합니다.

## 7. Leakage / source bias

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/leakage/source_knn/same_rate@20` | embedding kNN 20개 중 같은 source 비율 | 너무 높으면 위험 | 모델이 biological signal보다 source/batch를 외우는지 확인. |
| `val/leakage/source_knn/entropy@20` | 이웃 source 다양성 entropy | 높을수록 source mixing | source가 다양하게 섞이면 높음. |
| `val/leakage/source_knn/n_spots` | 계산 spot 수 | 참고값 | 샘플링 크기. |

`same_rate@20`이 높다고 항상 나쁜 것은 아닙니다. source별 organ/disease 구성이 다르면 biological 차이일 수도 있습니다. 다만 ablation 비교에서 특정 옵션만 source clustering을 강하게 만들면 주의해야 합니다.

## 8. Curated gene-set monitor

`val/set/<panel>/...` 지표는 endothelial, epithelial, immune, macrophage 같은 curated marker panel을 보조적으로 봅니다.

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/set/<name>/coverage` | 해당 marker set 중 vocab에 들어 있는 비율 | 높을수록 좋음 | 이 cell-type marker를 vocab이 충분히 포함하는가. |
| `val/set/<name>/pcc_mean` | marker set 내부 expression 관계의 평균 correlation monitor | 높을수록 좋음, 보조 | 해당 panel의 expression 패턴을 얼마나 보존하는가. |
| `val/set/<name>/cls_silhouette` | 해당 marker high/low spot이 embedding에서 분리되는가 | 높을수록 좋음, 보조 | marker-high spot이 embedding상 모이는가. |

이 지표들은 **주요 selection metric이 아니라 보조 모니터**로 두는 것이 맞습니다. panel coverage나 dataset composition에 민감합니다.

## 9. Alignment/metric 계열: Stage1에서는 보조 또는 legacy 로그

Stage1-only에서도 공통 training loop 때문에 alignment-style metric이 남을 수 있습니다.

| 지표 | 의미 | Stage1 해석 |
|---|---|---|
| `val/align/loss` | image/RNA alignment loss 계열 | Stage1-only에서는 핵심 아님. |
| `val/align/diag_cos` | paired diagonal cosine | Stage1-only에서는 보조/legacy. |
| `val/align/offdiag_cos` | unpaired off-diagonal cosine | Stage1-only에서는 보조/legacy. |
| `val/align/diag_minus_off` | paired similarity - unpaired similarity | Stage2에서 더 중요. |
| `val/metric/cosine_sim` | generic cosine similarity | Stage1에서는 핵심 아님. |
| `val/metric/gene_tx_mse` | gene prediction MSE 계열 | objective 설정에 따라 참고. |
| `val/metric/gene_tx_pcc` | gene prediction Pearson | 참고. |
| `val/metric/gene_tx_spearman` | gene prediction Spearman | 참고. |

Stage1 run을 판단할 때는 위 alignment 계열보다 `tx_self`, `linear_probe`, `intrinsic`, `gene_embedding`을 우선합니다.

## 10. DINO / View-JEPA / KoLeo auxiliary metrics

DINO나 View-JEPA를 켠 run에서만 나타납니다.

| 지표 | 의미 | 방향 | 직관 |
|---|---|---|---|
| `val/tx_self/dino_weight_effective` | warmup/ramp 후 실제 적용 중인 DINO loss weight | 참고값 | 0이면 아직 DINO가 꺼져 있음. |
| `val/tx_self/dino_warmup_scale` | DINO warmup scale | 참고값 | 0 -> 1로 올라감. |
| `val/tx_self/dino_consistency` | student h_tx와 teacher h_tx 간 consistency loss | 낮을수록 좋음 | corrupted view가 clean teacher view를 따라가는가. |
| `val/tx_self/dino_cosine_distance` | student/teacher cosine distance | 낮을수록 좋음 | 0에 가까울수록 비슷함. |
| `val/tx_self/view_jepa_weight_effective` | View-JEPA 실제 weight | 참고값 | warmup 후 켜지는지 확인. |
| `val/tx_self/view_jepa_warmup_scale` | View-JEPA warmup scale | 참고값 | 0 -> 1로 올라감. |
| `val/tx_self/view_jepa` | predictor가 clean teacher h_tx를 맞히는 loss | 낮을수록 좋음 | data2vec/JEPA식 latent prediction 품질. |
| `val/tx_self/view_jepa_cosine_distance` | View-JEPA prediction과 target의 cosine distance | 낮을수록 좋음 | prediction 방향이 target과 가까운가. |
| `val/tx_self/koleo_weight_effective` | KoLeo 실제 weight | 참고값 | 0이면 KoLeo 미적용. |
| `val/tx_self/koleo_warmup_scale` | KoLeo warmup scale | 참고값 | 0 -> 1로 올라감. |
| `val/tx_self/koleo` | nearest-neighbor entropy regularizer loss | 보통 낮을수록 loss 관점 좋음 | embedding collapse 방지용. 단 너무 강하면 biological neighbor 구조를 흐릴 수 있음. |

DINO/KoLeo를 켠 뒤 `effective_rank`가 급락하거나 `distance_spearman`, `linear_probe`가 무너지면 auxiliary가 representation을 방해하는 신호입니다. 이 경우 View-JEPA predictor 방식이나 더 긴 warmup을 우선 검토합니다.

## 11. 어떤 run을 좋은 run으로 볼까?

Stage1에서 좋은 run은 보통 아래 조건을 만족합니다.

1. `masked_symbol_top10_acc`와 `clean_msm/top10_acc`가 안정적으로 상승한다.
2. `masked_symbol_ce_norm`이 내려간다. vocab 크기가 달라도 비교 가능하다.
3. `linear_probe/hvg_rank/spot_rank_spearman`이 높다.
4. `linear_probe/masked_hvg/spearman_mean`이 유지되거나 오른다.
5. `gene_embedding/corr_spearman`이 너무 낮지 않다.
6. `effective_rank`가 collapse하지 않는다.
7. `source_knn/same_rate@20`이 과도하게 올라가지 않는다.

반대로 다음은 주의 신호입니다.

- MSM top-k는 오르는데 `masked_hvg` probe가 떨어짐: value shortcut 또는 symbol task 과적합 가능성.
- `effective_rank` 급락: embedding collapse 가능성.
- DINO/KoLeo weight가 켜진 직후 `distance_spearman`과 probe가 동시에 급락: auxiliary가 너무 강하거나 warmup이 짧음.
- raw CE만 좋아짐: vocab size 차이 때문일 수 있으므로 `ce_norm`과 top-k를 같이 확인해야 함.

## 12. 추천 dashboard 순서

학습 중에는 아래 순서로 보면 됩니다.

1. **MSM panel**: `masked_symbol_top10_acc`, `clean_msm/top10_acc`, `masked_symbol_ce_norm`
2. **Probe panel**: `hvg/spearman_mean`, `masked_hvg/spearman_mean`, `hvg_rank/spot_rank_spearman`
3. **Intrinsic panel**: `effective_rank`, `explained_top10`, `gene_embedding/corr_spearman`
4. **Aux panel**: `dino_weight_effective`, `view_jepa_weight_effective`, `koleo_weight_effective`
5. **Leakage panel**: `source_knn/same_rate@20`, `source_knn/entropy@20`

현재 자동 생성되는 `metric_curves.png`, `stage1_metric_curves.png`가 이 순서와 거의 맞도록 구성되어 있습니다.
