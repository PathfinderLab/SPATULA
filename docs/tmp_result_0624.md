# Temporary Result Report - 2026-06-24

프로젝트: `/workspace/mm_align`  
목적: Stage1/Stage1.25 기반 spot encoder의 현재 성능과 해석 정리

## 1. 현재 평가 대상

현재 확인한 주요 run은 다음과 같다.

| 구분 | Run | 의미 |
|---|---|---|
| Stage1 | `stage1_pipe_msm_v4096_spatula_mid__pipeline-sequential__stage125_mc_weight-0p10__stage125_target_chunks-2` | MSM-only로 학습한 기본 tx/spot encoder |
| Stage1.25 | `stage125_pipe_chunk_jepa_spatula_mid` | Stage1 checkpoint를 init으로 사용한 chunk-to-chunk JEPA refinement 초기 run |
| DLPFC eval | `results/eval/full_stage1_msm_spatula_mid/dlpfc_viz/` | `spot_state` 기반 gene-map SCC 및 DLPFC qualitative evaluation |

Stage1 기본 설정은 `global_median + vocab_clip4096 + random512 + mask_ratio=0.15 + spatula_mid + mixed value augmentation`이다. 현재 프로젝트의 핵심 가정인 “full vocab coverage보다 robust spot representation이 중요하다”는 방향에 맞춰져 있다.

## 2. Stage1 Test 성능 요약

출처: `results/runs/stage1_pipe_msm_v4096_spatula_mid__pipeline-sequential__stage125_mc_weight-0p10__stage125_target_chunks-2/test_stage1.csv`

| Metric | Value | 해석 |
|---|---:|---|
| `intrinsic_effective_rank` | 31.73 | collapse 없이 일정한 표현 차원을 사용 중 |
| `intrinsic_expression_knn_overlap@20` | 0.5269 | expression neighbor 구조를 절반 이상 보존 |
| `intrinsic_expression_distance_spearman` | 0.5871 | embedding 거리와 expression 거리의 상관이 양호 |
| `intrinsic_gene_embedding_corr_spearman` | 0.1648 | gene embedding이 co-expression 구조를 일부 반영 |
| `linear_probe_hvg_pearson_mean` | 0.7433 | frozen embedding으로 HVG value를 강하게 예측 가능 |
| `linear_probe_hvg_spearman_mean` | 0.4417 | gene expression rank/order signal도 보존 |
| `linear_probe_hvg_r2_mean` | 0.5599 | linear probe 기준 expression 설명력이 양호 |
| `linear_probe_hvg_rank_spot_rank_spearman` | 0.5414 | spot 내부 gene relative salience 복원 신호가 좋음 |
| `linear_probe_hvg_rank_top10_overlap` | 0.5406 | top salient gene set의 절반 이상을 복원 |
| `linear_probe_masked_hvg_pearson_mean` | 0.7334 | 일부 gene이 가려져도 expression 예측력이 유지됨 |
| `linear_probe_masked_hvg_spearman_mean` | 0.4341 | masked setting에서도 rank signal 유지 |
| `mvm_pearson` | 0.1327 | direct value-head imputation은 약함 |
| `mvm_spearman` | 0.0901 | direct masked value reconstruction은 주된 강점이 아님 |
| `mvm_r2` | -2.0508 | absolute value prediction objective와는 불일치 |

### 해석

Stage1은 direct MVM value reconstruction보다는 **frozen spot embedding을 통한 downstream expression probe**에서 강한 신호를 보인다. 이는 현재 Stage1을 “gene expression imputation model”이 아니라 **spot representation foundation**으로 보는 해석과 잘 맞는다.

특히 `linear_probe_hvg_rank_spot_rank_spearman = 0.5414`와 `top10_overlap = 0.5406`은 global_median normalization의 목적, 즉 absolute scale보다 **relative gene salience**를 학습하려는 방향과 일관된다.

## 3. h_tx / z_chunk / z_spot 비교

Stage1 test에서 chunk/spot-state representation도 함께 평가되었다.

| Representation | HVG Spearman | HVG Rank Spearman | 해석 |
|---|---:|---:|---|
| `h_tx` | 0.4417 | 0.5414 | 기본 clean/sampled tx encoder representation |
| `chunk_state` | 0.4143 | 0.5080 | partial gene chunk만으로도 상당한 spot signal 유지 |
| `spot_state` | 0.4322 | 0.5287 | multi-chunk aggregate가 chunk보다 spot-level task에 더 적합 |

`z_chunk`는 partial view이므로 `h_tx`보다 낮은 것이 자연스럽다. 중요한 점은 `spot_state`가 `chunk_state`보다 rank/linear probe에서 개선된다는 점이다. 이는 multi-chunk aggregation이 단순한 noise가 아니라 spot-level representation 복원에 기여할 가능성을 보여준다.

다만 현재 Stage1 test의 `spot_state`는 Stage1.25 JEPA refinement 이후가 아니라 Stage1 encoder에서 chunk view를 추출/aggregate한 결과로 보는 것이 안전하다. Stage1.25가 충분히 학습된 후에는 같은 evaluation으로 다시 비교해야 한다.

## 4. Stage1.25 Chunk-JEPA 현재 상태

출처: `results/runs/stage125_pipe_chunk_jepa_spatula_mid/val_history.json`, `sequential_meta.json`

Stage1.25 설정:

| 항목 | 값 |
|---|---|
| objective | `multi_chunk_jepa_only` |
| init checkpoint | Stage1 MSM checkpoint |
| capacity | `spatula_mid` |
| batch/rank | 384 |
| multi_chunk_weight | 0.1 |
| target chunks | 2 |
| context scale | 0.45 - 0.65 |
| target scale | 0.15 - 0.25 |
| target id scale | 0.25 |

현재 `val_history.json`에는 2 epoch만 기록되어 있어, 아직 학습 완료 성능으로 해석하기에는 이르다.

| Metric | Epoch 1 | Epoch 2 | 해석 |
|---|---:|---:|---|
| `val/tx_self/multi_chunk_jepa` | 0.0987 | 0.1044 | 아직 안정화 전, 큰 개선/악화 판단 불가 |
| `val/tx_self/multi_chunk_jepa_weighted` | 0.0020 | 0.0042 | warmup scale 증가 영향 |
| `val/tx_self/multi_chunk_weight_effective` | 0.0200 | 0.0400 | warmup 중 |
| `val/tx_self/multi_chunk_context_target_cosine_distance` | 1.2734 | 1.2090 | context-target latent가 어느 정도 가까워지는 방향 |
| `val/tx_self/multi_chunk_query_only_smoothl1` | 0.3907 | 0.4787 | query-only shortcut만으로 target 설명 어려움 |
| `val/tx_self/multi_chunk_context_gain_over_query_only` | 0.2920 | 0.3743 | context 정보 사용 신호가 존재 |
| `val/tx_self/multi_chunk_ctx_tgt_overlap` | 0.0 | 0.0 | context/target gene overlap 없음, disjoint contract 유지 |
| `val/tx_self/multi_chunk_single_chunk_frac` | 0.0 | 0.0 | single chunk degenerate case 없음 |

### 해석

Stage1.25는 아직 초기 단계이지만, JEPA objective 자체는 trivial하게 collapse한 모습은 아니다. 특히 `query_only_smoothl1`보다 context를 사용한 predictor가 더 나은 `context_gain_over_query_only`를 보이고, `ctx_tgt_overlap=0`으로 shortcut 가능성이 제한되어 있다.

다만 기록이 2 epoch뿐이므로 최종 판단은 불가능하다. 최소 warmup/ramp가 끝난 이후의 `multi_chunk_jepa`, `context_gain_over_query_only`, `effective_rank`, 그리고 downstream `z_spot` probe를 함께 봐야 한다.

## 5. DLPFC Gene-map 평가

출처: `results/eval/full_stage1_msm_spatula_mid/dlpfc_viz/gene_map_scc_all.csv`

`spot_state`에 대해 DLPFC leave-one-sample style gene-map probe를 수행했다. 전체 66개 sample/gene 조합에서:

| Summary | Spearman SCC |
|---|---:|
| mean | 0.3912 |
| median | 0.3600 |
| min | 0.1119 |
| max | 0.7410 |

Gene별 평균 SCC:

| Gene | Mean SCC | 해석 |
|---|---:|---|
| MBP | 0.5949 | myelin/white-matter signal이 잘 복원됨 |
| SNAP25 | 0.5161 | neuronal marker spatial pattern이 비교적 잘 보존 |
| GFAP | 0.4903 | astrocyte/glial spatial signal이 양호 |
| MOBP | 0.2860 | 중간 수준 |
| PCP4 | 0.2491 | layer-related marker이나 예측 난도가 있음 |
| CARTPT | 0.2106 | 가장 약함 |

관련 figure:

- `results/eval/full_stage1_msm_spatula_mid/dlpfc_viz/.../spot_state/gene_map_scc_barplot.png`
- `results/eval/full_stage1_msm_spatula_mid/dlpfc_viz/.../spot_state/method_summary.png`
- `results/eval/full_stage1_msm_spatula_mid/dlpfc_viz/.../h_tx/method_summary.png`
- `results/eval/full_stage1_msm_spatula_mid/dlpfc_viz/.../chunk_state/method_summary.png`

### 해석

DLPFC gene-map 결과는 현재 spot representation이 단순 masked-symbol accuracy만 학습한 것이 아니라, 실제 spatial tissue에서 marker gene의 공간적 패턴을 어느 정도 보존한다는 근거가 된다. 특히 MBP, SNAP25, GFAP처럼 공간적/세포형 특성이 강한 gene에서 SCC가 높다.

다만 현재는 Stage1 `spot_state` 중심 평가이며, Stage1.5 spatial JEPA 이후 `z_spot_tx_spatial`과 직접 비교해야 진짜 spatial context의 이득을 판단할 수 있다.

## 6. Vocab / Normalization QC 관련 관찰

최근 추가한 QC 결과는 다음 경로에 있다.

- `results/eval/vocab_quality.csv`
- `results/eval/normalization_quality.csv`
- `results/eval/normalization_rank_shift.csv`
- `results/figures/vocab_norm_qc/`

핵심 관찰:

| 항목 | 관찰 |
|---|---|
| vocab4096 marker coverage | 152/154 curated markers retained |
| vocab4096 protein-coding ratio | 97.3% |
| global_median vs none distribution | histogram은 유사하지만 rank-level 변화는 큼 |
| vocab4096 global_median rank Spearman vs none | 0.4513 |
| vocab4096 global_median top50 overlap vs none | 0.4858 |
| vocab4096 global_median top10 overlap vs none | 0.2465 |

즉 `global_median`은 값 분포 그림만 보면 `none`과 비슷해 보이지만, spot 내부 gene rank를 상당히 재정렬한다. 이는 “raw abundance보다 gene-wise relative salience를 학습한다”는 가정과 부합한다.

## 7. 현재까지의 긍정적 결론

1. Stage1 MSM encoder는 absolute MVM value reconstruction보다는 **frozen embedding downstream probe**에서 강하다.
2. `global_median + vocab4096` 조합은 marker coverage를 거의 유지하면서 정보량 높은 gene 중심으로 학습하게 해준다.
3. `h_tx`, `chunk_state`, `spot_state`를 분리한 evaluation이 가능해졌고, `spot_state`가 chunk보다 spot-level task에 더 적합한 신호를 보인다.
4. DLPFC gene-map SCC에서 MBP/SNAP25/GFAP 등 biologically meaningful marker의 spatial pattern이 비교적 잘 복원된다.
5. Stage1.25 chunk-JEPA는 아직 초기지만, `query_only` shortcut보다 context 사용 이득이 관찰되어 objective가 완전히 trivial하지는 않아 보인다.

## 8. 주의점 / 한계

1. Stage1.25는 현재 2 epoch 기록만 있어 성능 판단에는 부족하다.
2. Stage1의 `mvm_r2`는 낮다. 하지만 이는 현재 모델 목적이 direct imputation이 아니라 representation learning이라는 점을 고려해 해석해야 한다.
3. source probe accuracy와 source kNN leakage가 높다. 데이터 source/domain signal이 embedding에 남아 있을 수 있으므로, 추후 source-balanced sampling 또는 domain leakage control을 확인해야 한다.
4. DLPFC gene-map은 Stage1 spot_state 기준이며, Stage1.5 spatial encoder 이후와 비교해야 한다.
5. spatial domain clustering metric은 아직 SDMBench-style metric까지 완전히 확장된 상태는 아니다.

## 9. 다음 액션

우선순위는 다음과 같다.

1. Stage1.25를 warmup/ramp 이후까지 충분히 학습하고, 같은 Stage1 test + DLPFC eval을 다시 수행한다.
2. Stage1 `h_tx`, Stage1 `spot_state`, Stage1.25 refined `spot_state`, Stage1.5 `z_spot_tx_spatial`을 같은 DLPFC/gene-map task에서 비교한다.
3. DLPFC spatial metric에 `CHAOS`, `PAS`, `Moran's I`, `Geary's C`, `ASW`를 추가한다.
4. gene-map metric에 Spearman 외 Pearson, RMSE, SSIM, JSD를 함께 기록한다.
5. source leakage를 줄이는 ablation 또는 source-stratified metric을 추가한다.

## 10. 한 줄 요약

현재 모델은 **MSM 기반 Stage1 spot encoder로서 relative gene salience와 downstream expression probe에서는 강한 신호를 보이고 있으며**, DLPFC gene-map에서도 biologically meaningful spatial marker pattern을 일부 복원한다. Stage1.25 chunk-JEPA는 objective shortcut 방지 신호는 긍정적이지만, 아직 초기 run이므로 충분한 학습 후 downstream 재평가가 필요하다.
