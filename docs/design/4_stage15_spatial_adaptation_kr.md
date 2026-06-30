# 4. Stage 1.5 Spatial Adaptation

작성일: 2026-06-24  
역할: spatial-JEPA, spot/region token, sample-local context adaptation, trivial smoothing 위험과 ablation 정리

## 1. Stage 1.5의 역할

Stage 1.5는 Stage 1/1.25에서 얻은 sample-agnostic spot representation을 spatial graph 위에서 contextualize한다.

```text
input : z_spot_tx or z_spot_tx_chunk + neighbor graph + coordinates
output: z_spot_tx_spatial
```

이 단계는 pure pretraining이라기보다 **spatial self-supervised adaptation**에 가깝다. 이유는 sample-local neighbor graph와 coordinate를 사용하기 때문이다.

## 2. HyperST / ST-JEPA / STFormer와의 연결

| Reference 관점 | 프로젝트 해석 |
|---|---|
| HyperST spot/niche | `anchor spot + neighbors = region` |
| ST-JEPA masked neighborhood prediction | spatial graph에서 visible context로 masked spot latent 예측 |
| STFormer spatial coding | coordinate/neighbor-aware spatial representation |
| I-JEPA block prediction | graph block/neighbor context -> target spot/region latent |

즉 Stage 1.5는 다음 질문을 푼다.

> 이 spot은 자기 gene expression만 보면 어떤 상태인가? 그리고 주변 tissue context까지 보면 어떤 의미를 갖는가?

## 3. 기본 데이터 흐름

```text
Stage1 frozen tx_encoder
  anchor spot hvg -> h_tx_anchor
  neighbor hvg aggregate -> h_region_tx
  image patch / neighbor image -> h_img, h_region_img
  coordinates / graph edges -> pos / edge features

SpatialEncoder
  input: spot token + region token + graph
  output: z_spatial
```

Separate token mode에서는 다음 구조가 명확하다.

```text
spot token   = anchor spot state
region token = neighbor/region context
```

## 4. Spatial-JEPA objective

현재 main candidate는 `region -> spot`이다.

```text
student: masked target spot token + visible region context
teacher: clean graph
loss = SmoothL1(z_student[target], stopgrad(z_teacher[target]))
```

간단히 쓰면:

```text
z_i^T = Teacher(clean graph)_i
z_i^S = Student(masked graph)_i
L_spatial = mean_{i in M} SmoothL1(z_i^S, stopgrad(z_i^T))
```

## 5. 중요한 우려: trivial spatial smoothing

Stage1.5는 neighbor spot과 center spot이 gene list/expression이 비슷하기 때문에 trivial해질 수 있다.

위험한 shortcut:

1. neighbor average만으로 center를 맞춘다.
2. sample-specific expression prior를 외운다.
3. coordinate smoothness만 학습한다.
4. tissue boundary나 rare niche를 못 배운다.

따라서 Stage1.5를 “spatial context learning”이라고 주장하려면 smoothing baseline을 넘어야 한다.

## 6. 필수 diagnostic baseline

| Baseline | 의미 | 왜 필요한가 |
|---|---|---|
| neighbor mean | `z_center ≈ mean(z_neighbors)` | 단순 smoothing보다 나은지 확인 |
| coordinate-only | `(x,y)` 또는 graph distance만 사용 | spatial layout prior만 쓰는지 확인 |
| random-neighbor | 같은 sample random spots | true locality가 필요한지 확인 |
| cross-sample random | 다른 sample/같은 organ spots | sample-specific prior인지 확인 |
| region-only | anchor 없이 region token만 | target leakage 여부 확인 |

핵심 metric:

```text
gain_over_neighbor_mean = L_neighbor_mean - L_spatial_jepa
```

`gain_over_neighbor_mean > 0`이면 learned spatial encoder가 단순 neighbor smoothing보다 낫다는 뜻이다.

## 7. 덜 trivial한 objective 후보

### 7.1 Residual target

neighbor 평균으로 설명되는 부분을 빼고 center-specific deviation을 예측한다.

```text
r_i = z_i - mean_{j in N(i)} z_j
L = SmoothL1(pred_i, stopgrad(r_i))
```

또는 expression level:

```text
r_i,g = x_i,g - mean_{j in N(i)} x_j,g
```

이 objective는 boundary, local deviation, niche-specific signal에 민감하다.

### 7.2 Boundary-aware target sampling

다음 spot을 더 자주 mask한다.

- neighbor와 expression distance가 큰 spot
- spatial gradient가 큰 marker gene spot
- cluster boundary spot
- tumor-stroma interface 후보

### 7.3 Multi-hop context

1-hop neighbor만 쓰면 너무 쉽다. context와 target 거리를 조절한다.

```text
context = outer ring / 2-hop neighbors
target  = center or inner ring
```

### 7.4 Edge dropout / neighbor dropout

dense neighborhood에 과의존하지 않게 한다.

```text
E' = dropout_edges(E, p)
N'_i = dropout_neighbors(N_i, p)
```

## 8. Ablation 대상

| Axis | 후보 | 질문 |
|---|---|---|
| mask target | `spot`, `region`, `both` | region->spot이 좋은가, inverse가 좋은가? |
| region token mode | `separate`, `fused` | JEPA contract가 명확한 구조가 이득인가? |
| region aggregation | `mean`, `weighted`, `attention/subsampled` | mean smoothing shortcut을 줄일 수 있는가? |
| graph | `ego`, `random`, `radius`, `knn` | 진짜 tissue locality가 중요한가? |
| mask strategy | random/block | block mask가 spatial reasoning을 강제하는가? |
| residual target | off/on | smoothing보다 boundary/deviation을 배우는가? |
| edge dropout | 0/0.1/0.2 | dense neighbor shortcut을 줄이는가? |
| image context | off/on | morphology context가 spatial RNA target에 기여하는가? |

## 9. Evaluation

Stage1.5는 loss만 보면 안 된다. 반드시 downstream/visualization을 본다.

| Eval | 의미 |
|---|---|
| DLPFC layer linear probe | spatial domain/layer를 잘 구분하는가 |
| zero-shot clustering ARI/NMI | label 없이 spatial domain이 분리되는가 |
| gene map SCC | 특정 gene spatial high/low pattern이 맞는가 |
| GT vs predicted heatmap | 정성적 spatial pattern 확인 |
| boundary subset SCC | boundary/transition에서 성능이 유지되는가 |
| neighbor mean 대비 gain | 단순 smoothing보다 나은가 |

## 10. 현재 유지할 main path

현재는 complexity를 막기 위해 아래 main path를 유지한다.

```text
Stage1/1.25 encoder frozen
+ separate spot/region token
+ mask_target=spot
+ region -> masked spot latent
+ diagnostic baseline 추가
```

즉 지금 당장 Stage1.5를 버리기보다, **trivial smoothing을 확인할 수 있는 baseline과 metric을 추가하는 방식**이 가장 안전하다.
