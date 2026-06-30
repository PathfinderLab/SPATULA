# Spot Encoder 설계: Stage 1 + Stage 1.5

작성일: 2026-06-23  
프로젝트: `/workspace/mm_align`  
관련 stage: Stage 1 RNA/Spot Foundation, Stage 1.5 Spatial Foundation

## 1. 핵심 요약

이 프로젝트에서 최종적으로 만들고 싶은 것은 단순한 transcriptomics encoder 하나가 아니라, **spot 자체의 gene-state와 주변 spatial context를 함께 이해하는 spot_encoder**이다.

이를 위해 spot_encoder는 두 단계로 구성된다.

| 단계 | 역할 | 학습하는 표현 | 직관 |
|---|---|---|---|
| Stage 1 | RNA/spot foundation | `h_tx`, `z_chunk`, `z_spot` | 한 spot 내부의 expressed gene set, relative expression salience, gene dependency를 이해한다. |
| Stage 1.5 | Spatial foundation | `z_spatial` | Stage 1 spot representation을 고정한 뒤, 주변 spot/region context를 통해 spatially informed spot embedding을 만든다. |

즉 최종 spot_encoder는 다음 계층 구조로 이해할 수 있다.

```text
raw spot expression
  -> Stage 1 tx_encoder
       -> h_tx: full spot-level transcriptomic representation
       -> z_chunk: sampled gene chunk representation
       -> z_spot: multi-chunk aggregated spot-state representation
  -> Stage 1.5 spatial_encoder
       -> z_spatial: spot + neighbor/region context-aware representation
```

Stage 1은 **spot을 하나의 문장처럼 보고 gene token들의 관계를 학습**한다. Stage 1.5는 그 spot을 tissue graph 위의 한 node로 보고, **neighbor/region context를 통해 spatially meaningful한 spot state로 확장**한다.

## 2. 왜 Stage 1과 Stage 1.5를 분리하는가?

처음부터 spatial context를 모두 넣어 학습할 수도 있지만, 현재 프로젝트의 논리는 두 단계를 분리하는 쪽이 더 안정적이다.

1. Stage 1은 gene-symbol semantics와 intra-spot dependency를 먼저 안정화한다.
2. Stage 1.5는 안정화된 spot representation을 입력으로 받아 spatial context를 학습한다.
3. Stage 2는 Stage 1.5 또는 Stage 1 target을 pathology image와 alignment한다.

이 분리는 다음 장점이 있다.

- **학습 난이도 분리**: gene vocabulary modeling과 spatial graph modeling을 동시에 풀지 않는다.
- **평가 명확성**: Stage 1은 gene/rank/spot representation 평가, Stage 1.5는 spatial clustering/gene map/spatial downstream 평가로 나눌 수 있다.
- **모듈 교체 가능성**: Stage 1 encoder, spatial encoder, image encoder를 각각 ablation할 수 있다.
- **HyperST/J(E)PA 논리와 정합성**: HyperST의 spot/niche 구조를 Stage 1.5에서 spot/region token으로 반영하고, JEPA식 latent prediction을 통해 context-aware representation을 학습한다.

## 3. Stage 1: RNA/Spot Foundation

### 3.1 목적

Stage 1의 목적은 **각 spot의 transcriptomic state를 표현하는 encoder**를 만드는 것이다. 여기서 중요한 것은 모든 gene expression value를 정확히 복원하는 것이 아니라, 다음 정보를 robust하게 담는 것이다.

- 어떤 gene들이 발현되었는가?
- gene들 사이의 co-expression/dependency는 어떤가?
- spot 내부에서 상대적으로 중요한 gene은 무엇인가?
- gene expression rank/salience가 spot representation에 반영되는가?
- downstream에서 HVG expression 또는 masked HVG를 linear probe로 예측할 수 있는가?

현재 Stage 1은 `top_hvg_gene` 기반 tx encoder를 사용한다. 입력은 vocab clipping, normalization, sequence sampling을 거친 spot-level expression vector이다.

### 3.2 입력 처리

Stage 1 입력은 prepared H5의 spot expression vector `hvg`이다.

현재 main 계열 설정은 다음과 같다.

| 항목 | 현재 추천 설정 | 이유 |
|---|---|---|
| gene normalization | `global_median` | Geneformer-style relative expression salience를 반영하기 위함 |
| vocab clip | `4096` | full biomarker discovery보다 robust spot representation을 우선함 |
| sequence sampling | `random` | 특정 top gene만 반복 학습하는 shortcut을 줄이고 다양한 partial view를 학습 |
| max seq len | `512` | 속도, memory, inference consistency 균형 |
| mask ratio | `0.15` 또는 speed 후보 `0.20` | MSM target 수와 난이도 균형 |
| value augmentation | `mixed` | masked symbol prediction에서 value shortcut을 줄임 |

`global_median` normalization은 absolute count reconstruction보다 **relative salience**를 강조한다. 따라서 MVM absolute value score만으로 Stage 1을 판단하면 안 되고, Spearman/rank/intrinsic manifold/linear probe를 함께 봐야 한다.

### 3.3 Primary Objective: MSM

Stage 1의 중심 objective는 **MSM, masked symbol modeling**이다.

```text
input spot gene sequence
  -> 일부 gene symbol을 [MASK]
  -> masked/unmasked value에 keep/noise/dropout augmentation
  -> encoder
  -> masked 위치의 gene symbol classification
  -> cross entropy loss
```

MSM의 직관은 BERT와 유사하다.

- 문장에서 단어를 가리고 주변 단어로 맞추듯이,
- spot에서 gene symbol을 가리고 주변 expressed gene/value context로 맞춘다.

여기서 value는 target symbol을 너무 쉽게 알려주는 shortcut이 될 수 있으므로, masked gene value와 unmasked gene value를 다르게 augmentation한다.

- masked gene value: keep / noise / dropout mixture
- unmasked gene value: 대부분 keep, 일부 noise
- 목적: value 하나만 보고 symbol을 맞추는 것이 아니라, cross-gene context를 보게 함

### 3.4 Multi-Chunk JEPA: Spot-State 보조 Objective

MSM은 gene dependency 학습에는 강하지만, 그것만으로는 `spot representation`이 안정적으로 형성된다고 보장하기 어렵다. 그래서 Stage 1에 multi-chunk JEPA를 보조 objective로 둔다.

#### 핵심 아이디어

한 spot의 non-zero gene sequence를 여러 chunk로 나눈다.

```text
spot sequence = [expressed genes ...]
  -> context chunks: C1, C2
  -> target chunks : T1, T2
```

Student는 context chunk만 본다. EMA teacher는 full clean spot을 보고 target gene 위치의 latent를 추출한다. Predictor는 context representation과 target gene-list cue를 이용해 target chunk latent를 예측한다.

```text
Student: context gene list + context values -> z_context
Teacher: full clean spot -> gather target gene token latents -> z_target_k
Predictor: z_context + target slot query + target gene-list query -> z_pred_k
Loss: SmoothL1(z_pred_k, stopgrad(z_target_k))
```

이 구조는 I-JEPA와 다음처럼 대응된다.

| I-JEPA | Stage 1 multi-chunk JEPA |
|---|---|
| image | one spot expression sequence |
| context block | context gene chunk |
| target block | target gene chunk |
| target position embedding | target gene-list query |
| target pixel content | target expression-dependent latent |
| predictor | context -> target latent predictor |

중요한 점은 **target value는 query로 주지 않는다**는 것이다. Target gene list는 “어느 block을 예측할지” 알려주는 위치 단서에 해당하지만, target expression value는 예측해야 할 semantic content이므로 숨겨야 한다.

### 3.5 Shortcut 제어

Target gene-list query는 I-JEPA의 position embedding보다 정보량이 많다. 특정 gene set 자체가 expression pattern의 prior를 갖기 때문이다. 그래서 shortcut을 감시하고 줄이기 위해 다음 장치를 둔다.

| 장치 | 의미 |
|---|---|
| `multi_chunk_target_id_scale=0.25` | target gene-list query가 너무 강해지지 않게 scale down |
| `multi_chunk_query_only_smoothl1` | context 없이 target gene-list query만으로 target을 맞추는 정도 |
| `multi_chunk_context_gain_over_query_only` | context를 넣었을 때 query-only보다 얼마나 좋아지는지 |
| `multi_chunk_ctx_tgt_overlap` | context-target gene overlap. 낮을수록 I-JEPA-like |
| `multi_chunk_koleo_weight=0.01` | context/predicted chunk latent collapse 방지 |

해석 기준은 다음과 같다.

```text
context_gain_over_query_only > 0
```

이면 target gene-list query만으로 푸는 것이 아니라 context gene/value 정보를 실제로 사용한다는 뜻이다.

### 3.6 Stage 1 Representation 종류

Stage 1에서는 여러 representation을 구분해서 평가한다.

| 표현 | 의미 | 사용 |
|---|---|---|
| `h_tx` | full sampled spot sequence를 한 번 encode한 기본 transcriptomic representation | Stage 1 primary, Stage 2 target 후보 |
| `z_chunk` / `chunk_state` | sampled gene chunk 하나의 representation | chunk-level diagnostic |
| `z_spot` / `spot_state` | 여러 chunk representation을 aggregate한 spot-state representation | multi-chunk JEPA의 목적 표현, Stage 1.5 입력 후보 |

현재 관찰상 `h_tx`가 test linear probe에서 가장 강하고, `spot_state`가 `chunk_state`보다 좋다. 따라서 현재는 다음처럼 해석하는 것이 안전하다.

- `h_tx`: 가장 안정적인 Stage 1 transcriptomic target
- `spot_state`: multi-chunk를 통해 얻는 spot-state 후보
- `z_spatial`: Stage 1.5에서 최종적으로 강화해야 할 spatial-aware spot representation

### 3.7 Stage 1 평가

Stage 1 평가는 loss만 보면 안 된다. 특히 vocab clip을 하면 CE scale이 full vocab보다 작아질 수 있고, global median normalization은 absolute value reconstruction에 불리할 수 있다.

주요 평가 축은 다음과 같다.

#### Intrinsic / Zero-shot-like

| Metric | 의미 |
|---|---|
| `masked_symbol_top{k}_acc` | masked gene symbol top-k prediction |
| `masked_symbol_ce_gain` | random CE 대비 얼마나 좋아졌는지 |
| `intrinsic_effective_rank` | embedding collapse 여부 |
| `intrinsic_expression_distance_spearman` | expression distance와 embedding distance의 rank correlation |
| `intrinsic_gene_embedding_corr_spearman` | ground-truth gene-gene correlation과 learned gene embedding similarity의 일치 |

#### Linear Probe

| Metric | 의미 |
|---|---|
| `linear_probe_hvg_*` | frozen embedding으로 HVG expression을 Ridge/linear probe로 예측 |
| `linear_probe_masked_hvg_*` | masked subset gene에 대한 expression 예측 |
| `linear_probe_hvg_rank_spot_rank_spearman` | spot 내부 gene rank/salience 보존 |
| `linear_probe_hvg_rank_top10_overlap` | top expressed/salient gene overlap |

#### Representation-specific

| Prefix | 의미 |
|---|---|
| `chunk_state_*` | chunk representation이 spot task를 얼마나 풀 수 있는지 |
| `spot_state_*` | chunk aggregate representation이 spot task를 얼마나 풀 수 있는지 |
| no prefix or `h_tx` | 기본 full spot tx representation |

### 3.8 현재 Stage 1 예시 결과

현재 완료된 run:

```text
stage1_speed_multi_chunk_late_gm_v4096_random512_mr20_mid_vamixed_msm_mc_w010_late_q0p25_k0p01_w0p10_warm10r10_lr1p5
```

설정 요약:

| 항목 | 값 |
|---|---|
| norm | global_median |
| vocab | 4096 |
| sampling | random512 |
| mask ratio | 0.20 |
| capacity | spatula_mid |
| value aug | mixed |
| objective | MSM + late multi-chunk JEPA |
| multi-chunk warmup | 10 epochs |
| multi-chunk ramp | 10 epochs |
| LR | 3e-4 |

Validation 흐름:

```text
val MSM CE        6.84 -> 4.25
val MSM top10     0.081 -> 0.579
clean MSM CE      6.50 -> 2.96
clean MSM top10   0.102 -> 0.758
```

Test 결과 핵심:

```text
linear_probe_hvg_spearman_mean        0.539
linear_probe_hvg_pearson_mean         0.734
linear_probe_hvg_r2_mean              0.547
linear_probe_hvg_rank_spot_spearman   0.530
masked_hvg_spearman_mean              0.525
```

Multi-chunk JEPA diagnostic:

```text
multi_chunk_jepa                     ~0.058
context_target_smoothl1              ~0.417
query_only_smoothl1                  ~0.209
context_gain_over_query_only         ~0.150
context_target_cosine_distance       ~1.79
```

해석:

- MSM은 잘 학습되었다.
- HVG/rank probe 성능도 좋다.
- multi-chunk JEPA는 trivial하지 않다.
- context gene/value가 target prediction에 실제로 기여한다.
- 다만 JEPA loss contribution은 작으므로, 현재 Stage 1은 “강한 MSM foundation + 약한 spot-state regularization”으로 보는 것이 맞다.

## 4. Stage 1.5: Spatial Foundation

### 4.1 목적

Stage 1.5의 목적은 Stage 1에서 얻은 spot-level representation을 기반으로 **spatial context-aware spot representation**을 만드는 것이다.

Stage 1이 한 spot 내부만 본다면, Stage 1.5는 다음을 본다.

- anchor spot의 transcriptomic state
- anchor spot의 image patch feature
- neighbor/region transcriptomic context
- neighbor/region image context
- spatial coordinate와 KNN graph

따라서 Stage 1.5는 다음 질문에 답해야 한다.

> 이 spot은 자기 gene expression만 보면 어떤 상태인가?  
> 그리고 주변 spot/region까지 보면 tissue context 안에서 어떤 의미를 갖는가?

이것이 최종 `spot_encoder`에 spatial awareness를 부여하는 단계다.

### 4.2 HyperST와의 연결

HyperST는 spot과 niche/region의 관계를 사용해 spatial context를 반영한다. 이 프로젝트에서는 이를 다음처럼 재해석한다.

| HyperST 개념 | 본 프로젝트 |
|---|---|
| spot | anchor spot token |
| niche | anchor + neighbors region token |
| spot/niche contrast | spot/region JEPA latent prediction |
| spatial context | KNN/ego graph + region aggregation |

차이점은 latent space를 contrastive하게 강제하기보다, **JEPA 방식으로 visible context가 masked target latent를 예측하도록 학습**한다는 점이다.

### 4.3 Stage 1.5 입력

Stage 1.5는 Stage 1 checkpoint를 사용한다. Stage 1 encoder는 frozen된다.

입력 채널은 다음과 같다.

| 채널 | 의미 |
|---|---|
| `h_tx` | anchor spot을 Stage 1 tx_encoder로 encode한 representation |
| `h_img` | anchor spot pathology/image patch feature, 예: UNI feature |
| `h_region_tx` | neighbor region의 transcriptomics aggregate를 Stage 1 tx_encoder로 encode한 representation |
| `h_region_img` | neighbor image feature pooling 또는 region image feature |
| `(x, y)` | spatial coordinate / positional encoding |

Region transcriptomics aggregation은 다음 방식이 가능하다.

| 방식 | 의미 |
|---|---|
| `mean` | neighbor log expression 평균 |
| `sum_log1p` | raw count space에서 sum 후 log1p |
| `weighted` | 거리 기반 weighted mean |

중요한 점은 Stage 1과 동일한 gene normalization/vocab clip을 Stage 1.5에서도 적용해야 한다는 것이다. Stage 1 encoder가 `global_median + vocab4096`으로 학습되었다면, Stage 1.5의 anchor/region tx 입력도 같은 방식으로 맞춰야 distribution shift를 줄일 수 있다.

### 4.4 Token 구성: fused vs separate

Stage 1.5는 spot/region 정보를 spatial backbone에 넣는 방식에 따라 두 가지 모드가 있다.

#### Fused mode

```text
[h_tx, h_img, h_region_tx, h_region_img, pos] -> MLP -> one token
```

장점:

- 단순하고 안정적이다.
- memory/compute가 낮다.

단점:

- spot과 region이 하나의 token으로 섞인다.
- JEPA에서 무엇이 context이고 무엇이 target인지 해석이 흐려질 수 있다.

#### Separate mode

```text
spot token   = f(h_tx, h_img, pos, type=spot)
region token = f(h_region_tx, h_region_img, pos, type=region)
```

장점:

- spot token과 region token을 분리한다.
- HyperST의 spot/niche 구조와 더 잘 맞는다.
- JEPA에서 “visible region -> masked spot” 구조를 명확하게 만들 수 있다.

단점:

- token 수가 2배가 된다.
- memory/compute가 더 크다.

Stage 1.5의 main candidate는 장기적으로 **separate token + spot mask**가 더 적합하다. 다만 현재 config 기본값은 `fused`로 되어 있으므로, main 실험에서는 명시적으로 `region_token_mode=separate`를 사용하는 것이 좋다.

### 4.5 Spatial JEPA Objective

Stage 1.5의 objective는 spatial graph 위의 JEPA이다.

```text
1. sample/region graph 구성
2. 일부 spot token을 block-wise mask
3. student spatial_encoder는 masked graph를 encode
4. EMA teacher spatial_encoder는 clean graph를 encode
5. masked target 위치에서 teacher latent를 target으로 사용
6. SmoothL1(student masked latent, stopgrad teacher latent)
```

현재 추천 구조:

| 항목 | 추천 |
|---|---|
| subgraph | `ego` |
| mask strategy | `block` |
| mask target | `spot` |
| token mode | `separate` |
| loss | `smooth_l1` |
| EMA | `0.999` |

이 구조의 직관은 다음과 같다.

> 주변 region/context를 보고, 가려진 anchor spot의 latent state를 예측한다.

이는 I-JEPA의 “visible context block으로 masked target block representation을 예측”하는 구조를 spatial transcriptomics graph에 옮긴 것이다.

### 4.6 Stage 1.5 Architecture

Stage 1.5 spatial encoder는 `configs/stage15/model.yaml` 기준으로 다음 요소를 갖는다.

| 항목 | 기본값 |
|---|---|
| arch | `kgnn` |
| fuse_dim | 256 |
| n_layers | 3 |
| n_heads | 4 |
| dropout | 0.10 |
| region_token_mode | `fused` 기본, main은 `separate` 권장 |

`kgnn`은 KNN graph와 edge/spatial 정보를 사용하는 lightweight spatial backbone이다. `smooth`는 non-parametric neighbor averaging control로 사용할 수 있고, `kxformer`는 더 큰 transformer-style candidate로 볼 수 있다.

### 4.7 Stage 1.5 평가

Stage 1.5는 validation loss만으로 판단하면 부족하다. Spatial representation이 실제로 좋아졌는지 봐야 한다.

#### Quantitative evaluation

| 평가 | 의미 |
|---|---|
| DLPFC layer linear probe | spatial embedding이 cortical layer annotation을 예측할 수 있는가 |
| zero-shot clustering | embedding만으로 spatial/cell-type cluster가 분리되는가 |
| KNN purity | 가까운 embedding들이 같은 spatial label/region에 속하는가 |
| gene map SCC | 특정 gene의 spatial expression pattern을 embedding으로 예측할 수 있는가 |
| GT vs predicted gene heatmap | 정성적으로 spatial pattern이 맞는가 |

특히 gene map SCC는 Stage 1.5 목표와 잘 맞는다.

```text
train split: z_spatial -> selected gene expression linear/Ridge probe 학습
test split : spot별 predicted gene expression map 생성
metric     : GT map vs predicted map Spearman SCC
visual     : spatial coordinate 위 GT/pred heatmap 비교
```

이 평가는 다음 질문에 답한다.

> spatial-aware embedding이 특정 gene의 공간적 high/low pattern을 보존하는가?

#### Qualitative visualization

Stage 1.5 test step에서는 다음 시각화가 중요하다.

- UMAP colored by sample/source/layer/cluster
- spatial cluster map
- GT gene expression spatial heatmap
- predicted gene expression spatial heatmap
- gene별 SCC barplot

이런 시각화는 loss보다 연구 설득력이 크다. 특히 downstream report에서는 “embedding이 tissue structure를 실제로 잡는가”를 보여주는 자료가 된다.

## 5. Stage 1 + Stage 1.5가 합쳐진 spot_encoder

최종 spot_encoder는 다음처럼 계층화된다.

```text
SpotEncoder(x_i, neighbors_i, image_i)

1. Stage1 tx_encoder
   input : spot i expression
   output: h_tx_i, optional z_spot_i

2. Region construction
   input : neighbors of spot i
   output: h_region_tx_i, h_region_img_i

3. Stage1.5 spatial_encoder
   input : spot token + region token + spatial graph
   output: z_spatial_i
```

즉 최종적으로 사용할 embedding은 task에 따라 달라질 수 있다.

| 사용처 | 추천 representation |
|---|---|
| pure transcriptomics baseline | `h_tx` |
| Stage 1 spot-state diagnostic | `z_spot` / `spot_state` |
| spatial downstream | `z_spatial` |
| Stage 2 image alignment target | `h_tx`, `z_spatial`, 또는 둘 다 ablation |
| image-to-expression prediction | `h_tx` 또는 `z_spatial` target + expression/rank probe |

현재 관점에서는 Stage 2 target을 하나로 고정하기보다 다음을 ablation하는 것이 좋다.

1. Image -> `h_tx`
2. Image -> `spot_state`
3. Image -> `z_spatial`
4. Image -> expression/rank target 직접 예측

## 6. Stage 2와의 연결

Stage 2는 pathology image encoder와 transcriptomics/spatial target을 alignment한다. 이때 Stage 1/1.5가 제공하는 target quality가 매우 중요하다.

- Stage 1만 쓰면 image는 spot-intrinsic molecular state와 align된다.
- Stage 1.5까지 쓰면 image는 local tissue context까지 반영한 molecular-spatial state와 align된다.

따라서 Stage 2 downstream은 다음을 봐야 한다.

| Task | 목적 |
|---|---|
| cross-modal retrieval | image와 RNA/spatial embedding이 같은 spot에서 가까운가 |
| image-to-HVG expression probe | image embedding이 molecular state를 예측하는가 |
| image-to-rank prediction | relative salience를 예측하는가 |
| HEST/PathBench/MSI/MIL | slide-level downstream에서 유용한가 |

## 7. 현재 추천 실험 흐름

### 7.1 Stage 1

현재 가장 유망한 후보는 다음이다.

```bash
bash scripts/ablation/run_all_stage1.sh speed_multi_chunk_late
```

또는 비교용으로:

```bash
bash scripts/ablation/run_all_stage1.sh \
  speed_multi_chunk_late \
  speed_multi_chunk \
  speed_msm
```

비교 포인트:

| 후보 | 의미 |
|---|---|
| `speed_msm` | MSM만 빠르게 saturation |
| `speed_multi_chunk` | multi-chunk를 epoch 6부터 빠르게 투입 |
| `speed_multi_chunk_late` | MSM을 epoch 10까지 먼저 안정화한 뒤 multi-chunk 투입 |

### 7.2 Stage 1.5

Stage 1.5는 Stage 1 checkpoint를 입력으로 사용한다.

```bash
STAGE1_CKPT=results/runs/<stage1_tag>/ckpt_tx_encoder_best.pt \
TAG=stage15_<stage1_tag>_spatial \
bash scripts/train/stage15_main.sh
```

권장 설정:

```text
region_token_mode = separate
mask_target       = spot
subgraph_kind     = ego
mask_strategy     = block
```

OOM이 발생하면:

```bash
TX_ENCODE_BATCH=128 BATCH_SIZE=8 \
STAGE1_CKPT=results/runs/<stage1_tag>/ckpt_tx_encoder_best.pt \
TAG=stage15_<stage1_tag>_spatial \
bash scripts/train/stage15_main.sh
```

### 7.3 Stage 1.5 test/eval

DLPFC spatial downstream:

```bash
python scripts/eval/dlpfc_eval.py \
  --dlpfc-dir /data/spatiallibd \
  --ckpts results/runs/<stage1_tag>/ckpt_tx_encoder_best.pt \
  --representations h_tx,spot_state \
  --out results/eval/dlpfc_<stage1_tag>.csv \
  --per-sample-out results/eval/dlpfc_<stage1_tag>_per_sample.csv \
  --viz-out-dir results/figures/dlpfc_<stage1_tag> \
  --genes MBP SNAP25 PCP4 GFAP
```

Stage 1.5 spatial gene map:

```bash
python scripts/eval/stage15_gene_map.py \
  --stage1-ckpt results/runs/<stage1_tag>/ckpt_tx_encoder_best.pt \
  --spatial-ckpt results/runs/<stage15_tag>/ckpt_spatial_best.pt \
  --split test \
  --genes MKI67 EPCAM COL1A1 CD3D \
  --probe-train-samples 20 \
  --max-train-spots 20000
```

## 8. 성공 기준

Stage 1 성공 기준:

- MSM/clean MSM top-k가 안정적으로 상승한다.
- HVG linear probe Spearman/Pearson/R2가 높다.
- rank-based probe가 좋다.
- intrinsic effective rank가 collapse되지 않는다.
- multi-chunk 사용 시 `context_gain_over_query_only > 0`이다.

Stage 1.5 성공 기준:

- spatial clustering/layer probe가 Stage 1 representation보다 좋아진다.
- gene map SCC가 의미 있게 나온다.
- spatial heatmap에서 GT와 predicted pattern이 정성적으로 맞는다.
- Stage 2 image alignment target으로 썼을 때 retrieval/expression/downstream이 개선된다.

최종 spot_encoder 성공 기준:

```text
h_tx는 molecular state를 잘 담고,
z_spatial은 molecular state + spatial tissue context를 더 잘 담으며,
Stage 2에서 pathology image와 alignment했을 때 downstream task 성능이 오른다.
```

## 9. 현재 해석 요약

현재 Stage 1 `speed_multi_chunk_late` run은 다음 의미가 있다.

- MSM saturation을 빠르게 당기는 데 성공했다.
- clean MSM top10이 약 0.76까지 상승했다.
- test HVG linear probe 성능이 좋다.
- multi-chunk JEPA는 target gene-list shortcut만으로 풀리는 task는 아니다.
- 다만 multi-chunk loss contribution은 작으므로, Stage 1 representation의 중심은 아직 MSM이다.

따라서 현재 전략은 다음처럼 정리할 수 있다.

1. Stage 1은 MSM 기반 strong transcriptomics foundation으로 둔다.
2. Multi-chunk JEPA는 spot-state regularizer로 사용한다.
3. Stage 1.5에서 본격적으로 spatial context를 주입한다.
4. 최종 spot_encoder는 Stage 1 `h_tx`와 Stage 1.5 `z_spatial`의 계층적 조합으로 정의한다.

## 부록: Stage 1 Multi-Chunk JEPA 및 정성 평가 업데이트

### I-JEPA식 multi-target chunk sampling

현재 Stage 1의 `msm_multi_chunk` 설정은 고정된 두 target chunk만 쓰는 방식에서 벗어나, `multi_chunk_target_chunks: auto`를 사용할 수 있다. 이 경우 `multi_chunk_n_chunks=4`라면 context chunk 1개와 target chunk 3개를 사용한다. 이는 I-JEPA의 “one/few context block + multiple target blocks” 구성에 더 가깝다.

chunk 길이도 완전 고정이 아니다. `multi_chunk_len`은 hard upper bound로만 쓰고, 실제 target chunk는 spot별 expressed-gene sequence 길이의 `multi_chunk_target_scale` 범위에서 샘플링된다. 기본값은 `[0.15, 0.25]`이다. context chunk는 target union과 가능한 한 disjoint하게 샘플링되며, 기본 context scale은 `[0.45, 0.65]`이다. 짧은 spot에서는 fallback으로 overlap이 생길 수 있으므로, 학습 로그의 `multi_chunk_ctx_tgt_overlap`, `multi_chunk_context_eff_len_mean`, `multi_chunk_target_eff_len_mean`, `multi_chunk_n_targets`를 같이 확인해야 한다.

핵심 해석은 다음과 같다.

- context chunk: 일부 gene list와 value를 관찰한 부분 표현
- target chunks: 숨겨진 여러 gene block의 teacher latent target
- predictor input: context representation + target slot query + target gene-id query
- predictor target: teacher가 full clean spot을 본 뒤 target gene positions에서 추출한 latent

따라서 target expression value는 predictor에 직접 주어지지 않고, target gene identity는 I-JEPA의 target position embedding에 대응하는 조건 정보로만 사용된다.

### Stage 1 downstream/visual test hook

`run_all_stage1.sh`는 기본적으로 빠른 Stage1 test CSV까지만 생성한다. DLPFC/spatialLIBD의 layer probe, zero-shot clustering, kNN purity, gene-map SCC, cluster panel, gene expression heatmap까지 같이 보고 싶다면 다음처럼 실행한다.

```bash
ABL_RUN_DLPFC=1 ABL_DLPFC_DIR=/data/spatiallibd ABL_DLPFC_VIZ=1 bash scripts/ablation/run_all_stage1.sh speed_multi_chunk_late
```

출력은 다음 위치에 생성된다.

- `results/eval/dlpfc_stage1_core_full_compare.csv`
- `results/eval/dlpfc_stage1_core_full_per_sample.csv`
- `results/figures/dlpfc_stage1_core_full/`

이 평가에서는 `h_tx`, `chunk_state(z_chunk)`, `spot_state(z_spot aggregate)`를 분리해서 report할 수 있다. spot-level downstream에서는 `spot_state`가 가장 중요한 inference representation이지만, `chunk_state`와 비교하면 multi-chunk aggregation이 실제로 downstream signal을 보강하는지 확인할 수 있다.

## 부록: MSM -> Chunk-JEPA -> Spatial-JEPA Sequential Ablation

현재 mainline은 Stage 1 안에서 `MSM only`와 `MSM + multi-chunk JEPA`를 비교하고, 그 다음 Stage 1.5에서 spatial JEPA를 학습하는 방식이다. 다만 stage의 의미를 더 엄밀히 분리하기 위해 다음 순차 ablation도 지원한다.

```text
Stage 1    : MSM only
Stage 1.25 : multi-chunk JEPA refinement
Stage 1.5  : spatial JEPA
```

실행 스크립트는 다음과 같다.

```bash
bash scripts/ablation/run_all_stage1_chunk_stage15.sh
```

기존 MSM checkpoint를 재사용하려면 다음처럼 지정한다.

```bash
STAGE1_CKPT=results/runs/<stage1_msm_run>/ckpt_tx_encoder_best.pt SKIP_BASE_STAGE1=1 bash scripts/ablation/run_all_stage1_chunk_stage15.sh
```

이 경로의 목적은 “Stage 1의 gene-language learning”과 “chunk-level spot-state learning”을 분리해서, `MSM + chunk-JEPA`를 한 번에 학습하는 현재 후보와 비교하는 것이다. 실험적으로는 다음 비교가 중요하다.

- `Stage1 MSM only` vs `Stage1.25 chunk-refined`
- `Stage1 MSM + chunk-JEPA joint` vs `MSM -> chunk-JEPA sequential`
- 각 ckpt의 `h_tx`, `chunk_state`, `spot_state` DLPFC/gene-map 성능
- chunk-refined ckpt를 넣은 Stage 1.5 spatial JEPA의 spatial clustering/gene-map SCC

이 ablation은 연구적으로 깔끔하지만 학습 비용이 더 든다. 따라서 mainline을 대체하기보다는, chunk-JEPA가 실제로 spot-state downstream을 개선하는지 확인하는 보조 경로로 둔다.

## 부록: 단일 Python Pipeline Entrypoint

Stage 1부터 Stage 1.5까지 checkpoint boundary를 유지하면서 한 번에 실행하려면 다음 entrypoint를 사용한다.

```bash
python scripts/run_spot_encoder_pipeline.py --pipeline joint
```

`joint`는 현재 mainline처럼 Stage 1 objective를 학습한 뒤 best tx checkpoint를 Stage 1.5 spatial JEPA에 넘긴다.

```text
Stage 1 ckpt:    results/runs/<stage1_tag>/ckpt_tx_encoder_best.pt
Stage 1.5 ckpt:  results/runs/<stage15_tag>/ckpt_spatial_best.pt
```

MSM과 chunk-JEPA를 분리하는 sequential 경로는 다음과 같다.

```bash
python scripts/run_spot_encoder_pipeline.py --pipeline sequential
```

이 경우 실행 흐름은 다음과 같다.

```text
Stage 1    : MSM only
Stage 1.25 : chunk-JEPA refinement from Stage 1 best tx checkpoint
Stage 1.5  : spatial JEPA from Stage 1.25 best tx checkpoint
```

각 stage 이후 hold-out/test evaluation을 실행한다.

- Stage 1 / Stage 1.25: `scripts/eval/stage1_tx.py`
- DLPFC qualitative/downstream: `scripts/eval/dlpfc_eval.py`
- Stage 1.5 spatial: `scripts/eval/stage15_indist.py`
- Optional Stage 1.5 gene map: `--stage15-gene-map`

예시:

```bash
python scripts/run_spot_encoder_pipeline.py   --pipeline sequential   --capacity spatula_mid   --stage1-epochs 50   --stage125-epochs 30   --stage15-epochs 30   --make-viz   --stage15-gene-map
```

기존 checkpoint를 재사용할 수도 있다.

```bash
python scripts/run_spot_encoder_pipeline.py   --pipeline sequential   --stage1-ckpt results/runs/<stage1_run>/ckpt_tx_encoder_best.pt
```

### Pipeline ablation grid / sweep

`run_spot_encoder_pipeline.py`는 단일 후보뿐 아니라 ablation grid도 실행할 수 있다. 내부 학습은 기존 bash launcher와 `accelerate`를 그대로 사용한다. Stage 1과 Stage 1.25는 `--num-proc-stage1` 값으로 multi-GPU accelerate를 사용한다. Stage 1.5는 현재 trainer 안정성을 위해 기본 `--num-proc-stage15 1`을 권장한다.

Predefined grid:

```bash
# Stage 1 objective 비교: MSM only vs MSM+multi-chunk
python scripts/run_spot_encoder_pipeline.py --grid stage1_core

# capacity 비교
python scripts/run_spot_encoder_pipeline.py --grid stage1_capacity

# MSM -> chunk-JEPA sequential 후보 비교
python scripts/run_spot_encoder_pipeline.py --grid sequential_core

# spatial-JEPA 방향/region aggregation 비교
python scripts/run_spot_encoder_pipeline.py   --grid stage15_spatial   --stage1-ckpt results/runs/<stage1_run>/ckpt_tx_encoder_best.pt
```

Custom sweep:

```bash
python scripts/run_spot_encoder_pipeline.py   --pipeline sequential   --sweep stage125_mc_weight=0.10,0.30   --sweep stage125_target_chunks=2,auto
```

지원하는 주요 sweep key는 CLI argument 이름에서 `--`를 빼고 `_`로 바꾼 형태다.

- `pipeline`: `joint`, `sequential`
- `capacity`: `spatula_lite`, `spatula_mid`, `spatula_large`
- `stage1_objective`: `msm_only`, `msm_multi_chunk`, `view_jepa_w005`, `view_jepa_w010`, `dino_late_no_koleo`
- `stage125_mc_weight`, `stage125_target_chunks`, `stage125_target_scale`, `stage125_context_scale`
- `stage15_mask_target`: `spot`, `region`, `both`
- `stage15_region_tx_agg`: `mean`, `sum_log1p`, `weighted`
- `stage15_subgraph_kind`: `ego`, `random`

각 조합은 tag/eval prefix에 sweep suffix를 붙여 저장되므로 run directory 이름만 봐도 실험 조건을 알 수 있다.

### Bash preset wrapper

Python pipeline을 직접 호출하지 않고, 자주 쓰는 실험/ablation preset을 bash로 실행할 수도 있다.

```bash
# 단일 실험: PIPELINE=joint 또는 sequential 지정 가능
bash scripts/ablation/run_spot_encoder_pipeline.sh single

# Stage1 objective ablation
bash scripts/ablation/run_spot_encoder_pipeline.sh stage1

# MSM -> chunk-JEPA -> spatial-JEPA sequential ablation
bash scripts/ablation/run_spot_encoder_pipeline.sh sequential

# Stage1.5 spatial-JEPA 방향/aggregation ablation
STAGE1_CKPT=results/runs/<stage1_run>/ckpt_tx_encoder_best.pt bash scripts/ablation/run_spot_encoder_pipeline.sh spatial

# curated grids를 순서대로 실행
# 순서: sequential -> joint MSM+chunk-JEPA -> vocab -> capacity -> spatial
bash scripts/ablation/run_spot_encoder_pipeline.sh all
```

`all` preset의 기본 순서는 smoke test 이후 실제 ablation 우선순위에 맞춰져 있다.

1. `sequential`: MSM -> chunk-JEPA -> spatial-JEPA
2. `joint_chunk`: Stage1 MSM+chunk-JEPA -> spatial-JEPA
3. `vocab`: 4096 / 8192 / full
4. `capacity`: spatula_lite / spatula_mid / spatula_large
5. `spatial`: spot-target vs region-target, mean vs weighted region aggregation

주요 환경 변수:

```bash
CAPACITY=spatula_mid
STAGE1_EPOCHS=50
STAGE125_EPOCHS=30
STAGE15_EPOCHS=30
NUM_PROC_STAGE1=8
NUM_PROC_STAGE15=1
MAKE_VIZ=1
SKIP_EVAL=0
DRY_RUN=1
```

custom sweep도 가능하다.

```bash
EXTRA_ARGS='--pipeline sequential --sweep stage125_mc_weight=0.10,0.30 --sweep stage125_target_chunks=2,auto' bash scripts/ablation/run_spot_encoder_pipeline.sh custom
```

