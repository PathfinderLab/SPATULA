# 2. Stage 1 Objectives

작성일: 2026-06-24  
역할: Stage 1 transcriptomic spot pretraining objective, multi-chunk JEPA, DINO/View-JEPA 후보의 철학과 ablation 정리

## 1. Stage 1의 목표

Stage 1은 sample graph나 pathology image를 보지 않고, spot 내부 transcriptomic state를 표현하는 encoder를 학습한다.

```text
input:  one spot gene/value sequence
output: z_spot_tx or h_tx
```

이 단계의 핵심 질문은 다음이다.

- spot 안의 gene symbol 관계를 학습하는가?
- gene-value context로 masked gene을 추론할 수 있는가?
- relative expression salience가 embedding에 남는가?
- spot-level downstream에 유용한 representation을 얻는가?

## 2. MSM: Primary objective

MSM은 masked symbol modeling이다. BERT/Geneformer류와 유사하게 일부 gene symbol을 가리고, 주변 gene/value context로 정답 gene을 맞춘다.

```text
S_i = {(g_1,v_1),...,(g_L,v_L)}
M_i ~ Bernoulli(mask_ratio)
student_input = mask_symbols(S_i, M_i)
loss_msm = CE(p(g_l | S_i \ M_i), g_l), for l in M_i
```

직관:

```text
spot = gene sentence
masked gene = missing word
neighboring expressed genes + values = context
```

MSM은 Stage 1의 중심 objective로 유지한다. 이유는 gene dependency와 transcriptomic state를 직접 학습하기 때문이다.

## 3. Multi-Chunk JEPA

MSM은 vocab classification이라 수렴이 느리고, spot-level aggregate representation이 반드시 안정적으로 형성된다는 보장이 약하다. 이를 보완하기 위해 multi-chunk JEPA를 사용한다.

### 3.1 기본 아이디어

한 spot의 nonzero sequence에서 context chunk와 target chunk를 나눈다.

```text
S_i -> context chunks C_i^1 ... C_i^m
    -> target chunks  T_i^1 ... T_i^k
```

Student는 context chunks만 보고, EMA teacher는 full clean spot을 본 뒤 target gene positions의 latent를 추출한다.

```text
z_context = Encoder_student(C_i^1,...,C_i^m)
z_target^k = pool(Teacher_full(S_i).per_token[target_positions_k])
z_pred^k = Predictor(z_context, q_slot^k, q_gene^k)
loss_jepa = mean_k SmoothL1(z_pred^k, stopgrad(z_target^k))
```

여기서 `q_gene`은 target gene list query다. target expression value는 주지 않는다.

### 3.2 I-JEPA와의 대응

| I-JEPA | Stage 1 multi-chunk JEPA |
|---|---|
| image | one spot expression sequence |
| context block | context gene chunk |
| target block | target gene chunk |
| target position embedding | target slot query + target gene-list query |
| target pixel content | target expression-dependent latent |
| predictor | context latent -> target latent |

### 3.3 왜 target gene list query를 쓰는가

Image에서는 target block 위치를 coordinate/positional embedding으로 알려준다. Gene sequence에서는 fixed spatial coordinate가 없으므로 “어떤 gene block을 예측해야 하는지”를 알려주는 query가 필요하다.

단, gene list는 coordinate보다 정보량이 많다. 특정 gene set 자체가 expression prior를 가질 수 있기 때문이다. 그래서 아래 shortcut diagnostics가 필요하다.

## 4. Shortcut diagnostics

| 지표 | 의미 | 해석 |
|---|---|---|
| `query_only_smoothl1` | context 없이 target query만으로 target을 맞추는 loss | 낮으면 shortcut 위험 |
| `context_gain_over_query_only` | context가 query-only보다 얼마나 이득을 주는가 | 양수면 context를 사용 |
| `ctx_tgt_overlap` | context와 target gene overlap | 0에 가까울수록 I-JEPA-like |
| `target_id_query_norm` | target gene-list query magnitude | 너무 크면 query dominance 위험 |
| `effective_rank` | representation collapse 여부 | 너무 낮으면 collapse |

핵심 판단식:

```text
gain = L_query_only - L_context_predictor
```

`gain > 0`이면 context gene/value가 target latent prediction에 기여한다.

## 5. DINO / View-JEPA / KoLeo 후보

### 5.1 DINO-style consistency

Teacher는 clean/raw view, student는 masked/noisy view를 본다.

```text
z_teacher = Teacher(clean spot)
z_student = Student(masked/noisy spot)
loss_dino = distance(z_student, stopgrad(z_teacher))
```

장점:
- view consistency를 줄 수 있다.
- corrupted input에서도 stable spot embedding을 만들 수 있다.

위험:
- predictor/centering/regularization이 약하면 collapse 또는 geometry 훼손 가능.
- MSM과 경쟁할 수 있다.

### 5.2 View-JEPA predictor

DINO보다 안전하게 predictor head를 둔다.

```text
z_pred = Predictor(z_student_corrupted)
loss_view_jepa = SmoothL1(z_pred, stopgrad(z_teacher_clean))
```

장점:
- predictor가 mismatch를 흡수하므로 base encoder가 덜 흔들릴 수 있다.
- data2vec/JEPA류에 더 가깝다.

### 5.3 KoLeo regularizer

embedding이 collapse하지 않도록 nearest-neighbor distance를 벌린다.

```text
loss_koleo = - mean_i log min_{j != i} || normalize(z_i) - normalize(z_j) ||_2
```

주의:
- 너무 강하면 biological neighbors까지 밀어낼 수 있다.
- Stage1에서는 작은 weight부터 시작한다.

## 6. Ablation 대상

| Axis | 후보 | 질문 |
|---|---|---|
| objective | `msm_only`, `msm_multi_chunk`, `msm_dino`, `msm_view_jepa` | auxiliary가 MSM representation을 돕는가? |
| multi_chunk weight | `0.10`, `0.30` | JEPA signal이 충분히 강한가? |
| target chunks | `2`, `auto` | multi-target이 더 좋은가? 비용은? |
| target_id_scale | `0.0`, `0.25`, `0.5` | gene-list query가 필요한가, shortcut인가? |
| warmup/ramp | early/late | MSM이 안정화된 후 JEPA를 켜는 것이 좋은가? |
| KoLeo | `0`, `0.01`, `0.05` | collapse 방지와 biological geometry 보존의 균형 |
| mask ratio | `0.15`, `0.20` | MSM saturation 속도와 난이도 균형 |

현재 추천 순서:

1. `msm_only`: 가장 해석 가능한 baseline
2. `sequential msm -> chunk_jepa`: Stage 1과 1.25 역할 분리
3. `joint msm_multi_chunk`: 동시 학습의 이득 확인
4. DINO/View-JEPA는 보조 ablation

## 7. 좋은 Stage 1 run의 조건

좋은 Stage 1 run은 다음이 함께 좋아야 한다.

- `masked_symbol_top10_acc` 상승
- `clean_msm_top10_acc` 상승
- `masked_symbol_ce_norm` 감소
- `linear_probe/masked_hvg/spearman` 상승
- `linear_probe/hvg_rank/spot_rank_spearman` 유지/상승
- `effective_rank` collapse 없음
- multi-chunk 사용 시 `context_gain_over_query_only > 0`
