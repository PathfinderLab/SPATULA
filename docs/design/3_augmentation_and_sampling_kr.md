# 3. Augmentation and Sampling

작성일: 2026-06-24  
역할: Stage 1 value augmentation, symbol masking, sequence/chunk sampling의 철학과 ablation 정리

## 1. 왜 augmentation이 필요한가

Stage 1 MSM은 masked gene symbol을 맞추는 objective다. 그런데 expression value가 그대로 남아 있으면 모델은 cross-gene context를 배우기보다 value shortcut으로 symbol을 맞출 수 있다.

따라서 augmentation의 목적은 다음이다.

- masked symbol prediction이 너무 쉽지 않게 만든다.
- value 하나가 gene identity를 누설하지 않게 한다.
- unmasked context는 너무 망가뜨리지 않아 spot state를 유지한다.
- partial sequence/chunk view에서도 stable representation을 만들게 한다.

## 2. Masked gene과 unmasked gene을 분리해서 봐야 하는 이유

MSM에서는 두 종류의 token이 있다.

```text
masked gene   : symbol이 가려진 target token
unmasked gene : context token
```

이 둘에 같은 value augmentation을 적용하면 목적이 흐려진다.

| 위치 | 목표 | augmentation 철학 |
|---|---|---|
| masked gene | symbol을 맞혀야 하는 target | value shortcut을 줄이기 위해 keep/noise/dropout mixture |
| unmasked gene | context 제공 | context를 유지하되 일부 noise로 robust하게 함 |

## 3. 현재 추천 정책

```text
mask_ratio = 0.15
value_aug = mixed
sequence_sampling = random
max_seq_len = 512
```

직관:

- `mask_ratio=0.15`: BERT/Geneformer식 안정적 시작점
- `mixed value augmentation`: shortcut 방지와 학습 가능성의 균형
- `random512`: 매 epoch 다른 partial view를 보게 하여 overfitting 감소

## 4. 간단한 수식

spot sequence를 `S = {(g_l, v_l)}`라고 하자. mask indicator `m_l`은 다음과 같다.

```text
m_l ~ Bernoulli(r_mask)
```

masked token의 symbol은 `[MASK]`로 바뀐다.

```text
g'_l = [MASK] if m_l = 1 else g_l
```

value augmentation은 위치별로 다르게 적용한다.

```text
v'_l = A_mask(v_l)      if m_l = 1
v'_l = A_context(v_l)   if m_l = 0
```

예시:

```text
A_mask(v) =
  keep(v)      with p_keep
  v + eps      with p_noise
  0            with p_dropout

eps ~ Normal(0, sigma^2)
```

## 5. Sampling 철학

### 5.1 zero expressed removal

발현되지 않은 gene은 문장에 등장하지 않는 단어처럼 본다.

```text
S_i = {(g,v) | x_i,g > 0}
```

### 5.2 sequence sampling

nonzero sequence가 너무 길면 모든 gene을 보지 않고 일부만 sample한다.

```text
C_i ~ sample(S_i, L=max_seq_len)
```

이것은 단순 속도 최적화가 아니라 augmentation이다. 매 epoch 다른 gene subset을 보게 하여 spot state가 특정 gene subset에 과적합되지 않게 한다.

### 5.3 random vs top-k

| 방식 | 장점 | 단점 |
|---|---|---|
| `random` | 다양한 partial view, shortcut 감소, multi-chunk와 잘 맞음 | salience 높은 gene을 놓칠 수 있음 |
| `top_k` | high expression/salience gene 중심 | 매번 같은 gene을 보아 overfit/shortcut 위험 |
| `none` | full sequence 사용 | 느리고 noisy, vocab/full에서 memory 부담 |

현재는 `random`을 default로 둔다.

## 6. Chunk sampling

multi-chunk JEPA에서는 sequence sampling과 chunk sampling을 구분한다.

```text
nonzero sequence S_i
  -> sampled chunk C_i^k
```

즉 `random512`는 전체 spot을 대표하는 하나의 sampled view이면서, multi-chunk에서는 context/target chunk의 source가 된다.

I-JEPA-like dynamic chunking에서는 target/context length가 고정되지 않을 수 있다.

```text
|T| ~ Uniform(scale_min, scale_max) * |S|
|C| ~ Uniform(context_min, context_max) * |S|
```

단, 너무 짧은 sequence에서는 single-chunk spot이 되어 JEPA signal이 약해질 수 있다. 이 경우:

- target chunk 수를 줄인다.
- chunk size를 dynamic하게 낮춘다.
- MSM-only로 처리한다.

## 7. Ablation 대상

| Axis | 후보 | 질문 |
|---|---|---|
| mask ratio | `0.15`, `0.20` | MSM saturation을 당길 수 있는가? |
| masked value policy | keep/noise/dropout/mixed | target value shortcut을 얼마나 줄일 것인가? |
| unmasked value policy | keep/noise/mixed | context를 얼마나 보존할 것인가? |
| noise sigma | low/high | robust view를 만들되 signal을 망가뜨리지 않는가? |
| sequence sampling | random/top_k | salience와 diversity 중 무엇이 좋은가? |
| max seq len | 256/512 | 속도와 representation trade-off |
| target chunk scale | 0.15-0.25 등 | JEPA target 난이도 조절 |
| context chunk scale | 0.45-0.65 등 | context 정보량 조절 |

## 8. 좋은 augmentation의 조건

좋은 augmentation은 다음 패턴을 만든다.

- `masked_symbol_top10_acc`가 상승한다.
- `clean_msm`도 같이 좋아진다.
- `masked_hvg` probe가 유지/상승한다.
- value shortcut만 늘어나지 않는다.
- `effective_rank`가 collapse하지 않는다.
- multi-chunk에서는 `context_gain_over_query_only`가 양수다.

반대로 MSM top-k만 좋아지고 masked-HVG/linear probe가 떨어지면 shortcut 가능성을 의심한다.
