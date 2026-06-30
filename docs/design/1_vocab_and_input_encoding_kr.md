# 1. Vocab and Input Encoding

작성일: 2026-06-24  
역할: gene vocabulary, normalization, tokenization, sequence construction의 철학과 ablation 대상 정리

## 1. 왜 vocab이 중요한가

Stage 1 tx encoder는 spot을 `(gene symbol, expression value)` token sequence로 본다. 이때 vocab은 단순히 “사용 가능한 gene 목록”이 아니라, 모델이 학습 자원을 어디에 쓸지 결정하는 bottleneck이다.

```text
raw gene matrix
  -> gene filtering / vocab selection
  -> gene_norm
  -> nonzero token packing
  -> sampled sequence
  -> tx_encoder
```

vocab이 너무 크면 long noisy sequence가 생기고, vocab이 너무 작으면 meaningful marker를 잃는다. 따라서 vocab size는 목표가 아니라 **representation capacity allocation의 결과**로 봐야 한다.

## 2. 현재 철학

현재 프로젝트는 모든 gene symbol을 커버하여 새로운 biomarker를 exhaustive하게 찾는 것보다, **robust spot representation**을 얻는 것을 우선한다.

따라서 기본 철학은 다음이다.

- clinically meaningful / high-information genes는 유지한다.
- 너무 드물거나 source-specific noise에 가까운 gene은 줄인다.
- housekeeping처럼 모든 spot에 흔한 gene은 rank/normalization으로 영향력을 낮춘다.
- vocab clip은 모델이 non-informative token에 capacity를 쓰지 않게 하는 장치다.

## 3. 현재 기본값

| 항목 | 기본값 | 이유 |
|---|---|---|
| gene normalization | `global_median` | Geneformer-style relative salience 반영 |
| vocab clip | `4096` | Geneformer V2 input size와 유사, 속도/표현력 균형 |
| zero removal | on | 발현되지 않은 gene은 sentence에 등장하지 않는 word처럼 취급 |
| sequence sampling | `random` | 같은 top genes만 반복되는 shortcut 방지 |
| max seq len | `512` | 속도, memory, inference consistency 균형 |
| min seq len | 너무 짧은 sequence 제외 | 문장 길이가 너무 짧으면 spot state가 불안정 |

## 4. 간단한 수식

spot `i`의 raw expression을 `x_i,g`라고 하자. global median normalization은 gene별 baseline `m_g`를 이용해 상대적 발현 salience를 만든다.

```text
v_i,g = log(1 + x_i,g / (m_g + eps))
```

또는 더 일반적으로:

```text
v_i,g = normalize_g(x_i,g)
```

이후 nonzero gene만 token으로 사용한다.

```text
S_i = {(g, v_i,g) | x_i,g > 0 and g in V}
```

sequence가 너무 길면 sampling한다.

```text
C_i ~ sample(S_i, L = max_seq_len)
```

## 5. Ablation 대상

| Axis | 후보 | 질문 |
|---|---|---|
| vocab size | `4096`, `8192`, `full` | 더 큰 vocab이 표현력을 높이는가, noise를 늘리는가? |
| normalization | `global_median`, `nonzero_z`, `none` | relative salience가 downstream에 더 좋은가? |
| max seq len | `256`, `512` | 짧은 chunk가 충분한가? 속도와 성능 trade-off는? |
| sampling | `random`, `top_k` | random partial view가 더 robust한가, top salience가 더 좋은가? |
| min seq len | low/high threshold | 짧은 sequence를 학습에서 빼는 것이 안정적인가? |
| must-include genes | on/off | marker gene coverage가 downstream에 영향을 주는가? |

현재 추천 ablation 기본값:

```text
default = global_median + vocab4096 + random512 + mask_ratio0.15 + mixed value augmentation
```

## 6. 주의할 점

### 6.1 CE loss scale

vocab size가 작으면 random CE도 작아진다.

```text
CE_random ≈ log(|V|)
```

따라서 `vocab4096`과 `full vocab`의 raw CE를 직접 비교하면 안 된다. 반드시 다음을 함께 본다.

- `masked_symbol_ce_norm = CE / log(|V|)`
- top-k accuracy
- clean MSM
- linear probe
- gene map / downstream

### 6.2 global_median과 MVM

`global_median`은 absolute count reconstruction에는 불리할 수 있다. 이것은 결함이라기보다 설계 철학의 결과다. 이 경우 value prediction은 raw target, normalized target, rank target으로 분리해서 평가해야 한다.

### 6.3 sample별 gene list 차이

샘플마다 measured gene list와 expressed gene list가 다를 수 있다. 따라서 vocab QC는 단순 global frequency만 보면 안 되고, source/sample prevalence도 같이 봐야 한다.

## 7. 얻고 싶은 결과

좋은 vocab/input pipeline은 다음 조건을 만족한다.

1. spot당 sequence가 너무 짧지 않다.
2. marker gene coverage가 유지된다.
3. MSM top-k가 안정적으로 오른다.
4. masked-HVG probe가 유지된다.
5. source leakage가 과도하지 않다.
6. Stage 1.5/Stage 2에서 distribution shift를 만들지 않는다.
