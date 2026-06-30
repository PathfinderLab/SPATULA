# RNA Foundation Model: MSM + JEPA Pretraining Design

## Overview

본 연구는 Spatial Transcriptomics 및 Single-cell Transcriptomics 데이터의 non-zero expressed gene sequence를 입력으로 사용하는 Foundation Model을 목표로 한다.

기본 목표는 다음과 같다.

1. Gene Symbol Prediction (Masked Symbol Modeling)
2. Gene Module Representation Learning (JEPA)
3. Spot-level Embedding Learning
4. 향후 WSI-RNA Alignment를 위한 표현 학습

---

# 1. Input Representation

각 Spot은 다음과 같은 Token으로 표현된다.

```text
(Symbol, Expression)
```

예시:

```text
(TP53, 8.2)
(MKI67, 5.4)
(CD3D, 7.1)
...
```

여기서

- Symbol = Semantic Token
- Expression = Relative Position / Importance Signal

로 해석한다.

---

# 2. Expression의 역할

본 연구에서는 Expression Value를 단순 Attribute가 아니라 일종의 Position Signal로 가정한다.

NLP:

```text
Word + Positional Encoding
```

RNA:

```text
Gene Symbol + Expression Encoding
```

즉,

```text
Expression
≈
Position Information
```

이라는 가설을 검증한다.

---

# 3. Non-Zero Gene Sequence Construction

Spot의 Non-zero Gene만 사용한다.

```text
Spot
↓
Non-zero Genes
↓
Gene Sequence
```

Zero Expression Gene은 제외한다.

---

# 4. Working Sequence Sampling

전체 Non-zero Gene을 매번 사용하는 것은 비효율적이다.

따라서 Spot마다 Working Sequence를 샘플링한다.

예시:

```text
Non-zero Gene Count = 6000

↓

Working Sequence = 2048
```

추천 구성:

| Type | Ratio |
|--------|--------|
| High Expression | 40% |
| Random Non-zero | 30% |
| HVG / Marker | 20% |
| Low Expression | 10% |

---

# 5. Primary Objective: Masked Symbol Modeling (MSM)

## Motivation

Expression으로부터 Symbol을 예측하도록 학습한다.

입력:

```text
(TP53, 8.2)
```

↓

```text
(MASK, 8.2)
```

학습 목표:

```text
Expression + Context
→
Gene Symbol Prediction
```

---

## MSM Mask Strategy

```text
Mask Symbol Only
Keep Expression
```

예시:

```text
(TP53, 8.2)
↓

(MASK, 8.2)
```

Mask Ratio:

```text
15% ~ 30%
```

---

# 6. Auxiliary Objective: JEPA

## Motivation

Gene Identity Prediction을 넘어서

```text
Gene Module
↔
Gene Module
```

관계를 학습한다.

---

# 7. JEPA Data Construction

Working Sequence:

```text
S = 2048 genes
```

예시:

```text
[G1 G2 G3 G4 G5 G6 G7 G8]
```

Target Selection:

```text
40~60%
```

예시:

```text
Target:
[G3 G4 G5]

Context:
[G1 G2 G6 G7 G8]
```

---

# 8. Context Construction

본 연구의 핵심 가설:

```text
Expression = Position Information
```

따라서

Target Gene은 제거하지 않는다.

대신

```text
Symbol → MASK
Expression → KEEP
```

를 사용한다.

예시:

```text
(TP53, 8.2)
↓

(MASK, 8.2)
```

---

# 9. JEPA Architecture

## Student Encoder

입력:

```text
Context Sequence
```

예시:

```text
(G1, v1)
(G2, v2)
(MASK, v3)
(MASK, v4)
...
```

---

## Teacher Encoder (EMA)

입력:

```text
Original Target Genes
```

예시:

```text
(G3, v3)
(G4, v4)
(G5, v5)
```

---

## Predictor

```text
Context Representation
↓
Predictor
↓
Target Representation
```

---

# 10. JEPA Loss

```text
L_JEPA
=
MSE(
z_pred,
z_target
)
```

또는

```text
Cosine Similarity Loss
```

사용 가능.

---

# 11. Collapse Prevention

JEPA 학습 시 Collapse를 방지하기 위해

```text
VICReg
```

사용을 권장한다.

Loss:

```text
L
=
L_MSM
+
λ L_JEPA
+
β L_VICReg
```

---

# 12. Training Strategy

## Stage-free Training

별도의 Stage 분리 없이

Warmup Scheduling 사용.

---

## Epoch 0 ~ E1

MSM Only

```text
L
=
L_MSM
```

---

## Epoch E1 ~ E2

MSM + Weak JEPA

```text
L
=
L_MSM
+
0.1 L_JEPA
```

---

## Epoch E2 ~

MSM + Strong JEPA

```text
L
=
L_MSM
+
0.3~1.0 L_JEPA
```

---

# 13. Why One-Step Scheduling?

완전한 Stage 분리보다

```text
MSM
→
MSM + JEPA
→
JEPA 강화
```

형태가 더 자연스럽다.

이유:

1. Encoder가 이미 안정적인 Symbol Representation을 확보
2. JEPA가 Gene Module Representation을 추가 학습
3. 두 Objective가 동일 Encoder를 공유
4. 학습 비용 감소

---

# 14. Proposed Final Pipeline

```text
Spot
↓
Non-zero Gene Sequence
↓
Working Sequence Sampling
↓
MSM Masking
↓
JEPA Target Sampling
↓
Shared Encoder

 ├── MSM Head
 │
 └── JEPA Predictor

↓
Joint Optimization
```

Loss:

```text
L_total
=
L_MSM
+
λ(t) L_JEPA
+
β L_VICReg
```

---

# 15. Future Extensions

## Spot-Level Foundation Model

```text
Spot Embedding
```

학습

---

## Sample-Level Encoder

JEPA 기반 Spot Aggregation

```text
Spot
↓
Sample Encoder
↓
Sample Embedding
```

---

## WSI-RNA Alignment

```text
UNI Encoder
+
RNA Encoder
```

JEPA 기반 Cross-modal Alignment

---

# Key Hypothesis

본 연구의 핵심 가설은 다음과 같다.

```text
Gene Symbol = Word

Expression = Position Signal
```

따라서

```text
Mask Symbol
Keep Expression
```

전략이 RNA Foundation Model에서
효과적인 Representation Learning을 제공할 것이다.