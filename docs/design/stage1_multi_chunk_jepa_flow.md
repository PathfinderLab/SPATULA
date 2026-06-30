# Stage 1 Multi-Chunk JEPA Flow

This note describes the current Stage 1 `msm_multi_chunk` objective.

Figure: `results/figures/stage1_multi_chunk_jepa_flow.png`

![Stage 1 Multi-Chunk JEPA Flow](../../results/figures/stage1_multi_chunk_jepa_flow.png)

## 목적

Stage 1의 기본 objective는 MSM(masked symbol modeling)이다. MSM은 gene-symbol semantics를 직접 학습하지만 vocab classification 문제라 수렴이 느릴 수 있다. Multi-chunk JEPA는 같은 spot 안의 서로 다른 expressed-gene chunk 사이에서 latent prediction을 수행하여, `z_spot`이 단순 gene dependency embedding을 넘어 spot-state representation으로 안정화되도록 돕는 보조 objective다.

## 데이터 흐름

### 1. 입력

- 입력은 vocab clip과 normalization 이후의 spot expression vector `hvg`다.
- zero-expressed genes는 encoder 내부에서 제거된다.
- `max_seq_len=512` 정책에서는 너무 긴 non-zero sequence가 random sampling으로 cap된다.

### 2. MSM main path

1. Student가 full spot view를 받는다.
2. 일부 gene symbol이 `[MASK]`로 바뀐다.
3. masked value에는 noise/dropout/keep 정책이 적용된다.
4. encoder가 masked symbol logits를 출력한다.
5. CE loss로 masked gene symbol을 맞춘다.

이 경로가 primary objective다.

### 3. Multi-chunk JEPA path

1. 같은 spot의 non-zero gene sequence를 random partition한다.
2. context chunks `C1, C2`와 target chunks `T1, T2`를 만든다.
3. context와 target은 최대한 disjoint하게 샘플링한다.
4. Student는 context chunks만 clean하게 encode한다.
5. `z_context = mean(z_C1, z_C2)`를 만든다.
6. EMA teacher는 target chunk만 보는 것이 아니라, full clean spot을 먼저 encode한다.
7. teacher의 full contextual per-token latent에서 target gene positions만 gather한다.
8. target token latent를 pool하고 `cls_proj + LayerNorm`을 적용해 `z_target_k`를 만든다.
9. Predictor는 `z_context + target_slot_query + target_gene_identity_query`를 입력으로 받는다.
10. target별 `z_pred_k`를 만들고, target별 Smooth-L1 loss를 평균한다.

## I-JEPA와 맞춘 부분

I-JEPA reference에서 중요한 점은 다음과 같다.

- target encoder는 full image를 본 뒤 target block latent를 잘라낸다.
- predictor는 mask token과 positional embedding을 통해 어느 target block을 예측해야 하는지 안다.
- 여러 target block을 각각 예측하고 loss를 평균한다.

현재 Stage 1 multi-chunk JEPA는 이를 gene-set setting에 맞게 바꾼다.

- full image -> full clean spot
- context block -> context gene chunks
- target block -> target gene chunks
- target position embedding -> target gene identity query
- target patch latent -> full-teacher contextual target gene latent

## Shortcut 방지

Target gene identity query는 target gene의 symbol embedding 평균만 사용한다. Expression value는 query에 들어가지 않는다. 따라서 predictor는 어떤 gene set을 예측해야 하는지는 알지만, 그 gene들의 expression value를 직접 보지는 않는다.

## 주요 로그

다음 metric으로 JEPA가 실제로 의미 있는지 확인한다.

- `val/tx_self/multi_chunk_jepa`
  - raw Smooth-L1 JEPA loss.
- `val/tx_self/multi_chunk_jepa_weighted`
  - final loss에 더해지는 weighted JEPA term.
- `val/tx_self/multi_chunk_jepa_to_msm_ratio`
  - MSM CE 대비 JEPA contribution 비율.
- `val/tx_self/multi_chunk_context_target_smoothl1`
  - predictor 없이 raw context와 target이 얼마나 가까운지.
- `val/tx_self/multi_chunk_predictor_smoothl1_gain`
  - positive이면 predictor가 raw context보다 target을 더 잘 예측한다는 뜻.
- `val/tx_self/multi_chunk_ctx_tgt_overlap`
  - context-target gene overlap 비율. 낮을수록 좋다.
- `val/tx_self/multi_chunk_target_id_query_norm`
  - target gene identity query가 실제로 활성화되는지 확인한다.

## Evaluation representation

- `z_chunk`
  - 하나의 sampled chunk embedding.
  - `val/chunk_state/*` metric으로 report된다.
- `z_spot`
  - 여러 chunk embedding의 aggregate.
  - inference/eval용 spot representation.
  - `val/spot_state/*` metric으로 report된다.
- JEPA target은 `z_spot`이 아니라 target chunk latent다.

## 해석 기준

`multi_chunk_predictor_smoothl1_gain > 0`이면 predictor가 raw context보다 target을 더 잘 맞추고 있다는 뜻이다. 이 값이 계속 0 이하라면, JEPA predictor가 유용한 latent prediction을 배우지 못하고 있을 가능성이 높다.

`multi_chunk_jepa_to_msm_ratio`가 너무 낮으면 JEPA가 loss에 거의 영향을 주지 못한다. 단, MSM CE와 JEPA Smooth-L1은 loss scale이 다르므로 raw value를 직접 비교하면 안 된다.

## Shortcut Control and Regularization

The target gene list plays the role closest to I-JEPA's target position embedding: it tells the predictor *which held-out block* should be predicted. Target expression values are deliberately not provided, because they are the held-out semantic content to infer. However, a gene list is more semantically informative than a spatial coordinate, so it can become a shortcut if used too strongly.

Current safeguards:

- `multi_chunk_target_id_scale`: scales the target gene-list query before it is added to the context representation. The default ablation setting uses `0.25` so the query identifies the target block without dominating the context signal.
- `multi_chunk_query_only_smoothl1`: measures how well the predictor can match the target using only the target slot/gene-list query and no context chunk.
- `multi_chunk_context_gain_over_query_only`: positive values mean context gene/value information improves prediction beyond the target gene list alone.
- `multi_chunk_ctx_tgt_overlap`: monitors context-target gene overlap. Values near zero indicate an I-JEPA-like disjoint setup.
- `multi_chunk_koleo_weight`: optional KoLeo entropy regularizer on context and predicted chunk latents. This discourages representational collapse and overly easy low-variance matching.

Interpretation:

- If `multi_chunk_jepa` is low but `multi_chunk_context_gain_over_query_only` is near zero or negative, the JEPA task may be mostly solved by the gene-list query and should be weakened or redesigned.
- If `multi_chunk_context_gain_over_query_only` is positive while `ctx_tgt_overlap` remains near zero, the model is using context gene/value semantics to infer target-block latent information.

