# SPATULA / mm_align Research Overview Update

작성일: 2026-06-19  
대상 프로젝트: `/workspace/mm_align`  
연결 문서: `docs/archive/contexts/spatula_research_overview.pdf`

이 문서는 기존 `spatula_research_overview.pdf`에 이어 붙이기 위한 **현재 진행상황 업데이트 context**입니다.  
기존 overview가 “spatial transcriptomics와 pathology image를 연결하기 위한 큰 연구 방향”을 설명했다면, 이 문서는 현재 코드베이스에서 실제로 구현된 stage 구조, ablation 결과, 평가 체계, 긍정적인 신호, 앞으로의 보완 방향을 직관적으로 정리합니다.

---

## 1. 한 줄 요약

현재 프로젝트는 단순히 gene expression 값을 복원하는 모델이 아니라, **spot-level RNA representation을 먼저 견고하게 만들고, 그 위에 spatial context를 얹은 뒤, 최종적으로 pathology image와 transcriptomics를 alignment하는 representation hierarchy**로 정리되고 있습니다.

```text
Stage 1   : spot / RNA encoder
Stage 1.5 : spatial context-aware encoder
Stage 2   : pathology image - transcriptomics alignment
```

가장 중요한 변화는 다음입니다.

- Stage 1의 목표를 “absolute expression value 복원”이 아니라 **relative gene salience와 robust spot representation 학습**으로 명확히 재정의했습니다.
- HyperST의 niche 개념을 `spot + neighbors(region)` 구조로 재해석했고, Stage 1.5에서 이를 Spatial-JEPA로 학습하도록 설계했습니다.
- Stage 2는 HEST/SEAL/PathBench/Loki 계열 downstream evaluation을 염두에 두고, retrieval뿐 아니라 image-to-expression prediction과 slide-level MIL까지 확장했습니다.
- `val_history.json`의 metric을 loss 중심에서 벗어나 MSM, linear probe, intrinsic, gene-gene structure, leakage, auxiliary regularization까지 해석 가능하게 정리했습니다.

보고서/발표용 figure는 아래 폴더에 정리되어 있습니다.

```text
results/figures/report/
```

대표 figure:

![Project overview](../../../results/figures/report/fig_project_overview.png)

---

## 2. 현재 연구 가설

### 2.1 기존 SPATULA 방향과의 연결

기존 overview의 핵심은 spatial transcriptomics와 pathology image 사이의 의미 있는 representation bridge를 만드는 것입니다. 현재 프로젝트에서는 이 목표를 세 단계로 쪼갰습니다.

| Stage | 질문 | 직관적 의미 |
|---|---|---|
| Stage 1 | 이 spot이 어떤 transcriptomic state를 갖는가? | spot 하나를 “gene sentence”처럼 이해하는 RNA encoder |
| Stage 1.5 | 이 spot은 주변 tissue context 안에서 어떤 의미를 갖는가? | HyperST의 niche를 `spot + neighbors`로 재해석 |
| Stage 2 | tissue image만 보고 molecular/spatial state를 얼마나 알 수 있는가? | pathology image와 RNA/spatial representation 정렬 |

즉, 현재 구조는 다음 흐름입니다.

```text
raw spot expression
  -> relative gene salience representation
  -> spatially contextualized spot representation
  -> image-aligned molecular representation
```

### 2.2 Stage 1의 핵심 가설

Stage 1은 `tx_encoder` 또는 `spot_encoder`에 해당합니다. 여기서는 하나의 spot을 `(gene symbol, expression value)` token set으로 봅니다.

핵심 가설은 다음입니다.

> 모든 gene을 완벽하게 복원하는 것보다, spot 안에서 중요한 expressed gene과 gene 간 관계, 그리고 상대적 expression salience를 잘 담는 representation이 Stage 1.5/Stage 2에 더 유용하다.

이 때문에 현재 Stage 1은 다음 선택을 선호합니다.

| 설계 | 이유 |
|---|---|
| `global_median` normalization | Geneformer처럼 absolute count보다 relative salience를 학습하기 위함 |
| `vocab_clip=4096` | 모든 symbol 커버보다 중요한 gene 중심의 dense representation을 얻기 위함 |
| `mask_ratio=0.15` | BERT/Geneformer 계열처럼 안정적인 masked modeling 시작점 |
| mixed value augmentation | masked symbol을 value shortcut으로 맞히지 않게 하기 위함 |
| clean MSM + masked-HVG probe | 단순 top-k가 아니라 representation 품질을 같이 보기 위함 |

### 2.3 Stage 1.5의 핵심 가설

Stage 1.5는 Stage 1에서 얻은 spot representation을 spatial context-aware하게 바꿉니다.

기존 HyperST가 spot과 niche를 나누어 생각했다면, 현재 프로젝트에서는 niche를 다음처럼 해석합니다.

```text
niche ≈ spot + spatial neighbors
```

그리고 i-JEPA 관점으로 보면:

```text
niche / local region = 하나의 image-like context
spot = 그 안의 patch token
```

따라서 Stage 1.5의 목적은 visible neighbor context를 보고 masked spot representation을 예측하는 것입니다.

```text
visible neighbor region  ->  predict masked spot latent
```

이 구조가 중요한 이유는 pathology image와 맞출 target이 단순 spot expression이 아니라, **주변 tissue context까지 반영한 molecular representation**이 되기 때문입니다.

### 2.4 Stage 2의 핵심 가설

Stage 2는 image encoder와 RNA/spatial encoder를 alignment합니다.

이때 단순 retrieval만 보는 것이 아니라 아래 downstream을 봐야 합니다.

| Eval | 의미 |
|---|---|
| cross-modal retrieval | image patch와 RNA spot이 서로 맞는지 |
| image-to-HVG prediction | image embedding이 molecular expression을 예측할 수 있는지 |
| image-to-relative/rank expression | absolute value가 아니라 gene salience를 맞히는지 |
| slide-level MIL | MSI/subtype 등 slide-level pathology benchmark로 확장 가능한지 |

---

## 3. 현재 구현 구조

현재 프로젝트의 stage 구조는 아래 figure로 정리했습니다.

![Project overview](../../../results/figures/report/fig_project_overview.png)

### 3.1 Stage별 산출물

| Stage | 입력 | 학습 | 산출물 |
|---|---|---|---|
| Stage 1 | HVG expression token set | MSM, optional View-JEPA/DINO | `ckpt_tx_encoder_best.pt` |
| Stage 1.5 | Stage1 spot embedding + spatial graph | Spatial-JEPA | `ckpt_spatial_best.pt` |
| Stage 2 | pathology image feature + RNA/spatial target | JEPA/CLIP/Barlow/S2L 등 | `ckpt_align_best.pt` |

### 3.2 주요 코드 위치

| 영역 | 파일 |
|---|---|
| Stage 1 encoder | `src/mm_align/models/tx/top_hvg_gene.py` |
| Stage 1 objective | `src/mm_align/objectives/tx/masked.py` |
| Stage 1 metric | `src/mm_align/evaluation/stage1_benchmarks.py` |
| Stage 1.5 spatial encoder | `src/mm_align/models/spatial/encoder.py` |
| Stage 1.5 sampler | `src/mm_align/data/spatial_sampler.py` |
| Stage 2 alignment objectives | `src/mm_align/objectives/alignment/` |
| 공통 학습 루프 | `scripts/train.py` |
| Stage1->Stage1.5 cascade | `scripts/run_all.sh` |

---

## 4. 데이터와 vocab 진행상황

현재 데이터는 HEST, ST1K, SpatialCorpus를 prepared shard 형태로 통합해 사용합니다. 보고서용 데이터셋 figure는 아래와 같습니다.

![Dataset overview](../../../results/figures/report/fig_dataset_overview.png)

### 4.1 데이터셋 관점에서 긍정적인 부분

- 여러 source를 통합했기 때문에 단일 dataset에 과적합된 RNA encoder가 아니라, 더 일반적인 spot representation을 만들 수 있습니다.
- source가 섞이면서 학습 난이도는 올라갔지만, Stage 2 alignment로 갈 때 더 현실적인 foundation setting이 됩니다.
- sequence length와 vocab hit rate를 별도로 QC하고 있어, 특정 source가 지나치게 짧은 token sequence를 만드는지 확인할 수 있습니다.

![Sequence QC](../../../results/figures/report/fig_sequence_qc.png)

### 4.2 Vocab 설계

현재 vocab 설계는 “모든 gene을 다 커버한다”보다 “spot representation에 정보량이 높은 gene을 안정적으로 사용한다”에 가깝습니다.

![Vocab overview](../../../results/figures/report/fig_vocab_overview.png)

핵심 해석:

- full vocab은 더 많은 symbol을 포함하지만, long noisy sequence를 만들 수 있습니다.
- `vocab_clip=4096`은 새로운 biomarker discovery 목적보다는 robust spot representation 목적에 더 적합합니다.
- vocab clip을 쓰면 CE loss scale이 작아질 수 있으므로 raw CE만 보면 안 되고, `ce_norm`, top-k, probe metric을 함께 봐야 합니다.

![Vocab filter quality](../../../results/figures/report/fig_vocab_filter_quality.png)

---

## 5. Stage 1 진행상황

### 5.1 Stage 1의 현재 기본 방향

현재 가장 중요한 Stage 1 candidate는 다음 계열입니다.

```text
global_median normalization
+ vocab_clip 4096
+ max_seq_len 1024 random sampling
+ mask_ratio 0.15
+ mixed value augmentation
+ MSM primary objective
```

이 조합은 absolute value reconstruction에서는 불리할 수 있지만, 프로젝트의 핵심 목적에 더 가까운 **relative salience / expression manifold / spot representation** 측면에서 긍정적인 신호를 보였습니다.

![Normalization before after](../../../results/figures/report/fig_norm_before_after.png)

### 5.2 주요 결과: baseline vs rel4096

| Metric | baseline | rel4096 | 해석 |
|---|---:|---:|---|
| HVG Spearman probe | 0.5223 | 0.5319 | rel4096가 expression rank signal을 약간 더 잘 보존 |
| spot-rank Spearman | 0.4001 | 0.5202 | rel4096가 spot 내부 gene 상대순위에서 뚜렷하게 우세 |
| expression distance Spearman | 0.1907 | 0.7144 | rel4096 embedding geometry가 expression manifold를 강하게 보존 |
| gene embedding corr Spearman | 0.2159 | 0.1771 | gene embedding co-expression alignment는 baseline이 약간 우세 |
| MVM Spearman | 0.0505 | -0.1038 | rel4096는 absolute value reconstruction과는 잘 맞지 않음 |

직관적 해석:

- baseline은 value reconstruction 쪽에서는 더 자연스러울 수 있습니다.
- rel4096는 “어떤 gene이 이 spot에서 상대적으로 중요한가”를 더 잘 보존합니다.
- 따라서 rel4096는 imputation model이라기보다 **spot representation foundation**으로 해석하는 것이 맞습니다.

### 5.3 Ablation 요약

현재 ablation 결과는 아래 report figure로 정리했습니다.

![Ablation summary](../../../results/figures/report/fig_ablation_summary.png)

주요 해석:

| Ablation | 관찰 | 긍정적 의미 |
|---|---|---|
| normalization | `global_median`이 rank/manifold 측면에서 강함 | Geneformer-style relative expression 가정과 잘 맞음 |
| vocab clip | clip4096/8192가 full vocab 대비 손해가 크지 않음 | dense vocab이 spot representation에 충분할 수 있음 |
| value augmentation | mixed/noise/dropout은 MSM top-k를 낮출 수 있으나 manifold/probe에는 도움 | shortcut 방지 목적과 부합 |
| mask ratio | 0.15를 보수적 default로 채택 | 과도한 masking으로 context를 깨지 않음 |
| JEPA/DINO auxiliary | direct DINO/KoLeo는 조심 필요, View-JEPA predictor가 더 안정적 후보 | auxiliary는 primary MSM을 대체하지 않고 보조로 사용해야 함 |

---

## 6. Stage 1 평가 체계 변화

초기에는 organ probe 같은 쉬운 task가 성능을 좋아 보이게 만들 수 있었습니다. 현재는 Stage 1의 목표에 맞춰 평가를 재정리했습니다.

### 6.1 현재 Stage 1에서 먼저 보는 지표

| 지표 | 직관 | 방향 |
|---|---|---|
| `masked_symbol_top10_acc` | 가려진 gene symbol을 top-10 안에 맞추는가 | 높을수록 좋음 |
| `clean_msm/top10_acc` | augmentation 없이도 symbol context를 이해하는가 | 높을수록 좋음 |
| `masked_hvg/spearman_mean` | target gene을 가려도 expression signal이 embedding에 남는가 | 높을수록 좋음 |
| `hvg_rank/spot_rank_spearman` | spot 내부 gene 상대 순위를 보존하는가 | 높을수록 좋음 |
| `gene_embedding/corr_spearman` | co-expression gene끼리 embedding도 가까운가 | 높을수록 좋음 |
| `effective_rank` | embedding이 collapse하지 않았는가 | 너무 낮으면 위험 |

더 자세한 설명은 아래 문서에 정리했습니다.

```text
docs/design/stage1_validation_metrics_kr.md
```

### 6.2 `distance_spearman`에 대한 해석 수정

`distance_spearman`은 spot 간 expression 거리와 embedding 거리의 rank correlation입니다.  
즉, expression이 비슷한 spot들이 embedding에서도 가까운지 보는 지표입니다.

하지만 중요한 점은:

> MSM은 masked gene symbol을 맞히는 objective이지, nearest-neighbor 구조 보존을 직접 학습하는 objective가 아닙니다.

따라서 Stage 1에서 `distance_spearman`은 유용한 보조 지표이지만, Stage 1의 primary selection metric은 아닙니다. spatial neighbor 구조 보존은 Stage 1.5에서 더 직접적으로 봐야 합니다.

---

## 7. DINO / View-JEPA / KoLeo 관련 변경점

### 7.1 기존 아이디어

이미지 self-supervised learning에서처럼 teacher view와 student view를 둘 수 있습니다.

```text
view_1: clean/raw expression view  -> teacher
view_2: masked + noisy expression view -> student
```

처음에는 DINO-style consistency와 KoLeo regularizer를 넣어 실험했습니다.

### 7.2 관찰된 문제

DINO/KoLeo를 너무 일찍 또는 강하게 켜면 다음 현상이 나타날 수 있습니다.

- MSM 자체는 좋아지는 듯 보이지만,
- expression geometry나 linear probe가 흔들릴 수 있고,
- `distance_spearman`이나 representation health가 떨어질 수 있습니다.

직관적으로는, Stage 1의 목적이 image instance discrimination이 아니라 **gene context를 통한 spot representation**이기 때문에, direct DINO consistency가 항상 잘 맞지는 않을 수 있습니다.

### 7.3 현재 보완 방향

현재는 다음 방향으로 정리했습니다.

| 선택 | 이유 |
|---|---|
| `msm_only` 유지 | 가장 해석이 쉽고 안정적인 baseline |
| DINO/KoLeo warmup | 처음부터 auxiliary를 켜지 않고 MSM이 어느 정도 자리 잡은 뒤 적용 |
| View-JEPA predictor head | student representation을 바로 teacher에 붙이지 않고 predictor가 clean latent를 예측하게 함 |
| KoLeo는 작은 weight부터 | collapse 방지용이지만 너무 강하면 biological similarity를 밀어낼 수 있음 |

현재 추천 grid:

```text
A0: msm_only
A1: view_jepa_w005
A2: view_jepa_w010
A3: dino_late_no_koleo
A4: spatula_mid
A5: spatula_large
```

---

## 8. Legacy 코드 비교에서 얻은 시사점

`/workspace`와 `/workspace/spatula`의 legacy 코드를 비교한 결과, 과거 MSM/MGM 성능이 더 좋아 보였던 이유는 다음일 가능성이 큽니다.

| Legacy 특징 | 현재 해석 |
|---|---|
| globalnorm tensor 사용 | 이미 global-median류 normalization이 반영됐을 가능성 |
| masked gene value를 더 많이 유지 | symbol prediction이 쉬워지는 shortcut 가능성 |
| 더 큰 model capacity | MSM top-k 성능을 직접 올릴 수 있음 |
| 약한 augmentation | clean symbol prediction에는 유리 |
| dataset 구성이 다름 | 현재 HEST/ST1K/SpatialCorpus 혼합보다 쉬웠을 수 있음 |

따라서 legacy를 그대로 복귀하기보다, 현재 코드에 아래 옵션을 추가했습니다.

```bash
STAGE1_CAPACITY=spatula_lite
STAGE1_CAPACITY=spatula_mid
STAGE1_CAPACITY=spatula_large
```

| Capacity profile | Token dim | Layers | Heads | 자동 batch/rank | 용도 |
|---|---:|---:|---:|---:|---|
| `spatula_lite` | 256 | 4 | 4 | 512 | 빠른 기준선 |
| `spatula_mid` | 384 | 6 | 6 | 384 | 현재 가장 유력한 main capacity 후보 |
| `spatula_large` | 512 | 6 | 8 | 256 | capacity upper-bound 확인 |

이 옵션은 `h_tx=512` 출력은 유지하면서 내부 token transformer capacity만 키웁니다.  
즉 Stage 1.5/Stage 2 연결부를 깨지 않고, **capacity 효과만 분리해서 검증**할 수 있습니다.

실행 예:

```bash
STAGE1_OBJECTIVE=msm_only STAGE1_CAPACITY=spatula_mid STAGE1_TAG=stage1_main_msm_only_spatula_mid bash scripts/run_all.sh
```

---

## 9. Stage 1.5 진행상황

Stage 1.5는 Stage 1 checkpoint를 받아 spatial context-aware encoder를 학습합니다.

현재 설계:

| 항목 | 현재 선택 | 직관 |
|---|---|---|
| input | Stage1 frozen spot embedding | spot 자체의 transcriptomic state |
| context | spatial neighbors / region | HyperST의 niche에 해당 |
| objective | Spatial-JEPA | visible context로 masked spot latent 예측 |
| token mode | separate | spot token과 region token을 구분 |
| evaluation | spatial kNN, gene map SCC, qualitative maps | spatially informative한지 확인 |

중요한 구현 보완:

- Stage 1.5 OOM 원인은 batch size 자체보다 frozen Stage1 encoder가 region HVG를 한 번에 처리하던 경로였습니다.
- `tx_encode_batch` micro-batch를 넣어 이 문제를 완화했습니다.

실행 예:

```bash
STAGE1_CKPT=results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt TAG=stage15_rel4096_spatial EPOCHS=30 bash scripts/train/stage15_main.sh
```

OOM 발생 시:

```bash
STAGE1_CKPT=results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt TAG=stage15_rel4096_spatial EPOCHS=30 TX_ENCODE_BATCH=128 BATCH_SIZE=8 bash scripts/train/stage15_main.sh
```

---

## 10. Stage 1.5 / Stage 2에서 추가된 qualitative evaluation

정량 metric뿐 아니라, 보고서와 연구 해석을 위해 정성적 figure가 중요합니다.

### 10.1 Stage 1.5 gene map evaluation

목표:

> 특정 marker gene의 실제 spatial expression map과, Stage 1.5 embedding에서 예측한 expression map을 비교한다.

평가 방식:

1. Stage 1.5 spatial encoder를 freeze합니다.
2. train split에서 `z_spatial -> selected gene expression` Ridge probe를 학습합니다.
3. test sample에서 gene별 GT map과 predicted map을 그립니다.
4. Spearman SCC로 공간적 상대 패턴 보존을 측정합니다.

실행 예:

```bash
python scripts/eval/stage15_gene_map.py   --stage1-ckpt results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt   --spatial-ckpt results/runs/stage15_rel4096_spatial/ckpt_spatial_best.pt   --split test   --genes MKI67 EPCAM COL1A1 CD3D   --probe-train-samples 20   --max-train-spots 20000
```

직관적 의미:

- GT와 prediction map이 비슷하면, spatial encoder가 단순 expression vector가 아니라 **공간적 molecular pattern**을 담는다는 뜻입니다.
- SCC는 absolute scale보다 spot 간 상대적 spatial pattern을 보므로, Stage 1의 relative salience 철학과도 잘 맞습니다.

### 10.2 Stage 2 downstream

Stage 2는 다음 평가를 중심으로 보강되었습니다.

| Eval | 의미 | 스크립트 |
|---|---|---|
| zero-shot retrieval | image와 RNA가 같은 spot끼리 가까운가 | `scripts/eval/zero_shot.py` |
| image-to-HVG probe | image embedding으로 gene expression을 예측할 수 있는가 | `scripts/eval/linear_probe.py` |
| slide-level MIL | MSI/subtype 같은 slide-level task로 확장 가능한가 | `scripts/eval/slide_mil.py` |

Stage 2 ablation 축:

| 축 | 후보 |
|---|---|
| image encoder | UNI/UNI2, H0-mini, GigaPath |
| method | JEPA, CLIP, Barlow, S2L |
| tx target | ours Stage1, Stage1.5 spatial target, Novae baseline |
| downstream | HEST expression, MSI/subtype, PathBench-style MIL |

---

## 11. 현재까지의 긍정적인 메시지

현재 결과를 부정적으로 보면 “absolute MVM이 낮다” 또는 “DINO가 바로 잘 붙지 않는다”로 볼 수 있습니다.  
하지만 연구 목표 관점에서는 더 긍정적인 해석이 가능합니다.

### 11.1 방향성이 명확해졌다

초기에는 organ probe처럼 쉬운 task가 성능을 좋아 보이게 만들 수 있었습니다.  
지금은 Stage 1의 목적을 다음처럼 더 정확히 맞췄습니다.

```text
좋은 Stage1 = gene symbol을 잘 맞히는 것
            + target gene shortcut 없이 expression signal을 담는 것
            + gene-gene relation을 embedding에 반영하는 것
            + Stage1.5/Stage2에 넘길 수 있는 robust h_tx를 만드는 것
```

### 11.2 rel4096 결과가 연구 가설과 맞는다

`global_median + clip4096`은 absolute expression 복원에서는 약할 수 있습니다.  
하지만 spot 내부 gene rank와 expression manifold 보존에서는 baseline보다 좋은 신호를 보였습니다.

이는 다음 가설과 잘 맞습니다.

> 우리는 expression imputation model을 만드는 것이 아니라, image-alignment와 spatial reasoning에 유용한 transcriptomics representation을 만들고 있다.

### 11.3 HyperST식 spatial context를 JEPA로 재해석했다

HyperST의 niche 개념을 그대로 contrastive latent space에 강제하기보다, 현재 프로젝트에서는 JEPA 방식으로 context prediction을 사용합니다.

```text
HyperST: spot + niche representation
Current: spot + neighbors -> Spatial-JEPA target prediction
```

이 방식은 latent space를 억지로 contrastive하게 묶지 않고, spatial context를 표현학습 objective로 자연스럽게 넣는 장점이 있습니다.

### 11.4 Stage 2 evaluation이 훨씬 현실적으로 확장됐다

단순 retrieval만으로는 pathology-RNA alignment가 downstream에 유용한지 알기 어렵습니다.  
현재는 image-to-expression, marker map, slide-level MIL까지 연결되므로, HEST/SEAL/PathBench류 benchmark와 연결하기 쉬운 형태가 되었습니다.

---

## 12. 남은 리스크와 대응 방향

| 리스크 | 현재 해석 | 대응 |
|---|---|---|
| MSM top-k가 shortcut으로 좋아질 수 있음 | masked value keep이 강하면 쉬운 문제 가능 | masked/unmasked value augmentation 분리, masked-HVG probe 사용 |
| DINO/KoLeo가 geometry를 흔들 수 있음 | image SSL 방식이 RNA token set에 그대로 맞지는 않음 | warmup, View-JEPA predictor, 작은 weight ablation |
| MVM이 낮음 | global_median은 absolute value 복원과 목적이 다름 | rank/Spearman/relative target을 함께 평가 |
| vocab clip이 gene discovery에는 제한적 | 목적이 biomarker discovery가 아니라 robust spot representation | full/8192/4096 ablation으로 trade-off 확인 |
| Stage1.5 OOM | frozen tx encoder가 region HVG를 크게 처리 | `tx_encode_batch`로 micro-batch 처리 |
| Stage2 downstream label 부족 | evaluator는 준비됐지만 label CSV 필요 | HEST/PathBench label 연결 필요 |

---

## 13. 다음 실험 우선순위

### 13.1 Stage 1

우선순위:

1. `msm_only`를 안정적인 기준선으로 유지합니다.
2. `spatula_mid` capacity를 먼저 돌려 model size 효과를 확인합니다.
3. `view_jepa_w005`, `view_jepa_w010`으로 predictor-based consistency가 도움이 되는지 확인합니다.
4. direct DINO/KoLeo는 warmup이 있는 후보만 제한적으로 봅니다.

추천 실행:

```bash
# 안정 baseline
STAGE1_OBJECTIVE=msm_only STAGE1_CAPACITY=spatula_lite STAGE1_TAG=stage1_main_msm_only_spatula_lite bash scripts/run_all.sh

# SPATULA capacity effect
STAGE1_OBJECTIVE=msm_only STAGE1_CAPACITY=spatula_mid STAGE1_TAG=stage1_main_msm_only_spatula_mid bash scripts/run_all.sh

# view-JEPA candidate
STAGE1_OBJECTIVE=view_jepa_w005 STAGE1_CAPACITY=spatula_lite STAGE1_TAG=stage1_main_view_jepa_w005 bash scripts/run_all.sh
```

### 13.2 Stage 1.5

Stage 1에서 선택된 checkpoint를 넘겨 Spatial-JEPA를 학습합니다.

```bash
STAGE1_CKPT=results/runs/<stage1_tag>/ckpt_tx_encoder_best.pt TAG=stage15_<stage1_tag> EPOCHS=30 bash scripts/train/stage15_main.sh
```

### 13.3 Stage 2

Stage 2는 아래 순서가 좋습니다.

1. Stage1 target 기반 image-RNA alignment
2. Stage1.5 spatial target 기반 alignment
3. image encoder ablation: UNI/UNI2 -> H0-mini -> GigaPath
4. method ablation: JEPA -> CLIP -> Barlow/S2L
5. HEST expression + slide-level MIL downstream 연결

---

## 14. 이 문서를 context로 사용할 때의 핵심 문장

아래 문장들이 현재 연구 진행상황을 가장 짧게 설명합니다.

1. 현재 Stage 1은 expression imputation이 아니라, relative gene salience를 담는 spot/RNA foundation encoder로 재정의되었다.
2. `global_median + vocab_clip4096`은 absolute MVM에는 불리하지만, spot 내부 gene rank와 expression manifold 측면에서 긍정적 신호를 보였다.
3. HyperST의 niche 개념은 현재 프로젝트에서 `spot + neighbors`로 재해석되며, Stage 1.5에서는 이를 Spatial-JEPA로 학습한다.
4. DINO/KoLeo는 직접 적용 시 representation geometry를 흔들 수 있어, 현재는 `msm_only` 기준선과 View-JEPA predictor 후보를 중심으로 본다.
5. Stage 2는 retrieval뿐 아니라 image-to-expression, marker spatial map, slide-level MIL까지 평가를 확장해 실제 pathology downstream과 연결한다.
6. legacy 코드의 높은 MSM 성능은 globalnorm, 약한 augmentation, 큰 capacity, masked value shortcut 가능성의 영향이 있으므로, 현재는 capacity effect만 분리해 ablation한다.

---

## 15. 관련 파일

| 목적 | 경로 |
|---|---|
| 프로젝트 가이드 | `PROJECT_GUIDE_KR.md` |
| Stage1 metric 설명 | `docs/design/stage1_validation_metrics_kr.md` |
| 연구 진행 리포트 | `docs/reports/research_progress_report_kr.md` |
| 보고서용 figure manifest | `results/figures/report/figure_manifest.md` |
| Stage1->Stage1.5 cascade | `scripts/run_all.sh` |
| Stage1 ablation | `scripts/ablation/run_all.sh` |
| report figure 생성 | `scripts/viz/report_figures.py` |

---

## 16. Figure 목록

| Figure | 파일 | 용도 |
|---|---|---|
| Project overview | `results/figures/report/fig_project_overview.png` | 전체 stage 구조 설명 |
| Dataset overview | `results/figures/report/fig_dataset_overview.png` | 데이터 source 구성 설명 |
| Vocab overview | `results/figures/report/fig_vocab_overview.png` | vocab 설계와 health 설명 |
| Sequence QC | `results/figures/report/fig_sequence_qc.png` | sequence length / vocab hit rate 설명 |
| Ablation summary | `results/figures/report/fig_ablation_summary.png` | Stage1 ablation 핵심 결과 설명 |
| Normalization | `results/figures/report/fig_norm_before_after.png` | global-median normalization 직관 설명 |
| Vocab filter QC | `results/figures/report/fig_vocab_filter_quality.png` | vocab filtering 품질 설명 |
