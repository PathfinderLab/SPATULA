# 연구 진행 리포트: Spatial Transcriptomics–Pathology Alignment

작성일: 2026-06-18  
프로젝트: `/workspace/mm_align`

## 1. Executive Summary

현재 프로젝트는 **spot-level transcriptomics representation을 먼저 견고하게 만들고(Stage 1), 이후 spatial context-aware encoder(Stage 1.5)를 통해 spot + neighbor/global context를 반영한 representation으로 확장한 뒤, 최종적으로 pathology image와 transcriptomics를 alignment하는 Stage 2**로 진행 중이다.

현재까지 가장 중요한 진전은 다음과 같다.

- Stage 1 RNA/spot encoder는 단순 organ classification 같은 쉬운 태스크가 아니라, masked symbol modeling, expression manifold preservation, HVG linear probe, relative rank probe, gene embedding correlation 등으로 평가 체계를 재정비했다.
- Geneformer-style `global_median` normalization + vocab clip 4096 + mixed value augmentation 조합(`stage1_full_rel4096`)은 absolute value reconstruction만 보면 불리하지만, **relative expression salience와 expression manifold 보존 측면에서는 baseline보다 더 긍정적인 신호**를 보였다.
- Stage 1.5는 HyperST의 spot/niche 관점을 `spot + neighbors(region)`로 재해석하고, JEPA 방식으로 visible region context가 masked spot representation을 예측하도록 구성했다.
- Stage 2는 Loki/HEST/SEAL/PathBench류 downstream을 반영할 수 있도록 image-to-expression linear probe, cross-modal retrieval, slide-level MIL task 설계 및 evaluator가 추가되었다.

요약하면, 현재 결과는 “absolute expression value를 그대로 복원하는 모델”보다 **relative salience, spatial context, multimodal alignment를 위한 representation foundation**이라는 프로젝트의 방향성과 더 잘 맞는 쪽으로 정리되고 있다.

## 2. 연구 목표와 Stage 정의

| Stage | 목적 | 핵심 질문 | 현재 상태 |
|---|---|---|---|
| Stage 1 | Spot/RNA encoder | spot의 expressed gene set과 상대적 발현 salience를 잘 표현하는가? | full baseline/rel4096 완료, 다수 ablation 완료 |
| Stage 1.5 | Spatial context-aware encoder | spot 자체뿐 아니라 neighbor/region context를 통해 spatially informative representation을 만들 수 있는가? | rel4096 ckpt 기반 실행 준비, OOM 경로 수정 완료 |
| Stage 2 | Image-transcriptomics alignment | pathology image embedding이 RNA/spatial representation과 정렬되고 downstream task에 유용한가? | retrieval/linear probe/eval scaffold 구축, 일부 기존 alignment run 존재 |

## 3. 현재 RUNS 현황

### Stage 1 완료/진행 RUNS

주요 완료 run:

| Run | 목적 | 상태 |
|---|---|---|
| `stage1_full_baseline` | nonzero_z + clip4096 + keep value augmentation | 완료, test CSV 존재 |
| `stage1_full_rel4096` | global_median + clip4096 + mixed value augmentation | 완료, test CSV 존재 |
| `stage1_full_rel8192` | global_median + clip8192 + mixed value augmentation | metadata만 존재, 실패/중단 상태로 판단 |
| `stage1_norm_*` | normalization ablation | 완료 |
| `stage1_vocab_*` | vocab clip size ablation | 완료 |
| `stage1_va_*` | value augmentation ablation | 완료 |
| `stage1_mask_*` | mask ratio ablation | 완료 |
| `stage1_jepa_*` | Gene-JEPA auxiliary ablation | 완료 |
| `stage1_samp_*` | sequence sampling ablation | 일부 완료 |

현재 full core ablation 설계는 `scripts/ablation/run_all.sh` 기준으로 다음 가정을 검증한다.

| Candidate | Normalization | Vocab | Seq cap | Value aug | JEPA | 의미 |
|---|---:|---:|---:|---|---|---|
| baseline | nonzero_z | 4096 | 1024 random | keep | off | 안정적인 표현 baseline |
| rel4096 | global_median | 4096 | 1024 random | mixed | off | Geneformer-style relative salience 가정 검증 |
| rel8192 | global_median | 8192 | 1024 random | mixed | off | 더 큰 informative vocab의 이득 확인 |
| rel4096_jepa | global_median | 4096 | 1024 random | mixed | lite | relative salience + Gene-JEPA 보조효과 확인 |

주의: `stage1_full_baseline`은 과거 `batch/rank=1024`, `stage1_full_rel4096`은 OOM 대응 이후 `batch/rank=512`로 실행되었다. 따라서 최종 논문/보고용 엄밀 비교는 같은 batch 설정으로 baseline 재실행이 필요하다.

## 4. Stage 1 주요 결과와 긍정적 해석

### 4.1 Full baseline vs rel4096 test 결과

출처:

- `results/eval/stage1_test_stage1_full_baseline.csv`
- `results/eval/stage1_test_stage1_full_rel4096.csv`

| Metric | baseline | rel4096 | 해석 |
|---|---:|---:|---|
| `linear_probe_hvg_spearman_mean` | 0.5223 | 0.5319 | rel4096가 HVG expression의 rank/order signal을 더 잘 보존 |
| `linear_probe_hvg_rank_spot_rank_spearman` | 0.4001 | 0.5202 | rel4096가 spot 내부 gene 상대순위 복원에서 강한 개선 |
| `linear_probe_hvg_rank_top10_overlap` | 0.5191 | 0.5005 | top salience gene overlap은 baseline과 유사한 수준 유지 |
| `intrinsic_expression_distance_spearman` | 0.1907 | 0.7144 | rel4096 embedding geometry가 expression manifold를 훨씬 잘 보존 |
| `intrinsic_gene_embedding_corr_spearman` | 0.2159 | 0.1771 | gene embedding co-expression alignment는 baseline이 약간 우세 |
| `mvm_spearman` | 0.0505 | -0.1038 | rel4096는 absolute masked value reconstruction에는 불리 |
| `mvm_r2` | -0.0259 | -2.8720 | absolute value scale prediction은 rel4096의 주 목적과 맞지 않음 |

가장 긍정적인 포인트는 `rel4096`이 **absolute expression value reconstruction에서는 약하지만, 프로젝트의 핵심 가정인 relative salience와 expression manifold 보존에서는 baseline보다 강한 신호를 보인다**는 점이다.

이는 현재 Stage 1을 “gene expression imputation model”로 보기보다, **spot representation foundation model**로 보는 해석과 잘 맞는다. 특히 `global_median` normalization은 특정 gene/spot의 absolute scale보다 gene token의 상대적 중요도와 rank를 모델링하려는 의도였고, 실제로 rank-based probe와 expression manifold metric에서 긍정적인 방향을 보였다.

### 4.2 Normalization ablation 해석

주요 validation 경향:

| Run | Top10 acc | Linear probe Pearson | Expression distance Spearman | 해석 |
|---|---:|---:|---:|---|
| `stage1_norm_none` | 0.0469 | 0.4577 | 0.2610 | symbol modeling에는 부적합, raw scale 영향 큼 |
| `stage1_norm_nonzero_z` | 0.4041 | 0.4546 | 0.6888 | 안정적인 baseline |
| `stage1_norm_global_median` | 0.5944 | 0.3664 | 0.7144 | symbol/rank/manifold 쪽 강점, absolute value probe는 약함 |

긍정적 해석:

- `global_median`은 masked symbol top-k와 expression manifold에서 강하다.
- 이는 Geneformer식 “relative expression salience as language-like token order” 가정과 부합한다.
- 다만 absolute value prediction은 약해질 수 있으므로 Stage 2 image-to-expression에서는 raw/normalized target을 분리해 평가해야 한다.

### 4.3 Vocab clip ablation 해석

주요 validation 경향:

| Run | Top10 acc | Linear probe Pearson | Expression distance Spearman | 해석 |
|---|---:|---:|---:|---|
| `stage1_vocab_clip2048` | 0.4158 | 0.4456 | 0.5905 | 작고 빠르지만 표현 다양성 제한 가능 |
| `stage1_vocab_clip4096` | 0.4069 | 0.4524 | 0.6588 | 효율과 표현력 균형이 좋음 |
| `stage1_vocab_clip8192` | 0.4123 | 0.4429 | 0.6967 | manifold 보존은 좋지만 compute/memory 부담 증가 |
| `stage1_vocab_full` | 0.4150 | 0.4437 | 0.6159 | full vocab이 항상 우월하지는 않음 |

긍정적 해석:

- full vocab보다 clip vocab이 representation 관점에서 손해가 크지 않다.
- `clip4096`은 Geneformer V2의 4096 input size와도 해석적으로 맞고, compute/memory 효율이 좋다.
- `clip8192`는 manifold 보존 가능성이 있으나 Stage1 full run에서 실패/중단된 상태라, batch/sequence memory 안정화 후 재검증이 필요하다.

### 4.4 Value augmentation ablation 해석

| Run | Top10 acc | Clean MSM Top10 | Linear probe Pearson | Expression distance Spearman | 해석 |
|---|---:|---:|---:|---:|---|
| `stage1_va_keep` | 0.3910 | 0.3919 | 0.4464 | 0.6328 | clean symbol prediction에는 안정적 |
| `stage1_va_mixed` | 0.2979 | 0.3622 | 0.4525 | 0.7270 | shortcut 방지와 representation manifold 측면에서 긍정적 |
| `stage1_va_noise` | 0.2948 | 0.3580 | 0.4385 | 0.7154 | mixed와 유사한 regularization 효과 |
| `stage1_va_dropout` | 0.2992 | 0.3598 | 0.4449 | 0.7277 | expression manifold 보존이 좋음 |

긍정적 해석:

- value augmentation은 top-k symbol accuracy를 일부 낮출 수 있지만, expression manifold 보존과 downstream probe에는 긍정적인 경향이 있다.
- 이는 “masked symbol에서 value shortcut을 방지하고, cross-gene context를 학습한다”는 설계 의도와 부합한다.
- 따라서 full main candidate에서 `mixed`를 사용하는 것은 타당하다.

### 4.5 JEPA ablation 해석

| Run | Top10 acc | Linear probe Pearson | Expression distance Spearman | 해석 |
|---|---:|---:|---:|---|
| `stage1_jepa_off` | 0.4022 | 0.3979 | 0.7719 | manifold 보존이 강함 |
| `stage1_jepa_lite` | 0.4010 | 0.4526 | 0.6952 | linear probe 향상 가능성 |
| `stage1_jepa_paper` | 0.4108 | 0.4450 | 0.6862 | top-k/linear probe 균형 |

긍정적 해석:

- JEPA auxiliary는 primary MSM을 대체하기보다 보조 regularizer로 해석하는 것이 적절하다.
- `lite`/`paper` 모두 linear probe 성능을 유지하거나 개선하는 신호가 있다.
- Stage 1.5/Stage 2에서 JEPA를 핵심 objective로 사용하는 방향과도 방법론적으로 연결된다.

## 5. Stage 1 결론

현재 Stage 1에서 가장 유망한 방향은 다음과 같다.

1. **Stage1 main candidate:** `stage1_full_rel4096`
   - `global_median + clip4096 + mixed value augmentation + random seq cap`
   - relative rank 및 expression manifold 보존에서 긍정적
2. **Baseline 재실행 필요:** `stage1_full_baseline`
   - 현재 batch가 rel4096과 달라 엄밀 비교를 위해 `ABL_BATCH=512`로 재실행 권장
3. **rel8192는 재시도 필요**
   - memory/compute 안정화 후 `ABL_BATCH=512` 또는 더 낮은 batch로 재실행
4. **MVM metric 해석 주의**
   - rel4096는 absolute value reconstruction에는 약하지만, 연구 목적상 이는 치명적 결함이라기보다 normalization 철학의 결과로 볼 수 있다.
   - Stage 2 image-to-expression에서는 raw-scale target과 rank/relative target을 분리 평가해야 한다.

## 6. Stage 1.5 진행 상황

Stage 1.5는 spatial context-aware encoder로, HyperST의 niche 개념을 `spot + neighbors(region)`로 재구성한다. 현재 main candidate는 다음과 같다.

| 구성 | 값 | 의미 |
|---|---|---|
| frozen Stage1 ckpt | `stage1_full_rel4096/ckpt_tx_encoder_best.pt` | relative salience 기반 spot encoder 사용 |
| subgraph | ego | 실제 tissue locality를 반영 |
| token mode | separate | spot token과 region token을 분리 |
| mask target | spot | visible region context로 masked spot representation 예측 |
| objective | spatial JEPA | latent prediction 방식으로 spatial context 학습 |
| tx_encode_batch | 256 | frozen Stage1 tx_encoder OOM 방지용 micro-batch |

최근 OOM 원인은 Stage1.5 batch size가 아니라, 내부에서 frozen Stage1 tx_encoder가 `region_hvg` 전체를 한 번에 처리하던 경로였다. 현재는 `tx_encode_batch` micro-batch를 도입해 이 경로를 완화했다.

실행 예시:

```bash
STAGE1_CKPT=results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt TAG=stage15_rel4096_spatial EPOCHS=30 bash scripts/train/stage15_main.sh
```

OOM 발생 시:

```bash
STAGE1_CKPT=results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt TAG=stage15_rel4096_spatial EPOCHS=30 TX_ENCODE_BATCH=128 BATCH_SIZE=8 bash scripts/train/stage15_main.sh
```

## 7. Stage 1.5 Test Evaluation 계획

Stage1.5 test evaluation은 validation loss와 별도로, spatially informative representation을 직접 확인하는 방향으로 구성했다.

새 evaluator:

```bash
scripts/eval/stage15_gene_map.py
```

평가 방식:

1. Stage1.5 spatial encoder를 freeze한다.
2. train split에서 spatial embedding `z_spatial -> selected gene expression` Ridge probe를 학습한다.
3. test sample에서 gene별 GT expression map과 predicted expression map을 그린다.
4. gene별 Spearman SCC를 계산한다.

실행 예시:

```bash
python scripts/eval/stage15_gene_map.py   --stage1-ckpt results/runs/stage1_full_rel4096/ckpt_tx_encoder_best.pt   --spatial-ckpt results/runs/stage15_rel4096_spatial/ckpt_spatial_best.pt   --split test   --genes MKI67 EPCAM COL1A1 CD3D   --probe-train-samples 20   --max-train-spots 20000
```

출력:

- `gene_map_scc.csv`: sample/gene별 Spearman SCC
- `<sample_id>/<gene>.png`: GT vs predicted spatial expression map
- `<sample_id>/<gene>_spots.csv`: spot별 `x,y,gt,pred`

긍정적 해석 포인트:

- 특정 marker gene의 발현 공간 패턴을 prediction map이 따라간다면, Stage1.5 representation이 단순 spot embedding이 아니라 spatial context를 반영하고 있음을 보여줄 수 있다.
- SCC는 absolute expression scale보다 공간적 상대 패턴 보존을 보기 때문에, Stage1의 relative salience 철학과 일관된다.
- 이 평가는 downstream image-to-transcriptomics prediction과도 연결된다. Stage2에서 image embedding이 이 spatially informed RNA target을 예측/정렬하게 되면, pathology image가 local molecular pattern뿐 아니라 neighborhood context까지 반영하는지 평가할 수 있다.

## 8. Stage 2 진행 상황과 계획

기존 Stage2 관련 run:

- `full_human_jepa_uni_lora`
- `full_human_clip_uni_lora`
- `full_human_barlow_uni_lora`
- `full_human_s2l_uni_lora`
- `ours_tx_jepa_uni_lora`

현재 Stage2는 다음 평가 축을 갖도록 보강되었다.

| Eval | 목적 | 스크립트 |
|---|---|---|
| zero-shot retrieval | image-RNA alignment 확인 | `scripts/eval/zero_shot.py` |
| image-to-HVG linear probe | pathology image embedding의 molecular predictability 확인 | `scripts/eval/linear_probe.py` |
| slide-level MIL | Loki/HEST/PathBench-style MSI/subtype task 확장 | `scripts/eval/slide_mil.py` |

Stage2 ablation 축:

| Axis | 후보 |
|---|---|
| image encoder | UNI/UNI2, H0-mini, GigaPath 등 |
| alignment method | JEPA, CLIP, Barlow, CCA, S2L |
| tx target | ours Stage1, Novae, HVG MLP baseline |
| downstream task | HEST expression, MSI/subtype, slide-level MIL |

긍정적 해석:

- Stage1/1.5가 relative/spatial transcriptomics target을 강화하면, Stage2는 단순 image patch retrieval보다 더 생물학적으로 의미 있는 image-to-molecular alignment를 학습할 수 있다.
- Loki가 제시하는 visual-omics foundation 방향과 맞게, retrieval + expression prediction + slide-level downstream으로 평가 축을 확장했다.

## 9. 현재 리스크와 대응

| 리스크 | 현재 상태 | 대응 |
|---|---|---|
| full baseline과 rel4096 batch 차이 | baseline=1024, rel4096=512 | baseline을 512로 재실행해 공정 비교 |
| rel8192 실패/중단 | ckpt 없음, meta만 존재 | batch 512 이하, 필요 시 `max_seq_len`/`tx_encode_batch` 조정 |
| MVM에서 rel4096 낮음 | global_median 철학상 absolute value 불리 | rank/Spearman/manifold metric 중심 해석, Stage2 target 분리 |
| Stage1.5 OOM | tx_encoder 내부 batch 문제 확인 | `tx_encode_batch` 도입, 256/128/64 조절 가능 |
| Stage1.5 test 아직 미실행 | evaluator 구현 완료 | spatial ckpt 생성 후 gene map SCC 실행 |
| Stage2 downstream label 필요 | evaluator는 준비됨 | HEST/PathBench label CSV 확보 필요 |

## 10. 다음 액션 아이템

우선순위 높은 순서:

1. `stage1_full_baseline`을 `ABL_BATCH=512`로 재실행해 `rel4096`과 공정 비교한다.
2. `stage1_full_rel8192`를 batch 512 또는 더 낮은 설정으로 재시도한다.
3. `stage1_full_rel4096` ckpt로 Stage1.5 main run을 완료한다.
4. Stage1.5 ckpt에 대해 `stage15_gene_map.py`로 marker gene spatial map/SCC test를 생성한다.
5. Stage2에서 `stage1_full_rel4096` 또는 Stage1.5 spatial target을 활용한 alignment run을 설계한다.
6. HEST/PathBench slide-level label CSV를 연결해 `slide_mil.py`를 실행한다.

## 11. 보고용 핵심 메시지

현재 연구는 단순히 loss를 낮추는 방향이 아니라, **spatial transcriptomics와 pathology image alignment에 적합한 representation hierarchy를 만드는 방향**으로 진전되고 있다.

특히 Stage1의 `rel4096` 결과는 다음 메시지를 뒷받침한다.

- absolute expression reconstruction은 약할 수 있지만,
- relative gene salience와 expression manifold 보존은 더 강하며,
- 이는 Geneformer-style language-like transcriptomics representation이라는 설계 철학과 부합한다.

따라서 현재까지의 결과는 “표현학습 foundation을 만드는 관점”에서는 긍정적이다. 다음 단계는 이 representation이 spatial context(Stage1.5)와 pathology image alignment(Stage2)에서도 유지/증폭되는지를 test gene map, retrieval, image-to-expression, slide-level MIL로 확인하는 것이다.
