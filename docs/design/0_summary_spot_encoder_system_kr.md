# 0. Spot Encoder System Summary

작성일: 2026-06-24  
프로젝트: `/workspace/mm_align`  
역할: 전체 설계 요약, stage별 산출 embedding 정의, pretraining/adaptation/evaluation 경계 정리

## 1. 최종적으로 얻고 싶은 것

이 프로젝트의 최종 목표는 단순한 gene expression imputation 모델이 아니라, **pathology image와 spatial transcriptomics를 연결할 수 있는 계층적 spot encoder system**을 만드는 것이다.

핵심 산출물은 여러 종류의 embedding이다.

| 이름 | 생성 stage | 입력 | 의미 | 주 사용처 |
|---|---|---|---|---|
| `z_spot_tx` / `h_tx` | Stage 1 | spot gene symbol/value sequence | sample-agnostic transcriptomic spot state | RNA intrinsic eval, Stage 2 target |
| `z_chunk` | Stage 1.25 또는 joint Stage 1 | sampled gene chunk | spot 내부 partial gene-state | chunk diagnostic, JEPA objective |
| `z_spot_tx_chunk` | Stage 1.25 또는 joint Stage 1 | multiple chunks | intra-spot gene dependency를 aggregate한 spot state | Stage 1.5 input 후보 |
| `z_spot_tx_spatial` / `z_spatial` | Stage 1.5 | spot + spatial neighbors + graph | sample-local spatial context가 반영된 spot state | spatial downstream, image alignment target |
| `z_patch_img` | PFM | pathology patch | raw patch-level morphology representation | Stage 2 input baseline |
| `z_patch_img_aligned` | Stage 2 | patch + RNA/spatial target | transcriptomics signal이 주입된 pathology representation | retrieval, image-to-expression, MIL |
| `z_spot_mm_spatial` | future Stage 2.5/3 | aligned image/RNA + spatial graph | multimodal spatial context representation | sample/slide-level downstream |

요약하면 최종 시스템은 다음 hierarchy를 가진다.

```text
raw expression
  -> vocab / normalization / tokenization
  -> Stage 1 transcriptomic spot encoder
       -> z_spot_tx
  -> Stage 1.25 intra-spot chunk refinement
       -> z_chunk, z_spot_tx_chunk
  -> Stage 1.5 spatial context adaptation
       -> z_spot_tx_spatial
  -> Stage 2 pathology-RNA alignment
       -> z_patch_img_aligned, z_spot_mm
  -> future multimodal spatial contextualization
       -> z_spot_mm_spatial
```

## 2. Pretraining, adaptation, tuning의 경계

reference 기준으로 보면 Geneformer/scGPT류는 대규모 unlabeled transcriptome corpus에서 masked/generative objective로 cell embedding을 학습하는 단계를 pretraining으로 본다. I-JEPA는 latent prediction을 self-supervised pretraining으로 본다. 반면 SEAL/Loki류는 pretrained pathology foundation model을 ST signal에 맞춰 alignment/fine-tuning하는 성격이 강하다.

이 프로젝트에서는 다음 용어가 가장 안전하다.

| 단계 | 이름 | 성격 | 이유 |
|---|---|---|---|
| Stage 1 | transcriptomic spot pretraining | pretraining | sample graph 없이 spot 내부 gene symbol/value 구조를 학습 |
| Stage 1.25 | intra-spot self-supervised refinement | pretraining 또는 continual refinement | 같은 spot 내부 chunk 간 dependency를 학습 |
| Stage 1.5 | spatial self-supervised adaptation | adaptation / pre-finetuning | sample-local neighbor graph와 coordinate를 사용 |
| Stage 2 | pathology-transcriptomics alignment | multimodal tuning | pretrained PFM을 molecular/spatial target에 맞춤 |
| Stage 2.5/3 | multimodal spatial contextualization | downstream-oriented adaptation | aligned modality를 다시 sample graph 위에서 coding |

따라서 보고 문장으로는 다음 표현을 권장한다.

> We pretrain a sample-agnostic transcriptomic spot encoder, refine it with intra-spot chunk-level self-supervision, adapt it to sample-level spatial context using spatial JEPA, and finally align it with pathology patch representations.

## 3. 핵심 철학

### 3.1 Absolute imputation보다 representation

이 프로젝트의 Stage 1은 expression value를 완벽히 복원하는 imputation model이 아니다. 목표는 이후 spatial/context/multimodal 단계에서 쓸 수 있는 robust spot representation이다.

따라서 주요 관심은 다음이다.

- gene symbol context를 이해하는가?
- spot 내부 relative expression salience를 보존하는가?
- gene-gene dependency가 embedding에 반영되는가?
- masked target gene 없이도 주변 gene context로 expression/rank를 예측할 수 있는가?
- sample/source shortcut 없이 across-sample generalization이 가능한가?

### 3.2 Spatial context는 sample-local adaptation

Stage 1.5는 sample-level graph를 사용한다. 따라서 `z_spot_tx_spatial`은 sample-agnostic embedding이라기보다 **sample-local spatially contextualized embedding**이다. 이것은 약점이 아니라 역할 구분이다.

- `z_spot_tx`: sample 간 비교와 molecular state 비교에 적합
- `z_spot_tx_spatial`: spatial domain, tissue niche, boundary, local field 평가에 적합

### 3.3 PFM도 patch-local이다

Pathology foundation model의 patch embedding은 보통 patch-level local morphology representation이다. 따라서 Stage 2 이후에도 image side에 spatial context를 넣을 여지가 있다.

권장 확장 순서:

1. Stage 1/1.25/1.5를 안정화한다.
2. Stage 2에서 PFM patch와 RNA/spatial target을 alignment한다.
3. Stage 2.5/3에서 aligned image/RNA embedding을 spatial graph 위에서 다시 contextualize한다.

## 4. Reference와의 관계

| Reference 계열 | 프로젝트 반영 |
|---|---|
| Geneformer | global-median/rank-like relative salience, masked gene modeling, vocab-size budget |
| scGPT | self-supervised transcriptome pretraining과 downstream/fine-tuning 분리 |
| I-JEPA | context block -> target block latent prediction, full teacher + masked/partial student |
| ST-JEPA/STFormer | spatial coordinate/neighbor graph를 이용한 spatial contextualization |
| HyperST | spot/niche 관점을 `spot + neighbors(region)`으로 재해석 |
| SEAL/Loki/HEST | pathology PFM과 ST alignment, image-to-expression, retrieval, slide-level downstream |

## 5. 최종 평가 원칙

각 embedding은 같은 task에서 비교되어야 한다.

| Evaluation | `z_spot_tx` | `z_spot_tx_chunk` | `z_spot_tx_spatial` | `z_patch_img_aligned` |
|---|---:|---:|---:|---:|
| MSM / masked symbol | main | optional | no | no |
| HVG expression probe | main | main | main | image-to-HVG |
| masked-HVG probe | main | main | optional | image-to-masked-HVG |
| gene map SCC | yes | yes | main | yes if paired |
| DLPFC layer probe | yes | yes | main | yes if paired |
| spatial clustering | optional | optional | main | optional/main |
| cross-modal retrieval | no | no | target | main |
| slide-level MIL | weak | weak | strong candidate | main |

핵심은 “어느 stage가 더 좋다”가 아니라, **어떤 embedding이 어떤 biological question에 적합한가**를 분리해 보는 것이다.
