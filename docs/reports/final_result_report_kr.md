## 연구 제목 및 연구책임자

국문 연구 제목: 공간전사체 기반 멀티모달 파운데이션 모델 개발

영문 연구 제목: SPATULA: Spatial Predictive Alignment for Transcriptomics Using Latent Associations

연구책임자: 김은수

## 연구의 필요성 및 배경

중간 연구계획서에서는 공간전사체(spatial transcriptomics)와 병리 이미지(Whole Slide Image, WSI)를 통합적으로 이해하는 멀티모달 파운데이션 모델의 필요성을 제시하였다. 기존 공간전사체 및 pathology AI 연구는 특정 downstream task에 특화된 경우가 많고, gene expression과 morphology 사이의 공통 representation을 충분히 학습하지 못하며, 대규모 공간전사체 데이터를 foundation model 수준으로 활용하는 데 한계가 있었다.

본 연구는 이러한 문제의식에서 출발하여 gene–image–spatial 정보를 통합적으로 이해하는 representation을 학습하는 것을 목표로 하였다. 특히 spot 하나를 단순 수치 벡터가 아니라 gene symbol과 expression value가 결합된 biological token set으로 보고, 이후 spatial neighborhood와 pathology image까지 단계적으로 결합하는 구조를 설계하였다. 이는 계획서에서 제시한 Intra-spot → Inter-spot → Inter-modality의 계층적 학습 전략과 직접적으로 연결된다.

## 연구의 핵심 아이디어 또는 가설

본 연구의 첫 번째 가설은 spot 내부의 expressed gene set이 일종의 biological sentence처럼 작동할 수 있다는 것이다. 각 gene은 symbol embedding으로 정체성을 표현하고, expression value는 Fourier value encoding으로 연속적인 정량 정보를 표현한다. 두 정보를 결합한 gene embedding sequence를 transformer encoder에 입력하여 spot-level representation을 얻는다.

두 번째 가설은 spot representation이 spatial context를 통해 확장될 수 있다는 것이다. HyperST의 spot/niche 관점을 본 연구에서는 spot + neighbors 구조로 재해석하였고, Stage 1.5에서 Spatial JEPA를 통해 visible neighbor context로 masked center spot representation을 예측하도록 설계하였다.

세 번째 가설은 병리 이미지와 transcriptomics의 관계가 단순한 contrastive matching만으로 충분하지 않다는 것이다. 따라서 Stage 2에서는 CLIP 방식의 alignment뿐 아니라 JEPA 방식의 predictive mapping을 함께 고려하여 morphology가 molecular state를 예측할 수 있는 representation을 학습하는 방향으로 확장하였다.

## 연구의 목표

연구의 최종 목표는 공간전사체와 병리 이미지를 통합적으로 이해하는 멀티모달 파운데이션 모델을 개발하고, 다양한 생물학적 및 임상적 downstream task에 적용 가능한 representation을 학습하는 것이다. 이를 위해 Stage 1에서는 RNA/spot encoder를 사전학습하고, Stage 1.5에서는 spatial context-aware encoder를 구축하며, Stage 2에서는 pathology image와 transcriptomics representation을 alignment하는 구조로 연구를 추진하였다.

세부 목표는 다음과 같다. 첫째, gene symbol과 expression value를 분리하여 모델링하는 spot encoder를 구현한다. 둘째, masked symbol modeling(MSM), value corruption, DINO-style consistency, Gene-JEPA 등 self-supervised objective를 비교할 수 있는 ablation 구조를 구축한다. 셋째, Stage 1.5에서 spatial neighbor 정보를 반영하는 Spatial JEPA encoder와 정성·정량 평가 도구를 구현한다. 넷째, Stage 2에서 image-to-transcriptomics prediction, cross-modal retrieval, slide-level MIL benchmark로 확장 가능한 평가 scaffold를 마련한다.

## 연구 추진 내용 및 연구 개발 결과

연구 추진은 계획서의 3단계 구조에 맞추어 진행하였다. Stage 1에서는 top-HVG gene encoder를 구현하였다. 이 encoder는 zero-expressed gene을 제거한 뒤, gene symbol embedding과 Fourier value embedding을 결합하고 transformer를 통해 spot embedding을 생성한다. 기본 objective는 MSM이며, value shortcut을 방지하기 위해 masked gene과 unmasked context gene의 value augmentation을 분리하였다. 또한 DINO-style teacher-student consistency, KoLeo regularizer, Gene-JEPA auxiliary loss를 선택적으로 켤 수 있도록 구성하였다.

평가 체계도 단순 loss tracking에서 확장하였다. Stage 1의 in-training 및 test evaluation에는 masked symbol top-k accuracy, CE/log(V) 정규화 지표, HVG linear probe, relative rank probe, expression manifold preservation, gene embedding correlation alignment, source leakage, MVM downstream check를 포함하였다. 특히 vocab clip을 사용할 경우 CE loss scale이 log(num_classes)에 따라 달라지므로, raw CE만으로 full vocab과 clip vocab을 비교하지 않도록 masked_symbol_ce_norm을 추가하였다.

Stage 1.5에서는 Stage 1 encoder가 만든 spot embedding을 frozen target으로 활용하고, spatial graph 위에서 neighbor/region context를 반영하는 SpatialEncoder 및 Spatial JEPA objective를 구현하였다. OOM 문제는 frozen tx encoder가 region_hvg 전체를 한 번에 처리하는 경로에서 발생함을 확인하고, tx_encode_batch micro-batch를 도입하여 완화하였다. 또한 test 단계에서 특정 marker gene의 ground-truth spatial map과 predicted spatial map을 비교하고 Spearman SCC를 계산하는 evaluator를 구현하였다.

Stage 2에서는 image encoder와 transcriptomics encoder를 정렬하기 위한 CLIP, JEPA, Barlow, CCA, S2L objective scaffold를 정리하였다. 또한 HEST/SEAL/PathBench류 downstream을 반영할 수 있도록 cross-modal retrieval, image-to-HVG linear probe, slide-level MIL evaluator를 추가하였다. 이는 계획서에서 제시한 image → transcript prediction, cross-modal retrieval, biomarker prediction 방향으로 확장 가능한 기반이다.

## 표 1. Stage 1 test 주요 결과

| 평가 지표 | baseline | rel4096 | 해석 |

|---|---:|---:|---|

| HVG Spearman | 0.5223 | 0.5319 | 상대 발현 순서 보존 소폭 개선 |

| Spot rank Spearman | 0.4001 | 0.5202 | spot 내부 gene 상대순위 복원 개선 |

| Expression distance Spearman | 0.1907 | 0.7144 | expression manifold 보존 크게 개선 |

| Gene embedding corr. | 0.2159 | 0.1771 | co-expression alignment는 baseline이 약간 우세 |

| MVM Spearman | 0.0505 | -0.1038 | absolute value 복원은 rel4096에 불리 |



## 결과 고찰 및 앞으로의 계획

Stage 1 결과는 본 연구가 단순 gene expression imputation 모델이 아니라 spot representation foundation model을 목표로 한다는 점을 뒷받침한다. rel4096 설정은 absolute value reconstruction에서는 불리했지만, relative expression rank와 expression manifold 보존에서는 baseline보다 강한 신호를 보였다. 이는 Geneformer-style global median normalization과 vocab clip을 통해 상대적 발현 salience를 학습하려는 연구 가설과 잘 맞는다.

다만 해석상 주의점도 확인하였다. 기존 일부 vocab clip run은 input dimension은 4096으로 줄었지만 symbol head가 full vocab size를 유지한 상태였음을 확인하였다. 이에 따라 vocab_clip 사용 시 clipped vocab_dict를 자동으로 연결하도록 수정하였고, CE를 log(num_classes)로 정규화하는 지표를 추가하였다. 따라서 향후 핵심 ablation은 수정된 pipeline에서 재실행하여 공정 비교를 수행할 계획이다.

앞으로의 계획은 세 단계이다. 첫째, Stage 1에서 rel4096, rel4096_dino, rel8192, rel4096_jepa를 수정된 vocab/head 설정으로 재실행한다. 둘째, 가장 안정적인 Stage 1 checkpoint를 기반으로 Stage 1.5 Spatial JEPA를 학습하고, marker gene spatial map 및 SCC로 spatially informative representation을 평가한다. 셋째, Stage 2에서 pathology image embedding과 Stage 1.5 target을 JEPA/CLIP 방식으로 alignment하고, image-to-expression prediction, cross-modal retrieval, slide-level MIL benchmark로 downstream 성능을 검증한다.

## 기타

본 연구 과정에서 실험 재현성과 보고 가능성을 높이기 위해 프로젝트 구조 리팩토링, stage별 config 정리, ablation metadata logging, test-time qualitative visualization을 추가하였다. Stage 1 test에서는 UMAP, label barplot, spatial gene expression panel을 생성할 수 있으며, Stage 1.5 test에서는 GT vs predicted gene map, SCC barplot, spatial embedding UMAP을 생성할 수 있다. 이는 최종 결과 해석에서 정량 지표뿐 아니라 representation이 실제 tissue structure와 biological marker pattern을 반영하는지 정성적으로 확인하기 위한 장치이다.

## 참고문헌(Reference)

Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP).

Assran et al., Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).

Cao et al., stFormer: a foundation model for spatial transcriptomics.

Chen et al., A visual–omics foundation model to bridge histopathology with spatial transcriptomics (OmiCLIP).

Tejada-Lapuerta et al., Nicheformer: a foundation model for single-cell and spatial omics.

Litman et al., GeneJepa: A Predictive World Model of the Transcriptome.

ElSheikh et al., Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics.

HEST, SEAL, PathBench 및 관련 spatial transcriptomics benchmark 자료.
