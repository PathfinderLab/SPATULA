# 5. Stage 2 Multimodal Alignment

작성일: 2026-06-24  
역할: pathology foundation model과 transcriptomics/spatial embedding alignment 설계, downstream task, ablation 정리

## 1. Stage 2의 목적

Stage 2는 pretrained pathology foundation model(PFM)의 patch-level image representation을 transcriptomics/spatial target과 align한다.

```text
image patch -> PFM -> z_patch_img
RNA/spatial encoder -> z_spot_tx or z_spot_tx_spatial
alignment objective -> z_patch_img_aligned
```

SEAL/Loki/HEST reference와 같은 방향으로, 목표는 pathology image가 molecular signal을 담도록 만드는 것이다.

## 2. 중요한 전제

PFM patch embedding은 기본적으로 patch-local morphology representation이다. 이것이 sample-level spatial tissue context를 충분히 encode한다고 보긴 어렵다.

따라서 Stage 2는 두 가지 역할을 가진다.

1. patch morphology와 spot-level molecular state를 align한다.
2. future stage에서 image side에도 spatial context를 넣을 기반을 만든다.

## 3. Alignment target 후보

| Target | 의미 | 장점 | 단점 |
|---|---|---|---|
| `z_spot_tx` | Stage1 transcriptomic spot embedding | sample-agnostic, 안정적 | spatial context 없음 |
| `z_spot_tx_chunk` | Stage1.25 refined spot embedding | intra-spot dependency 반영 | 아직 spatial context 없음 |
| `z_spot_tx_spatial` | Stage1.5 spatial embedding | tissue context 반영 | sample-local adaptation 성격 |
| expression/rank target | HVG expression/rank 직접 예측 | HEST benchmark와 직접 연결 | representation alignment보다 task-specific |

권장 ablation:

```text
Image -> z_spot_tx
Image -> z_spot_tx_chunk
Image -> z_spot_tx_spatial
Image -> expression/rank target
```

## 4. Objective 후보

| Objective | 수식 직관 | 질문 |
|---|---|---|
| CLIP contrastive | paired image/RNA 가까이, unpaired 멀리 | retrieval에 좋은가? |
| JEPA latent prediction | image context -> RNA/spatial target latent | semantic molecular prediction에 좋은가? |
| Barlow/CCA | cross-modal correlation matching | modality collapse 없이 align되는가? |
| regression/probe | image embedding -> gene expression | HEST benchmark 성능이 오르는가? |
| MIL | patch/spot aggregate -> slide label | slide-level task로 확장되는가? |

간단한 CLIP식 수식:

```text
sim_ij = cosine(z_img_i, z_rna_j) / tau
L_clip = CE(sim_i*, i) + CE(sim_*i, i)
```

JEPA식 수식:

```text
z_pred = Predictor(z_img)
L_jepa = SmoothL1(z_pred, stopgrad(z_rna_or_spatial))
```

## 5. Downstream task

Stage 2는 retrieval만 보면 부족하다. reference 기준으로 SEAL/HEST/Loki가 보는 task를 반영해야 한다.

| Task | 설명 | Metric |
|---|---|---|
| cross-modal retrieval | image patch와 matched RNA spot retrieval | Recall@k, MRR |
| image-to-HVG | image embedding으로 HVG expression 예측 | Pearson/Spearman/R2 |
| image-to-rank | relative expression salience 예측 | spot-rank Spearman, top-k overlap |
| gene map prediction | image 기반 spatial expression heatmap | gene-wise SCC, heatmap |
| slide-level MIL | patch/spot bags로 MSI/subtype 등 예측 | AUROC, accuracy |
| pathway prediction | pathway score/gene set score 예측 | Pearson/AUROC |

## 6. Image encoder ablation

| Axis | 후보 | 질문 |
|---|---|---|
| PFM | UNI/UNI2, H0-mini, GigaPath, H-Optimus 등 | 어떤 PFM이 ST alignment에 잘 맞는가? |
| tuning | frozen, LoRA, partial, full | molecular signal 주입에 얼마나 update가 필요한가? |
| patch size | HEST default, larger context | morphology context 범위가 중요한가? |
| aggregation | spot patch, neighbor patch, MIL bag | patch-local vs region-level image context |

## 7. Stage2 이후 spatial context

Stage1.5가 RNA side에 spatial context를 넣는다면, Stage2 이후에는 image/RNA aligned embedding에도 spatial context를 넣을 수 있다.

```text
Stage2: z_patch_img_aligned
Stage2.5/3: spatial graph over aligned image/RNA embeddings
          -> z_spot_mm_spatial
```

이 방향은 장기적으로 매우 자연스럽다. 단, Stage1.5와 Stage2를 처음부터 합치면 collator/objective/debugging 복잡도가 크므로, 우선은 분리된 baseline을 안정화한 뒤 확장한다.

## 8. Ablation 대상

| Axis | 후보 | 질문 |
|---|---|---|
| target embedding | tx, chunk, spatial | 어떤 target이 image alignment에 가장 유리한가? |
| objective | CLIP, JEPA, Barlow/S2L | retrieval vs prediction trade-off |
| image encoder | UNI, UNI2, H0-mini, GigaPath | PFM 선택 영향 |
| tuning | frozen/LoRA/partial | PFM update 필요성 |
| downstream | retrieval, gene prediction, MIL | 어느 task에서 강한가? |
| spatial after alignment | off/on | image side spatial context가 추가 이득을 주는가? |

## 9. 좋은 Stage 2의 조건

좋은 Stage 2는 다음을 동시에 만족해야 한다.

- retrieval이 좋아진다.
- image-to-expression/rank prediction이 좋아진다.
- slide-level downstream에서 raw PFM보다 개선된다.
- modality gap이 줄어든다.
- spatial heatmap이 정성적으로 plausible하다.
- 특정 organ/sample에만 과적합하지 않는다.
