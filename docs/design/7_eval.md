# 7. Evaluation Methods and Metrics Reference

작성일: 2026-06-24  
역할: scripts/eval/* 에서 사용하는 모든 평가 방법, metric, figure의 정의와 의미 정리. val_history.json에 기록되는 training-time validation metric도 함께 설명.

## 1. 평가 파이프라인 전체 구조

평가는 크게 두 위치에서 일어난다.

| 위치 | 언제 실행 | 출력 |
|---|---|---|
| **training-time validation** (train.py) | 매 epoch 또는 N epoch마다 | `val_history.json`, `history.json`, `metric_curves.png` |
| **post-training eval** (scripts/eval/*.py) | training 종료 후 ckpt를 받아 따로 호출 | `results/eval/*/...csv`, 다양한 figure |

두 위치의 metric은 일부 겹치지만 호출 컨텍스트가 다르다. validation은 모델 선택과 early stopping에 쓰이는 빠른 진단이고, post-training eval은 정량/정성 비교를 위한 정밀 평가다.

## 2. 입력 데이터 처리 — 어떤 scale을 평가하는가

`results/cache/prepared_expanded/` 안의 shard는 다음 순서로 생성된다.

```text
raw counts
  -> sc.pp.normalize_total (target_sum = 1e4)
  -> np.log1p          (-> hvg_log)
  -> [선택] gene_norm.apply_np (global_median 등) -> 모델 입력
```

이 순서에서 두 개의 "표현 공간"이 존재한다.

* **log1p_cp10k** (= shard 안의 `hvg_log`):
    * `sc.pp.normalize_total` + `np.log1p`. 생물학적으로 흔히 쓰는 scale.
    * 모든 downstream GT는 이 scale을 기준으로 한다.
* **gene_norm space** (= 모델 입력으로 들어가는 scale):
    * 기본은 `global_median`: `(hvg_log - 0) / nonzero_median_per_gene`. zero-preserve + clip(±8) 적용.
    * 모델이 학습하는 loss는 이 scale 위에서 계산된다.

두 scale은 per-gene 양의 affine 변환 관계라서 **Spearman / Pearson 같은 rank/linear-invariant metric은 동일**하다 (clip이 거의 안 걸리는 경우). 그러나 **RMSE / SSIM / JSD는 NOT scale-invariant**이고, 모델의 loss surface는 gene_norm space 위에 있으므로 그 scale에서 본 RMSE/SSIM/JSD가 실제 loss와 가깝다.

평가 스크립트는 이 사실을 반영해 `*_raw`와 `*_norm` 두 컬럼을 같이 emit한다.

* `*_raw`: log1p_cp10k 위에서 계산한 metric (해석성 우선)
* `*_norm`: gene_norm 변환 후 metric (모델이 보는 scale)

## 3. Stage 1 transcriptomic eval — `scripts/eval/stage1_tx.py`

목적: frozen Stage-1 tx_encoder가 만든 spot embedding의 정량적 health, downstream probe, leakage, modality intrinsic을 한 번에 본다. 세 가지 representation (`h_tx`, `chunk_state`, `spot_state`)에 같은 metric을 모두 적용한다.

### 3.1 평가하는 representation 세 종류

| Representation | 정의 | 출력 차원 |
|---|---|---|
| `h_tx` | encoder의 CLS token output | embed_dim (예: 512) |
| `chunk_state` | 무작위로 sampling한 한 chunk만 통과시킨 CLS | embed_dim |
| `spot_state` | n_chunks 개 chunk의 CLS를 평균 (inference pooling) | embed_dim |

### 3.2 Health / intrinsic metric

| Metric | 식 | 의미 |
|---|---|---|
| `intrinsic_effective_rank` | exp(entropy of singular values) | embedding이 차원을 얼마나 활용하는지. 1에 가까울수록 collapsed |
| `intrinsic_explained_top10` | top-10 singular value의 분산 비율 | 첫 10개 component에 정보가 집중되어 있는지 |
| `intrinsic_norm_mean / std` | L2 norm 통계 | embedding magnitude 분포 |
| `intrinsic_expression_knn_overlap@20` | h_tx KNN과 HVG KNN의 Jaccard | embedding 위 가까운 spot이 실제 expression도 가까운지 |
| `intrinsic_expression_distance_spearman` | cos(h_tx) 와 cos(HVG)의 Spearman | manifold geometry 유지 |
| `intrinsic_gene_embedding_corr_spearman` | gene-token embedding의 cosine 과 co-expression의 Spearman | encoder의 gene token이 의미 있는 거리 갖는지 |

### 3.3 Downstream probe (Ridge 회귀)

`hvg_linear_probe(h_tx, hvg_eff)` — frozen embedding → 선택된 HVG 발현 예측. SEAL / HEST 식.

| Argument | 의미 |
|---|---|
| `--linear-probe-genes 256` | high-variance + nonzero_frac > 0.01 인 유전자 중 분산 상위 256개 |
| `--probe-pca-n 256` | embedding → StandardScaler → PCA(256) → Ridge (HEST 표준) |
| `--probe-alpha auto` | `α = 100 / (features × targets)` HEST/SEAL 공식. float이면 그대로 |
| `--probe-metric-suite spatial_bench` | Pearson/Spearman/R²/RMSE_norm + SSIM/JSD/RMSE_zscore + per-gene median/Q1/Q3 |
| `--gene-held-out-folds 5` | target gene을 5 fold로 나눠 fold마다 mask + 재인코딩 + 그 fold gene만 예측 → cross-gene generalization |

per-gene 결과는 `return_per_gene=True`로 받아 figure에 활용. `probe_figures/per_gene_pearson_hist.png` 와 `pearson_vs_ssim_*.png` 가 생성된다.

### 3.4 Source / batch leakage

| Metric | 식 | 의미 |
|---|---|---|
| `source_probe_acc` | LogisticRegression(h_tx → source 라벨) 5-fold | source가 임베딩에서 얼마나 선형 분리되는지. 1에 가까우면 batch effect 강함 |
| `leakage_source_knn_same_rate@20` | h_tx KNN 20개 중 같은 source 비율 | KNN topology에서 source clustering 정도 |
| `leakage_source_knn_entropy@20` | KNN source 분포의 entropy | 높을수록 source가 골고루 섞임 (좋음) |

### 3.5 MVM (value head 점검)

| Metric | 의미 |
|---|---|
| `mvm_pearson / spearman / r2` | masked value head가 expression value를 얼마나 회복하는지 (R²<0 이면 학습이 안 됐다는 신호) |
| `mvm_rmse_norm` | value MSE / target variance의 제곱근 |

### 3.6 Qualitative figure (`--make-viz`)

| Figure | 설명 |
|---|---|
| `eval_pool_barplots.png` | source / organ 별 spot count |
| `embedding_umap_by_source.png` | UMAP을 source로 색칠 (PCA(50) 후 UMAP, n_neighbors=50, min_dist=0.3 — sample-cluster bias를 더 정직하게 표시) |
| `embedding_umap_by_organ.png` | organ 색칠 |
| `embedding_umap_by_sample_id.png` | sample 색칠 |
| `gt_spatial_gene_maps/{sample}_gt_gene_maps.png` | 지정 gene의 GT spatial expression panel |
| `probe_figures/per_gene_pearson_hist.png` | linear vs gene_held_out probe의 per-gene Pearson 분포 |
| `probe_figures/pearson_vs_ssim_*.png` | per-gene Pearson vs SSIM 산점도 (uniform mediocrity 인지 long-tail인지) |

## 4. Masked Symbol Modeling eval — `scripts/eval/msm_eval.py`

목적: encoder의 내부 마스킹 파이프라인을 강제로 켜고 (`_force_mask_in_eval = True`) symbol_head의 top-K 정확도를 측정. training-time MSM loss와 같은 형태지만 test split 위에서 정량 평가한다.

### 4.1 Top-K accuracy

| Variant | 의미 |
|---|---|
| `top1_micro` ~ `top20_micro` | 전체 masked position을 풀어 풀에서 K-best 안에 정답이 있는 비율 |
| `top1_macro` ~ `top10_macro` | gene 단위로 정확도 계산 후 평균. 흔한 gene이 dominate하지 않음 |
| `nll` | masked position의 cross-entropy 평균. 낮을수록 좋음 |
| `mean_conf_gt` / `median_conf_gt` | softmax probability at GT token의 평균/중앙값 |

**primary metric은 top-10**. top-1은 vocab=4096에 대비 ~0.18 수준이어서 변별력이 약하고, top-10은 ~0.53 수준으로 변별력이 좋다.

### 4.2 Per-gene 분석

`per_gene.csv` — gene별 (token_id, n_obs, top1, top5, top10).  
`best_genes.csv` / `worst_genes.csv` — n_obs ≥ 20인 gene 중 top-10 기준 상/하위 50개.

해석 가이드:
* "Best" gene은 보통 조직 특이성이 강한 marker (PTPN5, FGFBP2, ETNPPL, TRBC1 등).
* "Worst" gene은 면역계 (IFIT2, SLAMF7, ICOSLG, IGHD), drift 빠른 stress response (MT1H, HAS2), 또는 분포가 너무 sparse한 유전자.

### 4.3 Per-organ / per-source breakdown

`per_organ.csv` / `per_source.csv` — 각 grouping에서 n ≥ 20인 그룹의 top-10 acc + top-1 acc + n. figure는 top-10 으로 막대 그래프.

### 4.4 Figures

| Figure | 설명 |
|---|---|
| `fig_topk_bar.png` | micro vs macro top-K bar (top-10 컬럼 녹색 음영) |
| `fig_per_gene_hist.png` | per-gene top-10 / top-1 분포 |
| `fig_best_worst.png` | top-10 기준 best 25 / worst 25 |
| `fig_per_organ.png` | organ별 top-10 |
| `fig_per_source.png` | source별 top-10 |
| `fig_confidence.png` | softmax p(GT token) histogram |

## 5. DLPFC eval — `scripts/eval/dlpfc_eval.py`

목적: spatialLIBD의 12개 DLPFC sample을 받아 frozen encoder가 cortical layer 정보를 보존하는지 검증. supervised layer probe + zero-shot clustering + gene-map probe.

### 5.1 데이터 로드

`load_dlpfc_sample()` ([src/mm_align/data/dlpfc.py:125](src/mm_align/data/dlpfc.py)):
1. 10X H5 → raw counts
2. project hvg vocab 으로 컬럼 재정렬
3. `sc.pp.normalize_total(1e4) + np.log1p` → `hvg_log` (log1p_cp10k)

**중요**: 이 단계에서 `gene_norm`은 적용하지 않는다. dlpfc_eval은 GT를 raw log1p_cp10k scale에서 그대로 본다.

### 5.2 Linear layer probe

`linear_probe(emb, layer_labels)` — sklearn LogisticRegression 5-fold. 출력 `layer_probe_acc`, `layer_probe_f1_macro`. supervised이므로 임베딩에 cortical layer 정보가 들어있는지 직접 측정.

### 5.3 KNN purity

`knn_purity_at_{k}` — embedding KNN 중 같은 layer 라벨 비율. k ∈ {5, 10, 20}. unsupervised geometry 측정.

### 5.4 Clustering metric (`--cluster-methods kmeans,leiden,gmm`)

세 method를 동시에 돌려 prefix별로 결과를 분리해서 표시한다.

| Method | 알고리즘 |
|---|---|
| `kmeans` | sklearn KMeans, n_init=10 |
| `leiden` | scanpy `sc.tl.leiden`, resolution을 target k로 bisection 튜닝 |
| `gmm` | sklearn GaussianMixture, covariance_type='tied' (SDMBench의 mclust 대체) |

각 method마다 다음 metric을 emit (prefix `{method}_`):

| Metric | 식 / 의미 |
|---|---|
| `ari` | Adjusted Rand Index vs GT layers |
| `nmi` | Normalized Mutual Information |
| `homogeneity` | each predicted cluster가 한 layer로만 이루어졌는지 |
| `completeness` | each layer가 한 cluster로만 모이는지 |
| `silhouette` | embedding 공간의 silhouette score |
| `asw_spatial` | spatial xy 거리 위의 silhouette (precomputed, SDMBench 식) |
| `chaos` | 1-NN 내부 cluster 거리 합 / n. cluster가 공간상 응집됐는지 (낮을수록 좋음) |
| `pas` | k=10 이웃 majority label과 다른 spot 비율 (낮을수록 좋음) |
| `fide` | F1 over Intra-Domain Edges (novae 식). 같은 cluster spot이 공간상 KNN인지. 0~1, 높을수록 좋음 |
| `entropy_norm` | log2(K)로 정규화한 cluster 크기 entropy. 0~1, balanced cluster면 1 |
| `heuristic` | `fide × entropy_norm / log2(K)` (novae) — 공간 응집 + 균형의 단일 score |
| `marker_morans_i_median` | top-5 marker 유전자의 Moran's I 중앙값 (squidpy 안 씀, KNN graph로 직접 계산) |
| `marker_gearys_c_median` | 동일하게 Geary's C |

GT layer 자체에 대해서도 같은 spatial-continuity metric을 한 번 emit (`gt_layer_*` prefix). 모델이 GT만큼 응집된 cluster를 만들었는지 비교 기준이 된다.

`--cluster-repeats N`: 각 method를 N번 random seed 변경해서 돌리고 mean + `_std` emit.

### 5.5 Two-mode gene-map probe

Leave-one-sample-out Ridge probe (alpha=10): N-1개 sample의 (emb, hvg_eff)로 학습 → held-out sample 예측 → per-gene metric 계산.

기본 출력 gene = canonical DLPFC layer markers (Maynard et al. 2021 Nature Neuroscience, spatialLIBD).

| Gene | 의미 |
|---|---|
| `MBP`, `MOBP` | myelin basic / oligodendrocyte → white matter |
| `SNAP25` | pan-neuronal (모든 gray matter layers) |
| `PCP4` | layer 5 marker |
| `GFAP` | astrocyte / white matter boundary |
| `CARTPT` | layer 4 marker |

STAGATE / BANKSY / GraphST / BayesSpace 모두 같은 6종을 사용. dlpfc_eval은 두 가지 scale에서 metric을 모두 emit:

| Column | 의미 |
|---|---|
| `spearman_scc_raw`, `pearson_raw` | log1p_cp10k 위 |
| `ssim_raw`, `jsd_raw`, `rmse_zscore_raw` | log1p_cp10k 위 |
| `spearman_scc_norm`, `pearson_norm` | gene_norm 후 (`global_median` 등) |
| `ssim_norm`, `jsd_norm`, `rmse_zscore_norm` | gene_norm 후 |

**SCC / PCC는 두 모드에서 같다** (수학적 invariance — per-gene 양의 affine 변환). **SSIM / JSD / RMSE_zscore는 다르다** — gene_norm 모드가 encoder의 loss와 가까운 척도다.

### 5.6 Figures

| Figure | 설명 |
|---|---|
| `{ckpt}/{rep}/clusters/{sample}_clusters.png` | GT layer + 모든 cluster method 결과 side-by-side (4-panel) |
| `{ckpt}/{rep}/method_summary.png` | 각 method의 ARI/NMI/HOM/COM/CHAOS/PAS 막대그래프 + std |
| `{ckpt}/{rep}/gene_maps/{sample}/{gene}.png` | 2-panel: GT (left) vs Pred (right). 제목에 raw + normalised metric 동시에 표시. **Pred 가 smooth해 보이는 것은 정상 — Ridge가 dominant signal만 남기고 noise를 제거하기 때문.** GT는 raw 데이터의 drop-out / sparse measurement 때문에 점박이 패턴이 나온다. |
| `{ckpt}/{rep}/gene_map_scc_barplot.png` | SCC bar + SSIM(raw vs norm) bar — gene-norm이 SSIM에 영향을 주는지 확인 |

## 6. SVG eval — `scripts/eval/svg_eval.py`

목적: foundation encoder가 공간적으로 변하는 유전자(SVG)의 ranking을 보존하는지 측정. SVG_Benchmarking의 protocol을 따른다.

### 6.1 두 ranking을 비교

1. **GT ranking**: 실제 log1p hvg에서 Moran's I per gene → 내림차순 정렬
2. **Predicted ranking**: emb → Ridge → predicted log1p map의 Moran's I per gene → 정렬

### 6.2 Metric

| Metric | 의미 |
|---|---|
| `kendall_tau_morans` | 두 ranking의 Kendall tau (전체 4096개 유전자) |
| `spearman_morans` | Spearman correlation of Moran's I 값 |
| `pearson_morans` | Pearson |
| `spearman_gearys` | Geary's C 위에서의 Spearman (보조) |
| `top_K_overlap` | top-K (K ∈ 25/50/100/200) gene의 Jaccard |
| `aupr_top_k` | GT top-K를 positive로 하고 predicted Moran's I를 score로 한 PR-AUC |

### 6.3 Shard 필터링

placeholder coords를 가진 shard (`coords.std == 0` 등)는 자동 제외. 그렇지 않으면 figure가 한 점으로 collapse한다.

### 6.4 Figures

| Figure | 설명 |
|---|---|
| `{ckpt}/rank_scatter/{sample}.png` | GT 순위 vs Pred 순위 산점도 + top-K 점선 |
| `{ckpt}/top_svg_maps/{sample}.png` | top SVG 6~8개의 GT vs Pred spatial map (2 row) |
| `{ckpt}/top_k_overlap.png` | sample별 top-K overlap bar |

## 7. Spot deconvolution — `scripts/eval/spot_deconv.py`

목적: frozen embedding이 cell-type proportion을 얼마나 잘 회복하는지. 입력 데이터 가용성에 따라 3-mode + synthetic fallback.

| Mode | 입력 | 출력 |
|---|---|---|
| `proportion` | per-spot cell-type proportion CSV | Ridge probe → per-celltype PCC/RMSE/SSIM/JSD |
| `reference` | scRNA centroid CSV | encoder로 centroid embedding → cosine softmax → proportions. optional GT가 있으면 metric overlay |
| `hard` | per-spot hard label CSV | LogisticRegression → accuracy + macro-F1 + confusion matrix |
| `synthetic` | 없음 — 알고리즘이 smoke-검증용 synthetic data 생성 | 모든 metric/figure path를 무 데이터로 검증 |

metric은 SpatialBenchmarking의 GenesMetrics convention과 동일 (PCC, SSIM, RMSE_zscore, JSD).

## 8. Training-time validation — `val_history.json`

train.py가 매 `val_every_epoch` 마다 emit하는 dict. 50개 항목 정도. 키는 모두 `val/` 로 시작하고 `/` 로 계층화된다.

### 8.1 Loss 계열

| Key | 의미 |
|---|---|
| `val/loss` | objective의 총 weighted loss |
| `val/tx_self/loss` | tx_self objective (MSM + 옵션들)만의 합 |
| `val/clean_msm/loss` | augmentation 없는 깨끗한 MSM 평가 (value_aug 영향 분리) |
| `val/align/loss` | image-RNA alignment loss (Stage 2 only) |

### 8.2 Masked Symbol 진단 (`val/tx_self/masked_symbol_*`, `val/clean_msm/...`)

| Key | 의미 |
|---|---|
| `masked_symbol_acc` (= top1_acc) | masked position에서 정답 token이 argmax인 비율 |
| `masked_symbol_top5/10_acc` | top-5/10 안에 정답 |
| `masked_symbol_ce` | cross-entropy loss (낮을수록 좋음) |
| `masked_symbol_ce_norm` | `ce / log(vocab_size)` — vocab size 무관 normalised CE |
| `masked_symbol_ce_gain` | `random_ce - actual_ce`. random baseline 대비 얼마나 좋아졌는지 (높을수록 좋음) |
| `masked_symbol_random_ce` | random uniform baseline의 CE |
| `masked_symbol_vocab_size` | vocab_size (sanity) |
| `n_masked_mean` | 한 spot당 평균 masked position 수 |
| `mask_actual_ratio` | 실제 mask ratio (목표값 ≈ 0.15) |

### 8.3 시퀀스 길이 진단 (`val/tx_self/*seq_len*`)

| Key | 의미 |
|---|---|
| `pre_sampling_seq_len_*` | sampling 전 non-zero gene 수의 분포 (min/p10/median/mean/p90/max) |
| `post_sampling_seq_len_*` | random_512 / top_k 등 sampling 후 길이 |
| `dataset_post_sampling_seq_len_mean` | dataset-level 평균 |
| `sampling_retention_ratio` | post / pre 평균 비율 — 정보 손실 정도 |

### 8.4 Intrinsic geometry (`val/intrinsic/*`, `val/{chunk,spot}_state/intrinsic/*`)

`hvg_linear_probe` 와 비슷한 metric을 매 epoch 빠르게 한 번 실행. 자세한 정의는 §3.2 참조.

### 8.5 Linear probe (`val/linear_probe/*`, `val/{chunk,spot}_state/linear_probe/*`)

| Key | 의미 |
|---|---|
| `hvg/pearson_mean`, `hvg/spearman_mean`, `hvg/r2_mean`, `hvg/rmse_norm` | high-variance HVG 256개에 대한 frozen Ridge probe (`hvg_linear_probe`) |
| `masked_hvg/*` | encoder 입력에서 target gene을 mask out한 후 같은 probe (`masked_hvg_linear_probe_from_encoder`). 정보가 다른 gene 컨텍스트로부터 회복되는지 |
| `hvg_rank/spot_rank_spearman` | spot 내부 gene 상대 순위가 보존되는지 (Geneformer 식) |
| `hvg_rank/bin_acc` | rank을 8개 bin으로 quantise한 분류 정확도 |
| `hvg_rank/top10_overlap` | spot 내부 top-10 gene 일치 비율 |

### 8.6 Leakage / batch (`val/leakage/*`)

| Key | 의미 |
|---|---|
| `source_knn/same_rate@20` | h_tx KNN 20개 중 같은 source 비율 (낮을수록 좋음) |
| `source_knn/entropy@20` | KNN source 분포의 entropy (높을수록 좋음) |
| `source_knn/n_spots` | 측정에 쓰인 spot 수 |

### 8.7 Gene set monitor (`val/set/*`)

지정한 marker gene set (예: endothelial, immune_T, proliferation 등)에 대해:

| Key | 의미 |
|---|---|
| `set/{name}/pcc_mean` | 그 set의 평균 expression 과 그 set 안의 individual gene의 Pearson 평균 |
| `set/{name}/coverage` | set 안의 gene 중 vocab에 존재하는 비율 |
| `set/{name}/cls_silhouette` | h_tx 공간에서 그 set이 한 cluster로 모이는지 (silhouette) |

### 8.8 Alignment & metric (`val/align/*`, `val/metric/*`) — Stage 2 only

| Key | 의미 |
|---|---|
| `align/diag_cos` | aligned image vs aligned tx의 cosine (paired) |
| `align/offdiag_cos` | aligned image vs aligned tx의 cosine (unpaired) |
| `align/diag_minus_off` | retrieval gap — 양수일수록 paired가 잘 떨어짐 |
| `metric/gene_tx_pcc` / `gene_tx_spearman` | gene reconstruction PCC/SCC |
| `metric/cosine_sim` | full pair cosine |

## 9. Metric scale 정리표

| Metric | scale-invariant 인가 | 어느 mode에서 보는 게 의미있나 |
|---|---|---|
| top-K accuracy | 분류 metric, scale 무관 | — |
| Spearman SCC | per-gene 양의 affine 변환에 invariant | raw / norm 동일 |
| Pearson | 양의 linear 변환에 invariant | raw / norm 동일 (clip 안 걸리면) |
| R² | 동일 (양의 linear 변환에 invariant) | raw / norm 동일 |
| RMSE_zscore | invariant (입력을 z-score 먼저) | raw / norm 동일 |
| RMSE_norm | invariant | raw / norm 동일 |
| **SSIM (1D)** | **NOT invariant** — min-max scaling에 의존 | norm이 encoder의 loss와 가까움 |
| **JSD** | **NOT invariant** — distribution shape에 민감 | norm |
| ARI / NMI / HOM / COM | 분류 metric, embedding scale 무관 | — |
| CHAOS / PAS / ASW_spatial | 공간 좌표만 본다, embedding scale 무관 | — |
| FIDE | 공간 좌표만 본다 | — |
| Moran's I / Geary's C | gene expression scale 의존 (그러나 standardise 내장) | 보통 raw로 발표 |
| effective rank | embedding 자체 — log1p_cp10k 와 무관 | — |
| `mvm_*` | value head의 직접 회귀 — scale 의존 | — |

## 10. 해석 priority 가이드

`스테이지 1 ckpt가 좋은가`를 빠르게 판단할 때 보는 순서:

1. `val_history.json[-1]/val/tx_self/masked_symbol_top10_acc` 가 random_ce 대비 `ce_gain ≥ 1.0` 인가
2. `val/linear_probe/hvg/spearman_mean` 가 0.3 이상으로 saturation 했는가
3. `val/leakage/source_knn/same_rate@20` 가 떨어지고 있는가
4. `val/intrinsic/effective_rank` 가 collapse하지 않는가

post-training은:

1. **MSM**: `top10_macro` (gene-balanced) 가 충분히 큰가
2. **DLPFC layer probe**: `h_tx layer_probe_acc` 가 0.6 이상인가
3. **DLPFC clustering**: `kmeans_ari`, `leiden_ari` (와 `gt_layer_fide` 대비 `kmeans_fide`)
4. **SVG**: 적어도 spatial-coords가 있는 sample 에서 `kendall_tau_morans ≥ 0.3` 인가
5. **gene_map**: 표준 marker (`MBP`, `SNAP25`, `PCP4`, `GFAP`) 에서 SCC ≥ 0.3 인가
