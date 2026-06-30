# Stage 1 configs — baseline vs main-candidate

Two parallel config sets live under `configs/stage1/`:

| File | Role |
|---|---|
| `data.yaml` / `model.yaml` / `train.yaml` | Shared by both runs (data pool, encoder shape, optimizer) |
| `experiment.yaml` | **Baseline** — conservative MSM, mask 0.15, value_aug=keep, JEPA off |
| `experiment_main.yaml` | **PDF main-candidate** — `B2 × C4 + Gene-JEPA E2` |
| `model_main.yaml` | Same as `model.yaml` plus `top_hvg_gene.value_aug = mixed` (C4) |

## Diff table — baseline → main-candidate

| Knob | YAML key | baseline | main-candidate |
|---|---|---|---|
| Mask ratio | `experiment.tokenizer.mask_ratio` | **0.15** (B1) | **0.30** (B2) |
| Value augmentation | `model.transcriptomics.top_hvg_gene.value_aug.mode` | `keep` (C1) | `mixed` (C4) |
| Value aug keep / noise / drop | `value_aug.{keep_p,noise_p,drop_p}` | n/a | 0.8 / 0.1 / 0.1 |
| Gene-level Predictive JEPA | `experiment.tx_self.enable_masked_jepa` | `false` | **`true`** |
| Gene-JEPA loss weight | `experiment.tx_self.jepa_weight` | 0.3 (idle, JEPA off) | **0.1** (E2 — paper λ) |
| MSM symbol weight | `experiment.tx_self.symbol_weight` | 1.0 | 1.0 |
| MVM (value reconstruction) weight | `experiment.tx_self.value_weight` | 0.0 (downstream only) | 0.0 |
| `mask_kind` | `experiment.tx_self.masking_obj` | `symbol` | `symbol` |
| EMA momentum (teacher) | `experiment.tx_self.ema_momentum` | 0.999 | 0.999 |

PDF sections this corresponds to: p.4 (foundation), p.6 (B/C grid),
p.8 (E series).

## Launch

```bash
# Baseline
bash scripts/train/stage1.sh

# Main-candidate
bash scripts/train/stage1_main.sh
```

Both runs reuse the SAME `data.yaml` / `train.yaml`; only the experiment
+ model (for value_aug) yaml differ.  `train.py` reads:

- `experiment.tx_self.{masking_obj,symbol_weight,value_weight,enable_masked_jepa,jepa_weight,ema_momentum}`
- `experiment.tokenizer.mask_ratio`
- `model.transcriptomics.top_hvg_gene.value_aug.*`

(These are the keys ablation scripts also override — see `scripts/ablation/`.)
