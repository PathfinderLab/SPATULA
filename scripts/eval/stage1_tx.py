"""Unified Stage-1 tx_encoder evaluation.

Compares multiple `ckpt_tx_encoder_best.pt` checkpoints (typically the
outputs of different ablation runs) on a shared evaluation suite.

Metrics computed per ckpt:
  1. Intrinsic representation health + expression-manifold preservation.
  2. Short downstream probes: frozen h_tx -> HVG, masked-HVG values,
     and relative expression rank/bin recovery.
  3. Gene-token embedding alignment with ground-truth gene-gene correlation.
  4. Clean MVM value-head imputation as a downstream reconstruction check.
  5. Source leakage checks; lower same-source kNN / source probe is better.
  6. Optional HEST organ probe/retrieval via --include-organ-probe.

Outputs `results/eval/tx_compare.csv` with one row per ckpt.

Usage:
    python scripts/eval/stage1_tx.py \\
        --prepared-dir results/cache/prepared_expanded \\
        --ckpts results/runs/stage1_obj_msm/ckpt_tx_encoder_best.pt \\
                 results/runs/stage1_obj_msm_jepa/ckpt_tx_encoder_best.pt \\
        --val-samples 76 --pool-spots 5000
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# scripts/eval/stage1_tx.py → repo root is parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_align.utils import get_logger
from mm_align.data.gene_norm import GeneNormalizer
from mm_align.evaluation.stage1_benchmarks import (
    embedding_health_metrics,
    expression_manifold_metrics,
    gene_embedding_correlation_alignment,
    gene_embeddings_from_encoder,
    hvg_linear_probe,
    hvg_rank_probe,
    masked_hvg_linear_probe_from_encoder,
    chunk_view_embeddings_from_encoder,
    source_knn_leakage_metrics,
)

log = get_logger("eval_tx")


# ─────────────────────────────────────────────────────────────────────────
# Load a tx_encoder from a saved ckpt, plus the input-pipeline knobs
# (vocab_clip + gene_norm) it expects at inference time.
# ─────────────────────────────────────────────────────────────────────────



def load_ablation_meta(ckpt_path: Path) -> dict:
    """Flatten per-run ablation metadata into CSV-friendly columns."""
    meta_path = ckpt_path.parent / "ablation_meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return {}
    out = {
        "ablation_group": meta.get("ablation_group", ""),
        "ablation_variant": meta.get("ablation_variant", ""),
        "ablation_profile": meta.get("profile", ""),
    }
    changed = meta.get("changed_args", {})
    if isinstance(changed, dict):
        out["ablation_changed_args"] = json.dumps(changed, sort_keys=True, ensure_ascii=False)
        for k, v in changed.items():
            col = "arg_" + str(k).replace(".", "_").replace("/", "_")
            out[col] = v
    else:
        out["ablation_changed_args"] = str(changed)
    speed = meta.get("speed_overrides", {}) if isinstance(meta.get("speed_overrides", {}), dict) else {}
    train = meta.get("train_overrides", {}) if isinstance(meta.get("train_overrides", {}), dict) else {}
    out["speed_limit_train"] = speed.get("limit_train", "")
    out["speed_vocab_clip_keep_indices"] = speed.get("vocab_clip_keep_indices", "")
    out["speed_max_seq_len"] = speed.get("max_seq_len", "")
    out["speed_sampling_strategy"] = speed.get("sampling_strategy", "")
    out["train_epochs"] = train.get("epochs", "")
    out["train_val_every_epoch"] = train.get("val_every_epoch", "")
    return out


def _recover_cfg_data(ckpt_path: Path, sd: dict) -> dict:
    """Pull `data` config from ckpt, falling back to <run_dir>/config.json.

    Older Stage-1 ckpts persisted only `prepared_dir` in `cfg_tx['data']`
    (see src/mm_align/training/checkpoints.py before 2026-06-14).  The
    run-dir config.json always carries the full data block.
    """
    data = {}
    # New `cfg_tx['data']` (with gene_norm + vocab_clip after the fix).
    cfg_tx = sd.get("cfg_tx") or sd.get("cfg", {})
    if isinstance(cfg_tx, dict) and "data" in cfg_tx:
        data.update(cfg_tx["data"])
    # Stage 1 train.py persists `config.json` next to the ckpt.
    cfg_json = ckpt_path.parent / "config.json"
    if cfg_json.exists():
        try:
            data_fallback = json.loads(cfg_json.read_text()).get("data", {})
            for k, v in data_fallback.items():
                data.setdefault(k, v)
        except Exception:
            pass
    return data


def load_tx_encoder(ckpt_path: Path, device: str = "cuda", tx_pooling_mode: str | None = None):
    """Returns (encoder, full_cfg, vocab_keep, gene_norm) — caller applies the
    last two to every input before encoding so that each ckpt's pipeline is
    reproduced, regardless of vocab clip / normalisation choice."""
    from mm_align.models.tx.factory import build_tx_encoder
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # `build_tx_encoder` consumes the full cfg dict ({"model": ..., "data": ...}).
    full_cfg = sd.get("cfg_tx") or sd.get("cfg")
    if full_cfg is None:
        # Legacy: only `tx_config` saved.  Try to reconstruct.
        tx_cfg = sd.get("tx_config")
        if tx_cfg is None:
            raise RuntimeError(f"{ckpt_path} has neither cfg_tx/cfg nor tx_config")
        # Fall back to config.json next to the ckpt.
        cfg_json = ckpt_path.parent / "config.json"
        if not cfg_json.exists():
            raise RuntimeError(f"{ckpt_path}: cannot recover full cfg "
                                f"(no cfg_tx/cfg in ckpt, no {cfg_json})")
        full_cfg = json.loads(cfg_json.read_text())
    full_cfg = copy.deepcopy(full_cfg)
    if tx_pooling_mode and tx_pooling_mode not in ("", "ckpt"):
        gcfg = (full_cfg.setdefault("model", {})
                        .setdefault("transcriptomics", {})
                        .setdefault("top_hvg_gene", {}))
        gcfg["pooling_mode"] = tx_pooling_mode
    if "model" not in full_cfg or "embed_dim" not in full_cfg["model"]:
        # Recover from neighbouring config.json (older ckpts saved partial cfg).
        cfg_json = ckpt_path.parent / "config.json"
        if cfg_json.exists():
            jc = json.loads(cfg_json.read_text())
            full_cfg.setdefault("model", {}).update(jc.get("model", {}))
            full_cfg.setdefault("data", {}).update(jc.get("data", {}))
    enc = build_tx_encoder(full_cfg)
    # State dict — supports tx_encoder_only / trainable_only / legacy.
    state = sd.get("tx_encoder") or sd.get("tx_state_dict") or sd.get("model", {})
    missing, _ = enc.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"{ckpt_path.name}: {len(missing)} missing keys (head/aux ok)")
    enc.eval().to(device)

    # Recover the input pipeline so this ckpt's eval matches its training.
    data_cfg = _recover_cfg_data(ckpt_path, sd)
    vocab_keep = None
    vc = data_cfg.get("vocab_clip") or {}
    keep_path = vc.get("keep_indices_path") if isinstance(vc, dict) else None
    if keep_path and Path(keep_path).exists():
        vocab_keep = np.load(keep_path)
    gene_norm_cfg = data_cfg.get("gene_norm")
    log.info(f"  {ckpt_path.parent.name}: "
              f"vocab_keep={'-' if vocab_keep is None else len(vocab_keep)}, "
              f"gene_norm={'-' if not gene_norm_cfg else gene_norm_cfg.get('mode')}, "
              f"pooling={getattr(enc, 'pooling_mode', '-')}")
    return enc, full_cfg, vocab_keep, gene_norm_cfg


# ─────────────────────────────────────────────────────────────────────────
# Eval suite
# ─────────────────────────────────────────────────────────────────────────

def _prepare_input(hvg_raw: np.ndarray, vocab_keep: np.ndarray | None,
                    normalizer: GeneNormalizer | None) -> np.ndarray:
    """Apply the ckpt's vocab clip + gene_norm to a raw (N, D_full) hvg pool.
    Returns float32, shape (N, D_eff)."""
    x = hvg_raw
    if vocab_keep is not None:
        x = x[:, vocab_keep]
    if normalizer is not None and normalizer:
        x = normalizer.apply_np(x)
    return x.astype(np.float32, copy=False)



def _safe_eval_batch(requested: int, x_norm: np.ndarray) -> int:
    """Clamp eval batch from observed clean sequence length before CUDA alloc."""
    requested = max(1, int(requested))
    if x_norm.size == 0:
        return requested
    nz = (x_norm != 0).sum(axis=1)
    p95 = int(np.percentile(nz, 95)) if nz.size else 0
    if p95 >= 4096:
        cap = 1
    elif p95 >= 2048:
        cap = 2
    elif p95 >= 1024:
        cap = 4
    elif p95 >= 512:
        cap = 8
    else:
        cap = 16
    return max(1, min(requested, cap))


@torch.no_grad()
def encode_pool(enc, hvg: np.ndarray, vocab_keep: np.ndarray | None,
                  normalizer: GeneNormalizer | None,
                  batch: int = 64, device: str = "cuda") -> np.ndarray:
    """Encode an (N, D_full) raw-log1p pool using memory-safe micro-batches."""
    x_norm = _prepare_input(hvg, vocab_keep, normalizer)
    cur_batch = _safe_eval_batch(batch, x_norm)
    if cur_batch < max(1, int(batch)):
        log.info("encode_pool auto-clamped encode_batch=%s -> %s for long clean sequences", batch, cur_batch)
    out = []
    r0 = 0
    while r0 < x_norm.shape[0]:
        try:
            x = torch.from_numpy(x_norm[r0:r0 + cur_batch]).to(device)
            out.append(enc(novae_latent=None, hvg=x)["h_tx"].cpu().numpy())
            r0 += cur_batch
        except torch.cuda.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if cur_batch <= 1:
                raise
            cur_batch = max(1, cur_batch // 2)
            log.warning("encode_pool OOM; retrying with batch=%d", cur_batch)
    return np.concatenate(out, axis=0)

def linear_probe(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """5-fold linear-probe top-1 acc + macro-F1.

    Wraps LogisticRegression in a Pipeline with StandardScaler so LBFGS
    actually converges within `max_iter=1000` on raw 512-d embeddings
    (without scaling, the `ConvergenceWarning` you saw is endemic).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    if len(np.unique(y)) < 2:
        return {"acc": float("nan"), "f1_macro": float("nan")}
    accs, f1s = [], []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=4),
        ).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return {"acc": float(np.mean(accs)), "f1_macro": float(np.mean(f1s))}


def retrieval_recall(X: np.ndarray, y: np.ndarray, ks=(1, 5, 10)) -> dict:
    """For each spot, find k nearest spots in embedding space; fraction with same label."""
    from sklearn.neighbors import NearestNeighbors
    out = {}
    if len(np.unique(y)) < 2:
        return {f"recall_at_{k}": float("nan") for k in ks}
    nn = NearestNeighbors(n_neighbors=max(ks) + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                     # drop self
    for k in ks:
        nbr_labels = y[idx[:, :k]]
        hit = (nbr_labels == y[:, None]).any(axis=1).mean()
        out[f"recall_at_{k}"] = float(hit)
    return out


def collapse_metrics(X: np.ndarray) -> dict:
    """Back-compat wrapper for older CSV column names."""
    m = embedding_health_metrics(X, prefix="intrinsic")
    return {k.removeprefix("intrinsic/"): v for k, v in m.items()}


def masked_value_imputation(enc, hvg_raw: np.ndarray,
                              vocab_keep: np.ndarray | None,
                              normalizer: GeneNormalizer | None,
                              mask_ratio: float = 0.30,
                              batch: int = 16, device: str = "cuda") -> dict:
    """MVM as DOWNSTREAM task — pipeline-aware version.

    Pipeline (matches Stage 1 training exactly):
        raw_hvg → vocab_clip → gene_norm → mask non-zeros → encoder → value_head

    Returns vocab-scale-INVARIANT metrics so vocab-clip ablations compare
    fairly:
        mvm_pearson       — corr(pred, target) over masked tokens          [vocab-invariant]
        mvm_r2            — coefficient of determination                    [vocab-invariant]
        mvm_mse_per_tok   — MSE averaged per masked token                   [partially invariant — depends on input scale]
        mvm_rmse_norm     — RMSE / target_std (scale-invariant version)     [vocab-invariant]
    """
    from mm_align.models.tx.top_hvg_gene import TopHVGGeneEncoder
    if not isinstance(enc, TopHVGGeneEncoder) or not hasattr(enc, "value_head"):
        return {"mvm_pearson": float("nan"), "mvm_spearman": float("nan"), "mvm_r2": float("nan"),
                "mvm_mse_per_tok": float("nan"), "mvm_rmse_norm": float("nan")}
    x_full = _prepare_input(hvg_raw, vocab_keep, normalizer)         # (N, D_eff)
    pred_all, tgt_all = [], []
    cur_batch = _safe_eval_batch(batch, x_full)
    if cur_batch < max(1, int(batch)):
        log.info("MVM auto-clamped encode_batch=%s -> %s for long clean sequences", batch, cur_batch)
    r0 = 0
    while r0 < x_full.shape[0]:
        try:
            x = torch.from_numpy(x_full[r0:r0 + cur_batch]).to(device)
            nz = (x != 0)               # nonzero in normalised space; nonzero_z preserves zeros
            rand = torch.rand_like(x.float())
            mask_pos = nz & (rand < mask_ratio)
            x_masked = torch.where(mask_pos, torch.zeros_like(x), x)
            with torch.no_grad():
                out = enc(novae_latent=None, hvg=x_masked)
                # Encoder output uses `per_token` (legacy aliased `tx_per_token`).
                per_tok = out.get("per_token") if out.get("per_token") is not None else out.get("tx_per_token")
                orig_pos = out.get("orig_positions")
                if per_tok is None or orig_pos is None:
                    return {"mvm_pearson": float("nan"), "mvm_spearman": float("nan"), "mvm_r2": float("nan"),
                            "mvm_mse_per_tok": float("nan"), "mvm_rmse_norm": float("nan")}
                val_pred_per_tok = enc.value_head(per_tok).squeeze(-1)
                val_pred = torch.zeros_like(x).scatter_(1, orig_pos, val_pred_per_tok)
            pred_all.append(val_pred[mask_pos].cpu().numpy())
            tgt_all.append(x[mask_pos].cpu().numpy())
            r0 += cur_batch
        except torch.cuda.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if cur_batch <= 1:
                raise
            cur_batch = max(1, cur_batch // 2)
            log.warning("MVM OOM; retrying with batch=%d", cur_batch)
    if not pred_all or sum(p.size for p in pred_all) < 2:
        return {"mvm_pearson": float("nan"), "mvm_spearman": float("nan"), "mvm_r2": float("nan"),
                "mvm_mse_per_tok": float("nan"), "mvm_rmse_norm": float("nan")}
    p = np.concatenate(pred_all); t = np.concatenate(tgt_all)
    mse = float(((p - t) ** 2).mean())
    pearson = float(np.corrcoef(p, t)[0, 1])
    from mm_align.evaluation.stage1_benchmarks import _spearman
    spearman = _spearman(p, t)
    t_var = float(t.var())
    r2 = float(1.0 - mse / max(t_var, 1e-12))
    rmse_norm = float(np.sqrt(mse) / max(np.sqrt(t_var), 1e-12))
    return {"mvm_pearson": pearson, "mvm_spearman": spearman, "mvm_r2": r2,
            "mvm_mse_per_tok": mse, "mvm_rmse_norm": rmse_norm}


# ─────────────────────────────────────────────────────────────────────────
# Pool builder — sample N spots from val shards
# ─────────────────────────────────────────────────────────────────────────

def build_eval_pool(prepared: Path, n_samples: int, n_spots: int,
                      split: str = "test", seed: int = 0,
                      return_meta: bool = False):
    """Source-balanced eval pool builder.

    Picks `splits[split]` (default `test` — the held-out set the encoder
    never sees, not even for early stopping).  `val` was used during Stage-1
    training for ES, so reporting numbers on `val` would be a leak.

    splits[*] is alphabetical, which skews the first N IDs toward
    spatialcorpus/ST1K (no organ labels) — `--val-samples 20` would deliver
    zero HEST samples and make organ_probe / retrieval metrics meaningless.
    Fix: bucket IDs by source first, then draw a round-robin sample so
    HEST is always represented when it exists.  HEST is prioritised because
    organ labels live only in HEST metadata.
    """
    splits = json.loads((prepared / "splits.json").read_text())
    if split not in splits:
        raise ValueError(f"--split={split!r} not in splits.json (keys: {list(splits)})")
    val_ids = splits[split]
    try:
        from mm_align.evaluation.labels import hest_metadata
        meta = hest_metadata()
    except Exception:
        meta = None

    # Bucket by source — pick whichever shard exists for each id.
    buckets: dict[str, list[tuple[str, str]]] = {"hest": [], "st1k": [], "spatialcorpus": []}
    for sid in val_ids:
        for suf, src in [("", "hest"), (".st1k", "st1k"), (".spatialcorpus", "spatialcorpus")]:
            if (prepared / f"{sid}{suf}.h5").exists():
                buckets[src].append((sid, suf))
                break

    # Round-robin pick, HEST first.
    order = ["hest", "st1k", "spatialcorpus"]
    chosen: list[tuple[str, str, str]] = []
    while len(chosen) < n_samples:
        progress = False
        for src in order:
            if buckets[src]:
                sid, suf = buckets[src].pop(0)
                chosen.append((sid, suf, src))
                progress = True
                if len(chosen) >= n_samples:
                    break
        if not progress:
            break

    rng = np.random.default_rng(seed)
    per = max(1, n_spots // max(len(chosen), 1))
    hvg_list, organ_list, src_list = [], [], []
    coord_list, sample_list = [], []
    for sid, suf, src in chosen:
        p = prepared / f"{sid}{suf}.h5"
        with h5py.File(p, "r") as f:
            if "hvg_log" not in f:
                continue
            n = f["hvg_log"].shape[0]
            take = min(n, per)
            sel = np.sort(rng.choice(n, take, replace=False))
            hvg_list.append(f["hvg_log"][sel].astype(np.float32))
            if "coords" in f:
                coord_list.append(f["coords"][sel].astype(np.float32))
            else:
                coord_list.append(np.full((take, 2), np.nan, dtype=np.float32))
        organ = "Unknown"
        # hest_metadata() returns {column_name: {sid: value, ...}}.
        if src == "hest" and isinstance(meta, dict):
            organ = str(meta.get("organ", {}).get(sid, "Unknown"))
        organ_list.extend([organ] * take)
        src_list.extend([src] * take)
        sample_list.extend([sid] * take)
    hvg = np.concatenate(hvg_list, axis=0)
    organ = np.array(organ_list)
    source = np.array(src_list)
    log.info(f"pool[{split}]: HEST={(source=='hest').sum()}  ST1K={(source=='st1k').sum()}  "
              f"spatialcorpus={(source=='spatialcorpus').sum()}  organs={len(np.unique(organ))}")
    if return_meta:
        meta = {
            "coords": np.concatenate(coord_list, axis=0) if coord_list else np.empty((0, 2), dtype=np.float32),
            "sample_id": np.array(sample_list),
        }
        return hvg, organ, source, meta
    return hvg, organ, source


# Back-compat alias.
build_val_pool = build_eval_pool



# ─────────────────────────────────────────────────────────────────────────
# Qualitative test visualisations
# ─────────────────────────────────────────────────────────────────────────

def _load_gene_names(prepared: Path, vocab_keep: np.ndarray | None = None) -> list[str]:
    p = prepared / "hvg_vocab.json"
    if p.exists():
        genes = json.loads(p.read_text())
    else:
        genes = [f"gene_{i}" for i in range(0 if vocab_keep is None else int(len(vocab_keep)))]
    if vocab_keep is not None:
        genes = [genes[int(i)] for i in vocab_keep]
    return [str(g) for g in genes]


def _safe_scanpy_umap(emb: np.ndarray, labels: dict[str, np.ndarray], out_prefix: Path,
                      *, max_spots: int = 5000, seed: int = 0,
                      n_neighbors: int = 50, min_dist: float = 0.3,
                      pca_n: int = 50) -> None:
    """UMAP of an embedding matrix, coloured by every column in `labels`.

    Two robustness knobs vs. the naive default:
      - PCA pre-reduction (default 50 PCs) before computing neighbours.  Raw
        h_tx is 512-D and per-sample-clustered; UMAP on the raw space draws
        thin disconnected strings (one per sample).  PCA→UMAP recovers the
        global topology.
      - Larger n_neighbors (50) and min_dist (0.3) to spread tight clusters.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(emb.shape[0])
    # For label-coloured UMAPs, drop Unknown/NA labels when possible.  Unknown
    # mostly comes from non-HEST spots in organ plots and otherwise dominates
    # the legend without telling us whether the representation separates biology.
    if labels:
        keep = np.ones(emb.shape[0], dtype=bool)
        for vals in labels.values():
            vv = np.asarray(vals).astype(str)
            known = ~np.isin(np.char.lower(vv), ["unknown", "unk", "na", "nan", "none", ""])
            if known.any():
                keep &= known
        if keep.sum() >= 50:
            idx = idx[keep]
    if idx.size > max_spots:
        idx = np.sort(rng.choice(idx, max_spots, replace=False))
    X = emb[idx].astype(np.float32, copy=False)
    obs = {k: np.asarray(v)[idx].astype(str) for k, v in labels.items()}
    try:
        import scanpy as sc
        import anndata as ad
        adata = ad.AnnData(X=X, obs=pd.DataFrame(obs))
        # PCA pre-reduction — scanpy's default neighbour pipeline.
        eff_pca = min(int(pca_n), X.shape[1], max(2, X.shape[0] - 1))
        if X.shape[1] > eff_pca and eff_pca > 2:
            sc.pp.pca(adata, n_comps=eff_pca, random_state=seed)
            use_rep = "X_pca"
        else:
            use_rep = "X"
        eff_neigh = min(int(n_neighbors), max(2, X.shape[0] - 1))
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=eff_neigh,
                         random_state=seed)
        sc.tl.umap(adata, min_dist=float(min_dist), random_state=seed)
        for key in obs:
            sc.pl.umap(adata, color=key, show=False, frameon=False, size=12)
            plt.savefig(out_prefix.parent / f"{out_prefix.name}_umap_by_{key}.png", dpi=160, bbox_inches="tight")
            plt.close()
    except Exception as e:
        log.warning(f"scanpy UMAP failed ({e}); using sklearn/umap fallback")
        try:
            import umap  # type: ignore
            xy = umap.UMAP(n_components=2, n_neighbors=min(30, max(2, X.shape[0] - 1)),
                           min_dist=0.1, random_state=seed).fit_transform(X)
        except Exception:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=seed).fit_transform(X)
        for key, vals in obs.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            uniq = np.unique(vals)
            cmap = plt.cm.tab20.colors if len(uniq) <= 20 else plt.cm.viridis(np.linspace(0, 1, len(uniq)))
            for i, u in enumerate(uniq):
                m = vals == u
                ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.7, color=cmap[i % len(cmap)], label=str(u))
            if len(uniq) <= 15:
                ax.legend(fontsize=7, markerscale=2)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"UMAP by {key}")
            fig.tight_layout()
            fig.savefig(out_prefix.parent / f"{out_prefix.name}_umap_by_{key}.png", dpi=160)
            plt.close(fig)


def _plot_label_barplots(organ: np.ndarray, source: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, vals, title in [(axes[0], source, "source"), (axes[1], organ, "organ")]:
        vc = pd.Series(vals.astype(str)).value_counts().head(20).sort_values()
        ax.barh(vc.index, vc.values, color="#4c78a8")
        ax.set_title(f"Eval pool {title} counts")
        ax.set_xlabel("spots")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_spatial_gene_panels(hvg_eff: np.ndarray, coords: np.ndarray, sample_id: np.ndarray,
                              genes: list[str], requested: list[str], out_dir: Path,
                              *, max_samples: int = 3) -> None:
    valid_xy = np.isfinite(coords).all(axis=1)
    if not valid_xy.any() or not requested:
        return
    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    chosen_genes = [g for g in requested if g.upper() in gene_to_idx]
    if not chosen_genes:
        log.warning(f"no requested viz genes found in effective vocab: {requested}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in list(dict.fromkeys(sample_id[valid_xy].tolist()))[:max_samples]:
        m = valid_xy & (sample_id == sid)
        if int(m.sum()) < 10:
            continue
        n = len(chosen_genes)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False, constrained_layout=True)
        for j, g in enumerate(chosen_genes):
            vals = hvg_eff[m, gene_to_idx[g.upper()]]
            ax = axes[0, j]
            sc = ax.scatter(coords[m, 0], coords[m, 1], c=vals, s=7, cmap="viridis")
            ax.set_title(g)
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"GT expression spatial maps | {sid}")
        fig.savefig(out_dir / f"{sid}_gt_gene_maps.png", dpi=170)
        plt.close(fig)


def render_stage1_test_viz(*, out_dir: Path, ckpt_name: str, emb: np.ndarray,
                           hvg_eff: np.ndarray, organ: np.ndarray, source: np.ndarray,
                           meta: dict, genes: list[str], viz_genes: list[str],
                           max_spots: int = 5000) -> None:
    ck_dir = out_dir / ckpt_name
    ck_dir.mkdir(parents=True, exist_ok=True)
    _safe_scanpy_umap(
        emb,
        {"source": source, "organ": organ, "sample_id": meta.get("sample_id", np.array([""] * emb.shape[0]))},
        ck_dir / "embedding",
        max_spots=max_spots,
    )
    _plot_label_barplots(organ, source, out_dir / "eval_pool_barplots.png")
    _plot_spatial_gene_panels(
        hvg_eff, meta.get("coords", np.empty((0, 2))), meta.get("sample_id", np.array([])),
        genes, viz_genes, out_dir / "gt_only_spatial_gene_maps",
    )

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def _render_probe_figures(*, out_dir: Path,
                            per_gene_linear: dict | None,
                            per_gene_held_out: dict | None) -> None:
    """Per-gene Pearson histogram (+ SSIM if present) for both probe variants
    side-by-side, plus a Pearson vs SSIM scatter when both metrics are
    available.  Helps decide if a model is uniformly mediocre or strong on a
    minority of genes (long-tail).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    if per_gene_linear:
        panels.append(("linear", per_gene_linear))
    if per_gene_held_out:
        panels.append(("gene_held_out", per_gene_held_out))
    if not panels:
        return
    # 1) Per-gene Pearson histogram.
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                              constrained_layout=True, squeeze=False)
    for ax, (name, pg) in zip(axes[0], panels):
        pearson = pg.get("pearson", np.array([]))
        valid = pearson[np.isfinite(pearson)]
        if valid.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(valid, bins=30, color="#4c78a8", edgecolor="white")
        med = float(np.median(valid))
        ax.axvline(med, color="#e15759", linestyle="--", linewidth=1,
                    label=f"median = {med:.3f}")
        ax.set_xlabel("per-gene Pearson")
        ax.set_ylabel("# genes")
        ax.set_title(f"{name} probe  (n={valid.size})")
        ax.legend()
    fig.savefig(out_dir / "per_gene_pearson_hist.png", dpi=160)
    plt.close(fig)
    # 2) Pearson vs SSIM scatter — only if both present (spatial_bench mode).
    for name, pg in panels:
        pearson = pg.get("pearson")
        ssim = pg.get("ssim")
        if pearson is None or ssim is None:
            continue
        valid = np.isfinite(pearson) & np.isfinite(ssim)
        if valid.sum() < 5:
            continue
        fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        ax.scatter(pearson[valid], ssim[valid], s=10, alpha=0.6, color="#4c78a8")
        ax.set_xlabel("per-gene Pearson")
        ax.set_ylabel("per-gene SSIM (1-D)")
        ax.set_title(f"{name} probe  | n={int(valid.sum())} genes")
        ax.grid(alpha=0.3)
        fig.savefig(out_dir / f"pearson_vs_ssim_{name}.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=("test", "val", "train"),
                     help="splits.json key to use.  Default 'test' = true holdout. "
                          "'val' was used during Stage-1 ES so it's a soft leak.")
    ap.add_argument("--val-samples", type=int, default=76,
                     help="(deprecated name; applies to whichever --split is set).")
    ap.add_argument("--pool-spots", type=int, default=5000)
    ap.add_argument("--encode-batch", type=int, default=64,
                     help="Micro-batch size for tx_encoder inference during eval. Lower on OOM.")
    ap.add_argument("--tx-pooling-mode", default="ckpt",
                     choices=("ckpt", "cls", "token_mean", "cls_token_mean_sum", "cls_token_mean_avg",
                              "cls_mean_sum", "cls_mean_avg", "mean"),
                     help="Override top_hvg_gene spot readout at eval time. 'ckpt' preserves saved config.")
    ap.add_argument("--linear-probe-genes", type=int, default=256,
                     help="Number of high-variance HVG targets for frozen h_tx -> expression probe.")
    ap.add_argument("--neural-linear-probe", action="store_true",
                     help="Also fit a trainable nn.Linear head for 50-ish epochs with early stopping. "
                          "This is slower than Ridge and intended for final/report eval.")
    ap.add_argument("--neural-probe-epochs", type=int, default=50)
    ap.add_argument("--neural-probe-patience", type=int, default=8)
    ap.add_argument("--neural-probe-lr", type=float, default=1e-3)
    ap.add_argument("--neural-probe-weight-decay", type=float, default=1e-4)
    ap.add_argument("--neural-probe-batch", type=int, default=512)
    ap.add_argument("--probe-pca-n", type=int, default=0,
                     help="Optional PCA(n) on h_tx before Ridge — HEST/SEAL use 256. 0 = no PCA.")
    ap.add_argument("--probe-alpha", default="1.0",
                     help="Ridge alpha. Float, or 'auto' for HEST/SEAL formula 100/(features*targets).")
    ap.add_argument("--probe-metric-suite", default="spatial_bench",
                     choices=("legacy", "spatial_bench"),
                     help="legacy = pearson/spearman/r2/rmse_norm. "
                          "spatial_bench additionally emits SSIM, JSD, RMSE_zscore + Q1/Q3 across genes.")
    ap.add_argument("--gene-held-out-folds", type=int, default=0,
                     help="If > 0, also run a gene-fold CV imputation probe with K folds "
                          "(target genes split into K disjoint groups, each masked at input then "
                          "predicted from the resulting embedding). Comparable to SpatialBenchmarking.")
    ap.add_argument("--probe-figures", action="store_true",
                     help="Render per-gene Pearson histogram + per-gene scatter figure for the "
                          "first ckpt-rep pair.")
    ap.add_argument("--include-organ-probe", action="store_true",
                     help="Run HEST organ probe/retrieval. Off by default because it is often trivial.")
    ap.add_argument("--skip-source-probe", action="store_true",
                     help="Skip source classifier leakage check.")
    ap.add_argument("--out", default="results/eval/tx_compare.csv")
    ap.add_argument("--make-viz", action="store_true",
                     help="Write qualitative test artifacts: UMAPs, label barplots, spatial GT gene maps.")
    ap.add_argument("--viz-out-dir", default=None,
                     help="Directory for qualitative artifacts. Default: <out_csv_stem>_viz next to --out.")
    ap.add_argument("--viz-max-spots", type=int, default=5000)
    ap.add_argument("--viz-genes", nargs="*", default=["MKI67", "EPCAM", "COL1A1", "CD3D"],
                     help="Genes for GT spatial expression panels when coords exist.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prep = Path(args.prepared_dir)
    log.info(f"building eval pool [{args.split}] "
              f"({args.pool_spots} spots from {args.val_samples} samples)...")
    pool = build_eval_pool(prep, args.val_samples, args.pool_spots, split=args.split,
                           return_meta=bool(args.make_viz))
    if args.make_viz:
        hvg, organ, source, pool_meta = pool
    else:
        hvg, organ, source = pool
        pool_meta = {}
    log.info(f"pool: {hvg.shape} | organs={len(np.unique(organ))} | sources={list(np.unique(source))}")

    rows = []
    full_hvg_dim = hvg.shape[1]
    # HEST-only subset for organ probe / retrieval — non-HEST spots have
    # `organ='Unknown'`, which becomes a single dominant class and lets the
    # probe "cheat" by predicting Unknown.  We keep those spots in `source`
    # so the source-probe (batch-leakage) signal stays meaningful.
    hest_mask = (source == "hest") & (organ != "Unknown")
    n_hest = int(hest_mask.sum())
    if args.include_organ_probe:
        log.info(f"organ probe will run on {n_hest} HEST spots "
                 f"({len(np.unique(organ[hest_mask]))} organs).")
    else:
        log.info("organ probe disabled by default; use --include-organ-probe to run it.")
    for ck in args.ckpts:
        ck = Path(ck)
        log.info(f"== {ck} ==")
        enc, cfg, vocab_keep, gene_norm_cfg = load_tx_encoder(ck, device=device, tx_pooling_mode=args.tx_pooling_mode)
        d_eff = int(len(vocab_keep)) if vocab_keep is not None else full_hvg_dim
        normalizer = GeneNormalizer(
            gene_norm_cfg, full_hvg_dim=full_hvg_dim, hvg_dim=d_eff,
            vocab_keep_indices=vocab_keep,
        ) if gene_norm_cfg else None
        hvg_eval = _prepare_input(hvg, vocab_keep, normalizer)
        emb = encode_pool(enc, hvg, vocab_keep, normalizer,
                          batch=args.encode_batch, device=device)
        if args.make_viz:
            viz_root = Path(args.viz_out_dir or (Path(args.out).with_suffix("").as_posix() + "_viz"))
            genes_eff = _load_gene_names(prep, vocab_keep)
            try:
                render_stage1_test_viz(
                    out_dir=viz_root,
                    ckpt_name=ck.parent.name,
                    emb=emb,
                    hvg_eff=hvg_eval,
                    organ=organ,
                    source=source,
                    meta=pool_meta,
                    genes=genes_eff,
                    viz_genes=args.viz_genes,
                    max_spots=args.viz_max_spots,
                )
            except Exception as e:
                log.warning(f"stage1 qualitative viz failed for {ck.parent.name}: {e}", exc_info=True)
        row = {"ckpt": ck.parent.name, "ckpt_path": str(ck),
                "split": args.split,
                "embed_dim": emb.shape[1], "input_dim": d_eff,
                "gene_norm": (gene_norm_cfg or {}).get("mode", "none"),
                "n_hest": n_hest, "n_all": int(emb.shape[0])}
        row.update(load_ablation_meta(ck))

        # 1. intrinsic representation health + expression manifold      [label-free]
        intrinsic = {}
        intrinsic.update(embedding_health_metrics(emb, prefix="intrinsic"))
        intrinsic.update(expression_manifold_metrics(
            emb, hvg_eval,
            max_spots=min(args.pool_spots, 2000),
            k=20,
            prefix="intrinsic/expression",
        ))
        gene_emb = gene_embeddings_from_encoder(enc)
        if gene_emb is not None:
            intrinsic.update(gene_embedding_correlation_alignment(
                hvg_eval, gene_emb,
                n_genes=min(512, hvg_eval.shape[1]),
                prefix="intrinsic/gene_embedding",
            ))
        for k, v in intrinsic.items():
            row[k.replace("/", "_")] = v
        # Back-compat flat columns used by older notebooks/CSVs.
        row.update(collapse_metrics(emb))

        # 2. short downstream linear probes: frozen h_tx -> HVG values  [higher = better]
        _probe_alpha = (args.probe_alpha if args.probe_alpha == "auto"
                         else float(args.probe_alpha))
        _probe_pca = int(args.probe_pca_n) if int(args.probe_pca_n) > 0 else None
        lp_res = hvg_linear_probe(
            emb, hvg_eval,
            n_targets=args.linear_probe_genes,
            max_spots=args.pool_spots,
            alpha=_probe_alpha,
            pca_n=_probe_pca,
            metric_suite=args.probe_metric_suite,
            prefix="linear_probe/hvg",
            return_per_gene=args.probe_figures,
        )
        if args.probe_figures:
            lp, lp_pg = lp_res
        else:
            lp, lp_pg = lp_res, None
        lp.update(hvg_rank_probe(
            emb, hvg_eval,
            n_targets=args.linear_probe_genes,
            max_spots=args.pool_spots,
            prefix="linear_probe/hvg_rank",
        ))
        lp.update(masked_hvg_linear_probe_from_encoder(
            enc, hvg_eval,
            n_targets=args.linear_probe_genes,
            max_spots=args.pool_spots,
            batch_size=_safe_eval_batch(args.encode_batch, hvg_eval),
            seed=0,
            device=device,
            prefix="linear_probe/masked_hvg",
        ))
        neural_probe_cfg = None
        if args.neural_linear_probe:
            from mm_align.evaluation.neural_linear_probe import (
                NeuralProbeConfig, neural_hvg_regression_probe,
            )
            neural_probe_cfg = NeuralProbeConfig(
                epochs=int(args.neural_probe_epochs),
                patience=int(args.neural_probe_patience),
                lr=float(args.neural_probe_lr),
                weight_decay=float(args.neural_probe_weight_decay),
                batch_size=int(args.neural_probe_batch),
                max_spots=int(args.pool_spots),
                seed=0,
                device=device,
            )
            lp.update(neural_hvg_regression_probe(
                emb, hvg_eval,
                n_targets=args.linear_probe_genes,
                config=neural_probe_cfg,
                prefix="neural_linear_probe/hvg",
            ))
        gho_pg = None
        if int(args.gene_held_out_folds) > 0:
            from mm_align.evaluation.stage1_benchmarks import gene_held_out_probe
            gho_res = gene_held_out_probe(
                enc, hvg_eval,
                n_targets=args.linear_probe_genes,
                gene_folds=int(args.gene_held_out_folds),
                max_spots=args.pool_spots,
                batch_size=_safe_eval_batch(args.encode_batch, hvg_eval),
                seed=0,
                pca_n=_probe_pca,
                alpha=_probe_alpha,
                device=device,
                metric_suite=args.probe_metric_suite,
                prefix="linear_probe/gene_held_out",
                return_per_gene=args.probe_figures,
            )
            if args.probe_figures:
                _gho, gho_pg = gho_res
                lp.update(_gho)
            else:
                lp.update(gho_res)
        if args.probe_figures and (lp_pg or gho_pg):
            viz_root = Path(args.viz_out_dir or (Path(args.out).with_suffix("").as_posix() + "_viz"))
            try:
                _render_probe_figures(
                    out_dir=viz_root / ck.parent.name / "probe_figures",
                    per_gene_linear=lp_pg,
                    per_gene_held_out=gho_pg,
                )
            except Exception as e:
                log.warning(f"probe figures failed for {ck.parent.name}: {e}", exc_info=True)
        # 2b. chunk-view eval: z_chunk is a single sampled partial view; z_spot
        # follows I-JEPA inference and encodes the full clean non-zero sequence.
        try:
            chunk_views = chunk_view_embeddings_from_encoder(
                enc, hvg_eval,
                n_chunks=4,
                chunk_len=min(256, hvg_eval.shape[1]),
                dynamic=True,
                batch_size=max(1, min(_safe_eval_batch(args.encode_batch, hvg_eval), 16)),
                max_spots=args.pool_spots,
                seed=0,
                device=device,
            )
            for rep_name, rep_prefix in (("z_chunk", "chunk_state"), ("z_spot", "spot_state")):
                z = chunk_views[rep_name]
                hvg_z = hvg_eval[:z.shape[0]]
                m_rep = {}
                m_rep.update(embedding_health_metrics(z, prefix=f"{rep_prefix}/intrinsic"))
                m_rep.update(expression_manifold_metrics(
                    z, hvg_z,
                    max_spots=min(args.pool_spots, 2000),
                    k=20,
                    prefix=f"{rep_prefix}/intrinsic/expression",
                ))
                m_rep.update(hvg_linear_probe(
                    z, hvg_z,
                    n_targets=args.linear_probe_genes,
                    max_spots=args.pool_spots,
                    alpha=_probe_alpha,
                    pca_n=_probe_pca,
                    metric_suite=args.probe_metric_suite,
                    prefix=f"{rep_prefix}/linear_probe/hvg",
                ))
                m_rep.update(hvg_rank_probe(
                    z, hvg_z,
                    n_targets=args.linear_probe_genes,
                    max_spots=args.pool_spots,
                    prefix=f"{rep_prefix}/linear_probe/hvg_rank",
                ))
                if args.neural_linear_probe and neural_probe_cfg is not None:
                    from mm_align.evaluation.neural_linear_probe import neural_hvg_regression_probe
                    m_rep.update(neural_hvg_regression_probe(
                        z, hvg_z,
                        n_targets=args.linear_probe_genes,
                        config=neural_probe_cfg,
                        prefix=f"{rep_prefix}/neural_linear_probe/hvg",
                    ))
                lp.update(m_rep)
        except Exception as e:
            log.warning(f"chunk/spot-state eval failed for {ck.parent.name}: {e}", exc_info=True)
        for k, v in lp.items():
            row[k.replace("/", "_")] = v

        # 3. optional organ probe — HEST-only                          [vocab-invariant]
        if args.include_organ_probe and n_hest >= 50:
            organ_h = organ[hest_mask]
            emb_h = emb[hest_mask]
            organ_idx = np.unique(organ_h, return_inverse=True)[1]
            row.update({f"organ_hest_probe_{k}": v
                          for k, v in linear_probe(emb_h, organ_idx).items()})
            row.update({f"organ_hest_{k}": v
                          for k, v in retrieval_recall(emb_h, organ_idx).items()})
        elif args.include_organ_probe:
            for k in ("acc", "f1_macro"):
                row[f"organ_hest_probe_{k}"] = float("nan")
            for k in (1, 5, 10):
                row[f"organ_hest_recall_at_{k}"] = float("nan")

        # 4. source leakage — full mixed pool (lower same/source-probe = better)
        if not args.skip_source_probe:
            src_idx = np.unique(source, return_inverse=True)[1]
            row.update({f"source_probe_{k}": v
                         for k, v in linear_probe(emb, src_idx).items()})
            for k, v in source_knn_leakage_metrics(
                emb, source,
                k=20,
                max_spots=args.pool_spots,
                prefix="leakage/source_knn",
            ).items():
                row[k.replace("/", "_")] = v

        # 5. MVM imputation value-head check                           [downstream reconstruction]
        row.update(masked_value_imputation(
            enc, hvg, vocab_keep, normalizer,
            batch=_safe_eval_batch(args.encode_batch, hvg_eval),
            device=device,
        ))
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"saved {out}")
    if args.make_viz:
        log.info(f"saved qualitative artifacts under {Path(args.viz_out_dir or (Path(args.out).with_suffix('').as_posix() + '_viz'))}")
    print()
    print("─" * 90)
    print(f"Eval pool = splits['{args.split}']  ({n_hest} HEST spots with organ labels, "
          f"{int(emb.shape[0])} total).")
    print("Primary Stage-1 ablation metrics:")
    print("  intrinsic_*                 label-free embedding health            [rank high, top10 not too dominant]")
    print("  intrinsic_expression_*      expression manifold preservation       [higher = better]")
    print("  intrinsic_gene_embedding_*  gene corr vs gene-token cosine         [higher = better]")
    print("  linear_probe_hvg_*          frozen h_tx -> HVG expression probe    [pearson/r2 high, rmse low]")
    print("  linear_probe_masked_hvg_*   masked-input h_tx -> held-out genes    [pearson/r2 high, rmse low]")
    print("  linear_probe_hvg_rank_*     relative expression rank/bin probe      [higher = better]")
    print("  mvm_pearson/spearman/r2     value-head imputation check            [higher = better]")
    print("  source_probe/leakage_*      cross-source leakage                   [LOWER same/probe, HIGHER entropy]")
    print("Optional metrics:")
    print("  organ_hest_probe/retrieval  HEST organ classification/retrieval    [--include-organ-probe]")
    print("─" * 90)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
