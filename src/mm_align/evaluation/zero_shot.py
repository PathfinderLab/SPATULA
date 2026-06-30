"""Pure zero-shot evaluation (NO supervised probe, NO label-dependent training).

Tasks:
  - Retrieval (image ↔ tx) — R@K + MRR
  - Clustering (KMeans + organ-label ARI/NMI when labels are available) + silhouette
  - RankMe (Garrido et al. 2023): effective rank of the representation manifold
  - Alignment / Uniformity (Wang & Isola 2020) — only when image- and tx-sides share dim
  - Modality gap: ||mean(z_img) - mean(z_tx)|| on L2-normalized projections
  - Per-organ retrieval (image queries grouped by organ): MAP@K
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from tqdm.auto import tqdm


# -------------------- representation extraction --------------------

@torch.no_grad()
def encode_loader(model, loader, device, max_batches: int | None = None,
                  desc: str = "encode") -> dict:
    """Run the model on a loader and collect all reps + auxiliary fields.

    Returns dict with stacked numpy arrays. Keys:
      h_image, h_tx, z_image, z_tx, image_feat (raw UNI), novae_latent,
      hvg (None if missing), sample_idx (per-spot int label index for biological grouping).
    """
    model.eval()
    keys = {"h_image": [], "h_tx": [], "z_image": [], "z_tx": [],
            "image_feat": [], "novae_latent": [], "hvg": [],
            "sample_idx": [], "spot_idx": []}
    total = max_batches if max_batches is not None else len(loader)
    pbar = tqdm(enumerate(loader), total=total, desc=desc, leave=False, position=2)
    for i, batch in pbar:
        if max_batches is not None and i >= max_batches:
            break
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
        out = model(b)
        keys["h_image"].append(out["h_image"].float().cpu().numpy())
        keys["h_tx"].append(out["h_tx"].float().cpu().numpy())
        keys["z_image"].append(out["z_image"].float().cpu().numpy())
        keys["z_tx"].append(out["z_tx"].float().cpu().numpy())
        keys["image_feat"].append(b["image"].float().cpu().numpy())
        keys["novae_latent"].append(b["tx_latent"].float().cpu().numpy())
        if "hvg" in b:
            keys["hvg"].append(b["hvg"].float().cpu().numpy())
        if "sample_idx" in b:
            si = b["sample_idx"]
            keys["sample_idx"].append(si.cpu().numpy() if torch.is_tensor(si) else np.asarray(si))
        if "spot_idx" in b:
            sp = b["spot_idx"]
            keys["spot_idx"].append(sp.cpu().numpy() if torch.is_tensor(sp) else np.asarray(sp))
    model.train()
    out = {}
    for k, v in keys.items():
        if not v:
            out[k] = None
        else:
            out[k] = np.concatenate(v) if v[0].ndim >= 1 else np.array(v)
    return out


# -------------------- metric primitives --------------------

def _pca_align(X_img: np.ndarray, X_tx: np.ndarray,
               *, target_dim: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Project both arms to a shared dim via independent PCA.

    Used when X_img.shape[1] != X_tx.shape[1] so we can still compute cosine
    retrieval.  Each side gets its own PCA fit (preserves intra-modality
    structure); we then L2-normalise.  This is *not* a learnt cross-modal
    alignment — it's just a fair-comparison projection."""
    from sklearn.decomposition import PCA
    d = min(target_dim, X_img.shape[1], X_tx.shape[1], X_img.shape[0] - 1, X_tx.shape[0] - 1)
    d = max(2, d)
    pi = PCA(n_components=d, random_state=0).fit_transform(X_img.astype(np.float32))
    pt = PCA(n_components=d, random_state=0).fit_transform(X_tx.astype(np.float32))
    return pi, pt


def _retrieval_core(X_img: np.ndarray, X_tx: np.ndarray,
                     ks: tuple[int, ...] = (1, 5, 10, 50),
                     *, pca_align_to: int = 64) -> dict[str, float]:
    """Cosine retrieval between rows of X_img and X_tx (paired diagonals).

    Cross-dim arms (multimodal vs tx-only, baseline_uni vs novae) are first
    PCA-projected to `pca_align_to` dim per modality so retrieval is well-defined.
    """
    if X_img.shape[1] != X_tx.shape[1]:
        X_img, X_tx = _pca_align(X_img, X_tx, target_dim=pca_align_to)
    a = F.normalize(torch.from_numpy(X_img.astype(np.float32)), dim=-1)
    b = F.normalize(torch.from_numpy(X_tx.astype(np.float32)), dim=-1)
    sim = a @ b.t()
    n = sim.size(0); diag = torch.arange(n)
    rank_i2t = (sim.argsort(dim=-1, descending=True) == diag.unsqueeze(-1)).float()
    rank_t2i = (sim.t().argsort(dim=-1, descending=True) == diag.unsqueeze(-1)).float()
    out: dict[str, float] = {}
    for k in ks:
        ka = min(k, n)
        out[f"i2t_R@{k}"] = float(rank_i2t[:, :ka].sum(-1).clamp(max=1).mean())
        out[f"t2i_R@{k}"] = float(rank_t2i[:, :ka].sum(-1).clamp(max=1).mean())
    pos_i2t = (sim.argsort(dim=-1, descending=True) == diag.unsqueeze(-1)).int()
    pos_t2i = (sim.t().argsort(dim=-1, descending=True) == diag.unsqueeze(-1)).int()
    out["i2t_MRR"] = float((1.0 / (pos_i2t.argmax(dim=-1).float() + 1)).mean())
    out["t2i_MRR"] = float((1.0 / (pos_t2i.argmax(dim=-1).float() + 1)).mean())
    out["n"] = float(n)
    return out


def _retrieval_per_group(X_img: np.ndarray, X_tx: np.ndarray, groups: np.ndarray,
                          ks: tuple[int, ...] = (1, 5, 10, 50),
                          *, min_group_size: int = 5,
                          pca_align_to: int = 64) -> dict[str, float]:
    """Mean retrieval metrics computed *within* each group (e.g. one slide
    sample, one organ).  Groups smaller than `min_group_size` are skipped.

    Returns the average R@K / MRR across groups (each group weighted equally).
    """
    keys = [f"i2t_R@{k}" for k in ks] + [f"t2i_R@{k}" for k in ks] + ["i2t_MRR", "t2i_MRR"]
    sums = {k: 0.0 for k in keys}
    n_groups = 0
    n_total_spots = 0
    for g in np.unique(groups):
        mask = (groups == g)
        n = int(mask.sum())
        if n < min_group_size:
            continue
        sub_img = X_img[mask]
        sub_tx = X_tx[mask]
        m = _retrieval_core(sub_img, sub_tx, ks=ks, pca_align_to=pca_align_to)
        for k in keys:
            sums[k] += m[k]
        n_groups += 1
        n_total_spots += n
    if n_groups == 0:
        return {k: float("nan") for k in keys} | {"n_groups": 0.0, "n_total_spots": 0.0}
    out = {k: sums[k] / n_groups for k in keys}
    out["n_groups"] = float(n_groups)
    out["n_total_spots"] = float(n_total_spots)
    return out


def _retrieval(X_img: np.ndarray, X_tx: np.ndarray,
               ks: tuple[int, ...] = (1, 5, 10, 50),
               *,
               sample_idx: np.ndarray | None = None,
               organ_labels: np.ndarray | None = None,
               pca_align_to: int = 64) -> dict[str, float]:
    """Compute retrieval at three granularities:
       global    — over the full pool (every other spot is a candidate)
       per_organ — within-organ (only same-organ spots are candidates)
       per_sample — within-slide (only same-slide spots are candidates)
    """
    out: dict[str, float] = {}
    # global
    g = _retrieval_core(X_img, X_tx, ks=ks, pca_align_to=pca_align_to)
    for k, v in g.items():
        out[f"global/{k}"] = v
    # per organ
    if organ_labels is not None:
        po = _retrieval_per_group(X_img, X_tx, np.asarray(organ_labels),
                                   ks=ks, pca_align_to=pca_align_to)
        for k, v in po.items():
            out[f"per_organ/{k}"] = v
    # per sample (slide)
    if sample_idx is not None:
        ps = _retrieval_per_group(X_img, X_tx, np.asarray(sample_idx),
                                   ks=ks, pca_align_to=pca_align_to)
        for k, v in ps.items():
            out[f"per_sample/{k}"] = v
    return out


def _rankme(Z: np.ndarray) -> float:
    """RankMe (Garrido 2023): effective rank via singular value entropy.

    Lower is bad (collapse). Upper bound = embedding dim.
    """
    if Z.shape[0] < 2:
        return float("nan")
    X = Z - Z.mean(0, keepdims=True)
    # SVD on min(B, D)-rank matrix
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = s + 1e-12
    p = s / s.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def _alignment_uniformity(X_img: np.ndarray, X_tx: np.ndarray,
                          alpha: float = 2.0, t: float = 2.0,
                          max_n: int = 4096) -> dict[str, float]:
    if X_img.shape[1] != X_tx.shape[1]:
        return {"alignment": float("nan"), "uniformity_img": float("nan"),
                "uniformity_tx": float("nan")}
    a = F.normalize(torch.from_numpy(X_img.astype(np.float32)), dim=-1)
    b = F.normalize(torch.from_numpy(X_tx.astype(np.float32)), dim=-1)
    if a.shape[0] > max_n:
        idx = torch.randperm(a.shape[0])[:max_n]
        a, b = a[idx], b[idx]
    align = (a - b).norm(dim=-1).pow(alpha).mean().item()
    # Uniformity: log E[exp(-t * pdist^2)]
    def _uni(x):
        d = torch.pdist(x).pow(2)
        return torch.log(torch.exp(-t * d).mean() + 1e-12).item()
    return {"alignment": float(align), "uniformity_img": float(_uni(a)),
            "uniformity_tx": float(_uni(b))}


def _modality_gap(X_img: np.ndarray, X_tx: np.ndarray) -> float:
    if X_img.shape[1] != X_tx.shape[1]:
        return float("nan")
    a = F.normalize(torch.from_numpy(X_img.astype(np.float32)), dim=-1).mean(0)
    b = F.normalize(torch.from_numpy(X_tx.astype(np.float32)), dim=-1).mean(0)
    return float((a - b).norm().item())


def _cluster_metrics(X: np.ndarray, k: int, labels: np.ndarray | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if X.shape[0] < k + 1:
        return {"silhouette": float("nan"), "ari": float("nan"), "nmi": float("nan")}
    try:
        pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    except Exception:
        return {"silhouette": float("nan"), "ari": float("nan"), "nmi": float("nan")}
    try:
        out["silhouette"] = float(silhouette_score(X, pred))
    except Exception:
        out["silhouette"] = float("nan")
    if labels is not None:
        out["ari"] = float(adjusted_rand_score(labels, pred))
        out["nmi"] = float(normalized_mutual_info_score(labels, pred))
    else:
        out["ari"] = float("nan"); out["nmi"] = float("nan")
    return out


def _per_group_map(X_img: np.ndarray, X_tx: np.ndarray, groups: np.ndarray,
                   topk: int = 10) -> float:
    """For each image query, true positives are TX spots from the same group (e.g. organ).
    Returns mean MAP@K. Useful as a coarse "do spots from the same tissue cluster" check.
    """
    if X_img.shape[1] != X_tx.shape[1] or groups is None:
        return float("nan")
    a = F.normalize(torch.from_numpy(X_img.astype(np.float32)), dim=-1)
    b = F.normalize(torch.from_numpy(X_tx.astype(np.float32)), dim=-1)
    sim = a @ b.t()
    order = sim.argsort(dim=-1, descending=True)[:, :topk].cpu().numpy()
    groups = np.asarray(groups)
    aps = []
    for i in range(order.shape[0]):
        target = groups[i]
        ret = groups[order[i]]
        hits = (ret == target).astype(np.float32)
        if hits.sum() == 0:
            continue
        cum_hits = np.cumsum(hits)
        precision_at_k = cum_hits / np.arange(1, topk + 1)
        aps.append((precision_at_k * hits).sum() / hits.sum())
    return float(np.mean(aps)) if aps else float("nan")


# -------------------- main entrypoint --------------------

def run_zero_shot_eval(model, val_loader, device, cfg: dict, epoch: int,
                       sample_id_for_idx: list[str] | None = None,
                       organ_for_sample: dict[str, str] | None = None) -> dict:
    """Returns a flat metrics dict. NO supervised training inside.

    Args:
      sample_id_for_idx: list mapping val dataset's sample_idx -> sample id string
      organ_for_sample : dict sample_id -> organ label
    """
    # `per_epoch_eval` block was removed from train.yaml when we switched to
    # end-of-training eval only — keep defaults so this function still runs
    # standalone via scripts/eval/zero_shot.py.
    pcfg = cfg["train"].get("per_epoch_eval") or {
        "arms": ["ours_image_only", "ours_multimodal", "ours_tx_only",
                 "baseline_uni", "baseline_novae"],
        "n_clusters": 10,
        "max_eval_batches": None,
        "max_eval_spots": 10000,
    }
    arms: list[str] = list(pcfg["arms"])
    n_clusters: int = pcfg["n_clusters"]
    max_b = pcfg.get("max_eval_batches")

    pool = encode_loader(model, val_loader, device, max_batches=max_b,
                         desc=f"zs[ep{epoch}] val→latents")
    if pool["h_image"] is None:
        return {}

    # Subsample the pool so per-epoch eval stays cheap on large val sets.
    cap = pcfg.get("max_eval_spots")
    n = pool["h_image"].shape[0]
    if cap and n > cap:
        rng = np.random.default_rng(0)  # deterministic across epochs → comparable
        idx = rng.choice(n, cap, replace=False)
        for k, v in list(pool.items()):
            if v is None:
                continue
            try:
                pool[k] = v[idx]
            except Exception:
                pass

    # Build organ labels per spot if metadata available.
    spot_organs = None
    if pool["sample_idx"] is not None and sample_id_for_idx and organ_for_sample:
        try:
            spot_organs = np.array([
                organ_for_sample.get(sample_id_for_idx[int(si)], None)
                for si in pool["sample_idx"]
            ])
            # Convert None -> "Unknown" string
            spot_organs = np.array(["Unknown" if v is None else v for v in spot_organs])
        except Exception:
            spot_organs = None

    def _img_of(arm: str) -> np.ndarray:
        if arm == "ours_image_only":   return pool["h_image"]
        if arm == "ours_multimodal":   return np.concatenate([pool["h_image"], pool["h_tx"]], axis=-1)
        if arm == "ours_tx_only":      return pool["h_tx"]
        if arm == "baseline_uni":      return pool["image_feat"]
        if arm == "baseline_novae":    return pool["novae_latent"]
        raise ValueError(arm)

    def _tx_of(arm: str) -> np.ndarray:
        if arm == "ours_image_only":   return pool["h_tx"]
        if arm == "ours_multimodal":   return pool["h_tx"]
        if arm == "ours_tx_only":      return pool["h_image"]
        if arm == "baseline_uni":      return pool["novae_latent"]
        if arm == "baseline_novae":    return pool["image_feat"]
        raise ValueError(arm)

    metrics: dict[str, float] = {"epoch": float(epoch)}

    sample_idx_arr = pool.get("sample_idx")
    for arm in tqdm(arms, desc=f"zs[ep{epoch}] arms", leave=False, position=2):
        Xi = _img_of(arm); Xt = _tx_of(arm)

        # Retrieval at 3 granularities — global / per_organ / per_sample.
        # Cross-dim arms (multimodal vs tx, baseline_uni vs novae) are PCA-aligned
        # so they're no longer NaN.
        retr = _retrieval(Xi, Xt,
                           sample_idx=sample_idx_arr,
                           organ_labels=spot_organs)
        for k, v in retr.items():
            metrics[f"{arm}/retr/{k}"] = v

        # Legacy MAP@10 (image queries × all spots, hit if same organ).
        if spot_organs is not None:
            metrics[f"{arm}/retr/organ_MAP@10"] = _per_group_map(Xi, Xt, spot_organs, topk=10)

        # Alignment / uniformity (works only when dims match)
        au = _alignment_uniformity(Xi, Xt)
        for k, v in au.items():
            metrics[f"{arm}/au/{k}"] = v

        # Modality gap
        metrics[f"{arm}/modality_gap"] = _modality_gap(Xi, Xt)

        # RankMe on image-side rep
        metrics[f"{arm}/rankme/image"] = _rankme(Xi)
        metrics[f"{arm}/rankme/tx"] = _rankme(Xt)

        # Clustering (silhouette + ARI/NMI vs organ if available)
        cluster_labels = spot_organs if spot_organs is not None else None
        cm = _cluster_metrics(Xi, n_clusters, cluster_labels)
        for k, v in cm.items():
            metrics[f"{arm}/cluster/{k}"] = v

    return metrics


# -------------------- figure renderer --------------------

def render_zero_shot_curves(history: list[dict], out_path: Path, title: str = "") -> None:
    """Renders a 3x3 panel: retrieval, organ-MAP, RankMe, modality gap,
    alignment, uniformity, silhouette, ARI, NMI — all over epochs."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    arms = sorted({k.split("/")[0] for h in history for k in h if "/" in k})
    color_for = {a: c for a, c in zip(arms, plt.cm.tab10.colors[: len(arms)])}

    def _series(arm, suffix):
        return [h.get(f"{arm}/{suffix}", np.nan) for h in history]

    panels = [
        ("Retrieval i2t R@10", "retr/i2t_R@10", "↑"),
        ("Retrieval mean MRR", None, "↑"),                 # custom: avg of i2t/t2i MRR
        ("Per-organ retrieval MAP@10", "retr/organ_MAP@10", "↑"),
        ("RankMe (image side)", "rankme/image", "↑"),
        ("Modality gap (‖μ_img-μ_tx‖)", "modality_gap", "↓"),
        ("Alignment (‖z_img-z_tx‖²)", "au/alignment", "↓"),
        ("Cluster silhouette", "cluster/silhouette", "↑"),
        ("Cluster ARI vs organ", "cluster/ari", "↑"),
        ("Cluster NMI vs organ", "cluster/nmi", "↑"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for ax, (label, key, arrow) in zip(axes.flatten(), panels):
        for arm in arms:
            if key is None:
                a = np.array(_series(arm, "retr/i2t_MRR"))
                b = np.array(_series(arm, "retr/t2i_MRR"))
                ys = 0.5 * (a + b)
            else:
                ys = _series(arm, key)
            ax.plot(epochs, ys, marker="o", color=color_for[arm], label=arm)
        ax.set_title(f"{label}  ({arrow})"); ax.set_xlabel("epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
