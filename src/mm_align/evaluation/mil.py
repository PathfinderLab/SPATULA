"""Slide-level MIL evaluation utilities for frozen Stage-2 embeddings.

The functions here intentionally stay lightweight: they consume per-spot
embeddings and sample indices produced by `encode_loader`, build slide bags,
and train only a small downstream head. This matches the Loki/HEST/PathBench
style of asking whether image-side representations support slide-level tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class SlideBags:
    bags: list[np.ndarray]
    labels: np.ndarray
    sample_ids: list[str]
    label_names: list[str]


def make_slide_bags(spot_embeddings: np.ndarray,
                    sample_idx: np.ndarray,
                    sample_ids_by_index: Sequence[str],
                    labels_by_sample: dict[str, str | int | float],
                    min_spots: int = 1,
                    label_names: Sequence[str] | None = None) -> SlideBags:
    """Group spot embeddings into per-slide bags with labels.

    Pass `label_names` from the train split when constructing val/test bags so
    class indices stay stable even if the eval split lacks one class.
    """
    z = np.asarray(spot_embeddings, dtype=np.float32)
    sidx = np.asarray(sample_idx).astype(int)
    bags: list[np.ndarray] = []
    labels_raw: list[str] = []
    kept_ids: list[str] = []
    for idx in np.unique(sidx):
        if idx < 0 or idx >= len(sample_ids_by_index):
            continue
        sid = str(sample_ids_by_index[idx])
        if sid not in labels_by_sample:
            continue
        bag = z[sidx == idx]
        if bag.shape[0] < min_spots:
            continue
        label = labels_by_sample[sid]
        if label is None or str(label).strip() == "" or str(label).lower() == "nan":
            continue
        bags.append(bag)
        labels_raw.append(str(label))
        kept_ids.append(sid)
    if label_names is None:
        label_names = sorted(set(labels_raw))
    else:
        label_names = [str(v) for v in label_names]
    label_to_idx = {v: i for i, v in enumerate(label_names)}
    kept = [(b, v, sid) for b, v, sid in zip(bags, labels_raw, kept_ids) if v in label_to_idx]
    if not kept:
        return SlideBags(bags=[], labels=np.zeros((0,), dtype=np.int64),
                         sample_ids=[], label_names=list(label_names))
    bags = [b for b, _, _ in kept]
    labels = np.array([label_to_idx[v] for _, v, _ in kept], dtype=np.int64)
    kept_ids = [sid for _, _, sid in kept]
    return SlideBags(bags=bags, labels=labels, sample_ids=kept_ids, label_names=list(label_names))


def mean_pool_bags(bags: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pool variable-length bags into one vector per slide."""
    if not bags:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(b, dtype=np.float32).mean(axis=0) for b in bags], axis=0)


def max_pool_bags(bags: Sequence[np.ndarray]) -> np.ndarray:
    """Max-pool variable-length bags into one vector per slide."""
    if not bags:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(b, dtype=np.float32).max(axis=0) for b in bags], axis=0)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray | None = None) -> dict[str, float]:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score, roc_auc_score)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_eval": float(len(y_true)),
        "n_classes": float(len(np.unique(y_true))),
    }
    if y_prob is not None and len(np.unique(y_true)) >= 2:
        try:
            if y_prob.shape[1] == 2:
                out["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                out["auprc"] = float(average_precision_score(y_true, y_prob[:, 1]))
            else:
                out["auroc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            pass
    return out


def run_pooled_slide_probe(train_bags: SlideBags,
                           eval_bags: SlideBags,
                           pooling: str = "mean",
                           C: float = 1.0,
                           max_iter: int = 2000) -> dict[str, float]:
    """Logistic-regression slide classifier on pooled frozen bags."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if pooling == "mean":
        x_tr = mean_pool_bags(train_bags.bags)
        x_ev = mean_pool_bags(eval_bags.bags)
    elif pooling == "max":
        x_tr = max_pool_bags(train_bags.bags)
        x_ev = max_pool_bags(eval_bags.bags)
    else:
        raise ValueError(f"unknown pooling={pooling!r}")
    y_tr = train_bags.labels
    y_ev = eval_bags.labels
    if x_tr.shape[0] < 4 or x_ev.shape[0] < 1 or len(np.unique(y_tr)) < 2:
        return {"skipped": 1.0, "n_train": float(x_tr.shape[0]), "n_eval": float(x_ev.shape[0])}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"),
    )
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_ev)
    prob = clf.predict_proba(x_ev) if hasattr(clf, "predict_proba") else None
    out = _classification_metrics(y_ev, pred, prob)
    out.update({"skipped": 0.0, "n_train": float(x_tr.shape[0])})
    return out


def run_attention_mil(train_bags: SlideBags,
                      eval_bags: SlideBags,
                      epochs: int = 50,
                      lr: float = 1e-3,
                      hidden_dim: int = 128,
                      weight_decay: float = 1e-4,
                      seed: int = 0,
                      device: str | None = None) -> dict[str, float]:
    """Small attention-MIL head trained on frozen slide bags.

    This is intentionally modest: it is a downstream evaluator, not a new
    representation learner. Use pooled probes as the stable baseline and this
    attention head as the stronger MIL check.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if len(train_bags.bags) < 4 or len(eval_bags.bags) < 1 or len(np.unique(train_bags.labels)) < 2:
        return {"skipped": 1.0, "n_train": float(len(train_bags.bags)), "n_eval": float(len(eval_bags.bags))}
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = int(train_bags.bags[0].shape[1])
    n_classes = int(max(train_bags.labels.max(), eval_bags.labels.max()) + 1)

    class AttentionMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.cls = nn.Linear(in_dim, n_classes)

        def forward(self, bag):
            score = self.attn(bag).squeeze(-1)
            weight = torch.softmax(score, dim=0).unsqueeze(-1)
            pooled = (weight * bag).sum(dim=0)
            return self.cls(pooled)

    model = AttentionMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = np.bincount(train_bags.labels, minlength=n_classes).astype(np.float32)
    weights = weights.sum() / np.maximum(weights, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    train_order = np.arange(len(train_bags.bags))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(int(epochs)):
        rng.shuffle(train_order)
        for i in train_order:
            bag = torch.from_numpy(train_bags.bags[i]).float().to(device)
            y = torch.tensor([int(train_bags.labels[i])], dtype=torch.long, device=device)
            logit = model(bag).unsqueeze(0)
            loss = F.cross_entropy(logit, y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for bag_np in eval_bags.bags:
            bag = torch.from_numpy(bag_np).float().to(device)
            prob = torch.softmax(model(bag), dim=-1).cpu().numpy()
            probs.append(prob)
            preds.append(int(prob.argmax()))
    prob_arr = np.stack(probs, axis=0)
    out = _classification_metrics(eval_bags.labels, np.array(preds), prob_arr)
    out.update({"skipped": 0.0, "n_train": float(len(train_bags.bags))})
    return out
