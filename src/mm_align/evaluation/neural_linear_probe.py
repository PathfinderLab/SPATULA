"""Neural linear probes for frozen spot embeddings.

These are report-oriented probes: a trainable nn.Linear head is fitted for a
small number of epochs with early stopping, while the encoder embeddings stay
frozen.  Ridge probes remain the default quick/benchmark path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .gene_imputation_metrics import pearson_per_gene, spearman_per_gene
from .stage1_benchmarks import _select_high_var_genes


@dataclass
class NeuralProbeConfig:
    epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    train_frac: float = 0.70
    val_frac: float = 0.15
    max_spots: int = 8000
    seed: int = 0
    device: str = "auto"


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _split_indices(n: int, *, train_frac: float, val_frac: float, seed: int,
                   stratify: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if stratify is None:
        perm = rng.permutation(n)
        n_train = max(2, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    y = np.asarray(stratify)
    try:
        train_idx, rest_idx = train_test_split(
            idx, train_size=train_frac, random_state=seed, stratify=y
        )
        rel_val = val_frac / max(1e-8, 1.0 - train_frac)
        rest_y = y[rest_idx]
        val_idx, test_idx = train_test_split(
            rest_idx, train_size=rel_val, random_state=seed + 1, stratify=rest_y
        )
        return train_idx, val_idx, test_idx
    except Exception:
        return _split_indices(n, train_frac=train_frac, val_frac=val_frac, seed=seed, stratify=None)


def _standardise_train_val_test(x: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray):
    mu = x[tr].mean(axis=0, keepdims=True)
    sd = x[tr].std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((x[tr] - mu) / sd, (x[va] - mu) / sd, (x[te] - mu) / sd, mu, sd)


def _loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y))
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, generator=g, drop_last=False)


def neural_hvg_regression_probe(emb: np.ndarray, hvg: np.ndarray, *,
                                n_targets: int = 256,
                                config: NeuralProbeConfig | None = None,
                                prefix: str = "neural_linear_probe/hvg") -> dict[str, float]:
    """Frozen embedding -> selected HVG values via trainable nn.Linear.

    This complements Ridge.  Targets are standardised on the probe train split
    for stable optimisation, then predictions are mapped back to the original
    expression scale for metrics.
    """
    cfg = config or NeuralProbeConfig()
    x = np.asarray(emb, dtype=np.float32)
    y_all = np.asarray(hvg, dtype=np.float32)
    n = min(x.shape[0], y_all.shape[0])
    if n < 40 or x.ndim != 2 or y_all.ndim != 2:
        return {f"{prefix}/pearson_mean": float("nan"), f"{prefix}/best_epoch": 0.0}
    x, y_all = x[:n], y_all[:n]
    rng = np.random.default_rng(cfg.seed)
    if n > cfg.max_spots:
        sel = rng.choice(n, int(cfg.max_spots), replace=False)
        x, y_all = x[sel], y_all[sel]
        n = x.shape[0]
    genes = _select_high_var_genes(y_all, n_targets)
    if genes.size == 0:
        return {f"{prefix}/pearson_mean": float("nan"), f"{prefix}/best_epoch": 0.0}
    y = y_all[:, genes]

    tr, va, te = _split_indices(n, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed)
    xtr, xva, xte, _, _ = _standardise_train_val_test(x, tr, va, te)
    y_mu = y[tr].mean(axis=0, keepdims=True)
    y_sd = y[tr].std(axis=0, keepdims=True)
    y_sd[y_sd < 1e-6] = 1.0
    ytr = ((y[tr] - y_mu) / y_sd).astype(np.float32)
    yva = ((y[va] - y_mu) / y_sd).astype(np.float32)

    dev = _device(cfg.device)
    model = nn.Linear(xtr.shape[1], ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()
    train_loader = _loader(xtr, ytr, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    xva_t = torch.from_numpy(xva.astype(np.float32)).to(dev)
    yva_t = torch.from_numpy(yva.astype(np.float32)).to(dev)

    best_loss = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev); yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva_t), yva_t).detach().cpu())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_z = model(torch.from_numpy(xte.astype(np.float32)).to(dev)).detach().cpu().numpy()
    pred = pred_z * y_sd + y_mu
    target = y[te]
    err = pred - target
    mse = (err ** 2).mean(axis=0)
    var = target.var(axis=0)
    r2 = 1.0 - mse / np.maximum(var, 1e-12)
    rmse_norm = np.sqrt(mse) / np.maximum(np.sqrt(var), 1e-12)
    pearson = pearson_per_gene(pred, target)
    spearman = spearman_per_gene(pred, target)
    return {
        f"{prefix}/pearson_mean": float(np.nanmean(pearson)),
        f"{prefix}/spearman_mean": float(np.nanmean(spearman)),
        f"{prefix}/r2_mean": float(np.nanmean(r2)),
        f"{prefix}/rmse_norm": float(np.nanmean(rmse_norm)),
        f"{prefix}/best_epoch": float(best_epoch),
        f"{prefix}/best_val_loss": float(best_loss),
        f"{prefix}/n_targets": float(len(genes)),
        f"{prefix}/n_spots": float(n),
    }


def neural_classification_probe(emb: np.ndarray, labels: np.ndarray, *,
                                config: NeuralProbeConfig | None = None,
                                prefix: str = "neural_linear_probe/class") -> dict[str, float]:
    """Frozen embedding -> class labels via trainable nn.Linear + CE."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    cfg = config or NeuralProbeConfig(max_spots=20000)
    x = np.asarray(emb, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(labels))
    n_classes = len(le.classes_)
    if x.ndim != 2 or x.shape[0] < 40 or n_classes < 2:
        return {f"{prefix}/acc": float("nan"), f"{prefix}/best_epoch": 0.0}
    n = x.shape[0]
    rng = np.random.default_rng(cfg.seed)
    if n > cfg.max_spots:
        sel = rng.choice(n, int(cfg.max_spots), replace=False)
        x, y = x[sel], y[sel]
        n = x.shape[0]
    tr, va, te = _split_indices(n, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed, stratify=y)
    xtr, xva, xte, _, _ = _standardise_train_val_test(x, tr, va, te)
    ytr, yva, yte = y[tr].astype(np.int64), y[va].astype(np.int64), y[te].astype(np.int64)

    dev = _device(cfg.device)
    model = nn.Linear(xtr.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    train_loader = _loader(xtr, ytr, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    xva_t = torch.from_numpy(xva.astype(np.float32)).to(dev)
    yva_t = torch.from_numpy(yva).to(dev)

    best_loss = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev); yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva_t), yva_t).detach().cpu())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xte.astype(np.float32)).to(dev)).argmax(dim=1).detach().cpu().numpy()
    return {
        f"{prefix}/acc": float(accuracy_score(yte, pred)),
        f"{prefix}/f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        f"{prefix}/best_epoch": float(best_epoch),
        f"{prefix}/best_val_loss": float(best_loss),
        f"{prefix}/n_classes": float(n_classes),
        f"{prefix}/n_spots": float(n),
    }
