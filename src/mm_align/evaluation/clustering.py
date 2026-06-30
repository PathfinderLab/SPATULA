from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering_metrics(emb: np.ndarray, labels: np.ndarray | None = None,
                       k: int = 10) -> dict[str, float]:
    if emb.shape[0] < k + 1:
        return {}
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
    pred = km.labels_
    out = {"cluster/silhouette": float(silhouette_score(emb, pred))}
    if labels is not None:
        out["cluster/ARI"] = float(adjusted_rand_score(labels, pred))
        out["cluster/NMI"] = float(normalized_mutual_info_score(labels, pred))
    return out
