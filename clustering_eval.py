#!/usr/bin/env python3
"""
clustering_eval.py
-----------------

Shared clustering evaluation utilities (Hartigan k-means + ARI/NMI/Purity/ASW).
Used by Python and R pipelines to ensure identical clustering behavior.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["evaluate_clustering"]


def _contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    _, y_true = np.unique(y_true, return_inverse=True)
    _, y_pred = np.unique(y_pred, return_inverse=True)
    m = np.zeros((y_true.max() + 1, y_pred.max() + 1), dtype=np.int64)
    for i in range(y_true.size):
        m[y_true[i], y_pred[i]] += 1
    return m


def _purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = _contingency_matrix(y_true, y_pred)
    return float(np.sum(np.max(m, axis=0)) / np.sum(m)) if m.size else float("nan")


def _adjusted_rand_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = _contingency_matrix(y_true, y_pred)
    n = m.sum()
    if n < 2:
        return float("nan")

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2

    sum_comb = comb2(m).sum()
    a = m.sum(axis=1)
    b = m.sum(axis=0)
    sum_a = comb2(a).sum()
    sum_b = comb2(b).sum()
    expected = sum_a * sum_b / comb2(np.array([n]))[0]
    max_index = 0.5 * (sum_a + sum_b) - expected
    if max_index == 0:
        return 0.0
    return float((sum_comb - expected) / max_index)


def _normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = _contingency_matrix(y_true, y_pred).astype(float)
    n = m.sum()
    if n == 0:
        return float("nan")
    p_ij = m / n
    p_i = p_ij.sum(axis=1, keepdims=True)
    p_j = p_ij.sum(axis=0, keepdims=True)
    denom = p_i @ p_j
    nz = (p_ij > 0) & (denom > 0)
    I = np.sum(p_ij[nz] * np.log(p_ij[nz] / denom[nz]))
    H_i = -np.sum(p_i[p_i > 0] * np.log(p_i[p_i > 0]))
    H_j = -np.sum(p_j[p_j > 0] * np.log(p_j[p_j > 0]))
    if (H_i + H_j) == 0:
        return 1.0
    return float(2 * I / (H_i + H_j))


def _hkmeans_cluster(
    X: np.ndarray,
    k: int,
    n_init: int = 1000,
    max_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    try:
        from hkmeans import HKMeans
    except Exception as exc:
        raise RuntimeError(
            "HKMeans is required. Install via: pip install hartigan-kmeans"
        ) from exc

    model = HKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=int(max(1, n_init)),
        n_jobs=1,
        max_iter=int(max(1, max_iter)),
        verbose=False,
    )
    labels = model.fit_predict(np.asarray(X, dtype=np.float64))
    return np.asarray(labels, dtype=np.int32)


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    n = X.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return float("nan")
    sq = np.sum(X * X, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (X @ X.T)
    dist = np.sqrt(np.maximum(dist2, 0.0))
    sil = np.zeros(n, dtype=np.float32)
    for i in range(n):
        same = labels == labels[i]
        if same.sum() <= 1:
            a = 0.0
        else:
            a = dist[i, same].sum() / (same.sum() - 1)
        b = np.inf
        for cl in uniq:
            if cl == labels[i]:
                continue
            mask = labels == cl
            if mask.sum() == 0:
                continue
            b = min(b, dist[i, mask].mean())
        if not np.isfinite(b):
            sil[i] = 0.0
        else:
            denom = max(a, b)
            sil[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(sil.mean())


def evaluate_clustering(
    imputed_data: np.ndarray,
    true_labels: np.ndarray,
    *,
    n_components: int = 50,
    k: Optional[int] = None,
    n_init: int = 1000,
    max_iter: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    X = np.asarray(imputed_data, dtype=np.float32)
    X = np.nan_to_num(X)
    y = np.asarray(true_labels)
    if X.ndim == 1:
        X = X[:, None]
    emb = X
    if k is None:
        k = max(2, len(np.unique(y)))
    cl = _hkmeans_cluster(emb, int(k), n_init=n_init, max_iter=max_iter, seed=seed)
    return {
        "ASW": round(_silhouette_score(emb, cl), 4),
        "ARI": round(_adjusted_rand_score(y, cl), 4),
        "NMI": round(_normalized_mutual_info(y, cl), 4),
        "PS": round(_purity_score(y, cl), 4),
    }
