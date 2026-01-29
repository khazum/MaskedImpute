#!/usr/bin/env python3
"""
clustering_eval.py
-----------------

Shared clustering evaluation utilities (PCA + k-means + ARI/NMI/Purity/ASW).
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


def _pca_embed(X: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    closest_dist2 = ((X - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = float(closest_dist2.sum())
        if not np.isfinite(total) or total <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            probs = closest_dist2 / total
            idx = int(rng.choice(n, p=probs))
        centers[i] = X[idx]
        dist2 = ((X - centers[i]) ** 2).sum(axis=1)
        closest_dist2 = np.minimum(closest_dist2, dist2)
    return centers


def _kmeans(
    X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_init = max(1, int(n_init))
    best_labels = None
    best_inertia = np.inf
    for _ in range(n_init):
        if n >= k:
            centers = _kmeans_pp_init(X, k, rng)
        else:
            centers = X[rng.choice(n, size=k, replace=True)]
        labels = None
        for _ in range(max_iter):
            dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist2.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    new_centers[j] = X[rng.integers(0, n)]
                else:
                    new_centers[j] = X[mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        if labels is None:
            continue
        inertia = float(np.sum((X - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels


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
    n_init: int = 10,
    max_iter: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    X = np.asarray(imputed_data, dtype=np.float32)
    X = np.nan_to_num(X)
    y = np.asarray(true_labels)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    n_components = max(1, min(int(n_components), n, d))
    emb = _pca_embed(X, n_components)
    if k is None:
        k = max(2, len(np.unique(y)))
    cl = _kmeans(emb, int(k), n_init=n_init, max_iter=max_iter, seed=seed)
    return {
        "ASW": round(_silhouette_score(emb, cl), 4),
        "ARI": round(_adjusted_rand_score(y, cl), 4),
        "NMI": round(_normalized_mutual_info(y, cl), 4),
        "PS": round(_purity_score(y, cl), 4),
    }
