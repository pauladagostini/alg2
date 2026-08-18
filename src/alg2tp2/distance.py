"""Distância de Minkowski e operações derivadas usadas pelos algoritmos de k-centros."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2) -> float:
    """Distância de Minkowski entre dois pontos (p=2 é a distância Euclidiana)."""
    return float(np.sum(np.abs(x - y) ** p) ** (1 / p))


def pairwise_distances(X: np.ndarray, p: float = 2) -> np.ndarray:
    """Matriz n x n com a distância de Minkowski entre cada par de pontos de X."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sum(np.abs(diff) ** p, axis=-1) ** (1 / p)


def _distances_to_centers(X: np.ndarray, indices: Sequence[int], p: float = 2) -> np.ndarray:
    centers = X[list(indices)]
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(np.abs(diff) ** p, axis=-1) ** (1 / p)


def distances_to_set(X: np.ndarray, indices: Sequence[int], p: float = 2) -> np.ndarray:
    """Para cada ponto de X, a distância até o mais próximo dos pontos em `indices`."""
    if len(indices) == 0:
        return np.full(X.shape[0], np.inf)
    return _distances_to_centers(X, indices, p).min(axis=1)


def nearest_center_labels(X: np.ndarray, indices: Sequence[int], p: float = 2) -> np.ndarray:
    """Para cada ponto de X, a posição (em `indices`) do centro mais próximo."""
    return _distances_to_centers(X, indices, p).argmin(axis=1)
