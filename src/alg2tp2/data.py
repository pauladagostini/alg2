"""Geração de dados sintéticos 2D, sempre junto de seus rótulos verdadeiros."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs


def generate_blobs(
    n_samples: int = 1000,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera dados sintéticos em blobs gaussianos e retorna (X, rótulos verdadeiros)."""
    return make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def generate_gaussian_mixture(
    n_samples: int = 1000,
    centers: int = 3,
    std_range: Tuple[float, float] = (0.1, 1.0),
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera uma mistura de gaussianas 2D com desvios padrão variados e retorna (X, rótulos verdadeiros)."""
    rng = np.random.default_rng(random_state)
    means = [rng.random(2) * 10 for _ in range(centers)]
    std_devs = rng.uniform(std_range[0], std_range[1], centers)
    samples_per_cluster = n_samples // centers

    points, labels = [], []
    for label, (mean, std) in enumerate(zip(means, std_devs)):
        points.append(rng.multivariate_normal(mean, np.eye(2) * std, samples_per_cluster))
        labels.append(np.full(samples_per_cluster, label))

    return np.vstack(points), np.concatenate(labels)
