"""Métricas de qualidade de clustering."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[float]]:
    """Calcula a silhueta e, se `true_labels` for informado, o índice de Rand ajustado."""
    silhouette = silhouette_score(X, labels)
    rand_index = adjusted_rand_score(true_labels, labels) if true_labels is not None else None
    return silhouette, rand_index
