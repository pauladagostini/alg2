"""Execução e agregação de experimentos comparando algoritmos de k-centros."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .distance import nearest_center_labels
from .evaluation import evaluate_clustering
from .k_center import calculate_radius

KCenterAlgorithm = Callable[..., List[int]]


@dataclass
class RunResult:
    algorithm: str
    radius: float
    silhouette: float
    ari: Optional[float]
    runtime_seconds: float


@dataclass
class ExperimentSummary:
    algorithm: str
    n_runs: int
    radius_mean: float
    radius_std: float
    silhouette_mean: float
    silhouette_std: float
    ari_mean: Optional[float]
    ari_std: Optional[float]
    runtime_mean: float
    runtime_std: float


def run_algorithm_trials(
    X: np.ndarray,
    k: int,
    algorithm: KCenterAlgorithm,
    algorithm_name: str,
    n_runs: int = 30,
    true_labels: Optional[np.ndarray] = None,
    **algorithm_kwargs,
) -> List[RunResult]:
    """Executa `algorithm` `n_runs` vezes sobre (X, k) e coleta as métricas de cada execução."""
    results = []
    for _ in range(n_runs):
        start = time.perf_counter()
        centers = algorithm(X, k, **algorithm_kwargs)
        runtime = time.perf_counter() - start

        radius = calculate_radius(X, centers)
        labels = nearest_center_labels(X, centers)
        silhouette, ari = evaluate_clustering(X, labels, true_labels)

        results.append(RunResult(algorithm_name, radius, silhouette, ari, runtime))
    return results


def summarize(results: Sequence[RunResult]) -> ExperimentSummary:
    """Agrega uma lista de RunResult em médias e desvios-padrão."""
    if not results:
        raise ValueError("results não pode ser vazio")

    radii = np.array([r.radius for r in results])
    silhouettes = np.array([r.silhouette for r in results])
    runtimes = np.array([r.runtime_seconds for r in results])
    aris = [r.ari for r in results if r.ari is not None]

    return ExperimentSummary(
        algorithm=results[0].algorithm,
        n_runs=len(results),
        radius_mean=float(radii.mean()),
        radius_std=float(radii.std()),
        silhouette_mean=float(silhouettes.mean()),
        silhouette_std=float(silhouettes.std()),
        ari_mean=float(np.mean(aris)) if aris else None,
        ari_std=float(np.std(aris)) if aris else None,
        runtime_mean=float(runtimes.mean()),
        runtime_std=float(runtimes.std()),
    )
