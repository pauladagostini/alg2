"""Algoritmos 2-aproximados para o problema de k-centros.

Ambos os algoritmos recebem uma matriz de pontos X e um k, e retornam os
índices (em X) dos pontos escolhidos como centros.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .distance import distances_to_set, pairwise_distances


def calculate_radius(X: np.ndarray, centers: Sequence[int], p: float = 2) -> float:
    """Raio da solução: maior distância entre um ponto e seu centro mais próximo."""
    return float(distances_to_set(X, centers, p=p).max())


def gonzalez_k_center(
    X: np.ndarray,
    k: int,
    p: float = 2,
    seed: Optional[int] = None,
) -> list[int]:
    """
    Algoritmo greedy 2-aproximado de Gonzalez (1985), também conhecido como
    farthest-first traversal: parte de um centro aleatório e, a cada passo,
    adiciona o ponto mais distante do conjunto de centros já escolhidos.
    """
    if k <= 0:
        raise ValueError("k precisa ser positivo")
    if k >= X.shape[0]:
        return list(range(X.shape[0]))

    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(X.shape[0]))]

    while len(centers) < k:
        dists = distances_to_set(X, centers, p=p)
        centers.append(int(np.argmax(dists)))

    return centers


def hochbaum_shmoys_k_center(X: np.ndarray, k: int, p: float = 2) -> list[int]:
    """
    Algoritmo 2-aproximado de Hochbaum & Shmoys (1985): busca binária sobre
    as distâncias par a par candidatas ao raio ótimo, verificando a cada
    passo se é possível cobrir todos os pontos com até k centros usando um
    conjunto dominante guloso no grafo de threshold 2*raio.
    """
    if k <= 0:
        raise ValueError("k precisa ser positivo")
    n = X.shape[0]
    if k >= n:
        return list(range(n))

    dist = pairwise_distances(X, p=p)
    candidate_radii = np.unique(dist)

    lo, hi = 0, len(candidate_radii) - 1
    best_centers: Optional[list[int]] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        centers = _greedy_dominating_set(dist, candidate_radii[mid], k)
        if centers is not None:
            best_centers = centers
            hi = mid - 1
        else:
            lo = mid + 1

    assert best_centers is not None  # k < n garante viabilidade no maior raio candidato
    return best_centers


def _greedy_dominating_set(dist: np.ndarray, radius: float, k: int) -> Optional[list[int]]:
    """
    Tenta cobrir todos os pontos com no máximo k centros, onde cada centro
    cobre os pontos a até 2*radius de distância. Retorna None se não houver
    cobertura viável com até k centros nesse raio.
    """
    n = dist.shape[0]
    uncovered = np.ones(n, dtype=bool)
    centers: list[int] = []

    while uncovered.any():
        if len(centers) >= k:
            return None
        center = int(np.argmax(uncovered))  # primeiro ponto ainda não coberto
        centers.append(center)
        uncovered &= dist[center] > 2 * radius

    return centers
