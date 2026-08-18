import numpy as np
import pytest

from alg2tp2.k_center import calculate_radius, gonzalez_k_center, hochbaum_shmoys_k_center

TIGHT_CLUSTERS = np.array(
    [
        [0, 0],
        [0.1, 0],
        [0, 0.1],
        [10, 10],
        [10.1, 10],
        [10, 10.1],
    ]
)


def test_gonzalez_k_center_returns_k_distinct_indices():
    centers = gonzalez_k_center(TIGHT_CLUSTERS, k=2, seed=0)
    assert len(centers) == 2
    assert len(set(centers)) == 2


def test_gonzalez_k_center_separates_clusters():
    centers = gonzalez_k_center(TIGHT_CLUSTERS, k=2, seed=0)
    radius = calculate_radius(TIGHT_CLUSTERS, centers)
    assert radius < 1.0


def test_hochbaum_shmoys_is_deterministic():
    assert hochbaum_shmoys_k_center(TIGHT_CLUSTERS, k=2) == hochbaum_shmoys_k_center(TIGHT_CLUSTERS, k=2)


def test_hochbaum_shmoys_separates_clusters():
    centers = hochbaum_shmoys_k_center(TIGHT_CLUSTERS, k=2)
    radius = calculate_radius(TIGHT_CLUSTERS, centers)
    assert radius < 1.0


def test_k_greater_or_equal_to_n_returns_all_points():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    assert sorted(gonzalez_k_center(X, k=3, seed=0)) == [0, 1, 2]
    assert sorted(hochbaum_shmoys_k_center(X, k=5)) == [0, 1, 2]


@pytest.mark.parametrize("algorithm", [gonzalez_k_center, hochbaum_shmoys_k_center])
def test_k_center_rejects_non_positive_k(algorithm):
    X = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        algorithm(X, k=0)
