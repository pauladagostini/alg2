import numpy as np

from alg2tp2.distance import (
    distances_to_set,
    minkowski_distance,
    nearest_center_labels,
    pairwise_distances,
)


def test_minkowski_distance_euclidean():
    assert minkowski_distance(np.array([0, 0]), np.array([3, 4])) == 5.0


def test_minkowski_distance_manhattan():
    assert minkowski_distance(np.array([0, 0]), np.array([3, 4]), p=1) == 7.0


def test_pairwise_distances_symmetric_and_zero_diagonal():
    X = np.array([[0, 0], [3, 4], [6, 8]])
    dist = pairwise_distances(X)
    assert np.allclose(np.diag(dist), 0)
    assert np.allclose(dist, dist.T)
    assert dist[0, 1] == 5.0


def test_distances_to_set_uses_nearest_center():
    X = np.array([[0, 0], [10, 0], [5, 0]])
    dist = distances_to_set(X, [0, 1])
    assert np.allclose(dist, [0, 0, 5])


def test_distances_to_set_empty_indices_returns_infinity():
    X = np.array([[0, 0], [1, 1]])
    dist = distances_to_set(X, [])
    assert np.all(np.isinf(dist))


def test_nearest_center_labels():
    X = np.array([[0, 0], [10, 0], [4, 0]])
    labels = nearest_center_labels(X, [0, 1])
    assert list(labels) == [0, 1, 0]
