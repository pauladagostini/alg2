import numpy as np

from alg2tp2.evaluation import evaluate_clustering


def test_evaluate_clustering_without_true_labels():
    X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]])
    labels = np.array([0, 0, 1, 1])
    silhouette, rand_index = evaluate_clustering(X, labels)
    assert silhouette > 0.9
    assert rand_index is None


def test_evaluate_clustering_with_true_labels():
    X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]])
    labels = np.array([0, 0, 1, 1])
    true_labels = np.array([0, 0, 1, 1])
    _, rand_index = evaluate_clustering(X, labels, true_labels)
    assert rand_index == 1.0
