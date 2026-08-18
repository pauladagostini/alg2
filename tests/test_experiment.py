import numpy as np
import pytest

from alg2tp2.experiment import run_algorithm_trials, summarize
from alg2tp2.k_center import hochbaum_shmoys_k_center


def test_run_algorithm_trials_and_summarize():
    X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]])
    true_labels = np.array([0, 0, 1, 1])

    results = run_algorithm_trials(
        X,
        k=2,
        algorithm=hochbaum_shmoys_k_center,
        algorithm_name="hochbaum-shmoys",
        n_runs=3,
        true_labels=true_labels,
    )
    assert len(results) == 3

    summary = summarize(results)
    assert summary.algorithm == "hochbaum-shmoys"
    assert summary.n_runs == 3
    assert summary.ari_mean == 1.0
    assert summary.radius_std == pytest.approx(0.0, abs=1e-9)  # algoritmo determinístico -> mesmo raio em toda execução
