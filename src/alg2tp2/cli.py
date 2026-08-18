"""Interface de linha de comando para comparar os algoritmos de k-centros."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .data import generate_blobs
from .experiment import ExperimentSummary, run_algorithm_trials, summarize
from .k_center import gonzalez_k_center, hochbaum_shmoys_k_center

ALGORITHMS = {
    "gonzalez": gonzalez_k_center,
    "hochbaum-shmoys": hochbaum_shmoys_k_center,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara algoritmos 2-aproximados para o problema de k-centros."
    )
    parser.add_argument("--n-samples", type=int, default=700, help="tamanho do dataset sintético")
    parser.add_argument("--centers", type=int, default=5, help="número de clusters gerados nos dados sintéticos")
    parser.add_argument("--k", type=int, default=5, help="número de centros buscado pelos algoritmos")
    parser.add_argument("--runs", type=int, default=30, help="número de execuções por algoritmo")
    parser.add_argument("--random-state", type=int, default=42, help="semente para a geração dos dados")
    parser.add_argument("--algorithm", choices=[*ALGORITHMS, "all"], default="all")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    X, true_labels = generate_blobs(
        n_samples=args.n_samples, centers=args.centers, random_state=args.random_state
    )

    algorithms = ALGORITHMS if args.algorithm == "all" else {args.algorithm: ALGORITHMS[args.algorithm]}

    for name, algorithm in algorithms.items():
        results = run_algorithm_trials(
            X, args.k, algorithm, name, n_runs=args.runs, true_labels=true_labels
        )
        _print_summary(summarize(results))


def _print_summary(summary: ExperimentSummary) -> None:
    print(f"--- {summary.algorithm} ({summary.n_runs} execuções) ---")
    print(f"raio        : média={summary.radius_mean:.4f}  desvio={summary.radius_std:.4f}")
    print(f"silhueta    : média={summary.silhouette_mean:.4f}  desvio={summary.silhouette_std:.4f}")
    if summary.ari_mean is not None:
        print(f"ARI         : média={summary.ari_mean:.4f}  desvio={summary.ari_std:.4f}")
    print(f"tempo (s)   : média={summary.runtime_mean:.4f}  desvio={summary.runtime_std:.4f}")
    print()


if __name__ == "__main__":
    main()
