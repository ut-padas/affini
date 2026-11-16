"""
One run of the manuscript's iterative-refinement experiment (Figure 9, uniform resampling) through
``p_out_iterative_experiment.run_iterative_refinement``, with its CSV under ``--results``.

The experiment script fixes its own sizes: 20,000 * 1.3^d initial points, 20 d initial clusters,
50 d points per cluster, 40% new points per iteration, 20,000 test points. Its "gaussian" is the
isotropic Gaussian and its wave domain is [-4, 4] (the manuscript describes the anisotropic
Gaussian and [-5, 5]); see CONVERGENCE_STUDY.md.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--function", required=True, choices=("gaussian", "quadratic", "rosenbrock", "wave"))
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=10)
    a = parser.parse_args()

    import p_out_iterative_experiment as experiment

    a.results.mkdir(parents=True, exist_ok=True)
    experiment.RESULTS_DIR = a.results  # read when the run starts
    csv = a.results / "active_learning_halo" / f"halo_metrics_dim{a.dimension}_f_{a.function}_tol{a.tol}_sampling_uniform.csv"
    if csv.exists() and sum(1 for _ in csv.open()) == a.n_iter + 2:  # header + iterations 0..n_iter
        print(f"{csv} is complete; skipping")
        return
    experiment.run_iterative_refinement(function=a.function, sampling="uniform", d=a.dimension, tol=a.tol, n_iter=a.n_iter)


if __name__ == "__main__":
    main()
