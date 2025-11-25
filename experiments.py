import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from tqdm import tqdm

from src.problems.six_hump_camelback import (
    six_hump_camelback,
    six_hump_camelback_grad,
    six_hump_camelback_hess,
)
from src.surrogate import DataSampler, NearODESurrogate
from src.utils import get_matrix_2_norm

logger = logging.getLogger(__name__)

CWD = Path.cwd()
FIGURES = CWD / "figures" / "experiments"
RESULTS = CWD / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def run_1D_experiment(N_train=2000, N_test=4000, ATOL=1e-2, RTOL=None):
    from src.problems.oned import one_d_example, one_d_example_grad, one_d_example_hess

    domain = np.array([0.0, 10.0])
    Lipschitz_est = get_matrix_2_norm(one_d_example_hess, domain)
    logger.info(f"Estimated Global Lipschitz Constant L = {Lipschitz_est:.4f}")

    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=0)

    surrogate = NearODESurrogate(data_dim=1, f_dim=1)
    surrogate.train(
        X_train=X_train,
        f=one_d_example,
        grad=one_d_example_grad,
        ATOL=ATOL,
        RTOL=RTOL,
        NUM_INITIAL_CLUSTERS=10,
        MIN_PTS_PER_CLUSTER=1,
    )

    X_test = DataSampler.sample_grid(domain, num_points_per_dim=N_test)
    Y_pred, outside_mask = surrogate.predict(
        X_test,
        ATOL=ATOL,
        RTOL=RTOL,
        k_NEIGHBOURS=1,
    )
    Y_true = one_d_example(X_test)

    abs_error = np.abs(Y_pred - Y_true)
    rmse = np.sqrt(np.mean(abs_error**2))

    source_centroids = surrogate.get_source_centroids()

    plt.subplots(1, 2, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_test, Y_true, "-k", label="Exact")
    plt.plot(X_test, Y_pred, "--m", label="Surrogate")
    plt.scatter(
        source_centroids,
        one_d_example(source_centroids),
        c="blue",
        marker="x",
        label=f"Sources $(N_{{sources}}={len(source_centroids)})$",
    )
    plt.scatter(
        X_test[outside_mask],
        one_d_example(X_test[outside_mask]),
        marker="o",
        c="orange",
        label=f"Unguaranteed Points $(N={np.sum(outside_mask)})$",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")

    plt.subplot(1, 2, 2)
    plt.semilogy(X_test, abs_error, "-k", label="Absolute Error")
    plt.axhline(ATOL, color="r", linestyle="--", label="Absolute Tolerance")
    plt.scatter(
        X_test[outside_mask],
        abs_error[outside_mask],
        marker="o",
        c="orange",
        label=f"Unguaranteed Points $(N={np.sum(outside_mask)})$",
    )
    above_tol_mask = abs_error > ATOL
    plt.scatter(
        X_test[above_tol_mask],
        abs_error[above_tol_mask],
        c="r",
        marker="x",
        label=f"Points Above ATOL $(N={np.sum(above_tol_mask)})$",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("Absolute Error $|f_{surrogate} - f_{exact}|$")

    img_path = FIGURES / f"1D_experiment_ATOL{ATOL}_RTOL{RTOL}.png"
    plt.savefig(img_path, dpi=400)
    logger.info(f"Saved 1D experiment figure to {img_path}")

    print(f"Max Error: {np.max(abs_error):.4e}")
    print(f"RMSE: {rmse:.4e}")


@dataclass
class BenchmarkProblem:
    name: str
    f: Callable
    grad: Callable
    hess: Optional[Callable]  # Optional, used for estimating L
    domain: np.ndarray  # Shape (D, 2)
    data_dim: int
    f_dim: int
    num_initial_clusters: int
    min_pts_per_cluster: int

    # Evaluation params
    L_global: Optional[float] = None  # Can be pre-set or computed

    # Seed
    seed: int = 0


@dataclass
class ExperimentResult:
    problem_name: str
    n_train: int
    n_test: int
    atol: float
    rtol: Optional[float]
    num_initial_clusters: int
    min_pts_per_cluster: int
    n_clusters: int
    time_train: float
    time_predict: float
    max_error: float
    rms_error: float
    n_guaranteed_good: int
    n_guaranteed_bad: int
    n_unguaranteed_good: int
    n_unguaranteed_bad: int


def estimate_lipschitz_constant(
    problem: BenchmarkProblem, N_samples: int = 1000
) -> float:
    """
    Estimates Global Lipschitz constant L via max Hessian norm.
    For high dimensions, samples random points instead of a grid.
    """
    if problem.L_global is not None:
        return problem.L_global

    if problem.hess is None:
        logger.warning(f"No Hessian provided for {problem.name}, using default L=1.")
        return 1.0

    xp = array_namespace(problem.domain)
    X = DataSampler.sample_latin_hypercube(problem.domain, N_samples)
    vals = problem.hess(X)  # (N, F_DIM, D, D)

    norms = xp.linalg.norm(vals, ord=2, axis=(2, 3))  # (N, F_DIM)
    max_norm = float(xp.max(norms))
    max_point = X[xp.argmax(norms // vals.shape[1])]

    logger.info(
        f"[{problem.name}] Estimated max Hessian norm (Global L): {max_norm} at point {max_point}"
    )
    return max_norm


def run_experiment(
    prob: BenchmarkProblem,
    N_train: int,
    N_test: int,
    ATOL: float,
    RTOL: Optional[float] = None,
):
    """
    Trains surrogate -> Evaluates -> Computes Guarantee Statistics.
    """
    xp = array_namespace(prob.domain)

    # 1. Estimate Lipschitz constant
    Lipschitz_est = estimate_lipschitz_constant(prob)

    # 2. Train
    surrogate = NearODESurrogate(data_dim=prob.data_dim, f_dim=prob.f_dim)
    X_train = DataSampler.sample_latin_hypercube(prob.domain, N=N_train, seed=prob.seed)

    t0 = time.perf_counter()
    surrogate.train(
        X_train=X_train,
        f=prob.f,
        grad=prob.grad,
        NUM_INITIAL_CLUSTERS=prob.num_initial_clusters,
        MIN_PTS_PER_CLUSTER=prob.min_pts_per_cluster,
        ATOL=ATOL,
        RTOL=RTOL,
        SEED=prob.seed,
    )
    train_time = time.perf_counter() - t0

    # 3. Evaluate
    X_test = DataSampler.sample_uniform(prob.domain, N_test, prob.seed)
    t0 = time.perf_counter()
    Y_pred, ungauranteed_mask = surrogate.predict(
        targets=X_test,
        ATOL=ATOL,
        RTOL=RTOL,
        L_global=Lipschitz_est,
        k_NEIGHBOURS=1,
    )
    predict_time = time.perf_counter() - t0

    # 4. Compute Errors
    Y_true = prob.f(X_test)  # (N, F_DIM)
    abs_error = xp.linalg.norm(Y_pred - Y_true, axis=1)
    rel_error = abs_error / xp.linalg.norm(Y_true, axis=1)

    # 5. Categorize points
    guaranteed_mask = xp.logical_not(ungauranteed_mask)
    is_good_mask = xp.logical_or(abs_error <= ATOL, rel_error <= RTOL)
    is_bad_mask = xp.logical_not(is_good_mask)

    n_guaranteed_good = xp.sum(guaranteed_mask & is_good_mask)
    n_guaranteed_bad = xp.sum(guaranteed_mask & is_bad_mask)
    n_unguaranteed_good = xp.sum(ungauranteed_mask & is_good_mask)
    n_unguaranteed_bad = xp.sum(ungauranteed_mask & is_bad_mask)

    max_err = float(xp.max(abs_error))
    rms_err = float(xp.sqrt(xp.mean(abs_error**2)))
    N_clusters = surrogate.N_src

    return ExperimentResult(
        problem_name=prob.name,
        n_train=N_train,
        n_test=N_test,
        atol=ATOL,
        rtol=RTOL,
        n_clusters=N_clusters,
        num_initial_clusters=prob.num_initial_clusters,
        min_pts_per_cluster=prob.min_pts_per_cluster,
        time_train=train_time,
        time_predict=predict_time,
        max_error=max_err,
        rms_error=rms_err,
        n_guaranteed_good=int(n_guaranteed_good),
        n_guaranteed_bad=int(n_guaranteed_bad),
        n_unguaranteed_good=int(n_unguaranteed_good),
        n_unguaranteed_bad=int(n_unguaranteed_bad),
    )


if __name__ == "__main__":
    from src.adaptive_clustering import _HAS_GPU
    from src.logging_config import setup_logging

    xp = cp if _HAS_GPU else np

    setup_logging(console_level=logging.INFO)

    # Run simple 1D experiment
    run_1D_experiment()

    # Define benchmark problems
    problems = [
        BenchmarkProblem(
            name="Six-Hump Camelback",
            f=six_hump_camelback,
            grad=six_hump_camelback_grad,
            hess=six_hump_camelback_hess,
            domain=xp.array([[-3, 3], [-2, 2]]),
            data_dim=2,
            f_dim=1,
            num_initial_clusters=100,
            min_pts_per_cluster=100,
            seed=0,
        )
    ]

    # Define Sweep
    tolerances = [
        (0.1, 0.02),
        (0.05, 0.01),
        (0.01, 0.002),
    ]

    N_TRAIN = 100_000
    N_TEST = 50_000

    # Run Loop
    for prob in tqdm(problems, desc="Benchmark Problems"):
        results = []
        print(f"\n--- Running Benchmarks for {prob.name} ---")
        for atol, rtol in tqdm(tolerances, desc="Tolerances", leave=False):
            print(f"  Running atol={atol}, rtol={rtol}...")
            result = run_experiment(
                prob=prob,
                N_train=N_TRAIN,
                N_test=N_TEST,
                ATOL=atol,
                RTOL=rtol,
            )
            results.append(result)

        # Conver to DataFrame
        df = pd.DataFrame([r.__dict__ for r in results])
        print("\n--- Experiment Results ---")
        print(df.to_string(index=False))

        # Save to CSV
        csv_path = RESULTS / f"benchmark_{prob.name.replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved benchmark results to {csv_path}")

        # Generate LaTeX Table
        latex_str = df.to_latex(
            index=False,
            float_format="%.4e",
            columns=[
                "n_initial_clusters",
                "atol",
                "rtol",
                "n_clusters",
                "n_unguaranteed_bad",
                "n_unguaranteed_good",
                "n_guaranteed_bad",
                "n_guaranteed_good",
                "max_error",
                "rms_error",
                "time_train",
                "time_predict",
            ],
            header=[
                r"$N_{\mathrm{init}}$",
                r"$\mathrm{ATOL}$",
                r"$\mathrm{RTOL}$",
                r"$N_{\mathrm{sources}}$",
                r"$N_{\mathrm{ung. bad}}$",
                r"$N_{\mathrm{ung. good}}$",
                r"$N_{\mathrm{g. bad}}$",
                r"$N_{\mathrm{g. good}}$",
                r"$\max \mathrm{Error}$",
                r"$\mathrm{RMSE}$",
                r"$\mathrm{Train\ Time\ (s)}$",
                r"$\mathrm{Predict\ Time\ (s)}$",
            ],
        )
        latex_path = RESULTS / f"benchmark_{prob.name.replace(' ', '_')}.tex"
        with open(latex_path, "w") as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {latex_path}")

    print("\nAll experiments completed.")
