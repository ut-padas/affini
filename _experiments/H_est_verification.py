import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from _experiments._deprecated.adaptive_clustering_non_mpi import adaptive_clustering
from experiments_utils import create_summary_dataframe
from ics26_experiments import get_N_INIT_CLUSTERS, get_N_MIN_PER_CLUSTER, get_N_TRAIN
from src import xp

# Type alias for problem tuple: (f_fn, g_fn, h_fn)
ProblemDef = Tuple[Callable, Callable, Callable]


def run_hessian_convergence_sweep(
    problems: Dict[str, ProblemDef],
    fdim: int,
    output_dir: Path,
    tols: List[float],
    dims: List[int],
    error_metrics: List[str] = ["Linf-global", "Linf-local"],
    prob_lims: Tuple[float, float] = (-1.0, 3.0),
    synth_budget_factor: float = 1000.0,
):
    """
    Runs a Hessian convergence sweep for either scalar (fdim=1) or vector-valued (fdim>1) functions.

    Args:
        problems: Dict mapping problem_name -> (f_fn, g_fn, h_fn).
                  h_fn must return (N, d, d) if fdim=1, or (N, m, d, d) if fdim>1.
        fdim: Output dimension of the function (1 for scalar, >1 for vector).
        output_dir: Path to save JSON configs and summary.csv.
        tols: List of tolerances to sweep.
        dims: List of input dimensions (d) to sweep.
        error_metrics: List of error metric strings to sweep.
        prob_lims: Input domain limits (min, max).
        synth_budget_factor: Multiplier for synthetic point budget.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    create_summary_dataframe(output_dir)
    output_csv = output_dir / "summary.csv"

    results_list = []

    # Flatten sweep parameters for tqdm
    sweep_configs = [
        (p_name, d, t, m)
        for p_name in problems.keys()
        for d in dims
        for t in tols
        for m in error_metrics
    ]

    print(f"Starting Sweep: {len(sweep_configs)} configurations.")
    print(f"Mode: {'Vector (Best-Match)' if fdim > 1 else 'Scalar (Direct)'}")
    print("-" * 60)

    for prob_name, dim, tol, metric_name in tqdm(sweep_configs, desc="Sweep"):

        # Skip known slow configuration (Dim 4 + Linf-local + Tight Tol)
        if dim == 4 and metric_name == "Linf-local" and tol <= 2e-2:
            continue

        json_name = output_dir / f"{prob_name}_dim{dim}_tol{tol}_{metric_name}.json"

        if json_name.exists():
            # Skip re-running if exists
            with open(json_name, "r") as f:
                results_list.append(json.load(f))
            continue

        # Setup Problem
        f_fn, g_fn, h_fn = problems[prob_name]
        N_TRAIN = get_N_TRAIN(dim)
        K0 = get_N_INIT_CLUSTERS(dim)
        MIN_PTS = get_N_MIN_PER_CLUSTER(dim)

        # 1. Generate Data
        xp.random.seed(42)
        X = xp.random.uniform(prob_lims[0], prob_lims[1], size=(N_TRAIN, dim))

        start_time = time.perf_counter()

        try:
            # 2. Run Clustering
            res = adaptive_clustering(
                X,
                f_fn=f_fn,
                g_fn=g_fn,
                TOL=tol,
                NUM_INITIAL_CLUSTERS=K0,
                MIN_PTS_PER_CLUSTER=MIN_PTS,
                ERROR_METRIC=metric_name,  # type: ignore
                SYNTH_BUDGET_FACTOR=synth_budget_factor,
                LOG_MEM_STATS=False,
                STORE_HESSIAN_SQRT_PER_CLUSTER=True,
                INITIAL_CLUSTERING_METHOD="kmeans",
            )
            elapsed = time.perf_counter() - start_time
            centroids = res["centroids"]
            n_clusters = centroids.shape[0]

            if n_clusters == 0:
                raise ValueError("0 Clusters found.")

            # --- Hessian Convergence Logic ---

            # A. Estimated Eigenvalues (Common Step)
            # H_est = H_sqrt @ H_sqrt.T
            H_sqrt = res["H_sqrt"]  # (K, d, d)
            H_est_mat = xp.einsum("kij,kjl->kil", H_sqrt, H_sqrt)
            est_vals, _ = xp.linalg.eigh(H_est_mat)
            # Sort descending: (K, d)
            lambda_est = xp.sort(est_vals, axis=1)[:, ::-1]

            # B. Ground Truth Comparison
            if fdim == 1:
                # --- SCALAR CASE: Direct Match ---
                # H_true shape: (K, d, d)
                H_true = h_fn(centroids)
                true_vals, _ = xp.linalg.eigh(H_true)
                # Take absolute value (estimator approximates |H|)
                lambda_true = xp.sort(xp.abs(true_vals), axis=1)[:, ::-1]

                # Error
                diff = lambda_est - lambda_true
                total_diff_norm = float(xp.linalg.norm(diff))
                total_true_norm = float(xp.linalg.norm(lambda_true))

            else:
                # --- VECTOR CASE: Best Match Active Component ---
                # H_true shape: (K, m, d, d)
                H_true_tensor = h_fn(centroids)
                m_outputs = H_true_tensor.shape[1]

                # 1. Compute Eigenvalues for ALL components: (K, m, d)
                # Flatten to diagonalize in batch
                H_flat = H_true_tensor.reshape(-1, dim, dim)  # (K*m, d, d)
                vals_flat, _ = xp.linalg.eigh(H_flat)
                # Abs and Sort
                lambda_true_all = xp.sort(xp.abs(vals_flat), axis=1)[:, ::-1]
                lambda_true_all = lambda_true_all.reshape(n_clusters, m_outputs, dim)

                # 2. Compute Distance to Estimator for every component
                # lambda_est: (K, d) -> (K, 1, d)
                # lambda_true: (K, m, d)
                # diffs: (K, m, d)
                diffs = lambda_est[:, None, :] - lambda_true_all
                diff_norms = xp.linalg.norm(diffs, axis=2)  # (K, m) L2 error per comp

                # 3. Pick Best Match (Minimum Error)
                best_match_idx = xp.argmin(diff_norms, axis=1)  # (K,)

                # 4. Extract Errors corresponding to best match
                row_idx = xp.arange(n_clusters)
                selected_diff_norms = diff_norms[row_idx, best_match_idx]

                # Extract corresponding true norms for relative error calc
                true_norms_all = xp.linalg.norm(lambda_true_all, axis=2)  # (K, m)
                selected_true_norms = true_norms_all[row_idx, best_match_idx]

                # Aggregate
                total_diff_norm = float(xp.linalg.norm(selected_diff_norms))
                total_true_norm = float(xp.linalg.norm(selected_true_norms))

            # Final Metrics
            l2_rel_error = total_diff_norm / (total_true_norm + 1e-12)
            l2_abs_error = total_diff_norm / xp.sqrt(n_clusters)

            record = {
                "problem": prob_name,
                "dim": dim,
                "tol": tol,
                "metric": metric_name,
                "n_clusters": n_clusters,
                "l2_rel_error": l2_rel_error,
                "l2_abs_error": l2_abs_error,
                "time_sec": elapsed,
                "status": "success",
            }
            results_list.append(record)

        except Exception as e:
            record = {
                "problem": prob_name,
                "dim": dim,
                "tol": tol,
                "metric": metric_name,
                "n_clusters": 0,
                "l2_rel_error": None,
                "l2_abs_error": None,
                "time_sec": 0.0,
                "status": "failed",
                "error_msg": str(e),
            }
            results_list.append(record)

        # Save individual JSON
        with open(json_name, "w") as f:
            json.dump(record, f, indent=4)

    # Save Summary CSV
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    print(f"Sweep Complete. Saved to {output_csv}")
    return df


# --- Example Usage ---
if __name__ == "__main__":
    from ics26_experiments import ICS26_DIR
    from src.problems.quadratic import quadratic, quadratic_grad, quadratic_hess
    from src.problems.vector_quadratic import (
        vector_quadratic,
        vector_quadratic_grad,
        vector_quadratic_hess,
    )

    # 1. Define Vector Problem
    problems_vec = {
        "VectorQuadratic": (
            vector_quadratic,
            vector_quadratic_grad,
            vector_quadratic_hess,
        )
    }

    # 2. Run Vector Sweep
    run_hessian_convergence_sweep(
        problems=problems_vec,
        fdim=2,
        output_dir=ICS26_DIR / "hessian_estimation_vector_quadratic",
        tols=[0.5, 0.1, 0.05],
        dims=[2, 3],
    )

    problems_scalar = {"Quadratic": (quadratic, quadratic_grad, quadratic_hess)}
    run_hessian_convergence_sweep(
        problems=problems_scalar,
        fdim=1,
        output_dir=ICS26_DIR / "hessian_estimation_quadratic",
        tols=[0.5, 0.1],
        dims=[2, 3],
    )
