import json
import time

import pandas as pd
from tqdm import tqdm

from experiments_utils import create_summary_dataframe
from ics26_experiments import (
    ICS26_DIR,
    get_N_INIT_CLUSTERS,
    get_N_MIN_PER_CLUSTER,
    get_N_TRAIN,
)
from src import xp
from src.adaptive_clustering import adaptive_clustering
from src.problems.quadratic import quadratic, quadratic_grad, quadratic_hess

# from src.problems.rosenbrock import rosenbrock, rosenbrock_grad, rosenbrock_hess

# --- Configuration ---
TOLS = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3]
DIMS = [2, 3, 4]
ERROR_METRICS_LIST = ["Linf-global", "Linf-local"]  # Added Sweep List

# Constant Parameters
PROB_LIMS = (-1.0, 3.0)

# TEMP_JSON_DIR = ICS26_DIR / "hessian_estimation_rosenbrock"
TEMP_JSON_DIR = ICS26_DIR / "hessian_estimation_quadratic"
TEMP_JSON_DIR.mkdir(exist_ok=True)
create_summary_dataframe(TEMP_JSON_DIR)

OUTPUT_FILE = TEMP_JSON_DIR / "summary.csv"
results_list = []
print(f"Starting sweep over TOLS={TOLS}, DIMS={DIMS}, METRICS={ERROR_METRICS_LIST}...")
print("-" * 60)

for dim in tqdm(DIMS, desc="Dimensions"):
    for tol in tqdm(TOLS, desc="Tolerances", leave=False):
        for metric_name in ERROR_METRICS_LIST:
            if dim == 4 and metric_name == "Linf-local" and tol <= 2e-2:
                # Skip known hard config to save time
                print(
                    f"Skipping DIM={dim}, TOL={tol}, METRIC={metric_name} due to known high compute time."
                )
                continue

            json_name = (
                TEMP_JSON_DIR / f"config_dim{dim}_tol{tol}_metric{metric_name}.json"
            )
            if not json_name.exists():
                # Retrieve dimension-specific params
                N_TRAIN = get_N_TRAIN(dim)
                NUM_INITIAL_CLUSTERS = get_N_INIT_CLUSTERS(dim)
                MIN_PTS_PER_CLUSTER = get_N_MIN_PER_CLUSTER(dim)

                print(
                    f"Running: DIM={dim}, TOL={tol}, METRIC={metric_name} "
                    f"[N={N_TRAIN}, K0={NUM_INITIAL_CLUSTERS}]"
                )

                # 1. Setup Data (Re-seed for consistency across runs)
                xp.random.seed(42)
                X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, dim))

                start_time = time.perf_counter()

                try:
                    # 2. Run Clustering
                    res = adaptive_clustering(
                        X,
                        f_fn=quadratic,
                        g_fn=quadratic_grad,
                        TOL=tol,
                        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                        ERROR_METRIC=metric_name,  # type: ignore
                        SYNTH_BUDGET_FACTOR=100 * 10,
                        LOG_MEM_STATS=False,
                        STORE_HESSIAN_SQRT_PER_CLUSTER=True,
                    )
                    elapsed_time = time.perf_counter() - start_time

                    # 3. Process Results
                    centroids = res["centroids"]
                    n_clusters = centroids.shape[0]

                    if n_clusters == 0:
                        raise ValueError("No valid clusters found.")

                    # --- Eigenvalue Error Calculation ---

                    # A. Estimated Eigenvalues (from H_sqrt)
                    H_sqrt = res["H_sqrt"]
                    # H_approx = H_sqrt @ H_sqrt.T
                    H_est_mag = xp.einsum("kij,kjl->kil", H_sqrt, H_sqrt)
                    H_est_vals, _ = xp.linalg.eigh(H_est_mag)

                    # Sort descending per cluster
                    H_est_lambda = xp.stack(
                        [
                            xp.sort(H_est_vals[k], descending=True)
                            for k in range(n_clusters)
                        ],
                        axis=0,
                    )

                    # B. True Eigenvalues (from analytical Hessian)
                    H_true = quadratic_hess(centroids)
                    H_true_vals, _ = xp.linalg.eigh(H_true)

                    # Take Magnitude (Abs) because our estimator approximates |H|
                    H_true_lambda_unsorted = xp.abs(H_true_vals)

                    # Sort descending per cluster to align with H_est
                    H_true_lambda = xp.stack(
                        [
                            xp.sort(H_true_lambda_unsorted[k], descending=True)
                            for k in range(n_clusters)
                        ],
                        axis=0,
                    )

                    # C. Compute Errors
                    diff = H_est_lambda - H_true_lambda

                    # Ensure we cast to standard python floats for CSV storage
                    l2_rel_error = float(
                        xp.linalg.norm(diff) / xp.linalg.norm(H_true_lambda)
                    )
                    l2_abs_error = float(xp.linalg.norm(diff) / xp.sqrt(n_clusters))

                    # 4. Store Record
                    record = {
                        "dim": dim,
                        "tol": tol,
                        "metric": metric_name,  # Store metric name
                        "n_clusters": n_clusters,
                        "l2_rel_error": l2_rel_error,
                        "l2_abs_error": l2_abs_error,
                        "time_sec": elapsed_time,
                        "status": "success",
                    }
                    results_list.append(record)
                    print(
                        f"  -> Success: {n_clusters} clusters | Rel Err: {l2_rel_error:.4e} | Time: {elapsed_time:.2f}s"
                    )

                except Exception as e:
                    print(f"  -> Failed: {e}")
                    record = {
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

                # --- Save Intermediate JSON ---
                with open(json_name, "w") as f:
                    json.dump(record, f, indent=4)

            else:
                # read existing record
                with open(json_name, "r") as f:
                    record = json.load(f)
                results_list.append(record)


# --- Save to CSV ---
df = pd.DataFrame(results_list)
df.to_csv(OUTPUT_FILE, index=False)
print("-" * 60)
print(f"Sweep Complete. Results saved to {OUTPUT_FILE}")
print(df[["dim", "tol", "metric", "n_clusters", "l2_rel_error", "status"]])
df.to_clipboard(index=False)
print("Results copied to clipboard.")
