import csv
import time

from mpi4py import MPI

from _experiments.ics_deprecated.surrogate import HATANNSurrogate
from src import RESULTS_DIR, _get_mpi_rank, _to_device, _to_host, xp
from src.logging_utils import setup_logging
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

setup_logging()

# Constants
TINY_EPS = 1e-12
SEED = 0
DIM = 5
TOL = 1e-2
N_INITIAL_TRAIN = int(1e5)
N_TEST = int(5e4)
NUM_INITIAL_CLUSTERS = 100 * DIM
MIN_PTS_PER_CLUSTER = 100 * DIM
ERROR_METRIC = "Linf-local"
INITIAL_CLUSTERING_METHOD = "uniform_grid"
DOMAIN = [-1.0, 3]
K_NEIGHBOURS = 1

# Active Learning Hyperparameters
MAX_ITERATIONS = 10
CANDIDATE_POOL_SIZE = int(4e5)  # Points to sample when hunting for gaps
MAX_NEW_POINTS_PER_ITER = int(1e5)  # Limit growth to avoid memory explosion

f_fn = rosenbrock
g_fn = rosenbrock_grad


def evaluate_surrogate(surrogate, X_test, ground_truth, iteration):
    """Rank 0: Evaluates the surrogate against a fixed test set and returns metrics."""
    start_predict = time.perf_counter()
    predictions, ungauranteed_mask = surrogate.predict(
        X_test,
        k_nn=K_NEIGHBOURS,
        strict_trust_region=False,
        use_saved_hessian=False,
    )
    end_predict = time.perf_counter()

    n_out = int(xp.sum(ungauranteed_mask))
    n_in = N_TEST - n_out
    good_idx = ~ungauranteed_mask

    print(f"\n--- Iteration {iteration} Evaluation ---")
    print(f"Prediction time: {end_predict - start_predict:.2f} seconds")

    metrics = {
        "n_out": n_out,
        "p_out": 100.0 * n_out / N_TEST,
        "n_in": n_in,
        "n_in_good": 0,
        "p_in_good": 0.0,
        "n_in_bad": 0,
        "p_in_bad": 0.0,
        "l2_rel_err": 0.0,
    }

    if n_in > 0:
        gt_in = ground_truth[good_idx]
        pred_in = predictions[good_idx]

        diff = pred_in - gt_in
        norm_diff = xp.linalg.norm(diff)
        norm_gt = xp.linalg.norm(gt_in)
        in_l2_rel_err = float(norm_diff / (norm_gt + TINY_EPS))

        # Pointwise Relative Error
        element_rel_error = xp.abs(diff) / (xp.abs(gt_in) + TINY_EPS)
        ptwise_rel_error = (
            xp.max(element_rel_error, axis=1)
            if element_rel_error.ndim > 1
            else element_rel_error
        )

        n_in_good = int(xp.sum(ptwise_rel_error <= TOL))
        n_in_bad = int(n_in - n_in_good)

        metrics.update(
            {
                "n_in_good": n_in_good,
                "p_in_good": 100.0 * n_in_good / N_TEST,
                "n_in_bad": n_in_bad,
                "p_in_bad": 100.0 * n_in_bad / N_TEST,
                "l2_rel_err": in_l2_rel_err,
            }
        )

        print(f"Total test points: {N_TEST}")
        print(
            f"Out of surrogate region (p_out): {metrics['n_out']} ({metrics['p_out']:.2f}%)"
        )
        print(
            f"Inside surrogate region: {metrics['n_in']} ({metrics['p_in_good']:.2f}% good, {metrics['p_in_bad']:.2f}% bad)"
        )
        print(
            f"Inside surrogate region - L2 Relative Error: {metrics['l2_rel_err']:.4e}"
        )
    else:
        print("No test points were inside the surrogate region.")

    return metrics


def main():
    comm = MPI.COMM_WORLD
    rank = _get_mpi_rank()

    OUTPUT_DIR = RESULTS_DIR / "p_out_exp"
    csv_path = OUTPUT_DIR / f"active_learning_metrics_dim{DIM}_tol{TOL}.csv"

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Initialize CSV with headers
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Iteration",
                    "N_Train",
                    "Train_Time_s",
                    "N_c",
                    "N_Out",
                    "P_Out_%",
                    "N_In",
                    "N_In_Good",
                    "P_In_Good_%",
                    "N_In_Bad",
                    "P_In_Bad_%",
                    "L2_Rel_Err",
                    "Dim",
                    "Tol",
                ]
            )

    xp.random.seed(SEED + rank)  # slight offset for safety if workers ever gen randoms

    # 1. Initialize Baseline Data
    X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_INITIAL_TRAIN, DIM))

    # Fixed test set to consistently evaluate progress over iterations
    X_test = None
    ground_truth = None
    if rank == 0:
        X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))
        ground_truth = f_fn(X_test)

    surrogate = HATANNSurrogate()

    # 2. The Active Learning Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        if rank == 0:
            current_n_train = X_train.shape[0]
            print(f"\n========== ITERATION {iteration} ==========")
            print(f"Training on {current_n_train} points...")
            start_train = time.perf_counter()

        # All MPI Ranks participate in training
        surrogate.train(
            X_train=X_train,
            f_fn=f_fn,
            g_fn=g_fn,
            TOL=TOL,
            NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
            SEED=SEED,
            ERROR_METRIC=ERROR_METRIC,
            CLUSTERING_MATRIX="Hessian_LSQ",
            IGNORE_GRADIENT=False,
            PROFILE_TIMING=False,
            save_model=False,
            STORE_HESSIAN=False,
        )

        continue_flag = False

        if rank == 0:
            num_clusters = surrogate.get_number_of_sources()
            train_time = time.perf_counter() - start_train
            print(f"Training time: {train_time:.2f} seconds")

            # Evaluate current performance and get metrics dict
            metrics = evaluate_surrogate(surrogate, X_test, ground_truth, iteration)

            # Append iteration data to CSV
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        iteration,
                        current_n_train,
                        f"{train_time:.4f}",
                        num_clusters,
                        metrics["n_out"],
                        f"{metrics['p_out']:.2f}",
                        metrics["n_in"],
                        metrics["n_in_good"],
                        f"{metrics['p_in_good']:.2f}",
                        metrics["n_in_bad"],
                        f"{metrics['p_in_bad']:.2f}",
                        f"{metrics['l2_rel_err']:.4e}",
                        DIM,
                        TOL,
                    ]
                )

            # --- GAP IDENTIFICATION (Monte Carlo) ---
            print("\nScanning domain for unguaranteed regions...")
            X_candidates = xp.random.uniform(
                DOMAIN[0], DOMAIN[1], size=(CANDIDATE_POOL_SIZE, DIM)
            )
            _, cand_mask = surrogate.predict(X_candidates, k_nn=1)

            gap_points = X_candidates[cand_mask]
            num_gaps_found = len(gap_points)
            print(
                f"Found {num_gaps_found} candidate points lying outside current trust regions."
            )

            if num_gaps_found > 0:
                # Subsample if we found too many gaps to prevent memory explosion
                if num_gaps_found > MAX_NEW_POINTS_PER_ITER:
                    indices = xp.random.choice(
                        num_gaps_found, MAX_NEW_POINTS_PER_ITER, replace=False
                    )
                    gap_points = gap_points[indices]

                print(
                    f"Adding {len(gap_points)} new points to the training set for the next iteration."
                )
                X_train = xp.vstack([X_train, gap_points])
                continue_flag = True if iteration < MAX_ITERATIONS else False
            else:
                print("Domain fully covered! No more iterations needed.")
                continue_flag = False

        # 3. Synchronize Ranks
        # Broadcast the decision to continue or stop
        continue_flag = comm.bcast(continue_flag, root=0)

        if not continue_flag:
            break

        # If continuing, broadcast the updated X_train to all workers.
        # (Converting to host memory ensures mpi4py handles it safely)
        if rank == 0:
            X_train_host = _to_host(X_train)
        else:
            X_train_host = None

        X_train_host = comm.bcast(X_train_host, root=0)
        X_train = _to_device(X_train_host)  # Push back to CuPy if using GPU backend

    # Clean Exit
    if rank != 0:
        MPI.Finalize()
        raise SystemExit(0)
    else:
        print(f"\nActive Learning Complete. Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
