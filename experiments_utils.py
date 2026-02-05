import json
import time
from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

from src import IS_CUPY_BACKEND, RESULTS_DIR, Array, xp
from src.logging_utils import capture_evaluations
from src.surrogate import HATANNSurrogate

ICS26_DIR = RESULTS_DIR / "ics26"
ICS26_DIR.mkdir(parents=True, exist_ok=True)
TINY_EPS = 1e-12
SEED = 0


def create_summary_dataframe(results_dir: Path):
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return

    json_files = list(results_dir.glob("*.json"))
    if len(json_files) == 0:
        print(f"No json result files found in {results_dir}!")
        return
    data_dicts = [json.load(open(f, "r")) for f in json_files]
    df = pd.DataFrame(data_dicts)
    df = df.sort_values(by=["dim", "tol"], ascending=[True, False])

    # Extract profiling stats to a separate CSV if available
    if "profiling_stats" in df.columns:
        profile_rows = []
        for _, row in df.iterrows():
            stats = row["profiling_stats"]

            # Base entry with identifiers
            entry = {
                "problem_tag": row.get("problem_tag"),
                "dim": row.get("dim"),
                "tol": row.get("tol"),
                "num_levels": row.get("num_levels"),
                "N0": row.get("N0"),
                "Nc": row.get("num_final_clusters"),
            }

            if isinstance(stats, dict):
                # Add scalar timing keys
                for k, v in stats.items():
                    if k != "level_stats":
                        entry[k] = v

                # Add level stats as comma-separated strings
                level_info = stats.get("level_stats", [])
                if isinstance(level_info, list) and level_info:
                    # Gather all keys present across levels
                    keys = set().union(*(d.keys() for d in level_info))
                    for k in sorted(keys):
                        vals = [str(lvl.get(k, "")) for lvl in level_info]
                        entry[k] = ",".join(vals)

            profile_rows.append(entry)

        if profile_rows:
            prof_df = pd.DataFrame(profile_rows)
            # Reorder columns to put identifiers first
            cols = list(prof_df.columns)
            prio_cols = ["problem_tag", "dim", "tol", "num_levels", "time_total"]
            sorted_cols = [c for c in prio_cols if c in cols] + [
                c for c in cols if c not in prio_cols
            ]
            prof_df = prof_df[sorted_cols]

            prof_path = results_dir / "training_profiles.csv"
            prof_df.to_csv(prof_path, index=False)
            print(f"Training profiles dataframe saved to {prof_path}")

        # Drop profiling_stats for the main summary (it's a dict and breaks CSV structure)
        df_summary = df.drop(columns=["profiling_stats"])
    else:
        df_summary = df

    df_summary.to_csv(results_dir / "summary.csv", index=False)
    print(f"Summary dataframe saved to {results_dir / 'summary.csv'}")

    try:
        # Count number of error cases
        n_errors = df["error_tag"].apply(lambda x: x != "").sum()
        if n_errors == 0:
            # print in green color
            print(f"\033[92mNo error cases in {results_dir}!\033[0m")
        else:
            # print in red color
            print(f"\033[91mError cases found: {n_errors} in {results_dir}!\033[0m")
    except Exception:
        pass


def plot_error_histogram(
    ground_truth: Array, predictions: Array, json_fpath: Path, TOL: float
):
    img_fpath = json_fpath.with_suffix(".png")
    img_title = json_fpath.stem

    # 1. Compute Element-wise Relative Error
    # Shape: (N, m) for vectors, (N,) for scalars
    element_rel_error = xp.abs(predictions - ground_truth) / (
        xp.abs(ground_truth) + TINY_EPS
    )

    # 2. Reduce to Point-wise Max (Worst Component)
    # If vector (ndim > 1), take max over components (axis=1)
    # This aligns with the "Pointwise Relative Error" definition where a point
    # is only good if ALL its components are good.
    if element_rel_error.ndim > 1:
        ptwise_rel_error = xp.max(element_rel_error, axis=1)
    else:
        ptwise_rel_error = element_rel_error

    # 3. Compute Stats
    n_within_tol = xp.sum(ptwise_rel_error <= TOL)
    perct_within_tol = 100.0 * n_within_tol / ptwise_rel_error.shape[0]

    # 4. Plotting
    plt.figure(figsize=(10, 6))

    # Extract data to CPU for plotting
    data_to_plot = ptwise_rel_error
    if IS_CUPY_BACKEND:
        data_to_plot = data_to_plot.get()
    plt.hist(data_to_plot.flatten(), bins=50, log=True)

    plt.axvline(
        x=TOL,
        color="r",
        linestyle="--",
        label=f"TOL={TOL} (Good Points={perct_within_tol:.2f}%)",
    )

    plt.ylabel("Frequency (log scale)")
    plt.xlabel(r"Pointwise Max Relative Error (log scale)")
    plt.title(img_title)
    plt.legend()
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(img_fpath, dpi=300)
    plt.close()


def del_all_error_files_in_dir(results_dir: Path):
    error_files = list(results_dir.glob("*.error.txt"))
    for ef in error_files:
        print(f"Deleting error file: {ef}")
        ef.unlink()

    # Get all json files
    json_files = list(results_dir.glob("*.json"))
    for jf in json_files:
        # Check if the error_tag is non-empty
        with open(jf, "r") as f:
            data = json.load(f)
            if data.get("error_tag", "") != "":
                print(f"Deleting json file with error: {jf}")
                jf.unlink()


def create_problem_result_dict(
    problem_name: str,
    problem_tag: str,
    train_time: float | None,
    predict_time: float | None,
    ground_truth: Array,
    predictions: Array,
    ungauranteed_mask: Array,
    TOL: float,
    DIM: int,
    F_DIM: int,
    N_TRAIN: int,
    N_CLUSTERS: int | None,
    error_tag: str,
    num_levels: int | None,
    **kwargs,
) -> dict:
    if ground_truth is None:
        # Experiment failed
        return {
            "problem_name": problem_name,
            "problem_tag": problem_tag,
            "tol": TOL,
            "dim": DIM,
            "f_dim": F_DIM,
            "n_train": N_TRAIN,
            "n_test": None,
            "n_clusters": N_CLUSTERS,
            "in_ptwise_max_rel_error_linf": None,
            "in_ptwise_median_rel_error_linf": None,
            "in_l2_relative_error": None,
            "in_l2_abs_error": None,
            "out_l2_relative_error": None,
            "out_l2_abs_error": None,
            "out_ptwise_max_rel_error_linf": None,
            "out_ptwise_median_rel_error_linf": None,
            "all_ptwise_max_rel_error_linf": None,
            "all_ptwise_median_rel_error_linf": None,
            "all_l2_relative_error": None,
            "all_l2_abs_error": None,
            "n_out": None,
            "perct_out": None,
            "n_in": None,
            "n_in_good": None,
            "perct_in_good": None,
            "n_in_bad": None,
            "perct_in_bad": None,
            "train_time_sec": None,
            "predict_time_sec": None,
            "error_tag": error_tag,
            "num_levels": num_levels,
            **kwargs,
        }

    # ground_truth shape is (N_TEST, F_DIM)
    N_TEST = ground_truth.shape[0]

    n_out = xp.sum(ungauranteed_mask)
    n_in = N_TEST - n_out
    g_idx = ~ungauranteed_mask

    if n_in == 0:
        # Handle the case where NO points are guaranteed
        in_ptwise_max_rel_error_linf = None
        in_ptwise_median_rel_error_linf = None
        in_l2_relative_error = None
        in_l2_abs_error = None

        n_in_good = 0
        n_in_bad = 0
        perct_in_good = None
        perct_in_bad = None
    else:
        # 1. Filter to valid (guaranteed) points
        gt_in = ground_truth[g_idx]
        pred_in = predictions[g_idx]

        # 2. Compute Element-wise Diff
        diff = pred_in - gt_in

        # 3. Global L2 Metrics (Frobenius Norm)
        # We calculate the norm over the entire flattened array of differences.
        # This corresponds to sqrt(sum(|y_ij - yhat_ij|^2)) for all i, j.
        norm_diff = xp.linalg.norm(diff)
        norm_gt = xp.linalg.norm(gt_in)
        in_l2_relative_error = float(norm_diff / (norm_gt + TINY_EPS))
        in_l2_abs_error = float(norm_diff / xp.sqrt(n_in))

        # 4. Pointwise Relative Error (L_inf style)
        element_rel_error = xp.abs(diff) / (xp.abs(gt_in) + TINY_EPS)

        if element_rel_error.ndim > 1:
            # Vector Case: A point is "good" only if ALL components are good.
            # Equivalently, the point error is the MAX error across components.
            ptwise_rel_error = xp.max(element_rel_error, axis=1)  # Shape: (N_in,)
        else:
            # Scalar Case
            ptwise_rel_error = element_rel_error  # Shape: (N_in,)

        # 5. Aggregate Stats
        in_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error))
        in_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error))

        # 6. Good/Bad Counts
        n_in_good = int(xp.sum(ptwise_rel_error <= TOL))
        n_in_bad = int(n_in - n_in_good)

        perct_in_good = 100.0 * n_in_good / N_TEST
        perct_in_bad = 100.0 * n_in_bad / N_TEST

    # Calculate similar error metrics for OUTSIDE trust region points
    if n_out == 0:
        out_l2_relative_error = None
        out_l2_abs_error = None
        out_ptwise_max_rel_error_linf = None
        out_ptwise_median_rel_error_linf = None
    else:
        gt_out = ground_truth[ungauranteed_mask]
        pred_out = predictions[ungauranteed_mask]
        diff = pred_out - gt_out
        norm_diff = xp.linalg.norm(diff)
        norm_gt = xp.linalg.norm(gt_out)
        out_l2_relative_error = float(norm_diff / (norm_gt + TINY_EPS))
        out_l2_abs_error = float(norm_diff / xp.sqrt(n_out))
        element_rel_error = xp.abs(diff) / (xp.abs(gt_out) + TINY_EPS)
        if element_rel_error.ndim > 1:
            ptwise_rel_error_out = xp.max(element_rel_error, axis=1)
        else:
            ptwise_rel_error_out = element_rel_error
        out_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error_out))
        out_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error_out))

    # Calculate simple error metrics for all points
    diff = predictions - ground_truth
    norm_diff = xp.linalg.norm(diff)
    norm_gt = xp.linalg.norm(ground_truth)
    all_l2_relative_error = float(norm_diff / (norm_gt + TINY_EPS))
    all_l2_abs_error = float(norm_diff / xp.sqrt(N_TEST))
    element_rel_error = xp.abs(diff) / (xp.abs(ground_truth) + TINY_EPS)
    if element_rel_error.ndim > 1:
        ptwise_rel_error_all = xp.max(element_rel_error, axis=1)
    else:
        ptwise_rel_error_all = element_rel_error
    all_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error_all))
    all_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error_all))

    # pout: percentage of points outside the trust region
    perct_out = float(100.0 * n_out / N_TEST)

    # Mandatory metrics
    result_dict = {
        "problem_name": problem_name,
        "problem_tag": problem_tag,
        "tol": TOL,
        "dim": DIM,
        "f_dim": F_DIM,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_clusters": N_CLUSTERS,
        "in_ptwise_max_rel_error_linf": in_ptwise_max_rel_error_linf,
        "in_ptwise_median_rel_error_linf": in_ptwise_median_rel_error_linf,
        "in_l2_relative_error": in_l2_relative_error,
        "in_l2_abs_error": in_l2_abs_error,
        "out_l2_relative_error": out_l2_relative_error,
        "out_l2_abs_error": out_l2_abs_error,
        "out_ptwise_max_rel_error_linf": out_ptwise_max_rel_error_linf,
        "out_ptwise_median_rel_error_linf": out_ptwise_median_rel_error_linf,
        "all_ptwise_max_rel_error_linf": all_ptwise_max_rel_error_linf,
        "all_ptwise_median_rel_error_linf": all_ptwise_median_rel_error_linf,
        "all_l2_relative_error": all_l2_relative_error,
        "all_l2_abs_error": all_l2_abs_error,
        "n_out": int(n_out),
        "perct_out": perct_out,
        "n_in": int(n_in),
        "n_in_good": int(n_in_good),
        "perct_in_good": perct_in_good,
        "n_in_bad": int(n_in_bad),
        "perct_in_bad": perct_in_bad,
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "error_tag": error_tag,
        "num_levels": num_levels,
    }
    # Add any additional metrics
    for key, value in kwargs.items():
        result_dict[key] = value

    return result_dict


def Problem_experiment_setup(
    json_fpath: Path,
    DOMAIN: list[float],
    f_fn: Callable,
    g_fn: Callable,
    DIM: int,
    F_DIM: int,
    N_TRAIN: int,
    N_INIT_CLUSTERS: int,
    TOL: float,
    N_MIN_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    N_TEST: int,
    K_NEIGHBOURS: int,
    USE_HESSIAN_INFORMED_SPLIT: bool,
    CLUSTERING_VERSION: Literal["v1", "v2"],
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    SAMPLING: Literal["Uniform", "Trajectories"] = "Uniform",
    PLOT_CONTOUR_IN_LOG_SCALE: bool = False,
    STORE_TRAIN_EVALS: bool = False,
    TRAIN_EVALS_PREFIX: str = "train_evals",
    TRAIN_EVALS_DIR: Optional[Path] = None,
    IGNORE_GRADIENT: bool = False,
    BATCH_SIZE: int = 10_000_000,
    PROFILE_TIMING: bool = False,
    SAVE_TRAINED_DATA: bool = False,
    SAVE_TEST_DATA: bool = False,
):
    if json_fpath.exists():
        print(f"Result file {json_fpath} already exists. Skipping experiment.")
        return

    # Sample data from a uniform distribution (if we choose to do so)
    if SAMPLING == "Uniform":
        xp.random.seed(SEED)
        X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
        X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))

    # Can also sample along trajectories, which is relevant for ODE applications.
    elif SAMPLING == "Trajectories":
        xp.random.seed(SEED)
        N_timesteps = 10
        N_train_init = N_TRAIN // N_timesteps
        N_test_init = N_TEST // N_timesteps

        # Save as tensor, then flatten at the end
        X_train = xp.zeros(shape=(N_timesteps, N_train_init, DIM))
        X_test = xp.zeros(shape=(N_timesteps, N_train_init, DIM))

        X_train[0] = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_train_init, DIM))
        X_test[0] = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_test_init, DIM))
        for j in range(1, N_timesteps):
            X_train[j] = f_fn(X_train[j - 1])
            X_test[j] = f_fn(X_test[j - 1])

        X_train = X_train.reshape((N_train_init * N_timesteps, DIM))
        X_test = X_test.reshape((N_train_init * N_timesteps, DIM))

    surrogate = HATANNSurrogate(version=CLUSTERING_VERSION)
    error_tag = ""

    FAILED = False
    _TRAINED, _PREDICTED = False, False
    try:

        if STORE_TRAIN_EVALS:
            assert (
                TRAIN_EVALS_DIR is not None
            ), "TRAIN_EVALS_DIR must be provided if STORE_TRAIN_EVALS is True"
            TRAIN_EVALS_DIR.mkdir(parents=True, exist_ok=True)
            f_debug = capture_evaluations(
                f_fn,
                save_dir=TRAIN_EVALS_DIR,
                prefix=TRAIN_EVALS_PREFIX + "_f",
            )
            g_debug = capture_evaluations(
                g_fn,
                save_dir=TRAIN_EVALS_DIR,
                prefix=TRAIN_EVALS_PREFIX + "_g",
            )

            # Train
            print("\n--- Training ---")
            _start_train_time = time.perf_counter()
            surrogate.train(
                X_train=X_train,
                f_fn=f_debug,
                g_fn=g_debug,
                TOL=TOL,
                NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
                MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
                MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                SEED=SEED,
                ERROR_METRIC=ERROR_METRIC,
                USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
                IGNORE_GRADIENT=IGNORE_GRADIENT,
                BATCH_SIZE=BATCH_SIZE,
                PROFILE_TIMING=PROFILE_TIMING,
            )
            _end_train_time = time.perf_counter()
            _TRAINED = True
        else:
            # Train
            print("\n--- Training ---")
            _start_train_time = time.perf_counter()
            surrogate.train(
                X_train=X_train,
                f_fn=f_fn,
                g_fn=g_fn,
                TOL=TOL,
                NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
                MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
                MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                SEED=SEED,
                ERROR_METRIC=ERROR_METRIC,
                USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
                IGNORE_GRADIENT=IGNORE_GRADIENT,
                PROFILE_TIMING=PROFILE_TIMING,
            )
            _end_train_time = time.perf_counter()
            _TRAINED = True

        if SAVE_TRAINED_DATA:
            trained_data_path = json_fpath.with_name(
                json_fpath.stem + "_trained_clusters_data.npz"
            )
            _t_d = surrogate.get_trained_data_as_dict()
            np.savez_compressed(trained_data_path, **_t_d)
            print(f"Trained surrogate data saved to {trained_data_path}")

        # Evaluate
        print("--- Evaluation ---")
        ground_truth = f_fn(X_test)

        if SAVE_TEST_DATA:
            test_data_path = json_fpath.with_name(json_fpath.stem + "_test_data.npz")
            np.savez_compressed(
                test_data_path, X_test=X_test, ground_truth=ground_truth
            )
            print(f"Test data saved to {test_data_path}")

        # Example utilizing the new signature with Guarantee checks
        _start_predict_time = time.perf_counter()
        preds, ungauranteed_mask = surrogate.predict(
            X_test,
            k_nn=K_NEIGHBOURS,
            strict_trust_region=False,
        )
        _end_predict_time = time.perf_counter()
        _PREDICTED = True

        train_time = _end_train_time - _start_train_time
        predict_time = _end_predict_time - _start_predict_time

        N_CLUSTERS = surrogate.get_number_of_sources()
        num_levels = surrogate.profiling_stats["num_levels"]
        num_generated_pts = surrogate.profiling_stats["num_generated_pts"]
        total_f_evals = surrogate.profiling_stats["total_f_evals"]

    except Exception as e:
        FAILED = True
        error_tag = f"Experiment failed with exception: {e} | TRAINED={_TRAINED}, PREDICTED={_PREDICTED}"
        print(error_tag)

        error_path = json_fpath.with_suffix(".error.txt")
        with open(error_path, "w") as ef:
            ef.write(error_tag)

        ground_truth = None
        preds = None
        ungauranteed_mask = None
        train_time = None
        predict_time = None
        N_CLUSTERS = 0
        num_levels = None
        num_generated_pts = None
        total_f_evals = None

    result_dict = create_problem_result_dict(
        problem_name="Problem_experiment",
        problem_tag=f"{DIM}D_N{N_TRAIN}_InitClust{N_INIT_CLUSTERS}_Tol{TOL}_ErrScale{ERROR_METRIC}_MinPts{N_MIN_PER_CLUSTER}_MaxClustEst{MAX_CLUSTERS_EST}_K{K_NEIGHBOURS}",
        error_tag=error_tag,
        train_time=train_time,
        predict_time=predict_time,
        ground_truth=ground_truth,
        predictions=preds,
        ungauranteed_mask=ungauranteed_mask,
        num_levels=num_levels,
        num_generated_pts=num_generated_pts,
        total_f_evals=total_f_evals,
        TOL=TOL,
        DIM=DIM,
        F_DIM=F_DIM,
        N_TRAIN=N_TRAIN,
        N_CLUSTERS=N_CLUSTERS,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        K_NEIGHBOURS=K_NEIGHBOURS,
        CLUSTERING_VERSION=CLUSTERING_VERSION,
        USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
        IGNORE_GRADIENT=IGNORE_GRADIENT,
        ERROR_METRIC=ERROR_METRIC,
        BATCH_SIZE=BATCH_SIZE,
        PROFILE_TIMING=PROFILE_TIMING,
        profiling_stats=surrogate.profiling_stats,
    )

    with open(json_fpath, "w") as jf:
        json.dump(result_dict, jf, indent=4)

    if FAILED:
        return None

    plot_error_histogram(
        ground_truth=ground_truth, predictions=preds, json_fpath=json_fpath, TOL=TOL
    )

    if DIM == 2 and N_CLUSTERS < 1000 and F_DIM == 1:
        # 1 plot 2 subplots (1X2)
        # Subplot 1: contourf of shifted gaussians function
        # Subplot 2: scatter of final cluster centroids with trust ellipses on contourf
        x = xp.linspace(DOMAIN[0], DOMAIN[1], 200)
        y = xp.linspace(DOMAIN[0], DOMAIN[1], 200)
        X, Y = xp.meshgrid(x, y)
        Z = f_fn(xp.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)

        if IS_CUPY_BACKEND:
            X = xp.asnumpy(X)
            Y = xp.asnumpy(Y)
            Z = xp.asnumpy(Z)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        if PLOT_CONTOUR_IN_LOG_SCALE:
            cs = axs[0].contourf(
                X,
                Y,
                Z,
                levels=50,
                cmap="viridis",
                norm=LogNorm(vmin=Z.min() + 1e-6, vmax=Z.max()),
            )
        else:
            cs = axs[0].contourf(X, Y, Z, levels=50, cmap="viridis")
        fig.colorbar(cs, ax=axs[0])
        axs[0].set_title(
            f"Function Contour Plot (N_CLUSTERS={N_CLUSTERS})",
            fontsize=10,
        )
        axs[0].set_xlabel(r"$x_1$")
        axs[0].set_ylabel(r"$x_2$")
        axs[0].set_aspect("equal")

        # Subplot 2: scatter of cluster centroids with trust ellipses
        if PLOT_CONTOUR_IN_LOG_SCALE:
            cs2 = axs[1].contourf(
                X,
                Y,
                Z,
                levels=50,
                cmap="viridis",
                norm=LogNorm(vmin=Z.min() + 1e-6, vmax=Z.max()),
            )
        else:
            cs2 = axs[1].contourf(X, Y, Z, levels=50, cmap="viridis")

        centroids = surrogate.X_src
        Qs = surrogate.Qs_src
        radii = surrogate.trust_radii_src
        if centroids is None or Qs is None or radii is None:
            raise ValueError("Surrogate has no cluster centroids to plot.")

        if IS_CUPY_BACKEND:
            centroids = xp.asnumpy(centroids)
            Qs = xp.asnumpy(Qs)
            radii = xp.asnumpy(radii)

        axs[0].scatter(
            centroids[:, 0], centroids[:, 1], c="red", s=10, label="Cluster Centroids"
        )

        fig.colorbar(cs2, ax=axs[1])
        axs[1].scatter(
            centroids[:, 0], centroids[:, 1], c="red", s=10, label="Cluster Centroids"
        )

        # Draw ellipses
        for i in range(centroids.shape[0]):
            mu = centroids[i]
            Q = Qs[i]

            # 1. Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(Q)

            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # 2. Calculate Axis Lengths (Inverse Relationship)
            # The trust region boundary is: (x-mu)^T Q (x-mu) <= r^2
            # Length of axis = r / sqrt(eigenvalue)
            # We multiply by 2 to get full width/height (diameter, not radius)
            # Add epsilon to avoid division by zero
            axes_lengths = 2.0 * 1.0 / np.sqrt(eigvals + 1e-12)

            # 3. Assign Width/Height
            # We map index 0 to width and index 1 to height.
            width = axes_lengths[0]
            height = axes_lengths[1]

            # 4. Calculate Angle
            # Since 'width' is based on index 0, we calculate the angle of eigenvector 0.
            # eigvecs[:, 0] is the vector [x_component, y_component]
            v0 = eigvecs[:, 0]
            angle = np.degrees(np.arctan2(v0[1], v0[0]))

            # 5. Create Ellipse
            ellipse = Ellipse(
                xy=mu,
                width=width,
                height=height,
                angle=angle,
                edgecolor="black",
                facecolor="none",
                linestyle="-",
                alpha=0.7,
                lw=1.0,
            )
            axs[1].add_patch(ellipse)

        axs[1].set_title(
            f"Cluster Centroids with Trust Ellipses (N_CLUSTERS={N_CLUSTERS})",
            fontsize=10,
        )
        axs[1].set_xlabel(r"$x_1$")
        axs[1].set_ylabel(r"$x_2$")
        axs[1].set_aspect("equal")
        # axs[1].set_xlim(DOMAIN[0], DOMAIN[1])
        # axs[1].set_ylim(DOMAIN[0], DOMAIN[1])
        plt.legend()

        fname = json_fpath.name.replace(".json", "_centroids_ellipses.png")
        plot_path = json_fpath.parent / fname
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
