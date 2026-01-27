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
from src.surrogate import NearODESurrogate

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

    df.to_csv(results_dir / "summary.csv", index=False)
    print(f"Summary dataframe saved to {results_dir / 'summary.csv'}")
    # Count number of error cases
    n_errors = df["error_tag"].apply(lambda x: x != "").sum()
    if n_errors == 0:
        # print in green color
        print(f"\033[92mNo error cases in {results_dir}!\033[0m")
    else:
        # print in red color
        print(f"\033[91mError cases found: {n_errors} in {results_dir}!\033[0m")


def plot_error_histogram(
    ground_truth: Array, predictions: Array, json_fpath: Path, TOL: float
):
    img_fpath = json_fpath.with_suffix(".png")
    img_title = json_fpath.stem

    rel_error_linf = xp.abs(predictions - ground_truth) / (
        xp.abs(ground_truth) + TINY_EPS
    )
    n_within_tol = xp.sum(rel_error_linf <= TOL)
    perct_within_tol = 100.0 * n_within_tol / ground_truth.shape[0]

    plt.figure(figsize=(10, 6))
    if IS_CUPY_BACKEND:
        plt.hist(rel_error_linf.get().flatten(), bins=50, log=True)
    else:
        plt.hist(rel_error_linf.flatten(), bins=50, log=True)

    plt.axvline(
        x=TOL,
        color="r",
        linestyle="--",
        label=f"TOL={TOL} (N_within_tol={perct_within_tol:.2f}%)",
    )

    plt.ylabel("Frequency (log scale)")
    plt.xlabel(r"Relative Error $\| . \|_{\infty}$ (log scale)")
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
            "n_train": N_TRAIN,
            "n_test": None,
            "n_clusters": None,
            "max_rel_error_linf": None,
            "max_rel_error_l2": None,
            "median_rel_error_linf": None,
            "median_rel_error_l2": None,
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
            "num_levels": None,
            **kwargs,
        }

    N_TEST = ground_truth.shape[0]

    n_out = xp.sum(ungauranteed_mask)
    n_in = N_TEST - n_out
    g_idx = ~ungauranteed_mask

    if n_in == 0:
        # Handle the case where NO points are guaranteed
        ptwise_max_rel_error_linf = 0.0  # Or None/NaN depending on preference
        ptwise_median_rel_error_linf = 0.0
        l2_relative_error = 0.0
        l2_abs_error = 0.0

        perct_in_good = 0.0
        perct_in_bad = 0.0
        n_in_good = 0
        n_in_bad = 0
    else:
        # Original Logic
        ptwise_rel_error_linf = xp.abs(predictions[g_idx] - ground_truth[g_idx]) / (
            xp.abs(ground_truth[g_idx]) + TINY_EPS
        )
        ptwise_max_rel_error_linf = xp.nanmax(ptwise_rel_error_linf)
        ptwise_median_rel_error_linf = xp.nanmedian(ptwise_rel_error_linf)

        l2_relative_error = float(
            xp.linalg.norm(ground_truth[g_idx] - predictions[g_idx], ord=2)
            / xp.linalg.norm(ground_truth[g_idx], ord=2)
        )
        l2_abs_error = float(
            xp.linalg.norm(ground_truth[g_idx] - predictions[g_idx], ord=2)
            / xp.sqrt(n_in)  # Use n_in instead of g_idx.sum() for speed
        )

        n_in_good = xp.sum(ptwise_rel_error_linf <= TOL)
        n_in_bad = xp.sum(ptwise_rel_error_linf > TOL)
        perct_in_good = 100.0 * n_in_good / N_TEST
        perct_in_bad = 100.0 * n_in_bad / N_TEST

    # pout: percentage of points outside the trust region
    # pin_good: percentage of points inside the trust region with error below tolerance
    # pin_bad: percentage of points inside the trust region with error above tolerance
    perct_out = 100.0 * n_out / N_TEST
    perct_in_good = 100.0 * n_in_good / N_TEST
    perct_in_bad = 100.0 * n_in_bad / N_TEST

    # Mandatory metrics
    result_dict = {
        "problem_name": problem_name,
        "problem_tag": problem_tag,
        "tol": TOL,
        "dim": DIM,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_clusters": N_CLUSTERS,
        "ptwise_max_rel_error_linf": float(ptwise_max_rel_error_linf),
        "ptwise_median_rel_error_linf": float(ptwise_median_rel_error_linf),
        "l2_relative_error": l2_relative_error,
        "l2_abs_error": l2_abs_error,
        "n_out": int(n_out),
        "perct_out": float(perct_out),
        "n_in": int(n_in),
        "n_in_good": int(n_in_good),
        "perct_in_good": float(perct_in_good),
        "n_in_bad": int(n_in_bad),
        "perct_in_bad": float(perct_in_bad),
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
    PLOT_CONTOUR_IN_LOG_SCALE: bool = False,
    STORE_TRAIN_EVALS: bool = False,
    TRAIN_EVALS_PREFIX: str = "train_evals",
    TRAIN_EVALS_DIR: Optional[Path] = None,
    IGNORE_GRADIENT: bool = False,
):
    # Sample data from a uniform distribution
    xp.random.seed(SEED)
    X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
    X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))

    surrogate = NearODESurrogate(version=CLUSTERING_VERSION)
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
                USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
                ERROR_METRIC=ERROR_METRIC,
                IGNORE_GRADIENT=IGNORE_GRADIENT
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
                USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
                ERROR_METRIC=ERROR_METRIC,
                IGNORE_GRADIENT=IGNORE_GRADIENT
            )
            _end_train_time = time.perf_counter()
            _TRAINED = True

        # Evaluate
        print("--- Evaluation ---")
        ground_truth = f_fn(X_test)

        # Example utilizing the new signature with Guarantee checks
        _start_predict_time = time.perf_counter()
        preds, ungauranteed_mask = surrogate.predict(
            X_test,
            k_NEIGHBOURS=K_NEIGHBOURS,
        )
        _end_predict_time = time.perf_counter()
        _PREDICTED = True

        train_time = _end_train_time - _start_train_time
        predict_time = _end_predict_time - _start_predict_time

        N_CLUSTERS = surrogate.get_number_of_sources()
        num_levels = surrogate.training_num_levels

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

    result_dict = create_problem_result_dict(
        problem_name="Quadratic",
        problem_tag=f"{DIM}D_N{N_TRAIN}_InitClust{N_INIT_CLUSTERS}_Tol{TOL}_ErrScale{ERROR_METRIC}_MinPts{N_MIN_PER_CLUSTER}_MaxClustEst{MAX_CLUSTERS_EST}_K{K_NEIGHBOURS}",
        error_tag=error_tag,
        train_time=train_time,
        predict_time=predict_time,
        ground_truth=ground_truth,
        predictions=preds,
        ungauranteed_mask=ungauranteed_mask,
        num_levels=num_levels,
        TOL=TOL,
        DIM=DIM,
        N_TRAIN=N_TRAIN,
        N_CLUSTERS=N_CLUSTERS,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        K_NEIGHBOURS=K_NEIGHBOURS,
        CLUSTERING_VERSION=CLUSTERING_VERSION,
        USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
        ERROR_METRIC=ERROR_METRIC,
    )

    with open(json_fpath, "w") as jf:
        json.dump(result_dict, jf, indent=4)

    if FAILED:
        return None

    plot_error_histogram(
        ground_truth=ground_truth, predictions=preds, json_fpath=json_fpath, TOL=TOL
    )

    if DIM == 2 and N_CLUSTERS < 1000:
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
            f"Shifted Gaussians with Cluster Centroids (TOL={TOL})", fontsize=10
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
            radius = radii[i]

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(Q)
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            # Width and height are "2 * sqrt(eigenvalue)" for 1 std dev
            width, height = 2 * np.sqrt(eigvals)
            ellipse = Ellipse(
                xy=mu,
                width=width,
                height=height,
                angle=angle,
                edgecolor="blue",
                facecolor="none",
                linestyle="--",
                alpha=0.7,
                lw=0.8,
            )
            axs[1].add_patch(ellipse)

            # Draw trust radius circle
            circle = Ellipse(
                xy=mu,
                width=2 * radius,
                height=2 * radius,
                angle=0,
                edgecolor="black",
                facecolor="none",
                linestyle=":",
                alpha=0.7,
                lw=0.8,
            )
            axs[1].add_patch(circle)

        axs[1].set_title(
            f"Cluster Centroids with Trust Ellipses (N_CLUSTERS={N_CLUSTERS})",
            fontsize=10,
        )
        axs[1].set_xlabel(r"$x_1$")
        axs[1].set_ylabel(r"$x_2$")
        axs[1].set_aspect("equal")
        axs[1].set_xlim(DOMAIN[0], DOMAIN[1])
        axs[1].set_ylim(DOMAIN[0], DOMAIN[1])
        plt.legend()

        fname = json_fpath.name.replace(".json", "_centroids_ellipses.png")
        plot_path = json_fpath.parent / fname
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
