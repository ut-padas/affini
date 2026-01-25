import json
import time
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from src import IS_CUPY_BACKEND, RESULTS_DIR, Array, xp
from src.data_sampler import DataSampler
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.surrogate import NearODESurrogate

ICS26_DIR = RESULTS_DIR / "ics26"
ICS26_DIR.mkdir(parents=True, exist_ok=True)

TINY_EPS = 1e-12

SEED = 0
N_INIT_CLUSTERS = 25
N_MIN_PER_CLUSTER = 100
MAX_CLUSTERS_EST = 500_000


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

    # Consider only guaranteed points for error metrics
    g_idx = ~ungauranteed_mask
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
        / xp.sqrt(g_idx.sum())
    )

    n_in = N_TEST - n_out
    # pout: percentage of points outside the trust region
    # pin_good: percentage of points inside the trust region with error below tolerance
    # pin_bad: percentage of points inside the trust region with error above tolerance
    perct_out = 100.0 * n_out / N_TEST

    n_in_good = xp.sum(ptwise_rel_error_linf <= TOL)
    n_in_bad = xp.sum(ptwise_rel_error_linf > TOL)
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


def rosenbrock_experiment_setup(
    json_fpath: Path,
    DIM: int,
    N_TRAIN: int,
    N_INIT_CLUSTERS: int,
    ROSENBROCK_A: float,
    TOL: float,
    N_MIN_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    N_TEST: int,
    K_NEIGHBOURS: int,
    CLUSTERING_VERSION: Literal["v1", "v2"],
    ERROR_SPLIT_CRITERION: Literal["L2", "Linf"] = "L2",
    USE_HESSIAN_INFORMED_SPLIT: bool = True,
    SYNTH_BUDGET_FACTOR: int = 100,
):
    scaled_rosenbrock = partial(rosenbrock, a=ROSENBROCK_A)
    scaled_rosenbrock_grad = partial(rosenbrock_grad, a=ROSENBROCK_A)

    # 1. Setup Domain
    domain = xp.array([[-3, 3] * DIM]).reshape(DIM, 2)

    # Sample Training Data
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_TRAIN, seed=SEED)
    X_train = xp.asarray(X_train)

    # 2. Initialize Surrogate
    surrogate = NearODESurrogate(version=CLUSTERING_VERSION)
    error_tag = ""

    FAILED = False
    _TRAINED, _PREDICTED = False, False
    try:
        # 3. Train
        print("--- Training ---")
        _start_train_time = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=scaled_rosenbrock,
            g_fn=scaled_rosenbrock_grad,
            TOL=TOL,
            NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
            MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            SEED=SEED,
            SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
            USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
            ERROR_SPLIT_CRITERION=ERROR_SPLIT_CRITERION,
        )
        _end_train_time = time.perf_counter()
        _TRAINED = True

        # 4. Evaluate
        print("\n--- Evaluation ---")
        X_test = DataSampler.sample_uniform(domain, N=N_TEST, seed=SEED)
        X_test = xp.asarray(X_test)

        # Example utilizing the new signature with Guarantee checks
        _start_predict_time = time.perf_counter()
        preds, ungauranteed_mask = surrogate.predict(
            X_test,
            k_NEIGHBOURS=K_NEIGHBOURS,
        )
        _end_predict_time = time.perf_counter()
        _PREDICTED = True

        ground_truth = scaled_rosenbrock(X_test)
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
        train_time = -1.0
        predict_time = -1.0
        N_CLUSTERS = -1
        num_levels = -1

    result_dict = create_problem_result_dict(
        problem_name="Rosenbrock",
        problem_tag="adaptive_clustering_global_vs_local",
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
        ROSENBROCK_A=ROSENBROCK_A,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        K_NEIGHBOURS=K_NEIGHBOURS,
        SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
        CLUSTERING_VERSION=CLUSTERING_VERSION,
        USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
        ERROR_SPLIT_CRITERION=ERROR_SPLIT_CRITERION,
    )

    with open(json_fpath, "w") as f:
        json.dump(result_dict, f, indent=4)

    if FAILED:
        return
    plot_error_histogram(
        ground_truth=ground_truth,
        predictions=preds,
        json_fpath=json_fpath,
        TOL=TOL,
    )

    if DIM == 2:
        # plot contours of the function and the centroids on domain -3 to 3
        x1 = xp.linspace(-3, 3, 100)
        x2 = xp.linspace(-3, 3, 100)

        X1, X2 = xp.meshgrid(x1, x2)
        Z = scaled_rosenbrock(xp.vstack([X1.ravel(), X2.ravel()]).T).reshape(100, 100)

        plt.figure(figsize=(10, 6))
        cp = plt.contourf(X1, X2, Z, levels=50, cmap="viridis", norm=LogNorm())

        centroids = surrogate.X_src
        radii = surrogate.trust_radii
        if centroids is None or radii is None:
            print("Centroids or radii are None, skipping centroid plot.")
            raise ValueError("Centroids or radii are None.")

        for i in range(centroids.shape[0]):
            circle = Circle(
                (centroids[i, 0], centroids[i, 1]),
                radii[i],
                color="red",
                fill=False,
                linestyle="--",
                alpha=0.5,
            )
            plt.gca().add_patch(circle)

        # plot centroids
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            color="red",
            marker="o",
            label="Centroids",
        )
        plt.colorbar(cp)

        fname = json_fpath.with_name(json_fpath.stem + "_centroids.png")
        plt.title(
            f"Rosenbrock Function Contours and Centroids\n{json_fpath.stem}",
            fontsize=8,
        )
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.axis("equal")
        plt.savefig(fname, dpi=300)


def experiment_rosenbrock_convergence_error_splitting() -> Path:
    N_TRAIN = int(2e5)
    N_TEST = int(1e5)
    K_NEIGHBOURS = 1
    USE_HESSIAN_INFORMED_SPLITs = [True, False]
    ERROR_SPLIT_CRITERIONs = ["L2", "Linf"]
    CLUSTERING_VERSION = "v1"

    EXP_DIR = (
        ICS26_DIR
        / "rosenbrock_convergence_v1_hessian_informed_error_splitting_experiments"
    )
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 3]
    ROSENBROCK_As = [1, 100]
    TOLs = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]

    for (
        DIM,
        ROSENBROCK_A,
        TOL,
        USE_HESSIAN_INFORMED_SPLIT,
        ERROR_SPLIT_CRITERION,
    ) in product(
        DIMs, ROSENBROCK_As, TOLs, USE_HESSIAN_INFORMED_SPLITs, ERROR_SPLIT_CRITERIONs
    ):
        if DIM > 2 and TOL <= 1e-3:
            print(
                f"Skipping infeasible case: DIM={DIM}, TOL={TOL}, ROSENBROCK_A={ROSENBROCK_A}"
            )
            continue
        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_hessian_informed_{USE_HESSIAN_INFORMED_SPLIT}_error_split_criterion_{ERROR_SPLIT_CRITERION}.json"
        )
        json_fpath.parent.mkdir(parents=True, exist_ok=True)
        if json_fpath.exists():
            print(f"Skipping existing: {json_fpath}")
            continue

        rosenbrock_experiment_setup(
            json_fpath=json_fpath,
            DIM=DIM,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            ROSENBROCK_A=ROSENBROCK_A,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            SYNTH_BUDGET_FACTOR=10,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
            ERROR_SPLIT_CRITERION=ERROR_SPLIT_CRITERION,  # type: ignore
        )

    return EXP_DIR


def experiment_rosenbrock_convergence_v2() -> Path:
    N_TRAIN = int(2e5)
    N_TEST = int(1e5)
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "rosenbrock_v2"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 3, 4, 5, 10]
    ROSENBROCK_As = [1, 100]
    TOLs = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    for DIM, ROSENBROCK_A, TOL in product(DIMs, ROSENBROCK_As, TOLs):
        if DIM > 3 and TOL <= 1e-3:
            print(
                f"Skipping infeasible case: DIM={DIM}, TOL={TOL}, ROSENBROCK_A={ROSENBROCK_A}"
            )
            continue

        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}.json"
        )
        json_fpath.parent.mkdir(parents=True, exist_ok=True)
        if json_fpath.exists():
            print(f"Skipping existing: {json_fpath}")
            continue

        rosenbrock_experiment_setup(
            json_fpath=json_fpath,
            DIM=DIM,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            ROSENBROCK_A=ROSENBROCK_A,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            SYNTH_BUDGET_FACTOR=100 * 10,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            USE_HESSIAN_INFORMED_SPLIT=True,
            ERROR_SPLIT_CRITERION="L2",  # type: ignore
        )

    return EXP_DIR


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()

    # dir_exp_1 = experiment_rosenbrock_convergence_error_splitting()
    # create_summary_dataframe(dir_exp_1)

    dir_exp_2 = experiment_rosenbrock_convergence_v2()
    create_summary_dataframe(dir_exp_2)
