import json
import time
from functools import partial
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import IS_CUPY_BACKEND, RESULTS_DIR, Array, xp
from src.data_sampler import DataSampler
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.surrogate import NearODESurrogate

ICS26_DIR = RESULTS_DIR / "ics26"
ICS26_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
TINY_EPS = 1e-12


def create_summary_dataframe(results_dir: Path):
    assert results_dir.exists(), f"Directory {results_dir} does not exist."

    json_files = list(results_dir.glob("*.json"))
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
    train_time: float,
    predict_time: float,
    ground_truth: Array,
    predictions: Array,
    ungauranteed_mask: Array,
    TOL: float,
    DIM: int,
    N_TRAIN: int,
    N_CLUSTERS: int,
    error_tag: str,
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
            "n_test": 0,
            "n_clusters": None,
            "max_rel_error_linf": None,
            "max_rel_error_l2": None,
            "n_ungauranteed": None,
            "perct_ungauranteed": None,
            "n_within_tol": None,
            "perct_within_tol": None,
            "train_time_sec": None,
            "predict_time_sec": None,
            "error_tag": error_tag,
            **kwargs,
        }

    N_TEST = ground_truth.shape[0]
    rel_error_linf = xp.abs(predictions - ground_truth) / (
        xp.abs(ground_truth) + TINY_EPS
    )
    max_rel_error_linf = xp.nanmax(rel_error_linf)

    rel_error_l2 = xp.linalg.norm(ground_truth - predictions, axis=1) / (
        xp.linalg.norm(ground_truth, axis=1) + TINY_EPS
    )
    max_rel_error_l2 = xp.nanmax(rel_error_l2)

    n_ungauranteed = xp.sum(ungauranteed_mask)
    perct_ungauranteed = 100.0 * n_ungauranteed / N_TEST

    n_within_tol = xp.sum(rel_error_linf <= TOL)
    perct_within_tol = 100.0 * n_within_tol / N_TEST

    # Mandatory metrics
    result_dict = {
        "problem_name": problem_name,
        "problem_tag": problem_tag,
        "tol": TOL,
        "dim": DIM,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_clusters": N_CLUSTERS,
        "max_rel_error_linf": float(max_rel_error_linf),
        "max_rel_error_l2": float(max_rel_error_l2),
        "n_ungauranteed": int(n_ungauranteed),
        "perct_ungauranteed": float(perct_ungauranteed),
        "n_within_tol": int(n_within_tol),
        "perct_within_tol": float(perct_within_tol),
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "error_tag": error_tag,
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
    ERROR_SCALING_METHOD: str,
    N_MIN_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    N_TEST: int,
    K_NEIGHBOURS: int,
    CONTAINMENT_LEVEL: float,
    TRUST_REGION_CONTAINMENT: bool,
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
    surrogate = NearODESurrogate()
    error_tag = ""

    FAILED = False
    try:
        # 3. Train
        print("--- Training ---")
        _start_train_time = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=scaled_rosenbrock,
            g_fn=scaled_rosenbrock_grad,
            TOL=TOL,
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,  # type: ignore
            NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
            MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            SEED=SEED,
            SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
        )
        _end_train_time = time.perf_counter()

        # 4. Evaluate
        print("\n--- Evaluation ---")
        X_test = DataSampler.sample_uniform(domain, N=N_TEST, seed=SEED)
        X_test = xp.asarray(X_test)

        # Example utilizing the new signature with Guarantee checks
        _start_predict_time = time.perf_counter()
        preds, ungauranteed_mask = surrogate.predict(
            X_test,
            k_NEIGHBOURS=K_NEIGHBOURS,
            trust_region_containment=TRUST_REGION_CONTAINMENT,
            containment_level=CONTAINMENT_LEVEL,
        )
        _end_predict_time = time.perf_counter()

        ground_truth = scaled_rosenbrock(X_test)
        train_time = _end_train_time - _start_train_time
        predict_time = _end_predict_time - _start_predict_time

        N_CLUSTERS = surrogate.get_number_of_sources()
    except Exception as e:
        FAILED = True
        error_tag = f"Experiment failed with exception: {e}"
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

    result_dict = create_problem_result_dict(
        problem_name="Rosenbrock",
        problem_tag="adaptive_clustering_global_vs_local",
        error_tag=error_tag,
        train_time=train_time,
        predict_time=predict_time,
        ground_truth=ground_truth,
        predictions=preds,
        ungauranteed_mask=ungauranteed_mask,
        TOL=TOL,
        DIM=DIM,
        N_TRAIN=N_TRAIN,
        N_CLUSTERS=N_CLUSTERS,
        ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
        ROSENBROCK_A=ROSENBROCK_A,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        K_NEIGHBOURS=K_NEIGHBOURS,
        TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
        CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
        SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
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


def experiment_adaptive_clustering_global_vs_local() -> Path:
    DIM = 2
    N_TRAIN = int(2e5)
    N_INIT_CLUSTERS = 50
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 50_000
    N_TEST = int(5e4)
    K_NEIGHBOURS = 5
    CONTAINMENT_LEVEL = 1.0
    TRUST_REGION_CONTAINMENT = True

    EXP_DIR = ICS26_DIR / "adaptive_clustering_global_vs_local"
    del_all_error_files_in_dir(EXP_DIR)
    ERROR_SCALING_METHODs = ["global", "local"]
    ROSENBROCK_As = [100**-1, 1, 100]
    TOLs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]

    for ERROR_SCALING_METHOD, ROSENBROCK_A, TOL in product(
        ERROR_SCALING_METHODs, ROSENBROCK_As, TOLs
    ):
        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_errscaling_{ERROR_SCALING_METHOD}.json"
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
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
        )

    return EXP_DIR


def experiment_adaptive_clustering_global_vs_n_init_clusters() -> Path:
    DIM = 2
    N_TRAIN = int(2e5)
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 50_000
    N_TEST = int(5e4)
    K_NEIGHBOURS = 5
    CONTAINMENT_LEVEL = 1.0
    TRUST_REGION_CONTAINMENT = True
    ERROR_SCALING_METHOD = "global"

    EXP_DIR = ICS26_DIR / "adaptive_clustering_global_vs_n_init_clusters"
    del_all_error_files_in_dir(EXP_DIR)
    N_INIT_CLUSTERSs = [1, 5, 25, 125, 625, 1000]
    TOLs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    ROSENBROCK_As = [1, 100]

    for N_INIT_CLUSTERS, TOL, ROSENBROCK_A in product(
        N_INIT_CLUSTERSs, TOLs, ROSENBROCK_As
    ):
        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_errscaling_{ERROR_SCALING_METHOD}.json"
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
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
            SYNTH_BUDGET_FACTOR=500,
        )
    return EXP_DIR


def experiment_adaptive_clustering_global_local_vs_min_pts_cluster():
    DIM = 2
    N_TRAIN = int(2e5)
    N_INIT_CLUSTERS = 50
    MAX_CLUSTERS_EST = 50_000
    N_TEST = int(5e4)
    K_NEIGHBOURS = 5
    CONTAINMENT_LEVEL = 1.0
    TRUST_REGION_CONTAINMENT = True

    EXP_DIR = ICS26_DIR / "adaptive_clustering_global_local_vs_min_pts_per_cluster"
    del_all_error_files_in_dir(EXP_DIR)
    ERROR_SCALING_METHODs = ["global", "local"]
    N_MIN_PER_CLUSTERs = [5, 25, 125, 625, 1000]
    TOLs = [1e-1, 1e-2, 1e-3]
    ROSENBROCK_As = [1, 100]

    for ERROR_SCALING_METHOD, N_MIN_PER_CLUSTER, TOL, ROSENBROCK_A in product(
        ERROR_SCALING_METHODs, N_MIN_PER_CLUSTERs, TOLs, ROSENBROCK_As
    ):
        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_errscaling_{ERROR_SCALING_METHOD}.json"
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
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
            SYNTH_BUDGET_FACTOR=500,
        )

    return EXP_DIR


def experiment_ellipsoidal_knn_containment() -> Path:
    DIM = 2
    N_TRAIN = int(2e5)
    N_INIT_CLUSTERS = 25
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 50_000
    N_TEST = int(5e4)
    ERROR_SCALING_METHOD = "global"
    ROSENBROCK_A = 100

    EXP_DIR = ICS26_DIR / "ellipsoidal_knn_containment"
    del_all_error_files_in_dir(EXP_DIR)
    K_NEIGHBOURSs = [1, 2, 3, 5, 10]
    CONTAINMENT_LEVELs = [1.0, 2.0, 3.0]
    TRUST_REGION_CONTAINMENTs = [True, False]
    TOLs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for K_NEIGHBOURS, CONTAINMENT_LEVEL, TRUST_REGION_CONTAINMENT, TOL in product(
        K_NEIGHBOURSs, CONTAINMENT_LEVELs, TRUST_REGION_CONTAINMENTs, TOLs
    ):
        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_errscaling_{ERROR_SCALING_METHOD}_containment{CONTAINMENT_LEVEL}_trustregion{TRUST_REGION_CONTAINMENT}.json"
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
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
            SYNTH_BUDGET_FACTOR=500,
        )

    return EXP_DIR


def experiment_rosenbrock_convergence() -> Path:
    N_INIT_CLUSTERS = 25
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 50_000
    N_TRAIN = int(2e5)
    N_TEST = int(1e5)
    K_NEIGHBOURS = 5
    CONTAINMENT_LEVEL = 1.0
    TRUST_REGION_CONTAINMENT = True
    ERROR_SCALING_METHOD = "global"

    EXP_DIR = ICS26_DIR / "rosenbrock_convergence"
    del_all_error_files_in_dir(EXP_DIR)
    DIMs = [2, 5]
    ROSENBROCK_As = [1, 100]
    TOLs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]

    for DIM, ROSENBROCK_A, TOL in product(DIMs, ROSENBROCK_As, TOLs):
        json_fpath = (
            EXP_DIR
            / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_errscaling_{ERROR_SCALING_METHOD}.json"
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
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
            SYNTH_BUDGET_FACTOR=300,
        )

    return EXP_DIR


if __name__ == "__main__":
    dir_exp_1 = experiment_adaptive_clustering_global_vs_local()
    create_summary_dataframe(dir_exp_1)

    dir_exp_3 = experiment_adaptive_clustering_global_local_vs_min_pts_cluster()
    create_summary_dataframe(dir_exp_3)

    dir_exp_4 = experiment_ellipsoidal_knn_containment()
    create_summary_dataframe(dir_exp_4)

    dir_exp_5 = experiment_rosenbrock_convergence()
    create_summary_dataframe(dir_exp_5)

    dir_exp_2 = experiment_adaptive_clustering_global_vs_n_init_clusters()
    create_summary_dataframe(dir_exp_2)

    _dirs = [dir_exp_1, dir_exp_2, dir_exp_3, dir_exp_4, dir_exp_5]
    for d in _dirs:
        create_summary_dataframe(d)
