import json
import time
from functools import partial
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
    **kwargs,
) -> dict:
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

    plt.figure(figsize=(10, 8))
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


def experiment_adaptive_clustering_global_vs_local() -> Path:
    DIM = 2
    TOL = 1e-1
    N_TRAIN = int(2e5)
    N_INIT_CLUSTERS = 50
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 50_000
    N_TEST = 5e4
    K_NEIGHBOURS = 5
    CONTAINMENT_LEVEL = 1.0
    TRUST_REGION_CONTAINMENT = True

    EXP_DIR = ICS26_DIR / "adaptive_clustering_global_vs_local"
    for ERROR_SCALING_METHOD in ["global", "local"]:
        for ROSENBROCK_A in [100**-1, 1, 100]:
            json_fpath = (
                EXP_DIR
                / f"rosenbrock_a{ROSENBROCK_A}_dim{DIM}_tol{TOL}_train{N_TRAIN}_test{N_TEST}_initclusters{N_INIT_CLUSTERS}_minpercluster{N_MIN_PER_CLUSTER}_k{K_NEIGHBOURS}_errscaling_{ERROR_SCALING_METHOD}.json"
            )
            json_fpath.parent.mkdir(parents=True, exist_ok=True)
            if json_fpath.exists():
                print(f"Skipping existing: {json_fpath}")
                continue

            scaled_rosenbrock = partial(rosenbrock, a=ROSENBROCK_A)
            scaled_rosenbrock_grad = partial(rosenbrock_grad, a=ROSENBROCK_A)

            # 1. Setup Domain
            domain = xp.array([[-3, 3] * DIM]).reshape(DIM, 2)

            # Sample Training Data
            X_train = DataSampler.sample_latin_hypercube(domain, N=N_TRAIN, seed=SEED)
            X_train = xp.asarray(X_train)

            # 2. Initialize Surrogate
            surrogate = NearODESurrogate()

            # 3. Train
            print("--- Training ---")
            _start_train_time = time.perf_counter()
            surrogate.train(
                X_train=X_train,
                f_fn=scaled_rosenbrock,
                g_fn=scaled_rosenbrock_grad,
                NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
                MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
                TOL=TOL,
                MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                SEED=SEED,
                SYNTH_BUDGET_FACTOR=100,
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

            result_dict = create_problem_result_dict(
                problem_name="Rosenbrock",
                problem_tag="adaptive_clustering_global_vs_local",
                train_time=_end_train_time - _start_train_time,
                predict_time=_end_predict_time - _start_predict_time,
                ground_truth=ground_truth,
                predictions=preds,
                ungauranteed_mask=ungauranteed_mask,
                TOL=TOL,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_CLUSTERS=surrogate.get_number_of_sources(),
                ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
                ROSENBROCK_A=ROSENBROCK_A,
                N_INIT_CLUSTERS=N_INIT_CLUSTERS,
                N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
                K_NEIGHBOURS=K_NEIGHBOURS,
                TRUST_REGION_CONTAINMENT=TRUST_REGION_CONTAINMENT,
            )
            with open(json_fpath, "w") as f:
                json.dump(result_dict, f, indent=4)
            plot_error_histogram(
                ground_truth=ground_truth,
                predictions=preds,
                json_fpath=json_fpath,
                TOL=TOL,
            )

    return EXP_DIR


if __name__ == "__main__":
    dir_exp_1 = experiment_adaptive_clustering_global_vs_local()

    _dirs = [dir_exp_1]
    for d in _dirs:
        create_summary_dataframe(d)
