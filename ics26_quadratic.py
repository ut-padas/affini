import json
import time
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

from ics26_experiments import (
    ICS26_DIR,
    SEED,
    create_problem_result_dict,
    create_summary_dataframe,
    del_all_error_files_in_dir,
    plot_error_histogram,
)
from src import IS_CUPY_BACKEND, xp
from src.problems.quadratic import quadratic, quadratic_grad
from src.surrogate import NearODESurrogate

N_INIT_CLUSTERS = 25
N_MIN_PER_CLUSTER = 100
MAX_CLUSTERS_EST = 10_000
K_NEIGHBOURS = 1
CONTAINMENT_LEVEL = 1.0
ENFORCE_CONTAINMENT = False
TOLs = [i / 100.0 for i in [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]]


def Quadratic_experiment_setup(
    json_fpath: Path,
    DIM: int,
    N_TRAIN: int,
    N_INIT_CLUSTERS: int,
    TOL: float,
    ERROR_SCALING_METHOD: Literal["global", "local"],
    CLUSTER_SPLITTING_METHOD: Literal["hessian_pca", "pca_only"],
    N_MIN_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    N_TEST: int,
    K_NEIGHBOURS: int,
    CONTAINMENT_LEVEL: float,
    ENFORCE_CONTAINMENT: bool,
    Q_SHRINK: float,
    CLUSTERING_VERSION: Literal["v1", "v2"],
    SYNTH_BUDGET_FACTOR: int = 10,
):
    # Sample data from a uniform distribution
    DOMAIN = [-2.0, 2.0]
    xp.random.seed(SEED)
    X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
    X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))

    surrogate = NearODESurrogate(version=CLUSTERING_VERSION)
    error_tag = ""

    FAILED = False
    _TRAINED, _PREDICTED = False, False
    try:
        # Train
        print("\n--- Training ---")
        _start_train_time = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=quadratic,
            g_fn=quadratic_grad,
            TOL=TOL,
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            CLUSTER_SPLITTING_METHOD=CLUSTER_SPLITTING_METHOD,
            NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
            MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            SEED=SEED,
            SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
            Q_SHRINK=Q_SHRINK,
        )
        _end_train_time = time.perf_counter()
        _TRAINED = True

        # Evaluate
        print("--- Evaluation ---")
        ground_truth = quadratic(X_test)

        # Example utilizing the new signature with Guarantee checks
        _start_predict_time = time.perf_counter()
        preds, ungauranteed_mask = surrogate.predict(
            X_test,
            k_NEIGHBOURS=K_NEIGHBOURS,
            enforce_containment=ENFORCE_CONTAINMENT,
            containment_level=CONTAINMENT_LEVEL,
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
        problem_tag=f"{DIM}D_N{N_TRAIN}_InitClust{N_INIT_CLUSTERS}_Tol{TOL}_ErrScale{ERROR_SCALING_METHOD}_MinPts{N_MIN_PER_CLUSTER}_MaxClustEst{MAX_CLUSTERS_EST}_K{K_NEIGHBOURS}_ContainLvl{CONTAINMENT_LEVEL}_EnfContain{ENFORCE_CONTAINMENT}_QShrink{Q_SHRINK}",
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
        ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
        CLUSTER_SPLITTING_METHOD=CLUSTER_SPLITTING_METHOD,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        K_NEIGHBOURS=K_NEIGHBOURS,
        ENFORCE_CONTAINMENT=ENFORCE_CONTAINMENT,
        CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
        Q_SHRINK=Q_SHRINK,
        SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
        CLUSTERING_VERSION=CLUSTERING_VERSION,
    )

    with open(json_fpath, "w") as jf:
        json.dump(result_dict, jf, indent=4)

    if FAILED:
        return None

    plot_error_histogram(
        ground_truth=ground_truth, predictions=preds, json_fpath=json_fpath, TOL=TOL
    )

    if DIM == 2 and N_CLUSTERS < 3000:
        # 1 plot 2 subplots (1X2)
        # Subplot 1: contourf of shifted gaussians function
        # Subplot 2: scatter of final cluster centroids with trust ellipses on contourf
        x = xp.linspace(DOMAIN[0], DOMAIN[1], 200)
        y = xp.linspace(DOMAIN[0], DOMAIN[1], 200)
        X, Y = xp.meshgrid(x, y)
        Z = quadratic(xp.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)

        if IS_CUPY_BACKEND:
            X = xp.asnumpy(X)
            Y = xp.asnumpy(Y)
            Z = xp.asnumpy(Z)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        cs = axs[0].contourf(X, Y, Z, levels=50, cmap="viridis")
        fig.colorbar(cs, ax=axs[0])
        axs[0].set_title("Quadratic with Cluster Centroids")
        axs[0].set_xlabel(r"$x_1$")
        axs[0].set_ylabel(r"$x_2$")
        axs[0].set_aspect("equal")

        # Subplot 2: scatter of cluster centroids with trust ellipses
        cs2 = axs[1].contourf(X, Y, Z, levels=50, cmap="viridis")

        centroids = surrogate.X_src
        if centroids is None:
            raise ValueError("Cluster centroids are None")

        if CLUSTERING_VERSION == "v2":
            Qs = surrogate.trust_ellipses_src
            if Qs is None:
                raise ValueError("Trust Ellipses are None")

            if IS_CUPY_BACKEND:
                centroids = xp.asnumpy(centroids)
                Qs = xp.asnumpy(Qs)

            fig.colorbar(cs2, ax=axs[1])
            axs[1].scatter(
                centroids[:, 0],
                centroids[:, 1],
                c="red",
                s=10,
                label="Cluster Centroids",
            )

            # Draw ellipses
            for i in range(centroids.shape[0]):
                mu = centroids[i]
                Q = Qs[i]
                w, V = np.linalg.eigh(Q)
                w = np.maximum(w, 1e-12)
                order = w.argsort()[::-1]
                w = w[order]
                V = V[:, order]

                a = np.sqrt(1.0 / w[0])
                b = np.sqrt(1.0 / w[1])
                angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
                ellipse = Ellipse(
                    xy=mu,
                    width=2 * a,
                    height=2 * b,
                    angle=angle,
                    edgecolor="red",
                    fc="None",
                    lw=1,
                    alpha=0.5,
                )
                axs[1].add_patch(ellipse)
        else:
            trust_radii = surrogate.trust_ellipses_src  # (N_CLUSTERS, )
            if trust_radii is None:
                raise ValueError("Trust Radii are None")

            if IS_CUPY_BACKEND:
                centroids = xp.asnumpy(centroids)
                trust_radii = xp.asnumpy(trust_radii)

            # Draw circles
            for i in range(centroids.shape[0]):
                mu = centroids[i]
                r = trust_radii[i]
                circle = Circle(
                    xy=mu,
                    radius=r,
                    edgecolor="red",
                    fc="None",
                    lw=1,
                    alpha=0.5,
                )
                axs[1].add_patch(circle)

        axs[1].set_title(
            f"Cluster Centroids with Trust Ellipses (N_CLUSTERS={N_CLUSTERS})"
        )
        axs[1].set_xlabel(r"$x_1$")
        axs[1].set_ylabel(r"$x_2$")
        axs[1].set_aspect("equal")
        plt.legend()

        fname = json_fpath.name.replace(".json", "_centroids_ellipses.png")
        plot_path = json_fpath.parent / fname
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)


def experiment_with_q_shrink() -> Path:
    CLUSTER_SPLITTING_METHOD = "hessian_pca"
    Q_SHRINK = 0.05
    CLUSTERING_VERSION = "v1"

    EXP_DIR = ICS26_DIR / "quadratic_with_q_shrink"
    del_all_error_files_in_dir(EXP_DIR)
    DIMs = [2, 3, 4, 5]
    ERROR_SCALING_METHODs = ["local"]

    for DIM, TOL, ERROR_SCALING_METHOD in product(DIMs, TOLs, ERROR_SCALING_METHODs):
        json_path = (
            EXP_DIR
            / f"ics26_quadratic_dim{DIM}_tol{TOL}_errscale{ERROR_SCALING_METHOD}_qshrink{Q_SHRINK}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue

        if DIM == 2:
            N_TRAIN = 100_000
            N_TEST = 10_000
        else:
            N_TRAIN = 100_000
            N_TEST = 10_000

        Quadratic_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,  # type: ignore
            CLUSTER_SPLITTING_METHOD=CLUSTER_SPLITTING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            ENFORCE_CONTAINMENT=ENFORCE_CONTAINMENT,
            Q_SHRINK=Q_SHRINK,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            SYNTH_BUDGET_FACTOR=1000,
        )

    return EXP_DIR


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()

    dir1 = experiment_with_q_shrink()
    create_summary_dataframe(dir1)
