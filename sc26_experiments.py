import json
import logging
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from src import IS_CUPY_BACKEND, RESULTS_DIR, _get_mpi_rank, _get_mpi_size, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.heuristic_parameters import (
    get_MIN_PTS_PER_CLUSTER,
    get_N_TRAIN,
    get_NUM_INITIAL_CLUSTERS,
)
from src.plotter_utils import plot_adaptive_clustering_results, plot_error_histogram
from src.surrogate import AFINNISurrogate, compute_error_metrics

SC26_DIR = RESULTS_DIR / "sc26"
SC26_DIR.mkdir(parents=True, exist_ok=True)
TINY_EPS = 1e-12
SEED = 0

logger = logging.getLogger(__name__)
_size = _get_mpi_size()
_rank = _get_mpi_rank()


def cap_initial_augmentation_budget(
    n_train: int,
    num_initial_clusters: int,
    min_pts_per_cluster: int,
    *,
    min_initial_clusters: int = 16,
) -> tuple[int, int]:
    """
    Keep the level-0 rejected-cluster augmentation budget compatible with the
    available training set once ``N_TRAIN`` has been capped.

    Rejected clusters are padded to at least ``2 * MIN_PTS_PER_CLUSTER`` before
    splitting.  The raw dimension-scaling heuristics keep growing after
    ``N_TRAIN`` is capped, which can otherwise make level 0 request far more
    synthetic points than the capped experiment can support.
    """
    max_min_pts_for_budget = max(2, n_train // (2 * min_initial_clusters))
    capped_min_pts = min(min_pts_per_cluster, max_min_pts_for_budget)
    max_initial_clusters_for_budget = max(1, n_train // (2 * capped_min_pts))
    capped_num_initial_clusters = min(
        num_initial_clusters, max_initial_clusters_for_budget
    )
    return capped_num_initial_clusters, capped_min_pts


def conductExperiment(
    # Problem-specific
    json_fpath: Path,
    DOMAIN: list[float],
    F_DIM: int,
    DIM: int,
    N_TRAIN: int,
    N_TEST: int,
    # Training-specific
    f_fn,
    g_fn,
    TOL: float,
    ERROR_EVAL_MODE: SurrogateMode,
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"],
    INITIAL_CLUSTERING_METHOD: Literal["kmeans", "knndbscan", "uniform_grid"],
    NUM_INITIAL_CLUSTERS: int,
    MIN_PTS_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int = 100_000,
    MAX_LEVELS: int = 1000,
    USE_WORK_STEALING: bool = True,
    USE_V2=True,
    # Prediction-specific
    K_NEIGHBOURS: int = 1,
    # Post-processing-specific
    CENTROIDS_PLOT_IN_LOG_SCALE: bool = True,
    SAVE_ALL_GEN_TRAINING_DATA: bool = False,
    SAVE_FINAL_CENTROIDS: bool = False,
):
    if json_fpath.exists():
        if _rank == 0:
            print(f"Experiment already exists at {json_fpath}. Skipping...")
        return
    json_fpath.parent.mkdir(parents=True, exist_ok=True)

    if _rank == 0:
        xp.random.seed(SEED)
        X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
        X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))
    else:
        X_train = None
        X_test = None

    surrogate = AFINNISurrogate()
    error_tag = ""
    FAILED = False
    _TRAINED, _PREDICTED = False, False
    try:
        model_save_path = json_fpath.with_name(json_fpath.stem + "_surrogate_model.npz")
        if _rank == 0:
            print("\n--- TRAINING ---")
        _start_train_time = time.perf_counter()

        # All ranks participate in training
        surrogate.train(
            X_train=X_train,
            USE_V2=USE_V2,
            f_fn=f_fn,
            g_fn=g_fn,
            mode=ERROR_EVAL_MODE,
            save_model=None,
            load_if_exists=True,
            TOL=TOL,
            ERROR_METRIC=ERROR_METRIC,
            INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
            NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            USE_WORK_STEALING=USE_WORK_STEALING,
            MAX_LEVELS=MAX_LEVELS,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            LOG_MEM_STATS=True,
            PROFILE_TIMING=True,
            STORE_AFFINE_FIT=True,
            SAVE_ALL_GEN_TRAINING_DATA=SAVE_ALL_GEN_TRAINING_DATA,
        )
        _train_time = time.perf_counter() - _start_train_time
        _TRAINED = True

        if _rank == 0:
            print("\n--- PREDICTION ---")

        # Save centroid data if user requests it
        if _rank == 0 and SAVE_FINAL_CENTROIDS is True:
            xp.save(f"centroids_tol_{TOL}_d{DIM}.npy", surrogate.X_src)

        _start_predict_time = time.perf_counter()

        # FIX: ALL ranks MUST participate in predict() because of comm.scatter/gather
        preds, ung_mask = surrogate.predict(
            query_points=X_test, k_nn=K_NEIGHBOURS, strict_trust_region=False
        )
        _predict_time = time.perf_counter() - _start_predict_time
        _PREDICTED = True

    except Exception as e:
        error_tag = f"Experiment failed with exception: {e} | TRAINED={_TRAINED}, PREDICTED={_PREDICTED}"
        msg = f"R{_rank:02}: {error_tag}"
        print(msg)
        logger.error(msg, exc_info=True)
        if _rank == 0:
            error_path = json_fpath.with_suffix(".error.txt")
            with open(error_path, "w") as ef:
                ef.write(error_tag)
        raise e

    # Workers return HERE, after completing the MPI-collective
    if _rank != 0:
        return

    # Compute ground truth and error metrics (Rank 0 only)
    ground_truth = f_fn(X_test)
    error_metrics = compute_error_metrics(
        tol=TOL,
        preds=preds,
        gt=ground_truth,
        ug_mask=ung_mask,
    )

    # Save JSON result
    json_data = {
        "IS_CUPY_BACKEND": IS_CUPY_BACKEND,
        "mpi_size": _size,
        "dim": DIM,
        "f_dim": F_DIM,
        "tol": TOL,
        "N_0": N_TRAIN,
        "N_test": N_TEST,
        "N_c": surrogate.N_src,
        "Train_time_s": _train_time,
        "Pred_time_s": _predict_time,
        "N_levels_max": surrogate.profiling_stats.get("num_levels", None),
        "num_generated_pts": surrogate.profiling_stats.get("num_generated_pts", None),
        "N": surrogate.profiling_stats.get("total_f_evals", None),
        "error_metric": ERROR_METRIC,
        "error_eval_mode": (
            (ERROR_EVAL_MODE.value + "_v2") if USE_V2 else ERROR_EVAL_MODE.value
        ),
        "initial_clustering_method": INITIAL_CLUSTERING_METHOD,
        "num_initial_clusters": NUM_INITIAL_CLUSTERS,
        "min_pts_per_cluster": MIN_PTS_PER_CLUSTER,
        "use_work_stealing": USE_WORK_STEALING,
        "error_metrics": error_metrics,
    }
    with open(json_fpath, "w") as jf:
        json.dump(json_data, jf, indent=4)
    print(f"Experiment completed and saved to {json_fpath}")

    plot_save_path = json_fpath.with_name(json_fpath.stem + "_error_histogram.png")
    plot_error_histogram(
        tol=TOL,
        ground_truth=ground_truth,
        predictions=preds,
        img_fpath=plot_save_path,
    )

    if DIM == 2 and surrogate.N_src <= 10_000:
        plot_save_path = json_fpath.with_name(json_fpath.stem + "_centroid_plot.png")
        plot_adaptive_clustering_results(
            centroids=_to_host(surrogate.X_src),
            Q=_to_host(surrogate.Qs_src),
            radii=_to_host(surrogate.trust_radii_src),
            domain=DOMAIN,  # type: ignore
            f_fn=f_fn,
            img_fpath=plot_save_path,
            plot_in_log=CENTROIDS_PLOT_IN_LOG_SCALE,
        )


def create_summary_dataframe(exp_dir: Path):
    if _rank != 0:
        # Only rank 0 creates the summary dataframe. Other ranks
        # workers return immediately after MPI-collective in conductExperiment().
        return

    import pandas as pd

    json_files = list(exp_dir.glob("*.json"))
    key_list = [
        "dim",
        "tol",
        "N_0",
        "N_test",
        "N",
        "N_c",
        "error_eval_mode",
        "error_metric",
        "initial_clustering_method",
        "Train_time_s",
        "Pred_time_s",
    ]
    data_dict = defaultdict(list)
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        for key in key_list:
            data_dict[key].append(data[key])

        data_dict["e"].append(data["error_metrics"]["all"]["l2_rel_error"])
        data_dict["e_in"].append(data["error_metrics"]["in"]["l2_rel_error"])
        data_dict["p_out"].append(data["error_metrics"]["perct_out"])

    df = pd.DataFrame(data_dict)
    df = df.sort_values(
        by=["dim", "tol", "error_eval_mode"], ascending=[True, False, True]
    )
    order = [
        "dim",
        "tol",
        "error_eval_mode",
        "error_metric",
        "initial_clustering_method",
        "N_0",
        "N_test",
        "N",
        "N_c",
        "e",
        "e_in",
        "p_out",
        "Train_time_s",
        "Pred_time_s",
    ]
    df = df[order]
    df.to_csv(exp_dir / "_summary.csv", index=False)


def anisotropic_quadratic_experiment():
    from src.problems.anisotropic_quadratic import aniso_quadratic, aniso_quadratic_grad

    EXP_DIR = SC26_DIR / "anisotropic_quadratic"

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    DIMs = [10]
    ERROR_EVAL_MODES = [SurrogateMode.AFFINE_COV_NO_HESS]

    combos = list(product(DIMs, ERROR_EVAL_MODES))
    for DIM, ERROR_EVAL_MODE in tqdm(combos, desc="Experiments"):

        # TOLs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
        #         0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11,
        #         0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        TOLs = [0.458, 0.381]
        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)

        for TOL in TOLs:
            if DIM == 10 and TOL <= 0.005:
                print("This is throwing OOM errors. Skipping for now.")
                continue
            json_fpath = (
                EXP_DIR
                / f"mpi_sz{_size}_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_{ERROR_EVAL_MODE.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
            )
            conductExperiment(
                # Problem-specific
                json_fpath=json_fpath,
                DOMAIN=[-3.0, 3.0],
                F_DIM=F_DIM,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                # Training-specific
                f_fn=aniso_quadratic,
                g_fn=aniso_quadratic_grad,
                TOL=TOL,
                ERROR_EVAL_MODE=ERROR_EVAL_MODE,
                ERROR_METRIC=ERROR_METRIC,
                INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                USE_WORK_STEALING=USE_WORK_STEALING,
                USE_V2=USE_V2,
            )

            create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def anisotropic_gaussian_experiment():
    from src.problems.anisotropic_gaussian import aniso_gaussian, aniso_gaussian_grad

    EXP_DIR = SC26_DIR / "anisotropic_gaussian"

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    DIMs = [5]
    ERROR_EVAL_MODES = [SurrogateMode.AFFINE_COV_NO_HESS]

    combos = list(product(DIMs, ERROR_EVAL_MODES))
    for DIM, ERROR_EVAL_MODE in tqdm(combos, desc="Experiments"):

        # TOLs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
        #         0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11,
        #         0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        TOLs = [0.0907, 0.00307]
        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)

        for TOL in TOLs:
            json_fpath = (
                EXP_DIR
                / f"mpi_sz{_size}_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_{ERROR_EVAL_MODE.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
            )
            conductExperiment(
                # Problem-specific
                json_fpath=json_fpath,
                DOMAIN=[-3.0, 3.0],
                F_DIM=F_DIM,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                # Training-specific
                f_fn=aniso_gaussian,
                g_fn=aniso_gaussian_grad,
                TOL=TOL,
                ERROR_EVAL_MODE=ERROR_EVAL_MODE,
                ERROR_METRIC=ERROR_METRIC,
                INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                USE_WORK_STEALING=USE_WORK_STEALING,
                USE_V2=USE_V2,
                CENTROIDS_PLOT_IN_LOG_SCALE=False,
            )

            create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def rosenbrock_experiment():
    from functools import partial

    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    EXP_DIR = SC26_DIR / "rosenbrock"

    ROSENBROCK_A = 100.0
    rosenbrock = partial(rosenbrock, a=ROSENBROCK_A)
    rosenbrock_grad = partial(rosenbrock_grad, a=ROSENBROCK_A)

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    DIMs = [5]
    ERROR_EVAL_MODES = [
        SurrogateMode.AFFINE_COV_HESS,
        SurrogateMode.AFFINE_COV_NO_HESS,
        SurrogateMode.NEAREST_NEIGHBOR,
    ]

    combos = list(product(DIMs, ERROR_EVAL_MODES))
    for DIM, ERROR_EVAL_MODE in tqdm(combos, desc="Experiments"):
        if ERROR_EVAL_MODE == SurrogateMode.AFFINE_COV_HESS:
            TOLs = [0.05]
        else:
            # TODO for each ERROR_EVAL_MODE
            print(
                f"Skipping ERROR_EVAL_MODE={ERROR_EVAL_MODE.value} for now. TODO: Add other TOLs for this mode."
            )
            continue

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)

        for TOL in TOLs:
            if (DIM > 2 and TOL < 0.01) or (
                DIM > 2 and TOL < 0.05 and ERROR_METRIC == "Linf-local"
            ):
                print("skip for now")
                continue

            json_fpath = (
                EXP_DIR
                / f"mpi_sz{_size}_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_{ERROR_EVAL_MODE.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
            )
            conductExperiment(
                # Problem-specific
                json_fpath=json_fpath,
                DOMAIN=[-2.0, 3.0],
                F_DIM=F_DIM,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                # Training-specific
                f_fn=rosenbrock,
                g_fn=rosenbrock_grad,
                TOL=TOL,
                ERROR_EVAL_MODE=ERROR_EVAL_MODE,
                ERROR_METRIC=ERROR_METRIC,  # type: ignore
                INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                USE_WORK_STEALING=USE_WORK_STEALING,
                USE_V2=USE_V2,
                SAVE_ALL_GEN_TRAINING_DATA=True,
                SAVE_FINAL_CENTROIDS=True,
            )

            create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def lorenz96_experiment(tf):

    # from src.problems.lorenz96_flowmap import lorenz96_flowmap, lorenz96_flowmap_grad
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched

    # Lorenz flowmap parameters set here
    FORCING = 1
    FINAL_TIME = tf

    # Custom oracles to return only the first component
    def lorenz(x):
        if IS_CUPY_BACKEND:
            import cupy
            import jax.numpy as jnp

            # x will be type CuPy. Convert to jax type.
            x_jax = jnp.from_dlpack(x)
            f = lorenz96_flowmap_jax_batched(x_jax, FORCING, FINAL_TIME)
            # output is jax type; convert to CuPy.
            return cupy.from_dlpack(f[:, :, 0])
        else:
            f = lorenz96_flowmap_jax_batched(x, FORCING, FINAL_TIME)
            return f

    # def lorenz_grad(x):
    #     g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
    #     N, d, _ = g.shape
    #     return g[:, 0, :].reshape((N, 1, d))

    EXP_DIR = SC26_DIR / f"lorenz96_dt_{tf}"
    F_DIM = 1
    ERROR_METRIC = "Linf-global"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    DIMs = [5]
    ERROR_EVAL_MODES = [SurrogateMode.AFFINE_COV_HESS]

    combos = list(product(DIMs, ERROR_EVAL_MODES))
    for DIM, ERROR_EVAL_MODE in tqdm(combos, desc="Experiments"):

        # TOLs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
        #         0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11,
        #         0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        TOLs = [0.1, 0.05]
        N_TRAIN = min(get_N_TRAIN(DIM), 1000000)
        N_TEST = N_TRAIN // 2
        # NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
        NUM_INITIAL_CLUSTERS = 144
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)

        for TOL in TOLs:
            json_fpath = (
                EXP_DIR
                / f"mpi_sz{_size}_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_{ERROR_EVAL_MODE.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
            )
            conductExperiment(
                # Problem-specific
                json_fpath=json_fpath,
                DOMAIN=[-5.0, 5.0],
                F_DIM=F_DIM,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                # Training-specific
                f_fn=lorenz,
                g_fn=None,
                TOL=TOL,
                ERROR_EVAL_MODE=ERROR_EVAL_MODE,
                ERROR_METRIC=ERROR_METRIC,
                INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                USE_WORK_STEALING=USE_WORK_STEALING,
                USE_V2=USE_V2,
            )

            create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def lorenz96_dimension_scaling_experiment():
    from src.problems.lorenz96_flowmap import lorenz96_flowmap, lorenz96_flowmap_grad

    # Lorenz flowmap parameters set here
    FORCING = 1
    FINAL_TIME = 0.1

    # Custom oracles to return only the first component
    def lorenz(x):
        f = lorenz96_flowmap(x, FORCING, FINAL_TIME)
        N, _ = f.shape
        return f[:, 0].reshape((N, 1))

    def lorenz_grad(x):
        g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
        N, d, _ = g.shape
        return g[:, 0, :].reshape((N, 1, d))

    EXP_DIR = SC26_DIR / "dim_scaling_lorenz96"
    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    DIMs = [5, 10, 20, 40, 80]
    ERROR_EVAL_MODES = [SurrogateMode.AFFINE_COV_HESS]

    combos = list(product(DIMs, ERROR_EVAL_MODES))
    for DIM, ERROR_EVAL_MODE in tqdm(combos, desc="Experiments"):
        TOLs = [0.1]

        N_TRAIN = min(get_N_TRAIN(DIM), 1_000_000)
        N_TEST = N_TRAIN // 2
        raw_num_initial_clusters = get_NUM_INITIAL_CLUSTERS(DIM)
        raw_min_pts_per_cluster = get_MIN_PTS_PER_CLUSTER(DIM)
        NUM_INITIAL_CLUSTERS, MIN_PTS_PER_CLUSTER = cap_initial_augmentation_budget(
            N_TRAIN,
            raw_num_initial_clusters,
            raw_min_pts_per_cluster,
        )

        if _rank == 0 and (
            NUM_INITIAL_CLUSTERS != raw_num_initial_clusters
            or MIN_PTS_PER_CLUSTER != raw_min_pts_per_cluster
        ):
            print(
                "Adjusted level-0 heuristics for capped training budget: "
                f"dim={DIM}, N_TRAIN={N_TRAIN:,}, "
                f"NUM_INITIAL_CLUSTERS={raw_num_initial_clusters:,}->{NUM_INITIAL_CLUSTERS:,}, "
                f"MIN_PTS_PER_CLUSTER={raw_min_pts_per_cluster:,}->{MIN_PTS_PER_CLUSTER:,}"
            )

        for TOL in TOLs:
            json_fpath = (
                EXP_DIR
                / f"mpi_sz{_size}_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_{ERROR_EVAL_MODE.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
            )
            conductExperiment(
                # Problem-specific
                json_fpath=json_fpath,
                DOMAIN=[-5.0, 5.0],
                F_DIM=F_DIM,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                # Training-specific
                f_fn=lorenz,
                g_fn=lorenz_grad,
                TOL=TOL,
                ERROR_EVAL_MODE=ERROR_EVAL_MODE,
                ERROR_METRIC=ERROR_METRIC,
                INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                USE_WORK_STEALING=USE_WORK_STEALING,
                USE_V2=USE_V2,
            )
            create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def indefinite_quadratic_experiment():

    d = 5
    randmat = xp.random.uniform(-5, 5, (d, d))
    H = 0.5 * (randmat + randmat.T)

    # avec = xp.array([1, -10, 50, -100, 200])
    # The Hessian matrix will be diagonal.
    def indefinite_quadratic(x):
        import numpy as np

        return np.einsum("ni,ij,nj->n", x, H, x)[:, np.newaxis]

    def indefinite_quadratic_grad(x):
        return 2 * x @ H

    EXP_DIR = SC26_DIR / "indef_quad"
    F_DIM = 1
    ERROR_METRIC = "Linf-global"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = False

    DIMs = [d]
    ERROR_EVAL_MODES = [
        SurrogateMode.AFFINE_COV_HESS,  # cluster via X^T @ H @ X
        SurrogateMode.AFFINE_COV_NO_HESS,  # cluster via X^T @ I @ X
        SurrogateMode.TAYLOR_COV_HESS,
        SurrogateMode.TAYLOR_LSQ_HESS,  # cluster via X^T @ H @ X, with very accurate H
    ]

    combos = list(product(DIMs, ERROR_EVAL_MODES))
    for DIM, ERROR_EVAL_MODE in tqdm(combos, desc="Experiments"):
        TOLs = [0.1, 0.05, 0.02]

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)

        for TOL in TOLs:
            json_fpath = (
                EXP_DIR
                / f"mpi_sz{_size}_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_{ERROR_EVAL_MODE.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
            )
            conductExperiment(
                # Problem-specific
                json_fpath=json_fpath,
                DOMAIN=[-2.0, 3.0],
                F_DIM=F_DIM,
                DIM=DIM,
                N_TRAIN=N_TRAIN,
                N_TEST=N_TEST,
                # Training-specific
                f_fn=indefinite_quadratic,
                g_fn=indefinite_quadratic_grad,
                TOL=TOL,
                ERROR_EVAL_MODE=ERROR_EVAL_MODE,
                ERROR_METRIC=ERROR_METRIC,
                INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                USE_WORK_STEALING=USE_WORK_STEALING,
                USE_V2=USE_V2,
            )

            create_summary_dataframe(EXP_DIR)

    return EXP_DIR


if __name__ == "__main__":
    import logging

    from src.logging_utils import setup_logging

    setup_logging(console_level=logging.DEBUG)
    # tf = float(sys.argv[1])

    # dir1 = anisotropic_quadratic_experiment()
    # dir2 = anisotropic_gaussian_experiment()
    # dir3 = rosenbrock_experiment()
    # dir1 = lorenz96_experiment(tf=0.15)
    dir2 = lorenz96_dimension_scaling_experiment()
    # dir5 = rosenbrock_diff_error_metrics_experiment()
    # dir6 = indefinite_quadratic_experiment()

    # _dirs = [dir1, dir2, dir3, dir4, dir5]
    # for d in _dirs:
    #     if _rank == 0:
    #         print(f"\nCreating summary dataframe for {d}...")
    #         create_summary_dataframe(d)
