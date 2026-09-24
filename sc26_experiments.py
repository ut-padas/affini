import json
import logging
import time
import jax
import cupy as cp
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal
import argparse
from tqdm import tqdm

from src import IS_CUPY_BACKEND, RESULTS_DIR, _get_mpi_rank, _get_mpi_size, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.heuristic_parameters import (
    get_MIN_PTS_PER_CLUSTER,
    get_N_TRAIN,
    get_NUM_INITIAL_CLUSTERS,
)
from src.plotter_utils import plot_adaptive_clustering_results, plot_error_histogram, plot_cluster_projections_for_wave
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


def conduct_experiment_given_data(
    # Problem-specific
    json_fpath: Path,
    DOMAIN: list[float],
    F_DIM: int,
    DIM: int,
    N_TRAIN: int,
    N_TEST: int,
    # Training-specific
    X_train,
    X_test,
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
    Y_train=None,
    HESSIAN=None,
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

    surrogate = AFINNISurrogate()
    error_tag = ""
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
            Y_train=None,
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
            HESSIAN=HESSIAN,
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
            query_points=X_test, k_nn=K_NEIGHBOURS, strict_trust_region=False, coarse_factor=DIM**2
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

    if DIM == 2 and surrogate.N_src <= 100_000:
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

    
    if json_fpath.parent.name == "wave":
        plot_save_path = json_fpath.with_name(json_fpath.stem + "_projection_plot.png")
        plot_cluster_projections_for_wave(_to_host(surrogate.X_src), plot_save_path, DIM)




def conduct_experiment_generate_uniform_data(
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
    HESSIAN=None,
    # Prediction-specific
    K_NEIGHBOURS: int = 1,
    # Post-processing-specific
    CENTROIDS_PLOT_IN_LOG_SCALE: bool = False,
    SAVE_ALL_GEN_TRAINING_DATA: bool = False,
    SAVE_FINAL_CENTROIDS: bool = False,
):

    if _rank == 0:
        xp.random.seed(SEED)
        X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
        X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))
    else:
        X_train = None
        X_test = None

    conduct_experiment_given_data(
        json_fpath,
        DOMAIN,
        F_DIM,
        DIM,
        N_TRAIN,
        N_TEST,
        X_train,
        X_test,
        f_fn,
        g_fn,
        TOL,
        ERROR_EVAL_MODE,
        ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER,
        MAX_CLUSTERS_EST,
        MAX_LEVELS,
        USE_WORK_STEALING,
        USE_V2,
        None,
        HESSIAN,
        K_NEIGHBOURS,
        CENTROIDS_PLOT_IN_LOG_SCALE,
        SAVE_ALL_GEN_TRAINING_DATA,
        SAVE_FINAL_CENTROIDS
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


"""
This is deprecated at the moment (incomplete).
"""
class NelsonBatchSolver:

    def __init__(self, env):

        Exception("ERROR: Nelson batch solver is not implemented yet.")

        # First, import stuff
        import sys
        from pathlib import Path
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root / "chemistry-benchmark-surrogates" / "src"))
        from PythonODE import parser # type: ignore

        # solver settings
        atol = 1e-16
        rtol = 1e-4

        # Parse the KIDA files
        print("Pasrsing gas network.\n")
        net = parser.KidaNetwork(grains = False)
        net.load_from_disk("chemistry-benchmark-surrogates/networks/nelson/gas_reactions.in")
        dropped = net.drop_passive_species()
        print(f"Dropped passive species: {dropped}")
        print("Constructing matrices A and B.")
        self.A, self.B = net.get_tensors(env)


    def batch_solve(self, X, dt):

        import sys
        from pathlib import Path
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root / "chemistry-benchmark-surrogates" / "src"))
        from PythonODE import parser # type: ignore

        atol = 1e-16
        rtol = 1e-4

        Y = xp.zeros_like(X)
        for j, state in enumerate(X):
            q = parser.QuadraticSolverPoint()
            [t,y] = q.solve(self.A, self.B, [0, dt], state, atol=atol, rtol=rtol)
            Y[j] = y[:,-1]

        return Y



### --- BEGIN EXPERIMENT FUNCTIONS --- ###

def anisotropic_quadratic_experiment(dim: int, tol: float, error_eval_mode: SurrogateMode):
    from src.problems.anisotropic_quadratic import aniso_quadratic, aniso_quadratic_grad, aniso_quadratic_hess

    EXP_DIR = SC26_DIR / "anisotropic_quadratic"

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    if dim >= 10 and tol <= 0.005:
        print("This is throwing OOM errors. Skipping for now.")
        return

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )
    conduct_experiment_generate_uniform_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-3.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        # Training-specific
        f_fn=aniso_quadratic,
        g_fn=aniso_quadratic_grad,
        HESSIAN=aniso_quadratic_hess,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
    )

    create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def anisotropic_quadratic_modified_sampling(dim: int, tol: float, error_eval_mode: SurrogateMode):
    from src.problems.anisotropic_quadratic import aniso_quadratic, aniso_quadratic_grad, _get_params
    EXP_DIR = SC26_DIR / "anisotropic_quadratic_modified_sampling"

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    if dim >= 10 and tol <= 0.005:
        print("This is throwing OOM errors. Skipping for now.")
        return

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )

    # Sample initial data from modified distribution
    decay_exp = 3.0; beta = 1.0
    w, H = _get_params(dim, xp.float64, decay_exp)
    mean = -0.5 * xp.linalg.solve(H, w)
    cov = 0.5 * xp.linalg.inv(H)
    X_train = xp.random.multivariate_normal(mean, cov, size=N_TRAIN)
    X_test = xp.random.multivariate_normal(mean, cov, size=N_TEST)
    
    conduct_experiment_given_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-3.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        X_train=X_train,
        X_test=X_test,
        # Training-specific
        f_fn=aniso_quadratic,
        g_fn=aniso_quadratic_grad,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
    )

    create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def anisotropic_gaussian_experiment(dim: int, tol: float, error_eval_mode: SurrogateMode):
    from src.problems.anisotropic_gaussian import aniso_gaussian, aniso_gaussian_grad, aniso_gaussian_hess

    EXP_DIR = SC26_DIR / "anisotropic_gaussian"

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    # NUM_INITIAL_CLUSTERS = 2
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )
    conduct_experiment_generate_uniform_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-3.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        # Training-specific
        f_fn=aniso_gaussian,
        g_fn=aniso_gaussian_grad,
        HESSIAN=aniso_gaussian_hess,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
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


def anisotropic_gaussian_modified_sampling(dim: int, tol: float, error_eval_mode: SurrogateMode):
    from src.problems.anisotropic_gaussian import aniso_gaussian, aniso_gaussian_grad, _get_random_rotation, _get_precision_eigenvalues

    EXP_DIR = SC26_DIR / "anisotropic_gaussian_modified_sampling"

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )

    # Sample initial data from modified distribution
    decay_exp = 3
    beta = 1
    Q = _get_random_rotation(dim, xp.float64)
    Lambda = xp.diag(_get_precision_eigenvalues(dim, xp.float64, decay_exp))
    Ainv = Q @ (Lambda ** -1.0) @ Q.T
    X_train = xp.random.randn(N_TRAIN, dim) @ Ainv
    X_test = xp.random.randn(N_TEST, dim) @ Ainv

    conduct_experiment_given_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-3.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        X_train=X_train,
        X_test=X_test,
        # Training-specific
        f_fn=aniso_gaussian,
        g_fn=aniso_gaussian_grad,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
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



def rosenbrock_experiment(dim: int, tol: float, error_eval_mode: SurrogateMode):
    from functools import partial

    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad, rosenbrock_hess

    EXP_DIR = SC26_DIR / "rosenbrock"

    ROSENBROCK_A = 100.0
    rosenbrock = partial(rosenbrock, a=ROSENBROCK_A)
    rosenbrock_grad = partial(rosenbrock_grad, a=ROSENBROCK_A)
    rosenbrock_hess = partial(rosenbrock_hess, a=ROSENBROCK_A)

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )
    conduct_experiment_generate_uniform_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-2.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        # Training-specific
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        HESSIAN=rosenbrock_hess,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,  # type: ignore
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
        SAVE_ALL_GEN_TRAINING_DATA=False,
        SAVE_FINAL_CENTROIDS=True,
    )

    create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def rosenbrock_experiment_modified_sampling(dim: int, tol: float, error_eval_mode: SurrogateMode):
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

    if error_eval_mode != SurrogateMode.AFFINE_COV_HESS:
        # TODO for each ERROR_EVAL_MODE
        print(
            f"Skipping ERROR_EVAL_MODE={error_eval_mode.value} for now. TODO: Add other TOLs for this mode."
        )
        return

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )

    # Generate Rosenbrock distribution data
    if dim > 2:
        print("Not implemented yet for dimensions other than d=2.")
        return
    a = 1.0; b = 5.0; mu = 1.0;
    x1_train = xp.random.randn(N_TRAIN) / (2*a) + mu
    x2_train = xp.random.randn(N_TRAIN) / (2*b) + (x1_train**2)
    x1_test = xp.random.randn(N_TEST) / (2*a) + mu
    x2_test = xp.random.randn(N_TEST) / (2*b) + (x1_train**2)
    X_train = xp.column_stack((x1_train, x2_train))
    X_test = xp.column_stack((x1_test, x2_test))

    conduct_experiment_given_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-2.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        X_train=X_train,
        X_test=X_test,
        # Training-specific
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,  # type: ignore
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
        SAVE_ALL_GEN_TRAINING_DATA=False,
        SAVE_FINAL_CENTROIDS=True,
    )

    create_summary_dataframe(EXP_DIR)

    return EXP_DIR


"""
Does the experiment for mapping to a single component of the output.
"""
def lorenz96_experiment(
        dim: int,
        tol: float,
        error_eval_mode: SurrogateMode,
        tf: float,
        forcing: float = 1.0,
        output_component: int = 0):
    from src.problems.lorenz96_flowmap import lorenz96_flowmap
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched

    # Custom oracles to return only the first component
    def lorenz(x):
        N, d = x.shape
        if IS_CUPY_BACKEND:
            # For now, just assume that X lives on a GPU.
            X_jax = jax.dlpack.from_dlpack(x)
            Y_jax = lorenz96_flowmap_jax_batched(X_jax, forcing, tf)
            return cp.from_dlpack(Y_jax)[:,output_component,0].reshape(N,1)
        else:
            f = lorenz96_flowmap(x, forcing, tf)[:,output_component]
            return f.reshape(N, 1)

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

    N_TRAIN = min(get_N_TRAIN(dim), 1000000)
    N_TEST = N_TRAIN // 2
    # NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
    NUM_INITIAL_CLUSTERS = 144
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )
    conduct_experiment_generate_uniform_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-5.0, 5.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        # Training-specific
        f_fn=lorenz,
        g_fn=None,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
    )

    create_summary_dataframe(EXP_DIR)

    return EXP_DIR



# TODO: probably can remove this
def lorenz96_dimension_scaling_experiment():
    from src.problems.lorenz96_flowmap import (
        lorenz96_flowmap_auto,
        lorenz96_flowmap_grad_auto,
    )

    # Lorenz flowmap parameters set here
    FORCING = 1
    FINAL_TIME = 0.1

    # Custom oracles to return only the first component
    # def lorenz(x):
    #     f = lorenz96_flowmap_auto(x, FORCING, FINAL_TIME)
    #     N, _ = f.shape
    #     return f[:, 0].reshape((N, 1))
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched
    def lorenz(X):
        N, d = X.shape
        # For now, just assume that X lives on a GPU.
        X_jax = jax.dlpack.from_dlpack(X)
        Y_jax = lorenz96_flowmap_jax_batched(X_jax, FORCING, FINAL_TIME)
        return cp.from_dlpack(Y_jax)[:,0,0].reshape(N,1)

    # def lorenz_grad(x):
    #     g = lorenz96_flowmap_grad_auto(x, FORCING, FINAL_TIME)
    #     N, d, _ = g.shape
    #     return g[:, 0, :].reshape((N, 1, d))

    EXP_DIR = SC26_DIR / "dim_scaling_lorenz96_dt0.1"
    F_DIM = 1
    ERROR_METRIC = "Linf-global"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    DIMs = [80]
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
            conduct_experiment_generate_uniform_data(
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



# TODO: what to do with this guy?
def lorenz96_low_rank_experiment(d, D, dt, tol):
    '''
    Goal: vary some of the variables and keep others fixed.
    Does HATANN behave like a low-rank function in this case? We hope so.
    Compare the behavior of the algorithm to running Lorenz for the low number of variables.
    We would like to run Lorenz d=5,10,20,40,80, but only varying 3 variables at a time.
    For now, fix the time step at dt=0.2.
    So in other words the experiments we will run will be:
    - d=5, D=10, dt=0.2, tol=varying
    - d=5, D=20, dt=0.2, tol=varying
    - d=5, D=40, dt=0.2, tol=varying
    '''

    print(f"Running simulation for d={d}, D={D}, dt={dt}, tol={tol}...")

    # Parameters for experiment
    dims2vary = xp.arange(d)
    dims2fix = xp.setdiff1d(xp.arange(D), dims2vary)
    d_fixed = D - len(dims2vary)
    l = -5.0; r = 5.0
    domain = [l, r]
    N_train = get_N_TRAIN(d) # use low-dim amount of data
    N_test = N_train // 2
    forcing = 1
    dt = 0.2

    # Generate initial data in the low rank subspace. Fix the variables def. by dims2fix
    fixed_row = xp.random.uniform(low=l, high=r, size=d_fixed)
    X_train = xp.empty((N_train, D))
    X_train[:,dims2vary] = xp.random.uniform(l, r, size=(N_train, d))
    X_train[:,dims2fix] = xp.outer(xp.ones(N_train), fixed_row)

    # In the same way, generate the testing data
    X_test = xp.empty((N_test, D))
    X_test[:,dims2vary] = xp.random.uniform(l, r, size=(N_test, d))
    X_test[:,dims2fix] = xp.outer(xp.ones(N_test), fixed_row)

    # Use the entire function
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched
    def f_fn(X):
        N, d = X.shape
        # For now, just assume that X lives on a GPU.
        X_jax = jax.dlpack.from_dlpack(X)
        Y_jax = lorenz96_flowmap_jax_batched(X_jax, forcing, dt)
        return cp.from_dlpack(Y_jax)[:,0,:]
        # return cp.from_dlpack(Y_jax)[:,0,0].reshape(N,1)

    # Put this into HATANN
    EXP_DIR = SC26_DIR / "lorenz_lowrankexp"
    json_fpath = (
        EXP_DIR / f"ambdim_{d}_fulldim{D}_dt{dt}_tol{tol}_ntrain{N_train}.json"
    )

    conduct_experiment_given_data(
        json_fpath = json_fpath,
        DOMAIN = domain,
        F_DIM = D,
        DIM = D,
        N_TRAIN = N_train,
        N_TEST = N_test,
        X_train = X_train,
        X_test = X_test,
        f_fn = f_fn,
        g_fn = None,
        TOL = tol,
        ERROR_EVAL_MODE = SurrogateMode.AFFINE_COV_HESS,
        ERROR_METRIC = 'Linf-global',
        INITIAL_CLUSTERING_METHOD = 'kmeans',
        NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(d),
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(d),
        USE_WORK_STEALING = False
    )

    create_summary_dataframe(EXP_DIR)
    return EXP_DIR



def lorenz_trajectories_experiment(d, tol, error_eval_mode, dt):
    '''
    Goal: use time step data as training and testing to do experiments.
    I.e. use manifolds of solutions as initial data.
    '''
    # Parameters for experiment
    l = -5.0; r = 5.0; domain = [l, r]
    N_test = 50_000
    forcing = 1

    # Define the function
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched
    def f_fn(X):
        N, d = X.shape
        # For now, just assume that X lives on a GPU.
        X_jax = jax.dlpack.from_dlpack(X)
        Y_jax = lorenz96_flowmap_jax_batched(X_jax, forcing, dt)
        return cp.from_dlpack(Y_jax)[:,0,:]
        # return cp.from_dlpack(Y_jax)[:,0,0].reshape(N,1)

    # Use this function to generate data.
    N_start_train = 20000; N_start_test = 5000; N_steps = 9;
    N_train = N_start_train * (N_steps + 1)
    N_test = N_start_test * (N_steps + 1)
    X_train = xp.zeros((N_start_train * (N_steps+1), d))
    X_train[:N_start_train,:] = xp.random.uniform(l, r, size=(N_start_train, d))
    X_test = xp.zeros((N_start_test * (N_steps+1), d))
    X_test[:N_start_test,:] = xp.random.uniform(l, r, size=(N_start_test, d))
    for j in range(1, N_steps + 1):
        X_train[(j*N_start_train):((j+1)*N_start_train)] = f_fn(X_train[((j-1)*N_start_train):(j*N_start_train)])
        X_test[(j*N_start_test):((j+1)*N_start_test)] = f_fn(X_train[((j-1)*N_start_test):(j*N_start_test)])

    # Put this into HATANN
    EXP_DIR = SC26_DIR / "lorenz_trajexp"
    json_fpath = (
        EXP_DIR / f"dim{d}_dt{dt}_tol{tol}_ntrain{N_start_train * (N_steps+1)}.json"
    )

    conduct_experiment_given_data(
        json_fpath = json_fpath,
        DOMAIN = domain,
        F_DIM = d,
        DIM = d,
        N_TRAIN = N_train,
        N_TEST = N_test,
        X_train = X_train,
        X_test = X_test,
        f_fn = f_fn,
        g_fn = None,
        TOL = tol,
        ERROR_EVAL_MODE = error_eval_mode,
        ERROR_METRIC = 'Linf-global',
        INITIAL_CLUSTERING_METHOD = 'kmeans',
        NUM_INITIAL_CLUSTERS = min(get_NUM_INITIAL_CLUSTERS(d), 1000),
        MIN_PTS_PER_CLUSTER = min(get_MIN_PTS_PER_CLUSTER(d), 500),
        USE_WORK_STEALING = False
    )

    create_summary_dataframe(EXP_DIR)
    return EXP_DIR



def nelson_trajectories_experiment():
    '''
    Goal: use time step data as training and testing to do experiments.
    '''
    # Parameters for experiment
    # Sample uniformly in powers, then take log, to cover all scales
    l = -20.0; r = 0.0; domain = [l, r]
    d = 14

    # --- Define the function ---

    # Nelson solver settings
    env = dict(
        T=10.0,             # Gas temperature 
        nH=2e4,             # Hydrogen number density
        Av=15.0,            # Visual Extinction
        uv_flux=1,          # Standard interstellar radiation field, G_0
        Tcap_2body=True,    # Set temperature to tmin or tmax if outside [tmin, tmax]
        Tgrain=10.0,        # Grain temperature
    )
    solver = NelsonBatchSolver(env)
    dt = 1000 * 365 * 24 * 3600 # 1000 years
    f_fn = lambda X: solver.batch_solve(X, dt)

    # Use this function to generate data.
    N_start_train = 25000; N_start_test = 12500; N_steps = 39;
    N_train = N_start_train * (N_steps + 1)
    N_test = N_start_test * (N_steps + 1)
    X_train = xp.zeros((N_start_train * (N_steps+1), d))
    powers = xp.random.uniform(l, r, size=(N_start_train, d))
    X_train[:N_start_train,:] = 10 ** powers
    X_test = xp.zeros((N_start_test * (N_steps+1), d))
    powers = xp.random.uniform(l, r, size=(N_start_test, d))
    X_test[:N_start_test,:] = 10 ** powers
    for j in range(1, N_steps + 1):
        X_train[(j*N_start_train):((j+1)*N_start_train)] = f_fn(X_train[((j-1)*N_start_train):(j*N_start_train)])
        X_test[(j*N_start_test):((j+1)*N_start_test)] = f_fn(X_train[((j-1)*N_start_test):(j*N_start_test)])

    # Put this into HATANN
    tol = 0.05
    EXP_DIR = SC26_DIR / "nelson_trajexp"
    json_fpath = (
        EXP_DIR / f"dim{d}_tol{tol}_ntrain{N_train}.json"
    )

    conduct_experiment_given_data(
        json_fpath = json_fpath,
        DOMAIN = domain,
        F_DIM = d,
        DIM = d,
        N_TRAIN = N_train,
        N_TEST = N_test,
        X_train = X_train,
        X_test = X_test,
        f_fn = f_fn,
        g_fn = None,
        TOL = tol,
        ERROR_EVAL_MODE = SurrogateMode.AFFINE_COV_HESS,
        ERROR_METRIC = 'Linf-global',
        INITIAL_CLUSTERING_METHOD = 'kmeans',
        NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(d),
        MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(d),
        USE_WORK_STEALING = False
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
            conduct_experiment_generate_uniform_data(
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


def wave_experiment(dim: int, tol: float, error_eval_mode: SurrogateMode):

    from src.problems.wave import wave, wave_grad, wave_hess

    EXP_DIR = SC26_DIR / "wave"

    wave_fn = lambda x: wave(x)
    wave_fn_grad = lambda x: wave_grad(x)
    wave_fn_hess = lambda x: wave_hess(x)

    F_DIM = 1
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = get_N_TRAIN(dim)
    N_TEST = N_TRAIN // 2
    # NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    NUM_INITIAL_CLUSTERS = 1
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )
    conduct_experiment_generate_uniform_data(
        # Problem-specific
        json_fpath=json_fpath,
        DOMAIN=[-2.0, 3.0],
        F_DIM=F_DIM,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        # Training-specific
        f_fn=wave_fn,
        g_fn=wave_fn_grad,
        HESSIAN=wave_fn_hess,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,  # type: ignore
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
        SAVE_ALL_GEN_TRAINING_DATA=False,
        SAVE_FINAL_CENTROIDS=False,
    )

    create_summary_dataframe(EXP_DIR)

    return EXP_DIR



def setup_argparse():
    parser = argparse.ArgumentParser()

    # Experiment basics
    parser.add_argument("--experiment", help="The name of the experiment to run.", type=str, default="rosenbrock_dim_scaling")
    parser.add_argument("--tols", help="The tolerances to use for the experiments.", type=float, nargs='+', default=[0.1, 0.05])
    parser.add_argument("--dims", help="The dims to use for the experiments.", type=int, nargs='+', default=[2, 5])

    # Other optional stuff for specific experiments (i.e. dt for Lorenz or other flow maps)
    parser.add_argument("--dt", help="The time horizon to use for the flow map experiment. \
                        Only does something if a flow map experiment is specified.", type=float, default=0.1)
    parser.add_argument("--error_eval_mode", help="The error eval mode to use.", type=str, default="affine_cov_hess")
    parser.add_argument("--sampling", help="The sampling strategy to use for initial points.", type=str, default="uniform")

    args = parser.parse_args()

    # Check if specified strings are valid
    possible_experiments = [
        "anisotropic_quadratic",
        "anisotropic_gaussian",
        "rosenbrock",
        "lorenz96",
        "wave"
    ]
    possible_error_eval_modes = [
        "affine_cov_hess",
        "affine_cov_no_hess",
        "nearest_neighbor",
        "quadratic",
        "taylor_cov_hess",
        "taylor_cov_no_hess",
        "taylor_lsq_hess",
        "taylor_exact_hess",
        "affine_exact_hess"
    ]
    possible_sampling = [
        "uniform",
        "modified"
    ]
    if args.experiment not in possible_experiments:
        Exception(f"ERROR: experiment specified is not one of the implemented test functions.\nPossible options are: {possible_experiments}")
    if args.error_eval_mode not in possible_error_eval_modes:
        Exception(f"ERROR: error eval. mode specified is not implemented.\nPossible options are: {possible_error_eval_modes}")
    if args.sampling not in possible_sampling:
        Exception(f"ERROR: sampling strategy specified is not implemented. Possible options are: {possible_sampling}")

    return args


def run_experiment_from_args(args):

    # Grab values from dictionary
    tolvec = args.tols
    dvec = args.dims
    function_name = args.experiment
    sampling = args.sampling

    # Pick correct enum for error_eval_mode
    enum_map = {
        "affine_cov_hess": SurrogateMode.AFFINE_COV_HESS,
        "affine_exact_hess": SurrogateMode.AFFINE_EXACT_HESS,
        "affine_cov_no_hess": SurrogateMode.AFFINE_COV_NO_HESS,
        "nearest_neighbor": SurrogateMode.NEAREST_NEIGHBOR,
        "quadratic": SurrogateMode.QUADRATIC,
        "taylor_cov_hess": SurrogateMode.TAYLOR_COV_HESS,
        "taylor_exact_hess": SurrogateMode.TAYLOR_EXACT_HESS,
        "taylor_cov_no_hess": SurrogateMode.TAYLOR_COV_NO_HESS,
        "taylor_lsq_hess": SurrogateMode.TAYLOR_LSQ_HESS
    }
    error_eval_mode = enum_map[args.error_eval_mode]

    for tol in tolvec:
        for d in dvec:
            if function_name == "anisotropic_quadratic":
                if sampling == "uniform":
                    anisotropic_quadratic_experiment(d, tol, error_eval_mode)
                elif sampling == "modified":
                    anisotropic_quadratic_modified_sampling(d, tol, error_eval_mode)
            elif function_name == "anisotropic_gaussian":
                if sampling == "uniform":
                    anisotropic_gaussian_experiment(d, tol, error_eval_mode)
                elif sampling == "modified":
                    anisotropic_gaussian_modified_sampling(d, tol, error_eval_mode)
            elif function_name == "rosenbrock":
                if sampling == "uniform":
                    rosenbrock_experiment(d, tol, error_eval_mode)
                elif sampling == "modified":
                    rosenbrock_experiment_modified_sampling(d, tol, error_eval_mode)
            elif function_name == "lorenz96":
                if sampling == "uniform":
                    lorenz96_experiment(d, tol, error_eval_mode, args.dt)
                elif sampling == "modified":
                    lorenz_trajectories_experiment(d, tol, error_eval_mode, args.dt)
            elif function_name == "wave":
                wave_experiment(d, tol, error_eval_mode)
            else:
                print(f"Function {function_name} not recognized.")


if __name__ == "__main__":
    import logging

    from src.logging_utils import setup_logging

    setup_logging(console_level=logging.DEBUG)

    args = setup_argparse()
    run_experiment_from_args(args)

    # Nelson low-ambient dimension experiment 1 (enforced ambient dimension):
    # d=5; dt=0.2;
    # Dvec = xp.array([10, 20, 40])
    # tolvec = xp.array([0.02])
    # for D in Dvec:
    #     for tol in tolvec:
    #         lorenz96_low_rank_experiment(d, int(D), dt, float(tol))

    # Nelson low-ambient dimension experiment 2 (learned ambient dimension):
    # dvec = xp.array([5, 10, 20, 40, 80])
    # dt = 0.1
    # tolvec = xp.array([0.1])
    # for d in dvec:
    #     for tol in tolvec:
    #         lorenz_trajectories_experiment(int(d), dt, float(tol))
