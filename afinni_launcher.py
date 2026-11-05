import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import jax

from src import IS_CUPY_BACKEND, RESULTS_DIR, _get_mpi_rank, _get_mpi_size, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.heuristic_parameters import (
    get_MIN_PTS_PER_CLUSTER,
    get_N_TRAIN,
    get_NUM_INITIAL_CLUSTERS,
)
from src.plotter_utils import (
    plot_adaptive_clustering_results,
    plot_cluster_projections_for_wave,
    plot_error_histogram,
    plot_near_centers,
    plot_wave_2d_errors,
    plot_wave_line_errors,
)
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
    Q_reg=1e-6,
    HESSIAN=None,
    # Prediction-specific
    K_NEIGHBOURS: int = 1,
    COARSE_FACTOR: int = 10,
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
            Y_train=None,
            f_fn=f_fn,
            g_fn=g_fn,
            mode=ERROR_EVAL_MODE,
            ATOL=TOL,
            RTOL=TOL,
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
            Q_REG=Q_reg,
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
        # TODO: how to correctly choose coarse_factor? Another way to do nearest neighbor search?
        preds, ung_mask = surrogate.predict(
            query_points=X_test,
            k_nn=K_NEIGHBOURS,
            strict_trust_region=False,
            coarse_factor=COARSE_FACTOR,
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
        atol=TOL,
        rtol=TOL,
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
        rtol=TOL,
        ground_truth=ground_truth,
        predictions=preds,
        img_fpath=plot_save_path,
    )

    if DIM == 2 and surrogate.N_src <= 100_000:
        plot_save_path = json_fpath.with_name(json_fpath.stem + "_centroid_plot.png")
        plot_adaptive_clustering_results(
            centroids=_to_host(surrogate.X_src),
            Q=_to_host(surrogate.Qs_src),
            domain=DOMAIN,  # type: ignore
            f_fn=f_fn,
            img_fpath=plot_save_path,
            plot_in_log=CENTROIDS_PLOT_IN_LOG_SCALE,
        )

    if json_fpath.parent.name == "wave":

        proj_plot_save_path = json_fpath.with_name(
            json_fpath.stem + "_projection_plot.png"
        )
        plot_cluster_projections_for_wave(
            _to_host(surrogate.X_src), proj_plot_save_path, DIM
        )

        if DIM == 2:
            err_plot_save_path = json_fpath.with_name(
                json_fpath.stem + "_error_plot.png"
            )
            plot_wave_2d_errors(surrogate, DOMAIN, f_fn, err_plot_save_path)

        err_plot2_save_path = json_fpath.with_name(
            json_fpath.stem + "_projerror_plot.png"
        )
        plot_wave_line_errors(surrogate, DOMAIN, f_fn, err_plot2_save_path)

        center_plot_save_path = json_fpath.with_name(
            json_fpath.stem + "_centers_plot.png"
        )
        target = xp.array([3, -3.5, 2, 3, 4])
        plot_near_centers(surrogate, target, DOMAIN, f_fn, center_plot_save_path)

        print("Plots complete")


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
    Q_reg=1e-6,
    # Prediction-specific
    K_NEIGHBOURS: int = 1,
    COARSE_FACTOR: int = 10,
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
        Q_reg,
        HESSIAN,
        K_NEIGHBOURS,
        COARSE_FACTOR,
        CENTROIDS_PLOT_IN_LOG_SCALE,
        SAVE_ALL_GEN_TRAINING_DATA,
        SAVE_FINAL_CENTROIDS,
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


### --- PROBLEMS --- ###


@dataclass(frozen=True)
class Problem:
    """
    A test function and how its experiment is set up.

    Results go to ``SC26_DIR / dir_name``. Training and test points are drawn
    uniformly from ``domain`` (on rank 0), unless ``sample(dim, n_train, n_test)``
    is given, which returns ``(X_train, X_test)``.
    """

    dir_name: str
    f: Callable
    g: Callable | None = None
    hessian: Callable | None = None
    domain: tuple[float, float] = (-3.0, 3.0)
    error_metric: Literal["L2", "Linf-global", "Linf-local"] = "Linf-local"
    n_train: Callable[[int], int] = get_N_TRAIN  # default N_TRAIN for a dimension
    n_test: Callable[[int], int] = lambda n_train: n_train // 2
    sample: Callable | None = None
    centroids_plot_in_log_scale: bool = False
    save_final_centroids: bool = False
    only_modes: tuple[SurrogateMode, ...] | None = None  # other modes are skipped
    max_dim: int | None = None  # higher dimensions are skipped


def anisotropic_quadratic(args) -> Problem:
    from src.problems.anisotropic_quadratic_2 import (
        aniso_quadratic,
        aniso_quadratic_grad,
        aniso_quadratic_hess,
    )

    return Problem(
        "anisotropic_quadratic_2",
        aniso_quadratic,
        aniso_quadratic_grad,
        aniso_quadratic_hess,
    )


def anisotropic_quadratic_modified_sampling(args) -> Problem:
    from src.problems.anisotropic_quadratic import (
        _get_params,
        aniso_quadratic,
        aniso_quadratic_grad,
    )

    def sample(dim, n_train, n_test):
        # f(x) = beta + x^T H x has its minimum at the origin, so the Gaussian is
        # centred there with covariance shaped by the curvature (wider along flat
        # directions).
        H = xp.asarray(_get_params(dim, xp.float64, 3.0))  # helper returns NumPy
        mean = xp.zeros(dim, dtype=xp.float64)
        cov = 0.5 * xp.linalg.inv(H)
        X_train = xp.random.multivariate_normal(mean, cov, size=n_train)
        X_test = xp.random.multivariate_normal(mean, cov, size=n_test)
        return X_train, X_test

    return Problem(
        "anisotropic_quadratic_modified_sampling",
        aniso_quadratic,
        aniso_quadratic_grad,
        sample=sample,
        centroids_plot_in_log_scale=True,
    )


def anisotropic_gaussian(args) -> Problem:
    from src.problems.anisotropic_gaussian import (
        aniso_gaussian,
        aniso_gaussian_grad,
        aniso_gaussian_hess,
    )

    return Problem(
        "anisotropic_gaussian", aniso_gaussian, aniso_gaussian_grad, aniso_gaussian_hess
    )


def anisotropic_gaussian_modified_sampling(args) -> Problem:
    from src.problems.anisotropic_gaussian import (
        _get_precision_eigenvalues,
        _get_random_rotation,
        aniso_gaussian,
        aniso_gaussian_grad,
    )

    def sample(dim, n_train, n_test):
        Q = xp.asarray(_get_random_rotation(dim, xp.float64))  # helpers return NumPy
        Lambda = xp.diag(xp.asarray(_get_precision_eigenvalues(dim, xp.float64, 3)))
        Ainv = Q @ (Lambda**-1.0) @ Q.T
        X_train = xp.random.randn(n_train, dim) @ Ainv
        X_test = xp.random.randn(n_test, dim) @ Ainv
        return X_train, X_test

    return Problem(
        "anisotropic_gaussian_modified_sampling",
        aniso_gaussian,
        aniso_gaussian_grad,
        sample=sample,
    )


def rosenbrock(args) -> Problem:
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad, rosenbrock_hess

    return Problem(
        "rosenbrock",
        partial(rosenbrock, a=100.0),
        partial(rosenbrock_grad, a=100.0),
        partial(rosenbrock_hess, a=100.0),
        domain=(-2.0, 3.0),
        n_test=lambda n_train: 50_000,
    )


def rosenbrock_modified_sampling(args) -> Problem:
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    def sample(dim, n_train, n_test):
        # Points concentrated around the valley x2 = x1^2.
        a, b, mu = 1.0, 5.0, 1.0
        x1_train = xp.random.randn(n_train) / (2 * a) + mu
        x2_train = xp.random.randn(n_train) / (2 * b) + (x1_train**2)
        x1_test = xp.random.randn(n_test) / (2 * a) + mu
        x2_test = xp.random.randn(n_test) / (2 * b) + (x1_test**2)
        return (
            xp.column_stack((x1_train, x2_train)),
            xp.column_stack((x1_test, x2_test)),
        )

    return Problem(
        "rosenbrock",
        partial(rosenbrock, a=100.0),
        partial(rosenbrock_grad, a=100.0),
        domain=(-2.0, 3.0),
        sample=sample,
        centroids_plot_in_log_scale=True,
        save_final_centroids=True,
        only_modes=(SurrogateMode.AFFINE_COV_HESS,),
        max_dim=2,
    )


def lorenz96(args) -> Problem:
    """The first component of the Lorenz-96 flow map over time ``args.dt``."""
    from src.problems.lorenz96_flowmap import lorenz96_flowmap
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched

    forcing, tf, output_component = 1.0, args.dt, 0

    def f(x):
        N, d = x.shape
        if IS_CUPY_BACKEND:
            # For now, just assume that X lives on a GPU.
            X_jax = jax.dlpack.from_dlpack(x)
            Y_jax = lorenz96_flowmap_jax_batched(X_jax, forcing, tf)
            return cp.from_dlpack(Y_jax)[:, output_component, 0].reshape(N, 1)
        else:
            f = lorenz96_flowmap(x, forcing, tf)[:, output_component]
            return f.reshape(N, 1)

    return Problem(
        f"lorenz96_dt_{tf}",
        f,
        domain=(-5.0, 5.0),
        error_metric="Linf-global",
        n_train=lambda dim: min(get_N_TRAIN(dim), 1000000),
    )


def wave(args) -> Problem:
    from src.problems.wave import wave, wave_grad, wave_hess

    return Problem("wave", wave, wave_grad, wave_hess, domain=(-5.0, 5.0))


def degenerate_quadratic(args) -> Problem:
    from src.problems.degenerate_quadratic import (
        degenerate_quadratic,
        degenerate_quadratic_grad,
        degenerate_quadratic_hess,
    )

    return Problem(
        "degenerate_quadratic",
        degenerate_quadratic,
        degenerate_quadratic_grad,
        degenerate_quadratic_hess,
    )


def indef_degen_quadratic(args) -> Problem:
    from src.problems.indef_degen_quadratic import (
        indef_degen_quad,
        indef_degen_quad_grad,
        indef_degen_quad_hess,
    )

    return Problem(
        "indef_degen_quadratic",
        indef_degen_quad,
        indef_degen_quad_grad,
        indef_degen_quad_hess,
    )


# --experiment name -> the problem it runs. With --sampling modified, the
# problems in MODIFIED_SAMPLING use that variant instead (the others ignore
# --sampling), and lorenz96 runs lorenz_trajectories_experiment.
PROBLEMS = {
    "anisotropic_quadratic": anisotropic_quadratic,
    "anisotropic_gaussian": anisotropic_gaussian,
    "degenerate_quadratic": degenerate_quadratic,
    "indef_degen_quadratic": indef_degen_quadratic,
    "rosenbrock": rosenbrock,
    "lorenz96": lorenz96,
    "wave": wave,
}
MODIFIED_SAMPLING = {
    "anisotropic_quadratic": anisotropic_quadratic_modified_sampling,
    "anisotropic_gaussian": anisotropic_gaussian_modified_sampling,
    "rosenbrock": rosenbrock_modified_sampling,
}


def run_problem(
    problem: Problem,
    dim: int,
    tol: float,
    error_eval_mode: SurrogateMode,
    Q_reg: float,
    coarse_factor: int,
    N_train: int | None = None,
):
    """Train and test one surrogate for ``problem``; skip if the result exists."""
    if problem.only_modes is not None and error_eval_mode not in problem.only_modes:
        print(
            f"Skipping ERROR_EVAL_MODE={error_eval_mode.value}: this sampling is only "
            f"set up for {[m.value for m in problem.only_modes]}."
        )
        return None
    if problem.max_dim is not None and dim > problem.max_dim:
        print(f"Not implemented yet for dimensions above d={problem.max_dim}.")
        return None

    EXP_DIR = SC26_DIR / problem.dir_name
    ERROR_METRIC = problem.error_metric
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = problem.n_train(dim) if N_train is None else N_train
    N_TEST = problem.n_test(N_TRAIN)
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim)

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_tol{tol}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}.json"
    )
    settings = dict(
        json_fpath=json_fpath,
        DOMAIN=list(problem.domain),
        F_DIM=1,
        DIM=dim,
        N_TRAIN=N_TRAIN,
        N_TEST=N_TEST,
        f_fn=problem.f,
        g_fn=problem.g,
        HESSIAN=problem.hessian,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        USE_V2=USE_V2,
        Q_reg=Q_reg,
        COARSE_FACTOR=coarse_factor,
        CENTROIDS_PLOT_IN_LOG_SCALE=problem.centroids_plot_in_log_scale,
        SAVE_FINAL_CENTROIDS=problem.save_final_centroids,
    )
    if problem.sample is None:
        conduct_experiment_generate_uniform_data(**settings)
    else:
        X_train, X_test = problem.sample(dim, N_TRAIN, N_TEST)
        conduct_experiment_given_data(X_train=X_train, X_test=X_test, **settings)

    create_summary_dataframe(EXP_DIR)
    return EXP_DIR


def lorenz_trajectories_experiment(
    d, tol, error_eval_mode, dt, Q_reg, coarse_factor, N_train=None
):
    """
    Goal: use time step data as training and testing to do experiments.
    I.e. use manifolds of solutions as initial data.
    """
    # Parameters for experiment
    l = -5.0
    r = 5.0
    domain = [l, r]
    N_test = 50_000
    forcing = 1

    # Define the function
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched

    def f_fn(X):
        N, d = X.shape
        # For now, just assume that X lives on a GPU.
        X_jax = jax.dlpack.from_dlpack(X)
        Y_jax = lorenz96_flowmap_jax_batched(X_jax, forcing, dt)
        return cp.from_dlpack(Y_jax)[:, 0, :]
        # return cp.from_dlpack(Y_jax)[:,0,0].reshape(N,1)

    # Use this function to generate data.
    N_start_train = 20000
    N_start_test = 5000
    N_steps = 9
    if N_train is not None:
        # Total training points = N_start_train * (N_steps + 1)
        N_start_train = max(1, N_train // (N_steps + 1))
    N_train = N_start_train * (N_steps + 1)
    N_test = N_start_test * (N_steps + 1)
    X_train = xp.zeros((N_start_train * (N_steps + 1), d))
    X_train[:N_start_train, :] = xp.random.uniform(l, r, size=(N_start_train, d))
    X_test = xp.zeros((N_start_test * (N_steps + 1), d))
    X_test[:N_start_test, :] = xp.random.uniform(l, r, size=(N_start_test, d))
    for j in range(1, N_steps + 1):
        X_train[(j * N_start_train) : ((j + 1) * N_start_train)] = f_fn(
            X_train[((j - 1) * N_start_train) : (j * N_start_train)]
        )
        X_test[(j * N_start_test) : ((j + 1) * N_start_test)] = f_fn(
            X_test[((j - 1) * N_start_test) : (j * N_start_test)]
        )

    # Put this into HATANN
    EXP_DIR = SC26_DIR / "lorenz_trajexp"
    json_fpath = (
        EXP_DIR / f"dim{d}_dt{dt}_tol{tol}_ntrain{N_start_train * (N_steps+1)}.json"
    )

    conduct_experiment_given_data(
        json_fpath=json_fpath,
        DOMAIN=domain,
        F_DIM=d,
        DIM=d,
        N_TRAIN=N_train,
        N_TEST=N_test,
        X_train=X_train,
        X_test=X_test,
        f_fn=f_fn,
        g_fn=None,
        TOL=tol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC="Linf-global",
        INITIAL_CLUSTERING_METHOD="kmeans",
        NUM_INITIAL_CLUSTERS=min(get_NUM_INITIAL_CLUSTERS(d), 1000),
        MIN_PTS_PER_CLUSTER=min(get_MIN_PTS_PER_CLUSTER(d), 500),
        USE_WORK_STEALING=False,
        Q_reg=Q_reg,
        COARSE_FACTOR=coarse_factor,
    )

    create_summary_dataframe(EXP_DIR)
    return EXP_DIR


def setup_argparse():
    parser = argparse.ArgumentParser()

    # Experiment basics
    parser.add_argument(
        "--experiment",
        help="The name of the experiment to run.",
        type=str,
        default="rosenbrock_dim_scaling",
    )
    parser.add_argument(
        "--tols",
        help="The tolerances to use for the experiments.",
        type=float,
        nargs="+",
        default=[0.1, 0.05],
    )
    parser.add_argument(
        "--dims",
        help="The dims to use for the experiments.",
        type=int,
        nargs="+",
        default=[2, 5],
    )

    # Other optional stuff for specific experiments (i.e. dt for Lorenz or other flow maps)
    parser.add_argument(
        "--dt",
        help="The time horizon to use for the flow map experiment. \
                        Only does something if a flow map experiment is specified.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--error_eval_mode",
        help="The error eval mode to use.",
        type=str,
        default="affine_cov_hess",
    )
    parser.add_argument(
        "--sampling",
        help="The sampling strategy to use for initial points.",
        type=str,
        default="uniform",
    )
    parser.add_argument(
        "--coarse_factor",
        help="The number of nearest neighbor points to use during inference.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--Q_reg",
        help="The regularization to use for the trust region construction, using Tikhonov.",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--N_train",
        help="The number of initial training points to use.",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # Check if specified strings are valid
    possible_experiments = [
        "anisotropic_quadratic",
        "anisotropic_gaussian",
        "degenerate_quadratic",
        "indef_degen_quadratic",
        "rosenbrock",
        "lorenz96",
        "wave",
    ]
    possible_error_eval_modes = [
        "affine_cov_hess",
        "affine_cov_no_hess",
        "affine_cov_hess_exact_grad",
        "nearest_neighbor",
        "taylor_cov_hess",
        "taylor_cov_no_hess",
        "taylor_lsq_hess",
        "taylor_exact_hess",
        "affine_exact_hess",
    ]
    possible_sampling = ["uniform", "modified"]
    if args.experiment not in possible_experiments:
        raise ValueError(
            f"ERROR: experiment specified is not one of the implemented test functions.\nPossible options are: {possible_experiments}"
        )
    if args.error_eval_mode not in possible_error_eval_modes:
        raise ValueError(
            f"ERROR: error eval. mode specified is not implemented.\nPossible options are: {possible_error_eval_modes}"
        )
    if args.sampling not in possible_sampling:
        raise ValueError(
            f"ERROR: sampling strategy specified is not implemented. Possible options are: {possible_sampling}"
        )

    return args


def run_experiment_from_args(args):

    # Grab values from dictionary
    tolvec = args.tols
    dvec = args.dims
    function_name = args.experiment
    sampling = args.sampling
    Q_reg = args.Q_reg
    coarse_factor = args.coarse_factor
    N_train = args.N_train

    # Pick correct enum for error_eval_mode
    enum_map = {
        "affine_cov_hess": SurrogateMode.AFFINE_COV_HESS,
        "affine_exact_hess": SurrogateMode.AFFINE_EXACT_HESS,
        "affine_cov_no_hess": SurrogateMode.AFFINE_COV_NO_HESS,
        "affine_cov_hess_exact_grad": SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD,
        "nearest_neighbor": SurrogateMode.NEAREST_NEIGHBOR,
        "taylor_cov_hess": SurrogateMode.TAYLOR_COV_HESS,
        "taylor_exact_hess": SurrogateMode.TAYLOR_EXACT_HESS,
        "taylor_cov_no_hess": SurrogateMode.TAYLOR_COV_NO_HESS,
        "taylor_lsq_hess": SurrogateMode.TAYLOR_LSQ_HESS,
    }
    error_eval_mode = enum_map[args.error_eval_mode]

    for tol in tolvec:
        for d in dvec:
            if sampling == "modified" and function_name == "lorenz96":
                lorenz_trajectories_experiment(
                    d, tol, error_eval_mode, args.dt, Q_reg, coarse_factor, N_train
                )
                continue
            if sampling == "modified" and function_name in MODIFIED_SAMPLING:
                problem = MODIFIED_SAMPLING[function_name](args)
            else:
                problem = PROBLEMS[function_name](args)
            run_problem(problem, d, tol, error_eval_mode, Q_reg, coarse_factor, N_train)


if __name__ == "__main__":
    import logging

    from src.logging_utils import setup_logging

    setup_logging(console_level=logging.DEBUG)

    args = setup_argparse()
    run_experiment_from_args(args)
