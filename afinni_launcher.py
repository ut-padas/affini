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
import matplotlib.pyplot as plt

from src import IS_CUPY_BACKEND, RESULTS_DIR, get_mpi_rank, get_mpi_size, to_host, xp
from src.components.modes import SurrogateMode
from src.components.sampling import SAMPLERS
from src.config import (
    TrainingConfig,
    get_MIN_PTS_PER_CLUSTER,
    get_N_TRAIN,
    get_NUM_INITIAL_CLUSTERS,
)
from src.core.evaluation import Evaluator
from src.infra.backend import MPI
from src.plotting import (
    plot_adaptive_clustering_results,
    plot_cluster_projections_for_wave,
    plot_error_histogram,
    plot_near_centers,
    plot_profile_dashboard,
    plot_wave_2d_errors,
    plot_wave_line_errors,
)
from src.surrogate import AFINNISurrogate, compute_error_metrics

TINY_EPS = 1e-12
DEFAULT_SAMPLING_METHOD = "convex_combination"


DEFAULT_Q_REG = 1e-6
DEFAULT_COARSE_FACTOR = 10


def _sampling_suffix(sampling_method: str) -> str:
    """File-name suffix for a non-default sampler (default runs keep their names)."""
    return "" if sampling_method == DEFAULT_SAMPLING_METHOD else f"_aug{sampling_method}"


def _settings_suffix(sampling_method: str, Q_reg: float, coarse_factor: int) -> str:
    """
    File-name suffix for every non-default setting that changes results, so a run with
    different settings never skips because of an unrelated cached result. Runs with
    the defaults keep their historical names, so existing results are still found.
    """
    suffix = _sampling_suffix(sampling_method)
    if Q_reg != DEFAULT_Q_REG:
        suffix += f"_qreg{Q_reg:g}"
    if coarse_factor != DEFAULT_COARSE_FACTOR:
        suffix += f"_coarse{coarse_factor}"
    return suffix


def _tolerance_tag(atol: float | None, rtol: float | None, scaled: bool = False) -> str:
    """
    File-name part for the tolerances: ``tol{t}`` for exactly what ``--tols t`` runs
    (atol = rtol = t, or atol only where atol is scaled by max|f|), so existing results
    are still found; otherwise ``atol{a}_rtol{r}``, with ``none`` for an omitted test.
    """
    if atol is not None and rtol == (None if scaled else atol):
        return f"tol{atol}"
    return f"atol{'none' if atol is None else atol}_rtol{'none' if rtol is None else rtol}"


def tolerance_pairs(
    tols: list[float], atols: list[float | None] | None, rtols: list[float | None] | None, scaled: bool
) -> list[tuple[float | None, float | None]]:
    """
    The (atol, rtol) pairs to run: every combination of ``atols`` x ``rtols`` (``None``
    leaves that test out, a missing list means none; the pair without any tolerance is
    skipped), or, when neither list is given, ``atol = rtol = t`` for each of ``tols``,
    with atol only where it is scaled by max|f| (``scaled``), as before.
    """
    if atols is None and rtols is None:
        return [(t, None if scaled else t) for t in tols]
    pairs = [(a, r) for a in (atols or [None]) for r in (rtols or [None])]
    if (None, None) in pairs:
        pairs = [p for p in pairs if p != (None, None)]
        if _rank == 0:
            print("Skipping atol=none, rtol=none: an acceptance test needs a tolerance.")
    return pairs


SEED = 0

logger = logging.getLogger(__name__)
_size = get_mpi_size()
_rank = get_mpi_rank()


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
    ATOL: float | None,
    RTOL: float | None,
    ERROR_EVAL_MODE: SurrogateMode,
    ERROR_METRIC: Literal["L2", "Linf"],
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
    SAMPLING_METHOD: str = DEFAULT_SAMPLING_METHOD,
    SCALE_ATOL_BY_MAX_F: bool = False,
):
    """
    Train, predict and score one experiment, writing ``json_fpath`` and its plots (rank 0).

    The acceptance test uses ``ATOL`` and ``RTOL`` (scalars applied to every output
    component; ``None`` leaves that test out). With ``SCALE_ATOL_BY_MAX_F``, ``ATOL``
    is multiplied by max|f_j| per output component over the training data; that
    evaluates ``f`` on the training points here and passes the values on as
    ``Y_train``, so they are not evaluated twice. The JSON records the requested
    ``atol``/``rtol`` and, when scaled, the per-component ``atol_scaled`` used.
    """

    if json_fpath.exists():
        if _rank == 0:
            print(f"Experiment already exists at {json_fpath}. Skipping...")
        return
    json_fpath.parent.mkdir(parents=True, exist_ok=True)

    surrogate = AFINNISurrogate()
    error_tag = ""
    _TRAINED, _PREDICTED = False, False
    try:
        if _rank == 0:
            print("\n--- TRAINING ---")
        atol_used = ATOL
        if SCALE_ATOL_BY_MAX_F and ATOL is not None:
            if _rank == 0:
                Y_train = f_fn(X_train) if Y_train is None else Y_train
                atol_used = tuple(float(ATOL * m) for m in to_host(xp.max(xp.abs(Y_train.reshape(len(Y_train), -1)), axis=0)))
            # Only rank 0 holds the data, but every rank must train with the same tolerances.
            atol_used = MPI.COMM_WORLD.bcast(atol_used, root=0)
        _start_train_time = time.perf_counter()
        config = TrainingConfig(
            mode=ERROR_EVAL_MODE,
            atol=atol_used,
            rtol=RTOL,
            error_metric=ERROR_METRIC,
            initial_clustering_method=INITIAL_CLUSTERING_METHOD,
            num_initial_clusters=NUM_INITIAL_CLUSTERS,
            min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
            use_work_stealing=USE_WORK_STEALING,
            max_levels=MAX_LEVELS,
            max_clusters_est=MAX_CLUSTERS_EST,
            log_mem_stats=True,
            profile_timing=True,
            save_all_gen_training_data=SAVE_ALL_GEN_TRAINING_DATA,
            q_reg=Q_reg,
            sampling_method=SAMPLING_METHOD,
        )
        # All ranks participate in training
        surrogate.train(X_train, config, Evaluator(f_fn=f_fn, g_fn=g_fn, hessian_fn=HESSIAN), Y_train=Y_train)
        _train_time = time.perf_counter() - _start_train_time
        _TRAINED = True

        if _rank == 0:
            print("\n--- PREDICTION ---")

        # Save centroid data if user requests it
        if _rank == 0 and SAVE_FINAL_CENTROIDS is True:
            xp.save(f"centroids_{_tolerance_tag(ATOL, RTOL, SCALE_ATOL_BY_MAX_F)}_d{DIM}.npy", surrogate.X_src)

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
        atol=atol_used,
        rtol=RTOL,
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
        "atol": ATOL,
        "rtol": RTOL,
        **({"atol_scaled": atol_used} if atol_used is not ATOL else {}),
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
        "sampling_method": SAMPLING_METHOD,
        "Q_reg": Q_reg,
        "k_neighbours": K_NEIGHBOURS,
        "coarse_factor": COARSE_FACTOR,
        "converged": surrogate.converged,
        "num_unresolved_pts": surrogate.num_unresolved_pts,
        "config": surrogate.config,  # every effective training setting and component
        "error_metrics": error_metrics,
        # Under this key the saved file can be passed straight to the src.plotting profile plots.
        "profiling_stats": surrogate.profiling_stats,
    }
    with open(json_fpath, "w") as jf:
        # Array scalars (NumPy/CuPy) must stay numbers for the plots; anything else unknown is saved as text.
        json.dump(json_data, jf, indent=4, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"Experiment completed and saved to {json_fpath}")

    plt.close(plot_profile_dashboard(surrogate, json_fpath.with_name(json_fpath.stem + "_profile.png")))

    plot_save_path = json_fpath.with_name(json_fpath.stem + "_error_histogram.png")
    plot_error_histogram(
        rtol=RTOL,
        ground_truth=ground_truth,
        predictions=preds,
        img_fpath=plot_save_path,
    )

    if DIM == 2 and surrogate.N_src <= 100_000:
        plot_save_path = json_fpath.with_name(json_fpath.stem + "_centroid_plot.png")
        plot_adaptive_clustering_results(
            centroids=to_host(surrogate.X_src),
            Q=to_host(surrogate.Qs_src),
            domain=DOMAIN,  # type: ignore
            # The contours show a scalar function: the first output component.
            f_fn=f_fn if F_DIM == 1 else (lambda X: f_fn(X)[:, 0]),
            img_fpath=plot_save_path,
            plot_in_log=CENTROIDS_PLOT_IN_LOG_SCALE,
        )

    if json_fpath.parent.name == "wave":

        proj_plot_save_path = json_fpath.with_name(
            json_fpath.stem + "_projection_plot.png"
        )
        plot_cluster_projections_for_wave(
            to_host(surrogate.X_src), proj_plot_save_path, DIM
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
        if target.shape[0] == DIM:
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
    ATOL: float | None,
    RTOL: float | None,
    ERROR_EVAL_MODE: SurrogateMode,
    ERROR_METRIC: Literal["L2", "Linf"],
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
    SAMPLING_METHOD: str = DEFAULT_SAMPLING_METHOD,
    SCALE_ATOL_BY_MAX_F: bool = False,
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
        ATOL,
        RTOL,
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
        SAMPLING_METHOD=SAMPLING_METHOD,
        SCALE_ATOL_BY_MAX_F=SCALE_ATOL_BY_MAX_F,
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
        # Results written before --atols/--rtols store one "tol" (atol = rtol = tol,
        # or atol only, then recorded with rtol None).
        atol, rtol = (data["atol"], data["rtol"]) if "tol" not in data else (data["tol"], data.get("rtol", data["tol"]))
        data_dict["atol"].append(atol)
        data_dict["rtol"].append(rtol)

        data_dict["sampling_method"].append(data.get("sampling_method", DEFAULT_SAMPLING_METHOD))
        data_dict["num_initial_clusters"].append(data.get("num_initial_clusters"))
        data_dict["min_pts_per_cluster"].append(data.get("min_pts_per_cluster"))
        data_dict["e"].append(data["error_metrics"]["all"]["l2_rel_error"])
        data_dict["e_in"].append(data["error_metrics"]["in"]["l2_rel_error"])
        data_dict["p_out"].append(data["error_metrics"]["perct_out"])

    df = pd.DataFrame(data_dict)
    df = df.sort_values(
        by=["dim", "atol", "rtol", "error_eval_mode"], ascending=[True, False, False, True]
    )
    order = [
        "dim",
        "atol",
        "rtol",
        "error_eval_mode",
        "error_metric",
        "initial_clustering_method",
        "sampling_method",
        "num_initial_clusters",
        "min_pts_per_cluster",
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

    Results go to ``RESULTS_DIR / dir_name``. Training and test points are drawn
    uniformly from ``domain`` (on rank 0), unless ``sample(dim, n_train, n_test)``
    is given, which returns ``(X_train, X_test)``. With ``scale_atol_by_max_f`` the
    acceptance test is absolute only, scaled by max|f| (see conduct_experiment_given_data).
    """

    dir_name: str
    f: Callable
    g: Callable | None = None
    hessian: Callable | None = None
    domain: tuple[float, float] = (-3.0, 3.0)
    error_metric: Literal["L2", "Linf"] = "Linf"
    scale_atol_by_max_f: bool = False
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
        aniso_gaussian,
        aniso_gaussian_grad,
        get_random_rotation,
    )

    def sample(dim, n_train, n_test):
        Q = xp.asarray(get_random_rotation(dim, xp.float64))  # helpers return NumPy
        eigvals = xp.asarray(_get_precision_eigenvalues(dim, xp.float64, 3))
        Ainv = Q @ xp.diag(1.0 / eigvals) @ Q.T
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
        "rosenbrock_modified_sampling",
        partial(rosenbrock, a=100.0),
        partial(rosenbrock_grad, a=100.0),
        domain=(-2.0, 3.0),
        sample=sample,
        centroids_plot_in_log_scale=True,
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
            import cupy as cp

            # The batched JAX flow map returns (N, 1, d): one saved time, tf.
            X_jax = jax.dlpack.from_dlpack(x)
            Y_jax = lorenz96_flowmap_jax_batched(X_jax, forcing, tf)
            return cp.from_dlpack(Y_jax)[:, 0, output_component].reshape(N, 1)
        else:
            f = lorenz96_flowmap(x, forcing, tf)[:, output_component]
            return f.reshape(N, 1)

    return Problem(
        f"lorenz96_dt_{tf}",
        f,
        domain=(-5.0, 5.0),
        scale_atol_by_max_f=True,
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
    atol: float | None,
    rtol: float | None,
    error_eval_mode: SurrogateMode,
    Q_reg: float,
    coarse_factor: int,
    N_train: int | None = None,
    sampling_method: str = DEFAULT_SAMPLING_METHOD,
    num_initial_clusters: int | None = None,
    min_pts_per_cluster: int | None = None,
):
    """
    Train and test one surrogate for ``problem``; skip if the result exists.
    ``num_initial_clusters`` / ``min_pts_per_cluster`` default to the dimension heuristics.
    """
    if problem.only_modes is not None and error_eval_mode not in problem.only_modes:
        print(
            f"Skipping ERROR_EVAL_MODE={error_eval_mode.value}: this sampling is only "
            f"set up for {[m.value for m in problem.only_modes]}."
        )
        return None
    if problem.max_dim is not None and dim > problem.max_dim:
        print(f"Not implemented yet for dimensions above d={problem.max_dim}.")
        return None

    EXP_DIR = RESULTS_DIR / problem.dir_name
    ERROR_METRIC = problem.error_metric
    INITIAL_CLUSTERING_METHOD = "kmeans"
    USE_V2 = True
    USE_WORK_STEALING = True

    N_TRAIN = problem.n_train(dim) if N_train is None else N_train
    N_TEST = problem.n_test(N_TRAIN)
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(dim) if num_initial_clusters is None else num_initial_clusters
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(dim) if min_pts_per_cluster is None else min_pts_per_cluster

    json_fpath = (
        EXP_DIR
        / f"mpi_sz{_size}_dim{dim}_{_tolerance_tag(atol, rtol, problem.scale_atol_by_max_f)}_ntrain{N_TRAIN}_{error_eval_mode.value}_{ERROR_METRIC}_{INITIAL_CLUSTERING_METHOD}_ninit{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}_v2{USE_V2}{_settings_suffix(sampling_method, Q_reg, coarse_factor)}.json"
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
        ATOL=atol,
        RTOL=rtol,
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
        SAMPLING_METHOD=sampling_method,
        SCALE_ATOL_BY_MAX_F=problem.scale_atol_by_max_f,
    )
    if problem.sample is None:
        conduct_experiment_generate_uniform_data(**settings)
    else:
        # Like the uniform path: rank 0 holds the data, drawn from a fixed seed.
        X_train, X_test = None, None
        if _rank == 0:
            xp.random.seed(SEED)
            X_train, X_test = problem.sample(dim, N_TRAIN, N_TEST)
        conduct_experiment_given_data(X_train=X_train, X_test=X_test, **settings)

    create_summary_dataframe(EXP_DIR)
    return EXP_DIR


def lorenz_trajectories_experiment(
    d,
    atol,
    rtol,
    error_eval_mode,
    dt,
    Q_reg,
    coarse_factor,
    N_train=None,
    sampling_method=DEFAULT_SAMPLING_METHOD,
    num_initial_clusters=None,
    min_pts_per_cluster=None,
):
    """
    Goal: use time step data as training and testing to do experiments.
    I.e. use manifolds of solutions as initial data. ``atol`` is scaled by max|f_j|
    per state component (see conduct_experiment_given_data). ``num_initial_clusters``
    / ``min_pts_per_cluster`` default to the heuristics capped at 1000 / 500; other
    values are added to the file name.
    """
    # Parameters for experiment
    l = -5.0
    r = 5.0
    domain = [l, r]
    N_test = 50_000
    forcing = 1

    # Define the function: the full Lorenz-96 state after time dt
    from src.problems.lorenz96_flowmap import lorenz96_flowmap
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched

    def f_fn(X):
        if not IS_CUPY_BACKEND:
            return lorenz96_flowmap(X, forcing, dt)
        import cupy as cp

        # The batched JAX flow map returns (N, 1, d): one saved time, dt.
        Y_jax = lorenz96_flowmap_jax_batched(jax.dlpack.from_dlpack(X), forcing, dt)
        return cp.from_dlpack(Y_jax)[:, 0, :]

    # Use this function to generate data.
    N_start_train = 20000
    N_start_test = 5000
    N_steps = 9
    if N_train is not None:
        # Total training points = N_start_train * (N_steps + 1)
        N_start_train = max(1, N_train // (N_steps + 1))
    N_train = N_start_train * (N_steps + 1)
    N_test = N_start_test * (N_steps + 1)
    # Like the other experiments: rank 0 holds the data, drawn from a fixed seed.
    X_train, X_test = None, None
    if _rank == 0:
        xp.random.seed(SEED)
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

    # Put this into AFINNI. The mode and MPI size are only in the name when they
    # differ from the historical defaults, so existing results are still found.
    EXP_DIR = RESULTS_DIR / "lorenz_trajexp"
    mode_suffix = "" if error_eval_mode == SurrogateMode.AFFINE_COV_HESS else f"_{error_eval_mode.value}"
    size_suffix = "" if _size == 1 else f"_mpisz{_size}"
    default_ninit, default_minpts = min(get_NUM_INITIAL_CLUSTERS(d), 1000), min(get_MIN_PTS_PER_CLUSTER(d), 500)
    num_initial_clusters = default_ninit if num_initial_clusters is None else num_initial_clusters
    min_pts_per_cluster = default_minpts if min_pts_per_cluster is None else min_pts_per_cluster
    cluster_suffix = ("" if num_initial_clusters == default_ninit else f"_ninit{num_initial_clusters}") + (
        "" if min_pts_per_cluster == default_minpts else f"_minpts{min_pts_per_cluster}"
    )
    json_fpath = EXP_DIR / (
        f"dim{d}_dt{dt}_{_tolerance_tag(atol, rtol, scaled=True)}_ntrain{N_start_train * (N_steps + 1)}{mode_suffix}{size_suffix}{cluster_suffix}"
        f"{_settings_suffix(sampling_method, Q_reg, coarse_factor)}.json"
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
        ATOL=atol,
        RTOL=rtol,
        ERROR_EVAL_MODE=error_eval_mode,
        ERROR_METRIC="Linf",
        SCALE_ATOL_BY_MAX_F=True,
        INITIAL_CLUSTERING_METHOD="kmeans",
        NUM_INITIAL_CLUSTERS=num_initial_clusters,
        MIN_PTS_PER_CLUSTER=min_pts_per_cluster,
        USE_WORK_STEALING=False,
        Q_reg=Q_reg,
        COARSE_FACTOR=coarse_factor,
        SAMPLING_METHOD=sampling_method,
    )

    create_summary_dataframe(EXP_DIR)
    return EXP_DIR


def _tolerance_value(text: str) -> float | None:
    """One --atols/--rtols value: a finite number >= 0, or ``none``."""
    if text.lower() == "none":
        return None
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number or 'none'") from None
    if not (value >= 0 and value != float("inf")):
        raise argparse.ArgumentTypeError(f"{text!r} must be a finite number >= 0")
    return value


def _positive_int(text: str) -> int:
    """An integer >= 1, for the cluster-size arguments."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be >= 1")
    return value


def setup_argparse(argv: list[str] | None = None):
    """Parse the command line (``argv``, default ``sys.argv``); --tols with --atols/--rtols exits with an error."""
    parser = argparse.ArgumentParser()

    # Experiment basics
    parser.add_argument(
        "--experiment",
        help="The name of the experiment to run.",
        choices=sorted(PROBLEMS),
        default="rosenbrock",
    )
    parser.add_argument(
        "--tols",
        help="Tolerances to sweep, each used as atol = rtol (default 0.1 0.05; for problems "
        "that scale atol by max|f|, atol only). Not allowed with --atols/--rtols.",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--atols",
        help="Absolute tolerances to sweep (scalars, applied to every output component; "
        "'none' leaves the absolute test out). Every combination with --rtols is run.",
        type=_tolerance_value,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--rtols",
        help="Relative tolerances to sweep, as --atols.",
        type=_tolerance_value,
        nargs="+",
        default=None,
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
        choices=[m.value for m in SurrogateMode],
        default=SurrogateMode.AFFINE_COV_HESS.value,
    )
    parser.add_argument(
        "--sampling",
        help="The sampling strategy to use for initial points.",
        choices=["uniform", "modified"],
        default="uniform",
    )
    parser.add_argument(
        "--coarse_factor",
        help="The number of nearest neighbor points to use during inference.",
        type=int,
        default=DEFAULT_COARSE_FACTOR,
    )
    parser.add_argument(
        "--Q_reg",
        help="The regularization to use for the trust region construction, using Tikhonov.",
        type=float,
        default=DEFAULT_Q_REG,
    )
    parser.add_argument(
        "--N_train",
        help="The number of initial training points to use.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_initial_clusters",
        help="Number of initial clusters (default: get_NUM_INITIAL_CLUSTERS(dim)).",
        type=_positive_int,
        default=None,
    )
    parser.add_argument(
        "--min_pts_per_cluster",
        help="Minimum points per cluster; rejected clusters are augmented to twice this "
        "before splitting (default: get_MIN_PTS_PER_CLUSTER(dim)).",
        type=_positive_int,
        default=None,
    )
    parser.add_argument(
        "--sampling_method",
        help="How new points are sampled inside rejected clusters during training "
        "(--sampling chooses the initial training data instead).",
        choices=sorted(SAMPLERS),
        default=DEFAULT_SAMPLING_METHOD,
    )

    args = parser.parse_args(argv)
    if args.tols is not None and (args.atols is not None or args.rtols is not None):
        parser.error("--tols cannot be combined with --atols/--rtols")
    if args.tols is None:
        args.tols = [0.1, 0.05]
    return args


def run_experiment_from_args(args):
    """Run every (atol, rtol) pair from ``tolerance_pairs`` in every dimension of ``--dims``; existing results are skipped."""

    # Grab values from dictionary
    dvec = args.dims
    function_name = args.experiment
    sampling = args.sampling
    Q_reg = args.Q_reg
    coarse_factor = args.coarse_factor
    N_train = args.N_train

    error_eval_mode = SurrogateMode(args.error_eval_mode)

    if sampling == "modified" and function_name == "lorenz96":
        for atol, rtol in tolerance_pairs(args.tols, args.atols, args.rtols, scaled=True):
            for d in dvec:
                lorenz_trajectories_experiment(
                    d, atol, rtol, error_eval_mode, args.dt, Q_reg, coarse_factor, N_train, args.sampling_method,
                    args.num_initial_clusters, args.min_pts_per_cluster,
                )  # fmt: skip
        return
    if sampling == "modified" and function_name in MODIFIED_SAMPLING:
        problem = MODIFIED_SAMPLING[function_name](args)
    else:
        problem = PROBLEMS[function_name](args)
    for atol, rtol in tolerance_pairs(args.tols, args.atols, args.rtols, problem.scale_atol_by_max_f):
        for d in dvec:
            run_problem(
                problem, d, atol, rtol, error_eval_mode, Q_reg, coarse_factor, N_train, args.sampling_method,
                args.num_initial_clusters, args.min_pts_per_cluster,
            )  # fmt: skip


if __name__ == "__main__":
    import logging

    from src.infra.logging_utils import setup_logging

    setup_logging(console_level=logging.DEBUG)

    args = setup_argparse()
    run_experiment_from_args(args)
