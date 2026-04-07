from functools import partial
from itertools import product
from pathlib import Path

from tqdm import tqdm

from experiments_utils import (
    ICS26_DIR,
    Problem_experiment_setup,
    create_summary_dataframe,
    del_all_error_files_in_dir,
)
from src.logging_utils import consolidate_evaluations
from src.problems.vector_quadratic_2 import vector_quadratic_2, vector_quadratic_2_grad
from src.problems.vector_quadratic_3 import vector_quadratic_3, vector_quadratic_3_grad
from src.problems.anisotropic_gaussian import aniso_gaussian, aniso_gaussian_grad
from src.problems.anisotropic_quadratic import aniso_quadratic, aniso_quadratic_grad
from src.problems.lorenz96_flowmap import (
    lorenz96_flowmap,
    lorenz96_flowmap_grad,
    lorenz96_flowmap_oracle,
    lorenz96_flowmap_oracle_first_component,
)
from src.problems.vanderpol_flowmap import (
    vanderpol_flowmap,
    vanderpol_flowmap_grad,
    vanderpol_flowmap_oracle,
    vanderpol_flowmap_oracle_first_component
)
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.problems.shifted_gaussians import (
    CENTERS_2D,
    CENTERS_3D,
    CENTERS_4D,
    CENTERS_5D,
    shifted_gaussians,
    shifted_gaussians_grad,
)


def get_N_TRAIN(dim: int, *, base: float = 1.3, factor: int = 50_000) -> int:
    """
    Heuristic to determine N_TRAIN based on dimension and tolerance.
        factor * (base)^dim
    """
    N_TRAIN = int(factor * (base**dim))
    return N_TRAIN


def get_N_INIT_CLUSTERS(dim: int, *, base: float = 1.3, factor: int = 25) -> int:
    """
    Heuristic to determine N_INIT_CLUSTERS based on dimension.
        factor * (base)^dim
    """
    N_INIT_CLUSTERS = int(factor * (base**dim))
    return N_INIT_CLUSTERS


def get_N_MIN_PER_CLUSTER(dim: int, *, base: float = 1.3, factor: int = 100) -> int:
    """
    Heuristic to determine N_MIN_PER_CLUSTER based on dimension.
        factor * (base)^dim
    """
    N_MIN_PER_CLUSTER = int(factor * (base**dim))
    return N_MIN_PER_CLUSTER


def ics_aniso_gaussian_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "gaussian_cov_taylor"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 5]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        json_path = (
            EXP_DIR
            / f"ics26_random_anisotropic_gaussian_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-3, 3],
            f_fn=aniso_gaussian,
            g_fn=aniso_gaussian_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX="Cov_Lvl_curves",
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
            STORE_HESSIAN=False,
            USE_HESSIAN_FOR_INFERENCE=False,
            ERROR_EVAL_MODE="taylor"
        )

    return EXP_DIR


def ics_aniso_quadratic_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "quadratic_do_not_use_hessian_for_inference"
    create_summary_dataframe(EXP_DIR)
    DIMs = [5, 10]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM == 10 and TOL <= 0.005:
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        json_path = (
            EXP_DIR
            / f"ics26_anisotropic_quadratic_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-3, 3],
            f_fn=aniso_quadratic,
            g_fn=aniso_quadratic_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,
            CLUSTERING_MATRIX="Cov_Lvl_curves",
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
            PLOT_CONTOUR_IN_LOG_SCALE=True,
            STORE_HESSIAN=False,
            USE_HESSIAN_FOR_INFERENCE=False,
        )

    return EXP_DIR


# Tests the alg on vector_quadratic_2, which has multiple outputs with different Hessians.
# Compare to not using no Hessian (will they average out to isotropic?)
def quadratic_2_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "quadratic_2_none"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM == 10 and TOL <= 0.005:
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        json_path = (
            EXP_DIR
            / f"ics26_anisotropic_quadratic_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=2,
            DOMAIN=[-3, 3],
            f_fn=vector_quadratic_2,
            g_fn=vector_quadratic_2_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,
            CLUSTERING_MATRIX="None",
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
            PLOT_CONTOUR_IN_LOG_SCALE=True,
            STORE_HESSIAN=False,
            USE_HESSIAN_FOR_INFERENCE=False,
        )

    return EXP_DIR



# Tests the alg on vector_quadratic_3, which has multiple outputs with the same Hessian.
def quadratic_3_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    output_d = 1
    EXP_DIR = ICS26_DIR / f"quadratic_3_outd_{output_d}_covlvlcurves"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM == 10 and TOL <= 0.005:
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        json_path = (
            EXP_DIR
            / f"ics26_anisotropic_quadratic_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)

        def f_fn(X):
            return vector_quadratic_3(X, d_out=output_d)
        def g_fn(X):
            return vector_quadratic_3_grad(X, d_out=output_d)

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=output_d,
            DOMAIN=[-3, 3],
            f_fn=f_fn,
            g_fn=g_fn,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,
            CLUSTERING_MATRIX="Cov_Lvl_curves",
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
            PLOT_CONTOUR_IN_LOG_SCALE=True,
            STORE_HESSIAN=False,
            USE_HESSIAN_FOR_INFERENCE=False,
        )

    return EXP_DIR






def ics_aniso_quadratic_runtime_scalability_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "ics_anisotropic_quadratic_runtime_scalability"
    create_summary_dataframe(EXP_DIR)

    # DO NOT CHANGE ANY OF THE SETTINGS BELOW
    DIMs = [1, 2, 4, 8, 16, 20, 32, 40, 64]
    TOLs = [0.01]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM >= 64 and TOL <= 0.1:
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        base = 1.1
        N_TRAIN = get_N_TRAIN(DIM, base=base, factor=100_000)
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM, base=1.01, factor=10)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM, base=base, factor=100)

        N_TRAIN = N_TRAIN // 2
        N_INIT_CLUSTERS = N_INIT_CLUSTERS // 2
        N_MIN_PER_CLUSTER = N_MIN_PER_CLUSTER // 2
        N_TEST = max(N_TRAIN // 4, 500)

        if DIM >= 128:
            BATCH_SIZE = 500_000
        elif DIM >= 32:
            BATCH_SIZE = 10_000_000
        elif DIM >= 16:
            BATCH_SIZE = 50_000_000
        else:
            BATCH_SIZE = 100_000_000

        json_path = (
            EXP_DIR
            / f"ics26_anisotropic_quadratic_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_batch{BATCH_SIZE}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        print(
            f"N_TRAIN: {N_TRAIN:,}, N_INIT_CLUSTERS: {N_INIT_CLUSTERS:,}, N_MIN_PER_CLUSTER: {N_MIN_PER_CLUSTER:,}, BATCH_SIZE: {BATCH_SIZE:,}, N_TEST: {N_TEST:,}"
        )

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-3, 3],
            f_fn=aniso_quadratic,
            g_fn=aniso_quadratic_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX='Hessian_LSQ',
            STORE_TRAIN_EVALS=False,
            BATCH_SIZE=BATCH_SIZE,
            PROFILE_TIMING=True,
        )

        create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def ics_rosenbrock_runtime_scalability_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "ics_rosenbrock_runtime_scalability"
    create_summary_dataframe(EXP_DIR)
    DIMs = [5]
    TOLs = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        N_TRAIN = get_N_TRAIN(DIM)
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)
        N_TEST = max(N_TRAIN // 4, 500)

        if DIM >= 128:
            BATCH_SIZE = 500_000
        elif DIM >= 32:
            BATCH_SIZE = 10_000_000
        elif DIM >= 16:
            BATCH_SIZE = 50_000_000
        else:
            BATCH_SIZE = 100_000_000

        json_path = (
            EXP_DIR
            / f"ics26_rosenbrock_dim{DIM}_tol{TOL}_ntrain{N_TRAIN:.2e}_batch{BATCH_SIZE}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        print(
            f"N_TRAIN: {N_TRAIN:,}, N_INIT_CLUSTERS: {N_INIT_CLUSTERS:,}, N_MIN_PER_CLUSTER: {N_MIN_PER_CLUSTER:,}, BATCH_SIZE: {BATCH_SIZE:,}, N_TEST: {N_TEST:,}"
        )

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2, 3],
            f_fn=rosenbrock,
            g_fn=rosenbrock_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX='Hessian_LSQ',
            STORE_TRAIN_EVALS=False,
            BATCH_SIZE=BATCH_SIZE,
            PROFILE_TIMING=True,
            PLOT_CONTOUR_IN_LOG_SCALE=True,
        )
        create_summary_dataframe(EXP_DIR)

    return EXP_DIR


def ics_rosenbrock_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "rosenbrock"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)

    DIMs = [2]
    TOLs = [0.05]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Rosenbrock Experiments"):
        if DIM == 10 and (
            (TOL <= 0.1 and ERROR_METRIC == "Linf-local")
            or (TOL <= 0.02 and ERROR_METRIC == "Linf-global")
        ):
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        json_path = (
            EXP_DIR / f"ics26_rosenbrock_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = 50000
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2.0, 3.0],
            f_fn=rosenbrock,
            g_fn=rosenbrock_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX="Hessian_LSQ",
            IGNORE_GRADIENT=False,
            STORE_HESSIAN=False,
            USE_HESSIAN_FOR_INFERENCE=False,
        )

    return EXP_DIR


def ics_rosenbrock_n_min_per_cluster_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    EXP_DIR = ICS26_DIR / "ics_rosenbrock_n_min_per_cluster"
    create_summary_dataframe(EXP_DIR)

    DIMs = [5]
    TOLs = [0.1, 0.05, 0.02]
    ERROR_METRICs = ["Linf-global", "Linf-local"]
    N_MIN_PER_CLUSTER_FACTORS = [100, 500, 1000, 5000]

    combos = list(product(DIMs, TOLs, ERROR_METRICs, N_MIN_PER_CLUSTER_FACTORS))
    for DIM, TOL, ERROR_METRIC, N_MIN_PER_CLUSTER_FACTOR in tqdm(
        combos, desc="Rosenbrock Experiments"
    ):
        if DIM >= 5 and (
            (
                ERROR_METRIC == "Linf-local"
                and (
                    (TOL <= 0.05 and N_MIN_PER_CLUSTER_FACTOR >= 5000)
                    or (TOL <= 0.02 and N_MIN_PER_CLUSTER_FACTOR >= 1000)
                )
            )
            or (
                ERROR_METRIC == "Linf-global"
                and TOL <= 0.02
                and N_MIN_PER_CLUSTER_FACTOR >= 1000
            )
        ):
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC}, N_MIN_PER_CLUSTER_FACTOR={N_MIN_PER_CLUSTER_FACTOR} due to resource constraints."
            )
            continue

        N_TRAIN = get_N_TRAIN(DIM)
        N_TEST = N_TRAIN // 2
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM, factor=N_MIN_PER_CLUSTER_FACTOR)

        json_path = (
            EXP_DIR
            / f"ics26_rosenbrock_n_min_per_cluster_dim{DIM}_tol{TOL}_err{ERROR_METRIC}_n_min_factor{N_MIN_PER_CLUSTER_FACTOR}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2.0, 3.0],
            f_fn=rosenbrock,
            g_fn=rosenbrock_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX='Cov_Lvl_curves',
        )

    return EXP_DIR


def generate_train_data_for_neural_network() -> None:
    TRAIN_EVALS_ROOT = ICS26_DIR / "ics_neural_network_train_evals"
    TOL = 0.05
    ERROR_METRIC = "Linf-local"
    USE_HESSIAN_INFORMED_SPLIT = True
    _prefix = "train_evals"
    K_NEIGHBOURS = 1

    # 2D, 5D Gaussians
    DIMs = [2, 5]
    for DIM in DIMs:
        _evals_dir = TRAIN_EVALS_ROOT / f"shifted_gaussians_{DIM}d_tol{TOL}"
        _centers = {
            2: CENTERS_2D,
            3: CENTERS_3D,
            4: CENTERS_4D,
            5: CENTERS_5D,
        }
        shifted_gaussians_partial = partial(
            shifted_gaussians,
            centers=_centers[DIM],
        )
        shifted_gaussians_grad_partial = partial(
            shifted_gaussians_grad,
            centers=_centers[DIM],
        )
        Problem_experiment_setup(
            json_fpath=TRAIN_EVALS_ROOT
            / f"ics26_shifted_gaussians_{DIM}d_tol{TOL}_train_evals.json",
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2.0, 2.0],
            TOL=TOL,
            f_fn=shifted_gaussians_partial,
            g_fn=shifted_gaussians_grad_partial,
            N_TRAIN=get_N_TRAIN(DIM),
            N_INIT_CLUSTERS=get_N_INIT_CLUSTERS(DIM),
            N_MIN_PER_CLUSTER=get_N_MIN_PER_CLUSTER(DIM),
            MAX_CLUSTERS_EST=100_000,
            N_TEST=get_N_TRAIN(DIM) // 2,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,
            STORE_TRAIN_EVALS=True,
            TRAIN_EVALS_DIR=_evals_dir,
            TRAIN_EVALS_PREFIX=_prefix,
            CLUSTERING_MATRIX='Cov_Lvl_curves',
            SAVE_TEST_DATA=True,
            SAVE_TRAINED_DATA=True,
        )
        for _p in ["_f", "_g"]:
            consolidate_evaluations(
                save_dir=_evals_dir,
                prefix=_prefix + _p,
                output_path=_evals_dir / f"shifted_gaussians_{DIM}d_tol{TOL}{_p}.npz",
                delete_individual_files_after=True,
            )

    # 5D Rosenbrock
    DIMs = [2, 5]
    for DIM in DIMs:
        _evals_dir = TRAIN_EVALS_ROOT / f"rosenbrock_{DIM}d_tol{TOL}"
        Problem_experiment_setup(
            json_fpath=TRAIN_EVALS_ROOT
            / f"ics26_rosenbrock_{DIM}d_tol{TOL}_train_evals.json",
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2.0, 3.0],
            TOL=TOL,
            f_fn=rosenbrock,
            g_fn=rosenbrock_grad,
            N_TRAIN=get_N_TRAIN(DIM),
            N_INIT_CLUSTERS=get_N_INIT_CLUSTERS(DIM),
            N_MIN_PER_CLUSTER=get_N_MIN_PER_CLUSTER(DIM),
            MAX_CLUSTERS_EST=100_000,
            N_TEST=get_N_TRAIN(DIM) // 2,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,
            STORE_TRAIN_EVALS=True,
            TRAIN_EVALS_DIR=_evals_dir,
            TRAIN_EVALS_PREFIX=_prefix,
            CLUSTERING_MATRIX='Cov_Lvl_curves',
            SAVE_TEST_DATA=True,
            SAVE_TRAINED_DATA=True,
        )
        for _p in ["_f", "_g"]:
            consolidate_evaluations(
                save_dir=_evals_dir,
                prefix=_prefix + _p,
                output_path=_evals_dir / f"rosenbrock_{DIM}d_tol{TOL}{_p}.npz",
                delete_individual_files_after=True,
            )

    # Lorenz 5D, F=1
    DIM = 5
    FORCING = 1
    FINAL_TIME = 0.1
    OUTPUT_VARIABLE = 0

    def lorenz_f(x):
        f = lorenz96_flowmap(x, FORCING, FINAL_TIME)  # (N, d) vector
        N, _ = f.shape
        return f[:, OUTPUT_VARIABLE].reshape((N, 1))

    def lorenz_g(x):
        g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
        N, d, _ = g.shape
        return g[:, OUTPUT_VARIABLE, :].reshape((N, 1, d))

    _evals_dir = (
        TRAIN_EVALS_ROOT
        / f"lorenz96_{DIM}d_output_var{OUTPUT_VARIABLE}_F{FORCING}_tf{FINAL_TIME}_tol{TOL}"
    )
    Problem_experiment_setup(
        json_fpath=TRAIN_EVALS_ROOT
        / f"ics26_lorenz96_{DIM}d_output_var{OUTPUT_VARIABLE}_F{FORCING}_tf{FINAL_TIME}_tol{TOL}.json",
        DIM=DIM,
        F_DIM=1,
        DOMAIN=[-5.0, 5.0],
        TOL=TOL,
        f_fn=lorenz_f,
        g_fn=lorenz_g,
        N_TRAIN=get_N_TRAIN(DIM),
        N_INIT_CLUSTERS=get_N_INIT_CLUSTERS(DIM),
        N_MIN_PER_CLUSTER=get_N_MIN_PER_CLUSTER(DIM),
        MAX_CLUSTERS_EST=100_000,
        N_TEST=get_N_TRAIN(DIM) // 2,
        K_NEIGHBOURS=K_NEIGHBOURS,
        ERROR_METRIC=ERROR_METRIC,
        STORE_TRAIN_EVALS=True,
        TRAIN_EVALS_DIR=_evals_dir,
        TRAIN_EVALS_PREFIX=_prefix,
        CLUSTERING_MATRIX='Cov_Lvl_curves',
        SAVE_TEST_DATA=True,
        SAVE_TRAINED_DATA=True,
    )
    for _p in ["_f", "_g"]:
        consolidate_evaluations(
            save_dir=_evals_dir,
            prefix=_prefix + _p,
            output_path=_evals_dir
            / f"lorenz96_{DIM}d_F{FORCING}_tf{FINAL_TIME}_tol{TOL}{_p}.npz",
            delete_individual_files_after=True,
        )


def ics_lorenz_experiment() -> Path:
    """
    Lorenz surrogate for a SINGLE time step.
    We are not worrying about sampling along manifolds in this case.
    Imagine this is the very first time step in a string of simulations.
    """
    FORCING = 1
    FINAL_TIME = 0.05

    def f(x):
        f = lorenz96_flowmap(x, FORCING, FINAL_TIME)  # (N, d) vector
        N, _ = f.shape
        return f[:, 0].reshape((N, 1))

    def grad_f(x):
        g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
        N, d, _ = g.shape
        return g[:, 0, :].reshape((N, 1, d))

    # def oracle(x):
    #     """
    #     Compute the flowmap, and the gradient of the flowmap, for the given input x,
    #     and return for the specified output variable.
    #     """
    #     N, d = x.shape
    #     f, g = lorenz96_flowmap_oracle_first_component(x, FORCING, FINAL_TIME)
    #     return f.reshape(N, 1), g.reshape(N, 1, d)

    # def f(x):
    #     f = lorenz96_flowmap(x, FORCING, FINAL_TIME)
    #     return f

    # def grad_f(x):
    #     g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
    #     return g

    # def oracle(x):
    #     f, g = lorenz96_flowmap_oracle(x, FORCING, FINAL_TIME)
    #     return f, g

    MAX_CLUSTERS_EST = 20_000
    K_NEIGHBOURS = 1
    # EXP_DIR = (
    #     ICS26_DIR / f"ics_lorenz_finaltime{FINAL_TIME}_outputvar0_forcing{FORCING}"
    # )
    EXP_DIR = (
        ICS26_DIR / f"lorenz_cov_lvl_curves_1d_out_no_gradient"
    )
    # del_all_error_files_in_dir(EXP_DIR)
    # create_summary_dataframe(EXP_DIR)

    DIMs = [5]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Lorenz Experiments"):
        json_path = (
            EXP_DIR
            / f"ics26_lorenz_dim{DIM}_tol{TOL}_err{ERROR_METRIC}_outputvar0.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = min(get_N_TRAIN(DIM), int(1e6))
        N_TEST = 50_000
        N_INIT_CLUSTERS = min(get_N_INIT_CLUSTERS(DIM), 1000)
        N_MIN_PER_CLUSTER = min(get_N_MIN_PER_CLUSTER(DIM), 500)
        print(
            f"N_TRAIN: {N_TRAIN:,}, N_INIT_CLUSTERS: {N_INIT_CLUSTERS:,}, N_MIN_PER_CLUSTER: {N_MIN_PER_CLUSTER:,}"
        )

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-5.0, 5.0],
            f_fn=f,
            g_fn=grad_f,
            # oracle_fn=oracle,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX="None",
            IGNORE_GRADIENT=True,
            PROFILE_TIMING=True,
        )
        create_summary_dataframe(EXP_DIR)

    return EXP_DIR




def vanderpol_experiment() -> Path:
    """
    Van der Pol surrogate for a single time step.
    We are not worrying about sampling along manifolds in this case.
    Imagine this is the very first time step in a string of simulations.
    """

    MU = 5.0
    FINAL_TIME = 5.0

    def f(x):
        f = vanderpol_flowmap(x, MU, FINAL_TIME)  # (N, d) vector
        N, _ = f.shape
        return f[:, 0].reshape((N, 1))

    def grad_f(x):
        g = vanderpol_flowmap_grad(x, MU, FINAL_TIME)
        N, d, _ = g.shape
        return g[:, 0, :].reshape((N, 1, d))

    def oracle_f(x):
        return vanderpol_flowmap_oracle_first_component(x, MU, FINAL_TIME)

    MAX_CLUSTERS_EST = 20_000
    K_NEIGHBOURS = 1
    EXP_DIR = (
        ICS26_DIR / f"vanderpol"
    )
    # del_all_error_files_in_dir(EXP_DIR)
    # create_summary_dataframe(EXP_DIR)

    TOLs = [0.01]
    ERROR_METRICs = ["Linf-global"]

    combos = list(product(TOLs, ERROR_METRICs))
    for TOL, ERROR_METRIC in tqdm(combos, desc="Lorenz Experiments"):
        json_path = (
            EXP_DIR
            / f"vanderpol_tol{TOL}_err{ERROR_METRIC}_outputvar1.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        N_TRAIN = min(get_N_TRAIN(2), int(1e6))
        N_TEST = 50_000
        N_INIT_CLUSTERS = min(get_N_INIT_CLUSTERS(2), 1000)
        N_MIN_PER_CLUSTER = min(get_N_MIN_PER_CLUSTER(2), 500)
        print(
            f"N_TRAIN: {N_TRAIN:,}, N_INIT_CLUSTERS: {N_INIT_CLUSTERS:,}, N_MIN_PER_CLUSTER: {N_MIN_PER_CLUSTER:,}"
        )

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=2,
            F_DIM=1,
            DOMAIN=[-4.0, 4.0],
            f_fn=f,
            g_fn=grad_f,
            oracle_fn=oracle_f,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            CLUSTERING_MATRIX="Hessian_LSQ",
            IGNORE_GRADIENT=False,
            PROFILE_TIMING=True,
            MAX_LEVELS=30
        )
        create_summary_dataframe(EXP_DIR)

    return EXP_DIR




if __name__ == "__main__":
    from src import IS_CUPY_BACKEND
    from src.logging_utils import setup_logging

    setup_logging()

    if IS_CUPY_BACKEND:
        
        print("Running on GPU node.")
        dir = ics_aniso_gaussian_experiment()
        create_summary_dataframe(dir)

    else:
        print("Running on CPU node.")
