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
from src.problems.lorenz96_flowmap import lorenz96_flowmap, lorenz96_flowmap_grad
from src.problems.quadratic import quadratic, quadratic_grad
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


def ics_quadratic_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "ics_quadratic"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 5, 10]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local", "Linf-global"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if (
            DIM == 10
            and (TOL <= 0.01 and ERROR_METRIC == "Linf-local")
            or (TOL <= 0.005 and ERROR_METRIC == "Linf-global")
        ):
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        json_path = (
            EXP_DIR / f"ics26_quadratic_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
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
            DOMAIN=[-2, 2],
            f_fn=quadratic,
            g_fn=quadratic_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
            STORE_TRAIN_EVALS=False,
        )

    return EXP_DIR


def ics_quadratic_experiment_no_hess_no_grad() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "ics_quadratic_no_hessian_no_gradient"
    create_summary_dataframe(EXP_DIR)
    DIMs = [5]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local"]
    case = [
        (True, False),
        (False, False),
        (False, True),
    ]  # (use_hessian, ignore_gradient)

    combos = list(product(DIMs, TOLs, ERROR_METRICs, case))
    for DIM, TOL, ERROR_METRIC, (USE_HESSIAN_INFORMED_SPLIT, IGNORE_GRADIENT) in tqdm(
        combos, desc="Experiments"
    ):
        if (
            DIM == 10
            and (TOL <= 0.01 and ERROR_METRIC == "Linf-local")
            or (TOL <= 0.005 and ERROR_METRIC == "Linf-global")
        ):
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        json_path = (
            EXP_DIR
            / f"ics26_quadratic_dim{DIM}_tol{TOL}_err{ERROR_METRIC}_nohess{USE_HESSIAN_INFORMED_SPLIT}_nograd{IGNORE_GRADIENT}.json"
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
            DOMAIN=[-2, 2],
            f_fn=quadratic,
            g_fn=quadratic_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
            IGNORE_GRADIENT=IGNORE_GRADIENT,
            STORE_TRAIN_EVALS=False,
        )

    return EXP_DIR


def ics_quadratic_runtime_scalability_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "ics_quadratic_runtime_scalability"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)
    DIMs = [1, 2, 4, 8, 16, 20, 32, 40, 64]
    TOLs = [0.1]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM >= 16 and TOL <= 0.05:
            print(
                f"Skipping: DIM={DIM}, TOL={TOL}, ERROR_METRIC={ERROR_METRIC} due to resource constraints."
            )
            continue

        base = 1.01
        N_TRAIN = get_N_TRAIN(DIM, base=base, factor=100_000)
        N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM, base=base, factor=100)
        N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM, base=base, factor=100)
        N_TEST = max(N_TRAIN // 2, 500)

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
            / f"ics26_quadratic_dim{DIM}_tol{TOL}_ntrain{N_TRAIN}_batch{BATCH_SIZE}_err{ERROR_METRIC}.json"
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
            DOMAIN=[-2, 2],
            f_fn=quadratic,
            g_fn=quadratic_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
            STORE_TRAIN_EVALS=False,
            BATCH_SIZE=BATCH_SIZE,
            PROFILE_TIMING=True,
        )

    return EXP_DIR


def ics_shifted_gaussians_experiment() -> Path:
    CLUSTERING_VERSION = "v2"
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1

    EXP_DIR = ICS26_DIR / "ics_shifted_gaussians"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 5]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-local", "Linf-global"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM >= 4 and TOL <= 5e-4:
            print(
                f"Skipping experiment for DIM={DIM} with very small TOL={TOL} due to resource constraints."
            )
            continue
        json_path = (
            EXP_DIR / f"ics26_sgaussians_dim{DIM}_tol{TOL}_metric{ERROR_METRIC}.json"
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

        # Set up partial functions for shifted gaussians with centers
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
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2, 2],
            f_fn=shifted_gaussians_partial,
            g_fn=shifted_gaussians_grad_partial,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=False,
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
        )

    return EXP_DIR


def ics_shifted_gaussians_no_hess_experiment() -> Path:
    CLUSTERING_VERSION = "v2"
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1

    EXP_DIR = ICS26_DIR / "ics_shifted_gaussians_no_hessian"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 5]
    # TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    TOLs = [
        0.2,
        0.19,
        0.18,
        0.17,
        0.16,
        0.15,
        0.14,
        0.13,
        0.12,
        0.11,
        0.1,
        0.09,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.03,
        0.02,
        0.01,
    ]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM >= 4 and TOL <= 5e-4:
            print(
                f"Skipping experiment for DIM={DIM} with very small TOL={TOL} due to resource constraints."
            )
            continue
        json_path = (
            EXP_DIR / f"ics26_sgaussians_dim{DIM}_tol{TOL}_metric{ERROR_METRIC}.json"
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

        # Set up partial functions for shifted gaussians with centers
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
            json_fpath=json_path,
            DIM=DIM,
            F_DIM=1,
            DOMAIN=[-2, 2],
            f_fn=shifted_gaussians_partial,
            g_fn=shifted_gaussians_grad_partial,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=False,
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
        )

    return EXP_DIR


def ics_rosenbrock_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "ics_rosenbrock"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)

    DIMs = [2, 5, 10]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-global", "Linf-local"]

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
        N_TEST = N_TRAIN // 2
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
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


def ics_rosenbrock_n_min_per_cluster_experiment() -> Path:
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

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
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


def generate_train_data_for_neural_network() -> None:
    TRAIN_EVALS_ROOT = ICS26_DIR / "ics_neural_network_train_evals"
    TOL = 0.05
    ERROR_METRIC = "Linf-local"
    USE_HESSIAN_INFORMED_SPLIT = True
    CLUSTERING_VERSION = "v2"
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
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,
            STORE_TRAIN_EVALS=True,
            TRAIN_EVALS_DIR=_evals_dir,
            TRAIN_EVALS_PREFIX=_prefix,
            USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
        )
        for _p in ["_f", "_g"]:
            consolidate_evaluations(
                save_dir=_evals_dir,
                prefix=_prefix + _p,
                output_path=_evals_dir / f"shifted_gaussians_{DIM}d_tol{TOL}{_p}.npz",
                delete_individual_files_after=True,
            )

    # 5D Rosenbrock
    DIM = 5
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
        CLUSTERING_VERSION=CLUSTERING_VERSION,
        ERROR_METRIC=ERROR_METRIC,
        STORE_TRAIN_EVALS=True,
        TRAIN_EVALS_DIR=_evals_dir,
        TRAIN_EVALS_PREFIX=_prefix,
        USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
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
    lorenz_f = partial(lorenz96_flowmap, forcing=FORCING, tf=FINAL_TIME)
    lorenz_g = partial(lorenz96_flowmap_grad, forcing=FORCING, tf=FINAL_TIME)
    _evals_dir = (
        TRAIN_EVALS_ROOT / f"lorenz96_{DIM}d_F{FORCING}_tf{FINAL_TIME}_tol{TOL}"
    )
    Problem_experiment_setup(
        json_fpath=TRAIN_EVALS_ROOT
        / f"ics26_lorenz96_{DIM}d_F{FORCING}_tf{FINAL_TIME}_tol{TOL}_train_evals.json",
        DIM=DIM,
        F_DIM=DIM,
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
        CLUSTERING_VERSION=CLUSTERING_VERSION,
        ERROR_METRIC=ERROR_METRIC,
        STORE_TRAIN_EVALS=True,
        TRAIN_EVALS_DIR=_evals_dir,
        TRAIN_EVALS_PREFIX=_prefix,
        USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
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

    # We must modify the functions to include forcing and tf as constants.

    FORCING = 1
    FINAL_TIME = 0.1
    # OUTPUT_VARIABLE = 0

    # For now, just do gradient in the first variable.
    # This way the output of the function is 1d.
    def f(x):
        f = lorenz96_flowmap(x, FORCING, FINAL_TIME)  # (N, d) vector
        # N, d = f.shape
        # return f[:, OUTPUT_VARIABLE].reshape((N, 1))
        return f

    def grad_f(x):
        g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
        # N, d, _ = g.shape
        # return g[:, OUTPUT_VARIABLE, :].reshape((N, 1, d))
        return g

    MAX_CLUSTERS_EST = 20_000

    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "lorenz_all_d_out_tf_0.1_F_1_v2"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)

    DIMs = [5, 10]
    TOLs = [
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.08,
        0.05,
        0.04,
        0.02,
        0.01,
        0.005,
    ]
    ERROR_METRICs = ["Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Lorenz Experiments"):

        # Resource constraints checks
        if DIM == 10 and TOL < 0.01:
            print(
                f"Skipping experiment for DIM={DIM}, TOL={TOL} due to high computational cost."
            )
            continue

        json_path = EXP_DIR / f"ics26_lorenz_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
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
            DOMAIN=[-5.0, 5.0],
            f_fn=f,
            g_fn=grad_f,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


if __name__ == "__main__":
    from src.logging_utils import setup_logging

    setup_logging()

    # dir1 = ics_quadratic_experiment()
    # create_summary_dataframe(dir1)

    # dir2 = ics_shifted_gaussians_no_hess_experiment()
    # create_summary_dataframe(dir2)

    # dir2 = ics_shifted_gaussians_experiment()
    # create_summary_dataframe(dir2)

    # dir3 = ics_rosenbrock_experiment()
    # create_summary_dataframe(dir3)

    # dir4 = ics_rosenbrock_n_min_per_cluster_experiment()
    # create_summary_dataframe(dir4)

    # dir4 = ics_lorenz_experiment()
    # create_summary_dataframe(dir4)

    # dir5 = ics_quadratic_runtime_scalability_experiment()
    # create_summary_dataframe(dir5)

    dir6 = ics_quadratic_experiment_no_hess_no_grad()
    create_summary_dataframe(dir6)

    # generate_train_data_for_neural_network()
