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
        )

    return EXP_DIR


def ics_shifted_gaussian_experiment() -> Path:
    CLUSTERING_VERSION = "v2"
    MAX_CLUSTERS_EST = 100_000
    K_NEIGHBOURS = 1

    EXP_DIR = ICS26_DIR / "ics_shifted_gaussians"
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 5]
    TOLs = [0.1, 0.05, 0.02, 0.01, 0.005]
    ERROR_METRICs = ["Linf-global", "Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM >= 4 and TOL <= 5e-4:
            print(
                f"Skipping experiment for DIM={DIM} with very small TOL={TOL} due to resource constraints."
            )
            continue
        json_path = (
            EXP_DIR / f"ics26_sgaussian_dim{DIM}_tol{TOL}_metric{ERROR_METRIC}.json"
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
            USE_HESSIAN_INFORMED_SPLIT=True,
            IGNORE_GRADIENT=False,
            STORE_TRAIN_EVALS=False,
        )

    return EXP_DIR


if __name__ == "__main__":
    from src.logging_utils import setup_logging

    setup_logging()

    dir1 = ics_quadratic_experiment()
    create_summary_dataframe(dir1)

    dir2 = ics_shifted_gaussian_experiment()
    create_summary_dataframe(dir2)
