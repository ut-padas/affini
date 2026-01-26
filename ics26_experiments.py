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
from src.problems.rastrigin import rastrigin, rastrigin_grad
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.problems.shifted_gaussians import (
    CENTERS_2D,
    CENTERS_3D,
    CENTERS_4D,
    shifted_gaussians,
    shifted_gaussians_grad,
)
from src.problems.lorenz96_flowmap import lorenz96_flowmap, lorenz96_flowmap_grad


def quadratic_experiment_with_v2() -> Path:
    N_TRAIN = 100_000
    N_TEST = 10_000
    N_INIT_CLUSTERS = 1
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 10_000
    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "quadratic_v2"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)
    DIMs = [2, 3, 4, 5, 8, 10]
    TOLs = [i / 100.0 for i in [20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05]]
    ERROR_METRICs = ["L2", "Linf-global", "Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM > 5 and TOL <= 1e-4:
            print(
                f"Skipping experiment for DIM={DIM}, TOL={TOL} due to high computational cost."
            )
            continue
        elif DIM >= 8 and TOL <= 1e-2:
            print(
                f"Skipping experiment for DIM={DIM}, TOL={TOL} due to high computational cost."
            )
            continue
        elif ERROR_METRIC == "L2" and TOL <= 0.2 and DIM >= 4:
            print(
                f"Skipping experiment for ERROR_METRIC={ERROR_METRIC}, TOL={TOL} and DIM={DIM} due to high computational cost."
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

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
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
            SYNTH_BUDGET_FACTOR=1000,
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


def gaussian_experiment_with_v2() -> Path:
    CLUSTERING_VERSION = "v2"
    N_INIT_CLUSTERS = 4
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 10_000
    K_NEIGHBOURS = 1

    EXP_DIR = ICS26_DIR / "shifted_gaussians_v2"
    create_summary_dataframe(EXP_DIR)
    del_all_error_files_in_dir(EXP_DIR)
    DIMs = [2, 3, 4]
    TOLs = [i / 100.0 for i in [50, 35, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01]]
    ERROR_METRICs = ["Linf-global", "Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Experiments"):
        if DIM >= 3 and TOL <= 1e-3:
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

        if DIM == 2:
            N_TRAIN = 200_000
            N_TEST = 50_000
        else:
            N_TRAIN = 300_000
            N_TEST = 100_000

        # Set up partial functions for shifted gaussians with centers
        shifted_gaussians_partial = partial(
            shifted_gaussians,
            centers={
                2: CENTERS_2D,
                3: CENTERS_3D,
                4: CENTERS_4D,
            }[DIM],
        )
        shifted_gaussians_grad_partial = partial(
            shifted_gaussians_grad,
            centers={
                2: CENTERS_2D,
                3: CENTERS_3D,
                4: CENTERS_4D,
            }[DIM],
        )

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
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
            SYNTH_BUDGET_FACTOR=500,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


def rastrigin_experiment_with_v2() -> Path:
    # Rastrigin is complex (oscillatory), so we need more initial clusters and data
    N_TRAIN = 300_000
    N_TEST = 50_000
    N_INIT_CLUSTERS = 50
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 50_000
    K_NEIGHBOURS = 5  # Higher K needed to bridge gaps between the many "grid" clusters
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "rastrigin_v2"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)

    # Rastrigin becomes extremely difficult in high dimensions for local surrogates
    # due to the exponential growth of local minima. We stick to 2, 3, 4.
    DIMs = [2, 3, 4]

    # Global Max is approx 80 for d=2, 120 for d=3.
    # TOLs: 0.5 (50%) down to 0.005 (0.5%)
    TOLs = [i / 100.0 for i in [50, 25, 10, 5, 2, 1, 0.5]]

    # L2 is excluded by default as it tends to over-segment the oscillating "floor"
    ERROR_METRICs = ["Linf-global", "Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Rastrigin Experiments"):

        # Resource constraints check
        if DIM >= 3 and TOL <= 1e-2:
            print(f"Skipping DIM={DIM}, TOL={TOL} (High Compute Cost)")
            continue

        json_path = (
            EXP_DIR / f"ics26_rastrigin_dim{DIM}_tol{TOL}_err{ERROR_METRIC}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue
        print(f"Running experiment: {json_path}")

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            DOMAIN=[-5.12, 5.12],  # Standard Rastrigin Domain
            f_fn=rastrigin,
            g_fn=rastrigin_grad,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CLUSTERING_VERSION=CLUSTERING_VERSION,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            SYNTH_BUDGET_FACTOR=1000,  # High budget for complex padding
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


def rosenbrock_experiment_with_v2() -> Path:
    # Rosenbrock has a narrow, curved valley.
    # We need slightly more data than Quadratic, but less than Rastrigin.
    N_TRAIN = 200_000
    N_TEST = 50_000

    # We start with a few clusters to help isolate the valley early on
    N_INIT_CLUSTERS = 50
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 20_000

    # K=2 helps bridge the linear approximations along the curve better than K=1
    K_NEIGHBOURS = 2
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "rosenbrock_v2"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)

    DIMs = [2, 5, 10]

    # Rosenbrock scale is 100, so gradients are steep.
    # We test tolerances from 50% down to 0.1%
    TOLs = [i / 100.0 for i in [50, 20, 10, 5, 2, 1, 0.5, 0.1]]

    ERROR_METRICs = ["Linf-global", "Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Rosenbrock Experiments"):

        # Resource constraints checks
        if DIM == 10 and TOL <= 0.005:
            print(
                f"Skipping experiment for DIM={DIM}, TOL={TOL} due to high computational cost."
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

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            # Domain [-2, 3] captures the global min at 1.0 and the steep walls
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
            SYNTH_BUDGET_FACTOR=1000,
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR


"""
Lorenz surrogate for a SINGLE time step.
We are not worrying about sampling along manifolds in this case.
Imagine this is the very first time step in a string of simulations.
"""
def lorenz_experiment_with_v2() -> Path:

    # We must modify the functions to include forcing and tf as constants.
    FORCING = 1
    FINAL_TIME = 0.1
    OUTPUT_VARIABLE = 0
    # For now, just do gradient in the first variable.
    # This way the output of the function is 1d.
    def f(x):
        f = lorenz96_flowmap(x, FORCING, FINAL_TIME)
        return f[:,OUTPUT_VARIABLE]
    def grad_f(x):
        g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
        return g[:,OUTPUT_VARIABLE,:]

    N_TRAIN = 200_000
    N_TEST = 50_000

    N_INIT_CLUSTERS = 50
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 20_000

    K_NEIGHBOURS = 1
    CLUSTERING_VERSION = "v2"

    EXP_DIR = ICS26_DIR / "lorenz_v2"
    del_all_error_files_in_dir(EXP_DIR)
    create_summary_dataframe(EXP_DIR)

    DIMs = [4, 5, 6, 10]
    TOLs = [i / 100.0 for i in [50, 20, 10, 5, 2, 1]]
    ERROR_METRICs = ["Linf-global", "Linf-local"]

    combos = list(product(DIMs, TOLs, ERROR_METRICs))
    for DIM, TOL, ERROR_METRIC in tqdm(combos, desc="Lorenz Experiments"):

        # Resource constraints checks
        if DIM == 10 and TOL <= 0.005:
            print(
                f"Skipping experiment for DIM={DIM}, TOL={TOL} due to high computational cost."
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

        Problem_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            DOMAIN=[-10.0, 10.0],
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
            SYNTH_BUDGET_FACTOR=1000,
            USE_HESSIAN_INFORMED_SPLIT=True,
        )

    return EXP_DIR








if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()

    # dir1 = quadratic_experiment_with_v2()
    # create_summary_dataframe(dir1)

    # dir2 = gaussian_experiment_with_v2()
    # create_summary_dataframe(dir2)

    # dir3 = rastrigin_experiment_with_v2()
    # create_summary_dataframe(dir3)

    dir4 = lorenz_experiment_with_v2()
    create_summary_dataframe(dir4)
