"""
This example builds a surrogate for the Rosenbrock function in two dimensions.
It uses the workflow described in the paper. It runs as one process, or on several
MPI ranks (`mpiexec -n 4 python -m examples.example01_rosenbrock_afinni`), where rank 0 reports.
"""

from pathlib import Path

from src import get_mpi_rank, xp
from src.components.modes import SurrogateMode
from src.config import (
    TrainingConfig,
    default_min_pts_per_cluster,
    default_n_train,
    default_num_initial_clusters,
)
from src.core.evaluation import Evaluator
from src.plotting import plot_adaptive_clustering_results
from src.problems.rosenbrock import rosenbrock
from src.surrogate import AFINNISurrogate

if __name__ == "__main__":

    # Parameters for training
    SEED = 67
    DIM = 2
    N_TRAIN = default_n_train(DIM)
    N_TEST = N_TRAIN // 2
    DOMAIN = xp.array([-2.0, 3.0])
    TOL = 0.02
    N_init_clusters = default_num_initial_clusters(DIM)
    min_ppc = default_min_pts_per_cluster(DIM)

    # --- Generate initial data ---
    xp.random.seed(SEED)
    X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
    f_train = rosenbrock(X_train)

    # --- Build AFINNI surrogate ---
    surrogate = AFINNISurrogate()
    config = TrainingConfig(
        mode = SurrogateMode.AFFINE_COV_HESS,
        atol = TOL,
        rtol = TOL,
        error_metric = 'Linf',
        num_initial_clusters = N_init_clusters,
        min_pts_per_cluster = min_ppc,
        seed = SEED,
        use_work_stealing = True,
        profile_timing = True,
    )
    surrogate.train(X_train, config, Evaluator(f_fn=rosenbrock), Y_train=f_train)

    # --- Compute errors ---
    X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))
    f_test_pred, _ = surrogate.predict(X_test)  # (predictions, trust-region mask)
    # Under MPI, predictions come back on rank 0 only; the other ranks are done.
    if get_mpi_rank() != 0:
        raise SystemExit(0)
    f_test_exact = rosenbrock(X_test).reshape(f_test_pred.shape)
    ptwise_errs = xp.abs(f_test_exact - f_test_pred).flatten()
    l2_rel_err = xp.linalg.norm(ptwise_errs) / xp.linalg.norm(f_test_exact)
    print(f"L2 Relative Global Error: {l2_rel_err}")

    # --- Plot clusters ---
    plot_adaptive_clustering_results(
        centroids = surrogate.X_src,
        Q = surrogate.Qs_src,
        domain = DOMAIN,
        f_fn = rosenbrock,
        img_fpath = Path("rosenbrock_adaptive_clustering_results.png"),
        title = f"AFINNI on Rosenbrock Function, tol={TOL}"
    )
