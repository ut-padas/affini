"""
This example builds a surrogate for the Rosenbrock function in two dimensions.
It uses the workflow described in the paper. This example uses no parallelism.
"""

from pathlib import Path
from src import xp
from src.adaptive_clustering_utils import SurrogateMode
from src.heuristic_parameters import get_MIN_PTS_PER_CLUSTER, get_N_TRAIN, get_NUM_INITIAL_CLUSTERS
from src.surrogate import AFINNISurrogate
from src.problems.rosenbrock import rosenbrock
from src.plotter_utils import plot_adaptive_clustering_results


if __name__ == "__main__":

    # Parameters for training
    SEED = 67
    DIM = 2
    N_TRAIN = get_N_TRAIN(DIM)
    N_TEST = N_TRAIN // 2
    DOMAIN = xp.array([-2.0, 3.0])
    TOL = 0.02
    N_init_clusters = get_NUM_INITIAL_CLUSTERS(DIM)
    min_ppc = get_MIN_PTS_PER_CLUSTER(DIM)

    # --- Generate initial data ---
    xp.random.seed(SEED)
    X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
    f_train = rosenbrock(X_train)

    # --- Build AFFINI surrogate ---
    surrogate = AFINNISurrogate()
    surrogate.train(
        X_train = X_train,
        Y_train = f_train,
        f_fn = rosenbrock,
        mode = SurrogateMode.AFFINE_COV_HESS,
        TOL = TOL,
        ERROR_METRIC = 'Linf-local',
        NUM_INITIAL_CLUSTERS = N_init_clusters,
        MIN_PTS_PER_CLUSTER = min_ppc,
        SEED = SEED,
        STORE_AFFINE_FIT = True,
        USE_WORK_STEALING = True,
        PROFILE_TIMING = True
    )

    # --- Compute errors ---
    X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))
    f_test_exact = rosenbrock(X_test)
    f_test_pred = surrogate.predict(X_test)
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
