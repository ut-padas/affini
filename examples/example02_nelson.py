"""
This example builds a surrogate for the Nelson flow map in two dimensions.
It uses the workflow described in the paper, with training and testing data
read in from a file instead of generated randomly. This example uses no parallelism.
"""

import sys
from pathlib import Path

import numpy as np

from src import xp
from src.adaptive_clustering_utils import SurrogateMode
from src.heuristic_parameters import get_MIN_PTS_PER_CLUSTER, get_NUM_INITIAL_CLUSTERS
from src.surrogate import AFINNISurrogate

# KIDApy's examples/ is added directly (rather than importing examples.hydro) so it doesn't collide with this repo's examples/
KIDAPY_ROOT = Path(__file__).resolve().parent.parent / "libs" / "KIDApy"
sys.path[:0] = [str(KIDAPY_ROOT), str(KIDAPY_ROOT / "examples")]

from hydro import HydroData
from parser import Network, load_abundances

NETWORK_PATH    = KIDAPY_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = KIDAPY_ROOT / "networks" / "nelson" / "abundances.in"
TRACE_FILE      = KIDAPY_ROOT / "datasets" / "grav_collapse" / "R0.1_M6_trace_cells.npy"

net = Network(grains=True, self_shielding=True, pd_fac=True)
net.load_from_disk(str(NETWORK_PATH))
net.drop_passive_species()


def nelson_flow_map(X):
    """
    Evaluates the Nelson flow map at the points X.

    TODO: implement.
    """
    raise NotImplementedError("nelson_flow_map is not implemented yet")


def get_nelson_train_test_data(m, test_frac=1/3, seed=0):
    """
    Reads in the first m rows of KIDApy's gravitational-collapse dataset, each starting from
    the network's initial abundances (as in KIDApy's nelson_batch_solve.demo_inputs), and
    splits them randomly into training and testing sets.

    Each input row is [states (14 species abundances), params (nH, T, Tgrain, Av, uv1, uv3, uv4)].
    Outputs are evaluated with nelson_flow_map.

    Returns X_train, f_train, X_test, f_test.
    """
    h = HydroData(three_uv_bands=True)
    h.load_hydro_data(str(TRACE_FILE), max_tracers=1, abundances=ABUNDANCES_PATH)
    params = h.params[:m]

    abund = load_abundances(str(ABUNDANCES_PATH))
    x0 = np.zeros(len(net.species))
    for name, val in abund.items():
        if name in net.species_map:
            x0[net.species_map[name]] = val
    states = np.tile(x0, (params.shape[0], 1))

    X = xp.asarray(np.hstack([states, params]))
    perm = np.random.default_rng(seed).permutation(X.shape[0])
    n_test = int(round(test_frac * X.shape[0]))
    X_test, X_train = X[perm[:n_test]], X[perm[n_test:]]

    return X_train, nelson_flow_map(X_train), X_test, nelson_flow_map(X_test)


if __name__ == "__main__":

    # Parameters for training
    SEED = 67
    N_ROWS = 804  # all rows of the single tracer in the dataset
    DIM = 2
    DOMAIN = xp.array([-2.0, 3.0])  # TODO: set to the domain of the Nelson data
    TOL = 0.02
    N_init_clusters = get_NUM_INITIAL_CLUSTERS(DIM)
    min_ppc = get_MIN_PTS_PER_CLUSTER(DIM)

    # --- Load data ---
    X_train, f_train, X_test, f_test_exact = get_nelson_train_test_data(N_ROWS, seed=SEED)

    # --- Build AFINNI surrogate ---
    surrogate = AFINNISurrogate()
    surrogate.train(
        X_train = X_train,
        Y_train = f_train,
        f_fn = nelson_flow_map,
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
    f_test_pred = surrogate.predict(X_test)
    ptwise_errs = xp.abs(f_test_exact - f_test_pred).flatten()
    l2_rel_err = xp.linalg.norm(ptwise_errs) / xp.linalg.norm(f_test_exact)
    print(f"L2 Relative Global Error: {l2_rel_err}")

    # --- Plot trajectories ---
