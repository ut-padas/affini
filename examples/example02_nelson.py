"""
This example builds a surrogate for the Nelson flow map in two dimensions.
It uses the workflow described in the paper, with training and testing data
read in from a file instead of generated randomly. This example uses no parallelism.
"""

from functools import partial
import sys
from pathlib import Path

import numpy as np

from src import _to_device, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.heuristic_parameters import get_MIN_PTS_PER_CLUSTER, get_NUM_INITIAL_CLUSTERS
from src.surrogate import AFINNISurrogate

# KIDApy's modules import each other by bare name (e.g. parser.py does `import pdfac`), so its root must be on sys.path
KIDAPY_ROOT = Path(__file__).resolve().parent.parent / "libs" / "KIDApy"
sys.path.insert(0, str(KIDAPY_ROOT))

from parser import Network
from solver import QuadraticSolver

NETWORK_PATH    = KIDAPY_ROOT / "networks" / "nelson" / "gas_reactions.in"

YEAR = 3600.0 * 24.0 * 365.25

ATOL = 1e-20
RTOL = 1e-3

BAND_FLOOR = 1e-30    # pdfac takes log10 of the bands, so they must stay positive
LOG_FLOOR  = 1e-30    # floor before taking log10 of inputs/outputs, since some abundances are zero

# Environment columns, in the order HydroData(three_uv_bands=True) returns them
ENV_COLS = ["nH", "T", "Tgrain", "Av", "uv1", "uv3", "uv4"]

net = Network(grains=True, self_shielding=True, pd_fac=True)
net.load_from_disk(str(NETWORK_PATH))
net.drop_passive_species()


def params_to_env(p):
    """One row of params -> the env dict the network expects."""
    env = dict(zip(ENV_COLS, map(float, p)))
    env["Tcap_2body"] = True
    for band in ("uv1", "uv3", "uv4"):
        env[band] = max(env[band], BAND_FLOOR)
    return env


def nelson_flow_map(states, params, dt=250.0 * YEAR, n_workers=1, atol=ATOL, rtol=RTOL, verbose=False):
    """
    Evaluates the Nelson flow map, following KIDApy's nelson_batch_solve.batch_solve:
    each row of states is integrated for a time dt at the fixed environment in the same row of params.

    Parameters
    ----------
    states : np.ndarray, shape (M, 14)
        Initial abundances per H nucleus, columns in net.species order.
    params : np.ndarray, shape (M, 7)
        Environment per row, columns ENV_COLS.
    dt : float
        Integration time [s], the same for every row. Defaults to 250 years.
    n_workers : int
        Number of worker processes. 1 runs serially.

    Returns
    -------
    final : np.ndarray, shape (M, 14)
        Abundances after dt. Rows whose solve failed are NaN.
    """
    states = np.asarray(states, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    N = len(net.species)
    if states.ndim != 2 or states.shape[1] != N:
        raise ValueError(f"states must have shape (M, {N}); got {states.shape}")
    if params.ndim != 2 or params.shape[1] != len(ENV_COLS):
        raise ValueError(f"params must have shape (M, {len(ENV_COLS)}); got {params.shape}")
    if states.shape[0] != params.shape[0]:
        raise ValueError(f"states and params need the same number of rows; got {states.shape[0]} and {params.shape[0]}")

    tensors = []
    for p in params:
        A, B = net.get_operators_vectorized(params_to_env(p))
        tensors.append((A, B))

    results, failures = QuadraticSolver().solve_multiple(
        tensors,
        t_span=(0.0, dt),
        x0=states,
        atol=atol,
        rtol=rtol,
        method="BDF",
        t_eval=[dt],
        n_workers=n_workers,
        verbose=verbose,
        return_failures=True,
    )

    final = np.full_like(states, np.nan)
    for i, result in enumerate(results):
        if result is not None:
            final[i] = result[1][:, -1]
    for i, exc in failures:
        print(f"[WARN] nelson_flow_map: row {i} failed: {exc}")
    return final


def to_log(a):
    """log10, with values floored at LOG_FLOOR so zero abundances stay finite."""
    return np.log10(np.maximum(a, LOG_FLOOR))


def nelson_flow_map_wrapper(X, output=3, dt=250.0 * YEAR, n_workers=8):
    """
    Evaluates a single component of the Nelson flow map at the log-scaled points X.

    Parameters
    ----------
    X : array, shape (M, 21)
        Each row is log10 of [states (14 abundances, in net.species order), params (ENV_COLS)].
    output : int
        Column of the output state to return. Defaults to 3 (CO).
    dt : float
        Integration time [s]. Defaults to 250 years.

    Returns
    -------
    array, shape (M, 1)
        log10 of the output abundance (floored at LOG_FLOOR).
    """
    X = np.asarray(_to_host(X), dtype=np.float64)
    N = len(net.species)
    if X.ndim != 2 or X.shape[1] != N + len(ENV_COLS):
        raise ValueError(f"X must have shape (M, {N + len(ENV_COLS)}); got {X.shape}")
    X = 10.0 ** X
    final = nelson_flow_map(X[:, :N], X[:, N:], dt=dt, n_workers=n_workers)
    return _to_device(to_log(final[:, [output]]))


# Column layout of the collapse-chemistry output files (see the dataset's README)
FILE_PARAM_COLS   = slice(2, 9)     # nH, T, Tgrain, Av_pdfac, uv1, uv3, uv4 -- same order as ENV_COLS
FILE_SPECIES_COLS = slice(9, 23)    # C, C+, CHx, CO, H2, H3+, HCO+, He, He+, M, M+, O, OHx, e- -- same order as net.species


def get_nelson_train_test_data(filepath, t, test_frac=1/3, f_fn=nelson_flow_map_wrapper):
    """
    Reads every (state, environment) sample from the first t tracers of a KIDApy collapse-chemistry
    output file, and splits them by tracer into training and testing sets: the last
    round(test_frac * t) tracers are the test set, the rest the training set.

    Each input row is log10 of [states (14 species abundances), params (nH, T, Tgrain, Av, uv1, uv3, uv4)].
    Outputs are evaluated with f_fn (nelson_flow_map_wrapper by default), so are also log10.
    Rows with a NaN (failed solves) and unwritten tracers are dropped.

    Parameters
    ----------
    filepath : str or Path
        Path to one of the R*_M6_trace_cells.npy output files, an (N_rows, 25) array.
    t : int
        Number of tracers to read, starting from the first.

    Returns X_train, f_train, X_test, f_test.
    """
    mm = np.load(filepath, mmap_mode="r")
    if mm.ndim != 2 or mm.shape[1] != 25:
        raise ValueError(f"{filepath} should have shape (N_rows, 25); got {mm.shape}")

    # Tracers are contiguous blocks of rows_per_tracer rows; column 0 (idx) changes at the first block boundary.
    first_idx = mm[:20000, 0]
    boundaries = np.flatnonzero(first_idx != first_idx[0])
    rows_per_tracer = int(boundaries[0]) if boundaries.size else mm.shape[0]
    n_tracers = mm.shape[0] // rows_per_tracer
    if t > n_tracers:
        raise ValueError(f"t={t} tracers requested but {filepath} only has {n_tracers}")

    n_test = int(round(test_frac * t))
    if not 0 < n_test < t:
        raise ValueError(f"test_frac={test_frac} with t={t} tracers leaves an empty training or test set")

    def read_tracers(first, last):
        block = np.asarray(mm[first * rows_per_tracer:last * rows_per_tracer])
        block = block[block[:, 2] != 0.0]                   # unwritten tracers
        X = np.hstack([block[:, FILE_SPECIES_COLS], block[:, FILE_PARAM_COLS]])
        X = X[~np.isnan(X).any(axis=1)]                     # failed solves
        return xp.asarray(to_log(X))

    X_train = read_tracers(0, t - n_test)
    X_test = read_tracers(t - n_test, t)

    return X_train, f_fn(X_train), X_test, f_fn(X_test)


if __name__ == "__main__":

    # Parameters for training
    SEED = 67
    DATA_FILE = Path("TODO/piecewise_constant/R0.1_M6_trace_cells.npy")  # TODO: set to the collapse-chemistry output file
    N_TRACERS = 3
    DIM = 21
    TOL = 0.2
    N_init_clusters = get_NUM_INITIAL_CLUSTERS(DIM)
    min_ppc = get_MIN_PTS_PER_CLUSTER(DIM)
    output = 3 # CO abundance
    dt = 250.0 * YEAR
    n_workers = 144 # number of available cores on node
    f_fn = partial(nelson_flow_map_wrapper, output=output, dt=dt, n_workers=n_workers)

    # --- Load data ---
    X_train, f_train, X_test, f_test_exact = get_nelson_train_test_data(DATA_FILE, N_TRACERS, f_fn=f_fn)

    # --- Build AFINNI surrogate ---
    surrogate = AFINNISurrogate()
    surrogate.train(
        X_train = X_train,
        Y_train = f_train,
        f_fn = f_fn,
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
