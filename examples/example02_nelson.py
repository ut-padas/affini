"""
This example builds a surrogate for the Nelson flow map in two dimensions.
It uses the workflow described in the paper, with training and testing data
read in from a file instead of generated randomly. This example uses no parallelism.
"""

from functools import partial
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def get_nelson_train_test_data(filepath, n_train_tracers=100, rows_per_tracer=4016, n_rows_until_output=5):
    """
    Reads (state, environment) samples from a KIDApy collapse-chemistry output file. The first
    n_train_tracers tracers are the training set and the next tracer is the test set.

    Each input row is log10 of [states (14 species abundances), params (nH, T, Tgrain, Av, uv1, uv3, uv4)],
    the layout nelson_flow_map_wrapper expects. The output for a row is log10 of species column `output`
    n_rows_until_output rows later in the same tracer, so the last n_rows_until_output rows of each tracer
    (which have no output) are cut off. Pairs with a NaN (failed solves) or from unwritten tracers are dropped.

    Parameters
    ----------
    filepath : str or Path
        Path to one of the R*_M6_trace_cells.npy output files, an (N_rows, 25) array.
    n_train_tracers : int
        Number of tracers in the training set, starting from the first.
    rows_per_tracer : int
        Number of contiguous rows per tracer in the file.
    n_rows_until_output : int
        How many rows later in a tracer the output is.

    Returns X_train, f_train, X_test, f_test.
    """
    mm = np.load(filepath, mmap_mode="r")
    if mm.ndim != 2 or mm.shape[1] != 25:
        raise ValueError(f"{filepath} should have shape (N_rows, 25); got {mm.shape}")
    if (n_train_tracers + 1) * rows_per_tracer > mm.shape[0]:
        raise ValueError(f"{filepath} has fewer than {n_train_tracers + 1} tracers of {rows_per_tracer} rows")

    def read_tracers(first, last):
        block = np.asarray(mm[first * rows_per_tracer:last * rows_per_tracer])
        block = block.reshape(last - first, rows_per_tracer, mm.shape[1])
        inp = block[:, :-n_rows_until_output].reshape(-1, mm.shape[1])
        out = block[:, n_rows_until_output:].reshape(-1, mm.shape[1])

        X = np.hstack([inp[:, FILE_SPECIES_COLS], inp[:, FILE_PARAM_COLS]])
        f = out[:, FILE_SPECIES_COLS]
        keep = (inp[:, 2] != 0.0) & (out[:, 2] != 0.0)              # unwritten tracers
        keep &= ~np.isnan(X).any(axis=1) & ~np.isnan(f).any(axis=1)  # failed solves
        return xp.asarray(to_log(X[keep])), xp.asarray(to_log(f[keep]))

    X_train, f_train = read_tracers(0, n_train_tracers)
    X_test, f_test = read_tracers(n_train_tracers, n_train_tracers + 1)
    return X_train, f_train, X_test, f_test


def read_tracer(filepath, tracer, rows_per_tracer=4016):
    """
    Reads every row of one tracer, in time order and unfiltered, as log10 of [states, params]
    (the same layout as get_nelson_train_test_data's inputs). Returns an array of shape (rows_per_tracer, 21).
    """
    mm = np.load(filepath, mmap_mode="r")
    block = np.asarray(mm[tracer * rows_per_tracer:(tracer + 1) * rows_per_tracer])
    return to_log(np.hstack([block[:, FILE_SPECIES_COLS], block[:, FILE_PARAM_COLS]]))


def rollout(surrogates, X_true, n_rows_until_output=5):
    """
    Rolls a trajectory forward by feeding the surrogates' predictions back in as inputs.

    Starting from the true first row, each step takes the previous rolled-out row, predicts the species
    abundances n_rows_until_output rows later, and uses them as the next input. Species without a
    surrogate, and the environment params, are taken from the true trajectory at each step.

    Parameters
    ----------
    surrogates : dict of int -> AFINNISurrogate
        Maps a species column (in net.species order) to the surrogate that predicts it.
    X_true : array, shape (rows_per_tracer, 21)
        The true trajectory of one tracer, from read_tracer.
    n_rows_until_output : int
        How many rows later in a tracer each surrogate's output is.

    Returns
    -------
    rows : np.ndarray, shape (n_steps,)
        Row index along the tracer of each rolled-out step.
    X_roll : np.ndarray, shape (n_steps, 21)
        The rolled-out trajectory, in the same layout as X_true.
    """
    rows = np.arange(0, len(X_true), n_rows_until_output)
    X_roll = np.array(X_true[rows])
    for k in range(1, len(rows)):
        x_prev = X_roll[k - 1:k]
        for col, surrogate in surrogates.items():
            pred, _ = surrogate.predict(x_prev)
            X_roll[k, col] = _to_host(pred).ravel()[0]
    return rows, X_roll


if __name__ == "__main__":

    # Parameters for training
    SEED = 67
    DIM = 21
    TOL = 0.2
    N_init_clusters = get_NUM_INITIAL_CLUSTERS(DIM)
    min_ppc = get_MIN_PTS_PER_CLUSTER(DIM)
    output = 3 # CO abundance
    dt = 250.0 * YEAR
    n_workers = 144 # number of available cores on node
    f_fn = partial(nelson_flow_map_wrapper, output=output, dt=dt, n_workers=n_workers)

    # --- Load data ---
    datapath = Path("/scratch/10630/benjo74/R0.1_M6_trace_cells.npy")
    n_train_tracers = 200
    rows_per_tracer = 4016
    n_rows_until_output = 5
    X_train, f_train, X_test, f_test_exact = get_nelson_train_test_data(datapath, n_train_tracers, rows_per_tracer, n_rows_until_output)

    # --- Build AFINNI surrogate ---
    surrogate = AFINNISurrogate()
    f_train_component = f_train[:,output][:,None]
    f_test_exact_component = f_test_exact[:,output][:,None]
    surrogate.train(
        X_train = X_train,
        Y_train = f_train_component,
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

    # --- Roll out surrogate along test trajectory, feeding its output back in ---
    X_test_traj = read_tracer(datapath, n_train_tracers, rows_per_tracer)
    rollout_rows, X_rollout = rollout({output: surrogate}, X_test_traj, n_rows_until_output)

    # --- Plot test trajectory ---
    species_name = net.species[output]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(X_test_traj)), X_test_traj[:, output], color="#0b0b0b", linestyle="-", linewidth=2, label=f"{species_name}")
    ax.plot(rollout_rows, X_rollout[:, output], color="#2a78d6", linestyle=":", linewidth=2)
    ax.set_xlabel("Row along test tracer")
    ax.set_ylabel(f"log10 abundance")
    ax.set_title(f"AFINNI on Nelson flow map, {species_name}, tol={TOL}")
    ax.grid(True, color="#e5e4df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"nelson_test_trajectory_{species_name}.png", dpi=300)

