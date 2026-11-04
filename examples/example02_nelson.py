
from functools import partial
import sys
import time
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
FILE_TIME_COL     = 1               # time since the start of the tracer [s]
FILE_PARAM_COLS   = slice(2, 9)     # nH, T, Tgrain, Av_pdfac, uv1, uv3, uv4 -- same order as ENV_COLS
FILE_SPECIES_COLS = slice(9, 23)    # C, C+, CHx, CO, H2, H3+, HCO+, He, He+, M, M+, O, OHx, e- -- same order as net.species


def get_nelson_train_test_data(filepath, n_train_tracers=100, n_test_tracers=1, rows_per_tracer=4016, n_rows_until_output=5):
    """
    Reads (state, environment) samples from a KIDApy collapse-chemistry output file. The first
    n_train_tracers tracers are the training set and the next n_test_tracers tracers are the test set.

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
    n_test_tracers : int
        Number of tracers in the test set, immediately following the training tracers.
    rows_per_tracer : int
        Number of contiguous rows per tracer in the file.
    n_rows_until_output : int
        How many rows later in a tracer the output is.

    Returns X_train, f_train, X_test, f_test, t_test. X_train and f_train are arrays pooled over the training
    tracers. X_test, f_test and t_test are lists with one array per test tracer, in tracer order; t_test
    holds the time [s] since the start of the tracer of each X_test row.
    """
    mm = np.load(filepath, mmap_mode="r")
    if mm.ndim != 2 or mm.shape[1] != 25:
        raise ValueError(f"{filepath} should have shape (N_rows, 25); got {mm.shape}")
    n_tracers = n_train_tracers + n_test_tracers
    if n_tracers * rows_per_tracer > mm.shape[0]:
        raise ValueError(f"{filepath} has fewer than {n_tracers} tracers of {rows_per_tracer} rows")

    def read_tracers(first, last):
        block = np.asarray(mm[first * rows_per_tracer:last * rows_per_tracer])
        block = block.reshape(last - first, rows_per_tracer, mm.shape[1])
        inp = block[:, :-n_rows_until_output].reshape(-1, mm.shape[1])
        out = block[:, n_rows_until_output:].reshape(-1, mm.shape[1])

        X = np.hstack([inp[:, FILE_SPECIES_COLS], inp[:, FILE_PARAM_COLS]])
        f = out[:, FILE_SPECIES_COLS]
        keep = (inp[:, 2] != 0.0) & (out[:, 2] != 0.0)              # unwritten tracers
        keep &= ~np.isnan(X).any(axis=1) & ~np.isnan(f).any(axis=1)  # failed solves
        return xp.asarray(to_log(X[keep])), xp.asarray(to_log(f[keep])), inp[keep, FILE_TIME_COL]

    X_train, f_train, _ = read_tracers(0, n_train_tracers)
    tests = [read_tracers(t, t + 1) for t in range(n_train_tracers, n_tracers)]
    X_test = [X for X, _, _ in tests]
    f_test = [f for _, f, _ in tests]
    t_test = [t for _, _, t in tests]
    return X_train, f_train, X_test, f_test, t_test


def read_tracer(filepath, tracer, rows_per_tracer=4016):
    """
    Reads every row of one tracer, in time order and unfiltered. Returns (time, X): the time [s] since the
    start of the tracer, shape (rows_per_tracer,), and log10 of [states, params] (the same layout as
    get_nelson_train_test_data's inputs), shape (rows_per_tracer, 21).
    """
    mm = np.load(filepath, mmap_mode="r")
    block = np.asarray(mm[tracer * rows_per_tracer:(tracer + 1) * rows_per_tracer])
    return block[:, FILE_TIME_COL], to_log(np.hstack([block[:, FILE_SPECIES_COLS], block[:, FILE_PARAM_COLS]]))


def rollout(surrogates, X_true_list, n_rows_until_output=5):
    """
    Rolls several trajectories forward at once by feeding the surrogates' predictions back in as inputs.

    Starting from the true first row of each tracer, each step takes the previous rolled-out row of every
    tracer, predicts the species abundances n_rows_until_output rows later, and uses them as the next input.
    The tracers are advanced together, so each step makes one batched predict call per surrogate rather
    than one per tracer. Species without a surrogate, and the environment params, are taken from the true
    trajectory at each step.

    Parameters
    ----------
    surrogates : dict of int -> AFINNISurrogate
        Maps a species column (in net.species order) to the surrogate that predicts it.
    X_true_list : list of arrays, shape (n_rows_t, 21)
        The true trajectory of each tracer. Lengths may differ.
    n_rows_until_output : int
        How many rows later in a tracer each surrogate's output is.

    Returns
    -------
    rows_list : list of np.ndarray, shape (n_steps_t,)
        Row index along each tracer of each rolled-out step.
    X_roll_list : list of np.ndarray, shape (n_steps_t, 21)
        The rolled-out trajectories, in the same layout as X_true_list.
    trust_list : list of np.ndarray, shape (n_steps_t - 1,), bool
        Whether each step was inside the trust region of every surrogate.
    """
    X_true_list = [np.asarray(_to_host(X)) for X in X_true_list]
    rows_list = [np.arange(0, len(X), n_rows_until_output) for X in X_true_list]
    X_roll_list = [np.array(X[rows]) for X, rows in zip(X_true_list, rows_list)]
    trust_list = [np.ones(len(rows) - 1, dtype=bool) for rows in rows_list]
    n_steps_all = np.array([len(rows) for rows in rows_list])

    for k in range(1, n_steps_all.max()):
        active = np.flatnonzero(n_steps_all > k)   # tracers that still have a step k
        x_prev = np.stack([X_roll_list[t][k - 1] for t in active])
        x_new = np.stack([X_roll_list[t][k] for t in active])
        for col, surrogate in surrogates.items():
            pred, ung = surrogate.predict(x_prev)
            x_new[:, col] = _to_host(pred).ravel()
            for i, t in enumerate(active):
                trust_list[t][k - 1] &= not bool(_to_host(ung).ravel()[i])
        for i, t in enumerate(active):
            X_roll_list[t][k] = x_new[i]

    n_trust = sum(int(tr.sum()) for tr in trust_list)
    n_total = sum(len(tr) for tr in trust_list)
    print(f"Number of trustworthy evals: {n_trust}/{n_total} "
          f"({100.0 * n_trust / n_total:.2f}% inside, {100.0 * (n_total - n_trust) / n_total:.2f}% outside trust region)")
    return rows_list, X_roll_list, trust_list


def one_step_l2_relative_error(surrogates, X_test, f_test):
    """
    Global relative L2 error of the surrogates' one-step predictions (no rollout): every test input is the
    true state, so errors do not accumulate. Computed in the surrogates' output space (log10 abundance).

    Parameters
    ----------
    surrogates : dict of int -> AFINNISurrogate
        Maps a species column (in net.species order) to the surrogate that predicts it.
    X_test : list of arrays, shape (M_t, 21)
        Test inputs, one array per test tracer.
    f_test : list of arrays, shape (M_t, 14)
        Exact outputs for X_test, one array per test tracer.

    Returns
    -------
    errors : dict of int -> float
        ||pred - exact||_2 / ||exact||_2 per species, over all test rows pooled across tracers.
    total : float
        The same ratio over all surrogate species together.
    """
    X = xp.concatenate([xp.asarray(x) for x in X_test], axis=0)
    F = np.concatenate([np.asarray(_to_host(f)) for f in f_test], axis=0)
    errors, num_sq, den_sq = {}, 0.0, 0.0
    for col, surrogate in surrogates.items():
        pred, _ = surrogate.predict(X)
        diff = np.asarray(_to_host(pred), dtype=np.float64).ravel() - F[:, col]
        errors[col] = np.linalg.norm(diff) / np.linalg.norm(F[:, col])
        num_sq += np.sum(diff ** 2)
        den_sq += np.sum(F[:, col] ** 2)
    return errors, np.sqrt(num_sq / den_sq)


def rollout_final_relative_error(surrogates, X_true_list, rows_list, X_roll_list):
    """
    Relative error of each tracer's rollout at its final rolled-out step: ||roll - true||_2 / ||true||_2 over
    the surrogate species (columns in `surrogates`), in log10 abundance. Returns an array of shape (n_tracers,).
    """
    cols = list(surrogates)
    errors = []
    for X_true, rows, X_roll in zip(X_true_list, rows_list, X_roll_list):
        true = np.asarray(_to_host(X_true))[rows[-1], cols]
        errors.append(np.linalg.norm(X_roll[-1, cols] - true) / np.linalg.norm(true))
    return np.array(errors)


if __name__ == "__main__":

    # Parameters for training
    SEED = 67
    DIM = 21
    TOL = 0.02
    N_init_clusters = get_NUM_INITIAL_CLUSTERS(DIM)
    min_ppc = get_MIN_PTS_PER_CLUSTER(DIM)
    dt = 250.0 * YEAR
    n_workers = 144 # number of available cores on node

    # --- Load data ---
    datapath = Path("/scratch/10630/benjo74/R0.1_M6_trace_cells.npy")
    n_train_tracers = 300
    rows_per_tracer = 4016
    n_test_tracers = 20
    n_rows_until_output = 5
    print(f"Loading Nelson data from {datapath}...")
    X_train, f_train, X_test, f_test_exact, t_test = get_nelson_train_test_data(datapath, n_train_tracers, n_test_tracers, rows_per_tracer, n_rows_until_output)
    print("Nelson data loaded.")

    # --- Build AFINNI surrogate for each component ---
    surrogate_dict = dict()
    for output in range(14):

        f_fn = partial(nelson_flow_map_wrapper, output=output, dt=dt, n_workers=n_workers)
        surrogate = AFINNISurrogate()
        f_train_component = f_train[:,output][:,None]
        print(f"Beginning training AFINNI surrogate for output {net.species[output]}...")
        surrogate.train(
            X_train = X_train,
            Y_train = f_train_component,
            f_fn = f_fn,
            mode = SurrogateMode.AFFINE_COV_HESS,
            ATOL = TOL,
            RTOL = TOL,
            ERROR_METRIC = 'Linf-local',
            NUM_INITIAL_CLUSTERS = N_init_clusters,
            MIN_PTS_PER_CLUSTER = min_ppc,
            SEED = SEED,
            STORE_AFFINE_FIT = True,
            USE_WORK_STEALING = True,
            PROFILE_TIMING = True,
            SAVE_INITIAL_CLUSTERING = True,
            USE_EXISTING_INITIAL_CLUSTERING = True
        )

        print("-----------------------------------------------------------")
        print(f"AFINNI training complete for output {net.species[output]}.")
        print(f"N_clusters = {surrogate.N_src}")
        print("-----------------------------------------------------------")
        surrogate_dict[output] = surrogate


    # --- One-step (non-rollout) error on the test tracers ---
    step_errors, step_total = one_step_l2_relative_error(surrogate_dict, X_test, f_test_exact)
    print("One-step relative L2 error (log10 abundance, pooled over test tracers):")
    for col, err in step_errors.items():
        print(f"  {net.species[col]:>5}: {err:.3e}")
    print(f"  Global: {step_total:.3e}")

    # --- Roll out surrogates along each test tracer, feeding their output back in, and plot ---
    n_species = len(net.species)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:n_species]
    roll_start_time = time.perf_counter()
    rollout_rows_list, X_rollout_list, _ = rollout(surrogate_dict, X_test, n_rows_until_output)
    roll_total_time = time.perf_counter() - roll_start_time
    print(f"Time rollout for all {len(X_test)} tracers (batched): {roll_total_time:.2f} seconds")

    final_errors = rollout_final_relative_error(surrogate_dict, X_test, rollout_rows_list, X_rollout_list)
    print("Pointwise relative error at the end of the rollout (log10 abundance, over tracers):")
    print(f"  Mean: {final_errors.mean():.3e}")
    print(f"  Max:  {final_errors.max():.3e} (tracer {final_errors.argmax()})")

    for t, (X_t, time_t) in enumerate(zip(X_test, t_test)):
        rollout_rows, X_rollout = rollout_rows_list[t], X_rollout_list[t]

        fig, ax = plt.subplots(figsize=(16, 8))
        for i, name in enumerate(net.species):
            ax.plot(time_t / (1e3 * YEAR), X_t[:, i], color=colors[i], linestyle="-", linewidth=2, label=f"{name}")
            ax.plot(time_t[rollout_rows] / (1e3 * YEAR), X_rollout[:, i], color=colors[i],
                linestyle="none", marker="o", markersize=10, markerfacecolor="none", markeredgewidth=1.5, markevery=10, alpha=0.7)
        # Line-style key: solid = exact, dotted = predicted
        ax.plot([], [], color="k", linestyle="-", linewidth=2, label="Exact")
        ax.plot([], [], color="k", linestyle="none", marker="o", markersize=10, markerfacecolor="none", markeredgewidth=1.5, alpha=0.7, label="Surrogate")
        ax.set_xlabel("Time along test tracer [kyr]", fontsize=16)
        ax.set_ylabel("log10 abundance", fontsize=20)
        ax.set_title(f"AFINNI on Nelson flow map", fontsize=24)
        ax.tick_params(labelsize=14)
        ax.grid(True, color="#e5e4df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=14, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(f"examples/nelson_test_trajectory_tracer{t}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
