import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from _experiments.ics_deprecated.surrogate import HATANNSurrogate

# --- User Imports ---
# Assuming these are available in your environment as per your previous snippet
from experiments_utils import create_summary_dataframe
from src import xp

# ==========================================
# 1. Physics: Parameterized Logistic Equation
# ==========================================
DT = 0.5


def logistic_deriv(t, X_aug, r):
    """
    RHS for Logistic Equation + Sensitivity Analysis.
    Args:
        t: Time (scalar)
        X_aug: Augmented state [x, s_x, s_r]
        r: Growth rate parameter (passed via args)
    """
    x, s_x, s_r = X_aug[0], X_aug[1], X_aug[2]

    # Primal: dx/dt = r*x*(1-x)
    dx = r * x * (1 - x)

    # Sensitivities
    # df/dx = r - 2rx
    # df/dr = x(1-x)
    df_dx = r - 2 * r * x
    df_dr = x * (1 - x)

    # Dynamics of sensitivities
    d_sx = df_dx * s_x
    d_sr = df_dx * s_r + df_dr

    return np.array([dx, d_sx, d_sr])


def compute_flow_and_grads(X_input_batch, dt):
    """
    Integrates batch of inputs [x0, r] forward by dt.
    """
    N, D = X_input_batch.shape
    F = np.zeros((N, 1))
    G = np.zeros((N, 1, 2))

    for i in range(N):
        x0 = X_input_batch[i, 0]
        r0 = X_input_batch[i, 1]

        # Init Augmented State: [x0, 1.0, 0.0]
        # s_x(0) = 1 (dx/dx = 1)
        # s_r(0) = 0 (dx/dr = 0 at t=0)
        Z0 = np.array([x0, 1.0, 0.0])

        # Pass 'r0' explicitly via args
        sol = solve_ivp(
            logistic_deriv,
            [0, dt],
            Z0,
            t_eval=[dt],
            method="RK45",
            args=(r0,),  # <--- Parameter passing
        )

        final = sol.y[:, -1]
        F[i, 0] = final[0]
        G[i, 0, 0] = final[1]  # ds/dx0
        G[i, 0, 1] = final[2]  # ds/dr

    return F, G


# Wrapper functions for Surrogate
def f_logistic(X):
    F, _ = compute_flow_and_grads(X, DT)
    return F


def g_logistic(X):
    _, G = compute_flow_and_grads(X, DT)
    return G


# ==========================================
# 2. Experiment Utilities
# ==========================================
def create_problem_result_dict(
    problem_name,
    tol,
    dim,
    n_train,
    n_clusters,
    ground_truth,
    predictions,
    ungauranteed_mask,
    train_time,
    predict_time,
    error_tag,
    **kwargs,
):
    """
    Packages results into a dictionary using the exact requested metric definitions.
    """
    if ground_truth is None:
        return {
            "problem_name": problem_name,
            "tol": tol,
            "dim": dim,
            "n_train": n_train,
            "n_test": None,
            "n_clusters": None,
            "ptwise_max_rel_error_linf": None,
            "ptwise_median_rel_error_linf": None,
            "l2_relative_error": None,
            "l2_abs_error": None,
            "n_out": None,
            "perct_out": None,
            "n_in": None,
            "n_in_good": None,
            "perct_in_good": None,
            "n_in_bad": None,
            "perct_in_bad": None,
            "train_time_sec": None,
            "predict_time_sec": None,
            "error_tag": error_tag,
            "success": False,
            **kwargs,
        }

    N_TEST = ground_truth.shape[0]
    TINY_EPS = 1e-8

    n_out = xp.sum(ungauranteed_mask)
    n_in = N_TEST - n_out
    g_idx = ~ungauranteed_mask

    if n_in == 0:
        # Handle the case where NO points are guaranteed
        ptwise_max_rel_error_linf = 0.0
        ptwise_median_rel_error_linf = 0.0
        l2_relative_error = 0.0
        l2_abs_error = 0.0

        n_in_good = 0
        n_in_bad = 0
    else:
        # Original Logic
        # Note: xp.abs handles arrays. For vectors, ensure dimensions match.
        diff = xp.abs(predictions[g_idx] - ground_truth[g_idx])
        denom = xp.abs(ground_truth[g_idx]) + TINY_EPS
        ptwise_rel_error_linf = diff / denom

        # If output is multi-dimensional, take max over dimensions for L_inf
        if ptwise_rel_error_linf.ndim > 1:
            ptwise_rel_error_linf = xp.max(ptwise_rel_error_linf, axis=1)

        ptwise_max_rel_error_linf = xp.nanmax(ptwise_rel_error_linf)
        ptwise_median_rel_error_linf = xp.nanmedian(ptwise_rel_error_linf)

        l2_relative_error = float(
            xp.linalg.norm(ground_truth[g_idx] - predictions[g_idx], ord=2)
            / xp.linalg.norm(ground_truth[g_idx], ord=2)
        )
        l2_abs_error = float(
            xp.linalg.norm(ground_truth[g_idx] - predictions[g_idx], ord=2)
            / xp.sqrt(n_in)
        )

        n_in_good = xp.sum(ptwise_rel_error_linf <= tol)
        n_in_bad = xp.sum(ptwise_rel_error_linf > tol)

    # Calculate Percentages relative to Total Test Set
    perct_out = 100.0 * n_out / N_TEST
    perct_in_good = 100.0 * n_in_good / N_TEST
    perct_in_bad = 100.0 * n_in_bad / N_TEST

    return {
        "problem_name": problem_name,
        "tol": tol,
        "dim": dim,
        "n_train": n_train,
        "n_test": N_TEST,
        "n_clusters": int(n_clusters) if n_clusters is not None else 0,
        "ptwise_max_rel_error_linf": float(ptwise_max_rel_error_linf),
        "ptwise_median_rel_error_linf": float(ptwise_median_rel_error_linf),
        "l2_relative_error": l2_relative_error,
        "l2_abs_error": l2_abs_error,
        "n_out": int(n_out),
        "perct_out": float(perct_out),
        "n_in": int(n_in),
        "n_in_good": int(n_in_good),
        "perct_in_good": float(perct_in_good),
        "n_in_bad": int(n_in_bad),
        "perct_in_bad": float(perct_in_bad),
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "error_tag": error_tag,
        "success": True,
        **kwargs,
    }


def run_logistic_experiment(
    output_dir: Path,
    tol: float,
    n_train: int = 5000,
    n_test: int = 2000,
    n_init_clusters: int = 10,
    ERROR_METRIC: str = "Linf-global",
):
    """Runs a single experiment for a specific TOL."""

    # Experiment Config
    DIM = 2
    # Domain: x in [0.1, 1.2], r in [0.5, 3.5]
    DOMAIN_MIN = np.array([0.1, 0.5])
    DOMAIN_MAX = np.array([1.2, 3.5])

    # JSON Output Path
    fname = f"logistic_N{n_train}_TOL{tol:.1e}_{ERROR_METRIC}.json"
    json_path = output_dir / fname

    if json_path.exists():
        print(f"Skipping {fname} (Already exists)")
        return

    # Data Generation
    xp.random.seed(0)
    # Scaled uniform sampling
    raw_train = xp.random.uniform(0, 1, size=(n_train, DIM))
    X_train = raw_train * (DOMAIN_MAX - DOMAIN_MIN) + DOMAIN_MIN

    raw_test = xp.random.uniform(0, 1, size=(n_test, DIM))
    X_test = raw_test * (DOMAIN_MAX - DOMAIN_MIN) + DOMAIN_MIN

    # Initialize Surrogate
    surrogate = HATANNSurrogate(version="v2")

    train_time = 0.0
    predict_time = 0.0
    ground_truth = None
    preds = None
    mask = None
    error_tag = ""
    n_clusters = 0

    try:
        # Train
        t0 = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=f_logistic,
            g_fn=g_logistic,
            TOL=tol,
            NUM_INITIAL_CLUSTERS=n_init_clusters,
            MIN_PTS_PER_CLUSTER=50,
            MAX_CLUSTERS_EST=2000,
            SEED=42,
            ERROR_METRIC=ERROR_METRIC,  # type: ignore
            USE_HESSIAN_INFORMED_SPLIT=True,
        )
        train_time = time.perf_counter() - t0
        n_clusters = surrogate.get_number_of_sources()

        # Test
        ground_truth = f_logistic(X_test)

        t0 = time.perf_counter()
        preds, mask = surrogate.predict(X_test, k_nn=1)
        predict_time = time.perf_counter() - t0

    except Exception as e:
        error_tag = str(e)
        print(f"  [FAILED] TOL={tol}: {e}")

    # Package Results
    res = create_problem_result_dict(
        problem_name="LogisticMap",
        tol=tol,
        dim=DIM,
        n_train=n_train,
        n_clusters=n_clusters,
        ground_truth=ground_truth,
        predictions=preds,
        ungauranteed_mask=mask,
        train_time=train_time,
        predict_time=predict_time,
        error_tag=error_tag,
        ERROR_METRIC=ERROR_METRIC,
    )

    # Save
    with open(json_path, "w") as f:
        json.dump(res, f, indent=4)

    print(
        f"  [DONE] TOL={tol:.1e} -> Clusters: {n_clusters}, L2 Abs Error: {res.get('l2_abs_error', 'N/A')}"
    )


# ==========================================
# 3. Main Loop
# ==========================================
if __name__ == "__main__":

    # 1. Setup Directory
    EXP_DIR = Path("results/ics26/logistic_convergence")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    create_summary_dataframe(EXP_DIR)

    print(f"--- Starting Logistic Equation Convergence Study ---")
    print(f"Output Directory: {EXP_DIR.absolute()}")

    # 2. Define Tolerances to Sweep
    TOLs = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4]
    ERROR_METRICs = ["Linf-local", "Linf-global"]

    combos = list(product(TOLs, ERROR_METRICs))

    # 3. Run Experiments
    for TOL, ERROR_METRIC in tqdm(combos, desc="Tolerance Sweep"):
        run_logistic_experiment(
            output_dir=EXP_DIR,
            tol=TOL,
            n_train=20_000,
            n_test=10_000,
            ERROR_METRIC=ERROR_METRIC,
        )

    print("\nAll experiments completed.")
    create_summary_dataframe(EXP_DIR)
