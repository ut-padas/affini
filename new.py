import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cupy as cp
import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from scipy.integrate import solve_ivp

# Import specific problems (Add your imports here)
from src.problems.six_hump_camelback import (
    six_hump_camelback,
    six_hump_camelback_grad,
    six_hump_camelback_hess,
)

# Import Surrogate and Sampler
from src.surrogate import DataSampler, NearODESurrogate

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("experiments")


# -----------------------------------------------------------------------------
# 1. Problem Configuration
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkProblem:
    name: str
    f: Callable
    grad: Callable
    hess: Optional[Callable]  # Optional, used for estimating L
    domain: np.ndarray  # Shape (D, 2)
    data_dim: int
    f_dim: int = 1

    # Evaluation params
    L_global: float | None = None  # Can be pre-set or computed
    test_samples: int = 100_000


# -----------------------------------------------------------------------------
# 2. Problem Definitions (Restored from Deprecated Script)
# -----------------------------------------------------------------------------


# --- Borehole (8D) ---
def borehole(x):
    """8D Borehole function."""
    xp = array_namespace(x)
    if x.ndim == 1:
        x = x[None, :]
    rw, r, Tu, Hu, Tl, Hl, L, Kw = [x[:, i] for i in range(8)]

    frac1 = 2 * L * Tu / (xp.log(r / rw) * Kw * rw**2)
    term1 = xp.log(r / rw) * (1 + frac1)
    return 2 * np.pi * Tu * (Hu - Hl) / term1


def borehole_grad(x):
    """Placeholder gradient for 8D Borehole."""
    xp = array_namespace(x)
    return xp.zeros((x.shape[0], 1, 8))


# --- Dixon-Price (nD) ---
def dixon_price(x):
    xp = array_namespace(x)
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[1]

    x1 = x[:, 0]
    term1 = (x1 - 1) ** 2

    sum_term = 0
    for i in range(1, n):
        # j in formula is i+1 (1-based indexing)
        j = i + 1
        xi = x[:, i]
        xi_prev = x[:, i - 1]
        sum_term += j * (2 * xi**2 - xi_prev) ** 2

    return (term1 + sum_term)[:, None]  # (N, 1)


def dixon_price_grad(x):
    xp = array_namespace(x)
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[1]
    g = xp.zeros_like(x)

    # Element 0
    g[:, 0] = 2 * (x[:, 0] - 1) - 4 * (2 * x[:, 1] ** 2 - x[:, 0])

    # Element n-1 (last)
    g[:, -1] = 2 * n * (2 * x[:, -1] ** 2 - x[:, -2]) * (4 * x[:, -1])

    # Elements 1 to n-2
    for i in range(1, n - 1):
        j = i + 1  # 1-based index for current term
        term1 = 2 * j * (2 * x[:, i] ** 2 - x[:, i - 1]) * (4 * x[:, i])
        term2 = 2 * (j + 1) * (2 * x[:, i + 1] ** 2 - x[:, i])
        g[:, i] = term1 - term2

    return g[:, None, :]  # (N, 1, D)


# --- Affine (nD -> mD) ---
# Random A and c must be fixed for reproducibility
np.random.seed(42)
AFFINE_A = np.random.uniform(0, 1, size=(10, 20))  # 20 -> 10
AFFINE_c = np.random.uniform(0, 1, size=(10))


def affine(x):
    xp = array_namespace(x)
    # Ensure A, c are on correct device
    A = xp.asarray(AFFINE_A, dtype=x.dtype)
    c = xp.asarray(AFFINE_c, dtype=x.dtype)

    # x: (N, 20), A.T: (20, 10) -> (N, 10)
    return x @ A.T + c


def affine_grad(x):
    xp = array_namespace(x)
    N = x.shape[0]
    A = xp.asarray(AFFINE_A, dtype=x.dtype)

    # Jacobian is constant A (10, 20). We need (N, 10, 20)
    J = xp.expand_dims(A, axis=0)
    return xp.repeat(J, N, axis=0)


# --- Lorenz-96 (ODE) ---
# Note: Running ODE solvers inside the training loop for 100k points is slow.
# We will define the vector field F(x) here, as that's what the surrogate usually models
# in "Near-ODE" contexts (emulating the RHS).
# If you specifically need to surrogate the *flow map* (x_t -> x_{t+dt}),
# we need the solver. Assuming we model the Vector Field f(x) here for speed/demo.


def lorenz96_velocity(x):
    """Compute velocity (RHS) of Lorenz 96."""
    xp = array_namespace(x)
    # x: (N, D)
    # F = (x_{i-1} - x_{i-2}) * x_{i+1} - x_i + F_force
    # Assuming F_force is parameter p[0]. Let's fix F=8.0 standard
    F_force = 8.0

    # Roll logic needs care with N-dim batches.
    # xp.roll shifts along last axis by default? Check docs.
    # Manual cyclic indexing is safer for batch (N, D)

    D = x.shape[1]
    x_m1 = xp.roll(x, 1, axis=1)  # i-1
    x_m2 = xp.roll(x, 2, axis=1)  # i-2
    x_p1 = xp.roll(x, -1, axis=1)  # i+1

    dxdt = (x_p1 - x_m2) * x_m1 - x + F_force
    return dxdt


def lorenz96_jacobian(x):
    """Jacobian of L96 Velocity."""
    xp = array_namespace(x)
    N, D = x.shape
    J = xp.zeros((N, D, D), dtype=x.dtype)

    indices = xp.arange(D)
    im2 = (indices - 2) % D
    im1 = (indices - 1) % D
    ip1 = (indices + 1) % D

    # J[i, i-2] = -x[i-1]
    # But wait, x is (N, D). We need broadcasting.
    # For all batches b, all rows i:
    # J[b, i, im2[i]] = -x[b, im1[i]]

    # This vectorized assignment is tricky in pure array API.
    # Simplified loop for clarity (J calc is usually cheap compared to clustering):
    for i in range(D):
        J[:, i, im2[i]] = -x[:, im1[i]]
        J[:, i, im1[i]] = x[:, ip1[i]] - x[:, im2[i]]
        J[:, i, i] = -1.0
        J[:, i, ip1[i]] = x[:, im1[i]]

    return J


# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------


def estimate_lipschitz_constant(
    problem: BenchmarkProblem, samples: int = 1000
) -> float:
    """
    Estimates Global Lipschitz constant L via max Hessian norm.
    For high dimensions, samples random points instead of a grid.
    """
    if problem.L_global is not None:
        return problem.L_global

    if problem.hess is None:
        logger.warning(f"No Hessian provided for {problem.name}, using default L=10.0")
        return 10.0

    xp = array_namespace(problem.domain)

    # Use dense grid for 2D, Random LHS for High-D
    if problem.data_dim == 2:
        d = problem.domain
        x = xp.linspace(d[0, 0], d[0, 1], int(np.sqrt(samples)))
        y = xp.linspace(d[1, 0], d[1, 1], int(np.sqrt(samples)))
        xv, yv = xp.meshgrid(x, y)
        pts = xp.column_stack((xv.flatten(), yv.flatten()))
    else:
        pts = DataSampler.sample_latin_hypercube(problem.domain, N=samples, seed=999)

    # Compute Hessian Norms
    # Expected Hess shape: (N, D, D) or (N, F_DIM, D, D)
    hess_matrices = problem.hess(pts)

    # Compute spectral norm (max singular value)
    # flattening to (N, D, D) if needed
    if hess_matrices.ndim == 4:
        # Take max over F_dim or just first dim?
        # Strictly L is max over all outputs.
        # For simplicity taking norm of F=0 slice if F>1 is messy without full tensor logic.
        hess_matrices = hess_matrices[:, 0, :, :]

    norms = xp.linalg.norm(hess_matrices, ord=2, axis=(1, 2))
    L_est = float(xp.max(norms))

    logger.info(f"[{problem.name}] Estimated Global L = {L_est:.4f}")
    return L_est


# -----------------------------------------------------------------------------
# 4. Core Experiment Logic
# -----------------------------------------------------------------------------


def run_experiment(
    prob: BenchmarkProblem, n_train: int, atol: float, rtol: float
) -> dict:
    """
    Trains surrogate -> Evaluates -> Computes Guarantee Statistics.
    """
    xp = array_namespace(prob.domain)

    # 1. Estimate L (if needed for fallback)
    L = estimate_lipschitz_constant(prob)

    # 2. Train
    surrogate = NearODESurrogate(data_dim=prob.data_dim, f_dim=prob.f_dim)
    X_train = DataSampler.sample_latin_hypercube(prob.domain, N=n_train, seed=42)

    start_t = time.perf_counter()
    surrogate.train(
        X_train=X_train,
        f=prob.f,
        grad=prob.grad,
        NUM_INITIAL_CLUSTERS=50,
        MIN_PTS_PER_CLUSTER=10,
        ATOL=atol,
        RTOL=rtol,
        SEED=42,
    )
    train_time = time.perf_counter() - start_t

    # 3. Evaluate (Test Set)
    X_test = DataSampler.sample_uniform(prob.domain, N=prob.test_samples, seed=123)

    preds, outside_mask = surrogate.predict(
        X_test, k_NEIGHBOURS=1, ATOL=atol, RTOL=rtol, L_global=L, ALPHA=1.0
    )

    # 4. Compute Errors
    ground_truth = prob.f(X_test)

    # Reshape for consistency: (N,) or (N, F)
    # If output is 1D, flatten both
    if prob.f_dim == 1:
        if preds.ndim > 1:
            preds = preds.flatten()
        if ground_truth.ndim > 1:
            ground_truth = ground_truth.flatten()

        abs_err = xp.abs(preds - ground_truth)
        rel_err = abs_err / (xp.abs(ground_truth) + 1e-12)
        norm_truth = xp.linalg.norm(ground_truth, ord=2)
        norm_err = xp.linalg.norm(abs_err, ord=2)
    else:
        # Vector output (e.g. Affine, L96)
        # Error is usually 2-norm per point
        abs_err = xp.linalg.norm(preds - ground_truth, axis=1)
        norm_truth_pt = xp.linalg.norm(ground_truth, axis=1) + 1e-12
        rel_err = abs_err / norm_truth_pt

        # Global norms for RMS
        norm_truth = xp.linalg.norm(ground_truth)  # Frobenius/Flat
        norm_err = xp.linalg.norm(preds - ground_truth)

    # 5. Categorize Points
    # "Good Error": Point satisfies tolerance requirements
    is_good_err = (abs_err <= atol) | (rel_err <= rtol)

    # Masks
    mask_guaranteed = ~outside_mask

    # Categories
    n_in_bad = float(xp.sum(mask_guaranteed & (~is_good_err)))
    n_in_good = float(xp.sum(mask_guaranteed & is_good_err))
    n_out = float(xp.sum(outside_mask))

    n_total = float(len(X_test))

    # 6. Metrics
    red_errors = abs_err[mask_guaranteed & (~is_good_err)]

    return {
        "Problem": prob.name,
        "N_init": n_train,
        "atol": atol,
        "rtol": rtol,
        "N_clusters": len(surrogate.X_src),  # type: ignore
        "Train_Time": train_time,
        "p_out": n_out / n_total,
        "p_in_good": n_in_good / n_total,
        "p_in_bad": n_in_bad / n_total,
        "max_err": float(xp.max(abs_err)),
        "rms_err": float(norm_err / norm_truth) if norm_truth > 0 else 0.0,
        "max_red_err": float(xp.max(red_errors)) if len(red_errors) > 0 else 0.0,
    }


# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------


def main():

    # --- 1. Define Domains ---
    # Dixon-Price 4D Domain: [0, 1]^4
    d_dp = 4
    domain_dp = np.column_stack((np.zeros(d_dp), np.ones(d_dp)))

    # Affine 20D Domain: [0, 1]^20
    d_aff = 20
    domain_aff = np.column_stack((np.zeros(d_aff), np.ones(d_aff)))

    # Lorenz 5D Domain: [0.5, 1.5]^5
    d_l96 = 5
    domain_l96 = np.column_stack((0.5 * np.ones(d_l96), 1.5 * np.ones(d_l96)))

    # --- 2. Problem List ---
    problems = [
        BenchmarkProblem(
            name="Camelback_2D",
            f=six_hump_camelback,
            grad=six_hump_camelback_grad,
            hess=six_hump_camelback_hess,
            domain=np.array([[-3.0, 3.0], [-2.0, 2.0]]),
            data_dim=2,
        ),
        BenchmarkProblem(
            name="Borehole_8D",
            f=borehole,
            grad=borehole_grad,
            hess=None,  # L will default to 10
            domain=np.array(
                [
                    [0.05, 0.15],
                    [100, 50000],
                    [63070, 115600],
                    [990, 1110],
                    [63.1, 116],
                    [700, 820],
                    [1120, 1680],
                    [9855, 12045],
                ]
            ),
            data_dim=8,
        ),
        BenchmarkProblem(
            name="DixonPrice_4D",
            f=dixon_price,
            grad=dixon_price_grad,
            hess=None,
            domain=domain_dp,
            data_dim=d_dp,
        ),
        BenchmarkProblem(
            name="Affine_20D_to_10D",
            f=affine,
            grad=affine_grad,
            hess=None,
            domain=domain_aff,
            data_dim=d_aff,
            f_dim=10,
        ),
        BenchmarkProblem(
            name="Lorenz96_5D",
            f=lorenz96_velocity,
            grad=lorenz96_jacobian,
            hess=None,
            domain=domain_l96,
            data_dim=d_l96,
            f_dim=d_l96,
        ),
    ]

    # Define Sweep
    tolerances = [
        (0.1, 0.02),
        (0.05, 0.01),
        (0.01, 0.002),
    ]

    N_TRAIN = 100_000
    results = []

    # Run Loop
    for prob in problems:
        print(f"\n--- Running Benchmarks for {prob.name} ---")
        for atol, rtol in tolerances:
            print(f"  Running atol={atol}, rtol={rtol}...")
            stats = run_experiment(prob, N_TRAIN, atol, rtol)
            results.append(stats)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Format for display
    print("\n=== Experiment Results ===")
    print(df.to_string(index=False))

    # Save CSV
    df.to_csv("experiment_results.csv", index=False)

    # Generate LaTeX
    latex_str = df.to_latex(
        index=False,
        columns=[
            "Problem",
            "N_init",
            "atol",
            "rtol",
            "N_clusters",
            "p_out",
            "p_in_good",
            "p_in_bad",
            "rms_err",
        ],
        header=[
            "Prob",
            "$N_{init}$",
            "atol",
            "rtol",
            "$N_{src}$",
            "$p_{out}$",
            "$p_{good}$",
            "$p_{bad}$",
            "RMS",
        ],
        float_format="%.4f",
        caption="Surrogate Performance Benchmarks",
        label="tab:gpu_benchmarks",
    )

    with open("experiment_results_latex.txt", "w") as f:
        f.write(latex_str)

    print("\nSaved 'experiment_results.csv' and 'experiment_results_latex.txt'")


if __name__ == "__main__":
    main()
