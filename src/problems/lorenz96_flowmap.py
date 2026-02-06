import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

# Assumes these modules exist in your project structure
from src import IS_CUPY_BACKEND, Array, xp
from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.sensitivities_solver import (
    parallel_sensitivities_solver,
    parallel_sensitivities_solver_single_component
)


def _ensure_numpy(x: Array) -> np.ndarray:
    """Helper to ensure data is on CPU for scipy/multiprocessing."""
    if IS_CUPY_BACKEND:
        return xp.asnumpy(x)
    return x


def _ensure_backend(x: np.ndarray) -> Array:
    """Helper to return data to the configured backend (CPU or GPU)."""
    if IS_CUPY_BACKEND:
        return xp.asarray(x)
    return x


def lorenz96_flowmap(X0: Array, forcing: float = 1.0, tf: float = 0.1) -> Array:
    """
    Computes the flowmap of the Lorenz96 system.

    This function runs in a single process using scipy.integrate.solve_ivp.
    It relies on vectorization for batch efficiency. For massive batches,
    consider a parallel approach similar to the gradient functions.

    Args:
        X0: (N, d) data matrix of initial states.
        forcing: Forcing term 'F'.
        tf: Final integration time.

    Returns:
        XF: (N, d) data matrix of states at time tf.
    """
    X0_np = _ensure_numpy(X0)
    n_trajectories, d = X0_np.shape
    X0_flat = X0_np.ravel()

    # Define the batched RHS locally.
    # Since this function is NOT using multiprocessing, a closure is safe/fast.
    # solve_ivp requires f(t, y), but L96 is autonomous (independent of t).
    def batched_ode_rhs(t, x_flat):
        x_batch = x_flat.reshape(n_trajectories, d)
        # Note: We assume lorenz96 returns (N, d)
        dxdt_batch = lorenz96(x_batch, forcing=forcing)
        return dxdt_batch.ravel()

    t_span = [0, tf]

    # Run integration
    sol = solve_ivp(
        batched_ode_rhs, t_span, X0_flat, method="RK45", t_eval=[tf], rtol=1e-6
    )

    # Reshape result: sol.y is (n_vars, 1) -> transpose -> (1, n_vars) -> reshape
    XF = sol.y.T.reshape(n_trajectories, d)

    return _ensure_backend(XF)


def lorenz96_flowmap_grad(X0: Array, forcing: float = 1.0, tf: float = 0.1) -> Array:
    """
    Computes the gradients (Jacobians) of the Lorenz96 flowmap.

    Uses multiprocessing to solve sensitivities in parallel chunks.

    Args:
        X0: (N, d) initial states.
        forcing: Forcing term.
        tf: Final time.

    Returns:
        DXF: (N, d, d) Jacobian matrices d(x_tf)/d(x_0).
    """
    X0_np = _ensure_numpy(X0)

    # KEY FIX: Use partial instead of lambda.
    # Multiprocessing relies on pickle, which cannot serialize lambda functions.
    # partial creates a picklable callable with 'forcing' baked in.
    rhs = partial(lorenz96, forcing=forcing)

    # We only need the sensitivities (index 1), ignoring the flowmap state (index 0)
    _, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return _ensure_backend(DXF)


def lorenz96_flowmap_oracle(
    X0: Array, forcing: float = 1.0, tf: float = 0.1
) -> tuple[Array, Array]:
    """
    Computes both the flowmap and its gradients in parallel.

    Args:
        X0: (N, d) initial states.
        forcing: Forcing term.
        tf: Final time.

    Returns:
        Tuple (XF, DXF).
    """
    X0_np = _ensure_numpy(X0)

    # Use partial for picklability
    rhs = partial(lorenz96, forcing=forcing)

    XF, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return _ensure_backend(XF), _ensure_backend(DXF)



def lorenz96_flowmap_oracle_single_component(
    X0: Array, forcing: float = 1.0, tf: float = 0.1
) -> tuple[Array, Array]:
    """
    Computes both the flowmap and its gradients in parallel.

    Args:
        X0: (N, d) initial states.
        forcing: Forcing term.
        tf: Final time.

    Returns:
        Tuple (XF, DXF).
    """
    X0_np = _ensure_numpy(X0)

    # Use partial for picklability
    rhs = partial(lorenz96, forcing=forcing)

    XF, DXF = parallel_sensitivities_solver_single_component(rhs, lorenz96_grad, X0_np, tf)

    return _ensure_backend(XF), _ensure_backend(DXF)



if __name__ == "__main__":
    # Essential for Windows compatibility when using multiprocessing
    mp.freeze_support()

    # --- Test Configuration ---
    n_trajectories = 5
    dim = 5
    X0_test = 0.5 * np.ones((n_trajectories, dim))
    F_val = 8.0
    t_final = 0.05

    print(f"--- Running Lorenz96 Benchmarks (N={n_trajectories}, d={dim}) ---")

    # 1. Test Standard Flowmap
    X_out = lorenz96_flowmap(X0_test, forcing=F_val, tf=t_final)
    print(f"Flowmap Shape: {X_out.shape}")

    # 2. Test Gradient Only
    DX_out = lorenz96_flowmap_grad(X0_test, forcing=F_val, tf=t_final)
    print(f"Gradient Shape: {DX_out.shape}")

    # 3. Test Oracle (Both)
    X_or, DX_or = lorenz96_flowmap_oracle(X0_test, forcing=F_val, tf=t_final)
    print(f"Oracle Shapes: State={X_or.shape}, Grad={DX_or.shape}")

    # Simple sanity check: Gradients should be Identity-like for very small t
    if t_final < 1e-3:
        is_identity = np.allclose(DX_or, np.eye(dim), atol=1e-2)
        print(f"Small t sanity check (expect Identity): {is_identity}")
