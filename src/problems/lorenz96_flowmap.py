import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

# Assumes these modules exist in your project structure
from src import IS_CUPY_BACKEND, Array, _to_device, _to_host, xp
from src.problems.adjoint_solver import parallel_adjoint_solver
from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.sensitivities_solver import parallel_sensitivities_solver


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
    X0_np = _to_host(X0)
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

    return _to_device(XF)


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
    X0_np = _to_host(X0)
    N, d = X0_np.shape

    # KEY FIX: Use partial instead of lambda.
    # Multiprocessing relies on pickle, which cannot serialize lambda functions.
    # partial creates a picklable callable with 'forcing' baked in.
    rhs = partial(lorenz96, forcing=forcing)

    # We only need the sensitivities (index 1), ignoring the flowmap state (index 0)
    _, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return _to_device(DXF)


def lorenz96_flowmap_oracle_first_component(
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
    X0_np = _to_host(X0)
    N, d = X0_np.shape

    # Use partial for picklability
    rhs = partial(lorenz96, forcing=forcing)

    e1 = np.eye(d)[0]
    XF, DXF = parallel_adjoint_solver(rhs, lorenz96_grad, X0_np, tf, e1)

    return _to_device(XF), _to_device(DXF)
