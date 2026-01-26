import numpy as np
from array_api_compat import array_namespace
from scipy.integrate import solve_ivp

from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.sensitivities_solver import sensitivities_solver

# Issue: value and derivative are part of same lorenz solve.
# How to deal with splitting the solve into each function?


def lorenz96_flowmap(X0, forcing=1.0, tf=0.1):
    """
    Lorenz96 flowmap implementation.
    Computes a batched ode solve of the data given in X0.

    Args:
        X0: (n_samples, F) data matrix where each row represents a state of the system.
        forcing: Forcing term in the Lorenz96 ODE
        tf: final time for solves

    Returns:
        XF: (n_samples, F) data matrix of states at later time.
    """
    xp = array_namespace(X0)

    n_trajectories, dim = X0.shape

    # reshape ODEs
    X0_flat = X0.ravel()

    def lorenz96_batch_ode(t, x_flat):
        x_batch = x_flat.reshape(n_trajectories, dim)
        dydt_batch = lorenz96(x_batch, forcing=forcing)
        return dydt_batch.ravel()

    # return solution at final time
    t_span = [0, tf]
    t_eval = [tf]
    sol = solve_ivp(lorenz96_batch_ode, t_span, X0_flat, "RK45", t_eval, rtol=1e-6)

    # sol.y contains (1, n_trajectories * dim) array. Reshape accordingly
    XF = sol.y.reshape(n_trajectories, dim)

    return XF


def lorenz96_flowmap_grad(X0, forcing=1.0, tf=0.1):
    """
    Lorenz96 flowmap implementation.
    Computes a batched ode solve of the data given in X0.

    Args:
        X0: (n_samples, F) data matrix where each row represents a state of the system.
        forcing: Forcing term in the Lorenz96 ODE
        tf: final time for solves

    Returns:
        DXF: (n_samples, F, F) data matrix of states at later time.
    """

    _, DXF = sensitivities_solver(lambda X: lorenz96(X, forcing), lorenz96_grad, X0, tf)
    
    # For each Jacobian matrix above, the rows correspond to the same derivative, not the function.
    # We need to take a transpose of each matrix.
    return DXF.transpose((0,2,1))


if __name__ == "__main__":
    n_trajectories = 1
    dim = 5
    X0 = 0.5 * np.ones((n_trajectories, dim))
    F = 1
    t_final = 0.01
    XF = lorenz96_flowmap(X0, forcing=F, tf=t_final)
    DXF = lorenz96_flowmap_grad(X0, forcing=F, tf=t_final)
