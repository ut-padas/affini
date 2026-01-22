
from array_api_compat import array_namespace
from scipy.integrate import solve_ivp


def sensitivities_solver(rhs, DfDx, X0, tf):
    """
    Modifies an ODE rhs to also solve for sensitivities.
    If x' = f(x), we modify the initial condition and rhs function to solve
        sensitivities in conjuction with the standard ODE.
    For now, does not include parameters as a variable in the system; assume fixed.

    Args:
        rhs: (function) The right hand side of the ODE
        DfDx: (function) The Jacobian of the right hand side
        DfDp: (function) The derivative with respect to parameters
        x0: (n_trajectories, n_equations) The initial conditions. Each row represents a trajectory
        p: (vector) The physical parameters of the system.
        tf: (scalar) final time

    Returns:
        xf: (n_trajectories, dim) The solution at tf
        dxf: (n_trajectories, n_equations, n_equations)
            The derivatives, of each equation, with respect to each other equation and parameter at tf
    """
    xp = array_namespace(X0)

    n_trajectories, n_equations = X0.shape

    # Modify initial condition so that it contains sensitivity variables
    I_flat = xp.eye(n_equations).ravel()
    X0_sensitivities = xp.outer(xp.ones(n_trajectories), I_flat)
    X0_combined = xp.column_stack((X0, X0_sensitivities))
    X0_flat = X0_combined.ravel()

    # Modify rhs to deal with sensitivities
    def new_rhs(x_s_flat, f, DfDx):

        # x refers to regular variables, s refers to sensitivity variables

        # first reshape
        x_s_batch = x_s_flat.reshape(n_trajectories, n_equations * (1 + n_equations))
        x_batch = x_s_batch[:, :n_equations] # (n_trajectories, n_equations)
        s_batch = x_s_batch[:, n_equations:] # (n_trajectories, n_equations * n_equations)
        s_batch_matrices = s_batch.reshape(n_trajectories, n_equations, n_equations) # (n_trajectories, n_equations, n_equations)

        # apply f to x_batch
        rhs_x_batch = f(x_batch) # (n_trajectories, n_equations)

        # apply J to each vector in s_batch (sensitivity RHS)
        J = DfDx(x_batch) # (n_trajectories, n_equations, n_equations)
        rhs_s_batch_tensor = xp.einsum('ijk,ikl->ijl', J, s_batch_matrices) # (n_trajectories, n_equations, n_equations)

        # combine these back together and flatten
        rhs_s_batch_matrix = rhs_s_batch_tensor.reshape(n_trajectories, n_equations * n_equations)
        rhs = xp.column_stack((rhs_x_batch, rhs_s_batch_matrix))
        return rhs.ravel()

    # solve ODEs
    t_span = [0, tf]
    t_eval = [tf]
    sol = solve_ivp(
        lambda t,x: new_rhs(x,rhs,DfDx),
        t_span,
        X0_flat,
        method='RK45',
        t_eval=t_eval
    )

    # restructure and return
    XF_SF_batch = sol.y.reshape(n_trajectories, n_equations * (1 + n_equations))
    XF_batch = XF_SF_batch[:, :n_equations] # (n_trajectories, n_equations)
    SF_batch = XF_SF_batch[:, n_equations:] # (n_trajectories, n_equations * n_equations)
    SF_batch_matrices = SF_batch.reshape(n_trajectories, n_equations, n_equations)

    return XF_batch, SF_batch_matrices
