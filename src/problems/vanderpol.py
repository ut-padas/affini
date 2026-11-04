"""Van der Pol oscillator and its gradient."""

import numpy as np

from src import Array


def vanderpol(X: Array, mu=5.0) -> Array:
    """
    Van der Pol oscillator implementation.

    Args:
        X: (N, 2) data matrix where each row represents a state of the system.
        mu: Nonzero oscillator parameter controlling the time scales.

    Returns:
        dXdt: (N, 2) time derivatives at each state in X.
    """
    x = X[:, 0]
    y = X[:, 1]
    dxdt = mu * (x - (x**3 / 3) - y)
    dydt = x / mu
    return np.column_stack((dxdt, dydt))


def vanderpol_grad(X, mu=5.0) -> Array:
    """
    Jacobian of the Van der Pol vector field.

    Args:
        X: (N, 2) data matrix where each row represents a state of the system.
        mu: Nonzero oscillator parameter controlling the time scales.

    Returns:
        J: (N, 2, 2) Jacobian matrices at each state in X.
    """
    N = X.shape[0]
    x = X[:, 0]
    J = np.zeros((N, 2, 2))

    # df1/dx = mu * (1 - x^2)
    J[:, 0, 0] = mu * (1 - x**2)

    # df1/dy = -mu
    J[:, 0, 1] = -mu

    # df2/dx = 1/mu
    J[:, 1, 0] = 1.0 / mu

    # df2/dy = 0 (already zero from np.zeros)
    return J


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Example usage and testing
    dim = 2
    mu = 5.0
    rng = np.random.default_rng(0)

    # Test 1: Evaluate Van der Pol function on a grid
    print("Test 1: Evaluating Van der Pol on a 2D grid")
    grid_points = 100
    x = np.linspace(-3, 3, grid_points)
    X1, X2 = np.meshgrid(x, x)
    X_test = np.zeros((grid_points * grid_points, dim))
    X_test[:, 0] = X1.ravel()
    X_test[:, 1] = X2.ravel()

    # Evaluate Van der Pol directly on the grid
    dxdt = vanderpol(X_test, mu=mu)
    # Take the first component for visualization
    dxdt_component = dxdt[:, 0].reshape(grid_points, grid_points)

    # Plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(X1, X2, dxdt_component, levels=50, cmap="viridis")
    plt.colorbar(label="dX₀/dt")
    plt.xlabel("X₀")
    plt.ylabel("X₁")
    plt.title("Van der Pol: First Component Time Derivative")

    # Test 2: Batch integration of multiple trajectories
    print("\nTest 2: Batch integration of multiple trajectories")
    n_trajectories = 5
    x0_batch = rng.standard_normal((n_trajectories, dim))
    x0_flat = x0_batch.ravel()

    def vanderpol_batch_ode(t, y_flat):
        y_batch = y_flat.reshape(n_trajectories, dim)
        dydt_batch = vanderpol(y_batch, mu=mu)
        return dydt_batch.ravel()

    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 1000)

    # Integrate all independent trajectories in one solve_ivp call.
    sol = solve_ivp(vanderpol_batch_ode, t_span, x0_flat, t_eval=t_eval, method="RK45")
    if not sol.success:
        raise RuntimeError(f"Van der Pol integration failed: {sol.message}")

    # Reshape solution: (n_trajectories * dim, n_times) -> (n_trajectories, dim, n_times)
    sol_batch = sol.y.reshape(n_trajectories, dim, -1)

    plt.subplot(1, 2, 2)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_trajectories))

    for traj_idx in range(n_trajectories):
        # Plot first component of each trajectory
        plt.plot(
            sol.t,
            sol_batch[traj_idx, 0, :],
            color=colors[traj_idx],
            alpha=0.7,
            label=f"Traj {traj_idx + 1}",
        )

    plt.xlabel("Time")
    plt.ylabel("X₀ State")
    plt.title(f"Van der Pol: {n_trajectories} Trajectories (First Component)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vanderpol_test.png", dpi=150)
    plt.close()
    print(f"\nPlot saved as 'vanderpol_test.png' ({n_trajectories} trajectories)")

    # Test 3: Evaluate the Jacobian
    print("\nTest 3: Evaluating Jacobian")
    X_single = rng.standard_normal((1, dim))
    J = vanderpol_grad(X_single, mu=mu)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian at sample point:\n{J[0]}")
