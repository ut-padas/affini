"""Lorenz96 problem and its gradient."""

import numpy as np

from src import Array


def lorenz96(X: Array, forcing=8.0) -> Array:
    """
    Lorenz96 function implementation.

    Args:
        X: (N, d) data matrix where each row represents a state of the system.
        forcing: Forcing term in the Lorenz96 equations.

    Returns:
        dXdt: (n_samples, d) time derivatives at each state in X.
    """
    dXdt = (
        (np.roll(X, -1, axis=1) - np.roll(X, 2, axis=1)) * np.roll(X, 1, axis=1)
        - X
        + forcing
    )
    return dXdt


def lorenz96_grad(X) -> Array:
    """
    Gradient (Jacobian) of the Lorenz96 function.

    Args:
        X: (N, d) data matrix where each row represents a state of the system.

    Returns:
        J: (N, d, d) Jacobian matrices at each state in X.
    """
    N, d = X.shape
    J = np.zeros((N, d, d), dtype=X.dtype)

    X_im2 = np.roll(X, 2, axis=1)
    X_im1 = np.roll(X, 1, axis=1)
    X_ip1 = np.roll(X, -1, axis=1)

    # Create index arrays for the band diagonals
    i_idx = np.arange(d)
    im2_idx = (i_idx - 2) % d
    im1_idx = (i_idx - 1) % d
    ip1_idx = (i_idx + 1) % d

    # Populate Jacobian using vectorized indexing
    J[:, i_idx, im2_idx] = -X_im1
    J[:, i_idx, im1_idx] = X_ip1 - X_im2
    J[:, i_idx, i_idx] = -1
    J[:, i_idx, ip1_idx] = X_im1

    return J


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp

    # Example usage and testing
    dim = 30
    forcing = 8.0

    # Test 1: Evaluate Lorenz96 function on a grid
    print("Test 1: Evaluating Lorenz96 on a 2D grid")
    grid_points = 100
    x = np.linspace(-10, 10, grid_points)
    X1, X2 = np.meshgrid(x, x)
    X_test = np.zeros((grid_points * grid_points, dim))
    X_test[:, 0] = X1.ravel()
    X_test[:, 1] = X2.ravel()

    # Evaluate Lorenz96 directly on the grid
    dXdt = lorenz96(X_test, forcing=forcing)
    # Take the first component for visualization
    dXdt_component = dXdt[:, 0].reshape(grid_points, grid_points)

    # Plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(X1, X2, dXdt_component, levels=50, cmap="viridis")
    plt.colorbar(label="dX₀/dt")
    plt.xlabel("X₀")
    plt.ylabel("X₁")
    plt.title("Lorenz96: First Component Time Derivative")

    # Test 2: Batch integration of multiple trajectories
    print("\nTest 2: Batch integration of multiple trajectories")
    n_trajectories = 5
    x0_batch = np.random.randn(n_trajectories, dim)
    x0_flat = x0_batch.ravel()

    def lorenz96_batch_ode(t, y_flat):
        y_batch = y_flat.reshape(n_trajectories, dim)
        dydt_batch = lorenz96(y_batch, forcing=forcing)
        return dydt_batch.ravel()

    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 1000)

    # Single solve_ivp call integrates ALL trajectories simultaneously!
    sol = solve_ivp(lorenz96_batch_ode, t_span, x0_flat, t_eval=t_eval, method="RK45")

    # Reshape solution: (n_trajectories * dim, n_times) -> (n_trajectories, dim, n_times)
    sol_batch = sol.y.reshape(n_trajectories, dim, -1)

    plt.subplot(1, 2, 2)
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_trajectories))

    for traj_idx in range(n_trajectories):
        # Plot first component of each trajectory
        plt.plot(
            sol.t,
            sol_batch[traj_idx, 0, :],
            color=colors[traj_idx],
            alpha=0.7,
            label=f"Traj {traj_idx+1}",
        )

    plt.xlabel("Time")
    plt.ylabel("X₀ State")
    plt.title(f"Lorenz96: {n_trajectories} Trajectories (First Component)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lorenz96_test.png", dpi=150)
    print(f"\nPlot saved as 'lorenz96_test.png' ({n_trajectories} trajectories)")

    # Test 3: Verify Jacobian
    print("\nTest 3: Verifying Jacobian computation")
    X_single = np.random.randn(1, dim)
    J = lorenz96_grad(X_single)
    print(f"Jacobian shape: {J.shape}")
    print(f"Jacobian at sample point:\n{J[0]}")
