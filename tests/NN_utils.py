import numpy as np

"""Shifted Gaussian function and its derivatives."""
ROSENBROCK_SCALE = 100.0



CENTERS_2D = np.array([[1.0, 1.0], [-1.0, -1.0]])
CENTERS_3D = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0]])
CENTERS_4D = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ]
)
CENTERS_5D = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0, 1.0],
    ]
)


def shifted_gaussians(
    X,
    centers,
    shift: float = 1.0,
    std_dev: float = 0.5,
):
    """
    Computes the shifted Gaussian function for D-dimensional input points.

    Args:
        X: (N, D) data matrix
        centers: (M, D) centers of the Gaussians
        shift: additive shift
        std_dev: standard deviation of the Gaussians

    Returns:
        F: (N, 1) function values
    """

    # X: (N, D) -> (N, 1, D)
    # centers: (M, D) -> (1, M, D)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]

    # Squared Euclidean distance: (N, M)
    sq_dist = np.sum(diff**2, axis=2)

    # Gaussian sum + shift
    f_x = np.sum(np.exp(-sq_dist / (2 * std_dev**2)), axis=1) + shift

    return f_x[:, np.newaxis]  # (N, 1)


def rosenbrock(X, a: float = ROSENBROCK_SCALE):
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (N, d) data matrix
        a: scale parameter

    Returns:
        F: (N, 1) function values at X
    """
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = np.sum(
        a * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0,
        axis=1,
        keepdims=True,
    )
    return F

def rosenbrock_grad(X, a: float = 100.0):
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (N, d) data matrix
        a: scale parameter

    Returns:
        J: (N, 1, d) Jacobian at X
    """
    N, d = X.shape
    J = np.zeros((N, 1, d), dtype=X.dtype)

    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)
    J[:, 0, :-1] -= 4 * a * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)
    J[:, 0, 1:] += 2 * a * (X[:, 1:] - X[:, :-1] ** 2)
    return J




def shifted_gaussians_grad(X, centers, std_dev: float = 0.5):
    """
    Computes the gradient of the shifted Gaussian function.

    Args:
        X: (N, D) data matrix
        centers: (M, D) centers of the Gaussians
        std_dev: standard deviation of the Gaussians

    Returns:
        grad_f_x: (N, 1, D) gradient at each point in X
    """
    # X: (N, D), centers: (M, D)
    # diff shape: (N, M, D)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]

    # sq_dist shape: (N, M)
    sq_dist = np.sum(diff**2, axis=2)

    # gaussians shape: (N, M)
    gaussians = np.exp(-sq_dist / (2 * std_dev**2))

    # 1. Compute coefficients: shape (N, M)
    # 2. Reshape coefficients to (N, M, 1) to multiply with diff (N, M, D)
    # 3. Sum across the M centers (axis 1)
    coeff = -gaussians / (std_dev**2)
    grad_f_x = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad_f_x[:, np.newaxis, :]  # Return shape (N, 1, D)



def lorenz96(X, forcing=8.0):
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

# test_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/TestData/SG_2D.npz"
# test_pts = np.load(test_path)
# print(list(test_pts.keys()))
# x = test_pts['X_test']
# print(x.shape)

# f = shifted_gaussians(x, CENTERS_2D) 
# f = rosenbrock(x)
# print(f.shape)