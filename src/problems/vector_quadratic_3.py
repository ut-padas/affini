# A third quadratic which returns identical multi-output.

from src import Array, xp

H = xp.diag(xp.array([10, 1]))


def vector_quadratic_3(X: Array, d_out=5) -> Array:
    """
    Computes f(x) = (<x, H1*x>, <x, H2*x>), which maps R2 to R2.
    For now assume H1 and H2 are diagonal matrices.
    Takes in X with shape (N, 2) and returns matrix (N, d).
    """
    N, d_in = X.shape
    val = xp.sum(X * (H @ X.T).T, axis=1)  # (N,)
    return xp.repeat(val[:, None], d_out, axis=1)  # (N, d_out)


def vector_quadratic_3_grad(X: Array, d_out=5) -> Array:
    """
    Computes the gradient: f'(x) = (2*H1*x, 2*H2*x)
    Takes in X with shape (N, 2) and returns tensor (N, d, 2).
    Output axes:
    - 0: batch dimension (N)
    - 1: output dimension (d) (one for each output component)
    - 2: input dimension (2)
    """
    g = 2.0 * (X @ H)  # (N, d_out)
    return xp.repeat(g[:, None, :], d_out, axis=1)  # (N, d_out, d_in)


def vector_quadratic_3_hess(X: Array, d_out=5) -> Array:
    """
    Computes the Hessian: f''(x) = (2*H1, 2*H2)
    Takes in X with shape (N, 2) and returns tensor (N, d_out, d_in, d_in).
    """
    N, d_in = X.shape
    h = xp.repeat((2.0 * H)[None, :, :], N, axis=0)  # Shape (N, 2, 2)
    return xp.repeat(h[:, None, :, :], d_out, axis=1)  # (N, d, 2, 2)


if __name__ == "__main__":
    X = xp.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    print("Function values:")
    print(vector_quadratic_3(X))
    print(vector_quadratic_3(X).shape)
    print("Gradients:")
    print(vector_quadratic_3_grad(X))
    print(vector_quadratic_3_grad(X).shape)
    print("Hessians:")
    print(vector_quadratic_3_hess(X))
    print(vector_quadratic_3_hess(X).shape)
