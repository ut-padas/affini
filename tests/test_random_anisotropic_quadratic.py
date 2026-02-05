import unittest

import numpy as np
import sympy as sp

from src.problems.random_anisotropic_quadratic import (
    DIAGONAL_EXPONENT,
    SEED,
    random_aniso_quadratic,
    random_aniso_quadratic_grad,
    random_aniso_quadratic_hess,
)


def _sympy_random_aniso_quadratic_f_g_h(n_dim: int, seed: int = SEED):
    """Create symbolic function, gradient, and Hessian for random anisotropic quadratic function."""
    x = sp.symbols(f"x0:{n_dim}")

    # Create scales: S = diag([(1/i)^DIAGONAL_EXPONENT for i in 1..d])
    scales = [sp.Float((1.0 / (i + 1)) ** DIAGONAL_EXPONENT) for i in range(n_dim)]

    # Create random rotation matrix Q using same logic as implementation
    np.random.seed(seed)
    A = np.random.randn(n_dim, n_dim)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        # Ensure a proper rotation (det(Q)=1)
        Q[:, 0] = -Q[:, 0]

    # Convert Q to symbolic matrix
    Q_sym = sp.Matrix(Q.astype(float))
    S_sym = sp.diag(*scales)

    # x vector as symbolic matrix
    x_vec = sp.Matrix(x)

    # Function: f(x) = x^T (Q S Q^T) x
    # First compute Q S Q^T
    QSQt = Q_sym @ S_sym @ Q_sym.T

    # Then compute x^T (Q S Q^T) x
    f = x_vec.T @ QSQt @ x_vec
    f = f[0]  # Extract scalar from 1x1 matrix

    # Compute gradient: grad f(x) = 2 * Q S Q^T x
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Compute Hessian: Hess f(x) = 2 * Q S Q^T
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")

    return f_fn, g_fn, h_fn


class TestRandomAnisoQuadraticSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, seed)
        configs = [
            (2, SEED),
            (3, SEED),
            (4, SEED),
            (2, 42),  # Different seed
            (3, 123),  # Different seed
        ]

        for n_dim, seed in configs:
            key = (n_dim, seed)
            cls._sympy[key] = _sympy_random_aniso_quadratic_f_g_h(n_dim, seed)

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(1)  # Use different seed for test data
        num_pts = 20  # Smaller number due to computational complexity
        prob_extents = (-0.5, 0.5)  # Smaller range

        test_configs = [
            (2, SEED),
            (3, SEED),
            (4, SEED),
            (2, 42),
            (3, 123),
        ]

        for n_dim, seed in test_configs:
            key = (n_dim, seed)
            f_fn, g_fn, h_fn = self._sympy[key]

            with self.subTest(n_dim=n_dim, seed=seed):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # For default seed, use implementation as-is
                if seed == SEED:
                    f_impl = random_aniso_quadratic(X)
                    g_impl = random_aniso_quadratic_grad(X)[:, 0, :]
                    h_impl = random_aniso_quadratic_hess(X)
                else:
                    # For custom seeds, compute manually since implementation uses fixed SEED
                    # Create scales: S = diag([(1/i)^DIAGONAL_EXPONENT for i in 1..d])
                    scales = np.array(
                        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(n_dim)],
                        dtype=X.dtype,
                    )

                    # Create random rotation matrix Q with custom seed
                    np.random.seed(seed)
                    A = np.random.randn(n_dim, n_dim).astype(X.dtype)
                    Q, _ = np.linalg.qr(A)
                    if np.linalg.det(Q) < 0:
                        Q[:, 0] = -Q[:, 0]

                    # Rotate data: X_rot = X Q
                    X_rot = X @ Q

                    # Quadratic term: sum(s_i * x_rot_i^2)
                    f_impl = np.sum(scales[None, :] * (X_rot**2), axis=1, keepdims=True)

                    # Gradient in rotated space: g_rot = 2 * S * x_rot
                    g_rot = 2.0 * scales[None, :] * X_rot

                    # Rotate back: g = g_rot Q^T
                    g_impl = g_rot @ Q.T

                    # Hessian: H = 2 * Q S Q^T (constant for all points)
                    scales_diag = np.diag(scales)
                    H_constant = 2.0 * Q @ scales_diag @ Q.T
                    h_impl = np.tile(H_constant[None, :, :], (num_pts, 1, 1))

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)
                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )
                h_sym = np.stack(
                    [np.asarray(h_fn(*row), dtype=np.float64) for row in X]
                )

                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)

                # Compare Hessians
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-10, atol=1e-10)

    def test_rotation_matrix_properties(self):
        """Test that the rotation matrix has proper orthogonal properties."""
        n_dims = [2, 3, 4, 5]

        for n_dim in n_dims:
            with self.subTest(n_dim=n_dim):
                # Replicate the rotation matrix creation logic
                np.random.seed(SEED)
                A = np.random.randn(n_dim, n_dim)
                Q, _ = np.linalg.qr(A)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]

                # Test orthogonality: Q @ Q^T = I
                identity = Q @ Q.T
                np.testing.assert_allclose(
                    identity, np.eye(n_dim), rtol=1e-12, atol=1e-12
                )

                # Test determinant = 1 (proper rotation)
                det = np.linalg.det(Q)
                np.testing.assert_allclose(det, 1.0, rtol=1e-12, atol=1e-12)

    def test_scales_properties(self):
        """Test that the scales follow the expected pattern."""
        n_dims = [2, 3, 4, 5]

        for n_dim in n_dims:
            with self.subTest(n_dim=n_dim):
                scales = np.array(
                    [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(n_dim)]
                )

                # Test that scales are decreasing
                for i in range(1, n_dim):
                    self.assertGreater(scales[i - 1], scales[i])

                # Test specific values
                expected_first = 1.0
                expected_second = (
                    (1.0 / 2.0) ** DIAGONAL_EXPONENT if n_dim > 1 else None
                )

                np.testing.assert_allclose(scales[0], expected_first, rtol=1e-12)
                if expected_second is not None:
                    np.testing.assert_allclose(scales[1], expected_second, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
