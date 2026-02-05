import unittest

import numpy as np
import sympy as sp

from src.problems.anisotropic_quadratic import (
    BETA,
    DECAY_EXPONENT,
    SEED,
    _get_params,
    _get_random_rotation,
    aniso_quadratic,
    aniso_quadratic_grad,
    aniso_quadratic_hess,
)


def _sympy_aniso_quadratic_f_g_h(
    n_dim: int,
    beta: float = BETA,
    decay_exp: float = DECAY_EXPONENT,
):
    """Create symbolic function, gradient, and Hessian for anisotropic quadratic function."""
    x = sp.symbols(f"x0:{n_dim}")

    # Create random rotation matrix Q using the same seed and method as implementation
    np.random.seed(SEED)
    A = np.random.randn(n_dim, n_dim).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Convert to sympy matrix
    Q_sym = sp.Matrix(Q)

    # Create eigenvalues: Lambda_ii = i^(-decay_exp)
    indices = np.arange(1, n_dim + 1)
    lambdas = indices ** (-decay_exp)

    # Create diagonal lambda matrix
    Lambda_diag = sp.diag(*[sp.Float(lam) for lam in lambdas])

    # Create weight vector w = [1, 1, ..., 1]
    w = sp.Matrix([1.0] * n_dim)

    # Full H matrix: H = Q * Lambda * Q^T
    H = Q_sym @ Lambda_diag @ Q_sym.T

    # Create x vector
    x_vec = sp.Matrix(x)

    # Function: f(x) = beta + w^T * x + x^T * H * x
    linear_term = w.T @ x_vec
    quadratic_term = x_vec.T @ H @ x_vec

    # Extract scalars from 1x1 matrices
    linear_term = linear_term[0]
    quadratic_term = quadratic_term[0]

    f = sp.Float(beta) + linear_term + quadratic_term

    # Compute gradient
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Compute Hessian
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")

    return f_fn, g_fn, h_fn, Q, lambdas


class TestAnisoQuadraticSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, beta, decay_exp)
        cls.test_configs = [
            (2, BETA, DECAY_EXPONENT),
            (3, BETA, DECAY_EXPONENT),
            (4, 5.0, 2.0),
            (5, 15.0, 4.0),
        ]

        for n_dim, beta, decay_exp in cls.test_configs:
            key = (n_dim, beta, decay_exp)
            cls._sympy[key] = _sympy_aniso_quadratic_f_g_h(n_dim, beta, decay_exp)

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(42)  # Different seed for test data
        num_pts = 20
        prob_extents = (-2.0, 2.0)  # Reasonable range for quadratic function

        for n_dim, beta, decay_exp in self.test_configs:
            key = (n_dim, beta, decay_exp)
            f_fn, g_fn, h_fn, Q_expected, lambdas_expected = self._sympy[key]

            with self.subTest(n_dim=n_dim, beta=beta, decay_exp=decay_exp):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # Use implementation with parameters
                f_impl = aniso_quadratic(X, beta=beta, decay_exponent=decay_exp)
                g_impl = aniso_quadratic_grad(X, decay_exponent=decay_exp)[:, 0, :]
                h_impl = aniso_quadratic_hess(X, decay_exponent=decay_exp)

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)

                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )

                h_sym = np.stack(
                    [np.asarray(h_fn(*row), dtype=np.float64) for row in X]
                )

                # Compare with standard tolerances for quadratic functions
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-12, atol=1e-14)

    def test_rotation_matrix_properties(self):
        """Test that the rotation matrix has proper orthogonal properties."""
        n_dim = 3

        # Import the actual function to test

        Q = _get_random_rotation(n_dim, np.float64)

        # Test orthogonality: Q^T @ Q should be identity
        identity_test = Q.T @ Q
        np.testing.assert_allclose(identity_test, np.eye(n_dim), rtol=1e-10, atol=1e-12)

        # Test determinant is 1 (proper rotation)
        det = np.linalg.det(Q)
        self.assertAlmostEqual(det, 1.0, places=10)

    def test_eigenvalues_properties(self):
        """Test properties of the eigenvalues."""

        n_dim = 4
        w, H = _get_params(n_dim, np.float64, DECAY_EXPONENT)

        # Weight vector should be all ones
        expected_w = np.ones(n_dim)
        np.testing.assert_allclose(w, expected_w, rtol=1e-12, atol=1e-14)

        # H should be symmetric
        np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-14)

        # H should be positive definite (all eigenvalues > 0)
        eigenvals = np.linalg.eigvals(H)
        self.assertTrue(np.all(eigenvals > 0))

        # Eigenvalues should follow the decay pattern
        eigenvals_sorted = np.sort(eigenvals)[::-1]  # Sort in descending order

        # The largest eigenvalue should correspond to 1^(-DEFAULT_DECAY_EXPONENT) = 1
        expected_largest = 1.0 ** (-DECAY_EXPONENT)
        self.assertAlmostEqual(eigenvals_sorted[0], expected_largest, places=10)

    def test_function_properties(self):
        """Test basic properties of the quadratic function."""
        rng = np.random.default_rng(123)
        X = rng.uniform(-1.0, 1.0, size=(50, 2)).astype(np.float64)

        f_vals = aniso_quadratic(X)

        # Function values should be finite
        self.assertTrue(np.all(np.isfinite(f_vals)))

        # At origin, function should equal beta (since w^T * 0 + 0^T * H * 0 = 0)
        X_origin = np.zeros((1, 2), dtype=np.float64)
        f_origin = aniso_quadratic(X_origin)
        np.testing.assert_allclose(f_origin, [[BETA]], rtol=1e-12, atol=1e-14)

    def test_gradient_properties(self):
        """Test properties of the gradient."""
        rng = np.random.default_rng(456)
        X = rng.uniform(-1.0, 1.0, size=(20, 3)).astype(np.float64)

        g_vals = aniso_quadratic_grad(X)

        # Gradient should be finite
        self.assertTrue(np.all(np.isfinite(g_vals)))

        # Shape should be (N, 1, d)
        self.assertEqual(g_vals.shape, (20, 1, 3))

        # At origin, gradient should equal w = [1, 1, 1]
        X_origin = np.zeros((1, 3), dtype=np.float64)
        g_origin = aniso_quadratic_grad(X_origin)
        expected_grad_origin = np.ones((1, 1, 3))
        np.testing.assert_allclose(
            g_origin, expected_grad_origin, rtol=1e-12, atol=1e-14
        )

    def test_hessian_properties(self):
        """Test properties of the Hessian."""
        rng = np.random.default_rng(789)
        X = rng.uniform(-1.0, 1.0, size=(10, 2)).astype(np.float64)

        h_vals = aniso_quadratic_hess(X)

        # Hessian should be finite
        self.assertTrue(np.all(np.isfinite(h_vals)))

        # Shape should be (N, d, d)
        self.assertEqual(h_vals.shape, (10, 2, 2))

        # Hessian should be constant (same for all points)
        for i in range(1, 10):
            np.testing.assert_allclose(h_vals[i], h_vals[0], rtol=1e-12, atol=1e-14)

        # Each Hessian should be symmetric
        for i in range(10):
            np.testing.assert_allclose(h_vals[i], h_vals[i].T, rtol=1e-12, atol=1e-14)

        # Each Hessian should be positive definite
        for i in range(10):
            eigenvals = np.linalg.eigvals(h_vals[i])
            self.assertTrue(np.all(eigenvals > 0))


if __name__ == "__main__":
    unittest.main()
