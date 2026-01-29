import unittest

import numpy as np
import sympy as sp

from src.problems.vector_quadratic import (
    vector_quadratic,
    vector_quadratic_grad,
    vector_quadratic_hess,
)


def _get_diagonals_sympy(n_dim: int):
    """
    Create symbolic diagonal matrices A1 and A2 following the pattern:
    A1: 100 on Odd indices (0, 2...), 1 on Even indices (1, 3...)
    A2: 1 on Odd indices (0, 2...), 100 on Even indices (1, 3...)
    """
    diag1 = [sp.Float(100.0) if i % 2 == 0 else sp.Float(1.0) for i in range(n_dim)]
    diag2 = [sp.Float(1.0) if i % 2 == 0 else sp.Float(100.0) for i in range(n_dim)]

    A1 = sp.diag(*diag1)
    A2 = sp.diag(*diag2)

    return A1, A2


def _sympy_vector_quadratic_f_g_h(n_dim: int):
    """Create symbolic functions, gradients, and Hessians for vector quadratic functions."""
    x = sp.symbols(f"x0:{n_dim}")
    x_vec = sp.Matrix(x)

    A1, A2 = _get_diagonals_sympy(n_dim)

    # f1(x) = (x - 1)^T A1 (x - 1)
    center1 = sp.ones(n_dim, 1)  # center at +1
    diff1 = x_vec - center1
    f1 = diff1.T @ A1 @ diff1
    f1 = f1[0, 0]  # Extract scalar from 1x1 matrix

    # f2(x) = (x + 1)^T A2 (x + 1)
    center2 = -sp.ones(n_dim, 1)  # center at -1
    diff2 = x_vec - center2
    f2 = diff2.T @ A2 @ diff2
    f2 = f2[0, 0]  # Extract scalar from 1x1 matrix

    # Compute gradients
    g1 = sp.Matrix([sp.diff(f1, xi) for xi in x])
    g2 = sp.Matrix([sp.diff(f2, xi) for xi in x])

    # Compute Hessians
    h1 = sp.Matrix([[sp.diff(f1, xi, xj) for xj in x] for xi in x])
    h2 = sp.Matrix([[sp.diff(f2, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f1_fn = sp.lambdify(x, f1, "numpy")
    f2_fn = sp.lambdify(x, f2, "numpy")
    g1_fn = sp.lambdify(x, g1, "numpy")
    g2_fn = sp.lambdify(x, g2, "numpy")
    h1_fn = sp.lambdify(x, h1, "numpy")
    h2_fn = sp.lambdify(x, h2, "numpy")

    return f1_fn, f2_fn, g1_fn, g2_fn, h1_fn, h2_fn


class TestVectorQuadraticSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension,)
        dims = [2, 3, 4, 5]

        for n_dim in dims:
            cls._sympy[n_dim] = _sympy_vector_quadratic_f_g_h(n_dim)

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(0)
        num_pts = 50
        prob_extents = (-2.0, 2.0)

        test_dims = [2, 3, 4, 5]

        for n_dim in test_dims:
            f1_fn, f2_fn, g1_fn, g2_fn, h1_fn, h2_fn = self._sympy[n_dim]

            with self.subTest(n_dim=n_dim):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # Compute using implementation
                f_impl = vector_quadratic(X)  # (N, 2)
                g_impl = vector_quadratic_grad(X)  # (N, 2, d)
                h_impl = vector_quadratic_hess(X)  # (N, 2, d, d)

                # Compute using SymPy
                f1_sym = np.array([float(f1_fn(*row)) for row in X], dtype=np.float64)
                f2_sym = np.array([float(f2_fn(*row)) for row in X], dtype=np.float64)
                f_sym = np.stack([f1_sym, f2_sym], axis=1)  # (N, 2)

                g1_sym = np.stack(
                    [np.asarray(g1_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )  # (N, d)
                g2_sym = np.stack(
                    [np.asarray(g2_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )  # (N, d)
                g_sym = np.stack([g1_sym, g2_sym], axis=1)  # (N, 2, d)

                h1_sym = np.stack(
                    [np.asarray(h1_fn(*row), dtype=np.float64) for row in X]
                )  # (N, d, d)
                h2_sym = np.stack(
                    [np.asarray(h2_fn(*row), dtype=np.float64) for row in X]
                )  # (N, d, d)
                h_sym = np.stack([h1_sym, h2_sym], axis=1)  # (N, 2, d, d)

                # Compare function values
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-12)

                # Compare gradients
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)

                # Compare Hessians
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-10, atol=1e-10)

    def test_diagonal_pattern_verification(self):
        """Test that the diagonal pattern matches expected values."""
        n_dim = 4

        # Create test point
        X = np.array([[0.5, -0.3, 1.2, -0.8]], dtype=np.float64)

        # Get Hessians
        H = vector_quadratic_hess(X)  # (1, 2, 4, 4)

        # Expected diagonal patterns
        expected_diag1 = [
            100.0,
            1.0,
            100.0,
            1.0,
        ]  # A1: 100 on odd indices (0,2), 1 on even (1,3)
        expected_diag2 = [
            1.0,
            100.0,
            1.0,
            100.0,
        ]  # A2: 1 on odd indices (0,2), 100 on even (1,3)

        # Check H1 (function 1)
        H1_diag = np.diag(H[0, 0, :, :])  # Extract diagonal of first Hessian
        np.testing.assert_array_equal(H1_diag, 2.0 * np.array(expected_diag1))

        # Check H2 (function 2)
        H2_diag = np.diag(H[0, 1, :, :])  # Extract diagonal of second Hessian
        np.testing.assert_array_equal(H2_diag, 2.0 * np.array(expected_diag2))

        # Verify off-diagonal elements are zero
        H1_off_diag = H[0, 0, :, :] - np.diag(H1_diag)
        H2_off_diag = H[0, 1, :, :] - np.diag(H2_diag)
        np.testing.assert_allclose(H1_off_diag, 0.0, atol=1e-15)
        np.testing.assert_allclose(H2_off_diag, 0.0, atol=1e-15)

    def test_centers_verification(self):
        """Test that the functions are minimized at their respective centers."""
        # f1 should be minimized at x = [1, 1, 1, ...]
        # f2 should be minimized at x = [-1, -1, -1, ...]

        for n_dim in [2, 3, 4]:
            with self.subTest(n_dim=n_dim):
                # Test points at centers
                center1 = np.ones((1, n_dim), dtype=np.float64)
                center2 = -np.ones((1, n_dim), dtype=np.float64)

                # Function values at centers should be zero
                f1_at_center = vector_quadratic(center1)
                f2_at_center = vector_quadratic(center2)

                np.testing.assert_allclose(
                    f1_at_center[0, 0], 0.0, atol=1e-15
                )  # f1 at center1
                np.testing.assert_allclose(
                    f2_at_center[0, 1], 0.0, atol=1e-15
                )  # f2 at center2

                # Gradients at centers should be zero
                g1_at_center = vector_quadratic_grad(center1)
                g2_at_center = vector_quadratic_grad(center2)

                np.testing.assert_allclose(
                    g1_at_center[0, 0, :], 0.0, atol=1e-15
                )  # grad f1 at center1
                np.testing.assert_allclose(
                    g2_at_center[0, 1, :], 0.0, atol=1e-15
                )  # grad f2 at center2


if __name__ == "__main__":
    unittest.main()
