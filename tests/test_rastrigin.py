import unittest

import numpy as np
import sympy as sp

from src.problems.rastrigin import RASTRIGIN_A, rastrigin, rastrigin_grad


def _sympy_rastrigin_f_g(n_dim: int, A: float = 10.0):
    """Create symbolic function and gradient for Rastrigin function."""
    x = sp.symbols(f"x0:{n_dim}")

    # Build the Rastrigin function: f(x) = A*d + sum(x^2 - A*cos(2*pi*x))
    d = sp.Float(n_dim)
    A_sym = sp.Float(A)

    # Sum term: sum(x^2 - A*cos(2*pi*x))
    sum_term = sum(x[i] ** 2 - A_sym * sp.cos(2 * sp.pi * x[i]) for i in range(n_dim))  # type: ignore

    # Full function: f(x) = A*d + sum_term
    f = A_sym * d + sum_term  # type: ignore

    # Compute gradient: df/dx_i = 2*x_i + 2*pi*A*sin(2*pi*x_i)
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")

    return f_fn, g_fn


class TestRastriginSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, A)
        configs = [
            (2, RASTRIGIN_A),
            (2, 5.0),
            (2, 1.0),
            (3, RASTRIGIN_A),
            (3, 20.0),
            (4, 15.0),
        ]

        for n_dim, A in configs:
            key = (n_dim, A)
            cls._sympy[key] = _sympy_rastrigin_f_g(n_dim, A)

    def test_value_grad_match_sympy(self):
        rng = np.random.default_rng(42)
        num_pts = 50
        prob_extents = (-5.12, 5.12)  # Standard Rastrigin domain

        test_configs = [
            (2, RASTRIGIN_A),
            (2, 5.0),
            (2, 1.0),
            (3, RASTRIGIN_A),
            (3, 20.0),
            (4, 15.0),
        ]

        for n_dim, A in test_configs:
            key = (n_dim, A)
            f_fn, g_fn = self._sympy[key]

            with self.subTest(n_dim=n_dim, A=A):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # Compute using implementation
                f_impl = rastrigin(X, A)
                g_impl = rastrigin_grad(X, A)[:, 0, :]

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)
                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )

                # Compare
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)

    def test_global_minimum(self):
        """Test that global minimum is at origin with value 0."""
        # Test at origin
        X = np.zeros((1, 2))
        f_val = rastrigin(X, RASTRIGIN_A)
        g_val = rastrigin_grad(X, RASTRIGIN_A)

        np.testing.assert_allclose(f_val, 0.0, atol=1e-12)
        np.testing.assert_allclose(g_val, 0.0, atol=1e-12)

    def test_gradient_shape(self):
        """Test gradient shape consistency."""
        rng = np.random.default_rng(123)

        for n_dim in [2, 3, 5]:
            for n_points in [1, 10, 50]:
                with self.subTest(n_dim=n_dim, n_points=n_points):
                    X = rng.uniform(-2.0, 2.0, size=(n_points, n_dim))
                    g = rastrigin_grad(X, RASTRIGIN_A)

                    expected_shape = (n_points, 1, n_dim)
                    self.assertEqual(g.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
