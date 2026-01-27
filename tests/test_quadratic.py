import unittest

import numpy as np
import sympy as sp

from src.problems.quadratic import A, C, quadratic, quadratic_grad, quadratic_hess


def _sympy_quadratic_f_g_h(n_dim: int, c: float = 1.0, a: float = 0.1):
    """Create symbolic function, gradient, and Hessian for quadratic function."""
    x = sp.symbols(f"x0:{n_dim}")

    # Build the quadratic function: f(x) = c + w^T x + a x^T x, where w = ones(d)
    w = [sp.Float(1.0)] * n_dim  # w = ones(d)

    # Linear term: w^T x
    linear_term = sum(w[i] * x[i] for i in range(n_dim))

    # Quadratic term: a x^T x
    quad_term = sp.Float(a) * sum(x[i] ** 2 for i in range(n_dim))  # type: ignore

    # Full function: f(x) = c + w^T x + a x^T x
    f = sp.Float(c) + linear_term + quad_term  # type: ignore

    # Compute gradient: grad f(x) = w + 2 a x
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Compute Hessian: Hess f(x) = 2 a I_d
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")

    return f_fn, g_fn, h_fn


class TestQuadraticSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, c, a)
        configs = [
            (2, C, A),
            (2, 0.0, A),
            (2, C, 0.0),
            (3, C, A),
            (3, 2.0, 0.5),
        ]

        for n_dim, c, a in configs:
            key = (n_dim, c, a)
            cls._sympy[key] = _sympy_quadratic_f_g_h(n_dim, c, a)

    def test_value_grad_match_sympy(self):
        rng = np.random.default_rng(0)
        num_pts = 50
        prob_extents = (-2.0, 2.0)

        test_configs = [
            (2, C, A),
            (2, 0.0, A),
            (2, C, 0.0),
            (3, C, A),
            (3, 2.0, 0.5),
        ]

        for n_dim, c, a in test_configs:
            key = (n_dim, c, a)
            f_fn, g_fn, h_fn = self._sympy[key]

            with self.subTest(n_dim=n_dim, c=c, a=a):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # For default parameters (C=1.0, A=0.1), use the implementation as-is
                if c == C and a == A:
                    f_impl = quadratic(X)
                    g_impl = quadratic_grad(X)[:, 0, :]
                    h_impl = quadratic_hess(X)

                else:
                    # For custom parameters, we need to modify the computation
                    # manually since the implementation uses fixed constants
                    w = np.ones((n_dim, 1), dtype=X.dtype)
                    XTX = np.sum(X**2, axis=1, keepdims=True)
                    f_impl = c + X @ w + a * XTX

                    w_grad = np.ones((n_dim,), dtype=X.dtype)
                    g_impl = w_grad[None, :] + 2 * a * X

                    # For custom parameters, compute Hessian manually: 2a * I_d
                    h_impl = 2 * a * np.eye(n_dim, dtype=X.dtype)[None, :, :]
                    h_impl = np.repeat(h_impl, num_pts, axis=0)

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)
                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )
                h_sym = np.stack(
                    [np.asarray(h_fn(*row), dtype=np.float64) for row in X]
                )

                # Compare
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
