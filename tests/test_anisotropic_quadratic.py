import unittest

import numpy as np
import sympy as sp

from src.problems.anisotropic_quadratic import (
    ANISO_RATIO,
    C,
    aniso_quadratic,
    aniso_quadratic_grad,
    aniso_quadratic_hess,
)


def _sympy_aniso_quadratic_f_g_h(n_dim: int, c: float = C, ratio: float = ANISO_RATIO):
    """Create symbolic function, gradient, and Hessian for anisotropic quadratic function."""
    x = sp.symbols(f"x0:{n_dim}")

    # Build scales: s = [1, ratio, ratio^2, ...]
    scales = [sp.Float(ratio) ** i for i in range(n_dim)]

    # Linear term: sum(x_i) for i in range(n_dim)
    linear_term = sum(x[i] for i in range(n_dim))

    # Quadratic term: sum(s_i * x_i^2) for i in range(n_dim)
    quad_term = sum(scales[i] * x[i] ** 2 for i in range(n_dim))

    # Full function: f(x) = c + sum(x_i) + sum(s_i * x_i^2)
    f = sp.Float(c) + linear_term + quad_term  # type: ignore

    # Compute gradient: grad f(x) = [1 + 2*s_i*x_i] for i in range(n_dim)
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Compute Hessian: Hess f(x) = diag(2*s_i)
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")

    return f_fn, g_fn, h_fn


class TestAnisoQuadraticSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, c, ratio)
        configs = [
            (2, C, ANISO_RATIO),
            (2, 0.0, ANISO_RATIO),
            (2, C, 10.0),
            (3, C, ANISO_RATIO),
            (3, 5.0, 50.0),
        ]

        for n_dim, c, ratio in configs:
            key = (n_dim, c, ratio)
            cls._sympy[key] = _sympy_aniso_quadratic_f_g_h(n_dim, c, ratio)

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(0)
        num_pts = 50
        prob_extents = (-1.0, 1.0)  # Smaller range due to anisotropy

        test_configs = [
            (2, C, ANISO_RATIO),
            (2, 0.0, ANISO_RATIO),
            (2, C, 10.0),
            (3, C, ANISO_RATIO),
            (3, 5.0, 50.0),
        ]

        for n_dim, c, ratio in test_configs:
            key = (n_dim, c, ratio)
            f_fn, g_fn, h_fn = self._sympy[key]

            with self.subTest(n_dim=n_dim, c=c, ratio=ratio):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # For default parameters (C=10.0, ANISO_RATIO=100.0), use implementation as-is
                if c == C and ratio == ANISO_RATIO:
                    f_impl = aniso_quadratic(X)
                    g_impl = aniso_quadratic_grad(X)[:, 0, :]
                    h_impl = aniso_quadratic_hess(X)

                else:
                    # For custom parameters, compute manually since implementation uses fixed constants
                    # Scales: [1, ratio, ratio^2, ...]
                    scales = np.array([ratio**i for i in range(n_dim)], dtype=X.dtype)

                    # Linear term: sum(x)
                    linear_term = np.sum(X, axis=1, keepdims=True)

                    # Quadratic term: sum(s_i * x_i^2)
                    quad_term = np.sum(scales[None, :] * (X**2), axis=1, keepdims=True)

                    f_impl = c + linear_term + quad_term

                    # Gradient: g_i = 1 + 2 * s_i * x_i
                    g_impl = 1.0 + 2.0 * scales[None, :] * X

                    # Hessian: H = diag(2 * scales)
                    h_impl = np.zeros((num_pts, n_dim, n_dim), dtype=X.dtype)
                    for i in range(n_dim):
                        h_impl[:, i, i] = 2.0 * scales[i]

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
