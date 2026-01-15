import unittest

import numpy as np
import sympy as sp

from src.problems.rosenbrock import (
    ROSENBROCK_SCALE,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)


def _sympy_rosenbrock_f_g_h(n_dim: int):
    x = sp.symbols(f"x0:{n_dim}")
    f = sp.Integer(0)
    for i in range(n_dim - 1):
        f += ROSENBROCK_SCALE * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    g = sp.Matrix([sp.diff(f, xi) for xi in x])
    h = sp.hessian(f, x)

    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")
    return f_fn, g_fn, h_fn


class TestRosenbrockSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {d: _sympy_rosenbrock_f_g_h(d) for d in (2, 5)}

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(0)
        num_pts = 60
        prob_extents = (-3.0, 3.0)

        for n_dim, (f_fn, g_fn, h_fn) in self._sympy.items():
            with self.subTest(n_dim=n_dim):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                f_impl = rosenbrock(X)  # (n, 1)
                g_impl = rosenbrock_grad(X)[:, 0, :]  # (n, d)
                h_impl = rosenbrock_hess(X)  # (n, d, d)

                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)[
                    :, None
                ]
                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )
                h_sym = np.stack(
                    [np.asarray(h_fn(*row), dtype=np.float64) for row in X]
                )

                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-9, atol=1e-9)
