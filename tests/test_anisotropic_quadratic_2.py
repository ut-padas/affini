"""
Tests for src/problems/anisotropic_quadratic_2.py (used by the paper's
uniform-sampling anisotropic-quadratic experiments).

f(x) = beta + x^T H x,  H = Q diag(decay_base^-1, ..., decay_base^-d) Q^T
"""

import unittest

import numpy as np
import sympy as sp

from src.problems.anisotropic_quadratic_2 import (
    BETA,
    DECAY_BASE,
    SEED,
    _get_params,
    _get_random_rotation,
    aniso_quadratic,
    aniso_quadratic_grad,
    aniso_quadratic_hess,
)


def _sympy_aniso_quadratic_2_f_g_h(n_dim: int, beta: float, decay_base: float):
    """Independent SymPy reference: f(x) = beta + x^T H x with geometric eigenvalue decay."""
    x = sp.symbols(f"x0:{n_dim}")

    # Same rotation construction (and seed) as the implementation
    np.random.seed(SEED)
    A = np.random.randn(n_dim, n_dim).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Eigenvalues: Lambda_ii = decay_base^(-i), i = 1..d
    lambdas = [sp.Float(decay_base) ** (-i) for i in range(1, n_dim + 1)]
    H = sp.Matrix(Q) @ sp.diag(*lambdas) @ sp.Matrix(Q).T

    x_vec = sp.Matrix(x)
    f = sp.Float(beta) + (x_vec.T @ H @ x_vec)[0]
    g = sp.Matrix([sp.diff(f, xi) for xi in x])
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    return (
        sp.lambdify(x, f, "numpy"),
        sp.lambdify(x, g, "numpy"),
        sp.lambdify(x, h, "numpy"),
    )


class TestAnisoQuadratic2(unittest.TestCase):
    # (dimension, beta, decay_base)
    test_configs = [
        (2, BETA, DECAY_BASE),
        (3, BETA, DECAY_BASE),
        (4, 5.0, 2.0),
        (5, 15.0, 10.0),
    ]

    @classmethod
    def setUpClass(cls):
        cls._sympy = {
            cfg: _sympy_aniso_quadratic_2_f_g_h(*cfg) for cfg in cls.test_configs
        }

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(42)
        for n_dim, beta, decay_base in self.test_configs:
            f_fn, g_fn, h_fn = self._sympy[(n_dim, beta, decay_base)]
            with self.subTest(n_dim=n_dim, beta=beta, decay_base=decay_base):
                X = rng.uniform(-2.0, 2.0, size=(20, n_dim))

                f_impl = aniso_quadratic(X, beta=beta, decay_base=decay_base)
                g_impl = aniso_quadratic_grad(X, decay_base=decay_base)[:, 0, :]
                h_impl = aniso_quadratic_hess(X, decay_base=decay_base)

                f_sym = np.array([float(f_fn(*row)) for row in X]).reshape(-1, 1)
                g_sym = np.stack([np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X])
                h_sym = np.stack([np.asarray(h_fn(*row), dtype=np.float64) for row in X])

                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-12, atol=1e-14)

    def test_rotation_matrix_properties(self):
        Q = _get_random_rotation(3, np.float64)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(Q), 1.0, places=10)

    def test_h_properties(self):
        n_dim = 4
        H = _get_params(n_dim, np.float64)

        np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-14)  # symmetric

        # Eigenvalues are exactly decay_base^-1, ..., decay_base^-d (so H is positive definite)
        eigenvals = np.sort(np.linalg.eigvalsh(H))[::-1]
        expected = DECAY_BASE ** -np.arange(1, n_dim + 1, dtype=np.float64)
        np.testing.assert_allclose(eigenvals, expected, rtol=1e-10, atol=1e-14)

    def test_decay_base_argument_is_used(self):
        """Regression: _get_params used the module constant instead of its argument."""
        n_dim, other_base = 3, 2.0
        self.assertNotEqual(other_base, DECAY_BASE)

        eigenvals = np.sort(np.linalg.eigvalsh(_get_params(n_dim, np.float64, other_base)))[::-1]
        np.testing.assert_allclose(eigenvals, other_base ** -np.arange(1, n_dim + 1.0), rtol=1e-10)

        X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(10, n_dim))
        self.assertFalse(np.allclose(aniso_quadratic(X, decay_base=other_base), aniso_quadratic(X)))

    def test_origin_is_minimum(self):
        n_dim = 3
        X0 = np.zeros((1, n_dim))
        np.testing.assert_allclose(aniso_quadratic(X0), [[BETA]], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(aniso_quadratic_grad(X0), np.zeros((1, 1, n_dim)), atol=1e-14)

        # H is positive definite, so every other point lies strictly above the minimum
        X = np.random.default_rng(1).uniform(-1.0, 1.0, size=(50, n_dim))
        self.assertTrue(np.all(aniso_quadratic(X) > BETA))

    def test_shapes_and_constant_hessian(self):
        X = np.random.default_rng(2).uniform(-1.0, 1.0, size=(10, 2))
        self.assertEqual(aniso_quadratic(X).shape, (10, 1))
        self.assertEqual(aniso_quadratic_grad(X).shape, (10, 1, 2))

        h_vals = aniso_quadratic_hess(X)
        self.assertEqual(h_vals.shape, (10, 2, 2))
        np.testing.assert_allclose(h_vals, np.broadcast_to(h_vals[0], h_vals.shape), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
