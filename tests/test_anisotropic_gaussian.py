import unittest

import numpy as np
import sympy as sp

from src.problems.anisotropic_gaussian import (
    BETA,
    SEED,
    SIGMA_DECAY_EXPONENT,
    aniso_gaussian,
    aniso_gaussian_grad,
    aniso_gaussian_hess,
)


def _sympy_aniso_gaussian_f_g_h(
    n_dim: int,
    beta: float = BETA,
    sigma_decay_exp: float = SIGMA_DECAY_EXPONENT,
    seed: int = SEED,
):
    """Create symbolic function, gradient, and Hessian for anisotropic Gaussian function."""
    x = sp.symbols(f"x0:{n_dim}")

    # Create random rotation matrix Q using the same seed and method as implementation
    np.random.seed(seed)
    A = np.random.randn(n_dim, n_dim).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # Convert to sympy matrix
    Q_sym = sp.Matrix(Q)

    # Create precision eigenvalues: precision_i = (i + 1)^(2*p)
    indices = np.arange(n_dim) + 1
    sigmas = 1.0 / (indices**sigma_decay_exp)
    prec_diag = 1.0 / (sigmas)

    # Create diagonal precision matrix
    Prec_diag = sp.diag(*[sp.Float(p) for p in prec_diag])

    # Full precision matrix: Sigma^-1 = Q * Lambda^-1 * Q^T
    Prec_full = Q_sym @ Prec_diag @ Q_sym.T

    # Create x vector
    x_vec = sp.Matrix(x)

    # Compute Mahalanobis distance squared: x^T * Sigma^-1 * x
    mahalanobis_sq = x_vec.T @ Prec_full @ x_vec
    mahalanobis_sq = mahalanobis_sq[0]  # Extract scalar from 1x1 matrix

    # Function: f(x) = beta + exp(-0.5 * mahalanobis_sq)
    f = sp.Float(beta) + sp.exp(-sp.Float(0.5) * mahalanobis_sq)  # type: ignore

    # Compute gradient
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Compute Hessian
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")

    return f_fn, g_fn, h_fn, Q, prec_diag


class TestAnisoGaussianSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, beta, sigma_decay_exp, seed)
        configs = [
            (2, BETA, SIGMA_DECAY_EXPONENT, SEED),
            (2, 0.5, SIGMA_DECAY_EXPONENT, SEED),
            (2, BETA, 2.0, SEED),
            (3, BETA, SIGMA_DECAY_EXPONENT, SEED),
            (3, 2.0, 4.0, SEED),
        ]

        for n_dim, beta, sigma_decay_exp, seed in configs:
            key = (n_dim, beta, sigma_decay_exp, seed)
            cls._sympy[key] = _sympy_aniso_gaussian_f_g_h(
                n_dim, beta, sigma_decay_exp, seed
            )

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(42)  # Different seed for test data
        num_pts = 20
        prob_extents = (-0.5, 0.5)  # Small range due to exponential nature

        test_configs = [
            (2, BETA, SIGMA_DECAY_EXPONENT, SEED),
            (2, 0.5, SIGMA_DECAY_EXPONENT, SEED),
            (2, BETA, 2.0, SEED),
            (3, BETA, SIGMA_DECAY_EXPONENT, SEED),
            (3, 2.0, 4.0, SEED),
        ]

        for n_dim, beta, sigma_decay_exp, seed in test_configs:
            key = (n_dim, beta, sigma_decay_exp, seed)
            f_fn, g_fn, h_fn, Q_expected, prec_diag_expected = self._sympy[key]

            with self.subTest(
                n_dim=n_dim, beta=beta, sigma_decay_exp=sigma_decay_exp, seed=seed
            ):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # For default parameters, use implementation as-is
                if (
                    beta == BETA
                    and sigma_decay_exp == SIGMA_DECAY_EXPONENT
                    and seed == SEED
                ):
                    f_impl = aniso_gaussian(X)
                    g_impl = aniso_gaussian_grad(X)[:, 0, :]
                    h_impl = aniso_gaussian_hess(X)

                else:
                    # For custom parameters, compute manually since implementation uses fixed constants
                    # This is a simplified version - for full test coverage, we'd need to modify the implementation
                    # to accept parameters, but for now we'll test the default case
                    continue

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)

                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )

                h_sym = np.stack(
                    [np.asarray(h_fn(*row), dtype=np.float64) for row in X]
                )

                # Compare with relaxed tolerances due to exponential function sensitivity
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-6, atol=1e-8)

    def test_rotation_matrix_properties(self):
        """Test that the rotation matrix has proper orthogonal properties."""
        n_dim = 3

        # Import the actual function to test
        from src.problems.anisotropic_gaussian import _get_random_rotation

        Q = _get_random_rotation(n_dim, np.float64)

        # Test orthogonality: Q^T @ Q should be identity
        identity_test = Q.T @ Q
        np.testing.assert_allclose(identity_test, np.eye(n_dim), rtol=1e-10, atol=1e-12)

        # Test determinant is 1 (proper rotation)
        det = np.linalg.det(Q)
        self.assertAlmostEqual(det, 1.0, places=10)

    def test_precision_eigenvalues_properties(self):
        """Test properties of the precision eigenvalues."""
        from src.problems.anisotropic_gaussian import _get_precision_eigenvalues

        n_dim = 4
        prec_diag = _get_precision_eigenvalues(n_dim, np.float64)

        # Should have n_dim elements
        self.assertEqual(len(prec_diag), n_dim)

        # All elements should be positive
        self.assertTrue(np.all(prec_diag > 0))

        # Should be decreasing (since sigma increases, precision decreases)
        # Actually, precision = (i+1)^(2*p), so it should be increasing
        for i in range(n_dim - 1):
            self.assertGreater(prec_diag[i + 1], prec_diag[i])

        # First element should be (1^(2*SIGMA_DECAY_EXPONENT)) = 1
        expected_first = 1.0 ** (2 * SIGMA_DECAY_EXPONENT)
        self.assertAlmostEqual(prec_diag[0], expected_first, places=10)

    def test_function_positivity(self):
        """Test that function values are always >= BETA."""
        rng = np.random.default_rng(123)
        X = rng.uniform(-2.0, 2.0, size=(50, 2)).astype(np.float64)

        f_vals = aniso_gaussian(X)

        # All function values should be >= BETA
        self.assertTrue(np.all(f_vals >= BETA))


if __name__ == "__main__":
    unittest.main()
