import unittest

import numpy as np
import sympy as sp

from src.problems.anisotropic_gaussian import (
    BETA,
    DIAGONAL_EXPONENT,
    SEED,
    random_aniso_gaussian,
    random_aniso_gaussian_grad,
    random_aniso_gaussian_hess,
)


def _sympy_aniso_gaussian_f_g_h(n_dim: int, seed: int = SEED):
    """Create symbolic function, gradient, and Hessian for anisotropic Gaussian function."""
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

    # Compute M = Q S Q^T
    M = Q_sym @ S_sym @ Q_sym.T

    # Quadratic form: x^T M x
    quad_form = x_vec.T @ M @ x_vec
    quad_form = quad_form[0]  # Extract scalar from 1x1 matrix

    # Function: f(x) = BETA + exp(-0.5 * x^T M x)
    f = BETA + sp.exp(-sp.Rational(1, 2) * quad_form)

    # Compute gradient: g(x) = -f(x) * M * x
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Compute Hessian: H(x) = f(x) * [ (Mx)(Mx)^T - M ]
    h = sp.Matrix([[sp.diff(f, xi, xj) for xj in x] for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")
    h_fn = sp.lambdify(x, h, "numpy")

    return f_fn, g_fn, h_fn


class TestAnisoGaussianSympyDerivatives(unittest.TestCase):
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
            cls._sympy[key] = _sympy_aniso_gaussian_f_g_h(n_dim, seed)

    def test_value_grad_hess_match_sympy(self):
        rng = np.random.default_rng(1)  # Use different seed for test data
        num_pts = 15  # Smaller number due to computational complexity
        prob_extents = (-0.3, 0.3)  # Small range to avoid numerical issues

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
                    f_impl = random_aniso_gaussian(X)
                    g_impl = random_aniso_gaussian_grad(X)[:, 0, :]
                    h_impl = random_aniso_gaussian_hess(X)
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
                    quad_term = np.sum(
                        scales[None, :] * (X_rot**2), axis=1, keepdims=True
                    )

                    # Function: f(x) = BETA + exp(-0.5 * quad_term)
                    exp_part = np.exp(-0.5 * quad_term)
                    f_impl = BETA + exp_part

                    # Gradient: g(x) = -exp(-0.5 * quad_term) * (Q S Q^T x)
                    # First compute M x = Q (S (Q^T x))
                    temp = scales[None, :] * X_rot  # X Q S
                    Mx = temp @ Q.T  # X Q S Q^T
                    g_impl = -exp_part * Mx

                    # Hessian: H(x) = exp(-0.5 * quad_term) * [ (Mx)(Mx)^T - M ]
                    # Construct M = Q S Q^T
                    S_mat = np.diag(scales)
                    M = Q @ S_mat @ Q.T  # (d, d)

                    # Compute outer product (Mx)(Mx)^T for each point
                    term1 = Mx[:, :, None] * Mx[:, None, :]  # (N, d, d)

                    # M broadcasted to (N, d, d)
                    term2 = M[None, :, :]  # (1, d, d) -> (N, d, d)

                    # H = exp_part * (term1 - term2), exp_part is (N, 1)
                    h_impl = exp_part[:, :, None] * (term1 - term2)

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)
                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )
                h_sym = np.stack(
                    [np.asarray(h_fn(*row), dtype=np.float64) for row in X]
                )

                # Compare function values
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-12)

                # Compare gradients
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)

                # Compare Hessians
                np.testing.assert_allclose(h_impl, h_sym, rtol=1e-10, atol=1e-10)

    def test_function_properties(self):
        """Test properties of the Gaussian function without normalization."""
        n_dims = [2, 3, 4]

        for n_dim in n_dims:
            with self.subTest(n_dim=n_dim):
                scales = np.array(
                    [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(n_dim)]
                )

                # Test that scales are positive
                self.assertTrue(np.all(scales > 0.0))

                # Test that scales are decreasing (anisotropy property)
                for i in range(1, n_dim):
                    self.assertGreater(scales[i - 1], scales[i])

    def test_function_positivity(self):
        """Test that the Gaussian function is always positive."""
        rng = np.random.default_rng(2)
        n_dims = [2, 3, 4]

        for n_dim in n_dims:
            with self.subTest(n_dim=n_dim):
                X = rng.uniform(-1.0, 1.0, size=(20, n_dim)).astype(np.float64)

                f_values = random_aniso_gaussian(X)

                # All function values should be positive
                self.assertTrue(np.all(f_values > 0))

                # Function should be bounded (exponential with negative argument)
                self.assertTrue(np.all(f_values < np.inf))

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
                if expected_second is not None:
                    np.testing.assert_allclose(scales[1], expected_second, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
