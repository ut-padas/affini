import unittest

import numpy as np
import sympy as sp

from src.problems.shifted_gaussians import (
    CENTERS_2D,
    CENTERS_3D,
    CENTERS_4D,
    shifted_gaussians,
    shifted_gaussians_grad,
)


def _sympy_shifted_gaussians_f_g(
    n_dim: int, centers: np.ndarray, shift: float = 1.0, std_dev: float = 0.5
):
    """Create symbolic function and gradient for shifted Gaussians."""
    x = sp.symbols(f"x0:{n_dim}")

    # Build the sum of Gaussians
    f = sp.Float(shift)
    for center in centers:
        sq_dist = sum((x[i] - sp.Float(center[i])) ** 2 for i in range(n_dim))
        f += sp.exp(-sq_dist / (2 * sp.Float(std_dev) ** 2))  # type: ignore

    # Compute gradient
    g = sp.Matrix([sp.diff(f, xi) for xi in x])

    # Create numerical functions
    f_fn = sp.lambdify(x, f, "numpy")
    g_fn = sp.lambdify(x, g, "numpy")

    return f_fn, g_fn


class TestShiftedGaussiansSympyDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._sympy = {}

        # Test configurations: (dimension, centers, shift, std_dev)
        configs = [
            (2, CENTERS_2D, 1.0, 0.5),
            (2, CENTERS_2D, 0.0, 0.5),
            (2, CENTERS_2D, 1.0, 1.0),
            (3, CENTERS_3D, 1.0, 0.5),
            (3, CENTERS_3D, 0.5, 0.3),
            (4, CENTERS_4D, 1.0, 0.5),
            (4, CENTERS_4D, 0.5, 0.3),
        ]

        for n_dim, centers, shift, std_dev in configs:
            # Convert to numpy if needed
            centers_np = np.array(centers) if hasattr(centers, "__array__") else centers
            key = (n_dim, tuple(map(tuple, centers_np)), shift, std_dev)
            cls._sympy[key] = _sympy_shifted_gaussians_f_g(
                n_dim, centers_np, shift, std_dev
            )

    def test_value_grad_match_sympy(self):
        rng = np.random.default_rng(42)
        num_pts = 50
        prob_extents = (-2.0, 2.0)

        test_configs = [
            (2, CENTERS_2D, 1.0, 0.5),
            (2, CENTERS_2D, 0.0, 0.5),
            (2, CENTERS_2D, 1.0, 1.0),
            (3, CENTERS_3D, 1.0, 0.5),
            (3, CENTERS_3D, 0.5, 0.3),
            (4, CENTERS_4D, 1.0, 0.5),
            (4, CENTERS_4D, 0.5, 0.3),
        ]

        for n_dim, centers, shift, std_dev in test_configs:
            centers_np = np.array(centers) if hasattr(centers, "__array__") else centers
            key = (n_dim, tuple(map(tuple, centers_np)), shift, std_dev)
            f_fn, g_fn = self._sympy[key]

            with self.subTest(n_dim=n_dim, shift=shift, std_dev=std_dev):
                X = rng.uniform(
                    low=prob_extents[0],
                    high=prob_extents[1],
                    size=(num_pts, n_dim),
                ).astype(np.float64)

                # Compute using implementation
                f_impl = shifted_gaussians(X, centers_np, shift=shift, std_dev=std_dev)
                g_impl = shifted_gaussians_grad(X, centers_np, std_dev=std_dev)[:, 0, :]

                # Compute using SymPy
                f_sym = np.array([float(f_fn(*row)) for row in X], dtype=np.float64)
                f_sym = f_sym.reshape(-1, 1)
                g_sym = np.stack(
                    [np.asarray(g_fn(*row), dtype=np.float64).reshape(-1) for row in X]
                )

                # Compare
                np.testing.assert_allclose(f_impl, f_sym, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(g_impl, g_sym, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
