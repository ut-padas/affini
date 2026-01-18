import logging
import time

from scipy.stats import qmc

from src import IS_CUPY_BACKEND, Array, xp

logger = logging.getLogger(__name__)


class DataSampler:
    """Helper to generate training data X in a backend-agnostic way."""

    @staticmethod
    def sample_uniform(domain, N: int | float, seed: int = 0) -> Array:
        """
        Sample N points uniformly from domain.
        Backend matches domain's backend.
        """
        N = int(N)
        logger.debug("Sampling %d points uniformly from domain.", N)
        t0 = time.perf_counter()

        rng = xp.random.default_rng(seed)

        # Parse domain
        if domain.ndim == 1:
            l, r = domain[0], domain[1]
            n_dim = 1
        else:
            l = domain[:, 0]
            r = domain[:, 1]
            n_dim = len(domain)

        # Generate (N, D)
        X = rng.uniform(l, r, (N, n_dim))
        logger.info(f"Uniform Sampling Time: {time.perf_counter()-t0:.4f}s")
        return xp.astype(X, xp.float64)

    @staticmethod
    def sample_grid(domain, num_points_per_dim: int | float) -> Array:
        """
        Sample points on a grid from domain.
        Backend matches domain's backend.
        """
        num_points_per_dim = int(num_points_per_dim)
        logger.debug(
            "Sampling grid with %d points per dimension from domain.",
            num_points_per_dim,
        )
        t0 = time.perf_counter()

        # Parse domain
        if domain.ndim == 1:
            l, r = domain[0], domain[1]
            n_dim = 1
            grids = [xp.linspace(l, r, num_points_per_dim)]
        else:
            l = domain[:, 0]
            r = domain[:, 1]
            n_dim = len(domain)
            grids = [xp.linspace(l[d], r[d], num_points_per_dim) for d in range(n_dim)]

        # Create meshgrid and reshape to (N, D)
        mesh = xp.meshgrid(*grids, indexing="ij")
        X = xp.stack(mesh, axis=-1).reshape(-1, n_dim)

        logger.info(f"Grid Sampling Time: {time.perf_counter()-t0:.4f}s")
        return xp.astype(X, xp.float64)

    @staticmethod
    def sample_latin_hypercube(domain, N: int | float, seed: int = 0) -> Array:
        """
        Sample N points using Latin Hypercube Sampling.
        Uses Scipy (CPU) internally, then moves to domain's backend.
        """
        N = int(N)
        logger.debug("Sampling %d points using Latin Hypercube from domain.", N)
        t0 = time.perf_counter()

        # Convert domain bounds to NumPy for scipy
        if domain.ndim == 1:
            n_dim = 1
            l_np = xp.asarray([float(domain[0])])
            r_np = xp.asarray([float(domain[1])])
        else:
            l_np = xp.asarray(domain[:, 0])
            r_np = xp.asarray(domain[:, 1])
            n_dim = len(domain)

        # Generate LHS
        sampler = qmc.LatinHypercube(d=n_dim, rng=seed)
        sample = sampler.random(n=N)

        # Scale to domain bounds
        if IS_CUPY_BACKEND:
            l_np = xp.asnumpy(l_np)
            r_np = xp.asnumpy(r_np)

        sample_scaled = qmc.scale(sample, l_np, r_np)

        # Convert to target backend
        X = xp.asarray(sample_scaled, dtype=xp.float64)
        logger.info(f"LHS Sampling Time: {time.perf_counter()-t0:.4f}s")
        return X


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging(console_level=logging.DEBUG)

    # Test samplers
    domain_1d = xp.array([0.0, 1.0])
    domain_2d = xp.array([[0.0, 1.0], [0.0, 1.0]])

    sampler = DataSampler()

    X_uniform_1d = sampler.sample_uniform(domain_1d, N=5, seed=42)
    print("Uniform 1D:\n", X_uniform_1d)

    X_grid_2d = sampler.sample_grid(domain_2d, num_points_per_dim=3)
    print("Grid 2D:\n", X_grid_2d)

    X_lhs_2d = sampler.sample_latin_hypercube(domain_2d, N=5, seed=42)
    print("LHS 2D:\n", X_lhs_2d)
