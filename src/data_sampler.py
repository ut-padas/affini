import logging
import time

import numpy as np
from array_api_compat import array_namespace
from scipy.stats import qmc

logger = logging.getLogger(__name__)


class DataSampler:
    """Helper to generate training data X in a backend-agnostic way."""

    @staticmethod
    def sample_uniform(domain, N: int, seed: int = 0):
        """
        Sample N points uniformly from domain.
        Backend matches domain's backend.
        """
        logger.debug("Sampling %d points uniformly from domain.", N)
        t0 = time.perf_counter()

        xp = array_namespace(domain)
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
    def sample_grid(domain, num_points_per_dim: int):
        """
        Sample points on a grid from domain.
        Backend matches domain's backend.
        """
        logger.debug(
            "Sampling grid with %d points per dimension from domain.",
            num_points_per_dim,
        )
        t0 = time.perf_counter()

        xp = array_namespace(domain)

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
    def sample_latin_hypercube(domain, N: int, seed: int = 0):
        """
        Sample N points using Latin Hypercube Sampling.
        Uses Scipy (CPU) internally, then moves to domain's backend.
        """
        logger.debug("Sampling %d points using Latin Hypercube from domain.", N)
        t0 = time.perf_counter()

        xp = array_namespace(domain)
        if xp.__name__ == "array_api_compat.cupy":
            domain = xp.asnumpy(domain)

        # Convert domain bounds to NumPy for scipy
        if domain.ndim == 1:
            n_dim = 1
            l_np = np.asarray([float(domain[0])])
            r_np = np.asarray([float(domain[1])])
        else:
            l_np = np.asarray(domain[:, 0])
            r_np = np.asarray(domain[:, 1])
            n_dim = len(domain)

        # Generate LHS
        sampler = qmc.LatinHypercube(d=n_dim, rng=seed)
        sample = sampler.random(n=N)

        # Scale to domain bounds
        sample_scaled = qmc.scale(sample, l_np, r_np)

        # Convert to target backend
        X = xp.asarray(sample_scaled, dtype=xp.float64)
        logger.info(f"LHS Sampling Time: {time.perf_counter()-t0:.4f}s")
        return X
