import logging

from array_api_compat import array_namespace

from src.data_sampler import DataSampler

logger = logging.getLogger(__name__)


def get_matrix_2_norm(func, domain, num_points_per_dim=300):
    """
    Estimate the 2-norm of a matrix-valued function over a given domain.

    Args:
        func: Function that maps (N, DATA_DIM) -> (N, F_DIM)
        domain: Domain to sample points from. Shape (2,) or (DATA_DIM, 2)
        num_points_per_dim: Number of points per dimension for grid sampling.

    Returns:
        Estimated maximum 2-norm of the function over the domain.
    """
    xp = array_namespace(domain)
    X = DataSampler.sample_grid(domain, num_points_per_dim=num_points_per_dim)

    vals = func(X)  # (M, K)
    norms = xp.linalg.norm(vals, axis=1)  # (M,)

    max_norm = xp.max(norms)
    max_point = X[xp.argmax(norms)]

    logger.info(f"Estimated max 2-norm: {max_norm} at point {max_point}")
    return max_norm
