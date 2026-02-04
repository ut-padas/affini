import logging

from array_api_compat import array_namespace

from src import Array, xp
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
        max_norm/L : Estimated maximum 2-norm of the function over the domain. Proxy for Lipschitz constant.
    """
    xp = array_namespace(domain)
    X = DataSampler.sample_grid(domain, num_points_per_dim=num_points_per_dim)

    vals = func(X)  # (M, K)
    norms = xp.linalg.norm(vals, axis=1)  # (M,)

    max_norm = xp.max(norms)
    max_point = X[xp.argmax(norms)]

    logger.info(f"Estimated max 2-norm: {max_norm} at point {max_point}")
    return max_norm


def repeats(values: Array, repeats: Array) -> Array:
    """
    Repeat elements of an array.

    Args:
        values (Array): The values to repeat (N,)
        repeats (Array): How many times to repeat each value (N,)

    Returns:
        cp.ndarray: The expanded array

    Example::
        values = cp.array([10, 20, 30])
        repeats = cp.array([3, 0, 2])
        result = repeats(values, repeats)
        # result is cp.array([10, 10, 10, 30, 30])
    """
    # 1. Calculate the output positions (offsets)
    # If repeats is [3, 0, 2], offsets is [0, 3, 3, 5]
    offsets = xp.zeros(len(repeats) + 1, dtype=xp.int32)
    offsets[1:] = xp.cumsum(repeats)

    total_size = int(offsets[-1])

    # 2. Create the output array
    # We use -1 as a placeholder to identify start positions
    indices = xp.zeros(total_size, dtype=xp.int32)

    # 3. Mark the start of each segment
    # We use boolean masking to handle cases where repeats[i] == 0
    diff_mask = repeats > 0
    indices[offsets[:-1][diff_mask]] = 1

    # 4. Use cumsum to propagate the segment IDs
    # [1, 0, 0, 1, 0] -> [1, 1, 1, 2, 2]
    # Then subtract 1 to get back to 0-based indexing
    segment_ids = xp.cumsum(indices) - 1

    # 5. Map these indices back to the original values
    # Filter 'values' to only include those actually being repeated
    # to maintain alignment with the cumulative segment IDs
    valid_values = values[diff_mask]
    return valid_values[segment_ids]
