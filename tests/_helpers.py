"""Helpers shared by tests that run on both the NumPy and the CuPy backend."""

import numpy as np

from src import IS_CUPY_BACKEND, _to_host

# CuPy reductions/einsum may order floating-point operations differently from
# NumPy (and from one code path to another), so results agree to rounding only.
GPU_RTOL = 1e-12
# Two separate GPU trainings: atomicAdd order differs per run and the differences
# pass through the least-squares solves, reaching a few 1e-12 on small values.
GPU_RUNS_RTOL = 1e-10


def host(a):
    """``a`` as a NumPy array, whatever the backend."""
    return np.asarray(_to_host(a))


def assert_backend_equal(
    actual, desired, rtol: float = GPU_RTOL, atol: float = 0.0, separate_runs: bool = False
):
    """
    Exact equality on the NumPy backend; equality to ``rtol`` (relative) and
    ``atol`` on CuPy, where operation order can differ. ``separate_runs`` compares
    results of two trainings, with ``GPU_RUNS_RTOL`` and an absolute floor at 1e-12
    of the largest value (for entries near zero).
    """
    actual, desired = host(actual), host(desired)
    if IS_CUPY_BACKEND and separate_runs:
        scale = float(np.max(np.abs(desired))) if desired.size else 0.0
        rtol, atol = max(rtol, GPU_RUNS_RTOL), max(atol, 1e-12 * scale)
    if IS_CUPY_BACKEND:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual, desired)
