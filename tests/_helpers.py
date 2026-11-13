"""Helpers shared by tests: backend-aware comparisons (NumPy or CuPy) and a training shorthand."""

import numpy as np

from src import IS_CUPY_BACKEND, _to_host, xp
from src.config import ClusteringConfig
from src.core.evaluation import Evaluator
from src.surrogate import AFINNISurrogate

# CuPy reductions/einsum may order floating-point operations differently from
# NumPy (and from one code path to another), so results agree to rounding only.
GPU_RTOL = 1e-12
# Two GPU trainings that may take different code paths (e.g. batch sizes in AFINNI,
# whose augmentation seed depends on batches): rounding differences pass through the
# least-squares solves and reach a few 1e-12 on small values. Repeated GPU runs of one
# configuration are bit-identical (no atomic float sums on the training path).
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


EVALUATOR_FIELDS = ("f_fn", "g_fn", "oracle_fn", "hessian_fn")


def train_surrogate(X, surrogate=None, *, Y_train=None, initial_labels=None, **options):
    """
    Train ``surrogate`` (a new ``AFINNISurrogate`` by default) and return it. The
    ``Evaluator`` fields among ``options`` build the evaluator and the rest the
    ``ClusteringConfig``, so tests can keep one dict of options per case.
    """
    evaluator = Evaluator(**{k: options.pop(k) for k in EVALUATOR_FIELDS if k in options})
    surrogate = AFINNISurrogate() if surrogate is None else surrogate
    surrogate.train(xp.asarray(X), ClusteringConfig(**options), evaluator, Y_train=Y_train, initial_labels=initial_labels)
    return surrogate
