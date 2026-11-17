"""
Lorenz-96 flow map x(0) -> x(tf) (the batched ensemble integrator, both backends) and its Jacobian
(SciPy sensitivities, parallelized over processes through shared memory).
"""

import os
from functools import partial

import numpy as np

from src import Array, to_device, to_host
from src.numerics import integrate_ensemble
from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.sensitivities_solver import (
    parallel_sensitivities_solver,
)

# The flow map's tolerances: about 1e-6 relative accuracy, far below every study tolerance.
LORENZ96_RTOL = LORENZ96_ATOL = 1e-6


def _limit_worker_threads() -> None:
    """Pin BLAS/OpenMP to one thread per worker, so N processes do not oversubscribe the cores."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _lorenz96_sensitivity_parallel_policy(
    X0_np: np.ndarray,
    *,
    min_augmented_variables: int = 20_000_000,
) -> bool:
    """
    Decide whether Lorenz96 sensitivity evaluation is large enough to justify
    multiprocessing.
    A sensitivity solve carries ``d + d*d`` variables per trajectory.  The
    policy is intentionally based on that augmented size rather than only on N,
    because high-dimensional gradients become expensive much faster than the
    state-only flowmap.
    """
    N, d = X0_np.shape
    return N * d * (d + 1) >= min_augmented_variables


def _lorenz96_num_workers(default: int = 64) -> int:
    """
    Worker count for Lorenz96 multiprocessing paths.
    """
    cpu_count = max(os.cpu_count() or 1, 2)
    return min(cpu_count - 1, default)


def lorenz96_flowmap(X0: Array, forcing: float = 1.0, tf: float = 0.1) -> Array:
    """
    (N, d) Lorenz-96 states at time ``tf`` from the rows of ``X0``, in float64 on the active backend:
    Dormand-Prince 5(4) per trajectory at rtol = atol = 1e-6 (``integrate_ensemble``), so a row's value
    does not depend on the other rows of the call.
    """
    return integrate_ensemble("lorenz96", X0, tf, [forcing], LORENZ96_RTOL, LORENZ96_ATOL)


def lorenz96_flowmap_grad(X0: Array, forcing: float = 1.0, tf: float = 0.1) -> Array:
    """
    Computes the gradients (Jacobians) of the Lorenz96 flowmap.

    Uses multiprocessing to solve sensitivities in parallel chunks.

    Args:
        X0: (N, d) initial states.
        forcing: Forcing term.
        tf: Final time.

    Returns:
        DXF: (N, d, d) Jacobian matrices d(x_tf)/d(x_0).
    """
    X0_np = to_host(X0)
    rhs = partial(lorenz96, forcing=forcing)

    # We only need the sensitivities (index 1), ignoring the flowmap state (index 0)
    _, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return to_device(DXF)


def _parallel_flowmap_grad_solve(
    X0_np: np.ndarray,
    forcing: float = 1.0,
    tf: float = 0.1,
    *,
    workers: int | None = None,
    mp_start_method: str | None = None,
    show_progress: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Array:
    rhs = partial(lorenz96, forcing=forcing)
    if workers is None:
        workers = _lorenz96_num_workers()

    _, DXF = parallel_sensitivities_solver(
        rhs,
        lorenz96_grad,
        X0_np,
        tf,
        MIN_PARALLEL_TRAJECTORIES=1,
        MIN_TRAJECTORIES_PER_WORKER=max(1, (X0_np.shape[0] + workers - 1) // workers),
        MAX_SAFE_WORKERS=workers,
        NUM_WORKERS=workers,
        MP_START_METHOD=mp_start_method,
        SHOW_PROGRESS=show_progress,
        RTOL=rtol,
        ATOL=atol,
    )
    return DXF


def lorenz96_flowmap_grad_auto(
    X0: Array,
    forcing: float = 1.0,
    tf: float = 0.1,
    *,
    workers: int | None = None,
    min_augmented_variables: int = 20_000_000,
    mp_start_method: str | None = None,
    show_progress: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Array:
    """
    Hybrid Lorenz96 flowmap Jacobian evaluation.

    Small batches stay in one serial batched sensitivity solve to avoid process
    overhead.  Large batches use the shared-memory multiprocessing solver.
    """
    X0_np = to_host(X0)
    if _lorenz96_sensitivity_parallel_policy(
        X0_np, min_augmented_variables=min_augmented_variables
    ):
        return to_device(
            _parallel_flowmap_grad_solve(
                X0_np,
                forcing,
                tf,
                workers=workers,
                mp_start_method=mp_start_method,
                show_progress=show_progress,
                rtol=rtol,
                atol=atol,
            )
        )

    rhs = partial(lorenz96, forcing=forcing)
    _, DXF = parallel_sensitivities_solver(
        rhs,
        lorenz96_grad,
        X0_np,
        tf,
        MIN_PARALLEL_TRAJECTORIES=X0_np.shape[0] + 1,
        SHOW_PROGRESS=show_progress,
        RTOL=rtol,
        ATOL=atol,
    )
    return to_device(DXF)
