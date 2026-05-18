import multiprocessing as mp
import os
from functools import partial
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.integrate import solve_ivp

from src import Array, _to_device, _to_host
from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.sensitivities_solver import (
    _preferred_mp_start_method,
    parallel_sensitivities_solver,
)


def _serial_flowmap_solve(
    X0_np: np.ndarray,
    forcing: float,
    tf: float,
    *,
    rtol: float = 1e-6,
    atol: float | None = None,
) -> np.ndarray:
    """
    One batched SciPy solve over all trajectories.
    """
    n_trajectories, d = X0_np.shape

    def batched_ode_rhs(t, x_flat):
        x_batch = x_flat.reshape(n_trajectories, d)
        dxdt_batch = lorenz96(x_batch, forcing=forcing)
        return dxdt_batch.ravel()

    solve_kwargs = {
        "method": "RK45",
        "t_eval": [tf],
        "rtol": rtol,
    }
    if atol is not None:
        solve_kwargs["atol"] = atol

    sol = solve_ivp(
        batched_ode_rhs,
        [0, tf],
        X0_np.ravel(),
        **solve_kwargs,
    )
    if not sol.success:
        raise RuntimeError(f"Lorenz96 flowmap solve_ivp failed: {sol.message}")
    return sol.y.T.reshape(n_trajectories, d)


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _flowmap_chunk_worker(args):
    (
        x0_shm_name,
        xf_shm_name,
        shape,
        dtype,
        start,
        end,
        forcing,
        tf,
        rtol,
        atol,
    ) = args
    _limit_worker_threads()

    shm_x0 = SharedMemory(name=x0_shm_name)
    shm_xf = SharedMemory(name=xf_shm_name)
    try:
        X0_all = np.ndarray(shape, dtype=dtype, buffer=shm_x0.buf)
        XF_all = np.ndarray(shape, dtype=dtype, buffer=shm_xf.buf)
        XF_all[start:end] = _serial_flowmap_solve(
            X0_all[start:end],
            forcing,
            tf,
            rtol=rtol,
            atol=atol,
        )
        return True
    finally:
        shm_x0.close()
        shm_xf.close()


def _parallel_flowmap_solve(
    X0_np: np.ndarray,
    forcing: float,
    tf: float,
    *,
    workers: int | None = None,
    mp_start_method: str | None = None,
    rtol: float = 1e-6,
    atol: float | None = None,
) -> np.ndarray:
    """
    Chunk the state-only Lorenz96 flowmap across worker processes.
    Workers write into shared memory to avoid returning large arrays through
    multiprocessing pipes.
    """
    if workers is None:
        workers = _lorenz96_num_workers()

    shm_x0 = SharedMemory(create=True, size=X0_np.nbytes)
    shm_xf = SharedMemory(create=True, size=X0_np.nbytes)
    try:
        X0_shared = np.ndarray(X0_np.shape, dtype=X0_np.dtype, buffer=shm_x0.buf)
        X0_shared[:] = X0_np

        n_trajectories = X0_np.shape[0]
        chunk_size = (n_trajectories + workers - 1) // workers
        tasks = []
        for start in range(0, n_trajectories, chunk_size):
            end = min(start + chunk_size, n_trajectories)
            tasks.append(
                (
                    shm_x0.name,
                    shm_xf.name,
                    X0_np.shape,
                    X0_np.dtype,
                    start,
                    end,
                    forcing,
                    tf,
                    rtol,
                    atol,
                )
            )

        ctx = mp.get_context(_preferred_mp_start_method(mp_start_method))
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_flowmap_chunk_worker, tasks)
        if not all(results):
            raise RuntimeError("Some Lorenz96 flowmap workers failed.")

        return np.ndarray(X0_np.shape, dtype=X0_np.dtype, buffer=shm_xf.buf).copy()
    finally:
        for shm in (shm_x0, shm_xf):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


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


def _lorenz96_flowmap_parallel_policy(
    X0_np: np.ndarray,
    *,
    min_state_variables: int = 20_000_000,
) -> bool:
    """
    Decide whether the state-only flowmap is large enough to chunk.

    Your standalone benchmark showed multiprocessing starts paying off around
    tens of millions of state variables, with 4 workers generally better than
    using more workers blindly.
    """
    N, d = X0_np.shape
    return N * d >= min_state_variables


def _lorenz96_num_workers(default: int = 64) -> int:
    """
    Worker count for Lorenz96 multiprocessing paths.
    """
    cpu_count = max(os.cpu_count() or 1, 2)
    return min(cpu_count - 1, default)


def lorenz96_flowmap(X0: Array, forcing: float = 1.0, tf: float = 0.1) -> Array:
    """
    Computes the flowmap of the Lorenz96 system.

    This function runs in a single process using scipy.integrate.solve_ivp.
    It relies on vectorization for batch efficiency. For massive batches,
    consider a parallel approach similar to the gradient functions.

    Args:
        X0: (N, d) data matrix of initial states.
        forcing: Forcing term 'F'.
        tf: Final integration time.

    Returns:
        XF: (N, d) data matrix of states at time tf.
    """
    X0_np = _to_host(X0)
    return _to_device(_serial_flowmap_solve(X0_np, forcing, tf))


def lorenz96_flowmap_auto(
    X0: Array,
    forcing: float = 1.0,
    tf: float = 0.1,
    *,
    workers: int | None = None,
    min_state_variables: int = 20_000_000,
    mp_start_method: str | None = None,
    rtol: float = 1e-6,
    atol: float | None = None,
) -> Array:
    """
    Hybrid state-only flowmap.

    Uses the serial batched SciPy solve for small calls and shared-memory
    multiprocessing for large calls.
    """
    X0_np = _to_host(X0)
    if _lorenz96_flowmap_parallel_policy(
        X0_np, min_state_variables=min_state_variables
    ):
        return _to_device(
            _parallel_flowmap_solve(
                X0_np,
                forcing,
                tf,
                workers=workers,
                mp_start_method=mp_start_method,
                rtol=rtol,
                atol=atol,
            )
        )
    return _to_device(_serial_flowmap_solve(X0_np, forcing, tf, rtol=rtol, atol=atol))


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
    X0_np = _to_host(X0)
    rhs = partial(lorenz96, forcing=forcing)

    # We only need the sensitivities (index 1), ignoring the flowmap state (index 0)
    _, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return _to_device(DXF)


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
    X0_np = _to_host(X0)
    if _lorenz96_sensitivity_parallel_policy(
        X0_np, min_augmented_variables=min_augmented_variables
    ):
        return _to_device(
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
    return _to_device(DXF)
