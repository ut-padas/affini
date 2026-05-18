import logging
import multiprocessing as mp
import os
import sys
import traceback
from functools import partial
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Assumes these modules exist in your project structure
from src import Array, _to_device, _to_host
from src.problems.adjoint_solver import parallel_adjoint_solver
from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.sensitivities_solver import parallel_sensitivities_solver

logger = logging.getLogger(__name__)


def _serial_flowmap_solve(X0: np.ndarray, rhs, tf: float) -> np.ndarray:
    """
    Integrate a batch of trajectories together in one SciPy solve.
    """
    if X0.shape[0] == 0:
        return np.empty_like(X0)

    n_trajectories, d = X0.shape

    def batched_ode_rhs(t, x_flat):
        x_batch = x_flat.reshape(n_trajectories, d)
        return rhs(x_batch).ravel()

    sol = solve_ivp(
        batched_ode_rhs,
        [0, tf],
        X0.ravel(),
        method="RK45",
        t_eval=[tf],
        rtol=1e-6,
    )
    return sol.y.T.reshape(n_trajectories, d)


def _solve_flowmap_and_write_chunk(
    x0_shm_name,
    xf_shm_name,
    x0_shape,
    xf_shape,
    dtype,
    start_idx,
    end_idx,
    rhs,
    tf,
):
    """
    Worker function that solves one shared-memory chunk of flowmap states.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        shm_x0 = SharedMemory(name=x0_shm_name)
        shm_xf = SharedMemory(name=xf_shm_name)

        X0_all = np.ndarray(x0_shape, dtype=dtype, buffer=shm_x0.buf)
        XF_all = np.ndarray(xf_shape, dtype=dtype, buffer=shm_xf.buf)

        X0_chunk = X0_all[start_idx:end_idx]
        if X0_chunk.shape[0] > 0:
            XF_all[start_idx:end_idx] = _serial_flowmap_solve(X0_chunk, rhs, tf)

        shm_x0.close()
        shm_xf.close()
        return True
    except Exception as exc:
        try:
            sys.stderr.write(f"Flowmap worker error: {exc}\n")
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        return False


def _flowmap_worker_wrapper_shm(args):
    return _solve_flowmap_and_write_chunk(*args)


def parallel_flowmap_solver(
    rhs,
    X0: np.ndarray,
    tf: float,
    MIN_PARALLEL_TRAJECTORIES: int = 20_000,
    MIN_TRAJECTORIES_PER_WORKER: int = 8_000,
    MAX_SAFE_WORKERS: int = 60,
) -> np.ndarray:
    """
    Parallel state-only flowmap solver using shared memory.

    This mirrors the sensitivity solver's chunking policy, but only stores the
    final state rather than an ``(N, d, d)`` sensitivity tensor.
    """
    if hasattr(X0, "device"):
        X0 = np.array(X0)

    N, d = X0.shape
    dtype = X0.dtype

    if N < MIN_PARALLEL_TRAJECTORIES:
        logger.info(
            f"Small flowmap batch ({N} < {MIN_PARALLEL_TRAJECTORIES}): running serially."
        )
        return _serial_flowmap_solve(X0, rhs, tf)

    shm_x0 = SharedMemory(create=True, size=X0.nbytes)
    shm_xf = SharedMemory(create=True, size=N * d * X0.itemsize)

    input_array = np.ndarray(X0.shape, dtype=dtype, buffer=shm_x0.buf)
    input_array[:] = X0[:]

    x0_shape = X0.shape
    xf_shape = (N, d)

    try:
        total_cores = os.cpu_count() or 1
        max_possible_workers = min(max(1, total_cores - 2), MAX_SAFE_WORKERS)
        needed_workers = (
            N + MIN_TRAJECTORIES_PER_WORKER - 1
        ) // MIN_TRAJECTORIES_PER_WORKER
        num_workers = max(1, min(max_possible_workers, needed_workers))

        chunk_size = (N + num_workers - 1) // num_workers
        indices = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
        tasks = [
            (
                shm_x0.name,
                shm_xf.name,
                x0_shape,
                xf_shape,
                dtype,
                start,
                end,
                rhs,
                tf,
            )
            for start, end in indices
            if start < end
        ]

        logger.info(
            f"Flowmap solver: {N:,} trajectories, {num_workers} workers "
            f"(N/Worker: {(N // num_workers):,})."
        )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_flowmap_worker_wrapper_shm, tasks),
                    total=len(tasks),
                    desc="Solving flowmaps",
                )
            )

        if not all(results):
            raise RuntimeError("Some flowmap workers failed. Check stderr logs.")

        return np.ndarray(xf_shape, dtype=dtype, buffer=shm_xf.buf).copy()
    finally:
        try:
            shm_x0.close()
            shm_x0.unlink()
        except Exception:
            pass
        try:
            shm_xf.close()
            shm_xf.unlink()
        except Exception:
            pass


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
    rhs = partial(lorenz96, forcing=forcing)
    return _to_device(_serial_flowmap_solve(X0_np, rhs, tf))


def lorenz96_flowmap_parallel(
    X0: Array, forcing: float = 1.0, tf: float = 0.1
) -> Array:
    """
    Computes the Lorenz96 flowmap with multiprocessing across trajectory chunks.

    Small batches fall back to the serial batched solve to avoid process-spawn
    overhead; large batches use the same shared-memory/chunking policy as the
    parallel sensitivity solver.
    """
    X0_np = _to_host(X0)
    rhs = partial(lorenz96, forcing=forcing)
    return _to_device(parallel_flowmap_solver(rhs, X0_np, tf))


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
    N, d = X0_np.shape

    # KEY FIX: Use partial instead of lambda.
    # Multiprocessing relies on pickle, which cannot serialize lambda functions.
    # partial creates a picklable callable with 'forcing' baked in.
    rhs = partial(lorenz96, forcing=forcing)

    # We only need the sensitivities (index 1), ignoring the flowmap state (index 0)
    _, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return _to_device(DXF)


def lorenz96_flowmap_oracle(X0: Array, forcing: float = 1.0, tf: float = 0.1) -> Array:
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
    N, d = X0_np.shape

    # KEY FIX: Use partial instead of lambda.
    # Multiprocessing relies on pickle, which cannot serialize lambda functions.
    # partial creates a picklable callable with 'forcing' baked in.
    rhs = partial(lorenz96, forcing=forcing)

    # We only need the sensitivities (index 1), ignoring the flowmap state (index 0)
    XF, DXF = parallel_sensitivities_solver(rhs, lorenz96_grad, X0_np, tf)

    return _to_device(XF), _to_device(DXF)


def lorenz96_flowmap_oracle_first_component(
    X0: Array, forcing: float = 1.0, tf: float = 0.1
) -> tuple[Array, Array]:
    """
    Computes both the flowmap and its gradients in parallel.

    Args:
        X0: (N, d) initial states.
        forcing: Forcing term.
        tf: Final time.

    Returns:
        Tuple (XF, DXF).
    """
    X0_np = _to_host(X0)
    N, d = X0_np.shape

    # Use partial for picklability
    rhs = partial(lorenz96, forcing=forcing)

    e1 = np.eye(d)[0]
    XF, DXF = parallel_adjoint_solver(rhs, lorenz96_grad, X0_np, tf, e1)

    return _to_device(XF), _to_device(DXF)
