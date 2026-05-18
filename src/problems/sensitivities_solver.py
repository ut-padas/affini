import logging
import multiprocessing as mp
import os
import platform
import sys
import traceback
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _preferred_mp_start_method(requested: str | None = None) -> str:
    """
    Pick a multiprocessing start method for heavy numerical worker pools.

    ``fork`` is much cheaper on Linux because children inherit the already
    imported modules via copy-on-write.  macOS defaults to ``spawn`` because
    forking a process that may have active native-library threads can be unsafe;
    users can still opt into another method with ``HATANN_MP_START_METHOD`` or
    the explicit argument.
    """
    requested = requested or os.environ.get("HATANN_MP_START_METHOD")
    available = mp.get_all_start_methods()
    if requested:
        if requested not in available:
            raise ValueError(
                f"Requested multiprocessing start method {requested!r} is not "
                f"available. Available methods: {available}"
            )
        return requested

    system = platform.system()
    if system == "Linux" and "fork" in available:
        return "fork"
    if "spawn" in available:
        return "spawn"
    return available[0]


def _serial_solve(X0, rhs, DfDx, tf, rtol=1e-6, atol=1e-9):
    """
    Core solver logic acting on a batch of trajectories.
    Returns (XF, SF).
    """
    if X0.shape[0] == 0:
        return None, None

    # We assume numpy for the solver internals
    xp = np
    n_trajectories, n_equations = X0.shape

    # Modify initial condition so that it contains sensitivity variables
    I_flat = xp.eye(n_equations).ravel()
    X0_sensitivities = xp.outer(xp.ones(n_trajectories), I_flat)
    X0_combined = xp.column_stack((X0, X0_sensitivities))
    X0_flat = X0_combined.ravel()

    # Define the coupled ODE system (State + Sensitivities)
    def coupled_rhs(t, x_s_flat):
        # x refers to regular variables, s refers to sensitivity variables

        # 1. Reshape flat input into batch format
        # Shape: (n_trajectories, n_equations * (1 + n_equations))
        x_s_batch = x_s_flat.reshape(n_trajectories, n_equations * (1 + n_equations))

        x_batch = x_s_batch[:, :n_equations]
        s_batch = x_s_batch[:, n_equations:]

        # Reshape sensitivities into matrices for matrix multiplication
        # Shape: (n_trajectories, n_equations, n_equations)
        s_batch_matrices = s_batch.reshape(n_trajectories, n_equations, n_equations)

        # 2. Compute Physics RHS
        rhs_x_batch = rhs(x_batch)

        # 3. Compute Sensitivity RHS (Jacobian * Sensitivities)
        J = DfDx(x_batch)  # Shape: (n_trajectories, n_equations, n_equations)

        # Batched Matrix Multiplication: J @ S
        rhs_s_batch_tensor = xp.einsum("ijk,ikl->ijl", J, s_batch_matrices)

        # Flatten the sensitivity matrices back to vectors
        rhs_s_batch_matrix = rhs_s_batch_tensor.reshape(
            n_trajectories, n_equations * n_equations
        )

        # 4. Combine and flatten
        result = xp.column_stack((rhs_x_batch, rhs_s_batch_matrix))
        return result.ravel()

    # Solve ODEs
    t_span = [0, tf]
    # RK45 is generally robust, t_eval ensures we get output exactly at tf
    sol = solve_ivp(
        coupled_rhs,
        t_span,
        X0_flat,
        method="RK45",
        t_eval=[tf],
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Sensitivity solve_ivp failed: {sol.message}")

    # Restructure results
    # sol.y shape is (n_flat_vars, 1) because t_eval has length 1
    final_flat = sol.y.T  # Transpose to get (1, n_flat_vars)

    XF_SF_batch = final_flat.reshape(n_trajectories, n_equations * (1 + n_equations))
    XF_batch = XF_SF_batch[:, :n_equations]
    SF_batch = XF_SF_batch[:, n_equations:].reshape(
        n_trajectories, n_equations, n_equations
    )

    return XF_batch, SF_batch


def _solve_and_write_chunk(
    x0_shm_name,
    xf_shm_name,
    sf_shm_name,
    x0_shape,
    xf_shape,
    sf_shape,
    dtype,
    start_idx,
    end_idx,
    rhs,
    DfDx,
    tf,
    rtol,
    atol,
):
    """
    Worker function that reads from shared input, solves, and writes to shared output.
    """
    # 0. Restrict threads to avoid oversubscription bombing the CPU
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        # Attach to shared memory
        shm_x0 = SharedMemory(name=x0_shm_name)
        shm_xf = SharedMemory(name=xf_shm_name)
        shm_sf = SharedMemory(name=sf_shm_name)

        # Reconstruct numpy arrays backed by shared memory
        X0_all = np.ndarray(x0_shape, dtype=dtype, buffer=shm_x0.buf)
        XF_all = np.ndarray(xf_shape, dtype=dtype, buffer=shm_xf.buf)
        SF_all = np.ndarray(sf_shape, dtype=dtype, buffer=shm_sf.buf)

        # 1. Read Input Chunk
        X0_chunk = X0_all[start_idx:end_idx]

        # 2. Solve (Reuse existing logic inline for simplicity/pickling)
        if X0_chunk.shape[0] > 0:
            XF_batch, SF_batch_matrices = _serial_solve(
                X0_chunk, rhs, DfDx, tf, rtol=rtol, atol=atol
            )

            # 3. Write Output directly to Shared Memory
            XF_all[start_idx:end_idx] = XF_batch
            SF_all[start_idx:end_idx] = SF_batch_matrices

        # Clean up
        shm_x0.close()
        shm_xf.close()
        shm_sf.close()
        return True

    except Exception as e:
        # Print to stderr carefully
        try:
            sys.stderr.write(f"Worker Error: {e}\n")
            traceback.print_exc(file=sys.stderr)
        except:
            pass
        return False


def _worker_wrapper_shm(args):
    return _solve_and_write_chunk(*args)


def parallel_sensitivities_solver(
    rhs,
    DfDx,
    X0,
    tf,
    MIN_PARALLEL_TRAJECTORIES=20_000,
    MIN_TRAJECTORIES_PER_WORKER=8_000,
    MAX_SAFE_WORKERS=60,
    NUM_WORKERS=None,
    MP_START_METHOD=None,
    SHOW_PROGRESS=True,
    RTOL=1e-6,
    ATOL=1e-9,
):
    """
    Parallel solver using Shared Memory to avoid pipe overhead.

    Args:
        MIN_PARALLEL_TRAJECTORIES (int): If N is below this threshold, run serially on the main process to avoid spawn overhead.
        MIN_TRAJECTORIES_PER_WORKER (int): Minimum workload per core to justify spawning a new process. Prevents over-parallelization on medium datasets.
        MAX_SAFE_WORKERS (int): Absolute cap on number of worker processes to prevent resource exhaustion on very large machines. Even on 70+ core machines, spawning more than ~50 workers for heavy math tasks can lead to excessive overhead and instability.
    """
    # Ensure Inputs are NumPy (CPU) for SharedMemory compatibility
    if hasattr(X0, "device"):  # Handle CuPy/Torch/JAX arrays
        X0 = np.array(X0)

    N, d = X0.shape
    dtype = X0.dtype

    # Check for small dataset serial bypass
    if N < MIN_PARALLEL_TRAJECTORIES:
        logging.info(
            f"Small dataset ({N} < {MIN_PARALLEL_TRAJECTORIES}): running serially."
        )
        return _serial_solve(X0, rhs, DfDx, tf, rtol=RTOL, atol=ATOL)

    # 1. Allocate Shared Memory for Inputs and Outputs
    # Inputs
    shm_x0 = SharedMemory(create=True, size=X0.nbytes)
    # Outputs
    xf_nbytes = N * d * X0.itemsize
    sf_nbytes = N * d * d * X0.itemsize

    shm_xf = SharedMemory(create=True, size=xf_nbytes)
    shm_sf = SharedMemory(create=True, size=sf_nbytes)

    # 2. Populate Input Shared Memory
    input_array = np.ndarray(X0.shape, dtype=dtype, buffer=shm_x0.buf)
    input_array[:] = X0[:]

    # Define shapes for workers to reconstruct
    x0_shape = X0.shape
    xf_shape = (N, d)
    sf_shape = (N, d, d)

    try:
        # 3. Configure Workers
        total_cores = os.cpu_count() or 1
        max_possible_workers = min(max(1, total_cores - 2), MAX_SAFE_WORKERS)

        # Calculate optimal number of workers based on density
        # We only want to add a worker if it has at least 'min_trajectories_per_worker' to do.
        needed_workers = (
            N + MIN_TRAJECTORIES_PER_WORKER - 1
        ) // MIN_TRAJECTORIES_PER_WORKER

        # Dispatch only as many workers as needed, up to the hardware limit.
        # NUM_WORKERS gives callers a way to use benchmark-informed policies
        # without changing this generic solver's defaults.
        if NUM_WORKERS is None:
            num_workers = min(max_possible_workers, needed_workers)
        else:
            num_workers = min(max_possible_workers, needed_workers, NUM_WORKERS)
        num_workers = max(1, num_workers)

        # Split work into equal chunks based on worker count
        chunk_size = (N + num_workers - 1) // num_workers
        indices = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

        tasks = []
        for start, end in indices:
            if start >= end:
                continue
            tasks.append(
                (
                    shm_x0.name,
                    shm_xf.name,
                    shm_sf.name,
                    x0_shape,
                    xf_shape,
                    sf_shape,
                    dtype,
                    start,
                    end,
                    rhs,
                    DfDx,
                    tf,
                    RTOL,
                    ATOL,
                )
            )

        logging.info(
            f"Solver: {N:,} trajectories, {num_workers} workers (N/Worker: {(N // num_workers):,})."
        )

        start_method = _preferred_mp_start_method(MP_START_METHOD)
        logging.info(f"Sensitivity solver multiprocessing start method: {start_method}")
        ctx = mp.get_context(start_method)
        with ctx.Pool(processes=num_workers) as pool:
            iterator = pool.imap(_worker_wrapper_shm, tasks)
            if SHOW_PROGRESS:
                iterator = tqdm(iterator, total=len(tasks), desc="Solving ODEs")
            results = list(iterator)

        if not all(results):
            raise RuntimeError("Some workers failed. Check stderr logs.")

        # 4. Read Final Results from Shared Memory
        # Create final arrays (Copying out of shared memory to standard memory)
        # We copy because SharedMemory will be unlinked/destroyed.
        # This copy operation is fast (memory-to-memory memcpy).
        XF_final = np.ndarray(xf_shape, dtype=dtype, buffer=shm_xf.buf).copy()
        SF_final = np.ndarray(sf_shape, dtype=dtype, buffer=shm_sf.buf).copy()

        return XF_final, SF_final

    finally:
        # 5. Cleanup Shared Memory
        # Close and unlink locally.
        # The workers have already closed their handles in _solve_and_write_chunk.
        try:
            shm_x0.close()
            shm_x0.unlink()
        except:
            pass

        try:
            shm_xf.close()
            shm_xf.unlink()
        except:
            pass

        try:
            shm_sf.close()
            shm_sf.unlink()
        except:
            pass
