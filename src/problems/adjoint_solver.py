
import logging
import multiprocessing as mp
import os
import sys
import traceback
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def adjoint_solve(x0, f, DfDx, T, c):
    '''
    FOR A SINGLE COMPONENT:
    We have an ode dx/dt = f(x,p)$, with IC x_0 = x(0).
    x is a vector.
    Goal: compute the derivative of <c, x(T)> with respect to vector IC x_0.
    This is mostly just a toy code.
    '''

    # 1. Compute the solution to final time x(T)
    primal_solve = solve_ivp(
        fun = f,
        t_span = [0,T],
        y0 = x0,
        method='RK45',
        rtol=1e-6,
        dense_output=True
    )

    # 2. With the solution in hand, solve adjoint equation backwards
    def adjoint_rhs(t, a, forward_sol):
        x_t = forward_sol.sol(t)
        jac = DfDx(t, x_t)
        return -jac.T @ a

    adjoint_solve = solve_ivp(
        fun = lambda t, a: adjoint_rhs(t, a, primal_solve),
        t_span = [T, 0],
        y0 = c,
        method = 'RK45',
        rtol = 1e-6
    )

    return np.dot(c, primal_solve.y[:,-1]), adjoint_solve.y[:,-1]



def batched_adjoint_solve(X0, rhs, DfDx, tf, c):
    """
    A serial solver for a bunch of trajectories at once.

    Inputs:
        X0: (N_traj, d) array containing initial conditions
        rhs: The right hand side, taking in batched states and returning batched rhs.
        DfDx: The Jacobian, taking in batched states (N_traj, d) and returning batch Jacobian (N_traj, d, d)
        tf: Final time for the solve
        c: CONSTANT vector defining the quantity of interest. Shape: (d)

    Outputs:
        q: the quantity of interest (q = c dot x(tf))
        dq_dX0: the gradient
    """
    if X0.shape[0] == 0:
        return None, None

    # We assume numpy for the solver internals
    xp = np
    n_trajectories, n_equations = X0.shape

    # 1. Forward solve (save entire state)

    # Modified right hand side for forward solve
    # Takes in a flattened version, reshapes, computes, and reflattens
    # This will be used as rhs in scipy integrate solver
    def modified_forward_rhs(t, x_flat):
        x_batch = x_flat.reshape(n_trajectories, n_equations)
        rhs_x_batch = rhs(x_batch)
        return rhs_x_batch.ravel()

    # Solve the ODEs
    forward_solve = solve_ivp(
        fun = modified_forward_rhs,
        t_span = [0,tf],
        y0 = X0.ravel(),
        method='RK45',
        rtol=1e-6,
        dense_output=True
    )

    # 2. Backward solve

    # Modified adjoint rhs
    # Again, a (the adjoint variable) is flattened
    # So we need to apply a Jacobian to every block
    def modified_adjoint_rhs(t, a):
        x_t = forward_solve.sol(t) # flattened state
        x_t_batch = x_t.reshape(n_trajectories, n_equations)
        jac_batch = DfDx(x_t_batch) # batched Jacobian (n_traj, n_eq, n_eq)
        a_batch = a.reshape(n_trajectories, n_equations)
        # make sure to take transposes of Jacobians
        adjoint_rhs_batch = -1 * np.einsum('ikj,ik->ij', jac_batch, a_batch)
        return adjoint_rhs_batch.ravel()

    # Modify the right hand side
    batch_adjoint_ic = np.tile(c, n_trajectories)

    adjoint_solve = solve_ivp(
        fun = modified_adjoint_rhs,
        t_span = [tf, 0],
        y0 = batch_adjoint_ic,
        method = 'RK45',
        rtol = 1e-6
    )

    xf_batch = forward_solve.y[:,-1].reshape(n_trajectories, n_equations)
    q_batch = xf_batch @ c
    a_batch = adjoint_solve.y[:,-1].reshape(n_trajectories, n_equations)
    return q_batch, a_batch



"""
Everything after this is taken from Prajwal and slightly modified
to do adjoint solves scattered across multiple processes.
"""


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
    c,
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
            XF_batch, SF_batch_matrices = batched_adjoint_solve(X0_chunk, rhs, DfDx, tf, c)

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


def parallel_adjoint_solver(
    rhs,
    DfDx,
    X0,
    tf,
    c,
    MIN_PARALLEL_TRAJECTORIES=20_000,
    MIN_TRAJECTORIES_PER_WORKER=8_000,
    MAX_SAFE_WORKERS=60,
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
        return batched_adjoint_solve(X0, rhs, DfDx, tf, c)

    # 1. Allocate Shared Memory for Inputs and Outputs
    # Inputs
    shm_x0 = SharedMemory(create=True, size=X0.nbytes)
    # Outputs
    xf_nbytes = N * X0.itemsize
    sf_nbytes = N * d * X0.itemsize
    shm_xf = SharedMemory(create=True, size=xf_nbytes)
    shm_sf = SharedMemory(create=True, size=sf_nbytes)

    # 2. Populate Input Shared Memory
    input_array = np.ndarray(X0.shape, dtype=dtype, buffer=shm_x0.buf)
    input_array[:] = X0[:]

    # Define shapes for workers to reconstruct
    x0_shape = X0.shape
    xf_shape = (N)
    sf_shape = (N, d)

    try:
        # 3. Configure Workers
        total_cores = os.cpu_count() or 1
        max_possible_workers = min(max(1, total_cores - 2), MAX_SAFE_WORKERS)

        # Calculate optimal number of workers based on density
        # We only want to add a worker if it has at least 'min_trajectories_per_worker' to do.
        needed_workers = (
            N + MIN_TRAJECTORIES_PER_WORKER - 1
        ) // MIN_TRAJECTORIES_PER_WORKER

        # Dispatch only as many workers as needed, up to the hardware limit
        num_workers = min(max_possible_workers, needed_workers)
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
                    c,
                )
            )

        logging.info(
            f"Solver: {N:,} trajectories, {num_workers} workers (N/Worker: {(N // num_workers):,})."
        )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker_wrapper_shm, tasks),
                    total=len(tasks),
                    desc="Solving ODEs",
                )
            )

        if not all(results):
            raise RuntimeError("Some workers failed. Check stderr logs.")

        # 4. Read Final Results from Shared Memory
        # Create final arrays (Copying out of shared memory to standard memory)
        # We copy because SharedMemory will be unlinked/destroyed.
        # This copy operation is fast (memory-to-memory memcpy).
        XF_final = np.ndarray(xf_shape, dtype=dtype, buffer=shm_xf.buf).copy()
        SF_final = np.ndarray(sf_shape, dtype=dtype, buffer=shm_sf.buf).copy()

        # might need to reshape here
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

