import multiprocessing as mp
import os

import numpy as np
from array_api_compat import array_namespace
from scipy.integrate import solve_ivp
from tqdm import tqdm


def _worker_solve(X0_chunk, rhs, DfDx, tf):
    """
    Worker function to solve a chunk of trajectories.
    Must be at the top level to be picklable by multiprocessing.
    """
    # If the chunk is empty (can happen if n_chunks > n_trajectories), return empty
    if X0_chunk.shape[0] == 0:
        return None, None

    xp = array_namespace(X0_chunk)
    n_trajectories, n_equations = X0_chunk.shape

    # Modify initial condition so that it contains sensitivity variables
    I_flat = xp.eye(n_equations).ravel()
    X0_sensitivities = xp.outer(xp.ones(n_trajectories), I_flat)
    X0_combined = xp.column_stack((X0_chunk, X0_sensitivities))
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
    sol = solve_ivp(coupled_rhs, t_span, X0_flat, method="RK45", t_eval=[tf])

    # Restructure results
    # sol.y shape is (n_flat_vars, 1) because t_eval has length 1
    final_flat = sol.y.T  # Transpose to get (1, n_flat_vars)

    XF_SF_batch = final_flat.reshape(n_trajectories, n_equations * (1 + n_equations))
    XF_batch = XF_SF_batch[:, :n_equations]
    SF_batch = XF_SF_batch[:, n_equations:]
    SF_batch_matrices = SF_batch.reshape(n_trajectories, n_equations, n_equations)

    return XF_batch, SF_batch_matrices


def _worker_solve_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing.imap"""
    return _worker_solve(*args)


def parallel_sensitivities_solver(rhs, DfDx, X0, tf):
    """
    Parallel version of sensitivities_solver using multiprocessing.

    Args:
        rhs: Picklable function for ODE RHS
        DfDx: Picklable function for Jacobian
        X0: Initial conditions (n_trajectories, n_equations)
        tf: Final time
    """
    xp = array_namespace(X0)

    # Determine number of workers
    total_cores = os.cpu_count() or 1
    num_workers = max(1, total_cores - 1)

    # Split X0 into chunks
    # Note: array_split handles cases where n_trajectories < num_workers gracefully
    chunks = np.array_split(X0, num_workers)

    # Prepare arguments for each worker
    # We must pass the functions rhs and DfDx. They must be picklable (defined at module level)
    tasks = [(chunk, rhs, DfDx, tf) for chunk in chunks if chunk.shape[0] > 0]
    mean_chunk_size = X0.shape[0] / len(tasks) if tasks else 0
    print(
        f"Total trajectories: {X0.shape[0]:,}, Workers: {len(tasks):,}, Mean chunk size: {mean_chunk_size:.2f}"
    )

    # Run in parallel
    with mp.Pool(processes=len(tasks)) as pool:
        results = list(
            tqdm(
                pool.imap(_worker_solve_wrapper, tasks),
                total=len(tasks),
                desc="Solving ODEs (Chunks)",
            )
        )

    # Unpack and combine results
    # results is a list of tuples: [(xf_1, sf_1), (xf_2, sf_2), ...]
    xf_list = [r[0] for r in results if r[0] is not None]
    sf_list = [r[1] for r in results if r[1] is not None]

    XF_combined = xp.concat(xf_list, axis=0)
    SF_combined = xp.concat(sf_list, axis=0)

    return XF_combined, SF_combined
