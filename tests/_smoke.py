"""
Shared end-to-end workload: train + predict a small 2-D Rosenbrock surrogate.

Used by both the in-process backend test (tests/integration/test_backend.py) and the MPI
worker (tests/mpi/_mpi_worker.py), so every configuration - CPU, GPU, serial,
multi-rank - exercises exactly the same code path and can be compared.
"""

from __future__ import annotations

import numpy as np

SMOKE_TOL = 0.1


def run_smoke(
    n_train: int = 20_000, n_query: int = 5_000, tol: float = SMOKE_TOL, seed: int = 0
) -> dict | None:
    """Run the workload. Returns a result dict on rank 0, ``None`` on other ranks."""
    import src
    from src.problems.rosenbrock import rosenbrock
    from src.surrogate import AFINNISurrogate
    from tests._helpers import train_surrogate

    xp = src.xp
    rank = src.get_mpi_rank()
    rng = np.random.default_rng(seed)

    # Data lives on rank 0 only; refine scatters it to other ranks.
    X = xp.asarray(rng.uniform(-2, 2, (n_train, 2))) if rank == 0 else xp.empty((0, 2))
    surrogate = AFINNISurrogate()
    train_surrogate(
        X, surrogate,
        f_fn=rosenbrock,
        atol=tol,
        rtol=tol,
        num_initial_clusters=16,
        min_pts_per_cluster=50,
    )

    Q = xp.asarray(rng.uniform(-2, 2, (n_query, 2))) if rank == 0 else None
    pred, _mask = surrogate.predict(Q, k_nn=1, coarse_factor=10)
    if rank != 0:
        return None

    pred_h = np.asarray(src._to_host(pred), dtype=np.float64)
    f_true = np.asarray(src._to_host(rosenbrock(Q)), dtype=np.float64).reshape(pred_h.shape)
    max_rel_err = float(np.abs(pred_h - f_true).max() / np.abs(f_true).max())

    return {
        "n_clusters": int(surrogate.N_src),
        "max_rel_err": max_rel_err,
        "n_ranks": src.get_mpi_size(),
        "has_mpi": src.has_mpi(),
        "is_cupy_backend": src.IS_CUPY_BACKEND,
    }
