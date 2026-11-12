"""
Tree tracking across ranks, launched under MPI by tests/mpi/test_mpi.py.

Each rank grows the trees of its share of the initial clusters (no work
stealing); rank 0 merges the forests. Checks that the merged forest is consistent
with the leaves (``tests._helpers.tree_problems``) on every rank, and reports the
node counts.
"""

import json

import numpy as np

from tests.mpi._mpi_worker import RESULT_PREFIX


def main() -> None:
    import src
    from src.backend import MPI
    from src.surrogate import AFINNISurrogate
    from tests._helpers import train_surrogate, tree_problems

    xp = src.xp
    world = MPI.COMM_WORLD
    rank = world.Get_rank()

    def f(x):
        return (xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]) + x[:, 2] ** 2)[:, None]

    rng = np.random.default_rng(4)
    X = xp.asarray(rng.uniform(-1, 1, (6_000, 3))) if rank == 0 else xp.empty((0, 3))
    s = AFINNISurrogate()
    train_surrogate(
        X, s, f_fn=f, atol=0.02, rtol=0.02, num_initial_clusters=7,
        min_pts_per_cluster=40, use_work_stealing=False, track_trees=True,
    )  # fmt: skip
    problems = world.gather(tree_problems(s), root=0)
    if rank == 0:
        result = {
            "n_ranks": world.Get_size(),
            "problems": problems,
            "n_leaves": int(s.N_src),
            "n_internal": int(s.tree_forest.n_internal),
            "n_leaf_nodes": int(s.tree_forest.n_leaves),
            "converged": bool(s.converged),
        }
        print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
