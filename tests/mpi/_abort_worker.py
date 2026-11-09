"""
One rank fails in a chosen training phase (argv[1]); the whole job must end with a
non-zero status and the original diagnostic, instead of leaving peers blocked.
"""

import sys

import numpy as np


class _FailingSendComm:
    """COMM_WORLD proxy whose result send (rank 1) raises, to fail during collection."""

    def __init__(self, comm, tag):
        self._comm, self._tag = comm, tag

    def send(self, obj, dest, tag=0):
        if tag == self._tag:
            raise RuntimeError("injected failure on rank 1 (collection)")
        return self._comm.send(obj, dest=dest, tag=tag)

    def __getattr__(self, name):
        return getattr(self._comm, name)


def main(phase: str) -> None:
    import src
    import src.adaptive_clustering as ac
    import src.kmeans
    from src.backend import MPI
    from src.problems.rosenbrock import rosenbrock
    from tests._helpers import train_surrogate

    rank = MPI.COMM_WORLD.Get_rank()

    def f(x):
        if phase == "root_initial_oracle" and rank == 0:
            raise RuntimeError("injected failure on rank 0 (initial evaluation)")
        if phase == "worker_refinement" and rank == 1 and len(x) < 1000:
            raise RuntimeError("injected failure on rank 1 (refinement)")
        return rosenbrock(x)

    if phase == "root_initial_clustering" and rank == 0:
        def failing_kmeans(*args, **kwargs):
            raise RuntimeError("injected failure on rank 0 (initial clustering)")

        src.kmeans.kmeans_labels = failing_kmeans
    if phase == "root_public_result" and rank == 0:
        import src.surrogate as su

        engine = su._adaptive_clustering

        def engine_then_fail(*args, **kwargs):
            engine(*args, **kwargs)
            raise RuntimeError("injected failure on rank 0 (public result conversion)")

        su._adaptive_clustering = engine_then_fail
    if phase == "worker_index" and rank == 1:
        import src.surrogate as su

        def failing_index(_centroids):
            raise RuntimeError("injected failure on rank 1 (index construction)")

        su._make_knn_index = failing_index
    if phase == "worker_collection" and rank == 1:
        class _MPI:
            COMM_WORLD = _FailingSendComm(MPI.COMM_WORLD, ac.LOCAL_RESULT_TAG)

            def __getattr__(self, name):
                return getattr(MPI, name)

        proxy = _MPI()
        proxy.COMM_TYPE_SHARED = MPI.COMM_TYPE_SHARED
        ac.MPI = proxy

    X = src.xp.asarray(np.random.default_rng(0).uniform(-2, 2, (20_000, 2))) if rank == 0 else src.xp.empty((0, 2))
    train_surrogate(X, f_fn=f, atol=0.01, rtol=0.01, num_initial_clusters=16, min_pts_per_cluster=50)


if __name__ == "__main__":
    main(sys.argv[1])
