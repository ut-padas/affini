"""
New surrogate schemes, registered with ``src.structure.bench``.

ChebTree and SparseCheb gather each refinement round's new points into one oracle call. Under
``mpiexec`` that call is distributed: rank 0 runs the scheme and scatters each batch of points over
every rank (``array_split``), each rank evaluates its share, and rank 0 gathers the values, the same
division of oracle work the refinement engine uses. The other ranks only serve evaluations until
rank 0 is done. A trained model lives on rank 0; prediction there is serial.
"""

import numpy as np

from src import to_host, xp
from src.infra.backend import MPI
from src.structure import chebtree, sparsegrid
from src.structure.bench import Model, register

_EVALUATE, _STOP = "evaluate", "stop"


def _host(f):
    return lambda X: np.ravel(to_host(f(xp.asarray(X))))


class ScatterOracle:
    """Rank 0's oracle: every call scatters its points over the ranks and gathers their values in order."""

    def __init__(self, f, comm):
        self.f, self.comm = f, comm

    def __call__(self, X):
        self.comm.bcast(_EVALUATE, root=0)
        local = self.comm.scatter(np.array_split(np.asarray(X), self.comm.Get_size()), root=0)
        values = self.comm.gather(self.f(local) if len(local) else np.empty(0), root=0)
        return np.concatenate(values)

    def stop(self):
        self.comm.bcast(_STOP, root=0)


def serve(f, comm) -> None:
    """A non-root rank's loop: evaluate the share of every batch rank 0 scatters, until it stops."""
    while comm.bcast(None, root=0) == _EVALUATE:
        local = comm.scatter(None, root=0)
        comm.gather(f(local) if len(local) else np.empty(0), root=0)


def distributed(train):
    """Run ``train(oracle, lo, hi, atol, seed) -> Model`` on rank 0 with the oracle spread over every rank."""

    def method(f, lo, hi, atol, seed) -> Model:
        comm = MPI.COMM_WORLD
        f_host = _host(f)
        if comm.Get_size() == 1:
            return train(f_host, lo, hi, atol, seed)
        if comm.Get_rank() != 0:
            serve(f_host, comm)
            return Model(lambda Q: None, None, 0, 0)
        oracle = ScatterOracle(f_host, comm)
        try:
            return train(oracle, lo, hi, atol, seed)
        finally:
            oracle.stop()  # also on failure, so the serving ranks never wait forever

    return method


def _chebtree(f, lo, hi, atol, seed) -> Model:
    """ChebTree (deterministic: the seed is unused)."""
    t = chebtree.build(f, lo, hi, atol)
    return Model(t.predict, t.leaf_of, len(t.leaves), t.evals, t.refuses)


def _sparsecheb(f, lo, hi, atol, seed) -> Model:
    """SparseCheb, the dimension-adaptive sparse-grid Chebyshev interpolant (its check points from ``seed``)."""
    m = sparsegrid.build(f, lo, hi, atol, seed=seed)
    return Model(m.predict, None, len(m.coef), m.evals)


register("chebtree", "ChebTree: Chebyshev boxes, coefficient decay", "#c0392b", distributed(_chebtree))
register("sparsecheb", "SparseCheb: adaptive sparse Chebyshev grid", "#6a1b9a", distributed(_sparsecheb))
