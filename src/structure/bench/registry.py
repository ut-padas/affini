"""
New surrogate schemes, registered with ``src.structure.bench.functions``.

ChebTree and SparseCheb gather each refinement round's new points into one oracle call. Under
``mpiexec`` that call is distributed: rank 0 runs the scheme and scatters each batch of points over
every rank (``array_split``), each rank evaluates its share, and rank 0 gathers the values, the same
division of oracle work the refinement engine uses. The other ranks only serve evaluations until
rank 0 is done. A trained model lives on rank 0; prediction there is serial.
"""

import numpy as np

from src import to_host, xp
from src.infra.backend import MPI
from src.structure.bench.functions import Model, register
from src.structure.schemes import chebtree, chebtt, chebtt_adaptive, kernel, sparsegrid

_EVALUATE, _STOP = "evaluate", "stop"


def _host(f):
    """f on host arrays: (N,) values, or (N, m) for a vector problem."""

    def g(X):
        V = np.asarray(to_host(f(xp.asarray(X))))
        return V.reshape(len(X), -1) if V.ndim == 2 and V.shape[1] > 1 else V.ravel()

    return g


class ScatterOracle:
    """Rank 0's oracle: every call scatters its points over the ranks and gathers their values in order."""

    def __init__(self, f, comm):
        self.f, self.comm = f, comm

    def __call__(self, X):
        self.comm.bcast(_EVALUATE, root=0)
        local = self.comm.scatter(np.array_split(np.asarray(X), self.comm.Get_size()), root=0)
        values = self.comm.gather(self.f(local) if len(local) else None, root=0)
        return np.concatenate([v for v in values if v is not None])

    def stop(self):
        self.comm.bcast(_STOP, root=0)


def serve(f, comm) -> None:
    """A non-root rank's loop: evaluate the share of every batch rank 0 scatters, until it stops."""
    while comm.bcast(None, root=0) == _EVALUATE:
        local = comm.scatter(None, root=0)
        comm.gather(f(local) if len(local) else None, root=0)


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


def _chebtt(f, lo, hi, atol, seed) -> Model:
    """ChebTT, the Chebyshev tensor train by cross approximation (its random fibres and check points from ``seed``)."""
    probe = np.asarray(f(np.asarray(lo, dtype=np.float64)[None]))
    if probe.ndim == 2 and probe.shape[1] > 1:  # a vector problem: one train per output, one cache of f
        m = chebtt.build_vector(f, lo, hi, atol, probe.shape[1], seed=seed)
    else:
        m = chebtt.build(f, lo, hi, atol, seed=seed)
    return Model(m.predict, None, 1, m.evals + 1)


def _chebtt_metric(f, lo, hi, atol, seed) -> Model:
    """ChebTT in the coordinates of the outputs' pullback metric where that pays (``chebtt.build_vector`` with ``rotate``)."""
    probe = np.asarray(f(np.asarray(lo, dtype=np.float64)[None]))
    m_out = probe.shape[1] if probe.ndim == 2 else 1
    m = chebtt.build_vector(f, lo, hi, atol, m_out, rotate=True, seed=seed)
    rotated = sum(len(cols) for cols, Q, _ in m.groups if Q is not None)
    predict = m.predict if probe.ndim == 2 and m_out > 1 else (lambda X: m.predict(X)[:, 0])
    return Model(predict, None, 1 + rotated, m.evals + 1)


def _chebtt_adaptive(f, lo, hi, atol, seed) -> Model:
    """ChebTT-A: ChebTT with piecewise axes plus one interface term where f jumps across an oblique or curved interface or has a sharp front."""
    probe = np.asarray(f(np.asarray(lo, dtype=np.float64)[None]))
    if probe.ndim == 2 and probe.shape[1] > 1:
        m = chebtt_adaptive.build_vector(f, lo, hi, atol, probe.shape[1], seed=seed)
        pieces = 1 + sum(mm.interface is not None for mm in m.models)
    else:
        g = (lambda X: np.asarray(f(X)).reshape(len(X))) if probe.ndim == 2 else f
        m = chebtt_adaptive.build(g, lo, hi, atol, seed=seed)
        pieces = 1 + (m.interface is not None)
    return Model(m.predict, None, pieces, m.evals + 1)


def _kernel(f, lo, hi, atol, seed) -> Model:
    """MetricKernel, the learned-metric greedy kernel interpolant (its Sobol batches from ``seed``)."""
    m = kernel.build(f, lo, hi, atol, seed=seed, max_evals=50_000)
    return Model(m.predict, None, len(m.X), m.evals)


register("chebtree", "ChebTree: Chebyshev boxes, coefficient decay", "#c0392b", distributed(_chebtree))
register("sparsecheb", "SparseCheb: adaptive sparse Chebyshev grid", "#6a1b9a", distributed(_sparsecheb))
register("chebtt", "ChebTT: Chebyshev tensor train, cross", "#e08a00", distributed(_chebtt))
register("chebtt_metric", "ChebTT-M: ChebTT in the pullback-metric coordinates", "#1b7f5b", distributed(_chebtt_metric))
register("kernel", "MetricKernel: learned-metric greedy kernel", "#2e7d9a", distributed(_kernel))
register("chebtt_adaptive", "ChebTT-A: ChebTT with piecewise axes and interface terms", "#b5179e", distributed(_chebtt_adaptive))
