"""
Every method the function suite runs, by name (``METHODS``: label, colour, and ``(f, lo, hi, atol, seed) ->
Model``): the baselines AFINNI and ART (with affine and with quadratic leaves) through ``AdaptiveSurrogate``,
and the schemes of ``src.schemes``.

Under ``mpiexec`` the baselines train and predict collectively on every rank, with work stealing. A scheme
trains on rank 0 and spreads each round's oracle calls over every rank (``src.schemes.oracle.distributed``);
the other ranks only serve evaluations until rank 0 is done. A trained model lives on rank 0, where
prediction is serial.
"""

from dataclasses import dataclass

import numpy as np

from src import IS_CUPY_BACKEND, to_host, xp
from src.config import ARTSettings, TrainingConfig
from src.core.evaluation import Evaluator
from src.infra.backend import MPI
from src.schemes import chebtree, chebtt, hub, kernel, sparsegrid
from src.schemes.chebtt import adaptive as chebtt_adaptive
from src.schemes.oracle import distributed as spread
from src.surrogate import AdaptiveSurrogate


@dataclass
class Model:
    predict: object  # (N, d) host points -> (N,) host values
    leaf_of: object | None  # (N, d) -> (N,) leaf ids, or None
    n_leaves: int
    evals: int
    refuses: object | None = None  # (N, d) -> (N,) bool: queries the model declines, answered by the oracle


def _ranks() -> int:
    return MPI.COMM_WORLD.Get_size()


def setup() -> str:
    """
    The device and rank count, e.g. ``cpu-1rank``. Runs are cached and plotted per setup and never
    compared across setups: times depend on both, and ART's split decisions differ between CPU and
    GPU, whose sums differ at rounding level.
    """
    return SETUP or f"{'gpu' if IS_CUPY_BACKEND else 'cpu'}-{_ranks()}rank"


SETUP = None  # ``--setup``: read and plot another setup's cache (``--overview`` only; never trains into it)


def _collective_predict(m):
    """``predict`` for a collective AdaptiveSurrogate: every rank calls it, rank 0 gets the values (None elsewhere)."""

    def predict(Q):
        out = m.predict(xp.asarray(Q))
        return None if out is None or out[0] is None else np.ravel(to_host(out[0]))

    return predict


def _art_config(lo, hi, atol, seed, local_model=None):
    extra = {"local_model": local_model} if local_model is not None else {}
    return TrainingConfig(algorithm="art", atol=atol, rtol=None, domain=(lo, hi), num_initial_clusters=1,
                          use_work_stealing=_ranks() > 1, seed=seed, max_levels=20,
                          art=ARTSettings(sample_multiplier=20, split_search="soft"), **extra)  # fmt: skip


def afinni(f, lo, hi, atol, seed) -> Model:
    d = len(lo)
    X = np.random.default_rng(seed).uniform(lo, hi, (int(min(5e4 * 1.3**d, 1e6)) if d > 2 else 600, d))
    cfg = TrainingConfig(atol=atol, rtol=None, num_initial_clusters=4, min_pts_per_cluster=int(min(100 * 1.3**d, 400)) if d > 2 else 24,
                         max_levels=20, seed=seed, use_work_stealing=_ranks() > 1)  # fmt: skip
    m = AdaptiveSurrogate()
    m.train(xp.asarray(X), cfg, Evaluator(f_fn=f))

    def leaf_of(Q):
        out = m.predict_local(xp.asarray(Q), k_nn=1, coarse_factor=10, strict_trust_region=False, return_cluster_assignments=True)
        return np.asarray(to_host(out[2])).reshape(len(Q), -1)[:, 0]

    return Model(_collective_predict(m), leaf_of, m.N_src, m.profiling_stats["total_f_evals"])


def _art(f, lo, hi, atol, seed, local_model=None) -> Model:
    X0 = np.random.default_rng(seed).uniform(lo, hi, (60, len(lo)))
    m = AdaptiveSurrogate()
    m.train(xp.asarray(X0), _art_config(lo, hi, atol, seed, local_model), Evaluator(f_fn=f))
    return Model(_collective_predict(m), lambda Q: np.asarray(to_host(m.route_leaves(xp.asarray(Q)))),
                 m.N_src, m.profiling_stats["total_f_evals"])  # fmt: skip


METHODS = {
    "afinni": ("AFINNI", "#6d3b0f", afinni),
    "art": ("ART, affine leaves", "#1f5fa8", lambda f, lo, hi, atol, seed: _art(f, lo, hi, atol, seed)),
    "art_quad": ("ART, quadratic leaves", "#2e7d32", lambda f, lo, hi, atol, seed: _art(f, lo, hi, atol, seed, "quadratic_lsq")),
}
BASELINES = ("afinni", "art", "art_quad")


def register(name: str, label: str, colour: str, fn) -> None:
    METHODS[name] = (label, colour, fn)


def _host(f):
    """f on host arrays: (N,) values, or (N, m) for a vector problem."""

    def g(X):
        V = np.asarray(to_host(f(xp.asarray(X))))
        return V.reshape(len(X), -1) if V.ndim == 2 and V.shape[1] > 1 else V.ravel()

    return g


def distributed(train):
    """Run ``train(oracle, lo, hi, atol, seed) -> Model`` on rank 0 with the oracle spread over every rank (``src.schemes.oracle``)."""

    def method(f, lo, hi, atol, seed) -> Model:
        with spread(_host(f)) as oracle:
            if oracle is None:  # a serving rank
                return Model(lambda Q: None, None, 0, 0)
            return train(oracle, lo, hi, atol, seed)

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


def _hub(f, lo, hi, atol, seed) -> Model:
    """The hub expansion (its pilot, design and hold-out from ``seed``); its pieces are its output groups."""
    m = hub.build(f, lo, hi, atol, seed=seed)
    return Model(m.predict, None, len(m.groups), m.evals)


register("chebtree", "ChebTree: Chebyshev boxes, coefficient decay", "#c0392b", distributed(_chebtree))
register("sparsecheb", "SparseCheb: adaptive sparse Chebyshev grid", "#6a1b9a", distributed(_sparsecheb))
register("chebtt", "ChebTT: Chebyshev tensor train, cross", "#e08a00", distributed(_chebtt))
register("chebtt_metric", "ChebTT-M: ChebTT in the pullback-metric coordinates", "#1b7f5b", distributed(_chebtt_metric))
register("kernel", "MetricKernel: learned-metric greedy kernel", "#2e7d9a", distributed(_kernel))
register("chebtt_adaptive", "ChebTT-A: ChebTT with piecewise axes and interface terms", "#b5179e", distributed(_chebtt_adaptive))
register("hub", "Hub: dense Chebyshev hub in the steep directions, weak terms in the rest", "#5d6d00", distributed(_hub))
