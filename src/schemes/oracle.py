"""
The function a scheme trains on (``Oracle``), and the same function spread over MPI ranks
(``DistributedOracle``, ``serve``, ``distributed``).

An ``Oracle`` evaluates on host arrays: values (N,) for one output or (N, m) for several, and, when the
caller supplies one, the Jacobian (N, m, d), or (N, d) for one output (the library's layout of gradients).
It counts the points and the Jacobians evaluated and the seconds spent in them, which are a training
report's ``evals``, ``jacobians`` and ``oracle_seconds``. It does not deduplicate: a scheme that may ask
for a point twice keeps its own store. (It does not wrap ``core.evaluation.Evaluator``: that one evaluates
f with every Jacobian and returns device arrays, while the schemes are host code and some ask for
Jacobians alone.)

Under MPI the scheme trains on rank 0 and every batch it asks for is cut into one even chunk per rank,
evaluated there and gathered in order, so the values do not depend on the rank count. A chunk that fails
on any rank is re-raised on rank 0 after the gather, so no rank is left waiting in a collective.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

from src.infra.backend import MPI


class Oracle:
    """f, and optionally its Jacobian, on host float64 arrays, with evaluation counts and time."""

    def __init__(self, f: Callable, jacobian: Callable | None = None):
        self.f, self.jacobian_fn = f, jacobian
        self.evals = 0
        self.jacobians = 0
        self.seconds = 0.0

    @property
    def has_jacobian(self) -> bool:
        return self.jacobian_fn is not None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        start = time.perf_counter()
        try:
            V = self.evaluate_values(X)
        finally:
            self.seconds += time.perf_counter() - start
        self.evals += len(X)
        return V

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """(N, m, d), or (N, d) for one output."""
        if self.jacobian_fn is None:
            raise ValueError("this oracle has no Jacobian")
        X = np.asarray(X, dtype=np.float64)
        start = time.perf_counter()
        try:
            J = self.evaluate_jacobian(X)
        finally:
            self.seconds += time.perf_counter() - start
        self.jacobians += len(X)
        return J

    def evaluate_values(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.f(X), dtype=np.float64)

    def evaluate_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.jacobian_fn(X), dtype=np.float64)


def as_oracle(f: Callable | Oracle, jacobian: Callable | None = None) -> Oracle:
    """``f`` itself if it is an Oracle (a ``jacobian`` then must not be given separately), else a new Oracle."""
    if isinstance(f, Oracle):
        if jacobian is not None:
            raise ValueError("pass the Jacobian to the Oracle, not beside it")
        return f
    return Oracle(f, jacobian)


class PointCache:
    """
    f (host (N, d) -> (N, m)) behind one cache keyed by the point's exact coordinates: one call gives every
    output, so separate trains of a vector function's outputs pay once for any point one of them asks
    for. ``evals`` counts the distinct points f saw.
    """

    def __init__(self, f):
        self.f, self.cache = f, {}

    @property
    def evals(self) -> int:
        return len(self.cache)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        keys = [x.tobytes() for x in X]
        first: dict[bytes, int] = {}
        for i, key in enumerate(keys):
            if key not in self.cache and key not in first:
                first[key] = i
        if first:
            V = np.asarray(self.f(X[list(first.values())]), dtype=np.float64).reshape(
                len(first), -1
            )
            self.cache.update(zip(first, V, strict=True))
        return np.array([self.cache[key] for key in keys])


_VALUES, _JACOBIAN, _STOP = "values", "jacobian", "stop"


def _evaluate_chunk(fn: Callable, chunk: np.ndarray):
    """("ok", values) or ("error", message): a failure is carried to rank 0 instead of breaking the collective."""
    if not len(chunk):
        return ("ok", None)
    try:
        return ("ok", np.asarray(fn(chunk), dtype=np.float64))
    except Exception as e:  # noqa: BLE001 (any failure must reach rank 0)
        return ("error", f"{type(e).__name__}: {e}")


class DistributedOracle(Oracle):
    """Rank 0's oracle: every call scatters its points over the ranks (``serve`` on the others) and gathers the results in order."""

    def __init__(self, f: Callable, jacobian: Callable | None = None, comm=None):
        super().__init__(f, jacobian)
        self.comm = comm if comm is not None else MPI.COMM_WORLD

    def _scatter(self, kind: str, fn: Callable, X: np.ndarray) -> np.ndarray:
        comm = self.comm
        comm.bcast(kind, root=0)
        local = comm.scatter(np.array_split(X, comm.Get_size()), root=0)
        results = comm.gather(_evaluate_chunk(fn, local), root=0)
        errors = [f"rank {r}: {msg}" for r, (status, msg) in enumerate(results) if status == "error"]
        if errors:
            raise RuntimeError("oracle failed on " + "; ".join(errors))
        return np.concatenate([v for _, v in results if v is not None])

    def evaluate_values(self, X: np.ndarray) -> np.ndarray:
        return self._scatter(_VALUES, self.f, X)

    def evaluate_jacobian(self, X: np.ndarray) -> np.ndarray:
        return self._scatter(_JACOBIAN, self.jacobian_fn, X)

    def stop(self) -> None:
        """Release the serving ranks (call exactly once, also when training failed)."""
        self.comm.bcast(_STOP, root=0)


def serve(f: Callable, jacobian: Callable | None = None, comm=None) -> None:
    """A non-root rank's loop: evaluate its chunk of every batch rank 0 scatters, until rank 0 stops."""
    comm = comm if comm is not None else MPI.COMM_WORLD
    while (kind := comm.bcast(None, root=0)) != _STOP:
        local = comm.scatter(None, root=0)
        comm.gather(_evaluate_chunk(f if kind == _VALUES else jacobian, local), root=0)


@contextmanager
def distributed(f: Callable, jacobian: Callable | None = None, comm=None) -> Iterator[Oracle | None]:
    """
    On rank 0, an oracle spread over every rank (a plain ``Oracle`` on one rank); on the others, serve its
    batches until rank 0 leaves the block, then yield None. Every rank must enter the block.
    """
    comm = comm if comm is not None else MPI.COMM_WORLD
    if comm.Get_size() == 1:
        yield Oracle(f, jacobian)
        return
    if comm.Get_rank() != 0:
        serve(f, jacobian, comm)
        yield None
        return
    oracle = DistributedOracle(f, jacobian, comm)
    try:
        yield oracle
    finally:
        oracle.stop()  # also on failure, so the serving ranks never wait forever
