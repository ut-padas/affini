"""
The ChebTT model: a Chebyshev interpolant in tensor-train form (``ChebTT``; one per output, in shared
coordinates, in ``VectorTT``), its contractions, and its compression after acceptance.

Each core holds Chebyshev coefficients along its axis, shape (r_k, n_k + 1, r_k+1), or on an axis with
breakpoints one block of n_k + 1 rows per piece. A query costs ``sum_k r_k (n_k + 1) r_k+1`` multiply-adds:
linear in d. The model with one core changed is one core's work from the train's environments
(``environments``, ``through``), which the checks and the compression use instead of whole contractions.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.schemes.chebyshev import chebyshev_bases


@dataclass
class ChebTT:
    lo: np.ndarray
    hi: np.ndarray
    cores: list  # per axis, (r_k, n_k + 1, r_k+1) Chebyshev coefficients along the middle axis
    evals: int = 0
    ranks: tuple = ()
    degrees: tuple = ()
    breaks: (
        tuple
    ) = ()  # per axis, its interior breakpoints (empty: one piece); () for none on any axis
    bands: (
        tuple
    ) = ()  # per axis, each breakpoint's band: the model is not claimed within it

    def pieces(self, k: int) -> int:
        return len(self.breaks[k]) + 1 if self.breaks else 1

    def predict(self, X: np.ndarray, chunk: int = 16_384) -> np.ndarray:
        """In chunks that bound the temporaries (``contract``), over the active axes only (``active``)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        axes, model, const = self.active
        if model is None:
            return np.full(len(X), const)
        return np.concatenate(
            [
                model.contract(model.bases(X[s : s + chunk][:, axes]))
                for s in range(0, len(X), chunk)
            ]
        )

    @cached_property
    def active(self) -> tuple:
        """
        ``(axes, model, constant)``: the axes whose cores depend on their coordinate, and the same model
        as a train over those axes alone. A core of degree 0 (one piece) is a constant matrix C_0, so it
        is multiplied into its left neighbour (or, before the first active axis, into the next one) and
        dropped: exact, and a query then pays only for its active axes. A function of few of the inputs
        (each output of a gradient of a sum of local terms, a species coupled to few others) keeps a
        handful of its d cores. With no active axis, ``model`` is None and ``constant`` the model's value.
        """
        axes, cores, pending = [], [], None
        for k, C in enumerate(self.cores):
            if C.shape[1] == 1 and self.pieces(k) == 1:
                M = C[:, 0, :]
                if cores:
                    cores[-1] = np.einsum("ajb,bc->ajc", cores[-1], M)
                else:
                    pending = M if pending is None else pending @ M
                continue
            if pending is not None:
                C, pending = np.einsum("ab,bjc->ajc", pending, C), None
            axes.append(k)
            cores.append(C)
        if not cores:
            return [], None, float(pending[0, 0])
        if len(axes) == len(self.cores):
            return axes, self, 0.0
        breaks = tuple(self.breaks[k] for k in axes) if self.breaks else ()
        bands = tuple(self.bands[k] for k in axes) if self.bands else ()
        return (
            axes,
            ChebTT(self.lo[axes], self.hi[axes], cores, breaks=breaks, bands=bands),
            0.0,
        )

    def bases(self, X: np.ndarray) -> list:
        """
        Per axis, the basis values at the rows of X (``contract``'s input): the Chebyshev values of the
        axis, or on an axis with breakpoints those of each query's own piece in its block of rows, zero
        in the other pieces' (one product per core still serves every query).
        """
        sizes = [C.shape[1] for C in self.cores]
        if not any(len(b) for b in self.breaks):
            return chebyshev_bases(self.local(X), sizes)
        return [self.axis_values(X, k, n1) for k, n1 in enumerate(sizes)]

    def axis_values(self, X: np.ndarray, k: int, n1: int) -> np.ndarray:
        """Axis k's entry of ``bases`` (n1, Q), for a core of ``n1`` rows (``n1 / pieces`` per piece on an axis with breakpoints)."""
        if not self.breaks or not len(self.breaks[k]):
            return chebyshev_bases(
                2.0 * (X[:, k : k + 1] - self.lo[k]) / (self.hi[k] - self.lo[k]) - 1.0,
                [n1],
            )[0]
        a, b = piece_bounds(
            self.lo[k],
            self.hi[k],
            self.breaks[k],
            self.bands[k] if self.bands else None,
        )
        m1 = n1 // len(a)
        p = np.clip(
            np.searchsorted(self.breaks[k], X[:, k], side="right"), 0, len(a) - 1
        )
        T = np.zeros((n1, len(X)))
        local = chebyshev_bases(
            (2.0 * (X[:, k] - a[p]) / (b[p] - a[p]) - 1.0)[:, None], [m1]
        )[0]
        T[p[None, :] * m1 + np.arange(m1)[:, None], np.arange(len(X))[None, :]] = local
        return T

    def local(self, X: np.ndarray) -> np.ndarray:
        """X in the cores' coordinates, [-1, 1] per axis."""
        return 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0

    def contract(self, T: list) -> np.ndarray:
        """
        The model at the points whose Chebyshev values are ``T`` (``chebyshev_bases``; at least ``n_k + 1``
        rows per axis, so models of different degrees can share one set). Per core, the values times the
        core first, one matrix product (Q, n + 1) x (n + 1, r_k r_k+1), then the running row vector
        times that: about ``sum_k r_k (n_k + 1) r_k+1`` multiply-adds per point, 1.6-4x faster than
        multiplying the row vector into the core first, which leaves a strided (Q, n + 1, r) temporary.
        """
        v = np.ones((T[0].shape[1], 1))
        for C, Tk in zip(self.cores, T, strict=True):
            v = advance(v, C, Tk)
        return v[:, 0]

    def environments(self, T: list) -> tuple[list, list]:
        """
        Per core k, the train's product left of it, (Q, r_k), and right of it, (Q, r_k+1), at the points
        of ``T``: the model with only core k changed to C' is ``through(left[k], C', T[k], right[k])``, one
        core's work instead of the whole train's. ``left[d]`` holds the model itself (as ``contract``'s
        bits: the same steps), as a (Q, 1) column.
        """
        left = [np.ones((T[0].shape[1], 1))]
        for C, Tk in zip(self.cores, T, strict=True):
            left.append(advance(left[-1], C, Tk))
        right = [np.ones((T[0].shape[1], 1))]
        for C, Tk in zip(self.cores[:0:-1], T[:0:-1], strict=True):
            right.append(np.einsum("qab,qb->qa", core_values(C, Tk), right[-1]))
        return left, right[::-1]


def core_values(C: np.ndarray, Tk: np.ndarray) -> np.ndarray:
    """(Q, r_k, r_k+1): core C at the points whose Chebyshev values on its axis are ``Tk`` (``contract``'s per-core matrix)."""
    ra, n1, rb = C.shape
    return (Tk[:n1].T @ C.transpose(1, 0, 2).reshape(n1, ra * rb)).reshape(-1, ra, rb)


def advance(v: np.ndarray, C: np.ndarray, Tk: np.ndarray) -> np.ndarray:
    """The row vectors v (Q, r_k) carried through core C: one step of ``contract``."""
    if (
        C.shape[0] == 1
    ):  # every first core, every core of a rank-1 train: a scaling, without einsum's overhead
        return v * (C[0].T @ Tk[: C.shape[1]]).T
    return np.einsum("qa,qab->qb", v, core_values(C, Tk))


def through(
    left: np.ndarray, C: np.ndarray, Tk: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """The train at Q points with core k replaced by C, from its environments (``ChebTT.environments``)."""
    return np.einsum("qb,qb->q", advance(left, C, Tk), right)


INSET = 1e-10  # a piece's end nodes sit this share of the axis inside an interior breakpoint, so a jump is never sampled on the wrong side


def piece_bounds(
    lo: float, hi: float, breaks, bands=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    The intervals [a_p, b_p] that carry each piece's Chebyshev nodes: [lo, hi] cut at ``breaks``, inset at
    each interior breakpoint by ``INSET`` of the width or twice its band (``axis_edges``), whichever is
    larger, so that no node falls on the wrong side of an edge that is located only to within its band.
    """
    e = np.concatenate([[lo], np.asarray(breaks, dtype=np.float64), [hi]])
    w = (hi - lo) * INSET
    inset = (
        np.full(len(e) - 2, w)
        if bands is None
        else np.maximum(w, 2.0 * np.asarray(bands, dtype=np.float64))
    )
    if len(
        inset
    ):  # never more than a quarter of either neighbouring piece, so no piece can invert
        inset = np.minimum(inset, 0.25 * np.minimum(np.diff(e)[:-1], np.diff(e)[1:]))
    a, b = e[:-1].copy(), e[1:].copy()
    a[1:] += inset
    b[:-1] -= inset
    return a, b


@dataclass
class VectorTT:
    """
    Every output its own scalar ChebTT, in its group's coordinates ``u = (x - centre) Q`` (``Q`` None:
    the box's own axes). Predictions are (N, outputs).
    """

    groups: list  # (output indices, Q or None, their ChebTTs)
    centre: np.ndarray
    outputs: int
    evals: int = 0

    def predict(self, X: np.ndarray, chunk: int = 16_384) -> np.ndarray:
        """A group's outputs share its coordinates and their Chebyshev values (``ChebTT.contract``)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty((len(X), self.outputs))
        for cols, Q, models in self.groups:
            shared = not any(
                m.breaks for m in models
            )  # pieces differ between outputs: each its own bases
            sizes = [
                max(m.cores[k].shape[1] for m in models)
                for k in range(len(models[0].cores))
            ]
            for s in range(0, len(X), chunk):
                U = (
                    X[s : s + chunk]
                    if Q is None
                    else (X[s : s + chunk] - self.centre) @ Q
                )
                T = chebyshev_bases(models[0].local(U), sizes) if shared else None
                for c, model in zip(cols, models, strict=True):
                    axes, small, const = model.active
                    if small is None:
                        out[s : s + chunk, c] = const
                    elif (
                        shared
                    ):  # the shared values of its active axes (at least n_k + 1 rows each)
                        out[s : s + chunk, c] = small.contract([T[k] for k in axes])
                    else:
                        out[s : s + chunk, c] = small.contract(small.bases(U[:, axes]))
        return out


def keep_degrees(C: np.ndarray, pieces: int, keep: int) -> np.ndarray:
    """The core with only the first ``keep`` coefficients of every piece."""
    if pieces == 1:
        return C[:, :keep]
    return C.reshape(C.shape[0], pieces, C.shape[1] // pieces, C.shape[2])[
        :, :, :keep
    ].reshape(C.shape[0], pieces * keep, C.shape[2])


def round_tt(cores: list, eps: float) -> list:
    """
    TT-SVD rounding (Oseledets 2011): right-orthogonalise the cores, then truncate every bond, left to
    right, to the smallest rank whose discarded singular values stay below ``eps * |TT| / sqrt(d - 1)``
    (a relative Frobenius error of at most eps overall, on the coefficient tensor).
    """
    cores = [C.copy() for C in cores]
    d = len(cores)
    for k in range(d - 1, 0, -1):
        ra, n1, rb = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(ra, n1 * rb).T)
        cores[k] = Q.T.reshape(-1, n1, rb)
        cores[k - 1] = np.einsum("ajb,cb->ajc", cores[k - 1], R)
    delta = eps * np.linalg.norm(cores[0]) / np.sqrt(max(d - 1, 1))
    for k in range(d - 1):
        ra, n1, rb = cores[k].shape
        U, sv, Vt = np.linalg.svd(cores[k].reshape(ra * n1, rb), full_matrices=False)
        tail = np.sqrt(np.cumsum(sv[::-1] ** 2))[
            ::-1
        ]  # tail[i]: the norm of the singular values from i on
        r = max(1, int(np.sum(tail > delta)))
        cores[k] = U[:, :r].reshape(ra, n1, r)
        cores[k + 1] = np.einsum("ab,bjc->ajc", sv[:r, None] * Vt[:r], cores[k + 1])
    return cores


def compress(model: ChebTT, probe: np.ndarray, budget: float) -> ChebTT:
    """
    The accepted model with its ranks rounded (the largest eps, by bisection) and then its trailing
    degrees chopped axis by axis, each step kept only while the predictions at ``probe`` stay within
    ``budget`` of the accepted model's. Acceptance leaves the model at most half the tolerance from f at
    its check points, usually far less, so compressing within a quarter of it cuts the cores 3-10x and
    inference with them (10-D Lotka-Volterra: 2,700 coefficients to 293, 8.7 to 0.35 us per query).
    The probe is the check set plus uniform points: model-only predictions, no evaluation of f.
    """
    T = model.bases(probe)  # once: every trial keeps or cuts degrees
    ref = model.contract(T)

    def close(values):
        return float(np.max(np.abs(values - ref))) <= budget

    # Rank rounding: the bisection on eps meets few distinct rank profiles, and a profile fixes the
    # rounded cores (the orthogonalisation does not depend on eps), so each profile is judged once.
    cores, lo_e, hi_e, judged = model.cores, 0.0, 1.0, {}
    for _ in range(16):
        mid = 0.5 * (lo_e + hi_e)
        trial = round_tt(model.cores, mid)
        ranks = tuple(C.shape[2] for C in trial[:-1])
        if ranks not in judged:
            judged[ranks] = close(ChebTT(model.lo, model.hi, trial).contract(T))
        if judged[ranks]:
            cores, lo_e = trial, mid
        else:
            hi_e = mid
    # Degree cuts, left to right, repeated while any axis shrinks. Cutting axis k's top degrees changes the
    # model by sum_j T_j(t_k) (L_k C_k[:, j, :] R_k+1) over the degrees cut, so one product of core k gives
    # every degree's contribution at the probe, and how many top degrees (in every piece at once) would
    # stay within the budget one after the other. An axis of degree <= 32 loses one degree per visit,
    # judged as that single cut; above it, a visit takes half of what fits: a high-degree axis needs
    # O(log n) passes instead of one per degree, and since no axis takes its whole share at once, the
    # axes still share the budget about as single cuts do. The right products are those of the pass's starting cores (no core right of k has changed
    # yet in this pass); the left one is carried through the cores as they are cut. An axis with
    # breakpoints is judged with its own piece layout (``axis_values`` for the core as it is now).
    layouts: dict = {}

    def values_for(k: int, n1: int) -> np.ndarray:
        """Axis k's Chebyshev values at the probe for a core of n1 rows: T's own rows, or on an axis with pieces its layout for n1 (cached)."""
        if model.pieces(k) == 1:
            return T[k]
        if (k, n1) not in layouts:
            layouts[(k, n1)] = model.axis_values(probe, k, n1)
        return layouts[(k, n1)]

    shrunk = True
    while shrunk:
        shrunk = False
        _, right = ChebTT(model.lo, model.hi, cores).environments(T)
        cores, v = list(cores), np.ones((len(ref), 1))
        for k in range(len(cores)):
            C, P = cores[k], model.pieces(k)
            m1 = C.shape[1] // P
            if (
                m1 > 33
            ):  # every degree's contribution at once: how many top degrees fit, one after the other
                Tk = values_for(k, C.shape[1])
                ra, n1, rb = C.shape
                w = np.einsum(
                    "qjb,qb->qj",
                    (v @ C.reshape(ra, n1 * rb)).reshape(-1, n1, rb),
                    right[k],
                )  # (Q, n1)
                part = Tk[:n1].T * w
                dev = part.sum(axis=1) - ref
                fits = 0
                while fits < m1 - 1:
                    dev = dev - part[
                        :, [p_ * m1 + m1 - 1 - fits for p_ in range(P)]
                    ].sum(axis=1)
                    if float(np.max(np.abs(dev))) > budget:
                        break
                    fits += 1
                if fits:
                    cores[k], shrunk = keep_degrees(C, P, m1 - (fits + 1) // 2), True
            elif m1 > 1:  # one degree, judged as a single cut
                trial = keep_degrees(C, P, m1 - 1)
                if close(through(v, trial, values_for(k, trial.shape[1]), right[k])):
                    cores[k], shrunk = trial, True
            v = advance(v, cores[k], values_for(k, cores[k].shape[1]))
    return ChebTT(model.lo, model.hi, cores, model.evals, tuple(C.shape[2] for C in cores[:-1]),
                  tuple(C.shape[1] // model.pieces(k) - 1 for k, C in enumerate(cores)), model.breaks, model.bands)  # fmt: skip
