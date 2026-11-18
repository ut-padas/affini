"""
Adaptive ChebTT: one ChebTT with piecewise axes, plus one interface term where f jumps across an
oblique or curved interface or has a sharp smooth front.

    f ~ ChebTT(f - S) + S,   S = 0 for smooth and axis-aligned non-smooth functions

A sum of products of univariate functions can be non-smooth only along axis-aligned planes, which
``chebtt.build``'s piecewise axes handle. An oblique or curved jump has high rank at any degree, and a
tree refined towards it needs about (1/h)^(d-1) leaves, so it is modelled by a term aligned with the
interface instead (the interface a plane or a diagonal quadric: a level set of a low-parametric phi):

1. ChebTT with piecewise axes, which stops (``chebtt.ObliqueEdge``) as soon as its edge search meets a
   jump across an oblique interface, handing over the points it located on it. Smooth and axis-edge
   functions end here and pay nothing more.
2. Interface points: segments from the most extreme check point to random points, each scanned (17
   points) and bisected on the value at its largest jump (a segment crosses a region of volume share p
   with probability ~2p(1-p) in any dimension, an axis fibre with ~1/sqrt(d)); phi is the simplest of
   plane and diagonal quadric whose null-space fit has a clear singular-value gap and leaves the best
   90% of the points on it to 1e-7.
3. The jump measured at 2d + 8 interface points by one-sided fits along the normal (degree 6 on 8
   points per side, within min(0.2, the room in the box): no evaluation outside the box). A constant
   jump becomes a 1-D profile S = h(phi(x)), sampled along the in-box segment from the smallest to the
   largest phi; a varying one becomes jump fields S = [side] sum_k J_k(foot) rho^k / k! (rho the signed
   distance, foot the nearest interface point), the J_k polynomials of the foot whose degree grows (0,
   1, 2) until held-out interface points agree to 0.1 of the tolerance. They live on the side whose
   foot map is regular: the other side of a closed quadric holds its centre, where the foot is undefined.
4. A function with no jump that ChebTT cannot fit on its first budget, whose segments show steep but
   smooth crossings, is a front: phi is fitted to crossings of the mid-level (max + min) / 2 on segments
   from the extreme point, and S = h(phi(x)).
5. ChebTT with piecewise axes on f - S.

The claim is the tolerance outside the bands of the breakpoints and of the interface (a query within a
location error of an interface may be on the wrong side). Interfaces must be level sets of a plane or a
diagonal quadric, and only one is modelled; a feature no probe crosses is missed.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from src.structure import chebtt


def _features(Z: np.ndarray, kind: str) -> np.ndarray:
    one = np.ones((len(Z), 1))
    if kind == "plane":
        return np.concatenate([Z, one], axis=1)
    return np.concatenate([Z**2, Z, one], axis=1)  # a diagonal quadric


def _poly(X: np.ndarray, deg: int) -> np.ndarray:
    """Total-degree ``deg`` monomials of X (0, 1 or 2), the jump fields' basis."""
    one = np.ones((len(X), 1))
    if deg == 0:
        return one
    if deg == 1:
        return np.concatenate([one, X], axis=1)
    iu = np.triu_indices(X.shape[1])
    return np.concatenate([one, X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]], axis=1)


@dataclass
class Interface:
    """The level set phi = 0 (a plane or a diagonal quadric, in the box's [-1, 1] coordinates) and its term S."""

    kind: str
    w: np.ndarray  # null-space coefficients of ``_features``
    centre: np.ndarray
    scale: np.ndarray
    side: float = 1.0  # the jump fields live where rho * side > 0
    profile: chebtt.ChebTT | None = None  # S = h(phi): h on [u0, u1]
    u0: float = 0.0
    u1: float = 0.0
    degree: int = -1  # jump fields: their polynomial degree, and per jet order the coefficients
    fields: list = field(default_factory=list)

    def phi(self, X: np.ndarray) -> np.ndarray:
        return _features((X - self.centre) / self.scale, self.kind) @ self.w

    def grad(self, X: np.ndarray) -> np.ndarray:
        """Analytic gradient of phi (in the original coordinates)."""
        d = X.shape[1]
        if self.kind == "plane":
            return np.repeat((self.w[:d] / self.scale)[None], len(X), axis=0)
        Z = (X - self.centre) / self.scale
        return (2.0 * self.w[:d] * Z + self.w[d : 2 * d]) / self.scale

    def geometry(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance rho (positive where phi > 0) and the foot point: closed form for a plane, Newton projection for a quadric."""
        if self.kind == "plane":
            g = self.grad(X[:1])[0]
            nrm = float(np.linalg.norm(g))
            rho = self.phi(X) / nrm
            return rho, X - rho[:, None] * (g / nrm)[None]
        Y = X.copy()
        for _ in range(8):
            G = self.grad(Y)
            Y = Y - (self.phi(Y) / np.maximum(np.sum(G * G, axis=1), 1e-300))[:, None] * G
        G = self.grad(Y)
        n = G / np.maximum(np.linalg.norm(G, axis=1), 1e-300)[:, None]
        return np.sum((X - Y) * n, axis=1), Y

    def normal(self, X: np.ndarray) -> np.ndarray:
        G = self.grad(X)
        return G / np.maximum(np.linalg.norm(G, axis=1), 1e-300)[:, None]

    def term(self, X: np.ndarray) -> np.ndarray:
        if self.profile is not None:
            return self.profile.predict(np.clip(self.phi(X), self.u0, self.u1)[:, None])
        rho, foot = self.geometry(X)
        B = _poly(foot, self.degree)
        S = sum((B @ c) * rho**k / math.factorial(k) for k, c in enumerate(self.fields))
        return S * (rho * self.side > 0)


@dataclass
class AdaptiveTT:
    """ChebTT (with piecewise axes) of f - S plus the interface term S (none when ``interface`` is None)."""

    tt: chebtt.ChebTT
    interface: Interface | None
    kind: str
    evals: int = 0
    stages: dict = field(default_factory=dict)  # evaluations per stage of the build

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = self.tt.predict(X)
        return out + self.interface.term(X) if self.interface is not None else out


def segment_edges(F, anchor: np.ndarray, others: np.ndarray, m: int = 17) -> tuple[np.ndarray, int]:
    """
    Jumps on the segments from ``anchor`` to each of ``others``, located by value bisection and confirmed
    by ``chebtt.edge_at``; and the number of steep cells that are not jumps (a front's crossings).
    """
    pts, steep = [], 0
    t = np.linspace(0.0, 1.0, m)
    for P in others:
        dvec = P - anchor
        if float(np.linalg.norm(dvec)) < 1e-9:
            continue

        def g(s, dvec=dvec):
            return F(anchor[None] + np.asarray(s)[:, None] * dvec[None])

        v = g(t)
        dv = np.abs(np.diff(v))
        c = int(np.argmax(dv))
        if dv[c] <= 8.0 * np.median(dv) + 1e-14 * max(float(np.abs(v).max()), 1e-300):
            continue
        x0, delta, _ = chebtt.locate_edge(g, t[c], t[c + 1], True, 1e-12)
        if chebtt.edge_at(g, x0, max(delta, 1e-13), True):
            pts.append(anchor + x0 * dvec)
            continue
        # a steep kink is not a front: zoom and test for a kink before counting the cell as steep
        x1, d1, _ = chebtt.locate_edge(g, t[max(c - 1, 0)], t[min(c + 2, m - 1)], False, 1e-6)
        if not chebtt.edge_at(g, x1, max(d1, 1e-13), False):
            steep += 1
    return np.array(pts).reshape(-1, len(anchor)), steep


def fit_interface(P: np.ndarray, lo, hi, kinds=("plane", "diagonal")) -> Interface | None:
    """The simplest kind whose null-space fit has a clear singular-value gap (< 1e-3) and leaves the best 90% of the points on it to 1e-7."""
    centre, scale = 0.5 * (lo + hi), 0.5 * (hi - lo)
    Z = (P - centre) / scale
    for kind in kinds:
        A = _features(Z, kind)
        if len(A) < A.shape[1] + 3:
            continue
        keep = np.arange(len(A))
        for _ in range(2):  # consensus: refit on the best 90%
            w = np.linalg.svd(A[keep], full_matrices=False)[2][-1]
            keep = np.argsort(np.abs(A @ w))[: max(A.shape[1] + 3, int(0.9 * len(A)))]
        _, sv, Vt = np.linalg.svd(A[keep], full_matrices=False)
        w = Vt[-1]
        if float(np.max(np.abs(A[keep] @ w)) / np.linalg.norm(w)) < 1e-7 and sv[-1] < 1e-3 * sv[-2]:
            return Interface(kind, w, centre, scale)
    return None


def _room(x, u, lo, hi) -> float:
    """The largest t >= 0 with x + t u in the box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(u > 0, (hi - x) / u, np.where(u < 0, (lo - x) / u, np.inf))
    return float(np.min(tt))


_NS, _DEG = 8, 6
_TS = 0.5 * (1.0 + np.cos(np.pi * (np.arange(_NS) + 0.5) / _NS))  # in (0, 1)


def one_sided_jumps(F, x, n, lo, hi, kmax: int = 2, L0: float = 0.2) -> np.ndarray | None:
    """Jumps of f, f', f'' along n at x (the positive side minus the negative), from degree-6 fits on 8 points per side; None with too little room."""
    Lp, Lm = min(L0, _room(x, n, lo, hi)), min(L0, _room(x, -n, lo, hi))
    if min(Lp, Lm) < 1e-3:
        return None
    ders = []
    for sgn, L in ((1.0, Lp), (-1.0, Lm)):
        t = sgn * L * _TS
        c = np.polynomial.polynomial.polyfit(t / L, F(x[None] + t[:, None] * n[None]), _DEG)
        ders.append([math.factorial(k) * c[k] / L**k for k in range(kmax + 1)])
    return np.array(ders[0]) - np.array(ders[1])


def interface_jets(F, iface: Interface, lo, hi, rng, n: int, tries: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """n interface points inside the box (the feet of random points) and their one-sided jumps."""
    P, J = [], []
    for _ in range(tries * n):
        if len(P) >= n:
            break
        y = rng.uniform(lo, hi)[None]
        _, foot = iface.geometry(y)
        if not np.all((foot[0] >= lo) & (foot[0] <= hi)):
            continue
        j = one_sided_jumps(F, foot[0], iface.normal(foot)[0], lo, hi)
        if j is not None:
            P.append(foot[0])
            J.append(j)
    return np.array(P).reshape(-1, len(lo)), np.array(J).reshape(-1, 3)


def _profile(F, iface: Interface, lo, hi, atol) -> None:
    """S = h(phi): h a 1-D piecewise ChebTT along the in-box segment from the smallest to the largest phi (for a plane, the corners -/+ sign(normal))."""
    d = len(lo)
    c, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    if iface.kind == "plane":
        s = np.sign(iface.grad(c[None])[0])
        x0, x1 = c - s * half, c + s * half
    else:  # a convex quadric: from its centre (the minimiser of phi, clipped to the box) to the corner farthest from it
        Z0 = -iface.w[d : 2 * d] / np.where(np.abs(iface.w[:d]) > 1e-300, 2.0 * iface.w[:d], 1e-300)
        x0 = np.clip(iface.centre + iface.scale * Z0, lo, hi)
        x1 = np.where(x0 - lo > hi - x0, lo, hi)
        if iface.phi(x1[None])[0] < iface.phi(x0[None])[0]:
            x0, x1 = x1, x0
    u0, u1 = float(iface.phi(x0[None])[0]), float(iface.phi(x1[None])[0])

    def x_of_u(u):  # bisection on the segment parameter (phi monotone along it), to rounding
        a, b = np.zeros(len(u)), np.ones(len(u))
        for _ in range(60):
            mid = 0.5 * (a + b)
            up = iface.phi(x0 + mid[:, None] * (x1 - x0)[None]) < u
            a, b = np.where(up, mid, a), np.where(up, b, mid)
        return x0 + (0.5 * (a + b))[:, None] * (x1 - x0)[None]

    iface.profile = chebtt.build(lambda U: F(x_of_u(U[:, 0])), np.array([u0]), np.array([u1]), 0.25 * atol, max_evals=20_000)
    iface.u0, iface.u1 = u0, u1


def _fields(iface: Interface, P, J, atol, kmax: int = 2) -> None:
    """Jump fields of the lowest degree (0, 1, 2) whose held-out jets (a fifth of the points) agree to 0.1 atol (jet k scaled by 0.2^k / k!)."""
    nv = max(len(P) // 5, 3)
    Pt, Jt, Pv, Jv = P[nv:], J[nv:], P[:nv], J[:nv]
    for deg in (0, 1, 2):
        A = _poly(Pt, deg)
        if len(A) < A.shape[1] + 2:
            break
        coefs = [np.linalg.lstsq(A, Jt[:, k], rcond=1e-12)[0] for k in range(kmax + 1)]
        B = _poly(Pv, deg)
        miss = max(float(np.max(np.abs(B @ coefs[k] - Jv[:, k]))) * 0.2**k / math.factorial(k) for k in range(kmax + 1))
        iface.degree, iface.fields = deg, coefs
        if miss <= 0.1 * atol:
            break


def level_crossings(F, anchor: np.ndarray, others: np.ndarray, level: float, m: int = 17) -> np.ndarray:
    """
    Crossings of ``level`` on the segments from ``anchor`` (the most extreme check point, on the far side
    of a front from most of the box) to each of ``others``, by bisection. The level is fixed by the caller
    for every round: crossings of different levels lie on different surfaces.
    """
    pts = []
    t = np.linspace(0.0, 1.0, m)
    for P in others:
        dvec = P - anchor
        v = F(anchor[None] + t[:, None] * dvec[None])
        sg = np.sign(v - level)
        for c in np.flatnonzero(sg[1:] != sg[:-1])[:1]:
            a, b = t[c], t[c + 1]
            for _ in range(45):
                mid = 0.5 * (a + b)
                if np.sign(F((anchor + mid * dvec)[None])[0] - level) == sg[c]:
                    a = mid
                else:
                    b = mid
            pts.append(anchor + 0.5 * (a + b) * dvec)
    return np.array(pts).reshape(-1, len(anchor))


def build(f, lo, hi, atol: float, seed: int = 0, max_evals: int = 300_000, first_budget: int | None = None) -> AdaptiveTT:
    """An ``AdaptiveTT`` of ``f`` (host (N, d) -> (N,)) on [lo, hi] to tolerance ``atol`` (see the module docstring)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    cache = chebtt._SharedOracle(f)  # one cache for every stage: a retry re-evaluates nothing

    def F(X):
        return cache(X)[:, 0]

    rng = np.random.default_rng(seed + 11)
    stages: dict = {}

    def done(tt, iface, kind):
        return AdaptiveTT(tt, iface, kind, cache.evals, stages)

    pts: list = []
    try:
        tt = chebtt.build(F, lo, hi, atol, seed=seed, max_evals=first_budget or 5000 + 500 * d, interface=pts)
        stages["chebtt"] = cache.evals
        return done(tt, None, "chebtt, piecewise axes" if tt.breaks else "chebtt")
    except chebtt.ObliqueEdge:
        stages["chebtt (stopped at an oblique edge)"] = cache.evals
    except RuntimeError:
        stages["chebtt (first budget)"] = cache.evals
    Xs = chebtt.check_points(lo, hi, 64 + 32 * d, seed)
    fs = F(Xs)
    anchor = Xs[int(np.argmax(np.abs(fs - np.median(fs))))]
    P = np.asarray(pts).reshape(-1, d)
    iface = None
    if not len(P):  # no oblique edge met: probe along segments for a jump or a front
        e0 = cache.evals
        P, steep = segment_edges(F, anchor, Xs[rng.choice(len(Xs), min(len(Xs), 2 * d + 8), replace=False)])
        stages["segment probe"] = cache.evals - e0
        if len(P) < 2:
            e0 = cache.evals
            if steep >= 2:  # a front, before a long retry
                Q = np.zeros((0, d))
                level = 0.5 * (float(fs.max()) + float(fs.min()))  # the mid-level; the median can sit on a front's flat tail
                for _ in range(6):
                    Q = np.concatenate([Q, level_crossings(F, anchor, rng.uniform(lo, hi, (2 * d + 8, d)), level)])
                    iface = fit_interface(Q, lo, hi)
                    if iface is not None:
                        break
                stages["front discovery"] = cache.evals - e0
            if iface is None:
                e0 = cache.evals
                tt = chebtt.build(F, lo, hi, atol, seed=seed, max_evals=max_evals)
                stages["chebtt (full budget)"] = cache.evals - e0
                return done(tt, None, "chebtt, piecewise axes" if tt.breaks else "chebtt")
            e0 = cache.evals
            _profile(F, iface, lo, hi, atol)
            stages["front profile"] = cache.evals - e0
            kind = f"front ({iface.kind}): h(phi)"
    if iface is None:
        e0 = cache.evals
        for _ in range(8):  # segments from the extreme point until a plane or diagonal quadric fits
            if len(P) >= d + 4:
                iface = fit_interface(P, lo, hi)
                if iface is not None:
                    break
            P = np.concatenate([P, segment_edges(F, anchor, rng.uniform(lo, hi, (2 * d + 8, d)))[0]])
        stages["interface discovery"] = cache.evals - e0
        if iface is None:
            raise RuntimeError(f"no plane or diagonal quadric fits the {len(P)} interface points")
        if iface.kind != "plane":  # the jump fields' side: away from the quadric's centre
            Y = rng.uniform(lo, hi, (2000, d))
            rho, _ = iface.geometry(Y)
            gn = np.linalg.norm(iface.grad(Y), axis=1)
            if np.any(rho < 0) and np.any(rho > 0) and gn[rho > 0].min() < gn[rho < 0].min():
                iface.side = -1.0
        e0 = cache.evals
        Pj, Jj = interface_jets(F, iface, lo, hi, rng, 2 * d + 8)
        stages["jets"] = cache.evals - e0
        e0 = cache.evals
        constant = len(Jj) >= 4 and all(np.ptp(Jj[:, k]) * 0.2**k / math.factorial(k) <= 0.1 * atol for k in range(3))
        if constant:
            _profile(F, iface, lo, hi, atol)
            kind = f"interface ({iface.kind}), constant jump: h(phi)"
        else:
            P2, J2 = interface_jets(F, iface, lo, hi, rng, max(3 * (d + 1) * (d + 2) // 2 - len(Pj), 0))
            _fields(iface, np.concatenate([Pj, P2]), np.concatenate([Jj, J2]), atol)
            if iface.side < 0:  # S on the negative side carries the jump with the opposite sign
                iface.fields = [-c for c in iface.fields]
            kind = f"interface ({iface.kind}), jump fields of degree {iface.degree}"
        stages["interface term"] = cache.evals - e0
    e0 = cache.evals
    tt = chebtt.build(lambda X: F(X) - iface.term(X), lo, hi, atol, seed=seed, max_evals=max_evals)
    stages["remainder chebtt"] = cache.evals - e0
    return done(tt, iface, kind)


@dataclass
class VectorAdaptiveTT:
    """One ``AdaptiveTT`` per output, built on one shared cache of f; predictions are (N, outputs)."""

    models: list
    evals: int = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.stack([m.predict(X) for m in self.models], axis=1)


def build_vector(f, lo, hi, atol: float, outputs: int, **kw) -> VectorAdaptiveTT:
    """``build`` per output of a vector-valued f (host (N, d) -> (N, outputs)), all on one cache, so a point any output's build asks for serves the others."""
    cache = chebtt._SharedOracle(f)
    models = [build(lambda X, c=c: cache(X)[:, c], lo, hi, atol, **kw) for c in range(outputs)]
    return VectorAdaptiveTT(models, cache.evals)
