"""
Test problems for the structure-discovery experiments: functions, domains, tolerances, test sets.

Every function takes (N, d) points on the active backend and returns (N, 1) values. Tolerances
are relative to max |f| on the problem's test set, so one tolerance means the same thing for
every function, and depend on the dimension only (``TOLERANCES``): loose ones in high dimension
first, so every method runs quickly while a scheme is still being proven there. The test set is a 300 x 300 grid in 2-D (for the maps) and 50,000 fixed uniform
points above.
"""

from dataclasses import dataclass

import numpy as np

from src import to_host, xp
from src.problems.anisotropic_gaussian import aniso_gaussian
from src.problems.wave import wave

GRID = 300
_SCALES: dict = {}  # (problem, d) -> per-component max |f| of a vector problem on its test set
TOLERANCES = {2: (0.1, 0.01, 0.001), 3: (0.1, 0.03, 0.01), 5: (0.1, 0.05, 0.02), 10: (0.2, 0.1, 0.05)}
N_TEST = 50_000


def _column(g):
    return lambda x: xp.reshape(g(x), (x.shape[0], -1))


def rosenbrock(x):
    """Rosenbrock's valley with unit scale, summed over consecutive pairs (the companion's example at d = 2)."""
    return xp.sum((1 - x[:, :-1]) ** 2 + (x[:, 1:] - x[:, :-1] ** 2) ** 2, axis=1, keepdims=True)


def front(x):
    """A sharp smooth front on the unit sphere: tanh((|x| - 1) / 0.05)."""
    return xp.tanh((xp.sqrt(xp.sum(x**2, axis=1, keepdims=True)) - 1.0) / 0.05)


def lorenz96(forcing: float, tf: float, state: bool = False):
    """
    The first state (or with ``state``, every state) of the Lorenz-96 flow map at ``tf`` (the library's
    per-trajectory DOPRI5, rtol = atol = 1e-6).
    """
    from src.problems.lorenz96_flowmap import lorenz96_flowmap

    return (lambda x: lorenz96_flowmap(x, forcing, tf)) if state else (lambda x: lorenz96_flowmap(x, forcing, tf)[:, :1])


def lotka_volterra(x, tf: float = 2.0, steps: int = 200):
    """The first species of ``lotka_volterra_state``."""
    return lotka_volterra_state(x, tf, steps)[:, :1]


def lotka_volterra_state(x, tf: float = 2.0, steps: int = 200):
    """
    Every species of a competitive Lotka-Volterra network at ``tf``: ``x_i' = x_i (r_i - sum_j A_ij x_j)``,
    with growth rates and interactions drawn once from a fixed seed (A_ii = 1, off-diagonal in [0, 0.8]),
    by fixed-step RK4 (row by row, so a row's value does not depend on the others; error about 1e-10).
    """
    d = int(x.shape[1])
    rng = np.random.default_rng(1_000 + d)
    r_ = xp.asarray(rng.uniform(0.5, 1.5, d))
    A = rng.uniform(0.0, 0.8, (d, d))
    np.fill_diagonal(A, 1.0)
    A = xp.asarray(A)

    def rhs(y):
        return y * (r_ - y @ A.T)

    y, h = xp.asarray(x, dtype=xp.float64), tf / steps
    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# --- Standard N-dimensional test functions (optimisation and sparse-grid benchmarks) ----------
def ackley(x):
    """Ackley: many local minima on a cone, with a kink at the centre (the square root of the mean square)."""
    d = x.shape[1]
    r = xp.sqrt(xp.sum(x**2, axis=1, keepdims=True) / d)
    c = xp.sum(xp.cos(2 * np.pi * x), axis=1, keepdims=True) / d
    return -20.0 * xp.exp(-0.2 * r) - xp.exp(c) + 20.0 + np.e


def rastrigin(x):
    """Rastrigin: a separable bowl with a dense grid of cosine wells."""
    return 10.0 * x.shape[1] + xp.sum(x**2 - 10.0 * xp.cos(2 * np.pi * x), axis=1, keepdims=True)


def griewank(x):
    """Griewank: a bowl times a product of cosines whose frequencies fall as 1 / sqrt(i)."""
    i = xp.arange(1, x.shape[1] + 1)
    return 1.0 + xp.sum(x**2, axis=1, keepdims=True) / 4000.0 - xp.prod(xp.cos(x / xp.sqrt(i)), axis=1, keepdims=True)


def styblinski_tang(x):
    """Styblinski-Tang: a separable quartic with 2^d local minima."""
    return 0.5 * xp.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1, keepdims=True)


def michalewicz(x, m: int = 10):
    """Michalewicz: steep narrow ridges, sharper as m grows (here m = 10), on [0, pi]^d."""
    i = xp.arange(1, x.shape[1] + 1)
    return -xp.sum(xp.sin(x) * xp.sin(i * x**2 / np.pi) ** (2 * m), axis=1, keepdims=True)


# Genz's test functions on [0, 1]^d, with his difficulty constants b (the sum of the a_i).
_GENZ_B = {"oscillatory": 9.0, "product_peak": 7.25, "corner_peak": 1.85, "gaussian": 7.03, "continuous": 20.4}


def genz(kind: str):
    """One of Genz's families, with a and w drawn once per d from a fixed seed and a scaled to sum to b."""

    def f(x):
        d = int(x.shape[1])
        rng = np.random.default_rng(7_000 + d)
        a = rng.uniform(0.0, 1.0, d)
        a = xp.asarray(a * _GENZ_B[kind] / a.sum())
        w = xp.asarray(rng.uniform(0.0, 1.0, d))
        if kind == "oscillatory":
            return xp.cos(2 * np.pi * w[0] + x @ a)[:, None]
        if kind == "product_peak":
            return xp.prod(1.0 / (a ** (-2.0) + (x - w) ** 2), axis=1, keepdims=True)
        if kind == "corner_peak":
            return ((1.0 + x @ a) ** (-(d + 1)))[:, None]
        if kind == "gaussian":
            return xp.exp(-xp.sum(a**2 * (x - w) ** 2, axis=1, keepdims=True))
        return xp.exp(-xp.sum(a * xp.abs(x - w), axis=1, keepdims=True))  # continuous: kinks at x_i = w_i

    return f


# --- Discontinuous test functions ----------------------------------------------------------
def genz_discontinuous(x):
    """Genz's discontinuous family: exp(a.x) where x_1 < w_1 and x_2 < w_2, else 0 (an axis-aligned corner jump)."""
    d = int(x.shape[1])
    rng = np.random.default_rng(7_000 + d)
    a = rng.uniform(0.0, 1.0, d)
    a = xp.asarray(a * 4.3 / a.sum())
    w = rng.uniform(0.0, 1.0, d)
    inside = (x[:, 0] < w[0]) & ((x[:, 1] < w[1]) if d > 1 else True)
    return (xp.exp(x @ a) * inside)[:, None]


def oblique_jump(x):
    """A smooth function with a unit jump across an oblique hyperplane: sin(2 q.x) + [q.x > 0.3], q a fixed random unit vector."""
    d = int(x.shape[1])
    q = np.random.default_rng(8_000 + d).normal(size=d)
    q = xp.asarray(q / np.linalg.norm(q))
    s = x @ q
    return (xp.sin(2.0 * s) + (s > 0.3))[:, None]


def ball_jump(x):
    """A smooth bump that drops to zero across a sphere: exp(-|x - c|^2) inside |x - c| < 0.8, 0 outside (a curved jump)."""
    c = 0.2
    r2 = xp.sum((x - c) ** 2, axis=1, keepdims=True)
    return xp.exp(-r2) * (r2 < 0.64)


# --- Vector-valued problems (R^d -> R^m) ---------------------------------------------------------
def rosenbrock_gradient(x):
    """The gradient of ``rosenbrock`` (a polynomial vector field, m = d)."""
    a, b = x[:, :-1], x[:, 1:]
    g = xp.zeros_like(x)
    g[:, :-1] += -2.0 * (1.0 - a) - 4.0 * a * (b - a**2)
    g[:, 1:] += 2.0 * (b - a**2)
    return g


def genz_triple(x):
    """Genz's oscillatory, product-peak and corner-peak functions side by side: outputs of different structure, m = 3 for any d."""
    return xp.concatenate([genz(k)(x) for k in ("oscillatory", "product_peak", "corner_peak")], axis=1)


def wave_pair(x):
    """``(cos 1.5 q.x, sin 1.5 q.x)``, q a fixed random unit vector: an oblique ridge in two outputs (m = 2)."""
    d = int(x.shape[1])
    q = np.random.default_rng(9_000 + d).normal(size=d)
    s = x @ xp.asarray(q / np.linalg.norm(q))
    return xp.stack([xp.cos(1.5 * s), xp.sin(1.5 * s)], axis=1)


def swirl(x):
    """
    Consecutive coordinate pairs rotated by the angle ``2 exp(-|x|^2)``: a smooth vortex, strong near the
    centre and fading outside it (m = d; for odd d the last coordinate is scaled by the cosine).
    """
    d = int(x.shape[1])
    th = 2.0 * xp.exp(-xp.sum(x**2, axis=1, keepdims=True))
    c, s = xp.cos(th), xp.sin(th)
    y = c * x
    pairs = d // 2
    if pairs:
        u, v = x[:, 0 : 2 * pairs : 2], x[:, 1 : 2 * pairs : 2]
        y[:, 0 : 2 * pairs : 2] = c * u - s * v
        y[:, 1 : 2 * pairs : 2] = s * u + c * v
    return y



def shared_latent(x):
    """
    Six nonlinear functions of two shared coordinates, oblique to every axis and curved
    (``g1 = 1.5 q1.x + 0.6 (q2.x)^2``, ``g2 = 1.5 q2.x``, q1, q2 orthonormal, drawn once per d): a
    slow-manifold proxy, where the outputs share structure that the axes do not show (m = 6).
    """
    d = int(x.shape[1])
    Q, _ = np.linalg.qr(np.random.default_rng(7_700 + d).normal(size=(d, 2)))
    s1, s2 = x @ xp.asarray(Q[:, 0]), x @ xp.asarray(Q[:, 1])
    g1, g2 = 1.5 * s1 + 0.6 * s2**2, 1.5 * s2
    return xp.stack([xp.tanh(2 * g1), xp.exp(-((g1 - 0.5) ** 2)) * xp.cos(2 * g2), xp.sin(g1 + g2), g1 * g2 / 3,
                     1 / (1 + (g1 + 1) ** 2 + g2**2), xp.cos(3 * g2 - g1)], axis=1)  # fmt: skip


@dataclass(frozen=True)
class Problem:
    name: str
    lo: float
    hi: float
    f: object
    vector: bool = False  # f returns (N, m): its components are scaled to max |f_c| = 1 on the test set

    def tolerances(self, d: int) -> tuple[float, ...]:
        """The tolerances run at dimension d (``TOLERANCES``, the nearest listed dimension below d)."""
        return TOLERANCES[max(k for k in TOLERANCES if k <= max(d, min(TOLERANCES)))]

    def domain(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(d, self.lo), np.full(d, self.hi)

    def test_points(self, d: int) -> np.ndarray:
        lo, hi = self.domain(d)
        if d == 2:
            g = (np.arange(GRID) + 0.5) / GRID
            return np.stack(np.meshgrid(lo[0] + g * (hi[0] - lo[0]), lo[1] + g * (hi[1] - lo[1])), axis=-1).reshape(-1, 2)
        return np.random.default_rng(20_261_117).uniform(lo, hi, (N_TEST, d))

    def scale(self, d: int) -> np.ndarray:
        """
        A vector problem's per-component max |f_c| on the test set (1 for a scalar problem), cached. The
        components are divided by it, so that one tolerance is the same relative accuracy for every
        component and the error metric, max over points and components of |error_c| / max |f_c|, does
        not let a component of large range hide the errors of a small one.
        """
        if not self.vector:
            return np.ones(1)
        key = (self.name, d)
        if key not in _SCALES:
            V = np.asarray(to_host(self.f(xp.asarray(self.test_points(d)))), dtype=np.float64)
            _SCALES[key] = np.maximum(np.max(np.abs(V), axis=0), 1e-300)
        return _SCALES[key]

    def oracle(self, d: int):
        """The function the methods train on: f, or for a vector problem f with its components scaled."""
        if not self.vector:
            return self.f
        s = xp.asarray(self.scale(d))
        return lambda x: self.f(x) / s

    def values(self, X: np.ndarray) -> np.ndarray:
        """(N,) for a scalar problem, (N, m) scaled components for a vector one."""
        V = np.asarray(to_host(self.oracle(int(X.shape[1]))(xp.asarray(X))), dtype=np.float64)
        return V.reshape(len(X), -1) if self.vector else V.ravel()


PROBLEMS = {
    "rosenbrock": Problem("rosenbrock", -2.0, 2.0, rosenbrock),
    "gaussian": Problem("gaussian", -3.0, 3.0, _column(aniso_gaussian)),
    "wave": Problem("wave", -5.0, 5.0, _column(wave)),
    "front": Problem("front", -2.0, 2.0, front),
    "ackley": Problem("ackley", -5.0, 5.0, ackley),
    "rastrigin": Problem("rastrigin", -5.12, 5.12, rastrigin),
    "griewank": Problem("griewank", -10.0, 10.0, griewank),
    "styblinski_tang": Problem("styblinski_tang", -5.0, 5.0, styblinski_tang),
    "michalewicz": Problem("michalewicz", 0.0, np.pi, michalewicz),
    **{f"genz_{k}": Problem(f"genz_{k}", 0.0, 1.0, genz(k)) for k in _GENZ_B},
    # discontinuous
    "genz_discontinuous": Problem("genz_discontinuous", 0.0, 1.0, genz_discontinuous),
    "oblique_jump": Problem("oblique_jump", -1.0, 1.0, oblique_jump),
    "ball_jump": Problem("ball_jump", -1.0, 1.0, ball_jump),
    # ODE flow maps
    "lorenz96": Problem("lorenz96", -5.0, 5.0, lorenz96(1.0, 0.1)),
    "lorenz96_chaotic": Problem("lorenz96_chaotic", -5.0, 5.0, lorenz96(8.0, 0.5)),
    "lotka_volterra": Problem("lotka_volterra", 0.1, 1.0, lotka_volterra),
}
VECTOR_PROBLEMS = {
    "rosenbrock_gradient": Problem("rosenbrock_gradient", -2.0, 2.0, rosenbrock_gradient, vector=True),
    "genz_triple": Problem("genz_triple", 0.0, 1.0, genz_triple, vector=True),
    "lotka_volterra_state": Problem("lotka_volterra_state", 0.1, 1.0, lotka_volterra_state, vector=True),
    "swirl": Problem("swirl", -2.0, 2.0, swirl, vector=True),
    "wave_pair": Problem("wave_pair", -3.0, 3.0, wave_pair, vector=True),
    "shared_latent": Problem("shared_latent", -1.0, 1.0, shared_latent, vector=True),
    "lorenz96_state": Problem("lorenz96_state", -5.0, 5.0, lorenz96(1.0, 0.1, state=True), vector=True),
}
PROBLEMS.update(VECTOR_PROBLEMS)
