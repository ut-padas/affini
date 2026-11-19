"""Draw the panels of the global-approximants companion (docs/reports/global-approximants/figures/*.svg) and write example.json.

Usage, from the repository root (``make report-global-figures``):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \\
        docs/reports/global-approximants/figures/make_figures.py

Every data panel comes from real training runs of the schemes, through ``src.schemes``. ChebTT's stages are
recorded by wrapping them (the wrappers change no result, and every recorded build is checked against an
unrecorded one); what the panels show is then rebuilt from its definition (the cross as a skeleton
decomposition, the coefficient tails, the edge search, the compression's budget, the metric groups, the
hub's terms) and the script stops unless the rebuild matches the library. The benchmark panels read the
store (``src.benchmark.store``) and mark the records older than a method's last change to its results, as
the README's table does, so running the script again after a sweep refreshes them. Every number the report
quotes is written to example.json, which the report reads.

BLAS runs on one thread: the hub expansion's Gram products round differently with the thread count.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # before NumPy loads BLAS

import json  # noqa: E402
import logging  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import src.schemes.chebtt.train as chebtt_train  # noqa: E402
from src.schemes import chebtt  # noqa: E402
from src.schemes.chebtt import ChebTTSettings  # noqa: E402
from src.schemes.chebyshev import cheb_points, chebyshev_tail, coefficients_along  # noqa: E402

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

OUT = Path(__file__).resolve().parent
COMPANION = OUT.parents[1] / "adaptive-surrogates" / "figures" / "example.json"  # AFINNI's and ART's runs on Rosenbrock

# ----------------------------------------------------------------------------- the running examples
# Rosenbrock's valley, as in the AFINNI and ART companion: the same box and tolerance.
R_LO, R_HI = np.array([-2.0, -1.0]), np.array([2.0, 3.0])
R_ALPHA = 1.0
SEED = 1

# The feature example on [-1, 1]^2: a separable base with an axis-aligned kink, an oblique ridge, an oblique jump.
F_LO, F_HI = -np.ones(2), np.ones(2)
F_ALPHA = 0.01
KINK = 0.4  # the base's kink, at x1 = KINK
RIDGE_ANGLE, RIDGE_OFFSET, RIDGE_STEEP, RIDGE_HEIGHT = 30.0, 0.2, 6.0, 0.5  # r = H tanh(S (q.x + offset))
JUMP_ANGLE, JUMP_OFFSET, JUMP_HEIGHT = -55.0, 0.6, 0.4  # j = H [p.x > offset]
Q = np.array([np.cos(np.radians(RIDGE_ANGLE)), np.sin(np.radians(RIDGE_ANGLE))])
P = np.array([np.cos(np.radians(JUMP_ANGLE)), np.sin(np.radians(JUMP_ANGLE))])
# Axis slices of F (section 2): x1 fixed for the smooth one and the jump one, x2 fixed for the kink one.
SLICES = {"smooth": (0, -0.6), "kink": (1, 0.6), "jump": (0, 0.8)}  # (fixed axis, its value); the slice runs along the other
ANGLES = (0, 5, 10, 20, 30, 45)  # the ridge alone, turned: ChebTT's rank against the angle

# ----------------------------------------------------------------------------- style
PALETTE = ["#c0392b", "#e1a800", "#2e7d32", "#1f5fa8", "#6d3b0f", "#ef7d1a", "#6a1b9a", "#d81b8a",
           "#00838f", "#7cb342", "#5c6bc0", "#a1887f", "#f06292", "#26a69a", "#8d6e63", "#3949ab"]  # fmt: skip
GREY, LIGHT, DARK, RED, BLUE, GOLD, GREEN = "#8a8a8a", "#d6d6d6", "#424242", "#c0392b", "#1f5fa8", "#e1a800", "#2e7d32"
PINK, CYAN = "#d81b8a", "#00a6c7"
# The feature example's three parts, in every panel that draws them.
FEATURE = {"kink": GOLD, "ridge": CYAN, "jump": PINK}
# Each method's colour in the benchmark's plots (src/benchmark/methods.py), so a reader can move between them.
METHOD = {"chebtt": "#e08a00", "chebtt_metric": "#1b7f5b", "chebtt_adaptive": "#b5179e", "hub": "#5d6d00",
          "afinni": "#6d3b0f", "art": "#1f5fa8", "art_quad": "#2e7d32", "chebtree": "#c0392b", "sparsecheb": "#6a1b9a",
          "kernel": "#2e7d9a", "slowtable": "#8e24aa"}  # fmt: skip
# Two families of fibres in the cross: along x1 and along x2.
FIBRE = {0: "#1f5fa8", 1: "#ef7d1a"}
# Level sets of a function, light where it is small: the background of every map of the domain.
BACKGROUND = LinearSegmentedColormap.from_list("level", ["#ffffff", "#e9edf3", "#c3cad6"])
# |f - f_hat| / alpha over the domain: blues within the tolerance, reds beyond it (as the AFINNI and ART companion).
ERROR_LEVELS = [0, 0.25, 0.5, 0.75, 1.0, 2.0, 1e9]
ERROR_CMAP = ListedColormap(["#f4f8fc", "#d6e6f5", "#aecde9", "#7fb0db", "#f0a79e", "#c0392b"])
ERROR_NORM = BoundaryNorm(ERROR_LEVELS, ERROR_CMAP.N)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "svg.fonttype": "path",
    # A fixed salt and no date make the SVGs identical from run to run.
    "svg.hashsalt": "global-approximants",
    "axes.linewidth": 0.6, "axes.edgecolor": "#777777", "hatch.linewidth": 0.4,
})  # fmt: skip


def rosenbrock(x: np.ndarray) -> np.ndarray:
    return (1.0 - x[:, 0]) ** 2 + (x[:, 1] - x[:, 0] ** 2) ** 2


def base(x: np.ndarray) -> np.ndarray:
    """b: separable (rank 1), analytic in x2, with a kink across the line x1 = KINK."""
    return (0.3 * np.abs(x[:, 0] - KINK) + 0.5 * np.cos(1.5 * x[:, 0])) * (1.0 + 0.3 * np.sin(2.0 * x[:, 1]))


def ridge(x: np.ndarray) -> np.ndarray:
    """r: a smooth, steep step across the oblique line q.x = -RIDGE_OFFSET."""
    return RIDGE_HEIGHT * np.tanh(RIDGE_STEEP * (x @ Q + RIDGE_OFFSET))


def jump(x: np.ndarray) -> np.ndarray:
    """j: a jump of JUMP_HEIGHT across the oblique line p.x = JUMP_OFFSET (the box's lower-right corner)."""
    return JUMP_HEIGHT * (x @ P > JUMP_OFFSET)


LAYERS = {"b": base, "br": lambda x: base(x) + ridge(x), "brj": lambda x: base(x) + ridge(x) + jump(x)}


def grid(lo: np.ndarray, hi: np.ndarray, n: int) -> np.ndarray:
    g = (np.arange(n) + 0.5) / n
    return np.stack(np.meshgrid(lo[0] + g * (hi[0] - lo[0]), lo[1] + g * (hi[1] - lo[1])), axis=-1).reshape(-1, 2)


def worst(model, f, lo, hi, alpha, n: int = 300) -> float:
    """The worst |error| / tolerance on an n x n grid (the benchmark's 2-D test set)."""
    X = grid(lo, hi, n)
    return float(np.max(np.abs(np.ravel(model.predict(X)) - f(X)))) / alpha


def r3(v: float) -> float:
    return float(f"{v:.3g}")


# ----------------------------------------------------------------------------- recording ChebTT
@dataclass
class Run:
    """One recorded ChebTT build: every oracle call with its stage and round, the fibres the sweeps asked for, each round's checks."""

    calls: list = field(default_factory=list)  # (stage, round, X, values)
    fibres: list = field(default_factory=list)  # per call of the grid's values: (round, points, values, multi-indices)
    checks: list = field(default_factory=list)  # (round, Sobol and look-ahead points before the round, RoundCheck, model)
    accepted: object = None  # the model the compression received
    probe: np.ndarray | None = None
    model: object = None  # the returned (compressed) model
    error: str | None = None  # the build's failure, if it failed


STAGES = ("axis_degrees", "_sweeps", "check_round", "refine", "axis_edges", "compress")


def record_chebtt(f, lo, hi, atol, check: bool = True, **settings) -> Run:
    """
    ``chebtt.build`` with its stages wrapped: each oracle call is tagged with the innermost stage running and
    with the round (``-1`` before the first sweeps). With ``check``, the build is repeated without the wrappers
    and must give the same evaluations and the same predictions bit for bit.
    """
    run, stage, rnd = Run(), ["sobol"], [-1]
    originals = {name: getattr(chebtt_train, name) for name in STAGES}
    values = chebtt_train._Grid.values

    def oracle(X):
        V = f(X)
        run.calls.append((stage[-1], rnd[0], np.array(X, copy=True), np.array(V, dtype=np.float64, copy=True)))
        return V

    def staged(name):
        def wrapper(*a, **k):
            stage.append(name)
            if name == "_sweeps":
                rnd[0] += 1
            try:
                out = originals[name](*a, **k)
            finally:
                stage.pop()
            if name == "check_round":
                run.checks.append((rnd[0], np.array(a[3], copy=True), out, a[0]))
            if name == "compress":
                run.accepted, run.probe = a[0], np.array(a[1], copy=True)
            return out

        return wrapper

    def grid_values(self, idx, max_evals):
        V = values(self, idx, max_evals)
        points = np.stack([self.nodes[k][idx[:, k]] for k in range(idx.shape[1])], axis=1)
        run.fibres.append((rnd[0], points, V.copy(), idx.copy()))
        return V

    for name in STAGES:
        setattr(chebtt_train, name, staged(name))
    chebtt_train._Grid.values = grid_values
    try:
        run.model = chebtt.build(oracle, lo, hi, atol, seed=SEED, **settings)
    except RuntimeError as e:
        run.error = str(e)
    finally:
        for name, fn in originals.items():
            setattr(chebtt_train, name, fn)
        chebtt_train._Grid.values = values
    if check:
        try:
            plain = chebtt.build(f, lo, hi, atol, seed=SEED, **settings)
        except RuntimeError as e:
            if run.error != str(e):
                raise SystemExit("Recording ChebTT changed whether it fails.") from e
            return run
        X = np.random.default_rng(0).uniform(lo, hi, (2000, len(lo)))
        if run.model is None or plain.evals != run.model.evals or not np.array_equal(plain.predict(X), run.model.predict(X)):
            raise SystemExit("Recording ChebTT changed its model.")
    return run


def last_sweep_skeleton(run: Run) -> dict:
    """
    In two dimensions a ChebTT is a skeleton (cross) decomposition: the last left-to-right sweep evaluates f on
    the columns J (every x1 node at r chosen x2 nodes) and the rows I (r chosen x1 nodes, every x2 node), and the
    accepted model's node values are f(:, J) f(I, J)^-1 f(I, :). Rebuild them from the recorded fibres, check
    them against the accepted model at its nodes, and check that the model interpolates f on the cross.
    """
    last = max(r for r, *_ in run.fibres)
    cols, rows = [fb for fb in run.fibres if fb[0] == last][-2:]
    _, Pc, Vc, idx_c = cols
    _, Pr, Vr, idx_r = rows
    n0 = len(np.unique(idx_c[:, 0]))
    J = idx_c[: len(idx_c) // n0, 1]  # the r chosen x2 nodes
    r = len(J)
    n1 = len(idx_r) // r
    rows_i = idx_r[::n1, 0]  # the r chosen x1 nodes, the maxvol rows of f(:, J)
    A, B = Vc.reshape(n0, r), Vr.reshape(r, n1)
    M = A @ np.linalg.solve(A[rows_i], B)
    x1, x2 = Pc[::r, 0], Pr[:n1, 1]
    G = np.stack(np.meshgrid(x1, x2, indexing="ij"), axis=-1).reshape(-1, 2)
    model = run.accepted.predict(G).reshape(n0, n1)
    scale = float(np.max(np.abs(M)))
    if not np.allclose(model, M, rtol=0, atol=1e-9 * scale):
        raise SystemExit("The accepted ChebTT is not the skeleton decomposition of its last sweep.")
    if not (np.allclose(M[:, J], A, atol=1e-9 * scale) and np.allclose(M[rows_i], B, atol=1e-9 * scale)):
        raise SystemExit("The skeleton decomposition does not interpolate f on its cross.")
    return {"rank": int(r), "I": x1[rows_i], "J": x2[J], "x1": x1, "x2": x2}


# ----------------------------------------------------------------------------- drawing helpers
def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def box_axes(ax, lo, hi, pad=0.0) -> None:
    w = hi - lo
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - pad * w[0], hi[0] + pad * w[0])
    ax.set_ylim(lo[1] - pad * w[1], hi[1] + pad * w[1])


def level_sets(ax, f, lo, hi, bands=9, log=False, lines=True, n=160) -> None:
    """f's level sets over the box, light where f is small (log-spaced for Rosenbrock's shallow valley)."""
    X = grid(lo, hi, n)
    Z = f(X).reshape(n, n)
    G0, G1 = X[:, 0].reshape(n, n), X[:, 1].reshape(n, n)
    if log:
        levels = np.r_[0.0, np.geomspace(0.08, Z.max(), bands), Z.max() + 1]
    else:
        levels = np.linspace(Z.min() - 1e-9, Z.max() + 1e-9, bands + 1)
    ax.contourf(G0, G1, Z, levels=levels, cmap=BACKGROUND, norm=BoundaryNorm(levels, BACKGROUND.N), zorder=0)
    if lines:
        ax.contour(G0, G1, Z, levels=levels[1:-1], colors="white", linewidths=0.4, zorder=0)


def feature_lines(ax, parts=("kink", "ridge", "jump"), lw=1.0) -> None:
    """The feature example's kink, the ridge's centre line and the jump's line, in their colours."""
    t = np.linspace(-1.5, 1.5, 2)
    if "kink" in parts:
        ax.plot([KINK, KINK], [-1, 1], color=FEATURE["kink"], lw=lw, ls=(0, (3, 1.5)), zorder=4)
    for name, n, c in (("ridge", Q, -RIDGE_OFFSET), ("jump", P, JUMP_OFFSET)):
        if name in parts:
            p0 = c * n
            line = p0 + np.outer(t, [-n[1], n[0]])
            ax.plot(*line.T, color=FEATURE[name], lw=lw, ls=(0, (3, 1.5)) if name == "ridge" else "-", zorder=4)
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)


def error_map(ax, model, f, lo, hi, alpha, n=240) -> np.ndarray:
    """|f - f_hat| / alpha over the box, in the error colours; returns the errors."""
    X = grid(lo, hi, n)
    e = np.abs(np.ravel(model.predict(X)) - f(X)) / alpha
    ax.imshow(e.reshape(n, n), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), cmap=ERROR_CMAP, norm=ERROR_NORM,
              interpolation="nearest", zorder=1)  # fmt: skip
    return e


def error_key() -> None:
    """The colour key of every error map, as in the AFINNI and ART companion."""
    fig, ax = plt.subplots(figsize=(3.4, 0.5))
    cells = len(ERROR_LEVELS) - 1
    x0, cw = 0.05, 0.55
    for i in range(cells):
        ax.add_patch(Rectangle((x0 + i * cw, 0.42), cw, 0.26, fc=ERROR_CMAP(i), ec="none"))
    ax.add_patch(Rectangle((x0, 0.42), cells * cw, 0.26, fc="none", ec=GREY, lw=0.5))
    for i, v in enumerate(ERROR_LEVELS[:-1]):
        ax.text(x0 + i * cw, 0.37, f"{v:g}", ha="center", va="top", fontsize=6.5, color=DARK)
    tol = ERROR_LEVELS.index(1.0)
    ax.plot([x0 + tol * cw] * 2, [0.38, 0.72], color=DARK, lw=0.8)
    ax.text(x0 + tol * cw / 2, 0.76, "within the tolerance", ha="center", va="bottom", fontsize=6.5, color=BLUE)
    ax.text(x0 + (tol + cells) * cw / 2, 0.76, "beyond it", ha="center", va="bottom", fontsize=6.5, color=RED)
    ax.text(x0 + cells * cw / 2, 0.06, "$|f - \\hat{f}| \\,/\\, \\alpha_{tol}$", ha="center", va="top", fontsize=7, color=DARK)
    ax.set_xlim(0, x0 + cells * cw + 0.05)
    ax.set_ylim(-0.12, 1.0)
    ax.axis("off")
    save(fig, "error-key")


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2, pad=1)


def halo(lw=2.2):
    return [patheffects.withStroke(linewidth=lw, foreground="white")]


# ----------------------------------------------------------------------------- section 1: the problem
def section_problem(rosen: Run) -> dict:
    """The two running examples, and where ChebTT's evaluations on Rosenbrock go."""
    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    box_axes(ax, R_LO, R_HI)
    level_sets(ax, rosenbrock, R_LO, R_HI, log=True)
    save(fig, "rosenbrock")
    for name, parts in (("b", ("kink",)), ("br", ("kink", "ridge")), ("brj", ("kink", "ridge", "jump"))):
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        box_axes(ax, F_LO, F_HI)
        level_sets(ax, LAYERS[name], F_LO, F_HI, bands=12)
        feature_lines(ax, parts)
        save(fig, f"feature-{name}")
    # Rosenbrock's evaluated points, by the stage that asked for them.
    groups = {"checks": ("sobol", "check_round"), "degrees": ("axis_degrees",), "fibres": ("_sweeps",)}
    colours = {"checks": GREY, "degrees": GOLD, "fibres": FIBRE[0]}
    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    box_axes(ax, R_LO, R_HI)
    level_sets(ax, rosenbrock, R_LO, R_HI, log=True)
    counts = {}
    for key, stages in groups.items():
        X = np.concatenate([c[2] for c in rosen.calls if c[0] in stages])
        counts[key] = int(len(X))
        ax.scatter(*X.T, s={"checks": 3, "degrees": 5, "fibres": 7}[key], c=colours[key], lw=0, zorder=3 if key == "fibres" else 2)
    if sum(counts.values()) != rosen.model.evals:
        raise SystemExit("Rosenbrock's recorded oracle calls do not add up to the model's evaluations.")
    save(fig, "rosenbrock-points")
    distinct = len({x.tobytes() for c in rosen.calls for x in c[2]})
    return {"evals": int(rosen.model.evals), "distinct": int(distinct), "by_stage": counts,
            "ranks": list(rosen.model.ranks), "degrees": list(rosen.model.degrees),
            "worst": r3(worst(rosen.model, rosenbrock, R_LO, R_HI, R_ALPHA)),
            "accepted_rank": int(max(rosen.accepted.ranks))}  # fmt: skip


# ----------------------------------------------------------------------------- section 2: Chebyshev interpolation
def slice_values(name: str, t: np.ndarray) -> np.ndarray:
    axis, value = SLICES[name]
    X = np.empty((len(t), 2))
    X[:, axis], X[:, 1 - axis] = value, t
    return LAYERS["brj"](X)


def cheb_interpolant(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The degree len(v) - 1 interpolant through values v at the second-kind points, at t (by its coefficients)."""
    a = coefficients_along(v, 0)
    return np.cos(np.outer(np.arccos(np.clip(t, -1, 1)), np.arange(len(a)))) @ a


def equispaced_interpolant(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The interpolant through values v at equispaced points of [-1, 1], at t (barycentric weights (-1)^j C(n, j))."""
    n = len(v) - 1
    x = np.linspace(-1, 1, n + 1)
    w = np.array([(-1) ** j * float(np.prod(np.arange(n - j + 1, n + 1)) / np.prod(np.arange(1, j + 1))) for j in range(n + 1)])
    D = t[:, None] - x[None]
    hit = np.isclose(D, 0.0, atol=1e-15)
    D[hit] = 1.0
    K = w / D
    out = (K @ v) / K.sum(axis=1)
    rows, cols = np.nonzero(hit)
    out[rows] = v[cols]
    return out


def section_chebyshev() -> dict:
    """Chebyshev points against equispaced ones, coefficient decay on three slices, and the tail estimate."""
    out: dict = {}
    # The points: cosines of equally spaced angles, and the first Chebyshev polynomials.
    n = 8
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(2.6, 2.7), gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.15})
    th = np.linspace(0, np.pi, 200)
    top.plot(np.cos(th), np.sin(th), color=GREY, lw=0.7)
    for j in range(n + 1):
        a = np.pi * j / n
        top.plot([np.cos(a)] * 2, [0, np.sin(a)], color=LIGHT, lw=0.6, zorder=1)
        top.scatter([np.cos(a)], [np.sin(a)], s=7, c=GREY, lw=0, zorder=2)
    top.scatter(cheb_points(n), np.zeros(n + 1), s=12, c=DARK, lw=0, zorder=3)
    top.axhline(0, color=GREY, lw=0.6)
    top.set_aspect("equal")
    top.axis("off")
    t = np.linspace(-1, 1, 400)
    for j, c in zip(range(5), (DARK, BLUE, GREEN, GOLD, PINK), strict=True):
        bottom.plot(t, np.cos(j * np.arccos(t)), color=c, lw=1.0)
        bottom.text(1.03, np.cos(j * np.arccos(1.0)) - 0.12 * j, f"$T_{j}$", color=c, fontsize=6.5, va="center")
    tidy(bottom)
    bottom.set_ylim(-1.15, 1.15)
    save(fig, "cheb-points")
    out["points_degree"] = n

    # Where equispaced points stumble: the ridge along the line x = t q through the centre.
    profile = lambda s: RIDGE_HEIGHT * np.tanh(RIDGE_STEEP * (s + RIDGE_OFFSET))  # noqa: E731
    tt = np.linspace(-1, 1, 2001)
    m = 16
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.75), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.plot(tt, profile(tt), color=CYAN, lw=1.6, zorder=1)
    xe, xc = np.linspace(-1, 1, m + 1), cheb_points(m)
    ax.plot(tt, equispaced_interpolant(profile(xe), tt), color=RED, lw=0.9, zorder=2)
    ax.plot(tt, cheb_interpolant(profile(xc), tt), color=DARK, lw=0.9, ls=(0, (3, 1.5)), zorder=3)
    ax.scatter(xe, profile(xe), s=6, c=RED, lw=0, zorder=4)
    ax.scatter(xc, profile(xc), s=6, c=DARK, lw=0, zorder=4)
    ax.set_ylim(-0.9, 0.9)
    tidy(ax)
    degrees = [4, 8, 12, 16, 24, 32, 48, 64]
    e_eq = [float(np.max(np.abs(equispaced_interpolant(profile(np.linspace(-1, 1, k + 1)), tt) - profile(tt)))) for k in degrees]
    e_ch = [float(np.max(np.abs(cheb_interpolant(profile(cheb_points(k)), tt) - profile(tt)))) for k in degrees]
    ax = axes[1]
    ax.semilogy(degrees, e_eq, "o-", color=RED, ms=2.5, lw=0.9)
    ax.semilogy(degrees, e_ch, "o-", color=DARK, ms=2.5, lw=0.9)
    ax.axhline(F_ALPHA, color=GREY, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("degree $n$", fontsize=6.5, labelpad=1)
    tidy(ax)
    save(fig, "runge")
    out["runge"] = {"degree": m, "equispaced": r3(e_eq[degrees.index(m)]), "chebyshev": r3(e_ch[degrees.index(m)]),
                    "equispaced_64": r3(e_eq[-1]), "chebyshev_64": r3(e_ch[-1]),
                    "chebyshev_within": next(k for k, e in zip(degrees, e_ch, strict=True) if e <= F_ALPHA)}  # fmt: skip

    # Decay of the coefficients on three slices of F.
    N = 256
    nodes = cheb_points(N)
    fig, axes = plt.subplots(2, 3, figsize=(5.6, 2.9), gridspec_kw={"height_ratios": [0.8, 1.2], "hspace": 0.45, "wspace": 0.3})
    colour = {"smooth": CYAN, "kink": GOLD, "jump": PINK}
    decay = {}
    j = np.arange(N + 1)
    for col, name in enumerate(SLICES):
        v = slice_values(name, nodes)
        a = np.abs(coefficients_along(v, 0))
        env = np.maximum(a[:-1], a[1:])  # pairwise maxima, so the zero coefficients of a symmetric slice do not read as decay
        ax = axes[0, col]
        ax.plot(tt, slice_values(name, tt), color=colour[name], lw=1.2)
        ax.set_ylim(-1.0, 1.6)
        tidy(ax)
        ax = axes[1, col]
        ax.loglog(j[1:], np.maximum(a[1:], 1e-17), ".", color=colour[name], ms=1.6, alpha=0.5)
        ax.loglog(j[1:-1], np.maximum(env[1:], 1e-17), color=colour[name], lw=1.0)
        if name == "smooth":
            keep = (env > 1e-13) & (j[:-1] >= 4)
            slope, icpt = np.polyfit(j[:-1][keep], np.log(env[keep]), 1)
            ax.loglog(j[4:120], np.exp(icpt + slope * j[4:120]), color=DARK, lw=0.6, ls=(0, (3, 1.5)))
            decay[name] = {"rho": r3(float(np.exp(-slope))), "floor_degree": int(j[:-1][keep].max())}
        else:
            sel = (j[:-1] >= 16) & (j[:-1] <= 200)
            p_fit, icpt = np.polyfit(np.log(j[:-1][sel]), np.log(env[sel]), 1)
            ref = 2 if name == "kink" else 1
            ax.loglog(j[8:], np.exp(icpt) * j[8:] ** -float(ref), color=DARK, lw=0.6, ls=(0, (3, 1.5)))
            decay[name] = {"power": r3(-p_fit), "reference": ref}
        ax.set_ylim(1e-17, 3)
        ax.set_xlabel("degree $j$", fontsize=6.5, labelpad=1)
        tidy(ax)
    save(fig, "decay")
    out["decay"] = decay
    out["slices"] = {k: {"axis": a + 1, "value": v} for k, (a, v) in SLICES.items()}

    # The tail estimate on the smooth slice and on the kink slice: the top quarter, the upper half, the truth.
    share = ChebTTSettings().safety * F_ALPHA / 2
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.85), gridspec_kw={"wspace": 0.3})
    tails = {}
    degs = [8, 16, 32, 64, 128, 256]
    for ax, name in zip(axes, ("smooth", "kink"), strict=True):
        rows = []
        for k in degs:
            v = slice_values(name, cheb_points(k))
            a = np.abs(coefficients_along(v, 0))
            quarter = 2.0 * float(a[max(1, k - k // 4 + 1) :].sum())
            if quarter != chebyshev_tail(v):
                raise SystemExit("The top-quarter tail differs from chebyshev_tail.")
            half = 2.0 * float(a[k // 2 + 1 :].sum())
            true = float(np.max(np.abs(cheb_interpolant(v, tt) - slice_values(name, tt))))
            rows.append((k, quarter, half, true))
        k_, q_, h_, t_ = (np.array(c) for c in zip(*rows, strict=True))
        ax.loglog(k_, t_, "o-", color=colour[name], ms=2.5, lw=1.2)
        ax.loglog(k_, q_, "s-", color=DARK, ms=2.2, lw=0.8)
        ax.loglog(k_, h_, "^-", color=GREY, ms=2.2, lw=0.8)
        ax.axhline(share, color=RED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xticks(degs, [str(d) for d in degs])
        ax.minorticks_off()
        ax.set_xlabel("degree $n$", fontsize=6.5, labelpad=1)
        tidy(ax)
        first = lambda e, k_=k_: next((int(k) for k, x in zip(k_, e, strict=True) if x <= share), None)  # noqa: E731
        tails[name] = {"quarter_degree": first(q_), "half_degree": first(h_), "true_degree": first(t_),
                       "quarter": [r3(x) for x in q_], "true": [r3(x) for x in t_]}  # fmt: skip
    save(fig, "tail")
    out["tail"] = tails | {"share": r3(share), "degrees": degs}
    return out


# ----------------------------------------------------------------------------- section 3: ChebTT
CHECK_ROUND = 4  # the round of ChebTT on b + r whose three check sets are drawn


def matrix_panel() -> dict:
    """f sampled on a tensor grid is a matrix; its singular values say how many products of univariate functions it needs."""
    n = 128
    t = cheb_points(n)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.9), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.35})
    x1, x2 = R_LO[0] + (t[::-1] + 1) * 0.5 * (R_HI - R_LO)[0], R_LO[1] + (t[::-1] + 1) * 0.5 * (R_HI - R_LO)[1]
    G = np.stack(np.meshgrid(x1, x2, indexing="ij"), axis=-1).reshape(-1, 2)
    # log(1 + f): the valley is shallow against the corners, and a linear scale shows only the corners
    axes[0].imshow(np.log1p(rosenbrock(G)).reshape(n + 1, n + 1), cmap=BACKGROUND, aspect="equal", interpolation="nearest")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("column: $x_2$", fontsize=6.5, labelpad=2)
    axes[0].set_ylabel("row: $x_1$", fontsize=6.5, labelpad=2)
    ax = axes[1]
    cases = (("rosenbrock", rosenbrock, R_LO, R_HI, DARK), ("b", LAYERS["b"], F_LO, F_HI, FEATURE["kink"]),
             ("br", LAYERS["br"], F_LO, F_HI, FEATURE["ridge"]), ("brj", LAYERS["brj"], F_LO, F_HI, FEATURE["jump"]))  # fmt: skip
    for name, f, lo, hi, c in cases:
        y1, y2 = lo[0] + (t[::-1] + 1) * 0.5 * (hi - lo)[0], lo[1] + (t[::-1] + 1) * 0.5 * (hi - lo)[1]
        M = f(np.stack(np.meshgrid(y1, y2, indexing="ij"), axis=-1).reshape(-1, 2)).reshape(n + 1, n + 1)
        s = np.linalg.svd(M, compute_uv=False)
        s = s / s[0]
        ax.semilogy(np.arange(1, 41), np.maximum(s[:40], 1e-17), "o-", color=c, ms=1.8, lw=0.8)
        out[name] = {"rank_exact": int(np.sum(s > 1e-12)), "rank_1e3": int(np.sum(s > 1e-3))}
    out["grid"] = n + 1
    ax.set_ylim(1e-17, 3)
    ax.set_xlabel("$k$", fontsize=6.5, labelpad=1)
    ax.set_ylabel("$\\sigma_k / \\sigma_1$", fontsize=6.5, labelpad=1)
    tidy(ax)
    save(fig, "matrix")
    return out


def cross_panel(name: str, run: Run, f, lo, hi, rnd: int, log=False, everything=True, size=2.0) -> dict:
    """The fibres of one round's last left-to-right sweep: lines along x1 at the chosen x2 nodes, along x2 at the chosen x1 nodes."""
    fibres = [fb for fb in run.fibres if fb[0] == rnd][-2:]
    (_, Pc, _, idx_c), (_, Pr, _, idx_r) = fibres
    n0 = len(np.unique(idx_c[:, 0]))
    r = len(idx_c) // n0
    J, Irows = Pc[:r, 1], Pr[:: len(Pr) // r, 0]
    fig, ax = plt.subplots(figsize=(size, size))
    box_axes(ax, lo, hi)
    level_sets(ax, f, lo, hi, log=log, bands=9 if log else 12, lines=False)
    if everything:  # every point evaluated up to this round, faint
        X = np.concatenate([c[2] for c in run.calls if c[1] <= rnd])
        ax.scatter(*X.T, s=1.2, c=GREY, lw=0, alpha=0.6, zorder=1)
    for x2 in J:
        ax.plot([lo[0], hi[0]], [x2, x2], color=FIBRE[0], lw=0.6, zorder=2)
    for x1 in Irows:
        ax.plot([x1, x1], [lo[1], hi[1]], color=FIBRE[1], lw=0.6, zorder=2)
    ax.scatter(*Pc.T, s=3, c=FIBRE[0], lw=0, zorder=3)
    ax.scatter(*Pr.T, s=3, c=FIBRE[1], lw=0, zorder=3)
    piv = np.stack(np.meshgrid(Irows, J), axis=-1).reshape(-1, 2)
    ax.scatter(*piv.T, s=14, c=DARK, lw=0.5, edgecolors="white", zorder=5)
    save(fig, name)
    return {"rank": int(r), "x1_nodes": int(n0), "x2_nodes": int(len(Pr) // r)}


def checks_panel(run: Run, f, rnd: int) -> dict:
    """One round's three check sets on b + r, each point coloured by the model's error there."""
    n_check = 64 + 32 * 2
    rd, Xc_in, chk, model = next(c for c in run.checks if c[0] == rnd)
    sobol, old_look = Xc_in[:n_check], Xc_in[n_check:]
    free = chk.Xall[len(Xc_in) :]
    new_look = chk.Xc[len(Xc_in) :]
    e_all = chk.err / F_ALPHA
    e_look = np.abs(model.predict(new_look) - f(new_look)) / F_ALPHA
    if not np.isclose(e_look.max(), chk.look / F_ALPHA, rtol=1e-9, atol=1e-12):
        raise SystemExit("The look-ahead errors rebuilt from the model do not match the check's.")
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.95), gridspec_kw={"wspace": 0.08})
    sets = ((sobol, e_all[:n_check], "o"), (free, e_all[len(Xc_in) :], "o"), (new_look, e_look, "o"))
    for ax, (X, e, _), k in zip(axes, sets, range(3), strict=True):
        box_axes(ax, F_LO, F_HI)
        level_sets(ax, f, F_LO, F_HI, bands=12, lines=False)
        if k == 2:
            ax.scatter(*chk.bases.T, s=40, marker="x", c=DARK, lw=1.4, zorder=6, path_effects=halo(3))
        order = np.argsort(e)
        ax.scatter(*X[order].T, s=3.5 if k != 1 else 1.6, c=e[order], cmap=ERROR_CMAP, norm=ERROR_NORM, lw=0.15,
                   edgecolors=GREY, zorder=4)  # fmt: skip
    save(fig, "checks")
    return {"round": rnd, "sobol": int(len(sobol)), "old_look": int(len(old_look)), "free": int(len(free)), "look": int(len(new_look)),
            "worst_sobol": r3(e_all[:n_check].max()), "worst_free": r3(e_all[len(Xc_in) :].max()), "worst_look": r3(e_look.max())}  # fmt: skip


def rounds_panel(run: Run) -> list:
    """Each round of ChebTT on b + r: its worst check error, its rank and degrees, and what it did next."""
    rows = []
    for (rnd, _, chk, model), rec in zip(run.checks, run.model.report.rounds, strict=True):
        rows.append({"round": rnd, "worst": r3(chk.worst / F_ALPHA), "rank": int(max(model.ranks)), "degrees": list(model.degrees),
                     "evals": int(rec["evals"]), "action": rec["action"]})  # fmt: skip
    fig, ax = plt.subplots(figsize=(2.6, 1.8))
    marks = {"rank": ("o", FIBRE[0]), "degree": ("s", FIBRE[1]), "breakpoints": ("D", GOLD), "accept": ("*", GREEN)}
    x = [r["evals"] for r in rows]
    ax.semilogy(x, [r["worst"] for r in rows], color=GREY, lw=0.8, zorder=1)
    for r in rows:
        m, c = marks[r["action"]]
        ax.scatter([r["evals"]], [r["worst"]], marker=m, s=34 if m == "*" else 16, c=c, zorder=3, lw=0.4, edgecolors="white")
    ax.axhline(ChebTTSettings().safety, color=RED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("evaluations", fontsize=6.5, labelpad=1)
    ax.set_ylabel("worst check error / $\\alpha_{tol}$", fontsize=6.5, labelpad=1)
    tidy(ax)
    save(fig, "rounds")
    return rows


def edges_panel(run: Run) -> dict:
    """The edge search on b's kink, read off the oracle calls it made: the scan, the zoom, the confirming fibres."""
    calls = [(X, V) for st, _, X, V in run.calls if st == "axis_edges"]
    scan_X, scan_V = calls[0]
    x2_base = float(scan_X[0, 1])
    on_base = [(X, V) for X, V in calls if np.all(X[:, 1] == x2_base)]
    zooms = [(X, V) for X, V in on_base if len(X) == 9]
    far = [(X, V) for X, V in calls if not np.all(X[:, 1] == x2_base)]
    # the zoom's last step, from its definition: the cell of largest second difference and its neighbours
    t, v = zooms[-1][0][:, 0], zooms[-1][1]
    c = int(np.argmax(np.abs(v[2:] - 2.0 * v[1:-1] + v[:-2]))) + 1
    x0 = 0.5 * (t[c - 1] + t[c + 1])
    found = float(run.model.breaks[0][0])
    if x0 != found:
        raise SystemExit("The kink rebuilt from the zoom's last step is not the model's breakpoint.")
    fig, axes = plt.subplots(1, 3, figsize=(5.9, 1.95), gridspec_kw={"wspace": 0.42, "width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    x = scan_X[:, 0]
    d2 = np.abs(scan_V[2:] - 2 * scan_V[1:-1] + scan_V[:-2])
    ax.plot(x, scan_V, color=DARK, lw=0.9)
    ax.scatter(x, scan_V, s=2.5, c=DARK, lw=0)
    tw = ax.twinx()
    tw.bar(x[1:-1], d2, width=(x[1] - x[0]) * 0.8, color=np.where(d2 > 8 * np.median(d2), FEATURE["kink"], LIGHT), zorder=0)
    tw.set_yticks([])
    tw.spines["top"].set_visible(False)
    ax.set_zorder(tw.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel(f"$x_1$ along the fibre $x_2 = {x2_base:.2f}$", fontsize=6.5, labelpad=1)
    tidy(ax)
    ax = axes[1]
    # value bisection halves the bracket at every evaluation, from the scan's cell it started in
    start = abs(on_base[1][0][1, 0] - on_base[1][0][0, 0])
    halving = [start / 2**i for i in range(len([X for X, _ in on_base if len(X) == 1]) + 1)]
    zoom_w = [float(Z[-1, 0] - Z[0, 0]) for Z, _ in zooms] + [float(t[c + 1] - t[c - 1])]
    # against the evaluations spent: one per halving of the value bisection, one zoom grid per step after it
    spent = len(halving) + 3  # its two end values and the two-scale test's two
    ax.semilogy(np.arange(len(halving)) + 2, halving, "-", color=GREY, lw=0.9)
    ax.semilogy(spent + 9 * np.arange(len(zoom_w)), zoom_w, "o-", color=FEATURE["kink"], ms=2.5, lw=1.0)
    ax.set_xlabel("evaluations on the fibre", fontsize=6.5, labelpad=1)
    ax.set_ylabel("bracket width", fontsize=6.5, labelpad=1)
    tidy(ax)
    ax = axes[2]
    box_axes(ax, F_LO, F_HI)
    level_sets(ax, LAYERS["b"], F_LO, F_HI, bands=12, lines=False)
    ax.plot(x, np.full_like(x, x2_base), color=DARK, lw=0.8)
    far_x2 = sorted({float(X[0, 1]) for X, _ in far})
    for y in far_x2:
        ax.plot([-1, 1], [y, y], color=GREY, lw=0.5, ls=(0, (2, 2)))
    for X, _ in far:
        ax.scatter(*X.T, s=5, c=FEATURE["kink"], lw=0, zorder=4)
    ax.plot([found, found], [-1, 1], color=FEATURE["kink"], lw=0.8, zorder=3)
    save(fig, "edges")
    w = F_HI[0] - F_LO[0]
    return {"scan": int(len(scan_X)), "base_x2": r3(x2_base), "found": float(f"{found:.8f}"), "error": r3(abs(found - KINK) / w),
            "zoom_steps": len(zooms), "zoom_width": r3(zoom_w[-1] / w), "bisect_steps": len(halving) - 1,
            "confirm": len(far_x2), "evals": int(sum(len(X) for X, _ in calls))}  # fmt: skip


def compress_panel(run: Run) -> dict:
    """The accepted train and the compressed one, core by core, and how far the compression moved the model."""
    budget = ChebTTSettings().chop_share * F_ALPHA
    again = chebtt.compress(run.accepted, run.probe, budget)
    if not np.array_equal(again.predict(run.probe), run.model.predict(run.probe)):
        raise SystemExit("Compressing the recorded accepted model does not give the returned one.")
    change = np.abs(run.model.predict(run.probe) - run.accepted.predict(run.probe))
    if change.max() > budget:
        raise SystemExit("The compression moved the model by more than its budget.")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.25})
    ax = axes[0]
    y = 0.0
    shapes = []
    for k, (A, B) in enumerate(zip(run.accepted.cores, run.model.cores, strict=True)):
        rowsA, colsA = A.shape[1], A.shape[0] * A.shape[2]
        rowsB, colsB = B.shape[1], B.shape[0] * B.shape[2]
        ax.add_patch(Rectangle((0, y), rowsA, colsA, fc=LIGHT, ec=GREY, lw=0.5))
        ax.add_patch(Rectangle((0, y), rowsB, colsB, fc=FIBRE[k], ec=DARK, lw=0.5, alpha=0.85))
        ax.text(-1.0, y + colsA / 2, f"core {k + 1}", ha="right", va="center", fontsize=6.5)
        shapes.append({"before": list(A.shape), "after": list(B.shape)})
        y += colsA + 3
    ax.set_xlim(-9, max(c.shape[1] for c in run.accepted.cores) + 1)
    ax.set_ylim(-1, y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax = axes[1]
    ax.hist(change / F_ALPHA, bins=40, color=GREY, lw=0)
    ax.axvline(ChebTTSettings().chop_share, color=RED, lw=0.7, ls=(0, (2, 2)))
    ax.set_xlabel("$|\\hat f_{compressed} - \\hat f_{accepted}|\\, /\\, \\alpha_{tol}$ at the probe", fontsize=6.5, labelpad=1)
    ax.set_yticks([])
    tidy(ax)
    save(fig, "compress")
    entries = lambda m: int(sum(C.size for C in m.cores))  # noqa: E731
    return {"shapes": shapes, "entries_before": entries(run.accepted), "entries_after": entries(run.model),
            "change": r3(change.max() / F_ALPHA), "probe": int(len(run.probe)), "budget": ChebTTSettings().chop_share}  # fmt: skip


def angle_panel() -> list:
    """The ridge alone, turned from the x1 axis: ChebTT's rank and evaluations against the angle."""
    rows = []
    for deg in ANGLES:
        q = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        f = lambda x, q=q: RIDGE_HEIGHT * np.tanh(RIDGE_STEEP * (x @ q + RIDGE_OFFSET))  # noqa: E731
        m = chebtt.build(f, F_LO, F_HI, F_ALPHA, seed=SEED)
        rows.append({"angle": deg, "rank": int(max(m.ranks)), "evals": int(m.evals), "degrees": list(m.degrees),
                     "worst": r3(worst(m, f, F_LO, F_HI, F_ALPHA))})  # fmt: skip
    fig, axes = plt.subplots(2, 1, figsize=(2.4, 2.2), sharex=True, gridspec_kw={"hspace": 0.15})
    a = [r["angle"] for r in rows]
    axes[0].plot(a, [r["rank"] for r in rows], "o-", color=FEATURE["ridge"], ms=3, lw=1.0)
    axes[0].set_ylabel("rank", fontsize=6.5, labelpad=1)
    axes[1].plot(a, [r["evals"] for r in rows], "o-", color=METHOD["chebtt"], ms=3, lw=1.0)
    axes[1].set_ylabel("evaluations", fontsize=6.5, labelpad=1)
    axes[1].set_xlabel("angle of the ridge to the $x_1$ axis (degrees)", fontsize=6.5, labelpad=1)
    for ax in axes:
        tidy(ax)
    save(fig, "angle")
    return rows


def model_summary(m, f, lo, hi, alpha) -> dict:
    return {"evals": int(m.evals), "ranks": [int(r) for r in m.ranks], "degrees": [int(n) for n in m.degrees],
            "pieces": [m.pieces(k) for k in range(len(m.cores))], "worst": r3(worst(m, f, lo, hi, alpha)),
            "madds": int(sum(C.size for C in m.cores))}  # fmt: skip


def section_chebtt(rosen: Run) -> dict:
    out: dict = {"matrix": matrix_panel()}
    out["rosenbrock_cross"] = cross_panel("cross-rosenbrock", rosen, rosenbrock, R_LO, R_HI, max(c[0] for c in rosen.checks), log=True,
                                          everything=False, size=2.3)  # fmt: skip
    out["rosenbrock_cross"] |= {"skeleton_rank": last_sweep_skeleton(rosen)["rank"]}
    br = record_chebtt(LAYERS["br"], F_LO, F_HI, F_ALPHA)
    sk = last_sweep_skeleton(br)
    out["skeleton"] = {"rank": sk["rank"]}
    frames = []
    for rnd in sorted({c[0] for c in br.checks}):
        frames.append(cross_panel(f"cross-br-{rnd}", br, LAYERS["br"], F_LO, F_HI, rnd) | {"round": rnd})
    out["cross"] = frames
    out["rounds"] = rounds_panel(br)
    out["checks"] = checks_panel(br, LAYERS["br"], CHECK_ROUND)
    out["compress"] = compress_panel(br)
    b = record_chebtt(LAYERS["b"], F_LO, F_HI, F_ALPHA)
    out["edges"] = edges_panel(b)
    plain = chebtt.build(LAYERS["b"], F_LO, F_HI, F_ALPHA, seed=SEED, edges=False)
    out["models"] = {"rosenbrock": model_summary(rosen.model, rosenbrock, R_LO, R_HI, R_ALPHA),
                     "b": model_summary(b.model, LAYERS["b"], F_LO, F_HI, F_ALPHA),
                     "b_no_edges": model_summary(plain, LAYERS["b"], F_LO, F_HI, F_ALPHA),
                     "br": model_summary(br.model, LAYERS["br"], F_LO, F_HI, F_ALPHA)}  # fmt: skip
    brj = record_chebtt(LAYERS["brj"], F_LO, F_HI, F_ALPHA, check=False)
    out["models"]["brj"] = {"failed": brj.error, "evals": int(sum(len(c[2]) for c in brj.calls))}
    out["angle"] = angle_panel()
    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    box_axes(ax, R_LO, R_HI)
    e = error_map(ax, rosen.model, rosenbrock, R_LO, R_HI, R_ALPHA)
    save(fig, "error-rosenbrock")
    out["models"]["rosenbrock"]["grid_worst"] = r3(e.max())
    s = ChebTTSettings()
    out["defaults"] = {"safety": s.safety, "rank0": s.rank0, "rank_step": s.rank_step, "rank_max": s.rank_max, "sweeps": s.sweeps,
                       "n_fibre": s.n_fibre, "n_max": s.n_max, "n_free": s.n_free, "chop_share": s.chop_share, "n_probe": s.n_probe,
                       "max_pieces": s.max_pieces, "lattice": chebtt.LATTICE}  # fmt: skip
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    from src import IS_CUPY_BACKEND

    if IS_CUPY_BACKEND:
        raise SystemExit("Run with SURROGATE_DEVICE=cpu.")
    companion = json.loads(COMPANION.read_text())
    rosen = record_chebtt(rosenbrock, R_LO, R_HI, R_ALPHA)
    numbers = {
        "rosenbrock": {"lower": R_LO.tolist(), "upper": R_HI.tolist(), "alpha": R_ALPHA, "afinni": companion["compare"]["afinni"],
                       "art": companion["compare"]["art"], "grid": companion["compare"]["grid"]},
        "feature": {"alpha": F_ALPHA, "kink": KINK, "ridge": {"angle": RIDGE_ANGLE, "offset": RIDGE_OFFSET, "steep": RIDGE_STEEP,
                    "height": RIDGE_HEIGHT}, "jump": {"angle": JUMP_ANGLE, "offset": JUMP_OFFSET, "height": JUMP_HEIGHT}},
        "colours": {"palette": PALETTE, "grey": GREY, "light": LIGHT, "dark": DARK, "red": RED, "blue": BLUE, "gold": GOLD,
                    "green": GREEN, "pink": PINK, "cyan": CYAN, "feature": FEATURE, "method": METHOD,
                    "fibre": {str(k): v for k, v in FIBRE.items()}, "error": [matplotlib.colors.to_hex(c) for c in ERROR_CMAP.colors]},
        "seed": SEED,
    }  # fmt: skip
    error_key()
    numbers["rosenbrock"]["chebtt"] = section_problem(rosen)
    numbers["chebyshev"] = section_chebyshev()
    numbers["chebtt"] = section_chebtt(rosen)
    (OUT / "example.json").write_text(json.dumps(numbers, indent=2) + "\n")


if __name__ == "__main__":
    main()
