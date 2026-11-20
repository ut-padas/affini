"""The problems appendix of the schemes' companion: a map of every benchmark function and of Robertson's step map.

Usage, from the repository root (``make report-global-figures`` runs it):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off SURROGATE_ODE_WORKERS=8 PYTHONPATH=. .venv/bin/python \\
        docs/reports/global-approximants/figures/problems.py

Every function of ``src.problems.FUNCTIONS`` at d = 2, the smallest dimension the benchmark runs it at, on its
box, one row per class of ``src.benchmark.table.FUNCTION_CLASSES``; a vector function by its first output,
scaled as the benchmark scales it. Robertson's step map, the increment over one step in dex, by each of its
three outputs over the plane of its two widest axes in the box of the states its training trajectories visit,
the third at the box's centre. The bands hold equal shares of the plane (quantiles of the values), so
functions of any range show their shape. Writes problems-functions.svg, problems-robertson.svg and
problems.json (the domains, grids and settings the appendix's tables quote).
"""

import json
import textwrap

import make_figures as gm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from src.benchmark import functions
from src.benchmark import step_maps as sm
from src.benchmark.table import FUNCTION_CLASSES
from src.problems import FUNCTIONS
from src.problems.genz import GENZ_DIFFICULTY
from src.problems.registry import GRID, N_TEST

N = 120  # grid points per axis of a function's map
N_STEP = 60  # of the step map's: each point is a Radau solve
BANDS = 9


def plane(lo: np.ndarray, hi: np.ndarray, axes: tuple[int, int], n: int) -> np.ndarray:
    """n x n cell centres over two axes of the box, the other coordinates at its centre."""
    g = (np.arange(n) + 0.5) / n
    a, b = axes
    G0, G1 = np.meshgrid(lo[a] + g * (hi[a] - lo[a]), lo[b] + g * (hi[b] - lo[b]))
    X = np.tile((lo + hi) / 2, (n * n, 1))
    X[:, a], X[:, b] = G0.ravel(), G1.ravel()
    return X


def function_map(ax, U: np.ndarray, V: np.ndarray, Z: np.ndarray, n: int, lines_raster: bool = False) -> None:
    """
    Equal-area bands of Z over the plane (U, V), light where Z is small, as the companion's other maps. Where one
    value covers more of the plane than a band (a function that is 0 outside a corner), it gets one band and the
    rest of the values share the others.
    """
    levels = np.unique(np.quantile(Z, np.linspace(0, 1, BANDS + 1)))
    above = np.greater(Z, Z.min())
    if len(levels) < BANDS // 2 + 1 and above.any():
        levels = np.unique(np.r_[Z.min(), np.quantile(Z[above], np.linspace(0, 1, BANDS))])
    if len(levels) < 2:  # a constant function: one band
        levels = np.array([levels[0] - 1, levels[0] + 1])
    levels[-1] += 1e-12 * max(1.0, abs(levels[-1]))
    U, V, Z = U.reshape(n, n), V.reshape(n, n), Z.reshape(n, n)
    # the bands as an image inside the SVG: as paths they are most of its size; the lines stay paths
    ax.contourf(U, V, Z, levels=levels, cmap=gm.BACKGROUND, norm=BoundaryNorm(levels, gm.BACKGROUND.N), zorder=0, rasterized=True)
    lines = ax.contour(U, V, Z, levels=levels[1:-1], colors=gm.GREY, linewidths=0.3, linestyles="solid", zorder=1)
    lines.set_rasterized(lines_raster)


def save(fig, name: str) -> None:
    """gm.save at a fixed raster resolution for the bands."""
    fig.savefig(gm.OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, dpi=220, metadata={"Date": None})
    plt.close(fig)


def functions_panel() -> list[dict]:
    """One row per class, one panel per function at d = 2; returns each function's domain and outputs."""
    rows = list(FUNCTION_CLASSES.items())
    cols = max(len(names) for _, names in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(6.6, 0.98 * len(rows) + 0.1))
    out = []
    for i, (cls, names) in enumerate(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(names):
                ax.set_axis_off()
                continue
            p = FUNCTIONS[names[j]]
            lo, hi = p.domain(2)
            X = plane(lo, hi, (0, 1), N)
            Y = np.asarray(p.values(X)).reshape(len(X), -1)
            function_map(ax, X[:, 0], X[:, 1], Y[:, 0], N, lines_raster=True)  # Rastrigin's lines alone are a megabyte
            ax.set_aspect("equal")
            ax.set_title(names[j], fontsize=5.2, family="monospace", pad=1.5)
            out.append({"name": names[j], "class": cls, "lo": p.lo, "hi": p.hi, "vector": p.vector, "outputs_2d": int(Y.shape[1])})
        axes[i, 0].set_ylabel(textwrap.fill(cls, 13), fontsize=5.8, labelpad=3)
    fig.subplots_adjust(wspace=0.12, hspace=0.32)
    save(fig, "problems-functions")
    return out


def robertson_panel() -> dict:
    """Robertson's increment in dex, each output over the plane of the box's two widest axes, with the visited states."""
    p = sm.STEP_MAPS["robertson"]
    train, _ = sm.data("robertson")
    Z = sm.visited(train)
    lo, hi = sm.state_box(Z)
    widest = tuple(int(k) for k in sorted(np.argsort(hi - lo)[-2:]))
    X = plane(lo, hi, widest, N_STEP)
    dZ = p.step(X) - X
    names = p.system.components
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    for k, ax in enumerate(axes):
        function_map(ax, X[:, widest[0]], X[:, widest[1]], dZ[:, k], N_STEP)
        ax.scatter(Z[:, widest[0]], Z[:, widest[1]], s=0.6, c=gm.DARK, lw=0, alpha=0.5, zorder=2, rasterized=True)
        ax.set_xlim(lo[widest[0]], hi[widest[0]])
        ax.set_ylim(lo[widest[1]], hi[widest[1]])
        ax.set_xlabel(f"$\\log_{{10}} {names[widest[0]][0]}_{names[widest[0]][1:]}$", fontsize=6.5, labelpad=1)
        if k == 0:
            ax.set_ylabel(f"$\\log_{{10}} {names[widest[1]][0]}_{names[widest[1]][1:]}$", fontsize=6.5, labelpad=1)
        ax.set_title(f"increment of $\\log_{{10}} {names[k][0]}_{names[k][1:]}$", fontsize=6.5, pad=2)
        ax.tick_params(length=2, pad=1, labelsize=5.5)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "problems-robertson")
    other = next(k for k in range(len(lo)) if k not in widest)
    start, later = train[:, 0, widest[0]], train[:, 1:-1, widest[0]]  # the initial states, and every later one
    return {"start": [gm.r3(float(start.min())), gm.r3(float(start.max()))], "later": [gm.r3(float(later.min())), gm.r3(float(later.max()))],
            "axes": list(widest), "fixed": other, "fixed_at": gm.r3(float((lo[other] + hi[other]) / 2)),
            "lo": [gm.r3(float(v)) for v in lo], "hi": [gm.r3(float(v)) for v in hi], "grid": N_STEP, "visited": int(len(Z))}


def step_rows() -> list[dict]:
    """Every step map's settings, as ``src.problems.ode`` defines them."""
    rows = []
    for name, p in sm.STEP_MAPS.items():
        s = p.system
        rows.append({"name": name, "family": sm.family(name), "d": sm.dim(name), "regime": p.regime, "log": bool(s.log),
                     "method": s.method, "rtol": s.rtol, "atol": s.atol, "dt": p.dt, "steps": p.steps,
                     "n_train": p.n_train, "n_test": p.n_test, "held": int(s.held)})  # fmt: skip
    return rows


def main() -> None:
    out = {"bands": BANDS, "grid": N, "functions": functions_panel(), "classes": {c: list(n) for c, n in FUNCTION_CLASSES.items()}}
    out["dims"] = [2, 3, 5, 10, 20]  # the default --d of ``python -m src.benchmark.sweep functions``
    out["tolerances"] = {str(d): list(functions.tolerances(d)) for d in out["dims"]}
    out["seeds"] = list(functions.SEEDS)
    out["test"] = {"grid": GRID, "points": N_TEST}
    out["step_maps"] = step_rows()
    out["step_tolerances"] = list(sm.STEP_TOLERANCES)
    out["robertson"] = robertson_panel()
    out["genz"] = dict(GENZ_DIFFICULTY)
    # two functions that are (nearly) zero on the whole test set at the largest d
    big = max(out["dims"])
    out["degenerate"] = {name: float(np.abs(np.asarray(FUNCTIONS[name].values(FUNCTIONS[name].test_points(big)))).max())
                         for name in ("ball_jump", "genz_product_peak")} | {"d": big}  # fmt: skip
    # the companion's own examples: the constants make_figures.py builds them from
    out["examples"] = {"second_output": list(gm.SECOND_OUTPUT), "weak": gm.WEAK, "lift_dims": list(gm.LIFT_DIMS),
                       "hub_d": gm.HUB_D, "rosen_box": [gm.R_LO.tolist(), gm.R_HI.tolist()],
                       "f_box": [gm.F_LO.tolist(), gm.F_HI.tolist()]}  # fmt: skip
    (gm.OUT / "problems.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
