"""Draw the companion's gallery of the functions it trains on (apx-functions.svg, apx-functions.json).

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/adaptive-surrogates/figures/appendix/functions.py

The manuscript study's five functions (``studies/manuscript``) and the convergence study's two vector
functions (``src.problems``), each over the plane of its first two coordinates. A function the study runs at
d = 2 is drawn on its box; one that runs at d = 5 on the plane through the box's centre, the other coordinates
at the centre. A vector function is drawn one output per panel, scaled as the study scales it. The bands hold
equal shares of the plane (quantiles of the values), so functions of any range show their shape.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from src.problems import FUNCTIONS  # noqa: E402
from studies.manuscript import case, spec  # noqa: E402

N = 120  # grid points per axis
BANDS = 9


def plane(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """N x N cell centres over the first two coordinates of the box, the others at its centre."""
    g = (np.arange(N) + 0.5) / N
    G0, G1 = np.meshgrid(lo[0] + g * (hi[0] - lo[0]), lo[1] + g * (hi[1] - lo[1]))
    X = np.tile((lo + hi) / 2, (N * N, 1))
    X[:, 0], X[:, 1] = G0.ravel(), G1.ravel()
    return X


def function_map(ax, X: np.ndarray, Z: np.ndarray, lo, hi) -> None:
    """Equal-area bands of Z over the plane, light where Z is small, as every map of the companion."""
    levels = np.unique(np.quantile(Z, np.linspace(0, 1, BANDS + 1)))
    levels[-1] += 1e-12 * max(1.0, abs(levels[-1]))
    G0, G1, Zg = X[:, 0].reshape(N, N), X[:, 1].reshape(N, N), Z.reshape(N, N)
    norm = BoundaryNorm(levels, mf.BACKGROUND.N)
    # the bands as an image inside the SVG: as paths they are most of its size; the lines stay paths
    ax.contourf(G0, G1, Zg, levels=levels, cmap=mf.BACKGROUND, norm=norm, zorder=0, rasterized=True)
    ax.contour(G0, G1, Zg, levels=levels[1:-1], colors=mf.GREY, linewidths=0.35, linestyles="solid", zorder=1)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([lo[0], hi[0]])
    ax.set_yticks([lo[1], hi[1]])
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.set_aspect("equal")


def panels() -> list[dict]:
    """Every panel: its label, function, box, the d it is drawn at and the output it shows."""
    out = []
    for name in ("quadratic", "gaussian", "rosenbrock", "wave", "lorenz96"):
        fn = spec.FUNCTIONS[name]
        d = 5 if name == "lorenz96" else 2
        f, _ = case.functions(name)
        out.append({"key": name, "label": fn.name + (r", $x_1(0.1)$" if name == "lorenz96" else ""), "d": d,
                    "lo": [fn.domain[0]] * d, "hi": [fn.domain[1]] * d, "output": 0,
                    "f": lambda X, f=f: np.asarray(f(X)).reshape(len(X), -1)})  # fmt: skip
    for name, outputs in (("shared_latent", (0, 1)), ("lorenz96_state", (1,))):
        p = FUNCTIONS[name]
        lo, hi = p.domain(5)
        for k in outputs:
            out.append({"key": f"{name}-{k + 1}", "label": f"{name}, output {k + 1}", "mono": True, "d": 5, "lo": lo.tolist(), "hi": hi.tolist(), "output": k,
                        "f": lambda X, p=p: np.asarray(p.values(X)).reshape(len(X), -1)})  # fmt: skip
    return out


def main() -> None:
    rows = panels()
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.55))
    for ax, row in zip(axes.ravel(), rows, strict=True):
        lo, hi = np.array(row["lo"]), np.array(row["hi"])
        X = plane(lo, hi)
        Z = row["f"](X)[:, row["output"]]
        function_map(ax, X, Z, lo, hi)
        ax.set_title(row["label"], fontsize=5.8 if row.get("mono") else 6.5, pad=4, family="monospace" if row.get("mono") else None)
        ax.text(0.96, 0.95, f"d = {row['d']}", transform=ax.transAxes, fontsize=5.5, color=mf.DARK, ha="right", va="top", zorder=3)
        row["min"], row["max"] = float(Z.min()), float(Z.max())
    fig.subplots_adjust(wspace=0.35, hspace=0.38)
    fig.savefig(mf.OUT / "apx-functions.svg", bbox_inches="tight", pad_inches=0.02, dpi=220, metadata={"Date": None})
    plt.close(fig)
    keep = ("key", "d", "lo", "hi", "output", "min", "max")
    data = {"grid": N, "bands": BANDS, "panels": [{k: r[k] for k in keep} for r in rows],
            "domains": {name: [FUNCTIONS[name].lo, FUNCTIONS[name].hi] for name in ("shared_latent", "lorenz96_state")}}
    (mf.OUT / "apx-functions.json").write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
