"""Draw the running example's panels of the global-approximants companion and write example.json.

Usage, from the repository root (``make report-global-figures``):

    THEORY_TOL=0.01 SURROGATE_DEVICE=cpu SURROGATE_MPI=off OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
        .venv/bin/python docs/reports/global-approximants/figures/make_figures.py

The running example is the step map of one Brusselator cell (``bruss_d2``: two species in log10
coordinates, the benchmark's 100 training and 40 test trajectories of 40 steps) at the tolerance 0.01.
Every method is built exactly as the comparison study builds it (``theory/panel/p25_compare.py``'s
``build``), and the script stops unless each method's rollout error equals the comparison's record for
the same case, so the pictures show the models the results describe. The solver calls of the box
methods are recorded by a subclass of the oracle cache (it changes no result). Every number the report
quotes about the example is written to example.json.
"""

import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("THEORY_TOL", "0.01")
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "theory" / "panel"), str(ROOT / "theory" / "recipe")]
import src.benchmark.step_maps as sm  # noqa: E402
from src.infra.records import RecordStore  # noqa: E402
from src.schemes.oracle import PointCache  # noqa: E402

OUT = Path(__file__).resolve().parent
NAME = "bruss_d2"
METHODS = ("chebtt_roll", "afinni_roll", "art_roll", "art_quad_roll", "recipe_100")
COLOURS = {"chebtt_roll": "#ef7d1a", "afinni_roll": "#1f5fa8", "art_roll": "#2e7d32", "art_quad_roll": "#00a6c7",
           "recipe_100": "#6a1b9a", "recipe_340": "#d81b8a", "solver": "#424242"}  # fmt: skip
PALETTE = {"dark": "#424242", "grey": "#8a8a8a", "light": "#d6d6d6", "red": "#c0392b", "blue": "#1f5fa8", "refused": "#bdbdbd"}
# |error| / tol: blues within the tolerance, reds beyond it (the other companion's error colours)
ERROR_LEVELS = [0, 0.25, 0.5, 0.75, 1.0, 2.0, 1e9]
ERROR_COLOURS = ["#f4f8fc", "#d6e6f5", "#aecde9", "#7fb0db", "#f0a79e", "#c0392b"]
ERROR_CMAP = ListedColormap(ERROR_COLOURS)
GRID = 64
LABELS = {"chebtt_roll": "ChebTT", "afinni_roll": "AFINNI", "art_roll": "ART (affine)", "art_quad_roll": "ART (quadratic)", "recipe_100": "chart recipe"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "svg.fonttype": "path",
    # A fixed salt and no date make the SVGs identical from run to run.
    "svg.hashsalt": "global-approximants",
    "axes.linewidth": 0.6, "axes.edgecolor": "#777777", "hatch.linewidth": 0.4,
})  # fmt: skip


class Recording(PointCache):
    """The oracle cache, remembering every instance so the points a build paid for can be drawn afterwards."""

    made: list = []

    def __init__(self, f):
        super().__init__(f)
        Recording.made.append(self)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def frame(ax, lo, hi, box=True):
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel(r"$\log_{10} u$")
    ax.set_ylabel(r"$\log_{10} v$")
    if box:
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, lw=0.8, ec=PALETTE["dark"], ls="--"))


def trajectories(ax, Z, colour, lw=0.4, alpha=0.6):
    for z in Z:
        ax.plot(z[:, 0], z[:, 1], color=colour, lw=lw, alpha=alpha)


def lagrange_lebesgue(nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """sum_j |l_j(x)| for interpolation at ``nodes`` (the Lebesgue function of Proposition 5.1 for an interpolant)."""
    L = np.ones((len(x), len(nodes)))
    for j, xj in enumerate(nodes):
        for i, xi in enumerate(nodes):
            if i != j:
                L[:, j] *= (x - xi) / (xj - xi)
    return np.abs(L).sum(1)


def show(ax, values, lo, hi, **kw):
    """A map over the box from row-major grid values (an image: no seams between cells in the SVG)."""
    return ax.imshow(values.reshape(GRID, GRID), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), aspect="auto", interpolation="nearest", **kw)


def least_squares_lebesgue(A: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """sum_i |l_i(t)| with l(t) = A (A^T A)^-1 phi(t): the least-squares fit's weights on its data at each row of Phi."""
    return np.abs(Phi @ np.linalg.pinv(A, rcond=1e-10)).sum(1)


def clean(v):
    """JSON without NaN or infinity (Typst's reader rejects them): a non-finite number becomes null."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def main() -> None:
    sm.PointCache = Recording
    import p25_compare as pc

    from src.problems.ode.reaction_diffusion import BRUSS_A, BRUSS_B

    runs = []
    fit = pc.run

    def captured(*a, **k):
        out = fit(*a, **k)
        runs.append(out)
        return out

    pc.run = captured  # the recipe's selection table, for its panel; the result is unchanged
    import generic4 as g4

    select, chosen = g4.select, {}

    def recorded_degree(*a, **k):
        """The selection, with the degree the comparison recorded: the selection compares measured costs, so a machine under a
        different load can choose another degree, and the panels must show the model the records describe."""
        out = select(*a, **k)
        chosen["this_run"] = out["degree"]
        if chosen.get("recorded") is not None:
            out["degree"] = chosen["recorded"]
        return out

    g4.select = recorded_degree
    from generic import TOL, design, multi_indices

    P = sm.STEP_MAPS[NAME]
    train, test = sm.data(NAME)
    lo, hi = sm.state_box(sm.visited(train))
    d = ROOT / "theory" / "panel" / "data" / "records"
    dstore = RecordStore(d / "p13_data", path_of=lambda k: d / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(d / "p25", path_of=lambda k: d / "p25" / f"{k['problem']}_{k['method']}_{k['tol']}.json")

    import generic3
    import generic4

    from src.schemes.chebtt import ChebTTSettings, VectorSettings

    cs = ChebTTSettings()
    settings = {"rcond": generic4.RCOND, "beta": generic4.BETA, "lambdas": list(generic4.LAMBDAS), "alpha": generic4.ALPHA, "gamma": pc.GAMMA,
                      "margin": generic3.MARGIN, "served_share": generic3.SERVED_SHARE, "sketch_columns": inspect.signature(generic4.Fit4.sketch).parameters["cols"].default, "splits": len(generic3.SPLITS),
                      "chebtt": {k: getattr(cs, k) for k in ("safety", "rank0", "rank_step", "rank_max", "sweeps", "n_fibre", "n_max", "n_free", "chop_share", "max_pieces")} | {"axes_rank": VectorSettings().axes_rank}}  # fmt: skip
    g = [np.linspace(lo[j], hi[j], GRID) for j in range(2)]
    G0, G1 = np.meshgrid(*g)
    Zg = np.stack([G0.ravel(), G1.ravel()], 1)
    Yg = P.step(Zg)
    ex = {"problem": NAME, "tol": TOL, "box": {"lo": lo.tolist(), "hi": hi.tolist()}, "train": int(len(train)), "test": int(len(test)),
          "steps": int(train.shape[1] - 1), "grid": GRID, "settings": settings, "bruss": {"A": BRUSS_A, "B": BRUSS_B}, "colours": {"method": COLOURS, **PALETTE, "error": ERROR_COLOURS}, "methods": {}}  # fmt: skip

    # the states: trajectories collapse onto a curve, the box's corners are never visited
    V = sm.visited(train)
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.plot(V[:, 0], V[:, 1], ".", ms=1.2, color=PALETTE["grey"], alpha=0.6)
    trajectories(ax, test, PALETTE["dark"], lw=0.5, alpha=0.8)
    ax.plot(train[:, 0, 0], train[:, 0, 1], "o", ms=2.0, mfc="none", mec=PALETTE["dark"], mew=0.5)
    frame(ax, lo, hi)
    save(fig, "states")

    def cell(Z):
        return [tuple(c) for c in np.floor((Z - lo) / (hi - lo) * 10).clip(0, 9).astype(int)]

    visited_cells = set(cell(V))
    ex["visited_cells_share"] = len(visited_cells) / 100

    models = {}
    for method in METHODS:
        Recording.made.clear()
        rec = store.read({"problem": NAME, "method": method, "tol": TOL})
        chosen["recorded"] = rec.get("degree") if rec and rec.get("status") == "done" else None
        model, pol, info = pc.build(P, NAME, method, dstore)
        roll = sm.rollout(P, pol(), test)
        if rec is not None and rec.get("status") == "done" and not np.isclose(rec["rollout_worst"], roll["rollout_worst"], rtol=1e-9, atol=0):
            raise SystemExit(f"{method}: rollout {roll['rollout_worst']:.6e} differs from the comparison's record {rec['rollout_worst']:.6e}")
        pts = np.array([np.frombuffer(k) for c in Recording.made for k in c.cache]) if Recording.made else np.empty((0, 2))
        r = np.asarray(model.refuses(Zg), dtype=bool)
        e = np.max(np.abs(Zg + model.predict(Zg) - Yg), axis=1)
        # one-step errors along the test trajectories (the true states), beside the rollout's accumulated errors
        X = test[:, :-1].reshape(-1, 2)
        ro = np.asarray(model.refuses(X), dtype=bool)
        one = np.where(ro, 0.0, np.max(np.abs(X + model.predict(X) - test[:, 1:].reshape(-1, 2)), axis=1)).reshape(len(test), -1).max(0)
        models[method] = {"model": model, "points": pts, "refused": r, "error": e, "roll": roll, "one_step": one}
        ex["methods"][method] = {"solver_calls": int(info["solver_calls"]), "points_drawn": int(len(pts)),
                                 "points_in_empty_cells": float(np.mean([c not in visited_cells for c in cell(pts)])) if len(pts) else None, "rollout_worst": roll["rollout_worst"],
                                 "one_step_worst": float(one.max()), "grid_refused_share": float(r.mean()),
                                 "grid_worst_accepted": float(e[~r].max()) if (~r).any() else None,
                                 "grid_beyond_tol_share": float(np.mean(e[~r] > TOL)) if (~r).any() else None,
                                 "test_refused_share": float(np.mean(roll["rollout_solved_by_step"])),
                                 "record": {k: rec.get(k) for k in ("rollout_fresh_rate", "rollout_steady_rate", "evaluate_us_1000", "denominator_us")}
                                 | {"long": rec.get("long"), "columns": rec.get("columns"), "mixing": rec.get("mixing")} if rec else None,
                                 **{k: v for k, v in info.items() if k in ("degree", "one_step_tol", "margin_rounds")}}  # fmt: skip
        if method.startswith("recipe"):
            ex["methods"][method]["degree_selected_this_run"] = chosen.get("this_run")
            ex["methods"][method]["selection"] = [{k: r[k] for k in ("degree", "terms", "chain_worst", "solver_share", "evaluate_us", "cost_us", "served")}
                                                  for r in runs[-1]["selection"]]  # fmt: skip
        if method == "chebtt_roll":
            ex["methods"][method]["degrees"] = [list(m.degrees) for m in model.models]
            ex["methods"][method]["ranks"] = [list(m.ranks) for m in model.models]
        print(f"{method}: rollout {roll['rollout_worst']:.2e}, one-step {one.max():.2e}, {len(pts)} points, grid refused {r.mean():.2f}")

    # where each method paid for its data, and its error over the box
    for method, m in models.items():
        fig, ax = plt.subplots(figsize=(2.2, 2.0))
        if method.startswith("recipe"):
            trajectories(ax, train, COLOURS[method], lw=0.35, alpha=0.7)
        else:
            ax.plot(m["points"][:, 0], m["points"][:, 1], ".", ms=0.9, color=COLOURS[method], alpha=0.7)
        frame(ax, lo, hi)
        save(fig, f"samples-{method}")
        fig, ax = plt.subplots(figsize=(2.2, 2.0))
        E = np.where(m["refused"], np.nan, m["error"] / TOL).reshape(GRID, GRID)
        show(ax, np.where(m["refused"], 1.0, np.nan), lo, hi, cmap=ListedColormap([PALETTE["refused"]]))
        show(ax, E, lo, hi, cmap=ERROR_CMAP, norm=BoundaryNorm(ERROR_LEVELS, ERROR_CMAP.N))
        trajectories(ax, test, PALETTE["dark"], lw=0.3, alpha=0.5)
        frame(ax, lo, hi)
        save(fig, f"error-{method}")

    fig, ax = plt.subplots(figsize=(3.6, 0.32))
    for i, c in enumerate(ERROR_COLOURS + [PALETTE["refused"]]):
        ax.add_patch(Rectangle((i, 0), 1, 1, color=c, ec="#999999", lw=0.3))
    ax.set_xlim(0, 7), ax.set_ylim(0, 1), ax.set_yticks([])
    ax.set_xticks(np.arange(7) + 0.5, ["< 0.25", "< 0.5", "< 0.75", "< 1", "< 2", "≥ 2", "refused"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "error-key")

    # Proposition 5.1's Lebesgue function: a tensor Chebyshev grid against least squares on the trajectory states
    n = int(max(max(dg) for dg in ex["methods"]["chebtt_roll"]["degrees"]))
    nodes = np.cos(np.pi * np.arange(n + 1) / n)
    S = (2 * Zg - lo - hi) / (hi - lo)
    lam_grid = lagrange_lebesgue(nodes, S[:, 0]) * lagrange_lebesgue(nodes, S[:, 1])
    q = int(ex["methods"]["recipe_100"]["degree"])
    idx = multi_indices(2, q)
    vlo, vhi = V.min(0), V.max(0)
    A = design(V, q, vlo, vhi, idx)
    lam_ls = least_squares_lebesgue(A, design(Zg, q, vlo, vhi, idx))
    T = test[:, :-1].reshape(-1, 2)
    lam_ls_test = least_squares_lebesgue(A, design(T, q, vlo, vhi, idx))
    St = (2 * T - lo - hi) / (hi - lo)
    lam_grid_test = lagrange_lebesgue(nodes, St[:, 0]) * lagrange_lebesgue(nodes, St[:, 1])
    ex["lebesgue"] = {"cheb_degree": n, "ls_degree": q, "ls_terms": len(idx), "ls_samples": int(len(V)),
                      "cheb_box_max": float(lam_grid.max()), "cheb_test_max": float(lam_grid_test.max()),
                      "ls_box_max": float(lam_ls.max()), "ls_test_max": float(lam_ls_test.max()), "ls_test_median": float(np.median(lam_ls_test)),
                      "ls_box_share_above_10": float(np.mean(lam_ls > 10))}  # fmt: skip
    vmax = np.log10(max(lam_grid.max(), lam_ls.max()))
    for name, lam in (("lebesgue-cheb", lam_grid), ("lebesgue-ls", lam_ls)):
        fig, ax = plt.subplots(figsize=(2.3, 2.0))
        im = show(ax, np.log10(lam), lo, hi, cmap="magma_r", vmin=0, vmax=vmax)
        trajectories(ax, test, "#ffffff", lw=0.35, alpha=0.7)
        frame(ax, lo, hi)
        fig.colorbar(im, ax=ax, label=r"$\log_{10}\Lambda$", fraction=0.05, pad=0.03)
        save(fig, name)

    # the same in one dimension: interpolation at designed nodes against least squares on samples that cover only part of the line,
    # as a trajectory's states cover only part of a box
    x = np.linspace(-1, 1, 801)
    m1, n1, edge = 10, 40, -0.2
    cheb = np.cos(np.pi * np.arange(m1 + 1) / m1)
    part = np.linspace(-1, edge, n1)
    lam_part = least_squares_lebesgue(np.polynomial.chebyshev.chebvander(part, m1), np.polynomial.chebyshev.chebvander(x, m1))
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    ax.semilogy(x, lagrange_lebesgue(cheb, x), color=COLOURS["chebtt_roll"], lw=1)
    ax.semilogy(x, lam_part, color=COLOURS["recipe_100"], lw=1)
    ax.plot(cheb, np.full_like(cheb, 0.45), "|", color=COLOURS["chebtt_roll"], ms=5)
    ax.plot(part, np.full_like(part, 0.3), "|", color=COLOURS["recipe_100"], ms=4)
    ax.set_xlabel("$t$"), ax.set_ylabel(r"$\Lambda_X(t)$")
    ax.set_ylim(0.22, None)
    save(fig, "lebesgue-1d")
    ex["lebesgue_1d"] = {"degree": m1, "samples": n1, "edge": edge, "cheb_max": float(lagrange_lebesgue(cheb, x).max()),
                         "part_inside_max": float(lam_part[x <= edge].max()), "part_max": float(lam_part.max())}  # fmt: skip

    # rollouts: the error carried along 40 test trajectories, and the one-step errors that feed it
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.2))
    k = np.arange(1, train.shape[1])
    for method, m in models.items():
        a1.semilogy(k, m["roll"]["rollout_error_by_step"], color=COLOURS[method], lw=1, label=LABELS[method])
        a1.semilogy(k, np.maximum(m["one_step"], 1e-12), color=COLOURS[method], lw=0.6, ls=":")
        a2.plot(k, 100 * np.asarray(m["roll"]["rollout_solved_by_step"]), color=COLOURS[method], lw=1)
    a1.axhline(TOL / 2, color=PALETTE["red"], lw=0.6, ls="--")
    a1.set_xlabel("step"), a1.set_ylabel("worst error over 40 trajectories")
    a2.set_xlabel("step"), a2.set_ylabel("refused (%)")
    fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "rollout")
    for method, m in models.items():
        e = np.asarray(m["roll"]["rollout_error_by_step"])
        ex["methods"][method]["amplification"] = float(e.max() / max(m["one_step"].max(), 1e-300))

    # Corollary 5.3: the speed-up against the refusal rate, for the evaluation costs the comparison measured
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    rho = np.logspace(-5, 0, 200)
    den = None
    for method in METHODS:
        r = (ex["methods"][method]["record"] or {})
        if not r.get("evaluate_us_1000"):
            continue
        den = r["denominator_us"]
        ax.loglog(rho, den / (r["evaluate_us_1000"] + rho * den), color=COLOURS[method], lw=0.8, alpha=0.8, label=LABELS[method])
        lg = r.get("long") or {}
        if lg.get("speedup_steady"):
            ax.plot(max(lg["steady_rate"], 1e-5), lg["speedup_steady"], "o", ms=3.5, color=COLOURS[method])
    ax.loglog(rho, 1 / rho, color=PALETTE["dark"], lw=0.8, ls="--")
    ax.set_xlabel(r"refusal rate $\rho$"), ax.set_ylabel("speed-up $S$")
    ax.set_ylim(1, None)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    save(fig, "speedup")

    # the recipe's selection: each degree's held-out chain error and expected cost per step
    sel = ex["methods"]["recipe_100"]["selection"]
    fig, a1 = plt.subplots(figsize=(3.2, 2.0))
    q = [r["degree"] for r in sel]
    a1.semilogy(q, [r["chain_worst"] for r in sel], "o-", color=COLOURS["recipe_100"], ms=3, lw=0.8)
    a1.axhline(generic3.MARGIN * TOL, color=PALETTE["red"], lw=0.6, ls="--")
    a1.set_xlabel("degree $q$"), a1.set_ylabel("held-out chain error", color=COLOURS["recipe_100"])
    a2 = a1.twinx()
    a2.semilogy(q, [r["cost_us"] for r in sel], "s-", color=PALETTE["dark"], ms=3, lw=0.8)
    a2.set_ylabel("expected cost per step (µs)")
    a1.axvline(ex["methods"]["recipe_100"]["degree"], color=PALETTE["grey"], lw=0.6)
    save(fig, "selection")
    ex["denominator_us"] = den

    (OUT / "example.json").write_text(json.dumps(clean(ex), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
