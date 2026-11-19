"""The convergence section: how AFINNI's and ART's leaves, evaluations and errors grow as the tolerance shrinks.

Usage, from the repository root (``make report-convergence``, about 15 minutes on a laptop's CPU):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/adaptive-surrogates/figures/convergence.py

Trains AFINNI and ART on the running example and on its chain extension to d = 3, 4, 5, at tolerances that
halve from rung to rung, through the public ``AdaptiveSurrogate`` API. AFINNI runs twice per tolerance: from
the running example's fixed initial sample, and from a sample that grows with the leaves (``grown_size``; where
the leaves follow the leaf-count law it grows like tol^(-d/2), the design of the manuscript's absolute-tolerance
experiment). Each model is measured on a fixed test set: its leaves and
evaluations, the share of the domain AFINNI flags (with the exact search), and the error inside, in units of
the tolerance. The script stops unless the d = 2 problem is the running example's function, and unless every
ART leaf polygon it rebuilds from the stored planes contains the leaf's centroid. Writes convergence.json and
the convergence-*.svg panels.

The tolerances end where the slowest method's run, ART's, takes about two minutes on one laptop CPU; past
that point each halving costs ART about 2^(d/2) times more.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import make_figures as mf
import matplotlib.pyplot as plt
import numpy as np

from src import to_host
from src.config import ARTSettings, TrainingConfig
from src.core.evaluation import Evaluator
from src.problems import FUNCTIONS
from src.surrogate import AdaptiveSurrogate

OUT = Path(__file__).resolve().parent
SEED = mf.SEED
# Per dimension, the loosest tolerance and the number of halvings. f grows with d (a sum of d - 1 valleys), so
# the loosest tolerance grows with it.
TOLERANCES = {2: (1.0, 11), 3: (1.0, 5), 4: (8.0, 5), 5: (64.0, 5)}
# Two problems of the benchmark's function suite beside the chain: (d, loosest tolerance, halvings, the number of
# directions f curves in, which sets the slope of the leaf-count law). shared_latent is six functions of two
# oblique coordinates; lorenz96_state is the Lorenz-96 flow over t = 0.1, the manuscript's ODE, every component.
PROBLEMS = {"shared_latent": (5, 0.1, 4, 2), "lorenz96_state": (5, 0.1, 3, 5)}
STRIP = (1.0, 0.25, 0.0625)  # the 2-D tolerances whose leaves the first figure shows
POINTS_PER_LEAF = 100  # the grown sample holds about this many points per leaf it makes (grown_size)
MAX_SAMPLE = 20_000_000  # a grown sample is never larger: the run would not fit a laptop's memory
FIT_LEAVES = 40  # the rates are fitted over the rungs with at least this many leaves
METHODS = ("art", "afinni_scaled", "afinni")
COLOUR = {"afinni": "#6d3b0f", "afinni_scaled": "#6d3b0f", "art": "#1f5fa8"}  # AFINNI's and ART's colours in the benchmark


def chain(x: np.ndarray) -> np.ndarray:
    """The running example's valley repeated along a chain: sum_i (1 - x_i)^2 + (x_(i+1) - x_i^2)^2."""
    return np.sum((1.0 - x[:, :-1]) ** 2 + (x[:, 1:] - x[:, :-1] ** 2) ** 2, axis=1)


def box(d: int) -> tuple[np.ndarray, np.ndarray]:
    """The running example's box, with every further axis like its second."""
    return np.r_[mf.LOWER[0], np.full(d - 1, mf.LOWER[1])], np.r_[mf.UPPER[0], np.full(d - 1, mf.UPPER[1])]


@dataclass(frozen=True)
class Problem:
    """A function on a box (host (N, d) -> (N,) or (N, m)) and its test set."""

    name: str
    f: object
    lo: np.ndarray
    hi: np.ndarray
    X: np.ndarray
    truth: np.ndarray


def chain_problem(d: int) -> Problem:
    lo, hi = box(d)
    X = mf.domain_grid(300) if d == 2 else np.random.default_rng(20_261_119).uniform(lo, hi, (100_000, d))
    return Problem("chain", chain, lo, hi, X, chain(X))


def library_problem(name: str, d: int) -> Problem:
    """A problem of the benchmark's function suite, with its test set; a vector one's outputs scaled to max |f_c| = 1."""
    p = FUNCTIONS[name]
    lo, hi = p.domain(d)
    X = p.test_points(d)
    return Problem(name, p.values, lo, hi, X, p.values(X))


def initial_points(method: str, prob: Problem, n_grown: int | None) -> np.ndarray:
    """ART's root node; AFINNI's fixed sample, the running example's rule of 200 (d + 1) points; or the grown sample."""
    d = len(prob.lo)
    n = {"art": mf.ART_MULTIPLIER * (d + 1), "afinni": 200 * (d + 1), "afinni_scaled": n_grown}[method]
    return np.random.default_rng(SEED).uniform(prob.lo, prob.hi, (n, d))


def grown_size(d: int, fixed: list[dict], grown: list[dict]) -> int:
    """
    The grown sample at the next tolerance: POINTS_PER_LEAF points per leaf it is expected to make, so that
    its leaves are made of sample points. The leaves are extrapolated from the last measured count by its
    recent growth per halving (the grown runs', or the fixed-sample run's at first), at least 1 and at most the
    2^(d/2) of the leaf-count law.
    """
    rows = grown if len(grown) >= 2 else fixed[: len(grown) + 1]
    counts = [r["leaves"] for r in rows[-4:]]
    growth = 1.0 if len(counts) < 2 else (counts[-1] / counts[0]) ** (1 / (len(counts) - 1))
    growth = min(max(growth, 1.0), 2 ** (d / 2))
    expected = fixed[0]["leaves"] if not grown else grown[-1]["leaves"] * growth
    return max(round(POINTS_PER_LEAF * expected), 200 * (d + 1))


def config(method: str, prob: Problem, tol: float) -> TrainingConfig:
    lo, hi, d = prob.lo, prob.hi, len(prob.lo)
    common = {"atol": tol, "rtol": None, "max_levels": 50, "seed": SEED, "use_work_stealing": False}
    if method == "art":
        return TrainingConfig(**common, algorithm="art", domain=(lo, hi), num_initial_clusters=1,
                              art=ARTSettings(sample_multiplier=mf.ART_MULTIPLIER))  # fmt: skip
    # the running example's cluster sizes, per dimension: N_min = 8 (d + 1), four initial clusters
    return TrainingConfig(**common, num_initial_clusters=mf.AFINNI_SETTINGS["num_initial_clusters"],
                          min_pts_per_cluster=8 * (d + 1))  # fmt: skip


def train(method: str, prob: Problem, tol: float, n_grown: int | None = None) -> AdaptiveSurrogate:
    def f_fn(x):
        v = prob.f(np.asarray(x))
        return v.reshape(len(v), -1)

    s = AdaptiveSurrogate()
    s.train(initial_points(method, prob, n_grown), config(method, prob, tol), Evaluator(f_fn=f_fn))
    return s


def measure(s: AdaptiveSurrogate, method: str, prob: Problem, tol: float) -> dict:
    """Leaves, evaluations, the share flagged outside every trust region, and the error inside (worst output) in units of tol."""
    exact = method != "art"
    X = prob.X
    P, outside = s.predict(X, k_nn=1, coarse_factor=mf.KAPPA, exact_outside=exact) if exact else s.predict(X)
    P, outside = np.asarray(to_host(P)).reshape(len(X), -1), np.ravel(to_host(outside)).astype(bool)
    e = np.max(np.abs(P - prob.truth.reshape(len(X), -1)), axis=1)[~outside] / tol
    return {"tol": tol, "leaves": int(s.N_src), "evals": int(s.profiling_stats["total_f_evals"]),
            "flagged": round(100 * float(outside.mean()), 2),
            "within": round(100 * float(np.mean(e <= 1)), 2), "p99": round(float(np.quantile(e, 0.99)), 3),
            "worst": round(float(e.max()), 2), "levels": int(s.profiling_stats["num_levels"]),
            "unresolved": int(s.profiling_stats["num_unresolved_clusters"])}  # fmt: skip


def art_polygons(s: AdaptiveSurrogate) -> list[np.ndarray]:
    """ART's leaf polygons in 2-D, cut out of the box by the stored planes (``mf.clip``), parents before children."""
    r = s.routing
    is_leaf, w, z, codes = (to_host(getattr(r, k)) for k in ("is_leaf", "normal", "offset", "node_locational_code"))
    polygon = {1: np.array([mf.LOWER, [mf.UPPER[0], mf.LOWER[1]], mf.UPPER, [mf.LOWER[0], mf.UPPER[1]]])}
    leaves = []
    for i in np.argsort([int(c) for c in codes], kind="stable"):
        code = int(codes[i])
        if code > 1:
            p = int(np.flatnonzero(codes == code >> 1)[0])
            polygon[code] = mf.clip(polygon[code >> 1], w[p], float(z[p]), keep_right=bool(code & 1))
        if is_leaf[i]:
            leaves.append(polygon[code])
    centres = to_host(s.X_src)
    for poly, c in zip(leaves, centres, strict=True):
        if not inside_polygon(poly, c):
            raise SystemExit("An ART leaf polygon rebuilt from the stored planes does not contain the leaf's centroid.")
    return leaves


def inside_polygon(poly: np.ndarray, x: np.ndarray) -> bool:
    """Whether x lies in the convex polygon (vertices in order, either orientation), up to rounding."""
    edges = np.roll(poly, -1, axis=0) - poly
    cross = edges[:, 0] * (x[1] - poly[:, 1]) - edges[:, 1] * (x[0] - poly[:, 0])
    return bool(np.all(cross >= -1e-9) or np.all(cross <= 1e-9))


def strip_panels(models: dict) -> None:
    """The 2-D partitions at the STRIP tolerances: ART's polygons, AFINNI's trust ellipses from both samples."""
    for tol in STRIP:
        polys = art_polygons(models["art", tol])

        def art(ax, polys=polys):
            for poly in polys:
                ax.fill(*poly.T, fc="none", ec=COLOUR["art"], lw=0.35, zorder=2)

        mf.panel(f"convergence-art-{STRIP.index(tol)}", art, size=(1.55, 1.55))
        for method in ("afinni", "afinni_scaled"):
            s = models[method, tol]
            C, T = to_host(s.X_src), to_host(s.trust_matrices_src)

            def afinni(ax, C=C, T=T):
                for c, t in zip(C, T, strict=True):
                    mf.ellipse(ax, c, t, COLOUR["afinni"], lw=0.35)

            mf.panel(f"convergence-{method.replace('_', '-')}-{STRIP.index(tol)}", afinni, size=(1.55, 1.55))


def fitted_rate(rows: list[dict], key: str = "leaves") -> float | None:
    """The least-squares slope of log(key) against log(1 / tol), over the rungs with at least FIT_LEAVES leaves."""
    use = [r for r in rows if r["leaves"] >= FIT_LEAVES]
    if len(use) < 3:
        return None
    t, n = np.log([1 / r["tol"] for r in use]), np.log([r[key] for r in use])
    return round(float(np.polyfit(t, n, 1)[0]), 2)


def rates_panel(runs: dict) -> None:
    """Leaves against 1 / tol, one panel per dimension, with the slope d / 2 of the theory."""
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.85), sharey=True)
    for ax, d in zip(axes, TOLERANCES, strict=True):
        for method in METHODS:
            rows = runs[method][d]
            x, y = [1 / r["tol"] for r in rows], [r["leaves"] for r in rows]
            hollow = method == "afinni"
            ax.loglog(x, y, "o-" if not hollow else "o--", color=COLOUR[method], ms=2.6, lw=0.9,
                      mfc="white" if hollow else COLOUR[method], mew=0.8, zorder=3)  # fmt: skip
        x0 = 1 / TOLERANCES[d][0]
        x = np.array([x0, x0 * 2 ** (TOLERANCES[d][1] - 1)])
        ref = runs["art"][d][0]["leaves"]
        ax.loglog(x, ref * 0.45 * (x / x0) ** (d / 2), color=mf.GREY, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.set_title(f"$d = {d}$", fontsize=7, pad=2)
        ax.set_xlabel(r"$1 / \alpha_{\mathrm{tol}}$", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.minorticks_off()
    axes[0].set_ylabel("leaves", fontsize=6.5, labelpad=1)
    fig.tight_layout(w_pad=0.4)
    mf.save(fig, "convergence-rates")


def error_panel(runs: dict) -> None:
    """Per method, the error inside (its 99th percentile, in units of tol) and the share AFINNI flags; circles d = 2, squares d = 4."""
    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(4.6, 1.75))
    for d, marker in ((2, "o"), (4, "s")):
        for method in METHODS:
            rows = runs[method][d]
            x = [1 / r["tol"] for r in rows]
            hollow = method == "afinni"
            kw = {"color": COLOUR[method], "ms": 2.6, "lw": 0.9, "mew": 0.8, "mfc": "white" if hollow else COLOUR[method]}
            ax_e.semilogx(x, [r["p99"] for r in rows], marker + ("--" if hollow else "-"), **kw)
            if method != "art":
                ax_f.semilogx(x, [r["flagged"] for r in rows], marker + ("--" if hollow else "-"), **kw)
    ax_e.axhline(1.0, color=mf.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_e.set_ylabel(r"99th percentile of $|f - \hat f| / \alpha_{\mathrm{tol}}$", fontsize=6, labelpad=1)
    ax_f.set_ylabel("share of the domain flagged (%)", fontsize=6, labelpad=1)
    for ax in (ax_e, ax_f):
        ax.set_xlabel(r"$1 / \alpha_{\mathrm{tol}}$", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.set_ylim(bottom=0)
        ax.minorticks_off()
    fig.tight_layout(w_pad=1.0)
    mf.save(fig, "convergence-error")


def problems_panel(problems: dict) -> None:
    """Leaves against 1 / tol on the two library problems, with the slopes k / 2 (grey, k the directions f curves in) and d / 2 (dotted)."""
    fig, axes = plt.subplots(1, len(problems), figsize=(4.6, 1.85), sharey=True)
    for ax, (name, pr) in zip(np.atleast_1d(axes), problems.items(), strict=True):
        for method in METHODS:
            rows = pr["runs"][method]
            hollow = method == "afinni"
            ax.loglog([1 / r["tol"] for r in rows], [r["leaves"] for r in rows], "o--" if hollow else "o-",
                      color=COLOUR[method], ms=2.6, lw=0.9, mfc="white" if hollow else COLOUR[method], mew=0.8, zorder=3)  # fmt: skip
        rows = pr["runs"]["art"]
        x = np.array([1 / rows[0]["tol"], 1 / rows[-1]["tol"]])
        for slope, ls in ((pr["shape"] / 2, (0, (3, 2))), (pr["d"] / 2, ":")):
            ax.loglog(x, rows[0]["leaves"] * 0.45 * (x / x[0]) ** slope, color=mf.GREY, lw=0.8, ls=ls, zorder=2)
        ax.set_title(name, fontsize=7, pad=2)
        ax.set_xlabel(r"$1 / \alpha_{\mathrm{tol}}$", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.minorticks_off()
    np.atleast_1d(axes)[0].set_ylabel("leaves", fontsize=6.5, labelpad=1)
    fig.tight_layout(w_pad=0.6)
    mf.save(fig, "convergence-problems")


def ladders(prob: Problem, tols: list[float], models: dict | None = None) -> dict:
    """Every method at each tolerance: the fixed-sample AFINNI first, since it sizes the grown sample."""
    d = len(prob.lo)
    runs: dict = {}
    for method in ("afinni", "afinni_scaled", "art"):
        rows = []
        for tol in tols:
            n_grown = grown_size(d, runs["afinni"], rows) if method == "afinni_scaled" else None
            if n_grown is not None and n_grown > MAX_SAMPLE:
                break
            s = train(method, prob, tol, n_grown)
            rows.append({**measure(s, method, prob, tol), **({"sample": n_grown} if n_grown else {})})
            if models is not None and tol in STRIP:
                models[method, tol] = s
            print(prob.name, method, d, rows[-1], flush=True)
        runs[method] = rows
    return runs


def main() -> None:
    if not np.array_equal(chain(mf.domain_grid(40)), mf.f_np(mf.domain_grid(40))):
        raise SystemExit("The chain at d = 2 is not the running example's function.")
    runs: dict = {m: {} for m in METHODS}
    models: dict = {}
    for d, (tol0, rungs) in TOLERANCES.items():
        by_method = ladders(chain_problem(d), [tol0 / 2**k for k in range(rungs)], models if d == 2 else None)
        for m in METHODS:
            runs[m][d] = by_method[m]
    problems = {}
    for name, (d, tol0, rungs, shape) in PROBLEMS.items():
        prob = library_problem(name, d)
        by_method = ladders(prob, [tol0 / 2**k for k in range(rungs)])
        problems[name] = {"d": d, "outputs": int(prob.truth.reshape(len(prob.X), -1).shape[1]), "shape": shape,
                          "runs": by_method, "rates": {m: fitted_rate(rows) for m, rows in by_method.items()}}  # fmt: skip
    strip_panels(models)
    rates_panel(runs)
    error_panel(runs)
    problems_panel(problems)
    out = {
        "tolerances": {str(d): {"loosest": t, "rungs": n} for d, (t, n) in TOLERANCES.items()},
        "strip": list(STRIP),
        "box": {"lower": list(map(float, mf.LOWER)), "upper": list(map(float, mf.UPPER))},
        "points_per_leaf": POINTS_PER_LEAF,
        "max_sample": MAX_SAMPLE,
        "fit_leaves": FIT_LEAVES,
        "runs": {m: {str(d): rows for d, rows in by_d.items()} for m, by_d in runs.items()},
        "rates": {m: {str(d): {"leaves": fitted_rate(rows), "evals": fitted_rate(rows, "evals"), "theory": d / 2}
                      for d, rows in by_d.items()} for m, by_d in runs.items()},  # fmt: skip
        "strip_leaves": {m: [runs[m][2][[r["tol"] for r in runs[m][2]].index(t)]["leaves"] for t in STRIP] for m in METHODS},
        "problems": problems,
    }
    (OUT / "convergence.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
