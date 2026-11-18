"""
Run, cache and plot surrogate methods on the structure-discovery problems, one CPU thread.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        PYTHONPATH=. .venv/bin/python -m src.structure.bench --problems rosenbrock --d 2 --methods afinni,art,art_quad

Every run (method, problem, d, tolerance, seed) is cached in ``src/structure/cache`` as one
``.npz``: its metrics, and for seed 1 its predictions and leaf ids on the test set. A run is only
trained when its cache file is missing, so the slow baselines (AFINNI, ART with affine and with
quadratic leaves) run once and every new scheme is plotted against them. ``--plot`` draws, per
problem and d, the maps (2-D: each method's pieces and its error map) and the cost curves
(evaluations against the worst error, the mean error and the share within tolerance).

A method is a function ``(f, lo, hi, atol, seed) -> Model``; ``Model`` gives ``predict(X)``,
``leaf_of(X)`` (or None), ``n_leaves`` and ``evals``. New schemes register in ``METHODS``.

Under ``mpiexec`` (``SURROGATE_MPI=on``) the baselines train and predict collectively on every rank,
with work stealing; the new schemes run on rank 0 and spread each round's oracle calls over every
rank (``src.structure.schemes``). Rank 0 alone decides what is cached, scores and writes. Each run
records its rank count, and the plots name it beside the method, since its times are then not
one-thread times.
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from src import to_host, xp  # noqa: E402
from src.config import ARTSettings, TrainingConfig  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.infra.backend import MPI  # noqa: E402
from src.structure.problems import GRID, PROBLEMS  # noqa: E402
from src.surrogate import AdaptiveSurrogate  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "output"
SEEDS = (1, 2, 3)


@dataclass
class Model:
    predict: object  # (N, d) host points -> (N,) host values
    leaf_of: object | None  # (N, d) -> (N,) leaf ids, or None
    n_leaves: int
    evals: int


class CountingOracle:
    """``f`` that counts its evaluated points (the cost every method is charged)."""

    def __init__(self, f):
        self.f, self.count = f, 0

    def __call__(self, X):
        self.count += int(X.shape[0])
        return self.f(X)


# --- Baselines -------------------------------------------------------------------------
def _ranks() -> int:
    return MPI.COMM_WORLD.Get_size()


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


# --- Views of an N-dimensional function -----------------------------------------------------
SLICE = 160


def directions(problem: str, d: int) -> np.ndarray:
    """
    (d, 2) the two directions in which f changes most: the leading eigenvectors of the mean
    gradient outer product, from forward differences at 400 fixed points (cached per problem and d).
    They span the slice plane and the shadow coordinates of the high-dimensional views.
    """
    path = CACHE / f"{problem}_d{d}" / "directions.npz"
    if path.exists():
        return np.load(path)["W"]
    p = PROBLEMS[problem]
    lo, hi = p.domain(d)
    X = np.random.default_rng(5).uniform(lo, hi, (400, d))
    h = 1e-5 * (hi - lo)
    v = p.values(np.concatenate([X] + [X + np.eye(d)[k] * h[k] for k in range(d)])).reshape(d + 1, len(X))
    G = ((v[1:] - v[0]) / h[:, None]).T
    lam, W = np.linalg.eigh(G.T @ G / len(X))
    W = W[:, np.argsort(lam)[::-1][:2]]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=W, eigenvalues=np.sort(lam)[::-1])
    return W


def slice_points(problem: str, d: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(points, inside mask, half-width) of the SLICE x SLICE grid on the plane through the centre spanned by ``directions``."""
    p = PROBLEMS[problem]
    lo, hi = p.domain(d)
    c, R = 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo))
    W = directions(problem, d)
    g = np.linspace(-R, R, SLICE)
    S, T = np.meshgrid(g, g)
    X = c + S.reshape(-1, 1) * W[:, 0] + T.reshape(-1, 1) * W[:, 1]
    inside = np.all((lo <= X) & (hi >= X), axis=1)
    return X, inside, R


# --- Running and caching -----------------------------------------------------------------
def cache_path(method, problem, d, tol, seed) -> Path:
    return CACHE / f"{problem}_d{d}" / f"{method}_tol{tol:g}_s{seed}.npz"


def run(method, problem, d, tol, seed, force=False) -> dict | None:
    """
    The run's metrics, from the cache or by training it now (and caching it). Collective under MPI:
    every rank calls it, rank 0 returns the metrics and the other ranks None.
    """
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    path = cache_path(method, problem, d, tol, seed)
    cached = comm.bcast(path.exists() and not force if rank == 0 else None, root=0)
    if cached:
        return json.loads(str(np.load(path)["metrics"])) if rank == 0 else None
    p = PROBLEMS[problem]
    lo, hi = p.domain(d)
    X = p.test_points(d)
    t0 = time.perf_counter()
    fX = p.values(X)
    oracle_us = 1e6 * (time.perf_counter() - t0) / len(X)
    atol = tol * float(np.max(np.abs(fX)))
    warnings.simplefilter("ignore")
    comm.Barrier()
    t0 = time.perf_counter()
    try:
        model = METHODS[method][2](p.f, lo, hi, atol, seed)
    except RuntimeError as err:  # a scheme over its evaluation budget: recorded, not fatal
        metrics = {"method": method, "problem": problem, "d": d, "tol": tol, "seed": seed, "atol": atol,
                   "failed": str(err)[:300], "train_s": time.perf_counter() - t0, "ranks": size}  # fmt: skip
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, metrics=json.dumps(metrics))
            print(json.dumps(metrics), flush=True)
        return metrics if rank == 0 else None
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X)
    predict_s = time.perf_counter() - t0
    slice_pred = None
    if d > 2 and seed == SEEDS[0]:
        Xs, inside, _ = slice_points(problem, d)
        slice_pred = model.predict(Xs[inside])
    if rank != 0:
        return None
    pred = np.asarray(pred, dtype=np.float64).ravel()
    e = np.abs(pred - fX)
    metrics = {"method": method, "problem": problem, "d": d, "tol": tol, "seed": seed, "atol": atol,
               "leaves": int(model.n_leaves), "evals": int(model.evals), "within_percent": float(100 * np.mean(e <= atol)),
               "mean_error": float(e.mean()), "max_error": float(e.max()), "max_ratio": float(e.max() / atol),
               "train_s": train_s, "predict_us": 1e6 * predict_s / len(X), "oracle_us": oracle_us, "ranks": size}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if seed == SEEDS[0]:
        extra["pred"] = pred.astype(np.float32)
        if model.leaf_of is not None and d == 2:
            extra["leaf"] = np.asarray(model.leaf_of(X)).astype(np.int32)
        if d > 2:
            extra["slice_pred"] = np.asarray(slice_pred, dtype=np.float64).ravel().astype(np.float32)
            if model.leaf_of is not None:
                extra["slice_leaf"] = np.asarray(model.leaf_of(Xs[inside])).astype(np.int32)
    np.savez_compressed(path, metrics=json.dumps(metrics), **extra)
    print(json.dumps(metrics), flush=True)
    return metrics


# --- Plots -------------------------------------------------------------------------------
_PIECES = ListedColormap(["#d6e6f5", "#f6d5a8", "#cfe8cf", "#ead0e8", "#f7e3a3", "#c9e4ea", "#e8c9c0", "#dcdcf0",
                          "#e4ecc0", "#f1cfdc", "#cde0d8", "#e9dccb"])  # fmt: skip
_ERR = ListedColormap(["#f4f8fc", "#d6e6f5", "#aecde9", "#7fb0db", "#f0a79e", "#c0392b"])
_ERR_NORM = BoundaryNorm([0, 0.25, 0.5, 0.75, 1.0, 2.0, 1e9], _ERR.N)


def label_of(method: str, metrics: dict | None) -> str:
    """The method's label, with its rank count when it ran on several."""
    ranks = (metrics or {}).get("ranks", 1)
    return METHODS[method][0] + (f" ({ranks} ranks)" if ranks > 1 else "")


def plot_maps(problem: str, methods: list[str], tol: float, out: Path) -> Path:
    """Each method's pieces (top) and |error| / tolerance (bottom) on the 2-D grid, seed 1."""
    p = PROBLEMS[problem]
    lo, hi = p.domain(2)
    fX = p.values(p.test_points(2))
    ext = (lo[0], hi[0], lo[1], hi[1])
    fig, axes = plt.subplots(2, len(methods), figsize=(3.0 * len(methods), 6.2), constrained_layout=True, squeeze=False)
    for j, method in enumerate(methods):
        z = np.load(cache_path(method, problem, 2, tol, SEEDS[0]))
        m = json.loads(str(z["metrics"]))
        if "failed" in m:
            for a in axes[:, j]:
                a.set_xticks([]), a.set_yticks([])
                a.text(0.5, 0.5, "over its evaluation budget", ha="center", va="center", fontsize=8, transform=a.transAxes)
            axes[0, j].set_title(METHODS[method][0], fontsize=8)
            continue
        e = (np.abs(z["pred"].astype(np.float64) - fX) / m["atol"]).reshape(GRID, GRID)
        ax = axes[0, j]
        if "leaf" in z:
            leaf = z["leaf"].reshape(GRID, GRID).astype(np.int64)
            ax.imshow((leaf * 2654435761) % 12, origin="lower", extent=ext, cmap=_PIECES, vmin=0, vmax=11, interpolation="nearest", aspect="auto")
            edge = (np.diff(leaf, axis=0, prepend=leaf[:1]) != 0) | (np.diff(leaf, axis=1, prepend=leaf[:, :1]) != 0)
            ax.imshow(np.ma.masked_where(~edge, edge), origin="lower", extent=ext, cmap=ListedColormap(["#555555"]),
                      interpolation="nearest", aspect="auto")  # fmt: skip
        ax.contour(*np.meshgrid(np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID)), fX.reshape(GRID, GRID),
                   levels=12, colors="#999999", linewidths=0.4, alpha=0.7)  # fmt: skip
        ax.set_title(f"{label_of(method, m)}\n{m['leaves']} pieces, {m['evals']:,} evaluations", fontsize=8)
        axes[1, j].imshow(e, origin="lower", extent=ext, cmap=_ERR, norm=_ERR_NORM, interpolation="nearest", aspect="auto")
        axes[1, j].set_title(f"error / tolerance: {m['within_percent']:.1f}% within, worst {m['max_ratio']:.2f}", fontsize=8)
        for a in axes[:, j]:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{problem}, tolerance {tol:g} of max |f| (seed {SEEDS[0]}); grey lines: contours of f", fontsize=10)
    path = out / f"{problem}-d2-tol{tol:g}-maps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_views(problem: str, d: int, methods: list[str], tol: float, out: Path) -> Path:
    """
    The high-dimensional views, seed 1. Top row: f on the slice plane, f over the shadow
    coordinates (every test point projected on the two directions), and the distribution of
    |error| / tolerance of every method. Then one row per method: its error on the slice, its
    pieces on the slice (if it has pieces), and its error over the shadow coordinates.
    """
    p = PROBLEMS[problem]
    lo, hi = p.domain(d)
    W = directions(problem, d)
    c = 0.5 * (lo + hi)
    X = p.test_points(d)
    fX = p.values(X)
    Y = (X - c) @ W
    Xs, inside, R = slice_points(problem, d)
    fs = np.full(len(Xs), np.nan)
    fs[inside] = p.values(Xs[inside])
    ext = (-R, R, -R, R)
    runs = []
    for method in methods:
        path = cache_path(method, problem, d, tol, SEEDS[0])
        if path.exists():
            z = np.load(path)
            m = json.loads(str(z["metrics"]))
            runs.append((method, m, z if "failed" not in m else None))
    fig, axes = plt.subplots(1 + len(runs), 3, figsize=(11.0, 3.3 * (1 + len(runs))), constrained_layout=True, squeeze=False)
    axes[0, 0].imshow(fs.reshape(SLICE, SLICE), origin="lower", extent=ext, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("f on the slice plane", fontsize=8)
    order = np.argsort(fX)
    axes[0, 1].scatter(Y[order, 0], Y[order, 1], c=fX[order], s=1, cmap="viridis", rasterized=True)
    axes[0, 1].set_title("f over the shadow coordinates (all test points)", fontsize=8)
    for method, m, z in runs:
        if z is None:
            continue
        e = np.sort(np.abs(z["pred"].astype(np.float64) - fX) / m["atol"])
        axes[0, 2].plot(e, 1.0 - np.arange(len(e)) / len(e), color=METHODS[method][1], label=label_of(method, m), lw=1.4)
    axes[0, 2].set_xscale("log"), axes[0, 2].set_yscale("log"), axes[0, 2].axvline(1.0, color="#999999", lw=0.8, ls="--")
    axes[0, 2].set_xlabel("|error| / tolerance"), axes[0, 2].set_ylabel("share of test points above")
    axes[0, 2].legend(fontsize=6, frameon=False), axes[0, 2].set_title("error distribution", fontsize=8)
    for i, (method, m, z) in enumerate(runs, start=1):
        label = label_of(method, m)
        if z is None:
            for a in axes[i]:
                a.text(0.5, 0.5, "over its evaluation budget", ha="center", va="center", fontsize=8, transform=a.transAxes)
            axes[i, 0].set_title(label, fontsize=8)
            continue
        es = np.full(len(Xs), np.nan)
        es[inside] = np.abs(z["slice_pred"].astype(np.float64) - fs[inside]) / m["atol"]
        axes[i, 0].imshow(es.reshape(SLICE, SLICE), origin="lower", extent=ext, cmap=_ERR, norm=_ERR_NORM, aspect="auto")
        axes[i, 0].set_title(f"{label}: error on the slice\n{m['evals']:,} evaluations, worst {m['max_ratio']:.2f} x tol", fontsize=8)
        if "slice_leaf" in z:
            leaf = np.full(len(Xs), -1, dtype=np.int64)
            leaf[inside] = z["slice_leaf"]
            img = np.ma.masked_where(leaf < 0, (leaf * 2654435761) % 12).reshape(SLICE, SLICE)
            axes[i, 1].imshow(img, origin="lower", extent=ext, cmap=_PIECES, vmin=0, vmax=11, aspect="auto", interpolation="nearest")
            L = leaf.reshape(SLICE, SLICE)
            edge = ((np.diff(L, axis=0, prepend=L[:1]) != 0) | (np.diff(L, axis=1, prepend=L[:, :1]) != 0)) & (L >= 0)
            axes[i, 1].imshow(np.ma.masked_where(~edge, edge), origin="lower", extent=ext, cmap=ListedColormap(["#555555"]),
                              aspect="auto", interpolation="nearest")  # fmt: skip
            axes[i, 1].set_title(f"its {m['leaves']} pieces, cut by the slice", fontsize=8)
        else:
            axes[i, 1].text(0.5, 0.5, "one global expansion: no pieces", ha="center", va="center", fontsize=8,
                            transform=axes[i, 1].transAxes)  # fmt: skip
        e = np.abs(z["pred"].astype(np.float64) - fX) / m["atol"]
        o = np.argsort(e)
        axes[i, 2].scatter(Y[o, 0], Y[o, 1], c=e[o], s=1, cmap=_ERR, norm=_ERR_NORM, rasterized=True)
        axes[i, 2].set_title(f"error over the shadow coordinates: {m['within_percent']:.1f}% within", fontsize=8)
    for a in axes.ravel():
        if a is not axes[0, 2]:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{problem}, d = {d}, tolerance {tol:g}: slice plane and shadow coordinates span the two directions "
                 "in which f changes most", fontsize=9)  # fmt: skip
    path = out / f"{problem}-d{d}-tol{tol:g}-views.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_cost(problem: str, d: int, methods: list[str], out: Path) -> Path:
    """
    Evaluations against the worst error / tolerance, the mean error, the % within, the training
    time and the inference time per query, over the problem's tolerances (seed means). The
    inference panel marks the oracle's own time per point: a surrogate pays off below it.
    """
    p = PROBLEMS[problem]
    tols = p.tolerances(d)
    keys = ("max_ratio", "mean_error", "within_percent", "train_s", "predict_us")
    labels = ("worst |error| / tolerance", "mean |error|", "% within tolerance", "training time (s)",
              "inference time per query (us)")  # fmt: skip
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.6), constrained_layout=True)
    oracle = []
    for method in methods:
        colour = METHODS[method][1]
        label = METHODS[method][0]
        rows = []
        for tol in tols:
            ms = [json.loads(str(np.load(cache_path(method, problem, d, tol, s))["metrics"])) for s in SEEDS
                  if cache_path(method, problem, d, tol, s).exists()]  # fmt: skip
            ms = [m for m in ms if "failed" not in m]
            if ms:
                rows.append([np.mean([m.get(k, np.nan) for m in ms]) for k in ("evals",) + keys])
                oracle += [m["oracle_us"] for m in ms if "oracle_us" in m]
                label = label_of(method, ms[0])
        if not rows:
            continue
        cols = np.array(rows).T
        for ax, y in zip(axes, cols[1:], strict=True):
            ax.plot(cols[0], y, "o-", color=colour, label=label, lw=1.6, ms=4)
    for ax, yl in zip(axes, labels, strict=True):
        ax.set_xscale("log"), ax.set_xlabel("oracle evaluations"), ax.set_ylabel(yl), ax.grid(alpha=0.3)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in (0, 1, 3, 4):
        axes[i].set_yscale("log")
    axes[0].axhline(1.0, color="#999999", lw=0.8, ls="--")
    if oracle:
        axes[4].axhline(float(np.median(oracle)), color="#999999", lw=0.8, ls="--")
        axes[4].text(0.02, 0.96, f"oracle: {np.median(oracle):.2g} us/point", transform=axes[4].transAxes, fontsize=7,
                     va="top", color="#666666")  # fmt: skip
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{problem}, d = {d}: cost against accuracy and time over tolerances {', '.join(map(str, tols))} "
                 "(seed means, one CPU thread)", fontsize=10)  # fmt: skip
    path = out / f"{problem}-d{d}-cost.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_summary(d: int, methods: list[str], out: Path) -> Path:
    """``summary-d<d>.md``: every cached problem's seed-mean metrics per tolerance and method."""
    lines = [f"# Summary, d = {d} (seed means; worst = worst |error| / tolerance)", "",
             "| problem | tol | method | ranks | pieces | evaluations | worst | % within | mean error | train s | infer us/query |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for problem, p in PROBLEMS.items():
        for tol in p.tolerances(d):
            for method in methods:
                ms = [json.loads(str(np.load(cache_path(method, problem, d, tol, s))["metrics"])) for s in SEEDS
                      if cache_path(method, problem, d, tol, s).exists()]  # fmt: skip
                if not ms:
                    continue
                if all("failed" in m for m in ms):
                    lines.append(f"| {problem} | {tol:g} | {METHODS[method][0]} | {ms[0].get('ranks', 1)} | | over budget | | | | | |")
                    continue
                ms = [m for m in ms if "failed" not in m]
                a = {k: np.mean([m[k] for m in ms]) for k in ("leaves", "evals", "max_ratio", "within_percent", "mean_error",
                                                               "train_s", "predict_us")}  # fmt: skip
                lines.append(f"| {problem} | {tol:g} | {METHODS[method][0]} | {ms[0].get('ranks', 1)} | {a['leaves']:.0f} | "
                             f"{a['evals']:,.0f} | {a['max_ratio']:.3g} | {a['within_percent']:.2f} | {a['mean_error']:.2e} | "
                             f"{a['train_s']:.2f} | {a['predict_us']:.2f} |")  # fmt: skip
    path = out / f"summary-d{d}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--problems", default="rosenbrock,gaussian,wave")
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--methods", default=",".join(BASELINES))
    parser.add_argument("--tols", default=None, help="comma-separated; default: the problem's")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot", action="store_true")
    a = parser.parse_args(argv)
    import src.structure.schemes  # noqa: F401  (registers the new schemes)
    from src.structure import (
        bench as module,  # run as __main__, the schemes registered in the imported module
    )

    METHODS.update(module.METHODS)

    methods = a.methods.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    for problem in a.problems.split(","):
        tols = [float(t) for t in a.tols.split(",")] if a.tols else PROBLEMS[problem].tolerances(a.d)
        for tol in tols:
            for method in methods:
                for seed in map(int, a.seeds.split(",")):
                    run(method, problem, a.d, tol, seed, force=a.force)
        if a.plot and MPI.COMM_WORLD.Get_rank() == 0:
            plot_cost(problem, a.d, methods, OUT)
            for tol in tols:
                if a.d == 2:
                    plot_maps(problem, methods, tol, OUT)
                else:
                    plot_views(problem, a.d, methods, tol, OUT)
            write_summary(a.d, methods, OUT)


if __name__ == "__main__":
    main()
