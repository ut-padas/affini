"""
Run, cache and plot surrogate methods on the structure-discovery problems.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        PYTHONPATH=. .venv/bin/python -m src.structure.bench --problems rosenbrock --d 2 --methods afinni,art,art_quad

Every run (method, problem, d, tolerance, seed) is cached in ``src/structure/cache/<setup>`` as one
``.npz``: its metrics, and for seed 1 its predictions and leaf ids on the test set. A run is only
trained when its cache file is missing, so the slow baselines (AFINNI, ART with affine and with
quadratic leaves) run once and every new scheme is plotted against them. ``--plot`` draws, per
problem and d, the maps (2-D: each method's pieces and its error map) and the cost curves
(evaluations against the worst error, the mean error and the share within tolerance).

A method is a function ``(f, lo, hi, atol, seed) -> Model``; ``Model`` gives ``predict(X)``,
``leaf_of(X)`` (or None), ``n_leaves``, ``evals`` and optionally ``refuses(X)``. New schemes
register in ``METHODS``. A refused query is answered by the oracle, as a simulation would fall back
to its solver: the errors are those of the answers given, and ``refused_percent`` is the share of
test points that needed the oracle.

Under ``mpiexec`` (``SURROGATE_MPI=on``) the baselines train and predict collectively on every rank,
with work stealing; the new schemes run on rank 0 and spread each round's oracle calls over every
rank (``src.structure.schemes``). Rank 0 alone decides what is cached, scores and writes. Each run
records its device and rank count, and the cache and ``output/`` hold one directory per setup
(``setup``), so one comparison never mixes devices or rank counts.
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from src import IS_CUPY_BACKEND, to_host, xp  # noqa: E402
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
    refuses: object | None = None  # (N, d) -> (N,) bool: queries the model declines, answered by the oracle


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


def setup() -> str:
    """
    The device and rank count, e.g. ``cpu-1rank``. Runs are cached and plotted per setup and never
    compared across setups: times depend on both, and ART's split decisions differ between CPU and
    GPU, whose sums differ at rounding level.
    """
    return SETUP or f"{'gpu' if IS_CUPY_BACKEND else 'cpu'}-{_ranks()}rank"


SETUP = None  # ``--setup``: read and plot another setup's cache (``--overview`` only; never trains into it)


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
    They span the slice plane and the shadow coordinates of the high-dimensional views. Under MPI
    every rank may compute them at once; each writes its own file and renames it into place, so a
    rank never reads a file another rank is still writing.
    """
    path = CACHE / "directions" / f"{problem}_d{d}.npz"  # shared by every setup: f alone fixes them
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
    tmp = path.with_name(f"{problem}_d{d}-{MPI.COMM_WORLD.Get_rank()}.tmp.npz")
    np.savez(tmp, W=W, eigenvalues=np.sort(lam)[::-1])
    tmp.replace(path)
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
    return CACHE / setup() / f"{problem}_d{d}" / f"{method}_tol{tol:g}_s{seed}.npz"


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
                   "failed": str(err)[:300], "train_s": time.perf_counter() - t0, "ranks": size,
                   "device": setup().split("-")[0]}  # fmt: skip
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
    refused = np.zeros(len(X), dtype=bool) if model.refuses is None else np.asarray(model.refuses(X), dtype=bool)
    pred = np.where(refused, fX, pred)
    if slice_pred is not None and model.refuses is not None:
        r = np.asarray(model.refuses(Xs[inside]), dtype=bool)
        slice_pred = np.where(r, p.values(Xs[inside]), np.asarray(slice_pred, dtype=np.float64).ravel())
    e = np.abs(pred - fX)
    metrics = {"method": method, "problem": problem, "d": d, "tol": tol, "seed": seed, "atol": atol,
               "leaves": int(model.n_leaves), "evals": int(model.evals), "within_percent": float(100 * np.mean(e <= atol)),
               "mean_error": float(e.mean()), "max_error": float(e.max()), "max_ratio": float(e.max() / atol),
               "train_s": train_s, "predict_us": 1e6 * predict_s / len(X), "oracle_us": oracle_us, "ranks": size,
               "refused_percent": float(100 * refused.mean()),
               "device": setup().split("-")[0]}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if seed == SEEDS[0]:
        extra["pred"] = pred.astype(np.float32)
        if refused.any():
            extra["refused"] = refused
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
        refused = f", {m['refused_percent']:.2f}% refused (grey)" if m.get("refused_percent") else ""
        if "refused" in z:
            r = z["refused"].reshape(GRID, GRID)
            axes[1, j].imshow(np.ma.masked_where(~r, r), origin="lower", extent=ext, cmap=ListedColormap(["#9e9e9e"]),
                              interpolation="nearest", aspect="auto")  # fmt: skip
        axes[1, j].set_title(f"error / tolerance: {m['within_percent']:.1f}% within, worst {m['max_ratio']:.2f}{refused}",
                             fontsize=8)  # fmt: skip
        for a in axes[:, j]:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{problem}, tolerance {tol:g} of max |f| (seed {SEEDS[0]}); grey lines: contours of f; {setup()}", fontsize=10)
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
                 f"in which f changes most; {setup()}", fontsize=9)  # fmt: skip
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
                 f"(seed means, {setup()})", fontsize=10)  # fmt: skip
    path = out / f"{problem}-d{d}-cost.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def seed_runs(method: str, problem: str, d: int, tol: float) -> list[dict]:
    """The cached metrics of every seed of one case (failed runs included, marked ``failed``)."""
    paths = [cache_path(method, problem, d, tol, s) for s in SEEDS]
    return [json.loads(str(np.load(p)["metrics"])) for p in paths if p.exists()]


# Worst |error| / tolerance, binned on a diverging scale centred at 1 (ColorBrewer RdBu, safe for
# colour-blind readers): blues pass, reds fail, darker is farther from the tolerance. The fail arm
# starts a step darker than the pass arm, so a cell just over the tolerance never reads as one just
# under it. A ratio equal to an edge falls in the lower bin, so exactly 1 passes.
_RATIO_EDGES = (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0)
_RATIO_COLOURS = ("#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f4a582", "#d6604d", "#b2182b", "#67001f")
_RATIO_NAMES = ("<= 0.01", "0.01-0.1", "0.1-0.5", "0.5-1", "1-2", "2-10", "10-100", "> 100")
_COST = matplotlib.colors.LinearSegmentedColormap.from_list("cost", ["#f4f3f0", "#3d3c3a"])  # one hue: darker costs more
_INK, _MUTED, _ABSENT = "#1a1a19", "#6b6a66", "#f4f3f0"


def _compact(v: float, count: bool = False) -> str:
    """A number in at most five characters, so a value fits one narrow grid cell."""
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e4:
        return f"{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    if count or v >= 100:
        return f"{v:.0f}"
    return f"{v:.0e}".replace("e-0", "e-") if v < 1e-3 else f"{v:.2g}"


def _overview_cell(method: str, problem: str, d: int, tol: float) -> dict | None:
    """
    One case's overview entry, or None when the method was not run on it. Errors are the worst seed's
    (the tolerance must hold for every seed), costs the seed means of the runs that finished; a case
    with any seed over its evaluation budget does not pass.
    """
    ms = seed_runs(method, problem, d, tol)
    if not ms:
        return None
    ok = [m for m in ms if "failed" not in m]
    cell = {"over": len(ms) - len(ok), "ranks": {m.get("ranks", 1) for m in ms}}
    if ok:
        cell |= {k: float(np.mean([m.get(k, 0.0) for m in ok]))
                 for k in ("evals", "leaves", "train_s", "predict_us", "refused_percent", "oracle_us")}  # fmt: skip
        cell["max_ratio"] = max(m["max_ratio"] for m in ok)
        cell["mean_ratio"] = float(np.mean([m["mean_error"] / m["atol"] for m in ok]))
    cell["passed"] = bool(ok) and not cell["over"] and cell["max_ratio"] <= 1.0
    return cell


def _ratio_colour(r: float) -> str:
    return _RATIO_COLOURS[int(np.searchsorted(_RATIO_EDGES, r, side="left"))]


def _ink_on(colour) -> str:
    """Dark or white text, whichever reads on the cell's fill (relative luminance threshold)."""
    r, g, b = matplotlib.colors.to_rgb(colour)
    return _INK if 0.2126 * r**2.2 + 0.7152 * g**2.2 + 0.0722 * b**2.2 > 0.25 else "white"


def _overview_grid(ax, problems, tols, methods, cells, fill, text, title, subtitle, extra_rows=(), row_labels=None,
                   mark_best=False) -> callable:  # fmt: skip
    """
    Draw one panel of the overview: a row per problem, a column group per tolerance, a column per
    method inside it, so the methods compared on one case sit side by side. ``fill(cell)`` and
    ``text(cell)`` give a cell's colour and label; a case not run is a dotted outline, one over its
    evaluation budget is hatched. ``mark_best`` outlines the cheapest method that holds the
    tolerance. ``extra_rows`` are (label, {(tol, method): text}) summary rows below the grid.
    Returns the left x of the column of (tolerance index, method index).
    """
    gap = 0.7
    col = lambda ti, mi: ti * (len(methods) + gap) + mi  # noqa: E731
    box = lambda x, y, **kw: plt.Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, **kw)  # noqa: E731  (a 2-point gap between cells)
    for ri, problem in enumerate(problems):
        for ti, tol in enumerate(tols):
            for mi, method in enumerate(methods):
                c, x = cells.get((method, problem, tol)), col(ti, mi)
                if c is None:
                    ax.add_patch(box(x, ri, fc="none", ec="#cfcec8", lw=0.7, ls=":"))
                    continue
                colour = fill(c) if "max_ratio" in c else "#d9d8d3"
                ax.add_patch(box(x, ri, fc=colour, lw=0))
                if c["over"]:
                    ax.add_patch(box(x, ri, fc="none", ec="#8f8e89", lw=0, hatch="///", hatch_linewidth=0.6))
                best = mark_best and c.get("best")
                if best:
                    ax.add_patch(box(x, ri, fc="none", ec=_INK, lw=1.8, zorder=3))
                ax.text(x + 0.5, ri + 0.5, text(c) if "max_ratio" in c else "over", ha="center", va="center", fontsize=7,
                        color=_ink_on(colour), linespacing=1.1, fontweight="bold" if best else "normal")  # fmt: skip
    n = len(problems)
    for k, (label, values) in enumerate(extra_rows):
        y = n + 0.55 + 0.6 * k
        ax.text(-0.3, y, label, ha="right", va="center", fontsize=7.5, color=_MUTED)
        for (tol, method), value in values.items():
            ax.text(col(tols.index(tol), methods.index(method)) + 0.5, y, value, ha="center", va="center", fontsize=7.5,
                    color=_INK)  # fmt: skip
    head = 6 + 4.3 * max(len(_short(m)) for m in methods)  # points above the grid taken by the method names
    above = lambda dy: matplotlib.transforms.offset_copy(ax.transData, fig=ax.figure, y=dy, units="points")  # noqa: E731
    for ti, tol in enumerate(tols):
        x0, x1 = col(ti, 0), col(ti, len(methods) - 1) + 1
        ax.text((x0 + x1) / 2, -0.2, f"tolerance {tol:g}", ha="center", va="bottom", fontsize=8.5, color=_INK,
                transform=above(head + 6), clip_on=False)  # fmt: skip
        ax.plot([x0 + 0.05, x1 - 0.05], [-0.2, -0.2], color=_MUTED, lw=0.6, clip_on=False, transform=above(head + 3))
        for mi, method in enumerate(methods):
            x = col(ti, mi)
            ax.add_patch(plt.Rectangle((x + 0.2, -0.2), 0.6, 0.12, fc=METHODS[method][1], lw=0, clip_on=False))
            ax.text(x + 0.5, -0.2, _short(method), rotation=90, ha="center", va="bottom", fontsize=7.5, color=_INK,
                    transform=above(2), clip_on=False)  # fmt: skip
    ax.text(0, -0.2, subtitle, ha="left", va="bottom", fontsize=7.5, color=_MUTED, transform=above(head + 24), clip_on=False,
            linespacing=1.3)  # fmt: skip
    ax.text(0, -0.2, title, ha="left", va="bottom", fontsize=10, color=_INK, transform=above(head + 30 + 11 * subtitle.count("\n")
            + 12), clip_on=False, fontweight="bold")  # fmt: skip
    ax.set_xlim(-0.05, col(len(tols) - 1, len(methods) - 1) + 1.05)
    ax.set_ylim(n + 0.25 + 0.6 * len(extra_rows) + (0.1 if extra_rows else 0), -0.25)
    ax.set_yticks(np.arange(n) + 0.5, row_labels or problems, fontsize=8)
    ax.tick_params(length=0, pad=4)
    ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    return col


def _short(method: str) -> str:
    """The method's label up to its first colon, for a column header."""
    return METHODS[method][0].split(":")[0]


def plot_overview(d: int, out: Path) -> list[Path]:
    """
    Every cached problem and tolerance at dimension d against every method, as grids with a row per
    problem and the methods side by side within each tolerance. ``overview-d<d>.png`` answers which
    methods hold the tolerance and which of those is cheapest: colour is the worst |error| /
    tolerance, the number the oracle evaluations, the outline the cheapest method that holds it.
    ``overview-d<d>-cost.png`` gives, in the same layout, the training and inference times (beside
    the oracle's own time per point), the pieces and the mean |error| / tolerance.
    """
    cases = [(problem, tol) for problem, p in PROBLEMS.items() for tol in p.tolerances(d)]
    cells = {(m, pr, t): c for m in METHODS for pr, t in cases if (c := _overview_cell(m, pr, d, t)) is not None}
    if not cells:
        return []
    methods = [m for m in METHODS if any(k[0] == m for k in cells)]
    problems = [p for p in PROBLEMS if any(k[1] == p for k in cells)]
    tols = PROBLEMS[problems[0]].tolerances(d)
    for pr, t in cases:  # the cheapest method that holds the tolerance on every seed; ties all marked
        held = [cells[m, pr, t] for m in methods if (m, pr, t) in cells and cells[m, pr, t]["passed"]]
        for c in held:
            c["best"] = c["evals"] == min(h["evals"] for h in held)
    ranks = "/".join(map(str, sorted(set().union(*(c["ranks"] for c in cells.values())))))
    where = f"{setup()}, {ranks} ranks, {len(SEEDS)} seeds"
    grid_w = len(tols) * (len(methods) + 0.7)  # data units across one panel
    paths = [out / f"overview-d{d}.png", out / f"overview-d{d}-cost.png"]

    fig, ax = plt.subplots(figsize=(0.56 * grid_w + 1.6, 0.4 * len(problems) + 1.0))
    summary = (("holds the tolerance", {}), ("cheapest that holds", {}))
    for (m, _, t), c in cells.items():
        held, best = summary[0][1].get((t, m), "0/0").split("/"), summary[1][1].get((t, m), "0")
        summary[0][1][t, m] = f"{int(held[0]) + c['passed']}/{int(held[1]) + 1}"
        summary[1][1][t, m] = str(int(best) + bool(c.get("best")))
    col = _overview_grid(ax, problems, tols, methods, cells, lambda c: _ratio_colour(c["max_ratio"]),
                         lambda c: f"{_compact(c['evals'], count=True)}\n{_compact(c['max_ratio'])}",
                         f"d = {d}: which methods hold the tolerance, and at what cost",
                         "Colour: the worst |error| / tolerance over the test set (the worst seed): blue holds, red exceeds.\n"
                         f"Top number: oracle evaluations (seed mean); bottom: the worst |error| / tolerance. {where}.",
                         summary, mark_best=True)  # fmt: skip
    for (m, pr, t), c in cells.items():
        if c.get("refused_percent", 0) > 0:
            ax.plot(col(tols.index(t), methods.index(m)) + 0.83, problems.index(pr) + 0.22, "o", ms=2.6, color=_ink_on(
                _ratio_colour(c["max_ratio"])), mew=0, zorder=4)  # fmt: skip
    handles = [matplotlib.patches.Patch(fc=c, label=n) for c, n in zip(_RATIO_COLOURS, _RATIO_NAMES, strict=True)]
    handles += [matplotlib.patches.Patch(fc="none", ec=_INK, lw=1.8, label="cheapest that holds"),
                matplotlib.patches.Patch(fc="#d9d8d3", hatch="///", ec="#8f8e89", lw=0, label="a seed over the evaluation budget"),
                matplotlib.patches.Patch(fc="none", ec="#b5b4ae", ls=":", label="not run"),
                matplotlib.lines.Line2D([], [], ls="", marker="o", ms=3, color=_INK, label="some queries refused")]  # fmt: skip
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=6, fontsize=7.5, frameon=False,
              title="worst |error| / tolerance", title_fontsize=8, alignment="left", handlelength=1.6, columnspacing=1.2)  # fmt: skip
    fig.savefig(paths[0], dpi=140, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    oracle = {pr: float(np.median([c["oracle_us"] for (_, p, _), c in cells.items() if p == pr and "oracle_us" in c] or [np.nan]))
              for pr in problems}  # fmt: skip
    panel_w, panel_h, label_w, head_h = 0.46 * grid_w, 0.3 * len(problems) + 0.1, 2.0, 2.35  # inches
    ncol = 2 if grid_w < 20 else 1  # wide grids (many methods) stacked, so the image stays readable without zooming
    fig = plt.figure(figsize=(ncol * (label_w + panel_w), 4 / ncol * (head_h + panel_h)))
    W, H = fig.get_size_inches()
    panels = (("train_s", "training time (s)", None), ("predict_us", "inference time per query (us)",
              [f"{p}  (oracle {_compact(oracle[p])})" for p in problems]), ("leaves", "pieces", None),
              ("mean_ratio", "mean |error| / tolerance", None))  # fmt: skip
    for k, (key, title, labels) in enumerate(panels):
        ax = fig.add_axes(((label_w + (k % ncol) * (label_w + panel_w)) / W, (4 // ncol - 1 - k // ncol) * (head_h + panel_h) / H,
                           panel_w / W, panel_h / H))  # fmt: skip
        if key == "mean_ratio":
            fill, note = (lambda c: _ratio_colour(c["mean_ratio"])), "Colour: the scale of the accuracy overview (blue below 1)."
        else:
            vals = [c[key] for c in cells.values() if "max_ratio" in c and c[key] > 0]
            norm = matplotlib.colors.LogNorm(min(vals), max(vals)) if vals else None
            fill = lambda c, key=key, norm=norm: _COST(norm(max(c[key], norm.vmin)) if norm is not None else 0.0)  # noqa: E731
            note = "Seed means; darker costs more (log scale)."
            if key == "predict_us":
                note += " Beside each problem: the oracle's time per point."
        _overview_grid(ax, problems, tols, methods, cells, fill, lambda c, key=key: _compact(c[key], count=key == "leaves"),
                       title, note, row_labels=labels)  # fmt: skip
    fig.text(0.02, 1 + 0.15 / H, f"d = {d}: times, pieces and mean error ({where})", fontsize=11, fontweight="bold",
             color=_INK, va="bottom")  # fmt: skip
    fig.savefig(paths[1], dpi=120, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return paths


def write_summary(d: int, methods: list[str], out: Path) -> Path:
    """``summary-d<d>.md``: every cached problem's seed-mean metrics per tolerance and method."""
    lines = [f"# Summary, d = {d} (seed means, except the worst seed's worst; worst = worst |error| / tolerance; "
             "refused queries are answered by the oracle)", "",
             "| problem | tol | method | ranks | pieces | evaluations | worst | worst seed | % within | % refused | mean error | "
             "train s | infer us/query |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for problem, p in PROBLEMS.items():
        for tol in p.tolerances(d):
            for method in methods:
                ms = [json.loads(str(np.load(cache_path(method, problem, d, tol, s))["metrics"])) for s in SEEDS
                      if cache_path(method, problem, d, tol, s).exists()]  # fmt: skip
                if not ms:
                    continue
                if all("failed" in m for m in ms):
                    lines.append(f"| {problem} | {tol:g} | {METHODS[method][0]} | {ms[0].get('ranks', 1)} | | over budget | | | | | | | |")
                    continue
                ms = [m for m in ms if "failed" not in m]
                a = {k: np.mean([m.get(k, 0.0) for m in ms]) for k in ("leaves", "evals", "max_ratio", "within_percent",
                                                                        "refused_percent", "mean_error", "train_s", "predict_us")}  # fmt: skip
                lines.append(f"| {problem} | {tol:g} | {METHODS[method][0]} | {ms[0].get('ranks', 1)} | {a['leaves']:.0f} | "
                             f"{a['evals']:,.0f} | {a['max_ratio']:.3g} | {max(m['max_ratio'] for m in ms):.3g} | "
                             f"{a['within_percent']:.2f} | {a['refused_percent']:.2f} | {a['mean_error']:.2e} | "
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
    parser.add_argument("--overview", action="store_true", help="only redraw overview-d<d>.png and overview-d<d>-cost.png from the cache")
    parser.add_argument("--replot", action="store_true", help="only redraw the problems' plots and the overview from the cache")
    parser.add_argument("--setup", default=None, help="with --overview or --replot: the cached setup to draw, e.g. cpu-8rank")
    a = parser.parse_args(argv)
    import src.structure.schemes  # noqa: F401  (registers the new schemes)
    from src.structure import (
        bench as module,  # run as __main__, the schemes registered in the imported module
    )

    METHODS.update(module.METHODS)

    if a.overview or a.replot:
        global SETUP
        SETUP = module.SETUP = a.setup
    out = OUT / setup()
    out.mkdir(parents=True, exist_ok=True)
    if a.overview:
        print(plot_overview(a.d, out))
        return
    methods = a.methods.split(",")
    for problem in a.problems.split(","):
        tols = [float(t) for t in a.tols.split(",")] if a.tols else PROBLEMS[problem].tolerances(a.d)
        if not a.replot:
            for tol in tols:
                for method in methods:
                    for seed in map(int, a.seeds.split(",")):
                        run(method, problem, a.d, tol, seed, force=a.force)
        if (a.plot or a.replot) and MPI.COMM_WORLD.Get_rank() == 0:  # after every problem, so results show as they arrive
            plot_problem(problem, a.d, tols, out)
            write_summary(a.d, list(METHODS), out)
            plot_overview(a.d, out)


def plot_problem(problem: str, d: int, tols, out: Path) -> None:
    """Every plot of one problem, with every method cached for it (not only those of the current run)."""
    methods = [m for m in METHODS if any(seed_runs(m, problem, d, t) for t in tols)]
    plot_cost(problem, d, methods, out)
    for tol in tols:  # the maps and views draw seed 1's predictions
        drawn = [m for m in methods if cache_path(m, problem, d, tol, SEEDS[0]).exists()]
        if drawn and d == 2:
            plot_maps(problem, drawn, tol, out)
        elif drawn:
            plot_views(problem, d, drawn, tol, out)


if __name__ == "__main__":
    main()
