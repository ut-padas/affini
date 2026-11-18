"""Draw the panels of the technical companion (docs/report/figures/*.svg) and write example.json.

Usage, from the repository root (``make report-figures``):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/make_figures.py

Every data panel comes from real training runs on the running example, through the public
``AdaptiveSurrogate`` API. The script records each level of the runs by wrapping stage methods
of the two plans (the wrappers change no result), rebuilds the trees from the records, and
stops unless what it rebuilt is exactly the trained model. Every number the report quotes
is written to example.json, which the report reads, so the text, the panels and the diagrams
cannot disagree.

The work-stealing panels show one recorded 4-rank run, stored in steal_run.json, because the
timing of steals differs from run to run (the tree does not, and the recording checks that).
Record it again with ``make report-steals``, which runs this script with ``--record-steals``
under ``mpiexec -n 4``.
"""

import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import src.art.plan as art_plan_module  # noqa: E402
from src import IS_CUPY_BACKEND, get_mpi_rank, to_host  # noqa: E402
from src.afinni.plan import AFINNIPlan  # noqa: E402
from src.art.constants import MAX_DEPTH_LIMIT  # noqa: E402
from src.art.geometry import plane_scores  # noqa: E402
from src.art.plan import ARTPlan  # noqa: E402
from src.art.split_search import SplitRows, hard_sse, soft_gates, soft_objective  # noqa: E402
from src.components.splitting import SPLIT_STRATEGIES, SplitData  # noqa: E402
from src.config import ARTSettings, TrainingConfig  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.kernels import source as kernel_source  # noqa: E402
from src.neighbors import build_box_tree  # noqa: E402
from src.numerics import grid_cell_coords, leading_eigvec  # noqa: E402
from src.numerics.segments import SEGMENT_CHUNK  # noqa: E402
from src.surrogate import AdaptiveSurrogate  # noqa: E402

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

OUT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- the example
LOWER, UPPER = np.array([-2.0, -1.0]), np.array([2.0, 3.0])
WIDTH = UPPER - LOWER
ALPHA = 1.0  # the absolute tolerance alpha_tol of both runs
SEED = 1
N0 = 600  # AFINNI's initial points
AFINNI_SETTINGS = {"atol": ALPHA, "rtol": None, "num_initial_clusters": 4, "min_pts_per_cluster": 24,
                   "max_levels": 12}  # fmt: skip
ART_MULTIPLIER = 20  # points per node = ART_MULTIPLIER * (d + 1)
ART_SETTINGS = {"atol": ALPHA, "rtol": None, "max_levels": 12}
KAPPA = 10  # Euclidean candidates at inference
# Leaves per bucket of the box tree drawn in the exact-flag panels: the example's few leaves need
# small buckets to show several levels (the library uses CPU_BUCKET).
FIG_BUCKET = 2
AUDIT_POINTS = 2000
FOREST_ROOTS = 4  # the grid-root run of the locational-code section
# The 4-rank run of the work-stealing panels: a tighter tolerance and small batches, so the
# tree is large enough, and its levels have enough batches, for steals to happen.
STEAL = {"atol": 0.01, "batch_size": 240, "steal_min_load": 240, "steal_size": 120, "max_levels": 20, "ranks": 6,
         "repeats": 3}

# ----------------------------------------------------------------------------- style
PALETTE = ["#c0392b", "#e1a800", "#2e7d32", "#1f5fa8", "#6d3b0f", "#ef7d1a", "#6a1b9a", "#d81b8a",
           "#00838f", "#7cb342", "#5c6bc0", "#a1887f", "#f06292", "#26a69a", "#8d6e63", "#3949ab"]  # fmt: skip
GREY, LIGHT, DARK, RED, BLUE, GOLD, GREEN = "#8a8a8a", "#d6d6d6", "#424242", "#c0392b", "#1f5fa8", "#e1a800", "#2e7d32"
PINK, CYAN = "#d81b8a", "#00a6c7"
RANKS = ["#1f5fa8", "#ef7d1a", "#2e7d32", "#6a1b9a", "#c0392b", "#00838f", "#e1a800", "#d81b8a"]
# Level sets of f, light where f is small: the background of every map of the domain.
BACKGROUND = LinearSegmentedColormap.from_list("level", ["#ffffff", "#e9edf3", "#c3cad6"])
# Signed error of a local model in units of the tolerance: white inside it, saturated beyond.
RESIDUAL_MID = "#f2f2f2"
RESIDUAL = LinearSegmentedColormap.from_list("residual", ["#1f5fa8", "#9ec1e6", RESIDUAL_MID, "#f0a79e", "#c0392b"])
# |f - f_hat| / alpha over the domain: blues within the tolerance, reds beyond it.
ERROR_LEVELS = [0, 0.25, 0.5, 0.75, 1.0, 2.0, 1e9]
ERROR_CMAP = ListedColormap(["#f4f8fc", "#d6e6f5", "#aecde9", "#7fb0db", "#f0a79e", "#c0392b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "svg.fonttype": "path",
    # A fixed salt and no date make the SVGs identical from run to run.
    "svg.hashsalt": "afinni",
    "axes.linewidth": 0.6, "axes.edgecolor": "#777777", "hatch.linewidth": 0.4,
})  # fmt: skip


def f_np(x: np.ndarray) -> np.ndarray:
    """The running example, Rosenbrock's valley with unit scale: (1 - x1)^2 + (x2 - x1^2)^2."""
    return (1.0 - x[:, 0]) ** 2 + (x[:, 1] - x[:, 0] ** 2) ** 2


def f_xp(x):
    """The same function as the oracle the library calls, (N, 1) on its array backend."""
    return (1.0 - x[:, 0:1]) ** 2 + (x[:, 1:2] - x[:, 0:1] ** 2) ** 2


def domain_grid(n: int) -> np.ndarray:
    g = (np.arange(n) + 0.5) / n
    return np.stack(np.meshgrid(LOWER[0] + g * WIDTH[0], LOWER[1] + g * WIDTH[1]), axis=-1).reshape(-1, 2)


def affine_fit(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    The affine least-squares model of ``AffineLSQModel`` on one cluster, restated: slope ``w``
    and value ``b`` at the centroid, with the library's ridge, and the residuals
    ``f - f_hat`` at the cluster's points.
    """
    c = X.mean(axis=0)
    dx, df = X - c, F - F.mean()
    S = dx.T @ dx
    w = np.linalg.solve(S + max(1e-8 * np.trace(S), 1e-12) * np.eye(X.shape[1]), dx.T @ df)
    return w, float(F.mean()), F - F.mean() - dx @ w


# ----------------------------------------------------------------------------- trees
@dataclass(eq=False)  # nodes are compared by identity: their fields hold arrays
class Node:
    """A node of either algorithm's forest, as the panels need it."""

    level: int
    parent: int | None
    X: np.ndarray  # the points it was fitted on
    leaf: bool = False
    children: list[int] = field(default_factory=list)
    centroid: np.ndarray | None = None
    code: int = 1  # ART's locational code
    polygon: np.ndarray | None = None  # ART's polytope, as its vertices
    plane: tuple[np.ndarray, float] | None = None  # ART's split, in normalized coordinates
    inherited: int = 0  # ART: how many of X came from the parent
    colour: str = GREY


def colour_nodes(nodes: list[Node]) -> None:
    """
    A colour per node, numbered within its level in queue order, so siblings always differ,
    and skipping its ancestors' colours, so the planes along any path (each drawn in the
    colour of the node that made it) differ too.
    """
    seen: dict[int, int] = {}
    previous: dict[int, str] = {}
    for n in nodes:
        k = seen.get(n.level, 0)
        seen[n.level] = k + 1
        taken, a = {previous.get(n.level)}, n.parent
        while a is not None:
            taken.add(nodes[a].colour)
            a = nodes[a].parent
        start = k + 3 * n.level
        n.colour = next(PALETTE[(start + t) % len(PALETTE)] for t in range(len(PALETTE))
                        if PALETTE[(start + t) % len(PALETTE)] not in taken)  # fmt: skip
        previous[n.level] = n.colour


def tree_layout(nodes: list[Node]) -> dict[int, tuple[float, float]]:
    """x from the in-order position of the leaves below a node, y from its level."""
    pos: dict[int, tuple[float, float]] = {}
    nxt = [0.0]

    def place(i: int) -> float:
        n = nodes[i]
        if not n.children:
            x = nxt[0]
            nxt[0] += 1.0
        else:
            x = float(np.mean([place(c) for c in n.children]))
        pos[i] = (x, -float(n.level))
        return x

    for i, n in enumerate(nodes):
        if n.parent is None:
            place(i)
            nxt[0] += 0.7  # a gap between the trees of a forest
    return pos


def place_text(ax, text: str, xy, offsets, paths: list[MPath], boxes: list[Bbox], pad=1.2, **kw) -> None:
    """
    Annotate ``xy`` at the first of ``offsets`` (points, with the text's alignment) whose box,
    grown by ``pad`` points, stays in the axes and meets none of ``paths`` and ``boxes``
    (display coordinates); the box joins ``boxes``. Call it after the axes limits are final:
    the test is made in display coordinates, with the figure at the 72 dpi the SVG canvas
    measures text at (``ax.figure.set_dpi(72)`` before computing ``paths`` and ``boxes``).
    Without a free offset the first one is used.
    """
    grow = pad * ax.figure.dpi / 72
    for dx, dy, ha, va in offsets:
        t = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **kw)
        e = t.get_window_extent()
        box = Bbox.from_extents(e.x0 - grow, e.y0 - grow, e.x1 + grow, e.y1 + grow)
        inside = ax.bbox.x0 <= box.x0 and box.x1 <= ax.bbox.x1 and ax.bbox.y0 <= box.y0 and box.y1 <= ax.bbox.y1
        if inside and not any(p.intersects_bbox(box, filled=False) for p in paths) and not any(box.overlaps(b) for b in boxes):
            boxes.append(box)
            return
        t.remove()
    print(f"warning: no free place for the label {text!r}")
    dx, dy, ha, va = offsets[0]
    ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **kw)


def draw_tree(ax, nodes: list[Node], upto: int | None = None, path: tuple[int, ...] = (), size=30, labels=False) -> None:
    """
    The forest down to level ``upto`` (all of it when None). Nodes of that level are in
    their own colour, ringed dark if accepted and red if rejected; earlier leaves keep
    their colour and earlier split nodes turn grey. ``path`` highlights a route.
    """
    pos = tree_layout(nodes)
    last = max(n.level for n in nodes) if upto is None else upto
    for i, n in enumerate(nodes):
        for c in n.children:
            if nodes[c].level <= last:
                on_path = i in path and c in path
                ax.plot(*zip(pos[i], pos[c], strict=True), color=RED if on_path else GREY, lw=1.6 if on_path else 0.6,
                        zorder=2 if on_path else 1)  # fmt: skip
    for i, n in enumerate(nodes):
        if n.level > last:
            continue
        current = upto is not None and n.level == upto
        if current:
            ax.scatter(*pos[i], s=size + 6, c=n.colour, edgecolors=DARK if n.leaf else RED, linewidths=1.3, zorder=3)
        else:
            ax.scatter(*pos[i], s=size, c=n.colour if n.leaf else LIGHT, edgecolors=RED if i in path else GREY,
                       linewidths=1.3 if i in path else 0.5, zorder=3)  # fmt: skip
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(-max(n.level for n in nodes) - (1.3 if labels else 0.6), 0.75)
    ax.axis("off")
    if labels:  # locational codes, clear of every node, edge and other code
        ax.figure.set_dpi(72)
        px = ax.transData.transform
        paths = [MPath(px([pos[i], pos[c]])) for i, n in enumerate(nodes) for c in n.children]
        r = 0.5 * np.sqrt(size) * ax.figure.dpi / 72
        boxes = [Bbox.from_extents(x - r, y - r, x + r, y + r) for x, y in px(list(pos.values()))]
        for i, n in sorted(enumerate(nodes), key=lambda t: (t[1].level, pos[t[0]][0])):
            # A split node's code goes above it, on the side away from the edge to its parent;
            # a leaf's below it, lower still where a neighbour's code is in the way.
            out = 0 if n.parent is None else (1 if pos[i][0] > pos[n.parent][0] else -1)
            side = {1: "left", -1: "right", 0: "center"}
            if n.leaf:
                offsets = [(0, -5.5, "center", "top"), (0, -14.5, "center", "top")]
            else:
                offsets = [(0, 5.5, "center", "bottom"), (out * 4, 4, side[out], "bottom"),
                           (out * 6, 0, side[out], "center"), (-out * 4, 4, side[-out], "bottom")]  # fmt: skip
            place_text(ax, format(n.code, "b"), pos[i], offsets, paths, boxes, fontsize=6, color=DARK, zorder=4)


# ----------------------------------------------------------------------------- AFINNI run
@dataclass
class Level:
    """One AFINNI level (a single batch in the example): its clusters and what became of them."""

    X: np.ndarray | None  # points of every queued cluster, cluster by cluster
    counts: np.ndarray | None
    centroids: np.ndarray
    good: np.ndarray  # accepted, after the deployed-model re-test
    split: SplitData | None = None  # the rejected clusters after augmentation (host copies)
    child_sizes: np.ndarray | None = None
    order: np.ndarray | None = None


def record_afinni() -> tuple[AdaptiveSurrogate, list[Level], np.ndarray]:
    """Train AFINNI on the running example, recording every level through the plan's stages."""
    levels: list[Level] = []
    pb, fit, split = AFINNIPlan.process_batch, AFINNIPlan._fit_and_accept, AFINNIPlan._split_rejected

    def process_batch(self, batch, **kw):
        if batch.index != 0:
            raise SystemExit("The example must fit one batch per level.")
        X, counts = to_host(batch.X).copy(), to_host(batch.counts).copy()
        out = pb(self, batch, **kw)
        lv = levels[-1]
        lv.X, lv.counts, lv.good = X, counts, to_host(lv.good).copy()
        return out

    def fit_and_accept(self, X, F, G, batch, components, evaluator, timer):
        out = fit(self, X, F, G, batch, components, evaluator, timer)
        # is_good is kept by reference: the deploy stage may still reject a cluster in place.
        levels.append(Level(X=None, counts=None, centroids=to_host(out[0].centroids).copy(), good=out[2]))
        return out

    def split_rejected(self, data, bad_idx, depth, splitter, next_level, timer):
        order, child_sizes = splitter.split(data)
        fields = ("X", "F", "G", "labels", "counts", "centroids", "f_c", "g_c")
        levels[-1].split = SplitData(*(None if getattr(data, k) is None else to_host(getattr(data, k)).copy()
                                       for k in fields))  # fmt: skip
        levels[-1].order, levels[-1].child_sizes = to_host(order), to_host(child_sizes)
        return split(self, data, bad_idx, depth, splitter, next_level, timer)

    AFINNIPlan.process_batch, AFINNIPlan._fit_and_accept, AFINNIPlan._split_rejected = (
        process_batch, fit_and_accept, split_rejected)  # fmt: skip
    try:
        X0 = np.random.default_rng(SEED).uniform(LOWER, UPPER, (N0, 2))
        s = AdaptiveSurrogate()
        s.train(X0, TrainingConfig(**AFINNI_SETTINGS, use_work_stealing=False, seed=SEED), Evaluator(f_fn=f_xp))
    finally:
        AFINNIPlan.process_batch, AFINNIPlan._fit_and_accept, AFINNIPlan._split_rejected = pb, fit, split
    leaves = np.vstack([lv.centroids[lv.good] for lv in levels])
    if not np.array_equal(leaves, to_host(s.X_src)):
        raise SystemExit("The recorded AFINNI levels do not reproduce the trained model's leaves.")
    return s, levels, X0


def afinni_forest(levels: list[Level]) -> list[Node]:
    """
    Rebuild AFINNI's forest, which the library does not store. Without work stealing the
    queue keeps its order: rejected clusters are split in ascending order and each appends
    its two children consecutively, so children 2j and 2j + 1 of the next level belong to
    the j-th rejected cluster. Each level's recorded points check the rebuilt queue.
    """
    nodes: list[Node] = []
    queue: list[int] = []
    for k, lv in enumerate(levels):
        starts = np.r_[0, np.cumsum(lv.counts)[:-1]]
        if k == 0:
            for s0, n in zip(starts, lv.counts, strict=True):
                nodes.append(Node(0, None, lv.X[s0 : s0 + n]))
                queue.append(len(nodes) - 1)
        for c, nid in enumerate(queue):
            if not np.array_equal(nodes[nid].X, lv.X[starts[c] : starts[c] + lv.counts[c]]):
                raise SystemExit("The rebuilt AFINNI forest does not match the recorded queue.")
            nodes[nid].leaf, nodes[nid].centroid = bool(lv.good[c]), lv.centroids[c]
        rejected = [nid for c, nid in enumerate(queue) if not lv.good[c]]
        queue = []
        if lv.split is None:
            continue
        Xs = lv.split.X[lv.order]
        bounds = np.r_[0, np.cumsum(lv.child_sizes)]
        for j, parent in enumerate(rejected):
            for side in (0, 1):
                a, b = bounds[2 * j + side], bounds[2 * j + side + 1]
                if b > a:  # an empty child is dropped
                    nodes.append(Node(k + 1, parent, Xs[a:b]))
                    nodes[parent].children.append(len(nodes) - 1)
                    queue.append(len(nodes) - 1)
    colour_nodes(nodes)
    return nodes


# ----------------------------------------------------------------------------- ART run
@dataclass
class SplitRecord:
    """What the split search saw and chose for one node (normalized coordinates)."""

    U: np.ndarray
    Y: np.ndarray  # outputs in units of the tolerance
    Phi: np.ndarray
    bank: np.ndarray  # (D, d) candidate normals
    bins: tuple[np.ndarray, float, float]  # plane and SSE of the bins search
    final: tuple[np.ndarray, float] | None = None  # the plane after the soft search


def art_config(**overrides) -> TrainingConfig:
    settings = {**ART_SETTINGS, "algorithm": "art", "domain": (LOWER, UPPER), "num_initial_clusters": 1,
                "use_work_stealing": False, "seed": SEED, "art": ARTSettings(sample_multiplier=ART_MULTIPLIER)}  # fmt: skip
    return TrainingConfig(**{**settings, **overrides})


def art_initial_points() -> np.ndarray:
    """ART samples its own points; the root starts from one node's worth of uniform ones."""
    return np.random.default_rng(SEED).uniform(LOWER, UPPER, (ART_MULTIPLIER * 3, 2))


def record_art() -> tuple[AdaptiveSurrogate, dict, dict, ARTPlan]:
    """Train ART on the running example, recording each node's points and each split search."""
    points: dict[int, tuple[np.ndarray, int]] = {}  # locational code -> (training points, inherited count)
    searches: dict[int, SplitRecord] = {}  # locational code -> its node's split search
    plans: list[ARTPlan] = []
    pb, cfd, split_fn = ARTPlan.process_batch, art_plan_module.ClusterFitData, ARTPlan._split
    bank_fn, bins_fn, soft_fn = ARTPlan._direction_bank, ARTPlan._bins_search, ARTPlan._soft_refine
    current: dict = {}

    def process_batch(self, batch, **kw):
        if not plans:
            plans.append(self)
        current["codes"] = to_host(batch.queue.locational_code[batch.cut]).tolist()
        current["inherited"] = to_host(batch.counts).tolist()
        return pb(self, batch, **kw)

    def fit_data(**kw):
        X, starts, counts = to_host(kw["X"]), to_host(kw["starts"]), to_host(kw["counts"])
        for code, inherited, s0, n in zip(current["codes"], current["inherited"], starts, counts, strict=True):
            points[code] = (X[s0 : s0 + n].copy(), int(inherited))
        return cfd(**kw)

    def split(self, idx, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, locational_code, *rest):
        # The j-th block of rows of the split search belongs to the j-th node of idx.
        current["split_codes"] = [int(p) for p in to_host(locational_code)[to_host(idx)]]
        return split_fn(self, idx, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, locational_code, *rest)

    def direction_bank(self, rows, lead, V, M, width, keys):
        bank = bank_fn(self, rows, lead, V, M, width, keys)
        current["bank"] = to_host(bank)
        return bank

    def bins_search(self, rows, bank):
        w, z, sse = bins_fn(self, rows, bank)
        U, Y, Phi, starts, counts = (to_host(a) for a in (rows.U, rows.Y, rows.Phi, rows.starts, rows.counts))
        for j, (code, s0, n) in enumerate(zip(current["split_codes"], starts, counts, strict=True)):
            sl = slice(s0, s0 + n)
            searches[code] = SplitRecord(U[sl].copy(), Y[sl].copy(), Phi[sl].copy(), current["bank"][j],
                                         (to_host(w)[j].copy(), float(to_host(z)[j]), float(to_host(sse)[j])))  # fmt: skip
        return w, z, sse

    def soft_refine(self, rows, w0, z0, sse0, progress):
        w, z = soft_fn(self, rows, w0, z0, sse0, progress)
        for j, code in enumerate(current["split_codes"]):
            searches[code].final = (to_host(w)[j].copy(), float(to_host(z)[j]))
        return w, z

    ARTPlan.process_batch, art_plan_module.ClusterFitData, ARTPlan._split = process_batch, fit_data, split
    ARTPlan._direction_bank, ARTPlan._bins_search, ARTPlan._soft_refine = direction_bank, bins_search, soft_refine
    try:
        s = AdaptiveSurrogate()
        s.train(art_initial_points(), art_config(), Evaluator(f_fn=f_xp))
    finally:
        ARTPlan.process_batch, art_plan_module.ClusterFitData, ARTPlan._split = pb, cfd, split_fn
        ARTPlan._direction_bank, ARTPlan._bins_search, ARTPlan._soft_refine = bank_fn, bins_fn, soft_fn
    return s, points, searches, plans[0]


def clip(polygon: np.ndarray, a: np.ndarray, z: float, keep_right: bool) -> np.ndarray:
    """Clip a convex polygon (x coordinates) by the plane a^T u = z of normalized coordinates u."""
    side = ((polygon - LOWER) / WIDTH) @ a - z
    if not keep_right:
        side = -side
    out = []
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        if side[i] >= 0:
            out.append(polygon[i])
        if (side[i] >= 0) != (side[j] >= 0):
            t = side[i] / (side[i] - side[j])
            out.append(polygon[i] + t * (polygon[j] - polygon[i]))
    return np.array(out)


def art_tree(s: AdaptiveSurrogate, points: dict) -> list[Node]:
    """ART's tree from the model's routing table, with every node's polytope and recorded points."""
    r = s.routing
    is_leaf, w, z, codes = (to_host(getattr(r, k)) for k in ("is_leaf", "normal", "offset", "node_locational_code"))
    order = np.argsort([int(p) for p in codes], kind="stable")  # breadth first: a parent before its children
    index = {int(codes[i]): k for k, i in enumerate(order)}
    box = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER, [LOWER[0], UPPER[1]]])
    nodes: list[Node] = []
    for i in order:
        code = int(codes[i])
        parent = index.get(code >> 1) if code > 1 else None
        polygon = box if parent is None else clip(nodes[parent].polygon, *nodes[parent].plane, keep_right=bool(code & 1))
        X, inherited = points[code]
        nodes.append(Node(code.bit_length() - 1, parent, X, bool(is_leaf[i]), code=code, polygon=polygon,
                          plane=None if is_leaf[i] else (w[i], float(z[i])), inherited=inherited,
                          centroid=X.mean(axis=0)))  # fmt: skip
        if parent is not None:
            nodes[parent].children.append(len(nodes) - 1)
    for n in nodes:  # the recorded points lie in the polytope rebuilt from the stored planes
        if n.parent is not None:
            p = nodes[n.parent]
            # The first right-hand point defines z, so it sits on the plane up to rounding.
            score = ((n.X - LOWER) / WIDTH) @ p.plane[0] - p.plane[1]
            if not np.all(score >= -1e-12 if n.code & 1 else score < 1e-12):
                raise SystemExit("A recorded ART node's points are not on its side of its parent's plane.")
    colour_nodes(nodes)
    return nodes


# ----------------------------------------------------------------------------- drawing helpers
def box_axes(ax, pad=0.04) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(LOWER[0] - pad * WIDTH[0], UPPER[0] + pad * WIDTH[0])
    ax.set_ylim(LOWER[1] - pad * WIDTH[1], UPPER[1] + pad * WIDTH[1])


def background(ax, bands=9) -> None:
    """f's level sets (log-spaced, since the valley is shallow), drawn beyond the domain for zoomed panels."""
    g = np.linspace(-0.6, 1.6, 110)  # coarse on purpose: this is most of every panel's file size
    G0, G1 = np.meshgrid(LOWER[0] + g * WIDTH[0], LOWER[1] + g * WIDTH[1])
    Z = f_np(np.c_[G0.ravel(), G1.ravel()]).reshape(G0.shape)
    levels = np.r_[0.0, np.geomspace(0.08, f_np(domain_grid(60)).max(), bands), Z.max() + 1]
    ax.contourf(G0, G1, Z, levels=levels, cmap=BACKGROUND, norm=BoundaryNorm(levels, BACKGROUND.N), zorder=0)
    ax.contour(G0, G1, Z, levels=levels[1:-1], colors="white", linewidths=0.4, zorder=0)


def zoom(ax, X: np.ndarray, pad=0.25) -> None:
    lo, hi = X.min(axis=0), X.max(axis=0)
    mid, half = (lo + hi) / 2, max(hi - lo) / 2 + pad
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def panel(name: str, draw, size=(2.3, 2.3), back=True, view: np.ndarray | None = None, pad=0.25) -> None:
    """One square map of the domain, or of the bounding box of ``view``; without the level sets, framed at the domain's edge."""
    fig, ax = plt.subplots(figsize=size)
    box_axes(ax, pad=0.04 if back else 0.0)
    if back:
        background(ax)
    draw(ax)
    if view is not None:
        zoom(ax, view, pad)
    save(fig, name)


def ellipse(ax, c, T, colour, lw=0.8, alpha=1.0, zorder=3) -> None:
    """The trust region (x - c)^T T (x - c) <= 1."""
    vals, vecs = np.linalg.eigh(T)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(c, *(2 / np.sqrt(vals)), angle=angle, fc="none", ec=colour, lw=lw, alpha=alpha, zorder=zorder))


def cut_line(ax, point: np.ndarray, normal: np.ndarray, colour=DARK, lw=1.0, ls="--", zorder=4) -> None:
    """The line through ``point`` with the given normal (x coordinates), long enough for any panel."""
    t = np.array([-10.0, 10.0])[:, None] * np.array([-normal[1], normal[0]]) / np.linalg.norm(normal)
    ax.plot(*(point + t).T, color=colour, lw=lw, ls=ls, zorder=zorder)


def arrow(ax, a, b, colour=DARK, lw=0.9, scale=7, zorder=5) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale, color=colour, lw=lw, shrinkA=0,
                                 shrinkB=0, zorder=zorder))  # fmt: skip


def cross(ax, xy, colour, size=46, lw=2.0, zorder=6) -> None:
    """A cluster centre: a bold cross with a white halo, so it stands out on any points beneath it."""
    xy = np.atleast_2d(xy)
    ax.scatter(*xy.T, s=size, c=colour, marker="x", lw=lw, zorder=zorder,
               path_effects=[patheffects.withStroke(linewidth=lw + 2.2, foreground="white")])  # fmt: skip


def residual_points(ax, X, e, size=4.0) -> None:
    """Points coloured by their signed error in units of the tolerance; those beyond it are ringed."""
    over = np.abs(e) > ALPHA
    ax.scatter(*X[~over].T, s=size, c=e[~over] / ALPHA, cmap=RESIDUAL, vmin=-2, vmax=2, lw=0, zorder=3)
    ax.scatter(*X[over].T, s=size + 3, c=e[over] / ALPHA, cmap=RESIDUAL, vmin=-2, vmax=2, edgecolors=DARK, lw=0.35,
               zorder=4)  # fmt: skip


def residual_key() -> None:
    """
    The colour key of every error map: f - f_hat in units of the tolerance, and the ring
    that marks a point beyond it. Narrow, so its text stays legible where it is printed
    small, beside a storyboard step.
    """
    fig, ax = plt.subplots(figsize=(2.5, 0.85))
    grad = np.linspace(-2, 2, 400)[None, :]
    ax.imshow(grad, extent=(-2, 2, 0, 0.3), cmap=RESIDUAL, vmin=-2, vmax=2, aspect="auto", zorder=1)
    for v in (-1, 1):  # the tolerance: between these marks a point is within it
        ax.plot([v, v], [-0.05, 0.35], color=DARK, lw=0.8, zorder=2)
    for v, t in ((-2, "−2"), (-1, "−1"), (0, "0"), (1, "+1"), (2, "+2")):
        ax.text(v, -0.09, t, ha="center", va="top", fontsize=6.5, color=DARK)
    ax.text(-2, 0.4, "model too high", ha="left", va="bottom", fontsize=6.5, color=BLUE)
    ax.text(2, 0.4, "model too low", ha="right", va="bottom", fontsize=6.5, color=RED)
    ax.text(0, -0.5, "$f - \\hat{f}$ in units of $\\alpha_{tol}$;", ha="right", va="top", fontsize=6.5, color=DARK)
    ax.scatter([0.22], [-0.66], s=14, c=[RESIDUAL(0.95)], edgecolors=DARK, lw=0.5, zorder=3)
    ax.text(0.36, -0.5, "ringed: beyond it", ha="left", va="top", fontsize=6.5, color=DARK)
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-0.95, 0.72)
    ax.axis("off")
    save(fig, "residual-key")


def hull_edges(ax, X: np.ndarray) -> None:
    """The convex hull of 2-D points, dashed (Andrew's monotone chain)."""
    P = X[np.lexsort((X[:, 1], X[:, 0]))]

    def chain(points):
        out = []
        for p in points:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    hull = np.array(chain(P) + chain(P[::-1]))
    ax.add_patch(Polygon(hull, closed=True, fc="none", ec=GREY, lw=0.7, ls="--", zorder=2))


# ----------------------------------------------------------------------------- AFINNI panels
def afinni_levels(nodes: list[Node], levels: list[Level]) -> dict:
    """Level k twice: the clusters with the forest, and the same clusters as errors of their models."""
    out = []
    for k, lv in enumerate(levels):
        fig, (ax, at) = plt.subplots(1, 2, figsize=(4.7, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
        box_axes(ax)
        background(ax)
        for n in nodes:
            if n.leaf and n.level < k:  # retired at an earlier level: faded
                ax.scatter(*n.X.T, s=2.5, c=n.colour, alpha=0.28, lw=0, zorder=2)
            elif n.level == k:
                ax.scatter(*n.X.T, s=3.5, c=n.colour, lw=0, zorder=3)
                cross(ax, n.centroid, DARK if n.leaf else RED)
        draw_tree(at, nodes, k)
        fig.tight_layout(pad=0.2, w_pad=0.4)
        save(fig, f"afinni-level-{k}")

        def errors(ax, k=k):
            for n in nodes:
                if n.level == k:
                    residual_points(ax, n.X, affine_fit(n.X, f_np(n.X))[2])
                    cross(ax, n.centroid, DARK if n.leaf else RED)

        if not lv.good.all():  # the report shows the errors of the levels that still reject clusters
            panel(f"afinni-error-{k}", errors)
        worst = [float(np.abs(affine_fit(n.X, f_np(n.X))[2]).max()) for n in nodes if n.level == k]
        out.append({"clusters": int(len(lv.counts)), "accepted": int(lv.good.sum()), "points": int(lv.counts.sum()),
                    "worst_error": round(max(worst), 2)})  # fmt: skip
    return {"levels": out, "n_levels": len(levels)}


def check_affine_fit(s: AdaptiveSurrogate, nodes: list[Node]) -> None:
    """The restated fit must be the library's: compare slope and centroid value on every leaf."""
    A, b, C = to_host(s.A_fits_src), to_host(s.b_fits_src), to_host(s.X_src)
    for k, n in enumerate(n for n in nodes if n.leaf):
        w, mean, _ = affine_fit(n.X, f_np(n.X))
        if not (np.allclose(w, A[k, 0], rtol=1e-8, atol=1e-10) and np.isclose(mean, b[k, 0] + A[k, 0] @ C[k], rtol=1e-8)):
            raise SystemExit("The restated affine fit differs from the library's leaf model.")


def child_error(X: np.ndarray, order: np.ndarray, child_counts: np.ndarray) -> float:
    """The larger of the two children's worst errors, each child fitted on its own points."""
    halves = (X[order[: child_counts[0]]], X[order[child_counts[0] :]])
    return max(float(np.abs(affine_fit(h, f_np(h))[2]).max()) for h in halves)


def afinni_split(levels: list[Level]) -> dict:
    """One rejected cluster: its error pattern, the plain and the Hessian-informed cut, the Hessian estimate."""
    hess = TrainingConfig(**AFINNI_SETTINGS).components().split_strategy
    plain = SPLIT_STRATEGIES.create("covariance")
    n_min, d = AFINNI_SETTINGS["min_pts_per_cluster"], 2

    def one_cluster(lv: Level, j: int) -> SplitData:
        rows = lv.split.labels == j
        return SplitData(X=lv.split.X[rows], F=lv.split.F[rows], G=None, labels=np.zeros(int(rows.sum()), dtype=np.int64),
                         counts=np.array([int(rows.sum())]), centroids=lv.split.centroids[j][None],
                         f_c=lv.split.f_c[j][None], g_c=None)  # fmt: skip

    def direction(strategy, data: SplitData):
        M, C = (to_host(a)[0] for a in strategy.metric(data))
        a = M @ to_host(leading_eigvec(C[None]))[0]  # the library's u_1, with its sign, so the halves match its cut
        order, child_counts = (to_host(v) for v in strategy.split(data))
        return a / np.linalg.norm(a), order, child_counts

    # Every rejected cluster under both cuts; the panels show the one where they differ most.
    table = []
    for k, lv in enumerate(levels):
        for j in range(0 if lv.split is None else len(lv.split.counts)):
            data = one_cluster(lv, j)
            e_h = child_error(data.X, *direction(hess, data)[1:])
            e_p = child_error(data.X, *direction(plain, data)[1:])
            table.append((e_p / e_h, k, j, e_p, e_h))
    def parent_error(t):
        rows = levels[t[1]].split.labels == t[2]
        return float(np.abs(affine_fit(levels[t[1]].split.X[rows], levels[t[1]].split.F[rows, 0])[2]).max())

    # ... among the clusters that are large and clearly beyond the tolerance, so the pattern shows.
    _, k, j, e_plain, e_hess = max(t for t in table if levels[t[1]].split.counts[t[2]] >= 2 * n_min
                                   and parent_error(t) > 2 * ALPHA)  # fmt: skip
    lv = levels[k]
    data = one_cluster(lv, j)
    X, F, c, fc = data.X, data.F[:, 0], data.centroids[0], float(data.f_c[0, 0])
    a_h, order_h, sizes_h = direction(hess, data)
    a_p, order_p, sizes_p = direction(plain, data)

    # The Hessian estimate, restated to draw its two point sets, and checked against the metric used.
    n_grad, n_cov = max(2 * d, n_min // 4), max(1, n_min // 3)
    near = np.argsort(np.linalg.norm(X - c, axis=1), kind="stable")[:n_grad]
    S = (X[near] - c).T @ (X[near] - c)
    g = np.linalg.solve(S + max(1e-8 * np.trace(S), 1e-12) * np.eye(d), (X[near] - c).T @ (F[near] - fc))
    taylor = np.abs(F - fc - (X - c) @ g)
    level_set = np.argsort(taylor, kind="stable")[:n_cov]
    Sigma = (X[level_set] - c).T @ (X[level_set] - c)
    lam, U = np.linalg.eigh(Sigma)
    if not np.allclose(U @ np.diag(1 / np.sqrt(np.maximum(np.abs(lam), 1e-12))) @ U.T, to_host(hess.metric(data)[0])[0],
                       rtol=1e-9, atol=1e-12):  # fmt: skip
        raise SystemExit("The restated Hessian estimate differs from the library's metric.")
    H_true = np.array([[2 + 12 * c[0] ** 2 - 4 * c[1], -4 * c[0]], [-4 * c[0], 2.0]])
    ev, evec = np.linalg.eigh(H_true)
    H_abs = evec @ np.diag(np.abs(ev)) @ evec.T

    def halves(ax, a, order, child_counts):
        s = (X - c) @ a
        tau = 0.5 * (np.sort(s)[child_counts[0] - 1] + np.sort(s)[child_counts[0]])
        ax.scatter(*X[order[: child_counts[0]]].T, s=5, c=PINK, lw=0, zorder=3)
        ax.scatter(*X[order[child_counts[0] :]].T, s=5, c=CYAN, lw=0, zorder=3)
        cut_line(ax, c + tau * a, a)
        arrow(ax, c, c + 0.55 * a)
        cross(ax, c, RED, size=60)

    def marked(ax, rows, colour):
        ax.scatter(*X.T, s=5, c="white", edgecolors=GREY, lw=0.3, zorder=2)
        ax.scatter(*X[rows].T, s=9, c=colour, lw=0, zorder=3)
        cross(ax, c, RED, size=60)

    inv, dx = np.linalg.inv(Sigma), X[level_set] - c
    shape = inv / np.max(np.einsum("ij,jk,ik->i", dx, inv, dx))

    def level_set_panel(ax):
        marked(ax, level_set, PINK)
        ellipse(ax, c, shape, PINK, lw=1.0)
        # The exact |H| at the centroid, as the ellipse of the same area, for comparison.
        ellipse(ax, c, H_abs * np.sqrt(np.linalg.det(shape) / np.linalg.det(H_abs)), DARK, lw=0.7)

    # One view for the five panels of this cluster: its points and the whole level-set ellipse.
    view = np.vstack([X, c + np.sqrt(np.diag(np.linalg.inv(shape))), c - np.sqrt(np.diag(np.linalg.inv(shape)))])
    panel("afinni-split-error", lambda ax: residual_points(ax, X, affine_fit(X, F)[2], size=7), view=view)
    panel("afinni-split-plain", lambda ax: halves(ax, a_p, order_p, sizes_p), view=view)
    panel("afinni-split-hessian", lambda ax: halves(ax, a_h, order_h, sizes_h), view=view)
    panel("afinni-hess-gradient", lambda ax: marked(ax, near, GREEN), view=view)
    panel("afinni-hess-levelset", level_set_panel, view=view)

    # Augmentation: the rejected cluster that received the most new points (originals come first).
    grown = [(int(lv_.split.counts[jj] - lv_.counts[~lv_.good][jj]), kk, jj) for kk, lv_ in enumerate(levels)
             if lv_.split is not None for jj in range(len(lv_.split.counts))]  # fmt: skip
    n_new, ka, ja = max(grown)
    Xa = levels[ka].split.X[levels[ka].split.labels == ja]
    na = len(Xa) - n_new

    def augmented(ax):
        hull_edges(ax, Xa[:na])
        ax.scatter(*Xa[:na].T, s=8, c=DARK, lw=0, zorder=3)
        ax.scatter(*Xa[na:].T, s=8, c=PINK, lw=0, zorder=4)

    panel("afinni-augment", augmented, view=Xa, pad=0.12)
    return {
        "split": {"level": k, "points": int(len(X)), "plain_error": round(e_plain, 2), "hessian_error": round(e_hess, 2),
                  "parent_error": round(float(np.abs(affine_fit(X, F)[2]).max()), 2),
                  "angle": round(float(np.degrees(np.arccos(min(1.0, abs(a_h @ a_p)))))), "n_grad": n_grad, "n_cov": n_cov,
                  "rejected": len(table), "hessian_wins": sum(t[0] > 1 for t in table)},
        "augment": {"level": ka, "before": int(na), "after": int(len(Xa)), "new": int(n_new)},
    }  # fmt: skip


def afinni_trust(s: AdaptiveSurrogate, nodes: list[Node]) -> dict:
    """One leaf's trust ellipse, every leaf's, and three queries at inference."""
    C, T = to_host(s.X_src), to_host(s.trust_matrices_src)
    leaves = [n for n in nodes if n.leaf]
    k = int(np.argmax([len(n.X) for n in leaves]))
    P = leaves[k].X
    dist = np.einsum("ij,jk,ik->i", P - C[k], T[k], P - C[k])
    if not np.isclose(dist.max(), 1.0, rtol=1e-9):
        raise SystemExit("The farthest point of a leaf is not on its trust ellipse.")
    vals, vecs = np.linalg.eigh(T[k])
    corners = C[k] + np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * np.abs(vecs / np.sqrt(vals)).sum(axis=1)

    def one(ax):
        ax.scatter(*P.T, s=5, c=leaves[k].colour, lw=0, zorder=3)
        ax.scatter(*P[np.argmax(dist)], s=26, facecolors="none", edgecolors=RED, lw=1.1, zorder=5)
        ellipse(ax, C[k], T[k], DARK, lw=1.0)
        cross(ax, C[k], DARK, size=60)  # dark: an accepted centre

    def every(ax):
        for kk, n in enumerate(leaves):
            ellipse(ax, C[kk], T[kk], n.colour, lw=0.8)
            cross(ax, C[kk], n.colour, size=24, lw=1.3, zorder=4)

    panel("afinni-trust-one", one, view=corners, pad=0.1)
    panel("afinni-trust-all", every)

    # Inference with kappa candidates and one leaf. Three queries found on a grid, one per case:
    # A, the Euclidean-nearest centre is not the leaf the trust-region distance picks;
    # B, the query lies in several ellipses; C, it lies in none and is flagged.
    grid = domain_grid(60)
    E = np.linalg.norm(grid[:, None] - C[None], axis=2)
    cand = np.argsort(E, axis=1, kind="stable")[:, :KAPPA]
    D = np.einsum("qkd,qkde,qke->qk", grid[:, None] - C[cand], T[cand], grid[:, None] - C[cand])
    rows = np.arange(len(grid))
    nu = cand[rows, np.argmin(D, axis=1)]
    dmin, inside = D.min(axis=1), (D <= 1).sum(axis=1)
    scores = {
        "A": np.where((nu != cand[:, 0]) & (dmin < 0.6), E[rows, nu] - E[rows, cand[:, 0]], -np.inf),
        "B": np.where((inside >= 2) & (nu == cand[:, 0]), -dmin, -np.inf),
        "C": np.where(dmin > 1, -np.abs(dmin - 2.0), -np.inf),
    }
    choice: dict[str, int] = {}
    for name, score in scores.items():  # each query at least a quarter of the domain from the others
        for i in choice.values():
            score = np.where(np.linalg.norm((grid - grid[i]) / WIDTH, axis=1) > 0.4, score, -np.inf)
        # as far inside the domain's edges as a query of this kind exists, so its letter has room
        for margin in (0.1, 0.05, 0.025, 0.0):
            inner = np.where(np.all((grid - LOWER > margin * WIDTH) & (UPPER - grid > margin * WIDTH), axis=1), score, -np.inf)
            if np.isfinite(inner).any():
                break
        else:
            raise SystemExit(f"No grid query of kind {name} for the inference panel.")
        choice[name] = int(np.argmax(inner))
    found = {}
    for name, i in choice.items():
        pred, out = (np.ravel(to_host(a)) for a in s.predict(grid[i][None], k_nn=1, coarse_factor=KAPPA))
        if bool(out[0]) != bool(dmin[i] > 1):
            raise SystemExit("The restated inference disagrees with the library's outside flag.")
        found[name] = {"nu": int(nu[i]), "euclid": int(cand[i, 0]), "d_nu": round(float(dmin[i]), 2),
                       "inside": int(inside[i]), "outside": bool(out[0]), "pred": round(float(pred[0]), 2),
                       "f": round(float(f_np(grid[i][None])[0]), 2)}  # fmt: skip

    def inference(ax):
        for kk, n in enumerate(leaves):
            ellipse(ax, C[kk], T[kk], n.colour, lw=0.5, alpha=0.4)
        for name, i in choice.items():
            q, r = grid[i], found[name]
            ellipse(ax, C[r["nu"]], T[r["nu"]], leaves[r["nu"]].colour, lw=1.6)
            if r["euclid"] != r["nu"]:
                ax.plot(*zip(q, C[r["euclid"]], strict=True), color=GREY, lw=0.9, ls=":", zorder=5)
                cross(ax, C[r["euclid"]], GREY, size=30, lw=1.5)
            ax.plot(*zip(q, C[r["nu"]], strict=True), color=RED if r["outside"] else DARK, lw=0.9, zorder=5)
            cross(ax, C[r["nu"]], leaves[r["nu"]].colour, size=36, lw=1.7)
            ax.scatter(*q, s=20, c=RED if r["outside"] else "black", zorder=7)
        # The letters keep clear of the highlighted ellipses, the lines and the marks; a white
        # halo lifts them off the faded ellipses they may still cross.
        ax.figure.set_dpi(72)
        px = ax.transData.transform
        paths = [p.get_transform().transform_path(p.get_path()) for p in ax.patches if p.get_linewidth() > 1]
        paths += [MPath(px(line.get_xydata())) for line in ax.lines]
        marks = px(np.vstack([np.vstack([grid[i], C[found[k]["nu"]], C[found[k]["euclid"]]]) for k, i in choice.items()]))
        boxes = [Bbox.from_extents(x - 3, y - 3, x + 3, y + 3) for x, y in marks]
        around = [(dx * r, dy * r, {-1: "right", 0: "center", 1: "left"}[dx], {-1: "top", 0: "center", 1: "bottom"}[dy])
                  for r in (4, 8, 12, 16, 20) for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))]  # fmt: skip
        for name, i in choice.items():
            place_text(ax, name, grid[i], around, paths, boxes, fontsize=8, weight="bold", zorder=8,
                       path_effects=[withStroke(linewidth=2, foreground="white")])  # fmt: skip

    panel("afinni-inference", inference, size=(2.7, 2.7))
    return {"trust_points": int(len(P)), "queries": found}


def afinni_exact_flag(s: AdaptiveSurrogate, nodes: list[Node]) -> dict:
    """
    The exact outside flag: a query the candidates flag although a farther leaf contains it, the
    leaves' bounding boxes against their balls, and one query's walk down the box tree. Every
    containment is recomputed over all leaves and checked against the library's.
    """
    C, T = to_host(s.X_src), to_host(s.trust_matrices_src)
    leaves = [n for n in nodes if n.leaf]
    grid = domain_grid(80)
    D = np.einsum("qnd,nde,qne->qn", grid[:, None] - C[None], T, grid[:, None] - C[None])
    inside = D.min(axis=1) <= 1
    for kappa in range(1, KAPPA + 1):  # the fewest candidates with a query they flag although a leaf contains it
        out = np.ravel(to_host(s.predict(grid, k_nn=1, coarse_factor=kappa)[1])).astype(bool)
        if np.any(out & inside):
            break
    exact = np.ravel(to_host(s.predict(grid, k_nn=1, coarse_factor=kappa, exact_outside=True)[1])).astype(bool)
    if not np.array_equal(exact, ~inside):
        raise SystemExit("The library's exact outside flag disagrees with the search over every leaf.")
    E = np.linalg.norm(grid[:, None] - C[None], axis=2)
    rank = np.argsort(np.argsort(E, axis=1, kind="stable"), axis=1, kind="stable")  # Euclidean rank of every leaf
    nu = np.argmin(np.where(D <= 1, D, np.inf), axis=1)
    miss = np.flatnonzero(out & inside)
    clear = miss[D[miss, nu[miss]] <= 0.8]  # well inside its leaf, so the picture is unambiguous
    pool = clear if clear.size else miss
    q_i = int(pool[np.argmax(rank[pool, nu[pool]])])  # the containing leaf farthest down the Euclidean order
    q, leaf = grid[q_i], int(nu[q_i])
    cand = np.argsort(E[q_i], kind="stable")[:kappa]
    # Boxes and balls: half-width sqrt((T^-1)_kk) on axis k, radius 1 / sqrt(lambda_min).
    half = np.sqrt(np.einsum("nkk->nk", np.linalg.inv(T)))
    radius = 1.0 / np.sqrt(np.linalg.eigvalsh(T)[:, 0])
    in_box = np.all(np.abs(grid[:, None] - C[None]) <= half[None], axis=2)
    in_ball = np.linalg.norm(grid[:, None] - C[None], axis=2) <= radius[None]
    if np.any((D <= 1) & ~in_box) or np.any((D <= 1) & ~in_ball):
        raise SystemExit("A trust ellipse leaves its bounding box or its ball.")
    # The leaf whose ball most exceeds its box, among those whose ball the panel shows whole.
    fits = np.all((C - radius[:, None] >= LOWER - 0.04 * WIDTH) & (C + radius[:, None] <= UPPER + 0.04 * WIDTH), axis=1)
    ratio = np.where(fits if fits.any() else True, radius**2 / np.prod(half, axis=1), -np.inf)
    long_leaf = int(np.argmax(ratio))

    # The tree over the boxes, and one query's walk: a node is tested when its parent's box
    # contains the query, entered when its own box does.
    tree = build_box_tree(C, C - half, C + half, FIG_BUCKET)
    left, n_nodes = tree["left"], len(tree["left"])
    depth, tested, entered = np.zeros(n_nodes, int), np.zeros(n_nodes, bool), np.zeros(n_nodes, bool)
    tested[0] = True
    for nd in range(n_nodes):  # children are numbered after their parents
        if tested[nd]:
            entered[nd] = bool(np.all((q >= tree["node_lo"][nd]) & (q <= tree["node_hi"][nd])))
        if left[nd] >= 0:
            depth[left[nd]] = depth[left[nd] + 1] = depth[nd] + 1
            tested[left[nd]] = tested[left[nd] + 1] = entered[nd]
    buckets = (left < 0) & entered
    tested_leaves = sum(int(tree["end"][b] - tree["start"][b]) for b in np.flatnonzero(buckets))
    shown = list(range(1, depth.max() + 1))[:4]

    def miss_panel(ax):
        for kk, n in enumerate(leaves):
            ellipse(ax, C[kk], T[kk], n.colour, lw=0.5, alpha=0.35)
        ellipse(ax, C[leaf], T[leaf], leaves[leaf].colour, lw=1.6)
        for c in cand:
            ax.plot(*zip(q, C[c], strict=True), color=GREY, lw=0.9, ls=":", zorder=5)
            cross(ax, C[c], GREY, size=30, lw=1.5)
        ax.plot(*zip(q, C[leaf], strict=True), color=DARK, lw=0.9, zorder=5)
        cross(ax, C[leaf], leaves[leaf].colour, size=36, lw=1.7)
        ax.scatter(*q, s=34, facecolors="none", edgecolors=RED, lw=1.1, zorder=7)  # flagged by the candidates
        ax.scatter(*q, s=12, c="black", zorder=8)  # answered by the containing leaf

    def box_panel(ax):
        for kk, n in enumerate(leaves):
            ellipse(ax, C[kk], T[kk], n.colour, lw=0.5, alpha=0.5)
            ax.add_patch(Rectangle(C[kk] - half[kk], *(2 * half[kk]), fc="none", ec=n.colour, lw=0.4, alpha=0.5))
        c, h = C[long_leaf], half[long_leaf]
        ellipse(ax, c, T[long_leaf], leaves[long_leaf].colour, lw=1.6)
        ax.add_patch(Rectangle(c - h, *(2 * h), fc="none", ec=leaves[long_leaf].colour, lw=1.3, zorder=4))
        ax.add_patch(Ellipse(c, 2 * radius[long_leaf], 2 * radius[long_leaf], fc="none", ec=DARK, lw=0.9, ls="--",
                             zorder=4))  # fmt: skip

    def level_panel(k):
        def draw(ax):
            for kk, n in enumerate(leaves):
                ellipse(ax, C[kk], T[kk], n.colour, lw=0.4, alpha=0.3)
            for nd in np.flatnonzero(depth == k):
                lo, hi = tree["node_lo"][nd], tree["node_hi"][nd]
                colour, lw = (DARK, 1.2) if entered[nd] else ((RED, 0.8) if tested[nd] else (LIGHT, 0.6))
                ax.add_patch(Rectangle(lo, *(hi - lo), fc="none", ec=colour, lw=lw, zorder=4 if tested[nd] else 3))
            ax.scatter(*q, s=14, c="black", zorder=8)
        return draw

    panel("afinni-flag-miss", miss_panel, size=(2.5, 2.5))
    panel("afinni-flag-box", box_panel, size=(2.5, 2.5))
    for k in shown:
        panel(f"afinni-flag-tree-{k}", level_panel(k), size=(1.7, 1.7))
    level_counts = [{"depth": int(k), "nodes": int(np.sum(depth == k)), "tested": int(np.sum(tested & (depth == k))),
                     "entered": int(np.sum(entered & (depth == k)))} for k in shown]  # fmt: skip
    return {"exact_flag": {
        "kappa": int(kappa), "leaf_rank": int(rank[q_i, leaf]) + 1, "d_leaf": round(float(D[q_i, leaf]), 2),
        "grid": int(len(grid)), "missed": int(miss.size),
        "flagged_percent": round(100 * float(out.mean()), 1), "exact_percent": round(100 * float(exact.mean()), 1),
        "per_query": {"ellipses": round(float((D <= 1).sum(axis=1).mean()), 2),
                      "boxes": round(float(in_box.sum(axis=1).mean()), 2),
                      "balls": round(float(in_ball.sum(axis=1).mean()), 2)},
        "long_leaf": {"radius": round(float(radius[long_leaf]), 2), "half": [round(float(v), 2) for v in half[long_leaf]],
                      "box_area": round(float(4 * np.prod(half[long_leaf])), 2),
                      "ball_area": round(float(np.pi * radius[long_leaf] ** 2), 2),
                      "ellipse_area": round(float(np.pi / np.sqrt(np.linalg.det(T[long_leaf]))), 2)},
        "tree": {"bucket": FIG_BUCKET, "nodes": int(n_nodes), "leaves": int(len(C)), "levels": level_counts,
                 "tested": int(tested.sum()), "entered": int(entered.sum()), "tested_leaves": int(tested_leaves)},
    }}  # fmt: skip


def queue_strips(nodes: list[Node], levels: list[Level]) -> None:
    """Each level's queue as one strip: clusters stored back to back, children next to each other."""
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(levels) + 0.3))
    total = max(int(lv.counts.sum()) for lv in levels)
    left: dict[int, tuple[float, float]] = {}
    for k in range(len(levels)):
        x = 0.0
        for i, n in enumerate(nodes):
            if n.level != k:
                continue
            w = len(n.X)
            ax.add_patch(plt.Rectangle((x, -k - 0.32), w, 0.64, fc=n.colour, ec="white", lw=0.8,
                                       alpha=0.3 if n.leaf else 1.0))  # fmt: skip
            if not n.leaf:
                ax.add_patch(plt.Rectangle((x, -k - 0.32), w, 0.64, fc="none", ec="black", lw=1.4, zorder=3))
            left[i] = (x, w)
            if n.parent is not None:
                px, pw = left[n.parent]
                ax.plot([px + pw / 2, x + w / 2], [-k + 1 - 0.32, -k + 0.32], color=GREY, lw=0.45, zorder=1)
            x += w
        ax.text(-0.012 * total, -k, f"level {k}", ha="right", va="center", fontsize=6, color=DARK)
    ax.set_xlim(-0.09 * total, total * 1.01)
    ax.set_ylim(-len(levels) + 0.45, 0.55)
    ax.axis("off")
    save(fig, "queue-strips")


def queue_layout(nodes: list[Node], levels: list[Level], level: int = 2, batch_size: int = 130) -> dict:
    """
    One level's queue as it is stored, and the next level it produces: a cluster is a run
    of rows, a batch a run of whole clusters (cut by the library's ``AFINNIPlan.batch_end``
    for a small ``batch_size``), and the j-th rejected cluster's two children are entries
    2j and 2j + 1 of the next level, grown by the points augmentation added. The children's
    order is the rebuilt forest's, which ``afinni_forest`` checked against the recorded queue.
    """
    here = [i for i, n in enumerate(nodes) if n.level == level]
    counts = np.array([len(nodes[i].X) for i in here])
    cumsum = np.cumsum(counts)
    starts = np.r_[0, cumsum[:-1]]

    def batch_ends(cum: np.ndarray) -> list[int]:
        cutter = object.__new__(AFINNIPlan)  # batch_end needs only its grid offset
        ends, start, index = [], 0, 0
        while start < len(cum):
            end = cutter.batch_end(cum, start, index, batch_size)
            ends.append(end)
            start, index = end, index + 1
        return ends

    ends = batch_ends(cumsum)
    rejected = [i for i in here if not nodes[i].leaf]
    children = [c for i in rejected for c in nodes[i].children]
    child_sizes = np.array([len(nodes[c].X) for c in children])
    # A child's new points: rows its parent did not have before augmentation.
    parent_rows = {i: {tuple(r) for r in nodes[i].X} for i in rejected}
    child_new = {c: sum(tuple(r) not in parent_rows[nodes[c].parent] for r in nodes[c].X) for c in children}
    total = max(int(cumsum[-1]), int(child_sizes.sum()))

    fig, ax = plt.subplots(figsize=(6.8, 3.55))
    y_zoom, y_br, y_top, y_bot, h = 2.05, 1.5, 1.0, -0.55, 0.44

    def ink(colour: str) -> str:
        """Black or white, whichever reads on the fill."""
        r, g, b = matplotlib.colors.to_rgb(colour)
        return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"

    def block(x, y, w, n, label, fade=False, new=0):
        ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, fc=n.colour, ec="white", lw=0.8, alpha=0.3 if fade else 1.0,
                                   zorder=2))  # fmt: skip
        if new:  # the share of new points, hatched at the block's end
            ax.add_patch(plt.Rectangle((x + w - new, y - h / 2), new, h, fc="none", ec="white", lw=0, hatch="////",
                                       zorder=3))  # fmt: skip
        if not n.leaf and not fade:
            ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, fc="none", ec="black", lw=1.4, zorder=4))
        ax.text(x + w / 2, y + (0.07 if "\n" in label else 0), label, ha="center", va="center", fontsize=6.3, zorder=5,
                color=DARK if fade else ink(n.colour), linespacing=1.1)  # fmt: skip

    # 1. Level `level`: clusters back to back; each rejected one labelled with its rank j among the rejected.
    for k, (i, x0) in enumerate(zip(here, starts, strict=True)):
        j = rejected.index(i) if i in rejected else None
        block(x0, y_top, counts[k], nodes[i], str(k) if j is None else f"{k}\n", fade=nodes[i].leaf)
        if j is not None:
            ax.text(x0 + counts[k] / 2, y_top - 0.11, f"j={j}", ha="center", va="center", fontsize=5.0, zorder=5,
                    color=ink(nodes[i].colour))  # fmt: skip

    # 2. Its rows: one cluster magnified, a run of consecutive rows of X and F.
    k_show = int(np.argmax(counts))
    x0, w0 = int(starts[k_show]), int(counts[k_show])
    zh = 0.22
    rows = [x0, x0 + 1, x0 + 2, None, x0 + w0 - 2, x0 + w0 - 1]
    cell = 0.055 * total
    zx = x0 + w0 / 2 - len(rows) * cell / 2
    colour = nodes[here[k_show]].colour
    for r, row in enumerate(rows):
        cx = zx + r * cell
        if row is None:
            ax.text(cx + cell / 2, y_zoom, "…", ha="center", va="center", fontsize=7, color=DARK)
            continue
        ax.add_patch(plt.Rectangle((cx, y_zoom - zh / 2), cell, zh, fc=colour, ec="white", lw=0.8, zorder=3))
        ax.text(cx + cell / 2, y_zoom, str(row), ha="center", va="center", fontsize=5.4, zorder=4, color=ink(colour))
    for xa, xb in ((x0, zx), (x0 + w0, zx + len(rows) * cell)):
        ax.plot([xa, xb], [y_top + h / 2, y_zoom - zh / 2], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.text(zx + len(rows) * cell + 0.015 * total, y_zoom, f"cluster {k_show}: rows {x0} to {x0 + w0 - 1} of X and F,\n"
            f"starts[{k_show}] = {x0}, counts[{k_show}] = {w0}", ha="left", va="center", fontsize=5.8, color=DARK)  # fmt: skip

    # 3. Batches: whole clusters, each ending at the first cluster that crosses the next multiple of batch_size.
    for g in range(1, int(cumsum[-1] // batch_size) + 1):
        # Above the clusters only, ending in a mark on the row: the line must not hide a label.
        ax.plot([g * batch_size] * 2, [y_top + h / 2 + 0.02, y_br + 0.02], color=BLUE, lw=0.8, ls=":", zorder=5)
        ax.plot(g * batch_size, y_top + h / 2 + 0.03, marker="v", ms=3.2, color=BLUE, zorder=6)
    lo = 0
    for b, end in enumerate(ends):
        a, z = starts[lo], cumsum[end - 1]
        ax.annotate("", (a + 1, y_br), (z - 1, y_br),
                    arrowprops={"arrowstyle": "|-|", "color": BLUE, "lw": 0.8, "mutation_scale": 2.5})  # fmt: skip
        ax.text((a + z) / 2, y_br + 0.06, f"batch {b}", ha="center", va="bottom", fontsize=6.0, color=BLUE)
        lo = end
    note = cumsum[-1] + 0.02 * total  # the notes column, right of the level's queue
    ax.text(note, y_br, "batches run one after another;\none ends at the first cluster\npast a mark (every batch_size points)",
            ha="left", va="center", fontsize=5.6, color=BLUE, linespacing=1.15)  # fmt: skip

    # 4. The next level: the j-th rejected cluster's children at 2j and 2j + 1, joined to their parent.
    x = 0.0
    for j, i in enumerate(rejected):
        k = here.index(i)
        for side, c in enumerate(nodes[i].children):
            w = len(nodes[c].X)
            block(x, y_bot, w, nodes[c], f"{2 * j + side}", new=child_new[c])
            ax.plot([starts[k] + counts[k] * (0.3 + 0.4 * side), x + w / 2], [y_top - h / 2, y_bot + h / 2],
                    color=nodes[i].colour, lw=0.9, alpha=0.85, zorder=1)  # fmt: skip
            x += w
    # The next level's batches, cut the same way, below its row; then its tail, below them.
    child_cum = np.cumsum(child_sizes)
    child_ends = batch_ends(child_cum)
    y_br3 = y_bot - h / 2 - 0.2
    for g in range(1, int(child_cum[-1] // batch_size) + 1):
        ax.plot([g * batch_size] * 2, [y_br3 - 0.02, y_bot - h / 2 - 0.02], color=BLUE, lw=0.8, ls=":", zorder=5)
        ax.plot(g * batch_size, y_bot - h / 2 - 0.03, marker="^", ms=3.2, color=BLUE, zorder=6)
    lo = 0
    child_starts = np.r_[0, child_cum[:-1]]
    for b, end in enumerate(child_ends):
        a, z = child_starts[lo], child_cum[end - 1]
        ax.annotate("", (a + 1, y_br3), (z - 1, y_br3),
                    arrowprops={"arrowstyle": "|-|", "color": BLUE, "lw": 0.8, "mutation_scale": 2.5})  # fmt: skip
        ax.text((a + z) / 2, y_br3 - 0.06, f"batch {b}", ha="center", va="top", fontsize=6.0, color=BLUE)
        lo = end
    tail = child_sizes[-2:].sum()
    y_tail = y_br3 - 0.42
    ax.annotate("", (x - tail, y_tail), (x, y_tail),
                arrowprops={"arrowstyle": "|-|", "color": RED, "lw": 0.8, "mutation_scale": 2.5})  # fmt: skip
    ax.text(x - tail - 0.012 * total, y_tail, "tail: the last clusters to start;\nan idle rank may take them",
            ha="right", va="center", fontsize=5.8, color=RED, linespacing=1.15)  # fmt: skip
    # Right of the last split line, so the note crosses none.
    ax.text(x + 0.015 * total, (y_top + y_bot) / 2, "the $j$-th rejected cluster\nwrites its two children\nas entries $2j$ and $2j{+}1$",
            ha="left", va="center", fontsize=5.8, color=DARK, linespacing=1.15)  # fmt: skip
    ax.add_patch(plt.Rectangle((x + 0.015 * total, y_bot - 0.09), 0.035 * total, 0.18, fc=GREY, ec="white", lw=0,
                               hatch="////", zorder=3))  # fmt: skip
    ax.text(x + 0.06 * total, y_bot, "new points,\nadded before\nthe cut", ha="left", va="center", fontsize=5.6,
            color=DARK, linespacing=1.15)  # fmt: skip

    for y, name, n in ((y_top, f"level {level}", int(cumsum[-1])), (y_bot, f"level {level + 1}", int(child_sizes.sum()))):
        ax.text(-0.015 * total, y, f"{name}\n{n} points", ha="right", va="center", fontsize=6.3, color=DARK)
    accepted = [k for k, i in enumerate(here) if nodes[i].leaf]
    ax.text(cumsum[-1] + 0.015 * total, y_top, f"accepted: {', '.join(map(str, accepted))}\nto the leaves $\\mathcal{{S}}$",
            ha="left", va="center", fontsize=6.0, color=DARK)  # fmt: skip
    ax.set_xlim(-0.12 * total, total * 1.24)
    ax.set_ylim(y_bot - h / 2 - 0.75, y_zoom + zh / 2 + 0.08)
    ax.axis("off")
    save(fig, "queue-layout")
    return {"layout": {"level": level, "clusters": len(here), "rejected": len(rejected), "points": int(cumsum[-1]),
                       "next_points": int(child_sizes.sum()), "new_points": int(sum(child_new.values())),
                       "batch_size": batch_size, "batches": len(ends), "next_batches": len(batch_ends(np.cumsum(child_sizes))),
                       "shown_cluster": k_show, "first_accepted": accepted[0],
                       "first_after": rejected.index(next(i for i in rejected if here.index(i) > accepted[0]))}}  # fmt: skip


# ----------------------------------------------------------------------------- ART panels
def art_levels(nodes: list[Node]) -> dict:
    """Depth k: the polytopes of that depth with their points, and the tree down to it."""
    depth = max(n.level for n in nodes)
    out = []
    for k in range(depth + 1):
        fig, (ax, at) = plt.subplots(1, 2, figsize=(4.7, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
        box_axes(ax)
        background(ax)
        for n in nodes:
            if n.leaf and n.level < k:
                ax.add_patch(Polygon(n.polygon, closed=True, fc=n.colour, ec="white", lw=0.6, alpha=0.3, zorder=1))
            elif n.level == k:
                ax.add_patch(Polygon(n.polygon, closed=True, fc=n.colour, ec="none", alpha=0.42, zorder=1))
                ax.add_patch(Polygon(n.polygon, closed=True, fc="none", ec=DARK if n.leaf else RED, lw=0.9, zorder=3))
                ax.scatter(*n.X[: n.inherited].T, s=2.6, c=DARK, lw=0, zorder=4)
                ax.scatter(*n.X[n.inherited :].T, s=3.2, c="white", edgecolors=DARK, lw=0.3, zorder=4)
        draw_tree(at, nodes, k)
        fig.tight_layout(pad=0.2, w_pad=0.4)
        save(fig, f"art-level-{k}")
        here = [n for n in nodes if n.level == k]
        out.append({"nodes": len(here), "accepted": sum(n.leaf for n in here),
                    "new_points": int(sum(len(n.X) - n.inherited for n in here))})  # fmt: skip

    def partition(ax):
        for n in nodes:
            if n.leaf:
                ax.add_patch(Polygon(n.polygon, closed=True, fc=n.colour, ec="white", lw=0.9, alpha=0.5, zorder=1))

    panel("art-partition", partition)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # the whole width for the codes of neighbouring leaves
    draw_tree(ax, nodes, labels=True, size=34)
    save(fig, "art-tree")
    return {"levels": out, "depth": depth}


def lineage_of(nodes: list[Node], n: Node) -> list[int]:
    """The node, then its ancestors up to the root, as indices."""
    out = [nodes.index(n)]
    while nodes[out[-1]].parent is not None:
        out.append(nodes[out[-1]].parent)
    return out


def domain_box(ax) -> None:
    ax.add_patch(plt.Rectangle(LOWER, *WIDTH, fc="none", ec=DARK, lw=1.2, zorder=5))


def polytope_node(nodes: list[Node]) -> Node:
    """The deep node whose polytope has the most corners: the one the region figures follow."""
    deep = [n for n in nodes if n.level >= 2 and len(n.X) > n.inherited]
    return max(deep, key=lambda n: (len(n.polygon), len(n.X) - n.inherited))


def ancestor_planes(ax, nodes: list[Node], lineage: list[int], polygon: np.ndarray, kept: bool = False) -> None:
    """
    The planes of ``lineage`` (root first, as indices; the last entry is the node itself), each
    numbered by its depth + 1 in the colour of the node it split, at the spot on the plane clearest of
    ``polygon`` inside the box; with ``kept``, an arrow from the plane into the side the node keeps.
    """
    centre = polygon.mean(axis=0)
    for j, (a, child) in enumerate(zip(lineage[:-1], lineage[1:], strict=True)):
        w, z = nodes[a].plane
        colour = nodes[a].colour  # a plane is drawn in the colour of the node it split
        normal = (w / WIDTH) / np.linalg.norm(w / WIDTH)
        u = (centre - LOWER) / WIDTH
        foot = LOWER + (u + (z - w @ u) * w / (w @ w)) * WIDTH
        cut_line(ax, foot, normal, colour=colour, lw=0.9, ls="-", zorder=3)
        along = np.array([-normal[1], normal[0]])
        spots = [foot + t * along for t in np.linspace(-6, 6, 241)]
        spots = [q for q in spots if np.all(q > LOWER + 0.06 * WIDTH) and np.all(q < UPPER - 0.06 * WIDTH)]
        spot = max(spots, key=lambda q: min(np.linalg.norm(polygon - q, axis=1).min(), 1.2))
        ax.text(*spot, str(j + 1), ha="center", va="center", fontsize=7, weight="bold", color=colour, zorder=6,
                bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.9})  # fmt: skip
        if kept:  # the left child (even code) keeps a^T u <= z, the right one a^T u >= z
            inward = normal if nodes[child].code % 2 else -normal
            arrow(ax, spot + 0.09 * WIDTH[0] * along, spot + 0.09 * WIDTH[0] * along + 0.16 * WIDTH[0] * inward,
                  colour=colour, lw=1.0)  # fmt: skip


def art_sampling(nodes: list[Node]) -> dict:
    """
    One deep node: its polytope, the domain box and the numbered ancestor planes that bound
    it, and its points. Then a hit-and-run chain in the widest deep region, chosen so that
    consecutive directions differ clearly and every step can be seen.
    """
    node = polytope_node(nodes)
    lineage = lineage_of(nodes, node)

    def polytope(ax):
        ax.add_patch(Polygon(node.polygon, closed=True, fc=node.colour, ec="none", alpha=0.45, zorder=2))
        ax.add_patch(Polygon(node.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2))
        domain_box(ax)
        ancestor_planes(ax, nodes, lineage[::-1], node.polygon)
        ax.scatter(*node.X[: node.inherited].T, s=7, c=DARK, lw=0, zorder=4)
        ax.scatter(*node.X[node.inherited :].T, s=7, c=PINK, lw=0, zorder=4)

    panel("art-polytope", polytope)

    # A chain as the sampler walks it (a direction uniform on the circle, then a uniform point
    # of the chord), drawn with its own random numbers: the library's are keyed by node and draw.
    # The widest deep region shows the chords best.
    def area(poly: np.ndarray) -> float:
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())

    walk_node = max((n for n in nodes if n.level >= 2 and n.inherited), key=lambda n: area(n.polygon))
    A, b = [[1, 0], [-1, 0], [0, 1], [0, -1]], [UPPER[0], -LOWER[0], UPPER[1], -LOWER[1]]
    for child in lineage_of(nodes, walk_node)[:-1]:  # one half-plane per ancestor, in x: normal . x <= offset
        w, z = nodes[nodes[child].parent].plane
        normal, offset = w / WIDTH, z + (w / WIDTH) @ LOWER
        sign = -1.0 if nodes[child].code & 1 else 1.0
        A.append(sign * normal)
        b.append(sign * offset)
    A, b = np.array(A), np.array(b)

    def chain(rng) -> list:
        x = walk_node.X[rng.integers(walk_node.inherited)]
        steps = []
        for _ in range(4):
            g = rng.normal(size=2)
            g /= np.linalg.norm(g)
            ratio = (b - A @ x) / (A @ g)
            t_min, t_max = ratio[A @ g < 0].max(), ratio[A @ g > 0].min()
            new = x + (t_min + (t_max - t_min) * rng.uniform()) * g
            steps.append((x + t_min * g, x + t_max * g, x, new, g))
            x = new
        return steps

    # Among the first 400 seeds, the chain whose consecutive directions differ most (the
    # smallest turn between two steps is largest), with steps long enough to see.
    diameter = np.ptp(walk_node.polygon, axis=0).max()

    def score(steps) -> float:
        turns = [np.degrees(np.arccos(min(1.0, abs(s1[4] @ s2[4])))) for s1, s2 in zip(steps, steps[1:], strict=False)]
        lengths = [np.linalg.norm(s[3] - s[2]) for s in steps]
        return min(turns) if min(lengths) > 0.12 * diameter else -1.0

    steps = max((chain(np.random.default_rng(SEED + k)) for k in range(400)), key=score)
    if score(steps) < 30:
        raise SystemExit("No hit-and-run chain for the picture turns clearly at every step.")

    def walk(ax):
        ax.add_patch(Polygon(walk_node.polygon, closed=True, fc=walk_node.colour, ec="none", alpha=0.3, zorder=2))
        ax.add_patch(Polygon(walk_node.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2))
        domain_box(ax)
        for j, (lo, hi, a, c, _) in enumerate(steps):
            last = j == len(steps) - 1
            ax.plot(*zip(lo, hi, strict=True), color=GREY, lw=0.8, ls=":", zorder=3)
            arrow(ax, a, c, colour=DARK, lw=0.9, scale=7)
            ax.scatter(*c, s=30 if last else 10, c=PINK if last else "white", edgecolors=DARK, lw=0.6, zorder=6)
            mid = (a + c) / 2
            ax.text(*mid, str(j + 1), fontsize=6.5, color=DARK, weight="bold", ha="center", va="center", zorder=7,
                    bbox={"fc": "white", "ec": "none", "pad": 0.4, "alpha": 0.85})  # fmt: skip
        ax.scatter(*steps[0][2], s=26, c=DARK, lw=0, zorder=6)

    panel("art-hit-and-run", walk, view=walk_node.polygon, pad=0.1)
    return {"sampling": {"code": format(node.code, "b"), "depth": node.level, "rows": len(lineage) - 1,
                         "inherited": node.inherited, "new": len(node.X) - node.inherited,
                         "walk_steps": len(steps), "walk_min_turn": round(score(steps))}}  # fmt: skip


def art_carry(nodes: list[Node]) -> dict:
    """
    The block of rows a node carries, depth by depth down to the polytope figure's node: one
    frame per node on the path, with the planes so far and the side each keeps. The rows are
    checked against the polygons: every corner of a node satisfies all of its rows.
    """
    node = polytope_node(nodes)
    path = lineage_of(nodes, node)[::-1]
    rows = []
    for parent, child in zip(path[:-1], path[1:], strict=True):
        w, z = nodes[parent].plane
        sign = -1.0 if nodes[child].code % 2 else 1.0  # right child: -a^T u <= -z
        rows.append(sign * np.r_[w, z])
    for depth, i in enumerate(path):
        U = (nodes[i].polygon - LOWER) / WIDTH
        for r in rows[:depth]:
            if np.any(U @ r[:-1] - r[-1] > 1e-9):
                raise SystemExit("A carried row does not hold on its node's polytope.")
    for depth, i in enumerate(path):
        n = nodes[i]

        def frame(ax, n=n, depth=depth):
            ax.add_patch(Polygon(n.polygon, closed=True, fc=n.colour, ec="none", alpha=0.45, zorder=2))
            ax.add_patch(Polygon(n.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2))
            domain_box(ax)
            ancestor_planes(ax, nodes, path[: depth + 1], n.polygon, kept=True)

        panel(f"art-carry-{depth}", frame, size=(1.6, 1.6))
    return {"carry": {"codes": [format(nodes[i].code, "b") for i in path],
                      "colours": [nodes[i].colour for i in path],
                      "plane_colours": [nodes[i].colour for i in path[:-1]],
                      # the nodes queued at each depth: the k of that level's (k, h, d + 1) block array
                      "level_nodes": [sum(n.level == t for n in nodes) for t in range(len(path))],
                      "rows": [[round(float(v), 2) for v in r] for r in rows]}}  # fmt: skip


def art_queue_blocks(nodes: list[Node]) -> dict:
    """
    ART's queue level by level, as the planes it carries: every node is its block of rows, one
    row per ancestor plane in the colour of the node that made it (hatched where the row is
    negated, for a right child). A rejected node (outlined) writes two children whose blocks
    are its own plus one row; an accepted one leaves for the leaves, and its block with it.
    """
    from matplotlib.patches import Rectangle

    levels = sorted({n.level for n in nodes})
    cell, gap, row_gap = 1.0, 1.1, 2.6
    width = 3 * cell  # d + 1 = 3 numbers per row
    pos: dict[int, tuple[float, float]] = {}  # node -> (x of its left edge, y of its top)
    y = 0.0
    shapes = []
    for lv in levels:
        here = [i for i, n in enumerate(nodes) if n.level == lv]
        for k, i in enumerate(here):
            pos[i] = (k * (width + gap), y)
        shapes.append([len(here), lv, sum(not nodes[i].leaf for i in here)])
        y -= max(lv, 0.6) * cell + row_gap
    fig_w = max(len([n for n in nodes if n.level == lv]) for lv in levels) * (width + gap)
    fig, ax = plt.subplots(figsize=(6.6, 0.33 * -y))
    for i, n in enumerate(nodes):
        x, yt = pos[i]
        path = lineage_of(nodes, n)[::-1]  # root first
        alpha = 0.35 if n.leaf else 1.0
        if n.level == 0:
            ax.add_patch(Rectangle((x, yt - 0.6 * cell), width, 0.6 * cell, fc="white", ec=GREY, lw=0.6, ls="--"))
            ax.text(x + width + 0.4, yt - 0.3 * cell, "no rows: the box only", ha="left", va="center", fontsize=5.5,
                    color=GREY)  # fmt: skip
        for j in range(n.level):  # row j: the plane of ancestor path[j], negated if the path turned right there
            colour = nodes[path[j]].colour
            right = bool(nodes[path[j + 1]].code & 1)
            ax.add_patch(Rectangle((x, yt - (j + 1) * cell), width, cell, fc=colour, ec="white", lw=0.5, alpha=alpha,
                                   hatch="////" if right else None))  # fmt: skip
        h = max(n.level, 0.6) * cell
        if not n.leaf:
            ax.add_patch(Rectangle((x, yt - h), width, h, fc="none", ec=DARK, lw=1.3, zorder=4))
            for c in n.children:
                cx, cy = pos[c]
                ax.plot([x + width / 2, cx + width / 2], [yt - h, cy], color=n.colour, lw=0.8, zorder=1)
        ax.text(x + width / 2, yt + 0.25 * cell, format(n.code, "b"), ha="center", va="bottom", fontsize=4.6,
                color=DARK if not n.leaf else GREY)  # fmt: skip
    for (K, lv, rejected), lvy in zip(shapes, sorted({pos[i][1] for i in pos}, reverse=True), strict=True):
        ax.text(-1.2, lvy - max(lv, 0.6) * cell / 2, f"level {lv}", ha="right", va="center", fontsize=6.5, color=DARK)
        ax.text(fig_w + 0.6, lvy - max(lv, 0.6) * cell / 2, f"({K}, {lv}, 3)   {K - rejected} accepted",
                ha="left", va="center", fontsize=6, color=DARK)  # fmt: skip
    ax.text(fig_w + 0.6, 0.9, "(K, h, d + 1)", ha="left", va="bottom", fontsize=6, color=DARK, weight="bold")
    ax.set_xlim(-6.5, fig_w + 11)
    ax.set_ylim(y + row_gap - 0.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "art-queue-blocks")
    return {"queue_blocks": {"shapes": shapes}}


def route_passes(s: AdaptiveSurrogate, nodes: list[Node], n_queries: int = 8) -> dict:
    """
    Routing as the library runs it, pass by pass, for a few uniform queries: each pass moves
    every query not yet at a leaf one level down. Stops unless the queries end in the
    library's leaves.
    """
    by_code = {n.code: n for n in nodes}
    rng = np.random.default_rng(SEED)
    q = LOWER + rng.random((n_queries, 2)) * WIDTH
    u = (q - LOWER) / WIDTH
    code = np.ones(n_queries, dtype=np.int64)
    passes = [code.copy()]
    while not all(by_code[int(c)].leaf for c in code):
        for i, c in enumerate(code):
            n = by_code[int(c)]
            if not n.leaf:
                w, z = n.plane
                code[i] = 2 * c + int(u[i] @ w - z >= 0)
        passes.append(code.copy())
    leaves = np.asarray(to_host(s.route_leaves(q)))
    if not np.array_equal(np.asarray(s.art_leaves["locational_code"])[leaves], code):
        raise SystemExit("The library routes the example queries to other leaves.")
    return {"passes": {"codes": [[format(int(c), "b") for c in row] for row in passes],
                       "leaf": [[bool(by_code[int(c)].leaf) for c in row] for row in passes]}}  # fmt: skip


def art_split(nodes: list[Node], searches: dict, plan: ARTPlan) -> dict:
    """One node's split search: the error pattern before and after, the candidates, the two objectives."""
    settings = plan.settings
    by_code = {n.code: n for n in nodes}

    def rows_of(rec: SplitRecord) -> SplitRows:
        return SplitRows(rec.U, rec.Y, rec.Phi, np.zeros(len(rec.U), dtype=np.int64), np.array([len(rec.U)]))

    def sse(rec: SplitRecord, w, z) -> float:
        return float(hard_sse(rows_of(rec), np.asarray(w)[None], np.array([z]), plan.min_side)[0][0])

    # One recorded search per node the plan tried to split, and one for every split node of the tree.
    split_codes = {n.code for n in nodes if n.plane is not None}
    if len(searches) != plan.counters["splits"] + plan.counters["split_failed"] or not split_codes <= searches.keys():
        raise SystemExit("The recorded split searches are not one per split node of the tree.")
    for code, rec in searches.items():  # the recorded planes are the ones stored in the tree
        plane = by_code[code].plane
        if plane is not None and not (np.allclose(plane[0], rec.final[0]) and np.isclose(plane[1], rec.final[1])):
            raise SystemExit("A recorded split search does not end at the plane stored in the tree.")
    gains = {code: 1 - sse(rec, *rec.final) / rec.bins[2] for code, rec in searches.items()
             if by_code[code].plane is not None}  # fmt: skip
    code = max(gains, key=lambda p: (round(gains[p], 6), -p))
    rec, node = searches[code], by_code[code]
    X = LOWER + rec.U * WIDTH
    F = f_np(X)
    w, z = rec.final
    w_b, z_b, sse_b = rec.bins
    right = rec.U @ w - z >= 0
    rows = rows_of(rec)
    score = rec.U @ w
    T = settings.soft_temperature * float(np.std(score))

    def x_normal(a):  # a normal in u, as a unit direction in x
        return (a / WIDTH) / np.linalg.norm(a / WIDTH)

    def x_point(a, offset):  # the centroid moved onto the plane a^T u = offset, in x
        u = (node.centroid - LOWER) / WIDTH
        return LOWER + (u + (offset - a @ u) * a / (a @ a)) * WIDTH

    def outline(ax):
        ax.add_patch(Polygon(node.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2))

    def before(ax):
        outline(ax)
        residual_points(ax, X, affine_fit(X, F)[2], size=9)

    # AFINNI's cut of the same points, for comparison: along its Hessian-informed direction
    # (the first direction of the bank, the red arrow), at the median score, by count.
    lead = rec.bank[0]
    ranked = np.argsort(rec.U @ lead, kind="stable")
    half = len(ranked) // 2
    afinni_low = np.zeros(len(X), dtype=bool)
    afinni_low[ranked[:half]] = True
    scores_sorted = (rec.U @ lead)[ranked]
    z_afinni = 0.5 * (scores_sorted[half - 1] + scores_sorted[half])

    def worst(mask) -> float:
        """The larger of the two children's worst errors, in units of the tolerance."""
        return max(float(np.abs(affine_fit(X[m], F[m])[2]).max()) / ALPHA for m in (mask, ~mask))

    def mse(mask) -> float:
        return sum(float(np.sum(affine_fit(X[m], F[m])[2] ** 2)) for m in (mask, ~mask)) / ALPHA**2 / len(X)

    def after(ax):
        outline(ax)
        for side in (right, ~right):
            residual_points(ax, X[side], affine_fit(X[side], F[side])[2], size=9)
        cut_line(ax, x_point(w, z), x_normal(w), colour=DARK, lw=1.2, ls="-")
        cut_line(ax, x_point(lead, z_afinni), x_normal(lead), colour=RED, lw=1.1, ls=(0, (3, 2)))

    def candidates(ax):
        outline(ax)
        ax.scatter(*X.T, s=6, c="white", edgecolors=GREY, lw=0.35, zorder=3)
        length = 0.3 * np.ptp(node.polygon, axis=0).max()
        for j, a in enumerate(rec.bank):
            arrow(ax, node.centroid, node.centroid + length * x_normal(a), colour=RED if j == 0 else BLUE,
                  lw=1.1 if j == 0 else 0.8)  # fmt: skip
        cut_line(ax, x_point(w_b, z_b), x_normal(w_b), colour=GOLD, lw=1.2, ls="--")
        cut_line(ax, x_point(w, z), x_normal(w), colour=DARK, lw=1.2, ls="-")

    def gates(ax):
        outline(ax)
        _, WR, _ = soft_gates(rows, np.asarray(w)[None], np.array([z]), np.array([T]), settings.soft_weight_floor)
        ax.scatter(*X.T, s=11, c=to_host(WR), cmap=LinearSegmentedColormap.from_list("gate", [PINK, "#f2f2f2", CYAN]),
                   vmin=0, vmax=1, edgecolors=DARK, lw=0.25, zorder=4)  # fmt: skip
        cut_line(ax, x_point(w, z), x_normal(w), colour=DARK, lw=1.2, ls="-")
        for sign in (-1, 1):
            cut_line(ax, x_point(w, z + sign * T), x_normal(w), colour=GREY, lw=0.6, ls=":")

    for name, draw in (("art-split-before", before), ("art-split-after", after), ("art-split-candidates", candidates),
                       ("art-split-gates", gates)):  # fmt: skip
        panel(name, draw, view=node.polygon, pad=0.12)

    # The bins search along the direction it chose: the node's rows sorted by score and cut into
    # n_bins bins of nearly equal count, at the positions ARTPlan._bins_search uses; every bin
    # boundary is a candidate cut, and each cut's error comes from the two fits it leaves.
    nb, m = settings.n_bins, plan.min_side
    s_b = to_host(plane_scores(rec.U, np.broadcast_to(w_b, rec.U.shape)))  # the library's scores, to the bit
    n_pts = len(s_b)
    pos = np.clip(np.round(np.arange(1, nb) * n_pts / nb).astype(np.int64), m, max(n_pts - m, m))
    pos = np.clip(pos, 1, max(n_pts - 1, 1))
    z_cuts = np.sort(s_b)[pos]
    cut_err = np.array([sse(rec, w_b, v) for v in z_cuts]) / n_pts
    chosen = int(np.argmin(np.abs(z_cuts - z_b)))
    if not (np.isclose(z_cuts[chosen], z_b) and np.isclose(cut_err[chosen], sse_b / n_pts)):
        raise SystemExit("The bins search's plane is not one of the recomputed bin boundaries.")
    rank = np.empty(n_pts, dtype=np.int64)
    rank[np.argsort(s_b, kind="stable")] = np.arange(n_pts)
    bin_of = np.searchsorted(pos, rank, side="right")

    def bins_points(ax):
        outline(ax)
        inside = Polygon(node.polygon, closed=True, transform=ax.transData)
        normal = x_normal(w_b)
        along = 10.0 * np.array([-normal[1], normal[0]])
        for j, v in enumerate(z_cuts):  # every bin boundary, clipped to the node
            q = x_point(w_b, v)
            line, = ax.plot(*np.array([q - along, q + along]).T, color=GOLD if j == chosen else GREY,
                            lw=1.3 if j == chosen else 0.45, ls="--" if j == chosen else "-", zorder=2)  # fmt: skip
            line.set_clip_path(inside)
        left = bin_of < chosen + 1  # the chosen cut's two sides, as the children C_-1 and C_1
        ax.scatter(*X[left].T, s=10, c=PINK, lw=0, zorder=3)
        ax.scatter(*X[~left].T, s=10, c=CYAN, lw=0, zorder=3)

    panel("art-bins-points", bins_points, view=node.polygon, pad=0.12)
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.bar(np.arange(1, nb), cut_err, color=[GOLD if j == chosen else LIGHT for j in range(nb - 1)], width=0.75)
    one_model = float(np.sum(affine_fit(X, F)[2] ** 2)) / ALPHA**2 / n_pts
    ax.axhline(one_model, color=DARK, lw=0.8, ls="--")
    ax.text(0.6, one_model * 0.97, "one model for the node", ha="left", va="top", fontsize=6.5, color=DARK)
    ax.set_xticks(np.arange(1, nb, 2))
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.set_xlabel("cut between bins $j$ and $j + 1$", fontsize=7, labelpad=1)
    ax.set_ylabel("mean squared error", fontsize=7, labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "art-bins-errors")
    bins_numbers = {"n_bins": nb, "per_bin": round(n_pts / nb, 1), "chosen_cut": chosen + 1,
                    "chosen_error": round(float(cut_err[chosen]), 3)}  # fmt: skip

    # Both objectives along the final normal, as functions of the offset: the hard one is a
    # staircase with a step at every point, the soft one a smooth curve an optimizer can descend.
    zs = np.linspace(*np.quantile(score, [0.04, 0.96]), 400)
    n = len(score)
    hard = np.array([sse(rec, w, v) for v in zs]) / n
    soft = np.array([float(to_host(soft_objective(rows, np.asarray(w)[None], np.array([v]), np.array([T]),
                                                  settings.soft_weight_floor)[0])[0]) for v in zs])  # fmt: skip
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.axvspan(z - T, z + T, color=GREY, alpha=0.14, lw=0)
    ax.plot(zs, hard, color=DARK, lw=0.9, drawstyle="steps-mid", label="hard: each child refitted")
    ax.plot(zs, soft, color=CYAN, lw=1.4, label="soft: gates of width $h$")
    ax.axvline(z, color=RED, lw=0.9, ls="--")
    ax.set_xlabel("offset $z$ along the chosen normal", fontsize=8, labelpad=1)
    ax.set_ylabel("mean squared error / $\\alpha_{tol}^2$", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7, length=2, pad=1)
    ax.set_xlim(zs[0] - 0.02 * np.ptp(zs), zs[-1] + 0.02 * np.ptp(zs))
    ax.set_ylim(0, 1.2 * max(hard.max(), soft.max()))
    # The legend starts right of the gate band, in the headroom above both curves.
    band_end = (z + T - ax.get_xlim()[0]) / np.diff(ax.get_xlim())[0]
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(band_end + 0.02, 1.0), handlelength=1.6,
              borderaxespad=0.2)  # fmt: skip
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "art-objective")
    return {"split": {"code": format(code, "b"), "depth": node.level, "points": n, "directions": int(len(rec.bank)),
                      "sse_parent": round(float(np.sum(affine_fit(X, F)[2] ** 2)) / ALPHA**2 / n, 2),
                      "sse_bins": round(sse_b / n, 3), "sse_final": round(sse(rec, w, z) / n, 3),
                      "gain_percent": round(100 * gains[code]), "improved": sum(g > 1e-12 for g in gains.values()),
                      "searched": len(gains), "left": int((~right).sum()), "right": int(right.sum()),
                      "n_bins": settings.n_bins, "soft_iters": settings.soft_iters,
                      "temperature": settings.soft_temperature,
                      "art_worst": round(worst(right), 2), "afinni_worst": round(worst(afinni_low), 2),
                      "art_mse": round(mse(right), 3), "afinni_mse": round(mse(afinni_low), 3),
                      "bins": bins_numbers}}  # fmt: skip


def locational_code_panels(nodes: list[Node]) -> dict:
    """
    Naming a node. One figure follows the route to the deepest leaf in the tree (top) and in
    the domain (bottom, one frame per depth, each showing the cut that adds that depth's bit).
    Three panels then set the locational code beside a Morton code: a Morton code decodes to
    its box, an ART code to a route whose shape needs the stored planes, and the levels a
    64-bit code holds as the dimension grows.
    """
    deepest = max((n for n in nodes if n.leaf), key=lambda n: (n.level, -n.code))
    route = [nodes.index(deepest)]
    while nodes[route[-1]].parent is not None:
        route.append(nodes[route[-1]].parent)
    route = route[::-1]
    leaves = [n for n in nodes if n.leaf]
    pos = tree_layout(nodes)
    mono = {"family": "DejaVu Sans Mono"}

    def code_text(ax, x, y, code: str, size=6.5, **kw) -> None:
        """
        A code with its last bit, the latest turn, in red; the root's lone 1 is the place-holder
        bit, not a turn, and stays dark. Monospace, so a red copy padded with spaces overlays it.
        """
        if len(code) == 1:
            ax.text(x, y, code, fontsize=size, color=DARK, weight="bold", fontdict=mono, **kw)
            return
        ax.text(x, y, code[:-1] + " ", fontsize=size, color=DARK, fontdict=mono, **kw)
        ax.text(x, y, " " * (len(code) - 1) + code[-1], fontsize=size, color=RED, weight="bold", fontdict=mono, **kw)

    # 1. The walk: the tree with the route, and one frame per depth below it.
    fig = plt.figure(figsize=(6.8, 3.7))
    tree_ax = fig.add_axes([0.02, 0.34, 0.96, 0.64])
    for i, n in enumerate(nodes):
        for c in n.children:
            on = i in route and c in route
            tree_ax.plot(*zip(pos[i], pos[c], strict=True), color=RED if on else LIGHT, lw=2.0 if on else 0.7,
                         zorder=2 if on else 1)  # fmt: skip
            if on:
                mx, my = (pos[i][0] + pos[c][0]) / 2, (pos[i][1] + pos[c][1]) / 2
                side = nodes[c].code & 1
                tree_ax.text(mx + (0.35 if side else -0.35), my, str(side), ha="center", va="center", fontsize=7,
                             color=RED, weight="bold", fontdict=mono)  # fmt: skip
    for i, n in enumerate(nodes):
        here = i in route
        tree_ax.scatter(*pos[i], s=34 if here else 12, c=(RED if n is deepest else BLUE) if here else LIGHT,
                        edgecolors=DARK if here else GREY, lw=0.6, zorder=3)  # fmt: skip
        if here:  # the label goes on the side away from the node's sibling, which is free
            sibling = next((j for j, m in enumerate(nodes) if m.code == (n.code ^ 1) and n.parent is not None
                            and m.parent == n.parent), None)  # fmt: skip
            left = sibling is not None and pos[sibling][0] > pos[i][0]
            code_text(tree_ax, pos[i][0] + (-0.55 if left else 0.55), pos[i][1], format(n.code, "b"),
                      ha="right" if left else "left", va="center")  # fmt: skip
    xs = [p[0] for p in pos.values()]
    tree_ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    tree_ax.set_ylim(-max(n.level for n in nodes) - 0.5, 0.6)
    tree_ax.axis("off")
    tree_ax.text(min(xs) - 0.6, 0.2, "0 = side $a^\\top u < z$\n1 = side $a^\\top u \\geq z$", fontsize=6.3,
                 color=DARK, va="top", linespacing=1.3)  # fmt: skip

    k = len(route)
    width = 0.96 / k
    for depth, i in enumerate(route):
        n = nodes[i]
        ax = fig.add_axes([0.02 + depth * width + 0.006, 0.06, width - 0.012, (width - 0.012) * 6.8 / 3.7])
        box_axes(ax, pad=0.0)
        for leaf in leaves:
            ax.add_patch(Polygon(leaf.polygon, closed=True, fc="none", ec=GREY, lw=0.25, alpha=0.5, zorder=1))
        if n.parent is not None:
            p = nodes[n.parent]
            ax.add_patch(Polygon(p.polygon, closed=True, fc=LIGHT, ec="none", alpha=0.8, zorder=1))
            w, z = p.plane
            normal = (w / WIDTH) / np.linalg.norm(w / WIDTH)
            u = (p.polygon.mean(axis=0) - LOWER) / WIDTH
            foot = LOWER + (u + (z - w @ u) * w / (w @ w)) * WIDTH  # the parent's centre moved onto its plane
            cut_line(ax, foot, normal, colour=DARK, lw=1.0, ls="-", zorder=4)
            reach = 0.16 * np.ptp(p.polygon, axis=0).max()
            for sgn, bit in ((-1, "0"), (1, "1")):
                ax.text(*(foot + sgn * reach * normal), bit, ha="center", va="center", fontsize=6.5, weight="bold",
                        color=RED if int(bit) == (n.code & 1) else GREY, fontdict=mono, zorder=6)  # fmt: skip
        ax.add_patch(Polygon(n.polygon, closed=True, fc=RED if n is deepest else BLUE, ec=DARK, lw=0.7, alpha=0.5,
                             zorder=2))  # fmt: skip
        ax.scatter(*deepest.polygon.mean(axis=0), s=6, c="black", zorder=7)
        code_text(ax, 0.5, -0.16, format(n.code, "b"), size=6.0, ha="center", va="top", transform=ax.transAxes)
    save(fig, "locational-code-walk")

    # 2. A Morton code decodes to its box.
    def morton(ax):
        ax.add_patch(plt.Rectangle(LOWER, *WIDTH, fc="white", ec=DARK, lw=0.8, zorder=1))
        half = WIDTH / 2
        ax.plot([LOWER[0] + half[0]] * 2, [LOWER[1], UPPER[1]], color=DARK, lw=0.8, zorder=2)
        ax.plot([LOWER[0], UPPER[0]], [LOWER[1] + half[1]] * 2, color=DARK, lw=0.8, zorder=2)
        for qy in (0, 1):
            for qx in (0, 1):
                corner = LOWER + np.array([qx, qy]) * half
                if (qy, qx) == (1, 0):  # the quadrant split again
                    for t in (0.5,):
                        ax.plot([corner[0] + t * half[0]] * 2, [corner[1], corner[1] + half[1]], color=GREY, lw=0.6, zorder=2)
                        ax.plot([corner[0], corner[0] + half[0]], [corner[1] + t * half[1]] * 2, color=GREY, lw=0.6, zorder=2)
                    for sy in (0, 1):
                        for sx in (0, 1):
                            c2 = corner + (np.array([sx, sy]) + 0.5) * half / 2
                            hit = (sy, sx) == (0, 1)
                            if hit:
                                ax.add_patch(plt.Rectangle(corner + np.array([sx, sy]) * half / 2, *(half / 2), fc=RED,
                                                           alpha=0.5, ec=DARK, lw=0.8, zorder=1))  # fmt: skip
                            ax.text(*c2, f"{sy}{sx}", ha="center", va="center", fontsize=6.5, fontdict=mono,
                                    color=DARK, weight="bold" if hit else "normal", zorder=3)  # fmt: skip
                else:
                    ax.text(*(corner + half / 2), f"{qy}{qx}", ha="center", va="center", fontsize=8, fontdict=mono,
                            color=DARK, zorder=3)  # fmt: skip

    panel("morton-code", morton, size=(1.9, 1.9), back=False)

    # 3. An ART code gives the turns; the stored planes give the shape.
    def art_leaf(ax):
        for leaf in leaves:
            ax.add_patch(Polygon(leaf.polygon, closed=True, fc="none", ec=LIGHT, lw=0.4, zorder=1))
        centre = deepest.polygon.mean(axis=0)
        for j, a in enumerate(route[:-1]):
            w, z = nodes[a].plane
            colour = PALETTE[(j + 3) % len(PALETTE)]
            cut_line(ax, LOWER + (w * z / (w @ w)) * WIDTH, w / WIDTH, colour=colour, lw=0.9, ls="-", zorder=3)
            # Number the cut by its depth, on its line, just outside the leaf.
            u = (centre - LOWER) / WIDTH
            foot = LOWER + (u + (z - w @ u) * w / (w @ w)) * WIDTH
            normal = (w / WIDTH) / np.linalg.norm(w / WIDTH)
            along = np.array([-normal[1], normal[0]])
            spot = foot + 0.55 * along + 0.12 * np.sign((foot - centre) @ normal) * normal
            ax.text(*spot, str(j + 1), ha="center", va="center", fontsize=6.5, weight="bold", color=colour, zorder=5,
                    bbox={"fc": "white", "ec": "none", "pad": 0.5, "alpha": 0.85})  # fmt: skip
        ax.add_patch(Polygon(deepest.polygon, closed=True, fc=RED, alpha=0.5, ec=DARK, lw=0.8, zorder=2))

    panel("morton-art", art_leaf, size=(1.9, 1.9), back=False)

    # 4. Levels that fit in a 64-bit code: d bits per level for Morton, one for the locational code.
    dims = np.arange(1, 41)
    fig, ax = plt.subplots(figsize=(2.3, 1.9))
    ax.plot(dims, 63 // dims, color=BLUE, lw=1.4, drawstyle="steps-mid", label="Morton: $d$ bits a level")
    ax.plot(dims, np.full_like(dims, MAX_DEPTH_LIMIT), color=RED, lw=1.4, label="locational: 1 bit a cut")
    ax.set_xlabel("dimension $d$", fontsize=6.5, labelpad=1)
    ax.set_ylabel("levels in 64 bits", fontsize=6.5, labelpad=1)
    ax.set_ylim(0, 70)
    ax.tick_params(labelsize=5.8, length=2, pad=1)
    ax.legend(fontsize=5.8, frameon=False, loc="center right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "code-capacity")
    return {"locational_code": {"codes": [format(nodes[i].code, "b") for i in route], "depth": deepest.level,
                                "leaf": format(deepest.code, "b"), "value": deepest.code,
                                "parent": format(deepest.code >> 1, "b"), "sibling": format(deepest.code ^ 1, "b"),
                                "children": [format(2 * deepest.code, "b"), format(2 * deepest.code + 1, "b")],
                                "max_depth": MAX_DEPTH_LIMIT, "morton_levels_d40": 63 // 40}}  # fmt: skip


def art_route(s: AdaptiveSurrogate, nodes: list[Node]) -> dict:
    """One query on its way down: the polytope it is in at each depth, and its path in the tree."""
    deepest = max((n for n in nodes if n.leaf), key=lambda n: (n.level, -n.code))
    q = deepest.polygon.mean(axis=0)
    path = [nodes.index(deepest)]
    while nodes[path[-1]].parent is not None:
        path.append(nodes[path[-1]].parent)
    path = path[::-1]
    leaf = int(to_host(s.route_leaves(q[None]))[0])
    if int(s.art_leaves["locational_code"][leaf]) != deepest.code:
        raise SystemExit("The library routes the example query to another leaf.")

    fig, (ax, at) = plt.subplots(1, 2, figsize=(4.7, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    box_axes(ax)
    background(ax)
    for n in nodes:
        if n.leaf:
            ax.add_patch(Polygon(n.polygon, closed=True, fc="none", ec=GREY, lw=0.4, zorder=1))
    for depth, i in enumerate(path):
        n = nodes[i]
        ax.add_patch(Polygon(n.polygon, closed=True, fc=deepest.colour if n.leaf else "none", alpha=0.55 if n.leaf else 1,
                             ec=RED, lw=0.5 + 0.25 * depth, zorder=2 + depth))  # fmt: skip
    ax.scatter(*q, s=20, c="black", zorder=20)
    draw_tree(at, nodes, path=tuple(path))
    fig.tight_layout(pad=0.2, w_pad=0.4)
    save(fig, "art-route")
    u = (q - LOWER) / WIDTH
    return {"route": {"depth": deepest.level, "code": format(deepest.code, "b"),
                      "sides": [int(u @ nodes[i].plane[0] - nodes[i].plane[1] >= 0) for i in path[:-1]]}}  # fmt: skip


# ----------------------------------------------------------------------------- both
def compare(afinni: AdaptiveSurrogate, art: AdaptiveSurrogate, afinni_nodes: list[Node], art_nodes: list[Node]) -> dict:
    """Both surrogates against f: along one line, and as error maps of the domain."""
    n = 240
    grid = domain_grid(n)
    G0, G1 = grid[:, 0].reshape(n, n), grid[:, 1].reshape(n, n)
    truth = f_np(grid)
    pa, oa = (np.ravel(to_host(v)) for v in afinni.predict(grid, k_nn=1, coarse_factor=KAPPA))
    oa_exact = np.ravel(to_host(afinni.predict(grid, k_nn=1, coarse_factor=KAPPA, exact_outside=True)[1]))
    pr, orr = (np.ravel(to_host(v)) for v in art.predict(grid))
    ea, er = np.abs(pa - truth) / ALPHA, np.abs(pr - truth) / ALPHA
    extent = (LOWER[0], UPPER[0], LOWER[1], UPPER[1])
    norm = BoundaryNorm(ERROR_LEVELS, ERROR_CMAP.N)

    def error_afinni(ax):
        ax.imshow(ea.reshape(n, n), origin="lower", extent=extent, cmap=ERROR_CMAP, norm=norm, zorder=1,
                  interpolation="nearest")  # fmt: skip
        outside = oa.reshape(n, n).astype(float)
        ax.contourf(G0, G1, outside, levels=[0.5, 1.5], colors="none", hatches=["//////"], zorder=2)
        ax.contour(G0, G1, outside, levels=[0.5], colors=DARK, linewidths=0.5, zorder=3)
        cross(ax, to_host(afinni.X_src), DARK, size=12, lw=1.0, zorder=4)

    def error_art(ax):
        ax.imshow(er.reshape(n, n), origin="lower", extent=extent, cmap=ERROR_CMAP, norm=norm, zorder=1,
                  interpolation="nearest")  # fmt: skip
        for nd in art_nodes:
            if nd.leaf:
                ax.add_patch(Polygon(nd.polygon, closed=True, fc="none", ec=DARK, lw=0.45, zorder=3))

    panel("error-afinni", error_afinni, back=False)
    panel("error-art", error_art, back=False)

    # The key under both maps: the error bands, then each mark drawn as the maps draw it.
    fig, ax = plt.subplots(figsize=(5.1, 0.66))
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
    ax.text(x0 + cells * cw / 2, 0.06, "$|f - \\hat{f}| \\,/\\, \\alpha_{tol}$", ha="center", va="top", fontsize=7,
            color=DARK)  # fmt: skip
    mx, tx = 3.95, 4.25
    cross(ax, (mx, 0.86), DARK, size=12, lw=1.0)
    ax.text(tx, 0.86, "AFINNI: a leaf's centre", va="center", fontsize=6.5, color=DARK)
    ax.add_patch(Rectangle((mx - 0.13, 0.42), 0.26, 0.2, fc="none", ec=DARK, lw=0.5, hatch="//////"))
    ax.text(tx, 0.52, "AFINNI: outside every trust region", va="center", fontsize=6.5, color=DARK)
    ax.add_patch(Polygon([(mx - 0.15, 0.08), (mx + 0.05, 0.1), (mx + 0.15, 0.26), (mx - 0.1, 0.3)], closed=True,
                         fc=ERROR_CMAP(0), ec=DARK, lw=0.45))  # fmt: skip
    ax.text(tx, 0.19, "ART: a leaf's region", va="center", fontsize=6.5, color=DARK)
    ax.set_xlim(0, 6.85)
    ax.set_ylim(-0.12, 1.0)
    ax.axis("off")
    save(fig, "error-map-key")

    # A line across the valley: f, each surrogate, and the band of width alpha around f.
    x2 = 2.0
    t = np.linspace(LOWER[0], UPPER[0], 600)
    line = np.c_[t, np.full_like(t, x2)]
    fl = f_np(line)
    la, lo_ = (np.ravel(to_host(v)) for v in afinni.predict(line, k_nn=1, coarse_factor=KAPPA))
    lr = np.ravel(to_host(art.predict(line)[0]))
    fig, axes = plt.subplots(2, 1, figsize=(4.0, 2.9), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax, pred, colour, name in ((axes[0], la, BLUE, "AFINNI"), (axes[1], lr, GREEN, "ART")):
        ax.fill_between(t, fl - ALPHA, fl + ALPHA, color=GREY, alpha=0.22, lw=0)
        ax.plot(t, fl, color=DARK, lw=0.8)
        # A leaf's model is a straight line: break the curve wherever its second difference is not zero.
        kinks = np.flatnonzero(np.abs(np.diff(pred, 2)) > 1e-6) + 1
        for piece in np.split(np.arange(len(t)), np.unique(np.r_[kinks, kinks + 1])):
            if len(piece) > 1:
                ax.plot(t[piece], pred[piece], color=colour, lw=1.3)
        ax.text(0.5, 0.86, name, transform=ax.transAxes, ha="center", fontsize=8.5, color=colour, weight="bold")
        ax.tick_params(labelsize=7, length=2, pad=1)
        ax.set_ylim(-1.8, fl.max() + 1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    flagged = lo_.astype(bool)
    if flagged.any():
        axes[0].scatter(t[flagged], np.full(flagged.sum(), -1.1), s=1.5, c=RED, lw=0, zorder=5)
    axes[1].set_xlabel(f"$x_1$ along the line $x_2 = {x2:g}$", fontsize=8, labelpad=1)
    save(fig, "slice")

    def summary(err, outside, model, nodes):
        stats, inside = model.profiling_stats, ~outside.astype(bool)
        return {"leaves": int(model.N_src), "evals": int(stats["total_f_evals"]),
                "generated": int(stats["num_generated_pts"]), "max_error": round(float(err.max()), 2),
                "within_percent": round(100 * float(np.mean(err <= 1)), 1),
                "outside_percent": round(100 * float(np.mean(outside)), 1),
                "within_inside_percent": round(100 * float(np.mean(err[inside] <= 1)), 1),
                "depth": max(nd.level for nd in nodes)}  # fmt: skip

    return {"afinni": summary(ea, oa, afinni, afinni_nodes) | {"outside_exact_percent": round(100 * float(np.mean(oa_exact)), 1)},
            "art": summary(er, orr, art, art_nodes),
            "slice_x2": x2, "grid": n, "flagged_on_line": int(flagged.sum()), "slice_points": int(t.size)}  # fmt: skip


# ----------------------------------------------------------------------------- work stealing
# The recorded runs: one root or one grid root per rank, each with and without stealing.
STEAL_RUNS = {"root_static": (1, False), "root_steal": (1, True), "grid_static": (STEAL["ranks"], False),
              "grid_steal": (STEAL["ranks"], True)}  # fmt: skip


def record_steals() -> None:
    """
    Under ``mpiexec -n 4``: train ART four times (a single root or one grid root per rank, each
    with and without work stealing) after one warm-up training, so that no recorded run pays
    for compiling the CPU kernels; check that stealing leaves each tree unchanged, and store
    the runs.
    """

    def train(roots: int, stealing: bool) -> AdaptiveSurrogate:
        s = AdaptiveSurrogate()
        s.train(art_initial_points(), art_config(
            atol=STEAL["atol"], max_levels=STEAL["max_levels"], batch_size=STEAL["batch_size"],
            num_initial_clusters=roots, use_work_stealing=stealing, profile_events=True,
            art=ARTSettings(sample_multiplier=ART_MULTIPLIER, steal_min_load=STEAL["steal_min_load"],
                            steal_size=STEAL["steal_size"], roots="grid" if roots > 1 else "box")),
                Evaluator(f_fn=f_xp))  # fmt: skip
        return s

    train(1, True)  # warm-up: compiles every kernel on every rank
    # Every configuration several times, interleaved; the median run is the one recorded.
    tries = {name: [] for name in STEAL_RUNS}
    for _ in range(STEAL["repeats"]):
        for name, spec in STEAL_RUNS.items():  # collective: every rank trains each
            tries[name].append(train(*spec))
    if get_mpi_rank() != 0:
        return
    if any(len(m.profiling_stats["per_rank"]) != STEAL["ranks"] for ms in tries.values() for m in ms):
        raise SystemExit(f"Run under mpiexec -n {STEAL['ranks']}.")
    if not all(m.converged for ms in tries.values() for m in ms):
        raise SystemExit("A recorded run stopped at max_levels; raise STEAL['max_levels'].")
    walls = {name: [steal_metrics([r["events"] for r in m.profiling_stats["per_rank"]])["wall"]
                    for m in ms] for name, ms in tries.items()}  # fmt: skip
    models = {name: ms[int(np.argsort(walls[name])[len(ms) // 2])] for name, ms in tries.items()}

    def keyed(model: AdaptiveSurrogate) -> tuple[dict, list]:
        """Every split plane by its node's (root, locational code), and the sorted leaves: the tree, whatever the row order."""
        a = {k: to_host(v) for k, v in model.routing.arrays().items()}
        planes = {(int(r), int(c)): (w.tobytes(), float(z)) for r, c, w, z in
                  zip(a["split_root"], a["split_locational_code"], a["normal"], a["offset"], strict=True)}  # fmt: skip
        return planes, sorted(zip(a["leaf_root"].tolist(), a["leaf_locational_code"].tolist(), strict=True))

    for kind in ("root", "grid"):
        trees_seen = [keyed(m) for name in (f"{kind}_steal", f"{kind}_static") for m in tries[name]]
        if any(t != trees_seen[0] for t in trees_seen):
            raise SystemExit(f"The {kind} trees differ between runs, with and without work stealing.")
    keep = ("kind", "t", "dur", "phase", "level", "victim", "thief", "n_pts", "n_clusters", "rank", "batch")
    kinds = ("run_start", "done", "steal_grant", "steal_received")
    runs, trees = {}, {}
    for name, m in models.items():
        stats = m.profiling_stats
        a = {k: to_host(v) for k, v in m.routing.arrays().items()}
        runs[name] = {
            "walls": [round(w, 4) for w in walls[name]],
            "leaves_per_rank": [int(r["num_final_clusters"]) for r in stats["per_rank"]],
            # Leaves are gathered in rank order, so the first leaves_per_rank[0] were built by rank 0.
            "leaves": [[int(r), int(c)] for r, c in zip(a["leaf_root"], a["leaf_locational_code"], strict=True)],
            "events": [[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k in keep}
                        for e in r["events"] if e["kind"] in kinds or e.get("phase") == "time_idle"]
                       for r in stats["per_rank"]],
        }  # fmt: skip
        trees[name.split("_")[0]] = {"bins": a["root_bins"].tolist(), "split": [
            [int(r), int(c), w.tolist(), float(z)]
            for r, c, w, z in zip(a["split_root"], a["split_locational_code"], a["normal"], a["offset"], strict=True)]}
    # The CPUs the ranks ran on: timings are only comparable on cores of one speed.
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    (OUT / "steal_run.json").write_text(json.dumps({"settings": STEAL, "cpus": cpus, "runs": runs, "trees": trees}) + "\n")
    print("recorded", {k: v["leaves_per_rank"] for k, v in runs.items()}, "; stealing leaves both trees unchanged")


def forest_region(planes: dict, bins: np.ndarray, r: int, code: int) -> np.ndarray:
    """
    The polygon (x coordinates) of node ``code`` of root ``r``: root r's grid cell, row-major
    over ``bins`` as ``Roots.root_of`` numbers them, cut by the planes ``planes[(r, code)]``
    of its ancestors from the root down.
    """
    lo_u, hi_u = grid_cell_coords(np.array([r]), bins)[0] / bins, (grid_cell_coords(np.array([r]), bins)[0] + 1) / bins
    lo, hi = LOWER + lo_u * WIDTH, LOWER + hi_u * WIDTH
    poly = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    for shift in range(code.bit_length() - 2, -1, -1):  # one bit per level
        poly = clip(poly, *planes[(r, code >> (shift + 1))], keep_right=bool((code >> shift) & 1))
    return poly


def art_forest_panel() -> dict:
    """
    ART with several roots: the running example trained with ``roots="grid"``. Every root
    cell grows its own tree numbered from 1, so the same locational code names a node in
    several trees; the panel marks one such code in two roots.
    """
    s = AdaptiveSurrogate()
    s.train(art_initial_points(), art_config(num_initial_clusters=FOREST_ROOTS,
                                             art=ARTSettings(sample_multiplier=ART_MULTIPLIER, roots="grid")),
            Evaluator(f_fn=f_xp))  # fmt: skip
    a = {k: to_host(v) for k, v in s.routing.arrays().items()}
    bins = np.asarray(a["root_bins"])
    planes = {(int(r), int(c)): (np.asarray(w), float(z)) for r, c, w, z in
              zip(a["split_root"], a["split_locational_code"], a["normal"], a["offset"], strict=True)}  # fmt: skip

    def root_cell(r: int) -> np.ndarray:
        return forest_region(planes, bins, r, 1)

    def region(r: int, code: int) -> np.ndarray:
        return forest_region(planes, bins, r, code)

    nodes = {(int(r), int(c)) for r, c in zip(a["leaf_root"], a["leaf_locational_code"], strict=True)} | set(planes)
    by_code: dict[int, list[int]] = {}
    for r, c in nodes:
        by_code.setdefault(c, []).append(r)
    shared = max((c for c, rs in by_code.items() if len(rs) >= 2), key=lambda c: (c.bit_length(), len(by_code[c]), -c))
    pair = sorted(by_code[shared])[:2]
    tints = [BLUE, GOLD, GREEN, PINK]

    def draw(ax):
        for r, c in zip(a["leaf_root"], a["leaf_locational_code"], strict=True):
            ax.add_patch(Polygon(region(int(r), int(c)), closed=True, fc=tints[int(r) % len(tints)], ec="white", lw=0.6,
                                 alpha=0.35, zorder=1))  # fmt: skip
        for r in range(int(np.prod(bins))):
            cell = root_cell(r)
            ax.add_patch(Polygon(cell, closed=True, fc="none", ec=DARK, lw=1.4, zorder=3))
            ax.text(*(cell[3] + np.array([0.04, -0.04]) * WIDTH), f"root {r}", ha="left", va="top", fontsize=6.2,
                    color=DARK, weight="bold", zorder=5, bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.8})  # fmt: skip
        for r in pair:
            poly = region(r, shared)
            ax.add_patch(Polygon(poly, closed=True, fc=RED, ec=DARK, lw=1.0, alpha=0.6, zorder=4))
            x, y = poly[:, 0], poly[:, 1]
            cross = x * np.roll(y, -1) - np.roll(x, -1) * y  # shoelace terms: the area centroid, inside a thin region
            centre = np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (3 * cross.sum())
            ax.text(*centre, f"({r}, {format(shared, 'b')})", ha="center", va="center", fontsize=6.3,
                    color="black", weight="bold", zorder=6, fontdict={"family": "DejaVu Sans Mono"},
                    bbox={"fc": "white", "ec": "none", "pad": 0.7, "alpha": 0.85})  # fmt: skip

    panel("art-forest", draw, back=False)
    return {"forest": {"roots": int(np.prod(bins)), "leaves": int(s.N_src), "code": format(shared, "b"),
                       "pair": pair, "trees_with_code": len(by_code[shared])}}  # fmt: skip


def steal_metrics(events: list) -> dict:
    """Wall time and, per rank, the time it worked: from its start to its end, minus its idle phases."""
    start = min(e["t"] for ev in events for e in ev if e["kind"] == "run_start")
    end = max(e["t"] for ev in events for e in ev if e["kind"] == "done")
    busy = []
    for ev in events:
        own_start = min(e["t"] for e in ev if e["kind"] == "run_start")
        own_end = max(e["t"] for e in ev if e["kind"] == "done")
        idle = sum(e["dur"] for e in ev if e.get("phase") == "time_idle")
        busy.append(own_end - own_start - idle)
    wall = end - start
    return {"wall": wall, "busy": busy, "idle_percent": 100 * (1 - sum(busy) / (len(events) * wall))}


def steal_panels() -> dict:
    """The recorded runs: who was working when, which rank built which leaf, and what stealing changed."""
    data = json.loads((OUT / "steal_run.json").read_text())
    runs, P = data["runs"], data["settings"]["ranks"]
    names = list(STEAL_RUNS)
    metrics = {n: steal_metrics(runs[n]["events"]) for n in names}
    titles = {"root_steal": "one root, stealing", "root_static": "one root, no stealing",
              "grid_steal": f"{P} roots, stealing", "grid_static": f"{P} roots, no stealing"}  # fmt: skip

    # The single-root run without stealing on its own time axis (it would squeeze the others);
    # the other three on one shared axis, so they compare directly.
    slow = max(names, key=lambda m: metrics[m]["wall"])
    shared_end = max(metrics[m]["wall"] for m in names if m != slow)
    fig, axes = plt.subplots(len(names), 1, figsize=(6.2, 0.16 * P * len(names) + 0.5), gridspec_kw={"hspace": 0.75})
    for ax, n in zip(axes, names, strict=True):
        end = metrics[n]["wall"] if n == slow else shared_end
        events = runs[n]["events"]
        t0 = min(e["t"] for ev in events for e in ev if e["kind"] == "run_start")
        for r, ev in enumerate(events):
            own_end = max(e["t"] for e in ev if e["kind"] == "done") - t0
            ax.add_patch(plt.Rectangle((0, r - 0.32), own_end, 0.64, fc=RANKS[r], ec="none", zorder=1))
            for e in ev:
                if e.get("phase") == "time_idle":
                    ax.add_patch(plt.Rectangle((e["t"] - t0, r - 0.32), e["dur"], 0.64, fc="#eeeeee", ec="none", zorder=2))
        for e in (e for ev in events for e in ev if e["kind"] == "steal_received"):
            a_, b_ = e["victim"], e["rank"]
            arrow(ax, (e["t"] - t0, a_ + (0.32 if b_ > a_ else -0.32)), (e["t"] - t0, b_ - (0.32 if b_ > a_ else -0.32)),
                  colour=DARK, lw=0.6, scale=4)  # fmt: skip
        ax.set_yticks(range(P), [f"rank {r}" for r in range(P)], fontsize=4.8)
        ax.set_ylim(P - 0.5, -0.6)
        ax.set_xlim(0, end)
        ax.set_title(f"{titles[n]}: {metrics[n]['wall']:.2f} s, {metrics[n]['idle_percent']:.0f}% idle",
                     fontsize=6.5, loc="left", pad=2, color=DARK)  # fmt: skip
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("time (s)", fontsize=5.5, labelpad=0)
    save(fig, "steal-timeline")

    trees = {}
    for kind, t in data["trees"].items():
        trees[kind] = ({(r, c): (np.array(w), z) for r, c, w, z in t["split"]}, np.array(t["bins"]))
    for n in names:
        planes, bins = trees[n.split("_")[0]]
        owner = np.repeat(np.arange(P), runs[n]["leaves_per_rank"])

        def owners(ax, n=n, planes=planes, bins=bins, owner=owner):
            for (r, code), k in zip(runs[n]["leaves"], owner, strict=True):
                ax.add_patch(Polygon(forest_region(planes, bins, r, code), closed=True, fc=RANKS[k], ec="white",
                                     lw=0.3, alpha=0.8, zorder=1))  # fmt: skip

        panel(f"steal-owners-{n.replace('_', '-')}", owners, back=False)

    base = metrics["root_static"]["wall"]
    # Formatted here, so the table shows 3.0, not 3: the numbers are quoted as recorded.
    table = {n: {"wall": f"{metrics[n]['wall']:.2f}", "speedup": f"{base / metrics[n]['wall']:.1f}",
                 "wall_range": f"{min(runs[n]['walls']):.2f}–{max(runs[n]['walls']):.2f}",
                 "idle_percent": round(metrics[n]["idle_percent"]), "leaves_per_rank": runs[n]["leaves_per_rank"],
                 "grants": sum(e["kind"] == "steal_received" for ev in runs[n]["events"] for e in ev)}
             for n in names}  # fmt: skip
    one = runs["root_steal"]
    return {"leaves": sum(one["leaves_per_rank"]), "leaves_per_rank": one["leaves_per_rank"],
            "steals": table["root_steal"]["grants"], "ranks": P, "atol": data["settings"]["atol"],
            "batch_size": data["settings"]["batch_size"],
            "depth": max(c.bit_length() - 1 for _, c in one["leaves"]), "runs": table,
            "grid_leaves": sum(runs["grid_steal"]["leaves_per_rank"]), "repeats": data["settings"]["repeats"],
            "cpus": len(data["cpus"])}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-steals", action="store_true", help="record steal_run.json (run under mpiexec -n 4)")
    args = parser.parse_args()
    if IS_CUPY_BACKEND:
        raise SystemExit("Run with SURROGATE_DEVICE=cpu: the panels and their numbers are the CPU results.")
    if args.record_steals:
        record_steals()
        return
    afinni, levels, X0 = record_afinni()
    forest = afinni_forest(levels)
    check_affine_fit(afinni, forest)
    art, art_points, searches, art_plan = record_art()
    tree = art_tree(art, art_points)

    numbers = {
        "alpha": ALPHA, "N0": N0, "seed": SEED, "kappa": KAPPA,
        "domain": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        # Library defaults the text quotes, read from TrainingConfig so a change shows up here.
        "defaults": {"q_reg_exponent": int(round(np.log10(TrainingConfig(atol=1.0).q_reg))),
                     "segment_chunk": SEGMENT_CHUNK, "n_directions": ARTSettings().n_directions,
                     "n_screened": ARTSettings().n_screened,
                     "walk_max_d": int(re.search(r"#define MAXD (\d+)", kernel_source("hit_and_run")).group(1))},
        "settings": {"afinni": {k: v for k, v in AFINNI_SETTINGS.items() if v is not None},
                     "art": {"sample_multiplier": ART_MULTIPLIER, "node_points": ART_MULTIPLIER * 3,
                             "max_levels": ART_SETTINGS["max_levels"]}},
        "colours": {"palette": PALETTE, "grey": GREY, "light": LIGHT, "dark": DARK, "red": RED, "blue": BLUE,
                    "gold": GOLD, "green": GREEN, "pink": PINK, "cyan": CYAN, "ranks": RANKS,
                    "residual_mid": RESIDUAL_MID, "error": [matplotlib.colors.to_hex(c) for c in ERROR_CMAP.colors]},
    }  # fmt: skip
    panel("example", lambda ax: None)
    numbers["afinni"] = afinni_levels(forest, levels) | afinni_split(levels) | afinni_trust(afinni, forest)
    numbers["afinni"] |= afinni_exact_flag(afinni, forest)
    queue_strips(forest, levels)
    residual_key()
    numbers["afinni"] |= queue_layout(forest, levels)
    numbers["art"] = art_levels(tree) | art_sampling(tree) | art_split(tree, searches, art_plan) | art_route(art, tree)
    numbers["art"] |= locational_code_panels(tree) | art_forest_panel()
    numbers["art"] |= art_carry(tree) | art_queue_blocks(tree) | route_passes(art, tree)
    numbers["compare"] = compare(afinni, art, forest, tree)
    # The library's own accuracy claim for the ART model: fresh uniform points, a one-sided bound.
    audit = art.audit(Evaluator(f_fn=f_xp), AUDIT_POINTS, seed=SEED)
    numbers["art"]["audit"] = {"points": AUDIT_POINTS, "violations": int(audit["n_violations"]),
                               "rate_percent": round(100 * audit["violation_rate"], 1),
                               "bound_percent": round(100 * audit["upper_bound"], 1),
                               "confidence_percent": round(100 * audit["confidence"])}  # fmt: skip
    numbers["steal"] = steal_panels()
    (OUT / "example.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({k: numbers[k] for k in ("compare",)}))


if __name__ == "__main__":
    main()
