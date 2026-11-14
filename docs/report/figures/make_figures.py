"""Draw the panels of the technical companion (docs/report/figures/*.svg) and write example.json.

Usage, from the repository root (``make report-figures``):

    AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/make_figures.py

Every data panel comes from real training runs on the running example, through the public
``AFINNISurrogate`` API. The script records each level of the runs by wrapping stage methods
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
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import src.art.plan as art_plan_module  # noqa: E402
from src import IS_CUPY_BACKEND, get_mpi_rank, to_host  # noqa: E402
from src.afinni.plan import AFINNIPlan  # noqa: E402
from src.art.plan import ARTPlan  # noqa: E402
from src.art.split_search import SplitRows, hard_sse, soft_gates, soft_objective  # noqa: E402
from src.components.splitting import SPLIT_STRATEGIES, SplitData  # noqa: E402
from src.config import ARTSettings, TrainingConfig  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402

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
AUDIT_POINTS = 2000
# The 4-rank run of the work-stealing panels: a tighter tolerance and small batches, so the
# tree is large enough, and its levels have enough batches, for steals to happen.
STEAL = {"atol": 0.05, "batch_size": 240, "steal_min_load": 240, "steal_size": 120, "max_levels": 20, "ranks": 4}

# ----------------------------------------------------------------------------- style
PALETTE = ["#c0392b", "#e1a800", "#2e7d32", "#1f5fa8", "#6d3b0f", "#ef7d1a", "#6a1b9a", "#d81b8a",
           "#00838f", "#7cb342", "#5c6bc0", "#a1887f", "#f06292", "#26a69a", "#8d6e63", "#3949ab"]  # fmt: skip
GREY, LIGHT, DARK, RED, BLUE, GOLD, GREEN = "#8a8a8a", "#d6d6d6", "#424242", "#c0392b", "#1f5fa8", "#e1a800", "#2e7d32"
PINK, CYAN = "#d81b8a", "#00a6c7"
RANKS = ["#1f5fa8", "#ef7d1a", "#2e7d32", "#6a1b9a"]
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
    path: int = 1  # ART's path code
    polygon: np.ndarray | None = None  # ART's polytope, as its vertices
    plane: tuple[np.ndarray, float] | None = None  # ART's split, in normalized coordinates
    inherited: int = 0  # ART: how many of X came from the parent
    colour: str = GREY


def colour_nodes(nodes: list[Node]) -> None:
    """A colour per node, numbered within its level in queue order, so siblings always differ."""
    seen: dict[int, int] = {}
    for n in nodes:
        k = seen.get(n.level, 0)
        seen[n.level] = k + 1
        n.colour = PALETTE[(k + 3 * n.level) % len(PALETTE)]


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
    if labels:  # path codes, clear of every node, edge and other code
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
            place_text(ax, format(n.path, "b"), pos[i], offsets, paths, boxes, fontsize=6, color=DARK, zorder=4)


# ----------------------------------------------------------------------------- AFINNI run
@dataclass
class Level:
    """One AFINNI level (a single batch in the example): its clusters and what became of them."""

    X: np.ndarray | None  # points of every queued cluster, cluster by cluster
    sizes: np.ndarray | None
    centroids: np.ndarray
    good: np.ndarray  # accepted, after the deployed-model re-test
    split: SplitData | None = None  # the rejected clusters after augmentation (host copies)
    child_sizes: np.ndarray | None = None
    order: np.ndarray | None = None


def record_afinni() -> tuple[AFINNISurrogate, list[Level], np.ndarray]:
    """Train AFINNI on the running example, recording every level through the plan's stages."""
    levels: list[Level] = []
    pb, fit, split = AFINNIPlan.process_batch, AFINNIPlan._fit_and_accept, AFINNIPlan._split_rejected

    def process_batch(self, batch, **kw):
        if batch.index != 0:
            raise SystemExit("The example must fit one batch per level.")
        X, sizes = to_host(batch.X).copy(), to_host(batch.sizes).copy()
        out = pb(self, batch, **kw)
        lv = levels[-1]
        lv.X, lv.sizes, lv.good = X, sizes, to_host(lv.good).copy()
        return out

    def fit_and_accept(self, X, F, G, batch, components, evaluator, timer):
        out = fit(self, X, F, G, batch, components, evaluator, timer)
        # is_good is kept by reference: the deploy stage may still reject a cluster in place.
        levels.append(Level(X=None, sizes=None, centroids=to_host(out[0].centroids).copy(), good=out[2]))
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
        s = AFINNISurrogate()
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
        starts = np.r_[0, np.cumsum(lv.sizes)[:-1]]
        if k == 0:
            for s0, n in zip(starts, lv.sizes, strict=True):
                nodes.append(Node(0, None, lv.X[s0 : s0 + n]))
                queue.append(len(nodes) - 1)
        for c, nid in enumerate(queue):
            if not np.array_equal(nodes[nid].X, lv.X[starts[c] : starts[c] + lv.sizes[c]]):
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


def record_art() -> tuple[AFINNISurrogate, dict, dict, ARTPlan]:
    """Train ART on the running example, recording each node's points and each split search."""
    points: dict[int, tuple[np.ndarray, int]] = {}  # path code -> (training points, inherited count)
    searches: dict[int, SplitRecord] = {}  # path code -> its node's split search
    plans: list[ARTPlan] = []
    pb, cfd, split_fn = ARTPlan.process_batch, art_plan_module.ClusterFitData, ARTPlan._split
    bank_fn, bins_fn, soft_fn = ARTPlan._direction_bank, ARTPlan._bins_search, ARTPlan._soft_refine
    current: dict = {}

    def process_batch(self, batch, **kw):
        if not plans:
            plans.append(self)
        current["paths"] = to_host(batch.queue.path_code[batch.cut]).tolist()
        current["sizes"] = to_host(batch.sizes).tolist()
        return pb(self, batch, **kw)

    def fit_data(**kw):
        X, starts, sizes = to_host(kw["X"]), to_host(kw["starts"]), to_host(kw["sizes"])
        for path, inherited, s0, n in zip(current["paths"], current["sizes"], starts, sizes, strict=True):
            points[path] = (X[s0 : s0 + n].copy(), int(inherited))
        return cfd(**kw)

    def split(self, idx, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, path_code, *rest):
        # The j-th block of rows of the split search belongs to the j-th node of idx.
        current["split_paths"] = [int(p) for p in to_host(path_code)[to_host(idx)]]
        return split_fn(self, idx, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, path_code, *rest)

    def direction_bank(self, rows, lead, V, M, width, keys):
        bank = bank_fn(self, rows, lead, V, M, width, keys)
        current["bank"] = to_host(bank)
        return bank

    def bins_search(self, rows, bank):
        w, z, sse = bins_fn(self, rows, bank)
        U, Y, Phi, starts, counts = (to_host(a) for a in (rows.U, rows.Y, rows.Phi, rows.starts, rows.counts))
        for j, (path, s0, n) in enumerate(zip(current["split_paths"], starts, counts, strict=True)):
            sl = slice(s0, s0 + n)
            searches[path] = SplitRecord(U[sl].copy(), Y[sl].copy(), Phi[sl].copy(), current["bank"][j],
                                         (to_host(w)[j].copy(), float(to_host(z)[j]), float(to_host(sse)[j])))  # fmt: skip
        return w, z, sse

    def soft_refine(self, rows, w0, z0, sse0, progress):
        w, z = soft_fn(self, rows, w0, z0, sse0, progress)
        for j, path in enumerate(current["split_paths"]):
            searches[path].final = (to_host(w)[j].copy(), float(to_host(z)[j]))
        return w, z

    ARTPlan.process_batch, art_plan_module.ClusterFitData, ARTPlan._split = process_batch, fit_data, split
    ARTPlan._direction_bank, ARTPlan._bins_search, ARTPlan._soft_refine = direction_bank, bins_search, soft_refine
    try:
        s = AFINNISurrogate()
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


def art_tree(s: AFINNISurrogate, points: dict) -> list[Node]:
    """ART's tree from the model's routing table, with every node's polytope and recorded points."""
    r = s.routing
    is_leaf, w, z, paths = (to_host(getattr(r, k)) for k in ("is_leaf", "normal", "offset", "node_path"))
    order = np.argsort([int(p) for p in paths], kind="stable")  # breadth first: a parent before its children
    index = {int(paths[i]): k for k, i in enumerate(order)}
    box = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER, [LOWER[0], UPPER[1]]])
    nodes: list[Node] = []
    for i in order:
        path = int(paths[i])
        parent = index.get(path >> 1) if path > 1 else None
        polygon = box if parent is None else clip(nodes[parent].polygon, *nodes[parent].plane, keep_right=bool(path & 1))
        X, inherited = points[path]
        nodes.append(Node(path.bit_length() - 1, parent, X, bool(is_leaf[i]), path=path, polygon=polygon,
                          plane=None if is_leaf[i] else (w[i], float(z[i])), inherited=inherited,
                          centroid=X.mean(axis=0)))  # fmt: skip
        if parent is not None:
            nodes[parent].children.append(len(nodes) - 1)
    for n in nodes:  # the recorded points lie in the polytope rebuilt from the stored planes
        if n.parent is not None:
            p = nodes[n.parent]
            # The first right-hand point defines z, so it sits on the plane up to rounding.
            score = ((n.X - LOWER) / WIDTH) @ p.plane[0] - p.plane[1]
            if not np.all(score >= -1e-12 if n.path & 1 else score < 1e-12):
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


def residual_points(ax, X, e, size=4.0) -> None:
    """Points coloured by their signed error in units of the tolerance; those beyond it are ringed."""
    over = np.abs(e) > ALPHA
    ax.scatter(*X[~over].T, s=size, c=e[~over] / ALPHA, cmap=RESIDUAL, vmin=-2, vmax=2, lw=0, zorder=3)
    ax.scatter(*X[over].T, s=size + 3, c=e[over] / ALPHA, cmap=RESIDUAL, vmin=-2, vmax=2, edgecolors=DARK, lw=0.35,
               zorder=4)  # fmt: skip


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
                ax.scatter(*n.centroid, s=18, c=DARK if n.leaf else RED, marker="x", lw=1.1, zorder=4)
        draw_tree(at, nodes, k)
        fig.tight_layout(pad=0.2, w_pad=0.4)
        save(fig, f"afinni-level-{k}")

        def errors(ax, k=k):
            for n in nodes:
                if n.level == k:
                    residual_points(ax, n.X, affine_fit(n.X, f_np(n.X))[2])
                    ax.scatter(*n.centroid, s=18, c=DARK if n.leaf else RED, marker="x", lw=1.1, zorder=5)

        if not lv.good.all():  # the report shows the errors of the levels that still reject clusters
            panel(f"afinni-error-{k}", errors)
        worst = [float(np.abs(affine_fit(n.X, f_np(n.X))[2]).max()) for n in nodes if n.level == k]
        out.append({"clusters": int(len(lv.sizes)), "accepted": int(lv.good.sum()), "points": int(lv.sizes.sum()),
                    "worst_error": round(max(worst), 2)})  # fmt: skip
    return {"levels": out, "n_levels": len(levels)}


def check_affine_fit(s: AFINNISurrogate, nodes: list[Node]) -> None:
    """The restated fit must be the library's: compare slope and centroid value on every leaf."""
    A, b, C = to_host(s.A_fits_src), to_host(s.b_fits_src), to_host(s.X_src)
    for k, n in enumerate(n for n in nodes if n.leaf):
        w, mean, _ = affine_fit(n.X, f_np(n.X))
        if not (np.allclose(w, A[k, 0], rtol=1e-8, atol=1e-10) and np.isclose(mean, b[k, 0] + A[k, 0] @ C[k], rtol=1e-8)):
            raise SystemExit("The restated affine fit differs from the library's leaf model.")


def child_error(X: np.ndarray, order: np.ndarray, sizes: np.ndarray) -> float:
    """The larger of the two children's worst errors, each child fitted on its own points."""
    halves = (X[order[: sizes[0]]], X[order[sizes[0] :]])
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
        a = M @ np.linalg.eigh(C)[1][:, -1]
        order, sizes = (to_host(v) for v in strategy.split(data))
        return a / np.linalg.norm(a), order, sizes

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

    def halves(ax, a, order, sizes):
        s = (X - c) @ a
        tau = 0.5 * (np.sort(s)[sizes[0] - 1] + np.sort(s)[sizes[0]])
        ax.scatter(*X[order[: sizes[0]]].T, s=5, c=PINK, lw=0, zorder=3)
        ax.scatter(*X[order[sizes[0] :]].T, s=5, c=CYAN, lw=0, zorder=3)
        cut_line(ax, c + tau * a, a)
        arrow(ax, c, c + 0.55 * a)
        ax.scatter(*c, s=26, c=RED, marker="x", lw=1.2, zorder=6)

    def marked(ax, rows, colour):
        ax.scatter(*X.T, s=5, c="white", edgecolors=GREY, lw=0.3, zorder=2)
        ax.scatter(*X[rows].T, s=9, c=colour, lw=0, zorder=3)
        ax.scatter(*c, s=26, c=RED, marker="x", lw=1.2, zorder=6)

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
    grown = [(int(lv_.split.counts[jj] - lv_.sizes[~lv_.good][jj]), kk, jj) for kk, lv_ in enumerate(levels)
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


def afinni_trust(s: AFINNISurrogate, nodes: list[Node]) -> dict:
    """One leaf's trust ellipse, every leaf's, and three queries at inference."""
    C, T = to_host(s.X_src), to_host(s.Qs_src)
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
        ax.scatter(*C[k], s=26, c=DARK, marker="x", lw=1.2, zorder=5)  # dark: an accepted centre

    def every(ax):
        for kk, n in enumerate(leaves):
            ellipse(ax, C[kk], T[kk], n.colour, lw=0.8)
            ax.scatter(*C[kk], s=8, c=n.colour, marker="x", lw=0.8, zorder=4)

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
        choice[name] = int(np.argmax(score))
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
                ax.scatter(*C[r["euclid"]], s=14, c=GREY, marker="x", lw=1.0, zorder=6)
            ax.plot(*zip(q, C[r["nu"]], strict=True), color=RED if r["outside"] else DARK, lw=0.9, zorder=5)
            ax.scatter(*C[r["nu"]], s=16, c=leaves[r["nu"]].colour, marker="x", lw=1.2, zorder=6)
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
                  for r in (4, 8, 12) for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))]  # fmt: skip
        for name, i in choice.items():
            place_text(ax, name, grid[i], around, paths, boxes, fontsize=8, weight="bold", zorder=8,
                       path_effects=[withStroke(linewidth=2, foreground="white")])  # fmt: skip

    panel("afinni-inference", inference, size=(2.7, 2.7))
    return {"trust_points": int(len(P)), "queries": found}


def queue_strips(nodes: list[Node], levels: list[Level]) -> None:
    """Each level's queue as one strip: clusters stored back to back, children next to each other."""
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(levels) + 0.3))
    total = max(int(lv.sizes.sum()) for lv in levels)
    left: dict[int, tuple[float, float]] = {}
    for k in range(len(levels)):
        x = 0.0
        for i, n in enumerate(nodes):
            if n.level != k:
                continue
            w = len(n.X)
            ax.add_patch(plt.Rectangle((x, -k - 0.32), w, 0.64, fc=n.colour, ec="white", lw=0.8,
                                       alpha=0.5 if n.leaf else 1.0))  # fmt: skip
            if not n.leaf:
                ax.add_patch(plt.Rectangle((x, -k - 0.32), w, 0.64, fc="none", ec=RED, lw=0.9, zorder=3))
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


def art_sampling(nodes: list[Node]) -> dict:
    """One deep node: its polytope and the planes that bound it, its points, and a hit-and-run chain."""
    deep = [n for n in nodes if n.level >= 2 and len(n.X) > n.inherited]
    node = max(deep, key=lambda n: (len(n.polygon), len(n.X) - n.inherited))
    lineage = [nodes.index(node)]  # the node, then its ancestors up to the root
    while nodes[lineage[-1]].parent is not None:
        lineage.append(nodes[lineage[-1]].parent)

    def polytope(ax):
        ax.add_patch(Polygon(node.polygon, closed=True, fc=node.colour, ec="none", alpha=0.45, zorder=2))
        ax.add_patch(Polygon(node.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2))
        for j, a in enumerate(lineage[:0:-1]):
            w, z = nodes[a].plane
            cut_line(ax, LOWER + (w * z / (w @ w)) * WIDTH, w / WIDTH, colour=PALETTE[(j + 3) % len(PALETTE)], lw=0.9,
                     ls="-", zorder=3)  # fmt: skip
        ax.scatter(*node.X[: node.inherited].T, s=7, c=DARK, lw=0, zorder=4)
        ax.scatter(*node.X[node.inherited :].T, s=7, c=PINK, lw=0, zorder=4)

    panel("art-polytope", polytope)

    # A chain as the sampler walks it (a direction uniform on the sphere, then a uniform point
    # of the chord), drawn with its own random numbers: the library's are keyed by node and draw.
    A, b = [[1, 0], [-1, 0], [0, 1], [0, -1]], [UPPER[0], -LOWER[0], UPPER[1], -LOWER[1]]
    for child in lineage[:-1]:  # one half-plane per ancestor, in x: normal . x <= offset
        w, z = nodes[nodes[child].parent].plane
        normal, offset = w / WIDTH, z + (w / WIDTH) @ LOWER
        sign = -1.0 if nodes[child].path & 1 else 1.0
        A.append(sign * normal)
        b.append(sign * offset)
    A, b = np.array(A), np.array(b)

    def chain(rng) -> list:
        x = node.X[rng.integers(node.inherited)]
        steps = []
        for _ in range(5):
            g = rng.normal(size=2)
            g /= np.linalg.norm(g)
            ratio = (b - A @ x) / (A @ g)
            t_min, t_max = ratio[A @ g < 0].max(), ratio[A @ g > 0].min()
            new = x + (t_min + (t_max - t_min) * rng.uniform()) * g
            steps.append((x + t_min * g, x + t_max * g, x, new))
            x = new
        return steps

    # The first chain, by seed, whose points stay apart, so that each point and arrow of the
    # picture can be told from the others, and which ends far from its start, as the text says.
    diameter = np.ptp(node.polygon, axis=0).max()
    for k in range(100):
        steps = chain(np.random.default_rng(SEED + k))
        visited = np.array([steps[0][2]] + [s[3] for s in steps])
        gaps = np.linalg.norm(visited[:, None] - visited[None], axis=2)[np.triu_indices(len(visited), 1)]
        if gaps.min() > 0.045 * diameter and np.linalg.norm(visited[-1] - visited[0]) > 0.5 * diameter:
            break
    else:
        raise SystemExit("No hit-and-run chain for the picture has distinct points and ends far from its start.")

    def walk(ax):
        ax.add_patch(Polygon(node.polygon, closed=True, fc=node.colour, ec="none", alpha=0.3, zorder=2))
        ax.add_patch(Polygon(node.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2))
        for j, (lo, hi, a, c) in enumerate(steps):
            last = j == len(steps) - 1
            ax.plot(*zip(lo, hi, strict=True), color=GREY, lw=0.7, ls=":", zorder=3)
            arrow(ax, a, c, colour=DARK, lw=0.8, scale=6)
            ax.scatter(*c, s=30 if last else 9, c=PINK if last else "white", edgecolors=DARK, lw=0.6, zorder=6)
        ax.scatter(*steps[0][2], s=26, c=DARK, lw=0, zorder=6)

    panel("art-hit-and-run", walk, view=node.polygon, pad=0.1)
    return {"sampling": {"path": format(node.path, "b"), "depth": node.level, "rows": len(lineage) - 1,
                         "inherited": node.inherited, "new": len(node.X) - node.inherited}}  # fmt: skip


def art_split(nodes: list[Node], searches: dict, plan: ARTPlan) -> dict:
    """One node's split search: the error pattern before and after, the candidates, the two objectives."""
    settings = plan.settings
    by_path = {n.path: n for n in nodes}

    def rows_of(rec: SplitRecord) -> SplitRows:
        return SplitRows(rec.U, rec.Y, rec.Phi, np.zeros(len(rec.U), dtype=np.int64), np.array([len(rec.U)]))

    def sse(rec: SplitRecord, w, z) -> float:
        return float(hard_sse(rows_of(rec), np.asarray(w)[None], np.array([z]), plan.min_side)[0][0])

    # One recorded search per node the plan tried to split, and one for every split node of the tree.
    split_paths = {n.path for n in nodes if n.plane is not None}
    if len(searches) != plan.counters["splits"] + plan.counters["split_failed"] or not split_paths <= searches.keys():
        raise SystemExit("The recorded split searches are not one per split node of the tree.")
    for path, rec in searches.items():  # the recorded planes are the ones stored in the tree
        plane = by_path[path].plane
        if plane is not None and not (np.allclose(plane[0], rec.final[0]) and np.isclose(plane[1], rec.final[1])):
            raise SystemExit("A recorded split search does not end at the plane stored in the tree.")
    gains = {path: 1 - sse(rec, *rec.final) / rec.bins[2] for path, rec in searches.items()
             if by_path[path].plane is not None}  # fmt: skip
    path = max(gains, key=lambda p: (round(gains[p], 6), -p))
    rec, node = searches[path], by_path[path]
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

    def after(ax):
        outline(ax)
        for side in (right, ~right):
            residual_points(ax, X[side], affine_fit(X[side], F[side])[2], size=9)
        cut_line(ax, x_point(w, z), x_normal(w), colour=DARK, lw=1.2, ls="-")

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
    return {"split": {"path": format(path, "b"), "depth": node.level, "points": n, "directions": int(len(rec.bank)),
                      "sse_parent": round(float(np.sum(affine_fit(X, F)[2] ** 2)) / ALPHA**2 / n, 2),
                      "sse_bins": round(sse_b / n, 3), "sse_final": round(sse(rec, w, z) / n, 3),
                      "gain_percent": round(100 * gains[path]), "improved": sum(g > 1e-12 for g in gains.values()),
                      "searched": len(gains), "left": int((~right).sum()), "right": int(right.sum()),
                      "n_bins": settings.n_bins, "soft_iters": settings.soft_iters,
                      "temperature": settings.soft_temperature}}  # fmt: skip


def art_route(s: AFINNISurrogate, nodes: list[Node]) -> dict:
    """One query on its way down: the polytope it is in at each depth, and its path in the tree."""
    deepest = max((n for n in nodes if n.leaf), key=lambda n: (n.level, -n.path))
    q = deepest.polygon.mean(axis=0)
    path = [nodes.index(deepest)]
    while nodes[path[-1]].parent is not None:
        path.append(nodes[path[-1]].parent)
    path = path[::-1]
    leaf = int(to_host(s.route_leaves(q[None]))[0])
    if int(s.art_leaves["path_code"][leaf]) != deepest.path:
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
    return {"route": {"depth": deepest.level, "path": format(deepest.path, "b"),
                      "sides": [int(u @ nodes[i].plane[0] - nodes[i].plane[1] >= 0) for i in path[:-1]]}}  # fmt: skip


# ----------------------------------------------------------------------------- both
def compare(afinni: AFINNISurrogate, art: AFINNISurrogate, afinni_nodes: list[Node], art_nodes: list[Node]) -> dict:
    """Both surrogates against f: along one line, and as error maps of the domain."""
    n = 240
    grid = domain_grid(n)
    G0, G1 = grid[:, 0].reshape(n, n), grid[:, 1].reshape(n, n)
    truth = f_np(grid)
    pa, oa = (np.ravel(to_host(v)) for v in afinni.predict(grid, k_nn=1, coarse_factor=KAPPA))
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
        ax.scatter(*to_host(afinni.X_src).T, s=5, c=DARK, marker="x", lw=0.6, zorder=4)

    def error_art(ax):
        ax.imshow(er.reshape(n, n), origin="lower", extent=extent, cmap=ERROR_CMAP, norm=norm, zorder=1,
                  interpolation="nearest")  # fmt: skip
        for nd in art_nodes:
            if nd.leaf:
                ax.add_patch(Polygon(nd.polygon, closed=True, fc="none", ec=DARK, lw=0.45, zorder=3))

    panel("error-afinni", error_afinni, back=False)
    panel("error-art", error_art, back=False)

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

    return {"afinni": summary(ea, oa, afinni, afinni_nodes), "art": summary(er, orr, art, art_nodes),
            "slice_x2": x2, "grid": n, "flagged_on_line": int(flagged.sum())}  # fmt: skip


# ----------------------------------------------------------------------------- work stealing
def record_steals() -> None:
    """Under ``mpiexec -n 4``: train ART with work stealing, check its tree against a run without, store the run."""

    def train(stealing: bool) -> AFINNISurrogate:
        s = AFINNISurrogate()
        s.train(art_initial_points(), art_config(
            atol=STEAL["atol"], max_levels=STEAL["max_levels"], batch_size=STEAL["batch_size"],
            use_work_stealing=stealing, profile_events=True,
            art=ARTSettings(sample_multiplier=ART_MULTIPLIER, steal_min_load=STEAL["steal_min_load"],
                            steal_size=STEAL["steal_size"])), Evaluator(f_fn=f_xp))  # fmt: skip
        return s

    s = train(True)
    reference = train(False)  # collective too: without stealing, every rank but 0 idles
    if get_mpi_rank() != 0:
        return
    stats = s.profiling_stats
    if len(stats["per_rank"]) != STEAL["ranks"]:
        raise SystemExit(f"Run under mpiexec -n {STEAL['ranks']}.")
    def keyed(model: AFINNISurrogate) -> tuple[dict, np.ndarray]:
        """Every split plane by its node's path code, and the sorted leaf codes: the tree, whatever the row order."""
        a = {k: to_host(v) for k, v in model.routing.arrays().items()}
        planes = {int(path): (w.tobytes(), float(z)) for path, w, z in zip(a["split_path"], a["w"], a["z"], strict=True)}
        return planes, np.sort(a["leaf_path"])

    tree = {k: to_host(v) for k, v in s.routing.arrays().items()}
    (planes, leaves), (planes_ref, leaves_ref) = keyed(s), keyed(reference)
    if planes != planes_ref or not np.array_equal(leaves, leaves_ref):
        raise SystemExit("The tree built with work stealing differs from the one built without.")
    counts = [int(r["num_final_clusters"]) for r in stats["per_rank"]]
    keep = ("kind", "t", "dur", "phase", "level", "victim", "thief", "n_pts", "n_clusters", "rank", "batch")
    kinds = ("run_start", "done", "steal_grant", "steal_received")
    events = [[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k in keep}
               for e in r["events"] if e["kind"] in kinds or e.get("phase") == "time_idle"]
              for r in stats["per_rank"]]  # fmt: skip
    data = {"settings": STEAL, "leaves": int(s.N_src), "leaves_per_rank": counts,
            # Leaves are gathered in rank order, so the first counts[0] leaves were built by rank 0.
            "leaf_paths": np.asarray(s.art_leaves["path_code"]).tolist(),
            "split_paths": tree["split_path"].tolist(), "w": tree["w"].tolist(), "z": tree["z"].tolist(),
            "events": events}  # fmt: skip
    (OUT / "steal_run.json").write_text(json.dumps(data) + "\n")
    print(f"recorded {data['leaves']} leaves, {counts} per rank; the tree equals the one built without stealing")


def steal_panels() -> dict:
    """The recorded 4-rank run: who was working when, and which rank built which leaf."""
    run = json.loads((OUT / "steal_run.json").read_text())
    events, P = run["events"], run["settings"]["ranks"]
    end = max(e["t"] for ev in events for e in ev if e["kind"] == "done")
    fig, ax = plt.subplots(figsize=(6.2, 1.2))
    for r, ev in enumerate(events):
        ax.add_patch(plt.Rectangle((0, r - 0.3), end, 0.6, fc=RANKS[r], ec="none", zorder=1))
        for e in ev:
            if e.get("phase") == "time_idle":
                ax.add_patch(plt.Rectangle((e["t"], r - 0.3), e["dur"], 0.6, fc="#eeeeee", ec="none", zorder=2))
    received = [e for ev in events for e in ev if e["kind"] == "steal_received"]
    for e in received:
        a, b = e["victim"], e["rank"]
        arrow(ax, (e["t"], a + (0.3 if b > a else -0.3)), (e["t"], b - (0.3 if b > a else -0.3)), colour=DARK, lw=0.7, scale=5)
    ax.set_yticks(range(P), [f"rank {r}" for r in range(P)], fontsize=6.5)
    ax.set_ylim(P - 0.45, -0.55)
    ax.set_xlim(0, end)
    ax.set_xlabel("time (s)", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, "steal-timeline")

    planes = {int(path): (np.array(w), z) for path, w, z in zip(run["split_paths"], run["w"], run["z"], strict=True)}
    owner = np.repeat(np.arange(P), run["leaves_per_rank"])  # leaves are gathered in rank order
    box = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER, [LOWER[0], UPPER[1]]])

    def polygon(path: int) -> np.ndarray:
        poly = box
        for shift in range(path.bit_length() - 2, -1, -1):  # from the root down: one bit per level
            poly = clip(poly, *planes[path >> (shift + 1)], keep_right=bool((path >> shift) & 1))
        return poly

    def owners(ax):
        for path, r in zip(run["leaf_paths"], owner, strict=True):
            ax.add_patch(Polygon(polygon(path), closed=True, fc=RANKS[r], ec="white", lw=0.3, alpha=0.8, zorder=1))

    panel("steal-owners", owners, back=False)
    return {"leaves": run["leaves"], "leaves_per_rank": run["leaves_per_rank"], "steals": len(received),
            "ranks": P, "atol": run["settings"]["atol"], "batch_size": run["settings"]["batch_size"],
            "depth": max(p.bit_length() - 1 for p in run["leaf_paths"])}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-steals", action="store_true", help="record steal_run.json (run under mpiexec -n 4)")
    args = parser.parse_args()
    if IS_CUPY_BACKEND:
        raise SystemExit("Run with AFINNI_DEVICE=cpu: the panels and their numbers are the CPU results.")
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
        "settings": {"afinni": {k: v for k, v in AFINNI_SETTINGS.items() if v is not None},
                     "art": {"sample_multiplier": ART_MULTIPLIER, "node_points": ART_MULTIPLIER * 3,
                             "max_levels": ART_SETTINGS["max_levels"]}},
        "colours": {"palette": PALETTE, "grey": GREY, "light": LIGHT, "dark": DARK, "red": RED, "blue": BLUE,
                    "gold": GOLD, "green": GREEN, "pink": PINK, "cyan": CYAN, "ranks": RANKS,
                    "residual_mid": RESIDUAL_MID, "error": [matplotlib.colors.to_hex(c) for c in ERROR_CMAP.colors]},
    }  # fmt: skip
    panel("example", lambda ax: None)
    numbers["afinni"] = afinni_levels(forest, levels) | afinni_split(levels) | afinni_trust(afinni, forest)
    queue_strips(forest, levels)
    numbers["art"] = art_levels(tree) | art_sampling(tree) | art_split(tree, searches, art_plan) | art_route(art, tree)
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
