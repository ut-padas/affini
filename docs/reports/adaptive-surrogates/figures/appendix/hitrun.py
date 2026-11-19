"""Draw the panels of the appendix section on why hit-and-run from inherited points is uniform, and write apx-hitrun.json.

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/adaptive-surrogates/figures/appendix/hitrun.py

Two ART runs of the running example are trained through the public API (make_figures'
``record_art``, and the same configuration at make_figures' ``STEAL["atol"]`` for a deeper tree),
with ``HitAndRunSampler._walk`` wrapped to count the steps every chain walks. The node areas come
from the trees' polytopes. For one depth-1 child, the library's own sampler (the plan's
``HitAndRunSampler``, with the node's keys, rows and inherited points) is called again: the script
stops unless it reproduces the node's new points exactly, and then draws many more chains from the
same starts, and from uniform starts, to measure how far their endpoints are from uniform.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
import numpy as np  # noqa: E402
from make_figures import DARK, GREY, LOWER, PINK, WIDTH, Polygon, plt  # noqa: E402
from scipy.stats import kstwo  # noqa: E402

from src import to_host  # noqa: E402
from src.art.constants import PURPOSE_TRAIN  # noqa: E402
from src.art.rng import node_keys  # noqa: E402
from src.art.sampling import HitAndRunSampler, NodePolytopes  # noqa: E402
from src.components.sampling import ClusterPoints  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.surrogate import AdaptiveSurrogate  # noqa: E402

CHAINS = 4000  # chains per experiment on the depth-1 child
TAUS = [
    0,
    1,
    2,
    3,
    5,
    10,
    20,
    40,
    80,
]  # walk lengths of the experiment; 0 is the start points themselves


def area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()))


class StepCounter:
    """Wraps ``HitAndRunSampler._walk`` to count chain-steps and the chains that started walking."""

    def __init__(self):
        self.steps = self.chains = self.recovery = 0
        self.wrapped = HitAndRunSampler._walk

    def __enter__(self):
        counter, wrapped = self, self.wrapped

        def walk(
            sampler,
            clusters,
            geo,
            labels,
            draw,
            U,
            rows,
            t_from,
            t_to,
            bad,
            coordinate=False,
        ):
            k = int(to_host(rows).sum())
            counter.steps += k * max(t_to - t_from, 0)
            counter.chains += k if (t_from == 0 and not coordinate) else 0
            counter.recovery += k if coordinate else 0
            return wrapped(
                sampler,
                clusters,
                geo,
                labels,
                draw,
                U,
                rows,
                t_from,
                t_to,
                bad,
                coordinate,
            )

        HitAndRunSampler._walk = walk
        return self

    def __exit__(self, *exc):
        HitAndRunSampler._walk = self.wrapped


def deep_areas() -> tuple[np.ndarray, np.ndarray, StepCounter, int]:
    """Depth and area of every node of the running example at the tighter tolerance, from its routing table."""
    with StepCounter() as counter:
        s = AdaptiveSurrogate()
        s.train(mf.art_initial_points(), mf.art_config(atol=mf.STEAL["atol"], max_levels=mf.STEAL["max_levels"]),
                Evaluator(f_fn=mf.f_xp))  # fmt: skip
    r = s.routing
    is_leaf, w, z, codes = (
        to_host(getattr(r, k))
        for k in ("is_leaf", "normal", "offset", "node_locational_code")
    )
    box = np.array([LOWER, [mf.UPPER[0], LOWER[1]], mf.UPPER, [LOWER[0], mf.UPPER[1]]])
    poly, plane = {}, {}
    for i in np.argsort(
        [int(c) for c in codes], kind="stable"
    ):  # a parent before its children
        code = int(codes[i])
        poly[code] = (
            box
            if code == 1
            else mf.clip(poly[code >> 1], *plane[code >> 1], keep_right=bool(code & 1))
        )
        if not is_leaf[i]:
            plane[code] = (w[i], float(z[i]))
    depth = np.array([c.bit_length() - 1 for c in poly])
    areas = np.array([area(p) for p in poly.values()])
    leaves = int(is_leaf.sum())
    if not np.isclose(
        areas[np.isin(list(poly), codes[is_leaf].astype(np.int64))].sum(), area(box)
    ):
        raise SystemExit("The leaves of the deeper run do not tile the domain.")
    return depth, areas, counter, leaves


def marginal_cdf(polygon: np.ndarray, j: int, s: np.ndarray) -> np.ndarray:
    """The exact CDF of u_j under the uniform law on the polygon, from the area left of u_j = s."""
    e = np.eye(2)[j]
    total = area(polygon)
    return np.array([area(c) / total if len(c := mf.clip(polygon, e, float(v), keep_right=False)) >= 3 else 0.0
                     for v in s])  # fmt: skip


def ks(polygon: np.ndarray, X: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance to the uniform law on the polygon, the larger of the two coordinates'."""
    U = (X - LOWER) / WIDTH
    out = 0.0
    for j in range(2):
        s = np.sort(U[:, j])
        F = marginal_cdf(polygon, j, s)
        n = len(s)
        out = max(
            out,
            float(
                np.max(np.maximum(np.arange(1, n + 1) / n - F, F - np.arange(n) / n))
            ),
        )
    return out


def uniform_in(polygon: np.ndarray, n: int, rng) -> np.ndarray:
    """Independent uniform points of the polygon, by rejection from its bounding box (a reference, not the library)."""
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    path = mf.MPath(polygon)
    out = np.empty((0, 2))
    while len(out) < n:
        X = rng.uniform(lo, hi, (4 * n, 2))
        out = np.vstack([out, X[path.contains_points(X)]])
    return out[:n]


def main() -> None:
    if mf.IS_CUPY_BACKEND:
        raise SystemExit(
            "Run with SURROGATE_DEVICE=cpu: the panels and their numbers are the CPU results."
        )
    with StepCounter() as counter:
        s, points, _, plan = mf.record_art()
    tree = mf.art_tree(s, points)
    tau = plan.sampler.walk_steps
    new_total = sum(len(n.X) - n.inherited for n in tree)
    if (
        counter.chains != new_total
        or counter.steps != tau * new_total
        or counter.recovery
    ):
        raise SystemExit(
            "A chain of the example walked other than walk_steps steps; the text assumes it did not."
        )
    root_area = area(tree[0].polygon)
    if (
        not np.allclose(tree[0].X, mf.art_initial_points())
        or len(tree[0].X) != tree[0].inherited
    ):
        raise SystemExit("The root is expected to hold exactly the initial points.")
    deep_depth, deep_area, deep_counter, deep_leaves = deep_areas()

    # (a) Expected rejection proposals per point against the walk's fixed steps.
    ratio = np.array([root_area / area(n.polygon) for n in tree])
    depth = np.array([n.level for n in tree])
    deep_ratio = (
        area(
            np.array(
                [LOWER, [mf.UPPER[0], LOWER[1]], mf.UPPER, [LOWER[0], mf.UPPER[1]]]
            )
        )
        / deep_area
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    jitter = np.random.default_rng(mf.SEED).uniform(-0.22, 0.22, len(deep_depth))
    ax.scatter(
        deep_depth + jitter, deep_ratio, s=1.2, c=GREY, lw=0, alpha=0.5, zorder=1
    )
    med = [np.median(deep_ratio[deep_depth == k]) for k in range(deep_depth.max() + 1)]
    ax.plot(
        range(len(med)),
        med,
        color=GREY,
        lw=1.0,
        zorder=2,
        label=f"$\\alpha_{{tol}}$ = {mf.STEAL['atol']:g}: {len(deep_depth)} nodes and their median",
    )
    ax.scatter(
        depth,
        ratio,
        s=14,
        c=[n.colour for n in tree],
        edgecolors=DARK,
        lw=0.4,
        zorder=4,
        label=f"$\\alpha_{{tol}}$ = {mf.ALPHA:g}: the running example's {len(tree)} nodes",
    )
    k = np.arange(deep_depth.max() + 1)
    ax.plot(k, 2.0**k, color=DARK, lw=0.8, ls="--", zorder=3)
    ax.axhline(tau, color=PINK, lw=1.2, zorder=3)
    ax.text(
        k[-1],
        tau / 1.3,
        f"hit-and-run: $\\tau$ = {tau} steps per point",
        fontsize=6.5,
        color=PINK,
        ha="right",
        va="top",
    )
    ax.legend(
        fontsize=6.3,
        frameon=False,
        loc="upper left",
        handlelength=1.2,
        borderaxespad=0.1,
        markerscale=0.8,
    )
    ax.text(
        k[-1] - 0.3,
        2.0 ** (k[-1] - 0.3),
        "$2^{\\mathrm{depth}}$",
        fontsize=7,
        color=DARK,
        ha="right",
        va="bottom",
    )
    ax.set_yscale("log", base=10)
    ax.set_xlabel("depth", fontsize=7, labelpad=1)
    ax.set_ylabel(
        "$|\\Omega| / |\\Omega_C|$: rejection proposals per point",
        fontsize=7,
        labelpad=1,
    )
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    mf.save(fig, "apx-hitrun-area")

    # (b) A depth-1 child: the library's sampler reproduces its new points, then walks many more chains.
    child = min((n for n in tree if n.level == 1), key=lambda n: n.inherited)
    parent = tree[child.parent]
    row = np.r_[parent.plane[0], parent.plane[1]] * (
        -1.0 if child.code & 1 else 1.0
    )  # as ARTPlan._split
    geo = NodePolytopes(np.array([0]), np.array([child.level]), node_keys(mf.SEED, np.array([0]), np.array([child.code])),
                        row[None, None, :], PURPOSE_TRAIN)  # fmt: skip
    starts = child.X[: child.inherited]
    n_new = len(child.X) - child.inherited
    again = plan.sampler.sample(
        ClusterPoints(starts, np.array([len(starts)]), polytopes=geo), np.array([n_new])
    )
    if not np.array_equal(to_host(again), child.X[child.inherited :]):
        raise SystemExit(
            "The library's sampler, called again, does not reproduce the child's new points."
        )
    rng = np.random.default_rng(mf.SEED)
    uniform_starts = uniform_in(child.polygon, CHAINS, rng)
    reference = uniform_in(child.polygon, CHAINS, rng)

    def endpoints(X0: np.ndarray, steps: int) -> np.ndarray:
        """``CHAINS`` endpoints after ``steps`` steps from starts picked among ``X0``: the plan's sampler
        with the drift check off, so every chain walks exactly ``steps`` steps on the plan's random stream."""
        sampler = HitAndRunSampler(
            plan.domain, plan.roots, steps, plan.sampler.max_walk_steps, mixing_z=np.inf
        )
        g = NodePolytopes(geo.root, geo.depth, geo.keys, geo.constraints, geo.purpose)
        out = to_host(
            sampler.sample(
                ClusterPoints(X0, np.array([len(X0)]), polytopes=g), np.array([CHAINS])
            )
        )
        if bool(to_host(g.insufficient)[0]):
            raise SystemExit(
                "A chain of the uniformity experiment failed the sampler's own checks."
            )
        return out

    walks = {t: endpoints(starts, t) for t in TAUS if t}
    if not np.array_equal(walks[tau][:n_new], child.X[child.inherited :]):
        raise SystemExit(
            "The experiment's chains do not extend the child's own chains."
        )
    D_few = [ks(child.polygon, starts)] + [
        ks(child.polygon, walks[t]) for t in TAUS if t
    ]
    D_uni = [ks(child.polygon, uniform_starts)] + [
        ks(child.polygon, endpoints(uniform_starts, t)) for t in TAUS if t
    ]
    D_ref = ks(child.polygon, reference)
    critical = float(
        kstwo.ppf(1 - 0.05 / 2, CHAINS)
    )  # 5% for the larger of two coordinates (Bonferroni)

    def child_map(ax, X, label_new):
        ax.add_patch(
            Polygon(
                child.polygon,
                closed=True,
                fc=child.colour,
                ec="none",
                alpha=0.25,
                zorder=1,
            )
        )
        ax.add_patch(
            Polygon(child.polygon, closed=True, fc="none", ec=DARK, lw=0.9, zorder=2)
        )
        mf.domain_box(ax)
        if label_new:
            ax.scatter(*X.T, s=5, c=PINK, lw=0, zorder=3)
        else:
            ax.scatter(*X.T, s=0.6, c=PINK, lw=0, alpha=0.6, zorder=3)
        ax.scatter(*starts.T, s=9, c=DARK, lw=0, zorder=4)

    mf.panel(
        "apx-hitrun-child",
        lambda ax: child_map(ax, child.X[child.inherited :], True),
        size=(1.9, 1.9),
    )
    mf.panel(
        "apx-hitrun-walk-1", lambda ax: child_map(ax, walks[1], False), size=(1.9, 1.9)
    )
    mf.panel(
        f"apx-hitrun-walk-{tau}",
        lambda ax: child_map(ax, walks[tau], False),
        size=(1.9, 1.9),
    )

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    x = np.arange(len(TAUS))
    ax.axhspan(
        0,
        critical,
        color="#e9edf3",
        lw=0,
        zorder=0,
        label=f"5% level for {CHAINS} independent uniform points",
    )
    ax.plot(
        x,
        D_few,
        color=DARK,
        lw=1.0,
        marker="o",
        ms=3,
        zorder=3,
        label=f"from the child's {len(starts)} inherited points",
    )
    ax.plot(x, D_uni, color=DARK, lw=1.0, ls="--", marker="o", ms=3, mfc="white", zorder=3,
            label=f"from {CHAINS} uniform points")  # fmt: skip
    ax.set_xticks(x, [str(t) for t in TAUS])
    ax.set_yscale("log")
    ax.set_xlabel(
        "steps per chain $\\tau$ (0: the start points)", fontsize=7, labelpad=1
    )
    ax.set_ylabel(f"KS distance to uniform, {CHAINS} endpoints", fontsize=7, labelpad=1)
    ax.legend(
        fontsize=6.5,
        frameon=False,
        loc="upper right",
        handlelength=2.2,
        borderaxespad=0.1,
    )
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    mf.save(fig, "apx-hitrun-ks")

    # Blocks of the child's size: the 4000 chains (the child's own block is the first) and the
    # independent reference points, each judged at the 5% level for n_new independent points.
    crit_new = float(kstwo.ppf(1 - 0.05 / 2, n_new))
    blocks = CHAINS // n_new
    chain_blocks = [
        ks(child.polygon, walks[tau][b * n_new : (b + 1) * n_new])
        for b in range(blocks)
    ]
    ref_blocks = [
        ks(child.polygon, reference[b * n_new : (b + 1) * n_new]) for b in range(blocks)
    ]
    first_ok = next(t for t, D in zip(TAUS, D_few, strict=True) if critical >= D)
    numbers = {
        "tau": tau, "max_walk_steps": plan.sampler.max_walk_steps, "new_points": new_total,
        "steps": counter.steps, "max_ratio": round(float(ratio.max())), "max_ratio_depth": int(depth[ratio.argmax()]),
        "depth": int(depth.max()), "above_tau": int((ratio > tau).sum()), "nodes": len(tree),
        "deep": {"atol": mf.STEAL["atol"], "nodes": len(deep_depth), "leaves": deep_leaves,
                 "depth": int(deep_depth.max()), "median_deepest": f"{med[-1]:.0f}",
                 "max_ratio": f"{deep_ratio.max():.0f}", "above_tau_percent": round(100 * float(np.mean(deep_ratio > tau))),
                 "steps_per_point": round(deep_counter.steps / deep_counter.chains, 2),
                 "recovery_chains": deep_counter.recovery},
        "child": {"code": format(child.code, "b"), "inherited": len(starts), "new": n_new,
                  "area_percent": round(100 * area(child.polygon) / root_area, 1), "chains": CHAINS,
                  "D_starts": round(D_few[0], 3), "D_first": round(D_few[1], 3), "D_tau": round(D_few[TAUS.index(tau)], 3),
                  "D_uniform_min": round(min(D_uni), 3), "D_uniform_max": round(max(D_uni), 3), "D_reference": round(D_ref, 3), "critical": round(critical, 3),
                  "first_tau_within": first_ok,
                  "D_node_new": round(ks(child.polygon, child.X[child.inherited :]), 3),
                  "critical_node_new": round(crit_new, 3), "blocks": blocks,
                  "blocks_above": int(np.sum(np.array(chain_blocks) > crit_new)),
                  "reference_blocks_above": int(np.sum(np.array(ref_blocks) > crit_new))},
    }  # fmt: skip
    (mf.OUT / "apx-hitrun.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers))


if __name__ == "__main__":
    main()
