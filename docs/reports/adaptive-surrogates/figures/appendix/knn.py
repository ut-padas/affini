"""Draw the panel of the companion's appendix on the exact nearest-centre search (apx-knn-*.svg, apx-knn.json).

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/adaptive-surrogates/figures/appendix/knn.py

The running example's AFINNI leaves are indexed with ``CpuExactKNN``. Its ``kneighbors`` takes
the cKDTree path in two dimensions, so the script calls the verified search of higher
dimensions (``_verified``, ``_search32``, ``_error_bound``) directly on the same index. It
records the rows ``_verified`` sends to the float64 scan by wrapping ``_fallback``, restates
the certificate from the float32 results and stops unless the restatement flags the same rows,
the two paths return the same neighbours, and those are the nearest by a scan of every leaf.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402

from src import IS_CUPY_BACKEND, to_host  # noqa: E402
from src.neighbors import knn  # noqa: E402

GRID = 240  # the queries: the grid of the companion's error maps
FAR_EXP, MERGE_FAR, DENSE = 9, 10, 40  # the demonstration sets of main()


def recorded_fallback():
    """Wrap ``knn._fallback`` so every mask of rows ``_verified`` sends to the float64 scan is kept."""
    masks: list[np.ndarray] = []
    original = knn._fallback

    def fallback(index, xp, q64, k, flagged, out_d, out_i):
        masks.append(np.asarray(flagged).copy())
        return original(index, xp, q64, k, flagged, out_d, out_i)

    knn._fallback = fallback
    return masks, original


def certificate(index, queries: np.ndarray, k: int) -> dict:
    """
    The certificate of ``_verified``, restated row by row from the float32 candidates: the k'-th
    float32 squared distance minus the bound must exceed the k-th float64 squared distance.
    """
    n, d = index.points64.shape
    kk = min(n, k + max(knn._VERIFY_SLACK, k))
    d32, cand = index._search32(queries, kk)
    # As _verified: by index, then a stable sort by distance.
    cand = np.sort(cand, axis=1)
    d64 = knn._candidate_d2(np, queries, index.points64, cand)
    order = np.argsort(d64, axis=1, kind="stable")
    dk = np.take_along_axis(d64, order, axis=1)[:, k - 1]
    qc = (queries - index.center).astype(np.float32).astype(np.float64)
    qc64 = queries - index.center
    x_sq = float(index.sq_norms.astype(np.float64).max())
    beta = knn._error_bound(
        np, (qc * qc).sum(axis=1), x_sq, d, (qc64 * qc64).sum(axis=1), 2.0 * x_sq
    )
    outside_min = d32[:, kk - 1].astype(np.float64) ** 2 - beta
    return {"kk": kk, "d32": d32, "cand": cand, "d64": d64, "order": order, "dk": dk, "beta": beta,
            "outside_min": outside_min, "certified": outside_min > dk}  # fmt: skip


def search_demo(points: np.ndarray, query: float, k: int) -> dict:
    """One query on a one-dimensional set: the float32 top k as found, the verified search, and the float64 scan."""
    index = knn.CpuExactKNN(points[:, None])
    q = np.array([[query]])
    _, top32 = index._search32(q, k)
    masks, original = recorded_fallback()
    try:
        _, exact = knn._verified(index, np, q, k)
    finally:
        knn._fallback = original
    _, scan = knn._direct_float64(q, index.points64, k)
    if not np.array_equal(exact, scan):
        raise SystemExit(
            "The verified search disagrees with the float64 scan on a demonstration set."
        )
    cert = certificate(index, q, k)
    return {"k": k, "kk": int(cert["kk"]), "float32_top": [float(points[i]) for i in top32[0]],
            "exact": [float(points[i]) for i in exact[0]], "beta": f"{cert['beta'][0]:.0e}",
            "fallback_rows": int(index.last_search_stats["fallback_rows"]),
            "flagged": int(sum(int(m.sum()) for m in masks))}  # fmt: skip


def main() -> None:
    if IS_CUPY_BACKEND:
        raise SystemExit(
            "Run with SURROGATE_DEVICE=cpu: the panels and their numbers are the CPU results."
        )
    s, levels, _ = mf.record_afinni()
    leaves = [nd for nd in mf.afinni_forest(levels) if nd.leaf]
    C = to_host(s.X_src)
    n, d = C.shape
    k = mf.KAPPA
    grid = mf.domain_grid(GRID)

    # The search predict runs in two dimensions (the cKDTree), and the verified search of d > 10.
    index = knn.CpuExactKNN(C)
    _, tree_idx = index.kneighbors(grid, k)
    tree_stats = dict(index.last_search_stats)
    masks, original = recorded_fallback()
    try:
        _, ver_idx = knn._verified(index, np, grid, k)
    finally:
        knn._fallback = original
    ver_stats = dict(index.last_search_stats)
    if tree_stats["strategy"] != "kdtree" or ver_stats["strategy"] != "verified":
        raise SystemExit(
            f"Unexpected search strategies: {tree_stats['strategy']}, {ver_stats['strategy']}."
        )
    full = knn._candidate_d2(
        np, grid, index.points64, np.broadcast_to(np.arange(n), (len(grid), n))
    )
    # Every leaf, ties to the smaller index.
    scan_idx = np.argsort(full, axis=1, kind="stable")[:, :k]
    if not (np.array_equal(tree_idx, scan_idx) and np.array_equal(ver_idx, scan_idx)):
        raise SystemExit(
            "The two searches and the scan of every leaf disagree on the nearest centres."
        )

    cert = certificate(index, grid, k)
    flagged = masks[0] if masks else np.zeros(len(grid), bool)
    if (
        not np.array_equal(flagged, ~cert["certified"])
        or int(flagged.sum()) != ver_stats["fallback_rows"]
    ):
        raise SystemExit(
            "The restated certificate does not flag the rows the library sends to the float64 scan."
        )
    # The lemma's premise, |d~^2 - d^2| <= beta, for every leaf and every query.
    d32_all, cand_all = index._search32(grid, n)
    err = np.abs(
        d32_all.astype(np.float64) ** 2 - np.take_along_axis(full, cand_all, axis=1)
    )
    ratio = err / cert["beta"][:, None]
    if ratio.max() > 1:
        raise SystemExit(
            "A float32 squared distance is farther from the float64 one than the bound allows."
        )
    # How far each row is from failing, in units of its bound.
    margin = (cert["outside_min"] - cert["dk"]) / cert["beta"]

    # The scene: the grid query nearest the centre of the domain.
    qi = int(np.argmin(np.linalg.norm(grid - (mf.LOWER + mf.UPPER) / 2, axis=1)))
    q = grid[qi]
    kk = cert["kk"]
    # The candidates, nearest first in float64.
    ranked = cert["cand"][qi, cert["order"][qi]]
    nearest, others = ranked[:k], ranked[k:]
    omitted = np.setdiff1d(np.arange(n), ranked)
    r_k = float(np.sqrt(cert["dk"][qi]))
    r_kk = float(cert["d32"][qi, kk - 1])
    beta = float(cert["beta"][qi])
    # Exaggerate the band by the largest 1, 2 or 5 times a power of ten that keeps it within 40% of
    # the gap between the two squared radii, so the margin stays visible.
    gap2 = r_kk**2 - r_k**2
    scale_exp = int(np.floor(np.log10(0.4 * gap2 / beta)))
    scale_factor = max(f for f in (1, 2, 5) if f * 10.0**scale_exp * beta <= 0.4 * gap2)
    scale = scale_factor * 10.0**scale_exp
    r_in = float(np.sqrt(r_kk**2 - scale * beta))
    if r_in <= r_k:
        raise SystemExit(
            "The exaggerated band reaches the k-th radius: choose a smaller scale."
        )
    d2 = full[qi]

    fig = mf.plt.figure(figsize=(2.6, 3.4))
    ax = fig.add_axes((0.0, 0.28, 1.0, 0.72))
    mf.box_axes(ax)
    mf.background(ax)
    ax.add_patch(
        Wedge(
            q, r_kk, 0, 360, width=r_kk - r_in, fc=mf.GREY, alpha=0.45, lw=0, zorder=2
        )
    )
    ax.add_patch(Circle(q, r_kk, fc="none", ec=mf.DARK, lw=0.9, ls="--", zorder=3))
    ax.add_patch(Circle(q, r_k, fc="none", ec=mf.DARK, lw=1.1, zorder=3))
    for i in omitted:
        mf.cross(ax, C[i], leaves[i].colour, size=18, lw=1.0, zorder=4)
        ax.collections[-1].set_alpha(0.35)
    for i in others:
        mf.cross(ax, C[i], leaves[i].colour, size=18, lw=1.0, zorder=5)
    for i in nearest:
        mf.cross(ax, C[i], leaves[i].colour, size=30, lw=1.7, zorder=6)
    ax.scatter(*q, s=16, c="black", zorder=7)

    # The same leaves on the axis of squared distance: the certificate is a statement on this line.
    rl = fig.add_axes((0.04, 0.02, 0.92, 0.2))
    top = 1.08 * d2.max()
    tall = 1 / 1.3  # the lines stop below the band's label
    rl.axvspan(r_in**2, r_kk**2, ymax=tall, color=mf.GREY, alpha=0.45, lw=0)
    rl.axvline(r_k**2, ymax=tall, color=mf.DARK, lw=1.1)
    rl.axvline(r_kk**2, ymax=tall, color=mf.DARK, lw=0.9, ls="--")
    for group, size, lw, alpha in (
        (omitted, 14, 0.9, 0.35),
        (others, 14, 0.9, 1.0),
        (nearest, 22, 1.4, 1.0),
    ):
        for i in group:
            rl.scatter(
                d2[i],
                0.5,
                s=size,
                c=leaves[i].colour,
                marker="x",
                lw=lw,
                alpha=alpha,
                zorder=4,
            )
    rl.annotate(
        "",
        (r_in**2, 0.12),
        (r_k**2, 0.12),
        arrowprops={"arrowstyle": "<->", "lw": 0.6, "color": mf.DARK},
    )
    rl.text(
        (r_k**2 + r_in**2) / 2,
        0.2,
        "margin",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=mf.DARK,
    )
    for x, label, ha in (
        (r_k**2, "$d^2_{(k)}$", "right"),
        (r_kk**2, "$\\tilde d^{\\,2}_{(k')}$", "left"),
    ):
        rl.text(
            x + (0.04 if ha == "left" else -0.04) * top / 6,
            1.0,
            label,
            ha=ha,
            va="top",
            fontsize=6,
            color=mf.DARK,
        )
    # The band is drawn scale times wider than beta, and says so.
    rl.text(
        (r_in**2 + r_kk**2) / 2,
        1.04,
        f"$\\beta \\times {scale_factor}{{\\cdot}}10^{{{scale_exp}}}$ (exaggerated)",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=mf.DARK,
    )
    rl.set_xlim(0, top)
    rl.set_ylim(0, 1.3)
    rl.set_yticks([])
    rl.spines[["left", "right", "top"]].set_visible(False)
    rl.tick_params(axis="x", labelsize=5.5, length=2, pad=1)
    rl.set_xlabel("squared distance to the query", fontsize=6, labelpad=1)
    mf.save(fig, "apx-knn-scene")

    # Two one-dimensional sets with a point near 10^FAR_EXP. In the first, 0 and 1 (listed in that
    # order, so the tie goes to 1) merge in float32 once centred; in the second, forty points
    # spaced by 1 lie closer together than float32 resolves near their mean.
    far = 10.0**FAR_EXP
    big = np.r_[1.0, 0.0, far + 1e6 * np.arange(MERGE_FAR)]
    dense = np.r_[np.arange(float(DENSE)), far]
    numbers = {
        "k": k, "kk": int(kk), "slack": knn._VERIFY_SLACK, "leaves": int(n), "tree_max_d": knn.CPU_TREE_MAX_D,
        "grid": int(len(grid)),
        "fallback_rows": int(ver_stats["fallback_rows"]), "tree_fallback_rows": int(tree_stats["fallback_rows"]),
        "min_margin": f"{margin.min():.1e}", "max_ratio_percent": int(round(100 * ratio.max())),
        "beta_max": f"{cert['beta'].max():.1e}",
        "scene": {"query": [round(float(v), 2) for v in q], "r_k": round(r_k, 3), "r_kk": round(r_kk, 3),
                  "beta": f"{beta:.1e}", "scale_factor": scale_factor, "scale_exp": scale_exp, "omitted": int(len(omitted)),
                  "margin": f"{(cert['outside_min'][qi] - cert['dk'][qi]) / beta:.1e}"},
        "far_exp": FAR_EXP,
        "merge": {"query": 0.25, "far": MERGE_FAR, "spacing": float(np.spacing(np.float32(big.mean()))),
                  **search_demo(big, 0.25, 1)},
        "dense": {"query": 0.0, "count": DENSE, "spacing": float(np.spacing(np.float32(dense.mean()))),
                  **search_demo(dense, 0.0, k)},
    }  # fmt: skip
    if (
        numbers["merge"]["float32_top"] == numbers["merge"]["exact"]
        or numbers["merge"]["fallback_rows"] != 0
    ):
        raise SystemExit(
            "The merge demonstration no longer shows a float32 miss repaired by the extra candidates."
        )
    c32 = (big[:2] - big.mean()).astype(np.float32)
    if c32[0] != c32[1]:
        raise SystemExit("0 and 1 no longer round to the same centred float32 number.")
    if (
        numbers["dense"]["fallback_rows"] != 1
        or numbers["dense"]["float32_top"] != numbers["dense"]["exact"]
    ):
        raise SystemExit(
            "The dense demonstration no longer fails the certificate on a right float32 list."
        )
    (mf.OUT / "apx-knn.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers))


if __name__ == "__main__":
    main()
