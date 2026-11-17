"""
Plots of trained surrogates: leaves and trust ellipsoids over the function, error
distributions, and problem-specific views (wave, near-centre and Hessian studies).

Put new plots of surrogates or predictions here, and plots of training profiles in
``src.plotting.profiling``. Plots should accept host or device arrays, return or save
a figure, and use ``src.plotting.style`` for a consistent look.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from src import Array, to_device, to_host
from src.plotting.style import finish, plot_style

if TYPE_CHECKING:
    from src.surrogate import AdaptiveSurrogate


def _host_domain(domain) -> tuple[float, float]:
    """``(lo, hi)`` as Python floats, so a CuPy ``domain`` cannot make NumPy return CuPy arrays to matplotlib."""
    lo, hi = (float(to_host(v)) for v in domain)
    return lo, hi


def plot_adaptive_clustering_results(
    centroids: np.ndarray,
    trust_matrices: np.ndarray,
    domain: tuple[float, float],
    f_fn: Callable,
    img_fpath: Path,
    title: str | None = None,
    plot_in_log: bool = True,
    plot_f_abs: bool = False,
):
    """
    Save a 2-D figure of ``f_fn`` over ``domain`` x ``domain`` with the leaf centroids
    (left) and their trust ellipsoids ``dx^T trust_matrix dx = 1`` (right).
    """
    # Matplotlib needs host arrays; accept device (CuPy) inputs too.
    centroids, trust_matrices = to_host(centroids), to_host(trust_matrices)
    domain = _host_domain(domain)

    # Get dimensions of X
    d = centroids.shape[1]
    if d != 2:
        print(
            f"Warning: Plotting is designed for 2D functions, but centroids have dimension {d}. Skipping plot."
        )
        return

    # Create grid for contour plot
    _n = 400
    x = np.linspace(domain[0], domain[1], _n)
    y = np.linspace(domain[0], domain[1], _n)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Evaluate function on grid
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = to_host(f_fn(to_device(grid_pts)).reshape(X_grid.shape))
    if plot_f_abs:
        Z_grid = np.abs(Z_grid)
    cbar_norm = LogNorm() if plot_in_log else None

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # -- Subplot 1: Contours + Centroids --
    cs = axs[0].contourf(
        X_grid,
        Y_grid,
        Z_grid,
        levels=50,
        cmap="viridis",
        norm=cbar_norm,
    )
    fig.colorbar(cs, ax=axs[0])
    axs[0].set_title("Function Contour")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal")
    axs[0].scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=5,
        label="Centroids",
    )

    # -- Subplot 2: Contours + OBBs --
    cs2 = axs[1].contourf(
        X_grid,
        Y_grid,
        Z_grid,
        levels=50,
        cmap="viridis",
        norm=cbar_norm,
    )
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title("Adaptive Clustering with Trust Ellipsoids")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_aspect("equal")

    # Plot Centroids
    axs[1].scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=5,
        zorder=10,
    )

    # Draw Ellipsoids and Circles
    for i in range(centroids.shape[0]):
        _centroid = centroids[i]
        # Eigen-decomposition of the leaf's trust matrix
        eigvals, eigvecs = np.linalg.eigh(trust_matrices[i])
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        a, b = 2.0 * np.sqrt(1.0 / eigvals)  # 2-sigma lengths
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = Ellipse(
            xy=_centroid,
            width=a,
            height=b,
            angle=angle,
            edgecolor="blue",
            facecolor="none",
            lw=0.5,
            alpha=0.7,
        )
        axs[1].add_patch(ellipse)

    # Update legend to include ellipses and circles
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Centroids",
            markerfacecolor="red",
            markersize=5,
        ),
        Line2D([0], [0], color="blue", lw=0.5, label="Trust Ellipsoid"),
    ]
    axs[1].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if title is not None:
        plt.suptitle(title, fontsize=11)
    else:
        plt.suptitle(
            f"Adaptive Clustering (N_CLUSTERS={len(centroids):,})",
            fontsize=11,
        )
    plt.subplots_adjust(top=0.88)

    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {img_fpath}")


# Cases ordered by d or tolerance take steps of one blue ramp, light to dark; the shares of the test
# points take the first four categorical slots in their fixed order, each with its own marker.
CASE_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b")  # fmt: skip
INK, MUTED = "#0b0b0b", "#52514e"
SHARE_STYLES = {
    "trusted, within tolerance": ("#2a78d6", "o"),
    "trusted, beyond tolerance": ("#eb6834", "s"),
    "flagged, inside a leaf": ("#1baf7a", "^"),
    "flagged, outside every leaf": ("#eda100", "D"),
    "flagged": ("#1baf7a", "^"),
}


def _case_labels(results: Sequence[Mapping]) -> list[str]:
    """``d = 5`` when the cases differ in d, ``tol = 0.1`` when in tolerance, both when in both."""
    dims = [r["dim"] for r in results]
    tols = [r["atol"] if r.get("atol") is not None else r.get("rtol") for r in results]
    parts = [[] for _ in results]
    if len(set(dims)) > 1 or len(results) == 1:
        for p, d in zip(parts, dims, strict=True):
            p.append(f"d = {d}")
    if len(set(tols)) > 1 or len(results) == 1:
        for p, tol in zip(parts, tols, strict=True):
            p.append(f"tol = {tol:g}")
    return [", ".join(p) or f"case {i}" for i, p in enumerate(parts)]


def _test_point_shares(result: Mapping) -> dict[str, float]:
    """Percentages of the test points by trust and tolerance; flagged points split by the exact flag when recorded."""
    m = result["error_metrics"]
    shares = {"trusted, within tolerance": m["perct_in_good"], "trusted, beyond tolerance": m["perct_in_bad"]}
    exact = result.get("perct_out_exact")
    if exact is None:
        shares["flagged"] = m["perct_out"]
    else:
        shares["flagged, inside a leaf"] = max(m["perct_out"] - exact, 0.0)
        shares["flagged, outside every leaf"] = exact
    return shares


def _draw_ratio_survival(ax, counts, edges, label: str, **line) -> None:
    """
    Share of points (%) whose tolerance ratio is at least x, from its histogram, labelled with the
    point count; nothing for no points. The curve starts at the second edge: the first bin also holds
    every ratio below the range, so the share at the first edge is 100% by construction.
    """
    counts = np.asarray(counts, float)
    n = int(counts.sum())
    if n == 0:
        return
    share = 100.0 * np.cumsum(counts[::-1])[::-1] / n
    x, share = np.asarray(edges[1:-1]), share[1:]
    keep = share > 0
    ax.plot(x[keep], share[keep], label=f"{label} (n = {n:,})", **line)


def _ratio_axes(ax, title: str) -> None:
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("tolerance", (1.0, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=MUTED)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("error / tolerance  (the acceptance test passes at <= 1)")
    ax.set_ylabel("points with at least this error (%)")
    ax.set_title(title, loc="left")


def _draw_shares(ax, results: Sequence[Mapping], labels: list[str]) -> None:
    """One row per case: each share of the test points as a marker on a log axis (zero shares are omitted)."""
    seen = set()
    for row, result in enumerate(results):
        for name, value in _test_point_shares(result).items():
            if value > 0:
                colour, marker = SHARE_STYLES[name]
                ax.plot(value, row, marker, color=colour, ms=8, mec="white", mew=1.0,
                        label=None if name in seen else name)  # fmt: skip
                seen.add(name)
    ax.set_yticks(range(len(results)), labels)
    ax.set_ylim(len(results) - 0.5, -0.5)
    ax.set_xscale("log")
    ax.set_xlim(right=150)
    ax.set_xlabel("share of the test points (%)")
    ax.set_title("Where the test points fall", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)


def _draw_count(ax, results: Sequence[Mapping], labels: list[str], key: str, title: str) -> None:
    values = [float(r[key]) if r.get(key) is not None else np.nan for r in results]
    ax.plot(range(len(values)), values, "-", color=MUTED, linewidth=1, zorder=1)
    for i, v in enumerate(values):
        ax.plot(i, v, "o", color=CASE_RAMP[_ramp_index(i, len(values))], ms=8, mec="white", mew=1.0, zorder=2)
    ax.set_xticks(range(len(values)), labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_title(title, loc="left")


def _ramp_index(i: int, n: int) -> int:
    return round(i * (len(CASE_RAMP) - 1) / max(n - 1, 1)) if n > 1 else 4


def plot_error_summary(results: Sequence[Mapping], path=None, *, title: str | None = None) -> plt.Figure:
    """
    The errors of one or more launcher results (their JSON dicts), in units of the tolerance: for
    each case the share of trusted and of flagged test points whose error is at least x tolerances
    (from ``error_metrics["tolerance_ratio"]``), where the test points fall (trusted within or beyond
    tolerance, flagged inside a leaf or outside every leaf), and, for several cases, the leaves N_c
    and oracle evaluations N. Cases are drawn in the given order, labelled by what differs (d or
    tolerance). Results without the histogram still get the share and count panels.
    """
    results = list(results)
    labels = _case_labels(results)
    hist = [r["error_metrics"].get("tolerance_ratio") for r in results]
    with plot_style():
        if len(results) == 1:
            fig, (ax_dist, ax_share) = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=(1.4, 1))
            if hist[0] is not None:
                for key, name, (colour, _) in (("in", "trusted", SHARE_STYLES["trusted, within tolerance"]),
                                               ("out", "flagged", SHARE_STYLES["trusted, beyond tolerance"])):  # fmt: skip
                    _draw_ratio_survival(ax_dist, hist[0][key], hist[0]["edges"], name, color=colour, linewidth=2)
                ax_dist.legend(loc="lower left")
            _ratio_axes(ax_dist, "Error distribution")
        else:
            fig = plt.figure(figsize=(13, 8))
            grid = fig.add_gridspec(2, 6)
            ax_in, ax_out = fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])
            ax_share = fig.add_subplot(grid[1, :2])
            ax_nc, ax_n = fig.add_subplot(grid[1, 2:4]), fig.add_subplot(grid[1, 4:])
            for ax, key, name in ((ax_in, "in", "Trusted points"), (ax_out, "out", "Flagged points")):
                for i, (h, label) in enumerate(zip(hist, labels, strict=True)):
                    if h is not None:
                        _draw_ratio_survival(ax, h[key], h["edges"], label,
                                             color=CASE_RAMP[_ramp_index(i, len(results))], linewidth=2)  # fmt: skip
                _ratio_axes(ax, name)
                ax.legend(loc="lower left")
            _draw_count(ax_nc, results, labels, "N_c", "Leaves N_c")
            _draw_count(ax_n, results, labels, "N", "Oracle evaluations N")
        _draw_shares(ax_share, results, labels)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return finish(fig, path)


def plot_cluster_projections_for_wave(clusters: Array, img_fpath: Path, dim: int):
    """Save centroids projected on the wave problem's direction q against their distance from span(q)."""
    from src.problems.wave import get_random_direction
    q = to_host(get_random_direction(dim, clusters.dtype))
    q = q / np.linalg.norm(q)
    clusters = to_host(clusters)

    # Project onto span(q) and span(q)^(perp)
    q_proj = clusters @ q
    residual = clusters - np.outer(q_proj, q)
    q_perp = np.linalg.norm(residual, axis=1)

    # Plot scatter plot
    plt.figure(figsize=(10,8))
    plt.xlabel("$||proj_q c||$")
    plt.ylabel("$||proj_{q^\\perp} c||$")
    plt.title("Projections of centers onto wave direction")
    plt.scatter(q_proj, q_perp)
    plt.savefig(img_fpath, dpi=300)
    plt.close()


def plot_wave_2d_errors(surrogate: AdaptiveSurrogate, domain, f_fn, img_fpath: Path):
    """Save the 2-D wave function, the surrogate and their error side by side."""

    # Generate grid of points
    centers = to_host(surrogate.X_src)
    Nc, d = centers.shape
    _n = 500
    domain = _host_domain(domain)
    x = np.linspace(domain[0], domain[1], _n)
    y = np.linspace(domain[0], domain[1], _n)
    X_grid, Y_grid = np.meshgrid(x, y)
    # Evaluate function on grid
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    exact = to_host(f_fn(to_device(grid_pts)).reshape(X_grid.shape))
    # Evaluate surrogate
    coarse_factor = 50
    output = surrogate.predict(grid_pts, coarse_factor=coarse_factor)
    approx = to_host(output[0]).reshape(X_grid.shape)
    trust = to_host(output[1]).reshape(X_grid.shape)

    # Compute error
    error = np.abs(exact - approx) + 1e-12

    # Plot errors
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # -- Subplot 1: Exact function --
    cs = axs[0,0].contourf(X_grid, Y_grid, exact, cmap="viridis", levels=20)
    fig.colorbar(cs, ax=axs[0,0])
    axs[0,0].set_title("Exact Function")
    axs[0,0].set_xlabel("x")
    axs[0,0].set_ylabel("y")
    axs[0,0].set_aspect("equal")

    # -- Subplot 2: Surrogate --
    cs2 = axs[0,1].contourf(X_grid, Y_grid, approx, cmap="viridis", levels=20)
    fig.colorbar(cs2, ax=axs[0,1])
    axs[0,1].scatter(centers[:, 0], centers[:, 1], c="red", s=5, label="Centroids")
    axs[0,1].set_title("AFINNI Surrogate")
    axs[0,1].set_xlabel("x")
    axs[0,1].set_ylabel("y")
    axs[0,1].set_aspect("equal")

    # -- Subplot 3: Log Errors --
    cs3 = axs[1,0].contourf(X_grid, Y_grid, np.log10(error), levels=10, cmap="viridis")
    fig.colorbar(cs3, ax=axs[1,0])
    axs[1,0].set_title("Log errors")
    axs[1,0].set_xlabel("x")
    axs[1,0].set_ylabel("y")
    axs[1,0].set_aspect("equal")

    # -- Subplot 4: Trust Regions
    binary_cmap = ListedColormap(["green", "red"])
    axs[1,1].contourf(X_grid, Y_grid, trust.astype(int), cmap=binary_cmap)
    axs[1,1].set_title("Trust Regions")
    axs[1,1].set_xlabel("x")
    axs[1,1].set_ylabel("y")
    axs[1,1].set_aspect("equal")

    plt.tight_layout()

    plt.suptitle(f"$f(x) = cos(2q^\\top x)$, $N_c={Nc}$", fontsize=11)

    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {img_fpath}")


def plot_wave_line_errors(
    surrogate: AdaptiveSurrogate,
    domain,
    f_fn,
    img_fpath: Path,
    coarse_factor: int = 50,
    betas=(0.0, 1.0, 2.0, 3.0),
):
    """Save errors along lines parallel to the wave direction, offset by each beta along its normal."""
    centers = to_host(surrogate.X_src)
    Nc, d = centers.shape

    # Random direction and its orthogonal complement (2D only)
    from src.problems.wave import get_random_direction
    q = get_random_direction(d, centers.dtype)
    e = np.zeros(d)
    e[0] = 1
    q_perp = e - np.dot(q, e) * q
    q_perp /= np.linalg.norm(q_perp)

    # Alpha sweep along the domain
    domain = _host_domain(domain)
    alpha = np.linspace(domain[0], domain[1], 2000)

    # Projection of centers onto q (used to scatter them on the x-axis)
    centers_proj = centers @ q

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.ravel()

    for i, beta in enumerate(betas):
        # Points along alpha*q + beta*q_perp
        pts = alpha[:, None] * q[None, :] + beta * q_perp[None, :]

        # Keep only points inside DOMAIN (all coordinates within bounds)
        in_domain = np.all((pts >= domain[0]) & (pts <= domain[1]), axis=1)
        alpha_valid = alpha[in_domain]
        pts_valid = pts[in_domain]

        ax = axs[i]

        if pts_valid.size == 0:
            ax.set_title(f"$\\beta = {beta}$ (no points in domain)")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel("Error")
            continue

        # Evaluate exact function
        exact = to_host(f_fn(to_device(pts_valid)))

        # Evaluate surrogate
        output = surrogate.predict(pts_valid, coarse_factor=coarse_factor)
        approx = to_host(output[0])

        # Compute error
        error = np.abs(exact - approx) + 1e-12

        ax.semilogy(alpha_valid, error, color="tab:blue")

        # Scatter centers' projections onto q, along the x-axis
        # (only those whose corresponding line point is in-domain)
        centers_proj_valid = centers_proj[
            (centers_proj >= alpha_valid.min()) & (centers_proj <= alpha_valid.max())
        ] if alpha_valid.size > 0 else centers_proj

        ax.scatter(
            centers_proj_valid,
            np.zeros_like(centers_proj_valid),
            transform=ax.get_xaxis_transform(),
            marker="|",
            s=100,
            color="red",
            clip_on=False,
            label="Centroid projections",
        )

        ax.set_title(f"$\\beta = {beta}$")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Error")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    axs[0].legend(loc="upper right")

    plt.tight_layout()
    plt.suptitle(
        f"$f(x) = \\cos(2q^\\top x)$, $N_c={Nc}$, line errors along $\\alpha q + \\beta q_\\perp$",
        fontsize=11,
    )

    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {img_fpath}")



def plot_near_centers(
    surrogate: AdaptiveSurrogate,
    target,
    domain,
    bg_fn,
    img_fpath: Path,
    coarse_factor: int = 50,
    n_grid: int = 500,
    n_report: int = 10,
):
    """Save the leaves the two-stage search considers for ``target`` and print the closest ones."""
    centers = to_host(surrogate.X_src)  # (Nc, d)
    trust_matrices = to_host(surrogate.trust_matrices_src)  # (Nc, d, d)
    Nc, d = centers.shape

    target = np.asarray(to_host(target)).reshape(1, d)

    # --- Coarse step: Euclidean nearest neighbors to target ---
    dists = np.linalg.norm(centers - target, axis=1)
    nn_idx = np.argsort(dists)[:coarse_factor]

    # --- Fine step: anisotropic quadratic form dx^T trust_matrix dx per candidate ---
    diffs = target - centers[nn_idx]         # (coarse_factor, d)
    trust_matrices_candidates = trust_matrices[nn_idx]  # (coarse_factor, d, d)
    quad_vals = np.einsum("ni,nij,nj->n", diffs, trust_matrices_candidates, diffs)

    # --- Print smallest n_report quadratic-form values, with centroid locations ---
    sort_order = np.argsort(quad_vals)
    topk_local = sort_order[:n_report]
    topk_idx = nn_idx[topk_local]            # global indices into `centers`
    topk_vals = quad_vals[topk_local]
    topk_centers = centers[topk_idx]

    print(f"Smallest {n_report} values of x^T T x (value, centroid location):")
    for rank, (val, loc) in enumerate(zip(topk_vals, topk_centers, strict=True), start=1):
        loc_str = ", ".join(f"{c:.5f}" for c in loc)
        print(f"  {rank}: {val:.5f}  @ ({loc_str})")

    # --- Plotting only supported in 2D ---
    if d != 2:
        print(f"Skipping plot: plot_near_centers only supports d=2 (got d={d}).")
        return

    # Select the single best candidate under this metric
    selected_idx = topk_idx[0]
    center_sel = centers[selected_idx]

    # --- Masks for plotting ---
    is_nn = np.zeros(Nc, dtype=bool)
    is_nn[nn_idx] = True
    is_selected = np.zeros(Nc, dtype=bool)
    is_selected[selected_idx] = True
    is_topk = np.zeros(Nc, dtype=bool)
    is_topk[topk_idx] = True

    is_topk_rest = is_topk & ~is_selected          # ranks 2..n_report
    is_other_nn = is_nn & ~is_topk                  # remaining NN candidates
    is_not_nn = ~is_nn

    # --- Background contour: user-provided function over the grid ---
    domain = _host_domain(domain)
    x = np.linspace(domain[0], domain[1], n_grid)
    y = np.linspace(domain[0], domain[1], n_grid)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    field = to_host(bg_fn(to_device(grid_pts)).reshape(X_grid.shape))

    fig, ax = plt.subplots(figsize=(8, 8))
    cs = ax.contourf(X_grid, Y_grid, field, levels=20, cmap="viridis", alpha=0.7)
    fig.colorbar(cs, ax=ax)

    # --- Ellipses x^T T x = 1 for the top-k candidates (centered at each candidate) ---
    for idx in topk_idx:
        c_i = centers[idx]
        diff_grid = grid_pts - c_i
        quad_field = np.einsum("ni,ij,nj->n", diff_grid, trust_matrices[idx], diff_grid).reshape(X_grid.shape)
        ax.contour(
            X_grid, Y_grid, quad_field,
            levels=[1.0],
            colors="red",
            linestyles="dotted",
            linewidths=1.0,
            zorder=3.5,
        )
    # Dummy line for a single legend entry (contour objects don't auto-label)
    ax.plot([], [], color="red", linestyle="dotted", linewidth=1.0, label=r"$x^\top T x = 1$")

    # Centers not among the coarse_factor nearest neighbors
    ax.scatter(
        centers[is_not_nn, 0], centers[is_not_nn, 1],
        c="pink", s=5, label="Not selected (coarse)", zorder=1,
    )
    # Centers among the coarse_factor nearest neighbors (excluding top-k)
    ax.scatter(
        centers[is_other_nn, 0], centers[is_other_nn, 1],
        c="red", s=20, label=f"{coarse_factor}-NN candidates", zorder=2,
    )
    # Top-k smallest x^T T x candidates (excluding the selected one)
    ax.scatter(
        centers[is_topk_rest, 0], centers[is_topk_rest, 1],
        c="blue", s=35, edgecolors="black", linewidths=0.5,
        label=f"Top-{n_report} candidates", zorder=4,
    )
    # The single center ultimately selected
    ax.scatter(
        center_sel[0], center_sel[1],
        c="orange", s=80, edgecolors="black", label="Selected center", zorder=5,
    )

    # Target point
    ax.scatter(
        target[0, 0], target[0, 1],
        marker="x", c="black", s=120, linewidths=3, label="Target", zorder=6,
    )

    ax.set_title(f"Nearest-center selection (coarse_factor={coarse_factor})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {img_fpath}")


def plot_exact_hess_vs_estimated_hess(
        est_hess_surrogate: AdaptiveSurrogate,
        exact_hess_surrogate: AdaptiveSurrogate,
        bg_fn, domain, img_fpath
    ):
    """Save the leaves of two surrogates trained with estimated and exact Hessians side by side."""

    # Compute background
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    n_grid = 500
    domain = _host_domain(domain)
    x = np.linspace(domain[0], domain[1], n_grid)
    y = np.linspace(domain[0], domain[1], n_grid)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    field = to_host(bg_fn(to_device(grid_pts)).reshape(X_grid.shape))
    # Color background on each axis
    cs = axs[0].contourf(X_grid, Y_grid, field, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axs[0])
    cs = axs[1].contourf(X_grid, Y_grid, field, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axs[1])

    # Plot centroids on each axis
    centers1, centers2 = to_host(est_hess_surrogate.X_src), to_host(exact_hess_surrogate.X_src)
    assert centers1.shape[1] == 2 and centers2.shape[1] == 2
    Nc1, Nc2 = centers1.shape[0], centers2.shape[0]
    axs[0].scatter(centers1[:, 0], centers1[:, 1], c="red", s=20, label="Centers", marker='^')
    axs[1].scatter(centers2[:, 0], centers2[:, 1], c="red", s=20, label="Centers", marker='o')

    # Other specifications
    axs[0].set_title(f"Hessian approx. with LSQ gradient ($N_c = {Nc1}$)")
    axs[1].set_title(f"Hessian approx. with exact gradient ($N_c = {Nc2}$)")
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')


    # Save
    plt.tight_layout()
    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {img_fpath}")



