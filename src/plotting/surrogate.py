"""
Plots of trained surrogates: leaves and trust ellipsoids over the function, the
split trees (``track_trees``), error distributions, and problem-specific views (wave, near-centre and Hessian studies).

Put new plots of surrogates or predictions here, and plots of training profiles in
``src.plotting.profiling``. Plots should accept host or device arrays, return or save
a figure, and use ``src.plotting.style`` for a consistent look.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from src import Array, _to_device, _to_host, xp
from src.plotting.style import afinni_style, finish

if TYPE_CHECKING:
    from src.surrogate import AFINNISurrogate


def _host_domain(domain) -> tuple[float, float]:
    """``(lo, hi)`` as Python floats, so a CuPy ``domain`` cannot make NumPy return CuPy arrays to matplotlib."""
    lo, hi = (float(_to_host(v)) for v in domain)
    return lo, hi


def _add_trust_ellipses(ax: plt.Axes, centroids: np.ndarray, Q: np.ndarray) -> EllipseCollection:
    """Draw each 2-D trust region ``dx^T Q dx = 1`` around its centroid as one collection (fast for many leaves)."""
    eigvals, eigvecs = np.linalg.eigh(Q)  # ascending: column 1 is the short axis
    ellipses = EllipseCollection(
        widths=2.0 / np.sqrt(eigvals[:, 1]),  # full axis lengths of the boundary
        heights=2.0 / np.sqrt(eigvals[:, 0]),
        angles=np.degrees(np.arctan2(eigvecs[:, 1, 1], eigvecs[:, 0, 1])),
        units="xy",
        offsets=centroids,
        offset_transform=ax.transData,
        edgecolors="blue",
        facecolors="none",
        linewidths=0.5,
        alpha=0.7,
    )
    ax.add_collection(ellipses)
    return ellipses


def plot_adaptive_clustering_results(
    centroids: np.ndarray,
    Q: np.ndarray,
    domain: tuple[float, float],
    f_fn: Callable,
    img_fpath: Path,
    title: str | None = None,
    plot_in_log: bool = True,
    plot_f_abs: bool = False,
):
    """
    Save a 2-D figure of ``f_fn`` over ``domain`` x ``domain`` with the leaf centroids
    (left) and their trust ellipsoids ``dx^T Q dx = 1`` (right).
    """
    # Matplotlib needs host arrays; accept device (CuPy) inputs too.
    centroids, Q = _to_host(centroids), _to_host(Q)
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
    Z_grid = _to_host(f_fn(_to_device(grid_pts)).reshape(X_grid.shape))
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

    _add_trust_ellipses(axs[1], centroids, Q)

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


def plot_error_histogram(
    ground_truth: Array, predictions: Array, img_fpath: Path, rtol: float | None
):
    """Save a log-log histogram of pointwise relative errors (worst output component per point); ``rtol`` (if given) is marked with the share of points within it."""
    TINY_EPS = 1e-12
    img_title = img_fpath.stem

    # 1. Compute Element-wise Relative Error
    # Shape: (N, m) for vectors, (N,) for scalars
    element_rel_error = xp.abs(predictions - ground_truth) / (
        xp.abs(ground_truth) + TINY_EPS
    )

    # 2. Reduce to Point-wise Max (Worst Component)
    # If vector (ndim > 1), take max over components (axis=1)
    # This aligns with the "Pointwise Relative Error" definition where a point
    # is only good if ALL its components are good.
    if element_rel_error.ndim > 1:
        ptwise_rel_error = xp.max(element_rel_error, axis=1)
    else:
        ptwise_rel_error = element_rel_error

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # Extract data to CPU for plotting
    data_to_plot = _to_host(ptwise_rel_error)
    plt.hist(data_to_plot.flatten(), bins=50, log=True)

    if rtol is not None:
        perct_within_tol = 100.0 * xp.sum(ptwise_rel_error <= rtol) / ptwise_rel_error.shape[0]
        plt.axvline(
            x=rtol,
            color="r",
            linestyle="--",
            label=f"RTOL={rtol} (Good Points={perct_within_tol:.2f}%)",
        )
        plt.legend()

    plt.ylabel("Frequency (log scale)")
    plt.xlabel(r"Pointwise Max Relative Error (log scale)")
    plt.title(img_title)
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(img_fpath, dpi=300)
    plt.close()


def plot_cluster_projections_for_wave(clusters: Array, img_fpath: Path, dim: int):
    """Save centroids projected on the wave problem's direction q against their distance from span(q)."""
    from src.problems.wave import _get_random_direction
    q = _to_host(_get_random_direction(dim, clusters.dtype))
    q = q / np.linalg.norm(q)

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


def plot_wave_2d_errors(surrogate: AFINNISurrogate, domain, f_fn, img_fpath: Path):
    """Save the 2-D wave function, the surrogate and their error side by side."""

    # Generate grid of points
    centers = surrogate.X_src
    Nc, d = centers.shape
    _n = 500
    domain = _host_domain(domain)
    x = np.linspace(domain[0], domain[1], _n)
    y = np.linspace(domain[0], domain[1], _n)
    X_grid, Y_grid = np.meshgrid(x, y)
    # Evaluate function on grid
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    exact = _to_host(f_fn(_to_device(grid_pts)).reshape(X_grid.shape))
    # Evaluate surrogate
    coarse_factor = 50
    output = surrogate.predict(grid_pts, coarse_factor=coarse_factor)
    approx = _to_host(output[0]).reshape(X_grid.shape)
    trust = _to_host(output[1]).reshape(X_grid.shape)

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
    surrogate: AFINNISurrogate,
    domain,
    f_fn,
    img_fpath: Path,
    coarse_factor: int = 50,
    betas=(0.0, 1.0, 2.0, 3.0),
):
    """Save errors along lines parallel to the wave direction, offset by each beta along its normal."""
    centers = surrogate.X_src
    Nc, d = centers.shape

    # Random direction and its orthogonal complement (2D only)
    from src.problems.wave import _get_random_direction
    q = _get_random_direction(d, centers.dtype)
    e = np.zeros(d); e[0] = 1
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
        exact = _to_host(f_fn(_to_device(pts_valid)))

        # Evaluate surrogate
        output = surrogate.predict(pts_valid, coarse_factor=coarse_factor)
        approx = _to_host(output[0])

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
    surrogate: AFINNISurrogate,
    target,
    domain,
    bg_fn,
    img_fpath: Path,
    coarse_factor: int = 50,
    n_grid: int = 500,
    n_report: int = 10,
):
    """Save the leaves the two-stage search considers for ``target`` and print the closest ones."""
    centers = _to_host(surrogate.X_src)  # (Nc, d)
    Qs = _to_host(surrogate.Qs_src)  # (Nc, d, d)
    Nc, d = centers.shape

    target = np.asarray(_to_host(target)).reshape(1, d)

    # --- Coarse step: Euclidean nearest neighbors to target ---
    dists = np.linalg.norm(centers - target, axis=1)
    nn_idx = np.argsort(dists)[:coarse_factor]

    # --- Fine step: anisotropic quadratic form x^T Q x per candidate ---
    diffs = target - centers[nn_idx]         # (coarse_factor, d)
    Q_candidates = Qs[nn_idx]                # (coarse_factor, d, d)
    quad_vals = np.einsum("ni,nij,nj->n", diffs, Q_candidates, diffs)

    # --- Print smallest n_report quadratic-form values, with centroid locations ---
    sort_order = np.argsort(quad_vals)
    topk_local = sort_order[:n_report]
    topk_idx = nn_idx[topk_local]            # global indices into `centers`
    topk_vals = quad_vals[topk_local]
    topk_centers = centers[topk_idx]

    print(f"Smallest {n_report} values of x^T Q x (value, centroid location):")
    for rank, (val, loc) in enumerate(zip(topk_vals, topk_centers), start=1):
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
    field = _to_host(bg_fn(_to_device(grid_pts)).reshape(X_grid.shape))

    fig, ax = plt.subplots(figsize=(8, 8))
    cs = ax.contourf(X_grid, Y_grid, field, levels=20, cmap="viridis", alpha=0.7)
    fig.colorbar(cs, ax=ax)

    # --- Ellipses x^T Q x = 1 for the top-k candidates (centered at each candidate) ---
    ellipse_label_added = False
    for idx in topk_idx:
        c_i = centers[idx]
        Q_i = Qs[idx]
        diff_grid = grid_pts - c_i
        quad_field = np.einsum("ni,ij,nj->n", diff_grid, Q_i, diff_grid).reshape(X_grid.shape)
        ax.contour(
            X_grid, Y_grid, quad_field,
            levels=[1.0],
            colors="red",
            linestyles="dotted",
            linewidths=1.0,
            zorder=3.5,
        )
    # Dummy line for a single legend entry (contour objects don't auto-label)
    ax.plot([], [], color="red", linestyle="dotted", linewidth=1.0, label=r"$x^\top Q x = 1$")

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
    # Top-k smallest x^T Q x candidates (excluding the selected one)
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
        est_hess_surrogate: AFINNISurrogate,
        exact_hess_surrogate: AFINNISurrogate,
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
    field = _to_host(bg_fn(_to_device(grid_pts)).reshape(X_grid.shape))
    # Color background on each axis
    cs = axs[0].contourf(X_grid, Y_grid, field, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axs[0])
    cs = axs[1].contourf(X_grid, Y_grid, field, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axs[1])

    # Plot centroids on each axis
    centers1 = est_hess_surrogate.X_src; centers2 = exact_hess_surrogate.X_src
    Nc1, d1 = centers1.shape; Nc2, d2 = centers2.shape
    assert d1 == 2; assert d2 == 2;
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

        



def plot_split_trees(
    surrogate: AFINNISurrogate,
    domain,
    f_fn: Callable,
    path: str | Path | None = None,
    plot_in_log: bool = False,
    n_grid: int = 400,
) -> plt.Figure:
    """
    Draw the split trees of a surrogate trained with ``track_trees`` over the contours
    of ``f_fn`` on ``domain`` x ``domain`` (2-D only). Left: root centroids in blue,
    internal nodes below the roots in orange, accepted leaves as red crosses, and an
    edge from every node to its parent. Right: the leaves and their trust regions
    ``dx^T Q dx = 1``. A root's split sits on its root centroid, so it adds no point
    or edge of its own. Unresolved clusters (dropped at ``max_levels``) have no
    centroid and are left out.
    """
    forest = surrogate.tree_forest
    if forest is None:
        raise ValueError(
            "The surrogate has no split trees: train it with ClusteringConfig(track_trees=True)."
        )
    roots = _to_host(forest.root_centroids)
    if roots.shape[1] != 2:
        raise ValueError(f"plot_split_trees draws 2-D trees only; got d={roots.shape[1]}.")
    internal = {k: _to_host(v) for k, v in forest.internal_nodes.items()}
    leaves = {k: _to_host(v) for k, v in forest.leaf_nodes.items()}

    below_root = internal["parent"] >= 0
    accepted = leaves["leaf"] >= 0
    leaf_xy = _to_host(surrogate.X_src)[leaves["leaf"][accepted]]
    leaf_parent = leaves["parent"][accepted]
    child_xy = np.concatenate([internal["centroid"][below_root], leaf_xy[leaf_parent >= 0]])
    parent_rows = np.concatenate([internal["parent"][below_root], leaf_parent[leaf_parent >= 0]])
    edges = np.stack([internal["centroid"][parent_rows], child_xy], axis=1)  # (n_edges, 2, 2)

    lo, hi = _host_domain(domain)
    x = np.linspace(lo, hi, n_grid)
    X_grid, Y_grid = np.meshgrid(x, x)
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = _to_host(f_fn(_to_device(grid_pts))).reshape(X_grid.shape)
    leaf_style = dict(c="red", marker="x", s=22, linewidths=0.9, zorder=4)

    with afinni_style():
        fig, (ax_tree, ax_trust) = plt.subplots(1, 2, figsize=(15, 7))
        for ax in (ax_tree, ax_trust):
            cs = ax.contourf(
                X_grid, Y_grid, Z_grid, levels=50, cmap="viridis",
                norm=LogNorm() if plot_in_log else None,
            )  # fmt: skip
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.colorbar(cs, ax=[ax_tree, ax_trust], label="f", shrink=0.85)

        ax_tree.add_collection(LineCollection(edges, colors="white", linewidths=0.6, alpha=0.8, zorder=2))
        ax_tree.scatter(*internal["centroid"][below_root].T, c="orange", s=16, edgecolors="black",
                        linewidths=0.3, zorder=3,
                        label=f"Internal nodes ({int(below_root.sum()):,})")  # fmt: skip
        ax_tree.scatter(*leaf_xy.T, label=f"Leaves ({len(leaf_xy):,})", **leaf_style)
        ax_tree.scatter(*roots.T, c="blue", s=25, edgecolors="black", linewidths=0.5, zorder=5,
                        label=f"Roots ({len(roots):,})")  # fmt: skip
        ax_tree.set_title("Split trees")
        ax_tree.legend(loc="upper right", frameon=True)

        leaf_Q = _to_host(surrogate.Qs_src)[leaves["leaf"][accepted]]
        _add_trust_ellipses(ax_trust, leaf_xy, leaf_Q)
        ax_trust.scatter(*leaf_xy.T, **leaf_style)
        ax_trust.set_title("Leaves and trust regions")
        ax_trust.legend(
            handles=[
                Line2D([], [], color="red", marker="x", linestyle="none", label="Leaves"),
                Line2D([], [], color="blue", lw=0.5, label=r"Trust region $dx^\top Q\, dx = 1$"),
            ],
            loc="upper right", frameon=True,
        )  # fmt: skip
    return finish(fig, path)
