from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from src.surrogate import AFINNISurrogate
from src import Array, _to_device, _to_host, xp

_LEVEL_STAT_COL_LABELS: dict[str, str] = {
    "good_clusters": "Good Clusters (Accepted)",
    "clusters_start": "Clusters at Start of Level",
    "clusters_level_end": "Clusters at End of Level",
    "num_pts_next_level": "Points for Next Level",
    "size_min": "Min Cluster Size",
    "size_mean": "Mean Cluster Size",
    "size_median": "Median Cluster Size",
    "size_max": "Max Cluster Size",
    "runtime_sec": "Runtime (s)",
}


def plot_mpi_level_heatmap(
    results: dict[str, Any],
    img_fpath: Path,
    values_col: Literal[
        "good_clusters",
        "clusters_start",
        "clusters_level_end",
        "num_pts_next_level",
        "size_min",
        "size_mean",
        "size_median",
        "size_max",
        "runtime_sec",
    ] = "good_clusters",
    title: str | None = None,
) -> None:
    """
    Plots a heatmap of a per-rank, per-level statistic from ``level_stats``
    collected during adaptive clustering.

    Rows are BFS levels, columns are MPI ranks.  Cell colour and annotation
    reflect the chosen ``values_col``.  Ranks / levels that were idle at a
    given level are shown in white (zero).

    Args:
        results: Return value of ``adaptive_clustering`` (rank 0 only).
        img_fpath: Output path for the saved figure.
        values_col: Which field from each ``level_stats`` entry to visualise.
        title: Optional figure title.
    """
    profiling_stats = results.get("profiling_stats", {})
    level_stats: list[dict] = profiling_stats.get("level_stats", [])

    if not level_stats:
        print("plot_mpi_level_heatmap: no level_stats found, skipping plot.")
        return

    # Collect unique ranks and levels (sorted)
    ranks = sorted({e["rank"] for e in level_stats})
    levels = sorted({e["level"] for e in level_stats})

    # Build dense matrix: rows=levels, cols=ranks, default 0
    matrix = np.zeros((len(levels), len(ranks)), dtype=float)
    rank_idx = {r: i for i, r in enumerate(ranks)}
    level_idx = {l: i for i, l in enumerate(levels)}
    for entry in level_stats:
        r = rank_idx[entry["rank"]]
        l = level_idx[entry["level"]]
        matrix[l, r] = float(entry.get(values_col, 0) or 0)

    # Set annotation format
    fmt = ".1e"

    # Mask truly-zero cells so they render white
    mask = matrix == 0

    col_label = _LEVEL_STAT_COL_LABELS.get(values_col, values_col)
    row_labels = [f"Lvl {l}" for l in levels]
    col_labels = [str(r) for r in ranks]

    fig_h = max(6, len(levels) * 0.42)
    fig_w = max(6, len(ranks) * 0.72 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        matrix,
        mask=mask,
        ax=ax,
        cmap="Reds",
        annot=True,
        fmt=fmt,
        linewidths=0.4,
        linecolor="#e0e0e0",
        cbar_kws={"label": col_label, "shrink": 0.8},
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot_kws={"size": 8},
    )
    # Fill masked (zero) cells explicitly with white
    ax.set_facecolor("white")

    ax.set_xlabel("MPI Rank")
    ax.set_ylabel("Level")
    ax.tick_params(axis="both", labelsize=9)

    default_title = f"MPI Workload Heatmap by Level\n(White = 0  |  {col_label})"
    ax.set_title(title if title is not None else default_title, fontsize=11)

    plt.tight_layout()
    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MPI heatmap to {img_fpath}")


def plot_mpi_rank_time_breakdown(
    profiling_stats,
    img_fpath: Path,
    top_n_annotate=10,
    min_val_annotate=0.05,
    figsize=(16, 10),
    tick_fontsize=13,
    axis_title_fontsize=14,
    bottom_log_scale=False,
    bottom_ylim=None,
    top_to_bottom_ratio=4,
):
    """
    Plot a stacked 100% bar chart of time breakdown per MPI rank,
    with a bottom panel showing total train time per rank.

    Args:
        profiling_stats: Dict containing "per_rank" list of dicts with timing info.
        img_fpath: Path to save the resulting figure.
        top_n_annotate: Number of top segments (by proportion) to annotate per bar.
        min_val_annotate: Minimum proportion threshold for annotation.
        figsize: Figure size.
        tick_fontsize: Font size for tick labels.
        axis_title_fontsize: Font size for axis titles.
        bottom_log_scale: If True, use a log y-axis for the bottom subplot.
        bottom_ylim: Optional y-limits tuple for the bottom subplot.
        top_to_bottom_ratio: Height ratio between the top and bottom subplots.
            1 means equal height, 2 means the top subplot is twice as tall.
    """
    rename_props = {
        "Preprocessing": ["time_init"],
        "H2D: Level Transfer": ["time_transition", "time_slicing", "time_gc"],
        "Augment Points": ["time_find_req_synth", "time_synth_gen"],
        "Error Estimation": ["time_error_calc", "time_filter"],
        "Trust Region Calculation": ["time_good_Q"],
        "D2H: Accepted Leaves": ["time_good_d2h"],
        "Split Bad Clusters": ["time_bad_split"],
        "D2H: Split Clusters Buffer": ["time_bad_d2h"],
        "MPI Scatter (Init Wait)": ["time_scatter"],
        "MPI Gather": ["time_gather"],
        "MPI Idle": ["time_idle"],
        "MPI Stealing": ["time_state_update", "time_mpi_send", "time_mpi_recv"],
    }

    num_ranks = len(profiling_stats["per_rank"])

    # --- Build per-rank DataFrame ---
    rows = []
    for rank in range(num_ranks):
        rank_data = profiling_stats["per_rank"][rank]
        row = {"MPI Rank": rank}
        for category, props in rename_props.items():
            row[category] = sum(rank_data.get(p, 0.0) for p in props)
        # Total time from the raw profiling data
        row["Total Time (s)"] = rank_data.get("time_total", 0.0)
        rows.append(row)

    rank_df = pd.DataFrame(rows)

    # --- Proportional DataFrame for stacked bars ---
    bar_cols = list(rename_props.keys())
    plot_df = rank_df.set_index("MPI Rank")[bar_cols]
    plot_df_100 = plot_df.div(plot_df.sum(axis=1), axis=0)

    colors = sns.color_palette("tab20", n_colors=len(bar_cols))

    # --- Figure ---
    fig, axs = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [top_to_bottom_ratio, 1]},
    )
    plt.subplots_adjust(hspace=0, top=0.88)

    # ── TOP: Stacked 100% bar ──────────────────────────────────────────────
    ax_bar = axs[0]
    plot_df_100.plot(
        kind="bar",
        stacked=True,
        ax=ax_bar,
        color=colors,
        width=0.9,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
    )

    # Annotations
    ranks_order = plot_df_100.rank(axis=1, method="min", ascending=False)
    for col_idx, container in enumerate(ax_bar.containers):
        for row_idx, patch in enumerate(container):
            rank_val = int(ranks_order.iloc[row_idx, col_idx])  # type: ignore
            if rank_val <= top_n_annotate:
                val = float(plot_df_100.iloc[row_idx, col_idx])  # type: ignore
                if val < min_val_annotate:
                    continue
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_y() + patch.get_height() / 2
                ax_bar.text(
                    x,
                    y,
                    f"{val:.0}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                )

    ax_bar.set_ylabel("Proportion of Time", fontsize=axis_title_fontsize)
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bar.set_xlabel("")
    ax_bar.tick_params(axis="y", labelsize=tick_fontsize)

    # ── BOTTOM: Total time per rank ────────────────────────────────────────
    ax_line = axs[1]
    x_indices = np.arange(num_ranks)
    y_total = rank_df["Total Time (s)"].values

    sns.lineplot(
        x=x_indices,
        y=y_total,
        marker="o",
        ax=ax_line,
        color="#333333",
        linewidth=2,
        markersize=8,
    )
    for x, y in zip(x_indices, y_total):
        ax_line.text(
            x,
            y,
            f"{y:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    if bottom_log_scale:
        ax_line.set_yscale("log")
        if bottom_ylim is None:
            ax_line.set_ylim(min(y_total) * 0.3, max(y_total) * 2.5)
        else:
            ax_line.set_ylim(bottom_ylim)
    elif bottom_ylim is not None:
        ax_line.set_ylim(bottom_ylim)

    ax_line.set_ylabel("Total Time (s)", fontsize=axis_title_fontsize)
    ax_line.set_xlabel("MPI Rank", fontsize=axis_title_fontsize)
    ax_line.set_xticks(x_indices)
    ax_line.set_xticklabels([str(r) for r in rank_df["MPI Rank"]])
    ax_line.tick_params(axis="both", labelsize=tick_fontsize)
    ax_line.grid(True, which="both", linestyle="--", alpha=0.3)

    # ── Legend + Title ─────────────────────────────────────────────────────
    handles, labels = ax_bar.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=6,
        frameon=False,
        fontsize=12,
    )

    fig.savefig(img_fpath, bbox_inches="tight")


def plot_adaptive_clustering_results(
    centroids: np.ndarray,
    Q: np.ndarray,
    radii: np.ndarray,
    domain: tuple[float, float],
    f_fn: Callable,
    img_fpath: Path,
    title: str | None = None,
    plot_circle: bool = False,
    plot_in_log: bool = True,
    plot_f_abs: bool = False,
):
    """
    Plots the results of adaptive clustering on the given function and saves the figure
    to img_fpath.
    The function creates a figure with two subplots:
    1. A contour plot of the function with the final cluster centroids overlaid.
    2. A contour plot of the function with the final cluster centroids, trust ellipsoids,
       and circles corresponding to the trust region radii overlaid.

    Args:
        centroids: (N, 2) array of cluster centroids,
        Q: (N, 2, 2) array of precision matrices for each cluster,
        radii: (N,) array of trust region radii for each cluster,
        domain: A tuple (min, max) defining the range of x and y values for the plot.
        f_fn: A callable that takes an (N, 2) array of points and returns an (N,) array
            of function values.
        img_fpath: A Path object where the resulting figure will be saved.
        title: Optional string for the overall figure title. If None, a default title
            will be generated based on the number of clusters.
        plot_circle: Whether to plot circles for the trust region radii (default: False).
        plot_in_log: Whether to render contour color normalization on a log scale.
        plot_f_abs: Whether to take the absolute value of the function for plotting.
    """
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

    # Draw Ellipsoids and Circles
    for i in range(centroids.shape[0]):
        _centroid = centroids[i]
        _Q = Q[i]

        # Eigen-decomposition of Q
        eigvals, eigvecs = np.linalg.eigh(_Q)
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

        # Draw radius circle with dashed line
        if plot_circle:
            _radius = radii[i]
            circle = Circle(
                xy=_centroid,
                radius=_radius,
                edgecolor="black",
                facecolor="none",
                lw=0.5,
                alpha=0.7,
                linestyle="dashed",
            )
            axs[1].add_patch(circle)

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
    if plot_circle:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=0.5,
                linestyle="dashed",
                label="Trust Radius",
            )
        )
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
    ground_truth: Array, predictions: Array, img_fpath: Path, tol: float
):
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

    # 3. Compute Stats
    n_within_tol = xp.sum(ptwise_rel_error <= tol)
    perct_within_tol = 100.0 * n_within_tol / ptwise_rel_error.shape[0]

    # 4. Plotting
    plt.figure(figsize=(10, 6))

    # Extract data to CPU for plotting
    data_to_plot = _to_host(ptwise_rel_error)
    plt.hist(data_to_plot.flatten(), bins=50, log=True)

    plt.axvline(
        x=tol,
        color="r",
        linestyle="--",
        label=f"TOL={tol} (Good Points={perct_within_tol:.2f}%)",
    )

    plt.ylabel("Frequency (log scale)")
    plt.xlabel(r"Pointwise Max Relative Error (log scale)")
    plt.title(img_title)
    plt.legend()
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(img_fpath, dpi=300)
    plt.close()


def plot_cluster_projections_for_wave(clusters: Array, img_fpath: Path, dim: int):
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

    # Generate grid of points
    centers = surrogate.X_src
    Nc, d = centers.shape
    _n = 500
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
    centers = surrogate.X_src
    Nc, d = centers.shape

    # Random direction and its orthogonal complement (2D only)
    from src.problems.wave import _get_random_direction
    q = _get_random_direction(d, centers.dtype)
    e = np.zeros(d); e[0] = 1
    q_perp = e - np.dot(q, e) * q
    q_perp /= np.linalg.norm(q_perp)

    # Alpha sweep along the domain
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
    centers = surrogate.X_src      # (Nc, d)
    Qs = surrogate.Qs_src          # (Nc, d, d)
    Nc, d = centers.shape

    target = np.asarray(target).reshape(1, d)

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
