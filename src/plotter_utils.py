from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

from src import _to_device, _to_host

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
    figsize=(12, 12),
    tick_fontsize=13,
    axis_title_fontsize=14,
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
        "MPI Stealing": ["time_heartbeat", "time_mpi_send", "time_mpi_recv"],
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
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.8, 1]}
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
                    f"{val:.0%}",
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

    ax_line.set_yscale("log")
    ax_line.set_ylim(min(y_total) * 0.3, max(y_total) * 2.5)
    ax_line.set_ylabel("Total Train (s)", fontsize=axis_title_fontsize)
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
    # Save figure
    plt.savefig(img_fpath, dpi=200, bbox_inches="tight")


def plot_adaptive_clustering_results(
    results: dict[str, Any],
    domain: tuple[float, float],
    f_fn: Callable,
    img_fpath: Path,
    title: str | None = None,
):
    """
    Plots the results of adaptive clustering on the given function and saves the figure
    to img_fpath.
    The function creates a figure with two subplots:
    1. A contour plot of the function with the final cluster centroids overlaid.
    2. A contour plot of the function with the final cluster centroids, trust ellipsoids,
       and circles corresponding to the trust region radii overlaid.

    Args:
        results: A dictionary containing the results of adaptive clustering, including:
            - "centroids": (N, 2) array of cluster centroids,
            - "Q": (N, 2, 2) array of precision matrices for each cluster,
            - "radii": (N,) array of trust region radii for each cluster.
        domain: A tuple (min, max) defining the range of x and y values for the plot.
        f_fn: A callable that takes an (N, 2) array of points and returns an (N,) array
            of function values.
        img_fpath: A Path object where the resulting figure will be saved.
        title: Optional string for the overall figure title. If None, a default title
            will be generated based on the number of clusters.
    """
    # Get dimensions of X
    d = results["centroids"].shape[1]
    if d != 2:
        print(
            f"Warning: Plotting is designed for 2D functions, but centroids have dimension {d}. Skipping plot."
        )
        return

    # Move all data to host for plotting
    results = {k: _to_host(v) for k, v in results.items()}

    # Create grid for contour plot
    _n = 400
    x = np.linspace(domain[0], domain[1], _n)
    y = np.linspace(domain[0], domain[1], _n)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Evaluate function on grid
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = _to_host(f_fn(_to_device(grid_pts)).reshape(X_grid.shape))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # -- Subplot 1: Contours + Centroids --
    cs = axs[0].contourf(
        X_grid,
        Y_grid,
        Z_grid,
        levels=50,
        cmap="viridis",
        norm=LogNorm(),
    )
    fig.colorbar(cs, ax=axs[0])
    axs[0].set_title("Function Contour")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal")
    axs[0].scatter(
        results["centroids"][:, 0],
        results["centroids"][:, 1],
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
        norm=LogNorm(),
    )
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title("Adaptive Clustering with Trust Ellipsoids")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_aspect("equal")

    # Plot Centroids
    axs[1].scatter(
        results["centroids"][:, 0],
        results["centroids"][:, 1],
        c="red",
        s=5,
        zorder=10,
    )

    # Draw Ellipsoids and Circles
    centroids = results["centroids"]
    Qs = results["Q"]
    radii = results["radii"]

    for i in range(centroids.shape[0]):
        center = centroids[i]
        Q = Qs[i]
        radius = radii[i]

        # Eigen-decomposition of Q
        eigvals, eigvecs = np.linalg.eigh(Q)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        a, b = 2.0 * np.sqrt(1.0 / eigvals)  # 2-sigma lengths
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = Ellipse(
            xy=center,
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
        circle = Circle(
            xy=center,
            radius=radius,
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
        Line2D(
            [0],
            [0],
            color="black",
            lw=0.5,
            linestyle="dashed",
            label="Trust Radius",
        ),
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
