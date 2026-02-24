from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
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
