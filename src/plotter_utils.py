from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

from src import _to_device, _to_host


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
