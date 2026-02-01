import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import Array, xp
from src.v1_adaptive_clustering import AdaptiveClusterer


class AnisotropicQuadratic:
    def __init__(self, ndim: int, steepness: float = 100.0):
        # Diagonal matrix D: [steep, 1, steep, 1, ...]
        self.D = xp.ones((ndim,), dtype=np.float64)
        self.D[::2] = steepness
        self.D = self.D.reshape(1, 1, -1)  # Shape for broadcasting (1, 1, d)

        # True Lipschitz Constant of the gradient is the max eigenvalue of Hessian
        # Hessian = 2 * D
        self.true_lipschitz = 2.0 * steepness

    def f(self, X: Array) -> Array:
        # f(x) = x^T D x
        quad_term = xp.sum((X**2) * self.D.reshape(1, -1), axis=1, keepdims=True)
        return quad_term  # (N, 1)

    def grad(self, X: Array) -> Array:
        # grad(x) = 2 D x
        G = 2.0 * X * self.D.reshape(1, -1)  # (N, d)
        return G[:, None, :]  # (N, 1, d)


def benchmark_hessian_estimation():
    # 1. Setup Data
    N = 10_000
    D = 2  # Keep 2D for easy visualization
    SEED = 0
    TOL = 0.1
    STEEPNESS = 50 * 1000

    # Create Anisotropic Function (Steepness 50 vs 1)
    func_obj = AnisotropicQuadratic(ndim=D, steepness=STEEPNESS)

    # Generate Random Data
    np.random.seed(SEED)
    X = np.random.uniform(-2, 2, size=(N, D))

    # Configurations to test: (Error Metric, Use Hessian Split)
    # Now testing L2, Linf-global, and Linf-local
    configs = [
        ("L2", False),
        ("L2", True),
        ("Linf-global", False),
        ("Linf-global", True),
        ("Linf-local", False),
        ("Linf-local", True),
    ]

    results = []
    summary_results = []
    print(
        f"Benchmarking Hessian Estimation on Anisotropic Quadratic Function (True L = {func_obj.true_lipschitz}, Tol = {TOL})"
    )
    print(
        f"{'Error Metric':<15} | {'Split Method':<15} | {'Clusters':<10} | {'Est. Lipschitz':<15} | {'Bias':<10}"
    )
    print("-" * 80)

    # Setup Visualization Figure: 3 rows (criteria) x 2 columns (split method)
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    axes_flat = axes.flatten()

    # Pre-compute Contour Grid
    grid_res = 200
    x_lin = np.linspace(-2, 2, grid_res)
    y_lin = np.linspace(-2, 2, grid_res)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_pts = np.c_[xx.ravel(), yy.ravel()]

    # Evaluate function on grid for background
    Z_vals = func_obj.f(grid_pts)
    if hasattr(Z_vals, "get"):
        Z_vals = Z_vals.get()
    Z = Z_vals.reshape(xx.shape)

    # Variable to store the contour plot for the colorbar
    last_cp = None

    for idx, (error_metric, use_hessian) in enumerate(configs):
        # 2. Run Clustering
        clusterer = AdaptiveClusterer(
            tol=TOL,
            min_points=10,
            use_hessian_informed_split=use_hessian,
            should_generate_extra_points=True,
            seed=SEED,
        )

        clusters, _ = clusterer.fit(
            X, func_obj.f, func_obj.grad, cupy_f=None, error_metric=error_metric  # type: ignore
        )

        # 3. Analyze Estimation Quality
        lipschitz_estimates = []
        centroids_list = []
        radii_list = []

        for c in clusters:
            centroids_list.append(c.centroid)
            radii_list.append(c.radius)

            if c.radius < 1e-9:
                continue

            lipschitz_estimates.append(c.lipschitz_obs)

            results.append(
                {
                    "Error Metric": error_metric,
                    "Hessian Split": use_hessian,
                    "Estimated_L": c.lipschitz_obs,
                    "True_L": func_obj.true_lipschitz,
                    "Error_Type": error_metric,  # Use detailed criterion name
                }
            )

        lipschitz_estimates = np.array(lipschitz_estimates)
        centroids_arr = np.array(centroids_list)
        radii_arr = np.array(radii_list)

        # Calculate mean and bias
        lip_mean = lipschitz_estimates.mean() if len(lipschitz_estimates) > 0 else 0.0
        bias = lip_mean / func_obj.true_lipschitz if lip_mean > 0 else 0.0

        # Print Summary stats
        split_name = "Hessian" if use_hessian else "Geometric"
        print(
            f"{error_metric:<15} | {split_name:<15} | {len(clusters):<10} | {lip_mean:.2f}           | {bias:.2f}"
        )

        # Store summary results
        summary_results.append(
            {
                "Tol": TOL,
                "Error Metric": error_metric,
                "Hessian Split": use_hessian,
                "N_Clusters": len(clusters),
                "Mean_Estimated_L": lip_mean,
                "True_L": func_obj.true_lipschitz,
                "Bias": bias,
            }
        )

        # 4. Plotting on Subplot
        ax = axes_flat[idx]

        # Plot contours
        last_cp = ax.contourf(xx, yy, Z, levels=30, cmap="viridis", alpha=0.8)

        # Plot Trust Circles
        for i in range(len(centroids_arr)):
            circle = patches.Circle(
                xy=(centroids_arr[i, 0], centroids_arr[i, 1]),
                radius=radii_arr[i],
                linewidth=1,
                edgecolor="red",
                facecolor="none",
                alpha=0.6,  # Slightly more transparent
            )
            ax.add_patch(circle)

        # Plot Centroids
        if len(centroids_arr) > 0:
            ax.scatter(
                centroids_arr[:, 0],
                centroids_arr[:, 1],
                c="red",
                s=10,
                edgecolors="white",
                linewidth=0.5,
                zorder=10,
            )

        split_title = "Hessian Split" if use_hessian else "Geometric Split"
        ax.set_title(
            f"{error_metric} | {split_title}\nN_Clusters = {len(clusters)}", fontsize=11
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("equal")

        # Only add legend to the first plot
        if idx == 0:
            ax.legend(["Centroids"], loc="upper right")

    # Add single shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # type: ignore
    if last_cp:
        fig.colorbar(last_cp, cax=cbar_ax, label="Function Value")

    plt.savefig("centroid_distribution_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved centroid distribution map to centroid_distribution_comparison.png")
    plt.close()

    # 5. Boxplot Visualization
    df = pd.DataFrame(results)
    df["Estimation_Bias"] = df["Estimated_L"] / df["True_L"]

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        x="Error Metric",
        y="Estimation_Bias",
        hue="Hessian Split",
        showfliers=False,
    )
    plt.axhline(1.0, color="r", linestyle="--", label="Perfect Estimate")
    plt.title(
        f"Lipschitz Estimation Quality: |∇g|/|∆x| Method (Tol={TOL}, Steepness={STEEPNESS})"
    )
    plt.ylabel("Ratio (Estimated / True)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Split Method")
    plt.tight_layout()
    plt.savefig("lipschitz_estimation_quality.png", dpi=300)
    print("Saved Lipschitz estimation plot to lipschitz_estimation_quality.png")
    plt.close()

    # Convert summary results to DataFrame and print
    summary_df = pd.DataFrame(summary_results)
    # save summary to CSV
    summary_df.to_csv("hessian_estimation_summary.csv", index=False)
    print("Saved summary results to hessian_estimation_summary.csv")


if __name__ == "__main__":
    benchmark_hessian_estimation()
