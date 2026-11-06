import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist


def sample_chord(X, num_synth, rng):
    """Current Approach: Interpolate between 2 random points."""
    m, d = X.shape
    idx1 = rng.integers(0, m, num_synth)
    idx2 = (idx1 + rng.integers(1, m, num_synth)) % m
    alpha = rng.random((num_synth, 1))

    return X[idx1] * alpha + X[idx2] * (1.0 - alpha)


def sample_dirichlet(X, num_synth, rng):
    """Proposed Approach: Dirichlet weights over d+1 points."""
    m, d = X.shape
    # If the cluster has fewer points than d+1, we cap the sample size
    num_pts = min(d + 1, m)

    # Create permutation/weight matrix B of size (m, num_synth)
    B = np.zeros((m, num_synth))
    for i in range(num_synth):
        # Choose d+1 random points
        idxs = rng.choice(m, num_pts, replace=False)
        # Generate Dirichlet weights
        weights = rng.dirichlet(np.ones(num_pts))
        B[idxs, i] = weights

    # X is (m, d). B is (m, num_synth).
    # X_new = B^T @ X -> results in (num_synth, d)
    return B.T @ X


def compute_metrics(X_orig, X_synth):
    """Calculate the 4 core metrics for evaluating the synthetic distribution."""
    # 1. Variance retained: Trace of synthetic covariance / Trace of original covariance
    cov_orig = np.cov(X_orig, rowvar=False)
    cov_synth = np.cov(X_synth, rowvar=False)

    # Handle edge case where d=1 or scalar variance
    if cov_orig.ndim == 0:
        var_retained = cov_synth / cov_orig
        cov_error = np.abs(cov_synth - cov_orig) / np.abs(cov_orig)
    else:
        var_retained = np.trace(cov_synth) / np.trace(cov_orig)
        # 4. Relative covariance error: Frobenius norm of diff / Frobenius norm of original
        cov_error = np.linalg.norm(cov_synth - cov_orig, ord="fro") / np.linalg.norm(
            cov_orig, ord="fro"
        )

    # 2. Mean radius ratio
    centroid_orig = np.mean(X_orig, axis=0)
    centroid_synth = np.mean(X_synth, axis=0)

    mean_radius_orig = np.mean(np.linalg.norm(X_orig - centroid_orig, axis=1))
    mean_radius_synth = np.mean(np.linalg.norm(X_synth - centroid_synth, axis=1))
    radius_ratio = mean_radius_synth / mean_radius_orig

    # 3. Mean pairwise distance ratio
    mean_pdist_orig = np.mean(pdist(X_orig))
    mean_pdist_synth = np.mean(pdist(X_synth))
    pdist_ratio = mean_pdist_synth / mean_pdist_orig

    return var_retained, radius_ratio, pdist_ratio, cov_error


def run_dimension_sweep():
    """Sweeps across dimensions and creates the 2x2 metric plots."""
    rng = np.random.default_rng(42)

    m = 4000  # Original data size
    num_synth = 400  # Synthetic data size to generate
    _n_max = 8
    dims = [2**i for i in range(1, _n_max)]

    # Storage dictionaries for metrics
    metrics = {
        "chord": {"var": [], "rad": [], "pdist": [], "cov": []},
        "diric": {"var": [], "rad": [], "pdist": [], "cov": []},
    }

    print("\nRunning Dimension Sweep")
    for d in dims:
        # Create a cluster with some variance structure
        cov_matrix = np.diag(np.linspace(10, 1, d))
        X = rng.multivariate_normal(mean=np.zeros(d), cov=cov_matrix, size=m)

        # Generate synthetic points
        X_chord = sample_chord(X, num_synth, rng)
        X_diric = sample_dirichlet(X, num_synth, rng)

        # Compute metrics for both
        vc, rc, pc, cc = compute_metrics(X, X_chord)
        vd, rd, pd, cd = compute_metrics(X, X_diric)

        # Store Chord
        metrics["chord"]["var"].append(vc)
        metrics["chord"]["rad"].append(rc)
        metrics["chord"]["pdist"].append(pc)
        metrics["chord"]["cov"].append(cc)

        # Store Dirichlet
        metrics["diric"]["var"].append(vd)
        metrics["diric"]["rad"].append(rd)
        metrics["diric"]["pdist"].append(pd)
        metrics["diric"]["cov"].append(cd)

    # Plotting the 2x2 Grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    x_labels = [f"$2^{i}$" for i in range(1, _n_max)]

    def format_ax(ax, title, target_line=1.0):
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Dimension d", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.axhline(target_line, linestyle="--", color="steelblue")

    # 1. Variance Retained
    ax = axs[0, 0]
    ax.plot(
        dims,
        metrics["chord"]["var"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Random chord",
    )
    ax.plot(
        dims,
        metrics["diric"]["var"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Dirichlet",
    )
    format_ax(ax, "Variance retained")
    ax.legend(fontsize=12)

    # 2. Mean Radius Ratio
    ax = axs[0, 1]
    ax.plot(
        dims,
        metrics["chord"]["rad"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Random chord",
    )
    ax.plot(
        dims,
        metrics["diric"]["rad"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Dirichlet",
    )
    format_ax(ax, "Mean radius ratio")
    ax.legend(fontsize=12)

    # 3. Mean Pairwise Distance Ratio
    ax = axs[1, 0]
    ax.plot(
        dims,
        metrics["chord"]["pdist"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Random chord",
    )
    ax.plot(
        dims,
        metrics["diric"]["pdist"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Dirichlet",
    )
    format_ax(ax, "Mean pairwise distance ratio")
    ax.legend(fontsize=12)

    # 4. Relative Covariance Error
    ax = axs[1, 1]
    ax.plot(
        dims,
        metrics["chord"]["cov"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Random chord",
    )
    ax.plot(
        dims,
        metrics["diric"]["cov"],
        marker="o",
        markersize=8,
        linewidth=2,
        label="Dirichlet",
    )
    format_ax(ax, "Relative covariance error", target_line=0.0)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("dimension_scaling_metrics.png", dpi=300)


def main():
    rng = np.random.default_rng(0)

    # ==========================================
    # 1. LOW DIMENSION VISUALIZATION (d = 2)
    # ==========================================
    print("Running Low Dimension (d=2) Test...")
    m_low = 20  # Original size
    d_low = 2
    num_synth_low = 9000  # Synthetic size

    mean_low = np.zeros(d_low)
    cov_low = np.array(
        [
            [1, 1],
            [0, 1],
        ]
    )
    # Symmetric covariance
    cov_low = (cov_low + cov_low.T) / 2

    X_low = rng.multivariate_normal(
        mean=mean_low,
        cov=cov_low,
        size=m_low,
    )

    X_chord_low = sample_chord(X_low, num_synth_low, rng)
    X_dirichlet_low = sample_dirichlet(X_low, num_synth_low, rng)

    # Compute 2-sigma ellipse correctly
    eigenvalues, eigenvectors = np.linalg.eigh(cov_low)

    # Sort eigenvalues and eigenvectors in descending order so index 0 is the major axis
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate angle using the major eigenvector (index 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 4 * np.sqrt(eigenvalues)  # 2-sigma width and height

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Plot Chord
    ax1.scatter(
        X_chord_low[:, 0],
        X_chord_low[:, 1],
        c="blue",
        alpha=0.5,
        label="Synthetic",
        s=15,
    )
    ax1.scatter(
        X_low[:, 0], X_low[:, 1], c="black", edgecolor="white", s=60, label="Original"
    )

    # Use mean_low for the center
    ellipse1 = Ellipse(
        xy=mean_low,
        width=width,
        height=height,
        angle=angle,
        linestyle="--",
        edgecolor="green",
        facecolor="none",
        linewidth=2,
        label="2-sigma",
    )
    ax1.add_patch(ellipse1)
    ax1.set_title("Random Chord (d=2)")
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    # Plot Dirichlet
    ax2.scatter(
        X_dirichlet_low[:, 0],
        X_dirichlet_low[:, 1],
        c="red",
        alpha=0.5,
        label="Synthetic",
        s=15,
    )
    ax2.scatter(
        X_low[:, 0], X_low[:, 1], c="black", edgecolor="white", s=60, label="Original"
    )

    # Use mean_low for the center
    ellipse2 = Ellipse(
        xy=mean_low,
        width=width,
        height=height,
        angle=angle,
        linestyle="--",
        edgecolor="green",
        facecolor="none",
        linewidth=2,
        label="2-sigma",
    )
    ax2.add_patch(ellipse2)
    ax2.set_title("Dirichlet (d=2)")
    ax2.legend()
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig("low_dim_comparison.png", dpi=300)
    # ==========================================
    # 2. RUN DIMENSION SWEEP AND PLOT METRICS
    # ==========================================
    run_dimension_sweep()


if __name__ == "__main__":
    main()
