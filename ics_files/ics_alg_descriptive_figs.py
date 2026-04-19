### --- PLOTTING TOOLS --- ###

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

# Plotting tools


def plot_points(centroids, ax: Axes, color="red", marker="o", size=4, label=""):
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=size,
        marker=marker,
        c=color,
        label=label,
        alpha=1,
    )


def plot_evaluation_points(points, ax: Axes, color="black", marker="o"):
    ax.scatter(points[:, 0], points[:, 1], s=20, marker=marker, c=color)

    # Also plot visual labels (letters) for each point
    shift = 0.02
    import string

    letters = list(string.ascii_uppercase)

    for j in range(len(points)):
        x = points[j, 0] + shift
        y = points[j, 1] + shift
        ax.text(x, y, letters[j], horizontalalignment="left")


def plot_ellipses(ellipses, ax: Axes, plot_numbers=False):
    for j, ellipse in enumerate(ellipses):
        ax.add_patch(ellipse)

    # Plot ellipse numbers
    if plot_numbers:
        shift = 0.02
        for j in range(len(ellipses)):
            xy = ellipses[j].get_center()
            x = xy[0] + shift
            y = xy[1] + shift
            ax.text(x, y, f"{j+1}", horizontalalignment="left")


def convert_to_ellipse_object(Q, centroid, color="red", alpha=0.3, filled=True):

    D, U = np.linalg.eigh(Q)
    order = np.argsort(D)
    D = D[order]
    U = U[:, order]
    # Length of axes are lambda_i^{-0.5}.
    # So total diameter is 2 / D**(1/2).
    lengths = 2.0 / np.sqrt(D + 1e-12)
    width = lengths[0]
    height = lengths[1]
    # Angle is given by the rotation matrix U.
    # In particular, the first singular vector (which corresponds to the width)
    v0 = U[:, 0]
    angle = np.degrees(np.arctan2(v0[1], v0[0]))
    if filled:
        return Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            color=color,
            alpha=alpha,
        )
    else:
        return Ellipse(
            xy=centroid,
            width=width,
            height=height,
            angle=angle,
            alpha=1,
            facecolor="none",
            edgecolor=color,
            linewidth=1,
            # linestyle='dotted'
        )


### --- CLUSTER TOOLS --- ###

from scipy.spatial import ConvexHull


# Each of the following sampling functions are centered at origin and are based on unit ball
# In other words, no ellipses
def generate_unit_ball_points_uniform(N, d=2):
    directions = np.random.randn(N, d)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = np.random.random(N) ** (1 / d)
    return directions * radii[:, None]


def generate_unit_ball_points_boundary_uniform(N, d=2):
    directions = np.random.randn(N, d)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions


def generate_random_polytope(V, N, d=2):

    # Generate vertices and candidate points
    candidates = generate_unit_ball_points_uniform(N, d)
    polytope_vertices = generate_unit_ball_points_boundary_uniform(V, d)

    # Build convex hull
    convex_hull = ConvexHull(polytope_vertices)
    equations = convex_hull.equations
    A = equations[:, :-1]
    b = equations[:, -1]

    # Test candidates: a point is in if Ax + b < 0.
    in_half_plane = ((candidates @ A.T) + b) < 0
    in_conv_hull_msk = in_half_plane.all(axis=1)
    polytope_points = candidates[in_conv_hull_msk]
    return polytope_points


def matrix_sqrt(Q):
    D, U = np.linalg.eigh(Q)
    return (U * (np.abs(D) ** 0.5)) @ U.T


if __name__ == "__main__":

    ### --- FIGURE 4: Entire description of algorithm --- ###
    """
    Underlying function: Rosenbrock 2D
    Fig (a): input points
    Fig (b): final points (centroids) after running algorithm
    Fig (c): colored leaf clusters (input points but reclustered)
    Fig (d): tree
    """

    import matplotlib.colors as mcolors

    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
    from src.surrogate import AFINNISurrogate

    # Training data
    d = 2
    N_train = 20000
    X_train = np.column_stack(
        (np.random.uniform(-2, 2, N_train), np.random.uniform(-1, 3, N_train))
    )

    # Testing data
    N_test = 1000000
    X_test = np.column_stack(
        (np.random.uniform(-2, 2, N_test), np.random.uniform(-1, 3, N_test))
    )

    N_visual = 500
    X_train_visual = np.column_stack(
        (np.random.uniform(-2, 2, N_visual), np.random.uniform(-1, 3, N_visual))
    )

    # Adaptive clustering parameters
    N_init_clusters = 4
    tol = 0.1

    # Trained surrogate
    surrogate = AFINNISurrogate()
    surrogate.train(
        X_train,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        save_model=False,
        TOL=tol,
        ERROR_METRIC="Linf-local",
        NUM_INITIAL_CLUSTERS=N_init_clusters,
        MIN_PTS_PER_CLUSTER=15,
        MAX_CLUSTERS_EST=10000,
        SEED=1,
        CLUSTERING_MATRIX="Cov_Lvl_curves",
        IGNORE_GRADIENT=False,
        STORE_AFFINE_FIT=True,
    )

    centroids = surrogate.get_source_centroids()
    TRs = surrogate.get_trust_regions()
    inf_vals, unguaranteed, nn_idx = surrogate.predict(
        X_test, predict_mode="affine", return_cluster_assignments=True
    )

    p = np.random.permutation(len(centroids))
    centroids = centroids[p]
    TRs = TRs[p]
    labels = nn_idx.flatten()

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.tight_layout()

    # gray colormap
    cmap_gray = plt.get_cmap("gray")
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated_gray", cmap_gray(np.linspace(0.3, 1.0, 100))
    )

    # contour data
    x = np.linspace(-2.1, 2.1, 400)
    y = np.linspace(-1.1, 3.1)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z_flat = rosenbrock(XY, a=100)
    Z = Z_flat.reshape(X.shape)

    # Fig (a): training data
    p = 0
    ax[p].contourf(X, Y, np.log10(Z), levels=10, cmap=new_cmap)
    # ax[p].contour(X, Y, np.log10(Z), levels=10, colors="black", linewidths=1.5, alpha=0.5, zorder=0)
    # plot_points(X_train, ax[0], color="#CA0000", marker='o', label='Training Data', size=25)
    plot_points(
        X_train_visual,
        ax[0],
        color="#CA0000",
        marker="o",
        label="Training Data",
        size=25,
    )
    xlims = ax[p].get_xlim()
    ylims = ax[p].get_ylim()
    ax[p].legend(loc="upper right")
    ax[p].set_xticks([])
    ax[p].set_yticks([])

    # Fig (b): centroids after running alg
    p = 1
    ellipses = []
    for j, TR in enumerate(TRs):
        ellipses.append(
            convert_to_ellipse_object(
                TR, centroids[j], "#cb00d9", filled=False, alpha=0.7
            )
        )
    ax[p].contourf(X, Y, np.log10(Z), levels=10, cmap=new_cmap)
    # ax[p].contour(X, Y, np.log10(Z), levels=10, colors="black", linewidths=1.5, alpha=0.5, zorder=0)
    plot_ellipses(ellipses, ax[p], False)
    plot_points(
        centroids, ax[p], color="#4C38FF", marker="x", size=30, label="Centroids"
    )
    ax[p].set_xlim(xlims)
    ax[p].set_ylim(ylims)
    ax[p].legend(loc="upper right")
    ax[p].set_xticks([])
    ax[p].set_yticks([])

    # Fig (c): colored input points (but reclustered)
    p = 2
    # ax[p].contourf(X, Y, np.log10(Z), levels=10, cmap=new_cmap)
    # ax[p].contour(X, Y, np.log10(Z), levels=10, colors="black", linewidths=1.5, alpha=0.5, zorder=0)
    # cmap = plt.get_cmap('tab20b', len(centroids))
    # plot_colors = [cmap(x) for x in np.linspace(0, 1, len(centroids))]
    for j in range(len(centroids)):
        ax[p].scatter(X_test[:, 0][labels == j], X_test[:, 1][labels == j], s=3)
    plot_points(
        centroids, ax[p], color="black", marker="x", size=30, label="Cluster Regions"
    )
    xlims = ax[0].get_xlim()
    ylims = ax[0].get_ylim()
    ax[p].set_xlim(xlims)
    ax[p].set_ylim(ylims)
    # plot_points(centroids, ax[p], color="#000000", marker='x', size=10)
    # plot_ellipses(ellipses, ax[p], False)
    ax[p].set_xticks([])
    ax[p].set_yticks([])
    ax[p].legend(loc="upper right")

    fig.savefig(
        "figures/ellipses_2_sideways.png", bbox_inches="tight", pad_inches=0, dpi=300
    )
