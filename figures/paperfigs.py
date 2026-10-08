import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.adaptive_clustering import adaptive_clustering
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull


# FIGURE TO EXPLAIN CLUSTERING
def methods_adaptive_clustering():
    
    np.random.seed(42)
    data = np.random.uniform([-2,-1], [2,3], size=(1000,2))

    def plot_level_of_algorithm(level, xloc, yloc, use_colors):
        output = adaptive_clustering(
            X = data,
            f_fn = rosenbrock,
            g_fn = rosenbrock_grad,
            TOL = 0.1,
            ERROR_METRIC='Linf-local',
            NUM_INITIAL_CLUSTERS=4,
            SEED=0
        )
        centers = output['']
        labels = _assign_labels(centers, data)
        for j in range(len(centers)):
            if use_colors:
                axs[xloc,yloc].scatter(data[labels==j][:,0], data[labels==j][:,1], marker='o', s=5, c=colors[j])
        axs[xloc,yloc].scatter(centers[:,0], centers[:,1], c='black', marker='o', s=20)
        axs[xloc,yloc].set_xlim(-2, 2)
        axs[xloc,yloc].set_ylim(-1, 3)
    
    x = np.linspace(-2,2,200)
    y = np.linspace(-1,3,200)
    X, Y = np.meshgrid(x,y)
    # Z = rosenbrock(X,Y)

    colors = ['red', 'gold', 'green', 'blue',
              "#671313", "#af5252", "#FFC74D", "#CEA200",
              '#2b5d34', "#657c02", "#66b6f3", "#2203ab",]
    
    # This can maybe be ignored
    fig, axs = plt.subplots(nrows=5, ncols=2, gridspec_kw={'width_ratios': [1, 2]})
    fig.tight_layout()

    # Original clusters
    plot_level_of_algorithm(0, 1, 0, True)

    # Make the corresponding tree
    circles = [
        patches.Circle((0.3,0.8), 0.05, facecolor=colors[0], edgecolor='black'),
        patches.Circle((0.7,0.8), 0.05, facecolor=colors[1], edgecolor='black'),
        patches.Circle((1.1,0.8), 0.05, facecolor=colors[2], edgecolor='black'),
        patches.Circle((1.5,0.8), 0.05, facecolor=colors[3], edgecolor='black'),
    ]
    for circle in circles:
        axs[1,1].add_patch(circle)
    
    # Plot second level
    plot_level_of_algorithm(1, 2, 0, True)

    # Make the tree for the second plot
    lines = [
        patches.Arrow(0.3, 0.8, -0.1, -0.3, width=0.05, color='black'),
        patches.Arrow(0.3, 0.8,  0.1, -0.3, width=0.05, color='black'),
        patches.Arrow(1.1, 0.8, -0.1, -0.3, width=0.05, color='black'),
        patches.Arrow(1.1, 0.8,  0.1, -0.3, width=0.05, color='black'),
        patches.Arrow(1.5, 0.8, -0.1, -0.3, width=0.05, color='black'),
        patches.Arrow(1.5, 0.8,  0.1, -0.3, width=0.05, color='black'),
    ]
    circles = [
        patches.Circle((0.3,0.8), 0.05, facecolor='grey', edgecolor='black'),
        patches.Circle((0.7,0.8), 0.05, facecolor=colors[1], edgecolor='black'),
        patches.Circle((1.1,0.8), 0.05, facecolor='grey', edgecolor='black'),
        patches.Circle((1.5,0.8), 0.05, facecolor='grey', edgecolor='black'),
        patches.Circle((0.2,0.5), 0.05, facecolor=colors[5], edgecolor='black'),
        patches.Circle((0.4,0.5), 0.05, facecolor=colors[4], edgecolor='black'),
        patches.Circle((1.0,0.5), 0.05, facecolor=colors[6], edgecolor='black'),
        patches.Circle((1.2,0.5), 0.05, facecolor=colors[7], edgecolor='black'),
        patches.Circle((1.4,0.5), 0.05, facecolor=colors[9], edgecolor='black'),
        patches.Circle((1.6,0.5), 0.05, facecolor=colors[8], edgecolor='black'),
    ]
    for line in lines:
        axs[2,1].add_patch(line)
    for circle in circles:
        axs[2,1].add_patch(circle)

    
    for i in range(len(axs)):
        for j in range(len(axs[0])):
            if j == 1 and i <= 2:
                axs[i,1].set_yticklabels([])
                axs[i,1].set_xticklabels([])
                axs[i,1].set_xlim(0,1.8)
                axs[i,1].set_ylim(0,1)
            axs[i,j].set_aspect('equal', adjustable='box')


    
        

    # Now do the much lower levels
    plot_level_of_algorithm(3, 3, 0, False)
    plot_level_of_algorithm(5, 3, 1, False)
    plot_level_of_algorithm(8, 4, 0, False)
    plot_level_of_algorithm(-1, 4, 1, False)

    
    plt.show()



def error_approximation_illustration():

    fig, ax = plt.subplots(1,2)
    ax1 = ax[0]
    ax2 = ax[1]

    # figure 1
    cluster_x = np.array([0.05, 0.15, 0.3, 0.4, 0.6, 0.7, 0.75, 0.875, 0.95])
    center = np.mean(cluster_x)
    a, x0, y0 = 2, 0.4, 0.3
    def f(x):
        return a * (x-x0)**2 + y0
    def slope(x):
        return 2 * a * (x-x0)
    def g(x):
        return slope(center) * (x - center) + f(center)

    xgrid = np.linspace(0,1,200)
    fgrid = f(xgrid)
    ggrid = g(xgrid)
    cluster_y_exact = f(cluster_x)
    cluster_y_approx = g(cluster_x)

    for j, x in enumerate(cluster_x):
        ax1.plot([x,x],[cluster_y_exact[j],cluster_y_approx[j]],'--r')
    ax1.plot(xgrid, fgrid, '-k')
    ax1.plot(xgrid, ggrid, '--b')
    ax1.scatter(center, 0, marker='x', s=60, c='black')
    ax1.scatter(cluster_x, np.zeros_like(cluster_x), marker='o', s=50, c='grey')
    ax1.set_xlim(0,1)
    ax1.set_ylim(-0.1,0.95)
    

    # figure 2
    km = KMeans(n_clusters=2)
    labels = km.fit_predict(np.expand_dims(cluster_x,axis=1))
    cluster_x_1 = cluster_x[labels==0]
    cluster_x_2 = cluster_x[labels==1]
    center1 = np.minimum(km.cluster_centers_[0], km.cluster_centers_[1])
    center2 = np.maximum(km.cluster_centers_[0], km.cluster_centers_[1])
    midpoint = 0.5 * (center1 + center2)
    def g1(x):
        return slope(center1) * (x - center1) + f(center1)
    def g2(x):
        return slope(center2) * (x - center2) + f(center2)

    xgrid1 = np.linspace(0,midpoint,100)
    xgrid2 = np.linspace(midpoint,1,100)
    ggrid1 = g1(xgrid1)
    ggrid2 = g2(xgrid2)
    cluster_y_approx_1 = g1(cluster_x_1)
    cluster_y_exact_1 = cluster_y_exact[labels==0]
    cluster_y_approx_2 = g2(cluster_x_2)
    cluster_y_exact_2 = cluster_y_exact[labels==1]

    for j, x in enumerate(cluster_x_1):
        ax2.plot([x,x],[cluster_y_exact_1[j],cluster_y_approx_1[j]],'--r')
    for j, x in enumerate(cluster_x_2):
        ax2.plot([x,x],[cluster_y_exact_2[j],cluster_y_approx_2[j]],'--r')
    ax2.plot(xgrid, fgrid, '-k')
    ax2.plot(xgrid1, ggrid1, '--b')
    ax2.plot(xgrid2, ggrid2, '--b')
    ax2.scatter(km.cluster_centers_, [0,0], marker='x', s=60, c='black')
    ax2.scatter(cluster_x, np.zeros_like(cluster_x), marker='o', s=50, c='grey')
    ax2.set_xlim(0,1)
    ax2.set_ylim(-0.1,0.95)

    plt.show()



# extremely slow (with for loop) but fine for plotting
def _assign_labels(sources, targets):
    assignments = np.zeros(len(targets))
    for j, target in enumerate(targets):
        assignments[j] = np.argmin(np.linalg.norm(sources-target, 2, axis=1))
    return assignments



def illustrate_hessian_approx(
    save_path="hessian_approx.png",
    domain=(0.5, 2.5, 0.5, 2.5),
    n_points=1000,
    seed=0,
):
    """Build and save the convex-region rejection-sampling figure.
 
    Generates a random convex polygon inside `domain`, draws `n_points`
    samples from it via rejection sampling, and renders two side-by-side
    panels (both with a contour background of f(x,y) = x^2 + y^2) showing
    the sampled points. Saves the figure to `save_path` as a .png.
 
    Returns nothing.
    """
    rng = np.random.default_rng(seed=seed)
    xmin, xmax, ymin, ymax = domain
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    avg_radius = 0.25 * min(xmax - xmin, ymax - ymin)
 
    # --------------------------------------------------------------
    # Paper-ready style
    # --------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
 
    # --------------------------------------------------------------
    # Random convex region: scatter irregularly-spaced, irregularly-
    # sized points around a center, then take their convex hull.
    # --------------------------------------------------------------
    def random_convex_polygon(n_seed=14, irregularity=0.5, spikiness=0.45):
        irreg = irregularity * 2 * np.pi / n_seed
        spike = spikiness * avg_radius
 
        steps = rng.uniform(2 * np.pi / n_seed - irreg,
                            2 * np.pi / n_seed + irreg, n_seed)
        steps *= 2 * np.pi / steps.sum()
        angles = np.cumsum(steps) + rng.uniform(0, 2 * np.pi)
 
        radii = np.clip(rng.normal(avg_radius, spike, n_seed),
                        0.25 * avg_radius, 1.75 * avg_radius)
 
        seed_pts = np.column_stack([
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles),
        ])
        hull = ConvexHull(seed_pts)
        verts = seed_pts[hull.vertices]
 
        # Re-center on the polygon's true (area-weighted) centroid,
        # not just the seed-scatter center: irregular radii otherwise
        # tend to pull the hull's visual "weight" off to one side.
        vx, vy = verts[:, 0], verts[:, 1]
        vx1, vy1 = np.roll(vx, -1), np.roll(vy, -1)
        cross = vx * vy1 - vx1 * vy
        area = cross.sum() / 2
        cx = ((vx + vx1) * cross).sum() / (6 * area)
        cy = ((vy + vy1) * cross).sum() / (6 * area)
        verts += np.array([center[0] - cx, center[1] - cy])
 
        return verts
 
    vertices = random_convex_polygon()
    region_path = Path(vertices)
 
    # --------------------------------------------------------------
    # Rejection sampling: draw uniform candidates in the polygon's
    # bounding box, keep only those inside the convex polygon.
    # --------------------------------------------------------------
    def rejection_sample_polygon(path, bbox, n_target, batch_size=2000):
        bxmin, bxmax, bymin, bymax = bbox
        accepted = []
        n_proposed = 0
        while sum(len(a) for a in accepted) < n_target:
            candidates = rng.uniform([bxmin, bymin], [bxmax, bymax],
                                      size=(batch_size, 2))
            n_proposed += batch_size
            inside = path.contains_points(candidates)
            accepted.append(candidates[inside])
        pts = np.vstack(accepted)[:n_target]
        acceptance_rate = n_target / n_proposed
        return pts, acceptance_rate
 
    bbox_xmin, bbox_ymin = vertices.min(axis=0)
    bbox_xmax, bbox_ymax = vertices.max(axis=0)
    points, acc_rate = rejection_sample_polygon(
        region_path, (bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax), n_points
    )
    print(f"Accepted {n_points} points; approx. acceptance rate = {acc_rate:.2%}")
 
    # --------------------------------------------------------------
    # Background function and contour grid, f(x,y) = x^2 + y^2
    # --------------------------------------------------------------
    grid_n = 300
    xg = np.linspace(xmin, xmax, grid_n)
    yg = np.linspace(ymin, ymax, grid_n)
    Xg, Yg = np.meshgrid(xg, yg)
    Fg = 10 * Xg**2 + Yg**2
 
    # --------------------------------------------------------------
    # Centroid of the sampled point cluster
    # --------------------------------------------------------------
    centroid = points.mean(axis=0)


    # Compute nearest neighbors for gradient approximation
    k1 = 35
    diff = points - centroid
    dist = np.linalg.norm(diff, axis=1)
    nn_idx_grad = np.argsort(dist)[:k1]

    # Compute Taylor expansion for Hessian approximation
    # --- Contour function f(x, y) = x^2 + y^2 and its gradient (2x, 2y) ---
    def f(x, y):
        return 10 * x**2 + y**2

    def grad_f(x, y):
        return np.stack([20 * x, 2 * y], axis=-1)

    # --- Evaluate at the centroid ---
    # (assumes `centroid` is a length-2 array/tuple: [cx, cy])
    f_centroid = f(centroid[0], centroid[1])
    grad_centroid = grad_f(centroid[0], centroid[1])

    print(f"f(centroid)      = {f_centroid:.6f}")
    print(f"grad f(centroid) = {grad_centroid}")

    # --- Evaluate at every sampled point in the convex region ---
    # (assumes `points` is an (N, 2) array of [x, y] samples)
    k2 = 100
    f_values = f(points[:, 0], points[:, 1])          # shape (N,)
    taylor_err = f_values - f_centroid - diff @ grad_centroid
    nn_idx_hess = np.argsort(taylor_err)[:k2]
 
    # --------------------------------------------------------------
    # Figure with two side-by-side panels. Both panels: contour
    # background + sampled points + centroid marker. No region
    # shading, no colorbar.
    # --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    contour_levels = 14
 
    for ax, title, iter in zip(
        axes,
        ["Gradient approx. pts.",
         r"Hessian approx. pts."],
         [0, 1]
    ):
        ax.contourf(Xg, Yg, Fg, levels=contour_levels, cmap="viridis",
                    alpha=0.75, zorder=0)
        ax.contour(Xg, Yg, Fg, levels=contour_levels, colors="white",
                   linewidths=0.5, alpha=0.6, zorder=1)
 
        ax.scatter(points[:, 0], points[:, 1], s=14, facecolor="white",
                   linewidth=0.3, alpha=0.95, zorder=3)

        if iter == 0:
            ax.scatter(points[nn_idx_grad, 0], points[nn_idx_grad, 1], s=16, facecolor="green",
                        edgecolor="black", zorder=4, label="Gradient Approx. Pts.", linewidth=0.3)
        elif iter == 1:
            ax.scatter(points[nn_idx_hess, 0], points[nn_idx_hess, 1], s=16, facecolor="pink",
                        edgecolor="black", zorder=4, label="Hessian Approx. Pts.", linewidth=0.3)
 
        ax.scatter(centroid[0], centroid[1], marker="x", s=60, c="red",
                   linewidth=2, zorder=5)
 
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
 
    fig.savefig(save_path)
    plt.close(fig)






if __name__ == '__main__':
    if 0:
        methods_adaptive_clustering()
    if 1:
        error_approximation_illustration()
    if 0:
        illustrate_hessian_approx()
    