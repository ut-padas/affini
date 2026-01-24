import logging
from collections.abc import Callable

import numpy as np
from sklearn.cluster import KMeans

from ben_ellipse import (
    fit_ellipse_PCA,
    fit_ellipse_PCA_scaled,
    metric_adaptive_clustering,
)


# bi-tree structures that we use in the recursive algorithm implemented below
class ClusterTree:

    def __init__(self, params, left, right):
        self.left = left
        self.right = right
        self.params = params


class Cluster:

    def __init__(self, index, params, QoI_value, gradient, trust_ellipse, n_pts):
        self.index = index  # leaf index
        self.params = params  # centroids of the cluster
        self.QoI_value = QoI_value  # output value
        self.gradient = gradient  # gradient value
        self.trust_ellipse = trust_ellipse  # ellipse that defines analytic trust region
        self.n_pts = n_pts  # number of points that generate the cluster


def adaptive_cluster_algorithm(
    training_data: np.ndarray,  # each row contains a set of parameters used to cluster
    eval_function: Callable,  # callable function; takes numpy array to numpy array
    # if use gradient is true, eval_function should return both value and gradient.
    # otherwise eval_function only returns value
    # modify so that if this takes a vector to a vector and a matrix (R^n to R^m),
    # it still works
    atol: float = 0.1,
    rtol: float = 0.1,
    N_init_clusters: int = 10,  # number of initial clusters to split into
    min_ppc: int = 1,  # minimum number of points used in a cluster (to measure error)
    use_gradient: bool = False,
    function_vals: (
        np.ndarray | None
    ) = None,  # if this is not none, uses these as solves. Otherwise runs eval_function every time
    seed=None,  # seed used for kmeans rng
    num_levels=-1,  # specify the number of levels for the algorithm to go in the tree.
    # If this is -1, then just go forever.  Otherwise only go up to this level in the tree.
):

    # check dimensions of training data; add another if necessary
    if training_data.ndim == 1:
        training_data = np.expand_dims(training_data, axis=1)

    # also check dimensions of function vals
    # do solves if necessary. TODO: specify dimension of output of eval_function
    if function_vals is None:
        function_vals = np.zeros(len(training_data))
        if not use_gradient:
            for j, row in enumerate(training_data):
                function_vals[j] = eval_function(row)
        else:
            for j, row in enumerate(training_data):
                function_vals[j], _ = eval_function(row)
    if function_vals.ndim == 1:
        function_vals = np.expand_dims(function_vals, axis=1)

    p1, n = training_data.shape
    p2, m = function_vals.shape
    if p1 != p2:
        raise Exception(
            "Error: length of training data and function values lists do not match"
        )

    k = 0  # counting number of clusters as we go
    cluster_structure = np.empty(shape=N_init_clusters, dtype=object)

    centroids, labels = _cluster(training_data, N_init_clusters, seed)

    for j in range(N_init_clusters):

        params_in_cluster = training_data[labels == j]
        fvals_in_cluster = function_vals[labels == j]

        centroid_params = centroids[j]
        fval_c, grad_c = _compute_oracle_at_point(
            centroid_params, eval_function, use_gradient
        )

        # check size of gradient to see if it is appropriate
        if use_gradient and grad_c.shape != (m, n):  # type: ignore
            raise Exception("Error: gradient function does return shape m*n")

        k, cluster_structure[j] = _adaptive_cluster_algorithm_helper(
            params_in_cluster,
            fvals_in_cluster,
            centroid_params,
            fval_c,
            grad_c,  # type: ignore
            eval_function,
            N_init_clusters * 2,
            atol,
            rtol,
            min_ppc,
            use_gradient,
            k,
            seed,
            num_levels - 1,
        )

    return k, cluster_structure


def _adaptive_cluster_algorithm_helper(
    training_data: np.ndarray,
    function_vals: np.ndarray,
    prev_centroid: np.ndarray,
    f_prev_centroid: np.ndarray,
    grad_prev_centroid: np.ndarray,
    eval_function: Callable,
    total_clusters: float,  # required to compute global absolute error
    atol=None,
    rtol=None,
    min_ppc: int = 1,
    use_gradient: bool = False,
    k: int = 0,  # current cluster index (for recursive purposes)
    seed=None,
    num_levels=-1,
):

    if len(training_data) <= 1:
        # Dead branch of the tree. Not enough variance of points to compute error (or a single point).
        return k, None

    # Add extra points to the cluster (randomly sampled) until we get to the minimum ppc
    training_data, function_vals = _generate_extra_points(
        training_data, function_vals, eval_function, min_ppc, use_gradient
    )

    # With metric adaptive clustering, we check error first
    # prev_centroid is the centroid of the data
    # check linear interpolant fit around this centroid
    distance = training_data - prev_centroid
    if not use_gradient:
        evec = np.linalg.norm(function_vals - f_prev_centroid, 2, axis=1)
    elif use_gradient:
        # make sure we compute in euclidean, non-log scaled distance
        evec = np.linalg.norm(
            function_vals - (f_prev_centroid + (distance @ grad_prev_centroid.T)),
            2,
            axis=1,
        )
    else:
        raise Exception("Error: use_gradient must be a boolean")

    abs_error = np.linalg.norm(evec, 2)  # TODO: are we using the correct norm?
    rel_error = abs_error / np.linalg.norm(function_vals, "fro")
    with_good_error = np.logical_or(evec <= atol, evec <= (rtol * f_prev_centroid))

    # from this, decide if we should split or not
    # TODO: look at the error criterion.
    if (
        (rtol is not None and rel_error <= rtol)
        or (atol is not None and abs_error <= (atol / np.sqrt(total_clusters)))
        or num_levels == 0
    ):

        # Create a cluster. Use ellipse formulation
        trust_ellipse = fit_ellipse_PCA_scaled(distance[with_good_error])
        k += 1
        return k, Cluster(
            k - 1,
            prev_centroid,
            f_prev_centroid,
            grad_prev_centroid,
            trust_ellipse,
            len(evec),
        )

    # Otherwise we plan to split the cluster. Use metric-adaptive.
    good_error_abs = evec <= atol
    good_error_rel = evec <= (rtol * f_prev_centroid)
    good_error = np.logical_or(good_error_abs, good_error_rel)

    # sort points, compute SVD of points with good error
    _, dim = training_data.shape
    if good_error.sum() > dim:
        Q = fit_ellipse_PCA(distance[good_error])
    else:
        Q = np.eye(dim)

    # with this psd matrix in hand, do metric adaptive clustering
    centroids_scaled, labels = metric_adaptive_clustering(distance, Q, n_clusters=2)
    centroids = centroids_scaled + prev_centroid

    # First cluster (N=0)
    pts_cluster = training_data[labels == 0]
    fvals_cluster = function_vals[labels == 0]
    centroid_params = centroids[0]
    fval_c, grad_c = _compute_oracle_at_point(
        centroid_params, eval_function, use_gradient
    )

    k, left = _adaptive_cluster_algorithm_helper(
        pts_cluster,
        fvals_cluster,
        centroid_params,
        fval_c,
        grad_c,  # type: ignore
        eval_function,
        total_clusters * 2,
        atol,
        rtol,
        min_ppc,
        use_gradient,
        k,
        seed,
        num_levels - 1,
    )

    # Second cluster (N=1)
    pts_cluster = training_data[labels == 1]
    fvals_cluster = function_vals[labels == 1]
    centroid_params = centroids[1]
    fval_c, grad_c = _compute_oracle_at_point(
        centroid_params, eval_function, use_gradient
    )

    k, right = _adaptive_cluster_algorithm_helper(
        pts_cluster,
        fvals_cluster,
        centroid_params,
        fval_c,
        grad_c,  # type: ignore
        eval_function,
        total_clusters * 2,
        atol,
        rtol,
        min_ppc,
        use_gradient,
        k,
        seed,
        num_levels - 1,
    )

    return k, ClusterTree(prev_centroid, left, right)


# At the moment, this implementation is really bad and will be slow. Redo how we sample (for loop).
# We require more than one point in the cluster (this number can change).
def _generate_extra_points(pts_cluster, fvals_cluster, oracle, min_ppc, use_gradient):

    n_pts_sample = min_ppc - len(pts_cluster)

    if n_pts_sample <= 0:
        return pts_cluster, fvals_cluster

    # take random convex combinations
    lambdas = np.random.uniform(0, 1, n_pts_sample)
    indices = np.random.randint(0, len(pts_cluster), (2, n_pts_sample))

    # deal with columns where the indices are the same
    # probably a much better way to do this...
    resample = (indices[1] - indices[0]) == 0
    all_inds = np.arange(len(pts_cluster))
    for j, val in enumerate(resample):
        if j:
            indices[:, j] = np.random.choice(all_inds, 2, replace=False)

    # construct convex combinations
    new_pts_cluster = np.multiply(
        pts_cluster[indices[0]], lambdas[:, np.newaxis]
    ) + np.multiply(pts_cluster[indices[1]], (1 - lambdas)[:, np.newaxis])

    # do ode solves for these new points
    new_fvals_cluster = np.zeros(shape=(n_pts_sample, fvals_cluster.shape[1]))
    for i, row in enumerate(new_pts_cluster):
        # compute true value (save in qi)
        if use_gradient:
            new_fvals_cluster[i], _ = oracle(row)
        else:
            new_fvals_cluster[i] = oracle(row)

    # put these new points together with the old ones
    pts_cluster = np.vstack((pts_cluster, new_pts_cluster))
    fvals_cluster = np.vstack((fvals_cluster, new_fvals_cluster))

    return pts_cluster, fvals_cluster


def _compute_oracle_at_point(x, oracle, use_gradient):

    if use_gradient:
        fval, grad = oracle(x)
        if np.isscalar(grad):
            grad = np.array([grad])
        if grad.ndim == 1:
            grad = np.expand_dims(grad, axis=1)
    else:
        fval = oracle(x)
        grad = None
    if np.isscalar(fval):
        fval = np.array([fval])

    return fval, grad


def _cluster(data: np.ndarray, n_clusters, seed):

    # normalize the training data
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_normalized = (data - mean) / std

    # Call clustering library
    km = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = km.fit_predict(data_normalized)
    centroids_normalized = km.cluster_centers_
    centroids = centroids_normalized * std + mean

    return centroids, labels


def flatten_cluster_centers_with_gradient(nc, n, m, tree):
    centroids = np.zeros(shape=(nc, n))
    QoI_values = np.zeros(shape=(nc, m))
    gradients = np.zeros(shape=(nc, m, n))
    trust_ellipses = np.zeros(shape=(nc, n, n))
    n_pts = np.zeros(shape=nc)
    tree_levels = np.zeros(shape=nc)

    k = 0
    lev = 0
    for val in tree:
        k = _flatten_cluster_centers_with_gradient_helper(
            val,
            k,
            lev + 1,
            centroids,
            QoI_values,
            gradients,
            trust_ellipses,
            n_pts,
            tree_levels,
        )

    return centroids, QoI_values, gradients, trust_ellipses, n_pts, tree_levels


def _flatten_cluster_centers_with_gradient_helper(
    tree, k, lev, centroids, QoI_values, gradients, trust_ellipses, n_pts, tree_levels
):
    if type(tree) is Cluster:
        centroids[k] = tree.params
        QoI_values[k] = tree.QoI_value
        if tree.gradient is None:
            gradients[k] = np.zeros((len(QoI_values[k]), len(centroids[k])))
        else:
            gradients[k] = tree.gradient
        trust_ellipses[k] = tree.trust_ellipse
        n_pts[k] = tree.n_pts
        tree_levels[k] = lev
        k += 1
    elif type(tree) is ClusterTree:
        k = _flatten_cluster_centers_with_gradient_helper(
            tree.left,
            k,
            lev + 1,
            centroids,
            QoI_values,
            gradients,
            trust_ellipses,
            n_pts,
            tree_levels,
        )
        k = _flatten_cluster_centers_with_gradient_helper(
            tree.right,
            k,
            lev + 1,
            centroids,
            QoI_values,
            gradients,
            trust_ellipses,
            n_pts,
            tree_levels,
        )
    return k


def flatten_cluster_centers(nc, n, m, tree):
    centroids = np.zeros(shape=(nc, n))
    QoI_values = np.zeros(shape=(nc, m))
    k = 0
    for val in tree:
        if type(val) is Cluster:
            centroids[k] = val.params
            QoI_values[k] = val.QoI_value
            k += 1
        elif type(val) is ClusterTree:
            k = _flatten_cluster_centers_helper(val.left, k, centroids, QoI_values)
            k = _flatten_cluster_centers_helper(val.right, k, centroids, QoI_values)
    return centroids, QoI_values


def _flatten_cluster_centers_helper(tree, k, centroids, QoI_values):
    if type(tree) is Cluster:
        centroids[k] = tree.params
        QoI_values[k] = tree.QoI_value
        k += 1
    elif type(tree) is ClusterTree:
        k = _flatten_cluster_centers_helper(tree.left, k, centroids, QoI_values)
        k = _flatten_cluster_centers_helper(tree.right, k, centroids, QoI_values)
    return k


if __name__ == "__main__":
    import argparse
    from functools import partial

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from src.logging_config import setup_logging
    from src.problems.quadratic import quadratic, quadratic_grad
    from src.problems.shifted_gaussians import (
        CENTERS_2D,
        shifted_gaussians,
        shifted_gaussians_grad,
    )

    setup_logging(console_level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tol", type=float, default=0.1, help="Tolerance for adaptive clustering"
    )
    args = parser.parse_args()
    TOL = args.tol

    # -----------------------------------------
    # Gaussian Function
    shif_g_fn = partial(shifted_gaussians, centers=CENTERS_2D)
    shif_g_grad_fn = partial(shifted_gaussians_grad, centers=CENTERS_2D)

    def evaluate_function(x):
        # If x is 1D, make it 2D
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        f_val = shif_g_fn(x).flatten()  # (N, )
        g_val = shif_g_grad_fn(x)  # (N, 1, D)
        g_val = g_val.reshape(g_val.shape[0], g_val.shape[2])  # (N, D)
        return f_val, g_val

    # Generate Data
    N = 10_000
    D = 2
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(N, D))

    k, clusters = adaptive_cluster_algorithm(
        training_data=X,
        eval_function=evaluate_function,
        atol=TOL,
        rtol=TOL,
        N_init_clusters=1,
        min_ppc=50,
        use_gradient=True,
        seed=0,
    )
    # flatten the clusters
    centroids, QoI_values, gradients, trust_ellipses, n_pts, tree_levels = (
        flatten_cluster_centers_with_gradient(k, D, 1, clusters)
    )

    # Plot centroids with ellipses on contourf
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    F_grid = shifted_gaussians(X_grid, centers=CENTERS_2D).reshape(X1.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, F_grid, levels=50, cmap="viridis")

    for i in range(len(centroids)):
        mu = centroids[i]
        Q = trust_ellipses[i]
        # Compute eigenvalues and eigenvectors of Q
        w, V = np.linalg.eigh(Q)
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]
        a = 1.0 / np.sqrt(w[0])
        b = 1.0 / np.sqrt(w[1])
        angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

        ellipse = Ellipse(
            xy=mu,
            width=2 * a,
            height=2 * b,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            lw=2,
            alpha=0.7,
        )
        plt.gca().add_patch(ellipse)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=50)
    plt.colorbar(label="Function Value")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(
        f"Adaptive Clustering with Ellipsoids on Shifted Gaussians (TOL={TOL}, N = {len(centroids)})"
    )
    fname = "ben_ac_code_gaussian.png"
    plt.savefig(fname)
    plt.close()

    # -----------------------------------------
    # Quadratic Function
    def evaluate_quadratic(x):
        # If x is 1D, make it 2D
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        f_val = quadratic(x).flatten()  # (N, )
        g_val = quadratic_grad(x)  # (N, 1, D)
        g_val = g_val.reshape(g_val.shape[0], g_val.shape[2])  # (N, D)
        return f_val, g_val

    # Generate Data
    N = 10_000
    D = 2
    rng = np.random.default_rng(0)
    X = rng.uniform(-10, 10, size=(N, D))

    k, clusters = adaptive_cluster_algorithm(
        training_data=X,
        eval_function=evaluate_quadratic,
        atol=TOL,
        rtol=TOL,
        N_init_clusters=1,
        min_ppc=50,
        use_gradient=True,
        seed=0,
    )
    centroids, QoI_values, gradients, trust_ellipses, n_pts, tree_levels = (
        flatten_cluster_centers_with_gradient(k, D, 1, clusters)
    )

    # Plot centroids with ellipses on contourf
    x1 = np.linspace(-10, 10, 100)
    x2 = np.linspace(-10, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    F_grid = quadratic(X_grid).reshape(X1.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, F_grid, levels=50, cmap="viridis")

    for i in range(len(centroids)):
        mu = centroids[i]
        Q = trust_ellipses[i]
        # Compute eigenvalues and eigenvectors of Q
        w, V = np.linalg.eigh(Q)
        order = np.argsort(w)[::]
        w = w[order]
        V = V[:, order]
        a = 1.0 / np.sqrt(w[0])
        b = 1.0 / np.sqrt(w[1])
        angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

        ellipse = Ellipse(
            xy=mu,
            width=2 * a,
            height=2 * b,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            lw=2,
            alpha=0.7,
        )
        plt.gca().add_patch(ellipse)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=50)
    plt.colorbar(label="Function Value")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(
        f"Adaptive Clustering with Ellipsoids on Quadratic Function (TOL={TOL}, N = {len(centroids)})"
    )
    fname = "ben_ac_code_quadratic.png"
    plt.savefig(fname)
    plt.close()
    print(f'Saved figure to "{fname}"')
