
import numpy as np
from collections.abc import Callable
from sklearn.cluster import KMeans
from ellipse_test import metric_adaptive_two_means_PCA_clustering, \
                         fit_ellipse_PCA_scaled, solve_hessian_least_squares_symm


# bi-tree structures that we use in the recursive algorithm implemented below
class ClusterTree:

    def __init__(self, params, left, right):
        self.left = left
        self.right = right
        self.params = params
    

class Cluster:

    def __init__(self, index, params, QoI_value, gradient, trust_ellipse, n_pts):
        self.index = index # leaf index
        self.params = params # centroids of the cluster
        self.QoI_value = QoI_value # output value
        self.gradient = gradient # gradient value
        self.trust_ellipse = trust_ellipse # ellipse that defines analytic trust region
        self.n_pts = n_pts # number of points that generate the cluster



def adaptive_cluster_algorithm(
    training_data: np.ndarray, # each row contains a set of parameters used to cluster
    eval_function: Callable, # callable function; takes numpy array to numpy array
        # if use gradient is true, eval_function should return both value and gradient.
        # otherwise eval_function only returns value
        # modify so that if this takes a vector to a vector and a matrix (R^n to R^m),
        # it still works
    atol: float = None,
    rtol: float = None,
    N_init_clusters: int = 10, # number of initial clusters to split into
    min_ppc: int = 1, # minimum number of points used in a cluster (to measure error)
    use_gradient: bool = False,
    function_vals: np.ndarray = None, # if this is not none, uses these as solves. Otherwise runs eval_function every time
    gradients: np.ndarray = None,
    seed = None, # seed used for kmeans rng
    num_levels = -1, # specify the number of levels for the algorithm to go in the tree.
        # If this is -1, then just go forever.  Otherwise only go up to this level in the tree.
    soft_clustering_factor = 0, # used for clustering "blurring" to attempt to make trust regions better
    ):

    # check dimensions of training data; add another if necessary
    if training_data.ndim == 1:
        training_data = np.expand_dims(training_data, axis=1)
    
    # also check dimensions of function vals
    # do solves if necessary. TODO: specify dimension of output of eval_function
    if function_vals is None or gradients is None:
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
        raise Exception('Error: length of training data and function values lists do not match')

    k = 0 # counting number of clusters as we go
    cluster_structure = np.empty(shape=N_init_clusters, dtype=object)

    centroids, labels = _cluster(training_data, N_init_clusters, seed)

    fmax = np.max(np.abs(function_vals))

    for j in range(N_init_clusters):

        params_in_cluster = training_data[labels==j]
        fvals_in_cluster = function_vals[labels==j]
        grads_in_cluster = gradients[labels==j]
            
        centroid_params = centroids[j]
        fval_c, grad_c = _compute_oracle_at_point(centroid_params, eval_function, use_gradient)

        # check size of gradient to see if it is appropriate
        if use_gradient and grad_c.shape != (m,n):
            raise Exception('Error: gradient function does return shape m*n')

        k, cluster_structure[j] = _adaptive_cluster_algorithm_helper(
            params_in_cluster,
            fvals_in_cluster,
            grads_in_cluster,
            centroid_params,
            fval_c,
            grad_c,
            eval_function,
            fmax,
            atol,
            rtol,
            min_ppc,
            use_gradient,
            k,
            seed,
            num_levels-1,
            soft_clustering_factor
        )

    return k, cluster_structure



def _adaptive_cluster_algorithm_helper(
    training_data: np.ndarray,
    function_vals: np.ndarray,
    gradients: np.ndarray,
    prev_centroid: np.ndarray,
    f_prev_centroid: np.ndarray,
    grad_prev_centroid: np.ndarray,
    eval_function: Callable,
    fmax: np.ndarray,
    atol = None,
    rtol = None,
    min_ppc: int = 1,
    use_gradient: bool = False,
    k: int = 0, # current cluster index (for recursive purposes)
    seed = None,
    num_levels = -1,
    soft_clustering_factor = 0,
):

    if len(training_data) <= 1:
        # Dead branch of the tree. Not enough variance of points to compute error (or a single point).
        return k, None

    # Add extra points to the cluster (randomly sampled) until we get to the minimum ppc
    training_data, function_vals, gradients = \
        _generate_extra_points(training_data, function_vals, gradients, eval_function, min_ppc, use_gradient)

    # With metric adaptive clustering, we check error first
    # prev_centroid is the centroid of the data
    # check linear interpolant fit around this centroid
    distance = training_data - prev_centroid
    if not use_gradient:
        errors = function_vals - f_prev_centroid
    elif use_gradient:
        errors = function_vals - (f_prev_centroid + (distance @ grad_prev_centroid.T))

    error_norms = np.linalg.norm(errors, ord=np.inf, axis=1)
    with_good_error = np.logical_or(error_norms <= atol, error_norms <= (rtol * fmax))

    # from this, decide if we should split or not
    if with_good_error.all() or num_levels == 0:
        
        # Create a cluster. Use ellipse formulation
        df_diffs = gradients - grad_prev_centroid
        trust_ellipse = solve_hessian_least_squares_symm(distance, df_diffs, absolute_value=True)
        # trust_ellipse = fit_ellipse_SDP_khachiyan(distance)
        k += 1
        return k, Cluster(k-1, prev_centroid, f_prev_centroid, grad_prev_centroid, trust_ellipse, len(errors))
        

    # Otherwise we plan to split the cluster. Use metric-adaptive.

    # sort points, compute optimal ellipse for clustering
    _, dim = training_data.shape
    df_diffs = gradients - grad_prev_centroid
    Q = solve_hessian_least_squares_symm(distance, df_diffs, absolute_value=True)
    
    # with this psd matrix in hand, do metric adaptive clustering
    centroids_scaled, labels = metric_adaptive_two_means_PCA_clustering(distance, Q, soft_clustering_factor)
    centroids = centroids_scaled + prev_centroid

    # First cluster (N=0)
    pts_cluster = training_data[labels <= 0.5]
    fvals_cluster = function_vals[labels <= 0.5]
    grads_cluster = gradients[labels <= 0.5]
    centroid_params = centroids[0]
    fval_c, grad_c = _compute_oracle_at_point(centroid_params, eval_function, use_gradient)

    k, left = _adaptive_cluster_algorithm_helper(
            pts_cluster, fvals_cluster, grads_cluster, centroid_params, fval_c, grad_c, eval_function,
            fmax, atol, rtol, min_ppc, use_gradient, k, seed, num_levels-1, soft_clustering_factor)

    # Second cluster (N=1)
    pts_cluster = training_data[labels >= 0.5]
    fvals_cluster = function_vals[labels >= 0.5]
    grads_cluster = gradients[labels >= 0.5]
    centroid_params = centroids[1]
    fval_c, grad_c = _compute_oracle_at_point(centroid_params, eval_function, use_gradient)

    k, right = _adaptive_cluster_algorithm_helper(
            pts_cluster, fvals_cluster, grads_cluster, centroid_params, fval_c, grad_c, eval_function,
            fmax, atol, rtol, min_ppc, use_gradient, k, seed, num_levels-1, soft_clustering_factor)


    return k, ClusterTree(prev_centroid, left, right)



# At the moment, this implementation is really bad and will be slow. Redo how we sample (for loop).
# We require more than one point in the cluster (this number can change).
def _generate_extra_points(pts_cluster, fvals_cluster, grads_cluster, oracle, min_ppc, use_gradient):

    n_pts_sample = min_ppc - len(pts_cluster)

    if n_pts_sample <= 0:
        return pts_cluster, fvals_cluster, grads_cluster

    # take random convex combinations
    lambdas = np.random.uniform(0, 1, n_pts_sample)
    indices = np.random.randint(0, len(pts_cluster), (2, n_pts_sample))

    # deal with columns where the indices are the same
    # probably a much better way to do this...
    resample = (indices[1] - indices[0]) == 0
    all_inds = np.arange(len(pts_cluster))
    for j, val in enumerate(resample):
        if j:
            indices[:,j] = np.random.choice(all_inds, 2, replace=False)

    # construct convex combinations
    new_pts_cluster = np.multiply(pts_cluster[indices[0]], lambdas[:, np.newaxis]) + \
            np.multiply(pts_cluster[indices[1]], (1 - lambdas)[:, np.newaxis])

    # do ode solves for these new points
    new_fvals_cluster = np.zeros(shape=(n_pts_sample, fvals_cluster.shape[1]))
    new_grads_cluster = np.zeros(shape=(n_pts_sample, grads_cluster.shape[1], grads_cluster.shape[2]))
    for i, row in enumerate(new_pts_cluster):
        # compute true value (save in qi)
        if use_gradient:
            new_fvals_cluster[i], new_grads_cluster[i] = oracle(row)
        else:
            new_fvals_cluster[i] = oracle(row)

    # put these new points together with the old ones
    pts_cluster = np.vstack((pts_cluster, new_pts_cluster))
    fvals_cluster = np.vstack((fvals_cluster, new_fvals_cluster))
    grads_cluster = np.vstack((grads_cluster, new_grads_cluster))

    return pts_cluster, fvals_cluster, grads_cluster


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
                val, k, lev+1, centroids, QoI_values,
                gradients, trust_ellipses, n_pts, tree_levels)

    return centroids, QoI_values, gradients, trust_ellipses, n_pts, tree_levels



def _flatten_cluster_centers_with_gradient_helper(
        tree,
        k,
        lev,
        centroids,
        QoI_values,
        gradients,
        trust_ellipses,
        n_pts,
        tree_levels
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
            tree.left, k, lev+1, centroids, QoI_values,
            gradients, trust_ellipses, n_pts, tree_levels)
        k = _flatten_cluster_centers_with_gradient_helper(
            tree.right, k, lev+1, centroids, QoI_values,
            gradients, trust_ellipses, n_pts, tree_levels)
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

