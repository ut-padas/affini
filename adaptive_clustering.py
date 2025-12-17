
import numpy as np
from collections.abc import Callable
from sklearn.cluster import KMeans


# bi-tree structures that we use in the recursive algorithm implemented below
class ClusterTree:

    def __init__(self, params, left, right):
        self.left = left
        self.right = right
        self.params = params
    

class Cluster:

    def __init__(self, index, params, QoI_value, gradient, lipschitz, radius, n_pts, lipschitz_var):
        self.index = index # leaf index
        self.params = params # centroids of the cluster
        self.QoI_value = QoI_value # output value
        self.gradient = gradient # gradient value
        self.lipschitz = lipschitz # hessian
        self.radius = radius # trust radius
        self.n_pts = n_pts # number of points that generate the cluster
        self.lipschitz_var = lipschitz_var # variance of Lipschitz constant (the one we use is the max)


    def __str__(self):
        return f'k={self.index}\tn_h={10**self.params[0]:.3f}\tT={10**self.params[1]:.3f}\tG0={self.params[2]:.6f}'



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
    seed = None, # seed used for kmeans rng
    num_levels = -1, # specify the number of levels for the algorithm to go in the tree.
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
        raise Exception('Error: length of training data and function values lists do not match')

    # normalize the training data
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    training_data_normalized = (training_data - mean) / std

    k = 0 # counting number of clusters as we go

    # Call clustering library
    km = KMeans(n_clusters=N_init_clusters, random_state=seed)
    labels = km.fit_predict(training_data_normalized)
    centroids_normalized = km.cluster_centers_
    centroids = centroids_normalized * std + mean

    cluster_structure = np.empty(shape=N_init_clusters, dtype=object)

    for j in range(N_init_clusters):

        # Compute error inside each cluster
        params_in_cluster = training_data[labels==j]
        fvals_cluster = function_vals[labels==j]
            
        # Compute exact value and gradient at cluster centroid
        # NOTE: if using KMedoids, we don't have to do a solve. Just pull the value
        centroid_params = centroids[j]
        if use_gradient:
            fval_c, grad_c = eval_function(centroid_params)
            # TODO: make this solve vectorized in one go (for scaling)
            if np.isscalar(grad_c):
                grad_c = np.array([grad_c])
            if grad_c.ndim == 1:
                grad_c = np.expand_dims(grad_c, axis=1)
        else:
            fval_c = eval_function(centroid_params)
            grad_c = None
        if np.isscalar(fval_c):
            fval_c = np.array([fval_c])

        # check size of gradient to see if it is appropriate
        if use_gradient and grad_c.shape != (m,n):
            raise Exception('Error: gradient function does return shape m*n')

        # Compute error by taking largest distances away and averaging errors
        distance = params_in_cluster - centroid_params
        if not use_gradient:
            evec = np.linalg.norm(fval_c-fvals_cluster, 2, axis=1) # TODO: what norm to use?
        elif use_gradient:
            # make sure we compute in euclidean distance
            evec = np.linalg.norm(fvals_cluster - (fval_c + np.matmul(distance, grad_c.T)), 2, axis=1) # TODO: what norm to use?
        
        abs_error = np.linalg.norm(evec, 2)
        rel_error = abs_error / np.linalg.norm(fvals_cluster, 'fro')
        radii = np.linalg.norm(distance, axis=1)
        radius = np.max(radii)

        if (rtol is not None and rel_error < rtol) \
                or (atol is not None and abs_error < atol) \
                or num_levels == 0:
            # Tolerance is good; save values in cluster column
            if use_gradient:
                lipschitz_constants = 2 / (radii**2) * evec
            else:
                lipschitz_constants = evec / radii
            lipschitz = np.max(lipschitz_constants)
            lipschitz_var = np.var(lipschitz_constants)
            cluster_structure[j] = Cluster(k, centroid_params, fval_c, grad_c, lipschitz, radius, len(evec), lipschitz_var)
            k += 1
        else:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = _adaptive_cluster_algorithm_helper(
                    params_in_cluster, fvals_cluster, centroid_params, eval_function, N_init_clusters * 2,
                    atol, rtol, min_ppc, use_gradient, k, seed, num_levels-1)

    return k, cluster_structure



def _adaptive_cluster_algorithm_helper(
    training_data: np.ndarray,
    function_vals: np.ndarray,
    prev_centroid: np.ndarray,
    eval_function: Callable,
    total_clusters: float, # required to compute global absolute error
    atol = None,
    rtol = None,
    min_ppc: int = 1,
    use_gradient: bool = False,
    k: int = 0, # current cluster index (for recursive purposes)
    seed = None,
    num_levels = -1,
):
    
    # record dimensions of matrices
    p1, n = training_data.shape
    p2, m = function_vals.shape
    if p1 != p2:
        raise Exception('Length of training data and function vals does not match')

    # normalize for clustering
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    # quick fix to the issue if std is 0 in one column
    if (std == 0).any():
        std += (std == 0) * 1
    params_normalized = (training_data - mean) / std

    N = 2  # split in half
    km = KMeans(n_clusters=N, random_state=seed)
    labels = km.fit_predict(params_normalized)
    centroids_normalized = km.cluster_centers_
    centroids = centroids_normalized * std + mean

    # First cluster (N=0)
    pts_cluster = training_data[labels==0]
    fvals_cluster = function_vals[labels==0]

    centroid_params = centroids[0]
    if use_gradient:
        fval_c, grad_c = eval_function(centroid_params)
        if np.isscalar(grad_c):
            grad_c = np.array([grad_c])
        if grad_c.ndim == 1:
            grad_c = np.expand_dims(grad_c, axis=1)
    else:
        fval_c = eval_function(centroid_params)
        grad_c = None
    if np.isscalar(fval_c):
        fval_c = np.array([fval_c])

    # Add extra points to the cluster (randomly sampled) until we get to the minimum ppc
    # At the moment, this implementation is really bad and will be slow. Redo how we sample (for loop).
    # We require more than one point in the cluster (this number can change).
    if len(pts_cluster) <= min_ppc and len(pts_cluster) > 1:
        new_pts_cluster, new_fval_vec = \
            _generate_extra_points(pts_cluster, fvals_cluster, eval_function, min_ppc, use_gradient)
    else:
        new_pts_cluster = np.zeros(shape=(0,n))
        new_fval_vec = np.zeros(shape=(0,m))


    # put old points and new points together into a single list
    if len(new_fval_vec) > 0:
        pts_cluster = np.vstack((pts_cluster, new_pts_cluster))
        fvals_cluster = np.vstack((fvals_cluster, new_fval_vec))

    # using true value, compute error from qc
    distance = pts_cluster - centroid_params
    if not use_gradient:
        evec = np.linalg.norm(fval_c-fvals_cluster, 2, axis=1)
    elif use_gradient:
        # make sure we compute in euclidean, non-log scaled distance
        evec = np.linalg.norm(fvals_cluster - (fval_c + np.matmul(distance, grad_c.T)), 2, axis=1)
        
    abs_error = np.linalg.norm(evec, 2)
    rel_error = abs_error / np.linalg.norm(fvals_cluster, 'fro')
    radii = np.linalg.norm(distance, axis=1)
    radius = np.max(radii)

    if radius == 0:
        # Dead branch of the tree. Not enough points to compute error
        left = None
    elif (rtol is not None and rel_error < rtol) \
            or (atol is not None and abs_error < (atol / np.sqrt(total_clusters))) \
            or num_levels == 0:
        # Error good
        if use_gradient:
            lipschitz_constants = 2 / (radii**2) * evec
        else:
            lipschitz_constants = evec / radii
        lipschitz = np.max(lipschitz_constants)
        lipschitz_var = np.var(lipschitz_constants)
        left = Cluster(k, centroid_params, fval_c, grad_c, lipschitz, radius, len(evec), lipschitz_var)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, left = _adaptive_cluster_algorithm_helper(
                pts_cluster, fvals_cluster, centroid_params, eval_function, total_clusters * 2,
                atol, rtol, min_ppc, use_gradient, k, seed, num_levels-1)



    # Second cluster (N=1)
    pts_cluster = training_data[labels==1]
    fvals_cluster = function_vals[labels==1]
    
    centroid_params = centroids[1]
    if use_gradient:
        fval_c, grad_c = eval_function(centroid_params)
        if np.isscalar(grad_c):
            grad_c = np.array([grad_c])
        if grad_c.ndim == 1:
            grad_c = np.expand_dims(grad_c, axis=1)
    else:
        fval_c = eval_function(centroid_params)
        grad_c = None
    if np.isscalar(fval_c):
        fval_c = np.array([fval_c])


    # Add extra points to the cluster (randomly sampled) until we get to the minimum ppc
    if len(pts_cluster) <= min_ppc and len(pts_cluster) > 1:
        new_pts_cluster, new_fval_vec = \
            _generate_extra_points(pts_cluster, fvals_cluster, eval_function, min_ppc, use_gradient)
    else:
        new_pts_cluster = []
        new_fval_vec = []


    # put old points and new points together into a single list
    if len(new_fval_vec) > 0:
        pts_cluster = np.vstack((pts_cluster, new_pts_cluster))
        fvals_cluster = np.vstack((fvals_cluster, new_fval_vec))
    

    # using true value, compute error from qc
    distance = pts_cluster - centroid_params
    if not use_gradient:
        evec = np.linalg.norm(fval_c-fvals_cluster, 2, axis=1) # TODO: what norm to use?
    elif use_gradient:
        # make sure we compute in euclidean, non-log scaled distance
        evec = np.linalg.norm(fvals_cluster - (fval_c + np.matmul(distance, grad_c.T)), 2, axis=1) # TODO: what norm to use?

    abs_error = np.linalg.norm(evec, 2)
    rel_error = abs_error / np.linalg.norm(fvals_cluster, 'fro')
    radii = np.linalg.norm(distance, axis=1)
    radius = np.max(radii)

    if radius == 0:
        # Dead branch of the tree. Not enough variance of points to compute error (or a single point).
        right = None
    elif (rtol is not None and rel_error <= rtol) \
            or (atol is not None and abs_error <= (atol / np.sqrt(total_clusters))) \
            or num_levels == 0:
        if use_gradient:
            lipschitz_constants = 2 / (radii**2) * evec
        else:
            lipschitz_constants = evec / radii
        lipschitz = np.max(lipschitz_constants)
        lipschitz_var = np.var(lipschitz_constants)
        right = Cluster(k, centroid_params, fval_c, grad_c, lipschitz, radius, len(evec), lipschitz_var)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, right = _adaptive_cluster_algorithm_helper(
                pts_cluster, fvals_cluster, centroid_params, eval_function, total_clusters * 2,
                atol, rtol, min_ppc, use_gradient, k, seed, num_levels-1)

    return k, ClusterTree(prev_centroid, left, right)



def _generate_extra_points(pts_cluster, fvals_cluster, oracle, min_ppc, use_gradient):

    n_pts_sample = min_ppc - len(pts_cluster)

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
    for i, row in enumerate(new_pts_cluster):
        # compute true value (save in qi)
        if use_gradient:
            new_fvals_cluster[i], _ = oracle(row)
        else:
            new_fvals_cluster[i] = oracle(row)

    return new_pts_cluster, new_fvals_cluster




def flatten_cluster_centers_with_gradient(nc, n, m, tree):
    centroids = np.zeros(shape=(nc, n))
    QoI_values = np.zeros(shape=(nc, m))
    gradients = np.zeros(shape=(nc, m, n))
    lipschitzes = np.zeros(shape=nc)
    radii = np.zeros(shape=nc)
    n_pts = np.zeros(shape=nc)
    lipschitz_vars = np.zeros(shape=nc)
    tree_levels = np.zeros(shape=nc)

    k = 0
    lev = 0
    for val in tree:
        if type(val) is Cluster:
            centroids[k] = val.params
            QoI_values[k] = val.QoI_value
            if val.gradient is None:
                gradients[k] = np.zeros((m,n))
            else:
                gradients[k] = val.gradient
            lipschitzes[k] = val.lipschitz
            lipschitz_vars[k] = val.lipschitz_var
            radii[k] = val.radius
            n_pts[k] = val.n_pts
            tree_levels[k] = lev
            k += 1
        elif type(val) is ClusterTree:
            k = _flatten_cluster_centers_with_gradient_helper(
                val.left, k, lev+1, centroids, QoI_values, gradients,
                lipschitzes, radii, n_pts, lipschitz_vars, tree_levels)
            k = _flatten_cluster_centers_with_gradient_helper(
                val.right, k, lev+1, centroids, QoI_values, gradients,
                lipschitzes, radii, n_pts, lipschitz_vars, tree_levels)
    return centroids, QoI_values, gradients, lipschitzes, radii, n_pts, lipschitz_vars, tree_levels



def _flatten_cluster_centers_with_gradient_helper(
        tree,
        k,
        lev,
        centroids,
        QoI_values,
        gradients,
        lipschitzes,
        radii,
        n_pts,
        lipschitz_vars,
        tree_levels
    ):
    if type(tree) is Cluster:
        centroids[k] = tree.params
        QoI_values[k] = tree.QoI_value
        if tree.gradient is None:
            gradients[k] = np.zeros((len(QoI_values[k]), len(centroids[k])))
        else:
            gradients[k] = tree.gradient
        lipschitzes[k] = tree.lipschitz
        lipschitz_vars[k] = tree.lipschitz_var
        radii[k] = tree.radius
        n_pts[k] = tree.n_pts
        tree_levels[k] = lev
        k += 1
    elif type(tree) is ClusterTree:
        k = _flatten_cluster_centers_with_gradient_helper(
            tree.left, k, lev+1, centroids, QoI_values, gradients,
            lipschitzes, radii, n_pts, lipschitz_vars, tree_levels)
        k = _flatten_cluster_centers_with_gradient_helper(
            tree.right, k, lev+1, centroids, QoI_values, gradients,
            lipschitzes, radii, n_pts, lipschitz_vars, tree_levels)
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

