
import numpy as np
from collections.abc import Callable
from sklearn.cluster import KMeans
from kmedoids import KMedoids


# bi-tree structures that we use in the recursive algorithm implemented below
class ClusterTree:

    def __init__(self, params, left, right):
        self.left = left
        self.right = right
        self.params = params
    

class Cluster:

    def __init__(self, index, params, QoI_value, gradient, hessian, radius):
        self.index = index # leaf index
        self.params = params # centroids of the cluster
        self.QoI_value = QoI_value # output value 
        self.gradient = gradient # gradient value
        self.hessian = hessian # hessian
        self.radius = radius # trust radius TODO: change name

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
    use_gradient: bool = False, #TODO: make sure this runs (todo for a masters student)
    function_vals: np.ndarray = None, # if this is not none, uses these as solves. Otherwise runs eval_function every time
    seed = None, # seed used for kmeans rng
    ):

    # check dimensions of training data; add another if necessary
    if training_data.ndim == 1:
        training_data = np.expand_dims(training_data, axis=1)
    
    # also check dimensions of function vals
    # do solves if necessary. TODO: specify dimension of output of eval_function
    if function_vals is None:
        function_vals = np.zeros(len(training_data))
        if use_gradient == False:
            for j, row in enumerate(training_data):
                function_vals[j] = eval_function(row)
        else:
            for j, row in enumerate(training_data):
                function_vals[j], _ = eval_function(row)
    if function_vals.ndim == 1:
        function_vals = np.expand_dims(function_vals, axis=1)

    # normalize the training data
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    training_data_normalized = (training_data - mean) / std

    k = 0 # counting number of clusters as we go

    # Call clustering library
    if len(training_data_normalized) > 0: # TODO: talk about which clustering algorithm to use
        km = KMeans(n_clusters=N_init_clusters, random_state=seed)
    else:
        km = KMedoids(n_clusters=N_init_clusters, metric='euclidean', random_state=seed)
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
        if use_gradient == True:
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

        # Compute error by taking largest distances away and averaging errors
        if use_gradient == False:
            evec = np.linalg.norm(fval_c-fvals_cluster, 2, axis=1) # TODO: what norm to use?
        elif use_gradient == True:
            # make sure we compute in euclidean distance
            distance = params_in_cluster - centroid_params
            evec = np.linalg.norm(fvals_cluster - (fval_c + np.matmul(distance, grad_c.T)), 2, axis=1) # TODO: what norm to use?

        abs_error = np.max(evec)

        if (rtol is not None and abs_error < np.min(np.linalg.norm(fvals_cluster, 2, axis=1)) * rtol) or (atol is not None and abs_error < atol):
            # Tolerance is good; save values in cluster column
            radius = np.max(np.linalg.norm(distance, axis=1))
            hessian = 2 / (radius**2) * np.max(evec)
            cluster_structure[j] = Cluster(k, centroid_params, fval_c, grad_c, hessian, radius)
            k += 1
        else:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = _adaptive_cluster_algorithm_helper(
                    params_in_cluster, centroid_params, eval_function,
                    atol, rtol, min_ppc, use_gradient, fvals_cluster, k, seed)

    return k, cluster_structure



def _adaptive_cluster_algorithm_helper(
    params: np.ndarray,
    prev_centroid: np.ndarray,
    eval_function: Callable,
    atol = None,
    rtol = None,
    min_ppc: int = 1,
    use_gradient: bool = False,
    function_vals: np.ndarray = None,
    k: int = 0, # current cluster index (for recursive purposes)
    seed = None,
):

    # normalize for clustering
    mean = params.mean(axis=0)
    std = params.std(axis=0)
    # quick fix to the issue if std is 0 in one column
    if (std == 0).any():
        std += (std == 0) * 1
    params_normalized = (params - mean) / std

    N = 2  # split in half
    if len(params_normalized) > 0: # TODO: talk about which clustering algorithm to use
        km = KMeans(n_clusters=N, random_state=seed)
    else:
        km = KMedoids(n_clusters=N, metric='euclidean', random_state=seed)
    labels = km.fit_predict(params_normalized)
    centroids_normalized = km.cluster_centers_
    centroids = centroids_normalized * std + mean

    # First cluster (N=0)
    params_in_cluster_0 = params[labels==0]
    fvals_cluster = function_vals[labels==0]

    centroid_params = centroids[0]
    if use_gradient == True:
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
    if len(params_in_cluster_0) <= min_ppc and len(params_in_cluster_0) > 1:

        num_points_to_sample = min_ppc - len(params_in_cluster_0)
        # take random convex combinations
        lambdas = np.random.uniform(0, 1, num_points_to_sample)
        indices = np.random.randint(0, len(params_in_cluster_0), (2, num_points_to_sample))
        # deal with columns where the indices are the same
        resample = (indices[1] - indices[0]) == 0
        all_inds = np.arange(len(params_in_cluster_0))
        for j, val in enumerate(resample):
            if j == True:
                indices[:,j] = np.random.choice(all_inds, 2, replace=False)
        # construct convex combinations
        new_params_in_cluster_0 = np.multiply(params_in_cluster_0[indices[0]], lambdas[:, np.newaxis]) + \
                np.multiply(params_in_cluster_0[indices[1]], (1 - lambdas)[:, np.newaxis])

        # do ode solves for these new points
        new_fval_vec = np.zeros(shape=(len(new_params_in_cluster_0),function_vals.shape[1]))
        for i, row in enumerate(new_params_in_cluster_0):
            # compute true value (save in qi)
            if use_gradient == False:
                f = eval_function(row)
            elif use_gradient == True:
                f, _ = eval_function(row)
            if np.isscalar(f):
                new_fval_vec[i] = np.array([f])
            else:
                new_fval_vec[i] = f

    else:
        new_fval_vec = []


    # put old points and new points together into a single list
    if len(new_fval_vec) > 0:
        params_in_cluster_0 = np.vstack((params_in_cluster_0, new_params_in_cluster_0))
        fvals_cluster = np.vstack((fvals_cluster, new_fval_vec))

    # using true value, compute error from qc
    if use_gradient == False:
        evec = np.linalg.norm(fval_c-fvals_cluster, 2, axis=1) # TODO: what norm to use?
    elif use_gradient == True:
        # make sure we compute in euclidean, non-log scaled distance
        distance = params_in_cluster_0 - centroid_params
        evec = np.linalg.norm(fvals_cluster - (fval_c + np.matmul(distance, grad_c.T)), 2, axis=1) # TODO: what norm to use?
        

    abs_error = np.max(evec)

    radius = np.max(np.linalg.norm(distance, axis=1))
    if radius == 0:
        # Dead branch of the tree. Not enough points to compute error
        left = None
    elif (rtol is not None and abs_error < np.min(np.linalg.norm(fvals_cluster, 2, axis=1)) * rtol) or (atol is not None and abs_error < atol):
        # Error good
        hessian = 2 / (radius**2) * np.max(evec)
        left = Cluster(k, centroid_params, fval_c, grad_c, hessian, radius)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, left = _adaptive_cluster_algorithm_helper(
                params_in_cluster_0, centroid_params, eval_function,
                atol, rtol, min_ppc, use_gradient,
                fvals_cluster, k, seed)



    # Second cluster (N=1)
    params_in_cluster_1 = params[labels==1]
    fvals_cluster = function_vals[labels==1]
    
    centroid_params = centroids[1]
    if use_gradient == True:
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
    if len(params_in_cluster_1) <= min_ppc and len(params_in_cluster_1) > 1:

        num_points_to_sample = min_ppc - len(params_in_cluster_1)
        # take random convex combinations
        lambdas = np.random.uniform(0, 1, num_points_to_sample)
        indices = np.random.randint(0, len(params_in_cluster_1), (2, num_points_to_sample))
        # deal with columns where the indices are the same
        resample = (indices[1] - indices[0]) == 0
        all_inds = np.arange(len(params_in_cluster_1))
        for j, val in enumerate(resample):
            if j == True:
                indices[:,j] = np.random.choice(all_inds, 2, replace=False)
        # construct convex combinations
        new_params_in_cluster_1 = np.multiply(params_in_cluster_1[indices[0]], lambdas[:, np.newaxis]) + \
                np.multiply(params_in_cluster_1[indices[1]], (1 - lambdas)[:, np.newaxis])

        # do ode solves for these new points
        new_fval_vec = np.zeros(shape=(len(new_params_in_cluster_1),function_vals.shape[1]))
        for i, row in enumerate(new_params_in_cluster_1):
            # compute true value (save in qi)
            if use_gradient == False:
                f = eval_function(row)
            elif use_gradient == True:
                f, _ = eval_function(row)
            if np.isscalar(f):
                new_fval_vec[i] = np.array([f])
            else:
                new_fval_vec[i] = f
            
        
    else:
        new_fval_vec = []


    # put old points and new points together into a single list
    if len(new_fval_vec) > 0:
        params_in_cluster_1 = np.vstack((params_in_cluster_1, new_params_in_cluster_1))
        fvals_cluster = np.vstack((fvals_cluster, new_fval_vec))
    

    # using true value, compute error from qc
    if use_gradient == False:
        evec = np.linalg.norm(fval_c-fvals_cluster, 2, axis=1) # TODO: what norm to use?
    elif use_gradient == True:
        # make sure we compute in euclidean, non-log scaled distance
        distance = params_in_cluster_1 - centroid_params
        evec = np.linalg.norm(fvals_cluster - (fval_c + np.matmul(distance, grad_c.T)), 2, axis=1) # TODO: what norm to use?


    abs_error = np.max(evec)

    radius = np.max(np.linalg.norm(distance, axis=1))
    if radius == 0:
        # Dead branch of the tree. Not enough variance of points to compute error (or a single point).
        right = None
    elif (rtol is not None and abs_error <= np.min(np.linalg.norm(fvals_cluster, 2, axis=1)) * rtol) or (atol is not None and abs_error < atol):
        hessian = 2 / (radius**2) * np.max(evec)
        right = Cluster(k, centroid_params, fval_c, grad_c, hessian, radius)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, right = _adaptive_cluster_algorithm_helper(
                params_in_cluster_1, centroid_params, eval_function,
                atol, rtol, min_ppc, use_gradient, fvals_cluster, k, seed)

    return k, ClusterTree(prev_centroid, left, right)



def flatten_cluster_centers_with_gradient(nc, n, m, tree):
    centroids = np.zeros(shape=(nc, n))
    QoI_values = np.zeros(shape=(nc, m))
    gradients = np.zeros(shape=(nc, m, n))
    hessians = np.zeros(shape=nc)
    dims = np.zeros(shape=nc)

    k = 0
    for val in tree:
        if type(val) == Cluster:
            centroids[k] = val.params
            QoI_values[k] = val.QoI_value
            gradients[k] = val.gradient
            hessians[k] = val.hessian
            dims[k] = val.radius
            k += 1
        elif type(val) == ClusterTree:
            k = _flatten_cluster_centers_with_gradient_helper(val.left, k, centroids, QoI_values, gradients, hessians, dims)
            k = _flatten_cluster_centers_with_gradient_helper(val.right, k, centroids, QoI_values, gradients, hessians, dims)
    return centroids, QoI_values, gradients, hessians, dims



def _flatten_cluster_centers_with_gradient_helper(tree, k, centroids, QoI_values, gradients, hessians, dims):
    if type(tree) == Cluster:
        centroids[k] = tree.params
        QoI_values[k] = tree.QoI_value
        gradients[k] = tree.gradient
        hessians[k] = tree.hessian
        dims[k] = tree.radius
        k += 1
    elif type(tree) == ClusterTree:
        k = _flatten_cluster_centers_with_gradient_helper(tree.left, k, centroids, QoI_values, gradients, hessians, dims)
        k = _flatten_cluster_centers_with_gradient_helper(tree.right, k, centroids, QoI_values, gradients, hessians, dims)
    return k




def flatten_cluster_centers(nc, n, m, tree):
    centroids = np.zeros(shape=(nc, n))
    QoI_values = np.zeros(shape=(nc, m))
    k = 0
    for val in tree:
        if type(val) == Cluster:
            centroids[k] = val.params
            QoI_values[k] = val.QoI_value
            k += 1
        elif type(val) == ClusterTree:
            k = _flatten_cluster_centers_helper(val.left, k, centroids, QoI_values)
            k = _flatten_cluster_centers_helper(val.right, k, centroids, QoI_values)
    return centroids, QoI_values



def _flatten_cluster_centers_helper(tree, k, centroids, QoI_values):
    if type(tree) == Cluster:
        centroids[k] = tree.params
        QoI_values[k] = tree.QoI_value
        k += 1
    elif type(tree) == ClusterTree:
        k = _flatten_cluster_centers_helper(tree.left, k, centroids, QoI_values)
        k = _flatten_cluster_centers_helper(tree.right, k, centroids, QoI_values)
    return k

