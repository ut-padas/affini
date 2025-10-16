
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

    def __init__(self, index, params, QoI_value, gradient, hessian, dim):
        self.index = index
        self.params = params
        self.QoI_value = QoI_value
        self.gradient = gradient
        self.hessian = hessian
        self.dim = dim

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
    unpack_log = [],
    function_vals: np.ndarray = None, # if this is not none, uses these as solves. Otherwise runs eval_function every time
    seed = None, # seed used for kmeans rng
    ):

    # check dimensions of function vals
    if function_vals.ndim == 1:
        function_vals = function_vals.reshape(len(function_vals),1)

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
        if function_vals is None:
            function_vals_in_cluster = None
        else:
            function_vals_in_cluster = function_vals[labels==j]
            
        # Compute exact value and gradient at cluster centroid
        # NOTE: if using KMedoids, we don't have to do a solve. Just pull the value
        centroid_params = centroids[j]
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
        

        # Compute error via max_{j=1,...,ss}(q_c - q_j) / q_c
        # param_sample = params_in_cluster.sample(ss_new)
        # dvec = np.empty(shape=ss_new)
        # for i, row in param_sample.reset_index(drop=True).iterrows():
        #     qi = self._solve_nelson_network(row.to_numpy(), x0, QoI, time)
        #     dvec[i] = np.abs(qc-qi)
        # error = np.max(dvec) / qc

        # Compute error by taking largest distances away and averaging errors
        if function_vals_in_cluster is None:

            # ODE solves not defined. We need to loop through and solve
            fval_vec = np.empty(shape=(len(params_in_cluster),len(fval_c)))
            for i, row in enumerate(params_in_cluster):
                # compute true value (save in qi)
                if use_gradient == False:
                    f = eval_function(row)
                elif use_gradient == True:
                    f, _ = eval_function(row)
                if np.isscalar(f):
                    fval_vec[i] = np.array([f])
                else:
                    fval_vec[i] = f

        else:

            # ODE solves are defined. No need to loop through
            # Also avoid the sample size and use every single point (we get away with it for free)
            fval_vec = function_vals_in_cluster

        # using true value, compute error from qc
        if use_gradient == False:
            evec = np.linalg.norm(fval_c-fval_vec, 2, axis=1) # TODO: what norm to use?
        elif use_gradient == True:
            # make sure we compute in euclidean, non-log scaled distance
            # use unpack_log to know which variables to do
            params_nonlog = np.copy(params_in_cluster)
            centroid_nonlog = np.copy(centroid_params)
            if len(unpack_log) > 0:
                for l, val in enumerate(unpack_log):
                    if val == True:
                        params_nonlog[:,l] = 10 ** params_nonlog[:,l]
                        centroid_nonlog[l] = 10 ** centroid_nonlog[l]
            distance = params_nonlog - centroid_nonlog
            evec = np.linalg.norm(fval_vec - (fval_c + np.matmul(distance, grad_c)), 2, axis=1) # TODO: what norm to use?

        abs_error = np.max(evec)

        if (rtol is not None and abs_error < np.min(np.linalg.norm(fval_vec, 2, axis=1)) * rtol) or (atol is not None and abs_error < atol):
            # Tolerance is good; save values in cluster column
            dim = np.max(np.linalg.norm(distance, axis=1))
            hessian = 2 / (dim**2) * np.max(evec)
            cluster_structure[j] = Cluster(k, centroid_params, fval_c, grad_c, hessian, dim)
            k += 1
        else:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = _adaptive_cluster_algorithm_helper(
                    params_in_cluster, centroid_params, eval_function,
                    atol, rtol, min_ppc, use_gradient, unpack_log,
                    function_vals_in_cluster, k, seed)

    return k, cluster_structure



def _adaptive_cluster_algorithm_helper(
    params: np.ndarray,
    prev_centroid: np.ndarray,
    eval_function: Callable,
    atol = None,
    rtol = None,
    min_ppc: int = 1,
    use_gradient: bool = False,
    unpack_log = [],
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
    if function_vals is None:
        function_vals_in_cluster_0 = None
    else:
        function_vals_in_cluster_0 = function_vals[labels==0]

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


    # Compute error by taking largest distances away and averaging errors
    if function_vals_in_cluster_0 is None:

        # ODE solves not defined. We need to loop through and solve
        fval_vec = np.empty(shape=(len(params_in_cluster_0),len(fval_c)))
        for i, row in enumerate(params_in_cluster_0):
            # compute true value (save in qi)
            if use_gradient == False:
                f = eval_function(row)
            elif use_gradient == True:
                f, _ = eval_function(row)
            if np.isscalar(f):
                fval_vec[i] = np.array([f])
            else:
                fval_vec[i] = f

    else:

        # ODE solves are defined. No need to loop through
        # Also avoid the sample size and use every single point (we get away with it for free)
        fval_vec = function_vals_in_cluster_0

    # put old points and new points together into a single list
    if len(new_fval_vec) > 0:
        params_in_cluster_0 = np.vstack((params_in_cluster_0, new_params_in_cluster_0))
        fval_vec = np.vstack((fval_vec, new_fval_vec))
        function_vals_in_cluster_0 = fval_vec

    # using true value, compute error from qc
    if use_gradient == False:
        evec = np.linalg.norm(fval_c-fval_vec, 2, axis=1) # TODO: what norm to use?
    elif use_gradient == True:
        # make sure we compute in euclidean, non-log scaled distance
        # use unpack_log to know which variables to do
        params_nonlog = np.copy(params_in_cluster_0)
        centroid_nonlog = np.copy(centroid_params)
        if len(unpack_log) > 0:
            for l, val in enumerate(unpack_log):
                if val == True:
                    params_nonlog[:,l] = 10 ** params_nonlog[:,l]
                    centroid_nonlog[l] = 10 ** centroid_nonlog[l]
        distance = params_nonlog - centroid_nonlog
        evec = np.linalg.norm(fval_vec - (fval_c + np.matmul(distance, grad_c)), 2, axis=1) # TODO: what norm to use?
        

    abs_error = np.max(evec)

    dim = np.max(np.linalg.norm(distance, axis=1))
    if dim == 0:
        # Dead branch of the tree. Not enough points to compute error
        left = None
    elif (rtol is not None and abs_error < np.min(np.abs(fval_vec)) * rtol) or (atol is not None and abs_error < atol):
        # Error good
        hessian = 2 / (dim**2) * np.max(evec)
        left = Cluster(k, centroid_params, fval_c, grad_c, hessian, dim)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, left = _adaptive_cluster_algorithm_helper(
                params_in_cluster_0, centroid_params, eval_function,
                atol, rtol, min_ppc, use_gradient, unpack_log,
                function_vals_in_cluster_0, k, seed)



    # Second cluster (N=1)
    params_in_cluster_1 = params[labels==1]
    if function_vals is None:
        function_vals_in_cluster_1 = None
    else:
        function_vals_in_cluster_1 = function_vals[labels==1]
    
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


    # Compute error by taking largest distances away and averaging errors
    if function_vals_in_cluster_1 is None:

        # ODE solves not defined. We need to loop through and solve
        fval_vec = np.empty(shape=(len(params_in_cluster_1),len(fval_c)))
        for i, row in enumerate(params_in_cluster_1):
            # compute true value (save in qi)
            if use_gradient == False:
                f = eval_function(row)
            elif use_gradient == True:
                f, _ = eval_function(row)
            if np.isscalar(f):
                fval_vec[i] = np.array([f])
            else:
                fval_vec[i] = f

    else:

        # ODE solves are defined. No need to loop through
        # Also avoid the sample size and use every single point (we get away with it for free)
        fval_vec = function_vals_in_cluster_1

    # put old points and new points together into a single list
    if len(new_fval_vec) > 0:
        params_in_cluster_1 = np.vstack((params_in_cluster_1, new_params_in_cluster_1))
        fval_vec = np.vstack((fval_vec, new_fval_vec))
        function_vals_in_cluster_1 = fval_vec
    

    # using true value, compute error from qc
    if use_gradient == False:
        evec = np.linalg.norm(fval_c-fval_vec, 2, axis=1) # TODO: what norm to use?
    elif use_gradient == True:
        # make sure we compute in euclidean, non-log scaled distance
        # use unpack_log to know which variables to do
        params_nonlog = np.copy(params_in_cluster_1)
        centroid_nonlog = np.copy(centroid_params)
        if len(unpack_log) > 0:
            for l, val in enumerate(unpack_log):
                if val == True:
                    params_nonlog[:,l] = 10 ** params_nonlog[:,l]
                    centroid_nonlog[l] = 10 ** centroid_nonlog[l]
        distance = params_nonlog - centroid_nonlog
        evec = np.linalg.norm(fval_vec - (fval_c + np.matmul(distance, grad_c)), 2, axis=1) # TODO: what norm to use?


    abs_error = np.max(evec)

    dim = np.max(np.linalg.norm(distance, axis=1))
    if dim == 0:
        # Dead branch of the tree. Not enough variance of points to compute error (or a single point).
        right = None
    elif (rtol is not None and abs_error < np.min(np.abs(fval_vec)) * rtol) or (atol is not None and abs_error < atol):
        hessian = 2 / (dim**2) * np.max(evec)
        right = Cluster(k, centroid_params, fval_c, grad_c, hessian, dim)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, right = _adaptive_cluster_algorithm_helper(
                params_in_cluster_1, centroid_params, eval_function,
                atol, rtol, min_ppc, use_gradient, unpack_log,
                function_vals_in_cluster_1, k, seed)

    return k, ClusterTree(prev_centroid, left, right)



def flatten_cluster_centers_with_gradient(nc, n_p, m, tree):
    centroids = np.zeros(shape=(nc, n_p))
    QoI_values = np.zeros(shape=(nc, m))
    gradients = np.zeros(shape=(nc, n_p, m))
    hessians = np.zeros(shape=nc)
    dims = np.zeros(shape=nc)

    k = 0
    for val in tree:
        if type(val) == Cluster:
            centroids[k] = val.params
            if val.QoI_value.ndim == 1:
                QoI_values[k] = np.expand_dims(val.QoI_value, axis=1)
            else:
                QoI_values[k] = val.QoI_value
            if val.gradient.ndim == 1:
                gradients[k] = np.expand_dims(val.gradient, axis=1)
            else:
                gradients[k] = val.gradient
            hessians[k] = val.hessian
            dims[k] = val.dim
            k += 1
        elif type(val) == ClusterTree:
            k = _flatten_cluster_centers_with_gradient_helper(val.left, k, centroids, QoI_values, gradients, hessians, dims)
            k = _flatten_cluster_centers_with_gradient_helper(val.right, k, centroids, QoI_values, gradients, hessians, dims)
    return centroids, QoI_values, gradients, hessians, dims



def _flatten_cluster_centers_with_gradient_helper(tree, k, centroids, QoI_values, gradients, hessians, dims):
    if type(tree) == Cluster:
        centroids[k] = tree.params
        if tree.QoI_value.ndim == 1:
            QoI_values[k] = np.expand_dims(tree.QoI_value, axis=1)
        else:
            QoI_values[k] = tree.QoI_value
        if tree.gradient.ndim == 1:
            gradients[k] = np.expand_dims(tree.gradient, axis=1)
        else:
            gradients[k] = tree.gradient
        hessians[k] = tree.hessian
        dims[k] = tree.dim
        k += 1
    elif type(tree) == ClusterTree:
        k = _flatten_cluster_centers_with_gradient_helper(tree.left, k, centroids, QoI_values, gradients, hessians, dims)
        k = _flatten_cluster_centers_with_gradient_helper(tree.right, k, centroids, QoI_values, gradients, hessians, dims)
    return k




def flatten_cluster_centers(nc, n_p, m, tree):
    centroids = np.zeros(shape=(nc, n_p))
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

