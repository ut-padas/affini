
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
    atol: float = None,
    rtol: float = None,
    N_init_clusters: int = 10, # number of initial clusters to split into
    min_ppc: int = 1, # minimum number of points used in a cluster (to measure error)
    use_gradient: bool = False,
    unpack_log = [],
    ode_solves: np.ndarray = None, # if this is not none, uses these as solves. Otherwise runs eval_function every time
    ):

    # normalize the training data
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    training_data_normalized = (training_data - mean) / std

    k = 0 # counting number of clusters as we go

    # Call clustering library
    km = KMeans(n_clusters=N_init_clusters)
    labels = km.fit_predict(training_data_normalized)
    centroids_normalized = km.cluster_centers_
    centroids = centroids_normalized * std + mean

    cluster_structure = np.empty(shape=N_init_clusters, dtype=object)

    for j in range(N_init_clusters):

        # Compute error inside each cluster
        params_in_cluster = training_data[labels==j]
        if ode_solves is None:
            ode_solves_in_cluster = None
        else:
            ode_solves_in_cluster = ode_solves[labels==j]
        
        centroid_params = centroids[j]
        if use_gradient == True:
            qc, dqdpc = eval_function(centroid_params)
        else:
            qc = eval_function(centroid_params)
            dqdpc = None
        

        # Compute error via max_{j=1,...,ss}(q_c - q_j) / q_c
        # param_sample = params_in_cluster.sample(ss_new)
        # dvec = np.empty(shape=ss_new)
        # for i, row in param_sample.reset_index(drop=True).iterrows():
        #     qi = self._solve_nelson_network(row.to_numpy(), x0, QoI, time)
        #     dvec[i] = np.abs(qc-qi)
        # error = np.max(dvec) / qc

        # Compute error by taking largest distances away and averaging errors
        if ode_solves_in_cluster is None:

            # ODE solves not defined. We need to loop through and solve
            qivec = np.empty(shape=len(params_in_cluster))
            for i, row in enumerate(params_in_cluster):
                # compute true value (save in qi)
                if use_gradient == False:
                    qivec[i] = eval_function(row)
                elif use_gradient == True:
                    qivec[i], _ = eval_function(row)

        else:

            # ODE solves are defined. No need to loop through
            # Also avoid the sample size and use every single point (we get away with it for free)
            qivec = ode_solves_in_cluster

        # using true value, compute error from qc
        if use_gradient == False:
            dvec = np.abs(qc-qivec) # no gradient (0th order)
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
            dvec = np.abs(qivec - (qc + np.sum(dqdpc * distance, axis=1)))

        abs_error = np.max(dvec)

        if (rtol is not None and abs_error < np.min(np.abs(qivec)) * rtol) or (atol is not None and abs_error < atol):
            # Tolerance is good; save values in cluster column
            dim = np.max(np.linalg.norm(distance, axis=1))
            hessian = 2 / (dim**2) * np.max(dvec)
            cluster_structure[j] = Cluster(k, centroid_params, qc, dqdpc, hessian, dim)
            k += 1
        else:
            # Split cluster in half (recursive step)
            k, cluster_structure[j] = _adaptive_cluster_algorithm_helper(
                    params_in_cluster, centroid_params, eval_function,
                    atol, rtol, min_ppc, use_gradient, unpack_log,
                    ode_solves_in_cluster, k)

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
    ode_solves: np.ndarray = None,
    k: int = 0, # current cluster index (for recursive purposes)
):

    # normalize for clustering
    mean = params.mean(axis=0)
    std = params.std(axis=0)
    # quick fix to the issue if std is 0 in one column
    if (std == 0).any():
        std += (std == 0) * 1
    params_normalized = (params - mean) / std

    N = 2  # split in half
    km = KMeans(n_clusters=N)
    labels = km.fit_predict(params_normalized)
    centroids_normalized = km.cluster_centers_
    centroids = centroids_normalized * std + mean

    # First cluster (N=0)
    params_in_cluster_0 = params[labels==0]
    if ode_solves is None:
        ode_solves_in_cluster_0 = None
    else:
        ode_solves_in_cluster_0 = ode_solves[labels==0]

    centroid_params = centroids[0]
    if use_gradient == True:
        qc, dqdpc = eval_function(centroid_params)
    else:
        qc = eval_function(centroid_params)
        dqdpc = None

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
        new_qivec = np.zeros(shape=len(new_params_in_cluster_0))
        for i, row in enumerate(new_params_in_cluster_0):
            # compute true value (save in qi)
            if use_gradient == False:
                new_qivec[i] = eval_function(row)
            elif use_gradient == True:
                new_qivec[i], _ = eval_function(row)

    else:
        new_qivec = []


    # Compute error by taking largest distances away and averaging errors
    if ode_solves_in_cluster_0 is None:

        # ODE solves not defined. We need to loop through and solve
        qivec = np.empty(shape=len(params_in_cluster_0))
        for i, row in enumerate(params_in_cluster_0):
            # compute true value (save in qi)
            if use_gradient == False:
                qivec[i] = eval_function(row)
            elif use_gradient == True:
                qivec[i], _ = eval_function(row)

    else:

        # ODE solves are defined. No need to loop through
        # Also avoid the sample size and use every single point (we get away with it for free)
        qivec = ode_solves_in_cluster_0

    # put old points and new points together into a single list
    if len(new_qivec) > 0:
        params_in_cluster_0 = np.row_stack((params_in_cluster_0, new_params_in_cluster_0))
        qivec = np.append(qivec, new_qivec)
        ode_solves_in_cluster_0 = qivec

    # using true value, compute error from qc
    if use_gradient == False:
        dvec = np.abs(qc-qivec) # no gradient (0th order)
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
        dvec = np.abs(qivec - (qc + np.sum(dqdpc * distance, axis=1)))

    abs_error = np.max(dvec)

    dim = np.max(np.linalg.norm(distance, axis=1))
    if dim == 0:
        # Dead branch of the tree. Not enough points to compute error
        left = None
    elif (rtol is not None and abs_error < np.min(np.abs(qivec)) * rtol) or (atol is not None and abs_error < atol):
        # Error good
        hessian = 2 / (dim**2) * np.max(dvec)
        left = Cluster(k, centroid_params, qc, dqdpc, hessian, dim)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, left = _adaptive_cluster_algorithm_helper(
                params_in_cluster_0, centroid_params, eval_function,
                atol, rtol, min_ppc, use_gradient, unpack_log,
                ode_solves_in_cluster_0, k)



    # Second cluster (N=1)
    params_in_cluster_1 = params[labels==1]
    if ode_solves is None:
        ode_solves_in_cluster_1 = None
    else:
        ode_solves_in_cluster_1 = ode_solves[labels==1]
    
    centroid_params = centroids[1]
    if use_gradient == False:
        qc = eval_function(centroid_params)
        dqdpc = None
    elif use_gradient == True:
        qc, dqdpc = eval_function(centroid_params)


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
        new_qivec = np.empty(shape=len(new_params_in_cluster_1))
        for i, row in enumerate(new_params_in_cluster_1):
            # compute true value (save in qi)
            if use_gradient == False:
                new_qivec[i] = eval_function(row)
            elif use_gradient == True:
                new_qivec[i], _ = eval_function(row)
        
    else:
        new_qivec = []


    # Compute error by taking largest distances away and averaging errors
    if ode_solves_in_cluster_1 is None:

        # ODE solves not defined. We need to loop through and solve
        qivec = np.empty(shape=len(params_in_cluster_1))
        for i, row in enumerate(params_in_cluster_1):
            # compute true value (save in qi)
            if use_gradient == False:
                qivec[i] = eval_function(row)
            elif use_gradient == True:
                qivec[i], _ = eval_function(row)

    else:

        # ODE solves are defined. No need to loop through
        # Also avoid the sample size and use every single point (we get away with it for free)
        qivec = ode_solves_in_cluster_1

    # put old points and new points together into a single list
    if len(new_qivec) > 0:
        params_in_cluster_1 = np.row_stack((params_in_cluster_1, new_params_in_cluster_1))
        qivec = np.append(qivec, new_qivec)
        ode_solves_in_cluster_1 = qivec
    

    # using true value, compute error from qc
    if use_gradient == False:
        dvec = np.abs(qc-qivec) # no gradient (0th order)
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
        dvec = np.abs(qivec - (qc + np.sum(dqdpc * distance, axis=1)))

    abs_error = np.max(dvec)

    dim = np.max(np.linalg.norm(distance, axis=1))
    if dim == 0:
        # Dead branch of the tree. Not enough variance of points to compute error (or a single point).
        right = None
    elif (rtol is not None and abs_error < np.min(np.abs(qivec)) * rtol) or (atol is not None and abs_error < atol):
        hessian = 2 / (dim**2) * np.max(dvec)
        right = Cluster(k, centroid_params, qc, dqdpc, hessian, dim)
        k += 1
    else:
        # Error too large; split cluster in half (recursive step)
        k, right = _adaptive_cluster_algorithm_helper(
                params_in_cluster_1, centroid_params, eval_function,
                atol, rtol, min_ppc, use_gradient, unpack_log,
                ode_solves_in_cluster_1, k)

    return k, ClusterTree(prev_centroid, left, right)



def flatten_cluster_centers_with_gradient(nc, n_p, tree):
    centroids = np.zeros(shape=(nc, n_p))
    QoI_values = np.zeros(shape=nc)
    gradients = np.zeros(shape=(nc, n_p))
    hessians = np.zeros(shape=nc)
    dims = np.zeros(shape=nc)

    k = 0
    for val in tree:
        if type(val) == Cluster:
            centroids[k] = val.params
            QoI_values[k] = val.QoI_value
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
        QoI_values[k] = tree.QoI_value
        gradients[k] = tree.gradient
        hessians[k] = tree.hessian
        dims[k] = tree.dim
        k += 1
    elif type(tree) == ClusterTree:
        k = _flatten_cluster_centers_with_gradient_helper(tree.left, k, centroids, QoI_values, gradients, hessians, dims)
        k = _flatten_cluster_centers_with_gradient_helper(tree.right, k, centroids, QoI_values, gradients, hessians, dims)
    return k




def flatten_cluster_centers(nc, n_p, tree):
    centroids = np.zeros(shape=(nc, n_p))
    QoI_values = np.zeros(shape=nc)
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

