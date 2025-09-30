from collections.abc import Callable
from types import NoneType
import numpy as np
from adaptive_clustering import adaptive_cluster_algorithm, flatten_cluster_centers_with_gradient
from nn_inference import NN_Inference
from scipy.stats import qmc
import time


class NearODESurrogate:

    def __init__(self):

        self._model_trained = False

    
    ###
    # Call this function to train model from scratch.
    # After this function has been called, we can call the predict function with a set of testing data.
    # Inputs:
    # - features: the input data {x} that we use as cluster centroids (we can refine these via adaptive clustering)
    # - labels: the output data {f(x)}
    # - gradients: the gradient of the output data {Df(x)} (which has the same shape as features)
    # - do_clustering: if True, we run adaptive clustering. If False, we use every point in features in the NN searches.
    # - oracle: the exact function. We need to evaluate arbitrary points in the adaptive clustering, so this is required if above is True
    # - N_init_clusters: the number of initial clusters
    # - ss: the number of points to sample from each cluster (might change this when if we change how we measure error)
    # - rel_err_tol: the error tolerance that decides if we split a cluster in half or not
    def train_model_from_data(self,
        
        # necessary data
        features: np.ndarray,
        labels: np.ndarray,
        gradients: np.ndarray,

        # clustering parameters
        do_clustering = True,
        oracle = None,
        N_init_clusters = 10, # initial number of clusters
        ss = 10,
        atol = 0.01,
        rtol = None,

        # other
        # i.e. should we add a transform into log space? or leave that up to user?
        check_guarantee_N_times: int = 0,
        L_global = None, # number of times we check all starting points if they are within
            # computational guarantee and recluster the ones that are not
        alpha = 1,
        disp_time = True,

    ):
        
        start_time = time.perf_counter()
        
        ### CHECK INPUTS
        # features and gradients should have the same shape
        # labels should have same length
        if features.ndim != 2:
            # throw error for wrong number of dimensions
            return -1
        if gradients.ndim != 2:
            # throw error for wrong number of dimensions
            return -1
        if features.shape != gradients.shape:
            # throw error for shapes not matching
            return -1
        mf, nf = features.shape
        if labels.ndim != 1:
            # throw error for wrong number of dimensions
            return -1
        if mf != len(labels):
            # throw error for number of outputs not matching number of inputs
            return -1
        
        self.atol = atol
        self.rtol = rtol
        self.oracle = oracle
        self.nf = nf
        self.mf = mf


        # Cluster if user selects the option
        # If do_clustering is True, the user must provide an "oracle" function that returns
        # both the function value and its gradient as a tuple.
        if do_clustering == True:

            # make sure oracle is defined
            if not callable(oracle):
                # throw error for oracle not being a function
                return -1

            # call clustering algorithm
            k, cluster_tree = adaptive_cluster_algorithm(features, oracle, atol, rtol, \
                    N_init_clusters, ss, True, [], labels, gradients)
            new_features, new_labels, new_gradients, new_hessians, new_dims = \
                    flatten_cluster_centers_with_gradient(k, nf, cluster_tree)
        

        else:
            new_features = features
            new_labels = labels
            new_gradients = gradients
            new_hessians = None
            new_dims = None
            k = len(features)


        # Put these into FAISS
        self.KNN_search = NN_Inference(new_features, new_labels, new_gradients, new_hessians, new_dims)
        self._model_trained = True
        self.N_sources = k


        # Computational guarantee: check distance to all training points
        if do_clustering == True and type(L_global) != NoneType and check_guarantee_N_times > 0:
            for j in range(check_guarantee_N_times):
                bad_features = self._test_guarantee(features, atol, L_global, alpha)
                self.add_sources(features[bad_features])

        end_time = time.perf_counter()
        if disp_time == True:
            print(f'Time to train: {end_time-start_time:.2f} seconds')

        return 0


    ###
    # Train model from just a function
    # Sample training points using _______ sampling algorithm (Latin Hypercube Sampling?)
    # Solve function at these points to create data and then call above function
    # Inputs:
    # - oracle: the true function
    # - domain: a p*2 matrix with the computational domain, where p is the number of parameters
    # - M: the number of samples from the distribution
    # - do_clustering: if True, we run adaptive clustering. If False, we use every point in features in the NN searches.
    # - N: the number of initial clusters
    # - ss: the number of points to sample from each cluster (might change this when if we change how we measure error)
    # - rel_err_tol: the error tolerance that decides if we split a cluster in half or not
    def train_model_from_sampling(self,

        oracle: Callable,

        # sampling parameters (TBD)
        # need to define domain, # of parameters, sampling distribution, etc.
        domain: np.ndarray,
        N: int, # number of samples
        distribution = 'uniform', # type of distribution to sample from (either uniform or LHS for now)

        # clustering parameters
        do_clustering = True,
        N_init_clusters = 10, # initial number of clusters
        ss = 10, # sample size for each cluster
        atol = 0.01, # absolute error tolerance
        rtol = None, # relative error tolerance
        fmax = None, # function max to use in relative error computation

        # other
        # i.e. should we add a transform into log space? or leave that up to user?
        check_guarantee_N_times: int = 0,
        L = None, # number of times we check all starting points if they are within
            # computational guarantee and recluster the ones that are not
        alpha = 1,
        disp_time = True, # display time

    ):
        
        start_time = time.perf_counter()
        
        # do sampling
        if domain.ndim > 2:
            # throw error since domain is not a matrix
            return -1
        if domain.ndim == 1:
            p = 1
            q = len(domain)
            l = domain[0]
            r = domain[1]
        else:
            p, q = domain.shape
            l = domain[:,0]
            r = domain[:,1]
        if q != 2:
            # throw error because this dimension should be a range
            return -1
        
        # draw parameters
        distributions = ['uniform', 'LHS']
        if distribution not in distributions:
            # throw error because distribution is not valid
            return -1
        
        if distribution == 'uniform':
            features = np.random.uniform(l, r, (N, p))
        elif distribution == 'LHS':
            sampler = qmc.LatinHypercube(d=p)
            features = sampler.random(n=N)
            features = qmc.scale(features, l, r)

        # do solves (maybe this isn't necessary but it's certainly easier)
        labels = np.empty(len(features))
        gradients = np.empty_like(features)
        for jj, feature in enumerate(features):
            labels[jj], gradients[jj] = oracle(feature)
        
        # plug this into above function
        self.train_model_from_data(features, labels, gradients, do_clustering,
                oracle, N_init_clusters, ss, atol, rtol, check_guarantee_N_times, L, alpha, False)

        end_time = time.perf_counter()
        if disp_time == True:
            print(f'Time to train: {end_time-start_time:.2f} seconds')

    
    ###
    # The function that evaluates some arbitrary list of points.
    # If targets is a 1D array, treat as N*1 matrix.
    # Otherwise, targets should be an N*p matrix, where p is the number of parameters.
    def evaluate(self,
        targets: np.ndarray,
        k=1,
        disp_time=False,
        L=None, # if defined, will give warning if targets are outside of the guaranteed tolerance
        recluster=False, # if True, will take points outside of computational guarantee
            # and put them in cluster algorithm. Then add these to faiss index
        alpha=1, # fudge factor for the Hessian approximations
    ):

        if self._model_trained == True:
            if targets.ndim == 1:
                predictions, unguaranteed_points = self.KNN_search.predict(targets.reshape(len(targets), 1), k, disp_time=disp_time, L_global=L, tol=self.err_tol, alpha=alpha)
            else:
                predictions, unguaranteed_points = self.KNN_search.predict(targets, k, disp_time=disp_time, L_global=L, tol=self.atol, alpha=alpha)

            if np.sum(unguaranteed_points) > 3 * self.N_sources:
                print('WARNING: too many points (3 * N_clusters) are outside of guarantee. Need more clusters.')

            if recluster == True:
                self.add_sources(targets[unguaranteed_points])
            
            return predictions, unguaranteed_points
                
                

    ###
    # This function returns the source points used for nearest neighbors searches
    def get_sources(self):
        return self.KNN_search.features


    def add_sources(self, features):
        k, cluster_tree = adaptive_cluster_algorithm(features,
            self.oracle, self.atol, self.rtol, 2, 10, True, [])
        new_features, new_labels, new_gradients, new_hessians, new_dims = \
                    flatten_cluster_centers_with_gradient(k, self.nf, cluster_tree)
        self.KNN_search.add_extra_features(new_features, new_labels, new_gradients, new_hessians, new_dims)
        self.N_sources += len(new_features)


    def _test_guarantee(self, targets, tol, L_global, alpha):
        D, I = self.KNN_search._faiss_index.search(targets, 1)
        local_tols = np.minimum(self.KNN_search.dims[I[:,0]] ** 2, 2 * tol / (alpha * self.KNN_search.hessians[I[:,0]]))
        global_tol = 2 * tol / L_global
        good_points = np.logical_or((D[:,0] < local_tols), np.logical_and(local_tols < D[:,0], D[:,0] < global_tol))
        return np.logical_not(good_points)
