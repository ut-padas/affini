import time
import numpy as np
import faiss
import matplotlib.pyplot as plt
import pandas as pd

# TODO eventually make labels multi-d. This would make labels a matrix and gradients a rank-3 tensor.


# Contains machinery that takes in features
# Does nearest neighbors searches via FAISS
# Includes the option to do gradient boosting via a Taylor series expansion
class NN_Inference:

    # CONSTRUCTOR
    def __init__(self,
        features: np.ndarray,
        labels: np.ndarray,
        gradients: np.ndarray = None,
        trust_ellipses: np.ndarray = None,
        n_pts_per_cluster: np.ndarray = None,
        tree_levels: np.ndarray = None
        ):
        
        # --- INPUT ERROR CHECKING ---

        # check that length of vectors is correct
        p1, n1 = features.shape
        p2, m1 = labels.shape
        if gradients is not None:
            p3, m2, n2 = gradients.shape
    
        # TODO do some checking

        # save data
        self.features = features
        self.labels = labels
        self.gradients = gradients
        self.trust_ellipses = trust_ellipses
        self.n_pts_per_cluster = n_pts_per_cluster
        self.tree_levels = tree_levels
        self.P = p1
        self.M = m1
        self.N = n1

        # build FAISS index
        self._faiss_index = faiss.IndexFlatL2(self.N)
        self._faiss_index.add(features)


    # given a matrix of features, uses FAISS to return the predicted value.
    # uses the average of the k-nearest neighbors (default only uses 1 nearest neighbor)
    def predict(self,
        targets: np.ndarray,
        atol,
        rtol,
        fmax,
        disp_time=False,
    ):

        N = len(targets)
        K = 10

        start_time = time.perf_counter()
        # TODO check if the length of each feature vector matches the length of training vectors

        # use FAISS to do k-nearest neighbors search
        # Find additional nearest neighbors, only use ones which are in trust region.
        _, Ind = self._faiss_index.search(targets, K)

        near_ind = np.full(shape=N, fill_value=-1)
        for j in reversed(range(K)):
            near_neighbors = self.features[Ind[:,j]]
            diffs = targets - near_neighbors
            near_trust_ellipses = self.trust_ellipses[Ind[:,j]]
            trust_distances = np.einsum('ni,nij,nj->n', diffs, near_trust_ellipses, diffs)
            in_trust_region = trust_distances <= 2 * np.maximum(atol, rtol * fmax)
            near_ind[in_trust_region] = Ind[in_trust_region,j]

        # what if a point has no nearest neighbors in trust region? Just use nearest neighbor
        bad_points = near_ind == -1
        _, Ind = self._faiss_index.search(targets[bad_points], 1)
        near_ind[bad_points] = Ind[:,0]

        # All points have an assigned source now. Do computations
        near_neighbors = self.features[near_ind]
        near_fvals = self.labels[near_ind]
        near_gradients = self.gradients[near_ind]
        near_trust_ellipses = self.trust_ellipses[near_ind]

        diffs = targets - near_neighbors
        if self.gradients is None:
            # gradient not specified; use only nearest neighbor
            predictions = near_fvals
        else:
            predictions = near_fvals + np.einsum('ijk,ik->ij', near_gradients, diffs)

        if disp_time:
            end_time = time.perf_counter()
            print(f'Time elapsed for predictions: {end_time-start_time} seconds')

        return predictions, bad_points


    def find_nearest_sources(self, targets):
        _, Ind = self._faiss_index.search(targets, 1)
        return Ind[:,0]


    def add_extra_features(self, new_features, new_labels, new_gradients, new_hessians, new_dims):
        self.features = np.vstack((self.features, new_features))
        self.labels = np.append(self.labels, new_labels)
        self.gradients = np.vstack((self.gradients, new_gradients))
        self.hessians = np.append(self.hessians, new_hessians)
        self.trust_radii = np.append(self.trust_radii, new_dims)
        self._faiss_index.add(new_features)


    def show_source_information(self):
        # each should be length N_sources
        grad_norms = np.linalg.norm(self.gradients, axis=(1,2))
        hessian_norms = np.abs(self.hessians)

        # gradient norms plot
        fig1, ax1 = plt.subplots()
        ax1.hist(grad_norms)
        ax1.set_title(f'Histogram of Gradient Norms ($N_{{sources}} = {len(grad_norms)}$)')
        
        # hessian norms plot
        fig2, ax2 = plt.subplots()
        ax2.hist(hessian_norms)
        ax2.set_title(f'Histogram of Hessian Norms ($N_{{sources}} = {len(hessian_norms)}$)')

        plt.show()



    def save_all_data(self):
        # Returns cluster statistics (every data point except x, y, dy/dx)
        all_data = np.empty(shape=(len(self.features), 5))
        all_data[:,0] = self.hessians
        all_data[:,1] = self.hessian_vars
        all_data[:,2] = self.trust_radii
        all_data[:,3] = self.n_pts_per_cluster
        all_data[:,4] = self.tree_levels
        col_headers = ['Lipschitz Constants',
                  'Lipschitz Constant Variance',
                  'Trust Radius',
                  '# Pts per Cluster',
                  'Tree Level']
        return pd.DataFrame(all_data, columns=col_headers)

