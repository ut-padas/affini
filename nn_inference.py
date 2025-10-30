import time
import numpy as np
import faiss
import matplotlib.pyplot as plt

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
        hessians: np.ndarray = None,
        trust_radii: np.ndarray = None
        ):
        
        # --- INPUT ERROR CHECKING ---

        # check that length of vectors is correct
        p1, n1 = features.shape
        p2, m1 = labels.shape
        p3, m2, n2 = gradients.shape
    
        # TODO do some checking

        # save data
        self.features = features
        self.labels = labels
        self.gradients = gradients
        self.hessians = hessians
        self.trust_radii = trust_radii
        self.P = p3
        self.M = m2
        self.N = n2

        # build FAISS index
        self._faiss_index = faiss.IndexFlatL2(self.N)
        self._faiss_index.add(features)


    # given a matrix of features, uses FAISS to return the predicted value.
    # uses the average of the k-nearest neighbors (default only uses 1 nearest neighbor)
    def predict(self,
        targets: np.ndarray,
        k: int,
        unwrap_log=[],
        disp_time=False,
        L_global = None, # if both L and tol are defined, use this to check if distance to nearest
        atol = None, # neighbors are within computational guarantee.
        rtol = None,
        alpha = 1, # extra fudge factor for the hessian approximations
    ):

        start_time = time.perf_counter()
        # TODO check if the length of each feature vector matches the length of training vectors

        # use FAISS to do k-nearest neighbors search
        D, I = self._faiss_index.search(targets, k)
        
        # if necessary we deal with distances by unwrapping log
        features_unlog = self.features.copy()
        targets_unlog = targets.copy()
        if len(unwrap_log) > 0:
            for j, val in enumerate(unwrap_log):
                if val == True:
                    targets_unlog[:,j] = 10 ** targets_unlog[:,j]
                    features_unlog[:,j] = 10 ** features_unlog[:,j]
        
        predictions = np.zeros((len(targets),self.M)) # this should be a matrix
        for j in range(k):
            if self.gradients is None:
                # gradient not specified; use only nearest neighbor
                predictions += self.labels[I[:,j]]
            else:
                distance_vecs = targets_unlog - features_unlog[I[:,j]]
                predictions += self.labels[I[:,j]] + np.einsum('ijk,ik->ij', self.gradients[I[:,j]], distance_vecs)

        # check for points that are outside the theoretical guarantee
        # FAISS distances are squared, so do not consider square root on rhs
        if (atol is not None and rtol is not None and k==1):
            local_atols = np.minimum(self.trust_radii[I[:,0]] ** 2, 2 * atol / (alpha * self.hessians[I[:,0]]))
            local_rtols = np.minimum(self.trust_radii[I[:,0]] ** 2, 2 * rtol * np.linalg.norm(predictions, 2, axis=1) / (alpha * self.hessians[I[:,0]]))
            good_points = (D[:,0] < np.maximum(local_atols, local_rtols))
        elif (atol is not None and L_global is not None and k==1):
            local_tols = np.minimum(self.trust_radii[I[:,0]] ** 2, 2 * atol / (alpha * self.hessians[I[:,0]]))
            global_tol = 2 * atol / L_global
            good_points = np.logical_or((D[:,0] < local_tols), np.logical_and(local_tols < D[:,0], D[:,0] < global_tol))
        elif (atol is not None and k==1):
            local_tols = np.minimum(self.trust_radii[I[:,0]] ** 2, 2 * atol / (alpha * self.hessians[I[:,0]]))
            good_points = (D[:,0] < local_tols)
        else:
            good_points = np.full(len(D), True)


        if disp_time == True:
            end_time = time.perf_counter()
            print(f'Time elapsed for predictions: {end_time-start_time} seconds')

        # return average
        return predictions / k, np.logical_not(good_points)
        # bad points is all False when tol is not specified


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
    
