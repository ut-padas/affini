import time
from types import NoneType
import numpy as np
import faiss

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
        dims: np.ndarray = None
        ):
        
        # --- INPUT ERROR CHECKING ---

        # check that length of vectors is correct
        self._mf, self._nf = features.shape
        self._ml           = len(labels)

        if type(gradients) != NoneType:
            self._mg, self._ng = gradients.shape
            if self._nf != self._ng:
                raise ValueError('Number of features does not match length of each gradient vector')
            if self._mf != self._mg:
                raise ValueError('Features and gradients has a mismatching size of points')
        if self._mf != self._ml:
            raise ValueError('Features and labels has a mismatching size of points')
        

        # save data
        self.features = features
        self.labels = labels
        self.gradients = gradients
        self.hessians = hessians
        self.dims = dims

        # build FAISS index
        self._faiss_index = faiss.IndexFlatL2(self._nf)
        self._faiss_index.add(features)


    # given a matrix of features, uses FAISS to return the predicted value.
    # uses the average of the k-nearest neighbors (default only uses 1 nearest neighbor)
    def predict(self,
        targets: np.ndarray,
        k: int,
        unwrap_log=[],
        disp_time=False,
        L_global = None, # if both L and tol are defined, use this to check if distance to nearest
        tol = None, # neighbors are within computational guarantee.
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

        # check for points that are outside the theoretical guarantee
        if (type(tol) != NoneType and type(L_global) != NoneType and k==1):
            # FAISS distances are squared, so do not consider square root on rhs
            local_tols = np.minimum(self.dims[I[:,0]] ** 2, 2 * tol / (alpha * self.hessians[I[:,0]]))
            global_tol = 2 * tol / L_global
            good_points = np.logical_or((D[:,0] < local_tols), np.logical_and(local_tols < D[:,0], D[:,0] < global_tol))
        else:
            good_points = np.full(len(D), True)
        
        predictions = np.zeros(len(targets))
        for j in range(k):
            if type(self.gradients) == NoneType:
                # gradient not specified; use only nearest neighbor
                predictions += self.labels[I[:,j]]
            else:
                distance_vecs = targets_unlog - features_unlog[I[:,j]]
                predictions += self.labels[I[:,j]] + np.sum(self.gradients[I[:,j]] * distance_vecs, axis=1)

        if disp_time == True:
            end_time = time.perf_counter()
            print(f'Time elapsed for predictions: {end_time-start_time} seconds')

        # return average
        return predictions / k, np.logical_not(good_points)
        # bad points is all False when tol is not specified


    def add_extra_features(self, new_features, new_labels, new_gradients, new_hessians, new_dims):
        self.features = np.row_stack((self.features, new_features))
        self.labels = np.append(self.labels, new_labels)
        self.gradients = np.row_stack((self.gradients, new_gradients))
        self.hessians = np.append(self.hessians, new_hessians)
        self.dims = np.append(self.dims, new_dims)
        self._faiss_index.add(new_features)
    
