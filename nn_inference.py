"""
Nearest-neighbor inference with optional first-order (Taylor) correction.
"""

import time

import faiss
import numpy as np

# TODO eventually make labels multi-d. This would make labels a matrix and gradients a rank-3 tensor.


class NN_Inference:
    """FAISS-backed kNN inference with optional Taylor gradient boosting.

    This class stores feature vectors (sources), their labels (targets), and
    optionally Jacobians (per-source gradients) along with per-source Hessian
    proxies and trust radii. It builds a FAISS IndexFlatL2 to enable fast
    nearest-neighbor queries, and exposes `predict` to compute surrogate
    outputs.

    Prediction modes:
      * No gradients provided  -> neighbor label averaging
      * Gradients provided     -> local linear model:
           f(x) ≈ f(x_i) + J_i (x - x_i)  (optionally averaged across k NNs)

    Guarantee logic (k == 1):
      * Uses per-source trust radius and scalar Hessian proxy to decide whether
        a query lies inside a local Taylor trust region (and optional global
        Lipschitz fallback); returns a boolean mask of "outside guarantee".

    Attributes:
      features (np.ndarray): Source feature matrix (P, n).
      labels (np.ndarray): Source labels (P, m).
      gradients (np.ndarray | None): Per-source Jacobians (P, m, n) or None.
      hessians (np.ndarray | None): Per-source scalar Hessian proxies (P,) or None.
      trust_radii (np.ndarray | None): Per-source trust radii (P,) or None.
      P (int): Number of source points.
      M (int): Output dimension.
      N (int): Input dimension.
      _faiss_index (faiss.Index): Flat L2 index over features (dimension N).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        gradients: np.ndarray | None = None,
        hessians: np.ndarray | None = None,
        trust_radii: np.ndarray | None = None,
    ):
        """Initialize the inference object and build the FAISS index.

        Args:
          features: Array of input points used as cluster centroids with shape (P, n).
          labels: Labels at sources, shape (P, m).
          gradients: Optional Jacobians at sources, shape (P, m, n).
          hessians: Optional scalar Hessian proxies per source, shape (P,).
          trust_radii: Optional trust radii per source (squared-distance space
            comparisons are used), shape (P,).

        Notes:
          * This constructor assumes `gradients` is provided and shaped as
            (P, m, n) when used; minimal checks are left as-is to preserve
            original logic.
          * FAISS distances are squared L2 norms; see comments in `predict`.
        """
        pf, n = features.shape
        pl, m = labels.shape
        if pf != pl:
            raise ValueError("Number of features and labels must match (P).")

        if gradients is not None:
            pg, mg, ng = gradients.shape
            if pf != pg or m != mg or n != ng:
                raise ValueError(
                    "Gradient shape must be (P, m, n) and match features/labels."
                )

        # Save data as provided (no copies unless necessary downstream)
        self.features = features
        self.labels = labels
        self.gradients = gradients
        self.hessians = hessians
        self.trust_radii = trust_radii
        self.P = pf  # number of training points
        self.M = m  # output dimension (m)
        self.N = n  # input dimension (n)

        # Build a flat L2 FAISS index over feature vectors.
        # Note: IndexFlatL2 expects float32 arrays; the original code relies on
        # faiss accepting the dtype in use. We preserve that behavior here.
        self._faiss_index = faiss.IndexFlatL2(self.N)
        self._faiss_index.add(features)  # type: ignore

    def predict(
        self,
        targets: np.ndarray,
        k: int,
        unwrap_log: list = [],
        disp_time: bool = False,
        L_global: float | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        alpha: float = 1.0,
    ):
        """Predict surrogate outputs for a batch of target points.

        For each target row x*, obtain indices/distances of k nearest neighbors
        via FAISS. If `gradients` are present, compute local linear predictions
        using the Jacobian at each neighbor with an offset vector (x* - x_i),
        then average across k neighbors. Otherwise, average neighbor labels.

        Args:
          targets: Query matrix of shape (Q, N).
          k: Number of nearest neighbors to use.
          unwrap_log: TODO
          disp_time: If True, print elapsed prediction time.
          L_global: Optional global Lipschitz constant for guarantee fallback.
          atol: Absolute tolerance for guarantee checks (k == 1 path).
          rtol: Relative tolerance for guarantee checks (k == 1 path).
          alpha: Extra multiplicative factor (fudge) for Hessian-based bounds.

        Returns:
          tuple[np.ndarray, np.ndarray]:
            - predictions: Array of shape (Q, M) with averaged surrogate outputs.
            - outside_mask: Boolean array (Q,) where True marks points deemed
              *outside* the theoretical guarantee.
                * If no tolerance is provided or k != 1, this mask is all False
                  (i.e., no guarantee screening).

        Notes:
          * FAISS distances D are **squared** L2 norms; all comparisons to
            trust radii or tolerance-derived bounds are in squared-distance
            space. Do not apply square roots on the right-hand side.
          * Local linear mode uses JVP via np.einsum('ijk,ik->ij', J, dx),
            with shapes:
              J:  (Q, M, N)  (neighbor Jacobians gathered for each query)
              dx: (Q, N)     (offset vectors x* - x_i)
              out: (Q, M)
        """
        start_time = time.perf_counter()

        # TODO check if the length of each feature vector matches the length of training vectors

        # k-NN search via FAISS; D are squared L2 distances, I are indices.
        D, I = self._faiss_index.search(targets, k)  # type: ignore

        # Optional “unlog” of selected dimensions prior to computing offsets.
        # (Distances from FAISS remain unchanged; only offsets for Taylor are
        # affected. Preserve original copy semantics.)
        features_unlog = self.features.copy()
        targets_unlog = targets.copy()
        if len(unwrap_log) > 0:
            for j, val in enumerate(unwrap_log):
                if val:
                    targets_unlog[:, j] = 10 ** targets_unlog[:, j]
                    features_unlog[:, j] = 10 ** features_unlog[:, j]

        # Allocate predictions; accumulate over k neighbors, average at return.
        predictions = np.zeros((len(targets), self.M))
        for j in range(k):
            if self.gradients is None:
                # No gradients: neighbor label averaging (accumulate).
                predictions += self.labels[I[:, j]]
            else:
                # Local linear correction:
                # distance_vecs = (x* - x_i) in the “unlogged” space (if used).
                distance_vecs = targets_unlog - features_unlog[I[:, j]]

                # Batched JVP: for each query q, (M×N) @ (N,) -> (M,)
                predictions += self.labels[I[:, j]] + np.einsum(
                    "ijk,ik->ij", self.gradients[I[:, j]], distance_vecs
                )

        # ---------------- Guarantee / trust-region checks (k == 1) ----------------
        # All bounds below compare squared distances (D[:, 0]) against squared
        # thresholds (e.g., trust_radii**2) or tolerance-derived values.
        if self.trust_radii is None or self.hessians is None:
            raise RuntimeError(
                "Hessian proxies and trust radii are required for guarantee checks."
            )
        else:
            if atol is not None and rtol is not None and k == 1:
                # Combine absolute and relative tolerances; pick the looser (max).
                local_atols = np.minimum(
                    self.trust_radii[I[:, 0]] ** 2,
                    2 * atol / (alpha * self.hessians[I[:, 0]]),
                )
                local_rtols = np.minimum(
                    self.trust_radii[I[:, 0]] ** 2,
                    2
                    * rtol
                    * np.linalg.norm(predictions, 2, axis=1)
                    / (alpha * self.hessians[I[:, 0]]),
                )
                good_points = D[:, 0] < np.maximum(local_atols, local_rtols)
            elif atol is not None and L_global is not None and k == 1:
                # Absolute tolerance with global Lipschitz fallback.
                local_tols = np.minimum(
                    self.trust_radii[I[:, 0]] ** 2,
                    2 * atol / (alpha * self.hessians[I[:, 0]]),
                )
                global_tol = 2 * atol / L_global
                good_points = np.logical_or(
                    (D[:, 0] < local_tols),
                    np.logical_and(local_tols < D[:, 0], D[:, 0] < global_tol),
                )
            elif atol is not None and k == 1:
                # Absolute tolerance only (no rtol, no global Lipschitz).
                local_tols = np.minimum(
                    self.trust_radii[I[:, 0]] ** 2,
                    2 * atol / (alpha * self.hessians[I[:, 0]]),
                )
                good_points = D[:, 0] < local_tols
            else:
                # No guarantee screening (k != 1 or no tolerance): mark all as good.
                good_points = np.full(len(D), True)

        if disp_time:
            end_time = time.perf_counter()
            print(f"Time elapsed for predictions: {end_time - start_time} seconds")

        # Average over neighbors and return outside-of-guarantee mask.
        # Bad points are all False when tol is not specified.
        return predictions / k, np.logical_not(good_points)

    def add_extra_features(
        self, new_features, new_labels, new_gradients, new_hessians, new_dims
    ):
        """Append newly created sources to the index and internal arrays.

        Args:
          new_features: New source features, shape (R, N).
          new_labels: New labels, shape (R, M) or compatible with append.
          new_gradients: New Jacobians, shape (R, M, N).
          new_hessians: New Hessian proxies, shape (R,) or compatible.
          new_dims: New trust radii (or related dims), shape (R,) or compatible.

        Notes:
          * This method preserves original array update semantics (append/vstack).
          * The FAISS index is incrementally updated by calling `.add` on the
            new features.
        """
        self.features = np.vstack((self.features, new_features))
        self.labels = np.append(self.labels, new_labels)
        if self.gradients is not None:
            self.gradients = np.vstack((self.gradients, new_gradients))
        if self.hessians is not None:
            self.hessians = np.append(self.hessians, new_hessians)
        if self.trust_radii is not None:
            self.trust_radii = np.append(self.trust_radii, new_dims)
        self._faiss_index.add(new_features)  # type: ignore
