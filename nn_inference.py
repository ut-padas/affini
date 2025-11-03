"""
Nearest-neighbor inference with optional first-order (Taylor) correction.
"""

import time

import faiss
import numpy as np


class NN_Inference:
    """
    FAISS-backed kNN inference with optional Taylor gradient boosting.

    Shape Conventions:
        - X.shape: (n_pts, d) where n_pts=number of source points, d=input dimension
        - f_vals.shape: (n_pts, fd) where fd=function output dimension
        - gradients.shape: (n_pts, fd, d) where gradients are Jacobian matrices
        - hessians.shape: (n_pts,) scalar Hessian proxies per source
        - trust_radii.shape: (n_pts,) trust radii per source

    Note: If fd=1 (scalar function), f_vals.shape=(n_pts, 1) and gradients.shape=(n_pts, 1, d)

    This class stores X (source points), their f_vals (function values), and
    optionally Jacobians (per-source gradients) along with per-source Hessian
    proxies and trust radii. It builds a FAISS IndexFlatL2 to enable fast
    nearest-neighbor queries, and exposes `predict` to compute surrogate
    outputs.

    Prediction modes:
      * No gradients provided  -> neighbor f_vals averaging
      * Gradients provided     -> local linear model:
           f(x) ≈ f(x_i) + J_i (x - x_i)  (optionally averaged across k NNs)

    Guarantee logic (k == 1):
      * Uses per-source trust radius and scalar Hessian proxy to decide whether
        a query lies inside a local Taylor trust region (and optional global
        Lipschitz fallback); returns a boolean mask of "outside guarantee".

    Attributes:
      X (np.ndarray): Source point matrix (n_pts, d).
      f_vals (np.ndarray): Source function values (n_pts, fd).
      gradients (np.ndarray | None): Per-source Jacobians (n_pts, fd, d) or None.
      hessians (np.ndarray | None): Per-source scalar Hessian proxies (n_pts,) or None.
      trust_radii (np.ndarray | None): Per-source trust radii (n_pts,) or None.
      P (int): Number of source points.
      M (int): Function output dimension.
      N (int): Input dimension.
      _faiss_index (faiss.Index): Flat L2 index over X (dimension N).
    """

    def __init__(
        self,
        X: np.ndarray,
        f_vals: np.ndarray,
        gradients: np.ndarray | None = None,
        hessians: np.ndarray | None = None,
        trust_radii: np.ndarray | None = None,
    ):
        """Initialize the inference object and build the FAISS index.

        Args:
          X: Array of input points used as cluster centroids with shape (n_pts, d).
          f_vals: Function values at sources, shape (n_pts, fd).
          gradients: Optional Jacobians at sources, shape (n_pts, fd, d).
          hessians: Optional scalar Hessian proxies per source, shape (n_pts,).
          trust_radii: Optional trust radii per source (squared-distance space
            comparisons are used), shape (n_pts,).

        Notes:
          * Shape validation is performed once here to ensure consistency
          * FAISS distances are squared L2 norms; see comments in `predict`.
        """
        # Validate shapes once
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (n_pts, d), got shape {X.shape}"
            )
        if f_vals.ndim != 2:
            raise ValueError(
                f"f_vals must be 2D array with shape (n_pts, fd), got shape {f_vals.shape}"
            )

        n_pts, d = X.shape
        n_f_vals, fd = f_vals.shape
        if n_pts != n_f_vals:
            raise ValueError(
                f"Number of X ({n_pts}) and f_vals ({n_f_vals}) must match."
            )

        if gradients is not None:
            if gradients.ndim != 3:
                raise ValueError(
                    f"gradients must be 3D array with shape (n, fd, d), got shape {gradients.shape}"
                )
            n_grad, fd_grad, d_grad = gradients.shape
            if n_pts != n_grad or fd != fd_grad or d != d_grad:
                raise ValueError(
                    f"Gradient shape {gradients.shape} must match (n_pts, fd, d) = ({n_pts}, {fd}, {d})"
                )

        # Save data as provided (no copies unless necessary downstream)
        self.X = X
        self.f_vals = f_vals
        self.gradients = gradients
        self.hessians = hessians
        self.trust_radii = trust_radii
        self.P = n_pts  # number of training points
        self.M = fd  # function output dimension (fd)
        self.N = d  # input dimension (d)

        # Build a flat L2 FAISS index over X vectors.
        # Note: IndexFlatL2 expects float32 arrays; the original code relies on
        # faiss accepting the dtype in use. We preserve that behavior here.
        self._faiss_index = faiss.IndexFlatL2(self.N)
        self._faiss_index.add(X)  # type: ignore

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
          targets: Query matrix of shape (n_query, d).
          k: Number of nearest neighbors to use.
          unwrap_log: TODO
          disp_time: If True, print elapsed prediction time.
          L_global: Optional global Lipschitz constant for guarantee fallback.
          atol: Absolute tolerance for guarantee checks (k == 1 path).
          rtol: Relative tolerance for guarantee checks (k == 1 path).
          alpha: Extra multiplicative factor (fudge) for Hessian-based bounds.

        Returns:
          tuple[np.ndarray, np.ndarray]:
            - predictions: Array of shape (n_query, fd) with averaged surrogate outputs.
            - outside_mask: Boolean array (n_query,) where True marks points deemed
              *outside* the theoretical guarantee.
                * If no tolerance is provided or k != 1, this mask is all False
                  (i.e., no guarantee screening).

        Notes:
          * FAISS distances D are **squared** L2 norms; all comparisons to
            trust radii or tolerance-derived bounds are in squared-distance
            space. Do not apply square roots on the right-hand side.
          * Local linear mode uses JVP via np.einsum('ijk,ik->ij', J, dx),
            with shapes:
              J:  (n_query, fd, d)  (neighbor Jacobians gathered for each query)
              dx: (n_query, d)     (offset vectors x* - x_i)
              out: (n_query, fd)
        """
        start_time = time.perf_counter()

        # k-NN search via FAISS; D are squared L2 distances, I are indices.
        D, I = self._faiss_index.search(targets, k)  # type: ignore

        # Optional "unlog" of selected dimensions prior to computing offsets.
        # (Distances from FAISS remain unchanged; only offsets for Taylor are
        # affected. Preserve original copy semantics.)
        X_unlog = self.X.copy()
        targets_unlog = targets.copy()
        if len(unwrap_log) > 0:
            for j, val in enumerate(unwrap_log):
                if val:
                    targets_unlog[:, j] = 10 ** targets_unlog[:, j]
                    X_unlog[:, j] = 10 ** X_unlog[:, j]

        # Allocate predictions; accumulate over k neighbors, average at return.
        predictions = np.zeros((len(targets), self.M))
        for j in range(k):
            if self.gradients is None:
                # No gradients: neighbor f_vals averaging (accumulate).
                predictions += self.f_vals[I[:, j]]
            else:
                # Local linear correction:
                # distance_vecs = (x* - x_i) in the "unlogged" space (if used).
                distance_vecs = targets_unlog - X_unlog[I[:, j]]

                # Batched JVP: for each query q, (M×N) @ (N,) -> (M,)
                predictions += self.f_vals[I[:, j]] + np.einsum(
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

    def add_extra_sources(
        self, new_X, new_f_vals, new_gradients, new_hessians, new_dims
    ):
        """Append newly created sources to the index and internal arrays.

        Args:
          new_X: New source points, shape (r, d).
          new_f_vals: New function values, shape (r, fd) or compatible with append.
          new_gradients: New Jacobians, shape (r, fd, d).
          new_hessians: New Hessian proxies, shape (r,) or compatible.
          new_dims: New trust radii (or related dims), shape (r,) or compatible.

        Notes:
          * This method preserves original array update semantics (append/vstack).
          * The FAISS index is incrementally updated by calling `.add` on the
            new X points.
        """
        self.X = np.vstack((self.X, new_X))
        self.f_vals = np.append(self.f_vals, new_f_vals)
        if self.gradients is not None:
            self.gradients = np.vstack((self.gradients, new_gradients))
        if self.hessians is not None:
            self.hessians = np.append(self.hessians, new_hessians)
        if self.trust_radii is not None:
            self.trust_radii = np.append(self.trust_radii, new_dims)
        self._faiss_index.add(new_X)  # type: ignore
