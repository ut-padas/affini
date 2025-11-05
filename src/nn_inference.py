"""
Nearest-neighbor inference with optional first-order (Taylor) correction.
"""

import time

import faiss
import numpy as np

NP_DTYPE = np.float64


class NN_Inference:
    """
    FAISS-backed kNN inference with optional Taylor gradient boosting.

    Shape Conventions:
        - X.shape: (N, D) where N=number of source points, D=input dimension
        - f_vals.shape: (N, FD) where FD=function output dimension
        - gradients.shape: (N, FD, D) where gradients are Jacobian matrices
        - hessians.shape: (N,) scalar Hessian proxies per source
        - trust_radii.shape: (N,) trust radii per source

    Note: If FD=1 (scalar function), f_vals.shape=(N, 1) and gradients.shape=(N, 1, D)

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
      X (np.ndarray): Source point matrix (N, D).
      f_vals (np.ndarray): Source function values (N, FD).
      gradients (np.ndarray | None): Per-source Jacobians (N, FD, D) or None.
      hessians (np.ndarray | None): Per-source scalar Hessian proxies (N,) or None.
      trust_radii (np.ndarray | None): Per-source trust radii (N,) or None.
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
        """
        Initialize the inference object and build the FAISS index.

        Args:
          X: Source points, shape (N, D).
          f_vals: Source function values, shape (N, FD).
          gradients: Optional Jacobians, shape (N, FD, D).
          hessians: Optional Hessian proxies, shape (N,).
          trust_radii: Optional trust radii, shape (N,).
        """
        # Validate shapes once
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (N, D), got shape {X.shape}"
            )
        if f_vals.ndim != 2:
            raise ValueError(
                f"f_vals must be 2D array with shape (N, FD), got shape {f_vals.shape}"
            )

        N, D = X.shape
        N_F, FD = f_vals.shape
        if N != N_F:
            raise ValueError(f"Number of X ({N}) and f_vals ({N_F}) must match.")

        if gradients is not None:
            if gradients.ndim != 3:
                raise ValueError(
                    f"gradients must be 3D array with shape (n, fd, D), got shape {gradients.shape}"
                )
            n_grad, fd_grad, d_grad = gradients.shape
            if N != n_grad or FD != fd_grad or D != d_grad:
                raise ValueError(
                    f"Gradient shape {gradients.shape} must match (N, fd, D) = ({N}, {FD}, {D})"
                )

        self.X = X
        self.f_vals = np.asarray(f_vals)
        self.gradients = np.asarray(gradients) if gradients is not None else None
        self.hessians = np.asarray(hessians) if hessians is not None else None
        self.trust_radii = np.asarray(trust_radii) if trust_radii is not None else None

        self.N_PTS, self.D = X.shape
        self.FD = f_vals.shape[1]

        self._faiss_index = faiss.IndexFlatL2(self.D)
        self._faiss_index.add(X)  # type: ignore

    def predict(
        self,
        targets: np.ndarray,
        k: int,
        unwrap_log: list = [],
        L_global: float | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        alpha: float = 1.0,
    ):
        """
        Predict surrogate outputs for a batch of target points.

        For each target row x*, obtain indices/distances of k nearest neighbors
        via FAISS. If `gradients` are present, compute local linear predictions
        using the Jacobian at each neighbor with an offset vector (x* - x_i),
        then average across k neighbors. Otherwise, average neighbor labels.

        Args:
          targets: Target points, shape (M, D).
          k: Number of nearest neighbors to use.
          unwrap_log: TODO
          disp_time: If True, print elapsed prediction time.
          L_global: Optional global Lipschitz constant for guarantee fallback.
          atol: Absolute tolerance for guarantee checks (k == 1 path).
          rtol: Relative tolerance for guarantee checks (k == 1 path).
          alpha: Extra multiplicative factor (fudge) for Hessian-based bounds.

        Returns:
          tuple[np.ndarray, np.ndarray]:
            - predictions: Array of shape (M, FD) with averaged surrogate outputs.
            - outside_mask: Boolean array of shape (M,) where True marks points
                deemed outside the thereotical guarantee.
                - If no tolerance is provided or k != 1, all entries are False (i.e.,
                  no guarantee screening is performed).
        """
        start_time = time.perf_counter()

        # k-NN search via FAISS
        distances, indices = self._faiss_index.search(targets, k)  # type: ignore

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
        predictions = np.zeros((len(targets), self.FD), dtype=NP_DTYPE)
        for j in range(k):
            if self.gradients is None:
                # No gradients: neighbor f_vals averaging (accumulate).
                predictions += self.f_vals[indices[:, j]]
            else:
                # Local linear correction:
                # distance_vecs = (x* - x_i) in the "unlogged" space (if used).
                distance_vecs = targets_unlog - X_unlog[indices[:, j]]

                # Batched JVP: for each query q, (M×N) @ (N,) -> (M,)
                predictions += self.f_vals[indices[:, j]] + np.einsum(
                    "ijk,ik->ij", self.gradients[indices[:, j]], distance_vecs
                )

        # ---------------- Guarantee / trust-region checks (k == 1) ----------------
        # All bounds below compare squared distances (D[:, 0]) against squared
        # thresholds (e.g., trust_radii**2) or tolerance-derived values.
        if self.trust_radii is None or self.hessians is None:
            print("Hessian proxies and trust radii are required for guarantee checks.")
        else:
            if atol is not None and rtol is not None and k == 1:
                # Combine absolute and relative tolerances; pick the looser (max).
                local_atols = np.minimum(
                    self.trust_radii[indices[:, 0]] ** 2,
                    2 * atol / (alpha * self.hessians[indices[:, 0]]),
                )
                local_rtols = np.minimum(
                    self.trust_radii[indices[:, 0]] ** 2,
                    2
                    * rtol
                    * np.linalg.norm(predictions, 2, axis=1)
                    / (alpha * self.hessians[indices[:, 0]]),
                )
                good_points = distances[:, 0] < np.maximum(local_atols, local_rtols)
            elif atol is not None and L_global is not None and k == 1:
                # Absolute tolerance with global Lipschitz fallback.
                local_tols = np.minimum(
                    self.trust_radii[indices[:, 0]] ** 2,
                    2 * atol / (alpha * self.hessians[indices[:, 0]]),
                )
                global_tol = 2 * atol / L_global
                good_points = np.logical_or(
                    (distances[:, 0] < local_tols),
                    np.logical_and(
                        local_tols < distances[:, 0], distances[:, 0] < global_tol
                    ),
                )
            elif atol is not None and k == 1:
                # Absolute tolerance only (no rtol, no global Lipschitz).
                local_tols = np.minimum(
                    self.trust_radii[indices[:, 0]] ** 2,
                    2 * atol / (alpha * self.hessians[indices[:, 0]]),
                )
                good_points = distances[:, 0] < local_tols
            else:
                # No guarantee screening (k != 1 or no tolerance): mark all as good.
                good_points = np.full(len(distances), True)

        end_time = time.perf_counter()
        print(f"Time elapsed for predictions: {end_time - start_time} seconds")

        # Average over neighbors and return outside-of-guarantee mask.
        # Bad points are all False when tol is not specified.
        return predictions / k, np.logical_not(good_points)

    def add_extra_sources(
        self,
        new_X: np.ndarray,
        new_f_vals: np.ndarray,
        new_gradients: np.ndarray | None,
        new_hessians: np.ndarray | None,
        new_radii: np.ndarray | None,
    ):
        """Append newly created sources to the index and internal arrays.

        Args:
          new_X: New source points, shape (r, D).
          new_f_vals: New function values, shape (r, FD) or compatible with append.
          new_gradients: New Jacobians, shape (r, fd, D).
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
            self.gradients = np.vstack((self.gradients, new_gradients))  # type: ignore
        if self.hessians is not None:
            self.hessians = np.append(self.hessians, new_hessians)  # type: ignore
        if self.trust_radii is not None:
            self.trust_radii = np.append(self.trust_radii, new_radii)  # type: ignore
        self._faiss_index.add(new_X)  # type: ignore
