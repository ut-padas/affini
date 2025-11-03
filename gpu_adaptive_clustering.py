from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cuml
import cupy as cp
from cuml.cluster import KMeans as cuKMeans

# Keep all cuML outputs on device
cuml.set_global_output_type("cupy")

CP_DTYPE = cp.float64


@dataclass
class Cluster:
    index: int
    centroid: cp.ndarray  # (d,)
    value: cp.ndarray  # (m,)
    gradient: Optional[cp.ndarray] = None  # (m, d) or None
    hessian: float = 0.0
    radius: float = 0.0


@dataclass
class TreeNode:
    cluster: Optional[Cluster] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    def leaves(self) -> list[Cluster]:
        if self.cluster is not None:
            return [self.cluster]
        out = []
        if self.left:
            out.extend(self.left.leaves())
        if self.right:
            out.extend(self.right.leaves())
        return out


class GPUAdaptiveClusterer:
    """
    Fully GPU-resident adaptive clustering:
    - X, F, centroids, labels, reductions all on CuPy
    - cuML KMeans for initial and binary splits
    - Function/gradient evaluation via user-provided f_gpu (CuPy-based)
    """

    def __init__(
        self,
        *,
        atol: float = 0.01,
        rtol: float = 0.01,
        n_clusters: int = 10,
        min_points: int = 1,
        use_gradient: bool = False,
        seed: int = 0,
        batch_eval: int = 131072,  # for f_gpu batching (tune as needed)
    ):
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.n_clusters = int(n_clusters)
        self.min_points = int(min_points)
        self.use_gradient = bool(use_gradient)
        self.seed = int(seed)
        self.batch_eval = int(batch_eval)
        self._cluster_id = 0

    # ---------- public API ----------

    def fit(
        self,
        X_gpu: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
        f_vals_gpu: Optional[cp.ndarray] = None,
    ) -> list[TreeNode]:
        """
        X_gpu: (N, d) CuPy float64
        f_gpu: accepts (B, d) CuPy, returns:
               - values (B, m) CuPy, or
               - (values, gradients) where gradients is (B, m, d) CuPy
        f_vals_gpu: optional precomputed values (N, m) CuPy
        """
        X = self._ensure_2d_cp(X_gpu).astype(CP_DTYPE, copy=False)

        if f_vals_gpu is not None:
            F = self._ensure_2d_cp(f_vals_gpu).astype(CP_DTYPE, copy=False)
            if F.shape[0] != X.shape[0]:
                raise ValueError("X and provided f_vals_gpu length mismatch")
        else:
            F = self._eval_function_gpu(X, f_gpu)

        # initial clustering
        labels, centroids = self._initial_cluster_gpu(X)

        trees: list[TreeNode] = []
        for c_id in range(self.n_clusters):
            mask = labels == c_id
            if not bool(mask.any()):
                continue
            Xi = X[mask]
            Fi = F[mask]
            ci = centroids[c_id]

            if self._meets_tolerance_gpu(Xi, Fi, ci, f_gpu):
                leaf = self._create_cluster_gpu(Xi, Fi, ci, f_gpu)
                trees.append(TreeNode(cluster=leaf))
            else:
                subtree = self._split_recursive_gpu(Xi, Fi, f_gpu)
                if subtree is not None:
                    trees.append(subtree)

        return trees

    # ---------- GPU helpers ----------

    @staticmethod
    def _ensure_2d_cp(A: cp.ndarray) -> cp.ndarray:
        if A.ndim == 1:
            return A.reshape(-1, 1)
        return A

    def _eval_function_gpu(
        self,
        X: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
    ) -> cp.ndarray:
        """
        Evaluate f on GPU in batches. Returns values (N, m) CuPy.
        If self.use_gradient is True, we still only cache values here;
        gradients are re-evaluated at centroids when needed.
        """
        N = X.shape[0]
        vals: list[cp.ndarray] = []
        for s in range(0, N, self.batch_eval):
            e = min(N, s + self.batch_eval)
            out = f_gpu(X[s:e])
            if isinstance(out, tuple):
                v, _ = out
            else:
                v = out
            if v.ndim == 1:
                v = v[:, None]
            vals.append(v.astype(CP_DTYPE, copy=False))
        return cp.concatenate(vals, axis=0)

    def _normalize_gpu(
        self, X: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = cp.where(std == 0, 1.0, std)
        Xn = (X - mean) / std
        return Xn, mean, std

    def _initial_cluster_gpu(self, X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        Xn, mean, std = self._normalize_gpu(X)
        km = cuKMeans(
            n_clusters=self.n_clusters, random_state=self.seed, init="k-means++"
        )
        labels = km.fit_predict(Xn)  # cupy (due to global output type)
        cents_n = km.cluster_centers_  # (k, d) cupy
        cents = cents_n * std + mean  # denorm on GPU
        return labels, cents

    def _split_binary_gpu(self, X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        Xn, mean, std = self._normalize_gpu(X)
        km = cuKMeans(n_clusters=2, random_state=self.seed, init="k-means++")
        labels = km.fit_predict(Xn)
        cents_n = km.cluster_centers_
        cents = cents_n * std + mean
        return labels, cents

    def _compute_max_error_gpu(
        self,
        X: cp.ndarray,
        F: cp.ndarray,
        centroid: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
    ) -> float:
        """
        Returns Python float but performs all math on GPU.
        """
        if not self.use_gradient:
            f_c = f_gpu(centroid[None, :])
            if isinstance(f_c, tuple):
                f_c = f_c[0]
            if f_c.ndim == 1:
                f_c = f_c[None, :]
            diffs = F - f_c
        else:
            out = f_gpu(centroid[None, :])
            if not isinstance(out, tuple):
                raise ValueError(
                    "use_gradient=True requires f_gpu to return (values, gradients)."
                )
            f_c, g_c = out
            if f_c.ndim == 1:
                f_c = f_c[None, :]
            if g_c.ndim == 2:
                g_c = g_c[None, :, :]  # (1, m, d)
            dx = X - centroid[None, :]  # (N, d)
            # linear model: f_c + dx @ g^T  → (N, m)
            f_lin = f_c + dx @ g_c[0].T
            diffs = F - f_lin

        errs = cp.linalg.norm(diffs, axis=1)  # (N,)
        max_err = errs.max()
        return float(max_err)  # extract scalar only

    def _meets_tolerance_gpu(
        self,
        X: cp.ndarray,
        F: cp.ndarray,
        centroid: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
    ) -> bool:
        max_err = self._compute_max_error_gpu(X, F, centroid, f_gpu)
        if max_err <= self.atol:
            return True
        # relative check
        norms = cp.linalg.norm(F, axis=1)
        if norms.size > 0:
            min_norm = float(norms.min())
            if max_err <= self.rtol * max(min_norm, 1e-300):
                return True
        return False

    def _create_cluster_gpu(
        self,
        X: cp.ndarray,
        F: cp.ndarray,
        centroid: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
    ) -> Cluster:
        if not self.use_gradient:
            v = f_gpu(centroid[None, :])
            if isinstance(v, tuple):
                v = v[0]
            if v.ndim == 1:
                v = v[None, :]
            grad = None
        else:
            v, g = f_gpu(centroid[None, :])
            if v.ndim == 1:
                v = v[None, :]
            if g.ndim == 2:
                g = g[None, :, :]  # (1, m, d)
            grad = g[0]  # (m, d)

        # radius, hessian proxy (all on GPU; convert to scalars at the end)
        dx = X - centroid[None, :]
        radius = float(cp.linalg.norm(dx, axis=1).max()) if X.shape[0] > 0 else 0.0
        max_err = self._compute_max_error_gpu(X, F, centroid, f_gpu)
        hess = (2.0 * max_err / (radius * radius)) if radius > 0.0 else 0.0

        cl = Cluster(
            index=self._cluster_id,
            centroid=centroid.astype(CP_DTYPE, copy=False).reshape(-1),
            value=v[0].astype(CP_DTYPE, copy=False).reshape(-1),
            gradient=None if grad is None else grad.astype(CP_DTYPE, copy=False),
            hessian=float(hess),
            radius=float(radius),
        )
        self._cluster_id += 1
        return cl

    def _generate_extra_points_gpu(
        self,
        sub_X: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
        need: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Generate 'need' convex-combination points entirely on GPU and evaluate f_gpu on them.
        Returns (new_X, new_F) both CuPy.
        """
        Q = sub_X.shape[0]
        if Q < 2:
            raise ValueError("Need at least two points to generate extra points.")
        rng = cp.random.default_rng(self.seed)
        lam = rng.random(need, dtype=CP_DTYPE)  # (need,)
        idx0 = rng.integers(0, Q, size=need)
        offs = rng.integers(1, Q, size=need)
        idx1 = (idx0 + offs) % Q
        x0 = sub_X[idx0]  # (need, d)
        x1 = sub_X[idx1]
        lamv = lam[:, None]
        new_X = lamv * x0 + (1.0 - lamv) * x1  # (need, d)
        out = f_gpu(new_X)
        new_F = out[0] if isinstance(out, tuple) else out
        if new_F.ndim == 1:
            new_F = new_F[:, None]
        return new_X, new_F

    def _process_subcluster_gpu(
        self,
        X: cp.ndarray,
        F: cp.ndarray,
        labels: cp.ndarray,
        label: int,
        centroid: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
    ) -> Optional[TreeNode]:
        mask = labels == label
        if not bool(mask.any()):
            return None

        sub_X = X[mask]
        sub_F = F[mask]

        if sub_X.shape[0] <= 1:
            return None

        # ensure minimum points by synthesizing convex-combo points on GPU
        if sub_X.shape[0] <= self.min_points:
            need = self.min_points - sub_X.shape[0] + 1
            extra_X, extra_F = self._generate_extra_points_gpu(sub_X, f_gpu, need)
            sub_X = cp.concatenate([sub_X, extra_X], axis=0)
            sub_F = cp.concatenate([sub_F, extra_F], axis=0)

        # dead branch?
        radius = float(cp.linalg.norm(sub_X - centroid[None, :], axis=1).max())
        if cp.isclose(cp.asarray(radius), 0.0):
            return None

        if self._meets_tolerance_gpu(sub_X, sub_F, centroid, f_gpu):
            return TreeNode(
                cluster=self._create_cluster_gpu(sub_X, sub_F, centroid, f_gpu)
            )
        else:
            return self._split_recursive_gpu(sub_X, sub_F, f_gpu)

    def _split_recursive_gpu(
        self,
        X: cp.ndarray,
        F: cp.ndarray,
        f_gpu: Callable[[cp.ndarray], cp.ndarray | Tuple[cp.ndarray, cp.ndarray]],
    ) -> Optional[TreeNode]:
        if X.shape[0] < 2:
            return None

        labels, cents = self._split_binary_gpu(X)

        left = self._process_subcluster_gpu(X, F, labels, 0, cents[0], f_gpu)
        right = self._process_subcluster_gpu(X, F, labels, 1, cents[1], f_gpu)

        if left is None and right is None:
            return None
        return TreeNode(left=left, right=right)


# -------- optional: flatten to CuPy tensors (no host copies) --------


def flatten_cluster_centers_with_gradient_gpu(nc: int, n: int, m: int, tree_or_trees):
    """
    Return CuPy arrays:
      centroids: (nc, n), values: (nc, m), gradients: (nc, m, n), hessians: (nc,), radii: (nc,)
    Everything stays on GPU.
    """
    centroids = cp.empty((nc, n), dtype=CP_DTYPE)
    values = cp.empty((nc, m), dtype=CP_DTYPE)
    grads = cp.empty((nc, m, n), dtype=CP_DTYPE)
    hess = cp.empty((nc,), dtype=CP_DTYPE)
    radii = cp.empty((nc,), dtype=CP_DTYPE)

    stack = []
    try:
        for node in tree_or_trees:
            if node is not None:
                stack.append(node)
    except TypeError:
        if tree_or_trees is not None:
            stack.append(tree_or_trees)

    k = 0
    while stack:
        node = stack.pop()
        cl = node.cluster
        if cl is not None:
            c = cl.centroid.reshape(-1)
            v = cl.value.reshape(-1)
            if c.size != n:
                raise ValueError(f"Centroid size {c.size} != n")
            if v.size != m:
                raise ValueError(f"Value size {v.size} != m")
            centroids[k] = c
            values[k] = v

            if cl.gradient is None:
                grads[k].fill(0)
            else:
                g = cl.gradient
                if g.shape == (m, n):
                    grads[k] = g
                elif g.ndim == 1 and m == 1 and g.size == n:
                    grads[k, 0, :] = g
                elif g.ndim == 2 and g.shape == (n, m):
                    grads[k] = g.T
                elif g.size == m * n:
                    grads[k] = g.reshape(m, n)
                else:
                    raise ValueError(
                        f"Gradient shape {g.shape} not compatible with (m,n)=({m},{n})"
                    )

            hess[k] = CP_DTYPE(cl.hessian)
            radii[k] = CP_DTYPE(cl.radius)
            k += 1
            if k > nc:
                raise ValueError(f"More leaves ({k}) than expected ({nc})")
        else:
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    if k != nc:
        raise ValueError(f"Expected nc={nc} leaves, found {k}")
    return centroids, values, grads, hess, radii


if __name__ == "__main__":
    import time

    # Example f_gpu: returns (values, gradients) on GPU
    def f_gpu(Xb: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        # Xb: (B, 3)
        x1 = Xb[:, 0]
        x2 = Xb[:, 1]
        x3 = Xb[:, 2]
        val0 = x1**2 + x2**2
        val1 = x1 - x2 + x3
        values = cp.stack([val0, val1], axis=1)  # (B, 2)

        # gradient for each row: shape (B, m, d) = (B, 2, 3)
        g0 = cp.stack([2 * x1, 2 * x2, cp.zeros_like(x1)], axis=1)  # (B,3)
        g1 = cp.stack([cp.ones_like(x1), -cp.ones_like(x1), cp.ones_like(x1)], axis=1)
        grads = cp.stack([g0, g1], axis=1)  # (B,2,3)
        return values, grads

    # Build data on GPU
    rng = cp.random.default_rng(0)
    X_gpu = rng.uniform(-10, 10, size=(800_000, 3)).astype(cp.float64)
    print("Data X_gpu shape:", X_gpu.shape)

    clusterer = GPUAdaptiveClusterer(
        atol=1.0, rtol=0.01, n_clusters=5, min_points=5, use_gradient=True, seed=0
    )
    trees = clusterer.fit(X_gpu, f_gpu)  # stays entirely on GPU

    # If you need flattened outputs on GPU:
    leaves = sum(len(t.leaves()) for t in trees)
    C, V, G, H, R = flatten_cluster_centers_with_gradient_gpu(leaves, 3, 2, trees)
    print(f"Found {leaves} clusters.")
    print(
        f"Centroids shape: {C.shape}, Values shape: {V.shape}, Gradients shape: {G.shape}"
    )
