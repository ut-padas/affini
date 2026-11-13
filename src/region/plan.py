"""``RegionPlan``: one region-mode batch from fit and validation to accepted leaves and children."""

import contextlib
import math

import numpy as np

from src import IS_CUPY_BACKEND, Array, _to_host, xp
from src.components.acceptance import tolerance_vector
from src.components.local_models import ClusterFitData, deploy_leaves, output_columns
from src.components.sampling import (
    ClusterPoints,
    augment_clusters,
    augmentation_targets,
)
from src.components.splitting import SplitData
from src.engine.clusters import ParamSpec
from src.numerics import (
    _reduce_max_1d,
    _reduce_sum_1d,
    det_einsum,
    det_matmul,
    sort_by_block_then_value,
)
from src.region.constants import (
    ACCEPTED,
    FAILED,
    INSUFFICIENT_SAMPLING,
    MAX_DEPTH,
    MAX_DEPTH_LIMIT,
    NOT_PERFORMED,
    PASSED,
    PURPOSE_DIRECTION,
    PURPOSE_TRAIN,
    PURPOSE_VALIDATION,
    SPLIT_FAILED,
)
from src.region.geometry import Domain, Roots, gap_cuts, plane_scores
from src.region.rng import _normals, counter_uniform, node_keys
from src.region.sampling import HitAndRunSampler, RegionPoints
from src.region.split_search import (
    _hard_sse,
    _interleave,
    _soft_gates,
    _soft_objective,
    _soft_step,
    _SplitRows,
    _unit,
    segment_sum,
)
from src.numerics import repeats


# --- The plan ------------------------------------------------------------------------
class RegionPlan:
    """
    The region algorithm's plan methods over the shared executor
    (``src.engine.refinement._run_local_refinement``): batch sizing, steal settings,
    output columns and ``process_batch``. ``counters`` accumulates this rank's split
    diagnostics (reported as ``profiling_stats["region"]``).
    """

    def __init__(self, config, d: int, fd: int, domain: Domain, roots: Roots):
        self.config, self.d, self.fd = config, d, fd
        self.domain, self.roots = domain, roots
        s = config.region
        self.settings = s
        self.model = config.components().local_model
        self.p_eff = self.model.n_features(d)  # ART: sample multiplier x the leaf model's features
        self.convex_sampling = s.sampling == "convex_combination"
        self.n_target = 0 if self.convex_sampling else math.ceil(s.sample_multiplier * self.p_eff)
        self.n_val = int(s.validation_samples)
        self.min_side = self.p_eff + int(s.min_side_margin)
        if not self.convex_sampling and self.n_target < 2 * self.min_side:
            raise ValueError(
                f"Region nodes get {self.n_target} training points, fewer than twice the "
                f"minimum side size {self.min_side}: no split could ever be made. Raise "
                f"region.sample_multiplier or lower region.min_side_margin."
            )
        if config.max_levels > MAX_DEPTH_LIMIT:
            raise ValueError(f"Region mode supports max_levels <= {MAX_DEPTH_LIMIT}; got {config.max_levels}.")
        self.sampler = HitAndRunSampler(domain, roots, s.walk_steps, s.max_walk_steps, s.mixing_z)
        node_pts = max(self.n_target, 2 * config.min_pts_per_cluster) + self.n_val
        self.steal_settings = {
            "min_load": s.steal_min_load or 4 * node_pts,
            "steal_size": s.steal_size or 2 * node_pts,
            "lease_received": True,
            "work_sizes": self.work_sizes,  # one work unit for loads, eligibility and grants
        }
        self.counters = {
            "splits": 0, "split_failed": 0, "rank_deficient": 0, "insufficient_sampling": 0,
            "soft_improved": 0, "soft_gradient_converged": 0, "soft_line_search_stalled": 0,
            "soft_gate_saturated": 0, "soft_nonfinite": 0, "soft_iteration_limit": 0,
        }  # fmt: skip

    def work_sizes(self, sizes: np.ndarray) -> np.ndarray:
        """Points each queued node will occupy on the device: after top-up (or augmentation) and validation."""
        if self.convex_sampling:
            return np.asarray(_to_host(augmentation_targets(xp.asarray(sizes), self.config.min_pts_per_cluster)))
        return np.maximum(sizes, self.n_target) + self.n_val

    def bytes_per_point(self) -> int:
        """
        Peak device bytes per processed point (training, validation and split work arrays).
        B3/B4's per-node candidate arrays (``n_bins - 1`` Grams per side, the screen) are
        charged to the fewest points a node can hold, so small nodes do not undercount.
        Carried constraints are charged per point (``2 h (d+1)``), which overestimates deep
        trees several times over. That is kept on purpose: charging them per node (d = 32,
        depth 12: 5.2 GB in use instead of 1.2 GB) built the same tree 3.4x slower on an
        8 GB laptop GPU under WSL, where a nearly full card is paged to host memory
        (design 42.11).
        """
        d, fd, h = self.d, self.fd, self.config.max_levels + 1
        p = self.p_eff
        g = fd * d if self.config.requires_grad_everywhere else 0
        search = 0
        if self.settings.split_search != "gap":
            # Gathered feature and output blocks (one copy of each) for the batched Grams.
            search = 2 * (p + fd) + 4
            nb, D = self.settings.n_bins, self.settings.n_directions
            per_node = 3 * (nb - 1) * (p * p + p * fd + 1) + 2 * D * (nb - 1)
            search += -(-per_node // max(1, int(self.work_sizes(np.zeros(1, np.int64))[0])))
        return 8 * (4 * d + 3 * fd + 2 * g + 2 * h * (d + 1) + 12 + search)

    def leaf_columns(self, model) -> dict[str, ParamSpec]:
        cols = output_columns(model, self.d, self.fd)
        if not self.settings.store_Q:
            del cols["Q"]
        return {
            **cols,
            "root": ParamSpec((), np.int64),
            "path_code": ParamSpec((), np.int64),
            "terminal_reason": ParamSpec((), np.int8),
            "validation_status": ParamSpec((), np.int8),
        }

    def split_columns(self) -> dict[str, ParamSpec]:
        return {
            "root": ParamSpec((), np.int64),
            "path_code": ParamSpec((), np.int64),
            "w": ParamSpec((self.d,)),
            "z": ParamSpec(()),
        }

    def initial_queue(self, X: np.ndarray, F: np.ndarray, G: np.ndarray | None):
        """Every root as a queued node (possibly with no points), points sorted by root."""
        from src.engine.clusters import ClusterBatch

        root = _to_host(self.roots.root_of(self.domain.normalize(xp.asarray(X)))).astype(np.int64)
        order = np.argsort(root, kind="stable")
        R = self.roots.n_roots
        return ClusterBatch(
            X[order], F[order], G[order] if G is not None else None,
            np.bincount(root, minlength=R).astype(np.int64),
            root=np.arange(R, dtype=np.int64), path_code=np.ones(R, dtype=np.int64),
            constraints=np.zeros((R, 0, self.d + 1)),
        )  # fmt: skip

    def process_batch(self, *, X, F, G, sizes, depth, root, path_code, constraints, components, evaluator,
                      accepted, splits, next_level, progress=None, timer=None) -> dict[str, int]:  # fmt: skip
        """
        Top up, fit, deploy, validate and split one batch of nodes (see the module
        docstring), appending leaves to ``accepted``, split records to ``splits`` and
        children to ``next_level``. ``progress()``, if given, is called between bounded
        chunks of long split work so steal requests are answered meanwhile. ``timer(key)``
        (the executor's profiler phases) times each stage under the shared phase keys:
        sampling as ``time_augment``, statistics and fit as ``time_error_calc``, deploying
        leaves as ``time_good_Q``, splitting as ``time_bad_split``. Returns counts for the
        executor's statistics.
        """
        timer = timer or (lambda key: contextlib.nullcontext())
        cfg = self.config
        model = components.local_model
        acceptance, trust_region = components.acceptance, components.trust_region
        K = int(sizes.shape[0])
        keys = node_keys(cfg.seed, root, path_code)
        geo = RegionPoints(root, depth, keys, constraints, PURPOSE_TRAIN)
        stats = {"generated": 0, "fallback_clusters": 0, "fallback_pts": 0}

        # 1. Training points: inherited ones, then new ones inside the polytope.
        with timer("time_augment"):
            # B1 keeps each node's points, except that an empty root (possible for grid or
            # Voronoi cells) is sampled up to 2 * p_eff points so it can be fitted.
            n_train = (
                xp.where(sizes == 0, 2 * self.p_eff, sizes) if self.convex_sampling else xp.maximum(sizes, self.n_target)
            )
            aug = augment_clusters(X, F, G, sizes, n_train, self.sampler, evaluator, None, region=geo)
            Xt, Ft, Gt, labels = aug.X, aug.F, aug.G, aug.labels
            stats["generated"] += 0 if aug.X_new is None else int(aug.X_new.shape[0])
            starts = xp.r_[0, xp.cumsum(n_train)[:-1]]
            del aug
            insufficient = geo.insufficient if geo.insufficient is not None else xp.zeros(K, dtype=bool)
            Xv = Fv = Gv = None
            if self.n_val:
                geo_v = RegionPoints(root, depth, keys, constraints, PURPOSE_VALIDATION)
                Xv = self.sampler.sample(ClusterPoints(X, sizes, region=geo_v), xp.full(K, self.n_val))
                Fv, Gv = evaluator(Xv, need_grad=Gt is not None)
                stats["generated"] += int(Xv.shape[0])
                insufficient = insufficient | geo_v.insufficient
            self.counters["insufficient_sampling"] += int(xp.sum(insufficient))

        # 2. Statistics and fit on the training points only.
        with timer("time_error_calc"):
            mus = _reduce_sum_1d(Xt, starts) / n_train[:, None]
            if cfg.requires_f_centroids or cfg.requires_grad_centroids:
                f_mus, g_mus = evaluator(mus, need_grad=cfg.requires_grad_centroids)
            else:  # no component reads f(c): column "f" is NaN rather than an oracle call
                f_mus, g_mus = xp.full((K, self.fd), xp.nan), None
            fit = ClusterFitData(X=Xt, F=Ft, G=Gt, labels=labels, starts=starts, sizes=n_train,
                                 centroids=mus, f_c=f_mus, g_c=g_mus)  # fmt: skip
            params = model.fit(fit)
            F_hat = model.evaluate({**params, "centroids": mus, "f": f_mus}, Xt, labels)
            good, E_abs, E_rel = acceptance(Ft, F_hat, starts, n_train)
            del F_hat

        # 3. Deploy the screened models, then check them on the validation points.
        with timer("time_good_Q"):
            store_Q = self.settings.store_Q
            is_leaf = xp.zeros(K, dtype=bool)
            status_v = xp.full(K, NOT_PERFORMED, dtype=xp.int8)
            good_idx = xp.nonzero(good)[0]
            # Set after validation below: a node whose chains drifted is reported as such
            # whatever its validation said, since its points need not represent its cell.
            if int(good_idx.shape[0]) > 0:
                leaves = deploy_leaves(model, trust_region, acceptance, fit, good_idx, params, E_abs, E_rel, store_Q=store_Q)
                if self.n_val and int(leaves.idx.shape[0]) > 0:
                    ok = self._validate(model, leaves, Xv, Fv, components)
                    status_v[leaves.idx] = xp.where(ok, PASSED, FAILED).astype(xp.int8)
                    keep = xp.nonzero(ok)[0]
                    leaves.idx, leaves.rows = leaves.idx[keep], {k: v[keep] for k, v in leaves.rows.items()}
                status_v = xp.where(insufficient, INSUFFICIENT_SAMPLING, status_v).astype(xp.int8)
                self._append_leaves(accepted, leaves, root, path_code, ACCEPTED, status_v)
                is_leaf[leaves.idx] = True

        # 4. Split the rejected nodes; those that cannot split keep their model as a leaf.
        rejected = ~is_leaf
        fallback = rejected & (depth >= cfg.max_levels)
        to_split = xp.nonzero(rejected & ~fallback)[0]
        failed_split = to_split[:0]
        if int(to_split.shape[0]) > 0:
            with timer("time_bad_split"):
                failed_split, n_gen = self._split(
                    to_split, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, path_code,
                    constraints, splits, next_level, progress,
                )  # fmt: skip
            stats["generated"] += n_gen
        status_v = xp.where(insufficient, INSUFFICIENT_SAMPLING, status_v).astype(xp.int8)
        for idx, reason in ((xp.nonzero(fallback)[0], MAX_DEPTH), (failed_split, SPLIT_FAILED)):
            if int(idx.shape[0]) > 0:
                with timer("time_good_Q"):
                    leaves = deploy_leaves(model, trust_region, acceptance, fit, idx, params, E_abs, E_rel,
                                           retest=False, store_Q=store_Q)  # fmt: skip
                self._append_leaves(accepted, leaves, root, path_code, reason, status_v)
                stats["fallback_clusters"] += int(idx.shape[0])
                stats["fallback_pts"] += int(xp.sum(n_train[idx]))
        stats.update(
            accepted=int(xp.sum(is_leaf)), rejected=K - int(xp.sum(is_leaf)), n_pts=int(Xt.shape[0])
        )
        return stats

    def _validate(self, model, leaves, Xv, Fv, components) -> Array:
        """Whether each deployed leaf passes the acceptance test on its node's validation points."""
        L = int(leaves.idx.shape[0])
        rows = (leaves.idx[:, None] * self.n_val + xp.arange(self.n_val)).ravel()
        labels = repeats(xp.arange(L), xp.full(L, self.n_val))
        F_hat = model.evaluate(leaves.rows, Xv[rows], labels)
        sizes = xp.full(L, self.n_val, dtype=xp.int64)
        ok, _, _ = components.acceptance(Fv[rows], F_hat, xp.arange(L) * self.n_val, sizes)
        return ok

    def _append_leaves(self, accepted, leaves, root, path_code, reason: int, status_v) -> None:
        idx = leaves.idx
        accepted.append({
            **leaves.rows,
            "root": root[idx], "path_code": path_code[idx],
            "terminal_reason": xp.full(int(idx.shape[0]), reason, dtype=xp.int8),
            "validation_status": status_v[idx],
        })  # fmt: skip

    # --- Splitting -------------------------------------------------------------------

    def _split(self, idx, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, path_code, constraints,
               splits, next_level, progress) -> tuple[Array, int]:  # fmt: skip
        """
        Split nodes ``idx`` on their training points plus their (now spent) validation
        points; return the nodes whose split was invalid and the number of new oracle
        points (B1's augmentation). Validation points of a failed node may train its
        children, but never validate them.
        """
        d = self.d
        k = int(idx.shape[0])
        sel = xp.zeros(int(fit.sizes.shape[0]), dtype=bool)
        sel[idx] = True
        train_rows = sel[fit.labels]
        counts = fit.sizes[idx]
        X, F = fit.X[train_rows], fit.F[train_rows]
        G = fit.G[train_rows] if fit.G is not None else None
        n_generated = 0
        if self.n_val:
            # Interleave per node: its training points, then its validation points.
            v_rows = (idx[:, None] * self.n_val + xp.arange(self.n_val)).ravel()
            total = counts + self.n_val
            labels = repeats(xp.arange(k), total)
            pos = xp.arange(int(xp.sum(total))) - (xp.cumsum(total) - total)[labels]
            is_train = pos < counts[labels]
            X = _interleave(X, Xv[v_rows], is_train)
            F = _interleave(F, Fv[v_rows], is_train)
            G = _interleave(G, Gv[v_rows], is_train) if G is not None else None
            counts = total
        if self.convex_sampling:
            # B1: AFINNI's augmentation (convex combinations, inside the parent's hull and
            # so inside its polytope), with a per-batch stream as in AFINNI mode.
            targets = augmentation_targets(counts, self.config.min_pts_per_cluster)
            rng = xp.random.default_rng(int(_to_host(keys[idx[0]])) % 2**63)
            aug = augment_clusters(X, F, G, counts, targets, self.config.components().sampler, evaluator, rng,
                                   centroids=fit.centroids[idx])  # fmt: skip
            X, F, G, labels, counts = aug.X, aug.F, aug.G, aug.labels, targets
            n_generated = 0 if aug.X_new is None else int(aug.X_new.shape[0])
        elif not self.n_val:
            labels = repeats(xp.arange(k), counts)
        data = SplitData(X=X, F=F, G=G, labels=labels, counts=counts, centroids=fit.centroids[idx],
                         f_c=fit.f_c[idx], g_c=fit.g_c[idx] if fit.g_c is not None else None,
                         hessian_fn=evaluator.hessian_fn)  # fmt: skip
        U = self.domain.normalize(X)
        w, z = self._propose(data, U, components, keys[idx], progress)

        # Verify: the stored predicate defines the children; both keep min_side points.
        scores = plane_scores(U, w[labels])
        order = sort_by_block_then_value(scores, labels)
        right = (scores >= z[labels]).astype(xp.int64)
        starts = xp.r_[0, xp.cumsum(counts)[:-1]]
        n_right = xp.rint(segment_sum(right.astype(xp.float64), starts)).astype(xp.int64)
        cut = counts - n_right
        valid = xp.all(xp.isfinite(w), axis=1) & xp.isfinite(z) & (cut >= self.min_side) & (n_right >= self.min_side)
        side = plane_scores(U[order], w[labels]) - z[labels] >= 0
        pos = xp.arange(int(labels.shape[0])) - starts[labels]
        if not bool(xp.all((side == (pos >= cut[labels])) | ~valid[labels])):
            raise RuntimeError("A region split predicate does not reproduce its children.")
        good = xp.nonzero(valid)[0]
        self.counters["splits"] += int(good.shape[0])
        self.counters["split_failed"] += k - int(good.shape[0])
        if int(good.shape[0]) > 0:
            keep = valid[labels][order]
            child_sizes = xp.stack([cut[good], counts[good] - cut[good]], axis=1).ravel()
            node, pc = idx[good], path_code[idx[good]]
            row = xp.concatenate([w[good], z[good][:, None]], axis=1)  # left: w^T u <= z
            parent_c = constraints[node]
            children_c = xp.stack([
                xp.concatenate([parent_c, row[:, None, :]], axis=1),
                xp.concatenate([parent_c, -row[:, None, :]], axis=1),
            ], axis=1).reshape(2 * int(good.shape[0]), -1, d + 1)  # fmt: skip
            rows_sorted = order[keep]
            next_level.append(
                X[rows_sorted], F[rows_sorted], G[rows_sorted] if G is not None else None, child_sizes,
                depth=xp.repeat(depth[node] + 1, 2), root=xp.repeat(root[node], 2),
                path_code=xp.stack([2 * pc, 2 * pc + 1], axis=1).ravel(), constraints=children_c,
            )  # fmt: skip
            splits.append({"root": root[node], "path_code": pc, "w": w[good], "z": z[good]})
        return idx[~valid], n_generated

    def _propose(self, data: SplitData, U: Array, components, keys: Array, progress) -> tuple[Array, Array]:
        """Hyperplanes ``(w, z)`` in ``u`` for every node of ``data`` (invalid ones are caught by the caller)."""
        mode = self.settings.split_search
        V, M = self._metric_basis(data, components.split_strategy)
        width = xp.asarray(self.domain.width)
        lead = _unit(det_einsum("kij,kj->ki", M, V[:, :, -1]) * width)  # scores (x - c)^T M v, in u
        if mode == "gap":
            scores = plane_scores(U, lead[data.labels])
            order = sort_by_block_then_value(scores, data.labels)
            ok, _, z = gap_cuts(scores[order], data.labels, data.counts, self.min_side)
            return lead, xp.where(ok, z, xp.nan)
        rows = self._split_rows(data, U)
        bank = self._direction_bank(rows, lead, V, M, width, keys)
        w, z, sse = self._threshold_search(rows, bank)
        if mode == "soft":
            if progress is not None:
                progress()
            w, z = self._soft_refine(rows, w, z, sse, progress)
        return w, z

    def _metric_basis(self, data: SplitData, splitter) -> tuple[Array, Array]:
        """The split strategy's metric ``M`` and the eigenvectors ``V`` of ``M C M`` (ascending)."""
        M, CovZ = splitter.metric(data)
        return xp.linalg.eigh(CovZ)[1], M

    def _split_rows(self, data: SplitData, U: Array) -> _SplitRows:
        """The leaf model's split features and tolerance-scaled outputs, so components combine by their tolerances."""
        fd = data.F.shape[1]
        acc = self.config.components().acceptance
        atol = xp.maximum(tolerance_vector("atol", acc.atol, fd), 0.0)
        rtol = xp.maximum(tolerance_vector("rtol", acc.rtol, fd), 0.0)
        starts = xp.r_[0, xp.cumsum(data.counts)[:-1]]
        Fmax = xp.stack([_reduce_max_1d(xp.abs(data.F[:, j]), starts) for j in range(fd)], axis=1)
        scale = xp.maximum(xp.maximum(atol[None, :], rtol[None, :] * Fmax), 1e-300)
        Y = data.F / scale[data.labels]
        if self.model.centred_features:
            mean = _reduce_sum_1d(U, starts) / data.counts[:, None]
            dU = U - mean[data.labels]
            radius = xp.sqrt(_reduce_sum_1d(xp.sum(dU**2, axis=1), starts) / data.counts)
            Phi = self.model.features(dU / xp.where(radius > 0, radius, 1.0)[data.labels][:, None])
        else:  # affine: [u, 1] on the raw normalized coordinates
            Phi = self.model.features(U)
        return _SplitRows(U, Y, Phi, data.labels, data.counts)

    def _direction_bank(self, rows: _SplitRows, lead, V, M, width, keys) -> Array:
        """
        (k, D, d) unit candidate normals in ``u``: the split strategy's leading direction,
        its next eigenvectors, the leading covariance direction of the points, and
        node-keyed random directions to fill ``n_directions``.
        """
        D, d = self.settings.n_directions, self.d
        dirs = [lead]
        for j in range(2, min(d, 3) + 1):  # further metric eigenvectors span near-tied eigenspaces
            dirs.append(_unit(det_einsum("kij,kj->ki", M, V[:, :, -j]) * width))
        mean = rows.reduce(rows.U) / rows.counts[:, None]
        dU = rows.U - mean[rows.labels]
        cov = rows.cross(dU, dU)
        evals, evecs = xp.linalg.eigh(cov)
        # Points on a lower-dimensional set (e.g. unmixed samples): the fit is then ridge-
        # regularized along the missing directions, which is worth seeing in the counters.
        self.counters["rank_deficient"] += int(xp.sum(evals[:, 0] <= 1e-12 * xp.maximum(evals[:, -1], 1e-300)))
        dirs.append(evecs[:, :, -1])
        n_rand = max(0, D - len(dirs))
        if n_rand:
            r = counter_uniform(keys[:, None], PURPOSE_DIRECTION, xp.arange(2 * n_rand * d)[None, :])
            g = _normals(r[:, : n_rand * d], r[:, n_rand * d :]).reshape(-1, n_rand, d)
            dirs.extend(_unit(g[:, i]) for i in range(n_rand))
        return xp.stack(dirs[:D], axis=1)

    def _threshold_search(self, rows: _SplitRows, bank: Array) -> tuple[Array, Array, Array]:
        """
        B3: along every bank direction, evaluate ``n_bins - 1`` quantile cut positions
        from per-bin sums (``_SplitRows.cut_sums``: one data pass per direction), keep the
        ``n_screened`` best per node as a screen, then recompute their SSE directly
        (``_hard_sse``) and return the best ``(w, z)`` and its SSE. Only cuts between
        distinct scores that leave ``min_side`` points per side are candidates.
        """
        k, D, _ = bank.shape
        nb = self.settings.n_bins
        b = xp.arange(1, nb)
        n = rows.counts
        # Only nodes that can keep min_side points on both sides have candidates (sparse B1
        # nodes may not); every index below stays inside its own node's block.
        eligible = n >= 2 * self.min_side
        pos = xp.clip(xp.round(b[None, :] * n[:, None] / nb).astype(xp.int64), self.min_side, xp.maximum(n - self.min_side, self.min_side)[:, None])
        pos = xp.clip(pos, 1, xp.maximum(n - 1, 1)[:, None])
        screen = xp.full((k, D, nb - 1), xp.inf)
        z_all = xp.full((k, D, nb - 1), xp.nan)
        for j in range(D):
            s = plane_scores(rows.U, bank[rows.labels, j])
            order = sort_by_block_then_value(s, rows.labels)
            ss = s[order]
            at = rows.starts[:, None] + pos  # first right-side row of each candidate
            ok = eligible[:, None] & (pos >= self.min_side) & (n[:, None] - pos >= self.min_side) & (ss[at] > ss[at - 1])
            # Per node only (batched products over its own sorted rows), so nothing
            # depends on the other nodes of the batch.
            GL, BL, CL, GR, BR, CR = rows.cut_sums(order, pos)
            sse = xp.zeros((k, nb - 1))
            for Gs, Bs, Cs in ((GL, BL, CL), (GR, BR, CR)):
                beta = rows.solve(Gs, Bs)
                sse = sse + Cs - 2 * xp.sum(beta * Bs, axis=(2, 3)) + xp.sum(beta * det_matmul(Gs, beta), axis=(2, 3))
            screen[:, j] = xp.where(ok, sse, xp.inf)
            z_all[:, j] = ss[at]
        flat = screen.reshape(k, -1)
        n_keep = min(self.settings.n_screened, flat.shape[1])
        best_w, best_z = bank[:, 0], xp.full(k, xp.nan)
        best_sse = xp.full(k, xp.inf)
        cand = xp.argsort(flat, axis=1)[:, :n_keep]
        for t in range(n_keep):
            c = cand[:, t]
            j, i = c // (nb - 1), c % (nb - 1)
            w = bank[xp.arange(k), j]
            z = z_all[xp.arange(k), j, i]
            z = xp.where(xp.isfinite(flat[xp.arange(k), c]), z, xp.nan)
            sse, _ = _hard_sse(rows, w, xp.where(xp.isfinite(z), z, xp.inf), self.min_side)
            better = sse < best_sse
            best_w = xp.where(better[:, None], w, best_w)
            best_z = xp.where(better, z, best_z)
            best_sse = xp.minimum(sse, best_sse)
        return best_w, best_z, best_sse

    def _soft_refine(self, rows: _SplitRows, w0: Array, z0: Array, sse0: Array, progress) -> tuple[Array, Array]:
        """
        B4: polish each B3 plane by projected gradient descent on the consistent soft
        objective ``J(w, z) = min_B F`` (Section 33.3 of the design), with a fixed ridge, a
        fixed temperature ``T = soft_temperature * std(w^T u)``, floored weights
        ``W_R = delta + (1 - 2 delta) sigma(s)``, ``W_L = delta + (1 - 2 delta) sigma(-s)``,
        and frozen-model Armijo steps. A step is accepted only if its decrease exceeds the
        inner solves' estimated suboptimality ``tr(R^T H^-1 R) / n`` (``R`` from the data
        residual) plus rounding. This is numerically monitored descent, not a certified
        optimum. Each node's termination is counted once: ``soft_gate_saturated`` (gate
        derivative vanished), ``soft_gradient_converged`` (even a first-order step of
        one gate width could not beat the margin), ``soft_line_search_stalled`` (no trial
        qualified otherwise), ``soft_nonfinite`` (objective or gap not finite) or
        ``soft_iteration_limit``. The result is hardened (``z`` moved to the first
        right-side score) and kept only if its directly refit hard SSE beats the B3
        incumbent's; otherwise the incumbent is returned.
        """
        s_cfg = self.settings
        delta = s_cfg.soft_weight_floor
        n = rows.counts.astype(xp.float64)
        lab = rows.labels
        ok0 = xp.isfinite(z0)
        w, z = w0.copy(), xp.where(ok0, z0, 0.0)
        eps = float(np.finfo(np.float64).eps)
        s0 = plane_scores(rows.U, w[lab])
        # Centred variance (the raw-moment form cancels to zero or below for a narrow node
        # far from the origin, which made T ~ 1e-151 and the gates hard), floored at the
        # scores' rounding resolution.
        mean = rows.reduce(s0) / n
        var = rows.reduce((s0 - mean[lab]) ** 2) / n
        resolution = 64 * eps * xp.maximum(xp.abs(mean), 1e-300)
        T = s_cfg.soft_temperature * xp.sqrt(xp.maximum(var, resolution**2))
        active = ok0.copy()

        def gate(w_, z_):
            return _soft_gates(rows, w_, z_, T, delta)

        reasons = ("soft_nonfinite", "soft_gate_saturated", "soft_gradient_converged", "soft_line_search_stalled")
        stops = xp.zeros(len(reasons), dtype=xp.uint64)  # summed on the device, read once at the end
        for it in range(s_cfg.soft_iters):
            if not bool(xp.any(active)):
                break
            if progress is not None and it % 4 == 3:
                progress()
            J, g_w, g_z, rL, rR, pen, gap, sp_sum = _soft_objective(rows, w, z, T, delta)
            if IS_CUPY_BACKEND:  # fused step: w, z, active and stops updated in place, no host reads
                _soft_step(rows, w, z, g_w, g_z, J, gap, pen, rL, rR, sp_sum, T, n, delta, active, stops, 12)
                continue
            g_w = g_w - xp.sum(g_w * w, axis=1, keepdims=True) * w  # tangent to |w| = 1
            gn2 = xp.sum(g_w**2, axis=1) + g_z**2
            alpha = T / xp.sqrt(xp.maximum(gn2, 1e-300))  # a first trial moves the gate about one width
            margin = gap + 64 * eps * (xp.abs(J) + 1e-300)
            finite = xp.isfinite(J) & xp.isfinite(gap) & xp.isfinite(gn2)
            taken = xp.zeros_like(active)
            for _ in range(12):
                if not bool(xp.any(active & finite & ~taken)):
                    break  # every node has stepped; further halvings could not accept anything
                w_t = _unit(w - alpha[:, None] * g_w)
                z_t = z - alpha * g_z
                WLt, WRt, _ = gate(w_t, z_t)
                Ft = (rows.reduce(WLt * rL + WRt * rR) + pen) / n
                accept = active & finite & ~taken & (Ft <= J - 1e-4 * alpha * gn2) & (J - Ft > margin)
                w = xp.where(accept[:, None], w_t, w)
                z = xp.where(accept, z_t, z)
                taken = taken | accept
                alpha = alpha * 0.5
            stopped = active & ~taken
            saturated = finite & (sp_sum / n < 1e-6)
            flat = finite & ~saturated & (T * xp.sqrt(gn2) <= margin)
            stops += xp.stack([xp.sum(stopped & ~finite), xp.sum(stopped & saturated), xp.sum(stopped & flat),
                               xp.sum(stopped & finite & ~saturated & ~flat)]).astype(stops.dtype)  # fmt: skip
            active = active & taken
        for name, count in zip(reasons, _to_host(stops).tolist(), strict=True):
            self.counters[name] += int(count)
        self.counters["soft_iteration_limit"] += int(xp.sum(active))

        # Harden: move z to the first right-side score so the predicate reproduces the cut.
        s = plane_scores(rows.U, w[lab])
        first_right = sort_by_block_then_value(xp.where(s >= z[lab], s, xp.inf), lab)[rows.starts]
        z_h = s[first_right]
        z_h = xp.where(s[first_right] >= z, z_h, xp.nan)
        sse, _ = _hard_sse(rows, w, xp.where(xp.isfinite(z_h), z_h, xp.inf), self.min_side)
        better = ok0 & (sse < sse0)
        self.counters["soft_improved"] += int(xp.sum(better))
        return xp.where(better[:, None], w, w0), xp.where(better, z_h, z0)
