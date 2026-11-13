"""
AFINNI as a plan of the shared refinement loop: fit the local model on every cluster of a
batch, deploy the accepted ones with their trust regions, and augment and bisect the rest.

The batch boundaries (``batch_end``), the random seed per batch, the 50,000-point batch floor
and the memory estimate fix AFINNI's batch compositions and with them its trees: changing any
of them changes the results, which ``scripts/regression_check.py`` detects.
"""

import logging

import numpy as np

from src import IS_CUPY_BACKEND, to_host, xp
from src.components.local_models import ClusterFitData, deploy_leaves, output_columns
from src.components.sampling import augment_clusters, augmentation_targets
from src.components.splitting import SplitData
from src.config import TrainingConfig
from src.core.clusters import ClusterBatch, ParamSpec
from src.core.plan import DeviceBatch
from src.infra.io import save_generated_data
from src.infra.logging_utils import log_mem_stats
from src.numerics import reduce_sum_1d, repeats

logger = logging.getLogger(__name__)

_CHECK_CHUNK = 1 << 22  # rows per gather in the split check (bounds its temporary memory)


def _check_split(splitter, order, child_sizes, counts, labels) -> None:
    """
    A split must permute every row exactly once, keep each row inside its parent
    cluster (``labels`` is sorted, so ``labels[order]`` must equal it), and cut each
    cluster into two children whose sizes add up to the parent's.
    """
    n_rows = int(labels.shape[0])
    if order.shape != (n_rows,) or child_sizes.shape != (2 * len(counts),):
        raise ValueError(
            f"Split strategy {splitter.name!r} returned order {tuple(order.shape)} and "
            f"child sizes {tuple(child_sizes.shape)}; expected ({n_rows},) and "
            f"({2 * len(counts)},)."
        )
    if not bool(
        xp.all(child_sizes >= 0) & xp.all(child_sizes[0::2] + child_sizes[1::2] == counts)
    ):
        raise ValueError(
            f"Split strategy {splitter.name!r} returned child sizes that are negative "
            f"or do not add up to their cluster's size."
        )
    # ``order`` has n_rows entries, so it is a permutation iff it marks every row;
    # a boolean mark array (1 byte per row) avoids bincount's large temporaries.
    marked = xp.zeros(n_rows, dtype=xp.bool_)
    marked[order] = True
    same_cluster = all(
        bool(xp.all(labels[order[s : s + _CHECK_CHUNK]] == labels[s : s + _CHECK_CHUNK]))
        for s in range(0, n_rows, _CHECK_CHUNK)
    )
    if not (bool(xp.all(marked)) and same_cluster):
        raise ValueError(
            f"Split strategy {splitter.name!r} returned an order that drops, duplicates "
            f"or moves rows between clusters."
        )


class AFINNIPlan:
    """AFINNI's members of ``src.core.plan.Plan``; ``rank`` enters the per-batch seed and the logs."""

    label = "AFINNI"
    steal_settings: dict = {}
    # The batch cap never goes below 50,000 points (part of the batch boundaries, see above).
    batch_floor = 50_000
    keeps_fallback_leaves = False  # clusters still above tolerance at max_levels are unresolved

    def __init__(self, config: TrainingConfig, d: int, fd: int, rank: int):
        self.config, self.d, self.fd, self.rank = config, d, fd, rank
        self._grid_offset = 0
        self.mempool = None
        if IS_CUPY_BACKEND:
            import cupy as cp

            self.mempool = cp.get_default_memory_pool()

    def initial_queue(self, X: np.ndarray, F: np.ndarray, G: np.ndarray | None, labels: np.ndarray) -> ClusterBatch:
        """The initial clusters (one per label), points sorted by label."""
        sort_idx = np.argsort(labels)
        counts = np.unique(labels[sort_idx], return_counts=True)[1]
        logger.info(f"Initial clustering done. Number of clusters: {counts.shape[0]}.")
        return ClusterBatch(X[sort_idx], F[sort_idx], G[sort_idx] if G is not None else None, counts.astype(np.int64))

    def leaf_columns(self, model) -> dict[str, ParamSpec]:
        return output_columns(model, self.d, self.fd)

    def split_columns(self) -> None:
        return None

    def work_sizes(self, sizes: np.ndarray) -> np.ndarray:
        return sizes

    def bytes_per_point(self) -> int:
        """
        Approximate live storage during AFFINE_COV_HESS rejected-cluster handling: the base
        batch X/F/labels, plus twice (augmentation doubles a rejected cluster) the augmented
        X/F/labels, dx, residual work arrays, split scores and sort buffers, including the
        (N, fd, d) ``A_tangent[label]`` temporary of the NumPy Taylor residual
        (``taylor_residual_norms``). The CuPy kernel is fused, so that term is conservative
        on the GPU; it is not tightened, because a smaller estimate would change batch sizes, and
        with them the results.
        """
        d, fd = self.d, self.fd
        base_batch_bytes_per_point = (d + fd + 1) * 8
        split_augmented_bytes_per_point = (
            3 * d  # X_aug, dx, selected/source point work
            + fd * d
            + 3 * fd
            + 8  # labels, masks, scalar work arrays, sort/scores
        ) * 8
        augmentation_factor = 2
        return base_batch_bytes_per_point + augmentation_factor * split_augmented_bytes_per_point

    def constraint_rows(self, queue: ClusterBatch) -> None:
        return None

    def batch_end(self, cumsum: np.ndarray, start: int, index: int, batch_size: int) -> int:
        """
        Batches end at the first cluster crossing the grid (index + 1) * batch_size of
        cumulative points, so a batch can exceed batch_size by its last cluster. A cluster
        that would start a batch beyond the grid is processed alone and the grid is
        re-anchored after it; the anchor resets with each level's first batch.
        """
        if index == 0:
            self._grid_offset = 0
        target_aug_count = (index + 1) * batch_size + self._grid_offset
        end = min(int(np.searchsorted(cumsum, target_aug_count)) + 1, len(cumsum))
        if end <= start:
            self._grid_offset = int(cumsum[start]) - (index + 1) * batch_size
            end = start + 1
        return end

    def process_batch(self, batch: DeviceBatch, *, components, evaluator, accepted, splits, next_level,
                      timer) -> dict[str, int]:  # fmt: skip
        """
        Fit the local model on every cluster, append the accepted ones (with their trust
        regions) to ``accepted``, and augment and split the rejected ones into ``next_level``.
        Each stage releases what the next does not need: the batch's points are dropped before
        augmentation and splitting, which hold the batch's peak memory.
        """
        depth = batch.queue.depth[batch.cut]
        X, F, G = batch.take()
        n_pts = int(X.shape[0])
        # 1. Fit and screen every cluster; ``fit`` then holds the only references to the points.
        fit, params, is_good, E_abs, E_rel = self._fit_and_accept(X, F, G, batch, components, evaluator, timer)
        del X, F, G
        if self.config.log_mem_stats:
            log_mem_stats(f" R{self.rank:02} L{batch.level:02} B{batch.index:02}/{batch.n_batches:02}: After Error Metrics")

        # 2. Deploy the accepted clusters as leaves; a cluster its finalized model fails is rejected.
        n_good = self._deploy(fit, params, is_good, E_abs, E_rel, components, accepted, timer)

        with timer("time_filter"):
            bad_idx = xp.nonzero(~is_good)[0]
            n_bad = int(bad_idx.shape[0])
        generated = 0
        if n_bad > 0:
            # 3. Augment the rejected clusters (this releases the batch's points), then
            # 4. bisect each into two children of the next level.
            data, generated = self._augment_rejected(fit, is_good, bad_idx, batch, components, evaluator, timer)
            self._split_rejected(data, bad_idx, depth, components.split_strategy, next_level, timer)
            with timer("time_gc"):
                del data
                if self.mempool is not None:
                    self.mempool.free_all_blocks()

        with timer("time_gc"):
            del fit
            if self.mempool is not None:
                self.mempool.free_all_blocks()
        return {
            "accepted": n_good, "rejected": n_bad, "generated": generated,
            "fallback_clusters": 0, "fallback_pts": 0, "n_pts": n_pts,
        }  # fmt: skip

    def _fit_and_accept(self, X, F, G, batch: DeviceBatch, components, evaluator, timer):
        """
        Fit the local model on every cluster at its points (with ``f`` and, if the mode needs
        it, the gradient at the centroid) and screen it with the acceptance test. Returns the
        fit data, the model's parameters and the acceptance result ``(is_good, E_abs, E_rel)``.
        """
        model, acceptance = components.local_model, components.acceptance
        sizes, starts = batch.sizes, batch.starts
        labels = repeats(xp.arange(len(sizes)), sizes)
        with timer("time_error_calc"):
            mus = reduce_sum_1d(X, starts) / sizes[:, None]
            f_mus, g_mus = evaluator(mus, need_grad=self.config.requires_grad_centroids)
            fit = ClusterFitData(X=X, F=F, G=G, labels=labels, starts=starts, sizes=sizes,
                                 centroids=mus, f_c=f_mus, g_c=g_mus)  # fmt: skip
            params = model.fit(fit)
            F_hat = model.evaluate({**params, "centroids": mus, "f": f_mus}, X, labels)
            is_good, E_abs, E_rel = acceptance(F, F_hat, starts, sizes)
        return fit, params, is_good, E_abs, E_rel

    def _deploy(self, fit: ClusterFitData, params, is_good, E_abs, E_rel, components, accepted, timer) -> int:
        """
        Append the accepted clusters (finalized model, trust region) to ``accepted`` and return
        how many there are. A cluster whose finalized model fails the re-test is marked
        rejected in ``is_good`` (in place), so it is split instead.
        """
        with timer("time_filter"):
            good_idx = xp.nonzero(is_good)[0]
            n_good = int(good_idx.shape[0])
        if n_good == 0:
            return 0
        with timer("time_good_Q"):
            leaves = deploy_leaves(components.local_model, components.trust_region, components.acceptance,
                                   fit, good_idx, params, E_abs, E_rel)  # fmt: skip
            if int(leaves.failed.shape[0]) > 0:
                is_good[leaves.failed] = False
                n_good -= int(leaves.failed.shape[0])
        with timer("time_good_d2h"):
            accepted.append(leaves.rows)
        return n_good

    def _augment_rejected(self, fit: ClusterFitData, is_good, bad_idx, batch: DeviceBatch, components, evaluator,
                          timer) -> tuple[SplitData, int]:  # fmt: skip
        """
        Grow every rejected cluster to ``augmentation_targets`` points and return them as the
        split's input with the number of new points. Releases the batch's points from ``fit``
        first: from here on only the rejected rows are needed.
        """
        config = self.config
        with timer("time_augment"):
            bad_mask = (~is_good)[fit.labels]
            X_bad, F_bad = fit.X[bad_mask], fit.F[bad_mask]
            G_bad = fit.G[bad_mask] if fit.G is not None else None
            fit.X = fit.F = fit.G = None
            del bad_mask
            counts = fit.sizes[bad_idx]
            targets = augmentation_targets(counts, config.min_pts_per_cluster)

            n_in, n_target = int(xp.sum(counts)), int(xp.sum(targets))
            logger.info(
                f"R{self.rank:02} L{batch.level:02} B{batch.index + 1}: rejected-cluster augmentation plan | "
                f"bad_clusters={int(bad_idx.shape[0]):,} input_pts={n_in:,} "
                f"bad_count[min/mean/max]={int(xp.min(counts)):,}/{float(xp.mean(counts)):,.1f}/{int(xp.max(counts)):,} "
                f"target_pts={n_target:,} synthetic_pts={n_target - n_in:,} expansion={n_target / max(n_in, 1):.2f}x"
            )  # fmt: skip

            # One stream per (level, batch, rank): part of what fixes AFINNI's trees.
            rng = xp.random.default_rng(config.seed + batch.level + batch.index + self.rank * 1000)
            aug = augment_clusters(X_bad, F_bad, G_bad, counts, targets, components.sampler, evaluator, rng,
                                   centroids=fit.centroids[bad_idx])  # fmt: skip
            generated = 0
            if aug.X_new is not None:
                generated = int(aug.X_new.shape[0])
                if config.save_all_gen_training_data:
                    save_generated_data(aug.X_new, config.generated_data_name(self.d))
            data = SplitData(
                X=aug.X, F=aug.F, G=aug.G, labels=aug.labels, counts=targets, centroids=fit.centroids[bad_idx],
                f_c=fit.f_c[bad_idx], g_c=fit.g_c[bad_idx] if fit.g_c is not None else None,
                hessian_fn=evaluator.hessian_fn,
            )  # fmt: skip
            # The augmented arrays hold copies of the rejected rows.
            del aug, X_bad, F_bad, G_bad
            if self.mempool is not None:
                # Hand cached blocks back before the split, the batch's largest allocation
                # burst (a few cudaFree calls per batch).
                self.mempool.free_all_blocks()
        return data, generated

    def _split_rejected(self, data: SplitData, bad_idx, depth: np.ndarray, splitter, next_level, timer) -> None:
        """
        Cut each rejected cluster in two along its split direction (its points sorted by their
        projection, ``child_sizes`` the size of each half) and append the children, one level
        deeper, to ``next_level``.
        """
        with timer("time_bad_split"):
            order, child_sizes = splitter.split(data)
            _check_split(splitter, order, child_sizes, data.counts, data.labels)
        with timer("time_bad_d2h"):
            G = data.G[order] if self.config.requires_grad_everywhere and data.G is not None else None
            next_level.append(data.X[order], data.F[order], G, child_sizes,
                              depth=np.repeat(depth[to_host(bad_idx)] + 1, 2))  # fmt: skip

    def profile(self) -> dict:
        return {}

    def merge_results(self, merged: dict, profile: dict, per_rank: list[dict]) -> None:
        return None
