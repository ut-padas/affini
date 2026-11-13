"""
AFINNI as a plan of the shared refinement loop: fit the local model on every cluster of a
batch, deploy the accepted ones with their trust regions, and augment and bisect the rest.

The batch boundaries (``batch_end``), the random seed per batch, the 50,000-point batch floor
and the memory estimate are AFINNI's original arithmetic: changing any of them changes batch
compositions and with them the trees, so ``scripts/regression_check.py`` must stay
bit-identical across refactors of this module.
"""

import logging

import numpy as np

from src import IS_CUPY_BACKEND, _to_host, xp
from src.components.local_models import ClusterFitData, deploy_leaves, output_columns
from src.components.sampling import augment_clusters, augmentation_targets
from src.components.splitting import SplitData
from src.config import ClusteringConfig
from src.core.clusters import ClusterBatch, ParamSpec
from src.core.plan import DeviceBatch
from src.infra.io import save_generated_data
from src.infra.logging_utils import log_mem_stats
from src.numerics import _reduce_sum_1d, repeats

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
    # The batch cap never goes below 50,000 points, so batches (and seeds) stay bit-identical.
    batch_floor = 50_000
    keeps_fallback_leaves = False  # clusters still above tolerance at max_levels are unresolved

    def __init__(self, config: ClusteringConfig, d: int, fd: int, rank: int):
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
        on the GPU; it is kept because a smaller estimate would change batch sizes, and
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
        """
        config, d, rank, mempool = self.config, self.d, self.rank, self.mempool
        level, b, num_batches = batch.level, batch.index, batch.n_batches
        sizes, starts = batch.sizes, batch.starts
        depth_cpu = batch.queue.depth[batch.cut]
        model, splitter, acceptance = components.local_model, components.split_strategy, components.acceptance
        trust_region, sampler = components.trust_region, components.sampler
        generated = 0
        aug_sizes = sizes
        K_batch = len(sizes)

        X_batch, F_batch, G_batch = batch.take()  # this method now holds the only references

        aug_starts = starts
        aug_ids = repeats(xp.arange(K_batch), sizes)

        with timer("time_error_calc"):
            mus = _reduce_sum_1d(X_batch, aug_starts) / aug_sizes[:, None]

            f_mus, g_mus = evaluator(
                mus, need_grad=config.requires_grad_centroids
            )

            # Fit the local model on every cluster, evaluate it at the
            # cluster's points, and accept the clusters within tolerance.
            fit_data = ClusterFitData(
                X=X_batch,
                F=F_batch,
                G=G_batch,
                labels=aug_ids,
                starts=aug_starts,
                sizes=aug_sizes,
                centroids=mus,
                f_c=f_mus,
                g_c=g_mus,
            )
            surrogate_params = model.fit(fit_data)
            F_hat = model.evaluate(
                {**surrogate_params, "centroids": mus, "f": f_mus},
                X_batch,
                aug_ids,
            )
            is_good, E_abs, E_rel = acceptance(F_batch, F_hat, aug_starts, aug_sizes)
            del F_hat, fit_data

        if config.log_mem_stats:
            log_mem_stats(
                f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Error Metrics"
            )

        # 5. Handle Accepted ("Good") Clusters
        with timer("time_filter"):
            good_idx_local = xp.nonzero(is_good)[0]
            K_good_batch = int(good_idx_local.shape[0])

        if K_good_batch > 0:
            with timer("time_good_Q"):
                leaves = deploy_leaves(
                    model, trust_region, acceptance,
                    ClusterFitData(
                        X=X_batch, F=F_batch, G=G_batch, labels=aug_ids, starts=aug_starts,
                        sizes=aug_sizes, centroids=mus, f_c=f_mus, g_c=g_mus,
                    ),
                    good_idx_local, surrogate_params, E_abs, E_rel,
                )  # fmt: skip
                if int(leaves.failed.shape[0]) > 0:
                    # Clusters the finalized model fails are split instead.
                    is_good[leaves.failed] = False
                    K_good_batch -= int(leaves.failed.shape[0])

            with timer("time_good_d2h"):
                accepted.append(leaves.rows)
                del leaves

        # 6. Handle Rejected ("Bad") Clusters -> Split
        with timer("time_filter"):
            bad_idx_local = xp.nonzero(~is_good)[0]
            n_bad = int(bad_idx_local.shape[0])
            n_batch_pts = int(X_batch.shape[0])

        if n_bad > 0:
            with timer("time_augment"):
                bad_mask = (~is_good)[aug_ids]
                X_bad = X_batch[bad_mask]
                F_bad = F_batch[bad_mask]
                G_bad = G_batch[bad_mask] if G_batch is not None else None
                # The whole batch is not needed past this point: release it before
                # augmentation and splitting, which hold the batch's peak memory.
                X_batch = F_batch = G_batch = bad_mask = None
                bad_counts = aug_sizes[bad_idx_local]

                # --- Augment small rejected clusters before splitting ---
                target_bad_counts = augmentation_targets(
                    bad_counts, config.min_pts_per_cluster
                )

                total_target_pts = int(xp.sum(target_bad_counts))
                total_bad_input_pts = int(xp.sum(bad_counts))
                planned_synth_pts = total_target_pts - total_bad_input_pts
                bad_count_min = int(xp.min(bad_counts))
                bad_count_mean = float(xp.mean(bad_counts))
                bad_count_max = int(xp.max(bad_counts))
                expansion = total_target_pts / max(total_bad_input_pts, 1)

                logger.info(
                    f"R{rank:02} L{level:02} B{b + 1}: rejected-cluster augmentation plan | "
                    f"bad_clusters={n_bad:,} "
                    f"input_pts={total_bad_input_pts:,} "
                    f"bad_count[min/mean/max]={bad_count_min:,}/{bad_count_mean:,.1f}/{bad_count_max:,} "
                    f"target_pts={total_target_pts:,} "
                    f"synthetic_pts={planned_synth_pts:,} "
                    f"expansion={expansion:.2f}x"
                )

                rng = xp.random.default_rng(config.seed + level + b + rank * 1000)
                aug = augment_clusters(
                    X_bad,
                    F_bad,
                    G_bad,
                    bad_counts,
                    target_bad_counts,
                    sampler,
                    evaluator,
                    rng,
                    centroids=mus[bad_idx_local],
                )
                X_bad_aug, F_bad_aug, G_bad_aug = aug.X, aug.F, aug.G
                aug_dense_labels_bad = aug.labels
                if aug.X_new is not None:
                    generated += int(aug.X_new.shape[0])
                    if config.save_all_gen_training_data:
                        save_generated_data(
                            aug.X_new,
                            config.generated_data_name(d),
                        )
                # The augmented arrays hold copies of the rejected rows.
                del aug, X_bad, F_bad, G_bad
                if mempool is not None:
                    # Hand cached blocks back before the split, the batch's largest
                    # allocation burst (a few cudaFree calls per batch).
                    mempool.free_all_blocks()

            with timer("time_bad_split"):
                # Cut each rejected cluster in two along its split direction:
                # its points are sorted by their projection onto that direction,
                # and child_sizes gives the size of each half.
                sort_order, child_sizes = splitter.split(
                    SplitData(
                        X=X_bad_aug,
                        F=F_bad_aug,
                        G=G_bad_aug,
                        labels=aug_dense_labels_bad,
                        counts=target_bad_counts,
                        centroids=mus[bad_idx_local],
                        f_c=f_mus[bad_idx_local],
                        g_c=g_mus[bad_idx_local] if g_mus is not None else None,
                        hessian_fn=evaluator.hessian_fn,
                    )
                )
                _check_split(
                    splitter, sort_order, child_sizes, target_bad_counts,
                    aug_dense_labels_bad,
                )  # fmt: skip

            with timer("time_bad_d2h"):
                # Children: each cluster cut along its split direction.
                next_level.append(
                    X_bad_aug[sort_order],
                    F_bad_aug[sort_order],
                    G_bad_aug[sort_order]
                    if config.requires_grad_everywhere and G_bad_aug is not None
                    else None,
                    child_sizes,
                    depth=np.repeat(depth_cpu[_to_host(bad_idx_local)] + 1, 2),
                )

            with timer("time_gc"):
                del (
                    X_bad_aug,
                    F_bad_aug,
                    G_bad_aug,
                    sort_order,
                )
                if mempool is not None:
                    mempool.free_all_blocks()

        with timer("time_gc"):
            del X_batch, F_batch, G_batch
            if mempool is not None:
                mempool.free_all_blocks()
        return {
            "accepted": K_good_batch, "rejected": n_bad, "generated": generated,
            "fallback_clusters": 0, "fallback_pts": 0, "n_pts": n_batch_pts,
        }  # fmt: skip

    def profile(self) -> dict:
        return {}

    def merge_results(self, merged: dict, profile: dict, per_rank: list[dict]) -> None:
        return None
