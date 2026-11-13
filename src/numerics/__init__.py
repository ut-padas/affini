"""
Batched numerical building blocks shared by the local models, split strategies,
acceptance criteria and trust regions.

Every function works on a batch of clusters stored contiguously (``labels`` gives
each row's cluster, ``starts`` the first row of each cluster) and runs as a few
vectorized passes, using custom CUDA kernels on the CuPy backend and NumPy
fallbacks otherwise. Nothing here knows about the refinement loop.

Submodules: ``segments`` (segmented sums and products), ``linalg`` (small batched solves and
products), ``stats`` (per-cluster statistics and projections).
"""

from src.numerics.linalg import (
    TINY_EPS,
    det_einsum,
    det_matmul,
    lsq_hessian_in_eigenbasis,
    solve_regularized_covariance_inplace,
    spd_factor,
    spd_solve,
    spd_solve_factored,
)
from src.numerics.segments import (
    _reduce_max_1d,
    _reduce_sum_1d,
    count_groups,
    get_n_smallest_per_block,
    grouped_cross,
    repeats,
    segment_cross,
    segment_layout,
    segment_sum,
    sort_by_block_then_value,
)
from src.numerics.stats import (
    _symmetric_point_covariance,
    centroid_distances,
    convert_to_2d_array,
    convert_to_3d_array,
    run_fused_cross_cov,
    run_fused_mahalanobis_projection,
    run_fused_split_projection,
    run_fused_stats,
    taylor_residual_norms,
)

__all__ = ["TINY_EPS", "_reduce_max_1d", "_reduce_sum_1d", "_symmetric_point_covariance", "centroid_distances", "convert_to_2d_array", "convert_to_3d_array", "count_groups", "det_einsum", "det_matmul", "get_n_smallest_per_block", "grouped_cross", "lsq_hessian_in_eigenbasis", "repeats", "run_fused_cross_cov", "run_fused_mahalanobis_projection", "run_fused_split_projection", "run_fused_stats", "segment_cross", "segment_layout", "segment_sum", "solve_regularized_covariance_inplace", "sort_by_block_then_value", "spd_factor", "spd_solve", "spd_solve_factored", "taylor_residual_norms"]
