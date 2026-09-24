from src import IS_CUPY_BACKEND, Array, xp, _to_host
from typing import Callable
from src.adaptive_clustering_utils import \
    get_n_smallest_per_block, \
    run_fused_stats, \
    run_fused_cross_cov, \
    solve_regularized_covariance_inplace
import matplotlib.pyplot as plt
import numpy as np


def compute_and_plot_relative_errors_per_row(
    exact_fn: Callable,
    eval_pts: Array,
    approx: Array,
):

    exact = exact_fn(eval_pts)
    rel_err = xp.linalg.norm(exact - approx, axis=1) / xp.linalg.norm(exact, axis=1)
    plt.figure()
    minimum = _to_host(min(rel_err))
    maximum = _to_host(max(rel_err))
    width = 0.05
    bins = np.arange(minimum, maximum + width, width)
    plt.hist(_to_host(rel_err), bins=bins, edgecolor='black')
    plt.savefig("rel_err_histogram.png", dpi=200)
