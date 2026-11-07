"""
Former home of the plotting helpers, kept so existing imports work; new code should
import from ``src.plotting``.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.plotting.profiling import plot_level_heatmap, plot_rank_time_breakdown
from src.plotting.surrogate import (  # noqa: F401
    _host_domain,
    plot_adaptive_clustering_results,
    plot_cluster_projections_for_wave,
    plot_error_histogram,
    plot_exact_hess_vs_estimated_hess,
    plot_near_centers,
    plot_wave_2d_errors,
    plot_wave_line_errors,
)


def plot_mpi_level_heatmap(
    results: dict[str, Any], img_fpath: Path, values_col: str = "good_clusters", title: str | None = None
) -> None:
    """Old signature of ``src.plotting.plot_level_heatmap``."""
    plt.close(plot_level_heatmap(results, img_fpath, value=values_col, title=title))


def plot_mpi_rank_time_breakdown(profiling_stats, img_fpath: Path, **_ignored) -> None:
    """Old signature of ``src.plotting.plot_rank_time_breakdown`` (styling arguments are ignored)."""
    plt.close(plot_rank_time_breakdown(profiling_stats, img_fpath))
