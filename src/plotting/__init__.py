"""
Plotting for AFINNI. Import plots from here; add new ones to the module that fits:

- ``src.plotting.profiling``: training profiles (time per phase and rank, level
  heatmaps, forest growth, timelines, load balance and work stealing, dashboard).
- ``src.plotting.surrogate``: trained surrogates and their errors.
- ``src.plotting.style``: the shared style (``afinni_style``) and ``finish`` (save/close).

Conventions for new plots: accept training outputs directly (see
``profiling.profile_of``), draw inside ``afinni_style()``, take an optional ``path``,
and return the figure, so plots work in notebooks, scripts and tests alike. Put the
data preparation in a ``*_frame`` helper when it is useful on its own.
"""

from src.plotting.profiling import (
    events_frame,
    level_stats_frame,
    phase_breakdown_frame,
    plot_level_heatmap,
    plot_level_progress,
    plot_load_over_time,
    plot_profile_dashboard,
    plot_rank_time_breakdown,
    plot_steal_matrix,
    plot_timeline,
    profile_of,
    steals_frame,
)
from src.plotting.style import afinni_style, finish
from src.plotting.surrogate import (
    plot_adaptive_clustering_results,
    plot_cluster_projections_for_wave,
    plot_error_histogram,
    plot_exact_hess_vs_estimated_hess,
    plot_near_centers,
    plot_wave_2d_errors,
    plot_wave_line_errors,
)

__all__ = [
    "afinni_style",
    "events_frame",
    "finish",
    "level_stats_frame",
    "phase_breakdown_frame",
    "plot_adaptive_clustering_results",
    "plot_cluster_projections_for_wave",
    "plot_error_histogram",
    "plot_exact_hess_vs_estimated_hess",
    "plot_level_heatmap",
    "plot_level_progress",
    "plot_load_over_time",
    "plot_near_centers",
    "plot_profile_dashboard",
    "plot_rank_time_breakdown",
    "plot_steal_matrix",
    "plot_timeline",
    "plot_wave_2d_errors",
    "plot_wave_line_errors",
    "profile_of",
    "steals_frame",
]
