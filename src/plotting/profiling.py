"""
Plots of training profiles: where time goes per rank, how work moves between ranks,
and how the forest grows level by level.

Every function accepts what training returns: ``AFINNISurrogate.profiling_stats``,
the ``adaptive_clustering`` result dict, a surrogate, or a dict of event streams
from ``src.profiling.read_event_streams`` (timeline plots only). Each returns the
figure and saves it when ``path`` is given.

Data requirements: breakdown plots need ``PROFILE_TIMING=True`` (otherwise phase
totals are zero); timeline, load and steal plots need ``PROFILE_EVENTS=True`` or
``PROFILE_DIR``; level plots always work. The ``*_frame`` helpers return the tidy
``pandas`` tables behind the plots, for custom analysis.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.plotting.style import afinni_style, finish, fmt_count
from src.profiling import PHASE_BY_KEY, group_colors, phase_groups

UNTIMED = "Other (untimed)"
_UNTIMED_COLOR = "#e5e5e5"


# --- Inputs -----------------------------------------------------------------------


def profile_of(source: Any) -> dict[str, Any]:
    """The consolidated ``profiling_stats`` dict from any supported training output."""
    if hasattr(source, "profiling_stats"):
        source = source.profiling_stats
    if isinstance(source, Mapping) and "profiling_stats" in source:
        source = source["profiling_stats"]
    if not isinstance(source, Mapping):
        raise TypeError(f"Expected profiling stats, a result dict or a surrogate; got {type(source)}.")
    return dict(source)


def _per_rank(stats: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Per-rank stats; a single rank's stats (no ``per_rank``) count as one rank."""
    return list(stats.get("per_rank") or [stats])


# --- Tidy tables --------------------------------------------------------------------


def phase_breakdown_frame(source: Any) -> pd.DataFrame:
    """Seconds per rank (rows) and display group (columns), plus untimed and total time."""
    stats = profile_of(source)
    groups = phase_groups()
    rows = []
    for r, ps in enumerate(_per_rank(stats)):
        row = {
            group: sum(ps.get(k, 0.0) for k in keys if not PHASE_BY_KEY[k].nested)
            for group, keys in groups.items()
        }
        total = float(ps.get("time_total", 0.0))
        row[UNTIMED] = max(0.0, total - sum(row.values()))
        row["total"] = total
        rows.append(pd.Series(row, name=r))
    df = pd.DataFrame(rows)
    df.index.name = "rank"
    return df


def level_stats_frame(source: Any) -> pd.DataFrame:
    """One row per (rank, level) from ``level_stats``."""
    stats = profile_of(source)
    entries = stats.get("level_stats") or [e for ps in _per_rank(stats) for e in ps.get("level_stats", [])]
    return pd.DataFrame(entries)


def events_frame(source: Any) -> pd.DataFrame:
    """Every recorded event (all ranks), sorted by time; empty if events were not recorded."""
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, Mapping) and source and all(isinstance(k, int) for k in source):
        events = [e for evs in source.values() for e in evs]  # read_event_streams output
    else:
        events = [e for ps in _per_rank(profile_of(source)) for e in ps.get("events", [])]
    df = pd.DataFrame(events)
    return df.sort_values("t", kind="stable").reset_index(drop=True) if len(df) else df


def steals_frame(source: Any) -> pd.DataFrame:
    """Granted steals: time, victim, thief, points and clusters moved."""
    ev = events_frame(source)
    if ev.empty or "steal_grant" not in set(ev["kind"]):
        return pd.DataFrame(columns=["t", "victim", "thief", "n_pts", "n_clusters", "mid_level"])
    # The granting rank is the victim (a "victim" column from requests is unrelated).
    g = ev[ev["kind"] == "steal_grant"].drop(columns=["victim"], errors="ignore")
    g = g.rename(columns={"rank": "victim"})
    return g[["t", "victim", "thief", "n_pts", "n_clusters", "mid_level"]].astype(
        {"victim": int, "thief": int, "n_pts": int, "n_clusters": int}
    )


def _require(df: pd.DataFrame, what: str) -> None:
    if df.empty:
        raise ValueError(f"No {what} recorded; train with PROFILE_EVENTS=True (or PROFILE_DIR).")


# --- Plots ---------------------------------------------------------------------------


def plot_rank_time_breakdown(
    source: Any,
    path: str | Path | None = None,
    normalize: bool = True,
    annotate_min: float = 0.04,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Stacked bars of each rank's time per phase group (fractions when ``normalize``),
    with each rank's total training time below.

    Time not covered by a timed phase is shown as "Other (untimed)", so bars are
    honest when only part of the loop is instrumented.
    """
    df = phase_breakdown_frame(source)
    if df.drop(columns=["total", UNTIMED]).to_numpy().sum() == 0:
        raise ValueError("All phase times are zero; train with PROFILE_TIMING=True.")
    parts = df.drop(columns=["total"])
    parts = parts.loc[:, parts.sum() > 0]
    shown = parts.div(parts.sum(axis=1), axis=0) if normalize else parts
    colors = {**group_colors(), UNTIMED: _UNTIMED_COLOR}
    n = len(df)
    with afinni_style():
        fig, (ax, ax_tot) = plt.subplots(
            2, 1, sharex=True, figsize=figsize or (max(9.0, 0.6 * n + 4), 7.5),
            gridspec_kw={"height_ratios": [4, 1]},
        )  # fmt: skip
        bottom = np.zeros(n)
        for col in shown.columns:
            vals = shown[col].to_numpy()
            ax.bar(df.index, vals, bottom=bottom, color=colors.get(col), label=col, width=0.85, edgecolor="white", linewidth=0.5)
            if normalize:
                for x, (v, b) in enumerate(zip(vals, bottom, strict=True)):
                    if v >= annotate_min:
                        ax.text(df.index[x], b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8)
            bottom += vals
        ax.set_ylabel("Fraction of time" if normalize else "Time (s)")
        if normalize:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=min(4, len(shown.columns)))

        ax_tot.plot(df.index, df["total"], marker="o", color="#333333")
        for x, t in zip(df.index, df["total"], strict=True):
            ax_tot.annotate(f"{t:.3g}s", (x, t), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax_tot.set_ylabel("Total (s)")
        ax_tot.set_xlabel("MPI rank")
        ax_tot.set_xticks(df.index)
        lo, hi = df["total"].min(), df["total"].max()
        pad = max(0.05 * hi, 1e-9)
        ax_tot.set_ylim(max(0.0, lo - 3 * pad), hi + 3 * pad)
    return finish(fig, path)


_VALUE_LABELS = {
    "good_clusters": "Accepted clusters",
    "clusters_start": "Clusters at start of level",
    "clusters_level_end": "Clusters queued for next level",
    "num_pts_next_level": "Points queued for next level",
    "generated_pts": "Generated points",
    "size_mean": "Mean cluster size",
    "size_max": "Max cluster size",
    "runtime_sec": "Level runtime (s)",
}


def plot_level_heatmap(
    source: Any,
    path: str | Path | None = None,
    value: str = "clusters_start",
    cmap: str = "Reds",
    annotate: bool | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap of one ``level_stats`` field per level (rows) and rank (columns); blank = no work."""
    df = level_stats_frame(source)
    if df.empty or value not in df:
        raise ValueError(f"No level statistics with field {value!r}.")
    grid = df.pivot_table(index="level", columns="rank", values=value, aggfunc="sum")
    grid = grid.reindex(columns=range(int(df["rank"].max()) + 1))
    data = grid.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(np.where(data == 0, np.nan, data))
    n_lvl, n_rank = data.shape
    annotate = (n_lvl * n_rank <= 800) if annotate is None else annotate
    stats = profile_of(source)
    with afinni_style():
        fig, ax = plt.subplots(figsize=(max(5.0, 0.62 * n_rank + 2.5), max(4.0, 0.32 * n_lvl + 1.5)))
        im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_xticks(range(n_rank))
        ax.set_yticks(range(n_lvl), [f"L{lvl}" for lvl in grid.index])
        ax.set_xlabel("MPI rank")
        ax.set_ylabel("Level")
        if annotate:
            vmax = np.nanmax(masked.filled(np.nan)) if masked.count() else 1.0
            for (i, j), v in np.ndenumerate(data):
                if np.isfinite(v) and v != 0:
                    label = f"{v:.2g}" if value == "runtime_sec" else fmt_count(v)
                    ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white" if v > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label=_VALUE_LABELS.get(value, value), shrink=0.8)
        default = f"{_VALUE_LABELS.get(value, value)} by level and rank"
        extra = []
        if stats.get("time_total"):
            extra.append(f"train {stats['time_total']:.3g}s")
        if stats.get("num_final_clusters"):
            extra.append(f"{stats['num_final_clusters']:,} leaves")
        ax.set_title(title or (default + (f"\n{', '.join(extra)}" if extra else "")))
    return finish(fig, path)


def plot_level_progress(source: Any, path: str | Path | None = None) -> plt.Figure:
    """Per level (all ranks): accepted vs. split clusters, and points processed / generated."""
    df = level_stats_frame(source)
    if df.empty:
        raise ValueError("No level statistics recorded.")
    if "generated_pts" not in df:  # profiles recorded before this field existed
        df["generated_pts"] = 0
    agg = df.groupby("level").agg(
        start=("clusters_start", "sum"), accepted=("good_clusters", "sum"),
        next_pts=("num_pts_next_level", "sum"), generated=("generated_pts", "sum"),
    )  # fmt: skip
    agg["split"] = agg["start"] - agg["accepted"]
    with afinni_style():
        fig, (ax, ax_pts) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        ax.bar(agg.index, agg["accepted"], color="#59a14f", label="accepted (leaves)")
        ax.bar(agg.index, agg["split"], bottom=agg["accepted"], color="#e15759", label="rejected (split)")
        ax.set_ylabel("Clusters")
        ax.set_yscale("log" if agg["start"].max() > 50 * max(1, agg["start"].min()) else "linear")
        ax.legend()
        ax.set_title("Forest growth by level")
        ax_pts.plot(agg.index, agg["next_pts"], marker="o", label="points queued for next level")
        ax_pts.plot(agg.index, agg["generated"], marker="s", label="generated points")
        ax_pts.set_ylabel("Points")
        ax_pts.set_xlabel("Level")
        ax_pts.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_count(v)))
        ax_pts.legend()
    return finish(fig, path)


def plot_timeline(
    source: Any, path: str | Path | None = None, show_steals: bool = True, min_dur: float = 0.0
) -> plt.Figure:
    """
    Gantt chart: one lane per rank, coloured by phase group over time, with arrows
    from victim to thief for every granted steal. Nested phases are not drawn.
    """
    ev = events_frame(source)
    _require(ev, "events")
    phases = ev[(ev["kind"] == "phase")].copy()
    phases = phases[~phases["phase"].map(lambda k: PHASE_BY_KEY[k].nested if k in PHASE_BY_KEY else False)]
    phases = phases[phases["dur"] >= min_dur]
    colors = group_colors()
    ranks = sorted(ev["rank"].unique())
    with afinni_style():
        fig, ax = plt.subplots(figsize=(11, max(2.5, 0.45 * len(ranks) + 1.5)))
        used = set()
        for (rank, key), grp in phases.groupby(["rank", "phase"]):
            group = PHASE_BY_KEY[key].group if key in PHASE_BY_KEY else key
            used.add(group)
            ax.broken_barh(list(zip(grp["t"], grp["dur"], strict=True)), (rank - 0.4, 0.8), color=colors.get(group, "#999999"), linewidth=0)
        if show_steals:
            for s in steals_frame(ev).itertuples():
                ax.annotate("", xy=(s.t, s.thief), xytext=(s.t, s.victim), arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
        ax.set_yticks(ranks)
        ax.set_ylabel("MPI rank")
        ax.set_xlabel("Time since start (s)")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        handles = [Patch(color=colors[g], label=g) for g in colors if g in used]
        if show_steals and not steals_frame(ev).empty:
            handles.append(plt.Line2D([], [], color="black", marker=">", lw=0.8, label="steal (victim → thief)"))
        fig.legend(handles=handles, loc="outside upper center", ncol=min(5, len(handles)))
    return finish(fig, path)


def plot_load_over_time(source: Any, path: str | Path | None = None) -> plt.Figure:
    """Queued points per rank over time (step plot), with steal grants marked on the victim's curve."""
    ev = events_frame(source)
    _require(ev, "events")
    loads = ev[ev["kind"].isin(["load", "level_start"])].dropna(subset=["n_pts"])
    _require(loads, "load samples")
    with afinni_style():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        cmap = plt.get_cmap("tab20")
        for i, (rank, grp) in enumerate(loads.groupby("rank")):
            ax.step(grp["t"], grp["n_pts"], where="post", color=cmap(i % 20), label=f"rank {rank}")
        steals = steals_frame(ev)
        if not steals.empty:
            ax.scatter(steals["t"], np.zeros(len(steals)), marker="v", color="black", s=18, label="steal granted", zorder=3)
        ax.set_xlabel("Time since start (s)")
        ax.set_ylabel("Queued points")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_count(v)))
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1 + (loads["rank"].nunique() + 1) // 16)
        ax.set_title("Work per rank over time")
    return finish(fig, path)


def plot_steal_matrix(source: Any, path: str | Path | None = None, value: str = "n_pts") -> plt.Figure:
    """Heatmap of work moved by stealing: victim (rows) to thief (columns), summed ``value``."""
    steals = steals_frame(source)
    _require(steals, "steals")
    mat = _steal_matrix(source, steals, value)
    n = len(mat)
    with afinni_style():
        fig, ax = plt.subplots(figsize=(max(4.5, 0.5 * n + 2.5), max(4.0, 0.5 * n + 1.5)))
        im = ax.imshow(np.ma.masked_equal(mat, 0), cmap="Blues")
        ax.grid(False)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("Thief rank")
        ax.set_ylabel("Victim rank")
        for (i, j), v in np.ndenumerate(mat):
            if v:
                ax.text(j, i, fmt_count(v), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label={"n_pts": "Points moved", "n_clusters": "Clusters moved"}.get(value, value), shrink=0.8)
        ax.set_title(f"Work stealing: {len(steals)} grants")
    return finish(fig, path)


def _steal_matrix(source: Any, steals: pd.DataFrame, value: str) -> np.ndarray:
    """Victim x thief totals of ``value``, including ranks that never stole or were robbed."""
    n = int(steals[["victim", "thief"]].to_numpy().max()) + 1
    if not (isinstance(source, Mapping) and source and all(isinstance(k, int) for k in source)):
        n = max(n, len(_per_rank(profile_of(source))))
    mat = np.zeros((n, n))
    np.add.at(mat, (steals["victim"].to_numpy(), steals["thief"].to_numpy()), steals[value].to_numpy())
    return mat


def plot_profile_dashboard(source: Any, path: str | Path | None = None) -> plt.Figure:
    """One-page summary: time breakdown, forest growth, and (if recorded) load and steals."""
    stats = profile_of(source)
    has_events = not events_frame(stats).empty
    has_timing = phase_breakdown_frame(stats).drop(columns=["total", UNTIMED]).to_numpy().sum() > 0
    panels = [p for p, ok in (
        ("breakdown", has_timing), ("levels", True), ("load", has_events),
        ("steals", has_events and not steals_frame(stats).empty),
    ) if ok]  # fmt: skip
    with afinni_style():
        fig = plt.figure(figsize=(14, 4.2 * ((len(panels) + 1) // 2)))
        for i, name in enumerate(panels):
            ax = fig.add_subplot((len(panels) + 1) // 2, 2, i + 1)
            _dashboard_panel(ax, name, stats)
        fig.suptitle(
            f"AFINNI training profile — {len(_per_rank(stats))} rank(s), "
            f"{stats.get('num_final_clusters', 0):,} leaves, {stats.get('time_total', 0):.3g}s",
            fontweight="bold",
        )
    return finish(fig, path)


def _dashboard_panel(ax: plt.Axes, name: str, stats: dict) -> None:
    """Draw one compact dashboard panel; the full-size versions are the plot_* functions."""
    if name == "breakdown":
        df = phase_breakdown_frame(stats).drop(columns=["total"])
        df = df.loc[:, df.sum() > 0]
        colors = {**group_colors(), UNTIMED: _UNTIMED_COLOR}
        df.div(df.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax, color=[colors[c] for c in df.columns], width=0.85, legend=False)
        ax.set_ylabel("Fraction of time")
        ax.set_xlabel("MPI rank")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_title("Time per phase")
    elif name == "levels":
        df = level_stats_frame(stats).groupby("level")[["clusters_start", "good_clusters"]].sum()
        ax.bar(df.index, df["good_clusters"], color="#59a14f", label="accepted")
        ax.bar(df.index, df["clusters_start"] - df["good_clusters"], bottom=df["good_clusters"], color="#e15759", label="split")
        ax.set_xlabel("Level")
        ax.set_ylabel("Clusters")
        if df["clusters_start"].max() > 50 * max(1, df["clusters_start"].min()):
            ax.set_yscale("log")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_title("Forest growth")
    elif name == "load":
        ev = events_frame(stats)
        loads = ev[ev["kind"].isin(["load", "level_start"])].dropna(subset=["n_pts"])
        for rank, grp in loads.groupby("rank"):
            ax.step(grp["t"], grp["n_pts"], where="post", label=f"rank {rank}", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Queued points")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_count(v)))
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1 + len(loads["rank"].unique()) // 16)
        ax.set_title("Load per rank")
    elif name == "steals":
        s = steals_frame(stats)
        mat = _steal_matrix(stats, s, "n_pts")
        im = ax.imshow(np.ma.masked_equal(mat, 0), cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(mat)))
        ax.set_yticks(range(len(mat)))
        ax.set_xlabel("Thief")
        ax.set_ylabel("Victim")
        ax.grid(False)
        ax.figure.colorbar(im, ax=ax, label="Points moved")
        ax.set_title(f"Steals ({len(s)})")
