"""
Plots for ART models (``algorithm="art"``) and for comparing surrogates.

- ``plot_art_partition``: a 2-D ART tree's cells, coloured by depth, by terminal
  reason and (with ``f_fn``) by pointwise relative error, with the cell boundaries.
- ``plot_art_tree_stats``: leaves per depth stacked by terminal reason, and the
  validation statuses.
- ``plot_error_maps``: pointwise relative error over a 2-D domain for several models side
  by side, with the tolerance contour.
- ``plot_error_cdf``: the distribution of pointwise relative errors of several models.
- ``plot_method_comparison``: bar panels of scalar metrics (errors, leaves, evaluations,
  time) for several methods.

Predictions are passed as callables ``X -> (n, fd)``, so an AFINNI model (kNN predictor)
and an ART model (routed) are compared on exactly the same points.
"""

import textwrap
from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

from src import to_host, xp
from src.art import TERMINAL_REASONS, VALIDATION_STATUSES
from src.infra.profiling import fmt_count
from src.plotting.style import afinni_style, finish

REASON_COLOURS = ("#4c9a6a", "#d1495b", "#edae49")  # accepted, max_depth, split_failed
STATUS_COLOURS = ("#9e9e9e", "#4c9a6a", "#d1495b", "#7b5ea7")


def pointwise_relative_error(truth, pred, floor: float = 1e-12) -> np.ndarray:
    """(n,) worst-component ``|pred - truth| / max(|truth|, floor)`` on the host (the reference ART's definition)."""
    truth, pred = np.asarray(to_host(truth)), np.asarray(to_host(pred))
    rel = np.abs(pred - truth) / np.maximum(np.abs(truth), floor)
    return rel.max(axis=1) if rel.ndim > 1 else rel


def _grid(lower, upper, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ``resolution x resolution`` grid over the box, as (X, Y) mesh and (n, 2) points."""
    gx = np.linspace(lower[0], upper[0], resolution)
    gy = np.linspace(lower[1], upper[1], resolution)
    XX, YY = np.meshgrid(gx, gy)
    return XX, YY, np.column_stack([XX.ravel(), YY.ravel()])


def _boundaries(ids: np.ndarray) -> np.ndarray:
    """Mask of grid cells whose right or upper neighbour belongs to another leaf."""
    edge = np.zeros(ids.shape, dtype=bool)
    edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
    return edge


def plot_art_partition(surrogate, path=None, *, f_fn: Callable | None = None, tol: float | None = None,
                       resolution: int = 400, title: str | None = None) -> plt.Figure:  # fmt: skip
    """
    The cells of a 2-D ART model over its domain: depth of the routed leaf, its
    terminal reason, and (with ``f_fn``) the pointwise relative error with ``tol``
    contoured. Boundaries are drawn where neighbouring grid points route to different
    leaves, i.e. along the stored split planes.
    """
    if surrogate.routing is None or surrogate.X_src.shape[1] != 2:
        raise ValueError("plot_art_partition needs a 2-D ART model.")
    dom = surrogate.routing.domain
    XX, YY, P = _grid(dom.lower, dom.upper, resolution)
    leaf = np.asarray(to_host(surrogate.route_leaves(xp.asarray(P)))).reshape(XX.shape)
    paths = surrogate.art_leaves["path_code"]
    depth = np.array([int(p).bit_length() - 1 for p in paths])[leaf]
    reason = surrogate.art_leaves["terminal_reason"].astype(int)[leaf]
    edges = _boundaries(leaf)
    extent = (dom.lower[0], dom.upper[0], dom.lower[1], dom.upper[1])
    n_panels = 3 if f_fn is not None else 2
    with afinni_style():
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6))
        im = axes[0].imshow(depth, origin="lower", extent=extent, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=axes[0], label="leaf depth")
        axes[0].set_title(f"Cells ({surrogate.N_src:,} leaves)")
        cmap = ListedColormap(REASON_COLOURS)
        axes[1].imshow(reason, origin="lower", extent=extent, cmap=cmap, aspect="auto",
                       norm=BoundaryNorm(np.arange(len(TERMINAL_REASONS) + 1) - 0.5, cmap.N))  # fmt: skip
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REASON_COLOURS]
        counts = surrogate.leaf_status_counts()["terminal_reason"]
        axes[1].legend(handles, [f"{r} ({counts[r]})" for r in TERMINAL_REASONS], loc="upper right", fontsize=8)
        axes[1].set_title("Terminal reason")
        if f_fn is not None:
            pred = surrogate.local_model.evaluate(surrogate.model_params(), xp.asarray(P), xp.asarray(leaf.ravel()))
            err = pointwise_relative_error(f_fn(xp.asarray(P)), pred).reshape(XX.shape)
            im = axes[2].imshow(np.maximum(err, 1e-12), origin="lower", extent=extent, aspect="auto",
                                norm=LogNorm(vmin=max(1e-8, float(err[err > 0].min()) if np.any(err > 0) else 1e-8)),
                                cmap="magma")  # fmt: skip
            fig.colorbar(im, ax=axes[2], label="pointwise relative error")
            if tol is not None:
                axes[2].contour(XX, YY, err, levels=[tol], colors="cyan", linewidths=0.8)
            axes[2].set_title("Pointwise error" + (f" (cyan: {tol:g})" if tol is not None else ""))
        for ax in axes:
            ax.imshow(np.ma.masked_where(~edges, edges), origin="lower", extent=extent, cmap="gray_r",
                      aspect="auto", alpha=0.6, interpolation="nearest")  # fmt: skip
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.grid(False)
        if title:
            fig.suptitle(title)
    return finish(fig, path)


def plot_art_tree_stats(surrogate, path=None, title: str | None = None) -> plt.Figure:
    """Leaves per depth, stacked by terminal reason, and the leaves' validation statuses."""
    if surrogate.routing is None:
        raise ValueError("plot_art_tree_stats needs an ART model.")
    depth = np.array([int(p).bit_length() - 1 for p in surrogate.art_leaves["path_code"]])
    reason = surrogate.art_leaves["terminal_reason"].astype(int)
    levels = np.arange(depth.max() + 1)
    with afinni_style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1.3]})
        bottom = np.zeros(len(levels))
        for code, (name, colour) in enumerate(zip(TERMINAL_REASONS, REASON_COLOURS, strict=True)):
            n = np.bincount(depth[reason == code], minlength=len(levels))[: len(levels)]
            ax.bar(levels, n, bottom=bottom, color=colour, label=name)
            bottom += n
        ax.set_xlabel("depth")
        ax.set_ylabel("leaves")
        ax.set_title(f"Leaves per depth ({surrogate.N_src:,} leaves)")
        ax.legend()
        status = surrogate.leaf_status_counts()["validation_status"]
        ax2.barh(list(VALIDATION_STATUSES), [status[s] for s in VALIDATION_STATUSES], color=STATUS_COLOURS)
        ax2.set_title("Validation status")
        ax2.set_xlabel("leaves")
        if title:
            fig.suptitle(title)
    return finish(fig, path)


def plot_error_maps(models: Mapping[str, Callable], f_fn: Callable, lower, upper, path=None, *,
                    tol: float | None = None, resolution: int = 300, title: str | None = None) -> plt.Figure:  # fmt: skip
    """
    Pointwise relative error over a 2-D box for each ``{label: predict(X) -> (n, fd)}``,
    on a shared log colour scale, with the ``tol`` contour and the share of the box within it.
    """
    XX, YY, P = _grid(np.asarray(lower, float), np.asarray(upper, float), resolution)
    truth = f_fn(xp.asarray(P))
    errs = {label: pointwise_relative_error(truth, predict(xp.asarray(P))).reshape(XX.shape)
            for label, predict in models.items()}  # fmt: skip
    positive = np.concatenate([e[e > 0] for e in errs.values()])
    norm = LogNorm(vmin=max(1e-8, float(positive.min()) if positive.size else 1e-8),
                   vmax=max(1e-6, float(positive.max()) if positive.size else 1.0))  # fmt: skip
    extent = (lower[0], upper[0], lower[1], upper[1])
    with afinni_style():
        fig, axes = plt.subplots(1, len(errs), figsize=(5.0 * len(errs) + 1.0, 4.4), squeeze=False)
        for ax, (label, err) in zip(axes[0], errs.items(), strict=True):
            im = ax.imshow(np.maximum(err, norm.vmin), origin="lower", extent=extent, norm=norm, cmap="magma",
                           aspect="auto")  # fmt: skip
            sub = label
            if tol is not None:
                ax.contour(XX, YY, err, levels=[tol], colors="cyan", linewidths=0.8)
                sub += f"\n{100 * np.mean(err <= tol):.1f}% of the box within {tol:g}"
            ax.set_title(sub)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.grid(False)
        fig.colorbar(im, ax=axes[0].tolist(), label="pointwise relative error")
        if title:
            fig.suptitle(title)
    return finish(fig, path)


def plot_error_cdf(errors: Mapping[str, np.ndarray], path=None, *, tol: float | None = None,
                   title: str | None = None) -> plt.Figure:  # fmt: skip
    """Empirical CDF of each ``{label: pointwise relative errors}`` on a log axis, with ``tol`` marked."""
    with afinni_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for label, err in errors.items():
            e = np.sort(np.maximum(np.asarray(err, float), 1e-16))
            share = f" ({100 * np.mean(e <= tol):.1f}% <= tol)" if tol is not None else ""
            ax.step(e, np.arange(1, e.size + 1) / e.size, where="post", label=label + share)
        if tol is not None:
            ax.axvline(tol, color="k", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("pointwise relative error")
        ax.set_ylabel("share of test points")
        ax.set_title(title or "Error distribution on independent test points")
        ax.legend(loc="lower right")
    return finish(fig, path)


# Metric key -> (panel title, log scale allowed when the values span more than 10x)
COMPARISON_METRICS = {
    "relative_l2": ("relative L2 error", True),
    "q99_pointwise": ("99th pct pointwise rel. error", True),
    "pct_above_tol": ("% of points above tol", False),
    "pout": ("% outside (trust region / domain)", False),
    "leaves": ("leaves", True),
    "evals": ("oracle evaluations", True),
    "train_seconds": ("training time [s]", True),
}


def plot_method_comparison(metrics: Mapping[str, Mapping[str, float]], path=None, *,
                           keys=tuple(COMPARISON_METRICS), title: str | None = None) -> plt.Figure:  # fmt: skip
    """One bar panel per metric key for each ``{method: {key: value}}`` (missing keys are skipped)."""
    keys = [k for k in keys if any(k in m for m in metrics.values())]
    labels = list(metrics)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    with afinni_style():
        fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys) + 0.5, 3.4), squeeze=False)
        for ax, key in zip(axes[0], keys, strict=True):
            name, log = COMPARISON_METRICS.get(key, (key, False))
            values = [float(metrics[m].get(key, np.nan)) for m in labels]
            bars = ax.bar(range(len(labels)), values, color=colours[: len(labels)])
            for bar, v in zip(bars, values, strict=True):
                ax.annotate(fmt_count(v) if v >= 1 else f"{v:.2g}", (bar.get_x() + bar.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=8)  # fmt: skip
            ax.set_xticks(range(len(labels)), [textwrap.fill(lbl, 16) for lbl in labels], fontsize=8)
            ax.set_title(name, fontsize=10)
            finite = [v for v in values if np.isfinite(v)]
            # Log only when the values span more than 10x; otherwise a linear axis from 0, so
            # near-equal values look near-equal (an auto-zoomed log axis exaggerates them).
            if log and finite and min(finite) > 0 and max(finite) > 10 * min(finite):
                ax.set_yscale("log")
                ax.set_ylim(min(finite) / 3, max(finite) * 3)
            else:
                ax.set_ylim(0, (max(finite) if finite else 1) * 1.15)
        if title:
            fig.suptitle(title)
    return finish(fig, path)
