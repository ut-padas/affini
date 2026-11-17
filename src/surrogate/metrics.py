"""Error metrics of a surrogate's predictions against reference values (relative L2, pointwise tolerance shares, outside-region counts)."""

from typing import Any

import numpy as np

from src import (
    Array,
    to_host,
    xp,
)
from src.components.acceptance import Tolerance, normalize_tolerance, tolerance_vector

TINY_EPS = 1e-12
# Bins of the tolerance ratio stored with every result: 8 per decade from 1e-6 to 1e3. A ratio below
# or above the range is counted in the first or last bin, a NaN one (a NaN prediction) in the last.
TOLERANCE_RATIO_EDGES = np.logspace(-6, 3, 9 * 8 + 1)


def tolerance_ratio(preds: Array, gt: Array, atol: Tolerance, rtol: Tolerance) -> Array:
    """
    Each point's error in units of its tolerance: over the output components, the largest
    min(|error| / atol, relative error / rtol), with a missing tolerance never accepting. A point
    passes the acceptance test of ``compute_error_metrics`` (and of training) when this is <= 1, up
    to the rounding of the division; 3 means three tolerances off.
    """
    atol, rtol = normalize_tolerance("atol", atol), normalize_tolerance("rtol", rtol)
    err = xp.abs(preds - gt).reshape(gt.shape[0], -1)
    rel = err / (xp.abs(gt).reshape(err.shape) + TINY_EPS)
    fd = err.shape[1]

    def in_units(e, name, tol):
        tv = tolerance_vector(name, tol, fd)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = e / tv
        # an omitted tolerance (-inf) never accepts; a zero one accepts only an exact value
        return xp.where(tv < 0, xp.inf, xp.where(e == 0, 0.0, r))

    r = xp.minimum(in_units(err, "atol", atol), in_units(rel, "rtol", rtol))
    return xp.max(xp.where(xp.isnan(r), xp.inf, r), axis=1)


def tolerance_ratio_counts(ratio: Array) -> list[int]:
    """Counts of ``ratio`` in the bins of ``TOLERANCE_RATIO_EDGES`` (a list, so it can be stored as JSON)."""
    r = np.nan_to_num(to_host(ratio), nan=np.inf)
    r = np.clip(r, TOLERANCE_RATIO_EDGES[0], TOLERANCE_RATIO_EDGES[-1])
    return np.histogram(r, TOLERANCE_RATIO_EDGES)[0].tolist()


def compute_error_metrics(
    preds,
    gt,
    ug_mask,
    atol: Tolerance = None,
    rtol: Tolerance = None,
    return_ptwise_error_arrays: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Array]]:
    """
    L2/Linf errors over all queries and separately inside/outside the trust regions
    (``ug_mask`` marks outside), plus the share of inside points that are good: every
    component within ``rtol`` relative or ``atol`` absolute error (inclusive), with the
    tolerances given as in training (see ``src.components.acceptance``). Sections with no points
    report NaN. ``tolerance_ratio`` holds the histogram of ``tolerance_ratio`` inside and outside
    (``src.plotting.plot_error_summary`` draws it). Optionally also returns the pointwise relative
    errors.
    """
    atol, rtol = normalize_tolerance("atol", atol), normalize_tolerance("rtol", rtol)
    if atol is None and rtol is None:
        raise ValueError("Give atol, rtol or both to count the points within tolerance.")

    n_test = gt.shape[0]
    n_out = xp.sum(ug_mask)
    n_in = n_test - n_out

    good_idx = ~ug_mask

    if n_in > 0:
        # 1. Filter to guaranteed points
        gt_in = gt[good_idx]
        preds_in = preds[good_idx]
        diff = preds_in - gt_in
        norm_diff = xp.linalg.norm(diff)
        norm_gt = xp.linalg.norm(gt_in)
        norm_diff_inf = xp.linalg.norm(diff, ord=xp.inf)
        norm_gt_inf = xp.linalg.norm(gt_in, ord=xp.inf)

        in_l2_rel_error = float(norm_diff / (norm_gt + TINY_EPS))
        in_l2_abs_error = float(norm_diff / xp.sqrt(n_in))
        in_linf_rel_error = float(norm_diff_inf / (norm_gt_inf + TINY_EPS))
        in_linf_abs_error = float(norm_diff_inf / xp.sqrt(n_in))

        element_abs_error = xp.abs(diff)
        element_rel_error = xp.abs(diff) / (xp.abs(gt_in) + TINY_EPS)
        # A vector point's reported error is its largest component error.
        ptwise_rel_error = xp.max(element_rel_error, axis=1) if element_rel_error.ndim > 1 else element_rel_error

        in_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error))
        in_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error))

        abs_2d = element_abs_error.reshape(len(element_abs_error), -1)
        rel_2d = element_rel_error.reshape(abs_2d.shape)
        fd = abs_2d.shape[1]
        within = (rel_2d <= tolerance_vector("rtol", rtol, fd)) | (abs_2d <= tolerance_vector("atol", atol, fd))
        n_in_good = int(xp.sum(xp.all(within, axis=1)))
        n_in_bad = int(n_in - n_in_good)
    else:
        in_l2_rel_error = float("nan")
        in_l2_abs_error = float("nan")
        in_linf_rel_error = float("nan")
        in_linf_abs_error = float("nan")
        in_ptwise_max_rel_error_linf = float("nan")
        in_ptwise_median_rel_error_linf = float("nan")
        n_in_good = 0
        n_in_bad = 0

    if n_out > 0:
        gt_out = gt[ug_mask]
        preds_out = preds[ug_mask]
        diff_out = preds_out - gt_out
        norm_diff_out = xp.linalg.norm(diff_out)
        norm_diff_out_inf = xp.linalg.norm(diff_out, ord=xp.inf)
        norm_gt_out = xp.linalg.norm(gt_out)
        norm_gt_out_inf = xp.linalg.norm(gt_out, ord=xp.inf)

        out_l2_rel_error = float(norm_diff_out / (norm_gt_out + TINY_EPS))
        out_l2_abs_error = float(norm_diff_out / xp.sqrt(n_out))
        out_linf_rel_error = float(norm_diff_out_inf / (norm_gt_out_inf + TINY_EPS))
        out_linf_abs_error = float(norm_diff_out_inf / xp.sqrt(n_out))

        element_rel_error_out = xp.abs(diff_out) / (xp.abs(gt_out) + TINY_EPS)
        if element_rel_error_out.ndim > 1:
            ptwise_rel_error_out = xp.max(element_rel_error_out, axis=1)
        else:
            ptwise_rel_error_out = element_rel_error_out

        out_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error_out))
        out_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error_out))
    else:
        out_l2_rel_error = float("nan")
        out_l2_abs_error = float("nan")
        out_linf_rel_error = float("nan")
        out_linf_abs_error = float("nan")
        out_ptwise_max_rel_error_linf = float("nan")
        out_ptwise_median_rel_error_linf = float("nan")

    # Metrics for all points
    diff_all = preds - gt
    norm_diff_all = xp.linalg.norm(diff_all)
    norm_diff_all_inf = xp.linalg.norm(diff_all, ord=xp.inf)
    norm_gt_all = xp.linalg.norm(gt)
    norm_gt_all_inf = xp.linalg.norm(gt, ord=xp.inf)

    all_l2_rel_error = float(norm_diff_all / (norm_gt_all + TINY_EPS))
    all_l2_abs_error = float(norm_diff_all / xp.sqrt(n_test))
    all_linf_rel_error = float(norm_diff_all_inf / (norm_gt_all_inf + TINY_EPS))
    all_linf_abs_error = float(norm_diff_all_inf / xp.sqrt(n_test))
    element_rel_error_all = xp.abs(diff_all) / (xp.abs(gt) + TINY_EPS)
    if element_rel_error_all.ndim > 1:
        ptwise_rel_error_all = xp.max(element_rel_error_all, axis=1)
    else:
        ptwise_rel_error_all = element_rel_error_all
    all_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error_all))
    all_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error_all))

    # Calculate percentages
    perct_out = float(100.0 * n_out / n_test)
    perct_in_good = float(100.0 * n_in_good / n_test)
    perct_in_bad = float(100.0 * n_in_bad / n_test)

    ratio = tolerance_ratio(preds, gt, atol, rtol)
    ratio_in, ratio_out = ratio[good_idx], ratio[ug_mask]
    # Create error metrics dictionary
    error_metrics = {
        "n_test": int(n_test),
        "n_out": int(n_out),
        "perct_out": perct_out,
        "n_in_good": n_in_good,
        "perct_in_good": perct_in_good,
        "n_in_bad": n_in_bad,
        "perct_in_bad": perct_in_bad,
        "all": {
            "l2_rel_error": all_l2_rel_error,
            "l2_abs_error": all_l2_abs_error,
            "linf_rel_error": all_linf_rel_error,
            "linf_abs_error": all_linf_abs_error,
            "ptwise_max_rel_error_linf": all_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": all_ptwise_median_rel_error_linf,
        },
        "in": {
            "l2_rel_error": in_l2_rel_error,
            "l2_abs_error": in_l2_abs_error,
            "linf_rel_error": in_linf_rel_error,
            "linf_abs_error": in_linf_abs_error,
            "ptwise_max_rel_error_linf": in_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": in_ptwise_median_rel_error_linf,
        },
        "out": {
            "l2_rel_error": out_l2_rel_error,
            "l2_abs_error": out_l2_abs_error,
            "linf_rel_error": out_linf_rel_error,
            "linf_abs_error": out_linf_abs_error,
            "ptwise_max_rel_error_linf": out_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": out_ptwise_median_rel_error_linf,
        },
        "tolerance_ratio": {
            "edges": TOLERANCE_RATIO_EDGES.tolist(),
            "in": tolerance_ratio_counts(ratio_in),
            "out": tolerance_ratio_counts(ratio_out),
            "max_in": float(xp.max(ratio_in)) if n_in > 0 else float("nan"),
            "max_out": float(xp.max(ratio_out)) if n_out > 0 else float("nan"),
        },
    }
    if not return_ptwise_error_arrays:
        return error_metrics
    else:
        return error_metrics, {
            "all": to_host(ptwise_rel_error_all),
            "in": to_host(ptwise_rel_error),
            "out": to_host(ptwise_rel_error_out),
        }
