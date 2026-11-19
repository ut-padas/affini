"""
Chebyshev arithmetic shared by the schemes: points, bases (with derivatives), coefficients and their tails.

Each function keeps the exact operation order its scheme was validated with; two functions that compute
the same quantity in different orders are kept apart, since merging them would change results bitwise.
"""

from typing import NamedTuple

import numpy as np
from scipy.fft import dct


def cheb_tables(t, n):
    """(N, n + 1) Chebyshev values and derivatives at t."""
    T, D = np.zeros((len(t), n + 1)), np.zeros((len(t), n + 1))
    T[:, 0] = 1.0
    if n:
        T[:, 1], D[:, 1] = t, 1.0
    for j in range(2, n + 1):
        T[:, j] = 2 * t * T[:, j - 1] - T[:, j - 2]
        D[:, j] = 2 * T[:, j - 1] + 2 * t * D[:, j - 1] - D[:, j - 2]
    return T, D


def cheb_points(n: int) -> np.ndarray:
    """Chebyshev points of the second kind on [-1, 1], ``cos(pi j / n)``, j = 0..n (descending)."""
    return np.cos(np.pi * np.arange(n + 1) / n)


def cheb_coefficients(V: np.ndarray) -> np.ndarray:
    """Tensor Chebyshev coefficients of the interpolant through values ``V`` on the second-kind grid."""
    a = np.asarray(V, dtype=np.float64)
    for k in range(a.ndim):
        n = a.shape[k] - 1
        a = dct(a, type=1, axis=k) / n
        edge = [slice(None)] * a.ndim
        for j in (0, n):
            edge[k] = j
            a[tuple(edge)] /= 2.0
    return a


def cheb_eval(coef: np.ndarray, T: np.ndarray) -> np.ndarray:
    """``sum_i coef[i] prod_k T_k[q, i_k]`` for per-axis Chebyshev values ``T`` (list of (Q, n_k + 1))."""
    Q = T[0].shape[0]
    R = T[0] @ coef.reshape(coef.shape[0], -1)  # (Q, rest)
    for k in range(1, coef.ndim):
        R = R.reshape(Q, coef.shape[k], -1)
        R = np.einsum("qi,qir->qr", T[k], R)
    return R[:, 0]


def cheb_basis(t: np.ndarray, n: int) -> np.ndarray:
    """(Q, n + 1) values T_0..T_n at ``t`` in [-1, 1]."""
    return np.cos(np.outer(np.arccos(np.clip(t, -1.0, 1.0)), np.arange(n + 1)))


class AxisFit(NamedTuple):
    """One axis's slab sums ``s`` and the two decay laws fitted on the upper half of its degrees."""

    s: np.ndarray
    C: float  # geometric law s_j ~ C exp(-sigma j)
    sigma: float  # inf: resolved to rounding level; 0: too few degrees to fit (no decay)
    Ca: float  # algebraic law s_j ~ Ca (j + 1)^-p
    p: float
    exponential: bool  # the fit falls by at least ``min_drop`` across the fitted degrees


def axis_decay(a: np.ndarray, k: int, min_drop: float) -> AxisFit:
    """
    Fit axis k's slab sums of |a| on the upper half of the degrees, geometrically (Mavriplis) and
    algebraically (the local regularity of Houston and Suli). The fits read the pairwise maxima
    ``max(s_j, s_j+1)``, so that the zero coefficients of a symmetric f do not read as decay, and
    skip entries below the rounding floor; the last pair below the floor means resolved, as
    Chebfun's chopping reads a plateau at rounding level. The decay counts as exponential when the
    geometric fit falls by ``min_drop`` across the fitted degrees: a threshold on the rate alone
    cannot tell a kink from a slowly converging analytic f on a few degrees, but a power law
    ``j^-p`` falls by only 2^p across ``[n/2, n]``, whatever n.
    """
    s = np.moveaxis(a, k, 0).reshape(a.shape[k], -1).sum(axis=1)
    n = len(s) - 1
    flat = AxisFit(s, float(s.max()), 0.0, float(s.max()), 0.0, False)
    if n < 2:
        return flat
    j = np.arange(min(n // 2, n - 2), n)
    y = np.maximum(s[:-1], s[1:])[j]
    floor = 1e-13 * max(float(s.max()), 1e-300)
    if y[-1] <= floor:
        return AxisFit(s, 0.0, np.inf, 0.0, np.inf, True)
    keep = y > floor
    if keep.sum() < 2:
        return flat
    slope, intercept = np.polyfit(j[keep], np.log(y[keep]), 1)
    power, log_ca = np.polyfit(np.log(j[keep] + 1.0), np.log(y[keep]), 1)
    sigma = float(-slope)
    return AxisFit(s, float(np.exp(intercept)), sigma, float(np.exp(log_ca)), float(-power), sigma * (j[-1] - j[0]) >= np.log(min_drop))


def tail_estimate(fit: AxisFit) -> float:
    """
    Twice the fitted tail of the omitted degrees (interpolation folds them back once), and at least
    the last slab. Where the decay is exponential the tail is the larger of the geometric and the
    algebraic fit's, since a few degrees barely tell the two laws apart. Otherwise f is assumed no
    smoother than a kink there: the algebraic tail with the power capped at 2 (a kink's
    coefficients decay as j^-2; a power fitted on a few pre-asymptotic degrees reads higher and
    underestimates the tail), anchored at the last pair of slabs. A power of at most 1 (a jump)
    never passes.
    """
    s, C, sigma, Ca, p, exponential = fit
    n = len(s) - 1
    if sigma == np.inf:
        return float(s[-1])
    algebraic = Ca * (n + 1.5) ** (1.0 - p) / (p - 1.0) if p > 1.0 else np.inf
    if exponential:
        r = np.exp(-sigma)
        tail = max(C * r ** (n + 1) / (1.0 - r), algebraic)
    else:
        tail = max(float(max(s[-2], s[-1])) * n / (min(p, 2.0) - 1.0), algebraic) if p > 1.0 else np.inf
    return max(2.0 * tail, float(s[-1]))


def degree_needed(fit: AxisFit, share: float) -> float:
    """The smallest degree at which the fitted geometric law's tail estimate meets ``share`` (used only where the decay is exponential)."""
    if fit.sigma == np.inf:
        return 0.0
    r = np.exp(-fit.sigma)
    return max(np.log(2.0 * fit.C * r / ((1.0 - r) * share)), np.log(fit.C / share)) / fit.sigma


def chop(coef: np.ndarray, budget: float) -> np.ndarray:
    """Drop trailing degree slabs, the smallest first, while the dropped |coefficients| sum to at most ``budget``; |T_j| <= 1 makes that sum a bound on the change."""
    spent = 0.0
    while True:
        w = [np.abs(np.take(coef, -1, axis=k)).sum() if coef.shape[k] > 1 else np.inf for k in range(coef.ndim)]
        k = int(np.argmin(w))
        if spent + w[k] > budget:
            return coef
        spent += w[k]
        coef = np.take(coef, np.arange(coef.shape[k] - 1), axis=k)


def coefficients_along(G: np.ndarray, axis: int) -> np.ndarray:
    """Chebyshev coefficients along ``axis`` of node values on the second-kind grid."""
    n = G.shape[axis] - 1
    if n == 0:
        return G.copy()
    a = dct(G, type=1, axis=axis) / n
    edge = [slice(None)] * G.ndim
    for j in (0, n):
        edge[axis] = j
        a[tuple(edge)] /= 2.0
    return a


def chebyshev_tail(v: np.ndarray) -> float:
    """
    Twice the sum of |coefficients| in the top quarter of the degrees of 1-D node values (interpolation
    folds the omitted tail back once). Not the upper half: for geometric decay the error is about the
    last coefficient, a_n ~ a_{n/2}^2 / a_0, so asking the upper half to meet a share asks for about the
    square of the accuracy, and doubling overshoots further.
    """
    a = np.abs(coefficients_along(v, 0))
    n = len(a) - 1
    return 2.0 * float(a[max(1, n - n // 4 + 1) :].sum())


def chebyshev_bases(t: np.ndarray, sizes: list) -> list:
    """Per axis k, the (sizes[k], Q) values T_j(t[:, k]) by the three-term recurrence (no arccos or cos), rows contiguous."""
    out = []
    for k, n1 in enumerate(sizes):
        T = np.empty((n1, len(t)))
        T[0] = 1.0
        if n1 > 1:
            x = np.ascontiguousarray(t[:, k])
            T[1] = x
        for j in range(2, n1):
            np.multiply(x, T[j - 1], out=T[j])
            T[j] *= 2.0
            T[j] -= T[j - 2]
        out.append(T)
    return out
