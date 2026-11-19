"""Genz's test families on [0, 1]^d (oscillatory, product peak, corner peak, Gaussian, continuous, discontinuous), with his difficulty constants, and the three smooth ones side by side as a vector problem."""

import numpy as np

from src import xp

# Genz's difficulty constants b (the sum of the a_i).
GENZ_DIFFICULTY = {"oscillatory": 9.0, "product_peak": 7.25, "corner_peak": 1.85, "gaussian": 7.03, "continuous": 20.4}


def genz(kind: str):
    """One of Genz's families, with a and w drawn once per d from a fixed seed and a scaled to sum to b."""

    def f(x):
        d = int(x.shape[1])
        rng = np.random.default_rng(7_000 + d)
        a = rng.uniform(0.0, 1.0, d)
        a = xp.asarray(a * GENZ_DIFFICULTY[kind] / a.sum())
        w = xp.asarray(rng.uniform(0.0, 1.0, d))
        if kind == "oscillatory":
            return xp.cos(2 * np.pi * w[0] + x @ a)[:, None]
        if kind == "product_peak":
            return xp.prod(1.0 / (a ** (-2.0) + (x - w) ** 2), axis=1, keepdims=True)
        if kind == "corner_peak":
            return ((1.0 + x @ a) ** (-(d + 1)))[:, None]
        if kind == "gaussian":
            return xp.exp(-xp.sum(a**2 * (x - w) ** 2, axis=1, keepdims=True))
        return xp.exp(-xp.sum(a * xp.abs(x - w), axis=1, keepdims=True))  # continuous: kinks at x_i = w_i

    return f


def genz_discontinuous(x):
    """Genz's discontinuous family: exp(a.x) where x_1 < w_1 and x_2 < w_2, else 0 (an axis-aligned corner jump)."""
    d = int(x.shape[1])
    rng = np.random.default_rng(7_000 + d)
    a = rng.uniform(0.0, 1.0, d)
    a = xp.asarray(a * 4.3 / a.sum())
    w = rng.uniform(0.0, 1.0, d)
    inside = (x[:, 0] < w[0]) & ((x[:, 1] < w[1]) if d > 1 else True)
    return (xp.exp(x @ a) * inside)[:, None]


def genz_triple(x):
    """Genz's oscillatory, product-peak and corner-peak functions side by side: outputs of different structure, m = 3 for any d."""
    return xp.concatenate([genz(k)(x) for k in ("oscillatory", "product_peak", "corner_peak")], axis=1)
