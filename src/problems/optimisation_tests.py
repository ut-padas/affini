"""Standard d-dimensional optimisation test functions: Ackley, Griewank, Styblinski-Tang, Michalewicz (Rastrigin is in ``rastrigin``)."""

import numpy as np

from src import xp


def ackley(x):
    """Ackley: many local minima on a cone, with a kink at the centre (the square root of the mean square)."""
    d = x.shape[1]
    r = xp.sqrt(xp.sum(x**2, axis=1, keepdims=True) / d)
    c = xp.sum(xp.cos(2 * np.pi * x), axis=1, keepdims=True) / d
    return -20.0 * xp.exp(-0.2 * r) - xp.exp(c) + 20.0 + np.e


def griewank(x):
    """Griewank: a bowl times a product of cosines whose frequencies fall as 1 / sqrt(i)."""
    i = xp.arange(1, x.shape[1] + 1)
    return 1.0 + xp.sum(x**2, axis=1, keepdims=True) / 4000.0 - xp.prod(xp.cos(x / xp.sqrt(i)), axis=1, keepdims=True)


def styblinski_tang(x):
    """Styblinski-Tang: a separable quartic with 2^d local minima."""
    return 0.5 * xp.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1, keepdims=True)


def michalewicz(x, m: int = 10):
    """Michalewicz: steep narrow ridges, sharper as m grows (here m = 10), on [0, pi]^d."""
    i = xp.arange(1, x.shape[1] + 1)
    return -xp.sum(xp.sin(x) * xp.sin(i * x**2 / np.pi) ** (2 * m), axis=1, keepdims=True)
