"""Functions with an interface: a sharp smooth front on a sphere, and jumps across an oblique plane and across a sphere."""

import numpy as np

from src import xp


def front(x):
    """A sharp smooth front on the unit sphere: tanh((|x| - 1) / 0.05)."""
    return xp.tanh((xp.sqrt(xp.sum(x**2, axis=1, keepdims=True)) - 1.0) / 0.05)


def oblique_jump(x):
    """A smooth function with a unit jump across an oblique hyperplane: sin(2 q.x) + [q.x > 0.3], q a fixed random unit vector."""
    d = int(x.shape[1])
    q = np.random.default_rng(8_000 + d).normal(size=d)
    q = xp.asarray(q / np.linalg.norm(q))
    s = x @ q
    return (xp.sin(2.0 * s) + (s > 0.3))[:, None]


def ball_jump(x):
    """A smooth bump that drops to zero across a sphere: exp(-|x - c|^2) inside |x - c| < 0.8, 0 outside (a curved jump)."""
    c = 0.2
    r2 = xp.sum((x - c) ** 2, axis=1, keepdims=True)
    return xp.exp(-r2) * (r2 < 0.64)
