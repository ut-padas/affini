"""Vector-valued test problems R^d -> R^m: Rosenbrock's gradient, an oblique wave pair, a vortex, and six outputs of two shared oblique coordinates."""

import numpy as np

from src import xp


def rosenbrock_gradient(x):
    """The gradient of the unit-scale Rosenbrock function, ``rosenbrock(x, a=1)`` (a polynomial vector field, m = d)."""
    a, b = x[:, :-1], x[:, 1:]
    g = xp.zeros_like(x)
    g[:, :-1] += -2.0 * (1.0 - a) - 4.0 * a * (b - a**2)
    g[:, 1:] += 2.0 * (b - a**2)
    return g


def wave_pair(x):
    """``(cos 1.5 q.x, sin 1.5 q.x)``, q a fixed random unit vector: an oblique ridge in two outputs (m = 2)."""
    d = int(x.shape[1])
    q = np.random.default_rng(9_000 + d).normal(size=d)
    s = x @ xp.asarray(q / np.linalg.norm(q))
    return xp.stack([xp.cos(1.5 * s), xp.sin(1.5 * s)], axis=1)


def swirl(x):
    """
    Consecutive coordinate pairs rotated by the angle ``2 exp(-|x|^2)``: a smooth vortex, strong near the
    centre and fading outside it (m = d; for odd d the last coordinate is scaled by the cosine).
    """
    d = int(x.shape[1])
    th = 2.0 * xp.exp(-xp.sum(x**2, axis=1, keepdims=True))
    c, s = xp.cos(th), xp.sin(th)
    y = c * x
    pairs = d // 2
    if pairs:
        u, v = x[:, 0 : 2 * pairs : 2], x[:, 1 : 2 * pairs : 2]
        y[:, 0 : 2 * pairs : 2] = c * u - s * v
        y[:, 1 : 2 * pairs : 2] = s * u + c * v
    return y


def shared_latent(x):
    """
    Six nonlinear functions of two shared coordinates, oblique to every axis and curved
    (``g1 = 1.5 q1.x + 0.6 (q2.x)^2``, ``g2 = 1.5 q2.x``, q1, q2 orthonormal, drawn once per d): a
    slow-manifold proxy, where the outputs share structure that the axes do not show (m = 6).
    """
    d = int(x.shape[1])
    Q, _ = np.linalg.qr(np.random.default_rng(7_700 + d).normal(size=(d, 2)))
    s1, s2 = x @ xp.asarray(Q[:, 0]), x @ xp.asarray(Q[:, 1])
    g1, g2 = 1.5 * s1 + 0.6 * s2**2, 1.5 * s2
    return xp.stack([xp.tanh(2 * g1), xp.exp(-((g1 - 0.5) ** 2)) * xp.cos(2 * g2), xp.sin(g1 + g2), g1 * g2 / 3,
                     1 / (1 + (g1 + 1) ** 2 + g2**2), xp.cos(3 * g2 - g1)], axis=1)  # fmt: skip
