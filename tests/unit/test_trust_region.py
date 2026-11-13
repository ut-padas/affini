"""Tests for adaptive_clustering_utils.trust_region_shapes and lsq_hessians."""

import numpy as np

from src import to_host, xp
from src.components.local_models import lsq_hessians
from src.components.trust_regions import trust_region_shapes

rng = np.random.default_rng(0)
SIZES = np.array([50, 80, 30])
D = 3
LABELS = np.repeat(np.arange(len(SIZES)), SIZES)
X = (
    rng.standard_normal((SIZES.sum(), D)) @ rng.standard_normal((D, D))
    + 5 * LABELS[:, None]
)
STARTS = np.r_[0, np.cumsum(SIZES)[:-1]]
CENTROIDS = np.add.reduceat(X, STARTS) / SIZES[:, None]


def test_farthest_point_lies_on_the_trust_region_boundary():
    Q = to_host(
        trust_region_shapes(
            xp.asarray(X),
            xp.asarray(CENTROIDS),
            xp.asarray(LABELS),
            xp.asarray(SIZES),
            q_reg=0.0,
        )
    )
    assert Q.shape == (len(SIZES), D, D)
    np.testing.assert_allclose(Q, Q.transpose(0, 2, 1), atol=1e-12)
    dx = X - CENTROIDS[LABELS]
    r = np.einsum("nc,ncd,nd->n", dx, Q[LABELS], dx)  # dx^T Q dx per point
    np.testing.assert_allclose(np.maximum.reduceat(r, STARTS), 1.0, rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_lsq_hessians_recover_a_quadratic():
    # f(x) = 1/2 x^T A x on each cluster: G - g_c = A (x - c) exactly.
    A = rng.standard_normal((D, D))
    A = A + A.T
    G = (X @ A)[:, None, :]  # (N, 1, d)
    g_c = (CENTROIDS @ A)[:, None, :]
    H = to_host(
        lsq_hessians(
            xp.asarray(X),
            xp.asarray(G),
            xp.asarray(CENTROIDS),
            xp.asarray(g_c),
            xp.asarray(LABELS),
            xp.asarray(SIZES),
        )
    )
    np.testing.assert_allclose(H, np.broadcast_to(A, H.shape), atol=1e-8)
