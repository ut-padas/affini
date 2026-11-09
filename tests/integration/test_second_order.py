"""
Second-order Taylor surrogates (TAYLOR_LSQ_HESS with STORE_HESSIAN) end to end.

Every output component has its own Hessian. Previously one Hessian, summed over
the outputs, was applied to all of them, which silently gave wrong predictions for
vector-valued functions (e.g. f = [x^2, -x^2] predicted ~0).
"""

import numpy as np
import pytest

from src import xp
from src.modes import SurrogateMode
from src.surrogate import AFINNISurrogate
from tests._helpers import host, train_surrogate


def train(X, f, g, **kw):
    s = AFINNISurrogate()
    opts = dict(
        mode=SurrogateMode.TAYLOR_LSQ_HESS, store_hess=True, atol=10.0, rtol=10.0,
        num_initial_clusters=1, min_pts_per_cluster=8, max_clusters_est=10, use_work_stealing=False,
    )  # fmt: skip
    opts.update(kw)
    train_surrogate(X, s, f_fn=f, g_fn=g, **opts)
    return s


def test_opposite_curvatures_are_not_cancelled():
    X = np.linspace(-1, 1, 101)[:, None]
    s = train(X, lambda x: xp.concatenate((x * x, -x * x), axis=1), lambda x: xp.stack((2 * x, -2 * x), axis=1))
    assert s.N_src == 1
    np.testing.assert_allclose(host(s.Hs_src), [[[[2.0]], [[-2.0]]]], atol=1e-10)
    pred, _ = s.predict(xp.asarray([[0.5]]))
    np.testing.assert_allclose(host(pred), [[0.25, -0.25]], atol=1e-10)


A = np.array([[[2.0, 0.6], [0.6, -1.0]], [[-3.0, 0.2], [0.2, 4.0]], [[0.0, 1.0], [1.0, 0.0]]])
B = np.array([[1.0, -2.0], [0.5, 0.5], [0.0, 3.0]])


def quad_f(x):
    return x @ xp.asarray(B).T + 0.5 * xp.einsum("nc,fcd,nd->nf", x, xp.asarray(A), x)


def quad_g(x):
    return xp.asarray(B)[None] + xp.einsum("fcd,nd->nfc", xp.asarray(A), x)


@pytest.mark.parametrize("k", [1, 4])
def test_vector_quadratics_are_reproduced_exactly(k):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2_000, 2))
    s = train(X, quad_f, quad_g, num_initial_clusters=6, min_pts_per_cluster=32, max_clusters_est=64)
    H = host(s.Hs_src)
    assert H.shape == (s.N_src, 3, 2, 2)
    np.testing.assert_allclose(H, np.broadcast_to(A, H.shape), atol=1e-9)
    Q = rng.uniform(-1.5, 1.5, (400, 2))
    pred, _ = s.predict(xp.asarray(Q), k_nn=k)
    np.testing.assert_allclose(host(pred), host(quad_f(xp.asarray(Q))), rtol=1e-9, atol=1e-9)


def test_scalar_second_order_is_unchanged():
    X = np.linspace(-1, 1, 101)[:, None]
    s = train(X, lambda x: x * x, lambda x: (2 * x)[:, None, :])
    assert host(s.Hs_src).shape == (1, 1, 1, 1)
    pred, _ = s.predict(xp.asarray([[0.5]]))
    np.testing.assert_allclose(host(pred), [[0.25]], atol=1e-12)
