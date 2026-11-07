"""AFINNISurrogate stores and predicts with whatever columns the local model declares."""

import numpy as np

import src.local_models as lm
from src import _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.clusters import ParamSpec
from src.problems.rosenbrock import rosenbrock
from src.surrogate import AFINNISurrogate
from tests._helpers import assert_backend_equal


class RenamedAffineModel(lm.LocalModel):
    """AffineLSQModel with its parameters stored under new column names."""

    name = "renamed_affine"
    _affine = lm.AffineLSQModel()

    def columns(self, d, fd):
        return {"slope": ParamSpec((fd, d)), "intercept": ParamSpec((fd,))}

    def fit(self, data):
        p = self._affine.fit(data)
        return {"slope": p["A_fit"], "intercept": p["b_fit"]}

    def evaluate(self, params, X, cluster_ids):
        renamed = {"A_fit": params["slope"], "b_fit": params["intercept"]}
        return self._affine.evaluate(renamed, X, cluster_ids)


def _train_and_predict(X, Q, local_model=None):
    surrogate = AFINNISurrogate()
    surrogate.train(
        xp.asarray(X),
        f_fn=rosenbrock,
        mode=SurrogateMode.AFFINE_COV_HESS,
        local_model=local_model,
        ATOL=0.1,
        RTOL=0.1,
        NUM_INITIAL_CLUSTERS=16,
        MIN_PTS_PER_CLUSTER=50,
        USE_WORK_STEALING=False,
    )
    pred, unguaranteed = surrogate.predict(xp.asarray(Q), k_nn=4)[:2]
    return surrogate, _to_host(pred), _to_host(unguaranteed)


def test_new_model_columns_reach_predict():
    rng = np.random.default_rng(0)
    X, Q = rng.uniform(-2, 2, (20_000, 2)), rng.uniform(-2, 2, (2_000, 2))
    _, pred_ref, ung_ref = _train_and_predict(X, Q)

    surrogate, pred, ung = _train_and_predict(X, Q, local_model=RenamedAffineModel())

    assert set(surrogate.leaf_params) == {"slope", "intercept"}
    assert surrogate.A_fits_src is None  # no "A_fit" column for this model
    assert surrogate.local_model.name == "renamed_affine"
    assert_backend_equal(pred, pred_ref, separate_runs=True)
    np.testing.assert_array_equal(ung, ung_ref)
