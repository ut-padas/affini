"""The launcher's result JSON (one file per run, feeding the summary CSV and the profile plots) and its tolerances."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import afinni_launcher as launcher  # noqa: E402
from src import plotting as P  # noqa: E402
from src import xp  # noqa: E402
from src.adaptive_clustering_utils import SurrogateMode  # noqa: E402
from tests._helpers import host  # noqa: E402


def test_result_json_holds_the_profile_and_feeds_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "RESULTS_DIR", tmp_path)
    exp_dir = launcher.run_problem(
        launcher.rosenbrock(None), dim=2, tol=0.1, error_eval_mode=SurrogateMode.AFFINE_COV_HESS,
        Q_reg=launcher.DEFAULT_Q_REG, coarse_factor=launcher.DEFAULT_COARSE_FACTOR, N_train=2000,
    )  # fmt: skip

    (result_path,) = exp_dir.glob("*.json")  # the profile must not be a second JSON the summary would read
    result = json.loads(result_path.read_text())
    assert result["profiling_stats"]["num_final_clusters"] == result["N_c"]
    assert isinstance(result["profiling_stats"]["time_total"], float)  # array scalars saved as numbers, not text

    summary = pd.read_csv(exp_dir / "_summary.csv")
    assert summary[["dim", "tol", "N_c"]].values.tolist() == [[2, 0.1, result["N_c"]]]

    assert isinstance(P.plot_profile_dashboard(result), plt.Figure)


def test_scaled_atol_is_tol_times_max_f_per_component(tmp_path):
    rng = np.random.default_rng(0)
    X_train, X_test = (xp.asarray(rng.uniform(-1, 1, (n, 2))) for n in (2000, 500))

    def f(x):
        return xp.stack((1 + x[:, 0] ** 2 + 3 * x[:, 1] ** 2, 100 * x[:, 0] * x[:, 1]), axis=1)

    json_fpath = tmp_path / "run.json"
    launcher.conduct_experiment_given_data(
        json_fpath=json_fpath, DOMAIN=[-1.0, 1.0], F_DIM=2, DIM=2, N_TRAIN=2000, N_TEST=500,
        X_train=X_train, X_test=X_test, f_fn=f, g_fn=None, TOL=0.1,
        ERROR_EVAL_MODE=SurrogateMode.AFFINE_COV_HESS, ERROR_METRIC="Linf", INITIAL_CLUSTERING_METHOD="kmeans",
        NUM_INITIAL_CLUSTERS=4, MIN_PTS_PER_CLUSTER=32, USE_WORK_STEALING=False, SCALE_ATOL_BY_MAX_F=True,
    )  # fmt: skip
    result = json.loads(json_fpath.read_text())
    expected = 0.1 * np.max(np.abs(host(f(X_train))), axis=0)
    np.testing.assert_allclose(result["atol"], expected, rtol=1e-15)
    assert result["rtol"] is None and result["config"]["rtol"] is None
