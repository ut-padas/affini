"""The launcher's result JSON (one file per run, feeding the summary CSV and the profile plots) and its tolerance arguments."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import afinni_launcher as launcher  # noqa: E402
from src import plotting as P  # noqa: E402
from src import xp  # noqa: E402
from src.components.modes import SurrogateMode# noqa: E402
from tests._helpers import host  # noqa: E402


def test_result_json_holds_the_profile_and_feeds_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "RESULTS_DIR", tmp_path)
    exp_dir = launcher.run_problem(
        launcher.rosenbrock(None), dim=2, atol=0.1, rtol=0.1, error_eval_mode=SurrogateMode.AFFINE_COV_HESS,
        Q_reg=launcher.DEFAULT_Q_REG, coarse_factor=launcher.DEFAULT_COARSE_FACTOR, N_train=2000,
    )  # fmt: skip

    (result_path,) = exp_dir.glob("*.json")  # the profile must not be a second JSON the summary would read
    result = json.loads(result_path.read_text())
    assert result["profiling_stats"]["num_final_clusters"] == result["N_c"]
    assert isinstance(result["profiling_stats"]["time_total"], float)  # array scalars saved as numbers, not text

    assert result_path.name.startswith("mpi_sz1_dim2_tol0.1_")  # --tols names are unchanged
    heuristic = f"_ninit{launcher.get_NUM_INITIAL_CLUSTERS(2)}_minpts{launcher.get_MIN_PTS_PER_CLUSTER(2)}_"
    assert heuristic in result_path.name  # the defaults without --num_initial_clusters / --min_pts_per_cluster
    summary = pd.read_csv(exp_dir / "_summary.csv")
    assert summary[["dim", "atol", "rtol", "N_c"]].values.tolist() == [[2, 0.1, 0.1, result["N_c"]]]

    assert isinstance(P.plot_profile_dashboard(result), plt.Figure)


def test_scaled_atol_is_tol_times_max_f_per_component(tmp_path):
    rng = np.random.default_rng(0)
    X_train, X_test = (xp.asarray(rng.uniform(-1, 1, (n, 2))) for n in (2000, 500))

    def f(x):
        return xp.stack((1 + x[:, 0] ** 2 + 3 * x[:, 1] ** 2, 100 * x[:, 0] * x[:, 1]), axis=1)

    json_fpath = tmp_path / "run.json"
    launcher.conduct_experiment_given_data(
        json_fpath=json_fpath, DOMAIN=[-1.0, 1.0], F_DIM=2, DIM=2, N_TRAIN=2000, N_TEST=500,
        X_train=X_train, X_test=X_test, f_fn=f, g_fn=None, ATOL=0.1, RTOL=None,
        ERROR_EVAL_MODE=SurrogateMode.AFFINE_COV_HESS, ERROR_METRIC="Linf", INITIAL_CLUSTERING_METHOD="kmeans",
        NUM_INITIAL_CLUSTERS=4, MIN_PTS_PER_CLUSTER=32, USE_WORK_STEALING=False, SCALE_ATOL_BY_MAX_F=True,
    )  # fmt: skip
    result = json.loads(json_fpath.read_text())
    expected = 0.1 * np.max(np.abs(host(f(X_train))), axis=0)
    np.testing.assert_allclose(result["atol_scaled"], expected, rtol=1e-15)
    np.testing.assert_allclose(result["config"]["atol"], expected, rtol=1e-15)
    assert result["atol"] == 0.1 and result["rtol"] is None and result["config"]["rtol"] is None


def test_tolerance_arguments_accept_none_and_refuse_tols_with_atols_or_rtols(capsys):
    args = launcher.setup_argparse(["--atols", "none", "0.01", "--rtols", "0.1", "NONE"])
    assert args.atols == [None, 0.01] and args.rtols == [0.1, None]
    assert launcher.setup_argparse([]).tols == [0.1, 0.05]
    for argv in (["--tols", "0.1", "--atols", "0.1"], ["--tols", "0.1", "--rtols", "0.1"], ["--atols", "-1"]):
        with pytest.raises(SystemExit) as stop:
            launcher.setup_argparse(argv)
        assert stop.value.code == 2
    assert "--tols cannot be combined" in capsys.readouterr().err


def test_every_atol_rtol_combination_is_run_except_the_one_without_any_tolerance():
    pairs = launcher.tolerance_pairs([0.5], [None, 0.01], [0.1, None], scaled=False)
    assert pairs == [(None, 0.1), (0.01, 0.1), (0.01, None)]
    assert launcher.tolerance_pairs([0.5], None, [0.2], scaled=False) == [(None, 0.2)]
    # --tols alone keeps its meaning: atol = rtol, or atol only where atol is scaled by max|f|.
    assert launcher.tolerance_pairs([0.1, 0.05], None, None, scaled=False) == [(0.1, 0.1), (0.05, 0.05)]
    assert launcher.tolerance_pairs([0.1], None, None, scaled=True) == [(0.1, None)]


@pytest.mark.parametrize(
    "atol,rtol,scaled,tag",
    [(0.1, 0.1, False, "tol0.1"), (0.1, None, True, "tol0.1"), (0.1, 0.1, True, "atol0.1_rtol0.1"),
     (None, 0.05, False, "atolnone_rtol0.05"), (0.01, 0.1, False, "atol0.01_rtol0.1")],
)  # fmt: skip
def test_file_names_keep_tol_only_for_what_tols_runs(atol, rtol, scaled, tag):
    assert launcher._tolerance_tag(atol, rtol, scaled) == tag


def test_an_absolute_only_run_is_recorded_and_summarized(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "RESULTS_DIR", tmp_path)
    exp_dir = launcher.run_problem(
        launcher.rosenbrock(None), dim=2, atol=1.0, rtol=None, error_eval_mode=SurrogateMode.AFFINE_COV_HESS,
        Q_reg=launcher.DEFAULT_Q_REG, coarse_factor=launcher.DEFAULT_COARSE_FACTOR, N_train=2000,
    )  # fmt: skip
    (result_path,) = exp_dir.glob("*.json")
    assert "_atol1.0_rtolnone_" in result_path.name
    result = json.loads(result_path.read_text())
    assert result["atol"] == 1.0 and result["rtol"] is None and "atol_scaled" not in result
    assert result["config"]["atol"] == 1.0 and result["config"]["rtol"] is None
    summary = pd.read_csv(exp_dir / "_summary.csv")
    assert summary["atol"].tolist() == [1.0] and summary["rtol"].isna().all()


def test_cluster_size_arguments_default_to_the_heuristics_and_must_be_positive():
    args = launcher.setup_argparse([])
    assert args.num_initial_clusters is None and args.min_pts_per_cluster is None
    args = launcher.setup_argparse(["--num_initial_clusters", "7", "--min_pts_per_cluster", "60"])
    assert (args.num_initial_clusters, args.min_pts_per_cluster) == (7, 60)
    for argv in (["--num_initial_clusters", "0"], ["--min_pts_per_cluster", "2.5"]):
        with pytest.raises(SystemExit):
            launcher.setup_argparse(argv)


def test_given_cluster_sizes_are_used_named_and_summarized(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "RESULTS_DIR", tmp_path)
    exp_dir = launcher.run_problem(
        launcher.rosenbrock(None), dim=2, atol=0.1, rtol=0.1, error_eval_mode=SurrogateMode.AFFINE_COV_HESS,
        Q_reg=launcher.DEFAULT_Q_REG, coarse_factor=launcher.DEFAULT_COARSE_FACTOR, N_train=2000,
        num_initial_clusters=5, min_pts_per_cluster=60,
    )  # fmt: skip
    (result_path,) = exp_dir.glob("*.json")
    assert "_ninit5_minpts60_" in result_path.name
    result = json.loads(result_path.read_text())
    assert result["config"]["num_initial_clusters"] == 5 and result["config"]["min_pts_per_cluster"] == 60
    summary = pd.read_csv(exp_dir / "_summary.csv")
    assert summary[["num_initial_clusters", "min_pts_per_cluster"]].values.tolist() == [[5, 60]]
