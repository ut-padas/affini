"""The launcher's result JSON is the single file per run: it feeds the summary CSV and the profile plots."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import afinni_launcher as launcher  # noqa: E402
from src import plotting as P  # noqa: E402
from src.adaptive_clustering_utils import SurrogateMode  # noqa: E402


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
