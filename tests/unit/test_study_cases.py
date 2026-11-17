"""The manuscript study's run matrix (studies/convergence/paper/cases.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies" / "convergence" / "paper"))
import cases  # noqa: E402


def test_figure8_exact_gradient_runs_use_the_coarse_factor_of_the_lsq_runs_beside_them():
    def coarse(row):
        args = row[6]
        return int(args[args.index("--coarse_factor") + 1])

    rows = cases.full_matrix([0], [1])
    exact = [r for r in rows if r[1] == "wave_exact"]
    lsq = [r for r in rows if r[0].startswith("conv_wave_d5_")]
    assert len(exact) == 5 and len(lsq) == 5
    assert {coarse(r) for r in exact} == {coarse(r) for r in lsq} == {50}
