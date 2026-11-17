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


def test_every_case_runs_with_the_coarse_factor_its_results_are_collected_by():
    import collect_paper
    import paper_values

    for case_id, group, _tier, _np, _seed, kind, args in cases.full_matrix([0, 1], [1, 2]):
        if kind != "launcher" or group in ("conv_k10", "mpi"):  # a diagnostic, and not collected by matching
            continue
        d = int(args[args.index("--dims") + 1])
        assert int(args[args.index("--coarse_factor") + 1]) == cases.group_coarse_factor(group, d), case_id

    rec = next(r for r in paper_values.records() if r["group"] == "wave_exact" and r["mode"].endswith("exact_grad"))
    run = {"problem": "wave", "dim": 5, "mode": rec["mode"], "mpi_size": 1, "atol": rec["tol"], "rtol": rec["tol"], "N_0": 1}
    assert collect_paper.matching([{**run, "coarse_factor": 50}], rec) != []
    assert collect_paper.matching([{**run, "coarse_factor": 10}], rec) == []
