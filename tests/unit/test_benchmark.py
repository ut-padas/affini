"""The benchmark's store (src/benchmark/store.py) and sweep (src/benchmark/sweep.py): where runs go, and which cases a sweep runs."""

import json

import numpy as np
import pytest

from src.benchmark import functions, step_maps, sweep
from src.benchmark.store import Store, write_json


@pytest.fixture
def store(tmp_path):
    return Store("functions", "cpu-1rank", root=tmp_path / "results", figures_root=tmp_path / "figures")


def test_a_run_is_one_record_with_its_arrays_under_suite_setup_problem_d_method(store):
    path = store.write({"problem": "rosenbrock", "d": 2, "method": "chebtt", "tol": 0.01, "seed": 1, "evals": 10}, {"pred": np.arange(3.0)})
    assert path == store.root / "runs" / "functions" / "cpu-1rank" / "rosenbrock" / "d2" / "chebtt" / "tol0.01-s1.json"
    record = store.read("rosenbrock", 2, "chebtt", 0.01, 1)
    assert record["evals"] == 10 and record["suite"] == "functions" and record["setup"] == "cpu-1rank"
    assert {"recorded", "commit"} <= set(record)
    assert np.array_equal(store.arrays("rosenbrock", 2, "chebtt", 0.01, 1)["pred"], np.arange(3.0))
    assert [r["method"] for r in store.runs(problem="rosenbrock")] == ["chebtt"]
    assert store.read("rosenbrock", 2, "chebtt", 0.1, 1) is None and store.arrays("rosenbrock", 2, "chebtt", 0.1, 1) == {}


def test_a_record_is_written_whole_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "case" / "record.json"
    write_json(path, {"worst": 0.5})
    write_json(path, {"worst": 0.25})
    assert json.loads(path.read_text()) == {"worst": 0.25} and [p.name for p in path.parent.iterdir()] == [path.name]


def test_slice_directions_are_kept_with_the_problems_data(store, monkeypatch):
    monkeypatch.setattr(functions, "store", lambda: store)
    W = functions.directions("rosenbrock", 3)
    assert W.shape == (3, 2)
    assert [p.name for p in (store.root / "data" / "functions" / "rosenbrock").iterdir()] == ["d3-directions.npz"]


class _NoRedraw:
    def __init__(self, log):
        pass

    def start(self, argv):
        pass

    def wait(self):
        pass


def test_a_step_map_sweep_runs_the_loosest_tolerance_of_one_regime_from_a_dimension(monkeypatch, tmp_path):
    cases = []
    monkeypatch.setattr(step_maps, "STORE", Store("step_maps", "cpu-1rank", root=tmp_path, figures_root=tmp_path))
    for name in ("data", "solver_cost", "dynamics"):
        monkeypatch.setattr(step_maps, name, lambda n: None)
    monkeypatch.setattr(sweep, "run_job", lambda argv, limit, log, env=None: cases.append(argv) or False)  # every case over the limit
    monkeypatch.setattr(sweep, "Redrawer", _NoRedraw)
    sweep.main(["step_maps", "--methods", "chebtt_roll", "--min-d", "10", "--max-d", "20", "--regimes", "stiff", "--loosest", "--limit", "1"])
    names = [argv[4] for argv in cases]
    assert {argv[8] for argv in cases} == {"0.01"}
    assert all(step_maps.dim(n) >= 10 and step_maps.STEP_MAPS[n].regime == "stiff" for n in names)
    assert "primordial" in names and "pollu" in names and "bruss_d20" not in names  # stuck at bruss_d10: d20 skipped
    assert all("failed" in r for r in step_maps.STORE.runs())


def test_a_function_sweep_stops_at_the_first_failed_tolerance_and_skips_higher_d(monkeypatch, tmp_path):
    cases = []
    monkeypatch.setattr(functions, "store", lambda: Store("functions", "cpu-2rank", root=tmp_path, figures_root=tmp_path))
    monkeypatch.setattr(sweep, "run_job", lambda argv, limit, log, env=None: cases.append(argv) or False)
    monkeypatch.setattr(sweep, "Redrawer", _NoRedraw)
    sweep.main(["functions", "--problems", "rosenbrock", "--d", "2,3", "--methods", "chebtt", "--ranks", "2", "--limit", "1"])
    assert len(cases) == 1  # the loosest tolerance at 2-D failed: no tighter one, and no 3-D
    assert cases[0][cases[0].index("--tols") + 1] == "0.1" and cases[0][cases[0].index("--d") + 1] == "2"


def test_since_reruns_older_records_with_force_and_keeps_newer_ones(monkeypatch, tmp_path):
    store = Store("functions", "cpu-2rank", root=tmp_path, figures_root=tmp_path)
    for tol in (0.1, 0.01, 0.001):
        store.write({"problem": "rosenbrock", "d": 2, "method": "chebtt", "tol": tol, "seed": 1, "evals": 1})
    old = store.run_path("rosenbrock", 2, "chebtt", 0.1, 1)
    old.write_text(old.read_text().replace('"recorded": "', '"recorded": "1999'))  # the loosest: written by older code
    for s in (2, 3):
        for tol in (0.1, 0.01, 0.001):
            store.write({"problem": "rosenbrock", "d": 2, "method": "chebtt", "tol": tol, "seed": s, "evals": 1})
    cases = []
    monkeypatch.setattr(functions, "store", lambda: store)
    monkeypatch.setattr(sweep, "run_job", lambda argv, limit, log, env=None: cases.append(argv) or True)
    monkeypatch.setattr(sweep, "Redrawer", _NoRedraw)
    sweep.main(["functions", "--problems", "rosenbrock", "--d", "2", "--methods", "chebtt", "--ranks", "2", "--since", "2000-01-01T00:00"])
    assert len(cases) == 1 and cases[0][-1] == "--force" and cases[0][cases[0].index("--tols") + 1] == "0.1"
