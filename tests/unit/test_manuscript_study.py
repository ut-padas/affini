"""The manuscript study (studies/manuscript): its specification, one case (AFINNI and ART), the runner's decisions and failure classes, and collecting into tables."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from studies.manuscript import case as case_module
from studies.manuscript import collect, run, spec
from studies.manuscript.case import run_case

TINY = {"n0": 3000, "k0": 8, "m": 40}
MACHINE = {"device": "cpu", "distributed_device": "cpu", "mpi_launcher": ["mpiexec"], "max_ranks": 2,
           "case_timeout_h": 4.0, "host_memory_gib": 100.0, "exact_max_dim": None}  # fmt: skip


def test_the_specification_has_every_experiment_once_per_setting():
    cases = spec.cases()
    keys = [c.key for c in cases]
    assert len(keys) == len(set(keys))
    afinni = [c for c in cases if c.method == "afinni"]
    count = {e: sum(e in c.experiments for c in afinni) for e in
             ("convergence", "dimension", "absolute", "gradient", "progression", "distributed", "iterative")}  # fmt: skip
    # 4 functions x 3 d + Lorenz-96 at d = 5, 5 tolerances, 2 modes, 5 seeds; Lorenz-96 at 5 d; ...
    assert count == {"convergence": 650, "dimension": 25, "absolute": 20, "gradient": 25, "progression": 1,
                     "distributed": 96, "iterative": 20}  # fmt: skip
    shared = next(c for c in cases if c.key == "lorenz96_d5_tol0.1_hessian_s0")
    assert set(shared.experiments) == {"convergence", "dimension"}  # one run serves both
    for method in spec.ART_METHODS:  # ART runs AFINNI's hessian cases of the error experiments, nothing else
        art = [c for c in cases if c.method == method]
        twins = {c.key.replace(method, "hessian") for c in art}
        assert twins == {c.key for c in afinni if c.mode == "hessian" and set(c.experiments) & {"convergence", "dimension", "absolute"}}
        assert len(art) == 365 and all(set(c.experiments) <= {"convergence", "dimension", "absolute"} for c in art)
    assert [spec.sizes(d)[1:] for d in (2, 5, 10, 20, 40, 80)] == [(42, 169), (92, 371), (250, 1378), (26, 19004),
                                                                     (16, 31250), (16, 31250)]  # fmt: skip
    absolute = sorted({(c.atol, c.n0) for c in cases if "absolute" in c.experiments}, reverse=True)
    assert absolute == [(579.8, 50000), (289.9, 282843), (144.95, 1600000), (72.475, 9050967)]
    order = sorted(cases, key=spec.cost_order)
    assert all(c.seed == 0 for c in order[: sum(c.seed == 0 for c in cases)])  # every seed 0 first


def test_a_case_records_both_searches_and_the_exact_one_flags_no_more(tmp_path):
    case = spec.Case("quadratic", 2, "hessian", 0, **TINY, tol=0.1, experiments=("convergence",))
    r = run_case(case.as_dict(), tmp_path / "q.json", n_test=4000)
    assert r["status"] == "done" and json.loads((tmp_path / "q.json").read_text())["N_c"] == r["N_c"]
    assert r["exact"]["p_o"] <= r["candidates"]["p_o"]
    assert r["level_stats"] and all("points_start" in lv and "host_rss_gib" in lv for lv in r["level_stats"])
    assert r["provenance"]["commit"] and r["memory"]["host_peak_gib"] > 0
    assert set(r["provenance"]["environment"]) <= set(case_module.ENVIRONMENT)  # no variable that could hold a path

    lorenz = spec.Case("lorenz96", 5, "hessian", 0, 2000, 8, 60, tol=0.1)
    r = run_case(lorenz.as_dict(), tmp_path / "l.json", n_test=2000)
    assert r["status"] == "done" and r["tolerances"]["rtol"] is None and len(r["tolerances"]["atol"]) == 1

    iterative = spec.Case("rosenbrock", 2, "hessian", 0, **TINY, tol=0.05, rounds=2)
    r = run_case(iterative.as_dict(), tmp_path / "i.json", n_test=4000)
    assert [x["N_train"] for x in r["rounds"]] == [3000, 4200, 5400]  # each round adds 0.4 N_0 outside points


def test_an_art_case_routes_every_test_point_and_records_its_leaves(tmp_path):
    from src.art.constants import MAX_DEPTH_LIMIT

    assert spec.ART_MAX_LEVELS == MAX_DEPTH_LIMIT
    for method in spec.ART_METHODS:
        case = spec.Case("rosenbrock", 2, "hessian", 0, **TINY, tol=0.1, method=method, experiments=("convergence",))
        r = run_case(case.as_dict(), tmp_path / f"{method}.json", n_test=3000)
        assert r["status"] == "done" and r["effective_config"]["max_levels"] == spec.ART_MAX_LEVELS and r["art"]
        assert r["exact"] == r["candidates"] and r["exact"]["p_o"] == 0.0  # routing: one leaf per query, nothing outside
        assert r["exact"]["in"]["l2_rel_error"] == r["exact"]["all"]["l2_rel_error"]
        assert sum(r["art"]["terminal_reasons"].values()) == r["N_c"] and 0.0 <= r["art"]["p_unaccepted"] <= 100.0


def test_a_failing_case_writes_its_failure_class_without_this_machines_paths(tmp_path):
    case = spec.Case("nonexistent", 2, "hessian", 0, **TINY, tol=0.1)
    r = run_case(case.as_dict(), tmp_path / "x.json", n_test=100)
    assert r["status"] == "failed" and r["failure"]["class"] == "error" and "nonexistent" in r["failure"]["message"]
    assert str(case_module.ROOT) not in r["failure"]["traceback"] and "<repo>/studies/manuscript/case.py" in r["failure"]["traceback"]
    assert case_module.portable(f"{Path.home()}/x and /work/1/{os.environ.get('USER', 'u')}/y") == "~/x and /work/1/<user>/y"

    class OutOfMemoryError(MemoryError):  # what CuPy raises when the device is full
        pass

    assert case_module.failure_class(OutOfMemoryError()) == "device_out_of_memory"
    assert case_module.failure_class(MemoryError()) == "host_memory_guard"


def _result(case, status="done", wall=10.0, gib=1.0, failure=None):
    return {"case": case.as_dict(), "status": status, "wall_s": wall, "memory": {"host_peak_gib": gib},
            "failure": {"class": failure} if failure else None}  # fmt: skip


def test_the_runner_projects_costs_along_a_series_and_skips_what_will_not_fit():
    series = [spec.Case("rosenbrock", 5, "hessian", 0, *spec.sizes(5), tol=t) for t in spec.TOLERANCES]
    other_seed = spec.Case("rosenbrock", 5, "hessian", 1, *spec.sizes(5), tol=0.1)
    all_cases = [*series, other_seed]
    by_key = {series[0].key: _result(series[0], wall=10, gib=2), series[1].key: _result(series[1], wall=40, gib=8)}
    # the next tolerance repeats the last step's growth, 4x per halving here (0.05 -> 0.02 is 1.32 steps)
    wall, gib = run.project(series[2], by_key, all_cases)
    assert wall == pytest.approx(40 * 4 ** (1.3219), rel=1e-3) and gib == pytest.approx(8 * 4 ** 1.3219, rel=1e-3)
    assert run.project(other_seed, by_key, all_cases) == (10, 2)  # another seed of the same case
    small = {**MACHINE, "host_memory_gib": 20.0}
    state, reason, _ = run.decide(series[2], small, by_key, all_cases, force=False)
    assert state == "skip" and "GiB" in reason
    assert run.decide(series[2], small, by_key, all_cases, force=True)[0] == "run"
    by_key[series[2].key] = _result(series[2], "failed", failure="host_memory_guard")
    state, reason, _ = run.decide(series[3], MACHINE, by_key, all_cases, force=False)
    assert state == "skip" and "host_memory_guard" in reason  # a looser tolerance ran out of memory
    wide = spec.Case("rosenbrock", 5, "hessian", 0, *spec.sizes(5), tol=0.05, ranks=8)
    assert run.decide(wide, MACHINE, {}, [wide], force=False)[0] == "skip"  # more ranks than the machine


@pytest.mark.parametrize(("script", "timeout_h", "klass"), [
    ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", 1.0, "killed"),
    ("import sys; sys.exit(1)", 1.0, "crashed"),
    ("import time; time.sleep(60)", 1 / 3600, "timeout"),
])  # fmt: skip
def test_the_runner_records_a_case_that_could_not_record_itself(tmp_path, script, timeout_h, klass):
    case = spec.Case("quadratic", 2, "hessian", 0, **TINY, tol=0.1)

    def launch(cmd, **kw):
        assert kw["stdin"] is subprocess.DEVNULL  # a launcher that forwards stdin must see nothing
        return subprocess.Popen([sys.executable, "-c", script], **kw)

    r = run.execute(case, {**MACHINE, "case_timeout_h": timeout_h}, tmp_path, launch=launch)
    assert r["status"] == "failed" and r["failure"]["class"] == klass
    assert json.loads(run.result_path(tmp_path, case).read_text())["failure"]["class"] == klass


def test_collect_builds_tables_with_the_runs_without_hessian_splits_and_art_beside(tmp_path):
    for mode, method in (("hessian", "afinni"), ("no_hessian", "afinni"), ("hessian", "art_affine")):
        for seed in (0, 1):
            case = spec.Case("quadratic", 2, mode, seed, **TINY, tol=0.1, method=method, experiments=("convergence",))
            run_case(case.as_dict(), run.result_path(tmp_path, case), n_test=3000)
    old = json.loads(run.result_path(tmp_path, case).read_text())  # a result written before the study ran ART
    old["case"].pop("method")
    old["case"]["key"] = "older"
    (tmp_path / "cases" / "older.json").write_text(json.dumps(old))
    collect.main(["--results", str(tmp_path)])
    with (tmp_path / "data" / "tables" / "convergence.csv").open() as fh:
        afinni, art = list(csv.DictReader(fh))
    assert afinni["method"] == "afinni" and afinni["N_c.seeds"] == "3" and afinni["e_nh.seeds"] == "2" and afinni["p_o"]
    assert float(afinni["N_c.min"]) <= float(afinni["N_c"]) <= float(afinni["N_c.max"])
    assert art["method"] == "art_affine" and art["N_c.seeds"] == "2" and art["e"] and afinni["e_in"]
    assert art["p_o"] == art["p_o_candidates"] == art["e_in"] == art["e_nh"] == "" and art["p_o.seeds"] == "0"  # printed as "–"
    with (tmp_path / "data" / "cases.csv").open() as fh:
        cases = list(csv.DictReader(fh))
    assert len(cases) == 7 and all(c["status"] == "done" and c["exact.e_in"] for c in cases)
    assert {c["method"] for c in cases if c["key"] == "older"} == {"afinni"}
    assert all(c["art.accepted"] for c in cases if c["method"] == "art_affine")


def test_the_smoke_run_keeps_the_two_loosest_tolerances_of_each_series():
    small = run.smoke_cases(spec.cases())
    assert {c.n0 for c in small} == {run.SMOKE["n0"]} and max(c.ranks for c in small) == 2
    assert {c.tol for c in small if "convergence" in c.experiments} == {0.1, 0.05}
    assert {c.tol for c in small if "distributed" in c.experiments} == {0.05, 0.005}  # a series of two
    assert sorted({c.atol for c in small if "absolute" in c.experiments}) == [289.9, 579.8]


def test_a_smoke_run_goes_through_the_runner_end_to_end(tmp_path):
    machine = tmp_path / "m.toml"
    machine.write_text('device = "cpu"\nmpi_launcher = ["mpiexec"]\n')
    run.main(["run", "--smoke", "--machine", str(machine), "--results", str(tmp_path / "out"), "--seeds", "0",
              "--only", r"^(gaussian_d2_tol0\.1_hessian|wave_d5_tol0\.1_exact_gradient)"])  # fmt: skip
    results = sorted(Path(tmp_path / "out" / "cases").glob("*.json"))
    assert len(results) == 2 and all(json.loads(p.read_text())["status"] == "done" for p in results)


def test_every_distributed_case_runs_on_cpu_ranks_with_one_thread_each(tmp_path):
    one = spec.Case("rosenbrock", 5, "hessian", 0, *spec.sizes(5, 10**5), tol=0.05, experiments=("distributed",))
    eight = spec.Case("rosenbrock", 5, "hessian", 0, *spec.sizes(5, 10**5), tol=0.05, ranks=8, experiments=("distributed",))
    gpu = spec.Case("rosenbrock", 5, "hessian", 0, *spec.sizes(5), tol=0.05, experiments=("convergence",))
    machine = {**MACHINE, "device": "gpu", "max_ranks": 128}
    for case in (one, eight):
        cmd, env = run.command(case, machine, tmp_path / "x.json")
        assert env["SURROGATE_DEVICE"] == "cpu" and env["OMP_NUM_THREADS"] == env["NUMBA_NUM_THREADS"] == "1"
    assert run.command(eight, machine, tmp_path / "x.json")[0][:3] == ["mpiexec", "-n", "8"]
    cmd, env = run.command(gpu, machine, tmp_path / "x.json")
    assert env["SURROGATE_DEVICE"] == "gpu" and cmd[0] == sys.executable and env["SURROGATE_MPI"] == "off"
