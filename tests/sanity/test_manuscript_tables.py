"""The companion appendix's manuscript tables (docs/reports/adaptive-surrogates/figures/manuscript_tables.py) read records as record_results.py writes them, and Table 2's sizes."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "adaptive-surrogates" / "figures"))
sys.path.insert(0, str(ROOT / "studies" / "sanity" / "paper"))

import cases  # noqa: E402
import manuscript_tables as mt  # noqa: E402
import paper_values as pv  # noqa: E402
import record_results as rr  # noqa: E402


def write_record(path: Path, drop: tuple[str, str] | None = None, scale: float = 1.0, only: str | None = None,
                 commit: str = "0123456789abcdef") -> None:
    """
    A record in record_results.py's format whose every value is the manuscript's own times ``scale``,
    except ``drop`` (unmeasured); ``only`` keeps the records of one table.
    """
    comparison = []
    for rec in pv.records():
        if only is not None and not rec["ref"].startswith(only):
            continue
        for q in rr.QUANTITIES:
            if q in rec:
                value = float("nan") if drop == (rec["ref"], q) else rec[q] * scale
                comparison.append({"ref": rec["ref"], "loc": rec["loc"], "problem": rec["problem"], "dim": rec["dim"],
                                   "tol": rec.get("tol", rec.get("atol")), "quantity": q, "paper": rec[q],
                                   "rerun_seed0": value, "n_seeds": 1, "seed_min": value, "seed_max": value,
                                   "rel_diff": 0.0})  # fmt: skip
    path.mkdir()
    rr.write_csv(path / "manuscript_results.csv", rr.results_rows(comparison))
    rr.write_csv(path / "cost.csv", [{"case": "mpi_np2_tol0.005", "seed": 0, "wall_s": 887.0, "peak_rss_gib_rank0": 1.0}])
    prov = {"commit": commit, "recorded": "2026-11-17T12:00:00", "machine": {"label": "test node"}, "gpu_runs": "1 of 1"}
    (path / "provenance.json").write_text(json.dumps(prov))


def test_a_record_of_the_manuscripts_own_values_reproduces_its_tables(tmp_path):
    write_record(tmp_path / "record")
    values, walls, abouts = mt.read_records([tmp_path / "record"])
    assert abouts == [{"name": "record", "commit": "0123456", "recorded": "2026-11-17", "machine": "test node", "gpu_runs": "1 of 1"}]
    by_key = {t["key"]: t for t in mt.tables(values, walls)}
    assert all(t["sources"] == ([] if t["key"] == "table-5" else [0]) for t in by_key.values())
    def cells(rows):
        return [r.get("cells", r.get("group")) for r in rows]

    for key in ("table-1a", "table-1b", "table-2", "table-3", "figure-8"):
        assert cells(by_key[key]["rerun"]) == cells(by_key[key]["paper"]), key
        assert by_key[key]["differences"] == [] and not any(m for r in by_key[key]["rerun"] for m in r.get("marks", []))
    for paper, rerun in zip(by_key["table-4"]["paper"], by_key["table-4"]["rerun"], strict=True):
        if "cells" in paper and paper["cells"][1] == "AFINNI":
            assert rerun["cells"][:4] == paper["cells"][:4] and rerun["cells"][4] == mt.MISSING
        elif "cells" in paper:
            assert rerun["cells"][2:] == [mt.MISSING] * 3
    assert all(c == mt.MISSING for r in by_key["table-5"]["rerun"] for c in r["cells"][1:])
    dynamic = {r["cells"][0]: r["cells"][3] for r in by_key["table-6"]["rerun"]}
    assert dynamic["2"] == "887.0" and dynamic["1"] == mt.MISSING


def test_a_number_the_record_lacks_is_a_dash_and_no_record_leaves_every_rerun_cell_empty(tmp_path):
    write_record(tmp_path / "record", drop=("Table 1a #3", "e_nh"))
    rows = mt.tables(*mt.read_records([tmp_path / "record"])[:2])[0]["rerun"]
    row3 = next(r["cells"] for r in rows if r.get("cells", [""])[0] == "3")
    assert row3[-1] == mt.MISSING and row3[2] == mt.sci(2.91e6)
    for t in mt.tables({}, {}):
        measured = [c for r in t["rerun"] for c in r.get("cells", [])]
        assert mt.MISSING in measured and t["sources"] == []
    assert mt.read_records([]) == ({}, {}, [])


def test_a_later_record_replaces_the_tables_it_holds_and_is_named_as_their_source(tmp_path):
    write_record(tmp_path / "main")
    write_record(tmp_path / "table2", scale=2.0, only="Table 2", commit="fedcba9876543210")
    values, walls, abouts = mt.read_records([tmp_path / "main", tmp_path / "table2"])
    assert [a["commit"] for a in abouts] == ["0123456", "fedcba9"]
    by_key = {t["key"]: t for t in mt.tables(values, walls)}
    assert by_key["table-2"]["sources"] == [1] and by_key["table-1a"]["sources"] == [0]
    first = by_key["table-2"]["rerun"][0]["cells"]
    assert first[3] == mt.count(2 * pv._LORENZ_DIMS[0][3])  # N_c from the supplementary record
    assert [r.get("cells") for r in by_key["table-1b"]["rerun"]] == [r.get("cells") for r in by_key["table-1b"]["paper"]]
    # doubling every Table 2 number: worse for counts and errors, beyond the 25% band
    assert {d["mark"] for d in by_key["table-2"]["differences"]} == {"worse"}


def test_table2_is_sized_as_the_code_that_produced_it_and_says_so():
    # cap_initial_augmentation_budget at 91ddd97, applied to the uncapped 1.3^d heuristics
    assert [cases.lorenz_dims_sizes(d)[1:] for d in (5, 10, 20, 40, 80)] == [
        (92, 371), (250, 1378), (26, 19004), (16, 31250), (16, 31250)]
    note = next(t for t in mt.tables({}, {}) if t["key"] == "table-2")["note"]
    assert "$K_0 = $ 16, $M = $ 31,250" in note and "$K_0 <= 1000$" in note


def test_cells_follow_the_manuscripts_formats():
    assert [mt.sci(2.32e5), mt.sci(7.43e-2), mt.sci(3e-2, 1), mt.sci(None)] == ["2.32E5", "7.43E-2", "3E-2", mt.MISSING]
    assert [mt.count(5772), mt.count(1.25e5), mt.pct(0.0), mt.pct(16.9), mt.pct(20.0, "%")] == ["5,772", "1.25E5", "0.00", "16.9", "20.0%"]
    assert [mt.tol(0.05), mt.tol(0.005)] == ["0.05", "5E-3"]


def test_a_difference_beyond_the_noise_band_is_marked_better_or_worse():
    assert mt.compare("e", 1.0, 1.2) is None and mt.compare("e", 1.0, 1.3) == "worse"
    assert mt.compare("N_c", 100, 70) == "better"
    assert mt.compare("p_o", 0.3, 1.2) is None and mt.compare("p_o", 10.0, 2.0) == "better"  # 1-point floor
    assert mt.compare("p_in_good", 50.0, 96.7) == "better" and mt.compare("e", 1.0, None) is None
    lines = mt.differences([("#1", "e", 1.0, 2.0, "worse"), ("#2", "e", 1.0, 3.0, "worse"), ("#3", "N", 10.0, 2.0, "better")])
    assert lines == [{"mark": "worse", "text": "$e$ 2–3× the manuscript's (#1, #2)", "cause": "Not analysed."},
                     {"mark": "better", "text": "$N$ 0.2× the manuscript's (#3)", "cause": "Not analysed."}]
    # a difference the lab has analysed carries its note
    (known,) = mt.differences([("#36", "e", 8.82e-2, 9.4e-3, "better")], "table-1b")
    assert known["cause"] == mt.CAUSES[("table-1b", "e", "better")] and "typo" in known["cause"]
