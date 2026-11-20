"""The shared records of long runs (src.infra.records): resuming, staleness, upgrades and the probes."""

import json
import time

import numpy as np

from src.infra import records
from src.infra.records import RecordStore


def test_a_case_is_computed_once_and_resumed_from_its_record(tmp_path):
    store = RecordStore(tmp_path)
    calls = []

    def compute(tol):
        calls.append(tol)
        return {"tol": tol, "evals": np.int64(7), "worst": np.float64(0.5)}

    key = {"method": "chebtt", "data": "abc", "tol": repr(0.1)}
    first = store.get(key, compute, 0.1)
    again = RecordStore(tmp_path).get(key, compute, 0.1)
    assert calls == [0.1] and first == again == {"tol": 0.1, "evals": 7, "worst": 0.5}
    record = store.read(key)
    assert record["key"] == key and record["status"] == "done" and record["schema"] == records.SCHEMA
    assert {"commit", "code", "cpu", "libraries", "environment"} <= set(record["provenance"])


def test_a_record_older_than_its_methods_last_change_is_computed_again(tmp_path):
    store = RecordStore(tmp_path)
    key = {"method": "art", "tol": "0.1"}
    store.get(key, lambda: {"evals": 1})
    path = store.path(key)
    path.write_text(path.read_text().replace('"recorded": "', '"recorded": "1999'))
    later = {"art": (time.strftime("%Y-%m-%dT%H:%M:%S"), "a change")}
    assert store.get(key, lambda: {"evals": 2}, changed={"afinni": later["art"]}) == {"evals": 1}  # another method's change
    assert store.get(key, lambda: {"evals": 2}, changed=later) == {"evals": 2}
    assert store.get(key, lambda: {"evals": 3}, changed=later) == {"evals": 2}


def test_an_unreadable_record_counts_as_missing(tmp_path):
    store = RecordStore(tmp_path)
    key = {"method": "hub"}
    store.path(key).parent.mkdir(parents=True, exist_ok=True)
    store.path(key).write_text('{"evals": 1')  # cut short by something other than write_json
    assert store.read(key) is None and store.get(key, lambda: {"evals": 4}) == {"evals": 4}


def test_a_first_schema_record_is_upgraded_with_what_it_recorded():
    old = {"method": "chebtt", "failed": "over the limit", "commit": "abc1234", "recorded": "2026-11-18T10:00:00"}
    new = records.upgrade(old)
    assert new["schema"] == records.SCHEMA and new["status"] == "failed" and new["recorded"] == old["recorded"]
    assert new["provenance"]["describe"] == "abc1234" and new["provenance"]["cpu"] is None
    assert records.upgrade(new) is new


def test_failure_keeps_the_traceback_without_this_machines_paths():
    try:
        raise ValueError(f"bad file {records.PROJECT_DIR}/x.json")
    except ValueError as err:
        info = records.failure(err)
    assert info["type"] == "ValueError" and "<repo>/x.json" in info["message"]
    assert str(records.PROJECT_DIR) not in info["traceback"] and "ValueError" in info["traceback"]


def test_the_probes_measure_time_memory_and_the_oracle():
    f, total = records.timed_oracle(lambda X: X.sum(axis=1))
    with records.Probe() as probe:
        big = np.ones((4000, 4000))  # 122 MiB, held while the sampler runs
        time.sleep(0.1)
        f(big[:10])
        f(big[:20])
        del big
    assert probe.seconds >= 0.1 and probe.peak_mb is not None and probe.peak_mb > 60
    assert total["calls"] == 2 and total["s"] >= 0
    out, us = records.per_query_us(lambda X: X * 2, np.ones((1000, 3)))
    assert out.shape == (1000, 3) and us > 0


def test_saved_bytes_is_the_file_size_or_none_when_saving_fails():
    def save(model, path):
        np.savez(path, a=model)
        return path

    def broken(model, path):
        raise TypeError("cannot save")

    assert records.saved_bytes(save, np.zeros(100)) > 800
    assert records.saved_bytes(broken, np.zeros(100)) is None


def test_error_stats_and_json_of_numpy_values(tmp_path):
    s = records.error_stats(np.array([0.5, 0.9, 2.0, 0.1]))
    assert s["worst"] == 2.0 and s["within"] == 75.0 and s["mean"] == 0.875
    assert records.error_stats(np.array([]))["worst"] is None
    records.write_json(tmp_path / "r.json", {"n": np.int32(3), "x": np.float32(0.5), "p": tmp_path})
    assert json.loads((tmp_path / "r.json").read_text()) == {"n": 3, "x": 0.5, "p": str(tmp_path)}
