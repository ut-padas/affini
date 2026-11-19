"""What the schemes share: settings, the oracle's counts, the training report, and saving and loading models."""

import dataclasses

import numpy as np
import pytest

from src.problems import FUNCTIONS
from src.schemes import base, chebtree, hub, kernel, sparsegrid, tabulation
from src.schemes.oracle import Oracle, distributed
from src.schemes.profiling import PHASES


def small(name, d):
    p = FUNCTIONS[name]
    lo, hi = p.domain(d)
    return p.values, lo, hi


def trained():
    """One small model of each scheme converted to the shared contract, with the points to predict at."""
    f, lo, hi = small("genz_gaussian", 2)
    rng = np.random.default_rng(3)
    yield "chebtree", chebtree.build(f, lo, hi, 0.01), rng.uniform(lo, hi, (50, 2))
    yield "chebtree_rotated", chebtree.build_rotated(f, lo, hi, 0.01, seed=1), rng.uniform(lo, hi, (50, 2))
    yield "sparsegrid", sparsegrid.build(f, lo, hi, 0.01, seed=1), rng.uniform(lo, hi, (50, 2))
    yield "kernel", kernel.build(f, lo, hi, 0.05, seed=1, max_evals=3000), rng.uniform(lo, hi, (50, 2))
    g, glo, ghi = small("shared_latent", 3)
    yield "hub", hub.build(g, glo, ghi, 0.1, n0=300, max_evals=2000), rng.uniform(glo, ghi, (50, 3))
    P = np.diag([0.9, 0.5])
    Z = rng.uniform(-1, 1, (60, 2))
    yield "tabulation", tabulation.build(lambda X: X @ P.T, Z, 1e-3, seed=1), rng.uniform(-1, 1, (50, 2))


@pytest.mark.parametrize("name, model, X", list(trained()), ids=lambda v: v if isinstance(v, str) else "")
def test_a_saved_model_loads_and_predicts_bit_for_bit(tmp_path, name, model, X):
    path = base.save(model, tmp_path / name)
    loaded = base.load(path)
    assert type(loaded) is type(model)
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert loaded.evals == model.evals
    assert loaded.report.scheme == model.report.scheme and loaded.report.rounds == model.report.rounds
    if hasattr(model, "refuses"):
        assert np.array_equal(loaded.refuses(X), model.refuses(X))


@pytest.mark.parametrize("name, model, X", list(trained()), ids=lambda v: v if isinstance(v, str) else "")
def test_every_training_reports_its_evaluations_phases_and_rounds(name, model, X):
    r = model.report
    assert r.evals == model.evals
    assert set(r.phases) == set(PHASES) and r.seconds >= sum(r.phases.values()) - 1e-9
    assert r.rounds and all("round" in rec and "evals" in rec for rec in r.rounds)
    assert 0.0 <= r.oracle_seconds <= r.seconds


def test_settings_take_overrides_and_reject_another_schemes_settings():
    s = sparsegrid.SparseGridSettings.resolve(None, {"seed": 3})
    assert s.seed == 3 and s.safety == sparsegrid.SparseGridSettings().safety
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.seed = 4
    with pytest.raises(TypeError):
        sparsegrid.SparseGridSettings.resolve(kernel.KernelSettings(), {})


def test_the_oracle_counts_points_jacobians_and_time():
    oracle = Oracle(lambda X: X.sum(axis=1), jacobian=lambda X: np.ones_like(X))
    oracle(np.zeros((5, 3)))
    oracle.jacobian(np.zeros((2, 3)))
    assert (oracle.evals, oracle.jacobians) == (5, 2) and oracle.seconds >= 0.0
    with pytest.raises(ValueError):
        Oracle(np.sum).jacobian(np.zeros((1, 3)))


def test_a_scheme_given_an_oracle_counts_on_it():
    f, lo, hi = small("genz_gaussian", 2)
    oracle = Oracle(f)
    model = sparsegrid.build(oracle, lo, hi, 0.01, seed=1)
    assert oracle.evals == model.evals == sparsegrid.build(f, lo, hi, 0.01, seed=1).evals


def test_one_rank_distributes_to_a_plain_oracle():
    with distributed(np.sum) as oracle:
        assert type(oracle) is Oracle


def test_saving_refuses_what_it_cannot_rebuild(tmp_path):
    @dataclasses.dataclass
    class Foreign:
        x: int = 1

    with pytest.raises(TypeError):
        base.save(Foreign(), tmp_path / "foreign")
