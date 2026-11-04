"""Gap sampling must retry empty batches and stop at its configured limit."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    # Importing the script should not replace pytest's logging handlers.
    with monkeypatch.context() as patch:
        patch.setattr("src.logging_utils.setup_logging", lambda: None)
        module = importlib.import_module("p_out_iterative_experiment")
    monkeypatch.setattr(module, "xp", np)
    return module


@pytest.fixture
def surrogate():
    return SimpleNamespace(X_src=np.zeros((2, 2)))


def generate(experiment, surrogate, strategy="halo", **kwargs):
    return experiment.generate_new_points(
        surrogate, 10, strategy, 0.4, [-3, 3], 1.5, **kwargs
    )


@pytest.mark.parametrize("strategy", ["halo", "cloud", "uniform", "chord"])
def test_empty_batches_warn_then_fail_at_default_limit(
    experiment, surrogate, monkeypatch, strategy
):
    sampler = Mock(return_value=np.empty((0, 2)))
    monkeypatch.setattr(experiment, f"{strategy}_sampling", sampler)
    with (
        pytest.warns(RuntimeWarning, match="produced no gap points") as caught,
        pytest.raises(RuntimeError, match="100 sampling attempts: collected 0 of 4"),
    ):
        generate(experiment, surrogate, strategy)
    assert sampler.call_count == len(caught) == 100
    assert "attempt 1/100" in str(caught[0].message)
    assert "attempt 100/100" in str(caught[-1].message)


def test_retry_can_succeed_on_last_attempt(experiment, surrogate, monkeypatch):
    points = np.arange(8).reshape(4, 2)
    sampler = Mock(side_effect=[np.empty((0, 2)), points[:2], points[2:]])
    monkeypatch.setattr(experiment, "halo_sampling", sampler)
    with pytest.warns(RuntimeWarning, match="attempt 1/3") as caught:
        actual = generate(experiment, surrogate, max_sampling_attempts=3)
    np.testing.assert_array_equal(actual, points)
    assert sampler.call_count == 3
    assert len(caught) == 1


def test_productive_attempts_also_count_toward_limit(
    experiment, surrogate, monkeypatch
):
    sampler = Mock(return_value=np.ones((1, 2)))
    monkeypatch.setattr(experiment, "halo_sampling", sampler)
    with pytest.raises(RuntimeError, match="2 sampling attempts: collected 2 of 4"):
        generate(experiment, surrogate, max_sampling_attempts=2)
    assert sampler.call_count == 2


def test_completed_batch_is_trimmed_to_target(experiment, surrogate, monkeypatch):
    sampler = Mock(return_value=np.ones((9, 2)))
    monkeypatch.setattr(experiment, "halo_sampling", sampler)
    result = generate(experiment, surrogate, max_sampling_attempts=1)
    assert result.shape == (4, 2)
    assert sampler.call_count == 1


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_attempt_limit_fails_before_sampling(
    experiment, surrogate, monkeypatch, limit
):
    sampler = Mock()
    monkeypatch.setattr(experiment, "halo_sampling", sampler)
    with pytest.raises(ValueError, match="at least 1"):
        generate(experiment, surrogate, max_sampling_attempts=limit)
    sampler.assert_not_called()


@pytest.mark.parametrize("strategy", ["halo", "cloud"])
def test_retries_use_new_candidates_and_remain_reproducible(
    experiment, monkeypatch, strategy
):
    model = SimpleNamespace(
        X_src=np.zeros((1, 2)),
        Qs_src=np.eye(2)[None, :, :],
        filter_unguaranteed_local=lambda points, **kwargs: np.ones(
            len(points), dtype=bool
        ),
    )
    sampler = getattr(experiment, f"{strategy}_sampling")
    args = (model, 8, 1.5, -100, 100) if strategy == "halo" else (model, 8, -100, 100)
    # Preserve NumPy's process-wide RNG state for unrelated tests.
    state = np.random.get_state()  # noqa: NPY002
    try:
        np.random.seed(0)  # noqa: NPY002
        first = sampler(*args)
        second = sampler(*args)
        assert not np.array_equal(first, second)
        np.random.seed(0)  # noqa: NPY002
        np.testing.assert_array_equal(sampler(*args), first)
    finally:
        np.random.set_state(state)  # noqa: NPY002


def test_cli_attempt_limit(experiment, monkeypatch):
    monkeypatch.setattr("sys.argv", ["p_out_iterative_experiment.py"])
    assert experiment.parse_args().max_sampling_attempts == 100
    monkeypatch.setattr(
        "sys.argv", ["p_out_iterative_experiment.py", "--max-sampling-attempts", "7"]
    )
    assert experiment.parse_args().max_sampling_attempts == 7
    monkeypatch.setattr(
        "sys.argv", ["p_out_iterative_experiment.py", "--max-sampling-attempts", "0"]
    )
    with pytest.raises(SystemExit) as error:
        experiment.parse_args()
    assert error.value.code == 2
