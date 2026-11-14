"""``initial_clustering`` gives exactly the labels training starts from, so callers can compute them once."""

import dataclasses
from importlib.util import find_spec

import numpy as np
import pytest

import src.engine.driver as driver
from src.config import TrainingConfig
from tests._helpers import host, train_surrogate

X = np.random.default_rng(4).uniform(-1, 1, (3_000, 2))
OPTIONS = {"atol": 0.05, "rtol": 0.05, "num_initial_clusters": 9, "min_pts_per_cluster": 40, "use_work_stealing": False, "seed": 5}
needs_knndbscan = pytest.mark.skipif(find_spec("knndbscan") is None, reason="needs the knndbscan package")


def f(x):
    return 1 + x[:, 0] ** 2 + 3 * x[:, 1] ** 2


@pytest.mark.parametrize("method", ["kmeans", "uniform_grid", pytest.param("knndbscan", marks=needs_knndbscan)])
def test_labels_equal_those_training_starts_from(monkeypatch, method):
    options = {**OPTIONS, "initial_clustering_method": method}
    used, original = [], driver.initial_clustering

    def recording(*args, **kwargs):
        used.append(original(*args, **kwargs))
        return used[-1]

    monkeypatch.setattr(driver, "initial_clustering", recording)
    train_surrogate(X, f_fn=f, **options)
    monkeypatch.undo()

    labels = driver.initial_clustering(X, TrainingConfig(**options))
    assert len(used) == 1 and labels.dtype == np.int64 and labels.shape == (len(X),)
    np.testing.assert_array_equal(labels, used[0])
    # kNN-DBSCAN's fixed eps=0.5 joins uniformly spread points into one cluster
    assert len(np.unique(labels)) > 1 or method == "knndbscan"


@needs_knndbscan
def test_knndbscan_separates_blobs_farther_apart_than_eps():
    rng = np.random.default_rng(7)
    blobs = np.concatenate([rng.normal(0.0, 0.05, (400, 2)), rng.normal(3.0, 0.05, (400, 2))])
    labels = driver.initial_clustering(blobs, TrainingConfig(atol=0.1, min_pts_per_cluster=20, initial_clustering_method="knndbscan"))
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.repeat([0, 1], 400))


def test_unknown_method_is_rejected():
    config = dataclasses.replace(TrainingConfig(atol=0.1), initial_clustering_method="spectral")
    with pytest.raises(ValueError, match="spectral"):
        driver.initial_clustering(X, config)


def test_given_labels_give_the_same_model_without_clustering_again(monkeypatch):
    computed = train_surrogate(X, f_fn=f, **OPTIONS)
    labels = driver.initial_clustering(X, TrainingConfig(**OPTIONS))

    def forbidden(*args, **kwargs):
        raise AssertionError("initial_clustering must not run when labels are given")

    monkeypatch.setattr(driver, "initial_clustering", forbidden)
    given = train_surrogate(X, f_fn=f, initial_labels=labels, **OPTIONS)
    assert given.N_src == computed.N_src
    np.testing.assert_array_equal(host(given.X_src), host(computed.X_src))
    assert given.profiling_stats["initial_labels_given"] is True
    assert computed.profiling_stats["initial_labels_given"] is False


@pytest.mark.parametrize(
    "labels", [np.zeros(len(X) - 1, dtype=int), np.zeros(len(X)), np.full(len(X), -1), np.zeros((len(X), 1), dtype=int)]
)
def test_invalid_labels_are_rejected_before_evaluating(labels):
    evaluations = []

    def counted(x):
        evaluations.append(len(x))
        return f(x)

    with pytest.raises(ValueError, match="initial_labels must be"):
        train_surrogate(X, f_fn=counted, initial_labels=labels, **OPTIONS)
    assert not evaluations
