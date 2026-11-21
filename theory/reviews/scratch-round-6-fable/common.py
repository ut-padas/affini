"""Shared setup for the round-6 tests: the chart, decoder and reduced surrogate of theory/demos/curved_lift.py, rebuilt identically."""
import itertools
import pickle
from pathlib import Path

import numpy as np

from src.benchmark.step_maps import STEP_MAPS, Projected, data, invariants_of, visited

HERE = Path(__file__).resolve().parent


def poly(U, q):
    cols = [np.ones(len(U))]
    for deg in range(1, q + 1):
        for idx in itertools.combinations_with_replacement(range(U.shape[1]), deg):
            cols.append(np.prod(U[:, list(idx)], axis=1))
    return np.stack(cols, axis=1)


class Reduced:
    def __init__(self, model, mu, V, lo, hi):
        self.model, self.mu, self.V, self.lo, self.hi = model, mu, V, lo, hi

    def refuses(self, Z):
        U = (Z - self.mu) @ self.V
        return np.any((self.lo > U) | (self.hi < U), axis=1)

    def predict(self, Z):
        U = np.clip((Z - self.mu) @ self.V, self.lo, self.hi)
        return np.asarray(self.model.predict(U)) - Z


class FnModel:
    """Any next-state function of the chart value as an increment model."""

    def __init__(self, fn, mu, V, lo, hi):
        self.fn, self.mu, self.V, self.lo, self.hi = fn, mu, V, lo, hi

    def refuses(self, Z):
        U = (Z - self.mu) @ self.V
        return np.any((self.lo > U) | (self.hi < U), axis=1)

    def predict(self, Z):
        return self.fn(np.clip((Z - self.mu) @ self.V, self.lo, self.hi)) - Z


def setup(name="hires", k=3, q=4):
    p = STEP_MAPS[name]
    train, test = data(name)
    X = visited(train)
    Y = train[:, 1:].transpose(1, 0, 2).reshape(-1, train.shape[2])
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:k].T
    coef = np.linalg.lstsq(poly((X - mu) @ V, q), X, rcond=None)[0]
    U = (X - mu) @ V
    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    return dict(p=p, train=train, test=test, X=X, Y=Y, mu=mu, V=V, coef=coef, q=q, lo=lo - pad, hi=hi + pad, U=U, W=invariants_of(p, train))


def load_reduced(s):
    model = pickle.loads((HERE / "hires_k3_model.pkl").read_bytes())
    red = Reduced(model, s["mu"], s["V"], s["lo"], s["hi"])
    return Projected(red, s["p"], s["W"]) if len(s["W"]) else red
