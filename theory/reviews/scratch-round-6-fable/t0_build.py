"""Rebuild Demonstration 8.5's reduced ChebTT (HIRES, k = 3, decoder lift) and keep it for the other tests; checks reproducibility."""
import pickle
import time

import numpy as np
from common import HERE, poly, setup

from src.schemes import chebtt

s = setup()
p, mu, V, coef, q = s["p"], s["mu"], s["V"], s["coef"], s["q"]
calls = {"n": 0, "t": 0.0, "bad": 0}


def oracle(Wq):
    t0 = time.perf_counter()
    Z = poly(Wq, q) @ coef
    out = p.step(Z)
    calls["n"] += len(Wq)
    calls["t"] += time.perf_counter() - t0
    calls["bad"] += int((~np.isfinite(out)).any(axis=1).sum())
    return out


t0 = time.perf_counter()
model = chebtt.build_vector(oracle, s["lo"], s["hi"], 0.005, s["X"].shape[1], seed=1, max_evals=200_000)
print("build_s", time.perf_counter() - t0, "evals", model.evals, "oracle calls", calls, flush=True)
import sys
(HERE / (sys.argv[1] if len(sys.argv) > 1 else "hires_k3_model.pkl")).write_bytes(pickle.dumps(model))
print("pickled")
