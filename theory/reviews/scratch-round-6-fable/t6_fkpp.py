"""
Test 6 (Fisher-KPP ladder): Demonstration 8.4's claim that the chart dimension for the lift error does not grow with the grid, on the
finer grids the benchmark already has trajectories for (d = 40, 80): theory/demos/reduced_dims.py's lift error for k = 5..12.
"""
import json
import sys

import numpy as np
from common import HERE

from src.benchmark.step_maps import STEP_MAPS, data, visited

out = {}
for name in sys.argv[1:]:
    p = STEP_MAPS[name]
    train, test = data(name)
    X = visited(train)
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    mu = X.mean(axis=0)
    Vt = np.linalg.svd(X - mu, full_matrices=False)[2]
    idx = np.random.default_rng(0).choice(len(Xe), min(400, len(Xe)), replace=False)
    m = p.scored(Ye[idx])
    rows = {}
    for k in range(5, 13):
        proj = mu + (Xe[idx] - mu) @ Vt[:k].T @ Vt[:k]
        rows[k] = float(np.max(np.where(m, np.abs(p.step(proj) - Ye[idx]), 0.0)))
        print(name, k, f"{rows[k]:.2e}", flush=True)
    out[name] = rows
    (HERE / f"t6_fkpp_{name}.json").write_text(json.dumps(rows, indent=1))
