"""
Post hoc, after p10_tube.py (no predictions): the tube arms' one-step defects on mixed inputs *without the gate*. Where an arm's gate
refuses every input (its calibrated maximum exceeds the budget), p10's accepted-defect scores are undefined, and the question the tube
experiment asks (does an arm own the first-order off-bundle term?) needs the defect itself. For each finished record of p10: the arm's
surrogate rebuilt from its record (generic4.rebuild, the same variant and extra pairs), the test trajectories' states mixed once (w =
0.01, 0.1; seed 9, as p10), and on plain test inputs: the median, 99% quantile and maximum of the worst scored defect per state, and
the sign coherence. Writes data/p10b_tube_ungated.json.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from generic2 import mixed_once
from generic4 import rebuild
from p10_tube import ARMS, OUT
from r5_data_scaling import extra_pairs

from src.benchmark.step_maps import STEP_MAPS, data


def stats(P, sur, X, Y) -> dict:
    with np.errstate(all="ignore"):
        e = np.where(P.scored(Y), sur.evaluate(X, cache=False)[0] - Y, 0.0)
    m = np.max(np.abs(e), axis=1)
    m = np.where(np.isfinite(m), m, np.inf)
    return {"median": float(np.median(m)), "q99": float(np.quantile(m, 0.99)), "max": float(np.max(m)),
            "coherence": float(np.nanmax(np.abs(e.mean(0)) / np.maximum(np.abs(e).mean(0), 1e-300)))}  # fmt: skip


def main() -> None:
    out = {}
    names = sys.argv[1:]
    for f in sorted(glob.glob(str(OUT / "records" / "p10" / "*.json"))):
        rec = json.loads(Path(f).read_text())
        name, arm = rec["key"]["problem"], rec["key"]["arm"]
        if rec.get("status") != "done" or (names and name not in names):
            continue
        P = STEP_MAPS[name]
        train, test = data(name)
        extra = extra_pairs(name, P, 1600) if arm == "T0+" else None
        sur = rebuild(P, train, rec, extra, ARMS[arm])
        res = {"plain": stats(P, sur, test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2]))}
        rng = np.random.default_rng(9)
        for w in (0.01, 0.1):
            X = mixed_once(P, test, w, rng)
            res[str(w)] = stats(P, sur, X, P.step(X))
        out[f"{name}/{arm}"] = res
        print(f"{name} {arm}: plain median {res['plain']['median']:.1e} max {res['plain']['max']:.1e}; mixed 0.01 median {res['0.01']['median']:.1e} "
              f"q99 {res['0.01']['q99']:.1e} max {res['0.01']['max']:.1e}; 0.1 median {res['0.1']['median']:.1e} max {res['0.1']['max']:.1e}")  # fmt: skip
    (OUT / "p10b_tube_ungated.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
