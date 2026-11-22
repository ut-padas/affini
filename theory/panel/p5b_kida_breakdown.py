"""
Post hoc, after p5_kida_screen.py (no predictions): where the KIDA arms' one-step errors sit. For each fitted arm of p5 (rebuilt as p5 builds
it, on the same 160 fit tracers), on the 40 test tracers' plain inputs: per key species, the median and worst error and the share of states
above the tolerance (where the truth exceeds 1e-12); per step of the tracers, the median and worst key-species error; and the oracle
coverage, the largest share of states any gate could accept (the share whose key-species error is within the tolerance and within half
of it). No solves. Writes data/p5b_kida_breakdown.json.
"""

import json

import numpy as np
from p5_kida_screen import OUT, QOI, TOL, Data, LinearArm, LocalArm, MLPArm, Persistence


def main() -> None:
    D = Data()
    Xf, Yf = D.pairs(D.fit)
    X, Y = D.pairs(D.test)
    steps = D.test.shape[1] - 1
    sp = list(D.P.system.names) if hasattr(D.P.system, "names") else None
    idx = np.flatnonzero(D.key)
    names = [sp[i] for i in idx] if sp else QOI
    out = {}
    for arm, build in (("persistence", Persistence), ("linear", lambda: LinearArm(D, "linear", Xf, Yf)),
                       ("chart2", lambda: LinearArm(D, "chart2", Xf, Yf)), ("local", lambda: LocalArm(D, Xf, Yf)),
                       ("mlp", lambda: MLPArm(D, Xf, Yf))):  # fmt: skip
        model = build()
        P = model.predict(X)
        above = Y > -12
        e = np.abs(P - Y)
        species = {}
        for n, i in zip(names, idx, strict=True):
            m = above[:, i]
            ei = e[m, i]
            species[n] = {"median": float(np.median(ei)) if len(ei) else None, "worst": float(ei.max()) if len(ei) else None,
                          "above_tol_percent": float(100 * np.mean(ei > TOL)) if len(ei) else None}  # fmt: skip
        key = np.max(np.where(above & D.key, e, 0.0), axis=1)
        by_step = key.reshape(len(D.test), steps)
        out[arm] = {"species": species, "step_median": np.median(by_step, axis=0).tolist(), "step_worst": by_step.max(axis=0).tolist(),
                    "oracle_coverage_percent": float(100 * np.mean(key <= TOL)), "oracle_coverage_half_percent": float(100 * np.mean(key <= TOL / 2))}  # fmt: skip
        top = sorted(species.items(), key=lambda kv: -(kv[1]["above_tol_percent"] or 0))[:4]
        print(f"{arm}: oracle coverage {out[arm]['oracle_coverage_percent']:.1f}% (half tol {out[arm]['oracle_coverage_half_percent']:.1f}%); "
              f"step medians {by_step[:, 0].mean():.2e} first, {np.median(by_step[:, -1]):.2e} last; worst species "
              + ", ".join(f"{n} {v['above_tol_percent']:.0f}% (median {v['median']:.1e})" for n, v in top))  # fmt: skip
    (OUT / "p5b_kida_breakdown.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
