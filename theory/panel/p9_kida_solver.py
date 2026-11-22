"""
KIDA's exact baseline (council round 2: Sol's first action, Fable's step 6c): before any speed-up against KIDA's solver is quoted, how far
the stored labels are from a tighter reference, and what a sparse Jacobian saves. On 120 dark-range states (the 40 test tracers' states
at steps 0, 8 and 16; serial, one core), each stepped by:
  dense      the deployed integrator (BDF, rtol 1e-6, atol 1e-24, the analytic Jacobian densified to 582 x 582);
  sparse     the same tolerances with the Jacobian kept sparse (SciPy's BDF then factorises with a sparse LU);
  reference  BDF at rtol 1e-9, atol 1e-28, sparse Jacobian.
Scores: seconds per step (median, 90th percentile, maximum), the solver's function, Jacobian and LU counts; against the reference, the
worst key-species and strict errors (dex, where the reference exceeds 1e-12) of the dense labels and of the sparse ones. Writes
data/p9_kida_solver.json.

Prediction, committed before the run: the sparse Jacobian makes a step at least 2x faster at the median (a 582 x 582 dense LU against a
sparse one with about 10 nonzeros per row), the stored labels are within 1e-3 dex of the reference on the key species, and the sparse
and dense labels agree within 1e-4 dex.
"""

import json
import time

import numpy as np
import r3_kida
import scipy.sparse as sp
from p5_kida_screen import OUT, Data
from scipy.integrate import solve_ivp


def main() -> None:
    D = Data()
    system = D.P.system
    N = len(r3_kida._net.species)
    Z = np.concatenate([D.test[:, k] for k in (0, 8, 16)])
    Y = 10.0**Z
    key = D.key

    pattern = sp.csc_matrix(system.jac(Y[0]))
    out = {"states": int(len(Y)), "nonzeros_per_row": float(pattern.nnz / pattern.shape[0])}
    # the sparse arm must not pay the densification it removes: the network's sparse Jacobian, built directly
    from solver import _make_jac  # noqa: PLC0415

    ops = {}

    def jac_sparse(v):
        env = tuple(np.round(v[N:], 12))
        if env not in ops:
            A, B = r3_kida._net.get_operators_vectorized({"nH": v[N], "T": v[N + 1], "Av": v[N + 2], "uv_flux": v[N + 3], "Tcap_2body": True})
            ops[env] = _make_jac(A.tocsr(), B, N)
        Jn = ops[env](0.0, v[:N]).tocsc()
        return sp.bmat([[Jn, None], [None, sp.csc_matrix((4, 4))]], format="csc")

    arms = {"dense": {"jac": lambda t, v: system.jac(v), "rtol": system.rtol, "atol": system.atol},
            "sparse": {"jac": lambda t, v: jac_sparse(v), "rtol": system.rtol, "atol": system.atol},
            "reference": {"jac": lambda t, v: jac_sparse(v), "rtol": 1e-9, "atol": 1e-28}}  # fmt: skip
    res = {}
    for name, kw in arms.items():
        nxt, secs, nfev, njev, nlu = np.empty_like(Y), [], [], [], []
        for i, y in enumerate(Y):
            t0 = time.perf_counter()
            s = solve_ivp(lambda t, v: system.rhs(v), (0.0, D.P.dt), y, method="BDF", **kw)
            secs.append(time.perf_counter() - t0)
            nxt[i] = s.y[:, -1] if s.success else np.nan
            nfev.append(s.nfev), njev.append(s.njev), nlu.append(s.nlu)
        res[name] = system.coords(nxt)
        out[name] = {"median_s": float(np.median(secs)), "p90_s": float(np.quantile(secs, 0.9)), "max_s": float(np.max(secs)),
                     "nfev_median": float(np.median(nfev)), "njev_median": float(np.median(njev)), "nlu_median": float(np.median(nlu))}  # fmt: skip
        print(name, out[name])
    ref = res["reference"]
    above = ref > -12
    for name in ("dense", "sparse"):
        e = np.abs(res[name] - ref)
        out[name]["key_worst_vs_reference"] = float(np.max(np.where(above & key, e, 0.0)))
        out[name]["strict_worst_vs_reference"] = float(np.max(np.where(above & ~np.isin(np.arange(len(key)), np.arange(N, len(key))), e, 0.0)))
        print(name, "vs reference: key", out[name]["key_worst_vs_reference"], "strict", out[name]["strict_worst_vs_reference"])
    e = np.abs(res["dense"] - res["sparse"])
    out["dense_vs_sparse_key_worst"] = float(np.max(np.where(above & key, e, 0.0)))
    print("dense vs sparse, key:", out["dense_vs_sparse_key_worst"])
    stored = np.concatenate([D.test[:, k + 1] for k in (0, 8, 16)])
    out["stored_vs_dense_key_worst"] = float(np.max(np.where(above & key, np.abs(stored - res["dense"]), 0.0)))
    print("stored labels vs this run's dense step, key:", out["stored_vs_dense_key_worst"])
    (OUT / "p9_kida_solver.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
