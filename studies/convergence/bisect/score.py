"""
Score saved runs (`capture_launcher.py` or `era_old.py` outputs) with one metric definition.

    python score.py run1.npz run2.npz ... [--ref-test fixed_test.npz]

Per run: `N_c`, `N`, `p_o` (% of test points the run flagged outside every trust region), `e_in`
and `e` (relative L2 over the inside points and over all points, the manuscript's definitions),
the launcher's own numbers when saved, and, when the leaf centres and trust matrices were saved,
`p_o*`: the share of test points outside the union of all ellipsoids, by an exact search over
every leaf (what any candidate rule can reach at best with these trust regions). `p_o - p_o*` is
the part of `p_o` caused by the candidate search (the `kappa` nearest centres in L2).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def outside_union(Xt, C, Q, chunk=2048):
    """True where no leaf has (x - c)^T Q (x - c) <= 1; exact over all leaves."""
    res = np.empty(len(Xt), bool)
    for s in range(0, len(Xt), chunk):
        dx = Xt[s : s + chunk, None, :] - C[None]
        res[s : s + chunk] = np.einsum("nki,kij,nkj->nk", dx, Q, dx, optimize=True).min(1) > 1.0
    return res


def outside_kappa(Xt, C, Q, kappa, chunk=2048):
    """The inference rule of every era: the kappa nearest centres in L2 (exact), then the smallest
    (x - c)^T Q (x - c) among them; True where that is > 1."""
    kappa = min(kappa, len(C))
    res = np.empty(len(Xt), bool)
    for s in range(0, len(Xt), chunk):
        q = Xt[s : s + chunk]
        d2 = (q * q).sum(1)[:, None] - 2 * q @ C.T + (C * C).sum(1)[None]
        idx = np.argpartition(d2, kappa - 1, axis=1)[:, :kappa]
        dx = q[:, None, :] - C[idx]
        res[s : s + chunk] = np.einsum("nki,nkij,nkj->nk", dx, Q[idx], dx, optimize=True).min(1) > 1.0
    return res


def score(path, ref_test=None, kappas=()):
    z = np.load(path, allow_pickle=False)
    out, pred, gt = z["out"].astype(bool), z["pred"], z["gt"]
    rel = lambda m: float(np.linalg.norm(pred[m] - gt[m]) / np.linalg.norm(gt[m]))  # noqa: E731
    row = {"run": Path(path).stem, "N_c": int(z["N_c"]), "N": int(z["N"]), "p_o": 100 * out.mean(),
           "e_in": rel(~out), "e": rel(np.ones_like(out)), "n_test": len(out)}  # fmt: skip
    if "metrics" in z.files:
        m = json.loads(str(z["metrics"]))
        row["launcher_p_o"] = m.get("perct_out")
        row["launcher_e"] = m.get("all", {}).get("l2_rel_error")
    if "predict_args" in z.files:
        row["coarse_factor"] = json.loads(str(z["predict_args"])).get("coarse_factor")
    if "Xt" in z.files and ref_test is not None:
        row["same_test"] = bool(np.array_equal(z["Xt"], ref_test[: len(z["Xt"])]))
    if "X_src" in z.files and "Qs" in z.files and "Xt" in z.files:
        row["p_o_union"] = 100 * outside_union(z["Xt"], z["X_src"], z["Qs"]).mean()
        for k in kappas:
            row[f"p_o_k{k}"] = 100 * outside_kappa(z["Xt"], z["X_src"], z["Qs"], k).mean()
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--ref-test", default=None)
    p.add_argument("--kappas", default="", help="comma-separated candidate counts to re-score p_o with")
    a = p.parse_args()
    ref = np.load(a.ref_test)["Xt"] if a.ref_test else None
    for r in a.runs:
        row = score(r, ref, [int(k) for k in a.kappas.split(",") if k])
        print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
