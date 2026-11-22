"""
Status: not run. Smoke tests on hires (theory/SYNTHESIS.md, section 3.15) showed that no gate tried (a neighbour distance; the local
Proposition 5.1 term below alone; the term below as written) transfers from calibration states to chain states: each accepted chain states
with one-step defects of 0.15-1.0. The predictions O1-O5 were written for a run that was stopped before it started.

Online local tables (council: Sol's B arm, Astra's frozen-against-growing comparison; ISAT's design, TH-62): does reusing the pairs a run
has already solved make a local surrogate serve where the global fits refuse? One local model in three arms, on the same test streams:
  frozen  the table is the training pairs (the fit tracers), never grown;
  online  the same table, and every refused query's solver step is inserted once its step of the stream ends;
  cold    an empty table grown only by insertions (the cost of a cold start).
The model: coordinates are the free components' principal coordinates (at most 20) and the held inputs, scaled by their spread on the fit
states; a query's increment is the intercept of a ridge affine fit on its K = min(64, max(16, 2(m+1))) nearest stored pairs (m coordinates);
its risk is the larger of Proposition 5.1 applied to that local fit, (1 + Lambda) times the largest scored residual of the fit on its
neighbours (Lambda the 1-norm of the intercept's weights on the neighbours' increments), and the largest scored disagreement between the
increment fit and the same fit to the successors themselves (they agree inside the neighbours' span and differ by an off-span offset of
the query, which the residual cannot see), refused above a threshold r*. r* is calibrated once on the frozen
table: the largest risk whose accepted calibration states (plain) all err within half the tolerance (none: r* = 0, every query refused).
Neither a distance to the neighbours nor the residual term alone is enough: in smoke tests on hires each, calibrated this way, accepted
chain states with one-step defects of 0.15-0.2 (states drifted off the neighbours' span in a direction the step contracts). The stream: all test trajectories advance together one step at a time
(a simulation's cells), each query scored before any insertion from its own step; held inputs follow the true path.
Scores per arm: the worst scored error of the chains with fallback (KIDA: the 13 key species, theory/KIDA_QOI.md), the solver share by step,
the worst one-step defect on the arm's own accepted inputs (solved for the audit only, never inserted), the final table size and its memory,
prediction and gate time per query (one local fit gives both). Data: the step maps' benchmark data (train split by trajectory into fit 80% and calibration 20%), KIDA's
dark range as p5 (fit 160, calibration 20 tracers). One record per (problem, arm) under data/records/p7/; a case is limited to 1,800 s.
Usage: p7_online.py [problem ...].

Predictions, committed before the run:
  O1  every arm's accepted chains stay within the tolerance and its own-input defect within it (the calibrated distance gate holds);
  O2  where the frozen table refuses at least 20% of the stream, the online arm's solver share over the last five steps is at least 10
      points below its share over the first five on at least half of those problems (later states recur near earlier solved ones);
  O3  on KIDA the online arm still refuses more than 80% of the stream (40 tracers in different environments do not recur at 0.01 dex);
  O4  cold refuses more than online on every problem over the whole stream, and its final table is smaller than the stream's solves
      under the frozen table plus the training pairs;
  O5  on the lattice controls (allen_cahn_d10, fisher_kpp_d10) and on hires the frozen table refuses more than the chart recipe did in
      R4c or r6 (a global fit serves more where it holds).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.benchmark.step_maps import STEP_MAPS, data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
import r3_kida  # noqa: E402,F401 -- registers the KIDA system before any solver pool is forked
from common import log, recorded  # noqa: E402

from src.infra.records import RecordStore  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
PROBLEMS = ("hires", "chemakzo", "oregonator", "van_der_pol", "primordial", "nelson", "pollu", "allen_cahn_d10", "fisher_kpp_d10", "kida_dark")
ARMS = ("frozen", "online", "cold")


def setup(name: str):
    """(problem with its scoring, fit trajectories, calibration trajectories, test trajectories)."""
    if name == "kida_dark":
        from p5_kida_screen import Data, KeyScored

        D = Data()
        return KeyScored(D.P, D.key), D.fit, D.cal, D.test
    P = STEP_MAPS[name]
    train, test = data(name)
    order = np.random.default_rng(0).permutation(len(train))
    n_fit = len(train) * 4 // 5
    return P, train[order[:n_fit]], train[order[n_fit:]], test


class Table:
    """Stored one-step pairs and the local affine model over them (module docstring)."""

    def __init__(self, P, fit, X=None, Y=None):
        self.P, self.h = P, P.system.held
        S = fit[:, :-1].reshape(-1, fit.shape[2])
        self.free = fit.shape[2] - self.h
        self.cut = -14.0 if P.system.log else -np.inf
        F = self._clip(S)[:, : self.free]
        self.mu = F.mean(0)
        k = min(self.free, 20)
        self.V = np.linalg.svd(F - self.mu, full_matrices=False)[2][:k].T
        U = self._coords(S, scale=False)
        self.sc = np.maximum(U.std(0), 1e-9)
        self.m = U.shape[1]
        self.K = min(64, max(16, 2 * (self.m + 1)))
        self.X = np.empty((0, fit.shape[2])) if X is None else X
        self.Y = np.empty((0, fit.shape[2])) if Y is None else Y
        self._index()

    def _clip(self, Z):
        out = np.array(Z, dtype=float)
        out[:, : self.free] = np.maximum(out[:, : self.free], self.cut)
        return out

    def _coords(self, Z, scale=True):
        Zc = self._clip(Z)
        U = np.concatenate([(Zc[:, : self.free] - self.mu) @ self.V, Zc[:, self.free :]], axis=1)
        return U / self.sc if scale else U

    def _index(self):
        self.U = self._coords(self.X) if len(self.X) else np.empty((0, self.m))
        self.dZ = self._clip(self.Y)[:, : self.free] - self._clip(self.X)[:, : self.free] if len(self.X) else None
        self.mask = self.P.scored(self.Y)[:, : self.free] if len(self.X) else None  # residuals count where the outputs are scored
        self.Yc = self._clip(self.Y)[:, : self.free] if len(self.X) else None
        self.nn = NearestNeighbors(n_neighbors=min(self.K, len(self.X))).fit(self.U) if len(self.X) >= self.m + 1 else None

    def insert(self, X, Y):
        if len(X):
            self.X, self.Y = np.concatenate([self.X, X]), np.concatenate([self.Y, Y])
            self._index()

    def local(self, Z):
        """
        (increment, risk) per row of Z. The increment is the intercept of the ridge affine fit on the query's nearest stored pairs, in
        coordinates centred at the query, so it is a linear combination sum_j l_j dZ_j of the neighbours' increments; the risk is
        the larger of Proposition 5.1 applied to that local fit, (1 + sum_j |l_j|) times the largest scored residual of the fit on its
        neighbours, and the disagreement between the increment fit and the same fit to the successors (an off-span offset; the comment
        at the line). Held components are unchanged.
        """
        inc = np.zeros_like(Z, dtype=float)
        risk = np.full(len(Z), np.inf)
        if self.nn is None:
            return inc, risk
        Uq = self._coords(Z)
        _, idx = self.nn.kneighbors(Uq)
        base = self._clip(Z)[:, : self.free]
        for i in range(len(Z)):
            A = np.concatenate([self.U[idx[i]] - Uq[i], np.ones((idx.shape[1], 1))], axis=1)
            H = np.linalg.solve(A.T @ A + 1e-8 * np.eye(A.shape[1]), A.T)  # coef = H dZ; the intercept's weights are H's last row
            dZ = self.dZ[idx[i]]
            coef = H @ dZ
            inc[i, : self.free] = base[i] + coef[-1] - Z[i, : self.free]
            res = np.where(self.mask[idx[i]], np.abs(A @ coef - dZ), 0.0)
            # the same local fit to the successors themselves: inside the neighbours' span it is the increment fit plus the (affine)
            # input, so the two agree; off the span they differ by the query's offset, which the increment fit carries forward and a
            # contracting step removes, so their disagreement measures how far off-span the query is in the output's own units
            nxt_b = H[-1] @ self.Yc[idx[i]]
            off = np.where(self.mask[idx[i]].any(axis=0), np.abs(base[i] + coef[-1] - nxt_b), 0.0)
            risk[i] = max((1.0 + np.abs(H[-1]).sum()) * res.max(), off.max())
        return inc, risk

    def risk(self, Z):
        return self.local(Z)[1]

    def predict(self, Z):
        return self.local(Z)[0]


def worst(P, Zp, Zt):
    e = np.where(P.scored(Zt), np.abs(Zp - Zt), 0.0)
    return np.where(np.all(np.isfinite(e), axis=1), np.max(e, axis=1), np.inf)


def calibrate(P, table, cal) -> float:
    X, Y = cal[:, :-1].reshape(-1, cal.shape[2]), cal[:, 1:].reshape(-1, cal.shape[2])
    inc, r = table.local(X)
    e = worst(P, X + inc, Y)
    order = np.argsort(r)
    bad = np.flatnonzero(e[order] > TOL / 2)
    if not len(bad):
        return float(r.max())
    return float(r[order][bad[0] - 1]) if bad[0] > 0 else 0.0


def stream(P, table, rstar, test, grow) -> dict:
    """All test chains advanced together; refused states solved (and inserted when ``grow``); accepted ones audited against the solver."""
    h = P.system.held
    Z = test[:, 0].copy()
    by_step, solved, own, pred_s, n_q = [], [], 0.0, 0.0, 0
    for k in range(test.shape[1] - 1):
        if h:
            Z[:, -h:] = test[:, k, -h:]
        t = time.perf_counter()
        inc, risk = table.local(Z)  # one fit per query gives the prediction and its risk
        pred_s += time.perf_counter() - t
        r = ~(risk <= rstar)
        nxt = Z + inc
        n_q += len(Z)
        exact = P.step(Z)  # the fallback for refused states and the audit of accepted ones (never inserted)
        if (~r).any():
            own = max(own, float(np.max(worst(P, nxt[~r], exact[~r]))))
        nxt[r] = exact[r]
        if grow:
            table.insert(Z[r], exact[r])
        Z = nxt
        by_step.append(float(np.max(worst(P, Z, test[:, k + 1]))))
        solved.append(float(r.mean()))
    return {"rollout_worst": max(by_step), "rollout_error_by_step": by_step, "solver_share": float(np.mean(solved)),
            "solver_share_by_step": solved, "own_input_defect_worst": own, "table_size": int(len(table.X)),
            "table_bytes": int(table.X.nbytes + table.Y.nbytes + table.U.nbytes), "predict_and_gate_us": 1e6 * pred_s / n_q}  # fmt: skip


def case(name: str, arm: str) -> dict:
    P, fit, cal, test = setup(name)
    t0 = time.perf_counter()
    X, Y = fit[:, :-1].reshape(-1, fit.shape[2]), fit[:, 1:].reshape(-1, fit.shape[2])
    frozen = Table(P, fit, X, Y)
    rstar = calibrate(P, frozen, cal)
    table = frozen if arm != "cold" else Table(P, fit)
    out = {"rstar": rstar, "coords": table.m, "neighbours": table.K, "train_pairs": int(len(X)), "build_s": time.perf_counter() - t0}
    out.update(stream(P, table, rstar, test, grow=arm != "frozen"))
    s = np.array(out["solver_share_by_step"])
    log(f"  {name} {arm}: rollout {out['rollout_worst']:.2e}, solver {100 * out['solver_share']:.1f}% (first 5 {100 * s[:5].mean():.0f}%, last 5 "
        f"{100 * s[-5:].mean():.0f}%), own-input defect {out['own_input_defect_worst']:.2e}, table {out['table_size']}, r* {rstar:.3g}")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "p7", path_of=lambda key: OUT / "records" / "p7" / f"{key['problem']}_{key['arm']}.json")
    for name in names:
        for arm in ARMS:
            recorded(store, {"problem": name, "arm": arm, "tol": TOL}, case, name, arm, limit=1800)
    fields = ("status", "rstar", "coords", "rollout_worst", "solver_share", "own_input_defect_worst", "table_size", "table_bytes", "predict_and_gate_us")
    summary = {f"{r['key']['problem']}/{r['key']['arm']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "p7_online.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
