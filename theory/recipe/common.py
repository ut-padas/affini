"""
Shared pieces of the recipe runs (memo section R): the problems, the chart and decoder, the chart fit, the deployed model and its scores.

The recipe, for a step-map problem P with training trajectories (n, steps + 1, d) in the surrogates' coordinates:
  floor    in a log system with an abundance floor every free component is read as max(z, log10(floor) - 2), so species far below the
           scoring floor do not carry their log10 noise into the chart (the surrogate predicts them at that level);
  chart    u = (V_k^T (c(z_free) - mu), z_held): the top-k principal directions of the visited free states, and the held inputs (an
           environment) as further chart coordinates;
  decoder  D(u): a least-squares polynomial of total degree 4 (lower when the pairs are short) from u to c(z_free);
  k        the smallest k whose lift error max |Phi(D(u), held) - Phi(z)| on validation states (visited states of a fifth of the training
           trajectories, held out of the chart and decoder) is within half the tolerance (the lift error is a floor under the one-step
           error, so it must leave room), else the smallest k within twice the smallest lift error seen: a larger k than the data's
           dimension leaves the fit unconstrained across the extra coordinates, so the smallest adequate k is preferred; solves counted;
  fit      G(u), total degree q, from u to the next free state, values only (no solve), for every q with at least 1.5 pairs per term;
  rule     the q with the smallest worst chained error over all training trajectories, of the deployed model (fit, output clip, projection
           onto the input's linear invariants); it passes when that error is within the tolerance;
  scores   one-step worst on the test states; worst rollout over the test trajectories (the benchmark's `rollout`); worst rollout with
           mixing before every step (each free state replaced by (1 - w) y + w y_partner, in the system's own variables, the partner a
           random other test trajectory's true state at that step; w = 0.01 and 0.1), against the solver chained with the same mixing: a
           proxy for transport between cells in a split simulation.
"""

import glob
import json
import signal
import time
from pathlib import Path

import numpy as np

from src.benchmark.step_maps import Projected, invariants_of, rollout, visited
from src.infra.records import RecordStore, failure

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def multi_indices(m: int, q: int) -> list[tuple]:
    if m == 0:
        return [()]
    return [(a, *rest) for a in range(q + 1) for rest in multi_indices(m - 1, q - a)]


def n_terms(m: int, q: int) -> int:
    from math import comb

    return comb(m + q, q)


class ChartPoly:
    """Total-degree-q Chebyshev polynomials of the chart coordinates, scaled to [-1, 1] on the chart box."""

    def __init__(self, q, lo, hi):
        self.q, self.lo, self.hi = q, lo, hi
        self.idx = multi_indices(len(lo), q)

    def design(self, U):
        s = np.clip((2 * U - self.lo - self.hi) / (self.hi - self.lo), -1.0, 1.0)
        T = [np.polynomial.chebyshev.chebvander(s[:, j], self.q) for j in range(s.shape[1])]
        out = np.empty((len(U), len(self.idx)))
        for t, m in enumerate(self.idx):
            col = T[0][:, m[0]].copy()
            for j in range(1, len(m)):
                if m[j]:
                    col *= T[j][:, m[j]]
            out[:, t] = col
        return out

    def fit(self, U, Y):
        self.coef = np.linalg.lstsq(self.design(U), Y, rcond=None)[0]
        return self

    def __call__(self, U):
        return self.design(U) @ self.coef


class Setup:
    """A problem's chart pieces from its training trajectories: floor clip, held split, principal directions, chart box."""

    def __init__(self, P, train):
        self.P = P
        self.h = P.system.held
        d = train.shape[2]
        self.free = d - self.h
        self.cut = np.log10(P.floor) - 2.0 if (P.system.log and P.floor > 0) else -np.inf
        X = visited(train)
        self.Xc = self.clip(X)
        self.Y = self.clip(train[:, 1:].transpose(1, 0, 2).reshape(-1, d))[:, : self.free]
        self.mu = self.Xc[:, : self.free].mean(axis=0)
        self.Vall = np.linalg.svd(self.Xc[:, : self.free] - self.mu, full_matrices=False)[2].T
        self.sing = np.linalg.svd(self.Xc[:, : self.free] - self.mu, compute_uv=False)
        self.zlo, self.zhi = self.Xc.min(0) - 2.0, self.Xc.max(0) + 0.5
        self.W = invariants_of(P, train)

    def clip(self, Z):
        return np.maximum(Z, self.cut) if np.isfinite(self.cut) else Z

    def chart(self, Z, k):
        Zc = self.clip(Z)
        return np.concatenate([(Zc[:, : self.free] - self.mu) @ self.Vall[:, :k], Zc[:, self.free :]], axis=1)


class Chart:
    """The chart at k coordinates (plus the held inputs) and its decoder, fitted on the given visited states."""

    def __init__(self, S: Setup, k: int, Xfit: np.ndarray, dec_degree: int = 4):
        self.S, self.k = S, k
        U = S.chart(Xfit, k)
        self.lo, self.hi = U.min(0), U.max(0)
        pad = 0.02 * np.maximum(self.hi - self.lo, 1e-6)
        self.lo, self.hi = self.lo - pad, self.hi + pad
        q = dec_degree
        while q > 1 and n_terms(U.shape[1], q) * 1.5 > len(U):
            q -= 1
        self.dec_degree = q
        self.dec = ChartPoly(q, self.lo, self.hi).fit(U, S.clip(Xfit)[:, : S.free])

    def u(self, Z):
        return np.clip(self.S.chart(Z, self.k), self.lo, self.hi)

    def decode(self, Z):
        """The decoded state of each row of Z: D(pi z) for the free components, the held inputs as they are."""
        return np.concatenate([self.dec(self.u(Z)), Z[:, self.S.free :]], axis=1)


class Model:
    """The deployed increment model: G(pi z), clipped to a margin around the training states (held inputs unchanged)."""

    def __init__(self, chart: Chart, G: ChartPoly):
        self.chart, self.G, self.evals = chart, G, 0

    def refuses(self, Z):
        return np.zeros(len(Z), dtype=bool)

    def predict(self, Z):
        S = self.chart.S
        out = np.array(Z, dtype=float)
        out[:, : S.free] = np.clip(self.G(self.chart.u(Z)), S.zlo[: S.free], S.zhi[: S.free])
        return out - Z


def deployed(S: Setup, chart: Chart, G: ChartPoly):
    return Projected(Model(chart, G), S.P, S.W) if len(S.W) else Model(chart, G)


def worst_scored(P, Zpred, Ztrue) -> float:
    return float(np.max(np.where(P.scored(Ztrue), np.abs(Zpred - Ztrue), 0.0))) if len(Ztrue) else 0.0


def choose_k(S: Setup, train: np.ndarray, tol: float, kmax: int, rng) -> dict:
    """k by the rule of the module docstring, from the lift-error curve on validation states of held-out training trajectories."""
    n = len(train)
    order = rng.permutation(n)
    val, fit = order[: max(1, n // 5)], order[max(1, n // 5) :]
    Xf = visited(train[fit])
    Xv = visited(train[val])
    Yv = train[val][:, 1:].transpose(1, 0, 2).reshape(-1, train.shape[2])
    pick = rng.choice(len(Xv), min(100, len(Xv)), replace=False)
    Xv, Yv = Xv[pick], Yv[pick]
    rows, solves = [], 0
    for k in range(1, min(kmax, S.free) + 1):
        ch = Chart(S, k, Xf)
        lift = S.P.step(ch.decode(Xv))
        solves += len(Xv)
        e = worst_scored(S.P, S.clip(lift), S.clip(Yv))
        rows.append({"k": k, "lift_error": e, "decoder_degree": ch.dec_degree})
        log(f"  k={k}: lift error {e:.3e} (decoder degree {ch.dec_degree})")
        if e <= tol / 2:
            break
    ok = [r for r in rows if r["lift_error"] <= tol / 2]
    floor = min(r["lift_error"] for r in rows)
    k = ok[0]["k"] if ok else next(r["k"] for r in rows if r["lift_error"] <= 2 * floor)
    return {"k": k, "k_passes": bool(ok), "curve": rows, "solves": solves}


def mixing_rollout(P, sur, true: np.ndarray, w: float, rng) -> float:
    """Worst scored error of the surrogate and the solver chained from the same states, both mixed before every step (module docstring)."""
    h = P.system.held
    free = true.shape[2] - h
    n = len(true)
    partners = np.array([[rng.choice([j for j in range(n) if j != i]) for i in range(n)] for _ in range(true.shape[1] - 1)])
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)

    def mix(Z, k):
        out = Z.copy()
        out[:, :free] = back((1 - w) * lin(Z[:, :free]) + w * lin(true[partners[k], k, :free]))
        if h:
            out[:, free:] = true[:, k, free:]
        return out

    Zs, Zt, worst = true[:, 0].copy(), true[:, 0].copy(), 0.0
    for k in range(true.shape[1] - 1):
        Zt = P.step(mix(Zt, k))
        Zin = mix(Zs, k)
        Zs = Zin + sur.predict(Zin)
        worst = max(worst, worst_scored(P, Zs, Zt))
        if not np.isfinite(worst) or worst > 10:
            return float(worst) if np.isfinite(worst) else float("inf")
    return worst


def run_recipe(P, train, test, tol=0.01, kmax=12, qmax=12, mix_w=(0.01, 0.1), seed=0) -> dict:
    """The recipe end to end (module docstring) on one problem; every number the memo quotes is in the returned dict."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    S = Setup(P, train)
    out = {"d": int(train.shape[2]), "held": int(S.h), "n_train": int(len(train)), "n_test": int(len(test)), "steps": int(train.shape[1] - 1),
           "invariants": int(len(S.W)), "singular_values": S.sing[:20].tolist()}  # fmt: skip
    kc = choose_k(S, train, tol, kmax, rng)
    out.update(k=kc["k"], k_passes=kc["k_passes"], k_curve=kc["curve"], solves=kc["solves"])
    chart = Chart(S, kc["k"], visited(train))
    U = chart.u(visited(train))
    m = U.shape[1]
    rows, fits = [], {}
    for q in range(1, qmax + 1):
        if n_terms(m, q) * 1.5 > len(U) or n_terms(m, q) > 3000:
            break
        G = ChartPoly(q, chart.lo, chart.hi).fit(U, S.Y)
        sur = deployed(S, chart, G)
        chain = float(rollout(P, sur, train)["rollout_worst"])
        rows.append({"degree": q, "terms": n_terms(m, q), "train_chains_worst": chain})
        fits[q] = G
        log(f"  q={q} ({n_terms(m, q)} terms): training chains {chain:.3e}")
    best = min(rows, key=lambda r: r["train_chains_worst"] if np.isfinite(r["train_chains_worst"]) else np.inf)
    q = best["degree"]
    sur = deployed(S, chart, fits[q])
    Xe = test[:, :-1].reshape(-1, test.shape[2])
    Ye = test[:, 1:].reshape(-1, test.shape[2])
    one = worst_scored(P, S.clip(Xe + sur.predict(Xe)), S.clip(Ye))
    roll = rollout(P, sur, test)
    out.update(degree_curve=rows, degree=q, passes=bool(best["train_chains_worst"] <= tol), decoder_degree=chart.dec_degree,
               one_step_worst=one, rollout_worst=float(roll["rollout_worst"]),
               rollout_worst_20=float(rollout(P, sur, test[:20])["rollout_worst"]),
               mixing_rollout_worst={str(w): mixing_rollout(P, sur, test, w, np.random.default_rng(seed + 1)) for w in mix_w},
               seconds=time.perf_counter() - t0)  # fmt: skip
    mixing = ", ".join(f"w={w}: {e:.3e}" for w, e in out["mixing_rollout_worst"].items())
    log(f"  selected k={out['k']} q={q}: one-step {one:.3e}, rollout {out['rollout_worst']:.3e}; mixing rollouts {mixing} ({out['seconds']:.0f} s)")
    return out


def _alarm(_signum, _frame):
    raise TimeoutError("the case exceeded its time limit")


def recorded(store: RecordStore, key: dict, compute, *args, limit: int = 120) -> dict:
    """
    One case through the record store: reused when a finished record exists, otherwise computed and written the moment it ends. A
    failure, or a case past ``limit`` seconds (SIGALRM), is recorded with its traceback and the run goes on. Arrays the case returns
    under ``_arrays`` go to the ``.npz`` beside its record.
    """
    rec = store.read(key)
    if rec is not None and rec.get("status") == "done":
        log(f"reuse {key}")
        return rec
    log(f"start {key} (limit {limit} s)")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit)
    try:
        res = {"status": "done", **compute(*args)}
    except Exception as err:  # noqa: BLE001 -- a failed case is recorded with its traceback and the run goes on
        res = {"status": "failed", "failure": failure(err)}
        log(f"FAILED {key}: {err!r}")
    finally:
        signal.alarm(0)
    arrays = res.pop("_arrays", None)
    return store.write(key, json.loads(json.dumps(res, default=float)), arrays)


def best_record(problem: str, tol: float = 0.01) -> dict | None:
    """The finished benchmark record with the smallest worst rollout for ``problem`` at ``tol`` (seed 1; read only)."""
    best = None
    for f in glob.glob(str(HERE.parents[1] / "results" / "benchmark" / "runs" / "step_maps" / "cpu-1rank" / problem / "*" / "*" / f"tol{tol}-s1.json")):
        r = json.loads(Path(f).read_text())
        if r.get("status", "done") != "done" or "failed" in r or r.get("rollout_worst") is None:
            continue
        if best is None or r["rollout_worst"] < best["rollout_worst"]:
            best = {"method": r["method"], "rollout_worst": r["rollout_worst"], "one_step_worst": r.get("worst"), "evals": r.get("evals")}
    return best
