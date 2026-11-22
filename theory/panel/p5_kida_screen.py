"""
The KIDA screen (council: Sol's and Astra's first experiment, Fable's C5-C7): candidate surrogates of KIDA uva 2024's step map in the
charge-neutral dark range (theory/recipe/r3_kida.py; 578 species and 4 held environment inputs), on equal data, scored on the 13 key species
of theory/KIDA_QOI.md at 0.01 dex wherever the true abundance exceeds 1e-12 (the strict metric over every species above 1e-12 beside it).

Data: the dark range's 200 training tracers (24 steps each), split by whole tracers into fit (160), calibration (20) and selection (20);
the 40 test tracers for the scores. Inputs are log10 abundances clipped at 1e-14 and the four log10 environment inputs; each arm predicts the
increment of every species (clipped targets), so every species stays in the state.

Arms (each with its own refusal test, calibrated for the maximum on the calibration tracers and on their mixed inputs, w = 0.01 and 0.1):
  persistence  no change (the floor of usefulness); never refuses;
  linear       ridge least squares from the 582 inputs and a constant to the increments; refuses by the Lebesgue bound of Proposition 5.1
               (exact for a fit linear in its data);
  chart2       the same with the inputs replaced by a total-degree-2 polynomial of the top 20 principal coordinates of the clipped
               abundances and the 4 environment inputs (the generic recipe's class); Lebesgue bound;
  local        for each query, ridge least squares on its 64 nearest fitted states (principal coordinates and environment, scaled); refuses
               by the distance to the 8th neighbour;
  mlp          five multilayer perceptrons (scikit-learn, two hidden layers of 256, ReLU, early stopping, seeds 0-4) on standardised inputs
               and increments, the mean as the prediction; refuses by the ensemble's spread (largest standard deviation over the key
               species).
Gate calibration: for a candidate threshold on the arm's risk score, the worst key-species error over the accepted calibration states
(plain and mixed); the largest threshold whose accepted worst stays within half the tolerance (none: refuse everything). Scores on the test
tracers: one-step accepted worst (key and strict) and refused share, on plain and mixed (w = 0.01, 0.1) inputs; rollouts over the 40 test
tracers with the solver stepping refused inputs (worst key-species error, solver share); training time, prediction and gate time per state.
One record per arm under data/records/p5/; a case is limited to 1,800 s. Usage: p5_kida_screen.py [arm ...].

Predictions, committed before the run (informed by the council's post hoc checks on the dense range: persistence worst 3 dex, a held-out
state 0.68 dex from its nearest training state; this is the charge-neutral dark range, which no run has fitted):
  K1  persistence and the linear fit err more than 0.1 dex (key species) on the accepted test inputs at any refusal below 50%;
  K2  no arm holds 0.01 dex on the key species over the accepted test inputs with less than 50% refused (200 tracers are too few);
  K3  the MLP has the smallest median one-step error, but its calibrated spread threshold does not hold the tolerance on the test tracers
      (accepted worst above 0.01) or refuses more than half;
  K4  the gated linear and chart arms keep their accepted test worst within the tolerance wherever they accept anything (their gate is an
      inequality), at a refusal above 80%.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
import r3_kida  # noqa: E402,F401 -- registers the KIDA system before any solver pool is forked
from common import log, recorded  # noqa: E402
from generic import mixing  # noqa: E402
from r4_generic import kida  # noqa: E402

from src.benchmark.step_maps import rollout  # noqa: E402
from src.infra.records import RecordStore  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
QOI = ["H2", "CO", "C", "C+", "O", "OH", "H2O", "e-", "H3+", "HCO+", "N2H+", "S+", "HCN"]
CUT = -14.0
N_SP = 578


def clip(Z):
    out = np.array(Z, dtype=float)
    out[..., :N_SP] = np.maximum(out[..., :N_SP], CUT)
    return out


class Data:
    """The dark range's tracers, split by tracer, with the key-species mask and the features each arm needs."""

    def __init__(self):
        self.P, train, self.test = kida("dark")
        rng = np.random.default_rng(0)
        order = rng.permutation(len(train))
        self.fit, self.cal, self.sel = train[order[:160]], train[order[160:180]], train[order[180:]]
        sp = list(r3_kida._net.species)
        self.key = np.zeros(self.test.shape[2], dtype=bool)
        self.key[[sp.index(q) for q in QOI]] = True
        X = clip(self.fit[:, :-1].reshape(-1, self.fit.shape[2]))
        self.mu = X[:, :N_SP].mean(0)
        self.V = np.linalg.svd(X[:, :N_SP] - self.mu, full_matrices=False)[2][:20].T

    def pairs(self, trajs):
        d = trajs.shape[2]
        return trajs[:, :-1].reshape(-1, d), trajs[:, 1:].reshape(-1, d)

    def mixed(self, trajs, rng):
        """Each state mixed once (w = 0.01 and 0.1) with another trajectory's state at the same step, and its solver step."""
        from generic2 import mixed_once

        X = np.concatenate([mixed_once(self.P, trajs, w, rng) for w in (0.01, 0.1)])
        return X, self.P.step(X)

    def errors(self, Zpred, Ztrue):
        """(key-species worst, strict worst) per state; an entry counts where the truth exceeds 1e-12."""
        above = Ztrue > -12
        e = np.abs(Zpred - Ztrue)
        key = np.max(np.where(above & self.key, e, 0.0), axis=1)
        strict = np.max(np.where(self.P.scored(Ztrue), e, 0.0), axis=1)
        return key, strict

    def chart(self, Z):
        Zc = clip(Z)
        return np.concatenate([(Zc[:, :N_SP] - self.mu) @ self.V, Zc[:, N_SP:]], axis=1)


def features(D: Data, Z, arm):
    Zc = clip(Z)
    if arm == "linear":
        return np.concatenate([Zc, np.ones((len(Z), 1))], axis=1)
    U = D.chart(Z)
    cols = [np.ones(len(U))] + [U[:, i] for i in range(U.shape[1])]
    cols += [U[:, i] * U[:, j] for i in range(U.shape[1]) for j in range(i, U.shape[1])]
    return np.stack(cols, axis=1)


class LinearArm:
    """Ridge least squares on fixed features; risk = the Lebesgue bound sqrt(N phi^T (A^T A + lam I)^-1 phi)."""

    def __init__(self, D, arm, X, Y, lam=1e-6):
        self.D, self.arm = D, arm
        A = features(D, X, arm)
        self.sc = np.maximum(np.abs(A).max(0), 1e-12)
        A = A / self.sc
        self.N = len(A)
        G = A.T @ A
        G += lam * np.trace(G) / len(G) * np.eye(len(G))
        self.L = np.linalg.cholesky(G)
        self.W = np.linalg.solve(self.L.T, np.linalg.solve(self.L, A.T @ (clip(Y)[:, :N_SP] - clip(X)[:, :N_SP])))

    def predict(self, Z):
        out = np.array(Z, dtype=float)
        out[:, :N_SP] = clip(Z)[:, :N_SP] + (features(self.D, Z, self.arm) / self.sc) @ self.W
        return out

    def risk(self, Z):
        Phi = features(self.D, Z, self.arm) / self.sc
        S = np.linalg.solve(self.L, Phi.T)
        return np.sqrt(self.N * np.sum(S * S, axis=0))


class LocalArm:
    """Ridge least squares on the 64 nearest fitted states per query (scaled chart coordinates); risk = distance to the 8th neighbour."""

    def __init__(self, D, X, Y):
        from sklearn.neighbors import NearestNeighbors

        self.D = D
        U = D.chart(X)
        self.sc = np.maximum(U.std(0), 1e-6)
        self.U = U / self.sc
        self.dZ = clip(Y)[:, :N_SP] - clip(X)[:, :N_SP]
        self.nn = NearestNeighbors(n_neighbors=64).fit(self.U)

    def predict(self, Z):
        Uq = self.D.chart(Z) / self.sc
        dist, idx = self.nn.kneighbors(Uq)
        out = np.array(Z, dtype=float)
        for i in range(len(Z)):
            A = np.concatenate([self.U[idx[i]] - Uq[i], np.ones((64, 1))], axis=1)
            G = A.T @ A + 1e-6 * np.eye(A.shape[1])
            coef = np.linalg.solve(G, A.T @ self.dZ[idx[i]])
            out[i, :N_SP] = clip(Z[i : i + 1])[0, :N_SP] + coef[-1]
        return out

    def risk(self, Z):
        return self.nn.kneighbors(self.D.chart(Z) / self.sc, n_neighbors=8)[0][:, -1]


class MLPArm:
    """Five MLPs on standardised inputs and increments; the mean predicts, the spread over the key species is the risk."""

    def __init__(self, D, X, Y):
        from sklearn.neural_network import MLPRegressor

        self.D = D
        Xi = clip(X)
        T = clip(Y)[:, :N_SP] - Xi[:, :N_SP]
        self.xm, self.xs = Xi.mean(0), np.maximum(Xi.std(0), 1e-6)
        self.tm, self.ts = T.mean(0), np.maximum(T.std(0), 1e-6)
        self.nets = [MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, max_iter=300, random_state=s).fit((Xi - self.xm) / self.xs, (T - self.tm) / self.ts)
                     for s in range(5)]  # fmt: skip
        self.keycols = np.flatnonzero(D.key[:N_SP])

    def _all(self, Z):
        Xi = (clip(Z) - self.xm) / self.xs
        return np.stack([n.predict(Xi) * self.ts + self.tm for n in self.nets])

    def predict(self, Z):
        out = np.array(Z, dtype=float)
        out[:, :N_SP] = clip(Z)[:, :N_SP] + self._all(Z).mean(0)
        return out

    def risk(self, Z):
        return self._all(Z)[:, :, self.keycols].std(0).max(1)


class Persistence:
    def predict(self, Z):
        return np.array(Z, dtype=float)

    def risk(self, Z):
        return np.zeros(len(Z))


class Gated:
    """A rollout-ready surrogate: increments from the arm's prediction, refusal by its risk against the calibrated threshold."""

    def __init__(self, arm, thr):
        self.arm, self.thr, self.evals = arm, thr, 0

    def refuses(self, Z):
        r = self.arm.risk(Z)
        return ~np.isfinite(r) | (r > self.thr)

    def predict(self, Z):
        return self.arm.predict(Z) - Z


def calibrate(D, arm, sets):
    """The largest threshold on the arm's risk whose accepted calibration states all err within half the tolerance (key species)."""
    risk = np.concatenate([arm.risk(X) for X, _ in sets])
    err = np.concatenate([D.errors(arm.predict(X), Y)[0] for X, Y in sets])
    order = np.argsort(risk)
    bad = np.flatnonzero(err[order] > TOL / 2)
    if not len(bad):
        return float(np.inf)
    first = bad[0]
    return float(risk[order][first - 1]) if first > 0 else -np.inf


def case(name: str) -> dict:
    t0 = time.perf_counter()
    D = Data()
    Xf, Yf = D.pairs(D.fit)
    rng = np.random.default_rng(1)
    cal_sets = [D.pairs(D.cal), D.mixed(D.cal, rng)]
    build = {"persistence": lambda: Persistence(), "linear": lambda: LinearArm(D, "linear", Xf, Yf), "chart2": lambda: LinearArm(D, "chart2", Xf, Yf),
             "local": lambda: LocalArm(D, Xf, Yf), "mlp": lambda: MLPArm(D, Xf, Yf)}[name]  # fmt: skip
    arm = build()
    train_s = time.perf_counter() - t0
    thr = float("inf") if name == "persistence" else calibrate(D, arm, cal_sets)
    sur = Gated(arm, thr)
    out = {"arm": name, "threshold": thr, "train_s": train_s, "sets": {}}
    test_sets = {"test": D.pairs(D.test), "mixed": D.mixed(D.test, np.random.default_rng(2))}
    for label, (X, Y) in test_sets.items():
        pred = arm.predict(X)
        key, strict = D.errors(pred, Y)
        r = np.asarray(sur.refuses(X), dtype=bool)
        out["sets"][label] = {"refused_percent": float(100 * r.mean()), "key_worst_accepted": float(key[~r].max()) if (~r).any() else None,
                              "strict_worst_accepted": float(strict[~r].max()) if (~r).any() else None, "key_median_all": float(np.median(key)),
                              "key_worst_all": float(key.max()), "key_above_tol_percent_all": float(100 * np.mean(key > TOL))}  # fmt: skip
        log(f"  {name} {label}: refused {out['sets'][label]['refused_percent']:.1f}%, accepted worst {out['sets'][label]['key_worst_accepted']}, "
            f"ungated median {np.median(key):.3e}, worst {key.max():.3e}")  # fmt: skip
    D.P.scored = lambda Zt: (Zt > -12) & D.key  # the rollout scores the key species
    roll = rollout(D.P, sur, D.test)
    out["rollout_worst_key"] = float(roll["rollout_worst"])
    out["rollout_solver_share"] = float(np.mean(roll["rollout_solved_by_step"]))
    if out["rollout_solver_share"] < 1:
        mix = {str(w): mixing(D.P, sur, D.test, w, np.random.default_rng(3)) for w in (0.01, 0.1)}
        out["mixing_chained"] = mix
    X = D.pairs(D.test)[0][:960]
    t1 = time.perf_counter()
    arm.predict(X)
    out["predict_us"] = 1e6 * (time.perf_counter() - t1) / len(X)
    t1 = time.perf_counter()
    arm.risk(X)
    out["gate_us"] = 1e6 * (time.perf_counter() - t1) / len(X)
    log(f"  {name}: rollout worst {out['rollout_worst_key']:.3e} (solver {100 * out['rollout_solver_share']:.0f}%), train {train_s:.0f} s, "
        f"predict {out['predict_us']:.0f} us, gate {out['gate_us']:.0f} us")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or ["persistence", "linear", "chart2", "local", "mlp"]
    store = RecordStore(OUT / "records" / "p5", path_of=lambda key: OUT / "records" / "p5" / f"{key['arm']}.json")
    for name in names:
        recorded(store, {"arm": name, "problem": "kida_dark", "tol": TOL}, case, name, limit=1800)
    summary = {r["key"]["arm"]: {k: v for k, v in r.items() if k not in ("key", "provenance", "schema", "recorded")} for r in store.records()}
    (OUT / "p5_kida_screen.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
