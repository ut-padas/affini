"""
The recipe (common.py) on KIDA uva 2024 (KIDApy, 578 species with its three pseudo-grain species, 8,275 reactions), set up as the
benchmark sets up Nelson: abundances per H nucleus and the environment (n_H, T, A_v, UV) in log10, the environment held over a 66.7 kyr
step and driven between steps along each tracer (`astro.environment_path`'s oscillation about a random starting environment), burn-in
1e4 yr from KIDApy's initial abundances, 24 steps, species below 1e-12 not scored. Two environment ranges (log-uniform):
  diffuse  Nelson's: n_H 1e2-1e4 cm^-3, T 50-3000 K, A_v 0.1-2 mag, UV 50-200 Draine;
  dense    n_H 1e3-1e5 cm^-3, T 10-50 K, A_v 3-10 mag, UV 1-10 Draine (a dark cloud, where far more species are abundant).
200 training and 40 test tracers per range, generated in chunks of 24 (one record each, with its trajectories beside it, under
data/records/r3/), then the recipe per range (kmax 20, a 40-minute limit). Usage: r3_kida.py [diffuse|dense ...].

Predictions, committed before the run (tolerance 0.01 dex; a two-state smoke test had shown 28-57 species above the floor in the diffuse
range and 120-128 in the dense range):
  K1  the chart needs far fewer coordinates than species are scored: k <= 15 in the diffuse range and k <= 25 in the dense range, while
      at least 30 (diffuse) and 80 (dense) species are scored on average;
  K2  the rule fails in both ranges: the degree is capped at 3-4 by the pairs (4,800) against k + 4 chart coordinates, and the test
      rollout errs 2 to 30 times the tolerance;
  K3  both mixing rollouts fail the tolerance in both ranges.
"""

import json
import sys

import numpy as np
from common import OUT, log, recorded, run_recipe

from src.infra.records import RecordStore
from src.problems.ode import astro
from src.problems.ode.step_maps import StepMapProblem
from src.problems.ode.system import SYSTEMS, trajectories

RANGES = {"diffuse": ((2, 4), (np.log10(50), np.log10(3000)), (-1, np.log10(2)), (np.log10(50), np.log10(200))),
          "dense": ((3, 5), (1, np.log10(50)), (np.log10(3), 1), (0, 1))}  # fmt: skip
CHUNK, N_TRAIN, N_TEST = 24, 200, 40
_net, _x0 = astro._load("kida.uva.2024/gas_reactions_kida.uva.2024.in", "kida.uva.2024/abundances.in", grains=True)
KIDA = astro.kidapy_system("kida_uva_2024", _net)
SYSTEMS[KIDA.name] = KIDA  # the step pool's workers look systems up by name


def environment(n: int, seed: int, rng_range) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([10.0 ** rng.uniform(lo, hi, n) for lo, hi in rng_range], axis=1)


def environment_path(n: int, seed: int, steps: int, rng_range) -> np.ndarray:
    """`astro.environment_path`'s oscillation (amplitudes 0.4 / 0.3 / 0.2 / 0.1 dex, periods 8-30 steps) about this range's start."""
    rng = np.random.default_rng(seed + 7919)
    base = np.log10(environment(n, seed, rng_range))
    amp = np.array([0.4, 0.3, 0.2, 0.1])
    period = rng.uniform(8, 30, (n, 1, 4))
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 4))
    k = np.arange(steps + 1)[None, :, None]
    return 10.0 ** (base[:, None, :] + amp * (np.sin(2 * np.pi * k / period + phase) - np.sin(phase)))


def problem(name: str) -> StepMapProblem:
    rr = RANGES[name]
    nel_dt = 1600.0e3 * astro.YEAR / 24
    return StepMapProblem(KIDA, nel_dt, lambda n, seed: np.concatenate([np.tile(np.maximum(_x0, 1e-20), (n, 1)), environment(n, seed, rr)], axis=1),
                          24, N_TRAIN, N_TEST, burn=1.0e4 * astro.YEAR, floor=1e-12, regime="longstep",
                          held_path=lambda n, seed, steps: environment_path(n, seed, steps, rr))  # fmt: skip


def chunk(P: StepMapProblem, seed: int) -> dict:
    Y = P.initial(CHUNK, seed)
    Y = P.system.step(Y, P.burn)
    Z = trajectories(P.system, Y, P.dt, P.steps, P.held_path(CHUNK, seed, P.steps))
    if not np.isfinite(Z).all():
        raise FloatingPointError(f"{int((~np.isfinite(Z)).any(axis=(1, 2)).sum())} tracers with a failed solve")
    return {"tracers": CHUNK, "_arrays": {"Z": Z}}


def dataset(name: str, store: RecordStore) -> tuple[np.ndarray, np.ndarray]:
    P = problem(name)
    parts = {}
    for role, n, base in (("train", N_TRAIN, 1000), ("test", N_TEST, 9000)):
        Zs = []
        for c in range(n // CHUNK + (n % CHUNK > 0)):
            rec = recorded(store, {"problem": f"kida_{name}", "role": role, "chunk": c, "seed": base + c}, chunk, P, base + c, limit=1800)
            if rec.get("status") != "done":
                raise RuntimeError(f"chunk {role} {c} failed: {rec.get('failure', {}).get('message')}")
            Zs.append(np.load(store.path(rec["key"]).with_suffix(".npz"))["Z"])
            log(f"kida_{name} {role}: chunk {c} ready ({sum(len(z) for z in Zs)} tracers)")
        parts[role] = np.concatenate(Zs)[:n]
    return parts["train"], parts["test"]


def case(name: str, train: np.ndarray, test: np.ndarray) -> dict:
    P = problem(name)
    scored = P.scored(train[:, 1:]).sum(axis=2)
    out = run_recipe(P, train, test, kmax=20, qmax=8)
    return {"scored_species_mean": float(scored.mean()), "scored_species_max": int(scored.max()), **out}


def main() -> None:
    names = sys.argv[1:] or list(RANGES)
    data_store = RecordStore(OUT / "records" / "r3_data", path_of=lambda k: OUT / "records" / "r3_data" / f"{k['problem']}_{k['role']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "r3", path_of=lambda k: OUT / "records" / "r3" / f"{k['problem']}.json")
    for name in names:
        train, test = dataset(name, data_store)
        log(f"kida_{name}: {train.shape} training, {test.shape} test")
        recorded(store, {"problem": f"kida_{name}", "method": "recipe", "tol": 0.01}, case, name, train, test, limit=2400)
    fields = ("status", "d", "k", "degree", "passes", "scored_species_mean", "one_step_worst", "rollout_worst", "mixing_rollout_worst", "solves",
              "seconds", "k_curve", "degree_curve")  # fmt: skip
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r3_kida.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
