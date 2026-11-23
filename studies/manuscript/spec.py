"""
The manuscript study's specification: every experiment of the paper as data, from which the case
list, the runner's order and the tables follow. The companion's Appendix A states the same
settings in prose; a change here is a change there.

A case is one training run (and its predictions), identified by its settings alone. Experiments
are tags on cases, so one run serves every experiment it belongs to (the Lorenz-96 d = 5 case at
tol 0.1 is in both the convergence and the dimension experiments). The convergence, dimension and
absolute-tolerance experiments also run ART, with affine and with quadratic leaves, on the same
samples, tolerances and test points as AFINNI.
"""

import math
from dataclasses import asdict, dataclass, field

TOLERANCES = (0.1, 0.05, 0.02, 0.01, 0.005)
SEEDS = (0, 1, 2, 3, 4)
# Candidates of the nearest-centre search; flags and predictions are completed by the exact search.
CANDIDATES = 50
N_TEST = 100_000
# The fixed test set of a (function, d) is drawn from this seed and the pair, never from the run's.
TEST_SEED = 20_261_117
MODES = {"hessian": "affine_cov_hess", "no_hessian": "affine_cov_no_hess", "exact_gradient": "affine_cov_hess_exact_grad"}
# The methods: AFINNI, and ART with the local model of each name (ARTSettings' defaults otherwise).
ART_METHODS = {"art_affine": "affine_lsq", "art_quadratic": "quadratic_lsq"}
# The deepest tree ART's locational codes allow (src.art.constants.MAX_DEPTH_LIMIT); AFINNI's levels are not limited.
ART_MAX_LEVELS = 62
# Iterative refinement: rounds, and the points each round adds, as a share of N_0.
ROUNDS = 10
ROUND_SHARE = 0.4


@dataclass(frozen=True)
class Function:
    """A test function: its name in the companion, its domain [lo, hi]^d and the dimensions it runs at."""

    name: str
    domain: tuple[float, float]
    dims: tuple[int, ...]


FUNCTIONS = {
    "quadratic": Function("Quadratic", (-3.0, 3.0), (2, 5, 10)),
    "gaussian": Function("Gaussian", (-3.0, 3.0), (2, 5, 10)),
    "rosenbrock": Function("Rosenbrock", (-2.0, 3.0), (2, 5, 10)),
    "wave": Function("Wave", (-5.0, 5.0), (2, 5, 10)),
    "lorenz96": Function("Lorenz-96", (-5.0, 5.0), (5, 10, 20, 40, 80)),
}


def sizes(d: int, n0: int | None = None) -> tuple[int, int, int]:
    """
    (N_0, K_0, M) at dimension d: N_0 = min(5e4 1.3^d, 1e6) unless given, then M = 100 1.3^d lowered
    until 16 clusters of 2 M points fit in N_0, and K_0 = 25 1.3^d lowered until K_0 clusters do. Each
    initial cluster is topped up to 2 M points before it is split, so the level-0 budget fits the sample.
    """
    n0 = min(int(5e4 * 1.3**d), 10**6) if n0 is None else int(n0)
    m = min(int(100 * 1.3**d), max(2, n0 // (2 * 16)))
    k0 = min(int(25 * 1.3**d), max(1, n0 // (2 * m)))
    return n0, k0, m


@dataclass(frozen=True)
class Case:
    """
    One run's settings. ``tol`` sets a_tol = r_tol = tol (Lorenz-96: a_tol = tol max|f|, absolute only);
    ``atol`` alone sets an unscaled absolute test (the absolute-tolerance experiment). ``rounds`` > 0
    makes it an iterative-refinement run. ``method`` is "afinni" or a key of ART_METHODS; an ART case
    keeps ``mode`` "hessian" (its splits come from ARTSettings.split_search) and ignores ``k0`` and ``m``.
    """

    function: str
    d: int
    mode: str
    seed: int
    n0: int
    k0: int
    m: int
    tol: float | None = None
    atol: float | None = None
    ranks: int = 1
    work_stealing: bool = True
    rounds: int = 0
    method: str = "afinni"
    experiments: tuple[str, ...] = field(default=(), compare=False)

    @property
    def key(self) -> str:
        """A file-name-safe identity from the settings alone (experiments excluded); an ART case names its method where AFINNI's names its mode."""
        test = f"tol{self.tol:g}" if self.tol is not None else f"atol{self.atol:g}"
        extra = (f"_n0{self.n0}" if (self.n0, self.k0, self.m) != sizes(self.d) else "") + (
            f"_np{self.ranks}{'' if self.work_stealing else '_static'}" if self.ranks > 1 or not self.work_stealing else "") + (
            f"_rounds{self.rounds}" if self.rounds else "")  # fmt: skip
        return f"{self.function}_d{self.d}_{test}_{self.mode if self.method == 'afinni' else self.method}{extra}_s{self.seed}"

    @property
    def series(self) -> tuple:
        """The cases that differ only in tolerance (and seed): the runner projects costs along one."""
        return (self.function, self.d, self.mode, self.n0 if self.atol is not None else None, self.ranks,
                self.work_stealing, self.rounds, self.method)  # fmt: skip

    def as_dict(self) -> dict:
        return {**asdict(self), "key": self.key}


def _case(function: str, d: int, mode: str, seed: int, experiment: str, n0: int | None = None, **kw) -> Case:
    return Case(function, d, mode, seed, *sizes(d, n0), experiments=(experiment,), **kw)


def cases() -> list[Case]:
    """Every case of the study, each once, with the experiments it serves."""
    out = []
    for name, fn in FUNCTIONS.items():
        for d in fn.dims if name != "lorenz96" else (5,):
            for tol in TOLERANCES:
                for mode in ("hessian", "no_hessian"):
                    out += [_case(name, d, mode, s, "convergence", tol=tol) for s in SEEDS]
    out += [_case("lorenz96", d, "hessian", s, "dimension", tol=0.1) for d in FUNCTIONS["lorenz96"].dims for s in SEEDS]
    for k in range(4):  # a_tol halves and N_0 grows by 2^(5/2) per step, so N_0 ~ a_tol^(-5/2)
        out += [_case("rosenbrock", 5, "hessian", s, "absolute", n0=round(5e4 * 2 ** (2.5 * k)), atol=579.8 / 2**k)
                for s in SEEDS]  # fmt: skip
    out += [_case("wave", 5, "exact_gradient", s, "gradient", tol=tol) for tol in TOLERANCES for s in SEEDS]
    for method in ART_METHODS:  # the experiments of errors and evaluations, as AFINNI's hessian runs
        for name, fn in FUNCTIONS.items():
            for d in fn.dims if name != "lorenz96" else (5,):
                out += [_case(name, d, "hessian", s, "convergence", tol=tol, method=method) for tol in TOLERANCES for s in SEEDS]
        out += [_case("lorenz96", d, "hessian", s, "dimension", tol=0.1, method=method)
                for d in FUNCTIONS["lorenz96"].dims for s in SEEDS]  # fmt: skip
        for k in range(4):
            out += [_case("rosenbrock", 5, "hessian", s, "absolute", n0=round(5e4 * 2 ** (2.5 * k)), atol=579.8 / 2**k,
                          method=method) for s in SEEDS]  # fmt: skip
    out += [_case("rosenbrock", 10, "hessian", 0, "progression", tol=0.05)]
    for tol in (0.05, 0.005):
        for p in (1, 2, 4, 8, 16, 32, 64, 128):
            for ws in (True, False):
                out += [_case("rosenbrock", 5, "hessian", s, "distributed", n0=10**5, tol=tol, ranks=p, work_stealing=ws)
                        for s in (0, 1, 2)]  # fmt: skip
    out += [_case(name, 2, "hessian", s, "iterative", tol=0.05, rounds=ROUNDS)
            for name in ("quadratic", "gaussian", "rosenbrock", "wave") for s in SEEDS]  # fmt: skip
    merged: dict[str, Case] = {}
    for c in out:
        prev = merged.get(c.key)
        merged[c.key] = c if prev is None else Case(**{**asdict(prev), "experiments": tuple(sorted(set(prev.experiments) | set(c.experiments)))})
    return list(merged.values())


def cost_order(case: Case) -> tuple:
    """
    The runner's order: seed 0 of everything before the other seeds (a series' first seed calibrates
    the projected cost of the others), then cheap before expensive: small d, loose tolerances, few ranks.
    """
    tol = case.tol if case.tol is not None else case.atol / 5798
    return (case.seed != 0, case.seed, case.rounds > 0, case.d, -math.log(tol), case.ranks, case.mode, case.function, case.method)
