# The schemes

Global approximants of an expensive function, beside the library's adaptive surrogate (AFINNI and ART):
each samples the oracle where its own error estimate asks and fits one model, aiming at the **worst
pointwise error** within a tolerance with few oracle evaluations and fast inference. They live in
`src/schemes`, standalone: they import only the library's infrastructure (`src`, `infra`, `numerics`,
`kernels`, `core`), never AFINNI, ART or the refinement engine. The benchmark that compares them with
AFINNI and ART is in [BENCHMARK.md](BENCHMARK.md); the experiments that led to them, what each taught and
the literature behind them are in [SURROGATE_RESEARCH.md](SURROGATE_RESEARCH.md) and
[SURROGATE_ITERATIONS.md](SURROGATE_ITERATIONS.md).

Every scheme has a frozen settings dataclass and ``build(f, lo, hi, atol, settings=None, *, jacobian=None,
stream_dir=None, **overrides)``; its model predicts on host arrays, counts its training evaluations in
``evals`` and carries a ``TrainingReport`` (evaluations, time per phase, one record per round) in
``report``. ``src.schemes.base.save`` and ``load`` write and read any trained model.

## Files

| File | Contents |
|---|---|
| `src/schemes/base.py`, `oracle.py`, `profiling.py` | settings, save/load and check points; the oracle (counts, Jacobians, the distributed oracle under MPI); phases, rounds and the training report |
| `src/schemes/chebyshev.py`, `directions.py` | Chebyshev arithmetic and gradient directions the schemes share |
| `src/schemes/chebtree.py` | **ChebTree**: an adaptive tree of tensor Chebyshev boxes, refined by the decay of their coefficients: per-axis geometric and algebraic decay fits give the error estimate, the degree where the decay is exponential and a split where it is not; a few check points per box, seeded by its locational code, catch aliasing; trailing coefficients are chopped after acceptance. Also `build_rotated`, ChebTree in active-subspace coordinates |
| `src/schemes/chebtt/` (`model.py`, `train.py`) | **ChebTT**: a Chebyshev interpolant in tensor-train form, built by TT-cross (maxvol fibres); cost linear in d; accepted on a Sobol set, free checks at earlier sweeps' grid points and a next-nested-grid look-ahead through the worst point; degrees doubled where an axis's upper degrees still matter, else rank raised. Vector outputs (`build_vector`): one train per output on one shared cache of f; with `rotate`, outputs that are ridges of shared oblique coordinates are trained in the eigenvectors of their pullback metric (benchmark method `chebtt_metric`, ChebTT-M) |
| `src/schemes/chebtt/adaptive.py` | **ChebTT-A**: ChebTT with piecewise axes, plus one interface term where f jumps across an oblique or curved interface or has a sharp front |
| `src/schemes/hub.py` | **the hub expansion**: per output group, a dense Chebyshev hub in the steep directions of the gradient covariance and degree-1/2 weak terms in the rest, fitted by least squares; uses the oracle's gradients (`jac`) when given, else a finite-difference pilot |
| `src/schemes/kernel.py` | **MetricKernel**: greedy Matern-5/2 kernel interpolation in a metric learned from the model's gradients (a prototype that loses to ChebTT above 2-D) |
| `src/schemes/sparsegrid.py` | **SparseCheb**: a dimension-adaptive sparse Chebyshev grid (Smolyak, Gerstner-Griebel), stopped by its surpluses and a Sobol-plus-corners check at half the tolerance; a failed check refines every active index once (a one-step look-ahead) |
| `src/schemes/tabulation.py` | **SlowTable**: in-situ tabulation of a step map over the states a simulation visits (ISAT's design: linear records in ellipsoids of accuracy) |

Problems: `rosenbrock`, `gaussian`, `wave`, `front` (a sharp tanh front on a sphere); the Genz
families `genz_oscillatory`, `genz_product_peak`, `genz_corner_peak`, `genz_gaussian`,
`genz_continuous`, `genz_discontinuous`; `ackley`, `rastrigin`, `griewank`, `styblinski_tang`,
`michalewicz`; the jumps `oblique_jump` and `ball_jump`; and the ODE flow maps `lorenz96`
(forcing 1, t = 0.1, the manuscript study's), `lorenz96_chaotic` (forcing 8, t = 0.5) and
`lotka_volterra` (a competitive network). All of them are defined for any d.

## Running

The benchmark that compares the schemes with AFINNI and ART is described in [BENCHMARK.md](BENCHMARK.md).

## Results so far (2-D, tolerances 0.1, 0.01, 0.001 of max |f|, 3 seeds, 8 ranks)

AFINNI and both ART variants exceed the tolerance at their worst points on every problem, typically
by 1.5-5x for ART with quadratic leaves, more on fronts (13-24x), and by 10-2,800x for AFINNI.

| Problem, tolerance 0.001 | ART, quadratic leaves: evaluations / worst | ChebTree | SparseCheb |
|---|---|---|---|
| gaussian | 18,462 / 2.7 | 2,331 / 0.07 | 1,748 / 0.59 (one seed misses) |
| wave | 5,896 / 1.7 | 1,399 / 1e-10 | 217 / 0.68 |
| genz_product_peak | 13,948 / 2.8 | 1,035 / 0.06 | 361 / 0.23 |
| griewank | 245,504 / 3.4 | 4,743 / 0.002 | 361 / 0.6 |
| rastrigin | 1,046,840 / 4.4 | 41,391 / 0.003 | 301 / exact |
| front | 186,293 / 17 | 25,775 / 0.07 | 66,601 / 0.03 |
| genz_continuous (kinks) | 25,578 / 24 | 9,315 / 1.4 | 231,465 / 0.67 |

Rosenbrock and Styblinski-Tang (polynomials) are reproduced exactly by both schemes with 61-81
evaluations. Inference is 0.1-1 us per query for every method except SparseCheb at tight
tolerances (up to 10 us). The full tables are in the figures' `summary-d2.md` after a run.

## Known weaknesses (the next work)

1. **ChebTree aliasing at coarse resolution.** On Ackley and Michalewicz at tolerance 0.1 it accepts
   9 x 9 boxes that cannot see the oscillation (worst 1.4-1.8x). Fix: test a box at a few
   independent points before accepting it, and double its degree if they fail.
2. **ChebTree at kinks.** Coefficients decay algebraically there, and the tail estimate (the two
   highest degrees) is optimistic (genz_continuous, worst 1.4-3.7x). Fix: estimate each axis's
   decay rate, extrapolate the tail with it, and split where the decay is slow instead of raising
   the degree. Also chop trailing coefficients after acceptance (faster inference), and avoid
   overshooting smooth functions (wave reaches 1e-10 of the tolerance: wasted evaluations).
3. **SparseCheb's stopping test.** The surplus estimate stops early where some interaction is never
   opened, and 20 d random check points missed small regions (gaussian, genz_corner_peak, griewank,
   worst 1.0-2.1x). A Sobol-plus-corners check at half the tolerance, with a one-step look-ahead when
   it fails, removes those misses; it costs 1.3-3x the evaluations.
4. **SparseCheb at kinks and jumps** needs one global expansion to resolve them: 1e5-1e6 evaluations
   on Ackley, genz_continuous and the jumps. A tree of sparse grids (ChebTree's splitting with a
   sparse grid in each box) would localise them.
5. **Dimension.** Tensor ChebTree costs (n + 1)^d points per box: it reaches d = 3 easily, d = 4 only
   for simple functions. SparseCheb reaches d = 5-10 for functions of low interaction order
   (rosenbrock, lotka_volterra, lorenz96 with the check), but not for oblique features (the 5-D wave
   exceeds 2 million evaluations). Rotation to active-subspace coordinates (`build_rotated`) helps
   ridges only and evaluates f outside the box.

Plan, in stages, each started only when the previous one is satisfactory: fix 1-3 and rerun 2-D;
then d = 3, 5, 10 at the looser tolerances of `TOLERANCES`; tighten them once a scheme holds there;
then chemistry-like problems trained on trajectory data (Robertson, a primordial H/He/H2 network).

A finding about the engine, from the distributed experiment on a GH200 node: with work stealing on
and off, AFINNI's trees, evaluation counts and times are identical from 2 to 64 ranks at tolerance
0.05, so stealing rarely or never triggers there; worth examining with the engine's event
profiling before relying on it.
