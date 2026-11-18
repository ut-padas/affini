# Structure experiments: Chebyshev surrogates against AFINNI and ART

An experiment, outside the library's public API. It asks which surrogate scheme keeps the **worst
pointwise error** below a tolerance with the fewest oracle evaluations, and compares new
Chebyshev-based schemes with AFINNI and ART on standard test functions and ODE flow maps. The end
goal is a surrogate for astrochemical reaction networks with hundreds of species, called by MHD
simulations in place of the chemistry time steps, so the worst error and the inference time both
matter.

The experiments that led here, what each one taught, and three literature reviews behind them
are in [docs/SURROGATE_RESEARCH.md](../../docs/SURROGATE_RESEARCH.md).

## Files

| File | Contents |
|---|---|
| `problems.py` | the problems (`PROBLEMS`), the tolerances per dimension (`TOLERANCES`), the test sets: a 300 x 300 grid in 2-D, 50,000 fixed uniform points above |
| `bench.py` | runs, caches and plots every method (`METHODS`): `afinni`, `art`, `art_quad` (ART with quadratic leaves) and the registered schemes |
| `schemes.py` | registers the new schemes; under MPI, rank 0 runs a scheme and spreads each round's oracle calls over every rank |
| `chebtree.py` | **ChebTree**: an adaptive tree of tensor Chebyshev boxes, refined by the decay of their coefficients: per-axis geometric and algebraic decay fits give the error estimate, the degree where the decay is exponential and a split where it is not; a few check points per box, seeded by its locational code, catch aliasing; trailing coefficients are chopped after acceptance. Also `build_rotated`, ChebTree in active-subspace coordinates |
| `chebtt.py` | **ChebTT**: a Chebyshev interpolant in tensor-train form, built by TT-cross (maxvol fibres); cost linear in d; accepted on a Sobol set, free checks at earlier sweeps' grid points and a next-nested-grid look-ahead through the worst point; degrees doubled where an axis's upper degrees still matter, else rank raised |
| `chemistry.py` | stiff chemical networks as oracles (the step map in log10 abundances, Radau per state): Robertson, and a nine-species isothermal primordial H/He/H2 network with rates from Grackle; trajectories of visited states |
| `kernel.py` | **MetricKernel**: greedy Matern-5/2 kernel interpolation in a metric learned from the model's gradients (a prototype that loses to ChebTT above 2-D) |
| `sparsegrid.py` | **SparseCheb**: a dimension-adaptive sparse Chebyshev grid (Smolyak, Gerstner-Griebel), stopped by its surpluses and a Sobol-plus-corners check at half the tolerance; a failed check refines every active index once (a one-step look-ahead) |

Problems: `rosenbrock`, `gaussian`, `wave`, `front` (a sharp tanh front on a sphere); the Genz
families `genz_oscillatory`, `genz_product_peak`, `genz_corner_peak`, `genz_gaussian`,
`genz_continuous`, `genz_discontinuous`; `ackley`, `rastrigin`, `griewank`, `styblinski_tang`,
`michalewicz`; the jumps `oblique_jump` and `ball_jump`; and the ODE flow maps `lorenz96`
(forcing 1, t = 0.1, the manuscript study's), `lorenz96_chaotic` (forcing 8, t = 0.5) and
`lotka_volterra` (a competitive network). All of them are defined for any d.

## Running

From the repository root, every method on the same 8 MPI ranks (AFINNI and ART train with work
stealing; the schemes spread their oracle calls):

```bash
SURROGATE_DEVICE=cpu SURROGATE_MPI=on OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PATH=$PWD/.venv/bin:$PATH PYTHONPATH=. .venv/bin/mpiexec -n 8 .venv/bin/python -m src.structure.bench \
  --problems rosenbrock,gaussian --d 2 --methods afinni,art,art_quad,chebtree,sparsecheb --plot </dev/null
```

- Runs, caches and plots are kept per setup, the device and the rank count (`cache/cpu-8rank/`,
  `output/cpu-8rank/`), so that one comparison never mixes them: times depend on both, and ART's
  split decisions differ between CPU and GPU.
- A run is trained once and cached in `cache/<setup>/` (metrics for every seed; predictions, leaf ids and,
  above 2-D, the slice predictions for seed 1). A new scheme is plotted against the cached
  baselines without retraining them. `--force` retrains.
- `--plot` writes to `output/<setup>/`, per problem, after its runs:
  - `<problem>-d<d>-cost.png`: evaluations against the worst error / tolerance, the mean error,
    the % within tolerance, the training time and the inference time per query (with the oracle's
    time per point);
  - in 2-D, `<problem>-d2-tol<t>-maps.png`: each method's pieces over contours of f, and its error map;
  - above 2-D, `<problem>-d<d>-tol<t>-views.png`: the slice plane and the shadow coordinates spanned
    by the two directions in which f changes most (the leading eigenvectors of the gradient
    covariance), each method's error on both, and every method's error distribution;
  - `summary-d<d>.md`: every method's seed means;
  - `overview-d<d>.png`: every cached problem (rows) and tolerance (column groups) against every
    method, coloured by the worst |error| / tolerance (worst seed; blue holds, red exceeds) and
    labelled with the oracle evaluations; the cheapest method that holds the tolerance is outlined,
    and the bottom rows count, per method, the cases held and won;
  - `overview-d<d>-cost.png`: the same grid for the training time, the inference time per query
    (beside the oracle's time per point), the pieces and the mean |error| / tolerance.
  `--overview` redraws both overviews from the cache alone (`--setup` picks another setup's cache).
- `cache/` and `output/` are not committed (`.gitignore`).
- The new schemes are NumPy only. `SURROGATE_DEVICE=gpu` moves AFINNI and ART to the GPU; the
  schemes would need CuPy versions of their per-box arithmetic, which is milliseconds per run, so
  only their oracle calls matter. The baselines' runs here (up to about 10^6 evaluations, d <= 5)
  train no faster on one GPU than on 8 CPU ranks and predict several times slower, since their
  batches are too small to hide kernel launches; so the recipe uses 8 CPU ranks.
- AFINNI at d = 10 (10^6 training points) needs several GiB of host memory per rank: run it with
  fewer ranks, and watch the memory.

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
tolerances (up to 10 us). The full tables are in `output/summary-d2.md` after a run.

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
