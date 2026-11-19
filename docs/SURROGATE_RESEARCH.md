# Surrogates with a small worst-case error: experiments and literature

This document collects the experiments run beside the library to find a surrogate scheme whose
**worst pointwise error** stays below a tolerance with few oracle evaluations and fast inference,
and the literature reviews that informed them. The code of the current experiments is in
[`src/structure/`](../src/structure/README.md).

**The goal.** The surrogate is meant to replace the time steps of astrochemical reaction networks
(hundreds of species) inside MHD simulations of early cosmic evolution and of galaxy and stellar
evolution: tracers produce data, a surrogate is built from it, and the simulation calls the
surrogate instead of the stiff chemistry solver. So the error that matters is the worst one over the
states the simulation visits, then the cost in oracle evaluations, then the inference time per call.

This file is also the experiments' **working memory**: section 0 is the current state, kept up to
date as results arrive, so that the work can be picked up from it alone. Every work any review read
is registered in [LITERATURE.md](LITERATURE.md) (check it before a new review, append to it after).

Contents:

0. [Working memory: the current state](#0-working-memory-the-current-state)
1. [What the experiments taught](#1-what-the-experiments-taught)
2. [Review I: spending fewer evaluations in adaptive trees](#2-review-i-spending-fewer-evaluations-in-adaptive-trees)
3. [Review II: structure-aware surrogates with max-norm control](#3-review-ii-structure-aware-surrogates-with-max-norm-control)
4. [Review III: from the experiments' failures to chemistry tabulation](#4-review-iii-from-the-experiments-failures-to-chemistry-tabulation)
5. [Review IV: kink splits, the limits in high dimension, a design for chemistry](#5-review-iv-kink-splits-the-limits-in-high-dimension-a-design-for-chemistry)
6. [Review V: ChebTT, high-rank structure, surrogates that learn structure, a chemistry oracle](#6-review-v-chebtt-high-rank-structure-surrogates-that-learn-structure-a-chemistry-oracle)
7. [Synthesis and next steps](#7-synthesis-and-next-steps)

---

## 0. Working memory: the current state

Updated 2026-11-18. Numbers: worst |error| / tolerance over the test set (below 1 is within),
evaluations of the oracle; "k/N" counts cases (problem x tolerance) within on every seed.

### 0.1 Where things are

- Code: [`src/structure/`](../src/structure/README.md) (its README says how to run the bench).
  - `schemes/`: `chebtt.py` (ChebTT: `build`, and `build_vector` for several outputs), `chebtt_adaptive.py`
    (ChebTT-A), `chebtree.py` (ChebTree), `sparsegrid.py` (SparseCheb), `kernel.py` (MetricKernel),
    `tabulation.py` (SlowTable).
  - `problems/functions.py`: the scalar problems, the vector-valued ones (`VECTOR_PROBLEMS`) and `TOLERANCES`
    (2-D 0.1/0.01/0.001, 3-D 0.1/0.03/0.01, 5-D 0.1/0.05/0.02, 10-D and above 0.2/0.1/0.05).
  - `problems/ode/`: the ODE systems (`system.py`: the `System` oracle; `chemistry.py`: Robertson,
    primordial H/He/H2, HIRES, POLLU, Akzo Nobel; `reaction_diffusion.py`: Brusselator, Allen-Cahn,
    Fisher-KPP; `lorenz.py`: Lorenz-96 at F = 1; `classic.py`: Van der Pol, the Oregonator, Burgers;
    `astro.py`: KIDApy networks).
  - `bench/functions.py` runs, caches and plots every method on the box problems (the schemes are
    registered in `bench/registry.py`); `bench/functions_fill.py` drives it over the whole grid
    (section 1.15): `python -m src.structure.bench.functions_fill --d 2,3,5,10,20 --limit 90`, with
    `--vector` for the vector-valued problems.
  - `bench/ode.py` is the ODE bench (section 1.12: `python -m src.structure.bench.ode --problems
    robertson --plot`) and `bench/ode_fill.py` its ladder (`--first lorenz96` runs a family first).
    Each problem declares a regime (stiff, non-stiff, chaotic): overviews per regime,
    `output/ode/overview-<regime>-{loose,tight}.png`, and speedups over the integrator its system
    would be solved with (named baselines `scipy-radau-cpu1`, `scipy-rk45-cpu1`).
  - Prefix every command with `SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=.` and use
    `.venv/bin/python` (the fill starts its own 8-rank `mpiexec` jobs).
- Results (not committed): `src/structure/cache/cpu-8rank/<problem>_d<d>/<method>_tol<t>_s<seed>.npz`,
  each with a JSON `metrics` entry (`max_ratio`, `evals`, `train_s`, `predict_us`; `failed` for a run
  over its evaluation budget or the fill's time limit; `component_ratio` for vector problems). Slice
  directions are in `cache/directions/` (they depend on f alone, so every setup shares them).
  Plots in `output/cpu-8rank/`: start from `overview-d<d>.png` and `overview-vector-d<d>.png`
  (pass/fail grids of every problem, tolerance and method with the evaluations; hatched cells are
  over a budget or the time limit) and `overview-d<d>-cost.png` (training and inference time,
  pieces, mean error); `summary-d<d>.md` has the seed means; then per problem `<problem>-d<d>-cost.png`
  and the maps (2-D) or slice and shadow views. The fill logs to `output/cpu-8rank/fill.log`.
  `python -m src.structure.bench.functions --replot --setup cpu-8rank --d <d> --problems ...` redraws them.
  ODE systems: trajectories in `cache/ode/<problem>.npz`, runs in
  `cache/ode/<problem>/<method>_tol<t>.json`, plots and `summary-ode.md` in `output/ode/`.
- Literature: [LITERATURE.md](LITERATURE.md) (every work reviewed, IDs I-* to V-*); reviews I-V are
  sections 2-6 below, Review VI (vector-valued outputs) section 8, Review VII (metrics by dynamical
  class) section 9, Review VIII (piecewise TTs, interfaces) section 10.

### 0.2 How the work proceeds (standing rules)

- **Climb dimensions on fast problems first.** The fast set: `rosenbrock`, `genz_oscillatory`,
  `genz_corner_peak`, `genz_product_peak`, `genz_gaussian`, `lotka_volterra`. Run a scheme on it at
  2, 3, 5, 10, 20 dimensions; only when it holds there does it earn the slow problems (`wave` and
  `lorenz96` above 3-D for the baselines, the rotated `gaussian`, `genz_continuous`, the jumps,
  `ackley`, `rastrigin`, `front`, `lorenz96_chaotic`). Never tune a scheme for 2-D.
- **Fill every case that trains in reasonable time**, not fixed per-method rules (such as "baselines
  at one seed from 10-D up"): the fill driver runs every method, problem, dimension and tolerance
  under a 90 s wall-time limit per run and records what exceeds it (section 1.15). It runs ChebTT,
  AFINNI, ART with affine and with quadratic leaves; ChebTree, SparseCheb and MetricKernel are
  recorded in section 1 and dropped, since ChebTT beats them from 3-D up. AFINNI is left out at
  d >= 10: 45 GB over 8 ranks on the 10-D wave, which a time limit does not catch before the machine
  runs out of memory.
- **Inference time matters as much as training**: a surrogate is trained once and called millions of
  times. Spend training time on compressing a model whenever that cuts its inference time at least
  2x (section 1.14).
- **The surrogate should discover structure** (manifolds, metrics, low rank, interaction order), not
  assume a basis.
- **Vector-valued problems before chemistry.** The chemistry bench is paused, by the user's decision,
  until the vector-valued box ladder (2, 3, 5, 10, 20-D) is done.
- **Same setup for every method in a comparison**: 8 MPI ranks, one thread per rank
  (OMP/NUMBA/OPENBLAS = 1), CPU. **One heavy job set at a time**: six parallel jobs exhausted 30 GB,
  and an 8-process chemistry pool beside the 8-rank bench oversubscribed the 16 cores (load 14-22),
  which distorts the bench's timings. Serial A/B checks in scratch scripts are fine with
  `OPENBLAS_NUM_THREADS` set low (BLAS threads competing with the bench made QR 50x slower).
- Fail fast: a variant that loses on the fast set is removed, with its numbers recorded in section 1.
- Every result reported goes through the bench (cache and plots); serial scratch scripts are only for
  quick A/B checks. The bench redraws a problem's plots, the summary and the overview after every
  problem, with every cached method, so a new scheme's column appears after its first problem.
- A change to the cache layout reaches only bench processes started after it: restart running jobs
  (or let them finish) before relying on the new layout.

### 0.3 Standings (bench, 8 ranks, 3 seeds, as of 2026-11-18)

From the cache: cases within tolerance on every seed / cases run (budget or time-limit failures
count as run); "–" not run. The fill is in progress: ChebTT's cached runs were cleared when
compression (section 1.14) changed its models, and are being re-run (2-D: 8 scalar problems so far);
3-D to 20-D have no ChebTT runs in the cache yet, and most of the baselines' 10-D and 20-D cases have
seed 1 only. In brackets for ChebTT: the last complete bench runs before compression (`summary-d<d>.md` of
those runs; fast set, and `wave` at 5-D), whose evaluations compression does not change.

| Scheme | 2-D | 3-D | 5-D | 10-D | 20-D | Limits |
|---|---|---|---|---|---|---|
| AFINNI | 0/46 | 5/30 | 0/20 | left out (memory) | left out | least squares on random points; up to 11.8M evaluations at 5-D |
| ART, affine | 1/46 | 3/30 | 0/20 | 0/2 | – | as AFINNI; 2.9M evaluations on 10-D Rosenbrock 0.1 |
| ART, quadratic | 1/46 | 3/30 | 2/21 | 0/8 (3 more within on seed 1) | 0/6 (4 within on seed 1) | as AFINNI; at 10-D and 20-D within only at 0.2, with a single quadratic leaf |
| ChebTree | **46/46** (median 487) | **30/30** (median 1.3k) | dropped | – | – | (n + 1)^d points per box: dead above 3-4 D |
| SparseCheb | 43/45 (median 261) | 27/30 | dropped | – | – | pays for kinks globally; misses small regions |
| MetricKernel | 13/24 | – | serial: 2.2-10.9 on two problems | serial: over budget | – | misses errors at the boundary (section 1.10) |
| **ChebTT** | **24/24** (median 433) | re-running | re-running [21/21, median 1.1k] | re-running [18/18, median 1.7k] | re-running [18/18, median 3.5k] | high rank: rotated narrow peaks, chaotic maps; kinks at tight tolerances |

Vector-valued problems, 2-D so far (five of the six; `lorenz96_state` and the baselines on
`wave_pair` still running): ChebTT 15/15 (median 440 evaluations, 0.05-0.86 us per query), ART with
quadratic leaves 2/15, ART with affine leaves 0/12, AFINNI 0/12 (componentwise worst up to 81x).
Inference at 2-D, median (range) us per query: ChebTT 0.15 (0.07-0.76), both ARTs 0.12-0.14
(0.06-1.4), AFINNI 0.31 (0.1-2.4), MetricKernel 1.4 (0.9-3.7), SparseCheb 0.9 (0.2-68).

Serial ChebTT runs outside the bench (one seed): 10-D 24/24 cases of 8 problems (1.6k-101k
evaluations, including `wave` and `lorenz96`), 20-D 20/21 of 7 problems (3k-130k; `lotka_volterra`
at 0.05 over the 300k budget). Over budget at 5-D: the rotated Gaussian, `lorenz96_chaotic`.

### 0.4 Conclusions that drive the next steps

1. Least-squares pieces on random points (AFINNI, ART) never control the worst error; their mean
   error is small. Interpolation at nodes chosen for the max norm (Chebyshev), with checks at the
   points a refinement would sample next, does: worst errors land at 0.1-0.5 of the tolerance.
2. Dimension is the decisive cost. Tensor boxes stop at 3-D, sparse grids at oblique features at 5-D.
   A tensor train sampled only along maxvol fibres (ChebTT) costs linear in d and holds the fast set
   to 20-D: 1.6k-12k evaluations on five of the six problems; Lotka-Volterra 1.5k-8.7k at 10-D and
   46k-620k at 20-D (218 s at 0.05, beyond the fill's time limit). ART with quadratic leaves needs
   2.7M evaluations and 362 s for 10-D Rosenbrock at 0.05 and still misses it (worst 2.05).
3. No box-sampling scheme can certify the worst error in high dimension (Novak-Woźniakowski): the 5-D
   rotated Gaussian fills ~1e-4 of its box, and every random check set misses it. ChebTT finds it only
   because its cross is seeded at the centre, where that peak happens to be. Kernel interpolation on
   random batches fails in the same way at the boundary, where most of a high-dimensional box's
   volume is (MetricKernel, section 1.10).
4. **Inference is a first-class cost.** An accepted tensor train is far larger than it needs to be:
   rounding its ranks and chopping its degrees within a quarter of the tolerance, then evaluating by
   the Chebyshev recurrence, took 10-D ChebTT from 1.8-8.7 us to 0.2-1.1 us per query, on par with
   ART and AFINNI (0.1-0.7 us), for 0.2-2.3 s more training (section 1.14).
5. **Chemistry's structure is a slow manifold.** Measured: the Robertson step map in log
   coordinates has Jacobian singular values ~1.7, 1.0, 1e-10 for every step dt >= 0.1 (rank 2 of 3:
   the fast direction collapses); the primordial network's fast species collapse at every step (two
   singular values ~1e-10, rank 3-6 of 9 at 1e6 yr). For a large network the rank is the slow
   manifold's dimension, not the species count; this is the structure a chemistry surrogate must
   find. So the chemistry stage trains and tests on visited states (held-out trajectories), not on a
   box.
6. **A chained surrogate's error adds up when it keeps its sign.** The chemistry dynamics contract
   (an early error is amplified 1.4-1.7x at most, and less later), yet a smooth box surrogate's
   rollout error grew linearly to 36x its one-step error over 40 Robertson steps. A chained
   surrogate needs about tol / N per step for systematic errors (tol / sqrt N for random ones), or
   errors that change sign along a trajectory (section 1.12).
7. **Sharing one model across outputs pays only when the outputs share structure.** A vector ChebTT
   (outputs as a last, discrete axis) costs one oracle call for every output, but its ranks are the
   sum of the outputs' own: five different ridges in 6-D cost 7.8k evaluations together against 4.7k
   apart (section 1.16). Whether species on one slow manifold share enough is still to be measured.
8. The surrogate should *discover* structure rather than assume a basis. What has worked so far is
   adaptive sampling that exposes structure: a tree (locality), maxvol fibres (low rank and coupling),
   dimension-adaptive sparse grids (interaction order). A metric learned from gradients
   (MetricKernel) did not survive the box ladder; ISAT-style ellipsoids shaped by the Jacobian
   (SlowTable) cover visited chemistry states but refuse most new trajectories of the primordial
   network (section 1.12).
- **Vector outputs share coordinates, not models.** Without noise, one model over all outputs cannot
  be more accurate than separate ones at the same points; in a tensor train it pays the union of the
  outputs' ranks squared. Train each output separately on one cache of f, and rotate groups of outputs
  that are ridges of shared oblique coordinates to the eigenvectors of their pullback metric:
  3.6-17x fewer evaluations on two shared coordinates at 5-10-D (section 1.19). The metric must be
  per output: one global rotation destroys separable outputs.

### 0.5 In progress and next

- Running: the fill driver on the vector-valued problems, now seven with `shared_latent`
  (`fill --vector --d 2,3,5,10,20 --limit 90`), with ChebTT (separate trains, one cache) and ChebTT-M
  beside the baselines; then the scalar fill (`fill --d 2,3,5,10,20 --limit 90`), which re-runs ChebTT
  and adds ChebTT-M everywhere. ChebTT's vector runs and its scalar 2-D runs were cleared: they came
  from the joint train or the slower contraction (section 1.19).
- Next, once the vector ladder is done: compare ChebTT's inference time at 10-D and 20-D with the
  baselines' in `overview-d<d>-cost.png`, with ChebTT compressed; then resume the chemistry bench.
- To do after the fill (no `src` edits while it runs): cache every evaluation in `chebtt.build` itself.
  Its degree pre-pass and check points re-evaluate points already seen; ChebTT-M, whose shared cache
  never does, used 4-25% fewer evaluations than ChebTT with identical worst errors on most 2-3-D
  scalar cases (Rosenbrock 310 against 334, 2-D Gaussian 908 against 1,208). ChebTT-M differs for
  real only where it escalates: the 3-D rotated Gaussian, 11k-80k against 13k-31k over the seeds
  (the pilot misses the narrow peak), and `front` at 0.1, over the time limit.
- Also after the fill: the bench should cache a `MemoryError` (AFINNI's host-memory guard: 5-D
  `rastrigin` at 0.1 needed a 7.2M-point queue) as a failure, as it does a run over its evaluation
  budget; the fill reported it as a crash, and its record was written by hand. AFINNI on 5-D
  `michalewicz` at 0.1 then exited before the time limit with nothing in the log (no traceback, no MPI
  termination message; probably the same memory exhaustion, killed from outside): rerun it alone,
  watching memory, after the fill; until then its cell is not run. And `swirl` needs a
  domain that shrinks with d: on [-2, 2]^20 its angle 2 exp(-|x|^2) is about 0 at every test point,
  so ART holds it with 400 evaluations while ChebTT, seeded at the centre, resolves the vortex there.
- Running after the box fill (chained): the stiff ladder (`ode.fill --limit 240`, 12 solver workers),
  then the primordial network with the same four methods. Then: per-class horizons and metrics from
  Review VII (valid prediction time for chaotic members; orbit and phase error for oscillatory ones;
  horizons in slow-relaxation times), and a chaotic family on the ladder.
- Chemistry, when resumed: `chebtt_metric` (ChebTT-M per species in log abundances) beside
  `chebtt_box` and SlowTable; first print the species' pullback-metric spectrum and groups, which
  say whether they share coordinates. Then the KL projection onto the conserved totals (section 8.4).
- Chemistry, when resumed: run HIRES, POLLU and Akzo Nobel (only SlowTable at Akzo 0.01 so far);
  per-species against vector ChebTT on the species' shared slow manifold; per-step tolerances of
  tol / N for chained runs; SlowTable's ellipsoids sized by curvature instead of the linear change,
  and grown during use, to cut the primordial network's 95% refusals. Scheme to build: ISAT's table
  of visited states with solver fallback, a local model in slow coordinates read off the step map's
  Jacobian, a learned metric for where a record is trusted, and look-ahead checks before a record
  grows (section 5).
- Open, not acted on: the library takes different ART/AFINNI decisions on CPU and GPU for the same
  seed (2-D wave 0.001: 11,469 evaluations on GPU, 23,187 on CPU), which AGENTS.md forbids; work
  stealing rarely triggers in the distributed runs.

---

## 1. What the experiments taught

All numbers are seed means on one problem set: tolerances relative to max |f| on the test set; in
2-D a 300 x 300 grid, above it 50,000 fixed uniform points. "Worst" is the worst |error| / tolerance
on the test set. Baselines: AFINNI, ART with affine leaves, ART with quadratic leaves
(`local_model="quadratic_lsq"`), all from the library.

### 1.1 Where an adaptive tree spends its evaluations

Every evaluated point is inherited down the tree and ends in exactly one leaf, so ART's cost is
about (leaves) x (points per leaf). Rejecting an interior node early saves nothing: its points pass
to its children. The only levers are fewer leaves (better leaf models or better pieces) and fewer
points per leaf (a cheaper acceptance test).

### 1.2 Quadric (curved) splits in ART

ART's bins and soft searches read a node's points only through the score `a^T u`, so feeding them the
lifted coordinates `phi(u) = [u, u_i u_j]` searches quadric surfaces with the same batched,
deterministic sums. Every ancestor quadric was carried in the engine's constraint column as a block
of d + 1 rows, and hit-and-run moved along the line section of the non-convex region (the roots of
one quadratic per ancestor). The prototype ran on the refinement engine unchanged.

- Closed quadrics (ellipsoids from the |H| metric, level sets of the residual quadratic) gave
  pieces in several components. A piece without inherited points was never sampled, and its leaf
  answered there untested (worst error 25x the tolerance on the running example).
- Sampling the whole line section, not only the current component, reached every piece, but
  annular and multi-component leaves still had worst errors far above tolerance.
- With a bound on how much a surface may bend (its normal within 60 degrees of its mean across the
  node), only gently bent planes survived. They changed the number of leaves by +-3%.
- On a sharp tanh front, residual quadrics gave 20% fewer evaluations with better accuracy, the one
  case where curved cuts helped.

Lesson: on smooth functions the curvature belongs in the leaf model, not in the cut; curved cuts pay
only for curved features, and they cost the convexity that makes hit-and-run sampling reliable.

### 1.3 Polynomial leaves accepted on leave-one-out residuals

ART with quadratic leaves was already 13x cheaper than with affine ones on Rosenbrock. Three changes
were then tested separately (ablation over Rosenbrock, a rotated Gaussian and a ridge function):
cubic leaves; acceptance on leave-one-out residuals `e_i / (1 - h_ii)` (the PRESS statistic, free
from the least-squares fit) instead of in-sample ones; and 6 instead of 20 points per coefficient.

| Change (against ART with quadratic leaves) | Evaluations | Accuracy |
|---|---|---|
| bent-plane splits | +-3% | unchanged |
| leave-one-out test alone | +5-10% | slightly better |
| 6 points per coefficient alone | 3-4x fewer | 97-99% within, worst about 2x |
| 6 points per coefficient with the leave-one-out test | 3x fewer | 98-99.8% within |
| cubic leaves alone | 10-40% fewer | better |
| all three | 3-5x fewer | same share within |

At d = 5 (wave, tolerance 0.01) the combination used 36,390 evaluations against 119,425 with 100% of
the test points within tolerance, but its worst error was 9x the baseline's (0.35 against 0.04):
cubic leaves overshoot in the sparsely sampled corners of their cells. Too few points per coefficient
also break the leave-one-out test itself: leverages near 1 inflate the residuals and good leaves are
split (quadratic leaves on the 5-D wave at 6 points per coefficient grew 854 leaves, against 176
for ART with quadratic leaves at its default budget).

Lesson: least-squares pieces on random points have no handle on the error between the points, and
that gap grows with the dimension.

### 1.4 ChebTree: tensor Chebyshev boxes refined by coefficient decay

Each box holds the tensor interpolant at Chebyshev points of the second kind; its coefficients (one
DCT-I per axis) give the error estimate (the two highest-degree slabs), choose degree doubling where
they decay (the nested points are reused) and a split where they do not.

| 2-D, tolerance 0.001 | ART, quadratic leaves: evaluations / worst | ChebTree |
|---|---|---|
| Gaussian | 18,462 / 2.7 | 2,331 / 0.07 |
| wave | 5,896 / 1.7 | 1,399 / 1e-10 |
| Genz product peak | 13,948 / 2.8 | 1,035 / 0.06 |
| Griewank | 245,504 / 3.4 | 4,743 / 0.002 |
| Rastrigin | 1,046,840 / 4.4 | 41,391 / 0.003 |
| sharp front | 186,293 / 17 | 25,775 / 0.07 |
| Genz continuous (kinks) | 25,578 / 24 | 9,315 / 1.4 |

Polynomials (Rosenbrock, Styblinski-Tang) are exact with 81 evaluations. AFINNI and both ART variants
exceed the tolerance at their worst points on every problem. Training takes milliseconds and
inference 0.2-0.3 us per query.

Failures: aliasing at coarse resolution (Ackley and Michalewicz at tolerance 0.1: 9 x 9 boxes that
cannot see the oscillation, worst 1.4-1.8x); optimistic tails at kinks, where coefficients decay
algebraically (worst 1.4-3.7x); overshoot on smooth functions; and the (n + 1)^d cost of tensor boxes,
which stops it at d = 3-4 (the 3-D front needed 1.3 million evaluations, the 4-D Gaussian more than 3
million).

### 1.5 SparseCheb: a dimension-adaptive sparse Chebyshev grid

The Smolyak combination of tensor interpolants on nested Clenshaw-Curtis grids, grown where the
hierarchical surpluses are largest (Gerstner-Griebel), stopped when the active surpluses fall below
half the tolerance and a check on 20 d random points passes.

- Cheapest on smooth and separable functions: Rastrigin exact with 301 evaluations (3,500x fewer
  than ART with quadratic leaves), Michalewicz with 237, Griewank with 361, the 5-D Rosenbrock exact
  with 105, the 10-D one with 325.
- The surplus estimate stops early when some interaction is never opened: the 5-D Lorenz-96 flow map
  stopped at 1-12% within tolerance until the check points were added (then 1,829 evaluations, worst
  0.42).
- Random check points miss small regions: the Genz corner peak (worst 1.37), the Gaussian and
  Griewank at some seeds (worst 1.0-1.7).
- One global expansion pays for every kink or jump: 1e5-1e6 evaluations on Ackley, Genz continuous
  and the jumps.
- Oblique ridges need every interaction: the 5-D wave exceeded 2 million evaluations.
- Its first implementation summed one small tensor per index, in Python, during training and at
  every prediction (up to 10 us per query, 90 s per run on jumps). A second version holds one flat
  Chebyshev expansion and checks on a Sobol set plus the corners; it removed the misses but costs
  1.5-3x the evaluations, and its flat evaluation is slower than expected for many terms.

### 1.6 Active-subspace rotation

Rotating to the eigenvectors of the gradient covariance `E[grad f grad f^T]` made the ridge function
one-dimensional (the 5-D wave in 180,000 tensor evaluations, worst 0.001) but did nothing for radial
features (the front, where the covariance has no gap), and the rotated box makes f be evaluated
outside the domain. Starting every axis at degree 2 to save points made aliasing worse (worst 13x).

### 1.7 ChebTree revised: fitted decay laws, seeded checks, fibre verification

The acceptance was rebuilt: per-axis geometric and algebraic fits of the coefficient slab sums
(exponential only if the fit falls >= 16x across the upper half of the degrees; otherwise treated as a
kink), the degree predicted from the fitted rate, four check points per box seeded by its locational
code (deterministic), chopping after acceptance, refusal at the depth cap. 2-D: within on all 46 cases
(was 28 of 33), 3-D 30/30.

- A blind spot no coefficient fit can see: a kink at about 2 pi / n of a box's width from a face makes
  the slab sums follow half a period of the kink's cos(j theta) modulation, which reads as fast decay
  (estimate 3.8e-4 against a true error 3.4e-2; 3-D `genz_continuous` 0.01, worst 2.49 at 1.1M
  evaluations). Fix: before acceptance, one more round evaluates, along each axis through the box's
  centre, the points the next nested grid would add (a full period of the modulation). Worst 0.63, for
  3-13% more evaluations in 2-D.
- Rejected: splitting at a kink located from the grid values. It fixed the 3-D case (0.43) but cost
  1.5-14x elsewhere. A located cut that misses by 1e-3-1e-4 of the width leaves the kink just inside a
  child's face, where it can never pass and is bisected toward the face; the trigger ("decay not
  exponential") also fires on smooth features not yet resolved; and refusals at a curved jump grew from
  1.3% to 4%. Locate to rounding level (Chebfun bisects to the last bit) or cut at the midpoint.

### 1.8 ChebTT: a Chebyshev tensor train by cross approximation

f is sampled only along fibres of a Chebyshev grid chosen by maxvol (TT-cross), so sampling and
inference cost grow linearly in d. Products of univariate functions have rank 1 and `cos(q.x)` rank 2.
What made it reliable:

- Degrees decided by the model: an axis doubles its degree when its top degrees still change the
  prediction at the check points (the top quarter, section 1.13); otherwise the rank grows. Raising
  only the rank at a fixed degree could never resolve a narrow peak (the rank climbed to 74 at degree
  8).
- Free checks: grid points of earlier sweeps and lower degrees, which the final cores do not
  interpolate, concentrated where the cross explored. They caught a narrow 5-D peak (0.022 > 0.01)
  that uniform check points missed.
- A look-ahead at the next nested grid's points along every axis through the worst check point and
  through the largest |f| seen; the right index sets seeded at the largest |f| and the centre.
- Results: 5-D bench 21/21 (median 1.1k evaluations; baselines 0-2 of 21 with 10-1000x more); bench
  fast set 18/18 at 10-D and at 20-D (section 1.15); serial 10-D 24/24, 20-D 20/21. Over budget: the
  5-D rotated anisotropic Gaussian (high rank in axis coordinates), `lorenz96_chaotic`,
  `genz_continuous` at 0.02 (kinks).
- With the degree tail read from the upper half of the coefficients, many smooth cases ended far
  below the tolerance (1e-6 of it): the tail rule asked for about the square of the accuracy
  (section 1.13).

### 1.9 Rejected: global rotation, and ChebTT in tree cells

- Global active-subspace rotation (gradient covariance from 10 d random points) made SparseCheb exact on
  the 5-D wave with 629 evaluations (plain: over 300k) and Genz oscillatory with 581, but: it makes f be
  evaluated outside the box (Lotka-Volterra blows up there, NaN), its gradients are noise where f is
  concentrated (the rotated Gaussian), and ChebTT got more expensive on non-ridge functions (corner
  peak 722 -> 1.2k). A rotation must be local and must stay inside the domain.
- A tree of ChebTT cells (split along the axis whose fibres need the highest degree): 2.5-4x plain
  ChebTT's evaluations on the kinks and no help for the Gaussian or the chaotic map.

### 1.10 MetricKernel: a learned-metric kernel model (`kernel.py`)

Matern-5/2 interpolation in a metric M learned from the average gradient outer product of the current
model (the mechanism of recursive feature machines), the scale chosen by the worst leave-one-out error
(Rippa's formula), grown by adding the worst-residual points of fresh Sobol batches, stopped when two
fresh batches stay below half the tolerance. 2-D: the learned metric halves the evaluations against
an isotropic kernel (Gaussian 609 vs 1,057 at 0.01) and fixes the wave (worst 0.21 vs 1.7 at 0.1), but
random batches miss corner peaks and fine scales (worst 1.8-2.4 on wave and corner peak at 0.01); 5-D
rotated Gaussian worst 2.2-4.5 (the needle, as for every box sampler). On the fast set (serial, one
seed) it loses to ChebTT everywhere: 5-D worst 0.3-0.9 where it holds and 2.2-10.9 on Genz oscillatory
and corner peak; 10-D over its 20k budget on Rosenbrock and 1.8-2.7 on the corner peak. Kernel
interpolation is weakest near the boundary, where most of a high-dimensional box's volume is, and a
few hundred random check points do not find it. Bench at 2-D: 13/24 cases within (median 225
evaluations, 0.9-3.7 us per query, against 0.07-0.76 for ChebTT). Not promoted: it is not run above
2-D, and the fill leaves it out. Its natural setting is a set of visited states, where the candidates
come from the simulation and boundaries matter less.

### 1.11 The chemistry premise, measured

Robertson's stiff network (k1 = 0.04, k2 = 3e7, k3 = 1e4), states from trajectories at random times
after Dirichlet-random initial conditions, central differences of the step map in log10 coordinates:
singular values ~(1.3-1.7, 1.0, 1e-2-1e-4) at dt = 1e-3 and ~(1.2-1.7, 0.5-1.0, 1e-10) for every
dt from 0.1 to 1e3. The fast direction collapses once dt exceeds the fast time scale.

The primordial network (nine species, isothermal; `src/structure/problems/ode/chemistry.py`), states relaxed for
1e5 yr from random ionisation and H2 fractions, central differences in log10 number densities:

| n_H (cm^-3), T (K) | dt = 1e2 yr | dt = 1e4 yr | dt = 1e6 yr |
|---|---|---|---|
| 1e1, 300 | 7 of ~1, two at 2e-7, 2e-10 | 7 of 0.9-1.4, two at 1e-11 | 1.3-3.1, 0.8, 0.1, 0.08, 2e-3, two at 1e-11 |
| 1e3, 1000 | 7 of ~1, two at 5e-11, 2e-11 | 7 of 0.65-1.4, two below 4e-11 | 2.9, 1.7, 0.8, 0.05, 3e-3, 3e-5, three below 6e-11 |
| 1e4, 3000 | 7 of ~1, two below 2e-11 | 6 of 0.9-1.9, 0.19, two below 2e-10 | 46, 1.7, 1.1, 0.6, 0.04, 0.03, three below 2e-8 |

The two fast species (H-, H2+) collapse onto quasi-equilibrium at every step (singular values ~1e-10),
and longer steps relax more modes (numerical rank 3-6 of 9 at 1e6 yr). At short steps the other
singular values are ~1: the map is close to the identity on the slow species. So a surrogate should
learn the increment `log y(t + dt) - log y(t)`, which is small and carries the chemistry, and its
input dimension is the number of slow modes, which falls as dt grows.

### 1.12 The chemistry bench: one-step and rollout errors

`src/structure/bench/ode.py` trains each surrogate of a network's step map (in log10 abundances, the
increment `Phi_dt(z) - z`) on training trajectories and judges it on held-out ones in two ways: the
**one-step error** (a surrogate step from the true state against the solver's step: how well the
visited states are covered) and the **rollout error** (20 test trajectories chained through the
surrogate from their initial states, the solver stepping what it refuses: what a one-zone run
accumulates). A fourth panel shows how much the solver's own dynamics grow an early error (the
largest singular value of the product of step Jacobians), to tell the surrogate's drift from the
dynamics'. Schemes: SlowTable (`tabulation.py`, ISAT's design: linear records in ellipsoids of
accuracy shaped by the Jacobian, grown and shrunk on the training stream, refusing outside them) and
ChebTT per species on the training states' bounding box.

Robertson (3 inputs, dt = 1, 200 training and 100 test trajectories of 40 steps):

| tol (dex) | scheme | training solves | one-step worst | refused | rollout worst after 40 steps |
|---|---|---|---|---|---|
| 0.01 | SlowTable | 841 | 0.040 (4x, only in the initial transient) | 1.5% | 0.018 |
| 0.01 | ChebTT box | 955 | 0.0019 | 0 | 0.020 |
| 0.001 | SlowTable | 1,379 | 0.00063 | 2.5% | 0.00074 |
| 0.001 | ChebTT box | 1,130 | 0.00005 | 0 | 0.0017 |

- The dynamics contract: the amplification of an early error stays at 1.4-1.7 and falls; chemistry is
  not chaotic. Yet the box ChebTT's rollout error grows linearly to 36x its one-step error after 40
  steps: a smooth surrogate's error has the same sign step after step and adds up. SlowTable's errors
  change sign from record to record and partly cancel. So the tolerance a chained surrogate needs per
  step is about tol / N for systematic errors (tol / sqrt N for random ones), or the surrogate must be
  built so that its errors do not share a sign along a trajectory.
- SlowTable's only misses are in the first step, the initial transient (the fast species far off the
  slow manifold), where it already refuses 50-90% of the states.
- Primordial (10 inputs, 60 training trajectories): SlowTable answered within 0.2 of the tolerance but
  refused 95% of the test states at 0.01 dex. Its ellipsoids, sized by where the *linear change* stays
  within the tolerance, are ~0.005 dex across, while its error is second order; and test trajectories
  from other densities, temperatures and fractions pass through regions 60 training trajectories never
  visited. Ellipsoids sized by curvature, and growth during use (in situ), are the next steps.
- Three standard stiff problems were added from the Test Set for IVP Solvers (Bari), transcribed from
  its Fortran sources and matching its reference solutions to 1e-12: HIRES (8 species), POLLU (20) and
  Akzo Nobel (its algebraic component substituted out, an exact reduction to 5 species). The bench
  starts their trajectories after the initial transient and leaves species below 1e-12 out of the
  error (an error in dex means nothing there). First run, Akzo Nobel at 0.01 dex with SlowTable:
  2,988 solves, one-step worst 0.045 (4.5x, the first steps), 2.7% refused, rollout worst 0.054.
- The bench also trains a vector ChebTT (one train for every species, section 1.16) beside the
  per-species one, to measure which fits a network whose species share one slow manifold. The
  chemistry bench is paused there until the vector-valued box ladder is done.

### 1.13 ChebTT's degree tail from the top quarter of the coefficients

For geometric decay a_j ~ rho^-j the interpolation error is about the last coefficient a_n, while the
upper half of the degrees sums to about a_{n/2}, roughly sqrt(a_n a_0). Asking the upper half to meet
an axis's share of the tolerance therefore asked for about the square of the needed accuracy, and
degrees doubled too far. Both degree tests (`chebyshev_tail` and the truncated-model test) read the
top quarter instead. Serial runs on the fast set at 5-D and 10-D:

| 5-D, evaluations | upper half | top quarter |
|---|---|---|
| Genz product peak, 0.02 | 1,632 | 1,160 |
| Genz Gaussian | 1,372 | 996 |

Up to 30-40% fewer evaluations at 5-D and 10-D, every case still within tolerance, worst errors closer
to it. Rejected: seeding new right indices at the worst check points (global pivots, as in TCI)
instead of at random. Neutral or worse: 10-D Lotka-Volterra at 0.1 went from 4.5k to 9.0k evaluations.

### 1.14 ChebTT inference: compression after acceptance, evaluation by recurrence

Profiled at 10-D, the prediction spent its time in the Chebyshev basis (arccos and cos per query and
axis) and in a planned einsum, not in the 300-2,700 multiply-adds the cores need. And the accepted
cores are far larger than the tolerance requires, since the cross grows ranks and doubles degrees in
coarse steps. Two changes, kept:

- **Compression** (`compress`): TT-SVD rank rounding with the largest relative eps found by bisection,
  then per-axis chopping of trailing degrees, each step kept while the model changes by at most
  0.25 atol at its check points and 4,000 uniform points. No oracle calls: the comparison is between
  models.
- **Prediction by recurrence**: the Chebyshev values from the three-term recurrence and one matrix
  product per core, in chunks. An uncompressed model's predictions match the einsum evaluation to
  1e-15.

| 10-D, serial | us per query, before | after | cores after compression |
|---|---|---|---|
| Lotka-Volterra | 8.7 | 0.35-1.0 | 2,700 core entries -> 293 (ranks ~2-5) |
| wave | 4.7 | 0.65-1.1 | |
| Genz product peak | 1.8 | 0.2 | rank 1 |

Worst errors 0.13-0.35 of the tolerance; compression adds 0.2-2.3 s of training. ART and AFINNI
predict at 0.1-0.7 us, so compressed ChebTT is on par with them; before, at 10-D and 20-D in the
bench it took 1.6-13 us, and 33-90 us on 20-D Lotka-Volterra. Rejected: a fused Numba loop per
query, no faster than the NumPy matrix products (both sit at the scalar flop floor). The principle (a
standing rule): spend training time on compression whenever it cuts inference at least 2x, since a
surrogate is trained once and called millions of times.

### 1.15 The box ladder to 20-D, and the fill driver

Bench runs of the fast set at 10-D and 20-D (8 ranks; ChebTT at 3 seeds, the baselines mostly at
seed 1; before compression):

| | ChebTT | ART, quadratic leaves | ART, affine leaves |
|---|---|---|---|
| 10-D | 18/18, 1.5k-8.7k evaluations, worst seed <= 0.41 | 3 of 8 within on seed 1, all at 0.2 with one leaf; Rosenbrock 0.05: 2.7M evaluations, 362 s, worst 2.05 | Rosenbrock 0.1: 2.9M evaluations, 75-95 s per run, worst 3.3-4.0 |
| 20-D | 18/18, 3.2k-12k evaluations (Lotka-Volterra 46k-620k), worst seed <= 0.5 | 4 of 6 within on seed 1, all at 0.2 with one leaf | – |

AFINNI is left out at d >= 10 for its memory (45 GB over 8 ranks on the 10-D wave, one rank 15 GB).

Fixed rules for which method runs where ("baselines at one seed and the loosest tolerance from 10-D
up") left holes and kept slow cases in by hand. They are replaced by `src/structure/bench/functions_fill.py`: every
method, problem, dimension and tolerance, from the loosest tolerance to the tightest (the cost only
grows as it tightens). Seed 1 runs first as its own 8-rank job under a 90 s wall-time limit (start-up
included); the other seeds run only if it finished. A run over the limit is killed with its whole
process group (so no rank is left waiting in a collective), cached as failed (hatched in the
overview, like a run over its evaluation budget), and the method's tighter tolerances for that problem
are skipped. Cached cases are never rerun. It runs ChebTT, AFINNI and both ARTs (ChebTree,
SparseCheb and MetricKernel are recorded and dropped: ChebTT beats them from 3-D up), the scalar
problems with the fast set first, or the vector-valued ones with `--vector`.

### 1.16 Vector-valued problems

The chemistry step map has one output per species, so the box ladder gained problems R^d -> R^m,
defined for any d (`VECTOR_PROBLEMS`): `rosenbrock_gradient` (m = d, a polynomial field),
`genz_triple` (Genz oscillatory, product peak and corner peak side by side, m = 3),
`lotka_volterra_state` and `lorenz96_state` (the full flow maps, m = d), `swirl` (coordinate pairs
rotated by 2 exp(-|x|^2), m = d) and `wave_pair` (cos and sin of one oblique ridge, m = 2). Each
component is scaled to max |f_c| = 1 on the test set, so one tolerance is the same relative accuracy
for every component, and the error is the componentwise relative max norm (each component's worst is
recorded too). The 2-D maps add a column with the field (|f| with streamlines); the views use the
Jacobian's Gram matrix; the overview is `overview-vector-d<d>.png`.

ChebTT first treated the outputs as a discrete last axis (one train, one oracle call serving every
output). Its cost is in the ranks, which add up across outputs with different structure: five outputs
cos(a_c . x) in 6-D needed 7.8k evaluations as one vector train against 4.7k for five separate trains
(rank 10, two per ridge). It was replaced by separate trains on one shared cache (section 1.19).

First results, 2-D:

| Case | ChebTT | ART, quadratic | ART, affine | AFINNI |
|---|---|---|---|---|
| `swirl`, 3 tolerances | 644-1,626 / 0.21-0.26 | 1.3k-33k / 1.8-3.0 | 1.4k-135k / 1.6-2.7 | 1.0k-46k / 4.0-81 |
| `rosenbrock_gradient`, 3 tolerances | 342 / 1e-12-1e-14 (exact) | 243-975 / 0.53-1.69 | 247-19.7k / 1.6-2.9 | 614-15.7k / 1.3-49 |
| `genz_triple`, 3 tolerances | 474-1,042 / 0.24-0.28 | 975-21.7k / 2.8-5.1 | 1.3k-120k / 2.7-3.9 | 1.4k-8.8k / 3.9-55 |
| `lotka_volterra_state`, 3 tolerances | 310-561 / 0.18-0.26 | 121-3.9k / 1.1-2.5 | 247-21k / 1.4-3.0 | 604 / 1.1 (0.1) |
| `wave_pair`, 3 tolerances | 376-440 / 0.13-0.23 | 975-4.4k / 1.2-3.2 | running | running |

(evaluations / worst over the seeds; ChebTT predicts at 0.05-0.86 us per query.) The vector ladder
(2, 3, 5, 10, 20-D) runs through the fill driver.

### 1.17 Engineering edge cases met

- Every rank computed and wrote a problem's cached slice directions at once; a rank read a file another
  was writing and the run crashed. Fix: each rank writes its own file and renames it into place. The
  directions live in `cache/directions/`, one file per problem and dimension, shared by every setup;
  a test guards that nothing else is written there.
- A bench run with a subset of methods redrew the plots with that subset only. Fix: plots draw every
  cached method; `--replot` redraws from the cache.
- Stopping a collective MPI run leaves the other ranks waiting: stop by the PIDs of the launcher and
  its ranks; never `pkill -f`. Collecting PIDs with `ps | grep <pattern>` inside a shell whose own
  command line contains the pattern kills that shell too: list the PIDs first, then kill them by number.
  The fill driver starts each job in its own process group and kills the group.
- A change to the cache layout reaches only bench processes started after it: a 10-D job already
  running wrote three stray directories under the old layout.
- An 8-process chemistry pool beside the 8-rank bench oversubscribed the 16 cores (load 14-22) and
  would have distorted the bench's timings: one heavy job set at a time.
- Fixed per-method rules for slow cases (baselines at one seed from 10-D up) left the grid uneven; a
  wall-time limit per run (section 1.15) keeps long cases out without them.
- The GPU never beat 8 CPU ranks on these workloads (launch- and transfer-bound, 200k kernel launches)
  and predicted 2-50x slower per query.

- **Editing `src` while a fill runs breaks it**: every job imports the code afresh, so a half-written
  `chebtt.py` crashed the remaining jobs (a syntax error on import), and the driver then stopped on the
  missing cache entry. The fill now reports a crashed job and records nothing for it; the rule is
  still not to edit `src` during a fill.

### 1.18 Engine findings from the same work

- A stable counting sort plus per-block merge sort replaced two full-batch argsorts in
  `sort_by_block_then_value`: single-thread AFINNI training 36-39% faster, bit-identical results.
- In the distributed experiment, work stealing on and off gave identical trees, evaluation counts
  and times from 2 to 64 ranks at tolerance 0.05: stealing rarely or never triggers for that workload.

### 1.19 Vector outputs: separate trains on one cache, rotated by the outputs' shared metric

Review VI (section 8) set the frame: f gives every output at every point without noise, so sharing a
model cannot make an interpolant more accurate at a given point set (the cancellation of inter-task
transfer, VI-2); sharing can only save points and inference work. In a tensor train, one train over
all outputs pays the union of their factor spaces: each bond rank lies between the largest output's
and the sum of theirs, and the sampling pays it squared.

**Separate trains, one cache** (`chebtt.build_vector`, bench method `chebtt`). One scalar ChebTT per
output; every oracle call stores all m outputs, and the trains share the degree pre-pass, the Sobol
checks and the first fibres (same seed), so they cost 1.3-1.9x fewer evaluations than independent
trains with identical models, and beat the joint train on 10 of 13 cases by 1.1-34x (one seed,
serial; 10-D swirl 31k against 1.07M, 10-D Lotka-Volterra 33k against 116k). The joint train won only
on coherent outputs (`wave_pair`, the cos and sin of one ridge: 763 against 1,255 at 5-D).

**What the outputs share is coordinates** (`build_vector(rotate=True)`, bench method `chebtt_metric`,
ChebTT-M). A pilot of forward differences at 4 + 2d Sobol points (rounded up to a power of two; all
outputs at once) gives every output's pullback metric G_c = mean grad f_c grad f_c^T. An output is
rotated when its eigenvectors hold 99% of trace(G_c) in at most half as many directions as the axes do
(a separable output never is: a rotation destroys its rank 1). Outputs whose leading subspaces merge
without growing form a group trained in the eigenvectors of their summed, trace-normalised metric, on
the box bounding the domain in those coordinates. That box extends beyond the domain, so a group keeps
the axes where f at 2d random points of its box is not finite or exceeds 10x its largest pilot value.
One seed, serial, evaluations / worst (tolerance units):

| Case | separate, one cache | ChebTT-M | joint train |
|---|---|---|---|
| `shared_latent` without curvature, 5-D 0.05 | 45k / 0.41 (4.7k coefficients) | **12.6k** / 0.40 (383) | 109k / 0.50 |
| `shared_latent` (curved), 5-D 0.05 | 430k / 0.64 (20k) | **78k** / 0.54 (917) | 449k / 0.46 |
| `shared_latent` without curvature, 10-D 0.1 | 922k / 0.32 (36k), 4.7 us/query | **54k** / 0.26 (519), 1.6 us | – |
| `shared_latent` (curved), 10-D 0.1 | over 2M | running in the fill | – |
| `genz_triple`, 5-D 0.05 | 1.4k / 0.30 (177) | 2.1k / 0.30 (54; all three rank 1) | 2.6k |
| `wave_pair`, 5-D 0.05 | 1.3k / 0.24 (182) | 1.5k / 0.18 (35, rank 1) | 763 |
| `swirl`, `rosenbrock_gradient` | as separate | axes kept (metric isotropic or axis-aligned) | – |
| `lotka_volterra_state` | as separate | rotation refused: the ODE leaves its orthant | – |

- **Escalation instead of an up-front decision.** Deciding from the pilot first cost the pilot on
  every case and rotated outputs that are already cheap on the axes: on the bench, 2-D and 3-D
  `genz_triple` and `shared_latent` cost 30-60% more than ChebTT (the rotated box is larger and shares
  no points with the outputs left on the axes; at 2-3-D the axis ranks are 1-4, so there is nothing to
  gain). Now every output is first trained on the axes with its rank capped at 6; one that meets the
  tolerance keeps that model, identical to ChebTT's at the same cost, and only outputs reaching the cap
  get the pilot, then a rotated train or a full-rank axis train from the cache. Cap 4 / 6 / 8 on
  `shared_latent` (curved): 3-D 15.1k / 13.6k / 13.6k (the axes 13.6k), 5-D 40k / 49k / 74k (the
  axes 358k), 10-D 227k / 256k / 241k (the axes over 2M). Every other bench case tested costs exactly
  ChebTT's evaluations at cap 6.
- The gain grows with d: 3.6x at 5-D, 17x at 10-D on the shared oblique coordinates, and 12-70x fewer
  coefficients. Curvature halves it but does not remove it (5.5x at 5-D).
- One global rotation for all outputs is wrong: on `genz_triple` the oscillatory output dominates the
  summed metric (0.95 of it on one direction) and the rotation made the product peak rank 10 (20k
  evaluations against 1.4k); on isotropic `swirl` it cost 6x. Hence the per-output decision and groups.
- Where nothing is gained, ChebTT-M costs the pilot, (d + 1) times 8-64 points; at 2-D it costs 25-40%
  more than ChebTT on `shared_latent` (two shared coordinates in 2-D are the whole space).
- Limits: the pilot sees nothing of a feature its points miss (the narrow rotated Gaussian), and the
  rotation is global: a curved manifold needs local coordinates, which is the SlowTable direction for
  chemistry (records in the local Jacobian metric). In log abundances the step map is defined off the
  visited states, so the refusal test should pass there.

**Faster contraction (all ChebTT models).** Per core, the Chebyshev values times the core first,
(Q, n + 1) x (n + 1, r_k r_k+1) in one matrix product, then the running row vector times that (a
plain scaling when the left rank is 1), instead of the row vector into the core first, which leaves a
strided (Q, n + 1, r) temporary. A vector model's group shares the Chebyshev values of its
coordinates, and `compress` computes the probe's values once instead of in each of its ~70 trials.
Interleaved A/B, 2 rounds, scalar ChebTT: identical evaluations and worst errors on 6 cases (5-D to
20-D); training 1.7-5.7x faster, inference 1.2-2.4x. Vector models, 10-D: Lotka-Volterra 2.1 -> 1.3
us per query, `shared_latent` 1.1 -> 0.36, `rosenbrock_gradient` 0.75 -> 0.28, 5-D `swirl` 1.45 ->
0.46. Rejected: stacking a group's outputs zero-padded into one contraction (faster only at batches of
about 1k; up to 4.6x the coefficients), and merging the outputs into one rounded train (faster only on
Lotka-Volterra, 0.70 against 1.18 us, and it moves predictions by up to 0.25 of the tolerance).


### 1.20 The stiff-system ladder: errors by dynamical class, and rollouts that hold

The vector ladder's next rung before chemistry (`ode/fill.py`, `ode/bench.py`): surrogates of the step
map of well-known stiff systems at growing sizes, trained and judged on visited states. Families
(`chemistry.LADDER`): the Brusselator with diffusion (Hairer-Wanner; no-flux boundaries, B = 1.9 near
its Hopf point, log coordinates; 2n inputs), Allen-Cahn (u' = u_xx + (u - u^3)/0.05, linear coordinates)
and Fisher-KPP (u' = u_xx + 10 u (1 - u), log coordinates), n = 2..80 grid points; the fixed-size test
problems (Robertson, Akzo Nobel, HIRES, POLLU) at their sizes. 40 steps per trajectory, 100 training
and 40 test trajectories. Methods: `chebtt_roll`, `chebtt_box`, `chebtt_metric`, `slowtable`.

**What every run measures** (Review VII, section 9):

- Speed: the solver's wall time per state and step (SciPy Radau, 0.7-2.6 ms at 2-20 inputs) and its
  operation counts (27-68 right-hand sides, 4-9 LU factorisations per step), against the surrogate's
  inference time; the ratio is `speedup`. SciPy's per-call overhead flatters a surrogate: a compiled
  solver doing the same counts on a small system takes microseconds, so the counts are the fair basis.
- The dynamics' class, from the problem's own trajectories (`dynamics`): step-map stiffness
  s = max -ln min|eig J_k| (Jay-Calvo: mildly stiff >= 10, stiff >= 20); the growth |J_k ... J_1|;
  contracting / neutral / chaotic; and the accumulation factor A_n = sum_j |J_{n-1} ... J_j| of the
  classical global-error bound (rollout error <= one-step error x A_n, to first order).
- Errors: the one-step worst; the part of it that survives the next exact step, |Phi(S(z)) - Phi(Phi(z))|;
  the rollout worst over 20 test trajectories; rho = rollout / (one-step x A_n); and the same surrogate
  with every step projected onto the system's linear invariants (found as the null space of the
  right-hand side at the training states: Robertson 1, HIRES 1, POLLU 6, Akzo Nobel 0).

**The stiff test problems are neutral, not contracting.** Robertson's step map keeps a singular value
of about 1 (the conserved mass) and HIRES and Akzo Nobel drift slowly along their slow manifolds: the
growth stays at 1.4-2.7 over 40 steps, so A_40 = 30 (Fisher-KPP, the one contracting problem) to 280
(HIRES). A rollout may therefore be up to A_n times the one-step error for any surrogate, and "the
rollout holds the tolerance" needs one-step errors well below tol / A_n unless errors cancel.

**Rollout-verified training** (`chebtt_roll`): chain the surrogate along the training trajectories (known,
so no solve), and retrain at the one-step tolerance times the shortfall until the rollouts hold;
ChebTT's spectral convergence makes a tighter tolerance cheap, and the retraining reuses every cached
solve. Robertson (one seed):

| tol | method | solves | one-step / tol | survived / tol | rollout / tol | rho |
|---|---|---|---|---|---|---|
| 0.01 | ChebTT box | 955 | 0.25 | 0.17 | 5.05 | 0.67 |
| 0.01 | ChebTT, rollout-verified | 1,159 | 0.04 | 0.02 | **0.82** | 0.98 |
| 0.001 | ChebTT box | 1,130 | 0.18 | 0.18 | 4.34 | 0.97 |
| 0.001 | ChebTT, rollout-verified | 2,176 | 0.03 | 0.01 | **0.26** | 0.58 |
| 0.01 | SlowTable | 841 | 4.03 | 0.10 | 1.78 | 0.28 |

- ChebTT's rho is 0.6-1: its errors keep their sign and add up (the 36x of section 1.12), which a
  rollout-aware tolerance fixes at 1.2-1.9x the solves.
- SlowTable's one-step error is 4x the tolerance, but only 0.1 of it survives the next step: its errors
  lie in fast directions the dynamics damp. The one-step error alone misjudges such a surrogate.
- Projection onto the invariants barely helps Robertson (5.05 -> 4.93): its drift is along the slow
  manifold, not the conserved total.
- Scratch, 4-D Brusselator at 0.01: training at tol/2 and tol/4 (7.0k and 6.9k solves against 3.8k)
  takes the rollout from 1.14 to 0.74 and 0.28 of the tolerance; a box in the states' principal axes
  cost 83k solves (22x) and its one-step error was 2x the tolerance: rejected. A box in the step map's
  singular vectors did not finish within 1,000 s: rejected. Longer steps (a surrogate of Phi_2dt chained
  20 times over the same span) cost 11k solves (2.9x) and rolled out to 2.86 of the tolerance (against
  1.14): the two-step map is harder to fit, not smoother, here: rejected.


### 1.21 Adaptive ChebTT for kinks, jumps and interfaces

ChebTT is one global model and does not use the refinement engine; its adaptivity is in degrees,
ranks and sample placement. It failed every non-smooth benchmark: kinks (`genz_continuous`), axis
jumps (`genz_discontinuous`), the oblique and spherical jumps, the sharp front. Review VIII (section 10)
and a consult agreed on the split of the problem: a sum of products of univariate functions can only
be non-smooth along axis-aligned planes, so axis-aligned edges belong inside the train (piecewise
axes) and oblique or curved interfaces need a term aligned with the interface; a partition refined
towards a codimension-1 interface needs about (1/h)^(d-1) leaves that are not boxes, so the engine
does not help there (it could host ChebTT leaves for localised features: a new plan, with ragged leaf
transport between ranks and empty-node handling; parked).

**Piecewise axes (in `chebtt.py`).** An axis may carry breakpoints; its pieces share one degree,
nodes are the pieces' Chebyshev points (inset by 1e-10 at an interior breakpoint), keys carry a
piece offset, and truncation, look-ahead, compression and prediction work per piece. Edges:
`axis_edges` scans a fibre (65 points), locates an outlying second difference by zooming (17 points
per 5x, to 1e-12 of the width; bisecting by midpoint deviations converged 1.4e-3 off when the kink sat
near a quarter point), confirms it by a two-scale test (a jump's or kink's one-sided difference keeps
its size from 1e-6 to 1e-8 of the width) and by a fibre displaced 1% in the other coordinates. It is
searched on fibres through the worst check point and the largest |f|, only where the axis's
coefficients decay algebraically: in the pre-pass on an axis unconverged at degree 32 (else a kink
takes it to degree 256 at once), on axes that must refine, and before the rank grows (a mild kink's
top-quarter tail can look converged, and the rank then grows in vain). One seed, evaluations / worst
over the tolerance:

| problem | before | piecewise axes |
|---|---|---|
| 10 smooth cases, 5-D and 10-D | 722-4,114 | identical, bit for bit |
| `genz_continuous` 2-D / 3-D / 5-D | fails / 4.7k, 0.15 / fails | 7.5k, 0.00 / 10.9k, 0.25 / 23.7k, 0.00 |
| `genz_continuous` 10-D / 20-D | 3.4k / 3.5k | identical (no breaks needed at 0.1) |
| `genz_discontinuous` 2 / 3 / 5 / 10 / 20-D | all fail (over 80k-150k) | 7.7k / 5.1k / 5.0k / 5.9k / 7.7k, worst 0.07-0.15 |
| `front` 2-D | 87.8k, 0.35 | 99.9k, 0.35 (searches before rank growth found nothing) |

Given the true breakpoints by hand, the piecewise train is exact on both families at 2-20-D (rank 2,
2.4k-7.9k evaluations): the remaining cost is detection, about 1.4k evaluations per searched axis.

**Interfaces (scratch prototypes).** Interface points are edges located along axis-parallel fibres
(every axis through the largest and smallest f of a 256-point sample, then random ones, until a fit
holds), or median-level crossings when no fibre has an edge; the interface phi is an algebraic fit
(plane, diagonal quadric, full quadric; residual below 1e-7). (a) Composite: f ~ h(phi(x)) + remainder,
h a 1-D piecewise ChebTT in u = phi sampled along the in-box segment from the smallest to the largest
phi (inverted by bisection: an interpolated inverse misplaced the jump by 1e-4 and left a thin O(1)
spike), the remainder an ordinary ChebTT. (b) Aligned rotation for a plane whose jump varies along it:
rotate the normal to the first axis, then piecewise axes (f evaluated in the rotated box).

| problem | 2-D | 5-D | 10-D | 20-D |
|---|---|---|---|---|
| `oblique_jump`, composite | exact, 5.9k | exact, 10.5k | exact, 19.3k | exact, 40.7k |
| `ball_jump`, composite | 0.01, 7.2k | 0.19, 11.8k | no interface points | no interface points |
| `front`, composite | 0.06, 11.3k | 0.06, 12.1k | over budget | no interface points |
| varying jump on a plane, aligned rotation | exact, 7.0k | exact, 8.9k | 0.18, 14.8k | - |

At 10-D and above the ball and the front fill a vanishing share of the box (a uniform point lies ~1.9
from the ball's centre against a radius of 0.8): almost no test point is inside them, f is the
constant background almost everywhere, and discovery from samples is at the Novak-Wozniakowski limit.
They are degenerate benchmarks there and need a domain that shrinks with d.


### 1.22 One adaptive ChebTT across smooth, kinked, jumping and interface functions

A second consult (after Reviews VII-VIII and the experiments of 1.21) set the target form: one model,
f ~ g(x) + S(x), g a ChebTT with piecewise axes and S interface terms added only when the data demands
them. Done since (one seed, serial):

- **Cheaper, safer edge detection (in `chebtt.py`).** A jump is bisected on the value (about 34
  evaluations to 1e-12 of the width) and tried before a kink (a steep kink's slope is an outlier against
  a tiny median difference and was misread as a jump); a kink is located only to 0.01 share / |[f']|.
  Confirmation at 2d random far fibres: present within a band of 1e-6 of the width (the largest
  displacement is recorded as the edge's band, and the pieces' end nodes are inset by twice it, at most a
  quarter of either neighbouring piece), absent (its factor vanishing there: no veto), or found
  elsewhere (an oblique interface: rejected; the earlier 1%-displaced copy let tilts below 1e-4 pass).
  A candidate within 10x its location precision of a known edge is the same edge (a jump registered
  twice, once coarsely as a kink, inverted a piece: worst 1.5e6 tol). Search cost per axis: ~1.4k ->
  100-300 evaluations. Kinks 2.4k / 3.6k / 8.5k at 2 / 3 / 5-D, axis jumps 1.9k-7.8k at 2-20-D,
  smooth cases bit-identical; tilts <= 1e-6 held outside their band, >= 1e-4 rejected.
- **Jump fields (scratch).** For a jump whose height varies along the interface, S(x) = [side] sum_k
  J_k(foot(x)) rho^k / k!, rho the signed distance, J_0..J_2 quadratic fields of the foot point fitted to
  one-sided degree-6 fits along the normal (8 points per side, within min(0.2, room in the box)) at
  3(d+1)(d+2)/2 interface points: no evaluation outside the box. Applied on the side whose foot map is
  regular (the other side of a closed quadric holds its centre, where the foot is undefined: the 2-D ball
  failed on the wrong side). With the interface given: plane 2 / 3 / 5 / 10-D 0.6k / 1.2k / 2.6k / 4.8k,
  sphere 2 / 3 / 5-D 0.6k / 1.4k / 3.8k, residual jump ~1e-7 tol.
- **The unified prototype (scratch).** One oracle cache; ChebTT with piecewise axes on 5000 + 500 d
  evaluations first (smooth and axis-edge functions pay nothing else); on failure, diagnosis: jumps on
  axis fibres through the most extreme check point, tested at 4 far fibres ("moving" = interface
  samples), then random-fibre edge points in rounds until a fit (plane, diagonal, full quadric) has
  points to spare (one round at 10-D found 8 points where a plane needs 14, and the build fell to the
  front path); a fit that is not an axis-aligned plane is an interface (jump fields, then ChebTT on the
  remainder); no edges at all and ChebTT fails: a front, phi fitted to mid-level crossings
  ((max + min) / 2: the median sat on the front's flat tail), S = h(phi).

| class (cases) | evaluations | worst / tol | path |
|---|---|---|---|
| smooth fast set, 5-D (6) | 722-1,617, as ChebTT | 0.00-0.26 | ChebTT |
| kinks 2 / 5 / 10-D | 1.7k / 14.3k / 3.3k | 0.00 / 0.00 / 0.13 | piecewise axes |
| axis jumps 2 / 5 / 10-D | 1.7k / 2.4k / 4.5k | 0.15 / 0.14 / 0.07 | piecewise axes |
| oblique jump 2 / 5 / 10 / 20-D | 7.3k / 12.5k / 22.8k / 52.1k | 0.15-0.26 | plane, jump fields |
| spherical jump 2 / 3 / 5-D | 8.8k / 10.4k / 15.5k | 0.19-0.23 | quadric, jump fields |
| varying jump on a plane 2 / 5 / 10-D | 7.3k / 14.8k / 21.9k | 0.10 / 0.10 / 0.45 | plane, jump fields |
| varying jump on a sphere 2 / 3 / 5-D | 8.5k / 11.1k / 16.8k | 0.24-0.29 | quadric, jump fields |
| sharp front 2 / 3 / 5-D | 80k / 149k / 147k | 0.35 / 0.06 / 0.06 | ChebTT / h(phi) / h(phi) |

All 28 cases hold; plain ChebTT fails 18 of them. Open: fronts (a failed first attempt, then a large
remainder), the first attempt wasted on interface functions, constant jumps paying for full quadratic
jump fields (the composite h(phi) was exact on `oblique_jump` at 5.9k-40.7k), and 5-D kinks costing more
through the retry (7.9k -> 14.3k). The scheme's honest claim is "within the tolerance outside the
reported bands of breakpoints and interfaces".

---

## 2. Review I: spending fewer evaluations in adaptive trees

Context of this review: ART with plane splits, affine leaves and a fixed per-node budget; the
question was how to reach the same accuracy with fewer oracle evaluations. Citations were checked by
search; abstracts and summaries were read, not full texts; items marked "(recalled)" were not checked.

**Curved, oblique and model trees.**
- OC1 [I-1] searches oblique splits by hill-climbing with random restarts; it gives smaller, usually
  more accurate trees than axis-parallel ones, at ART's per-node cost.
- NDT [I-2] adds nonlinear feature combinations and splits obliquely in the lifted space, so curved
  boundaries become hyperplanes (the quadric construction of section 1.2).
- Omnivariate trees [I-3] choose the node type (univariate, linear, nonlinear) per node by a
  statistical test; they report that few nonlinear nodes are chosen, mostly near the root, and that
  nonlinear nodes everywhere overfit: the main pitfall for quadric splits.
- Hinging hyperplanes [I-4] fit a bent model of two planes joined at a hinge by alternating least
  squares.
- Model trees with polynomial leaves: SUPPORT [I-5] (optionally smoothed across pieces) and GUIDE
  [I-6] (split variables chosen by residual tests, cheaper than exhaustive search); M5 uses affine
  leaves (recalled).

**Higher-order leaves.** A full quadratic in d dimensions has (d + 1)(d + 2)/2 coefficients (6 at
d = 2, 66 at d = 10); Powell [I-7] fits one from 2d + 1 points by the least Frobenius norm of the
Hessian. Local error is O(h^2) for affine and O(h^3) for quadratic pieces, so for smooth f the leaf
counts scale like tol^(-d/2) and tol^(-d/3) (recalled).

**Sample-efficient acceptance.**
- Hoeffding trees [I-8] split as soon as a concentration bound separates the candidates.
- Simplex stochastic collocation [I-9] and adaptive sparse grids [I-10] use the old model's residual
  at a new point as a free per-cell error indicator; ME-gPC [I-11] splits elements by a local
  indicator.
- Treed and dynamic Gaussian processes [I-12, I-13] do active learning within leaves; too heavy per
  leaf here.
- The rule of three [I-14]: if k fresh uniform points all pass, the violating volume fraction is at
  most about 3/k at 95% confidence, so a budget of 20(d + 1) points guards about 5% of the volume at
  d = 2 and 0.7% at d = 20.

**Splits that follow the function.**
- Optimal anisotropic meshes [I-15]: piecewise-linear interpolation is near-optimal on meshes
  quasi-uniform in the metric |H|; the error then scales with det|H|^(1/d) instead of lambda_max.
- Greedy bisection [I-16, I-17] reaches the optimal aspect ratio, with error O(N^-1) in 2-D.
- BSP partitions with polynomial pieces [I-18] have an approximation theory for arbitrary planes.
- Surflets [I-19]: cells cut by polynomial surfaces reach near-optimal rates for piecewise-smooth f.
- Discontinuity localisation by an actively refined SVM boundary [I-20], tested up to d = 11.

**Ranked ideas from this review.** (1) Staged acceptance on fresh points: a small fit budget plus k
validation points, about 2-3x fewer evaluations at d >= 5, with a volume guarantee of 3/k. (2) A
minimax (L-infinity) affine fit by Lawson's reweighted least squares: (4/3)^(d/2) fewer leaves on
convex pieces, nothing on saddles. (3) Choosing the leaf model per node (affine, diagonal, full
quadratic). (4) Parent-Hessian-informed splits at no evaluation cost. (5) Low-rank quadric splits,
gated by an omnivariate test. (6) Merging sibling leaves bottom-up for faster inference.

**References.**
- [I-1] Murthy, Kasif, Salzberg 1994, *A system for induction of oblique decision trees*, https://arxiv.org/abs/cs/9408103
- [I-2] Ittner, Schlosser 1996, *Non-linear decision trees (NDT)*, ICML, https://mlanthology.org/icml/1996/ittner1996icml-non/
- [I-3] Yildiz, Alpaydin 2001, *Omnivariate decision trees*, IEEE TNN, doi:10.1109/72.963795
- [I-4] Breiman 1993, *Hinging hyperplanes for regression, classification, and function approximation*, IEEE TIT, doi:10.1109/18.256506
- [I-5] Chaudhuri, Huang, Loh, Yao 1994, *Piecewise-polynomial regression trees*, Statistica Sinica 4, https://pages.stat.wisc.edu/~loh/treeprogs/guide/smooth.pdf
- [I-6] Loh 2002, *Regression trees with unbiased variable selection and interaction detection*, Statistica Sinica 12, https://www3.stat.sinica.edu.tw/statistica/j12n2/j12n21/j12n21.htm
- [I-7] Powell 2004, *Least Frobenius norm updating of quadratic models*, Math. Prog. 100
- [I-8] Domingos, Hulten 2000, *Mining high-speed data streams*, KDD, https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf
- [I-9] Witteveen, Iaccarino 2012, *Simplex stochastic collocation*, SISC, doi:10.1137/100817504
- [I-10] Jakeman, Roberts 2011, *Local and dimension adaptive sparse grid interpolation and quadrature*, https://arxiv.org/abs/1110.0010
- [I-11] Wan, Karniadakis 2005, *An adaptive multi-element generalized polynomial chaos method*, https://www.math.lsu.edu/~xlwan/papers/journal/megpc1.pdf
- [I-12] Gramacy, Lee 2008, *Bayesian treed Gaussian process models*, https://arxiv.org/abs/0710.4536
- [I-13] Taddy, Gramacy, Polson 2011, *Dynamic trees for learning and design*, https://arxiv.org/abs/0912.1586
- [I-14] Hanley, Lippman-Hand 1983, *If nothing goes wrong, is everything all right?*, JAMA
- [I-15] Chen, Sun, Xu 2007, *Optimal anisotropic meshes for minimizing interpolation errors in L^p-norm*, Math. Comp. 76, https://www.math.uci.edu/~chenlong/Papers/Chen.L%3BSun.P%3BXu.J2007.pdf
- [I-16] Mirebeau, Cohen 2012, *Greedy bisection generates optimally adapted triangulations*, https://arxiv.org/abs/1101.1452
- [I-17] Cohen, Dyn, Hecht, Mirebeau 2012, *Adaptive multiresolution analysis based on anisotropic triangulations*, https://arxiv.org/abs/1101.1512
- [I-18] Dekel, Leviatan 2005, *Adaptive multivariate approximation using binary space partitions and geometric wavelets*, SINUM, doi:10.1137/040604649
- [I-19] Chandrasekaran, Wakin, Baron, Baraniuk 2009, *Representation and compression of multidimensional piecewise functions using surflets*, https://dspace.mit.edu/handle/1721.1/52325
- [I-20] Gorodetsky, Marzouk 2014, *Efficient localization of discontinuities in complex computational simulations*, https://arxiv.org/abs/1402.2845

---

## 3. Review II: structure-aware surrogates with max-norm control

Context of this review: least-squares pieces on random points fail the worst-error requirement in
higher dimensions; the question was which schemes discover the function's own structure and control
the worst error. Eval-count estimates for d = 10 are the reviewer's own, from known rates and ranks.

**Adaptive Chebyshev and spectral methods.**
- Chebfun in 1-D [II-2] samples on 2^k + 1 Chebyshev points, stops when the coefficients level off
  at the tolerance ("chopping") and checks two extra points ("sampleTest"). For f analytic on the
  Bernstein ellipse E_rho with |f| <= M there, ||f - p_n|| <= 4 M rho^-n / (rho - 1) [II-3, Thm 8.2]:
  a certificate only when rho and M are known; in practice the tail estimates the error.
- Chebfun2 and Chebfun3 [II-1, II-4] build low-rank Chebyshev approximations; Chebfun3F cuts
  evaluations by about 75% [II-5]. Very effective for d = 2-3; ranks grow for rotated or radial
  features.
- Overlapping Chebyshev patches on a recursive bisection, split where coefficients do not decay and
  blended by a partition of unity [II-6]; spectral-element hp strategies [II-7]. Tensor patches limit
  these to about d = 4.

**High-dimensional spectral and low-rank methods.**
- Smolyak sparse grids on Clenshaw-Curtis points [II-8] converge like N^-k times a log factor whose
  power grows with d; the dimension-adaptive version [II-9] grows the index set by surplus. Excellent
  for axis-aligned anisotropy and few interactions, poor for rotated features; the surplus can stop
  refinement early.
- TT-cross [II-10, II-11] interpolates on fibres chosen by maxvol, within a modest factor of the best
  rank-r error [II-12], at about d n r^2 evaluations per sweep. Spectral TT [II-13], the function
  train [II-14] and the extended functional TT [II-15]. The max error is estimated only on check
  points, and pivots can miss a localised feature.
- Isotropy in the hypercube [II-16]: approximability follows the Euclidean degree, so rotated features
  cost more in every axis-based format.

**Structure discovery.**
- Active subspaces [II-17]: eigenvectors of `E[grad f grad f^T]`; a mean-square (Poincare) bound, not
  max-norm; no gap for a radial front centred in the box.
- Ridge recovery with uniform error: Fornasier-Schnass-Vybiral [II-18] (high probability, polynomial
  in d) and Cohen et al. [II-19] (deterministic, needs a positive or sparse direction); polynomial
  ridge fits by variable projection [II-20].
- Nonlinear coordinates: active manifolds [II-21], nonlinear level-set learning [II-22], pseudo-reversible
  networks [II-23], polynomial feature maps matched to grad f [II-24]; all report L2 errors and need
  many gradients.
- Anisotropic metric adaptation [II-25, I-16]: optimal L^p error for piecewise-linear pieces, still
  algebraic (N^(-2/d)) and practical only for d <= 3.

**Certified max error for d > 3.** Novak and Wozniakowski [II-26]: even for C-infinity functions with
all derivatives bounded by 1, uniform approximation needs at least 2^(floor(d/2)) samples, so no
black-box method certifies the max error in high dimension. Certificates come only from an assumed
class: a Bernstein ellipse [II-3], an RKHS norm bound for Gaussian processes [II-27], or a Lipschitz
constant. Failure modes of the practical estimators: aliasing (a false plateau of coefficients),
features between samples ("fooling" functions [II-28]), symmetry zeroing surpluses, TT pivots missing
a peak, and Gibbs oscillation at discontinuities (detectable on sparse grids [II-29]).

**Ranked schemes from this review.** (1) Chebyshev TT-cross after a gradient-based rotation when the
gradient covariance has a gap, with degrees from coefficient decay and validation on random points
plus a local maximisation of |f_hat - f|. (2) A dimension-adaptive Clenshaw-Curtis sparse grid (an
existing library, e.g. Tasmanian, can do it). (3) A polynomial level-set map phi with grad phi
parallel to grad f, then f ~ h(phi) with h an adaptive 1-D Chebyshev series. (4) Chebyshev
partition-of-unity patches with hp refinement from coefficient decay (d <= 4). (5) Ridge recovery
with uniform error, when f is a ridge.

**References.**
- [II-1] Townsend, Trefethen 2013, *An extension of Chebfun to two dimensions*, SISC 35(6), https://people.maths.ox.ac.uk/trefethen/publication/PDF/townsend_trefethen13.pdf
- [II-2] Aurentz, Trefethen 2017, *Chopping a Chebyshev series*, ACM TOMS 43(4), https://arxiv.org/abs/1512.01803
- [II-3] Trefethen 2013/2019, *Approximation Theory and Approximation Practice*, SIAM, https://people.maths.ox.ac.uk/trefethen/ATAP/
- [II-4] Hashemi, Trefethen 2017, *Chebfun in three dimensions*, SISC 39(5), https://people.maths.ox.ac.uk/trefethen/chebfun3_final.pdf
- [II-5] Dolgov, Kressner, Stroessner 2021, *Functional Tucker approximation using Chebyshev interpolation*, SISC 43(3), https://arxiv.org/abs/2007.16126
- [II-6] Aiton, Driscoll 2019, *An adaptive partition of unity method for multivariate Chebyshev polynomial approximations*, SISC 41(5), https://arxiv.org/abs/1805.00423
- [II-7] Mavriplis 1994, *Adaptive mesh strategies for the spectral element method*, CMAME 116, https://ntrs.nasa.gov/api/citations/19920022021/downloads/19920022021.pdf
- [II-8] Barthelmann, Novak, Ritter 2000, *High dimensional polynomial interpolation on sparse grids*, Adv. Comput. Math., https://users.fmi.uni-jena.de/~novak/sparse.html
- [II-9] Gerstner, Griebel 2003, *Dimension-adaptive tensor-product quadrature*, Computing 71, https://ins.uni-bonn.de/media/public/publication-media/dimadapt.pdf
- [II-10] Oseledets, Tyrtyshnikov 2010, *TT-cross approximation for multidimensional arrays*, LAA 432, https://doi.org/10.1016/j.laa.2009.07.024
- [II-11] Oseledets, *Black box cross approximation*, https://oseledets.github.io/news/black-box-cross-approximation-for-multidimensional-arrays
- [II-12] Savostyanov 2014, *Quasioptimality of maximum-volume cross interpolation of tensors*, LAA 458, https://arxiv.org/abs/1305.1818
- [II-13] Bigoni, Engsig-Karup, Marzouk 2016, *Spectral tensor-train decomposition*, SISC 38(4), https://arxiv.org/abs/1405.5713
- [II-14] Gorodetsky, Karaman, Marzouk 2019, *A continuous analogue of the tensor-train decomposition*, CMAME 347, https://arxiv.org/abs/1510.09088
- [II-15] Stroessner, Sun, Kressner 2024, *Approximation in the extended functional tensor train format*, Adv. Comput. Math., https://arxiv.org/abs/2211.11338
- [II-16] Trefethen 2017, *Cubature, approximation, and isotropy in the hypercube*, SIAM Rev., https://epubs.siam.org/doi/10.1137/16M1066312
- [II-17] Constantine, Dow, Wang 2014, *Active subspace methods in theory and practice*, SISC 36(4), https://arxiv.org/abs/1304.2070
- [II-18] Fornasier, Schnass, Vybiral 2012, *Learning functions of few arbitrary linear parameters in high dimensions*, FoCM 12, https://arxiv.org/abs/1008.3043
- [II-19] Cohen, Daubechies, DeVore, Kerkyacharian, Picard 2012, *Capturing ridge functions in high dimensions from point queries*, Constr. Approx. 35, https://link.springer.com/article/10.1007/s00365-011-9147-6
- [II-20] Hokanson, Constantine 2018, *Data-driven polynomial ridge approximation using variable projection*, SISC 40(3), https://arxiv.org/abs/1702.05859
- [II-21] Bridges, Gruber, Felder, Verma, Hoff 2019, *Active manifolds*, ICML, https://proceedings.mlr.press/v97/bridges19a.html
- [II-22] Zhang, Zhang, Hinkle 2019, *Learning nonlinear level sets for dimensionality reduction in function approximation*, NeurIPS, https://neurips.cc/virtual/2019/poster/14262
- [II-23] Teng, Wang, Ju, Gruber, Zhang 2023, *Level set learning with pseudo-reversible neural networks*, SISC 45(3), https://arxiv.org/abs/2112.01438
- [II-24] Bigoni, Marzouk, Prieur, Zahm 2022, *Nonlinear dimension reduction for surrogate modeling using gradient information*, https://arxiv.org/abs/2102.10351
- [II-25] Loseille, Alauzet 2011, *Continuous mesh framework I/II*, SINUM 49, https://hal.archives-ouvertes.fr/hal-03167252
- [II-26] Novak, Wozniakowski 2009, *Approximation of infinitely differentiable multivariate functions is intractable*, J. Complexity, https://users.fmi.uni-jena.de/~novak/ca1.pdf
- [II-27] Fiedler, Scherer, Trimpe 2021, *Practical and rigorous uncertainty bounds for Gaussian process regression*, AAAI, https://arxiv.org/abs/2105.02796
- [II-28] Hickernell, Jiang, Liu, Owen 2013, *Guaranteed conservative fixed width confidence intervals via Monte Carlo sampling*, https://arxiv.org/abs/1208.4318
- [II-29] Jakeman, Archibald, Xiu 2011, *Characterization of discontinuities in high-dimensional stochastic problems on adaptive sparse grids*, JCP 230, https://www.osti.gov/servlets/purl/1345533

---

## 4. Review III: from the experiments' failures to chemistry tabulation

Context of this review: the failures of section 1 (aliasing, optimistic tails at kinks, overshoot,
early stopping of sparse grids, the dimension ceiling) and the chemistry goal. Items marked
"(recalled)" were not re-checked online; "(abstract)" were checked against an abstract or summary only.

**Robust Chebyshev acceptance.**
- Chebfun's `standardChop` [III-1] takes the monotone upper envelope of the coefficients and is
  satisfied only by a plateau that is long, flat and low enough (up to tol^(2/3) if perfectly flat),
  then chops at its elbow. Coefficients alone cannot see aliasing: on 17, 33 and 65 points T_128
  looks like the constant 1. So `sampleTest` [III-2] compares f with the interpolant at two fixed
  off-grid points (x = -0.358 and 0.0368, at about sqrt(eps) times the scale). The authors note that
  a deterministic test can be outwitted and a randomised one defeats any fixed adversary; neither is
  a worst-case guarantee.
- Bernstein-ellipse bounds [II-3] (recalled): |a_k| <= 2 M rho^-k and the tail is 2 M rho^-n / (rho - 1).
  With rho fitted from the coefficients the bound blows up as rho -> 1, a natural split trigger,
  but it is rigorous only when M and rho are known.
- Euclidean degree [III-3]: for analytic f on the hypercube the right polynomial space is bounded
  by the 2-norm of the exponent vector; total degree favours axes over diagonals and maximal degree
  the reverse, which bears directly on oblique ridges.

**Estimating smoothness for hp decisions.**
- Mavriplis [II-7]: fit the last four Legendre coefficients to C e^(-sigma j); raise the degree if
  sigma > 1, split otherwise; the fit also estimates the truncation tail.
- Houston and Suli [III-4] (abstract): local Sobolev regularity by a root test on the coefficients.
- Eibner and Melenk [III-5] (abstract): coefficients decay exponentially if and only if f is
  analytic; fit a decay law per element, raise the degree if it fits, split otherwise.
- Mitchell and McClain [III-6] compare 13 hp strategies on 20 elliptic problems: reference-solution
  strategies need the fewest degrees of freedom but are costly, and coefficient decay (Mavriplis) is
  the best general strategy, in the energy norm, not the max norm.
- ME-gPC [I-11] (abstract) splits an element when its high-mode variance share times its measure
  exceeds a threshold; the weighting stops refinement at a minimum size near a discontinuity.

**Robust and local sparse grids.**
- Dimension-adaptive index sets [II-9] use heuristic surplus indicators; nothing prevents early
  stopping when an interaction is never opened.
- Locally refined hierarchical sparse grids: ASGC [III-7] (abstract), local plus dimension
  adaptivity [I-10] (abstract), spatially adaptive grids with modified boundary bases [III-8]
  (abstract).
- Sparse grids inside the elements of an adaptive partition, ME-PCM [III-9] (abstract): sparse grids
  in a tree of subdomains, with an L^2 analysis.
- Leja sequences are nested at any size, so the degree can grow by one instead of doubling [III-10]
  (abstract); greedy sparse interpolation on Leja points has sup-norm rates for analytic, anisotropic
  f under holomorphy assumptions [III-11] (abstract).
- No source gives a surplus stopping rule guaranteed not to stop early.

**High dimension with uniform error.**
- Maximum-volume TT cross is quasi-optimal in the max norm on the tensor grid, with a constant that
  need not grow exponentially in d, if f is close to low rank [II-12]; nothing is said off the grid.
- Spectral and functional TT with Chebyshev or Legendre bases, cross approximation and rank
  adaptivity [II-13, II-14] (abstract): exponential convergence for smooth f. Failure modes:
  underestimated ranks, localised features missed by fibre sampling, dependence on the variable order,
  rank inflation at kinks.
- Realistic dimensions: tensor boxes about 4; sparse grids about 10-20 at low interaction; TT tens or
  more at small ranks; ISAT about 50 or more, in situ.

**Chemistry tabulation.**
- ISAT (Pope 1997) [III-12] tabulates only the accessed part of composition space, with
  piecewise-linear maps valid inside ellipsoids of accuracy (EOAs). Each query is a retrieve, a grow
  or an add, with a direct ODE solve as the fallback: 16 species, about 1000x speedup.
- Lu and Pope 2009 [III-13] add better searching and error checking and correction: randomly chosen
  retrieves (about 10% of the work) are checked against a direct solve and an offending EOA shrinks.
  Tested in 54 dimensions; the error distribution has no sharp cut-off at the tolerance, and the
  authors control its 90% quantile, not its maximum.
- Liu and Pope 2005 [III-14]: local errors are well controlled with a small probability of large
  ones, caused by non-convex regions of accuracy; the global error is linear in the tolerance.
- PRISM [III-15] (abstract): quadratic polynomials fitted on demand in hypercubes of composition
  space, a ChebTree built in situ.
- Dynamic adaptive chemistry [III-16] (abstract), coupled with ISAT [III-17] (abstract): local
  mechanism reduction on the fly, up to 857 species, about 10x.
- Slow manifolds: ILDM [III-18] and CSP [III-19] (recalled); principal-component transport for
  combustion [III-20] (abstract).

**Astrochemistry surrogates** (accuracy as reported; none controls the worst error or falls back to
the solver, and no use of ISAT in astrochemistry was found, in a search that was not exhaustive):

| Source | Method | Size | Accuracy | Speedup |
|---|---|---|---|---|
| Chemulator [III-21] | neural emulator of UCLCHEM | not recorded | log MSE about 2e-4 per 1000-yr step, 6e-3 after 1000 steps; large errors for some models | about 5e4 |
| Grassi et al. 2022 [III-22] | autoencoder and latent ODE | 29 species, 224 reactions to 5 latent species, 12 reactions | not recorded | 65x |
| Sulzer and Buck 2023 [III-23] | latent neural ODE | not recorded | relative error below the baseline | 55x; linear latent model up to 4000x |
| CODES [III-24] | 4 surrogate architectures | includes osu2008, 29 species | mean relative error below 4% | 10^3 or more |
| MACE [III-25] | autoencoder and latent ODE, circumstellar envelopes | not recorded | not recorded | about 26x |
| Branca and Pallottini 2023 [III-26] | physics-informed network | 9 species | about 10% | about 200x |
| Branca and Pallottini 2024 [III-27] | DeepONet | 9 species | about 1e-2 | 128x |
| Ono and Sugimura 2025 [III-28] | five DeepONets, primordial chemistry | 6 species | below 10% relative error in over 90% of cases | 10x CPU, over 1000x GPU |
| Palud et al. 2023 [III-29] | emulator of the Meudon PDR code | not recorded | mean 4.5%; worst case 3x better than interpolation | not recorded |

**Error-controlled fallback.** ISAT-style methods answer from the table inside an EOA and solve and
tabulate otherwise. In practice the bound is statistical: the tolerance sets the local linearisation
error when a record grows, error checking re-checks a random sample of answers, and the reports give
quantiles, not a maximum [III-13, III-14]. A guaranteed worst error needs a verified local bound or a
check of every answer, which pays only if checking is cheaper than solving.

**References.**
- [III-1] Aurentz, Trefethen 2017, *Chopping a Chebyshev series*, ACM TOMS 43(4):33, https://arxiv.org/abs/1512.01803 (= [II-2])
- [III-2] Chebfun, `@chebtech/sampleTest.m`, https://github.com/chebfun/chebfun
- [III-3] Trefethen 2017, *Multivariate polynomial approximation in the hypercube*, Proc. AMS 145:4837, https://arxiv.org/abs/1608.02216
- [III-4] Houston, Suli 2005, *A note on the design of hp-adaptive finite element methods for elliptic partial differential equations*, CMAME 194:229
- [III-5] Eibner, Melenk 2007, *An adaptive strategy for hp-FEM based on testing for analyticity*, Comput. Mech. 39:575, https://doi.org/10.1007/s00466-006-0107-0
- [III-6] Mitchell, McClain 2014, *A comparison of hp-adaptive strategies for elliptic partial differential equations*, ACM TOMS, https://math.nist.gov/~WMitchell/papers/nistir7824.pdf
- [III-7] Ma, Zabaras 2009, *An adaptive hierarchical sparse grid collocation algorithm*, JCP, https://doi.org/10.1016/j.jcp.2009.01.006
- [III-8] Pflueger 2010, *Spatially Adaptive Sparse Grids for High-Dimensional Problems*, Dr. Hut (TUM thesis)
- [III-9] Foo, Wan, Karniadakis 2008, *The multi-element probabilistic collocation method*, JCP 227:9572, https://www.math.lsu.edu/~xlwan/papers/journal/mepcm.pdf
- [III-10] Narayan, Jakeman 2014, *Adaptive Leja sparse grid constructions for stochastic collocation*, SISC 36:A2952, https://arxiv.org/abs/1404.5663
- [III-11] Chkifa, Cohen, Schwab 2014, *High-dimensional adaptive sparse polynomial interpolation and applications to parametric PDEs*, FoCM 14:601
- [III-12] Pope 1997, *Computationally efficient implementation of combustion chemistry using in situ adaptive tabulation*, CTM 1:41, https://tcg.mae.cornell.edu/pubs/Pope_CTM_97.pdf
- [III-13] Lu, Pope 2009, *An improved algorithm for in situ adaptive tabulation*, JCP 228:361, https://tcg.mae.cornell.edu/pubs/Lu_LRP_JCP_09.pdf
- [III-14] Liu, Pope 2005, *The performance of in situ adaptive tabulation in computations of turbulent flames*, CTM 9:549, https://tcg.mae.cornell.edu/pubs/Liu_P_CTM_2005.pdf
- [III-15] Tonse, Moriarty, Brown, Frenklach 1999, *PRISM: piecewise reusable implementation of solution mapping*, Israel J. Chem. 39:97, https://escholarship.org/uc/item/6sr15042
- [III-16] Liang, Stevens, Farrell 2009, *A dynamic adaptive chemistry scheme for reactive flow computations*, Proc. Combust. Inst. 32:527
- [III-17] Contino et al. 2011, *Coupling of in situ adaptive tabulation and dynamic adaptive chemistry*, Proc. Combust. Inst. 33:3057
- [III-18] Maas, Pope 1992, *Simplifying chemical kinetics: intrinsic low-dimensional manifolds in composition space*, Combust. Flame 88:239
- [III-19] Lam, Goussis 1994, *The CSP method for simplifying kinetics*, Int. J. Chem. Kinet. 26:461
- [III-20] Sutherland, Parente 2009, *Combustion modeling using principal component analysis*, Proc. Combust. Inst. 32:1563, https://doi.org/10.1016/j.proci.2008.06.147
- [III-21] Holdship, Viti, Haworth, Ilee 2021, *Chemulator: fast, accurate thermochemistry for dynamical models through emulation*, A&A 653:A76, https://arxiv.org/abs/2106.14789
- [III-22] Grassi et al. 2022, *Reducing the complexity of chemical networks via interpretable autoencoders*, A&A 668:A139, https://arxiv.org/abs/2104.09516
- [III-23] Sulzer, Buck 2023, *Speeding up astrochemical reaction networks with autoencoders and neural ODEs*, NeurIPS ML4PS workshop, https://arxiv.org/abs/2312.06015
- [III-24] Janssen, Sulzer, Buck 2024, *CODES: benchmarking coupled ODE surrogates*, https://arxiv.org/abs/2410.20886
- [III-25] Maes, De Ceuster, Van de Sande, Decin 2024, *MACE: a machine learning approach to chemistry emulation*, ApJ, https://arxiv.org/abs/2405.03274
- [III-26] Branca, Pallottini 2023, *Neural networks: solving the chemistry of the interstellar medium*, MNRAS, https://arxiv.org/abs/2211.15688
- [III-27] Branca, Pallottini 2024, *Emulating the interstellar medium chemistry with neural operators*, A&A 684:A203, https://arxiv.org/abs/2402.12435
- [III-28] Ono, Sugimura 2025, *Neural-network chemical emulator for first-star formation*, ApJ 996:9, https://arxiv.org/abs/2508.16114
- [III-29] Palud et al. 2023, *Neural network-based emulation of interstellar medium models*, A&A 678:A198, https://arxiv.org/abs/2309.01724

---

## 5. Review IV: kink splits, the limits in high dimension, a design for chemistry

Scope: builds on docs/SURROGATE_RESEARCH.md (refs I-*, II-*, III-* are used without repeating them).
New references are IV-* (the IDs of docs/LITERATURE.md). Verification tags: **[v]** title/venue/claim checked against the paper or its
abstract in this review; **[a]** checked against an abstract or summary only; **[r]** recalled, not checked.
Numbers marked "measured" come from short runs of `src/structure/schemes/chebtree.py` as of the fibre-check commit.

### 5.0 Bottom line

1. Kink-located splits (section 1.7) cost more for three reasons, each reproduced: (a) a kink a small but non-negligible distance from a
   face reads as a jump, so its box can never pass and is bisected toward the face; (b) the secant locator is
   too coarse and often returns nothing, so it creates exactly those near-face kinks; (c) the gate
   "non-exponential axis" fires on under-resolved smooth features and on curved jumps, where no single plane
   helps. The near-face kink that ChebTree accepted (section 1.7) is a separate blind spot of the decay classifier, and it is structural.
2. Beyond d = 4, no tensor or Smolyak scheme on the box can promise the worst error without strong structure.
   The structure chemistry has is low intrinsic dimension (a slow manifold) and Jacobians of the time-step map
   with low numerical rank. So the next scheme should be built on the visited states, not on a box: an
   ISAT-type table with local models in local slow coordinates, accepted by look-ahead checks, with refusal
   and a solver fallback.
3. Your fibre check at the next nested points carries over to every candidate scheme as "check where the
   refinement would sample next". It is the most reliable estimator class in this review; it is still not a
   bound.

### 5.1 Why splitting at a located kink costs more

**Mechanism A, the near-face trap (measured).** Take 1-D `exp(-a|x - x0|)` on a box of width w, a = 11.9,
n = 16. With the kink more than 3% of w from a face, the fitted power is p ≈ 1.8–2.6 and the estimate is
3–9x the true error. Within 1% of a face, the slab sums are flat (p ≈ 0) and `tail_estimate` returns inf
("a jump never passes"). The true error there is only 1–3e-3. The reason: a kink at x0 = cos θ0 has
coefficients ∝ sin θ0 · cos(jθ0)/j², which stay flat for j < 1/θ0 when θ0 is small. The box can only be
accepted after splitting toward the face until it falls under the rounding floor. The 2-D run
(genz_continuous at 0.1, kink mode) shows this directly: the split values on axis 1 approach w1 from above
as w1 + 3.3e-2, 1.6e-2, 8.1e-3, …, 1.2e-4, a halving sequence. The run ends with 87 leaves against 23, a
smallest width of 2e-5 against 3e-2, and 11.5k evaluations against 2.9k.

**Mechanism B, the locator (measured).** Of 86 calls, 61 returned None and fell back to the midpoint
(at n = 8 it returns None even for interior kinks at 0.3w and 0.5w). Of the 25 hits, 14 landed 1–5% from a
face. When the locator does fire, its error is about 1e-3 to 1e-4 of w. That leaves the kink in a child at
exactly the distance that triggers mechanism A. Placing the kink somewhere in the 1e-12 to 1e-2 band is
worse than placing it at random.

**Mechanism C, the gate.** "Non-exponential" means "not resolved yet", not "kinked". On the Gaussian, 8 of
13 calls returned a "kink" in a smooth function (1.2k → 1.9k evaluations). On ball_jump the interface is
curved, so a plane fitted to one fibre does not separate it. Off-centre cuts also shrink boxes more slowly
per level, and `max_depth` counts splits, not widths. So the untrusted band at the depth cap is wider:
refused queries go from 1.3% to 4%.

**The near-face blind spot (measured, structural).** The wrongly accepted 3-D box has its kink at 0.0102/0.125 = 0.082 of
its width. At n = 8 its slab sums for j = 3..8 are 8.4e-3, 6.7e-3, 4.0e-3, 1.7e-3, 1.0e-4, 2.3e-4. That is
the first half-period of the cos(jθ0) modulation (θ0 = 0.58, period 10.8 in j), read as a 36x drop, so the
axis counts as "exponential". The tail estimate is 3.8e-4 against a true error of 3.4e-2. A scan over kink
positions gives a smallest estimate/true ratio of 0.03 (at 0.081, n = 8) and 0.04 (at 0.02, n = 16),
whatever a and w are. Any kink at about 2π/n of a box's width from a face fools the classifier. Your fibre
check at the next nested points works because degree 2n covers a full period of the modulation.

**What the literature does.**
- Chebfun's splitting [IV-1, v] works on intervals that fail ("nafs"). `detectedge` estimates |f'|…|f''''|
  by finite differences on a 50-point and then 15-point equispaced grid, zooming 7x per level as long as an
  estimate grows by at least 1.2x. A jump in f' is bisected "to the very last bit". This costs about 300
  evaluations per edge, against about 7000 for Chebyshev bisection. An edge closer than 1e-14(b - a) to an
  end is cut at 0.01(b - a) instead. With no edge, the cut is at the midpoint, and these "removable" cuts
  are merged afterwards. So chebfun never leaves a kink inside the band where mechanism A bites.
- Polynomial annihilation [IV-2, v] computes a jump function that converges to the jump at an edge and to
  O(h^m) elsewhere, with minmod over the orders m. It tells jumps from steep gradients on scattered data: the
  missing gate. Concentration factors [IV-3, r] give the same from spectral coefficients, so they could run
  on the coefficients a box already has.
- On sparse grids, PA guides refinement of the discontinuity only. In the Jakeman-Archibald-Xiu work
  (II-29) the number of points for a hyperspherical jump grew linearly in d up to d = 100 [IV-4, a: from the
  DOE final report]. Minimal multi-element collocation [IV-5, a] cuts elements along the detected
  discontinuity instead of along the axes.
- hp practice: the cells next to a singularity get low degree, and the graded mesh does the work
  (Babuška-Guo [IV-20, r]). Wilka-Lang [IV-6, a] do the same in sparse grids: local low-order bases in the
  non-smooth region, with optional kink detection to choose the degree.

**Fixes, ranked by gain over cost.**
1. *Locate precisely or not at all.* Run chebfun-style edge detection on one fibre (about 50–300
   evaluations, far less than one 9x17x17 box), refine to about 1e-12 w, and confirm on a second fibre
   (agreement within the refinement width means an axis-aligned plane). If the kink is not confirmed,
   split at the midpoint. Expected: the 3-D kink case falls to the cost of about 8 smooth boxes
   (unmeasured), and the smooth cases and the jumps go back to their midpoint costs.
2. *Gate with PA:* take a kink split only when the jump in f' is confirmed and the jump in f is about 0.
3. *Make the depth cap a width cap* (h_min per axis), and size each child's degree by its width.
4. *Keep the next-nested fibre check* as the acceptance test (it is already validated).

All of this is low-dimensional engineering. Ship fixes 1, 2 and 4 into whatever partition scheme comes
next, and invest no further in tensor boxes.

### 5.2 What is achievable in high dimension from point queries

| Structure assumed | Best sup-norm behaviour | Algorithm | Fits chemistry? |
|---|---|---|---|
| C^∞, all derivatives ≤ 1 | intractable: at least 2^⌊d/2⌋ samples (II-26) | none | – |
| mixed smoothness r | about N^-r (log N)^((d-1)(r+1)) [IV-7, r] | sparse grids | the log factor bites beyond d ≈ 10; axis-bound |
| anisotropic holomorphy, (b_j) ∈ ℓ^p | best n-term n^-(1/p - 1), independent of d [IV-8, a] | adaptive Leja / lower sets (III-10, III-11) | needs decaying variable importance; raw species coordinates do not have it |
| ridge g(Ax), A of size k x d | samples polynomial in d, uniform error (II-18 w.h.p., II-19) | ridge recovery | globally no; locally yes (below) |
| low TT rank r | d n r² per sweep; quasi-optimal on the grid only (II-12) | TT-cross, C3, spectral TT (II-10, II-13, II-14) | depends on variable order; rank grows at kinks |
| f in a Sobolev RKHS of order τ on an m-dimensional manifold | rates set by m, not d [IV-9, v]; P-greedy quasi-optimal [IV-10, v]; f-greedy better by about n^-1/2 [IV-11, v] | VKOGA, β-greedy [IV-12, v; IV-13, a] | yes: tracer data lie on the manifold |
| f within ε of a known n-dimensional space V | 2n samples recover within (1 + C√n) ε [IV-14, a] | sampling projection | yes, if V comes from local POD or Jacobians |

The key fact for chemistry is the reviewer's inference from ILDM/CSP theory (III-18, III-19), not from a
citation. Over a step Δt much longer than the fast time scales, the fast modes contract like exp(λΔt). So the
Jacobian ∂Φ_Δt/∂x has numerical rank about m, the number of slow modes, and the map is locally a ridge in m
directions. The useful assumption is therefore "locally a ridge with small m, on a manifold visited by the
simulation", not "smooth on a box". This should be measured on Robertson and on the primordial network
before the design is built on it.

Kernels give the one rigorous pointwise bound in this list: |f - s|(x) ≤ P_X(x) · ||f||_H. ||f||_H is unknown,
and the growth of ||s_n||_H gives only a heuristic plateau estimate. P_X(x) is still an honest "how far is x
from the data" measure, which makes it a natural refusal test.

### 5.3 A combination for 10–100 inputs on a low-dimensional manifold

**Design (ranked components):**

1. **Query-driven table (ISAT's contract, III-12, III-13).** Each record holds a state x_i, the map value
   Φ(x_i), the Jacobian J_i (from a sensitivity solve, cheaper than d + 1 solves [r]) and its SVD
   J_i = U Σ V^T, which splits slow from fast directions.
2. **A local model in slow coordinates.** Use the linear map in every direction, plus a correction in the m
   leading right singular directions: a quadratic, a small Leja sparse grid in m dimensions, or a VKOGA
   kernel model. This is a local active-subspace model, with the subspace taken from J and no sampling of
   gradients.
3. **A validity region and refusal.** An ellipsoid of accuracy shaped by Σ (ISAT), intersected with a kernel
   power-function bound. A query outside both goes to the solver and becomes a grow or an add.
4. **Acceptance by look-ahead.** Accept a grow only after a direct solve at the points the next refinement
   would use: ± the EOA semi-axis along the leading slow directions, the analogue of your fibre check.
5. **Error checking.** Re-solve a random share of retrievals and shrink the offending records
   (Lu-Pope 2009). Report the maximum, not only the 90% quantile.
6. Neural compression of mature subtrees (Neural-ISAM [IV-15, a]), only after the table is trusted. Its error
   control is not stated in the abstract.

Element conservation, positivity, and errors measured on log-abundances are part of the contract. Species
span 20 orders of magnitude, so a relative tolerance on max |f| is meaningless.

**Cheap 5-D tests (each under a minute, analytic oracles):**

- T1, the local-rank premise. On Lorenz-96 (forcing 1) and a 5-D Davis-Skodje-type system [IV-24, r] (1 slow and 4
  fast modes with a known manifold), plot the singular values of J_Δt along trajectories against Δt. Go on
  only if the numerical rank is clearly below d.
- T2, the table against ChebTree and SparseCheb on trajectory queries. Train on 50 trajectories, test on
  50 held-out ones. Report the worst error, the solver-call rate, the table size, and whether the
  ±semi-axis look-ahead catches the failures random checking misses.
- T3, calibration of the kernel refusal. Fit VKOGA f/P-greedy on the visited set and plot the true error
  against P_X(x)·||s||_H on held-out on-manifold and off-manifold points. The bound should rank the errors
  even where it is not conservative.
- T4, a box fallback for the bench: a tree of Leja sparse grids with PA-gated kink cuts (fixes 1–2) and the
  forward-neighbour look-ahead. Run it on 5-D genz_continuous, corner_peak, oblique_jump and the rotated
  Gaussian, against SparseCheb.
- T5 (later): Chebyshev TT-cross in a cell for the 5-D wave and the rotated Gaussian, checked on random
  fibres. Keep it only if it beats T4 there.

Chemistry benchmarks, in order:
1. Robertson (3 species, the standard stiff test [IV-22, IV-23, r]).
2. H2-air: about 9 species and 20 reactions (Li et al. 2004 [IV-21, r]); ISAT papers also use CH4 and GRI-Mech
   [r].
3. A primordial H/He/H2 network as in KROME [IV-16, v; MNRAS 439:2386 r] or GRACKLE (H, D, He, H2 chemistry
   and cooling) [IV-17, v]. The 6-species network of Ono-Sugimura (III-28) and osu2008 (CODES, III-24) give
   published surrogate baselines.

No use of ISAT in astrochemistry turned up in this search either.

### 5.4 Pitfalls

- **Checks that share the grid's blind spot.** Coefficient fits and random points both miss a kink at
  about 2π/n of a face's width (section 1). Look-ahead checks miss features finer than the next level. None
  is a bound (II-26).
- **Near-face slivers.** Any partition scheme with data-driven cuts must locate a cut to rounding level or
  cut at the midpoint. Anything in between manufactures hard cells.
- **Gates that read "unresolved" as "singular"** waste cuts on smooth functions and on curved interfaces.
- **Non-convex accuracy regions** give a tail of large errors in ISAT (III-14). Report the maximum and the
  share above the tolerance on held-out trajectories.
- **Query order and MPI.** Tables depend on the order of insertion. Fix the merge order, or results differ
  between ranks.
- **Transients** (ignition, thermal runaway, shock heating): the Jacobian's rank rises and the solver-call
  rate can spike. Measure the rate per phase, not only on average.
- **Kernel conditioning.** f-greedy picks clustered points. Use stabilised β-greedy [IV-12] and a fixed
  minimum power-function value.
- **Global rank and sparsity assumptions** (TT, Leja lower sets) hold in good coordinates. Raw species
  abundances are rarely good coordinates; log abundances or a local SVD basis are better.

### 5.5 References (new)

- [IV-1] Pachón, Platte, Trefethen 2010, *Piecewise-smooth chebfuns*, IMA J. Numer. Anal. 30(4):898–916, doi:10.1093/imanum/drp008; algorithm read in report NA-08-07, https://www.cs.ox.ac.uk/files/717/NA-08-07.pdf [v]
- [IV-2] Archibald, Gelb, Yoon 2005, *Polynomial fitting for edge detection in irregularly sampled signals and images*, SINUM 43(1):259–279, doi:10.1137/S0036142903435259 [a]
- [IV-3] Gelb, Tadmor 1999, *Detection of edges in spectral data*, Appl. Comput. Harmon. Anal. 7:101–135 [r]
- [IV-4] Xiu (PI), DOE final report DE-SC0011615, *Advanced dynamically adaptive algorithms for stochastic simulations on extreme scales*, https://www.osti.gov/servlets/purl/1345533 [v] (summarises II-29 and IV-5)
- [IV-5] Jakeman, Narayan, Xiu 2013, *Minimal multi-element stochastic collocation for uncertainty quantification of discontinuous functions*, JCP 242, doi:10.1016/j.jcp.2013.02.035 [a]
- [IV-6] Wilka, Lang 2024/2025, *Adaptive hp-polynomial based sparse grid collocation algorithms for piecewise smooth functions with kinks*, https://arxiv.org/abs/2404.02556 [a]
- [IV-7] Bungartz, Griebel 2004, *Sparse grids*, Acta Numerica 13:147–269 [r]
- [IV-8] Cohen, DeVore, Schwab 2010/2011, *Convergence rates of best N-term Galerkin approximations for a class of elliptic sPDEs*, FoCM 10; *Analytic regularity and polynomial approximation of parametric and stochastic elliptic PDEs*, Anal. Appl. 9 [a]
- [IV-9] Fuselier, Wright 2012, *Scattered data interpolation on embedded submanifolds with restricted positive definite kernels: Sobolev error estimates*, SINUM 50(3):1753–1776, https://arxiv.org/abs/1007.2825 [a]
- [IV-10] Santin, Haasdonk 2017, *Convergence rate of the data-independent P-greedy algorithm in kernel-based approximation*, Dolomites Res. Notes Approx. 10:68–78, https://arxiv.org/abs/1612.02672 [a]
- [IV-11] Wenzel, Santin, Haasdonk 2023, *Analysis of target data-dependent greedy kernel algorithms: convergence rates for f-, f·P- and f/P-greedy*, Constr. Approx. 57:45–74, doi:10.1007/s00365-022-09592-3; refined in Santin, Wenzel, Haasdonk, SINUM 2024, https://arxiv.org/abs/2307.09811 [a]
- [IV-12] Wenzel, Santin, Haasdonk 2021, *A novel class of stabilized greedy kernel approximation algorithms: convergence, stability and uniform point distribution*, J. Approx. Theory (venue [r]), https://arxiv.org/abs/1911.04352 [a]
- [IV-13] Wirtz, Haasdonk 2013, *A vectorial kernel orthogonal greedy algorithm*, Dolomites Res. Notes Approx. 6:83–100, doi:10.14658/PUPJ-DRNA-2013-Special_Issue-10 [a]
- [IV-14] Krieg, Pozharska, Ullrich, Ullrich 2024, *Sampling projections in the uniform norm*, https://arxiv.org/abs/2401.02220 [a]
- [IV-15] Fush, Bonilla, Schroeder, Yao, Mueller 2026, *Neural-ISAM: a hybrid in-situ machine learning approach for complex manifold-based combustion models in LES*, https://arxiv.org/abs/2605.10028 [a]
- [IV-16] Grassi et al. 2014, *KROME – a package to embed chemistry in astrophysical simulations*, MNRAS, https://arxiv.org/abs/1311.1070 [a]
- [IV-17] Smith et al. 2017, *Grackle: a chemistry and cooling library for astrophysics*, MNRAS 466:2217, https://arxiv.org/abs/1610.09591 [a]
- [IV-18] Ren, Liu, Lu, Lu, Oluwole, Goldin 2014, *The use of dynamic adaptive chemistry and tabulation in reactive flow simulations*, Combust. Flame 161(1):127–137, doi:10.1016/j.combustflame.2013.08.018 [a] (ISAT alone about 100x on a methane PaSR)
- [IV-19] Hiremath, Ren, Pope 2011, *Combined dimension reduction and tabulation strategy using ISAT-RCCE-GALI for the efficient implementation of combustion chemistry*, Combust. Flame 158:2113–2127 [a]
- [IV-20] Babuška, Guo 1986, *The h-p version of the finite element method, Parts 1 and 2*, Comput. Mech. 1 [r]
- [IV-21] Li, Zhao, Kazakov, Dryer 2004, *An updated comprehensive kinetic model of hydrogen combustion*, Int. J. Chem. Kinet. 36:566–575 [r]
- [IV-22] Robertson 1966, *The solution of a set of reaction rate equations*, in Numerical Analysis: An Introduction, Academic Press [r]
- [IV-23] Hairer, Wanner 1996, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, Springer [r]
- [IV-24] Davis, Skodje 1999, *Geometric investigation of low-dimensional manifolds in systems approaching equilibrium*, J. Chem. Phys. 111:859 [r]
- III-7 (Ma, Zabaras, ASGC): the DOI resolves to JCP 2009; volume 228, pages 3084–3113 [r].

---

## 6. Review V: ChebTT, high-rank structure, surrogates that learn structure, a chemistry oracle

Builds on docs/LITERATURE.md (I-* to IV-* cited by ID) and Review IV (section 5); new works are V-1 to V-49. Tags: **[v]**
source read (full text, named sections, or code), **[a]** abstract or catalogue only, **[r]** recalled. "Measured"
marks short runs made for the review (scripts not kept; the rates are in `src/structure/problems/ode/chemistry.py`).

### 6.0 Bottom line (ranked by gain over cost)

1. **Most of the overshoot comes from one rule, not from the doubling.** Both of ChebTT's degree tests measure the
   *upper half* of the coefficients (`chebyshev_tail` and `_truncated` drop degrees n/2+1..n). For geometric decay
   a_j ~ ρ^-j, the upper half is about a_{n/2}, but the interpolation error is about a_n ≈ a_{n/2}²/a_0. Requiring
   a_{n/2} ≤ share therefore gives an error near share²/max|f|, which squares the relative accuracy, and doubling
   adds up to another factor of ρ^-n. That fits worst errors of about 1e-6 of the tolerance (reviewer's derivation).
   Fix: estimate the tail from the last coefficients (a Chebfun-style plateau [III-1], or a fitted rate
   extrapolated past n), and chop each core after acceptance (§2.3). This needs no new evaluations.
2. **Replace "rank + 2 with random right indices" by global pivots from the worst check points**, as in TCI
   [V-3 §4.3.5]. Split each failing point σ into σ_≤k and σ_>k at every bond and append the parts to the index sets.
   This keeps nesting, costs nothing extra (the points are already evaluated) and aims the rank exactly where the
   model fails. Then add residual-driven (DMRG-greedy or rook) pivots inside the sweep [V-2 Alg. 2, 5]. The present
   scheme is the ALS-maxvol cross, which "can not update the ranks" [V-2 §3.3.1]; random enrichment is its weakest
   repair.
3. **Rotated or peaked positive f: approximate g = log(f + c) with c ≈ atol, and accept on f.** For the rotated
   Gaussian, log f = -xᵀPx is a quadratic: degree 2 per axis and TT ranks ≤ 2 + min(k, d−k) (at most 4 at d = 5).
   The needed accuracy in g is atol/(f + c), which is tight only where f ≫ c, and there g ≈ log f is exactly
   quadratic. This is the cheapest test in this memo. It also matches chemistry, where log abundances are the
   natural coordinates (§3).
4. **For chemistry, rank local tabulation above global TT.** Lorenz-96 at F = 8 has a flow-map Jacobian with no
   usable low rank, and its trouble is global coupling (§5, measured). Stiff chemistry contracts onto slow
   manifolds, which favours local models in local metrics (ISAT and AFINNI trust regions) with refusal. Use ChebTT
   as the leaf model in cells where the local rank is low, and on smooth box maps.
5. **Learned metrics are the "structure discovery" worth building**: a Hessian or Jacobian metric field per cell
   (AFINNI's `exact_hessian` split already computes |H|^{1/2}) used three ways: to shape trust regions as error
   ellipsoids, to rotate the cell's TT or kernel coordinates, and to refuse queries. Kernels on the learned metric
   (VKOGA, IV-13) come next. Manifold GPs, diffusion maps and neural operators give no sup-norm control at tight
   tolerances (§4).
6. **Build the chemistry oracle from Grackle's rate code** (pinned commit, §6): citable, numpy and scipy only.
   Start isothermal, then add T as a state.

### 6.1 What ChebTT does, against the literature

ChebTT is the ALS maxvol cross of Oseledets-Tyrtyshnikov [II-10], which Dolgov and Savostyanov classify as Alg. 3
[V-2 §3.3.1]: sweeps update the index sets by maxvol at fixed ranks. Rank adaptivity in the literature comes from
three mechanisms.

- **DMRG-maxvol** [V-1, a; V-2 Alg. 4, v]: compute the two-site superblock in full, truncate its SVD at ε, take
  maxvol of the factors. Rank-revealing, but O(d n² r²) evaluations: expensive at n = 64-512.
- **DMRG-greedy** [II-12, now read; V-2 Alg. 5]: add one cross per bond per half-sweep. The pivot is the
  largest-error entry of the superblock, found by random sampling followed by rook pivoting (alternating row and
  column maxima until the pivot dominates its row and column) [V-2 Alg. 2]. Nesting holds at every step, and the
  cost is O((m + n) r) entries per rank-one step. The authors call it heuristic: "its accuracy can not be guaranteed
  in general" [V-2 Remark 2].
- **AMEn-cross** (TT-Toolbox `amen_cross.m` [V-9, v]; theory in V-8 [a]): keep a rank-5 (`zrank`) TT
  approximation Z of the residual f − f̃, enrich each core with `kickrank = 2` directions from Z plus `zrank/2`
  random ones, and stop when the Frobenius change of every supercore between consecutive sweeps (`max_dx`) falls
  below tol, or after the error has risen twice (`max_err_jumps = 2`). The stopping rule is a convergence test,
  not an error bound.
- **TCI / prrLU** [V-3, v]: 2-site updates by partial rank-revealing LU, with the aim ‖F − F̃‖∞ < τ. The error
  estimate is the largest pivot error on the 2-site slices, normalised by the largest sampled |F|
  (`normalizeerror = true`). It must stay below τ for 3 consecutive iterations (`ncheckhistory = 3`). TCI is exact
  on the slices it built from, and nothing more [V-3 §4.3].

**Sup-norm reliability.** The best bound is on the grid only: with exact maximum-volume sets,
|A − Ã| ≤ (2r + κr + 1)^⌈log₂ d⌉ (r + 1)² E_C, where E_C is the best rank-r error in the max norm [II-12, Thm. 1,
v]. The measured factor on random tensors was about 2^3.2 for the greedy cross and 2^1.3 after DMRG sweeps. Error
bounds for the whole tensor appear only recently [V-17, a]. Nothing controls the error between grid points.
ChebTT's three check sets (Sobol, free cached points, next-grid look-ahead) are a stronger estimator than TCI's
in-sample pivot error, so keep them.

**Narrow peaks: the "ergodicity" problem.** TCI names "multivariate functions with very narrow peaks" as a case
where the local updates stay in one part of the space [V-3 §4.3.6]. Every such case its authors met was fixed by
**global pivots**: user-proposed points split at every bond and appended to the index sets, then a prrLU on the
pivot matrices to drop spurious ones, then a few reset sweeps [V-3 §4.3.5]. They suggest a separate global search
for points of large |F − F̃| that proposes pivots in a loop. The maxvol property makes such a search cheap: the
largest element of a maximum-volume r×r submatrix is at least |A|_max / r² [V-7, r; quoted in V-5 §2.3, v]. TTOpt
[V-5] and optima_tt [V-6, a] turn this into optimisers, with up to d = 100 in V-6. For us: run a low-rank maxvol
cross on the residual |f − f̃| (mapped through atan as in TTOpt eq. 7) as the global-pivot finder. It costs
O(d n R²) evaluations per sweep, and each evaluation is also a check point.

### 6.2 Recommendations for ChebTT (cheap, in order)

**2.1 Global pivots from failures.** After each failed round, take the k worst points among the Sobol set, the
free checks and the look-ahead, snap them to the current grid (the look-ahead points lie on the next grid, so
snap after doubling), and add them as global pivots. Keep "largest |f| seen" as one. Expected: rank grows only
where needed, and the Gaussian's peak is caught as soon as one check point lands near it. Test: the rotated
Gaussian and Lorenz-96 F = 8 at 5-D; compare evaluations and final ranks with rank + 2.

**2.2 Rank-revealing truncation.** Add TCI's rule: drop pivots whose prrLU pivot falls below τ_loc ≈ τ, so a rank
that was overshot shrinks. Use a QR with column pivoting of the fibre matrix in place of plain QR plus maxvol, and
truncate at τ/√d relative to max|f|.

**2.3 Degrees: estimate, chop, round.**
(a) Tail estimate: in `_fibre_degree` and in the degree-or-rank test, replace "upper half of the degrees" by the
mean of the last 4 coefficients times a fitted ρ/(1 − ρ), and fall back to the upper half when the decay is not
geometric (a kink, see review_1).
(b) Post-hoc chop: for each core, keep the smallest degree n'_k at which dropping coefficients above n'_k changes
the model by at most share at all check points. This needs no evaluations and makes inference faster.
(c) Optional growth by 1.5 instead of 2. Second-kind points of degree n are nested in degree m·n for every integer
m, so with `LATTICE = 3·2^12` the chain 8 → 12 → 24 → 48 is nested except at the first step (the even nodes are
shared).
(d) TT rounding after the cross (SVD at ε/√(d−1)) cuts rank and inference cost. It is safe only if checked again
at the check points.

**2.4 Kinks along axes (Genz continuous).** f = exp(−Σ a_i |x_i − u_i|) has TT rank 1, so the whole cost is
degree. Use piecewise-Chebyshev univariate bases, split at kinks found on a fibre by chebfun's edge detection
[IV-1]. This is the C3 function train with piecewise polynomial cores [II-14]. The failure at tol 0.02 suggests
early acceptance at a coarse degree whose check points miss the kink (mechanism of review_1 §1; unmeasured).

### 6.3 High rank from rotated or coupled structure

**Why the rotated Gaussian fails.** Rohrbach, Dolgov, Grasedyck and Scheichl [V-10, v, Thm. 3.1] bound the
functional TT ranks of exp(−xᵀΓx/2) by (1 + 7(σ/λ_min) log(·/ε))^l, where l is the rank of the sub-diagonal block
of Γ at that bond, σ its largest singular value and λ_min Γ's smallest eigenvalue. The rank is exponential in l
and grows with the condition number of the precision matrix. With a dense rotation Q, l = min(k, d − k) and σ/λ_min
approaches 125: the format's worst case. The ranks also depend on the variable ordering [V-10 §3], but no ordering
helps a dense Q.

**Remedies, ranked for our goal:**
1. *TT of a transform.* log(f + c) (§0.3). I found no TT paper that uses "TT of log f, then exp" for pointwise
   surrogates. The TT density literature works with √f, because it keeps the approximation non-negative and allows
   sampling [V-14, a; V-15, a]. A compressed exp of a TT is a hard problem of its own [V-16, a], but we never need
   it: f̂ = exp(g̃) − c is evaluated pointwise at the same O(d n r²) cost. Pitfalls: f must be positive (wave and
   oscillatory Genz cannot use it); f underflows where xᵀPx > 745, which is why the shift c is needed; a mixture of
   peaks makes log-sum-exp kinked between peaks; and the cross must pick pivots by the weighted residual
   (f + c)|g̃ − g| while acceptance stays on f.
2. *Local rotations inside cells.* A cell's Hessian metric (AFINNI's `exact_hessian` or the gradient-LSQ split)
   gives its rotation; the cell's TT works in rotated coordinates. This is a local version of coordinate flows
   [V-11, a], which finds a global linear map by Riemannian gradient descent and notes that a rotation can turn a
   rotated high-rank Gaussian back into rank one. A rotated box sticks out of the domain: restrict the TT to the
   rotated box inscribed in the cell, cover the rest with child cells, and never evaluate f outside the domain
   (Lotka-Volterra returns NaN there).
3. *A Hessian at the peak, not gradient covariance at random points.* The active subspace failed because the peak
   fills about 1e-4 of the box. Find the peak first (largest |f| seen, then a TTOpt search), estimate the Hessian
   there by finite differences (O(d²) evaluations) and rotate. This helps one peak, not a field of features.
4. *Tree formats.* HT with an adapted tree, and tree search by node swaps [V-12, a; black-box HT V-13, r], help
   when the coupling is sparse but badly ordered. They do not help a dense rotation.
5. *Quantics TT* [V-4, a; V-18, r]: very fine resolution in 1-3 dimensions; an option for sharp 1-D features only.

### 6.4 Surrogates that learn structure: metrics, kernels, ridges, networks

"Structure" is what a method learns; the guarantee is the best known sup-norm statement; d is the input
dimension, m the intrinsic one.

| Family | Structure discovered | Best sup-norm guarantee | Samples vs d | Inference | Fit to our goal |
|---|---|---|---|---|---|
| Metric-based (Riemannian) adaptation: local models in error ellipsoids | local anisotropy, through the Hessian or Jacobian metric | local: P1 error on a cell ≈ max over edges of eᵀ|H|e [I-15, r]; cells needed ~ ε^(−m/2)∫√det|H|_m | ε^(−m/2), m = number of non-negligible Hessian eigenvalues; ε^(−d/2) with no low-rank Hessian | tree lookup plus a local model | **high** (this is ISAT/AFINNI) |
| ChebTT (fixed as in §2) | global separable and low-rank coupling | on the grid only, quasi-optimal [II-12] | d n r² per sweep | d n r² | high on smooth box maps; low on chaotic maps or rotated peaks without §3 |
| Greedy kernels (VKOGA, f/P-greedy) with an ARD or AGOP metric | anisotropy, via learned length scales or a gradient outer product | \|f − s\|(x) ≤ P_X(x)·‖f‖_H; ‖f‖_H unknown [IV-11, IV-13; V-25, V-27, r] | rates set by m on a manifold [IV-9] | O(N d) per query | medium: good at 1e-2–1e-4, limited by conditioning at tight tolerances [V-26, r] |
| GPs on manifolds or graphs (Matérn via Laplace-Beltrami or graph Laplacian) | the data manifold | none in sup norm; posterior variance as an indicator | needs the manifold's spectrum, or a graph built on the visited states | O(N) per query | low: built for uncertainty, not tight tolerances [V-19, V-20, a] |
| Diffusion maps plus geometric harmonics | intrinsic coordinates of the visited states | none; Nyström extension degrades off the data | graph on N states | O(N) per query | coordinates only [V-21, a] |
| MLS on manifolds (MMLS) | local tangent charts | O(h^(m+1)) for noiseless samples [V-23, a] | depends on m, not d | local least squares per query | medium: a chart-based leaf model; slow per query [V-22, V-23, a] |
| Multi-index or ridge learning (GOP, AGOP, kernel active subspaces) | k leading directions (global or local) | uniform recovery w.h.p. for exact ridges [II-18, II-19] | polynomial in d | projection plus a low-dimensional model | high as a coordinate provider for rows 1–3 [V-28, V-29, V-30, a] |
| Neural: DeepONet, FNO, KAN, neural-ODE flow maps | learned features | none in practice: sup-norm sampling is hard for network approximation spaces [V-37, a]; trained networks fall short of their theoretical rates [V-36, a] | large | fast on GPU | low, except as a compressor behind a verified table [V-32–V-35, a] |
| Conformal refusal (any model) | none | coverage of a quantile, marginal over the distribution, exchangeability required [V-38, V-39, a]; covariate shift needs a known likelihood ratio [V-40, a] | – | – | an audit tool, not a max-error control |

**(a) Metric-based adaptation and AFINNI.** The continuous-mesh framework [II-25; review V-31, a] builds a metric
field M(x) = |H(x)| scaled by the tolerance and makes cells of unit size in it. For a linear model the error over a
displacement e is about ½eᵀHe, so the ellipsoid of accuracy is eᵀ(|H|/2ε)e ≤ 1. That is ISAT's EOA [III-12] and the
shape AFINNI's Hessian-metric splits already use. AFINNI's trust matrix is different: it is a *coverage* ellipsoid
(scaled inverse covariance of the cluster's points), and its own docstring says it is "not an error certificate".
**Recommendation:** intersect the coverage ellipsoid with the error ellipsoid from the cell's metric. The outside
flag then becomes an error-based refusal, which is ISAT's contract and the one the MHD coupling needs.

Does a learned metric field give worst-error control at d = 10–100? Only if it is (i) correct and (ii) nearly
low-rank. With m Hessian eigenvalues above the tolerance's scale, the cell count scales like ε^(−m/2). The ellipsoids
become long in flat directions, and only third derivatives and the domain bound them, so the curse moves from d to
m. Correctness is the weak part: a Hessian estimated from samples is a heuristic, and II-26 rules out certificates.
The control is per cell and empirical (look-ahead checks along the ellipsoid's axes, review_1 §3.4).

**(b) Kernels with learned metrics.** ARD length scales, Mahalanobis or AGOP metrics [V-29], and deep kernels
[V-24, a] all learn anisotropy, but only the power function gives a pointwise statement. For noiseless data the GP
posterior standard deviation equals the power function [V-25, r], so |f − s|(x) ≤ P_X(x)·‖f‖_H. Learning the metric
changes H, and no bound on ‖f‖_H survives that. At tight tolerances the kernel matrix grows ill-conditioned, the
trade-off principle [V-26, r]. Practical limit (reviewer's estimate): relative errors of 1e-4 to 1e-6 need
stabilised greedy selection [IV-12] and O(N³) work per cell. Kernels belong in cells, on the visited manifold.

**(c) Ridge and multi-index learning.** For chemistry, the Jacobian of the step map, which ISAT computes anyway, is
the local gradient outer product. Its leading right singular vectors are the local active subspace, with no
sampling of gradients (review_1 §3). Wu et al. [V-28] give convergence rates for gradient-based direction
estimates. Kernel active subspaces [V-30] are a nonlinear version with L2 targets only.

**(d) Neural models** offer speed, not control: use them only to compress mature table regions (IV-15), behind
refusal and fallback, and report the maximum apart from any conformal quantile.

**Cheap tests on the 5-D bench (each under a few minutes):**
- *Rotated Gaussian:* (1) ChebTT on log(f + atol) (expect hundreds of evaluations); (2) global pivots from worst
  checks (§2.1); (3) VKOGA with the metric P (the true Hessian/2) against an isotropic kernel at 5k points, to see
  what a perfect learned metric buys; (4) the AGOP metric from 200 random gradients, which should fail like the
  active subspace did.
- *Genz continuous:* piecewise-Chebyshev cores with kink detection (§2.4) against SparseCheb; a kernel with Wendland
  C0 functions as a control.
- *Lorenz-96 (F = 8, t = 0.5):* a full TT-SVD of the map sampled on a 9^5 grid (59k RK4 evaluations, vectorised in
  `problems/functions.py`) gives the true unfolding singular values, and so whether rank or degree is the obstacle (§5).
  Then train AFINNI and a kernel on attractor trajectories against the box.
- *Lotka-Volterra:* TT in Hessian-rotated cells (§3.2), checking that no evaluation leaves the domain.

**Ranking against ChebTT for our goal** (worst error over visited states, high d, chemistry): (1) local tabulation
in learned metrics with refusal (AFINNI/ISAT plus 4a); (2) ChebTT, fixed, as a box method and as a leaf model where
the local rank is low; (3) greedy kernels in Jacobian metrics on the visited manifold; (4) ridge and AGOP learning
as the coordinate provider for 1-3; (5) MMLS charts and diffusion-map coordinates for analysis; (6) manifold GPs and
neural surrogates, only behind refusal.

### 6.5 Flow maps and chemistry maps: what rank to expect

**Measured (Lorenz-96).** On the attractor, the Lyapunov spectra are: d = 5, F = 8: (0.53, −0.005, −0.57, −1.2,
−3.7); d = 10, F = 8: (1.26, 0.74, 0.12, …, −4.7); F = 1 has none positive. On the bench's box [−5, 5]^d, the
gradient of the output x_1(t) has median, 95% and maximum norms of 1.2, 2.9 and 4.0 (d = 5, F = 8, t = 0.5), against
0.96, 1.09 and 1.30 (F = 1, t = 0.1). So the chaotic map is not steep at t = 0.5. Its difficulty is coupling and
degree. A quadratic right-hand side composed over about 2–3 nonlinear times (|x| ~ 5) raises the polynomial degree
geometrically and spreads each output's dependence over all 5 variables, because the cyclic stencil spans the
whole ring at d = 5. For small t, Φ_t ≈ x + t f(x), whose TT rank is that of the right-hand side (3–4 here, local
in the index). Rank grows with t as the coupling spreads. The median finite-time Jacobian singular values at
F = 8, t = 0.5 run from 27 to 0.008 (d = 5) and from 27 to 0.003 (d = 10). Only the last one or two are negligible,
so the map is not locally a low-dimensional ridge.

**Chemistry differs through dissipation.** Over a step longer than the fast time scales, fast modes contract by
exp(λΔt), so the step map's Jacobian has numerical rank near the number of slow modes: the ILDM/CSP premise
[III-18, III-19], the reason ISAT's EOAs are long in fast directions [III-12], and the argument of Pope's
"manifold challenges" review [V-49, r]. Implications: (i) local tabulation on visited states matches this
structure and global TT does not; (ii) conservation is linear in number densities but a sum of exponentials in log
densities, which has high TT rank. **Measure T1 of review_1 on the primordial network before choosing
coordinates.** TT models of the right-hand side [V-41, a] model f, not Φ_t; rank-adaptive tensor integrators
[V-42, a] confirm that rank must be adapted in time.

### 6.6 Chemistry oracles (numpy and scipy only)

#### 6.6.1 Robertson [IV-22], verified against the IVP test set's `rober.f` [V-46, v]

y1' = −0.04 y1 + 1e4 y2 y3;  y2' = 0.04 y1 − 1e4 y2 y3 − 3e7 y2²;  y3' = 3e7 y2².  y(0) = (1, 0, 0); test interval
t ∈ [0, 1e11]. Rates are fixed (no temperature). Use Radau or BDF with rtol 1e-10 and atol 1e-20 for the
reference, inputs (log y1, log y2, log y3, log Δt), and y1 + y2 + y3 = 1 as the conservation check.

#### 6.6.2 Primordial H/He/H2: nine species (H, H+, H−, H2, H2+, e−, He, He+, He++)

**Source** [V-44, v]: Grackle, `src/clib/rate_functions.c` at commit 48e0733 (2026-02-05),
https://raw.githubusercontent.com/grackle-project/grackle/48e0733d079f216f083ea7715db5faf60d284509/src/clib/rate_functions.c.
Reaction identities come from the species updates in `src/clib/solve_rate_cool_g.F` (`step_rate_g`, lines
2225–2310), because **the comments in rate_functions.c mislabel k16–k19**. The updates show k16 and k17 acting on
H−·H+, and k18 and k19 on H2+. Grackle's defaults are used: `CaseBRecombination = 0`, `three_body_rate = 0`,
`h2_charge_exchange_rate = 1`, tiny = 1e-20 (`grackle_macros.h`). Units are cm³ s⁻¹ (two-body) and cm⁶ s⁻¹
(three-body). T is in K; T_eV = T/11605; ln T_eV is the natural log. Most fits derive from Abel et al. 1997
[V-43, a], whose model covers n < 1e4 cm⁻³ and 1 K < T < 1e8 K. Every row below was transcribed from the pinned
source and evaluated (now `primordial_rates` in `src/structure/problems/ode/chemistry.py`). As a check, k2(1e4 K) = 4.17e-13, the standard case-A value. **All
rows: verified against source.**

| k | Reaction | k(T) (cgs) | Source lines |
|---|---|---|---|
| k1 | H + e → H+ + 2e | exp(Σ c_i (ln T_eV)^i), c = (−32.71396786375, 13.53655609057, −5.739328757388, 1.563154982022, −0.2877056004391, 0.03482559773736999, −0.00263197617559, 1.119543953861e-4, −2.039149852002e-6); max(tiny, ·) if T_eV ≤ 0.8 | 36–54 |
| k2 | H+ + e → H + γ | T > 5500: exp(Σ c_i (ln T_eV)^i), c = (−28.61303380689232, −0.7241125657826851, −0.02026044731984691, −0.002380861877349834, −3.212605213188796e-4, −1.421502914054107e-5, 4.989108920299513e-6, 5.755614137575758e-7, −1.856767039775261e-8, −3.071135243196595e-9); else **k4(T)** (as in the source) | 97–127 |
| k3 | He + e → He+ + 2e | T_eV > 0.8: exp(Σ c_i (ln T_eV)^i), c = (−44.09864886561001, 23.91596563469, −10.75323019821, 3.058038757198, −0.5685118909884001, 0.06795391233790001, −0.005009056101857001, 2.067236157507e-4, −3.649161410833e-6); else tiny | 56–75 |
| k4 | He+ + e → He + γ | T_eV > 0.8: 1.54e-9(1 + 0.3/exp(8.099328789667/T_eV))/(exp(40.49664394833662/T_eV) T_eV^1.5) + 3.92e-13/T_eV^0.6353; else 3.92e-13/T_eV^0.6353 | 77–95 |
| k5 | He+ + e → He++ + 2e | T_eV > 0.8: exp(Σ c_i (ln T_eV)^i), c = (−68.71040990212001, 43.93347632635, −18.48066993568, 4.701626486759002, −0.7692466334492, 0.08113042097303, −0.005324020628287001, 1.975705312221e-4, −3.165581065665e-6); else tiny | 129–150 |
| k6 | He++ + e → He+ + γ | 3.36e-10 T^−1/2 (T/1e3)^−0.2 / (1 + (T/1e6)^0.7) | 152–169 |
| k7 | H + e → H− + γ | 3.0e-16 (T/300)^0.95 exp(−T/9320) | 171–176 |
| k8 | H + H− → H2 + e | 1.35e-9 (T^0.098493 + 0.32852 T^0.5561 + 2.771e-7 T^2.1826) / (1 + 6.191e-3 T^1.0461 + 8.9712e-11 T^3.0424 + 3.2576e-14 T^3.7741) | 178–185 |
| k9 | H + H+ → H2+ + γ | T < 30: 2.10e-20 (T/30)^−0.15; else 10^(−18.20 − 3.194 L + 1.786 L² − 0.2072 L³), L = log₁₀ min(T, 3.2e4) | 187–203 |
| k10 | H2+ + H → H2 + H+ | 6.0e-10 | 205–209 |
| k11 | H2 + H+ → H2+ + H | T_eV > 0.3: exp(−21237.15/T) Σ c_i (ln T)^i, c = (−3.3232183e-7, 3.3735382e-7, −1.4491368e-7, 3.4172805e-8, −4.7813720e-9, 3.9731542e-10, −1.8171411e-11, 3.5311932e-13) (Savin 2004 option); else tiny | 211–250 |
| k12 | H2 + e → 2H + e | T_eV > 0.3: 4.4886e-9 T^0.109127 exp(−101858/T); else tiny | 252–265 |
| k13 | H2 + H → 3H | T_eV > 0.3: 1.0670825e-10 T_eV^2.012 / (exp(4.463/T_eV)(1 + 0.2472 T_eV)^3.512); else tiny. Grackle's solver uses the density-dependent k13dd fit (lines 325–446); this low-density form is the minimal choice | 267–323 |
| k14 | H− + e → H + 2e | T_eV > 0.04: exp(Σ c_i (ln T_eV)^i), c = (−18.01849334273, 2.360852208681, −0.2827443061704, 0.01623316639567, −0.03365012031362999, 0.01178329782711, −0.001656194699504, 1.068275202678e-4, −2.631285809207e-6); else tiny | 448–469 |
| k15 | H− + H → 2H + e | T_eV > 0.1: exp(Σ c_i (ln T_eV)^i), c = (−20.37260896533324, 1.139449335841631, −0.1421013521554148, 0.00846445538663, −0.0014327641212992, 2.012250284791e-4, 8.66396324309e-5, −2.585009680264e-5, 2.4555011970392e-6, −8.06838246118e-8); else 2.56e-9 T_eV^1.78186 | 471–493 |
| k16 | H− + H+ → 2H | 2.4e-6 (1 + T/2e4)/√T | 495–500 |
| k17 | H− + H+ → H2+ + e | T > 1e4: 4.0e-4 T^−1.4 exp(−15100/T); else 1.0e-8 T^−0.4 | 502–512 |
| k18 | H2+ + e → 2H | T > 617: 1.32e-6 T^−0.76; else 1.0e-8 | 514–524 |
| k19 | H2+ + H− → H2 + H | 5.0e-7 √(100/T) | 526–530 |
| k22 | 3H → H2 + H | T ≤ 300: 1.3e-32 (T/300)^−0.38; else 1.3e-32 (T/300)^−1 (cm⁶ s⁻¹) | 543–588 |
| k57 | H + H → H+ + e + H | T > 3000: 1.2e-17 T^1.2 exp(−1.578e5/T); else tiny | 678–688 |
| k58 | H + He → H+ + e + He | T > 3000: 1.75e-17 T^1.3 exp(−1.578e5/T); else tiny | 690–700 |

For robustness tests, Glover and Abel [V-47, a] assess the uncertain H2 rates and give alternatives. Rate
equations use number densities, as in `step_rate_g` (where Grackle's mass-density fields carry /2 for H2 and /4
for He). For example, d n_H2/dt = k8 n_H n_H− + k10 n_H2+ n_H + k19 n_H2+ n_H− + k22 n_H³ − (k11 n_H+ + k12 n_e +
k13 n_H) n_H2. Grackle sets H2+ to its equilibrium value; the oracle should integrate it as a species. Radiative
rates (k24–k31) and dust are left out.

**State and coordinates.** Stage A (isothermal): inputs are log n_H ∈ [0, 6] (n in cm⁻³), log T ∈ [1, 4], the log
fractions x_i = n_i/n_H of 6 independent species, and log Δt. Nuclei and charge conservation fix the other three
(He/H = 0.079 by number for Grackle's default hydrogen mass fraction 0.76 [r]). That is 8 inputs.
Stage B adds T as a state with a cooling function (Grackle's ceHI and related rates, lines 755 onward), which is
the stiff, transient-rich case. Outputs: log x_i after Δt, floored at 1e-20.

**Ranges used in astrophysics.** Abel et al.'s model covers n < 1e4 and 1–1e8 K [V-43]. A primordial emulator
spans n = 1e-3–1e18 cm⁻³ and is checked in one-zone collapses with steps down to 1e-4 t_ff [III-28, a]. The
free-fall time is t_ff = √(3π/32Gρ) ≈ 47 Myr · n^−1/2 for μ = 1.22 (computed). Reasonable choices are Δt ∈ [1e-3,
1e-1] t_ff(n), or a fixed Δt ∈ [1e2, 1e5] yr for ISM runs. Initial ionisation fractions are 1e-6–1e-1 and H2
fractions 1e-8–0.5 (reviewer's choice, to cover the relic and shock-ionised cases).

#### 6.6.3 H2–O2 (optional): Cantera `h2o2.yaml` at commit 5876366 [V-45, v]

This is the hydrogen-oxygen submechanism of GRI-Mech 3.0 ("Modified from the original to include N2"), not Li et
al. 2004 [IV-21] or Burke et al. 2012 [V-48, r]. It has 10 species (H2, H, O, O2, OH, H2O, HO2, H2O2, AR, N2) and 29
reversible reactions. k = A T^b exp(−Ea/RT), with A in cm-mol-s and Ea in cal/mol. Reverse rates need the NASA-7
polynomials in the same file (equilibrium constants), which is the main work without Cantera. **All rows: verified
against source.**

| # | Reaction | A | b | Ea | Notes |
|---|---|---|---|---|---|
| 1 | 2O + M ⇌ O2 + M | 1.2e17 | −1.0 | 0 | eff. H2 2.4, H2O 15.4, AR 0.83 |
| 2 | O + H + M ⇌ OH + M | 5.0e17 | −1.0 | 0 | H2 2.0, H2O 6.0, AR 0.7 |
| 3 | O + H2 ⇌ H + OH | 3.87e4 | 2.7 | 6260 | |
| 4 | O + HO2 ⇌ OH + O2 | 2.0e13 | 0 | 0 | |
| 5 | O + H2O2 ⇌ OH + HO2 | 9.63e6 | 2.0 | 4000 | |
| 6 | H + O2 + M ⇌ HO2 + M | 2.8e18 | −0.86 | 0 | O2, H2O, N2, AR 0 (explicit in 7–10) |
| 7 | H + 2O2 ⇌ HO2 + O2 | 2.08e19 | −1.24 | 0 | |
| 8 | H + O2 + H2O ⇌ HO2 + H2O | 1.126e19 | −0.76 | 0 | |
| 9 | H + O2 + N2 ⇌ HO2 + N2 | 2.6e19 | −1.24 | 0 | |
| 10 | H + O2 + AR ⇌ HO2 + AR | 7.0e17 | −0.8 | 0 | |
| 11 | H + O2 ⇌ O + OH | 2.65e16 | −0.6707 | 17041 | |
| 12 | 2H + M ⇌ H2 + M | 1.0e18 | −1.0 | 0 | H2 0, H2O 0, AR 0.63 |
| 13 | 2H + H2 ⇌ 2H2 | 9.0e16 | −0.6 | 0 | |
| 14 | 2H + H2O ⇌ H2 + H2O | 6.0e19 | −1.25 | 0 | |
| 15 | H + OH + M ⇌ H2O + M | 2.2e22 | −2.0 | 0 | H2 0.73, H2O 3.65, AR 0.38 |
| 16 | H + HO2 ⇌ O + H2O | 3.97e12 | 0 | 671 | |
| 17 | H + HO2 ⇌ O2 + H2 | 4.48e13 | 0 | 1068 | |
| 18 | H + HO2 ⇌ 2OH | 8.4e13 | 0 | 635 | |
| 19 | H + H2O2 ⇌ HO2 + H2 | 1.21e7 | 2.0 | 5200 | |
| 20 | H + H2O2 ⇌ OH + H2O | 1.0e13 | 0 | 3600 | |
| 21 | OH + H2 ⇌ H + H2O | 2.16e8 | 1.51 | 3430 | |
| 22 | 2OH (+M) ⇌ H2O2 (+M) | k∞: 7.4e13, −0.37, 0; k0: 2.3e18, −0.9, −1700 | | | Troe A 0.7346, T3 94, T1 1756, T2 5182; H2 2.0, H2O 6.0, AR 0.7 |
| 23 | 2OH ⇌ O + H2O | 3.57e4 | 2.4 | −2110 | |
| 24 | OH + HO2 ⇌ O2 + H2O | 1.45e13 | 0 | −500 | duplicate of 29 |
| 25 | OH + H2O2 ⇌ HO2 + H2O | 2.0e12 | 0 | 427 | duplicate of 26 |
| 26 | OH + H2O2 ⇌ HO2 + H2O | 1.7e18 | 0 | 29410 | duplicate |
| 27 | 2HO2 ⇌ O2 + H2O2 | 1.3e11 | 0 | −1630 | duplicate of 28 |
| 28 | 2HO2 ⇌ O2 + H2O2 | 4.2e14 | 0 | 12000 | duplicate |
| 29 | OH + HO2 ⇌ O2 + H2O | 5.0e15 | 0 | 17330 | duplicate |

Typical conditions (reviewer's estimate, unverified): T₀ 900–1500 K, 1–10 atm, equivalence ratio 0.5–2, Δt 1e-7–1e-5 s.

### 6.7 Benchmark protocol for the chemistry stage

**What others report.** ISAT controls the local tabulation error ε (scaled), and its error checking bounds the 90%
quantile of the incurred error, not the maximum [III-13]. Rare large errors come from non-convex regions of
accuracy [III-14]. Astrochemistry emulators report mean or quantile relative errors: "below 10% in over 90% of
cases" [III-28], mean relative error below 4% [III-24], about 1e-2 [III-27]. None reports a maximum on held-out
trajectories with a fallback rate.

**Protocol (concrete):**
1. *Data.* Train from one-zone trajectories: isothermal and free-fall collapse (ρ̇ = ρ/t_ff), with initial states
   sampled from §6.2's ranges by a Sobol design (200 trajectories, about 1e5 states). Test on 100 held-out
   trajectories with different seeds and on 20 trajectories from a different forcing (shock-heated T jumps).
   Contrast with training on the box of §6.2 at the same budget.
2. *Error.* e = max over species with x_i > 1e-12 of |log₁₀ x̂_i − log₁₀ x_i|, plus the conservation residuals of
   H nuclei, He nuclei and charge. Report the maximum, the 99.9% and 99% quantiles, and the share above the
   tolerance (0.01 dex and 0.001 dex). Report them per phase (quiescent, ignition or collapse, transient), as
   review_1 §4 advised.
3. *Fallback.* Report the refused-query rate overall and per phase, and the error of accepted answers only. Re-solve
   a random 1% of accepted answers, as in Lu-Pope error checking [III-13], and report both the maxima it finds and
   the ones it misses.
4. *Accumulation.* Apply the surrogate repeatedly along 20 test trajectories (1000 steps) and report the drift of
   the log abundances and of the temperature (stage B) against the solver, as in III-28's one-zone tests.
5. *Cost.* Solver evaluations to train, table size, and inference time per query against a BDF solve at the same
   accuracy, on CPU and GPU.

### 6.8 Cheapest first experiments

In order: (1) ChebTT on log(f + atol) for the rotated Gaussian; (2) the last-coefficient tail and the chop on the
26 passing 5-D cases (target worst/tolerance 0.1–0.5); (3) global pivots in place of rank + 2; (4) a TT-SVD of
Lorenz-96 F = 8 on 9^5 points; (5) Robertson and stage-A primordial oracles with review_1's T1 in linear and log
coordinates; (6) AFINNI with error ellipsoids against coverage ellipsoids, under the §7 protocol.

Status (section 1): (2) done, as the top-quarter tail (1.13) and the chop in the compression (1.14);
(3) tried and rejected (1.13); (5) done (1.11, 1.12).

### 6.9 References (new: V-*)

Tensors and cross approximation
- V-1 Savostyanov, Oseledets 2011, *Fast adaptive interpolation of multi-dimensional arrays in tensor train format*, nDS'11, doi:10.1109/nDS.2011.6076873 [a]
- V-2 Dolgov, Savostyanov 2020, *Parallel cross interpolation for high-precision calculation of high-dimensional integrals*, CPC 246:106869, https://arxiv.org/abs/1903.11554 [v: §2–3.3]
- V-3 Núñez Fernández et al. 2025, *Learning tensor networks with tensor cross interpolation: new algorithms and libraries*, SciPost Phys. 18:104, https://arxiv.org/abs/2407.02454 [v: §4.3, 4.3.5–4.3.7, the crossinterpolate2 API]
- V-4 Ritter et al. 2024, *Quantics tensor cross interpolation…*, PRL 132:056501, https://arxiv.org/abs/2303.11819 [a]
- V-5 Sozykin et al. 2022, *TTOpt: a maximum volume quantized tensor train-based optimization…*, NeurIPS, https://arxiv.org/abs/2205.00293 [v: §2.3]
- V-6 Chertkov, Ryzhakov, Novikov, Oseledets 2022, *Optimization of functions given in the tensor train format*, https://arxiv.org/abs/2209.14808 [a]
- V-7 Goreinov, Oseledets, Savostyanov, Tyrtyshnikov, Zamarashkin 2010, *How to find a good submatrix*, in Matrix Methods: Theory, Algorithms and Applications, World Scientific [r; its bound is quoted from V-5]
- V-8 Dolgov, Savostyanov 2014, *Alternating minimal energy methods for linear systems in higher dimensions*, SISC 36(5):A2248, https://arxiv.org/abs/1301.6068, https://arxiv.org/abs/1304.1222 [a]
- V-9 TT-Toolbox, `cross/amen_cross.m`, https://github.com/oseledets/TT-Toolbox [v: options and stopping rule]
- V-10 Rohrbach, Dolgov, Grasedyck, Scheichl 2022, *Rank bounds for approximating Gaussian densities in the tensor-train format*, SIAM/ASA JUQ 10(3), https://arxiv.org/abs/2001.08187 [v: Thm. 3.1, §3]
- V-11 Dektor, Venturi 2022, *Tensor rank reduction via coordinate flows*, https://arxiv.org/abs/2207.11955 [a]
- V-12 Grelier, Nouy, Chevreuil 2018, *Learning with tree-based tensor formats*, https://arxiv.org/abs/1811.04455 [a]
- V-13 Ballani, Grasedyck, Kluge 2013, *Black box approximation of tensors in hierarchical Tucker format*, LAA 438 [r]
- V-14 Dolgov, Anaya-Izquierdo, Fox, Scheichl 2020, *Approximation and sampling of multivariate probability distributions in the tensor train decomposition*, Stat. Comput. 30:603, https://arxiv.org/abs/1810.01212 [a]
- V-15 Cui, Dolgov 2022, *Deep composition of tensor-trains using squared inverse Rosenblatt transports*, FoCM, https://arxiv.org/abs/2007.06968 [a]
- V-16 Eigel, Farchmin, Heidenreich, Trunschke 2021, *Efficient approximation of high-dimensional exponentials by tensor networks*, https://arxiv.org/abs/2105.09064 [a]
- V-17 Qin, Lidiak, Gong, Tang, Wakin, Zhu 2022, *Error analysis of tensor-train cross approximation*, https://arxiv.org/abs/2207.04327 [a]
- V-18 Khoromskij 2011, *O(d log N)-quantics approximation of N-d tensors in high-dimensional numerical modeling*, Constr. Approx. 34 [r]

Kernels, manifolds, ridges, metrics
- V-19 Borovitskiy, Terenin, Mostowsky, Deisenroth 2020, *Matérn Gaussian processes on Riemannian manifolds*, NeurIPS, https://arxiv.org/abs/2006.10160 [a]
- V-20 Borovitskiy et al. 2021, *Matérn Gaussian processes on graphs*, AISTATS, https://arxiv.org/abs/2010.15538 [a]
- V-21 Coifman, Lafon 2006, *Geometric harmonics: a novel tool for multiscale out-of-sample extension of empirical functions*, ACHA 21(1):31–52 [a]
- V-22 Sober, Levin 2016/2020, *Manifold approximation by moving least-squares projection (MMLS)*, https://arxiv.org/abs/1606.07104 [a]
- V-23 Sober, Aizenbud, Levin 2017/2021, *Approximation of functions over manifolds: a moving least-squares approach*, https://arxiv.org/abs/1711.00765 [a]
- V-24 Wilson, Hu, Salakhutdinov, Xing 2016, *Deep kernel learning*, AISTATS, https://arxiv.org/abs/1511.02222 [a]
- V-25 Kanagawa, Hennig, Sejdinovic, Sriperumbudur 2018, *Gaussian processes and kernel methods: a review on connections and equivalences*, https://arxiv.org/abs/1807.02582 [a; power function = posterior sd recalled]
- V-26 Schaback 1995, *Error estimates and condition numbers for radial basis function interpolation*, Adv. Comput. Math. 3:251 [r]
- V-27 Wendland 2005, *Scattered Data Approximation*, Cambridge University Press [r]
- V-28 Wu, Guinney, Maggioni, Mukherjee 2010, *Learning gradients: predictive models that infer geometry and statistical dependence*, JMLR 11:2175–2198 [a]
- V-29 Radhakrishnan, Beaglehole, Pandit, Belkin 2022/2024, *Mechanism of feature learning in deep fully connected networks and kernel machines that recursively learn features*, https://arxiv.org/abs/2212.13881 [a]
- V-30 Romor, Tezzele, Lario, Rozza 2022, *Kernel-based active subspaces…*, https://arxiv.org/abs/2008.12083 [a]
- V-31 Alauzet, Loseille 2016, *A decade of progress on anisotropic mesh adaptation for computational fluid dynamics*, CAD 72:13–39, doi:10.1016/j.cad.2015.09.005 [a]

Neural surrogates and refusal
- V-32 Lu, Jin, Karniadakis 2021, *DeepONet*, https://arxiv.org/abs/1910.03193 [a]
- V-33 Li et al. 2021, *Fourier neural operator for parametric PDEs*, ICLR, https://arxiv.org/abs/2010.08895 [a]
- V-34 Liu et al. 2024, *KAN: Kolmogorov-Arnold networks*, https://arxiv.org/abs/2404.19756 [a]
- V-35 Chen, Rubanova, Bettencourt, Duvenaud 2018, *Neural ordinary differential equations*, NeurIPS, https://arxiv.org/abs/1806.07366 [a]
- V-36 Adcock, Dexter 2021, *The gap between theory and practice in function approximation with deep neural networks*, https://arxiv.org/abs/2001.07523 [a]
- V-37 Grohs, Voigtlaender 2024, *Proof of the theory-to-practice gap in deep learning via sampling complexity bounds for neural network approximation spaces*, https://arxiv.org/abs/2104.02746 [a]
- V-38 Lei, G'Sell, Rinaldo, Tibshirani, Wasserman 2018, *Distribution-free predictive inference for regression*, JASA, https://arxiv.org/abs/1604.04173 [a]
- V-39 Angelopoulos, Bates 2021, *A gentle introduction to conformal prediction…*, https://arxiv.org/abs/2107.07511 [a]
- V-40 Tibshirani, Barber, Candès, Ramdas 2019, *Conformal prediction under covariate shift*, NeurIPS, https://arxiv.org/abs/1904.06019 [a]

Dynamics and chemistry
- V-41 Gelß, Klus, Eisert, Schütte 2019, *Multidimensional approximation of nonlinear dynamical systems* (MANDy), J. Comput. Nonlinear Dyn. 14:061006, https://arxiv.org/abs/1809.02448 [a]
- V-42 Dektor, Rodgers, Venturi 2021, *Rank-adaptive tensor methods for high-dimensional nonlinear PDEs*, J. Sci. Comput., https://arxiv.org/abs/2012.05962 [a]
- V-43 Abel, Anninos, Zhang, Norman 1997, *Modelling primordial gas in numerical cosmology*, New Astron. 2:181–207, https://arxiv.org/abs/astro-ph/9608040 [a]
- V-44 Grackle source, `src/clib/rate_functions.c` and `solve_rate_cool_g.F` at 48e0733, https://github.com/grackle-project/grackle [v]
- V-45 Cantera `data/h2o2.yaml` at 5876366 (GRI-Mech 3.0 H2/O2 submechanism), https://github.com/Cantera/cantera [v]
- V-46 Test Set for IVP Solvers (Bari), ROBER, `src/problems/rober.f`, https://archimede.uniba.it/~testset/ [v]
- V-47 Glover, Abel 2008, *Uncertainties in H2 and HD chemistry and cooling and their role in early structure formation*, MNRAS 388:1627 (venue r), https://arxiv.org/abs/0803.1768 [a]
- V-48 Burke, Chaos, Ju, Dryer, Klippenstein 2012, *Comprehensive H2/O2 kinetic model for high-pressure combustion*, Int. J. Chem. Kinet. 44:444 [r]
- V-49 Pope 2013, *Small scales, many species and the manifold challenges of turbulent combustion*, Proc. Combust. Inst. 34:1–31 [r]

Updated: II-12 (Savostyanov 2014) was read in this review: Thm. 1, and the greedy and DMRG cross (Alg. 1–2).

---

## 7. Synthesis and next steps

**What can be promised.** No method that only samples f certifies the worst error over a box in
high dimension [II-26]. The guarantee that can be built is narrower and is the one the MHD coupling
needs: the worst error over the states the simulation reaches, estimated on held-out tracers and by
checking a random share of answers against the solver, plus a per-query refusal where the local
estimate does not hold, answered by the true solver. That is ISAT's contract [III-12, III-13], and
AFINNI's trust regions with the exact outside flag already have its shape. ISAT's own reports
control a quantile of the error, not its maximum [III-13, III-14]; reaching the maximum needs either
verified local bounds or checking more of the answers.

**The experiments' failures and their remedies.**

| Failure (section 1) | Remedy from the literature |
|---|---|
| ChebTree accepts aliased boxes | off-grid samples before acceptance (`sampleTest`) and a plateau of minimum length (`standardChop`) [III-1, III-2] |
| ChebTree's tail is optimistic at kinks | fit the decay rate; raise the degree only for exponential decay, split otherwise, and use the fitted tail as the estimate [II-7, III-5, III-6] |
| ChebTree overshoots smooth functions | predict the degree from the fitted rate; Leja points, nested at any size, let it grow by one [III-10] |
| tensor boxes stop at d = 3-4 | sparse grids inside the cells of a tree [III-9, I-10]; functional TT inside verified cells [II-13, II-14] |
| SparseCheb stops early and misses small regions | look one step ahead (evaluate every admissible forward neighbour before stopping) and verify at the next split's points; no rule is guaranteed [II-9] |
| SparseCheb pays for kinks globally | multi-element splitting [I-11, III-9] |
| oblique ridges | Euclidean-degree spaces [III-3], or a rotation when the gradient covariance has a gap [II-17] |
| polynomial leaves overshoot in sparse corners | nodes that control the max norm (Chebyshev or Leja points), not random points |
| quadric cuts break sampling coverage | keep cuts convex; put curvature in the leaf model |

**Ranked next steps.**

1. **In-situ tabulation over the existing leaves, with solver fallback and error checking.** Build
   from the states an MHD run queries (or from recorded tracer trajectories), answer inside a leaf's
   region of accuracy, solve and tabulate outside it, and re-check a random share of answers.
   Effect: the dimension that matters becomes that of the accessed manifold. Risks: an error tail
   above the tolerance from non-convex regions [III-14], table memory, sharing the table across MPI
   ranks, and results that depend on query order.
2. **Fix ChebTree's acceptance** (plateau, decay-rate fit, a few off-grid checks per box seeded by the
   box's code, so runs stay deterministic). Effect: aliasing and kinks handled for d <= 4. Risk: more
   evaluations per box (unmeasured), and a fitted rate is still not a bound.
3. **Degree from the fitted rate on nested Leja points.** Effect: less overshoot on smooth boxes.
   Risk: larger Lebesgue constants, modest gains.
4. **SparseCheb inside tree cells, with one-step look-ahead.** Effect: localised kinks and corner
   peaks for d = 5-10. Risk: complexity; the indicators stay heuristic.
5. **Reduce the chemistry inputs before tabulating** (element conservation, principal components
   [III-20], slow manifolds [III-18, III-19], a learned latent space [III-22]) and measure the error
   in the full species space. Effect: from hundreds of inputs to perhaps tens. Risk: the
   reconstruction error is uncontrolled, so step 1's fallback stays necessary.
6. **Functional TT with Chebyshev bases** for smooth maps at d = 5-20 [II-13, II-14], used only inside
   verified cells, since it has no off-grid max-norm guarantee and misses localised features.

The experiments then move from box test functions to chemistry: Robertson's stiff problem, a
primordial H/He/H2 network, then larger sparse networks, all trained from trajectory data and judged
by the worst error on held-out trajectories, the solver-call rate and the inference time.

---

## 8. Review VI: surrogates of vector-valued functions

Scope: sharing structure across outputs R^d -> R^m, error metrics, the output index in a TT, conservation
and positivity, and experiments to choose a vector ChebTT design. Tags: [v] source read, [a] abstract or
summary, [r] recalled. New works VI-1 to VI-23 (docs/LITERATURE.md); earlier ones by their IDs.

### 8.0 Bottom line (ranked by gain over cost)

Scratch runs, one seed, serial, with `chebtt.build` and the joint `build_vector` it then had. Recommendations 1 and 5
were adopted, and 2-4 superseded by ChebTT-M (section 1.19). Evaluations:

| Problem, d, tol | vector TT (output last) | separate TTs | separate TTs, one shared cache |
|---|---|---|---|
| 5 x cos(a_c.x), 6, 0.01 | 18,264 | 4,986 | **2,765** |
| swirl, 2, 0.1 | **608** | 1,270 | 697 |
| swirl, 2, 0.01 | 1,426 | 2,216 | **1,229** |
| swirl, 5, 0.05 | 53,026 | 10,939 (worst 1.03 tol) | 7,770 (1.03 tol) |
| swirl, 10, 0.1 | 1,071,581 | 41,397 | **31,188** |
| rosenbrock_gradient, 5 / 10 | 3,856 / 44,935 | 3,937 / 16,570 | **2,067 / 9,120** |
| genz_triple, 5 / 10 | 2,594 / 8,290 | 2,270 / 4,917 | **1,395 / 3,363** |
| lotka_volterra_state, 5 / 10 | 4,319 / 115,802 | 7,090 / 43,815 | **3,841 / 32,780** |
| wave_pair, 5 / 10 | **763 / 1,471** | 1,815 / 3,416 | 1,255 / 2,647 |

Every model holds its tolerance (0.2-0.6 of it) except the separate TTs on 5-D swirl (1.03).

1. **Make "separate scalar ChebTTs with one shared oracle cache" the default vector ChebTT** (gain 1.3-1.9x
   over separate TTs and 1.1-34x over the vector TT on 10 of 13 cases; it costs about 20 lines). Every oracle
   call stores all m outputs, keyed by point. The m builds use the same seed, so they share the degree
   pre-pass fibres, the Sobol check set and the first maxvol fibres (all seeded at the peak and the centre),
   and the models equal independent builds (identical worst errors in every case). This is the "multiple
   TT-cross" of Olivier et al. [VI-15, a]: own cores and ranks per output, shared sample points.
2. **Pick joint or separate per group of outputs with a rank test** (a further 1.6-1.8x on coherent
   outputs, wave_pair). One TT over a group pays about R_k R_{k+1} oracle calls per core and per node. Here R_k is the
   joint bond rank, which lies between max_c r_{c,k} and sum_c r_{c,k}. Separate TTs pay
   sum_c r_{c,k} r_{c,k+1}. So joint wins when R_k < sqrt(sum_c r_{c,k}^2), about sqrt(m) times one
   component's rank. With ranks read off the 2-D and 5-D grids (section 3), this rule predicts the winner in 12 of the 13
   runs above; the miss is 2-D swirl at 0.1, within 15%.
3. **Fix `build_vector`'s checks: expand every check point to all m output nodes** (no extra oracle calls).
   Today each Sobol, free and look-ahead check tests only the output its snapped last coordinate names, so
   each output gets 1/m of the checks: a reliability fix for the worst error.
4. **Rank per bond, and the output core in the middle, if the vector TT is kept** (gain about 1.7x on the
   proxy). ChebTT grows one rank r on every bond at once. A joint TT's compressed ranks rise towards the
   output core (10-D Lotka-Volterra: 3, 6, 10, 15, 19, 21, 21, 21, 18, 10), so the sampling pays the largest
   bond rank everywhere. This explains most of the 10-D blow-ups (swirl 1.07M evaluations). AMEn or greedy
   cross adapts rank per bond [V-3; V-9, v]. On the coupled Lotka-Volterra species the middle
   placement alone beats separate TTs on the proxy (96 against 160), so this is the variant to test for
   chemistry.
5. **Drop "output PCA, then scalar TTs" for box surrogates.** On the rank proxy it is the worst option or
   tied everywhere: 1.0-9x the separate TTs' cost, 16x the best joint TT's on shared coordinates (section 3). Keep PCA of the outputs only as a *local*
   basis, which is the range of the step map's Jacobian (slow modes, §1.11), inside a record or a cell.
6. **Chemistry: project onto the conserved totals in log coordinates** (exact conservation and positivity
   at a few hundred flops; expected to remove the conserved-direction part of the rollout drift in §1.12).
   See section 4.
7. **Metric for chemistry:** report the worst |Δ log10 n_i| (dex) over species above an abundance floor,
   the worst relative conservation residual, and the rollout error. Do not scale a species' increment by
   its own maximum: that makes a species that barely changes as hard to fit as one that changes a lot.

### 8.1 How the families handle multiple outputs (Q1)

**What sharing can buy.** Our oracle returns all m outputs at every point, without noise: the "block
design" of Bonilla et al. [VI-2, a]. Under noise-free observations and a block design, an intrinsic
coregionalisation GP predicts each task from its own targets only, and "a cancellation of inter-task
transfer occurs". The same holds for any linear interpolant with a shared basis (polynomial, sparse grid,
kernel with one scalar kernel): f_c-hat = P f_c, componentwise. So **sharing never makes an interpolant
more accurate at a given point set.** It can only (a) let one oracle call serve every output, (b) choose
one point set that serves all outputs, and (c) share the inference work. What matters is how far the joint
design exceeds the union of each output's needs.

| Family | What is shared | Cost of the joint design | When sharing pays |
|---|---|---|---|
| Multi-output GP: LMC K = sum_q B_q k_q, ICM (Q = 1) [VI-1, a; VI-3, a; VI-6, r; VI-23, r]; co-kriging across fidelities [VI-22, r] | kernel hyperparameters, one Gram factorisation (ICM) | points = union | noisy or missing outputs, few data; not for our block design [VI-2] |
| Separable matrix-valued kernels, VKOGA [IV-13, a; VI-5, a; VI-7, r] | centres chosen by the residual of the whole vector; one Gram matrix, vector coefficients | the greedy set must serve the worst output | always cheap in points (no rank penalty); matrix-valued kernels help only when outputs correlate [VI-5] |
| Sparse grids / PCE with shared index sets [VI-8, a; VI-16, a] | one grid; refinement on one output or on all with weights (Tasmanian `iOut`) | union of the refinements | always fine: cost is the union |
| Output PCA (PCA-PCE, PCA-GP, PCA-Net, POD-DeepONet) [VI-4, a; VI-9, a; VI-10, a; VI-11, a/r] | an output basis U (m x r); r scalar surrogates of the coefficients U^T f | coefficients mix outputs | outputs of very high dimension (fields, time series) with fast PCA decay |
| EIM / DEIM [VI-12, r; VI-13, a] | a basis plus r "magic" output indices; f ≈ U (P^T U)^-1 P^T f | r oracle outputs instead of m | when outputs are expensive one by one; here all come at once |
| ISAT records [III-12, v] | one region (EOA) and one linear map A (m x d) per record for all species; error is a scaled 2-norm of the whole vector [r] | the region shrinks to the most demanding species | species share curvature directions (a slow manifold) |
| TT with an output index [VI-14, a; V-9, v; VI-15, a] | every core | bond ranks are those of the union (section 3) | coherent outputs: R_k ≈ max_c r_c |

The **"rank-sum" penalty is specific to tensor trains**: in point-based schemes (kernels, ISAT, sparse
grids, trees) a joint design costs about the union of the per-output designs; in a TT it costs the joint
rank squared, and incoherent outputs' joint rank is the *sum* of theirs. Output PCA does not reduce rank: a
coefficient U_j^T f mixes every component, so has the union's TT rank, and under the max norm its error is
multiplied by the row sums sum_j |U_cj|.

ART's and AFINNI's shared leaves are the point-based case; their swirl failures (worst 1.6-3.8) are the
scalar least-squares failure (§0.4 item 1), not a vector effect.

### 8.2 Error metrics for vector outputs (Q2)

Used in the literature:

- **Componentwise relative max norm**, max_x max_c |e_c(x)| / ||f_c||_inf. This is a weighted l-infinity
  norm of D^-1 e with D = diag(||f_c||). It is the bench's metric.
- **Scaled 2-norm**, ||B e||_2: ISAT's error measure, with B a scaling matrix [III-12, r for the form].
- **Relative 2-norm** ||e||/||f||: field emulators, PCA-based methods [VI-4, VI-9, a].
- **Mean errors in log abundance** (MSE, MAE in dex): astrochemistry emulators, CODES [III-24, a]. Few
  report a maximum.

For worst-case reporting there is no single standard. The l-infinity norm of the scaled error is the only
one of these that bounds every output, and it is what a solver fallback needs. Keep it for the bench. For
chemistry, the absolute error in log10 abundance is already scale-free: |Δ log10 n| ≤ ε means a relative
error of at most 10^ε − 1 ≈ 2.3ε. So report:

- the worst dex error over species above a floor (n_i / n_H > 1e-12, say), with species below the floor
  judged in linear space;
- the worst relative error of each conserved total, |E n-hat − E n| / |E n|;
- the rollout error (§1.12).

Scaling an increment Δz_c by its own maximum (the bench's rule applied to increments) inflates the demand
on species near equilibrium. A fixed dex tolerance on the increments is the physical one.

### 8.3 Where the output index goes in a TT (Q3)

**Rank facts** (elementary; checked numerically below). Let F_c^(k) be the unfolding (x_1..x_k ; x_{k+1}..x_d)
of output c.

- With the output index on the right of bond k, the bond rank is dim(sum_c colspace F_c^(k)).
- With it on the left, the rank is dim(sum_c rowspace F_c^(k)).
- Both lie between max_c r_{c,k} and sum_c r_{c,k}.
- The bond right next to the output core has rank at most the number of output modes.

So placement only chooses which union is taken. A block TT [VI-14, a] moves the output (block) index with
the sweep. TT-Toolbox's `amen_cross` takes a function returning an M x B block and reshapes each core as
(r_k n_k) x (r_{k+1} B) on the forward sweep, so the block travels with the core being updated [V-9, v].
That buys per-bond rank adaptivity, not an escape from the union rank.

**Measured unfolding ranks** (full Chebyshev grids, 1e-3 relative singular-value cut, components scaled to
max 1). The proxy counts oracle calls per sweep in units of n: the sum over cores of r_k r_{k+1}, where an
output index costs no oracle call.

| 5-D problem | per-component ranks | joint, output last | proxy: separate / last / best middle / PCA |
|---|---|---|---|
| swirl | 2-3 | 4, 6, 9, 10 | 104 / 182 / 110 / 135 |
| rosenbrock_gradient | 1-2 | 4, 5, 6, 7 | 44 / 103 / 54 / 137 |
| genz_triple | 1-3 | 3, 5, 5, 5 | 46 / 73 / 67 / 149 |
| lotka_volterra_state | 2-4 | 3, 6, 9, 10 | 160 / 175 / 96 / 337 |
| 5 x cos(a_c.x) | 2 | 5, 10, 10, 10 | 80 / 265 / 221 / 744 |
| wave_pair | 2 | 2, 2, 2, 2 | 32 / 16 / 16 / 32 |
| two shared oblique coordinates, 6 outputs | 3-22 | 4, 13, 23, 20 | 2352 / 835 / 579 / 13460 |

- The ranks rise towards the output core. Placing it in the middle (p = d/2) cuts the proxy 1.6-1.9x for
  swirl, Lotka-Volterra and the Rosenbrock gradient.
- Separate TTs still win on every bench problem except wave_pair, whose cos and sin of one ridge share
  their factors, and the coupled Lotka-Volterra species once the output core sits in the middle (96
  against 160).
- Only the synthetic "slow manifold" (six nonlinear mixtures of two shared oblique coordinates) favours
  the joint TT, by 2.8-4x. This is the case the chemistry hopes for.

**Alternatives.** Output POD times a TT per coefficient is a Tucker factor on the output axis, which a joint
TT with the output last already contains (its last bond rank is the number of output modes); scalar TTs of
the coefficients pay the union rank r times. Quantics on the output index [V-4, V-18] needs a smooth
species ordering; there is none.

**For coupled species on a slow manifold:** if the slow coordinates have low TT rank, the species' factor
spaces overlap and a joint TT (per-bond ranks, middle output core) is cheapest; if they are oblique, every
species is high-rank in a box TT anyway (§6.5) and no placement helps. Decide by the rank test on the
chemistry bench.

**Inference.** Compressed coefficients and time per query (numpy, one thread):

| Case | separate | vector |
|---|---|---|
| 10-D Lotka-Volterra | 3,370 coefficients, 2.9 us | 10,510, 1.3 us |
| 10-D Rosenbrock gradient | 302, 0.95 us | 2,564, 0.50 us |

Numpy times are per-call overhead; a compiled one-pass evaluator of the m TTs (block-diagonal cores,
shared Chebyshev recurrences) costs about the separate coefficient count in multiply-adds. A CPU core does ~10^4 multiply-adds in 0.2-1 us; at d ≈ m ≈ 100 a box TT per
species costs ~10^6. The end goal needs a local model in p slow coordinates with an m x p output map
(about m p operations once the record is found).

### 8.4 Conservation and positivity (Q4)

Approaches found:

- **Predict fluxes:** Δc = A S, A stoichiometric, conserves atoms for any S [VI-17, v] (targets from a
  constrained generalised inverse; 6 active species); an architecture version followed [VI-18, a].
- **Solve the constrained outputs as residuals of the others** (ACnet [VI-19, v]). It is exact to machine
  precision, but the residual outputs "show systematically larger errors". Under a worst-case metric that
  is disqualifying: the whole inconsistency lands on the residual species, typically the electron or a
  trace species.
- **Positivity-preserving projection with atom balance** [VI-20, a; details not read]; the RCCE form
  n_i = exp(−g_i − sum_j λ_j a_ij) [VI-21, r; IV-19].

**Recommended (cheap, exact).** Given a predicted log abundance z-hat, find z* = z-hat + E^T λ such that
E exp(z*) = b. Here E is the element and charge matrix (elements x species) and b the input's totals,
which the step conserves.

- It is the Kullback-Leibler projection of n-hat onto {E n = b}, with the convex dual
  min_λ sum_i n-hat_i exp((E^T λ)_i) − b^T λ; Newton on λ (3-6 unknowns: H, He, D, charge) takes 2-4 steps,
  a few hundred flops, and the result is positive and exactly conserving.
- The log correction (E^T λ)_i has the same size for every species carrying element e, so a trace
  species' dex error grows by at most |λ_e| max E_ei. And λ_e is about the relative error of the total,
  which the abundant carriers dominate, so the worst dex error rises by at most about one tolerance.
- It removes exactly the error component along the conserved directions, which a smooth surrogate
  accumulates with one sign over a rollout (§1.12).

Conservation in log coordinates has high TT rank (§6.5), so impose it afterwards rather than learn it.

### 8.5 Experiments (Q5), cheapest first

All on the bench (cache, plots, 3 seeds) once a scratch A/B shows a gain.

- **E1. Shared-cache separate ChebTTs as a scheme** (`chebtt` for vector problems, the vector TT kept as
  `chebtt_vector`) on the five vector problems at 2, 5, 10, then 20-D. Pass: within tolerance on every seed
  and fewer evaluations than the vector TT on 4 of 5 at 10-D. Watch swirl 5-D (seed 0 reached 1.03).
- **E2. Check expansion in `build_vector`.** Check every point at all m output nodes; over 3 seeds, count
  the cases beyond the tolerance, before and after, at an unchanged evaluation count.
- **E3. The rank test.** In the first sweep at the pre-pass degrees, compare the joint fibre-matrix ranks
  with the per-output ones. Check that R_k < sqrt(sum_c r_{c,k}^2) predicts the winner on the 7 problems
  above plus a mixed problem: wave_pair and genz_triple side by side (m = 5), where the best answer is two
  groups. Then group the outputs greedily (merge two groups while the joint rank stays below the rule).
- **E4. Output-core placement.** Generalise the fixed axis of `build` to any position p. Compare p = 0,
  d/2 and d, with and without per-bond rank growth, on lotka_volterra_state (5-D, 10-D), swirl (5-D) and
  the shared-coordinates problem. Expect about 1.7x from the middle placement and more from per-bond
  ranks.
- **E5. Chemistry, joint against separate.** Robertson (3 species) and the primordial network (9 species),
  on the increment Δz in dex with a fixed dex tolerance. Compare a vector ChebTT, shared-cache separate TTs
  and SlowTable on one-step worst dex, evaluations and inference time, after printing the rank test (do
  the species share factors in log coordinates?).
- **E6. KL projection.** Add the projection to every chemistry scheme. Measure the worst dex one-step,
  the conservation residual (to rounding), and the rollout error after 40 steps. Also split the rollout
  error into its components along and across the conserved directions, to know how much of the 36x
  growth the projection can remove.
- **E7. Inference.** A Numba evaluator of m TTs in one pass with shared recurrences, against the vector TT
  (10-D Lotka-Volterra, m = 10), under AGENTS.md's bar.

### 8.6 Pitfalls

- Section 0's table is one seed, serial, checked on 20k uniform points; the bench must confirm it. The rank
  proxy ignores repeated points in a joint TT's index sets, so it overstates the joint cost somewhat.
- Sharing does not help certification: the Novak-Woźniakowski limit [II-26] holds per output.
- `check_points` draws 64 + 32d Sobol points, not a power of two (scipy warns; balance is lost).

---

## 9. Review VII: error metrics and test protocols by dynamical class

Scope: how to classify a system from its trajectories (stiff, contracting, neutral or oscillatory, chaotic), how a
step-map surrogate's one-step error turns into its rollout error, and which metrics, horizons and normalisations
fit each class. Tags: [v] source read, [a] abstract or summary read, [r] recalled. New works VII-1 to VII-34
(docs/LITERATURE.md); earlier ones by their IDs. "Ours" marks a proposal that no source states.

Notation. z_k are the solver's states, Phi the step map, J_k = DPhi(z_k) by central differences (as
`ode.bench.dynamics` already computes), P_{j->n} = J_{n-1} ... J_j (P_{n->n} = I), d_k = S(z_k) - Phi(z_k)
the surrogate's defect at a true state, and delta = max_k |d_k| the one-step error.

### 9.0 Bottom line (ranked by gain over cost)

1. **Normalise every rollout by the accumulation factor A_N = sum_{k=0}^{N-1} ||P_{k+1->N}||** and report
   rho_N = e_N / (delta A_N). Linearised, e_N = sum_k P_{k+1->N} d_k, which is Lady Windermere's fan
   [VII-7, r], so rho_N <= 1 while the errors stay small. rho near 1 means coherent defects (same sign, aligned
   with the growing directions), rho near 1/sqrt(N) means random ones, and rho > 1 means the surrogate left the
   states it was tested on. The bench's `amplification` reports ||P_{1->N}||, the growth of an error made at
   step 1 only; A_N sums the growth over every start step. Measured here on three Robertson test trajectories
   (max norm): A_40 = 49-64. That gives rho_40 = 0.56 for ChebTT box at 0.001 (coherent: this is the "36x"
   of section 1.12), 0.17 for ChebTT box at 0.01 and 0.018 for SlowTable at 0.001. So A_N, not N, is the
   tolerance budget: delta <= tol / A_N keeps a rollout within tol in the worst case. This generalises
   section 0.4's tol / N.
2. **Classify each system from its Jacobian products, not by hand.** The stiffness index is
   s_k = -ln min_i |lambda_i(J_k)|. Jay and Calvo [VII-3, v] call a solution mildly stiff at
   ln S <= -10 and stiff at ln S <= -20, where S is the smallest eigenvalue modulus of the sensitivity matrix.
   Robertson's s_k is 24-27, so it is stiff within one step. The growth type of g_n = ||P_{0->n}|| gives the
   other class: exponential with lambda_1 > 0 is chaotic, polynomial is neutral (phase drift), and an A_N that
   saturates is contracting. A system can carry two labels: stiffness describes the bottom of the spectrum,
   the other class its top.
3. **For stiff systems, score the defect that survives one exact step: J_{k+1} d_k**, or exactly,
   Phi(S(z_k)) - Phi(Phi(z_k)). Robertson's fast singular value is about 1e-11, so a defect in the fast
   direction is erased in one step. ISAT scores |B(R - R_l)| with a free scaling matrix B [III-12, v]. Taking
   B = J of the next step is ours and is not in ISAT. Use this projector, not an orthogonal projection onto
   the slow tangent space: the fast fibres are oblique (CSP [III-19, r]).
4. **For chaotic systems, report the valid prediction time (VPT) in Lyapunov times**, not the error at a
   fixed horizon. Use the NRMSE with each component scaled by its standard deviation over time and the
   threshold eps = 0.5 (Vlachas et al. [VII-18, v]). Also report the VPT that the one-step error predicts,
   VPT* = (1/lambda_1) ln(eps / delta_n), where delta_n is the one-step NRMSE. Report long-term statistics
   (power spectrum, marginal Wasserstein distance, lambda_1 of the surrogate) from long free runs.
5. **For neutral and oscillatory systems, split the error into orbit (amplitude) and phase**:
   d_orb = min_tau |z^_N - z(t_N + tau)| and the drift tau*/t_N per period. The phase error grows linearly
   [VII-27, a], so report its slope, not its value at an arbitrary horizon.
6. **Choose horizons in the system's own time unit.** For a contracting system, run until A_N saturates
   (about 3 tau_slow), then report the stationary error. For a neutral one, use periods (report the error
   per period). For a chaotic one, run up to the VPT, and run separately for the statistics.

### 9.1 Classification from trajectories (Q1)

**Stiffness.** The logarithmic norm mu[A] = lim_{h->0+} (||I + hA|| - 1)/h (mu_2[A] = lambda_max((A + A^T)/2))
bounds ||e^{tA}|| <= e^{t mu[A]}. It was introduced independently by Dahlquist and Lozinskii in 1958 for error
bounds [VII-6, VII-8 r; VII-1 a]. Along a trajectory, ||Phi'|| <= exp(∫ mu), so ln||J_k|| / dt is a lower
estimate of the step's mean mu. Söderlind, Jay and Calvo [VII-2, a] build a stiffness indicator from the upper
and lower logarithmic norms of the vector field. It is computable along a solution and independent of the
method, it defines a local reference time scale, and a large ratio of the integration range to that scale is
necessary for stiffness. The slides behind that paper [VII-3, v] give the computable version: the stiffness
factor S(x, a, h) = min_{s <= t in [a, a+h]} min_j |lambda_j(R(t, s, x(s)))|, with R the sensitivity matrix,
that is our step-map Jacobian. The thresholds ln S <= -10 (mildly stiff) and <= -20 (stiff) come from
S = e^{h lambda} for Dahlquist's test equation and from the stability regions of explicit methods. For one step,
-ln S equals dt max|Re lambda(f_z)| for a frozen linear field: the classical stiffness index, without needing
f_z. The classical stiffness ratio max|Re lambda| / min|Re lambda| [IV-23, r] is ill-defined when the slow
eigenvalue is near zero, as Robertson's is (sigma_2 = 0.999 per step). Brugnano, Mazzia and Trigiante [VII-5, v]
define kappa_c = max_t ||Phi(t)||, gamma_c = (1/T) ∫ ||Phi(t)|| dt and the stiffness ratio
sigma_c = kappa_c / gamma_c >> 1 (no numerical cutoff given). Higham and Trefethen [VII-4, a]: stiffness is
transient and eigenvalues are asymptotic, so pseudospectra characterise it better. In practice compute
singular values beside the eigenvalues. Robertson shows why (computed here): sigma_max(J_k) is about 1.65 per
step, yet the product stays at about 1.5, so a bound built from per-step norms overstates the growth
geometrically.

**Expansion and contraction.** The finite-time Lyapunov exponent (FTLE) is
Lambda_n(z_0) = (1/(n dt)) ln sigma_max(P_{0->n}) [VII-14, a; formula r]. The maximal exponent
lambda_1 = lim Lambda_n is obtained by QR re-orthonormalisation of a bundle of tangent vectors [VII-13, r].
Gilpin applies Gram-Schmidt at every step until the smallest-magnitude exponent is below 1e-8 [VII-19, v].
The Lyapunov time is T_lambda = 1/lambda_1 [VII-18, v]. Chaos means lambda_1 > 0 on a bounded orbit [VII-18, v].

**Neutral directions.** A continuous-time chaotic flow has at least one positive exponent and exactly one zero
exponent, the one along the flow, and a dissipative one has at least one negative exponent [VII-19, v]. A limit
cycle has lambda_1 = 0 and the rest negative. Conservative systems (Lotka-Volterra) have sum lambda_i = 0
(det J = 1) [r]. No source gives a numerical threshold for "lambda_1 = 0". Ours: call a system neutral when
|lambda_1| t_test < 1 and ln g_n grows like p ln n (p of 1 or more, from shear between nearby orbits of
different period) rather than linearly in n. These are the bench's existing amplification curves, refitted.

### 9.2 Error propagation (Q2)

With z^_{k+1} = S(z^_k) = Phi(z^_k) + d^_k, the error obeys e_{k+1} = J_k e_k + d^_k + O(e^2), so
e_N = sum_k P_{k+1->N} d^_k and ||e_N|| <= delta A_N. This is the global-from-local argument of Hairer, Nørsett
and Wanner ("Lady Windermere's fan") [VII-7, r]. With a bound mu on the logarithmic norm along the orbit,
||P_{k->N}|| <= e^{mu (t_N - t_k)}. Then A_N <= 1/(1 - e^{mu dt}), about 1/(|mu| dt), bounded for mu < 0;
A_N = N for mu = 0; and A_N grows like e^{lambda t_N}/(lambda dt) for growth at rate lambda > 0 [r, from the
same sum]. B-convergence carries this to stiff problems: global error bounds that depend on the one-sided
Lipschitz constant, which can stay moderate for arbitrarily stiff systems, rather than on the Lipschitz
constant [VII-9, a; VII-10 via VII-3, v (C-stability)]. Many stiff problems violate a moderate one-sided
constant (a remark found while searching, not checked).

**Is there a standard "accumulation factor"?** Not in the surrogate literature found. The nearest standard
objects:
- Estep's **stability factor**, the norm of the solution of the dual (adjoint) problem, which weights local
  errors in a posteriori global error bounds [VII-11, a]. Our A_N is its discrete forward version.
- The **conditioning parameters** kappa_c and gamma_c of the initial value problem [VII-5, v].

For chaotic systems the stability factor caps even the reference solution. Estep and Johnson [VII-12, a] find
Lorenz pointwise computable on intervals of moderate length, reported as T = 30 in double precision. Kehlet
and Logg [VII-21, a] reach T = 1000 only at order 200 with 400 digits. A rollout beyond the reference's own horizon compares two meaningless trajectories.

**Proposed normalisation (ours).** Report rho_N = ||e_N|| / (delta A_N), together with
delta_eff = ||e_N|| / A_N (the per-step defect the rollout behaves as if it had). Measure the norm in the
bench's own norm (max over scored species, in dex), with A_N in the induced max norm. A second, exact
attribution uses the solver: E_k = Phi^{N-k-1}(S(z^_k)) - Phi^{N-k}(z^_k) gives each step's share of e_N
(the telescoping fan). It costs O(N^2) solver steps, which is fine for 20 trajectories of 40 steps.

### 9.3 Stiff systems: the part of a one-step error that survives (Q3)

A defect in a direction that J_{k+1} kills never reaches the rollout. The natural score is therefore
|J_{k+1} d_k| (linear), or exactly Phi(S(z_k)) - Phi(Phi(z_k)), one extra solve per test state. Why not
project onto the slow manifold:
- CSP and ILDM split fast and slow modes by eigenvectors that are not orthogonal [III-18, III-19, r]. A
  fast-direction defect shifts the slow coordinates by its component along the oblique fast fibre, which J
  captures and an orthogonal projection does not.
- Gear, Kaper, Kevrekidis and Zagaris [VII-29, a] stress that integrating forward to reach the manifold
  shifts the observables. That shift is what J d measures.

ISAT defines the local error as epsilon = |B(R(phi_q) - R_l(phi_q))| with a scaling matrix B, B = I the
simplest [III-12, v]; its region of accuracy is where epsilon <= epsilon_tol. Our choice B = J(Phi(z)) turns
ISAT's mapping error into the surviving error. Lu and Pope's error checking controls a quantile of epsilon
[III-13, v]. Liu and Pope find the global error linear in the tolerance [III-14, v]: the empirical counterpart
of A_N bounded.

**ML papers on stiff kinetics** measure trajectory errors; none found scores a propagated or slow-projected
defect:
- Kim et al. [VII-30, a]: the issue is scaling the outputs and losses for stiff neural ODEs.
- Goswami et al. [VII-31, a]: operator networks for ROBER, POLLU and syngas.
- CODES reports mean relative errors [III-24, v].
- The nearest is Valorani [VII-32, a]: the error in the fast eigenvalues of the learned Jacobian, on
  Davis-Skodje [IV-24]. That preprint was withdrawn by arXiv; do not build on it.

For the bench this matters: SlowTable's 4x misses sit in the initial transient (section 1.12), where the fast
species are far from the manifold. If the surviving defect there is within tolerance, the miss is harmless to
a chained run.

### 9.4 Chaotic systems (Q4)

- **NRMSE and VPT** [VII-18, v]. NRMSE(o^) = sqrt(<(o^ - o)^2 / sigma^2>), with sigma the per-component
  standard deviation over time and <.> the mean over components. VPT = (1/T_lambda) max{t_f : NRMSE < eps
  for all t <= t_f} with eps = 0.5, averaged over 100 initial conditions on the attractor. Lorenz-96 has
  lambda_1 of about 1.68 (F = 8) and 2.27 (F = 10). The best VPT there was about 2.3 Lyapunov times, and
  "predictive utility" was limited to one to two Lyapunov times. Long-term quality is judged by the error of
  the mean power spectral density, PSD(f) = 20 log10(2|U(f)|) dB.
- **Reservoir prediction and climate** [VII-16, v]. Pathak et al. (Chaos 2017): on Kuramoto-Sivashinsky the
  reservoir predicts for about 5 Lyapunov times and then keeps the attractor's "climate". The positive and
  zero Lyapunov exponents of the autonomous reservoir approximate the true ones. Climate failure is "fairly
  evident" in return maps. PRL 2018 [VII-17, r]: about 8 Lyapunov times for large KS.
- **dysts** [VII-19, v]. 131 systems, each with its dominant period from the power spectrum (random-phase
  surrogates for significance). Sampled at 15 (coarse) or 100 (fine) points per period; 10 periods train,
  2 validate, and the forecast covers the last 2 periods of an unseen trajectory. Eight metrics (MSE, MASE,
  MAE, MARRE, |CV|, 1 - r^2, sMAPE, MAPE), with sMAPE reported. The best forecast error correlates most with
  lambda_max, less with fractal dimension and entropy.
- **Gilpin 2023** [VII-20, v]. 135 systems, 24 methods. sMAPE eps(t) = (200/t) sum_{t'} |y - y^| /
  (|y| + |y^|). VPT is the last time with sMAPE < 50: 13.9 +- 2.8 Lyapunov times for the best model, and
  15 +- 5.3 by the first doubling of the accumulated error. 100 points per period, horizons to 50 periods,
  15-17 largely concordant metrics. In that regime accuracy no longer tracks lambda_1.
- **Correlation time** [VII-22, v]: the time until the mean Pearson correlation with the truth falls below
  0.8 or 0.9, following Kochkov et al. [VII-33, r] and Sun et al.
- **Invariant measure.**
  - Jiang et al. [VII-23, a]: optimal-transport losses preserve the invariant measure.
  - Park et al. [VII-24, a]: low test error does not imply statistical accuracy, and matching Jacobians in
    training helps. That speaks for our Jacobian-based tests.
  - Brenner et al. [VII-25, r]: the KL divergence of state-space densities (D_stsp) and the Hellinger
    distance of power spectra.
  - Mikhaeil et al. [VII-26, a]: training gradients on chaotic orbits always diverge.
- **Predictability time.** T_p = (1/lambda_1) ln(Delta / delta_0) for an initial error delta_0 and a
  tolerance Delta [VII-15 Boffetta et al., r]. Our VPT* applies it with delta_0 = one-step NRMSE, so
  VPT / VPT* says whether a surrogate's rollout is as good as its one-step error allows.
- **Standard values:** eps = 0.5 for the NRMSE (Vlachas), sMAPE 50 (Gilpin), correlation 0.8 or 0.9
  (PDE-Refiner), 100 initial conditions, a horizon of 2 to 50 dominant periods. sMAPE suits neither log
  abundances nor components that cross zero (its denominator is |y| + |y^|). Use the NRMSE scaled by the
  standard deviation.

### 9.5 Oscillatory and neutral systems (Q5)

- **Error growth.** Cano and Sanz-Serna [VII-27, a] give conditions under which the error on a periodic orbit
  grows asymptotically linearly with the number of periods. For Hamiltonian and reversible systems, methods
  with the matching geometric property have better growth. Recalled from Hairer, Lubich and Wanner
  [VII-28, r]: on integrable systems the error of symplectic or reversible methods grows linearly, that of
  general methods quadratically, because a drifting invariant changes the period and so the phase speed.
- **Mechanism.** On a limit cycle a perturbed state relaxes to the cycle but keeps a phase shift: its
  asymptotic phase, or isochron [VII-34, r]. That is the zero exponent.
- **Benchmarks.** None found reports phase and amplitude separately. dysts only aligns systems by dominant
  period [VII-19, v].
- **Ours.**
  - The orbit error min over |tau| <= T/2 of |z^_N - z(t_N + tau)|, against a densely stored true orbit.
  - The phase tau* and its drift rate per period.
  - For conservative systems (Lotka-Volterra), the drift of the invariant.
  - The plain pointwise error over a few periods only, labelled as such.

### 9.6 Rollout horizon by class (Q6)

- **Contracting.** e_N tends to a stationary value of at most delta A_infinity. Run until A_N/A_infinity
  >= 0.95, about 3 tau_slow / dt with tau_slow = -dt / ln|lambda_slow(J)| for the slowest non-unit
  eigenvalue, or the application's coupling length if shorter, and report the stationary error and rho. A
  slow mode with |lambda| near 1 makes the system neutral over any feasible horizon. Robertson's sigma_2 of
  0.999 per step gives tau_slow of about 1000 steps, so over 40 steps it is "stiff + neutral". A_N = 50-64
  is close to N, which is why its rollout grows linearly.
- **Neutral or oscillatory.** Measure in periods: at least 10 to estimate the slope (ours; no standard
  found). Report the error per period and the phase drift rate, which extrapolate.
- **Chaotic.** No fixed horizon: report the VPT, and stop at a few times VPT* (beyond it pointwise error is
  saturated noise). The statistics need separate free runs long enough to converge the spectrum and the
  marginals: hundreds of Lyapunov times or dominant periods (ours). Gilpin evaluates to 50 periods
  [VII-20, v]. The reference solver's own computable horizon applies [VII-12, VII-21].

### 9.7 Recommended protocol (Q7)

Implementable with the bench's finite-difference Jacobians and stored test trajectories.

**Diagnostics** (per test trajectory, cached beside `amplification`):
1. J_k for every step: 2d solver steps each, already done.
2. Stiffness: s = max_k(-ln min_i |lambda_i(J_k)|). Stiff at s >= 20, mildly stiff at s >= 10 [VII-3].
   The slow dimension m = #{sigma_i(J_k) > 1e-6}.
3. Growth: g_n = ||P_{0->n}||, Lambda_n = ln g_n / (n dt), and A_N by backward products
   (P <- P J_k, O(N) products per end point). On one long trajectory of the attractor, lambda_1 (and
   lambda_2) by QR.
4. Time units: T_lambda = 1/lambda_1, the dominant period T_peak from the power spectrum (as dysts does),
   and tau_slow.
5. Labels:
   - **chaotic**: lambda_1 T_run >= 5 with lambda_1 > 0 on every seed.
   - **contracting**: A_N saturates within the run (A_{2N}/A_N < 1.1).
   - **neutral**: neither, with polynomial g_n.
   - **stiff**: from step 2. Stiffness combines with any of the three.

**Metrics**, every class: delta (max norm, per-component scale: dex for chemistry, the standard deviation over
time otherwise), its 99.9% quantile, the speedup, and rho_N. Then by class:

| Class | One-step | Rollout | Horizon |
|---|---|---|---|
| contracting | delta | stationary e_N, rho_N | A_N within 5% of saturation, or the application's length |
| stiff | delta and the surviving delta_J = max \|J_{k+1} d_k\| | rho_N with delta_J and A_N from k + 2 | as its top class |
| neutral or oscillatory | delta | orbit error, phase drift per period, invariant drift | 10 or more periods |
| chaotic | delta, one-step NRMSE delta_n | VPT (eps = 0.5) in Lyapunov times, VPT / VPT*; PSD error, W1 per component, lambda_1 of the surrogate | to about 3 VPT*; free runs of hundreds of T_lambda for the statistics |

**Normalisations, so that systems of different stiffness and dimension compare:**
- Time in T_lambda, T_peak or tau_slow, never in steps.
- Components in dex or in standard deviations.
- Rollouts as rho_N = e_N / (delta A_N): dimensionless and between 0 and 1 while the linearisation holds.
- Tolerance budgets as delta <= tol / A_N.
- Dimension through the max over components (and the RMS beside it).

### 9.8 Pitfalls

- A_N and the surviving defect are linearisations at the true states. For large defects (the initial
  transient), use the exact solver forms. rho > 1 flags this, or a drift into untested states.
- Eigenvalue-based stiffness misses non-normal transient growth [VII-4]. Keep the singular values: on
  Robertson the per-step sigma_max of 1.65 and the product's 1.5 differ completely in what they predict.
- Jay and Calvo's thresholds use ln S over a time scale h, and the bench's h is dt. A larger dt makes any
  dissipative system "stiffer". Report s together with dt.
- FTLEs at short horizons are positive in contracting non-normal systems (transient growth); classify as
  chaotic only from a converged lambda_1, not from Lambda_n.
- Finite differences at h = 1e-5 resolve sigma down to about 1e-11 only. The 1e-12 values on Robertson are
  at that floor, so s is a lower bound there.
- Chaotic pointwise metrics at long horizons, and comparisons beyond the reference solver's own computable
  time, measure noise.
- The [r] items, in particular HNW's fan theorem, the log-norm bounds, Boffetta's T_p and the
  Hairer-Lubich-Wanner growth rates, are standard but were not re-read here. VII-32 was withdrawn by arXiv.

Measured for this review (light, three Robertson test trajectories, 40 steps, `src/structure/cache/ode`):
s_k = 24-27; sigma(J) = (1.2-1.66, 0.84-0.9996, 1e-12-2e-11); A_10, A_20, A_40 = 17, 34, 64 (max norm,
worst trajectory); rho_40 = 0.56 (ChebTT box 0.001), 0.17 (ChebTT box 0.01), 0.018 (SlowTable 0.001).

---

## 10. Review VIII: piecewise and partitioned tensor trains for kinks, jumps and fronts

Scope: ChebTT (`src/structure/schemes/chebtt.py`) fails on axis-aligned kinks (`genz_continuous`), axis-aligned
jumps (`genz_discontinuous`), an oblique planar jump (`oblique_jump`), a spherical jump (`ball_jump`)
and a sharp radial front (`front`). This review covers the literature for the two planned designs:
piecewise univariate bases per axis (design 1, already under way in the working tree: `axis_edges`,
`locate_edge`, `is_singular`, `piece_bounds`), and a spatial partition with ChebTT leaves (design 2).
IDs are those of docs/LITERATURE.md; new works are VIII-*. Tags: **[v]** read in the source,
**[a]** abstract or summary only, **[r]** recalled, not checked. **[m]** marks short numpy runs made
for this review (scratch scripts, not committed). They are not bench runs.

### 10.0 Bottom line (ranked by gain over cost)

1. **Design 1 is the right tool for (a) and (b), and only for them.** A product of univariate
   functions jumps or kinks only along axis-aligned hyperplanes, and a sum of r such products only
   along their union. So no per-axis piecewise basis, at any rank, represents an oblique or curved
   jump. For separable singularities every fibre along axis k meets the edge at the same coordinate,
   so one breakpoint per axis serves all of the core's rank functions. The function train finds the
   same structure per fibre [II-14, v]. Finish design 1, then measure two refinements: one degree
   per piece instead of one per axis, and a location precision tied to the tolerance for kinks
   (section 1).
2. **For (c) and (d), subtract a located jump instead of partitioning.** Locate interface points
   with the edge search design 1 already runs (on an oblique jump it finds the edge "moving" between
   the two fibres, and those points lie on the interface). Fit a plane, or a sphere or quadric, and
   confirm the fit on fresh fibres. Estimate the one-sided jumps j_0..j_m of f and its normal
   derivatives from the fibre fits. Then run the unchanged global ChebTT on
   g = f − H(φ) Σ_k j_k φ^k / k! and add the term back at inference. In 2-D, with an oblique line
   and a jump field that varies along it, the remainder's max error falls as n^-(m+1) (m = 3:
   2e-12 at n = 256, rank 4). Without the subtraction it stays at 1.5 for every n [m]. This keeps
   one TT, so smooth functions keep exactly their current cost. It is XFEM's enrichment, the
   immersed interface method's correction and Eckhoff's jump subtraction, in TT form
   [VIII-9, VIII-11, VIII-12].
3. **A partition (design 2) does not scale on curved or oblique interfaces unless its leaves are
   aligned with them.** A tree graded toward the interface needs about (1/h)^(d−1) cut leaves. Cut
   polytope leaves break ChebTT's product domain. A two-leaf split at the located interface, with
   each leaf on the full box fed by a jump-corrected oracle, is the subtraction above under another
   name. Keep design 2 for what subtraction cannot carry: several crossing interfaces, jump fields
   that are not smooth, and local high-rank features. Do not build it first for (c)-(e).
4. **Detection costs, when the interface is curved.** Edge tracking with an SVM [I-20, v] aims at
   99% of the volume correctly classified. The paper itself says full resolution grows exponentially
   with the surface's dimension. Hyperspherical sparse grids [VIII-3, v] needed 94k evaluations to
   place an off-centre sphere within 1e-3 at 5-D, and 513k for 1e-2 at 9-D, with linear bases. For
   a max-norm surrogate the interface must be exact (a parametric surface confirmed at rounding
   level) or have a refused band. A parametric fit (plane: d + 1 unknowns, sphere: d + 2) confirmed
   on extra fibres is cheap; anything else is a research problem.
5. **QTT is not a shortcut [m].** Interleaved quantics ranks of a half-plane indicator stay bounded
   for rational directions (4 at 45°, 17 for slope 0.7, 5 for (1,1,1) in 3-D) but grow for an
   irrational one (14 → 158 from 2^6 to 2^11 points per axis). For a disk they grow as about 2^(L/2)
   (26 → 267). Ordered axis by axis, which is ChebTT's coordinate TT, the plane's rank is about the
   grid size (717 of 1024). A staircase leaves an O(2^-L) band at the jump that the max norm sees.

### 10.1 Function trains with adaptive or piecewise bases (Q1)

**What the function train does [II-14, v].** Gorodetsky, Karaman and Marzouk build the FT by a
continuous TT-cross. Each fibre met during the cross "is approximated adaptively to a particular
tolerance as it is encountered" (sec. 4). Parameters are "refin[ed] ... independently for each
univariate function" (sec. 2), so the knots differ between the rank functions of one core (fig. 3:
"the basis for each function was chosen online, adaptively, and independently for each fiber").
The continuous maxvol ("dominant") searches each piece of a piecewise polynomial for the maximum
(sec. 4.2). Breakpoints are never shared or chosen by edge detection. Refinement is breadth-first:
split into three equal regions, fit degree 6, and split again while the leading coefficient
contributes more than 1e-10 to the squared norm (sec. 5.2, for the 2-D Genz discontinuous).
Reported costs: machine accuracy on 2-D Genz discontinuous at rank 1, evaluations clustered at the
jump. On the d-dimensional rank-1 Genz discontinuous (eq. 19, jumps at x_i = 1/2), evaluations grow
linearly in d up to d ≈ 125 (fig. 7; the axis reaches 4e4, and we could not read the curve values
from the text). Errors are relative errors of an integral and a derivative, not max norms. The paper
gives no oblique-jump example. Its singular example (eq. 20) reports RMS errors and ranks that grow
with d.

**C3 [VIII-1, v].** `piecewisepoly.c` defaults: `nregions = 5` equal pieces, `maxorder = 10`,
`coeff_check = 3`, `epsilon = 1e-15`, `minsize = 1e-2`. A piece is split again into equal pieces
while any of its last 3 coefficients squared exceeds epsilon (`adapt_help`). Sums and products of
two piecewise polynomials first take the union of their breakpoints (`piecewise_poly_match`). So
breakpoints are per function, unions appear only in arithmetic, and the refinement is
midpoint-style with a minimum piece size. In the max norm a jump that no knot hits leaves an error
of the jump's size in a piece of width ≥ `minsize`. The scheme is accurate in L2 and integrals,
which is what the paper reports, not in the max norm.

**Tensorized (QTT-in-axis) univariate bases [VIII-2, a].** Ali and Nouy show that splines with free
knots, and Besov functions, are approximated at near-optimal rates by tensorized univariate
functions. Breakpoints are found implicitly, at the cost of L binary cores per axis and a dyadic
resolution floor. On smooth axes this would replace a Chebyshev degree n by L·r² entries and break
"exactly the current cost". Rejected for the main path.

**For design 1 as written (working tree).**
- One breakpoint set per axis, shared by all rank functions, is correct for (a) and (b). FT's
  per-fibre bases matter only when the singularity moves with the other coordinates, and then the
  format cannot represent it anyway (bottom line 1). The confirmation fibre displaced by 1% is the
  right test of alignment. On `genz_discontinuous` the displaced fibre can land where f ≡ 0 along
  the axis (x_2 > w_2) and find no edge. That "absent" result should not veto the edge, since an
  unneeded breakpoint costs only n + 1 nodes.
- **Degree per piece (to measure).** `build` gives all pieces of an axis one degree. On Genz
  discontinuous one piece is identically 0, and on kinks the pieces have unequal widths. FT and C3
  adapt each piece separately [II-14, VIII-1, v].
- **Kink precision tied to the tolerance [derived].** A kink misplaced by δ leaves an error of about
  |[f′]|·δ, so δ ≤ 0.01·atol/|[f′]| suffices. A jump needs rounding level, plus refusal of the band
  |x_k − b| < δ for a formal max-norm claim. `locate_edge` runs 45 steps of 3 evaluations
  (2 + 135 + 8 = 145 per edge). For a jump, bisection on the value (1 evaluation per step) reaches
  2^-45; for a kink, about 15-25 steps suffice. This saves 60-100 evaluations per edge, which
  matters at 20-D with two bases and two fibres per axis.
- `INSET` (end nodes 1e-10 of the axis inside a breakpoint) is necessary: second-kind Chebyshev
  points include the endpoints, and a node on the edge takes either side's value at random.
- **Trigger:** "must refine" plus "algebraic decay". On a smooth function `is_singular` must reject
  every candidate. Count its fires on the smooth fast set and require 0. With 0 fires the smooth
  costs are identical by construction; otherwise each fire costs 2·65 + 145 per base.

### 10.2 Tensor trains on discontinuities (Q2)

- **Coordinate TT, oblique jump.** Sampled on an n × n grid, 1[x + y < c] is a triangular 0/1
  matrix of rank about n, with singular values decaying like 1/k [r]. We measured 717 for slope 0.7
  on 1024² [m]. No continuous approximant gets the max error below half the jump near the
  interface, whatever its rank. So ChebTT cannot pass (c) or (d) by more rank or degree. This is a
  representation limit, not a sampling one.
- **QTT [m, exact TT-SVD ranks, 2^10 points per axis].** Interleaved: 4, 17 (bounded in L: a carry
  automaton), 98 (golden-ratio slope, growing), 173 (disk); 3-D at 2^6: 5, 39, 248 (ball).
  Sequential: 1023, 717, 633, 336. sin(2s) + 1[s > 0.7] along (0.8, 0.6): 9 at 1e-8 for every L. QTT
  exploits lattice alignment, not obliqueness. QTT patching [VIII-7, a] (block-sparse patches of
  lower bond dimension, "substantial improvements for sharply localized functions") is the closest
  existing piecewise TT, without max-norm control.
- **TT for shocks [VIII-13, a]** (a TT stochastic finite-volume method) gives no rank result for jumps.
  **Tree partitions** with local models (Tree-PCE [VIII-8, a]; I-18, I-19, II-6, III-9, IV-6) never
  carry a TT per leaf with a max-norm target; section 1.9 (2.5-4x on kinks) is the only measurement.

### 10.3 Discontinuity detection, cost against dimension (Q3)

| Method | What it returns | Cost in evaluations | Fit with cross |
|---|---|---|---|
| 1-D edge search on fibres (chebfun [IV-1, v]; `locate_edge`) | one point at rounding level | ~50-300 per edge; 45 bisection steps for a jump | native: the fibres are the cross's |
| Polynomial annihilation, axis by axis [IV-2, a; VIII-5, a; II-29] | jump function on scattered/sparse-grid points | PA "to completion" grows exponentially with d and dominates beyond d = 3 [I-20, v, fig. 5.8]; with sparse-grid refinement, linear in d for a hypersphere up to d = 100 [IV-4, v] | gate: tells a jump from a steep gradient |
| PA seed + SVM edge tracking [I-20, v] | classifier for 99% of the volume | 10-20 edge points suffice to seed; the uncertainty-sampling stop is a space-filling design of the surface, so it is exponential in d − 1 for full resolution (their sec. 4) | none for max norm, except as a refusal mask |
| Hyperspherical sparse grid [VIII-3, v; VIII-4, a] | r(θ) for a star-shaped surface, each value by 1-D bisection | off-centre sphere, distance error e∞: 5-D 1e-3 in 94k, 9-D 1e-2 in 513k (Table 5.1, linear basis, adaptive + acceleration) | r(θ) could itself be a ChebTT whose entries are bisections [speculative] |
| Parametric fit (plane, sphere, quadric) of located points | exact surface if the hypothesis holds | plane (d + 2 points) × 45: about 1k at 20-D; sphere similar; general quadric (d + 1)(d + 2)/2 points: about 10^4 at 20-D | confirm on fresh fibres; refuse a band if the confirmation fails |

Saye's height-function recursion [VIII-6, a] is the standard way to decide, in a box, along which
axis an implicit surface is a graph: pick k with ∂φ/∂x_k bounded away from 0, else subdivide. It
needs only 1-D root finding. Use it if a curved interface is ever handled in graph coordinates.

### 10.4 Multi-element methods carrying a TT per element (Q4)

ME-gPC [I-11] and MEPCM [III-9] split boxes by coefficient-decay criteria. A TT per element is
direct, but the interface is then resolved by grading, at the (1/h)^(d−1) cost above.
Minimal multi-element collocation [IV-5, a] cuts elements along the detected discontinuity, so they
are not boxes, and a TT does not fit them without a map. We found no published method that carries
a TT per element of a discontinuity-fitted partition [searched; absence is not proof]. Two ways to
carry one:

- **Mapped leaves (transfinite / graph maps) [VIII-15, r]:** side A = {x_k < h(x_rest)} as
  x_k = lo_k + t·(h(x_rest) − lo_k), t ∈ [0, 1]. The map couples t with x_rest (rank rises), and
  where the plane exits a side face h leaves [lo_k, hi_k]: clipping puts an oblique kink in x_rest.
  Fine in 2-D, a (d−2)-dimensional problem again in high d.
- **Full-box leaves fed by an extended oracle** (f on its own side, f minus the Taylor jump model on
  the other) are subtraction (section 5); two TTs beat one only if the sides' ranks differ a lot.

### 10.5 Coordinates and corrections aligned to the interface (Q5)

- **Rotation** makes a planar interface axis-aligned (exp(a·x) stays rank 1 under any rotation), but
  the rotated box leaves the domain. Section 1.9 rejected global rotation because f must not be
  evaluated outside the box. Coordinate flows [V-11, a] lower FTT ranks by a linear change of
  variables; they do not address domains.
- **Polar and spherical coordinates** make a sphere axis-aligned (Chebfun's disk and sphere use
  low-rank double-Fourier forms [VIII-14, r]). Over a box, though, the radial coordinate's range
  depends on the angle, which is the same clipping problem.
- **Jump subtraction (recommended).** Following XFEM's Heaviside enrichment [VIII-11, r], the
  immersed interface method's jump corrections [VIII-12, r] and Eckhoff's subtraction of jump terms
  to restore spectral convergence [VIII-9, a; VIII-10, r], write
  f = g + H(φ)·Σ_{k≤m} j_k(π(x)) φ(x)^k / k!. Here φ is the fitted level set (signed distance for a
  plane, |x − c|² − R² for a sphere), π the projection onto the interface, and j_k the jumps of the
  k-th normal derivative. If the j_k are exact, g is C^m across the interface.
  2-D test [m]: line at angle 0.6, f_A = exp(0.5x + 0.3y), f_B = cos(x − 2y) + 0.5, exact j_k, max
  error of the full Chebyshev interpolant of g on an 801² grid:

  | m | n = 16 | 64 | 256 | rank at 1e-6 (n = 256) |
  |---|---|---|---|---|
  | none | 1.5 | 1.7 | 1.5 | 144 |
  | 0 | 2.5e-2 | 6.2e-3 | 1.6e-3 | 144 |
  | 1 | 3.3e-4 | 2.0e-5 | 1.3e-6 | 22 |
  | 2 | 6.1e-6 | 9.2e-8 | 1.5e-9 | 8 |
  | 3 | 1.4e-7 | 5.4e-10 | 2.1e-12 | 4 |

  The rate is n^-(m+1) and the rank falls with m. For the bench's cases, j_k is constant on the
  interface: `oblique_jump` has j_0 = 1 and j_k = 0 (g = sin(2 q·x), rank 2). `ball_jump` has
  constant j_k = −(d/dφ)^k e^-(0.64 + φ) at φ = 0, with φ = |x − c|² − R² applied outside the ball,
  which avoids ∇φ = 0 at the centre. So only a few interface samples are needed to confirm that j_k is
  constant. A j_0 error of δ leaves a jump of δ in g, which needs δ ≲ atol/3 (Gibbs adds about 9%).
  The j_k come from one-sided Chebyshev fits on the edge-search fibres. Along axis k, the k-th
  derivative jump equals j_k·q_k^k for a plane, so use the axis with the largest |q_k|. Equivalently,
  use the graph form Σ ĵ_k(x_rest)(x_k − h(x_rest))^k / k!, whose ĵ_k are exactly what a fibre
  along axis k measures.
- **Fronts (e).** `front` is tanh((|x| − 1)/0.05): f = P(φ(x)) with φ = |x|² (rank 2) and a 1-D
  profile P. The same subtraction applies with a smooth profile in place of H. Locate points with
  f = 0 by 1-D root finding on fibres (smooth, so a secant costs about 10 evaluations each), fit a
  quadric, read P off one normal fibre, verify at check points, and run ChebTT on f − P(φ). The
  remainder here is exactly 0 [derived]. A front whose amplitude varies along the surface leaves a
  remainder with the same sharpness but an amplitude that falls with the distance along the surface
  [speculative]. Composition structure F(Σ u_i(x_i)) is close to the active-manifold and level-set
  learning works already registered (II-21, II-22); none of them targets the max norm.

### 10.6 Recommended designs and cheap experiments, by gain over cost

1. **X1, design 1 as is (hours).** Run `genz_continuous` and `genz_discontinuous` at 2, 5, 10 and
   20-D with `edges` on and off, then the smooth fast set with edges on. Pass if the smooth cases
   have the same evaluation counts and models bit for bit (`is_singular` fires 0 times; log every
   `axis_edges` call and its cost), and (a) and (b) come within tolerance near the smooth median
   costs plus about 2·d·(2·65 + 145).
2. **X2, cheaper and per-piece (a day).** One degree per piece, value bisection for jumps, a
   tolerance-tied precision for kinks, and "absent on the confirmation fibre" accepted when that
   fibre is flat. Measure the savings at 20-D under the AGENTS.md bar.
3. **X3, planar subtraction for `oblique_jump` (2-3 days).** Collect the moving edges `axis_edges`
   already rejects. Locate d + 2 interface points (45 bisections each, brackets predicted by the
   running plane fit), fit the plane, and confirm on 2 fresh fibres (|predicted − located| ≤ 1e-9
   of the width). Estimate j_0..j_2 at 3-5 points; if they are constant within atol/10, use the
   constants. Wrap the oracle as g = f − S, train ChebTT on g, predict g + S, and refuse
   |φ| < 1e-9. Expected: near `sin(2 q·x)`'s smooth cost plus about 1-2k evaluations at 20-D.
   Run 2, 5, 10, 20-D.
4. **X4, spherical subtraction for `ball_jump`.** The same with a sphere (d + 2 unknowns) and the
   correction applied outside. Fall back to a diagonal quadric (2d + 1) if the sphere is not
   confirmed. In 2-D and 3-D first. Note that the sphere touches the faces x_i = 1 (c_i + R = 1), so
   fibres along e_i there graze it. The radial φ does not mind, but a graph form along e_i would.
5. **X5, profile subtraction for `front`** as in section 5. It is specific to level-set structure.
   Run it only after X3-X4 show that the interface machinery holds.
6. **X6, design 2 (only if X3-X5 fail, or for crossing interfaces).** Split at the located interface
   with full-box leaves fed by the extended oracle, or use a graded tree with ChebTT leaves and
   refusal at the depth cap in d ≤ 3. Do not expect it to scale in d.

Every step leaves smooth problems on the current path: subtraction runs only after `axis_edges`
has located a confirmed interface, and a single-piece ChebTT is the same object as today's.

### 10.7 Pitfalls

- A max-norm claim on a jump needs the interface to rounding or a refused band (report its volume
  share). SVM and sparse-grid detectors report misclassified volume (I-20) or Hausdorff distance
  (VIII-3), not max errors; FT's "machine accuracy" (II-14 sec. 5.2) is an L2/integral result.
- Kinks and jumps need different precisions: 2^-45 for both wastes evaluations, a 1e-3 cut for both
  recreates section 1.7's near-face trap for jumps.
- Piece ends must stay inside a breakpoint (`INSET`), and so must leaves of a jump-fitted partition.
- Subtraction moves the error into the j_k: a biased j_0 is a small jump ChebTT cannot remove, so
  check j_0 at fresh interface points.
- The 2-D subtraction test used exact jumps and a full tensor grid: it shows the representation
  works, not the estimation or the cross. X3 tests those.
