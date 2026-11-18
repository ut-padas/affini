# Surrogates with a small worst-case error: experiments and literature

This document collects the experiments run beside the library to find a surrogate scheme whose
**worst pointwise error** stays below a tolerance with few oracle evaluations and fast inference,
and three literature reviews that informed them. The code of the current experiments is in
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
6. [Synthesis and next steps](#6-synthesis-and-next-steps)

---

## 0. Working memory: the current state

Updated 2026-11-18. Numbers: worst |error| / tolerance over the test set (below 1 is within),
evaluations of the oracle; "k/N" counts cases (problem x tolerance) within on every seed.

### 0.1 Where things are

- Code: [`src/structure/`](../src/structure/README.md) (its README says how to run the bench).
  Schemes: `chebtree.py` (ChebTree), `sparsegrid.py` (SparseCheb), `chebtt.py` (ChebTT); they register
  in `schemes.py`; `bench.py` runs, caches and plots every method; `problems.py` holds the problems and
  `TOLERANCES`.
- Results (not committed): `src/structure/cache/<setup>/` and `output/<setup>/`, setup `cpu-8rank`.
  Start from `output/cpu-8rank/overview-d<d>.png` (a pass/fail grid of every problem, tolerance and
  method, with the evaluations) and `overview-d<d>-cost.png` (training and inference time, pieces,
  mean error); then per problem `<problem>-d<d>-cost.png` and the maps (2-D) or slice and shadow views.
  `python -m src.structure.bench --replot --setup cpu-8rank --d <d> --problems ...` redraws them.
- Literature: [LITERATURE.md](LITERATURE.md) (every work reviewed, IDs I-* to V-*); reviews I-IV are
  sections 2-5 below.

### 0.2 How the work proceeds (standing rules)

- **Climb dimensions on fast problems first.** The fast set: `rosenbrock`, `genz_oscillatory`,
  `genz_corner_peak`, `genz_product_peak`, `genz_gaussian`, `lotka_volterra`. Run a scheme on it at
  2, 5, 10, 20 dimensions; only when it holds there does it earn the slow problems (`wave` and
  `lorenz96` above 3-D for the baselines, the rotated `gaussian`, `genz_continuous`, the jumps,
  `ackley`, `rastrigin`, `front`, `lorenz96_chaotic`). Do not tune a scheme for 2-D.
- **Same setup for every method in a comparison**: 8 MPI ranks, one thread per rank
  (OMP/NUMBA/OPENBLAS = 1), CPU; one MPI job at a time (six parallel jobs exhausted 30 GB). AFINNI is
  not run at 10-D and above (tens of GB per rank). Serial A/B checks of a scheme in scratch scripts are
  fine, with `OPENBLAS_NUM_THREADS` set low: BLAS threads competing with the bench made QR 50x slower.
- Fail fast: a variant that loses on the fast set is removed, with its numbers recorded in section 1.
- Every result reported goes through the bench (cache and plots); serial scratch scripts are only for
  quick A/B checks. The bench redraws a problem's plots, the summary and the overview after every
  problem, with every cached method, so a new scheme's column appears after its first problem.
- ART with affine leaves is not run at 10-D and above either: 2.8M evaluations and 75-95 s per run on
  10-D Rosenbrock, still 3.3-4.0x the tolerance; ART with quadratic leaves represents ART there.

### 0.3 Standings (bench, 8 ranks, 3 seeds)

| Scheme | 2-D | 3-D | 5-D | 10-D | Limits |
|---|---|---|---|---|---|
| AFINNI | 0/46 | 5/30 | 0/20 | not run | least squares on random points; up to 11.8M evaluations at 5-D |
| ART, affine | 1/46 | 3/30 | 0/20 | not run (cost) | as AFINNI |
| ART, quadratic | 1/46 | 3/30 | 2/21 | 0/2 (running) | as AFINNI |
| ChebTree | **46/46** (median 487) | **30/30** (median 1.3k) | – | – | (n + 1)^d points per box: dead above 3-4 D |
| SparseCheb | 43/45 (median 293) | 27/30 | over budget on oblique features | – | pays for kinks globally; misses small regions |
| **ChebTT** | 12/12 | (backfill pending) | **21/21** (median 1.1k) | 3/3 (running) | high rank: rotated narrow peaks, chaotic maps; kinks at tight tolerances |

Serial ChebTT runs outside the bench (one seed): 10-D 24/24 cases of 8 problems (1.6k-101k
evaluations, including `wave` and `lorenz96`), 20-D 20/21 of 7 problems (3k-130k; `lotka_volterra`
at 0.05 over the 300k budget). Over budget at 5-D: the rotated Gaussian, `lorenz96_chaotic`.

### 0.4 Conclusions that drive the next steps

1. Least-squares pieces on random points (AFINNI, ART) never control the worst error; their mean
   error is small. Interpolation at nodes chosen for the max norm (Chebyshev), with checks at the
   points a refinement would sample next, does: worst errors land at 0.1-0.5 of the tolerance.
2. Dimension is the decisive cost. Tensor boxes stop at 3-D, sparse grids at oblique features at 5-D.
   A tensor train sampled only along maxvol fibres (ChebTT) costs linear in d and holds the fast set
   to 20-D with 10^3-10^5 evaluations.
3. No box-sampling scheme can certify the worst error in high dimension (Novak-Woźniakowski): the 5-D
   rotated Gaussian fills ~1e-4 of its box, and every random check set misses it. ChebTT finds it only
   because its cross is seeded at the centre, where that peak happens to be.
4. **Chemistry's structure is a slow manifold.** Measured: the Robertson step map in log
   coordinates has Jacobian singular values ~1.7, 1.0, 1e-10 for every step dt >= 0.1 (rank 2 of 3:
   the fast direction collapses). For a large network the rank is the slow manifold's dimension, not
   the species count; this is the structure a chemistry surrogate must find. So the chemistry stage
   trains and tests on visited states (held-out trajectories), not on a box.
5. The surrogate should *discover* structure rather than assume a basis. What has worked so far is
   adaptive sampling that exposes structure: a tree (locality), maxvol fibres (low rank and coupling),
   dimension-adaptive sparse grids (interaction order). Candidates that learn a metric or a manifold
   (a learned-metric kernel model, ISAT ellipsoids, diffusion maps, metric-based adaptation) are
   under review (Review V, in progress) and under test (section 1.10).

### 0.5 In progress and next

- Running: the bench's fast ladder at 10-D and 20-D (ChebTT, ART quadratic), then MetricKernel's bench
  runs on the fast set at 2-D and 5-D; Review V (TT-cross
  practice, Riemannian and manifold surrogates, a primordial-chemistry rate table).
- Next: a chemistry bench on visited states: Robertson (exact rates), then a primordial H/He/H2
  network from published rate tables (no Cantera: the networks are hand-coded and checked against
  their sources). Inputs in log abundances; judged by the worst error on held-out trajectories, the
  solver-call (refusal) rate and the inference time. Scheme to build: ISAT's table of visited states
  with solver fallback, a local model in slow coordinates read off the step map's Jacobian, a learned
  metric for where a record is trusted, and look-ahead checks before a record grows (section 5).
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

- Degrees decided by the model: an axis doubles its degree when its upper half of degrees still
  changes the prediction at the check points; otherwise the rank grows. Raising only the rank at a
  fixed degree could never resolve a narrow peak (the rank climbed to 74 at degree 8).
- Free checks: grid points of earlier sweeps and lower degrees, which the final cores do not
  interpolate, concentrated where the cross explored. They caught a narrow 5-D peak (0.022 > 0.01)
  that uniform check points missed.
- A look-ahead at the next nested grid's points along every axis through the worst check point and
  through the largest |f| seen; the right index sets seeded at the largest |f| and the centre.
- Results: 5-D bench 21/21 (median 1.1k evaluations; baselines 0-2 of 21 with 10-1000x more); serial
  10-D 24/24, 20-D 20/21. Over budget: the 5-D rotated anisotropic Gaussian (high rank in axis
  coordinates), `lorenz96_chaotic`, `genz_continuous` at 0.02 (kinks).
- Many smooth cases end far below the tolerance (1e-6 of it): the degree doubling overshoots, cheap
  here but wasted evaluations.

### 1.9 Rejected: global rotation, and ChebTT in tree cells

- Global active-subspace rotation (gradient covariance from 10 d random points) made SparseCheb exact on
  the 5-D wave with 629 evaluations (plain: over 300k) and Genz oscillatory with 581, but: it makes f be
  evaluated outside the box (Lotka-Volterra blows up there, NaN), its gradients are noise where f is
  concentrated (the rotated Gaussian), and ChebTT got more expensive on non-ridge functions (corner
  peak 722 -> 1.2k). A rotation must be local and must stay inside the domain.
- A tree of ChebTT cells (split along the axis whose fibres need the highest degree): 2.5-4x plain
  ChebTT's evaluations on the kinks and no help for the Gaussian or the chaotic map.

### 1.10 A learned-metric kernel model (prototype, not committed)

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
few hundred random check points do not find it. Not promoted beyond 5-D; its bench runs at 2-D and 5-D
are for the record. Its natural setting is a set of visited states, where the candidates come from the
simulation and boundaries matter less.

### 1.11 The chemistry premise, measured

Robertson's stiff network (k1 = 0.04, k2 = 3e7, k3 = 1e4), states from trajectories at random times
after Dirichlet-random initial conditions, central differences of the step map in log10 coordinates:
singular values ~(1.3-1.7, 1.0, 1e-2-1e-4) at dt = 1e-3 and ~(1.2-1.7, 0.5-1.0, 1e-10) for every
dt from 0.1 to 1e3. The fast direction collapses once dt exceeds the fast time scale.

### 1.12 Engineering edge cases met

- Every rank computed and wrote a problem's cached slice directions at once; a rank read a file another
  was writing and the run crashed (now written to a per-rank file and renamed into place).
- A bench run with a subset of methods redrew the plots with that subset only (now every cached
  method; `--replot` redraws from the cache).
- Stopping a collective MPI run leaves the other ranks waiting: stop by the PIDs of the launcher and
  its ranks; never `pkill -f`.
- The GPU never beat 8 CPU ranks on these workloads (launch- and transfer-bound, 200k kernel launches)
  and predicted 2-50x slower per query.

### 1.13 Engine findings from the same work

- A stable counting sort plus per-block merge sort replaced two full-batch argsorts in
  `sort_by_block_then_value`: single-thread AFINNI training 36-39% faster, bit-identical results.
- In the distributed experiment, work stealing on and off gave identical trees, evaluation counts
  and times from 2 to 64 ranks at tolerance 0.05: stealing rarely or never triggers for that workload.

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
Numbers marked "measured" come from short runs of `src/structure/chebtree.py` as of the fibre-check commit.

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

## 6. Synthesis and next steps

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
