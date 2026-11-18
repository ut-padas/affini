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

Contents:

1. [What the experiments taught](#1-what-the-experiments-taught)
2. [Review I: spending fewer evaluations in adaptive trees](#2-review-i-spending-fewer-evaluations-in-adaptive-trees)
3. [Review II: structure-aware surrogates with max-norm control](#3-review-ii-structure-aware-surrogates-with-max-norm-control)
4. [Review III: from the experiments' failures to chemistry tabulation](#4-review-iii-from-the-experiments-failures-to-chemistry-tabulation)
5. [Synthesis and next steps](#5-synthesis-and-next-steps)

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

### 1.7 Engine findings from the same work

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

## 5. Synthesis and next steps

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
