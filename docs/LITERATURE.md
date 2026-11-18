# Literature register

The register of every work reviewed for the surrogate experiments (`src/structure/`,
[SURROGATE_RESEARCH.md](SURROGATE_RESEARCH.md)), so that no work is reviewed twice.

**Rule.** Before searching the literature, check this register. After a review, append every work it read or
cited to the table of its topic, under a new ID series (I-* to III-* are the reviews in SURROGATE_RESEARCH.md,
IV-* the fourth review), and update the entries whose takeaway or use the review changed.

**Verified.** *yes*: the source itself was read. *abstract-only*: an abstract or summary was read. *unverified*:
the work was cited from memory, or the review did not record how it checked it. The status of I-* to III-* is
taken from the reviews' own statements: Review I read abstracts and summaries; Review II recorded nothing;
Review III marked its items "(abstract)" or "(recalled)" and left checked items unmarked. A work can be
listed under two IDs (II-2 = III-1); both are kept, since the documents cite both.

**Where.** "Review I/II/III" are sections 2-4 of SURROGATE_RESEARCH.md, "Synthesis" is its section 5, and
"Review IV" is the fourth review (kink splits, the limits in high dimension, the design for chemistry).

## Adaptive polynomial and Chebyshev approximation

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| II-1 | Townsend, Trefethen 2013, *An extension of Chebfun to two dimensions*, SISC 35(6), https://people.maths.ox.ac.uk/trefethen/publication/PDF/townsend_trefethen13.pdf | unverified | Low-rank Chebyshev approximation in 2-D; ranks grow for rotated or radial features | Review II |
| II-2 | Aurentz, Trefethen 2017, *Chopping a Chebyshev series*, ACM TOMS 43(4), https://arxiv.org/abs/1512.01803 | unverified | Stop when the coefficients level off at the tolerance; check at extra points | Review II; = III-1 |
| II-3 | Trefethen 2013/2019, *Approximation Theory and Approximation Practice*, SIAM, https://people.maths.ox.ac.uk/trefethen/ATAP/ | unverified | Bernstein-ellipse bound: a certificate only if rho and M are known | Reviews II, III; ChebTree's decay fit (the reason a fitted rate is not a bound) |
| II-4 | Hashemi, Trefethen 2017, *Chebfun in three dimensions*, SISC 39(5), https://people.maths.ox.ac.uk/trefethen/chebfun3_final.pdf | unverified | Low-rank Tucker-type Chebyshev approximation in 3-D | Review II |
| II-6 | Aiton, Driscoll 2019, *An adaptive partition of unity method for multivariate Chebyshev polynomial approximations*, SISC 41(5), https://arxiv.org/abs/1805.00423 | unverified | Overlapping Chebyshev patches split where the coefficients do not decay; tensor patches limit it to d of about 4 | Review II; ranked scheme 4 |
| II-16 | Trefethen 2017, *Cubature, approximation, and isotropy in the hypercube*, SIAM Rev., https://epubs.siam.org/doi/10.1137/16M1066312 | unverified | Rotated features cost more in every axis-based format | Review II |
| III-1 | Aurentz, Trefethen 2017, *Chopping a Chebyshev series*, ACM TOMS 43(4):33, https://arxiv.org/abs/1512.01803 | yes | standardChop: a plateau that is long, flat and low enough, chopped at its elbow; coefficients cannot see aliasing | Review III; ChebTree's chop and its check points |
| III-2 | Chebfun, `@chebtech/sampleTest.m`, https://github.com/chebfun/chebfun | yes | Compare f with the interpolant at two fixed off-grid points; a deterministic test can be outwitted | Review III; ChebTree's seeded check points |
| III-3 | Trefethen 2017, *Multivariate polynomial approximation in the hypercube*, Proc. AMS 145:4837, https://arxiv.org/abs/1608.02216 | yes | Euclidean-degree index sets suit analytic f; total degree favours the axes, maximal degree the diagonals | Review III; Synthesis (oblique ridges) |
| IV-1 | Pachón, Platte, Trefethen 2010, *Piecewise-smooth chebfuns*, IMA J. Numer. Anal. 30(4):898-916, doi:10.1093/imanum/drp008 (report NA-08-07: https://www.cs.ox.ac.uk/files/717/NA-08-07.pdf) | yes | Chebfun's splitting: an edge located to the last bit by finite-difference zooms (about 300 evaluations per edge), a midpoint cut otherwise, and removable cuts merged afterwards; it never leaves a kink just inside a face | Review IV: diagnosis of ChebTree's kink splits, fix 1 |

## Sparse grids and multi-element collocation

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| I-10 | Jakeman, Roberts 2011, *Local and dimension adaptive sparse grid interpolation and quadrature*, https://arxiv.org/abs/1110.0010 | abstract-only | The old model's residual at a new point is a free error indicator; local plus dimension adaptivity | Reviews I, III; Synthesis (sparse grids in tree cells) |
| II-8 | Barthelmann, Novak, Ritter 2000, *High dimensional polynomial interpolation on sparse grids*, Adv. Comput. Math., https://users.fmi.uni-jena.de/~novak/sparse.html | unverified | Smolyak on Clenshaw-Curtis points converges like N^-k times a log factor whose power grows with d | Review II; SparseCheb |
| II-9 | Gerstner, Griebel 2003, *Dimension-adaptive tensor-product quadrature*, Computing 71, https://ins.uni-bonn.de/media/public/publication-media/dimadapt.pdf | unverified | Grow the index set by surplus; the surplus can stop refinement early | Reviews II, III; SparseCheb's growth |
| III-7 | Ma, Zabaras 2009, *An adaptive hierarchical sparse grid collocation algorithm for the solution of stochastic differential equations*, JCP 228:3084-3113 (volume and pages unverified), https://doi.org/10.1016/j.jcp.2009.01.006 | abstract-only | ASGC: refine a point's children when its hierarchical surplus exceeds a threshold (local, piecewise-linear) | Review III |
| III-8 | Pflueger 2010, *Spatially Adaptive Sparse Grids for High-Dimensional Problems*, Dr. Hut (TUM thesis) | abstract-only | Spatially adaptive sparse grids with modified boundary bases | Review III |
| III-9 | Foo, Wan, Karniadakis 2008, *The multi-element probabilistic collocation method*, JCP 227:9572, https://www.math.lsu.edu/~xlwan/papers/journal/mepcm.pdf | abstract-only | Sparse grids inside the elements of an adaptive partition (L^2 analysis) | Review III; Synthesis (tree of sparse grids) |
| III-10 | Narayan, Jakeman 2014, *Adaptive Leja sparse grid constructions for stochastic collocation*, SISC 36:A2952, https://arxiv.org/abs/1404.5663 | abstract-only | Leja sequences are nested at any size: the degree can grow by one | Review III; Synthesis; Review IV (box fallback T4) |
| III-11 | Chkifa, Cohen, Schwab 2014, *High-dimensional adaptive sparse polynomial interpolation and applications to parametric PDEs*, FoCM 14:601 | abstract-only | Greedy Leja interpolation on lower sets has sup-norm rates for anisotropic holomorphic f | Reviews III, IV (what is achievable) |
| IV-4 | Xiu (PI), Jakeman, Archibald, DOE final report DE-SC0011615, *Advanced dynamically adaptive algorithms for stochastic simulations on extreme scales*, https://www.osti.gov/servlets/purl/1345533 | yes | Summarises II-29 and IV-5: refinement guided by polynomial annihilation puts points only near the discontinuity; points grew linearly in d for a hyperspherical jump up to d = 100 | Review IV |
| IV-6 | Wilka, Lang 2024 (v2 2025), *Adaptive hp-polynomial based sparse grid collocation algorithms for piecewise smooth functions with kinks*, https://arxiv.org/abs/2404.02556 | abstract-only | Local low-order bases where f is not smooth, variable order elsewhere, an optional kink detector to choose the degree | Review IV (tree of sparse grids, T4) |
| IV-7 | Bungartz, Griebel 2004, *Sparse grids*, Acta Numerica 13:147-269 | unverified | Mixed-smoothness rates with a log factor whose power grows with d | Review IV (what is achievable) |

## hp-adaptivity and edge detection

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| I-11 | Wan, Karniadakis 2005, *An adaptive multi-element generalized polynomial chaos method*, https://www.math.lsu.edu/~xlwan/papers/journal/megpc1.pdf | abstract-only | Split an element when its high-mode variance share times its measure exceeds a threshold | Reviews I, III; Synthesis (multi-element splitting) |
| I-20 | Gorodetsky, Marzouk 2014, *Efficient localization of discontinuities in complex computational simulations*, https://arxiv.org/abs/1402.2845 | abstract-only | Locates a discontinuity by an actively refined SVM boundary, up to d = 11 | Review I |
| II-7 | Mavriplis 1994, *Adaptive mesh strategies for the spectral element method*, CMAME 116, https://ntrs.nasa.gov/api/citations/19920022021/downloads/19920022021.pdf | unverified | Fit the last coefficients to C e^(-sigma j): raise the degree if the decay is fast, split otherwise | Reviews II, III; ChebTree's geometric decay fit |
| II-29 | Jakeman, Archibald, Xiu 2011, *Characterization of discontinuities in high-dimensional stochastic problems on adaptive sparse grids*, JCP 230, https://www.osti.gov/servlets/purl/1345533 (this URL serves the DOE report IV-4) | unverified | Polynomial annihilation detects jumps on adaptive sparse grids | Reviews II, IV |
| III-4 | Houston, Suli 2005, *A note on the design of hp-adaptive finite element methods for elliptic partial differential equations*, CMAME 194:229 | abstract-only | Local Sobolev regularity from a root test on the coefficients | Review III; ChebTree's algebraic decay fit |
| III-5 | Eibner, Melenk 2007, *An adaptive strategy for hp-FEM based on testing for analyticity*, Comput. Mech. 39:575, https://doi.org/10.1007/s00466-006-0107-0 | abstract-only | Coefficients decay exponentially if and only if f is analytic; fit a decay law per element | Review III; ChebTree's exponential test |
| III-6 | Mitchell, McClain 2014, *A comparison of hp-adaptive strategies for elliptic partial differential equations*, ACM TOMS, https://math.nist.gov/~WMitchell/papers/nistir7824.pdf | yes | Of 13 strategies, coefficient decay is the best general one, in the energy norm, not the max norm | Review III |
| IV-2 | Archibald, Gelb, Yoon 2005, *Polynomial fitting for edge detection in irregularly sampled signals and images*, SINUM 43(1):259-279, doi:10.1137/S0036142903435259 | abstract-only | Polynomial annihilation: a jump function that tends to 0 away from edges, minmod over orders; tells jumps from steep gradients on scattered data | Review IV: gate for kink splits (fix 2) |
| IV-3 | Gelb, Tadmor 1999, *Detection of edges in spectral data*, Appl. Comput. Harmon. Anal. 7:101-135 | unverified | Concentration factors detect edges directly from spectral coefficients | Review IV (an edge test on a box's existing coefficients) |
| IV-5 | Jakeman, Narayan, Xiu 2013, *Minimal multi-element stochastic collocation for uncertainty quantification of discontinuous functions*, JCP 242, doi:10.1016/j.jcp.2013.02.035 | abstract-only | Elements cut along the detected discontinuity, not along the axes, so the number of elements is minimal | Review IV |
| IV-20 | Babuška, Guo 1986, *The h-p version of the finite element method, Parts 1 and 2*, Comput. Mech. 1 | unverified | Geometric grading toward a singularity, low degree in the cells next to it | Review IV |

## Trees, partitions and local models

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| I-1 | Murthy, Kasif, Salzberg 1994, *A system for induction of oblique decision trees*, https://arxiv.org/abs/cs/9408103 | abstract-only | OC1: oblique splits by hill-climbing with restarts | Review I |
| I-2 | Ittner, Schlosser 1996, *Non-linear decision trees (NDT)*, ICML, https://mlanthology.org/icml/1996/ittner1996icml-non/ | abstract-only | Oblique splits in lifted coordinates give curved boundaries | Review I; quadric splits in ART (rejected: leaves in several components, worst errors far above the tolerance) |
| I-3 | Yildiz, Alpaydin 2001, *Omnivariate decision trees*, IEEE TNN, doi:10.1109/72.963795 | abstract-only | Choose the node type per node by a test; nonlinear nodes everywhere overfit | Review I |
| I-4 | Breiman 1993, *Hinging hyperplanes for regression, classification, and function approximation*, IEEE TIT, doi:10.1109/18.256506 | abstract-only | Two planes joined at a hinge, fitted by alternating least squares | Review I |
| I-5 | Chaudhuri, Huang, Loh, Yao 1994, *Piecewise-polynomial regression trees*, Statistica Sinica 4, https://pages.stat.wisc.edu/~loh/treeprogs/guide/smooth.pdf | abstract-only | SUPPORT: polynomial leaves, optionally smoothed across pieces | Review I |
| I-6 | Loh 2002, *Regression trees with unbiased variable selection and interaction detection*, Statistica Sinica 12, https://www3.stat.sinica.edu.tw/statistica/j12n2/j12n21/j12n21.htm | abstract-only | GUIDE: split variables chosen by residual tests | Review I |
| I-7 | Powell 2004, *Least Frobenius norm updating of quadratic models*, Math. Prog. 100 | abstract-only | A quadratic from 2d + 1 points by the least Frobenius norm of the Hessian | Review I |
| I-8 | Domingos, Hulten 2000, *Mining high-speed data streams*, KDD, https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf | abstract-only | Hoeffding trees split once a concentration bound separates the candidates | Review I |
| I-9 | Witteveen, Iaccarino 2012, *Simplex stochastic collocation*, SISC, doi:10.1137/100817504 | abstract-only | The old model's residual at a new point is a free per-cell error indicator | Review I |
| I-14 | Hanley, Lippman-Hand 1983, *If nothing goes wrong, is everything all right?*, JAMA | abstract-only | Rule of three: k passing fresh points bound the violating volume share by about 3/k at 95% | Review I (staged acceptance) |
| I-15 | Chen, Sun, Xu 2007, *Optimal anisotropic meshes for minimizing interpolation errors in L^p-norm*, Math. Comp. 76, https://www.math.uci.edu/~chenlong/Papers/Chen.L%3BSun.P%3BXu.J2007.pdf | abstract-only | Piecewise-linear interpolation is near-optimal on meshes quasi-uniform in the metric abs(H) | Review I |
| I-16 | Mirebeau, Cohen 2012, *Greedy bisection generates optimally adapted triangulations*, https://arxiv.org/abs/1101.1452 | abstract-only | Greedy bisection reaches the optimal aspect ratio | Reviews I, II |
| I-17 | Cohen, Dyn, Hecht, Mirebeau 2012, *Adaptive multiresolution analysis based on anisotropic triangulations*, https://arxiv.org/abs/1101.1512 | abstract-only | Anisotropic multiresolution by greedy bisection, error O(N^-1) in 2-D | Review I |
| I-18 | Dekel, Leviatan 2005, *Adaptive multivariate approximation using binary space partitions and geometric wavelets*, SINUM, doi:10.1137/040604649 | abstract-only | Approximation theory for BSP partitions with polynomial pieces | Review I |
| I-19 | Chandrasekaran, Wakin, Baron, Baraniuk 2009, *Representation and compression of multidimensional piecewise functions using surflets*, https://dspace.mit.edu/handle/1721.1/52325 | abstract-only | Cells cut by polynomial surfaces reach near-optimal rates for piecewise-smooth f | Review I |
| II-25 | Loseille, Alauzet 2011, *Continuous mesh framework I/II*, SINUM 49, https://hal.archives-ouvertes.fr/hal-03167252 | unverified | Metric-based anisotropic adaptation: optimal L^p error for linear pieces, practical only for d of 3 or less | Review II |

## Kernels and Gaussian processes

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| I-12 | Gramacy, Lee 2008, *Bayesian treed Gaussian process models*, https://arxiv.org/abs/0710.4536 | abstract-only | Treed GPs with active learning in the leaves; too heavy per leaf here | Review I (rejected: cost per leaf) |
| I-13 | Taddy, Gramacy, Polson 2011, *Dynamic trees for learning and design*, https://arxiv.org/abs/0912.1586 | abstract-only | Sequential trees with active learning | Review I (rejected: cost per leaf) |
| II-27 | Fiedler, Scherer, Trimpe 2021, *Practical and rigorous uncertainty bounds for Gaussian process regression*, AAAI, https://arxiv.org/abs/2105.02796 | unverified | A rigorous max-error bound needs an assumed RKHS-norm bound | Review II (certified error) |
| IV-9 | Fuselier, Wright 2012, *Scattered data interpolation on embedded submanifolds with restricted positive definite kernels: Sobolev error estimates*, SINUM 50(3):1753-1776, https://arxiv.org/abs/1007.2825 | abstract-only | Restricted kernels on a compact embedded submanifold: error rates set by the manifold's dimension | Review IV (kernels on the visited manifold) |
| IV-10 | Santin, Haasdonk 2017, *Convergence rate of the data-independent P-greedy algorithm in kernel-based approximation*, Dolomites Res. Notes Approx. 10:68-78, https://arxiv.org/abs/1612.02672 | abstract-only | P-greedy is quasi-optimal for Sobolev kernels and its points are asymptotically uniform | Review IV |
| IV-11 | Wenzel, Santin, Haasdonk 2023, *Analysis of target data-dependent greedy kernel algorithms: convergence rates for f-, f·P- and f/P-greedy*, Constr. Approx. 57:45-74, doi:10.1007/s00365-022-09592-3; sequel: Santin, Wenzel, Haasdonk, SINUM 2024, https://arxiv.org/abs/2307.09811 | abstract-only | Target-adaptive greedy selection beats the rates of uniform points; the beta-greedy scale from P-greedy (beta = 0) to f-greedy (beta = 1) | Review IV |
| IV-12 | Wenzel, Santin, Haasdonk 2021, *A novel class of stabilized greedy kernel approximation algorithms: convergence, stability and uniform point distribution*, J. Approx. Theory (venue unverified), https://arxiv.org/abs/1911.04352 | abstract-only | Restricting the greedy choice keeps it stable with optimal rates | Review IV (kernel conditioning) |
| IV-13 | Wirtz, Haasdonk 2013, *A vectorial kernel orthogonal greedy algorithm*, Dolomites Res. Notes Approx. 6:83-100, doi:10.14658/PUPJ-DRNA-2013-Special_Issue-10 | abstract-only | VKOGA: greedy kernel interpolation of vector-valued maps, with a-posteriori bounds | Review IV (local models, T3) |

## Low-rank tensors

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| II-5 | Dolgov, Kressner, Stroessner 2021, *Functional Tucker approximation using Chebyshev interpolation*, SISC 43(3), https://arxiv.org/abs/2007.16126 | unverified | Chebfun3F: about 75% fewer evaluations than Chebfun3 | Review II |
| II-10 | Oseledets, Tyrtyshnikov 2010, *TT-cross approximation for multidimensional arrays*, LAA 432, https://doi.org/10.1016/j.laa.2009.07.024 | unverified | Cross interpolation on maxvol fibres, about d n r^2 evaluations per sweep | Reviews II, IV |
| II-11 | Oseledets, *Black box cross approximation*, https://oseledets.github.io/news/black-box-cross-approximation-for-multidimensional-arrays | unverified | Practical black-box TT-cross | Review II |
| II-12 | Savostyanov 2014, *Quasioptimality of maximum-volume cross interpolation of tensors*, LAA 458, https://arxiv.org/abs/1305.1818 | unverified | Max-norm quasi-optimality on the grid only; nothing off the grid | Reviews II, III, IV |
| II-13 | Bigoni, Engsig-Karup, Marzouk 2016, *Spectral tensor-train decomposition*, SISC 38(4), https://arxiv.org/abs/1405.5713 | unverified | Spectral TT with cross and rank adaptivity | Reviews II, III, IV; Synthesis step 6 |
| II-14 | Gorodetsky, Karaman, Marzouk 2019, *A continuous analogue of the tensor-train decomposition*, CMAME 347, https://arxiv.org/abs/1510.09088 | unverified | Function train (C3) | Reviews II, III, IV; Synthesis step 6 |
| II-15 | Stroessner, Sun, Kressner 2024, *Approximation in the extended functional tensor train format*, Adv. Comput. Math., https://arxiv.org/abs/2211.11338 | unverified | Extended functional TT | Review II |

## Ridge functions, active subspaces and nonlinear coordinates

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| II-17 | Constantine, Dow, Wang 2014, *Active subspace methods in theory and practice*, SISC 36(4), https://arxiv.org/abs/1304.2070 | unverified | Eigenvectors of E[grad f grad f^T]; a mean-square bound; no gap for radial features | Review II; ChebTree `build_rotated` (helps ridges only) |
| II-18 | Fornasier, Schnass, Vybiral 2012, *Learning functions of few arbitrary linear parameters in high dimensions*, FoCM 12, https://arxiv.org/abs/1008.3043 | unverified | Ridge recovery with uniform error, with high probability, polynomial in d | Reviews II, IV |
| II-19 | Cohen, Daubechies, DeVore, Kerkyacharian, Picard 2012, *Capturing ridge functions in high dimensions from point queries*, Constr. Approx. 35, https://link.springer.com/article/10.1007/s00365-011-9147-6 | unverified | Deterministic ridge recovery; needs a positive or sparse direction | Reviews II, IV |
| II-20 | Hokanson, Constantine 2018, *Data-driven polynomial ridge approximation using variable projection*, SISC 40(3), https://arxiv.org/abs/1702.05859 | unverified | Polynomial ridge fits by variable projection | Review II |
| II-21 | Bridges, Gruber, Felder, Verma, Hoff 2019, *Active manifolds*, ICML, https://proceedings.mlr.press/v97/bridges19a.html | unverified | Nonlinear 1-D coordinates following the gradient; L2 errors | Review II |
| II-22 | Zhang, Zhang, Hinkle 2019, *Learning nonlinear level sets for dimensionality reduction in function approximation*, NeurIPS, https://neurips.cc/virtual/2019/poster/14262 | unverified | Learned level-set coordinates; L2 errors, many gradients | Review II |
| II-23 | Teng, Wang, Ju, Gruber, Zhang 2023, *Level set learning with pseudo-reversible neural networks*, SISC 45(3), https://arxiv.org/abs/2112.01438 | unverified | Pseudo-reversible networks for level sets | Review II |
| II-24 | Bigoni, Marzouk, Prieur, Zahm 2022, *Nonlinear dimension reduction for surrogate modeling using gradient information*, https://arxiv.org/abs/2102.10351 | unverified | Polynomial feature maps matched to grad f | Review II (ranked scheme 3) |

## Complexity, widths and certification

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| II-26 | Novak, Wozniakowski 2009, *Approximation of infinitely differentiable multivariate functions is intractable*, J. Complexity, https://users.fmi.uni-jena.de/~novak/ca1.pdf | unverified | Uniform approximation of C^∞ functions needs at least 2^(floor(d/2)) samples: no black-box certificate in high d | Review II; Synthesis; Review IV |
| II-28 | Hickernell, Jiang, Liu, Owen 2013, *Guaranteed conservative fixed width confidence intervals via Monte Carlo sampling*, https://arxiv.org/abs/1208.4318 | unverified | "Fooling" functions defeat any finite sample | Review II |
| IV-8 | Cohen, DeVore, Schwab 2010, *Convergence rates of best N-term Galerkin approximations for a class of elliptic sPDEs*, FoCM 10; and 2011, *Analytic regularity and polynomial approximation of parametric and stochastic elliptic PDEs*, Anal. Appl. 9 | abstract-only | Anisotropic holomorphy gives best n-term rates independent of the number of variables | Review IV (what is achievable) |
| IV-14 | Krieg, Pozharska, Ullrich, Ullrich 2024, *Sampling projections in the uniform norm*, https://arxiv.org/abs/2401.02220 | abstract-only | Any n-dimensional space has a sampling projection with at most 2n points and sup-norm of order sqrt(n) | Review IV (samples suffice once a good local space is known) |

## In-situ tabulation

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| III-12 | Pope 1997, *Computationally efficient implementation of combustion chemistry using in situ adaptive tabulation*, CTM 1:41, https://tcg.mae.cornell.edu/pubs/Pope_CTM_97.pdf | yes | Tabulate only the accessed region; linear maps in ellipsoids of accuracy; retrieve, grow, add, or solve | Review III; Synthesis step 1; Review IV design |
| III-13 | Lu, Pope 2009, *An improved algorithm for in situ adaptive tabulation*, JCP 228:361, https://tcg.mae.cornell.edu/pubs/Lu_LRP_JCP_09.pdf | yes | Error checking re-solves random retrievals and shrinks the offending records; it controls the 90% quantile, not the maximum | Reviews III, IV |
| III-14 | Liu, Pope 2005, *The performance of in situ adaptive tabulation in computations of turbulent flames*, CTM 9:549, https://tcg.mae.cornell.edu/pubs/Liu_P_CTM_2005.pdf | yes | Rare large errors from non-convex regions of accuracy; the global error is linear in the tolerance | Reviews III, IV (pitfalls) |
| III-15 | Tonse, Moriarty, Brown, Frenklach 1999, *PRISM: piecewise reusable implementation of solution mapping*, Israel J. Chem. 39:97, https://escholarship.org/uc/item/6sr15042 | abstract-only | Quadratic polynomials on demand in hypercubes of composition space: a ChebTree built in situ | Review III |
| III-16 | Liang, Stevens, Farrell 2009, *A dynamic adaptive chemistry scheme for reactive flow computations*, Proc. Combust. Inst. 32:527 | abstract-only | Mechanism reduced locally on the fly | Review III |
| III-17 | Contino et al. 2011, *Coupling of in situ adaptive tabulation and dynamic adaptive chemistry*, Proc. Combust. Inst. 33:3057 | abstract-only | ISAT with dynamic adaptive chemistry, up to 857 species, about 10x | Review III |
| IV-15 | Fush, Bonilla, Schroeder, Yao, Mueller 2026, *Neural-ISAM: a hybrid in-situ machine learning approach for complex manifold-based combustion models in LES of turbulent flames*, https://arxiv.org/abs/2605.10028 | abstract-only | Mature subtrees of the ISAT tree are replaced by networks trained in situ; the abstract states no error control | Review IV (component 6, only after the table is trusted) |
| IV-18 | Ren, Liu, Lu, Lu, Oluwole, Goldin 2014, *The use of dynamic adaptive chemistry and tabulation in reactive flow simulations*, Combust. Flame 161(1):127-137, doi:10.1016/j.combustflame.2013.08.018 | abstract-only | ISAT alone about 100x on methane/air PaSR; ISAT with dynamic adaptive chemistry adds more | Review IV |
| IV-19 | Hiremath, Ren, Pope 2011, *Combined dimension reduction and tabulation strategy using ISAT-RCCE-GALI for the efficient implementation of combustion chemistry*, Combust. Flame 158:2113-2127 | abstract-only | Dimension reduction (RCCE) combined with ISAT | Review IV |

## Chemistry reduction and astrochemistry surrogates

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| III-18 | Maas, Pope 1992, *Simplifying chemical kinetics: intrinsic low-dimensional manifolds in composition space*, Combust. Flame 88:239 | unverified | ILDM: slow manifolds | Reviews III, IV (low Jacobian rank of the step map) |
| III-19 | Lam, Goussis 1994, *The CSP method for simplifying kinetics*, Int. J. Chem. Kinet. 26:461 | unverified | Computational singular perturbation: fast/slow mode splitting | Reviews III, IV |
| III-20 | Sutherland, Parente 2009, *Combustion modeling using principal component analysis*, Proc. Combust. Inst. 32:1563, https://doi.org/10.1016/j.proci.2008.06.147 | abstract-only | Principal-component transport | Review III; Synthesis step 5 |
| III-21 | Holdship, Viti, Haworth, Ilee 2021, *Chemulator: fast, accurate thermochemistry for dynamical models through emulation*, A&A 653:A76, https://arxiv.org/abs/2106.14789 | yes | Neural emulator of UCLCHEM, about 5e4x; large errors for some models | Review III |
| III-22 | Grassi et al. 2022, *Reducing the complexity of chemical networks via interpretable autoencoders*, A&A 668:A139, https://arxiv.org/abs/2104.09516 | yes | 29 species to 5 latent species, 65x | Review III; Synthesis step 5 |
| III-23 | Sulzer, Buck 2023, *Speeding up astrochemical reaction networks with autoencoders and neural ODEs*, NeurIPS ML4PS workshop, https://arxiv.org/abs/2312.06015 | yes | Latent neural ODE 55x; linear latent model up to 4000x | Review III |
| III-24 | Janssen, Sulzer, Buck 2024, *CODES: benchmarking coupled ODE surrogates*, https://arxiv.org/abs/2410.20886 | yes | Benchmark of 4 architectures including osu2008; mean relative error below 4% | Reviews III, IV (benchmarks) |
| III-25 | Maes, De Ceuster, Van de Sande, Decin 2024, *MACE: a machine learning approach to chemistry emulation*, ApJ, https://arxiv.org/abs/2405.03274 | yes | Autoencoder and latent ODE for circumstellar envelopes, about 26x | Review III |
| III-26 | Branca, Pallottini 2023, *Neural networks: solving the chemistry of the interstellar medium*, MNRAS, https://arxiv.org/abs/2211.15688 | yes | Physics-informed network, 9 species, about 10% error | Review III |
| III-27 | Branca, Pallottini 2024, *Emulating the interstellar medium chemistry with neural operators*, A&A 684:A203, https://arxiv.org/abs/2402.12435 | yes | DeepONet, 9 species, about 1e-2 | Review III |
| III-28 | Ono, Sugimura 2025, *Neural-network chemical emulator for first-star formation*, ApJ 996:9, https://arxiv.org/abs/2508.16114 | yes | Primordial chemistry, 6 species; below 10% relative error in over 90% of cases | Reviews III, IV (baseline for the primordial benchmark) |
| III-29 | Palud et al. 2023, *Neural network-based emulation of interstellar medium models*, A&A 678:A198, https://arxiv.org/abs/2309.01724 | yes | Meudon PDR emulator, mean 4.5% | Review III |
| IV-16 | Grassi et al. 2014, *KROME - a package to embed chemistry in astrophysical simulations*, MNRAS (439:2386 unverified), https://arxiv.org/abs/1311.1070 | abstract-only | Generates network solvers to embed in simulation codes; source of the primordial H/He/H2 benchmark | Review IV (benchmarks) |
| IV-17 | Smith et al. 2017, *Grackle: a chemistry and cooling library for astrophysics*, MNRAS 466:2217, https://arxiv.org/abs/1610.09591 | abstract-only | Non-equilibrium primordial H, D, He chemistry and cooling | Review IV (benchmarks) |
| IV-21 | Li, Zhao, Kazakov, Dryer 2004, *An updated comprehensive kinetic model of hydrogen combustion*, Int. J. Chem. Kinet. 36:566-575 | unverified | A standard H2-air mechanism, about 9 species | Review IV (benchmarks) |

## ODE flow-map surrogates and stiff test problems

| ID | Reference | Verified | Takeaway for this project | Where used |
|---|---|---|---|---|
| IV-22 | Robertson 1966, *The solution of a set of reaction rate equations*, in Numerical Analysis: An Introduction, Academic Press | unverified | Robertson's 3-species stiff problem | Review IV (first chemistry benchmark) |
| IV-23 | Hairer, Wanner 1996, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, Springer | unverified | Standard stiff test set, including Robertson | Review IV |
| IV-24 | Davis, Skodje 1999, *Geometric investigation of low-dimensional manifolds in systems approaching equilibrium*, J. Chem. Phys. 111:859 | unverified | A model problem with a known slow manifold | Review IV (test T1) |

The flow maps of the bench (Lorenz-96, Lotka-Volterra) are defined in `src/structure/problems.py` and
cite no work.
