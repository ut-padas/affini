# A theory of surrogates for stiff ODE step maps: candidate foundations

This memo collects candidate theorems on which a surrogate scheme for the step maps of stiff ODEs could be built,
each with its assumptions, its proof (or the classical result it rests on), what it predicts, and a numerical check
of that prediction. It is written for the general autonomous system $x' = F(x)$; chemistry networks are one
instance. The theorems were chosen to explain what the benchmark and the convergence studies measured
(`docs/SURROGATE_RESEARCH.md`, `docs/SCHEMES_VS_ART.md`, both companions' `convergence.json`), and each is kept to
the complexity those measurements need. Revision 5 answers seven review rounds and a final external one (round 8;
`theory/REVIEWS.md`).

- **Status legend.** *Proved*: a complete proof is given here. *Proved, citing*: complete given a classical result
  marked **[C]**. *Conditional*: proved under a stated hypothesis not verified for the systems in question. *Sketch*: a
  step is not written out. *Conjecture*: stated, not proved.
- **Kinds of evidence.** *Verified bound*: a bound computed from validated enclosures. *Estimate*: a bound evaluated
  with sampled constants (maxima over finite sets, finite differences); it can be optimistic. *Observation*: a measured
  quantity. No number in this memo is a verified bound; the checks report estimates and observations.
- **Checks.** `theory/checks/t*.py` (NumPy and SciPy only, each under three minutes) write `theory/checks/data/t*.json`;
  the reviews' tests are in `theory/reviews/scratch-round-<n>/` ("review 1 test A", ...). Benchmark numbers come from the
  archived records under `results/benchmark/runs/` (seed 1 unless stated), not from the prose of the research log. A check
  that could not have failed is labelled a sanity check.
- **Prior art.** `theory/LITERATURE.md` (IDs `TH-*`); classical results under "References". References marked "(from
  memory)" were named by a reviewer and have not been read.
- **Scores and terms.** States of chemistry systems are in log10 abundances, errors in dex (decades). A worst error is the maximum
  over the problem's *scored* outputs (the benchmark's mask, which leaves out trace species below a floor) and over the states or steps
  named. A surrogate *holds* a tolerance when that worst error is within it. $\Phi$ is the exact step; every numerical reference uses
  the integrator and tolerances of its own experiment (Radau at the benchmark's tolerances for the chemistry step maps; DOP853 in
  `t3_lightcone`, for example), so every "exact" value carries that solver's error. A *one-step kick* perturbs a
  test state once and scores one step from it; a *rollout kicked at every step* perturbs the state before every step of a chain,
  for the surrogate and the solver alike. In the reduced surrogates, $\pi$ is the chart, $D$ the decoder, $D'$ its derivative and
  $J = D\Phi$ the step's Jacobian; outputs are clipped to a margin around the training states and projected onto the linear
  invariants (the benchmark's `Projected`).
- **Provenance.** A result is *pre-registered* when its predictions were committed before its output, *post hoc* when script and
  output arrived in one commit; post hoc observations motivate tests and are not conformance tests. Benchmark records live in the
  local store `results/benchmark/runs/`, which is not in git; a claim citing a record is reproducible only with that store.

  | Evidence | Predictions committed | Output committed | Status |
  |---|---|---|---|
  | T1–T7 checks (`checks/t*.py`) | with the output | `d0576af` (later edits `ffa0df0`) | post hoc |
  | Section F (`functions/f*.py`) | with the output | `a7e135a` | post hoc |
  | Demonstration 8.3 (`demos/robertson_leaf.py`) | with the output | `bc45824` | post hoc |
  | Demonstration 8.4 (`demos/reduced_dims.py`) | with the output | `0ab7a7e` | post hoc |
  | Demonstration 8.5 (`demos/curved_lift.py`) | `8c1720c` | `50f2f2f` | pre-registered |
  | G1 (`gradients/g1_static.py`) | `d0e5e27` | `9d0a142` (with the disclosed slope-fit fix) | pre-registered |
  | G2, G3 (`g2_hires.py`, `g3_hires_grid.py`) | `8f80735`, `e3997ea` | `e3997ea`, `6ea87a9` | pre-registered |
  | The HIRES recipe R1–R3, R4–R5 (`g4_hires_recipe.py`) | `3ce981b`, `6c92037` | `949ddf0` | pre-registered; R4–R5 designed after the first run |
  | P1–P4, Q1–Q4 (`g5_hires_selection.py`, `g6_hires_off_trajectory.py`) | `9d3156c`, `479f12b` | `479f12b`, `ac93af9` | pre-registered; P1–P2 informed by review 7's exploration |
  | Review tests (`reviews/scratch-round-<n>/`), `g4_diagnosis.py`, `g7_value_fit_kicks.py` | none | with the review or the script | post hoc |

**Notation.** $\Lambda$ the log-norm integral of T1; $\mathcal{L}_N$ the least-squares Lebesgue constant; $\Pi$ the $L^2$
projector of T5; $\delta = \Psi - \Phi_\tau$ a surrogate's defect; $\nu$ the diagonal bound of (A3); $\lambda_i$ eigenvalues;
$P = D\Phi(x^*)$ at a fixed point, $P = V\,\mathrm{diag}(\lambda)V^{-1}$; $\operatorname{dist}(p, S)$ the distance to a set; $M$ the set of
states a surrogate is trained on and asked to cover, distinct from a slow manifold $S$.

## 0. What known theory already predicts in our data, and what it does not

| Quantity (source) | Classical prediction | Measured |
|---|---|---|
| ART's evaluations against $1/\alpha_{\mathrm{tol}}$, Rosenbrock, $d = 2..5$ (adaptive `convergence.json`, `rates.art`) | piecewise-affine approximation of a function with a full-rank Hessian: slopes $d/2 = 1, 1.5, 2, 2.5$ | $1.00, 1.52, 2.10, 2.60$ |
| ART on a ridge, $d = 2, 3, 5$ (global `convergence.json`, `ridge_rates`) | an exact ridge needs only a 1-D piecewise-linear profile: slope $1/2$ | $0.541, 0.546, 0.816$ ($d = 5$ deviates) |
| ChebTT-M's degree per decade on the ridge (`ridge_degree`) | $1/\log_{10}\rho$ of the ridge profile | predicted $12.0, 14.6, 19.0$; measured $11.9, 14.6, 19.1$ |

The first two rows are consistent with optimal-mesh theory (Chen, Sun, Xu 2007, for piecewise-linear interpolation on
anisotropic meshes); that theory does not establish ART's oracle counts. Observations the memo sets out to explain:

- **U1.** Step maps over a longer step are harder: the 4-D Brusselator's archived short-step ChebTT box run used 3,782 solves;
  a long-step variant (scratch, `SURROGATE_RESEARCH.md` §1.20) needed about 11k.
- **U2.** Rollouts on Robertson (archived records, seed 1): ChebTT's rollout over (one-step worst $\times A_n$) is 0.38–0.73; the
  rollout over the worst error surviving one exact step is 18–30.5 for four configurations and 5.4 for a fifth (SlowTable at 0.001); SlowTable at 0.01 has one-step
  worst 0.040 dex and survived worst 0.00099 dex (0.1 of the tolerance). A separate experiment (§1.12) saw ChebTT's rollout
  reach about 36 times its one-step error in 40 steps.
- **U3.** The Robertson step map in log coordinates has singular values about $1.7, 1.0$ and a third at the solver's resolution
  ($10^{-10}$ in §0.4; the exact flow's derivative is invertible).
- **U4.** ChebTT's cost on `lotka_volterra` at $d = 20$ varied 26-fold between seeds in a scratch comparison (3,221 to 85,206),
  driven by checks at the box's vertices; without them, 3.2k–10.4k (`SCHEMES_VS_ART.md`).
- **U5.** ART with one quadratic leaf fails at the box's vertices (2.7 tol) in one case; in another, three of six refits fail.
- **U6.** Vector outputs: per-output tensor trains cost about $m$ times one output; ART reuses each sample for every output.
- **U7.** No acceptance test in the schemes is a certificate: a quadratic prefit passed validation at 0.04 and scored 4.86.

## 1. Setting and tools

$x' = F(x)$ on $\Omega \subset \mathbb{R}^d$; $x(t; x_0)$ the solution; $\Phi_\tau(x_0) = x(\tau; x_0)$; a box $B = \prod_j[a_j, b_j]$
($a_j < b_j$) with half-widths $h_j$; a surrogate $\Psi$; chains $x_{k+1} = \Phi_\tau(x_k)$, $y_{k+1} = \Psi(y_k)$, $y_0 = x_0$,
$e_k = y_k - x_k$.

**Logarithmic norms [C: Dahlquist 1958; Söderlind 2006].** $\mu(A) = \lim_{h\to0^+}(\lVert I + hA\rVert - 1)/h$.
(L1) $\mu$ is sublinear, $|\mu(A) - \mu(B)| \le \lVert A - B\rVert$. (L2) In $\lVert v\rVert_\beta = \max_j|v_j|/\beta_j$:
$\mu_\beta(A) = \max_i(\operatorname{Re}a_{ii} + \sum_{j\ne i}|a_{ij}|\beta_j/\beta_i)$. (L3) If $w' = A(t)w$, then
$\lVert w(t)\rVert \le \lVert w(0)\rVert e^{\int_0^t\mu(A)}$. (L4) $\mu(\int_0^1A) \le \int_0^1\mu(A)$. (L5) $\mu(A) \ge \max_i\operatorname{Re}\lambda_i(A)$;
a non-normal $A$ can have a large $\mu$ despite large negative eigenvalues.

**Bernstein ellipses and Chebyshev series [C: Trefethen, ATAP, Thms 4.2, 8.1, 8.2, 8.3, 15.2].** $E_\rho([a,b])$ is the open
ellipse with foci $a, b$ and semi-axes $h(\rho\pm\rho^{-1})/2$; every point of it lies within $h(\rho - \rho^{-1})/2$ of $[a, b]$; the
distance from $[a, b]$ to its boundary is $h(\rho - 1)^2/(2\rho)$, at the foci. If $f$ is analytic in the polyellipse $E_\rho$ with
$|f| \le M$: $|c_k| \le 2^{|\operatorname{supp}k|}M\rho^{-k}$ and $\lVert f - I_nf\rVert_{\infty,B} \le 2\sum_{k\not\le n}|c_k|$. In one variable,
$\limsup_nE_n(f)^{1/n} = 1/\rho_*$ with $\rho_*$ the largest ellipse of analyticity (Bernstein's converse; a root rate, without a
two-sided prefactor). The Lebesgue constant of $n + 1$ Chebyshev points is at most $\frac{2}{\pi}\log(n + 1) + 1$.

---

## F. Static functions first: scaling laws on the function suite

Before step maps, the same ideas were tested on the benchmark's static test functions, where an evaluation costs microseconds and
the singularities are known in closed form. These are tier-1 tests: each prediction is a scaling law (an exponent, a rank, a count)
derived from a stated mechanism, judged by whether the trend matches, not by constants. Scripts and outputs: `theory/functions/f*.py`,
`theory/functions/data/f*.json` (fresh ChebTT runs at small $d$, and the benchmark's records; the records' `ranks` field is the MPI rank
count, not a TT rank, so ranks come from fresh runs); a review's own tests are in `theory/reviews/scratch-round-5-fable/`. Each row says
what kind of check it is: a **test** (it could have failed), a **bound** (an "at most" that the data could not exceed), an **identity**
(true by construction), or a **record** (an observation in the benchmark's store). The predictions and their outcomes were committed
together; the next pass commits the predictions first.

| Class (problem) | Mechanism | Prediction | Measured | Kind; verdict |
|---|---|---|---|---|
| Smooth, separable (`genz_product_peak`, $d = 2$; three separable functions at $d = 3$, review 5 test 1) | poles at $w_i \pm i/a_i$ give a Bernstein $\rho_i$ per axis (§1) | TT rank 1; the degree in $x_i$ grows by $1/\log_{10}\rho_i$ per decade | rank 1 everywhere; $d = 2$: 3.27 and 4.30 against 3.15 and 4.66; $d = 3$: ratios 0.80–1.28 over nine axes, with the axes' order and a tenfold range of rates right | test; confirmed within about 30% |
| Kink (the profile $e^{-a\lvert x - w\rvert}$ of `genz_continuous`) | Chebyshev coefficients with an oscillatory $O(k^{-2})$ envelope across a kink | interpolation error $\propto n^{-1}$ | slope $-0.91$ | test; confirmed |
| Separable kinks, ChebTT with breakpoints (`genz_continuous`, $d = 2$; $d = 3, 4$ in review 5 test 3) | each piece entire, TT rank 1 | cost nearly flat in the tolerance | rank 1, 2 pieces per axis; 1,826–3,085 evaluations from $10^{-1}$ to $10^{-6}$ at $d = 2$, 2,084–4,003 at $d = 3$, 3,839–6,293 at $d = 4$ | test; confirmed. Without breakpoints the build exceeded 20,000 evaluations at a tolerance of 0.1 (review 5 G3): ChebTT's acceptance rule, not the $n^{-1}$ law, which predicts a cheap build there |
| Aligned kinks and jumps in high $d$ (`genz_continuous`, `genz_discontinuous`) | as above | ChebTT with breakpoints holds them at every $d$ | held at every $d$ to 20 (1,897 → 21,776 and 2,063 → 7,748 evaluations) | record; consistent |
| Oblique and curved jumps (`oblique_jump`, `ball_jump`) | no axis breakpoint represents an oblique interface; its TT rank is of order $n$ (`docs/LITERATURE.md`, Review VIII) | ChebTT fails; an interface term (ChebTT-A) holds | ChebTT-A held `oblique_jump` at every $d$ to 20 and `ball_jump` to $d = 5$; ChebTT's "0 held" are four aborted runs with no recorded cause, and ChebTT-M held `oblique_jump` at $d = 3$ (102,808) but not at $d = 2$ | record; the interface term's success is confirmed, the rank mechanism for ChebTT's failure untested |
| Sharp front on a sphere (`front`: $\tanh(20(\lVert x\rVert_2 - 1))$ on $[-2, 2]^d$) | analytic near the unit sphere with a nearby complex singularity across a curved surface, but not analytic on the box: at the origin the norm's corner gives one-sided slopes $\pm20\,\mathrm{sech}^2(20) \approx \pm3.4\times10^{-16}$, below rounding (so exact-arithmetic rates do not apply, the measured ones do); $\lvert x\rvert$ concentrates in high $d$ | ChebTT costly, ChebTT-A cheaper; in high $d$ the test set never sees the front | ChebTT 53,742 and 1,225,790 at $d = 2, 3$; ChebTT-A 6,889–10,404 to $d = 5$; at $d = 20$ one quadratic leaf "holds" with error ratios $10^{-8}$–$10^{-13}$ (the function is constant on the test set); at $d = 10$ the hub sees the front (ratio 0.88 at tolerance 0.05) | record; consistent (the artifact at $d = 20$ only) |
| Two shared oblique coordinates (`shared_latent`) | Jacobian rank 2 everywhere: coordinate count 2 | rotated models win; axis-aligned ChebTT does not | local rank 2 at $d = 5, 10$; ChebTT 1,792 → 376,037 ($d = 5$) → fails at 20; the hub 2,400–20,000; ChebTT-M 1,816 → 1,591,796 (it wins but does not scale mildly; tolerances loosen with $d$) | test and record; confirmed for the hub |
| Full-rank vector outputs (`rosenbrock_gradient`, `lotka_volterra_state`) | no reduction available from the Jacobian | a rotation gains nothing | evaluations with and without the rotation equal for `rosenbrock_gradient` and within 0.6% for `lotka_volterra_state` | record; confirmed |
| An oblique ridge with a trigonometric profile (`wave_pair`) | Jacobian rank 1 | *predicted:* axis-aligned ChebTT pays for the obliqueness | ChebTT cheap at every $d$ (517 → 5,424): $\cos, \sin$ of a linear form have TT rank at most 2 | test; **failed** |
| Shared parameters (`genz_triple`) | three outputs of different structure | *predicted:* full local rank | local rank 2 of 3: the oscillatory and corner-peak outputs share the direction $a$ | test; **failed** |
| A rational oblique ridge (review 5 test 2: $1/(b^2 + (q\cdot x)^2)$, $q$ random) | the general form of the corrected claim below | TT rank grows with the tolerance and the profile's sharpness, not with $d$ | rank 4 → 12 ($b = 0.7$, $d = 2$) and 6 → 18 at both $d = 3$ and 5; evaluations grow about as $d\,n\,r^2$ until over the cap | test; confirmed |
| Smooth, a cosine of a sum (`genz_oscillatory`) | $\cos(c + a\cdot x) = \operatorname{Re}\prod_je^{ia_jx_j}e^{ic}$ | TT rank 2 | 2 at every tolerance and $d$ | identity |
| A function of one aggregate (`genz_corner_peak`) | Lemma 7.2 with $\rho_s = 3.91$ | rank growth at most 1.69 per decade | ChebTT 0.7–1.0 per decade (quantised by its rank step of 2); the true ranks 0.65 per decade by SVD (review 5 test 5) | bound; not informative (2.6 times loose) |
| Jump, one piece / with the breakpoint | no polynomial is uniformly within half a jump; pieces analytic | error at least half the jump; geometric with the breakpoint | 0.71–1.12 against 0.57; $10^{-12}$ | identity (a theorem; the breakpoint is placed at the known $w$) |
| Aggregates (`lotka_volterra`, $d = 20$) | one dominant gradient direction; the uniform measure concentrates it | a few aggregate coordinates suffice on most of the box and fail at the vertices | eigenvalues 1, 0.0375, 0.0037; fits in 1–3 directions err 2.3–10% of the range on uniform points, 4–12 times more at vertices; but a full 20-coordinate quadratic fitted the same way shows the same ratio (8.4; review 5 G1) | test; **inconclusive**: the design cannot tell the aggregate mechanism from extrapolation |

**The corrected claim (review 5 G11).** Lemma 7.2 already implies it: a function that is a polynomial of degree $n$ in $k$ oblique coordinates
has TT ranks at most $\binom{n + k}{k}$ in any ordering, whatever $d$ (it separates across each cut), so an axis-aligned train has about
$d\,n\binom{n + k}{k}^2$ parameters at $n$ modes per axis (a representation size; a cross algorithm's oracle count needs its own argument). The
coordinate count $k$ is what a rotated model needs; ChebTT's cost is set by the rank of the profile in the given axes. This covers
`wave_pair` (a profile of rank 2), the corner peak and ChebTT's failure on `shared_latent` ($k = 2$, high degree); for a random unit $q$
on $[-1, 1]^d$ the range of $q\cdot x$ grows like $\sqrt{2d/\pi}$, so the rate does depend on $d$ unless the direction is scaled.

What this pass establishes. Where a function's analytic structure is known, the Chebyshev and TT mechanisms predict ChebTT's degrees
(within about 30%, axis order right) and ranks, the difference breakpoints make for kinks, and when a rotation pays; the interface term's
success on oblique jumps is consistent. Two predictions failed (`wave_pair`, `genz_triple`), both avoidable from the memo's own lemmas; one
test was inconclusive (aggregates). Four rows are identities or bounds, not tests. T5 (least-squares leaves) is not tested here: that
least-squares methods held none of the kink cases is not specific, since they also hold none of several smooth vector problems.

---

## T1. Analyticity radii of the step map from the vector field

**Question.** Chebyshev and tensor-train rates depend on the Bernstein parameter $\rho$. For a step map $F$ is known: can
$(\rho, M)$ be bounded *a priori*, robustly to stiffness? The classical complex Gronwall argument uses the Lipschitz constant $L$
and gives radii like $e^{-L\tau}$; the logarithmic norm is one-sided.

**Assumptions.** (A1) The real solutions exist for $x_0 \in B$, $t \in [0, \tau]$; $X$ is the set they visit. (A2) For weights
$\beta > 0$, $F$ extends holomorphically to an open neighbourhood of $T_\beta = \{x + v: x \in X, v \in \mathbb{C}^d, \lVert v\rVert_\beta \le 1\}$.
$m(t) = \sup\{\mu_\beta(J_F(x(t; x_0) + v)): x_0 \in B, \lVert v\rVert_\beta \le 1\}$, $\Lambda = \max_{t\le\tau}\int_0^tm \ (\ge 0)$.

**Lemma 1.1 (tube lemma; proved).** For $x_0 \in B$ and $\lVert w_0\rVert_\beta < e^{-\Lambda}$ the complex solution from $x_0 + w_0$
exists on $[0, \tau]$ and $\lVert z(t) - x(t; x_0)\rVert_\beta \le \lVert w_0\rVert_\beta e^{\int_0^tm} < 1$.
*Proof.* Let $t^*$ be the first time $\lVert w\rVert_\beta = 1$ or the solution ends. On $[0, t^*)$, $w' = A(t)w$ with
$A = \int_0^1J_F(x + sw)\,ds$, so (L3)–(L4) give $\lVert w\rVert_\beta \le q := \lVert w_0\rVert_\beta e^\Lambda < 1$; the solution stays in a
compact subset of the tube where $F$ is bounded, so it continues past $t^*$; hence $t^* > \tau$. $\square$

**Lemma 1.2 (componentwise tube; proved).** For each $x_0 \in B$ let $r(\cdot; x_0): [0, \tau] \to \mathbb{R}^d_{>0}$ be $C^1$ with, for
all $t$ and $|v_j| \le r_j(t; x_0)$, $J = J_F(x(t; x_0) + v)$: $\operatorname{Re}J_{ii}r_i + \sum_{j\ne i}|J_{ij}|r_j \le r_i'$ ($F$ holomorphic
near that tube). Then $|w_{0,j}| \le \theta r_j(0; x_0)$, $0 \le \theta < 1$, implies existence on $[0, \tau]$ and
$|w_j(t)| \le \theta r_j(t; x_0)$. If $\underline r_j := \inf_{x_0}r_j(0; x_0) > 0$, Theorem 1 holds with $\theta\underline r_j$ in place of
$\theta\beta_je^{-\Lambda}$ and $M_i = \sup_{x_0}(|x_i(\tau; x_0)| + \theta r_i(\tau; x_0))$, provided this supremum is finite (a uniform bound
on the terminal radii).
*Proof.* $w = 0$ stays 0 by uniqueness. Otherwise let $u = \max_j|w_j|/r_j$; at an active index $i$,
$D^+(|w_i|/r_i) \le u[\operatorname{Re}A_{ii}r_i + \sum_{j\ne i}|A_{ij}|r_j - r_i']/r_i \le 0$ (the row condition is convex in $J$, so it holds for
the mean $A$); the Dini derivative of a finite maximum is bounded by the active indices; first exit and continuation as in
Lemma 1.1 (Kamke–Müller comparison, Walter 1970, on a complex tube). $\square$

**Theorem 1 (proved, citing ATAP and [C: Ilyashenko–Yakovenko, Thm 1.1]).** For $\theta \in (0, 1)$ (the error bound divides by $\rho_j - 1$) let
$h_j(\rho_j - \rho_j^{-1})/2 = \theta\beta_je^{-\Lambda}$. Each $\Phi_{\tau,i}$ extends holomorphically to $E_\rho$ with
$|\Phi_{\tau,i}| \le M_i := \max_B|x_i(\tau)| + \theta\beta_i$, so $|c_k| \le 2^{|\operatorname{supp}k|}M_i\rho^{-k}$ and

$$\lVert\Phi_{\tau,i} - I_n\Phi_{\tau,i}\rVert_{\infty,B} \le 4M_i\sum_j\frac{\rho_j^{-n_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1}.$$

*Proof.* Every point of $E_\rho$ is $x_0 + w_0$ with $x_0 \in B$, $\lVert w_0\rVert_\beta < e^{-\Lambda}$ (a boundary point $c + A\cos u + ib'\sin u$ with
$|A\cos u| > h$ is within $b'$ because $(A - b')|\cos u| = h\rho^{-1}|\cos u| \le h$). Lemma 1.1 and [C] give holomorphy, with
$|z_i(\tau)| \le |x_i(\tau)| + \beta_i\theta e^{\int_0^\tau m - \Lambda}$. The error sums $\prod_jw_j(k_j)$ ($w(0) = 1$, $w(k) = 2\rho^{-k}$) over
$k \not\le n$. $\square$ The degree it implies for a tolerance is a sufficient bound and keeps $M_i$.

**Proposition 1.3 (a pointwise ceiling on certificates; proved).** If $\Phi_{\tau,i}$ is holomorphic on $E_\rho$ with
$|\Phi_{\tau,i}| \le M$, then $\operatorname{dist}(x_j, \partial E_{\rho_j}) \le M/|\partial_j\Phi_{\tau,i}(x)|$ for $x \in B$ (Cauchy on the disc centred at
$x_j$). For fixed interior $x_j = c_j + sh_j$ and $\rho_j - 1 \ll 1$ the left side is $h_j\ln\rho_j\sqrt{1 - s^2}(1 + o(1))$. It constrains
certificates $(\rho, M)$, not the true degree.

**Corollary 1.4 (the bound's regimes; proved).** If, for tubes $\beta = r\hat\beta$, $m_r(t) \le \mu_0 + cr^p$ ($c, p > 0$,
$0 < r \le r_{\max}$), then $R(\tau) = \sup_{r\le r_{\max}}re^{-\Lambda_r}$ is at least $\min(r_{\max}, (|\mu_0|/c)^{1/p})$ for every $\tau$ if
$\mu_0 < 0$; $(pce\tau)^{-1/p}$ if $\mu_0 = 0$; $e^{-\mu_0\tau}(pce\tau)^{-1/p}$ if $\mu_0 > 0$, the last two once $(pc\tau)^{-1/p} \le r_{\max}$.
Regimes of the **bound**: a neutral linear field ($F = 0$, $c = 0$) has a $\tau$-independent certificate.

**Proposition 1.5 (composition; proved).** For entire $F$ let $D_\tau$ be the complex initial data whose solution exists on
$[0, \tau]$. Then $\Phi_\tau$ is holomorphic on $D_\tau$ and $D_{\tau+\sigma} = D_\sigma \cap \Phi_\sigma^{-1}(D_\tau) \subset D_\tau$: the family of polyellipses
contained in $D_\tau$ shrinks with $\tau$, so the radii supported by complex existence alone can only shrink (a bound on $|\Phi_\tau|$ on
a fixed ellipse can still improve, as for a stable linear flow). One-sided: $\Phi_\tau$ may continue beyond $D_\tau$, and the true nearest
singularity need not approach $B$ monotonically. For the real planar polynomial system obtained from $w' = iw + w^2$ ($w = x + iy$),
$w(t) = e^{it}w_0/(1 - w_0(e^{it} - 1)/i)$: the time-$\pi$ map has a pole at finite complex initial data, while the time-$2\pi$ map is the
identity on its existence domain and continues to an entire map (review 8). Monotonicity for a stated class (the Brusselator; U1) is open.

**Remark 1.6 (linear invariants).** If $CF \equiv 0$, $\mu(J) \ge 0$ in every norm, so $\mu_0 \ge 0$ in full coordinates. For a fixed
tube $\Lambda$ is nondecreasing in the horizon; whether it grows depends on the weights (a conserved two-state exchange with
unequal rates has $\mu > 0$ with equal weights and $\mu = 0$ with weights in the ratio of the rates). Reduced Robertson has
$\inf_\beta\mu_\beta = -0.0089$ at $y(100)$ (review 2 test D); no $\tau$-uniform tube was found on boxes of $\pm10\%$ or $\pm50\%$.

**Checks (estimates).** `t1_scalar` ($x' = \mp x - x^3$, exact maps): $\rho_{\mathrm{thm}} \le \rho_{\mathrm{exact}}$ at every $\tau$; dissipative
$\rho_{\mathrm{thm}} = 1.731$ for $\tau \ge 1$ against $\rho_{\mathrm{exact}} = 2.544, 2.430$ at $\tau = 1, 2$ tending to 2.414; expanding: slope of
$\log(\rho - 1)$ $-1.14$ against $-1.03$; resolved coefficients below $2M\rho_{\mathrm{thm}}^{-k}$ (ratio at most 0.56). Review 1 test B
(Robertson, own coordinates, box $\pm50\%$ of $y(100)$, sampled starts, times and boundary phases, $\beta$ optimised): $\Lambda = 0.04$–$1.75$
for $\tau|\lambda_{\mathrm{fast}}|$ from 42 to 42,000; with $\beta \propto h$, $10^2$–$10^5$; the proxy degree $\log(10^3)/\log\rho$ is 19–202 against
measured 4.4–5.0. Review 2 test C (4-D Brusselator, entrywise bounds around sampled trajectories): proxy degrees $\log(100)/\log\rho$
of 1,272–32,236 against measured 2–4. Review 2 test D: Lemma 1.2 with entrywise majorants does not beat Theorem 1 on reduced
Robertson. Review 1 test A: fibre-coefficient slopes do not discriminate the regimes for $\tau \in [0.25, 4]$.

**U1** is not explained by T1. Review 2 test C reproduces its direction on a grid (per-axis degree (5, 5, 3, 3) to (7, 7, 4, 4) and
mid-unfolding count 4 to 6 at 0.01 dex when $\tau$ doubles; a cross-cost proxy larger by $1141/462 \approx 2.47$; a sampled proxy, not
an oracle count).

**Prior art.** Holomorphic dependence, the log-norm Gronwall bound, componentwise comparison and the Chebyshev estimates are
classical; Hansen–Schwab (TH-1) works in affine parameters with Lipschitz constants; Carleman-linearization bounds (Forets–Pouly;
Liu et al. 2021; abstracts only) treat the dissipative quadratic case; CAPD propagates real enclosures with log norms (unverified).
What is put together here is their combination on a complex tube for stiff step maps.

**Limits.** Informative only on small boxes; vacuous on the Brusselator's; weights must be optimised; no tube computation in log
coordinates; the product in Theorem 1 is exponential in $d$; no computation here is a validated enclosure.

---

## T2. How a surrogate's defect accumulates

**Proposition 2.1 (global bound; classical).** Let $C$ be convex and forward invariant up to $\tau$, $\mu(J_F) \le m$ on $C$,
$\Psi(C) \subset C$, $\sup_C\lVert\delta\rVert \le \varepsilon$, $x_0 \in C$. Then $\lVert e_n\rVert \le \varepsilon\sum_{j<n}e^{jm\tau}$ (one-step recurrence
$\lVert e_{k+1}\rVert \le e^{m\tau}\lVert e_k\rVert + \varepsilon$). With an invariant, $m \ge 0$.

**Proposition 2.2 (first-order representation; proved).** If $\Phi$ is $C^2$ on the segments $[x_k, y_k]$ with $\lVert D^2\Phi\rVert \le K_2$
there and $G_{n,j} = D\Phi(x_{n-1})\cdots D\Phi(x_j)$ ($G_{n,n} = I$): $e_n = \sum_{k=0}^{n-1}G_{n,k+1}\delta(y_k) + R_n$ with
$\lVert R_n\rVert \le \frac{K_2}{2}\sum_{k<n}\lVert G_{n,k+1}\rVert\lVert e_k\rVert^2$ (Taylor and unrolling). *Remark (hyperbolic fixed points,
classical).* At a fixed point of the reduced system with $I - P$ invertible and $\delta$ small in $C^1$ on a neighbourhood,
$y^* - x^* = (I - P)^{-1}\delta(x^*) + E$, $\lVert E\rVert \lesssim \lVert(I - P)^{-1}\rVert^2\lVert D\delta\rVert\varepsilon + \lVert(I - P)^{-1}\rVert^3K_2\varepsilon^2/2$; not
applicable to Robertson (singular reduced Jacobian at its equilibrium).

**Proposition 2.3 (neutral linear dynamics; proved).** Let $\Phi(x) = Ux$ with $U$ orthogonal, in the Euclidean norm (or any
$U$-invariant norm), let the defect satisfy $\lVert\delta(y_k)\rVert \le \varepsilon$ on the surrogate chain's own inputs, and let $\delta$ be
$\mathrm{Lip}(\delta)$-Lipschitz on the segments $[x_k, y_k]$. Then $e_n = U^{n-1}S_n + R_n$ with $S_n = \sum_{k<n}U^{-k}\delta(U^kx_0)$ and
$\lVert R_n\rVert \le \mathrm{Lip}(\delta)\varepsilon n(n - 1)/2$ (from $\lVert e_k\rVert \le k\varepsilon$, which uses the defect bound on the chain and the isometry; a
bound on the exact inputs alone gives no such estimate). When the pulled-back orbit average $\bar\delta$ exists, $S_n = n\bar\delta + o(n)$;
if the pulled-back sequence $U^{-k}\delta(U^kx_0)$ is a finite Fourier sum in $k$ with zero mean and no resonant (zero-frequency) mode,
$S_n$ is bounded (geometric sums). $\lVert e_n\rVert \approx \lVert S_n\rVert$ only while $R_n$ is small against $\lVert S_n\rVert$.

**Proposition 2.4 (normal stability; the organising statement; proved).** Let $S$ be compact with $\Phi(S) \subset S$ and
$U_r = \{p: \operatorname{dist}(p, S) \le r\}$. If $\operatorname{dist}(\Psi(p), S) \le \kappa_\Psi\operatorname{dist}(p, S) + \varepsilon_S$ on $U_r$ with
$0 \le \kappa_\Psi < 1$, $\varepsilon_S \ge 0$, $\varepsilon_S \le (1 - \kappa_\Psi)r$, then $\Psi(U_r) \subset U_r$ and
$\operatorname{dist}(y_n, S) \le \kappa_\Psi^n\operatorname{dist}(y_0, S) + \varepsilon_S/(1 - \kappa_\Psi)$. Sufficient cases (triangle inequalities): with the
exact map's contraction $\operatorname{dist}(\Phi(p), S) \le \kappa\operatorname{dist}(p, S)$ on $U_r$, (a) $\sup_{U_r}\lVert\delta\rVert \le \varepsilon_S$ gives
$\kappa_\Psi = \kappa$; (b) $\sup_S\lVert\delta\rVert \le \varepsilon_S$ and $\lVert\delta(p) - \delta(q)\rVert \le \ell\operatorname{dist}(p, S)$ ($q$ a nearest point) give
$\kappa_\Psi = \kappa + \ell$; and, without it, (c) $\Psi(U_r)$ within $\varepsilon_S$ of $S$ gives $\kappa_\Psi = 0$.
**Scope.** It bounds the distance to $S$ only: tangential errors are not controlled (a surrogate that projects onto an attracting
circle and rotates by the exact angle plus a bias satisfies (c) and drifts in phase by the bias every step). None of its hypotheses
is computable from a fitted surrogate and data on $S$; Theorem 8 is. *Invariants.* A linear invariant is neutral ($\kappa = 1$ along
it), so the tube is taken within a fixed conservation leaf and the surrogate must conserve the invariant on its output, in
physical concentrations: in log coordinates $C\,10^{\Psi(z)} = C\,10^z$. Feeding the invariant to the fit does not impose this (review 3
test A: such a HIRES chart diverged, its invariant drifting by 38 dex).

**Diagnostics, not theorems.** The error that survives one exact step, $\lVert\Phi(\Psi(z)) - \Phi(\Phi(z))\rVert$, is a useful diagnostic of
which defect directions the dynamics damp; it is not implied by normal stability to predict the rollout. Whether a scheme's
defects keep their sign along trajectories (the mechanism suggested for ChebTT) needs the weighted defect vectors or their
autocorrelation, which were not measured.

**Checks (observations).** Archived Robertson records (seed 1): rollout over survived worst 30.5, 24.4, 19.6, 18.0 for ChebTT box at 0.01
and 0.001, `chebtt_roll` at 0.001 (2,042 solves), SlowTable at 0.01, and 5.4 for SlowTable at 0.001; ChebTT's `rollout_over_bound`
0.38–0.73; SlowTable's deployed rollout includes solver fallbacks. Review 2 test A: a fit with one tube sample per visited state stayed
within the predicted offset of $S$ ($3.6\times10^{-3}$ against $6.3\times10^{-3}$); fits on visited states alone diverged. Sanity checks
(`t2_rollout`): the Davis–Skodje fixed point shifts $10.508\varepsilon$, $1.582\varepsilon$, $13.79\varepsilon$ for slow-mode, fast-mode and slow-variable
defects, against the classical $14.1\varepsilon$.

**Prior art.** Proposition 2.1: global error of one-step methods (TH-20); the fixed-point remark: persistence of hyperbolic fixed
points (TH-18, TH-19); attracting sets under one-step perturbations: Kloeden–Lorenz 1986, Stuart–Humphries 1996 (from memory);
Proposition 2.4 is a scalar invariant-tube argument (folklore).

---

## T3. A light cone for locally coupled systems

**Assumptions.** (A3) Along the trajectories from $B$ up to $\tau$: $\operatorname{Re}J_{ii} \le \nu$, $|J_{ij}| \le L_{ij}$ ($i \ne j$), $L \ge 0$ with
$L_{ii} = 0$; an edge $j \to i$ for each $L_{ij} > 0$; $D(i, j)$ the graph distance along such edges (from the inputs $j$ to the
output $i$); $\ell = \max L_{ij}$ and $\Delta$ the largest row count of $L$.

**Lemma 3.1 (proved).** Entrywise $|\partial\Phi_\tau/\partial x_0| \le e^{\nu\tau}\exp(\tau L)$, and $[\exp(\tau L)]_{ij} \le \sum_{k\ge D(i,j)}(\tau\ell\Delta)^k/k!$.
*Proof.* Variation of constants about the diagonal propagator; each Picard iterate is dominated by the series of $e^{\nu t}\exp(tL)$;
$(L^k)_{ij} = 0$ for $k < D(i, j)$ and $\le(\ell\Delta)^k$ (row sums). $\square$ **Corollary 3.2.** Freezing the coordinates outside
$W_r(i) = \{j: D(i, j) \le r\}$ at the box centre costs at most $e^{\nu\tau}\sum_{j\notin W_r(i)}h_j[\exp(\tau L)]_{ij} \le h^*e^{\nu\tau}\sum_{k>r}(\ell\Delta\tau)^k/k!$,
$h^* = \max_jh_j$ (the omitted inputs all lie at distance $> r$, and the row sums of $L^k$ are at most $(\ell\Delta)^k$). This is independent of $d$
when $h^*$, $e^{\nu\tau}$ and $\ell\Delta\tau$ are bounded uniformly in $d$ (a uniform $\nu$ alone is not enough if $\tau$ grows: $x_1' = x_1 + x_2/d$ with
$\tau = d$ has $\nu = 1$, $\ell\Delta\tau = 1$ and freezing error $(e^d - 1)/d$, review 9); no count of inputs per distance is needed. Without uniform scales it is not: one incoming
edge, $\ell\Delta\tau = 1$ and unit half-widths still allow $x_1' = dx_1 + x_2$ ($\nu = d$, $\tau = 1$), whose radius-zero freezing error is
$(e^d - 1)/d$ (review 8).
**Corollary 3.3.** The window function's tensor Chebyshev interpolant of degree $n$ has unfolding rank at most
$(n + 1)^{\min(|W\cap\mathrm{left}|, |W\cap\mathrm{right}|)}$ (frozen variables contribute rank one), independent of $d$ when the window size and the
degree are; on a Lorenz-96 ring $|W_r| = \min(d, 3r + 1)$.

**Scope.** $d$-independent only while $\ell\Delta\tau$ stays bounded as $d$ grows. A method-of-lines discretisation of a fixed domain has
$\ell = Dn^2$: review 1 test D (Fisher–KPP, $\tau = 0.01$) finds window radius 4, 9, 16, 31 sites for $n = 10..80$ at $10^{-3}$ and the whole
system at $10^{-5}$.

**Checks** (`t3_lightcone`, Lorenz-96, $F = 1$, $d = 40$, measured constants). At $\tau = 0.25$ the largest $|\partial\Phi_i/\partial x_j|$ falls from
0.81 at $D = 0$ to $1.9\times10^{-13}$ at $D = 11$, below the bound at every distance; the window error at $r = 4$ is $2.6, 3.2, 5.2\times10^{-5}$ and at
$r = 5$ $1.8, 1.5, 1.2\times10^{-6}$ for $d = 20, 40, 80$. The ladder's archived records (tolerance 0.01): ChebTT box Allen–Cahn 325, 530, 1,619,
17,103 ($d = 2, 3, 5, 10$), Fisher–KPP 409, 2,324, 14,358, 107,002, Lorenz-96 2,515, 7,230, 339,627 ($d = 4, 5, 10$); SlowTable Fisher–KPP 1,138,
2,560, 4,115, Allen–Cahn 686, 2,395, 4,423, Lorenz-96 7,437, 11,382. They do not test the light cone.

**Prior art.** A classical positive comparison applied to the variational equation; light cones for lattices (TH-22 to TH-26), decay
of $\exp(tA)$ (TH-28), light cone to rank for quantum propagators (TH-29) and linear dissipative ODE states (TH-30).

---

## T4. What can be certified from samples

**Proposition 4.1 (adaptive audits; proved).** Before round $r$, freeze (conditionally on the past) a candidate $s_r$, a score, a law
$\mu$ and a sample size $N_r$; draw $N_r$ points i.i.d. from $\mu$, independent of the past; accept iff every score is within the
tolerance (count exceptions, non-physical states and undefined steps as failures unless the fallback is part of the frozen map).
Then, for $0 < \alpha, \eta < 1$ and rounds $r = 1, 2, \ldots$, the probability of *ever* accepting a candidate whose violation set has
$\mu$-measure $> \eta$ is at most $\sum_r(1 - \eta)^{N_r}$, at most $\alpha$ for $N_r \ge \ln(2^r/\alpha)/(-\ln(1 - \eta))$. *Proof.* Conditionally on the past, a false pass at round $r$ has probability at
most $(1 - \eta)^{N_r}$; union bound over rounds. $\square$ (One round: Wilks 1941.) It is a statement about the procedure, not a
conditional probability given acceptance, and it is distributional: an arbitrarily large error on a set of measure below $\eta$ may be missed.
With numerical reference trajectories the tolerance must include a validated reference error.

**Corollary 4.1′ (rollout audit; proved).** With fresh initial states from the application's law as the points and the worst error over
an $n$-step rollout as the score, Proposition 4.1 bounds the probability of ever accepting a surrogate whose $n$-step rollout violates
the tolerance on a set of initial states of measure $> \eta$, at $N_rn$ solver steps. It is the distributional complement of a uniform
certificate. **From measure to sup.** If the rollout score is $L$-Lipschitz on the target set and every ball of radius $h$ centred there
has $\mu$-mass $> \eta$, then measure at most $\eta$ above $t$ implies the uniform bound $t + Lh$ (Proposition 4.2's argument); in $d$
dimensions $h \sim \eta^{1/d}$.

**Proposition 4.2 (measure to sup; proved).** With $d_B(x, y) = \max_j|x_j - y_j|/(b_j - a_j)$ and $\mu$ uniform on $B$, if $e \ge 0$ is
$L$-Lipschitz in $d_B$ and $\mu\{e > t\} < \eta \le 1$, then $\sup_Be \le t + L\eta^{1/d}$ (a $d_B$-ball of radius $h \le 1$ meets $B$ in volume at
least $h^d$). **Corollary 4.3.** For a frozen candidate and score, $N \ge 1$ points i.i.d. uniform on $B$ and $0 < \alpha < 1$, the event "every point scores within $t$
*and* $\sup_Be > t + L(1 - \alpha^{1/N})^{1/d}$" has probability at most $\alpha$ (and $(1 - \alpha^{1/N})^{1/d} \le (\ln(1/\alpha)/N)^{1/d}$). *Proof.* If
$\mu\{e > t\} \ge \eta = 1 - \alpha^{1/N}$, all points pass with probability at most $(1 - \eta)^N = \alpha$; otherwise Proposition 4.2 gives the
bound. $\square$ It is not a guarantee conditional on a pass: for $e(x) = x$ on $[0, 1]$, $N = 1$, $\alpha = 0.05$, $t = 0.01$, $L = 1$, every pass
reports 0.96 against the true supremum 1, and passes occur with probability 0.01 (review 8).
**Proposition 4.4 (sharpness; proved).** For any $N \ge 1$ points, $e(x) = L\min_id_B(x, x_i)$ is $L$-Lipschitz, vanishes at the points, and
has $\sup e \ge L/(2N^{1/d})$: the $N$ cubes of side $2h$ about the points cover $B$ only if $N(2h)^d \ge 1$, $h$ the covering radius. This
is the classical optimal-recovery bound (Sukharev 1979; Novak 1988).

**Proposition 4.5 (a certificate under analyticity; proved, citing ATAP 15.2).** Let $p$ have integer degree $n_j \le N_j$ in $x_j$ and
$(\rho, M)$ be verified as in Theorem 1. If $|\Phi_{\tau,i} - p| \le t$ at the nodes of the tensor Chebyshev–Lobatto grid of degrees $N_j$ (exact
nodal residuals, or $t$ enlarged by the reference solver's error), then
$\sup_B|\Phi_{\tau,i} - p| \le t\prod_j(\frac{2}{\pi}\log(N_j + 1) + 1) + 4M\sum_j\frac{\rho_j^{-N_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1}$.
*Proof.* $I_Np = p$, so $\Phi - p = I_N(\Phi - p) + (\Phi - I_N\Phi)$. $\square$ Elementary (validated-numerics tools do the same; Brisebarre–Joldeş
2010, unverified); with Theorem 1's present $\rho$ on the benchmark's boxes the grid is impractically large.

---

## T5. What a fit from samples does where it is used

**Question.** A surrogate fitted to samples (a tree leaf, a global polynomial, a chart fit from trajectories) is used at states it was not
fitted on: test states, states of a rollout, states moved by another process in a split simulation. What bounds its error there, and can
that bound be computed before the surrogate is used?

Setting: a space $V$ of dimension $p$ (polynomials of any degree in any coordinates, or any other finite basis $\varphi_1, \ldots, \varphi_p$),
samples $X = \{x_1, \ldots, x_N\}$ with values $f(x_i)$, $V_{ij} = \varphi_j(x_i)$ of full column rank, the least-squares fit
$\mathrm{LS}f = \varphi^{\mathsf T}(V^{\mathsf T}V)^{-1}V^{\mathsf T}f(X)$, and its weights $\ell(t) = V(V^{\mathsf T}V)^{-1}\varphi(t) \in \mathbb{R}^N$, so that
$\mathrm{LS}f(t) = \sum_i\ell_i(t)f(x_i)$. The *Lebesgue function of the sample at $t$* is $\Lambda_X(t) = \sum_i|\ell_i(t)|$; it depends on $X$, $V$ and $t$
only, not on $f$, so it is computable at any state before any value there is known. For vector outputs apply it componentwise.

**Proposition 5.1 (the error at a point of use; proved).** For every $t$ and every $q \in V$:
$|f(t) - \mathrm{LS}f(t)| \le |f(t) - q(t)| + \Lambda_X(t)\max_i|f(x_i) - q(x_i)|$; hence
$|f(t) - \mathrm{LS}f(t)| \le (1 + \Lambda_X(t))\inf_{q\in V}\lVert f - q\rVert_{\infty, X\cup\{t\}}$ and, over a target set $T$,
$\sup_T|f - \mathrm{LS}f| \le (1 + \Lambda_X(T))\inf_q\lVert f - q\rVert_{\infty, X\cup T}$ with $\Lambda_X(T) = \sup_{t\in T}\Lambda_X(t)$.
*Proof.* $\mathrm{LS}$ reproduces $V$, so $\mathrm{LS}f(t) - q(t) = \sum_i\ell_i(t)(f(x_i) - q(x_i))$. $\square$
*Bounds on $\Lambda$.* With $\varphi$ orthonormal under a measure $\mu$, $G = N^{-1}V^{\mathsf T}V$ and $K(t) = \lVert\varphi(t)\rVert_2^2$:
$\Lambda_X(t) \le \sqrt N\lVert\ell(t)\rVert_2 = \sqrt{\varphi(t)^{\mathsf T}G^{-1}\varphi(t)} \le \sqrt{K(t)/\lambda_{\min}(G)}$. For one quadratic leaf on a box ($T$ the box,
$\mu$ uniform) this is the classical Lebesgue inequality, with $\mathcal{L}_N = \Lambda_X(T)$.

*What it says, generically.* The error where a surrogate is used is the space's approximation error on the samples and the point of use,
amplified by how the samples sit relative to that point. Three regimes follow, whatever the problem: (a) target states among the samples
(a thin visited set that the samples cover): $\Lambda_X(T)$ stays moderate as $p$ grows, and the degree can be raised until the best
approximation error is small; (b) targets between sparse samples (an initial law that fills a box, states moved off the visited set):
$\Lambda_X(T)$ grows quickly with $p$, so a fit that is exact on the samples can be arbitrarily wrong at $T$, and in-sample checks cannot see
it; (c) designed samples (Chebyshev grids, the points a solver-queried scheme chooses): $\Lambda_X(B)$ is the grid's Lebesgue constant,
logarithmic in the degree per axis, so the box is covered uniformly. A rollout then multiplies the one-step error by the dynamics'
accumulation (Proposition 2.1, Theorem 8): the error after $n$ steps is at most of order $A_n\,(1 + \Lambda_X(T_n))\,e^*$, with $T_n$ the states
the rollout visits and $e^*$ the best approximation error. Every factor is computable from data: $\Lambda$ from the samples and the target
inputs, $A_n$ from Jacobians along trajectories, $e^*$ estimated by held-out residuals.

**Proposition 5.2 (the error at a point; proved).** With $r = f - \Pi f$: $f(x) - \mathrm{LS}f(x) = r(x) - \varphi(x)^{\mathsf T}G^{-1}b$,
$b = N^{-1}\sum_i\varphi(x_i)r(x_i)$, so $|f(x) - \mathrm{LS}f(x) - r(x)| \le \sqrt{K(x)}\,\lVert G^{-1}\rVert\,\lVert b\rVert$; for independent draws,
$\mathbb{E}\lVert b\rVert^2 = N^{-1}\mathbb{E}[K(X)r(X)^2] \le N^{-1}K\lVert r\rVert_\infty^2$ (since $\mathbb{E}[\varphi r] = 0$). A high-probability statement
combines this with a separate event on $\lVert G^{-1}\rVert$. ART's adaptive sampling needs its own hypotheses.

**Checks (estimates).** `t5_least_squares` (ART's $N = 20p + 1$ uniform points): $\lambda_{\min} = 0.57$–$0.67$ although $N$ is far below
Cohen–Davenport–Leviatan's sufficient size (their Thm 1; TH-40): 121 against 3,987 at $d = 2$, 1,321 against 108,292 at $d = 10$;
$\mathcal{L}_N$ between the sampled vertex estimate (4.4, 6.0, 9.7, 18.9, 34.4 for $d = 2, 3, 5, 10, 20$) and $\sqrt{K/\lambda_{\min}}$ (6.2, 9.1, 14.4,
29.2, 56.7). Review 1 test C (synthetic functions, $d = 10$): the leaf errs 0.5–1.6 times the best sup error found in the interior and 2–5.7
at the vertices, matching the $L^2$ projection's vertex error in three of four functions. This suggests a mechanism for U5 (the $L^2$
projection's own vertex error); it does not identify the cause of the specific benchmark failure.

**Prior art.** Cohen–Davenport–Leviatan (TH-40), weighted and subsampled least squares (TH-41 to TH-44).

---

## T6. Local rank and coordinate counts of a stiff step map

**Definitions.** For a tolerance $\eta$, a target set $T$ and a class of continuous encoders $c: T \to \mathbb{R}^m$ with arbitrary decoders,
the *coordinate count* $m_T(\eta)$ is the least $m$ with $\inf_{c,G}\sup_T\lVert\Phi_\tau - G\circ c\rVert \le \eta$ (norm stated). The *local rank* at
$p$ at threshold $\eta$ is $\#\{i: \sigma_i(D\Phi_\tau(p)) > \eta\}$. For nested targets and the same norm, tolerance and encoder class,
$m_{T'} \le m_T$ when $T' \subset T$. The local rank does not bound the coordinate count from above (Example 6.8); Proposition 6.5 bounds it
from below.

**Assumption (S) [C: Fenichel 1971, 1979; Hirsch–Pugh–Shub 1977; Eldering–Kvalheim–Revzen on global stable foliations].** A compact,
normally attracting, locally invariant $k$-manifold $S$, a forward-invariant neighbourhood $N$ (up to $\tau$) and a projection
$\pi: N \to S$ along stable fibres with $\lVert\Phi_t(p) - \Phi_t(\pi(p))\rVert \le C\operatorname{dist}(p, S)e^{-\gamma t}$. For derivative statements, the separate
hypothesis (S′) $\sup_N\lVert D(\Phi_t - \Phi_t\circ\pi)\rVert \le C_1e^{-\gamma_1t}$ (no distance factor: one would force $D\pi = I$ on $S$, impossible in
positive codimension). The regularity of the foliation and a neighbourhood covering an application box are not implied by
smoothness of $F$; theorem numbers to be pinned.

**Theorem 6 (conditional on (S)).** $\Phi_\tau = \Gamma_\tau\circ\pi + R_\tau$ with $\Gamma_\tau = \Phi_\tau|_S$ and $\lVert R_\tau(p)\rVert \le C\operatorname{dist}(p, S)e^{-\gamma\tau}$.
**Corollary 6.1 (conditional on (S′); citing Weyl).** $\sigma_{k+1}(D\Phi_\tau(p)) \le C_1e^{-\gamma_1\tau}$ on $N$, since $D(\Phi_\tau\circ\pi)$ has rank at most $k$.
**Corollary 6.2.** For a scalar output (or the max norm with componentwise midranges), the best surrogate that ignores a set of coordinates
errs by half the oscillation of $\Gamma_\tau\circ\pi$ along them, plus at most $\sup\lVert R_\tau\rVert$; for vector outputs in the Euclidean norm half the
diameter is only a lower bound for the enclosing radius. An $O(\varepsilon)$ estimate in slow–fast standard form needs uniform fast
attraction and control of the slow shift across the fibres; there is no general lower bound of that order (a decoupled slow equation has
zero tilt). **Corollary 6.3 (three budgets).** If $\Gamma_\tau$ and $\hat\Gamma$ are defined on a set $Q \supset \pi(N) \cup \hat\pi(N)$, and
$L = \sup_{p\in N,\,\hat\pi(p)\ne\pi(p)}\lVert\Gamma_\tau(\hat\pi(p)) - \Gamma_\tau(\pi(p))\rVert/\lVert\hat\pi(p) - \pi(p)\rVert$ is finite, then
$\sup_N\lVert\hat\Gamma\circ\hat\pi - \Phi_\tau\rVert \le \sup_{\hat\pi(N)}\lVert\hat\Gamma - \Gamma_\tau\rVert + L\sup_N\lVert\hat\pi - \pi\rVert + \sup_N\lVert R_\tau\rVert$
(write $\hat\Gamma(\hat\pi) - \Phi_\tau = [\hat\Gamma(\hat\pi) - \Gamma_\tau(\hat\pi)] + [\Gamma_\tau(\hat\pi) - \Gamma_\tau(\pi)] - R_\tau$); when $\hat\pi$ lands in $S$ and $S \subset Q$,
the first supremum may be taken over $S$. It may not otherwise: for $x' = -x$, $S = \{0\}$, $N = [-1, 1]$, $\tau = \log 2$, $\pi = 0$, $\hat\pi = 1$,
$\Gamma_\tau(x) = x/2$ and $\hat\Gamma(x) = 5x$, the error at 0 is 5 and the right side with $\sup_S$ is 1 (review 8).
**Corollary 6.4.** The same factorisation holds componentwise, with one $\pi$ for every output. Under (S), with $\pi$ continuous, an
embedding $\chi: S \to \mathbb{R}^q$ gives a factorisation through $q$ continuous coordinates up to the remainder $R_\tau$, so an upper count
$m_N(\eta) \le q$ holds at tolerances $\eta \ge \sup_N\lVert R_\tau\rVert$; $q = k$ needs a global chart.

**Proposition 6.5 (a lower bound on the coordinate count; proved, citing Borsuk–Ulam).** Let $\Phi$ be $C^2$ on the ball $B(p, r)$ with
$\lVert D^2\Phi(\xi)[v, v]\rVert_2 \le K_2\lVert v\rVert_2^2$ there and $\sigma_m(D\Phi(p)) \ge s$. For every continuous $c: B(p, r) \to \mathbb{R}^{m-1}$ and every $G$:
$\sup_{B(p,r)}\lVert\Phi - G\circ c\rVert_2 \ge sr - \frac12K_2r^2$; the best radius is $\min(r_0, s/K_2)$ ($r_0$ the largest admissible; $sr_0$ if $K_2 = 0$).
In the max norm over $d_{\mathrm{out}}$ outputs divide by $\sqrt{d_{\mathrm{out}}}$. Discontinuous encoders are not covered except within one cell
containing a ball. *Proof.* On the $(m - 1)$-sphere of radius $r$ in the span of the first $m$ right singular vectors, Borsuk–Ulam gives
$c(p + v) = c(p - v)$; $\lVert\Phi(p + v) - \Phi(p - v)\rVert \ge 2sr - K_2r^2$; one of the two errors is at least half. $\square$ On a $C^2$ submanifold $M$
apply it to $\Phi\circ q$ for a chart $q$ centred at $p$ with $Dq(0)$ isometric onto $T_pM$, with $K_2$ bounding the composite's Hessian. It is a
specialisation of the antipodal argument of DeVore, Howard, Micchelli (1989). *Scale comparison* (review 3 test B; sampled $K_2$ and $s$;
Robertson at a visited state, $\tau = 1$): bound 0.048 and 0.177 on balls of 0.05 and 0.2 dex, against 0.046 and 0.180 for the best sampled
linear one-coordinate encoder's slab half-diameter (not a verified covering radius): the same scale, not a proof of sharpness.

**Example 6.8 (the local rank does not bound the coordinate count; proved; review 4).** $x' = \omega Jx + a(1 - \lVert x\rVert_2^2)x$,
$J = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$, $a \gg |\omega| > 0$, on the invariant annulus $\frac12 \le \lVert x\rVert \le \frac32$. The radius solves $r' = a(1 - r^2)r$,
$r_t = [1 + (r^{-2} - 1)e^{-2at}]^{-1/2}$, the phase rotates at rate $\omega$; on the unit circle $\sigma(D\Phi_t) = (1, e^{-2at})$, and the second singular
value is uniformly small on the annulus for large $a$. Yet any continuous scalar encoder takes equal values at an antipodal pair of the circle
(Borsuk–Ulam), the exact step maps that pair to an antipodal pair at distance 2, so one of the two decoded outputs errs by at least 1
(Euclidean; $1/\sqrt2$ in the max norm). Two coordinates (the identity) suffice. So local rank 1 everywhere coexists with coordinate count 2,
even on the periodic orbit alone; the obstruction is known in the autoencoder literature (Batson et al.).

**Theorem 6.6 (singular values from volume contraction; proved, citing Muldowney 1990).** With $W$ fixed and invertible, $\alpha_1 \ge \alpha_2 \ge \ldots$
the eigenvalues of the symmetric part of $WJ_FW^{-1}$, $s_m(t) = \sup_{x_0\in B}\sum_{i\le m}\alpha_i(x(t; x_0))$: $\prod_{i\le m}\sigma_i(WD\Phi_\tau W^{-1}) \le \exp\int_0^\tau s_m$
(the $m$-th additive compound's Euclidean log norm; the Constantin–Foias trace argument, Douady–Oesterlé, R. A. Smith, from memory). For
$u' = D\Delta_nu + f(u)$ with diffusion coefficient $D > 0$ ($\Delta_n$ the cell-centred Neumann Laplacian on $n$ cells of the unit interval, $W = I$,
$|f'| \le L_f$): $s_m \le mL_f - D\sum_{i\le m}\lambda_i(-\Delta_n)$ with $\lambda_i = 4n^2\sin^2((i - 1)\pi/(2n)) \ge 4(i - 1)^2$ uniformly in $n$, and
$\sigma_m \le (\exp\int_0^\tau s_m)^{1/m}$; so at fixed $D, \tau, L_f$ the number of singular values above a fixed threshold is bounded independently of
the mesh; this bounds a local rank,
not a coordinate count or a cost.

**Proposition 6.7 (the staircase).** (a) *Proved, constant $J$.* If $J$ is constant and diagonalisable,
$e^{\tau\operatorname{Re}\lambda_{(i)}}/\kappa \le \sigma_i(e^{\tau J}) \le \kappa e^{\tau\operatorname{Re}\lambda_{(i)}}$, $\kappa$ the infimum of $\kappa(V)$ over eigenbases: the count above $\eta$
lies between the frozen counts at $\eta\kappa$ and $\eta/\kappa$ (on POLLU, $\kappa = 99$ to $6.9\times10^3$: thresholds shifted by 2–4 decades each way).
(b) *Proved, one-sided.* If $J(x(t)) = V(t)\Lambda(t)V(t)^{-1}$ with $V \in C^1$, then
$\prod_{i\le m}\sigma_i(D\Phi_\tau) \le (\lVert V(\tau)\rVert\lVert V(0)^{-1}\rVert)^m\exp\int_0^\tau(\sum_{i\le m}\operatorname{Re}\lambda_{(i)} + m\lVert V^{-1}V'\rVert)$ ($z = V^{-1}w$ solves
$z' = (\Lambda - V^{-1}V')z$; Ky Fan; Horn's product inequality). Its $m$-th root bounds $\sigma_m$ above, so it can only show that fewer than $m$
values exceed a threshold; it gives no lower bound. (c) *No path-averaged law.* With $J(t) = R(\omega t)\operatorname{diag}(0, -b)R(\omega t)^{\mathsf T}$,
$0 < |\omega| \ll b$, the instantaneous slow exponent is 0, but in the rotating frame the exponents are $(-b \pm \sqrt{b^2 - 4\omega^2})/2 < 0$, so the
leading singular value decays like $e^{-\tau(b - \sqrt{b^2 - 4\omega^2})/2}$, whose small-rotation expansion is $e^{-(\omega^2/b)\tau}$: a uniform-in-$\tau$ comparison with $\exp\int\operatorname{Re}\lambda_{(i)}$ fails however slowly the basis
turns. A finite-time estimate needs explicit accumulated perturbations.

**Checks (observations).** `t6_hires` (HIRES; $t_0 = 1, 10, 100$; $\tau = 0.01$ to 10): the frozen count matches the number of singular values above
$10^{-3}$ and $10^{-6}$ in 24 of 24 cases (thresholds in spectral gaps); individual values follow $e^{\tau\lambda_i}$ (at $t_0 = 1$, $\tau = 1$: 0.14, 0.10,
$1.7\times10^{-4}$, $1.9\times10^{-5}$ against 0.12, 0.091, $2.5\times10^{-4}$, $2.8\times10^{-5}$); Theorem 6.6 with $W = I$ is valid and useless there (about $10^{13}$ at $t_0 = 1$, $\tau = 1$; at $\tau = 10$ the sampled bounds reach $2.7\times10^{284}$
and saturate floating point).
Review 2 test B: the frozen count matches on POLLU (24/24) and the Brusselator (18/20, 20/20) and undercounts on Robertson at $\tau = 100$;
review 2 test E: the path average matches Robertson's $\sigma_2$ within 3–15% in 12 cases (an observation; (c) shows it is not a law).
`t6_slowfast` (Michaelis–Menten): the best function of $s$ alone errs $3.46\times10^{-2}$ at $\varepsilon = 0.1$ down to $3.51\times10^{-4}$ at $10^{-3}$, slope 0.999
(one example of Corollary 6.2's order); its $\sigma_2$ law and Robertson's $\sigma_3$ are Liouville's formula (sanity checks). Review 1 test E
(Fisher–KPP): 7 singular values above $10^{-2}$ and 10 above $10^{-4}$ at $n = 40$ and 80. U3: Robertson's two large ambient singular values and
one fitted coordinate on one sampled leaf (§8) are observations; its third value is at the solver's resolution.

**Prior art.** Fenichel fibration (TH-49), fibre projections (TH-50 to TH-52), isostables (TH-53), inertial manifolds (TH-56); ISAT, ILDM, CSP
and the G-Scheme use the Jacobian's singular values or modes; gradient-based dimension reduction (TH-48) gives $L^2$ bounds; Theorem 6 is
tracking at one time; Theorem 6.6 the classical compound-matrix estimate; Proposition 6.5 a specialisation of DeVore–Howard–Micchelli;
Example 6.8's obstruction is known for autoencoders.

---

## T7. Sup norm on the box against accuracy where the states are

**Proposition 7.1 (proved, citing Hoeffding).** For $f = g(s)$, $s = \frac1d\sum_jx_j$ with independent coordinates uniform on $[-1, 1]$:
$\Pr(|s| \ge t) \le 2e^{-dt^2/2}$, so a polynomial accurate on $|s| \le t_\eta = \min(1, \sqrt{2\ln(2/\eta)/d})$ violates the tolerance on measure at
most $\eta$, while the box's sup norm needs $|s| \le 1$. **Lemma 7.2 (proved).** Across any cut write $s = s_L + s_R$; a polynomial of degree $q$
in $s$ is $\sum_{a=0}^qs_L^ab_a(s_R)$, so its TT ranks are at most $q + 1$.

**Check** (`t7_concentration`; $g = 1/(1 + 0.8s)$, tol and $\eta = 10^{-3}$): sup-norm degree 13 for every $d$; measure-sense degree 13, 13, 13, 9, 5, 4
(Hoeffding) and 12, 8, 6, 4, 3, 3 (measured quantile) for $d = 2, 5, 10, 20, 50, 100$; vertex errors 102, 1,029, 1,495 tol at $d = 20, 50, 100$. An
illustration, not a mechanism for other functions; consistent with U4 in direction only.

**Scope.** For the maximum pointwise error the target is the sup norm on the states the application visits; a measure-sense quantile on a
box is a different, distributional target.

**Prior art.** Caflisch, Morokoff, Owen 1997; Trefethen 2017; Hoeffding 1963.

---

## 8. Surrogates in reduced coordinates

**Theorem 8 (finite-horizon rollout bound with computable constants; proved).** Fix a horizon $n \ge 1$ and an exact chain
$x_0, \ldots, x_{n-1}$ in a set $M$. Let $\hat\pi: \mathbb{R}^d \to \mathbb{R}^k$ be $L_\pi$-Lipschitz, $U \supset \{\hat\pi(x_j)\}$ and
$U_\rho = \{u: \operatorname{dist}(u, U) \le \rho\}$, $\hat G: U_\rho \to \mathbb{R}^d$ with $L_G = \mathrm{Lip}(\hat G|_{U_\rho})$, $\hat g = \hat\pi\circ\hat G$ with
$L_g = \mathrm{Lip}(\hat g|_{U_\rho})$, $A_j = \sum_{i<j}L_g^i$, and the defects $\varepsilon_G = \max_{j<n}\lVert\hat G(\hat\pi x_j) - \Phi(x_j)\rVert$,
$\varepsilon_u = \max_{j<n}\lVert\hat g(\hat\pi x_j) - \hat\pi\Phi(x_j)\rVert$ ($\le L_\pi\varepsilon_G$). For $\Psi = \hat G\circ\hat\pi$ and $y_0 = x_0$: if $\varepsilon_uA_{n-1} \le \rho$,
then for $1 \le j \le n$

$$\lVert y_j - x_j\rVert \le \varepsilon_G + L_G\,\varepsilon_u\,A_{j-1}.$$

*Proof.* $u_j = \hat\pi(y_j)$, $v_j = \hat\pi(x_j)$, $a_j = \lVert u_j - v_j\rVert$, $a_0 = 0$. While $u_j \in U_\rho$: $u_{j+1} = \hat g(u_j)$ and
$\lVert\hat g(v_j) - v_{j+1}\rVert \le \varepsilon_u$, so $a_{j+1} \le L_ga_j + \varepsilon_u$ and $a_j \le \varepsilon_uA_j \le \rho$ (so $u_j \in U_\rho$ by induction through
$j \le n - 1$); and $y_{j+1} - x_{j+1} = [\hat G(u_j) - \hat G(v_j)] + [\hat G(v_j) - \Phi(x_j)]$. $\square$
The proof uses no injectivity of $\hat\pi$ and only the inputs $u_0, \ldots, u_{n-1}$; for a family of initial states the maxima run over every
exact input through step $n - 1$. It is a discrete Lipschitz error estimate for the autoregressive map $\hat\pi\circ\hat G$. To be a *verified*
bound, $\varepsilon_G, \varepsilon_u$ need a verified net of the inputs with a defect modulus and a reference error ($\varepsilon \le \max_i\lVert\Psi(x_i) - \tilde\Phi(x_i)\rVert + L_\delta h + b$),
and $L_G, L_g$ need validated bounds on the fitted map (polynomial derivative bounds, interval arithmetic), with any output clipping included.
A chart with more coordinates than the data's local dimension leaves $\hat G$ unconstrained across the data, so $L_g \gg 1$. Linear invariants
must be conserved on the output (Proposition 2.4). *Scope.* The theorem applies to a deployed map exactly when that map is constant on the
fibres of its encoder $\hat\pi$, so that it factors as $\hat G\circ\hat\pi$. A projection onto a fixed invariant leaf can be absorbed into $\hat G$. A
normal correction $B(u)(z - D(u))$, a projection onto the input's own invariant when the domain spans several leaves, an abundance floor that
changes the invariant, or a refusal branch with the solver as fallback, are not constant on the fibres of a chart encoder. Such a map needs
either an augmented encoder (the invariant, or the residual, as further coordinates) that satisfies the theorem's hypotheses, or a
full-state certificate: a bound on the deployed defect along the chain, the exact map's propagation (Proposition 2.1, or 2.2 with its
remainder) on a stated input tube shown to close, and conservation and positivity verified with clipping and floors included.

**Proposition 8.2 (accuracy off $M$; conditional on (S); untested).** Under (S), with a chart (or an embedding) $\chi$ on the image of $\pi$,
$\hat\pi \approx \chi\circ\pi$, and $\hat G$ defined on $\hat\pi(N) \cup \chi(\pi(N))$ with the finite paired constant
$L = \sup_{p\in N,\,\hat\pi(p)\ne\chi(\pi(p))}\lVert\hat G(\hat\pi(p)) - \hat G(\chi(\pi(p)))\rVert/\lVert\hat\pi(p) - \chi(\pi(p))\rVert$:
$\sup_N\lVert\Psi - \Phi_\tau\rVert \le \sup_{\chi(\pi(N))}\lVert\hat G - \Gamma_\tau\circ\chi^{-1}\rVert + L\sup_N\lVert\hat\pi - \chi\circ\pi\rVert + C\sup_N\operatorname{dist}(\cdot, S)e^{-\gamma\tau}$
(insert $\hat G(\chi(\pi(p)))$ and $\Gamma_\tau(\pi(p))$ between $\hat G(\hat\pi(p))$ and $\Phi_\tau(p)$; the last term is Theorem 6's remainder). When $\hat\pi$
lands in $\chi(\pi(N))$, a Lipschitz bound of $\hat G$ there supplies $L$. A Lipschitz bound on $\chi(\pi(N))$ alone does not suffice otherwise: on
the attracting circle of Example 6.8 with $\omega = 0$, take the angle chart, inputs whose exact code image is $[-a, 0]$ inside it, a learned
code shifted by $h > 0$, and the decoder $(\cos u, \sin u) + A\max(u, 0)e_1$ (exact, and 1-Lipschitz, on the exact image); at angle 0 the first
coordinate errs $|\cos h - 1 + Ah|$, which grows with $A$ while the right side with the Lipschitz constant on the exact image does not (review
9). For an atlas, the chart-selection rule is part of the map: bound every chart the implementation can select and take the maximum. Data on $M \subset S$ do not give a fit on $\chi(\pi(N))$, and fibre constancy of $\hat\pi$ is not enough unless the
encoded base point is also right. This is the guarantee an operator-split simulation needs (the other operators move the state off $M$): the
main open problem.

**Checks (observations and estimates).**

- `t2_slow_surrogate` (Robertson, log coordinates, $\Delta t = 1$; fits on the visited states of 100 trajectories only; 40 held-out trajectories; the
  visited states lie on one curve of one leaf, parametrised by $z_1$). Worst errors (dex):

  | surrogate | one-step on visited states | on the tube ($\pm0.05$ dex, leaving the leaf) | rollout, 39 steps |
  |---|---|---|---|
  | V: degree 3 in $(z_1, z_2, z_3)$, increment $z + g(z)$ | $2.9\times10^{-10}$ | 1.74 | 39.9 (reaches the clip) |
  | V-full: the same, next state directly | $2.8\times10^{-10}$ | 2.04 | 39.6 (reaches the clip) |
  | S: degree 8 in $z_1$ alone, next state directly | $5.2\times10^{-4}$ | 0.60 | $5.2\times10^{-4}$ |
  | S-incr: degree 8 in $z_1$ alone, increment $z + g(z_1)$ | $3.2\times10^{-3}$ | 0.62 | $1.1\times10^{-2}$ (3.3 times its one-step error) |

  For S the sampled constants (maxima over the visited states; central differences on 4,001 points of the widened range) are
  $\varepsilon_G = 1.18\times10^{-3}$, $L_G = 36$, $L_g = 0.997$, giving the *estimate* 1.53 after 39 steps against the measured $5.2\times10^{-4}$ (with
  $L_\pi\varepsilon_G$ for $\varepsilon_u$); a diagnostic, not a verified bound. S's rollout equaling its one-step error reflects Robertson's contraction along
  its slow curve; its 0.60 on the tube is the mass direction (the tube leaves the leaf).
- Review 3 test A (HIRES, log coordinates, $\Delta t = 2$, 170 steps, fits on visited states only; charts by PCA of the visited states, by the leading
  eigenvectors of the mean $D\Phi^{\mathsf T}D\Phi$, or with the invariant as an input): all four full-coordinate fits reach the clip; the 3-coordinate PCA
  and Jacobian charts at degree 6 stay bounded (rollout $5.4$–$5.7\times10^{-2}$ dex after 169 steps, growth 89–157 against the exact dynamics'
  amplification of 52–61), while at degrees 3–4 the PCA chart's rollout reaches 2.15 and 1.03 dex ("bounded" here means avoiding the clip, not
  meeting a small tolerance); 4- and 5-coordinate charts reach the clip within 10 steps; the invariant-input chart diverges. The two main
  3-coordinate fits err 0.4–0.6 dex at 0.05 dex off the visited set (the larger charts 19.8–64.6, the invariant chart 1.45). None of these fits had
  verified constants or enforced conservation.
- What this supports: a pair of controlled experiments in which reduced coordinates of the data's dimension, fitted on the data and predicting
  the next state, avoided the divergence of full-coordinate fits, within a limited window of chart dimension and degree.

**Demonstration 8.3 (Theorem 8 on one Robertson leaf; `theory/demos/robertson_leaf.py`, `theory/demos/data/robertson_leaf.json`).**
Robertson in log10 abundances, $\Delta t = 1$, the leaf $y_1 + y_2 + y_3 = 1$, horizon 39 steps from step 1 (past the initial layer). $M$ is the
leaf's slow curve, traced by one reference trajectory from $(1, 0, 0)$ (Radau, rtol $10^{-12}$) and parametrised by $z_1$, which decreases along
it; the held-out chains' states lie within $1.0\times10^{-10}$ dex of it. The surrogate: $\hat\pi(z) = z_1$ ($L_\pi = 1$ in the max norm);
$\hat G(u) = (g_1(u), g_2(u), \log_{10}(1 - 10^{g_1} - 10^{g_2}))$ with $g_1, g_2$ the degree-$q$ Chebyshev interpolants of the next state's $z_1, z_2$
at the Chebyshev points of $U_\rho = [-0.305, -0.010]$ ($\rho = 0.01$), so every output conserves the leaf's mass exactly (offset at most
$5\times10^{-17}$, measured at the chains' final states). Constants: $L_g$ and $L_G$ are bounded rigorously for the fitted polynomials (a grid maximum plus half the
spacing times a second-derivative bound from the Chebyshev coefficients, $|T_k''| \le k^2(k^2 - 1)/3$; an interval bound for the
reconstructed component), up to floating-point rounding; $\varepsilon_G, \varepsilon_u$ are *estimated*: the defect's maximum on 20,001 curve points, plus
half the spacing times the defect's own Lipschitz constant estimated from the grid (a secant estimate, so not conservative in principle),
plus the reference error estimated from a second trajectory at rtol $10^{-10}$ ($1.1\times10^{-11}$). That estimate is too small: both
references share one linear-interpolation table from $z_1$ to time, whose error is about $9.5\times10^{-11}$ in the step map (review 5 G7), so
the rows with $q \ge 20$ sit at the reference's floor. The defects on the chains' own inputs equal those on the curve to 1% (review 5 test 6).

| degree $q$ (solver steps) | $L_g$ | $L_G$ | defect on the grid | Theorem 8's bound, 39 steps | measured worst rollout (40 chains) | bound / measured |
|---|---|---|---|---|---|---|
| 8 (9) | 0.9974 | 18.8 | $4.4\times10^{-5}$ | $2.1\times10^{-3}$ | $1.6\times10^{-4}$ | 13 |
| 12 (13) | 0.9975 | 18.8 | $8.8\times10^{-7}$ | $3.9\times10^{-5}$ | $2.1\times10^{-6}$ | 19 |
| 16 (17) | 0.9975 | 18.8 | $2.0\times10^{-8}$ | $8.5\times10^{-7}$ | $3.1\times10^{-8}$ | 27 |
| 20 (21) | 0.9975 | 18.8 | $5.0\times10^{-10}$ | $4.9\times10^{-8}$ | $3.0\times10^{-9}$ | 16 |

The bound holds at every degree and is 13 to 27 times the measured rollout; the horizon condition $\varepsilon_uA_{n-1} \le \rho$ holds. $L_G = 18.8$
comes from the conservation-reconstructed $\log_{10}y_3$ ($y_1/y_3$ is large early on); $L_g < 1$ (the flow contracts along the curve), so
$A_{38} \approx 36$. With the cruder margin (the Lipschitz constant of $\hat G$ plus that of the exact curve map) the bound is $5.9\times10^{-2}$ at degree 8 and
about $5.7\times10^{-2}$ at the higher degrees: the margin then dominates. The interval bound for the reconstructed component is 29% above its sampled supremum (18.8 against
14.6); subdividing $U_\rho$ would tighten it rigorously. *What this does and does not show.* The surrogate is a one-variable Chebyshev interpolant of the map along the curve on which it is
then tested, with $L_g < 1$ for free; the hard cases of Theorem 8 ($L_g > 1$, $k > 1$, a chart that is not a coordinate, inputs off $M$) are
not exercised. It is the first end-to-end evaluation of Theorem 8 with every
constant computed, half of them rigorously; the defects are estimates, because no validated (interval) ODE enclosure was used, and that is
the remaining step to a verified bound. The surrogate covers states on the slow curve of one leaf from step 1 on: the benchmark's
ChebTT box surrogate (955 and 1,130 evaluations at tolerances 0.01 and 0.001 dex; worst rollouts 0.051 and 0.0043 dex) also covers the
initial layer (step 0) and its whole bounding box, so the two are not the same task, and the reduced surrogate's cost excludes the reference
trajectory that traces the curve.

**Demonstration 8.4 (how many coordinates a reduced ChebTT would need on the benchmark's stiff step maps; `theory/demos/reduced_dims.py`,
`theory/demos/data/reduced_dims_*.json`).** Before building a reduced-coordinate ChebTT, the chart was measured on the benchmark's own
trajectories, at tolerance 0.01 dex: with the principal directions of the visited states (mean $\mu$, directions $V_k$), the *lift error*
$\max|\Phi(\mu + V_kV_k^{\mathsf T}(z - \mu)) - \Phi(z)|$ over 400 held-out visited states (a surrogate with $k$ inputs and a full-state output;
the step damps the dropped directions' fast part) and the *residual* $\max|z - \mu - V_kV_k^{\mathsf T}(z - \mu)|$ (what decoding a predicted
chart value costs). The smallest $k$ reaching a quarter of the tolerance:

| problem ($d$) | $k$ for the lift error | $k$ for the residual | the visited set's dimension (review 3) |
|---|---|---|---|
| Fisher–KPP (10) | 7 | 10 | |
| Fisher–KPP (20) | 8 | 13 | |
| HIRES (8) | 7 | 7 | 3 |
| POLLU (20) | 16 | 16 | |

*Outcomes of the committed predictions (post hoc: script and output in one commit).* Fisher–KPP: the count at which the lift error reaches
the tolerance did not grow from $d = 10$ to 20 (6 and 6 at 0.01 dex; 7 and 8 at a quarter of it, the table's criterion; lift errors $5.7\times10^{-3}$ against $8.6\times10^{-3}$ at $k = 6$, $2.1\times10^{-3}$ against $3.0\times10^{-3}$ at $k = 7$), and the input needs fewer
coordinates than the state (the step damps what the input drops): confirmed, as an empirical trend for these sampled ensembles. Theorem 6.6
motivated it but does not predict it: it bounds $\sigma_m(D\Phi)$, not the displacement made by projecting a visited state onto its PCA
subspace, and a discarded direction of small variance can be slow even with a constant Jacobian and a fast spectral gap. Connecting the
two needs a bound on $D\Phi$ along the actual lift segments. HIRES and POLLU: predicted equal to the visited set's dimension; measured 7
against 3 on HIRES and 16 against about 7 on POLLU: **failed** for the affine chart. Projecting a curved visited set onto a flat subspace
moves the state in slow directions the step does not damp. A first build
(`theory/demos/reduced_chebtt.py`, Fisher–KPP at $d = 10$, $k = 7$, ten output trains) did not finish within the benchmark's 240 s, where box
ChebTT used 107,002 evaluations in 210 s; that single unfinished run, at another tolerance and target than the chart measurement, does not show that an
affine lift cannot be cheaper (review 6 H7). On finer grids the chart dimension stays at 8: the lift error first reaches a quarter of the tolerance at
$k = 8$ for $d = 40$ ($1.8\times10^{-3}$) and $d = 80$ ($2.0\times10^{-3}$), the errors at each $k$ agreeing to 7% (review 6 test 6), so the
diffusion ladder's input dimension is mesh-independent from $d = 20$ to 80.

**Demonstration 8.5 (a curved lift, and what it is for; `theory/demos/curved_lift.py`, predictions committed in `8c1720c`; output
`theory/demos/data/curved_lift_hires_k3.json`; review 6's tests in `theory/reviews/scratch-round-6-fable/`).** The lift of a chart value $u$
(top-3 principal coordinates of HIRES's visited log-states) was made in three ways: affine ($\mu + Vu$), a *decoder* $D(u)$ (a least-squares
polynomial of total degree 4 from chart values to states, fitted on the visited states, no solve), and the decoder followed by constrained-run
iterations (relax with the solver, reset the chart coordinates along $V$). Lift errors $\max|\Phi(\mathrm{lift}(\pi(z))) - \Phi(z)|$ on 400 held-out
states: affine $3.85\times10^{-2}$, decoder $2.45\times10^{-4}$, decoder with 1–6 iterations $3.8$–$7.8\times10^{-3}$ (worse; the lift was chosen on
these test states, a selection the next pass should make on validation states). On the same three chart coordinates, the same test states, the same
projection onto the invariant and the benchmark's `rollout` (review 6 test 1):

| surrogate ($k = 3$) | solver calls beyond the training trajectories | one-step worst (dex) | worst rollout, 170 steps |
|---|---|---|---|
| reduced ChebTT, oracle $\Phi(D(u))$ on the chart box | 9,477 (10,287 and 207–209 s on two rebuilds) | $2.46\times10^{-3}$ | 0.30 |
| the ideal reduced map $\Phi(D(\pi z))$ | one per query | $3.4\times10^{-4}$ | $2.5\times10^{-2}$ |
| data fit: least-squares polynomial from chart value to next state on the training pairs, degree 5 | 0 | $2.1\times10^{-3}$ | $7.4\times10^{-2}$ |
| the same, degree 6 | 0 | $6.4\times10^{-4}$ | $1.5\times10^{-2}$ |
| the same, degree 7 | 0 | $1.5\times10^{-4}$ | 2.2 |
| the same, degree 8 | 0 | $5.7\times10^{-5}$ | 4.2 |
| the same, degree 9 (review 7) | 0 | $3.4\times10^{-5}$ | $4.3\times10^{-4}$ |
| the same, degree 10 (review 7; section G) | 0 | $6.7\times10^{-6}$ | $1.5\times10^{-4}$ |

*On the visited set a data fit is cheaper and better:* with no solve, the degree-10 fit holds $1.5\times10^{-4}$ in the rollout on the same 20
trajectories, 2,000 times below the reduced ChebTT (the degree-6 fit, an interior choice of the degrees first searched, 4–8, is 20 times better
than the ChebTT; degrees 9–10 were found by review 7, and degrees 7, 8, 11 and 12 diverge, so the degree must be chosen on the training chains;
section G). The decoder lift makes a reduced oracle well defined; what the solver-queried build buys is
accuracy off the data within the chart's span (kicks of 0.01 dex inside it: the reduced ChebTT errs $2.8\times10^{-3}$, the degree-6 data fit
$6.6\times10^{-2}$; on kicks of 0.01 dex in every coordinate, a different test, the value-only fits err 0.11 and 0.20 at degrees 9 and 10, `g7`). Box ChebTT is a different task (the 8-D bounding box) that did not finish within the benchmark's whole-run limit; the 188 s here
is the build alone, and 91–96% of the chart box is away from the data, where calls cost about 5 times more.

*Rollouts (review 6 test 3, H6).* The reduced ChebTT's rollout error is the exact dynamics' propagation of its defects: the first-order sum
$\sum_jJ_{n-1}\cdots J_{j+1}\delta_j$ with exact Jacobians is within a factor 1.7 of the measured rollout on the six worst trajectories, the error lies in
the chart's span, and $\max\lVert D\Psi\rVert = 1.24$ is below $\max\lVert D\Phi\rVert = 1.83$. The ratio of rollout to one-step error is not a property
of the dynamics (120, 75 and 23 for the three rows above, against HIRES's accumulation factor $A_{170} = 5{,}351$), and a small one-step error is not
sufficient: the degree-7 and 8 fits reach 2.2 and 4.2. Their single-state Jacobian norms are large (up to 16), but so are those of the degree-9 and 10
fits that hold the rollout (about 15), so a single-state maximum does not separate them (section G). Two diagnostics are computable on the
training chains: (i) the defects' first-order sum, and (ii) the surrogate's own propagation (products of $D\Psi$) against the dynamics'.
Neither is a sufficient condition, even with the Jacobians taken on the surrogate's chain: for the exact identity step, a start at 0 and
$\Psi(y) = y + \varepsilon + C\,h(y/\varepsilon)$ with $h(s) = 3s^2 - 2s^3$, the two-step first-order sum is $2\varepsilon$ and $D\Psi = 1$ at both inputs 0 and
$\varepsilon$, yet the two-step error is $2\varepsilon + C$, so for $\varepsilon, C > 0$ and any tolerance strictly between $2\varepsilon$ and $2\varepsilon + C$ both
diagnostics pass and the rollout fails (reviews 8, 9). A sufficient condition needs a tube that is shown to close: Theorem 8 for factored
maps, Proposition 2.2's Taylor representation with its remainder bound, or the exact mean-Jacobian recurrence along the exact-to-surrogate
segments. The chained error on the training chains, which section G's rule
uses, includes the nonlinear feedback and is a different quantity from (i). As a selection diagnostic it picked fits that pass the
unperturbed test chains in every `g5` and `g6` draw, but it does not separate passes from failures at every degree (in `g6`, degree 10
passes the training chains and fails the test chains in two draws).

*Off the visited set (H2).* The step amplifies some chart-normal input perturbations (gain 1.41 along an ambient normal direction; an
amplified displacement can still be turned into a tangent one, so this is not expansion in the distance to the set), and a kick of $\varepsilon$
gives an error of about $1.9\varepsilon$ ($1.9\times10^{-2}$ at 0.01 dex); nothing is refused. Two constructive results (review 6 test 9), from two
different arms. First, for the *uncorrected reduced ChebTT* ($e_0 = 2.46\times10^{-3}$ its error on the set, 400 kicked states): the decoder
residual $r(z) = \max|z - D(\pi z)|$, which needs no solve, tracked its error, the largest observed ratio to $e_0 + r$ being 1.532 (an
empirical constant of that sample, not a bound), a computable proxy for Proposition 8.2's middle term (a deterministic starting point is $\lVert G(u) - \Phi(D(u))\rVert + Lr$ on validated lift segments, before projection); as a refusal rule at a
quarter of the tolerance it admitted 5% of the states kicked by 0.003 dex and none at 0.01 or 0.05. Second, for the ideal reduced map
$\Phi(D(\pi z))$ (a solve at the decoded point, 100 kicked states): a first-order normal correction $\Psi_1(z) = \Phi(D(\pi z)) + D\Phi\,(z - D(\pi z))$,
with $D\Phi$ by central differences, cut the kicked error from $1.9\times10^{-2}$ to $2.0\times10^{-4}$ at 0.01 dex and from $9.4\times10^{-2}$ to
$4.9\times10^{-3}$ at 0.05 dex (a fitted $B(u) \approx D\Phi(D(u))$ is section G's step).

*Predictions.* P1 (decoder far below affine) held; its second half (polished within a quarter of the tolerance) failed. P2 was committed for the
polished lift and run with the decoder (the best on the test states); its timing part held for the build alone; "rollouts bounded" and P3 (the
invariant conserved, $7.6\times10^{-16}$ per step) could not fail, since `Projected` enforces the invariant and the outputs are clipped.

*POLLU (H3).* The earlier "locally five-dimensional, needs an atlas" is withdrawn: its neighbourhoods held a median of 3 trajectories, so they measured
single curves. At fixed steps the 100 training states have about six transverse directions (singular values down to 0.19–0.23, then 0.02–0.03): the set
is about 7-dimensional. At $k = 7$ a cubic decoder reaches a lift error of $2.0\times10^{-2}$ (affine $9.2\times10^{-2}$); degree 4 overfits (330
coefficients for 100 trajectories). The limit is the number of trajectories against a six-parameter family, not the geometry.

**Prior art.** Combustion tabulates on low-dimensional manifolds of progress variables (ILDM, Maas–Pope 1992; FGM, van Oijen–de Goey 2000; FPI;
REDIM, Bykov–Maas 2007; ISAT on RCCE manifolds, Tang–Pope 2002; ICE-PIC, Ren et al. 2006; ISAT–RCCE–GALI, Hiremath et al. 2011; from memory);
constrained runs fill a slow manifold at prescribed chart values (Gear, Kaper, Kevrekidis, Zagaris 2005; TH-51); learned-manifold reduced dynamics
include spectral submanifolds (Cenedese et al. 2022), Linot–Graham 2020, Floryan–Graham 2022 and chemistry autoencoders (Grassi et al. 2022,
Sulzer–Buck 2023, Holdship et al. 2021). Discrete-time error bounds for manifold Galerkin and LSPG (Lee–Carlberg 2020, Thm 4.3) and latent
stability and error propagation with encoder, decoder and latent dynamics through a Lipschitz recurrence (Farenga et al. 2024, arXiv:2408.15183,
§§3.2–3.3) are close to Theorem 8. Theorem 8 is therefore a specialisation of a known technique; a contribution would be a *verified*
end-to-end certificate for a stated class of stiff step maps.

## G. The gradient tier: values and Jacobians

When the oracle also returns derivatives (analytic Jacobians of a static function; forward sensitivities of a step map), each
evaluation carries $1 + k$ numbers along $k$ chart directions instead of 1. The theory then has two tiers. With gradients, the local
quantities the diagnostics use ($\sigma_m(D\Phi(p))$ in Proposition 6.5, the spectra of Proposition 6.7) are measured rather than differenced,
and the fits can use derivative data; without them, finite differences cost $k + 1$ (one-sided) or $2k$ (central) evaluations per gradient,
or the fits use values alone. An oracle Jacobian is a local derivative of the exact step and verifies nothing by itself: Theorem 8's gains
are those of the fitted map, bounded from its coefficients without oracle derivatives; Proposition 6.5 also needs a Hessian bound over a
ball, and Proposition 6.7(a) concerns a constant Jacobian. Scripts and
outputs: `theory/gradients/g*.py`, `theory/gradients/data/g*.json`; every script's predictions were committed before its run
(`d0e5e27`, `8f80735`, `e3997ea`).

| Test | Prediction | Measured | Verdict |
|---|---|---|---|
| Chebyshev–Hermite interpolation in one variable (`g1_static`; two product-peak factors and a Runge function) | values and first derivatives at $n$ points give degree $2n - 1$, so the points per decade halve: $1/(2\log_{10}\rho)$ | ratios 1.97, 2.01, 2.00; each rate within 1.5% of the Bernstein prediction (the slope fit stops at the rounding floor, a fix made after the first run fitted the plateau) | the half rate confirmed on three profiles. The Hermite contour formula with the squared Chebyshev nodal polynomial gives $O(r^{-2n})$ on any admissible contour $r > 1$ (DLMF §3.3); exact halving of the node counts does not follow from the degree alone |
| Gradient-enhanced least squares, $d = 5$, total degree 4 (126 terms), uniform random points (`g1_static`) | $1 + d$ equations per point: about 6 times fewer points for the same accuracy and conditioning | corner peak: with gradients 0.229 at 30 points and 0.253 at 60; values alone 0.247 at 180 and 0.253 at 360 (the sampled sup error of an approximate $L^2$ projection is 0.21); product peak (projection 0.12): 2–3 times; one random design per size, conditioning not measured | consistent at the tested budgets; the factor itself is not measured. A sample-saving theorem needs assumptions on the sampled block Gram matrix and the weights (at most $d$ points lie in a hyperplane, whose squared defining polynomial has zero values and gradients there) |
| HIRES reduced surrogate, 3 chart coordinates, random visited chart values (`g2_hires`; directional forward sensitivities) | G2.1 at degree 8, values with Jacobians at $N$ points match values alone at about $4N$ | at degree 8 every value-only fit fails (5.2–28 dex, the output clip); with Jacobians 2.7 at $N = 200$ and 0.048 at $N = 800$; at degree 6, 0.37 and 0.019 against 0.51 for values alone at 800 | G2.1 not testable as committed (the value-only fits never work); the run shows that Jacobians help on random visited points, not the factor 4. G2.2 (holding $2.5\times10^{-3}$ within a few hundred solves) failed |
| | G2.4 the rollout stays 50–150 times the one-step error, and falls in proportion | ratios 3.5 (degree 6, $N = 800$) and 6.7 (degree 8); on the grid (`g3_hires_grid`) 93–364 over both arms, the one-step error falling 19-fold from 5 to 13 points per axis and the rollout 8-fold | failed: the ratio is not a property of the dynamics (as Demonstration 8.5 found) |
| The same on tensor Chebyshev points of the chart box (`g3_hires_grid`) | G3.1 gradients save a factor 4 in points | no gain: at 13 points per axis values alone reach $1.27\times10^{-3}$, with Jacobians $2.42\times10^{-3}$; at 7 points with Jacobians $6.6\times10^{-3}$ against $2.3\times10^{-3}$ for values alone at 11 | **refuted** for the tested setup. Not because values determine the polynomial: for $q \ge n$, $T_n$ of each scaled coordinate vanishes on the grid and lies in the fitted space, so the value design has a nullspace (the best fits at 11 and 13 points have $q = 14, 15$). Attributing the loss to weighting or placement needs both arms in one unisolvent space ($q < n$) with ranks reported. Caveats: each configuration's degree was chosen on the test states (optimistic for both arms), and from 11 points per axis the cap of 900 terms stops both at degree 15 |
| | G3.2 some configuration holds $2.5\times10^{-3}$ with under 1,000 solves | values alone on 11 points per axis: $2.34\times10^{-3}$ with 1,331 solves (rollout 0.56 dex) | near miss, and optimistic (degree chosen on the test states): 7 times fewer solves than the reduced ChebTT's 9,477, above the predicted 1,000. A data fit on the training pairs with no solve does far better on the visited set (below) |
| | G3.3 with a sensitivity cost of $c$ plain solves, the gradient fit is cheaper in total than the value-only fit of the same accuracy if $c < 4$ | not tested | moot: it presupposes G3.1's factor 4, which failed, and the compiled-solver cost it names was never measured |
| Cost of a Jacobian | G2.3 a sensitivity solve costs at most 3 plain solves | 6.7 for 3 directions (11.8 against 1.8 ms, `g2_hires.json`); for all 8 directions 49 ms against a plain solve's 0.50 ms (medians over 100 training states, review 7's `t3_more.json`), while 16 plain solves by central differences take 19.6 ms and agree with the sensitivities to $2$–$8\times10^{-7}$ (`g4_diagnosis.json`) | failed here; in this Python implementation central differences are the cheaper route to a Jacobian, at 16 plain solves |

What the gradient tier establishes. Derivative data carry the information the theory says (Hermite halving exactly; several times fewer
points for least squares where the points are sparse or poorly placed), but on a well-designed Chebyshev point set they add little for the
values' sup error unless they are weighted for it. Where they pay: diagnostics (one Jacobian gives what Proposition 6.5, Proposition 6.7 and
the chart need), the constants of Theorem 8, and, on HIRES, a fit that converges with the degree and stays accurate off the visited set
(below). Open: weighting gradient rows by their contribution to the sup error; a compiled sensitivity solver.

**The HIRES recipe end to end** (`g4_hires_recipe.py`, `g4_diagnosis.py`, `g5_hires_selection.py`, `g6_hires_off_trajectory.py`, `g7_value_fit_kicks.py`; predictions
committed in `3ce981b`, `6c92037`, `9d3156c` and `479f12b`, each before its run, `g4_diagnosis` and `g7` post hoc; reviews 7 and 8). The recipe of §10 item 1 at $k = 3$, tolerance 0.01 dex:

- the chart and degree-4 decoder of Demonstration 8.5, and the invariant projection;
- a fit $G$ of total degree $q$ from chart values to next states on the training pairs;
- the normal correction $\Psi_1(z) = G(\pi z) + B(\pi z)\,(z - D(\pi z))$, with $B(u)$ fitted entrywise (degree 2 in $u$) to the
  log-coordinate Jacobians at 300 training states. These come from forward sensitivities in `g4`, and from central differences of the
  step in `g5`–`g6` (16 plain solves each), which reproduce `g4`'s figures to three digits. Both $B$ and the derivative rows below use $J$
  at the training state $X$, not at $D(\pi X)$; the difference is of the order of the decoder residual;
- the deployed map: $\Psi_1$ followed by clipping and the projection onto the input's invariant (with an abundance floor). HIRES's
  invariant $y_7 + y_8$ varies across trajectories, so even the uncorrected map reads it from the input, and the correction reads the full
  residual: neither is constant on the chart's fibres, so Theorem 8 does not apply as it stands (§8, Scope);
- the rule: the degree in 4–12 with the smallest worst training-chain error of the *deployed* surrogate over all 100 training
  trajectories; it passes when that error is within the tolerance.

`g4` ran a narrower rule: degrees 4–8, the uncorrected surrogate on 40 chains, and a single-state condition
$\max\lVert D\Psi\rVert \le \max\lVert D\Phi\rVert$. Review 7 showed that all three mislead. Its headline, a corrected rollout of
$7.3\times10^{-3}$, held on the benchmark's first 20 test trajectories only; on all 40 it is $1.5\times10^{-2}$, and over eight draws of
the Jacobian states $1.0$–$2.7\times10^{-2}$. The degree range also stopped one short of the value-only fit's good degrees.

The fits:

- **values:** $G$ fitted to $\Phi(X)$, no solve beyond the training trajectories;
- **consistent:** $G$ fitted to $\Phi(X) - B\,n_X$, the targets with the normal part removed;
- **gradients:** consistent, plus the chart-derivative rows $J\,D'(u)$ at the 300 Jacobian states;
- **gradients + off-trajectory:** also values and derivative rows of $\Phi(D(u'))$ at 300 chart values $u'$ displaced by up to 0.05 dex
  from the Jacobian states.

| surrogate ($k = 3$) | plain solves beyond the training trajectories | degree (rule) | training chains (100) | rollout, 20 / 40 test trajectories | one-step after a kick of 0.01 dex | rollout kicked by 0.003 dex at every step (40) |
|---|---|---|---|---|---|---|
| reduced ChebTT (Demonstration 8.5) | 9,477 | – | – | 0.30 / – | $2.8\times10^{-3}$ (kicks within the chart's span, a different test) | – |
| values (plain) | 0 | 10 | $1.0\times10^{-4}$ | $1.5\times10^{-4}$ / $1.5\times10^{-4}$ | 0.20 (degree 9: 0.11; `g7`) | 2.86 |
| gradients, corrected, 3 draws | 4,800 | 12 | $2.6$–$6.6\times10^{-4}$ | $3.6$–$9.8\times10^{-4}$ / $0.63$–$1.0\times10^{-3}$ | 0.24 (one draw, review 7) | 4.6, 4.7 and $\ge 27$ (stopped past 10 dex) |
| the same at degree 8 | 4,800 | – | $1.1$–$2.5\times10^{-2}$ | $4.5$–$7.3\times10^{-3}$ / $1.0$–$2.7\times10^{-2}$ | $8.1\times10^{-3}$ (one draw) | $2.3$–$8.0\times10^{-2}$ |
| the same uncorrected, degree 8 | 4,800 | – | – | – | – | 1.04–1.05 |
| gradients + off-trajectory, corrected, 3 draws | 9,900 | 11–12 | $0.78$–$1.6\times10^{-3}$ | – / $1.6$–$2.0\times10^{-3}$ | $2.0$–$4.0\times10^{-3}$ at every degree 8–12 | $2.5\times10^{-2}$–$0.10$ |
| exact $\Phi(D(u)) + J\,n$ (diagnosis, 40 kicked test states) | one solve and one Jacobian per query | – | – | – | $1.7\times10^{-4}$ | – |

What the runs show:

- **On the visited set, values alone suffice, at the right degree.** The rule picks degree 10 for the value-only fit, which holds
  $1.5\times10^{-4}$ on all 40 test rollouts with no solve: about 70 times below the tolerance, and 2,000 times below the reduced ChebTT's rollout
  on the same 20 trajectories. This is one fixed split of training and test trajectories.
  The fit is erratic in the degree (7, 8, 11 and 12 diverge), so the rule on the training chains is essential, and it identified every
  case. This supersedes Demonstration 8.5's degree-6 "best" data fit, which only sat at the edge of the degrees searched.
- **Jacobian rows make the fit converge with the degree** (an observation about training-chain scores). The corrected gradient fit's
  training-chain error falls monotonically from degree 8 to 12 in each of three draws of the Jacobian states, at 4,800 plain solves by
  differences. The draws vary the Jacobian states on one trajectory split and share the kick realisations. The rule's choice generalised
  in `g5`, but not at every degree: in `g6` degree 10 passes on the training chains for two draws ($8.8$ and $4.3\times10^{-3}$) and fails
  on the 40 test chains ($1.7$ and $1.2\times10^{-2}$), while the three selected fits pass.
- **The correction is right, and needs the consistent targets.** With $n = z - D(u)$ and $K$ a Hessian bound of $\Phi$ on the segments from
  $D(u)$ to $z$, before projection and clipping,
  $\lVert\Psi_1(z) - \Phi(z)\rVert \le \lVert G(u) - \Phi(D(u))\rVert + \lVert B(u) - J(D(u))\rVert\lVert n\rVert + K\lVert n\rVert^2/2$; only the last term is quadratic. With exact
  pieces the first two vanish: the diagnosis error rises 12.9-fold from $\varepsilon = 0.003$ to 0.01 (11.1 for second order; two kick sizes,
  40 states). With fitted pieces the measured slopes in $\varepsilon$ are 0.96–1.02 over $[0.003, 0.05]$, a finite-range observation that identifies neither
  the dominant term (the fit-value term need not vanish at $n = 0$ and varies across the kicked chart sites) nor an asymptotic order. The on-trajectory derivative rows use $J(X)D'(u)$, which differs from the derivative of $\Phi\circ D$ by at
  most $K\lVert X - D(u)\rVert\lVert D'(u)\rVert$ (the decoder residual's order), and $B$ is fitted to $J(X)$ against $u = \pi X$.
  On a fit to $\Phi(X)$ the correction counts the normal part twice. At degree 9, for example, the value-only fit gives $1.2\times10^{-3}$
  plain and $5.2\times10^{-2}$ corrected, while the consistent fit gives $5.6\times10^{-2}$ plain and $1.5\times10^{-3}$ corrected (review 7).
- **Off the set, data off the trajectories help.** The training states lie on a few trajectories, which are curves in the 3-D chart, and
  between them a high-degree fit is unconstrained. That is consistent with the value-only fit (0.20 at a 0.01 kick, degree 10) and the
  degree-12 gradient fit (0.24) failing. Values *and* derivative rows at chart values off the trajectories (`g6`, one intervention) hold a
  single kick of 0.01 within the tolerance at every degree; which of the two is needed is untested (separate arms at fixed sites are the
  next test). They cost about two degrees on the set: the rollout holds from degree 11, not 9 (prediction Q2), and at the selected degree it
  is 1.9, 2.5 and 2.9 times worse, paired by draw. They also double the solves, to 9,900, about the reduced ChebTT's count.
- **Kicked at every step, none of the tested configurations holds the tolerance.** This rollout is the operator-split proxy: another
  process moves the state between steps. Tested: the value-only fit at degree 10; the gradient fit corrected at degrees 8 and 12 and
  uncorrected at 8; the off-trajectory fit at degree 8 and at the selected degree; one realisation of the kicks, 40 trajectories. The best
  err $2.3\times10^{-2}$–$0.10$, and the value-only fit, the best on the set, errs 2.86. Without the correction the gradient fit errs 1.04, so
  the correction gives a 13–45-fold gain, but not enough. The off-trajectory fit's single-kick error at 0.03 dex is 0.6–$2.2\times10^{-2}$ at
  every degree, consistent with $B$'s degree-2 fit being the remaining first-order term (untested). A single surrogate for both regimes is
  open.
- **The single-state Jacobian condition is not a rollout condition.** Value-only fits with $\max\lVert D\Psi\rVert \approx 15$ (degrees 9
  and 10) hold the rollout, while degree 8 with 16.3 diverges. The gradient fit's constant 1.27 is the dynamics' gain along the chart
  ($\max\lVert J\,D'\,V^{\mathsf T}\rVert = 1.29$), and the deployed corrected surrogate has 1.87 against the dynamics' 1.86 (all sampled norms). The rule now
  uses the chained error alone, which picked fits that pass the unperturbed test chains in every draw (not a separation at every degree;
  see Demonstration 8.5); the product condition of Demonstration 8.5 along the chains is untested.
- **Refusal.** Refusing at $r > \mathrm{tol}/4$ refuses 96–100% of the kicked states, a property of the kicks (every coordinate moved by
  at least 0.003). For the gradient fit the ratio of the separate maxima of the corrected error and the residual is 0.30–0.37 over its four
  kicks (0.75–0.86 for the consistent fit, 2.9–3.6 for the data fit); it is not a pointwise calibration.
  A threshold must be calibrated on the paired errors and residuals of the deployed model, with its accepted-region error and coverage, and
  the fallback tested on full chains. The kicks also
  move the chart value as well as the normal part, so the kicked error mixes the two (review 6 separated them).
- **Provenance.**
  - `g4`'s "consistent" and "gradients" fits were designed from a diagnosis on kicked *test* states, after the first run's results had been
    seen. No test number entered a fit or the rule; a diagnosis on held-out training trajectories would have been cleaner.
  - `g5`'s predictions P1 and P2 were informed by review 7's exploratory runs; P3, P4 and `g6`'s Q1–Q4 by none.
- **Verdicts.**
  - R1 fails as a rule: no degree passed; the fallback degree 6 scored $1.48\times10^{-2}$ on 20 trajectories, within the looser
    test cutoff of twice the tolerance.
  - R2 holds only for the exploratory gradient arm at 0.01, and fails at 0.05 and for the data fit.
  - R3 is refuted for the fitted correction; the exact one is second order.
  - R4 fails both clauses: the corrected rollout is worse than the plain one, and the one-step error ($2.1\times10^{-3}$) exceeds twice the
    data fit's.
  - R5 (the fallback degree 8): its Jacobian clause holds for the plain map (sampled norms: 1.267 against the dynamics' 1.857) and fails
    for the corrected map that is deployed (sampled 1.873, review 7); the two uncorrected kick comparisons and the corrected cutoff at 0.01 hold; the slope fails (0.98
    against 1.7).
  - P1, P2 and P3 hold.
  - P4: the ordering clause holds in all three draws, the three-tolerance clause in one.
  - Q1 holds.
  - Q2 fails.
  - Q3 fails: the selected fit is 3–26% worse than the degree-8 fit without the rows, and over three times the tolerance in two draws.
  - Q4 fails.

**Prior art (section G).** Hermite interpolation and its contour error formula are classical (TH-63); regression with value and gradient
rows is gradient-enhanced least squares (TH-64). The first-order correction about a known state with the step's Jacobian, an accuracy region
and fallback to the integrator is ISAT's (Pope 1997, TH-62). What is specific here is the decoder-centred form $G(\pi z) + B(\pi z)(z - D(\pi z))$
with a fitted Jacobian field and targets with the normal part removed; the contribution sought remains a verified certificate for the deployed map.

## 9. Synthesis

Entries are observations (obs.), estimates (est.) or proofs (thm); no verified bound has been computed.

| | T1 | T2 | T4 | T5 | T6 | §8 |
|---|---|---|---|---|---|---|
| U1 longer steps harder | thm: valid bound, uninformative on the Brusselator; Prop. 1.5 for certificates | | | | | |
| U2 rollout growth | | thm: normal stability (distance only); obs.: rollout/survived ratios 18–30.5 (four configurations), 5.4 (a fifth) | thm: rollout audit (distributional) | | | thm: finite-horizon bound for factored maps; est.: 1.53 vs obs. $5.2\times10^{-4}$ (sampled constants, Robertson); est.: 13–27 times the measured rollout (Demonstration 8.3, half the constants rigorous); obs.: reduced charts avoid divergence on Robertson and HIRES; obs.: HIRES value-only fit $1.5\times10^{-4}$ on 40 rollouts (`g5`), and no tested fit within the tolerance kicked at every step (`g5`, `g6`) |
| U3 Robertson's singular values | | | | | obs.: two large ambient values, one fitted coordinate on one leaf; thm: lower count (Prop. 6.5) | |
| U4 vertex escalation | | | | | | (T7: illustration, direction only) |
| U5 ART-quad vertex failure | | | | thm + obs.: suggests the $L^2$ projection's vertex error (synthetic, 3 of 4) | | |
| U6 vector outputs | | | | | not explained | untested |
| U7 no certificate | | | thm: what an audit does and does not certify | | | |

(T3 explains none of U1–U7; its scope is lattices.)

## 10. Candidate foundations for a scheme, ranked (revised after the external review)

The target is a deterministic worst-case bound on rollouts from a stated set of initial states over a stated horizon, with the audit of
Corollary 4.1′ as a separately labelled distributional guarantee.

1. **Reduced-coordinate surrogates with a verified finite-horizon rollout bound (Theorem 8).** Specify the initial set and horizon; choose a
   chart of the data's local dimension and fit $\hat G$ predicting the next state with invariants conserved on the output; then *verify*
   $\varepsilon_G$, $\varepsilon_u$ on a net of the inputs with a defect modulus and a reference enclosure, and $L_G$, $L_g$ by validated bounds on the fitted
   map. If the verified residual tends to zero under refinement while the gains and tube radii stay controlled, Theorem 8 gives uniform
   convergence on that set; a rate needs a rate for the residual. Demonstrated on one Robertson leaf (8.3). The theorem covers factored maps
   $\hat G\circ\hat\pi$; the HIRES recipe below deploys a corrected, projected map, whose certificate needs a full-state bound on a stated
   input tube (open). On HIRES (8.5) the practical recipe that the evidence supports, run end to end in section G, is:
   - a fit on the training pairs in the chart coordinates, its degree chosen by the chained error of the deployed surrogate on all
     training chains;
   - for rollouts on the visited set, values alone: degree 10 holds $1.5\times10^{-4}$ over all 40 test trajectories with no solve;
   - for inputs off the set, a first-order normal correction on targets with the normal part removed, plus Jacobian rows on and off the
     trajectories: a single kick of 0.01 dex stays within the tolerance at every degree;
   - refusal or error tracking by the solve-free decoder residual (for the uncorrected reduced ChebTT the largest observed ratio of its
     error to $e_0 + r$ was 1.532), with a threshold calibrated on the deployed model's paired errors and residuals.

   None of the tested fits holds a rollout kicked at every step (the operator-split proxy; best $2.3\times10^{-2}$). The training-chain
   score is a selection diagnostic, not a certificate.
2. **Counts as diagnostics (Proposition 6.5, Theorem 6.6, Proposition 6.7(a)–(b)).** Proposition 6.5 is a rigorous obstruction given validated
   local constants; the spectral tests give candidate dimensions of $D\Phi$ and anisotropic sampling directions, not the count of a PCA lift
   (Demonstration 8.4); an upper count needs a verified global embedding or atlas (Example 6.8).
3. **Accuracy off the training set (Proposition 8.2):** conditional on corrected (S), a fit on $\chi(\pi(N))$ and a Lipschitz bound on a domain
   containing both $\hat\pi(N)$ and $\chi(\pi(N))$; the open problem for operator
   splitting, where the deployed map (with its solver fallback) must be certified on every kicked state.
4. **A priori analyticity (T1)**, open: valid wherever tried, informative only on small boxes.
5. **The light cone (T3)**, for lattices with $h^*$, $e^{\nu\tau}$ and $\ell\Delta\tau$ bounded uniformly in $d$.

Proposed for formalisation in Lean later (none is formalised yet): Lemmas 1.1, 1.2, 3.1, 7.2, Propositions 1.3, 1.5, 2.1–2.4, 4.1, 4.2, 4.4, 5.1, 5.2, Corollary 4.1′, Example 6.8 and Theorem 8 are
elementary or self-contained given their cited results; Proposition 6.5 cites Borsuk–Ulam; the Chebyshev bounds, holomorphic dependence,
Fenichel's theorem, Weyl's and Horn's inequalities and the compound-matrix log norm would enter as cited axioms.

## 11. Open items

- §8 and G: one HIRES surrogate for both regimes (on the set and kicked at every step), first through a better fit of $B(u)$ (the
  off-trajectory Jacobians, a higher degree; with the exact pair the error is second order in the kick, with the fitted pair it grew about
  linearly over the kicks tested), then through refusal with the
  solver as fallback at a calibrated residual threshold (the fallback branch then belongs to the deployed map); separate off-trajectory
  value and derivative arms at fixed sites; a rollout bound for the deployed corrected, projected map on a stated split-input tube; a proof of the residual bound under stated hypotheses; POLLU with several hundred trajectories and a decoder at $k = 7$–8
  (Demonstration 8.5); a validated (interval) ODE enclosure to turn Demonstration 8.3's estimated defects into verified ones; a tighter first-order
  version with a remainder and a validity region; Theorem 8 on the benchmark's schemes (a reduced-coordinate ChebTT against box ChebTT on HIRES,
  POLLU, Fisher–KPP); Proposition 8.2 for operator-split inputs.
- T1: the tube in log coordinates on a benchmark box; monotonicity of the true nearest singularity in $\tau$ for a stated class (it fails in
  general, Proposition 1.5; U1).
- T2: the sign autocorrelation of each scheme's weighted defects along trajectories.
- T6: a constructive upper coordinate count for a stated class (verified fibre projection and embedding); a finite-time version of 6.7 with
  accumulated perturbations.
- T7: the direct test on `lotka_volterra` at $d = 20$.

## References (classical results used)

- G. Dahlquist (1958); G. Söderlind, BIT 46 (2006); W. Walter, *Differential and Integral Inequalities*, Springer 1970.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (Thms 4.2, 8.1–8.3, 15.2); Proc. AMS 145 (2017).
- Yu. Ilyashenko, S. Yakovenko, *Lectures on Analytic Differential Equations*, AMS 2008 (Thm 1.1).
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*; P. E. Kloeden, J. Lorenz, SIAM J. Numer. Anal. 23 (1986);
  A. M. Stuart, A. R. Humphries, *Dynamical Systems and Numerical Analysis*, CUP 1996.
- N. Fenichel, Indiana Univ. Math. J. 21 (1971), J. Differential Equations 31 (1979); M. Hirsch, C. Pugh, M. Shub, LNM 583 (1977); C. Kuehn,
  *Multiple Time Scale Dynamics*, Springer 2015; A. Eldering, M. Kvalheim, S. Revzen, *Global linearization and fiber bundle structure of invariant
  manifolds*, arXiv:1711.03646.
- J. S. Muldowney, Rocky Mountain J. Math. 20 (1990); P. Constantin, C. Foias, Comm. Pure Appl. Math. 38 (1985); A. Douady, J. Oesterlé, C. R. Acad.
  Sci. 290 (1980); R. A. Smith, Proc. Roy. Soc. Edinburgh 104A (1986); W. A. Coppel, *Dichotomies in Stability Theory*, LNM 629 (1978).
- R. DeVore, R. Howard, C. Micchelli, Manuscripta Math. 63 (1989); J. Batson et al., arXiv:2102.08380 (§4).
- W. Hoeffding (1963); S. S. Wilks, Ann. Math. Stat. 12 (1941); A. G. Sukharev (1979); E. Novak, LNM 1349 (1988).
- A. Cohen, M. A. Davenport, D. Leviatan, FoCM 13 (2013); A. Cohen, G. Migliorati, SMAI J. Comput. Math. 3 (2017).
- R. E. Caflisch, W. Morokoff, A. Owen, J. Comput. Finance 1 (1997).
- L. Chen, P. Sun, J. Xu, Math. Comp. 76 (2007); M. J. Davis, R. T. Skodje, J. Chem. Phys. 111 (1999); L. A. Segel, M. Slemrod, SIAM Review 31
  (1989); S. B. Pope, Combust. Theory Model. 1 (1997); U. Maas, S. B. Pope, Combust. Flame 88 (1992); S. H. Lam, D. A. Goussis, Int. J. Chem.
  Kinet. 26 (1994); M. Valorani, S. Paolucci, J. Comput. Phys. 228 (2009); N. Brisebarre, M. Joldeş, ISSAC 2010.
- K. Lee, K. Carlberg, J. Comput. Phys. 404 (2020), arXiv:1812.08373 (Thm 4.3); Farenga et al., arXiv:2408.15183 (§§3.2–3.3); C. W. Gear, T. J.
  Kaper, I. G. Kevrekidis, A. Zagaris, SIAM J. Appl. Dyn. Syst. 4 (2005); M. Cenedese et al., Nat. Commun. 13 (2022); A. J. Linot, M. D. Graham,
  Phys. Rev. E 101 (2020); D. Floryan, M. D. Graham, Nat. Mach. Intell. 4 (2022); C. Foias, O. Manley, R. Temam, M2AN 22 (1988); M. Marion, R. Temam,
  SIAM J. Numer. Anal. 26 (1989) (from memory unless a link is given).
- A. Forets, A. Pouly, arXiv:1711.02552; J.-P. Liu et al., PNAS 118 (2021), arXiv:2011.03185.
