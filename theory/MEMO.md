# A theory of surrogates for stiff ODE step maps: candidate foundations

This memo collects candidate theorems on which a surrogate scheme for the step maps of stiff ODEs could be built,
each with its assumptions, its proof (or the classical result it rests on), what it predicts, and a numerical check
of that prediction. It is written for the general autonomous system $x' = F(x)$; chemistry networks are one
instance. The theorems were chosen to explain what the benchmark and the convergence studies measured
(`docs/SURROGATE_RESEARCH.md`, `docs/SCHEMES_VS_ART.md`, both companions' `convergence.json`), and each is kept to
the complexity those measurements need. Revision 4 answers three internal review rounds and an external one
(`theory/REVIEWS.md`).

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
  rollout over the worst error surviving one exact step is 18–30.5 for five configurations; SlowTable at 0.01 has one-step
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
$\theta\beta_je^{-\Lambda}$ and $M_i = \sup_{x_0}(|x_i(\tau; x_0)| + \theta r_i(\tau; x_0))$.
*Proof.* $w = 0$ stays 0 by uniqueness. Otherwise let $u = \max_j|w_j|/r_j$; at an active index $i$,
$D^+(|w_i|/r_i) \le u[\operatorname{Re}A_{ii}r_i + \sum_{j\ne i}|A_{ij}|r_j - r_i']/r_i \le 0$ (the row condition is convex in $J$, so it holds for
the mean $A$); the Dini derivative of a finite maximum is bounded by the active indices; first exit and continuation as in
Lemma 1.1 (Kamke–Müller comparison, Walter 1970, on a complex tube). $\square$

**Theorem 1 (proved, citing ATAP and [C: Ilyashenko–Yakovenko, Thm 1.1]).** For $\theta \in [0, 1)$ let
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
contained in $D_\tau$ shrinks with $\tau$, so certificates built from complex existence can only get worse. One-sided ($\Phi_\tau$
may continue beyond $D_\tau$); whether the true nearest singularity approaches $B$ monotonically is open (U1).

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

**Proposition 2.3 (neutral linear dynamics; proved).** If $\Phi(x) = Ux$, $U$ orthogonal, then $e_n = U^{n-1}S_n + R_n$ with
$S_n = \sum_{k<n}U^{-k}\delta(U^kx_0)$ and $\lVert R_n\rVert \le \mathrm{Lip}(\delta)\varepsilon n(n - 1)/2$. When the pulled-back orbit average
$\bar\delta$ exists, $S_n = n\bar\delta + o(n)$; a trigonometric polynomial of the phase with zero average has bounded $S_n$ (geometric
sums, for a rotation number off the resonances). $\lVert e_n\rVert \approx \lVert S_n\rVert$ only while $R_n$ is small against $\lVert S_n\rVert$.

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
$W_r(i) = \{j: D(i, j) \le r\}$ at the box centre costs at most $e^{\nu\tau}\sum_{j\notin W_r(i)}h_j[\exp(\tau L)]_{ij}$; this is independent of $d$
when the number of inputs at each distance is bounded (regular bounded-degree lattices; not, e.g., inward trees of large fan-in).
**Corollary 3.3.** The window function's tensor Chebyshev interpolant of degree $n$ has unfolding rank at most
$(n + 1)^{\min(|W\cap\mathrm{left}|, |W\cap\mathrm{right}|)}$ (frozen variables contribute rank one); on a Lorenz-96 ring $|W_r| = \min(d, 3r + 1)$.

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
Then the probability of *ever* accepting a candidate whose violation set has $\mu$-measure $> \eta$ is at most $\sum_r(1 - \eta)^{N_r}$, at
most $\alpha$ for $N_r \ge \ln(2^r/\alpha)/(-\ln(1 - \eta))$. *Proof.* Conditionally on the past, a false pass at round $r$ has probability at
most $(1 - \eta)^{N_r}$; union bound over rounds. $\square$ (One round: Wilks 1941.) It is a statement about the procedure, not a
conditional probability given acceptance, and it is distributional: an arbitrarily large error on a set of measure below $\eta$ passes.
With numerical reference trajectories the tolerance must include a validated reference error.

**Corollary 4.1′ (rollout audit; proved).** With fresh initial states from the application's law as the points and the worst error over
an $n$-step rollout as the score, Proposition 4.1 bounds the probability of ever accepting a surrogate whose $n$-step rollout violates
the tolerance on a set of initial states of measure $> \eta$, at $N_rn$ solver steps. It is the distributional complement of a uniform
certificate. **From measure to sup.** If the rollout score is $L$-Lipschitz on the target set and every ball of radius $h$ centred there
has $\mu$-mass $> \eta$, then measure at most $\eta$ above $t$ implies the uniform bound $t + Lh$ (Proposition 4.2's argument); in $d$
dimensions $h \sim \eta^{1/d}$.

**Proposition 4.2 (measure to sup; proved).** With $d_B(x, y) = \max_j|x_j - y_j|/(b_j - a_j)$ and $\mu$ uniform on $B$, if $e \ge 0$ is
$L$-Lipschitz in $d_B$ and $\mu\{e > t\} < \eta \le 1$, then $\sup_Be \le t + L\eta^{1/d}$ (a $d_B$-ball of radius $h \le 1$ meets $B$ in volume at
least $h^d$). **Corollary 4.3.** $N$ fresh uniform points within $t$ give, with probability $1 - \alpha$, $\sup_Be \le t + L(1 - \alpha^{1/N})^{1/d} \le t + L(\ln(1/\alpha)/N)^{1/d}$.
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

## T5. What one least-squares quadratic leaf does

Unweighted least squares on $N$ points with full column rank; $V_{ij} = \varphi_j(x_i)$ for an orthonormal basis $\varphi$ of the quadratics
under $\mu$; $G = N^{-1}V^{\mathsf T}V$; $K(x) = \lVert\varphi(x)\rVert_2^2$, $K = \sup_xK(x)$.

**Proposition 5.1 (Lebesgue inequality; proved).** $\lVert f - \mathrm{LS}f\rVert_\infty \le (1 + \mathcal{L}_N)\inf_q\lVert f - q\rVert_\infty$, and
$\mathcal{L}_N \le \sqrt{K/\lambda_{\min}(G)}$ since $\sum_i|\ell_i(x)| \le \sqrt N\lVert\ell(x)\rVert_2 = \sqrt{\varphi(x)^{\mathsf T}G^{-1}\varphi(x)}$.

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
zero tilt). **Corollary 6.3 (three budgets).** If $\hat\pi$ lands in $S$ (or $\Gamma_\tau$ and $\hat\Gamma$ share a domain containing both),
$\sup_N\lVert\hat\Gamma\circ\hat\pi - \Phi_\tau\rVert \le \sup_S\lVert\hat\Gamma - \Gamma_\tau\rVert + \mathrm{Lip}(\Gamma_\tau)\sup_N\lVert\hat\pi - \pi\rVert + \sup_N\lVert R_\tau\rVert$.
**Corollary 6.4.** The same factorisation holds componentwise, with one $\pi$ for every output. Under (S) an embedding $\chi: S \to \mathbb{R}^q$
gives a factorisation through $q$ continuous coordinates; $q = k$ needs a global chart.

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
$u' = D\Delta_nu + f(u)$ (Neumann, $W = I$, $|f'| \le L_f$): $s_m \le mL_f - D\sum_{i\le m}\lambda_i(-\Delta_n)$ with $\lambda_i \ge 4(i - 1)^2$ uniformly in $n$, so
at fixed $D, \tau, L_f$ the number of singular values above a fixed threshold is bounded independently of the mesh; this bounds a local rank,
not a coordinate count or a cost.

**Proposition 6.7 (the staircase).** (a) *Proved, constant $J$.* If $J$ is constant and diagonalisable,
$e^{\tau\operatorname{Re}\lambda_{(i)}}/\kappa \le \sigma_i(e^{\tau J}) \le \kappa e^{\tau\operatorname{Re}\lambda_{(i)}}$, $\kappa$ the infimum of $\kappa(V)$ over eigenbases: the count above $\eta$
lies between the frozen counts at $\eta\kappa$ and $\eta/\kappa$ (on POLLU, $\kappa = 99$ to $6.9\times10^3$: thresholds shifted by 2–4 decades each way).
(b) *Proved, one-sided.* If $J(x(t)) = V(t)\Lambda(t)V(t)^{-1}$ with $V \in C^1$, then
$\prod_{i\le m}\sigma_i(D\Phi_\tau) \le (\lVert V(\tau)\rVert\lVert V(0)^{-1}\rVert)^m\exp\int_0^\tau(\sum_{i\le m}\operatorname{Re}\lambda_{(i)} + m\lVert V^{-1}V'\rVert)$ ($z = V^{-1}w$ solves
$z' = (\Lambda - V^{-1}V')z$; Ky Fan; Horn's product inequality). Its $m$-th root bounds $\sigma_m$ above, so it can only show that fewer than $m$
values exceed a threshold; it gives no lower bound. (c) *No path-averaged law.* With $J(t) = R(\omega t)\operatorname{diag}(0, -b)R(\omega t)^{\mathsf T}$,
$0 < |\omega| \ll b$, the instantaneous slow exponent is 0, but in the rotating frame the exponents are $(-b \pm \sqrt{b^2 - 4\omega^2})/2 < 0$, so the
leading singular value decays like $e^{-(\omega^2/b)\tau}$: a uniform-in-$\tau$ comparison with $\exp\int\operatorname{Re}\lambda_{(i)}$ fails however slowly the basis
turns. A finite-time estimate needs explicit accumulated perturbations.

**Checks (observations).** `t6_hires` (HIRES; $t_0 = 1, 10, 100$; $\tau = 0.01$ to 10): the frozen count matches the number of singular values above
$10^{-3}$ and $10^{-6}$ in 24 of 24 cases (thresholds in spectral gaps); individual values follow $e^{\tau\lambda_i}$ (at $t_0 = 1$, $\tau = 1$: 0.14, 0.10,
$1.7\times10^{-4}$, $1.9\times10^{-5}$ against 0.12, 0.091, $2.5\times10^{-4}$, $2.8\times10^{-5}$); Theorem 6.6 with $W = I$ is valid and useless there (to $10^{13}$).
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
must be conserved on the output (Proposition 2.4).

**Proposition 8.2 (accuracy off $M$; conditional on (S); untested).** Under (S), with a chart $\chi$ on the image of $\pi$ (or a specified atlas),
$\hat\pi \approx \chi\circ\pi$ and $\hat G$ fitted on $\chi(\pi(N))$ and Lipschitz there:
$\sup_N\lVert\Psi - \Phi_\tau\rVert \le \sup_{\chi(\pi(N))}\lVert\hat G - \Gamma_\tau\circ\chi^{-1}\rVert + \mathrm{Lip}(\hat G)\sup_N\lVert\hat\pi - \chi\circ\pi\rVert + C\sup_N\operatorname{dist}(\cdot, S)e^{-\gamma\tau}$
(the one-step split of Theorem 6). Data on $M \subset S$ do not give a fit on $\chi(\pi(N))$, and fibre constancy of $\hat\pi$ is not enough unless the
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

**Prior art.** Combustion tabulates on low-dimensional manifolds of progress variables (ILDM, Maas–Pope 1992; FGM, van Oijen–de Goey 2000; FPI;
REDIM, Bykov–Maas 2007; ISAT on RCCE manifolds, Tang–Pope 2002; ICE-PIC, Ren et al. 2006; ISAT–RCCE–GALI, Hiremath et al. 2011; from memory);
constrained runs fill a slow manifold at prescribed chart values (Gear, Kaper, Kevrekidis, Zagaris 2005; TH-51); learned-manifold reduced dynamics
include spectral submanifolds (Cenedese et al. 2022), Linot–Graham 2020, Floryan–Graham 2022 and chemistry autoencoders (Grassi et al. 2022,
Sulzer–Buck 2023, Holdship et al. 2021). Discrete-time error bounds for manifold Galerkin and LSPG (Lee–Carlberg 2020, Thm 4.3) and latent
stability and error propagation with encoder, decoder and latent dynamics through a Lipschitz recurrence (Farenga et al. 2024, arXiv:2408.15183,
§§3.2–3.3) are close to Theorem 8. Theorem 8 is therefore a specialisation of a known technique; a contribution would be a *verified*
end-to-end certificate for a stated class of stiff step maps.

## 9. Synthesis

Entries are observations (obs.), estimates (est.) or proofs (thm); no verified bound has been computed.

| | T1 | T2 | T4 | T5 | T6 | §8 |
|---|---|---|---|---|---|---|
| U1 longer steps harder | thm: valid bound, uninformative on the Brusselator; Prop. 1.5 for certificates | | | | | |
| U2 rollout growth | | thm: normal stability (distance only); obs.: survived/rollout ratios 18–30.5 | thm: rollout audit (distributional) | | | thm: finite-horizon bound; est.: 1.53 vs obs. $5.2\times10^{-4}$; obs.: reduced charts avoid divergence on Robertson and HIRES |
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
   convergence on that set; a rate needs a rate for the residual. First demonstration: one invariant leaf of Robertson or HIRES, with a
   verified chart and an independent reference enclosure.
2. **Counts as diagnostics (Proposition 6.5, Theorem 6.6, Proposition 6.7(a)–(b)).** Proposition 6.5 is a rigorous obstruction given validated
   local constants; the spectral tests give candidate dimensions and anisotropic sampling directions; an upper count needs a verified global
   embedding or atlas (Example 6.8).
3. **Accuracy off the training set (Proposition 8.2):** conditional on corrected (S) and a fit on $\chi(\pi(N))$; the open problem for operator
   splitting, where the deployed map (with its solver fallback) must be certified on every kicked state.
4. **A priori analyticity (T1)**, open: valid wherever tried, informative only on small boxes.
5. **The light cone (T3)**, for lattices with $\ell\Delta\tau = O(1)$.

Lean (later): Lemmas 1.1, 1.2, 3.1, 7.2, Propositions 1.3, 1.5, 2.1–2.4, 4.1, 4.2, 4.4, 5.1, 5.2, Corollary 4.1′, Example 6.8 and Theorem 8 are
elementary or self-contained given their cited results; Proposition 6.5 cites Borsuk–Ulam; the Chebyshev bounds, holomorphic dependence,
Fenichel's theorem, Weyl's and Horn's inequalities and the compound-matrix log norm would enter as cited axioms.

## 11. Open items

- §8: the verified demonstration of item 1 (validated defect and Lipschitz bounds for a fitted reduced map on one leaf); a tighter first-order
  version with a remainder and a validity region; Theorem 8 on the benchmark's schemes (a reduced-coordinate ChebTT against box ChebTT on HIRES,
  POLLU, Fisher–KPP); Proposition 8.2 for operator-split inputs.
- T1: the tube in log coordinates on a benchmark box; monotonicity of the true nearest singularity in $\tau$ (U1).
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
