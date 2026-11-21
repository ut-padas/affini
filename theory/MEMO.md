# A theory of surrogates for stiff ODE step maps: candidate foundations

This memo collects candidate theorems on which a surrogate scheme for the step maps of stiff ODEs could be built,
each with its assumptions, its proof (or the classical result it rests on), what it predicts, and a numerical check
of that prediction. It is written for the general autonomous system $x' = F(x)$; chemistry networks are one
instance. The theorems were chosen to explain what the benchmark and the convergence studies measured
(`docs/SURROGATE_RESEARCH.md`, `docs/SCHEMES_VS_ART.md`, both companions' `convergence.json`), and each is kept to
the complexity those measurements need. Revision 1 answers the first external review (`theory/REVIEWS.md`, round 1).

- **Status legend.** *Proved*: a complete proof is given here. *Proved, citing*: complete given a classical result
  marked **[C]**. *Sketch*: a step is not written out. *Conjecture*: stated, not proved. Each statement lists its
  assumptions; nothing else is used.
- **Bounds and laws.** Most results are one-sided (an upper bound on a cost, a lower bound on a radius). A
  prediction drawn from a one-sided result is stated as a bound; only two-sided results are read as laws.
- **Checks.** `theory/checks/t*.py` (NumPy and SciPy only, each under three minutes) write
  `theory/checks/data/t*.json`; the review's own tests are in `theory/reviews/scratch-round-1/` (quoted as "review
  test A–E"). Every number below comes from those files or from the repository records named beside it. A check that
  could not have failed (an identity) is labelled as a sanity check, not as evidence.
- **Prior art.** `theory/LITERATURE.md` (IDs `TH-*`); classical results under "References".

**Notation.** $\Lambda$ is the log-norm integral of T1; $\mathcal{L}_N$ the least-squares Lebesgue constant of T5;
$\delta$ a surrogate's defect; $\nu$ the diagonal bound of (A3); $\lambda_i$ eigenvalues; $P = V\,\mathrm{diag}(\lambda)V^{-1}$.

## 0. What known theory already predicts in our data, and what it does not

| Quantity (source) | Classical prediction | Measured |
|---|---|---|
| ART's evaluations against $1/\alpha_{\mathrm{tol}}$, Rosenbrock, $d = 2..5$ (adaptive `convergence.json`, `rates.art`) | piecewise-affine approximation, full-rank Hessian: slopes $d/2 = 1, 1.5, 2, 2.5$ | $1.00, 1.52, 2.10, 2.60$ |
| ART on a ridge, $d = 2, 3, 5$ (global `convergence.json`, `ridge_rates`) | rank-$r$ Hessian: slope $r/2 = 0.5$ (Chen, Sun, Xu 2007) | $0.54, 0.55, 0.82$ ($d = 5$ deviates) |
| ChebTT-M's degree per decade on the ridge (`ridge_degree`) | $1/\log_{10}\rho$ of the ridge profile | predicted $12.0, 14.6, 19.0$; measured $11.9, 14.6, 19.1$ |

Observations no theory in the repository explains yet:

- **U1.** Costs differ by orders of magnitude between dynamical classes, and the 4-D Brusselator's map over
  $2\Delta t$ was harder to fit than two steps of $\Delta t$ (11k against 3.8k solves, `SURROGATE_RESEARCH.md` §1.20).
- **U2.** ChebTT's rollout error on Robertson reaches 36 times its one-step error in 40 steps (§1.12); SlowTable's
  one-step error is 4 times the tolerance, yet only 0.1 of it survives the next exact step (§1.20).
- **U3.** The Robertson step map in log coordinates has singular values about $1.7, 1.0, 10^{-10}$ (§0.4 item 5).
- **U4.** ChebTT's cost on `lotka_volterra` at $d = 20$ varies 26-fold between seeds (3,221 to 85,206), driven by
  checks at the box's vertices (`SCHEMES_VS_ART.md`).
- **U5.** ART with one quadratic leaf wins near-quadratic problems at loose tolerance but fails at the vertices (2.7
  tol); one refit of the same quadratic fails three times in six (`SCHEMES_VS_ART.md`).
- **U6.** Vector outputs: per-output tensor trains cost about $m$ times one output; ART's shared leaves do not.
- **U7.** No acceptance test in the schemes is a certificate: one false accept in 66 (validation 0.04, test 4.86).

## 1. Setting and tools

$x' = F(x)$ on $\Omega \subset \mathbb{R}^d$; $x(t; x_0)$ the solution; the step map $\Phi_\tau(x_0) = x(\tau; x_0)$; a
box $B = \prod_j[a_j, b_j]$ with half-widths $h_j$; a surrogate $\Psi \approx \Phi_\tau$ with defect $\delta = \Psi - \Phi_\tau$.

**Logarithmic norms [C: Dahlquist 1958; Söderlind 2006].** For a norm on $\mathbb{C}^d$ and its induced matrix norm,
$\mu(A) = \lim_{h\to0^+}(\lVert I + hA\rVert - 1)/h$.
(L1) $\mu$ is sublinear and $|\mu(A) - \mu(B)| \le \lVert A - B\rVert$.
(L2) In $\lVert v\rVert_\beta = \max_j|v_j|/\beta_j$: $\mu_\beta(A) = \max_i(\operatorname{Re}a_{ii} + \sum_{j\ne i}|a_{ij}|\beta_j/\beta_i)$.
(L3) If $w' = A(t)w$, then $D^+\lVert w\rVert \le \mu(A(t))\lVert w\rVert$, so $\lVert w(t)\rVert \le \lVert w(0)\rVert e^{\int_0^t\mu(A)}$.
(L4) $\mu(\int_0^1 A(s)\,ds) \le \int_0^1\mu(A(s))\,ds$ for continuous $A$.
(L5) $\mu(A) \ge \max_i\operatorname{Re}\lambda_i(A)$ in every norm.

**Bernstein ellipses and Chebyshev series [C: Trefethen, ATAP, Thms 4.2, 8.1, 8.2, 15.2].** $E_\rho([a,b])$ is the open
ellipse with foci $a, b$ and semi-axes $h(\rho\pm\rho^{-1})/2$; every point of it lies within $h(\rho - \rho^{-1})/2$ of
$[a, b]$, and the distance from $[a, b]$ to its boundary is $h(\rho - 1)^2/(2\rho)$ (attained at the foci). If $f$ is
analytic in the polyellipse $E_\rho$ with $|f| \le M$, then $|c_k| \le 2^{|\operatorname{supp}k|}M\rho^{-k}$; the
tensor Chebyshev–Lobatto interpolant obeys $\lVert f - I_nf\rVert_{\infty,B} \le 2\sum_{k\not\le n}|c_k|$; the
Lebesgue constant of $n + 1$ Chebyshev points is at most $\frac{2}{\pi}\log(n + 1) + 1$.

---

## T1. Analyticity radii of the step map from the vector field

**Question.** Chebyshev and tensor-train rates depend on the Bernstein parameter $\rho$ of the function, which
black-box schemes estimate from samples. For a step map $F$ is known: can $(\rho, M)$ be bounded *a priori*, with
constants that do not collapse for stiff systems? The classical complex Gronwall argument uses the Lipschitz constant
$L$ of $F$ and gives radii like $e^{-L\tau}$, useless when $L \sim 10^4$–$10^8$. The logarithmic norm is one-sided:
large negative eigenvalues do not hurt it.

**Assumptions.** (A1) The real solutions exist for $x_0 \in B$, $t \in [0, \tau]$; $X$ is the set they visit. (A2) For
weights $\beta > 0$, $F$ extends holomorphically to an open neighbourhood of the tube
$T_\beta = \{x + v: x \in X, v \in \mathbb{C}^d, \lVert v\rVert_\beta \le 1\}$.
Let $m(t) = \sup\{\mu_\beta(J_F(x(t; x_0) + v)): x_0 \in B, \lVert v\rVert_\beta \le 1\}$ and
$\Lambda = \max_{t\in[0,\tau]}\int_0^t m \ (\ge 0)$.

**Lemma 1.1 (tube lemma; proved).** Under (A1)–(A2), for $x_0 \in B$ and $\lVert w_0\rVert_\beta < e^{-\Lambda}$ the
complex solution from $x_0 + w_0$ exists on $[0, \tau]$ and $\lVert z(t) - x(t; x_0)\rVert_\beta \le \lVert w_0\rVert_\beta e^{\int_0^t m} < 1$.
*Proof.* With $w = z - x$, as long as $\lVert w\rVert_\beta \le 1$: $w' = A(t)w$, $A = \int_0^1 J_F(x + sw)\,ds$, so by
(L3)–(L4) $\lVert w(t)\rVert_\beta \le \lVert w_0\rVert_\beta e^{\int_0^t m} \le q := \lVert w_0\rVert_\beta e^\Lambda < 1$. On
$[0, t^*)$, $t^*$ the first time $\lVert w\rVert_\beta$ would reach 1 or the solution would end, $z$ stays in the
compact set $\{\lVert v\rVert_\beta \le q\}$ around $X$, where $F$ is bounded; it extends past $t^*$, so $t^* = \tau$. $\square$

**Lemma 1.2 (componentwise tube; proved; review N1).** Let $r: [0, \tau] \to \mathbb{R}^d_{>0}$ be $C^1$ with, for all
$t$, $x_0 \in B$ and $|v_j| \le r_j(t)$, $J = J_F(x(t; x_0) + v)$:
$\operatorname{Re}J_{ii}r_i + \sum_{j\ne i}|J_{ij}|r_j \le r_i'$ ($i = 1..d$), $F$ holomorphic on a neighbourhood of
that tube. Then $|w_{0,j}| \le \theta r_j(0)$, $\theta < 1$, implies existence on $[0, \tau]$ and $|w_j(t)| \le \theta r_j(t)$.
*Proof.* $u(t) = \max_j|w_j|/r_j$. At an index $i$ attaining the maximum,
$D^+|w_i| \le \operatorname{Re}A_{ii}|w_i| + \sum_{j\ne i}|A_{ij}||w_j|$, so
$D^+(|w_i|/r_i) \le u[\operatorname{Re}A_{ii}r_i + \sum_j|A_{ij}|r_j - r_i']/r_i \le 0$: the row condition is convex
in $J$ and passes to the mean $A$. The Dini derivative of a maximum of finitely many functions is at most the
largest over the active indices; so $u$ does not increase. The bootstrap is Lemma 1.1's. $\square$
Lemma 1.1 is the case $r(t) = \beta e^{\int_0^t m - \Lambda}$. Lemma 1.2 allows radii that differ by orders of
magnitude between variables and change in time (a non-normal transient, a light cone, a fast species).

**Theorem 1 (proved, citing ATAP and holomorphic dependence on initial data [C: Ilyashenko–Yakovenko, Thm 1.1]).**
Under Lemma 1.1's hypotheses, for $\theta \in (0, 1)$ let $h_j(\rho_j - \rho_j^{-1})/2 = \theta\beta_je^{-\Lambda}$ (or
$= \theta r_j(0)$ under Lemma 1.2). Each $\Phi_{\tau,i}$ extends holomorphically to $E_\rho$ with
$|\Phi_{\tau,i}| \le M_i := \max_B|x_i(\tau)| + \theta\beta_i$ (resp. $+\theta r_i(\tau)$), so
$|c_k(\Phi_{\tau,i})| \le 2^{|\operatorname{supp}k|}M_i\rho^{-k}$ and

$$\lVert\Phi_{\tau,i} - I_n\Phi_{\tau,i}\rVert_{\infty,B} \le 4M_i\sum_j\frac{\rho_j^{-n_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1}.$$

*Proof.* A point $\zeta = c + A\cos u + ib'\sin u$ of $\partial E_\rho([a, b])$ with $|A\cos u| > h$ has distance
$\sqrt{(A|\cos u| - h)^2 + b'^2\sin^2u} \le b'$ to $[a, b]$, since $(A - b')|\cos u| = h\rho^{-1}|\cos u| \le h$; points
with $|\operatorname{Re}\zeta - c| \le h$ are within $|\operatorname{Im}\zeta| < b'$; interior points lie on smaller
ellipses. So $E_\rho \subset \{x_0 + w_0: x_0 \in B, \lVert w_0\rVert_\beta < e^{-\Lambda}\}$, an open set on which
Lemma 1.1 and [C] make $\Phi_\tau$ holomorphic; $|z_i(\tau)| \le |x_i(\tau)| + \beta_i\theta e^{\int_0^\tau m - \Lambda}$.
The error bound sums $\prod_j w_j(k_j)$, $w(0) = 1$, $w(k) = 2\rho^{-k}$, over $k \not\le n$:
$\prod_jS_j - \prod_js_j \le \sum_j(S_j - s_j)\prod_{l\ne j}S_l$ with $S_j = (\rho_j + 1)/(\rho_j - 1)$,
$S_j - s_j = 2\rho_j^{-n_j}/(\rho_j - 1)$. $\square$

**Proposition 1.3 (a ceiling on certifiable analyticity; proved; review N2).** If $\Phi_{\tau,i}$ is holomorphic on
$E_\rho$ with $|\Phi_{\tau,i}| \le M$, then for each $j$: $(\rho_j - 1)^2/(2\rho_j) \le M/(h_j\lVert\partial_j\Phi_{\tau,i}\rVert_{\infty,B})$.
*Proof.* For $x \in B$ the disc of radius $h_j(\rho_j - 1)^2/(2\rho_j)$ about $x_j$ in the $j$-th variable lies in
$E_{\rho_j}$; Cauchy's estimate. $\square$
Theorem 1 gives a pair $(\rho, M)$ from below; Proposition 1.3 says no pair can do better than the Jacobian allows.
Where $\lVert D\Phi_\tau\rVert$ grows like $e^{\lambda\tau}$ on a bounded set (an attractor), every certificate has
$\rho_j - 1 \lesssim \sqrt{2\rho_jM/h_j}\,e^{-\lambda\tau/2}$. It bounds certificates, not the true degree: an entire
function with a large derivative only needs a large $M$.

**Corollary 1.4 (the bound's three regimes; proved).** If, for tubes $\beta = r\hat\beta$, $m_r(t) \le \mu_0 + cr^p$
($0 < r \le r_{\max}$), the admissible radius $R(\tau) = \sup_r re^{-\Lambda_r}$ is at least
$\min(r_{\max}, (|\mu_0|/c)^{1/p})$ for every $\tau$ if $\mu_0 < 0$; $(pce\tau)^{-1/p}$ if $\mu_0 = 0$;
$e^{-\mu_0\tau}(pce\tau)^{-1/p}$ if $\mu_0 > 0$. With $\rho - 1 \approx \theta R/h$ for small $R$, the degree the bound
certifies for a tolerance is about $\frac{h}{\theta R}[\log(1/\mathrm{tol}) + d\log\frac{2h}{\theta R}]$ (the second term
from the product in Theorem 1). These are regimes of the **bound**, set by the sign of $\mu_0$ (a log norm on a complex
tube), which need not coincide with the dynamical class (C1 below).

**Remark 1.5 (linear invariants; review C4, N6).** If $CF \equiv 0$ ($C$ of rank $q$), then $J$ has the eigenvalue 0
and $\mu(J) \ge 0$ in every norm (L5): $\mu_0 \ge 0$, the first regime is unavailable, and $\Lambda$ grows with $\tau$.
Every closed mass-action network has such invariants (Robertson 1, HIRES 1, POLLU 6). Theorem 1 should then be applied
to the system reduced to an invariant affine leaf $\{Cx = \text{const}\}$ (dimension $d - q$), with the polyellipse in
coordinates of the leaf. Whether reduced Robertson has $\mu_0 < 0$ for some $\beta$ is open.

**Predictions (as bounds).** (P1.1) Theorem 1's $\rho$ never exceeds the true one, on every system satisfying
(A1)–(A2), stiff ones included. (P1.2) The certified degree is bounded in $\tau$ only where $\mu_0 < 0$ on the leaf;
with Proposition 1.3 the true $\rho$ is bracketed between Theorem 1's value and the Jacobian's ceiling.

**Checks.**

- `t1_scalar`: $x' = \mp x - x^3$ on $[-1, 1]$, exact maps. $m_r = \mp1 + 3r^2$ (brute force agrees). Validity
  holds at every $\tau$ (dissipative: $\rho_{\mathrm{thm}} = 1.731$ against $\rho_{\mathrm{exact}} = 2.414$ for
  $\tau \ge 1$, a factor 1.6 in degree; expanding: $\log(\rho - 1)$ slope $-1.14$ against the exact $-1.03$), and every
  resolved coefficient lies below $2M\rho_{\mathrm{thm}}^{-k}$ (largest ratio 0.56). Here the bound's regime and the
  dynamics agree, because $\mu_0$ is attained at the equilibrium $x = 0$.
- Review test B (`b_robertson_t1.py`): Robertson in original coordinates, box at $y(100)$ (half-widths $0.5y$), fast
  rate 4,197, $\beta$ optimised from the Perron vector of the comparison matrix. The bound is valid in 24 of 24
  comparisons, and $\Lambda$ stays $0.04$–$1.75$ while $\tau|\lambda_{\mathrm{fast}}|$ goes from 42 to 42,000: the
  log norm is robust to stiffness, as claimed. Two limits: with $\beta \propto h$ the same computation gives
  $\Lambda = 10^2$–$10^5$ (the weights matter), and since $\mu_\beta \ge 0.022$ for every $\beta$ (Remark 1.5's
  invariant and the comparison matrix), $\Lambda$ grows with $\tau$: the certified degree for 3 digits is 19, 20, 30, 202
  at $\tau = 0.01, 0.1, 1, 10$ against measured 5.0, 4.7, 4.4, 4.7, an overestimate of 4- to 43-fold. The bound does not
  see that the map ignores the fast species ($\rho_{\mathrm{eff}} = \infty$ in $y_2$).
- `t1_lorenz96` and review test A: the decay of fibre coefficients does **not** discriminate the regimes in
  $\tau \in [0.25, 4]$: the slope of $\log(\rho_{\mathrm{eff}} - 1)$ against $\log\tau$ is $-0.8$ to $-1.1$ for
  $F = 0.5, 0.8, 1, 1.2$ and three seeds (the memo's earlier $-1.04$ was one seed at $F = 1$). This range reflects the
  short-step behaviour $\Phi_\tau \approx x + \tau F(x)$ common to all forcings; the regimes separate only for
  $\tau \ge 4$ ($\rho_{\mathrm{eff}}$ levels at 4–5.5 for $F = 0.5$ and keeps falling for $F \ge 1$). On the benchmark's
  box $\mu_\infty(J)$ reaches 5, so the bound's regime there is "expanding" for a system that is dynamically neutral.
  At $F = 8$ (chaotic, $\lambda_{\max} = 1.68$) $\rho_{\mathrm{eff}} - 1$ falls by a factor 5.3 between $\tau = 0.5$ and
  1, a rate 3.3, twice $\lambda_{\max}$; beyond $\tau = 1$ 129 points do not resolve the fibre.

**U1.** Corollary 1.4 is an upper bound on the cost; it is *consistent with* the dependence on the dynamical class
but does not explain U1, which needs a lower bound. The Brusselator's $\Lambda$ and Jacobian growth (Proposition 1.3)
were not computed; that is the first open item.

**Prior art** (`LITERATURE.md`, T1). Holomorphic dependence on initial data and the real log-norm Gronwall bound are
classical; Hansen–Schwab (TH-1) proves holomorphy in affine parameters with Lipschitz constants and a small-data
condition. Carleman-linearization error bounds for quadratic fields (Forets–Pouly 2017; Liu et al. 2021,
arXiv:2011.03185) give all-time convergence of a polynomial representation in the initial state under a negative log
norm of the linear part and a smallness condition, which is this corollary's first regime for quadratic $F$ with
Taylor instead of Chebyshev expansions (review C24; checked from abstracts only). Rigorous integrators (Zgliczyński's
Lohner-type algorithms, CAPD) propagate real enclosures with log norms (review C25; unverified). What remains new as
far as found: general holomorphic $F$, the tube bootstrap with a time-integrated $\Lambda$ and componentwise radii,
Bernstein polyellipses with weights, and the evaluation on a stiff network.

**Limits.** (A2) excludes non-holomorphic right-hand sides (tabulated rates, $\max$). The weights $\beta$ (or $r(t)$)
must be optimised; a fixed $\beta \propto h$ fails. The product in Theorem 1 is exponential in $d$: the bound is useful
in few (reduced) variables.

---

## T2. How a surrogate's defect accumulates

**Proposition 2.1 (global bound; classical).** Let $C$ be convex and forward invariant up to $\tau$, $\mu(J_F) \le m$
on $C$, $\Psi(C) \subset C$, $\sup_C\lVert\delta\rVert \le \varepsilon$. Then $\lVert y_n - x_n\rVert \le \varepsilon\sum_{j<n}e^{jm\tau}$.
*Proof.* (L3)–(L4) on the segment between two trajectories give the contraction factor; induct on
$e_{k+1} = \Phi(y_k) - \Phi(x_k) + \delta(y_k)$. $\square$
With an invariant, $m \ge 0$ (Remark 1.5) and the bound grows at least linearly; the bounded case $m < 0$ applies
only on the reduced leaf. The stiff test problems of §1.20 are neutral in this sense.

**Proposition 2.2 (first-order representation; proved).** If $\Phi$ is $C^2$ on a convex $U$ containing both chains,
$\lVert D^2\Phi\rVert \le K_2$, $G_{n,j} = D\Phi(x_{n-1})\cdots D\Phi(x_j)$:

$$e_n = \sum_{k=0}^{n-1}G_{n,k+1}\,\delta(y_k) + R_n,\qquad \lVert R_n\rVert \le \tfrac{K_2}{2}\sum_{k<n}\lVert G_{n,k+1}\rVert\,\lVert e_k\rVert^2.$$

*Proof.* $e_{k+1} = D\Phi(x_k)e_k + r_k + \delta(y_k)$, $\lVert r_k\rVert \le K_2\lVert e_k\rVert^2/2$; unroll. $\square$
This is the statement for transients, which is what the benchmark's 40-step rollouts are. *Remark (hyperbolic fixed
points, classical).* If $x^*$ is a fixed point of the system reduced by its invariants with $I - P$ invertible and
$\lVert\delta\rVert_{C^1} \le \varepsilon$, then $y^* - x^* = (I - P)^{-1}\delta(x^*) + O(\varepsilon^2)$, the constant
involving $\lVert(I - P)^{-1}\rVert^2K_2$ (so the expansion needs $\varepsilon\lVert(I - P)^{-1}\rVert^2K_2 \ll 1$, a real
restriction along slow modes); if $P$ is diagonalisable a defect along mode $i$ is amplified by $1/|1 - \lambda_i|$.

**Proposition 2.3 (neutral linear dynamics; proved).** If $\Phi(x) = Ux$, $U$ orthogonal, $x_k = U^kx_0$, then
$e_n = U^{n-1}S_n + R_n$ with $S_n = \sum_{k<n}U^{-k}\delta(U^kx_0)$ and $\lVert R_n\rVert \le \mathrm{Lip}(\delta)\,\varepsilon\,n^2/2$
($\varepsilon = \sup\lVert\delta\rVert$). So for $n \ll 1/\mathrm{Lip}(\delta)$, $\lVert e_n\rVert \approx \lVert S_n\rVert$:
it grows like $n\lVert\bar\delta\rVert$ with $\bar\delta$ the defect's average pulled back to the initial frame (for a
planar rotation by $\theta \notin \pi\mathbb{Q}$, the orbit average, by Weyl), and stays bounded when $\bar\delta = 0$
and $\delta$ is a trigonometric polynomial of the phase.

**Proposition 2.4 (accuracy is needed only on an attracting tube; proved; review N3).** Let $S$ be compact with
$\Phi(S) \subset S$, $U_r = \{p: \operatorname{dist}(p, S) \le r\}$, and suppose (i) $\operatorname{dist}(\Phi(p), S) \le \kappa\operatorname{dist}(p, S)$
on $U_r$ with $\kappa < 1$, (ii) $\sup_{U_r}\lVert\delta\rVert \le \varepsilon_1 \le (1 - \kappa)r$. Then $\Psi(U_r) \subset U_r$
and every chain from $y_0 \in U_r$ has $\operatorname{dist}(y_n, S) \le \kappa^n\operatorname{dist}(y_0, S) + \varepsilon_1/(1 - \kappa)$
for all $n$; Proposition 2.2 holds with every supremum over $U_r$.
*Proof.* $\operatorname{dist}(\cdot, S)$ is 1-Lipschitz, so $\operatorname{dist}(\Psi(p), S) \le \kappa\operatorname{dist}(p, S) + \varepsilon_1$;
iterate. $\square$
The normal part of the error does not accumulate; only the tangential part (along $S$) does, through Proposition 2.2.
Under T6, $\kappa$ is the fast contraction per step and $S$ the slow manifold.

**Remark 2.5 (invariants; review N6).** If $CF \equiv 0$, then $Ce_n = \sum_{k<n}C\delta(y_k)$ exactly: the defect's
invariant component is never damped. On Robertson projecting onto the invariant barely helps (rollout 5.05 to 4.93,
§1.20), so this is not the dominant error there.

**Predictions.** (P2.1) The error that survives one exact step, $\lVert\Phi(\Psi(z)) - \Phi(\Phi(z))\rVert \approx \lVert D\Phi\,\delta\rVert$,
predicts the rollout better than the one-step error. (P2.2) A smooth surrogate's defect varies little along a short
trajectory, so its errors add up and it nearly attains the classical bound $\varepsilon A_n$; a defect that changes
sign along the trajectory does not. (P2.3) Accuracy beyond the tube $U_r$ (states not near the attracting set) is not
needed for chained use.

**Checks.**

- P2.1 on the committed table (§1.20, Robertson, five rows; review C6): rollout over the survived error spans 18–41
  (a 2.3-fold range), rollout over the one-step error 0.44–24 (55-fold). P2.1 holds on these rows.
- P2.2: ChebTT's rollout over (one-step $\times A_n$), $\rho$ in §1.20, is 0.67–0.98 on four rows: the smooth surrogate
  attains the classical bound. SlowTable's is 0.28, and there the survived error describes it (0.10 of the tolerance
  survives against 4.03 one-step). The sign autocorrelation of each scheme's defects along a trajectory was not
  measured (open).
- P2.3: SlowTable's only misses on Robertson are in the first step, the initial layer outside the tube, where it also
  refuses 50–90% of the states (§1.12). Consistent; $\kappa$ and the tube width were not measured.
- Sanity checks (identities, could not fail; `t2_rollout`): the Davis–Skodje fixed point of $\Psi$ shifts by
  $10.508\varepsilon$ (slow mode), $1.582\varepsilon$ (fast mode) and $13.79\varepsilon$ (a defect in the slow *variable*
  $(1, 0)$), against the classical $14.1\varepsilon$: because the eigenvectors $(1, 1)$ and $(0, 1)$ are not orthogonal,
  a defect along a coordinate nearly attains the worst case. The split that helps is along the rows of $V^{-1}$ (the
  fibres), not along the species. The rotation identity of Proposition 2.3 holds (1,709.6 against 1,708.3; bounded
  defects at most 20.01 over 4,000 steps).

**Prior art** (T2). Proposition 2.1 is the global error of one-step methods and its stiff form (TH-20); the fixed-point
remark is the persistence of hyperbolic fixed points (TH-18, TH-19); learned flow-map papers use two-sided Lipschitz
versions (TH-15, TH-21). Proposition 2.4 is an elementary invariance argument; its use to define where a surrogate must
be accurate, and P2.1 as an acceptance criterion, are the surrogate-specific parts.

---

## T3. A light cone for locally coupled systems

**Assumptions.** (A3) Along the trajectories from $B$ up to $\tau$: $\operatorname{Re}J_{ii} \le \nu$, $|J_{ij}| \le L_{ij}$
($i \ne j$), $L \ge 0$ on the coupling graph's edges; $D(i, j)$ the directed graph distance; $\ell = \max L_{ij}$,
$\Delta$ the largest out-degree. The dimensionless speed is $\ell\Delta\tau$.

**Lemma 3.1 (proved).** For $x_0 \in B$, entrywise $|\partial\Phi_\tau/\partial x_0| \le e^{\nu\tau}\exp(\tau L)$, and
$[\exp(\tau L)]_{ij} \le \sum_{k\ge D(i,j)}(\tau\ell\Delta)^k/k!$.
*Proof.* $M' = (D + N)M$ with $D = \operatorname{diag}J$; with the diagonal propagator $U$ ($|U_{ii}(t, s)| \le e^{\nu(t-s)}$),
$M = U + \int UNM$. Each Picard iterate is bounded entrywise by the partial sums for $Q = e^{\nu t}\exp(tL)$, which solves
$Q = e^{\nu t}I + \int_0^te^{\nu(t-s)}LQ$; $(L^k)_{ij} = 0$ for $k < D(i, j)$ and $\le(\ell\Delta)^k$. $\square$

**Corollary 3.2 (window surrogate).** With $W_r(i) = \{j: D(i, j) \le r\}$ and $x^W$ frozen at the box centre outside it,
$\sup_B|\Phi_{\tau,i}(x) - \Phi_{\tau,i}(x^W)| \le e^{\nu\tau}\sum_{j\notin W_r(i)}h_j[\exp(\tau L)]_{ij}$. **Corollary 3.3.**
The tensor Chebyshev interpolant of degree $n$ of the window function has unfolding rank at most
$(n + 1)^{\min(|W\cap\mathrm{left}|, |W\cap\mathrm{right}|)}$ across any bond; for a ring in natural order the exponent is
at most $\lfloor|W_r|/2\rfloor$ (for Lorenz-96 $|W_r| = 3r + 1$).

**Scope (review C9).** The bounds are independent of $d$ only if $\ell\Delta\tau$ stays bounded as $d$ grows: lattices
at fixed spacing (Lorenz-96 with more sites, a growing domain). A method-of-lines discretisation refining a *fixed*
domain has $\ell = Dn^2$: the window is the heat kernel's width $n\sqrt{2D\tau}$ sites, a fixed fraction of the system.
Review test D (Fisher–KPP as in the ladder, $\tau = 0.01$): the window radius for $10^{-3}$ is 4, 9, 16, 31 sites for
$n = 10, 20, 40, 80$ (Lemma 3.1's bound: 4, 9, 17, 34, nearly sharp), and the whole system at $10^{-5}$. So T3 does not
apply to the stiff ladder; for those families the relevant structure is the damping of high modes (Theorem 6.5).

**Checks** (`t3_lightcone`, Lorenz-96 at $F = 1$, $d = 40$, the measured trajectory sups $\ell_1 = 2.25$, $\ell_2 = 1.89$,
so the lemma is checked with measured constants). At $\tau = 0.25$ the largest $|\partial\Phi_i/\partial x_j|$ falls from
0.81 at $D = 0$ to $1.9\times10^{-13}$ at $D = 11$, below the bound at every distance. The window error at $r = 4$ is
$2.6, 3.2, 5.2\times10^{-5}$ for $d = 20, 40, 80$ (maxima over 64 samples; they do not increase systematically with $d$:
at $r = 5$ they are 1.8, 1.5, $1.2\times10^{-6}$). At $\tau = 1$ the bound is valid but loose.

**The ladder's records** (`results/benchmark/runs/step_maps/cpu-1rank`, seed 1, tolerance 0.01, all $d$ outputs).
ChebTT box: Allen–Cahn 325, 530, 1,619, 17,103 ($d = 2, 3, 5, 10$); Fisher–KPP 409, 2,324, 14,358, 107,002; Lorenz-96 2,515,
7,230, 339,627 ($d = 4, 5, 10$): per output the cost grows 10- to 54-fold to $d = 10$. SlowTable (visited states): Fisher–KPP
1,138, 2,560, 4,115 ($d = 2, 3, 5$), Allen–Cahn 686, 2,395, 4,423, Lorenz-96 7,437, 11,382 ($d = 4, 5$). SlowTable's cost grows
more slowly on Fisher–KPP, but it is cheaper in only one of the eight pairs, and 2–4.5 times dearer on Allen–Cahn. The
records neither confirm nor test the light cone (Lorenz-96's window, 7–13 sites, covers $d \le 10$).

**Prior art** (T3). Light cones for classical lattices (TH-22 to TH-26), decay of $\exp(tA)$ (TH-28), light cone to
tensor-network rank for quantum propagators (TH-29) and linear dissipative ODE states (TH-30). Windowed dependence and
$d$-independent TT ranks for nonlinear step maps appear new. The rank bound is that of any function of $|W|$ variables:
$d$-independent, exponential in the window.

---

## T4. What can be certified from samples

**Proposition 4.1 (adaptive audits; proved).** Models $s_1, s_2, \ldots$; at round $r$, $N_r$ fresh i.i.d. points from
$\mu$, independent of the past; accept $s_r$ iff all are within the tolerance; stop at the first acceptance. Then
$\Pr[\mu(\text{violation set of the accepted model}) > \eta] \le \sum_r(1 - \eta)^{N_r}$, at most $\alpha$ when
$N_r \ge \ln(2^r/\alpha)/(-\ln(1 - \eta))$. *Proof.* Conditionally on the history $s_r$ is fixed; union bound. $\square$
With one round this is Wilks' non-parametric tolerance limit (1941). The schemes' Sobol sets, free checks, look-ahead
fibres and maxvol pivots are not fresh i.i.d. points, so their acceptances carry no such guarantee (U7).

**Proposition 4.2 (measure to sup; proved).** With $\mu$ uniform on $B$ and $d_B(x, y) = \max_j|x_j - y_j|/(b_j - a_j)$: if
$e \ge 0$ is $L$-Lipschitz in $d_B$ and $\mu\{e > t\} < \eta \le 1$, then $\sup_Be \le t + L\eta^{1/d}$.
*Proof.* If $e(x) > t + Lr$, $r = \eta^{1/d}$, then $e > t$ on $\{d_B(x, \cdot) \le r\}$, of measure at least $r^d$. $\square$
**Corollary 4.3.** $N$ fresh uniform points within $t$ give, with probability $\ge 1 - \alpha$,
$\sup_Be \le t + L(\ln(1/\alpha)/N)^{1/d}$. **Proposition 4.4 (sharpness; proved).** For any $N$ points there is an
$L$-Lipschitz $e \ge 0$ vanishing at all of them with $\sup e \ge L/(2m)$, $m = \lceil(N + 1)^{1/d}\rceil$ (a bump in a
sub-box of side $1/m$ containing no point in its interior). This is the classical optimal-recovery lower bound for
Lipschitz classes (Sukharev 1979; Bakhvalov): sampled sup-norm certificates for Lipschitz functions cost
$N \sim (L/\Delta)^d$.

**Proposition 4.5 (a sampled sup-norm certificate under analyticity; proved, citing Bernstein–Walsh and ATAP 15.2;
review N4).** Let $e = \Phi_{\tau,i} - p$, $p$ a polynomial of degree $n_j$ in $x_j$, $(\rho, M)$ as in Theorem 1. On
$E_\rho$, $|p| \le \lVert p\rVert_{\infty,B}\prod_j\rho_j^{n_j}$ (Bernstein–Walsh, one variable at a time), so $|e| \le M_e := M + \lVert p\rVert_{\infty,B}\prod_j\rho_j^{n_j}$.
If $|e| \le t$ on the tensor Chebyshev–Lobatto grid of degrees $N_j$, then
$\sup_B|e| \le t\prod_j(\frac{2}{\pi}\log(N_j + 1) + 1) + 4M_e\sum_j\frac{\rho_j^{-N_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1}$.
*Proof.* $e = I_Ne + (e - I_Ne)$: the Lebesgue constant bounds the first term, Theorem 1's display (with $M_e$) the
second. $\square$
The audit costs $\prod_j(N_j + 1)$ evaluations, with $N_j \approx n_j + \log(M_e/t)/\log\rho_j$: the order of the
interpolant itself, against $(L/\Delta)^d$ for Lipschitz functions. Still exponential in the number of variables, so it
is a certificate in few (reduced) variables. It is the step from Theorem 1's $(\rho, M)$ to a guarantee for a surrogate.

**Prior art** (T4). Known: adaptive data analysis (TH-38), fill-distance arguments (TH-17), Wilks' tolerance limits,
optimal recovery (Sukharev, Novak's 1988 monograph), Hickernell's cones (TH-33 to TH-35; sup-norm guarantees only in 1-D).

---

## T5. What one least-squares quadratic leaf does

**Proposition 5.1 (Lebesgue inequality; proved).** With $\mathrm{LS}f = \sum_if(x_i)\ell_i$ and
$\mathcal{L}_N = \sup_x\sum_i|\ell_i(x)|$: $\lVert f - \mathrm{LS}f\rVert_\infty \le (1 + \mathcal{L}_N)\inf_{q\in V}\lVert f - q\rVert_\infty$;
and $\mathcal{L}_N \le \sqrt{K/\lambda_{\min}(G)}$, since $\sum_i|\ell_i(x)| \le \sqrt N\lVert\ell(x)\rVert_2 = \sqrt{\varphi(x)^{\mathsf T}G^{-1}\varphi(x)}$
($G$ the empirical Gram matrix in an orthonormal basis $\varphi$, $K = \sup_x|\varphi(x)|^2$).

**Proposition 5.2 (the error at a point; exact identity proved, variance a sketch; review N5).** With $Pf$ the $L^2(\mu)$
projection and $r = f - Pf$: $f(x) - \mathrm{LS}f(x) = r(x) - \sum_i\ell_i(x)r(x_i)$ exactly (LS reproduces $V$). For i.i.d.
points and $\lVert G - I\rVert \le 1/2$, the second term has root-mean-square of order $\lVert r\rVert_{L^2}\sqrt{K(x)/N}$
(the Cohen–Davenport–Leviatan argument evaluated at a point; not written out).

**Checks.** `t5_least_squares` (quadratics on $[-1, 1]^d$, ART's $N = 20p + 1$ uniform points, 5 seeds): the empirical
Gram matrix is well conditioned ($\lambda_{\min} = 0.57$–$0.67$), although $N$ is far below the sufficient size of
Cohen–Davenport–Leviatan's Theorem 1 ($K \le \kappa N/\ln N$, $\kappa = (3\ln\frac32 - 1)/(2 + 2r)$, $r = 1$; TH-40): 121
against 3,987 at $d = 2$, 421 against 25,295 at $d = 5$, 1,321 against 108,292 at $d = 10$. $\mathcal{L}_N$ is bracketed
between the sampled estimate at vertices (4.4, 6.0, 9.7, 18.9, 34.4 for $d = 2, 3, 5, 10, 20$) and the bound
$\sqrt{K/\lambda_{\min}}$ (6.2, 9.1, 14.4, 29.2, 56.7). Review test C ($d = 10$, six seeds, $E_T$ the best sup error of any
quadratic on 4,000 uniform points and 1,000 vertices by linear programming): the least-squares leaf errs 0.5–1.6 times
$E_T$ in the interior and 2–5.7 times at the vertices, nearly seed-independent, and its vertex error matches the $L^2$
projection's for three of four functions. So the Lebesgue bound is far from attained, and the earlier prediction (a
leaf holds only if $E_T \le \mathrm{tol}/(1 + \mathcal{L}_N)$) is withdrawn.

**Prediction (from Proposition 5.2).** A quadratic leaf fitted by least squares errs about $|r(x)|$, the $L^2$
projection's own error, plus a seed-dependent term of relative size $\sqrt{K(x)/N}$: in the bulk about $E$, at the
vertices $|r(\text{vertex})|$. U5's vertex failure (2.7 tol) is that $L^2$ effect, the same as T7's; the three failed
refits in six at margins 0.83–1.17 are a score sitting at the tolerance with $\pm15\%$ scatter.

**Prior art** (T5). Known: Cohen–Davenport–Leviatan (TH-40), weighted and subsampled least squares (TH-41 to TH-44). No
sup-norm guarantee was found for adaptive trees with oblique splits and least-squares polynomial leaves.

---

## T6. The effective dimension of a stiff step map

**Assumption (S) [C: Fenichel 1971, 1979; Hirsch–Pugh–Shub 1977].** A compact, normally attracting, locally invariant
$k$-dimensional manifold $S$ with a $C^1$ invariant stable foliation of a forward-invariant neighbourhood $N$ (up to time
$\tau$): every $p \in N$ has a base point $\pi(p) \in S$ with
$\lVert\Phi_t(p) - \Phi_t(\pi(p))\rVert + \lVert D(\Phi_t - \Phi_t\circ\pi)(p)\rVert \le Ce^{-\gamma t}$, $\gamma$ exceeding
the tangential rates. *Caveats (review C15).* Fenichel's fibres exist in a neighbourhood of $S$ of size independent of
$\varepsilon$; covering a box needs every layer problem to be globally attracted within $N$ (automatic for one fast
variable on an invariant interval, not in general). The derivative estimate needs a $C^1$ foliation with a uniform
rate (true for $C^r$ fields, $r \ge 2$, any $\gamma$ below the normal rate); the theorem numbers are to be checked.

**Theorem 6 ((S) at $t = \tau$).** $\Phi_\tau = \Gamma_\tau\circ\pi + R_\tau$ with $\Gamma_\tau = \Phi_\tau|_S$, a function
on the $k$-manifold, and $\lVert R_\tau\rVert_{C^1} \le Ce^{-\gamma\tau}$. This restates (S); its content is in the
corollaries:

1. (6.1) $\sigma_{k+1}(D\Phi_\tau(p)) \le Ce^{-\gamma\tau}$ for every $p \in N$ ($D(\Gamma_\tau\circ\pi)$ has rank $\le k$; Weyl).
2. (6.2) The best surrogate that ignores a set of coordinates errs by half the oscillation of $\Gamma_\tau\circ\pi$ along
   them, up to $Ce^{-\gamma\tau}$. In the standard form $x' = f$, $\varepsilon y' = g$ ($\pi$'s base point
   $\xi = x + \varepsilon H(x, y)$, $\gamma \to \gamma/\varepsilon$), dropping $y$ costs at most
   $\varepsilon\,\mathrm{Lip}(\Gamma_\tau)\sup_x\operatorname{rad}_yH(x, \cdot) + Ce^{-\gamma\tau/\varepsilon}$ and at least half
   the oscillation of $\Gamma_\tau(\xi(x, \cdot))$: order $\varepsilon$. Outside the standard form the fibre tilt and the rate
   ratio are different numbers (Robertson: relaxing $y_2$ moves $y_1, y_3$ by $O(y_2) \approx 10^{-5}$, while the rate
   ratio is $10^{-4}$–$10^{-6}$).
3. (6.3) Approximating $\Gamma_\tau$ to $\mathrm{tol}/2$ and the base-point map $\xi$ to $\mathrm{tol}/(2\,\mathrm{Lip}\,\Gamma_\tau)$
   suffices. In the standard form, $H$ needs only $\mathrm{tol}/(2\varepsilon\,\mathrm{Lip}\,\Gamma_\tau)$: a tight tolerance in
   $k$ coordinates and a loose one, but still in $d$ variables, for $\xi$ (review C16a). It does not make the cost
   independent of $d$.
4. (6.4) All outputs share $\pi$: $\Phi_{\tau,i} = \Gamma_{\tau,i}\circ\pi$; one coordinate map serves every species.

$\Gamma_\tau$ involves the slow manifold, which is $C^r$ but in general not analytic and not unique, so Theorem 1 does
not apply to $\Gamma_\tau$ itself; it applies to $\Phi_\tau$ on a set, or to the reduced flow on the critical manifold
(analytic when $g$ is) up to $O(\varepsilon)$ (review C16b).

**Theorem 6.5 (singular values from volume contraction; proved, citing Muldowney 1990; review N7).** Let
$\alpha_1(x) \ge \alpha_2(x) \ge \ldots$ be the eigenvalues of the symmetric part of $WJ_F(x)W^{-1}$ ($W$ fixed and
invertible) and $s_m(t) = \sup_{x_0\in B}\sum_{i\le m}\alpha_i(x(t; x_0))$. Then
$\prod_{i\le m}\sigma_i(WD\Phi_\tau(x_0)W^{-1}) \le \exp\int_0^\tau s_m$, hence $\sigma_m(WD\Phi_\tau W^{-1}) \le \exp(\frac1m\int_0^\tau s_m)$.
*Proof.* $\prod_{i\le m}\sigma_i(M) = \lVert\wedge^mM\rVert_2$; $\wedge^m$ of the variational solution solves the linear
equation with the $m$-th additive compound of $WJW^{-1}$, whose Euclidean log norm is the sum of the $m$ largest
eigenvalues of its symmetric part [C]; (L3); and $\sigma_m \le (\prod_{i\le m}\sigma_i)^{1/m}$. $\square$
It needs no fibration, no standard form and no $\varepsilon$, and covers diffusion: for $u' = D\Delta_nu + f(u)$ with
$|f'| \le L_f$, $s_m \le mL_f - D\sum_{i\le m}\lambda_i(-\Delta_n)$ (Ky Fan), independent of $n$ once $n \ge m$.

**Proposition 6.6 (frozen-Jacobian singular values; proved for constant $J$).** If $J$ is constant and diagonalisable,
$J = V\,\mathrm{diag}(\lambda)V^{-1}$, then $e^{\tau\operatorname{Re}\lambda_{(i)}}/\kappa(V) \le \sigma_i(e^{\tau J}) \le \kappa(V)e^{\tau\operatorname{Re}\lambda_{(i)}}$
($\lambda_{(i)}$ ordered by decreasing real part; Horn's inequality $\sigma_i(ABC) \le \lVert A\rVert\sigma_i(B)\lVert C\rVert$ both
ways). So the number of singular values above $\eta$ is $\#\{i: \tau\operatorname{Re}\lambda_i > \ln\eta\}$ up to a factor
$\kappa(V)$ in $\eta$: the effective dimension is a staircase in $\tau$ set by the spectrum, one step per time scale. For
a nonlinear system it is the frozen-coefficient approximation, valid while $J$ varies little over $\tau$ (conjecture in
general).

**Checks.**

- `t6_slowfast`, Michaelis–Menten ($k = 1$): the error of the best function of $s$ alone is $3.46\times10^{-2}$ at
  $\varepsilon = 0.1$ down to $3.51\times10^{-4}$ at $\varepsilon = 10^{-3}$, slope 0.999 in $\varepsilon$ (Corollary 6.2's
  order $\varepsilon$; a fair test). Its $\sigma_2$ law (rate 1.21 against 1.2) and Robertson's $\sigma_3 = e^{-|\lambda|\tau}$ are
  Liouville's formula in disguise when one direction is fast and the others stay of order one (review C17): sanity checks
  only. (Robertson's second singular value is not always of order one: 0.45 at $t_0 = 1$, $\tau = 1$; $6.5\times10^{-3}$ at
  $t_0 = 0.01$, $\tau = 1$.)
- `t6_hires` (HIRES, 8 species, one invariant; states at $t_0 = 1, 10, 100$, steps $\tau = 0.01$ to 10): the number of
  singular values of $D\Phi_\tau$ above $10^{-3}$ and $10^{-6}$ equals Proposition 6.6's count from the eigenvalues of
  $J(x_0)$ in all 24 cases, and the individual values follow $e^{\tau\lambda_i}$ (at $t_0 = 1$, $\tau = 1$: 0.14, 0.10,
  $1.7\times10^{-4}$, $1.9\times10^{-5}$ against 0.12, 0.091, $2.5\times10^{-4}$, $2.8\times10^{-5}$; at $\tau = 10$: $8.0\times10^{-3}$,
  $6.1\times10^{-10}$, $4.0\times10^{-11}$ against $7.4\times10^{-3}$, $5.1\times10^{-10}$, $4.1\times10^{-11}$). The staircase:
  8, 7–8, 5, 3 singular values above $10^{-3}$ at $\tau = 0.01, 0.1, 1, 10$. HIRES is weakly nonlinear (one bilinear
  term), so this supports the frozen approximation only where $J$ varies little. Theorem 6.5 with $W = I$ is valid but
  useless here (bounds up to $10^{13}$ at $\tau = 1$): the symmetric part of a non-normal $J$ has large positive
  eigenvalues, so $W$ must be chosen, as $\beta$ in T1.
- Review test E (Fisher–KPP, $\tau = 0.01$): 7 singular values above $10^{-2}$ and 10 above $10^{-4}$ at both $n = 40$ and 80,
  each of the first twelve below the heat kernel's $0.92e^{(L_f - \lambda_m)\tau}$: the effective dimension of the stiff
  ladder does not grow with $n$, which is the structure a scheme should use there (instead of T3).

**Predictions.** (P6.1) The effective input dimension at a tolerance $\eta$ is the staircase
$\#\{i: \tau\operatorname{Re}\lambda_i > \ln\eta\}$ (Proposition 6.6) and is bounded independently of the grid for
diffusion-dominated families (Theorem 6.5). (P6.2) Ignoring fast variables costs order of the fibre tilt (Corollary 6.2).
(P6.3) A step-map surrogate needs a tight tolerance only in $k$ coordinates; ChebTT-M's metric-aligned coordinates and the
hub expansion are of that form. U3 is the staircase at $\tau \ge 0.1$ (its $10^{-10}$ is the solver's floor; the true value
is exponentially small). U6 is *not* explained here: it was measured on `lotka_volterra_state`, which is not stiff; its
explanation is that every ART point serves every output, while per-output trains repeat the vertex escalation (T7).

**Prior art** (T6). Classical dynamics: Fenichel fibration (TH-49), fibre projections (TH-50 to TH-52), isostables
(TH-53), inertial manifolds (TH-56); ISAT (Pope 1997) builds its ellipsoids from the singular value decomposition of the
mapping gradient and ILDM (Maas–Pope 1992) counts slow modes from the Jacobian; gradient-based dimension reduction
(TH-48) gives only $L^2$ bounds. The sup-norm statements (Corollaries 6.1–6.2, Proposition 6.6's count as an
approximation-complexity statement) appear new; the observation itself is not.

---

## T7. Sup norm on the box against accuracy where the states are

**Proposition 7.1 (proved).** For $f = g(s)$, $s = \frac1d\sum_jx_j$, $x$ uniform on $[-1, 1]^d$: Hoeffding,
$\Pr(|s| \ge t) \le 2e^{-dt^2/2}$; a polynomial accurate to tol on $|s| \le t_\eta = \sqrt{2\ln(2/\eta)/d}$ violates the
tolerance on a set of measure at most $\eta$, while the sup norm on the box needs $|s| \le 1$ (the vertices).
**Lemma 7.2 (proved).** A polynomial of degree $q$ in $s = \sum_jw_jx_j$ has TT ranks at most $q + 1$ in any ordering:
$\sum_pc_ps^p = \sum_{a\le q}s_L^a(\sum_{p\ge a}c_p\binom{p}{a}s_R^{p-a})$.

**Check** (`t7_concentration`; $g = 1/(1 + 0.8s)$, $\rho = 2$, tol and $\eta = 10^{-3}$): sup-norm degree 13 for every $d$;
measure-sense degree 13, 13, 13, 9, 5, 4 from Hoeffding's radius, and 12, 8, 6, 4, 3, 3 from the measured quantile, for
$d = 2, 5, 10, 20, 50, 100$; no violation on $10^5$ uniform points; vertex errors 102, 1,030, 1,495 times the tolerance at
$d = 20, 50, 100$.

**Predictions and their scope (review C18).** (P7.1) For aggregate-driven maps in $d \gtrsim 10$ the box sup norm costs
more rank than accuracy on the bulk, spent on a set of exponentially small measure. This is consistent with U4 *in
direction only*: Lemma 7.2 bounds ranks by 14 against 5–10 here (a factor 1.4–2.8), while U4's records show rank 4–10
against rank 1 (at degrees 1–2) and a 26-fold cost; a direct test on `lotka_volterra` (a polynomial in its aggregates
$\sum_jA_{1j}x_j$, measure-sense against vertex errors) is open. (P7.2) The project's metric is the maximum pointwise
error. The reading that keeps it is the **sup norm on the reachable set** (the tube of Proposition 2.4), not an
$\eta$-quantile under the uniform measure on a box; the vertex cost is the price of demanding the sup norm on states the
dynamics never visit.

**Prior art.** Effective dimension and concentration in high-dimensional integration (Caflisch, Morokoff, Owen 1997);
the hypercube's corners in polynomial approximation (Trefethen 2017); Hoeffding (1963).

---

## 8. Synthesis

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| U1 cost by class; $\Phi_{2\Delta t}$ harder | consistent (upper bound only) | | | | | | |
| U2 rollout growth; SlowTable's damped errors | | **explains** (P2.1 on 5 rows; ChebTT attains the classical bound) | | | | supports (fast = damped) | |
| U3 Robertson's rank 2 | | | | | | **explains** (staircase, Prop. 6.6) | |
| U4 vertex escalation | | | | | | | consistent in direction |
| U5 ART-quad wins and vertex failure | | | | | **explains** (Prop. 5.2: the $L^2$ projection's vertex error) | | same mechanism |
| U6 vector outputs | | | | | | not explained (non-stiff case) | per-output escalation |
| U7 no certificate | | | | restates (non-fresh checks carry no guarantee) | | | |

## 9. Candidate foundations for a scheme, ranked (revised after review round 1)

1. **Accuracy on the attracting tube (Proposition 2.4 with Proposition 2.2 and Proposition 4.1).** Demand the sup norm on
   a tube around the attracting set (the visited states), weight the tolerance by what survives one exact step (P2.1),
   and audit with fresh points on the visited measure. It needs no coordinates, explains U2 quantitatively, matches the
   benchmark's practice of training and testing on visited states, keeps a sup norm (P7.2), and says what to measure
   next: the contraction $\kappa$ and the tube's width per problem.
2. **A priori analyticity inside the tube (Theorem 1 with Lemma 1.2, Proposition 1.3, Proposition 4.5)**, in reduced
   variables (Remark 1.5), as the source of degrees and of certificates. Valid on a stiff network (review test B), but it
   overestimates degrees 4- to 43-fold with a fixed $\beta$; componentwise radii (Lemma 1.2) are the next test.
3. **The effective dimension (Proposition 6.6, Theorem 6.5)**, provable from $J$ without a fibration, and T6's coordinates
   once the base-point map can be learnt: it sets the number of coordinates that need the tight tolerance.
4. **The light cone (T3)**, only for lattices with $\ell\Delta\tau = O(1)$ as $d$ grows.

Lean (later): Lemmas 1.1, 1.2, 3.1, Propositions 1.3, 2.4, 4.1, 4.2, 4.4, 5.1, Lemma 7.2 and the identity of 5.2 are
self-contained and finite-dimensional; the Chebyshev bounds, holomorphic dependence, Fenichel's theorem and the
compound-matrix log norm would enter as cited axioms.

## 10. Open items

- T1: $\Lambda$ and the Jacobian ceiling (Proposition 1.3) on the 4-D Brusselator, against measured fibre coefficients, at
  $\tau = 0.25$ and 0.5 (the test of U1); Lemma 1.2's radii on Robertson at $\tau = 10$ (do they beat Theorem 1's
  1.04–1.35?); reduced Robertson's $\mu_0$.
- T2: the sign autocorrelation of ChebTT's and SlowTable's defects along benchmark trajectories (P2.2); $\kappa$ and the
  tube width (P2.3).
- T6: Proposition 6.6 on a strongly nonlinear multi-scale network (POLLU, the primordial network); a weighted $W$ that makes
  Theorem 6.5 useful on non-normal Jacobians; learning the base-point map.
- T7: the direct test on `lotka_volterra` at $d = 20$.

## References (classical results used)

- G. Dahlquist (1958); G. Söderlind, *The logarithmic norm. History and modern theory*, BIT 46 (2006).
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (Thms 4.2, 8.1, 8.2, 15.2); *Multivariate
  polynomial approximation in the hypercube*, Proc. AMS 145 (2017).
- Yu. Ilyashenko, S. Yakovenko, *Lectures on Analytic Differential Equations*, AMS 2008 (Thm 1.1).
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*, Springer.
- N. Fenichel, *Persistence and smoothness of invariant manifolds for flows*, Indiana Univ. Math. J. 21 (1971); *Geometric
  singular perturbation theory for ordinary differential equations*, J. Differential Equations 31 (1979); M. Hirsch, C. Pugh,
  M. Shub, *Invariant Manifolds*, LNM 583 (1977); C. Kuehn, *Multiple Time Scale Dynamics*, Springer 2015.
- J. S. Muldowney, *Compound matrices and ordinary differential equations*, Rocky Mountain J. Math. 20 (1990).
- W. Hoeffding (1963); S. S. Wilks, *Determination of sample sizes for setting tolerance limits*, Ann. Math. Stat. 12 (1941);
  A. G. Sukharev, *Optimal method of constructing best uniform approximations for functions of a certain class* (1979); E.
  Novak, *Deterministic and Stochastic Error Bounds in Numerical Analysis*, LNM 1349 (1988).
- A. Cohen, M. A. Davenport, D. Leviatan, FoCM 13 (2013); A. Cohen, G. Migliorati, SMAI J. Comput. Math. 3 (2017).
- R. E. Caflisch, W. Morokoff, A. Owen, *Valuation of mortgage-backed securities using Brownian bridges to reduce effective
  dimension*, J. Comput. Finance 1 (1997).
- L. Chen, P. Sun, J. Xu, Math. Comp. 76 (2007); M. J. Davis, R. T. Skodje, J. Chem. Phys. 111 (1999); L. A. Segel, M.
  Slemrod, SIAM Review 31 (1989); S. B. Pope, Combust. Theory Model. 1 (1997); U. Maas, S. B. Pope, Combust. Flame 88 (1992).
- A. Forets, A. Pouly, *Explicit error bounds for Carleman linearization*, arXiv:1711.02552; J.-P. Liu et al., *Efficient
  quantum algorithm for dissipative nonlinear differential equations*, PNAS 118 (2021), arXiv:2011.03185.
