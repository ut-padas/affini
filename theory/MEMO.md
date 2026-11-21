# A theory of surrogates for stiff ODE step maps: candidate foundations

This memo collects candidate theorems on which a surrogate scheme for the step maps of stiff ODEs could be built,
each with its assumptions, its proof (or the classical result it rests on), what it predicts, and a numerical check
of that prediction. It is written for the general autonomous system $x' = F(x)$; chemistry networks are one
instance. The theorems were chosen to explain what the benchmark and the convergence studies measured
(`docs/SURROGATE_RESEARCH.md`, `docs/SCHEMES_VS_ART.md`, both companions' `convergence.json`), and each is kept to
the complexity those measurements need. Revision 2 answers the second external review (`theory/REVIEWS.md`, rounds
1 and 2).

- **Status legend.** *Proved*: a complete proof is given here. *Proved, citing*: complete given a classical result
  marked **[C]**. *Sketch*: a step is not written out. *Conjecture*: stated, not proved. Each statement lists its
  assumptions; nothing else is used.
- **Bounds and laws.** A prediction drawn from a one-sided result is stated as a bound; only two-sided results are
  read as laws.
- **Checks.** `theory/checks/t*.py` (NumPy and SciPy only, each under three minutes) write `theory/checks/data/t*.json`;
  the reviews' tests are in `theory/reviews/scratch-round-<n>/` (quoted as "review 1 test A", "review 2 test B", ...).
  Every number below comes from those files or from the repository records named beside it. A check that could not have
  failed (an identity) is labelled a sanity check.
- **Prior art.** `theory/LITERATURE.md` (IDs `TH-*`); classical results under "References".

**Notation.** $\Lambda$ is the log-norm integral of T1; $\mathcal{L}_N$ the least-squares Lebesgue constant of T5;
$\delta = \Psi - \Phi_\tau$ a surrogate's defect; $\nu$ the diagonal bound of (A3); $\lambda_i$ eigenvalues;
$P = V\,\mathrm{diag}(\lambda)V^{-1}$; $\operatorname{dist}(p, S)$ the distance from a point to a set.

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
- **U4.** ChebTT's cost on `lotka_volterra` at $d = 20$ varies 26-fold between seeds (3,221 to 85,206), driven by checks
  at the box's vertices (`SCHEMES_VS_ART.md`).
- **U5.** ART with one quadratic leaf wins near-quadratic problems at loose tolerance but fails at the vertices (2.7
  tol); one refit of the same quadratic fails three times in six (`SCHEMES_VS_ART.md`).
- **U6.** Vector outputs: per-output tensor trains cost about $m$ times one output; ART's shared leaves do not.
- **U7.** No acceptance test in the schemes is a certificate: one false accept in 66 (validation 0.04, test 4.86).

## 1. Setting and tools

$x' = F(x)$ on $\Omega \subset \mathbb{R}^d$; $x(t; x_0)$ the solution; the step map $\Phi_\tau(x_0) = x(\tau; x_0)$; a
box $B = \prod_j[a_j, b_j]$ with half-widths $h_j$; a surrogate $\Psi \approx \Phi_\tau$.

**Logarithmic norms [C: Dahlquist 1958; Söderlind 2006].** $\mu(A) = \lim_{h\to0^+}(\lVert I + hA\rVert - 1)/h$.
(L1) $\mu$ is sublinear and $|\mu(A) - \mu(B)| \le \lVert A - B\rVert$.
(L2) In $\lVert v\rVert_\beta = \max_j|v_j|/\beta_j$: $\mu_\beta(A) = \max_i(\operatorname{Re}a_{ii} + \sum_{j\ne i}|a_{ij}|\beta_j/\beta_i)$.
(L3) If $w' = A(t)w$, then $D^+\lVert w\rVert \le \mu(A(t))\lVert w\rVert$, so $\lVert w(t)\rVert \le \lVert w(0)\rVert e^{\int_0^t\mu(A)}$.
(L4) $\mu(\int_0^1 A(s)\,ds) \le \int_0^1\mu(A(s))\,ds$ for continuous $A$. (L5) $\mu(A) \ge \max_i\operatorname{Re}\lambda_i(A)$.

**Bernstein ellipses and Chebyshev series [C: Trefethen, ATAP, Thms 4.2, 8.1, 8.2, 8.3, 15.2].** $E_\rho([a,b])$ is the
open ellipse with foci $a, b$ and semi-axes $h(\rho\pm\rho^{-1})/2$; every point of it lies within $h(\rho - \rho^{-1})/2$
of $[a, b]$; the distance from $[a, b]$ to its boundary is $h(\rho - 1)^2/(2\rho)$, at the foci. If $f$ is analytic in
the polyellipse $E_\rho$ with $|f| \le M$: $|c_k| \le 2^{|\operatorname{supp}k|}M\rho^{-k}$, and
$\lVert f - I_nf\rVert_{\infty,B} \le 2\sum_{k\not\le n}|c_k|$. Conversely (Bernstein, ATAP 8.3), the best-approximation
errors of a function on an interval decay like $\rho_*^{-n}$ with $\rho_*$ the largest ellipse of analyticity, whatever
$M$. The Lebesgue constant of $n + 1$ Chebyshev points is at most $\frac{2}{\pi}\log(n + 1) + 1$.

---

## T1. Analyticity radii of the step map from the vector field

**Question.** Chebyshev and tensor-train rates depend on the Bernstein parameter $\rho$, which black-box schemes estimate
from samples. For a step map $F$ is known: can $(\rho, M)$ be bounded *a priori*, robustly to stiffness? The classical
complex Gronwall argument uses the Lipschitz constant $L$ of $F$ and gives radii like $e^{-L\tau}$; the logarithmic norm is
one-sided, so large negative eigenvalues do not hurt it.

**Assumptions.** (A1) The real solutions exist for $x_0 \in B$, $t \in [0, \tau]$; $X$ is the set they visit. (A2) For
weights $\beta > 0$, $F$ extends holomorphically to an open neighbourhood of
$T_\beta = \{x + v: x \in X, v \in \mathbb{C}^d, \lVert v\rVert_\beta \le 1\}$.
$m(t) = \sup\{\mu_\beta(J_F(x(t; x_0) + v)): x_0 \in B, \lVert v\rVert_\beta \le 1\}$, $\Lambda = \max_{t\le\tau}\int_0^tm \ (\ge 0)$.

**Lemma 1.1 (tube lemma; proved).** For $x_0 \in B$ and $\lVert w_0\rVert_\beta < e^{-\Lambda}$ the complex solution from
$x_0 + w_0$ exists on $[0, \tau]$ and $\lVert z(t) - x(t; x_0)\rVert_\beta \le \lVert w_0\rVert_\beta e^{\int_0^tm} < 1$.
*Proof.* With $w = z - x$, while $\lVert w\rVert_\beta \le 1$: $w' = A(t)w$, $A = \int_0^1J_F(x + sw)\,ds$; (L3)–(L4) give
$\lVert w(t)\rVert_\beta \le q := \lVert w_0\rVert_\beta e^\Lambda < 1$; the solution stays in a compact subset of the tube,
where $F$ is bounded, so it extends to $\tau$. $\square$

**Lemma 1.2 (componentwise tube; proved; review 1 N1, review 2 D5).** For each $x_0 \in B$ let $r(\cdot; x_0): [0, \tau] \to \mathbb{R}^d_{>0}$
be $C^1$ with, for all $t$ and $|v_j| \le r_j(t; x_0)$, $J = J_F(x(t; x_0) + v)$:
$\operatorname{Re}J_{ii}r_i + \sum_{j\ne i}|J_{ij}|r_j \le r_i'$ ($F$ holomorphic near that tube). Then $|w_{0,j}| \le \theta r_j(0; x_0)$,
$\theta < 1$, implies existence on $[0, \tau]$ and $|w_j(t)| \le \theta r_j(t; x_0)$; Theorem 1 then holds with
$\inf_{x_0}r_j(0; x_0)$ in place of $\beta_je^{-\Lambda}$.
*Proof.* $u = \max_j|w_j|/r_j$; at an active index $i$,
$D^+(|w_i|/r_i) \le u[\operatorname{Re}A_{ii}r_i + \sum_j|A_{ij}|r_j - r_i']/r_i \le 0$ (the row condition is convex in
$J$, so it holds for the mean $A$); the Dini derivative of a finite maximum is bounded by the active indices. This is the
Kamke–Müller comparison argument (Walter 1970) on a complex tube. $\square$

**Theorem 1 (proved, citing ATAP and holomorphic dependence on initial data [C: Ilyashenko–Yakovenko, Thm 1.1]).** For
$\theta \in (0, 1)$ let $h_j(\rho_j - \rho_j^{-1})/2 = \theta\beta_je^{-\Lambda}$. Each $\Phi_{\tau,i}$ extends holomorphically
to $E_\rho$ with $|\Phi_{\tau,i}| \le M_i := \max_B|x_i(\tau)| + \theta\beta_i$, so $|c_k| \le 2^{|\operatorname{supp}k|}M_i\rho^{-k}$ and

$$\lVert\Phi_{\tau,i} - I_n\Phi_{\tau,i}\rVert_{\infty,B} \le 4M_i\sum_j\frac{\rho_j^{-n_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1}.$$

*Proof.* Every point of $E_\rho$ is $x_0 + w_0$ with $x_0 \in B$ and $\lVert w_0\rVert_\beta < e^{-\Lambda}$ (the distance
property in §1: a boundary point $c + A\cos u + ib'\sin u$ with $|A\cos u| > h$ is within $b'$ because
$(A - b')|\cos u| = h\rho^{-1}|\cos u| \le h$). Lemma 1.1 and [C] make $\Phi_\tau$ holomorphic there, with
$|z_i(\tau)| \le |x_i(\tau)| + \beta_i\theta e^{\int_0^\tau m - \Lambda}$. The error bound sums $\prod_jw_j(k_j)$
($w(0) = 1$, $w(k) = 2\rho^{-k}$) over $k \not\le n$. $\square$

**Proposition 1.3 (a pointwise ceiling on certificates; proved; review 2 D8, P4).** If $\Phi_{\tau,i}$ is holomorphic on
$E_\rho$ with $|\Phi_{\tau,i}| \le M$, then for every $x \in B$ and $j$: $\operatorname{dist}(x_j, \partial E_{\rho_j}) \le M/|\partial_j\Phi_{\tau,i}(x)|$.
*Proof.* Cauchy's estimate on the largest disc about $x_j$ in $E_{\rho_j}$ (other variables fixed in $B$). $\square$
For $x_j = c_j + sh_j$ and $\rho_j - 1 \ll 1$ the left side is $h_j\ln\rho_j\sqrt{1 - s^2}(1 + o(1))$, so where
$|\partial_j\Phi_\tau| \sim e^{\lambda\tau}$ at an interior point with $M$ bounded, every certificate has
$\ln\rho_j \lesssim (M/h_j)e^{-\lambda\tau}/\sqrt{1 - s^2}$: the exponent of Theorem 1's lower bound with $\lambda$ in
place of $\mu_0$. It constrains certificates $(\rho, M)$; the true asymptotic degree is set by the largest ellipse of
analyticity (Bernstein's converse), whatever $M$.

**Corollary 1.4 (the bound's three regimes; proved).** If, for tubes $\beta = r\hat\beta$, $m_r(t) \le \mu_0 + cr^p$
($0 < r \le r_{\max}$), the admissible radius $R(\tau) = \sup_{r\le r_{\max}}re^{-\Lambda_r}$ is at least
$\min(r_{\max}, (|\mu_0|/c)^{1/p})$ for every $\tau$ if $\mu_0 < 0$; $(pce\tau)^{-1/p}$ if $\mu_0 = 0$; and
$e^{-\mu_0\tau}(pce\tau)^{-1/p}$ if $\mu_0 > 0$, the last two once $(pc\tau)^{-1/p} \le r_{\max}$ (for shorter $\tau$ the
supremum is at $r_{\max}$). With $\rho - 1 \approx \theta R/h$, the certified degree for a tolerance is about
$\frac{h}{\theta R}[\log(1/\mathrm{tol}) + d\log\frac{2h}{\theta R}]$. These are regimes of the **bound**, set by $\mu_0$ (a log
norm on a complex tube), not the dynamical class.

**Proposition 1.5 (composition; proved; review 2 P5).** For entire $F$ let $D_\tau \subset \mathbb{C}^d$ be the initial
data whose complex solution exists on $[0, \tau]$. Then $\Phi_\tau$ is holomorphic on $D_\tau$, $D_{\tau+\sigma} = D_\sigma \cap \Phi_\sigma^{-1}(D_\tau) \subset D_\tau$,
and the largest polyellipse around $B$ inside $D_\tau$ shrinks weakly as $\tau$ grows. *Proof.* Existence on
$[0, \tau + \sigma]$ is existence on $[0, \sigma]$ followed by existence from $\Phi_\sigma(z_0)$ on $[0, \tau]$. $\square$
So every certificate built from the existence of complex solutions (Theorem 1, Lemma 1.2) can only get worse with the
step. It is one-sided: $\Phi_\tau$ may continue analytically beyond $D_\tau$ ($x' = x^2$: $\Phi_\tau = x_0/(1 - x_0\tau)$
continues across the ray $x_0 \ge 1/\tau$ except at a pole). Whether the true nearest singularity of $\Phi_\tau$ moves
monotonically towards $B$ as $\tau$ grows, for a class of fields, is the precise open question behind U1.

**Remark 1.6 (linear invariants; review 1 C4, review 2 D4).** If $CF \equiv 0$ ($C$ of rank $q$), $J$ has the eigenvalue 0
and $\mu(J) \ge 0$ in every norm (L5): $\mu_0 \ge 0$, so the first regime is unavailable in full coordinates. This alone
does not make $\Lambda$ grow (for linear conservative kinetics $\Lambda = 0$); $\Lambda$ grows when
$\inf_\beta\mu_\beta > 0$ on the tube or the tube's nonlinear term is positive. On the invariant leaf the reduced
Robertson system has $\inf_\beta\mu_\beta = -0.0089$ at $y(100)$ (review 2 test D), so the first regime exists pointwise;
on a box of $\pm10\%$ or $\pm50\%$ around that state no $\tau$-uniform tube was found with entrywise majorants.

**Predictions (as bounds).** (P1.1) Theorem 1's $\rho$ never exceeds the true one. (P1.2) The certified degree grows
weakly with $\tau$ (Proposition 1.5) and is bounded in $\tau$ only where $\mu_0 < 0$ on the leaf.

**Checks.**

- `t1_scalar`: $x' = \mp x - x^3$ on $[-1, 1]$, exact maps: valid at every $\tau$ (dissipative $\rho_{\mathrm{thm}} = 1.731$
  against $\rho_{\mathrm{exact}} = 2.414$ for $\tau \ge 1$, a factor 1.6 in degree; expanding: $\log(\rho - 1)$ slope
  $-1.14$ against $-1.03$); resolved coefficients below $2M\rho_{\mathrm{thm}}^{-k}$ (ratio at most 0.56). Here the bound's
  regime and the dynamics agree because $\mu_0$ is attained at the equilibrium.
- Review 1 test B (Robertson, own coordinates, box $\pm50\%$ of $y(100)$, $\beta$ optimised): valid in 24 of 24
  comparisons; $\Lambda = 0.04$–$1.75$ while $\tau|\lambda_{\mathrm{fast}}|$ goes from 42 to 42,000 (robust to stiffness);
  with $\beta \propto h$, $\Lambda = 10^2$–$10^5$; certified degree for 3 digits 19, 20, 30, 202 at $\tau = 0.01, 0.1, 1, 10$
  against measured 5.0, 4.7, 4.4, 4.7.
- Review 2 test C (the 4-D Brusselator on the bounding box of the benchmark's visited states, 0.5–0.8 dex, log
  coordinates): valid and **vacuous**: $\Lambda = 4.42$ and 6.35 at $\tau = 0.25$ and 0.5, certified $\rho - 1 = 3.6\times10^{-3}$
  to $1\times10^{-4}$ against measured $\rho_{\mathrm{eff}} = 3.0$–9.2, certified degrees for two digits 1,272–32,236 against
  measured 2–4. Not stiffness: $2uv - (B + 1)$ reaches about 11 at the box's corner. Proposition 1.3's (endpoint) ceiling
  is $\rho \le 10$ at both steps and does not move, since the Jacobian does not grow.
- Review 2 test D (Lemma 1.2 on reduced Robertson with entrywise majorants): no gain over Theorem 1 ($\rho = 1.07$–1.02
  against 1.44–1.04 on the $\pm50\%$ box); per-trajectory radii give a certificate at $\tau = 10$ where a common one fails.
  The comparison is Theorem 1 at its best against Lemma 1.2 at a cheap bound.
- `t1_lorenz96`, review 1 test A: the fibre-coefficient decay does not discriminate regimes for $\tau \in [0.25, 4]$
  (slopes $-0.8$ to $-1.1$ for $F = 0.5$ to 1.2, three seeds); they separate only for $\tau \ge 4$.

**U1.** Not explained by T1. On the Brusselator it is reproduced only by measurement (review 2 test C, full $13^4$ grid in
log coordinates): the per-axis degree for $10^{-2}$ dex goes from (5, 5, 3, 3) to (7, 7, 4, 4) and the middle unfolding
rank from 4 to 6 when $\tau$ doubles, a cross-cost proxy $\sum_jr_{j-1}r_j(n_j + 1)$ larger by 2.47 (record: 2.9).
Proposition 1.5 shows the direction for existence-based certificates; the statement for the true singularities is open.

**Prior art** (T1). Holomorphic dependence and the real log-norm Gronwall bound are classical; Hansen–Schwab (TH-1)
works in affine parameters with Lipschitz constants; Carleman-linearization bounds for quadratic fields (Forets–Pouly;
Liu et al. 2021) give all-time convergence in the initial state under a negative log norm of the linear part (abstracts
only); rigorous integrators (CAPD) propagate real enclosures with log norms (unverified); Lemma 1.2 is the Kamke–Müller
comparison theorem on a complex tube. New as far as found: the complex tube bootstrap with a time-integrated $\Lambda$
and Bernstein polyellipses with weights, evaluated on stiff networks.

**Limits.** Valid everywhere it was tried, but informative only on small boxes ($\pm50\%$ in own variables, review 1
test B) and vacuous on the benchmark's Brusselator box (review 2 test C). The weights must be optimised. In log
coordinates (where the surrogates work) no tube computation exists, and linear invariants are no longer affine leaves.
The product in Theorem 1 is exponential in $d$.

---

## T2. How a surrogate's defect accumulates, and when a rollout is stable

**Proposition 2.1 (global bound; classical).** Let $C$ be convex and forward invariant up to $\tau$, $\mu(J_F) \le m$ on
$C$, $\Psi(C) \subset C$, $\sup_C\lVert\delta\rVert \le \varepsilon$. Then $\lVert y_n - x_n\rVert \le \varepsilon\sum_{j<n}e^{jm\tau}$.
With an invariant, $m \ge 0$ (Remark 1.6): the bounded case applies only on the reduced leaf.

**Proposition 2.2 (first-order representation; proved).** If $\Phi$ is $C^2$ on the segments $[x_k, y_k]$ with
$\lVert D^2\Phi\rVert \le K_2$ there, and $G_{n,j} = D\Phi(x_{n-1})\cdots D\Phi(x_j)$:

$$e_n = \sum_{k=0}^{n-1}G_{n,k+1}\,\delta(y_k) + R_n,\qquad \lVert R_n\rVert \le \tfrac{K_2}{2}\sum_{k<n}\lVert G_{n,k+1}\rVert\,\lVert e_k\rVert^2.$$

*Proof.* $e_{k+1} = D\Phi(x_k)e_k + r_k + \delta(y_k)$ with $\lVert r_k\rVert \le K_2\lVert e_k\rVert^2/2$ (Taylor on the
segment); unroll. $\square$ The defects enter at the **surrogate's** states $y_k$, not at the visited ones. *Remark
(hyperbolic fixed points, classical).* At a fixed point of the system reduced by its invariants with $I - P$ invertible,
for $\lVert\delta\rVert_{C^1} \le \varepsilon$ small enough the perturbed fixed point exists and
$y^* - x^* = (I - P)^{-1}\delta(x^*) + E$ with $\lVert E\rVert \lesssim \lVert(I - P)^{-1}\rVert^2\lVert D\delta\rVert\varepsilon + \lVert(I - P)^{-1}\rVert^3K_2\varepsilon^2/2$.
It does not apply to Robertson: its reduced equilibrium $(y_1, y_2) = (0, 0)$ has the singular reduced Jacobian
$\begin{pmatrix}-k_1 & k_3\\ k_1 & -k_3\end{pmatrix}$ (review 2 D10).

**Proposition 2.3 (neutral linear dynamics; proved).** If $\Phi(x) = Ux$, $U$ orthogonal, then $e_n = U^{n-1}S_n + R_n$ with
$S_n = \sum_{k<n}U^{-k}\delta(U^kx_0)$ and $\lVert R_n\rVert \le \mathrm{Lip}(\delta)\,\varepsilon\,n^2/2$. For
$n \ll 1/\mathrm{Lip}(\delta)$, $\lVert e_n\rVert \approx \lVert S_n\rVert$, which grows like $n\lVert\bar\delta\rVert$ ($\bar\delta$ the
defect's average pulled back to the initial frame) and stays bounded when $\bar\delta = 0$ and $\delta$ is a trigonometric
polynomial of the phase. Beyond that window the error drifts at a rate of order $\mathrm{Lip}(\delta)\varepsilon$ per step
(review 2 test E: $0.5q\varepsilon^2$ per step for a phase harmonic of order $q$).

**Proposition 2.4 (normal stability of a rollout; proved; review 1 N3, review 2 P1).** Let $S$ be compact with
$\Phi(S) \subset S$ and $U_r = \{p: \operatorname{dist}(p, S) \le r\}$. Suppose the surrogate's **normal gain** satisfies

$$\operatorname{dist}(\Psi(p), S) \le \kappa_\Psi\operatorname{dist}(p, S) + \varepsilon_S\quad(p \in U_r),\qquad \kappa_\Psi < 1,\quad \varepsilon_S \le (1 - \kappa_\Psi)r.$$

Then $\Psi(U_r) \subset U_r$ and every chain from $y_0 \in U_r$ has $\operatorname{dist}(y_n, S) \le \kappa_\Psi^n\operatorname{dist}(y_0, S) + \varepsilon_S/(1 - \kappa_\Psi)$.
*Proof.* Iterate. $\square$ Three ways to meet the hypothesis:
(a) accuracy on the whole tube, $\sup_{U_r}\lVert\delta\rVert \le \varepsilon_S$, with the map's normal contraction
$\operatorname{dist}(\Phi(p), S) \le \kappa\operatorname{dist}(p, S)$: $\kappa_\Psi = \kappa$;
(b) accuracy on $S$ only, $\sup_S\lVert\delta\rVert \le \varepsilon_S$, with $\lVert\delta(p) - \delta(q)\rVert \le \ell\operatorname{dist}(p, S)$
for $q$ a nearest point of $S$: $\kappa_\Psi = \kappa + \ell$ (review 2 P1);
(c) a surrogate whose image lies within $\varepsilon_S$ of $S$: $\kappa_\Psi = 0$ (Theorem 8).
*Invariants (review 2 D2).* A linear invariant is neutral ($\kappa = 1$ along $C$), so with invariants the statement is
used on each leaf, $S \cap \{Cx = c\}$, which requires $C\Psi = C$ (the surrogate conserves them) or samples off the leaf.

**Predictions.** (P2.1) Under Proposition 2.4's hypothesis, the error surviving one exact step,
$\lVert\Phi(\Psi(z)) - \Phi(\Phi(z))\rVert$, predicts the rollout better than the one-step error; without the hypothesis
neither does. (P2.2) For ChebTT on Robertson the defect keeps its sign along a trajectory and the classical bound
$\varepsilon A_n$ is nearly attained; this is an observation about that scheme on that problem, not about smoothness.

**Checks.**

- `t2_slow_surrogate` (Robertson, log coordinates, $\Delta t = 1$, the benchmark's initial states, least squares on the
  visited states of 100 trajectories only, 40 held-out trajectories). The visited states lie on one curve, parametrised
  by $z_1 = \log_{10}y_1$. Worst errors (dex):

  | surrogate | one-step on visited states | on the tube ($\pm0.05$ dex) | rollout, 39 steps |
  |---|---|---|---|
  | V: degree 3 in $(z_1, z_2, z_3)$, increment $z + g(z)$ | $2.9\times10^{-10}$ | 1.74 | 40 (diverges) |
  | V-full: the same, next state directly | $2.8\times10^{-10}$ | 2.04 | 40 (diverges) |
  | S: degree 8 in $z_1$ alone, next state directly | $5.2\times10^{-4}$ | 0.60 | $5.2\times10^{-4}$ |
  | S-incr: degree 8 in $z_1$ alone, increment $z + g(z_1)$ | $3.2\times10^{-3}$ | 0.62 | $1.1\times10^{-2}$ |

  Accuracy on the visited states does not control the rollout (V), and accuracy on the tube is not necessary (S is
  inaccurate there and does not accumulate at all). What decides is the normal gain: S's image lies on the slow curve
  ($\kappa_\Psi \approx 0$, case (c)); an increment surrogate has derivative $I$ across the curve ($\kappa_\Psi \approx 1$: S-incr's
  errors persist without decaying); V has $|\partial\Psi/\partial z_2|$ up to 6.3 (review 2 test A: 6.3 and 1.23 for chains
  that diverge, 0.24 and 0.03 for chains that stay bounded).
- Review 2 test A: the same class fitted with one tube sample per visited state (case (a)) stays within
  $\varepsilon_1/(1 - \kappa)$ of $S$ as Proposition 2.4 predicts (chain offset $3.6\times10^{-3}$ against a tube error
  $6.3\times10^{-3}$; $\kappa_{\mathrm{fast}} \le 1.7\times10^{-5}$, $\kappa_{\mathrm{mass}} = 1.000000$). P2.1 holds for it (rollout over
  survived error 6.8 and 19.6) and fails without the hypothesis ($1.4\times10^3$ to $4\times10^9$).
- The committed table (§1.20, Robertson, five rows of ChebTT and SlowTable, which query the oracle off the visited set or
  use its Jacobian): rollout over survived error 18–41, over one-step error 0.44–24; ChebTT's $\rho$ = rollout /
  (one-step $\times A_n$) is 0.67–0.98.
- Sanity checks (identities; `t2_rollout`): the Davis–Skodje fixed point shifts by $10.508\varepsilon$ (slow mode),
  $1.582\varepsilon$ (fast mode) and $13.79\varepsilon$ (a defect in the slow *variable*), against the classical
  $14.1\varepsilon$: with non-orthogonal eigenvectors a coordinate-direction defect nearly attains the worst case, so the
  useful split is along the fibres (rows of $V^{-1}$).

**Prior art** (T2). Proposition 2.1 is the global error of one-step methods and its stiff form (TH-20); the fixed-point
remark is persistence of hyperbolic fixed points (TH-18, TH-19); persistence of attracting sets under one-step
perturbations is Kloeden–Lorenz (1986) and Stuart–Humphries (1996); restricting a tabulated map to the accessed region
is ISAT's premise (Pope 1997). The surrogate-specific content is the normal-gain condition and its three sufficient
cases, case (c) above all.

---

## T3. A light cone for locally coupled systems

**Assumptions.** (A3) Along the trajectories from $B$ up to $\tau$: $\operatorname{Re}J_{ii} \le \nu$, $|J_{ij}| \le L_{ij}$
($i \ne j$), $L \ge 0$ on the coupling graph's edges; $D(i, j)$ the directed graph distance, $\ell = \max L_{ij}$, $\Delta$
the largest out-degree.

**Lemma 3.1 (proved).** Entrywise $|\partial\Phi_\tau/\partial x_0| \le e^{\nu\tau}\exp(\tau L)$, and
$[\exp(\tau L)]_{ij} \le \sum_{k\ge D(i,j)}(\tau\ell\Delta)^k/k!$. *Proof.* Variation of constants about the diagonal
propagator; each Picard iterate is dominated by the series of $e^{\nu t}\exp(tL)$; $(L^k)_{ij} = 0$ for $k < D(i, j)$. $\square$
**Corollary 3.2 (window).** Freezing the coordinates outside $W_r(i) = \{j: D(i, j) \le r\}$ at the box centre costs at
most $e^{\nu\tau}\sum_{j\notin W_r(i)}h_j[\exp(\tau L)]_{ij}$. **Corollary 3.3.** The window function's tensor Chebyshev
interpolant of degree $n$ has unfolding rank at most $(n + 1)^{\min(|W\cap\mathrm{left}|, |W\cap\mathrm{right}|)}$; for a ring
in natural order the exponent is at most $\lfloor|W_r|/2\rfloor$ ($|W_r| = 3r + 1$ for Lorenz-96).

**Scope.** $d$-independent only while $\ell\Delta\tau$ stays bounded as $d$ grows (lattices at fixed spacing). A
method-of-lines discretisation of a fixed domain has $\ell = Dn^2$, so its window is a fixed fraction of the system:
review 1 test D (Fisher–KPP, $\tau = 0.01$) finds radius 4, 9, 16, 31 sites for $n = 10..80$ at $10^{-3}$ (Lemma 3.1's bound
4, 9, 17, 34) and the whole system at $10^{-5}$. For those families the relevant structure is the effective dimension
(Theorem 6.5).

**Checks** (`t3_lightcone`, Lorenz-96 at $F = 1$, $d = 40$, measured constants $\ell_1 = 2.25$, $\ell_2 = 1.89$). At
$\tau = 0.25$ the largest $|\partial\Phi_i/\partial x_j|$ falls from 0.81 at $D = 0$ to $1.9\times10^{-13}$ at $D = 11$, below
the bound at every distance; the window error at $r = 4$ is $2.6, 3.2, 5.2\times10^{-5}$ and at $r = 5$ $1.8, 1.5, 1.2\times10^{-6}$
for $d = 20, 40, 80$ (maxima over 64 samples). The ladder's records (`results/benchmark/runs/step_maps/cpu-1rank`, seed 1,
tolerance 0.01, all outputs): ChebTT box Allen–Cahn 325, 530, 1,619, 17,103 ($d = 2, 3, 5, 10$), Fisher–KPP 409, 2,324, 14,358,
107,002, Lorenz-96 2,515, 7,230, 339,627 ($d = 4, 5, 10$); SlowTable Fisher–KPP 1,138, 2,560, 4,115, Allen–Cahn 686, 2,395, 4,423,
Lorenz-96 7,437, 11,382 (cheaper than ChebTT in one of eight pairs). They do not test the light cone (Lorenz-96's window
covers $d \le 10$).

**Prior art** (T3). Light cones for classical lattices (TH-22 to TH-26), decay of $\exp(tA)$ (TH-28), light cone to rank
for quantum propagators (TH-29) and linear dissipative ODE states (TH-30); new as far as found for nonlinear step maps.

---

## T4. What can be certified from samples

**Proposition 4.1 (adaptive audits; proved).** Models $s_1, s_2, \ldots$; at round $r$, $N_r$ fresh i.i.d. points from
$\mu$, independent of the past; accept $s_r$ iff all are within the tolerance. Then
$\Pr[\mu(\text{violation set of the accepted model}) > \eta] \le \sum_r(1 - \eta)^{N_r}$, at most $\alpha$ when
$N_r \ge \ln(2^r/\alpha)/(-\ln(1 - \eta))$. (One round: Wilks 1941.) The measure $\mu$ must be where the chain goes: by
Proposition 2.2 the defects that matter are at the surrogate's own chain states, so a chained surrogate is audited on its
chain states (as `chebtt_roll` does) or on the tube, not on the visited states alone (review 2 D1).

**Proposition 4.2 (measure to sup; proved).** With $d_B(x, y) = \max_j|x_j - y_j|/(b_j - a_j)$, if $e \ge 0$ is $L$-Lipschitz
in $d_B$ and $\mu\{e > t\} < \eta \le 1$ (uniform $\mu$), then $\sup_Be \le t + L\eta^{1/d}$. **Corollary 4.3.** $N$ fresh
uniform points within $t$ give, with probability $1 - \alpha$, $\sup_Be \le t + L(\ln(1/\alpha)/N)^{1/d}$.
**Proposition 4.4 (sharpness; proved).** For any $N$ points there is an $L$-Lipschitz $e \ge 0$ vanishing at all of them
with $\sup e \ge L/(2\lceil(N + 1)^{1/d}\rceil)$: the classical optimal-recovery bound (Sukharev 1979; Novak 1988).

**Proposition 4.5 (a certificate under analyticity; proved; review 2 D11, P6).** Let $p$ have degree $n_j \le N_j$ in
$x_j$ and $(\rho, M)$ be as in Theorem 1. If $|\Phi_{\tau,i} - p| \le t$ on the tensor Chebyshev–Lobatto grid of degrees
$N_j$, then

$$\sup_B|\Phi_{\tau,i} - p| \le t\prod_j\Bigl(\tfrac{2}{\pi}\log(N_j + 1) + 1\Bigr) + 4M\sum_j\frac{\rho_j^{-N_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1}.$$

*Proof.* $I_Np = p$ for $N \ge n$, so $\Phi - p = I_N(\Phi - p) + (\Phi - I_N\Phi)$: the Lebesgue constant bounds the first
term and Theorem 1 the second. $\square$ The grid of degrees $N_j \approx \log(4M/t)/\log\rho_j$ already yields $I_N\Phi$, so
the proposition certifies a compressed $p$ at the price of the uncompressed interpolant; with Theorem 1's present $\rho$
on the benchmark's boxes (T1 checks) that grid has thousands of points per axis. Elementary; validated-numerics tools do
the same (Brisebarre–Joldeş 2010, unverified).

---

## T5. What one least-squares quadratic leaf does

**Proposition 5.1 (Lebesgue inequality; proved).** $\lVert f - \mathrm{LS}f\rVert_\infty \le (1 + \mathcal{L}_N)\inf_{q\in V}\lVert f - q\rVert_\infty$,
and $\mathcal{L}_N \le \sqrt{K/\lambda_{\min}(G)}$ since $\sum_i|\ell_i(x)| \le \sqrt N\lVert\ell(x)\rVert_2 = \sqrt{\varphi(x)^{\mathsf T}G^{-1}\varphi(x)}$
($G$ the empirical Gram matrix in an orthonormal basis $\varphi$, $K = \sup_x|\varphi(x)|^2$).

**Proposition 5.2 (the error at a point; proved; review 1 N5, review 2 D12).** With $Pf$ the $L^2(\mu)$ projection and
$r = f - Pf$: $f(x) - \mathrm{LS}f(x) = r(x) - \sum_i\ell_i(x)r(x_i)$ exactly. For the estimator with the exact Gram matrix
($G = I$), the second term is $\frac1N\sum_ik_x(x_i)r(x_i)$, $k_x(y) = \varphi(x)^{\mathsf T}\varphi(y)$, of mean 0 and variance
exactly $\frac1N\mathbb{E}[k_x(X)^2r(X)^2] \le \lVert r\rVert_\infty^2K(x)/N$ ($\mathbb{E}k_x^2 = K(x)$); with the empirical $G$ this holds
up to the factor $\lVert G^{-1}\rVert$.

**Checks.** `t5_least_squares` (ART's $N = 20p + 1$ uniform points, 5 seeds): $\lambda_{\min} = 0.57$–$0.67$, although $N$ is
far below Cohen–Davenport–Leviatan's sufficient size (their Thm 1, $K \le \kappa N/\ln N$, $\kappa = (3\ln\frac32 - 1)/(2 + 2r)$,
$r = 1$; TH-40): 121 against 3,987 at $d = 2$, 1,321 against 108,292 at $d = 10$. $\mathcal{L}_N$ lies between the sampled vertex
estimate (4.4, 6.0, 9.7, 18.9, 34.4 for $d = 2, 3, 5, 10, 20$) and $\sqrt{K/\lambda_{\min}}$ (6.2, 9.1, 14.4, 29.2, 56.7). Review 1
test C ($d = 10$, $E_T$ the best sup error by linear programming): the leaf errs 0.5–1.6 $E_T$ in the interior and 2–5.7 $E_T$ at
the vertices, nearly seed-independent; the vertex error matches the $L^2$ projection's for three of four functions.

**Prediction.** A least-squares quadratic leaf errs about $|r(x)|$ (the $L^2$ projection's own error) plus a seed-dependent
term of relative size $\sqrt{K(x)/N}$. U5's vertex failure is that $L^2$ effect in three of four tested functions, the
mechanism of T7; the three failed refits in six at margins 0.83–1.17 are a score at the tolerance with $\pm15\%$ scatter.

**Prior art** (T5). Cohen–Davenport–Leviatan (TH-40), weighted and subsampled least squares (TH-41 to TH-44). No sup-norm
guarantee was found for adaptive trees with oblique splits and least-squares polynomial leaves.

---

## T6. The effective dimension of a stiff step map

**Assumption (S) [C: Fenichel 1971, 1979; Hirsch–Pugh–Shub 1977].** A compact, normally attracting, locally invariant
$k$-dimensional manifold $S$ with a $C^1$ invariant stable foliation of a forward-invariant neighbourhood $N$: every
$p \in N$ has a base point $\pi(p) \in S$ with $\lVert\Phi_t(p) - \Phi_t(\pi(p))\rVert + \lVert D(\Phi_t - \Phi_t\circ\pi)(p)\rVert \le Ce^{-\gamma t}$,
$\gamma$ above the tangential rates. *Caveats.* Fenichel's fibres exist near $S$; covering a box needs every layer problem to
be globally attracted within $N$; the derivative estimate needs a $C^1$ foliation with a uniform rate ($C^r$ fields,
$r \ge 2$); theorem numbers to be checked.

**Theorem 6 ((S) at $t = \tau$).** $\Phi_\tau = \Gamma_\tau\circ\pi + R_\tau$, $\Gamma_\tau = \Phi_\tau|_S$, $\lVert R_\tau\rVert_{C^1} \le Ce^{-\gamma\tau}$.
Corollaries: (6.1) $\sigma_{k+1}(D\Phi_\tau(p)) \le Ce^{-\gamma\tau}$ on $N$ (Weyl). (6.2) The best surrogate that ignores a set of
coordinates errs by half the oscillation of $\Gamma_\tau\circ\pi$ along them, up to $Ce^{-\gamma\tau}$; in the standard form
$x' = f$, $\varepsilon y' = g$ ($\xi = x + \varepsilon H$) that is order $\varepsilon$. Outside the standard form the fibre tilt and the
rate ratio are different numbers (Robertson: $O(y_2) \approx 10^{-5}$ against $10^{-4}$–$10^{-6}$). (6.3) Approximating
$\Gamma_\tau$ to $\mathrm{tol}/2$ and $\pi$ to $\mathrm{tol}/(2\,\mathrm{Lip}\,\Gamma_\tau)$ suffices: a tight tolerance in $k$
coordinates, a loose one, still in $d$ variables, for $\pi$. (6.4) $\Phi_{\tau,i} = \Gamma_{\tau,i}\circ\pi + R_{\tau,i}$ for every
output with the same $\pi$. $\Gamma_\tau$ involves the slow manifold, which is $C^r$ but in general not analytic, so Theorem 1
applies to $\Phi_\tau$ on a set, not to $\Gamma_\tau$.

**Proposition 6.5 (a lower bound: fewer coordinates cannot work; proved; review 2 P2).** Let $\Phi$ be $C^2$ on the ball
$B(p, r)$ with $\lVert D^2\Phi\rVert \le K_2$ and $\sigma_m(D\Phi(p)) = s$. For every continuous $c: B(p, r) \to \mathbb{R}^{m-1}$ and
every map $G$: $\sup_{B(p,r)}\lVert\Phi - G\circ c\rVert \ge sr - \frac12K_2r^2$ ($\ge s^2/(2K_2)$ at $r = s/K_2$).
*Proof.* On the $(m - 1)$-sphere of radius $r$ in the span $E$ of the first $m$ right singular vectors, Borsuk–Ulam gives
$v$ with $c(p + v) = c(p - v)$, so $G\circ c$ agrees at $p \pm v$; and
$\lVert\Phi(p + v) - \Phi(p - v)\rVert \ge 2\lVert D\Phi(p)v\rVert - K_2r^2 \ge 2sr - K_2r^2$; one of the two errors is at least half. $\square$
With Corollary 6.1 the effective dimension is two-sided: $k$ coordinates suffice up to $Ce^{-\gamma\tau}\operatorname{diam}$,
and fewer than $m$ continuous coordinates, linear or not, cost at least $\sigma_mr - K_2r^2/2$. Robertson at $\tau = 1$
($\sigma_2 \approx 0.8$–1, $\sigma_3 \le 10^{-10}$): exactly 2 coordinates for every tolerance between the two.

**Theorem 6.6 (singular values from volume contraction; proved, citing Muldowney 1990).** With
$\alpha_1(x) \ge \alpha_2(x) \ge \ldots$ the eigenvalues of the symmetric part of $WJ_F(x)W^{-1}$ ($W$ fixed) and
$s_m(t) = \sup_{x_0\in B}\sum_{i\le m}\alpha_i(x(t; x_0))$: $\prod_{i\le m}\sigma_i(WD\Phi_\tau W^{-1}) \le \exp\int_0^\tau s_m$, hence
$\sigma_m \le \exp(\frac1m\int_0^\tau s_m)$. *Proof.* $\prod_{i\le m}\sigma_i = \lVert\wedge^m\cdot\rVert_2$, the $m$-th additive
compound's Euclidean log norm is the sum of the $m$ largest eigenvalues of the symmetric part [C], and (L3). $\square$ For
$u' = D\Delta_nu + f(u)$ with $|f'| \le L_f$: $s_m \le mL_f - D\sum_{i\le m}\lambda_i(-\Delta_n)$ (Ky Fan), and the Neumann
eigenvalues $4n^2\sin^2\frac{(i-1)\pi}{2n} \ge 4(i - 1)^2$ are bounded below uniformly in $n$, so the bound on the number of
singular values above a threshold does not grow with the grid. This is the Constantin–Foias trace argument (Douady–Oesterlé,
R. A. Smith for ODEs); its use for a surrogate's input dimension is the new part.

**Proposition 6.7 (the staircase; proved for constant $J$, sketch in general; review 2 D3, P3).** (a) If $J$ is constant and
diagonalisable, $e^{\tau\operatorname{Re}\lambda_{(i)}}/\kappa \le \sigma_i(e^{\tau J}) \le \kappa e^{\tau\operatorname{Re}\lambda_{(i)}}$ with
$\kappa$ the infimum of $\kappa(V)$ over eigenbases (from $\sigma_i(ABC) \le \lVert A\rVert\sigma_i(B)\lVert C\rVert$, a consequence of
the min-max characterisation, both ways); the number of singular values above $\eta$ lies between the counts
$\#\{i: \tau\operatorname{Re}\lambda_i > \ln(\eta\kappa)\}$ and $\#\{i: \tau\operatorname{Re}\lambda_i > \ln(\eta/\kappa)\}$. (b) Along a trajectory
with $J(x(t)) = V(t)\,\mathrm{diag}(\lambda(t))V(t)^{-1}$, an exponential dichotomy between the groups of eigenvalues to be
separated, and $\int_0^\tau\lVert V^{-1}V'\rVert$ small, the singular values lie within $\kappa(V(0))\kappa(V(\tau))(1 + o(1))$ of
$\exp\int_0^\tau\operatorname{Re}\lambda_{(i)}(x(t))\,dt$ (Levinson's theorem; Coppel 1978, Eastham 1989; finite-$\tau$ constants not
written out). The effective dimension at tolerance $\eta$ is the staircase $\#\{i: \int_0^\tau\operatorname{Re}\lambda_i > \ln\eta\}$,
computable from the Jacobians a solver forms along the step.

**Checks.**

- `t6_hires` (HIRES, 8 species; states at $t_0 = 1, 10, 100$; $\tau = 0.01$ to 10): the frozen count (eigenvalues of
  $J(x_0)$) matches the number of singular values above $10^{-3}$ and $10^{-6}$ in 24 of 24 cases; individual values follow
  $e^{\tau\lambda_i}$ (at $t_0 = 1$, $\tau = 1$: 0.14, 0.10, $1.7\times10^{-4}$, $1.9\times10^{-5}$ against 0.12, 0.091, $2.5\times10^{-4}$,
  $2.8\times10^{-5}$); staircase 8, 7–8, 5, 3 above $10^{-3}$ at $\tau = 0.01, 0.1, 1, 10$. Theorem 6.6 with $W = I$ is valid and
  useless there (bounds to $10^{13}$): non-normal Jacobians need a weight $W$.
- Review 2 test B: the frozen count holds on POLLU (20 species, 6 invariants; 24 of 24 at both thresholds, gaps at most 1.1
  decades) and on the 4-D Brusselator (18 of 20 and 20 of 20; the miss is a value at the threshold), and fails on
  Robertson at $\tau = 100$ from early states by 11–16 decades (count 1 against 2: it undercounts, the unsafe direction).
- Review 2 test E: the path-averaged exponent of (b) reproduces Robertson's $\sigma_2$ within 3–15% in 12 of 12 cases,
  including $\tau = 1000$ where the frozen one is off by 170 decades.
- `t6_slowfast` (Michaelis–Menten): the best function of $s$ alone errs $3.46\times10^{-2}$ at $\varepsilon = 0.1$ down to
  $3.51\times10^{-4}$ at $10^{-3}$, slope 0.999 (Corollary 6.2; a fair test). Its $\sigma_2$ law and Robertson's
  $\sigma_3 = e^{-|\lambda|\tau}$ are Liouville's formula when one direction is fast: sanity checks.
- Review 1 test E (Fisher–KPP, $\tau = 0.01$): 7 singular values above $10^{-2}$ and 10 above $10^{-4}$ at $n = 40$ and 80.

**Predictions.** (P6.1) The effective input dimension at tolerance $\eta$ is the path-averaged staircase, two-sided by
Proposition 6.5, and bounded in the grid size for diffusion-dominated families. (P6.2) Ignoring fast variables costs the
fibre tilt. (P6.3) Accuracy is needed tightly only in $k$ coordinates (with Theorem 8: and only on $S$). U3 is the
staircase at the benchmark's $\tau = 1$ (its $10^{-10}$ is the solver's floor). U6 is not explained here: it was measured on
`lotka_volterra_state`, which is not stiff.

**Prior art** (T6). Fenichel fibration (TH-49), fibre projections (TH-50 to TH-52), isostables (TH-53), inertial manifolds
(TH-56); ISAT (Pope 1997) and ILDM (Maas–Pope 1992) use the Jacobian's singular values or modes; CSP (Lam–Goussis 1994) and
the G-Scheme (Valorani–Paolucci 2009) count exhausted modes by $\tau|\lambda_i|$; gradient-based dimension reduction (TH-48)
gives $L^2$ bounds. Borsuk–Ulam lower bounds are the tool of continuous nonlinear widths (DeVore, Howard, Micchelli 1989;
from memory). New as far as found: the two-sided sup-norm effective dimension of a step map (Corollary 6.1 with
Proposition 6.5) and its path-averaged staircase as an approximation-complexity statement.

---

## T7. Sup norm on the box against accuracy where the states are

**Proposition 7.1 (proved).** For $f = g(s)$, $s = \frac1d\sum_jx_j$, $x$ uniform on $[-1, 1]^d$: $\Pr(|s| \ge t) \le 2e^{-dt^2/2}$,
so a polynomial accurate on $|s| \le t_\eta = \sqrt{2\ln(2/\eta)/d}$ violates the tolerance on measure at most $\eta$, while the
box's sup norm needs $|s| \le 1$. **Lemma 7.2 (proved).** A polynomial of degree $q$ in $s = \sum_jw_jx_j$ has TT ranks at most
$q + 1$ in any ordering.

**Check** (`t7_concentration`; $g = 1/(1 + 0.8s)$, tol and $\eta = 10^{-3}$): sup-norm degree 13 for every $d$; measure-sense
degree 13, 13, 13, 9, 5, 4 (Hoeffding) and 12, 8, 6, 4, 3, 3 (measured quantile) for $d = 2, 5, 10, 20, 50, 100$; vertex errors
102, 1,030, 1,495 tol at $d = 20, 50, 100$.

**Scope.** (P7.1) Consistent with U4 in direction only (ranks at most 14 against 5–10 here; U4 shows rank 4–10 against 1);
the direct test on `lotka_volterra` is open. (P7.2) The metric that keeps the maximum pointwise error is the sup norm where
the chain goes (Proposition 2.4's tube, on each invariant leaf, or $S$ with Theorem 8), not an $\eta$-quantile on a box; the
vertex cost is the price of demanding the sup norm on states the dynamics never visit.

**Prior art.** Caflisch, Morokoff, Owen 1997 (effective dimension); Trefethen 2017 (the hypercube's corners); Hoeffding 1963.

---

## 8. Theorem 8: surrogates in slow coordinates

**Theorem 8 (proved, given (S)).** Under (S), let $\hat\pi: N \to \mathbb{R}^k$ approximate a chart of the base-point map,
$\hat\pi \approx \chi\circ\pi$ with $\chi: S \to \mathbb{R}^k$ a chart, and let $\hat G$ approximate $\Gamma_\tau\circ\chi^{-1}$ on
$\chi(S)$, Lipschitz on a convex set containing $\chi(S)$ and $\hat\pi(N)$. The surrogate $\Psi = \hat G\circ\hat\pi$ satisfies

$$\sup_N\lVert\Psi - \Phi_\tau\rVert \le \sup_{\chi(S)}\lVert\hat G - \Gamma_\tau\circ\chi^{-1}\rVert + \mathrm{Lip}(\hat G)\sup_N\lVert\hat\pi - \chi\circ\pi\rVert + Ce^{-\gamma\tau},$$

and if $\hat G$ maps into the $\varepsilon_S$-neighbourhood of $S$, its normal gain is $\kappa_\Psi = 0$: every chain stays within
$\varepsilon_S$ of $S$ after one step (Proposition 2.4(c)).
*Proof.* $\Phi_\tau = \Gamma_\tau\circ\pi + R_\tau$ and
$\Psi - \Gamma_\tau\circ\pi = [\hat G\circ\hat\pi - \hat G\circ\chi\circ\pi] + [(\hat G - \Gamma_\tau\circ\chi^{-1})\circ\chi\circ\pi]$. $\square$
So a surrogate fitted on the slow set alone, in $k$ coordinates, is accurate on the whole neighbourhood (to the accuracy of
its coordinates) and normally stable in a rollout. With linear invariants the chart must include them as coordinates, or
the surrogate must conserve them (Proposition 2.4, invariants).

**Check.** `t2_slow_surrogate` (above): the slow-coordinate surrogate S, fitted on visited states only, rolls out with no
accumulation ($5.2\times10^{-4}$ after 39 steps, equal to its one-step error), while the far more accurate full-coordinate
surrogates diverge. Its chart $\hat\pi = z_1$ is the linear proxy; Robertson's visited states share one leaf, so the invariant
needs no coordinate here.

## 9. Synthesis

| | T1 | T2 | T3 | T4 | T5 | T6 | T7 | Thm 8 |
|---|---|---|---|---|---|---|---|---|
| U1 cost by class; $\Phi_{2\Delta t}$ harder | valid, uninformative on the Brusselator; Prop. 1.5 gives the direction for certificates | | | | | | | |
| U2 rollout growth; SlowTable's damped errors | | normal gain decides (Prop. 2.4); survived error under its hypothesis | | audit on chain states | | fast = damped | | slow-coordinate surrogates do not accumulate |
| U3 Robertson's rank 2 | | | | | | **explains** at $\tau = 1$ (staircase, two-sided with Prop. 6.5) | | |
| U4 vertex escalation | | | | | | | consistent in direction | |
| U5 ART-quad vertex failure | | | | | $L^2$ projection's vertex error (3 of 4 cases) | | same mechanism | |
| U6 vector outputs | | | | | | not explained (non-stiff) | per-output escalation | shared chart for all outputs (stiff case) |
| U7 no certificate | | | | restates | | | | |

## 10. Candidate foundations for a scheme, ranked (revised after review round 2)

1. **Slow-coordinate surrogates with normal stability (Theorem 8 with Proposition 2.4 and Proposition 2.2).** Fit
   $\hat G$ on the slow set in $k$ coordinates (plus the invariants), predict the next state, and audit on the surrogate's
   own chain states (Proposition 4.1). Both sides were measured on Robertson: the slow-coordinate surrogate does not
   accumulate, full-coordinate ones diverge, tube-sampled ones stay bounded. It is the only item whose hypothesis and
   conclusion were both measured on a stiff problem and held.
2. **The effective dimension (Corollary 6.1, Proposition 6.5, Proposition 6.7, Theorem 6.6).** Two-sided; the frozen count
   holds on HIRES, POLLU and the Brusselator, the path-averaged one on Robertson; it tells item 1 what $k$ is and which
   coordinates need the tight tolerance.
3. **A priori analyticity (T1, Lemma 1.2, Propositions 1.3, 1.5, 4.5)**, as an open problem: valid wherever tried,
   informative on small boxes only, vacuous on the Brusselator's box.
4. **The light cone (T3)**, only for lattices with $\ell\Delta\tau = O(1)$ as $d$ grows.

Lean (later): Lemmas 1.1, 1.2, 3.1, Propositions 1.3, 1.5, 2.4, 4.1, 4.2, 4.4, 5.1, 5.2, 6.5, Lemma 7.2 and Theorem 8 are
self-contained; the Chebyshev bounds, holomorphic dependence, Fenichel's theorem, Borsuk–Ulam, Levinson's theorem and the
compound-matrix log norm would enter as cited axioms.

## 11. Open items

- Theorem 8 on the benchmark's own schemes and problems: a learned chart $\hat\pi$ (from the Jacobian's dominant singular
  subspace, Proposition 6.7) on HIRES or POLLU, with the invariants as coordinates, against ChebTT's rollouts.
- T1: the tube in log coordinates on a benchmark box; whether the true nearest singularity of $\Phi_\tau$ approaches $B$
  monotonically in $\tau$ for dissipative polynomial fields (the open question behind U1).
- T2: the sign autocorrelation of each scheme's defects along trajectories (P2.2).
- T6: Proposition 6.7(b) with its finite-$\tau$ constants; a weight $W$ that makes Theorem 6.6 useful on non-normal Jacobians.
- T7: the direct test on `lotka_volterra` at $d = 20$.

## References (classical results used)

- G. Dahlquist (1958); G. Söderlind, BIT 46 (2006); W. Walter, *Differential and Integral Inequalities*, Springer 1970.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (Thms 4.2, 8.1–8.3, 15.2); Proc. AMS 145 (2017).
- Yu. Ilyashenko, S. Yakovenko, *Lectures on Analytic Differential Equations*, AMS 2008 (Thm 1.1).
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*; P. E. Kloeden, J. Lorenz, SIAM J. Numer. Anal.
  23 (1986); A. M. Stuart, A. R. Humphries, *Dynamical Systems and Numerical Analysis*, CUP 1996.
- N. Fenichel, Indiana Univ. Math. J. 21 (1971), J. Differential Equations 31 (1979); M. Hirsch, C. Pugh, M. Shub, LNM 583 (1977);
  C. Kuehn, *Multiple Time Scale Dynamics*, Springer 2015.
- J. S. Muldowney, Rocky Mountain J. Math. 20 (1990); P. Constantin, C. Foias, Comm. Pure Appl. Math. 38 (1985); A. Douady,
  J. Oesterlé, C. R. Acad. Sci. 290 (1980); R. A. Smith, Proc. Roy. Soc. Edinburgh 104A (1986).
- W. A. Coppel, *Dichotomies in Stability Theory*, LNM 629 (1978); M. S. P. Eastham, *The Asymptotic Solution of Linear
  Differential Systems*, OUP 1989.
- R. DeVore, R. Howard, C. Micchelli, *Optimal nonlinear approximation*, Manuscripta Math. 63 (1989).
- W. Hoeffding (1963); S. S. Wilks, Ann. Math. Stat. 12 (1941); A. G. Sukharev (1979); E. Novak, LNM 1349 (1988).
- A. Cohen, M. A. Davenport, D. Leviatan, FoCM 13 (2013); A. Cohen, G. Migliorati, SMAI J. Comput. Math. 3 (2017).
- R. E. Caflisch, W. Morokoff, A. Owen, J. Comput. Finance 1 (1997).
- L. Chen, P. Sun, J. Xu, Math. Comp. 76 (2007); M. J. Davis, R. T. Skodje, J. Chem. Phys. 111 (1999); L. A. Segel, M. Slemrod,
  SIAM Review 31 (1989); S. B. Pope, Combust. Theory Model. 1 (1997); U. Maas, S. B. Pope, Combust. Flame 88 (1992); S. H.
  Lam, D. A. Goussis, Int. J. Chem. Kinet. 26 (1994); M. Valorani, S. Paolucci, J. Comput. Phys. 228 (2009); N. Brisebarre,
  M. Joldeş, ISSAC 2010.
- A. Forets, A. Pouly, arXiv:1711.02552; J.-P. Liu et al., PNAS 118 (2021), arXiv:2011.03185.
