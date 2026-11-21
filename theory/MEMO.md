# A theory of surrogates for stiff ODE step maps: candidate foundations

This memo collects candidate theorems on which a surrogate scheme for the step maps of stiff ODEs could be built,
each with its assumptions, its proof (or the classical result it rests on), what it predicts, and a numerical check
of that prediction. It is written for the general autonomous system $x' = F(x)$; chemistry networks are one
instance. The theorems were chosen to explain what the benchmark and the convergence studies measured
(`docs/SURROGATE_RESEARCH.md`, `docs/SCHEMES_VS_ART.md`, both companions' `convergence.json`), not the other way
round, and each is kept to the complexity those measurements need.

- **Status legend.** *Proved*: a complete proof is given here. *Proved, citing*: the proof is complete given a
  classical result that is cited (marked **[C]**). *Sketch*: the argument is given but a step is not written out.
  *Conjecture*: stated, not proved. Every theorem lists its assumptions as (A1), (A2), ...; nothing else is used.
- **Checks.** `theory/checks/t*.py` (NumPy and SciPy only, each under two minutes) write
  `theory/checks/data/t*.json`; every number quoted below comes from those files or from the repository's
  records named beside it. A check can refute a prediction; it cannot prove a theorem.
- **Prior art.** `theory/LITERATURE.md` (IDs `TH-*`) records, per theorem, the closest published work and what is
  new. Classical results used here are listed under "References" at the end.

## 0. What known theory already predicts in our data, and what it does not

Two measured quantities already match classical predictions, which supports the approach of predicting the
experiments from assumptions on $f$:

| Quantity (source) | Prediction | Measured |
|---|---|---|
| ART's evaluations against $1/\alpha_{\mathrm{tol}}$ on Rosenbrock, $d = 2, 3, 4, 5$ (adaptive `convergence.json`, `rates.art`) | piecewise-affine approximation of a function with a full-rank Hessian: $N \sim \mathrm{tol}^{-d/2}$, slopes $1, 1.5, 2, 2.5$ | $1.00, 1.52, 2.10, 2.60$ |
| ART on a ridge $g(q\cdot x)$, $d = 2, 3, 5$ (global `convergence.json`, `ridge_rates`) | optimal anisotropic pieces for a rank-$r$ Hessian: $N \sim \mathrm{tol}^{-r/2}$ with $r = 1$ (Chen, Sun, Xu 2007) | $0.54, 0.55, 0.82$ (the $d = 5$ value deviates) |
| ChebTT-M's degree per decade on the ridge (global `convergence.json`, `ridge_degree`) | $1/\log_{10}\rho$ for the ridge profile's Bernstein parameter $\rho$ | predicted $12.0, 14.6, 19.0$; measured $11.9, 14.6, 19.1$ |

What no theory in the repository explains yet (the targets of sections T1–T7):

- **U1.** Step maps of chaotic, neutral and dissipative systems differ in cost by orders of magnitude, and a step map
  over $2\Delta t$ was *harder* to fit than two steps of $\Delta t$ (4-D Brusselator: 11k against 3.8k solves,
  `SURROGATE_RESEARCH.md` §1.20).
- **U2.** A smooth surrogate's rollout error grows linearly with the number of steps (ChebTT on Robertson: 36 times its
  one-step error after 40 steps, §1.12), while a table whose one-step error is 4 times the tolerance rolls out
  better, because only 0.1 of its error survives the next exact step (SlowTable, §1.20).
- **U3.** The Robertson step map in log coordinates has Jacobian singular values about $1.7, 1.0, 10^{-10}$ for every
  step $\Delta t \ge 0.1$ (§0.4 item 5): its effective input dimension is 2, not 3.
- **U4.** ChebTT's cost on `lotka_volterra` at $d = 20$ varies 26-fold between seeds (3,221 to 85,206 evaluations),
  driven by checks at the box's vertices; without them all seeds cost about 10k at rank 1 (`SCHEMES_VS_ART.md`).
- **U5.** ART with one quadratic leaf wins near-quadratic problems at loose tolerances but fails at the box's vertices
  (2.7 tol), and one refit of the same quadratic fails three times in six (`SCHEMES_VS_ART.md`).
- **U6.** Vector outputs: per-output tensor trains cost $m$ times one output, while ART's shared leaves do not.
- **U7.** No acceptance test in the schemes is a certificate: one false accept in 66 (validation 0.04, test 4.86).

## 1. Setting and tools

$x' = F(x)$, $F: \Omega \subset \mathbb{R}^d \to \mathbb{R}^d$; $x(t; x_0)$ the solution; the step map
$\Phi_\tau(x_0) = x(\tau; x_0)$; a box $B = \prod_j [a_j, b_j]$ with half-widths $h_j = (b_j - a_j)/2$. A
surrogate $\Psi \approx \Phi_\tau$; its defect $\delta = \Psi - \Phi_\tau$.

**Logarithmic norms [C: Dahlquist 1958; Söderlind 2006].** For a norm $\lVert\cdot\rVert$ on $\mathbb{C}^d$ and its
induced matrix norm, $\mu(A) = \lim_{h \to 0^+} (\lVert I + hA \rVert - 1)/h$. Used properties:

- (L1) $\mu(A + B) \le \mu(A) + \mu(B)$, $\mu(cA) = c\,\mu(A)$ for $c \ge 0$, $|\mu(A) - \mu(B)| \le \lVert A - B\rVert$.
- (L2) For the weighted max-norm $\lVert v \rVert_\beta = \max_j |v_j|/\beta_j$ ($\beta_j > 0$):
  $\mu_\beta(A) = \max_i \big(\operatorname{Re} a_{ii} + \sum_{j \ne i} |a_{ij}|\,\beta_j/\beta_i\big)$.
- (L3) If $w$ is $C^1$ and $w' = A(t) w$, then $D^+\lVert w(t)\rVert \le \mu(A(t))\,\lVert w(t)\rVert$ (upper right
  Dini derivative). Hence $\lVert w(t)\rVert \le \lVert w(0)\rVert \exp\int_0^t \mu(A)$.
- (L4) For continuous $A(s)$: $\mu\big(\int_0^1 A(s)\,ds\big) \le \int_0^1 \mu(A(s))\,ds$ (from L1 on Riemann sums,
  and the continuity of $\mu$).

**Bernstein ellipses and Chebyshev series [C: Trefethen, ATAP, Thms 8.1, 8.2 and 4.2].** $E_\rho([a,b])$ is the
open ellipse with foci $a, b$ and semi-axes $h(\rho \pm \rho^{-1})/2$. Every point of $E_\rho$ lies within distance
$h(\rho - \rho^{-1})/2$ of $[a, b]$ (shown in the proof of Theorem 1). $E_\rho = \prod_j E_{\rho_j}([a_j, b_j])$ is
a polyellipse. If $f$ is analytic in $E_\rho$ and $|f| \le M$ there, its tensor Chebyshev coefficients satisfy
$|c_k| \le 2^{|\operatorname{supp} k|} M \rho^{-k}$ (with $\rho^{-k} = \prod_j \rho_j^{-k_j}$; one variable at a
time). The tensor interpolant $I_n f$ on Chebyshev–Lobatto points maps $T_k$ to a product of $T$'s of sup-norm 1 and
reproduces every $T_k$ with $k \le n$ (aliasing), so $\lVert f - I_n f\rVert_{\infty, B} \le 2\sum_{k \not\le n} |c_k|$.

---

## T1. Analyticity radii of the step map from the vector field, robust to stiffness

**Question.** Chebyshev and tensor-train rates depend on the Bernstein parameter $\rho$ of the function, which
black-box schemes estimate from samples (and can be fooled). For a step map, $F$ is known: can $\rho$ and $M$ be
bounded *a priori* from $F$, with constants that do not blow up for stiff systems?

The classical complex Gronwall argument bounds the growth of complex perturbations by the Lipschitz constant $L$ of
$F$, which gives radii like $e^{-L\tau}$: for a stiff system ($L \sim 10^4$–$10^8$) the bound is useless. The
logarithmic norm replaces $L$ by a one-sided constant that is negative for dissipative systems, however stiff.

**Assumptions.** (A1) The real solutions $x(t; x_0)$ exist for $x_0 \in B$, $t \in [0, \tau]$; $X$ is the set they
visit. (A2) For a weight vector $\beta > 0$, $F$ extends holomorphically to an open neighbourhood of the tube
$T_\beta = \{x + v : x \in X,\ v \in \mathbb{C}^d,\ \lVert v\rVert_\beta \le 1\}$.

Define $m(t) = \sup\{\mu_\beta(J_F(x(t; x_0) + v)) : x_0 \in B,\ \lVert v\rVert_\beta \le 1\}$ (continuous in $t$, as
a supremum of a continuous function over a compact set) and $\Lambda = \max_{t \in [0, \tau]} \int_0^t m(s)\,ds$
($\Lambda \ge 0$, from $t = 0$).

**Lemma 1.1 (tube lemma; proved).** Under (A1)–(A2), for every $x_0 \in B$ and $w_0 \in \mathbb{C}^d$ with
$\lVert w_0\rVert_\beta < e^{-\Lambda}$, the complex solution $z(t)$ from $x_0 + w_0$ exists on $[0, \tau]$ and
$\lVert z(t) - x(t; x_0)\rVert_\beta \le \lVert w_0\rVert_\beta \exp \int_0^t m < 1$.

*Proof.* Let $w = z - x$. As long as $\lVert w\rVert_\beta \le 1$, the segment $x + s w$ ($s \in [0,1]$) lies in
$T_\beta$, and $w' = F(x + w) - F(x) = A(t) w$ with $A(t) = \int_0^1 J_F(x + s w)\,ds$ (the mean-value formula
holds for holomorphic maps on convex sets). By (L4), $\mu_\beta(A(t)) \le m(t)$, and by (L3)
$\lVert w(t)\rVert_\beta \le \lVert w_0\rVert_\beta e^{\int_0^t m} \le \lVert w_0\rVert_\beta e^{\Lambda} =: q < 1$.
Let $t^*$ be the supremum of the $t \le \tau$ up to which $z$ exists and $\lVert w\rVert_\beta < 1$. On $[0, t^*)$
the solution stays in the compact set $\{x + v: \lVert v\rVert_\beta \le q\} \subset T_\beta$, where $F$ is bounded,
so it extends past $t^*$ with $\lVert w\rVert_\beta \le q < 1$; hence $t^* = \tau$. $\square$

**Theorem 1 (proved, citing ATAP and holomorphic dependence on initial data [C: Ilyashenko–Yakovenko, Thm 1.1]).**
Under (A1)–(A2), fix $\theta \in (0, 1)$ and define $\rho_j > 1$ by $h_j(\rho_j - \rho_j^{-1})/2 = \theta\beta_j e^{-\Lambda}$.
Every component $\Phi_{\tau,i}$ extends holomorphically to $E_\rho$, with
$|\Phi_{\tau,i}| \le M_i := \max_{x_0 \in B} |x_i(\tau; x_0)| + \theta\beta_i$ there. Consequently
$|c_k(\Phi_{\tau,i})| \le 2^{|\operatorname{supp} k|} M_i \rho^{-k}$ and

$$\lVert \Phi_{\tau,i} - I_n \Phi_{\tau,i}\rVert_{\infty, B} \;\le\; 4 M_i \sum_j \frac{\rho_j^{-n_j}}{\rho_j - 1}\prod_{l \ne j}\frac{\rho_l + 1}{\rho_l - 1}.$$

*Proof.* A point $\zeta$ of $E_{\rho}([a, b])$ is $c + h\cos\phi$ with $\phi = u + i\nu$, $|\nu| < \log\rho$. Write
$\zeta = p + w$ with $p$ the nearest point of $[a, b]$. If $|\operatorname{Re}\zeta - c| \le h$, then
$|w| = |\operatorname{Im}\zeta| < h(\rho - \rho^{-1})/2$. Otherwise, with semi-axes $A = h(\rho + \rho^{-1})/2$ and
$b' = h(\rho - \rho^{-1})/2$ and $\zeta = c + A\cos u + i b'\sin u$ on the boundary, $|w|^2 = (A|\cos u| - h)^2 + b'^2 \sin^2 u \le b'^2$
because $(A - b')|\cos u| = h\rho^{-1}|\cos u| \le h$; the interior follows by monotonicity in $\rho$. So every point of
$E_\rho$ is $x_0 + w_0$ with $x_0 \in B$ and $\lVert w_0\rVert_\beta \le \theta e^{-\Lambda} < e^{-\Lambda}$. Lemma 1.1
gives a solution from it on $[0, \tau]$ inside the tube; the solution of a holomorphic ODE depends holomorphically on
its initial value where it exists [C], so $\Phi_\tau$ is holomorphic on the open set
$\{x_0 + w_0 : \lVert w_0\rVert_\beta < e^{-\Lambda}\} \supset E_\rho$ and agrees with the real map on $B$. The bound
$|z_i(\tau)| \le |x_i(\tau)| + \beta_i\lVert w(\tau)\rVert_\beta \le |x_i(\tau)| + \beta_i\theta e^{\int_0^\tau m - \Lambda}$
and $\int_0^\tau m \le \Lambda$ give $M_i$. The coefficient bound is §1's. For the interpolation error, with
$w(0) = 1$, $w(k) = 2\rho^{-k}$: $\sum_{k \in \mathbb{N}^d} \prod_j w_j(k_j) = \prod_j S_j$, $S_j = (\rho_j + 1)/(\rho_j - 1)$,
and the sum over $k \le n$ is $\prod_j s_j$ with $S_j - s_j = 2\rho_j^{-n_j}/(\rho_j - 1)$; then
$\prod S - \prod s \le \sum_j (S_j - s_j)\prod_{l \ne j} S_l$ and the aliasing bound give the display. $\square$

**Corollary 1.2 (three regimes; proved).** Fix a shape $\hat\beta$ and scale the tube, $\beta = r\hat\beta$. Suppose
that for $0 < r \le r_{\max}$ the tube's log-norm bound obeys $m_r(t) \le \mu_0 + c\,r^p$ ($c > 0$, $p > 0$; $p = 1$ when
$J_F$ is Lipschitz, $p = 2$ when the first-order term vanishes on the worst trajectory, as for $J = \mu_0 - 3z^2$ at
$z = 0$). Then the admissible radius $R(\tau) = \sup_{r \le r_{\max}} r e^{-\Lambda_r}$ satisfies

| Regime | $\mu_0$ | Lower bound on $R(\tau)$ | Degree for a tolerance, as $\tau$ grows |
|---|---|---|---|
| dissipative | $< 0$ | $\min(r_{\max}, (\lvert\mu_0\rvert/c)^{1/p})$, for every $\tau$ | bounded in $\tau$ |
| neutral | $= 0$ | $(p\,c\,e\,\tau)^{-1/p}$ (once below $r_{\max}$) | $\propto \tau^{1/p}$ |
| expanding | $> 0$ | $e^{-\mu_0\tau}(p\,c\,e\,\tau)^{-1/p}$ | $\propto \tau^{1/p} e^{\mu_0\tau}$ |

*Proof.* $\Lambda_r \le \max(0, \mu_0 + c r^p)\,\tau$. Dissipative: for $c r^p \le |\mu_0|$, $\Lambda_r = 0$. Neutral and
expanding: maximise $r e^{-c r^p\tau}$, at $r = (p c\tau)^{-1/p}$. The degree column uses $\rho - 1 \approx 2\theta R/h$ for
small $R$ and $n \approx \log(C/\mathrm{tol})/\log\rho$. $\square$

**Predictions.** (P1.1) Theorem 1's $\rho$ is a valid lower bound on the true analyticity parameter, and
the true $\rho$ follows the regime's law. (P1.2) In a dissipative system the cost of a step-map surrogate does not grow
with the step; in a neutral one the needed degree grows like $\tau^{1/p}$, so one long step costs more than the
same span in short steps whenever the cost grows faster than linearly in the degree (tensor or TT costs do); in a
chaotic one it grows exponentially. This is U1.

**Checks.**

- `t1_scalar` (closed forms; $B = [-1, 1]$). $x' = -x - x^3$: $m_r = -1 + 3r^2$ (brute force agrees to 4 digits), so
  $R \to 1/\sqrt3$ and $\rho_{\mathrm{thm}} \to 1.731$ for $\tau \ge 1$, against the exact $\rho \to 1 + \sqrt2 = 2.414$
  (the branch points $\pm i/\sqrt{1 - e^{-2\tau}}$). $x' = x - x^3$: $\rho_{\mathrm{thm}} \le \rho_{\mathrm{exact}}$ at
  every $\tau$, and $\log(\rho - 1)$ falls with slope $-1.14$ (theorem) against $-1.03$ (exact) for $\tau \ge 1$: the
  exponent $\mu_0 = 1$ is right, the bound loses only the factor $\tau^{-1/2}$ of $p = 2$. Every resolved Chebyshev
  coefficient of the exact maps lies below $2M\rho_{\mathrm{thm}}^{-k}$ (largest ratio 0.56).
- `t1_lorenz96` ($d = 8$; coefficients along a fibre in $x_0$). $F = 0$ on $[-0.05, 0.05]^8$: the explicit bound
  $\mu_\infty(J(z)) \le -1 + 4\max_j|z_j|$ with $\lVert x(t)\rVert_2 \le \sqrt d\,a\,e^{-t}$ (the quadratic term conserves
  $\sum x_i^2$) gives $\Lambda = 0$ and $\rho_{\mathrm{thm}} = 4.55$ for every $\tau$; every resolved coefficient lies
  below the bound (ratio at most 0.44), and the measured $\rho_{\mathrm{eff}}$ stays between 100 and 280 from
  $\tau = 0.5$ to 4, levelling off rather than falling towards 1 (P1.2, dissipative). $F = 1$ (the benchmark's box $[0, 2]^8$, a neutral travelling wave; $J$ is linear in $x$, so
  $p = 1$): $\rho_{\mathrm{eff}} - 1 \propto \tau^{-1.04}$ for $\tau \in [0.25, 4]$, against the predicted $\tau^{-1}$.
  $F = 8$ (chaotic, largest Lyapunov exponent 1.68): $\rho_{\mathrm{eff}}$ falls from 251 at $\tau = 1/16$ to 4.4 at
  $\tau = 1$ and to about 1 at $\tau = 2$ (where 129 points no longer resolve the fibre).

**Limits.** The theorem needs $F$ holomorphic on a complex tube (true for mass-action kinetics, polynomial and
rational right-hand sides; false for $|x|$, $\max$, tabulated rates), and a computable bound on the log norm over that
tube. The bound on $R$ is a lower bound: the true radius can be much larger (factor 1.4 in $\rho$ above), so the
predicted *degrees* are upper bounds. The tensor-product error bound is exponential in $d$; T3 and T6 are what make it
usable in high dimension.

---

## T2. How a surrogate's defect accumulates in a rollout

**Question.** Chaining $\Psi$ for $n$ steps, how large is $y_n - x_n$? The classical bound,
$\varepsilon A_n$ with $A_n = \sum_j \lVert G_{n,j}\rVert$, is what the benchmark reports
(`step_maps.py`, `rollout_over_bound`). U2 shows it is far from the whole story.

**Proposition 2.1 (global bound; proved, classical).** Let $C$ be convex and forward invariant under the flow up to
time $\tau$, with $\mu(J_F(x)) \le m$ for $x \in C$. Then $\lVert \Phi_\tau(x) - \Phi_\tau(y)\rVert \le e^{m\tau}\lVert x - y\rVert$
on $C$. If also $\Psi(C) \subset C$ and $\sup_C \lVert\delta\rVert \le \varepsilon$, then
$\lVert y_n - x_n\rVert \le \varepsilon \sum_{j < n} e^{jm\tau}$: at most $\varepsilon/(1 - e^{m\tau})$ for every $n$ when
$m < 0$.
*Proof.* $x(t) - y(t)$ obeys $w' = A w$ with $A$ the mean of $J_F$ over the segment, which lies in $C$; (L3)–(L4)
give the contraction; then $e_{k+1} = \Phi(y_k) - \Phi(x_k) + \delta(y_k)$ and induction. $\square$

**Proposition 2.2 (first order, and the fixed-point bias; proved).** Let $\Phi$ be $C^2$ on a convex set $U$
containing both chains, $\lVert D^2\Phi\rVert \le K_2$, and $G_{n,j} = D\Phi(x_{n-1})\cdots D\Phi(x_j)$ ($G_{n,n} = I$). Then

$$e_n = \sum_{k=0}^{n-1} G_{n,k+1}\,\delta(y_k) + R_n, \qquad \lVert R_n\rVert \le \tfrac{K_2}{2}\sum_{k<n}\lVert G_{n,k+1}\rVert\,\lVert e_k\rVert^2 .$$

If $x^*$ is a fixed point with $P = D\Phi(x^*)$, $I - P$ invertible, and $\lVert\delta\rVert_{C^1} \le \varepsilon$ is small,
then $\Psi$ has a unique fixed point $y^*$ near $x^*$ and

$$y^* - x^* = (I - P)^{-1}\delta(x^*) + O(\varepsilon^2) = V\,\mathrm{diag}\Big(\frac{1}{1 - \lambda_i}\Big)V^{-1}\delta(x^*) + O(\varepsilon^2) \quad (P = V\Lambda V^{-1}).$$

*Proof.* $e_{k+1} = D\Phi(x_k)e_k + r_k + \delta(y_k)$ with $\lVert r_k\rVert \le K_2\lVert e_k\rVert^2/2$ (Taylor on the
segment); unroll. For the fixed point, apply the implicit function theorem to $y = \Phi(y) + \delta(y)$ at
$(x^*, \delta = 0)$, then expand $\delta(y^*) = \delta(x^*) + O(\varepsilon\lVert y^* - x^*\rVert)$. $\square$

So a defect component along mode $i$ is amplified by $1/|1 - \lambda_i|$: about 1 for a fast mode
($\lambda_i \approx 0$), about $1/(|\lambda_{\mathrm{slow}}|\tau)$ for a slow one, without bound for a neutral one. The
classical bound charges every direction the worst amplification.

**Proposition 2.3 (neutral dynamics: only the defect's average accumulates; proved).** Let $\Phi(x) = Ux$ with $U$
orthogonal, and $x_k = U^k x_0$. Up to a remainder at most $\mathrm{Lip}(\delta)\,\varepsilon\,n^2/2$ (with $\varepsilon = \sup\lVert\delta\rVert$),
$e_n = U^{n-1}S_n$ with $S_n = \sum_{k<n} U^{-k}\delta(U^k x_0)$, the defect pulled back to the initial frame. If the
Cesàro mean $\bar\delta = \lim S_n/n$ exists, $\lVert e_n\rVert = n\lVert\bar\delta\rVert + o(n)$. For a planar rotation
by an angle $\theta \notin \pi\mathbb{Q}$ and continuous $\delta$, $\bar\delta = \frac{1}{2\pi}\int_0^{2\pi} R_{-\phi}\,\delta(R_\phi x_0)\,d\phi$
(Weyl equidistribution); if $\bar\delta = 0$ and $\delta$ is a trigonometric polynomial in the phase,
$\sup_n\lVert S_n\rVert < \infty$ (each harmonic sums geometrically).
*Proof.* $\Phi$ is linear, so $e_{k+1} = Ue_k + \delta(y_k)$ exactly; replace $\delta(y_k)$ by $\delta(x_k)$ at the
stated cost and unroll. $\square$

**Corollary 2.4 (a tolerance per mode).** Near a stable fixed point, $\lVert y^* - x^*\rVert_2 \le T$ holds if
$|(V^{-1}\delta)_i| \le T\,|1 - \lambda_i| / (\sqrt d\,\lVert V\rVert_2)$ for every $i$. A rollout-aware surrogate needs
tolerance about $T(1 - \lambda_{\mathrm{slow}})$ only along slow modes, and about $T$ along fast ones.

**Predictions.** (P2.1) Fast-direction defects barely accumulate: the "survived" error of `step_maps.py`
($\lVert\Phi(\Psi(z)) - \Phi(\Phi(z))\rVert \approx \lVert D\Phi\,\delta\rVert$) predicts rollout quality better than the
one-step error (U2, SlowTable). (P2.2) In neutral directions a smooth defect, which varies little along a short
trajectory, has $\bar\delta \approx \delta$ and grows linearly (U2, ChebTT's 36x and $\rho = 0.6$–1); a defect whose
sign changes along the orbit does not.

**Checks** (`t2_rollout`). Davis–Skodje ($\gamma = 10$, $\tau = 0.1$; exact flow; eigenvalues 0.905 and 0.368): a
constant slow-mode defect gives $y_n - x_n \to 10.508\,\varepsilon$ against the predicted $10.508\,\varepsilon$; a
fast-mode one $1.582\,\varepsilon$ against $1.582\,\varepsilon$; the classical bound is $A_n\varepsilon = 14.1\,\varepsilon$
for both. Rotation ($\tau = 0.1$, 4,000 steps): a generic linear defect grows as $n\lVert\bar\delta\rVert$ (1,709.6
against 1,708.3); defects with $\bar\delta \approx 0$ (a trace-free symmetric field, a phase harmonic, a constant)
stay at most 20 at every step, where the classical bound is 4,000.

**Limits.** Proposition 2.3 is for linear isometric dynamics; for a nonlinear neutral flow (a conserved quantity, a
limit cycle) the same statement holds to first order in the co-moving frame (averaging), which is not proved here.

---

## T3. A light cone for locally coupled systems, and tensor-train ranks independent of $d$

**Question.** Why can a tensor train represent a step map of a large network at all? For locally coupled systems
(reaction–diffusion on a grid, lattices, Lorenz-96, sparse reaction graphs), information travels a bounded graph
distance per unit time.

**Assumptions.** (A3) Along every trajectory from $B$ up to time $\tau$: $\operatorname{Re} J_{ii} \le \delta$ and
$|J_{ij}| \le L_{ij}$ for $i \ne j$, with $L \ge 0$ supported on the coupling graph's edges. $D(i, j)$ is the directed
graph distance from $i$ to $j$.

**Lemma 3.1 (proved).** For every $x_0 \in B$, entrywise
$|\partial\Phi_\tau/\partial x_0| \le e^{\delta\tau}\exp(\tau L)$, and
$[\exp(\tau L)]_{ij} \le \sum_{k \ge D(i,j)} (\tau\ell\Delta)^k/k! \le \frac{(\tau\ell\Delta)^{D(i,j)}}{D(i,j)!}e^{\tau\ell\Delta}$
with $\ell = \max L_{ij}$ and $\Delta$ the largest out-degree.
*Proof.* $M = \partial x(t)/\partial x_0$ solves $M' = (D + N)M$, $D = \operatorname{diag} J$. With $U(t, s)$ the diagonal
propagator of $D$ ($|U_{ii}| \le e^{\delta(t - s)}$), $M(t) = U(t, 0) + \int_0^t U(t, s)N(s)M(s)\,ds$. Every Picard iterate
of this equation is bounded entrywise by the corresponding partial sum of the series of
$Q(t) = e^{\delta t}\exp(tL)$, which solves $Q = e^{\delta t}I + \int_0^t e^{\delta(t - s)}LQ\,ds$; the iterates converge to
$M$. $(L^k)_{ij} = 0$ for $k < D(i, j)$, and $(L^k)_{ij} \le \ell^k\Delta^k$. $\square$

**Corollary 3.2 (window surrogate; proved).** Let $W_r(i) = \{j : D(i, j) \le r\}$ and $x^W$ equal $x$ on $W_r(i)$ and
the box centre $c$ elsewhere. Then
$\sup_B |\Phi_{\tau,i}(x) - \Phi_{\tau,i}(x^W)| \le e^{\delta\tau}\sum_{j \notin W_r(i)} h_j [\exp(\tau L)]_{ij}$,
which does not depend on $d$ beyond the number of sites at each distance.
*Proof.* The segment from $x^W$ to $x$ lies in $B$; integrate $\nabla\Phi_{\tau,i}$ along it and use Lemma 3.1. $\square$

**Corollary 3.3 (TT ranks; proved).** Let $p$ be the tensor Chebyshev interpolant of degree $n$ in each window
variable of $x \mapsto \Phi_{\tau,i}(x^W)$. For any ordering of the variables and any bond splitting them into a left
and a right set, the unfolding rank of $p$ is at most $(n + 1)^{\min(|W \cap \mathrm{left}|,\,|W \cap \mathrm{right}|)}$,
since $p = \sum_{k_{\mathrm{left}}} T_{k_{\mathrm{left}}}(x_{\mathrm{left}})\,q_{k_{\mathrm{left}}}(x_{\mathrm{right}})$. So
$\Phi_{\tau,i}$ has a TT approximation whose ranks do not depend on $d$, with error the window error plus the
interpolation error (Theorem 1 applied to the window function). For a ring in its natural order
$\min(\cdot) \le \lfloor |W_r|/2 \rfloor$.

**Predictions.** (P3.1) Jacobian entries of the step map decay super-exponentially in graph distance. (P3.2) At a
fixed step, ChebTT's ranks and cost per output on a locally coupled family (the method-of-lines ladder: Brusselator,
Allen–Cahn, Fisher–KPP at $n = 2..80$; Lorenz-96) stop growing once $d$ exceeds the light cone's width, and grow with
$\tau$. (P3.3) For densely coupled systems (`lotka_volterra`: every species sees every other) there is no light cone;
T7 is the relevant structure there.

**Checks** (`t3_lightcone`; Lorenz-96, $F = 1$, $d = 40$, the measured trajectory sups $\ell_1 = 2.25$, $\ell_2 = 1.89$,
so this checks the lemma with measured constants, not proved ones). At $\tau = 0.25$ the measured largest
$|\partial\Phi_i/\partial x_j|$ falls from 0.81 at $D = 0$ to $1.9\times10^{-13}$ at $D = 11$, always below the bound
($1.1$ to $6.6\times10^{-10}$). The window error at radius $r = 4$ is $2.6\times10^{-5}$, $3.2\times10^{-5}$ and
$5.2\times10^{-5}$ for $d = 20, 40, 80$: independent of $d$, as predicted. At $\tau = 1$ the bound is valid but loose
(the majorant $\exp(\tau L)$ ignores cancellations), and the window needs $r \approx 7$ for $10^{-4}$.

**Limits.** The rank bound $(n+1)^{\lfloor|W|/2\rfloor}$ is $d$-independent but large; it explains why ranks can stop
growing, not their measured values.

**P3.2 against the ladder's records** (`results/benchmark/runs/step_maps/cpu-1rank`, `chebtt_box`, total
evaluations for all $d$ outputs): Allen–Cahn at 0.01 costs 325, 530, 1,619, 17,103 for $d = 2, 3, 5, 10$; Fisher–KPP
409, 2,324, 14,358, 107,002; Lorenz-96 2,515, 7,230, 339,627 for $d = 4, 5, 10$. Per output the cost still grows 12- to
54-fold up to $d = 10$, and no $d = 20$ run finished. This does not test P3.2: at these steps the light cone is as wide
as the system (Lorenz-96 at $\tau = 0.25$ needs a window of radius 2–4, i.e. 7–13 sites, for $10^{-2}$–$10^{-5}$), so
$d$-independence could only appear beyond $d \approx 10$–20. It does show that a box-wide ChebTT in the original
coordinates pays a steep price in $d$ on these stiff families, while SlowTable, trained on the visited states only,
grows mildly (Fisher–KPP at 0.01: 1,138, 2,560, 4,115 for $d = 2, 3, 5$), which is what T6 (low effective dimension of the
visited set) and T7 (the visited measure, not the box) predict. P3.2 stays open.

---

## T4. What can be certified from samples

**Question.** Which acceptance tests are certificates (U7), and what does a sampled certificate cost?

**Proposition 4.1 (adaptive audits; proved).** A training procedure produces models $s_1, s_2, \ldots$; at round $r$
it draws $N_r$ points i.i.d. from $\mu$, independent of everything before, and accepts $s_r$ iff all of them are within
the tolerance; it stops at the first acceptance. Then
$\Pr[\text{the accepted model violates the tolerance on a set of } \mu\text{-measure} > \eta] \le \sum_r (1 - \eta)^{N_r}$;
with $N_r \ge \ln(2^r/\alpha)/(-\ln(1 - \eta))$ this is at most $\alpha$.
*Proof.* Given the history before round $r$, $s_r$ is fixed; if its violation set has measure $> \eta$, all $N_r$
fresh points miss it with probability $\le (1 - \eta)^{N_r}$. Union bound over rounds. $\square$

The schemes' current checks are not of this kind: Sobol sets are deterministic, free checks are earlier
evaluations, and look-ahead fibres and maxvol pivots are chosen from the model. They are good heuristics and no
certificate (U7).

**Proposition 4.2 (from a measure bound to a sup bound; proved).** With $\mu$ uniform on $B$ and
$d_B(x, y) = \max_j |x_j - y_j|/(b_j - a_j)$: if $e \ge 0$ is $L$-Lipschitz in $d_B$ and $\mu\{e > t\} < \eta \le 1$,
then $\sup_B e \le t + L\eta^{1/d}$.
*Proof.* If $e(x) > t + Lr$ with $r = \eta^{1/d}$, then $e > t$ on $\{y \in B : d_B(x, y) \le r\}$, whose measure is at
least $r^d = \eta$ (each axis keeps at least $r(b_j - a_j)$). $\square$

**Corollary 4.3.** $N$ fresh uniform points all within $t$ give, with probability $\ge 1 - \alpha$,
$\sup_B e \le t + L(\ln(1/\alpha)/N)^{1/d}$ (since $1 - \alpha^{1/N} \le \ln(1/\alpha)/N$).

**Proposition 4.4 (sharpness; proved).** For any $N$ points in $B$ there is an $L$-Lipschitz $e \ge 0$ that vanishes at
all of them with $\sup_B e \ge L/(2m)$, $m = \lceil (N + 1)^{1/d}\rceil$.
*Proof.* Of the $m^d > N$ congruent sub-boxes of side $1/m$ (in $d_B$), one contains no point in its interior; put
$e(y) = L\max(0, 1/(2m) - d_B(y, \text{its centre}))$ there. $\square$

So a sampled sup-norm certificate for Lipschitz functions costs $N \sim (L/\Delta)^d$ samples for a margin $\Delta$: the
curse of dimension (Novak–Woźniakowski) in its simplest form. Above $d \approx 4$ a sup-norm certificate must come
from structure (an a priori bound, Theorem 1), not from sampling; a measure-sense certificate (Proposition 4.1) costs
$\ln(1/\alpha)/\eta$ samples in any dimension.

---

## T5. When is one quadratic leaf reliable

**Proposition 5.1 (Lebesgue inequality for discrete least squares; proved).** For points $x_1..x_N$ and a space $V$
on which they are unisolvent, with $\mathrm{LS}f(x) = \sum_i f(x_i)\ell_i(x)$ and
$\Lambda_N = \sup_x \sum_i |\ell_i(x)|$: $\lVert f - \mathrm{LS}f\rVert_\infty \le (1 + \Lambda_N)\inf_{q \in V}\lVert f - q\rVert_\infty$.
*Proof.* $\mathrm{LS}$ reproduces $V$, so $f - \mathrm{LS}f = (f - q) - \mathrm{LS}(f - q)$ and
$\lVert\mathrm{LS}g\rVert_\infty \le \Lambda_N\max_i|g(x_i)|$. $\square$

**Check** (`t5_least_squares`; quadratics on $[-1, 1]^d$, ART's $N = 20p + 1$ uniform points, 5 seeds). The empirical
Gram matrix is well conditioned ($\lambda_{\min} \approx 0.57$–$0.67$), although $N$ is far below the
Cohen–Davenport–Leviatan sufficient size ($N/\ln N \ge (1 + r)K/\kappa$ with $r = 1$, $\kappa = (1 - \ln 2)/2$ and $K = 1 + 8d + 9d(d-1)/2$; the constants are to be verified against the paper, `LITERATURE.md`) ($N = 421$ against 17,006 at $d = 5$; 4,621 against 308,470 at $d = 20$).
The Lebesgue constant, estimated from below on random points, is 4.4, 6.0, 9.7, 18.9, 34.4 at the vertices for
$d = 2, 3, 5, 10, 20$, about twice its interior value.

**Predictions.** (P5.1) A single quadratic leaf holds a tolerance only if the best quadratic error is about
$1/(1 + \Lambda_N)$ of it: $1/20$ at $d = 10$, $1/35$ at $d = 20$. ART's one-leaf wins sit at 0.62–0.99 of the tolerance
on the test set, so they are within the $\Lambda_N$ margin by luck of the draw, and fail at the vertices, where
$\Lambda_N$ is largest (U5: 2.7 tol at vertices; three failed refits in six). (P5.2) Weighted least squares with
optimal sampling (Cohen–Migliorati) or a fresh-point audit (Proposition 4.1) is what would make such a leaf
reliable.

---

## T6. The effective dimension of a stiff step map

**Question.** Why does the Robertson step map have one negligible singular value (U3), why do a table's
fast-direction errors not matter (U2), and how should a surrogate exploit stiffness?

**Assumptions** (Fenichel's setting [C: Fenichel 1979; Jones 1995; Kuehn 2015, ch. 3]). (S1) Coordinates
$p = (x, y) \in \mathbb{R}^k \times \mathbb{R}^{d-k}$ with $x' = f(x, y, \varepsilon)$, $\varepsilon y' = g(x, y, \varepsilon)$, smooth, on a
compact domain $N$ forward invariant up to time $\tau$. (S2) The critical manifold $\{g = 0\} = \{y = h_0(x)\}$ is
uniformly attracting: the eigenvalues of $D_y g$ on it have real parts $\le -\gamma_0 < 0$. (S3) [C] For
$\varepsilon \le \varepsilon_0$ there are a locally invariant slow manifold $\{y = h_\varepsilon(x)\}$ and a smooth projection
along its stable fibres, $\pi(p) = (\xi(p), h_\varepsilon(\xi(p)))$ with base point $\xi(x, y) = x + \varepsilon H(x, y, \varepsilon)$, $H$
smooth and bounded with its derivatives, such that for $0 < \gamma < \gamma_0$:
$\lVert\Phi_t(p) - \Phi_t(\pi(p))\rVert + \lVert D[\Phi_t - \Phi_t\circ\pi](p)\rVert \le C e^{-\gamma t/\varepsilon}$. (The
derivative part of (S3) is what Fenichel's smooth fibres give; it is stated as an assumption to be checked against the
cited theorems.)

**Theorem 6 (proved, given (S3)).** $\Phi_\tau = \Gamma_\tau\circ\xi + R_\tau$ on $N$, with
$\Gamma_\tau(\zeta) = \Phi_\tau(\zeta, h_\varepsilon(\zeta))$ a function of $k$ variables and
$\lVert R_\tau\rVert + \lVert DR_\tau\rVert \le Ce^{-\gamma\tau/\varepsilon}$. Consequently:

1. (Corollary 6.1) $\sigma_{k+1}(D\Phi_\tau(p)) \le Ce^{-\gamma\tau/\varepsilon}$ at every $p$, since
   $D(\Gamma_\tau\circ\xi)$ has rank at most $k$ (Weyl's inequality for singular values).
2. (Corollary 6.2) If $\Gamma_\tau$ is Lipschitz on a convex set containing every $x + \varepsilon H(x, y)$, the best surrogate
   that is a function of $x$ alone has error at most
   $\varepsilon\,\mathrm{Lip}(\Gamma_\tau)\,\sup_x \operatorname{rad}_y H(x, \cdot) + Ce^{-\gamma\tau/\varepsilon}$ ($\operatorname{rad}$: the
   radius of the smallest ball containing $\{H(x, y)\}_y$), and no function of $x$ alone does better than
   $\tfrac12\sup_{x, y, y'}|\Gamma_\tau(\xi(x, y)) - \Gamma_\tau(\xi(x, y'))| - Ce^{-\gamma\tau/\varepsilon}$, which is of order
   $\varepsilon$ when $\Gamma_\tau$ and $H$ vary: a linear reduction (dropping the fast variables) is accurate to
   $O(\varepsilon)$, not exponentially.
3. (Corollary 6.3, tolerance split) Approximating $\Gamma_\tau$ to $\mathrm{tol}/2$ and $H$ to
   $\mathrm{tol}/(2\varepsilon\,\mathrm{Lip}\,\Gamma_\tau)$ approximates $\Phi_\tau$ to $\mathrm{tol}$ (plus the exponential
   term): the $d - k$ fast directions need a tolerance looser by $1/\varepsilon$, and none at all once
   $\varepsilon\,\mathrm{Lip}(\Gamma)\operatorname{osc}H \le \mathrm{tol}/2$.
4. (Corollary 6.4, shared coordinates) Every output $\Phi_{\tau,i}$ is $\Gamma_{\tau,i}\circ\xi$ with the same $\xi$: one
   coordinate map serves every species; only the $k$-variable maps differ.

*Proof.* (S3) at $t = \tau$ with $\Phi_\tau(\pi(p)) = \Gamma_\tau(\xi(p))$. For 2, take $g(x) = \Gamma_\tau(x + \varepsilon\bar H(x))$ with
$\bar H(x)$ the centre of the smallest ball containing $H(x, \cdot)$'s range; the lower bound holds because $g(x)$ is one
value for every $y$. For 3, $|\Gamma(\tilde\xi) - \Gamma(\xi)| \le \mathrm{Lip}(\Gamma)\,\varepsilon|\tilde H - H|$. $\square$

**Predictions.** (P6.1) $\sigma_{k+1}(D\Phi_\tau)$ decays like $e^{-\gamma\tau/\varepsilon}$ until the solver's tolerance
(U3's $10^{-10}$ is that floor). (P6.2) Ignoring the fast variables costs $O(\varepsilon)$. (P6.3) A step-map surrogate
needs full resolution only in $k$ coordinates: the hub expansion's form (dense in a few directions, low order in the
rest) and ChebTT-M's metric-aligned coordinates are the right structures for stiff maps, with fast-direction
resolution set by $\mathrm{tol}/\varepsilon$; and the defects in fast directions are also those the dynamics damp (T2),
which is why SlowTable's large one-step errors barely survive (U2).

**Checks** (`t6_slowfast`). Michaelis–Menten (Segel–Slemrod form, $k = 1$, box $s \in [0.2, 1]$, $c \in [0, 1]$):
$\sigma_2$ at $\varepsilon = 0.01$ falls from 0.28 to $3\times10^{-11}$ as $\tau/\varepsilon$ goes from 1 to 20, a rate
$1.21$ per unit $\tau/\varepsilon$ against the predicted $\gamma \approx \min(s + \kappa) = 1.2$; the best
function-of-$s$ surrogate's error is $3.46\times10^{-2}$ at $\varepsilon = 0.1$ down to $3.51\times10^{-4}$ at
$\varepsilon = 10^{-3}$, slope $0.999$ in $\varepsilon$ (P6.2). Robertson in log coordinates (not in standard form; the
conserved total is the second slow direction): $\sigma_3$ follows $e^{-|\lambda_{\mathrm{fast}}|\tau}$ closely
($2.7\times10^{-10}$ at $\tau|\lambda| = 22$ against $e^{-22} = 2.8\times10^{-10}$; 0.10 at $\tau|\lambda| = 2.2$
against 0.11) down to $10^{-17}$, and the two others settle at $1.64$–$1.73$ and $1.00$, reproducing U3 without its floor.

**Limits.** Real networks are not given in slow–fast form; $k$, $\xi$ and $\varepsilon$ must be found (the Jacobian's
singular spectrum gives $k$ and the tangent of $\xi$; a scheme must learn $H$). Several time scales give a hierarchy
($k$ depends on $\tau$; at $t_0 = 0.01$ Robertson's second singular value also decays, to $6.5\times10^{-3}$ at
$\tau = 1$). Assumption (S3)'s derivative estimate is cited, not proved here.

---

## T7. Sup-norm against measure: why vertex checks multiply the cost

**Question.** U4: ChebTT's checks at the vertices drive its rank up 26-fold between seeds. Is that a defect of the
scheme or a property of the target norm?

Many high-dimensional maps depend on aggregates ($s = \sum_j w_j x_j$: the competition term
$\sum_j A_{ij}x_j$ of `lotka_volterra`, a total concentration, an energy). Under the uniform measure an aggregate
concentrates; at the vertices it does not.

**Proposition 7.1 (proved).** Let $f(x) = g(s)$, $s = \frac1d\sum_j x_j$, $x$ uniform on $[-1,1]^d$. Hoeffding:
$\Pr(|s| \ge t) \le 2e^{-dt^2/2}$. So a polynomial $p$ with $\sup_{|s| \le t_\eta}|g - p| \le \mathrm{tol}$,
$t_\eta = \sqrt{2\ln(2/\eta)/d}$, violates the tolerance on a set of measure at most $\eta$, while sup-norm accuracy
needs $\sup_{|s| \le 1}|g - p| \le \mathrm{tol}$ (the vertices reach $s = \pm1$). For $g$ analytic in the Bernstein
ellipse $E_\rho([-1,1])$, the measure-sense degree is about $\log(1/\mathrm{tol})/\log\rho(t_\eta)$, with
$\rho(t) \to \infty$ as $t \to 0$, against $\log(1/\mathrm{tol})/\log\rho$ in the sup norm.

**Lemma 7.2 (proved).** A polynomial of degree $q$ in $s = \sum_j w_j x_j$ has TT ranks at most $q + 1$ in every
ordering: with $s = s_L + s_R$ across a bond,
$\sum_p c_p s^p = \sum_{a \le q} s_L^a\big(\sum_{p \ge a} c_p\binom{p}{a}s_R^{p-a}\big)$.

**Check** (`t7_concentration`; $g(s) = 1/(1 + 0.8s)$, $\rho = 2$, tol $= 10^{-3}$, $\eta = 10^{-3}$). The sup-norm degree is
13 (rank $\le 14$) for every $d$. The measure-sense degree from Hoeffding's radius is 13, 13, 13, 9, 5, 4 for
$d = 2, 5, 10, 20, 50, 100$, and from the measured $(1 - \eta)$ quantile of $|s|$ (sharp, not a bound) 12, 8, 6, 4, 3,
3. The measure-sense polynomial violated the tolerance on none of $10^5$ uniform points, and misses by 102, 1,030 and
1,490 times the tolerance at the vertices for $d = 20, 50, 100$.

**Predictions.** (P7.1) For aggregate-driven maps in $d \gtrsim 10$, a sup-norm target costs several times the rank
of a measure-sense one, and the extra cost is spent on a set of exponentially small measure. Maxvol pivots move to
large $|f|$, hence to vertices, so a cross with vertex checks pays the sup-norm price (U4: ranks 4–10 and 48k–85k
evaluations with them, rank 1 and 3.2k–10.4k without). (P7.2) The choice of norm ("Decision needed" in
`SCHEMES_VS_ART.md`) is a choice between these two costs; for trajectories that live on a manifold (chemistry) the
measure of the visited states is the relevant one.

**Limits.** Proposition 7.1 is for an exact function of one aggregate; real maps are approximately so (several
aggregates, weights $w_j$ of unequal size: the effective dimension in Hoeffding is $(\sum w_j)^2/\sum w_j^2$). The
connection to ChebTT's escalation is an explanation consistent with the records, not a proof that the cross behaves so.

---

## 8. Synthesis: which theorem explains which observation

| | T1 analyticity | T2 rollouts | T3 light cone | T4 certificates | T5 LS leaf | T6 slow–fast | T7 measure |
|---|---|---|---|---|---|---|---|
| U1 cost by dynamical class; $\Phi_{2\Delta t}$ harder | **explains** (regime laws) | | | | | | |
| U2 rollout growth; SlowTable's damped errors | | **explains** | | | | supports (fast = damped) | |
| U3 Robertson's rank 2 | | | | | | **explains** | |
| U4 ChebTT's vertex escalation | | | | | | | **explains** |
| U5 ART-quad wins and fragility | | | | supports | **explains** | | |
| U6 vector outputs | | | per-output windows | | | **explains** (shared $\xi$) | |
| U7 no certificate | | | | **explains** | | | |

## 9. Candidate foundations for a scheme, ranked

1. **Stiffness-adapted coordinates with a priori analyticity (T6 + T1 + T2).** Find the slow coordinates $\xi$ (the
   step map's dominant singular subspace gives $k$ and $D\xi$; learn $H$ at the loose tolerance
   $\mathrm{tol}/\varepsilon$), approximate the $k$-variable map $\Gamma_\tau$ with Chebyshev or TT degrees chosen from
   Theorem 1's $\rho$ (computed from $F$, not sampled), and allocate the tolerance per mode (Corollary 2.4). It
   explains U1, U2, U3 and U6 at once, applies to every stiff system with a spectral gap, and makes testable claims:
   cost depending on $k$, not $d$; rollouts bounded by $\varepsilon/(1 - \lambda_{\mathrm{slow}})$ in contracting
   systems; degrees bounded in the step for dissipative maps.
2. **Light-cone tensor trains (T3 + T1)** for spatially extended or locally coupled systems (method-of-lines PDEs,
   lattice chemistry): ranks independent of $d$; complements 1 where $k$ itself is large.
3. **Measure-sense targets with fresh-point audits (T7 + T4)** for high-dimensional black boxes without dynamics to
   exploit: costs that do not explode at vertices, and a certificate in any dimension.

Lean (later): Lemma 1.1, Lemma 3.1, Propositions 4.1, 4.2, 4.4, 5.1 and Lemma 7.2 are self-contained and finite
dimensional; the Chebyshev bounds, holomorphic dependence and Fenichel's theorem would enter as cited axioms.

## 10. Open items

- T1: a computable bound on $m(t)$ for a real network (mass action is polynomial, so it reduces to bounding the
  states on the tube); a TT (not tensor) version of the a priori error bound.
- T2: Proposition 2.3 for nonlinear neutral flows (averaging); the sign-coherence of each scheme's actual defects
  along benchmark trajectories.
- T3: P3.2 needs ladder runs beyond the light cone's width ($d \ge 20$ at the ladder's steps), with ranks recorded.
- T6: a constructive way to learn $\xi$ (and $H$) from samples, with its cost; several time scales.
- T7: aggregates with unequal weights, and several aggregates.

## References (classical results used)

- G. Dahlquist, *Stability and error bounds in the numerical integration of ordinary differential equations*, 1958; G.
  Söderlind, *The logarithmic norm. History and modern theory*, BIT 46 (2006).
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM 2013/2019 (Thms 4.2, 8.1, 8.2); multivariate
  versions: Trefethen, *Multivariate polynomial approximation in the hypercube*, Proc. AMS 145 (2017).
- Yu. Ilyashenko, S. Yakovenko, *Lectures on Analytic Differential Equations*, AMS 2008 (Thm 1.1: analytic dependence on
  initial conditions).
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*, Springer (global error of one-step
  methods, "Lady Windermere's fan").
- N. Fenichel, *Geometric singular perturbation theory for ordinary differential equations*, J. Differential Equations
  31 (1979); C. K. R. T. Jones, *Geometric singular perturbation theory*, LNM 1609 (1995); C. Kuehn, *Multiple Time
  Scale Dynamics*, Springer 2015.
- W. Hoeffding, *Probability inequalities for sums of bounded random variables*, JASA 58 (1963).
- A. Cohen, M. A. Davenport, D. Leviatan, *On the stability and accuracy of least squares approximations*, FoCM 13
  (2013); A. Cohen, G. Migliorati, *Optimal weighted least-squares methods*, SMAI J. Comput. Math. 3 (2017).
- E. Novak, H. Woźniakowski, *Approximation of infinitely differentiable multivariate functions is intractable*, J.
  Complexity 25 (2009).
- L. Chen, P. Sun, J. Xu, *Optimal anisotropic meshes for minimizing interpolation errors in $L^p$-norm*, Math. Comp. 76
  (2007).
- M. J. Davis, R. T. Skodje, *Geometric investigation of low-dimensional manifolds in systems approaching equilibrium*,
  J. Chem. Phys. 111 (1999) (the test system of `t2_rollout`); L. A. Segel, M. Slemrod, *The quasi-steady-state
  assumption*, SIAM Review 31 (1989) (the Michaelis–Menten scaling of `t6_slowfast`).
