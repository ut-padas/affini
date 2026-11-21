# Review of the theory memo, round 1 of 3

- **Reviewer:** Claude Fable 5.2 (independent review sub-agent; no subagents used; no existing file edited).
- **Date:** 2026-11-21. Branch `structure-experiments`, memo as of commit `0502542`.
- **Read in full:** `theory/MEMO.md`, `theory/LITERATURE.md`, `theory/README.md`, all of `theory/checks/*.py` and
  `theory/checks/data/*.json`. **Read for context:** `docs/SURROGATE_RESEARCH.md` §0.4, §1.12, §1.20;
  `docs/SCHEMES_VS_ART.md`; the `rates`, `ridge_rates`, `ridge_degree` entries of both `convergence.json`;
  `src/problems/ode/reaction_diffusion.py`, `step_maps.py` (the ladder's steps), `src/problems/lotka_volterra.py`; the
  ladder records under `results/benchmark/runs/step_maps/cpu-1rank` that the memo quotes.
- **Ran:** five tests of my own (`theory/reviews/scratch-round-1/`, NumPy and SciPy only, 1 s to 3 min 14 s each):
  four that could refute a prediction of the memo and one that tests a theorem I propose. I did not re-run the memo's seven checks: I compared every number the memo quotes with the JSON files and
  read each script against the claim it supports.
- **Literature:** three web searches (Carleman linearization bounds, ISAT, Wilks); everything else in C24–C27 is
  from memory and marked so.

**Verdict in one paragraph.** The proofs are mostly correct: I found no error in Lemma 1.1, Theorem 1, Lemma 3.1,
Corollaries 3.2–3.3, Propositions 4.1, 4.2, 4.4, 5.1 or Lemma 7.2, and the memo's numbers match its JSON files (three
small slips, C20). The weaknesses are elsewhere. (i) Several "predictions" are upper bounds read as laws, and the
checks that "confirm" them could not have failed (C1, C2, C6, C17). (ii) Three theorems are not applicable, as
stated, to the problems they are meant to explain: T3 to the stiff method-of-lines ladder (C9), T2's fixed-point
results and T1's dissipative regime to any network with a conservation law (C4, C7), T6 to `lotka_volterra_state`
(C16). (iii) T6, the top-ranked foundation, is its own assumption (S3) restated; its content is the corollaries, and
"cost depending on k, not d" does not follow from them (C15, C16). (iv) P5.1 is refuted by a direct test (C13). On
the positive side, my test B shows that Theorem 1 does what its title claims on Robertson: valid radii that the fast
rate 4,200 does not destroy. It also shows the limit: the bound decays with the step where the true radius does not.

---

## Critiques

Severity: **critical** (a headline claim is wrong or unsupported), **major** (a statement, check or assumption
needs to change), **minor** (wording, constant, citation).

### T1

**C1 (critical). The regime laws are upper bounds; they cannot "explain" U1, and the memo never evaluates them on
U1's system.**
*Location:* Corollary 1.2, P1.1 ("the true ρ follows the regime's law"), P1.2, §8 row U1 ("**explains**").
*Problem.* Theorem 1 gives a lower bound on the analyticity parameter, so Corollary 1.2's degree column is an upper
bound on the degree. An upper bound that grows with τ does not show that the cost grows with τ; U1 (the two-step
Brusselator map costs 2.9 times more) needs a lower bound on the cost, and the memo has none. Second, the regime in
the corollary is the sign of $\mu_0$, the log norm in a weighted max norm on a complex tube; it is not the dynamical
class. The two differ on every system I tried:

- Lorenz-96, $F = 1$, box $[0,2]^8$ (called "neutral" in the memo): $\mu_\infty(J)$ on the real box reaches 5
  (analytic sup; sampled 4.96 in `a_lorenz_regimes.py`), so the theorem puts it in the expanding regime with
  $R \lesssim e^{-5\tau}$, not $\tau^{-1}$.
- Robertson (test B): the comparison matrix of $J$ has spectral abscissa $0.022 > 0$ at $y(100)$, so
  $\mu_\beta(J) \ge 0.022$ for every weight vector (Collatz–Wielandt): "expanding" by the theorem, while the
  measured $\rho$ is flat in τ (4.0–4.8 for $\tau = 0.01$ to 10).
- The 4-D Brusselator of U1 is at $B = 1.9$, a stable focus ("the uniform state is stable … rings down slowly",
  `reaction_diffusion.py`), in log coordinates, trained on visited states. Dynamically it is dissipative, for which
  the corollary's table says "bounded in τ", the opposite of U1. The memo computes neither $\mu_0$ nor Λ for it.

*Fix.* (a) State P1.1–P1.2 as bounds. (b) Add a converse so the statement becomes two-sided; Cauchy's estimate
gives one in two lines (proposed theorem N2 below). (c) Evaluate Λ on the Brusselator at $\tau = 0.25$ and 0.5 and
compare with measured fibre coefficients before claiming U1. (d) In §8 change "explains" to "consistent with; upper
bound only".

**C2 (major). The Lorenz-96 checks do not discriminate between regimes; the quoted $-1.04$ is the most favourable
of three seeds.**
*Location:* check `t1_lorenz96`, the sentence "$\rho_{\mathrm{eff}} - 1 \propto \tau^{-1.04}$ … against the
predicted $\tau^{-1}$".
*Evidence* (`a_lorenz_regimes.py`: the memo's measurement, repeated for four forcings and three fibre seeds; slope of
$\log(\rho_{\mathrm{eff}} - 1)$ against $\log\tau$ on $\tau \in [0.25, 4]$):

| $F$ (dynamics at $d = 8$) | seed 1 | seed 2 | seed 3 | $\rho_{\mathrm{eff}}$ at $\tau = 4, 8, 16$ (seed 1) |
|---|---|---|---|---|
| 0.5 (stable fixed point, rate $-0.5$) | $-0.88$ | $-0.83$ | $-0.82$ | 4.2, 3.6, 4.0 |
| 0.8 (stable, rate $-0.2$) | $-0.96$ | $-0.80$ | $-0.84$ | 3.5, 2.9, 2.7 |
| 1.0 (critical: $\operatorname{Re}\lambda = -1 + F$ at $k = \pi/2$) | $-1.04$ | $-0.80$ | $-0.84$ | 2.9, 2.3, 1.8 |
| 1.2 (unstable, a wave) | $-1.11$ | $-0.82$ | $-0.88$ | 2.5, 1.6, 1.2 |

The slope on $[0.25, 4]$ is $-0.8$ to $-1.1$ for every forcing, the dissipative ones included. It is the
short-step behaviour common to all of them ($\Phi_\tau \approx x + \tau F(x)$ with quadratic $F$: the nonlinearity,
and so the distance to the singularities, scales with $1/\tau$), not a signature of neutrality. The regimes separate
only for $\tau \ge 4$ (plateau at 4–5.5 for $F = 0.5$; continued decline for $F \ge 1$), which the memo's fit does
not use. At $F = 1.2$ the three seeds give slopes $-1.3$, $-0.4$, $-0.5$ between $\tau = 4$ and 16: no clean law.
Further: (i) the $F = 0$ case fits $\rho_{\mathrm{eff}}$ to 2–4 coefficients (`k_used` 2–4 in the JSON; three rows
are `inf`), on a box of half-width 0.05 where the map is nearly linear, so "levels off between 100 and 280" is not
evidence of anything, and the bound there (4.55) is off by a factor 3.7 in degree; (ii) the chaotic case reports no
comparison with $\lambda_{\max} = 1.68$: between $\tau = 0.5$ and 1, $\rho - 1$ falls by 5.3, a rate of 3.3, twice
the Lyapunov exponent, and the JSON's fitted slope ($-2.46$) uses two unresolved rows ($\rho = 1.106$, 1.003).
*Fix.* Report all seeds; fit the law only where the regimes differ ($\tau \ge 4$); drop the $F = 0$ fit or enlarge
the box until at least ten coefficients are resolved; state the chaotic rate against $\lambda_{\max}$.

**C3 (major). Theorem 1 has never been evaluated on a stiff system, although "robust to stiffness" is its title.
When evaluated, it holds, with two caveats the memo should state.**
*Location:* T1's checks (two scalar ODEs and non-stiff Lorenz-96), "Limits", §10 first item.
*Evidence* (`b_robertson_t1.py`: Robertson in the original coordinates, box centred at $y(100)$ with half-widths
$0.5\,y$; $m(t)$ over the trajectories from the centre and the 8 corners and over the distinguished boundary of the
polydisc; β optimised; $\rho_{\mathrm{eff}}$ from 33-point fibres; fast rate $|\lambda| = 4{,}197$):

| τ | $\tau\lvert\lambda_{\mathrm{fast}}\rvert$ | Λ | $\rho_{\mathrm{thm}}$ per input $(y_1, y_2, y_3)$ | $\rho_{\mathrm{eff}}$ measured | degree for 3 digits, thm / measured (worst input) |
|---|---|---|---|---|---|
| 0.01 | 42 | 0.044 | 1.44, 5.67, 1.44 | 39, ∞, 4.0 | 19 / 5.0 |
| 0.1 | 420 | 0.090 | 1.42, 5.42, 1.42 | 41, ∞, 4.3 | 20 / 4.7 |
| 1 | 4,197 | 0.43 | 1.35, 3.75, 1.26 | 43, ∞, 4.8 | 30 / 4.4 |
| 10 | 41,970 | 1.75 | 1.11, 1.35, 1.04 | 16, ∞, 4.4 | 202 / 4.7 |

The bound is valid in all 24 comparisons (both box sizes) and Λ stays of order one while $\tau|\lambda|$ reaches
42,000: the central claim is right. The caveats: (a) it needs the right weights. With $\beta \propto h$ the same
computation gives $\Lambda = 10^2$ to $10^5$ (my first run); the usable β is near the Perron vector of the comparison
matrix, with $\beta_2/h_2 \approx 7\,\beta_1/h_1$. The theorem should say how β is chosen (minimise Λ; start from
the Perron vector). (b) Λ grows with τ (0.17 per unit time at $\tau = 10$) because $\mu_\beta \ge 0.022$ for every
β, so the bound decays while the true radius does not: the degree is overestimated 4-fold at $\tau \le 0.1$ and
43-fold at $\tau = 10$. Note also that the measured $\rho$ in $y_2$ is infinite (the map does not depend on the fast
species to $10^{-8}$): the theorem does not see T6's structure. My $m(t)$ uses 9 trajectories, so
$\rho_{\mathrm{thm}}$ here is slightly optimistic.
*Fix.* Add this (or a better) stiff evaluation to the memo; state the choice of β; replace "do not shrink with τ or
with stiffness" by "do not shrink with stiffness; shrink with τ unless $\mu_0 < 0$".

**C4 (major). The dissipative regime is empty for every system with a linear invariant, which includes every
closed mass-action network.**
*Location:* Corollary 1.2 (row "dissipative"), Proposition 2.1 ($m < 0$), "Limits" of T1.
*Problem.* If $c^{\mathsf T}F \equiv 0$ then $c^{\mathsf T}J = 0$, so $J$ has the eigenvalue 0 and
$\mu(J) \ge \max\operatorname{Re}\lambda(J) = 0$ in every norm. So $\mu_0 \ge 0$: Robertson, HIRES and POLLU (1, 1
and 6 invariants, §1.20) are never in the regime "radius independent of τ", and Proposition 2.1's bounded
accumulation never applies to them on a full-dimensional set. Test B shows the consequence (Λ grows with τ).
*Fix.* State T1 and Proposition 2.1 on the invariant affine leaf: eliminate the invariants (a reduced system of
dimension $d - q$), or use a seminorm that vanishes on the directions transverse to the leaf together with the
matching log seminorm. Then check whether the reduced Robertson system has $\mu_0 < 0$ for some β (I did not).

**C5 (minor). Constants and wording in T1.**
(a) Proof of Corollary 1.2: "$\rho - 1 \approx 2\theta R/h$" should be $\theta R/h$
($h(\rho - \rho^{-1})/2 = \theta R$ and $\rho - \rho^{-1} \approx 2(\rho - 1)$); the JSON agrees (expanding,
$\tau = 4$: $R = 0.00227$, $\rho_{\mathrm{thm}} = 1.0023$). (b) The degree column ignores the prefactor of Theorem
1's display, $\prod_l(\rho_l + 1)/(\rho_l - 1) \approx (2/(\rho - 1))^{d}$, which adds $d\log(1/(\rho - 1))$ to
$\log(C/\mathrm{tol})$; in the neutral regime the degree is $\tau^{1/p}\,(\log(1/\mathrm{tol}) + (d/p)\log\tau)$, not
$\tau^{1/p}$. (c) The symbol Λ is the log-norm integral in T1, the eigenvalue matrix in Proposition 2.2 and (with a
subscript) the Lebesgue constant in T5; δ is the defect in §1–T2 and the diagonal bound in (A3). (d) "factor 1.4 in
ρ" is a factor 1.6 in degree ($\log 2.414/\log 1.731$); the degree is what the reader needs. (e) The theorem allows
only weighted max norms. It could allow any norm whose unit ball contains the polydisc $\{|v_j| \le \beta_j\}$,
which matters when the fast block is not diagonally dominant.

### T2

**C6 (major). The checks verify linear algebra on synthetic defects; no prediction about a real surrogate is
tested, although the records allow it.**
*Location:* check `t2_rollout`, P2.1, P2.2.
*Problem.* Davis–Skodje with a constant defect converges to the fixed point of $\Psi$, which is
$(I - P)^{-1}\delta + O(\varepsilon^2)$ by the implicit function theorem: the check cannot fail. The rotation check
verifies an identity for a linear map. P2.1 ("the survived error predicts rollout quality better than the one-step
error") and P2.2 (sign coherence) are statements about ChebTT's and SlowTable's defects and are not tested.
*Evidence that P2.1 can be tested from the committed table* (§1.20, Robertson, five rows, in units of the
tolerance):

| row | one-step | survived | rollout | rollout / one-step | rollout / survived |
|---|---|---|---|---|---|
| ChebTT box, 0.01 | 0.25 | 0.17 | 5.05 | 20 | 30 |
| ChebTT rollout-verified, 0.01 | 0.04 | 0.02 | 0.82 | 21 | 41 |
| ChebTT box, 0.001 | 0.18 | 0.18 | 4.34 | 24 | 24 |
| ChebTT rollout-verified, 0.001 | 0.03 | 0.01 | 0.26 | 8.7 | 26 |
| SlowTable, 0.01 | 4.03 | 0.10 | 1.78 | 0.44 | 18 |

The ratio to the survived error spans 18–41 (2.3-fold), the ratio to the one-step error 0.44–24 (55-fold): P2.1
holds on these five rows. This belongs in the memo in place of (or beside) the synthetic check. P2.2 needs the sign
autocorrelation of the defects along a trajectory (`SCHEMES_VS_ART.md`, open experiment 6).

**C7 (major). Proposition 2.2's fixed-point part and Corollary 2.4 do not apply to the case U2 is about.**
*Location:* Proposition 2.2 (second display), Corollary 2.4, §8 row U2.
*Problem.* (a) Robertson conserves $y_1 + y_2 + y_3$, so $D\Phi$ has the eigenvalue 1 (left eigenvector
$(1, 1, 1)$) at every state, its fixed points are not isolated, and $I - P$ is singular: the hypothesis fails.
(b) U2's rollouts are 40 steps of a transient along the slow manifold, not the neighbourhood of a fixed point; what
applies is the first display, $e_n = \sum_k G_{n,k+1}\delta(y_k) + R_n$. (c) The memo says U2 "shows [the classical
bound] is far from the whole story", but for the smooth surrogate the bound is attained: $\rho = $ rollout /
(one-step $\times A_n$) is 0.67–0.98 on four ChebTT rows of §1.20. The classical bound fails to describe only
SlowTable (0.28), and there the survived error does (C6).
*Fix.* State the first display as the result for transients; restrict the fixed-point formula to hyperbolic fixed
points of the system reduced by its invariants; say that ChebTT attains the classical bound.

**C8 (minor). Smaller points in T2.**
(a) Proposition 2.3: "$\lVert e_n\rVert = n\lVert\bar\delta\rVert + o(n)$" is false as $n \to \infty$. The remainder
$\mathrm{Lip}(\delta)\,\varepsilon\,n^2/2$ is not $o(n)$; relative to the main term it is $\mathrm{Lip}(\delta)\,n/2$.
For $\delta(y) = \varepsilon By$ the chain is $(U + \varepsilon B)^n$, which grows geometrically or stays bounded.
State it for $n \ll 1/\mathrm{Lip}(\delta)$. (b) The JSON has a third Davis–Skodje case the memo omits: a unit defect
in the slow *variable*, $(1, 0)$, is amplified 13.79, more than the slow *mode*'s 10.51 and within 2% of the
classical 14.1, because the eigenvectors $(1,1)$ and $(0,1)$ are not orthogonal. So "the classical bound charges
every direction the worst amplification" understates it: a coordinate-direction defect nearly attains it. For
non-normal chemistry Jacobians the split must be along the fibres (rows of $V^{-1}$), not along the species.
(c) The $O(\varepsilon^2)$ in Proposition 2.2 hides $\lVert(I - P)^{-1}\rVert^2 K_2$; the expansion needs
$\varepsilon\,\lVert(I - P)^{-1}\rVert^2 K_2 \ll 1$, a real restriction for slow modes
($\lVert(I - P)^{-1}\rVert \approx 1/(|\lambda_{\mathrm{slow}}|\tau)$). (d) Corollary 2.4 holds to first order only,
and assumes $P$ diagonalisable; say so. (e) "stay at most 20": the JSON's maximum is 20.008.

### T3

**C9 (critical). There is no $d$-independent light cone on the method-of-lines ladder, by construction; P3.2 is
false for three of the four families it names.**
*Location:* (A3), P3.2, §9 item 2 ("method-of-lines PDEs"), the paragraph "P3.2 against the ladder's records"
("$d$-independence could only appear beyond $d \approx 10$–20").
*Problem.* The ladder refines a fixed domain $[0, 1]$: `_laplacian` multiplies by $n^2$, so (A3)'s constants are
$\ell = D n^2$ and $\delta \approx -2Dn^2$. For diffusion, $e^{\delta\tau}\exp(\tau L)$ is the discrete heat kernel,
of width $\sqrt{2D\tau}$ in space, i.e. $n\sqrt{2D\tau}$ sites: $0.14\,n$ for Fisher–KPP ($\tau = 0.01$),
$0.1\,n$ for Allen–Cahn, $0.5\,n$ for the Brusselator. The window is a fixed fraction of the system at every $n$.
*Evidence* (`d_mol_window.py`: Fisher–KPP as in the repository, $\tau = 0.01$, the middle output, box
$[0.05, 0.95]^n$, 48 samples; smallest window radius in sites):

| $n$ | radius for $10^{-3}$ (Lemma 3.1's bound) | radius for $10^{-5}$ (bound) | half the system |
|---|---|---|---|
| 10 | 4 (4) | 5 (5) | 5 |
| 20 | 9 (9) | 10 (10) | 10 |
| 40 | 16 (17) | 20 (20) | 20 |
| 80 | 31 (34) | 40 (40) | 40 |

The radius doubles with $n$; at $10^{-5}$ the window is the whole system at every size. (The lemma's bound is
nearly sharp here, which is a point in its favour.) So for stiff method-of-lines families the stiffness enters T3
through $\ell\tau = D\tau n^2$: T3 is the one theorem of the memo that is *not* robust to stiffness.
*Fix.* State (A3) with the dimensionless $\ell\tau$ and say that $d$-independence needs $\ell\tau$ bounded as $d$
grows (lattices at fixed spacing: Lorenz-96, a growing domain), which excludes refinement of a fixed domain. For
the refinement ladder the relevant structure is the opposite one: the high modes are damped, so the number of
singular values of $D\Phi_\tau$ above a tolerance does not grow with $n$ (proposed theorem N7; test E measures 7
above $10^{-2}$ and 10 above $10^{-4}$ for $n = 40$ and 80). That is a correct and tested replacement for P3.2 on
those families.

**C10 (minor). The rank bound is that of any function of $|W|$ variables.**
*Location:* Corollary 3.3, "Limits".
The memo says so honestly. Two additions: the window for Lorenz-96 has $3r + 1$ sites (one forward and two
backward per step), 13 at $r = 4$ and 22 at $r = 7$, so the bound is $(n + 1)^{6}$ to $(n + 1)^{11}$; and "the
window error is independent of $d$" is supported only loosely by the check ($2.6$, $3.2$, $5.2 \times 10^{-5}$ for
$d = 20, 40, 80$ at $r = 4$, a maximum over 64 samples that rises with $d$ there and falls at $r = 5$). A sharper
statement is available from proposed theorem N1: analyticity radii that grow with graph distance give coefficient
decay per variable, hence ranks from the decay and not from a hard window.

### T4

**C11 (no defect found).** Propositions 4.1, 4.2, 4.4 and Corollary 4.3 are correct as stated (I checked the corner
case of 4.2, the case $\ln(1/\alpha)/N > 1$ of 4.3 and the counting in 4.4).

**C12 (minor). Citations and one overstatement.**
(a) The lower bound for Lipschitz functions is classical optimal recovery (Sukharev 1979; Bakhvalov; Novak's 1988
monograph), not Novak–Woźniakowski 2009, which is about $C^\infty$ functions. (b) Proposition 4.1 with one round is
Wilks' non-parametric tolerance limit (1941): $N \ge \ln\alpha/\ln(1 - \eta)$, the "95/95 needs 59 samples" rule.
Cite it. (c) "Above $d \approx 4$ a sup-norm certificate must come from structure (… Theorem 1)": Theorem 1's own
display carries $\prod_l(\rho_l + 1)/(\rho_l - 1)$, exponential in $d$, and it bounds $\Phi$, not the error of a
given surrogate. The step from an a priori $(\rho, M)$ to a certificate for $\Psi$ is missing; it is short (proposed
theorem N4).

### T5

**C13 (major). P5.1 reads the Lebesgue inequality backwards, and a direct test refutes it.**
*Location:* P5.1 ("holds a tolerance only if the best quadratic error is about $1/(1 + \Lambda_N)$ of it: $1/20$ at
$d = 10$"; "within the $\Lambda_N$ margin by luck of the draw").
*Problem.* Proposition 5.1 is a sufficient condition: the error is at most $(1 + \Lambda_N)E$. "Only if" would need
the bound to be attained, which requires a residual whose signs at the sample points match those of $\ell_i(x)$.
*Evidence* (`c_ls_leaf.py`: $d = 10$, $N = 1{,}321$, 6 seeds; $E_T$ is the best sup error of any quadratic on 4,000
uniform points plus 1,000 vertices, by linear programming):

| function | LS worst, interior / $E_T$ | LS worst, vertices / $E_T$ | $L^2$ projection at vertices / $E_T$ |
|---|---|---|---|
| $1/(1 + 0.8\,\bar x)$ | 1.52–1.58 | 4.04–4.15 | 4.35 |
| $\exp(0.5\sum x_i/\sqrt d)$ | 0.92–1.00 | 5.48–5.70 | 5.24 |
| $\exp(-0.09\lvert x\rvert^2)$ | 1.13–1.31 | 3.94–4.14 | 8.66 |
| cubic | 0.49–0.63 | 1.96–2.07 | 2.05 |

Against $1 + \Lambda_N = 11.6$ (interior) and 19.9 (vertices): the measured factor is 0.5–1.6 in the interior and
2–5.7 at the vertices, and it barely varies with the seed. The vertex error is that of the $L^2$-best quadratic
(columns 3 and 4 agree for three of four functions): least squares minimises a mean square under the uniform
measure, and the vertices have measure zero. This is T7's phenomenon, not instability. The exact statement is
proposed theorem N5: the error at $x$ is $r(x) - \sum_i\ell_i(x)r(x_i)$ with $r$ the residual of the $L^2$
projection, and the second term has standard deviation $\lVert r\rVert_{L^2}\sqrt{K(x)/N}$, about
$0.6\lVert r\rVert_{L^2}$ at a vertex for $N = 20p$ at every $d$ (0.07–0.32 of $E_T$ here).
*Fix.* Replace P5.1 by: a least-squares quadratic leaf has worst error about $E$ in the bulk and
$|r(\text{vertex})|$ at the vertices; U5's 2.7 tol is the $L^2$ projection's vertex error; "three failed refits in
six" at margins 0.83–1.17 is the $\pm 15\%$ scatter of a score that sits at the tolerance, not a draw from a
20-fold worst case.

**C14 (minor).** (a) $\Lambda_N$ is "estimated from below"; an upper bound costs nothing:
$\Lambda_N(x) \le \sqrt{N}\,\lVert\ell(x)\rVert_2 = \sqrt{\varphi(x)^{\mathsf T}G^{-1}\varphi(x)} \le \sqrt{K(x)/\lambda_{\min}(G)}$,
which gives 29.2 at $d = 10$ and 56.7 at $d = 20$ against the estimates 18.9 and 34.4: two-sided to a factor 1.6,
and growth linear in $d$ at the vertices. (b) "about twice its interior value" holds for $d \ge 10$ only (ratios
1.07, 1.11, 1.36, 1.78, 1.92). (c) Two values of CDL's constant circulate, $(1 - \ln 2)/(2 + 2r)$ and
$(3\ln\frac32 - 1)/(2 + 2r)$ (the lower and the upper Chernoff tail); the memo uses the second, which
`LITERATURE.md` marks as read in the source. I did not re-verify it; it changes the sufficient sizes by a factor
1.4 and no conclusion.

### T6

**C15 (major). Theorem 6 is assumption (S3) at $t = \tau$; what is unproved is exactly what limits its use.**
*Location:* (S3), "Theorem 6 (proved, given (S3))", "Limits" last sentence.
*Problem.* $\Phi_\tau = \Gamma_\tau\circ\xi + R_\tau$ with the stated bound on $R_\tau$ and $DR_\tau$ is (S3)
rewritten. The label is honest, but the theorem ranked first in §9 then has no proof content beyond Weyl's
inequality. (S3) differs from the cited theorems in three ways. (a) Fenichel's fibres exist in a neighbourhood of
the slow manifold whose size does not depend on ε, not on all of a forward-invariant $N$; to cover a box one needs
that every layer problem $y' = g(x, y, 0)$ is globally attracted to $h_0(x)$ within $N$ (automatic for one fast
variable on an invariant interval under (S2), not for $d - k \ge 2$, where the layer problem can have other
attractors). (b) The derivative estimate needs the foliation to be $C^1$ jointly in the base point with a rate
uniform in ε; this holds for $C^r$, $r \ge 2$, vector fields with any $\gamma < \gamma_0$, but it should be cited
with the theorem number, as the memo itself says. (c) The form $\xi = x + \varepsilon H$ is specific to the standard
form. In a network that is not in standard form the fibre projection moves the slow coordinates by an amount set by
the fast species' abundance, not by the time-scale ratio: for Robertson, relaxing $y_2$ shifts $y_1$ and $y_3$ by
$O(y_2) \approx 10^{-5}$, while the time-scale ratio is $10^{-4}$ to $10^{-6}$ depending on the state. The "ε" of
Corollaries 6.2–6.3 (fibre tilt) and the "ε" of the exponential (rate ratio) are two different numbers outside the
standard form.
*Fix.* State T6 without coordinates: a compact normally attracting invariant manifold $S$ with a $C^1$ invariant
stable foliation of a forward-invariant neighbourhood $N$, contraction $Ce^{-\Gamma t}$ along fibres with Γ larger
than the tangential rates. Then $\Phi_\tau = \Gamma_\tau\circ\pi + R_\tau$, $\lVert R\rVert_{C^1} \le Ce^{-\Gamma\tau}$,
and Corollary 6.2 reads "the best surrogate ignoring a set of coordinates errs by half the oscillation of
$\Gamma_\tau\circ\pi$ along them". Give the standard form as the example in which the oscillation is $O(\varepsilon)$.

**C16 (major). "Cost depending on $k$, not $d$" does not follow, and U6 is attributed to T6 on a problem T6 does not
cover.**
*Location:* Corollary 6.3, §9 item 1 ("testable claims: cost depending on $k$, not $d$"), §8 row U6.
*Problem.* (a) $H$ (or ξ) is a function of all $d$ variables with values in $\mathbb{R}^k$. Corollary 6.3 loosens
its tolerance by $1/(\varepsilon\,\mathrm{Lip}\,\Gamma_\tau)$; it does not reduce its dimension. The cost is a
$k$-variate function at the tight tolerance plus a $d$-variate one at a loose tolerance, and the second is free only
when $\varepsilon\,\mathrm{Lip}(\Gamma)\operatorname{osc}H \le \mathrm{tol}/2$. (b) $\Gamma_\tau$ involves
$h_\varepsilon$, which is $C^r$ but in general neither analytic nor unique (slow manifolds differ by
$O(e^{-c/\varepsilon})$). Theorem 1 therefore cannot be applied to $\Gamma_\tau$ as §9 item 1 proposes ("degrees
chosen from Theorem 1's ρ"); it applies to $\Phi_\tau$ restricted to a set, or to the flow of the reduced equation
on the critical manifold $h_0$ (analytic when $g$ is) with an $O(\varepsilon)$ error. (c) U6 was measured on
`lotka_volterra_state` (competitive Lotka–Volterra, rates in $[0.5, 1.5]$, $t_f = 2$): not stiff, no slow manifold.
"Shared ξ" cannot explain that ART's shared leaves beat per-output trains there; `SCHEMES_VS_ART.md`'s own account
(each point serves every output; the vertex escalation repeated per output) does.
*Fix.* Reword the claim as "tight tolerance in $k$ coordinates, loose in $d$"; replace "T1 on $\Gamma_\tau$" by one
of the two options in (b); mark U6 as unexplained by T6 for the non-stiff case.

**C17 (major). Both T6 checks of the singular-value law are Liouville's formula, not tests of the fibre
structure.**
*Location:* check `t6_slowfast` (i) and the Robertson part.
*Problem.* In both examples exactly one direction is fast ($d - k = 1$). Then
$\sigma_d = |\det D\Phi_\tau|/\prod_{i<d}\sigma_i$ with $\det D\Phi_\tau = \exp\int_0^\tau\operatorname{tr}J$, and
$\operatorname{tr}J \approx \lambda_{\mathrm{fast}}$. So "$\sigma_2$ decays at rate 1.21 against 1.2" and
"$\sigma_3$ follows $e^{-|\lambda_{\mathrm{fast}}|\tau}$" hold for any system whose other singular values stay of
order one, with or without a fibration. The non-trivial content is only that the others do stay of order one, and
there the JSON is less clean than the text: "settle at 1.64–1.73 and 1.00" is true for $t_0 = 10^2$ and $10^4$; at
$t_0 = 1$ the second singular value is 0.45 at $\tau = 1$ (not mentioned), at $t_0 = 0.01$ it is $6.5\times10^{-3}$
(mentioned under "Limits"). "Down to $10^{-17}$" has two resolved points per state; the values below $10^{-12}$ are
rounding.
*Fix.* Test Corollary 6.1 where $d - k \ge 2$: HIRES, POLLU, or the primordial network (§0.4 item 5 reports two
collapsed singular values). There Liouville fixes only the product of the small singular values, and the theorem
predicts each is below $Ce^{-\gamma\tau}$ with γ the *slowest* fast rate. Check (ii), the $O(\varepsilon)$ slope
0.999, is a fair test and passes.

### T7

**C18 (major). T7's quantitative link to U4 is not made, and the norm it recommends conflicts with the project's
metric unless "measure" means the visited set.**
*Location:* P7.1, P7.2, §8 row U4, §9 item 3.
*Problem.* (a) Lemma 7.2 is an upper bound on the rank. For $d = 20$ the check gives degree 13 in the sup norm
against 9 (Hoeffding) or 4 (quantile) in measure: ranks at most 14 against at most 10 or 5, a factor 1.4–2.8. U4's
records show rank 1 against 4–10 and a 26-fold cost. Rank 1 at degrees 1–2 means the measure-sense model is a
product of nearly linear factors, which is not what "low degree in one aggregate" predicts (degree 4 in $s$ has
rank up to 5). T7 is consistent with U4 in direction only. A direct test is cheap: fit `lotka_volterra` at $d = 20$
by a polynomial in the aggregates $\sum_j A_{1j}x_j$ and compare its measure-sense and vertex errors with ChebTT's.
(b) For the application, the stated key metric is the maximum pointwise error. A measure-sense target on a box
abandons it. The reading that keeps it is the sup norm over the visited set (the tube of proposed theorem N3), not
an η-quantile under the uniform measure on a box. §9 item 3 should say which is meant.
*Fix.* Add the direct test; restate P7.2 as "sup norm on the reachable set against sup norm on the box".

**C19 (minor).** Concentration of aggregates and the role of the hypercube's corners in polynomial approximation
are known (effective dimension: Caflisch, Morokoff and Owen 1997; corners: Trefethen 2017, which the memo already
lists). "Prior art: not searched separately" should be replaced by these two references.

### Numbers, synthesis, ranking

**C20 (minor). Numbers that differ from the JSON or the records.**
(a) T7: "1,490 times the tolerance" at $d = 100$; the JSON has 1,495. (b) T3 records paragraph: "12- to 54-fold"
per output; Allen–Cahn is $(17{,}103/10)/(325/2) = 10.5$. (c) T2: "at most 20"; 20.008. Every other number I
compared (about 60) matches, including §0's three rows against both `convergence.json` files and the ladder
records.

**C21 (major). The records paragraph in T3 reads SlowTable selectively.**
*Location:* "while SlowTable … grows mildly (Fisher–KPP at 0.01: 1,138, 2,560, 4,115 for $d = 2, 3, 5$), which is
what T6 … and T7 … predict".
*Evidence* (same records, seed 1, tolerance 0.01; ChebTT box against SlowTable): Fisher–KPP 409 / 1,138,
2,324 / 2,560, 14,358 / 4,115; Allen–Cahn 325 / 686, 530 / 2,395, 1,619 / 4,423; Lorenz-96 ($d = 4, 5$) 2,515 /
7,437, 7,230 / 11,382. SlowTable is cheaper in one of the eight recorded pairs, and there is no SlowTable record at
$d = 10$ in these folders. The records support "SlowTable's cost grows more slowly on Fisher–KPP"; they do not show
that it is cheaper, and on Allen–Cahn it is 2–4.5 times dearer at every recorded size.
*Fix.* Quote all three families, or drop the sentence.

**C22 (major). The ranking: foundation 1 is not the strongest as stated.**
*Location:* §9.
- T6 is an assumption plus corollaries (C15); ξ must be learnt as a $d$-variate map (C16); its singular-value
  checks are tautological (C17).
- T1 is valid and survives stiffness (C3) but overestimates the degree 4- to 43-fold on Robertson, decays with τ
  where the truth does not (C1, C4), and cannot be applied to $\Gamma_\tau$ (C16b).
- T2 is classical; its fixed-point part does not cover the benchmark's neutral transients (C7), and
  "rollouts bounded by $\varepsilon/(1 - \lambda_{\mathrm{slow}})$ in contracting systems" has no instance in the
  suite: §1.20 states that the stiff test problems are neutral, not contracting.
- Foundation 2 (T3) does not apply to the stiff ladder (C9).

What the data support more directly is a different combination: **accuracy on the attracting tube, with a
survived-error tolerance and audits on the visited measure** (proposed N3 with Proposition 2.2's first display and
Proposition 4.1). It needs no coordinates ξ, it explains U2 quantitatively (C6's table), it matches §0.4 item 5
("trains and tests on visited states") and SlowTable's refusals in the first step, and it keeps a sup norm (C18b).
T1, generalised as N1 and made two-sided by N2, is then the source of a priori degrees and of certificates in the
$k$ reduced variables (N4). I would rank: 1. tube accuracy (N3 + T2 + T4.1); 2. a priori analyticity (N1 + N2 + N4)
inside it; 3. the effective dimension, from N7 (provable from $J$) and from T6's coordinates once ξ can be learnt; 4. T3 only for lattices with
$\ell\tau = O(1)$.

**C23 (minor). §8 over-claims.** Beyond C1, C16c and C18a: "T6 supports U2 (fast = damped)" is right; "T5 explains
U5" should become "T7's mechanism explains U5's vertex failure" (C13); "T4 explains U7" is a restatement (a
non-certificate can accept falsely), acceptable if worded so.

### Literature

**C24 (major). T1's dissipative regime has close prior art that `LITERATURE.md` does not list: Carleman
linearization bounds.** For $x' = F_1x + F_2x^{\otimes2}$, Forets and Pouly (arXiv:1711.02552) and Liu et al., *Efficient
quantum algorithm for dissipative nonlinear differential equations* (arXiv:2011.03185), bound the error of a
truncated polynomial (Carleman) representation of the flow in its initial state; convergence for all times is
obtained under a negative logarithmic norm of $F_1$ and a smallness condition of the form
$R \sim \lVert F_2\rVert\lVert x_0\rVert/|\mu(F_1)| < 1$; arXiv:2606.24473 (*New convergence results for Carleman
linearization*) states the infinite-horizon case needs $\mu_1 < 0$ and gives none for $\mu_1 \ge 0$. This is the
memo's dissipative row ($R \ge (|\mu_0|/c)^{1/p}$ with $c = 2\lVert F_2\rVert$, $p = 1$) for quadratic fields,
with Taylor instead of Chebyshev expansions. (Verified from search snippets and my memory of the papers, not from
the full texts.) T1's novelty claim should narrow to: general holomorphic $F$, the tube bootstrap with a
time-integrated Λ, Bernstein polyellipses and weights β.

**C25 (minor). Rigorous integration does use logarithmic norms** (the open question at TH-11): Zgliczyński's
$C^0$/$C^1$ Lohner algorithms and the CAPD library bound the propagation of sets with the logarithmic norm of the
Jacobian over an enclosure, the real-variable analogue of Lemma 1.1. From memory; worth confirming and citing as
the source of the technique.

**C26 (minor). T6's "singular-value drop is observed numerically in TH-55" understates the prior art.** ISAT (Pope
1997, already in the repository's register via SlowTable) is built on the singular value decomposition of the
mapping gradient, whose small singular values in the fast directions shape the ellipsoids of accuracy; ILDM
(Maas–Pope 1992, III-18) counts the slow modes from the Jacobian. T6's novelty is the statement as an approximation
bound, as the memo says, not the observation. (My search did not retrieve the passage of Pope 1997; from memory.)

**C27 (minor).** See C12 (Sukharev; Wilks) and C19.

---

## Own hypothesis tests

Scripts and outputs are in `theory/reviews/scratch-round-1/` (`*.py`, `*.json`). Run from that folder with
`../../../.venv/bin/python <script>`.

| Test | Target | Could it refute? | Outcome |
|---|---|---|---|
| A `a_lorenz_regimes.py` (2 s) | T1's check "$\tau^{-1.04}$ confirms the neutral law" | Yes: if the law appears only at $F = 1$, the check discriminates | **Refutes the check.** Slopes $-0.80$ to $-1.11$ on $[0.25, 4]$ for $F = 0.5, 0.8, 1, 1.2$ and three seeds; regimes separate only for $\tau \ge 4$. $\mu_\infty(J) = 5$ on the box: the theorem's regime there is "expanding" (C1, C2). |
| B `b_robertson_t1.py` (3 min 14 s) | Theorem 1 on a stiff system; "robust to stiffness" | Yes: $\rho_{\mathrm{thm}} > \rho_{\mathrm{eff}}$ would refute validity; $\Lambda \sim \tau|\lambda_{\mathrm{fast}}|$ would refute robustness | **Confirms validity and robustness to stiffness** (24 of 24 comparisons valid; $\Lambda \le 1.75$ at $\tau|\lambda| = 42{,}000$). **Refutes "does not shrink with τ"** and the regime reading: bound decays, truth flat; degree overestimated 4- to 43-fold (C3, C4). |
| C `c_ls_leaf.py` (7 s) | P5.1 (factor $1/(1 + \Lambda_N) = 1/20$ at $d = 10$) | Yes: ratios near 20 would confirm | **Refutes P5.1.** Ratios 0.5–1.6 (interior), 2–5.7 (vertices), nearly seed-independent; the vertex error is the $L^2$ projection's (C13). |
| D `d_mol_window.py` (39 s) | P3.2 on the method-of-lines ladder | Yes: a window radius independent of $n$ would confirm | **Refutes P3.2 for these families.** Radius 4, 9, 16, 31 for $n = 10, 20, 40, 80$ at $10^{-3}$; the whole system at $10^{-5}$ (C9). Lemma 3.1's bound itself is nearly sharp. |
| E `e_kpp_singular_values.py` (1 s) | Proposed theorem N7 (not a claim of the memo) | Yes: counts growing with $n$ would refute it | **Confirms N7.** 7 / 10 singular values above $10^{-2}$ / $10^{-4}$ at $n = 40$ and 80; each of the first twelve below $0.92\,e^{(L_f - \lambda_m)\tau}$. |

Limits of my tests. A: one output and one fibre direction, as in the memo. B: $m(t)$ from 9 trajectories (a lower
estimate of the sup, so $\rho_{\mathrm{thm}}$ is slightly optimistic), original coordinates (the benchmark uses
log coordinates), one centre state; the first version of the script failed to optimise β from poor starts and was
corrected (Perron-vector starts, a smooth objective). C: $E_T$ is a minimax over a finite set, a lower bound on the
box's; the $L^2$ projection is a Monte-Carlo estimate on 200,000 points, which probably explains the one
disagreement (the Gaussian, 8.66 against 4.0). D: the box $[0.05, 0.95]^n$ of independent values, rougher than the
ladder's smooth fields; on smooth fields the window error is smaller but its scaling with $n$ is the same heat
kernel. E: two states per size, one step; the general statement of N7 is a proof sketch, not checked line by line.

---

## Proposed additional or alternative theorems

**N1 (componentwise tube lemma; generalises Lemma 1.1 and contains Lemma 3.1).** Let
$r: [0, \tau] \to \mathbb{R}^d_{>0}$ be $C^1$ and suppose that for all $t$, $x_0 \in B$ and $v \in \mathbb{C}^d$ with
$|v_j| \le r_j(t)$, with $J = J_F(x(t; x_0) + v)$:

$$\operatorname{Re}J_{ii}\,r_i(t) + \sum_{j \ne i}|J_{ij}|\,r_j(t) \;\le\; r_i'(t) \qquad (i = 1..d).$$

Then for $|w_{0,j}| \le \theta r_j(0)$, $\theta < 1$, the complex solution exists on $[0, \tau]$ with
$|z_j(t) - x_j(t)| \le \theta r_j(t)$, and $\Phi_\tau$ is holomorphic on the polyellipse with
$h_j(\rho_j - \rho_j^{-1})/2 = \theta r_j(0)$, bounded by $\max|x_i(\tau)| + \theta r_i(\tau)$.
*Proof sketch.* With $u = \max_j|w_j|/r_j(t)$ and $i$ an index attaining it,
$D^+|w_i| \le \operatorname{Re}A_{ii}|w_i| + \sum_j|A_{ij}||w_j|$ gives
$D^+u \le u\,[\operatorname{Re}A_{ii}r_i + \sum_j|A_{ij}|r_j - r_i']/r_i \le 0$; the row condition is convex in
$J$, so it passes to the mean-value matrix $A$; the bootstrap is Lemma 1.1's.
*Why it is better.* Lemma 1.1 is the special case $r(t) = \beta\,e^{\int_0^t m - \Lambda}$. A time-dependent
weight vector follows non-normal transients (in test B a fixed β must compromise between the initial layer and the
slow phase). The admissible $r(0)$ are those from which the majorant system $r' = C(t, r)\,r$ ($C$ the comparison
matrix over the tube) has a solution up to τ. Radii may differ by orders of magnitude between variables: large
$r_j(0)$ for variables far in the coupling graph gives $\rho_j \gg 1$, i.e. Lemma 3.1's light cone as coefficient
decay, and large $r_j(0)$ in fast directions would express T6 inside T1. Prediction to test: on Robertson, N1's
radii at $\tau = 10$ are larger than Theorem 1's (1.04–1.35).

**N2 (converse: a ceiling on any analyticity parameter from the step map's Jacobian).** If $\Phi_{\tau,i}$ is
holomorphic on $E_\rho$ with $|\Phi_{\tau,i}| \le M$, then for every $x \in B$ and $j$,
$|\partial_j\Phi_{\tau,i}(x)| \le M/\operatorname{dist}_j(x, \partial E_{\rho_j})$, and since the distance from
$[a_j, b_j]$ to $\partial E_{\rho_j}$ is at least $h_j(\rho_j - 1)^2/(2\rho_j)$ and at most
$h_j(\rho_j - \rho_j^{-1})/2$ (at the centre),

$$\frac{(\rho_j - 1)^2}{2\rho_j} \;\le\; \frac{M}{h_j\,\lVert\partial_j\Phi_{\tau,i}\rVert_{\infty,B}}, \qquad \rho_j - \rho_j^{-1} \le \frac{2M}{h_j\,|\partial_j\Phi_{\tau,i}(c)|}.$$

*Proof.* Cauchy's estimate on the disc of that radius in the $j$-th variable. $\square$
*Use.* $\lVert D\Phi_\tau\rVert$ is what the benchmark already records (`dynamics`: the growth of the Jacobian
products). Where it grows like $e^{\lambda\tau}$ with $M$ bounded (an attractor), every admissible ρ obeys
$\rho - 1 \lesssim \sqrt{2M/h}\,e^{-\lambda\tau/2}$: with Theorem 1 this brackets the true parameter and turns
"cost by dynamical class" into a two-sided statement. Together with Bernstein's converse theorem
($\limsup E_n^{1/n} = 1/\rho_{\mathrm{true}}$) it gives a lower bound on the asymptotic degree, which is what U1
needs. Prediction to test: on the 4-D Brusselator, $\lVert D\Phi_{2\tau}\rVert/\lVert D\Phi_\tau\rVert > 1$ on the
visited set.

**N3 (accuracy is needed only on an attracting tube).** Let $S$ be compact with $\Phi(S) \subset S$ (a piece of the
slow manifold, or the closure of the visited set), $U_r = \{p : \operatorname{dist}(p, S) \le r\}$, and suppose
(i) $\operatorname{dist}(\Phi(p), S) \le \kappa\operatorname{dist}(p, S)$ on $U_r$ with $\kappa < 1$ (under T6,
$\kappa = Ce^{-\gamma\tau/\varepsilon}$), (ii) $\sup_{U_r}\lVert\Psi - \Phi\rVert \le \varepsilon_1 \le (1 - \kappa)r$.
Then $\Psi(U_r) \subset U_r$; every surrogate chain starting in $U_r$ stays in $U_r$ with
$\operatorname{dist}(y_n, S) \le \kappa^n\operatorname{dist}(y_0, S) + \varepsilon_1/(1 - \kappa)$ for all $n$ (the
normal error does not accumulate), and Proposition 2.2's first display holds with every supremum taken over $U_r$.
*Proof.* $\operatorname{dist}(\Psi(p), S) \le \operatorname{dist}(\Phi(p), S) + \varepsilon_1 \le \kappa r + (1 - \kappa)r$;
iterate. $\square$
*Predictions.* The sup-norm requirement lives on a $k$-dimensional tube of width
$\varepsilon_1/(1 - \kappa)$, not on the box; states outside it occur only in the first step (the initial layer),
which is where §1.12 reports SlowTable's only misses and its 50–90% refusals; the survived error is the tangential
part and is what accumulates (C6's table). It also says what to measure next: κ and the tube width per problem.

**N4 (a sampled sup-norm certificate under analyticity; links T1 to T4).** Let $e = \Phi_{\tau,i} - p$ with $p$ a
polynomial of degree $n_j$ in $x_j$, and let $(\rho, M)$ be as in Theorem 1. On $E_\rho$,
$|p| \le \lVert p\rVert_{\infty,B}\prod_j\rho_j^{n_j}$ (Bernstein–Walsh, one variable at a time), so
$|e| \le M_e := M + \lVert p\rVert_{\infty,B}\prod_j\rho_j^{n_j}$. If $|e| \le t$ on the tensor Chebyshev–Lobatto
grid of degrees $N_j$, then

$$\sup_B|e| \;\le\; t\prod_j\Lambda^{\mathrm{Leb}}_{N_j} + 4M_e\sum_j\frac{\rho_j^{-N_j}}{\rho_j - 1}\prod_{l \ne j}\frac{\rho_l + 1}{\rho_l - 1}, \qquad \Lambda^{\mathrm{Leb}}_N \le \tfrac{2}{\pi}\log(N + 1) + 1 .$$

*Proof.* $e = I_Ne + (e - I_Ne)$; the first term by the Lebesgue constant of Chebyshev points, the second by
Theorem 1's display applied to $e$. $\square$
*Use.* The audit costs $\prod_j(N_j + 1)$ evaluations with $N_j \approx n_j + \log(M_e/t)/\log\rho_j$: the order of
the interpolant itself, against $(L/\Delta)^d$ for Proposition 4.4's Lipschitz class. It is exponential in the
number of variables, so it is a certificate for the $k$ reduced variables, or per fibre. This is the missing step
between "T1 gives $(\rho, M)$" and "the scheme has a guarantee".

**N5 (what a least-squares leaf does at a point; replaces P5.1).** Let $Pf$ be the $L^2(\mu)$ projection on $V$,
$r = f - Pf$, $G$ the empirical Gram matrix in an orthonormal basis φ, $K(x) = \lvert\varphi(x)\rvert^2$. Then
$f(x) - \mathrm{LS}f(x) = r(x) - \sum_i\ell_i(x)r(x_i)$ exactly;
$\Lambda_N(x) \le \sqrt{K(x)/\lambda_{\min}(G)}$; and for i.i.d. points, conditionally on $\lVert G - I\rVert \le 1/2$,
$\mathbb{E}\big[(\sum_i\ell_i(x)r(x_i))^2\big] \le 4\,\lVert r\rVert^2_{L^2}\,K(x)/N$ up to the usual truncation
argument (Cohen–Davenport–Leviatan's proof, evaluated at a point instead of integrated). So the leaf's error at $x$
is $|r(x)| \pm O(\lVert r\rVert_{L^2}\sqrt{K(x)/N})$: the $L^2$ projection's own pointwise error plus a
seed-dependent term that is $0.6\lVert r\rVert_{L^2}$ at a vertex when $N = 20p$. Test C agrees with it. It
predicts U5: a quadratic leaf passes a uniform test set at about $\lVert r\rVert_\infty$ over the bulk and fails
the vertices by $|r(\text{vertex})|$, for every seed.

**N6 (linear invariants).** If $CF \equiv 0$ for a $q \times d$ matrix $C$: (i) $\mu(J_F) \ge 0$ in every norm (C4);
(ii) $C\,e_n = \sum_{k<n}C\,\delta(y_k)$ exactly, undamped, and zero iff $C\delta \equiv 0$; (iii) T1, Proposition
2.1 and Proposition 2.2 hold verbatim for the system reduced to a leaf $\{Cx = \text{const}\}$ with a surrogate
that conserves $C$. The records show (ii) is not Robertson's dominant error (projection onto the invariants:
5.05 to 4.93), so this is needed for the statements to apply, not as an explanation of U2.

**N7 (singular values of a step map from volume contraction; no slow manifold needed).** Let
$\alpha_1(x) \ge \alpha_2(x) \ge \dots$ be the eigenvalues of the symmetric part of $J_F(x)$ (in the Euclidean
norm, or of $WJ_FW^{-1}$ for a fixed invertible $W$), and $s_m(t) = \sup_{x_0 \in B}\sum_{i \le m}\alpha_i(x(t; x_0))$.
Then for every $x_0 \in B$ and $m \le d$,

$$\prod_{i \le m}\sigma_i\big(D\Phi_\tau(x_0)\big) \le \exp\int_0^\tau s_m(t)\,dt, \qquad\text{so}\qquad \sigma_m\big(D\Phi_\tau(x_0)\big) \le \exp\Big(\frac1m\int_0^\tau s_m\Big).$$

*Proof sketch.* $\prod_{i \le m}\sigma_i(M) = \lVert\wedge^mM\rVert_2$; the $m$-th exterior power of the variational
solution solves a linear equation whose generator is the $m$-th additive compound of $J$, and the Euclidean log norm
of that compound is the sum of the $m$ largest eigenvalues of the symmetric part of $J$ (Muldowney 1990); apply
(L3). $m = d$ is Liouville's formula. $\square$
*Why it matters.* It bounds the singular values that T6 is about from $J$ alone, with no fibration, no standard
form and no ε, and it is computable on a trajectory. It is weaker than Corollary 6.1 where a fibration exists (the
symmetric part punishes non-normality, so $W$ must be chosen, as β in T1), but it is a theorem where T6 is an
assumption, and it covers the diffusion ladder: for $u' = D\Delta_nu + f(u)$ with $|f'| \le L_f$,
$s_m \le mL_f - D\sum_{i \le m}\lambda_i$ with $\lambda_i$ the Laplacian's eigenvalues, hence
$\sigma_m(D\Phi_\tau) \le \exp\big(\tau(L_f - \tfrac{D}{m}\sum_{i \le m}\lambda_i)\big)$, independent of $n$ once
$n \ge m$. For Fisher–KPP at $\tau = 0.01$ the bound allows about 13 singular values above $10^{-2}$ at every $n$ (from the continuous eigenvalues $\pi^2i^2$).
*Test E* (`e_kpp_singular_values.py`, 1 s): the measured counts above $10^{-2}$, $10^{-4}$, $10^{-8}$ are 7, 10, 15 at
$n = 40$ and 7, 10, 14 at $n = 80$, the same at a rough and at a smooth state, and every one of the first twelve
singular values is below $0.92\,e^{(L_f - \lambda_m)\tau}$ (the heat kernel's own law, sharper than the bound I can
prove). So the effective input dimension of the Fisher–KPP ladder at 0.01 is about 7 from $n = 10$ on, which is the
structure a scheme should use there; it is consistent with the records showing no saturation up to $d = 10$.

---

## Overall assessment

| Theorem | Proof | Check | Assumptions against real stiff problems | Verdict |
|---|---|---|---|---|
| T1 | Correct (Lemma 1.1, Theorem 1, Corollary 1.2; one factor 2 in a remark) | Scalar check fair; Lorenz-96 check does not discriminate (C2); no stiff case (C3) | (A2) holds for mass action. The dissipative regime is unavailable with invariants (C4); β must be optimised | **Keep; needs work.** Valid and stiffness-robust (test B). Restate predictions as bounds, add a converse (N2), generalise to N1, evaluate on the Brusselator. |
| T2 | Correct; 2.3's asymptotic wording wrong (C8a) | Cannot fail (C6) | $I - P$ singular with invariants; benchmark rollouts are neutral transients (C7) | **Keep the first display**; test P2.1 on the records (it holds on five rows); move the fixed-point part to a remark. |
| T3 | Correct | Fair for Lorenz-96 with measured constants | (A3) is not uniform in $d$ on a refined domain (C9) | **Restrict to $\ell\tau = O(1)$**; replace P3.2 on the ladder by N7 (tested, E). |
| T4 | Correct | (none needed) | Verifiable | **Solid, known** (cite Wilks, Sukharev). Add N4 to connect with T1. |
| T5 | Proposition 5.1 correct | Fair as a measurement of $\Lambda_N$ and $\lambda_{\min}$ | n/a | **Drop P5.1** (refuted, C13); replace by N5. |
| T6 | Tautological given (S3) (C15) | Singular-value checks are Liouville's formula (C17); the ε-slope check is fair | (S3) on a box is stronger than Fenichel; standard form rarely given; two different ε (C15) | **Keep the idea, rewrite coordinate-free**; test where $d - k \ge 2$; do not claim "cost in $k$" (C16). |
| T7 | Correct | Fair for the toy function | Exact aggregate assumed | **Keep as a remark**; add the direct test on `lotka_volterra`; fix the norm it implies (C18). |

Solid as they stand: T4, Lemma 3.1, Lemma 1.1 and Theorem 1 as mathematics. Needing work before anything is built
on them: T1's predictions, T6's statement, T2's scope. To drop or replace: P5.1, P3.2 on the method-of-lines
ladder, "U6 explained by T6", "cost depending on $k$, not $d$". The ranking in §9 should be reordered around the
attracting tube (C22).

Sources consulted in the literature check:
[arXiv:2011.03185](https://arxiv.org/pdf/2011.03185), [arXiv:1711.02552](https://arxiv.org/abs/1711.02552),
[arXiv:2606.24473](https://arxiv.org/pdf/2606.24473), [arXiv:2405.12714](https://web3.arxiv.org/pdf/2405.12714),
[Lu and Pope 2009](https://tcg.mae.cornell.edu/pubs/Lu_LRP_JCP_09.pdf),
[NIST Dataplot, tolerance limits (Wilks 1941)](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/tolelimi.htm).
