# Review of the theory memo, round 2 of 3

- **Reviewer:** Claude Fable 5.2, 2026-11-21, on `theory/MEMO.md` at commit `907cf70` (revision 1).
- **Read:** `MEMO.md` in full, `REVIEWS.md`, `reviews/round-1.md` (critiques and proposals), `checks/t6_hires.py`,
  `checks/common.py`, `reviews/scratch-round-1/b_robertson_t1.{py,json}`, `docs/SURROGATE_RESEARCH.md` §0.4, §1.12,
  §1.20, `src/problems/ode/{system,chemistry,reaction_diffusion,step_maps}.py`, and `LITERATURE.md` by search.
- **Ran** (all in `theory/reviews/scratch-round-2/`, NumPy and SciPy, each under three minutes):
  `a_tube_robertson.py` (§9 item 1 on Robertson), `b_staircase.py` (Proposition 6.6 on Robertson, the 4-D Brusselator
  and POLLU), `c_bruss_u1.py` (U1: Theorem 1, Proposition 1.3 and the measured degrees and ranks on the 4-D
  Brusselator), `d_robertson_leaf.py` (Lemma 1.2 in leaf coordinates, reduced Robertson's $\mu_0$),
  `e_small_checks.py` (Proposition 2.3's bounded case; a path-averaged form of Proposition 6.6). Their JSON and log
  files are beside them.
- **Not done:** the primordial network; `lotka_volterra` at $d = 20$ (T7's open test); full texts of the references
  I name under "Literature" (cited from memory, marked as such).

**Verdict in one paragraph.** Revision 1 fixed what round 1 asked: the proofs I rechecked are correct, bounds are
called bounds, and the new statements are mostly sound. Three things are still wrong or overclaimed. (1) The new
top-ranked foundation is stated for "the visited states", but Proposition 2.4 needs accuracy on a full-dimensional
tube around a set that contains every neutral direction; a surrogate accurate to $4\times10^{-9}$ dex on held-out
visited states of Robertson diverges to 10 dex in 39 steps (test A). (2) Theorem 1 is valid but vacuous on the
problem it was meant to explain: on the 4-D Brusselator it certifies degrees 1,272 to 32,236 where 2 to 4 suffice,
and Lemma 1.2 does not rescue it on Robertson (tests C, D). (3) Proposition 6.6's frozen count holds on POLLU
(24 of 24) but is wrong by 11 to 170 decades on Robertson at long steps; a path-averaged exponent repairs it (tests B,
E). Proposition 4.5 is correct but weaker than a one-line argument gives, and its cost estimate is wrong.

---

## Audit of round-1 resolutions

Only resolutions that need a comment are listed; the others (C2, C5, C8 except one item, C10–C12, C14, C19–C21,
C23, C25–C27) are correct and sufficient as made.

| # | Resolution in revision 1 | Audit |
|---|---|---|
| C1 | Regimes called bounds; Proposition 1.3 added; Brusselator test left open | Sufficient in wording. The open test is now run (test C): Theorem 1 is valid and vacuous there, and Proposition 1.3's ceiling does not move between $\tau = 0.25$ and 0.5, so neither says anything about U1. See D6. |
| C3 | Test B quoted; $\beta$ must be optimised | Correct, but the box of test B ($\pm50\%$ of $y(100)$) is far smaller than the benchmark's (one decade in $y_2$ after step 1, six with step 0). "Valid on a stiff network" in §9 should carry that scope. See D7. |
| C4 | Remark 1.5: reduce to the leaf | Partly cosmetic: the remark asserts a route and leaves it open. Test D answers it: the reduced system has $\inf_\beta\mu_\beta = -0.0089$ at $y(100)$ (eliminating $y_3$ or $y_2$; $+0.0089$ eliminating $y_1$), but the margin is 8.7% of nothing useful: on the $\pm50\%$ box no tube uniform in $\tau$ exists. See D5. One sentence is wrong as written ("$\Lambda$ grows with $\tau$", D4). |
| C6 | The committed table is the check | Correct as far as it goes (five rows, one problem, one family of surrogates). Test A adds rows where P2.1 fails by nine orders; it is conditional on the tube hypothesis. See D1. |
| C7 | Fixed-point part demoted to a remark for the reduced system | Sufficient, with one more caveat: Robertson's reduced equilibrium is not hyperbolic (D10), so the remark never applies to it. |
| C8 | Proposition 2.3's remainder corrected | The remainder $\mathrm{Lip}(\delta)\varepsilon n^2/2$ is right. The sentence "stays bounded when $\bar\delta = 0$" is true only inside the first-order window (D9). |
| C9 | T3's scope restricted | Sufficient. |
| C13 | P5.1 withdrawn; Proposition 5.2 | Sufficient. The variance formula should be stated exactly (D12). |
| C15, C16 | (S) restated with caveats; "cost in $k$" withdrawn | Sufficient. Corollary 6.4 drops the remainder (D13). |
| C17 | `t6_hires` as the non-trivial check | Correct and a fair test. It is one weakly nonlinear network; test B extends it and finds where it fails (D3). |
| C18 | Sup norm on the reachable set | The right reading, but "reachable set" and "tube of Proposition 2.4" are different sets when there are invariants or when the surrogate is not normally contracting (D1, D2). |
| C22 | Ranking rebuilt around the tube | Accepted too readily. The proposition I proposed in round 1 (N3) is correct but its hypothesis (ii) is the whole difficulty, and the memo's reading of it ("the visited states", "audit … on the visited measure") is refuted by test A. See D1. |
| C24 | Carleman bounds cited | Sufficient (still from abstracts). |
| N1 | Lemma 1.2 | Proof correct. Stated with one tube for every $x_0$, which the proof does not need and which costs a factor in test D (D5). |
| N2 | Proposition 1.3 | Correct, but loose by a square root at interior points; the remark's exponent $\lambda\tau/2$ should be $\lambda\tau$ there (D8). |
| N4 | Proposition 4.5 | Correct but needlessly weak, with a wrong cost estimate (D11). My round-1 proposal had the same defect. |
| N7 | Theorem 6.5, Proposition 6.6 | Theorem 6.5 is the classical trace-formula argument (Literature). Proposition 6.6 is proved only for constant $J$; its extension is the claim that carries §8's "explains U3" and it fails at long steps (D3). |

---

## Critiques

### D1 (critical). §9 item 1, §8 row U2, P2.3: the tube is not "the visited states", and accuracy on visited states does not control a rollout

*Location.* §9 item 1 ("a tube around the attracting set (the visited states) … audit with fresh points on the
visited measure … matches the benchmark's practice of training and testing on visited states"); Proposition 2.4;
P2.1, P2.3; P7.2.

*Problem.* Proposition 2.4 is correct, but its hypothesis (ii) is $\sup_{U_r}\lVert\delta\rVert \le \varepsilon_1$ on
$U_r$, a neighbourhood of $S$ of full dimension $d$ and width $r \ge \varepsilon_1/(1 - \kappa)$. The visited states
of a stiff system lie within $e^{-\gamma t}$ of a set of dimension $k < d$; accuracy on them says nothing about
$\delta$ at distance $\varepsilon_1$ from $S$, which is where the chain is after one step. What is needed besides
accuracy on $S$ is that the surrogate itself contracts normal to $S$ (Proposition P1 below: $\mathrm{Lip}$ of $\delta$
across the tube below $1 - \kappa$). An audit on the visited measure cannot see this, and neither can the survived
error of P2.1, which is evaluated at exact states.

*Evidence (test A; Robertson, log10 coordinates, $\Delta t = 1$, states from step 1 on).*
- Visited states: fast-normal offset at most $2.7\times10^{-5}$ dex, mass offset $8\times10^{-16}$: they lie on one
  curve (dimension 1 in $d = 3$).
- Contraction: a displacement of $\eta = 0.01$ to 1 dex in the fast direction leaves $1.7\times10^{-5}$ dex after one
  step (the floor of the quasi-steady proxy), so $\kappa_{\mathrm{fast}} \le 1.7\times10^{-5}$ at $\eta = 1$. A
  displacement of the total mass is kept exactly: $\kappa_{\mathrm{mass}} = 1.000000$.
- Surrogate V (polynomial of total degree 3 in the three log coordinates, least squares on visited states only):
  one-step worst on held-out visited states $3.7\times10^{-9}$ dex, survived error $2.4\times10^{-9}$, error on the
  tube of half-width 0.05 dex 0.46, $|\partial\Psi/\partial z_2|$ up to 6.3. Rollout: $3.7\times10^{-9}$, $2.3\times10^{-8}$,
  $9.7\times10^{-7}$, $4.9\times10^{-5}$, $2.5\times10^{-3}$, **10.4** dex at steps 1, 2, 5, 10, 20, 39. Rollout over
  survived error: $4\times10^{9}$. At degree 5: one-step $1.3\times10^{-10}$, rollout $4.3\times10^{-8}$ and growing
  ($|\partial\Psi/\partial z_2| = 1.23$), ratio 1,379.
- Surrogate T (same class, fitted on visited states and one tube sample per state): one-step on visited states
  $3.7\times10^{-3}$ (degree 3) and $5.1\times10^{-5}$ (degree 5), on the tube $6.3\times10^{-3}$ and $5.6\times10^{-4}$;
  rollout $3.7\times10^{-3}$ and $9.2\times10^{-5}$; the chain's fast offset $3.6\times10^{-3}$ and $6.7\times10^{-5}$,
  below $\varepsilon_1/(1 - \kappa)$ as Proposition 2.4 predicts; rollout over survived 19.6 and 6.8.

So Proposition 2.4 **survives** with its hypothesis as stated (surrogate T), and the memo's reading of it is
**refuted**: V is $10^6$ times more accurate than T on the visited measure and useless in a chain. P2.1 holds for T
(6.8 and 19.6 beside the memo's 18 to 41) and fails for V. P2.3 ("accuracy beyond the tube is not needed") is
untested in the direction that matters, and "accuracy on the tube is needed" is now shown.

*Fix.* State the foundation as: sup-norm accuracy on $U_r$, or accuracy on $S$ together with a bound on the
surrogate's normal derivative (Proposition P1). Replace "audit on the visited measure" by an audit on the tube (visited
states displaced by up to $r$) or on the surrogate's own chain states, which is what Proposition 2.2 calls for
($\delta(y_k)$, not $\delta(x_k)$) and what `chebtt_roll` already does. Say that the benchmark's schemes satisfy the
hypothesis only because they query the oracle off the visited set (ChebTT on the bounding box) or use its Jacobian
(SlowTable), not because they are tested on visited states. P2.1 should carry the condition.

### D2 (major). Proposition 2.4 with invariants: $S$ must contain the neutral directions, where the benchmark has no states

*Location.* Proposition 2.4, "Under T6, $\kappa$ is the fast contraction per step and $S$ the slow manifold"; Remark 2.5.

*Problem.* Hypothesis (i) needs $\kappa < 1$ for every displacement off $S$. A linear invariant is neutral
($\kappa_{\mathrm{mass}} = 1$, test A), so $S$ must be the slow manifold across all leaves (dimension $k + q$), not
its intersection with the leaf the data live on. `robertson_initial` puts every state on the leaf $\sum y = 1$; the
tube $U_r$ then contains states of other total mass, never visited and never tested. The same holds for HIRES and
POLLU as far as their initial states share invariants. This is the assumption that fails for mass-action networks.

*Fix.* Either add to Proposition 2.4: "if $C\Psi = C$ on $U_r$ (the surrogate conserves the invariants), the
statement holds on each leaf with $S$ replaced by $S \cap \{Cx = c\}$", which makes projection onto the invariants a
structural requirement and not the optional step Remark 2.5 treats it as; or require samples off the leaf. Remark
2.5's "projecting barely helps (5.05 to 4.93)" is a fact about ChebTT on a box, which is accurate off the leaf anyway.

### D3 (major). Proposition 6.6 beyond constant $J$: holds on POLLU, fails on Robertson at long steps; §8's "explains U3" rests on the unproved part

*Location.* Proposition 6.6 ("valid while $J$ varies little over $\tau$"), P6.1, §8 row U3, §9 item 3.

*Evidence (test B; Magnus product for $D\Phi_\tau$, converged to $5\times10^{-4}$ decades).*
- POLLU (20 species, 6 invariants, $t_0 = 1, 10, 30$, $\tau = 0.01$ to 10, linear and log coordinates): the counts above
  $10^{-3}$ and $10^{-6}$ equal the frozen prediction in **24 of 24** cases each (15, 13–14, 13, 11 values at
  $\tau = 0.01, 0.1, 1, 10$). Individual values are within 1.1 decades ($\sigma_1 = 12$ against 1: a non-normal
  transient, inside $\kappa(V) = 99$). I expected this test to fail; it did not.
- 4-D Brusselator: 18 of 20 at $10^{-3}$, 20 of 20 at $10^{-6}$; the one miss ($t_0 = 2.5$, $\tau = 2$: 3 against 2) is a
  value at the threshold.
- Robertson: 26 of 30. The misses are large: from $t_0 = 0.01$ and 1 at $\tau = 100$ the frozen prediction of
  $\sigma_2$ is $2.9\times10^{-18}$ and $1.7\times10^{-13}$ against the true $3.1\times10^{-2}$ and $4.0\times10^{-2}$; at
  $\tau = 1000$ it is $10^{-176}$ against $3.5\times10^{-3}$ (test E). The frozen count gives rank 1 where the map has
  rank 2: it **undercounts**, the unsafe direction for a scheme that drops coordinates. At the benchmark's
  $\tau = 1$ the count is right (0.82 against 0.67).
- The eigenvalues at the *end* state give the right count in every Robertson case, and the path average
  $\exp\int_0^\tau\lambda_{\mathrm{slow}}(x(t))\,dt$ reproduces $\sigma_2$ within 3 to 15% in all twelve cases of test E
  (for example $2.85\times10^{-2}$ against $3.08\times10^{-2}$ at $t_0 = 0.01$, $\tau = 100$).

*Problem.* The proposition proves the constant case; "conjecture in general" is in the text, yet P6.1 and §8 use the
frozen count as the explanation of U3 and §9 item 3 calls it "provable from $J$". With $\kappa(V) = 99$ to
$6.9\times10^{3}$ (POLLU, linear and log) the proved two-sided bound is also loose by 2 to 4 decades in $\eta$ even for
constant $J$.

*Fix.* Replace the frozen exponent by the path average (Proposition P3 below, Levinson-type), state the count as
"between the frozen counts at $\eta\kappa(V)$ and $\eta/\kappa(V)$" in the proved case, and quote the Robertson
failure as the limit of the frozen form.

### D4 (minor). Remark 1.5: "$\Lambda$ grows with $\tau$" does not follow from an invariant

*Location.* Remark 1.5. *Problem.* An invariant gives $\mu(J) \ge 0$, hence $\mu_0 \ge 0$ and no first regime; that
is correct. It does not give $m > 0$: for linear conservative kinetics $x' = Ax$ with zero column sums and
non-negative off-diagonal entries, $\mu_1(A) = 0$ and the tube has no nonlinearity, so $\Lambda = 0$ for every $\tau$.
$\Lambda$ grows when $\inf_\beta\mu_\beta > 0$ on the tube (Robertson: 0.022) or when $c > 0$ in Corollary 1.4. *Fix.*
"$\Lambda$ cannot be made zero by a negative $\mu_0$; it grows with $\tau$ when $\inf_\beta\mu_\beta > 0$ or the
tube's nonlinear term is positive." Also Corollary 1.4's second and third formulas need $(pc\tau)^{-1/p} \le r_{\max}$
(for short $\tau$ the supremum is at $r_{\max}$).

### D5 (major). Lemma 1.2 and Remark 1.5 do not yet improve Theorem 1 on Robertson; the lemma is stated more narrowly than its proof

*Location.* Lemma 1.2 (statement), Remark 1.5, §9 item 2 ("componentwise radii are the next test"), §10.

*Evidence (test D; box at $y(100)$, entrywise majorant of the Jacobian in leaf coordinates $(y_1, y_2, c)$, equal
relative radius $s$ in every coordinate, trajectories from the centre and 8 corners).*
- Reduced $\mu_0$: $\inf_\beta\mu_\beta(J_{\mathrm{red}}) = -0.00893$ at $y(100)$ (the slow eigenvalue), so the open
  item "is reduced Robertson's $\mu_0 < 0$" is answered yes at a real state, for two of the three eliminations.
- On the $\pm50\%$ box the largest admissible $s$ is 0.064, 0.064, 0.035, none at $\tau = 0.01, 0.1, 1, 10$ with one
  tube for every $x_0$, and 0.064, 0.064, 0.047, 0.019 with one tube per trajectory: $\rho = 1.07$ to 1.02, degrees
  (from $\rho$ alone) 109 to 356, against Theorem 1's 19 to 202 in round-1 test B. On the $\pm10\%$ box: $\rho = 1.89$,
  1.89, 1.63, 1.17 against Theorem 1's 7.6, 7.4, 5.5, 2.2.
- So no $\tau$-uniform tube appears, and Lemma 1.2 in this form is worse than Theorem 1 with the exact supremum.

*Problem.* (a) The negative margin of the reduced comparison matrix is $(k_1 + k_3y_2)\,2k_2y_2$ against
$k_3y_3(k_1 + k_3y_2)$, 8.7% at $y(100)$; variation of $y_2/y_3$ across the box (ninefold at $\pm50\%$) and the shear
into leaf coordinates (the $y_3$ radius becomes $r_c + r_1 + r_2$, about four times $h_3$) consume it. The first
regime exists pointwise, not on a box of the benchmark's size. (b) The lemma requires one $r(t)$ for all $x_0 \in B$.
Its proof is per trajectory: any family $r(t; x_0)$ satisfying the row condition along its own trajectory works, and
Theorem 1 needs only $\inf_{x_0}r_j(0; x_0)$. In test D this is the difference between no certificate and a
certificate at $\tau = 10$.

*Caveat on my test.* The majorant bounds each entry separately; the exact supremum over the tube (as in test B)
would do better. The comparison is therefore between Theorem 1 at its best and Lemma 1.2 at a cheap bound; it shows
that Lemma 1.2 does not help *for free*, not that it cannot help.

*Fix.* State Lemma 1.2 with $r(t; x_0)$. Record test D's outcome under Remark 1.5 and withdraw the expectation in
§9 item 2 until an exact-supremum computation shows a gain.

### D6 (major). Theorem 1 and Proposition 1.3 on the Brusselator: valid, and silent about U1

*Location.* T1 "U1", P1.2 ("the true $\rho$ is bracketed"), §9 item 2 ("the source of degrees and of certificates"), §10.

*Evidence (test C; bounding box of the benchmark's visited states, 0.5 to 0.8 dex wide).*
- Theorem 1 (own variables, $\beta$ optimised, entrywise upper bound of $m$): $m$ between 14 and 22 at $\tau = 0.25$,
  $\Lambda = 4.42$ and 6.35 at $\tau = 0.25$ and 0.5; certified $\rho - 1 = 3.6\times10^{-3}$, $1.9\times10^{-3}$ and
  $3\times10^{-4}$, $1\times10^{-4}$; measured $\rho_{\mathrm{eff}} = 4.1, 4.2, 9.2, 9.2$ and $3.0, 3.1, 5.3, 5.3$. Certified
  degrees for two digits: 1,272 to 2,387 and 15,223 to 32,236, against measured 2.1 to 3.3 and 2.8 to 4.1. The bound is
  valid and overestimates the degree 400- to 8,000-fold. The reason is not stiffness: $2uv - (B + 1)$ reaches about 11
  at the box's corner, so the log norm is large and positive on a box whose trajectories all ring down.
- Proposition 1.3's ceiling: $\rho \le 10.0, 10.1, 16.8, 16.7$ at $\tau = 0.25$ and $10.0, 10.1, 11.4, 11.4$ at 0.5. The
  Jacobian does not grow ($h_j\lVert\partial_j\Phi\rVert = 1.08 \to 0.89$ and $0.58 \to 0.76$), so the ceiling does not
  tighten while the measured $\rho$ falls.
- What does change (full $13^4$ tensor grid in log coordinates): the degree needed for $10^{-2}$ dex goes from
  (5, 5, 3, 3) to (7, 7, 4, 4) per axis, $\rho_{\mathrm{eff}}$ from (2.67, 2.69, 4.78, 4.80) to (2.21, 2.22, 3.92, 3.92), the
  middle unfolding rank from 4 to 6; the cross-cost proxy $\sum_jr_{j-1}r_j(n_j + 1)$ rises 2.47-fold ($10^{-2}$) and
  2.34-fold ($10^{-3}$), against the recorded 2.9 (11k against 3.8k solves).

*Problem.* The bracket of P1.2 is $[1.004, 10]$ around a measured 4.1: three and a half orders of magnitude in degree.
U1 on the Brusselator is reproduced by direct measurement (degree and rank both grow under composition) and by
neither T1 statement. §9 item 2 cannot be "the source of degrees" in this state, and Proposition 4.5, which takes
Theorem 1's $(\rho, M)$, would need grids of $10^3$ to $10^4$ points per axis.

*Fix.* Report test C in T1's checks; in §8 change U1/T1 to "valid, uninformative on the Brusselator"; move T1 below
the effective dimension in §9 (see "Ranking"). The honest statement of what explains U1's direction is the
composition law P5 below.

### D7 (major). Realism of T1's assumptions for the benchmark's problems

*Location.* T1 assumptions and "Limits"; §9 item 2.

*Problem.* (a) The surrogates work in log10 coordinates on boxes several decades wide (Robertson's training box
spans $z_2 \in [-12, -4.4]$ with step 0, one decade without). Theorem 1 has been evaluated only in the system's own
variables on a box of $\pm50\%$ (test B) or less than one decade (test C). In own variables a box that wide contains
states whose fast rates differ by orders of magnitude, and one weight vector $\beta$ cannot serve them; in log
variables the vector field $F(10^z)/(10^z\ln 10)$ is entire, but no tube computation exists and linear invariants are
no longer affine leaves, so Remark 1.5's reduction does not carry over. (b) (A2) excludes the primordial network's
rate fits only where they use $\max$ or tables; that part is stated. (c) The product $\prod_l(\rho_l + 1)/(\rho_l - 1)$
with $\rho - 1 \sim 10^{-3}$ is $10^{13}$ in four variables: even where valid the display is far from the measured
error.

*Fix.* State the scope: "verified on boxes of relative half-width at most 0.5 in own variables". Add as an open item
the tube computation in log variables on the benchmark's box.

### D8 (major). Proposition 1.3 is loose by a square root; the remark's exponent is $\lambda\tau$, not $\lambda\tau/2$, away from the endpoints

*Location.* Proposition 1.3 and the two sentences after it; the Bernstein-ellipse paragraph of §1.

*Problem.* The proof uses the smallest disc, at a focus, with the largest derivative, wherever it is. For
$x_j = c_j + sh_j$ the largest disc inside $E_{\rho_j}$ has radius $\operatorname{dist}(x_j, \partial E_{\rho_j})$, which
for $\rho_j - 1 \ll 1$ is $h_j\ln\rho_j\sqrt{1 - s^2}\,(1 + o(1))$, not $h_j(\rho_j - 1)^2/(2\rho_j)$. So the
correct ceiling is pointwise:

$$\operatorname{dist}\bigl(x_j, \partial E_{\rho_j}\bigr) \le \frac{M}{|\partial_j\Phi_{\tau,i}(x)|}\quad\text{for every } x \in B,$$

and where the large derivative sits in the interior, $\rho_j - 1 \lesssim M/(h_j|\partial_j\Phi|\sqrt{1 - s^2})$: with
$\lVert D\Phi_\tau\rVert \sim e^{\lambda\tau}$ this is $e^{-\lambda\tau}$, the same exponent as Theorem 1's lower bound
has with $\mu_0$ in place of $\lambda$. The memo's own Lorenz-96 number agrees: at $F = 8$ the measured rate of
$\rho_{\mathrm{eff}} - 1$ is 3.3 against $\lambda_{\max} = 1.68$; the memo's ceiling allows any rate above 0.84, the
pointwise one any rate above 1.68. *Fix.* Replace the statement by the pointwise one (the present one is its
corollary at the foci) and the remark's exponent accordingly. Also say that the ceiling constrains pairs $(\rho, M)$
only through $M$; by Bernstein's converse theorem (ATAP Thm 8.3) the true asymptotic rate is set by the largest
ellipse of analyticity, whatever $M$.

### D9 (minor). Proposition 2.3: "stays bounded when $\bar\delta = 0$" holds only to first order

*Location.* Last sentence of Proposition 2.3; the sanity check "bounded defects at most 20.01 over 4,000 steps".
*Evidence (test E).* Rotation by the golden angle with the tangential defect $\varepsilon\cos(q\varphi)$ (zero orbit
average, a trigonometric polynomial): the phase error over $\varepsilon$ is 2.4, 3.5, 15.6, 153, 1,523 at
$n = 10, 10^2, 10^3, 10^4, 10^5$ for $\varepsilon = 10^{-2}$, $q = 3$; the drift per step is $0.51\,q\varepsilon^2$
(also at $\varepsilon = 10^{-3}$), $0.10\,q\varepsilon^2$ at $q = 1$, $1.43\,q\varepsilon^2$ at $q = 8$. The perturbed
circle map has a rotation number that differs at order $\varepsilon^2$, so the error grows linearly beyond
$n \sim 1/(q\varepsilon)$, which is exactly where the proposition's remainder $\mathrm{Lip}(\delta)\varepsilon n^2/2$
stops being small. *Fix.* "…and $S_n$ stays bounded; $e_n$ then stays bounded for $n \ll 1/\mathrm{Lip}(\delta)$ and
drifts at a rate of order $\mathrm{Lip}(\delta)\,\varepsilon$ per step beyond."

### D10 (minor). The fixed-point remark: constant, existence, and Robertson

*Location.* Remark after Proposition 2.2. (a) From $e = (I - P)^{-1}[\delta(x^*) + D\delta\,e + O(K_2\lVert e\rVert^2)]$ the
second-order term is $\lVert(I - P)^{-1}\rVert^2\lVert D\delta\rVert\varepsilon + \lVert(I - P)^{-1}\rVert^3K_2\varepsilon^2/2$:
the constant involves the cube with $K_2$ and the square with $\lVert\delta\rVert_{C^1}$; the smallness condition the
memo states is the right one for the first. Existence of $y^*$ needs the same condition and should be said. (b)
Robertson's equilibrium reduced by its invariant is $(y_1, y_2) = (0, 0)$ with
$J_{\mathrm{red}} = \begin{pmatrix}-k_1 & k_3\\ k_1 & -k_3\end{pmatrix}$, singular: the approach is algebraic and $I - P$ is
not invertible even after reduction. The remark applies to HIRES-like networks, not to Robertson.

### D11 (major). Proposition 4.5 is weaker than a one-line argument, and its cost estimate is wrong for its own bound

*Location.* Proposition 4.5 and the paragraph after it.

*Problem.* For $N_j \ge n_j$ the interpolation operator reproduces $p$, so $e - I_Ne = \Phi_{\tau,i} - I_N\Phi_{\tau,i}$
and

$$\sup_B|e| \le t\prod_j\Bigl(\tfrac{2}{\pi}\log(N_j + 1) + 1\Bigr) + 4M\sum_j\frac{\rho_j^{-N_j}}{\rho_j - 1}\prod_{l\ne j}\frac{\rho_l + 1}{\rho_l - 1},$$

with Theorem 1's $M$, not $M_e = M + \lVert p\rVert\prod_j\rho_j^{n_j}$. Bernstein–Walsh is not needed. With $M_e$ as
stated, the required grid is $N_j \ge [\sum_ln_l\log\rho_l + \log(\lVert p\rVert/t)]/\log\rho_j$, about $dn$ for
isotropic $\rho$, not "$n_j + \log(M_e/t)/\log\rho_j$ … the order of the interpolant itself": the stated bound costs
$d^d$ times more points than the sentence says. The iterated Bernstein–Walsh step itself is correct. Two further
points belong in the text: evaluating $\Phi$ on the full grid of degree $N$ already yields $I_N\Phi$, a surrogate with
the second term as its certified error, so the proposition certifies a compressed $p$ at the price of the uncompressed
interpolant; and with Theorem 1's current $\rho$ (D6) $N$ is in the thousands.

*Fix.* Restate with $M$ and the condition $N \ge n$; correct the cost sentence.

### D12 (minor). Proposition 5.2's variance

*Location.* Proposition 5.2, "root-mean-square of order $\lVert r\rVert_{L^2}\sqrt{K(x)/N}$". With $G = I$ the term
is $\frac1N\sum_ik_x(x_i)r(x_i)$, $k_x(y) = \varphi(x)^{\mathsf T}\varphi(y)$, of mean zero and variance exactly
$\frac1N\mathbb{E}[k_x(X)^2r(X)^2]$. Since $\mathbb{E}[k_x^2] = K(x)$, the memo's form holds when $r^2$ and $k_x^2$ are
uncorrelated; at a vertex $k_x^2$ concentrates near that vertex, where $|r|$ is largest, so the form with
$\lVert r\rVert_{L^2}$ underestimates and the rigorous bound is $\lVert r\rVert_\infty\sqrt{K(x)/N}$. *Fix.* State
the exact variance; it is one line and removes "sketch" from the leading term.

### D13 (minor). Smaller points

- Corollary 6.4: $\Phi_{\tau,i} = \Gamma_{\tau,i}\circ\pi + R_{\tau,i}$, not an equality.
- Theorem 6.5, "independent of $n$ once $n \ge m$": the eigenvalues of $-\Delta_n$ depend on $n$; they are bounded
  below uniformly (Neumann: $\lambda_i = 4n^2\sin^2\frac{(i-1)\pi}{2n} \ge 4(i - 1)^2$), which is what the bound needs.
- Proposition 6.6: the inequality $\sigma_i(ABC) \le \lVert A\rVert\sigma_i(B)\lVert C\rVert$ is the standard
  consequence of the min-max characterisation (a special case of Horn's product inequalities); for a repeated
  eigenvalue 0 (POLLU: six) $V$ is not unique and $\kappa(V)$ should be an infimum over eigenbases.
- Proposition 2.4, "Proposition 2.2 holds with every supremum over $U_r$": Proposition 2.2 needs a convex set, and
  $U_r$ is not convex around a curved $S$; take the suprema over the segments $[x_k, y_k]$ or over the convex hull.
- P2.2 ("a smooth surrogate … nearly attains $\varepsilon A_n$"): surrogate T of test A is smooth and its rollout is
  1.0 and 1.8 times its worst one-step error after 39 steps, against 9 to 24 for the four ChebTT rows. Its worst
  one-step error sits at a few early states and is not repeated along the trajectory. P2.2 is a statement about
  ChebTT on this problem, not about smoothness.
- §8, U5 "explains": round-1 test C had the vertex error match the $L^2$ projection for three of four functions;
  "explains in three of four cases" is the supported wording.

---

## Own hypothesis tests

| Test | Script | Prediction under test | Outcome |
|---|---|---|---|
| A | `a_tube_robertson.py` | Prop. 2.4; P2.1; P2.3; "tube = visited states" (§9 item 1) | Prop. 2.4 **survives** (chain offset $3.6\times10^{-3} \le \varepsilon_1 = 6.3\times10^{-3}$; $\kappa_{\mathrm{fast}} \le 1.7\times10^{-5}$). The reading "visited states" is **refuted** (one-step $3.7\times10^{-9}$, rollout 10.4 dex). P2.1 conditional: ratio 6.8–19.6 with tube accuracy, $1.4\times10^{3}$–$4\times10^{9}$ without. $\kappa_{\mathrm{mass}} = 1$: (i) fails for $S$ within a leaf. |
| B | `b_staircase.py` | Prop. 6.6's frozen count on nonlinear multi-scale systems | **Survives** on POLLU (24/24 at both thresholds, gaps $\le 1.1$ decades) and the Brusselator (18/20, 20/20). **Fails** on Robertson at $\tau = 100$ from early states (11–16 decades; count 1 against 2). |
| C | `c_bruss_u1.py` | Theorem 1 and Prop. 1.3 bracket the Brusselator's $\rho$ and account for U1 | Valid and vacuous: $\rho_{\mathrm{thm}} - 1 = 3.6\times10^{-3}$ and $3\times10^{-4}$ against measured 3.1 and 2.0; ceiling 10 at both $\tau$. U1 reproduced by measurement: cost proxy $\times2.47$ (record: $\times2.9$), from degree (5→7) and rank (4→6). A Bernstein lower bound on the degree gives only 1.2–1.7. |
| D | `d_robertson_leaf.py` | Lemma 1.2 in leaf coordinates beats Theorem 1 on Robertson; reduced $\mu_0 < 0$ | $\mu_0 = -0.0089$ at $y(100)$: **yes**. Beating Theorem 1: **no** with entrywise majorants ($\rho = 1.07$–1.02 against 1.44–1.04 on the $\pm50\%$ box; 1.89–1.17 against 7.6–2.2 on $\pm10\%$). Per-trajectory tubes help (0.047 against 0.035 at $\tau = 1$; a tube at $\tau = 10$ where the common one fails). |
| E | `e_small_checks.py` | Prop. 2.3's bounded case; a path-averaged Prop. 6.6 | Drift $0.5\,q\varepsilon^2$ per step: bounded only to first order. Path-averaged exponent matches $\sigma_2$ within 3–15% in 12 of 12 Robertson cases, including those where the frozen one is off by up to 170 decades. |

Limits of these tests. A: polynomial least squares of my own, not the benchmark's schemes; the quasi-steady proxy for
the distance to $S$ has a floor of $2\times10^{-5}$ dex. B: one trajectory per system. C: the supremum of $m$ is
bounded entrywise (so Theorem 1's exact value is somewhat better than quoted) over 41 sampled trajectories; ranks are
unfolding ranks of grid values, not of a cross approximation. D: as stated in D5.

---

## Ranking (§9) and remaining overclaims

1. **Item 1 as written is not a foundation yet**; with D1 and D2 repaired it is the right first item, because it is
   the only one whose hypothesis and conclusion were both measured on a real stiff problem and held (surrogate T). The
   repaired statement has three parts: accuracy on $S$, a bound on the surrogate's normal derivative, and exact
   conservation of the linear invariants. "Explains U2 quantitatively" is too strong: Proposition 2.4 gives no number
   for U2; P2.1's ratio (7 to 41) is an observation without a derivation.
2. **The effective dimension should be second**, ahead of T1: it survived on HIRES (24 of 24 counts) and POLLU (24 of 24 at each threshold), it
   becomes a two-sided statement with Proposition P2 below, and its failure mode is understood and repairable (P3).
3. **T1 third, as an open problem and not as "the source of degrees"**: valid everywhere it was tried, uninformative
   on the Brusselator (three orders of magnitude) and 4- to 43-fold off on Robertson, with no gain from Lemma 1.2 so
   far. Proposition 4.5 inherits this.
4. T3 as in the memo.

Remaining overclaims: §8 "U2 explains" (conditional on the tube hypothesis; five rows of one problem); §8 "U3
explains (staircase, Prop. 6.6)" (true at $\tau = 1$; the proposition is proved for constant $J$ and fails at long
steps); §9 item 2 "valid on a stiff network" (on a $\pm50\%$ box); §9 item 3 "provable from $J$ without a fibration"
(only Theorem 6.5 is, and it was useless without a weight on HIRES); P7.2 "the tube of Proposition 2.4" equated with
the reachable set (D2).

Realism for mass-action networks with conservation laws: (A1)–(A2) hold (polynomial fields); the first regime of
T1 is unavailable in full coordinates and too thin in leaf coordinates (D5); Proposition 2.4's hypothesis (i) fails
unless $S$ spans the leaves or the surrogate conserves the invariants (D2); (S) is plausible but unverified for
POLLU; the fixed-point remark fails for Robertson (D10).

---

## Proposed theorems

**P1 (the tube with a surrogate that is accurate only on $S$; repairs §9 item 1).** Let $S$, $U_r$, $\kappa$ be as
in Proposition 2.4(i), and suppose $\sup_S\lVert\delta\rVert \le \varepsilon_S$ and
$\lVert\delta(p) - \delta(q)\rVert \le \ell\,\operatorname{dist}(p, S)$ for $p \in U_r$ and $q$ a nearest point of $S$,
with $\kappa + \ell < 1$ and $r \ge \varepsilon_S/(1 - \kappa - \ell)$. Then $\Psi(U_r) \subset U_r$ and
$\operatorname{dist}(y_n, S) \le (\kappa + \ell)^n\operatorname{dist}(y_0, S) + \varepsilon_S/(1 - \kappa - \ell)$.
*Proof.* $\operatorname{dist}(\Psi(p), S) \le \kappa\operatorname{dist}(p, S) + \lVert\delta(q)\rVert + \ell\operatorname{dist}(p, S)$;
iterate. $\square$ Since $\ell \le \mathrm{Lip}_N(\Psi) + \mathrm{Lip}_N(\Phi)$ across the tube and the map's own normal
Lipschitz constant is of order $\kappa$, a sufficient design condition is that the surrogate's derivative normal to
$S$ is below $1 - 2\kappa$ in norm. Test A is both sides of it: $|\partial\Psi/\partial z_2| = 6.3$ and 1.23 (diverging
chains) against 0.24 and 0.03 (bounded chains). *With invariants:* if $C\Psi = C$ on $U_r$, apply the statement on
each leaf to $S \cap \{Cx = c\}$; without it $\kappa = 1$ along $C$ and no $r$ works. Consequences for a scheme: a
surrogate that is a function of slow coordinates only has $\ell \approx \kappa$ by construction; an increment
surrogate $z + g(z)$ needs $\partial_Ng \approx -I$, which a fit on visited states cannot learn.

**P2 (a lower bound that makes the effective dimension a law; T6, U3).** Let $\Phi$ be $C^2$ on the ball $B(p, r)$
with $\lVert D^2\Phi\rVert \le K_2$ and $\sigma_m(D\Phi(p)) = s$. For every continuous $c: B(p, r) \to \mathbb{R}^{m-1}$
and every map $G$,
$$\sup_{B(p,r)}\lVert\Phi - G\circ c\rVert \ge sr - \tfrac12K_2r^2\qquad(\text{so } \ge s^2/(2K_2)\text{ if } r \ge s/K_2).$$
*Proof.* Let $E$ be spanned by the first $m$ right singular vectors. By Borsuk–Ulam, $c$ restricted to the sphere
$\{p + v: v \in E, \lVert v\rVert = r\}$, of dimension $m - 1$, takes the same value at some antipodal pair $p \pm v$.
$G\circ c$ agrees there, while
$\lVert\Phi(p + v) - \Phi(p - v)\rVert \ge 2\lVert D\Phi(p)v\rVert - K_2r^2 \ge 2sr - K_2r^2$; one of the two errors is at
least half of that. $\square$ With Corollary 6.1 this is two-sided: $k$ coordinates suffice up to
$Ce^{-\gamma\tau}\operatorname{diam}$, and fewer than $m$ continuous coordinates, of any kind, cost at least
$\sigma_mr - K_2r^2/2$. For Robertson at $\tau = 1$ ($\sigma_2 \approx 0.8$–1.0, $\sigma_3 \le 10^{-10}$) the number of
coordinates is exactly 2 for every tolerance between the two. It holds in any coordinates, linear or not, which
answers C16a's concern from the other side: the count is necessary, not only sufficient.

**P3 (path-averaged staircase; replaces the frozen form of Proposition 6.6; sketch).** Along a trajectory let
$J(x(t)) = V(t)\,\mathrm{diag}(\lambda(t))\,V(t)^{-1}$ with $V$ of class $C^1$, and suppose the real parts are ordered
with an exponential dichotomy between the groups to be told apart ($\int_s^t\operatorname{Re}(\lambda_i - \lambda_j) \ge -c$
for $i$ in the slower group, $j$ in the faster, $s \le t$) and $\omega = \int_0^\tau\lVert V^{-1}V'\rVert\,dt$ small. Then
$D\Phi_\tau = V(\tau)\,[\mathrm{diag}(e^{\int_0^\tau\lambda_i}) + O(\omega e^c)\text{ relative, groupwise}]\,V(0)^{-1}$, and
$\sigma_i(D\Phi_\tau)$ lies within a factor $\kappa(V(0))\kappa(V(\tau))(1 + O(\omega e^c))$ of
$\exp\int_0^\tau\operatorname{Re}\lambda_{(i)}(x(t))\,dt$. *Sketch.* $Y = VZ$ gives $Z' = (\mathrm{diag}\lambda - V^{-1}V')Z$,
an $L$-diagonal system; Levinson's theorem (Coppel, *Dichotomies in Stability Theory*, 1978; Eastham, *The Asymptotic
Solution of Linear Differential Systems*, 1989) gives the diagonal form under the dichotomy; then the inequality of
Proposition 6.6. The finite-$\tau$ constants are not written out. *Evidence.* Test E: twelve Robertson cases within 3
to 15%, including $\tau = 1000$. The staircase becomes $\#\{i: \int_0^\tau\operatorname{Re}\lambda_i > \ln\eta\}$; a
scheme can evaluate it from Jacobians the solver already forms along the step.

**P4 (pointwise Cauchy ceiling; replaces Proposition 1.3).** Under Proposition 1.3's hypotheses, for every
$x \in B$ and $j$: $\operatorname{dist}(x_j, \partial E_{\rho_j}) \le M/|\partial_j\Phi_{\tau,i}(x)|$; for
$x_j = c_j + sh_j$ the left side is $h_j\ln\rho_j\sqrt{1 - s^2}(1 + O(\ln\rho_j))$. Proposition 1.3 is the case
$s = \pm1$. Proof and consequence in D8. A companion lower bound on the degree itself, free of $M$: if a polynomial
$p$ of degree $n_j$ in $x_j$ has $\lVert\Phi_{\tau,i} - p\rVert_{\infty,B} \le \mathrm{tol}$, then for any two points
$a < b$ of a fibre in direction $j$ (in units of $h_j$),
$n_j \ge (|f(b) - f(a)| - 2\,\mathrm{tol})\sqrt{1 - \max(a^2, b^2)}\,/\,[(b - a)(\tfrac12\operatorname{osc}f + \mathrm{tol})]$
(mean value theorem for $p$, Bernstein's inequality for $p$ minus a constant). It is a law for maps with steep
interior layers (relaxation oscillations, fronts, chaotic stretching: $n_j \gtrsim e^{\lambda\tau}$ as long as the
variation across the layer exceeds $2\,\mathrm{tol}$) and weak otherwise: on the Brusselator it gives 1.2 to 1.7
against needed degrees 3 to 10 (test C). So it does not explain U1 there.

**P5 (composition; what can be said about U1's direction).** For entire $F$ let $D_\tau \subset \mathbb{C}^d$ be the
open set of initial data whose complex solution exists on $[0, \tau]$. Then $\Phi_\tau$ is holomorphic on $D_\tau$,
$D_{\tau + \sigma} = D_\sigma \cap \Phi_\sigma^{-1}(D_\tau) \subset D_\sigma$, and the largest polyellipse around $B$ inside
$D_\tau$ shrinks (weakly) as $\tau$ grows: any certificate obtained from existence of complex solutions (Theorem 1,
Lemma 1.2) can only get worse with $\tau$. *Proof.* Existence on $[0, \tau + \sigma]$ is existence on $[0, \sigma]$
followed by existence from $\Phi_\sigma(z_0)$ on $[0, \tau]$. $\square$ This is one-sided: by Bernstein's converse the
true asymptotic degree is set by the largest polyellipse to which $\Phi_\tau$ *continues*, which can exceed $D_\tau$
($x' = x^2$: $\Phi_\tau(x_0) = x_0/(1 - x_0\tau)$ continues across the ray $x_0 \ge 1/\tau$ except at one pole). Whether
the nearest singularity of $\Phi_\tau$ is monotone in $\tau$ for a class of fields (it is for $x' = \mp x - x^3$ and for
$x' = x^2$) is the precise open question behind U1; a proof for dissipative polynomial fields would make "longer steps
are never asymptotically cheaper" a theorem. The measured fact on the Brusselator is $\rho_{\mathrm{eff}}$ from
(2.67, 4.8) to (2.21, 3.9) and the middle rank from 4 to 6 between $\tau = 0.25$ and 0.5.

**P6 (Proposition 4.5 as it should read).** In D11: for $N \ge n$,
$\sup_B|\Phi_{\tau,i} - p| \le t\,\Lambda_N + \lVert\Phi_{\tau,i} - I_N\Phi_{\tau,i}\rVert_{\infty,B}$, with Theorem 1's
bound and $M$ for the second term.

---

## Literature

Cited from memory; none re-read for this review.

- **Proposition 2.4 and "accuracy only where the states are".** Persistence of attracting sets under one-step
  perturbations is Kloeden and Lorenz, *Stable attracting sets in dynamical systems and in their one-step
  discretizations*, SIAM J. Numer. Anal. 23 (1986), and Stuart and Humphries, *Dynamical Systems and Numerical
  Analysis* (1996), already on `LITERATURE.md`'s list of texts to obtain. Restricting a tabulated map to the accessed
  region is the premise of ISAT (Pope 1997), which the memo cites for T6 only. The memo's "surrogate-specific part" is
  therefore ISAT's design principle; what would be new is P1's condition on the surrogate's normal derivative, if it
  is not already in the ISAT error analyses (not checked).
- **Lemma 1.2.** Componentwise differential inequalities with a Metzler majorant are the Kamke–Müller comparison
  theorem (Walter, *Differential and Integral Inequalities*, 1970) and the weighted $\ell_\infty$ contraction
  conditions of contraction theory (Bullo's monograph, listed in `LITERATURE.md`). The lemma is that argument on a
  complex tube; say so.
- **Theorem 6.5.** Bounding sums of Lyapunov exponents, hence products of singular values, by the trace of the
  symmetric part on $m$-dimensional subspaces is the Constantin–Foias trace formula (Comm. Pure Appl. Math. 38, 1985;
  Temam, *Infinite-Dimensional Dynamical Systems in Mechanics and Physics*, 1988, ch. V) and, for ODEs, Douady and
  Oesterlé (1980) and R. A. Smith (1986), on which Muldowney builds. The diffusion example is the standard
  attractor-dimension estimate. Not new; its use for a surrogate's input dimension is.
- **Proposition 6.6.** Counting modes with $\tau|\lambda_i|$ above a threshold is the criterion for exhausted modes
  in CSP (Lam and Goussis 1994, in `LITERATURE.md` as III-19) and the G-Scheme (Valorani and Paolucci 2009). "The
  count as an approximation-complexity statement" is new only with a lower bound such as P2. The path-averaged form
  is Levinson's theorem.
- **Proposition 1.3.** A Cauchy estimate; the sharp two-sided statement at fixed $\tau$ is Bernstein's converse
  theorem (ATAP Thm 8.3), which the memo should cite as the reason $\rho$ is a property of the map.
- **Proposition 4.5.** Rigorous a posteriori bounds for Chebyshev interpolants from ellipse data exist in validated
  numerics (Brisebarre and Joldeş, *Chebyshev interpolation polynomial-based tools for rigorous computing*, ISSAC
  2010; unverified). The proposition is elementary and should not be listed as new.
- No wrong novelty claim was found for Theorem 1 beyond round 1's C24 and C25.

---

## Overall assessment

Revision 1 is honest about what is proved and most of its new statements are correct; the errors found this round are
in how three of them are read. The foundation that can be built on today is: (i) the tube, restated with the
surrogate's normal contraction and conservation of invariants as hypotheses (P1), which test A supports on both
sides; (ii) the effective dimension, with the path-averaged exponent (P3) and the Borsuk–Ulam lower bound (P2), which
makes it the memo's first two-sided statement. T1 is correct mathematics without predictive value on the benchmark's
boxes so far, and U1 remains unexplained by a theorem: the data say that composition raises both degree and rank, and
P5 isolates the open question. For round 3 the memo should (a) rewrite §9 item 1 and P2.1 with the tube hypothesis
made explicit and tested on the benchmark's own schemes, (b) replace the frozen staircase, (c) correct Propositions
1.3 and 4.5, and (d) either compute Theorem 1 in log variables on a benchmark box or demote it.
