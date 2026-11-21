# Review of the theory memo, round 3 of 3

- **Reviewer:** Claude Fable 5.2, 2026-11-21, on `theory/MEMO.md` at commit `507c209` (revision 2).
- **Read:** `MEMO.md` in full; `REVIEWS.md`; `reviews/round-2.md` (verdict, audit, D1–D3); `checks/common.py`,
  `checks/t2_slow_surrogate.py` and every `checks/data/*.json` that the memo quotes in T2, T5, T6, T7 and §8;
  `reviews/scratch-round-2/{a_tube_robertson.json, b_staircase.py}`; `src/problems/ode/{system, chemistry, step_maps}.py`;
  `LITERATURE.md` by search (T6 entries).
- **Ran** (in `theory/reviews/scratch-round-3/`, NumPy and SciPy, each under three minutes; logs and JSON beside them):
  `a_hires_chart.py` (Theorem 8 on HIRES with learned charts against full-coordinate surrogates),
  `a2_hires_followup.py` (degrees, the exact dynamics' amplification, local dimension of the visited set),
  `b_borsuk_ulam.py` (Proposition 6.5's bound against the best one-coordinate fit found, Robertson).
- **Not done:** T1, T3, T4 (except Proposition 4.1), T5 and T7 were not re-derived (rounds 1 and 2 checked them and
  revision 2 did not change their proofs); no comparison with ChebTT's HIRES rollouts in the benchmark records (I did
  not find the records' rollout fields in the time available); POLLU; the references under "Literature" are from memory
  and marked so.

**Verdict in one paragraph.** The proofs in revision 2 are correct as far as they go, and Proposition 6.5 is
quantitatively sharp (test B). But the new top-ranked foundation proves less than the memo says it does, and its
one check passes for reasons the theorem does not contain. Theorem 8 with Proposition 2.4(c) bounds the chain's
distance to $S$, not the rollout error $\lVert y_n - x_n\rVert$, which is the stated metric (a constant map satisfies
it). On HIRES (three-dimensional visited set, several invariant leaves, learned charts) the slow-coordinate surrogate
is stable only when the chart has exactly the dimension of the *visited set* (3) and a moderate degree; with the 4 or
5 coordinates that the staircase of item 2 prescribes it diverges to the clip within ten steps, although its one-step
error is $10^{-5}$ to $10^{-6}$ dex (test A). Where it is stable, its error grows 90- to 120-fold over 169 steps, which
is the exact dynamics' own amplification (52 to 61), so "does not accumulate" is a property of Robertson, not of the
construction. The idea survives in a narrower and more useful form: every full-coordinate fit diverged, every
3-coordinate fit of degree 3 to 6 did not. Items 1 and 2 become a foundation with five changes (last section), the
main one being a rollout theorem that needs no Fenichel theory and whose constants are computable from the surrogate.

---

## Audit of round-2 resolutions

Resolutions not listed (D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, P4, P5, P6) are correct and sufficient.

| # | Resolution in revision 2 | Audit |
|---|---|---|
| D1 | Proposition 2.4 restated with the normal gain; `t2_slow_surrogate`; Theorem 8; P2.1 conditional; audits on chain states | The restatement is right, and the new check is a fair refutation of "accuracy on visited states suffices". But the resolution replaces one unverified hypothesis (accuracy on the tube) by another (the image of $\hat G$ lies near $S$ on all of $\hat\pi(U_r)$), verified on one system where it holds for a structural reason (one curve, one chart coordinate). On HIRES the same recipe fails for $k = 4, 5$ (E2). Partly cosmetic: the normal gain of the check's surrogates is asserted, not computed (E9). |
| D2 | Invariants: stated in Proposition 2.4 and Theorem 8 | Stated, not tested, and one of the two options is insufficient: a chart that *includes* the invariant as an input coordinate does not make $\Psi$ conserve it. On HIRES the chart (invariant, two Jacobian directions) has the best one-step error of the 3-coordinate charts ($2.9\times10^{-4}$) and is the only one of them that diverges (invariant drift 38 dex, test A). Only $C\Psi = C$ (or an output that is projected onto the leaf) closes the neutral direction (E5). |
| D3 | Proposition 6.7: constant $J$ proved, (b) a sketch citing Levinson | Labels are right. The citation is not: Levinson's theorem is asymptotic in $t$; a finite-$\tau$ statement is roughness of exponential dichotomies, with a smallness condition against the gap (E8). P6.1 and §9 still read (b) as established. |
| P1 | Case (b) of Proposition 2.4 | Correct (I rechecked: $\operatorname{dist}(\Psi(p), S) \le \operatorname{dist}(\Phi(p), S) + \lVert\delta(q)\rVert + \lVert\delta(p) - \delta(q)\rVert$). Its constant $\ell$ involves $\Phi$ off $S$, so it is not checkable from data on $S$ (E6). |
| P2 | Proposition 6.5 | Proof correct (rechecked line by line, E7); numerically sharp (test B). The sentence drawn from it ("exactly 2 coordinates for every tolerance between the two") overstates it and contradicts the memo's own 1-coordinate surrogate unless the set is named (E3). |
| P3 | Proposition 6.7(b) | See D3. |
| Lit. | Cited from the reviewer's memory | Acceptable for an internal round if flagged, and it is flagged in `REVIEWS.md` but not in the memo's "Prior art" paragraphs (only "from memory" for DeVore–Howard–Micchelli and "unverified" for two others). Before the external review each must be read or marked in the memo. |

---

## Critiques

### E1 (critical). Theorem 8 and Proposition 2.4 bound the distance to $S$, not the rollout error

*Location.* Proposition 2.4; Theorem 8 ("normally stable in a rollout"); §8 Check ("rolls out with no accumulation");
§9 row U2 ("slow-coordinate surrogates do not accumulate"); §10 item 1.

*Problem.* The conclusion of Proposition 2.4 is $\operatorname{dist}(y_n, S) \le \ldots$. The memo's metric is the
maximum pointwise error of a rollout, $\lVert y_n - x_n\rVert$. The constant map $\Psi \equiv s^* \in S$ satisfies
case (c) with $\varepsilon_S = 0$. Nothing in items 1 and 2 bounds the error *along* $S$: Proposition 2.1 would, but it
is stated for the full map with a log norm that is $\ge 0$ with invariants, and Proposition 2.2 is a representation, not
a bound. So the foundation as written carries no rollout error guarantee, which is the one a scheme must have.

*Evidence.* (i) In `t2_slow_surrogate.json` the rollout error of S equals its one-step error to every digit
($5.198866581643813\times10^{-4}$): the maximum is at step 1 and later errors are smaller, because Robertson's flow
contracts along its slow curve. That is a property of the problem. (ii) Test A2 (HIRES, 3 PCA coordinates): one-step
$1.8\times10^{-2}$, $6.6\times10^{-3}$, $6.4\times10^{-4}$ at degrees 3, 4, 6; rollout after 169 steps 2.15, 1.03,
$5.7\times10^{-2}$: growth 119, 157, 89. The exact
dynamics amplify a $10^{-4}$ relative change of an initial parameter by 52 (Pr) and 61 (E) over the same 169 steps
(1.2 at step 10, 2.7–2.9 at step 40). So the surrogate accumulates at about the rate the dynamics dictate, $\varepsilon A_n$,
which no surrogate can avoid, and which Theorem 8 does not state.

*Fix.* Replace the second half of Theorem 8 by a rollout theorem (proposal T8′ under "Minimal changes"): the chain of
$\Psi = \hat G\circ\hat\pi$ is exactly a $k$-dimensional iteration $u_{n+1} = \hat\pi(\hat G(u_n))$, so the error is
$\varepsilon_G(1 + L_GL_\pi A_{n-1})$ with $A_n$ the amplification of the *reduced* map. State "no accumulation" as the
case $A_n$ bounded (Robertson), and quote HIRES as the case where it is not.

### E2 (critical). "Fitted on the slow set alone, in $k$ coordinates" is not stable unless the data fill the chart; with the staircase's $k$ it diverges

*Location.* Theorem 8 ("if $\hat G$ maps into the $\varepsilon_S$-neighbourhood of $S$"); the sentence after its proof;
§10 item 1 ("Fit $\hat G$ on the slow set in $k$ coordinates"); item 2 ("it tells item 1 what $k$ is"); open item 1.

*Problem.* The hypothesis of case (c) is on $\hat G$ over $\hat\pi(U_r)$, an open set of $\mathbb{R}^k$. A fit on visited
states constrains $\hat G$ on $\hat\pi(\text{visited set})$. The two agree only if the visited set has dimension $k$
and is sampled at the resolution of $\hat G$. When the chart has more coordinates than the visited set has dimensions,
$\hat G$ is unconstrained across the data, exactly as the full-coordinate V was in round 2, and the chain leaves in
that direction. The theorem is not false (its hypothesis fails), but the practical claim drawn from it is.

*Evidence (test A; HIRES, log10, $\Delta t = 2$, burn 5, 170 steps, 100 training and 40 held-out trajectories, all fits on
visited states only; errors in dex, rollout capped at 40 by the clip).*

| surrogate | features | one-step, visited | tube $\pm0.05$ | rollout 10 | rollout 40 | rollout 169 | $\rho(D\Psi)$ max |
|---|---|---|---|---|---|---|---|
| PCA-2 (degree 8) | 45 | 0.48 | 0.46 | 0.52 | 0.72 | 2.95 | 1.86 |
| PCA-3 (degree 6) | 84 | $6.4\times10^{-4}$ | 0.51 | $8.8\times10^{-4}$ | $1.8\times10^{-3}$ | $5.7\times10^{-2}$ | 1.20 |
| JAC-3 (degree 6) | 84 | $5.4\times10^{-4}$ | 0.59 | $7.0\times10^{-4}$ | $1.3\times10^{-3}$ | $5.4\times10^{-2}$ | 1.07 |
| INV-3 (degree 6) | 84 | $2.9\times10^{-4}$ | 1.45 | $5.3\times10^{-4}$ | $8.5\times10^{-4}$ | 38 (diverges) | 4.67 |
| PCA-4, JAC-4, INV-4 (degree 5) | 126 | $6\times10^{-6}$–$2.7\times10^{-5}$ | 28–65 | 40 (diverges) | 40 | 40 | 46–297 |
| PCA-5, JAC-5, INV-5 (degree 4) | 126 | $4.7\times10^{-6}$–$2.1\times10^{-5}$ | 20–55 | 40 (diverges) | 40 | 40 | 8–248 |
| full, degree 2 (next state or increment) | 45 | $6.6\times10^{-4}$ | 184 | 40 (diverges) | 40 | 40 | 46 |
| full, degree 3 (next state or increment) | 165 | $4.6\times10^{-6}$ | 1,290 | 40 (diverges) | 40 | 40 | 234 |

PCA: principal directions of the visited states. JAC: leading eigenvectors of the mean of $D\Phi^{\mathsf T}D\Phi$ over
150 visited states. INV: $\log_{10}(y_7 + y_8)$ and JAC directions. The exact map has $\rho(D\Phi) \le 1.11$,
$\lVert D\Phi\rVert \le 1.85$. Follow-up (test A2): PCA-4 diverges already at degree 3 (35 features; rollout 0.23 at step
10, 9.5 at step 169) and only the nearly linear degree 2 stays bounded (one-step $3\times10^{-2}$); PCA-3 at degree 8
(165 features, one-step $5.7\times10^{-5}$, the best of all) diverges at step 169 ($\rho(D\Psi) = 2.5$). The visited
set's local PCA (40 neighbours) has singular values $1, 0.76, 1.6\times10^{-3}, 1.8\times10^{-4}$: it is thin even in
its third direction at the sampling of 100 trajectories.

So: (1) every full-coordinate surrogate diverges and every 3-coordinate chart of degree 3 to 6 without the invariant is
stable, which **supports** the idea of item 1; (2) the stable window is "chart dimension = dimension of the visited
set, degree below the data's resolution across it", which is **not** what Theorem 8 or §10 states; (3) the dimension
that item 2 delivers is the wrong one here: the staircase at $\tau = 2$ (median singular values of $D\Phi$ at visited
states $1.8, 1.2, 0.40, 2.8\times10^{-2}, 7.4\times10^{-3}, 3.9\times10^{-8}$) says 5 coordinates above $10^{-3}$ and 4
above $10^{-2}$, and those charts diverge.

*Is `t2_slow_surrogate` a fair test?* It is fair against "accuracy on visited states suffices" (V and V-full could have
been stable and were not). It is not a test of Theorem 8's practical claim, because it passes for three reasons
outside the theorem: the visited set is one curve and the chart has one coordinate (the data fill the chart), every
state is on one leaf (no neutral direction is excited), and the flow contracts along the curve (E1). HIRES removes all
three and the claim holds only in the window above.

*Fix.* State the hypothesis as a property of the fitted model that can be computed: the Lipschitz constant (or the
Jacobian's products along chains) of the reduced iteration $\hat g = \hat\pi\circ\hat G$ on a $\rho$-neighbourhood of
$\hat\pi(\text{data})$ (T8′). Say that it holds by construction only when the data fill an open set of the chart, and
give the two ways to get there: choose $k$ as the dimension of the data (the smallest $k$ at which a function of
$\hat\pi$ fits the next state, a test of injectivity), or fill $S$ (relaxed perturbations of visited states,
constrained runs; see Literature) when the application needs the larger $k$.

### E3 (major). The staircase's dimension and the surrogate's dimension are different numbers; "exactly 2 coordinates" contradicts the memo's own 1-coordinate surrogate

*Location.* Proposition 6.5's last paragraph; P6.1; §9 row U3; §10 item 2.

*Problem.* Proposition 6.5 is a statement on a full-dimensional ball $B(p, r) \subset \mathbb{R}^d$. The singular
values it uses count every non-contracted direction: the slow directions within a leaf, the invariants, and slow
directions that the family of initial states never excites. The surrogate of item 1 lives on the visited set (or on
one leaf). The memo joins them with "it tells item 1 what $k$ is" and "Robertson … exactly 2 coordinates for every
tolerance between the two", while its flagship surrogate S uses one coordinate at $5\times10^{-4}$.

*Evidence (test B; Robertson at a visited state, $\tau = 1$).* $\sigma(D\Phi) = 1.51, 0.99, 1.2\times10^{-12}$;
restricted to the tangent plane of the leaf: $1.38, 1.3\times10^{-11}$. On the ball of radius 0.05 (0.2) dex the best
single linear coordinate found errs 0.046 (0.180), $z_1$ errs 0.056 (0.232), a constant 0.074 (0.290); on the leaf
through $p$ the same $z_1$ errs at the resolution of the test's slabs only. So the count 2 is "one slow coordinate and
the mass", and it is the right count on a ball and the wrong one on the leaf. HIRES (E2): staircase 4–5, visited set 3.

*Fix.* Define two numbers and keep them apart: $m_{\mathrm{ball}}(\eta)$, the staircase (what a surrogate on a box or
on kicked states needs), and $m_M(\eta) = \#\{\sigma_i(D\Phi(p)|_{T_pM}) > \eta\} \le \dim M$ for the set $M$ the
surrogate is asked to cover. Restate Proposition 6.5 on a $C^2$ submanifold $M$ (Borsuk–Ulam on a sphere in a chart of
$M$; $K_2$ then includes the chart's second derivative). Replace "exactly 2 coordinates for every tolerance between
the two" by the range the proposition gives (E7), on a ball.

### E4 (major). The first half of Theorem 8 (accuracy on the neighbourhood) is unsupported by either test, and it is the half the application needs

*Location.* Theorem 8's display; "is accurate on the whole neighbourhood (to the accuracy of its coordinates)"; §10
item 1 ("the only item whose hypothesis and conclusion were both measured on a stiff problem and held").

*Problem.* The bound has the term $\mathrm{Lip}(\hat G)\sup_N\lVert\hat\pi - \chi\circ\pi\rVert$: the chart must be
constant along the stable fibres. No check measures it, and no chart tried satisfies it. On Robertson S errs 0.60 dex on
states displaced by 0.05 dex (the memo reports this and reads it as "accuracy on the tube is not necessary"); on HIRES
the 3-coordinate charts err 0.5–0.6 dex on the same tube, including JAC, whose kernel is the fibre direction to first
order where $D\Phi$ has a gap (it has none after the third value: 0.40, 0.028, 0.0074). In an operator-split
simulation the other operators displace the state off the visited set before every chemistry step, so the chain's
inputs are in $N$, not on $S$: there the first half is the guarantee, and the measured constant is an amplification of
ten.

Two further points on the check's tube: it displaces all three log coordinates, so it leaves the leaf (a change of the
total mass of up to 12%), and the column mixes the neutral direction with the fast one; test B shows that S's 0.60 is
the mass direction (a function of $z_1$ is as bad as a constant across leaves). And the hypothesis "measured" in §10
was not: neither the distance of $\hat G$'s image to $S$ nor the normal gain of the four fitted surrogates is in
`t2_slow_surrogate.json`; the quoted $|\partial\Psi/\partial z_2| = 6.3$ belongs to round 2's fit (one-step
$3.7\times10^{-9}$, rollout 10.4), not to the check's V ($2.9\times10^{-10}$, 40).

*Fix.* Split Theorem 8 into (i) the rollout statement for chains that start on $S$ (T8′, no fibres needed) and (ii)
the accuracy statement on $N$, marked as untested with the fibre projection as its open hypothesis. Add to the check
the two measured quantities (image-to-$S$ distance, $\rho(D\hat g)$) and a tube within the leaf. Reword §10 item 1.

### E5 (major). Theorem 8's hypotheses: the chart, the boundary, the leaves

*Location.* Theorem 8; Assumption (S); Proposition 2.4, "Invariants".

1. *A global chart need not exist.* $\chi: S \to \mathbb{R}^k$ with $k = \dim S$ exists only if $S$ embeds in
   $\mathbb{R}^k$. The benchmark's stiff suite has limit cycles (Oregonator, van der Pol, the Brusselator): $S$ a circle
   has no chart in one coordinate. The proof only needs $\chi$ injective with a Lipschitz inverse on $\chi(S)$, so
   "chart" should read "embedding into $\mathbb{R}^{k'}$, $k' \ge k$"; then $\chi(S)$ has empty interior in
   $\mathbb{R}^{k'}$ and E2's problem (data do not fill the chart) is unavoidable and must be handled by the
   Lipschitz condition of T8′, or by an atlas.
2. *Where $\hat G$ is assumed good.* "If $\hat G$ maps into the $\varepsilon_S$-neighbourhood of $S$" must say on which
   set: $\hat\pi(U_r)$. At the boundary of $\chi(S)$ the chain's coordinate can leave $\chi(S)$ (in test A up to 17% of
   INV-3's chain states leave the fitted box before it diverges). Needed: $\Gamma_\tau(S)$ inside $S$ with a margin
   larger than the reduced error, or the hypothesis on a neighbourhood.
3. *Invariants.* (S) asks $S$ to attract in every normal direction, so with $q$ invariants $S$ spans the leaves and has
   dimension $k_{\mathrm{leaf}} + q$; the check uses its slice by one leaf and a chart of dimension $k_{\mathrm{leaf}}$.
   That is legitimate only if $\Psi$ maps the leaf to itself. A polynomial $\hat G$ in log coordinates does not: on HIRES
   the invariant drifts by $2.8\times10^{-3}$ (PCA-3) and $3.9\times10^{-3}$ dex (JAC-3) over 169 steps, and along a
   neutral direction nothing damps it. "The chart must include them as coordinates, **or** the surrogate must conserve
   them" is wrong in its first branch (audit, D2): INV-3 includes it and diverges. Conservation of the output is the
   requirement; the chart coordinate is needed in addition when the data span several leaves.
4. *The constant.* In (S), $\lVert\Phi_t(p) - \Phi_t(\pi(p))\rVert \le Ce^{-\gamma t}$ has $C$ proportional to
   $\operatorname{dist}(p, S)$; written so, Theorem 8's last term is $C\operatorname{dist}(p, S)e^{-\gamma\tau}$ and
   vanishes on $S$, which the rollout statement uses.
5. *The legend.* "Proved, given (S)" is not one of the four status labels; (S) itself carries the caveat "theorem
   numbers to be checked". Until they are, Theorem 8 is "proved, citing" a result whose applicable form is not pinned.

### E6 (major). Proposition 2.4 is one line, and none of its three cases is checkable from what a scheme has

*Location.* Proposition 2.4; T2 Prior art ("the surrogate-specific content is the normal-gain condition").

*Problem.* The proof is "Iterate", correctly. The content is in the three sufficient cases, whose proofs are not
written (they are short and correct: (a) and (b) need $\Phi(S) \subset S$, which the proposition's proof does not
otherwise use; (b) needs $q$ to be a nearest point and $\kappa + \ell < 1$). What a mathematician will ask is how any
case is verified: (a) needs sup-norm accuracy on a $d$-dimensional tube (the cost the memo wants to avoid); (b) needs
$\ell$, a Lipschitz constant of $\Psi - \Phi$ across $S$, hence $\Phi$'s normal derivative; (c) needs
$\operatorname{dist}(\hat G(u), S)$ for $u$ off the data, hence $S$ off the data. None is computable from the surrogate
and samples on $S$. T8′ is: its constants are $\varepsilon_G$ on the data and Lipschitz constants of the fitted model.

*Fix.* Write the three proofs (three lines each). Present Proposition 2.4 as the organising statement and T8′ as the
one a scheme checks. Drop "above all" for case (c) unless its hypothesis is replaced by a computable one.

### E7 (major for the "two-sided" claim, minor for the proposition). Proposition 6.5

*Proof.* Correct. The sphere of radius $r$ in the $m$-dimensional span $E$ has dimension $m - 1$ and
$v \mapsto c(p + v)$ maps it continuously to $\mathbb{R}^{m-1}$, so Borsuk–Ulam applies; Taylor at $p$ gives
$\lVert\Phi(p + v) - \Phi(p - v)\rVert \ge 2\lVert D\Phi(p)v\rVert - K_2r^2$ and $\lVert D\Phi(p)v\rVert \ge \sigma_mr$ on $E$.
The remainder only needs $\sup\lVert D^2\Phi(\xi)[v, v]\rVert$, the quadratic form, on the two radii.

*Numerical prediction (test B).* Robertson, $\sigma_2 = 0.993$, $K_2 \approx 1.0$–1.1 (sampled second differences) on
balls of 0.05 and 0.2 dex: bound 0.0484 and 0.177; best one-coordinate fit found 0.0456 and 0.180. The bound is
attained within the test's sampling (the 6% deficit at $r = 0.05$ is the sampled ball not reaching its sphere and the
slabs' diameters being sampled). **Verdict: the proposition predicts the error of the best $(m-1)$-coordinate fit to
within a few per cent here.**

*What the text claims beyond it.*
1. "Fewer than $m$ continuous coordinates, **linear or not**" is right; it should add that discontinuous encoders are
   not covered (a tree's leaf index, a piecewise chart), since ART and SlowTable are such schemes. Within one cell
   containing a ball the bound applies again.
2. Norms: the bound is in the 2-norm of the output; in the memo's metric (maximum over components) it is divided by
   at most $\sqrt{d}$. State the norm.
3. Range: the bound is $\max_{r\le r_0}(sr - K_2r^2/2) \le s^2/(2K_2)$, with $r_0$ the radius of the largest ball in
   the domain. "Exactly 2 coordinates for every tolerance between the two [singular values]" should read: no fewer
   than 2 for tolerances below $\min(sr_0 - K_2r_0^2/2,\ s^2/(2K_2))$ on a ball (about 0.45 dex here), and 2 suffice
   down to $Ce^{-\gamma\tau}\operatorname{diam}$ **under (S)**.
4. "Two-sided": the lower bound is from the singular values at one point; the upper bound is from (S), a global
   geometric hypothesis that the singular values do not certify. No statement "if $\sigma_{k+1}(D\Phi) \le \eta$ on a
   convex set then $k$ coordinates suffice to $O(\eta\operatorname{diam})$" is in the memo, and it is not obvious
   (level sets of an approximately rank-$k$ map need not be the fibres of one global chart). The staircase computed
   from a solver's Jacobians therefore gives the lower count rigorously and the upper count heuristically. Say so in
   P6.1 and §10 item 2, or prove the missing implication for a class.

### E8 (major). Proposition 6.7(b): the cited theorem is not the one needed, and (a)'s proved bracket is much wider than the agreement quoted

*Location.* Proposition 6.7; P6.1; §9 row U3; §10 item 2.

*Problem.* Levinson's theorem (Coppel; Eastham) gives the asymptotics of solutions of $u' = (\Lambda(t) + R(t))u$ as
$t \to \infty$ for $R \in L^1$ under a dichotomy condition on $\Lambda$. At a finite $\tau$ "$(1 + o(1))$" has no
meaning. What (b) needs is roughness of exponential dichotomies (Coppel 1978, ch. 4): if $u' = \Lambda(t)u$ has a
dichotomy with constants $(K, \alpha)$ between the two groups and $\sup\lVert V^{-1}V'\rVert < \alpha/(4K^2)$, the
perturbed system has one with nearby constants, which brackets the *groups* of singular values by factors depending on
$K$, not each $\sigma_i$ by $1 + o(1)$. That is a pointwise smallness against the gap, not smallness of
$\int\lVert V^{-1}V'\rVert$. The label "sketch" is honest; the content of the sketch is not yet a correct statement.

For (a), the proved bracket is a factor $\kappa(V)$ each way: with $\kappa(V) = 99$ to $6.9\times10^3$ (round 2, POLLU)
it is 2 to 4 decades wide in $\eta$, so "24 of 24" is an observation far inside what (a) proves, on states where the
thresholds $10^{-3}$, $10^{-6}$ happen to sit in spectral gaps.

*Fix.* A one-sided statement that is provable now (proposal P7 below) for the product of the leading singular values;
(b) restated with the dichotomy hypothesis and labelled conjecture for individual values; P6.1 and §9 "explains U3"
reduced to "(a) proves it at constant $J$ within $\kappa(V)$; the path average matches within 3–15% in 12 cases".

### E9 (minor). Consistency, numbering, numbers, labels

1. T3 "Scope" cites "the effective dimension (Theorem 6.5)": 6.5 is now the Borsuk–Ulam proposition; it means Theorem
   6.6 with Proposition 6.7.
2. `REVIEWS.md`, round 1 table, uses the numbering of revision 1 ("Remark 1.5", "Theorem 6.5", "Proposition 6.6", "§8",
   "§9 item 1") without saying so; round 2's row D1 says "P2.3", which no longer exists. Add one line: "numbers are those
   of the revision under review".
3. `theory/README.md`: "seven candidate theorems T1–T7" (Theorem 8 missing), "each under two minutes" (memo: three),
   "external reviews" (also in the memo's first paragraph: the three rounds are by the same model family and internal;
   the external one is still to come).
4. Corollaries (6.1)–(6.4) are numbered inside Theorem 6's paragraph and then Proposition 6.5 continues the count; the
   Lean list names "Proposition 2.4 … and Theorem 8" as self-contained, but Theorem 8 needs (S).
5. Numbers against JSON: all that I compared agree (T2 sanity checks, T5's five rows, T7, the `t2_slow_surrogate`
   table) except T7's vertex error at $d = 50$: 1,028.5, printed as 1,030. "40 (diverges)" is the clip (39.88, 39.64):
   say "reaches the clip at $-40$".
6. `t2_slow_surrogate`'s paragraph states $\kappa_\Psi \approx 0$ (S) and $\kappa_\Psi \approx 1$ (S-incr) as findings;
   they are inferences. S-incr's rollout is 3.3 times its one-step error, which is growth, not "persist without
   decaying".
7. §9, row U6, column Thm 8: "shared chart for all outputs (stiff case)" is untested; row U2: add "on Robertson".
8. §0 says "Observations no theory in the repository explains yet" and §9 answers U3 with "**explains**": at
   $\tau = 1$ on a ball, by (a) only within $\kappa(V)$ (E8).

### E10 (minor). Proposition 4.1 on chain states

The proposition needs fresh i.i.d. points independent of the model. States along one chain are neither independent of
each other nor drawn from a fixed $\mu$ (their law depends on $s_r$). The statement that is true and useful: draw $N_r$
fresh *initial states*, roll the accepted model out on each, and accept iff every chain stays within the tolerance for
$n$ steps; then with probability $1 - \alpha$ the measure of initial states whose $n$-step rollout violates is at most
$\eta$. That is a probabilistic certificate of the rollout itself, the metric of the project, with no dynamical
hypothesis; it costs $N_rn$ oracle steps. §10 item 1 should name it as the audit.

---

## Own hypothesis tests

**A. `a_hires_chart.py`, `a2_hires_followup.py` (Theorem 8's practical claim).** Design, numbers and table under E2;
amplification and degrees under E1. Summary of verdicts:

| claim | verdict |
|---|---|
| A full-coordinate fit on visited states diverges in a rollout (round 2, D1) | **confirmed** on HIRES: 4 of 4, at one-step errors $4.6\times10^{-6}$ to $6.6\times10^{-4}$ |
| A slow-coordinate fit on visited states, next-state form, is stable (Theorem 8's claim) | **holds only** for $k = 3$ = the visited set's dimension and degrees 3–6 (4 of 4 without the invariant coordinate); **fails** for $k = 4, 5$ (6 of 6 diverge within 10 steps), for $k = 3$ at degree 8, and for the chart with the invariant |
| It "does not accumulate" | **refuted** as a general statement: growth 89–157 over 169 steps, the dynamics' own 52–61 |
| The staircase gives the chart's dimension (§10 item 2) | **refuted** on HIRES: staircase 4–5, stable charts 3 |
| The surrogate is accurate on the neighbourhood (first half of Theorem 8) | **not supported**: 0.41–0.59 dex at 0.05 dex off the visited set, for every chart |
| A chart learned from the Jacobian's dominant right singular subspace works | as well as PCA at $k = 3$ (rollout $5.4\times10^{-2}$ against $5.7\times10^{-2}$), no better off the visited set |

The test could have refuted the idea outright (if the 3-coordinate charts had diverged like the others) and did not.

**B. `b_borsuk_ulam.py` (Proposition 6.5).** Numbers under E3 and E7. Verdict: **confirmed and sharp** on a ball
(bound 0.0484, 0.177; best fit found 0.0456, 0.180); on the leaf one coordinate suffices, as $\sigma(D\Phi|_{\text{leaf}}) = (1.38, 10^{-11})$ says.

Limits of these tests: one system each; polynomial least squares with features clipped at 1.5 times the fitted range
(a different regulariser moves the degree at which a chart turns unstable, not the mechanism); $K_2$ and the slab errors
are sampled estimates; HIRES at 100 trajectories samples its third direction thinly.

---

## Literature

All from memory; none re-read for this review. Mark them so until checked.

- **The scheme of Theorem 8 is established practice in combustion.** Tabulating the state on a low-dimensional
  manifold as a function of $k$ progress variables and advancing only those is ILDM (Maas, Pope, Combust. Flame 88,
  1992), flamelet-generated manifolds (van Oijen, de Goey, Combust. Sci. Technol. 161, 2000), FPI (Gicquel, Darabiha,
  Thévenin, Proc. Combust. Inst. 28, 2000) and REDIM (Bykov, Maas, Combust. Theory Model. 11, 2007). The closest to
  $\Psi = \hat G\circ\hat\pi$ with a tabulated step map is ISAT in a reduced representation with species
  reconstruction: Tang, Pope, *Implementation of combustion chemistry by in situ adaptive tabulation of rate-controlled
  constrained equilibrium manifolds*, Proc. Combust. Inst. 29 (2002); Ren, Pope, Vladimirsky, Guckenheimer, *The
  invariant constrained equilibrium edge preimage curve method*, J. Chem. Phys. 124 (2006) (ICE-PIC: reconstruction on an
  invariant manifold from reduced coordinates); Hiremath, Ren, Pope, *Combined dimension reduction and tabulation
  strategy using ISAT–RCCE–GALI*, Combust. Flame 158 (2011). Their error control is empirical (ISAT's ellipsoids of
  accuracy bound the one-step retrieval error: Pope 1997; Liu, Pope, Combust. Theory Model. 9, 2005; Lu, Pope, J.
  Comput. Phys. 228, 2009); I know of no rollout theorem there.
- **How to obtain points on $S$ and the lift $\chi^{-1}$.** Gear, Kaper, Kevrekidis, Zagaris, *Projecting to a slow
  manifold: singularly perturbed systems and legacy codes*, SIAM J. Appl. Dyn. Syst. 4 (2005) (constrained runs);
  Zagaris, Gear, Kaper, Kevrekidis 2009 (TH-51, already listed: accuracy $O(\varepsilon^m)$ of the $m$-th scheme);
  Kevrekidis et al., *Equation-free, coarse-grained multiscale computation*, Commun. Math. Sci. 1 (2003); Gear,
  Kevrekidis, *Projective methods for stiff differential equations*, SIAM J. Sci. Comput. 24 (2003). These are the tools
  that fill $S$ at a prescribed chart value (E2's second route) and they come with error analyses of the projection,
  not of a tabulated map.
- **Reduced dynamics on a learned manifold.** Cenedese, Axås, Bäuerlein, Avila, Haller, *Data-driven modeling and
  prediction of non-linearizable dynamics via spectral submanifolds*, Nat. Commun. 13 (2022): exactly "coordinates on
  an attracting invariant manifold, reduced map, parametrisation back", with the manifold's existence from
  Haller–Ponsioen (TH-54). Linot, Graham, *Deep learning to discover and predict dynamics on an inertial manifold*,
  Phys. Rev. E 101 (2020); Floryan, Graham, *Data-driven discovery of intrinsic dynamics*, Nat. Mach. Intell. 4 (2022)
  (charts and atlases, relevant to E5.1); Lee, Carlberg, *Model reduction of dynamical systems on nonlinear manifolds
  using deep convolutional autoencoders*, J. Comput. Phys. 404 (2020) (a posteriori error bounds for projection on a
  learned manifold); Champion, Lusch, Kutz, Brunton, PNAS 116 (2019); Vlachas et al., *Multiscale simulations of complex
  systems by learning their effective dynamics*, Nat. Mach. Intell. 4 (2022). For chemistry: Grassi et al., *Reducing
  the complexity of chemical networks via interpretable autoencoders*, A&A 668 (2022); Sulzer, Buck, *Speeding up
  astrochemical reaction networks with autoencoders and neural ODEs*, ApJ 959 (2023); Holdship et al. (Chemulator),
  A&A 653 (2021). These advance a latent state and decode it, which is case (c); none that I recall states the normal
  stability or a rollout bound as a theorem.
- **Flow-map learning with error bounds.** Qin, Wu, Xiu, *Data driven governing equations approximation using deep
  neural networks*, J. Comput. Phys. 395 (2019): one-step flow-map surrogates with the classical $\varepsilon\sum L^j$
  rollout bound in full coordinates (Proposition 2.1).
- **Approximate inertial manifolds.** Foias, Manley, Temam, *Modelling of the interaction of small and large eddies
  in two dimensional turbulent flows*, M2AN 22 (1988); Marion, Temam, *Nonlinear Galerkin methods*, SIAM J. Numer. Anal.
  26 (1989): reduced dynamics in $k$ coordinates with the rest slaved, with error estimates in terms of the distance
  to the manifold; the PDE precedent for T8′.
- **Proposition 2.4(c) itself.** I know of no source that states it; it is one line, and a referee will call it
  folklore. The statement worth claiming is T8′ with computable constants and E10's audit, as a combination; neither
  ingredient is new (Proposition 2.1 in a chart; Wilks).
- **Proposition 6.5.** DeVore, Howard, Micchelli 1989 is the right source for the technique (continuous nonlinear
  widths are bounded below by Bernstein widths through Borsuk–Ulam); the pointwise form with $\sigma_m$ and $K_2$ is a
  corollary of that argument, so "new as far as found" should be "an application".
- **Proposition 6.7(b).** Coppel 1978, ch. 4 (roughness) rather than Levinson; for slowly varying $J$ also the
  adiabatic estimates in Kreiss, *Difference methods for stiff ordinary differential equations*, SIAM J. Numer. Anal.
  15 (1978), which treat exactly $J(t)$ with a spectral gap and bounded $V^{-1}V'$.

---

## Minimal changes for a foundation

1. **T8′ (rollout theorem; replaces the second half of Theorem 8; elementary, no (S)).** Let $M$ be compact with
   $\Phi(M) \subset M$ (the set the scheme is trained on), $\hat\pi: \mathbb{R}^d \to \mathbb{R}^k$ $L_\pi$-Lipschitz and
   injective on $M$, $U = \hat\pi(M)$, $U_\rho$ its $\rho$-neighbourhood, $\hat G: U_\rho \to \mathbb{R}^d$ with
   $\varepsilon_G = \sup_{x\in M}\lVert\hat G(\hat\pi(x)) - \Phi(x)\rVert$, $L_G = \mathrm{Lip}(\hat G)$, and
   $\hat g = \hat\pi\circ\hat G$ with amplification $A_n = \sum_{j<n}\mathrm{Lip}(\hat g|_{U_\rho})^j$
   (or, to first order, the products of $D\hat g$ along the chain, Proposition 2.2 in $k$ dimensions). For $x_0 \in M$
   and $\Psi = \hat G\circ\hat\pi$: while $L_\pi\varepsilon_GA_{n-1} \le \rho$,
   $$\lVert y_n - x_n\rVert \le \varepsilon_G\,(1 + L_GL_\pi A_{n-1}).$$
   *Proof.* $u_n = \hat\pi(y_n)$, $v_n = \hat\pi(x_n) \in U$. $u_{n+1} = \hat g(u_n)$ and
   $\lVert\hat g(v_n) - v_{n+1}\rVert = \lVert\hat\pi(\hat G(\hat\pi x_n)) - \hat\pi(\Phi x_n)\rVert \le L_\pi\varepsilon_G$, so
   $a_n = \lVert u_n - v_n\rVert$ obeys $a_{n+1} \le \mathrm{Lip}(\hat g)a_n + L_\pi\varepsilon_G$ while $u_n \in U_\rho$;
   and $y_{n+1} - x_{n+1} = [\hat G(u_n) - \hat G(v_n)] + [\hat G(\hat\pi x_n) - \Phi(x_n)]$. $\square$
   Every constant is a property of the fitted model and of its residuals on the data. It contains E2 (a chart with
   more coordinates than the data has dimensions has $\mathrm{Lip}(\hat g|_{U_\rho}) \gg 1$: 46–297 measured), E1
   ($A_n$ is the price, about the dynamics' own amplification when the fit is good: 52–61 on HIRES, bounded on
   Robertson) and needs neither Fenichel nor a normal gain. Check it on both tests' numbers before adopting (I did
   not evaluate $\mathrm{Lip}(\hat g)$ on $U_\rho$, only $\rho(D\Psi)$ at visited states).
2. **Name the set.** $M$ is the visited set (or what the application will visit), not Fenichel's $S$; $k$ is the
   smallest dimension at which a function of $\hat\pi$ fits the next state on $M$ (HIRES: 3; Robertson: 1), bounded
   below by the restricted Proposition 6.5 (E3). The staircase is the dimension needed off $M$. If the application
   kicks states off $M$ (operator splitting), say that $M$ must then be filled to the staircase's dimension (relaxed
   perturbations, constrained runs), and that accuracy off $M$ is the first half of Theorem 8, open (E4).
3. **Conservation as a requirement** on the output ($C\Psi = C$ exactly, by construction or by projection), not "chart
   coordinate or conservation" (E5.3); include the invariants as chart coordinates when the data span leaves.
4. **The audit is on whole rollouts** from fresh initial states (E10): a probabilistic certificate of the metric
   itself, complementary to T8′'s deterministic but pessimistic $A_n$.
5. **Item 2 restated as what is proved:** Proposition 6.5 (with its norm, range and continuity hypothesis, on a ball
   and on a submanifold) is the lower count; Proposition 6.7(a) brackets the frozen count within $\kappa(V)$; the upper
   count holds under (S) and the path-averaged form is conjecture with evidence (E7, E8). One provable addition:
   **P7 (one-sided path-averaged bound).** If $J(x(t)) = V(t)\Lambda(t)V(t)^{-1}$ with $V \in C^1$, then
   $z = V^{-1}w$ solves $z' = (\Lambda - V^{-1}V')z$, the Euclidean log norm of the $m$-th additive compound of
   $\Lambda - V^{-1}V'$ is at most $\sum_{i\le m}\operatorname{Re}\lambda_{(i)} + m\lVert V^{-1}V'\rVert$, and so
   $$\prod_{i\le m}\sigma_i(D\Phi_\tau) \le \bigl(\lVert V(\tau)\rVert\,\lVert V(0)^{-1}\rVert\bigr)^m\exp\int_0^\tau\Bigl(\sum_{i\le m}\operatorname{Re}\lambda_{(i)}(t) + m\lVert V^{-1}V'\rVert\Bigr)dt$$
   (Theorem 6.6's argument in the moving eigenbasis; the sort is pointwise in $t$). It is the weight $W$ that open
   item T6 asks for, it bounds the staircase from above, and it does not give the lower bound on $\sigma_m$ whose
   absence is the unsafe direction (undercounting); that still needs the dichotomy. I have not tested it numerically.

Smaller: fix E9's cross-references and labels; write the proofs of cases (a) and (b); read or flag the references.

---

## Overall assessment

What would still make a careful mathematician reject the memo: (1) the headline theorem's conclusion is not the
quantity the memo sets out to control (E1), and its key hypothesis is unverifiable as stated and false for the
recipe the ranking recommends (E2, E6); (2) the phrase "two-sided effective dimension" joins a sharp lower bound at a
point with an upper bound under a different, uncertified hypothesis, and is then used for a set it does not speak
about (E3, E7); (3) a sketch resting on the wrong classical theorem carries an "explains" in the synthesis (E8);
(4) "hypothesis and conclusion both measured" is not true of the one check (E4).

What is solid: every proof I rechecked (Propositions 2.2, 2.4 with its cases, 4.1, 6.5, 6.7(a), Corollary 6.1,
Theorem 8's inequality) is correct; Proposition 6.5 is sharp in a test that could have embarrassed it; the round-2
finding that full-coordinate fits on visited states diverge is now confirmed on a second, harder system, and the
slow-coordinate remedy survives there in a precise window.

Is the foundation strong enough to build a scheme on? Not as written; yes after changes 1–4, which are small. The
scheme they describe is: choose the chart's dimension by the fit on the data, fit the next state in those coordinates
with conserved outputs, compute the reduced map's amplification from the fitted model for the deterministic bound,
and certify rollouts by fresh-trajectory audits. Its open mathematical problem, which the external review should be
asked about, is accuracy off the data (the fibre projection), because that is what an operator-split simulation
exercises.
