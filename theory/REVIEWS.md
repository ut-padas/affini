# Reviews of the theory memo

Each round: the reviewer and its brief, a summary of its critiques, and the resolution of each (accepted and changed,
partly accepted, or rejected with the reason). The full reviews are in `reviews/round-<n>.md`, their tests in
`reviews/scratch-round-<n>/`; `git log -- theory/` shows each revision of the memo. Statement and section numbers in each round's table are those of the revision under review.

## Round 1: Claude Fable 5.2 (medium effort), on the memo at commit `0502542`

Brief: check every proof, every check and every number; run independent hypothesis tests that could refute a theorem;
judge the assumptions against real stiff problems, the ranking, and the novelty claims. Verdict: the proofs of T1, T3,
T4, Proposition 5.1 and Lemma 7.2 are correct; several predictions were upper bounds read as laws, several checks could
not have failed, three theorems did not apply as stated to the problems they were meant to explain, and one prediction
(P5.1) is refuted. Five tests of its own (A–E).

| # | Severity | Critique | Resolution |
|---|---|---|---|
| C1 | critical | Corollary 1.2's regimes are upper bounds; they cannot explain U1, and the bound's regime differs from the dynamical class (Lorenz-96 at $F = 1$ has $\mu_\infty = 5$; Robertson's $\mu \ge 0.022$) | Accepted. Predictions restated as bounds; Corollary 1.4 called "the bound's regimes"; §8 says "consistent (upper bound only)"; Proposition 1.3 (a ceiling, from N2) added; the Brusselator test is open item 1 |
| C2 | major | The Lorenz-96 slope $-1.04$ is one seed; slopes $-0.8$ to $-1.1$ appear at dissipative forcings too | Accepted. The check now reports test A's table and says it does not discriminate; the $F = 0$ fit and the "neutral law" claim are withdrawn; the chaotic rate is compared with $\lambda_{\max}$ |
| C3 | major | T1 never evaluated on a stiff system; test B: valid 24/24, $\Lambda \le 1.75$ at $\tau|\lambda| = 42{,}000$, but only with optimised $\beta$, and degrees overestimated 4- to 43-fold | Accepted. Test B's results are in the memo; the choice of $\beta$ is stated as necessary; "does not shrink with $\tau$" removed |
| C4 | major | With a linear invariant $\mu(J) \ge 0$ in every norm: the dissipative regime is empty for closed networks | Accepted. Remark 1.5 (reduce to the invariant leaf); Proposition 2.1 notes it |
| C5 | minor | $\rho - 1 \approx \theta R/h$ (not $2\theta R/h$); the degree misses a $d\log$ term; symbol clashes; 1.4 in $\rho$ is 1.6 in degree; other norms allowed | Accepted (a)–(d); (e) noted through Lemma 1.2's componentwise radii |
| C6 | major | T2's checks are identities on synthetic defects; P2.1 is testable from the committed table and holds on five rows | Accepted. The table is now the check; the synthetic checks are labelled sanity checks |
| C7 | major | The fixed-point formula needs $I - P$ invertible (false with invariants); U2's rollouts are transients; ChebTT attains the classical bound | Accepted. Proposition 2.2's first display is the result; the fixed-point part is a remark for the reduced system; §8 updated |
| C8 | minor | Proposition 2.3's "$+o(n)$" is wrong; the omitted Davis–Skodje case (13.79); the $O(\varepsilon^2)$ constant; Corollary 2.4 first order and diagonalisable; 20.008 | Accepted, all five; the coordinate-direction case now motivates splitting along the fibres (rows of $V^{-1}$) |
| C9 | critical | No $d$-independent light cone on the method-of-lines ladder ($\ell = Dn^2$); test D: window radius 4, 9, 16, 31 for $n = 10..80$ | Accepted. T3's scope restricted to $\ell\Delta\tau = O(1)$; the ladder moved to Theorem 6.5 (N7, test E) |
| C10 | minor | $|W_r| = 3r + 1$ for Lorenz-96; "independent of $d$" only loosely supported | Accepted; the $r = 5$ values added |
| C11 | – | No defect in T4 | – |
| C12 | minor | Cite Sukharev/Bakhvalov/Novak 1988 for the Lipschitz lower bound and Wilks 1941 for one-round audits; the step from $(\rho, M)$ to a certificate is missing | Accepted; Proposition 4.5 (from N4) added |
| C13 | major | P5.1 reads the Lebesgue inequality backwards; test C: factors 0.5–1.6 (interior), 2–5.7 (vertices), not 20 | Accepted. P5.1 withdrawn; Proposition 5.2 (from N5) and U5 explained by the $L^2$ projection's vertex error |
| C14 | minor | An upper bound on $\mathcal{L}_N$ is free; "twice" holds for $d \ge 10$ only; two CDL constants circulate | Accepted; `t5_least_squares` reports $\sqrt{K/\lambda_{\min}}$; the constant is the one read in TH-40 |
| C15 | major | Theorem 6 is (S3) restated; (S3) on a box is stronger than Fenichel; two different $\varepsilon$ outside the standard form | Accepted. T6 restated coordinate-free with the caveats; Theorem 6 labelled as (S) at $t = \tau$; the two $\varepsilon$ distinguished |
| C16 | major | "Cost in $k$, not $d$" does not follow ($\xi$ is $d$-variate); T1 cannot be applied to $\Gamma_\tau$; U6 measured on a non-stiff problem | Accepted, all three |
| C17 | major | Both singular-value checks are Liouville's formula; test where $d - k \ge 2$ | Accepted. Labelled sanity checks; new check `t6_hires` (8 species): Proposition 6.6's count matches in 24 of 24 cases |
| C18 | major | T7's link to U4 is directional only; a measure-sense target conflicts with the max-error metric | Accepted. P7.1 "consistent in direction"; P7.2 restated as the sup norm on the reachable set; the direct test is open |
| C19 | minor | Prior art for T7: Caflisch–Morokoff–Owen 1997, Trefethen 2017 | Accepted |
| C20 | minor | 1,490 → 1,495; 12-fold → 10.5-fold; 20 → 20.008 | Accepted |
| C21 | major | The SlowTable reading was selective (cheaper in one of eight pairs) | Accepted; all three families quoted |
| C22 | major | The ranking should be rebuilt around accuracy on the attracting tube | Accepted after checking the argument: the tube framing (Proposition 2.4) is what the benchmark already does (visited states), keeps a sup norm, and explains U2; ranking revised |
| C23 | minor | §8 over-claims | Accepted; table rewritten |
| C24 | major | Carleman-linearization bounds are close prior art for T1's dissipative regime | Accepted; cited (abstracts only) and the novelty claim narrowed |
| C25 | minor | Rigorous integrators use log norms (CAPD) | Accepted as unverified prior art |
| C26 | minor | ISAT and ILDM already use the Jacobian's singular values | Accepted |
| C27 | minor | See C12, C19 | – |
| N1 | proposal | Componentwise tube lemma | Adopted as Lemma 1.2 (proof checked: the row condition is convex in $J$, the Dini derivative of the max is bounded by the active indices) |
| N2 | proposal | Cauchy converse | Adopted as Proposition 1.3, with the scope narrowed: it bounds certificates $(\rho, M)$, not the true degree |
| N3 | proposal | Attracting tube | Adopted as Proposition 2.4 |
| N4 | proposal | Analytic sampled certificate | Adopted as Proposition 4.5 (Bernstein–Walsh checked one variable at a time) |
| N5 | proposal | Pointwise least-squares identity | Adopted as Proposition 5.2 (identity proved; the variance a sketch) |
| N6 | proposal | Linear invariants | Adopted as Remarks 1.5 and 2.5 |
| N7 | proposal | Compound-matrix singular-value bound | Adopted as Theorem 6.5; `t6_hires` shows it needs a weighting $W$ on non-normal Jacobians. Added Proposition 6.6 (frozen Jacobian), which the HIRES check supports |

## Round 2: Claude Fable 5.2 (medium effort), on revision 1 (commit `907cf70`)

Brief: audit the round-1 resolutions, check every new statement, test the new top-ranked foundation and the frozen
staircase on strongly nonlinear problems, run open items. Verdict: revision 1's proofs are correct and bounds are called
bounds; three readings were wrong: the tube read as "the visited states" (refuted by a surrogate accurate to $4\times10^{-9}$
dex on visited states that diverges to 10 dex), Theorem 1 called "the source of degrees" (vacuous on the Brusselator), and
the frozen staircase extended beyond constant $J$ (fails on Robertson at long steps). Five tests (A–E); proposals P1–P6.

| # | Severity | Critique | Resolution |
|---|---|---|---|
| Audit | – | C3, C4, C6, C18, C22 partly cosmetic or read too readily; others sufficient | Addressed through D1–D7 below |
| D1 | critical | The tube is not the visited states; accuracy on visited states does not control a rollout (test A) | Accepted, and taken further: Proposition 2.4 restated with the surrogate's **normal gain** and three sufficient cases; new check `t2_slow_surrogate` shows a slow-coordinate surrogate fitted on visited states only is stable (case (c)), which became Theorem 8; P2.1 made conditional; audits on chain states (Proposition 4.1) |
| D2 | major | Invariants are neutral ($\kappa_{\mathrm{mass}} = 1$); $S$ must span the leaves or $\Psi$ must conserve | Accepted; Proposition 2.4 and Theorem 8 state it |
| D3 | major | Frozen staircase fails on Robertson at $\tau = 100$ (11–16 decades, undercount); path average fits within 3–15% | Accepted; Proposition 6.7 proved for constant $J$ with the two-sided count, path-averaged form (b) as a sketch (Levinson); test B and E quoted |
| D4 | minor | "$\Lambda$ grows with $\tau$" does not follow from an invariant; Corollary 1.4 needs $(pc\tau)^{-1/p} \le r_{\max}$ | Accepted (Remark 1.6, Corollary 1.4) |
| D5 | major | Lemma 1.2 does not beat Theorem 1 on reduced Robertson with entrywise majorants; per-trajectory radii help | Accepted; Lemma 1.2 stated with $r(t; x_0)$; test D quoted; the expectation withdrawn |
| D6 | major | Theorem 1 valid but vacuous on the Brusselator (degrees 1,272–32,236 against 2–4); ceiling static; U1 reproduced only by measurement | Accepted; T1 demoted to item 3 of §10; test C quoted under U1; Proposition 1.5 (P5) added for the direction |
| D7 | major | Theorem 1 evaluated only on small boxes in own variables; log coordinates untested | Accepted; scope stated in "Limits"; open item |
| D8 | major | Proposition 1.3 loose by a square root; pointwise version; exponent $\lambda\tau$ | Accepted (P4's ceiling adopted; its companion degree bound, weak on the Brusselator, not included) |
| D9 | minor | Proposition 2.3 bounded only to first order (drift $0.5q\varepsilon^2$ per step) | Accepted |
| D10 | minor | Fixed-point remark: constants, existence; never applies to Robertson | Accepted |
| D11 | major | Proposition 4.5 weaker than a one-line argument; wrong cost | Accepted; restated as P6 with $N \ge n$ and Theorem 1's $M$; cost corrected; marked elementary |
| D12 | minor | Proposition 5.2's variance exact form | Accepted; the variance is now exact for $G = I$ |
| D13 | minor | Corollary 6.4 remainder; Neumann eigenvalue bound; $\kappa(V)$ infimum; convexity in Proposition 2.2; P2.2 wording; U5 "three of four" | Accepted, all |
| Lit. | – | Kloeden–Lorenz, Stuart–Humphries, ISAT for the tube; Kamke–Müller for Lemma 1.2; Constantin–Foias, Douady–Oesterlé, Smith for the compound bound; CSP, G-Scheme for the count; Bernstein's converse; Brisebarre–Joldeş | Accepted and cited (from the reviewer's memory, not re-read); DeVore–Howard–Micchelli added for the Borsuk–Ulam technique (from memory) |
| P1 | proposal | Tube with accuracy on $S$ and a normal Lipschitz bound | Adopted as case (b) of Proposition 2.4 |
| P2 | proposal | Borsuk–Ulam lower bound | Adopted as Proposition 6.5 (proof checked: the sphere has dimension $m - 1$, the second-order remainder is at most $K_2r^2$) |
| P3 | proposal | Path-averaged staircase | Adopted as Proposition 6.7(b), a sketch |
| P4 | proposal | Pointwise Cauchy ceiling | Adopted as Proposition 1.3 |
| P5 | proposal | Composition monotonicity of existence domains | Adopted as Proposition 1.5 |
| P6 | proposal | Proposition 4.5 as it should read | Adopted |

## Round 3: Claude Fable 5.2 (medium effort), on revision 2 (commit `507c209`)

Brief: audit the round-2 resolutions; test Theorem 8 and Proposition 2.4 in depth, with learned charts, $k \ge 2$ and several
leaves; test Proposition 6.5; check the memo's consistency. Verdict: the proofs are correct and Proposition 6.5 is sharp, but
items 1–2 of the ranking were not yet a foundation: Theorem 8 bounded the distance to $S$, not the rollout error, and its
practical claim held only for charts of the data's dimension. Tests A, A2 (HIRES), B (Borsuk–Ulam on Robertson).

| # | Severity | Critique | Resolution |
|---|---|---|---|
| Audit | – | D1 partly cosmetic (one hypothesis replaced by another, verified on one structurally easy system); D2's "chart coordinate or conservation" wrong in its first branch; D3 cites Levinson where roughness of dichotomies is needed | Addressed in E2, E5, E8 |
| E1 | critical | Theorem 8 with Proposition 2.4 bounds the distance to $S$, not $\lVert y_n - x_n\rVert$; on HIRES stable surrogates grow 89–157 over 169 steps, about the dynamics' 52–61 | Accepted. Theorem 8 is now the review's T8′, a rollout bound with constants from the fitted model (proof checked line by line); evaluated on Robertson's S (valid, about 3,000 times pessimistic); "no accumulation" attributed to Robertson's contraction |
| E2 | critical | On HIRES with learned charts only $k = 3$ (the visited set's dimension) and degrees 3–6 are stable; the staircase's $k = 4, 5$ diverge | Accepted; quoted in §8; $k$ is chosen by the fit on $M$; the staircase is $m_{\mathrm{ball}}$, not the surrogate's $k$ |
| E3 | major | The staircase's dimension and the surrogate's are different numbers | Accepted: $m_{\mathrm{ball}}$ and $m_M$ defined at the head of T6 and used throughout |
| E4 | major | Accuracy on the neighbourhood (first half of Theorem 8) is unsupported; errors 0.4–0.6 dex off the visited set | Accepted: split off as Proposition 8.2, conditional on (S), untested, the main open problem; "hypothesis and conclusion both measured" removed |
| E5 | major | No global chart for limit cycles; the set where $\hat G$ must be good; invariants need output conservation; $C \propto \operatorname{dist}(p, S)$; status label | Accepted: chart = injective map into $\mathbb{R}^k$; $U_\rho$ named; conservation required; (S) with the distance factor; "Conditional" added to the legend |
| E6 | major | Proposition 2.4's cases unproved and uncheckable | Accepted: proofs of (a)–(c) written; Proposition 2.4 called the organising statement, Theorem 8 the checkable one |
| E7 | major | Proposition 6.5 correct and sharp; norm, range, continuity; "two-sided" overclaimed | Accepted: norm, range and discontinuous encoders stated; test B quoted; the upper count marked as holding under (S) only |
| E8 | major | Proposition 6.7(b) cites the wrong theorem; (a)'s bracket is 2–4 decades wide | Accepted: (a) with its bracket, (b) replaced by the review's provable one-sided bound P7, (c) a conjecture under roughness of dichotomies |
| E9 | minor | Cross-references, README, "external", numbering, 1,029, "reaches the clip", inferred gains, U2/U6 wording | Accepted, all |
| E10 | minor | Proposition 4.1 on chain states is not i.i.d.; audit whole rollouts from fresh initial states | Accepted: Corollary 4.1′ |
| Lit. | – | Combustion manifold methods (ILDM, FGM, FPI, REDIM, ISAT–RCCE, ICE-PIC), constrained runs, learned-manifold reduced dynamics, approximate inertial manifolds; Proposition 2.4(c) folklore; Proposition 6.5 an application of DeVore–Howard–Micchelli | Accepted, cited as "from memory" |
| T8′ | proposal | Rollout theorem with computable constants | Adopted as Theorem 8 |
| P7 | proposal | One-sided path-averaged bound | Adopted as Proposition 6.7(b) (proof checked: the moving-eigenbasis equation and Horn's product inequality) |

## Round 4 (external): Codex GPT-6.2 (`gpt-6.2-sol`, medium effort), on revision 3 (commit `ffa0df0`)

Brief: check every proved, cited and conditional statement; check the synthesis against the evidence; judge the foundation; list final
edits. Verdict: Theorem 8's inequality is sound with a closed tube and verified constants, but items 1–2 of the ranking needed changes before
they could support a worst-case guarantee or prescribe a coordinate count. Its own test: an exact stiff flow on an attracting circle (Example 6.8).
All findings were checked against the memo before being applied; the archived Robertson records were queried to confirm F9.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| F1 | critical | The rollout audit bounds the probability of ever accepting a model bad on more than measure $\eta$, not the worst case, and not conditionally on acceptance | Accepted: Proposition 4.1 restated with frozen candidates and the joint false-pass reading; the audit labelled distributional; the Lipschitz-plus-ball-mass route to a uniform bound added; §10 targets a deterministic worst-case bound |
| F2 | critical | Theorem 8's constants in the check are sampled; the theorem needs a closed tube, coverage of the inputs, and validated constants | Accepted: Theorem 8 restated for a finite horizon with the closed tube, no injectivity, the latent defect $\varepsilon_u$; the Robertson figure called an estimate; what verification needs is stated |
| F3 | critical | (S) with the distance factor on the derivative forces $D\pi = I$ on $S$ | Accepted (an error introduced in revision 3): the distance factor kept for the state only; the derivative hypothesis (S′) separate |
| F4 | critical | Local rank does not bound the coordinate count (attracting circle); $m_M \le \dim M \le m_{\mathrm{ball}}$ false in general | Accepted: counts defined by tolerance, target set and encoder class; local rank separated; Example 6.8 added (proof checked) |
| F5 | major | 2.4(b) omits the exact map's contraction; normal stability leaves tangential errors free; P2.1 does not follow | Accepted: contraction stated; scope paragraph with the circle example; P2.1 and P2.2 demoted to diagnostics and a hypothesis; conservation in physical concentrations |
| F6 | major | The Robertson test does not establish sharpness of Proposition 6.5 | Accepted: "scale comparison"; status "proved, citing Borsuk–Ulam"; $K_2 = 0$, output dimension, chart version |
| F7 | major | Lemma 1.2's sum, endpoint bound and infimum; Remark 1.6's examples; degree claims; Bernstein's converse is a root rate | Accepted, all; exact dissipative $\rho$ corrected (2.544 at $\tau = 1$); "certified degree" replaced by the bound's proxy degree |
| F8 | major | The empirical-Gram variance claim does not follow | Accepted: Proposition 5.2 restated with the deterministic bound $\sqrt{K(x)}\lVert G^{-1}\rVert\lVert b\rVert$ and $\mathbb{E}\lVert b\rVert^2$ (checked) |
| F9 | major | The synthesis mixed runs and over-weighted sampled comparisons | Accepted: archived Robertson records used (rollout/survived 18–30.5, `rollout_over_bound` 0.38–0.73, 2,042 solves, confirmed by a query); Brusselator long step marked scratch; HIRES "bounded" qualified; §9 entries labelled obs./est./thm |
| F10 | major | A slowly rotating eigenbasis refutes the path-averaged law | Accepted: Proposition 6.7(c) replaced by the counterexample; (a)'s bracket stated each way; (b) as an upper bound only |
| F11 | major | Corollaries 6.2–6.3 omit norms, domains and the remainder | Accepted: norm and decoder class stated; three budgets with $\sup\lVert R_\tau\rVert$; no general $O(\varepsilon)$ lower bound |
| F12 | minor | Proof and domain repairs (2.1–2.3, 3.1–3.3, 4.2–4.5, 7.1–7.2, notation) | Accepted: Proposition 4.4 given the stronger covering proof $L/(2N^{1/d})$; window bound needs row-sum control; $\Pi$ for the $L^2$ projector; the rest as listed |
| F13 | major | Prior art contains the techniques and nearby latent rollout bounds (Lee–Carlberg; Farenga et al.; Batson et al.; DeVore–Howard–Micchelli) | Accepted: each statement credited; the contribution restated as a verified end-to-end certificate for a stated class |

## Round 5 (Fable): Claude Fable 5.2 (medium effort), on section F and Demonstration 8.3 (commit `a7e135a`)

Brief: check section F's mechanisms, numbers and verdicts against the scripts and JSON; run refutation tests; check Demonstration 8.3's
rigorous parts line by line. Verdict: no critical findings; the Bernstein parameters recompute exactly and the rigorous half of
Demonstration 8.3 is sound; several "confirmed" rows were bounds, identities or non-discriminating checks. Tests 1–7.

| # | Severity | Critique | Resolution |
|---|---|---|---|
| G1 | major | The aggregate check cannot separate the aggregate mechanism from extrapolation (a full quadratic shows the same vertex ratio) | Accepted: the row is marked inconclusive; a discriminating design (error against the aggregate in bins, the crossing radius against $d$) is the next step |
| G2 | major | The corner peak's bound is 2.6 times loose and cannot fail; ChebTT's ranks are quantised | Accepted: marked as a bound, not informative; the true ranks (0.65 per decade) quoted |
| G3 | major | "Did not finish at 0.1" is not in the JSON and contradicts the $n^{-1}$ law | Accepted: restated with the review's measurement (over 20,000 evaluations) and attributed to ChebTT's acceptance rule |
| G4 | major | Four rows cannot fail or do not discriminate | Accepted: every row now carries its kind (test, bound, identity, record); the ART sentence restated |
| G5 | major | ChebTT's "0 held" on oblique jumps are aborted runs; ChebTT-M fails at $d = 2$ | Accepted: the rank mechanism marked untested; the $d = 2$ failure stated |
| G6 | major | ChebTT-M does not scale mildly on `shared_latent` | Accepted: confirmed for the hub only |
| G7 | major | Demonstration 8.3's reference error is underestimated (shared interpolation table) | Accepted: the floor stated; the $q \ge 20$ rows marked at it |
| G8 | minor | Demonstration 8.3's interval bound is 29% loose; the modulus margin is a secant estimate; its scope | Accepted, all stated |
| G9 | minor | "Identical" counts hold for one of the two full-rank problems | Accepted: "within 0.6%" |
| G10 | minor | `genz_triple` a second failed prediction; `front` at $d = 10$; the jump profile; "within 10%"; predictions committed with results | Accepted: `genz_triple` and the rational ridge added; "within about 30%"; the next pass commits predictions first |
| G11 | minor | Lemma 7.2 already gives the corrected `wave_pair` claim: rank at most $\binom{n+k}{k}$ | Accepted: stated as "the corrected claim" |

## Round 6 (Fable): Claude Fable 5.2 (medium effort), on Demonstrations 8.4 and 8.5 (commit `50f2f2f`)

Brief: fairness of the comparisons, where the reduced surrogate is evaluated, the prediction verdicts, the POLLU conclusion; refutation tests.
Verdict: the reduced ChebTT was compared only with a failed record, while a solve-free data fit on the same chart beats it; the surrogate is accurate
only on the visited set; the POLLU dimension estimate measured single trajectories. Constructive: a solve-free residual tracks the off-set error and a
first-order normal correction removes most of it. Tests 0–9.

| # | Severity | Critique | Resolution |
|---|---|---|---|
| H1 | critical | A degree-6 data fit on the same chart (no solve) beats the reduced ChebTT: $6.4\times10^{-4}$ and $1.5\times10^{-2}$ against $2.46\times10^{-3}$ and 0.30 | Accepted: the baseline rows and the ideal map in Demonstration 8.5's table; the conclusion restated (the solver-queried build buys accuracy off the data, within the chart's span) |
| H2 | critical | Accurate only on the visited set (normal gain 1.41): a kick of 0.01 dex gives $1.9\times10^{-2}$, unrefused | Accepted: stated, with the review's constructive results (the residual bound, the normal correction) |
| H3 | major | POLLU's "locally five-dimensional, needs an atlas" measured single trajectories; the set is about 7-dimensional; a cubic decoder at $k = 7$ reaches $2.0\times10^{-2}$ | Accepted: withdrawn and replaced |
| H4 | major | Cost not like for like (build alone against a whole-run limit; rebuilds gave 10,287 calls, 207–209 s; calls off the data cost 5 times more; the lift chosen on test states) | Accepted: all stated |
| H5 | major | P2 committed for the polished lift, run with the decoder; "rollouts bounded" and P3 could not fail | Accepted: the verdicts restated by kind |
| H6 | major | The rollout rule is neither necessary nor sufficient | Accepted: replaced by the two conditions (first-order defect sum; the surrogate's own propagation) |
| H7 | major | "An affine lift cannot be cheaper" rests on one unfinished run | Accepted: withdrawn; the review's finer-grid result (chart dimension 8 at $d = 40$ and 80) added |
| H8, H9 | minor | Why the polish failed is asserted; smaller points | Noted; the polish's failure mechanism is left untested |
| Suggestions | – | Data fit chosen by the rollout conditions; residual tracking and refusal; a first-order normal correction; restricting the ChebTT domain; more POLLU trajectories; a baseline row in every table | Adopted as §10 item 1's recipe and §11's open items |

## Round 7 (Fable): Claude Fable 5.2 (medium effort), on section G and the HIRES recipe (commit `949ddf0`)

Brief: the mathematics and code of the gradient tier and the recipe; whether each verdict follows from the committed numbers and predictions; whether
the headline (a corrected rollout of $7.3\times10^{-3}$ with Jacobian rows) is fair; the cheapest decisive next experiments. Verdict: the code is
correct and the table recomputes from the JSON, but the headline holds on the first 20 test trajectories only, and the degree range hid a value-only
fit that holds the rollout tolerance 50 times better with no solve. Tests 1–5 in `reviews/scratch-round-7/`; the follow-up runs are
`g5_hires_selection.py` (the review's first suggested experiment, with the third) and `g6_hires_off_trajectory.py` (its second).

| # | Severity | Critique | Resolution |
|---|---|---|---|
| F1 | critical | The corrected rollout of $7.3\times10^{-3}$ holds on 20 test trajectories; on all 40 it is $1.5\times10^{-2}$, $1.0$–$2.7\times10^{-2}$ over eight draws | Accepted: both figures reported; the rule now runs on the deployed surrogate over all 100 training chains, and the test score on all 40 trajectories (`g5`) |
| F2 | critical | Degrees 4–8 stop short: the value-only fit at degrees 9–10 holds $1.2\times10^{-3}$ and $1.5\times10^{-4}$ with no solve; the gradient fit's gain is convergence in the degree and accuracy off the set | Accepted and reproduced: `g5` picks degree 10 for the value-only fit ($1.5\times10^{-4}$ on 40 trajectories) and 12 for the gradient fit (monotone in three draws); "needs the Jacobian rows" and "first … to do so" withdrawn; Demonstration 8.5's table gains degrees 9–10 |
| F3 | major | The single-state condition (ii) is not necessary, is failed by the deployed surrogate (1.873 against 1.857), and its 1.27 is the dynamics' gain along the chart | Accepted: dropped from the rule (reported only), the 1.27 explained, Demonstration 8.5's sentence corrected; the product condition marked untested |
| F4 | major | Cost: central differences (16 plain solves) reproduce the run at 1,600–4,800 plain solves; the compiled-solver overhead is unsourced; timings not in the JSON | Accepted: differences used in `g5`–`g6`, the cost row restated with the review's committed timings, the unsourced figure deleted |
| F5 | major | G2.1 recorded as confirmed at degree 6, not the committed 8, against value-only fits that all failed | Accepted: "not testable as committed" |
| F6 | major | G2.4 committed, failed, unreported | Accepted: row added, failed |
| F7 | major | `g3_hires_grid.py` chooses each degree on the test states; the 900-term cap binds from 11 points per axis | Accepted: both stated as caveats on G3.1 and G3.2 |
| F8 | major | "Falls 30-fold" is wrong: the error rises 12.9-fold (11.1 for second order) | Accepted: corrected |
| F9 | major | The headline was not a prediction; the fits were designed from kicked test states | Accepted: stated under "Provenance"; the follow-up runs committed predictions first |
| F10 | minor | Eight smaller points (the like-for-like kicked comparison, unsourced $3.5\times10^{-7}$, "within 1%", "about 6 times", ranges, $J$ at $X$, refusal as a property of the kick, kicks mixing chart and normal moves) | Accepted, all stated or corrected |
| Suggestions | – | The selection rerun on degrees 4–12 over three draws; off-trajectory Jacobian rows; a rollout kicked at every step | All run (`g5`, `g6`): off-trajectory rows hold a single 0.01 kick at every degree but cost two degrees on the set; kicked at every step no fit holds the tolerance (best $2.3\times10^{-2}$); §11 records the next step |

## Round 8 (external, final): GPT Sol 6.2 (high effort), on the whole memo (commit `65519b5`)

Brief: correctness of every statement and proof, every empirical claim against its JSON and committed predictions, internal consistency of
sections 9–11, novelty, readability for a reader without the code. Verdict: Theorem 8 is a sound starting point for a finite-horizon
certificate of a factored surrogate; two off-set estimates need larger domains, the deployed HIRES map needs its own rollout bound, and
several verdicts and summaries must be tied to their committed predictions and cohorts. It reran the Robertson demonstration (identical), the
static gradient script and `g6` (selected degrees and kicked rollouts within $5\times10^{-7}$ dex). Review: `reviews/round-8-sol.md`.

| # | Severity | Critique | Resolution |
|---|---|---|---|
| 1 | critical | Proposition 8.2's Lipschitz hypothesis covers only $\chi(\pi(N))$; counterexample on the circle of Example 6.8 | Accepted: the domain contains both images (or $\hat\pi$ lands in the chart image); counterexample and the atlas rule stated |
| 2 | critical | Corollary 6.3's extension clause is false with $\sup_S$; counterexample for $x' = -x$ | Accepted: the approximation term is over $\hat\pi(N)$, with the decomposition; 6.4 states continuity of $\pi$ and the remainder in the count |
| 3 | critical | "A rollout holds when" still reads as sufficient; counterexample with a smoothstep surrogate | Accepted: both quantities called diagnostics, the counterexample given, a sufficient condition needs a tube and remainder |
| 4 | critical | Corollary 4.3 reads as a conditional guarantee; counterexample $e(x) = x$ | Accepted: stated as the joint event "pass and violate" with probability at most $\alpha$, with proof; 4.1's schedule from $r = 1$, $0 < \alpha, \eta < 1$ |
| 12 | critical | R5 credits the plain map's Jacobian; R1 never passed | Accepted: both Jacobian comparisons in the R5 verdict; R1 failed as a rule; R2, R4 restated |
| 5 | major | The deployed HIRES map is not factored (correction, input-invariant projection) | Accepted: a Scope paragraph under Theorem 8, the deployed map in G's list, §10 item 1 separated from Theorem 8 |
| 6 | major | Nearest singularities can recede ($w' = iw + w^2$) | Accepted: Proposition 1.5 restated with the counterexample; §11's item restricted to a class |
| 7 | major | Proposition 2.3's defect budget and norm implicit | Accepted: Euclidean or $U$-invariant norm, chain defect budget, Lipschitz on the segments, the Fourier condition |
| 8 | major | Corollary 3.2 needs uniform scales | Accepted: the $h^*$ bound without a distance-shell count, the counterexample, 3.3's uniformity |
| 9 | major | Theorem 6.6 does not predict the PCA lift count; 8.4's literal outcomes | Accepted: outcomes reported (HIRES and POLLU failed), the trend called empirical; 6.6's mesh convention and $\sigma_m$ bound |
| 10 | major | G conflates derivative observations with verified constants and a Taylor law | Accepted: G's opening restated; the one-step error budget and the derivative-row offset given; slopes labelled observations |
| 11 | major | The residual multiplier and normal gain exceed their experiment | Accepted: attributed to the uncorrected reduced ChebTT with its pass shares; the gain restated; calibration on paired data |
| 13 | major | The tensor-grid explanation invokes unisolvence where the design has a nullspace; G3.3 omitted | Accepted: explanation withdrawn, the nullspace stated; G3.3 row added (moot) |
| 14 | major | G1.2 is a single-design comparison against an approximate $L^2$ projection; Hermite halving of node counts | Accepted: tested budgets and the $L^2$ reference reported, conditioning unmeasured; the contour bound cited |
| 15 | major | Narrow the selection and off-trajectory conclusions | Accepted: one split, shared kicks, `g6` as a joint intervention, the tested configurations listed, early-stop scores as lower bounds, `g6`'s degree-10 generalisation failure |
| 16 | major | The 0.11 is a corrected figure on a plain row; the cohort of the factor 2,000 | Accepted and measured (`g7_value_fit_kicks.py`): plain value-only 0.20 at degree 10 (0.11 at 9); the factor stated on 20 trajectories; chart-span kicks labelled |
| 17 | major | The spherical front is not analytic on the box | Accepted: the corner at the origin and its size stated; the kink envelope and the TT cost as representation size |
| 18 | major | Several observations fail the precommit standard | Accepted: a provenance table (prediction commit, output commit, status) in the opening |
| 19 | major | LITERATURE.md's novelty summary describes superseded claims; ISAT, Hermite and gradient-enhanced regression uncredited | Accepted: summary items 2–3 restated, items 8–9 and TH-62 to TH-64 added, a prior-art paragraph in G |
| 20 | minor | Numbers and directions in summaries | Accepted: U2 and §9 counts and direction, T6's $10^{13}$ scope, 8.3's margins and mass offset, 8.4's units, G2.4's range, the per-draw penalty, §9 rows |
| 21 | minor | An outside reader needs the score and deployment conventions | Accepted: a "Scores and terms" convention in the opening; $D'$ for the decoder's derivative; Review VIII's source; $0 < \theta < 1$; Lemma 1.2's finite terminal radius; 6.7(c)'s exact exponent; the Lean list marked as proposed |

## Round 9 (follow-up): GPT Sol 6.2 (medium effort), on the round-8 critical fixes (commit `afe797f`)

Brief: whether the fixes for findings 1–5 and 12 of round 8 are true as written, and whether the revision introduced new errors. Verdict:
findings 3, 4 and 12 resolved; 1, 2 and 5 not yet; seven new problems (three major, four minor). Review: `reviews/round-9-sol.md`.

| # | Severity | Critique | Resolution |
|---|---|---|---|
| 8.1 | – | Proposition 8.2 uses $\mathrm{Lip}_Q$ where only a paired constant is assumed; the circle example uses an embedding, not a chart | Accepted: a finite paired constant $L$ in the display; embeddings allowed; the angle-chart counterexample; the atlas maximum |
| 8.2 | – | Corollary 6.3's $\sup_S$ needs $S \subset Q$ | Accepted: the paired constant and $S \subset Q$ stated |
| 8.5 | – | Theorem 8's scope needs the fibre criterion, and Propositions 2.1–2.2 use the exact map's propagation | Accepted: scope stated by constancy on the encoder's fibres, the fixed-leaf case, augmented encoders, and the full-state certificate's parts |
| 8.3, 8.4, 8.12 | – | Resolved; parameter choices for the counterexample, 4.3's standing assumptions, "fallback" and "sampled" in R5 | Accepted: all stated |
| N1 | major | Corollary 3.2 also needs $e^{\nu\tau}$ uniform ($\tau = d$ counterexample) | Accepted, in T3 and §10 item 5 |
| N2 | minor | 8.4's literal prediction is at the full tolerance (6 against 6) | Accepted: both criteria reported |
| N3 | major | 8.5 groups two arms under one baseline; 1.53 is below the observed 1.532 | Accepted: the arms separated, the central-difference Jacobian named, 1.532 as an empirical constant |
| N4 | major | "Separated every case" overstates across G (`g6` degree 10) | Accepted: restated as selection, with the degree-wise failures |
| N5 | minor | The refusal ratio range covers one fit and misses one kick | Accepted: 0.30–0.37 for the gradient fit, the other fits' ranges given |
| N6 | minor | The Taylor budget's wording still implies first-order terms | Accepted: the slopes identify neither the dominant term nor an order |
| N7 | minor | "Radau at the benchmark's tolerances" is not every check's integrator | Accepted: each experiment's integrator |

## Recipe round 1 (external): GPT Sol 6.2 (high effort), on the HIRES-built recipe's failure across the panel (R2)

Review: `reviews/recipe-round-1-sol.md`. Diagnosis confirmed (in-sample degree selection; no reduction on box-filling problems), with
code findings (chart clamping hid CHEMAKZO's extrapolation; rank-deficient designs; an affine decoder beats the quartic on fisher_kpp_d10;
the singular invariant projection on primordial). Resolution: the generic recipe (`theory/panel/generic.py`): fold-fitted transforms,
grouped held-out selection, no clamp, a refusal rule, a minimum-norm projection (R4).

## Round 10 (Fable): Claude Fable 5.2 (medium effort), on section E, T5 and the generic recipe's results R4, P2, R5 (commit `40fa52f`)

Review: `reviews/round-10-fable.md` (3 critical, 7 major, 6 minor; four post hoc checks in its appendix). Verdict: section E,
Proposition 5.1 and Corollary 5.3 are correct for an exact least-squares fit; the recipe misapplies them.

| # | Severity | Critique | Resolution |
|---|---|---|---|
| 1 | critical | The refusal threshold tol/(2 e_v) - 1 double-counts sampling (e_v already holds 1 + Lambda) and has a floor (the bound is >= 1 and its supremum >= sqrt(p)): it refuses problems the fit serves (post hoc: 7 of 9 re-checked hold the tolerance with <= 2.5% solved) | Accepted: a calibrated gate (held-out scores err/(1 + b), a validated range of b), rerun with committed predictions (R4b) |
| 2 | critical | `served` is true in all records; unserved problems report degree 1 | Accepted: served = within tolerance and solver share below 50%, fixed before the rerun; the best configuration reported when none serves |
| 3 | critical | "Approximation-limited" on the ladders is not in the records: the held-out maximum is a first-step state (sampling); Allen-Cahn's floor is the chart's lift error; fisher_kpp_d3 is cap-limited | Accepted: the claim is withdrawn; per-factor diagnosis per degree to be recorded |
| 4 | major | Section P does not exist | Accepted: written after the rerun |
| 5 | major | HIRES-specific statements remain (section 10 item 1, section 11, section 9) | Accepted: replaced with the generic recipe and panel results |
| 6 | major | 5.3 (b) hides e*(X u {t}) off the data; ridge, rank, clipping, projection and label noise are outside the hypotheses; the cost-weighted refusal rate | Accepted: hypotheses stated; the residual decomposition added |
| 7 | major | Propagation (A_n) is absent from the rule | Accepted in part: held-out chains with the gate active carry it in the selection; a per-step margin is left open |
| 8 | major | R4 and R5 verdicts: several post hoc; G7 open; D3's reason refuted | Accepted: verdicts restated, post hoc ones labelled |
| 9 | major | The gate is blind across the chart's fibres (HIRES: error ~ 1.5 x the quartic-decoder residual, rank correlation 0.99-1.00) | Accepted: the residual term (refuse beyond the training maximum) added to the gate, tested on every problem with k < free |
| 10 | major | The gate's O(p^2) cost; double evaluation; unrepeated timings | Accepted: noted for the timing protocol; a sketched bound left open |
| 11-16 | minor | Caps not derived; independent inputs vs pairs; e_v a fragile maximum; clipping hides overflow; singular factor; two mixing protocols named alike; KIDA's k rule | Accepted: the degree cap raised with early stopping; overflow tested before clipping; pseudo-inverse for rank-deficient factors; names fixed; KIDA open |

## Council round 1 (external, independent): Fable 5.2 (medium), Sol 6.2 (high), Astra 6.2 (medium), on `SYNTHESIS.md` sections 1-10

Reviews: `reviews/council-fable.md`, `reviews/council-sol.md`, `reviews/council-astra.md`. Agreed: repair KIDA's reference comparison
first (Astra: the invariant projection, given the solver's own answer, missed by up to 293 dex); a deployment test with neighbours
carrying surrogate error before any model search; rollout-tuned ChebTT and the chart as controls; one bounded MLP against linear and local
fits on equal data; designed data compared at equal cost; stop full-panel reruns, architecture surveys and unconditional AFINNI/ART
sweeps. Resolution (`SYNTHESIS.md` section 3.9): KIDA's conservation laws declared in `src` (approved); the charge-neutral dark range and
13 key species; the third recipe (generic3: calibration for the maximum on states set apart, mixed calibration, a propagation margin, k
up to the free dimension); p4, p5, p6 and r6 run with predictions committed first.

## Council round 2 (external, independent): the same three, on `SYNTHESIS.md` sections 3.9-3.16, 9.6, 9.7 and 10 (commit `39422d9`)

Reviews: `reviews/council2-fable.md`, `reviews/council2-sol.md`, `reviews/council2-astra.md` (each with read-only checks).

| # | From | Critique | Resolution |
|---|---|---|---|
| 1 | Fable | The per-step budget is tol/2 everywhere while the accumulation factor A_n is about n or more on every problem (31-40 on the lattices, 140 CHEMAKZO, 6,669 HIRES); it predicts HIRES's 0.16 weak-mixing chain from files written before the run | Accepted: budgets tol/(2 A_n) and a coherence-based one recorded beside tol/2 |
| 2 | Fable | HIRES's mixed-input defect is first order in the off-chart offset (coherence 0.84); a constant normal term fitted on training trajectories' mixed states cuts its median 37-62x (not its maximum) | Accepted: the tube experiment (normal term, sensitivity chart, Jacobian labels, a plain-data control) with predictions first |
| 3 | Fable | Three served fits are singular (CHEMAKZO, fisher_kpp_d5, fisher_kpp_d10): the deployed coefficients are not the minimum-norm fit the bound describes, the 281-1,642 us gates are an SVD per call (a precomputed factor: 15 us, the same decisions), and the 4.7-5.0 "first-step errors" are the output clip of an undetermined fit | Accepted: a truncated-SVD fit, the bound and an out-of-span term from one precomputed factor; R4c and R4d rerun |
| 4 | Fable, Astra | Reading 9.7.1 does not follow: for ChebTT the column is within 0.4-2.1x of the plain rollout; the 2.7-13x is reference-partner mixing being easier, and p6 also changed the donor geometry | Accepted: chained mixing is no longer the coupled headline; paired forced- against closed-neighbour columns with the same geometry; D = 0.25 and unsorted cells |
| 5 | Fable, Astra | Reading 9.7.4 is wrong: fisher_kpp_d10's 1,600 arm moves the ungated worst from 4.9 to 0.118 (median 3.4 to 7.1e-3); its initial law has at most five shape parameters and cannot fill 10 dimensions; nearest-neighbour distance in the ambient metric is not a dimension | Accepted: the reading withdrawn; first-step data design stopped (the starts are the panel's i.i.d. initial law) |
| 6 | All | Selection must take the fastest configuration within the coupled constraints, not the smallest error; CHEMAKZO's 0.7 is mostly the SVD branch (Fable), and degree 5 served at 0.9% solver (Astra) | Accepted: selection by effective speed-up among configurations within the margin; chains stopped once they cannot be served |
| 7 | All | Conservation laws: Burgers d10 and d20's second law and one of POLLU's are spurious too (off-data checks); the effect of bruss_d20's on scores is 3.5e-8; in the benchmark's store only burgers_d10 is affected; declare laws per system, verify off data | Accepted: theory code uses laws verified off data; the `src` change (declared laws, an off-data check) awaits the user's approval; burgers_d10 marked |
| 8 | Astra | p5b labelled species in the wrong order (QOI order against sorted columns) | Accepted and fixed (`092220b`): the local arm's misses are S+, HCN, C and N2H+ |
| 9 | Sol, Fable | Verify the exact baseline before quoting speed-ups: KIDA's step densifies an analytic sparse Jacobian; compare with a sparse path and a tighter reference | Accepted: a KIDA solver audit (sparse against dense, tighter tolerance) before any KIDA speed-up is quoted |
| 10 | All | KIDA: no more model search until the coverage question is answered; measure the local arm against the number of tracers, the rank of the key species' futures, the attainable avoided solver time | Accepted: KIDA limited to these measurements |
| 11 | Fable | Basis for charts of 8 or more coordinates (POLLU, Nelson, primordial, lorenz96_d20): an adaptive index set against a trajectory-fitted TT (T-a) | Accepted, after the repair and the tube experiment |
| 12 | Sol, Astra | One model per output: a cheap grouped output-rank screen before a block TT | Accepted, with item 11 |

## Council round 3 (external, independent): the same three, on `SYNTHESIS.md` sections 3.17-3.22, 9.8 and 10 (commit `125b58e`)

Reviews: `reviews/council3-fable.md`, `reviews/council3-sol.md`, `reviews/council3-astra.md` (each with post hoc read-only checks).

| # | From | Critique | Resolution |
|---|---|---|---|
| 1 | Fable | POLLU is data-limited, not accumulation-limited: with 240 more training trajectories the same degree-3 fit's one-step median falls 2.0e-4 -> 1.7e-5 and its ungated rollout 3.9e-2 -> 2.8e-3; multi-step fits and periodic exact steps gave 1.4-2x | Accepted: learning curves in whole trajectories on every unserved problem, the budget in solver seconds |
| 2 | Fable | HIRES is refused by the product gate (1 + b) e_hat: b is about sqrt(rank), so one mixed calibration state refuses half the plain states; an additive envelope a0 + a1 b + a2 rho (rho the offset from the bundle) calibrated by a linear program refuses 0% of plain inputs (14.5% now) and 38% of weakly mixed ones (73%), every accepted defect under tol/2 | Accepted: the envelope gate, then the 18 problems rerun |
| 3 | All | The composite: the on-bundle fit frozen, the normal term as a residual that vanishes on the bundle, switched on beyond the decoder's residual; Fable's check: the plain fit unchanged, the median defect 20-37x lower inside the radius, the maximum unmoved | Accepted: the anchored tube at fixed degree, tube samples by rejection on the offset, a region-specific gate |
| 4 | Fable, Astra | CHEMAKZO's 7-14x from the sensitivity chart is an artefact: the arms' selection rows are identical to degree 5 and a timing tie picked another degree; what helps on full charts is tube data in the fit (selection chains 94-370x lower at degrees 6-7) | Accepted: section 3.20's reading withdrawn; cost ties broken by chain error |
| 5 | Astra, Sol | POLLU's peak is mostly inherited error (0.04627 of 0.04702 dex remains after an exact step from the hybrid input); the worst tracer's own-input defect is 4.6e-3, five times the teacher-forced estimate | Accepted: traced alongside item 1 (data first, since it is cheapest and decisive) |
| 6 | Fable, Astra | KIDA's step is not mostly its right-hand side: dense LU 42-46%, right-hand side 28-36%, SciPy's Python 23% | Accepted: section 3.22's reading corrected; a KIDA solver acceleration (fewer factorisations, a compiled path) before any fit |
| 7 | Fable | Speed-ups are against SciPy's per-call overhead: Robertson's 778x is about 8x against a compiled solver (estimated from operation counts), bruss_d20's 14x 2-4x | Accepted: a compiled baseline measured before any speed-up is quoted as final |
| 8 | Fable | Tensor trains are not supported by the records (the fitted coefficient tensors need 6-11x more coefficients as trains); shared outputs cannot pay when the prediction costs about 1 us | Accepted: T-a and T-b deferred |
| 9 | Fable | Nelson is bias-limited at degree 2 and data-limited at degree 3; primordial resists five times the data and local fits, with rare jumps of up to 11 dex in one step | Accepted: both in the learning curves; primordial's jumps examined |
| 10 | Sol, Astra | Qualification needs fresh closed split blocks and a measured 2x on independent coupled runs | Accepted: the final audit of every service policy |
