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
