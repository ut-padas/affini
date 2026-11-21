# Reviews of the theory memo

Each round: the reviewer and its brief, a summary of its critiques, and the resolution of each (accepted and changed,
partly accepted, or rejected with the reason). The full reviews are in `reviews/round-<n>.md`, their tests in
`reviews/scratch-round-<n>/`; `git log -- theory/` shows each revision of the memo.

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
