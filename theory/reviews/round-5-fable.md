# Review round 5 (Fable): section F and Demonstration 8.3

Reviewed at commit a7e135a (branch `structure-experiments`): section F of `theory/MEMO.md`, Demonstration 8.3, `theory/functions/f1`–`f4`
with their JSON, and `theory/demos/robertson_leaf.py` with its JSON. Standard applied: the author's "tier 1" (scaling laws from stated
mechanisms, judged by trend). Own scripts are in `theory/reviews/scratch-round-5-fable/` (`t1`–`t7`, each under 3 minutes, CPU, no MPI).
No existing file was edited.

Summary of the verdict. The table's numbers match the JSON with a few small exceptions (G9, G10). The mechanisms are stated correctly;
both Bernstein parameters recompute exactly (G0). The polynomial bounds of Demonstration 8.3 are correct as implemented and dominate a
brute-force check. The weaknesses are of three kinds: four "confirmed" rows rest on a check that could not have failed or that does not
separate the stated mechanism from an alternative (G1, G2, G3, G4); two rows report the favourable part of the records and leave out the
unfavourable part (G5, G6); and the reference-error term of Demonstration 8.3 is about 9 times too small, because both references share one
interpolation table (G7).

## Critiques

**G0 (no defect; recomputation).** With a formula different from the script's (sum of the distances to the foci, not the Joukowski
inverse), `t1_recompute.py` gives for the product peak at $d = 2$ ($a = (3.033, 4.217)$, $w = (0.141, 0.717)$, the registry's draw)
$\rho = (2.07527, 1.63841)$ and $1/\log_{10}\rho = (3.154, 4.664)$, and for the corner peak $\rho_s = 3.90616$, $1/\log_{10}\rho_s = 1.6899$.
Both agree with the JSON to every printed digit. `genz_params` in `f1` reproduces `src/problems/genz.py`'s draw. The kink's $n^{-1}$ rate
(coefficients $k^{-2}$), the half-jump lower bound, $\cos$ of a sum having TT rank 2, and Lemma 7.2's use for the corner peak are all
correct statements.

**G1 (major; row "Aggregates", `f4_aggregates.py`). The vertex check does not separate "a few aggregates fail at the vertices" from "any
least-squares polynomial fitted on uniform points extrapolates badly to the vertices".** The fit is trained on uniform points, whose
leading aggregate spans $[-0.71, 0.74]$, and tested on vertices, where it spans $[-1.02, 1.08]$ (box extreme $\pm1.10$): the vertices are
outside the training range of every model, whatever its coordinates. Control (`t5_aggregate_control.py`, same data and seeds as `f4`):

| model | uniform worst | vertex worst | ratio |
|---|---|---|---|
| top-2 directions, degree 3 (the memo's arm) | 2.5% | 26.8% | 10.8 |
| top-3 directions, degree 4 (the memo's arm) | 2.3% | 16.5% | 7.3 |
| **all 20 coordinates**, degree 2 (231 terms), uniform training | 4.7% | 39.8% | **8.4** |
| all 20 coordinates, degree 1, uniform training | 15.0% | 62.1% | 4.2 |
| top-3 directions, degree 4, trained on uniform points and 2,000 other vertices | 3.7% | 12.9% | 3.5 |
| all 20 coordinates, degree 2, trained on uniform points and vertices | 4.0% | 32.9% | 8.2 |

A model that drops no coordinate shows the same ratio (8.4) as the memo's arms (4–12), so the ratio measures extrapolation, not the
coordinate count. The part of T7 that survives the control: with the vertices in the training set the three-aggregate fit still errs 13%
at the vertices against 3.7% on uniform points. *Fix:* report the arm trained with vertices (or a Chebyshev-weighted design) beside a
full-coordinate control, and state the claim as "at equal training design, k aggregates leave x at the vertices against y in the bulk".
Also say what "suffice" means: the uniform error saturates at 2.3% of the range from $k = 2$, $q = 3$ on (2.48%, 2.47%, 2.25%, 2.27%),
which is the neglected directions' floor and is far above any tolerance the benchmark uses.

**G2 (major; row "Smooth, a function of one aggregate"). The check is too loose to fail, and the measured slopes are ChebTT's rank step,
not the function's ranks.** (a) The bound 1.69 per decade is 2.6 times the true growth: the sup-norm $\varepsilon$-ranks of the corner
peak at $d = 2$ (SVD on a 200 × 200 Chebyshev grid, `t1_recompute.py`) are 2, 3, 4, 6, 7 at $10^{-2}, \ldots, 10^{-10}$, 0.65 per decade.
(b) ChebTT reports 2, 4, 6, 6: every value is even because the cross adds rank in steps of `rank_step = 2` from `rank0 = 2` and the
rounding in `compress` did not remove the excess, so "0.7, 0.7, 1.0 per decade" is a four-point fit through quantized upper bounds. (c)
"Independent of $d$" is asserted from 0.7, 0.7, 1.0, which neither supports nor refutes it (the pole's order $d + 1$ raises the
prefactor with $d$; rank 8 at $d = 5$ against 6). (d) A sharper prediction from the same mechanism was available: the rank across a cut
is at most the degree in the better-conditioned side plus one, and for $x_1$ alone the pole is at $x_1 = -1/a_1$, giving
$\rho = 7.03$ and 1.18 per decade at $d = 2$. *Fix:* state the row as an upper bound that is "not violated", not "confirmed"; measure
$\varepsilon$-ranks by SVD at $d = 2$ and by TT-rounding a high-accuracy train at $d = 3, 5$; add a profile whose bound is nearly attained.

**G3 (major; rows "Kink" and "Separable kinks, whole function"). The statement that ChebTT without breakpoints "did not finish at 0.1
in 170 s" is not in the JSON, and it contradicts the row's own law.** The $n^{-1}$ law with the measured constants (0.10 at $n = 16$,
0.044 at $n = 32$ for one factor) puts the degree for a relative tolerance of 0.1 at a few tens per axis, a rank-1 train of a few
thousand evaluations. Measured (`t4_kink_d3.py`, $d = 2$, `edges=False`): tolerance 0.3 is accepted with degrees (12, 13) and 388
evaluations; tolerance 0.1 raises "would exceed 20,000 evaluations" after 5 s, and with a cap of 200,000 does not return within the
170 s limit. So the failure is real, but it is not the $n^{-1}$ law: ChebTT's acceptance (every check within `safety * atol`, and the
degree test on the top quarter of the coefficients at a share of `atol / d`) demands far more than sup-norm interpolation does. The memo
uses the non-finish as support for the mechanism. *Fix:* record the failed run (evaluations at the stop, the degrees reached) in the
JSON, and either explain it from ChebTT's acceptance rule or drop it from the evidence. Two smaller points on these rows: the slope
$-0.91$ is a six-point fit whose local slopes are $-1.10, -1.21, -0.23, -0.77, -1.68$ (the error depends on where the kink falls
between nodes), which separates $n^{-1}$ from $n^{-1/2}$ or $n^{-2}$ but no more; and the slope $-0.02$ of the evaluations is set by the
outlier at the loosest tolerance (3,085 at 0.1, then 1,826 → 2,165): without it the slope is $+0.018$. "Nearly flat" stands either way.

**G4 (major as a group; checks that cannot fail, counted as confirmations).**
- *Jump, one piece* ("error never below half the jump"): this is a theorem for every continuous approximant, so the measurement
  (0.71–1.12 against 0.576) can only fail through a bug. It is a sanity check of the script.
- *Kink and jump with the breakpoint*: the script places the breakpoint at the exact, known $w$ and interpolates an entire function on
  each piece; nothing about breakpoints is tested ($10^{-12}$ is the script's own inset `1e-12`, and the kink's floor $10^{-11}$ is that
  inset times the slope). The real test is the next row (ChebTT finds the breakpoints itself), which passes.
- *"Least-squares local models (ART, AFINNI) held none of the kink and jump cases … consistent with T5"*: in the same records ART also
  holds 0 of 3 on smooth problems (`wave_pair`, `shared_latent`, `lotka_volterra_state`, `genz_triple` at every $d$ to 10; `front` at
  $d = 2, 3$), so holding no kink case does not single out kinks, and T5 makes no prediction that this outcome tests.
- *Oscillatory, rank 2*: can fail upward (the cross starts at rank 2 and `compress` can reach 1, as the product peak shows), so it is a
  fair test of the implementation; as a statement about the function it is an identity.
*Fix:* mark these rows "sanity" or "identity", not "confirmed", so the count of confirmed predictions reflects predictions that could
have gone otherwise (the product peak's degrees, the breakpoint builds, the rotation rows, `wave_pair`).

**G5 (major; row "Oblique and curved jumps"). The records support less than the row says, and one contrary record is left out.**
(a) "ChebTT held 0 of the cases it ran": it ran four cases in all (`oblique_jump` and `ball_jump`, $d = 2, 3$, tolerance 0.1, seed 1),
each with status `failed`, no error ratio and no evaluation count (training times 16–79 s where recorded). The cause of the failure
(budget, time, rank limit) is not in what `f2_records.py` reads, so the mechanism "its TT rank is of order $n$" is not tested by these
records; four aborted runs at one tolerance are consistent with it and with other causes. (b) "A rotation aligning the plane with an
axis also works" cites ChebTT-M at $d = 3$ only. The same table has ChebTT-M **failing** `oblique_jump` at $d = 2$ (0 of 1, status
`failed` after 64 s), where the rotation is the easiest. One success and one failure do not confirm the sub-claim. *Fix:* report 1 of 2,
find out why $d = 2$ failed, and test the rank mechanism directly (the numerical rank of $[q\cdot x > 0.3]$ on an $n \times n$ grid
against $n$ takes seconds).

**G6 (major; row "Two shared oblique coordinates"). "Rotated models … scale mildly with $d$" is true of the hub and false of ChebTT-M.**
ChebTT-M's medians are 1,816, 9,017, 75,392, 256,211, 1,591,796 at $d = 2, 3, 5, 10, 20$: a factor 877, and 21 from $d = 5$ to 20. The
benchmark also loosens the tolerance as $d$ grows (0.01 at $d = 2$, 0.05 at $d = 5$, 0.1 at $d = 10, 20$ for the medians quoted), so the
growth at a fixed tolerance is steeper than the medians show. A coordinate count of 2 predicts a cost that does not depend on $d$ beyond
finding the rotation; 1.6 million evaluations at $d = 20$ needs another explanation (six separate trains, the rotation's accuracy), and
the mechanism does not give it. What the records do confirm: the hub (2,400 → 20,000) scales mildly, and ChebTT-M beats ChebTT from
$d = 5$ (75,392 against 376,037) and holds at $d = 20$ where ChebTT fails; at $d = 2, 3$ the two cost the same. *Fix:* split the verdict
(hub: confirmed; ChebTT-M: wins, but its growth in $d$ is unexplained), and compare across $d$ at one tolerance (0.1 is run at every $d$).
The same caveat on tolerance sets applies to every "→" across $d$ in the record rows; it is harmless for the kink rows, whose cost is
flat in the tolerance.

**G7 (major for the $q \ge 20$ rows, harmless below; Demonstration 8.3, `robertson_leaf.py`: `Curve.time`, `ref_err`). The reference
error is underestimated about 9 times, because both references share the error it should measure.** `Curve.time` inverts
$z_1 \mapsto t$ by linear interpolation in a table on the same 200,001 times for both tolerances. Its error (`t7_time_inversion.py`):
$\sup|z_1(\mathrm{time}(u)) - u| = 4.9\times10^{-11}$ on the defect grid, which changes the tabulated step map by
$(4.9, 9.5, 8.5)\times10^{-11}$ per component; two Newton steps on the dense output reduce the inversion error to $10^{-16}$. The
script's estimate, the difference between the rtol $10^{-12}$ and $10^{-10}$ references, is $1.1\times10^{-11}$: it cannot see an error
both share. Independent signs of the same thing (`t6_robertson_bounds.py`): the reference map minus a smooth fit of itself has a
residual of $6.3\times10^{-11}$ with point-to-point jumps of $8.9\times10^{-11}$; the held-out chains (which do not use the table)
differ from the curve's map by $9.4\times10^{-11}$; the "defect" at $q = 24$ ($1.03\times10^{-10}$) and `L_delta_estimate`
($6.5\times10^{-6}$, identical for $q = 16, 20, 24$) are this error and not the fit's; and the measured rollout stops improving at
$2.97\times10^{-9}$ for $q = 20$ and 24. Consequences: at $q = 20$, $\varepsilon_u$ should be about $1.1\times10^{-10}$, not
$7.0\times10^{-11}$, and the bound about $7\times10^{-8}$, not $4.9\times10^{-8}$; the bound still holds, and rows $q \le 16$ change by
a few percent at most. The ratio "16" at $q = 20$ compares a bound with a measurement at the reference's noise floor and says nothing about
tightness. *Fix:* invert by Newton on the dense output (or add the inversion error to `ref_err`), and either drop $q = 20$ from the
table or mark it as at the reference floor. The $q = 24$ row is computed and not shown; say why.

**G8 (minor; Demonstration 8.3, what is and is not rigorous).** Checked line by line:
- `cheb_fit`: correct normalisation and orientation (interpolates its nodes to $9\times10^{-16}$ at every $q$).
- `deriv_bounds`: $|T_k'| \le k^2$ and $|T_k''| \le k^2(k^2 - 1)/3$ are right; the grid contains both ends; the sup of $p$ uses the
  half-spacing times $\sup|p_s|$; the derivative bound's term `ds / 2 * sup_d2p_s * scale**2 / scale` equals
  $(\Delta s/2)\sup|p_{ss}|\cdot\text{scale}$, which is the right mean-value bound for $p_u = \text{scale}\cdot p_s$ (the expression is
  correct but written confusingly; its comment mixes steps in $u$ and in $s$). Against 2,000,001 sample points the bounds dominate at
  $q = 8, 16, 20, 24$ and are tight: $\sup|g_1'| = 0.997473 \le 0.997477$, $\sup|g_2'| = 4.0096 \le 4.0097$.
- $G_3' = -(10^{g_1}g_1' + 10^{g_2}g_2')/(1 - 10^{g_1} - 10^{g_2})$ (the $\ln 10$ cancels), and the interval bound is valid, with
  the denominator's positivity checked. It is 29% above the sampled sup (18.80 against 14.60) because it pairs the largest $y_1$ with
  the largest $|g_1'|$ and the smallest $y_3$; that looseness passes directly into the bound (a bound with the sampled $L_G$ would be
  $1.66\times10^{-3}$ at $q = 8$, not $2.12\times10^{-3}$). Subdividing $U_\rho$ into a few intervals would recover it rigorously.
- The horizon loop is right: the bound at step $j$ uses $A_{j-1}$, and the condition uses $A_{38}$ for $n = 39$.
- $\varepsilon_G, \varepsilon_u$ on the curve agree with the defects on the chains' own inputs (where Theorem 8 takes them) to 1%
  ($4.423\times10^{-5}$ both at $q = 8$; $1.962$ against $1.955\times10^{-8}$ at $q = 16$), so taking them on the curve is harmless
  here; the memo should say that the chains are within $10^{-10}$ of $M$ *and* that this offset has no term in the margin.
- "Modulus" margin: a Lipschitz constant estimated from secants of the same grid is a lower bound on the true one, so the margin is
  not conservative in principle; the memo labels it estimated, which is honest. The only margin not built from the grid's own secants
  for the surrogate is the "conservative" one, and its bound is $5.7\times10^{-2}$, 350 times the measurement at $q = 8$ and $2\times10^7$
  times at $q = 20$. The headline table is the modulus variant; the text says so, and the sentence "half of them rigorously" is fair.
- Scope, which the memo states but the table's heading does not: the surrogate is a one-variable Chebyshev interpolant of one
  trajectory's step map, fitted on the curve on which it is then tested, with $L_g < 1$ for free. The parts of Theorem 8 that are hard
  ($L_g > 1$, $k > 1$, a chart that is not a coordinate, inputs off $M$) are not exercised.

**G9 (minor; row "Full-rank vector outputs").** "Identical evaluation counts with and without the rotation at every $d$" holds for
`rosenbrock_gradient` and fails for `lotka_volterra_state` at $d = 10$ (tolerance 0.05: 67,107 against 67,479; 68,464 against 68,983)
and $d = 20$ (333,982 against 335,566). The differences are under 0.6%, so "a rotation gains nothing" stands; "identical" does not.

**G10 (minor; reporting).**
- `f3_vector.py`'s docstring predicts full rank for `genz_triple`; the measured local rank is 2 of 3 (the oscillatory and corner-peak
  outputs share the direction $a$ up to scale, so their gradients are parallel). That is a second failed prediction; the memo counts
  one ("One prediction failed") and omits `genz_triple` and `swirl` from the table.
- Row "front": "the hub and a single quadratic leaf hold at $d = 10, 20$": `art_quad` has no record at $d = 10$. The artifact claim
  is right at $d = 20$ (error ratios $10^{-8}$ and $10^{-13}$ with one leaf: the function is constant on the test set); at $d = 10$
  the hub's ratio is 0.88 at tolerance 0.05, so there the front is seen and the fit is merely within a loose tolerance.
- Row "Jump": the profile in `f1` is `exp(x)` below $w_1$, not $\exp(a_1x)$; it is not "one factor of `genz_discontinuous`".
- "to within 10% on the separable case" generalises one instance; see own test 1 (ratios 0.80–1.28 over nine axes).
- Were the predictions made before looking? The scripts and their outputs arrive in one commit, and `f1`'s docstring already contains
  outcomes (the non-finish at 0.1). The two wrong predictions kept in `f3`'s docstring are the only evidence of order. For the next
  pass, commit the predictions (the docstring, or a `predictions.json`) before the run that tests them.

**G11 (minor; row `wave_pair`, the corrected claim).** "A general profile $g(q\cdot x)$ would have growing rank" is right but leaves out
what Lemma 7.2, two rows above, already gives: the rank is at most the degree of $g$ in $s$ plus one, so it grows with the tolerance
and with the *range* of $s$, not with $d$ as such. For a random unit $q$ on $[-1, 1]^d$ the range is $\lVert q\rVert_1 \sim \sqrt{2d/\pi}$,
so $\rho_s \to 1$ as $d$ grows and the rate does depend on $d$; the corner peak is independent of $d$ only because its $a$ is scaled to
a fixed sum. The same lemma gives the general statement the row is reaching for: a degree-$n$ function of $k$ oblique coordinates has TT
ranks at most $\binom{n + k}{k}$, whatever $d$, so an axis-aligned train costs about $d\,n\binom{n + k}{k}^2$. This one formula covers
`wave_pair` (rank 2), the corner peak, and `shared_latent`'s failure for ChebTT ($k = 2$, high degree). The original wrong prediction
was therefore avoidable from the memo's own lemma. Own test 2 checks the corrected claim.

## Own tests

All in `theory/reviews/scratch-round-5-fable/`; run with `SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=<repo>:.`.

**Test 1, separable analytic functions at $d = 3$ (`t2_separable_d3.py`, 4 s). Prediction survives; scatter 20–28%, not 10%.** Fresh
ChebTT at tolerances $10^{-2}$ to $10^{-9}$; degree per decade against $1/\log_{10}\rho_i$:

| function | predicted | measured | ratio | TT ranks |
|---|---|---|---|---|
| `genz_product_peak`, $d = 3$ | 2.65, 1.41, 4.07 | 2.96, 1.58, 3.94 | 1.12, 1.12, 0.97 | 1 |
| $\prod 1/(b_i^2 + x_i^2)$ on $[-1, 1]^3$, $b = (0.2, 0.5, 1.5)$ | 11.59, 4.78, 1.93 | 11.74, 3.83, 2.26 | 1.01, 0.80, 1.17 | 1 |
| $\prod 1/(c_i - x_i)$, $c = (1.05, 1.5, 4)$ | 7.31, 2.39, 1.12 | 6.68, 3.07, 1.11 | 0.91, 1.28, 0.99 | 1 |

The ordering of the axes and the magnitudes over a range of 10 in the rate are right in every case. This is the strongest result in
section F, and it could have failed.

**Test 2, a general oblique ridge (`t3_ridge.py`, 98 s). The corrected `wave_pair` claim survives, with the qualifications of G11.**
$g(q\cdot x)$ on $[-1, 1]^d$, $q$ a random unit vector; (max TT rank, evaluations) at $10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}$; cap 300,000:

| profile | $d$ | result | rank per decade (Lemma 7.2's bound) |
|---|---|---|---|
| Runge, $b = 0.7$ | 2 | (4, 485), (6, 1,347), (10, 1,495), (12, 2,762) | 1.4 (≤ 4.3) |
| | 3 | (6, 2,049), (10, 8,082), (14, 16,160), (18, 32,437) | 2.0 (≤ 5.7) |
| | 5 | (6, 5,390), (10, 24,432), (14, 63,324), (18, 204,585) | 2.0 (≤ 5.2) |
| | 8 | (10, 26,486), (14, 105,981), over the cap, over the cap | – (≤ 7.7) |
| Runge, $b = 0.3$ | 2 | (6, 999), (10, 2,758), (16, 7,897), (20, 8,705) | 2.4 (≤ 9.6) |
| | 3 | (11, 13,404), (19, 64,397), (28, 125,882), over the cap | |
| | 5 | (11, 44,369), (18, 185,434), over the cap, over the cap | |
| | 8 | over the cap at every tolerance | |
| $\exp(2s)$ | 2–8 | rank 1 at every tolerance; 283 → 2,308 evaluations | 0 |

So a rational ridge profile does make axis-aligned ChebTT pay: the rank grows with the tolerance (and with the sharpness of the
profile), is the same at $d = 3$ and 5, and the evaluations grow about as $d\,n\,r^2$ until the build no longer fits. `wave_pair` and
$\exp$ are the exceptions (profiles that are sums of few products). Lemma 7.2's bound holds and is 2.6–4 times loose here too, which
repeats G2: "at most" rows cannot be the basis of a quantitative prediction.

**Test 3, kinks at $d = 3, 4$ (`t4_kink_d3.py`). Prediction survives.** `genz_continuous`, ChebTT with breakpoints, $10^{-1}$ to
$10^{-6}$: rank 1 throughout; 2 pieces on every axis except one axis of one case ($d = 3$, tolerance 0.1: pieces 2, 1, 2); evaluations
2,084, 4,003, 2,939, 3,047, 3,068, 3,185 at $d = 3$ and 6,293, 4,521, 3,872, 3,839, 4,004, 4,187 at $d = 4$: flat, with the excess at
the loose tolerances that the $d = 2$ data also show. The same script gives G3's numbers for the build without breakpoints.

**Test 4, the aggregate control (`t5_aggregate_control.py`). The row's check does not discriminate**; table in G1.

**Test 5, the corner peak's true ranks (`t1_recompute.py`).** 2, 3, 4, 6, 7 at $10^{-2}$ to $10^{-10}$ (0.65 per decade) against the
bound 1.69 and ChebTT's 2, 4, 6, 6; see G2.

**Test 6, Demonstration 8.3 (`t6_robertson_bounds.py`, 18 s; `t7_time_inversion.py`).** The polynomial and interval bounds dominate
brute force; the defects on the chains equal those on the curve; the reference's shared inversion error is $4.9\times10^{-11}$ in $z_1$
and $9.5\times10^{-11}$ in the step map against the script's $1.1\times10^{-11}$; see G7 and G8.

## Suggestions: what in the function suite would test T5, T6 and T7 before step maps

1. **T5 (one least-squares quadratic leaf), directly.** Nothing in section F tests it. On a box of half-width $h$ around a point of a
   smooth test function (the product peak, the corner peak), fit the quadratic by least squares from $N$ uniform points and measure the
   sup error against $h$ and $N$: Proposition 5.1 predicts (1 + $\mathcal{L}_N$) times the best quadratic's error, so slope 3 in $h$ with a
   constant from the third derivative; Proposition 5.2 predicts the sampling term's $N^{-1/2}$ decay and its $\sqrt{K(x)}$ growth toward
   the corners. Then the leaf count: ART's leaves on a smooth function should scale as $\mathrm{tol}^{-d/3}$. All are exponents, each can
   fail, and each run takes seconds at $d = 2, 3$. A kink through the leaf gives the contrasting slope 1 in $h$, which would make the
   sentence "consistent with T5" an actual test.
2. **T6 (local rank against coordinate count) with a planted fast contraction.** A static stand-in for a stiff step map:
   $f(x) = \Gamma(\pi(x)) + e^{-\gamma}R(x)$ with $\pi$ a curved (not linear) projection onto $k$ coordinates. Predictions:
   $\sigma_{k+1}(Df) \le Ce^{-\gamma}$ (Corollary 6.1), and a surrogate in $k$ coordinates errs by the oscillation term of Corollary
   6.2; sweep $\gamma$ and check both slopes. Add the two cases the memo already separates in words: a function whose local rank is
   $k$ everywhere but whose coordinate count is larger (Example 6.8's construction, as a suite function), and `shared_latent` with the
   curvature term scaled up, where a linear rotation stops being enough. `genz_triple`'s rank 2 of 3 (G10) is a free instance of a
   rank deficit that comes from shared parameters, not shared coordinates.
3. **T6 for tensor trains: the formula of G11.** For $k = 1, 2$ oblique coordinates and a polynomial or rational profile of known
   degree, compare the TT rank with $\binom{n + k}{k}$ and the evaluations with $d\,n\,r^2$, at one tolerance across $d$. This separates
   "coordinate count" (what a rotated model needs) from "TT rank in the given axes" (what ChebTT pays), which is the distinction the
   `wave_pair` row found by accident.
4. **T7 with a design that can fail (the repair of G1).** Proposition 7.1 predicts a degree: for $g(s)$, $s$ the mean of the coordinates,
   the measure-sense degree is that of $g$ on $|s| \le t_\eta = \sqrt{2\ln(2/\eta)/d}$. Use `genz_corner_peak` and `lotka_volterra` with
   the fit trained on a design that covers the box (vertices included), report the error as a function of $|s|$ in bins, and check that
   the radius where the error crosses the tolerance scales as $d^{-1/2}$ across $d = 5, 10, 20, 40$. Run the full-coordinate control at
   the same design. `front` at $d = 10, 20$ gives the same effect for a radial coordinate and is already in the records.
5. **Fixed-tolerance comparisons across $d$ in the records.** Tolerance 0.1 is run at every $d$; rebuild the "→" columns from it, and
   read the failure reason of each failed record (G5), so that "fails" is a statement about a mechanism.
6. **Demonstration 8.3's next rung, still cheap:** a two-coordinate chart on Robertson across leaves (the mass as the second input),
   where $L_g$ is no longer below 1 for free and $U_\rho$ is two-dimensional; and a Newton inversion of the curve's parametrisation so
   the reference floor drops from $10^{-10}$ to the solver's $10^{-12}$.

## Overall assessment

As a tier-1 pass the section does what it set out to do for the Chebyshev mechanisms: degrees from Bernstein ellipses, rank 1 and rank
2 structure, breakpoints for aligned kinks and jumps. Those predictions are correct, recompute exactly, and survive three new tests
that could have refuted them. The honest reporting of the failed `wave_pair` prediction is to the section's credit, and its corrected
claim also survives a test. The section overstates in its bookkeeping: of thirteen rows, about five are predictions that could have
failed and did not; four are identities, theorems or bounds too loose to fail; and four record-based rows quote the favourable records
and pass over ChebTT-M's failure at $d = 2$, its steep growth on `shared_latent`, and failures whose cause was not read. The aggregate
row, the one presented as T7's direct test, does not test T7 as designed. Demonstration 8.3's rigorous half is rigorous as implemented;
its estimated half has one concrete error (the shared inversion table) that leaves every bound standing but invalidates the last row's
ratio. None of this undermines the mechanisms; it limits how much confidence the pass can transfer to T5, T6 and T7, which it barely
touches.
