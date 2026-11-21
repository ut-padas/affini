# Round 7 (Fable): section G, the HIRES recipe end to end, and its statements in sections 8.5, 10 and 11

Reviewer: Claude Fable 5.2, independent review of `theory/MEMO.md` section G at commit `949ddf0` (branch `structure-experiments`),
with `theory/gradients/g1_static.py`, `g2_hires.py`, `g3_hires_grid.py`, `g4_hires_recipe.py`, `g4_diagnosis.py` and
`theory/gradients/data/*.json`. My tests are in `theory/reviews/scratch-round-7/` (`common.py` rebuilds `g4_hires_recipe.py`'s setup from
its own functions; CPU, `SURROGATE_MPI=off`, `OMP_NUM_THREADS=4`; every run under two minutes). No other file was edited.

**Verdict.** The code is correct where I checked it, and every number in the recipe's table recomputes from the JSON. The headline
does not survive two checks the run did not make. (1) The corrected rollout of $7.3\times10^{-3}$ is the worst of the benchmark's first
20 test trajectories; over all 40 it is $1.5\times10^{-2}$, and over eight draws of the sensitivity states it is $1.0$–$2.7\times10^{-2}$
every time. (2) The degree range 4–8 stops one short of where things change: the value-only data fit at degrees 9 and 10, with no
solve, has rollouts of $1.2\times10^{-3}$ and $1.5\times10^{-4}$ on all 40 test trajectories, and the training chains show it. So "the
Jacobian rows are what fix it" and "the recipe needs the Jacobian rows" are wrong for rollouts on the visited set. What the Jacobian
rows do buy is real but different: a fit whose rollout improves monotonically with the degree (down to $1.0\times10^{-3}$ at degree 12),
and accuracy off the visited set at degree 8. No single fit yet holds the tolerance both in the rollouts and on kicked states.

## Findings

### F1 (critical). The headline holds on the first 20 test trajectories only

*Location.* Section G, "a 20-trajectory rollout of $7.3\times10^{-3}$, **within the tolerance**"; section 10 item 1.

*Evidence* (`t2_headline.py`, `t3_more.py`; "gradients" fit, degree 8, corrected; worst rollout in dex).

| draw of the sensitivity states | Jacobians | 20 test trajectories | all 40 test trajectories | all 100 training chains |
|---|---|---|---|---|
| seed 3, 300 (the committed run) | sensitivities | $7.34\times10^{-3}$ | $1.50\times10^{-2}$ | $1.17\times10^{-2}$ |
| seed 3, 300 | central differences | $7.33\times10^{-3}$ | $1.49\times10^{-2}$ | $1.17\times10^{-2}$ |
| seed 11, 300 | sensitivities | $5.3\times10^{-3}$ | $1.03\times10^{-2}$ | $1.10\times10^{-2}$ |
| seed 12, 300 | sensitivities | $4.5\times10^{-3}$ | $2.67\times10^{-2}$ | $2.46\times10^{-2}$ |
| seed 13, 300 | central differences | $5.8\times10^{-3}$ | $1.55\times10^{-2}$ | $2.25\times10^{-2}$ |
| seed 14, 300 | central differences | $1.21\times10^{-2}$ | $1.41\times10^{-2}$ | $1.65\times10^{-2}$ |
| seed 3, 100 | sensitivities | $8.5\times10^{-3}$ | $1.78\times10^{-2}$ | $1.11\times10^{-2}$ |
| seed 11, 100 | central differences | $7.4\times10^{-3}$ | $1.11\times10^{-2}$ | $9.0\times10^{-3}$ |
| seed 3, 600 | central differences | $9.8\times10^{-3}$ | $2.35\times10^{-2}$ | $1.54\times10^{-2}$ |

The committed number reproduces. On the benchmark's 20 trajectories the tolerance holds in 8 of 9 rows; on all 40 it fails in 9 of
9 (one to three trajectories over). The training chains over all 100 trajectories predict the failure in 8 of 9 rows, so the
selection would have caught it had it been applied to the deployed surrogate on every training chain; the script chains the
uncorrected surrogate on 40.

*Fix.* Report both figures (20 for the comparison with the benchmark's records, 40 for the claim), over several draws. Replace
"within the tolerance" by "within the tolerance on the benchmark's 20 trajectories, 1.0–2.7 times the tolerance on all 40". Apply
condition (i) to the corrected surrogate on all training chains.

### F2 (critical). The degree range 4–8 hides that the value-only fit holds the rollout tolerance, with no solve

*Location.* Section G: "The data fit is the obstacle", "The Jacobian rows are what fix it", "This is the first reduced HIRES surrogate
in this memo to do so"; section 10 item 1: "the recipe needs the Jacobian rows"; Demonstration 8.5's table, which stops at degree 8.

*Evidence* (`t5_data_degrees.py`: the "data" fit, plain, no solve; `t3_more.py`, `t4_high_degree.py`: the "gradients" fit, corrected,
seed 3).

| degree (terms) | data fit: one-step | training chains (100) | rollout, 20 / 40 test | $\max\lVert D\Psi\rVert$ | gradients, corrected: training chains | rollout, 40 test | kicked $\varepsilon = 0.01$ |
|---|---|---|---|---|---|---|---|
| 6 (84) | $6.4\times10^{-4}$ | $4.8\times10^{-2}$ | $1.5$ / $2.6\times10^{-2}$ | 4.5 | 0.11 | $8.2\times10^{-2}$ | |
| 7 (120) | $1.5\times10^{-4}$ | 2.4 | 2.2 / 2.4 | 14.1 | $3.9\times10^{-2}$ | $6.6\times10^{-2}$ | |
| 8 (165) | $5.7\times10^{-5}$ | 3.7 | 3.7 / 3.7 | 16.3 | $1.17\times10^{-2}$ | $1.50\times10^{-2}$ | $8.1\times10^{-3}$ |
| 9 (220) | $3.4\times10^{-5}$ | $6.1\times10^{-4}$ | $4.3\times10^{-4}$ / $1.2\times10^{-3}$ | 15.7 | $6.3\times10^{-3}$ | $6.5\times10^{-3}$ | $1.17\times10^{-2}$ |
| 10 (286) | $6.7\times10^{-6}$ | $1.0\times10^{-4}$ | $1.5\times10^{-4}$ / $1.5\times10^{-4}$ | 14.3 | $1.5\times10^{-3}$ | $2.8\times10^{-3}$ | $1.40\times10^{-2}$ |
| 11 (364) | $6.4\times10^{-6}$ | 4.7 | 27 / 27 | 61 | $1.2\times10^{-3}$ | $1.9\times10^{-3}$ | $8.7\times10^{-2}$ |
| 12 (455) | $3.0\times10^{-6}$ | 27 | 28 / 28 | 70 | $6.4\times10^{-4}$ | $1.0\times10^{-3}$ | 0.24 |

1. The data fit at degree 10 holds $1.5\times10^{-4}$ on all 40 test rollouts, 50 times below the headline, with no solve, and
   condition (i) on the training chains selects it ($1.0\times10^{-4}$). It was not found because the script stops at degree 8 and
   degrees 7 and 8 diverge. Its behaviour is erratic in the degree (7, 8, 11, 12 diverge) and it is poor off the set ($0.11$ at
   $\varepsilon = 0.01$ for degree 9), so it is fragile, but "the data fit is the obstacle" is not what the numbers say.
2. The gradient fit's corrected rollout falls monotonically with the degree and holds the tolerance on all 40 trajectories from
   degree 9 (seeds 11 and 12 the same: $3.0$ and $2.0\times10^{-3}$ at degree 9, $6.5$ and $5.2\times10^{-4}$ at degree 12). That
   regularity is the Jacobian rows' real gain on the set. The fallback picked degree 8 because it is the edge of the grid.
3. Off the set the order reverses: the corrected kicked error at $\varepsilon = 0.01$ is within the tolerance only at degree 8 and
   grows with the degree (0.24 at degree 12; at $\varepsilon = 0.05$ it is 0.39 at degree 9 and 4.6 at degree 12). No degree holds
   both the 40-trajectory rollout and the kicked tolerance.
4. The double counting is shown cleanly at degree 9, better than by the memo's own rows: data fit plain $1.2\times10^{-3}$,
   corrected $5.2\times10^{-2}$; "consistent" fit plain $5.6\times10^{-2}$, corrected $1.5\times10^{-3}$. The plain gradient fit is
   0.06 at every degree for the same reason (it lacks $B\,n$).

*Fix.* Extend the degree range until the training chains stop improving; add the value-only rows to the table; restate the result as
"on the visited set a value-only fit of degree 9–10 holds the rollout tolerance but is erratic in the degree and wrong off the set;
the Jacobian rows make the rollout monotone in the degree and give the off-set accuracy at degree 8; no fit holds both". Withdraw
"needs the Jacobian rows" in section 10 and "first ... to do so" in section G, and add degrees 9–10 to Demonstration 8.5's table.

### F3 (major). Condition (ii) as implemented is not necessary, is failed by the surrogate that ships, and compares the wrong pair

*Location.* Section G, conditions (i)–(ii), "$\lVert D\Psi\rVert = 1.27$ at every degree"; Demonstration 8.5, "the degree-7 and 8 fits
reach 2.2 and 4.2 because their own Jacobians amplify (norm up to 10)".

*The 1.27 is the fit, not an artefact* (`t1_dpsi.py`). At the 200 states no chart value is clipped; without the invariant projection
the norm is 1.243 at the same state (training step 15); the projection alone contributes 1.00–1.02. The value equals the dynamics'
gain along the chart, $\max\lVert J\,DD\,V^{\mathsf T}\rVert = 1.29$ (`chart_gain` in `t2_headline.json`), and the median over the states is
1.23: a function of three chart coordinates that matches $\Phi\circ D$ has this norm whatever its degree. The low-degree value-only
fits have it too (1.268 at degree 4). Other draws give 1.28–1.36, tracking a chart gain of 1.28–1.34.

*Problems.*
1. Not necessary: the data fits of degree 9 and 10 have $\max\lVert D\Psi\rVert = 15.7$ and 14.3 and hold the rollout (F2), while degree
   8 with 16.3 diverges. A maximum of one-step norms does not separate them, so 8.5's explanation of degrees 7–8 is incomplete. 8.5
   states the condition for products of $D\Psi$ along the chain; the script tests a single-state maximum.
2. The corrected surrogate, which is the one deployed, has $\max\lVert D\Psi_1\rVert = 1.873$ against the dynamics' 1.857 at the same
   states (1.86–1.89 over the draws): it fails (ii) as written, by 1–2%. Its Jacobian is within 0.08 entrywise of $D\Phi$ (the plain
   one within 0.60), so this is the fitted $B$'s error, and a strict inequality is the wrong test.
3. For the plain surrogate the fair reference is the chart gain (1.29), not $\max\lVert D\Phi\rVert$ (1.86), which leaves 45% of slack.

*Fix.* Test (ii) on products along the training chains (or drop it in favour of the chained error (i), which did separate every case
above), with a stated margin; say that 1.27 is the dynamics' gain along the chart; correct 8.5's sentence.

### F4 (major). The cost statement is pessimistic where it is measured and unsupported where it is optimistic

*Location.* Section G, "Limits of the run", cost; the table's "Cost of a sensitivity solve" row.

*Evidence* (`t3_more.py`, 100 training states, one at a time): plain solve 0.50 ms median (1.21 ms mean); full 8-direction sensitivity
solve 49 ms median; central differences (16 plain solves) 19.6 ms median.

1. Central-difference Jacobians reproduce the run to three digits (F1, rows 1–2) at 16 plain solves each: 4,800 plain solves for
   300 states, and 1,600 for 100 states, which do as well (F1, rows 7–8). "No fewer than the ChebTT's 9,477" is therefore wrong: the
   Python sensitivity solve is simply dominated by differences here.
2. In time the comparison goes the other way as well: 300 sensitivity solves take 15 s and the differences 6 s, against 188–209 s for
   the reduced ChebTT's build, whose calls cost 20 ms each (review 6 H4).
3. "With a compiled forward-sensitivity solver (10–75% overhead) they would cost 330–525" has no source, and reads an overhead quoted
   per solve as covering eight sensitivity directions. Give a citation with the number of directions, or delete it.
4. The 50 ms and 0.6–1.8 ms are in no committed JSON (`g2_hires.json` holds only the 3-direction 11.8 ms and 1.77 ms).

*Fix.* State the cost as 16 plain solves per Jacobian by central differences (agreement with the sensitivities $2$–$8\times10^{-7}$,
`g4_diagnosis.json`), 1,600–4,800 in total; write the timings to the JSON.

### F5 (major). G2.1 is recorded as confirmed on a degree other than the committed one, between fits that both failed

*Location.* Section G table, row `g2_hires`.

The prediction names degree 8. The memo's numbers (0.37 against 0.51; 0.019 against 0.51) are the degree-6 rows of `g2_hires.json`. At
degree 8 the JSON has 2.7 with Jacobians at $N = 200$ against 27.7 with values alone at 800, and 0.048 against 27.7 at $N = 800$.
Every value-only fit errs 0.5–28 dex (the clip), so "matches the value-only fit at about $4N$" compares with a fit that never worked:
the run shows that Jacobians help on random visited points, not the factor 4.

*Fix.* Verdict "not testable as committed: the value-only fits fail at every $N$; with Jacobians the error falls to 0.019 (degree 6)
and 0.048 (degree 8) at $N = 800$"; name the degree in the row.

### F6 (major). G2.4 was committed and is not reported; it failed

*Location.* `g2_hires.py` docstring (G2.4: rollout 50–150 times the one-step error, falling in proportion); absent from the table.

`g2_hires.json`: 3.5 (0.067/0.019, degree 6, $N = 800$) and 6.7 (0.32/0.048, degree 8). `g3_hires_grid.json`: 307, 238, 184
($n = 13$, 11, 9, values alone) and 309 ($n = 13$, with Jacobians). The one-step error fell 19-fold from $n = 5$ to 13 and the rollout
8-fold. G3.3 is also unreported (moot once G3.1 failed).

*Fix.* Add the row with verdict "failed", consistent with review 6 H6 (the ratio is not a property of the dynamics).

### F7 (major). `g3_hires_grid.py` chooses each configuration's degree on the test states

*Location.* `g3_hires_grid.py:69` (`min(..., key=lambda r: r["one_step_worst"])`, the test error); memo rows G3.1 and G3.2.

Every quoted number of the grid study ($1.27$, $2.42$, $2.34\times10^{-3}$) is the best of up to 14 degrees on the test states. The
refutation of G3.1 stands (a selection that favours both arms cannot create the missing factor: with Jacobians $6.6\times10^{-3}$ at
$n = 7$ against $2.3\times10^{-3}$ for values alone at $n = 11$), but G3.2's "near miss" is an optimistic figure. Separately, the cap of
900 terms (`g3_hires_grid.py:63`) stops both arms at degree 15 for $n \ge 11$, so "no gain at 13 points per axis" compares two fits of
the same degree, where the Jacobian rows cannot show a higher degree.

*Fix.* Choose the degree on held-out training states; say that at $n \ge 11$ the cap binds.

### F8 (major). "Falls 30-fold" is wrong

*Location.* Section G, "The correction is right and second order ... The diagnosis row's error falls 30-fold from
$\varepsilon = 0.003$ to 0.01".

`g4_diagnosis.json`: $1.277\times10^{-5}$ at 0.003 and $1.650\times10^{-4}$ at 0.01. The error *rises* 12.9-fold, against 11.1 for the
square of the kick ratio (order 2.12). The conclusion (second order) is right; the sentence is not. It rests on two kick sizes and 40
states.

*Fix.* "rises 12.9-fold from 0.003 to 0.01, against 11.1 for second order".

### F9 (major). The headline was not a prediction, and test data informed the design

*Location.* Section G, verdicts; `g4_hires_recipe.py` docstring; `g4_diagnosis.py:30–35`.

1. R1–R5 say nothing about a corrected rollout within the tolerance (R4 asked only that the consistent fit's corrected rollout be no
   worse than the plain one, and failed). The headline is the best of six test rollouts (three fits, plain and corrected) read after
   the run. It needs a committed prediction and a rerun, which F1 and F2 now supply the material for.
2. The "consistent" and "gradients" fits were designed from `g4_diagnosis.py`, which uses kicked *test* states, after the first run's
   test results were seen. No test number enters a fit or the degree rule, and the fallback rule was in the code of `3ce981b` before
   the first run (though not in its docstring), so the selection itself is clean. "No test data in any choice" holds for the
   parameters, not for the design; the diagnosis should use held-out training trajectories.
3. R1's "holds only through the fallback rule" is fair. For the record: the rule's fallback picks by the plain training chain, and for
   the gradient fit that quantity (0.049 at degree 8) is the missing $B\,n$, not the quality of the deployed surrogate (F2 item 4).

### F10 (minor). Smaller points

- "5.7 against $5.5\times10^{-3}$; 1.88 against $1.87\times10^{-2}$" compares 300 kicked states (seed 3) with the diagnosis's 40 (seed
  5, other signs). Both are the size of $J\,n$ (`Jn_max` $5.5\times10^{-3}$, $1.87\times10^{-2}$), so the agreement says only that $G$'s
  own error at kicked chart values is below about $10^{-3}$. Compute $\Phi(D(u))$ on the recipe's own kicked states.
- "checked against finite differences to $3.5\times10^{-7}$" (table, `g2_hires` row) is in no JSON; `g4_diagnosis.json` has
  $8.0\times10^{-7}$ and $1.7\times10^{-7}$ for the full Jacobian at one state each.
- "each rate within 1% of the Bernstein prediction": `g1_static.json` gives 1.5% for `product_peak_axis0` (1.601 against 1.577),
  0.55% and 0.33% for the others.
- Corner peak, "about 6 times fewer points": the value-only threshold lies between 126 and 180 points and 30 is the smallest size run
  with gradients, so the factor is "at least 4.2", not measured as 6.
- "$\lVert D\Psi\rVert$ is 4.5–16 at degrees 6–8": the JSON has 4.46–16.6 over the two value-only fits.
- The chart-derivative rows use $J$ at the training state $X$, not at $D(\pi X)$, and $B(u)$ is fitted to $J(X)$ against $u = \pi X$.
  The difference is of the order of the decoder residual; say so, since section G writes $J\,DD(u)$ and $B(u) \approx D\Phi(D(u))$.
- "Refusing at $r > \mathrm{tol}/4$ refuses 96–100%" is a property of the kick (every coordinate moved by at least 0.003), not of the
  surrogate; worth saying.
- The kicks move the chart value as well as the normal part, so "kicked error" mixes the two; review 6 separated them.

## What checks out

- **Sensitivities.** `log_jacobian`: $S(0) = \operatorname{diag}(y_0)$ and $S_1/y_1$ is $\partial z_1/\partial z_0$ (the factors $\ln 10$ cancel). The
  directional version in `g2_hires.py` ($S(0) = y_0\odot DD(u)$) is the same identity. `np.kron(J, I_k)` matches the row-major
  ravel of $S$. The block-diagonal Newton matrix omits $\partial(JS)/\partial y$; that affects convergence only, and the result agrees with
  central differences of the benchmark's step to $2$–$8\times10^{-7}$. Central-difference Jacobians reproduce every figure of the run to
  three digits (F1).
- **Pairing and targets.** `Y` is time-major like `visited(train)`. The consistent targets $Y - B\,n$ use the same `normal_term` as
  the deployed correction. The gradient rows are weighted by the chart box's half-widths, i.e. derivatives in scaled coordinates,
  as in `g2_hires.py`. The kick signs are drawn once and shared by the three fits.
- **$D\Psi$.** Central differences with $h = 10^{-6}$ through `Projected`, no clipping active, the identity added for the increment
  model: correct (F3).
- **The table.** Every entry of the recipe's table, the slopes (0.98–1.02), the refusal shares, "$0.4\,r$" (0.33–0.37) and the
  verdicts R2–R5 agree with `g4_hires_recipe.json`. The Hermite ratios and the least-squares figures agree with `g1_static.json`
  apart from F10.
- **Like for like with 8.5.** Same chart, decoder, projection, 20 trajectories and `rollout`; the data fit's $1.5\times10^{-2}$ and
  $6.4\times10^{-4}$ reproduce.

## Own tests

| # | script | question | result |
|---|---|---|---|
| 1 | `t1_dpsi.py` | what sets $\max\lVert D\Psi\rVert = 1.27$? | the fit's own Jacobian (1.243 without the projection, no clipping); equals the chart gain 1.29; corrected 1.873 against the dynamics' 1.857 |
| 2 | `t2_headline.py` | is $7.3\times10^{-3}$ robust? | reproduces; 20 trajectories: within the tolerance in 8 of 9 draws; 40 trajectories: over it in 9 of 9; central differences equal sensitivities; 100 states do as well as 300 |
| 3 | `t3_more.py` | degrees 9–12; timings | corrected rollout on 40 trajectories $6.5\times10^{-3}$ to $1.0\times10^{-3}$ (three draws alike); plain 0.50 ms, sensitivity 49 ms, 16 differences 19.6 ms |
| 4 | `t4_high_degree.py` | condition (ii) and kicks at degrees 8–12 | $\lVert D\Psi_1\rVert = 1.87$ throughout; kicked error at 0.01 grows from $8.1\times10^{-3}$ (degree 8) to 0.24 (degree 12); double counting shown at degree 9 |
| 5 | `t5_data_degrees.py` | the value-only fit at degrees 6–12 | degrees 9, 10 hold $1.2\times10^{-3}$, $1.5\times10^{-4}$ on 40 trajectories with $\lVert D\Psi\rVert \approx 15$; 7, 8, 11, 12 diverge; the training chains predict each case |

## Suggested next experiments

1. **The recipe as it should have been run (about 90 s).** `g4_hires_recipe.py` with degrees 4–12, condition (i) applied to the
   deployed surrogate on all 100 training chains, three draws of 100 and 300 Jacobian states by central differences, scored on the
   benchmark's 20 and on all 40 test trajectories, for the "data" and "gradients" fits. Commit beforehand: the rule picks degree 10
   for the data fit and degree 12 for the gradient fit; both hold the tolerance on all 40 trajectories; the gradient fit is monotone
   in the degree in every draw and the data fit is not.
2. **One fit for both regimes (under 2 min).** The gradient fit at degrees 9–12 fails off the set because its Jacobian rows sit on the
   trajectories. Add value and chart-derivative rows of $\Phi(D(u))$ at chart values displaced from the 300 states by $\pm0.01$ to
   $\pm0.05$ (300–600 more Jacobians, 6–12 s by differences), and ask whether some degree then holds both the 40-trajectory rollout
   and the kicked error at $\varepsilon = 0.01$ and 0.03. A ridge on the coefficients is the solve-free control. This decides whether
   the recipe can be a single surrogate.
3. **The use the correction is for: a rollout that is kicked at every step (under 1 min, about 6,800 plain solves).** Chain each of the
   40 test trajectories with a random kick of 0.003 dex before every step, against the solver chained with the same kicks. Compare the
   data fit at degree 10, the gradient fit corrected at degrees 8 and 10, and the plain gradient fit. This is the operator-split
   proxy, and the only setting in which F2's value-only fit and the Jacobian recipe are expected to separate decisively.

## Overall assessment

Section G's static and grid results are sound apart from two verdicts (G2.1, the unreported G2.4) and a test-set selection in the grid
study. The mathematics and code of the recipe are right: the sensitivities, the consistent targets, the correction and the Jacobian
norm all check, and the constant 1.27 is the dynamics' gain along the chart. The headline is the problem. It holds on 20 of the 40
test trajectories at a degree fixed by the edge of the grid, was not predicted, and is credited to the Jacobian rows although a
value-only fit two degrees higher holds the same rollouts 50 times better at no cost. The defensible statement is narrower and still
useful: Jacobian rows at 100–300 states (1,600–4,800 plain solves by differences) turn an erratic fit into one that converges with the
degree on the visited set, and give the only fit within the tolerance at a kick of 0.01 dex; holding both at once is open.
