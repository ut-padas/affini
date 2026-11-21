# Round 6 (Fable): Demonstrations 8.4 and 8.5

Reviewer: Claude Fable 5.2, independent review of `theory/MEMO.md` section 8, Demonstrations 8.4 and 8.5, at commit `50f2f2f`
(branch `structure-experiments`), with `theory/demos/reduced_dims.py`, `reduced_chebtt.py`, `curved_lift.py` and their JSON outputs.
Scripts and logs of the tests below are in `theory/reviews/scratch-round-6-fable/` (CPU, `SURROGATE_MPI=off`, one BLAS thread unless
stated; every run under four minutes). No existing file was edited and nothing was written to the benchmark's store.

**Verdict.** The numbers in the two demonstrations recompute (the table of 8.4, the lift errors, the one-step error, the rollout, the
invariant), with one exception (the build's cost, H4). The conclusions drawn from them do not all follow. Two are wrong as stated:
the 9,477 solver calls of the reduced ChebTT buy nothing on the benchmark's own metrics against a least-squares fit on the same chart
that costs no solver call (H1), and the surrogate is accurate only on the visited set, which is not an attracting manifold, so a kick
of the size of the tolerance already breaks the tolerance without a refusal (H2). The POLLU conclusion ("needs an atlas") rests on a
dimension estimate that measures single trajectories (H3). What holds: the decoder lift is a real 157-fold gain over the affine one
on the visited set; the rollout error of the reduced ChebTT is the exact dynamics' amplification of its one-step defects, not an
instability of the surrogate (test 3); the Fisher–KPP chart dimension stays at 8 up to $d = 80$ (test 6).

## Critiques

### H1 (critical). The reduced ChebTT is compared with a failed record, not with the obvious baseline, which beats it

*Location.* Demonstration 8.5, the table and "a curved lift makes a reduced ChebTT feasible" (the commit's title).

*Problem.* The only comparator is box ChebTT's "over the 240 s limit" record, which has no error, no evaluation count and covers a
different task (the 8-D bounding box). The baseline on the *same* task (the same three chart coordinates, the same test states, the
same projection onto the invariant, the benchmark's `rollout`) is a least-squares polynomial from the chart value to the next state,
fitted on the training pairs every method is given: no solver call. `reduced_chebtt.py` computes such a fit ("data-only fit test") but
8.5 does not report one, and the memo's own review-3 result on the same problem (degree 6, rollout $5.4$–$5.7\times10^{-2}$) is already
five times better than 8.5's 0.30.

*Evidence (test 1).* Same chart, box, projection and scoring as `curved_lift.py`; all 6,800 test states; 20 rollouts of 170 steps.

| surrogate ($k = 3$) | solver calls beyond the trajectories | one-step worst | worst rollout |
|---|---|---|---|
| reduced ChebTT, decoder lift (memo) | 9,477 | $2.46\times10^{-3}$ | 0.30 |
| data-only, degree 5 (56 terms) | 0 | $2.1\times10^{-3}$ | $7.4\times10^{-2}$ |
| data-only, degree 6 (84 terms) | 0 | $6.4\times10^{-4}$ | $1.5\times10^{-2}$ |
| data-only, degree 7 (120 terms) | 0 | $1.5\times10^{-4}$ | 2.2 |
| data-only, degree 8 (165 terms) | 0 | $5.7\times10^{-5}$ | 4.2 |
| ideal reduced map $\Phi(D(\pi z))$ (the function ChebTT approximates) | one per query | $3.4\times10^{-4}$ | $2.5\times10^{-2}$ |

The degree-6 fit is 4 times better in one step and 20 times better in the rollout (1.5 times the tolerance), for nothing. The ideal
row shows where the ChebTT version loses: the lift error alone would give a rollout of $2.5\times10^{-2}$; the other factor of 12 is
ChebTT's own error ($2.4\times10^{-3}$ of the $5\times10^{-3}$ it was allowed), spent on a chart box of which the data occupy a few
percent (H4).

What the solver-queried build does buy, and the memo does not say: validity on the chart box away from the data. For kicks *inside
the chart's span* of 0.01 dex the reduced ChebTT errs $2.8\times10^{-3}$ and the degree-6 data fit $6.6\times10^{-2}$ (test 2). That is
the honest case for it, and it is a statement about inputs off the visited set, where H2 applies.

*Fix.* Add the data-only rows and the ideal row to the table; restate the conclusion as "a decoder lift makes the reduced oracle
well defined; on the visited set a data fit is cheaper and better; the oracle's value is off the data, in the chart's span only".
Compare with box ChebTT only as "a different task that did not finish".

### H2 (critical). Accurate only on the visited set, which does not attract: a kick of the tolerance's size breaks the tolerance, unrefused

*Location.* 8.5, "one-step worst on the test states $2.46\times10^{-3}$ ... none refused", and 8.4/8.5's "curved slow manifold".

*Problem.* The test states are held-out trajectories from the same family of initial conditions, so they lie on the visited set
(decoder residual: median $3\times10^{-5}$). `Reduced.refuses` looks only at the chart value, so a state at any distance from the
visited set with chart value in the box is answered. And the visited set is not a slow manifold that the step returns to: it is the
image of two initial-condition parameters and time (which is why its dimension is 3). The exact step *keeps* a kick: the median
singular values of $D\Phi$ at visited states are 1.80, 1.21, 0.40, 0.027, 0.0075, and restricted to the five directions normal to
the chart they are 1.41, 0.39, 0.24, 0.016, 0.002 (test 5). The chart's normal space contains directions the step amplifies. The
surrogate ignores them, so its error off the set is what the exact step keeps of the kick.

*Evidence (test 2; 400 held-out states, kicks of max-norm $\varepsilon$ with random signs; no state refused in any row).*

| kick | $\varepsilon$ (dex) | exact step keeps | reduced ChebTT error | ideal $\Phi(D(\pi z))$ | data-only degree 6 |
|---|---|---|---|---|---|
| every component | 0.001 | $2.2\times10^{-3}$ | $2.6\times10^{-3}$ | $1.9\times10^{-3}$ | $7.7\times10^{-3}$ |
| every component | 0.01 | $2.1\times10^{-2}$ | $1.9\times10^{-2}$ | $1.9\times10^{-2}$ | $7.3\times10^{-2}$ |
| every component | 0.05 | 0.11 | $9.6\times10^{-2}$ | $9.6\times10^{-2}$ | 0.41 |
| normal to the chart | 0.01 | $1.7\times10^{-2}$ | $1.7\times10^{-2}$ | $1.7\times10^{-2}$ | $1.7\times10^{-2}$ |
| in the chart's span | 0.01 | $1.3\times10^{-2}$ | $2.8\times10^{-3}$ | $1.3\times10^{-3}$ | $6.6\times10^{-2}$ |

The error grows linearly, about $1.9\,\varepsilon$, which is Proposition 8.2's middle term with $\mathrm{Lip} \approx \lVert D\Phi\rVert \approx 1.8$: the
encoded base point is wrong by the kick. A kick of 0.01 dex (the tolerance) gives twice the tolerance. This is the situation of an
operator-split simulation, and 8.5 does not mention it (8.3 had a "what this does and does not show" paragraph; 8.5 has none).

*Fix.* State the scope: tolerance held on the visited set only; error about $1.9\times$ the distance from it; nothing refused. Replace
"curved slow manifold" by "the visited set" throughout 8.4–8.5, and say what its dimension is (parameters of the initial conditions
plus time), since that, not a spectral gap, is what the PCA measures. Add a refusal or correction based on the decoder residual
(Suggestions 1–2; both tested below).

### H3 (major). POLLU: "locally five-dimensional, needs an atlas" is not supported; the estimate measures single trajectories

*Location.* 8.5, last sentence; no script or JSON for the POLLU numbers is committed.

*Problem and evidence (test 4).* I reproduce the quoted local singular values exactly (1, 0.80, 0.53, 0.056, 0.015, 0.0044 on
neighbourhoods of 60 states), and the neighbourhoods they come from hold a **median of 3 distinct trajectories (minimum 2)**: with 100
trajectories in a family with six varied emissions, 60 nearest states are mostly consecutive steps of two or three trajectories, so
the local PCA sees their curves, not the set. The set's dimension shows at a fixed step: the 100 training states at steps 0, 10, 30,
58 have singular values 1, 0.46–0.93, 0.41–0.45, 0.37–0.42, 0.24–0.29, 0.19–0.23, then 0.02–0.03. Six transverse directions plus
time: the visited set is about 7-dimensional, as the global spectrum in `reduced_dims_pollu.json` already says (seven values above
0.17, then 0.05). A chart with $k = 5$ is below the set's dimension, so no decoder and no atlas of 5-D charts can work; "not a graph
over five global principal coordinates" is true for a trivial reason.

A different decoder does change the result. Lift errors on 400 held-out states (dex; the residual equals the lift error to within a
factor 2, so the step damps little):

| $k$ | affine | degree 2 | degree 3 | degree 4 |
|---|---|---|---|---|
| 5 | 0.35 | 0.39 | 0.86 | 1.21 (memo) |
| 6 | 0.34 | 0.42 | 1.01 | |
| 7 | $9.2\times10^{-2}$ | $4.9\times10^{-2}$ | $2.0\times10^{-2}$ (120 terms) | 0.14 (330 terms) |
| 8 | $4.6\times10^{-2}$ | $2.8\times10^{-2}$ | (not run) | |

At $k = 7$ a global cubic decoder is 4.5 times better than the affine lift and 60 times better than the memo's $k = 5$ decoder, and
is twice the tolerance. Degree 4 at $k = 7$ has 330 coefficients for 100 trajectories and overfits; at $k = 5$ every degree above 1
is worse than affine for the same reason plus the missing dimensions. The limit is the number of trajectories against a
six-parameter family, not the geometry.

*Fix.* Withdraw "locally about five-dimensional" and "needs an atlas". State: dimension about 7; at $k = 7$ a cubic decoder reaches
$2\times10^{-2}$; whether a global decoder reaches a quarter of the tolerance is open and needs more trajectories (or samples of the
family by its parameters), not local charts. Commit the POLLU output and the local-PCA script, with the number of distinct
trajectories per neighbourhood.

### H4 (major). The cost comparison is not apples to apples, and the build's cost does not reproduce

*Location.* 8.5 table, "build 188 s, 9,477 solver calls" against "over the 240 s limit"; P2.

*Problems and evidence.*
1. *Scope of the limit.* The sweep's 240 s bounds the whole job (`sweep.run_job`: start-up, the build, `predict`, 20 rollouts, the
   projected rollouts, `survived`). The 188 s is `build_s` alone; `curved_lift.py` then also spends the lift variants (about 30 s in
   the JSON's own `seconds`) and the scoring.
2. *Reproduction.* Two rebuilds with the committed construction (`t0_build.py`; one BLAS thread, and default threads) each gave
   **10,287 solver calls and 207–209 s**, identical to each other and not to the JSON's 9,477 and 188 s. The model is otherwise the
   same to 2% (one-step $2.45\times10^{-3}$, rollout 0.29). I did not find the cause; the JSON carries no provenance (commit, threads,
   machine). At 208 s the build plus the scoring is at the limit, not within it.
3. *The cost unit.* 99% of the build is the solver (206 of 207 s), at 20 ms per call; on visited states a call costs 4.2 ms (10,200
   calls in 43 s, test 1). The queries sit where the decoder extrapolates: only 4% of the chart box lies within 0.05 (in box units) of
   a visited state and 9% within 0.1; on 40% of the box the decoder leaves the data's bounding box by more than 0.1 dex, and by up
   to 1.4 dex. So "9,477 solver calls" are about 45,000 on-set calls in time, most of them spent fitting $\Phi\circ D$ where $D$ means
   nothing.
4. *Uncounted work.* The chart and decoder use the 17,000 training transitions (free by the benchmark's convention, but the decoder
   is the surrogate's main ingredient). The lift variant is chosen by its error on the *test* states (`best = min(...)` over
   `lift_err`), at 7,600 solver calls that are not in the count; the choice should use training or validation states.
5. *The task.* Box ChebTT is asked for the 8-D box, the reduced one for a 3-D set; 8.3 said this about Robertson, 8.5 does not.

*Fix.* Report the whole job's time, threads and provenance (keep the case through `src/infra/records.py`, as the repository's rule
for long runs asks); count time or on-set-equivalent calls; choose the lift on training states; restrict the chart box to the data's
neighbourhood (or refuse there) so the oracle is not queried where the decoder extrapolates; state that the tasks differ.

### H5 (major). Two of the three predictions could not fail, and P2 was not run as committed

*Location.* 8.5, "Predictions: P1 ... P2 ... P3 ...".

*Problem.* P1 is a real test and is reported honestly (decoder held, polish failed). P2 was committed "with the polished lift"; the
polished lift was never built on (the script takes the best variant, which was the decoder), so P2 as written is untested and the
memo's "P2 held" should say that the lift was swapped. Within P2, "rollouts stay bounded" cannot fail: $\Psi = G(\mathrm{clip}(\pi z))$ maps
every state into the image of a continuous map on a compact box, and `rollout` hands refused states to the solver. P3 cannot fail
either: `Projected` solves $Wy' = Wy$ at every step by construction (I confirm $7.5\times10^{-16}$ per step and $1.1\times10^{-14}$ after
170 chained steps, test 5); it is an identity, in the sense of review 5 G4. The parts that could fail are P1 and, in P2, the time
(H4) and the one-step tolerance (held, on the visited set).

*Fix.* Label each prediction with its kind (test, identity, cannot fail), as section F now does; replace "bounded" by a number
committed in advance (for instance "within $A \times$ the one-step error" with $A$ from `dynamics.json`).

### H6 (major). The rollout rule ("one-step error near the tolerance over the amplification") is neither necessary nor sufficient

*Location.* 8.5, "The rollout is about 120 times the one-step error, roughly twice the exact dynamics' own amplification ... holding
the tolerance in a rollout needs a one-step error near the tolerance over that amplification."

*What holds (test 3).* For the reduced ChebTT the rollout error is the exact dynamics' propagation of its one-step defects: the
first-order sum $e_n = \sum_j J_{n-1}\cdots J_{j+1}\delta_j$ with the *exact* Jacobians gives 0.30, 0.23, 0.24, 0.15, 0.19, 0.10 on
the six worst trajectories against the measured 0.29, 0.28, 0.20, 0.20, 0.20, 0.16; the error lies in the chart's span (0.29 against
0.007 normal to it); and $\max\lVert D\Psi\rVert = 1.23$–$1.25$ is below $\max\lVert D\Phi\rVert = 1.81$–$1.84$. No instability of the surrogate.

*What does not.* The ratio of rollout to one-step error is not a property of the dynamics: 120 for the reduced ChebTT, 75 for the
ideal reduced map, 23 for the degree-6 data fit (test 1), and the degree-7 and degree-8 fits, with one-step errors of $1.5\times10^{-4}$
and $5.7\times10^{-5}$ (below $0.01/120$), reach 2.2 and 4.2. For degree 8 the exact-Jacobian sum predicts $6.6\times10^{-4}$ where
the rollout measures $1.6\times10^{-2}$, and the surrogate's own Jacobian (norm up to 10) predicts $1.56\times10^{-2}$: that one *is*
Theorem 8's $L_g$. "Twice 52–61" also compares a sum of 170 defects with the growth of one error; the benchmark's accumulation factor
for HIRES is $A_{170} = 5{,}351$ (`dynamics.json`; the growth peaks at 106, not at the end), so the measured 120 means the defects
mostly cancel or lie in damped directions.

*Fix.* State the rule as two conditions: the defects' first-order sum along the chain within the tolerance (computable from the
training chains without a solve beyond the Jacobians), and the surrogate's own propagation ($D\Psi$ products) no larger than the
dynamics'. Quote $A_n$ beside the growth.

### H7 (major). "An affine lift cannot be cheaper than the box" rests on one unrecorded, unfinished run

*Location.* 8.4, "a first build (`reduced_chebtt.py`, Fisher–KPP at $d = 10$, $k = 7$, ten output trains) did not finish within the
benchmark's 240 s, where box ChebTT used 107,002 evaluations in 210 s".

*Problem.* No output file exists (the script writes its JSON only at the end), so the evaluation count at the stop, the per-call cost
and the reason are unknown. The two builds are not the same problem: the reduced one asks each train for half the tolerance on the
next state through `build_vector`, the box one for the full tolerance on the increment through per-component `build`; and off the
visited set the affine lift queries states whose solves may cost several times more (H4.3 measured a factor 5 on HIRES). For HIRES and
POLLU the conclusion follows from the counts (7 of 8, 16 of 20); for Fisher–KPP (7 of 10, 8 of 20, and 8 of 40 and 80, test 6) it
does not follow from anything recorded.

*Fix.* Rerun with a record written per output train and an evaluation cap; report evaluations and time per call; or drop the
sentence and keep the count argument.

### H8 (minor). Why the polish failed is asserted, not tested, and H2 suggests another reason

*Location.* 8.5, "(worse: resetting along the flat directions $V$ moves the state off the curved manifold again)".

Constrained runs converge to the slow manifold in the fibre over $u$. The visited set is not that manifold (H2: the chart's normal
space has gain 1.41 under one step; the step map has three to four undamped directions and the family fills three). Relaxing with
the solver therefore moves the state along slow directions away from the visited set, whatever the reset does. The memo's
explanation may contribute, but the experiment does not separate the two; say so.

### H9 (minor). Smaller points

- The lift error $2.45\times10^{-4}$ is over 400 states; over all 6,800 test states it is $3.4\times10^{-4}$ (test 1).
- "One-step worst ... none refused" is over all 6,800 test states (checked in the code and by recomputation); correct.
- With `Projected` the surrogate is not $\hat G\circ\hat\pi$: it also reads the input's invariant $Wz$, so its effective chart has
  $k + 1$ numbers and Theorem 8's form applies to the pair. Worth one sentence.
- The 8.4 table recomputes from the JSON in every cell. Its last column is blank for three rows; the numbers exist: 2 parameters plus
  time for HIRES, 6 plus time for POLLU (H3).
- `curved_lift.py` and `reduced_chebtt.py` fit the decoder in unscaled monomials; harmless at $k = 3$, degree 4 (condition number
  about $10^4$ in scaled coordinates), but it should be scaled before higher degrees are tried.
- The "survived" figure ($6.8\times10^{-4}$) uses 200 states; fine, as the benchmark does.

## Own tests

All in `theory/reviews/scratch-round-6-fable/`; `common.py` rebuilds the chart, decoder and box of `curved_lift.py`.

| # | script | question | result | verdict on the memo |
|---|---|---|---|---|
| 0 | `t0_build.py` | does the build reproduce? | 10,287 calls, 207–209 s (twice, 1 thread and default); one-step $2.45\times10^{-3}$, rollout 0.29 | accuracy reproduces; cost does not (H4) |
| 1 | `t1_baselines.py` | what do the solver calls buy on the visited set? | degree-6 data fit: $6.4\times10^{-4}$ one-step, $1.5\times10^{-2}$ rollout, no calls; ideal reduced map: $3.4\times10^{-4}$, $2.5\times10^{-2}$; data within 0.05 of 4% of the chart box | refutes "feasible" as a gain (H1) |
| 2 | `t2_offset.py` | accuracy on states kicked off the visited set | error $\approx 1.9\varepsilon$ for kicks in every component, equal to what the exact step keeps; $1.9\times10^{-2}$ at $\varepsilon = 0.01$; nothing refused | Proposition 8.2's middle term, as feared (H2) |
| 3 | `t3_rollout.py` | dynamics' amplification or surrogate instability? | exact-Jacobian first-order sum within a factor 1.7 of the measured rollout on the six worst trajectories; error in the chart's span; $\lVert D\Psi\rVert < \lVert D\Phi\rVert$ | confirms the memo for the reduced ChebTT; the degree-8 control is the opposite case (H6) |
| 4 | `t4_pollu.py` | POLLU: geometry or decoder? | local neighbourhoods hold a median of 3 trajectories; fixed-step spectrum has 6 large values; $k = 7$ cubic decoder $2.0\times10^{-2}$ | refutes "five-dimensional, needs an atlas" (H3) |
| 5 | `t5_checks.py` | invariant; does the visited set attract? | $7.5\times10^{-16}$ per step, $1.1\times10^{-14}$ after 170; $D\Phi$ singular values 1.80, 1.21, 0.40, 0.027, 0.0075; normal to the chart 1.41, 0.39, 0.24 | invariant correct; not an attracting manifold (H2) |
| 6 | `t6_fkpp.py` | does the Fisher–KPP chart dimension grow on finer grids? | lift error $\le$ tolerance/4 first at $k = 8$ for $d = 40$ ($1.8\times10^{-3}$) and $d = 80$ ($2.0\times10^{-3}$); the errors at each $k$ agree to 7% between the two | confirms 8.4's claim, more strongly than 8.4 does |
| 7 | `t7_jacobian_chart.py` | does a chart from $D\Phi^{\mathsf T}D\Phi$ fix the kicks? | normal gain falls (0.38 at $k = 3$, 0.06 at $k = 4$) but the decoder is then evaluated off its data: kicked error 0.06 ($k = 3$), 0.27 ($k = 4$) at $\varepsilon = 0.01$ | no: a negative result of mine; the data must leave the visited set |
| 8 | `t8_jacobian_affine.py` | affine lift in the Jacobian chart | 0.49, 0.30, 0.057, 0.040, 0.042 for $k = 3..7$ (PCA: 0.039, 0.0082, 0.0050, 0.0046, 0.00011); stopped at the time limit in the last block | no: worse than PCA; a global linear fibre direction does not exist here |
| 9 | `t9_tracking.py` | can the error off the set be tracked or corrected? | error $\le 1.53\,(e_0 + r)$ for every kicked state, $r = \max\lvert z - D(\pi z)\rvert$ (no solve); first-order correction $\Phi(D(\pi z)) + D\Phi\,(z - D(\pi z))$: $1.9\times10^{-2} \to 2.0\times10^{-4}$ at $\varepsilon = 0.01$, $9.4\times10^{-2} \to 4.9\times10^{-3}$ at 0.05 | the constructive result (Suggestions 1–2) |

Details of test 3 (reduced ChebTT, six worst of the 20 rollout trajectories; dex):

| trajectory | measured worst (step) | first order, exact $J$ | first order, surrogate's $J$ | chart part / normal part |
|---|---|---|---|---|
| 7 | 0.290 (145) | 0.302 | 0.285 | 0.292 / 0.007 |
| 18 | 0.284 (165) | 0.235 | 0.281 | 0.283 / 0.009 |
| 19 | 0.199 (170) | 0.242 | 0.229 | 0.202 / 0.012 |
| 11 | 0.197 (111) | 0.153 | 0.194 | 0.194 / 0.005 |
| 14 | 0.196 (170) | 0.191 | 0.219 | 0.196 / 0.006 |
| 6 | 0.165 (105) | 0.097 | 0.164 | 0.161 / 0.004 |

Tests 7 and 8 were my own hypotheses for a fix and failed; they are reported so the idea is not tried again as stated.

## Suggestions

Towards a reduced surrogate whose rollouts hold the tolerance, with error tracking, in order of cost:

1. **Take the data fit as the base, and choose it by the two rollout conditions, not by the one-step error (no solver call).** On the
   training chains compute, for each candidate degree, the chained rollout (as `chebtt_roll` does) and $\max\lVert D\Psi\rVert$ or the
   product of the surrogate's chart Jacobians (Theorem 8's $L_g$, exact for a polynomial). Degree 6 passes both on HIRES ($1.5\times10^{-2}$
   on the test rollouts), degrees 7–8 fail the second; a rollout-weighted or Jacobian-penalised fit (penalise $\lVert Dg\rVert$ beyond the
   exact chain's) is the natural refinement to gain the last factor 1.5. This is the cheapest route to the tolerance on the visited
   set and should be tried before any further ChebTT build.
2. **Track the error with the decoder residual, and refuse on it.** $r(z) = \max\lvert z - D(\pi z)\rvert$ costs one polynomial evaluation;
   on every kicked state tested the error was at most $1.53\,(e_0 + r)$ (test 9), which is Proposition 8.2's middle term made
   computable, with the constant $\lVert D\Phi\rVert$ that `dynamics` already measures. Refusing at $r >$ tolerance/4 keeps the one-step
   tolerance by construction; the refused share in a split simulation is then the measured cost of having no off-set model. Carry
   $e_0 + \mathrm{Lip}\cdot r$ through T2's recursion for a running error estimate along a rollout.
3. **Add a first-order normal correction (an ISAT record on the reduced chart).** $\Psi_1(z) = G(\pi z) + B(\pi z)\,(z - D(\pi z))$ with
   $B(u) \approx D\Phi(D(u))$. With the exact Jacobian the kicked error fell 94-fold at 0.01 dex and stayed within half the tolerance
   at 0.05 dex (test 9). $B$ is a $d \times d$ function of three chart coordinates: fit it from Jacobians at a few hundred visited
   states (16 solver calls each on HIRES, or the solver's own sensitivities), with the same data-fit machinery. Its leading error is
   second order in $r$, so item 2's indicator becomes $e_0 + c\,r^2$. This addresses Proposition 8.2 directly, which a better chart
   alone does not (tests 7–8).
4. **If the solver-queried ChebTT is kept,** restrict its domain to where the decoder is supported (a mask or a refusal by distance
   to the visited chart values, or a box in coordinates where the data fill it, such as the two initial-condition parameters and
   time mapped through the chart): 91–96% of the present box is extrapolation and costs 5 times more per call (H4).
5. **POLLU:** more trajectories before any atlas (a six-parameter family needs several hundred for a cubic in 7 coordinates), a
   decoder at $k = 7$–8 of degree 2–3 chosen by held-out trajectories, then items 1–3. Estimate dimensions from fixed-step
   cross-sections, never from nearest-neighbour balls along trajectories.
6. **Protocol:** per-case records with provenance for every run quoted; predictions labelled by kind; a baseline row (data-only
   fit) in every table; model choices on training or validation states only.

## Overall assessment

The measurements in 8.4 and 8.5 are careful and almost all recompute; the Fisher–KPP claim survives a harder test than the memo gave
it, and the memo's reading of the HIRES rollout as the dynamics' amplification is right. The demonstrations overreach in what they
conclude. On the visited set the reduced ChebTT is dominated by a free least-squares fit on the same chart; off it, the surrogate
(any function of three chart coordinates) errs by twice the kick, because the visited set is a family of trajectories and not an
attracting manifold; and the POLLU verdict comes from a biased dimension estimate. The useful output of this round is constructive:
a solve-free residual tracks the off-set error within a factor 1.5, and a first-order normal correction removes it to second order.
Those two, on top of a data fit selected by the rollout conditions of Theorem 8, are the shortest path to a reduced surrogate that
holds the tolerance in a rollout and knows when it does not.
