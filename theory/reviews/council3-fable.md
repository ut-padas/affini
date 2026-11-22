# Council review, round 3 (Fable): what the third round's records say, and what to test next

Reviewer: Claude Fable 5.2, one of three independent council members. Material: `theory/SYNTHESIS.md` at commit `125b58e`
(sections 3.17-3.22, 9.8, 10), my round-2 review, the records under `theory/panel/data/` and the scripts that wrote them,
`theory/MEMO.md` (E, T2, T5, 8), `src/benchmark/step_maps.py`. My own checks are read-only and post hoc; the eleven scripts and
their outputs are in `/tmp/council3-fable/` (not committed) and are listed in the appendix. None was pre-registered, so each is a
reading to re-run with a committed prediction. Two of them failed to show what I expected (the accumulation-aware fits of check C,
the local fits of check H) and are reported as such.

## Summary of the recommendation

The records and eleven checks change four of 9.8's readings, and the changes point the same way: **the next phase is limited by
the training data's size and by the form of the gate, not by the model family.** (1) POLLU is not accumulation-limited: with 240
more training trajectories (41 s of solver time) the same degree-3 fit at the same chart has a one-step median of 1.7e-5 instead of
2.0e-4 and an ungated rollout of 2.8e-3 instead of 3.9e-2. Nelson is bias-limited at degree 2 and data-limited at degree 3; only
primordial resists both data and local fits. The fixed budget of 60-100 trajectories has become the experiment's main constraint
and should be replaced by a budget in solver seconds with a learning curve per problem. (2) The gate's product form
`(1 + b) e_hat` is what refuses HIRES: the typical bound `b` is the square root of the rank (about 20), so any calibration score
above 2.4e-4 refuses half the training states. An additive envelope `a0 + a1 b + a2 rho` (`rho` the offset from the bundle), which
is the form the tube's decomposition gives, refuses 0% of HIRES's plain inputs instead of 14.5% and 38% of weakly mixed ones
instead of 73%, with every accepted defect below half the tolerance. (3) The normal term's cost on the bundle is a radius effect:
fitted as a residual after the on-bundle fit, only on tube samples within 3e-3 dex of the bundle, and switched on beyond the
decoder's own residual, it leaves the plain fit untouched to every digit and cuts the median defect 20-37x inside that radius; it
does not move the maximum, so it lowers accepted errors (2-3x), not refusals. (4) "A sensitivity rotation gives 7-14x on a full
chart" is an artefact: CHEMAKZO's three tube arms have identical selection rows (a total-degree space is invariant under a
rotation); what helps there is tube data in the fit (selection chains 94-370x lower at degrees 6-7), and the selection's cost tie
picked a different degree. For speed: the gate is 85-92% of the evaluation and a 32-column sketch of it is 10x cheaper; every
quoted speed-up is against SciPy's per-call overhead and needs one compiled baseline; KIDA's step is 42% dense LU and 28%
right-hand side, not "its right-hand side". Tensor trains: the fitted maps' coefficient tensors need 6-11x more coefficients as a
train than as a total-degree set, and shared outputs cannot pay because the prediction is already 1 us; run one sample-efficiency
test on the two problems with expensive solves and otherwise stop.

---

## What was acted on from round 2

Done: the repair (truncated-SVD fit, one factor, batched projection, cost-based selection, early stopping), the three budgets
recorded, the tube experiment with a volume control, paired columns with D = 0.25 and unsorted cells, the index sets, KIDA's
coverage curve and solver audit, laws verified off the data. Not done: the offset from the bundle recorded in the coupled tests
(round 2, 1.4 and 5.3; p11 records no offset, which is why its C2 cannot be interpreted); the adaptive set grown by held-out gain
(p12 grew it by in-sample correlation, half the set per round); the trajectory-fitted tensor train; KIDA's `dt` and horizon from
an MFC run and the rank of the key species' futures; the local table with a successor model, offline checked growth and online
audits; a compiled-solver estimate beside every speed-up. One arm differs from its label: T0+ adds `r5_data_scaling.extra_pairs`,
which are 1,600 states of the *initial law* (`P.initial`, one step each), not pairs along trajectories, so it is a first-step-data
arm, not a control for data volume on the bundle. That explains U4's failure on bruss_d20, whose high-rank part is its first
steps.

---

## 1. The plan from here; which of 9.8's readings do not follow

Findings, most important first.

### 1.1 (critical) Reading 9.8.4 on POLLU is wrong: it is limited by the number of trajectories, not by accumulation

- Check D: R4e's configuration (k = 13, degree 3, 560 terms) refitted with the benchmark's 100 training trajectories plus 240
  fresh ones (`P.initial(240, 4321)`, 14,160 solves, 41 s), no gate, scored on 20 test trajectories:

  | trajectories | degree | one-step median / q99 / max | rollout worst |
  |---|---|---|---|
  | 100 | 3 | 2.0e-4 / 5.0e-4 / 7.0e-4 | 3.89e-2 |
  | 340 | 3 | 1.7e-5 / 9.3e-5 / 3.1e-4 | 2.75e-3 |
  | 340 | 4 (2,380 terms) | 5.7e-6 / 1.6e-5 / 7.7e-5 | 1.11e-2 |
  | 100 | 4 (rank 1,601 of 2,380) | 4.6e-2 / 0.15 / 0.18 | overflow |

  The fit has 560 coefficients and 100 independent trajectories (the 5,900 pairs along them are strongly dependent). The
  accumulation budget `tol / (2 A_n)` = 2.1e-5 is what the *median* defect must reach, and data reach it.
- Check B says what the accumulation is: the chain error grows linearly (6.6e-4 per step to 3.89e-2 at step 58), the gate
  refuses nothing, and on the worst component the chain error (+3.89e-2) equals the sum of the chain's signed own-input defects
  (+3.39e-2). The defects keep their sign; the chain's offset from the bundle does not grow (median 1.5e-4 to 2.7e-4). So
  Proposition 2.2 with neutral propagation describes it, and the lever is the size of the signed defect.
- Check C: fits aimed at the sum fail the bar. Window-sum rows (10,800 rows, weights 0.3-3) give 2.7-3.2e-2; the chain's own
  inputs added twice (4,000 solves) give 2.5e-2; both give 2.1e-2: 1.4-1.8x where 5x is needed. The sum of signed one-step
  errors along *test* trajectories stays 2e-2 in every variant: the bias is a generalisation error across trajectories, which
  only more trajectories (or fewer coefficients) remove.
- Consequence for p12's reading "POLLU is accumulation-limited" and for 9.8.5: the budget is not "too strict"; it is the
  correct target for the median, and it is wrongly tested against a calibrated maximum (section 3).

### 1.2 (critical) HIRES is refused by the gate's product form, not by the out-of-span term and not by "the calibrated maximum" as such

- `sum_i b(x_i)^2 = N trace(hat matrix) = N r`, so over the fitted states the root mean square of `b` is `sqrt(r)`: 20 at rank
  413. The gate accepts when `(1 + b) e_hat <= tol / 2`, so `e_hat` must be below `tol / (2 (1 + sqrt(r)))` = 2.4e-4 to accept a
  typical training state. R4e's HIRES has `e_hat` = 2.47e-4 (`records/r4e/hires.json`): accept needs `b < 19.2`.
- Check A (fit on 66 trajectories, calibrated on 34 with their mixed states, as generic4 does): `e_hat` 1.82e-4, accept needs
  `b < 26.5`; on plain test inputs 14.5% are refused, **all by the bound term, none by the out-of-span term, `b_max` or the
  residual term**; the accepted maximum is 1.0e-7. Section 3.18's sentence "the out-of-span term refuses 60-80% of the selection
  chains' early states" is not what the test inputs show; I did not attribute the selection chains' refusals.
- `e_hat` is `max e / (1 + b)` over calibration states that include mixed ones, whose defect is set by their offset from the
  bundle and does not scale with `b` (the chart cannot see the offset). The product then charges every plain state with the
  mixed states' defect times its own `1 + b`. Section 3 gives the form that follows from the theory and its result (plain
  refusal 0%).
- The same mechanism, not "the normal term costs accuracy", refuses all of HIRES under T1: its `e_hat` is 1.67e-3
  (`records/p10/hires_T1.json`), so accept needs `b < 2.0`, while `b >= 1` everywhere and is about 13 (the square root of its
  rank, 170) on a typical state.

### 1.3 (critical) Reading 9.8.3's last clause is an artefact: a rotation of a full chart changes nothing; tube data do

- A total-degree polynomial space is invariant under an invertible linear change of coordinates, so on a full chart (k equal to
  the free dimension) T2's fit is T1m's whenever the design has full rank, and T1 has no normal coordinates to add.
  `records/p10/chemakzo_{T1m,T1,T2}.json`: the selection rows are identical at degrees 1-5 (chain worst 5.26e-3, 1.17e-3,
  3.83e-4 at degrees 3, 4, 5 in all three; they separate only from degree 6, where the truncation differs). T1m and T1 chose
  degree 4 at an estimated cost of 4 us; T2 chose degree 5 at 4 us: a tie decided by timing. The "7-14x" compares degree 5
  with tube data against degree 5 without.
- What the same records do show is larger and was not reported: **with 1,600 mixed pairs in the fit, CHEMAKZO's plain selection
  chains fall from 1.40e-3 to 3.83e-4 at degree 5, from 4.71e-3 to 5.0e-5 at degree 6 and from 5.23e-3 to 1.4e-5 at degree 7**,
  with a solver share of 0.000 instead of 0.009 and 0.165 at degrees 5 and 6. Without tube data the chains saturate near the
  margin (4.7-5.7e-3 at degrees 6-9) however small the calibration error (7e-10): chain states leave the trajectories and the
  polynomial is unconstrained there. On full charts the tube is a statement about the *data*.
- Robertson's "U5 fails, 7-15x" is the same selection effect: T1m, T1 and T2 chose degree 3 where T0 chose 5; at degree 3 their
  chains equal T0's (2.52e-4 against 2.54e-4). All costs are 1-2 us, so the choice is timing noise.
- Fix the rule: among served configurations within a factor 1.5 (or 2 us) of the lowest cost, take the smallest chain error.

### 1.4 (major) Reading 9.8.3's "up to 3,300x in the maximum" compares blow-ups, and the normal term's cost is a radius effect

- Mixed inputs are far from the bundle in the log variables: the offset `rho(x) = max |x - D(u(x))|` (D the chart's decoder) is at
  most 6.4e-4 dex on plain test states; at w = 0.01 its median is 8.5e-4, its 90th percentile 0.11 and its maximum 0.38; at
  w = 0.1 they are 7.7e-3, 0.37 and 0.71 (check A2). A trace species takes its partner's abundance. T0's ungated maxima of 0.35
  and 300 (`p10b`) are a degree-12 polynomial at such states (300 is the output clip); T1 at degree 8 with those states in its
  fit does not blow up. That is coverage of far states, not ownership of a first-order term. Within 3e-3 dex of the bundle T0's
  maximum is 2.2e-2 and 1.1e-2, and no variant of the normal term I fitted moves it.
- The cost on the bundle comes from fitting a linear term to states 0.1-0.7 dex away, where the defect is not first order.
  Check A2, a residual model `B(u) delta` (affine in u, 20 columns) fitted after the on-bundle fit:

  | fitted on tube samples with offset | plain median / max | mixed, offset <= 3e-3: median / q99 / max (w = 0.01; 0.1) |
  |---|---|---|
  | none (T0) | 8.6e-9 / 1.4e-5 | 3.4e-4 / 4.0e-3 / 2.2e-2; 9.2e-4 / 4.4e-3 / 1.1e-2 |
  | any (as T1) | 1.1e-3 / 5.2e-2 | 1.0e-3 / 1.7e-2 / 5.8e-2; 1.0e-3 / 2.3e-2 / 6.4e-2 |
  | <= 3e-2 | 5.7e-5 / 2.3e-3 | 6.1e-5 / 8.0e-4 / 1.8e-2; 6.3e-5 / 1.1e-3 / 8.6e-3 |
  | <= 3e-3 | 1.4e-5 / 3.2e-4 | 1.7e-5 / 8.6e-4 / 2.0e-2; 2.5e-5 / 1.3e-3 / 1.3e-2 |
  | <= 3e-3, switched on beyond the decoder's residual | 8.6e-9 / 1.4e-5 | (section 2) |

  So the median falls 20-37x and the 99th percentile 3-5x inside the radius, the maximum does not move, and the plain fit can be
  kept exactly. Section 2 builds on this.

### 1.5 (major) Reading 9.8.4 on Nelson and primordial: three different limits under one label

Check G (R4e's chart, no gate, fresh trajectories added):

- **Nelson** is bias-limited at degree 2 and data-limited at degree 3. Degree 2 (153 terms): median 2.8e-2 at 60 trajectories,
  2.7e-2 at 120. Degree 3 (969 terms, which R4e's term rule excluded at 1,152 fitted pairs): 1.4e-2 at 75, 1.2e-2 at 90,
  7.3e-3 at 120 trajectories; its 99th percentile is still 0.35. Its persistence error is 0.25 dex per step at the median, 25
  times the tolerance: the step is long.
- **primordial** improves slowly with data and has rare jumps. 60 to 300 trajectories: degree 3 median 2.3e-2 to 9.9e-3,
  degree 4 to 6.0e-3; the worst from step 5 falls from 0.25 to 3.8e-2 at degree 3; the first steps stay at 0.5-1.3. Its
  persistence error has a median of 2.5e-2 and a maximum of 11 dex in one step. Local fits (check H: ridge on 32-256 nearest
  pairs, degrees 1-2) are no better than the global degree 4 on the same 300 trajectories (median 6.2e-3 against 6.0e-3, 99th
  percentile 0.35 against 0.074).
- "No index set helps" (3.21) is measured by the best held-out *worst*, which is set by a handful of first-step states (0.03-0.3
  on problems whose rollouts are 2e-3). By served rollout and cost the hyperbolic cross is the better set on bruss_d20 (2.1e-3
  at 13.8 us against 3.4e-3 at 31 us, `p12_basis.json`) and the worse on burgers_d20 and fisher_kpp_d10. The conclusion that
  the unserved four are not term-limited stands for POLLU, Nelson's degree 2 and primordial for other reasons (1.1 and above).

### 1.6 (major) Reading 9.8.1: the speed-up is real against SciPy and unmeasured against a solver

- Every `solve_us` is `solve_ivp` called once per state from Python (`baseline-scipy-radau-cpu1.json`, whose own note says
  "SciPy's per-call Python overhead dominates small systems"). Robertson's step is 52 right-hand sides and 9.4 factorisations
  of a 3 x 3 matrix in 1,218 us, 23 us per right-hand side; bruss_d20's is 70 and 7.6 of a 20 x 20 matrix in 1,589 us. A
  compiled Radau does that work in roughly 10 us and 100 us (my estimate from the counts, not measured), so 778x and 14x become
  about 8x and 2-4x. I asked for this estimate beside every speed-up in round 2; it is still missing, and it decides whether
  the served set is a speed result at all.
- Timings in the records differ by 2-4x between runs of one configuration: CHEMAKZO's evaluation is 22.1 us in
  `records/r4e/chemakzo.json` and 5.9 us in `records/p10/chemakzo_T0.json` (effective speed-up 54 and 202). The median "54" has
  that uncertainty. AGENTS.md's rule (pinned threads, nothing else running, interleaved rounds) applies to any number quoted.
- "No loss of accuracy": HIRES was served in R4d and is not in R4e (F1 failed), and fisher_kpp_d3 accepts one-step errors of
  7.3e-3 (`one_step_worst`, R4e) and own-input defects of 7.4e-3 to 1.1e-2 (p11), above the gate's own budget of 5e-3.

### 1.7 (major) Reading 9.8.5: the coherence budget is looser, and "refuses whole problems" is the product form again

- `records/r4e/*.json`, refused share under the accumulation budget against the coherence budget: bruss_d2 100% against 3.2%,
  bruss_d20 17.8 against 7.9, burgers_d20 31.7 against 11.0, fisher_kpp_d2 12.6 against 3.0. "No looser" holds only for the
  unserved problems, where both are 100%.
- `budget_shares` refuses when `(1 + b) e_hat > B`: once `e_hat > B / 2` it refuses everything, whatever the states' own
  errors. HIRES's plain defects have a median of 9e-9. With the additive envelope a budget of `tol / 20` refuses 2.3% of HIRES's
  plain inputs (check A3).

### 1.8 (major) Reading 9.8.6: KIDA's step is mostly linear algebra and SciPy, and the coverage law is fitted to the wrong curve

- Check E (ten dark-range states, one thread): one right-hand side 0.161 ms, one analytic Jacobian 0.36 ms, one dense LU of
  582 x 582 2.63 ms, one sparse LU 5.08 ms (fill 249 nonzeros per row from 35). A step (0.460 s, 805 right-hand sides, 74
  factorisations, 329 internal steps): **LU 42%, right-hand side 28%, triangular solves 7%, the remaining 23% SciPy's own
  per-step Python.** p9 measured counts and totals and inferred the split from the sparse arm being slower; the sparse arm is
  slower because SuperLU with this ordering is slower than LAPACK at n = 582.
- p8's extrapolation fits a power law to the coverage, which saturates. The median error is the stable curve: 0.047, 0.032,
  0.027, 0.0215 dex at 20, 40, 80, 160 tracers, exponent -0.31 to -0.37, so the median reaches 0.01 near 1,300-1,900 tracers;
  with the worst at 2 dex, a coverage near 95% needs the median several times below 0.01: 0.005 is about 1e4 tracers at that
  exponent and 0.002 about 1e5 (section 8).

### 1.9 (minor) Readings 9.8.2 and p11's C2

"Coupling does not compound through neighbours" holds (closed against forced within 1.37x). "The column's worst error tracks the
own-input defect" holds on seven problems; on bruss_d2, unsorted, D = 0.005 the worst is 6.4e-3 with an own-input defect of
1.4e-3 (`p11_paired_columns.json`), and on bruss_d20 and CHEMAKZO the inherited error is of the worst error's size, so there
the column accumulates. C2 failed because a closed column of 40 cells with one initial law homogenises: stronger exchange makes
every cell the mean state, and the errors fall with D on most rows. The column does not produce sustained offsets, and it does
not record them. A coupled test must keep supplying unlike states (section 9, step 4).

### The order

Learning curves and a solver-time budget; the envelope gate with a sketched bound; the residual normal term and tube data at fixed
degree; a coupled test with sustained, recorded offsets; one compiled baseline and KIDA's solver; KIDA's no-fit measurements; one
tensor-train test. Section 9 gives costs and decisive readings.

---

## 2. The composite tube model

**Recommendation: the normal term as a residual model fitted after the on-bundle fit, on tube samples inside a stated radius,
multiplied by a continuous switch that is zero where the decoder cannot tell a state from the bundle.** A blended joint fit (T1)
is what failed; a hard switch is discontinuous; a two-level design is this with the levels named.

### 2.1 Theory

Notation: chart `u(x)`, decoder `D`, base point `p(x) = D(u(x))`, offset `delta(x) = x - p(x)`, `rho(x) = |delta(x)|`,
`rho_0` the largest `rho` over the fit's plain states (the decoder's residual: 8.8e-4 on HIRES), `rho_1` the tube's radius.

`Psi(x) = G(u) + omega(rho) B(u) delta`,   `omega = 0` on `[0, rho_0]`, a smooth step to 1 on `[rho_0, 2 rho_0]`.

- Stage 1: `G` by least squares on the plain pairs, as now. Proposition 5.1: `|Phi(x) - G(u)| <= (1 + Lambda_1(u)) e_1` for
  bundle states.
- Stage 2: `B` by least squares of the residuals `Phi(x_i) - G(u_i)` on the features `delta (x) phi_1(u)` (offset coordinates
  times an affine basis of the chart: `(free - k)(k + 1)` columns; 20 on HIRES, 98 on POLLU, 110 on bruss_d20), over tube
  samples with `rho_0 < rho <= rho_1`. Proposition 5.1 applies to stage 2 as a fit of the function `x -> Phi(x) - G(u(x))` in
  the space `V_2 = span{delta_j phi_m(u)}`:

  `|Phi(x) - Psi(x)| <= (1 + Lambda_2(x)) e_2`,   `e_2 <= eps_1 + eta rho_1 + (H_Phi / 2) rho_1^2`,

  with `eps_1` the on-bundle error (it is the part of the residual `V_2` cannot hold), `eta = sup |D Phi(p) - B*(u)|` the error
  of the best affine-in-u normal derivative, and `H_Phi` the map's second derivative across the tube. The three terms are the
  tube decomposition of round 2 with the radius explicit.
- **What the radius trades.** `e_2` grows as `H rho_1^2`, and least squares spreads it over every state in the fit, plain ones
  included. This is the measured cost: fitted on every mixed state (offsets to 0.7 dex) the plain median is 1.1e-3; within
  3e-2, 5.7e-5; within 3e-3, 1.4e-5 (1.4). The radius where a first-order model holds is `rho_1 <= sqrt(2 B / H)` for a
  per-step budget `B`; it is a property of the map, to be measured, and beyond it the answer is refusal.
- **The switch.** For `rho <= rho_0`, `Psi = G(u)`: every bound and every record of the on-bundle fit is unchanged (measured:
  8.6e-9 / 2.8e-7 / 1.4e-5, T0's values). `Psi` is continuous, and its Lipschitz constant across the ramp is at most
  `Lip(G o u) + |B| (1 + rho |omega'|) <= Lip(G o u) + 3 |B|`, which Proposition 2.2's remainder needs to be finite. A hard
  switch at an offset threshold has a jump of `|B delta|` there and no such constant.
- **What the gate certifies.** `Lambda_2(x)`, the Lebesgue function of the stage-2 design at the query, is small only if tube
  samples exist near `u(x)` with offsets in the direction of `delta(x)` and at least its size. So the gate says "the first-order
  model was identified here"; it does not bound `H`, which is calibrated (section 3). Theorem 8 does not apply (the map is not
  constant on the chart's fibres); Propositions 2.1-2.2 with the composite's defect do.
- **Sampling.** `rho(x)` needs no solve, so tube samples are drawn at the radius wanted: generate mixed states (or the coupled
  test's own inputs), keep those with `rho_0 < rho <= rho_1`, solve only those. Mixing at w = 0.01 and 0.1 puts 35% and 65% of
  HIRES's states outside 3e-3; those solves bought nothing.
- **Full charts.** There is no `delta`; the composite reduces to tube data in the one fit (1.3), and Proposition 5.1's Lebesgue
  function over the tube samples is the whole statement.
- **Not recommended.** The sensitivity chart: on full charts it is the identity on the fit (1.3); on reduced charts a global
  linear sensitivity subspace of the chart's dimension does not exist on HIRES (T2's plain median is 5.1e-3, `p10b`) or POLLU
  (T2's calibration score is 1.7e-3 and it refuses everything), and `B(u)` is its local form. The offset in the transported
  (linear) variables: tested in check A2 (`10^Psi = 10^G + B n`, `n` the concentration offset), medians equal to the log form's
  inside the radius and blow-ups of 6 dex outside it.

### 2.2 What my check found, and what it did not

Check A3 (HIRES; composite with `rho_1` = 3e-3 and 3e-2, the envelope gate of section 3, budget `tol / 2`):

| model | plain: refused, accepted max | w = 0.01: refused, accepted median / max | w = 0.1: refused, accepted median / max |
|---|---|---|---|
| T0, product gate | 14.5%, 1.0e-7 | 72.6%, – / 1.3e-3 | 90.4%, – / 1.3e-3 |
| T0, envelope | 0.0%, 1.4e-5 | 38.5%, 3.1e-4 / 4.7e-3 | 65.3%, 9.3e-4 / 4.9e-3 |
| composite 3e-3, envelope | 0.0%, 1.4e-5 | 38.7%, 2.2e-4 / 2.1e-3 | 66.0%, 2.6e-4 / 3.0e-3 |
| composite 3e-2, envelope | 0.0%, 1.4e-5 | 37.6%, 2.3e-4 / 2.3e-3 | 63.0%, 2.1e-4 / 1.5e-3 |

The composite lowers what is accepted (maximum 2-3x, median up to 4x) at the same refusal. It does not lower the refusal,
because the envelope's slope in `rho` (1.1-1.6) is set by a few states with a defect of 1-2e-2 at offsets below 3e-3, which are
not first order. I did not identify them. Three hypotheses, each a column in the next run: they are early steps (curvature of
the transient); their offset lies along a sensitive direction that the max-norm `rho` hides (then `|B(u) delta|`, the size of
the predicted correction, is the right regressor for the gate); or the decoder's base point is wrong there.

### 2.3 The experiment that decides

Problems: HIRES, POLLU (with 340 trajectories, 1.1), bruss_d20, burgers_d20, fisher_kpp_d10; CHEMAKZO, Robertson, lorenz96_d4 for
tube data on full charts. Degree fixed at R4e's (no selection by cost, 1.3). Arms: T0; tube data in the fit (T1m) at fixed
degree; the composite at `rho_1` in {3e-3, 1e-2, 3e-2} with constant and affine `B`; each with the envelope gate. Tube samples
drawn by rejection on `rho`, 1,600 kept per arm. Scores: plain and mixed inputs by offset decile (median, 99th percentile,
maximum, refused), the closed column, the coupled test of section 9 step 4, and the large-defect states listed with step index,
`rho` and `|B delta|`.

Predictions to commit:

- P1 (replication): on HIRES the composite's plain scores equal T0's to three digits at every radius.
- P2: inside `rho_1` = 3e-3 the median mixed defect falls at least 10x on HIRES and at least 3x on POLLU and bruss_d20; the
  maximum inside the radius changes by less than 2x on all three.
- P3: at equal refusal (within 3 points) the accepted maximum on mixed inputs is at least 1.5x lower than T0's on HIRES and
  POLLU.
- P4: on the three full charts, tube data at fixed degree lower the plain rollout at least 3x at degrees where the fit has more
  than 250 terms, and change it by less than 2x below 60 terms.
- P5: more than half of the states with a defect above 5e-3 at `rho <= 3e-3` are in the first ten steps.

What changes the plan: P3 failing means the normal term buys only a median and belongs with the accumulation accounting
(section 4), not the gate. P5 holding means the burn-in rule removes the envelope's worst states and the refusal falls with it;
P5 failing sends the next run to `|B delta|` as the gate's regressor.

---

## 3. The gate after the repair

**Yes: an additive envelope in the bound and the offset, calibrated for the maximum by a linear program.** It is the form the
theory gives, it is state-based (no step index, which a coupled run does not have), and on HIRES it removes the plain refusals
and halves the mixed ones with every accepted defect inside the budget.

### 3.1 Theory

For a chart model `Psi = G o u` and a query `x` with base point `p`:

`|Phi(x) - Psi(x)| <= |Phi(p) - G(u)| + |Phi(x) - Phi(p)| <= (1 + b(u)) e* + L_n rho(x) + (H / 2) rho(x)^2`.

The first term is Proposition 5.1 on the bundle; `L_n` is the map's Lipschitz constant across the bundle (at most about 1 for a
map that contracts normal offsets, and the envelope measures 1.1-1.6 on HIRES in log variables). The product `(1 + b) e_hat`
multiplies the second term by `1 + b`, which is wrong by a factor `sqrt(rank)` on average (1.2). The gate:

`bound(x) = a_0 + a_1 b(x) + a_2 rho(x) + a_3 rho(x)^2`,   accept when `bound(x) <= B`, `b <= b_cap`, `rho <= rho_cap`,

with `a >= 0` minimising `sum_i bound(x_i)` subject to `bound(x_i) >= e_i` on every calibration state (plain and tube), and the
caps the calibration ranges. With the composite, `rho` is replaced or joined by `|B(u) delta|` and `Lambda_2`.

What it certifies: on the calibration states it is an upper envelope by construction; `a_1` and `a_2` are sample lower bounds
of `e*` and `L_n`, so the statement for a new state is distributional, as every calibrated gate's is (T4). Two honest
additions: the conformal level at the unit of a trajectory (`1 - 1 / (n_cal + 1)` for "no accepted state of a new trajectory
exceeds its bound"), which says how many calibration trajectories a stated level needs, and the online audit of round 2 (solve a
share `q` of accepted queries; after `m` clean audits the exceedance rate is below `3 / m` at 95%). Per-regime calibration and a
calibration per offset are this envelope made piecewise; fit it piecewise in `rho` if the linear program's slack is large in one
offset range. An error model in the step index is not a gate: the step index is not a function of the state.

### 3.2 Measured (check A3, HIRES, held-out trajectories for calibration, 20 test trajectories)

`e <= 9.0e-5 + 5.2e-8 b + 1.5 rho`. Refused at budget `tol / 2`: plain 0.0% (product gate 14.5%), w = 0.01 38.5% (72.6%),
w = 0.1 65.3% (90.4%); accepted maxima 1.4e-5, 4.7e-3, 4.9e-3, none above the budget. At `tol / 20`: 2.3%, 72%, 92%, accepted
maxima 1.3e-6, 4.1e-4, 4.0e-4. The product gate's accepted maximum is lower (1.3e-3) because it refuses twice as much.

### 3.3 The accumulation budget as an account, not a threshold

A uniform `tol / (2 A_n)` per step is the worst case for a cell that errs at the budget on every step. Proposition 2.2 gives the
honest alternative: each cell carries `S = sum_j g_j bound(x_j)` over its accepted steps (`g_j` the propagation factor; 1 for
neutral dynamics, the benchmark's growth factor otherwise), a refused step adds nothing (Corollary 5.3 (d)), and a cell whose
account reaches `tol / 2` is stepped by the solver from then on. Cells on the bundle (bound 1e-4 on HIRES) never spend it; cells
in a mixing zone do, and the solver share then *is* the cost of the guarantee. The inequality `|e_n| <= S_n` is checkable on
every cell of every recorded chain.

### 3.4 The experiment

Rerun R4e's 18 problems with the envelope (and the sketched bound of section 7), the budget `tol / 2` and the account.
Predictions to commit: HIRES is served on plain rollouts with under 3% refused; fisher_kpp_d3's accepted one-step worst falls
below 5e-3 (its product gate accepts 7.3e-3 now); the lattices' refused share on mixed inputs is at most R4e's on every problem
and at least 1.5x lower on three; no served problem's accepted test maximum exceeds the budget; `|e_n| <= S_n` holds on at
least 99% of cells and steps of the plain rollouts. If the accepted maximum exceeds the budget on more than one problem, the
envelope needs an inflation factor, calibrated at the trajectory level, before anything else is built on it.

---

## 4. Accumulation

The question's premise (POLLU) is answered by data (1.1). What the checks say in general:

- **The mechanism is a signed sum** (check B): chain error and the sum of signed own-input defects agree on the worst component
  (3.89e-2 against 3.39e-2); the map damps the rest (the largest signed sum over components is 9.6e-2).
- **Periodic exact steps do not help, by theory and by measurement.** An exact step adds no defect and propagates the earlier
  ones (Corollary 5.3 (d)); along a neutral direction nothing is removed. POLLU: an exact step every 10 gives 2.64e-2 at 8%
  solver, every 5 gives 1.90e-2 at 19%: no better than proportional to the steps replaced.
- **Fits aimed at the sum give 1.4-1.8x** (check C), below the bar. A multi-step loss is the exact first-order objective
  (`e_n = sum G delta` is linear in the coefficients), but its gain is bounded by how much of the bias is a property of the
  training trajectories, and on POLLU almost none is: the signed sum on test trajectories does not move. Do not build
  chain-fitted charts or residual correctors now.
- **Invariants** remove the error along the conserved directions only; POLLU's worst component is not one of them.
- **What lowers accumulated error is a smaller signed defect where the chain goes**: more trajectories (12x on the median,
  14x on the rollout, 41 s), then tube data on full charts (1.3), then the normal term's median gain (20-37x inside its radius
  on HIRES), which is the composite's real use (2.2).
- **What to record**: for every served problem the largest `|sum of signed defects|` over windows of held-out trajectories,
  beside the one-step maximum. It is the quantity selection chains already test, it is what the account of 3.3 bounds, and it
  separates POLLU at 100 trajectories (2.1e-2) from POLLU at 340 before any rollout.

Where data are expensive (Nelson at 78 ms a solve, KIDA at 0.46 s) the multi-step fit is worth one test, after their learning
curves: commit that it lowers the rollout at least 2x at equal data, and drop it otherwise.

---

## 5. Nelson and primordial

Three limits, three tests (1.5):

- **Nelson: more data at degree 3, then a smaller space.** The cheapest decisive test is the learning curve continued: 240 and
  480 trajectories (7.5 and 15 minutes of solves, stored as records) at degrees 2 and 3 with the envelope gate. Prediction to
  commit: the degree-3 median falls below 2e-3 at 480 trajectories and the 99th percentile stays above 1e-2, so Nelson is
  served with a refused share between 5% and 30%. If the 99th percentile does not fall with data, the tail is a regime (a
  jump), and the answer is refusal there. Because its solves are the panel's most expensive, Nelson is also the problem where a
  format with fewer coefficients than 969 would pay (section 6).
- **primordial: measure the map before fitting it again.** Data move the median as `n^-0.5` and local fits do not help. Two
  no-solve measurements on the 300 stored trajectories: (a) the modulus of continuity: for pairs of training inputs at
  standardised chart distance below `r`, the largest difference of their successors, as a function of `r`; a fit can hold 0.01
  only at a sample spacing where that difference is below 0.01, and the local dimension (local principal components at that
  scale) turns the spacing into a count; (b) the share of steps whose persistence error exceeds 1 dex, and whether a threshold
  on the state separates them. Prediction: fewer than 5% of steps carry every one-step error above 0.1, and a gate that refuses
  them leaves a map on which degree 3 holds 0.01 at 300 trajectories on at least 80% of the rest. If (a) shows successors
  differing by more than 0.1 dex at the closest available pairs, primordial at this step length is not a surrogate target and
  joins the oscillators.
- **Local fits (ART, AFINNI under the protocol)**: not before (a). Check H is crude (no adaptivity), but a local fit needs the
  same sample spacing as (a) measures, so (a) decides for both.
- **The generic rule**: a learning curve in whole trajectories at two degrees is the first diagnostic of any unserved problem.
  Flat in data at a degree means bias (raise the degree or change the space); falling means variance (buy trajectories, or a
  smaller space); a tail that does not fall means a regime to refuse.

---

## 6. Tensor trains and shared outputs

**Shared outputs: do not build.** The prediction is `p x d_out` multiplications, 0.1-1.1 us per state (check F); the evaluation's
cost is the gate, which does not depend on the outputs. The output mode's rank at 1e-3 is 20 of 20 on POLLU and lorenz96_d20,
10 of 10 on fisher_kpp_d10, 16-18 of 20 on burgers_d20, 9-11 of 20 on bruss_d20 only. An output core matters for one ChebTT per
output on a box, which T7 rules out at the target's scale.

**Tensor trains in the chart: the screen is against them, with a caveat.** Check F computes the TT ranks of each fitted
coefficient tensor (the unfoldings of the Chebyshev coefficients, truncated so that the function changes by at most 1e-3 in root
mean square):

| problem | k, degree | TT ranks (largest) | TT coefficients | total-degree coefficients |
|---|---|---|---|---|
| chemakzo | 5, 5 | 30 | 7,278 | 1,260 |
| fisher_kpp_d10 | 6, 5 | 65 | 44,100 | 4,620 |
| bruss_d20 | 9, 4 | 64 | 73,210 | 14,300 |
| pollu | 13, 3 | 69 | 97,596 | 11,200 |
| lorenz96_d20 | 20, 2 | 40 | 50,043 | 4,620 |

The ranks are 50-100% of the largest a total-degree tensor allows. Caveat: truncating a low-rank function to total degree raises
its rank (a product of univariate functions has rank 1 and its total-degree truncation up to `q + 1`), so the screen is biased
against the train; it says the *fitted* maps have no low-rank structure in principal chart coordinates, not that no low-rank
approximant of the step map exists.

**The one experiment that decides.** After the sketched gate the evaluation is 2-5 us at 250-715 terms, so a train cannot buy
2x in speed; its only possible gain is sample efficiency where solves are expensive. So: POLLU and Nelson, R4e's chart,
learning curves at 50, 100, 200 and 340 (Nelson 60, 120, 240) trajectories; arms: total degree 3; a train fitted by alternating
least squares at ranks 2, 4, 8 with degree 3 per coordinate and the outputs as a last core; gate: the tangent design's Lebesgue
function (round 2, section 2) inside the envelope. Scores: one-step median and 99th percentile from step 5, the signed-sum
statistic of section 4, the rollout. Predictions to commit: the train reaches POLLU's one-step median of 2e-5 with no fewer
than half the trajectories total degree needs, and on Nelson its median at 120 trajectories is above total degree 3's (7.3e-3).
Decision: if the train halves the trajectories needed on either problem, build T-a for charts of 10 or more coordinates and test
it on KIDA's chart next; otherwise tensor trains stay box schemes, and T-b, T-c and T-d are closed. CPU: about an hour, most of
it Nelson's solves, which section 5 needs anyway.

---

## 7. Speed: changes worth 2x or more

After the repair the selection takes 1-69 s per problem and is no longer worth optimising. What remains:

| change | evidence | expected | measurement that confirms it |
|---|---|---|---|
| **The gate from one product.** `W = Phi V_r`; `b` from `W / s`; the out-of-span share from `norm(Phi)^2 - norm(W)^2`, skipped when the rank is full. `gate_terms` does three `p x r` products. | Check F, us per state: bruss_d20 47.5 = design 2.1 + prediction 1.1 + gate 41.7; fisher_kpp_d10 47.3 = 1.6 + 0.6 + 43.6; fisher_kpp_d5 23.4 = 1.2 + 0.3 + 21.9; chemakzo 15.2 = 0.8 + 0.3 + 13.1. One product: 17.6, 17.5, 8.0, 5.1 us; `b` equal to 8e-9 relative | 2.3-2.5x on the evaluation at 250 or more terms | decisions identical on the test inputs of the ten served records; `evaluate_us` interleaved |
| **A sketched bound.** `b~ = sqrt(N) norm(Phi M S)`, `S` a fixed 32-column Gaussian sketch; the out-of-span share from `norm(Phi Q)`, `Q = (I - V_r V_r^T) S'`. Calibrate the gate on the sketched statistics, so the statistic deployed is the one calibrated. | 1.6, 1.1, 0.6, 0.6 us; `b~ / b` within 0.71-1.31 at the 1st and 99th percentiles | 9-14x on the evaluation (bruss_d20 47.5 to about 5 us), so the effective speed-up from step 10 rises by the same factor | refused share within 2 points of the exact gate's and accepted test maximum within the budget on every served problem; three sketch seeds |
| **One compiled baseline.** Not a speed-up: the denominator. | 1.6: SciPy spends 23 us per right-hand side on Robertson | decides which served problems are speed results | a compiled implicit integrator with a compiled right-hand side and Jacobian on robertson, bruss_d20, pollu, nelson: us per step at the benchmark's tolerances, agreement with the stored labels to 1e-6 |
| **KIDA's exact step: factorisations.** 74 dense LUs per step are 42% of it. Reuse a factorisation across step-size changes within a ratio (as VODE does), start from the previous step's step size and order instead of order 1 (329 internal steps from a cold start), or a one-step stiff method of high order (Rosenbrock, Radau) that does not restart. | check E | 2x if the count halves; more with a warm start | `nlu`, internal steps and seconds on p9's 120 states, labels within 1e-5 dex of the reference |
| **KIDA's right-hand side compiled** (a Numba loop over the reaction list instead of sparse products from Python). | 0.161 ms per call, 28% of the step; SciPy's own Python is another 23% | 1.3x alone; with the row above, 3-4x | the same 120 states; right-hand side values bit-compared or within 1e-13 relative |
| **Solver-time data budget.** Not a per-step speed-up: it converts unserved problems. | 1.1: 41 s served POLLU's fit | POLLU from 1x (unserved) to its nominal 30x or more | section 9, step 1 |
| **A GPU or batched path** for the evaluation and for refused states. | none on this machine | only at 1e5 or more cells per call | on TACC: cells per second against batch size, CPU and GPU, with the sketched gate |

Skipped as below the bar: a compiled design (2 us of 5 after the sketch), float32, reuse of factors across nested degrees, the
selection, thread tuning. Two caps to keep in view: under coupling the speed-up is at most the inverse of the refused share on
the coupled inputs (5-8% on the lattices' mixed inputs: 12-20x), so the gate of section 3 is also the largest speed lever for
coupled runs; and a sparse LU is not a gain at n = 582 with SuperLU's default ordering (5.1 ms against 2.6 ms dense), so "sparse"
needs another ordering or solver (KLU, a fill-reducing permutation computed once per network) before it is tried again.

---

## 8. KIDA's route

**A global fit at 0.01 dex on all 13 key species is not a realistic target at the data this laptop can make; accelerate the
exact solver first, measure three things without fitting, and let those choose between an in-situ table and a reduced state.**

- **The measured law.** The local arm's median falls as `N^-0.31` to `N^-0.37` (1.8). For a local-linear estimator the error
  scales as `N^(-2 / d)`, so the states the 160 tracers sample behave as a set of dimension about 6: the four driven environment
  inputs and about two chemical ones. The median reaches 0.01 near 1,300-1,900 tracers; holding 0.01 on 95% of states needs the
  median several times lower: 0.005 is about 1e4 tracers at this exponent (a local-quadratic estimator, `N^(-3 / d)`, would need
  about 2,500), and 0.002 about 1e5 (16,000). A tracer is 24 steps of 0.46 s: 1e4 tracers are 31 core-hours, a cluster job, and
  only worth submitting after the three measurements below.
- **The solver is both the fallback and the data source**, and it has 3-4x in it (section 7) with no modelling risk. Do that
  first; it also multiplies the data any later fit can afford.
- **Three measurements, no fits** (the first two are round 2's, still open): (a) `dt` and the horizon from an MFC run, with
  the persistence error over the tolerance at that `dt`; at the current `dt` persistence errs 0.167 dex at the median, 17
  times the tolerance, the Nelson regime; (b) the rank at the tolerance of the key species' futures with respect to the state
  (finite differences, 20 states, 5 steps, about 30 minutes): tens means a closed reduced state exists, hundreds means it does
  not; (c) the local dimension of the dark-range states (local principal components at the 64-neighbour scale, stored data):
  it turns the exponent above from an inference into a measurement and prices any table.
- **Then**: dimension about 6 and rank in the tens: a chart of that size with the recipe's fit, the envelope gate and tube data,
  on a cluster-sized data set; this is the global route, at a tolerance the learning curve supports. Dimension above 10 or rank
  in the hundreds: an in-situ table with checked growth and audits (round 2, section 6), whose coverage grows where the
  simulation goes; the panel's 2-minute cap cannot test it and a single MFC column can. A reduced network is a modelling choice
  outside this study; (b) says whether it can exist.
- **Stop**: model search on the 160 tracers; extrapolating coverage percentages; the sparse-Jacobian path as tried.

---

## 9. Plan, in order

| # | step | purpose | CPU cost | the result that would change the plan | stop doing |
|---|---|---|---|---|---|
| 1 | Learning curves in whole trajectories (the benchmark's set plus 1x, 3x, 7x fresh ones, kept as records) at two degrees for POLLU, Nelson, primordial, lorenz96_d20, allen_cahn_d5, HIRES, with the signed-sum statistic; from then on the data budget is solver seconds (state it per problem), not 100 trajectories. Predictions: POLLU served at 340 trajectories with rollout below 5e-3 and under 5% refused; Nelson as in section 5; lorenz96_d20's degree-3 chain below the margin by 800 trajectories. | Separate data limits from model limits before any model work | 40 min, half of it Nelson | POLLU not served with the gate on: the chain's refusals, not the fit, are its limit, and step 2 moves first. A curve flat at both degrees on any problem: that problem needs another space (step 8) | Reading an unserved problem at 100 trajectories as a property of the method; the held-out *worst* as a figure of merit; T0+ as a volume control |
| 2 | The envelope gate (3.1), the gate from one product and its sketch (section 7), the selection tie rule (1.3), the account (3.3); rerun R4e's 18 problems. Predictions in 3.4 and section 7. | An honest gate that does not refuse by construction, at a tenth of the cost | 1 h | Accepted maxima above the budget on more than one problem: add a trajectory-level inflation and re-run before steps 3-5. A sketch that changes the refused share by more than 2 points: keep the one-product gate only | The product gate; `tol / (2 A_n)` tested against `e_hat`; selection by raw cost |
| 3 | The composite and tube data at fixed degree (2.3), predictions P1-P5. | The off-bundle design | 1-2 h | P3 failing: the normal term is kept only for the account's median; P5 failing: `abs(B delta)` becomes the gate's regressor | Joint fits on mixed states at any offset (T1 as run); the sensitivity chart; mixing weights chosen without looking at the offset |
| 4 | A coupled test with sustained offsets, recorded: p11's column with (a) an inflow boundary fed by trajectories at a different step (cells of unlike age meet, as behind a front), (b) each step's neighbour drawn at a random age offset; record the inputs' `rho` distribution per step beside every score, and run the surrogates of steps 1-3. Prediction: the inputs' median `rho` stays above 1e-3 for the whole run in (a) on HIRES and POLLU (in p11 it decays), and every served surrogate's worst error is within 1.5x of its accepted own-input defect times the account's factor. | A test whose difficulty is the offset, the quantity MFC will set | 30 min | If offsets decay here too, the coupling the panel can script does not leave the tube, and the tube radius must come from an MFC run before more tube work | The closed column and chained mixing as coupled headlines |
| 5 | One compiled baseline on four problems; KIDA's factorisation count, warm start and compiled right-hand side (section 7). | The denominator of every speed-up; 3-4x on the target's fallback | 2-3 h | A compiled solver within 3x of the surrogate's evaluation on the lattices: the panel's speed results are restated, and speed work moves to problems with expensive right-hand sides (Nelson, KIDA) | Quoting speed-ups against per-call SciPy without the estimate beside them |
| 6 | KIDA without fits: `dt` and horizon from MFC (the user's), the rank of futures, the local dimension (section 8). | Choose KIDA's route | 45 min plus the user's | Section 8's two branches | Model search on KIDA |
| 7 | primordial's two measurements (section 5); Nelson's multi-step fit only if step 1 leaves it data-limited. | Close or open the two hard chemistry cases | 15 min | Successors differing by more than 0.1 dex at the closest pairs: primordial is refused whole, as the oscillators are | Index-set and local-fit arms on either before the measurement |
| 8 | The tensor-train sample-efficiency test (section 6). | Decide T-a once | 1 h, sharing step 1's data | The train halving the trajectories needed on POLLU or Nelson: build T-a for charts of 10 or more coordinates | T-b, T-c, T-d; further index sets by in-sample correlation |

Steps 1 and 2 are independent and each changes what the later numbers mean, so they come first; 3 needs both; 4 needs 3's
surrogates; 5 and 6 run alongside from the start.

---

## Appendix: the checks

All from `theory/panel` with
`SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=../..:.:../recipe ../../.venv/bin/python <script>`; each under three minutes;
scripts and outputs (`*.out`) in `/tmp/council3-fable/`. Other reviewers' runs shared the machine, so absolute times are rough;
check F's and E's ratios were taken with one thread.

- **A** `a_hires.py`, `a2_hires.py`, `a3_hires.py`: HIRES at R4e's k = 3, degree 12, decoder degree 4, fitted on 66 of the 100
  training trajectories (`generic4.build`), 34 for calibration with 3,000 of their mixed states, 20 test trajectories with
  their states mixed once at w = 0.01 and 0.1 (seed 9), solver labels cached. Offsets `rho`; the gate's four terms on test
  inputs; residual normal models in the offset's off-chart coordinates (constant to cubic in u; fitted on all mixed states, or
  within 3e-3 and 3e-2; the log and the linear-variable forms); the linear-program envelope on the calibration states with
  `b` below its 90th percentile; the composite with the smooth switch. Numbers in 1.2, 1.4, 2.2, 3.2.
- **B** `b_pollu.py`: `generic4.rebuild` of `records/r4e/pollu.json` (48 s); hybrid chains on 20 test trajectories with the
  solver from every chain state; per step the own-input defect, the defect on true states, the chain error, the running signed
  sum, the refused share, the decoder residual of chain and true states; an exact step every 10 and 5. Numbers in 1.1, 4.
- **C** `c_pollu.py`: the same configuration fitted on all 100 trajectories, gate off; window-sum rows (dyadic windows, half
  overlap, 10,800 rows); own inputs (2,000 chain states of the training trajectories solved and added, twice). Numbers in 1.1.
- **D** `d_pollu_deg4.py`: 240 fresh trajectories (seed 4321), degrees 3 and 4 at 100 and 340 trajectories. Table in 1.1.
- **E** `e_kida.py`: times of `system.rhs`, `system.jac`, `scipy.linalg.lu_factor` and `splu` of `I - 1e3 J` on one dark-range
  state, and SciPy BDF's counts on six states. Numbers in 1.8.
- **F** `f_split.py`: per problem `generic4.build` on all training trajectories at the record's configuration; times of the
  design, the prediction product, `gate_terms`, the projection on 2,000 test inputs; the one-product gate and a 32-column
  sketch; TT ranks of the coefficient tensor by the singular values of its unfoldings, outputs as the last mode, each
  unfolding truncated at a tail of `1e-3 / sqrt(m)` in the Frobenius norm. POLLU's separate gate time (63.8 us) exceeded its
  whole evaluation (47.0 us) in that run, a sign of the timing noise. Numbers in sections 6 and 7.
- **G** `g_curve.py`: fresh trajectories (seed 4321) for primordial (240) and Nelson (60) through `ode.trajectories` with the
  problem's held path; fits at R4e's chart, no gate. Numbers in 1.5. Persistence errors per step computed from the test
  trajectories inline.
- **H** `h_local.py`: primordial, for each test state a weighted ridge fit of degree 1 or 2 on its 32-256 nearest training
  pairs in standardised chart coordinates (charts of 6 and 9 coordinates), at 60 and 300 trajectories. Numbers in 1.5.
- Inline readings of `records/p10/*.json` (selection rows per arm), `records/r4e/*.json` (budgets, ranks, timings),
  `p11_paired_columns.json`, `p12_basis.json`, `p8_kida_coverage.json`, `p9_kida_solver.json`,
  `results/benchmark/data/step_maps/*/baseline-*.json`.

Not checked: which gate term refuses HIRES's selection chains (only its test inputs); the composite on any problem but HIRES;
the envelope on any problem but HIRES; chains or columns with the envelope or the composite; the sketched gate's decisions
(only its ratio to the exact bound); the compiled-solver times, which are estimates from operation counts; lorenz96_d20,
allen_cahn_d5 and the oscillators beyond their records; KIDA's rank of futures and local dimension.
