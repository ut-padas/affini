# Council review, round 4 (Fable): what R4f, p13 and p14 show, and the gate to build next

Reviewer: Claude Fable 5.2, one of three independent council members. Material: `theory/SYNTHESIS.md` at commit `8bf8868`
(sections 3.23-3.27, 9.9, 10), my round-3 review, `theory/panel/generic4.py`, `p13_learning.py`, `r4f_envelope.py`,
`p14_baseline.py` and their records. My five checks are read-only and post hoc. Their scripts and outputs are in
`/tmp/claude-1000/-home-prajwal-github-repos-affini/568029e2-a240-405f-b4dd-878afdf848c5/scratchpad/c4/` (not committed) and are
listed in the appendix. None was pre-registered, so each is a reading to re-run with a committed prediction. The machine was
shared with the other reviewers, so I quote time ratios taken inside one script, not absolute times.

## Summary of the recommendation

**R4f did not test the envelope gate I measured in round 3, and its two failures have one cause that is not the floor.** The
envelope was fitted over every calibration state with its caps "at their ranges", and at the degrees R4e serves, 10-20% of the
held-out states are whole trajectories outside the fitted data: their bound `b` reaches 1e8-1e9 and their errors reach 2-5. An
additive envelope cannot cover those and stay below the budget, so the three lattices were refused whole at R4e's degrees and
fell back to 28-56 terms; on Burgers d20 the same states drove `a1` to 3e-9 and `b_max` to 1.3e9, which switches the gate off.
Fitted inside a cap on `b` (three times the square root of the rank, or the 90th percentile), the envelope serves all three
lost lattices at R4e's degrees with rollouts of 1.5e-5 to 8.7e-4, and Burgers d20 at degree 5 with 1e-7 to 1.5e-6, in my
one-split check. Part of this is my error: round 3's section 3.1 said "the caps the calibration ranges" while my check had
fitted below the 90th percentile of `b`. **The account, by contrast, is not a bound as built**: it assumes errors propagate
with factor 1, and on HIRES under weak mixing the chain error exceeds the account on 2.4% of cells and steps with no accepted
defect above its bound (worst 1.27e-2, above the tolerance, on another partner draw); `allen_cahn_d5`'s record shows 7.7%. So
9.9.2 and 9.9.3 need rewriting, and the next gate is: a label-free cap, the envelope inside it with a trajectory-level
guarantee, and an account whose budget is calibrated on whole chains. For the expensive problems, Nelson and primordial are
bias-limited at a median of 3-6e-3 with every index set at 540 trajectories, so no gate serves them; the 2x there is the exact
solver, and LSODA already integrates all three cases where VODE fails or is slow (2.2-7x faster than the library's step, labels
within 2.2e-6). The publishable result is the refusal-first recipe on the ten served problems against the fastest exact solver,
with the negative results stated; the composite tube and "certifies coupled chains" should be dropped.

---

## 1. The plan from here; which readings of 9.9 do not follow

Findings, most important first.

### 1.1 (critical) 9.9.2, second bullet: the lattices were lost to uncapped calibration states, not to the floor `a0`

- `records/r4f/{bruss_d20,fisher_kpp_d5,fisher_kpp_d10}.json` deploy degree 2, 3 and 2 (55, 56, 28 terms), where R4e deployed
  degree 4, 6 and 5 (715, 462, 462 terms). Their deployed envelopes have `a0 = 0`; the floor is `a1 b` with `a1` = 2.7e-3,
  2.2e-3, 7.0e-3 and `b >= 1`. Section 3.25's "it needs a0 < tol / 80" describes HIRES (`a0` = 9.3e-5), not these three.
- At R4e's degrees the selection rows show refusal of everything, not an exhausted account: bruss_d20 degree 4 has a solver
  share of 1.00, fisher_kpp_d10 degree 5 has 1.00, fisher_kpp_d5 degree 6 has 0.97 with a chain of 2.0e-4. The rows' `b_max`
  are 1.7e9, 3.8e9 and 9.3e8, and their held-out worst errors 2.7, 5.0 and 4.9.
- Check A (split 0: 80 fit, 10 calibration, 10 selection trajectories) shows why. On fisher_kpp_d5 at degree 6 the
  calibration errors have a median of 3.0e-7, a 99th percentile of 1.9 and a maximum of 3.3; `b` has a median of 4.4, a 90th
  percentile of 534 and a maximum of 6.0e8. The share with `b` above three times the square root of the rank is 10-20% at
  every step, which with ten trajectories means one or two whole trajectories. The uncapped envelope is `a0` = 1.75e-2
  (fisher_kpp_d5), 9.4e-2 (bruss_d20), and `a2` = 432 (fisher_kpp_d10): above the per-step budget before any account.
- The product gate handles these states because its bound is proportional to `b`. The additive form needs them removed by a
  cap first. With the envelope fitted only on states inside a cap, same check, 40 test trajectories, gate only:

  | problem, degree | cap on `b` | envelope floor at the median state | rollout worst | solver share |
  |---|---|---|---|---|
  | fisher_kpp_d5, 6 | 3 sqrt(rank) = 63 | 8.5e-7 | 1.5e-5 | 16.1% |
  | fisher_kpp_d5, 6 | 90th percentile = 534 | 1.3e-6 | 1.5e-4 | 11.1% |
  | bruss_d20, 4 | 3 sqrt(rank) = 50 | 2.8e-5 | 6.1e-4 | 9.9% |
  | fisher_kpp_d10, 5 | 3 sqrt(rank) = 63 | 1.2e-5 | 4.3e-4 | 15.0% |
  | burgers_d20, 3 | 3 sqrt(rank) = 28 | 2.0e-5 | 9.3e-4 | 3.9% |
  | burgers_d20, 5 | 3 sqrt(rank) = 62 | 6.2e-8 | 1.0e-7 | 13.9% |

  The max-envelope account then never binds on them: the median bound times 40 steps is at most 1.1e-3, against a budget of
  5e-3, and the solver share is the same with and without the account.
- The price is the refused share, 10-16% against R4e's 5-7%, and it is the trajectories outside the data. More training
  trajectories are the lever (section 4), which p13 showed cost seconds on the lattices.

### 1.2 (critical) 9.9.2 "not conservative enough on Burgers d20" is the same uncapped fit, and it is one accepted step

- `records/r4f/burgers_d20.json`: degree 5, rank 460 of 462, `b_max` 1.3e9, `a1` 3.3e-9, `one_step_worst` 1.0e-2 equal to
  `rollout_worst` 1.0e-2. The violation is a single accepted step with an error of twice the budget, not accumulation. With
  `a1 b` negligible at every `b` the gate cannot see extrapolation; R4e chose degree 3 (84 terms) and held 1.4e-3.
- With the cap, degree 5 holds 1.0e-7 (table above) and the account is exceeded on 0.00% of cells and steps.

### 1.3 (critical) "Where the account binds, coupled chains are honest" (3.25) does not follow: the account ignores propagation

- `Account` sums bounds with factor 1 (`generic4.py:462-469`). Round 3's section 3.3 had `g_j`, "1 for neutral dynamics, the
  benchmark's growth factor otherwise"; the factor was not implemented, and `account_check` runs only plain rollouts.
- Check B rebuilds R4f's HIRES surrogate (envelope 9.31e-5, 2.29e-6, 1.49, as recorded) and runs chained mixing on 20 test
  trajectories with the account, solving every chain state. At w = 0.01: worst 1.27e-2 at 96.9% solver, `e_n > S_n` on 2.35% of
  cells and steps, 105 accepted steps of which **none** has a defect above its bound, the largest accepted defect 1.94e-3. The
  sum of accepted bounds is at most 5e-3 by construction, so the exact steps amplified the injected error at least 2.5 times.
  At w = 0.1: 3.4e-3, no excess.
- R4f's own record has 9.9e-3 at w = 0.01, 1% under the tolerance and twice the account's budget; G1 passed by that 1%.
- `records/r4f/allen_cahn_d5.json`: account exceeded on 7.69% of cells and steps, rollout 1.2e-2 at 95.8% solver, with a
  one-step worst of 2.2e-3. Check A reproduces it (1.6e-2 to 2.4e-2 at 94% solver). Section 3.25's "elsewhere the account
  holds" counts only served problems; bruss_d20 (0.69%) is also above zero.
- So the statement the records support is: the account bounds the chain where the exact map does not amplify errors in the
  scored coordinates (the lattices, Robertson, CHEMAKZO, lorenz96_d4 over 40 steps), and fails on HIRES and Allen-Cahn.

### 1.4 (major) 9.9.3: HIRES under coupling is limited by amplification of moderate defects, not shown to be rare large ones

- Section 3.26's "rare states whose defect is near 1e-2, summed over 170 steps" comes from smoke tests without a record. In
  check B no accepted defect exceeds 1.94e-3 and the chain still reaches 1.27e-2.
- "Neither data nor a first-order correction moves them" stands as measured (p13: 0.16-0.17 at every size; composite: 0.18 and
  0.16). What follows is narrower: HIRES's weak-mixing failure under the product gate is the gate accepting off-bundle inputs;
  under the envelope it is amplification. Neither is a statement about the fit on the bundle (plain rollout 4e-6).

### 1.5 (major) 9.9.1: data is decisive for plain rollouts, and three parts of the table need qualifying

- **POLLU's served flag flips at the margin.** `records/p13/pollu_220.json` and `pollu_340.json` have the same test rollout,
  5.5e-3, the same degree and the same one-step worst (2.9e-4). The flag differs because the selection chain is 5.1e-3 at 220
  and 4.4e-3 at 340, against a margin of 5e-3. The gain is from 100 to 220 trajectories (4.7e-2 to 5.5e-3); 220 to 340 buys
  nothing at degree 3.
- **POLLU at 580 is slower than VODE.** Degree 4 has 2,380 terms and an evaluation of 367 us; VODE's step is 343 us
  (`p14_baseline.json`). Its speed-up of 3.7 is against the baseline.
- **POLLU is not served under coupling at any size**: chained mixing has a solver share of 0.99-1.00 at 220, 340 and 580.
- **Allen-Cahn d5 at 580 refuses 30.1%** and its speed-up falls from 94 to 3.2 with the same degree and `e_hat`: more data made
  the deployed gate worse. Its served degrees are not monotone (5 and 7 served, 6 not, chains 5.2-5.4e-3 against the margin).
  It has no VODE time.
- **The header "100" is 60 for Nelson and primordial** (`n_train` 180, 300, 540).
- **"Nelson improves 4x", "primordial does not improve" are read from the held-out worst**, a maximum that I asked in round 3
  not to use: it rises with the sample. Check C, one-step errors on the 40 test trajectories, no gate:

  | problem, degree | 60 trajectories: median, share within tol/2 | 300 | 540 | within tol/20 at 540 |
  |---|---|---|---|---|
  | Nelson, 2 | 2.8e-2, 1% | 2.6e-2, 2% | 2.6e-2, 1% | 0% |
  | Nelson, 3 | 2.5e-2, 8% | 5.7e-3, 42% | 5.0e-3, 50% | 0% |
  | primordial, 3 | 4.3e-2, 8% | 1.3e-2, 15% | 1.3e-2, 18% | 0% |
  | primordial, 4 | 2.2e-1, 5% | 7.3e-3, 34% | 6.5e-3, 39% | 0% |

  Both improve with data and then saturate. My round-3 prediction (Nelson's degree-3 median below 2e-3 at 480) fails.

### 1.6 (major) 9.9.4: VODE is the wrong denominator on three problems, and Nelson's 78 ms is unconfirmed

- Check D, a few test states each, one process: LSODA with the analytic Jacobian integrates every state where VODE fails.

  | problem | library step | VODE (p14's call) | LSODA | LSODA's worst against the stored state |
  |---|---|---|---|---|
  | HIRES, 18 states | 1.44 ms | 4.72 ms | 0.20 ms | 2.3e-8 |
  | primordial, 9 | 28.2 ms | 11.2 ms, 7 of 9 converge | 7.2 ms, 9 of 9 | 6.4e-9 |
  | Nelson, 6 | 19.0 ms | 3,018 ms, 2 of 6 converge | 8.7 ms, 6 of 6 | 2.2e-6 |

  So "fails on Nelson and primordial" is VODE's behaviour, and the denominator should be the fastest exact solver that agrees
  with the labels, per problem.
- `records/p13_data/nelson_*.json` record 57.4 s for 480 trajectories of 24 steps, 5.0 ms per step, with
  `SURROGATE_ODE_WORKERS` at its default of 1 (`src/problems/ode/system.py:29`). The baseline record says 78 ms. Primordial:
  2.4 ms against 8.4 ms; HIRES 0.22 ms against 0.94 ms. I did not find the cause (`step_cost` takes a mean over 40 states, which
  a few expensive states can dominate). Until it is resolved, "Nelson 78 ms" overstates what a surrogate could save by up to 15x.
- p14's VODE has Python right-hand sides, so "compiled" describes the integrator only; the synthesis says so, the title does not.

### 1.7 (minor) Smaller points on 3.25 and 3.27

- G4's median of 70 is over plain rollouts. Under chained mixing the seven "served" problems include CHEMAKZO at 88% and 92%
  solver, fisher_kpp_d2 at 27% and 53%, bruss_d2 at 25% and 10% (`records/r4f/`).
- `evaluate_us` times `sur.evaluate` alone. `Account.refuses` then calls `gate_bound` again, which rebuilds the design and both
  sketched products, so the deployed cost under the account is up to twice the recorded one.
- The calibration set is ten trajectories per split (`thirds`: 80, 10, 10). Every "calibrated maximum" so far is over ten
  trajectories and their mixed states, pooled over three fits that are not the deployed one.
- p14's prediction "below 15" failed at 16; not material.

### The order

The capped envelope with a chain-calibrated account on all 18 problems; the denominators re-measured; a data budget with a
stopping rule; KIDA's exact solver; one test of another approximant on Nelson's stored data. Section 6 has costs and readings.

---

## 2. The gate that serves and certifies

**Recommendation: three parts, each with its own statement. A label-free cap that defines where the fit is determined; the
additive envelope fitted inside the cap, with a trajectory-level guarantee; an account whose charge is a model of the typical
defect and whose budget is calibrated on whole chains under the coupling scripts.**

### 2.1 Theory

Notation as in round 3: bound `b(x)`, offset `rho(x)`, rank `r`, `N` fitted states.

- **The cap.** `sum_i b(x_i)^2 = N r`, so the root mean square of `b` over the fit is `sqrt(r)`, and at most `1 / beta^2` of
  the fitted states have `b > beta sqrt(r)`. Accept only `b <= beta sqrt(r)` (default `beta` = 3), `rho <= rho_cap`, and the
  out-of-span share below its cap. This is Proposition 5.1's domain: inside it the fit's error is at most `(1 + beta sqrt(r))`
  times the best approximation error, and outside it nothing is claimed. It needs no labels, so it also measures data: the
  share of held-out trajectories inside the cap is a learning curve that costs no solves beyond the trajectories. Measured
  (check A, plain calibration states by step): 80-100% inside on the lattices; on POLLU at 100 trajectories 0% (median `b` 1.0e4 against
  `sqrt(r)` = 24), the label-free form of "POLLU is data-limited".
- **The envelope.** `E(x) = a0 + a1 b + a2 rho + a3 rho^2`, `a >= 0`, minimising the sum subject to `E >= e` on the calibration
  states inside the cap. This is a convex scenario program with four variables when a scenario is one calibration trajectory
  with its mixed states. For exchangeable trajectories and a fit that did not see them, the expected probability that a new
  trajectory has an accepted state with a defect above `E` is at most `4 / (n_cal + 1)` (Calafiore and Campi, the scenario
  approach). That is 36% at the present ten trajectories, 4% at 100 and 1% at 400. So the certificate is bought with
  calibration trajectories, which p13 showed are cheap; the pooled calibration of `final` is outside the statement, and the
  deployed fit should be the one calibrated.
- **The account.** Proposition 2.2 gives `|e_n| <= sum_j G_{n,j} |delta_j|` with `G` the exact map's propagation of an error
  from step `j` to `n`. Two facts from the checks set the design. First, `G` is not 1 on HIRES and Allen-Cahn (1.3), and the
  benchmark's `A_n` (6,669 on HIRES) is a worst case no budget survives. Second, inside the cap the chain error is close to the
  sum of *typical* defects: with a charge `c(x) = m0 + m1 b + m2 rho + m3 rho^2` fitted by non-negative least squares to the
  calibration errors, `max_n e_n / sum_j c(x_j)` is 0.1-1.5 at the median and at most 4.7 over test chains on seven problems
  (check A). So: charge `c`, not `E`; accept a step while `lambda (S + c) <= tol / 2`; choose `lambda` on whole chains.
- **What `lambda` certifies.** Run the calibration trajectories as chains under the deployment policy and each coupling script,
  for a grid of `lambda`, and take the smallest `lambda` for which at most a share `alpha` of chains exceed `tol / 2` anywhere.
  The loss "a chain exceeds the tolerance" is monotone in `lambda` up to the growth after an account closes, and conformal risk
  control then bounds the expected loss on a new chain by `alpha` when `n_cal >= 1 / alpha - 1`. The statement covers the
  amplification, the under-coverage of `c` and the horizon in one number, and it is limited to the calibrated scripts and
  horizon. It says nothing about states off the calibration law, which is what MFC will supply.
- **Online audits** cover that gap: solve a share `q` of accepted steps, compare the defect with `c(x)`, and raise `lambda` or
  close accounts in a region when the running ratio exceeds its calibrated range. After `m` clean audits the exceedance rate is
  below `3 / m` at 95%. This is a population statement over cells, not a bound per cell.
- **What it cannot do.** Where the map amplifies strongly (HIRES at w = 0.01), `lambda` will be large and the solver share
  high. That is the correct answer there, and the solver share is the cost of the guarantee.

### 2.2 What my check found, and what it did not

- Inside the cap the lattices, Burgers, CHEMAKZO and lorenz96_d4 are served on plain rollouts by the gate alone, and neither
  account changes their solver share (1.1; CHEMAKZO 1.3e-5 at 8.4% solver, lorenz96_d4 2.4e-5 at 5.6%).
- HIRES, plain rollouts, cap 3 sqrt(rank): gate only 7.5e-6 at 7.9% solver (R4e's product gate: 18.1%); the max-envelope
  account raises the solver share to 52.7%; the typical-defect account leaves it at 7.9% with no excess. So the charge model
  matters where the floor binds, and only there: the envelope's median bound is 2-5 times the typical charge on the lattices.
- POLLU at 100 trajectories is refused by the cap, as it should be.
- Allen-Cahn d5 at degree 3 exceeds every account (1-11% of cells and steps): its fit is coarse (mean defect 6e-4 a step) and
  the map amplifies. It needs p13's 340 trajectories and a `lambda` above 1.
- Not checked: any coupled chain with the cap (check B ran out of time after R4f's arm); `lambda` by the grid procedure (I
  took the largest ratio over ten selection chains, a level of 91% at best); 340 trajectories; the pooled fit.

### 2.3 Calibration and cost

- Calibration: at least 100 calibration trajectories per problem (bought as p13 bought its data), their mixed states, and
  chains of them under plain, w = 0.01 and w = 0.1 scripts for six values of `lambda`. Solver calls only for refused steps and
  the reference mixing chains.
- Evaluation: nothing added. `b` and `rho` are already computed; the account must reuse them instead of calling `gate_bound`.
- Selection: the cap makes high degrees pay for their refusals, so the cost rule needs no change; break ties as now.

### 2.4 The experiment (R4g), predictions to commit

On R4e's 18 problems at R4e's chart, 100 training trajectories plus 100 fresh calibration trajectories; then 340 plus 100.

- H1: all ten of R4e's served problems are served, plain rollouts and both mixing weights within `tol / 2`.
- H2: bruss_d20, fisher_kpp_d5 and fisher_kpp_d10 are served at R4e's degrees with a plain solver share below 20% at 100
  trajectories and below 8% at 340.
- H3: burgers_d20's accepted one-step worst is below 2e-3 at every served degree.
- H4: on every served problem the share of test trajectories with an accepted defect above `E` is at most 8% (twice the
  scenario bound at 100 calibration trajectories).
- H5: HIRES is served on plain rollouts with under 15% solver; under mixing at w = 0.01 its `lambda` is at least 2.5 and its
  solver share above 80% (a predicted failure to serve, stated).
- H6: `lambda` is at most 2 on the lattices, Robertson, CHEMAKZO and lorenz96_d4, and above 2 on allen_cahn_d5.

What changes the plan: H2 failing at 340 means the refusals are not a data effect, and the product gate stays for plain use
with the account as an audit only. H4 failing on more than one problem means the pooled or split fit breaks exchangeability
and the envelope needs an inflation. H6 failing on the lattices means the typical-defect charge is not stable across chains,
and the account goes back to the envelope inside the cap, which already serves them.

---

## 3. The expensive problems

**The shortest route to a factor of two on Nelson and primordial is the exact solver, and no gate or index set will serve them
with the present approximant.**

- **They are bias-limited.** Check E, 540 trajectories, stored data only:

  | problem | index set, terms | median | 99th percentile | within tol/2 | within tol/20 |
  |---|---|---|---|---|---|
  | Nelson | total degree 3, 969 | 5.0e-3 | 6.4e-2 | 50% | 0% |
  | Nelson | hyperbolic cross level 3, 1,273 | 3.0e-3 | 3.9e-2 | 75% | 0% |
  | Nelson | adaptive, 969 | 6.1e-3 | 5.4e-2 | 39% | 0% |
  | primordial | total degree 4, 330 | 6.5e-3 | 1.2e-1 | 39% | 0% |
  | primordial | hyperbolic cross level 4, 785 | 4.6e-3 | 1.3e-1 | 52% | 1% |
  | primordial | adaptive, 792 | 4.2e-3 | 1.8e-1 | 53% | 1% |

  Total degree 4 on Nelson has 4,845 terms, above the recipe's limit. A 24-step chain within `tol / 2` needs a typical defect
  near 2e-4; these are 15-30 times above it, and the curve in data is flat (1.5).
- **The ceiling.** A one-step gate that accepted exactly the states within `tol / 2` would serve 50-75% of steps: a speed-up of
  2-4 before accumulation, which the account then removes. Nelson's persistence error is 0.25 dex a step at the median, 25
  times the tolerance: the step is long for any smooth global fit.
- **The solver.** LSODA is 2.2 times faster than the library's step on Nelson and 3.9 times on primordial in check D, with
  correct labels. That is a measured factor of two with no modelling risk, and it lowers the price of every later data set.
- **KIDA.** The same order as round 3: the exact solver first (section 4), then the three no-fit measurements still open from
  rounds 2 and 3 (`dt` and horizon from an MFC run, the rank of the key species' futures, the local dimension). A fit before
  them repeats Nelson's result at 40 times the cost per solve.
- **One test of another approximant, on stored data.** An expensive step leaves room for an evaluation of 100 us or more,
  which the recipe's cost rule never used. Test on Nelson's 540 trajectories, with no new solves: this library's ART on the
  chart coordinates, and a local quadratic on the nearest 200-400 pairs. Commit beforehand: neither reaches a median below
  5e-4. If one does, it gets the gate of section 2; if neither does, Nelson and primordial are reported as not served.
- **Not recommended:** more trajectories for either at the present space; local fits on primordial without the modulus-of-
  continuity measurement asked for in round 3 (still not run).

---

## 4. Major speed-ups, against the fastest exact solver

| change | evidence | expected | measurement that confirms it |
|---|---|---|---|
| **Deploy the sketched gate on R4e's configurations.** p14's table uses R4e's product-gate evaluation (20-28 us at 462-715 terms). | R4f's selection rows time the sketched evaluation at R4e's degrees: bruss_d20 8.6 us, fisher_kpp_d5 4.1, fisher_kpp_d10 3.9; with R4e's solver shares and VODE's times the speed-ups become 13.5, 10.4 and 9.1 (my arithmetic from two records) | 2.8-2.9x on fisher_kpp_d5 and d10, 1.6x on bruss_d20, about 3x on CHEMAKZO | interleaved timing of both gates on the same test inputs, decisions compared |
| **Fewer refusals through data.** After the sketch the cost is `evaluate + share x solve`, and at a share of 5-7% and a solve of 87-412 us the second term is 5-28 us, equal to or above the first. | check A: the refused states are whole held-out trajectories outside the fit, 10-20% of them at 80 fit trajectories | up to 2x on the lattices if the share falls below 2% | section 2.4's H2 at 340 trajectories |
| **LSODA as the fallback and the data source** on HIRES, Nelson, primordial. | check D: 7x, 2.2x and 3.9x against the library's step | those factors on every refused step and every training solve | the same 40 states and steps as p14, one thread, labels within 1e-5 |
| **KIDA's step with a Fortran BDF that reuses factorisations** (LSODA, then VODE), then a compiled right-hand side. | round 3's check E: 74 dense LUs a step are 42% of it, SciPy's Python 23%; LSODA keeps a Jacobian across steps | 2x or more; not measured | p9's 120 states: seconds, LU count, labels within 1e-5 dex of the reference |
| **The account without a second gate evaluation.** | `Account.refuses` calls `evaluate` and `gate_bound` | up to 2x on the evaluation under the account | `evaluate_us` of the policy, not of the surrogate |

Below the bar or closed: lorenz96_d20 (1,771 terms, 208 us, a speed-up of 0.7 even against the baseline); POLLU at degree 4
(367 us against VODE's 343); a compiled design; the selection; GPU paths on this machine. HIRES and POLLU under coupling are
not speed targets at a solver share of 91-100%.

---

## 5. What would make the result publishable as a method

**Supported across the panel, with the wording the records allow:**

- A reduced or full chart with a truncated-SVD total-degree fit, a Lebesgue-function gate and solver fallback serves ten of 18
  step maps at 0.01 in plain rollouts and chained mixing under the product gate (`records/r4e/`), at 3-100 times a Fortran BDF
  with Python right-hand sides (p14). The identity `sum b^2 = N r` and what it says about the product form.
- Training data in whole trajectories is the first lever for a fit that is close: POLLU 4.7e-2 to 5.5e-3 from 100 to 220
  trajectories, Allen-Cahn d5 served at 340 (p13).
- The negative results, which are the paper's most useful part for a reader choosing a method: the oscillators are refused at
  every size; Nelson and primordial are bias-limited (1.5, section 3); the fitted maps have no tensor-train structure in chart
  coordinates (round 3); no index set changes the served set (p12, check E); a first-order off-bundle correction moves medians,
  not HIRES's coupled failure (3.26); the uniform accumulation budget refuses everything unserved.

**Need one more experiment:**

- "The gate certifies": R4g (2.4), with the trajectory-level statements and at least 100 calibration trajectories.
- Every speed-up: against the faster of VODE and LSODA on the rollouts' own states, with the sketched gate, pinned threads,
  interleaved (section 4, row 1 and 1.6).
- "Served" for POLLU and Allen-Cahn d5: three seeds of the extra trajectories, since both flip at the margin (1.5).
- A data budget with a stopping rule, in place of "a learning curve": stop when the held-out share inside the cap passes 95%
  and the chain statistic of 2.1 changes by less than 1.5x between sizes.

**Drop:**

- "The envelope plus account certifies coupled chains" as run in R4f (1.3).
- The composite tube and the normal term as components of the method; keep them as a recorded negative result.
- Speed-ups against per-call SciPy, and "3-800x".
- Any KIDA surrogate claim; KIDA appears as the target, with the solver audit and the coverage curve.
- lorenz96_d20 as a candidate.

---

## 6. Plan, in order

| # | step | purpose | CPU cost | the result that would change the plan | stop doing |
|---|---|---|---|---|---|
| 1 | R4g: the cap, the envelope inside it, the typical-defect account with `lambda` from chains (section 2), 100 fresh calibration trajectories per problem, at 100 and 340 training trajectories; predictions H1-H6 | A gate that serves R4e's ten and states what it certifies | 1.5 h | H2 failing at 340: keep the product gate for plain use. H4 failing on two problems: add an inflation before anything else | The R4f account; caps at calibration ranges; ten-trajectory calibration; reading `evaluate_us` as the policy's cost |
| 2 | p14 again with LSODA beside VODE on the rollouts' own states, the library's step re-timed on the same states, and R4e's table restated with the sketched gate | The denominator; resolves Nelson's 78 ms against 5 ms | 20 min | An exact solver within 3x of the surrogate's cost on a problem: that problem leaves the speed claims | `solve_us` from the baseline records as a denominator; VODE where it fails |
| 3 | The data budget with the stopping rule of section 5 on the ten served problems, POLLU and allen_cahn_d5, three seeds of extra trajectories | Robust served flags; the refusal share against data | 1 h | Served flags that differ between seeds on more than two problems: the margin rule needs a tolerance band, and the counts in the paper become ranges | Fixed 100 trajectories; the held-out worst as a statistic |
| 4 | KIDA's exact step: LSODA and VODE on p9's 120 states, then a compiled right-hand side | 2x on the target's fallback and data source | 1-2 h | No Fortran BDF within 1e-5 dex of the reference: the route is a Rosenbrock or Radau method with a warm start | Model search on KIDA; the sparse path as tried |
| 5 | Nelson on stored data: this library's ART and a local quadratic, prediction "neither below 5e-4" (section 3) | Close or open the expensive panel problems | 45 min | Either below 5e-4: it gets R4g's gate and a learning curve | Index sets and more trajectories on Nelson and primordial |
| 6 | KIDA's no-fit measurements and a coupled test with recorded offsets (round 3, steps 4 and 6), both needing states from an MFC run | The tube radius and `dt` the target will set | 45 min plus the user's | Offsets in MFC states beyond the calibrated `rho_cap`: the tube must be sampled from MFC, not from mixing scripts | Chained mixing at chosen weights as the coupled headline |

Steps 1 and 2 are independent and come first; 3 uses 1's gate; 4 and 5 run alongside; 6 waits on the user's MFC states.

---

## Appendix: the checks

All from `theory/panel` with
`SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=../..:.:../recipe ../../.venv/bin/python <script> <args>`, each under three
minutes, scripts and outputs (`*.out`) in the scratch directory named at the top.

- **A** `acct2.py <problem> [degree]`: `generic4.build` on split 0's 80 fit trajectories at R4e's chart and degree with the
  sketched gate; calibration statistics on the split's ten calibration trajectories and their mixed states; the envelope and a
  non-negative least-squares charge fitted inside a cap on `b` (the 90th percentile, and three times the square root of the
  rank); chains on the ten selection and 40 test trajectories with the gate only, the max-envelope account and the charge
  account. Run on fisher_kpp_d5, bruss_d20, fisher_kpp_d10, burgers_d20 (degrees 3 and 5), fisher_kpp_d3, allen_cahn_d5,
  hires, pollu, chemakzo, lorenz96_d4. POLLU's second arm stopped (no state inside the cap). Numbers in 1.1, 1.2, 2.1, 2.2.
- **B** `mix.py hires`: R4f's HIRES surrogate rebuilt (pooled calibration as `final`), chained mixing on 20 test trajectories
  with the account, every chain state solved to split `e_n > S_n` into accepted defects above their bound and propagation. The
  capped arm did not finish inside the time limit. Numbers in 1.3, 1.4.
- **C** `nel.py <problem> <degrees>`: Nelson and primordial at R4e's chart with 60, 300 and 540 trajectories
  (`records/p13_data/*.npz`), one-step errors on the 40 test trajectories, no gate. Table in 1.5.
- **D** `vode.py <problem> <n>`: the library's step, p14's VODE call, VODE with a small first step, VODE with a
  finite-difference Jacobian, LSODA through `scipy.integrate.ode` and through `solve_ivp`, on states at steps 0, 10 and 20 of
  the first test trajectories. Table in 1.6.
- **E** `basis.py <problem> <basis:level> ...`: total-degree, hyperbolic-cross and adaptive index sets at 540 trajectories,
  one-step errors on the test trajectories. Table in section 3.
- Inline readings of `records/r4e/`, `records/r4f/`, `records/p13/`, `records/p13_data/`, `p14_baseline.json` and
  `results/benchmark/data/step_maps/*/baseline-*.json`.

Not checked: the capped gate on the full 18 problems, with pooled calibration, or under any coupled chain; `lambda` by the
grid procedure; any run at 340 trajectories with the new gate; the cause of the baseline's 78 ms on Nelson; LSODA on the
lattices, POLLU and KIDA; whether the scenario bound's exchangeability holds for mixed states drawn from the same trajectories.
