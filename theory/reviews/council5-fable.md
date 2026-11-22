# Council review, round 5 (Fable): what sets the refusal rate, and what a GPU changes

Reviewer: Claude Fable 5.2, one of three independent council members. Material: `theory/SYNTHESIS.md` at commit `9de9461`
(sections 3.28-3.33, 9.10, 10), my round-4 review, `theory/panel/generic4.py`, `r4g_capped.py`, `p15_denominators.py`,
`p17_audit.py`, `p18_active.py`, `p19_seeds.py`, the untracked `theory/panel/gpu_eval.py`, their records, and `src/numerics/`
for the library's GPU paths. My four checks are read-only and post hoc; their scripts and outputs are in
`/tmp/claude-1000/-home-prajwal-github-repos-affini/32f886c8-be9f-433e-aef0-3afb16101b8f/scratchpad/c5/` (not committed) and are
listed in the appendix. None was pre-registered, so each is a reading to re-run with a committed prediction. Times are one thread
on a shared laptop; I quote ratios within one script. There is no GPU here: every GPU number below is an estimate from operation
counts and is marked as such.

## Summary of the recommendation

**The panel's refusal rate is not the quantity that will cap a deployment, and "refusal is the lever" (9.10.3) is true for a
reason the synthesis does not state.** On the nine problems that work in the closed columns, every refusal is in the first three
steps of a chain and the rate from step 3 on is 0.0%, so the reported 2-5% is the transient divided by a 40-step chain, and
p17's 2-20x is set by the evaluation time at 100 cells per call with the gate computed twice, not by 1/rho. Against that, the
account as built never forgets: it charges every accepted step and never decays, so each cell's budget runs out after a few hundred
to a couple of thousand accepted steps and the refusal rate then goes to 100% and stays there (measured on `fisher_kpp_d10`: 69% in steps 120-159, 100%
from step 160, while the true error falls from 5e-4 to 1e-13). In a simulation longer than the benchmark's chains this, not the
gate, is what caps the speed-up at about 1. So the first work is the account (a decay factor calibrated on long chains, a stated
horizon where the dynamics are neutral, and its transport with the cells), and a restated metric: transient, steady-state,
off-bundle and account refusals reported separately on chains at least ten times longer and on 1e5 or more fresh cell-steps.
For the coupled inputs, CHEMAKZO's refusals are all the cap, at 30-650 times the cap, so widening it is not an option; but the
bound `b` is the square root of N times the leverage, so adding refused states is optimal design with a bound on what it
achieves, and p18's own record shows it (CHEMAKZO's chained-mixing solver share falls from 98.5% to 2.5% with refused data and
stays at 98.1% with random data; the synthesis table does not show this column). It fails on reduced charts (POLLU, HIRES fall
to degree 2), where the chart must grow first. On a GPU the evaluation is about 36,000-66,000 floating-point operations per
cell, half to two thirds of them the sketched gate, so it batches well in float64 with a row-independent kernel; float32
turns 95% of the decisions into refusals on a rank-deficient fit; and a GPU makes the fallback the bottleneck at any refusal rate above
about 1e-3, so the comparator that decides whether this is worth building is a batched exact solver on the same GPU, against
which I expect 3-15x at d of 20 or less, not 40-460x. Training is dominated by exact-solver labels, not linear algebra; it
scales across devices with one reduction of R factors and independent selection fits, and is limited by the model's fixed
capacity, not by communication.

---

## 1. The plan from here; which readings of 9.10 do not follow

Findings, most important first.

### 1.1 (critical) The account never decays, so the refusal rate goes to 100% on long chains

- `generic4.Account.refuses` adds the charge `c` to `S` on every accepted step and never lowers it; a step is refused when
  `lambda (S + c) > tol / 2`. A refused step "adds nothing" and also removes nothing. So a cell has a fixed number of accepted
  steps, about `tol / (2 lambda c)`, and is refused for ever after.
- The records already show how close this is. `records/r4g/*_340.json`, `account_check.account_max` after one benchmark chain,
  as a share of the budget (`lambda S / (tol / 2)`): fisher_kpp_d10 0.56 after 40 steps, pollu 0.62 after 59, fisher_kpp_d3
  0.31, fisher_kpp_d5 0.30, burgers_d20 0.24, fisher_kpp_d2 0.21, bruss_d2 0.19, bruss_d20 0.17, lorenz96_d4 0.08; **hires 1.00
  after 170 steps**, which is its 85% solver share. Only robertson, chemakzo and allen_cahn_d5 (charges near 1e-7 to 1e-6) are
  far from it.
- Check D (`long.py`): the 40 test cells continued to 400 steps with the exact solver, the deployed surrogate and account.

  | problem | steps 0-39 | 80-119 | 120-159 | 160-399 | true worst error | median `lambda S / budget` at 400 |
  |---|---|---|---|---|---|---|
  | fisher_kpp_d10 | 4.4% | 1.7% | 69.3% | 100% | 5e-4, then 1e-4 falling to 2e-13 | 1.00 |
  | burgers_d20 | 2.5% | 0% | 0% | 0% | 5e-6 | 0.66 (full near step 600) |
  | lorenz96_d4 | 3.2% | 0% | 0% | 0% | 3e-4 rising to 1.2e-3 | 0.31 (full near step 1,300) |

  All the refusals after step 40 are the account's alone (the gate accepts those inputs). The charge per step falls after the
  transient, so a linear reading of `account_max` (170 steps for burgers_d20) is too short by a factor of three to four; the
  horizon is a few hundred to a couple of thousand steps, not infinite.
- The two cases teach different things. On fisher_kpp_d10 the dynamics contract (once the solver takes over, the error falls
  by about 10% a step), so the account charges for errors that no longer exist. On lorenz96_d4 the error grows linearly, by
  about 2.5e-6 a step, and the account tracks it (0.31 of the budget charged, 0.24 used): there the account is right, and
  what it says is that this surrogate has a horizon of about 2,000 steps at this tolerance, whatever the gate.
- Consequence for 9.10: "the capped gate ... is the current recipe ... serves 13 of 18" holds only for chains of the benchmark's
  length. No speed-up in 3.31 extrapolates to a longer run.

### 1.2 (critical) On the served problems the refusal rate is the transient, not a rate

- `records/r4g/*_340.json`, `rollout_solved_by_step`: the refusals sit in the first one to three steps (the first is refused at 60-100%
  on all but bruss_d2, which refuses nothing) and the share from step 10 on is 0.0% on bruss_d20, burgers_d20, fisher_kpp_d2/d3/d5/d10, lorenz96_d4 and robertson (0.2% allen_cahn_d5, 0.7% pollu,
  2.5% chemakzo). The quoted 2-5% is one to two refused steps out of 40.
- Check A (`causes.py`) confirms it in the closed columns (sorted, D = 0.05) with the cause of every refusal:

  | problem | refused, steps 0-2 | steps 3-9 | steps 10+ | cause | `b / cap` of the refused (median, 90th) | their true one-step error (median, 90th) |
  |---|---|---|---|---|---|---|
  | bruss_d20 | 64.3% | 0.0% | 0.0% | cap on `b` | 1.9e5, 1.4e6 | 0.77, 2.5 |
  | fisher_kpp_d10 | 47.0% | 0.0% | 0.0% | cap (43.3), envelope (3.7) | 6.2, 57 | 1.3e-2, 0.11 |
  | burgers_d20 | 31.3% | 0.0% | 0.0% | cap | 1.6, 2.5 | 6.4e-4, 1.1e-3 |
  | robertson | 68.0% | 7.0% | 0.0% | cap | 1.7, 4.4 | 2.8e-7, 7.8e-7 |

- So there are two kinds of transient refusal. On bruss_d20 and fisher_kpp_d10 they are correct: the surrogate is wrong by
  1e-2 to 2.5 on those states, and no gate should accept them. On burgers_d20 and robertson (and chemakzo's plain rollouts:
  `b / cap` 1.25 and 1.94, errors at most 2e-7) they are needless: the inputs sit one to four caps out and the fit is accurate.
- p18's "1.5-5x lower rho" is a change in steps 0-2 only (`records/p18/*`: bruss_d20's third step falls from 90% to 5% refused,
  its first two stay at 100%; every arm is 0.00% from step 10 on). It is a real effect on the transient and says nothing about
  a steady rate.
- What a deployment sees depends on how often its cells are in a transient. That is a property of the application, so the
  recipe should report three numbers, not one: the refusal rate on fresh starts, the steady rate, and the steps between them.
- The steady rate is not yet measured to the precision that matters. Zero refusals in 3,000-3,700 cell-steps bounds it below
  about 0.1% (the rule of three). Section 3 shows that on a GPU the fallback dominates from about 1e-3, so the audit needs 1e5
  to 1e6 fresh cell-steps per problem.

### 1.3 (critical) 9.10.2: p17's speed-ups are limited by the evaluation as timed, not by refusals

- From `data/p17_audit.json`, per column: `1 / rho` is 15-570 on the nine working problems, while `exact_s / evaluate_s` is
  2-37. Example, fisher_kpp_d10: `1 / rho` 25-31, exact over evaluation 2-3, speed-up 1.6-2.3. The binding term is the 33-48 us
  per cell-step of evaluation against LSODA's 84 us.
- That time has two removable parts. First, `Account.refuses` calls `sur.evaluate` and then builds the design and the gate
  terms again for the charge (`generic4.py` lines 493-500), so the gate is paid twice. Second, the columns evaluate 100 cells a
  call. Check C (`batch.py`, one thread):

  | us per cell | batch 100, as p17 times it | batch 100, one evaluation | batch 1,000, one evaluation |
  |---|---|---|---|
  | fisher_kpp_d10 (462 terms) | 14.9 | 7.8 | 3.4 |
  | bruss_d20 (715 terms) | 23.2 | 12.0 | 5.8 |

  p17 recorded 33-48 and 26-67 us for the same protocol, two to three times my one-thread figure, so load or threading was in
  its timing as well (I did not establish which).
- Restated with p17's own refusal rates and denominators: fisher_kpp_d10 at 3.4 us is 84 / (3.4 + 0.035 x 84) = 13, not 1.9;
  bruss_d20 at 5.8 us is 415 / (5.8 + 0.048 x 415) = 16, not 7.9. At the steady rate (no transient) they are 25 and 72.
- So 9.10.2's "2-20x on nine problems (median about 7)" understates the nine, by a protocol artefact, and 9.10.3's "refusal is
  the lever" does not follow from p17 for them. It does follow for the three that fail (1.4, 1.5) and for long chains (1.1).

### 1.4 (major) CHEMAKZO under coupling: the cap, by a factor of 30-650, on inputs the fit predicts well; data fixes it

- Check A, CHEMAKZO's column: 98.4% refused, every one by the cap on `b`; `b / cap` has a median of 30 and a 90th percentile
  of 650. Their true one-step error has a median of 6e-7, a 90th percentile of 4.5e-5 and a 99th of 9.7e-4; 98% are below
  tol / 20. The calibration states mixed at w = 0.01 are 0.7% inside the cap (`b / cap` median 249), at w = 0.1, 0.1% (median
  2,375), and there the errors outside reach 4.8e-2 at the 99th percentile.
- So "widening the cap where it is too strict" (question 2's third candidate) is not a small change: it means a cap hundreds
  of times larger, which is the uncapped envelope round 4 removed, and at w = 0.1 it would accept errors above the tolerance.
- The synthesis's table for p18 reports the plain-rollout share only. The same records hold the result that matters
  (`records/p18/chemakzo_{refused,random}.json`, `mixing_chained`):

  | CHEMAKZO, 100 trajectories | chained mixing w = 0.01, solver share | w = 0.1 | calibration states inside the cap |
  |---|---|---|---|
  | R4g | 98.5% | 99.9% | 38% |
  | + 8,782 random pairs | 98.1% | 99.9% | 39% |
  | + 8,782 refused pairs (8,441 of them mixed at w = 0.01) | **2.5%** | 37.9% | 79% |

  with a worst of 6.2e-6 and 1.0e-5. The same holds, smaller, on robertson (w = 0.1: 26.4% to 0.9%; 2.3% with
  random data, at another degree) and bruss_d2 (w = 0.01:
  14.4% to 0.4%, against 9.3% with random data). Section 2.2 gives the reason and the bound. It was not tested in closed
  columns, and the pool held no w = 0.1 states, which is why that weight stays at 38%.

### 1.5 (major) Refusal-driven data fails on reduced charts, and 3.32 reads it as HIRES "getting worse"

- `records/p18/{hires,pollu}_refused.json`: with all 11,800-31,627 refused pairs added, the selection falls to degree 2 and
  refuses everything (HIRES: degree 8 to 2; POLLU: 3 to 2, 94% solver against the random arm's 6.6%). Both have reduced charts
  (HIRES k = 3 of 8, POLLU k = 13 of 20); CHEMAKZO, robertson and bruss_d2, where it works, have full charts.
- That is 3.20's result again: on a reduced chart the mixed states differ across the chart's fibres, so their successors are
  not a function of the chart coordinates, and adding them is noise for the fit. Refused data needs the chart to hold the
  directions the refused states leave in (section 2.4).
- Also, when everything is refused the "refused" set is simply all the pool's states, plain and mixed, at twice the random
  arm's count (7,800 against 4,800 on Allen-Cahn d5, 31,627 against 20,400 on HIRES): on those rows the two arms differ in
  count and composition, so they are not the equal-cost comparison the script's docstring states.

### 1.6 (minor) Smaller points

- 9.10.1 "serves 13 of 18 on plain rollouts and chained mixing, robustly across draws": CHEMAKZO and POLLU pass chained mixing
  with a solver share of 100% (`records/r4g/{chemakzo,pollu}_340.json`), which is the exact solver passing. "Served" should
  require a solver share below one half in every script it is claimed for; by that rule 10 are served under mixing: CHEMAKZO, POLLU and
  HIRES (96-97%) are not. p19 covers five problems.
- The out-of-span cap is a comparison of rounding errors on full-rank fits. When the rank equals the number of terms, `s` is
  zero in exact arithmetic; `s_max` is then 1.4e-15 (burgers_d20) and 1.7e-15 (fisher_kpp_d10), and check A finds 0.8% of
  steps 0-2 refused by `s` alone on both. Drop the term when the fit has full rank: it removes those refusals and half of the
  sketch's cost on 9 of the 13 served fits.
- `gpu_eval.py` (untracked) reproduces `Surrogate4.evaluate` exactly in float64 on NumPy (0 decision flips, 0 difference on
  5,600 inputs, both problems): a sound base for the GPU measurements.

### The order

1.1 changes what "served" means, so it comes first, with 1.2's split of the refusal rate and 1.3's timing protocol in the same
experiment (they are one script). Then the leverage design in closed columns (1.4, 1.5). The GPU measurements can run in
parallel on another machine as soon as the evaluation protocol is fixed.

---

## 2. Cutting the refusal rate several-fold

### 2.1 The decomposition, and where each part is cut

For a cell followed for N steps, with the present recipe:

    rho = rho_transient + rho_leverage + rho_account + rho_bias

- **transient**: fresh states the fit is wrong on (bruss_d20, fisher_kpp_d10) or merely one to four caps out (burgers_d20,
  robertson, chemakzo): n_t / N for a single start, the application's share of fresh states otherwise.
- **leverage**: inputs with `b` far above the cap where the fit is nevertheless accurate (CHEMAKZO's coupled inputs).
- **account**: 1 - n* / N once N exceeds n* = tol / (2 lambda c).
- **bias**: the approximant cannot represent the map at the tolerance (Nelson, primordial, the oscillators, lorenz96_d20):
  100%, and nothing in this section moves it (p18's R4 held).

The speed-up is `t_exact / (t_eval + rho t_fallback)`. On a CPU core `t_eval / t_exact` is 0.002-0.04, so rho matters from a
few percent; on a GPU it matters from about 1e-3 (section 3.3). Ranked by what they buy:

### 2.2 Refusal-driven data is leverage design: the bound, and the decisive test

**Theory.** With A the fitted design (N rows, retained rank r) the gate's bound is

    b(x)^2 = N phi(x)^T (A^T A)^+ phi(x) = N h(x),

N times the leverage of x. The mean of h over the fitted rows is r / N, which is where the cap `b <= 3 sqrt(r)` comes from. So
a cap refusal is a statement about the design, not about the map: the data do not determine the fit at x.

- *One point.* Adding m rows with design vector phi(x) gives `h'(x) = h / (1 + m h) <= 1 / m` (Sherman-Morrison). The cap
  needs `N' h' <= 9 r`, so about `N / (9 r)` nearby rows bring a refused point inside. On CHEMAKZO at 340 trajectories that is
  28,900 / (9 x 724), four to five rows.
- *A distribution.* If a share alpha of the new design's rows is drawn from the refused inputs' distribution mu, the mean of
  `b^2` over mu is at most `r / alpha`, so by Markov at most `1 / (9 alpha)` of mu stays outside the cap: 22% at alpha = 1/2,
  whatever mu is. Usually far fewer do (p18's CHEMAKZO arm has alpha = 0.51 and leaves 2.5%). Random plain pairs have alpha = 0 with respect to mu, which is why the random arm does
  nothing for coupled inputs (1.4).
- *What survives.* Proposition 5.1 and the envelope are unchanged: the fit is still linear in its data and calibrated on fresh
  states. The refused pairs cost no extra solve (deployment solved them). What the bound does not give is accuracy at the new
  points: that needs the map to lie in the model there (true on CHEMAKZO: errors of 6e-7 at inputs 30 caps out; false on a
  reduced chart, 1.5). The guarantee stays distributional, and it now holds for the coupled inputs only if the calibration
  set contains inputs from the same script as deployment. Chained mixing at two weights is a model of coupling; the columns
  are another distribution. The calibration set must include refused states from the columns, held out from the fit.
- This is greedy G-optimal design (Kiefer-Wolfowitz: minimising the largest leverage). Iterating it converges when the
  refused share on fresh inputs of the same script stops falling; the stopping rule is the Markov bound above being slack.

**The experiment (p20), predictions to commit.** Full-chart problems (chemakzo, robertson, bruss_d2, lorenz96_d4,
allen_cahn_d5), 340 trajectories, closed columns at D = 0.005, 0.05, 0.25. Round j: run the deployed surrogate on a pool of
fresh columns, add the refused pairs (cap the count at the size of the fit, alpha = 1/2, sampled uniformly from the refused),
refit, recalibrate on fresh plain, mixed and column states, score on fresh columns. Three rounds. Control: the same count of
random column states (alpha = 1/2 from the columns' own distribution, not selected by refusal), which separates "data where
the deployment goes" from "data where it refuses".
- L1: CHEMAKZO's column solver share falls from 96-100% to below 22% after one round in every column and below 5% after
  three, within tol / 2.
- L2: the control does as well as the refused arm within a factor of 2 on CHEMAKZO (the lever is the columns' distribution;
  refusal only selects it cheaply). If L2 fails in favour of the refused arm, selection by refusal matters by itself.
- L3: the plain-rollout share from step 3 on stays below 0.5% on all five (the on-bundle fit is not degraded).
- L4: the shares at w = 0.1 fall below 22% once w = 0.1 refusals are in the pool.
- What changes the plan: L1 failing with the inside-cap share above 80% means the envelope, not the cap, refuses, and the
  off-bundle map is not in the model: go to 2.4 on full charts too.

### 2.3 The account: a decay factor, a horizon, and transport

**Theory.** The chain error obeys `e_{n+1} = Phi(x_n + e_n) - Phi(x_n) + delta_n`. If over windows of m steps the map's
derivative along the visited states satisfies `|D Phi^m| <= gamma^m` with gamma < 1 in the scored norm, then
`S_{n+1} = gamma S_n + c_n` bounds the error to first order, stays below `c / (1 - gamma)`, and never exhausts when
`lambda c / (1 - gamma) < tol / 2`. A refused step multiplies S by gamma as well: the exact solver contracts the inherited
error just as it contracts any perturbation. Where the growth is 1 or more (neutral or chaotic maps, lorenz96_d4), no account
is bounded: the honest output is a horizon, `n* = tol / (2 lambda c)`, and a per-step budget `tol / (2 A_N)` for a stated N.
The records already show what that costs (`budgets.accumulation.refused_percent` in `records/r4g/*_340.json`: 79% refused on
fisher_kpp_d10 and 100% on bruss_d2 under the 40-step accumulation budget, against 4% and 0%).

- gamma cannot be read from the benchmark's `dynamics` (its `accumulation_factor` is 30-140 after 40 steps on these
  problems: the worst-case norm does not contract inside the benchmark's horizon). It must be calibrated on chains, as lambda
  is: pairs of exact chains from starts displaced by a surrogate-sized error, the ratio of their distance m steps apart, taken
  at the 95th percentile over chains. This is the same chain-level, distributional guarantee as lambda, and lambda then has
  one job (the margin on c) instead of two.
- Round 4 found amplification above 1 on HIRES under weak mixing, so gamma is a function of the script. Calibrate it on the
  scripts deployed, and refuse to state a horizon-free guarantee when its upper quantile is 1 or more.
- **Transport.** p17 keeps S with the cell index while transport moves material between cells. For a transport step that is
  a convex combination (non-negative weights summing to 1, in the variables it acts on), the error after it is at most the
  same combination of the errors, so S must be transported by the same stencil (or by the stencil's maximum, which is
  conservative and cheaper). Without that, S is the account of a cell that no longer holds the material it charged.

**The experiment (p21), predictions to commit.** The ten problems that hold tol / 2 in p17, chains of 400 and 2,000 steps
(trajectories continued with the fastest qualified exact solver), plain and closed columns. Arms: the present account; no
account (gate only); the decaying account with gamma from separate long calibration chains.
- C1: with the present account the solver share is above 50% over steps 200-399 on fisher_kpp_d10 (check D), and over steps
  1,000-1,999 on burgers_d20, lorenz96_d4 and at least four of fisher_kpp_d2/d3/d5 and bruss_d2/d20 (from `account_max`,
  allowing for the charge falling after the transient as in check D).
- C2: with no account, the lattices and Burgers hold tol / 2 to 2,000 steps with a steady solver share below 0.5%;
  lorenz96_d4 exceeds tol / 2 between 1,500 and 3,000 steps (2.5e-6 a step in check D).
- C3: the decaying account's solver share from step 40 on is below 1% on every problem where the calibrated gamma is below 1,
  and the chain's error stays below the account on at least 95% of chains.
- What changes the plan: C2 failing on a contracting problem (errors accumulate without the account) means the gate's
  one-step statement is not enough even there, and the horizon becomes part of every claim.

### 2.4 Reduced charts under coupling (POLLU, HIRES): grow the chart where it is refused

The refusals here are also the cap (`inside_cap_share` 0.33 on POLLU, 0.60 on HIRES, and 100% and 96% solver under mixing),
but 1.5 shows data alone destroys the fit. 3.20 found the normal term owns the off-bundle defect and costs accuracy on the
bundle. The two together say: keep the on-bundle fit, and give off-bundle inputs their own fit on a larger chart, chosen by
the offset rho the gate already computes. Bound: each fit is linear in its own data with its own leverage and envelope; the
switch is a label-free function of the input (rho against the decoder's training range), so calibration splits cleanly in
two and the union costs one more term in the coverage statement. This differs from the withdrawn composite tube in that the
second fit is a full fit on k' > k coordinates with its own gate, not an affine correction with the gate inherited.
- Test on POLLU first (the cheaper of the two): the off-bundle fit at k' = 20 (degree 2, 231 terms) on column and mixed
  refusals, alpha = 1/2. Prediction: POLLU's column solver share falls below 30% within tol / 2. If the degree-2 full-chart
  fit is bias-limited there (held-out median above 1e-3), POLLU joins Nelson: stop.
- HIRES only if POLLU passes; its amplification (round 4) makes the account, not the gate, its limit.

### 2.5 The burn-in rule, and a second tier above the cap

- The gate already implements the burn-in rule where it is needed (bruss_d20, fisher_kpp_d10: errors of 1e-2 to 2.5 on the
  refused first steps). A fixed "solver takes the first n steps" rule would add nothing there and would be wrong for a
  deployment whose cells are not at their first step. No bound is needed, and nothing is to be built.
- Where the transient refusals are needless (burgers_d20, robertson, chemakzo plain: `b / cap` at most 4.4 at the 90th
  percentile, errors at most 1.1e-3 and usually 1e-7), a second tier removes them: a second envelope fitted on calibration
  states with `cap < b <= 5 cap`, accepted when it is below tol / 2. The guarantee is the same trajectory-level statement with
  one more tier. Prediction: it removes at least half the first-three-step refusals on those three and none on bruss_d20 and
  fisher_kpp_d10 (their tier-2 envelope exceeds tol / 2). It is worth building only if the application has a large share of
  fresh states; it does not touch the steady rate. Low priority.
- Data on the transient ("more independent starts", section 5.3 of the synthesis) is the other route, and p18 shows its size:
  one refused step fewer.

### 2.6 Local refits and better gates

- A local refit of a refused region is 2.2 with a partition. Its natural form in this library is the recipe's fit inside
  ART's leaves: per-leaf designs, per-leaf leverage, the leaf lookup as part of the gate. It answers the capacity limit of
  section 5 below and uses the engine's per-segment Gram sums as they are. It is a new method, not a repair: I would not
  start it before 2.2-2.4 show how far one global fit goes, and the calibration must then cover every leaf (a pooled
  conformal score normalised per leaf, or leaves without enough calibration states refuse).
- I have no better gate to propose than the capped envelope. The evidence says the gate is not the limit: steady-state
  refusals are 0.0%, the transient ones are mostly correct, and the coupled ones are the design.

---

## 3. GPU inference and batching

### 3.1 What one evaluation costs (from `Surrogate4.evaluate` and `gpu_eval.py`)

For p terms, m chart coordinates, `free` outputs and a sketch of 32 + 32 columns, per cell:

    design  p (m - 1)      prediction  p free      gate  64 p      decoder  p_dec free

- bruss_d20 (p = 715, m = 9, free = 20): 5,700 + 14,300 + 45,800, about 66,000 operations; fisher_kpp_d10 (p = 462): about
  36,000. The sketched gate is half to two thirds of them. Check C agrees: design 2.8, prediction 0.9, gate 3.7 us per cell
  on bruss_d20 (1.6, 0.4, 2.5 on fisher_kpp_d10).
- The design matrix is p doubles per cell (5.7 kB on bruss_d20), written once and read three times: the evaluation is
  memory-bound, which is why the CPU time per cell rises again above 1,000 cells a call (check C: 3.4 us at 1,000, 5.5 at
  50,000 on fisher_kpp_d10).

### 3.2 Is it worth batching, and at what sizes

Yes: every cell is independent, the work is three dense products with a small right factor, and there is no control flow.
Estimates, to be measured:
- **Throughput.** At 36,000-66,000 operations and 4-6 kB of traffic per cell, a data-centre GPU in float64 should reach 1e7
  to 5e7 cells a second (0.02-0.1 us per cell), a workstation card with weak float64 3e6 to 1e7. Against one CPU core at
  3-6 us that is 30-300x; against a 64-core node it is 1-7x, because the CPU evaluation is already BLAS-bound.
- **Batch size.** Launch overhead (a few tens of microseconds for about ten kernels) is amortised from about 1e4 cells; the
  design's memory caps a batch at about 1e5 to 1e6 cells per 8 GB unless it is tiled. Tile at a fixed size, or fuse the design
  and the three products in one kernel that never stores the design (one thread per cell and output block, the term loop
  sequential). Under AGENTS.md's rule the fused kernel has to buy a measured 2x over the tiled CuPy version to be kept.
- **Decisions must not depend on the batch.** A cell at the gate's boundary must get the same decision whatever cells share
  its batch, or a run changes with the domain decomposition. cuBLAS chooses its algorithm by shape (the reason for
  `det_matmul` in `src/numerics/linalg.py`), so the products that feed the gate and the prediction need a row-independent
  kernel with a fixed term order: the same contract as ART's and AFINNI's prediction, and the same style of kernel.

### 3.3 The 1/rho cap on a GPU: the fallback dominates from about 1e-3

Per cell-step, with C fallback cores per GPU: `t = t_eval + rho t_fallback / C`. With t_eval at 0.05 us and LSODA at 60-400 us
(p15), the two terms are equal at `rho = C x (1e-4 to 8e-4)`. So with one core the fallback dominates above 0.01-0.08%, with
64 cores above 0.8-5%. The transient rate of 2-5% therefore removes most of a GPU's gain, and the steady rate has to be known
to 1e-4 (1.2). This is the quantitative form of the user's first concern, and it is the reason section 2 comes before any GPU
code. The fallback must also be batched: refused cells are compacted on the device, solved, and scattered back.

### 3.4 Precision

- Check C, float32 through `DeviceSurrogate` on NumPy. Full-rank fit (fisher_kpp_d10): 0 decision flips in 5,600 inputs,
  predictions within 2.7e-6 of float64 (the accepted errors' median is 3e-5). Rank-deficient fit (bruss_d20, rank 632 of 715,
  singular values kept down to 1e-10 of the largest): **5,319 of 5,600 decisions flip, every one from accepted to refused**.
  The gate's factor `M = V_r / s` spans ten decades and the out-of-span cap is 3.5e-9, both below float32's resolution.
- So: float64 for the gate as built. The failure is in the safe direction, but it is total. A float32 evaluation needs a
  float32-conditioned fit (a cut-off near 1e-5 instead of 1e-10, the out-of-span cap above 1e-5), which is a different
  estimator with a higher refusal rate and has to be selected and calibrated as such. Robertson's and CHEMAKZO's errors (1e-7)
  are also below float32's prediction noise (1e-6), harmless at this tolerance and visible at tighter ones.
- Measure, per served problem: decision flips and prediction differences between float32 and float64 on 1e6 column inputs,
  and the same between two batch sizes in float64 (which must be zero).

### 3.5 Comparators, and the estimate that decides whether to build

- **One CPU core** (p15's denominators): the number already reported.
- **A CPU node** running the same surrogate: the GPU gains 1-7x over it (3.2). The honest claim for the GPU is not speed over
  a node but staying on the device beside a GPU physics solver.
- **A batched exact solver on the same GPU**: the comparator that matters in the user's picture, and it is missing. The
  library's `integrate_ensemble` (`src/numerics/ode.py`, `src/kernels/ode.cu`) is explicit Runge-Kutta with a step per
  trajectory and one right-hand side (Lorenz 96): it is the right structure and not a stiff solver. A stiff one costs, per
  cell-step at d = 20, about 20 internal steps of a Jacobian, an LU (d^3 / 3 = 2,700) and a few solves: of the order of 1e5 to
  1e6 operations, against the surrogate's 4e4 to 7e4. The CPU ratio of 25-460 comes largely from BLAS against scalar Fortran,
  and a batch on a GPU vectorises both. **My estimate is 3-15x for the surrogate over a batched stiff solver at d of 20 or
  less, at the steady refusal rate.** The gain grows with d (the solver's d^3 against the surrogate's p (m + free + 64)), so
  the case for the surrogate on a GPU is the large systems, which are the ones not yet served.
- Batched stiff ensemble solvers exist outside this library (SUNDIALS with batched linear solvers, Julia's DiffEqGPU are the
  ones I know of; from memory, to be checked). Measure one of them, or a Rosenbrock kernel beside `ode.cu`, before building
  the surrogate's GPU path: if it is within 5x of the surrogate on the panel's problems, the GPU surrogate is worth building
  only for d above about 50.

### 3.6 Changes of 2x or more to the evaluation

- Evaluate once per step (the account reuses the gate's `b` and rho): 1.9x on the account protocol (check C: 14.9 to 7.8 us).
- Fewer terms: the cost is linear in p. p12's bases are the route; nothing new to add.
- Not 2x alone, but free: drop the out-of-span sketch on full-rank fits (half the gate, a quarter to a third of the total).

---

## 4. GPU training, as ART and AFINNI do it

### 4.1 Where training time goes

`train_s` in `records/r4g/*_340.json` is 6-36 s on eleven problems and 78-664 s on the other seven (CHEMAKZO 87, pollu 180,
primordial 252, hires 664). My rebuild of CHEMAKZO's deployed fit took 118 s, of which the four SVDs of a 28,900 x 792 design are a few seconds;
the rest is `P.step` labelling the mixed calibration states with the library's solver (896 us a state, p15). So:
- **The first 2x-10x in training is not a GPU.** Label with the fastest qualified exact solver: p15 has LSODA or VODE 9-24x
  faster than the library's step on 8 of the 10 problems I read (bruss_d20: 415 against 4,204 us; 7.7x on HIRES, 2.2x on
  lorenz96_d4). The condition is label accuracy:
  Robertson's and CHEMAKZO's envelopes sit at 1e-7 to 1e-8, at the solvers' disagreement (p15's `worst_vs_stored` is 6e-9 to
  1.5e-7), so tighten the tolerances there or the envelope calibrates on label noise (3.29's H4 failure).
- Labels are independent per state and are CPU work: the library's MPI ranks and work stealing are the right tool, not a GPU,
  until a batched stiff solver exists (3.5).

### 4.2 The fits on a GPU

- **The Gram route does not reproduce the recipe.** `A^T A` by `segment_cross` (per-chunk sequential sums, combined in order)
  is deterministic and fast, but it squares the condition number: singular values below about 1e-8 of the largest are lost in
  float64, and the recipe keeps them down to 1e-10 (`RCOND`). On rank-deficient fits (bruss_d20, chemakzo, robertson, hires)
  the retained rank, the coefficients and `M` would change. That is a deliberate numerical change to the estimator and to
  the gate, to be selected and calibrated again, not a port.
- **A tall-skinny QR reproduces it.** Householder QR of fixed row blocks, the R factors (p x p) combined in a fixed binary
  tree, then the SVD of the final R on one device: the singular values and `V` are A's to rounding, the condition number is
  not squared, and `U^T Y` comes from applying the same reflections to Y. Cost: N p^2 for the blocks (2e10 at N = 3e4,
  p = 800: under a second on a GPU) and p^3 for the last SVD.
- **The selection** is up to 12 degrees x 3 splits of independent fits and chains: task-parallel, no shared state.
- **The calibration chains** are sequential in steps and parallel over chains; their cost is the exact solver's (4.1).
- **Worth it?** Not at the panel's sizes: the linear algebra is seconds. It becomes worth it at N of 1e6 to 1e7 pairs or p
  of several thousand, which section 2's refusal-driven data and section 5's larger models would produce. Build the QR path
  when a fit first takes more than a few minutes on a node, and not before (AGENTS.md's bar).

### 4.3 What must stay deterministic

- The fit: a fixed blocking of rows by global index and a fixed reduction tree, so the model's bits do not depend on the
  number of ranks or devices (the library's rule for scatter and merge). `MPI_Allreduce`'s own tree is not fixed.
- The gate's decisions at inference: independent of the batch (3.2).
- The selection: it breaks ties in cost within 10% by timing (`generic4.select`), which 3.23 already caught changing a
  result. On a GPU, select by an operation count (terms, sketch columns), not by a measured time.
- The sketch: its seed is part of the model (`Fit4.sketch(seed=0)`), and the calibrated statistic is the sketched one, so
  the sketch matrices are saved with the model, not regenerated on the device with another generator.
- The account: S is a per-cell sum in step order; it is deterministic if the transport of S is (2.3).

---

## 5. Scaling across several GPUs

- **The least squares.** One reduction of p x p factors for the whole fit (p = 3,000: 72 MB per rank, once), against a
  gradient all-reduce at every step of data-parallel deep learning. Communication is not the limit at any size considered.
  The tree-combined QR (4.2) scales in N linearly per device; the limit is the last p x p SVD on one device, p^3 time and
  8 p^2 bytes, comfortable to p of about 3e4.
- **The selection and the chains.** Independent work: near-ideal scaling to the number of (degree, split, variant) fits,
  about 36 here, then idle. Labels scale over CPU ranks.
- **Inference.** No communication between cells; each device evaluates its own cells. What stops ideal scaling is the
  fallback: refused cells cluster in space (fronts, fresh inflow), so one device's fallback queue is long while others are
  empty. That is the load imbalance the engine's work stealing was built for; the fallback queue should be shared across
  ranks, and its imbalance is a metric (section 6).
- **What makes deep learning scale and this recipe not.** A network's capacity grows with its width and depth, so more data
  and more devices buy accuracy. One global polynomial has p fixed by the chart dimension and degree; once the variance is
  gone, more data does nothing (Nelson and primordial are bias-limited at 540 trajectories, round 4; p18's R4). More devices
  then only shorten a job that is already short. For data and devices to buy accuracy the capacity must grow with the data:
  more terms (bounded by p^3 and by the Lebesgue function's growth), or a partition with a local fit per cell of it, which is
  what ART and AFINNI are and what their per-segment sums and MPI scatter already scale (2.6). That is the comparison to
  state: this recipe's training is cheaper and more scalable per unit of model than a network's, and its model is smaller
  than the problems need.
- **What would stop it.** The exact-solver labels (a serial fraction on CPUs until a batched stiff solver exists);
  calibration data per region if the model is partitioned; the refused-data loop, whose rounds are sequential (each round's
  refusals depend on the last fit), three to five rounds at most by 2.2's bound.

---

## 6. Other inference metrics, and how to measure each

Ordered by how much each could change a decision.

1. **The certified horizon.** Accepted steps per cell before the account refuses, `tol / (2 lambda c)`, or infinite with a
   decaying account. Report it beside every refusal rate (1.1). Measure: p21.
2. **The refusal rate, split.** Fresh-start rate, steps to the steady rate, steady rate with a confidence bound (1e5 to 1e6
   cell-steps), each by cause (cap, envelope, account). One number hides all of 1.2.
3. **Fallback latency, not only throughput.** A step ends when its slowest refused cell is solved. Measure the per-step wall
   time's distribution (median, 99th percentile), the refused count per step and its spatial clustering, in the columns and
   in 2-D if the application is. A stiff solve on a hard cell takes many times the mean, so the 99th percentile step can be
   set by one cell.
4. **Evaluation per cell against batch size**, at 1e2 to 1e7 cells, with the device's launch overhead and memory limit
   visible; one evaluation per step (3.6).
5. **Memory.** Per model: coefficients p x free and the sketch 64 p doubles (0.5 MB on bruss_d20), nothing per cell. Per
   cell in flight: the design, p doubles (4-6 kB), which sets the batch; the account, one double. With a partition, per-leaf
   models: the model memory then scales with leaves and is the metric to watch.
6. **Decision stability.** Flips between batch sizes (must be 0), between float32 and float64 (3.4), between CPU and GPU
   (rounding only: report the share of cells within 1e-12 relative of a threshold), and across a restart (the account's
   state must be checkpointed with the cells).
7. **Transfer and compaction.** If the physics stays on the GPU and the fallback on the CPU, refused cells cross the bus:
   8 d bytes each way per refused cell, small beside the solve; the compaction (a stream compaction per step) and scatter
   are the device-side cost. Measure them at the steady and the transient rates.
8. **Energy or cost per cell-step** if the comparison is a node against a GPU: by 3.2 the two are within a small factor on
   throughput, so cost may decide.
9. **Conservation after projection** in float64 and float32 (the projection exponentiates log states), as the largest
   violation of each law over the columns.

---

## 7. Plan, in order

| # | What | Purpose | Cost | The result that would change the plan |
|---|---|---|---|---|
| 1 | p21: chains of 400 and 2,000 steps, three account arms; refusals by cause and step; one evaluation per step; BLAS pinned to one thread | whether anything is served beyond the benchmark's horizon (1.1-1.3) | data: 10-50x the test chains with LSODA or VODE, minutes; runs under an hour | C2 failing on a contracting problem: the horizon becomes part of every claim, and the surrogate is a short-horizon accelerator |
| 2 | Restate 3.31's table from (1): fresh-start rate, steady rate with a bound from 1e5 or more cell-steps, horizon, speed-up at each | the metric the user's 1/rho argument needs | one audit run | a steady rate above 1e-3 on the lattices: section 2.2's loop applies to them too |
| 3 | p20: refusal-driven data as leverage design in closed columns, full charts, with the column-distribution control | the coupled refusals (CHEMAKZO) with a bound | three rounds x five problems, under two hours | L1 failing with the cap satisfied: the off-bundle map is not in the model; go to 2.4 |
| 4 | POLLU with an off-bundle fit on the full chart (2.4) | reduced charts under coupling | one problem, under an hour | bias-limited at degree 2: POLLU and HIRES are closed as negative results |
| 5 | On a GPU machine: `gpu_eval.py` in float64 at 1e3 to 1e7 cells, the decision-stability checks, and a batched stiff solver on the same card on three problems | whether a GPU path is worth building, and against what (3.5) | a day, no library code | the surrogate within 5x of the batched solver at d of 20 or less: build the GPU path only for large d |
| 6 | Labels with the fastest qualified solver at tightened tolerances | up to 9-24x less labelling time, no GPU | small | label noise above the envelope on Robertson or CHEMAKZO: keep the library's step there |
| 7 | Only after 1-5: the recipe's fit in ART's leaves (2.6, section 5) | capacity that grows with data; the bias-limited problems | a new method | - |

Stop doing:
- quoting one refusal rate, or a speed-up, from 40-step chains;
- timing the hybrid with the gate evaluated twice and unpinned threads;
- widening the cap for coupled inputs (1.4), and a fixed burn-in rule (2.5);
- adding refused data on reduced charts without growing the chart (1.5);
- float32 for the gate as built (3.4);
- a Gram-matrix port of the fit presented as the same estimator (4.2);
- calling a problem "served" under a script where the solver takes more than half the steps (1.6).

---

## Appendix: the checks

All in `/tmp/claude-1000/-home-prajwal-github-repos-affini/32f886c8-be9f-433e-aef0-3afb16101b8f/scratchpad/c5/`, run from
`theory/panel` with `SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=../..:.:../recipe:<that directory>`. `common5.py`
rebuilds R4g's deployed surrogate at 340 trajectories as p17 does (`generic4.rebuild`, the same training and calibration
trajectories and calibration sets), with lambda taken from the record instead of searched again.

| Check | Script, output | What it does | Limits |
|---|---|---|---|
| A | `causes.py <problem>`, `causes_*.txt` (CHEMAKZO's output is in the session log only) | plain test rollouts and p17's sorted column at D = 0.05 on its fresh cells; each refusal attributed to the cap on `b`, the out-of-span share, the offset, the envelope or the account; the true one-step error of every input from the library's step | one column per problem; five problems (chemakzo, bruss_d20, fisher_kpp_d10, burgers_d20, robertson); POLLU did not finish in three minutes |
| B | records only | `rollout_solved_by_step`, `account_check`, `budgets`, `mixing_chained`, `fresh_calibration` of `records/r4g`, `records/p18`, and `data/p17_audit.json` | - |
| C | `batch.py <problem>` | one-thread time per cell of `Surrogate4.evaluate`, of p17's protocol (`Account.refuses` then `predict`) and of `gpu_eval.DeviceSurrogate` in float64 and float32 on NumPy, at 100 to 50,000 cells; decision flips and prediction differences on 5,600 real inputs | two problems (fisher_kpp_d10, bruss_d20); float32 on a CPU stands in for a GPU's arithmetic, not its speed |
| D | `long.py <problem>`, `long_*.txt` | the 40 test cells continued to 400 steps with the library's step, the deployed surrogate and account; refusals by 40-step window, with those due to the account alone | three problems (fisher_kpp_d10, burgers_d20, lorenz96_d4); plain chains only |

The GPU figures in section 3 are estimates from operation counts and typical device throughputs; none was measured.
