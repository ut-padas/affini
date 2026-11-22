# Round 10 (Fable): the four-factor memo, the generic recipe (R4, P2, R5), and counsel on what to build next

Reviewer: Claude Fable 5.2. Reviewed at commit `40fa52f` plus the uncommitted `theory/panel/data/records/r5/` and
`data/r5_data_scaling.json` (complete: 24 records, `logs/r5.log` ends `EXIT 0`). Scope: `theory/MEMO.md` sections E, T5, 8–11;
`theory/panel/{generic,r4_generic,p2_tradeoff,r5_data_scaling,p1_descriptors}.py` and their records; `theory/recipe/`.

Four read-only checks were run (each under three minutes, no file written outside the session scratch directory). Their
sources are reproduced in the appendix at the end of this file, so they can be copied to `theory/reviews/scratch-round-10/`.
Run from `theory/panel` with
`SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=../..:.:../recipe ../../.venv/bin/python <script>`.
Every number from them is **post hoc** (computed after reading the records, thresholds chosen by the reviewer) and scored on the
test trajectories: they motivate pre-registered reruns and are not results.

**Counts: 3 critical, 7 major, 6 minor.**

## Summary

The mathematics of section E, Proposition 5.1 and Corollary 5.3 is correct for an exact least-squares fit. The trouble is in
how the recipe uses it. The refusal threshold `tol / (2 e_v) - 1` puts the *held-out error* where the proposition has the *best
approximation error*, and compares it with an upper bound of the Lebesgue function that cannot go below 1 and is typically 2–25
on in-distribution inputs. So the rule refuses every input unless the held-out worst error is 4 to 50 times below the tolerance.
On at least seven problems recorded as "refuses every input" the same fit, with no gate at all or with a threshold calibrated on
held-out data, holds the tolerance on all 40 test rollouts with 0–2.5% of the steps solved (check A, B, D). The headline "every
other problem refuses every input" is therefore a property of the threshold, not of the problems; and "extra data fixes the
sampling factor but not the approximation factor on the Fisher–KPP and Allen–Cahn ladders" is not what the records show (the
in-sample residual keeps falling with the degree; the held-out maximum sits at first-step states whose Lebesgue bound is
400–10^5; on Allen–Cahn the error floor is the chart's own, which the k rule allows up to tol/2). In the other direction the
gate is too lax where the chart discards coordinates: on HIRES the error of accepted mixed states is the decoder residual to a
rank correlation of 0.99–1.00, which a residual term would catch (check C). Section P, which the memo cites six times as the
generic evidence, does not exist in `MEMO.md`, and section 10's recipe and section 11's first open item are still HIRES-specific.

---

## Findings

### 1. (critical) The refusal threshold double-counts sampling and has a floor: it refuses problems the fit already serves

`generic.py:237` sets `lam_star = TOL / (2 * e_v) - 1`, with `e_v` the worst held-out one-step error over all validation states
of three splits (`generic.py:236, 260, 265`), and refuses when `lebesgue_bound > lam_star` (`generic.py:155`).

Three separate defects:

- **`e_v` is not `e*`.** Proposition 5.1 bounds the error at `t` by `(1 + Λ(t)) e*`. A held-out error at `v` is itself at most
  `(1 + Λ(v)) e*`: `e_v` already contains the sampling factor of the worst validation state (the highest-Λ one: a first-step state
  in every case examined). The rule multiplies it by `1 + Λ(t)` again.
- **The bound has a floor.** With constants in the space, `Λ_X(t) ≥ |Σ ℓ_i(t)| = 1`, and `sqrt(N) ‖ℓ‖_2 ≥ ‖ℓ‖_1`. So nothing is
  accepted unless `lam_star ≥ 1`, i.e. `e_v ≤ tol/4`. Worse, over the training inputs the mean of `N φᵀ(AᵀA)⁻¹φ` is exactly `p`
  (the trace of the hat matrix), and by the Kiefer–Wolfowitz equivalence theorem no design has `sup_t N φᵀ(AᵀA)⁻¹φ < p`: the
  bound's supremum is at least `sqrt(p)` for *any* sample set. Raising the degree raises the floor.
- **The factor 2** is an unexplained margin. Factor 3 of section E says the margin should be the accumulation `A_n`
  (`dynamics.json` has it per problem); 2 is too small for POLLU and too large for contracting problems.

Evidence (check A; the fit at the degree with the smallest `e_v`, refit on all training trajectories as `p2` does; test inputs):

| problem (extra) | q, p | `e_v` | rule's Λ* | bound min / median | true Λ min / median | ungated test one-step worst | rule refuses |
|---|---|---|---|---|---|---|---|
| lorenz96_d5 (0) | 7, 792 | 1.56e-3 | 2.21 | 5.6 / 8.0 | 2.3 / 3.3 | 2.75e-3 | 100% |
| lorenz96_d10 (1,600) | 4, 1001 | 1.48e-3 | 2.39 | 3.2 / 4.4 | 2.1 / 2.5 | 1.95e-3 | 100% |
| pollu (0) | 2, 231 | 4.89e-3 | 0.02 | 8.0 / 23.6 | 6.0 / 16.5 | 1.15e-3 | 100% |
| fisher_kpp_d2 (0) | 8, 45 | 1.92e-2 | −0.74 | 1.8 / 3.7 | 1.3 / 2.1 | 2.26e-3 | 100% |
| bruss_d4 (0) | 5, 126 | 9.93e-3 | −0.50 | 2.0 / 2.6 | 1.5 / 1.8 | 8.68e-3 | 100% |
| allen_cahn_d10 (1,600) | 3, 84 | 7.38e-3 | −0.32 | 2.5 / 3.9 | 1.4 / 2.4 | 6.55e-3 | 100% |
| burgers_d10 (0) | 2, 28 | 9.35e-3 | −0.47 | 1.06 / 1.07 | – | 9.11e-3 | 100% |
| allen_cahn_d3 (0) | 5, 56 | 9.08e-4 | 4.51 | 2.6 / 5.7 | 1.2 / 2.7 | 2.89e-4 | 72.8% |
| hires (0) | 8, 165 | 5.53e-4 | 8.04 | 3.8 / 6.2 | 1.7 / 3.0 | 5.73e-5 | 30.9% |

No test input errs beyond the tolerance in any row, ungated. Rollouts over all 40 test trajectories (checks B and D):

| problem (extra) | ungated: worst, solves | calibrated gate: worst, solver share | recorded (R4/R5) |
|---|---|---|---|
| lorenz96_d5 (0) | 2.75e-3, 0 | 5.2e-4, 0.1% | refuses all |
| lorenz96_d10 (1,600) | 1.95e-3, 0 | 1.1e-3, 1.4% | refuses all |
| fisher_kpp_d2 (0) | 3.13e-3, 0 | 3.8e-3, 1.1% | refuses all |
| burgers_d10 (0) | 9.11e-3, 0 | 5.0e-3, 2.5% | refuses all |
| bruss_d4 (0) | 1.59e-2 (fails) | 6.3e-3, 0.3% | refuses all |
| allen_cahn_d3 (0) | 8.67e-3, 0 | 8.7e-3, 0.8% | 3.7e-4 at 72.8% solved |
| hires (0) | 4.70 (fails) | 2.75e-3, 1.3% | 5.4e-4 at 30.9% solved |
| allen_cahn_d10 (1,600) | 4.03e-2 (fails) | 3.4e-2, 3.7% (fails) | refuses all |
| pollu (0) | 29.8 (fails) | 6.1e-2, 53% (fails) | refuses all |

"Calibrated" is one rule, chosen by the reviewer after seeing check A: accept when the bound is at most `L`, `L` the largest
held-out bound below the first held-out error above tol/2 (three splits pooled), capped at the 99% quantile of the held-out
bounds. Seven of nine hold the tolerance with at most 2.5% of the steps solved; POLLU and allen_cahn_d10 fail through
propagation (finding 7), and HIRES and POLLU show that *some* gate is needed (ungated they diverge). Mixed inputs were not
tested with this rule.

Fix. Calibrate the gate on held-out data instead of inserting `e_v` into the proposition:
1. score every held-out state by `s(v) = err(v) / (1 + b(v))`, `b` the bound; take `ê` as a high quantile of `s` over held-out
   *trajectories* (split-conformal at the trajectory level, which is T4's distributional statement and should be labelled so);
2. refuse when `(1 + b(t)) ê > tol / A` with `A` from factor 3 (finding 7), **and** when `b(t)` exceeds the largest bound at which
   an error was actually validated (no extrapolation in Λ beyond the calibration range);
3. state in the memo that `b ≥ 1` always and `sup b ≥ sqrt(p)`, so a threshold below these is "refuse everything" by
   construction;
4. pre-register the rerun of R4 and R5 with this rule before running it.

### 2. (critical) "Served" is true for all 37 problems, and the selection then reports degree 1 for every problem it does not serve

`generic.py:252–254` scores an all-refusing configuration as `chain_worst = 0.0`, `solver_share = 1.0`; `generic.py:263–265`
then counts it among the configurations "within the tolerance". Consequences, all visible in the records:

- `served` is `True` in all 37 R4 records and all 24 R5 records (`grep -c '"served": true' data/r4_generic.json`), including
  Nelson, Oregonator and van der Pol. G1, G2, G4 and G7 as committed are stated in terms of "serves"; read literally G1 fails
  (17 of 17 sparse problems "served") and G7 fails (Nelson "served"). The commit message of `40fa52f` instead applies "solver
  share below 50%", a definition introduced after the run. That is a reasonable definition, but the verdicts are then post hoc
  and must be labelled so (the memo's own provenance rule).
- Among tied all-refusing configurations the minimum of `(solver_share, degree)` is degree 1. So the recorded `degree`,
  `lambda_star`, `terms`, `rank` of 31 problems are those of the affine fit, not of the best one (R4 table: `q = 1`, `Λ* ≈ −0.9..−1`
  on every unserved row; e.g. lorenz96_d10 records `e_v = 0.40` where its degree-3 fit has 0.045). D1 and D2 are stated "at the
  selected degree" and "the selected degree rises with M", which this makes meaningless.
- Only `lam_star < 0` is treated as all-refusing, but everything is refused for `lam_star < min bound` (≥ 1): lorenz96_d5
  (Λ* = 1.84), lorenz96_d10_m1600 (2.39), pollu (0.02) run full solver chains for every configuration and split.
- A configuration that accepts inputs and exceeds the tolerance loses to an all-refusing one. That is the right fallback, but
  it is reported as a served problem.

Fix: `served = ok and solver_share < threshold` with the threshold fixed in the docstring (and the rule that `solver_share = 1`
is "not served"); when nothing is served, report the configuration with the smallest held-out one-step error; treat
`lam_star < 1` as all-refusing; write the refusal share of the *held-out* chains into the selection criterion's record.

### 3. (critical) "Approximation-limited" on the Fisher–KPP and Allen–Cahn ladders is not what the records show

The claim to be checked: "extra data fixes the sampling factor but not the approximation factor on the Fisher–KPP and
Allen–Cahn ladders (held-out error stalls at 0.02–0.2 for every degree up to 8)".

- **Range.** On Allen–Cahn the best held-out error with 1,600 extras is 7.4e-3 (d = 10, q = 3) and 1.0e-2 (d = 20), below or at
  the tolerance, not 0.02–0.2 (R5 records, `e_v by q`). On these ladders the degree stops at 7 (k = 6: `n_terms ≤ 3000`), not 8.
- **Non-monotone in q is the signature of sampling, not approximation.** `e*` is nonincreasing in the degree; a held-out error
  that falls and then rises (fisher_kpp_d10, M = 1,600: 0.63, 0.19, 0.18, 0.12, 0.092, 0.30, 0.35) is `(1 + Λ) e*` with Λ taking
  over. Check B on split 0 (M = 1,600):

  | fisher_kpp_d10, q | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | in-sample max (trajectories) | 0.64 | 0.24 | 0.083 | 0.029 | 0.013 | 6.8e-3 | 7.2e-4 |
  | held-out max | 0.58 | 0.16 | 0.069 | 0.032 | 0.039 | 0.089 | 0.99 |
  | held-out median | 1.3e-2 | 4.8e-3 | 1.3e-3 | 3.0e-4 | 1.0e-4 | 3.3e-5 | 8.0e-6 |
  | bound at the worst held-out state | 4.3 | 8.7 | 15 | 24 | 413 | 2.6e3 | 9.5e4 |
  | step of the worst held-out state | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

  The in-sample maximum (an upper bound of `e*` on the samples while `p` is well below the number of independent inputs, so
  through q = 5 here) falls by a factor 2.5–3 per degree with no stall. The held-out maximum is always a first-step state, with a
  bound that explodes from q = 5. This ladder is **sampling-limited at its first-step states, with a slowly converging
  approximation** (about 0.4 decades per degree): both factors, and more data of the right kind is still the lever.
- **Allen–Cahn's floor is the chart's.** On allen_cahn_d10 the held-out maximum is 4.0e-3 to 5.1e-3 for q = 3..6 (0% of
  held-out states beyond the tolerance), and the in-sample maximum 6.5e-3 to 2.7e-3. The k rule accepted k = 6 with a lift error
  of 3.7e-3 (`k_curve`; `generic.py:212` accepts up to tol/2). A fit in chart coordinates cannot beat the variation of the step
  over the chart's fibres, which that rule allows up to tol/2, while the refusal rule needs `e_v ≤ tol/4` at the very least
  (finding 1). The two rules are inconsistent with each other; the same holds for fisher_kpp_d10 (3.9e-3), bruss_d10 (2.1e-3),
  allen_cahn_d20 (3.8e-3).
- **fisher_kpp_d3 is cap-limited**: in-sample and held-out maxima fall together (0.34 at q = 2 to 6.3e-3 and 5.1e-3 at q = 8, about
  0.29 decades per degree, a Bernstein parameter near 2 in the chart coordinates) and 0% of held-out states exceed the tolerance
  at q = 8. With k = 3, degree 12 has 455 terms; the cap `qmax = 8` (`generic.py:297`) stops it.

Fix: the diagnosis per problem must separate the factors with three numbers per degree, all already computable from the
records' data: the in-sample maximum (factor 1 on the samples), the held-out median and maximum, and the bound and the step
index at the held-out maximum (factor 2). Classify "approximation-limited" only when the in-sample maximum stalls with the
degree while `p` is far below the number of independent inputs. Set the k rule's lift threshold from the error budget the
gate needs (for example tol/10), not tol/2, and record the lift error as the fit's floor.

### 4. (major) Section P does not exist; the generic results are not in the memo

`grep -n "section P\|Section P" theory/MEMO.md` gives lines 80, 132, 603, 715, each sending the reader to section P for the
evidence "on every problem"; `grep -n '^## ' theory/MEMO.md` has no such section, and `grep -n 'r4_generic\|p2_tradeoff\|r5_data\|p1_descriptors' theory/MEMO.md`
returns nothing. `generic.py:2` and `p1_descriptors.py:2` cite it too. So the memo's generic claims (U8's explanation, the
descriptors of section E, line 715's "the panel is [the evidence]") rest on text that is not there. Also stale: the header
(line 8, "Revision 5 answers seven review rounds"), the provenance table (no rows for R2, P1, R4, P2, R5, although their
predictions are pre-registered in `a42863b`, `298dabf`, `fff5107`, `bd86a36`), the synthesis table of section 9 (no U8 row,
no T5 entry for it), `theory/README.md` (no `p2`, `r5`) and `REVIEWS.md` (no entry for `recipe-round-1-sol.md`).

Fix: write section P after the rerun of finding 1, with one table per factor over all 38 problems by class, the predictions
and verdicts as committed, and the post hoc redefinitions marked.

### 5. (major) HIRES-specific and single-problem statements that remain outside Appendix A

| Where | Statement | What to do |
|---|---|---|
| Section 10, item 1, lines 669–681 | "On HIRES (8.5) the practical recipe that the evidence supports … degree 10 holds 1.5e-4 … a single kick of 0.01 dex … ratio 1.532 … best 2.3e-2" | This is the ranked foundation of the whole memo and it is one problem's recipe. Replace it with the generic recipe and its panel results; move these lines to A.2/A.3 |
| Section E, line 135–136 | "Section 10 turns them into a rule for choosing a scheme" | Section 10 contains no such rule. Write it (counsel (a), item 6) or delete the sentence |
| Section 11, first item, lines 697–704 | "one HIRES surrogate for both regimes … POLLU with several hundred trajectories … on HIRES, POLLU, Fisher–KPP" | Restate as panel items by class |
| Section 9, row U2, §8 column | "est.: 1.53 vs obs. 5.2e-4 (sampled constants, Robertson) … HIRES value-only fit 1.5e-4 on 40 rollouts (`g5`) … (`g5`, `g6`)" | Keep one clause with a pointer to Appendix A; add the panel's numbers |
| Section 8, Scope, lines 582–583 | "A normal correction B(u)(z − D(u)) … an abundance floor" | Generic in wording, but motivated only by the HIRES recipe; fine once section 10 is generic |
| T2, line 299 | "review 1 test A: such a HIRES chart diverged, its invariant drifting by 38 dex" | A single-problem check in a general section: label it "(one problem)" or move it |
| Section 0, U1–U5 | Brusselator, Robertson (U2, U3), `lotka_volterra` (U4) | These are observations to explain, legitimately specific; say so in one line, and add how many problems each was seen on |
| T1 checks (lines 240–250), T6 checks (525–533) | Robertson, HIRES, POLLU, Brusselator, Fisher–KPP | Checks of general statements on one to three problems. Either extend each to the panel by class or head the block "single-problem checks" as Appendix A does |
| Section G, line 635 | "measured on HIRES in Appendix A.3, and to be measured on the panel" | Correctly flagged; it is an open item until done |

T5 itself (Proposition 5.1, Corollary 5.3, the three regimes) and section E are stated generically.

### 6. (major) Proposition 5.1 and Corollary 5.3: correct, but the deployed map is outside their hypotheses, and (b) hides the term that matters off the data

Proposition 5.1 is correct as stated (exact least squares, full column rank; the proof is one line and right), and
`sqrt(N)‖ℓ‖_2 = sqrt(φᵀG⁻¹φ)` matches `Fit.lebesgue_bound`. Corollary 5.3 (a), (c), (d) are correct; (c) is an identity plus
`c_g, c_Ψ ≥ 0`. What needs stating:

- **(b) is true and nearly empty off the data.** "Every accepted input errs by at most `(1 + Λ*) e*_V(X ∪ {t})`": the best
  approximation error *includes the point of use*. When the space consists of functions of the chart coordinates and `t` is
  displaced across the chart's fibres, no `q ∈ V` can follow `Φ(t)`, so `e*_V(X ∪ {t})` is large while `Λ_X(t)` is small. That
  is the HIRES blind spot; it is not an exception to the corollary but its hidden hypothesis. State (b) with the explicit
  assumption `e*_V(X ∪ {t}) ≤ ê` on the accepted set, and add the decomposition for a chart space with a decoder `D`,
  `t' = D(π t)`: `|Φ(t) − Ψ(t)| ≤ |Φ(t) − Φ(t')| + (1 + Λ_X(πt)) e*_V(X ∪ {t'}) ≤ L_⊥ r(t) + (1 + Λ) ê`, `r(t) = ‖t − D(πt)‖`
  (needs `π∘D = id` on the chart box, or its defect as a further term). This is the residual term of finding 9.
- **Ridge.** With `ridge > 0` (`RIDGES = (0, 1e-6)`) the fit does not reproduce `V`, so the proof's first step fails. The
  correct statement has a bias term: `|f − Rf|(t) ≤ (1 + Λ^λ(t)) e* + |q*(t) − Rq*(t)|`. Either prove it with that term or state
  that the proposition covers `ridge = 0` only. Rank deficiency (full column rank is assumed; `Fit.rank` is recorded but
  nothing acts on it) needs the same care.
- **Post-processing.** The deployed map clips the output to `[zlo, zhi]` (`generic.py:148`), projects onto the invariants and
  floors (`project`). Clipping to a box that contains the truth does not increase the error; the box is fitted on the fold
  (`X.min − 2`, `X.max + 0.5`) and an off-data truth can lie outside it. The projection can increase the sup error of
  individual components. Say that 5.3 (b) bounds the raw fit and that the deployed map's error is measured, not bounded.
- **Label noise.** The values are the solver's: add `Λ_X(t) η` for a solver error `η`. With thresholds of 10²–10³ and
  benchmark tolerances this is not negligible in general.
- **"e* estimated by held-out residuals"** (line 421, 435): held-out residuals estimate `(1 + Λ) e*` at the validation states;
  they give a *lower* bound `e* ≥ max_v err(v)/(1 + Λ(v))` and, with the in-sample residuals, an upper bound of `e*` on
  `X ∪ val` only. Neither bounds `e*_V(X ∪ {t})` at a new `t` without an assumption. Say which estimate is used and what it
  assumes (finding 1).
- **Regime (a)** ("targets among the samples: Λ stays moderate as p grows") holds for the true Λ at a sample only in the
  limits (`Λ = 1` at interpolation); between them it is unproved, and the *computable bound* grows like `sqrt(p)` on the samples
  themselves. State it for the bound the recipe uses.
- **(c) assumes the solve costs the same on refused and accepted inputs.** Refused inputs are first-step, transient states,
  where an implicit solver takes the most steps. The cap is `1/ρ_c` with `ρ_c = E[c_Φ 1_refused] / E[c_Φ]`, the *cost-weighted*
  refusal rate. `p2` uses the mean of 40 random test states (`baseline_cost`, `step_maps.py:138–141`).

### 7. (major) The propagation factor is in the decomposition and absent from the rule

Section E's estimate is `A_n max_j (1 + Λ) e*`. The rule budgets tol/2 per step and ignores `A_n`. Evidence (check D): POLLU's
quadratic fit has a test one-step worst of 1.15e-3 (no state beyond the tolerance) and its rollout reaches 29.8 ungated and
6.1e-2 under the calibrated one-step gate with 53% of the steps solved; allen_cahn_d10 (M = 1,600) has one-step 6.5e-3 and
rollout 3.4e-2. Conversely, the selection's held-out chains do include propagation, which is why the recipe never accepted a
failing configuration. Fix: per-step budget `tol / A_n` with `A_n` from `dynamics.json` (or the held-out chain's own ratio of
chained to one-step error, which is cheaper and specific to the fit), predicted per class before the rerun: contracting
problems should need no margin, neutral and chaotic ones a margin of order `A_n`.

### 8. (major) R4 and R5 verdicts against the committed predictions

R4 (`40fa52f`), with "served" read as "solver share below 50%" (post hoc, finding 2):

| | Committed | Records | Verdict |
|---|---|---|---|
| G1 | at most 2 of 17 sparse served | chemakzo only | holds under the post hoc definition; fails as coded |
| G2 | at least 8 of 19 covered, among them bruss_d2, fisher_kpp_d2, allen_cahn_d2, robertson, hires; not oregonator, van_der_pol | 4 (allen_cahn_d2, bruss_d2, lorenz96_d4, hires), 5 with allen_cahn_d3 at 73% solved | fails. The stated reason "fisher_kpp_d2 needs degree > 8" is not supported: its degree-8 fit has test one-step 2.3e-3 and an ungated rollout of 3.1e-3; one held-out state in three splits sets `e_v = 1.9e-2`. The failure is the threshold |
| G3 | rollout within tol in ≥ 80% of served | 5 of 5 (2e-6 to 5e-4) | holds |
| G4 | solver share < 10% in at least half of served | 4 of 5 | holds under the post hoc definition |
| G5 | mixing within tol wherever served; share +10 points at w = 0.1 in most | hires 0.381 at w = 0.01, 2.0e-2 at 0.1; +10 points on 3 of 5 (not bruss_d2, lorenz96_d4) | first half fails (stated); second half holds 3 of 5, which the commit message does not report |
| G6 | primordial finishes | finishes after the lift guard added in `bd86a36` | fails as committed (stated honestly) |
| G7 | Nelson, both KIDA ranges not served | Nelson: refuses all. `records/r4/kida_dense.json` is `failed` (KeyError), `kida_diffuse` absent | **open**, not "holds": two of its three problems have no result |

R5 (uncommitted; predictions in `bd86a36`), at the best degree since the recorded one is degree 1 (finding 2):

| | Committed | Records | Verdict |
|---|---|---|---|
| D1 | held-out error falls ≥ 10-fold from M = 0 to 1,600 on fisher_kpp_d10, allen_cahn_d10, burgers_d10; ≥ 2 of 3 served | 0.505 → 0.092 (5.5×), 0.0636 → 0.0074 (8.6×), 0.00935 → 0.0014 (6.6×); burgers only | fails on both counts under the rule. With a calibrated gate allen_cahn_d10's one-step error is within tolerance and its rollout is not (finding 7) |
| D2 | selected degree rises with M; served rollouts within tol | burgers_d10 2 → 3, rollout 5.2e-5 | holds on the one served; untestable on the others |
| D3 | lorenz96_d10 not served at any M ("approximation and term count, not only sampling") | refuses all at Λ* = 2.39 with `e_v = 1.48e-3`, `e_v` by degree 0.40, 0.061, 0.0077, 0.0015 | holds by the rule, and **its stated reason is refuted**: convergence is geometric, the ungated rollout holds 1.95e-3 with no solve, and degree 5 was excluded by 3 terms (`n_terms(10, 5) = 3003 > 3000`, `generic.py:245`) |
| D4 | POLLU, Nelson not served at M = 1,600 | both refuse all | holds (POLLU also fails with a calibrated gate; Nelson's `e_v` is 1–9) |
| D5 | fisher_kpp_d3 and lorenz96_d5 served at M = 400 | lorenz96_d5: 2.8% solved, 7.4e-6; fisher_kpp_d3: refuses all | half fails. fisher_kpp_d3 is cap-limited (finding 3) |

Stated more strongly than the data allow, beyond the above:
- "every other problem refuses every input": true of the rule, false as a statement about the fits (finding 1).
- "with more data lorenz96_d5, burgers_d10, burgers_d20 [are served]": correct. But lorenz96_d5 and burgers_d10 hold the
  tolerance at M = 0 with a calibrated gate (5.2e-4 at 0.1% solved; 5.0e-3 at 2.5%), so "data fixed them" is not established;
  the extra data lowered `e_v` past the threshold's floor.
- k differs between R4 and R5 for the same problem (bruss_d10 9 → 8, pollu 20 → 9), so those rows compare different charts;
  POLLU's best `e_v` *rose* from 3.8e-3 (k = 20, M = 400) to 6.7e-3 (k = 9, M = 1,600).
- P2's T3 first half ("the effective speed-up is at most 1/ρ") could not have failed: it is the identity of 5.3 (c) evaluated
  with the same numbers. By the memo's legend it is a sanity check.

### 9. (major) The gate is blind across the chart's fibres; the decoder residual sees what it misses (HIRES, check C)

On HIRES test states mixed once (the states `p2` used), among the inputs the Lebesgue rule accepts:

| | w = 0.01 | w = 0.1 |
|---|---|---|
| Lebesgue rule: refused, accepted worst | 62.7%, 1.12e-2 | 73.8%, 1.24e-2 |
| rank correlation of the error with the Lebesgue bound | 0.11 | 0.38 |
| … with the linear normal residual `‖(I − V_k V_kᵀ)(c(z) − μ)‖` | −0.08 | 0.12 |
| … with the quadratic-decoder residual | 0.22 | 0.67 |
| … with the quartic-decoder residual (the degree the k rule validated) | **0.99** | **1.00** |
| median (max) of error / quartic residual | 1.48 (5.5) | 1.49 (3.5) |
| adding "residual ≤ its training maximum" (1.5e-3): refused, accepted worst | 67.4%, 2.2e-3 | 84.2%, 2.2e-3 |

The error of an accepted mixed state *is* its displacement from the decoder's image, times about 1.5 (the same ratio as
section 10's "1.532"). The linear residual is useless because the set is curved. This is one problem and post hoc. Note the
structural fact that makes it generic: the blind spot exists exactly when `k < free`; with a full chart the residual is zero and
the Lebesgue bound sees every direction (chemakzo, k = 5 = free: 91% refused under mixing, accepted worst 1.9e-5).

Fix and test: counsel, "The residual term" below.

### 10. (major) The gate's cost and the speed-ups

- `lebesgue_bound` solves a `p × p` triangular system per input: `O(p²)`, against `O(p · free)` for the fit. It is 6.6 times
  the fit's cost on lorenz96_d4 (9.9 against 1.5 µs) and it, with 5.8% refused, takes the speed-up from 184 to 10 (`p2.log`).
  A Gaussian sketch removes the square: precompute `S R⁻ᵀ` (`r × p`, `r` about 32) and use `sqrt(N) ‖S R⁻ᵀ φ‖`, `O(r p)` per
  input, accurate to about `1/sqrt(r)`, ample for a threshold calibrated on the same sketch. A box test in chart coordinates
  (`O(k)`) can accept or refuse most inputs before it.
- The deployed `refuses` also evaluates the whole prediction and projection (`generic.py:154`), and `predict` evaluates it
  again: the deployed cost per accepted step is `gate + 2 predict`, while `p2`'s formula has `gate + (1 − ρ) predict`
  (`p2_tradeoff.py:99`). The design matrix is built three times per step.
- The timings are one unrepeated call on at most 2,000 inputs with no warm-up (`p2_tradeoff.py:89–94`): 0.3 µs × 2,000 is
  0.6 ms. AGENTS.md's measuring rules ask for repeated, interleaved rounds. The 2,311× and 876× are per input *in a batch of
  2,000*, against a solve cost per state measured one state at a time; say so, and give the batch size.
- Speed-ups are quoted for the rule's refusal rate; under a calibrated gate hires goes from 31% to about 1% refused, so every
  "effective" number of `p2` changes with finding 1.

### 11. (minor) The degree cap, the term cap and the k rule

- `qmax = 8` binds on fisher_kpp_d2 (`e_v` still falling at q = 8, 45 terms) and fisher_kpp_d3; `n_terms ≤ 3000` excluded
  lorenz96_d10's degree 5 by three terms. Neither cap is derived. Replace both by the stability condition the theory gives:
  admit a degree while the held-out bound's median stays below a stated value, or while `p log p` is below the number of
  independent inputs (trajectories plus extras), and record the degree at which the cap, not the error, stopped the search.
- The pair count (`generic.py:244`) counts every step of every trajectory; the independent inputs are closer to the number of
  trajectories for the transient states (the R5 docstring says this itself). `1.5 p ≤ n_pairs` therefore admits degrees whose
  bound is 10³–10⁵ at first-step states (finding 3's table).
- The k rule validates a *polynomial decoder* by solving at decoded states, although the deployed fit never decodes; it is
  sufficient, not necessary, and its threshold of tol/2 is the floor of finding 3. It uses split 0's validation trajectories,
  which are fit trajectories of splits 1 and 2 (a mild leak into the selection, none into the test scores).
- For `free ≤ 40` a failed reduction falls back to the full chart, where the total-degree space allows degree 2–3 only (POLLU
  231 terms at q = 2, lorenz96_d40 q ≤ 2). "Not served" there says nothing about the problem; it says a total-degree space in
  20–40 coordinates is the wrong space.

### 12. (minor) `e_v` is a maximum over thousands of points and three splits

One state sets it (fisher_kpp_d2: 1.9e-2 held-out against 2.3e-3 on test). An extreme statistic of a heavy-tailed error (the
tail is the high-Λ states) is a noisy input to a threshold; the calibrated score of finding 1 removes the dependence, since
it divides each error by its own `1 + b`.

### 13. (minor) Clipping defeats the non-finite check

`_raw` clips the fit's output to `[zlo, zhi]` before `refuses` tests finiteness (`generic.py:148, 155`): an overflowing
polynomial becomes a finite value at the box's edge, and only NaN is caught. The Lebesgue bound is huge in such cases, so no
recorded result changes; test the unclipped output.

### 14. (minor) `np.linalg.solve` on an exactly singular factor

`Fit.__init__` falls back to `lstsq` when a diagonal entry of `R` is zero, but `lebesgue_bound` still calls
`np.linalg.solve(self.R.T, …)`, which raises `LinAlgError` (not silenced by `errstate`). No record hit it (ridge 0 fits with
rank deficiency would). Refuse everything when `rank < p`, or compute the bound from the pseudo-inverse.

### 15. (minor) The mixing in `p2` and in R4 differ

`p2_tradeoff.mixed` mixes each test state once and steps it; `generic.mixing` mixes at every step of a chain. The commit
message's "hires accepts states erring 1.3e-2" is the former, the 0.38 of G5 the latter. Both are useful; name them (one-step
mixed, chained mixed) wherever quoted.

### 16. (minor) KIDA

`recipe/logs/r3_kida.log` ends "STOPPED by the operator": the in-sample recipe stepped decoded states 3–7 dex off the data.
The generic recipe's `lift_error` guard (`Zd > 300`) would not stop that either (3–7 dex off is far below 300). For KIDA the
k rule should refuse to solve at a decoded state whose decoder residual on the fit data exceeds a bound, or use the
no-solve criterion of counsel (b). Until KIDA has run, no statement about the 578-species target is supported.

---

## Counsel

### (a) Making the Chebyshev schemes effective where the approximation is slow

What the records say first: after finding 3, the problems with a genuinely slow approximation are the Fisher–KPP ladder (0.3–0.4
decades per degree in chart coordinates), and possibly Allen–Cahn above its chart floor. Lorenz-96 converges fast and is limited
by the number of terms in 10+ coordinates; POLLU and Nelson have no validated reduction. So (a) addresses one or two classes,
and one test decides which of its tools apply.

**The deciding test for items 1–3 (fibres).** All three tools assume the slow decay comes from a feature that is *aligned with
an axis and stays at the same place as the other coordinates vary*. Measure that directly. On each of fisher_kpp_d3, d10,
allen_cahn_d10, burgers_d10 and one control (lorenz96_d5): take 30 visited states, and through each, one fibre per chart axis
at 65 Chebyshev points across the visited range, lifted by the validated decoder (about `30 × 65 × k` solves, 12,000 for k = 6;
each a millisecond-scale solve). Per fibre compute (i) the Chebyshev coefficients and whether the tail is geometric (`log|c_n|`
linear in `n`: a pole at distance δ) or algebraic (linear in `log n`: a kink, such as the floor clip of `Transform.clip`);
(ii) the nearest pole `x_0 ± iδ` of an AAA fit. Report per axis the spread of `x_0` across the 30 fibres relative to δ.
Pre-register: on Fisher–KPP the spread exceeds δ (a travelling front's position depends on the other coordinates). Caution from
the memo's own history: review 1 test A found that fibre-coefficient slopes did not discriminate regimes on Robertson, so
report the fit's residual with each slope.

1. **Per-axis rational bases (AAA, lightning poles).** Mechanism: a pole pair at distance δ from an interval of half-width `h`
   limits the Bernstein parameter to about `1 + δ/h`, so polynomials need `O(h/δ)` degrees per decade; rational functions with
   poles clustered at the singularity converge root-exponentially whatever δ. In a tensor format the basis must be *linear and
   shared along an axis*: polynomials plus partial fractions `1/(x − p_j)` with poles fixed per axis (taken from AAA on a few
   fibres). That works only if the fibre test finds the poles stationary (spread ≪ δ). Decisive number: on held-out fibres,
   the sup error of the shared-pole basis against Chebyshev at the same number of functions; drop the idea unless it is 10
   times smaller on at least one class. Risk: poles near the interval make the design ill-conditioned; real poles inside the
   visited range must be excluded.
2. **Splitting where coefficients decay slowly.** Mechanism: bisecting an interval doubles `δ/h`, so halves the degrees per
   decade and doubles the cells: no gain in total for a singularity that every cell sees, a large gain when it is localised
   (cells away from it get a large Bernstein parameter). So the estimator must give a *location*, not only a radius: split at
   the real part of the nearest pole (or at the kink), on the axis whose predicted degree `log(C/tol)/log ρ_j` is largest, and
   only when the far cell's predicted degree falls by at least half. In `d` coordinates a split on one axis doubles the cost of
   every other axis, so the criterion must compare predicted total sizes (I did not check how ChebTree's present split rule does this). Decisive
   number: total solver queries of ChebTree with the pole-located split against its current rule, at the same tolerance, on the
   five problems above.
3. **Coordinate maps clustering points at a front.** Mechanism (Tee and Trefethen 2006; Hale and Trefethen 2008): a conformal
   map that sends a strip around the pole pair to a large ellipse turns `O(h/δ)` into `O(log(h/δ))` degrees. Same condition
   as item 1 (a stationary pole along the axis), same test, same decisive number. If the fronts move, neither 1 nor 3 applies
   and the only global remedy is a nonlinear coordinate (the front position as a chart coordinate); otherwise local fits
   (ART, AFINNI) are the right scheme, and the decision rule should say so instead of a Chebyshev scheme being forced.
4. **Building on the target set instead of the bounding box.** This is the lever with the largest expected effect and it
   connects (a) to factor 2. A Chebyshev scheme queries the solver at designed points, so its sampling factor is logarithmic
   by construction; its cost is that the box is mostly never visited (T7), and that the step there can be harder or
   unsolvable. The generic recipe has the opposite profile. Combine them: **ChebTT (or ChebTree) on the box of the chart
   coordinates, querying `π(Φ(D(u)))` through the validated decoder.** Mechanism: `k` coordinates instead of `d`, designed
   points (Λ about `(2/π log n + 1)^k` instead of 10²–10⁵ at first-step states), and no total-degree cap (a rank-`r` train
   with `n` points per axis has about `k n r²` numbers against `C(k + q, q)`). Preconditions, each checkable without a new
   theory: a validated decoder (34 of the 37 R4 records say "validated", several of them with `k` equal to the number of free
   components, where the decoder is trivial and nothing is reduced), and decoded box corners that the solver can step
   (the KIDA failure, finding 16). So restrict the box: cells of a ChebTree that contain no visited state are not built, and
   inputs falling in them are refused. Decisive test, cheap: on the problems with a validated decoder and `k ≤ 6`, the share of
   the chart box's Chebyshev grid points (degree 8 per axis, a sample of 2,000) whose decoded state has a decoder residual
   within the training range and steps without failure. Where that share is high, run the reduced ChebTT and compare solver
   queries and test rollouts with R4. Demonstration 8.5 did this on HIRES; the panel is what is missing.
5. **Refusal and effective speed-up for Chebyshev schemes.** For designed samples the Lebesgue gate reduces to membership of
   the box (or of a built cell): `c_g` is `O(d)`, and there is no blind spot across fibres when the box is in the full state.
   So a box scheme's effective speed-up equals its nominal one times `(1 − ρ)`-terms with `ρ` the share of inputs outside the
   box: small on the visited set, and to be measured under mixing (a convex mix in linear abundances does not stay in a
   log-coordinate box in general). Test with no new training: from the benchmark's records, compute for every method and
   problem `c_Φ / (c_g + (1 − ρ) c_Ψ + ρ c_Φ)` with the recorded `refused_percent`, beside the generic recipe's. The approximation
   factor needs its own indicator there (the coefficient tail of the cell, already a ChebTree quantity).
6. **A decision rule across schemes from the descriptors.** Inputs, all computable before training a surrogate: the validated
   chart dimension `k` and whether a decoder validates; the predicted degrees per decade per chart axis and whether the poles
   are stationary (the fibre test); `A_n` and the kind (contracting, neutral, chaotic); the share of use inputs off the chart
   (decoder residual under the deployment's mixing); the solve cost. A first rule to pre-register:
   - `k ≤ 3`, predicted degree ≤ 12: total-degree fit from trajectories plus designed starts (the generic recipe);
   - `k ≤ 8`, decoder validated, Bernstein parameter ≥ 2: reduced ChebTT through the decoder;
   - a slow axis with stationary poles: the same with a map or rational basis on that axis; with moving fronts: ART or AFINNI;
   - no reduction (`k > 10`: Lorenz-96 from `d = 10`, POLLU, Nelson): local fits with solver queries, or no surrogate;
   - chaotic with `A_n · tol_step > tol` over the horizon: no surrogate at that horizon, whatever the scheme.

   Decisive test at no cost in runs: predict a scheme per problem from `p1_descriptors.json`, then compare with the benchmark
   store's best method per problem at tolerance 0.01 (a 37-row confusion matrix), fitting any threshold of the rule with one
   problem class left out and scoring on it. A rule that does not beat "always ART" on held-out classes is not a rule.

### (b) Data diversity where the sampling factor is the limit

After finding 3 this is the main lever for the ladders, not a side issue: the worst held-out state is a first-step state in
every row examined.

1. **Budgets from the number of terms.** The dependence on `C(k + q, q)` with the *chart* dimension is right; the linear
   scaling is not, for random draws. Stability of least squares from random samples needs `N ≳ K(p) log p` with
   `K(p) = sup Σ φ_j²` (Cohen, Davenport, Leviatan, already TH-40 in the memo): `K = p` only for the optimal (Christoffel)
   sampling measure, and up to `p²` for uniform-type draws. The records agree: fisher_kpp_d10 with about 1,660 scattered
   inputs has a worst held-out bound of 24 at `p = 210` (N/p = 8) and 413 at `p = 462` (N/p = 3.6). So: `N ≈ c p log p` with
   designed points, `p^1.6`–`p²` without. Test with **no solves**: for each problem, the maximum bound over a held-out pool of
   2,000 fresh starts as a function of `N` (random draws) at fixed `p`; report the `N` at which it falls below 10, against
   `p log p` and `p²`. Remember the floor: no design brings `sup b` below `sqrt(p)`, so the gate of finding 1 must be fixed
   first or no amount of data makes a problem "served".
2. **Starts by maximal Λ from a candidate pool.** This is greedy G-optimal design (equivalently pivoted QR of the pool's design
   matrix, approximate Fekete points): pick the candidate with the largest `φᵀ(AᵀA)⁻¹φ`, update by a rank-one formula, repeat.
   No solve ranks the candidates, as proposed. It attacks `sup_t Λ` on the pool directly, and the pool is the deployment's own
   law, so the points are physical. Two cautions: it selects the edge of the pool, where the approximation factor is largest
   (watch the in-sample maximum as well as the bound), and a pool of *starts* covers step 0 only (see item 4).
   Decisive test at **zero new solves**: the 1,600 extras of R5 are already labelled. Select 400 of them greedily and compare
   with the first 400 (R5's M = 400) and with all 1,600, at each degree, on all twelve R5 problems. Pre-register: greedy-400
   matches or beats random-1,600 in the held-out maximum at first-step states at the same degree, and admits one degree more
   at the same bound.
3. **Perturbed or mixed training states for split-simulation inputs.** Two different mechanisms, by whether the chart is
   full. *Full chart* (chemakzo: 91% refused under mixing, accepted errors 2e-5): mixed states are simply outside the samples'
   cover, and adding them to the fit is the direct fix; expect the refusal to fall with the number added. *Reduced chart*
   (HIRES): the mixed state differs from every training state in a coordinate the chart discards; adding such pairs to a fit
   in the same chart gives inconsistent labels at the same `u` (noise of size `L_⊥ r`), raising `e_v` and the refusal. There
   the data must come with coordinates: the chart's principal directions refitted on the deployment's law (k rises), or the
   normal correction `B(u)(z − D(u))` of Appendix A.3 fitted from the perturbed pairs. Test, on every problem served after the
   fix of finding 1: 400 one-step mixed pairs (w = 0.1) added (i) to the fit only, (ii) to the fit and the chart with `k`
   re-chosen; chained-mixing error and solver share. Pre-register: (i) cuts chemakzo's refusal by more than half and does not
   help HIRES; (ii) raises HIRES's `k` by one or two and brings its chained-mixing error within the tolerance.
4. **Varied sampling times and trajectory lengths.** Mechanism: a trajectory gives one scattered state and then states the
   dynamics pull together, so the transient layer (steps 0–2) is sampled once per trajectory per step index, and that is where
   every held-out maximum sits (check B: step 0 on fisher_kpp_d10, steps 1–2 on fisher_kpp_d3 and allen_cahn_d10). Inputs at
   random *offsets* into the transient, `Φ_{θ dt}(x_0)` with `θ` uniform on `[0, 3)`, each then stepped once by the fixed `dt`,
   fill the layer between the step indices; fresh starts fill step 0 only. Longer trajectories add states where the bound is
   already near its minimum, so they help only chaotic problems, whose attractor a longer orbit covers. Test: 400 offset
   inputs against 400 fresh starts (same solve count), held-out maximum by step index. Pre-register from the step index of the
   current worst state: offsets win on fisher_kpp_d3 and allen_cahn_d10, fresh starts on fisher_kpp_d10, length on Lorenz-96
   only.

### The residual term in the refusal rule

Rule: with the decoder `D` of the degree the k rule validated, `r(t) = max_i |c(t)_i − D(π t)_i|` over the free components;
refuse when `(1 + b(t)) ê + L_⊥ r(t) > tol / A`. It follows from the decomposition in finding 6. `L_⊥` is the step's
sensitivity across the chart's fibres: small for fast directions (the step damps them), of order one for slow ones (a change
of composition that mixing produces), 1.5 on HIRES. Estimate it from about 100 perturbed one-step pairs, or use `r ≤ r_max`
of the training states, which needs no solve and already brought HIRES's accepted worst from 1.2e-2 to 2.2e-3 for 5–10 more
points of refusal (check C). Cost: one decoder evaluation, `O(p_D · free)`.

Decisive test, generic: on every problem with `k < free` (the Allen–Cahn, Fisher–KPP, Burgers and Brusselator ladders, HIRES,
primordial, Nelson), one-step and chained mixing at w = 0.01 and 0.1, with and without the term. Pre-register: (i) among
accepted inputs the rank correlation of the error with the residual exceeds that with the Lebesgue bound wherever accepted
errors exceed tol/10; (ii) with the term, no accepted mixed state exceeds the tolerance on any problem; (iii) on problems with
`k = free` the term is identically zero and changes nothing; (iv) Burgers shows no blind spot (R5: 5.8e-5 under mixing), so the
term leaves its refusal within two points. A failure of (i) on a second problem would mean HIRES's 0.99 is that problem's
geometry, not a mechanism.

---

## Next steps, ranked, each across every problem class

1. **Fix the gate and the selection, then rerun R4 and R5 with new committed predictions** (findings 1, 2, 7, 11): the
   calibrated score, `A_n` in the budget, "served" defined with a solver-share threshold, the best degree reported, a lift
   threshold consistent with the budget, caps from the stability condition. Minutes of compute. Everything downstream depends on
   it, because the present sorting of the problems into served, data-limited and approximation-limited is produced by the
   threshold.
2. **Add the decoder-residual term and run the mixing test on every problem with `k < free`** (finding 9): it is the only
   failure found where the recipe returns a wrong answer rather than a slow one.
3. **Write the per-factor diagnosis as descriptors and as section P** (findings 3, 4, 5): per problem and degree, the in-sample
   maximum, the held-out median and maximum, the bound and step index at the maximum, the lift floor, `A_n`; classify each
   problem by its limiting factor from these; replace section 10's HIRES recipe by the generic one.
4. **Designed data at zero solves** (counsel (b) 2, then 1 and 4): greedy selection from R5's labelled extras; the
   `N` against `p log p` curve; offsets into the transient.
5. **The fibre test** (counsel (a)), which decides between maps or rational bases, splits, and local fits for the
   slow-approximation classes; then **reduced ChebTT through the decoder on the restricted chart box** wherever the decoder
   validates.
6. **The decision rule, scored against the benchmark store with a class left out** (counsel (a) 6), and effective speed-ups for
   every method from existing records with a properly repeated timing and the cost-weighted refusal rate (findings 6, 10).
7. **KIDA** (finding 16), with a k rule that never steps a state far off the data: the target problem has no result yet.

---

## Appendix: the checks

Four scripts, run as stated at the top. They read `data/records/r4`, `data/records/r5` and `data/r5_extra_*.npz`, write
nothing, and use `generic.build` unchanged with the recorded `k` and, per problem, the degree and ridge with the smallest
recorded `one_step_heldout`.

### Check A: the bound, the true Lebesgue function and ungated one-step errors (`chk_a.py`)

```python
import json, sys, numpy as np, time
from generic import TOL, build, design, pairs, split, worst_scored
from src.benchmark.step_maps import STEP_MAPS, data
def rec(name, m):
    f = f"data/records/r4/{name}.json" if m == 0 else f"data/records/r5/{name}_m{m}.json"
    return json.load(open(f))
def extra(name, m):
    if m == 0: return None
    z = np.load(f"data/r5_extra_{name}.npz"); return z["Z0"][:m], z["Z1"][:m]
print("name M k q p sqrt(p) e_v lam_rule | bound min/med/max | trueLam min/med/max (300 inputs) | ungated: worst, %>tol | rule refused% | accept if trueLam<=tol/e_v-1: refused%, worst | calibrated on held-out (bound thr): refused%, worst")
for name, m in [("lorenz96_d5",0),("lorenz96_d10",1600),("allen_cahn_d10",1600),("burgers_d10",0),("pollu",0),("pollu",400),("bruss_d4",0),("fisher_kpp_d2",0),("allen_cahn_d3",0),("hires",0),("lorenz96_d4",0)]:
    t0=time.time()
    r = rec(name, m); P = STEP_MAPS[name]; train, test = data(name); ex = extra(name, m)
    c = min(r["selection"], key=lambda c: c["one_step_heldout"]); q, ridge, e_v = c["degree"], c["ridge"], c["one_step_heldout"]
    sur, _ = build(P, train, r["k"], q, ridge, float(e_v), ex)
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    U = sur.T.chart(Xe, sur.k)
    with np.errstate(all="ignore"):
        b = sur.G.lebesgue_bound(U)
        err = np.max(np.where(P.scored(Ye), np.abs(Xe + sur.predict(Xe) - Ye), 0.0), axis=1)
    # true Lebesgue function on a subsample (ridge 0 only)
    X, Y = pairs(train)
    if ex is not None: X = np.concatenate([X, ex[0]])
    A = design(sur.T.chart(X, sur.k), q, sur.G.lo, sur.G.hi, sur.G.idx)
    rng = np.random.default_rng(0); sub = rng.choice(len(Xe), min(300, len(Xe)), replace=False)
    lam = np.full(len(sub), np.nan)
    if ridge == 0:
        Q, R = np.linalg.qr(A)
        Phi = design(U[sub], q, sur.G.lo, sur.G.hi, sur.G.idx)
        lam = np.abs(Q @ np.linalg.solve(R.T, Phi.T)).sum(0)
    p = len(sur.G.idx)
    # calibrated threshold: largest bound L such that every held-out point (3 splits) with bound <= L errs <= tol/2
    bs, es = [], []
    for s in (0,1,2):
        fit, val = split(train, s)
        sv, _ = build(P, fit, r["k"], q, ridge, float(e_v), ex)
        Xv, Yv = pairs(val)
        with np.errstate(all="ignore"):
            bs.append(sv.G.lebesgue_bound(sv.T.chart(Xv, sv.k)))
            es.append(np.max(np.where(P.scored(Yv), np.abs(Xv + sv.predict(Xv) - Yv), 0.0), axis=1))
    bs, es = np.concatenate(bs), np.concatenate(es)
    o = np.argsort(bs); bad = np.nonzero(es[o] > TOL/2)[0]
    Lcal = bs[o][bad[0]-1] if len(bad) and bad[0] > 0 else (np.inf if not len(bad) else -1)
    acc = b <= Lcal
    accl = lam <= TOL / e_v - 1
    f = lambda v: "/".join(f"{x:.3g}" for x in (np.nanmin(v), np.nanmedian(v), np.nanmax(v)))
    print(f"{name} {m} k={r['k']} q={q} p={p} sqrtp={p**.5:.1f} e_v={e_v:.2e} lam_rule={TOL/(2*e_v)-1:.2f} | {f(b)} | {f(lam)} | {err.max():.2e}, {100*np.mean(err>TOL):.2f}% | {100*np.mean(b>TOL/(2*e_v)-1):.1f}% | {100*(1-accl.mean()):.1f}%, {err[sub][accl].max() if accl.any() else 0:.2e} | Lcal={Lcal:.3g}: {100*(1-acc.mean()):.1f}%, {err[acc].max() if acc.any() else 0:.2e}  [{time.time()-t0:.0f}s]", flush=True)
```

### Check B: ungated rollouts; in-sample against held-out error by degree (`chk_b.py`)

```python
import json, numpy as np, time
from generic import TOL, build, design, pairs, split, multi_indices
from src.benchmark.step_maps import STEP_MAPS, data, rollout
def rec(name, m):
    return json.load(open(f"data/records/r4/{name}.json" if m == 0 else f"data/records/r5/{name}_m{m}.json"))
def extra(name, m):
    if m == 0: return None
    z = np.load(f"data/r5_extra_{name}.npz"); return z["Z0"][:m], z["Z1"][:m]
print("== ungated rollouts over all test trajectories (no solves), best-e_v degree")
for name, m in [("lorenz96_d5",0),("lorenz96_d10",1600),("pollu",0),("fisher_kpp_d2",0),("bruss_d4",0),("allen_cahn_d10",1600),("burgers_d10",0),("allen_cahn_d3",0),("hires",0)]:
    r = rec(name, m); P = STEP_MAPS[name]; train, test = data(name)
    c = min(r["selection"], key=lambda c: c["one_step_heldout"])
    sur, _ = build(P, train, r["k"], c["degree"], c["ridge"], float(c["one_step_heldout"]), extra(name, m))
    sur.lam_star = np.inf
    ro = rollout(P, sur, test)
    print(f"{name} M={m} q={c['degree']}: ungated rollout worst {ro['rollout_worst']:.2e}, solver share {np.mean(ro['rollout_solved_by_step']):.3f}; test shape {test.shape}", flush=True)
print("== in-sample against held-out by degree, split 0 (scored max)")
for name, m in [("fisher_kpp_d10",1600),("fisher_kpp_d3",1600),("allen_cahn_d10",1600)]:
    r = rec(name, m); P = STEP_MAPS[name]; train, test = data(name); ex = extra(name, m)
    fit, val = split(train, 0); Xf, Yf = pairs(fit); Xv, Yv = pairs(val)
    nsteps = train.shape[1]-1
    for q in sorted({c["degree"] for c in r["selection"]}):
        sur, e_v = build(P, fit, r["k"], q, 0.0, (Xv, Yv), ex)
        with np.errstate(all="ignore"):
            ein = np.max(np.where(P.scored(Yf), np.abs(Xf + sur.predict(Xf) - Yf), 0.0), axis=1)
            ev = np.max(np.where(P.scored(Yv), np.abs(Xv + sur.predict(Xv) - Yv), 0.0), axis=1)
            b = sur.G.lebesgue_bound(sur.T.chart(Xv, sur.k))
            ex_in = 0.0
            if ex is not None:
                ex_in = np.max(np.where(P.scored(ex[1]), np.abs(ex[0] + sur.predict(ex[0]) - ex[1]), 0.0))
        w = int(np.argmax(ev)); step = w // len(val)   # pairs() orders by step then trajectory
        print(f"{name} M={m} k={r['k']} q={q} p={len(sur.G.idx)}: in-sample traj max {ein.max():.2e} (extras {ex_in:.2e}), held-out max {ev.max():.2e} median {np.median(ev):.1e} 99% {np.quantile(ev,.99):.1e}; worst at step {step}/{nsteps}, bound there {b[w]:.3g}, median bound {np.median(b):.3g}; share of held-out > tol {100*np.mean(ev>TOL):.2f}%", flush=True)
```

### Check C: k curves; residual gates on mixed HIRES states (`chk_c.py`)

```python
import json, numpy as np
from generic import TOL, build, design, pairs, multi_indices
from p2_tradeoff import mixed
from src.benchmark.step_maps import STEP_MAPS, data, visited
print("== k rule: lift error at the chosen k")
for n in ["allen_cahn_d10","allen_cahn_d20","bruss_d10","burgers_d10","fisher_kpp_d10","lorenz96_d10","pollu","hires","chemakzo","primordial","nelson","allen_cahn_d5"]:
    r = json.load(open(f"data/records/r4/{n}.json")); c = r["k_curve"][-1]
    print(f"  {n}: k={r['k']} ({r['reduction']}), last curve entry k={c['k']} lift {c['lift_error']:.2e} decoder degree {c['decoder_degree']}, free={r['d']-r['held']}")
name = "hires"; r = json.load(open(f"data/records/r4/{name}.json")); P = STEP_MAPS[name]; train, test = data(name)
c = [c for c in r["selection"] if c["degree"] == r["degree"] and c["ridge"] == r["ridge"]][0]
sur, _ = build(P, train, r["k"], r["degree"], r["ridge"], float(c["one_step_heldout"]))
T, k = sur.T, sur.k
Xtr = visited(train); Utr = T.chart(Xtr, k)
def resid_lin(Z):
    C = T.clip(Z)[:, :T.free] - T.mu
    return np.linalg.norm(C - C @ T.V[:, :k] @ T.V[:, :k].T, axis=1)
lo, hi = Utr.min(0), Utr.max(0)
dec = {}
for deg in (2, 4):
    idx = multi_indices(k, deg)
    dec[deg] = (idx, np.linalg.lstsq(design(Utr, deg, lo, hi, idx), T.clip(Xtr)[:, :T.free], rcond=None)[0])
def resid_dec(Z, deg):
    idx, coef = dec[deg]
    return np.max(np.abs(T.clip(Z)[:, :T.free] - design(T.chart(Z, k), deg, lo, hi, idx) @ coef), axis=1)
from scipy.stats import spearmanr
for w in (0.01, 0.1):
    Z = mixed(P, test, w, np.random.default_rng(5)); Y = P.step(Z)
    with np.errstate(all="ignore"):
        b = sur.G.lebesgue_bound(T.chart(Z, k)); err = np.max(np.where(P.scored(Y), np.abs(Z + sur.predict(Z) - Y), 0.0), axis=1)
    acc = np.isfinite(b) & (b <= sur.lam_star)
    print(f"== hires mixed w={w}: n={len(Z)}, Lebesgue rule (Lambda*={sur.lam_star:.2f}) refuses {100*(1-acc.mean()):.1f}%, accepted worst {err[acc].max():.2e}, accepted > tol {100*np.mean(err[acc]>TOL):.2f}%, > tol/10 {100*np.mean(err[acc]>TOL/10):.1f}%")
    print(f"   Spearman(error, Lebesgue bound) among accepted: {spearmanr(err[acc], b[acc])[0]:.2f}")
    for label, f in (("linear normal residual", resid_lin), ("quadratic-decoder residual", lambda Z: resid_dec(Z, 2)), ("quartic-decoder residual", lambda Z: resid_dec(Z, 4))):
        rt, rz = f(Xtr), f(Z)
        for qn, thr in (("train max", rt.max()), ("train 99%", np.quantile(rt, .99))):
            a2 = acc & (rz <= thr)
            print(f"   {label}: Spearman(error, residual) among accepted {spearmanr(err[acc], rz[acc])[0]:.2f}; adding 'residual <= {qn}' ({thr:.2e}): refuses {100*(1-a2.mean()):.1f}%, accepted worst {err[a2].max() if a2.any() else 0:.2e}")
        # threshold that would be needed, and a linear model err <= e_v + L r
        L = np.max((err[acc]) / np.maximum(rz[acc], 1e-12)); print(f"      max err/residual among accepted {L:.3g}; median {np.median(err[acc]/np.maximum(rz[acc],1e-12)):.3g}")
```

### Check D: rollouts under a held-out-calibrated threshold (`chk_d.py`)

```python
import json, numpy as np, time
from generic import TOL, build, pairs, split
from src.benchmark.step_maps import STEP_MAPS, data, rollout
def rec(name, m):
    return json.load(open(f"data/records/r4/{name}.json" if m == 0 else f"data/records/r5/{name}_m{m}.json"))
def extra(name, m):
    if m == 0: return None
    z = np.load(f"data/r5_extra_{name}.npz"); return z["Z0"][:m], z["Z1"][:m]
print("held-out-calibrated gate: accept bound <= L, L = largest held-out bound below the first held-out error > tol/2 (all three splits), capped at the 99% quantile of held-out bounds")
for name, m in [("pollu",0),("lorenz96_d5",0),("lorenz96_d10",1600),("fisher_kpp_d2",0),("bruss_d4",0),("allen_cahn_d10",1600),("burgers_d10",0),("allen_cahn_d3",0),("hires",0)]:
    t0 = time.time()
    r = rec(name, m); P = STEP_MAPS[name]; train, test = data(name); ex = extra(name, m)
    c = min(r["selection"], key=lambda c: c["one_step_heldout"]); q, ridge, e_v = c["degree"], c["ridge"], c["one_step_heldout"]
    bs, es = [], []
    for s in (0, 1, 2):
        fit, val = split(train, s); sv, _ = build(P, fit, r["k"], q, ridge, float(e_v), ex); Xv, Yv = pairs(val)
        with np.errstate(all="ignore"):
            bs.append(sv.G.lebesgue_bound(sv.T.chart(Xv, sv.k))); es.append(np.max(np.where(P.scored(Yv), np.abs(Xv + sv.predict(Xv) - Yv), 0.0), axis=1))
    bs, es = np.concatenate(bs), np.concatenate(es); o = np.argsort(bs); bad = np.nonzero(es[o] > TOL / 2)[0]
    L = bs[o][bad[0] - 1] if len(bad) and bad[0] > 0 else (np.inf if not len(bad) else -1.0)
    L = min(L, np.quantile(bs, 0.99))
    sur, _ = build(P, train, r["k"], q, ridge, float(e_v), ex); sur.lam_star = L
    ro = rollout(P, sur, test)
    print(f"{name} M={m} q={q}: L={L:.3g} (held-out accepted share {100*np.mean(bs<=L):.1f}%); test rollout worst {ro['rollout_worst']:.2e}, solver share {np.mean(ro['rollout_solved_by_step']):.3f}  [{time.time()-t0:.0f}s]", flush=True)
```
