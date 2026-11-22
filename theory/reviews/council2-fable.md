# Council review, round 2 (Fable): what the second round's records say, and what to test next

Reviewer: Claude Fable 5.2, one of three independent council members. Material: `theory/SYNTHESIS.md` at commit `39422d9`
(sections 3.9-3.16, 9.6, 9.7, 10), the records under `theory/panel/data/`, the scripts that wrote them, `theory/MEMO.md` (E, T5, 8),
`src/benchmark/step_maps.py`. Every check of mine is read-only and post hoc; the five scripts are in `/tmp/council2-fable/` (not
committed) and are described in the appendix with their output. Nothing here was pre-registered, so each check is a reading to
be re-run with a committed prediction, not a result.

## Summary of the recommendation

Three things in the records change the plan more than any new model family would. (1) **The gate's budget ignores accumulation.**
Every gate accepts a one-step defect up to half the tolerance, while the benchmark's own cached factor `A_n` (`dynamics.json`) is
about the number of steps on every problem (40 on the lattices, 140 on CHEMAKZO, 6,669 on HIRES). A budget of `tol / (2 A_n)` per
step predicts section 3.16's failure from files written before the run; "selection must see coupled chains" treats the symptom.
(2) **The off-chart defect is first order and removable.** On HIRES the accepted defect on mixed inputs is a linear function of the
offset from the chart, with one sign (coherence 0.84); a constant linear term in the off-chart coordinates, fitted on 12
trajectories' mixed states, cuts its median 37-62x. That is the tube's theory in one line: Proposition 5.1 applied to a design
that contains normal coordinates, on samples that leave the bundle the way the coupling does. (3) **Three served fits are
numerically singular**, their coefficients are not the minimum-norm solution the gate's bound describes, the 281-1,642 us gate is
the cost of an SVD solve repeated per call (a precomputed factor gives the same decisions at 15 us), and the first steps lie outside
the span the data determine. So: fix the gate's arithmetic, its budget and its span term first (hours); then run one tube
experiment with four arms on six problems; then replace the total-degree basis where the chart has 8 or more coordinates by an
adaptive index set, with a trajectory-fitted tensor train as the second arm; keep KIDA to three no-fit measurements until the MFC
step and horizon are fixed. Stop designing data for first steps (they are the panel's i.i.d. starts: Brusselator d20's increments
have rank 20 with them and 2 without), stop reporting chained mixing as the coupled test, and do not rerun the panel for the
spurious laws before measuring their effect (3.5e-8 on bruss_d20).

---

## 1. The plan from here; what this round got wrong; which readings in 9.7 do not follow

What was done well: every run has predictions committed before it, the failed ones are reported as failed (V3, V5, E1, S1's second
clause, K3, W1), and the records hold enough to answer most of what follows without a rerun. Of my round-1 list, C1 (ChebTT under
mixing), C2 (a gate for the maximum, a propagation margin) and C3 (the chart range up to the free dimension) were run; C4 (a basis
that scales in the chart dimension) was not, and section 3.10 shows it is what POLLU, Nelson and lorenz96_d20 now wait for
(section 2 below).

Findings, most important first.

### 1.1 (critical) The per-step budget is `tol / 2` everywhere; the accumulation factor says it must be 40 to 6,700 times smaller

`generic2.py:109` refuses when `(1 + b) e_hat > TOL / 2`; `p7_online.py` and `p5_kida_screen.py` calibrate thresholds so that
accepted calibration states err within half the tolerance. Proposition 2.1 and Theorem 8 bound a chain by the one-step defect times
`A_n`. The benchmark computes `A_n` from the step map's Jacobians for every problem (`step_maps.py:481-543`), and the cached values
are (check 6):

| problem | steps | `A_n` (max) | kind |
|---|---|---|---|
| burgers_d20, lorenz96_d4, fisher_kpp_d3, fisher_kpp_d5 | 40 | 31-40 | neutral or contracting |
| robertson, bruss_d20 | 40 | 62, 67 | neutral, stiff |
| nelson (24 steps), primordial (30) | | 82, 99 | neutral / chaotic, stiff |
| allen_cahn_d5 | 40 | 123 | neutral |
| chemakzo | 85 | 140 | neutral, mildly stiff |
| pollu | 59 | 234 | neutral, stiff |
| hires | 170 | 6,669 | chaotic, stiff (growth 61) |

So on no problem do errors decay along a chain: `A_n` is about `n` or larger. With accepted defects of 2e-3 on mixed inputs
(section 3.16; my check 1 reproduces 2.2e-3) HIRES's bound is 13 and the observed 0.16 needs no further explanation. The plain
chains pass only because the plain defect is 1e-6 to 1e-8 (`records/r4d/hires.json`: chains 2.1e-6 at degree 12).

Consequences: (a) reading 9.7.2 is right about the fact and incomplete about the remedy: measuring the margin on mixed chains
certifies one coupling script; the budget `margin * tol / A` certifies the step whatever the script, and both should be recorded.
(b) The worst-case `A_n` is pessimistic for defects of mixed sign: measure the coherence `c = |mean e| / mean |e|` of accepted
defects (0.84 on HIRES at w = 0.01, 0.16 at w = 0.1; check 1) and budget `e (c n + sqrt(n)) <= tol / 2`. (c) Mixing with a
reference partner pulls the error back by `w` per step, so its factor is about `min(n, 1 / w)`: 10 at w = 0.1. That, not coupling,
is what the next finding is about.

### 1.2 (critical) HIRES's mixed-input defect is linear in the off-chart offset, and a linear normal term removes it in the median

Check 1 (`a_hires.py`): R4d's HIRES surrogate rebuilt (e_hat 1.882e-4 against the record's 1.884e-4); test states mixed once at
w = 0.01 and 0.1; over the accepted ones the defect's median is 2.55e-4 and 6.8e-4, its maximum 2.2e-3. Regressing the defect on the
five off-chart principal coordinates (a constant matrix, fitted on mixed states of 12 *training* trajectories, scored on the test
ones) leaves a median of 6.95e-6 and 1.10e-5 (37x and 62x smaller; R2 0.64 and 0.76). The maximum does not move (2.24e-3): a few
states carry a defect that is not first order, and I did not identify them. The sign coherence is 0.84 on every component at
w = 0.01.

Reading: the chart projects orthogonally (principal directions of the visited states, `generic.py:81`), the step map collapses
states along its own fast fibres, and the two differ by an angle; Proposition 8.2's middle term `L sup |pi_hat - chi o pi|` is
exactly this and is first order in the offset. Section 3.16's sentence "small defects add up along directions the dynamics do not
contract" is true; what it omits is that the defects are systematic, predictable from the input, and cheap to remove. This is the
basis of section 5 below.

### 1.3 (critical) Three served fits are singular; the coefficients, the bound and the gate's cost all follow from that

- `records/r4c/chemakzo.json`, `fisher_kpp_d5.json` and `records/r4d/fisher_kpp_d10.json`: the chosen configurations have ranks
  1,285 of 2,002, 603 of 1,287 and 600 of 924 terms (field `rank` per split; check 3). These are the only three records whose gate
  costs far more than the prediction (1,642 / 10, 525 / 9, 281 / 5 us); every full-rank fit's gate is 1 to 7 times its prediction.
- **The cost is a branch, not a complexity.** `generic2.py:51-52` calls `np.linalg.lstsq(G.R.T, Phi.T)` when the rank is deficient:
  an SVD of a 1,287 x 1,287 matrix per call. Check 3 on fisher_kpp_d5: 488 us per state; the same quantity from a factor computed
  once (`V_r S_r^-1`, numpy's default cutoff) costs 14.7 us, agrees to 2.9e-3 relative, and changes 0 of 1,600 decisions. A
  triangular solve on a full-rank factor costs 11 us at the same size. So "the gate's cost sets the speed" (3.10), bottleneck 8 of
  section 9.1 and question 7's example are one missing precomputation.
- **The deployed coefficients are not the estimator the bound is about.** `generic.py:111` back-substitutes through `R` whenever no
  diagonal entry is exactly zero; on fisher_kpp_d5 the factor's singular values run from 5.4e2 to 2.3e-15, and 1,112 of 1,287 are
  below 1e-3 of the largest. The pseudo-inverse bound describes the minimum-norm fit; the deployed map has arbitrary large
  coefficients along the undetermined directions.
- **The pseudo-inverse bound is blind where it matters.** It drops the part of `phi(t)` outside the retained row space, which is
  where the fit is undetermined (`Lambda` is infinite there). On fisher_kpp_d5 the test states' share of `|phi|` outside the span
  with singular values above 1e-8 of the largest is 5.1% at step 0, 0.15% at step 1, 1e-4 at step 2 and zero from step 5 (check 3):
  the first steps are out of span, later steps are in it.
- **Hence two readings.** The "ungated first-step worst" of 4.7-5.0 on every Fisher-KPP row of section 3.13, and the held-out worst
  of 4.97 at degrees 6-9 in `records/r4c/fisher_kpp_d5.json`, are the output clip (`zlo = min - 2`, `zhi = max + 0.5`,
  `generic.py:83,148`) of an undetermined fit; they are not error magnitudes, and "no arm moves the first step" (9.7.4) partly says
  that a clipped value cannot move. And CHEMAKZO's effective speed-up of 0.7 (9.7.3) is the SVD branch: with the gate at twice the
  prediction its late-phase value is 35-41 at the degree the rule chose (check 5; an estimate, to be measured).
- The two ridges offered are none and `1e-6 s_max^2`; the second serves nothing on these problems (every `ridge 1e-06` row of the
  three records is `inf` or unserved, with e_hat near 1e-3). There is no regularisation between a singular fit and a coarse one.

Fix (section 4 gives the theory): a truncated-SVD fit at a stated relative cutoff, the gate's two terms from the same factor
(`Lambda_r(t)` and the out-of-span share `rho(t)`), computed once and shared with the prediction.

### 1.4 (major) Reading 9.7.1 does not follow: the column equals the plain rollout; the "2.7-13x" is the mixing test's damping

For rollout-tuned ChebTT, `records/p4/*.json` (plain rollout over 20 trajectories, chained mixing) against `p6_column.json`
(check 2):

| problem | plain rollout | chained mixing 0.01 / 0.1 | column 0.005 / 0.05 | own-input defect |
|---|---|---|---|---|
| allen_cahn_d3 | 3.4e-3 | 3.2e-3 / 2.3e-4 | 3.3e-3 / 3.1e-3 | 4.7e-5 |
| allen_cahn_d2 | 6.7e-3 | 6.8e-3 / 9.0e-4 | 8.5e-3 / 3.4e-3 | 1.1e-4 |
| robertson | 4.3e-3 | 3.5e-3 / 1.1e-3 | 4.1e-3 / 4.0e-3 | 1.8e-4 |
| lorenz96_d10 | 7.2e-3 | 6.1e-3 / 1.8e-3 | 6.7e-3 / 6.6e-3 | 6.0e-4 |
| chemakzo | 5.7e-3 | 3.7e-3 / 1.7e-3 | 5.0e-3 / 4.6e-3 | 2.5e-4 |
| fisher_kpp_d5 | 4.4e-3 | 3.1e-3 / 1.4e-3 | 3.4e-3 / 1.6e-3 | 6.9e-4 |

Over the 15 ChebTT rows the column at D = 0.05 is between 0.4 and 2.1 times the *plain* rollout (2.1 on allen_cahn_d10, 2.4e-3
against 5.0e-3; at or below 1.4 on the other fourteen). The column adds nothing measurable to an uncoupled rollout. Chained mixing
at w = 0.1 is the outlier because it replaces a tenth of each cell by the truth every step (1.1 (c)). The correct statements:
(i) for a box scheme the column, the plain rollout and weak mixing are the same test, and strong mixing with a reference is an
easier one; (ii) ChebTT's chain is 3 to 70 times its one-step defect (accumulation), and its margin is thin by construction, since
`chebtt_roll` stops retraining at half the tolerance on the training rollouts (`step_maps.py:288`); (iii) the recipe's column worst
equals its own-input defect within 6% on six of its eight rows (fisher_kpp_d2, d3, d5, burgers_d20, lorenz96_d4, robertson): a
single accepted step, not accumulation. The two schemes fail in different ways and need different margins.

The column is also a mild test: 40 cells, 40 steps, D at most 0.05, cells sorted so neighbours are alike, every input a convex
combination of bundle states. A flow solver at a CFL number of order one exchanges tens of percent per step between cells that can
differ across a front. The quantity that sets the difficulty is the offset from the bundle, about `D kappa |x_i - x_{i+1}|^2`
(`kappa` the bundle's curvature); neither test controls it. Add D = 0.25 and an unsorted column (dissimilar neighbours); the ChebTT
columns take 1-30 s each (`wall_s` in the records).

### 1.5 (major) "Served" and "V2 holds" count cases where the solver does the work, and the speed-up uses the wrong share

- CHEMAKZO is "served" and "holds chained mixing" (V2) with the solver taking 89% and 99% of the mixed steps
  (`records/r4c/chemakzo.json`, `mixing_chained`); HIRES in R4d takes 63% and 78%. A test the solver passes is not evidence about
  the surrogate. "Served" needs the solver share below 50% in the coupled test too.
- The effective speed-up uses the refused share of the *test trajectories' own states* (`generic3.py:193,205`). CHEMAKZO refuses
  0.7% of those and 18.2% of its chain's states (`rollout_solver_share`): chain states leave the calibrated set 26 times as often,
  the same drift section 3.15 found in the local table. Under mixing HIRES's 5.7 becomes about 1.5 (938 / (23 + 0.37 * 18 + 0.63 * 938)).
  Record the speed-up with the share of the regime it is quoted for.
- The refusals sit in the first steps: over steps 10 onward the refused share is 0.000 on bruss_d20, burgers_d20, fisher_kpp_d3,
  0.002-0.007 on fisher_kpp_d5 and d10, and 0.094 on HIRES (check 5, from `refused_by_step`). The medians of V5 are set by the
  first ten of 40 steps.

### 1.6 (major) Reading 9.7.4: the first steps are the panel's i.i.d. starts, and the evidence for the cost law is two problems

- E1 failed (leverage beat random on 2 of 4 lattices and halved the speed on two); the reading leads with the one lattice where it
  won by 24x. The "0.46 against 0.09" comparison is one problem at d = 10 against one at d = 3.
- The rank of the increments at 1e-3 (largest residual of a truncated SVD over the training pairs; check 4) with and without the
  first five steps: bruss_d20 20 -> 2, burgers_d20 12 -> 6, fisher_kpp_d10 10 -> 6, allen_cahn_d10 10 -> 5, pollu 13 -> 9;
  unchanged on nelson (13), hires (7), chemakzo (5), lorenz96_d20 (20). On the lattices the high-dimensional problem *is* the first
  five steps.
- The last sentence of 9.7.4 is the right conclusion and the plan should act on it: the solver takes the first `m` steps as a rule
  (cost `m / n` of a run, zero in the limit), scores are reported from step `m` on as well as from 0, and no more runs go into
  choosing first-step data. This is a statement about where the target's states are, so it belongs with the tube (section 5).

### 1.7 (major) Reading 9.7.6: the KIDA conclusion holds; "coverage, not a hard species" needs two qualifications

`p5b_kida_breakdown.json`: (i) the species are not alike: H2, CO and O are within 0.01 dex on 95-100% of states for every fitted
arm, while C (45-77% above), S+ (57-69%), HCN (50-60%) and N2H+ (36-48%) carry the misses; a per-species gate would serve the first
group now. (ii) Persistence errs 0.167 dex at the median and 1.2 dex at the median of the first step: the tolerance is 6% of one
step's change at this `dt`, with 4,800 pairs over a 4-dimensional driven environment. Whether that is the target's regime depends
on the MFC step: a step 100 times shorter makes the map nearly the identity and moves the difficulty from approximation to
accumulation (1.1) over 100 times as many steps. Fix `dt` and the horizon from an MFC run before any more KIDA fits. The MLP arm is
one scikit-learn configuration trained for 18 s; it shows that 4,800 pairs do not train a network here and nothing more general.

### 1.8 (minor) Readings 9.7.3, 9.7.5, 9.7.7

9.7.3 (selection must see speed): adopt it, but the 0.7 is 1.3's SVD branch. 9.7.5 (calibration is fragile): true, and each
instance has a specific cause (section 6). 9.7.7 (spurious laws): the bug is real; its stated size and reach are not what the
records show (section 8).

### The order I recommend

Section 9 gives the steps with costs. In short: repair (1.3, 1.1, 1.5), then the tube experiment (section 5), then the basis for
charts of 8 or more coordinates (section 2), then the audited local table (section 6), with three KIDA measurements that need no fit
alongside.

---

## 2. Tensor trains made versatile

What limits the recipe in the records is the count of terms of a total-degree basis, `C(k + q, q)`, against the pairs:

- POLLU, k = 13: degree 3 has 560 terms and a held-out median of 1.0e-3 (coarse); degree 4 has 2,380 terms and overflows
  (`records/r4c/pollu.json`, held-out worst 299).
- Nelson, k = 12 plus held inputs: degree 1 errs 0.11 at the median; degree 2 (153 terms) has a held-out worst of 2.9; degree 3
  (969 terms) exceeds two thirds of its 1,152 pairs (60 training trajectories of 24 steps; at 78 ms a solve, that data set cost
  90 s).
- lorenz96_d20, k = 20: degree 3 (1,771 terms) holds its chains at 5.48e-3, 10% above the margin; degree 4 is out of reach.
- primordial, k = 6: degree 3 is coarse, degrees 4 and 5 overflow.

So the question is which format puts fewer coefficients where the map needs them. Ranking, with the theory each keeps:

1. **T-0 (new): an adaptive downward-closed index set in the chart coordinates** (anisotropic degrees, hyperbolic-cross-like
   interactions, grown greedily by held-out gain per added term). It is linear in the data, so Proposition 5.1 and the gate hold
   as they are, the fit is convex, and the gate costs `O(p^2)` with a small `p`. It is the tensor train's competitor at equal
   coefficients and should be run first because it is one function (`multi_indices`) away. *Bound:* Proposition 5.1 with `V` the
   span of the chosen set; the choice of the set on held-out data is a selection among linear models, so the split
   calibration covers it.
2. **T-a: a tensor train fitted to trajectory data in chart coordinates** (alternating least squares). Worth its non-convexity only
   where T-0 needs more than a third of the pairs, that is where interactions are of high order but low rank. *What the gate can
   certify:* with the other cores frozen, core `j` is a linear least-squares problem, but the bound of that one problem is not a
   bound for the model. The honest object is the **tangent design**: the rows `d f(x_i) / d (all cores)`, a matrix with
   `sum_j r_{j-1} (q + 1) r_j` columns minus the gauge. Its leverage `Lambda_T(t)` bounds the error at `t` relative to the best
   approximation *within the tangent space at the fitted point* (a first-order statement), and it is computable. The calibrated
   gate then works as for the chart: `e_hat` from held-out scores `err / (1 + Lambda_T)`. No statement survives about the global
   optimum; say so.
3. **T-e: ranks and degrees by held-out chains.** This is the selection rule for T-a and T-0, not a candidate of its own; use the
   margin of 1.1 and the speed of 9.7.3, and stop a chain as soon as it cannot be served (section 7).
4. **T-f (new): the tube box.** Coordinates `(u, r)`: chart coordinates and normal coordinates, the latter restricted to
   `|r| <= rho` at degree 1 or 2 (two or three nodes). The map is nearly affine in `r` (1.2), so the normal cores have rank about
   the number of fibre-mismatch directions. This is how a cross-built ChebTT moves from the box to the tube; it needs solves at
   designed points, and its precondition is that the bundle fills its chart box (a measurement, section 5.4).
5. **T-b: one train for all outputs.** Section 3: do it through an output basis first; the library's `build_vector` already shares
   the point cache across outputs (`docs/SCHEMES.md:83`), so what is left to share is the cores.
6. **T-c: order or tree from the coupling graph.** In chart coordinates the graph is gone (principal directions mix species), so
   T-c excludes T-a and the tube; it only helps a box scheme in the original variables, which T7 rules out at the target's scale.
   On the lattices the order is already the chain. Do not pursue.
7. **T-d: piecewise trains in leaves.** It targets the relaxation oscillators only. A jump is rare in time, and the generic answer
   to a rare event under a maximum metric is refusal: record the served share on the smooth phases instead.

Decisive first test for 1 and 2 (one run, equal data, `M = 0`, the repaired gate): POLLU, Nelson, primordial, lorenz96_d20, with
bruss_d20 and burgers_d20 as controls. Arms: total degree (the record), T-0, T-a at ranks 2, 4, 8 and degrees 3 to 6. Predictions
to commit: T-0 reaches a held-out median below 1e-4 on POLLU with fewer than 800 terms and serves lorenz96_d20 (its degree-3 chain
is already 5.5e-3); neither arm serves Nelson on its 1,152 pairs (its increments have full rank 13 of 14, check 4, and its
degree-2 held-out median is 3.7e-2: a data limit, so give Nelson a second data arm of 200 trajectories, about five minutes of solves); T-a beats T-0 at equal coefficients on at most one of the four. If T-a wins on
two or more, it becomes the default fit for k >= 8. For vector-valued *functions* the same arms run on the function suite's
problems with 8 or more inputs through `build_vector`'s interface; nothing in T-0 or T-a is specific to step maps.

---

## 3. One model per output

Three structures, three detectors, and what each guarantees.

1. **Outputs that are linear combinations of fewer outputs** (output rank). Detector: the rank at the tolerance of the increment
   matrix (check 4). After the transient it is 9 of 20 on POLLU while the states' rank is 17; 2 on bruss_d20; 6 on burgers_d20;
   13 of 14 on Nelson and 20 of 20 on lorenz96_d20 (nothing to gain). *Exploit with an interpolative decomposition, not an SVD:*
   choose `r` actual output components by pivoted QR and write the others as `C` times them. Then for every state
   `|err_other| <= ||C||_inf |err_selected| + tau`, `tau` the truncation residual, and strong rank-revealing QR keeps the entries of
   `C` bounded by 2. This is a sup-norm statement on real species, which an orthogonal basis does not give (its rows mix all
   outputs and cost a factor `sqrt(r)`). `tau` is measured on data and is exact only where the dependence is a conservation law;
   elsewhere it is calibrated like `e_hat`.
2. **Outputs that depend on the same few input directions** (shared active directions). Detector: the singular values of the
   stacked Jacobians, or of a reduced-rank regression of successors on states. This is the chart, and section 5 argues it should be
   built from the map's sensitivity, not from the states' variance.
3. **Outputs fixed by conservation laws.** Exact; the projection does it.

For ChebTT, a model per output multiplies build time and inference by `d_out` but not solves (shared cache). Whether that is what
stops it at d >= 8 is not in the records: every chemistry record is a 240 s time limit with no split of the time. Measure the
evaluations and seconds per output on CHEMAKZO (5 outputs, 11 s column build) before designing an output core.

**For the target the frame is different.** KIDA needs 13 species at 0.01 dex and carries 578 only because a closed chain needs its
own inputs. The right question is which state coordinates the 13 species' *futures* depend on: the rank at the tolerance of
`d (key species at steps 1..h) / d (state now)` on the tube. If that rank is tens, a closed reduced model exists, and the accuracy
needed on the other 565 species is set by their sensitivity, not by 0.01 dex. If it is hundreds, no shared-output device helps.
This is goal-oriented reduction; it is generic (any vector-valued map with declared quantities of interest), and it is a
measurement, not a fit (section 9, step 6).

---

## 4. Held-out against in-sample error

The diagnosis (the Lebesgue function between sparse first-step samples, at a cost set by the initial law's dimension) is incomplete
in three ways.

1. **Out of span, not between samples.** Where the design is rank-deficient (1.3) a first-step state has a component in directions
   the data do not determine. Proposition 5.1 needs a companion for a truncated fit. Let `A = U S V^T`, keep the `r` singular values
   above a cutoff, `P_r = V_r V_r^T`, and let `c*` be the coefficients of the best approximant `q*` with error `e*`. For the
   minimum-norm fit `p`:

   `|f(t) - p(t)| <= (1 + Lambda_r(t)) e* + ||(I - P_r) phi(t)|| ||c*||`,   `Lambda_r(t) = sqrt(N) ||S_r^-1 V_r^T phi(t)||`

   (insert `q*`; the fit reproduces `P_r c*` up to `Lambda_r e*`; the rest is the part of `q*` the data cannot see). The second
   term is the out-of-span share times the size of the function's coefficients; it is zero on the data and is what the
   pseudo-inverse bound drops. The gate needs both terms, and both come from one product `Phi V_r`.
2. **The estimator.** Neither a singular back-substitution nor a ridge of `1e-6 s_max^2` is a sensible fit (1.3). Truncate at a
   cutoff tied to what the labels support: the solver's relative tolerance (1e-8) times `s_max` is the natural default; select
   among 1e-6, 1e-8, 1e-10 by held-out chains as the ridge is now.
3. **The fit space.** Total degree in `k` coordinates is not identifiable from data on a set of dimension 2 to 6 (check 4): 731 of
   1,287 directions on fisher_kpp_d5. An index set grown by held-out gain (T-0) never adds the unidentifiable terms.

What makes the held-out maximum robust, most robust first, by class of problem:

- **The gate with the span term** (all classes): it turns an unbounded held-out error into a refusal, whatever the data.
- **Truncation at the label noise** (all classes): bounds the fit off the data.
- **A burn-in rule** (stiff and dissipative problems, the lattices): the transient is where rank is high and data are single
  samples; the solver takes it.
- **Designed data** (full-chart, low-dimensional problems only): section 3.13 shows it works at d = 3 and fails to matter at
  d = 10. Do not extend it.
- **A smaller basis** (k >= 8).

Prediction to commit for the repair: with truncation at 1e-8 and the span term, the ungated first-step worst on fisher_kpp_d5, d10
and CHEMAKZO falls below 0.5 (from the clip at 4.7-5.0), the refused share at steps 0-4 does not fall (those states are out of
span), and the served rollouts change by less than a factor 2.

---

## 5. A tube, not a box: the theory first

### 5.1 The statement

Let `S` be the set the trajectories settle on after the burn-in (measured rank 2-9 at 1e-3 on the lattices and POLLU, check 4),
`pi` a base-point map onto it, and a query `x = p + n`, `p = pi(x)`. For the step map `Phi` and a surrogate `Psi`, both twice
differentiable on the tube `T_rho = {|n| <= rho}`:

`Phi(x) - Psi(x) = [Phi(p) - Psi(p)] + [D Phi(p) - D Psi(p)] n + R`,   `|R| <= (H_Phi + H_Psi) rho^2 / 2`.

Three terms, three owners:

- **On the set:** `e_S`, what every fit so far controls (Proposition 5.1 on the bundle).
- **The normal derivative:** first order in `rho`. A fit trained on the bundle has no information about `D Phi(p) n`; what it does
  there is an accident of its form. An increment model carries `n` forward unchanged (`D Psi n = n`), the truth contracts it
  (`D Phi n` is small for a stiff map): the local table's drift in section 3.15. A chart model with orthogonal fibres sets
  `D Psi n = 0` along the wrong directions: HIRES in section 3.16, measured in 1.2.
- **Curvature:** second order; it sets the radius at which a first-order treatment stops (`rho <= sqrt(tol / (A H))`).

### 5.2 Three ways to own the first-order term, each generic

(a) **A sensitivity chart.** Choose the chart from what the map depends on, not from where the states vary: the top right singular
vectors of `D Phi` on the tube (the active subspace of the step map), or without Jacobians the row space of a reduced-rank
regression of successors on states that includes off-bundle samples. Then `Psi = G o pi` is constant along the directions `Phi` is
nearly constant along, and the first-order term is at most `sigma_{k+1}(D Phi) rho`, with `sigma_{k+1}` of order `exp(-gamma dt)`
for a stiff map. This makes Proposition 8.2's middle term vanish by construction and keeps Theorem 8's factored form. The
benchmark already holds the step map's Jacobians on 20 test trajectories (`dynamics`), so the chart can be compared with the
principal one before any fit.

(b) **A normal term in the design.** `Psi(x) = G(u) + B r(x)`, `r` the off-chart coordinates (all of them, or the few the coupling
excites), `B` constant or affine in `u`. The model stays linear in its data, so **Proposition 5.1 applies verbatim to the enlarged
design** `[phi(u), r, u (x) r]`, and its Lebesgue function at a query says whether the tube was sampled in that normal direction.
That is the tube's sampling factor and its gate in one object. Theorem 8 does not apply as it stands (the map is not constant on
the chart's fibres; the memo's scope note says so); the accumulation bound of Proposition 2.1 with the defect of the enlarged fit
does.

(c) **Jacobian labels.** Where the oracle gives `D Phi` at bundle states, `B(u)` is labelled without off-bundle solves (the memo's
gradient tier).

### 5.3 The design: where to sample

- Along the bundle: the trajectories, as now, from the burn-in on.
- Across it: the offsets the coupling produces. Transport makes convex combinations of bundle states, so its offsets lie in the
  bundle's secant space, whose dimension is the states' rank (3 to 17, check 4), not `d - k`. **The mixed pairs the calibration
  already labels (`generic3.mixed_pairs`) are tube samples; they should be in the fit, with a disjoint set kept for the gate.**
  A constant `B` needs about three times as many independent offsets as normal directions excited; an affine `B(u)` needs `(k + 1)`
  times that.
- Normal degree: two nodes at `+- rho / sqrt(2)` give degree 1 with error `H rho^2 / 4`; three give degree 2. Chebyshev points
  along `u` matter less than covering the normal directions at all.
- Radius: the offset one coupling step creates, `delta_c`, over one minus the normal contraction per step; for a stiff map
  `rho ~ delta_c`. Measure `delta_c` as the off-chart residual of the column's inputs at each `D`.

### 5.4 The gate and the closing condition

A query is served when (i) the enlarged design's `Lambda(x)` is below its threshold, (ii) the out-of-span share is below its
threshold (section 4), (iii) `|r(x)| <= rho_cal`, the largest normal offset in the calibration set. The gate certifies that the
query is where the first-order model was identified; it does not bound `H`, which stays calibrated. The tube closes over `n` steps
if `e A_n + delta_c <= rho` (Theorem 8's condition with the coupling's offset added), and the chain holds the tolerance if
`e (c n + sqrt(n)) <= tol / 2` (1.1). Both are checkable from recorded quantities.

For a tensor train: T-a fits `(u, r)` with degree 1 in `r`; T-f builds by cross on the tube box, which needs the bundle to fill its
chart box. Measure the occupied share of the box at the resolution of the nodes, and the solver's cost at random tube-box points,
before building (Demonstration 8.5 paid 9,477 solves for corners off the manifold).

### 5.5 The experiment that decides between variants

Problems: HIRES, POLLU, bruss_d20, burgers_d20, fisher_kpp_d10 (reduced charts), CHEMAKZO and robertson (full charts; the arms
should change nothing there). Arms, at equal solves:

- T0: R4d as recorded, with the repaired gate.
- T0+: T0 with as many extra *plain* pairs as the tube arms use mixed ones (the control for data volume).
- T1: principal chart plus a constant normal term, mixed pairs from the fit trajectories in the fit.
- T2: sensitivity chart (reduced-rank regression on plain and mixed pairs), fibre-constant.
- T3: T1 with `B` from the cached Jacobians and no mixed solves (where `dynamics` has them).

Scores: accepted defect on mixed inputs (median, maximum, coherence), chained mixing at 0.01 and 0.1, the column at D = 0.005,
0.05, 0.25, solver shares, with the budget of 1.1 and with `tol / 2`.

Predictions to commit (informed by check 1, so T1's first line is a replication, not a test):

- T1 cuts HIRES's median accepted mixed defect at least 20x and leaves its maximum within a factor 2 of 2.2e-3.
- With the budget `tol / (2 A)`, T0 on HIRES refuses more than 95% under mixing (it becomes honest), and T1 refuses at least 20
  points less than T0 at w = 0.01 while holding 0.01.
- T2 matches T1 within a factor 3 on the median defect with no normal term; if it does, prefer T2 (it keeps Theorem 8).
- T0+ does not improve the mixed defect by more than a factor 2 on any reduced-chart problem (plain data carry no normal
  information).
- CHEMAKZO and robertson change by less than a factor 2 in every score.

What would change the plan: if T1 lowers the median 20x and the chain still fails, the few large defects (1.2) drive it, and the
budget and refusal are the lever, not the model. If T0+ matches T1, the effect was data volume and the tube adds nothing here.

---

## 6. The gate

Each transfer failure in the records has a specific cause; fixing the causes comes before a new design.

1. **The local table (3.15).** Its model predicts the increment, so an off-span offset is carried forward and grows; its gate was
   calibrated on plain calibration states with a budget of `tol / 2` on a problem whose `A_n` is 6,669
   (`p7_online.py` docstring). The first repair is the model: predict the *successor* from the neighbours' successors, which
   collapses the off-span offset every step as the stiff map does, so that chain states return to the stored set. Prediction to
   commit: on HIRES, frozen arm, the own-input defect falls from 0.15 to below 5e-3 with the distance gate unchanged. I did not run
   this.
2. **The conjunction (3.16).** Consistent calibration is right; generalise it: each term is calibrated on the states the other
   terms accept, and each term's threshold is recorded with the count of states that set it.
3. **Calibrate on the deployed inputs.** The states that matter are the hybrid's own accepted inputs in a coupled chain. The column
   already audits them (`own_input_defect_worst`, `p6_column.py`). Make that the calibration loop: run the hybrid on calibration
   trajectories in the column, solve the accepted inputs, add them to the calibration set (and, in a second arm, to the fit), repeat
   until the largest accepted defect is within the budget for two rounds. This is ISAT's checked growth done offline, and it is
   generic across schemes.
4. **The budget** of 1.1.
5. **Online audits.** A sample-based gate cannot certify a maximum (the memo's T4). What survives a shift of inputs is an audit
   drawn from the deployed stream: solve a random share `q` of accepted queries. After `m` audits with no defect above the budget,
   the exceedance rate is below `3 / m` at 95% confidence; the speed cap is `1 / q` (1% caps at 100x and gives that statement at
   `3e-4` after 10,000 audits, a few steps of a 3-D run). With the accumulation budget, rare exceedances of bounded size do not
   break the tolerance. Report this rate, not a claim about the maximum.
6. **Conformal on chains.** Steps of one chain are not exchangeable; trajectories are. A conformal threshold on the trajectory's
   maximum is valid at level `1 - 1 / (n_cal + 1)`: 95% with the 20 calibration trajectories used now. It is the right unit (it is
   what the propagation margin is) and too weak to be the guarantee; it says how many calibration trajectories a stated level needs.

So: checked growth offline (3) plus audits online (5), on a gate whose terms are inequalities (Proposition 5.1 with the span and
normal terms). Do not add gate variants calibrated on plain states.

---

## 7. Speed: changes worth 2x or more

| change | where | evidence | expected | measurement that confirms it |
|---|---|---|---|---|
| Precompute the truncated factor `V_r S_r^-1`; the bound is one product | `generic2.bound` | check 3: 488 -> 14.7 us, 0 of 1,600 decisions change | 33x on the gate; effective 1.3 -> 4.4 on fisher_kpp_d5 at 20% refused, about 27 from step 10 on; CHEMAKZO 0.7 -> above 20 | `gate_us` and decisions before and after on chemakzo, fisher_kpp_d5, fisher_kpp_d10; bit-compare accepted sets |
| One design matrix per step, shared by prediction, bound and span term; `refuses` stops recomputing the prediction | `generic2.py:106-116`, `step_maps.py:471-472` (three designs, two projections per step) | full-rank records: gate is 1-7x the prediction | 2-4x on the step where refusal is near zero (fisher_kpp_d3: 16 -> about 4 us) | `predict_us + gate_us` against one fused call on the nine served records |
| Batched invariant projection (closed form, as `Projected.predict` does) | `generic2.py:89-92`: one `lstsq` per state in Python | robertson predicts in 18 us, same-size problems without laws in 1-2 us | 5x or more on the prediction wherever laws exist; KIDA has 15 laws on 578 species | `predict_us` with `W` and with `W` emptied, robertson, hires, KIDA |
| Burn-in by rule: the solver takes the first `m` steps, scores from `m` on | reporting and use | check 5: refused share from step 10 is 0.000-0.007 on all served lattices | 2x on bruss_d20 (13 -> 25) and burgers_d20 (14 -> 33) for long runs | `refused_by_step` on a run of 400 steps (ten times the horizon) |
| Stop a selection chain once it cannot be served (error above the margin, or solver share above one half after ten steps) | `generic3.select` | CHEMAKZO: 855-1,287 s, degrees 8-10 take 4-14 minutes each | 3x or more on selection where high degrees refuse | wall time per configuration, and the chosen configuration unchanged |
| Sparse Jacobian for KIDA | `astro.py:119-123` densifies a 582 x 582 matrix for SciPy BDF's dense LU | not measured | several times on the step; it also makes more tracers affordable, which is KIDA's limit | `nlu` and seconds per step, dense against `jac_sparsity`, 40 states |

Two notes on what a speed-up means. The baseline is `solve_ivp` called per state from Python (`system.py:62-64`); the baseline
records keep `nfev` and `nlu` so that a compiled solver's cost can be estimated (`step_cost`'s docstring). Quote every effective
speed-up against that estimate as well: at 20-40 us per state a surrogate of HIRES may not beat a compiled Radau, while Nelson
(78 ms per solve) and primordial (8 ms) are where a surrogate pays. And the algorithm itself is not the limit: once the gate costs
what the prediction does, the speed is `1 / rho`, so refusal (the burn-in, the tube, the budget) is the lever.

Skipped as below the bar: reusing one QR across nested degrees, dropping the ridge arm, thread tuning.

---

## 8. The spurious conservation laws

Check 7 (`b_inv.py`: the same null-space search on the training states and on 600 states drawn uniformly in their box) and check 8
(`c_bruss.py`):

| problem | laws on data | laws off data | smallest relative singular values on data |
|---|---|---|---|
| bruss_d20 | 1 | 0 | 2.8e-7, 2.2e-7, 4.8e-11 |
| burgers_d10 | 2 | 1 | 7.6e-3, 1.8e-13, 3.0e-14 |
| burgers_d20 | 2 | 1 | 3.3e-6, 5.8e-13, 5.9e-14 |
| robertson, hires | 1 | 1 | |
| bruss_d10, allen_cahn_d10 | 0 | 0 | |

1. **The fix's principle is right** (`W F(y) = 0` holds for every `y`, so sample `y` off the data), and it removes bruss_d20's law.
2. **Burgers is not clean.** 9.7.7 lists Burgers d10 and d20 among the true laws. One of each pair is real (the total); the other
   holds on the data to 1e-13 and fails off it: a symmetry of the initial law, not of the system. It holds on test trajectories
   because they share the initial law, and it would be wrong in a coupled run that breaks the symmetry.
3. **The size of the effect is not "100% of its scale" in any sense that reaches a score.** bruss_d20's law is orthogonal to the
   data, so its total is near zero and any change is large relative to it (the 38-265 of section 3.12 divides by that total).
   Relative to the state's scale it drifts 1.4e-8 along the test trajectories. With and without the projection R4c's bruss_d20
   surrogate differs by at most 3.5e-8 in a one-step prediction and not at all in seven digits of the 20-trajectory rollout
   (2.477944e-3 both). Measure this difference on d40 and d80 (16-55 laws) before rerunning anything; by construction those laws
   also hold on the data to 1e-9.
4. **Reach in the benchmark's store.** Among completed records at tolerance 0.01 only burgers_d10 and robertson carry invariants
   (`invariants` field; check 9); there is no completed record on a lattice of d >= 20 on this machine (burgers_d20's four are time
   limits). So the reach there is burgers_d10: its four ChebTT records have a rollout of 1.13e-2 unprojected and 3.0e-3 projected,
   and `chebtt_roll`'s acceptance uses the projected one (`step_maps.py:286-288`), so a symmetry of the initial law is part of why
   that record passes. The reach in `theory/panel` is every recipe run on a lattice of d >= 20 (`generic.py:82` calls
   `invariants_of` per fold) and p6's drift column; sizes as in point 3.
5. **Make it robust, not only fixed.** (a) Declare the laws wherever the system is known: every mass-action network from its
   stoichiometry, `conserved = ()` for Brusselator, Allen-Cahn, Fisher-KPP and Lorenz, the total for Burgers; keep the numerical
   search as a unit test that must agree with the declaration. (b) If a search stays, normalising each row by `|F|` hides slow
   reactions on a stiff network (a channel 1e-9 as fast as the fastest reads as a law): use the columns of the Jacobian, each
   normalised, at off-data states. (c) Accept a law only if it also holds along solver trajectories from off-law starts to the
   solver's tolerance.
6. This is a deliberate change to results for burgers_d10 (`RESULTS_CHANGED`), and needs the user's approval as a change to `src`.

---

## 9. Plan, in order

| # | step | purpose | CPU cost | the result that would change the plan | stop doing |
|---|---|---|---|---|---|
| 1 | Repair the gate: truncated-SVD fit, `Lambda_r` and the span term from one precomputed factor, one design per step, batched projection, effective speed-up by regime and from step 10. Rerun R4c's nine served problems and R4d. | Remove 1.3 and 1.5 from every later number | 1-2 h, mostly CHEMAKZO's selection | If the served set changes (a fit that was served through a singular solve stops holding), the R4c table is void and section 3.10 is rewritten first | Quoting gate costs or speed-ups from R4c/R4d |
| 2 | Add the budget `tol / (2 A)` and the coherence-based one beside `tol / 2`; record all three shares. No new fits. | Test 1.1 on existing surrogates | 20 min (rebuilds and chains) | If HIRES holds mixing under the coherence budget with under 50% refused, the tube arms are an optimisation, not a necessity | The propagation margin on plain chains as the only margin |
| 3 | Column at D = 0.25 and unsorted, for the 23 rows of p6. | A coupled test that differs from a rollout (1.4) | 15 min | If ChebTT's worst stays within 1.5x its plain rollout again, keep one test (the plain rollout plus own-input audits) and drop the column | Chained mixing as the headline coupled score |
| 4 | The tube experiment of 5.5 (five arms, seven problems), predictions committed first. | Decide the tube's variant | 2-3 h | T0+ matching T1: no tube term; T2 matching T1: adopt the sensitivity chart and keep Theorem 8 | First-step data design (leverage, random extras) |
| 5 | Basis for k >= 8 (section 2): total degree, T-0, T-a on POLLU, Nelson, primordial, lorenz96_d20 and two controls. | Decide whether a tensor train earns its non-convexity | 2-4 h | T-a ahead of T-0 on two or more: T-a becomes the default for k >= 8 and T-b follows; otherwise TT stays a box scheme | T-c, T-d, further ChebTT box variants at d >= 20 |
| 6 | KIDA, no fits: (a) `dt` and horizon from an MFC run, and persistence over tolerance at that `dt`; (b) the rank of the key species' futures with respect to the state, by finite differences on 20 states and 5 steps; (c) the sparse Jacobian; (d) the local arm's median against 50, 100, 200 tracers from the stored data. | Whether a closed reduced model exists, and how many tracers 0.01 dex needs | (a) the user's; (b) about 30 min of solves; (c) 1 h; (d) minutes | Rank in the hundreds: no reduction, the target needs a per-species gate and the solver elsewhere. A slope in (d) that needs more than 1e4 tracers: 0.01 dex on all 13 species is not a target for a global fit, and step 7 is the route | MLP arms and model search on KIDA |
| 7 | The local table with a successor model, offline checked growth and online audits (section 6), on HIRES, POLLU, Nelson, then KIDA. | The design built for inputs a coupled run produces | 2-3 h | Own-input defect not below 5e-3 on HIRES with the successor model: the local route needs the sensitivity metric of step 4 first | Gate variants calibrated on plain states |
| 8 | Conservation laws: declare them, measure the effect on d40 and d80, mark burgers_d10. | Correctness | 30 min plus approval | An effect above 1e-4 on any score: rerun those records | Rerunning the panel before the effect is measured |

Steps 1-3 need no new idea and decide how much of 4-7 is needed; 4 and 5 are independent and can run in either order; 6 runs
alongside.

---

## Appendix: the checks

All from `theory/panel` with
`SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=../..:.:../recipe ../../.venv/bin/python <script>`; each under three minutes.

1. `/tmp/council2-fable/a_hires.py`: `generic3.rebuild` of `records/r4d/hires.json` (64 s; e_hat 1.882e-4); `mixed_once` on 12
   training and 16 test trajectories at w = 0.01 and 0.1, labelled by the solver. Test, accepted: defect max 2.20e-3 / 2.18e-3,
   median 2.55e-4 / 6.80e-4, refused 72% / 88%; off-chart offset median 1.5e-2. With a constant matrix on the five off-chart
   principal coordinates fitted on the training mixed states: median 6.95e-6 / 1.10e-5, max 2.24e-3, R2 0.640 / 0.758. Coherence
   `|mean e| / mean |e|`: 0.84 / 0.16 on every component.
2. Inline: `p4_chebtt_mixing.json`, `p6_column.json` and `records/r4c/*.json` side by side (table in 1.4).
3. `/tmp/council2-fable/d_gate.py`: `rebuild` of `records/r4c/fisher_kpp_d5.json` (1,287 terms, rank 731 on all training data);
   per state on 1,600 test states: design 7.5 us, `bound` 488 us, prediction 6.0 us, `refuses` 505 us, triangular solve 11.3 us
   (different values: the factor is singular); precomputed pseudo-inverse factor (842 kept) 14.7 us, relative difference 2.9e-3,
   0 decisions differ; singular values 5.45e2 to 2.34e-15, rank 544 at 1e-8 and 809 at 1e-12; out-of-span share by step. Ranks
   and gate costs per record read from the `selection` field of `records/r4c` and `records/r4d`.
4. `/tmp/council2-fable/e_rank.py`, first part: the smallest rank whose truncated SVD of the centred increments (and states) over
   the training pairs leaves a largest residual of 1e-3, with all steps and with steps from 5.
5. The same script, second part: `refused_by_step`, `rollout_solved_by_step`, `gate_us`, `predict_us`, `solve_us` of the served
   records; the "gate at twice the prediction" column is an estimate, not a measurement.
6. Inline: `accumulation`, `growth`, `kind` from each problem's `dynamics.json` in the benchmark's store.
7. `/tmp/council2-fable/b_inv.py`: `invariants_of`'s search on the training states and on 600 uniform states in their box; drift
   of the data's laws along the test trajectories relative to the state's scale.
8. `/tmp/council2-fable/c_bruss.py`: `rebuild` of `records/r4c/bruss_d20.json`; predictions and the 20-trajectory rollout with
   `T.W` as built and emptied.
9. Inline: the `invariants`, `rollout_worst` and `rollout_worst_projected` fields of every
   `results/benchmark/runs/step_maps/cpu-1rank/*/*/*/tol0.01-s1.json`.

Not checked: p7's code beyond its docstring (uncommitted, not run); the sparse-Jacobian gain; the successor-model prediction of
section 6; anything on KIDA beyond the stored p5 and p5b records.
