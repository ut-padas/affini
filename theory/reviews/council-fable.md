# Council review (Fable): what to do next for surrogates of stiff step maps

Reviewer: Claude Fable 5.2, one of three independent council members. Material: `theory/SYNTHESIS.md` at commit `c758db4`, the
records it cites, the library's scheme documentation. Every check below is read-only and post hoc; the scripts are in
`/tmp/council-fable/` (not committed) and are described in the appendix so they can be rewritten in a few lines.

## Summary of the recommendation

Do not add a model family yet, and do not run section 9 in its written order. The records say three things the proposal does not
act on. (1) On the class the target belongs to (stiff chemistry with 8 or more species) almost nothing has been measured: AFINNI,
ART and SlowTable have no record on HIRES, POLLU, Nelson, primordial, the Oregonator or Van der Pol, the data-scaling run on Nelson
and POLLU used the gate that round 10 withdrew, and POLLU's "no validated reduction" is a search cap one step short of its
threshold. (2) The KIDA record carries no information about KIDA: its 298 dex error is in-sample and is a floor artifact, and the
data set itself (4,800 pairs) leaves a held-out state 0.68 dex from its nearest training state, so the decoder test of S4 would
fail for a sampling reason and the proposal's rule would then drop the idea for the wrong reason. (3) The largest effect in the
scheme table is not a basis or a gate but the propagation margin: the same ChebTT fails most of its rollouts without it
(0.02–0.10 on the rows of 1.3) and holds 15 problems with it (1.2e-3 to 7.2e-3), at effective speed-ups of 19 to 23,990 (median 1,088)
with at most 0.8% refused; section 9.1 lists
eight bottlenecks and propagation is not among them. So: first finish measuring what exists, on the chemistry rungs in order of
dimension (primordial, Nelson, POLLU, then KIDA), with the propagation margin and a gate calibrated for a maximum instead of a 99%
quantile; put ChebTT under chained mixing, where the framework predicts it passes because mixed states stay inside its box; make
KIDA a data-coverage question before it is a model question; and only then run one learned component (the nonlinear lift) and one
learned baseline (a gated MLP on Nelson, on this laptop), both judged on one curve: the worst accepted error against the share
refused, at equal data.

---

## 1. Is the proposal of section 9 right?

Its criterion (9.2) is the right starting point and most of its rejections stand. Its ranking, two of its readings of the data and
one decision rule are wrong. Findings, most important first.

### 1.1 (critical) The target class is nearly unmeasured, and the headline count hides it

"26 of 37 served" (SYNTHESIS 3.7) is a count over a panel that is 29 lattice problems, 2 oscillators and 6 chemistry problems
(`theory/panel/data/records/r4b/`). On chemistry plus oscillators the calibrated recipe serves Robertson, CHEMAKZO and HIRES; of
those, CHEMAKZO misses the tolerance on the 40 test trajectories (1.3e-2) and HIRES fails chained mixing (0.22 with 47% solved). So
one of eight target-class problems (Robertson) is clean, and KIDA is a ninth that fails.

The existing schemes were not run there. `results/benchmark/runs/step_maps/cpu-1rank/{hires,pollu,nelson,primordial,oregonator,van_der_pol}/`
each hold four ChebTT variants and nothing else (`ls`); every one of those ChebTT records is "over the 240 s time limit"
(`scheme_table.md`). AFINNI, ART, ART-quad and SlowTable have records only on the small ladders, Robertson and CHEMAKZO. Section 6
rates AFINNI and ART as fitting "curved slow manifolds" and SlowTable as fitting "in situ, near a slow manifold"; neither rating
has a record on a problem with a slow manifold above 5 dimensions. S2 and S3 are therefore not "score existing code under mixing":
they are first measurements, and they should start on the chemistry rungs, not on "every problem with an R4b result".

### 1.2 (critical) The KIDA record says nothing about KIDA, and S4's decision rule would misread the next one

- **The error is an artifact.** `records/r4b/kida_dense.json`: in-sample worst 298.3–298.7 dex and held-out worst 297.8–298.3 dex at
  every degree 1–4. An in-sample error of 298 dex at every degree is not a fit: it is `-300 - (-1.7)`, the floor in
  `generic2.py:93` (`np.log10(np.maximum(out, 1e-300))`) reached when the relative invariant projection (`generic2.py:89–92`,
  `out = yh + yh * z`) drives an abundance non-positive. (Mechanism inferred from the code and the number; I did not re-run the
  1,578 s case.) "The best held-out configuration erred 298 dex at its worst and 2.2 dex at the median" (SYNTHESIS 3.8) therefore
  measures the projection, on a chart of one coordinate that the rule itself marks "not validated".
- **The data are far too sparse for any fit.** On the committed dense trajectories (`recipe/data/records/r3_data/kida_dense_*.npz`,
  216 training and 48 test tracers, 24 steps; check K1–K3 in the appendix): 43 to 223 species are scored per state, only 23 are
  scored in every state and 294 are scored somewhere; the scored species move 0.04 dex per step at the median and the largest
  scored change per state is about 0.3 dex at the median. A held-out state's nearest training state (in 10 principal coordinates
  plus the environment) is 0.68 dex away in its worst scored species at the median, and that gap shrinks from 0.89 to 0.68 dex as
  the training set grows from 25 to 216 tracers. Copying the nearest neighbour's increment errs 0.38 dex at the median and a local
  linear fit on 60 neighbours 0.26 dex: no better than predicting no change. The share of held-out states predicted within 0.01
  dex is 0.0–0.3%.
- **Linear charts are hopeless, which agrees with the synthesis, but for a stronger reason.** An affine chart reconstructs held-out
  scored abundances to 2.1 dex at k = 20, 1.0 dex at k = 80 and 0.66 dex at k = 160 (worst; medians 0.05, 0.02, 0.009).
- **Even an exactly four-dimensional piece is undersampled.** Every tracer starts from the same abundances, so the state after the
  burn-in is a function of the four environment inputs alone. A degree-4 polynomial of those four inputs (70 terms, 216 samples)
  predicts the held-out step-0 states to 0.55 dex at worst and 0.009 dex at the median.

So the sampling factor on KIDA is orders of magnitude from the tolerance before any model is chosen. S4's "decisive first test"
(the worst held-out reconstruction error of a network decoder; "if no decoder is within the tolerance, the idea is dropped") will
fail on these 4,800 states for that reason, and the rule would drop the one learned component the proposal keeps. This is the
mistake of SYNTHESIS section 4, row 6 (reading a held-out maximum as an approximation limit) written into a decision rule. The rule
must be a scaling statement: reconstruction error against the number of tracers, beside the nearest-neighbour gap (section 4 below).

### 1.3 (critical) Propagation is the largest measured effect and is not in the bottleneck list

`scheme_table.md`, same scheme with and without a propagation margin (`chebtt_roll` retrains until the chained training
trajectories hold half the tolerance, `src/benchmark/step_maps.py:273`):

| problem | `chebtt_box` one-step / rollout | `chebtt_roll` one-step / rollout |
|---|---|---|
| allen_cahn_d2 | 1.2e-3 / 1.0e-1 | 1.1e-4 / 6.7e-3 |
| allen_cahn_d5 | 3.2e-3 / 9.4e-2 | 6.0e-5 / 1.2e-3 |
| allen_cahn_d10 | 3.9e-3 / 9.3e-2 | 1.1e-4 / 2.4e-3 |
| chemakzo | 3.3e-3 / 5.6e-2 | 3.0e-4 / 3.8e-3 |
| robertson | 2.5e-3 / 5.1e-2 | 1.8e-4 / 4.3e-3 |
| fisher_kpp_d3 | 1.8e-3 / 2.4e-2 | 6.7e-4 / 2.5e-3 |

On these rows the rollout is 10 to 80 times the one-step error, whatever the scheme: ART, ART-quad and AFINNI are near the tolerance
one step at a time (4.3e-3 to 2.0e-2) and fail every rollout they have (1.5e-2 to 0.45); SlowTable likewise (rollouts 1.8e-2 to 0.26). The only rows
of existing schemes that hold a rollout are the `chebtt_roll` rows (15 problems) and a few `chebtt_box` rows on mild dynamics.

The recipe has the same gap and no margin for it. On `allen_cahn_d5` the accepted one-step error is 1.9e-3 and the rollout 4.2e-2
(`records/r4b/allen_cahn_d5.json`: `mixing_once.worst_accepted`, `rollout_worst`), a ratio of 22, the same as ChebTT's on that
ladder; its gate accepts at `(1 + b) e_hat <= tol/2` (`generic2.py:109`), a one-step statement. So "Allen–Cahn not served" (d = 5,
40, 80) is factor 3, not sampling and not the front. Round 10's finding 7 said this ("the propagation factor is in the
decomposition and absent from the rule") and its resolution left the margin open. Section 9.1 names eight bottlenecks, none of them
propagation; S1 mentions "a propagation margin from held-out chains" in a subordinate clause. It should be the first change, for
every scheme, because it is already shown to work.

### 1.4 (major) The calibrated gate controls a 99th percentile, which is what section 7 holds against the literature

`generic2.py:141`: `e_hat` is the 99% quantile of the held-out scores `err / (1 + b)`. By construction about 1% of in-distribution
states have an error above what the gate assumes for them, with nothing bounding by how much, and the accepted range reaches
`b_max` values of 1e3 to 4e10 (`e_hat`, `b_max` in every record; `fisher_kpp_d10`: 3.8e10). With 1,600 test states on a ladder that is about
16 states per problem outside the calibration. The four C2 failures (1.25 to 2.5 times the tolerance) and the five full-chart
mixing failures are what such a gate is expected to produce; they are not, as 3.7 reads them, only "T4's point about what samples
certify". Two consequences:

- The proposal's criterion (9.2) says linear-in-data models have "a refusal rule with an accepted-error statement" and networks
  have only heuristic or marginal ones. In R4b the linear model's gate is itself a marginal, distributional score; the bound it
  multiplies is not used as a bound. The advantage of the Lebesgue score over an ensemble spread is then an empirical question
  (which score ranks the large errors better), and the project has not measured it. 9.2 claims more than R4b delivers.
- The fix is cheap and should precede everything: calibrate for the maximum (the largest held-out score with a finite-sample
  factor, pooled over splits), and pay for it in refusals. Section 4 gives the prediction.

### 1.5 (major) The selection rule takes the smallest margin it can find

`generic2.py:191` picks, among served configurations, the smallest solver share and then the lowest degree. On CHEMAKZO it chose
degree 3 (held-out chain 8.5e-3, 85% of the tolerance, share 0.00) over degree 5 (2.3e-4, share 0.01), degree 6 (6.8e-5) and
degree 7 (2.1e-5) (`records/r4b/chemakzo.json`, `selection`). The test rollout then erred 1.3e-2. R4 had 4.8e-5 on the same
problem. This C2 failure is a selection key without a margin, not a limit of held-out data. `chebtt_roll` demands half the
tolerance on the chains for exactly this reason. (For `fisher_kpp_d3` and `lorenz96_d20` no better configuration was available;
those two are the gate of 1.4.)

### 1.6 (major) Mixing is an off-data problem only for trajectory-trained schemes; the box schemes were never scored under it

The mixing protocol forms `(1 - w) x + w x'` in linear abundances (`generic2.py:203–205`). A convex combination of two states
inside a box is inside the box, in linear coordinates and, componentwise, in log coordinates. For a scheme with designed samples on
the bounding box of the training states (regime (c) of T5: logarithmic Lebesgue constant over the whole box), a mixed state is an
ordinary input. Open problem 1 ("no scheme yet holds the tolerance under chained mixing at an acceptable refusal rate") is stated
for all schemes but measured only on the chart recipe: no benchmark record has a mixing field. The framework's own prediction is
that `chebtt_roll` holds chained mixing on the problems it holds now, with nothing refused. If that is confirmed, the operator-split
regime is solved wherever a box scheme is affordable, and the open problem narrows to "problems too large for a box", which is a
different research question from the one S2 addresses.

### 1.7 (major) Two "no reduction" verdicts come from the search range

- POLLU: R4b ran with `kmax = 12` (`generic2.run` default; `r4b_calibrated.py` passes none for step maps). The lift error fell to
  8.4e-3 at k = 11 and 5.1e-3 at k = 12 against a threshold of 5.0e-3 (`records/r4b/pollu.json`, `k_curve`; `generic.py:214`), and
  the rule fell back to the full 20-dimensional chart. With 1,600 extra starts R5 validated k = 9 on POLLU
  (`r5_data_scaling.json`, `pollu_m1600`). "POLLU has no validated reduction" (SYNTHESIS 5.6, 8.3, 9.1) is not supported.
- On the 20-dimensional chart POLLU's degree-2 fit has a held-out one-step worst of 4.9e-3 and a test one-step worst of 1.1e-3
  with 10% refused, yet 89% of rollout steps are solved: the fit's own outputs leave the validated range, which a reduced chart
  with a decoder is there to prevent.
- Nelson validated at k = 12 of its 14 free coordinates, exactly at the cap.

This is the mistake of section 4, row 4 ("never let a search range end where the answer might start") again.

### 1.8 (major) R5 never tested data on the chemistry problems under a working gate

`r5_data_scaling.json`: `nelson_m400`, `nelson_m1600`, `pollu_m400`, `pollu_m1600` all report degree 1, 100% refused, under the
threshold that round 10 withdrew. Section 9.1 item 4 attributes these problems to "lifts at high ambient dimension", but the
records point at data and at the basis's size:

- primordial (60 trajectories, 1,440 fitted pairs): the in-sample worst falls 2.4, 0.66, 0.14, 4.7e-3 over degrees 1–4 while the
  held-out worst rises to 300 with bounds of 5e6 at step 0. Sampling, by the synthesis's own diagnostic.
- Nelson (60 trajectories, 16 chart coordinates with the 4 environment inputs): degree 2 has 153 terms and an in-sample worst of
  0.19; degree 3 would have 969 terms and is excluded by the two-thirds rule. A total-degree basis in 16 variables cannot reach
  the degree it needs at any affordable data: degree 4 has 4,845 terms, above the 3,000 cap.

The first is fixed by starts; the second needs a basis whose size does not grow like a total-degree space (section 2).

### 1.9 (major) The gate's cost is a property of the panel's cheap solvers, not of the target

Section 9.1 item 8 and 3.7 say the exact bound "costs about as much as the solve". In the records the solve costs 0.16–2.4 ms on
the lattice problems (`solve_us`), which is why a 0.1–0.3 ms gate matters there; and the prediction itself costs nearly as much as
the gate (`bruss_d80`: predict 220 µs, gate 330 µs, solve 2,400 µs; `allen_cahn_d80`: 100, 100, 2,000 µs with 84 terms, so the cost
is the decoder and projection, not the fit). On the chemistry problems the solve costs 8.4 ms (primordial), 78 ms (Nelson) and
about 200 ms (KIDA dense, SYNTHESIS 3.8) against gates of 10–20 µs. A sketched O(rp) bound buys nothing where the project is
going. Drop it from the plan.

### 1.10 (minor) Choices and rejections

- **End-to-end MLP: the rejection is too strong.** It rests on NN-39's 99th percentiles (networks of up to 37 species, data volumes
  and coverage not the project's) and on NN-65, a worst-case statement that APPROACHES 8.2 itself says polynomial classes share.
  After 1.4 the project's own gate is a percentile control too. "Whether learned surrogates are worth pursuing at all" (question 5)
  cannot be answered by citation. One baseline on Nelson is minutes on this CPU (section 5); it is not a run for its own sake.
- **Random or learned features (H1): correctly in reserve.** Its case rested on the withdrawn "approximation-limited fronts".
- **The fibre test and rational axes: correctly deferred**, and lower still: fronts are a lattice question, and the target has no
  lattice.
- **Latent neural ODEs, Koopman, PINNs, operators, SINDy, transformers, KANs: agreed**, for the reasons given.
- **Reduced ChebTT through a decoder: do not reject on Demonstration 8.5 alone.** One problem, before the propagation margin and
  before validated chart ranges; 1.6 makes a box scheme in chart coordinates the natural design if mixing passes on boxes. Keep it
  conditional on the result of that test.
- **S2 (in-situ tables): right idea, wrong first experiment.** SlowTable as recorded is the weakest scheme in the table (one-step
  1.1e-2 to 4.5e-2 on 8 of its 12 problems, 17–51% refused on six, effective speed-ups of 1–5 on those six). Scoring it under mixing
  on the lattice panel will show a scheme that already misses the tolerance without mixing. What decides in-situ tabulation for
  the target is how fast a table saturates on the states a coupled run produces, which needs no surrogate at all (section 5).
- **Section 3.7's table** prints `0.0e+00` rollouts for problems the solver steps entirely; print "solver" instead, since the
  column otherwise reads as a perfect result.

---

## 2. Existing schemes against new families, by problem class

All numbers from `scheme_table.md` (benchmark rollouts on 20 test trajectories) and `records/r4b/` (40 test trajectories). Every
method is given the training trajectories (4,000–17,000 solves, uncounted in every row); "solves" below are beyond those.

| Class | What holds 0.01 now | Where improving an existing scheme pays | Where it does not |
|---|---|---|---|
| Ladders and small chemistry, d ≤ 6 (13 problems incl. Robertson, CHEMAKZO) | `chebtt_roll` on all 13 (1.2e-3 to 6.7e-3; 325–57,128 solves; effective speed-ups 101–23,990; at most 0.8% refused). The recipe holds 9 of these on 40 tests at effective speed-ups of 14–396 | Nothing to build. Put the propagation margin of `chebtt_roll` into ART, AFINNI and SlowTable (their rollouts fail by 2–45x only for want of it), and score `chebtt_roll` under mixing | A new family has nothing to win here |
| Lattices, d = 10 | `chebtt_roll` on allen_cahn_d10 and lorenz96_d10 (19,387 and 682,454 solves), `chebtt_box` on fisher_kpp_d10 (107,002); the recipe on bruss, burgers, fisher, lorenz (770–1,750 lift solves) | The recipe needs 100–800x fewer solves where both finished (fisher_kpp_d10, lorenz96_d10). For Allen–Cahn the recipe needs the margin (1.3) | — |
| Lattices, d ≥ 20 | The recipe on bruss 20/40/80, burgers 20/40, fisher 20/40/80. ChebTT: every record is a 240 s limit or an evaluation cap, not a failed fit | A longer limit for `chebtt_roll` at d = 20 on one ladder tests T3's claim that ranks do not grow with d. Low priority: the target has no lattice structure | Chaotic full charts (lorenz96_d20/40): a total-degree fit in 20–40 variables at degree 2–3 with 4,000 pairs; no polynomial improvement fixes that, and no learned family is predicted to either at this data |
| Relaxation oscillators (Oregonator, Van der Pol) | Nothing | The in-sample worst does not fall with degree (Oregonator 4.7 to about 3 dex over degrees 1–12; Van der Pol 1.3e3 at every degree; bounds of 2–4 at the worst state at degrees 1–4): a genuine approximation limit of a global smooth space, the only one in the panel's records. Local schemes with a split at the jump (ART, ChebTree, ChebTT-A's interface) are the existing answer and have no record here | Van der Pol's accumulation factor reaches 3e7 and the Oregonator's 4e4 (`dynamics.json`): no surrogate holds a rollout through the jump; only refusal in the jump layer can, at a rate equal to that layer's share of the visits |
| Stiff chemistry, d = 8–20 (HIRES, primordial, Nelson, POLLU) | The recipe on HIRES only | **Most of the available gain is here.** (a) Data: fresh starts under the calibrated gate (never run, 1.8). (b) The chart range (1.7). (c) A basis that does not grow like a total-degree space in 12–20 chart coordinates: the library has SparseCheb's index sets and ART's local affine or quadratic leaves; applied to trajectory data in chart coordinates they are an adaptive version of the recipe, and ART's property that one sample serves every output is exactly what 14–578 outputs need. (d) A box scheme in the validated chart box, if 1.6 holds, for mixing | A new family is not indicated by any record: no chemistry problem has been shown limited by its space |
| KIDA (582 inputs) | Nothing, and nothing measured (1.2) | Coverage first (section 5). The step-0 map (four environment inputs to the state after burn-in) is a static function of four variables on a box: ChebTT, SparseCheb or ART apply directly with designed samples, and give the first honest number on how hard KIDA's outputs are to represent | A linear chart at any k up to 160 (K2). A nonlinear lift is the first place a learned component is indicated, but only once the data support a test |

The overall reading: improving and extending the existing schemes pays more than a new family in every class where the records
identify the limit, because the identified limits are propagation margins, gate calibration, search ranges and data, plus a
basis-size problem that the library's own sparse and local schemes address. The one place the records leave room for a learned
component is the lift at KIDA's dimension, and there the data must come first.

Nonlinear features specifically ("give them nonlinear functions"): the records support them in two places only. The decoder
(a curved lift on HIRES is 157 times better than an affine one, SYNTHESIS 2.2, `theory/demos/curved_lift.py`) and the relaxation
oscillators' jump. Rational or mapped axes address fronts, which round 10 showed are not the binding limit.

---

## 3. Published claims that deserve the least trust

Ranked by how much the proposal leans on them and how weak the evidence is. Verification status from `theory/LITERATURE.md` and
`docs/LITERATURE.md`; 65 lines there carry "abstract only", "search summary" or "memory".

1. **Speed-ups of 250x to 50,000x** (NN-32 abstract only; III-21; III-23; NN-36 abstract only). Nominal ratios against the
   authors' solvers, often batched GPU against serial CPU, with no refusal. Corollary 5.3 (c) caps the speed at 1/ρ; the panel's
   own nominal-to-effective ratios are 3 to 280 (`bruss_d2`: 3,908 against 14). *Test:* report every learned baseline with
   `speedup_effective` at the refusal rate needed for a stated accepted maximum, on the same CPU thread count as the solver.
2. **"Stable" long runs** (III-21: 1,000 iterations; NN-34: one-year runs, abstract through a search summary; NN-28 and NN-40,
   abstracts only). Stability is not accuracy: NN-34 reports regional biases, NN-33 a 13% ozone bias at 30 days. *Test:* the
   rollout's worst error on Nelson for an ungated network, beside its one-step error and the measured accumulation factor.
3. **"ISAT is proven at hundreds of species"** (III-17, abstract only, cited in S2). The 857-species result couples ISAT with
   dynamic adaptive chemistry (a reduction) and reports about 10x; ISAT controls a 90% quantile (III-13, read). It supports
   "tabulation can be made to work with reduction on combustion's accessed sets", not that a table saturates on astrochemical
   states at 0.01 dex. *Test:* the coverage curve of section 5 on KIDA tracers.
4. **"Ensembles flag the worst errors efficiently"** (NN-39, read) and the conformal literature's coverage (V-38 to V-40, NN-78 to
   NN-80). Statements about percentiles and marginal coverage. *Test:* the worst accepted error against the share refused, for the
   ensemble score and for the Lebesgue score on the same model class where both exist (a last-layer least-squares head).
5. **NN-65 as an argument against networks** (abstract only). A worst case over a class; polynomial classes share it (II-26, per
   APPROACHES 8.2), and after 1.4 neither side's gate is a bound. It should not carry a rejection.
6. **"Multi-scale sampling fixes robustness"** (NN-28, abstract only), "online training is needed" (NN-34): on the authors' cases.
   They agree with the panel's R5 result, which is why they are believed; they are not independent evidence of how much data.
   *Test:* S1's mixed-state training on HIRES and CHEMAKZO.
7. **"Direct-in-time emulation of 192 species within 0.2 dex"** (NN-36, abstract only). A function of time for fixed initial
   abundances and a static environment; it is not a step map and does not survive a driven environment. Not a counterexample to
   the difficulty measured in 1.2, and 0.2 dex is 20 tolerances.
8. **"Far outperforms Chebyshev"** for mapped points (NN-46, search summary) and "8 experts halve inference" (NN-25, abstract
   only): one-dimensional, known singularity locations; flamelet tables of 4–5 dimensions.
9. **Section 7's own negative claim** ("no published learned chemistry surrogate reports or controls a maximum error near 0.01
   dex") is the result of one day's search (APPROACHES, search date 2026-11-21) with most sources read as abstracts. It is probably
   true; it should be worded as "none found".

---

## 4. What the four factors predict, per candidate and class, and the cheapest decisive experiment

Predictions are mine, for the author to commit or amend before running. "Holds" means every test rollout within 0.01.

### C1. `chebtt_roll` under chained mixing (existing scheme, new measurement)

- *Factors.* Sampling: designed, logarithmic over the box. Off-data: none, mixed states stay in the box (1.6). Propagation: the
  margin was set on unmixed training chains; mixing with w = 0.1 contracts states toward each other and should not raise A_n.
- *Prediction.* Holds at w = 0.01 and 0.1 on at least 13 of the 15 problems it holds now, with at most 1% refused (box exits on
  Lorenz-96, as now). Risk: Allen–Cahn, where a mixed state can land nearer the unstable u = 0 than any training chain did.
- *Experiment.* Retrain the 15 models (train times 0–179 s each in the table) and run `generic.mixing` on them. About 30 minutes.
- *Rule.* 13 or more hold: the operator-split problem is one of affordability, and box schemes in validated chart boxes move up
  the plan. Fewer than 10: mixing breaks propagation margins generally, and S1's mixed-state data become the priority.

### C2. Gate for the maximum, propagation margin, margin-aware selection (the recipe, three small changes)

- *Change.* `e_hat` from the largest pooled held-out score times a finite-sample factor; accept at `(1 + b) e_hat <= tol / (2 A)`
  with A the ratio of held-out chain worst to accepted one-step worst at the chosen degree; among served configurations choose the
  smallest held-out chain, requiring it within half the tolerance.
- *Prediction by class.* Small ladders and small chemistry: the four C2 failures become holds or refusals; CHEMAKZO holds near
  1e-4 at under 5% refused (its degree 5–7 rows). Median refusal over served problems rises from 1.4% to 5–15%. Allen–Cahn d = 5
  becomes served at degree 5 with 10–30% refused; d = 40, 80 stay unserved (lift floor 4e-3 above tol / 2A). Full-chart mixing
  failures: `chemakzo` and `fisher_kpp_d2` hold; `fisher_kpp_d3`, `fisher_kpp_d5`, `lorenz96_d20` refuse more than 50% under
  mixing. Total served: 20–24 of 37, with no served problem above the tolerance on the 40 tests.
- *Experiment.* Rerun R4b's selection from its inputs; under an hour excluding KIDA.
- *Rule.* If any served problem still exceeds the tolerance on test, the Lebesgue score does not rank this problem's errors and
  the gate needs another score there (the decoder residual, a distance); record which.

### C3. Chemistry rungs with data and a full chart range (primordial, Nelson, POLLU, HIRES)

- *Factors.* Approximation: in-sample errors fall with degree on primordial and POLLU; Nelson is capped by basis size.
  Sampling: 60–100 trajectories; held-out worst at step 0. Propagation: worst-case accumulation over the horizon 99 (primordial), 82 (Nelson), 234 (POLLU), 5.3e3 (HIRES) (`dynamics.json`),
  so the margin of C2 matters here more than on the ladders (31–123).
- *Prediction.* With 1,600 fresh starts, `kmax` equal to the free dimension and C2's gate: POLLU validates k = 9–13 and is served
  with under 20% refused; primordial is served at degree 4 with under 30% refused; Nelson is not served by the total-degree
  recipe at any data (needs degree 4 or more in 16 variables). HIRES unchanged.
- *Experiment.* Extras exist for Nelson and POLLU (`r5_extra_nelson.npz`, `r5_extra_pollu.npz`); primordial needs 1,600 solves at
  8.4 ms. Minutes per problem.
- *Rule.* Both served: chemistry at 10–20 species is a data problem and the recipe is the baseline for the rest. Nelson unserved
  as predicted: run C4 on it. Primordial or POLLU unserved with in-sample error below 1e-3: the gate is the limit; back to C2.

### C4. A basis that scales in chart dimension: sparse index sets and local leaves on trajectory data

- *Change.* In the recipe, (a) hyperbolic-cross or anisotropic index sets in place of total degree (SparseCheb's sets), with the
  environment inputs given their own, higher degree; (b) ART-style local affine or quadratic leaves on the chart coordinates,
  fitted to the trajectory pairs, one Lebesgue bound per leaf.
- *Factors.* Approximation: the same or better at a fraction of the terms where interactions are low-order. Sampling: fewer terms
  per sample, so smaller bounds; leaves keep p small but need points per leaf.
- *Prediction.* Nelson: (a) reaches an in-sample worst below 5e-3 with under 600 terms and is served with 1,600 extras at 20–50%
  refused; (b) needs more than 20 points per term per leaf and is served only with 6,400 extras. Oregonator: (b) with a split at
  the jump is served with 20–40% refused; (a) is not. Lattices: no change.
- *Experiment.* (a) is an index-set option in `generic.multi_indices`; (b) a k-d partition of the fitted pairs. An afternoon each.
- *Rule.* Nelson served by either: the existing schemes' ideas cover chemistry at 18 dimensions, and KIDA's question is only the
  lift and the data. Nelson unserved with 6,400 extras by both: this is the first record of a space limit on chemistry, and the
  learned baseline C7 becomes informative rather than optional.

### C5. KIDA coverage (no surrogate)

- *Factors.* Sampling only. From K3: the nearest-neighbour gap falls as roughly N^-0.13, the rate of a set of about 8 effective
  dimensions (a rough estimate from four sizes and a median).
- *Prediction.* With 1,000 dense tracers the median gap is still above 0.5 dex and no method predicts more than 5% of held-out
  states within 0.01 dex. With tracers drawn from one narrow environment range (0.3 dex per input instead of 2) the gap falls
  below 0.2 dex and a local linear fit predicts more than half the states within 0.05 dex.
- *Experiment and rule.* Section 5, experiment 1.

### C6. A nonlinear lift for KIDA (S4, with the rule changed)

- *Factors.* Approximation: the lower bound of Proposition 6.5 allows k near the effective dimension (about 8–13 by construction:
  four environment inputs, the step, and the path's eight parameters). Sampling: no computable factor for a network decoder; its
  worst held-out reconstruction cannot be below the data's own resolution.
- *Prediction.* On the present 4,800 states a network decoder at k = 8–16 reconstructs held-out scored abundances to 0.3–1 dex at
  worst and 0.01–0.03 dex at the median: better than the affine chart (2.1 dex at k = 20) and far from the tolerance. On Nelson
  and POLLU it adds nothing over the quartic decoder.
- *Experiment.* Decoder only, at 50, 100 and 216 tracers; a small MLP on CPU, under an hour.
- *Rule.* Judge the slope, not the level: continue if the worst error falls at least as fast as the nearest-neighbour gap and the
  median is below 0.02 dex; drop the lift only if it does no better than the affine chart at equal k. Never drop it on the level
  at 4,800 states.

### C7. One learned baseline: a gated one-step MLP on Nelson (CPU now), KIDA dense later (GPU)

- *Factors.* Approximation: no advantage predicted on smooth chemistry maps. Sampling: no computable factor; an ensemble spread
  as the score. Propagation and off-data: as every scheme.
- *Prediction.* On Nelson with the training trajectories plus 6,400 extras, a 5-member ensemble of small residual MLPs in the
  recipe's coordinates: median one-step error 3e-3 to 1e-2, worst above 0.1; the worst accepted error stays above 0.01 until more
  than 70% is refused, against under 50% for the best linear-in-data fit of C3–C4 at the same data.
- *Experiment.* Minutes to train on this laptop; score with R4b's fields.
- *Rule.* If the network holds 0.01 at a lower refusal than every linear-in-data fit on Nelson, the criterion of 9.2 is wrong in
  practice and networks earn a place on KIDA when the GPU is available. If not, the rejection stands on the project's own data.

### C8. A box scheme in a validated chart box (conditional on C1)

- *Prediction.* On HIRES (k = 3) and POLLU (k = 9–13), ChebTT on the bounding box of the chart coordinates of the visited states,
  refusing by the decoder residual, with the margin of `chebtt_roll`: HIRES holds chained mixing at w = 0.01 with under 50%
  solved (now 0.22 at 47%). POLLU's box at k ≥ 9 is too large for ChebTT in the time limit.
- *Rule.* HIRES holds: designed samples in chart coordinates are the mixing design for low k. Fails by decoded corners being
  unphysical: restrict to the convex hull, or drop.

---

## 5. The minimum set that decides whether learned surrogates are worth pursuing for the target

Four experiments, in this order; the first three run on this laptop.

1. **KIDA coverage curve (no model).** Generate dense tracers to 1,000 (a burn-in of 3.6–10 s and 24 steps of 0.2 s each per
   tracer on one core, SYNTHESIS 3.8: about 4 core-hours, an hour on four workers, as resumable records). Record, against the
   number of tracers: the nearest-neighbour gap in scored species, and the nearest-increment and local-linear errors of K2–K3
   (maximum, 99th percentile, median, share within 0.01 and 0.1 dex). Repeat on one narrow environment range.
   *Decides:* how many states any data-driven surrogate needs. If the gap's slope implies more than about 1e7 states for a
   0.1 dex gap over the full range, an offline surrogate of the full dense range at 0.01 dex is out of reach for every family,
   learned or not, and the options are an in-situ table on the states one simulation produces, a narrower range, or a different
   metric.
2. **The scored set and the tolerance, decided by the user.** At present every species above 1e-12 is scored: 43–223 per state,
   with membership changing along a trajectory (K1). The same check on the 20–40 species an MHD run needs (coolants, charge
   carriers, observables) may differ by orders of magnitude in difficulty. This is a decision about the application, not an
   experiment, and it changes every number above; it should be made before any KIDA model is trained.
3. **Nelson as the rung.** C3, C4 and C7 on Nelson (14 species, 4 environment inputs, 78 ms solves): the recipe with data and a
   scaling basis, beside one gated MLP ensemble, on the curve of worst accepted error against share refused at 1,600 and 6,400
   extra starts. *Decides:* whether, where data are affordable, a network beats the linear-in-data designs at the tail. If nothing
   holds 0.01 on Nelson at under 50% refused with 6,400 extras, nothing will on KIDA.
4. **On the GPU, only if 1 and 3 allow it:** the lift of C6 and the MLP of C7 on KIDA dense at the data volume experiment 1 says
   is needed, with the same curve.

What would make learned surrogates not worth pursuing: experiment 1 showing no saturation at any affordable volume on the range
the application needs, and experiment 3 showing the network's tail no better than the polynomial's at equal data. What would make
them worth pursuing: experiment 3 showing a lower refusal rate at the tolerance for the network, or experiment 1 showing a low
effective dimension that only a nonlinear lift can exploit (K2's affine charts cannot).

---

## 6. Recommended plan

| # | Step | Purpose | CPU cost | What would change the plan | 
|---|---|---|---|---|
| 1 | Fix the three rule defects (C2): gate for the maximum, propagation margin, margin-aware selection; `kmax` to the free dimension; rerun R4b's selection without KIDA | Remove the failures that are the rule's own; get honest refusal rates | under 1 h | A served problem still above tolerance on test: the Lebesgue score fails there; add a second score before going on |
| 2 | `chebtt_roll` under chained mixing (C1) | Test whether mixing is off-data for box schemes at all | 30 min | Holds: C8 moves up, S2 moves down. Fails: mixed-state training data (S1) first |
| 3 | Chemistry rungs with fresh starts (C3), in order primordial, POLLU, Nelson | Separate data from space on the target's class | minutes each; Nelson's 6,400 extras about 10 min | Both small rungs served: chemistry at 10–20 species is settled as a data problem |
| 4 | Sparse index sets and local leaves in the recipe (C4), on Nelson and the Oregonator; the propagation margin in ART, AFINNI, SlowTable and their first runs on the chemistry problems with a 15-minute limit | The "more adaptive" versions of what exists, where the records ask for them | an afternoon to write, minutes to run | Nelson unserved by both: first evidence of a space limit; C7 becomes necessary |
| 5 | The scored set and tolerance for KIDA (user's decision), then the coverage curve (section 5, experiments 1–2); fix the projection floor so KIDA errors are finite | Make KIDA a measured problem | 1 h on four workers | No saturation: stop offline KIDA surrogates of the full range; consider in-situ only |
| 6 | One gated MLP ensemble on Nelson (C7), beside step 4's best | Test criterion 9.2 on the project's own terms | under 1 h | Network wins the tail: add it to the GPU list for KIDA |
| 7 | Decoder scaling test on KIDA (C6) at three data sizes; C8 on HIRES if step 2 held | The one learned component, judged by slope | under 1 h each | Lift no better than affine at equal k: drop it |
| 8 | In-situ table on KIDA tracers: retrieval share against table size, on the stream of one tracer family | Whether tabulation saturates on what a run visits | hours (a Jacobian per record is the cost; measure it first) | Retrieval share below 50% after 1e4 records: tabulation does not serve the target either |

**Stop doing:**
- climbing the lattice ladders past d = 10 and reporting a served count dominated by them; report by class, chemistry first;
- running the chart recipe on KIDA before step 5 (1,578 s for a record with no information);
- optimising the gate's cost (a sketched bound): irrelevant at 8–200 ms solves;
- the fibre test and rational or mapped axes, until a record shows a front as the binding limit;
- deciding against a family from a held-out level at one data size (S4's rule), or from published percentiles alone (9.4's MLP row);
- scoring SlowTable under mixing on the lattice panel as the test of in-situ tabulation.

---

## Appendix: the checks

All read-only, on committed data; run from `theory/` with
`SURROGATE_DEVICE=cpu SURROGATE_MPI=off OMP_NUM_THREADS=4 ../.venv/bin/python /tmp/council-fable/<script>`; each under one minute.
They are exploratory and post hoc: medians over 1,152 held-out states, one random subsample per size, no tuning. Treat the
effective dimension in particular as an order of magnitude.

**K1 (`kida1.py`): what the dense data are.** Arrays `(216, 25, 582)` and `(48, 25, 582)`, log10 abundances floored at -30 plus
four log10 environment inputs. Scored species (above -12) per state: mean 145.7, minimum 43, maximum 223; scored in every state
23; scored in some state 294. Change per step over scored entries: median 0.041 dex, 99th percentile 1.03, maximum 14.7.

**K2 (`kida2.py`): affine charts and local prediction.** Abundances clipped at -14; training on the first 200 tracers, scoring on
the first 40 test tracers (960 states), errors over species scored in the true state.
- Affine chart (principal directions of the training states), held-out reconstruction, worst / 99th percentile / median:
  k = 5: 6.47 / 1.12 / 0.126; k = 20: 2.09 / 0.477 / 0.053; k = 80: 1.01 / 0.207 / 0.019; k = 160: 0.659 / 0.116 / 0.0093.
  Singular values relative to the first: 0.29 at the 5th, 0.16 at the 10th, 0.077 at the 20th, 0.046 at the 30th.
- No change (persistence), per scored entry: worst 3.07, 99th percentile 1.01, median 0.041.
- Nearest training pair's increment, in 10 principal coordinates plus the environment (per state, worst scored species): median
  0.395, 90th percentile 1.19, worst 3.49; within 0.01 dex 0.0%, within 0.1 dex 3.7%. Among the 10% of states nearest to the
  data: worst 0.95, median 0.22.
- Local linear fit on 60 neighbours: median 0.256, worst 2.57; within 0.1 dex 13.5%. With 20 or 40 principal coordinates: worse.

**K3 (`kida3.py`): scaling and the step-0 map.** Nearest neighbour as in K2, 48 test tracers, training subsets of 25, 50, 100, 216
tracers: state gap in the worst scored species (median) 0.886, 0.791, 0.713, 0.676 dex; increment error (median) 0.419, 0.424,
0.365, 0.377. Median error by step with 216 tracers: 0.17 at step 0, 0.28–0.48 afterwards; the largest scored change per state
(median) is 2.49 at step 0 and 0.25–0.49 afterwards. Step-0 state as a polynomial of the four environment inputs (216 samples),
held-out worst / median: degree 1: 1.53 / 0.097; degree 2: 0.85 / 0.037; degree 3: 0.65 / 0.017; degree 4 (70 terms): 0.55 / 0.0093.

**Record readings** (no script kept; `python3` over `panel/data/records/r4b/*.json`): the fields `selection[*].insample_worst`,
`heldout_worst`, `chain_worst`, `solver_share`, `splits[*].step_of_worst`, `bound_at_worst`; `k_curve`; `e_hat`, `b_max`;
`predict_us`, `gate_us`, `solve_us`; `mixing_once`, `mixing_chained`. Accumulation factors from
`results/benchmark/data/step_maps/*/dynamics.json`.
