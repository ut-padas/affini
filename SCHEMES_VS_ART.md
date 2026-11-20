# Where AFINNI and ART beat the global schemes, and what to do about it

Findings of two independent reviews of the benchmark store (`results/benchmark/`, schema 2, records up to
2026-11-20) and of both companions' `convergence.json`: an analysis (Claude Opus 5.5, high effort) and a
critical review of it with its own experiments (Fable 5.2, medium effort). "Measured" marks numbers from the
reviewer's own runs; everything else comes from the records or the code. Neither review changed the
repository. Their scripts were in the session's scratch folder (`scratchpad/a2.py`, `a7.py`, `review/*.py`),
which is not kept.

## Summary

- The function suite has 402 cases (problem, d, tolerance). ART or ART with quadratic leaves uses fewer
  evaluations than every scheme in 21 of them; AFINNI in none. In the step-map suite ART wins one case,
  Robertson at 0.01. There is no case where AFINNI or ART holds the tolerance and every scheme fails.
  Schemes hold 376 cases; the baselines 36 (AFINNI 11/224, ART 13/282, ART-quad 33/282).
- The 21 wins are seven phenomena. Nine are one affine function (Lorenz-96 at d = 3, three problems times
  three tolerances), three are swirl at d = 20, where the test set only sees the identity.
- Almost every win comes from fixed evaluation floors: ART's one or few leaves of 20p points (p coefficients
  per leaf) against the schemes' tolerance-independent start-up and verification costs.
- Two real advantages of ART: its cost does not grow with the number of outputs (ChebTT builds one train per
  output), and its leaves follow oblique low-rank curvature (ridges, slow manifolds).
- The reviewer's main correction: ChebTT's seed-unstable cost on `lotka_volterra` at d = 20 is not a coarse
  refinement step. The expensive seeds buy accuracy at the box's vertices, which the 50,000 uniform test
  points never reach; there the cheap seed and ART-quad are both wrong. Which reading is right depends on
  the norm the benchmark claims (see "Decision" below).

## The winning cases

Seed means over seeds 1–3 (as `src/benchmark/table.py` counts them); evaluations, with worst error over
the tolerance in brackets. Records: `results/benchmark/runs/functions/cpu-8rank/<problem>/d<d>/<method>/tol<t>-s<seed>.json`
and `results/benchmark/runs/step_maps/cpu-1rank/robertson/d3/<method>/tol0.01-s1.json`.

| Pattern | Case (problem, d, tol) | Winner | Cheapest scheme that holds |
|---|---|---|---|
| A. Affine map | `lorenz96` 3, all three tolerances | ART 81 (0.00) | SparseCheb 177; ChebTT-A 388–415, ChebTT 389–424 |
| | `lorenz96_chaotic` 3, all three | ART 81 (0.00) | ChebTT-A 403 |
| | `lorenz96_state` 3, all three | ART 81 (0.00) | ChebTT 538 |
| A'. Identity on the test set | `swirl` 20 at 0.2, 0.1, 0.05 | ART 421 (0.004–0.014) | hub 6,000 (same errors); ChebTT 39k / 120k / 210k |
| B. Near-quadratic, loose tolerance | `genz_gaussian` 10 0.2 | ART-quad 1,321 (0.93–0.99) | ChebTT-A 1,685 (0.28–0.29) |
| | `genz_product_peak` 10 0.2 | ART-quad 1,321 (0.84–0.87) | ChebTT-A 1,664 |
| | `lotka_volterra` 5 0.1 | ART-quad 421 (0.62–0.65) | ChebTT-A 1,038 (seeds 762, 762, 1,591) |
| | `lotka_volterra` 10 0.2 | ART-quad 1,321 (0.62) | ChebTT-A 1,521 |
| | `lotka_volterra` 20 0.2 | ART-quad 4,621 (0.67–0.71) | hub 14,667; ChebTT 45.6k (seeds 48,509; 3,223; 85,208) |
| | `rosenbrock_gradient` 2 0.1 | ART-quad 243 (two leaves) | ChebTT 297 (exact) |
| | `lorenz96_state` 2 0.1 | ART-quad 243 (two leaves) | ChebTT 352 |
| C. Vector outputs, m = d | `lotka_volterra_state` 10 0.2 | ART-quad 1,321 (0.72–0.78) | ChebTT, ChebTT-M, ChebTT-A 12,417 each; hub 12,667 |
| | `lotka_volterra_state` 20 0.2 | ART-quad 4,621 (0.67–0.71) | ChebTT 360,828, ChebTT-M 362,372; ChebTT-A and hub over 90 s |
| D. Stiff step map | `robertson` 3 0.01 (one-step) | ART 655 (0.63) | ChebTT-box/A 955 (0.25); `chebtt_roll` 1,159 |

At `lotka_volterra_state` 10 0.1, ART-quad fails (1.44–1.55) and every scheme holds at about 33k: ART's
vector win is at the loosest tolerance only. On `lotka_volterra_state` all three ChebTT variants are the
same model (ChebTT-M trains on the axes first and all outputs succeed, so it rotates nothing; ChebTT-A adds
no interface term).

Other senses of "better":

- Rollouts on Robertson at 0.01: ART 0.0163 against ChebTT-box/A 0.0505, with 31% fewer evaluations;
  `chebtt_roll` is better than both (0.00434, 1,159 evaluations). `rollout_over_bound`: ART 0.17–0.55
  (13 records), ART-quad 0.12–0.90 (17), ChebTT-box 0.28–0.97 (32).
- Below the schemes' smallest budgets only ART has a model at all (convergence study: ridge d = 2 at 247–991
  evaluations against 1,223; Robertson 245–409 against 558).
- Inference: ART is faster only on vector problems at d >= 10 (`lotka_volterra_state` 20 0.2: 2.1 us against
  ChebTT's 10.6 us). On scalar problems compressed ChebTT is 3–8x faster than ART-quad.
- Training-time wins follow the evaluation wins times the oracle's cost, except swirl at d = 20, where 20
  ChebTT trains spend 4.4 s on a 0.8 us oracle.

## The floors (verified against the records)

- ART, one leaf: `ceil(20 p) + 1` (`sample_multiplier=20` in `src/benchmark/methods.py`, plus the centroid):
  affine p = d + 1 gives 81 at d = 3 and 421 at d = 20; quadratic p = (d+1)(d+2)/2 gives 421, 1,321, 4,621
  at d = 5, 10, 20.
- Hub: `200 d + 2000` (2,400 … 6,000 at d = 2 … 20).
- ChebTT: 64 + 32d Sobol checks, a degree pre-pass of 36d (start degree 8, `_fibre_degree(n0=8)`), about
  36(d − 1) per rank-2 cross sweep and a 16d look-ahead: about 120d–156d in all; checks and look-ahead are
  30–52% of it.
- SparseCheb at d = 3: 168 of its 177 evaluations are checks.

## Why, pattern by pattern

- **A, affine Lorenz-96 at d = 3 (artefact).** With d = 3 the indices i + 1 and i − 2 coincide mod 3, so
  (x_{i+1} − x_{i−2}) x_{i−1} vanishes; x' = −x + F and the flow map F + e^{−t}(x − F) is affine. ART's first
  affine leaf is exact; every scheme's spend is overhead. Measured: an affine least-squares fit on 80 Sobol
  points scores 0.00 tol.
- **A', swirl at d = 20 (artefact).** The test set's smallest |x|² is 7.676 (measured), so the rotation angle
  2 exp(−|x|²) is at most 9.3e-4 rad and f is the identity to about 1e-3 at every test point; the identity
  model scores 0.0036 tol. Neither ART nor ChebTT resolves the vortex: inside the unit ball the identity scores
  1.56 tol and ChebTT (39k evaluations) 1.54 (measured).
- **B, near-quadratic at a loose tolerance (partly real).** ART wins when the best quadratic error
  E₂(f) = inf_q ‖f − q‖∞ is below tol·max|f|. ChebTT's acceptance targets `safety = 0.5` of the tolerance, so
  it accepts at 0.28–0.50 tol. The margin is the safety factor, not the start degree (measured: at start degree
  4 the accepted models still score 0.23–0.31). ART accepts in-sample (`error_metric="Linf"`,
  `validation_samples=0`): its holds sit at 0.62–0.99, ART with affine leaves fails the same cases at 1.3–1.9,
  and one quadratic refitted on 192 Sobol points scores 0.83–1.17 tol on the genz cases (three of six refits
  fail; measured). Not every B case is one leaf: on `rosenbrock_gradient` 2 and `lorenz96_state` 2 a single
  quadratic scores 2.0–2.7 tol, and ART-quad used two leaves.
- **The seed-unstable cost on `lotka_volterra` 20 0.2 (corrected).** Measured round by round:

  | Seed | Round 0 worst (rank 2) | Path | Evaluations |
  |---|---|---|---|
  | 2 | 0.49 | accept | 3,221 |
  | 1 | 0.70 | rank 4, 6, 8 | 48,507 |
  | 3 | 0.80 | rank 4, 6, 8, 10 | 85,206 |

  Degrees never change; only the rank grows. At rank 4 the model already scores 0.30 on the Sobol checks
  and 0.38 on the test set; what fails are the free checks and the look-ahead, at points with 19–20 of 20
  coordinates on the boundary. Maxvol moves the cross to the vertices, and the check error rises with rank
  (0.70 → 1.02 → 1.03 for seed 1). At random vertices of the box (measured): ART-quad 2.7 tol, the cheap
  ChebTT seed 1.12, the expensive seeds 0.43 and 0.25. With no free checks (`n_free=0`) the three seeds cost
  10,213 / 3,221 / 10,360, compressed to rank 1, degrees 1–2, test error 0.37–0.54 (measured).
- **C, vector outputs (real).** `build_vector` trains one ChebTT per output on a shared point cache; only the
  Sobol set and the pre-pass coincide. On `lotka_volterra_state` 10 0.2 (measured): eight outputs accept at
  round 0 with 610–860 new points each and two need a rank round (about 3,000 each), 12,867 distinct points in
  all, 38% shared. At d = 20 the 360k is about 18k per output: the vertex escalation above, twenty times over.
  ART puts each point to use for every output.
- **Ridges and slow manifolds (real).** For g(q·x) the Hessian g'' q qᵀ has rank 1; optimal anisotropic
  piecewise-linear meshes then need N ~ tol^{−r/2} pieces with r the Hessian's rank (Chen, Sun and Xu 2007).
  ART's measured ridge rates are 0.54, 0.55, 0.82 at d = 2, 3, 5, against 1.0, 1.52, 2.59 on Rosenbrock. On
  the ridge at d = 5, tol 0.1: ChebTT 45,633, ChebTT-M 6,241 (rotated, rank 1), hub 3,000, ART 609. ChebTT-M
  and the hub already find the direction; their remaining cost is their floor at loose tolerance.
- **D, rollouts.** ChebTT's interpolation error is smooth and may keep its sign along slow trajectories
  (adding up like N|e|), while least-squares residuals change sign within a leaf (like sqrt(N)|e|). Not tested;
  consistent with ChebTT's smaller one-step error (0.25 against 0.63) and three times larger rollout.
  `chebtt_roll` already corrects it.

## Hypothesis tests (reviewer, measured)

| Hypothesis | Result | Verdict |
|---|---|---|
| A lower start degree cuts the floor | Start 4: `genz_gaussian` 10 1,684 → 1,044–1,104; `lotka_volterra` 5 761 → 482–501; `lorenz96` 3 422 → 290 (226 at start 2) | Confirmed |
| A lower start degree is safe elsewhere | `michalewicz` d = 2, 3 never converges (rank grows forever at worst 17.3); `griewank` 10 +77%; `rosenbrock` 10 +14%; `genz_continuous` 5 +17–20%; `lotka_volterra` 10 0.2 seed 1 escalates to 2,414 | Refuted |
| One quadratic on the Sobol checks holds B and C | Holds `lotka_volterra` 10, 20 and `_state` 10, 20 at 0.2 (test 0.59–0.82); fails both genz cases, `lotka_volterra` 5 0.1, everything at 0.1 | Partly |
| That fit is well conditioned | Condition number 10–11 at d = 10, 33–39 at d = 20 | Confirmed |
| That fit never falsely accepts | One false accept in 66: `genz_corner_peak` 20 seed 3 (validation 0.04, test 4.86) | Refuted |
| Finer rank steps or better pivots stabilise the cost | `rank_step=1` 100–107k; `sweeps=3` 75–130k; `rank0=4` 44–81k; worst-point pivots mixed | Refuted |
| The escalation is caused by checks at vertices | `n_free=0` gives 10k / 3.2k / 10k | Confirmed |
| Block TT with shared pivots cuts the vector cost | Shared-pivot cost 36–42 against Σr² 10–70 at 0.2, 143 against 118 at 0.1 (`lotka_volterra_state` 10) | Refuted |

## Fixes, with both reviews' verdicts

1. **A "level-0" affine or quadratic fit on ChebTT's own check points**, before any cross (in
   `src/schemes/chebtt/train.py`, after the checks are evaluated; one factorisation for all outputs in
   `build_vector`). Both agree; it is the only proposal that moves the vector cases (measured: 564 against
   12,417 at d = 10 and 1,064 against 360,828 at d = 20 on `lotka_volterra_state` 0.2; 160–214 on Lorenz-96 at
   d = 3). It needs fibre checks through the centre and through the largest |f| (without them it would accept
   swirl 20, `gaussian` 10, `griewank` 20) and a vertex probe (the `genz_corner_peak` false accept). It does
   nothing at tol 0.1. Least squares on uniform points needs about m² log m points for stability (Cohen,
   Davenport and Leviatan 2013); the measured condition numbers show the bound is far from binding at degree 2.
2. **Decide the norm before changing ChebTT** (reviewer). See below; it decides whether the 48k–85k seeds are
   the honest cost or the vertex checks can go.
3. **A boundary-aware second test set** in the function suite (random vertices; uniform points with half the
   coordinates pushed to an end), and ART reported with held-out validation (`validation_samples > 0`). Both
   reviews agree; cheap, and it would have exposed the swirl and Lotka–Volterra artefacts. Flag Lorenz-96 at
   d = 3 as linear, and shrink swirl's box with d.
4. **A lower start degree only with a recovery rule** (reviewer): decide degree growth from the look-ahead
   fibres' own coefficient tails (f on Chebyshev points through the worst check point), which show a missing
   degree where the model's own tail cannot. Untested; it is what would make start degree 4 safe, and it closes
   the same latent hole at degree 8. A plain lower start degree regresses (table above).
5. **Evaluate only the new points when the pre-pass doubles a degree** (it re-evaluates the nested ones: the
   +1–4% on the controls), and **share the look-ahead bases across outputs** in `build_vector` (16d instead of
   16d·m; estimated from the code).
6. **Hub: reuse the 200d pilot points in the fit** and size its first sample from the number of terms
   (analyst; not evaluated by the reviewer). The pilot is 67% of the hub's floor at d = 20. Gradient sketching
   (2k instead of 2d evaluations per pilot point; Constantine, Eftekhari and Wakin 2015) is a further option.
7. Not recommended: Dörfler marking or Leja growth for the rank escalation (the escalation is rank-only), and
   block TT for the vector outputs (measured above).

## Decision needed: what "held" means

- **Worst error over the closed box, vertices included.** Then the 48k–85k seeds are correct, and seed 2 and
  ART-quad pass only because the test set never reaches the vertices.
- **Error under the sampling measure** (the region the samples cover). Then ChebTT can drop its vertex checks:
  about 10k evaluations on `lotka_volterra` 20 0.2 with no other change, about 1k with the level-0 fit. The
  reviewer judged this reading closer to the astrochemistry target, whose states lie on a manifold, but left it
  as a decision.

## Open experiments, cheapest first

1. Re-score the stored models of the 21 winning cases on a vertex and centre test set (minutes; no training).
2. ART-quad with held-out validation on the B cases.
3. The level-0 fit with a vertex probe on `genz_corner_peak` 20 and the 22 controls.
4. Degree growth from the look-ahead's data, then start degree 4 on `michalewicz`, `griewank` and
   `genz_continuous`.
5. `lotka_volterra_state` 20 0.2 with `n_free=0`, to confirm the per-output cost falls to about 10k.
6. The rollout sign-coherence claim: the sign autocorrelation of one-step errors along a Robertson trajectory,
   for both models.

## Sources

- L. Chen, P. Sun and J. Xu, Optimal anisotropic meshes for minimizing interpolation errors in L^p-norm,
  Math. Comp. 76 (2007). https://www.ams.org/journals/mcom/2007-76-257/S0025-5718-06-01896-5
- A. Cohen, M. A. Davenport and D. Leviatan, On the stability and accuracy of least squares approximations,
  Found. Comput. Math. 13 (2013). https://arxiv.org/abs/1111.4422
- P. G. Constantine, A. Eftekhari and M. B. Wakin, Computing active subspaces efficiently with gradient
  sketching (2015). https://arxiv.org/abs/1506.04190
- S. Dolgov and D. Savostyanov, Parallel cross interpolation for high-precision calculation of
  high-dimensional integrals, Comput. Phys. Commun. (2020). https://arxiv.org/abs/1903.11554
- W. Dörfler, A convergent adaptive algorithm for Poisson's equation, SIAM J. Numer. Anal. 33 (1996).
- Chebfun's `standardChop` (needs at least 17 coefficients before chopping).
  https://github.com/chebfun/chebfun/blob/master/standardChop.m
