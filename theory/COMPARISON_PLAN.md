# Plan: a fair comparison of the chart recipe, AFINNI, ART and rollout-tuned ChebTT (for review, not yet run)

## What exists today

- **AFINNI and ART were run only on the small step maps**, by the benchmark (`results/benchmark/runs/step_maps/`), under its 90-300 s time
  limits. They finished on the d <= 4 lattices, Robertson and lorenz96_d4, and timed out on CHEMAKZO, fisher_kpp_d5, allen_cahn_d5
  (AFINNI and affine ART), lorenz96_d5 and at d >= 10. They were never run on HIRES, POLLU, Nelson, primordial, the oscillators or the
  d = 10-20 lattices.
- **Where they finished, they meet the one-step tolerance but not rollouts.** At tolerance 0.01 their one-step worst is 0.004-0.02 and
  their 20-trajectory rollouts 0.015-0.45 (fisher_kpp_d3: AFINNI 0.035, ART 0.041; Robertson: ART 0.016, quadratic ART 0.051). They are
  built as one-step approximants with no propagation margin, which is what `chebtt_roll` adds to ChebTT (Robertson 4.3e-3,
  CHEMAKZO 3.8e-3).
- **None of them was scored under this study's protocol**: closed columns, long chains, refusal by cause, speed-up against the fastest
  exact solver.
- **No method uses gradients.** The benchmark gives AFINNI and ART value-only oracles (`Evaluator(f_fn=...)`, no Jacobian or Hessian),
  ChebTT's cross approximation uses values only, and the chart recipe fits values only. The only gradient work was the early HIRES case
  study (`theory/gradients/`).

So no fair comparison exists. The differences that would make a naive one unfair:

| Aspect | Chart recipe (R4g) | AFINNI, ART | ChebTT (`chebtt_roll`) |
|---|---|---|---|
| Data access | trajectory data (fixed set); solver calls only for calibration, selection fallbacks, refused-state data | an oracle anywhere in the bounding box of the visited states, queried adaptively | an oracle on the box (cross approximation) |
| Accuracy target | one-step gate calibrated for the maximum, propagation margin in selection | one-step tolerance only | one-step tolerance, retrained until training chains hold tol / 2 |
| Refusal | capped gate plus account (fallback to the solver) | AFINNI: trust regions; ART: none inside the box; both refuse outside the box | refuses outside the box |
| Gradients | none | none (supported: an exact Hessian for one ART split strategy) | none |
| Devices | CPU (NumPy) | CPU or GPU, MPI with work stealing | CPU |

## The comparison

**Principle:** every method in its best version, given the same budget and judged by the same protocol.

1. **Best versions.**
   - **The chart recipe:** R4g (the capped gate, the decaying account) with refusal-driven data.
   - **`chebtt_roll`:** as the benchmark builds it.
   - **AFINNI, ART with affine leaves, ART with quadratic leaves:** each wrapped in the same propagation-margin loop `chebtt_roll` uses.
     Retrain at a tighter one-step tolerance until the chains along the training trajectories hold tol / 2, at most three rounds,
     reusing every cached solve. Without it, the comparison measures the missing margin, not the method.
   - **AFINNI and ART refuse outside their bounding box,** and AFINNI outside its trust regions, with the solver as fallback, so every
     method has a fallback and the maximum error is comparable.
2. **The same budgets, three ways.** All are reported, so no single axis decides.
   - **(a) Equal tolerance.** Two tolerances, 0.01 and 0.001, as the benchmark's grid. Each method's maximum error, solver calls,
     training time, refusal and speed-up at each.
   - **(b) Equal solver calls.** Solver calls are the data budget. Every call counts for every method: the recipe's trajectory steps,
     calibration labels and fallbacks; the box methods' oracle queries. Plot error against calls, with the two tolerances as two points.
   - **(c) Equal compute.** A fixed training wall time per problem (30 min on 4 cores), and who meets the maximum-error contract within
     it.
3. **The same protocol for everyone.** The study's suite, unchanged:
   - 40 test rollouts and chained mixing;
   - fresh closed columns (p17), sorted and unsorted at three exchanges;
   - 2,000-step long chains (p21), with start-up and steady refusal by cause and the certified horizon;
   - the speed-up against the fastest exact solver (p15), with evaluation timed at 1,000 cells per call;
   - model size and peak memory.
4. **Problems** (10, covering the classes):
   - small: bruss_d2, robertson, fisher_kpp_d3, lorenz96_d4;
   - medium: chemakzo, allen_cahn_d5, fisher_kpp_d5;
   - larger: fisher_kpp_d10, burgers_d20;
   - hard: pollu.

   HIRES, Nelson and primordial are added if time allows. AFINNI and ART at d >= 10 may not finish on CPU, and that is reported as a
   result, with the GPU follow-up marked.
5. **Gradients.** The primary comparison is value-only for every method, since none uses gradients today; this keeps it equal. A gradient
   arm is a phase 2 for all methods together, with each Jacobian charged at its real cost (a sensitivity solve, or d extra solves by
   finite differences). The recipe would take Jacobian labels (the tube's T3); AFINNI and ART would need a Jacobian input that the
   library does not yet expose; ChebTT's cross approximation cannot use them. Because support is unequal, a gradient arm would compare
   implementations more than methods, and it should be decided after phase 1.
6. **Records and rules.**
   - Every case is a record under `theory/panel/data/records/` (never the benchmark's store), with predictions committed before the run.
   - The methods are built through the benchmark's own builders (`src.benchmark.step_maps.METHODS`, `src.benchmark.methods`).
   - No `src` changes are needed, except possibly exposing a Jacobian input for phase 2.

## What it costs

| Item | Estimate |
|---|---|
| Code | a propagation-margin wrapper for AFINNI and ART (about 40 lines, modelled on `chebtt_roll`); an adapter so p17's columns and p21's long chains run on any model with `predict` and `refuses` (about 60 lines); solver-call accounting for the recipe; about 2 hours with smoke tests |
| Training, per problem and tolerance | the recipe 1-10 min; `chebtt_roll` 1-60 min (times out at d >= 20); AFINNI and ART: 3-70 s where they finished at 0.01, 90 s+ timeouts at d >= 5, so up to the 30 min cap with 4 MPI ranks and work stealing; at 0.001 several times longer |
| Cases | 10 problems x 2 tolerances x 5 methods = 100 training runs, plus the protocol (columns, long chains: 1-5 min each) |
| Wall time | serial about 15-20 hours; with three or four runs in parallel on this 16-core laptop, about 5-7 hours. Cut to the 0.01 tolerance first (about 3 hours), then 0.001 |

**Order:** the wrappers and adapter with smoke tests; tolerance 0.01 on all 10 problems; review; tolerance 0.001; phase 2 (gradients)
only if phase 1 leaves the ranking unclear.

## What it would tell us

At equal tolerance, solver calls and compute:
- which method meets the maximum-error contract in rollouts and under coupling, and at what speed-up against the fastest exact solver;
- where each method breaks: dimension, stiffness, coupling, long horizons;
- whether the local methods (AFINNI, ART) serve the problems the chart recipe cannot (Nelson, primordial, the oscillators' jumps), which
  is the question the round-5 council left open.
