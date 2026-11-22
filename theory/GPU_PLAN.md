# GPU measurement plan (before any GPU code)

The council's fifth round (`theory/reviews/council5-*.md`) answered whether the fourth recipe (`theory/panel/generic4.py`) is worth
running on a GPU. Its answer: probably, for large batches and large systems, but only measurements on a GPU machine decide, and the
refusal rate decides more than the evaluation does. This plan lists what to measure on a GPU machine (TACC, or a workstation with an
8 GB card) and what result would justify building a GPU path in the library. Every figure below is an estimate from operation counts;
none was measured here (this machine has no GPU).

## What one evaluation costs

| Quantity | Estimate | Source |
|---|---|---|
| Operations per cell | 36,000 (fisher_kpp_d10, 462 terms) to 66,000-120,000 (bruss_d20, 715 terms); the sketched gate is half to two thirds | Fable 3.1, Astra 3 |
| Memory per cell | the design row, p doubles: 5.7 kB at 715 terms, written once and read three times (memory-bound) | Fable 3.1 |
| Model per device | about 0.5 MB of float64 weights at 715 terms; replicated on every device | Astra 3 |
| Account state | one float64 per cell | Sol 6, Astra 6 |

## The cap the fallback sets

Per cell and step, `t = t_eval + rho t_fallback / C` with C fallback workers. With `t_eval` near 0.05 us on a data-centre GPU and an
exact step at 60-400 us (LSODA or VODE, `data/p15_denominators.json`), the fallback dominates above `rho` of about `C x (1e-4 to 8e-4)`:
above 0.01-0.08% with one core, above 1-5% with 64. A start-up refusal rate of 2-5% removes most of a GPU's gain, so the steady rate has
to be known to about 1e-4 (p21 measures it) and refused cells must be compacted on the device, solved in a batch, and scattered back.

## Measurements to make

1. **Throughput against batch size.** Cells per second and per-cell time at B = 1, 32, 128, 512, 2,048, 8,192, 32,768 (and 1e5-1e7 where
   memory allows, tiled), on a small fit (bruss_d2, 28 terms) and a large one (bruss_d20, 715 terms), resident on the device and
   host-fed. Predicted: launch overhead amortised from about 1e4 cells; 1e7-5e7 cells per second on a data-centre card in float64,
   3e6-1e7 on a workstation card with weak float64.
2. **Decisions independent of the batch.** The same cell must get the same accept or refuse decision whatever cells share its batch
   (otherwise a run changes with the domain decomposition). cuBLAS picks its algorithm by shape, so the products that feed the gate need
   a row-independent kernel with a fixed term order (the library's `det_matmul` contract, AGENTS.md). Measure decision flips and
   prediction differences between two batch sizes in float64: they must be zero.
3. **Precision.** Float64 for the gate as built. In float32, 95% of decisions on bruss_d20 (a rank-deficient fit) flipped from accept to
   refuse, and none on fisher_kpp_d10 (full rank) (Fable's check C, on the CPU). A float32 path needs a float32-conditioned fit (a
   singular-value cut near 1e-5 instead of 1e-10), selected and calibrated as its own estimator. Measure flips between float32 and float64
   on 1e6 column inputs per served problem.
4. **Comparators.** All at the same maximum-error contract and with the fallback included:
   - one CPU core (p15's numbers);
   - a whole CPU node running the same surrogate (the GPU is estimated to gain only 1-7x over it);
   - a batched stiff exact solver on the same GPU, which is the comparator that matters if the physics runs on a GPU. The library's
     `integrate_ensemble` is explicit Runge-Kutta, not stiff. Use an external batched stiff solver (SUNDIALS with batched linear
     solvers, or Julia's DiffEqGPU; to be checked), or a Rosenbrock kernel beside `src/kernels/ode.cu`. Estimated: the surrogate 3-15x
     faster than it at d <= 20, the gain growing with d (the solver's d^3 against the surrogate's p (m + outputs + 64)).
5. **Inference metrics beside the maximum error** (Sol 6, Astra 6):
   - the maximum error with its witness (cell, step, component);
   - start-up and steady refusal by cause;
   - split-step latency at p50, p95, p99 and the worst step;
   - cells per second under steady load;
   - fallback batch sizes and queue wait;
   - bytes per cell, peak device memory, transfers;
   - the account's state kept through compaction, migration and retries.

## Training on GPU

The first 2-10x in training is not a GPU: training time is dominated by exact-solver labels, and labelling with the fastest qualified
solver (LSODA or VODE, 9-24x faster than the library's step on most problems, p15) comes first. Then:
- **Dense work to the device**: the chart, design, truncated SVD and multi-output solve, in float64, keeping the same singular-value
  cut and comparing rank, predictions and gate factors.
- **On the CPU**: the four-coefficient linear program and NNLS of the gate.
- **Reproducibility**: fixed row IDs, splits, random streams, basis order, sketch matrices and SVD sign conventions, and reductions in an
  order independent of the batch, as ART's and AFINNI's GPU paths do.

## Several GPUs

- **Training: distribute independent work first.** Problems, folds, seed draws, calibration chains and exact-label batches parallelise
  without communication. Split one least-squares fit across devices only when it does not fit on one, and then by a tall-skinny QR with
  a fixed merge tree. Not by an all-reduce of Gram matrices: they square the condition number, and at the fit's cut of 1e-10 the
  relevant eigenvalues fall below float64 rounding, which changes the rank decision and the gate.
- **Inference: replicate the compact model.** Partition cells by the simulation's own ownership, keep each cell's account on its device,
  and use no communication between cells beyond the physics' halo exchange. This matches the main scaling advantage of data-parallel deep
  learning without its repeated gradient synchronisation.

## What would justify building the GPU path

Build it only if, on the intended hardware, the whole hybrid (evaluation, gate, account, compacted fallback) is at least 2x faster than a
batched stiff solver on the same GPU at the same maximum-error contract, on problems the recipe serves. If the batched solver is within 5x
of the surrogate's evaluation at d <= 20, the GPU path pays only for large systems (d above about 50), which the recipe does not yet serve.
A draft device evaluator (`theory/panel/gpu_eval.py`, NumPy or CuPy) exists for these measurements and is not yet reviewed or committed.
