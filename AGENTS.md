# AGENTS.md

Guidance for coding agents working in this repository (AFINNI: adaptive clustering
surrogates with CPU/GPU backends, MPI and work stealing).

## Environment

- Python environment: `.venv/` (managed by `uv`; see `Makefile` for install targets).
- Backend selection: `AFINNI_DEVICE=cpu|gpu|auto`, `AFINNI_MPI=off|on|auto`.
  Always set them explicitly in tests and benchmarks, so a missing GPU/MPI cannot
  produce a misleading pass.
- MPI launcher: `.venv/bin/mpiexec` (bundled MPICH); MPI tests use `AFINNI_MPI_NP` ranks.

## Verification

Run from the repository root:

```bash
# Unit/integration tests, CPU and GPU (both must pass)
AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"
AFINNI_DEVICE=gpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"

# Multi-rank tests (they launch their own mpiexec; never run pytest inside mpiexec)
AFINNI_DEVICE=cpu AFINNI_MPI=on AFINNI_MPI_NP=4 PATH=$PWD/.venv/bin:$PATH \
  .venv/bin/python -m pytest tests/mpi -o addopts='' -q

# Differential regression check against a reference run of another checkout
AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. .venv/bin/python scripts/regression_check.py \
  --save /tmp/new --compare /tmp/ref          # CPU refactors must be bit-identical
# on GPU add --rtol 1e-12 (decisions must still match exactly)
# the same check on 2 ranks covers the scatter of initial clusters and the merge of leaves
AFINNI_DEVICE=cpu AFINNI_MPI=on PATH=$PWD/.venv/bin:$PATH PYTHONPATH=. .venv/bin/mpiexec -n 2 \
  .venv/bin/python scripts/regression_check.py --save /tmp/new2 --compare /tmp/ref2
```

- A refactor must keep `scripts/regression_check.py` bit-identical on CPU (AFINNI and ART
  cases; on one rank and, for changes to scatter or merge, on two). Deliberate
  numerical changes must be listed and justified separately. Compare against references
  saved from the current baseline (`--save` on the base commit), not older ones: the
  custom k-means/kNN deliberately changed results relative to the original PR head.
- GPU training sums are deterministic and per segment (`segment_cross`, `_reduce_sum_1d`,
  `det_matmul`, `spd_solve` in `src/numerics/`): repeated GPU runs are bit-identical
  and ART trees do not depend on the batch size. Do not reintroduce `atomicAdd`
  float sums, CuPy's float `add.reduceat` (it subtracts prefix sums) or cuBLAS-batched
  products (algorithm chosen by batch count) on the training path. GPU results still
  differ from CPU results at rounding level: GPU tests compare with a tight tolerance
  (`tests/_helpers.py`: 1e-12, or 1e-10 with `separate_runs=True` when comparing two
  trainings), never with a loose one that would hide an algorithmic change. Leaf counts
  and evaluation counts must still match exactly.
- The GPU suite passes completely (AFINNI_ART_DESIGN.md Sections 42.9-42.10); treat any
  GPU failure as new.
- Profiling on GPU: `AFINNI_NVTX=1` emits one NVTX range per phase; profile with
  `nsys profile --trace=cuda,nvtx,osrt` (installed at `/usr/local/bin/nsys`).
  `py-spy`, `viztracer` and `kernprof` are in `.venv/bin`. ART's CUDA tests
  (`../Adaptive-Regression-Trees-ART-/tests/test_cuda_*.py`, run with this venv) need
  `c++` on PATH for their CPU reference checks, or they skip.
- CUDA kernels live in `src/kernels/*.cu` (one file per area) and are compiled at runtime by
  NVRTC through `src.kernels.module(name)` / `source(name)`; CuPy caches the binaries on disk,
  so there is no build step. Never add kernel source strings to `.py` files, and delete a
  kernel when its last caller goes. Format with
  `uvx --from clang-format==18.1.8 clang-format -i src/kernels/*.cu` (config in
  `src/kernels/.clang-format`); documentation rules in "CUDA documentation style" below. If
  the project ever gets a `[build-system]`, ship `src/kernels/*.cu` as package data.
- Large runs can exhaust host RAM (queues grow with rejected clusters x
  2 * min_pts_per_cluster); watch RSS when trying big N / small tolerances.

## Where things are

[README.md](README.md) has the project overview, installation, quick start, the documentation
index and the **code layout** of `src/` (which package owns what, and the import layers). Read
its "Code layout" section before adding a module; put new code in the package that owns the
concern, and import only from the same or a lower layer.

- Package `__init__` files re-export the names other modules import, so import from the package
  (`from src.art import ARTPlan`), not from the file a name happens to live in.
- Tests that monkeypatch a module-level name must patch the module that *uses* it (e.g.
  `src.art.split_search.IS_CUPY_BACKEND`, not `src.art`), or the patch silently misses.

Names (use them in code, docstrings, messages and docs):

- **AFINNI** (`algorithm="afinni"`, `src/afinni/`, `AFINNIPlan`) is the adaptive-clustering
  algorithm; the project and `AFINNISurrogate` share the name.
- **ART** (`algorithm="art"`, `src/art/`, `ARTPlan`, `ARTSettings`, `config.art`) is the
  adaptive regression tree. Do not call it "region mode": "region" is geometry only (a node's
  polytope is `NodePolytopes`, its rows `constraint_rows`), and "trust region" is AFINNI's.
  The external code `scripts/art_comparison.py` runs is "the reference ART"; ours is
  "AFINNI's ART" where the two must be told apart.
- **The refinement engine** (`src/engine/`) runs either plan: `driver.refine` (collective
  training), `refinement.refine_rank` (one rank's level-by-level loop), `work_stealing`.
  Plan-specific behaviour goes in the plan, behind `src/core/plan.py`, not in the engine.

## Code changes

- Before adding a helper, search for an existing one and reuse or extend it. Do not
  add small single-use functions, and do not duplicate logic that a module already
  owns (e.g. new components go through the registries in `src/components/registry.py`).
- When existing code gains new behaviour, rename the function, its docstring and
  related identifiers to describe what it now does, and update every caller, test
  and doc. Do not keep a name that only fits the old behaviour.
- Add or update tests with every behaviour change, so the reuse cannot quietly
  regress other callers.

Why: in a large codebase it is easy to miss the forest for the trees and add a
one-off function for each local problem. Reusing and extending shared modules,
with tests alongside, solves several use cases in fewer lines and makes the
codebase more robust and mature faster.

## Performance changes

Custom kernels, fused loops and other optimizations cost reading time, invariants a later
edit can break, and new failure modes; they must buy a measured improvement that matters.

- **Correctness first.** Code that exists to keep results independent of batch composition,
  run order or rank count (e.g. deterministic segmented sums instead of batched library
  calls) is kept regardless of speed. Never trade accuracy or those contracts for speed.
- **Speed or memory must be shown, not assumed.** A speed change must save, on a workload
  that matters (large, deep or high-dimensional trees) with baseline time `t`, at least

  `required(t) = max(2 s, min(5% of t, 10 s))`

  i.e. 2 s for runs under 40 s, 5% from 40 s to 200 s, and 10 s beyond (so 2% at 500 s, 1% at
  1000 s), or a 10x speedup of a short run (seconds to milliseconds). It must not slow other
  workloads by more than the same amount, nor raise peak memory unless a budget was agreed.
  A memory change must cut peak usage by at least 10%, or let a larger problem fit. Demand
  more from changes that add many lines or a fragile invariant; when the gain is marginal,
  prefer the library call or the simpler expression.
- **CPU loops:** prefer NumPy and BLAS; use Numba (already a dependency) only for fused loops
  NumPy cannot express without many temporaries, keep them serial unless one call carries a
  large amount of work (a parallel region's start-up cost made thousands of small calls slower),
  and set `error_model="numpy"` where IEEE infinities or NaNs are expected. Never replace a BLAS
  call with a hand-written loop without measuring: BLAS won every Gram size tested.
- **Measure properly:** interleaved A/B runs against a snapshot of the previous state (two
  or more rounds), the same power/charger state for both arms, and a bit-for-bit comparison
  of the complete model (leaf columns, statuses, counters, predictions) for changes claimed
  exact. Remove changes that do not meet the bar instead of keeping them behind a switch.

Why: what a user notices is the absolute time saved, which a fixed percentage misjudges at both
ends (10% of 3 s is invisible, 3% of 1000 s is 30 s). An optimization not measured on the
workloads that matter tends to optimize the wrong case while its complexity stays forever, and a
single noisy round misleads (AFINNI_ART_DESIGN.md Section 42.13).

## Code documentation style

Apply to all Python code you create, and to existing code you edit once you
understand it. Do not go hunting for old code to fix when a request needs no edits.

Why not Google-style docstrings: type hints and descriptive names already state
the parameters, types and return values. Restating them adds length without
information. Docstrings should carry the context the code cannot express.

- Every module starts with a concise docstring stating its purpose. Add usage
  examples or parameter details only when the code does not make them obvious.
- Every class, function and method gets a concise docstring. Prefer one useful
  sentence; write more only when the contract needs it.
- Exception: skip the docstring when the name and a short body already say
  everything, e.g. a property returning a field, `__len__`, an override that only
  implements a documented abstract method, a test named after the behaviour it
  checks, or a trivial package `__init__`. Once an edit makes such code non-trivial,
  add the docstring in the same edit.
- Document what the signature and implementation do not say: contracts, invariants,
  design reasons, side effects, ownership, threading, coordinate conventions,
  performance limits, failure behaviour.
- Do not paraphrase the name, parameters, return annotation or visible code.
- Comment only where a local decision would otherwise be easy to misread or break,
  and explain why, not what the next line does.
- Keep documentation next to the behaviour it governs, and update it when that
  behaviour changes.

```python
# Bad
def trust_region_shapes(...):
    """Compute the trust region shapes."""

# Good
def trust_region_shapes(...):
    """Scale each Q so the farthest training point lies on dx^T Q dx = 1; inference treats > 1 as outside."""


# Bad
class WorkStealer:
    """Handles work stealing between ranks."""

# Good
class WorkStealer:
    """Answer every steal request with exactly one grant or denial; an unanswered request deadlocks retirement."""


# Bad
order = xp.lexsort(xp.stack([scores, labels]))  # sort the scores

# Good: explain the decision, not the call
order = xp.lexsort(xp.stack([scores, labels]))  # keeps clusters contiguous so children can be cut by position
```


## CUDA documentation style

CUDA kernels are dense and their correctness often rests on choices the code cannot show
(which thread owns what, why an order is fixed, why a formula is not the obvious one). So
`.cu` files document more than Python does, but with the same rule: explain what the code
cannot say, never restate it.

- **File header:** one line naming the area and the Python callers.
- **Per kernel** (`__global__`): the thread mapping (what one thread or block owns) and the
  contract the caller relies on: ordering and determinism, in-place outputs, failure values
  (NaN, flags), size limits. Do not list parameters; the Python wrapper's docstring is the
  user-facing contract and shapes belong there.
- **Phases:** a short numbered comment before each major phase of a non-trivial kernel
  (load, accumulate, reduce, write), saying *why* it is done that way when that matters:
  memory traffic, bank conflicts, exactness, matching the NumPy path.
- **Invariants that must not be "optimized away"** get their own comment at the line:
  reassociation that would change bits, `_rn` intrinsics that forbid fused multiply-adds,
  NaN propagation that differs from `fmax`, counters that must match a NumPy layout.
- Keep comments next to the code they govern and update them with it.

```cuda
// Bad: restates the signature and the loop
// Computes the sum of A over rows r0..r1 for entry e and writes it to partial.

// Good: the mapping, the contract, and the reason for the order
// One thread per (chunk, entry). Chunks are counted from the segment's own start, never from
// the batch's, so a segment's bits do not depend on its neighbours.

// Bad
double e = exp(-fabs(s));  // compute exp

// Good: the decision, not the call
// Stable logistic: exp(-|s|) never overflows, and sigma(s), sigma(-s) come from the same e.
double e = exp(-fabs(s));

// Good: an invariant a reader could break
// ldexp: alpha * 2^-m exactly, the same bits as the CPU loop's repeated halving
double a = ldexp(alpha, -m);
```
