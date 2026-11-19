# The schemes

Global approximants of an expensive function, beside the library's adaptive surrogate (AFINNI and ART).
Each samples the oracle where its own error estimate asks and fits one model. The aim is the **worst
pointwise error** within a tolerance, with few oracle evaluations and fast inference.

The schemes live in `src/schemes` and are standalone. They import only the library's infrastructure
(`src`, `infra`, `numerics`, `kernels`, `core`), never AFINNI, ART or the refinement engine.
`tests/unit/test_layering.py` enforces this.

- The benchmark that compares them with AFINNI and ART is in [BENCHMARK.md](BENCHMARK.md), and the
  README's [Choosing a method](../README.md#choosing-a-method) summarises it.
- The experiments that led to them, what each taught and the literature behind them are in
  [SURROGATE_RESEARCH.md](SURROGATE_RESEARCH.md) and [SURROGATE_ITERATIONS.md](SURROGATE_ITERATIONS.md).

## The contract

Every scheme module has a frozen settings dataclass and a driver:

```python
model = scheme.build(f, lo, hi, atol, settings=None, *, jacobian=None, stream_dir=None, **overrides)
```

- **The oracle.** `f` takes host (N, d) arrays and returns (N,) values, or (N, m) for a vector
  function; it may also be an `Oracle` (`src/schemes/oracle.py`).
  - `jacobian` (N, m, d), or (N, d) for one output, is used where a scheme can use it: the hub
    expansion's directions and fit, and SlowTable's records. The other schemes ignore it.
  - The `Oracle` counts the points and Jacobians evaluated and the time spent in them. It does not
    deduplicate: a scheme that may ask for a point twice keeps its own store (ChebTT's lattice store,
    `PointCache` for vector trains).
- **Settings.** `overrides` replace single settings (`chebtt.build(f, lo, hi, 1e-3, seed=1)`), and a
  model's settings are frozen with it.
- **The model.** It predicts on host arrays, counts its training evaluations in `evals`, and carries a
  `TrainingReport` in `report`: evaluations, Jacobians, wall time per phase, oracle time and one
  record per round.
- **Under MPI.** `with oracle.distributed(f) as o:` gives rank 0 an oracle that cuts every batch into
  one even chunk per rank, while the other ranks serve those chunks until rank 0 leaves the block.
  - The values do not depend on the rank count.
  - A failure on any rank is re-raised on rank 0, so no rank is left waiting in a collective.
- **Saving.** `base.save(model, path)` writes any model to one `.npz` with a JSON header (format
  version, model, report), and `base.load(path)` rebuilds it, predicting bit for bit as trained.

### Profiling and logging

`src/schemes/profiling.py` times every stage under the same phase groups, so schemes compare stage by
stage:

| Phase | What it covers |
|---|---|
| `explore` | directions, edges, pilot gradients |
| `sample` | choosing points |
| `fit` | fitting the model |
| `check` | judging the model |
| `adapt` | deciding what to refine |
| `compress` | shrinking an accepted model |

The oracle's time is counted separately, and overlaps the phases that call it.

- **Rounds.** A round is a scheme's unit of refinement (a cross sweep, a tree level, a budget
  doubling). Each round ends with one INFO log line (`src.schemes.<scheme>`) and one record in the
  report.
- **Streaming.** With `stream_dir`, every phase and round is also streamed as JSON lines through
  `infra.profiling.Profiler`. That is the file format the monitor reads; a renderer for scheme runs
  is listed in TODO.md.
- **Cost.** Phases are coarse (per stage or round), so timing is always on.

## The schemes

### ChebTT (`chebtt/`, `ChebTTSettings`)

A Chebyshev interpolant in tensor-train form, built by TT-cross:

- **Sampling.** Maxvol fibres on the tensor grid of Chebyshev points, sweeping left to right and
  back. Sampling and inference are linear in d: a query costs `sum_k r_k (n_k + 1) r_{k+1}`.
- **Starting degrees.** Each axis's degree is read off random fibres.
- **Checks.** Each round is checked on a Sobol set, on free checks (grid points from earlier sweeps)
  and on a look-ahead along every axis through the worst point. Where an axis's top degrees still
  matter its degree doubles; otherwise the rank grows.
- **Piecewise axes.** An axis whose coefficients decay algebraically is searched for jumps and
  kinks, which become breakpoints: each piece gets its own Chebyshev basis.
- **Compression.** An accepted model is compressed by rank rounding and degree cuts, within a
  quarter of the tolerance.
- **Vector outputs.** `build_vector` trains one ChebTT per output on one shared point cache.
- **ChebTT-M.** `build_vector(..., rotate=True)` trains outputs that are ridges of shared oblique
  coordinates in the eigenvectors of their pullback metric (`VectorSettings`).

Limits: oblique or curved features have high rank at any degree, and the checks are estimates, not
bounds.

### ChebTT-A (`chebtt/adaptive.py`, `AdaptiveSettings`)

ChebTT with piecewise axes, plus one interface term where f jumps across an oblique or curved
interface or has a sharp front: `f ~ ChebTT(f - S) + S`.

- **The interface.** Its points are found on segments from the most extreme check point, and fitted
  by a plane or a diagonal quadric.
- **The term.** The jump is measured by one-sided fits along the normal. It becomes a profile
  `h(phi)` or jump fields of the foot point.
- **Smooth functions.** A function without such an interface ends after the first ChebTT attempt and
  pays nothing more.

Limits: only one interface is modelled, and it must be a level set of a plane or a diagonal quadric.

### ChebTree (`chebtree.py`, `ChebTreeSettings`; `build_rotated`, `RotatedChebTreeSettings`)

An adaptive tree of tensor Chebyshev boxes.

- **Acceptance.** A box is accepted when per-axis decay fits of its coefficients bound the tail
  within the tolerance, its check points agree, and fibres through its centre (the next nested
  grid's points) confirm it.
- **Refinement.** A failing axis with exponential decay raises its degree; one without it splits the
  box.
- **The depth cap.** Boxes accepted only because they reached it refuse their queries (`refuses`), so
  the caller answers them with the oracle.
- **Rotated.** `build_rotated` grows the tree in active-subspace coordinates.

Limits: tensor grids cost `prod(n_k + 1)` points per box, so ChebTree suits 2-3-D.

### SparseCheb (`sparsegrid.py`, `SparseGridSettings`)

A dimension-adaptive sparse Chebyshev grid (Smolyak, Gerstner-Griebel). It refines the active index
with the largest surplus. It stops when the surpluses and a Sobol-plus-corners check are within half
the tolerance; a failed check refines every active index once (a look-ahead).

Limits: the surpluses can stop early where an interaction is never opened, and kinks and jumps need
one global expansion.

### The hub expansion (`hub.py`, `HubSettings`)

Per group of outputs: a dense Chebyshev model (the hub) in the few directions the outputs are steep
in, the leading eigenvectors of their gradient covariance. Every other direction gets degree-1 and
degree-2 terms, whose coefficients are polynomials of the hub. The model is fitted by least squares.

- **Choices from the data.** The hub dimension, the output groups, the samples (pivoted QR plus
  vertex-directed points), the degree (by the hold-out's worst error) and the budget, which doubles
  until the hold-out is within half the tolerance or stops improving.
- **With a Jacobian.** It replaces the finite-difference pilot, and every sample's gradient enters
  the fit.

Limits: functions outside that class (many differently oriented features, jumps), and the hold-out is
an estimate, not a bound.

### MetricKernel (`kernel.py`, `KernelSettings`)

Greedy Matern-5/2 kernel interpolation in a metric learned from the model's own gradients (the square
root of their average outer product), with Sobol check batches.

Limits: a dense factorisation per round, and random checks rarely land on the worst point in high
dimension.

### SlowTable (`tabulation.py`, `TableSettings`)

In-situ tabulation of a step map over the states a simulation visits, following ISAT's design. Each
record is linear (a state, its increment and its Jacobian) inside an ellipsoid of accuracy. A query
inside an ellipsoid is retrieved, a share of retrievals is checked, and an uncovered query grows the
nearest ellipsoid or becomes a new record. The record's Jacobian comes from the oracle when it has
one, otherwise from forward differences.

Limits: a query outside every ellipsoid is refused; the ellipsoids are estimates.

## Files

| File | Contents |
|---|---|
| `base.py` | `Settings`, the `Model` protocol, the check points, `save` and `load` |
| `oracle.py` | `Oracle`, `PointCache`, `DistributedOracle`, `serve`, `distributed` |
| `profiling.py` | `Training` (phases, rounds, the stream) and `TrainingReport` |
| `chebyshev.py` | Chebyshev points, bases (with derivatives), coefficients, tails and decay fits |
| `directions.py` | gradients (Jacobian or finite differences), gradient covariance, output groups, the active subspace, the pullback-metric groups |
| `chebtt/model.py`, `chebtt/train.py`, `chebtt/adaptive.py` | ChebTT and VectorTT, their training, ChebTT-A |
| `chebtree.py`, `sparsegrid.py`, `hub.py`, `kernel.py`, `tabulation.py` | one scheme each: settings, model, stages, driver |

## Verification

`scripts/scheme_regression_check.py` trains every scheme on one small case and saves, or compares
against a reference, its evaluations, predictions and refusals:

```bash
SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python scripts/scheme_regression_check.py --save /tmp/new --compare /tmp/ref
```

- The whole check takes a few seconds.
- A refactor must reproduce every case bit for bit; a deliberate numerical change is listed and
  justified separately.
- The script pins BLAS to one thread, because the hub expansion's Gram products round differently
  with the thread count.
