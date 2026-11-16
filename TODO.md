# TODO

Open work, grouped by area. The README's "Roadmap" lists the user-facing milestones; this
file is the working list behind them. Remove an item in the change that completes it.

## Naming: code against the technical companion

The companion's symbols map to code names in `docs/ARCHITECTURE.md` ("Notation: from the
companion to the code"). These code names still differ from the companion's terms; each
rename below breaks something, which is why it is listed rather than done.

- [ ] Rename `ARTSettings.soft_temperature` to `soft_gate_width` (`h` in the companion).
  Breaks user configs and the keys of saved config JSON. Do it with the packaging work below.
- [ ] Rename `AFINNISurrogate.filter_unguaranteed_local` to a name that says what it
  returns, e.g. `outside_trust_regions_local`: a trust region is not a guarantee. Used by
  `p_out_iterative_experiment.py` and its test. Do it with the packaging work below.
- [ ] Rename `w`, `z`, `T` in the split search (`src/art/split_search.py`,
  `ARTPlan._soft_refine`) to `normal`, `offset`, `gate_width`. This touches GPU-only lines
  (the `soft_step` call, `_soft_objective_fused`) and kernel-argument lists, so run the GPU
  suite and the GPU regression check when doing it.
- [ ] Decide whether to rename the `AFINNISurrogate` leaf attributes (`X_src`, `f_src`,
  `N_src`, `A_fits_src`, `b_fits_src`, `E_cluster_*_src`, `counts_src`, `J_src`,
  `Hs_src`), e.g. to `centres`, `n_leaves` (as `trust_matrices_src` already is). They are also the keys of saved
  model files, so `load_model` would need a key map for old files. Current recommendation:
  leave them.
- [ ] Decide whether to rename the "covariance" names that hold an unnormalized scatter
  matrix (`CovZ`, `symmetric_point_covariance`, `run_fused_cross_cov`, ...). They are
  documented as scatter matrices; a rename would be broad. Current recommendation: leave
  them.

## Packaging, CI and releases

Modelled on `knndbscan` (its `pyproject.toml`, `.github/workflows/`, `CHANGELOG.md`,
`docs/RELEASING.md`). AFINNI is pure Python, so one `py3-none-any` wheel serves every OS
and Python.

- [ ] Make it a package: move `src/` to `src/afinni/` and rename every `from src...`
  import; add a `[build-system]` (`uv_build`); ship `src/kernels/*.cu` as package data;
  read `__version__` from the installed metadata; exclude `archive/`, `review_artifacts/`,
  `research/` and `scratch/` from the source archive. `scripts/regression_check.py` must
  keep running on older commits (import from either name).
- [ ] Choose a licence (PyPI needs one).
- [ ] Dependency policy: lower bounds only, no upper caps; floors found by testing with
  `--resolution lowest-direct`, not copied from the lock.
- [ ] Slim the core dependencies: `jax` and `diffrax` (one problem module and
  `afinni_launcher.py`) to an extra; `matplotlib` and `pandas` (`src/plotting/`) to a
  `plot` extra; drop `seaborn` (not imported anywhere in `src/`).
- [ ] Widen `requires-python` to `>=3.10` (tested: 3.10 to 3.15 pass the CPU and MPI suites
  with bit-identical regression results). Needs `class Registry[T]` in
  `src/components/registry.py` replaced by `Generic[T]`, and the floors `jax>=0.6.2`,
  `scipy>=1.15.3`.
- [ ] CI (`.github/workflows/ci.yml`): lint and `uv lock --check`; CPU tests on Linux
  x86_64 and arm64, macOS and Windows; MPI tests with the bundled MPICH; the differential
  regression check (base against PR, 1 and 2 ranks); the built wheel on every supported
  Python with the oldest and newest allowed dependencies. The GPU suite stays a manual run.
- [ ] Releases: `CHANGELOG.md` (Keep a Changelog, with a heading for changes that alter
  results), `docs/RELEASING.md`, a tag-driven workflow publishing to TestPyPI and then PyPI
  by trusted publishing, and GitHub releases from the changelog.
- [ ] Document that trained models are bit-for-bit reproducible only within one set of
  dependency versions (the lock): a NumPy upgrade can change a large tree.

## Python 3.14, 3.15 and NumPy 2.5

The project is developed and tested on Python 3.13 (`.python-version`).

- [ ] Support Python 3.14: the lock pins NumPy 2.2.6, which has no 3.14 wheels, and gives
  Linux 3.14 the pre-release Numba 0.63.0b1. `uv lock -P numpy==2.4.6 -P numba -P llvmlite
  -P cuml-cu12` fixes both; with it the CPU and MPI suites pass on 3.13 and 3.14 and the
  regression check stays bit-identical on 1 and 2 ranks.
- [ ] Decide whether to adopt NumPy 2.5 everywhere. It is the first NumPy with Python 3.15
  wheels, and it changes the large benchmark `vector_quadratic_d6` (173,084 to 173,946
  leaves, relative L2 0.012240 to 0.012265, L-infinity unchanged) while all 52 regression
  cases stay bit-identical. Adopting it is a deliberate numerical change: record it, and
  save the regression references and benchmark baselines again.
- [ ] Unblock Python 3.15 in the development lock: add `python_full_version < '3.15'` to
  `cuml-cu12` in the `research` group (cuML caps Numba below 0.65, which has no 3.15 wheels).
- [ ] The `gpu` extra on Python 3.15 waits for CuPy to publish 3.15 wheels.
- [ ] Add one larger case to `scripts/regression_check.py`: the 52 small cases did not
  detect the NumPy 2.5 change that the large benchmark showed.

## knndbscan

- [ ] Publish `knndbscan` to PyPI (it is on TestPyPI only), then drop the TestPyPI index
  from `pyproject.toml`. It is a core dependency, so this blocks publishing `afinni`.
- [ ] Make kNN-DBSCAN's `eps` a setting: `src/engine/initial.py` fixes it at 0.5, which joins
  uniformly spread points into one cluster.

## Algorithms

- [ ] The Hessian estimate's level set (`n_est_cov = max(1, min_pts_per_cluster // 3)` in
  `CovarianceHessianSplit.metric`) has no floor in `d`, unlike the gradient points
  (`max(2d, min_pts_per_cluster // 4)`). For `d > min_pts_per_cluster // 3` (33 with the
  default 100) the level-set scatter is singular, and its eigenvalues floored at 1e-12 give
  the unexplored directions the largest split weight. Decide on a floor (e.g. `max(2d, ...)`)
  or a pseudo-inverse that gives those directions no weight; either changes results, so it
  is a deliberate numerical change. The companion's pitfall in section 4.1 describes the
  current behaviour.
- [ ] The affine fit's ridge, `max(1e-8 tr(Cov), tiny_eps)` added to each cluster's covariance
  (`fused_affine_fit_regularize`, `src/kernels/small_linalg.cu`, and its NumPy twin), biases
  the fit by an amount that grows with `d`: an exactly linear function fits with a largest
  error of about 4e-7 at d = 10 and 3e-6 at d = 70 (unit-cube domain, 710 points), not about
  1e-15. With `atol` below about 1e-5 at high `d`, nodes are rejected that a plain
  least-squares fit would accept. Decide whether the ridge should scale with `atol`, shrink
  with the point count, or apply only when the covariance is near singular (a deliberate
  numerical change). `tests/integration/test_art_training.py` (70-D test) uses `atol=1e-4`
  because of it.

## Documentation that disagrees with the code

- [x] `docs/USAGE.md` says runs are reproducible "for a fixed `seed` and rank count". True
  for ART; false for AFINNI with work stealing on (the default), whose per-batch random
  streams depend on which clusters share a batch.
- [x] `q_reg` (comment in `src/config.py`, `docs/USAGE.md`) is described as a floor on the
  covariance eigenvalues. It floors the eigenvalues of the unnormalized scatter matrix,
  i.e. `q_reg / n` on the covariance.
- [x] The `max_levels` comment in `src/config.py` says "maximum number of refinement
  levels": depths 0 to `max_levels` are fitted, one more than it says.
- [x] `SCATTER_TAG` comment in `src/engine/driver.py` says the payload uses tags `+1..4`;
  the queue columns use up to `+8` (`QUEUE_COLUMNS`).
- [x] `docs/ARCHITECTURE.md` and `docs/USAGE.md` state that an ART tree does not depend on
  batch size, rank count or steals, without the exception: `sampling="convex_combination"`
  seeds a per-batch stream.
- [x] The `WorkStealer` docstring and its idle log mention a "level limit" that
  `refine_rank` never uses.

- [ ] Two unrelated functions are named `segment_sum`: `src/numerics/segments.py`
  (values, labels, K) and `src/art/split_search.py` (a, starts). The second is on the
  CuPy path, so a rename needs the GPU suite.

## Verification still to run

Settled on an RTX 2000 Ada laptop GPU (CuPy 14.2, CUDA 12.9 runtime) at `0b91ae2`: the GPU suite
passes except the two kNN-DBSCAN tests, whose GPU path in the `knndbscan` package needs cuML,
which the `gpu` extra does not install (AFINNI's own k-means and kNN do not use it).
`regression_check.py` on the GPU is bit-identical to `c8a6ff6` (its successor `b67c1a8`, which
only adds the ART cases, with the case and array names mapped to the current ones) on 1 and 2
ranks, and repeated GPU runs are bit-identical. `test_walks_above_256_dimensions_do_not_depend_on_the_chunk_size`
runs the chunked CuPy walk above 256 dimensions.

- [x] On the GPU machine: the GPU suite and the GPU regression check against the current
  tree. The naming pass changed code paths shared by CPU and GPU (`src/components/splitting.py`,
  `src/art/plan.py`, `src/art/routing.py`, `src/surrogate/model.py`); the CPU checks passed
  bit-identical. The trust-matrix rename (`Q` to `trust_matrix`) changed shared code in
  `src/components/local_models.py`, `src/art/plan.py`, `src/afinni/plan.py`,
  `src/surrogate/model.py`, `src/surrogate/persistence.py`, `src/infra/profiling.py` (the
  phase key, which also looks up the NVTX range) and `src/plotting/surrogate.py`, and only docstrings in
  `src/components/trust_regions.py`; no kernel, CuPy-only branch or kernel-argument list.
- [x] GPU suite after the grid-numbering change (`grid_strides`, `grid_cell_coords` in
  `src/numerics/grid.py`, used by `Roots.bounds`, `Roots.root_of` and the grid initial
  clustering, so ART and the grid clustering run beyond 64 dimensions). The helpers run on the
  host and only their int64 results reach the device; CPU 1- and 2-rank regression checks are
  bit-identical.
- [x] A 2-rank GPU regression comparison against the last GPU-verified commit (CPU 2-rank is
  bit-identical).
- [x] GPU suite and GPU regression check after the batch-layout vocabulary pass (`sizes` to
  `counts` in `ClusterBatch`, `QUEUE_COLUMNS`, `DeviceBatch`, `ClusterFitData`, the acceptance
  criteria and `child_counts`; `owner` to `labels` in `HitAndRunSampler`; `k` to `K` in
  `ARTPlan._split`, `ARTPlan._bins_search` and `SplitRows.cut_sums`). CPU 1- and 2-rank checks
  are bit-identical. CuPy-only lines changed, untested on a GPU:
  `src/components/local_models.py` `AffineLSQModel.fit` (the two `segment_cross(...,
  data.counts)` calls); `src/art/sampling.py` `HitAndRunSampler._walk` (the `d > 256` chunked
  branch, `labels[sel]`) and `_walk_fused` (the `hr_walk` launch, `labels.astype(xp.int64)`
  as the kernel's `owner`); `src/art/split_search.py` `SplitRows.cut_sums` (the CuPy branch's
  `K` in `edges`, the two `reshape(K, ...)`, `C.reshape(K, m + 1)`, the `L, R` buffers and the
  `bin_sides` launch). Left on purpose in GPU-only code: the local `k` (node count) in
  `soft_step` and `_soft_objective_fused` (`src/art/split_search.py`, with their `(k, ...)`
  shape comments), and every kernel parameter in `src/kernels/*.cu` (`hr_walk`'s `owner`,
  documented there as Python's `labels`).

## Technical companion (`docs/report/`)

- [ ] The manuscript's Hessian-estimate theorem (main text and appendix) defines the
  scatter matrix as a sum and bounds its inverse against `C H`; the bound holds for the
  mean (a missing `1/n`). Fix it in the manuscript; the companion states the mean.
- [x] Section 6 says the whole hit-and-run walk runs in one kernel; with the drift check it
  is at least two launches of `hr_walk`.
