# ART parity on Vista (GH200): this library's ART vs the reference ART

## Hand-off: applying these changes (read this first)

The appendix "The patch" at the end of this file is the complete `git diff` of the working tree
against `refinement-engine` at `1f273bc` (12 files, +500 / -77). Nothing here was committed,
staged or stashed. To apply it on a clean checkout of that commit, save the appendix's diff block
to a file and run `git apply <file>` (it applies cleanly to `1f273bc`; checked with
`git apply --check -R` against this working tree). This file (`ART_PARITY.md`) and
`REFERENCE_ART_ENV.md` (how the reference ART was compiled on Vista) are untracked notes, not
part of the patch.

### What the patch contains, by change (see "Changes to the library" for measurements)

| change | files | kind |
|---|---|---|
| Comparison harness: both sides warm up to depth 1 and time only the build; the reference's `--max-iters` drives `soft_iters`; the benchmark oracle on the GPU through the reference's `CudaOracleWorkspace`; `--library-affine-search-from`, `--library-host-oracle` | `scripts/art_comparison.py`, `docs/USAGE.md`, `studies/sanity/art/art_cases.py` | harness only |
| 1. Column-parallel `spd_solve_factored`; one factorization in `lstsq_auto` | `src/kernels/small_linalg.cu`, `src/numerics/linalg.py` | exact (bit-identical) |
| 3. Right-looking Cholesky in shared memory (`chol_packed_block`, `spd_factor_shared`) | same | exact |
| 4. Register-tiled Gram kernel `segment_cross_tiled2d` for P, Q >= 32 | `src/kernels/segment_cross.cu`, `src/numerics/segments.py` | exact |
| 5. Shared-memory solves with a parallel back substitution (`chol_packed_solve_block`, `spd_solve_shared`, `spd_solve_factored_shared`); `exact=True` keeps leaf fits on the exact kernel | `src/kernels/small_linalg.cu`, `src/numerics/linalg.py`, `src/components/local_models.py` | relaxed (deterministic, batch-independent, different rounding; only the split search's internal solves) |
| 6. `regression_check.py` cases `art_quadratic_d8`, `afinni_quadratic_d8` | `scripts/regression_check.py` | test coverage |
| 7. `ARTSettings.affine_search_from = 100`: affine fits in the bins and soft searches from 100 leaf coefficients | `src/art/plan.py`, `src/config.py`, `tests/unit/test_art.py` | algorithm (results change only for leaves with >= 100 coefficients) |

Suggested commits, in this order: the harness; the exact kernels (1, 3, 4); the regression
cases (6); the relaxed solves (5); the affine search (7). Changes 5 and 7 are deliberate
numerical changes and need to be listed and justified in their commit messages (AGENTS.md).

### Verified on the final working tree (Vista, GH200, CuPy 14.2, CUDA 12.9)

- `SURROGATE_DEVICE=cpu SURROGATE_MPI=off pytest -m "not mpi"`: 657 passed, 71 skipped.
- `SURROGATE_DEVICE=gpu SURROGATE_MPI=off pytest -m "not mpi"`: 695 passed, 33 skipped.
- `scripts/regression_check.py` against a baseline saved from `1f273bc` (with the two d = 8
  cases): CPU bit-identical, 54 cases, 904 array pairs, PASS. GPU at `--rtol 1e-12`: **one
  array fails**, `art_quadratic_d8|param:coef` (7.7e-13 absolute on coefficients up to 3.6;
  1.7e-11 relative on a near-zero element); every counter, status, mask and routing code is
  identical. The cause is change 5 (split planes move by about 1e-14). Adopting change 5 means
  stating that tolerance (e.g. 1e-10, the GPU tests' `separate_runs` tolerance) or a
  scale-relative comparison for that case.
- Repeated GPU runs bit-identical (Rosenbrock quadratic 10-D and 20-D models).
- The exact changes bit-identical to `1f273bc` on four complete benchmark models
  (`scratch/scripts/compare_models.py`) and in unit checks of each kernel; AFINNI at d = 8
  bit-identical to `1f273bc`.
- The reference ART's own CUDA tests (439 passed, 9 skipped) were run only to validate the
  reference's build environment, not this patch.
- After the last test-suite run, only `scripts/art_comparison.py` changed (GPU oracle, its
  docstring); it lints and compiles, and the final matrix's 96 GPU-oracle reruns used it.
- Not run: the MPI tests (`tests/mpi`) and the two-rank `regression_check.py`. The patch does not
  touch scatter, merge or work stealing, but AGENTS.md lists them; run them before pushing.

### Still to do when committing (not in the patch)

- **Technical companion:** `docs/report/adaptive-surrogates.typ` (appendix around its lines
  3357-3367) names `segment_cross_tiled` and `spd_factor_coop` as the large-system kernels;
  update it for `segment_cross_tiled2d`, `spd_factor_shared` and the shared-memory solves, describe
  `affine_search_from` where the bins and soft searches are explained, and rebuild the figures and
  the PDF (`make report-figures`, `make report`), as AGENTS.md requires for a changed default or stage.
- `docs/USAGE.md` line 250 still says the soft split search is "1-30x faster than the reference's
  GPU configuration"; that predates the reference's current GPU path and should be re-measured or
  removed.
- `docs/ARCHITECTURE.md`: add `affine_search_from` to the settings table if it lists ART settings.
- Format the kernels: `uvx --from clang-format==18.1.8 clang-format -i src/kernels/*.cu`.
- Coverage: the new CuPy launchers run only in the GPU suite (`make coverage-gpu`).

## Setup

- One GH200, modules `gcc/15.1.0` + `cuda/12.9`, this repository's `.venv` (CuPy 14.2.0, CUDA 12.9 wheels).
- Reference ART: compiled through `scratch/reference_env/` (see `REFERENCE_ART_ENV.md` at the
  repository root). Its CUDA tests: 439 passed, 9 skipped.
- Separate CuPy caches: `scratch/cupy-cache` (library), `scratch/cupy-cache-ref` (reference).

## First measurements (Rosenbrock 10-D, offset 1, affine leaves, depth 10, multiplier 10, 20 optimizer iterations)

| side | timed region | build s | leaves | queries | test rel. L2 | q99 | max pw rel |
|---|---|---|---|---|---|---|---|
| reference | `build_tree_benchmark` incl. depth-1 warm-up (art_comparison) | 1.05 | 1024 | 112,640 | 0.250 | 1.50 | 9.17 |
| reference | build only, after warm-up (its own `build_seconds`), 2 runs | 0.674 / 0.660 | 1024 | 112,640 | | | |
| reference | its warm-up alone | 0.63 / 0.48 | | | | | |
| library | `model.train`, no in-process warm-up, warm disk cache, 2 runs | 1.034 / 1.033 | 1024 | 114,687 | 0.292 | 1.71 | 5.71 |
| library | the same, cold CuPy disk cache | 88.7 | | | | | |

Observations:

- On the GH200 the library is not 7x slower here: 1.03 s against the reference's 0.66 s build
  (1.05 s with its warm-up). The laptop's 2 s was mostly launch and synchronisation overhead on
  a slower host.
- The comparison is not yet like for like: the reference's build time excludes its first-use
  costs (module loads, workspace allocation), while the library's 1.03 s includes them. Next:
  time the library after an in-process depth-1 warm-up as well.
- A cold kernel cache costs the library about 88 s (NVRTC compiles every kernel). This matters
  for first runs on a new machine, not for timing.

## Why the lab member saw ~1 s (reference) against up to ~7 s (library)

His report (2026-11-19): Rosenbrock, Gaussian and plane wave, depth 10, tolerance 1e-2 relative
L_inf; the reference stays about 1 s from 3-D to 10-D, the library grows to about 7 s.

Short answer: **not only the timing region.** The timed regions differed, but that is worth
about 0.5 s per run, in both directions. The main cause is most likely a **settings mismatch
that the comparison script did not catch**: the library ran 200 soft-search iterations against
the reference's 20.

### 1. Iteration cap: 200 (library) vs 20 (reference). Most likely the main cause.

Timeline, from both repositories' history:

| date | repository | change |
|---|---|---|
| 2026-11-16 | library `0b5531c` | `ARTSettings.soft_iters` 40 -> 200, "the reference ART's `--max-iters`" (its preset then: `max_iters=200`, `sample_multiplier=50`) |
| 2026-11-18 | reference `e082f05` | presets `max_iters` 200 -> 20, `sample_multiplier` 50 -> 10, GPU sampling and sample cache |
| 2026-11-19 | his comparison | runs after the reference's change |

`scripts/art_comparison.py` gives the library the reference's `--sample-multiplier` (so both
used 10 by default), but it did **not** pass the reference's `--max-iters`. Without
`--library-soft-iters 20` the library kept `ARTSettings.soft_iters=200`, 10x the reference's
optimizer iterations, and the soft search is the largest share of the library's build time (66%
of it at 20 iterations in the laptop profile). Whether he passed `--library-soft-iters` is not
recorded. How much of the 200 iterations actually runs (nodes stop early when converged or
stalled) is measured below with the `soft_iteration_limit` and other stop counters.

Laptop numbers from before (RTX 2000 Ada, Rosenbrock, affine, depth 10), for scale:
multiplier 50 + 200 iterations gave 3.90 / 5.48 / 11.41 s at d = 3 / 10 / 20; multiplier 10 + 20
iterations gave 2.07 / 1.93 / 2.41 s. Multiplier 10 with 200 iterations, the likely mismatch,
was not measured on the laptop. On Vista it makes the library 1.9-4.9x slower than the reference
(see "Iteration cap on Vista" below).

Fix (in the library's comparison script): `art_comparison.py` now passes the reference's
`--max-iters` as the library's `soft_iters` unless `--library-soft-iters` is given, so the two
iteration caps cannot drift apart again. The sanity study's `library_default` group passes
`ARTSettings().soft_iters` explicitly to keep its meaning.

### 2. Timed region: a few tenths of a second, in both directions

The script timed the reference's whole `build_tree_benchmark` call, depth-1 warm-up included,
and the library's `train` with no warm-up. So each side's time carried a different first-use
cost. On Vista, Rosenbrock 10-D:

- Reference warm-up 0.48-0.63 s. The script's time included it: 1.05 s against 0.66 s for the build alone.
- Library: 1.03 s without a warm-up, 0.63 s after one. Its first-use cost is about 0.4 s.

That is the same order on both sides, so it cannot turn ~1 s into ~7 s. If he took the
reference's time from its own harness (build only) and the library's from the script (no
warm-up), the library was charged about 0.4 s extra. Fix: both sides now warm up to depth 1 and
time only the build (`warmup_seconds` reported separately).

### 3. Other possible contributors (not verified)

- A cold CuPy kernel cache (a fresh `CUPY_CACHE_DIR`, e.g. a new job's node-local temp) makes the
  library compile every kernel on first use: 88.7 s here. This is far more than 7 s on the GH200,
  so it was probably not his case, unless the cache was only partly warm.
- Hardware: his machine for the depth-10 runs is not recorded. On the laptop at matched settings
  the library took about 2 s; on the GH200 0.63 s (after warm-up).

### Matched result on Vista (both warmed up, 20 iterations, multiplier 10)

Rosenbrock 10-D: library 0.63 s, reference 0.66-0.67 s, 1,024 leaves each, 114,687 vs 112,640
queries. The full matrix (3 benchmarks x d = 3, 5, 10, 20, affine and quadratic, two interleaved
rounds) is in "Timing matrix" below.

### Iteration cap on Vista: both codes at 20 and at 200 iterations

Multiplier 10, affine leaves, depth 10, offset 1, both sides warmed up, the reference on its GPU
BFS path; each cell gives two rounds, and "library - reference" uses the faster round of each.
Sources: the library at 20 and 200 iterations, interleaved (`scratch/results/iter_cap.jsonl`); the
reference at 20 iterations from the timing matrix (`timing_matrix.jsonl`); the reference at 200
iterations, run separately (`ref_iter200.jsonl`). So the reference's 20 and 200 runs were not
interleaved with each other, but each has two rounds and the spread within a cell is about 3% at most.

| benchmark | d | iters | reference build s | library build s | library - reference s | reference leaves / queries | library leaves / queries | reference test L2 | library test L2 |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | 3 | 20 | 0.20 / 0.21 | 0.45 / 0.46 | +0.25 | 28 / 1,120 | 37 / 1,553 | 0.00463 | 0.00434 |
| gaussian | 3 | 200 | 0.34 / 0.33 | 1.01 / 1.01 | +0.68 | 28 / 1,120 | 34 / 1,427 | 0.00428 | 0.00424 |
| gaussian | 5 | 20 | 0.41 / 0.40 | 0.61 / 0.63 | +0.21 | 347 / 20,820 | 394 / 24,427 | 0.00457 | 0.00463 |
| gaussian | 5 | 200 | 0.99 / 0.95 | 1.72 / 1.67 | +0.71 | 367 / 22,020 | 335 / 20,769 | 0.00449 | 0.0047 |
| gaussian | 10 | 20 | 0.88 / 0.88 | 0.64 / 0.65 | -0.24 | 1020 / 112,200 | 1016 / 113,791 | 0.0115 | 0.0149 |
| gaussian | 10 | 200 | 1.83 / 1.77 | 1.80 / 1.80 | +0.03 | 1020 / 112,200 | 1010 / 113,119 | 0.0111 | 0.013 |
| plane_wave | 3 | 20 | 0.76 / 0.77 | 0.64 / 0.65 | -0.12 | 1022 / 40,880 | 967 / 40,613 | 0.245 | 0.184 |
| plane_wave | 3 | 200 | 4.76 / 4.72 | 1.65 / 1.66 | -3.07 | 1019 / 40,760 | 878 / 36,875 | 0.148 | 0.273 |
| plane_wave | 5 | 20 | 0.90 / 0.93 | 0.64 / 0.65 | -0.25 | 1024 / 61,440 | 1024 / 63,487 | 0.539 | 0.516 |
| plane_wave | 5 | 200 | 5.76 / 5.62 | 1.68 / 1.70 | -3.94 | 1024 / 61,440 | 1024 / 63,487 | 0.55 | 0.488 |
| plane_wave | 10 | 20 | 0.61 / 0.61 | 0.62 / 0.62 | +0.01 | 1024 / 112,640 | 1024 / 114,687 | 0.62 | 0.634 |
| plane_wave | 10 | 200 | 3.09 / 3.19 | 1.77 / 1.76 | -1.33 | 1024 / 112,640 | 1024 / 114,687 | 0.62 | 0.632 |
| rosenbrock | 3 | 20 | 0.57 / 0.59 | 0.63 / 0.63 | +0.06 | 1005 / 40,200 | 755 / 31,709 | 0.0207 | 0.026 |
| rosenbrock | 3 | 200 | 2.82 / 2.85 | 1.61 / 1.59 | -1.22 | 991 / 39,640 | 759 / 31,877 | 0.0169 | 0.0233 |
| rosenbrock | 5 | 20 | 0.79 / 0.78 | 0.65 / 0.64 | -0.14 | 1024 / 61,440 | 1024 / 63,487 | 0.0995 | 0.0967 |
| rosenbrock | 5 | 200 | 4.48 / 4.36 | 1.72 / 1.74 | -2.64 | 1024 / 61,440 | 1023 / 63,425 | 0.0993 | 0.0989 |
| rosenbrock | 10 | 20 | 0.74 / 0.73 | 0.63 / 0.64 | -0.10 | 1024 / 112,640 | 1024 / 114,687 | 0.25 | 0.292 |
| rosenbrock | 10 | 200 | 5.38 / 5.38 | 1.79 / 1.80 | -3.59 | 1024 / 112,640 | 1024 / 114,687 | 0.253 | 0.291 |

The configuration he most likely ran (reference at 20 iterations, the library at its default 200):

| benchmark | d | reference, 20 iters | library, 200 iters | library / reference |
|---|---|---|---|---|
| gaussian | 3 | 0.20 | 1.01 | 4.9x |
| gaussian | 5 | 0.40 | 1.67 | 4.2x |
| gaussian | 10 | 0.88 | 1.80 | 2.1x |
| plane_wave | 3 | 0.76 | 1.65 | 2.2x |
| plane_wave | 5 | 0.90 | 1.68 | 1.9x |
| plane_wave | 10 | 0.61 | 1.76 | 2.9x |
| rosenbrock | 3 | 0.57 | 1.59 | 2.8x |
| rosenbrock | 5 | 0.78 | 1.72 | 2.2x |
| rosenbrock | 10 | 0.73 | 1.79 | 2.4x |

Results:

- **At the same iteration cap, the codes are on par.** At 20 iterations the library is within
  -0.25 to +0.25 s of the reference on every case. At 200 iterations the library is the faster
  one: the reference takes 2.8-5.8 s on Rosenbrock and plane wave, against the library's 1.6-1.8 s.
  The exception is Gaussian 3-D and 5-D, where the reference stays faster (0.33 / 0.95 s, against
  1.01 / 1.67 s).
- **Mismatched as he most likely ran it, the library is 1.9-4.9x slower** (about +0.8 to +1.2 s
  on the GH200). This reproduces the shape of his report, the library slower at every d, though
  not the 7 s. Nearly every node runs to the cap in the library at either setting (for example
  1006 of 1023 split nodes at Rosenbrock 10-D with 200; all of them with 20), so each extra
  iteration is paid in full. The soft search is launch- and synchronisation-bound (laptop
  profile: 30k launches, 0.29 s of GPU busy time in a 2 s build), so its cost per iteration grows
  with the host's launch overhead. On slower hardware the same mismatch plausibly gives several
  seconds. That is an inference: his hardware is not recorded.
- **Errors and trees, 20 vs 200:** mostly the same leaves and similar test L2 on both sides.
  The reference improves more from 200 iterations on plane wave 3-D (test L2 0.245 -> 0.148),
  while the library gets worse there (0.184 -> 0.273). The library builds fewer leaves with 200
  on Gaussian 5-D (335 vs 394) and plane wave 3-D (878 vs 967).
- Plane wave with `--offset 1` reaches zero (1 + cos), so its max pointwise relative error is
  not meaningful (~1e8). Use a larger offset for plane-wave error comparisons.

## Timing matrix: reference vs library, matched (Vista, one GH200)

Both sides: multiplier 10, 20 optimizer iterations, depth 10, tolerance 1e-2 max pointwise
relative, offset 1, 20,000 test points, depth-1 warm-up then the timed build, each run in a fresh
process, two rounds with the side order swapped in round 2 (`scratch/scripts/timing_matrix.py`,
`scratch/results/timing_matrix.jsonl`). The reference runs its GPU BFS path
(`--traversal bfs --optimizer cupy_batched_cholesky --gpu-sampling --no-check-split-gain`). Host MiB
is the process's peak RSS. GPU MiB is the process's peak device memory from nvidia-smi
(including the CuPy pool and the CUDA context, so it is an upper bound on what the build needs).
All 96 runs exited 0.

| benchmark | leaf | d | side | build s (rounds) | warm-up s | leaves | depth | queries | test rel L2 | q99 pw rel | max pw rel | host MiB | GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | affine | 3 | reference | 0.205 / 0.208 | 0.23 / 0.23 | 28 | 7 | 1,120 | 0.00463 | 0.0153 | 0.0356 | 892 | 978 |
| gaussian | affine | 3 | library | 0.466 / 0.472 | 0.43 / 0.44 | 37 | 7 | 1,553 | 0.00434 | 0.0151 | 0.0309 | 921 | 1726 |
| gaussian | affine | 5 | reference | 0.405 / 0.400 | 0.23 / 0.24 | 347 | 10 | 20,820 | 0.00457 | 0.0151 | 0.06 | 868 | 980 |
| gaussian | affine | 5 | library | 0.646 / 0.640 | 0.43 / 0.43 | 394 | 10 | 24,427 | 0.00463 | 0.015 | 0.0379 | 944 | 1732 |
| gaussian | affine | 10 | reference | 0.879 / 0.876 | 0.24 / 0.23 | 1020 | 10 | 112,200 | 0.0115 | 0.037 | 0.107 | 902 | 990 |
| gaussian | affine | 10 | library | 0.672 / 0.662 | 0.47 / 0.47 | 1016 | 10 | 113,791 | 0.0149 | 0.0488 | 0.117 | 944 | 1864 |
| gaussian | affine | 20 | reference | 0.733 / 0.739 | 0.26 / 0.26 | 1024 | 10 | 215,040 | 0.0175 | 0.0529 | 0.0878 | 903 | 1028 |
| gaussian | affine | 20 | library | 0.765 / 0.769 | 0.50 / 0.49 | 1024 | 10 | 217,087 | 0.0213 | 0.0651 | 0.163 | 1046 | 2112 |
| gaussian | quadratic | 3 | reference | 0.065 / 0.066 | 0.22 / 0.24 | 4 | 3 | 400 | 0.00116 | 0.0052 | 0.0181 | 994 | 980 |
| gaussian | quadratic | 3 | library | 0.062 / 0.063 | 0.47 / 0.48 | 2 | 1 | 203 | 0.00451 | 0.0258 | 0.0945 | 897 | 1726 |
| gaussian | quadratic | 5 | reference | 0.064 / 0.062 | 0.25 / 0.24 | 4 | 3 | 840 | 0.00187 | 0.00683 | 0.0455 | 986 | 996 |
| gaussian | quadratic | 5 | library | 0.254 / 0.253 | 0.48 / 0.50 | 7 | 4 | 1,483 | 0.00128 | 0.00497 | 0.0271 | 926 | 1746 |
| gaussian | quadratic | 10 | reference | 0.130 / 0.129 | 0.31 / 0.30 | 7 | 4 | 4,620 | 0.00117 | 0.00466 | 0.0198 | 1021 | 1122 |
| gaussian | quadratic | 10 | library | 0.257 / 0.260 | 0.53 / 0.53 | 5 | 4 | 3,309 | 0.00182 | 0.00748 | 0.0532 | 899 | 1784 |
| gaussian | quadratic | 20 | reference | 0.088 / 0.086 | 0.27 / 0.27 | 6 | 3 | 13,860 | 0.000917 | 0.00332 | 0.0151 | 995 | 2510 |
| gaussian | quadratic | 20 | library | 0.353 / 0.351 | 0.72 / 0.71 | 1 | 0 | 2,311 | 0.000935 | 0.00339 | 0.0144 | 846 | 1106 |
| plane_wave | affine | 3 | reference | 0.761 / 0.773 | 0.24 / 0.23 | 1022 | 10 | 40,880 | 0.245 | 375 | 1.28e+08 | 882 | 978 |
| plane_wave | affine | 3 | library | 0.678 / 0.668 | 0.43 / 0.43 | 967 | 10 | 40,613 | 0.184 | 160 | 4.23e+08 | 935 | 1738 |
| plane_wave | affine | 5 | reference | 0.897 / 0.926 | 0.23 / 0.23 | 1024 | 10 | 61,440 | 0.539 | 1.52e+03 | 1.4e+08 | 882 | 980 |
| plane_wave | affine | 5 | library | 0.691 / 0.661 | 0.44 / 0.44 | 1024 | 10 | 63,487 | 0.516 | 1.44e+03 | 2.9e+08 | 946 | 1768 |
| plane_wave | affine | 10 | reference | 0.613 / 0.610 | 0.24 / 0.22 | 1024 | 10 | 112,640 | 0.62 | 1.84e+03 | 3.23e+07 | 883 | 988 |
| plane_wave | affine | 10 | library | 0.630 / 0.651 | 0.48 / 0.47 | 1024 | 10 | 114,687 | 0.634 | 1.64e+03 | 4.1e+07 | 953 | 1854 |
| plane_wave | affine | 20 | reference | 0.645 / 0.647 | 0.26 / 0.24 | 1024 | 10 | 215,040 | 0.636 | 1.74e+03 | 6.83e+07 | 899 | 1024 |
| plane_wave | affine | 20 | library | 0.670 / 0.684 | 0.61 / 0.50 | 1024 | 10 | 217,087 | 0.651 | 1.77e+03 | 1.06e+08 | 1037 | 2112 |
| plane_wave | quadratic | 3 | reference | 0.538 / 0.555 | 0.22 / 0.21 | 981 | 10 | 98,100 | 0.102 | 59.5 | 3.84e+07 | 968 | 984 |
| plane_wave | quadratic | 3 | library | 0.631 / 0.626 | 0.52 / 0.48 | 307 | 10 | 31,313 | 0.00161 | 0.0386 | 2.31e+04 | 942 | 1734 |
| plane_wave | quadratic | 5 | reference | 0.673 / 0.673 | 0.30 / 0.23 | 1024 | 10 | 215,040 | 0.261 | 459 | 4.72e+07 | 992 | 1002 |
| plane_wave | quadratic | 5 | library | 0.662 / 0.656 | 0.53 / 0.50 | 1002 | 10 | 212,423 | 0.293 | 296 | 4.46e+07 | 952 | 2020 |
| plane_wave | quadratic | 10 | reference | 0.666 / 0.660 | 0.28 / 0.24 | 1024 | 10 | 675,840 | 0.612 | 1.75e+03 | 2.72e+07 | 997 | 1148 |
| plane_wave | quadratic | 10 | library | 1.305 / 1.296 | 0.58 / 0.52 | 1024 | 10 | 677,887 | 0.615 | 1.69e+03 | 3.15e+07 | 1052 | 4226 |
| plane_wave | quadratic | 20 | reference | 1.665 / 1.666 | 0.45 / 0.26 | 1024 | 10 | 2,365,440 | 0.622 | 1.78e+03 | 1.08e+08 | 1015 | 2874 |
| plane_wave | quadratic | 20 | library | 32.535 / 32.495 | 1.91 / 1.67 | 1024 | 10 | 2,367,487 | 0.627 | 1.59e+03 | 1.72e+08 | 1813 | 31746 |
| rosenbrock | affine | 3 | reference | 0.571 / 0.590 | 0.28 / 0.24 | 1005 | 10 | 40,200 | 0.0207 | 1.04 | 16.7 | 883 | 978 |
| rosenbrock | affine | 3 | library | 0.646 / 0.656 | 0.42 / 0.42 | 755 | 10 | 31,709 | 0.026 | 2.13 | 77.9 | 944 | 1736 |
| rosenbrock | affine | 5 | reference | 0.792 / 0.783 | 0.24 / 0.24 | 1024 | 10 | 61,440 | 0.0995 | 1.43 | 27.7 | 883 | 980 |
| rosenbrock | affine | 5 | library | 0.662 / 0.674 | 0.43 / 0.44 | 1024 | 10 | 63,487 | 0.0967 | 1.84 | 18.9 | 944 | 1768 |
| rosenbrock | affine | 10 | reference | 0.744 / 0.733 | 0.24 / 0.23 | 1024 | 10 | 112,640 | 0.25 | 1.5 | 9.17 | 901 | 990 |
| rosenbrock | affine | 10 | library | 0.644 / 0.649 | 0.46 / 0.48 | 1024 | 10 | 114,687 | 0.292 | 1.71 | 5.71 | 962 | 1856 |
| rosenbrock | affine | 20 | reference | 0.776 / 0.758 | 0.26 / 0.27 | 1024 | 10 | 215,040 | 0.322 | 1.11 | 3.59 | 900 | 1028 |
| rosenbrock | affine | 20 | library | 0.678 / 0.668 | 0.53 / 0.52 | 1024 | 10 | 217,087 | 0.347 | 1.19 | 3.23 | 1059 | 1986 |
| rosenbrock | quadratic | 3 | reference | 0.430 / 0.432 | 0.23 / 0.23 | 676 | 10 | 67,600 | 0.0019 | 0.0774 | 1.53 | 985 | 982 |
| rosenbrock | quadratic | 3 | library | 0.621 / 0.630 | 0.47 / 0.47 | 417 | 10 | 42,533 | 0.00206 | 0.0961 | 2.38 | 960 | 1748 |
| rosenbrock | quadratic | 5 | reference | 0.567 / 0.551 | 2.31 / 0.24 | 1023 | 10 | 214,830 | 0.017 | 0.341 | 8.73 | 1144 | 1002 |
| rosenbrock | quadratic | 5 | library | 0.660 / 0.666 | 0.49 / 0.49 | 1021 | 10 | 216,451 | 0.0148 | 0.32 | 3.83 | 952 | 1816 |
| rosenbrock | quadratic | 10 | reference | 0.614 / 0.614 | 4.73 / 0.27 | 1024 | 10 | 675,840 | 0.0798 | 0.5 | 3.55 | 1184 | 1154 |
| rosenbrock | quadratic | 10 | library | 1.289 / 1.304 | 0.50 / 0.51 | 1024 | 10 | 677,887 | 0.084 | 0.485 | 4.81 | 1087 | 4226 |
| rosenbrock | quadratic | 20 | reference | 1.658 / 1.651 | 2.51 / 0.35 | 1024 | 10 | 2,365,440 | 0.119 | 0.425 | 1.05 | 1177 | 2914 |
| rosenbrock | quadratic | 20 | library | 33.864 / 32.715 | 1.70 / 1.70 | 1024 | 10 | 2,367,487 | 0.131 | 0.465 | 1.25 | 2142 | 31746 |

| benchmark | leaf | d | library / reference (best of rounds) | library - reference s |
|---|---|---|---|---|
| gaussian | affine | 3 | 2.28 | +0.262 |
| gaussian | affine | 5 | 1.60 | +0.239 |
| gaussian | affine | 10 | 0.76 | -0.214 |
| gaussian | affine | 20 | 1.04 | +0.032 |
| gaussian | quadratic | 3 | 0.96 | -0.003 |
| gaussian | quadratic | 5 | 4.07 | +0.191 |
| gaussian | quadratic | 10 | 1.99 | +0.128 |
| gaussian | quadratic | 20 | 4.07 | +0.265 |
| plane_wave | affine | 3 | 0.88 | -0.093 |
| plane_wave | affine | 5 | 0.74 | -0.236 |
| plane_wave | affine | 10 | 1.03 | +0.021 |
| plane_wave | affine | 20 | 1.04 | +0.025 |
| plane_wave | quadratic | 3 | 1.16 | +0.088 |
| plane_wave | quadratic | 5 | 0.97 | -0.018 |
| plane_wave | quadratic | 10 | 1.96 | +0.636 |
| plane_wave | quadratic | 20 | 19.51 | +30.830 |
| rosenbrock | affine | 3 | 1.13 | +0.075 |
| rosenbrock | affine | 5 | 0.84 | -0.122 |
| rosenbrock | affine | 10 | 0.88 | -0.089 |
| rosenbrock | affine | 20 | 0.88 | -0.090 |
| rosenbrock | quadratic | 3 | 1.44 | +0.191 |
| rosenbrock | quadratic | 5 | 1.20 | +0.109 |
| rosenbrock | quadratic | 10 | 2.10 | +0.676 |
| rosenbrock | quadratic | 20 | 19.81 | +31.064 |

### Results

- **Affine leaves: on par.** Library / reference is 0.74-1.13 at d = 3-20 on all three
  benchmarks, except Gaussian 3-D and 5-D (2.3x and 1.6x). Those are small trees (28-394 leaves),
  where the library's fixed per-level cost dominates; the absolute gap is 0.24-0.26 s. Errors are
  similar. The library builds fewer leaves on Rosenbrock 3-D (755 vs 1005) and on plane wave 3-D
  (967 vs 1022, with a lower test L2: 0.18 vs 0.25).
- **Quadratic leaves, d <= 5: on par or within 0.2 s.** Plane wave 3-D quadratic is much better
  in the library: 307 leaves, test L2 1.6e-3, against 981 leaves and 0.10. This matches his point
  that the library gives better errors on some plane-wave runs.
- **Quadratic leaves, d = 10: about 2x slower** (1.30 vs 0.61-0.66 s, +0.64-0.68 s), at 4.2 GB of
  GPU memory against 1.1 GB.
- **Quadratic leaves, d = 20: 19.5x slower and about 11x the GPU memory.** 32.5-33.9 s against
  1.65-1.67 s, 31.7 GB against 2.9 GB, the same 1024 leaves and nearly the same queries
  (2.37M). This is the real scaling gap: p = 231 coefficients, 2,310 points per node. It is the
  first target for the profiles.
- **Gaussian quadratic d = 20:** the library accepts the root (1 leaf, 2,311 queries, test max
  1.44e-2). The reference splits to 6 leaves (13,860 queries, test max 1.51e-2). Both are cheap.
  Different root decisions on a near-tolerance fit; not investigated.
- **Warm-up:** the library's is 0.42-0.72 s (1.7-1.9 s for quadratic d = 20) against the
  reference's 0.21-0.45 s. The reference's first round of quadratic 5-D / 10-D / 20-D shows
  2.3-4.7 s warm-ups: its kernels for those shapes were compiled then. Builds are not affected.
- **Host memory:** 0.85-1.2 GB on both sides, except library quadratic d = 20 (2.1 GB).
- **Plane wave max pointwise relative errors (1e7-1e8) are not meaningful:** with offset 1 the
  target 1 + cos(.) reaches 0.

## Timing matrix after changes 1-5 (Vista, one GH200)

Same protocol as the first matrix (both sides warmed up, two interleaved rounds, fresh process per
run; `scratch/results/timing_matrix_solvesh.jsonl`), after changes 1-5 (change 5 before its
exact-leaf-fit refinement, which adds about 0.05 s at quadratic 20-D). All 96 runs exited 0.

| benchmark | leaf | d | side | build s (rounds) | warm-up s | leaves | depth | queries | test rel L2 | q99 pw rel | max pw rel | host MiB | GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | affine | 3 | reference | 0.210 / 0.208 | 0.24 / 0.23 | 28 | 7 | 1,120 | 0.00463 | 0.0153 | 0.0356 | 875 | 978 |
| gaussian | affine | 3 | library | 0.474 / 0.463 | 0.43 / 0.43 | 37 | 7 | 1,553 | 0.00434 | 0.0151 | 0.0309 | 917 | 1726 |
| gaussian | affine | 5 | reference | 0.395 / 0.392 | 0.23 / 0.24 | 347 | 10 | 20,820 | 0.00457 | 0.0151 | 0.06 | 875 | 980 |
| gaussian | affine | 5 | library | 0.648 / 0.634 | 0.44 / 0.44 | 394 | 10 | 24,427 | 0.00463 | 0.015 | 0.0379 | 944 | 1732 |
| gaussian | affine | 10 | reference | 0.863 / 0.873 | 0.24 / 0.23 | 1020 | 10 | 112,200 | 0.0115 | 0.037 | 0.107 | 884 | 990 |
| gaussian | affine | 10 | library | 0.663 / 0.650 | 0.47 / 0.46 | 1016 | 10 | 113,791 | 0.0149 | 0.0488 | 0.117 | 953 | 1856 |
| gaussian | affine | 20 | reference | 0.749 / 0.736 | 0.25 / 0.25 | 1024 | 10 | 215,040 | 0.0175 | 0.0529 | 0.0878 | 891 | 1028 |
| gaussian | affine | 20 | library | 0.758 / 0.745 | 0.48 / 0.49 | 1024 | 10 | 217,087 | 0.0213 | 0.0651 | 0.163 | 1037 | 2110 |
| gaussian | quadratic | 3 | reference | 0.068 / 0.069 | 0.23 / 0.23 | 4 | 3 | 400 | 0.00116 | 0.0052 | 0.0181 | 976 | 980 |
| gaussian | quadratic | 3 | library | 0.060 / 0.061 | 0.48 / 0.48 | 2 | 1 | 203 | 0.00451 | 0.0258 | 0.0945 | 879 | 1726 |
| gaussian | quadratic | 5 | reference | 0.066 / 0.064 | 0.24 / 0.25 | 4 | 3 | 840 | 0.00187 | 0.00683 | 0.0455 | 987 | 996 |
| gaussian | quadratic | 5 | library | 0.251 / 0.251 | 0.51 / 0.51 | 7 | 4 | 1,483 | 0.00128 | 0.00497 | 0.0271 | 926 | 1746 |
| gaussian | quadratic | 10 | reference | 0.133 / 0.132 | 0.30 / 0.30 | 7 | 4 | 4,620 | 0.00117 | 0.00466 | 0.0198 | 1004 | 1122 |
| gaussian | quadratic | 10 | library | 0.211 / 0.209 | 0.50 / 0.50 | 5 | 4 | 3,309 | 0.00182 | 0.00748 | 0.0532 | 917 | 1784 |
| gaussian | quadratic | 20 | reference | 0.086 / 0.086 | 0.27 / 0.27 | 6 | 3 | 13,860 | 0.000917 | 0.00332 | 0.0151 | 997 | 2510 |
| gaussian | quadratic | 20 | library | 0.022 / 0.022 | 0.38 / 0.37 | 1 | 0 | 2,311 | 0.000935 | 0.00339 | 0.0144 | 846 | 1106 |
| plane_wave | affine | 3 | reference | 0.780 / 0.768 | 0.24 / 0.23 | 1022 | 10 | 40,880 | 0.245 | 375 | 1.28e+08 | 882 | 978 |
| plane_wave | affine | 3 | library | 0.662 / 0.653 | 0.43 / 0.42 | 967 | 10 | 40,613 | 0.184 | 160 | 4.23e+08 | 935 | 1738 |
| plane_wave | affine | 5 | reference | 0.940 / 0.903 | 0.24 / 0.23 | 1024 | 10 | 61,440 | 0.539 | 1.52e+03 | 1.4e+08 | 900 | 980 |
| plane_wave | affine | 5 | library | 0.660 / 0.668 | 0.43 / 0.42 | 1024 | 10 | 63,487 | 0.516 | 1.44e+03 | 2.9e+08 | 921 | 1758 |
| plane_wave | affine | 10 | reference | 0.615 / 0.617 | 0.23 / 0.23 | 1024 | 10 | 112,640 | 0.62 | 1.84e+03 | 3.23e+07 | 883 | 988 |
| plane_wave | affine | 10 | library | 0.637 / 0.644 | 0.48 / 0.48 | 1024 | 10 | 114,687 | 0.634 | 1.64e+03 | 4.1e+07 | 953 | 1802 |
| plane_wave | affine | 20 | reference | 0.640 / 0.644 | 0.25 / 0.24 | 1024 | 10 | 215,040 | 0.636 | 1.74e+03 | 6.83e+07 | 900 | 1024 |
| plane_wave | affine | 20 | library | 0.658 / 0.661 | 0.50 / 0.49 | 1024 | 10 | 217,087 | 0.651 | 1.77e+03 | 1.06e+08 | 1037 | 2110 |
| plane_wave | quadratic | 3 | reference | 0.552 / 0.545 | 0.21 / 0.21 | 981 | 10 | 98,100 | 0.102 | 59.5 | 3.84e+07 | 978 | 984 |
| plane_wave | quadratic | 3 | library | 0.611 / 0.614 | 0.46 / 0.48 | 307 | 10 | 31,313 | 0.00161 | 0.0386 | 2.31e+04 | 942 | 1734 |
| plane_wave | quadratic | 5 | reference | 0.670 / 0.666 | 0.23 / 0.24 | 1024 | 10 | 215,040 | 0.261 | 459 | 4.72e+07 | 992 | 1002 |
| plane_wave | quadratic | 5 | library | 0.648 / 0.652 | 0.50 / 0.52 | 1002 | 10 | 212,423 | 0.293 | 296 | 4.46e+07 | 971 | 1800 |
| plane_wave | quadratic | 10 | reference | 0.654 / 0.655 | 0.24 / 0.24 | 1024 | 10 | 675,840 | 0.612 | 1.75e+03 | 2.72e+07 | 1007 | 1148 |
| plane_wave | quadratic | 10 | library | 0.845 / 0.845 | 0.51 / 0.50 | 1024 | 10 | 677,887 | 0.612 | 1.75e+03 | 5.8e+07 | 1052 | 4226 |
| plane_wave | quadratic | 20 | reference | 1.679 / 1.671 | 0.26 / 0.27 | 1024 | 10 | 2,365,440 | 0.622 | 1.78e+03 | 1.08e+08 | 1014 | 2874 |
| plane_wave | quadratic | 20 | library | 5.274 / 5.195 | 0.56 / 0.54 | 1024 | 10 | 2,367,487 | 0.629 | 1.63e+03 | 1.12e+08 | 1812 | 31746 |
| rosenbrock | affine | 3 | reference | 0.566 / 0.578 | 0.65 / 0.24 | 1005 | 10 | 40,200 | 0.0207 | 1.04 | 16.7 | 894 | 978 |
| rosenbrock | affine | 3 | library | 0.648 / 0.650 | 0.44 / 0.43 | 755 | 10 | 31,709 | 0.026 | 2.13 | 77.9 | 944 | 1736 |
| rosenbrock | affine | 5 | reference | 0.787 / 0.788 | 0.23 / 0.24 | 1024 | 10 | 61,440 | 0.0995 | 1.43 | 27.7 | 901 | 980 |
| rosenbrock | affine | 5 | library | 0.659 / 0.675 | 0.42 / 0.43 | 1024 | 10 | 63,487 | 0.0967 | 1.84 | 18.9 | 953 | 1768 |
| rosenbrock | affine | 10 | reference | 0.766 / 0.749 | 0.24 / 0.24 | 1024 | 10 | 112,640 | 0.25 | 1.5 | 9.17 | 901 | 990 |
| rosenbrock | affine | 10 | library | 0.639 / 0.651 | 0.47 / 0.47 | 1024 | 10 | 114,687 | 0.292 | 1.71 | 5.71 | 950 | 1854 |
| rosenbrock | affine | 20 | reference | 0.780 / 0.798 | 0.25 / 0.26 | 1024 | 10 | 215,040 | 0.322 | 1.11 | 3.59 | 900 | 1028 |
| rosenbrock | affine | 20 | library | 0.668 / 0.676 | 0.50 / 0.50 | 1024 | 10 | 217,087 | 0.347 | 1.19 | 3.23 | 1042 | 2112 |
| rosenbrock | quadratic | 3 | reference | 0.436 / 0.435 | 0.24 / 0.23 | 676 | 10 | 67,600 | 0.0019 | 0.0774 | 1.53 | 998 | 982 |
| rosenbrock | quadratic | 3 | library | 0.626 / 0.622 | 0.48 / 0.47 | 417 | 10 | 42,533 | 0.00206 | 0.0961 | 2.38 | 951 | 1748 |
| rosenbrock | quadratic | 5 | reference | 0.559 / 0.566 | 0.28 / 0.25 | 1023 | 10 | 214,830 | 0.017 | 0.341 | 8.73 | 993 | 1002 |
| rosenbrock | quadratic | 5 | library | 0.634 / 0.640 | 0.50 / 0.51 | 1021 | 10 | 216,451 | 0.0148 | 0.32 | 3.83 | 953 | 1992 |
| rosenbrock | quadratic | 10 | reference | 0.613 / 0.613 | 0.27 / 0.26 | 1024 | 10 | 675,840 | 0.0798 | 0.5 | 3.55 | 1013 | 1154 |
| rosenbrock | quadratic | 10 | library | 0.850 / 0.846 | 0.50 / 0.51 | 1024 | 10 | 677,887 | 0.0843 | 0.486 | 5.42 | 1087 | 4226 |
| rosenbrock | quadratic | 20 | reference | 1.656 / 1.663 | 0.38 / 0.36 | 1024 | 10 | 2,365,440 | 0.119 | 0.425 | 1.05 | 1024 | 2914 |
| rosenbrock | quadratic | 20 | library | 5.361 / 5.349 | 0.55 / 0.54 | 1024 | 10 | 2,367,487 | 0.131 | 0.47 | 1.19 | 2141 | 31746 |

| benchmark | leaf | d | library / reference (best of rounds) | library - reference s |
|---|---|---|---|---|
| gaussian | affine | 3 | 2.23 | +0.255 |
| gaussian | affine | 5 | 1.62 | +0.242 |
| gaussian | affine | 10 | 0.75 | -0.213 |
| gaussian | affine | 20 | 1.01 | +0.010 |
| gaussian | quadratic | 3 | 0.88 | -0.008 |
| gaussian | quadratic | 5 | 3.89 | +0.186 |
| gaussian | quadratic | 10 | 1.59 | +0.078 |
| gaussian | quadratic | 20 | 0.25 | -0.064 |
| plane_wave | affine | 3 | 0.85 | -0.115 |
| plane_wave | affine | 5 | 0.73 | -0.243 |
| plane_wave | affine | 10 | 1.03 | +0.021 |
| plane_wave | affine | 20 | 1.03 | +0.017 |
| plane_wave | quadratic | 3 | 1.12 | +0.066 |
| plane_wave | quadratic | 5 | 0.97 | -0.018 |
| plane_wave | quadratic | 10 | 1.29 | +0.190 |
| plane_wave | quadratic | 20 | 3.11 | +3.525 |
| rosenbrock | affine | 3 | 1.14 | +0.082 |
| rosenbrock | affine | 5 | 0.84 | -0.128 |
| rosenbrock | affine | 10 | 0.85 | -0.110 |
| rosenbrock | affine | 20 | 0.86 | -0.112 |
| rosenbrock | quadratic | 3 | 1.43 | +0.186 |
| rosenbrock | quadratic | 5 | 1.13 | +0.075 |
| rosenbrock | quadratic | 10 | 1.38 | +0.234 |
| rosenbrock | quadratic | 20 | 3.23 | +3.694 |

### Where it stands

| case | reference | library, start | library, now |
|---|---|---|---|
| Rosenbrock quadratic 20-D | 1.65-1.67 s | 32.5-33.9 s (19.8x) | 5.34-5.36 s (3.2x) |
| plane wave quadratic 20-D | 1.67-1.68 s | 32.5 s (19.5x) | 5.20-5.27 s (3.1x) |
| Rosenbrock quadratic 10-D | 0.61 s | 1.29-1.30 s (2.1x) | 0.85 s (1.4x) |
| plane wave quadratic 10-D | 0.65-0.66 s | 1.30 s (2.0x) | 0.85 s (1.3x) |
| Gaussian quadratic 20-D | 0.086 s | 0.35 s (4.1x) | 0.022 s (0.25x; the library accepts the root) |
| affine leaves, d = 3-20, except Gaussian 3-D / 5-D | | 0.74-1.13x | 0.73-1.14x (unchanged) |
| Gaussian affine 3-D / 5-D | 0.20 / 0.40 s | +0.26 / +0.24 s | +0.26 / +0.24 s (unchanged) |
| Gaussian quadratic 5-D / 10-D | 0.065 / 0.13 s | +0.19 / +0.13 s | +0.19 / +0.08 s |

Peak GPU memory at quadratic 20-D is still 31.7 GB against the reference's 2.9 GB.

### Split stages (GPU-synchronized, timed build only; `scratch/scripts/split_stage_times.py`)

| case | train | bins search | soft search | hard SSE (inside both) | metric basis | outside the split |
|---|---|---|---|---|---|---|
| Rosenbrock quadratic 20-D | 5.37 s | 2.24 s | 2.06 s | 0.55 s | 0.03 s | 0.77 s |
| Rosenbrock quadratic 10-D | 0.84 s | 0.20 s | 0.25 s | 0.06 s | 0.04 s | 0.23 s |
| Gaussian affine 3-D (7 levels) | 0.47 s | 0.09 s | 0.11 s | 0.02 s | 0.04 s | 0.14 s |
| Gaussian quadratic 5-D (4 levels) | 0.24 s | 0.05 s | 0.06 s | 0.01 s | 0.02 s | 0.06 s |

- At quadratic 20-D the soft search alone (2.06 s, 20 iterations) costs more than the reference's
  whole build (1.67 s), and the bins search is extra work the reference does not do: about 180
  factorizations and solves of 231 x 231 systems per node per split.
- The small trees are bound by per-level fixed costs: Gaussian affine 3-D spends about 65 ms
  per level (bins 13 ms, soft 16 ms, metric basis 6 ms, fit and sampling 20 ms) against the
  reference's 29 ms per level for its whole build.

## Final comparison: reference, original library, latest library (Vista, one GH200)

Protocol (`scratch/scripts/final_matrix.py`, `scratch/results/final_matrix.jsonl`): the reference's
preset settings for every case (sample multiplier 10, 20 optimizer iterations, depth 10; the
reference on its GPU BFS path), offset 1, 20,000 test points, depth-1 warm-up then the timed
build, fresh process per run. Five problem types x d = 3, 5, 10, 20 x tolerance 0.1 and 0.01
(max pointwise relative) x affine and quadratic leaves = 80 cases. Variants: **reference**;
**original** (this library before any change here: `scratch/snapshots/base`, whose `src/` equals
the branch head, run through the fixed comparison script); **latest** (the working tree at its
default settings, `affine_search_from = 100`). Two rounds, variant order rotated; times are the
faster round (all 480 runs exited 0). Trees and errors are deterministic, so leaves and errors are
the same in both rounds. Peak GPU memory includes the CuPy pool and the CUDA context.

| benchmark | leaf | tol | d | build s: ref / orig / latest | leaves: ref / orig / latest | test rel L2: ref / orig / latest | q99 pw rel: ref / orig / latest | GPU MiB: ref / orig / latest |
|---|---|---|---|---|---|---|---|---|
| gaussian | affine | 0.01 | 3 | 0.204 / 0.430 / 0.429 | 28 / 36 / 36 | 0.00463 / 0.00434 / 0.00434 | 0.0153 / 0.0151 / 0.0151 | 978 / 1730 / 1730 |
| gaussian | affine | 0.01 | 5 | 0.388 / 0.632 / 0.625 | 347 / 398 / 398 | 0.00457 / 0.00468 / 0.00468 | 0.0151 / 0.0152 / 0.0152 | 980 / 1742 / 1740 |
| gaussian | affine | 0.01 | 10 | 0.850 / 0.645 / 0.641 | 1020 / 1016 / 1016 | 0.0115 / 0.0148 / 0.0148 | 0.037 / 0.0491 / 0.0491 | 990 / 1868 / 1846 |
| gaussian | affine | 0.01 | 20 | 0.747 / 0.650 / 0.650 | 1024 / 1024 / 1024 | 0.0175 / 0.0213 / 0.0213 | 0.0529 / 0.0647 / 0.0647 | 1028 / 2116 / 2134 |
| gaussian | affine | 0.1 | 3 | 0.037 / 0.058 / 0.058 | 2 / 2 / 2 | 0.0223 / 0.026 / 0.026 | 0.0699 / 0.0809 / 0.0809 | 978 / 1724 / 1724 |
| gaussian | affine | 0.1 | 5 | 0.042 / 0.058 / 0.058 | 2 / 2 / 2 | 0.0337 / 0.0336 / 0.0336 | 0.118 / 0.113 / 0.113 | 978 / 1732 / 1732 |
| gaussian | affine | 0.1 | 10 | 0.059 / 0.058 / 0.058 | 3 / 2 / 2 | 0.0283 / 0.0313 / 0.0313 | 0.0926 / 0.0994 / 0.0994 | 984 / 1742 / 1742 |
| gaussian | affine | 0.1 | 20 | 0.029 / 0.015 / 0.015 | 1 / 1 / 1 | 0.0257 / 0.025 / 0.025 | 0.0756 / 0.0709 / 0.0709 | 1008 / 950 / 950 |
| gaussian | quadratic | 0.01 | 3 | 0.066 / 0.061 / 0.062 | 4 / 2 / 2 | 0.00116 / 0.00451 / 0.00451 | 0.0052 / 0.0258 / 0.0258 | 980 / 1730 / 1730 |
| gaussian | quadratic | 0.01 | 5 | 0.062 / 0.253 / 0.248 | 4 / 7 / 7 | 0.00187 / 0.00128 / 0.00128 | 0.00683 / 0.00497 / 0.00497 | 996 / 1756 / 1756 |
| gaussian | quadratic | 0.01 | 10 | 0.132 / 0.250 / 0.211 | 7 / 5 / 5 | 0.00117 / 0.00182 / 0.00182 | 0.00466 / 0.00748 / 0.00748 | 1122 / 1798 / 1798 |
| gaussian | quadratic | 0.01 | 20 | 0.085 / 0.346 / 0.023 | 6 / 1 / 1 | 0.000917 / 0.000935 / 0.000935 | 0.00332 / 0.00339 / 0.00339 | 2510 / 1128 / 1128 |
| gaussian | quadratic | 0.1 | 3 | 0.028 / 0.017 / 0.017 | 1 / 1 / 1 | 0.00536 / 0.00539 / 0.00539 | 0.0168 / 0.0255 / 0.0255 | 980 / 922 / 922 |
| gaussian | quadratic | 0.1 | 5 | 0.030 / 0.018 / 0.017 | 1 / 1 / 1 | 0.00428 / 0.00442 / 0.00442 | 0.0147 / 0.02 / 0.02 | 996 / 944 / 944 |
| gaussian | quadratic | 0.1 | 10 | 0.035 / 0.024 / 0.018 | 1 / 1 / 1 | 0.0022 / 0.00211 / 0.00211 | 0.00634 / 0.00895 / 0.00895 | 1122 / 988 / 988 |
| gaussian | quadratic | 0.1 | 20 | 0.040 / 0.346 / 0.023 | 1 / 1 / 1 | 0.000947 / 0.000935 / 0.000935 | 0.00324 / 0.00339 / 0.00339 | 2510 / 1128 / 1128 |
| gaussian_mixture | affine | 0.01 | 3 | 0.420 / 0.640 / 0.638 | 484 / 524 / 524 | 0.00693 / 0.00692 / 0.00692 | 0.0245 / 0.0273 / 0.0273 | 978 / 1734 / 1732 |
| gaussian_mixture | affine | 0.01 | 5 | 0.772 / 0.665 / 0.679 | 1015 / 1023 / 1023 | 0.0191 / 0.024 / 0.024 | 0.0651 / 0.0762 / 0.0762 | 980 / 1746 / 1756 |
| gaussian_mixture | affine | 0.01 | 10 | 0.773 / 0.638 / 0.639 | 1024 / 1024 / 1024 | 0.05 / 0.06 / 0.06 | 0.151 / 0.188 / 0.188 | 990 / 1816 / 1806 |
| gaussian_mixture | affine | 0.01 | 20 | 0.773 / 0.654 / 0.648 | 1024 / 1024 / 1024 | 0.0524 / 0.0628 / 0.0628 | 0.151 / 0.18 / 0.18 | 1028 / 2132 / 1956 |
| gaussian_mixture | affine | 0.1 | 3 | 0.124 / 0.409 / 0.415 | 18 / 24 / 24 | 0.0375 / 0.0364 / 0.0364 | 0.129 / 0.125 / 0.125 | 978 / 1726 / 1726 |
| gaussian_mixture | affine | 0.1 | 5 | 0.211 / 0.557 / 0.559 | 67 / 74 / 74 | 0.0422 / 0.0512 / 0.0512 | 0.14 / 0.154 / 0.154 | 978 / 1736 / 1736 |
| gaussian_mixture | affine | 0.1 | 10 | 0.571 / 0.630 / 0.621 | 619 / 756 / 756 | 0.0526 / 0.0605 / 0.0605 | 0.16 / 0.185 / 0.185 | 986 / 1818 / 1818 |
| gaussian_mixture | affine | 0.1 | 20 | 0.393 / 0.627 / 0.618 | 436 / 489 / 489 | 0.0551 / 0.0646 / 0.0646 | 0.158 / 0.186 / 0.186 | 1012 / 1920 / 1920 |
| gaussian_mixture | quadratic | 0.01 | 3 | 0.207 / 0.568 / 0.565 | 100 / 110 / 110 | 0.00279 / 0.0033 / 0.0033 | 0.00967 / 0.0119 / 0.0119 | 980 / 1738 / 1736 |
| gaussian_mixture | quadratic | 0.01 | 5 | 0.385 / 0.656 / 0.654 | 476 / 731 / 731 | 0.00657 / 0.00402 / 0.00402 | 0.0241 / 0.0144 / 0.0144 | 1000 / 1910 / 1910 |
| gaussian_mixture | quadratic | 0.01 | 10 | 0.697 / 1.259 / 0.823 | 1024 / 1024 / 1024 | 0.0143 / 0.016 / 0.0158 | 0.0488 / 0.0563 / 0.0554 | 1156 / 4242 / 4242 |
| gaussian_mixture | quadratic | 0.01 | 20 | 1.632 / 32.403 / 1.743 | 1024 / 1024 / 1024 | 0.0117 / 0.0175 / 0.0175 | 0.0387 / 0.0607 / 0.0605 | 2918 / 31772 / 24810 |
| gaussian_mixture | quadratic | 0.1 | 3 | 0.110 / 0.314 / 0.329 | 10 / 9 / 9 | 0.025 / 0.0308 / 0.0308 | 0.11 / 0.102 / 0.102 | 980 / 1736 / 1736 |
| gaussian_mixture | quadratic | 0.1 | 5 | 0.196 / 0.332 / 0.329 | 19 / 12 / 12 | 0.0219 / 0.0319 / 0.0319 | 0.0831 / 0.108 / 0.108 | 996 / 1758 / 1758 |
| gaussian_mixture | quadratic | 0.1 | 10 | 0.304 / 0.895 / 0.573 | 58 / 106 / 115 | 0.0245 / 0.0246 / 0.0243 | 0.0882 / 0.0871 / 0.0861 | 1122 / 1838 / 1846 |
| gaussian_mixture | quadratic | 0.1 | 20 | 0.337 / 11.505 / 0.697 | 11 / 22 / 37 | 0.0187 / 0.0241 / 0.023 | 0.0612 / 0.0775 / 0.0744 | 2510 / 2126 / 1940 |
| plane_wave | affine | 0.01 | 3 | 0.757 / 0.660 / 0.663 | 1022 / 967 / 967 | 0.245 / 0.184 / 0.184 | 375 / 160 / 160 | 978 / 1734 / 1738 |
| plane_wave | affine | 0.01 | 5 | 0.912 / 0.648 / 0.655 | 1024 / 1024 / 1024 | 0.539 / 0.516 / 0.516 | 1.52e+03 / 1.44e+03 / 1.44e+03 | 980 / 1760 / 1768 |
| plane_wave | affine | 0.01 | 10 | 0.613 / 0.646 / 0.624 | 1024 / 1024 / 1024 | 0.62 / 0.634 / 0.634 | 1.84e+03 / 1.64e+03 / 1.64e+03 | 988 / 1856 / 1834 |
| plane_wave | affine | 0.01 | 20 | 0.640 / 0.666 / 0.652 | 1024 / 1024 / 1024 | 0.636 / 0.651 / 0.651 | 1.74e+03 / 1.77e+03 / 1.77e+03 | 1024 / 2112 / 1932 |
| plane_wave | affine | 0.1 | 3 | 0.696 / 0.637 / 0.639 | 878 / 746 / 746 | 0.249 / 0.189 / 0.189 | 258 / 159 / 159 | 978 / 1738 / 1738 |
| plane_wave | affine | 0.1 | 5 | 0.895 / 0.659 / 0.659 | 1024 / 995 / 995 | 0.539 / 0.516 / 0.516 | 1.52e+03 / 1.4e+03 / 1.4e+03 | 980 / 1768 / 1768 |
| plane_wave | affine | 0.1 | 10 | 0.606 / 0.634 / 0.638 | 1024 / 1024 / 1024 | 0.62 / 0.637 / 0.637 | 1.84e+03 / 1.63e+03 / 1.63e+03 | 988 / 1854 / 1854 |
| plane_wave | affine | 0.1 | 20 | 0.649 / 0.662 / 0.660 | 1024 / 1024 / 1024 | 0.636 / 0.65 / 0.65 | 1.74e+03 / 1.67e+03 / 1.67e+03 | 1024 / 2112 / 2122 |
| plane_wave | quadratic | 0.01 | 3 | 0.538 / 0.616 / 0.618 | 981 / 307 / 307 | 0.102 / 0.00161 / 0.00161 | 59.5 / 0.0386 / 0.0386 | 984 / 1734 / 1734 |
| plane_wave | quadratic | 0.01 | 5 | 0.664 / 0.647 / 0.631 | 1024 / 1002 / 1002 | 0.261 / 0.293 / 0.293 | 459 / 296 / 296 | 1002 / 1898 / 2020 |
| plane_wave | quadratic | 0.01 | 10 | 0.653 / 1.296 / 0.844 | 1024 / 1024 / 1024 | 0.612 / 0.615 / 0.612 | 1.75e+03 / 1.69e+03 / 1.75e+03 | 1148 / 4226 / 4226 |
| plane_wave | quadratic | 0.01 | 20 | 1.670 / 32.493 / 1.864 | 1024 / 1024 / 1024 | 0.622 / 0.627 / 0.627 | 1.78e+03 / 1.59e+03 / 1.71e+03 | 2874 / 31746 / 24788 |
| plane_wave | quadratic | 0.1 | 3 | 0.527 / 0.600 / 0.603 | 764 / 172 / 172 | 0.0714 / 0.00804 / 0.00804 | 61.5 / 0.0825 / 0.0825 | 982 / 1732 / 1734 |
| plane_wave | quadratic | 0.1 | 5 | 0.676 / 0.646 / 0.639 | 1024 / 901 / 901 | 0.261 / 0.294 / 0.294 | 459 / 306 / 306 | 1002 / 1968 / 1804 |
| plane_wave | quadratic | 0.1 | 10 | 0.654 / 1.294 / 0.837 | 1024 / 1024 / 1024 | 0.612 / 0.616 / 0.612 | 1.75e+03 / 1.66e+03 / 1.78e+03 | 1148 / 4226 / 4226 |
| plane_wave | quadratic | 0.1 | 20 | 1.677 / 32.510 / 1.858 | 1024 / 1024 / 1024 | 0.622 / 0.626 / 0.625 | 1.78e+03 / 1.68e+03 / 1.68e+03 | 2874 / 31746 / 24788 |
| quadratic | affine | 0.01 | 3 | 0.592 / 0.627 / 0.644 | 841 / 685 / 685 | 0.00691 / 0.00626 / 0.00626 | 0.0551 / 0.0557 / 0.0557 | 978 / 1732 / 1734 |
| quadratic | affine | 0.01 | 5 | 0.700 / 0.642 / 0.641 | 1019 / 911 / 911 | 0.0261 / 0.0157 / 0.0157 | 0.245 / 0.138 / 0.138 | 980 / 1754 / 1738 |
| quadratic | affine | 0.01 | 10 | 0.779 / 0.647 / 0.644 | 1024 / 1024 / 1024 | 0.0643 / 0.0405 / 0.0405 | 0.499 / 0.38 / 0.38 | 988 / 1854 / 1854 |
| quadratic | affine | 0.01 | 20 | 0.844 / 0.760 / 0.750 | 1024 / 1024 / 1024 | 0.113 / 0.153 / 0.153 | 0.942 / 1.19 / 1.19 | 1024 / 2112 / 1986 |
| quadratic | affine | 0.1 | 3 | 0.200 / 0.599 / 0.595 | 102 / 82 / 82 | 0.0285 / 0.0283 / 0.0283 | 0.136 / 0.134 / 0.134 | 978 / 1726 / 1726 |
| quadratic | affine | 0.1 | 5 | 0.378 / 0.623 / 0.621 | 512 / 219 / 219 | 0.0312 / 0.0302 / 0.0302 | 0.238 / 0.155 / 0.155 | 978 / 1730 / 1730 |
| quadratic | affine | 0.1 | 10 | 0.637 / 0.643 / 0.634 | 945 / 708 / 708 | 0.0652 / 0.0455 / 0.0455 | 0.53 / 0.375 / 0.375 | 988 / 1804 / 1782 |
| quadratic | affine | 0.1 | 20 | 0.851 / 0.772 / 0.755 | 1024 / 1011 / 1011 | 0.113 / 0.154 / 0.154 | 0.942 / 1.17 / 1.17 | 1024 / 2110 / 2110 |
| quadratic | quadratic | 0.01 | 3 | 0.012 / 0.017 / 0.016 | 1 / 1 / 1 | 2.65e-16 / 6.3e-16 / 6.3e-16 | 1.47e-15 / 3.15e-15 / 3.15e-15 | 976 / 918 / 918 |
| quadratic | quadratic | 0.01 | 5 | 0.012 / 0.017 / 0.017 | 1 / 1 / 1 | 1.52e-15 / 8.91e-16 / 8.91e-16 | 1.42e-14 / 5.84e-15 / 5.84e-15 | 990 / 934 / 934 |
| quadratic | quadratic | 0.01 | 10 | 0.016 / 0.023 / 0.017 | 1 / 1 / 1 | 6.93e-16 / 1.12e-15 / 1.12e-15 | 5.07e-15 / 7.55e-15 / 7.55e-15 | 1110 / 974 / 974 |
| quadratic | quadratic | 0.01 | 20 | 0.024 / 0.349 / 0.023 | 1 / 1 / 1 | 1.16e-15 / 1.4e-15 / 1.4e-15 | 9.01e-15 / 1.03e-14 / 1.03e-14 | 2518 / 1106 / 1106 |
| quadratic | quadratic | 0.1 | 3 | 0.012 / 0.016 / 0.017 | 1 / 1 / 1 | 2.65e-16 / 6.3e-16 / 6.3e-16 | 1.47e-15 / 3.15e-15 / 3.15e-15 | 976 / 918 / 918 |
| quadratic | quadratic | 0.1 | 5 | 0.012 / 0.017 / 0.017 | 1 / 1 / 1 | 1.52e-15 / 8.91e-16 / 8.91e-16 | 1.42e-14 / 5.84e-15 / 5.84e-15 | 990 / 934 / 934 |
| quadratic | quadratic | 0.1 | 10 | 0.016 / 0.024 / 0.017 | 1 / 1 / 1 | 6.93e-16 / 1.12e-15 / 1.12e-15 | 5.07e-15 / 7.55e-15 / 7.55e-15 | 1110 / 974 / 974 |
| quadratic | quadratic | 0.1 | 20 | 0.024 / 0.349 / 0.023 | 1 / 1 / 1 | 1.16e-15 / 1.4e-15 / 1.4e-15 | 9.01e-15 / 1.03e-14 / 1.03e-14 | 2874 / 1106 / 1106 |
| rosenbrock | affine | 0.01 | 3 | 0.567 / 0.644 / 0.635 | 1005 / 754 / 754 | 0.0207 / 0.0264 / 0.0264 | 1.04 / 2.13 / 2.13 | 978 / 1736 / 1736 |
| rosenbrock | affine | 0.01 | 5 | 0.777 / 0.655 / 0.656 | 1024 / 1024 / 1024 | 0.0995 / 0.096 / 0.096 | 1.43 / 1.84 / 1.84 | 980 / 1778 / 1756 |
| rosenbrock | affine | 0.01 | 10 | 0.743 / 0.653 / 0.630 | 1024 / 1024 / 1024 | 0.25 / 0.292 / 0.292 | 1.5 / 1.79 / 1.79 | 990 / 1868 / 1868 |
| rosenbrock | affine | 0.01 | 20 | 0.772 / 0.648 / 0.656 | 1024 / 1024 / 1024 | 0.322 / 0.346 / 0.346 | 1.11 / 1.18 / 1.18 | 1028 / 2132 / 2020 |
| rosenbrock | affine | 0.1 | 3 | 0.358 / 0.647 / 0.635 | 544 / 277 / 277 | 0.0311 / 0.0394 / 0.0394 | 1.2 / 2.13 / 2.13 | 978 / 1730 / 1732 |
| rosenbrock | affine | 0.1 | 5 | 0.794 / 0.643 / 0.648 | 969 / 751 / 751 | 0.0995 / 0.0978 / 0.0978 | 1.36 / 1.83 / 1.83 | 980 / 1760 / 1760 |
| rosenbrock | affine | 0.1 | 10 | 0.739 / 0.635 / 0.649 | 1024 / 1024 / 1024 | 0.25 / 0.292 / 0.292 | 1.5 / 1.77 / 1.77 | 990 / 1868 / 1868 |
| rosenbrock | affine | 0.1 | 20 | 0.760 / 0.655 / 0.644 | 1024 / 1024 / 1024 | 0.322 / 0.349 / 0.349 | 1.11 / 1.21 / 1.21 | 1028 / 2132 / 2132 |
| rosenbrock | quadratic | 0.01 | 3 | 0.419 / 0.626 / 0.630 | 676 / 424 / 424 | 0.0019 / 0.00205 / 0.00205 | 0.0774 / 0.0844 / 0.0844 | 982 / 1752 / 1740 |
| rosenbrock | quadratic | 0.01 | 5 | 0.559 / 0.647 / 0.640 | 1023 / 1022 / 1022 | 0.017 / 0.0147 / 0.0147 | 0.341 / 0.319 / 0.319 | 1002 / 1814 / 1824 |
| rosenbrock | quadratic | 0.01 | 10 | 0.614 / 1.265 / 0.804 | 1024 / 1024 / 1024 | 0.0798 / 0.0848 / 0.0837 | 0.5 / 0.505 / 0.493 | 1154 / 4240 / 4240 |
| rosenbrock | quadratic | 0.01 | 20 | 1.657 / 32.432 / 1.762 | 1024 / 1024 / 1024 | 0.119 / 0.13 / 0.13 | 0.425 / 0.461 / 0.463 | 2914 / 31766 / 24810 |
| rosenbrock | quadratic | 0.1 | 3 | 0.236 / 0.660 / 0.612 | 229 / 155 / 155 | 0.0114 / 0.00946 / 0.00946 | 0.129 / 0.108 / 0.108 | 982 / 1740 / 1740 |
| rosenbrock | quadratic | 0.1 | 5 | 0.491 / 0.636 / 0.616 | 826 / 603 / 603 | 0.0183 / 0.0206 / 0.0206 | 0.346 / 0.317 / 0.317 | 1002 / 1880 / 1880 |
| rosenbrock | quadratic | 0.1 | 10 | 0.611 / 1.275 / 0.817 | 1024 / 1024 / 1024 | 0.0798 / 0.0841 / 0.0829 | 0.5 / 0.493 / 0.504 | 1154 / 4240 / 4240 |
| rosenbrock | quadratic | 0.1 | 20 | 1.657 / 32.383 / 1.775 | 1024 / 1024 / 1024 | 0.119 / 0.13 / 0.13 | 0.425 / 0.465 / 0.461 | 2914 / 31766 / 24810 |

| group | cases | time latest/ref: median (range) | time latest/orig: median (range) | time orig/ref: median | leaves latest/ref: median | leaves latest/orig: median (range) | test L2 latest/ref: median (range) | test L2 latest/orig: median (range) |
|---|---|---|---|---|---|---|---|---|
| affine, d <= 10 | 30 | 1.01 (0.72-3.34) | 1.00 (0.96-1.03) | 1.03 | 1.00 | 1.00 (1.00-1.00) | 1.00 (0.60-1.29) | 1.00 (1.00-1.00) |
| affine, d = 20 | 10 | 0.88 (0.54-1.57) | 0.99 (0.98-1.01) | 0.89 | 1.00 | 1.00 (1.00-1.00) | 1.13 (0.97-1.36) | 1.00 (1.00-1.00) |
| quadratic, d <= 10 | 30 | 1.30 (0.51-3.99) | 0.98 (0.64-1.05) | 1.48 | 1.00 | 1.00 (1.00-1.08) | 1.04 (0.02-3.88) | 1.00 (0.99-1.00) |
| quadratic, d = 20 | 10 | 1.07 (0.27-2.07) | 0.06 (0.05-0.07) | 19.43 | 1.00 | 1.00 (1.00-1.68) | 1.09 (0.99-1.49) | 1.00 (0.95-1.00) |

### Oracle on the GPU for this library's side

`scripts/art_comparison.py` evaluated the benchmark for this library's ART on the host (points to
the host, the reference's NumPy target, values back), while the reference with `--gpu-sampling`
evaluates Rosenbrock, Gaussian and Gaussian mixture on the GPU (`CudaOracleWorkspace`). At 2.37M
points in 20-D the host evaluation alone takes 2.47 s (mixture), 1.24 s (Gaussian), 0.26 s
(Rosenbrock). The script now evaluates those three through the reference's own
`CudaOracleWorkspace`, in fixed padded chunks of 65,536 rows so a point's value does not depend
on its batch (`_device_oracle`; `--library-host-oracle` restores the host path). Values agree
with the host target within 2e-16 to 7e-16 relative (d = 3 and 20) and are batch-independent.
Plane wave and the quadratic benchmark have no GPU oracle in the reference and stay on the host
on both sides. The tables above use the GPU oracle for both library variants on Rosenbrock,
Gaussian and Gaussian mixture (one round, `scratch/results/final_matrix_gpuoracle.jsonl`), and
the two-round host-oracle runs elsewhere.

The oracle's rounding-level differences change borderline accept/split decisions: in 33 of the
96 rerun library runs the leaf count changed by a few (once 533 -> 489, Gaussian mixture affine
20-D at tol 0.1), with test L2 within 0.92-1.05 of the host-oracle run, mostly identically in
both library variants. Trees on these problems are that sensitive to rounding in f.

### What changed

- **Time, quadratic leaves at d = 20 (10 cases):** latest / original median 0.06 (0.05-0.07), so
  14-20x faster (Rosenbrock 32.4 -> 1.76 s, Gaussian mixture 32.4 -> 1.74 s, plane wave
  32.5 -> 1.86 s). Latest / reference median 1.07: Rosenbrock 1.06-1.07x (+0.11 s), Gaussian
  mixture 1.07x at tol 0.01, plane wave 1.11-1.12x (+0.19 s); Gaussian mixture at tol 0.1 2.07x
  (0.70 vs 0.34 s: 37 leaves against the reference's 11). Where the tree is one leaf (Gaussian,
  quadratic benchmark), 0.27-1.0x the reference.
- **Time, quadratic leaves at d <= 10 (30 cases):** latest / original median 0.98, latest /
  reference median 1.30 (0.51-3.99). The largest ratios are small trees, at most +0.40 s
  (Gaussian 5-D 0.064 -> 0.25 s, Gaussian mixture 3-D, Rosenbrock 3-D), from per-level fixed
  costs (about 65 ms per level against the reference's 29 ms).
- **Time, affine leaves (40 cases):** unchanged from the original (median 1.00 / 0.99); against
  the reference median 1.01 at d <= 10 and 0.88 at d = 20, except small trees (Gaussian mixture
  3-D 3.3x, quadratic 3-D 3.0x, Gaussian 3-D 2.2x, each +0.26-0.40 s).
- **Leaves, latest vs original:** the same in 78 of 80 cases with the same oracle in both (also
  the same oracle queries). The others: Gaussian mixture quadratic at tol 0.1, 10-D 106 -> 115 and
  20-D 22 -> 37 leaves (the affine search at 20-D, the relaxed solves at 10-D), with test L2 within
  0.95-1.00. Against the reference: the same count in most depth-capped cases (1024); fewer on
  plane wave quadratic 3-D (307 vs 981 at 0.01) and Rosenbrock quadratic 3-D.
- **Accuracy, latest vs original:** test L2 ratio within 0.95-1.00 for quadratic leaves and
  1.00 for affine leaves (identical trees).
- **Accuracy, latest vs reference:** median test L2 ratio 1.00 (affine d <= 10), 1.13 (affine
  d = 20), 1.04 (quadratic d <= 10), 1.09 (quadratic d = 20). Better than the reference on plane
  wave quadratic 3-D (0.0016 vs 0.102 at 0.01: 63x lower, with a third of the leaves) and several
  5-D cases; worse on Gaussian quadratic 3-D (0.0045 vs 0.0012; 2 vs 4 leaves) and 10-D (1.55x).
- **Peak GPU memory:** quadratic 20-D 31.7 GB (original) -> 24.8 GB (latest), against the
  reference's 2.9 GB; other cases 1.1-2.1 GB (library) against 1.0-2.9 GB (reference).

## Numerical contract under review (not adopted; AGENTS.md unchanged)

Two guarantees in AGENTS.md are often both called "bit-identical":

1. **Refactors are bit-identical to the previous code.** A kernel rewrite must keep each sum's
   operation order, so the new kernel gives the old kernel's bits.
2. **GPU training is reproducible and batch-independent.** Repeated runs give the same bits, and
   a node's arithmetic depends only on its own rows (any batch size, rank count or steal
   schedule gives the same tree): fixed-order per-segment sums, no `atomicAdd` float sums, no
   cuBLAS-batched products.

Under review: **relax 1, keep 2**, as an experiment, adopted only if it reaches parity with the
reference. Kernels may then use a different fixed summation order (blocked or tree reductions,
warp shuffles, parallel triangular solves): still deterministic and per segment, but rounding
differently from the kernels they replace. That is a one-time deliberate numerical change on
the GPU, listed below. The CPU path does not change, so `regression_check.py` stays bit-identical
on CPU.

How a relaxed change is checked:

- GPU vs CPU within the GPU tests' tight tolerance (1e-12; 1e-10 for separate trainings), with
  identical leaf and evaluation counts (the existing GPU test rule), and the GPU suite passing.
- `regression_check.py` on the GPU against the snapshot with `--rtol 1e-12`: decisions identical.
- Repeated GPU runs bit-identical, and the same tree for different batch sizes (guarantee 2).
- The benchmark matrix rerun: leaves, queries and errors statistically unchanged. A borderline
  split can still flip on a large tree, the same kind of difference GPU and CPU trees already
  have; such cases are listed.

Each change below is marked **exact** (guarantee 1 kept) or **relaxed** (guarantee 1 dropped,
guarantee 2 kept).

## Changes to the library (working tree, not committed)

Checks after every change: the complete library model (every saved array and counter) bit for bit
against the snapshot `scratch/snapshots/base` on four ART cases (Rosenbrock quadratic 20-D and
10-D, Gaussian affine 3-D, plane wave quadratic 3-D; `scratch/scripts/save_models.sh`,
`compare_models.py`), `scripts/regression_check.py` exact (rtol 0) on GPU and CPU (52 AFINNI and ART
cases, all 2-D), and the AFINNI cases the regression check does not reach: quadratic leaves at
d = 8 (p = 45, the large-system path), tikhonov and lstsq fits, and affine leaves at d = 8
(`scratch/scripts/afinni_check.py`), bit for bit and timed against the snapshot.

### 1. Column-parallel factored solves, one factorization in `lstsq_auto`: kept (exact)

- What: `spd_solve_factored` (`src/kernels/small_linalg.cu`) runs one thread per (system,
  right-hand side) instead of one per system; each column's arithmetic is unchanged.
  `lstsq_auto` (`src/numerics/linalg.py`) factors each Gram once for the solution and the
  inverse used in its condition number.
- Why: the quadratic d = 20 profile spent 22.3 s of 33 s of GPU time in `spd_solve_factored`:
  `lstsq_auto`'s inverse has q = p = 231 columns, solved serially by one thread per system.
- Result, Rosenbrock quadratic 20-D: 32.9 -> 15.0 s ("Fit + error" 18.0 -> 0.53 s); quadratic
  10-D 1.37 -> 1.06 s; Gaussian affine 3-D and plane wave quadratic 3-D unchanged (0.47, 0.60 s).
  Models bit-identical on all four. regression_check exact on GPU and CPU: PASS (max difference
  0). AFINNI d = 8: bit-identical; tikhonov 1.49 vs 1.51-1.52 s, lstsq 1.12-1.13 vs 1.24-1.25 s,
  affine 1.42 vs 1.42-1.44 s (two interleaved rounds).

### 2. Factor with L^T mirrored into the upper triangle (row-wise back substitution): removed (exact)

- Bit-identical, but slower: `spd_solve_factored` 3.53 -> 5.23 s, the build 15.0 -> 16.8 s. Reverted.

### 3. Right-looking Cholesky in shared memory for 32 <= p <= 239: kept (exact)

- What: `spd_factor_shared` (`src/kernels/small_linalg.cu`), one block per system holding the
  packed lower triangle in shared memory. After each column is final, every warp updates the
  trailing triangle (one warp per row). Each entry gets the same subtractions in the same k
  order and with the same operands as `chol_inplace`, so the factor is bit-identical (checked
  directly against `spd_factor_coop` at p = 32, 45, 66, 120, 231, 239). `spd_factor` uses it
  while `8 (p (p + 1) / 2 + p)` bytes fit the card's opt-in shared memory (p <= 239 on the
  GH200); larger systems keep `spd_factor_coop`.
- Why: `spd_factor_coop` computes each pivot's sum serially and parallelizes only one column.
- Result, two interleaved rounds against the previous state (`scratch/snapshots/solvecols`):
  Rosenbrock quadratic 20-D 14.74 / 14.73 -> 12.73 / 12.73 s (-2.0 s, meets the 2 s bar);
  quadratic 10-D 0.97 / 0.98 -> 0.95 / 0.96 s; plane wave quadratic 3-D 0.60 / 0.60 -> 0.61 / 0.60 s;
  Gaussian affine 3-D 0.47 / 0.46 -> 0.45 / 0.48 s. Models bit-identical on all four.
  regression_check exact, GPU and CPU: PASS. AFINNI d = 8 (p = 45): bit-identical, time unchanged
  (1.52 vs 1.50-1.51 s tikhonov, 1.14 vs 1.12 s lstsq, 1.43-1.44 vs 1.42-1.43 s affine).

### 4. Register-tiled Gram kernel for P, Q >= 32: kept (exact)

- What: `segment_cross_tiled2d` (`src/kernels/segment_cross.cu`, launched by `segment_cross` in
  `src/numerics/segments.py` when both output dimensions are >= 32): one block per (weight set,
  chunk, 64 x 64 output tile), each thread a 4 x 4 register tile, so each shared-memory load
  feeds 4 products instead of half of one. Each entry keeps the same rows, row order and
  operands, so the bits are unchanged (checked directly against the other kernel for 24 shape
  and option combinations: P = 32-231, symmetric and not, weights, two weight sets, offsets,
  an empty segment).
- Why: the 231 x 231 Grams were 6.3 s of the 12.7 s build, bound by shared-memory loads.
- Result, two interleaved rounds against the previous state (`scratch/snapshots/shchol`):
  Rosenbrock quadratic 20-D 12.68 / 12.74 -> 8.03 / 8.04 s (-4.7 s); quadratic 10-D
  0.955 / 0.948 -> 0.949 / 0.941 s; plane wave quadratic 3-D and Gaussian affine 3-D unchanged
  (0.59-0.60 s, 0.46 s). Models bit-identical on all four; regression_check exact on GPU and
  CPU: PASS; AFINNI d = 8 bit-identical, times unchanged (its Grams are 45 x 45).

### 5. Shared-memory solves with a parallel back substitution for p >= 32: under review (relaxed)

- What (`src/kernels/small_linalg.cu`, `src/numerics/linalg.py`): `chol_packed_block` (the
  factorization of `spd_factor_shared`, now shared by three kernels) and
  `chol_packed_solve_block` (forward and back substitution by the whole block). `spd_solve`
  factors and solves each system in shared memory (`spd_solve_shared`) while the triangle and
  right-hand sides fit; `spd_solve_factored` stages an existing factor into shared memory
  (`spd_solve_factored_shared`). The forward substitution keeps the exact order. The back
  substitution is right-looking, so it subtracts `L_ki x_k` in decreasing k instead of
  increasing k: p parallel steps instead of a chain of p^2 / 2 dependent products, at a different
  rounding (fixed per system, so still reproducible and batch-independent). `spd_solve` and
  `spd_solve_factored(spd_factor(A), B)` still agree bit for bit with each other. The inverse in
  `lstsq_auto` (q = p = 231) does not fit and keeps the exact one-thread-per-column kernel.
- Why: after change 4, the single-right-hand-side solves were the largest kernel (3.5 s). The
  soft search's calls have about 2,000 systems each, so the serial back substitution set their
  time (about 5 ms per call), not throughput.
- Unit check: 1e-13 or better against NumPy at p = 20-231 and q = 1, 2, p; non-SPD systems
  still give all-NaN solutions.
- Result, two interleaved rounds against the exact state after change 4 (`scratch/snapshots/gram2d`):
  Rosenbrock quadratic 20-D 8.04 / 8.04 -> 5.36 / 5.36 s (-2.7 s); quadratic 10-D 0.944 / 0.943 ->
  0.815 / 0.814 s; plane wave quadratic 3-D 0.59-0.60 -> 0.60-0.61 s; Gaussian affine 3-D
  unchanged (0.46 s).
- Effect on the trees: Gaussian affine 3-D and plane wave quadratic 3-D are bit-identical (their
  systems are below 32 unknowns, or the changed solves only screen candidates). Rosenbrock
  quadratic 10-D and 20-D keep the same leaf counts (1024), statuses and queries, but their
  split planes differ: the soft optimizer amplifies the rounding change into different planes.
  Test relative L2 0.13102 -> 0.13144 (20-D) and 0.08404 -> 0.08428 (10-D), about +0.3%.
- Leaf fits stay exact: `spd_solve` and `spd_solve_factored` take `exact=True` (the
  one-thread-per-column kernel, increasing-k back substitution), used by `lstsq_auto`'s solution
  and the tikhonov quadratic fit in `src/components/local_models.py`. So only the split
  search's internal solves are relaxed, and AFINNI is bit-identical to the exact baseline (it has
  no other path into them). Cost: about +0.05 s at Rosenbrock quadratic 20-D (5.64 -> 5.68 s).
- Checks: GPU suite 694 passed, 33 skipped (as before); regression_check exact on CPU PASS (54
  cases, with the d = 8 cases of change 6); two separate runs of Rosenbrock quadratic 10-D and
  20-D bit-identical (reproducible); AFINNI d = 8 bit-identical to the exact baseline, times
  unchanged. regression_check on GPU at rtol 1e-12 **fails on one array**: `art_quadratic_d8`'s
  leaf coefficients differ by up to 7.7e-13 absolute (coefficients up to 3.6; 1.7e-11 relative
  on a near-zero element), while every decision is identical (counters, statuses, routing codes,
  masks, leaf codes). The relaxed split search moves the planes by about 1e-14, which moves the
  hit-and-run samples by about 5e-15 and, through an ill-conditioned quadratic fit, the leaf
  coefficients by 7.7e-13 and the predictions by 1.4e-13. This is the expected footprint of a
  relaxed split search; adopting it would need that case's array tolerance stated (1e-10, as
  the GPU tests' `separate_runs` tolerance) or a scale-relative comparison.

### 6. regression_check.py: two cases at d = 8 with quadratic leaves (test coverage)

- What: `art_quadratic_d8` (depth 3, multiplier 3, 5 soft iterations) and `afinni_quadratic_d8`
  (depth 4): 45 coefficients, above `SPD_COOP_MIN_P`, so the GPU factors, solves and forms Grams
  with the large-system kernels that the 2-D cases never reach. About 0.1-0.25 s each.
- Finding: on these cases the GPU and CPU already make different decisions in the exact baseline
  (ART: 1812 vs 1834 evaluations, 8 leaves each; AFINNI: 25 vs 24 leaves), before any change here.
  The GPU tests' rule that GPU and CPU leaf and evaluation counts match is checked on small
  problems only; at d = 8 with quadratic leaves it does not hold. Not investigated further.

### 7. Affine search fits from 100 leaf coefficients (`ARTSettings.affine_search_from = 100`): kept (algorithm, deliberate change)

- What (`src/art/plan.py` `_search_rows`, `src/config.py`, `--library-affine-search-from` in
  `scripts/art_comparison.py`, test `test_split_search_uses_affine_fits_from_affine_search_from_coefficients`):
  when the leaf model has at least `affine_search_from` coefficients, the bins search ranks
  its cuts (A') and the soft search optimizes its plane (A'') with affine fits `[u, 1]` (d + 1
  coefficients) instead of the leaf model's. The bins search's screened candidates and the soft
  result are still judged by the leaf model's hard SSE (`hard_sse`), and the leaves are fitted
  with the leaf model. Below the threshold nothing changes (bit for bit). With quadratic leaves
  the threshold of 100 means d >= 13.
- Why: at quadratic 20-D the bins search factored about 180 systems of 231 unknowns per node per
  split, and the soft search's 20 iterations each formed two 231 x 231 Grams and factored them.
- Measured with experiment switches (since replaced by the threshold): leaf model everywhere,
  affine bins screen (A'), affine bins screen and soft search (A' + A''); two interleaved
  rounds each, library alone (`scratch/results/screen_ab.jsonl`, `soft_ab.jsonl`), reference from
  the matrix after changes 1-5:

| benchmark | d | reference s | leaf / A' / A'+A'' s | ref / leaf / A' / A'+A'' leaves | ref / leaf / A' / A'+A'' test rel L2 | ref / leaf / A' / A'+A'' q99 |
|---|---|---|---|---|---|---|
| rosenbrock | 3 | 0.435 | 0.605 / 0.597 / 0.598 | 676 / 417 / 446 / 437 | 0.0019 / 0.00206 / 0.00252 / 0.00337 | 0.0774 / 0.0961 / 0.13 / 0.314 |
| rosenbrock | 5 | 0.559 | 0.620 / 0.610 / 0.612 | 1023 / 1021 / 1021 / 1021 | 0.017 / 0.0148 / 0.0221 / 0.0309 | 0.341 / 0.32 / 0.398 / 0.546 |
| rosenbrock | 10 | 0.613 | 0.828 / 0.766 / 0.680 | 1024 / 1024 / 1024 / 1024 | 0.0798 / 0.0843 / 0.0869 / 0.0894 | 0.5 / 0.486 / 0.505 / 0.518 |
| rosenbrock | 20 | 1.656 | 5.406 / 3.656 / 1.997 | 1024 / 1024 / 1024 / 1024 | 0.119 / 0.131 / 0.13 / 0.13 | 0.425 / 0.47 / 0.469 / 0.462 |
| gaussian | 3 | 0.068 | 0.059 / 0.060 / 0.191 | 4 / 2 / 2 / 6 | 0.00116 / 0.00451 / 0.00459 / 0.00122 | 0.0052 / 0.0258 / 0.0264 / 0.00482 |
| gaussian | 5 | 0.064 | 0.239 / 0.267 / 0.355 | 4 / 7 / 8 / 12 | 0.00187 / 0.00128 / 0.0015 / 0.00143 | 0.00683 / 0.00497 / 0.00568 / 0.0053 |
| gaussian | 10 | 0.132 | 0.203 / 0.196 / 0.522 | 7 / 5 / 7 / 25 | 0.00117 / 0.00182 / 0.00121 / 0.0013 | 0.00466 / 0.00748 / 0.00435 / 0.00473 |
| gaussian | 20 | 0.086 | 0.024 / 0.023 / 0.023 | 6 / 1 / 1 / 1 | 0.000917 / 0.000935 / 0.000935 / 0.000935 | 0.00332 / 0.00339 / 0.00339 / 0.00339 |
| plane_wave | 3 | 0.545 | 0.612 / 0.597 / 0.601 | 981 / 307 / 454 / 447 | 0.102 / 0.00161 / 0.00744 / 0.0615 | 59.5 / 0.0386 / 1.18 / 7.63 |
| plane_wave | 5 | 0.666 | 0.625 / 0.614 / 0.611 | 1024 / 1002 / 1012 / 1011 | 0.261 / 0.293 / 0.296 / 0.328 | 459 / 296 / 540 / 419 |
| plane_wave | 10 | 0.654 | 0.836 / 0.771 / 0.693 | 1024 / 1024 / 1024 / 1024 | 0.612 / 0.612 / 0.613 / 0.611 | 1.75e+03 / 1.75e+03 / 1.72e+03 / 1.85e+03 |
| plane_wave | 20 | 1.671 | 5.235 / 3.507 / 1.838 | 1024 / 1024 / 1024 / 1024 | 0.622 / 0.629 / 0.625 / 0.627 | 1.78e+03 / 1.63e+03 / 1.63e+03 / 1.71e+03 |

- d = 20: 2.7-2.8x faster (Rosenbrock 5.41 -> 2.00 s, plane wave 5.24 -> 1.84 s, against the
  reference's 1.66 / 1.67 s), test error unchanged (0.131 -> 0.130, 0.629 -> 0.627).
- d = 10: 1.2x (0.83 -> 0.68 s), Rosenbrock test L2 0.0843 -> 0.0894 (+6%).
- d <= 5: no speedup (p is small), and worse trees: Rosenbrock 5-D 0.0148 -> 0.0309 test L2,
  plane wave 3-D 0.0016 -> 0.0615 (447 instead of 307 leaves), Gaussian 3-D-10-D more leaves.
  Hence the threshold: the affine search only where p is large, the trees elsewhere unchanged.

### 8. FP64 tensor-core Gram kernel (`mma.m8n8k4.f64`): removed

- Bit-identical to `segment_cross_tiled2d` on the GH200 (the FP64 tensor core rounds as the
  scalar fused multiply-add chain does), but only 5% faster: 10.4 vs 11.0 ms for a two-weight
  231 x 231 Gram over 256 nodes of 2310 rows, about 3.8 TFMA/s. The Gram is bound by memory
  traffic, not arithmetic: each chunk of rows is re-read once per 64 x 64 output tile (10 tiles
  for a symmetric 231 x 231 output) and once per weight set.
- Soft search's kernel time at quadratic 20-D (1.95 s, nsys NVTX projection): Gram 0.83 s,
  single-column products Phi^T W y and Phi^T W E 0.38 s, factorization 0.25 s, solves 0.14 s,
  residuals 0.12 s; each is another pass over Phi (N x 231 doubles).

### Profile after change 1 (Rosenbrock quadratic 20-D, library)

Split search 14.0 s of the 15 s build, GPU-bound: `segment_cross_tiled` 6.3 s (the symmetric
231 x 231 Grams: soft search 2 per iteration, bins search 1 per direction, hard SSE 2 per
candidate), `spd_solve_factored` 3.5 s (single right-hand sides, mostly the bins search's 15 cuts
x 2 sides per direction), `spd_factor_coop` 3.3 s (same systems).

## Learnings from every experiment (summary)

Environment and measurement:

- The reference ART's NVRTC kernels do not compile with CuPy 14 + CUDA 12.9 wheels as written
  (`curand_kernel.h` not found, CCCL rejects `--std=c++11`, NVRTC has no `math.h`). A
  `sitecustomize.py` that only changes compiler options (C++17, the toolkit's include trees, two
  shim headers) fixes it without editing the reference (`REFERENCE_ART_ENV.md`).
- A cold CuPy kernel cache costs this library about 88.7 s on its first run (every kernel
  compiled by NVRTC); the reference's first quadratic runs show 2.3-4.7 s warm-ups for the same
  reason. Time only warm runs, with a separate cache directory per side.
- Timed regions must match: the reference's harness excludes its depth-1 warm-up; this library's
  first-use cost (module loads, workspaces) is about 0.4 s. `art_comparison.py` warms up both.
- The benchmark oracle was evaluated on the host for this library's side: 9 ms in total at
  affine 10-D (115k points), but 2.47 s (Gaussian mixture), 1.24 s (Gaussian) and 0.26 s
  (Rosenbrock) at 2.37M points in 20-D. It now uses the reference's GPU oracle.
- Plane wave with `--offset 1` reaches 0 (1 + cos with amplitude 1), so its max pointwise relative
  error (1e7-1e8) is meaningless; compare its L2 errors, or use a larger offset.
- Trees are sensitive to rounding in f: changing the oracle by 2e-16 to 7e-16 relative changed
  the leaf count in 33 of 96 runs (test L2 within 0.92-1.05). GPU and CPU training already make
  different decisions on the d = 8 regression cases at `1f273bc` (8 leaves each, 1812 vs 1834
  evaluations; AFINNI 25 vs 24 leaves).

The lab member's report (reference ~1 s, library up to ~7 s at depth 10):

- Mainly a settings mismatch the comparison script did not catch: the reference's presets
  changed to 20 iterations and multiplier 10 on 2026-11-18, the library's `soft_iters` default
  stayed 200, and the script passed the multiplier but not `--max-iters`. On the GH200 that
  mismatch makes the library 1.9-4.9x slower. At equal iteration caps the two are on par, and at
  200 iterations the reference is the slower one on Rosenbrock and plane wave (2.8-5.8 s vs
  1.6-1.8 s), though not on Gaussian 3-D / 5-D (0.33 / 0.95 s vs 1.01 / 1.67 s). The rest of the 7 s
  is most likely his slower host (the soft search is launch- and synchronization-bound); his
  hardware is not recorded.
- Nearly every node runs the soft search to its iteration cap, at 20 and at 200 iterations, on
  both sides.

Performance (Rosenbrock quadratic 20-D unless stated):

- Where the time went at the start (32.9 s): the leaf fit's condition-number inverse in
  `lstsq_auto` (231 right-hand sides solved serially by one thread per system, 22 s of GPU time),
  the 231 x 231 Grams (6.3 s), factorizations (3.3 s) and single-right-hand-side solves (3.5 s).
- Exact kernel work (changes 1, 3, 4) reached 8.0 s with bit-identical results; one exact idea
  failed (change 2: mirroring L^T for row-wise back substitution was slower, 3.5 -> 5.2 s).
- Back substitution in the exact operation order is a serial chain of about p^2 / 2 dependent
  products per system; at p = 231 and a few thousand systems per call that chain, not
  throughput, sets the time. Only a different order (change 5, relaxed) removes it: 8.0 -> 5.4 s.
- The FP64 tensor cores (`mma.m8n8k4.f64`) give the scalar fused multiply-add chain's bits on the
  GH200, but did not help (5%): the Grams are bound by memory traffic (each chunk re-read per
  output tile and per weight set), not arithmetic. The soft search as a whole is bound by
  repeated passes over the N x 231 feature matrix.
- The bins search is work the reference does not do: about 180 factorizations of 231 x 231
  systems per node per split. Affine fits in the bins and soft searches from 100 leaf
  coefficients (change 7) took 5.4 -> 1.8 s with unchanged test error at d = 20, but make trees
  worse at d <= 5 (up to 38x on plane wave 3-D), hence the threshold.
- Small trees (a few dozen to a few hundred leaves) cost about 65 ms per level against the
  reference's 29 ms (launch-bound fixed costs in the bins search, soft search, metric basis, fit
  and sampling). Closing that is worth 0.2-0.4 s per run, below AGENTS.md's bar of 2 s or 10x.
- Peak GPU memory follows the batch-size budget (`ARTPlan.bytes_per_point`), not need: 31.7 GB
  -> 24.8 GB at quadratic 20-D against the reference's 2.9 GB.

Not done from the original task:

- The reference was not profiled with nsys (only timed); the library was, per phase and kernel.
- `ncu` was not used; nsys kernel times and NVTX projections were enough to find the bottlenecks.
- Deep trees (depth 15-20) on both sides, as the lab member asked; dimensions above 20.
- The proposal to change `ARTSettings` defaults (multiplier 50, 200 soft iterations) to the
  reference's presets (10, 20): measured above (2.2-2.9x faster at 20 iterations, errors mixed),
  not changed.

## Open

- Small trees: 2-4x the reference in time (+0.2-0.4 s), from per-level fixed costs.
- Quadratic leaves at d = 20: 1.07-1.12x the reference in time after all changes, but 24.8 GB of
  GPU memory against 2.9 GB.
- Gaussian mixture, quadratic, 20-D, tol 0.1: 2.1x the reference (37 leaves against 11).
- Whether to adopt the relaxed contract (change 5) and the affine search (change 7); the
  GPU `regression_check` tolerance for `art_quadratic_d8` if change 5 is adopted.
- Deep trees (depth 15-20) and d > 20 on both sides with the same parameters.
- The doc follow-ups listed in the hand-off section (companion, USAGE, ARCHITECTURE).

## Scripts and data (in `scratch/`, not committed; on the Vista work file system)

- `reference_env/`: the reference ART's NVRTC wrapper and shim headers.
- `snapshots/base`: `src/` and `scripts/` at `1f273bc` with the fixed comparison script (the
  "original" variant); other snapshots hold intermediate states used for A/B runs.
- `scripts/final_matrix.py`, `final_table.py`: the three-way comparison and its tables;
  `timing_matrix.py`, `timing_table.py`: the two-way matrices; `iter_cap.sh`, `ref_iter200.sh`:
  the iteration-cap study; `screen_ab.sh`, `soft_ab.sh`: the affine-search A/B runs; `ab.sh`:
  interleaved A/B of two checkouts.
- `scripts/save_models.sh`, `compare_models.py`, `compare_models_tol.py`: complete-model
  bit-for-bit and tolerance checks; `afinni_check.py`: the AFINNI d = 8 cases;
  `regression_d8.sh`: regression check with new baselines; `check_tiled2d.py`, `check_dmma.py`:
  kernel unit checks; `oracle_check.py`: GPU vs host oracle.
- `scripts/profile_case.sh`: nsys profile with NVTX, kernel and API summaries;
  `split_stage_times.py`: GPU-synchronized split-stage times (and NVTX ranges per stage).
- `results/`: every JSONL record and table; `profiles/`: the nsys reports; `models/`: saved models;
  `regression/`: regression-check baselines; `results/art_parity.patch`: the patch below.

## Appendix: the patch

`git diff` against `refinement-engine` at `1f273bc`.

```diff
diff --git a/docs/USAGE.md b/docs/USAGE.md
index c86ffdd..2720eb5 100644
--- a/docs/USAGE.md
+++ b/docs/USAGE.md
@@ -242,9 +242,11 @@ batch size, rank count and steal schedule, and the same in every run on one back
 rounding level). `scripts/art_comparison.py` trains the
 reference ART code and this ART on the reference's benchmark problems and scores both on the
 same test set; pass `--sampler parallel_numba` to time against the reference's fastest CPU
-configuration, and add `--traversal bfs --optimizer cupy_batched_cholesky` for its fastest GPU
-configuration. `--skip-reference` / `--skip-library` run one side only; this ART alone also runs
-under `mpiexec`. On the GPU,
+configuration, and add `--traversal bfs --optimizer cupy_batched_cholesky --gpu-sampling` for its
+fastest GPU configuration. Both sides build the same tree to depth 1 before the timed build (the
+reference's `--warmup`; `--no-warmup` skips it on both), so `build_seconds` excludes kernel loads
+and workspace allocation on either side. `--skip-reference` / `--skip-library` run one side only;
+this ART alone also runs under `mpiexec`. On the GPU,
 this ART's soft split search is 1-30x faster than the reference's GPU configuration, and faster than this
 ART on the CPU whenever the CPU training takes more than about 1.5 s.
 
diff --git a/scripts/art_comparison.py b/scripts/art_comparison.py
index 7fd3fbd..75eef0a 100644
--- a/scripts/art_comparison.py
+++ b/scripts/art_comparison.py
@@ -7,7 +7,8 @@ The target function, its domain and the test set all come from an ART checkout
 both models are scored by ART's ``evaluate_tree_benchmark`` on the same independent
 uniform test points (``10000 * d`` by default, seed ``--test-seed``). This library's ART uses
 the reference's settings: affine leaves, the pointwise relative stopping rule with ART's
-``--error-tolerance``, ``--max-depth`` levels and ART's ``--sample-multiplier``, starting
+``--error-tolerance``, ``--max-depth`` levels, ART's ``--sample-multiplier`` and, as its soft
+search's iteration cap, ART's ``--max-iters`` (``--library-soft-iters`` overrides it), starting
 from ``ceil(multiplier * (d + 1))`` uniform points (ART's exact box root).
 
 Arguments not listed below are passed to ART's benchmark parser unchanged, so ART runs
@@ -20,12 +21,24 @@ GPU). Compare against that one when timing matters, e.g.::
         --art-path ../Adaptive-Regression-Trees-ART- --benchmark quadratic --dimension 2 \\
         --sampler parallel_numba
 
-ART's fastest GPU configuration is ``--traversal bfs --optimizer cupy_batched_cholesky``
-(resident batched soft optimizer; sampling, oracle and node fits stay on the CPU).
+ART's fastest GPU configuration is ``--traversal bfs --optimizer cupy_batched_cholesky
+--gpu-sampling`` (resident batched soft optimizer, hit-and-run, node fits and, for Rosenbrock and
+the Gaussians, the oracle on the GPU); without ``--gpu-sampling`` its sampling, oracle and node
+fits stay on the CPU.
 ``--skip-reference`` / ``--skip-library`` run one side only. This library's ART alone also runs
 under ``mpiexec`` (``--skip-reference``; ``--library-work-stealing`` enables stealing); rank 0
 prints.
 
+Both sides time the same thing. Under the reference's ``--warmup`` (its default; ``--no-warmup``
+turns it off for both), each side first builds the same tree to depth 1 in the same process,
+which loads its compiled kernels and allocates its workspaces, and ``build_seconds`` then times
+only the full build that follows: the reference's own ``build_seconds``, and this library's
+``AdaptiveSurrogate.train``. ``warmup_seconds`` reports the warm-up (``null`` without one).
+
+On the GPU, this library's ART evaluates Rosenbrock, Gaussian and Gaussian-mixture targets with the
+reference's own GPU oracle (``_device_oracle``), as the reference does with ``--gpu-sampling``;
+other targets, or ``--library-host-oracle``, evaluate on the host.
+
 Leaf degree 1 maps to ``affine_lsq`` and degree 2 to ``quadratic_lsq`` (both with an
 intercept). Refuses other degrees, ART leaves without an intercept, error metrics other than
 ``max_pointwise_relative`` and relative-error floors other than 1e-12, which this library's
@@ -42,19 +55,54 @@ import math
 import sys
 import time
 import warnings
+from dataclasses import replace
 from pathlib import Path
 
 import numpy as np
 
 sys.path.insert(0, str(Path.cwd()))
 
-from src import to_host, xp  # noqa: E402
+from src import IS_CUPY_BACKEND, to_host, xp  # noqa: E402
 from src.config import ARTSettings, TrainingConfig  # noqa: E402
 from src.core.evaluation import Evaluator  # noqa: E402
 from src.infra.backend import MPI  # noqa: E402
 from src.surrogate import AdaptiveSurrogate  # noqa: E402
 
 
+def _device_oracle(target, d: int):
+    """
+    The benchmark on the GPU through the reference ART's own oracle workspace (Rosenbrock,
+    Gaussian and Gaussian mixture, with their offsets), as the reference evaluates it with
+    ``--gpu-sampling``; None for other targets or on NumPy. Points go through in chunks of a
+    fixed size, padded, so every call has the same shape and a point's value never depends on
+    the other points of its batch.
+    """
+    if not IS_CUPY_BACKEND or not callable(getattr(target, "cuda_oracle_spec", None)):
+        return None
+    import cupy as cp
+    from art._cuda_oracles import CudaOracleWorkspace
+
+    n = 1 << 16
+    try:
+        ws = CudaOracleWorkspace(target, 1, n, d, cp.cuda.get_current_stream())
+    except ValueError:
+        return None
+    X_buf, y_buf, inherited = cp.zeros((1, n, d)), cp.empty((1, n)), np.zeros(1, dtype=np.int32)
+
+    def f(x):
+        x = cp.asarray(x, dtype=cp.float64)
+        out = cp.empty(int(x.shape[0]))
+        for s in range(0, int(x.shape[0]), n):
+            m = min(n, int(x.shape[0]) - s)
+            X_buf[0, :m] = x[s : s + m]
+            X_buf[0, m:] = 0.0
+            ws.evaluate_new(X_buf, y_buf, inherited)
+            out[s : s + m] = y_buf[0, :m]
+        return out[:, None]
+
+    return f
+
+
 class _LibraryARTModel:
     """The reference ART's evaluator only needs ``predict``; route on the active backend, return host values."""
 
@@ -89,8 +137,12 @@ def main() -> None:
     own.add_argument("--library-roots", default="box", choices=("box", "grid", "voronoi"))
     own.add_argument("--library-validation", type=int, default=0)
     own.add_argument("--library-walk-steps", type=int, default=20)
+    own.add_argument("--library-host-oracle", action="store_true",
+                     help="evaluate the benchmark on the host for this library's ART even where the GPU can")
     own.add_argument("--library-search", default="soft", choices=("gap", "bins", "soft"))
-    own.add_argument("--library-soft-iters", type=int, default=None, help="soft split search iterations (default: ARTSettings.soft_iters)")
+    own.add_argument("--library-affine-search-from", type=int, default=None,
+                     help="ARTSettings.affine_search_from (default: its default); 0: never")
+    own.add_argument("--library-soft-iters", type=int, default=None, help="soft split search iterations (default: the reference's --max-iters)")
     mine, art_argv = own.parse_known_args()
     sys.path.insert(0, str(mine.art_path.resolve()))
 
@@ -154,9 +206,7 @@ def main() -> None:
     if comm.Get_size() > 1 and not mine.skip_reference:
         raise SystemExit("the reference ART runs in one process: pass --skip-reference under mpiexec")
     if not mine.skip_reference:
-        t = time.perf_counter()
         run = build_tree_benchmark(args, domain, target)
-        seconds = time.perf_counter() - t
         evaluation, frac = evaluate(run.tree)
         from art.tree import LeafNode
 
@@ -165,16 +215,19 @@ def main() -> None:
             if isinstance(node, LeafNode):
                 statuses[str(node.status)] = statuses.get(str(node.status), 0) + 1
         print("COMPARE " + json.dumps({"method": "reference ART", **common, **_summary(
-            evaluation, run.tree.num_leaves(), run.tree.max_depth(), int(run.build_result.oracle_queries), seconds,
-            {"frac_gt_tol": frac, "terminal_reasons": statuses,
+            evaluation, run.tree.num_leaves(), run.tree.max_depth(), int(run.build_result.oracle_queries),
+            run.build_seconds,
+            {"frac_gt_tol": frac, "warmup_seconds": run.warmup_seconds, "terminal_reasons": statuses,
              "art_config": {"sampler": args.sampler, "optimizer": args.optimizer, "traversal": args.traversal}},
         )}), flush=True)  # fmt: skip
     if mine.skip_library:
         return
 
-    def f(x):
+    def f_host(x):
         return xp.asarray(np.asarray(target(np.asarray(to_host(x))), dtype=np.float64).reshape(-1, 1))
 
+    f = (None if mine.library_host_oracle else _device_oracle(target, args.dimension)) or f_host
+
     n_root = math.ceil(args.sample_multiplier * (args.dimension + 1))
     X0 = uniform(n_root, np.random.default_rng(args.seed))
     config = TrainingConfig(
@@ -185,8 +238,16 @@ def main() -> None:
         art=ARTSettings(roots=mine.library_roots, sample_multiplier=args.sample_multiplier,
                         validation_samples=mine.library_validation, walk_steps=mine.library_walk_steps,
                         split_search=mine.library_search,
-                        **({} if mine.library_soft_iters is None else {"soft_iters": mine.library_soft_iters})),
+                        **({} if mine.library_affine_search_from is None else
+                           {"affine_search_from": mine.library_affine_search_from or None}),
+                        soft_iters=args.max_iters if mine.library_soft_iters is None else mine.library_soft_iters),
     )  # fmt: skip
+    warmup_seconds = None
+    if args.warmup:
+        comm.Barrier()
+        t = time.perf_counter()
+        AdaptiveSurrogate().train(X0, replace(config, max_levels=min(config.max_levels, 1)), Evaluator(f_fn=f))
+        warmup_seconds = time.perf_counter() - t
     model = AdaptiveSurrogate()
     comm.Barrier()
     t = time.perf_counter()
@@ -203,6 +264,7 @@ def main() -> None:
         "reference": {"leaf_degree": args.leaf_degree, "leaf_include_bias": bias, "error_metric": args.error_metric,
                       "relative_error_floor": args.relative_error_floor, "sampler": args.sampler},
         "library_art": {"leaf_model": leaf_model, "sampling": config.art.sampling,
+                        "oracle": "host" if f is f_host else "gpu (the reference's CudaOracleWorkspace)",
                         "acceptance": "pointwise", "relative_error_floor": 1e-12,
                         "split_search": config.art.split_search, "roots": config.art.roots},
     }  # fmt: skip
@@ -214,7 +276,8 @@ def main() -> None:
     depth = int(max(int(p).bit_length() - 1 for p in model.art_leaves["locational_code"]))
     print("COMPARE " + json.dumps({"method": "this library's ART", **common, **_summary(
         evaluation, int(model.N_src), depth, int(model.profiling_stats["total_f_evals"]), seconds,
-        {"frac_gt_tol": frac, "terminal_reasons": model.leaf_status_counts()["terminal_reason"],
+        {"frac_gt_tol": frac, "warmup_seconds": warmup_seconds,
+         "terminal_reasons": model.leaf_status_counts()["terminal_reason"],
          "split_strategy": model.config["components"]["split_strategy"]["name"],
          "split_search": mine.library_search, "art_counters": model.profiling_stats.get("art", {})},
     )}), flush=True)  # fmt: skip
diff --git a/scripts/regression_check.py b/scripts/regression_check.py
index feece76..60f510e 100644
--- a/scripts/regression_check.py
+++ b/scripts/regression_check.py
@@ -4,7 +4,8 @@ Differential regression check for the training/inference pipeline.
 Trains small, fixed problems through the public ``AdaptiveSurrogate`` API for every
 ``SurrogateMode`` x error metric x output dimension (plus a few extra paths: oracle,
 precomputed Y, uniform-grid roots, second-order Taylor, tiny batches) and ART
-(every split search, root type, sampler, quadratic leaves, fallback leaves), predicts on
+(every split search, root type, sampler, quadratic leaves, fallback leaves), plus quadratic
+leaves at d = 8 for both plans (the GPU's large-system kernels), predicts on
 fixed queries, and stores every array a saved model holds, every prediction, trust mask
 and counter.
 
@@ -85,6 +86,11 @@ def wavy(fd):
     return f
 
 
+def wavy_8d(x):
+    """An 8-D function for the cases whose quadratic leaves (45 coefficients) take the GPU's large-system kernels."""
+    return (xp.sin(x @ xp.linspace(0.5, 1.5, 8)) + 0.5 * xp.sum(x**2, axis=1) + 2.0)[:, None]
+
+
 def model_arrays(s: AdaptiveSurrogate) -> dict[str, np.ndarray]:
     """Every numeric array ``save_model`` would write (leaves, local-model parameters, routing, leaf status)."""
     out = {k: getattr(s, k) for k in ("X_src", "f_src", "E_cluster_abs_src", "E_cluster_rel_src", "counts_src")}
@@ -131,6 +137,12 @@ def cases():
     yield "art_quadratic", 1, dict(art_base, local_model="quadratic_lsq", art=ARTSettings(**rs))
     yield "art_tiny_batches", 1, dict(art_base, batch_size=97, art=ARTSettings(**rs, soft_iters=5))
     yield "art_fallback_leaves", 2, dict(art_base, max_levels=3, art=ARTSettings(**rs, soft_iters=5))
+    # d = 8 with quadratic leaves: 45 coefficients, above SPD_COOP_MIN_P, so the GPU factors,
+    # solves and forms Grams with the large-system kernels the 2-D cases never reach.
+    yield "art_quadratic_d8", 1, dict(art_base, _dim=8, domain=([-1.0] * 8, [1.0] * 8), max_levels=3,
+                                      local_model="quadratic_lsq", art=ARTSettings(sample_multiplier=3, soft_iters=5))
+    yield "afinni_quadratic_d8", 1, dict(base, _dim=8, local_model="quadratic_lsq", min_pts_per_cluster=60,
+                                         max_levels=4)
 
 
 def run(out_prefix):
@@ -138,6 +150,8 @@ def run(out_prefix):
     rng = np.random.default_rng(82)
     X = xp.asarray(rng.uniform(-1, 1, (512, 2)))
     Q = xp.asarray(rng.uniform(-1.2, 1.2, (257, 2)))
+    X8 = xp.asarray(rng.uniform(-1, 1, (1500, 8)))
+    Q8 = xp.asarray(rng.uniform(-1.2, 1.2, (257, 8)))
     arrays, report = {}, {}
     for name, fd, kwargs in cases():
         art_mode = kwargs.get("algorithm") == "art"
@@ -146,21 +160,24 @@ def run(out_prefix):
             f = wavy(fd)
         oracle, use_Y = kwargs.pop("_oracle", False), kwargs.pop("_Y", False)
         fns = {"oracle_fn": lambda x, f=f, g=g: (f(x), g(x))} if oracle else {"f_fn": f, "g_fn": g}
+        X_case, Q_case = X, Q
+        if kwargs.pop("_dim", 2) == 8:  # function values only: the 2-D gradient and Hessian do not apply
+            f, h, fns, X_case, Q_case = wavy_8d, None, {"f_fn": wavy_8d}, X8, Q8
         t0 = time.perf_counter()
         try:
             s = AdaptiveSurrogate()
-            s.train(X, TrainingConfig(**kwargs), Evaluator(hessian_fn=h, **fns), Y_train=f(X) if use_Y else None)
+            s.train(X_case, TrainingConfig(**kwargs), Evaluator(hessian_fn=h, **fns), Y_train=f(X_case) if use_Y else None)
             train_time = time.perf_counter() - t0
             for attr, v in model_arrays(s).items():
                 arrays[f"{name}|{attr}"] = v.reshape(len(v), -1) if v.ndim else v
             if art_mode:
-                pred, mask = s.predict(Q)  # routed down the tree
+                pred, mask = s.predict(Q_case)  # routed down the tree
                 arrays[f"{name}|pred_route"], arrays[f"{name}|mask_route"] = to_host(pred), to_host(mask)
                 counters = {**s.profiling_stats.get("art", {}), **s.profiling_stats.get("terminal_reasons", {})}
                 arrays[f"{name}|art_counters"] = np.asarray([counters[k] for k in sorted(counters)], dtype=np.int64)
             else:
                 for k, coarse, strict in ((1, 10, False), (4, 10, False), (4, 10, True), (2000, 2000, False)):
-                    pred, mask = s.predict(Q, k_nn=k, coarse_factor=coarse, strict_trust_region=strict, batch_size=37)
+                    pred, mask = s.predict(Q_case, k_nn=k, coarse_factor=coarse, strict_trust_region=strict, batch_size=37)
                     arrays[f"{name}|pred_{k}_{strict}"] = to_host(pred)
                     arrays[f"{name}|mask_{k}_{strict}"] = to_host(mask)
             ps = s.profiling_stats
diff --git a/src/art/plan.py b/src/art/plan.py
index d8675be..1bd6a22 100644
--- a/src/art/plan.py
+++ b/src/art/plan.py
@@ -419,11 +419,12 @@ class ARTPlan:
             return lead, xp.where(ok, z, xp.nan)
         rows = self._split_rows(data, U)
         bank = self._direction_bank(rows, lead, V, M, width, keys)
-        w, z, sse = self._bins_search(rows, bank)
+        search_rows = self._search_rows(rows)
+        w, z, sse = self._bins_search(rows, bank, search_rows)
         if mode == "soft":
             if progress is not None:
                 progress()
-            w, z = self._soft_refine(rows, w, z, sse, progress)
+            w, z = self._soft_refine(rows, w, z, sse, progress, search_rows)
         return w, z
 
     def _metric_basis(self, data: SplitData, splitter) -> tuple[Array, Array]:
@@ -476,14 +477,27 @@ class ARTPlan:
             dirs.extend(unit_rows(g[:, i]) for i in range(n_rand))
         return xp.stack(dirs[:D], axis=1)
 
-    def _bins_search(self, rows: SplitRows, bank: Array) -> tuple[Array, Array, Array]:
+    def _search_rows(self, rows: SplitRows) -> SplitRows:
+        """
+        The rows whose fits rank the bins search's cuts and drive the soft search: ``rows`` (the
+        leaf model's features), or from ``ARTSettings.affine_search_from`` coefficients the same
+        points with affine features ``[u, 1]``, whose fits cost ``(d + 1)^3`` instead of ``p^3``
+        per system. Candidates are always judged by ``rows``' hard SSE.
+        """
+        start = self.settings.affine_search_from
+        if start is None or self.n_coefficients < start or rows.Phi.shape[1] == self.d + 1:
+            return rows
+        Phi = xp.concatenate([rows.U, xp.ones((int(rows.U.shape[0]), 1))], axis=1)
+        return SplitRows(rows.U, rows.Y, Phi, rows.labels, rows.counts, ridge=rows.ridge)
+
+    def _bins_search(self, rows: SplitRows, bank: Array, search_rows: SplitRows) -> tuple[Array, Array, Array]:
         """
         The bins search: along every bank direction, evaluate ``n_bins - 1`` quantile cut positions
-        from per-bin sums (``SplitRows.cut_sums``: one data pass per direction), keep the
-        ``n_screened`` best per node as a screen, then recompute their SSE directly
-        (``hard_sse``) and return the best ``(w, z)`` and its SSE. Only cuts between
-        distinct scores that leave ``min_side`` points per side are candidates. The SSE is
-        the hard objective ``SSE(a, z)`` of the companion.
+        from per-bin sums of ``search_rows`` (``SplitRows.cut_sums``: one data pass per direction),
+        keep the ``n_screened`` best per node as a screen, then recompute their SSE directly
+        with the leaf model's ``rows`` (``hard_sse``) and return the best ``(w, z)`` and its SSE.
+        Only cuts between distinct scores that leave ``min_side`` points per side are candidates.
+        The SSE is the hard objective ``SSE(a, z)`` of the companion.
         """
         K, D, _ = bank.shape
         nb = self.settings.n_bins
@@ -504,10 +518,10 @@ class ARTPlan:
             ok = eligible[:, None] & (pos >= self.min_side) & (n[:, None] - pos >= self.min_side) & (ss[at] > ss[at - 1])
             # Per node only (batched products over its own sorted rows), so nothing
             # depends on the other nodes of the batch.
-            GL, BL, CL, GR, BR, CR = rows.cut_sums(order, pos)
+            GL, BL, CL, GR, BR, CR = search_rows.cut_sums(order, pos)
             sse = xp.zeros((K, nb - 1))
             for Gs, Bs, Cs in ((GL, BL, CL), (GR, BR, CR)):
-                beta = rows.solve(Gs, Bs)
+                beta = search_rows.solve(Gs, Bs)
                 sse = sse + Cs - 2 * xp.sum(beta * Bs, axis=(2, 3)) + xp.sum(beta * det_matmul(Gs, beta), axis=(2, 3))
             screen[:, j] = xp.where(ok, sse, xp.inf)
             z_all[:, j] = ss[at]
@@ -529,10 +543,12 @@ class ARTPlan:
             best_sse = xp.minimum(sse, best_sse)
         return best_w, best_z, best_sse
 
-    def _soft_refine(self, rows: SplitRows, w0: Array, z0: Array, sse0: Array, progress) -> tuple[Array, Array]:
+    def _soft_refine(self, rows: SplitRows, w0: Array, z0: Array, sse0: Array, progress,
+                     search_rows: SplitRows) -> tuple[Array, Array]:  # fmt: skip
         """
         The soft search: polish each bins-search plane by projected gradient descent on the consistent soft
-        objective ``J(w, z) = min_B F`` (``J_h`` in the companion), with a fixed ridge (0 for
+        objective ``J(w, z) = min_B F`` (``J_h`` in the companion) of the children's fits on
+        ``search_rows`` (``rows``, or their affine features, see ``_search_rows``), with a fixed ridge (0 for
         ``leaf_fit="lstsq"``), a
         fixed gate width ``T = soft_temperature * std(w^T u)`` (``h`` in the companion), floored
         weights ``W_R = delta + (1 - 2 delta) sigma(s)``, ``W_L = delta + (1 - 2 delta) sigma(-s)``
@@ -545,8 +561,8 @@ class ARTPlan:
         one gate width could not beat the margin), ``soft_line_search_stalled`` (no trial
         qualified otherwise), ``soft_nonfinite`` (objective or gap not finite) or
         ``soft_iteration_limit``. The result is hardened (``z`` moved to the first
-        right-side score) and kept only if its directly refit hard SSE beats the bins-search
-        incumbent's; otherwise the incumbent is returned.
+        right-side score) and kept only if its directly refit hard SSE on ``rows`` (the leaf
+        model's) beats the bins-search incumbent's; otherwise the incumbent is returned.
         """
         s_cfg = self.settings
         delta = s_cfg.soft_weight_floor
@@ -566,7 +582,7 @@ class ARTPlan:
         active = ok0.copy()
 
         def gate(w_, z_):
-            return soft_gates(rows, w_, z_, T, delta)
+            return soft_gates(search_rows, w_, z_, T, delta)
 
         reasons = ("soft_nonfinite", "soft_gate_saturated", "soft_gradient_converged", "soft_line_search_stalled")
         stops = xp.zeros(len(reasons), dtype=xp.uint64)  # summed on the device, read once at the end
@@ -575,9 +591,9 @@ class ARTPlan:
                 break
             if progress is not None and it % 4 == 3:
                 progress()
-            J, g_w, g_z, rL, rR, pen, gap, sp_sum = soft_objective(rows, w, z, T, delta)
+            J, g_w, g_z, rL, rR, pen, gap, sp_sum = soft_objective(search_rows, w, z, T, delta)
             if IS_CUPY_BACKEND:  # fused step: w, z, active and stops updated in place, no host reads
-                soft_step(rows, w, z, g_w, g_z, J, gap, pen, rL, rR, sp_sum, T, n, delta, active, stops, 12)
+                soft_step(search_rows, w, z, g_w, g_z, J, gap, pen, rL, rR, sp_sum, T, n, delta, active, stops, 12)
                 continue
             g_w = g_w - xp.sum(g_w * w, axis=1, keepdims=True) * w  # tangent to |w| = 1
             gn2 = xp.sum(g_w**2, axis=1) + g_z**2
@@ -591,7 +607,7 @@ class ARTPlan:
                 w_t = unit_rows(w - alpha[:, None] * g_w)
                 z_t = z - alpha * g_z
                 WLt, WRt, _ = gate(w_t, z_t)
-                Ft = (rows.reduce(WLt * rL + WRt * rR) + pen) / n
+                Ft = (search_rows.reduce(WLt * rL + WRt * rR) + pen) / n
                 accept = active & finite & ~taken & (Ft <= J - 1e-4 * alpha * gn2) & (J - Ft > margin)
                 w = xp.where(accept[:, None], w_t, w)
                 z = xp.where(accept, z_t, z)
diff --git a/src/components/local_models.py b/src/components/local_models.py
index 14628db..04243df 100644
--- a/src/components/local_models.py
+++ b/src/components/local_models.py
@@ -376,8 +376,9 @@ class QuadraticLSQModel(LocalModel):
             return {"coef": xp.swapaxes(lstsq_auto(G, B, Phi, data.F, data.starts, data.counts), 1, 2), "radius": radius}
         lam = 1e-10 * xp.trace(G, axis1=1, axis2=2) / p + 1e-300
         # spd_solve: on the GPU its bits do not depend on the batch (a batched linalg.solve picks its
-        # algorithm by batch count, so cold and warm runs fitted different coefficients); NumPy: linalg.solve
-        beta = spd_solve(G + lam[:, None, None] * xp.eye(p), B)  # (K, p, fd)
+        # algorithm by batch count, so cold and warm runs fitted different coefficients); NumPy: linalg.solve.
+        # exact: the leaf's coefficients keep the exact substitution order, whatever the split search uses.
+        beta = spd_solve(G + lam[:, None, None] * xp.eye(p), B, exact=True)  # (K, p, fd)
         return {"coef": xp.swapaxes(beta, 1, 2), "radius": radius}
 
     def evaluate(self, params, X, cluster_ids):
diff --git a/src/config.py b/src/config.py
index 343e802..f8f0068 100644
--- a/src/config.py
+++ b/src/config.py
@@ -116,6 +116,14 @@ class ARTSettings:
     n_bins: int = 16
     n_directions: int = 6
     n_screened: int = 3
+    # From this many leaf coefficients p, the bins search ranks its cuts and the soft search
+    # optimizes its plane with affine fits ([u, 1], d + 1 coefficients) instead of the leaf
+    # model's; the screened candidates and the soft result are still judged by the leaf model's
+    # hard SSE. A quadratic leaf's p^3 work per fitted system becomes (d + 1)^3 (231 -> 21 at
+    # d = 20). Below about 100 coefficients the saving is small and the trees can get worse (up
+    # to several times the test error for quadratic leaves at d <= 5); at d = 20 the test error is
+    # unchanged on the reference ART's benchmarks. None: always the leaf model's fits.
+    affine_search_from: int | None = 100
     # The reference ART's iteration cap. Against 40, on its benchmark problems at d >= 3, it builds
     # 5-30% fewer leaves for the same test error (4x fewer on a 5-D plane wave), and so fewer
     # oracle calls, for about 2-3x the training time where the trees are about the same size.
@@ -150,6 +158,10 @@ class ARTSettings:
                 raise ValueError(f"ARTSettings.{name} must be one of {choices}; got {getattr(self, name)!r}.")
         if self.sampling == "convex_combination" and self.validation_samples:
             raise ValueError("validation_samples needs sampling='hit_and_run' (fresh points inside each node).")
+        if self.affine_search_from is not None and (
+            not isinstance(self.affine_search_from, numbers.Integral) or self.affine_search_from < 1
+        ):
+            raise ValueError(f"affine_search_from must be None or an integer >= 1; got {self.affine_search_from!r}.")
         for name, low in (("n_bins", 2), ("n_directions", 1), ("n_screened", 1), ("soft_iters", 0)):
             value = getattr(self, name)
             if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < low:
diff --git a/src/kernels/segment_cross.cu b/src/kernels/segment_cross.cu
index 6a27ef0..641e024 100644
--- a/src/kernels/segment_cross.cu
+++ b/src/kernels/segment_cross.cu
@@ -136,6 +136,82 @@ extern "C" __global__ void segment_cross_tiled(const double* A, const double* B,
     }
 }
 
+// The same partial sums as segment_cross_partial for large P x Q outputs (Grams of many
+// features), one block per (weight set, chunk, 64 x 64 output tile), 16 x 16 threads each
+// owning a 4 x 4 register tile. Per staged row a thread reads 4 values of A and 4 of B from
+// shared memory for 16 products, where segment_cross_tiled reads 2 per product, which made it
+// shared-memory-bound. Entry (r, c) still sums ((A - a_off) w)_r (B - b_off)_c over the chunk's
+// rows in row order, carried across the staged row tiles, so its bits are those of the other
+// two kernels. Only entries with eidx[r * Q + c] >= 0 are written (the upper triangle if
+// symmetric); the partials keep the untiled kernel's (weight set, chunk, entry) layout.
+extern "C" __global__ void segment_cross_tiled2d(const double* A, const double* B, const double* w,
+                                                 const double* a_off, const double* b_off, const long long* cum,
+                                                 const long long* starts, const long long* counts,
+                                                 const int* tile_r, const int* tile_c, const int* eidx,
+                                                 double* partial, long long n_chunks, int K, int E, int P, int Q,
+                                                 int CH, int flags, long long w_stride, long long b_stride) {
+    const int TR = 32;  // rows staged per pass: 2 x 32 x 64 doubles = 32 KB of shared memory
+    __shared__ double sA[TR][64], sB[TR][64];
+    // 1. Block -> (weight set, chunk) and output tile; the chunk's rows as in the untiled kernel.
+    long long sj = blockIdx.x;
+    long long sw = sj / n_chunks, j = sj - sw * n_chunks;
+    if (j >= cum[K - 1]) return;
+    int lo = 0, hi = K - 1;
+    while (lo < hi) {
+        int mid = (lo + hi) >> 1;
+        if (cum[mid] > j)
+            hi = mid;
+        else
+            lo = mid + 1;
+    }
+    long long k = lo;
+    long long first = cum[k] - (counts[k] + CH - 1) / CH;
+    long long r0 = starts[k] + (j - first) * CH;
+    long long r1 = min(r0 + CH, starts[k] + counts[k]);
+    w += sw * w_stride;
+    B += sw * b_stride;
+    const int R0 = tile_r[blockIdx.y] * 64, C0 = tile_c[blockIdx.y] * 64;
+    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
+    double acc[4][4];
+    for (int u = 0; u < 4; ++u)
+        for (int v = 0; v < 4; ++v) acc[u][v] = 0.0;
+    // 2. Stage TR rows of the tile's A and B columns (offset- and weight-adjusted as in the other
+    //    kernels; columns past P or Q as 0, never written), then accumulate in row order.
+    for (long long base = r0; base < r1; base += TR) {
+        int nrow = (int)min((long long)TR, r1 - base);
+        for (int t = threadIdx.x; t < TR * 64; t += blockDim.x) {
+            int row = t >> 6, col = t & 63;
+            double a = 0.0, b = 0.0;
+            if (row < nrow && R0 + col < P) {
+                a = A[(base + row) * P + R0 + col] - ((flags & 2) ? a_off[k * P + R0 + col] : 0.0);
+                if (flags & 1) a = a * w[base + row];
+            }
+            if (row < nrow && C0 + col < Q)
+                b = B[(base + row) * Q + C0 + col] - ((flags & 4) ? b_off[k * Q + C0 + col] : 0.0);
+            sA[row][col] = a;
+            sB[row][col] = b;
+        }
+        __syncthreads();
+        for (int row = 0; row < nrow; ++row) {
+            double a[4], b[4];
+            for (int u = 0; u < 4; ++u) a[u] = sA[row][ty + 16 * u];
+            for (int v = 0; v < 4; ++v) b[v] = sB[row][tx + 16 * v];
+            for (int u = 0; u < 4; ++u)
+                for (int v = 0; v < 4; ++v) acc[u][v] += a[u] * b[v];
+        }
+        __syncthreads();
+    }
+    // 3. One partial per computed entry, at its index in the entry list.
+    for (int u = 0; u < 4; ++u)
+        for (int v = 0; v < 4; ++v) {
+            int r = R0 + ty + 16 * u, c = C0 + tx + 16 * v;
+            if (r < P && c < Q) {
+                int e = eidx[r * Q + c];
+                if (e >= 0) partial[sj * E + e] = acc[u][v];
+            }
+        }
+}
+
 // One thread per (weight set, segment, entry): the segment's chunk partials summed in chunk
 // order; with sym, also written to the mirrored entry.
 extern "C" __global__ void segment_cross_reduce(const double* partial, const long long* cum,
diff --git a/src/kernels/small_linalg.cu b/src/kernels/small_linalg.cu
index d4984b8..929eb7f 100644
--- a/src/kernels/small_linalg.cu
+++ b/src/kernels/small_linalg.cu
@@ -19,22 +19,25 @@ __device__ bool chol_inplace(double* a, int p) {
     return true;
 }
 
-// Each right-hand side: forward substitution with L, then back substitution with L^T.
-__device__ void chol_solve(const double* a, double* x, int p, int q) {
-    for (int c = 0; c < q; ++c) {
-        for (int i = 0; i < p; ++i) {
-            double t = x[i * q + c];
-            for (int k = 0; k < i; ++k) t -= a[i * p + k] * x[k * q + c];
-            x[i * q + c] = t / a[i * p + i];
-        }
-        for (int i = p - 1; i >= 0; --i) {
-            double t = x[i * q + c];
-            for (int k = i + 1; k < p; ++k) t -= a[k * p + i] * x[k * q + c];
-            x[i * q + c] = t / a[i * p + i];
-        }
+// Right-hand side c of x (p, q): forward substitution with L, then back substitution with L^T.
+// Columns never read each other, so solving them in any order or in parallel gives the same bits.
+__device__ void chol_solve_column(const double* a, double* x, int p, int q, int c) {
+    for (int i = 0; i < p; ++i) {
+        double t = x[i * q + c];
+        for (int k = 0; k < i; ++k) t -= a[i * p + k] * x[k * q + c];
+        x[i * q + c] = t / a[i * p + i];
+    }
+    for (int i = p - 1; i >= 0; --i) {
+        double t = x[i * q + c];
+        for (int k = i + 1; k < p; ++k) t -= a[k * p + i] * x[k * q + c];
+        x[i * q + c] = t / a[i * p + i];
     }
 }
 
+__device__ void chol_solve(const double* a, double* x, int p, int q) {
+    for (int c = 0; c < q; ++c) chol_solve_column(a, x, p, q, c);
+}
+
 // One thread per system: factor and solve (see ``spd_solve``); a non-SPD system gets an
 // all-NaN solution.
 extern "C" __global__ void spd_solve(double* L, double* X, long long nb, int p, int q) {
@@ -93,12 +96,152 @@ extern "C" __global__ void spd_factor_coop(double* L, long long nb, int p) {
         for (int t = threadIdx.x; t < p * p; t += blockDim.x) a[t] = nan("");
 }
 
-// One thread per system: solves with a factor from spd_factor; the same arithmetic as the
-// solve half of spd_solve, so factor-then-solve gives spd_solve's bits.
-extern "C" __global__ void spd_solve_factored(const double* L, double* X, long long nb, int p, int q) {
-    long long b = (long long)blockDim.x * blockIdx.x + threadIdx.x;
+// Packed lower triangle in shared memory: row i starts at i (i + 1) / 2.
+__device__ __forceinline__ int tri(int i) { return i * (i + 1) / 2; }
+
+// Cholesky of the packed lower triangle sh (in shared memory) by the whole block; col (p doubles
+// of shared memory) is workspace. Right-looking, so once column j is final every warp updates the
+// trailing triangle (one warp per row). Entry (i, m) still receives a_im - L_i0 L_m0 - L_i1 L_m1 -
+// ... in k order with the same operands as chol_inplace's inner sum, and is divided by its pivot
+// last, so the factor is bit-identical to chol_inplace and spd_factor_coop; only the schedule
+// differs. Keep the row operand first in each product (L_ik * L_mk), as chol_inplace does.
+// Returns false (to every thread) for a non-SPD or NaN pivot, leaving sh partly factored.
+__device__ bool chol_packed_block(double* sh, double* col, int p, int* ok) {
+    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
+    if (threadIdx.x == 0) *ok = 1;
+    __syncthreads();
+    for (int j = 0; j < p; ++j) {
+        // 1. The pivot: a_jj already holds a_jj - sum_k<j L_jk^2, accumulated in k order.
+        if (threadIdx.x == 0) {
+            double s = sh[tri(j) + j];
+            if (!(s > 0.0)) *ok = 0;
+            sh[tri(j) + j] = sqrt(s);
+        }
+        __syncthreads();
+        if (!*ok) return false;
+        // 2. Column j below the pivot, also copied to col[] so the update reads it without bank conflicts.
+        double d = sh[tri(j) + j];
+        for (int i = j + 1 + threadIdx.x; i < p; i += blockDim.x) {
+            double v = sh[tri(i) + j] / d;
+            sh[tri(i) + j] = v;
+            col[i] = v;
+        }
+        __syncthreads();
+        // 3. Trailing update: a_im -= L_ij L_mj for j < m <= i.
+        for (int i = j + 1 + warp; i < p; i += n_warps) {
+            double li = col[i];
+            double* row = sh + tri(i);
+            for (int m = j + 1 + lane; m <= i; m += 32) row[m] -= li * col[m];
+        }
+        __syncthreads();
+    }
+    return true;
+}
+
+// One block per system, its packed lower triangle in shared memory (p (p + 1) / 2 + p doubles,
+// so p <= 239 on a 227 KB card): the factor of chol_packed_block, bit-identical to spd_factor.
+extern "C" __global__ void spd_factor_shared(double* L, long long nb, int p) {
+    extern __shared__ double sh[];  // packed lower triangle, then column workspace
+    __shared__ int ok;
+    long long b = blockIdx.x;
     if (b >= nb) return;
-    chol_solve(L + b * p * p, X + b * p * q, p, q);
+    double* a = L + b * p * p;
+    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
+    // 1. Load the lower triangle; the upper triangle is never read.
+    for (int i = warp; i < p; i += n_warps)
+        for (int m = lane; m <= i; m += 32) sh[tri(i) + m] = a[i * p + m];
+    // 2. Factor; 3. write the lower triangle back (the upper one keeps its input, as chol_inplace leaves it).
+    if (!chol_packed_block(sh, sh + tri(p), p, &ok)) {
+        for (int t = threadIdx.x; t < p * p; t += blockDim.x) a[t] = nan("");
+        return;
+    }
+    for (int i = warp; i < p; i += n_warps)
+        for (int m = lane; m <= i; m += 32) a[i * p + m] = sh[tri(i) + m];
+}
+
+// Solve the q right-hand sides x (p, q, in shared memory) with the packed factor sh, by the whole
+// block. The forward substitution is right-looking with chol_solve's order (x_i - L_i0 x_0 - L_i1
+// x_1 - ... then / L_ii), so its bits are chol_solve's. The back substitution is right-looking
+// too, which subtracts L_ki x_k in decreasing k where chol_solve subtracts in increasing k: it
+// rounds differently from chol_solve (a fixed order per system, so still reproducible and
+// independent of the batch), in exchange for p parallel steps instead of a serial chain of
+// p^2 / 2 dependent products.
+__device__ void chol_packed_solve_block(const double* sh, double* x, int p, int q) {
+    // 1. Forward: x_j final once divided; then every later row subtracts its L_ij x_j.
+    for (int j = 0; j < p; ++j) {
+        for (int c = threadIdx.x; c < q; c += blockDim.x) x[j * q + c] /= sh[tri(j) + j];
+        __syncthreads();
+        for (int t = threadIdx.x; t < (p - j - 1) * q; t += blockDim.x) {
+            int i = j + 1 + t / q, c = t % q;
+            x[i * q + c] -= sh[tri(i) + j] * x[j * q + c];
+        }
+        __syncthreads();
+    }
+    // 2. Back: x_k final once divided; then every earlier row subtracts L_ki x_k (row k of L: contiguous).
+    for (int k = p - 1; k >= 0; --k) {
+        for (int c = threadIdx.x; c < q; c += blockDim.x) x[k * q + c] /= sh[tri(k) + k];
+        __syncthreads();
+        for (int t = threadIdx.x; t < k * q; t += blockDim.x) {
+            int i = t / q, c = t % q;
+            x[i * q + c] -= sh[tri(k) + i] * x[k * q + c];
+        }
+        __syncthreads();
+    }
+}
+
+// One block per system: factor (as spd_factor_shared) and solve its q right-hand sides with the
+// factor still in shared memory (p (p + 1) / 2 + p + p q doubles), so L never goes to global
+// memory; the solve is chol_packed_solve_block's, which rounds differently from spd_solve. A
+// non-SPD system gets an all-NaN solution, as spd_solve.
+extern "C" __global__ void spd_solve_shared(const double* A, double* X, long long nb, int p, int q) {
+    extern __shared__ double sh[];  // packed lower triangle, column workspace, then x (p, q)
+    __shared__ int ok;
+    long long b = blockIdx.x;
+    if (b >= nb) return;
+    const double* a = A + b * p * p;
+    double* xg = X + b * p * q;
+    double* x = sh + tri(p) + p;
+    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
+    // 1. Load the lower triangle and the right-hand sides; 2. factor; 3. solve; 4. write x.
+    for (int i = warp; i < p; i += n_warps)
+        for (int m = lane; m <= i; m += 32) sh[tri(i) + m] = a[i * p + m];
+    for (int t = threadIdx.x; t < p * q; t += blockDim.x) x[t] = xg[t];
+    if (!chol_packed_block(sh, sh + tri(p), p, &ok)) {
+        for (int t = threadIdx.x; t < p * q; t += blockDim.x) xg[t] = nan("");
+        return;
+    }
+    chol_packed_solve_block(sh, x, p, q);
+    for (int t = threadIdx.x; t < p * q; t += blockDim.x) xg[t] = x[t];
+}
+
+// One block per system: solves with a factor from spd_factor, staged into shared memory as a
+// packed triangle (p (p + 1) / 2 + p q doubles) and solved as spd_solve_shared does, so its bits
+// are spd_solve_shared's. A factor that failed (all NaN) gives all-NaN solutions.
+extern "C" __global__ void spd_solve_factored_shared(const double* L, double* X, long long nb, int p, int q) {
+    extern __shared__ double sh[];  // packed lower triangle, then x (p, q)
+    long long b = blockIdx.x;
+    if (b >= nb) return;
+    const double* a = L + b * p * p;
+    double* xg = X + b * p * q;
+    double* x = sh + tri(p);
+    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
+    for (int i = warp; i < p; i += n_warps)
+        for (int m = lane; m <= i; m += 32) sh[tri(i) + m] = a[i * p + m];
+    for (int t = threadIdx.x; t < p * q; t += blockDim.x) x[t] = xg[t];
+    __syncthreads();
+    chol_packed_solve_block(sh, x, p, q);
+    for (int t = threadIdx.x; t < p * q; t += blockDim.x) xg[t] = x[t];
+}
+
+// One thread per (system, right-hand side): solves with a factor from spd_factor. Each column
+// keeps the solve half of spd_solve's arithmetic, so factor-then-solve gives spd_solve's bits,
+// and a system with many columns (an inverse, q = p) is no longer one thread's serial p^3 work.
+// Threads of one system are adjacent, so their x reads coalesce and their L reads are shared.
+extern "C" __global__ void spd_solve_factored(const double* L, double* X, long long nb, int p, int q) {
+    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
+    if (t >= nb * q) return;
+    long long b = t / q;
+    chol_solve_column(L + b * p * p, X + b * p * q, p, q, (int)(t - b * q));
 }
 
 // Batched small products with a fixed sequential inner sum (see ``det_matmul``).
diff --git a/src/numerics/linalg.py b/src/numerics/linalg.py
index 004a3c5..16c79df 100644
--- a/src/numerics/linalg.py
+++ b/src/numerics/linalg.py
@@ -15,14 +15,24 @@ if IS_CUPY_BACKEND:  # sources in src/kernels/*.cu
     _kern_spd_solve = _linalg.get_function("spd_solve")
     _kern_spd_factor = _linalg.get_function("spd_factor")
     _kern_spd_factor_coop = _linalg.get_function("spd_factor_coop")
+    _kern_spd_factor_shared = _linalg.get_function("spd_factor_shared")
+    # The packed triangle of the largest system spd_factor_shared can hold in one block's shared memory.
+    _SPD_SHARED_BYTES = int(xp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]) - int(
+        _kern_spd_factor_shared.shared_size_bytes)  # the opt-in limit counts the kernel's static flag too
+    _kern_spd_factor_shared.max_dynamic_shared_size_bytes = _SPD_SHARED_BYTES
+    _kern_spd_solve_shared = _linalg.get_function("spd_solve_shared")
+    _kern_spd_solve_shared.max_dynamic_shared_size_bytes = _SPD_SHARED_BYTES
     _kern_spd_solve_factored = _linalg.get_function("spd_solve_factored")
+    _kern_spd_solve_factored_shared = _linalg.get_function("spd_solve_factored_shared")
+    _kern_spd_solve_factored_shared.max_dynamic_shared_size_bytes = _SPD_SHARED_BYTES
     _kern_affine_reg = _linalg.get_function("fused_affine_fit_regularize")
     _kern_cholesky = _linalg.get_function("batched_inplace_cholesky_solve")
 
 
 TINY_EPS = 1e-12
 
-# From this many unknowns, SPD systems are factored by one block each (spd_factor_coop).
+# From this many unknowns, SPD systems are factored by one block each: in shared memory
+# (spd_factor_shared) while the packed triangle fits, else in global memory (spd_factor_coop).
 SPD_COOP_MIN_P = 32
 
 
@@ -87,17 +97,31 @@ def leading_eigvec(C: Array) -> Array:
     return v * xp.where(xp.take_along_axis(v, i[:, None], axis=1) < 0, -1.0, 1.0)
 
 
-def spd_solve(A: Array, B: Array) -> Array:
+def spd_solve(A: Array, B: Array, *, exact: bool = False) -> Array:
     """
     ``A^-1 B`` for stacks of small symmetric positive definite systems ``(..., p, p)``,
     ``(..., p, q)`` (equal leading shapes). NumPy: ``xp.linalg.solve``. CuPy: one thread
     per system (Cholesky of the lower triangle and two triangular solves), a single kernel
     instead of the dozen launches and ~0.4 ms of a batched ``linalg.solve``, with bits that
-    do not depend on the rest of the stack. A system that is not positive definite (or not
-    finite) gets an all-NaN solution, which callers already treat as an invalid fit.
+    do not depend on the rest of the stack. From ``SPD_COOP_MIN_P`` unknowns, one block per
+    system factors and solves in shared memory while it fits (``spd_solve_shared``), else
+    ``spd_factor`` then ``spd_solve_factored``; both give ``spd_solve_factored(spd_factor(A), B,
+    exact=exact)``'s bits (``exact``: that function's exact substitution order). A system that is
+    not positive definite (or not finite) gets an all-NaN solution, which callers already treat as
+    an invalid fit.
     """
     if IS_CUPY_BACKEND and A.shape[-1] >= SPD_COOP_MIN_P:
-        return spd_solve_factored(spd_factor(A), B)
+        p, q = A.shape[-1], B.shape[-1]
+        shared = 8 * (p * (p + 1) // 2 + p + p * q)
+        if shared > _SPD_SHARED_BYTES or exact:
+            return spd_solve_factored(spd_factor(A), B, exact=exact)
+        lead = A.shape[:-2]
+        A = xp.ascontiguousarray(A, dtype=xp.float64)
+        X = xp.array(xp.broadcast_to(B, (*lead, p, q)), dtype=xp.float64, order="C", copy=True)
+        nb = int(np.prod(lead, dtype=np.int64))
+        if nb:
+            _kern_spd_solve_shared((nb,), (256,), (A, X, np.int64(nb), np.int32(p), np.int32(q)), shared_mem=shared)
+        return X
     if not IS_CUPY_BACKEND:
         return solve_or_nan(A, B)
     p, q = A.shape[-1], B.shape[-1]
@@ -141,24 +165,38 @@ def spd_factor(A: Array) -> Array:
     p = A.shape[-1]
     L = xp.array(A, dtype=xp.float64, order="C", copy=True)
     nb = int(np.prod(A.shape[:-2], dtype=np.int64))
-    if nb and p >= SPD_COOP_MIN_P:  # one block per system (same bits): one thread is too serial
+    shared = 8 * (p * (p + 1) // 2 + p)
+    if nb and p >= SPD_COOP_MIN_P and shared <= _SPD_SHARED_BYTES:  # one block per system, in shared memory (same bits)
+        _kern_spd_factor_shared((nb,), (256,), (L, np.int64(nb), np.int32(p)), shared_mem=shared)
+    elif nb and p >= SPD_COOP_MIN_P:  # one block per system (same bits): one thread is too serial
         _kern_spd_factor_coop((nb,), (min(128, -(-p // 32) * 32),), (L, np.int64(nb), np.int32(p)))
     elif nb:
         _kern_spd_factor(((nb + 127) // 128,), (128,), (L, np.int64(nb), np.int32(p)))
     return L
 
 
-def spd_solve_factored(L: Array, B: Array) -> Array:
-    """``A^-1 B`` from ``L = spd_factor(A)``; bit-identical to ``spd_solve(A, B)`` on both backends."""
+def spd_solve_factored(L: Array, B: Array, *, exact: bool = False) -> Array:
+    """
+    ``A^-1 B`` from ``L = spd_factor(A)``, bit-identical to ``spd_solve(A, B)``. CuPy: from
+    ``SPD_COOP_MIN_P`` unknowns, while the factor and right-hand sides fit one block's shared
+    memory, one block per system (``spd_solve_factored_shared``, the substitution of
+    ``spd_solve_shared``); otherwise, or with ``exact``, one thread per (system, right-hand side),
+    the substitution of ``spd_solve``'s one-thread kernel, which keeps the back substitution's
+    increasing-k order (and so the bits of the CPU loop's order) at the cost of a serial chain.
+    """
     if not IS_CUPY_BACKEND:
         return solve_or_nan(L, B)
     p, q = L.shape[-1], B.shape[-1]
     lead = L.shape[:-2]
     X = xp.array(xp.broadcast_to(B, (*lead, p, q)), dtype=xp.float64, order="C", copy=True)
     nb = int(np.prod(lead, dtype=np.int64))
-    if nb:
-        _kern_spd_solve_factored(((nb + 127) // 128,), (128,), (xp.ascontiguousarray(L), X, np.int64(nb),
-                                                                 np.int32(p), np.int32(q)))  # fmt: skip
+    shared = 8 * (p * (p + 1) // 2 + p * q)
+    if nb and q and p >= SPD_COOP_MIN_P and shared <= _SPD_SHARED_BYTES and not exact:
+        _kern_spd_solve_factored_shared((nb,), (256,), (xp.ascontiguousarray(L), X, np.int64(nb), np.int32(p),
+                                                        np.int32(q)), shared_mem=shared)  # fmt: skip
+    elif nb and q:
+        _kern_spd_solve_factored(((nb * q + 127) // 128,), (128,), (xp.ascontiguousarray(L), X, np.int64(nb),
+                                                                     np.int32(p), np.int32(q)))  # fmt: skip
     return X
 
 
@@ -223,10 +261,13 @@ def lstsq_auto(G: Array, B: Array, design: Array, rhs: Array, starts: Array, cou
     from scipy import linalg as sla
 
     K, p = int(G.shape[0]), int(G.shape[-1])
-    X = spd_solve(G, B)
+    # One factorization for the solution and the inverse. The leaf coefficients keep the exact
+    # substitution order, so a leaf's model does not change with the split search's solver.
+    L = spd_factor(G)
+    X = spd_solve_factored(L, B, exact=True)
     if K == 0:
         return X
-    inv = spd_solve(G, xp.broadcast_to(xp.eye(p), (K, p, p)))
+    inv = spd_solve_factored(L, xp.broadcast_to(xp.eye(p), (K, p, p)))
     norm_1 = xp.max(xp.sum(xp.abs(G), axis=-2), axis=-1)
     norm_1_inv = xp.max(xp.sum(xp.abs(inv), axis=-2), axis=-1)
     rcond = 1.0 / (norm_1 * norm_1_inv)
diff --git a/src/numerics/segments.py b/src/numerics/segments.py
index 889d627..17b4552 100644
--- a/src/numerics/segments.py
+++ b/src/numerics/segments.py
@@ -16,6 +16,7 @@ if IS_CUPY_BACKEND:  # sources in src/kernels/*.cu
     _kern_seg_cross_partial = _seg.get_function("segment_cross_partial")
     _kern_seg_cross_reduce = _seg.get_function("segment_cross_reduce")
     _kern_seg_cross_tiled = _seg.get_function("segment_cross_tiled")
+    _kern_seg_cross_tiled2d = _seg.get_function("segment_cross_tiled2d")
 
 
 SEGMENT_CHUNK = 256
@@ -23,6 +24,9 @@ SEGMENT_CHUNK = 256
 # From this many output entries per segment, segment_cross loads each chunk into shared
 # memory once (same bits as the untiled kernel; fewer FP64 operations and global reads).
 SEGMENT_TILED_MIN_E = 32
+# From this many rows and columns of output, segment_cross uses 4 x 4 register tiles per thread
+# (segment_cross_tiled2d); below it a 64 x 64 tile is mostly padding.
+SEGMENT_TILED2D_MIN = 32
 
 
 def repeats(values: Array, repeats: Array) -> Array:
@@ -221,7 +225,15 @@ def segment_cross(
     same = B is A and a_off is b_off and w is None
     width = P if same or flags & 8 else P + Q
     TR = min(32, (40 * 1024) // (8 * max(1, width)))
-    if SEGMENT_TILED_MIN_E <= E and TR >= 4:  # entry groups of 8 x threads lift any cap on E
+    if P >= SEGMENT_TILED2D_MIN and Q >= SEGMENT_TILED2D_MIN and not flags & 8:
+        tile_r, tile_c, eidx = _segment_cross_tiles(P, Q, symmetric)
+        _kern_seg_cross_tiled2d(
+            (S * n_chunks, int(tile_r.shape[0])), (256,),
+            (A, B, *opt, cum, starts, counts, tile_r, tile_c, eidx, partial, np.int64(n_chunks), np.int32(K),
+             np.int32(E), np.int32(P), np.int32(Q), np.int32(CH), np.int32(flags), np.int64(N if multi else 0),
+             np.int64(b_stride)),
+        )  # fmt: skip
+    elif SEGMENT_TILED_MIN_E <= E and TR >= 4:  # entry groups of 8 x threads lift any cap on E
         threads = min(T, ((E + 31) // 32) * 32)
         _kern_seg_cross_tiled(
             (S * n_chunks, -(-E // (8 * threads))), (threads,),
@@ -245,6 +257,24 @@ def segment_cross(
     return out
 
 _SEGMENT_CROSS_ENTRIES: dict = {}
+_SEGMENT_CROSS_TILES: dict = {}
+
+
+def _segment_cross_tiles(P: int, Q: int, symmetric: bool) -> tuple[Array, Array, Array]:
+    """
+    ``segment_cross_tiled2d``'s 64 x 64 output tiles (row and column tile of each; only tiles
+    touching the upper triangle if symmetric) and the index of entry ``(r, c)`` in
+    ``_segment_cross_entries`` (-1 where none is computed), cached per shape.
+    """
+    key = (P, Q, symmetric)
+    if key not in _SEGMENT_CROSS_TILES:
+        nr, nc = -(-P // 64), -(-Q // 64)
+        tiles = np.array([(i, j) for i in range(nr) for j in range(nc) if not symmetric or j >= i], dtype=np.int32)
+        er, ec = (np.asarray(to_host(a)) for a in _segment_cross_entries(P, Q, symmetric))
+        eidx = np.full(P * Q, -1, dtype=np.int32)
+        eidx[er.astype(np.int64) * Q + ec] = np.arange(er.shape[0], dtype=np.int32)
+        _SEGMENT_CROSS_TILES[key] = (xp.asarray(tiles[:, 0].copy()), xp.asarray(tiles[:, 1].copy()), xp.asarray(eidx))
+    return _SEGMENT_CROSS_TILES[key]
 
 
 def _segment_cross_entries(P: int, Q: int, symmetric: bool) -> tuple[Array, Array]:
diff --git a/studies/sanity/art/art_cases.py b/studies/sanity/art/art_cases.py
index b15c1f2..b7250d9 100644
--- a/studies/sanity/art/art_cases.py
+++ b/studies/sanity/art/art_cases.py
@@ -11,7 +11,8 @@ Matched settings (group ``conv``): same target, domain, test set (``10000 d`` un
 seed 19), seed, leaf degree, pointwise relative tolerance (floor 1e-12), ``--max-depth``, sample
 multiplier 50 and soft-oblique split search with the same iteration cap (``--library-soft-iters
 200``, the reference's ``--max-iters``, pinned so the match does not depend on the default). Group
-``library_default`` reruns this library's side with its own default (``ARTSettings.soft_iters``). Problems follow the reference's recorded cluster presets
+``library_default`` reruns this library's side with its own default (``ARTSettings.soft_iters``, passed
+explicitly, since art_comparison.py otherwise uses the reference's ``--max-iters``). Problems follow the reference's recorded cluster presets
 (``tasmanian/oracle_presets.json``): offsets keep |f| away from 0, where a pointwise relative
 tolerance cannot be met by any leaf.
 
@@ -24,6 +25,8 @@ import argparse
 import math
 import re
 
+from src.config import ARTSettings
+
 GROUPS = ("conv", "library_default", "depth", "exact_target", "mpi", "repeat", "gpu_vs_cpu")
 TIERS = {"S": 0, "M": 1, "L": 2}
 SIGNS = [1, -1, 1, 1, -1, -1, 1, -1]  # the reference's recorded dense-sign plane-wave normals (d = 3, 5, 8)
@@ -66,7 +69,8 @@ def full_matrix(seeds, nps, gpu):
 
     def both(case, group, t, args, sides=("reference", "library"), np_=1, device="main", tag=""):
         for side in sides:
-            extra = ["--library-soft-iters", "200"] if group != "library_default" else []
+            # art_comparison.py gives this library's ART the reference's --max-iters unless told otherwise
+            extra = ["--library-soft-iters", str(200 if group != "library_default" else ARTSettings().soft_iters)]
             rows.append((case, group, t, side, np_, device, f"{group}/{case}.{side}{tag}.json", args + extra))
 
     for problem, (_, dims, degrees, tols, depth) in PROBLEMS.items():
diff --git a/tests/unit/test_art.py b/tests/unit/test_art.py
index ced9edd..ff6c8c9 100644
--- a/tests/unit/test_art.py
+++ b/tests/unit/test_art.py
@@ -251,7 +251,7 @@ def test_soft_refinement_never_returns_a_worse_hard_split_than_its_incumbent():
     rows = _rows(np.random.default_rng(2))
     w0, z0 = xp.asarray([[0.0, 1.0]]), xp.asarray([0.5])  # a poor incumbent: the wrong axis
     sse0, _ = hard_sse(rows, w0, z0, plan.min_side)
-    w, z = plan._soft_refine(rows, w0, z0, sse0, None)
+    w, z = plan._soft_refine(rows, w0, z0, sse0, None, rows)
     sse, ok = hard_sse(rows, w, z, plan.min_side)
     assert bool(ok[0]) and float(sse[0]) <= float(sse0[0])
     assert plan.counters["soft_improved"] == 1  # the step in x0 = 0.4 is found
@@ -260,6 +260,24 @@ def test_soft_refinement_never_returns_a_worse_hard_split_than_its_incumbent():
     assert sum(plan.counters[r] for r in reasons) == 1  # one termination reason per refined node
 
 
+def test_split_search_uses_affine_fits_from_affine_search_from_coefficients():
+    """Quadratic leaves at d = 3 have 10 coefficients: affine search rows from a threshold of 10, the leaf's below it."""
+    from src.art import ARTPlan
+    from src.config import TrainingConfig
+
+    rows, _, _, _ = _split_rows(np.random.default_rng(4), d=3)
+    for start, p_search in ((10, 4), (11, 10), (None, 10)):
+        cfg = TrainingConfig(atol=1e-3, algorithm="art", domain=([0] * 3, [1] * 3), local_model="quadratic_lsq",
+                             art={"sample_multiplier": 10, "affine_search_from": start})  # fmt: skip
+        plan = ARTPlan(cfg, 3, 2, Domain(np.zeros(3), np.ones(3)), Roots.build("box", 3, 1))
+        quad = rows.__class__(rows.U, rows.Y, plan.model.features(rows.U), rows.labels, rows.counts)
+        search = plan._search_rows(quad)
+        assert search.Phi.shape[1] == p_search
+        assert search is quad or (np.array_equal(host(search.Phi[:, :3]), host(quad.U)) and bool(xp.all(search.Phi[:, 3] == 1)))
+    with pytest.raises(ValueError, match="affine_search_from"):
+        TrainingConfig(algorithm="art", domain=([0], [1]), art={"affine_search_from": 0})
+
+
 def test_drift_check_flags_short_chains_from_packed_seeds_and_extension_clears_it():
     """Seeds packed in one corner of a node: 2 steps cannot spread them; doubling chains do."""
     domain, roots = Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1)
@@ -419,7 +437,7 @@ def test_narrow_score_clouds_far_from_the_origin_keep_a_resolved_temperature():
     U = np.c_[0.9 + 1e-8 * np.linspace(-1, 1, n), np.random.default_rng(0).random(n)]
     Y = np.where(U[:, :1] > 0.9 + 2e-9, 1.0, -1.0)
     rows = SplitRows(xp.asarray(U), xp.asarray(Y), xp.asarray(np.c_[U, np.ones(n)]), xp.zeros(n, dtype=xp.int64), xp.asarray([n]))
-    plan._soft_refine(rows, xp.asarray([[1.0, 0.0]]), xp.asarray([0.9]), xp.asarray([np.inf]), None)
+    plan._soft_refine(rows, xp.asarray([[1.0, 0.0]]), xp.asarray([0.9]), xp.asarray([np.inf]), None, rows)
     assert plan.counters["soft_gate_saturated"] == 0
 
 
@@ -447,11 +465,11 @@ def test_bins_search_skips_nodes_too_small_to_split_beside_valid_ones():
     labels = xp.asarray(np.r_[np.zeros(60), np.ones(n_small)].astype(np.int64))
     rows = SplitRows(U, Y, Phi, labels, xp.asarray([60, n_small]))
     bank = xp.asarray(np.broadcast_to(np.eye(2), (2, 2, 2)).copy())
-    _, z, sse = plan._bins_search(rows, bank)
+    _, z, sse = plan._bins_search(rows, bank, rows)
     assert np.isfinite(host(z)[0]) and np.isfinite(host(sse)[0])
     assert np.isnan(host(z)[1]) and np.isinf(host(sse)[1])
     alone = SplitRows(big.U, big.Y, big.Phi, big.labels, big.counts)
-    _, z_alone, _ = plan._bins_search(alone, bank[:1])
+    _, z_alone, _ = plan._bins_search(alone, bank[:1], alone)
     assert float(host(z_alone)[0]) == float(host(z)[0])
 
 
```
