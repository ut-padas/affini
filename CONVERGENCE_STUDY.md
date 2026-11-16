# Accuracy and convergence study: AFINNI paper re-run, reference ART vs AFINNI's ART

Prepared on 2026-11-16 against AFINNI `refinement-engine` @ `f60eb2b` and the reference ART
`main` @ `854924a` (2026-11-10, "BFS traversal, Multithreaded numba sampler, Batched GPU
optimizer"; `git pull --ff-only` reported "Already up to date"). Nothing was committed in either
repository. The heavy runs are left to you; what was run here (CPU only) is in
[Smoke tests run here](#smoke-tests-run-here).

## Goals

- **A.** Re-run every experiment behind the numbers in the AFINNI manuscript
  (`review_artifacts/26acm_afinni.zip`) with the paper's settings, and compare each reported
  number with the re-run. The test is accuracy: errors against tolerance, convergence slopes and
  trust-region coverage must meet the paper's claims. Leaf and evaluation counts may move where
  the code changed on purpose.
- **B.** Run a convergence study of the reference ART and AFINNI's ART (`algorithm="art"`) on the
  reference's problems under matched settings, and decide whether they agree:
  accuracy-vs-cost curves, convergence rates and tolerance compliance, not bit-identical trees.

## Files

Everything new is under `studies/convergence/` (its `output/` is git-ignored):

| File | Role |
|---|---|
| `common.sh` | paths, `run_logged` (timeout, log, done-marker, never aborts the sweep), provenance |
| `paper/run_paper.sh` | **Part A top level** |
| `paper/cases.py` | Part A run matrix (groups, tiers); `--dry-run` of the shell script prints it |
| `paper/paper_values.py` | every number the paper reports, with its TeX label, and the paper's internal inconsistencies |
| `paper/run_case.py` | one `afinni_launcher.py` run with results redirected and an optional seed; records wall time and peak RSS |
| `paper/run_iterative.py` | one run of `p_out_iterative_experiment.py` (Figure 9) |
| `paper/collect_paper.py` | paper-vs-re-run table, checks A1-A6, slopes, plots |
| `art/run_art.sh` | **Part B top level** |
| `art/art_cases.py` | Part B run matrix |
| `art/run_art_case.py` | one side of `scripts/art_comparison.py` (imported, not copied), saved as JSON with RMSE, max abs error, all quantiles, wall time, peak RSS |
| `art/exact_checks.py` | the exact-agreement sub-tests (root fit, thin-box fit) |
| `art/collect_art.py` | paired tables, slopes, checks B1-B8, plots |

## Quick start

```bash
cd ~/github-repos/affini            # the AFINNI root; scripts use .venv/bin/python
uv sync --extra gpu --extra mpi       # GPU machine (WSL); on TACC: --extra mpi-system, source scripts/hpc/tacc.sh

# tiny end-to-end versions (CPU, about 70 s and 45 s; each includes one 2-rank MPI run)
studies/convergence/paper/run_paper.sh --smoke
studies/convergence/art/run_art.sh --smoke

# Part A, cheapest tier first, on the GPU; then M and L on bigger hardware
studies/convergence/paper/run_paper.sh --device gpu --max-tier S
studies/convergence/paper/run_paper.sh --device gpu --max-tier M
studies/convergence/paper/run_paper.sh --device gpu --max-tier L      # HPC node, >= 128 GB host RAM

# Part B
studies/convergence/art/run_art.sh --device gpu --ref-device gpu --max-tier S
studies/convergence/art/run_art.sh --device gpu --ref-device gpu --max-tier M

# rebuild the reports at any time (also run at the end of each sweep)
.venv/bin/python studies/convergence/paper/collect_paper.py --out studies/convergence/output/paper
.venv/bin/python studies/convergence/art/collect_art.py --out studies/convergence/output/art
```

Common options: `--groups a,b`, `--only REGEX` (case ids), `--out DIR`, `--dry-run` (print the
commands with their `AFINNI_DEVICE`/`AFINNI_MPI`), `--seeds`, `--nps`. Every command sets
`AFINNI_DEVICE` and `AFINNI_MPI` explicitly. Environment overrides: `ART_ROOT`, `PYTHON`,
`MPIEXEC` (e.g. `MPIEXEC=ibrun` or `"srun --mpi=pmix"`), `STUDY_OUT`, `RUN_TIMEOUT` (default 6h
per run, `0` = none). Output layout: `OUT/results/...` (JSON), `OUT/logs/<case>.log`,
`OUT/done/<case>.done` (a rerun skips done cases; delete a marker to redo one),
`OUT/provenance.txt` (both commits, Python, backend, GPU), `OUT/report/` (tables, plots).
On SLURM, wrap one tier or group per job; the done-markers make resubmission safe.

---

## Part A: the AFINNI manuscript

### What the paper reports

Experiments ran on TACC Vista (GH200, CuPy, cuML k-means, Python 3.13, CUDA 12.9). Settings
common to Table 1: uniform training and test points, `[N_0, K_0, M] = 1.3^d x [50000, 25, 100]`
(`N_0` = `default_n_train`, `K_0` = `num_initial_clusters`, `M` = `min_pts_per_cluster`),
`rtol = atol` ("Linf": a cluster passes if its max error relative to its own max |f| is below
rtol, or its max error is below atol), covariance-Hessian splits (`affine_cov_hess`), `kappa = 2 d^2`
candidates (`coarse_factor`; the repo's script uses 10/50/200 for d = 2/5/10), `q_reg = 1e-6`,
convex-combination augmentation, k-means initial clusters, test set `N_0 / 2` points
(Rosenbrock: 50,000). Errors: `e_in`, `e` relative L2 inside the trust regions / over all test
points, `p_o` % of test points outside every trust region, `e_nh` = `e` with Hessian-free splits
(`M = I`, mode `affine_cov_no_hess`). Theory (`eq:theoretical_scaling`): `e ~ N_c^(-2/d)`.

**Table 1a/1b** (`convergence.tex`, `tab:combined_convergence_A`/`_B`; the matrix's tier in the last column):

| # | problem | d | rtol | N | N_c | p_o (%) | e_in | e | e_nh | tier |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Quadratic | 5 | 0.1 | 2.32e5 | 390 | 0.03 | 0.0217 | 0.0218 | 0.0743 | S |
| 2 | Quadratic | 5 | 0.05 | 4.93e5 | 964 | 0.42 | 0.0128 | 0.0132 | 0.64 | S |
| 3 | Quadratic | 5 | 0.02 | 2.91e6 | 5,772 | 6.15 | 0.00668 | 0.00936 | 0.0424 | S |
| 4 | Quadratic | 5 | 0.01 | 1.43e7 | 28,262 | 16.9 | 0.00473 | 0.00865 | 0.0257 | M |
| 5 | Quadratic | 5 | 0.005 | 6.37e7 | 1.25e5 | 33.1 | 0.00322 | 0.00755 | 0.0169 | M |
| 6 | Quadratic | 10 | 0.1 | 6.5e6 | 3,229 | 0.43 | 0.0295 | 0.0302 | 0.142 | S |
| 7 | Quadratic | 10 | 0.05 | 1.9e7 | 9,422 | 3.87 | 0.0206 | 0.0285 | 0.134 | M |
| 8 | Quadratic | 10 | 0.02 | 1.51e8 | 74,969 | 26.2 | 0.0153 | 0.0392 | 0.114 | L |
| 9 | Quadratic | 10 | 0.01 | 9.85e8 | 4.87e5 | 58.9 | 0.0113 | 0.044 | 0.09 | L |
| 10 | Gaussian | 2 | 0.1 | 8.46e4 | 48 | 0.02 | 0.0154 | 0.0154 | 0.0222 | S |
| 11 | Gaussian | 2 | 0.05 | 8.49e4 | 65 | 0.02 | 0.00763 | 0.00763 | 0.013 | S |
| 12 | Gaussian | 2 | 0.02 | 8.67e4 | 104 | 0.04 | 0.00376 | 0.00378 | 0.00693 | S |
| 13 | Gaussian | 2 | 0.01 | 9.92e4 | 176 | 0.15 | 0.00207 | 0.00218 | 0.00394 | S |
| 14 | Gaussian | 2 | 0.005 | 1.4e5 | 359 | 0.21 | 0.00102 | 0.00106 | 0.00177 | S |
| 15 | Gaussian | 5 | 0.1 | 2.28e5 | 173 | 0.05 | 0.00186 | 0.00189 | 0.00903 | S |
| 16 | Gaussian | 5 | 0.05 | 5.64e5 | 810 | 1.1 | 0.000256 | 0.0248 | 0.0138 | S |
| 17 | Gaussian | 5 | 0.02 | 1.64e6 | 2,847 | 1.7 | 0.000173 | 0.0116 | 0.0173 | S |
| 18 | Gaussian | 5 | 0.01 | 7.02e6 | 13,010 | 2.6 | 7.71e-5 | 0.0206 | 0.0092 | S |
| 19 | Gaussian | 5 | 0.005 | 1.73e7 | 32,471 | 3.0 | 2.88e-5 | 0.0286 | 0.00698 | M |
| 20 | Rosenbrock | 2 | 0.1 | 1.39e5 | 344 | 1.59 | 0.034 | 0.034 | 0.0421 | S |
| 21 | Rosenbrock | 2 | 0.05 | 2.27e5 | 746 | 1.48 | 0.0158 | 0.0158 | 0.0406 | S |
| 22 | Rosenbrock | 2 | 0.02 | 5.23e5 | 2,007 | 1.45 | 0.0051 | 0.00512 | 0.0346 | S |
| 23 | Rosenbrock | 2 | 0.01 | 1.08e6 | 4,232 | 1.84 | 0.00232 | 0.00234 | 0.0111 | S |
| 24 | Rosenbrock | 2 | 0.005 | 2.15e6 | 8,446 | 2.36 | 0.00118 | 0.0012 | 0.00626 | S |
| 25 | Rosenbrock | 5 | 0.1 | 1.42e7 | 27,916 | 14.2 | 0.0421 | 0.0515 | 0.0738 | M |
| 26 | Rosenbrock | 5 | 0.05 | 5.24e7 | 1.03e5 | 31.1 | 0.026 | 0.0402 | 0.0562 | M |
| 27 | Rosenbrock | 5 | 0.02 | 2.39e8 | 4.69e5 | 57.3 | 0.0144 | 0.0329 | 0.0423 | L |
| 28 | Rosenbrock | 5 | 0.01 | 7.03e8 | 1.38e6 | 73.8 | 0.0088 | 0.0289 | 0.0353 | L |
| 29 | Rosenbrock | 5 | 0.005 | 2.03e9 | 3.97e6 | 84.9 | 0.00513 | 0.0256 | 0.0303 | L |
| 30 | Lorenz96 | 5 | 0.1 | 1.86e5 | 139 | 0.0 | 0.0509 | 0.0509 | 0.055 | S |
| 31 | Lorenz96 | 5 | 0.05 | 2.03e5 | 355 | 0.01 | 0.0262 | 0.0262 | 0.0407 | S |
| 32 | Lorenz96 | 5 | 0.02 | 6.88e5 | 1,327 | 0.03 | 0.0135 | 0.0135 | 0.021 | S |
| 33 | Lorenz96 | 5 | 0.01 | 2.23e6 | 4,285 | 0.21 | 0.00888 | 0.00893 | 0.0134 | S |
| 34 | Lorenz96 | 5 | 0.005 | 7.55e6 | 14,473 | 2.04 | 0.00625 | 0.00649 | 0.00946 | S |
| 35 | Wave | 2 | 0.1 | 8.53e4 | 162 | 0.02 | 0.0195 | 0.0195 | 0.0491 | S |
| 36 | Wave | 2 | 0.05 | 9.0e4 | 240 | 0.04 | 0.00881 | 0.0882 | 0.0339 | S |
| 37 | Wave | 2 | 0.02 | 1.15e5 | 393 | 0.05 | 0.00421 | 0.00422 | 0.0213 | S |
| 38 | Wave | 2 | 0.01 | 1.77e5 | 678 | 0.1 | 0.00209 | 0.00209 | 0.0138 | S |
| 39 | Wave | 2 | 0.005 | 3.03e5 | 1,177 | 0.18 | 0.00105 | 0.00105 | 0.00826 | S |
| 40 | Wave | 5 | 0.1 | 5.76e6 | 15,448 | 2.49 | 0.0408 | 0.0431 | 0.35 | S |
| 41 | Wave | 5 | 0.05 | 1.14e7 | 30,432 | 6.85 | 0.0255 | 0.0312 | 0.225 | M |
| 42 | Wave | 5 | 0.02 | 3.02e7 | 81,035 | 22.4 | 0.0123 | 0.0359 | 0.215 | M |
| 43 | Wave | 5 | 0.01 | 6.4e7 | 1.72e5 | 40.3 | 0.00649 | 0.052 | 0.198 | M |
| 44 | Wave | 5 | 0.005 | 1.37e8 | 3.68e5 | 57.8 | 0.00355 | 0.0701 | 0.184 | L |

**Other numerical results** (all transcribed in `paper/paper_values.py`):

| Where | What | Settings |
|---|---|---|
| Figure 7 (`fig:convergence_plot`) | slopes of `e_in` vs `N_c` per problem, from Table 1; text: Gaussian d=2 -1.31, Quadratic d=10 -0.21; all at or below `-2/d` except Quadratic d=5 | recomputed from Table 1 by the collector |
| Figure 8 table (`fig:exact_grad_vs_approx_grad_comparison`) | Wave d=5, rtol 0.1 ... 0.005: LSQ gradient (= Table 1 rows 40-44) vs exact gradient (`N_c` 10,351 ... 42,626; `e_in` 4.28e-2 ... 2.75e-3) | mode `affine_cov_hess_exact_grad` |
| Table 2 (`tab:lorenz_dim_scaling`) | Lorenz-96, rtol 0.1, d = 5, 10, 20, 40, 80: `N` 1.86e5 ... 1.19e8, `N_c` 117, 1400, 1801, 1775, 2335, `e_in` 5.45e-2 ... 7.34e-2, `p_o` <= 0.5% | `N_0 = min(5e4 1.3^d, 1e6)`, `K_0 = min(25 1.3^d, 1000)`, `M = min(100 1.3^d, 500)` |
| Table 3 (`tab:rosenbrock_weak_scaling`) | Rosenbrock d=5, atol only = 579.8, 289.9, 145.0, 72.5 with `N_0` = 5e4, 282,842, 1,600,889, 9,061,035: `N_c` 7,443 ... 1,121,571, `e_in` 4.49e-2 ... 6.06e-3, `p_in,good` >= 96% | rtol off, `N_0 ~ eps^(-5/2)` |
| Table 4 (`tab:other_surrogates`), AFINNI rows | Rosenbrock d=2 (7.5e2, 1.6e-2), d=5 (1.0e5, 4.0e-2), Lorenz d=5 (7.8e2, 1.9e-2) | = Table 1 rows 21, 26 and (old) 31; other methods (Forest, GPR, FALKON, NN, Tasmanian) not in this repo |
| Figure 9 (`fig:iterative_refinement`) | `p_out` decreasing over 10 iterations of uniform gap resampling, 4 functions; no numbers in the text | 20,000 initial points, +40% per iteration |
| Figures 10-13, Tables 5-6 (`scalability.tex`) | training time 0.61 s -> 37.51 s (Rosenbrock d=5, rtol 1 -> 5e-3), `N` 1.9e5 -> >1e9, `N_c` -> 2.7e6, prediction 0.01 -> 0.41 s for 4.6e4 queries; D=10 run progression; MPI static/dynamic times on 1-1024 ranks | speed, informational; only the MPI accuracy invariance is checked (group `mpi`) |

The paper's internal inconsistencies (recorded in `PAPER_NOTES`): Table 4's Lorenz row matches
the commented-out older Lorenz rows (782, 1.87e-2), not Table 1b row 31 (355, 2.62e-2); Lorenz
d=5 rtol 0.1 appears twice with the same settings (Table 1b #30: `N_c` 139; Table 2 #1: 117),
which is the paper's own run-to-run spread (about 17%); Figure 8's LSQ column differs slightly
from rows 40-44; row 36's `e` = 8.82e-2 and row 2's `e_nh` = 0.64 are probably typos (8.82e-3,
6.4e-2); the text says rows 14 and 29 exceed the tolerance, but only row 29 does.

### How this repo reproduces it, and where it does not match

`run_paper_experiments.sh` (`make paper-experiments`) covers Table 1 (Hessian runs only), Table 2
and Figure 8, through `afinni_launcher.py`. On this branch the launcher works with the current
API (`TrainingConfig`, `Evaluator`, `AFINNISurrogate.train/predict`, results as JSON with every
effective setting). The study calls the launcher's own functions (`run_case.py` only redirects
`RESULTS_DIR` and the seed) so a case means exactly what the same flags mean to the launcher.
Gaps it fills:

- **`e_nh` column**: not in the script; group `conv_nh` runs every Table 1 row with
  `--error_eval_mode affine_cov_no_hess`.
- **Table 2** as scripted (`--dims 5 10 20 40 80 --tols 0.1`) uses the uncapped heuristics
  `K_0 = 25 1.3^d`, `M = 100 1.3^d` (3e10 clusters at d=80) and would fail; group `lorenz_dims`
  passes the caption's caps per dimension. The paper-era code (`91ddd97`) capped them
  differently, with `cap_initial_augmentation_budget` (`K_0 = 16`, `M = 31,250` at d=80);
  **decision needed** if the caption's values do not reproduce Table 2.
- **Table 3, Figure 9, MPI**: not scripted; groups `rosen_weak`, `iterative`, `mpi`.
- **Seeds**: the launcher has one seed (0); group `seeds` reruns the S-tier rows with seeds 1, 2
  (training/test data and `TrainingConfig.seed`) to measure the noise the verdicts use.

Settings that differ between paper text and code (the study follows the code, since the paper's
numbers came from it):

| Item | Paper text | Code (now and at the paper-era commit `91ddd97`) |
|---|---|---|
| Quadratic spectrum | `Lambda_ii = i^-3` | `anisotropic_quadratic_2`: `Lambda_ii = 5^-i` |
| Lorenz-96 acceptance | rtol = atol | absolute only, `atol = tol * max|f|` over the training data (was `Linf-global`) |
| Lorenz-96 horizon | not stated in the active text (commented table: `t_H = 0.05`) | `--dt 0.1` (passed explicitly) |
| `kappa` | `2 d^2` (8 for d=2) | `--coarse_factor` 10 / 50 / 200; Table 2 and Figure 8 with the default 10 |
| Figure 9 "Gaussian" | anisotropic Gaussian, wave on `[-5, 5]^d`, Rosenbrock tested uniformly | `p_out_iterative_experiment.py`: isotropic Gaussian, wave on `[-4, 4]^d`, Rosenbrock with its own test distribution, `20000 * 1.3^d` initial points |
| Lorenz-96 on GPU | JAX/diffrax (rtol = atol = 1e-3) | the same; on CPU SciPy RK45 (rtol 1e-6): the two flow maps differ by about 1e-3 |

**Deliberate numerical changes since the paper era** (from `git log` after `91ddd97`, 2026-10-08;
the paper's numbers predate them, so leaf and evaluation counts are not expected to match bits):

| Commit | Change | Effect on the re-run |
|---|---|---|
| `3fb78af` | test-problem constants (rotation `Q` of the quadratic and Gaussian, wave direction `q`) drawn with NumPy on every backend; problems no longer reseed the global RNG on each call | on the GPU the quadratic, Gaussian and wave are *different instances* (different rotation) than in the paper; also the random streams after each `f` call change |
| `15c2721` | own k-means\|\| + Lloyd and exact neighbour search replace cuML/scikit-learn | different initial clusters, so different trees |
| `7a1a973` | `Linf-local`/`Linf-global` merged into `Linf`; Lorenz uses `e < tol * max|f|` instead of `e / (max|f| + 1e-12) < tol` | Lorenz decisions at rounding level |
| `7186c93` | a cluster larger than a batch no longer ends the level early (clusters were silently dropped and the run still reported converged) | large GPU runs with small batch caps may now have more leaves than the paper |
| `7dc1c0a` and the GPU determinism work | fixed-order segmented sums replace atomics and batched cuBLAS | GPU results change at rounding level; decisions near the tolerance may flip |

Also expected: AFINNI's trees depend on the batch size (capped from free GPU memory: a laptop GPU
batches differently from a 96 GB H100), the rank count and the steal schedule (docs/USAGE.md); CPU
and GPU draw different random numbers.

### Run matrix and cost

`cases.py` lists every run; `run_paper.sh --dry-run` prints the commands. Counts by group and tier:

| Group | What | S | M | L |
|---|---|---|---|---|
| `conv` | Table 1, Hessian splits (also Figure 7, Table 4, Figure 8 LSQ column) | 29 | 9 | 6 |
| `conv_nh` | Table 1 `e_nh` | 29 | 9 | 6 |
| `seeds` | S rows of `conv` with seeds 1, 2 | 58 | | |
| `lorenz_dims` | Table 2 | 2 | 2 | 1 |
| `rosen_weak` | Table 3 | 1 | 1 | 2 |
| `wave_exact` | Figure 8, exact gradient | 3 | 2 | |
| `iterative` | Figure 9 (d=2, tol 0.05, 10 iterations, 4 functions) | 4 | | |
| `mpi` | Rosenbrock d=5, `N_0` = 1e5, P = 1, 2, 4, 8, CPU ranks; tol 0.05 (M) and 5e-3 (L) | | 4 | 4 |

Tiers come from the paper's evaluation count `N`: **S** `N <= 1e7`, **M** `<= 1e8`, **L** above.
Measured here on a 16-core laptop CPU: every d=2 Table 1 row takes 3-35 s with under 0.9 GB
peak RSS (see below). For the rest, scale from the paper's GH200 numbers: Rosenbrock d=5 at
`N = 2e9` trained in about 40 s on the GPU but the D=10 run peaked at 184 M live points with
100 GB device and 220 GB host memory (Table 5), i.e. roughly 1 kB of host RAM per live point.
Expect S rows in seconds to a few minutes on one GPU and < 8 GB host RAM; M rows minutes to
about an hour and 16-64 GB host RAM (fits a 64 GB WSL machine with a 16-24 GB GPU, maybe
not on 32 GB); L rows need an HPC node with >= 128 GB host RAM (Rosenbrock d=5 at 0.005 and
Quadratic d=10 at 0.01 most likely 200+ GB). The no-Hessian runs make more evaluations than
the Hessian runs of the same row. Each run records `wall_s` and `peak_rss_gib_rank0` in
`OUT/report/cost.csv`; run S first and extrapolate. Watch `free -g` on M/L runs (AGENTS.md:
queues grow with rejected clusters x `2 M`).

Lorenz-96 on the GPU needs a JAX with CUDA (`uv pip install "jax[cuda12]"`); with the default
CPU-only `jaxlib`, `run_paper.sh` runs the Lorenz cases on the CPU and says so
(`--lorenz-device` overrides).

### Collecting and judging (Part A)

`collect_paper.py` writes `OUT/report/comparison.{csv,md}` (one row per reported number: paper,
re-run with seed 0, seed min/max, relative difference, verdict), `checks.{csv,md}`, `slopes.csv`,
`cost.csv`, `runs.csv`, and plots: `convergence_e_in_vs_Nc.png` (paper open markers, re-run
filled, `N_c^(-2/d)` reference), `e_in_vs_tol.png` (with `e_in = tol`), `evaluations_vs_tol.png`,
`iterative_p_out.png`.

**"Same within noise"** for a count or error: `|log(re-run / paper)| <= max(3 sigma, log 1.25)`,
where `sigma` is the standard deviation of `log(value)` over the seed replicates and 25% is the
floor set by the paper's own 17% spread; for `p_o`: within `max(1 point, 30%, 3 sigma)`. A
`differs` verdict on `N`, `N_c` or `p_o` is not a failure by itself: it needs an explanation from
the change list above (on the GPU the quadratic, Gaussian and wave problems are different
instances). The pass/fail checks are the paper's accuracy claims:

| Check | Criterion | Rationale |
|---|---|---|
| A1 | `e_in <= 1.1 tol` for every Table 1, Table 2 and Figure 8 row (A1s: every seed) | the paper's claim; row 29 fails in the paper too (noted) |
| A2 | `e <= 1.25 e_nh` wherever the paper has `e <= e_nh` | Hessian-informed splits are not worse |
| A3 | Table 3: `p_in,good >= 95%` | the paper: "no less than 96%" |
| A4 | fitted slope of `log e_in` vs `log N_c` `<= -2/d + 0.1 (+2 se)` per problem; A4b: within `max(0.2, 2 se)` of the paper's slope over the same rows | Figure 7 (Quadratic d=5 is exempt in the paper) |
| A5 | `converged` and no unresolved points | a run that stopped early proves nothing |
| A6 | MPI: `e_in <= 1.1 tol` on every rank count; `N_c(P) / N_c(1)` in [0.8, 1.25] | AFINNI's trees depend on P; accuracy must not |

A1-A5 failing on rows the paper meets, or a slope clearly shallower than the paper's, means a
regression to investigate; counts that differ while A1-A5 pass are consistent with the changes.

---

## Part B: the reference ART vs AFINNI's ART

### What the reference repository contains (`854924a`)

- `art/`: `builder.py` (`RegressionTreeBuilder`, DFS or BFS traversal, node batches),
  `splitters.py` (`soft_oblique` with optimizers `cpu`, `hybrid_gram`, `cupy_qr`,
  `cupy_cholesky`, `cupy_batched_cholesky`; `hrt` hinge splitter for affine leaves),
  `sampling.py` (`hit_and_run`, `parallel_numba` multithreaded chains; thinning 20, 50 chains),
  `models.py` (affine / polynomial ridge, `ridge = 0` by default), `temperature.py`, `presets.py`
  (`DEFAULT_TREE_PRESET`: degree 1, max pointwise relative error, tol 1e-2, depth 10,
  multiplier 50, exact box root; `BFS_GPU_TREE_PRESET`: BFS + `cupy_batched_cholesky`).
- `examples/benchmark_functions.py`: quadratic (`beta=0`, `w=0`, spectrum `i^-3`, random
  rotation, `[-3,3]^d`), Gaussian (`[-1,1]^d`, condition 16), Gaussian mixture, plane wave
  (frequency 4, `[-3,3]^d`), spherical piecewise (fixed affine pieces, d=2-5, offset 4.15),
  Rosenbrock (`a=1, b=100`, `[-2,3]^d`), Rastrigin (`[-5.12,5.12]^d`), active-subspace Gaussian.
- Drivers: `tree_high_dim_benchmark.py` (any dimension; the one `art_comparison.py` uses),
  `tree_2D_benchmark.py`, `gaussian_testbench.py` (active-subspace Gaussian),
  `tree_shallow_water_benchmark.py` (Clawpack), `soft_oblique_*` stress tests;
  `tasmanian/` and `random_forest/` comparisons with **recorded TACC presets**
  (`tasmanian/oracle_presets.json`: Rosenbrock d=5 offset 1; Rastrigin d=3/4/5 on half-widths
  5.12/2.5/1.5, offset 1, depth 22/21/20, degree 2; plane wave d=3/5/8 with dense-sign normals,
  offset 2, degree 1; spherical piecewise d=2-5, depth 20; shallow water 5-D, depth 17).
- `tests/`: 35 files, about 300 tests, including CUDA kernels (`test_cuda_*`, need `c++` on PATH
  for their CPU references).
- Convergence there: no sweep driver; each benchmark run reports leaves, oracle queries, depth,
  test relative L2, RMSE, max absolute and pointwise-relative quantiles on `10000 d` uniform
  test points (seed 19) and leaf terminal statuses. Convergence is measured by sweeping
  `--error-tolerance` (or `--max-depth`) and reading those.

`scripts/art_comparison.py` still works with this commit (checked here on 7 problems, both sides).
It builds the reference's target, domain and test set once and scores both codes with the
reference's evaluator; AFINNI's ART gets affine (`affine_lsq`) or quadratic (`quadratic_lsq`)
leaves, the pointwise relative rule with floor 1e-12, `--max-depth` levels, multiplier 50, a box
root with `ceil(50 (d+1))` uniform points and the soft split search. `run_art_case.py` imports it,
runs one side per process (so each side can have its own device, timeout and done-marker) and
adds RMSE, max absolute error and all quantiles to its summary.

### Study design

Problems (`art_cases.py`), all with the reference's recorded presets; offsets keep `|f|` away from
0, where a pointwise relative tolerance cannot be met by any leaf (with offset 0 the quadratic is
0 at the origin and both codes would only hit the depth limit there):

| Problem | Arguments | d | Leaf degree | Tolerances | Max depth |
|---|---|---|---|---|---|
| quadratic | `--offset 1` | 2, 3, 4, 5 | 1 | 0.1, 0.05, 0.02, 0.01, 0.005 | 20 |
| gaussian | (positive) | 2, 3, 4, 5 | 1, 2 | 0.1 ... 0.005 | 20 |
| gaussian_mixture | | 2, 3 | 1 | 0.1 ... 0.01 | 20 |
| plane_wave | `--plane-wave-frequency 4 --offset 2`, dense-sign normal | 3, 5, 8 | 1 | 0.1 ... 0.001 | 24 |
| rosenbrock | `--offset 1` | 2, 3, 4, 5 | 1, 2 | 0.1 ... 0.01 | 20 |
| rastrigin | `--offset 1`, half-width 5.12 / 2.5 / 1.5 | 3, 4, 5 | 2 | 0.1, 0.05, 0.02 | 22 / 21 / 20 |
| spherical_piecewise | fixed affine pieces, offset 4.15 | 2, 3, 4, 5 | 1 | 0.01 | sweep 6 ... 16 |

Each (problem, d, degree, tolerance) runs with seeds 1, 2, 3 for both codes (the seed changes the
tree's random numbers and, for quadratic/Gaussian, the rotation; both codes always see the same
instance, so comparisons are paired). Matched settings: same target, domain, test set, seed,
leaf degree, tolerance semantics (max pointwise relative error on the node's samples, floor
1e-12), max depth, sample multiplier 50 (`n = ceil(50 p)` points per node), soft-oblique split
search with the same 200-iteration cap (`--afinni-soft-iters 200`). Groups:

| Group | Runs | Purpose |
|---|---|---|
| `conv` | 130 (problem, d, degree, tol) cells x 3 seeds = 390 runs per code (S 300, M 60, L 30) | the convergence curves |
| `afinni_default` | AFINNI side only, seed 1, its default 40 soft iterations | the shipped default against the same reference runs |
| `depth` | spherical piecewise, 4 d x 6 depths x 3 seeds x 2 codes | a discontinuous target: convergence of the share above tol with depth |
| `exact_target` | quadratic leaves on the quadratic, d=2, 3, 5 | must stop at the root, error ~ 1e-10 |
| `mpi` | AFINNI's ART, Rosenbrock d=3, P = 1, 2, 4 with stealing | identical tree on any rank count |
| `repeat` | two cases rerun, both codes | run-to-run determinism of each code |
| `gpu_vs_cpu` | 3 AFINNI cases also on the CPU (only with `--device gpu`) | GPU vs CPU agreement |

Plus `exact_checks.py` (always run first), see below. Recorded per run: leaves, oracle queries,
depth, test relative L2, RMSE, max absolute error, max / median / q95 / q99 pointwise relative
error, share of test points above the tolerance, leaf terminal reasons (depth-limited leaves),
build and wall time, peak RSS.

**Cost.** The reference on the CPU is the slow side (one core for the default `hit_and_run`
sampler and `cpu` optimizer, DFS). Measured here (CPU, d = 2): about 50 ms per reference leaf
(quadratic d=2, tol 0.02: 134 leaves in 7.1 s; AFINNI's ART 127 leaves in 3.3 s), under 0.3 GB
per run. Rosenbrock d=3 at tol 0.05, depth 20 had not finished after 11 minutes on one core
(it was stopped when the session ended): M and L cells take hours on the reference's CPU path. Use `--ref-sampler parallel_numba` (threads)
on CPU nodes and `--ref-device gpu` (BFS + `cupy_batched_cholesky`) on the GPU machine; AFINNI's
ART is GPU-batched (`--device gpu`). Memory stays small (< 2 GB) except d=8 plane waves and
L-tier Rastrigin. Run `--max-tier S` first; `OUT/report/runs.csv` has `build_s`, `wall_s` and
`peak_rss_gib` per run for extrapolation (time roughly proportional to leaves).

### What differences are expected, and what would indicate a bug

Expected (not bugs): different trees and leaf counts (different random generators: the reference
uses per-node `default_rng` seeds, AFINNI's ART counter-based streams keyed by node; the
reference tunes a temperature per node from nearest-neighbour distances, AFINNI's ART uses 0.1 x
the std of the node's scores; AFINNI starts the soft search from a covariance-Hessian/bins
direction; side-size rules differ: `min_side_fraction 0.05` vs `p + min_side_margin`); evaluation
counts that include AFINNI's centroid evaluations (one per split node with the covariance-Hessian
split); a non-zero share of test points above the tolerance in *both* codes (the stopping test
is on the node's samples, not a certificate); depth-limited leaves on the spherical interface in
both codes; timings.

AFINNI's leaf fits are regularized and the reference's are not: `affine_lsq` adds Tikhonov
`1e-8 * trace` of the node's scatter matrix, `quadratic_lsq` a ridge of `1e-10` of the Gram's
mean diagonal. On a cube this biases the fit by about 1e-8 relative (harmless), but it grows like
`1e-8 * aspect^2` on thin nodes, and repeated parallel cuts make thin nodes (plane waves,
Rosenbrock's valley). The thin-box check below shows the effect. A systematic accuracy gap that
grows at tight tolerances on those problems would trace to it.

Indicators of a bug in AFINNI's ART (the checks of `collect_art.py`; medians over seeds):

| Check | Criterion |
|---|---|
| B1 | share of test points above tol (AFINNI) `<= max(2 x reference, reference + 0.5%)`; B1b: AFINNI `q99 > 1.2 tol` where the reference has `q99 <= tol` |
| B2 | leaves AFINNI / reference in [0.5, 2] (WARN), [1/3, 3] (FAIL outside) at equal tolerance |
| B3 | slopes of log(rel L2) and log(q99) vs log(leaves) agree: `|s_AFINNI - s_ref| <= 2 sqrt(se_A^2 + se_R^2) + 0.15`; both are also listed beside the theory `-(k+1)/d` (`-(k+1)` for the rank-1 plane wave) |
| B4 | relative L2 at the same leaf count (each code's fit, at the reference's median leaf count) AFINNI / reference `<= 2` (FAIL above, WARN below 0.5) |
| B5 | depth-limited leaf share AFINNI `<=` reference + 5% |
| B6 | exact sub-tests (below) and `exact_target`: 1 leaf, max relative error `<= 1e-6` |
| B7 | AFINNI's ART: identical leaves, evaluations and every metric on 1, 2, 4 ranks with stealing |
| B8 | each code reproduces its own run (same seed); B8b AFINNI GPU vs CPU: same leaves and evaluations, errors within 1e-8 (WARN if only the errors differ) |

A failure of B1, B3 or B4 with B2 in range means AFINNI's ART converges, but to the wrong
accuracy or at the wrong rate; B2 alone out of range with B1, B3, B4 passing means a different
but equally accurate cost (a split-search difference, worth reporting, not a correctness bug).

### Exact agreement sub-tests (`exact_checks.py`)

Bit-identical trees are impossible below the root (different generators and split searches), and
no 1-D or axis-aligned case removes that (both codes still search splits stochastically). Two
tests are exact:

1. **Root fit.** The reference builds the root only (`--max-depth 0 --store-samples`); its root
   sample (`ceil(50 p)` points) is given to AFINNI's ART with `max_levels=0` (exactly `n` points:
   no top-up). Both must give the same leaf model (max prediction difference on 10,000 test
   points `<= 1e-6` relative, the ridges allow about 1e-8), the same training error, the same
   accept/reject decision, and AFINNI must evaluate `f` only at those points (+1 centroid). NumPy
   `lstsq` is the third opinion. Problems: all six, d = 2, 3, 5, degrees 1 and 2, plus an exactly
   representable case.
2. **Thin box.** An exactly affine target `f = 2 + x_0 + x_1 / w` on `[0,1] x [0,w]`,
   `1/w` = 1 ... 1e4, fitted by each code's leaf model on the same points. Both should reproduce
   it to rounding (criterion `<= 1e-6`).

---

## Smoke tests run here

All on this machine's CPU (16 cores, no GPU), `AFINNI_DEVICE=cpu`, outputs in the session scratch
directory. The final smoke runs ran one at a time under
`systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=400% nice -n 10 timeout 600` with
`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2` (the machine was shared with other jobs); the d=2
full-size runs, the full exact checks and the paper-era comparison ran earlier without the cap.

**Scripts end to end.**

| Run | Outcome |
|---|---|
| `paper/run_paper.sh --smoke` (11 cases: every group except `mpi`, tiny sizes) | all 11 ran, 66 s, peak RSS 0.6 GB; collector built every table and plot (all paper rows `missing`, as they must be at tiny sizes) |
| Part A `mpi` smoke (`mpiexec -n 2`, earlier in the session) | ran and was collected; a later repeat failed only because the host was out of memory (AFINNI's own guard, see Known gaps) |
| `art/run_art.sh --smoke` (exact checks + 18 runs: 6 problems x 2 codes, spherical depth, exact target, AFINNI's ART on 1 and 2 ranks) | all ran, 43 s, peak RSS 0.25 GB; checks: 29 pass, 12 missing (slopes need >= 3 tolerances), 2 FAIL (the thin-box finding below) |
| `--dry-run` of both full matrices | Part A 172 runs, Part B 1070 runs (with `--device gpu`), commands and `AFINNI_DEVICE`/`AFINNI_MPI` as intended |

**Part A at full paper size, d = 2 only (CPU, seeds 0 and 1, 3-35 s per run, < 0.9 GB).** All 15
Table 1 rows for Gaussian, Rosenbrock and Wave d=2, with and without Hessian splits:

- `N`, `N_c` and `e_in` agree with the paper within the 25% noise band in every row (e.g. Rosenbrock
  #21: `N_c` 759 vs 746, `N` 2.32e5 vs 2.27e5; Wave #39: 1,167 vs 1,177, 3.01e5 vs 3.03e5), except
  Rosenbrock #22 `e_in`, 27% *below* the paper. A1 (`e_in <= 1.1 tol`), A5 and the slope checks
  pass: slopes -1.28 (paper -1.31), -1.06 (-1.06), -1.39 (-1.45) for Gaussian, Rosenbrock, Wave,
  all steeper than the theoretical -1.
- `p_o` (and hence `e`) is higher than the paper at tight tolerances: Wave #38/#39 2.4% / 7.5% vs
  0.10% / 0.18%, Rosenbrock #21-24 about 1.7-2x. The paper-era code (`91ddd97`, extracted with
  `git archive`, run on the CPU with an exact brute-force stand-in for `faiss`) gives the same:
  Wave tol 0.005 `N_c` 1,177 (the paper's value exactly), `N` 3.03e5, but `p_o` 7.57%; Rosenbrock
  tol 0.1 `N_c` 344 (exactly), `p_o` 1.95% vs 1.59%. So the `p_o` gap is not a regression on this
  branch: the paper's `p_o` came from its GPU inference path (cuML neighbours, GPU instances), and
  only the GPU re-run can confirm it.
- `e_nh` (no Hessian) is 2-5x *lower* than the paper's in every row (e.g. Rosenbrock #22: 5.3e-3
  vs 3.46e-2), with 2-8x more leaves than the Hessian runs. The current `affine_cov_no_hess`
  baseline is much better than the paper's. Consequently A2 fails on Rosenbrock #24 and Wave
  #38-39, where the Hessian run's higher `p_o` makes its global `e` exceed the no-Hessian `e`
  (its `e_in` is still lower: 1.04e-3 vs 1.31e-3 at Rosenbrock #24).

**Part B (CPU).**

- `scripts/art_comparison.py` works with the reference at `854924a` for every problem of the
  matrix (6 benchmarks + spherical piecewise, degrees 1 and 2, 1 and 2 ranks).
- Small converged cases agree closely, e.g. quadratic d=2 tol 0.02: reference 134 leaves, rel L2
  4.24e-3, q99 0.0145, 0.14% of test points above tol; AFINNI's ART 127 leaves, 4.16e-3, 0.0142,
  0.11%. Gaussian d=2 tol 0.02: 15 vs 13 leaves, 4.06e-3 vs 4.01e-3.
- B7: AFINNI's ART gave the identical tree and metrics on 1 and 2 ranks with work stealing.
- Exact root fit (`exact_checks.py`, full set): 37/37 pass. AFINNI's root model matches the
  reference to 1e-10 (quadratic leaves) and 1e-9-3e-8 (affine leaves, the Tikhonov term), with the
  same training error and decision, and evaluates `f` only at the given points plus one centroid;
  NumPy `lstsq` matches the reference to 1e-15. The exactly representable quadratic is accepted
  at the root by both (AFINNI error 7e-10, reference 9e-15).
- **Thin box: AFINNI's ART fails to reproduce an exactly affine function on thin nodes; the
  reference reproduces it to 1e-14 at every aspect ratio.** Max relative error of AFINNI's leaf:

  | aspect ratio | 1 | 10 | 100 | 1e3 | 1e4 |
  |---|---|---|---|---|---|
  | affine leaf | 1.2e-8 | 2.9e-7 | 2.8e-5 | 2.8e-3 | 0.14 |
  | quadratic leaf | 3.5e-10 | 2.0e-9 | 1.6e-7 | 1.7e-5 | 1.7e-3 |

  The cause is the regularization in `src/components/local_models.py` (`AffineLSQModel` through
  `solve_regularized_covariance_inplace`: `lambda = 1e-8 * trace`; `QuadraticLSQModel`:
  `1e-10 * trace / p`), which on a node of aspect ratio `a` biases the slope along its thin side by
  about `1e-8 a^2`. Such nodes are rejected (`max_depth` at depth 0 in the probe) where the
  reference accepts them. At the study's tolerances (>= 1e-3) it matters once nodes reach aspect
  ratios of about 300 (affine) or 3,000 (quadratic). The plane-wave sweep down to 1e-3 is there to
  show whether real trees get there (look for B2, B4, B5 failures growing as the tolerance
  tightens).

## Known gaps and risks

- **AFINNI's host-memory guard is system-wide.** Before each level it checks the machine's
  available memory and keeps 3 GB free; on a machine shared with other jobs even tiny runs stop
  with `MemoryError: ... only 0.3 GiB is available` (seen here twice). On WSL give the VM enough
  memory (`.wslconfig`), and run one sweep at a time. To protect a shared desktop, wrap a sweep in
  `systemd-run --user --scope -p MemoryMax=24G -p CPUQuota=800% nice -n 10 <script>` (sizes to
  taste); the scripts themselves impose no cap beyond `RUN_TIMEOUT`.
- **Not reproducible here**: Table 4's other methods (Forest/GPR/FALKON/NN/Tasmanian need
  Optuna, GPyTorch, FALKON, Tasmanian; the reference ART's `tasmanian/` and `random_forest/`
  scripts compare against ART, not AFINNI); the speed figures and the MPI speedups (only their
  accuracy side is checked); Table 5's D=10 run progression (memory).
- **Instances differ on the GPU** (`3fb78af`): exact agreement with the paper's counts is not
  expected for the quadratic, Gaussian and wave; on the CPU the problems are the paper-era
  NumPy instances, but the paper ran on GPUs. Compare GPU re-runs to the paper with the noise
  rule and A1-A6, not bits.
- **Paper-era settings unknown**: which `coarse_factor` and caps Table 2 used, and the Lorenz
  horizon in the text. The study uses the script's and caption's values; decide if they fail.
- **Lorenz-96 GPU** needs `jax[cuda12]`; otherwise it runs on the CPU with SciPy, a slightly
  different (more accurate) flow map.
- **Batch-size dependence**: AFINNI's trees depend on the GPU batch cap; a 16 GB GPU and the
  paper's 96 GB H100 give statistically, not bitwise, equal trees.
- **Reference sampler and optimizer variants** (`parallel_numba`, CUDA optimizers) give different
  trees from the default CPU path; record which one ran (it is in each JSON's `art_config`).
- **Not covered by `art_comparison.py`**: the reference's active-subspace Gaussian testbench,
  shallow water (Clawpack), the HRT splitter, temperature tuning strategies.
- **Pointwise relative tolerance near f = 0**: with offset 0 the reference's quadratic,
  Rosenbrock and Rastrigin cannot meet the tolerance at their zeros; the study uses the recorded
  offsets. Running offset 0 would mostly measure the depth limit.

## Decisions for you

1. **AFINNI's ART leaf regularization (Part B finding).** The Tikhonov term (`1e-8 * trace`, from the
   AFINNI manuscript's Eq. 3) and the quadratic ridge make AFINNI's ART leaves inexact on thin
   nodes, unlike the reference's unregularized fits. Decide whether ART leaves should keep it,
   scale it by the smallest rather than the summed eigenvalue, or drop it (the reference uses
   ridge 0 with an rcond-based solver choice). Any change alters AFINNI's ART trees (and, if
   shared, AFINNI's): a deliberate numerical change to list and justify. Running Part B first
   shows how much it matters on the real problems.
2. **The `e_nh` column.** The current no-Hessian baseline is 2-5x more accurate than the paper's
   (d=2 rows). Decide whether the paper's `e_nh` came from different code or settings (worth a
   look at how `M = I` was run then), and whether the paper's claim "Hessian splits beat no
   Hessian" should be restated as "fewer leaves for the same `e_in`" (true here: 2-8x fewer).
3. **`p_o` vs the paper.** CPU runs, current and paper-era, give up to 40x the paper's `p_o` at tight
   tolerances with the same `N_c`. If the GPU re-run also does, the paper's `p_o` and `e` columns
   need re-measuring; if not, the GPU inference path (neighbour search) is what to examine.
4. **Table 2 settings.** The caption's caps (`K_0 <= 1000`, `M <= 500`) versus the paper-era
   code's `cap_initial_augmentation_budget` (`K_0` down to 16, `M` up to 31,250), and
   `coarse_factor` 10 (script) versus `2 d^2` (text) for d up to 80. The study uses the caption
   and the script; switch if Table 2 does not reproduce.
5. **Lorenz-96 on the GPU**: install `jax[cuda12]` (the paper's JAX flow map, rtol 1e-3) or accept
   the CPU SciPy flow map (rtol 1e-6) for those rows.
6. **Paper text vs code** (fix the text or the code): quadratic spectrum `i^-3` (text) vs `5^-i`
   (code); "rtol = atol in every experiment" vs Lorenz's absolute `tol * max|f|`; Figure 9's
   isotropic Gaussian, wave domain `[-4, 4]` and Rosenbrock test distribution; the
   typos and the Table 1b / Table 4 Lorenz mismatch listed above.
7. **Pass thresholds.** The 25% noise floor, `1.1 tol` for A1, and the B-check bands are my
   proposals; adjust them before reading the results as final.
