# Accuracy and convergence study: AFINNI paper re-run, reference ART vs this library's ART

Prepared on 2026-11-16 against this repository's `refinement-engine` @ `f60eb2b` and the reference ART
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
- **B.** Run a convergence study of the reference ART and this library's ART (`algorithm="art"`) on the
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
| `paper/run_manuscript.sh` | **every manuscript experiment with the current code**, then a record of the results (see [Recording the results](#recording-the-results)) |
| `paper/record_results.py` | writes a record: the numbers, the commit, the machine and the libraries |
| `results/` | the records, one folder per recording, listed in `results/INDEX.md` (tracked in git) |
| `art/run_art.sh` | **Part B top level** |
| `art/art_cases.py` | Part B run matrix |
| `art/run_art_case.py` | one side of `scripts/art_comparison.py` (imported, not copied), saved as JSON with RMSE, max abs error, all quantiles, wall time, peak RSS |
| `art/exact_checks.py` | the exact-agreement sub-tests (root fit, thin-box fit) |
| `art/collect_art.py` | paired tables, slopes, checks B1-B8, plots |
| `bisect/run_at.sh` | one launcher case at any commit, extracted with `git archive` (see [Where the differences come from](#where-the-differences-come-from)) |
| `bisect/capture_launcher.py` | runs that commit's own launcher unchanged and saves its test points, predictions, out-of-region mask, ground truth, leaf centres and trust matrices |
| `bisect/era_old.py` | one case through the pre-November API, for settings the old launchers hard-code (acceptance metric) |
| `bisect/score.py` | scores saved runs with one metric; re-scores `p_o` for other candidate counts and against the exact union of all trust regions |

## Quick start

```bash
cd ~/github-repos/affini            # the repository root; scripts use .venv/bin/python
uv sync --extra gpu --extra mpi       # GPU machine; on a cluster with its own MPI: --extra mpi-system and the cluster's setup script in scripts/hpc/

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
commands with their `SURROGATE_DEVICE`/`SURROGATE_MPI`), `--seeds`, `--nps`. Every command sets
`SURROGATE_DEVICE` and `SURROGATE_MPI` explicitly. Environment overrides: `ART_ROOT`, `PYTHON`,
`MPIEXEC` (e.g. `MPIEXEC=ibrun` or `"srun --mpi=pmix"`), `STUDY_OUT`, `RUN_TIMEOUT` (default 6h
per run, `0` = none). Output layout: `OUT/results/...` (JSON), `OUT/logs/<case>.log`,
`OUT/done/<case>.done` (a rerun skips done cases; delete a marker to redo one),
`OUT/provenance.txt` (both commits, Python, backend, GPU), `OUT/report/` (tables, plots).
On SLURM, wrap one tier or group per job; the done-markers make resubmission safe.

**What to run first on the GPU.** (1) Part A `--max-tier S` as above: Table 1's d=2 rows now run
with 50 candidates, as the paper did (D1), plus group `conv_k10` with the script's 10. (2) The
paper-era GPU check below (seconds per run), which decides D4 and possibly D3. Then M, L and Part B.

**Paper-era runs on the GPU** (D4: were the Rosenbrock and Gaussian rows GPU runs, before or after
`e37c315`? D3: does the old GPU path give the paper's `e_nh`?). Each old commit gets its own locked
environment (cupy, cuml-cu12 25.10, faiss-cpu, jax with CUDA):

```bash
export BISECT_DIR=~/afinni-bisect DEVICE=gpu CAP=0 FAISS_SHIM=   # no CPU stand-ins on the GPU
for c in 91ddd97 a6844cb; do
  mkdir -p $BISECT_DIR/$c && git archive $c | tar -x -C $BISECT_DIR/$c && (cd $BISECT_DIR/$c && uv sync)
done
R=studies/convergence/bisect/run_at.sh
P=$BISECT_DIR/91ddd97/.venv/bin/python   # paper-era code, Tikhonov trust regions
for t in 0.1 0.005; do
  PYTHON=$P $R 91ddd97 gpu_r$t -- --experiment rosenbrock --dims 2 --tols $t --error_eval_mode affine_cov_hess --coarse_factor 50
  PYTHON=$P $R 91ddd97 gpu_g$t -- --experiment anisotropic_gaussian --dims 2 --tols $t --error_eval_mode affine_cov_hess --coarse_factor 50
done
PYTHON=$P $R 91ddd97 gpu_g5d0.1 -- --experiment anisotropic_gaussian --dims 5 --tols 0.1 --error_eval_mode affine_cov_hess --coarse_factor 50
PYTHON=$P $R 91ddd97 gpu_w0.005nh -- --experiment wave --dims 2 --tols 0.005 --error_eval_mode affine_cov_no_hess --coarse_factor 10
PYTHON=$P $R 91ddd97 gpu_r0.02nh -- --experiment rosenbrock --dims 2 --tols 0.02 --error_eval_mode affine_cov_no_hess --coarse_factor 10
P=$BISECT_DIR/a6844cb/.venv/bin/python   # shrinkage trust regions; its launcher passes kappa = d^2
for t in 0.1 0.02 0.005; do
  PYTHON=$P $R a6844cb gpu_r${t}_shrink --launcher experiments.py -- --experiment rosenbrock --dims 2 --tols $t --error_eval_mode affine_cov_hess
done
PYTHON=$P $R a6844cb gpu_r0.02nh_shrink --launcher experiments.py -- --experiment rosenbrock --dims 2 --tols 0.02 --error_eval_mode affine_cov_no_hess
.venv/bin/python studies/convergence/bisect/score.py $BISECT_DIR/runs/gpu_*.npz --kappas 10,50
```

The commits before `d7ec052` pick the GPU whenever `cupy` imports (they ignore `SURROGATE_DEVICE`).
D4 is confirmed if the GPU trees match the paper: Rosenbrock #20 `N` about 1.39e5, `N_c` 344,
`e_in` 3.4e-2, and #24 `N_c` 8,446; Gaussian `N_c` 48 / 359 (#10 / #14) and 173 at d=5 (#15). The
`p_o` column printed as `p_o_k10` for the `a6844cb` runs should then give the paper's Rosenbrock
1.59 / 1.45 / 2.36%. D3 is explained only if the no-Hessian runs give `e` near 8.26e-3 (Wave #39)
and 3.46e-2 (Rosenbrock #22). Each run saves `runs/<tag>.npz` and `.log`.

---

## Recording the results

The manuscript's numbers are now re-measured with the current code, and every set of results is
kept with what produced it, so that later numbers can be compared with earlier ones.

```bash
MACHINE="GH200 node" studies/convergence/paper/run_manuscript.sh
```

- **What it runs:** every experiment of the manuscript on the GPU, at the L tier: Table 1 at all
  dimensions with and without Hessian splits, Table 2, Table 3, the Figure 8 and Figure 9 runs, the
  MPI runs and the seed replicates (172 cases). Not the 10-candidate diagnostic (`conv_k10`), and
  not Table 4's other methods. Options of `run_paper.sh` given after it override the defaults.
- **Before it starts:** it refuses a working tree with modified tracked files, so the recorded
  commit is the code that ran (`ALLOW_DIRTY=1` overrides). `MACHINE` is a neutral description of
  the hardware; no host name is recorded.
- **Resuming:** finished cases are skipped, so an interrupted run continues with the same command.
  The record is rewritten from whatever has finished at the end of each invocation.
- **The record:** `studies/convergence/results/<date>_<commit>_<machine>/`, meant to be committed:

| File | Contents |
|---|---|
| `RESULTS.md` | the provenance, then every number, one table per manuscript table |
| `manuscript_results.csv` | one row per number the manuscript reports: the value measured (seed 0), the seed replicates' range, and the paper's value and the relative difference beside it |
| `provenance.json` | the commit the runs started at (and a warning if they started at several, or with modified files), the `uv.lock` blob, CPU, cores, RAM, OS, GPU, driver, CUDA runtime, Python, library and MPI versions |
| `slopes.csv`, `checks.csv`, `cost.csv` | convergence slopes, accuracy checks, wall time and peak memory per case |

`results/INDEX.md` lists every record with its date, commit, machine and how many of the reported
numbers it measured.

## Part A: the AFINNI manuscript

### What the paper reports

Experiments ran on a GH200 node ( CuPy, cuML k-means, Python 3.13, CUDA 12.9). Settings
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
API (`TrainingConfig`, `Evaluator`, `AdaptiveSurrogate.train/predict`, results as JSON with every
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
| `kappa` | `2 d^2` (8 for d=2) | `--coarse_factor` 10 / 50 / 200 in the script; Table 2 and Figure 8 with the default 10. **Table 1's d=2 rows were measured with 50** (see below); the study now runs d=2 with 50 and keeps 10 as group `conv_k10` |
| Figure 9 "Gaussian" | anisotropic Gaussian, wave on `[-5, 5]^d`, Rosenbrock tested uniformly | `p_out_iterative_experiment.py`: isotropic Gaussian, wave on `[-4, 4]^d`, Rosenbrock with its own test distribution, `20000 * 1.3^d` initial points |
| Lorenz-96 on GPU | JAX/diffrax (rtol = atol = 1e-3) | the same; on CPU SciPy RK45 (rtol 1e-6): the two flow maps differ by about 1e-3 |

**Deliberate numerical changes since the paper era** (from `git log` after `91ddd97`, 2026-10-08;
the paper's numbers predate them, so leaf and evaluation counts are not expected to match bits):

| Commit | Change | Effect on the re-run |
|---|---|---|
| `3fb78af` | test-problem constants (rotation `Q` of the quadratic and Gaussian, wave direction `q`) drawn with NumPy on every backend; problems no longer reseed the global RNG on each call | on the GPU the quadratic, Gaussian and wave are *different instances* (different rotation) than in the paper; also the random streams after each `f` call change |
| `15c2721`, `0edf0e6` | own k-means\|\| + Lloyd and exact neighbour search replace cuML/scikit-learn (`15c2721` adds them, `0edf0e6` trains with them) | different initial clusters, so different trees (on the CPU from `0edf0e6`; Wave #39: `N_c` 1,177 -> 1,167) |
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

Lorenz-96 on the GPU uses JAX's CUDA build, which the `gpu` extra installs on Linux
(`uv sync --extra gpu`); with a CPU-only `jaxlib` (other platforms, or an environment without
the extra), `run_paper.sh` runs the Lorenz cases on the CPU and says so (`--lorenz-device`
overrides).

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

## Part B: the reference ART vs this library's ART

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
  `tasmanian/` and `random_forest/` comparisons with **recorded cluster presets**
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
reference's evaluator; this library's ART gets affine (`affine_lsq`) or quadratic (`quadratic_lsq`)
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
search with the same 200-iteration cap (`--library-soft-iters 200`). Groups:

| Group | Runs | Purpose |
|---|---|---|
| `conv` | 130 (problem, d, degree, tol) cells x 3 seeds = 390 runs per code (S 300, M 60, L 30) | the convergence curves |
| `library_default` | this library's side only, seed 1, its default soft iterations (`ARTSettings.soft_iters`, 200 since the comparison below; 40 before) | the shipped default against the same reference runs |
| `depth` | spherical piecewise, 4 d x 6 depths x 3 seeds x 2 codes | a discontinuous target: convergence of the share above tol with depth |
| `exact_target` | quadratic leaves on the quadratic, d=2, 3, 5 | must stop at the root, error ~ 1e-10 |
| `mpi` | this library's ART, Rosenbrock d=3, P = 1, 2, 4 with stealing | identical tree on any rank count |
| `repeat` | two cases rerun, both codes | run-to-run determinism of each code |
| `gpu_vs_cpu` | 3 cases of this library's ART also on the CPU (only with `--device gpu`) | GPU vs CPU agreement |

Plus `exact_checks.py` (always run first), see below. Recorded per run: leaves, oracle queries,
depth, test relative L2, RMSE, max absolute error, max / median / q95 / q99 pointwise relative
error, share of test points above the tolerance, leaf terminal reasons (depth-limited leaves),
build and wall time, peak RSS.

**Cost.** The reference on the CPU is the slow side (one core for the default `hit_and_run`
sampler and `cpu` optimizer, DFS). Measured here (CPU, d = 2): about 50 ms per reference leaf
(quadratic d=2, tol 0.02: 134 leaves in 7.1 s; this library's ART 127 leaves in 3.3 s), under 0.3 GB
per run. Rosenbrock d=3 at tol 0.05, depth 20 had not finished after 11 minutes on one core
(it was stopped when the session ended): M and L cells take hours on the reference's CPU path. Use `--ref-sampler parallel_numba` (threads)
on CPU nodes and `--ref-device gpu` (BFS + `cupy_batched_cholesky`) on the GPU machine; this library's
ART is GPU-batched (`--device gpu`). Memory stays small (< 2 GB) except d=8 plane waves and
L-tier Rastrigin. Run `--max-tier S` first; `OUT/report/runs.csv` has `build_s`, `wall_s` and
`peak_rss_gib` per run for extrapolation (time roughly proportional to leaves).

### What differences are expected, and what would indicate a bug

Expected (not bugs): different trees and leaf counts (different random generators: the reference
uses per-node `default_rng` seeds, this library's ART counter-based streams keyed by node; the
reference tunes a temperature per node from nearest-neighbour distances, this library's ART uses 0.1 x
the std of the node's scores; this library's ART starts the soft search from a covariance-Hessian/bins
direction; side-size rules differ: `min_side_fraction 0.05` vs `p + min_side_margin`); evaluation
counts that include this library's centroid evaluations (one per split node with the covariance-Hessian
split); a non-zero share of test points above the tolerance in *both* codes (the stopping test
is on the node's samples, not a certificate); depth-limited leaves on the spherical interface in
both codes; timings.

This library's leaf fits are regularized and the reference's are not: `affine_lsq` adds Tikhonov
`1e-8 * trace` of the node's scatter matrix, `quadratic_lsq` a ridge of `1e-10` of the Gram's
mean diagonal. On a cube this biases the fit by about 1e-8 relative (harmless), but it grows like
`1e-8 * aspect^2` on thin nodes, and repeated parallel cuts make thin nodes (plane waves,
Rosenbrock's valley). The thin-box check below shows the effect. A systematic accuracy gap that
grows at tight tolerances on those problems would trace to it.

Indicators of a bug in this library's ART (the checks of `collect_art.py`; medians over seeds):

| Check | Criterion |
|---|---|
| B1 | share of test points above tol (this library) `<= max(2 x reference, reference + 0.5%)`; B1b: this library's `q99 > 1.2 tol` where the reference has `q99 <= tol` |
| B2 | leaves this library / reference in [0.5, 2] (WARN), [1/3, 3] (FAIL outside) at equal tolerance |
| B3 | slopes of log(rel L2) and log(q99) vs log(leaves) agree: `|s_lib - s_ref| <= 2 sqrt(se_lib^2 + se_ref^2) + 0.15`; both are also listed beside the theory `-(k+1)/d` (`-(k+1)` for the rank-1 plane wave) |
| B4 | relative L2 at the same leaf count (each code's fit, at the reference's median leaf count) this library / reference `<= 2` (FAIL above, WARN below 0.5) |
| B5 | depth-limited leaf share of this library `<=` reference + 5% |
| B6 | exact sub-tests (below) and `exact_target`: 1 leaf, max relative error `<= 1e-6` |
| B7 | this library's ART: identical leaves, evaluations and every metric on 1, 2, 4 ranks with stealing |
| B8 | each code reproduces its own run (same seed); B8b this library's ART GPU vs CPU: same leaves and evaluations, errors within 1e-8 (WARN if only the errors differ) |

A failure of B1, B3 or B4 with B2 in range means this library's ART converges, but to the wrong
accuracy or at the wrong rate; B2 alone out of range with B1, B3, B4 passing means a different
but equally accurate cost (a split-search difference, worth reporting, not a correctness bug).

### Exact agreement sub-tests (`exact_checks.py`)

Bit-identical trees are impossible below the root (different generators and split searches), and
no 1-D or axis-aligned case removes that (both codes still search splits stochastically). Two
tests are exact:

1. **Root fit.** The reference builds the root only (`--max-depth 0 --store-samples`); its root
   sample (`ceil(50 p)` points) is given to this library's ART with `max_levels=0` (exactly `n` points:
   no top-up). Both must give the same leaf model (max prediction difference on 10,000 test
   points `<= 1e-6` relative, the ridges allow about 1e-8), the same training error, the same
   accept/reject decision, and this library's ART must evaluate `f` only at those points (+1 centroid). NumPy
   `lstsq` is the third opinion. Problems: all six, d = 2, 3, 5, degrees 1 and 2, plus an exactly
   representable case.
2. **Thin box.** An exactly affine target `f = 2 + x_0 + x_1 / w` on `[0,1] x [0,w]`,
   `1/w` = 1 ... 1e4, fitted by each code's leaf model on the same points. Both should reproduce
   it to rounding (criterion `<= 1e-6`).

---

### Split search: this library's ART against the reference ART (first results)

Run on the CPU at `a161574` (before the library rename; the side labels were then `art` /
`afinni`), with the reference ART at `854924a` (`hit_and_run` sampler, CPU optimizer), the `conv`
group's matched settings (`--library-soft-iters 200`) and, for this library's ART, also its own
default of 40 soft iterations. Incremental, as agreed: every quick cell first (each problem
family at its loosest tolerances), larger cells only while the quick ones hold. A reference run
was capped at 5 minutes (this library's ART at 10).

Leaves at seed 1 (reference / this library's ART, 200 iterations), with test relative L2 and the
share of test points above the tolerance comparable on both sides unless noted:

| problem | cells | leaves, reference / this library's ART |
|---|---|---|
| Gaussian d = 2-5, degree 1 and 2, tol 0.1-0.005 | 31 | 0.6-1.0x in 25 cells; 1-3 leaves more in 6 cells whose trees have 1-13 leaves |
| quadratic d = 2-5, degree 1, tol 0.1-0.005 | 10 | 0.74-0.91x at d = 2-3; 169 vs 416 at d = 4 and 253 vs 2,163 at d = 5 (tol 0.1) |
| Gaussian mixture d = 2, 3, tol 0.1 | 2 | 26 vs 22 at d = 2 (small tree); 402 vs 569 at d = 3 |
| Rosenbrock d = 2, degree 1, tol 0.1-0.01 | 4 | 0.28-0.33x (e.g. 2,803 vs 10,034 at tol 0.01) |
| Rosenbrock d = 2, degree 2, tol 0.1 and 0.05, seeds 1-3 | 6 | 0.77-0.89x; at tol 0.1, seed 1, this library's ART has 4x the reference's test L2 (6.2e-3 vs 1.5e-3), seeds 2 and 3 do not (3.3e-3 vs 3.4e-3, 4.6e-3 vs 4.9e-3): seed noise |
| Rosenbrock d = 3, degree 2, tol 0.1 | 1 | 376 vs 404 |
| plane wave d = 3, tol 0.1 | 1 | 77 vs 1,557 |

Where the reference finishes, this library's ART reaches the same accuracy with as many or fewer
leaves and oracle calls, and trains 5-50x faster on the CPU. The reference did not finish within
5 minutes at Rosenbrock d = 3-5 (degree 1) and d = 4, 5 (degree 2), plane wave d = 5 and 8, and
Rastrigin d = 3; those cells have no comparison yet (this library's ART: e.g. 2,292 leaves at
Rosenbrock d = 3, 7,255 at d = 4, 3,947 at plane wave d = 5).

This library's default of 40 soft iterations against the 200 matched to the reference: 5-30% more
leaves at d >= 3 (822 vs 627 at Gaussian d = 4, tol 0.01; 9,408 vs 7,255 at Rosenbrock d = 4), 4.3x
more at plane wave d = 5 (17,059 vs 3,947, and slower: 103 vs 72 s), about the same at d = 2.

### Soft iterations: 40 against 200 (this library's ART)

Behind the default `ARTSettings.soft_iters = 200` (the reference ART's cap). The library's side
only, seed 1, on the CPU at `e9879eb`, with the comparison's problems (d = 2-5, converged trees)
and, for d = 10-100, depth-capped trees, where both settings build trees of equal size and the
test error at equal leaves and evaluations compares the planes. Test error is comparable in every
converged cell, and oracle calls follow the leaves.

| regime | leaves, 200 / 40 | build time, 200 / 40 |
|---|---|---|
| trees under about 50 leaves, any problem | 0.9-1.1 | about 1 |
| Gaussian, quadratic, Rosenbrock at d = 2 | 0.91-1.07 | 1.2-2.4 |
| Gaussian d = 3-5, tol 0.05-0.005 | 0.69-1.0 | 1-2 |
| quadratic d = 3-5, tol 0.1-0.02 | 0.76-0.92 | about 2 |
| Rosenbrock d = 3-4, affine leaves, tol 0.1-0.02 | 0.80-0.96 | 2.5-3.3 |
| Rosenbrock d = 3-4, quadratic leaves | 0.85-0.95, and 14-22% lower test error | 3-3.3 |
| plane wave d = 3, tol 0.05-0.002 | 0.23-0.45 | 0.9-1.1 |
| plane wave d = 5, tol 0.1 and 0.05 | 0.22-0.23 (36,077 -> 7,876 at 0.05) | 0.5-0.7 |
| d = 10-100, depth-capped: anisotropic quadratic | equal; test error x 0.40-0.52 at d = 20-100 | 1.5-4 |
| d = 10-100, depth-capped: Gaussian, Rosenbrock, plane wave | equal; test error within 2% (trees far from resolved) | 1.5-4 |

On a laptop GPU, 200 iterations cost 2.1-2.4x the build time for 0.3-65% fewer leaves on
Rosenbrock d = 3, 4, plane wave d = 5 and Gaussian d = 4. Plane wave d = 8, Rosenbrock d = 5 with
quadratic leaves and Rastrigin d = 3 did not finish within 15 minutes on the CPU at either setting.

## Smoke tests run here

All on this machine's CPU (16 cores, no GPU), `SURROGATE_DEVICE=cpu`, outputs in the session scratch
directory. The final smoke runs ran one at a time under
`systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=400% nice -n 10 timeout 600` with
`OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2` (the machine was shared with other jobs); the d=2
full-size runs, the full exact checks and the paper-era comparison ran earlier without the cap.

**Scripts end to end.**

| Run | Outcome |
|---|---|
| `paper/run_paper.sh --smoke` (11 cases: every group except `mpi`, tiny sizes) | all 11 ran, 66 s, peak RSS 0.6 GB; collector built every table and plot (all paper rows `missing`, as they must be at tiny sizes) |
| Part A `mpi` smoke (`mpiexec -n 2`, earlier in the session) | ran and was collected; a later repeat failed only because the host was out of memory (the library's own guard, see Known gaps) |
| `art/run_art.sh --smoke` (exact checks + 18 runs: 6 problems x 2 codes, spherical depth, exact target, this library's ART on 1 and 2 ranks) | all ran, 43 s, peak RSS 0.25 GB; checks: 29 pass, 12 missing (slopes need >= 3 tolerances), 2 FAIL (the thin-box finding below) |
| `--dry-run` of both full matrices | Part A 172 runs, Part B 1070 runs (with `--device gpu`), commands and `SURROGATE_DEVICE`/`SURROGATE_MPI` as intended |

**Part A at full paper size, d = 2 only (CPU, seeds 0 and 1, 3-35 s per run, < 0.9 GB).** All 15
Table 1 rows for Gaussian, Rosenbrock and Wave d=2, with and without Hessian splits:

- `N`, `N_c` and `e_in` agree with the paper within the 25% noise band in every row (e.g. Rosenbrock
  #21: `N_c` 759 vs 746, `N` 2.32e5 vs 2.27e5; Wave #39: 1,167 vs 1,177, 3.01e5 vs 3.03e5), except
  Rosenbrock #22 `e_in`, 27% *below* the paper. A1 (`e_in <= 1.1 tol`), A5 and the slope checks
  pass: slopes -1.28 (paper -1.31), -1.06 (-1.06), -1.39 (-1.45) for Gaussian, Rosenbrock, Wave,
  all steeper than the theoretical -1.
- `p_o` (and hence `e`) is higher than the paper at tight tolerances: Wave #38/#39 2.4% / 7.5% vs
  0.10% / 0.18%, Rosenbrock #21-24 about 1.7-2x. These runs used the script's `--coarse_factor 10`;
  the paper's d=2 rows were measured with 50, which reproduces them (see
  [Where the differences come from](#where-the-differences-come-from), D1). Not a regression.
- `e_nh` (no Hessian) is 2-5x *lower* than the paper's in every row (e.g. Rosenbrock #22: 5.3e-3
  vs 3.46e-2), with 2-8x more leaves than the Hessian runs; the paper-era code gives the same (D3).
  With `--coarse_factor 10`, A2 fails on Rosenbrock #24 and Wave #38-39, where the Hessian run's
  higher `p_o` makes its global `e` exceed the no-Hessian `e`; with 50 the Hessian run's `e` falls
  to its `e_in` (Wave #39 1.06e-3, Rosenbrock #24 1.05e-3) and A2 passes on every d=2 row.

**Part B (CPU).**

- `scripts/art_comparison.py` works with the reference at `854924a` for every problem of the
  matrix (6 benchmarks + spherical piecewise, degrees 1 and 2, 1 and 2 ranks).
- Small converged cases agree closely, e.g. quadratic d=2 tol 0.02: reference 134 leaves, rel L2
  4.24e-3, q99 0.0145, 0.14% of test points above tol; this library's ART 127 leaves, 4.16e-3, 0.0142,
  0.11%. Gaussian d=2 tol 0.02: 15 vs 13 leaves, 4.06e-3 vs 4.01e-3.
- B7: this library's ART gave the identical tree and metrics on 1 and 2 ranks with work stealing.
- Exact root fit (`exact_checks.py`, full set): 37/37 pass. This library's root model matches the
  reference to 1e-10 (quadratic leaves) and 1e-9-3e-8 (affine leaves, the Tikhonov term), with the
  same training error and decision, and evaluates `f` only at the given points plus one centroid;
  NumPy `lstsq` matches the reference to 1e-15. The exactly representable quadratic is accepted
  at the root by both (this library's error 7e-10, the reference's 9e-15).
- **Thin box (with `leaf_fit="tikhonov"`, which was ART's fit when this ran): this library's ART fails
  to reproduce an exactly affine function on thin nodes; the reference reproduces it to 1e-14 at
  every aspect ratio.** With the default `leaf_fit="lstsq"` both pass. Max relative error of
  this library's leaf with the ridge:

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

## Differences from the manuscript

From the CPU re-runs at d = 2 (Gaussian, Rosenbrock, Wave, rows 10-14, 20-24, 35-39; the d >= 5
rows await the GPU). "Now" is `f60eb2b` with the script's settings unless stated.

| # | Difference | Size | Rows | Cause (details below) |
|---|---|---|---|---|
| D1 | `p_o` higher | Wave 2.43% vs 0.10% (#38), 7.50% vs 0.18% (#39), 0.61% vs 0.05% (#37); Rosenbrock 2.5-4.7% vs 1.5-2.4% (#21-24) | 37-39, 21-24 | Wave: setting, the paper used `kappa` = 50 candidates where the script records 10; Rosenbrock: `kappa` and the trust regions of the time (D4) |
| D2 | overall `e` higher | Wave 4.4e-3 vs 2.09e-3 (#38), 4.1e-3 vs 1.05e-3 (#39); Rosenbrock #24 1.7e-3 vs 1.2e-3 | 38, 39, 24 | follows from D1 (points outside every trust region are predicted by a far leaf) |
| D3 | `e_nh` 2-5x lower, so "Hessian splits beat no Hessian" (A2) fails with `kappa` = 10 | e.g. Wave #39 1.60e-3 vs 8.26e-3, Rosenbrock #22 5.3e-3 vs 3.46e-2 | all of 10-14, 20-24, 35-39 | not reproduced by any code state or setting tried; the paper's `e_nh` runs are not in the repository |
| D4 | Rosenbrock and Gaussian d=2 trees not the paper's | Rosenbrock #20: `N` 1.47e5 vs 1.39e5, `e_in` 2.8e-2 vs 3.4e-2 (`N_c` 344 both); #24 `N_c` 8,633 vs 8,446; Gaussian `N_c` 49 / 388 vs 48 / 359 (#10 / #14) | 10-14, 20-24 | confirmed: GPU runs of the paper-era code; Rosenbrock before `e37c315` (shrinkage trust regions), scored with 10 candidates; Gaussian with 10 (d=2) and 50 (d=5) |
| D5 | small tree changes since the paper era | Wave #39 `N_c` 1,177 -> 1,167, `N` 303,333 -> 300,528 | all | deliberate: own k-means replaces scikit-learn (`0edf0e6`); within noise |
| D6 | Rosenbrock test set 50,000 points, not 42,250 (`N_0 / 2`) | `p_o` 1.94% vs 1.99%, `e` unchanged at #24 | 20-24 | change of the launcher (`36d2992`); the first 42,250 points are the paper-era ones |
| D7 | settings that differ between text and code | see the table in "How this repo reproduces it" | Table 2, Figure 9, quadratic, Lorenz | settings and text; decisions 4-6 |

Agree within noise: `N`, `N_c` and `e_in` on every d=2 row except Rosenbrock #22's `e_in` (27% below
the paper, D4), the slopes, A1 and A5.
With `kappa` = 50 the current code also matches the Wave `p_o` and `e`: #37 0.045% / 4.30e-3
(paper 0.05 / 4.22e-3), #38 0.073% / 2.04e-3 (0.10 / 2.09e-3), #39 0.22% / 1.06e-3 (0.18 / 1.05e-3).

## Where the differences come from

**Method.** `studies/convergence/bisect/run_at.sh COMMIT TAG -- <launcher flags>` extracts a commit
with `git archive` (nothing is checked out) and runs *that commit's own launcher* unchanged
through `capture_launcher.py`, which wraps `AdaptiveSurrogate.train/predict` and
`compute_error_metrics` and saves the scored test points, predictions, out-of-region mask, ground
truth, leaf centres and trust matrices. `score.py` then scores every commit with the manuscript's
definitions (`p_o`, `e_in`, `e`) and re-scores `p_o` for other candidate counts and against the
exact union of all ellipsoids (`p_o*`, the best any candidate rule can do with those trust
regions). All runs: CPU, one at a time under the resource cap, 2-15 s each (about 35 runs).

**Test points.** Every era draws them the same way: NumPy seed 0, `X_train` then `X_test`, both
uniform, `N_test = N_0 / 2`; Rosenbrock uses 50,000 from `36d2992` (2026-11-03), whose first
42,250 are the old ones. Checked bitwise: the Wave and Rosenbrock test points and ground truth of
`91ddd97` and `f60eb2b` are identical, so the comparisons below separate model changes from metric
changes. The metric definitions (`compute_error_metrics`) did not change either: each launcher's
own `p_o` and `e` equal `score.py`'s.

**Stand-ins on this machine.** Commits before `d7ec052` import `faiss` and `cupy` unconditionally.
`faiss` is replaced by an exact brute-force search in float32 (`IndexFlatL2`: same as faiss's exact
index up to ties; `IndexHNSWFlat`, used by the paper-era CPU path above 5,000 leaves, becomes exact,
so for Rosenbrock #23-24 the old CPU `p_o` would be slightly higher with real faiss, not lower).
`cupy` is an import-only stub whose `is_available()` is False, so those commits run on NumPy and
scikit-learn, as they would on a machine without a GPU. Neither touches training.

**D1, D2: the candidate count, a setting.** Inference looks for the trust region containing `x`
only among the `kappa` centres nearest to `x` in L2 (every era; `predict(coarse_factor=...)`). AFINNI's
trust regions are long thin ellipsoids along the level sets (the wave's fronts, Rosenbrock's
valley), so the containing ellipsoid's centre is often not among the 10 nearest:

| Run (CPU) | `kappa` 4 | 10 | 20 | 50 | exact union `p_o*` | paper |
|---|---|---|---|---|---|---|
| Wave #39, `91ddd97` | 23.5% | 7.57% | 1.73% | **0.180%** | 0.166% | 0.18% |
| Wave #39, `f60eb2b` | | 7.50% | 1.61% | 0.218% | 0.213% | 0.18% |
| Rosenbrock #24, `91ddd97` | 11.5% | 4.27% | 2.63% | 2.07% | 1.96% | 2.36% |
| Rosenbrock #20 / #22 / #24, `f60eb2b` | | 1.84 / 2.55 / 4.65% | 0.61 / 1.36 / 2.66% | 0.06 / 0.71 / 1.99% | 0.06 / 0.52 / 1.85% | 1.59 / 1.45 / 2.36% |

The paper-era code `91ddd97` (2026-10-08, "Added bash file that runs experiments") with
`--coarse_factor 50` reproduces Wave #38 and #39 to the printed digits: `N` 176,987 / 303,333,
`N_c` 678 / 1,177, `p_o` 0.099% / 0.180%, `e_in` 2.086e-3 / 1.047e-3, `e` 2.089e-3 / 1.053e-3 (paper
1.77e5 / 3.03e5, 678 / 1,177, 0.10 / 0.18, 2.09e-3 / 1.05e-3, 2.09e-3 / 1.05e-3). With 10 (what
`run_paper_experiments.sh` passes for d=2) the same tree gives 7.57%. So the Wave rows were run on
a CPU path (the GPU draws other data, another wave direction and other k-means clusters) with 50
candidates. No commit is involved: `p_o` at `kappa` = 10 is the same at every commit where the tree
is the same (7.572% at `91ddd97`, `935f7ae`, `a071e41`, `15c2721`; 7.503% at `0edf0e6`, `736a4d9`,
`015adfd`, `f60eb2b`). Earlier launchers used other counts: `coarse_factor=DIM**2` (4 at d=2) from
`a4f9302` (2026-09-22) to `3b6d729` (2026-10-04, which added `--coarse_factor`, default 10).
*Classification:* change of settings (the script and the text, `2 d^2` = 8, misstate what the
table used). *Fix:* done in the study: `cases.py` runs d=2 with 50 and keeps the script's 10 as
group `conv_k10`; `run_paper_experiments.sh` (tracked, not edited here) should pass 50 for d=2,
and the manuscript should state the `kappa` it used. The underlying limit stays: even at 50,
`p_o - p_o*` is 0.01-0.2 points at d=2 and grows with d (Gaussian d=5 #15 at `91ddd97`: `p_o*`
0.010%, `kappa` 50 0.62%, 200 0.20%). A membership search that cannot miss (e.g. candidates within
`1 / sqrt(lambda_min(T))` of `x`, which bounds every ellipsoid that can contain it) would make `p_o`
independent of `kappa`; it changes predictions, not training, and would be a deliberate numerical
change to decide and list.

**D4: the Rosenbrock and Gaussian rows came from another code path.** The CPU tree for Rosenbrock
#20 is the same from May to November: `25ad31f` (2026-05-18), `a6844cb` (2026-10-04), `91ddd97` all
give `N_c` 344, `N` 146,812, `e_in` 2.83e-2 (paper: 344, 1.39e5, 3.40e-2), and #24 gives 8,633 leaves
at `91ddd97` (paper 8,446). The CPU Gaussian rows differ too (`N_c` 49 / 388 vs 48 / 359). Rosenbrock
has no random instance, so the only things that differ on a GPU are CuPy's random stream for the
data and cuML's k-means (300 Lloyd iterations; scikit-learn ran `max_iter=2` on the CPU). Their `p_o`
fits the trust regions of the time: until `e37c315` (2026-10-04, "Changed shrinkage to Tikhonov to
regularize the trust regions") each covariance was shrunk 5% towards `trace/d * I`, which fattens
thin ellipsoids. Rescoring the CPU trees of `a6844cb` with 10 candidates gives 1.52 / 1.68 / 2.66%
at #20 / #22 / #24 (paper 1.59 / 1.45 / 2.36), the flat profile of the table, which no single `kappa` gives
with the Tikhonov-regularized regions of `e37c315` onwards (now, at 20: 0.61 / 1.36 / 2.66%). The Rosenbrock rows also already appear, with
their `e_nh`, in the manuscript's older commented-out table, where the Wave rows still lack `e_nh`.
*Classification:* most likely a GPU run of the shrinkage era (before `e37c315`) with 10 candidates;
not a bug and not a later change. The Gaussian d=5 row #15 also differs on the CPU (`91ddd97`:
`N_c` 415 vs 173), as expected if the d=5 rows ran on the GPU (CuPy-drawn rotation). *Confirmed* on a GH200 GPU
(see "Paper-era runs on a GH200 GPU" below): run on the GPU, the paper-era code reproduces every
`N` and `N_c` of these rows exactly (Gaussian 48 / 359 / 173, Rosenbrock 344 / 2,007 / 8,446). The
Rosenbrock `p_o` comes from `a6844cb`'s shrinkage trust regions with 10 candidates (1.588 / 1.451 /
2.364%, the printed digits); the Gaussian `p_o` from 10 candidates at d=2 and 50 at d=5. So Table 1
mixes two trust-region versions, the CPU (Wave) and the GPU, and two candidate counts. *Fix:* none
needed in the code; re-measure Table 1 under one setting with the current code (the study does)
and report that.

**D3: `e_nh`, unexplained.** The paper-era script has no `e_nh` runs. No state tried reproduces the
column: Wave #39 without Hessian gives `e` = 1.62e-3 / 1.64e-3 at `91ddd97` (`kappa` 10 / 50), 1.60e-3
now, 1.23e-3 at `a6844cb` (whose launcher used `K_0` = 1000 for the wave), 2.09e-3 with `Linf-global`
and 4.67e-3 with `L2` acceptance (paper 8.26e-3); Rosenbrock #22 gives 6.5e-3 at `a6844cb` (paper
3.46e-2). At `91ddd97` the split for `M = I` is the principal axis of the cluster covariance, and
the current code's `e_nh` agrees with that commit's within 3%. The paper's `e_nh` exceeds the tolerance at tight tolerances (Wave #39 8.3e-3 > 5e-3,
Rosenbrock #22 3.5e-2 > 2e-2), which a converged `Linf-local` run cannot produce inside its trust
regions, so those runs were probably unconverged or used another acceptance rule. The paper-era GPU path
is excluded: on a GH200 it gives the same no-Hessian accuracy as the CPU (Rosenbrock #22 5.67e-3,
Wave #39 1.58e-3; see below). *Classification:* unknown; not a change of the code between 2026-10-04 and now. *Fix:* re-measure `e_nh` with the current code (group `conv_nh`) and restate the claim
as "fewer leaves for the same `e_in`" (2-8x here; 4.6x for Rosenbrock and 8.3x for Wave on the GPU).

**D5: tree changes since the paper era.** Bisected on Wave #39 (first-parent of
`refinement-engine`; the modularity branch's commits `711c69b`..`736a4d9` are on it, merged into
`main` by PR #5, `1088e53`, 2026-11-06): `N`, `N_c`, `p_o` bit-identical from `91ddd97` through the
merge of `feat/cross-platform-backends` (`935f7ae`, 2026-11-04), `a071e41` and `15c2721`; they change
at `0edf0e6` (2026-11-06, "Integrate custom search and enforce depth-aware refinement"), whose
message states that the initial clustering algorithm changes (own k-means|| + Lloyd instead of
scikit-learn), and stay the same through `736a4d9`, `015adfd` and `f60eb2b`. *Classification:*
deliberate change of a component; within noise. `3fb78af` (problem instances drawn without
reseeding the global RNG) does not change the CPU tree here.

**D6:** `36d2992` (2026-11-03, "Added N_train as a command line argument") set Rosenbrock's test set
to 50,000 points; scored on the paper-era 42,250, #24 gives `p_o` 1.94% and `e` 1.04e-3 instead of
1.99% and 1.05e-3. Change of test-set size, negligible; nothing to fix.

### Paper-era runs on a GH200 GPU

These runs confirm D4 and exclude the GPU as the source of D3. Each paper-era commit ran on one
GH200 (CUDA runtime 12.9) in its own locked environment: `uv sync` of a `git archive` extraction of
`91ddd97` and of `a6844cb`. Both install on aarch64, with cuml-cu12 25.10 and CuPy 13.6.0. Each case
runs that commit's own launcher unchanged; the scripts come from `studies/convergence/bisect/`:

```bash
export BISECT_DIR=<extractions and runs> DEVICE=gpu CAP=0 FAISS_SHIM=
R=studies/convergence/bisect/run_at.sh
P=$BISECT_DIR/91ddd97/.venv/bin/python
for t in 0.1 0.005; do
  PYTHON=$P $R 91ddd97 gpu_r$t -- --experiment rosenbrock --dims 2 --tols $t --error_eval_mode affine_cov_hess --coarse_factor 50
  PYTHON=$P $R 91ddd97 gpu_g$t -- --experiment anisotropic_gaussian --dims 2 --tols $t --error_eval_mode affine_cov_hess --coarse_factor 50
done
PYTHON=$P $R 91ddd97 gpu_g5d0.1 -- --experiment anisotropic_gaussian --dims 5 --tols 0.1 --error_eval_mode affine_cov_hess --coarse_factor 50
PYTHON=$P $R 91ddd97 gpu_w0.005nh -- --experiment wave --dims 2 --tols 0.005 --error_eval_mode affine_cov_no_hess --coarse_factor 10
PYTHON=$P $R 91ddd97 gpu_r0.02nh -- --experiment rosenbrock --dims 2 --tols 0.02 --error_eval_mode affine_cov_no_hess --coarse_factor 10
P=$BISECT_DIR/a6844cb/.venv/bin/python
for t in 0.1 0.02 0.005; do
  PYTHON=$P $R a6844cb gpu_r${t}_shrink --launcher experiments.py -- --experiment rosenbrock --dims 2 --tols $t --error_eval_mode affine_cov_hess
done
PYTHON=$P $R a6844cb gpu_r0.02nh_shrink --launcher experiments.py -- --experiment rosenbrock --dims 2 --tols 0.02 --error_eval_mode affine_cov_no_hess
python studies/convergence/bisect/score.py $BISECT_DIR/runs/gpu_*.npz --kappas 10,50
```

**D4.** `p_o` (%) at the candidate count whose value agrees with the paper:

| Row | Paper: N / N_c / p_o / e_in / e | Commit | N / N_c | p_o | e_in / e |
|---|---|---|---|---|---|
| #10 Gaussian d=2, 0.1 | 8.46e4 / 48 / 0.02 / 1.54e-2 / 1.54e-2 | `91ddd97` | 84,554 / 48 | 0.019 (10 and 50) | 1.542e-2 / 1.542e-2 |
| #14 Gaussian d=2, 0.005 | 1.4e5 / 359 / 0.21 / 1.02e-3 / 1.06e-3 | `91ddd97` | 140,088 / 359 | **0.206 (10)**; 0.083 (50) | 1.022e-3 / 1.029e-3 (50) |
| #15 Gaussian d=5, 0.1 | 2.28e5 / 173 / 0.05 / 1.86e-3 / 1.89e-3 | `91ddd97` | 227,533 / 173 | **0.052 (50)**; 0.85 (10) | 1.861e-3 / 1.891e-3 |
| #20 Rosenbrock d=2, 0.1 | 1.39e5 / 344 / 1.59 / 3.4e-2 / 3.4e-2 | `a6844cb` | 139,317 / 344 | **1.588 (10)** | 3.401e-2 / 3.406e-2 (4) |
| #22 Rosenbrock d=2, 0.02 | 5.23e5 / 2,007 / 1.45 / 5.1e-3 / 5.12e-3 | `a6844cb` | 523,036 / 2,007 | **1.451 (10)** | 5.113e-3 / 5.306e-3 (4) |
| #24 Rosenbrock d=2, 0.005 | 2.15e6 / 8,446 / 2.36 / 1.18e-3 / 1.2e-3 | `a6844cb` | 2,145,572 / 8,446 | **2.364 (10)** | 1.315e-3 / 2.633e-3 (4) |

`91ddd97`, with Tikhonov trust regions, builds the same Rosenbrock trees, but its `p_o` at 10
candidates is 1.97% / 3.96% at #20 / #24. `a6844cb`'s errors are at its launcher's 4 candidates; at
#24 the paper's are closer to `91ddd97`'s at 50 (1.129e-3 / 1.13e-3), so that row's errors may come
from a run that predicted with more candidates.

**D3.** Without Hessian splits:

| Row | Paper `e_nh` | Commit and candidates | N_c | `e` (no Hessian) | `p_o` | Hessian run's N_c |
|---|---|---|---|---|---|---|
| #22 Rosenbrock d=2, 0.02 | 3.46e-2 | `91ddd97`, 10 | 9,310 | 5.67e-3 | 1.77% | 2,007 |
| #22 Rosenbrock d=2, 0.02 | 3.46e-2 | `a6844cb`, 4 | 9,310 | 5.67e-3 | 2.07% | 2,007 |
| #39 Wave d=2, 0.005 | 8.26e-3 | `91ddd97`, 10 | 9,812 | 1.58e-3 | 3.43% | 1,177 |

The GPU path is as accurate as the CPU path (Rosenbrock #22 6.5e-3 on the CPU at `a6844cb`, Wave
#39 1.62e-3 at `91ddd97`), 5-6x below the paper's column, with 4.6x and 8.3x the Hessian runs'
leaves.

## Known gaps and risks

- **The library's host-memory guard is system-wide.** Before each level it checks the machine's
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
  NumPy instances. The paper mixed platforms: the Wave d=2 rows reproduce exactly on the CPU
  (D1), the Rosenbrock and Gaussian rows do not (D4). Compare GPU re-runs to the paper with the
  noise rule and A1-A6, not bits.
- **`p_o` depends on the candidate count** at every d (D1): compare it with the paper only at the
  paper's `kappa`, and read `p_o*` (`bisect/score.py`) for the coverage of the trust regions
  themselves.
- **Paper-era settings unknown**: which `coarse_factor` and caps Table 2 used, and the Lorenz
  horizon in the text. The study uses the script's and caption's values; decide if they fail.
- **Lorenz-96 GPU** needs JAX's CUDA build (in the `gpu` extra on Linux); otherwise it runs on the
  CPU with SciPy, a slightly different (more accurate) flow map.
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

1. **This library's ART leaf regularization (Part B finding): decided, ART fits as the reference does.**
   `ARTSettings.leaf_fit="lstsq"` (the default) drops the ridge in every ART fit, leaves and
   split search, and solves by Cholesky where the Gram's 1-norm reciprocal condition number is
   at least 1e-10, else by QR on the node's rows (`lstsq_auto`). AFINNI keeps Eq. 3's ridge and is
   bit-identical; `leaf_fit="tikhonov"` restores the ridge for ART. With it the thin-box check
   passes at every aspect ratio. The split search stays this library's (the bins screen, then the
   soft polish, 200 iterations as the reference allows): where the reference finishes, it reaches
   the reference's accuracy with as many or fewer leaves ("Split search" and "Soft iterations"
   above). The reference's start, gate width, minimum side and retries are not ported.
2. **The `e_nh` column (D3).** The no-Hessian baseline is 2-5x more accurate than the paper's, in
   the paper-era code as now; no code state or setting tried reproduces the column, and the
   paper-era script has no `e_nh` runs, and the paper-era GPU path gives today's values too.
   Re-measure `e_nh` and restate "Hessian splits beat no Hessian" as "fewer leaves for the same `e_in`" (2-8x
   fewer here). With `kappa` = 50, A2 (`e <= 1.25 e_nh`) passes on every d=2 row, though on
   Gaussian #12-14 `e_nh` is still 5-16% below the Hessian run's `e`.
3. **`kappa` (D1, D4).** The paper's Table 1 used different candidate counts row by row: 50 for
   Wave d=2 (on the CPU), 10 for Gaussian and Rosenbrock d=2, 50 at d=5 and 200 at d=10 (on the
   GPU). Table 1 is now re-measured under one rule, `kappa = max(50, 2 d^2)` (50 / 50 / 200 at
   d = 2 / 5 / 10), in `run_paper_experiments.sh` and in the study; the text should state that
   rule and report the re-measured table. Table 2 keeps 10: at d = 80 the rule would gather about 120 MB of
   trust matrices per query. Decide also whether inference
   should find the containing trust region exactly (a search bounded by `1 / sqrt(lambda_min(T))`
   instead of the `kappa` nearest centres), which makes `p_o` independent of `kappa`; a deliberate
   change of predictions (training unchanged).
4. **Table 2 settings.** The caption's caps (`K_0 <= 1000`, `M <= 500`) versus the paper-era
   code's `cap_initial_augmentation_budget` (`K_0` down to 16, `M` up to 31,250), and
   `coarse_factor` 10 (script) versus `2 d^2` (text) for d up to 80. The study uses the caption
   and the script; switch if Table 2 does not reproduce.
5. **Lorenz-96 on the GPU: decided.** The `gpu` extra installs `jax[cuda12]` on Linux, so the
   Lorenz rows run the paper's JAX flow map (rtol 1e-3) on the GPU.
6. **Paper text vs code** (fix the text or the code): quadratic spectrum `i^-3` (text) vs `5^-i`
   (code); "rtol = atol in every experiment" vs Lorenz's absolute `tol * max|f|`; Figure 9's
   isotropic Gaussian, wave domain `[-4, 4]` and Rosenbrock test distribution; the
   typos and the Table 1b / Table 4 Lorenz mismatch listed above.
7. **Pass thresholds.** The 25% noise floor, `1.1 tol` for A1, and the B-check bands are my
   proposals; adjust them before reading the results as final.
