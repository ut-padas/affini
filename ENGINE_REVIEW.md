# Refinement engine and high-dimensional runs: findings, software changes, algorithmic changes

A review of `src/engine/` (the driver, the per-rank loop and work stealing), of the contracts it
shares with `AFINNIPlan` and `ARTPlan`, and of what stops AFINNI and ART from reaching d ≈ 100,
with a scaling study to measure those limits on one GPU (`studies/high_dim/`).

**Scope of the next step.** For now the intention is to implement only **Section 3 (software
changes)**: changes to how the code runs that leave the algorithm's mathematics unchanged.
**Section 4 (algorithmic changes)** changes what the algorithms compute; every item there needs
a study (accuracy, convergence order, new regression references) before it is adopted, not just
an implementation.

**State of the code.** Every file:line reference is to `refinement-engine` at `f60eb2b`. The
references were checked against that commit; paths are relative to the repository root.

**What the evidence rests on.** Each result is labelled by its source:

- **read**: the code, with file:line references;
- **measured**: CPU runs on this machine (Intel Core Ultra 7 255H, NumPy, bundled MPICH, one
  thread per rank). The engine timings of Section 1.2 are single runs, not the interleaved A/B
  rounds AGENTS.md requires for a performance change; their effects (1.5x to 3x) are well above
  run-to-run noise, but treat them as indicative. They were taken on an earlier commit; the code
  paths they exercise are unchanged at `f60eb2b`, but the timings were not repeated;
- **predicted**: formulas evaluated by instantiating the plans on the CPU
  (`studies/high_dim/predict_limits.py`);
- **literature**: a survey checked against primary records where marked.

This machine has no CUDA device. **No GPU number in this document was measured here.** GPU
statements rest on the code and on the Nsight measurements recorded in
`docs/AFINNI_ART_DESIGN.md` (section 42).

AFINNI's trees depend on batch boundaries (Section 1.3, F10), so AFINNI runs with different
batch sizes build statistically equivalent but not identical trees (leaf counts within 0.7% in
Section 1.2).

---

## Summary

- **Keep the architecture.** The plan/engine boundary, contiguous batches of whole nodes with
  segmented reductions, deterministic per-segment GPU sums, ART's node-keyed random streams,
  self-contained migratable work items and the one-reply-per-request steal protocol are right.
- **The scheduler is where the time goes on CPU and on several ranks.** `batch_size` is a
  10M-point memory cap that also serves as cache block, steal granularity and (for AFINNI) part
  of the random stream; one batch per level made a CPU run 2x slower and made stealing almost
  inert with the defaults (Section 1.2).
- **At d ≈ 100 on one GPU, the first failures are implementation limits**, each fixable without
  changing results: leaf buffers that reserve `max_clusters_est x d^2` doubles at startup (80 GB at
  d = 100), memory charges that undercount quadratic leaves up to 4.9x and overcount affine ART
  nodes 10x, a kNN prediction gather of 4 GB per chunk, and host RAM for breadth-first queues.
- **Behind them are method limits** that only Section 4 can move: a Hessian estimate that is
  singular for d > 33 with the defaults, hit-and-run chains that barely move in high d, a fit
  ridge whose bias grows with d and with node aspect ratio (an exactly affine function is fitted
  with error 0.14 at aspect ratio 1e4), acceptance tested only at AFINNI's own points, and
  leaf counts that grow exponentially in the number of curved directions.
- **Most scheduler changes are bit-identical for ART but not for AFINNI**, whose random streams
  and batch grid depend on the schedule. Node-keyed AFINNI streams (A1) remove that dependence;
  until then, any software change that moves AFINNI's batch boundaries or rank assignment is a
  deliberate numerical change (Section 6, decision 1).

---

## 1. Findings (current code)

### 1.1 Verdict

**Keep.** These are the hard parts, and they are done well:

- The plan/engine boundary (`src/core/plan.py`), which lets one loop run both algorithms.
- Contiguous batches of whole nodes with segmented reductions and batched small solves, so no
  Python loop runs over nodes or points.
- Deterministic per-segment GPU sums, and ART's random streams keyed by `(root, locational_code)`
  (`src/art/plan.py:222`). ART's tree is therefore independent of batch size, rank count and steal
  schedule: the 4-rank ART run with stealing of Section 1.2 built the same 8,182 leaves from the
  same 2,470,963 evaluations as the run where rank 0 did everything.
- Self-contained migratable work items: a stolen node needs nothing from its victim.
- The one-reply-per-request steal protocol and the explicit MPI failure policy.

**Change.** Four properties of the scheduler cost real time or memory:

1. **A batch is usually a whole level.** `batch_size` is a memory cap, not a work quantum. It
   makes the CPU path about 2x slower and sets the latency of every steal reply to one level.
2. **Work stealing rarely fires with the defaults.** The thresholds are absolute point counts,
   the victim answers only between batches, and each thief asks one victim and blocks.
3. **Breadth-first order holds a whole level in host RAM**, the worst-case memory order for a
   tree, and at high d each queued node is megabytes.
4. **Rank 0 works alone at both ends**: `f(X0)`, the initial clustering, a scatter by cluster
   count, and a pickled gather.

One further property blocks most fixes for AFINNI: its results are a function of the schedule
(batch index, level, rank and free VRAM enter the random streams and the batch boundaries).

### 1.2 Measurements (CPU, single runs)

All runs: Rosenbrock, uniform points in `[-1, 1]^d`, `OMP_NUM_THREADS=1`. The timing script is
not in the repository.

**Where one rank's time goes (AFINNI, serial).** `d = 4`, `N0 = 200,000`, `atol = rtol = 3e-3`,
`K0 = 64`, `min_pts = 50`, `AFFINE_COV_HESS`: 561,680 leaves, 41.8M evaluations, 31 levels.

| Phase | Share of 159 s |
|---|---|
| Split rejected (`time_bad_split`) | 72% |
| Fit and error | 11% |
| Augment | 7% |
| Trust regions and finalize | 6% |
| Engine's own work: slicing, copies to the queue, level transition | about 2% |
| Oracle | 1.7% |

On the CPU the engine's bookkeeping is cheap; what it controls is how the plan's work is cut up
and where it runs.

**Batch size on one rank** (same case):

| `batch_size` | Wall | Split | Fit and error | Leaves |
|---|---|---|---|---|
| 10,000,000 (default; every level is one batch) | 159.5 s | 115.1 s | 17.3 s | 561,680 |
| 200,000 | 95.0 s | 63.2 s | 11.8 s | 557,857 |
| 50,000 | 80.1 s, 80.8 s | 52.4 s, 53.4 s | 9.9 s, 9.7 s | 560,599 |

A 2.0x speedup from one setting. The middle levels hold about 3M points each, so every temporary
is tens to hundreds of megabytes, far beyond the CPU's 24 MiB L3; with 50,000-point batches the
working set stays in cache. Which temporaries dominate was not profiled, so the cache explanation
is a hypothesis that fits the data, not a measured cause. Leaf counts differ because AFINNI's
trees depend on batch boundaries (F10).

**Work stealing, 4 ranks** (same AFINNI case; "idle" summed over ranks):

| Configuration | Wall | Idle | Steal requests / grants |
|---|---|---|---|
| No stealing | 39.3 s | 36.5 s | none |
| Stealing, defaults | 40.4 s | 30.1 s | 3 / 1 |
| Stealing, `batch_size = 200,000`, threshold 100,000 points | 26.1 s | 6.8 s | 13 / 9 |

A skewed workload (`d = 4`, `N0 = 100,000`, a function steep near one corner only):

| Configuration | Wall | Idle per rank |
|---|---|---|
| No stealing | 18.1 s | 3.6, 8.6, 0.0, 16.7 s |
| Stealing, defaults | 18.2 s | 2.9, 7.9, 0.2, 13.0 s |
| Stealing, small batches and threshold | 12.2 s | 0.7, 4.1, 0.3, 4.3 s |

With the defaults stealing does nothing on these runs: three requests in 40 seconds, while two
idle ranks waited 13 to 16 s each.

ART (`d = 5`, `N0 = 20,000`, `tol = 1e-2`, 13 levels, one box root):

| Configuration | Wall | Idle | Notes |
|---|---|---|---|
| No stealing | 76.0 s | 225.8 s | rank 0 holds the only root and does all the work |
| Stealing | 25.4 s | 25.7 s | 59 requests, 6 grants, 46 denials `MID_LEVEL_NO_CLUSTERS` |

Stealing is essential for ART (3.0x here), but a quarter of the rank-seconds are still idle;
perfect balance would give about 19 s. The 46 denials share one cause: the whole level is in the
current batch, so there is no unprocessed tail to give. In every stealing run, 6 to 8 s of the
idle time is `time_state_update`, spent by a thief spinning on `Iprobe` while it waits for its
reply (F6).

**Host memory.** A `d = 6`, `N0 = 100,000`, `tol = 1e-2` AFINNI run passed 7.4 GB of RSS before it
was stopped.

### 1.3 Findings in the code

"Status" is the verdict of a check against `f60eb2b`: **accurate** (the code does this),
**partly** (true, but its weight or fix needs qualifying), **not re-verified** (a measurement not
repeated).

| # | Finding | Evidence | Status | At high d |
|---|---|---|---|---|
| F1 | `batch_size` is a 10M-point memory cap, lowered only by the device cap (65% of free VRAM at the plan's `bytes_per_point`, clipped to `[batch_floor, 20M]`), so a level smaller than it is one batch | `src/config.py:200`; `src/engine/driver.py:315-347`; `src/engine/refinement.py:179,203`; `src/afinni/plan.py:123`, `src/art/plan.py:164-171` | accurate | On a GPU at d ≈ 100 the cap already gives small batches (ART affine about 211k points or 42 nodes per batch on 24 GB, AFINNI about 2.1M points), so "one batch per level" mostly disappears there |
| F2 | Steal requests are answered between batches and at level ends; ART's `progress()` answers from inside the soft search but can give away only nodes after the current batch, so with one batch per level it has nothing to give | `refinement.py:198,205-209,231-232`; `src/art/plan.py:424-426,574-575`; `src/engine/work_stealing.py:228` | accurate | multi-rank only |
| F3 | Absolute steal thresholds: `MIN_LOAD_THRESHOLD = 1,000,000` and `STEAL_SIZE = 500,000` points for AFINNI; ART uses 4 and 2 nodes' worth of points. Points are not cost: the split stage is 72% of AFINNI's time, and an ART node costs about `n p^2 + p^3` plus sampling | `work_stealing.py:38-39`; `art/plan.py:94-97` | accurate | ART's cost per node grows as `n p^2 ≈ m p^3` |
| F4 | Every idle rank picks the same victim (`argmax` of the gossiped loads) and blocks on its one request | `work_stealing.py:305,314-353` | accurate | degrades with rank count; not measured beyond 4 ranks |
| F5 | Load gossip is all-to-all: `P - 1` blocking 8-byte sends per update, relying on the eager limit, `O(P^2)` messages per level across the job | `work_stealing.py:190-197` | accurate | harmless at 4 to 64 ranks, will not scale to thousands |
| F6 | A thief waiting for its reply spins on `Iprobe` without sleeping (6 to 8 s of idle per stealing run in 1.2); the 50 ms sleep applies only after a denial or below the threshold | `work_stealing.py:320-353` (sleep at `:307,352`) | accurate (code); not re-verified (time) | cheap to fix (S8) |
| F7 | Premature retirement: a rank retires when its cached loads sum to zero, and never steals again once inside `_retire`. Loads from different senders are not ordered at a third rank, so a rank can see a victim's zero (published at its level end) before the thief's nonzero publish after `recv_clusters` | `work_stealing.py:298-303,335`; `refinement.py:231`; the `_retire` docstring states the limit (`:355-362`) | partly: the window exists by reading; not reproduced | one lost worker, never a wrong result (the barrier in `_retire` waits for everyone); termination by message counting (Mattern 1987) closes it |
| F8 | Breadth-first order holds a whole level in host RAM: the queue peaks at the widest level (`docs/AFINNI_ART_DESIGN.md:451` records d = 10 Rosenbrock growing from 0.69M to 184M points at an intermediate level). Breadth-first holds `O(width)` nodes, depth-first `O(depth)` | `refinement.py:171-177,220-232` | accurate | **worse at high d**: a queued ART child carries about n/2 inherited points, about 2.1 MB at d = 100 (2,525 x 101 doubles) against 33 KB of constraints, so 64 GB of host RAM holds about 30k queued nodes per level |
| F9 | The tail of a tree is many nearly empty levels (the run in 1.2 spends its last ten levels on fewer than 30,000 points each, 142 points at level 30); on a GPU each pays the full per-level launch and transfer cost (about 25 to 30 µs per launch in the design doc's Nsight measurements) | `refinement.py:171-232` | accurate (code); GPU cost not measured here | |
| F10 | Rank 0 works alone: it evaluates `f(X0)` while every other rank waits, clusters, scatters runs of equal cluster *count* (not work; ART with one root gives rank 0 everything), and receives every rank's result as a pickled `comm.send`, then concatenates and builds the routing table | `driver.py:150-153` and `:248-259` (`f(X0)`, clustering), `:300` (equal counts), `:193-199` (pickle); `art/plan.py:176-187` (routing) | accurate | the largest serial term for an expensive oracle under MPI; no effect on one GPU |
| F11 | AFINNI's results depend on the schedule. The augmentation stream is `seed + level + batch_index + rank * 1000`: streams coincide whenever `level + batch_index` matches (level 3 batch 0 and level 2 batch 1), and `seed + 1` reuses the streams of `seed` one level later. Batch boundaries come from `batch_size`, which the device cap derives from *free* memory, so a tree depends on what else uses the card. A stolen cluster runs under the thief's rank and level counters | `src/afinni/plan.py:249`; `afinni/plan.py:123-135` (batch grid); `driver.py:327-328,340-347` | accurate | also a reproducibility issue on a shared GPU; the high-d study must run on an otherwise idle card. This is why AFINNI's regression check pins batch size, rank count and stealing, and why multi-rank AFINNI results cannot be checked against serial ones |
| F12 | Host and device traffic: every batch is copied from pageable host memory with `xp.asarray`, children and leaves return through `to_host`; nothing is pinned and nothing overlaps compute. `free_all_blocks()` runs after every batch and twice more inside an AFINNI batch, returning the pool to the driver and forcing fresh `cudaMalloc` calls. The queue always lives on the host | `refinement.py:271-273,284`; `afinni/plan.py:171,176,267`; `src/core/clusters.py:315` | accurate (code); not measured | smaller at high d: ART's Grams cost `n p^2` flops per `n (d+1)` doubles moved, so arithmetic intensity grows with d |
| F13 | The 1.2 measurements: 72% in the split stage, 2.0x from `batch_size = 50,000` on CPU, stealing ineffective with defaults, 3.0x for ART with stealing | 1.2 | not re-verified | plausible; the AFINNI batch-size effect is reachable by a user setting today, but it changes AFINNI's trees statistically (F11) |

### 1.4 Further findings, most of them at high d

Each is ranked in Section 2 by the d at which it bites.

- **M1. The leaf buffers are reserved up front.** `ColumnBuffer` allocates
  `max_clusters_est // size` rows per rank at construction (default 1,000,000 in total), including
  the `(d, d)` `trust_matrix` column: 8 d² MB, 80 GB at d = 100 on one rank
  (`src/config.py:202`; `refinement.py:82,88-89`; `src/core/clusters.py:298-305`). The buffer
  already grows by doubling when full (`clusters.py:327-333`), so the large initial capacity
  buys nothing but avoided copies.
  - **Measured:** `np.empty((1_000_000, 100, 100))` raises `MemoryError` on this machine (30 GB
    RAM + 30 GB swap, heuristic overcommit); d = 80 passes. AFINNI therefore fails at startup
    somewhere between d ≈ 85 and d ≈ 160, depending on RAM and swap.
  - ART is hit the same way: `ARTSettings.store_trust_matrix=True` is the default
    (`config.py:127`), although only the kNN predictor reads the column.
- **M2. The quadratic-leaf memory charge misses the largest temporary.** `segment_cross`
  allocates one partial result per 256-row chunk and entry, `(N/256 + K) x p(p+1)/2` doubles for
  a symmetric Gram, about `n p^2 / 512` doubles per node (`src/numerics/segments.py:208-209`).
  ART's `bytes_per_point` (`art/plan.py:113-133`) does not charge it. For quadratic leaves this
  term dominates: predicted per-node peak over charge = 1.7 at d = 60, 3.0 at d = 80, 4.9 at
  d = 100. The batch cap therefore admits batches that cannot fit. For affine ART nodes the
  charge errs the other way, about 10x (M4).
- **M3. AFINNI's charge omits its d x d arrays per cluster** (the covariance, the level-set
  scatter, the eigenvectors, M, CovZ and the trust matrix: about 8 per cluster);
  `afinni/plan.py:99-118` charges only per-point terms. The omission is about +14% with the
  study's `min_pts_per_cluster = 3(d+1)` and about +43% at d = 100 with the default of 100: not
  yet an out-of-memory at 65% of free memory, but a thin margin.
- **M4. ART charges carried constraints per point with the maximum depth.** `bytes_per_point`
  charges `2 h (d+1)` doubles per point with `h = max_levels + 1` (`art/plan.py:123,133`), while a
  node carries `h (d+1)` doubles once, with `h` its actual depth (`art/plan.py:160-162`). That
  overestimates an affine node about 10x at high d. The docstring (`art/plan.py:115-121`) states
  why it is kept: charging per node admits batches that nearly fill the card, and a driver that
  pages a full card to host memory (as under WSL) builds the same tree several times slower.
- **M5. On the CPU the batch cap assumes 32 GiB.** Without CuPy the cap uses a fixed
  `32 * 1024**3` bytes, not the machine's RAM (`driver.py:330-331`). At low d this never binds
  (AFINNI's 480 bytes per point at d = 4 give a cap above 20M); at high d it sets CPU batch sizes,
  and for AFINNI with them the trees, independently of the host.
- **M6. The host-memory guard is system-wide.** `check_host_memory` refuses a queue append when
  the machine's available memory minus a reserve (10% of RAM, at least 2 GiB) is too small
  (`clusters.py:143-160,233`), so a run on a machine shared with other jobs can stop with
  `MemoryError` at tiny sizes (seen twice in the convergence study). The guard covers queue
  growth only, not the `ColumnBuffer` reservation of M1.
- **M7. Hit-and-run barely moves in high d** (measured, Section 2.2): at d = 100, 20 steps leave
  the endpoint correlated 0.98 with its start. The drift check (`src/art/sampling.py:63-82,301`)
  compares the halfway and final means, which do not move for a chain started at stationarity,
  so it cannot see this. Coordinate directions exist only as a recovery walk for chains that did
  not move (`art/sampling.py:110-121`; `src/kernels/hit_and_run.cu:83`), and each step recomputes
  all `h` slacks at `O(h d)`.
- **M8. The Hessian level set has no floor in d.** `CovarianceHessianSplit.metric` takes
  `N_cov = min_pts_per_cluster // 3` points (`src/components/splitting.py:255`), 33 with the
  default, so its scatter is singular for d > 33, and the eigenvalue floor then gives the
  unexplored directions the largest split weight (noted in `TODO.md`). ART takes its lead
  direction from the same metric (`art/plan.py:413-414,429-432`).
- **M9. The fit ridge biases thin and high-dimensional nodes** (ART: resolved, it fits without a
  ridge as the reference does, `ARTSettings.leaf_fit="lstsq"`; AFINNI keeps it). Affine fits add
  `max(1e-8 tr(Cov), 1e-12)` to the covariance (`src/numerics/linalg.py:121-139`,
  `src/kernels/small_linalg.cu:120-131`; the Tikhonov term of the AFINNI manuscript's Eq. 3).
  Quadratic fits add `1e-10 tr(G)/p` in radius-scaled features
  (`src/components/local_models.py:343-346`). The relative slope shrinkage along direction i is
  about `1e-8 Σ_j σ_j² / σ_i²`: about `1e-8 d` for a round node, growing as `A²` with aspect ratio
  `A` (as `A⁴` for the quadratic `z_s²` features).
  - **Measured (convergence study, thin box, exactly affine f):** max relative error of AFINNI's
    ART leaf 1.2e-8 at aspect ratio 1, 2.9e-7 at 10, 2.8e-5 at 100, 2.8e-3 at 1e3, 0.14 at 1e4
    (quadratic leaf: 3.5e-10 to 1.7e-3); the reference ART, which fits without a ridge,
    reproduces the function to about 1e-14 at every aspect ratio. Such nodes are rejected where
    the reference accepts them; at tolerances ≥ 1e-3 this matters from aspect ratios of about 300
    (affine) or 3,000 (quadratic). Deep ART nodes cut repeatedly along one direction (plane
    waves, Rosenbrock's valley) are exactly the elongated case.
  - **Measured (d = 100, CPU, exactly affine f):** the ART root's error is 3.6e-6 x std f, and
    AFINNI's leaves reach 4.5e-6, so a tolerance of 1e-6 x std makes ART split an exactly affine
    function (that run hit its time limit at level 6).
- **M10. AFINNI's kNN prediction gathers `k_coarse` trust matrices per query**:
  `5000 queries x 10 x d²` doubles, 4 GB per default chunk at d = 100
  (`src/surrogate/model.py:481,590,612-618`). Workaround: `predict(batch_size=...)`, which the
  study scripts set.
- **M11. Everything runs in fp64.** Consumer and laptop GPUs (the WSL case) run fp64 at 1/32 to
  1/64 of their fp32 rate, so the time per node differs by 10 to 50x between an RTX card and an
  H100 or GH200. Results must record the GPU model (the study scripts do).
- **M12. Several kernels lose parallelism above some d** (speed only):
  - `batched_inplace_cholesky_solve`, the affine fit's solve, runs one thread per cluster: d³/6
    serial multiply-adds per thread at d = 100, with strided access (`small_linalg.cu:135`).
  - `spd_solve_factored` runs one thread per system; the factor is cooperative from p ≥ 32
    (`spd_factor_coop`, `small_linalg.cu:66,98`; `src/numerics/linalg.py:23,49-59`), but the
    triangular solves are not, which is slow for quadratic p.
  - `cluster_stats.cu:70` caches `x - mu` in registers only for d ≤ 128.
  - `hit_and_run.cu:8` sets `MAXD 256`; above d = 256 on CuPy the walk falls back to chunked
    per-chain constraint gathers (`art/sampling.py:179-191`).
- **M13. AFINNI certifies only its own points.** Its augmentation takes convex combinations of
  two points of the same cluster (`src/components/sampling.py:82-90`), which stay inside the
  cluster's hull, and acceptance tests only the cluster's points. In high d the hull is a
  vanishing part of the cell a query lands in. ART's points are uniform in the node, it can check
  fresh `validation_samples` (default 0, `config.py:102`), and `AFINNISurrogate.audit`
  (`model.py:718`) gives a global claim for ART models only.
  - **Measured at d = 8 (smoke run):** AFINNI on Rosenbrock at tol 0.3 converged, but only 54% of
    uniform test points were within tolerance (ART: 99.7%).
- **M14. ART thieves hold received work until it is processed.** With `lease_received`, a rank
  never re-offers clusters it received until a level boundary (`work_stealing.py:111,221,228,257`).
  This stops work bouncing between idle ranks, but a thief that received a large grant cannot
  pass part of it on within the level.

---

## 2. What stops d ≈ 100

### 2.1 Blockers, ranked by the d at which they bite

"Impl" is fixable engineering (Section 3). "Method" is approximation theory or sampling, which
only a change of the algorithm moves (Section 4).

| d where it bites | Blocker | Kind | Numbers | Fix |
|---|---|---|---|---|
| > 33 | **B1.** The level set of `N_cov = min_pts // 3` points is singular, so the Hessian-informed split direction is noise (M8). Even with `N_cov ≈ d + 1` points the scatter is a poor estimate | impl (default) and method (estimator) | default `min_pts = 100` gives `N_cov = 33` | short term: `min_pts_per_cluster ≥ 3(d+1)` (the study does this); A2, A3 |
| ≥ 20, worst ≥ 50 | **B2.** Hit-and-run chains of 20 steps (80 at most) do not decorrelate (M7). The endpoints stay uniform because their starts are, but they are near-duplicates of inherited points: about half of each node's oracle calls add little information, and the Linf acceptance test sees fewer independent points than n | method (mixing ∝ d²) and impl (fixed τ) | start/end correlation 0.78 (d = 20), 0.94 (d = 50), 0.98 (d = 100) at τ = 20; independence needs τ ≈ d²/2 | A5, A6 |
| 40-80 | **B3.** Memory of ART quadratic leaves: n = 50 p points with p = (d+1)(d+2)/2, Grams of p², and the uncharged `segment_cross` partials (M2) | impl (charge, partials) and method (p ∝ d²) | predicted per-node peak 1.7 GB at d = 40, 13.8 GB at d = 60, 67 GB at d = 80, 237 GB at d = 100; fits 24 GB up to d ≈ 60, 80 GB up to d ≈ 80; 43k (d = 40) to 258k (d = 100) evaluations per node | S2 (charge or bound the partials); A6 (m = 4 fits 24 GB up to d ≈ 100), A7. The method limit stays: p² grows as d⁴/4 |
| 40-60 | **B4.** Compute of quadratic leaves: the soft search forms two weighted p x p Grams per iteration, 40 iterations | method (`n p^2 ∝ m d^6 / 8`) | predicted 3 s per node at d = 40 and 32 s at d = 60 at an effective 1 TFLOP/s fp64; on a consumer card divide the rate by 10 to 50 (M11) | the setting `split_search="bins"` for quadratic leaves; A7 |
| ≈ 85-160 | **B5.** The leaf buffers reserve `8 d² x 1M` bytes at startup (M1) | impl | 80 GB at d = 100; measured `MemoryError` at d = 100 on 30 GB RAM + 30 GB swap | S1 (the study sets `max_clusters_est = 20,000` and, for ART, `store_trust_matrix=False`) |
| ≥ 70 at small atol; any d for elongated nodes | **B6.** The ridge bias (M9) | impl, but fixing it changes results | 3.6e-6 x std at d = 100 for an exactly affine f; 0.14 at aspect ratio 1e4 | A4 |
| 100 (prediction) | **B7.** kNN gather of 4 GB per chunk (M10) | impl | `8 x 5000 x 10 x d²` bytes | S3 |
| > 128 / > 256 | **B8.** Register caches and kernel fallbacks (M12) | impl (speed) | – | S6 |
| all d, from about 20 | **B9.** Leaves needed for generic curvature (Section 4.11) | method | leaves grow like `∏_i max(1, sqrt(λ_i h² / ε))`, exponential in the number of curved directions | only structure helps: A3, A7 |
| all d | **B10.** Evaluations per leaf: n = m(d+1) for ART (5,050 at d = 100, m = 50); about 2 `min_pts` per rejected AFINNI cluster | method / settings | 10⁴ leaves ≈ 2 x 10⁴ nodes ≈ 10⁸ evaluations at d = 100 | A6; S11 for expensive oracles; ART already reuses inherited points |
| all d | **B11.** Host RAM for breadth-first queues (F8) | impl | about 2 MB per queued ART child at d = 100 | S4 |

### 2.2 Hit-and-run mixing (measured, CPU, the library's own sampler)

Setup: `studies/high_dim/hit_and_run_mixing.py`; 8 nodes of the unit cube, each cut by 4 random
planes; 400 exact uniform starts per node. **ρ** is the correlation of start and end along random
directions (1: the point did not move, 0: independent); **disp** is the mean squared displacement
over its value for two independent points. The drift check flagged **0%** of nodes in every case.

| d | τ = 20 (default) | τ = 80 (`max_walk_steps`) | τ = 320 | τ = 2000 |
|---|---|---|---|---|
| 10 | ρ 0.57, disp 0.42 | 0.15, 0.85 | 0.01, 1.00 | 0.00, 1.01 |
| 20 | 0.78, 0.21 | 0.42, 0.58 | 0.04, 0.95 | 0.00, 1.00 |
| 50 | 0.94, 0.06 | 0.81, 0.19 | 0.45, 0.55 | 0.02, 0.99 |
| 100 | 0.98, 0.02 | 0.94, 0.06 | 0.79, 0.21 | 0.26, 0.74 |
| 200 | 1.00, 0.00 | 0.98, 0.02 | 0.94, 0.06 | 0.69, 0.32 |

- ρ ≈ exp(−c τ/d²) with c ≈ 4 to 7, so a chain needs τ ≈ d²/2 steps (5,000 at d = 100) to give a
  nearly independent point, matching the O*(d²) warm-start mixing bound for hit-and-run (Lovász
  1999).
- The points are still uniform: starts are inherited uniform points and hit-and-run preserves
  uniformity, so a short chain costs independence, not correctness.
- The drift check compares means at the halfway point and the end, which do not move for a chain
  started at stationarity; it is blind to this by construction.

The predicted memory per node and the predicted out-of-memory boundary on one GPU are in
Section 5.2; the approximation-theoretic limits are in Section 4.11.

---

## 3. Software changes

Changes to how the code runs that leave the algorithms' mathematics unchanged. **This is the
section to implement now.**

**The bit-identity rule for this section.** ART's trees do not depend on batch size, batch
composition, processing order, rank count or steal schedule, so every change below is
bit-identical for ART on CPU (and on GPU within the existing GPU contract: repeated GPU runs
bit-identical, decisions equal to CPU at rounding level) if its claim holds, and the regression
check verifies it. AFINNI's trees depend on batch boundaries, rank and level counters, and free
VRAM (F11). A change that moves any of those is the same mathematics with different random
draws: **not bit-identical for AFINNI**, and so a deliberate numerical change under AGENTS.md
unless node-keyed streams (A1) land first. The column "AFINNI" below says which case applies.

**The bar.** AGENTS.md: a speed change must save `required(t) = max(2 s, min(5% of t, 10 s))` on
a workload that matters (large, deep or high-dimensional trees), or give a 10x speedup of a short
run, without slowing other workloads by as much; a memory change must cut peak usage by at least
10% or let a larger problem fit. Measure with interleaved A/B rounds against a snapshot, and
compare the complete model bit for bit for changes claimed exact.

**Overview, ordered by expected value for large and high-dimensional runs:**

| # | Change | Benefit | ART | AFINNI | Size |
|---|---|---|---|---|---|
| S1 | Leaf buffers on demand; trust matrix only when a predictor needs it | lets d ≥ 85 start (80 GB at d = 100) | identical | identical | small |
| S2 | Memory charges that match the allocations (quadratic partials, AFINNI d x d, constraints per node, CPU host cap); or bound the partials | stops quadratic OOMs at d = 60-80; up to 10x larger affine ART batches | identical | changes trees on GPU, and on CPU at high d | small to moderate |
| S3 | Prediction chunks sized from d; Mahalanobis distance without the d x d gather | prediction at d ≥ 100 without a manual chunk size | identical (routing unaffected) | identical on CPU; verify on GPU | small |
| S4 | A deque of runs instead of level-by-level queues | host RAM `O(quantum x depth)` instead of `O(width)`; steal granularity; tail | identical | changes trees (needs A1 to be exact) | large |
| S5 | The batch as a work quantum, separate from the memory cap | 2.0x on one CPU rank (measured); bounded steal latency | identical | changes trees | small |
| S6 | Kernels that keep their parallelism at large d | speed at d > 100 (quadratic p, d > 128, d > 256) | identical if inner-sum orders are kept | identical if orders are kept | moderate |
| S7 | Host-device traffic: pinned buffers, a copy stream, fewer pool flushes, a device-resident pool, fewer host syncs | GPU speed, largest at low d | identical | identical | moderate to large |
| S8 | A short sleep in the thief's reply loop | 6-8 s of busy-wait per stealing run | identical | schedule changes (already non-reproducible with stealing) | trivial |
| S9 | Steal policy in time units; randomized victims; lifelines; counting termination | idle time at 8+ ranks | identical | schedule changes | moderate |
| S10 | Balanced start and end: parallel `f(X0)`, scatter by work, one repartition, buffer gather | serial terms on rank 0 | identical | parallel `f(X0)` and the gather identical; scatter and repartition change trees | moderate |
| S11 | Oracle farm for expensive oracles | evaluation-bound runs (5,050 evaluations per ART node at d = 100) | identical | identical if batches are kept | large; decide per application |
| S12 | Shared ancestor-plane tables in path-ordered runs | ART constraint memory at low d and great depth; negligible at d ≈ 100 | identical | n/a | moderate |
| S13 | Lower precision where it does not reach training decisions | prediction speed on consumer GPUs | user's choice; not bit-identical | same | small, per site |

### S1. Leaf buffers on demand, trust matrix only when needed

- **What.** Start `ColumnBuffer` at a small capacity (or one derived from the first level's
  accepted count) and rely on its doubling growth; keep `max_clusters_est` as a hint, not a
  reservation. Store `trust_matrix` for ART only when the model will be predicted by kNN
  (`predictor="knn"`), instead of by default.
- **Where.** `src/core/clusters.py:298-305` (initial allocation), `:327-333` (growth);
  `src/engine/refinement.py:82,88-89`; `src/config.py:127,202`.
- **Benefit.** A larger problem fits: AFINNI and ART start at d ≥ 85 (the reservation is 80 GB at
  d = 100, `MemoryError` measured). Measurement: peak RSS at d = 100 before and after, and wall
  time of a large low-d run (the doubling copies must not cost more than `required(t)`; a
  geometric growth copies each row at most about twice).
- **Risk.** Doubling briefly holds two copies of a column; at d = 100 the trust-matrix column of
  `n` leaves is `8 d² n` bytes, so growth near the end of a large run peaks at about 3x the final
  size. Bound it with `check_host_memory` like the queue. Changing the ART default changes what a
  saved model contains; a kNN prediction of an ART model trained without the column must fail
  with a clear message.
- **Bit-identical.** Yes, both algorithms, all backends: no value depends on capacity.

### S2. Memory charges that match the allocations

- **What.** Make each plan's `bytes_per_point` (or a per-node term next to it) charge what its
  batch allocates:
  - ART quadratic leaves: charge the `segment_cross` partials, about `n p^2 / 512` doubles per
    node, or, better, bound them: launch `segment_cross` over groups of segments so the partial
    buffer never exceeds a fixed budget. Chunks are counted from each segment's own start, so the
    grouping does not change any segment's bits.
  - AFINNI: add its about 8 d x d arrays per cluster, charged per cluster through the fewest points
    a cluster holds (as ART charges its per-node search arrays).
  - ART constraints: charge `h (d+1)` per node with the batch's actual `h`, not `2 h (d+1)` per
    point with `h = max_levels + 1`. Its docstring's reason stands: under WSL a card filled to the
    brim pages and runs several times slower. Keep a safety margin through the 65% factor, not
    through a 10x overcharge, and test on WSL.
  - CPU: derive the cap from host RAM (`psutil`) instead of a fixed 32 GiB.
- **Where.** `src/art/plan.py:113-133`; `src/afinni/plan.py:99-118`; `src/numerics/segments.py:200-215`;
  `src/engine/driver.py:315-347`.
- **Benefit.** Quadratic ART runs at d = 60 to 80 stop running out of memory on batches the cap
  admitted (predicted per-node peak over charge 1.7 to 4.9, Section 5.2); affine ART batches grow
  up to 10x at high d, fewer launches per level. Measurement: `gpu_pool_peak_bytes` against the
  charge on the study's quadratic runs; wall time of affine ART at d = 60 to 100 (`required(t)`).
- **Risk.** A tighter charge admits batches nearer the card's limit (the WSL paging case). A
  grouped `segment_cross` adds launches; measure at low d where launches dominate.
- **Bit-identical.** ART: yes on every backend (trees do not depend on batches). AFINNI: the
  charge sets batch boundaries, so changing it changes AFINNI's trees on GPU (and on CPU at high d
  through the host cap). Either leave AFINNI's charge as is until A1, or record it as a
  deliberate numerical change; its docstring already declines a tighter estimate for this reason.

### S3. Prediction chunks sized from d

- **What.** Derive the default `predict` chunk from a memory budget and `k_coarse x d²`, and
  compute the Mahalanobis distances without materializing the `(chunk, k_coarse, d, d)` gather:
  loop over the `k_coarse` candidates, or over sub-chunks, with the same contraction per query.
- **Where.** `src/surrogate/model.py:481,565,604-618` (and the second kNN path at `:791-809`).
- **Benefit.** Prediction at d ≥ 100 without `predict(batch_size=...)` (4 GB per default chunk at
  d = 100). Measurement: peak memory (≥ 10% cut or the d = 100 case fits) and prediction time at low
  d (no slowdown beyond `required(t)`).
- **Risk.** CuPy's `einsum` may route a contraction through batched matrix products whose rounding
  depends on the batch count, so a different chunk can change predictions at rounding level on
  GPU, and with it a borderline trust-region decision. Storing a factor of the trust matrix
  instead would change rounding and the model format.
- **Bit-identical.** Routing prediction (ART) is untouched. kNN prediction: identical on CPU per
  query; verify on GPU against the current chunking.

### S4. A deque of runs instead of level-by-level queues

The structural change; it addresses the memory peak, the steal granularity and the small-level
tail together.

- **What.** Each rank keeps a double-ended queue of *runs*: contiguous `ClusterBatch`es of about
  one quantum of work, nodes in path order. The owner takes runs from the deep end (the most
  recently created children), processes one, and pushes the children back on the deep end,
  merging small runs so they stay full. A thief receives runs from the shallow end: the oldest,
  shallowest nodes, which root the largest remaining subtrees. A run is a self-contained buffer,
  so a grant is a send with no slicing and can be answered between any two quanta.
- **Where.** `src/engine/refinement.py` (`refine_rank` at `:47`, `_refine_level` at `:140-240`);
  `ClusterQueueBuilder` (`src/core/clusters.py`) becomes a run builder; the plans'
  `process_batch` does not change. ART already tolerates mixed depths in a batch (constraints
  padded to a common `h`, depth per node); `max_levels` already uses per-node depth; level
  statistics become per-depth histograms.
- **Benefit.** Host RAM of about `quantum x depth` points instead of a whole level (F8, B11: about
  2 MB per queued ART child at d = 100). Stealing shallow work moves large subtrees in few messages
  while the owner keeps working on data it just produced: owner last-in-first-out, thief
  first-in-first-out, the order the work-stealing efficiency bound depends on (Blumofe and
  Leiserson 1999). Small leftovers from different depths merge into one full batch instead of ten
  nearly empty levels (F9). Measurement: peak RSS on a deep run (≥ 10% or a larger problem
  fits, decided from the study's `host_rss_peak_bytes`); wall time on the Section 1.2 cases and on
  GPU.
- **Risk.** The largest rewrite; per-level logging and the companion's description of the
  level-by-level loop change with it.
- **Bit-identical.** ART: identical trees on 1, 2 and 4 ranks must be the acceptance test. AFINNI:
  changes the batches, so it needs A1 first to be checkable.

### S5. The batch as a work quantum

- **What.** Separate three numbers `batch_size` conflates:

  | Quantity | Meaning | How to set it |
  |---|---|---|
  | memory cap | most points the device can hold | `_cap_batch_size` |
  | quantum | points per batch | CPU: sized to cache (50,000 to 200,000 points measured best); GPU: the larger of "enough rows to keep the card busy" and about 0.2 to 0.5 s of work, never above the cap |
  | steal unit | smallest transferable piece | one quantum |
- **Where.** `src/config.py:200`; `src/engine/driver.py:315-347`; `src/engine/refinement.py:179-215`.
- **Benefit.** 2.0x on one CPU rank (Section 1.2, 159 s to 80 s, far above `required(t) = 7.95 s`);
  steal latency bounded by one quantum. At d ≈ 100 on GPU the cap already limits batches (F1), so
  expect less there. Dinan et al. (2007) found polling interval and chunk size first-order
  parameters for MPI work stealing, worth up to 2x.
- **Risk.** On GPU, smaller batches mean more launches; the GPU quantum must be found by
  measurement and will be much larger than the CPU one. Profile `splitter.split` at two batch
  sizes first: a cheaper fix inside the component may exist.
- **Bit-identical.** ART: yes (sweep the quantum, compare complete models). AFINNI: changes trees
  (statistically equivalent, leaf counts within 0.7%); needs A1 to be exact.

### S6. Kernels that keep their parallelism at large d

- **What.** Cooperative versions of the kernels that run one thread per system:
  `batched_inplace_cholesky_solve` (one block per cluster, as `spd_factor_coop` does for the
  quadratic factor) and the triangular solves of `spd_solve_factored`; tiled loops over d in
  place of the `MAXD 128` register cache of `cluster_stats.cu` and the `MAXD 256` limit of
  `hr_walk`.
- **Where.** `src/kernels/small_linalg.cu:98,135`; `src/numerics/linalg.py:121-133`;
  `src/kernels/cluster_stats.cu:69-97`; `src/kernels/hit_and_run.cu:8,70`;
  `src/art/sampling.py:179-191`.
- **Benefit.** Speed at d > 100 and for quadratic p; to be measured on a CUDA machine on the
  study's d = 100 to 300 runs. Only worth it where the kernel shows up in an Nsight profile
  (`AFINNI_NVTX=1`).
- **Risk.** Every entry must keep its sequential inner sum in the same order, or bits change; the
  `spd_factor_coop` comment states the pattern. Kernel tests must cover both sides of each
  threshold.
- **Bit-identical.** Yes, both algorithms, if the inner-sum orders are kept (GPU only; CPU
  unaffected).

### S7. Host-device traffic on the GPU

- **What.**
  - Pinned host buffers for batch uploads and child/leaf downloads, on a second stream so copies
    overlap compute (Daiss et al. 2022 measured a further 1.5x from overlap on top of aggregated
    kernels).
  - Flush the memory pool once per level, or when the pool exceeds a budget, instead of after
    every batch and twice inside an AFINNI batch; keep the flush before `_cap_batch_size`, which
    the cap's reading needs.
  - A device-resident pool with host spill: children stay on the device while the pool fits a
    budget, the shallow end spills to host; only spilled and stolen runs cross the bus. Children
    are compacted with prefix sums, which is deterministic (Merrill et al. 2012; Zhou et al. 2008
    build whole kd-trees this way). With S4, small subtrees are finished in place, without host
    round trips (cuML's forest builder and PLANET switch regime for small nodes).
  - Find host syncs in the batch path (`int(...)`, `bool(xp.any(...))`) with
    `nsys profile --trace=cuda,nvtx,osrt` and remove those that are not needed for a decision;
    `segment_cross` already sizes its partials from an upper bound to avoid one.
- **Where.** `src/engine/refinement.py:263-285`; `src/afinni/plan.py:171,176,267`;
  `src/core/clusters.py` (builders).
- **Benefit.** Largest for AFINNI at low d, where arithmetic intensity is lowest and the design
  doc's measurements name transfers and memory traffic as AFINNI's main GPU cost; smaller at high
  d (F12). Not measured. Measurement: Nsight timeline and wall time on a large low-d AFINNI run and
  a deep ART run; peak VRAM as the constraint for the flush frequency.
- **Risk.** A device-resident pool competes with the batch for VRAM; the cap must account for it.
  Pinned memory is a limited resource on shared nodes.
- **Bit-identical.** Yes, both algorithms, provided the batches stay the same (the pool budget must
  not feed into `batch_size`).

### S8. A short sleep in the thief's reply loop

- **What.** Sleep about 1 ms per iteration of the wait for a grant or denial, or block on
  `Probe` for either tag.
- **Where.** `src/engine/work_stealing.py:320-353`.
- **Benefit.** Removes the 6 to 8 s of `time_state_update` busy-wait per stealing run (Section 1.2),
  which competes for cores with the victims when ranks share a node. Measurement: idle and wall
  time on the 4-rank cases; the change is small enough that freeing cores is the main point.
- **Risk.** A longer sleep delays the thief's start by up to the sleep.
- **Bit-identical.** ART: yes. AFINNI with stealing is already schedule-dependent.

### S9. Steal policy in units of time

- **What.**
  - **Threshold.** Grant when the victim's remaining work exceeds about twice the cost of moving
    it plus one quantum, with work in estimated seconds from a per-phase throughput (EWMA) that
    each rank measures on itself; this replaces the absolute constants.
  - **Victims.** Pick at random, or sample two and ask the one with more advertised work, instead
    of `argmax`. Randomized selection is the provably efficient choice (Blumofe and Leiserson
    1999) and avoids the herd.
  - **Waiting.** After a few failed attempts, register with a small fixed set of lifeline
    neighbours and sleep; a neighbour that gets work pushes some along (Saraswat et al. 2011).
  - **Load information.** Drop all-to-all gossip; lifelines need none, and two-choice selection
    can read loads piggybacked on steal replies.
  - **Termination.** Count transfers sent and received and terminate when two successive waves
    agree (Mattern 1987), closing the window of F7.
  - **Leases.** Allow a thief to re-offer part of a large received grant (M14), keeping the rule
    that a grant cannot bounce back whole.
- **Where.** `src/engine/work_stealing.py:38-40,175-197,201-353,355-379`; `art/plan.py:94-100`.
- **Benefit.** Idle rank-seconds at 8 to 16 ranks on the skewed case of Section 1.2. Only once
  multi-rank runs matter.
- **Risk.** Protocol changes; the `tests/mpi` protocol tests must keep "exactly one reply per
  request".
- **Bit-identical.** ART: yes (identical to serial). AFINNI: schedule-dependent already.

### S10. Balanced start and end

- **What.**
  - Scatter `X0` first and evaluate `f(X0)` on every rank, then partition.
  - Scatter by work, not count: cut the root list at equal prefix sums of `plan.work_sizes`. For
    ART with few roots, refine on rank 0 until there are enough nodes to share, then scatter.
  - Repartition once after a pilot: after one or two levels each root's rejection rate is known
    and children cost roughly what their parents did. One weighted prefix cut over nodes in
    `(root, locational_code)` order, followed by `Alltoallv`, fixes the bulk imbalance before it
    is stolen away piece by piece (forest-of-octree codes partition this way: Burstedde et al.
    2011; Harlacher et al. 2012). The survey found no published head-to-head of curve
    repartitioning against work stealing on adaptive trees with unpredictable cost; the closest
    result (Lifflander et al. 2012) is that repartitioning wins when load persists and stealing
    when it does not. Hence: repartition early, steal for the rest.
  - Gather leaf columns with buffer `Gatherv`, not pickled dictionaries; with ordered runs each
    rank's leaves arrive sorted by key and rank 0 merges instead of sorting.
- **Where.** `src/engine/driver.py:148-166,193-199,248-259,294-313`; `src/art/plan.py:176-187`.
- **Benefit.** For an expensive oracle, parallel `f(X0)` removes the largest serial term; a
  weighted scatter removes ART's one-root start on rank 0. Measurement: wall time on multi-rank
  runs with a costly oracle; gather time on a large model.
- **Risk.** Moderate; the scatter's views and the MPI failure policy must be kept.
- **Bit-identical.** Parallel `f(X0)` (deterministic oracle) and the buffer gather: yes, both
  algorithms. Scatter by work and repartition: ART yes; AFINNI changes which rank processes which
  cluster, and the rank enters its stream (F11). Distributed initial clustering changes AFINNI's
  initial clusters and is listed as A9.

### S11. Oracle farm for expensive functions

- **What.** Collect the new sample points of a batch and hand them to oracle workers while the
  device fits another batch (a manager-worker ensemble, as in libEnsemble); balance ranks by
  pending evaluations; evaluate `f(X0)` in parallel (S10).
- **Benefit.** The benchmarks spend 1 to 9% of wall time in the oracle, so not for them. For a
  real solver as oracle the balance inverts, and at high d it bites sooner: an ART node costs
  n = 50(d+1) evaluations, 5,050 at d = 100. Decide from the measured oracle cost of the target
  application.
- **Bit-identical.** Yes if evaluations are returned in order and batches are unchanged.

### S12. Path order as the engine's curve, and shared ancestor-plane tables

- **What.** Sort by `(root, locational_code left-aligned)`: depth-first order, in which every
  subtree is one contiguous range. That order gives partitioning by key range (S10), steals of
  subtree fragments (S4), sorted assembly and a depth-first routing table for inference, and
  shared ancestor planes: a run carries one table of its distinct planes and `h` 32-bit indices
  per node (`4h` bytes plus a share of the table, against `8 h (d+1)`), gathered by the sampler
  through the index.
- **Where.** `src/core/clusters.py:46-60` (constraint columns); `src/art/sampling.py`,
  `src/kernels/hit_and_run.cu:52-75` (constraint reads); `src/art/routing.py`.
- **Benefit.** At high d the gain is **negligible**: constraints are `8 h (d+1)` bytes per node,
  33 KB at d = 100 with h = 41, against about 2 MB of points per node, under 2% of the memory. What
  limits ART's batches there is the charge (S2), not the storage. Plane tables matter only for
  deep trees at low d; measure the gather's cost in `hr_walk`.
- **Coordinate Morton or Hilbert keys do not transfer**: a 64-bit key holds `floor(64/d)` bits per
  dimension (three at d = 20, less than one at d = 100), the cuts are oblique and data-dependent,
  and single-pass (Karras-style) construction needs the tree to be a pure function of sorted
  keys. For AFINNI's initial clusters, a Hilbert order of the centres' leading two or three
  principal coordinates would give each rank a compact region; it matters only for distributed
  prediction of very large models. Low priority.
- **Bit-identical.** ART: yes if the sampler reads the same plane values in the same order.

### S13. Precision

- **What.** fp32 only at sites that do not reach a training decision, or by the user's explicit
  choice: e.g. the L2 coarse candidate search in kNN prediction, followed by the fp64 refinement.
- **Benefit.** Consumer and laptop GPUs run fp64 at 1/32 to 1/64 of fp32 (M11).
- **Risk.** A coarse search in fp32 can change the candidate set, and with it a prediction.
- **Bit-identical.** No; adopt only by the user's choice, with the GPU model recorded.

### CPU path

The CPU-specific items are in S2 (a host-based cap), S5 (a cache-sized quantum, the largest
measured CPU gain) and S6's counterpart: the CPU hit-and-run already runs one fused Numba walk per
chain (`art/sampling.py:179`). Before adding Numba kernels elsewhere, profile `splitter.split` at
two batch sizes (72% of AFINNI's CPU time); AGENTS.md's rules apply: BLAS for Grams, Numba only
for fused loops NumPy cannot express without many temporaries.

### Not recommended

| Idea | Why not |
|---|---|
| A task runtime (Charm++, HPX, Legion, Dask) | Their GPU work aggregates small tasks into the large batches this engine already forms, and determinism would be in someone else's hands |
| Coordinate Morton keys for the training tree | too few bits in high d, and the cuts are oblique (S12) |
| One-sided (RMA) stealing | MPICH emulates RMA with active messages, so a busy victim still blocks the thief; a small quantum gives the same responsiveness with two-sided MPI |
| CUDA-aware MPI for steals | stolen work is on the host; revisit only after S7's device-resident pool |
| Several ranks per GPU through MPS | helps only when a rank leaves the card idle; fix the idle time first |
| Library batched solvers (cuBLAS, cuSOLVER, MAGMA) | they choose algorithms by batch count, which breaks the determinism contract (also an AGENTS.md rule) |
| Global hash tables for lineage | carried or run-local planes need no lookup across ranks |

### References for this section

Bibliographic details were confirmed from publisher or archive records by the survey unless
noted. Where the text relies on a paper's mechanism rather than its abstract, it says so.

**Work stealing and termination**

- Blumofe, Leiserson. Scheduling multithreaded computations by work stealing. JACM 46(5), 1999.
  doi:10.1145/324133.324234
- Dinan, Olivier, Sabin, Prins, Sadayappan, Tseng. Dynamic load balancing of unbalanced
  computations using message passing. IPDPS workshops, 2007. https://www.cs.unc.edu/~olivier/pmeo07.pdf
- Dinan, Larkins, Sadayappan, Krishnamoorthy, Nieplocha. Scalable work stealing. SC 2009.
  doi:10.1145/1654059.1654113
- Olivier et al. UTS: an unbalanced tree search benchmark. LCPC 2006.
- Saraswat, Kambadur, Kodali, Grove, Krishnamoorthy. Lifeline-based global load balancing.
  PPoPP 2011. doi:10.1145/1941553.1941582
- Zhang et al. GLB: lifeline-based global load balancing library in X10. arXiv:1312.5691
- Perarnau, Sato. Victim selection and distributed work stealing performance. IPDPS 2014.
  doi:10.1109/IPDPS.2014.74
- Lifflander, Krishnamoorthy, Kale. Work stealing and persistence-based load balancers for
  iterative overdecomposed applications. HPDC 2012. doi:10.1145/2287076.2287103
- Menon, Kale. A distributed dynamic load balancer for iterative applications. SC 2013.
  doi:10.1145/2503210.2503284
- Mattern. Algorithms for distributed termination detection. Distributed Computing 2(3), 1987.
  doi:10.1007/BF01782776
- Shiina, Taura. Itoyori. SC 2023. doi:10.1145/3581784.3607049 (source of the MPICH RMA limitation)

**Curve partitioning and distributed trees**

- Burstedde, Wilcox, Ghattas. p4est. SISC 33(3), 2011. doi:10.1137/100791634
- Sundar, Sampath, Biros. Bottom-up construction and 2:1 balance refinement of linear octrees in
  parallel. SISC 30(5), 2008.
- Harlacher, Klimach, Roller, Siebert, Wolf. Dynamic load balancing for unstructured meshes on
  space-filling curves. IPDPSW 2012.
- Xiao, Biros. Parallel algorithms for nearest neighbor search problems in high dimensions. SISC
  2016. doi:10.1137/15M1026377
- Bahmani, Moseley, Vattani, Kumar, Vassilvitskii. Scalable k-means++. PVLDB 5(7), 2012.
  arXiv:1203.6402
- Panda, Herbach, Basu, Bayardo. PLANET. PVLDB 2, 2009. doi:10.14778/1687553.1687569

**GPU tree construction, batching and reproducible sums**

- Karras. Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. HPG 2012.
- Zhou, Hou, Wang, Guo. Real-time kd-tree construction on graphics hardware. ACM TOG 27(5), 2008.
  doi:10.1145/1409060.1409079
- Merrill, Garland, Grimshaw. Scalable GPU graph traversal. PPoPP 2012.
- Mitchell, Adinets, Rao, Frank. XGBoost: scalable GPU accelerated learning. arXiv:1806.11248
- Mitchell, Frank. Accelerating the XGBoost algorithm using GPU computing. PeerJ CS 3 (2017).
  doi:10.7717/peerj-cs.127
- Wen et al. ThunderGBM. JMLR 21 (2020).
- Daiss, Diehl, Marcello, Kheirkhahan, Kaiser, Pflueger. From task-based GPU work aggregation to
  stellar mergers. SC22 workshops. arXiv:2210.06438
- cuML random forest (vendor blog, not peer reviewed).
  https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/
- Haidar, Abdelfattah, Zounon, Tomov, Dongarra. A guide for achieving high performance with very
  small matrices on GPU: batched LU and Cholesky. IEEE TPDS 29(5) (2018).
  doi:10.1109/TPDS.2017.2783929
- Abdelfattah et al. Batched GEMM and Cholesky. ISC 2016, LNCS 9697.
- Ahrens, Demmel, Nguyen. Algorithms for efficient reproducible floating point summation. ACM TOMS
  46(3) (2020). doi:10.1145/3389360. Binned reproducible sums (ReproBLAS), the published
  alternative to fixed-order segmented sums if the order constraint ever becomes too costly;
  fixed-order segmented sums are a sound design.

**Random streams and oracles**

- Salmon, Moraes, Dror, Shaw. Parallel random numbers: as easy as 1, 2, 3. SC 2011.
- Hudson, Larson, Navarro, Wild. libEnsemble. IEEE TPDS 33(4), 2022. doi:10.1109/TPDS.2021.3082815

---

## 4. Algorithmic changes

Changes to what the algorithms compute. Each changes trees and results, so each is a deliberate
numerical change under AGENTS.md: listed and justified, with new regression references saved
from the new baseline. **None should be adopted on an implementation alone.** Each needs a study
that checks, at least:

- **accuracy**: test errors against the tolerance (max, RMS, 99th percentile, share of uniform test
  points within tolerance), not only the training-sample acceptance;
- **convergence and order of convergence** on the convergence study's problems
  (`CONVERGENCE_STUDY.md`: Part A, the AFINNI manuscript's experiments, `studies/convergence/paper/`;
  Part B, the reference ART against AFINNI's ART, `studies/convergence/art/`, including
  `exact_checks.py`), and on the high-dimensional study's problems (Section 5);
- **the theoretical estimate** the change rests on (a bias bound, a mixing bound, a rate);
- **cost**: leaves, evaluations, wall time and peak memory against the current method;
- **the companion**: `docs/report/` states the algorithms, so an adopted change updates it and
  its figures in the same change.

### A1. Node-keyed AFINNI random streams (enabling change)

- **What.** Give every AFINNI cluster the identity ART has: `root` = its initial cluster and a
  `locational_code` doubling at each split, and draw augmentation randoms from
  `counter_uniform(node_keys(seed, root, locational_code), ...)` as ART does. Make the batch grid
  (`afinni/plan.py:123-135`) independent of batch boundaries.
- **Why.** AFINNI's trees become independent of batch size, rank count, steal schedule and free
  VRAM (F11). Every software change of Section 3 then becomes bit-identical for AFINNI too and
  verifiable on any number of ranks; the stream collisions of F11 disappear; AFINNI's splits can be
  recorded as planes, which the ancestor-plane work for the sampler needs.
- **Expected effect.** Statistically equivalent trees (a different random stream); two `int64`
  columns on the queue, which the wire format already carries for ART
  (`src/core/clusters.py:56-57`).
- **Verify.** Train with three batch sizes, two rank counts and stealing on, and compare every leaf
  column bit for bit; check that no other path depends on batch composition (NumPy's grouped
  products in `count_groups` run per matrix, but the three-batch-size check settles it). Then
  Part A of the convergence study (errors and slopes within the paper's claims) and new regression
  references. Not a speed change: it is a reproducibility contract, which AGENTS.md exempts from
  the speed bar. Do it first if multi-rank AFINNI or any AFINNI-affecting item of Section 3 is
  wanted.

### A2. A floor for the Hessian level set

- **What.** Give `N_cov` a floor in d, e.g. `max(2d, min_pts_per_cluster // 3)` as the gradient
  points already have (`max(2d, min_pts_per_cluster // 4)`), or use a pseudo-inverse that gives
  unexplored directions no weight instead of the largest.
- **Why.** For d > 33 with the defaults the level-set scatter is singular, and the eigenvalue floor
  makes the split direction point along the directions the estimate knows least (M8, B1). ART's
  lead direction inherits it.
- **Expected effect.** A nonsingular level-set scatter at every d, not only for
  d < `min_pts_per_cluster // 3`. Where `2d ≤ min_pts_per_cluster // 3` (d ≤ 16 with the default
  100) a `max(2d, ...)` floor leaves `N_cov` unchanged, so low-d results stay bit-identical; above
  that the level set takes a larger share of each cluster's points, or clusters must be larger.
- **Verify.** Bit-identity of the regression check where the floor does not bind; the convergence
  study's Part A (the manuscript's Hessian-versus-no-Hessian comparison, leaf counts for equal
  error); the high-dimensional study's `aniso_quadratic` and `rosenbrock` runs with
  `--library-min-pts` against the scaled `min_pts` the study uses.

### A3. Better curvature directions: principal Hessian directions, active subspaces

- **What.** Estimate the residual Hessian per node from all n points:
  `M = Σ r_i (x_i − c)(x_i − c)^T`, with `r_i` the residuals of the affine fit (principal Hessian
  directions, Li 1992). Use its top eigenvectors as the split direction and in ART's direction
  bank in place of the random directions (`art/plan.py:421,453-475`). Where gradients are
  available (gradient modes), the active-subspace matrix `E[∇f ∇f^T]` is the alternative.
- **Why.** The split direction is the lever at high d, more than the sampler or the scheduler: the
  leaf count scales with the effective dimension only if the cuts find the curved directions. The
  current estimators are weak at d = 100: a level set of `N_cov` points (B1), and a bank of 6
  directions of which 2 are random, almost orthogonal (|cos| ≈ 0.1) to any fixed direction at
  d = 100. `M` costs `n d²` operations, as the covariance does, and is well determined for n ≫ d
  (ART has n = 50(d+1)); by Stein's lemma it estimates `Cov · H · Cov` for Gaussian-like designs.
- **Expected effect.** On `aniso_quadratic` (eigenvalues `2 i^-3`, rotated), a leaf count roughly
  independent of d, about 10⁴ to 10⁵ leaves at tol 0.01, if the directions are found; no gain on
  isotropic curvature (no structure to find).
- **Verify.** The high-dimensional study's `aniso_quadratic` case is the separating test (finest
  tolerance against d for both algorithms); the convergence study's Part A and B for no loss at low
  d; the theory: `M` estimates the Hessian only up to the design's covariance, which for thin
  nodes is anisotropic itself, so the estimator must be whitened by the node covariance.

### A4. A scale-invariant fit regularization

- **What.** Solve the leaf least squares in the node's whitened coordinates (or its covariance
  eigenbasis) with a *relative* eigenvalue floor, e.g. `1e-12 λ_max` applied only when an
  eigenvalue is that small; or QR / least squares on centred features with an rcond cut, as the
  reference ART does (ridge 0, rcond-based solver choice). For quadratic leaves, the same in the
  scaled features.
- **Why.** The ridge `max(1e-8 tr Cov, 1e-12)` shrinks the slope along direction i by about
  `1e-8 Σ_j σ_j² / σ_i²`: `1e-8 d` on a round node and growing as `A²` with aspect ratio A (`A⁴` for
  the quadratic features) (M9, B6). Measured: an exactly affine function fitted with relative
  error 1.2e-8 at aspect ratio 1 and 0.14 at 1e4 (reference ART about 1e-14); 3.6e-6 x std at
  d = 100 on a round node. Thin nodes are rejected where the reference accepts them, which costs
  leaves at tight tolerances on plane waves and Rosenbrock's valley. The term comes from the
  AFINNI manuscript's Eq. 3, so the decision is whether ART leaves (and AFINNI's, since
  `AffineLSQModel` is shared, `local_models.py:266-290`) keep it.
- **Expected effect.** Exact affine (and quadratic) reproduction at every aspect ratio and d;
  possibly fewer leaves at tight tolerances. A ridge also stabilizes near-singular fits (few
  points, degenerate clusters), so the floor must still catch those (`rank_deficient`).
- **Verify.** `exact_checks.py` thin box to about 1e-14 at aspect ratios 1 to 1e4; Part B of the
  convergence study (plane-wave sweep to 1e-3, whether accuracy gaps to the reference close at
  tight tolerances); Part A for AFINNI if the shared model changes (the manuscript's numbers were
  produced with the ridge); the high-dimensional `affine` case (the ridge floor flat in d).

### A5. Sampling that mixes in high d

- **What.** Replace 20 isotropic hit-and-run steps with coordinate hit-and-run whose slacks are
  updated incrementally (`O(h)` per step instead of `O(h d)`: a coordinate move changes each slack
  by one column times the step), run for about d sweeps (about d²/2 coordinate steps); keep exact
  sampling for box roots. The kernel already has a coordinate mode for recovery walks
  (`hit_and_run.cu:83`) that recomputes every slack. Further options: rounding (CHRR, volesti) for
  thin nodes, or affine-invariant barrier walks (Dikin, Vaidya: one d x d Cholesky per step,
  feasible as a batched kernel).
- **Why.** Measured: at d = 100, 20 steps leave endpoints correlated 0.98 with their starts (B2,
  Section 2.2); about half of each node's oracle calls add little information, and the acceptance
  test sees fewer independent points than n. A d-step coordinate sweep costs about one isotropic
  step with incremental slacks, and coordinate hit-and-run mixes in polynomial time (Laddha and
  Vempala 2023).
- **Expected effect.** Nearly independent new points per node at the same cost; a more reliable
  acceptance at high d.
- **Verify.** `studies/high_dim/hit_and_run_mixing.py` (ρ near 0 and disp near 1 at d = 10 to 200);
  a uniformity test of the endpoints in thin polytopes; the drift check replaced or complemented
  by a check that sees the start-end correlation; Part B of the convergence study (accuracy and
  leaves against the reference, which uses its own sampler) and the share within tolerance on test
  points in Section 5.

### A6. Fewer points per node

- **What.** Lower `sample_multiplier` m (50 by default, `config.py:99`), e.g. m = 10 for affine
  leaves and m = 4 for quadratic ones.
- **Why.** m = 50 is about 2x redundant at d = 100 because of B2; evaluations per node are
  `m (d+1)` (5,050 at d = 100) and quadratic nodes `m p` (B3, B10). m = 4 fits a quadratic node on
  24 GB up to d ≈ 100 (Section 5.2).
- **Expected effect.** Proportionally fewer evaluations and less memory per node; a less reliable
  sample maximum, so more test points above tolerance.
- **Verify.** The study's `art_affine_m10` and `art_quad_m4` variants against m = 50 on
  `aniso_quadratic` and `rosenbrock` (share within tolerance, evaluations to reach each tolerance);
  best after A5, since better points make fewer points enough.

### A7. Quadratic leaves in a subspace

- **What.** A quadratic in the top k eigen-directions of A3's estimate plus affine terms in the
  rest: `p = d + 1 + k(k+1)/2` coefficients instead of `(d+1)(d+2)/2`.
- **Why.** Full quadratic leaves hit memory at d = 60 to 80 and cost `n p² ∝ m d⁶ / 8` (B3, B4); a
  subspace quadratic keeps most of the error reduction where curvature is concentrated.
- **Expected effect.** Quadratic-leaf accuracy on low-effective-dimension functions at affine-like
  cost; no gain on isotropic curvature.
- **Verify.** `aniso_quadratic` (exact when k covers the curved directions) and `quadratic`
  (k < d cannot be exact: measures the loss) at d = 20 to 100; convergence order against full
  quadratic leaves at d ≤ 20 in Part B. Depends on A3.

### A8. Acceptance on fresh points

- **What.** Test AFINNI's accepted clusters on fresh points drawn uniformly in the region a query
  would be routed to (its trust region, or its cell), not only on its own points; for ART, a
  default `validation_samples > 0`. Report the share of uniform test points within tolerance with
  every run, and keep `AFINNISurrogate.audit` as the global claim (ART).
- **Why.** AFINNI's points are convex combinations inside the cluster's hull, a vanishing part of
  the cell in high d (M13); measured: converged at d = 8, tol 0.3, with 54% of test points within
  tolerance (ART 99.7%). For both algorithms, acceptance is a maximum over n samples, a statement
  about typical points, not the supremum (Section 4.11).
- **Expected effect.** Accepted leaves that hold on the region they serve; more leaves and
  evaluations for AFINNI.
- **Verify.** Part A of the convergence study (trust-region coverage, errors against tolerance, the
  manuscript's claims) and the `error_vs_tol` plot of Section 5 (points on or below the diagonal).

### A9. Distributed initial clustering for AFINNI

- **What.** Cluster a sample on rank 0, broadcast the centres, assign points locally and exchange
  them with one `Alltoallv`; k-means|| (Bahmani et al. 2012) is the standard scalable seeding if a
  sample is not enough.
- **Why.** The initial clustering is serial on rank 0 (F10).
- **Expected effect.** Different initial clusters, hence different trees; a shorter serial start.
- **Verify.** Part A of the convergence study (initial-cluster quality enters leaf counts and the
  manuscript's numbers); time of the start on large `N0`.

### 4.10 Literature, and what it implies

Bibliographic details were confirmed from primary records by a sub-agent unless marked **[U]**
(unconfirmed) or **[M]** (from memory, not checked). The key results are taken from abstracts.

**Adaptive and oblique trees, and partition approximation theory**

- Murthy, Kasif, Salzberg. A system for induction of oblique decision trees. JAIR 2 (1994).
  doi:10.1613/jair.63. Randomised hill-climbing for oblique splits.
- Wickramarachchi et al. HHCART: an oblique decision tree. CSDA 96 (2016).
  doi:10.1016/j.csda.2015.11.006. Splits along covariance eigenvectors via a Householder
  reflection, as AFINNI does.
- Cattaneo, Chandak, Klusowski. Convergence rates of oblique regression trees for flexible function
  libraries. Ann. Statist. 52(2) (2024). doi:10.1214/24-AOS2354. Greedy oblique trees adapt to
  sums of ridge functions: the only tree theory found that escapes the exponent in d, and only
  under that structure.
- Gramacy, Lee. Bayesian treed Gaussian process models. JASA 103 (2008).
  doi:10.1198/016214508000000689
- Chipman, George, McCulloch. Bayesian treed models. Mach. Learn. 48 (2002).
  doi:10.1023/A:1013916107446
- Friedman. Multivariate adaptive regression splines. Ann. Statist. 19(1) (1991).
  doi:10.1214/aos/1176347963. Works when interactions are of low order.
- Binev, Cohen, Dahmen, DeVore, Temlyakov. Universal algorithms for learning theory I. JMLR 6
  (2005); Part II in Constr. Approx. 26 (2007), doi:10.1007/s00365-006-0658-z. Adaptive-partition
  least squares reaches N^(−s/d)-type rates.
- Chen, Sun, Xu. Optimal anisotropic meshes for minimizing interpolation errors in L^p. Math. Comp.
  76 (2007). doi:10.1090/S0025-5718-06-01896-5. The error is about N^(−2/d) ‖det(H)^(1/d)‖, so
  anisotropy helps only through the eigenvalue spectrum.
- Stone. Optimal global rates of convergence for nonparametric regression. Ann. Statist. 10 (1982).
  doi:10.1214/aos/1176345969
- Novak, Woźniakowski. Approximation of infinitely differentiable multivariate functions is
  intractable. J. Complexity 25 (2009) [venue and DOI U]. Smooth functions still need at least
  2^⌊d/2⌋ values.
- The specific ART paper this code follows was **not found** by search [U]; cite it from the code's
  authors.

**Sparse grids**

- Bungartz, Griebel. Sparse grids. Acta Numerica 13 (2004). doi:10.1017/S0962492904000182
- Gerstner, Griebel. Dimension-adaptive tensor-product quadrature. Computing 71 (2003).
  doi:10.1007/s00607-003-0015-5
- Pflüger, Peherstorfer, Bungartz. Spatially adaptive sparse grids for high-dimensional data-driven
  problems. J. Complexity 26 (2010). doi:10.1016/j.jco.2010.04.001
- Jakeman, Roberts. Local and dimension adaptive stochastic collocation. LNCSE 88 (2013).
  doi:10.1007/978-3-642-31703-3_9

Sparse grids reach d ≈ 100 only when the effective dimension is low, and they are axis-aligned, so
they cannot exploit an oblique ridge.

**Low effective dimension**

- Constantine, Dow, Wang. Active subspace methods in theory and practice. SISC 36(4) (2014).
  doi:10.1137/130916138
- Constantine. Active Subspaces. SIAM (2015).
- Cohen, Daubechies, DeVore, Kerkyacharian, Picard. Capturing ridge functions in high dimensions
  from point queries. Constr. Approx. 35 (2012). doi:10.1007/s00365-011-9147-6
- Fornasier, Schnass, Vybíral. Learning functions of few arbitrary linear parameters in high
  dimensions. FoCM 12 (2012). doi:10.1007/s10208-012-9115-y
- Hokanson, Constantine. Data-driven polynomial ridge approximation using variable projection. SISC
  40(3) (2018). doi:10.1137/17M1117690
- Zahm, Constantine, Prieur, Marzouk. Gradient-based dimension reduction of multivariate
  vector-valued functions. SISC 42(1) (2020). arXiv:1801.07922
- Romor, Tezzele, Rozza. A local approach to parameter space reduction for regression and
  classification tasks. J. Sci. Comput. (2024). doi:10.1007/s10915-024-02542-0. Local active
  subspaces: the closest published analogue of a Hessian-aligned partition.
- Caflisch, Morokoff, Owen. Valuation of mortgage-backed securities using Brownian bridges to reduce
  effective dimension. J. Comput. Finance 1 (1997). doi:10.21314/JCF.1997.005. Defines effective
  dimension.
- Rabitz, Aliş. General foundations of high-dimensional model representations. J. Math. Chem. 25
  (1999). doi:10.1023/A:1019188517934
- Li, K.-C. On principal Hessian directions for data visualization and dimension reduction: another
  application of Stein's lemma. JASA 87(420) (1992) **[M]: from memory, not checked against the
  record.** The residual-weighted moment `Σ r_i (x_i − c)(x_i − c)^T` estimates `Cov · H · Cov`
  from all points.

**Polytope sampling**

- Smith. Oper. Res. 32(6) (1984). doi:10.1287/opre.32.6.1296. Introduces hit-and-run.
- Lovász. Hit-and-run mixes fast. Math. Program. 86 (1999). doi:10.1007/s101070050099. O*(d² R²/r²)
  steps from a warm start.
- Lovász, Vempala. Hit-and-run from a corner. SIAM J. Comput. 35(4) (2006).
  doi:10.1137/S009753970544727X
- Laddha, Vempala. Convergence of Gibbs sampling: coordinate hit-and-run mixes fast. DCG 70 (2023).
  arXiv:2009.11338
- Haraldsdóttir, Cousins, Thiele, Fleming, Vempala. CHRR: coordinate hit-and-run with rounding.
  Bioinformatics 33(11) (2017). doi:10.1093/bioinformatics/btx052. Rounding as the remedy for slow
  mixing in thin polytopes (deep ART nodes); works in thousands of dimensions.
- Cousins, Vempala. A practical volume algorithm. Math. Program. Comput. 8 (2016).
  doi:10.1007/s12532-015-0097-z
- Kannan, Narayanan. Random walks on polytopes and an affine interior point method for LP. Math.
  Oper. Res. 37(1) (2012). doi:10.1287/moor.1110.0519. The Dikin walk, which is affine-invariant.
- Chen, Dwivedi, Wainwright, Yu. Fast MCMC sampling algorithms on polytopes. JMLR 19(55) (2018).
  The Vaidya and John walks.
- Kook, Lee, Shen, Vempala. Sampling with Riemannian HMC in a constrained space. NeurIPS 2022.
  arXiv:2202.01908
- Gryazina, Polyak. Random sampling: billiard walk algorithm. EJOR 238(2) (2014).
- Chalkis, Fisikopoulos. volesti. R Journal 13(2) (2021). doi:10.32614/RJ-2021-077

**Other piecewise-linear surrogates in high d**

- Chang et al. Algorithm 1012: DELAUNAYSPARSE. ACM TOMS 46(4) (2020). doi:10.1145/3422818.
  Feasible at d ≈ 50, still N^(−2/d).
- Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks 94 (2017).
  doi:10.1016/j.neunet.2017.07.002
- Oseledets. Tensor-train decomposition. SISC 33(5) (2011). doi:10.1137/090752286
- Gorodetsky, Karaman, Marzouk. A continuous analogue of the tensor-train decomposition. CMAME 347
  (2019). doi:10.1016/j.cma.2018.12.015

**What the literature implies for this project**

1. **Is the approach right for d ≈ 100?** For generic f no method is; that is a theorem. For f
   with a few strongly curved, possibly oblique directions, yes: oblique, curvature-aligned
   partitions are what the theory (Cattaneo et al. 2024; local active subspaces) says can win, and
   axis-aligned methods (sparse grids, MARS, CART) cannot. The leaf count then scales with the
   effective dimension, while the cost per leaf still scales with d (affine) or d² (quadratic).
2. **The split direction is the lever** (A2, A3), more than the sampler or the scheduler.
3. **Mixing.** Hit-and-run's d² mixing is a known property; the practical remedies are rounding
   plus coordinate hit-and-run (CHRR, volesti) or affine-invariant barrier walks (A5).
4. **Reproducible sums.** Fixed-order segmented sums are sound; binned reproducible accumulation
   (ReproBLAS) is the published alternative.

### 4.11 Approximation limits (method)

**The rate.** An affine leaf of widths `h_i` along the Hessian's eigenvectors has residual
`½ Σ λ_i (y_i² − E y_i²)`. For a generic smooth f in d dimensions the number of cells grows as
`ε^(−d/2)` (the `N^(−2/d)` rate). Novak and Woźniakowski show that even C^∞ functions need at least
`2^⌊d/2⌋` values, so smoothness alone does not help.

**Sample-based acceptance concentrates.** Both methods accept on the maximum over their own n
samples, not on the supremum. In high d the residual is a sum of many independent terms, so its
values concentrate:

| isotropic case (d equal λ, h) | |
|---|---|
| sup over the cell | ≈ d λ h²/12 |
| standard deviation | ≈ √d λ h²/27 |
| max over n samples | ≈ √(2 ln n) x std |

Coarse tolerances are therefore reachable at any d, but the certificate is a statement about
typical points, and the error at a cell's corners is several times larger.

**The threshold below the root error.** Halving one direction removes 15/16 of that direction's
share of the residual variance; halving the residual standard deviation of an isotropic node
requires cutting about 3/4 of all directions, about `2^(0.75 d)` leaves. Below the error of a
single node the leaf count explodes:

`ε_min(d) ≈ ε_root(d) · B^(−2/d_eff)`

with B the leaf budget and `d_eff` the number of directions whose curvature exceeds the tolerance.
At d = 100 with B = 10⁶ and `d_eff ≈ d`, `B^(−2/d) = 0.76`: **refinement buys at most about 25%
over one global fit.**

**Root fits (`predict_limits.py --root-fit`, measured on CPU, d = 10 to 120).** Global least-squares
fits on n = 50 p uniform points, errors relative to std f; almost independent of d:

| problem | affine leaf, sample max / test RMS | quadratic leaf, sample max |
|---|---|---|
| affine | 1e-14 (exact) | 1e-14 |
| quadratic (isotropic) | 0.18 / 0.052 | 1e-14 (exact) |
| aniso_quadratic | 7-12 / 1.01 | 1e-14 (exact) |
| rosenbrock | 3.0-3.9 / 0.86 | 1.9-2.2 (d ≤ 40) |

**Predictions the high-dimensional study tests:**

| problem | prediction |
|---|---|
| affine | Every variant reaches every tolerance down to the ridge floor, about 4e-6 at d = 100 (B6). One leaf for ART, 64 for AFINNI (its initial clusters) |
| quadratic, affine leaves | tol 0.3 at the root at every d; tol 0.1 reachable at d = 20 (about 2^15 leaves); **at d ≥ 40 nothing finer than 0.3** |
| quadratic, quadratic leaves | one exact leaf until the out-of-memory boundary of Section 5.2 |
| aniso_quadratic | eigenvalues `2 i^(−3)`, so `Σ λ_i²` converges: about 6 directions need cuts at tol 0.01, an estimated 10⁴ to 10⁵ leaves roughly independent of d, **if the split directions find the rotated eigenvectors**. Random bank directions are almost orthogonal to them at d = 100 (\|cos\| ≈ 0.1) and AFINNI's level-set estimate is noisy (B1), so expect ART (bins over 6 directions plus soft polishing) to degrade with d more slowly than AFINNI. The case that separates the two |
| rosenbrock | a chain of d − 1 coupled quartic terms with a root error of about 3 x std; tol 1.0 needs most coordinates refined: reachable at d ≈ 20 (the smoke run reached it at d = 8 with 1,239 ART leaves), **unreachable at d ≥ 40**. Beyond d ≈ 20, piecewise-affine or piecewise-quadratic partitioning cannot approximate Rosenbrock better than about one standard deviation of f |

---

## 5. High-dimensional scaling study

Scripts in `studies/high_dim/`. Nothing in this section was run on a GPU.

### 5.1 Design

- **Problems** (`studies/high_dim/problems.py`; all on `[−1, 1]^d`; tolerance `tol` relative to the
  standard deviation of f over the test set, `atol = tol · std f`, `rtol` None, error metric Linf):

  | problem | role |
  |---|---|
  | `affine` | `1 + a^T x` with \|a\| = 1. Exact for every leaf, so whatever it cannot reach is an implementation floor |
  | `quadratic` | `src.problems.quadratic`: isotropic curvature. Exact for quadratic leaves, the worst case for affine leaves |
  | `aniso_quadratic` | `src.problems.anisotropic_quadratic`: rotated, eigenvalues `2 i^(−3)`, low effective dimension along oblique directions. Exact for quadratic leaves |
  | `rosenbrock` | `src.problems.rosenbrock`, a = 100: generic |

  `src/problems/affine.py` is fixed at d = 20, so the study defines its own affine function.
- **Variants** (`settings.py`):

  | variant | in default matrix | what it is |
  |---|---|---|
  | `afinni` | yes | AFFINE_COV_HESS, 64 initial clusters, X0 = 64 · 2 · min_pts uniform points |
  | `art_affine` | yes | one box root, soft search, m = 50, X0 = one node's budget |
  | `art_quad` | yes | quadratic leaves, m = 50 |
  | `art_affine_m10` | optional (`VARIANTS=`) | m = 10 |
  | `art_quad_m4` | optional | quadratic, m = 4, bins search |

  Study settings that differ from library defaults, recorded in every result's `config`:
  `min_pts_per_cluster = max(100, 3(d+1))` (B1; `--library-min-pts` restores 100);
  `max_clusters_est = 20,000` (B5; the buffers grow on demand); ART `store_trust_matrix=False`;
  `max_levels = 40`; work stealing off; `profile_timing=True`. AFINNI predicts with
  `coarse_factor = 50` (`--coarse-factor`, recorded as the run's `coarse_factor`; the library
  default is 10), since fewer candidates miss elongated trust regions and inflate `frac_outside`.
- **Matrix** (`sweep.py`): d ∈ {20, 40, 60, 80, 100, 120, 160, 200, 250, 300}. The tolerance ladder
  is 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001; where the leaf is exact it continues with 1e-4 … 1e-9
  to find the floor. At each d the ladder runs from coarse to fine and stops at the first
  tolerance not reached. For each (problem, variant), d stops rising when the *coarsest* tolerance
  fails with out-of-memory, the time limit, a kill or a crash; a run that only fails to converge
  does not stop d, so memory keeps being measured.
- **Per run** (`run_one.py`, one process each), the JSON record holds:
  - d, method, leaf model, tol and atol; status, reached and converged;
  - leaves, levels, the ART max depth, f evaluations, the effective batch size;
  - wall time per phase (`profiling_stats` `time_*`, with the oracle time), training and prediction
    time, per-level statistics;
  - **peak GPU memory three ways**: the exact peak of CuPy's pool through an allocator wrapper;
    NVML device memory sampled every 20 ms minus the baseline; NVML per-process memory where the
    driver reports it (not under WSL);
  - host peak RSS;
  - test errors on 10,000 fixed uniform points: max, RMS and 99th percentile relative to std f,
    the share within atol, and for AFINNI the share outside the trust regions;
  - ART leaf statuses (`max_depth`, `split_failed`, `insufficient_sampling`) and counters;
  - on failure: the mode (`gpu_oom`, `host_oom`, `time_limit`, `killed` for the kernel OOM killer,
    `crash`, `error`, or `not_converged`), the stage, the innermost `src/` frame, the message, and
    the last refinement level logged.
- **Robustness.** Each run is a fresh process; its own SIGALRM time limit records a clean
  `time_limit`, and the sweep kills it hard 120 s later if needed. The record is written before
  training starts, so a process killed outright still leaves its settings; the sweep then fills in
  the failure. Resumable: a rerun skips every final record; `--retry-failed` reruns errors, crashes
  and kills.

### 5.2 Predicted memory per node and limits on one GPU (`predict_limits.py`, formulas only)

- **charged** = `plan.bytes_per_point()` x the points a node occupies.
- **est** = a term-by-term reading of the allocations: point arrays, the `segment_cross` partials,
  the bins search's `4(n_bins − 1) p²` Gram stacks at the solve, and the soft search's `6 p²`.
- A node with **est** above the card's free memory (card minus 0.6 GB) is predicted to run out of
  memory.
- **nodes/batch** is the batch cap ÷ node size on a 24 GB card. Below 1, every batch is one node,
  which may still overrun.

| variant | d | p | points/node | charged GB | est GB | est/charged | nodes/batch (24 GB) | fits 24/48/80 GB | s/node at 1 TF |
|---|---|---|---|---|---|---|---|---|---|
| afinni | 100 | 101 | 606 | 0.0045 | 0.0051 | 1.14 | 3,400 | y/y/y | 4e-5 |
| afinni | 300 | 301 | 1,806 | 0.039 | 0.045 | 1.15 | 386 | y/y/y | 1e-3 |
| art_affine | 60 | 61 | 3,050 | 0.133 | 0.0125 | 0.09 | 115 | y/y/y | 1e-3 |
| art_affine | 100 | 101 | 5,050 | 0.363 | 0.0345 | 0.09 | 42 | y/y/y | 5e-3 |
| art_affine | 200 | 201 | 10,050 | 1.44 | 0.139 | 0.10 | 11 | y/y/y | 0.04 |
| art_affine | 300 | 301 | 15,050 | 3.22 | 0.319 | 0.10 | 4.7 | y/y/y | 0.13 |
| art_quad | 20 | 231 | 11,550 | 0.23 | 0.088 | 0.38 | 66 | y/y/y | 0.06 |
| art_quad | 40 | 861 | 43,050 | 2.08 | 1.70 | 0.82 | 7.3 | y/y/y | 3.0 |
| art_quad | 60 | 1,891 | 94,550 | 8.13 | 13.8 | 1.70 | 1.9 | y/y/y | 32 |
| art_quad | 80 | 3,321 | 166,050 | 22.1 | 67.1 | 3.04 | 0.69 | n/n/y | 174 |
| art_quad | 100 | 5,151 | 257,550 | 48.7 | 237 | 4.87 | 0.31 | n/n/n | 651 |
| art_quad_m4 | 100 | 5,151 | 20,604 | 12.7 | 23.1 | 1.82 | 1.2 | y (just)/y/y | 15 |
| art_quad_m4 | 120 | 7,381 | 29,524 | 25.6 | 54.9 | 2.15 | 0.60 | n/n/y | 45 |

The full table for d = 4 to 300 is written by `run_study.sh` to `results/predicted.{csv,md}`.

**Predicted out-of-memory boundary (d) on one GPU:**

| variant | 24 GB | 48 GB | 80 GB |
|---|---|---|---|
| art_quad (m = 50) | 60-80 | 60-80 | 80-100 |
| art_quad_m4 | about 100 | 100-120 | 120-160 |
| art_affine, afinni | not by per-node memory up to d = 300 | same | same |

For `art_affine` and `afinni` the limits are host RAM (B11), the startup leaf buffer if left at
the default (B5), the kNN gather (B7), and time.

**What to check against the measurements:** the ratio of the measured `gpu_pool_peak_bytes` to
`est_per_node` x nodes per batch, and whether quadratic runs at d = 60 to 80 run out of memory
although the batch charge says they fit (M2 predicts they do).

### 5.3 Commands

On the GPU machine, from the repository root, with `uv sync --extra gpu` done:

```bash
studies/high_dim/run_study.sh --dry-run          # list the matrix
studies/high_dim/run_study.sh                    # full sweep: AFINNI_DEVICE=gpu AFINNI_MPI=off by default
# first pass (recommended): fewer dims and a shorter limit, then widen; reruns resume
DIMS=20,60,100 TIME_LIMIT=900 studies/high_dim/run_study.sh
VARIANTS=art_affine_m10,art_quad_m4 studies/high_dim/run_study.sh     # the exploratory variants
EXTRA=--library-min-pts RESULTS=studies/high_dim/results_libminpts studies/high_dim/run_study.sh
studies/high_dim/run_study.sh --smoke            # CPU, d = 4 and 8, a few minutes: checks the pipeline
```

Pieces separately:

```bash
AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m studies.high_dim.predict_limits --out results/predicted.csv [--root-fit]
AFINNI_DEVICE=gpu AFINNI_MPI=off .venv/bin/python -m studies.high_dim.run_one --variant art_affine --problem rosenbrock --d 40 --tol 0.3 --out /tmp/r.json --time-limit 900
AFINNI_DEVICE=gpu AFINNI_MPI=off .venv/bin/python -m studies.high_dim.hit_and_run_mixing --dims 10,50,100,200 --taus 20,80,320,dd
.venv/bin/python -m studies.high_dim.collect --results studies/high_dim/results
```

Practical notes:

- Run on an otherwise idle card: AFINNI's trees depend on free VRAM (F11).
- On a shared login node or WSL, optionally cap host memory so that a host out-of-memory kills one
  run instead of the machine, e.g. prefix the command with `systemd-run --user --scope -p MemoryMax=…G`,
  or use the scheduler's memory limit. The scripts do not do this themselves.
- **Budget:** each (problem, variant, d) ends with at most one time-limited run, so the worst case
  is about 12 pairs x 10 d x the time limit: about a day at 1800 s. The first pass suggested above
  is bounded by about 12 x 3 x 15 min ≈ 9 h and is usually much shorter.

### 5.4 How to read the results

`collect.py` writes `runs.csv`, `finest.csv`, `memory.csv` and three plots into the results folder:

- **`finest_tol_vs_d.png`** (one panel per problem): the finest tolerance each variant reached
  against d; a cross means no tolerance of the ladder was reached.
  - `affine`: a flat line at the ridge floor confirms B6; a line rising with d quantifies it.
  - `quadratic`: Section 4.11 predicts affine leaves stuck at 0.3 from d ≥ 40.
  - `aniso_quadratic`: the separating case for A3.
  - `rosenbrock`: predicted unreachable at d ≥ 40.
- **`cost_vs_d.png`**: f evaluations, training time and peak GPU memory against d at tol = 0.3
  (`--ref-tol`). Dotted lines are the predicted `est_per_node`. The measured peak should sit at or
  above one node's estimate and track nodes/batch x estimate; a peak far above the estimate
  locates an uncharged temporary. Read `failure_where` for out-of-memory runs.
- **`error_vs_tol.png`**: the 99th-percentile test error and the share within tolerance against the
  requested tol, labelled by d. Points above the diagonal mean the acceptance test, computed on
  training samples, overstates accuracy (M13 predicts this for AFINNI). A share below about 0.99
  for ART means its samples did not represent the cell (B2).
- **`finest.csv`** gives, per (problem, variant, d), the next tolerance and why it failed:
  `time_limit` or `not_converged`: leaf explosion (the method); `gpu_oom` or `host_oom`: memory
  (implementation, compare with Section 5.2); ART `max_depth` or `split_failed` statuses: the tree
  hit `max_levels` or could not split; `insufficient_sampling` counts: the drift check fired.

### 5.5 Smoke tests run here (CPU, no GPU)

Unless noted the runs were tiny, at d ≤ 8; the cases at d = 60 and d = 100 are marked. Later runs
used a 6 GB, 4-core cap. All outputs are in a scratch directory, none in the repository.

| test | result |
|---|---|
| `run_one`, three variants, Rosenbrock d = 4, tol 0.1 | Records complete. AFINNI: 1,689 leaves, 1.2 s, 90% of test points within tolerance, 10% outside the trust regions. ART affine: 2,631 leaves, 17.7 s, 99.85% within. ART quadratic: 119 leaves, 5.2 s, 99.95% within |
| Time-limit path (ART affine, Rosenbrock d = 20, tol 0.01, 15 s) | `failure=time_limit`, stage `train`, where `segments.py:278 grouped_cross`, last level 7 |
| Host-OOM path (ART quadratic, quadratic f, d = 60 and 80, under `ulimit -v` 6 GB) | d = 60: one exact leaf (error 1e-10 x std, 94,551 evaluations, 8.5 s, RSS 4.5 GB). d = 80: `host_oom` at `local_models.py:327 features` (4 GB Phi), recorded; the sweep stopped raising d |
| `run_study.sh --smoke`: 4 problems x 3 variants x d ∈ {4, 8}, 58 runs | 5 min. 56 ok and 2 `time_limit`: ART affine and ART quadratic on Rosenbrock at d = 8, tol 0.3, 90 s limit. All exact cases reached 1e-6 with one ART leaf. AFINNI on Rosenbrock at d = 8, tol 0.3: converged, but only 54% within tolerance on the test set (M13). Resume verified: a rerun launched nothing. `collect.py` wrote the CSVs and plots, which were inspected. Rerun under the 6 GB, 4-core cap (`affine`, `aniso_quadratic`, `rosenbrock`; 44 runs): 4 min 23 s, the same leaf and evaluation counts, the same 2 time limits |
| `predict_limits.py`, d = 10 to 300, and `--root-fit`, d = 10 to 120 | the tables of Sections 5.2 and 4.11 |
| `hit_and_run_mixing.py`, d = 10 to 200 | the table of Section 2.2 |
| d = 100, `affine`, CPU (run before the caps were set) | ART: one leaf at tol 1e-4 and 1e-5 (error 3.56e-6 x std, 5,051 evaluations, 0.2 s). At 1e-6 it split an affine function and hit its 150 s limit at level 6. AFINNI: 64 leaves at 1e-4 (error 4.5e-6 x std, 3.9 s). B6 confirmed |
| `ruff check studies/high_dim` | clean |

---

## 6. Decisions for the user

1. **AFINNI under Section 3.** S2, S4, S5, S9 and S10 (scatter, repartition) move AFINNI's batch
   boundaries or rank assignment and so change AFINNI's trees. Options: (a) apply them to ART only
   for now (AFINNI keeps its current batching, charge and scatter); (b) accept each as a
   deliberate numerical change for AFINNI with new references; (c) do A1 first, after which every
   Section 3 item is bit-identical for both algorithms. S1, S3, S6, S7 (with unchanged batches), S8
   and the buffer gather do not raise this question.
2. **Suggested order within Section 3** (implement now):
   1. S1 (startup failure at high d) and S3 (prediction at high d): small, exact, unblock d ≈ 100.
   2. S2 for ART (quadratic partials charged or bounded; constraints per node, tested on WSL).
   3. S5 and S8 for ART: a CPU quantum (2x measured) and the reply-loop sleep.
   4. S10's parallel `f(X0)` and buffer gather; scatter by work for ART.
   5. S4, then S9 on top of it, if the study's `host_rss_peak_bytes` binds or multi-rank runs matter.
   6. S6 and S7, each decided by an Nsight profile on a CUDA machine; S12 and S11 when a workload
      needs them.
3. **Study settings.** Keep the scaled `min_pts_per_cluster`, `max_clusters_est = 20,000`,
   `store_trust_matrix=False` and `max_levels = 40`? Each is recorded per run; `--library-min-pts`
   measures the alternative.
4. **Study budget.** The time limit per run and which d to run first (suggested:
   `DIMS=20,60,100 TIME_LIMIT=900`).
5. **The deliberate numerical changes the study will motivate** (Section 4, each needing its own
   study and new regression references): the `N_cov` floor or principal Hessian directions (A2,
   A3); a scale-invariant ridge (A4; the convergence study's first decision); the hit-and-run
   scheme and m (A5, A6); node-keyed AFINNI streams (A1); acceptance on fresh points for AFINNI
   (A8).
6. **Open questions the existing runs do not answer:**
   - the GPU quantum, and whether small batches help or hurt there;
   - which split-stage temporaries cause the CPU cache effect of Section 1.2 (a profile of
     `splitter.split` at two batch sizes would say, and might allow a cheaper fix inside the
     component);
   - whether the premature retirement of F7 occurs in practice (an event-log check: a rank retires
     while another later receives a grant);
   - scaling beyond 4 ranks: all-to-all gossip and `argmax` victims should degrade with rank
     count, but only 4 ranks were measured;
   - how well "children cost what parents did" predicts load for the one-shot repartition (S10).
