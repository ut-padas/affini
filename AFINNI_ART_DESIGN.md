# AFINNI and ART: algorithms, comparison, and a design for combining them

This document describes two surrogate-construction algorithms developed in the group,
compares them, and proposes how ART's ideas can be brought into AFINNI without losing
AFINNI's batched GPU execution, out-of-core memory model, MPI distribution, or work
stealing. It ends with a minimal implementation plan that supports exactly these two
algorithms. A more general, PySPH-style property system is recorded in an appendix for
later.

| Part | Content |
|---|---|
| [I. AFINNI](#part-i-afinni) | Problem, mathematics, refinement loop, batched GPU execution, inference, MPI, work stealing, component API |
| [II. ART](#part-ii-art-adaptive-regression-trees) | Problem, polytope geometry, sampling, soft oblique and hinge splitters, builder, GPU cohorts, known weaknesses |
| [III. Comparison](#part-iii-afinni-and-art-compared) | Side-by-side comparison, and the observation that AFINNI's split is already a hyperplane |
| [IV. Integration](#part-iv-bringing-art-into-afinni) | Requirements, three approaches (1: global ids and parent links, 2: per-level buffers and path codes, 3: the combined plan), evaluation for GPU/MPI/work stealing, unbalanced splits, batched ART components, staged experiments |
| [V. Minimal plan](#part-v-minimal-implementation-plan-afinni-and-art-only) | Implementation status in Section 42. Two named plans (`afinni`, `region`) over one executor, fixed columns and wire format, the two pipelines, file-by-file changes, what is deferred, implementation order, resolved questions and what still needs measurement |
| [Appendix A](#appendix-a-deferred-a-pysph-style-property-system) | Deferred: PySPH-style property system, primitives and illustrative examples |

**Revision (Nov 10, 2026).** Revised after the technical review in
[AFINNI_ART_TECHNICAL_REVIEW.md](AFINNI_ART_TECHNICAL_REVIEW.md) ("the review"). The main
changes:
- a collision-free node identity;
- tie-safe geometric split predicates;
- a pre-fit top-up stage;
- corrected carried-path overheads;
- the majorization argument for frozen-model line search;
- a bounded hard threshold search before any soft optimization;
- a revised approach 3;
- a staged experiment plan;
- open questions marked resolved or still open;
- Part V reduced to a minimal plan for the two algorithms, with the property system
  moved to Appendix A.

**Second revision (Nov 11, 2026)**, after the review's §20–29 (its review of the first
revision and its answers to Q-A to Q-I):
- the AFINNI pipeline in Section 37 now follows the code's actual order
  (preliminary acceptance, then `Q`, finalize and re-test);
- region mode's differences are listed as named stage methods over one executor, not
  "two branch points";
- RNG streams are indexed by purpose, attempt, chain and step, not just the node key;
- the point `role` column is split into a training/validation field and a provenance
  flag, and fits see a training-only view;
- centroid values are required when any component needs them;
- one canonical stored predicate, an explicit MPI wire format, a region memory
  estimator, and a restricted first-version work-item contract;
- B1 redefined as one tree evaluated by both predictors;
- Q-A to Q-I resolved, with the answers in Section 41 and in the sections they affect.

**Third revision (Nov 11, 2026):** the whole plan is implemented on branch
`region-mode`; Section 42 records what was built, the CPU/MPI verification, the ART
comparison, the GPU test plan and what is deliberately deferred.

Review sections are cited as "review §n".

**Status labels.** *Implemented* describes code on the `modularity` branch of `affini`
(Nov 2026) or `main` of `Adaptive-Regression-Trees-ART-`. *Manuscript* refers to the
AFINNI paper draft as summarized in `review_artifacts/manuscript_reference.md`.
*Proposed* marks design that does not exist yet.

**Conventions.** Points are rows: `X` has shape `(N, d)`, values `F` have shape `(N, fd)`,
gradients `G` have shape `(N, fd, d)`. `K` is a number of clusters, `c_k` a cluster
centroid, `n_k` its number of points. The manuscript writes points as columns; its
`XXᵀ` is the code's sum of row outer products.

---

# Part I. AFINNI

## 1. Problem and goals

AFINNI (Adaptive Affine Nearest Neighbor Interpolation) builds a surrogate
`f̂ : Ω ⊂ ℝᵈ → ℝᶠᵈ` of a smooth, noise-free, expensive function `f` that is accessed
through an oracle. It has two simultaneous goals:

1. **Enrich** the sample where the initial data under-resolve `f`, by adding oracle
   evaluations only where a local model fails.
2. **Compress** all evaluated samples into a much smaller set of local models (leaves)
   that answer queries quickly.

The manuscript states the goal as pointwise relative accuracy,
`|f(x) − f̂(x)| ≤ ε |f(x)|`. The implemented acceptance test is weaker than that.
Its relative Linf error is normalized by the cluster's *largest* `|F|`, not each point's
own `|F|` (Section 4). So an accepted cluster can still contain points with large
pointwise relative error. Example (review §3.8): `F = [1, 1000]`, `F̂ = [11, 1000]`
gives a cluster error of 0.01, but a pointwise error of 10 at the first point. Before
comparing AFINNI and ART, specify which target is meant: integrated error,
cluster-normalized error, or the pointwise tolerance-violation rate. The method assumes
a Lipschitz-continuous Hessian; discontinuities, noise and extrapolation are out of
scope.

## 2. Overview

```text
X0 (N0 points) ──f──► F0
   │ initial clustering (k-means, K0 roots)
   ▼
queue of clusters ──► level loop: for each batch of whole clusters
                         fit local model ─► acceptance test
                           accepted ─► trust ellipsoid Q, leaf columns ─► output buffer
                           rejected ─► augment (sample + evaluate) ─► split in two
                                         children ─► next level's queue
leaves {c_k, f(c_k), model params, Q_k}  ─► inference: kNN + trust-region rerank
```

The forest (one tree per initial cluster) is conceptual: interior nodes are never
stored; only the accepted leaves are (Section 9).

## 3. Local models (`src/local_models.py`)

Each leaf `k` carries a model `f̂_k`. All models are fitted for every cluster of a batch
at once (vectorized over clusters, no Python loops).

| Model (registry name) | Form | Data needed | Used by modes |
|---|---|---|---|
| `constant` | `f̂(x) = f(c)` | — | `NEAREST_NEIGHBOR` |
| `taylor` | `f̂_j(x) = f_j(c) + ∇f_j(c)ᵀ(x−c)` | gradient at centroid | `TAYLOR_*` |
| `taylor`, `second_order=True` | `+ ½ (x−c)ᵀ H_j (x−c)` | gradient at every point | `TAYLOR_LSQ_HESS` + `store_hess` |
| `affine_lsq` | `f̂(x) = A x + b` | — | `AFFINE_*` (default) |

**Affine least squares.** With centred rows `Δx_i = x_i − c` and the mean output `ȳ`:

$$
S_{xx}=\sum_i \Delta x_i\Delta x_i^T,\qquad S_{xy}=\sum_i \Delta x_i (y_i-\bar y)^T,
$$

$$
(S_{xx}+\beta I)\,A^T = S_{xy},\qquad \beta=\max(10^{-8}\operatorname{tr}S_{xx},\,10^{-12}),\qquad b=\bar y-A c .
$$

`S_xx` comes from a fused segmented kernel (`run_fused_stats`), `S_xy` from
`run_fused_cross_cov`, and the batched solve from a custom in-place Cholesky on the GPU
(`solve_regularized_covariance_inplace`). For an exact unregularized fit, its training
residual is no larger than that of the centroid Taylor model; this is an empirical
least-squares statement, not a held-out bound.

**Second-order Taylor (LSQ Hessian).** For each output component `j`, the symmetric
`H_j` minimizing `‖H ΔX − ΔG_j‖_F` (ΔG = gradient residuals about `g(c)`) solves the
Lyapunov-type equation

$$
H\Sigma + \Sigma H = D + D^T,\qquad \Sigma=\Delta X\Delta X^T,\ D=\Delta X\,\Delta G_j^T .
$$

With `Σ = UΛUᵀ` and `B = Uᵀ(D + Dᵀ)U`, the solution is `H = U H̃ Uᵀ` with
`H̃_ij = B_ij / (λ_i + λ_j)` (`lsq_hessians`). It is computed only for accepted clusters
(`finalize`), and the deployed model is re-tested: clusters that fail with the quadratic
term are sent back to be split.

## 4. Acceptance (`src/acceptance.py`)

For cluster `k` and output component `j`, with residuals `e_ij = F_ij − F̂_ij` over the
cluster's own points:

| Metric | Absolute error `abs_kj` | Relative error `rel_kj` |
|---|---|---|
| `Linf` (default) | `max_i |e_ij|` | `abs_kj / max_i |F_ij|` |
| `L2` | `sqrt(Σ_i e_ij² / n_k)` (RMS) | `‖e_·j‖₂ / ‖F_·j‖₂` |

A cluster is accepted when **every** component passes either test:

$$
\forall j:\quad rel_{kj} < rtol_j \ \ \text{or}\ \ abs_{kj} < atol_j .
$$

`atol` and `rtol` are each a scalar, a per-component vector, or `None` (that test never
accepts). The errors `E_cluster_abs`, `E_cluster_rel` (shape `(K, fd)`) are stored with
each leaf. The test is on training points, so it is optimistic.

## 5. Splitting: curvature-informed spectral bisection (`src/splitting.py`)

A rejected cluster is split in two. Every strategy is a two-means relaxation in a metric
`M ≈ |H|^{1/2}`, where `|H| = U|D|Uᵀ` is the spectral absolute value of the Hessian
(eigenvalues replaced by their magnitudes):

```text
C   = Σ_i Δx_i Δx_iᵀ                    # cluster scatter (d × d)
CovZ = M C M                            # scatter of mapped points z_i = M Δx_i
v   = leading eigenvector of CovZ       # split direction in mapped space
s_i = vᵀ M (x_i − c)                    # projection score per point
sort points of each cluster by s_i; cut:
    median: n//2 | n − n//2   (default, balanced)
    sign:   s_i < 0 | s_i ≥ 0 (manuscript Algorithm 2; falls back to median if one side is empty)
```

Motivation (manuscript): for near-quadratic `f` the Taylor residual about `c` is
`½ (x−c)ᵀH(x−c)`, and `|dxᵀH dx| ≤ dxᵀ|H|dx`. So `½ (x−c)ᵀ|H|(x−c)` is an upper
envelope of the residual's magnitude, equal to it only when `H` is semidefinite. Two-means
in the `|H|^{1/2}` metric reduces that envelope for the two children. For indefinite `H`
the envelope can be loose: on `x₁x₂`, `|H| = I` and the metric has no preferred direction
(review §27.1). With `M = I` this is ordinary PCA bisection.

| Strategy | How `M` is estimated | Needs |
|---|---|---|
| `covariance` | `M = I` | — |
| `covariance_hessian`, `gradient="local_lsq"` (default for `AFFINE_COV_HESS`) | Fit a gradient by LSQ on the `max(2d, min_pts/4)` points nearest `c`; take the `max(1, min_pts/3)` points with the smallest Taylor residual (they trace a level set `E_δ`, approximately an ellipsoid of shape `|H|⁻¹`); `|H| ∝ (scatter of E_δ)⁻¹`; `M = |H|^{1/2}` via eigendecomposition with eigenvalue floor | — |
| `covariance_hessian`, `gradient="centroid"` | Same, with the exact gradient at `c` | `g(c)` |
| `exact_hessian` | `M = |H(c)|^{1/2}` from `hessian_fn` | `hessian_fn` |
| `gradient_lsq_hessian` | LSQ Hessian from gradient residuals of the output component with the largest relative Taylor error | `G` at every point |

The split is validated by `_check_split`: the returned order must be a permutation,
keep every row inside its parent cluster, and produce two non-negative child sizes per
cluster that add up to the parent's. Empty children are dropped.

**Note for Part IV.** Because `M` is symmetric, `s_i = (Mv)ᵀ(x_i − c)`. A cut at score
`τ` is therefore the hyperplane `wᵀx = z` with `w = Mv/‖Mv‖` and
`z = wᵀc + τ/‖Mv‖`. AFINNI's split is an oblique hyperplane split that is not
recorded. One caveat (review §3.1): the median cut is by *count*. When the scores
on either side of the median are equal, no plane separates the two children. Four
identical points split `[2, 2]` is the extreme case. So a recorded plane needs a
threshold chosen at a real gap between distinct scores (Section 31.7).

The current projection computes `M (x − c)` for every point and then dots it with `v`,
which costs `O(N d²)`. Computing `a = Mv` once per cluster and then `aᵀ(x − c)` costs
`O(K d² + N d)` (review §10.9). This changes floating-point results, so it belongs in
the new geometric mode only.

## 6. Augmentation (`src/sampling.py`)

Rejected clusters that are too small to estimate a model, a Hessian and a split are
topped up **before** splitting:

$$
n_k^{target} = \begin{cases}\max(2n_k,\ 2\,\text{min\_pts}) & n_k < 2\,\text{min\_pts}\\ n_k & \text{otherwise}\end{cases}
$$

so that each median child still has `min_pts` points. New points are placed by a
`Sampler`; the default `convex_combination` draws two distinct points of the cluster and
`α ~ U(0,1)`: `x_new = α x_i + (1−α) x_j`. Every new point costs an oracle evaluation.
Properties: new points stay inside the cluster's convex hull, cannot add a missing
direction, concentrate towards the interior, and never fill the gap between the hull and
the region the leaf is later asked to answer for (a known cause of out-of-region queries).

Order matters: **fit and test first, then augment rejected clusters, then split.** A
small cluster can be accepted on few points.

## 7. Trust ellipsoids (`src/trust_regions.py`)

For each accepted cluster, with `Σ = Σ_i Δx_iΔx_iᵀ`:

$$
\Sigma = U\Lambda U^T,\quad \Lambda \leftarrow \max(\Lambda, q_{reg}),\quad
\alpha=\max_i \Delta x_i^T\Sigma^{-1}\Delta x_i,\quad Q = \alpha^{-1}\Sigma^{-1}.
$$

The region `{x : (x−c)ᵀQ(x−c) ≤ 1}` contains every observed point of the cluster. It is
an empirical coverage region, not an error certificate.

## 8. The training loop (`src/adaptive_clustering.py`)

```text
rank 0: F0 = f(X0); labels = kmeans(X0, K0); sort by label; scatter clusters to ranks
each rank (_run_local_refinement):
  level = 0
  loop:
    if queue empty: work-stealing idle step (retire or receive work); continue
    for each batch b (≈ batch_size points, cut at cluster boundaries):
        serve pending steal requests (may give away the unprocessed tail)
        copy batch to device
        μ_k = segment means; f(μ_k) [, ∇f(μ_k)]
        params = model.fit(batch); F̂ = model.evaluate(params, X)
        is_good, E_abs, E_rel = acceptance(F, F̂)
        accepted: Q = trust_region.shapes(); final = model.finalize(); re-test if final adds terms
                  append leaf columns to host ColumnBuffer
        rejected: targets = augmentation_targets(n_k); augment (sampler + oracle)
                  order, child_sizes = splitter.split(); check
                  append children (depth + 1) to host ClusterQueueBuilder
    level boundary: children with depth > max_levels are counted unresolved and dropped;
                    publish load; serve steal requests; queue = children
gather leaves + profiles on rank 0; broadcast the model to every rank
```

**Surrogate modes** are named (local model, split strategy) presets:

| Mode | Local model | Split |
|---|---|---|
| `NEAREST_NEIGHBOR` | constant | covariance |
| `AFFINE_COV_NO_HESS` | affine LSQ | covariance |
| `AFFINE_COV_HESS` (default) | affine LSQ | covariance-Hessian, local LSQ gradient |
| `AFFINE_COV_HESS_EXACT_GRAD` | affine LSQ | covariance-Hessian, exact centroid gradient |
| `AFFINE_EXACT_HESS` | affine LSQ | exact Hessian |
| `TAYLOR_COV_NO_HESS` | Taylor | covariance |
| `TAYLOR_COV_HESS` | Taylor | covariance-Hessian, centroid gradient |
| `TAYLOR_EXACT_HESS` | Taylor | exact Hessian |
| `TAYLOR_LSQ_HESS` | Taylor (optionally second order) | gradient-LSQ Hessian |

**Key defaults** (`ClusteringConfig`): Linf metric, `num_initial_clusters=100`,
`min_pts_per_cluster=100`, `max_levels=50`, `batch_size=10M` points, `q_reg=1e-6`,
`seed=0`, work stealing on. Dimension heuristics used in experiments:
`N0 ≈ 50,000·1.3ᵈ`, `K0 ≈ 25·1.3ᵈ`, `min_pts ≈ 100·1.3ᵈ` (`src/heuristic_parameters.py`).

**Termination and status.** Depth travels with each cluster (`ClusterBatch.depth`), so
the `max_levels` limit does not depend on which rank processes a cluster. Clusters that
exceed it leave their region without leaves; the run reports `converged=False` and the
unresolved cluster/point counts. If no cluster at all is accepted, `train()` raises and
keeps the previous model.

## 9. Batched GPU execution

The engine's performance comes from treating **a batch of whole clusters** as one work
item, stored contiguously:

| Array | Shape | Meaning |
|---|---|---|
| `X`, `F`, `G` | `(N, d)`, `(N, fd)`, `(N, fd, d)` | points, values, optional per-point gradients |
| `sizes` | `(K,)` | points per cluster |
| `starts` | `(K,)` | first row of each cluster (`cumsum` of sizes) |
| `labels` | `(N,)` | cluster index of each row (`repeat(arange(K), sizes)`) |
| `depth` | `(K,)` | tree depth of each cluster (host only) |

Every operation is either a **segmented reduction** over rows (`reduceat`, segmented
max, `segment_sum` via a CSR indicator on CPU, fused statistics kernels on GPU) or a
**batched small dense operation** over clusters (`(K, d, d)` eigensolves, Cholesky,
`einsum`). No Python loop runs over clusters or points.

**Memory model (out-of-core with respect to the device).**

- A level's queue lives in host RAM (`ClusterBatch`); only one batch at a time is copied
  to the device. Children are copied back to a host `ClusterQueueBuilder` (written in
  place, capacity `2×` the level's points, grown by doubling, with an explicit
  host-memory check that raises `MemoryError` with advice instead of being OOM-killed).
- Accepted leaves go to a host `ColumnBuffer` declared from the model's columns.
- On GPU the batch size is capped from free VRAM (65% of free memory divided by an
  estimated peak of `(d+fd+1)·8 + 2·(3d + fd·d + 3fd + 8)·8` bytes per point), clipped
  to `[5·10⁴, 2·10⁷]`. The CuPy memory pool is released before the reading and between
  allocation bursts.
- The rejected half of a batch is released before augmentation, which together with the
  split is the peak.

Arithmetic intensity is low (manuscript: `≈ d/8` FLOP/byte for statistics), so
transfers and memory traffic, not FLOPs, are the main costs. Each level makes a
host→device→host round trip; this bounds device memory at the cost of PCIe/NVLink traffic.

**Ordering invariant (used in Part IV).** Within a level the queue order is preserved:
batches are processed in order, `xp.nonzero` returns rejected clusters in ascending
order, each parent appends its two children consecutively (low-score child first), and
`build()`/`select()` drop clusters without reordering. Without work stealing, the queue
at every level is therefore sorted by (root, path from the root).

## 10. Inference (`AFINNISurrogate.predict`)

```text
for each query x (batched):
    candidates = k_nn · coarse_factor nearest centroids (Euclidean, exact kNN in src/knn.py)
    q_j = (x − c_j)ᵀ Q_j (x − c_j) for each candidate          # trust-region distance
    keep the k_nn smallest q_j
    weights w_j = 1 / (q_j² + 1e-12)        # Shepard, p = 4 on distance √q
    prediction = Σ w_j f̂_j(x) / Σ w_j
    outside flag: min_j q_j > 1
```

With `k_nn=1` this is the manuscript's one-leaf rule. Euclidean preselection can miss
the leaf that is best in its own metric; `coarse_factor` controls recall.
`strict_trust_region` zeroes weights of candidates with `q_j > 1`. `src/knn.py`
implements an exact search with a float32 candidate pass, error-bounded verification,
and a float64 fallback (KD-tree on CPU for low `d`, a fused tiled top-k CUDA kernel on
GPU).

## 11. MPI execution

- **Scatter.** Rank 0 evaluates `f(X0)`, computes the initial labels, sorts points by
  label, splits the cluster list into `P` contiguous chunks of equal **cluster** count
  (`np.array_split` of the counts), and sends each chunk (`SCATTER_TAG=1000`, payload on
  `+1..+5`).
- **Refinement** runs independently per rank, with work stealing (Section 12).
- **Gather.** Every rank sends its leaves and profile to rank 0 (`LOCAL_RESULT_TAG=2000`);
  rank 0 concatenates them, consolidates statistics, and `train()` broadcasts the model so
  every rank can predict. `predict()` scatters rank 0's queries, predicts locally, gathers.
- **Failure policy** (`run_collective`). In one process an exception propagates. Under MPI
  an exception on one rank could leave others blocked in a collective, so the failing rank
  logs rank, phase and traceback and calls `MPI_Abort`. Nested collective calls (e.g. a
  `predict()` inside an oracle) clean up only their own profiler scope.
- **Profiling.** Per-phase timings, an optional event timeline, and per-rank JSON-lines
  streams that `python -m src.monitor` tails live without MPI traffic.

## 12. Work stealing (`src/work_stealing.py`)

Static partitioning is insufficient because refinement work is adaptive: some regions
generate many more levels and points than others.

**State.** Each rank keeps its load (queued points) and a cached vector of every rank's
load, updated by gossip: 8-byte blocking sends on `TAG_STATE_BROADCAST=101` whenever the
load changes (assumes eager delivery). An initial exchange makes every rank learn every
initial load before any decision.

**Victim** (between batches and at level boundaries, `serve_requests`). For every pending
request (`TAG_STEAL_REQ=102`): if more than one unprocessed cluster remains and the
unprocessed points exceed `MIN_LOAD_THRESHOLD=10⁶`, grant
`max(STEAL_SIZE=5·10⁵, open/2)` points' worth of whole clusters from the **tail** of the
queue, always keeping at least one; otherwise deny (`TAG_STEAL_DENY=104`) with a reason.
Clusters before the current batch start are never given away. A grant is a metadata
message on `TAG_STEAL_GRANT=110` followed by buffers `X, F, [G], sizes, depth` on tags
`111..115`.

**Thief** (`idle_step`). Publish load 0; deny any requests it receives; if the global
load is 0, retire; else ask the most loaded rank if its load exceeds the threshold, and
wait for the **explicit** reply (a zero-load update is not a reply), while denying
crossed requests from other thieves. A grant is kept and processed; on denial, sleep
`SLEEP_INTERVAL=0.05 s` and retry.

**Retirement.** A rank with no work and no outstanding request enters a non-blocking
barrier (`Ibarrier`) and keeps answering requests until every rank has entered it. It
then `allgather`s the number of load broadcasts each rank sent and receives exactly the
ones still buffered, so no message is pending at `MPI_Finalize`.

**Invariants** (each was a real bug at some point): exactly one reply per request; every
grant is consumed; global completion is never inferred from a stale load snapshot; a
migrated cluster carries all its state (points, values, gradients, depth); protocol
ownership stays in the execution layer, not in scientific components. Any new per-cluster
state (Part IV) must be added to the migration payload. That is why Part V puts every
column in one internal table, which the payload code iterates over.

## 13. Component API (`src/config.py`, `src/registry.py`)

The five swappable components each have a base class and a registry:

| Component | Base / registry | Contract |
|---|---|---|
| local model | `LocalModel` / `LOCAL_MODELS` | `fit`, optional `finalize`, `evaluate` (training `(N,)` ids, inference `(n,k)` ids), `columns(d, fd)` |
| split strategy | `SplitStrategy` / `SPLIT_STRATEGIES` | `metric` → `(M, MCM)` or override `split` → `(order, child_sizes)`; optional `child_sizes` cut rule |
| sampler | `Sampler` / `SAMPLERS` | `sample(ClusterPoints, n_new, rng)` → new points grouped by cluster |
| acceptance | `AcceptanceCriterion` / `ACCEPTANCE_CRITERIA` | `__call__(F, F̂, starts, sizes)` → `(is_good, E_abs, E_rel)` |
| trust region | `TrustRegion` / `TRUST_REGIONS` | `shapes(X, centroids, labels, counts)` → `Q` |

A component is given as an instance, a registered name, or `(name, settings)`.
Components declare the data they need (`needs_point_grad`, `needs_centroid_grad`,
`needs_hessian_fn`), and the loop evaluates and carries only what is needed. Under MPI
every rank builds its components from the same spec. Saved models store the local
model's name and settings and every declared column (versioned `.npz`, no pickling).

What the API does **not** yet cover: the sample budget (`augmentation_targets` is fixed),
the predictor (kNN + trust region is hard-coded), per-cluster state beyond the fixed
columns, and any notion of a domain or of tree structure.

## 14. Accuracy, complexity and limitations

- First-order model, quadratic-in-spacing error: with leaf size `h ~ N_c^{-1/d}`, the
  manuscript argues `‖f − f̂‖_{L²} = O(N_c^{-2/d})`; reported slopes roughly match at
  low `d` for in-region error. This needs representative samples and controlled coverage.
- Out-of-region rate `p_o` grows with tightening tolerance and dimension (e.g. Lorenz96:
  0.01% at `d=5` to 34.7% at `d=80`), driven by hull-limited augmentation and Euclidean
  candidate preselection.
- Construction cost is dominated by memory traffic; queues can expand to hundreds of times
  `N0` points at intermediate levels (d=10 Rosenbrock: 0.69M to 184M points).
- Open: nonconvex/high-dimensional Hessian estimation, higher-order models, better
  within-region sampling, sign vs median cut semantics.

---

# Part II. ART (Adaptive Regression Trees)

Source: `github-repos/Adaptive-Regression-Trees-ART-` (`art/`), and its handoff
`art/agents.md`.

## 15. Problem

ART approximates an expensive **scalar** oracle `f : D → ℝ` over a bounded domain `D`
(normally with the uniform measure `ν`) by an oblique regression tree. The resource to
minimize is the number of oracle queries. The ideal split of a node region `R` solves

$$
\min_{w,z}\ \Big[\,P_R(A)\inf_{\phi_L\in\mathcal M}\mathbb E\big[(f-\phi_L)^2\mid A\big]
+P_R(A^c)\inf_{\phi_R\in\mathcal M}\mathbb E\big[(f-\phi_R)^2\mid A^c\big]\Big],
\quad A=\{x\in R: w^Tx<z\},\ \|w\|_2=1,
$$

which ART replaces by samples drawn inside `R` and a smoothed empirical problem.
Construction is greedy (no global tree optimization).

## 16. Geometry: polytope regions (`art/domain.py`)

Every node owns a convex polytope `R = {x : A x ≤ b}`.

- The root box `[l, u]` is `A = [I; −I]`, `b = [u; −l]` (2d rows). Arbitrary convex
  polytopes are also accepted as roots.
- Splitting a node by `(w, z)` appends one row to a copy of its constraints:
  left child `wᵀx ≤ z`, right child `−wᵀx ≤ −z` (`split_region`).
- A depth-`k` node has `2d + k` rows: the box plus one signed hyperplane per ancestor.
  Rows are never simplified, so redundant constraints accumulate.
- Leaves form an exact partition of `D`. The interiors are disjoint and cover `D`. The
  closed children share their boundary plane, a set of zero measure. Routing makes
  ownership unique by sending `wᵀx − z ≥ 0` to the right (review §3.4). Sampling can
  treat regions as closed sets, but routing needs this half-open convention.

Both children's polytopes are created when the parent is split, independently of whether
either child will later converge. The polytope's role is **sampling inside the node**;
prediction does not use it.

## 17. The builder (`art/builder.py`)

```text
BUILD(R, inherited samples, depth):
    top up inherited samples to n_node with hit-and-run inside R; query f only at new points
    prepare the design matrix once (polynomial features)
    fit a fresh model; compute the stopping metric on the same samples
    if error ≤ threshold:  leaf("tolerance_met")
    if depth == max_depth: leaf("max_depth")
    choose temperature; run the splitter on the node samples; retry selected failures
    if no valid split or (optional) MSE gain too small: diagnostic leaf
    R_L, R_R = split_region(R, w, z)
    pass each child the parent samples on its side; BUILD both children
```

- **Sample budget**: `n_node = sample_multiplier · p_eff` (default multiplier 50), with
  `p_eff = d+1` (affine) or `C(d+m, m)` (degree-`m` polynomial). If a child inherits more
  than `n_node` points they are subsampled; otherwise it is topped up.
- **Stopping metrics**: MSE, relative L2, median or max pointwise relative error,
  computed on training samples (optimistic). Optional holdout fraction for the
  convergence fit.
- **Split acceptance**: minimum side size `max(min_side_points, ⌈min_side_fraction·n⌉)`
  on the hard partition; optional MSE gain checks
  (`gain_abs = MSE_parent − MSE_split`, `gain_rel = gain_abs / MSE_parent`); finite-result
  and numerical checks. Failed attempts may be retried with a new initialization, reusing
  the same samples (no new oracle calls).
- **Traversal**: DFS (default; the left subtree is finished before the right one) or BFS
  (a FIFO queue; cohorts of up to `node_batch_size=20` ready nodes; cohorts may cross
  levels). Each node is judged independently when it is processed.
- Every node has a path id (`root/L/R/...`); the full tree is stored.

## 18. Sampling inside a polytope (`art/sampling.py`)

**Hit-and-run.** From a feasible `x`, draw a direction `u = g/‖g‖`, `g ~ N(0, I)`;
compute the feasible chord

$$
I(x,u)=\{t: A(x+tu)\le b\} = \Big[\max_{(Au)_r<0}\frac{(b-Ax)_r}{(Au)_r},\ \min_{(Au)_r>0}\frac{(b-Ax)_r}{(Au)_r}\Big],
$$

draw `t ~ U(I)` and move to `x + t u`. Uniform measure on `R` is invariant. Each step
costs `O(m d)` for `m` constraints. Burn-in and fixed thinning (preset: 20) are
configurable.

- **Inheritance.** A child keeps the parent's samples on its side (no re-query) and seeds
  its chain at a random inherited point with zero burn-in. The argument is that a uniform
  parent sample conditioned on the child is uniform in the child. That holds only for
  ideal parent sampling **and a child region fixed independently of the samples**. ART
  chooses the plane from those same samples and their values. Conditional on the chosen
  plane, the inherited points are therefore not a uniform sample of the child. They are
  good feasible seeds and reusable oracle data, but they don't justify zero burn-in
  (review §7.1).
- **Covariance-shaped directions** (`isotropic_sampling=True`): a pilot chain of `50·d`
  steps estimates `Σ = QΛQᵀ`, eigenvalues are floored at `η·tr(Λ)/d`, and directions are
  `u ∝ QΛ^{1/2}g`, which improves mixing in elongated cells.
- **ACF thinning estimator**: implemented, experimental, not integrated.
- Root boxes are sampled exactly; batched and Numba-parallel samplers exist.

## 19. Local models (`art/models.py`)

Affine and polynomial ridge models solve a weighted, trace-scaled ridge problem

$$
G=\Phi^TW\Phi,\qquad \lambda = \text{ridge}\cdot\operatorname{tr}(G)/p,\qquad \beta=(G+\lambda I)^{-1}\Phi^TWy,
$$

with weights floored at `1e-12`. Solvers: `normal` (Cholesky), `qr`, `svd`, and `auto`
(Cholesky if LAPACK `dpocon` reciprocal condition ≥ `1e-10`, else pivoted QR on the
augmented system). Polynomial features are built once per node (`PreparedDesign`) and
reused by every weighted refit. Kernel ridge exists but lacks weighted fitting, so it is
not usable with the soft splitter.

## 20. Soft oblique splitter (`art/splitters.py`, `art/objectives.py`)

**Relaxation.** For `θ = (w, z)` with `‖w‖ = 1` and temperature `T > 0`, the soft
right-membership of sample `i` is

$$
\pi_i(\theta)=\sigma\!\left(\frac{w^Tx_i-z}{T}\right).
$$

At fixed `θ`, the left and right models are weighted least-squares fits with weights
`1 − π_i` and `π_i` (inner problem). The profiled outer objective is

$$
J_T(\theta)=\frac1n\sum_i\Big[(1-\pi_i)\,r_{L,i}^2+\pi_i\,r_{R,i}^2\Big],\qquad r_{\cdot,i}=y_i-\phi_\cdot(x_i).
$$

This is **variable projection**: the regression coefficients are eliminated by solving
the inner problem exactly, and only `θ` is optimized.

**Gradient.** By the envelope theorem (exact unregularized inner minimizers contribute
no derivative term):

$$
a_i=\frac{(r_{R,i}^2-r_{L,i}^2)\,\pi_i(1-\pi_i)}{n},\qquad
\nabla_wJ=\frac{X^Ta}{T},\qquad \nabla_zJ=-\frac{\sum_ia_i}{T}.
$$

**Unit-normal constraint.** Without `‖w‖ = 1`, rescaling `θ` would change the effective
temperature. The gradient is projected onto the sphere's tangent space,
`(I − wwᵀ)∇_wJ` (the Riemannian gradient on `S^{d−1} × ℝ`), and a step is retracted by
renormalizing `w`.

**Line search.** Armijo backtracking on the retraction,
`J(R_θ(αp)) ≤ J(θ) + cα⟨∇J, p⟩`, with an adaptive initial step (grow after zero
backtracks, shrink after failures, clipped to `[α_min, α_max]`). Stop when
`‖∇_proj J‖ ≤ atol + rtol‖∇_proj J(θ₀)‖`, at `max_iters`, or at `α_min`. By default the
line search **freezes** the models (`refit_during_line_search=False`). The candidate's
`π` is recomputed, but the residuals are reused, and the models are refit only after a
step is accepted.

**Frozen line search is a majorization step (review §8.3).** Write the frozen
loss as `F(θ, B_t)`, where `B_t` are the current coefficients. Suppose `B_t` exactly
minimizes the *same* objective at `θ_t`. Then for every θ,

$$
J(\theta)=\min_B F(\theta,B)\le F(\theta,B_t),\qquad J(\theta_t)=F(\theta_t,B_t),
$$

so the frozen loss is an upper bound that touches `J` at the current iterate. A frozen
Armijo step `F(θ_{t+1}, B_t) ≤ J(θ_t) + cα⟨grad J, p⟩` therefore gives
`J(θ_{t+1}) ≤ J(θ_t) + cα⟨grad J, p⟩` after the refit. Freezing does not lose
the descent property. The *inconsistencies* in Section 24 (a gate-dependent ridge,
independent weight floors, a clipped logit) and inexact inner solves are therefore the
possible causes of the loss increases observed in ART. Freezing itself is not. The
theorem does not say which cause produced a given historical increase; that needs its
trace. With a consistent objective, line-search
trials need no refits, and each accepted step costs exactly one pair of refits.

**Initialization and hardening.** A random unit `w` and `z` at the median projection
(an approximately balanced start); optional restarts keep the best valid result. The gate
is then hardened at `wᵀx = z`, side sizes are checked, and fresh unweighted child models
are fitted.

**Temperature.** `T` has units of distance (because `‖w‖ = 1`):
`T = c · median nearest-neighbour distance` (default) or
`c · median pairwise distance · n^{−1/d}`, on at most 512 points. Strategies: fixed `c`,
the splitter's absolute `T`, tune `c` at the root, or tune at every node (validation MSE
over `c ∈ {1e-4 … 1}`).

## 21. Hinge splitter (HRT)

Affine leaves only. Fit `h(x) = max(θ₁ᵀx̃, θ₂ᵀx̃)` (or `min`, or both and keep the better)
with `x̃ = [x; 1]`. Initialize with a random median hyperplane; iterate: assign points to
the active piece, reject partitions below the side size, fit each active set, update
`θ_j ← θ_j + μ(θ_j^{OLS} − θ_j)`. The boundary is `(θ₁ − θ₂)ᵀx̃ = 0`, normalized to
`(w, z)`. It is a fixed-point heuristic with no descent or convergence guarantee.

## 22. GPU execution in ART

- `hybrid_gram`: weighted Grams/right-hand sides on the GPU, solves on the CPU.
- `cupy_qr`, `cupy_cholesky`: one node's soft optimization resident on the GPU.
- `cupy_batched_cholesky` (BFS only): a workspace sized for `node_batch_size` nodes,
  allocated once per build; each cohort's packed `X, y` are uploaded, features built on
  the GPU, and projected gradient descent runs with independent per-node state, active-
  slot compaction each iteration (`batch_reorder=True`), and GPU-side initialization.
  Sampling, oracle queries, node fits and stopping decisions remain on the CPU. The
  path assumes node datasets of equal size (ART's fixed budget). Its fixed slots and
  active-pointer compaction are worth borrowing; it does not replace a distributed
  queue (review §2.2).

## 23. Inference and persistence

Route from the root: at each `SplitNode`, go right if `wᵀx − z ≥ 0`; evaluate the single
leaf's model. Batched by masking query indices per node. Trees are saved with joblib
(tree, models, diagnostics, run configuration). There is no domain check: a query
outside `D` is routed to some leaf and silently extrapolated (review §2.3). A routing
function that returns a value everywhere is not an extrapolation guarantee.

## 24. Known weaknesses (from `agents.md`)

1. Objective and gradient are inconsistent: inner fits are ridge-regularized with a
   weight-dependent `λ` and a weight floor, but the outer loss and envelope gradient
   ignore both; the sigmoid clips logits at ±50 while the gradient does not.
2. Frozen-model line search. ART's handoff says Armijo guarantees decrease only for
   the frozen surrogate. That is true of the current inconsistent objective. For a
   consistent objective the frozen loss is a majorizer, and decrease is guaranteed in
   exact arithmetic (Section 20; finite-precision checks in Section 33.3). The fix is the objective, not the line search.
8. Validation option. `_fit_convergence_model` holds out a fraction of the samples
   from the fit, but then evaluates the stopping metric on *all* samples, the training
   rows included. The leaf is then refit on all data. This is not a held-out acceptance
   test (review §3.7).
3. Degenerate gates: no balance constraint; the sigmoid can saturate with almost all
   mass on one side, giving tiny gradients and false convergence, rejected only at the
   end (observed on curved interfaces, where the plane isolates 1–2 of 150 points).
4. Optimistic stopping and split gain on training data.
5. MCMC dependence; fixed global thinning; zero-burn-in inheritance argument.
6. Constraint growth with depth, no LP feasibility, no minimum volume check.
7. Scalability: polynomial feature growth, two refits per iteration, serial restarts.

---

# Part III. AFINNI and ART compared

| Aspect | AFINNI | ART |
|---|---|---|
| Output | vector (`fd` components) | scalar |
| Data | user-supplied `X0`, plus augmentation of small rejected clusters | a fixed per-node budget sampled inside the node |
| Region of a leaf | implicit: the leaf's points; ellipsoid `Q` for membership | explicit convex polytope; exact partition of `D` |
| Domain | none (data defines it) | required (box or polytope) |
| Roots | `K0` k-means clusters (a forest) | one root (the domain) |
| Split rule | closed form: one eigenproblem (`M C M`), median or sign cut | iterative optimization of `(w, z)` minimizing children's fit error |
| What drives the split | curvature geometry of `f` (estimated `|H|^{1/2}`) | fitting error directly (variable projection) |
| Cost per split | a few passes over the cluster's points | tens of iterations, each with two weighted refits |
| Balance | median: exactly balanced; sign: unbalanced | unbalanced; minimum side size enforced |
| Sampling | convex combinations inside the convex hull | hit-and-run in the polytope (stationary measure uniform; finite runs are approximate and correlated, not iid uniform) |
| Stopping | per-component `atol`/`rtol`, Linf or L2, on cluster points | one metric and threshold on node samples; optional holdout |
| Leaf models | constant, Taylor (1st/2nd order), affine LSQ | affine or polynomial ridge |
| Stored structure | leaves only | full tree, path ids, diagnostics |
| Inference | kNN + trust-region rerank + Shepard blend; may flag "outside" | root-to-leaf routing; exactly one leaf; no domain check (extrapolates silently) |
| Inference cost | `O(kNN) + O(k·coarse·d²)` per query; dense `Q` of `d²` per leaf | `O(depth · d)` per query; `d + 1` per internal node |
| Parallelism | thousands of clusters per batch on GPU; MPI with work stealing | cohorts of ≤ 20 nodes on GPU; CPU sampling/oracle; no MPI |
| Theory | `O(N_c^{-2/d})` heuristic rate; Hessian-metric motivation | variable projection; Riemannian descent (pending a consistent objective) |

**AFINNI is already an oblique tree.** As derived in Section 5, AFINNI's split is the
hyperplane `w = Mv/‖Mv‖`, `z = wᵀc + τ/‖Mv‖`, except where tied scores are split by
row position (Section 31.7). Two further facts make the analogy close:

1. The k-means roots are the Voronoi cells of the final centres (both backends assign
   labels to the returned centres), which are convex polytopes:
   `cell_k = {x : (c_j − c_k)ᵀx ≤ (‖c_j‖² − ‖c_k‖²)/2  ∀j ≠ k}`. Caveat: assignment is
   float32, so near-ties may be on the wrong side of the exact bisector.
2. Convex combinations of points in a convex polytope stay inside it, so AFINNI's
   augmented points never leave their (implicit) polytope.

So, up to score ties at the median cut and float32 root assignment, every AFINNI tree
is implicitly a tree of convex polytopes with oblique hyperplane splits, which is ART's
structure. AFINNI discards `(w, z)` and replaces membership with a
nearest-centroid search in the `Q` metric. The trade-off:

- AFINNI pays for a cheap, curvature-informed split and gives up an explicit partition;
  the cost appears at inference (search, Euclidean misses, an "outside" flag).
- ART pays for an optimized split and MCMC sampling per node (much more work per node)
  and gets an exact partition, a stored tree, and `O(depth)` routing.

---

# Part IV. Bringing ART into AFINNI

The goal is to keep AFINNI's batched, out-of-core GPU loop and its MPI and
work-stealing execution, and to add ART's ideas as selectable components:
- hyperplane bookkeeping;
- sampling inside polytopes;
- better split selection;
- tree routing;
- ART's sample budgets and stopping metrics.

Four decisions can be made independently (review §4). Separating them is what lets
each experiment attribute its effect:

| Decision | Current AFINNI | New alternatives | What it changes |
|---|---|---|---|
| Regions and roots | data clusters; no stored partition | box tree, shallow hierarchy, exact Voronoi roots | which region owns a query; where sampling is feasible |
| Samples and budget | convex combinations, only after a rejection | top-up before the fit, inside the child's region; independent validation samples | coverage and oracle cost |
| Split selection | curvature/PCA direction with a count or sign cut | hard threshold search, a bank of directions, soft refinement, hinge proposals | approximation efficiency and training compute |
| Prediction | Euclidean candidates, `Q` rerank, optional blend | hard routing; routing to candidates, then a local rerank | query cost, continuity and deployed error |

## 25. What is needed beyond "record `(w, z)`"

1. **A domain.** `ClusteringConfig` has no domain. Without one, root cells are
   unbounded, so hit-and-run chords are infinite and routing outside the data is
   undefined. Proposed: `domain=BoxDomain(bounds)` (or a general polytope). This is a
   substantive requirement, not a convenience: a box inferred from the data's minima and
   maxima can contain physically invalid oracle inputs.
2. **Root cells.** There are three options (Section 31.6). Voronoi roots need the
   centres that defined the final assignment. `kmeans_labels` returns only labels, and
   the CPU path works on centred data, so the centres and the offset must also be
   returned. Assignment must be exact (float64) at the bisectors. `knndbscan` roots are
   not convex.
3. **A predicate that reproduces the training partition.** The median cut is by count,
   and tied scores can make it unrepresentable by any plane (Section 31.7).
4. **A node identity and a node table** (Sections 31.2 and 31.5).
5. **A terminal record for every region.** Every terminal region needs a model and a
   status, including max-depth, failed-split and budget-exhausted nodes. If a split is
   invalid, keep the parent as a terminal leaf rather than committing an empty child
   and inventing a model for it (review §13.5).
6. **Unbalanced splits**, handled safely (Section 30).
7. **Explicit oracle requirements.** The loop evaluates `f(c)` for every fitted
   cluster. An affine routed mode does not need it. Declare centroid values the same
   way gradients are declared (`needs_centroid_value`) and keep today's behaviour in
   the AFINNI modes (review §11.6).

## 26. Where each ART mechanism goes

| ART mechanism | AFINNI slot | Change needed |
|---|---|---|
| polytope per node | — | domain, root geometry, node identity, carried constraints |
| hit-and-run top-up | `Sampler` (its docstring anticipates a polytope field in `ClusterPoints`) | `ClusterPoints.regions`; batched GPU kernel |
| `n_node = multiplier·p_eff` before the fit | `augmentation_targets` (fixed, after rejection only) | a budget rule chosen by the mode, plus a pre-fit stage |
| split proposals | `SplitStrategy.split` | returns a stored predicate `(w, z)` and a validity mask |
| polynomial ridge leaves | `LocalModel` | a separate weighted-regression contract (Section 33.3); batched features |
| ART stopping metrics; pointwise violation rate | `AcceptanceCriterion` | new segmented reductions |
| minimum side / gain checks | inside the split strategy | per-cluster validity mask and fallback |
| root-to-leaf routing | prediction in `surrogate.py` (fixed) | a `predictor=` option: `"knn"` (today) or `"route"` |

Components are only interchangeable when they are compatible, and the configuration
must check this (review §4):
- tree routing needs routable splits and complete terminal records;
- polytope sampling needs a region and a feasible seed;
- soft splitting needs a model with a defined weighted objective. A Taylor model built
  from oracle derivatives does not qualify.

## 27. Approach 1: global ids, parent links, Morton ids and planes

*Interpretation of the whiteboard sketches.*

**Memory argument.** If each node stored its full lineage `A_i ∈ ℝ^{(d+1)×D}`, memory
would be `Σ_i ℓ_i (d+1)`. Each split node contributes exactly one hyperplane, so the true
memory is `(d+1)·(number of nodes)`. Store each plane once and reconstruct a node's
polytope from its ancestors.

**Arrays**, indexed by a global node id `i`:

| Array | Content |
|---|---|
| `mid[i]` | path id: the root id plus one bit per level, `child = 2·parent + side` |
| `pid[i]` | global id of the parent |
| `planes[i]` | `(w, z)` of the hyperplane that split node `i`: `d + 1` values |
| level, leaf data | level of the node; centroid, end offsets and model for leaves |

**Parallel allocation.** Given the `B` bad (rejected) nodes of a level and the current
end of the arrays `cp`, bad node `j` creates its children at

```text
L = cp + 2j,   R = cp + 2j + 1
mid[L] = 2·mid[parent] + 0,  mid[R] = 2·mid[parent] + 1
pid[L] = pid[R] = global id of parent
cp ← cp + 2B
```

No atomics and no prefix sum are needed, because `j` is the node's index in the bad
list. Appending `2B` children per level makes the global array a sequence of level
blocks, i.e. **breadth-first** storage.

**Lineage gather** ("requires a custom kernel"). `PLANES(leaf ids) → (d+1) × D × p`
collects, for `p` nodes, their `D` ancestor planes, by following `pid` or by looking up
the prefixes `mid >> k` in a hash table. **Reorder** everything by Morton order at the
end, for locality.

**Concern raised:** pointer chasing through `pid` or a hash table is irregular memory
access.

**Review notes.**
- The compact, authoritative node record is the right *permanent* representation.
- When parent indices are already known, no hash table is needed.
- The `cp` counter is per rank. Using it for global ids in the training loop would
  require coordination between ranks, so assign compact global indices at assembly
  instead (Section 31.5).
- "Morton id" here means a binary *path* code, not a spatial Morton code built by
  interleaving coordinate bits. It gives ancestry and an order, but no general
  Euclidean locality for oblique partitions (review §3.2).

## 28. Approach 2: per-level buffers and path-code backtracking

**Arrays.** Keep AFINNI's per-level `starts`/`ends` and batched operations. Add, for
each level `ℓ`, a buffer of the bad nodes' path ids `m_BAD,ℓ` and their planes `P_ℓ`.

**Lineage.** A node at level `ℓ` with path id `m` finds its ancestor at level `ℓ − k`
from `m >> k`, and reads that ancestor's plane from `P_{ℓ−k}`. Converged nodes are
pushed out with their path id. Bad nodes and their planes are pushed to the next level.

**Observations.**

1. Approach 2's per-level buffers are exactly approach 1's level blocks, cut at the `cp`
   boundaries. Memory is identical: one `(d+1)` row per split node.
2. Approach 2 needs a map from `m >> k` to a slot in `P_{ℓ−k}`. The sketch does not
   specify one; the options are a hash table or a search.
3. **Key insight of approach 2.** By AFINNI's ordering invariant (Section 9), each level's
   queue on a rank is already sorted by (root, path). So each level's path array is
   sorted for free, and the slot is `searchsorted(mid_{ℓ−k}, m >> k)`, a vectorized
   binary search with no hash table. Binary-search probes are dependent and can touch
   distant addresses, so this is not automatically coalesced. Sorted queries improve
   reuse (review §3.3). The order holds per rank. After migration, or across ranks,
   final assembly must establish the order it needs.
4. A converged leaf has no plane of its own. Its region is its ancestors' planes plus
   the side bits, which are the low bits of its path id. Only split nodes store planes.
5. Sorting by a path code with a leading sentinel bit groups nodes by depth first. That
   is BFS order, not the left-to-right subtree (DFS) order a routing layout may want.
   Choose the inference layout explicitly (review §3.3).

## 29. A clarification for approaches 1 and 2: splitting does not need the lineage

Both sketches describe gathering the ancestors' planes "to compute the hyperplane of the
current node". ART's splitters use only the node's samples `(X, y)`; they never read
ancestor planes. The lineage is needed only for:

1. **Sampling inside the node's polytope.** Every hit-and-run step needs all of the
   node's constraints.
2. **The polytope as an output.** This is optional, because routing does not use it.

So lineage access is needed by the sampler, once per cluster per batch. It is never
needed inside the split optimization's inner loop, which keeps the optimizer's
workspaces dense and reusable (review §1).

## 30. Unbalanced splits

AFINNI's data path already handles unbalanced children:
- `child_sizes` is an arbitrary `(2K,)` vector;
- the `sign` cut already produces unbalanced children;
- `_check_split` accepts any non-negative sizes;
- node-id allocation is per node, so it doesn't depend on point counts.

What depends on the median assumption, and the fixes:

1. **Tiny children can be falsely accepted.** This is the important one. Augmentation
   sizes the parent to `2·min_pts` so each median half gets `min_pts`, and only rejected
   clusters are augmented. An unbalanced cut can produce, say, a 5-point child. At the
   next level that child is fitted and tested on its 5 points, and an affine model in
   `d = 10` fits them exactly: zero error, accepted.

   **Fix:** a pre-fit stage that tops up every cluster below its budget *before* the fit
   and the acceptance test, together with a rank/conditioning check. A count alone is
   necessary but not sufficient: nearly collinear points, duplicates and correlated
   chains still give weak designs (review §13.2). Cap the number of resampling attempts
   so that a thin or invalid cell ends as a reported terminal failure, not an infinite
   loop. With a known polytope the top-up can be hit-and-run inside the child, which is
   ART's design.
2. **Empty children.** Never commit them; an invalid split leaves the parent terminal.
3. **Count versus fraction.** Distinguish a minimum *count*, needed for a stable fit,
   from a minimum *fraction*, imposed for balance. Region top-up can supply samples to
   a small-volume child, but a hard fractional rule forbids that child entirely. That
   can block a valid cut around a small, difficult feature, so track how often the rule
   rejects the best candidate (review §8.7). Decision: region mode drops the fraction and
   uses four separate safeguards (Section 37.3). Top-up can recover small caps, but it
   does not make their inherited-data scores reliable (review §25.3).
4. **Thin clusters give a degenerate `Q`.** This matters only for the kNN predictor.

## 31. Approach 3: the combined plan

Approach 3 uses approach 1's authoritative node records, approach 2's ordering insight, and
self-contained migratable work items. It separates three kinds of storage that have
different needs (review §5.1):

| Storage | Lifetime | Content | Needs |
|---|---|---|---|
| Live work item (the queue) | while unresolved | points, values, optional gradients; node key, true depth, root id, carried constraints, RNG and budget state | contiguous; complete enough to continue after migration |
| Node records (rank-local, append-only) | permanent | one record per committed split or terminal | immutable; compact; written by the processing owner |
| Inference model | after assembly | compact routing arrays and leaf columns | layout chosen for the predictor |

### 31.1 Principle: inheritance rules applied at split time

Lineage-dependent state is defined by a rule `child = rule(parent, split)`. Each rule is
applied once, batched over the rejected clusters, and its result travels with the
cluster:

```text
child.depth       = parent.depth + 1
child.path_code   = (parent.path_code << 1) | side          # see 31.2
child.root        = parent.root
child.constraints = parent.constraints ⊕ signed row of (w, z)  # see 31.4
child.rng_stream  = (child.root, child.path_code)            # identity only; see 31.9
node_records.append(split record of the parent)
```

A thief therefore needs nothing from the victim's history. That is the decisive
property for MPI.

### 31.2 Node identity (corrects the earlier packed encoding)

The earlier proposal, `mid = (root ≪ L_max) | (1 ≪ depth) | path`, collides at
`depth = L_max`. For example, with `L_max = 3`, `(root 0, depth 3, path 1)` and
`(root 1, depth 0, path 0)` both encode to 9 (review §3.2; verified). Use two fields:

```text
node_key  = (root_id, path_code)
path_code(root)   = 1                                   # leading sentinel encodes depth
path_code(child)  = (path_code(parent) << 1) | side     # side: 0 = left, 1 = right
path_code(parent) = path_code(child) >> 1
depth             = bit_length(path_code) − 1
```

If a single packed integer is needed, reserve `D_max + 1` bits for the whole path code:
`key = (root_id << (D_max + 1)) | path_code`. Parent and child operations then mask and
update only the path field, never shift the whole key. Check for overflow before
creating a child. Any fixed width has a depth limit: 63 bits holds
`⌈log₂ R⌉ + D_max + 1 ≤ 63`. Deeper trees need two words. With two fields the path
field has its own limit: a positive signed int64 code with the sentinel bit allows
depth ≤ 62 (review §20.3).

**One convention for upper hierarchies.** Box and grid roots use an explicit root table
followed by per-root path codes. A learned shallow binary hierarchy (Section 31.6)
instead uses one root and the full path prefix through that hierarchy. Never reset a
frontier cell to `path_code = 1` while also expecting parent lookup to reach its
upper-tree ancestors. Depth limits and constraint lengths use the same choice.

### 31.3 The migratable work item

A steal transfers whole ready nodes between processing stages, never an optimizer in
mid-iteration. The payload contains, or references through configuration broadcast once
(review §11.4):

- the node key, true depth, root id and signed ancestor constraints;
- samples, outputs, optional gradients, a training/validation field and a separate
  provenance field (inherited, new), and statistical weights;
- sample ids or provenance needed for deterministic reuse;
- RNG stream identity and counters for sampling, candidate directions and validation;
- any retained chain state and a frozen direction transform;
- budget state, retry count, and the selected model and feature specification;
- a fallback model, or enough data to build one if processing terminates.

Slacks can be recomputed on arrival. Shared root geometry and component settings are
broadcast once. Every rank verifies a schema/configuration digest and rejects
inconsistent jobs. Python objects and closures never enter the hot path.

**First-version contract (restricted, review §20.3 and §29.4).** Part V carries less than
the full list above, so it states what it supports instead:
- only **unstarted ready nodes** are stolen; a node's chains, attempt counters and
  partial results stay local to the task that processes it, and are reconstructed from
  the node key when a node is started;
- samples are unweighted, and per-sample ids are not carried;
- budgets are per node only. A global oracle budget would have to account for
  in-flight allocations across ranks; a shared config limit does not enforce it;
- the fallback model is the parent fitted before the split was committed;
- a received job is processed by its receiver before it may be offered onward
  (Section 33.7).

Each relaxation (mid-node migration, weights, global budgets) adds the corresponding
fields back.

### 31.4 Constraint storage: carry the full path first

Each live cluster carries its own `h` signed ancestor rows of size `d + 1`, sized by its
**true depth**. Depth must not come from the rank-local round counter, which differs
from depth after a steal (review §5.4). Each rank's queue is currently homogeneous in
depth, so a batch's blocks form a dense `(K, h, d+1)` tensor. If a scheduler later mixes
depths, use depth buckets, or ragged offsets padded within a GPU bucket.

Overhead (review §5.3), with `n` samples per live cluster, `q` outputs, and `g = 1` when
gradients are carried:

$$
B_{points}=8n(d+q+g\,dq),\qquad B_{path}=8h(d+1),\qquad
\frac{B_{path}}{B_{points}}=\frac hn\ \ (q=1,\ g=0).
$$

So for `n = 100` children the overhead is about 30% at depth 30 and 50% at depth 50,
independent of `d`. Under ART's budget, `n = 50(d+1)`, it falls to a few percent. The
sampling policy therefore changes the best storage policy. Inheriting the path also
copies about `2h(d+1)` numbers per split, which costs host time and migration traffic,
not just peak memory.

**Later, if measured overhead justifies it**, use a hybrid of approaches 3 and 1 (review §5.2):
- keep a rank-local, immutable node/plane store, and let ready jobs reference their
  ancestry by local handle;
- before sampling, build each batch's contiguous constraint buffer from those handles;
- when migrating, send the needed ancestry with the jobs, deduplicated within the
  transfer, and have the receiver intern it.

This needs no remote lookup and no global MPI window. Decide on measured RSS and MPI
bytes, sharing of path prefixes, and how often migration and sampling happen, not on
`d · depth` alone.

### 31.5 Node records, ownership and assembly

- The rank that processes a node writes that node's split or terminal record. Migrating
  an unprocessed node transfers the right to process it. Records already committed are
  never sent back to their original rank.
- At assembly, rank 0 gathers all records and validates them:
  - every key is unique;
  - every non-root node's parent exists;
  - every committed split has exactly two children;
  - every reachable terminal has exactly one model and status.

  It then assigns compact indices with approach 1's rule (`first_child = cp + 2j` within
  each sibling-adjacent block). A layout that does not keep siblings adjacent (DFS or
  blocked) stores two child indices.
- The single-rank gather is acceptable at first. Very large models will need sharded
  output and distributed assembly later (review §5.5).

### 31.6 Roots

| Option | Use | Notes |
|---|---|---|
| Single box | mathematical reference | Exposes little parallelism early: reaching `W` independent nodes needs at least `⌈log₂ W⌉` levels of splits |
| Shallow hierarchy (proposed default for concurrency) | many initial tasks with no Voronoi bookkeeping | Cheap cuts (axis-aligned, PCA or metric) persisted as the top of the tree. Axis cuts let descendants carry coordinate bounds (`O(d)` per sampling step); oblique cuts add about `log₂ R` planes. Size the frontier from GPU cohort capacity and rank count. |
| Exact Voronoi roots | compatibility with current AFINNI | Needs the actual centres and exact assignment. Store the centres once and evaluate the bisector inequalities in tiles; no per-root face list is persisted (review §23.1). A global minimal face list is not needed. Optional speed-up if root faces are a measured bottleneck: bound the chord with cached faces, then certify both endpoints against all centres (Section 33.4). Keeping just the nearest few centres without that certification is not exact. |

AFINNI's existing `uniform_grid_cluster_labels` is a first shallow-hierarchy baseline,
provided it uses the declared domain and keeps its bounds and bin structure instead of
discarding them (review §6.4). Geometrically non-empty roots with no initial samples
must be sampled, given an explicit fallback, or reported as unresolved. Empty sample
sets and empty regions are different conditions.

### 31.7 A geometric split predicate that reproduces the training membership

Store the predicate that routing will evaluate, with one evaluation convention for
training, sampling and inference. **Canonical form:** `wᵀu − z ≥ 0` in a fixed run-level
normalized coordinate `u = (x − x₀)/s`, where `x₀` and `s` come from the domain and are
stored once with the model. That keeps most of the numerical benefit of centring
without node-local anchors. A node-local form `aᵀ(x − c_ref) ≥ τ` is not used, because
converting it to `(w, z)` loses the centring and storing it needs `c_ref` per node. The
constraint rows, chord calculations and router all use the same `(w, z)` in `u`, so the
sampler samples exactly the cells the router owns, up to the boundary tolerance
(review §20.3). Then:

1. Pick the threshold at a **gap between distinct scores**, subject to the minimum side
   count. With a strict-left predicate, use the upper adjacent score, so a midpoint
   cannot round onto the lower value.
2. Re-partition the training points with the stored predicate and verify the
   membership.
3. If ties make the requested balance impossible, accept an unbalanced cut, try another
   direction, or keep the node terminal. Never split identical points by row index and
   call the result geometric.
4. Handle `Mv = 0`, zero covariance and non-finite planes explicitly.

This deliberately changes child counts relative to the median cut, so it lives behind
the new mode and cannot be bit-identical with AFINNI's median split.

### 31.8 Routing and status reporting

```text
node = root_of(x)          # box: 0; hierarchy: short traversal; Voronoi: exact 1-NN over centres
repeat until every query sits at a terminal:
    active = ~is_terminal[node]
    for active queries only:                 # never gather plane arrays at terminal ids
        side = (predicate(node, x) ≥ 0)
        node = child(node, side)
predict with the terminal's model
```

The cost is `O(d·h + d·q)` per query for affine leaves, where `h` is the path length.
Report the query-weighted mean, p95 and maximum depth, not only the leaf count. GPU
traversal variants to compare later (review §12.4):
- fused per-query traversal;
- grouped traversal by subtree;
- shallow upper levels followed by fused subtrees.

The dense `Q` (`d²` per leaf) becomes optional for routed-only models. At `d = 80`, a
million `Q` matrices take 51.2 GB, against about 0.65 GB for a million planes.

The predictor reports separate statuses, and AFINNI's `(prediction, outside_mask)`
contract is kept for its kNN predictor (review §12.5):

```text
outside_domain     # the only meaning of "outside" for the routed predictor
terminal_reason    # accepted, max_depth, budget_exhausted, split_failed, ...
validation_status  # passed, failed, insufficient, not_performed
outside_Q_support  # optional empirical diagnostic
leaf_id
```

Two cautions:
- Hard routing can introduce jumps across split planes (review §28). Hard routing is the
  first reference; boundary and trajectory stress tests are part of it from the start.
  - A band blend `(1−α)f_L + αf_R` adds a derivative term `(f_R − f_L)∇αᵀ`, of order
    `ε/b` for band width `b`, so a narrow band can make Jacobians and stiffness worse.
  - Capping the number of blended branches per query does not by itself give
    continuity: a dropped nonzero weight causes a jump. Only compact-support weights
    that vanish where a branch is dropped do.
  - Sibling affine fits agree on the face `wᵀx = z` if and only if
    `A_R − A_L = v wᵀ` and `b_R − b_L = −v z`. This can be enforced when a parent is
    split, but descendants must then preserve the inherited interface trace.
  - For a flow map, decide from measured one-step and multi-step error whether
    continuity is needed. For a vector field inside an ODE solver, or a downstream
    differentiable loop, treat continuity and Jacobian accuracy as explicit
    requirements and test a representation built for them.
- Routing a model trained by the current algorithm is a *diagnostic of the implicit
  partition*. It does not show that the coverage problem is solved, because the old
  leaves' samples may cover only a small part of their routed cell (review §12.3).

### 31.9 Reproducibility

Derive random streams from the stable node key plus the purpose: sampling, validation,
candidate directions, retries. The current seed, `seed + level + batch + rank·1000`,
changes whenever the batch cap, the rank count or the steal schedule changes.

**Identity is not position (review §20.3).** One node can ask for several top-ups,
warm-up extensions, validation attempts or split retries. If each restarts from the node
key, samples repeat. Every draw is therefore fully indexed by a counter-based generator
(e.g. Philox):

```text
stream = (root, path_code, purpose, attempt, chain)      # counter = step
```

Under the first-version contract (Section 31.3), attempt and step counters exist only
inside the task that processes a node, so nothing extra is migrated. Allowing a
partially processed node to migrate would require carrying those counters and any chain
state.

There are three separate contracts (review §11.5):

1. AFINNI mode on the CPU stays bit-identical under storage-only refactors.
2. New modes have logical random streams that don't depend on which rank processes a
   node.
3. CPU and GPU runs of the same algorithm agree within the repository's tight numerical
   tolerances, with exact structural decisions.

## 32. Evaluation of the three approaches

| Criterion | Approach 1: global id + `pid` | Approach 2: per-level + path backtracking | Approach 3: combined plan |
|---|---|---|---|
| Permanent plane memory | `(d+1)` per node | same | same |
| Lineage lookup during refinement | per batch, `pid` chasing (irregular) | per batch, `searchsorted` per level (dependent probes, no hash table) | none: carried; hybrid store later |
| Live-state overhead | none | none | `h/n` of point data (about 30% at depth 30, `n=100`; a few % under ART budgets) |
| Hash table | only for `mid` lookups | no | no |
| Id allocation | `cp + 2j`, per rank | per level | `(root, path_code)` keys; compact indices at assembly |
| Unique across MPI ranks | `cp` ids need coordination | path ids yes; slots no | yes, no coordination |
| Work stealing | the payload must ship the ancestry | same | the work item is self-contained |
| Unbalanced splits | unaffected | unaffected | unaffected; safeguards in Section 30 |
| Inference | `pid`/children tables | needs a child lookup | compact arrays chosen at assembly |
| Implementation effort | custom lineage kernel | `searchsorted` plus level buffers | the fixed internal column table (Section 36), node records and assembly |

Recommendation: approach 3 for the first implementation. It has the simplest story for
distributed ownership, which is worth more than saving path duplication before profiling
it. Approach 1's records and approach 2's ordering live inside it.

## 33. Batched components on AFINNI's GPU loop

Execution rule: nodes are grouped into **compatible cohorts**, i.e. similar feature
count, model type, sample-count bucket and constraint-count bucket. Inside a cohort,
keep stable slots and compact the list of active slots as nodes finish, as ART's batched
optimizer does. Masks alone do not skip work inside GEMMs or large kernels:
- full-cohort masked execution costs in proportion to `K · max I_k`;
- active-list execution costs about `Σ I_k`.

Refill finished slots at cohort boundaries at first (review §10.1). Every step is a
segmented pass over the cohort's points or a batched small dense operation. Never
materialize per-point outer products `φφᵀ`: use tiled reductions or batched products,
and reduce locally before any atomics (review §10.2).

### 33.1 Geometric version of AFINNI's split (baseline B1/B2)

- Compute `a = Mv` per cluster, the scores `aᵀ(x − c)`, and a gap threshold with a
  minimum side size (Section 31.7).
- Store the predicate.
- The `O(K d² + N d)` projection replaces the current `O(N d²)` one (Section 5 note).

### 33.2 Hard threshold search and a bank of directions (B3)

AFINNI already pays to find a good direction; the median cut optimizes balance, not the
children's fit error. Before optimizing all `d + 1` gate variables, search thresholds
along the chosen direction (review §9.1). With features `φ_i`, outputs `y_i`, and the
points sorted by score, accumulate the sufficient statistics of each side:

$$
G_s=\sum_{i\in s}\phi_i\phi_i^T,\qquad B_s=\sum_{i\in s}\phi_i y_i^T,\qquad C_s=\sum_{i\in s}\|y_i\|^2,
$$

$$
\mathrm{SSE}_s(\beta)=C_s-2\operatorname{tr}(\beta^TB_s)+\operatorname{tr}(\beta^TG_s\beta).
$$

- Bin the scores (8–32 quantile bins, as an experimental budget), accumulate per bin,
  and prefix-sum over bins. Never form a per-point prefix tensor of size `N·p²`.
- Exclude bins whose cut falls in a tied gap or leaves a side below the minimum size.
- Solve the candidate child systems in batch and evaluate the SSE. With ridge, include
  the penalty if it is part of the split objective.
- The prefix SSE formula subtracts large, nearly equal terms when the fit is good, so
  roundoff can make it negative or reorder near-ties. Use it only to **screen**; recompute
  the leading few candidates' residual sums directly over their rows before accepting
  a small gain (review §20.3). This matters most near the tolerance.
- No temperature, no line search; it fits the segmented kernels directly.
- Add a small bank of directions: the Hessian-informed one, the leading covariance
  direction, further metric eigenvectors, and a few deterministic random directions.
  Process them in bounded tiles; they also give the GPU parallel work near the root.
  Near repeated eigenvalues, include several directions spanning the tied eigenspace
  rather than trusting the one the eigensolver returns. Signed or absolute Hessian
  candidates are included only when derivatives are already available. Region mode
  does not buy extra gradient or Hessian evaluations for saddles: on `x₁x₂`, `|H| = I` is
  isotropic, and even the signed-eigenvector diagonal cut (child MSE `7/225`) loses to
  an axis cut (`1/36`) (review §27; checked numerically). Hard fit scoring over the bank
  is the protection. Adding one interaction feature to the leaf is a separate option
  worth testing alongside splitting.
- Its limitation: it can miss a good plane that is not along any candidate direction.

### 33.3 Consistent soft-oblique refinement (B4)

The soft optimizer is used only to polish the best hard candidate, within a fixed
budget, with that candidate kept as the fallback.

**Objective (review §8.2).** Per node:
- prepare a fixed feature matrix `Φ` (with an intercept);
- fix input normalization, output scaling `Y`, a penalty matrix `P ⪰ 0`, and the ridge
  `λ`, set once from the unweighted node Gram;
- freeze the temperature for the run.

Then

$$
F(\theta,B_L,B_R)=\tfrac1n\Big[\textstyle\sum_i(1-\tilde\pi_i)\|Y_i-\phi_iB_L\|^2+\sum_i\tilde\pi_i\|Y_i-\phi_iB_R\|^2+\lambda\operatorname{tr}(B_L^TPB_L)+\lambda\operatorname{tr}(B_R^TPB_R)\Big],
$$

$$
J(\theta)=\min_{B_L,B_R}F,\qquad (\Phi^TW_s\Phi+\lambda P)B_s=\Phi^TW_sY .
$$

- **Weights:** used identically in every fit and loss (or `δ = 0` with sufficient
  regularization), computed with a stable logistic and no clipping (review §8.4, §20.3):

  ```text
  W_R = δ + (1 − 2δ)·σ( s)
  W_L = δ + (1 − 2δ)·σ(−s)          # = 1 − W_R, without the cancellation
  dW_R/ds = (1 − 2δ)·σ(s)·σ(−s)     # not π̃(1 − π̃)
  ```
- **Gradient:** with `a_i = (‖r_R,i‖² − ‖r_L,i‖²) π̃'_i / n`,

  $$\nabla_wJ=X^Ta/T,\qquad \partial_zJ=-\textstyle\sum_ia_i/T,$$

  projected to the tangent space, `(I − wwᵀ)∇_w`. The penalty does not appear in the
  gradient because it does not depend on θ. A gate-dependent `λ` would add derivative
  terms, which is why `λ` is fixed per node.
- **Line search:** frozen-model Armijo is justified by the majorization argument
  (Section 20). Trials need only gate evaluations and cached squared residuals; each
  accepted step needs one pair of refits.
- **Finite-precision descent (review §24).** A fixed tolerance like `1e-12` means
  different things at different scales. Instead:
  - The inner solve's suboptimality is exactly
    `ε_inner = Σ_s tr(R_sᵀ H_s⁻¹ R_s)/n`, with `H_s = ΦᵀW_sΦ + λP` and residual
    `R_s = ΦᵀW_s(Y − ΦB̃_s) − λPB̃_s`, recomputed from the feature data, not from the
    rounded Gram. With `P = I` and `λ > 0`, `‖R_s‖²_F/(nλ)` is a rigorous but loose
    bound; solving `H_s Z = R_s` with the existing factors gives a tight estimate.
  - Accept a proposed step only if
    `Q̂_new + e_new ≤ F̂_old − e_old − ε̄_inner,old − m`, with `m > 0` and `e` the
    loss-evaluation errors. This implies `J_new ≤ J_old − m`. With engineering
    estimates of `e` and `ε`, call it *numerically monitored descent*, not a certificate.
  - Evaluate accepted losses with stable, fixed-order reductions. Never use the prefix
    SSE formula for the final comparison.
  - If the refit makes the frozen value worse beyond its uncertainty, keep the old
    coefficients at the new gate or reject the step.
  - Fallback order: reject non-finite values; repair the solve (refinement, QR/SVD);
    recompute state and gradient; try a bounded number of refitted trials; otherwise
    stop and return the best valid hard candidate. Raising the ridge changes the
    objective, so it restarts the node's optimization.
- **Complementary Grams:** `G_L = G_tot − G_R` is exact when the weights sum to one, but
  loses relative accuracy on a tiny side. Forming the smaller-mass side directly is a
  heuristic: a low-weight sample with a large feature norm can still dominate a
  direction. Scale features, check solve residuals, and fall back to two direct Grams
  near singularity (review §8.5, §20.3).
- **Saturation (review §25).** Floors keep the inner systems supported but do not keep
  `dW/ds = (1−2δ)σ(s)σ(−s)` away from zero. A saturated gradient is reported as
  `soft_optimizer_stalled`, never as convergence. The B3 candidate stays the feasible
  incumbent throughout.
- **Normalization and degeneracy:**
  - centre and scale the inputs, and convert the stored plane back to raw coordinates;
  - require a minimum hard side count;
  - keep the best feasible hard iterate;
  - use a short temperature continuation schedule;
  - compare candidates only after hardening and refitting. A lower soft loss is not
    evidence of a better tree.
- **Temperature:** with `Σ_k = C_k / n_k` (normalized; the earlier formula used the
  unnormalized scatter and had a spurious `√n`), use either
  `T = c·sqrt(tr Σ_k / d)·n_k^{−1/d}`, or the gate-specific width `c·sqrt(wᵀΣ_k w)`.
- **Weighted model contract:** this is a separate capability, not a `weights=`
  argument added to every `LocalModel`. The current affine model centres with
  *unweighted* means, so a weighted fit must use fixed features with a weighted
  intercept solve, or consistent weighted means. Oracle-based Taylor models don't
  participate (review §8.6).
- **Solver:** do not pass gate-weighted Grams through
  `solve_regularized_covariance_inplace`. Its internal trace ridge would make `λ`
  depend on the gate, and it overwrites its inputs. Add an explicit solver with fixed
  regularization, failure status and residual checks; send failing nodes to a separate
  QR/SVD batch; compare custom kernels against library batched factorizations at
  realistic `p` (review §10.8).
- **Multiple outputs:** fixed output scales derived from the tolerances, with the sum
  of squared normalized residuals as the objective. This keeps one Gram factorization
  with several right-hand sides. Acceptance stays per component.

### 33.4 Sampling inside polytopes

Sampling inside regions is a scientific change, not just another CUDA kernel
(review §7).

- **Seeds and warm-up (review §22).** Inherited points are feasible seeds, not
  stationary samples (Section 18). No step count based on dimension alone (`20`, `50d`,
  `100d`) guarantees mixing in a learned cell. The policy:
  - work in normalized coordinates and run a geometry-only pilot before querying the
    oracle;
  - start several chains from dispersed feasible points: inherited points, or interior
    points reached from LP extremes (an LP extreme itself is on the boundary and is a
    poor start);
  - freeze the direction transform within each pilot or production segment;
  - monitor normalized coordinates, a few fixed random projections, projections on
    the cut normals, the squared radius and normalized face slacks. Raw coordinate
    means alone miss rotated thin directions;
  - extend pilots in doubling blocks and compare between-chain and early/late
    behaviour (rank-normalized split-R̂, bulk/tail ESS, projection quantiles). These
    are diagnostics, not proofs of mixing, and need enough retained states to mean
    anything;
  - stop at a geometry-work budget. If diagnostics stay poor, recondition, change the
    sampler, request better seeds, or finish with
    `validation_status = insufficient_sampling`. Never accept a node because its
    chains ran out of budget without finding an error;
  - choose the kernel, transform and run length from the pilot, then run a separate
    production segment, so stopping when diagnostics look good does not select the
    retained samples.

  Biased but feasible training points are still useful for fitting; a probabilistic
  validation claim is a different matter (Section 33.5).
- **Chain layout.** Compare a few persistent chains per node against one chain per
  requested sample. With `n_new` samples, `C` chains and warm-up `b`, warm-up costs
  `C·b` steps against `n_new·b`. Persistent chains amortize warm-up for training.
  One endpoint per chain is easier to reason about for validation. Independent
  streams from the same seed point give conditionally independent endpoints, but all
  share that start's bias (review §22.4). Count oracle calls only for retained
  evaluated points.
- **Voronoi root faces.** Compute chords against all centres in tiles as the reference.
  Optional, if measured to be a bottleneck: bound the chord with box/path constraints
  and cached faces, then check both endpoints against every centre by comparing
  distances, not nearest-centre indices, since a boundary point can tie. If the root's
  centre is no farther than any other at both endpoints, every bisector inequality
  holds along the segment (each is affine in `t`), so the chord is exact. Otherwise add
  the violating centre's bisector, tighten and repeat, a bounded number of times
  before falling back to the full pass. Draw `t` only after certification, so no
  sampling bias is introduced. The review checked this on 300 cases (`d` = 2, 5, 10) to
  `2.2e-16`; it is not a measured GPU speed-up (review §23.3).
- **Per-step cost.** Box bounds are coordinate intervals, `O(d)` per step, not `2d`
  dense rows. A step costs `O(d + h·d)` with `h` carried planes, plus `(R−1)·d` with
  Voronoi faces. Keep the slacks `s = b − Ax` and update them as `s ← s − t·Au`, with
  periodic recomputation to bound drift. ART's batched sampler already does this.
- **Tolerances.** Use normalized constraints with scale-aware tolerances. Handle
  near-parallel directions, empty or non-finite chords and thin cells explicitly.
  Never silently clamp a violation and treat the sample as exact.
- **Alternatives.** Rejection from an exact parent sampler costs `1/p` proposals per
  sample, which is a good reference for shallow, balanced children. Coordinate
  hit-and-run is cheaper per step but mixes poorly in rotated thin cells. Compare
  useful exploration per second, not steps per second.
- **Target measure.** Uniform region sampling targets error under the uniform measure.
  If queries follow `X0`'s distribution, the measure must be named and supported.

### 33.5 Budgets, validation and oracle accounting

- The mode's budget rule decides each cluster's sample count before fitting (ART:
  `multiplier · p_eff`) or before splitting (AFINNI: `max(2n, 2·min_pts)`), plus any
  validation samples.
- **Oracle identity (review §13.3).** Suppose every processed node has `n` samples,
  children inherit without duplication, and each child is topped up to `n`. Then one
  split costs `2n − n = n` new evaluations, whatever the balance. A tree with `L`
  terminal leaves therefore costs `N_oracle = nL`, plus centroid, validation and retry
  evaluations. A splitter that reduces the leaf count saves oracle calls directly.
- **Validation (review §21).** Two levels with different claims:
  - **Global audit, the version-one claim.** Draw fresh iid points from the declared
    query measure (for a box and the uniform measure, directly). Evaluate the frozen
    deployed router and models, unresolved leaves included, and count tolerance
    violations. With `m` points and zero violations, the one-sided upper bound on the
    violation probability is `1 − δ^{1/m}`: below 1% at 95% confidence needs
    `m = 299`, below 0.1% needs `2,995` (`m ≥ log δ / log(1 − p₀)`). With observed
    failures, use a one-sided binomial interval. This tests the actual deployed
    selection rule, normalization and boundaries. It does not certify each leaf or give
    a bound uniform over the domain. If the run is changed because of an audit, the next
    reported audit uses fresh points.
  - **Per-node validation, a training diagnostic.** Validate the deployed (finalized)
    model on fresh rows, independent of its fit and split; revalidate after any refit.
    Hit-and-run rows are correlated and only approximately stationary, so the iid
    bound does not apply. ESS does not repair this: it describes chosen observables,
    not rare failure regions a chain never visits. Report the result as
    `validation_status`, not a confidence certificate.
  - **What would upgrade local validation.** If `m` validation points are jointly
    within total variation `η` of `m` iid draws, given the history, zero failures give
    `U = 1 − (δ − η)^{1/m}`. Independent chains with endpoint bounds `ε_i` give
    `η ≤ Σε_i`. These must be proven bounds for the kernel and run length; observed
    autocorrelation is not one.
  - **Adaptive choice of cells is fine** if each test is fresh and conditional on its
    history. Allocate error probability over *attempted* tests, failed and abandoned
    ones included. A per-key allocation needs no global counter across ranks:
    `δ_{root,path,j} = (δ_total/R)·2^{−(2h+1)}·6/(π² j²)` for attempt `j ≥ 1` at depth
    `h`. It is valid for any finite tree but very strict at depth (`2⁻⁶¹` at `h = 30`),
    which is another reason the global audit carries the claim.
  - **Sequential tests** (Bernoulli confidence sequences) can save oracle calls, but only
    once samples are iid and the predictor is fixed. Not in version one.
  - A failed parent's validation rows may be released into its children's training
    data. They are then spent and can never validate a descendant.
- **Pointwise violation metric** for reporting and as an optional acceptance rule:
  `e_i = max_j |f_j − f̂_j| / max(atol_j, rtol_j |f_j|)`, with a violation when
  `e_i > 1`.

### 33.6 Cost and memory

Split cost per cohort, with `I` accepted iterations and `a` line-search trials per
iteration (review §10.3):

$$
T_{split}\sim I\big[c_GNp^2+c_BNpq+c_SKp^3+c_ENpq+c_A\,aN(d+q)\big].
$$

Dense quadratic leaves grow very quickly under ART's budget `n = 50p`, with
`p = (d+1)(d+2)/2` (computed sizes, review §10.4):

| d | p | n per node | one Φ | two Grams | `n·p²` per Gram |
|---:|---:|---:|---:|---:|---:|
| 10 | 66 | 3,300 | 1.66 MiB | 0.07 MiB | 1.4e7 |
| 40 | 861 | 43,050 | 283 MiB | 11.3 MiB | 3.2e10 |
| 80 | 3,321 | 166,050 | 4,207 MiB | 168 MiB | 1.8e12 |

Per-node Gram work grows as `p³ ~ d⁶`, which rules out a blanket dense-quadratic
default. Instead:
- reduce the feature set for the split, e.g. split with affine features and use richer
  features only in the accepted leaves (review §9.4);
- size batches from a per-stage byte estimate that takes the maximum over overlapping
  lifetimes. Today's batch cap models only the `AFFINE_COV_HESS` path and has a
  50,000-point floor.

### 33.7 MPI and work stealing implications

- **Keep the ownership boundary.** Whole ready nodes are stolen between stages, and the
  existing request/grant/retire invariants are kept. Generic columns change the payload
  and its version, but not the protocol's state machine.
- **Make ART-sized queues stealable now.** A queued-point count no longer measures
  work: a rank with hundreds of expensive ART nodes sits below the 10⁶-point threshold
  (an end-to-end test produced 0 grants for this reason). Region mode gets its own
  thresholds and is tested with forced transfers of hundreds to thousands of points.
  This is required before any MPI efficiency claim (review §26.5).
- **Minimal steal policy first (review §26).**
  - Static per-model cost weights, a configurable grant size, measured transfer time
    and request-latency metrics.
  - A receiver processes a granted job through at least one node task before offering
    it onward. Its new children are new jobs and may be stolen. This lease stops
    untouched jobs bouncing between idle ranks.
  - A short backoff after denials; refresh stale load before asking the same victim.
  - Outstanding jobs are tracked separately from estimated seconds: a job whose
    estimate rounds to zero is still work, so time estimates are never the liveness
    signal.
- **Later: grant by predicted makespan.** With predicted finish times `T_v`, `T_h`,
  grant `S` only if `max(T_v, T_h) − max(T_v − w_v(S), T_h + t_move(S) + w_h(S))` exceeds
  an uncertainty margin. For identical workers this gives `w = (T_v − t_move)/2`, which
  explains half-load stealing but not "half the points". Predict the next bounded node
  task, not its unknown subtree; GPU cohort time is not additive across nodes. Calibrate
  per phase with medians or trimmed EWMA throughputs. Under ART's fixed `n = κp`,
  `np²` and `p³` are proportional, so their coefficients cannot be fitted separately
  from those runs alone. Record cold compilation and exceptional solver events
  separately.
- **Keep stealing responsive.** Add bounded progress points between groups of optimizer
  iterations or sampling chunks, so requests are answered while long cohorts run. Never
  announce zero load while GPU work, oracle calls or transfers are outstanding.
- **Batch oracle calls.** Collect new-sample requests across ready nodes. An oracle
  that is itself a distributed solver needs an explicit worker-group design.
- **Test on the right hardware.** On one GPU, use one GPU rank, and test multi-rank
  protocol behaviour on the CPU. Several ranks sharing one GPU does not measure scaling
  (review §11.7).

## 34. Staged experiments

| Label | Training | Predictor | Question answered |
|---|---|---|---|
| B0 | current AFINNI | kNN/`Q` | frozen reference |
| B1 | one tree, built once with AFINNI's sampling, an explicit domain, valid stored predicates and `Q` kept | kNN/`Q` *and* routing, on that same artifact | isolates the predictor; exposes coverage gaps |
| B1-grid | B1 with grid or box roots instead of k-means | both | a new construction baseline; not the unchanged B0 model |
| B2 | pre-fit budget, child-region sampling, geometric AFINNI split | routing | do coverage and budgeting deliver most of the gain? |
| B3 | B2 plus hard threshold and direction search | routing | does a bounded-cost residual splitter pay? |
| B4 | B3 plus consistent soft refinement on a fixed budget | routing | does nonlinear gate optimization pay for itself? |
| R | original ART at a pinned revision | its routing | reference behaviour |

Compare at equal oracle budget, at equal deployed error, and under equal resource
limits. Use common initial samples and an untouched global test set. Cross the model
class with the splitter, so an affine-versus-quadratic difference is not attributed to
splitting. Problem families: constant/affine; rotated anisotropic quadratics; indefinite
quadratics and `x₁x₂`, where the covariance-Hessian heuristic suffers from cancellation
(review §9.5); Rosenbrock; a localized bump; continuous oblique piecewise-affine; a curved
interface; vector outputs with different scales; a real flow map. Use `d` = 2, 10, 40, 80,
with five or more seeds.

A more expensive splitter is justified when
`ΔT_optimizer < ΔN_f · t_f + N_q · Δt_q`, i.e. it saves oracle time or query time. Report
the individual terms (review §15.2).

---

# Part V. Minimal implementation plan: AFINNI and ART only

## 35. Scope

The first implementation supports exactly two algorithms on one engine:

| Mode | What it is | Contract |
|---|---|---|
| `algorithm="afinni"` (default) | today's AFINNI | bit-identical on CPU (`scripts/regression_check.py`); same kNN/`Q` predictor and return shape |
| `algorithm="region"` | ART-style region tree on AFINNI's batched engine: domain, explicit predicates, top-up before the fit, polytope sampling, routing | new numerical behaviour, validated by the B1–B4 experiments (Section 34) |

There are no user-defined properties, no general stage scheduler, and no public
primitive API (deferred to [Appendix A](#appendix-a-deferred-a-pysph-style-property-system)).
All state is a **fixed, known set of columns**, and each mode is a **named plan**.

Three design decisions are kept from the property-system idea, because they make even
the two-algorithm version simpler and safer:

1. **One internal column table.** Every `ClusterBatch` operation iterates over one
   internal list of column specs: split, select, builder growth, MPI send and receive,
   save and load. Today that list is repeated by hand in five places. The list is
   fixed, and not a user API.
2. **Inheritance rules for lineage state.** `depth`, `(root, path_code)` and the
   constraint block are each updated by one batched rule at split time (Section 31.1).
   The RNG stream *identity* follows from the node key; RNG *positions* are counters
   local to the task that processes a node (Section 31.9). Together with stealing only
   unstarted nodes (Section 31.3), that is what makes stolen work self-contained.
3. **Two named plans over one executor (review §29).** One executor owns queue
   admission, host/device transfer, steal progress, timing, exception handling and job
   ownership. The numerical modules keep owning fitting, reductions, sampling,
   acceptance and splitting. Each mode is a named, validated plan, not a set of public
   per-stage switches:
   - independent switches admit incoherent settings, such as validation before the
     final model is built, or routing without complete topology;
   - moving the top-up before the fit in AFINNI mode would change its training data,
     random draws and oracle calls even for clusters that already pass, so that
     combination could never be the bit-identical mode anyway.

   B1–B4 are internal named configurations of the region plan, not public stage
   permutations.

## 36. Columns

`ClusterBatch` today has `X, F, G, sizes, depth`. Region mode adds a fixed set. A column
that a mode doesn't use is `None`, and is never allocated, sent or saved.

| Column | Scope | Shape | dtype | Modes | Migrated | Saved |
|---|---|---|---|---|---|---|
| `X`, `F`, `G` | point | `(n,d)`, `(n,fd)`, `(n,fd,d)` | f64 | both (`G` if needed) | yes | — |
| `use` | point | `(n,)` | int8 | region: `train` or `validation` | yes | — |
| `origin` | point | `(n,)` | int8 | region: `inherited` or `new` (provenance and diagnostics only) | yes | — |
| `sizes` (structural) | cluster | `(K,)` | int64 | both | yes | — |
| `n_train` | cluster | `(K,)` | int64 | region | yes | leaf |
| `depth` | cluster | `(K,)` | int64 | both | yes | leaf |
| `root` | cluster | `(K,)` | int32 | region | yes | node/leaf |
| `path_code` | cluster | `(K,)` | int64 | region | yes | node/leaf |
| `constraints` | cluster | `(K, h, d+1)`, `h` = depth | f64 | region | yes | — |
| `retries` | cluster | `(K,)` | int16 | region | yes | — |

AFINNI mode carries exactly today's columns, so its payloads and results are unchanged.
A rank's queue is homogeneous in depth (Section 31.4), so `constraints` is a dense block
per batch. There is no RNG column (Section 31.9).

**Training and validation are views, not a flag the fits could ignore.** The existing
fit functions and segmented reductions use every row they are given. So within each
cluster, training rows are stored first and validation rows after them
(`n_train` of `sizes`). Each batch gathers a packed training view and a packed
validation view. Fitting, statistics, normalization, threshold selection, centroid
Taylor-residual ranking and split search only ever receive the training view.
Validation outputs never enter them. The two views can share a transfer buffer, but
never a statistical role (review §20.3).

**Spent validation rows.** When a node fails, its validation rows may be released into
its children's training data (`use = train`). They are spent: they never validate a
descendant again.

**Wire format.** Each transfer starts with a schema header: format version, the present
columns and their dtypes and trailing shapes. Then one message per present column, at
`base_tag + offset`, where the offset is the column's position in the table. The
initial scatter and work stealing use the same function with different base tags.
The region columns above are 12 arrays (`X`, `F`, `G` counted separately), so the tag
range is derived from the table, not written as a fixed list (review §20.3). Persisted
leaves, transient jobs and node records share column descriptors, not one payload.

Outputs:

| Store | Columns | Modes |
|---|---|---|
| leaf columns (`ColumnBuffer`, today) | shared columns plus model columns; `Q` optional in region mode; plus `root`, `path_code`, `n_train`, `terminal_reason`, `validation_status` | both |
| node records (new, rank-local, append-only) | `root`, `path_code`, predicate `(w, z)` in normalized coordinates, `status` | region |
| model metadata | resolved plan (`afinni`, or `region` + B-variant), default predictor, domain, normalization `(x₀, s)` | both |

## 37. Pipelines

### 37.1 AFINNI plan: today's order, unchanged

This is the order in `src/adaptive_clustering.py`, which the AFINNI plan must keep,
together with its batching, reductions and RNG calls:

```text
1  load batch                  admission by the existing byte estimate
2  STATS                       centroids, f(c) [, ∇f(c)]
3  FIT                         model.fit
4  ACCEPT (preliminary)        acceptance on the cluster's points
5  accepted:                   Q = trust_region.shapes(...)
                               final = model.finalize(...)
                               if final adds columns: re-evaluate, re-test;
                                  clusters that now fail move to the rejected set
6  accepted survivors          leaf columns
7  rejected                    AUGMENT to max(2n, 2·min_pts) by convex combinations
                               SPLIT: metric direction, median or sign cut; _check_split
                               children (depth + 1) → next queue
8  depth limit                 children beyond max_levels stay unresolved; no leaf
```

### 37.2 Region plan

```text
1  load batch                  admission by the region byte estimate (below)
2  PREPARE_TRAINING_DATA       top up training rows to budget (multiplier·p_eff) by
                               hit-and-run inside the polytope (Section 33.4);
                               draw validation rows separately; rank check
3  STATS (training view)       centroids; f(c) and ∇f(c) only if some component needs them
4  FIT (training view)         model.fit
5  SCREEN (optional)           preliminary acceptance on training rows, to skip
                               finalizing and validating hopeless nodes
6  FINALIZE                    the candidate that will be deployed (+ Q if requested)
7  VALIDATE                    deployed candidate on validation rows → validation_status
8  accepted                    leaf columns, terminal record
9  rejected                    no augmentation (already topped up);
                               release validation rows to children as training rows
10 PROPOSE_SPLIT (train view)  B1/B2: metric direction, gap threshold
                               B3: threshold bins × direction bank
                               B4: + consistent soft refinement
11 VERIFY_SPLIT                predicate reproduces membership; finite plane; minimum
                               training count per side; invalid → terminal,
                               parent model kept, status split_failed
12 COMMIT                      children inherit depth, path_code, root, constraints;
                               node record written
13 depth or budget limit       terminal record with fallback (parent) model and status
```

### 37.3 What region mode changes, as named plan methods

Region mode changes more than two branch points (review §20.3). Each difference is a
small named method of the plan over the shared executor. The AFINNI plan's version of
each method is today's code, unchanged. They extend the modules that already own the
work (Section 38) instead of adding a single-use helper per line.

| Plan method | AFINNI | Region |
|---|---|---|
| `init_roots` | k-means labels or grid labels | box, grid with retained bounds, or Voronoi centres (Section 31.6) |
| `prepare_training_data` | none | top-up, validation draw, training/validation views |
| `required_oracle_quantities` | union of component flags (today) | the same union, over the training view |
| `fit_and_validate_candidate` | steps 3–6 of 37.1 | steps 3–8 of 37.2 |
| `prepare_rejected_nodes` | augmentation | release validation rows; nothing else |
| `propose_and_verify_split` | median or sign cut, `_check_split` | gap or search predicate, membership and support checks |
| `commit_terminal_or_children` | leaf or children; unresolved past depth | node records, terminal statuses, fallback models |
| `estimate_peak_bytes` | today's formula | region footprint (below) |
| assembly and persistence | leaves | leaves plus node records, validated (Section 31.5) |

**Node states.** `ready → sampling → fitted → audit-pending → split-ready → terminal`,
used only to define ownership and recovery. In version one only `ready` nodes are
stealable; all other state stays inside the processing task (review §29.4).

**Centroid requirements are a union (review §20.3).** An affine routed leaf does not
need `f(c)`, but `CovarianceHessianSplit` reads `data.f_c` on both of its gradient
paths, and B1/B2 reuse that metric. Components already declare `needs_centroid_grad`,
`needs_point_grad` and `needs_hessian_fn`, and `config.py` takes their union. Add
`needs_centroid_value` in the same way; STATS computes `f(c)` only when the union asks
for it. Replacing the oracle anchor with a fitted intercept would be a different
algorithm and needs its own comparison.

**Region memory estimate, needed from B2.** The admission mechanism (the VRAM cap and
batch split) is shared, but each plan has its own footprint function. The region
footprint counts top-up rows, validation rows, `constraints` (`K·h·(d+1)`), chain
state, slacks and packing temporaries, as the maximum over overlapping stage
lifetimes (Section 33.6).

**Split support (review §25).** No fractional balance rule. Four separate safeguards:

| Safeguard | First implementation |
|---|---|
| Geometric validity | distinct-score gap, finite plane, feasible inherited seeds, membership reproduced by the stored predicate |
| Statistical support | at least `p_eff` training rows per side plus a configurable margin, a rank/conditioning check, then child top-up and validation |
| Optimization stability (B4) | B3 candidate kept as incumbent; bounded steps, iterations and restarts; `soft_optimizer_stalled` on saturation |
| Resource and progress | depth, sample/oracle and split-attempt budgets; explicit terminal statuses |

A point count is not a volume fraction for inherited or MCMC points, so the support
rule is not a minimum-volume guarantee. Allowing a child with one seed, topping it up
and only then scoring would spend oracle calls to evaluate candidate splits; that
needs its own candidate-evaluation budget and is not part of version one.

## 38. Code changes, file by file

| File | Change |
|---|---|
| `src/config.py` | `algorithm` and `domain` fields; a `RegionSettings` group: `roots` (`box`, `grid`, `voronoi`), `sample_multiplier`, `validation_samples`, `audit_samples`, `min_side_margin`, `split_search` (`gap`, `bins`, `soft`), `n_bins`, `n_directions`, `soft_iters`, sampler settings (`n_chains`, `pilot_block`, `geometry_budget`), region steal thresholds; `requires_f_centroids` union next to the existing `requires_grad_*`; validation that region components are compatible |
| `src/clusters.py` | the internal column table; `ClusterBatch`, `ClusterQueueBuilder`, `split`, `select` iterate over it; training-first row order and the training/validation views; `NodeRecords` (append-only, rank-local) |
| `src/work_stealing.py` | `send_clusters`/`recv_clusters` iterate over the table: schema header, then `base_tag + offset` per present column (Section 36). Region-mode thresholds; receiver lease (no re-offering before processing). Request/grant/retire logic unchanged |
| `src/region.py` (new) | domain and normalization; root construction (`box`; `grid` from `uniform_grid_cluster_labels` keeping bounds; `voronoi` from k-means centres); gap-threshold predicate and membership check; constraint inheritance; the region plan methods (Section 37.3); terminal records; assembly and validation of node records; the router |
| `src/splitting.py` | split strategies return `SplitResult(order, child_sizes, w, z, valid)`; AFINNI's strategies leave `w, z` empty. New: `GapThresholdSplit` (B1/B2), `ThresholdSearchSplit` (B3, with direct recomputation of leading candidates), soft refinement (B4, later). `needs_centroid_value` on every strategy |
| `src/local_models.py` | `needs_centroid_value` flag next to the existing gradient flags |
| `src/sampling.py` | `HitAndRunSampler` (batched, several chains per node, slack updates, box as coordinate bounds, pilot/production segments, tiled Voronoi chords); a budget rule per mode (a function, not a component registry) |
| `src/acceptance.py` | `PointwiseViolation` criterion (optional; also used for the global audit) |
| `src/numerics.py` | `a = Mv` projection for region mode; binned prefix statistics for threshold search; an explicit-status solver with fixed regularization and residual output (B4) |
| `src/adaptive_clustering.py` | becomes the executor: the AFINNI plan is today's loop body; the region plan's methods are called at the same points (Section 37.3); collecting and gathering `NodeRecords` |
| `src/surrogate.py` | `predict(..., predictor="knn" \| "route")` with structured statuses for `route`; persist the resolved plan and default predictor; a model saved without `Q` defaults to routing and rejects an explicit `predictor="knn"`; save/load node records (bumps `MODEL_FORMAT_VERSION`); an `audit()` that runs the global audit (Section 33.5) |
| `src/kmeans.py` | return the centres (and the CPU centring offset) for Voronoi roots |

The `Predictor` and `SampleBudget` "registries" proposed earlier are replaced by a
`predictor=` option and a budget rule chosen by the mode. They can become components
later, if a third algorithm needs them.

## 39. Deliberately not built now

Status: still deferred, with triggers, in Section 42.6. Progress points inside long
loops, B4 and Voronoi roots have since been built.

- user-defined properties, the general stage/dependency scheduler, and the public
  primitives API (Appendix A);
- polynomial leaves beyond affine, except as a later B4 variant with a memory
  estimator;
- migrating partially processed nodes (mid-node chain state, attempt counters);
- sample weights, per-sample ids, global oracle budgets;
- calibrated time-based steal estimates and makespan-based grants (the minimal policy
  in Section 33.7 *is* built), and progress points inside long loops until B4;
- ancestry interning, device-resident descendants, host spilling;
- distributed assembly;
- per-node probabilistic certificates (they need proven mixing bounds), sequential
  tests;
- cached-face Voronoi chord certification, unless root faces are measured to be the
  bottleneck;
- boundary-band blending, continuity-constrained fits, and routed candidate reranking.

Each is in Part IV with the condition that would justify it.

## 40. Implementation order

Status: phases 1–5 are implemented (Section 42.1); phase 6 is Section 42.6.

1. **Baseline and contracts.**
   - Freeze a regression reference.
   - Define the domain and normalization, the tie and ownership conventions, the
     terminal and validation statuses, the accuracy metric (cluster-normalized versus
     pointwise), the RNG indexing and the compatibility rules.
   - Introduce the internal column table and the versioned wire format, and turn the
     loop into an executor with the AFINNI plan, keeping it bit-identical.
2. **Explicit tree with cheap splits (B1).**
   - Record valid predicates for a tree sampled by AFINNI's augmentation with `Q` kept, and compare
     `predictor="knn"` and `"route"` on that same artifact.
   - `algorithm="region"` with box and grid roots, gap-threshold predicates, node
     records, assembly, and a CPU reference router, then the GPU router. Report grid
     roots as B1-grid, a new baseline.
3. **Samples that support the routed regions (B2).**
   - Training/validation views, pre-fit top-up, validation rows, the region memory
     estimate.
   - A batched hit-and-run sampler with pilot/production segments, checked against
     low-dimensional reference sampling.
   - Optional `Q`, and `f(c)` only when the requirement union asks for it.
   - The global audit, and region-mode steal thresholds with forced-transfer tests.
4. **Hard split search (B3).** Threshold bins, a direction bank, direct recomputation of
   leading candidates.
5. **Consistent soft refinement (B4).** Finite-difference tests first; residual-based
   descent checks; ART-style fixed slots; warm start from the B3 candidate, which is
   kept as the fallback; bounded progress points for steal requests.
6. **Only if profiles demand it:** the items in Section 39.

## 41. Open questions

### 41.1 Resolved in the first review

| # | Question | Resolution |
|---|---|---|
| 1 | Single box root or Voronoi roots? | **Resolved: three options.** The box is the mathematical reference; a shallow hierarchy (new option) gives early GPU/MPI concurrency; Voronoi roots for compatibility experiments. See Section 31.6. |
| 2 | Sign or median cut for AFINNI's own hyperplanes? | **Resolved.** Neither as such: a valid geometric predicate at a score gap, with a minimum side count, chosen by a small hard threshold search. Exact count balance is subordinate to representable membership. See Sections 31.7 and 33.2. |
| 3 | How should the `fd` outputs combine in the split objective? | **Resolved (initial choice).** Fixed tolerance-derived output scales and the sum of squared normalized residuals for split proposals; acceptance stays per component; test worst-component failures explicitly. Derivative-based direction candidates should cover the relevant scaled components (review §27.3). |
| 4 | Carried constraint block or per-batch gather, and when to switch? | **Resolved.** Carry for the first version; measure `h/n`, prefix duplication and migration bytes; then use a rank-local immutable ancestry store with deduplicated migration. The switch is decided by measurement, not by a `d·depth` threshold. |
| 5 | Who writes a stolen cluster's tree records? | **Resolved.** The current processing owner. Stable logical keys; compact indices assigned at assembly; committed records never move back. |
| 6 | What does "outside" mean under routing? | **Resolved.** Outside the domain only. Unresolved termination and empirical `Q` support are reported as separate statuses; AFINNI's flag keeps its meaning for the kNN predictor. |
| 7 | Holdout or confidence-based acceptance? | **Resolved with Q-A.** Fresh per-node validation of the deployed model as a diagnostic; the statistical claim comes from the global audit. |

### 41.2 Resolved in the second review (Q-A to Q-I)

| Q | Question | Resolution |
|---|---|---|
| Q-A | What can validation claim under MCMC dependence and adaptive testing? | **Resolved (review §21).** Version one claims a global violation rate for the frozen deployed predictor from fresh iid audit points (`m = 299` for < 1%, `2,995` for < 0.1%, at 95%, zero failures). Per-node MCMC validation is a diagnostic. ESS is not a substitute for a joint total-variation bound `η`, which would give `1 − (δ − η)^{1/m}`. Adaptive cell choice is fine with fresh conditional tests and an allocation over attempted tests. Sequential tests later, only with iid samples. See Section 33.5. |
| Q-B | How long should child chains warm up? | **Resolved (review §22).** No dimension-only rule exists. Dispersed starts, a geometry pilot in doubling blocks with several projection and slack diagnostics, a work cap ending in `insufficient_sampling`, and a separate production segment. See Section 33.4. |
| Q-C | Exact Voronoi faces cheaply? | **Resolved (review §23).** Not needed: store centres and compute chords in tiles against all of them. Optional cached-face chord with endpoint certification, exact when both endpoints belong to the root's centre. Low-dimensional Voronoi/Delaunay adjacency is another option, measured rather than limited by a dimension cutoff. LP redundancy removal needs dual bounds and sequential removal. See Sections 31.6 and 33.4. |
| Q-D | Majorization with floors and finite precision? | **Resolved (review §24).** The exact gap `tr(RᵀH⁻¹R)/n` from recomputed normal-equation residuals, plus loss-evaluation errors, sets the margin a step must beat; refit trials only as a bounded fallback; a ridge change restarts the optimization. See Section 33.3. |
| Q-E | Minimum side count instead of a fraction? | **Resolved (review §25).** Drop the fraction; keep four separate safeguards. Count plus rank does not prevent saturated gates during optimization; that is handled by the incumbent and the `soft_optimizer_stalled` status. See Section 37.3. |
| Q-F | Calibrating steal costs and preventing bouncing? | **Resolved (review §26).** Version one: region-mode thresholds, static cost weights, a receiver lease, backoff, and liveness tracked by job count. Later: makespan-based grants with per-phase throughput estimates. See Section 33.7. |
| Q-G | Covariance-Hessian on saddles? | **Resolved (review §27).** Exact `|H|` is itself isotropic on `x₁x₂`, and the signed diagonal cut loses to an axis cut, so don't buy derivatives. Use the direction bank with hard fit scoring, tied-eigenspace candidates, and test an interaction feature in the leaf. See Section 33.2. |
| Q-H | Is a capped blend enough for continuity? | **Resolved (review §28).** No; capping can drop nonzero weights, and narrow bands inflate derivatives by `ε/b`. Hard routing as the reference with boundary and trajectory tests; continuity becomes a requirement only where the application needs it (vector fields, differentiable loops). See Section 31.8. |
| Q-I | Two fixed pipelines or one with per-stage options? | **Resolved (review §29).** Two named plans over one executor and shared numerical modules; no public stage permutations; plan and predictor persisted with the model. See Sections 35 and 37. |

### 41.3 Still open: needs measurements, not more design

First measurements are in Section 42.3 (defaults: `split_search="soft"`, multiplier 50,
no validation points, margin 0). The items below still need a sweep.

- The minimum training-count margin per side, and the validation and audit sample
  counts, at equal oracle budget.
- Persistent chains versus one endpoint per chain, and pilot block sizes, on the
  difficult cells of Section 34's problem families.
- Whether carried constraints or Voronoi root faces are bottlenecks worth the deferred
  optimizations.
- Whether routing without blending is accurate enough on the flow-map benchmark.
- Region-mode steal thresholds and grant sizes, from measured request latency.

## 42. Implementation status (branch `region-mode`, Nov 11, 2026)

Every phase of Section 40 has a working implementation on branch `region-mode`. This
does not mean every contract of Sections 33–38 holds: where the code is simpler it is
described in 42.2, the guarantees actually established are in 42.3, and the open
resource and sampling contracts are in 42.6 and 42.7. CPU and MPI (1, 2, 4 and 8 ranks,
stealing on and off) are verified on a 16-core laptop; the GPU has not been run and is
the last step (Section 42.5).

### 42.1 What was built

| Plan item | Where | Notes |
|---|---|---|
| Internal column table, versioned wire format | `src/clusters.py` (`QUEUE_COLUMNS`), `src/work_stealing.py` | `ClusterBatch.split`/`select`, the queue builder and `send/recv_clusters` iterate over the table. The header carries version, present columns, shapes and dtypes; column `i` goes on `base_tag + 1 + i`, so an AFINNI-mode transfer sends the same array payloads on the same tags as before. The header itself is new, so this is not wire compatibility with an older executable: mixed versions fail on the version check. |
| One executor, two named plans | `src/adaptive_clustering.py`, `src/region.py` (`RegionPlan`) | The AFINNI batch body is unchanged in order and arithmetic. The region plan supplies `work_sizes`, `bytes_per_point`, `leaf_columns`, `split_columns`, `initial_queue`, `process_batch`. Shared deploy step: `deploy_leaves` in `src/local_models.py` (Q, finalize, re-test), called by both plans. |
| `algorithm`, `domain`, `RegionSettings` | `src/config.py` | `domain=(lower, upper)` required; `"data_bbox"` is explicit opt-in and recorded. |
| Roots | `src/region.py` (`Roots`) | `box` (default, ART: no initial clustering), `grid` (shared `uniform_grid_bins`), `voronoi` (centres of the initial clustering; exact nearest-centre assignment, tiled; no face list; hit-and-run bisectors from centre differences `c_j − c_r`, normalized; root assignment re-decides the nearest-distance candidate with the same centre-difference test, so sampling, training and routing agree on thin cells. Points within rounding of a bisector remain ambiguous: there is no error-bounded tie test). |
| Counter-based RNG | `node_keys`, `counter_uniform`, `counter_uniform_pairs` | splitmix64 of (seed, root, path code, purpose, draw index); the same on NumPy and CuPy. The sampler takes two 32-bit uniforms per hash and both normals of each Box-Muller pair: `ceil(d/2) + 1` hashes per hit-and-run step. |
| Pre-fit top-up, validation rows, drift and movement checks | `HitAndRunSampler` | Box/grid roots sampled exactly; elsewhere one chain per new point, continued (doubling up to `max_walk_steps`) while its endpoints drift. There is no separate pilot and production segment: the diagnostic decides how long the same returned chains run, which is weaker than Section 33.4's scheme. Every chain that kept any coordinate at its start value (it has not left the start's face) is rerun with systematic coordinate sweeps on its own stream (at least two sweeps when `max_walk_steps` allows; in a box one sweep is exact), through the same drift check computed on the rerun chains only; nodes still drifting, with any chain still on its start face, or with an infeasible step or endpoint get `insufficient_sampling`. Coordinate moves cannot leave every corner of an oblique polytope; such nodes are flagged, not repaired. That recovery depends on the first walk's outcome: it is an adaptive repair, not an independent production run. Drift statistics are centred moments. Validation points are a diagnostic. |
| Split predicates | `RegionPlan._split`, `_propose` | `split_search="gap"` (B2), `"bins"` (B3: `n_bins` cuts × `n_directions` directions; cut positions depend only on a node's count, so per direction each node is sorted once, each bin's Gram is one batched product over its contiguous rows, and both sides of every cut are prefix and suffix sums over bins, the right side summed directly rather than subtracted from the total; one data pass per direction instead of one per cut; the sums are a screen and the `n_screened` best are recomputed directly), `"soft"` (B4, the default: consistent soft objective, envelope gradient, frozen Armijo, a margin `tr(RᵀH⁻¹R)/n` with `R` from the data residual, hardening, kept only if the direct hard SSE beats B3). B4 is numerically monitored descent with a hard incumbent, not a certified optimum: the margin is an estimate, not an error bound. Each refined node records one termination reason (`soft_gradient_converged`, `soft_line_search_stalled`, `soft_gate_saturated`, `soft_nonfinite`, `soft_iteration_limit`); with the original 20 iterations nearly every node stopped at the iteration limit, so the default is now 40 (Section 42.8). `rank_deficient` counts nodes entering B3/B4 whose point covariance has an eigenvalue ratio `λ_min/λ_max ≤ 1e-12` (singular-value ratio `≤ 1e-6`); it is a diagnostic only, not a check of every fit or child, and B2 does not compute it. B4's temperature uses the centred score variance, floored at the scores' rounding resolution. |
| B1 | `sampling="convex_combination"` | AFINNI's augmentation on (usually Voronoi) roots, `Q` kept, so both predictors run on one tree. |
| Terminal statuses and fallback leaves | `TERMINAL_REASONS`, `VALIDATION_STATUSES` | `accepted`, `max_depth`, `split_failed`; validation `not_performed`, `passed`, `failed`, `insufficient_sampling`. Every point of the domain has a leaf. |
| Assembly and routing | `RoutingTable` | Compact node arrays (child indices, planes, leaf rows): routing follows child indices, so any depth up to 62 routes. Building and loading check shapes, finite planes, unique `(root, path_code)` provenance, both children of every split, parents and roots; queries must be `(n, d)` and finite. Older files with packed keys still load. |
| Predictors, persistence, audit | `src/surrogate.py` | `predict(predictor="knn" \| "route")` (route is the default for region models), `route_leaves`, `route_details` (per query: leaf, outside, terminal reason and validation status), `leaf_status_counts`, `audit` (iid uniform on the domain, Clopper–Pearson bound; non-finite oracle or model values count as violations), model format 2; `store_Q=False` models refuse kNN. Training warns when leaves end with `insufficient_sampling`: `converged` only says that no node is still above tolerance. |
| Quadratic leaves | `QuadraticLSQModel` (`"quadratic_lsq"`), `LocalModel.n_features` / `features` | ART's degree-2 leaves: full quadratic in node-centred, radius-scaled coordinates, tiny fixed ridge. The region plan takes the sample budget (`multiplier × n_features`) and the split basis from the model; affine trees are unchanged. |
| `f(c)` only when needed | `needs_centroid_value` on models and split strategies, `config.requires_f_centroids` | The leaf `f` column is NaN when no component reads it. |
| Pointwise acceptance | `src/acceptance.py` (`"pointwise"`) | ART's `max_pointwise_relative` with `rtol` alone. |
| Stealing for region work | `WorkStealer(min_load, steal_size, lease_received, work_sizes)`, `progress()` | Region thresholds (a few nodes' worth), receiver lease (`LEASED` denials), and progress points inside B4 so requests are served mid-batch. Published loads, steal eligibility and grant sizes all use the plan's `work_sizes` (points after top-up and validation), so thieves and victims compare one unit. |
| Experiment harnesses | `scripts/art_comparison.py`, `scripts/afinni_benchmark.py` | ART vs region on ART's benchmarks, scored by ART's own evaluator (ART's configuration, e.g. its fastest CPU sampler `parallel_numba`, recorded in the output); AFINNI mode, base vs branch accuracy and runtime. |
| Example and benchmark problems | `examples/example04_region_mode.py`, `src/problems/spherical_piecewise.py` | Both modes on four problems (`--problem all` or one): ART's spherical piecewise preset (identical to ART's function in 2–5-D), anisotropic Gaussian, wave, Rosenbrock. Per-problem depth limits (and region rtol 0.02 for wave) give equal relative L2 in one process on one CPU; region mode then needs 3.5–18x fewer oracle calls at similar or lower error and fewer leaves on all but the spherical jump (where it is twice as accurate), AFINNI trains as fast or faster (table in `examples/USAGE.md`). Stages `afinni`, `region`, `compare`; `summary.png` across problems. |
| Plots | `src/plotting/region.py` | `plot_region_partition`, `plot_region_tree_stats`, `plot_error_maps`, `plot_error_cdf`, `plot_method_comparison`; profile dashboards name the mode and show fallback leaves per level. |
| Monitor | `src/monitor.py`, `run_start` event | The header names the mode (`AFINNI` or `AFINNI region mode (ART; …)`); region runs report points in fallback leaves instead of queued points. Region stages are timed under the shared phase keys, so profiles break down by phase. |
| Documentation | `examples/USAGE.md` ("AFINNI region mode (ART)", plots, monitor), `README.md` | |

### 42.2 Simplifications against Sections 36–38

- Validation points never enter the queue: they are drawn in `process_batch`, used, and
  either dropped (leaf) or given to the children as training points. So the queue needs
  no `use`, `origin` or `n_train` columns, and no `retries` column while each node draws
  each purpose once (attempt 0).
- The hit-and-run check compares halfway and final endpoints (coordinates and squared
  radius), plus a movement check (Section 42.7). R̂/ESS diagnostics and projections on
  cut normals (Section 33.4) would be extensions of `_drifting`.
- B2–B4 fit children in the leaf model's own basis (`LocalModel.features`: `[u, 1]` for
  affine leaves, node-centred quadratic monomials for `quadratic_lsq`), as ART fits
  children in its leaf design; B4 uses a temperature `soft_temperature * std(wᵀu)` and
  one fixed ridge per node.
- B1 uses a per-batch random stream for its convex combinations, as AFINNI mode
  does, so only B1 depends on the batch size.

### 42.3 Verification on this machine (CPU, MPI)

- **AFINNI-mode bit-identity:** `scripts/regression_check.py` on all 45 cases, bit-identical
  to the base commit except `extra_tiny_batches` (the bug fix below); bit-identical to
  the post-fix reference.
- **AFINNI-mode accuracy and runtime** (`scripts/afinni_benchmark.py`, base `015adfd` vs
  branch): leaves, evaluations, pout, in-tolerance share and errors identical on
  Rosenbrock d=4, vector quadratic d=6 and anisotropic Gaussian d=4, serial and with 4
  MPI ranks (stealing off and on). Serial medians 12.27/12.41 s, 43.9/44.4 s,
  1.82/1.84 s; alternating runs overlap (Rosenbrock 12.05–12.79 s base, 12.06–12.72 s
  branch); 4-rank runs within 1–2%.
- **Tests:** `pytest -m "not mpi"`: 472 passed, 35 skipped (432 before the branch).
  `tests/mpi` at 2, 4, 8 ranks: 12 passed (one AFINNI-mode test relaxed, see 42.4). Region
  tests cover the RNG, roots, exact and hit-and-run sampling (triangle mean), the drift
  check, gap cuts with ties, binned statistics, the B4 gradient against finite
  differences, B4 never worse than B3, routing and its validation, the wire format,
  batch-size independence of B2/B3/B4 trees, Voronoi and B1, `store_Q=False`, the audit,
  save/load and configuration errors. The MPI test checks identical trees and
  predictions for any rank count, with forced steals, including B4 (10 mid-batch grants
  through `progress()` in a 4-rank run).
- **ART comparison** (`scripts/art_comparison.py`). ART defaults: tolerance 1e-2 on the
  pointwise relative error, max depth 10, multiplier 50, the `cpu` soft optimizer (up to
  200 iterations per split), DFS. Every run is scored on ART's test set of `10000·d`
  uniform points by ART's own evaluator. Each cell is relative L2 error / share of test
  points above the tolerance / leaves / oracle queries / build time, one seed. All cells
  were re-measured in one session on the same 16-core laptop (on battery, balanced power
  profile), with the current code: bin-sum B3, B4 with 40 iterations (Section 42.8) and
  every review fix of 42.7. ART's code is unchanged.

  **Affine leaves** (ART `--leaf-degree 1`, region `affine_lsq`). ART is run with its
  default sampler (`hit_and_run`, the baseline) and with `--sampler parallel_numba`, its
  fastest CPU-only configuration (ART's other optimizers need a GPU).

| Problem | ART baseline (`hit_and_run`) | ART fastest CPU (`parallel_numba`) | Region B4 (default) | Region B3 | Region B2 |
|---|---|---|---|---|---|
| quadratic 2D | 2.6e-03 / 6.7% / 415 / 62k / 23 s | 2.6e-03 / 7.1% / 401 / 60k / 13 s | 2.2e-03 / 5.4% / 404 / 61k / 1.6 s | 2.3e-03 / 5.2% / 414 / 63k / 0.57 s | 2.0e-03 / 4.9% / 512 / 78k / 0.59 s |
| quadratic 3D | 3.8e-03 / 28% / 717 / 143k / 46 s | 4.1e-03 / 30% / 715 / 143k / 24 s | 3.9e-03 / 28% / 702 / 142k / 3.7 s | 4.3e-03 / 30% / 752 / 152k / 1.6 s | 3.5e-03 / 25% / 948 / 191k / 1.9 s |
| gaussian 2D | 2.4e-03 / 0.06% / 22 / 3k / 0.88 s | 2.6e-03 / 0.005% / 22 / 3k / 0.52 s | 3.0e-03 / 0.36% / 18 / 3k / 0.11 s | 2.3e-03 / 0.07% / 25 / 4k / 0.1 s | 2.3e-03 / 0.04% / 28 / 4k / 0.094 s |
| gaussian 3D | 2.5e-03 / 0.17% / 139 / 28k / 6.8 s | 2.8e-03 / 0.31% / 120 / 24k / 2.3 s | 2.8e-03 / 0.38% / 114 / 23k / 0.58 s | 2.6e-03 / 0.3% / 160 / 32k / 0.37 s | 2.4e-03 / 0.19% / 210 / 42k / 0.42 s |
| plane wave 2D | 0.018 / 39% / 931 / 140k / 56 s | 0.015 / 33% / 882 / 132k / 32 s | 2.5e-03 / 2.2% / 249 / 38k / 0.95 s | 4.2e-03 / 11% / 501 / 76k / 0.66 s | 7.0e-03 / 11% / 976 / 148k / 1.2 s |
| rosenbrock 2D | 2.6e-03 / 31% / 471 / 71k / 28 s | 2.3e-03 / 28% / 450 / 68k / 16 s | 2.6e-03 / 29% / 351 / 53k / 1.4 s | 2.6e-03 / 32% / 347 / 53k / 0.49 s | 2.2e-03 / 14% / 798 / 121k / 0.93 s |
| spherical piecewise 2D | 0.047 / 2.5% / 226 / 34k / 14 s | 0.074 / 2.4% / 148 / 22k / 6 s | 0.056 / 1.2% / 188 / 29k / 0.82 s | 0.045 / 1.3% / 218 / 33k / 0.34 s | 0.067 / 13% / 268 / 41k / 0.34 s |
| spherical piecewise 3D | 0.11 / 26% / 333 / 67k / 24 s | 0.11 / 25% / 320 / 64k / 12 s | 0.09 / 14% / 502 / 101k / 2.6 s | 0.1 / 22% / 520 / 105k / 1.2 s | 0.18 / 61% / 812 / 164k / 1.6 s |
| plane wave 3D | 0.14 / 88% / 1,024 / 205k / 75 s | 0.12 / 88% / 1,024 / 205k / 41 s | 0.048 / 55% / 887 / 179k / 4.6 s | 0.1 / 77% / 952 / 192k / 2.1 s | 0.056 / 70% / 1,023 / 207k / 2.1 s |
| rosenbrock 5D, [-2,3]⁵, offset 1 | 0.06 / 89% / 1,024 / 307k / 98 s | 0.063 / 89% / 1,024 / 307k / 41 s | 0.054 / 87% / 1,024 / 309k / 8.5 s | 0.07 / 90% / 1,024 / 309k / 3.9 s | 0.081 / 90% / 1,024 / 309k / 3.5 s |
| rosenbrock 20D, [-2,3]²⁰, offset 1 | 0.26 / 97% / 1,024 / 1.08M / 322 s | 0.27 / 97% / 1,024 / 1.08M / 135 s | 0.28 / 97% / 1,024 / 1.08M / 69 s | 0.29 / 97% / 1,024 / 1.08M / 34 s | 0.28 / 97% / 1,024 / 1.08M / 30 s |

  Against ART's fastest CPU configuration, region B4 builds 3.9–34× faster for `d <= 5`
  (median 6.5× over all rows) and 2× faster at 20-D, where per-node arithmetic
  (`n p^2 ~ d^3` under ART's budget) replaces the per-node call overhead that batching
  saves. Oracle queries and leaves are similar or lower. Accuracy, one seed each:
  - B4 is more accurate than both ART runs on 2-D plane wave (L2 6× lower, 2.2% of points
    above tolerance against 33–39%), 3-D plane wave (0.048 / 55% against 0.12–0.14 / 88%),
    3-D spherical piecewise (0.09 / 14% against 0.11 / 25%), 2-D quadratic and 5-D Rosenbrock;
  - comparable on 3-D quadratic, 2-D Rosenbrock and 20-D Rosenbrock; on 2-D spherical
    piecewise its L2 (0.056) lies between ART's two runs with half their share above
    tolerance;
  - behind on the Gaussians: 0.36% (2-D) and 0.38% (3-D) of points above tolerance against
    ART's 0.005–0.31%, with fewer leaves.

  B3 (no soft optimizer) is 1.1–3× faster than B4 and the fairer reference for speed; B4's
  40 iterations buy most of its advantage on the oblique-interface problems (plane waves).
  B2 (curvature direction at the median) fails on interfaces, as Section 9.5 predicted.
  The 3-D plane wave remains stream-sensitive (B4's L2 was 0.023–0.20 across streams with
  20 iterations). Oracle counts include centroid evaluations when the split strategy reads
  `f(c)` (the default covariance-Hessian metric does).

  **20-D** (last row): nothing reaches the tolerance with 1,024 affine leaves, so every run
  fills the depth-10 tree at the same accuracy and the row compares cost only.

  **Quadratic leaves** (ART `--leaf-degree 2 --sampler parallel_numba`, region
  `quadratic_lsq`; both sample `50 × (d+1)(d+2)/2` points per node and fit children in the
  quadratic basis). The last column repeats region B4 with affine leaves for reference.
  20-D is omitted (231 coefficients per leaf).

| Problem | ART fastest CPU (`parallel_numba`) | Region B4 | Region B3 | Region B4, affine leaves |
|---|---|---|---|---|
| quadratic 2D | 7.6e-16 / 0% / 1 / 300 / 0.46 s | 1.9e-10 / 0% / 1 / 301 / 0.0026 s | 1.9e-10 / 0% / 1 / 301 / 0.06 s | 2.2e-03 / 5.4% / 404 / 61k / 1.6 s |
| quadratic 3D | 1.3e-15 / 0% / 1 / 500 / 0.47 s | 2.1e-10 / 0% / 1 / 501 / 0.0026 s | 2.1e-10 / 0% / 1 / 501 / 0.059 s | 3.9e-03 / 28% / 702 / 142k / 3.7 s |
| gaussian 2D | 9.6e-04 / 0.15% / 5 / 2k / 0.55 s | 7.3e-04 / 0% / 6 / 2k / 0.063 s | 8.5e-04 / 0.005% / 5 / 2k / 0.077 s | 3.0e-03 / 0.36% / 18 / 3k / 0.11 s |
| gaussian 3D | 1.1e-03 / 0.07% / 9 / 4k / 0.61 s | 9.6e-04 / 0.013% / 10 / 5k / 0.15 s | 1.1e-03 / 0.14% / 10 / 5k / 0.11 s | 2.8e-03 / 0.38% / 114 / 23k / 0.58 s |
| plane wave 2D | 1.8e-03 / 2.3% / 530 / 159k / 18 s | 1.0e-03 / 0.035% / 141 / 43k / 1.1 s | 8.7e-04 / 0.06% / 243 / 73k / 0.62 s | 2.5e-03 / 2.2% / 249 / 38k / 0.95 s |
| rosenbrock 2D | 8.6e-04 / 1.4% / 245 / 74k / 10 s | 7.4e-04 / 0.44% / 324 / 98k / 2.4 s | 6.1e-04 / 0.74% / 267 / 81k / 0.71 s | 2.6e-03 / 29% / 351 / 53k / 1.4 s |
| spherical piecewise 2D | 0.028 / 3.1% / 217 / 65k / 11 s | 0.05 / 1.5% / 231 / 70k / 1.8 s | 0.036 / 1.9% / 209 / 63k / 0.57 s | 0.056 / 1.2% / 188 / 29k / 0.82 s |
| spherical piecewise 3D | 0.077 / 22% / 490 / 245k / 24 s | 0.069 / 18% / 462 / 232k / 6.6 s | 0.081 / 28% / 427 / 214k / 2.5 s | 0.09 / 14% / 502 / 101k / 2.6 s |
| plane wave 3D | 0.016 / 29% / 951 / 476k / 45 s | 1.2e-03 / 0.19% / 340 / 171k / 4.5 s | 0.075 / 22% / 747 / 375k / 4.2 s | 0.048 / 55% / 887 / 179k / 4.6 s |
| rosenbrock 5D, [-2,3]⁵, offset 1 | 9.4e-03 / 42% / 1,021 / 1.07M / 65 s | 8.6e-03 / 39% / 1,024 / 1.08M / 46 s | 0.017 / 59% / 1,024 / 1.08M / 16 s | 0.054 / 87% / 1,024 / 309k / 8.5 s |

  With quadratic leaves region B4 is as accurate as ART or more on every row except the L2
  of 2-D spherical piecewise (0.05 against 0.028, with fewer points above tolerance), and
  much more accurate on the plane waves: 2-D 141 leaves and 0.035% above tolerance against
  530 and 2.3%; 3-D 340 leaves, L2 1.2e-3 and 0.19% against 951, 0.016 and 29%. It builds
  3.6–16× faster for `d <= 3` (the one-leaf quadratics aside) but only 1.4× at 5-D, where
  B4's per-node work grows with `p^2 = 441`; B3 is 2.9× faster than B4 there, at a lower
  accuracy (59% above tolerance against 39%). Both methods represent the quadratics exactly with one leaf (region mode's
  1e-10 ridge leaves an L2 of 2e-10). Against affine leaves, quadratic leaves cut the share
  above tolerance on every smooth row, e.g. 5-D Rosenbrock 87% → 39% and 3-D plane wave 55% →
  0.19%, at up to 3.5× more oracle queries (the larger per-node budget).

### 42.4 Found while implementing

- **Fixed, region mode at high dimension:** B3 and B4 first formed every weighted Gram by
  materializing per-point outer products (`N p^2` numbers, about 1.9 GB per deep level at
  `d = 20`) and summing them, against Section 33's own rule. At 20-D that made B4 2.5×
  slower than ART (411 s). Grams are now batched matrix products over nodes grouped by point
  count (`_SplitRows.cross`), with no padding, so each node's arithmetic stays independent of
  its batch: B4 97 s, B3 63 s, unchanged at low dimension. The table in 42.3 was
  re-measured after this change and the sampler change below.
- **Faster sampling, statistics unchanged:** the hit-and-run sampler used `2d + 1`
  hashed uniforms per step and selected the box chord with nested `where`. It now takes two
  32-bit uniforms per hash and both Box-Muller normals per pair (about 4x fewer hashes at
  20-D), and bounds the box chord with `fmin`/`fmax` (bit-identical to the selection). The
  new random stream was accepted only after checking that the statistics did not change:
  against exact rejection samples in 2-D, 3-D and 7-D polytopes the two samplers have the
  same Kolmogorov-Smirnov distances (2-D, 3-D at the noise level; 7-D from packed seeds
  0.095 +- 0.007 before, 0.096 +- 0.005 after over five streams); on five of ART's benchmarks
  over three seeds every metric lies within the seed-to-seed spread, and the one borderline
  case (3-D Gaussian leaves) disappears with sixteen seeds (L2 0.002609 for both, p = 0.96;
  leaves p = 0.21). At 20-D, sampling fell from 40 s to 27 s, B4 from 97 s to 79 s.

- **Fixed, pre-existing bug in AFINNI mode:** a cluster larger than `batch_size` ended its
  level's batch loop, silently dropping every later cluster of that level while the run
  still reported `converged` (4,000 points with `batch_size=97`: 23 leaves instead of
  385). Now such a cluster is processed alone and the batch grid is re-anchored after
  it; batches are unchanged whenever the old code processed everything. Only the
  regression case `extra_tiny_batches` changes (1 leaf before, 7 after). The GPU batch
  cap (free VRAM, floor 50,000 points) could trigger it on large clusters.
- **Pre-existing test failure, resolved by relaxing the test:**
  `test_multi_rank_matches_serial` failed on the base commit too, because AFINNI's
  augmentation seed (`seed + level + batch + rank·1000`) makes multi-rank trees differ
  from serial ones (235 leaves serial, 239–252 on 2–8 ranks, the same maximum error
  0.081). Changing the seeding would break AFINNI's bit-identity contract (Section 35),
  so the test is now `test_multi_rank_agrees_with_serial_in_accuracy`: error within the
  training tolerance and within 25% of serial, leaf count within 15%. Region mode keeps
  its exact multi-rank check.

### 42.5 GPU test plan (to run last)

Run from the repository root on a CUDA machine with the GPU extra installed.

1. **Backends detected:** `uv run python -m src` reports CuPy.
2. **Unit and integration tests:**
   `AFINNI_DEVICE=gpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"`.
   Region tests compare predictions with `assert_backend_equal(..., separate_runs=True)`.
   The batch-size independence tests also compare leaf and evaluation counts exactly; on
   the GPU, `atomicAdd` inside the split metric (`run_fused_stats`) can reorder rounding
   between batch compositions and flip a near-tied B3/B4 candidate. Prediction tolerance
   and tree-decision equality are separate contracts, and a different leaf count is a
   different tree, not one "equal to rounding". If a decision test fails, do not loosen
   it: record the first divergent decision (node, candidate scores and their margin),
   check the margin is at rounding level, and then choose and document a policy (for
   example, deterministic per-node reductions for split statistics, or GPU trees that are
   reproducible per batch size only).
3. **AFINNI-mode regression:** save a GPU reference on the base commit (a worktree of
   `015adfd`) and compare:
   `AFINNI_DEVICE=gpu AFINNI_MPI=off PYTHONPATH=. python scripts/regression_check.py --save /tmp/gpu_ref`
   (base), then on the branch `... --save /tmp/gpu_new --compare /tmp/gpu_ref --rtol 1e-12`.
   Expected: every case passes except `extra_tiny_batches` (the bug fix).
4. **AFINNI-mode accuracy and runtime:** `AFINNI_DEVICE=gpu AFINNI_MPI=off PYTHONPATH=. python
   scripts/afinni_benchmark.py` in both checkouts, alternating runs. Leaves, evaluations
   and pout identical; errors equal to rounding; times within noise.
5. **Region mode against the CPU numbers:** `AFINNI_DEVICE=gpu ... python
   scripts/art_comparison.py --art-path ../Adaptive-Regression-Trees-ART- --skip-art
   --benchmark <b> --dimension <d>` for the rows of 42.3. Expect metrics close to the
   CPU table (not bit-equal) and the time to scale with batch size.
6. **Device code paths exercised only on GPU:** `segment_sum` (2-D `add.reduceat`),
   `_reduce_max_1d`, uint64 arithmetic in `counter_uniform`, batched `linalg.solve` on
   (k, B, p, p) arrays, Voronoi `root_of` tiles, `searchsorted` routing, and the region
   batch cap (`RegionPlan.bytes_per_point`, floor 1,000 points). Watch peak VRAM on
   rosenbrock 5D with `split_search="soft"`; an out-of-memory error means the footprint
   estimate is too low.
7. **MPI with a GPU:** one GPU rank per GPU (Section 33.7); the multi-rank protocol is
   already verified on CPU.
8. **Example 04 on the GPU:** `AFINNI_DEVICE=gpu uv run python -m
   examples.example04_region_mode --problem all`; the table should match the CPU table in
   `examples/USAGE.md` to rounding (leaf and evaluation counts may differ slightly for
   AFINNI, whose seed depends on batch boundaries set by the VRAM cap).

### 42.6 Deliberately deferred (not unfinished)

Each item below waits for the measurement named with it (Section 39 and Part IV):

| Item | Trigger |
|---|---|
| Migrating partially processed nodes; attempt counters in the payload | profiles showing idle ranks while long nodes run (progress points already serve requests between B4 iterations) |
| Makespan-based grants with calibrated phase costs | measured grant latency or imbalance with the static thresholds |
| Ancestry interning, device-resident descendants, host spilling | measured `h/n` overhead, RSS or migration bytes |
| Distributed assembly, sharded inference | models too large for rank 0 |
| Cached-face Voronoi chords | Voronoi root faces measured as the sampling bottleneck |
| Low-rank quadratic leaves (`d + 1 + r` features along estimated curvature directions) | full quadratic leaves (42.8) need `(d+1)(d+2)/2` coefficients, too many at high dimension |
| Boundary blending, continuity-constrained fits | an application that needs continuity or Jacobians (Q-H) |
| Per-node certificates, sequential tests | proven mixing bounds for the sampler (Q-A) |
| Weights, per-sample ids, global oracle budgets | an experiment that needs them |
| The property system and primitives | a third algorithm (Appendix A) |

### 42.7 Fixes from the second technical review

Found by the technical review (`AFINNI_ART_TECHNICAL_REVIEW.md`) and fixed; none changes an
AFINNI-mode result (`scripts/regression_check.py` bit-identical on CPU) or the region trees
of 42.3:

- **Audit:** a NaN or infinite oracle or model value counted as a pass, because the
  violation test was a `>` comparison. The shared `pointwise_violations` (also used by
  pointwise acceptance) now counts them as violations; `audit` validates its arguments,
  takes default tolerances from the trained acceptance settings and reports `n_invalid`.
- **Voronoi sampling:** a chain in a cell measured the bisector with its own centre
  (`a = 0`), which made a one-centre Voronoi root unsampleable. Own-centre planes are now
  masked; duplicate centres are removed when roots are built.
- **Stuck chains:** a 20-D chain seeded at a corner could stay where it started and pass
  the drift check, which compares endpoints. Nodes where fewer than half of the chains moved
  are rerun on a separate stream with coordinate directions; infeasible steps are counted;
  the node is marked `insufficient_sampling` if either remains.
- **Sparse B1 nodes:** B3 indexed past a node with fewer than `2 * min_side` points. Such
  nodes now have no candidate and end as `split_failed`.
- **Depth and loaded tables:** packed routing keys rejected trees shallower than the
  documented 62 levels when there were many roots; loaded tables were not validated and NaN
  queries routed silently. See the routing row of 42.1.
- **Stealing units:** grants and the load published after them counted inherited points
  while every other load used `work_sizes`.
- **B4 diagnostics:** the margin used the normal-equation residual from the sums (which
  cancels to rounding noise near a good fit), and `soft_converged` counted failed line
  searches. See the split-predicate row of 42.1. The search also stops halving once every
  node has stepped.
- **Memory estimate:** `bytes_per_point` now charges B3/B4's `n_bins`-dependent per-node
  candidate arrays; the executor warns when the estimate is below the batch floor and when
  one cluster exceeds the batch size.
- **Comparison script:** `art_comparison.py` refuses ART leaf degrees other than 1 and
  metrics other than `max_pointwise_relative`, and prints both resolved configurations.

Follow-up review (its Section 32), also fixed; AFINNI-mode results stay bit-identical:

- **Close Voronoi centres:** bisector coefficients and slacks were differences of dot
  products and squared norms of the centres, which cancel when centres are close: with
  centres `1e-12` apart, all 200 samples of the middle cell landed in a neighbour's cell
  and the node passed. They are now formed from the centre difference and normalized, so
  slack is a distance and the feasibility tolerance is scale-invariant; the returned
  endpoint is checked too. Samples stay inside the cell at separations `1e-6` to `1e-12`.
- **Cancelled variances:** B4's temperature and the drift statistic used
  `mean(s²) − mean(s)²`, which cancels for a narrow node far from the origin (`T ≈ 1e-151`
  instead of `5.8e-10`, so every gate saturated; rounding-sized shifts read as drift). Both
  use centred moments with a rounding-resolution floor; the drift radius is taken about
  the node's own mean.
- **Partly stuck nodes:** recovery ran only when most chains were stuck, so 8 of 20 points
  could stay on a 4-D corner unflagged. Every unmoved chain is now rerun, and any chain
  still unmoved flags the node.
- **Region batches:** the AFINNI grid admits the cluster crossing a grid line, so an
  oversized node could share a batch with a smaller one although the warning said it ran
  alone. Region batches now hold whole nodes up to `batch_size`, or one oversized node
  alone; AFINNI's batching is unchanged.
- **Loaded geometry:** the routing table now rejects non-finite or zero-width domains,
  invalid grid bins and non-finite or repeated Voronoi centres, and loading checks that
  the leaf metadata names the same `(root, path_code)` for every leaf row.
- **Comparison script:** relative-error floors other than `1e-12` are refused (ART fits an
  intercept for degree-1 leaves whatever `--leaf-bias` says, as region mode does); the
  `CONFIG` line prints the resolved components instead of the request.

Second follow-up (review Section 33), also fixed; AFINNI-mode results stay bit-identical:

- **Voronoi root assignment:** `root_of` compared full squared distances, whose shared
  tangential part absorbed the normal difference of close centres: in 2-D with centres
  `1e-8` apart, 21 of 200 samples of the middle cell routed to a neighbour (and at `1e-10`
  the start itself was misassigned). The nearest-distance candidate is now re-decided with
  the sampler's centre-difference test; all samples agree with exact rational distances
  at separations `1e-6` to `1e-12`, for the sampler, `root_of` and routing.
- **Points left on a face:** a recovered chain could move one coordinate and keep the
  others at the corner (7 of 20 points on faces in 4-D, unflagged). A chain now counts as
  recovered only when every coordinate left its start value, and recovery uses
  systematic sweeps instead of random axes.
- **Recovery drift on the wrong rows:** the drift statistic used all rows of a node while
  recovery walked a few, so a fixed majority hid the repaired chains' movement. It now
  uses the diagnosed chains' own count, means and variances.
- **Grid root counts:** a loaded grid whose bin product overflows int64 (e.g.
  `[2^62 + 1, 4]`) now fails validation instead of routing to negative indices.
- **Comparison provenance:** `CONFIG` starts with an `effective` block (ART's forced
  intercept, the region sampling actually used) before the requested options.
- **MPI:** the multi-rank region test also trains Voronoi roots, with and without forced
  steals, and checks identical trees.

Not changed, by decision: the load a victim publishes after a grant still includes its
processed prefix and in-flight batch (a scheduling estimate, not a correctness issue);
routing stays a level-wise reference implementation until GPU timings exist.

### 42.8 Algorithm changes after the reviews

Three changes from the architectural review (its Section 34), each measured before it was
kept; AFINNI-mode results stay bit-identical:

- **Bin-sum B3 (its E3).** B3 formed every cut's Gram from all the node's rows: 15 data
  passes per direction. Cut positions depend only on a node's count, so per direction each
  node is now sorted once, each of the 16 bins gets one batched product over its contiguous
  rows, and both sides of every cut are prefix and suffix sums over bins
  (`_SplitRows.cut_sums`). The right side is summed directly, not subtracted from the total.
  Splitting at 20-D fell from 29.6 s to 10.8 s (B3 total 88 s → 66 s); trees were identical
  on spherical piecewise 3-D and Rosenbrock 5-D/20-D and within seed noise elsewhere
  (rounding of the screen can reorder near-tied candidates).
- **B4 iterations 20 → 40.** With 20 iterations nearly every node stopped at the limit
  (`soft_iteration_limit`). A sweep over 0 (B3), 20, 40 and 80 iterations on ten of the
  benchmarks: 40 gave fewer leaves or lower error on most (2-D plane wave 278 → 249 leaves,
  L2 3.3e-3 → 2.5e-3; 3-D plane wave L2 0.20 → 0.049; 3-D Gaussian 128 → 114 leaves), for
  about 35% more build time. 80 helped the plane waves further (3-D L2 0.014) but was mixed
  elsewhere and cost 2–2.5× the time of 20. ART allows 200.
- **Quadratic leaves (its E5, in ART's full form).** `quadratic_lsq` and the
  `LocalModel.n_features` / `features` hooks, so the region plan sizes nodes and fits
  children in the leaf model's basis (42.1). Measured against ART's degree-2 trees in 42.3.
  The low-rank variant for high dimension is deferred (42.6).

---

# Appendix A. Deferred: a PySPH-style property system

*Not part of the current plan.* This appendix records the extensibility idea for later.
Sections 35–36 have already taken from it what the two-algorithm version needs: the
internal column table, inheritance rules, explicit stages, and versioned payloads and
model manifests.

## A.1 Motivation

PySPH (https://pysph.readthedocs.io/en/main/design/overview.html) separates *what is
stored* from *how it is computed*. A `ParticleArray` holds named properties
(`add_property`, with strides for vectors), and `Equation`s read and write those
properties by name (`d_rho`, `s_m`) inside a fixed integrator structure. The same
primitives that define density and velocity let a user add new properties later. The
long-term aim for AFINNI is the same:
- a user declares a per-point, per-cluster or lineage-dependent quantity and its
  computation;
- the engine allocates, carries, migrates, stores and saves it;
- components read and write properties by name.

**Order of work (review §14.1):** generalize storage first, scientific scheduling
second. `sizes` and the offsets stay structural, never user properties.

## A.2 Taxonomy

| Scope | Examples | Lives in / travels with |
|---|---|---|
| per point | `X`, `F`, `G`, sample role, original sample id, chain id | the queue; migrated |
| per cluster, local | centroid, `f(c)`, `∇f(c)`, fit parameters, errors, `Q`, counts, volume, condition number | the device batch; accepted ones go to the leaf table |
| per cluster, inherited | depth, `(root, path_code)`, constraint block, RNG key, budget state, warm-start direction | the queue; migrated |
| per tree node, persistent | predicate, terminal status, compact child indices | node records; saved |
| per run | tolerances, domain, temperature constant, configuration digest | the configuration |

## A.3 Declaration and contracts

*Illustrative.*

```python
@dataclass(frozen=True)
class PropSpec:
    name: str
    scope: Literal["point", "cluster", "node"]
    shape: tuple[int | str, ...]          # symbols "d", "fd", "depth", "d+1", resolved by a validated resolver
    dtype: Any = np.float64
    stage: Stage | None = None            # when it is computed (local props)
    inherit: Callable | None = None       # child value from parent value (inherited props)
    reads: tuple[str, ...] = ()           # checked against the stage
    persist: Literal[None, "leaf", "node"] = None
    migrate: bool = False
    init_new_rows: Callable | None = None # value for rows created by top-up/augmentation
    invalidated_by: tuple[str, ...] = ()  # e.g. ("samples",): recompute after rows change
    placement: Literal["host", "device"] = "host"
```

Contracts a general version must define (review §14.2):
- how new rows are initialized;
- what invalidates a value;
- ownership and mutability;
- the ragged representation;
- required migration (a component must not depend on a field marked `migrate=False`);
- persistence dependencies;
- host/device placement;
- failure semantics.

Executable callbacks are never serialized; model files store registered names, validated
settings and numeric arrays.

Examples:

```python
@CLUSTER_PROPS.register
class TrustLogVolume(ClusterProp):
    """log volume of the trust ellipsoid up to the constant log|B_d| (unit ball); Q must be PD."""
    name, shape, stage, persist = "trust_log_volume_rel", (), Stage.ACCEPTED, "leaf"
    reads = ("Q",)
    def compute(self, ctx):
        sign, logdet = xp.linalg.slogdet(ctx["Q"])   # sign > 0 alone does not prove PD
        return -0.5 * logdet

@CLUSTER_PROPS.register
class Constraints(InheritedProp):
    """Signed ancestor predicates, one (d+1) row per tree level; read by polytope samplers."""
    name, shape, migrate = "constraints", ("depth", "d+1"), True
    def root(self, ctx):
        return xp.empty((ctx.K, 0, ctx.d + 1))
    def inherit(self, parent, split):
        row = xp.concatenate([split.w, split.z[:, None]], 1)   # left: w·x <= z
        return xp.stack([append_row(parent, row), append_row(parent, -row)], 1)
```

## A.4 Primitives

User computations would be vectorized functions over a batch, built from primitives the
engine provides. These are the current `src/numerics.py`, made public and documented:
- segmented sum, max, mean and scatter matrix, and fused statistics;
- per-cluster gather and cluster-to-point broadcast;
- segmented n-smallest, and block-then-value sort;
- batched small linear algebra with an explicit status;
- binned prefix statistics;
- inheritance helpers;
- ancestry materialization.

Code generation (PySPH uses compyle) would come after that.

## A.5 How to get there from the two-algorithm version

1. Turn the internal column table (Section 36) into a public `PropSpec` registry, with
   the contracts in A.3 enforced at configuration time.
2. Expose the stage functions of Section 37 as hook points that take registered
   properties.
3. Validate the design with one user metric (e.g. trust log volume, or fit condition
   number used in an acceptance rule) before generalizing further.
