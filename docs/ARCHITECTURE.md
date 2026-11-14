# Architecture

How a training run and a prediction move through the code, the contracts the parts rely on,
and where to extend it. [README.md](../README.md#code-layout) has the package layers;
[AFINNI_ART_DESIGN.md](AFINNI_ART_DESIGN.md) has the mathematics and the design log.

## Two plans, one engine

AFINNI and ART are **plans** (`src/afinni/`, `src/art/`) of one **refinement engine**
(`src/engine/`). The engine owns everything both algorithms share: the queue of nodes waiting
to be refined, batching by device memory, host-to-device transfers, MPI distribution, work
stealing, level statistics and profiling. A plan owns what differs: its initial queue, its
output columns, how much device memory a queued node takes, where a batch may end, and what
happens to one batch. The boundary is the `Plan` protocol in `src/core/plan.py`;
`tests/unit/test_plans.py` checks both plans against it.

Both plans draw on the same **components** (`src/components/`): local models, split
strategies, acceptance tests, trust regions and samplers, each a registry of named classes
(`src/components/registry.py`) chosen by `TrainingConfig`.

## A training run

```
AFINNISurrogate.train                                    src/surrogate/model.py
└── driver.refine_unguarded         every rank           src/engine/driver.py
    ├── _initial_data               rank 0: f at the training points; AFINNI's initial clustering
    ├── _make_plan                  AFINNIPlan, or ARTPlan with the broadcast domain and roots
    ├── plan.initial_queue          rank 0: one queued node per cluster (AFINNI) or root (ART)
    ├── _scatter                    contiguous runs of nodes to every rank
    ├── _cap_batch_size             batch size from free device memory and plan.bytes_per_point
    ├── refinement.refine_rank      every rank                     src/engine/refinement.py
    │   └── per level, per batch:   plan.batch_end → DeviceBatch → plan.process_batch
    │                               leaves → accepted, split records → splits, children → next level
    │                               steal requests answered between batches, at level ends and
    │                               (ART) inside the soft split search, through batch.progress()
    └── _merge_results              rank 0: concatenate leaves; plan.merge_results (ART: routing table)
```

A plan's `process_batch` runs four stages on its batch:

| | AFINNI (`AFINNIPlan`) | ART (`ARTPlan`) |
|---|---|---|
| 1. points | the batch's own points | each node topped up inside its polytope; validation points drawn if `validation_samples` > 0 |
| 2. fit and accept | local model per cluster, acceptance test | local model per node, acceptance test |
| 3. deploy | accepted clusters with trust regions | accepted nodes, checked on their validation points if drawn |
| 4. refine the rest | augment, then bisect along a curvature-informed direction | cut by a plane from the split search; nodes that cannot split keep a fallback leaf |

## Prediction

- **AFINNI** (`predictor="knn"`): exact kNN over the leaf centroids (`src/neighbors/knn.py`); the
  `k_nn * coarse_factor` nearest are re-ranked by trust-region distance `dx^T Q dx`, and the best
  `k_nn` leaves' models are blended. A query whose best candidate lies at distance > 1 is flagged
  outside.
- **ART** (`predictor="route"`): `RoutingTable.route` (`src/art/routing.py`) walks each query down
  the split planes to its leaf; a query outside the domain is flagged and routed from its
  nearest domain point.

`predict` is collective under MPI (rank 0 scatters the queries and gathers the results);
`predict_local` and `predict_route_local` predict on one process.

## Data on the move

| Container | Where | Holds |
|---|---|---|
| `ClusterBatch` | host, `src/core/clusters.py` | a rank's queue: point columns (`X`, `F`, `G`) and node columns (`sizes`, `depth`; ART adds `root`, `path_code`, `constraints`), in the order of `QUEUE_COLUMNS`, which is also the MPI wire format |
| `DeviceBatch` | device, `src/core/plan.py` | one batch handed to the plan; `take()` passes ownership of the points, so the plan can release them before augmentation and splitting |
| `ColumnBuffer` | host, `src/core/clusters.py` | accepted leaves (and ART's split records), one row per node |
| model file | `.npz`, `src/surrogate/persistence.py` | leaves, local-model parameters, ART's routing table and leaf statuses, settings as JSON; versioned |

## Contracts

- **Determinism.** Training reductions sum in a fixed order per segment (`src/numerics/`,
  `src/kernels/`), so repeated runs give the same bits and an ART tree does not depend on batch
  size, rank count or steal schedule (its random streams are keyed by node). AFINNI's batch
  boundaries and per-batch seed define its trees; `scripts/regression_check.py` detects any
  change.
- **Memory.** The batch size is capped from the free device memory and the plan's
  `bytes_per_point`; a batch's points are released before augmentation and splitting, which
  hold its peak. Host queues check available RAM before they grow.
- **MPI failures.** `driver.run_collective` wraps every collective entry point: in one process an
  exception propagates; under MPI the failing rank logs it and aborts every rank.
- **Work stealing** (`src/engine/work_stealing.py`). Every steal request gets exactly one grant
  or denial; a grant takes nodes from the unprocessed tail of the victim's queue, never from
  its current batch.

## Notation: from the companion to the code

The technical companion ([report/adaptive-surrogates.typ](report/adaptive-surrogates.typ)) writes
the mathematics with the AFINNI manuscript's single-letter symbols; the code uses descriptive
names, and the docstring or comment where a name is defined gives its symbol. Paths are under
`src/`.

| Symbol | Meaning | Name in the code | Where |
|---|---|---|---|
| f, f̂, d | the function (oracle), the surrogate, the input dimension | `Evaluator`; `AFINNISurrogate.predict`; `d` | `core/evaluation.py`, `surrogate/model.py` |
| X₀, N | the initial points and their number | `train(X_train, ...)`; `profiling_stats["N0"]` | `surrogate/model.py`, `engine/driver.py` |
| C, X_C | a cluster (AFINNI) or node (ART), its points | a queued cluster: rows `starts[k] : starts[k] + sizes[k]` of `X`, `F` | `core/clusters.py` (`ClusterBatch`) |
| c, f(c) | a cluster's centre (the mean of its points), f there | `centroids` (`mus` in the plans), `f_c`; for the leaves `X_src`, `f_src` (leaf columns `"centroids"`, `"f"`) | `components/local_models.py` (`ClusterFitData`), `surrogate/model.py` |
| (w, b) | the local affine model f̂(x) = wᵀ(x − c) + b | leaf columns `"A_fit"` (one row wᵀ per output) and `"b_fit"` = b − wᵀc (the intercept at the origin); `AffineLSQModel` | `components/local_models.py` |
| e_a, e_r | a cluster's absolute and relative error | `E_abs`, `E_rel`; leaf columns `"E_cluster_abs"`, `"E_cluster_rel"` | `components/acceptance.py` |
| α_tol, r_tol | the absolute and relative tolerances | `atol`, `rtol` | `config.py` (`TrainingConfig`) |
| N_init, N_min | initial clusters; minimum points per cluster | `num_initial_clusters`, `min_pts_per_cluster` | `config.py` |
| H, abs(H)^½ | the Hessian; the split metric | `M`, the first value of `SplitStrategy.metric` | `components/splitting.py` |
| Σ | the scatter matrix Σ (x − c)(x − c)ᵀ | `Cov`, `C` (called an unnormalized covariance) | `numerics/stats.py` (`run_fused_stats`), `components/trust_regions.py` |
| ĝ | the least-squares gradient at c, from the max(2d, ⌊N_min/4⌋) nearest points | `grad_lsq`, `n_gradient_pts` | `components/splitting.py` (`CovarianceHessianSplit`) |
| N_cov | the points with the smallest Taylor error, which trace the level set | `n_est_cov` | `components/splitting.py` (`CovarianceHessianSplit`) |
| x̃ᵢ, u₁, sᵢ | the stretched points, the split direction, the scores | `M (x − c)`; `v`, the leading eigenvector of `CovZ`; `scores` | `components/splitting.py` (`SplitStrategy.split`) |
| C₋₁, C₁ | the two children of a split | AFINNI: the low and the high half (`child_sizes`); ART: left, aᵀu < z (path code 2π), and right (2π + 1) | `components/splitting.py`, `art/plan.py` (`ARTPlan._split`) |
| T, γ, α | the trust-region matrix, its eigenvalue floor, the farthest point's scaled distance | `Q` (leaf column), `Qs_src`; `q_reg`; `max_dist` | `components/trust_regions.py`, `config.py` |
| κ, dᵢ, ν | Euclidean candidates, trust-region distance, the answering leaf | `k_nn * coarse_factor` (`k_coarse`); `dist_sq`; `nn_idx[:, 0]` | `surrogate/model.py` (`predict`, `predict_local`) |
| 𝒮, N_c | the leaves, their number | the `accepted` buffer during training; `N_src` and the `*_src` arrays | `engine/refinement.py`, `surrogate/model.py` |
| u | coordinates of the domain, in [0, 1]^d | `U`, `Domain.normalize` | `art/geometry.py` |
| Ω_C = {u : A u ≤ z} | an ART node's region | the root cell and `constraints`, rows `[a \| z]` of the ancestors' planes; `NodePolytopes` | `core/clusters.py` (`QUEUE_COLUMNS`), `art/sampling.py` |
| (a, z) | an ART split plane, unit normal and offset; aᵀu ≥ z is the right child | `normal`, `offset` (`RoutingTable`, split records); `w`, `z` in the split search, its kernels and the model file's routing arrays | `art/routing.py`, `art/plan.py`, `art/split_search.py` |
| π | the path code | `path_code` | `core/clusters.py`, `art/routing.py` |
| n = ⌈m p⌉, p, m | points per node, coefficients per output, the multiplier | `ARTPlan.n_target`, `ARTPlan.n_coefficients`, `sample_multiplier` | `art/plan.py`, `config.py` (`ARTSettings`) |
| τ, g | hit-and-run steps, a step's direction | `walk_steps`; `g` | `config.py`, `art/sampling.py` (`HitAndRunSampler`) |
| n_dir, n_bins | search directions; cuts per direction plus one | `n_directions`, `n_bins` | `config.py`, `art/plan.py` (`_direction_bank`, `_bins_search`) |
| yᵢ, φᵢ | a node's values in units of the tolerance, its features | `SplitRows.Y`, `SplitRows.Phi` | `art/split_search.py` |
| SSE, J_h | the hard and soft split objectives | `hard_sse`, `soft_objective` | `art/split_search.py` |
| h, δ, λ | gate width, weight floor, ridge | `T` (`soft_temperature` times the std of the scores), `delta` (`soft_weight_floor`), `SplitRows.lam` | `art/plan.py` (`_soft_refine`), `art/split_search.py`, `config.py` |
| W₋₁, W₁, B₋₁, B₁ | the children's soft weights and coefficients | `WL`, `WR`; `beta` | `art/split_search.py` |
| η | the audit's confidence | `confidence` | `surrogate/model.py` (`audit`) |
| P, wᵢ, L | the ranks, a rank's load, the loads it knows | `comm.Get_size()`, `WorkStealer.local_load`, `WorkStealer.global_loads` | `engine/work_stealing.py` |
| w_*, w_s | share threshold, steal size | `min_load` (`MIN_LOAD_THRESHOLD`, `steal_min_load`), `steal_size` (`STEAL_SIZE`) | `engine/work_stealing.py`, `config.py` |

## Extending

- **A component** (local model, split strategy, sampler, acceptance test, trust region):
  subclass its base class and register it; select it by name in `TrainingConfig`
  ([USAGE.md](USAGE.md)).
- **A plan**: implement every member of `Plan` in `src/core/plan.py`, add it to `_make_plan`
  in `src/engine/driver.py`, and extend `tests/unit/test_plans.py`. The engine needs no other
  change.
- **A CUDA kernel**: one `.cu` file per area in `src/kernels/`, launched through
  `src.kernels.module` / `source`; `tests/unit/test_kernel_sources.py` checks that every kernel
  has a launcher. Conventions are in [AGENTS.md](../AGENTS.md).
