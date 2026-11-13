# Technical review of the proposed AFINNI and ART integration

First review and revision review: November 10, 2026. Response to
[AFINNI_ART_DESIGN.md](AFINNI_ART_DESIGN.md).

**Start with [section 20](#20-review-of-the-revised-design) for the current assessment.**
Sections 20-31 review Claude's revised plan and answer all nine questions, Q-A through
Q-I. Sections 1-19 retain the first review's derivations and source analysis so that
the revised design's existing section references remain valid. The corrections in
section 3 are historical findings, several of which the revised design now resolves.

The revised design reviewed here has SHA-256
`9d8e96eb43cbbf963c6f9d780b424dda51e77e0fc22669ac4c7e97dd99ca64ea`.
Both source revisions below remain unchanged. Changes to the design are not yet
changes to either implementation.

This review uses the actual local implementations, including ART's CUDA workspaces,
builder, sampler, objective and predictor. AFINNI is at commit
`015adfd4a34b1b5ac5f8c67d9fa38a18a5c8bae7`; ART is at
`854924af9ec8555a11db10de6ab75f429a31be2d`. ART's working tree was clean. The supplied
design is an untracked document in the AFINNI working tree. These identifiers pin the
review to the inspected versions, rather than to either repository's future default branch.

Small CPU checks of specific claims were run. There was no GPU performance study,
no training of the proposed integrated algorithm and no change to either implementation. Numerical
results below distinguish those checks from estimates and proposed experiments.

Reading map:

- [Assessment of the revised plan](#20-review-of-the-revised-design) and [question index](#202-answers-to-the-nine-questions).
- [Updated implementation decisions](#30-the-minimal-plan-to-implement-now) and [second-round checks](#31-second-round-verification-and-remaining-uncertainty).
- [Recommendation](#1-recommendation) and [implementation facts](#2-what-the-two-implementations-actually-do).
- [Corrections to the design](#3-corrections-to-the-original-design).
- [Storage designs](#5-evaluation-of-approaches-1-2-and-3), [root choices](#6-roots-and-early-parallelism), and [sampling](#7-sampling-is-a-scientific-change-not-just-another-cuda-kernel).
- [Consistent soft optimization and the descent proof](#8-make-the-soft-optimizer-mathematically-consistent).
- [Alternative splitters](#9-additional-split-strategies-worth-testing), [GPU costs](#10-gpu-execution-and-memory-analysis), and [MPI](#11-mpi-and-work-stealing).
- [Inference](#12-inference-after-integration) and [accuracy and budgets](#13-accuracy-stopping-and-oracle-budgets).
- [Property system](#14-the-property-system-proposal), [analytical conclusions](#15-what-can-be-decided-on-paper-and-what-needs-measurement), and [experiments](#16-a-staged-wslgpu-experiment-plan).
- [Implementation order](#17-recommended-implementation-order), [answers to open questions](#18-direct-answers-to-the-original-designs-open-questions), and [evidence](#19-evidence-and-limits-of-this-review).

## 1. Recommendation

Use AFINNI's execution machinery and introduce explicit region-aware tree training as
an optional algorithm. Keep training storage, migration state and inference storage
separate. A modified version of approach 3 is recommended for the first implementation,
with compact persistent node records and self-contained migratable work items.

Full soft-oblique ART optimization should **not** be the default splitter at the
start. The first serious candidate is:

1. A bounded domain with a cheap hierarchical partition to create initial work.
2. Explicit split planes and complete routing, including unsuccessful terminal nodes.
3. Sampling inside each child's region before deciding whether it is accurate enough.
4. AFINNI's curvature direction, followed by a small search over hard thresholds and
   possibly several directions.
5. Optional, bounded soft-oblique refinement of the best candidate.
6. Independent validation of the model and predictor that will actually be deployed.
7. GPU batches grouped by work shape, with MPI stealing based on estimated work and
   transfer cost rather than only the number of points.

This recommendation has a concrete reason. Explicit regions and sampling address
coverage. Routing addresses the growing inference search. Better split optimization
addresses approximation efficiency. These are separate benefits with separate costs.
Introducing all three at once would make it difficult to identify which change helped.

The original document gets the broad architecture right. Its most useful observation
is that ancestor constraints are needed by sampling, whereas ART's split optimization
needs only the node's samples. That permits dense, reusable optimization workspaces
without ancestor lookup in their inner loops.

The first review found a colliding node-ID encoding, unrepresentable count-based
median splits, unjustified stationary-seed assumptions, and an incomplete argument
for frozen-model line search. The revised design now addresses these in its main
architecture. The remaining implementation gaps and deeper answers are in sections
20-31; they should guide the next revision.

## 2. What the two implementations actually do

### 2.1 AFINNI is a sample-refinement engine with a separate model-selection rule

The principal implementation points are [adaptive_clustering.py](src/engine/adaptive_clustering.py),
[splitting.py](src/components/splitting.py), [sampling.py](src/components/sampling.py),
[local_models.py](src/components/local_models.py), and [surrogate.py](src/surrogate.py).

The engine stores contiguous points grouped into clusters. It fits and tests a batch,
stores accepted leaf models, augments rejected clusters, splits those clusters, and
copies the children into a host queue. Its numerical operations use segmented
reductions and batched linear algebra. Its work-stealing boundary is a set of whole,
unprocessed clusters.

The training partition is discarded. At prediction time, the code searches over leaf
centroids, selects Euclidean candidates, reranks by each candidate's quadratic trust
distance, and optionally blends several leaf predictions. Therefore the deployed
selection rule is not the partition used to fit the leaves.

That distinction matters even before ART is introduced. The current acceptance test
checks a leaf against its own training points. It does not check all points that the
eventual nearest-neighbor predictor will assign to that leaf. A different predictor
can change accuracy without changing any fitted coefficient.

The current out-of-core claim concerns device memory during refinement. The queue
still has to fit in host RAM. Initial clustering and oracle evaluation are centralized
on rank 0, and the final model is gathered and replicated. Those are separate scaling
limits. Retaining the existing loop does not automatically solve them.

### 2.2 ART is region-based adaptive approximation with expensive split selection

ART's [builder](../Adaptive-Regression-Trees-ART-/art/builder.py) constructs a tree of
convex regions. Children inherit the parent samples that fall on their side of a split.
Each child is topped up to its budget before its own fit and stopping decision.
Oversized inherited sets are subsampled. A finite-depth or failed-split node remains a
leaf with a model and a diagnostic status.

Its [soft objective](../Adaptive-Regression-Trees-ART-/art/objectives.py) fits two
weighted local models for a sigmoid gate and adjusts an oblique separating plane.
Its [sampling code](../Adaptive-Regression-Trees-ART-/art/sampling.py) uses exact box
sampling at suitable roots and hit-and-run in child polytopes. Geometry pilots do
not query the oracle.

ART already has more GPU structure than a simple per-node Python implementation.
The [batched optimizer](../Adaptive-Regression-Trees-ART-/art/_cuda_batched_optimizer.py)
and [batched Cholesky workspace](../Adaptive-Regression-Trees-ART-/art/_cuda_batched_cholesky.py)
keep features and numerical buffers in fixed GPU slots, reuse capacity, and compact
active pointer lists. They are worth borrowing from. However, that path assumes
equally sized node datasets; its builder still performs sampling and node fits on
the CPU. It is not a replacement for AFINNI's distributed queue.

### 2.3 ART's inference is already hard routing

In [art/tree.py](../Adaptive-Regression-Trees-ART-/art/tree.py), `SplitNode.route`
goes right when `w @ x - z >= 0`. `RegressionTree.predict` keeps a stack of nodes
and query-index subsets. At each split it partitions the current subset; at a leaf it
evaluates that leaf's model. It does not evaluate a sigmoid at inference, test all
polytope constraints, use kNN, or blend siblings.

It is currently a NumPy/Python traversal, with vectorized work inside each node.
Its asymptotic routing work is proportional to the sum of query path lengths times
dimension. Its Python-node overhead and repeated indexed gathers make it an
inappropriate GPU implementation to port literally.

There is no domain containment check or accuracy-status check in that predictor.
It will return a leaf prediction for an outside-domain point too. This was verified
with a one-leaf model on `[0, 1]`: querying `2` returned `2` for an identity affine
model. A total routing function is not an extrapolation guarantee.

## 3. Corrections to the original design

These findings refer to the original draft. The revised design incorporates most of
them. Source-code observations remain applicable at the pinned commits. Section 20
distinguishes resolved design issues from the issues that remain.

### 3.1 A median-by-count cut is not always a hyperplane cut

The algebra converting a projection direction into a plane is correct in exact
arithmetic. If `a = Mv`, the score is `a^T(x-c)`. For nonzero `a`, a score threshold
`tau` corresponds to

\[
w=a/\|a\|_2,\qquad z=w^Tc+\tau/\|a\|_2.
\]

The missing condition is that the count-based partition can actually be separated
by a threshold. If the two scores adjacent to the median are equal, their midpoint
does not separate them. Four identical points provide a decisive counterexample.
The existing `CovarianceSplit` returns child sizes `[2, 2]`, as a CPU check confirms.
No deterministic function of the query coordinates can route those identical points
to two different children.

Even without exact ties, recomputing the normalized plane can change a near-zero
comparison through rounding, cancellation, or different dot-product reductions.
Large coordinate offsets make `w^T x - z` especially vulnerable.

For the new geometric mode:

- Store the predicate that will actually be used, preferably in centered form
  `a^T(x-c_ref) < tau`, or enforce one common normalized evaluation convention.
- Select a threshold at a distinct-score gap subject to minimum-side requirements.
  With a strict-left predicate, using the upper adjacent score as the threshold can
  avoid a midpoint rounding onto the lower value.
- Repartition the training points with the stored predicate and verify membership.
- If ties make the requested balance impossible, permit an unbalanced cut, try a
  different direction, or retain an unresolved leaf. Do not split identical points
  by their row index and call the result a geometric partition.
- Handle `Mv = 0`, a zero covariance, and nonfinite planes explicitly.

This may deliberately change child counts from current median splitting. Keep it
behind the new mode. It cannot always be implemented as bit-identical bookkeeping.

### 3.2 The proposed packed identity has a collision

Section 31 proposes

```text
mid = (root << L_max) | (1 << depth) | path
```

The root field overlaps the sentinel at `depth = L_max`. With `L_max = 3`, root 0,
depth 3, path 1 gives `9`. Root 1, depth 0, path 0 also gives `9`. This is an actual
integer collision, not a floating-point issue.

Use two fields initially:

```text
node_key = (root_id, path_code)
path_code(root) = 1
path_code(child) = (path_code(parent) << 1) | side
path_code(parent) = path_code(child) >> 1
```

If a packed key is required, reserve `D_max + 1` bits for the entire path code:

```text
key = (root_id << (D_max + 1)) | path_code
```

Parent and child operations must mask and update the path field, preserving the root
field. Shifting the whole packed key is wrong. Check overflow before creating a child.
A fixed 128-bit representation still has a finite depth limit; arbitrary depth needs
a variable-length path or another identity scheme.

These are binary path codes, not spatial Morton codes obtained by interleaving
coordinate bits. They provide ancestry and an ordering, but no general Euclidean
locality guarantee for arbitrary oblique partitions.

### 3.3 Sorted lookup is useful, but it is not automatically coalesced

`searchsorted` over sorted per-level keys avoids a hash table. It still performs
dependent binary-search probes. Different lanes can access distant addresses.
Sorted queries may improve reuse; they do not prove coalesced access. Conversely,
an explicit-parent design does not require a hash table when parent indices are
already known.

Coalescing is a property of the addresses accessed by threads in a memory
instruction, not of the fact that a high-level operation is vectorized. NVIDIA's
[CUDA memory-access guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)
defines it in terms of memory transactions serving warp accesses.

Also distinguish breadth-first ordering, left-to-right ordering within a depth, and
depth-first subtree ordering. A sentinel path integer sorts by depth before path.
Sorting adaptive leaves from different depths by that integer does not produce
left-to-right subtree order. A normalized path-prefix order or a DFS traversal does.

The current queue's order can help local construction. MPI migration does not make
all ranks' appended node records globally ordered. Final assembly must explicitly
establish whichever layout the predictor needs.

### 3.4 Closed polytopes overlap on split boundaries

ART appends `w^T x <= z` to the left region and `w^T x >= z` to the right region.
Their boundaries overlap. The interiors are disjoint, and the overlap has zero
Lebesgue measure for a genuine split of a full-dimensional region. The routing rule
assigns equality to the right, making query ownership unique.

Say this explicitly. A consistent tie convention matters for boundary tests,
degenerate data, feasibility tolerances and reproducibility. Geometry may use closed
sets for numerical sampling, while routing uses a half-open ownership convention.

### 3.5 The temperature shortcut needs a normalized covariance

Section 33 suggests `T = c sqrt(tr(C)/d) n^(-1/d)`. Earlier, `C` denotes the
unnormalized scatter `sum_i (x_i-c)(x_i-c)^T`. With that definition the scale includes
an unintended factor `sqrt(n)`. Duplicating the same dataset changes the length
estimate even though the geometry is unchanged.

Use `Sigma = C/n`, or `C/(n-1)` for the usual sample covariance. Then, for example,

\[
T=c\sqrt{\operatorname{tr}(\Sigma)/d}\,n^{-1/d}.
\]

Even this is only a heuristic replacement for nearest-neighbor spacing. For an
anisotropic node, a gate-specific scale `sqrt(w^T Sigma w)` is more directly related
to the width along that gate. Freeze the chosen temperature during one optimization
run. Recomputing it as `w` changes introduces additional derivatives unless treated
as an explicit outer update.

### 3.6 The proposed stage order cannot express ART's stopping policy

The property-system table places augmentation after acceptance. ART needs a child
sample-budget decision and possible top-up before fitting and accepting that child.
A property attached to the existing post-acceptance stage cannot implement that
behavior correctly.

The engine needs an explicit pre-fit sampling stage. It also needs final-model
validation after any operation that changes the model, including second-order
finalization or a full-data refit. Names and lifecycle contracts should reflect
these operations rather than treating them as hidden property callbacks.

### 3.7 ART's current validation option is not a clean held-out acceptance test

In the inspected `builder.py`, `_fit_convergence_model` excludes a fraction of the
samples from fitting but predicts on the entire sample array. `_prepare_node` then
computes its stopping metric over all those predictions, including training rows.
`_work_leaf` subsequently refits on all data.

This is partial holdout training with a mixed evaluation set. It is not an estimate
computed solely on withheld samples, and the final refitted model has changed.
The integration should implement and test its intended validation contract directly.

### 3.8 AFINNI's relative Linf test is not a pointwise relative-error guarantee

The implemented relative metric is `max(abs(error)) / max(abs(F))` per component.
Consider `F = [1, 1000]` and predictions `[11, 1000]`. The implemented relative error
is `0.01`, so `rtol = 0.02` accepts. The first point has relative error `10`.
This was verified through the current `LinfAcceptance` implementation.

Therefore the design's introductory pointwise goal must not be confused with the
implemented stopping rule. Before comparing ART and AFINNI, specify whether the
target is integrated error, cluster-normalized error, or pointwise tolerance
violation. The same numerical tolerance value is not a fair comparison across
different definitions.

## 4. Separating the integration

The design is easier to reason about as four independently selectable decisions.

| Decision | Existing AFINNI | New alternatives | What it changes |
|---|---|---|---|
| Region and root construction | Data clusters, no retained domain partition | Box tree, hierarchical cells, exact Voronoi roots | Ownership and feasible sampling |
| Samples and budget | Convex combinations after rejection | Child top-up, validation streams, region sampling | Coverage and oracle cost |
| Split selection | Curvature/PCA direction and count/sign cut | Hard candidate search, soft refinement, hinge proposals | Approximation efficiency and training compute |
| Prediction | Euclidean candidates, Q rerank, optional blend | Hard routing, routed candidate search, controlled blending | Query cost, continuity and deployment error |

A `Predictor` registry is appropriate. So are explicit domain and sample-budget
contracts. However, component interchangeability needs compatibility checks.
Tree routing requires routable splits and complete terminal records. Polytope
sampling requires a region and a feasible seed. Soft splitting requires a model that
supports a mathematically defined weighted objective. A Taylor model based on oracle
derivatives is not automatically such a weighted regression model.

The recommendation is to extend the existing registries and storage specifications, and not first
to build a generic user-programmable dependency scheduler. The scientific behavior
changes are large enough without simultaneously replacing the whole execution model.

## 5. Evaluation of approaches 1, 2 and 3

### 5.1 The permanent tree and the live queue have different needs

The permanent tree should store each internal split once. A live work item should
contain enough state to continue after migration without a synchronous request to a
previous owner. These goals do not conflict.

| Approach | What to keep | What to change |
|---|---|---|
| 1: parent indices and persistent planes | Compact authoritative node records; direct parent links when locally available | Avoid globally coordinated node allocation in the training loop; gather ancestry once per sampler batch |
| 2: per-depth arrays and path lookup | Simple immutable arrays; useful local ordering; batched lookup | Remove the claim of guaranteed coalescing; distinguish logical keys from storage positions |
| 3: carried constraints plus node records | Self-contained migration; contiguous sampler inputs | Bound duplication; represent true depth; include RNG and sampler state; do not require the inference layout to match the queue |

For a first implementation, approach 3 has the simplest distributed ownership story. That is
more valuable than saving a few megabytes of path duplication before profiling.
At very large frontiers, use the hybrid below.

### 5.2 A practical hybrid of approaches 3 and 1

Maintain a rank-local immutable node/plane store. A ready job references its ancestry
through local handles. Before GPU sampling, materialize the relevant constraints
into a contiguous temporary buffer. When migrating a job, send the required ancestry
with it, deduplicated across all jobs in the same transfer. The receiver interns that
small ancestry store and creates its own local handles.

This saves repeated host copies when many siblings share long paths, while keeping
migration self-contained. It does not require remote GPU pointer chasing or a global
MPI window. The tradeoff is more packing and reference-management code.

Start with full carried paths and measure the fraction of RSS and MPI bytes they
consume. Add ancestry interning if that fraction is material. Do not choose a switch
threshold based only on `dimension * depth`; sample count, gradient columns, shared
prefixes, migration frequency and sampling frequency all affect the tradeoff.

### 5.3 The carried-path overhead is understated for live children

Let `q` be output dimension, `n` the live node's sample count, `h` depth, and `g` be 1
when every sample carries gradients. In float64, approximately

\[
B_{points}=8n(d+q+gdq),\qquad B_{path}=8h(d+1).
\]

For scalar values without gradients, the ratio is exactly `h/n`. At `d=10`,
`n=100`, `h=30`, the live child has 8,800 bytes of points/values and 2,640 bytes of
path data, a 30% addition. The design's 15% example compares against an augmented
parent with 200 points. After splitting, each child often holds approximately 100.

At `h=50`, `n=100`, the addition is 50%, independent of dimension for scalar output.
For ART's affine budget `n=50(d+1)`, it becomes much smaller as `d` grows. Thus the
sampling policy changes the best storage policy.

Full path inheritance also copies roughly `2 h(d+1)` numbers for each split, plus
the new rows. It is not only a peak-memory cost. Host copying and migration traffic
can become significant when the oracle and split are cheap.

### 5.4 Rank-local processing rounds are not global tree depths

AFINNI explicitly tracks a local round counter separately from `queue.depth`.
A thief may start processing a deeper subtree during an early local round.
The current homogeneous-depth queue discipline can support dense constraint tensors,
but their depth dimension must come from the jobs, not from the local round number.

If the scheduler later mixes ready jobs from different depths, use depth buckets or
ragged offsets with padding inside individual GPU buckets. A global level barrier is
unnecessary for scientific correctness and would reduce the benefit of work stealing.

### 5.5 Ownership and final assembly

The rank that processes a node writes that node's final split or terminal record.
Migrating an unprocessed node transfers the right to process it. Already committed
ancestor records remain immutable. Do not send completed node records back to their
original rank just to preserve an old ownership convention.

At assembly, validate unique keys, existing parents, exactly two children for every
committed split, and exactly one terminal model/status for every reachable terminal.
Then assign compact inference indices. A sibling-adjacent layout permits
`first_child + side`; a DFS or blocked layout may need two child indices or a right
offset. Choose the representation explicitly rather than assuming every final sort
preserves sibling adjacency.

The single-rank final gather is acceptable initially. For large models, retain
rank-local output shards and build a distributed or external-memory assembly path.
Tree identity alone does not make a centralized concatenation and sort scalable.

## 6. Roots and early parallelism

### 6.1 A single box is mathematically simple but exposes little node parallelism

One root gives one fit, one gate and one region until it splits. It does not fill a
GPU or many MPI ranks merely because the rest of the engine is batched. Under binary
growth, exposing `W` independent nodes requires at least `ceil(log2(W))` split levels
in the fully expanding case. Accepted or highly unbalanced branches can expose less.

Parallelize samples, validation and candidate directions at the root, but recognize
the dependency chain. Work stealing cannot steal uncreated children.

### 6.2 Exact Voronoi roots preserve the closest connection to current AFINNI

The correct root cell uses the centers that defined the final assignments, not the
mean of each final assigned cluster recomputed afterward. Those can differ after a
finite number of Lloyd iterations. Retain the actual centers and their coordinate
convention. The inspected GPU k-means path explicitly uses unverified float32
assignments; geometric mode needs a consistent root predicate and verified membership.

With `R` roots, storing all bisectors explicitly costs approximately
`8 R(R-1)(d+1)` bytes. Sharing them among descendants eliminates a frontier multiplier
but not this quadratic root cost. One can retain centers and generate bisectors per
root batch to reduce persistent storage, at the cost of repeated preparation.

Each hit-and-run step still needs all relevant root inequalities. Simply retaining
the nearest few centers is not exact. An omitted center may define an essential face,
and a sample can leave its true root cell. Neighbor pruning needs a geometric
redundancy argument or full-constraint validation with a correct fallback. Rejecting
invalid approximate chords afterward also needs a sampling-distribution argument.

### 6.3 The preferred additional option is a shallow partition tree

Build a shallow, cheap hierarchy over the domain to expose enough independent cells,
then adapt each cell with the selected scientific splitter. Choices include
axis-aligned spatial cuts, PCA cuts, or metric cuts. Persist this hierarchy as the
upper part of the tree.

For axis-aligned upper cuts, descendants can carry updated coordinate bounds. Root
selection is a short traversal, and the sampler avoids `R-1` root bisectors. Oblique
upper cuts add only the path's planes, approximately `log2(R)` for a balanced hierarchy.

The cost is an imposed coarse partition that may create more leaves or unnecessary
boundaries. It is an algorithmic change and needs comparison with Voronoi roots and
a single adaptive box. For GPU/MPI engineering, however, it offers a useful middle
ground: multiple initial tasks without high-dimensional Voronoi bookkeeping.

Size the initial frontier from measured GPU cohort capacity and MPI worker
count, with enough excess tasks to tolerate imbalance. Do not hard-code the old
100 roots or ART's 20-node cohort as a universal optimum.

### 6.4 Reuse the existing grid option, but retain its geometry

AFINNI already has `uniform_grid_cluster_labels` in
[adaptive_clustering_utils.py](src/engine/). It factors the
requested cluster count across dimensions and assigns points to bins. It can supply
a useful additional root baseline before implementing a new partitioner.

The current helper derives bounds from the observed data and returns labels and bin
counts; `initial_clustering` discards the bin counts. Region mode should instead use
the declared domain and retain the bounds, bin structure and routing convention.
Extend the existing implementation where appropriate rather than adding a second
grid-labeling helper. A hierarchical partition remains useful when its adaptive
shape or routing layout is preferable to that grid.

Also handle geometrically nonempty roots with no initial samples. The current
initialization keeps only labels observed in `X0`. That is appropriate for sample
clustering, but it cannot establish coverage of a declared domain partition.
Initialize such cells by sampling them, use a valid explicit fallback, or report
them as unresolved regions. Empty sample sets and empty geometric regions are
different conditions. A box cell can be sampled directly; an oblique cell without
an inherited seed may need a feasibility/interior-point procedure.

## 7. Sampling is a scientific change, not just another CUDA kernel

### 7.1 Feasible inherited samples are not automatically stationary seeds

For a fixed, independently specified child region, conditioning an ideal uniform
parent point on membership gives a uniform child point. ART chooses the child plane
using those same samples and their function values. Conditional on that selected
plane, the inherited data are not generally an independent uniform sample from the
child. MCMC dependence and AFINNI's original hull-biased samples add further problems.

Consequently, inherited points are useful feasible seeds and reusable oracle data.
They do not justify zero burn-in or independence. Launching many chains from the same
small pool removes some serial dependence but does not erase the starting bias.

The classical [hit-and-run analysis by Lovasz and Vempala](https://faculty.cc.gatech.edu/~vempala/papers/start.pdf)
establishes mixing under specified geometric and starting conditions. It does not
justify a universal fixed number of 20 steps in every learned high-dimensional cell.

### 7.2 Freeze the sampling metric during a chain segment

Ordinary hit-and-run selects a direction, computes the feasible chord, and samples
uniformly along it. A fixed positive-definite direction transform corresponds to
hit-and-run in linearly transformed coordinates. Its stationary measure is still
uniform in the original region, because the transformation has a constant Jacobian.

This supports ART's idea of covariance-shaped directions. It does not imply that a
finite pilot produces a reliable covariance, or that an endpoint conditioned on that
estimated covariance is stationary. Freeze the transform, then allow an appropriate
warm-up and diagnose the resulting chains. Adapting the metric to the current point
at every step is a different Markov kernel and needs separate justification.

### 7.3 Use several persistent chains per node as the main candidate

Compare a modest number of persistent chains per node with the proposed
one-chain-per-new-point scheme. Choose chain count to provide sufficient GPU work
across all active nodes. Keep chain state, RNG counters and optional chord slacks.
Generate outputs in chunks rather than allocating all possible chains at once.

One chain per requested sample offers parallelism when few nodes are ready. Its
weakness is that every chain needs warm-up from its seed. With `n_new` samples,
`C` persistent chains and `b` warm-up steps, the warm-up work is `Cb` rather than
`n_new b`. The alternatives can differ greatly in both cost and bias.

For dense path constraints `A`, maintain `s = b-Ax` and compute `v = Au` each step.
The new slack is `s_new = s-t v`. That removes a second dense matrix-vector product,
but adds `O(Cm)` slack storage and accumulation drift. Periodically recompute slacks
from coordinates and check feasibility. ART already has an incremental-slack batched
sampler; reuse that idea rather than recreating all of it from the scalar formula.

### 7.4 Box bounds should cost O(d) per step

The root's `2d` coordinate inequalities need not be stored or multiplied as a dense
matrix. Intersect the line with coordinate intervals directly. With `h` dense ancestor
planes and no Voronoi bisectors, a chord step costs roughly `O(d + hd)`. With all
Voronoi root faces, it becomes `O(d + (h+R-1)d)`.

Process constraints in tiles, reduce the lower/upper chord endpoints, and avoid
materializing a tensor of shape `samples * constraints * dimension`. Depending on
dimension, one warp or several warps can cooperate on a chain's dot products.
Benchmark a node-major layout that reuses plane tiles across that node's chains.

### 7.5 Boundary and feasibility rules need tests

Use normalized constraints or scale-aware tolerances. Handle nearly parallel
directions, negative slack, empty or nonfinite chords, duplicate planes, and thin
regions. A point on a sharp corner may have zero-length chords for many directions;
feasibility alone is insufficient for a good seed. A mean of feasible points remains
feasible in a convex region, but a rank-deficient point set may still have its mean
on the boundary. An interior-point or feasibility recovery path should produce an
explicit failure status if it cannot recover.

Do not silently clamp all constraint violations and then treat the resulting samples
as exact uniform samples. Define numerical tolerances, count repairs, and inspect
their frequency on rotated thin-cell tests.

### 7.6 Useful sampling alternatives

For a child occupying fraction `p` of its parent, rejection sampling from an
independent exact parent sampler needs `1/p` proposals per accepted sample on average.
It is attractive for a shallow, reasonably balanced child. It becomes poor for tiny
fractions or when the parent sampler itself is expensive or approximate.

Keep an independent reserve of geometry samples if testing this idea. Reusing the
same samples that selected the split recreates the adaptive-selection issue.
Exact box rejection is an excellent correctness reference in low dimension when
acceptance is reasonable. It is not a general high-dimensional solution.

Coordinate hit-and-run is another performance candidate: with cached slacks, a
coordinate update accesses a column of `A` rather than a dense direction product.
Its per-step arithmetic is lower, but mixing can deteriorate in rotated thin cells.
Compare effective exploration per second, not raw steps per second.

### 7.7 Match the target measure

Uniform region sampling optimizes uniform-domain error. It may be a poor use of
oracle calls when queries follow a nonuniform physical distribution or a lower-
dimensional manifold. AFINNI's supplied `X0` may encode such a distribution, while
convex combinations impose another one. The integration must name the target measure.

Start with full-dimensional boxes and uniform measure to make the comparison precise.
Later, support weighted samples or a domain-specific sampler. A box inferred from
coordinate minima and maxima can contain physically invalid oracle inputs; requiring
an explicit domain is substantive, not just a configuration convenience.

## 8. Make the soft optimizer mathematically consistent

### 8.1 The current inconsistency is real, but its size depends on configuration

ART's inner fits use trace-scaled ridge and floored weights. Its reported outer loss
contains only weighted residuals, and its gate gradient uses an unregularized envelope
formula. Its sigmoid also clips logits. These are different mathematical operations.
The shared ART benchmark preset currently uses zero ridge, which removes one source
of inconsistency but not all of them. Do not describe every configured ART run as
having the same gradient error.

The CPU objective was checked on 100 fixed random points in three dimensions, with
`y = x0^2 + 0.5*x1*x2 + 0.2*x0`, temperature `0.7`, unit normal `[0.8, 0.6, 0]`,
offset `0.1`, and tangent direction `[-0.6, 0.8, 0, 0]`. A central difference with
step `1e-6` gave:

| Ridge setting | Implemented directional gradient | Finite difference of implemented profiled loss | Absolute discrepancy |
|---|---:|---:|---:|
| `0` | 0.623007756400 | 0.623007756495 | 9.48e-11 |
| `0.1` | 0.601906140542 | 0.617678953430 | 1.577e-2 |

The second ridge value deliberately makes the mismatch visible; it is not a claim
about the magnitude of error at the default `1e-8` setting. The experiment supports
the structural objection, while the derivation explains why it occurs.

### 8.2 A consistent objective

Prepare a fixed node-local feature matrix `Phi`, including an intercept if needed.
Fix input normalization, output scaling and ridge strength for the duration of the
gate optimization. Let `Y` contain the scaled outputs, `B_L` and `B_R` their
coefficient matrices, and `P` be a fixed positive-semidefinite penalty matrix.
Define

\[
F(\theta,B_L,B_R)=\frac1n\left[
\sum_i(1-\pi_i)\|Y_i-\phi_iB_L\|_2^2+
\sum_i\pi_i\|Y_i-\phi_iB_R\|_2^2+
\lambda\operatorname{tr}(B_L^TPB_L)+
\lambda\operatorname{tr}(B_R^TPB_R)\right],
\]

\[
\pi_i=\sigma((w^Tx_i-z)/T),\qquad
J(\theta)=\min_{B_L,B_R}F(\theta,B_L,B_R).
\]

The two inner systems are

\[
(\Phi^TW_s\Phi+\lambda P)B_s=\Phi^TW_sY.
\]

Provided the minimizers are unique and differentiable in the region of interest,
the envelope theorem gives the familiar gate gradient with

\[
a_i=\frac{(\|r_{R,i}\|^2-\|r_{L,i}\|^2)\pi_i(1-\pi_i)}n,
\qquad
\nabla_wJ=X^Ta/T,\quad \partial_zJ=-\sum_i a_i/T.
\]

The ridge term is absent from this derivative because it is independent of `theta`,
not because regularization can be ignored. It is present in the objective used by
the inner solve and the line search.

For example, set `lambda` from the unweighted node Gram once. If instead it depends
on the gate through `tr(Phi^T W Phi)`, its derivative contributes terms proportional
to the coefficient penalty. Those terms must be derived and implemented. Keeping a
node-fixed scale is simpler.

Decide whether the intercept is penalized. If `P` has a zero intercept entry, ensure
the weighted system still has a unique solution. Strictly positive ridge on all
coefficients is simpler mathematically but changes translation/intercept behavior.
That choice belongs in the model specification and tests.

### 8.3 Frozen-model line search can be justified by majorization

This is the most useful mathematical correction to the original discussion.

At iteration `t`, suppose the fitted coefficients `B_t` exactly minimize the same
objective `F(theta_t, B)`. For any candidate `theta`,

\[
J(\theta)=\min_B F(\theta,B)\le F(\theta,B_t),
\qquad J(\theta_t)=F(\theta_t,B_t).
\]

Thus the frozen-coefficient loss is an upper bound on the profiled loss and touches
it at the current iterate. If the line search finds a retracted step satisfying

\[
F(\theta_{t+1},B_t)
\le F(\theta_t,B_t)+c\alpha\langle\operatorname{grad}J,p\rangle,
\]

with a descent direction `p`, then after refitting,

\[
J(\theta_{t+1})
\le F(\theta_{t+1},B_t)
\le J(\theta_t)+c\alpha\langle\operatorname{grad}J,p\rangle.
\]

So a frozen-model Armijo step can ensure decrease of the profiled objective. It
does not inherently lose that property. The original warning applies to the current
mismatched objective and fits, inaccurate solves, or other inconsistencies.

This is useful for GPU batching: line-search trials need gate evaluations and cached
squared residuals, with no new weighted fits. Each accepted iteration needs one pair
of refits. There is no need to pay for a full pair of fits at every rejected trial
just to obtain the descent argument.

A small separate NumPy check using fixed `lambda = 1` in the summed normal equations
gave a tangent-gradient discrepancy of `9.27e-11`. One candidate step produced:

```text
old profiled objective = 1.770259692582
new frozen objective   = 1.733179587232
new refitted objective = 1.731782835099
```

This illustrates the inequality; the proof above is the justification. It does not
prove convergence to the globally best split, nor that the eventual hard partition
improves validation error. Rank deficiency, finite precision, temperature changes
and approximate inner solves require their own checks.

In implementation, still record both frozen and refitted values and reject a
numerically inconsistent step beyond a stated tolerance. This is a cheap check after
the refit that was already required.

### 8.4 Weight floors must agree everywhere

Independently applying `max(pi, delta)` and `max(1-pi, delta)` destroys exact
complementarity near saturation. It also changes the inner objective relative to
the loss and gradient if those use the original gates.

One coherent option is

\[
\widetilde\pi=\delta+(1-2\delta)\sigma(s),
\qquad W_R=\widetilde\pi,\quad W_L=1-\widetilde\pi.
\]

Use these weights in every fit and loss, and multiply the gate derivative by
`1-2*delta`. Another option is no floor with a sufficiently regularized system.
These define different algorithms; either can be coherent.

Use a stable logistic evaluation rather than clipping the mathematical logit and
then differentiating an unclipped function. For extreme positive logits, computing
the left membership using `sigmoid(-s)` can also avoid losing its small value through
`1 - sigmoid(s)` cancellation.

### 8.5 Complementary Grams are an optimization with conditions

When left/right weights sum to one,

\[
G_L=G_{total}-G_R,\qquad b_L=b_{total}-b_R
\]

is algebraically correct. Add each side's ridge after forming the data Grams.
However, subtracting nearly equal matrices loses relative accuracy in a tiny side.
Computing the smaller-mass side directly and obtaining the larger side by subtraction
is one possible safeguard. Computing both directly is the reference implementation.

With independent weight floors, the complement formula is no longer even exact in
real arithmetic. Validate the optimization against the direct two-Gram path on
saturated gates, ill-conditioned designs and tiny child fractions.

### 8.6 Weighted affine fitting needs a real weighted-model contract

The current AFINNI affine model centers using unweighted cluster means. Adding a
`weights` argument without changing the centering, cross-covariance, intercept and
ridge semantics would be wrong.

Use fixed node-local features with a weighted intercept solve, or compute weighted
means and weighted centered statistics consistently. Expose weighted least-squares
capability for compatible models. Do not require every `LocalModel`, including
oracle-based Taylor models, to pretend it solves the same inner problem.

For vector outputs, fixed output scales permit one Gram factorization with multiple
right-hand sides. If the objective instead uses a different per-sample relative-error
weight for each output, each output generally has a different Gram. That is a real
performance consequence of the accuracy definition.

### 8.7 Normalize coordinates and control gate degeneracy

Center and scale inputs before optimization. Otherwise coordinate units determine
the unit-normal constraint, temperature and gradient step. Store the transform and
convert routing planes consistently. A plane in normalized coordinates can be
converted to raw coordinates, but rescaling its normal also requires rescaling its
offset and temperature if the soft gate is to remain the same.

Use a minimum hard-side count and retain the best feasible hard partition during
optimization. A soft effective-mass constraint can help prevent saturated gates, but
does not replace the hard-side check. An offset constrained between projection
quantiles is another candidate; the quantiles depend on direction and introduce
nonsmoothness if treated as part of the differentiable parameterization.

Use a short temperature continuation schedule, a finite iteration budget,
and a reliable hard-split fallback before attempting elaborate global optimization.
Compare candidates after hardening and refitting. A smaller soft loss is not itself
evidence of a better hard tree.

A minimum side fraction is also a modeling restriction. It can prevent a valid cut
around a rare, difficult feature. Distinguish a minimum sample count needed for a
stable fit from a minimum probability/volume fraction imposed for balance. Region
top-up can supply more samples to a small-volume child, whereas a hard fractional
balance rule forbids that child entirely. Track how often the rule rejects the best
candidate, especially in localized-bump and curved-interface experiments.

## 9. Additional split strategies worth testing

### 9.1 Curvature direction plus hard threshold search

AFINNI already pays to find a useful direction. Its median threshold prioritizes
balance, not the two children's fit error. Before optimizing all `d+1` gate variables,
search thresholds along that direction.

For fixed features `phi_i`, outputs `y_i`, and points sorted by projection, the
sufficient statistics on either side are

\[
G_s=\sum_{i\in s}\phi_i\phi_i^T,\quad
B_s=\sum_{i\in s}\phi_i y_i^T,\quad
C_s=\sum_{i\in s}\|y_i\|^2.
\]

For a coefficient matrix `Beta`, residual sum of squares is

\[
\mathrm{SSE}_s=C_s-2\operatorname{tr}(\mathrm{Beta}^TB_s)
+\operatorname{tr}(\mathrm{Beta}^TG_s\mathrm{Beta}).
\]

Solve a small set of candidate child systems and evaluate this expression. With ridge,
do not substitute the unregularized optimum formula blindly. Include the penalty if
that is the selected split objective, and compute residual loss separately when needed.

Testing every position can require too many `p^3` solves. Start with perhaps 8 to 32
quantile or histogram boundaries, excluding tied gaps and undersized sides. Those
numbers are experimental budgets, not recommended universal defaults. Accumulate
statistics per bin and prefix-sum the bins. Avoid storing a per-point prefix tensor
of size `N*p*p`.

This method has no temperature or line search, and fits AFINNI's segmented GPU
execution well. Its limitation is direction choice. It can miss a useful plane even
when threshold selection along the chosen direction is exact.

### 9.2 A small bank of directions, then optional soft polishing

Compare the Hessian-informed direction, the leading covariance direction, selected
additional metric eigenvectors, and a few deterministic random directions. Search
hard thresholds for each. Process direction candidates in bounded tiles.

Then either use the best valid candidate or use it to initialize a short, consistent
soft optimization. This supplies the optimizer with a feasible, scored alternative
and gives it a clear fallback. Candidate directions also provide parallel work near
the root when there are too few nodes.

The hypothesis is that many nodes need a reasonable cut, while only a minority
benefit from a long nonlinear solve. Measure whether the extra optimization reduces
subsequent oracle calls and leaf count enough to pay for itself. Do not select solely
by immediate training SSE; use validation and the final end-to-end tradeoff.

### 9.3 Hinge proposals are useful but do not ensure a continuous final tree

A scalar function `max(a_L^T x+b_L, a_R^T x+b_R)` is continuous across the equality
plane of its two pieces. That makes ART's hinge splitter an interesting proposal
mechanism on locally convex or concave structure.

However, the builder subsequently tops up children, refits models and recursively
subdivides. Those operations need not preserve agreement on the original boundary.
A tree constructed using hinge proposals is not automatically a continuous hinge
representation. Vector outputs also lack one natural shared max/min boundary.

Benchmark it as a bounded-cost scalar splitter, not assume continuity or a
general descent guarantee from the hinge form.

### 9.4 Separate split features from expensive leaf features

At high dimension, consider affine split optimization with richer accepted-leaf
models. A full quadratic leaf may fit a region without needing a quadratic-weighted
inner solve at every gate iteration. Fit and validate that richer model when deciding
whether to stop, then use an affine or reduced-feature split surrogate if splitting
is still necessary.

This is an approximation to ART's ideal objective with the richer model class. It
can choose a different partition, so measure its cost and accuracy. It may still be
far more economical than repeated dense quadratic inner fits.

Reduced-rank quadratics, diagonal quadratics in a learned coordinate frame, and a
small selected polynomial basis are also plausible options. Fix each node's feature
map during gate optimization so the objective and derivatives remain well defined.

### 9.5 Curvature estimates have a limitation that favors testing residual objectives

The low-Taylor-residual covariance heuristic is most intuitive when the residual
level sets resemble ellipsoids. With an indefinite Hessian, signed quadratic terms
can cancel. For `f(x1,x2)=x1*x2` around the origin, small absolute Taylor residuals
occur near both coordinate axes; they do not describe one ellipsoid shaped by
`|H|^-1`.

This does not invalidate every curvature split, but it weakens that heuristic's
geometric interpretation on saddles. Indefinite quadratics and mixed-curvature
functions are essential experiments. ART's residual-driven objectives are credible
alternatives there, even if their compute cost is higher.

## 10. GPU execution and memory analysis

### 10.1 Replace the universal lockstep rule with compatible cohorts

The design says all clusters should proceed in lockstep with masks. Masks are a
reasonable first implementation, but they do not necessarily skip work inside GEMM,
Cholesky or a large segmented kernel. A cohort with one slow node can waste most of
its capacity after other nodes converge.

For iteration counts `I_k`, full-cohort masked execution can cost in proportion to
`K * max(I_k)`, while ideal active execution is closer to `sum(I_k)`. The ratio can
be large when most nodes converge quickly and a few hit the iteration limit.

Use stable slots for `X`, features and outputs, and compact active slot IDs or batched
matrix pointers. ART already does this. Repack large arrays only when saved work
exceeds the copying cost. Refill completed slots at a safe cohort boundary initially;
a persistent continuously refilled scheduler can come later if profiling supports it.

Useful grouping dimensions are feature count, model type, sample-count bucket,
constraint-count bucket, dtype and solver path. True tree depth is relevant to
constraint shape but need not define a global scheduling barrier.

### 10.2 Ragged segmented kernels and dense batched BLAS both have a place

AFINNI's segmented layout is excellent for arbitrary node sizes and reductions.
ART's fixed budget naturally creates equal-size designs suitable for strided batched
matrix operations. There is no reason to force both into one kernel strategy.

For small affine systems, fused segmented statistics may minimize overhead. For
larger feature counts, tiled dense products may obtain much better data reuse.
Bucket similar sample counts and pad a little when the saved launch and indexing
cost exceeds padding work. Zero padding must be excluded consistently from sample
counts, gate losses, statistics and validation.

Never implement `segsum(phi*phi^T)` by materializing all pointwise outer products.
That turns an `O(K p^2)` output into an `O(N p^2)` temporary. Use tiled reductions or
batched products that reuse input tiles. A single global atomic per point and Gram
entry can also become a contention bottleneck; reduce locally before combining.

### 10.3 The cost model changes when ART is introduced

Let `K` be active nodes, `N` their total samples, `p` feature count, `q` output count,
`I` accepted optimizer iterations and `a` average line-search trials per iteration.
A rough operation model is

\[
T_{split}\sim I\left[
c_G Np^2+c_B Npq+c_S Kp^3+c_E Npq+c_A aN(d+q)
\right].
\]

Constants depend strongly on kernels and whether both Grams are formed directly.
With cached squared residual differences, frozen line-search trials can reduce the
`q` dependence further. The point is that `Np^2` and `Kp^3` are qualitatively
different from AFINNI's few-pass splitter.

For sampling, with `C` chains per node, `s` steps per chain, and `m` dense constraints,
the leading work is approximately `O(K C s m d)`, plus coordinate-bound operations.
Preconditioning can add dense direction transforms and a pilot covariance solve.

The old statement that construction is memory-bound need not remain true. Weighted
Gram formation can have high reuse and substantial compute intensity; small solves
can be latency-bound; sampling can be limited by reductions, random generation,
memory traffic or poor occupancy. Profile phases separately.

### 10.4 Dense quadratic models become large very quickly

For total degree two, `p=(d+1)(d+2)/2`. With ART's `n=50p` samples per node, a single
float64 feature matrix needs `8np = 400p^2` bytes. The following are computed sizes,
not measured allocator peaks:

| d | p | n per node | One Phi matrix | Two dense Grams | `n*p^2` multiply-add terms per full Gram |
|---:|---:|---:|---:|---:|---:|
| 10 | 66 | 3,300 | 1.66 MiB | 0.0665 MiB | 1.44e7 |
| 40 | 861 | 43,050 | 282.79 MiB | 11.31 MiB | 3.19e10 |
| 80 | 3,321 | 166,050 | 4,207.25 MiB | 168.29 MiB | 1.83e12 |

Two sides, repeated iterations, factor buffers, weighted features, residuals,
restarts and several simultaneous nodes multiply these costs. At `d=40`, 20 feature
matrices alone occupy about 5.52 GiB. At `d=80`, even one node can be problematic on
a small GPU. This is why general degree-two soft splitting should not be described as
cheap at moderate or high dimension.

Because `n` grows with `p`, per-node Gram work here grows as `p^3`; for quadratics,
that is approximately `d^6`. Feature storage grows approximately as `d^4`. These
growth rates are enough to rule out a blanket dense-quadratic default on paper.

### 10.5 Batch sizing must be based on bytes and actual stage lifetimes

The current hardware cap in `adaptive_clustering.py` estimates the
`AFFINE_COV_HESS` path and assumes a factor-two augmentation footprint. It is not a
memory model for ART features, chain state or optimizer buffers. It also clips the
point cap to a minimum of 50,000, and whole-cluster batching can exceed a nominal
point target when one cluster is oversized.

For the new engine, admit jobs using an estimate such as

\[
B_{peak}\approx B_{resident\ points}+B_{features}+B_{weighted\ buffers}
+B_{Grams/factors}+B_{gate/residuals}+B_{sampling}+B_{new\ points}
+B_{transfer\ staging}+B_{library\ workspace}.
\]

Take the maximum over stages whose buffers overlap, rather than summing unrelated
lifetimes blindly. Include the maximum intended sample budget before allocating.
Allow a batch smaller than any fixed point-count floor. If one node exceeds capacity,
stream its sufficient-statistic and loss passes in tiles, use a CPU fallback, or
stop with a reported resource limit. Silently overflowing a point cap is not an
out-of-core strategy.

Streaming a giant node is mathematically possible for weighted least squares, but
the optimizer may require rereading its features every iteration. This creates a
transfer cost that resident-node benchmarks will not reveal.

### 10.6 Retain selected descendants on the device

The current round trip for every refinement level is simple and bounds VRAM. It can
become wasteful when a node's small children will immediately need several expensive
resident operations.

After the initial implementation works, test a bounded device queue: keep a batch's
children resident for a few refinement steps while capacity permits, and spill older
or excess jobs to the host. Keep a host-visible pool available for stealing. Do not
hold all useful work in one rank's private GPU queue while other ranks idle.

Use reusable workspaces and bounded pinned staging buffers. Overlapping transfers
with compute requires appropriate streams, pinned memory and explicit lifetime
management, as described in NVIDIA's [transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-and-overlapping-transfers-with-computation).
An asynchronous API call alone does not establish useful overlap.

### 10.7 Host out-of-core operation needs a separate design

Track current queues, next queues, accepted records, cached ancestry, temporary
copies, incoming MPI payloads and final-gather memory. A host-memory check is a useful
failure mechanism, but it does not make the run disk-backed.

For workloads exceeding RAM, persist immutable sample/node chunks and spill cold
ready jobs to a local store with a manifest and checksums. Keep a small scheduling
record in memory. Migrate portable numeric payloads, not process pointers or local
file paths that another node cannot access. Measure spill bytes and I/O time.

This should be a later, explicit feature unless the first experiments require it.
First reduce unnecessary frontier growth with sensible sample budgets and bounded
ready scheduling. Those changes may eliminate more memory pressure than a complex
spill subsystem.

### 10.8 The existing small covariance solver cannot be reused unchanged

In [src/numerics.py](src/numerics.py), `solve_regularized_covariance_inplace` adds
trace-dependent ridge internally. Its GPU Cholesky kernel assigns one thread to
each small matrix and clamps a nonpositive pivot to a small positive value. That
choice belongs to the existing AFINNI implementation; it is not a generic solver
contract for the proposed profiled objective.

Passing a gate-weighted Gram through that function would make its ridge depend on
the gate again, defeating the fixed-ridge derivation. Its in-place mutation also
means an unregularized Gram needed for later complement formation must be retained
separately. A one-thread-per-matrix cubic solve should not be assumed efficient for
hundreds or thousands of polynomial features.

Extend the shared numerical layer with an explicit, tested regularization and status
contract, preserving the old call path for legacy regression. Compare small custom
kernels with library batched factorizations at representative feature counts. ART's
GPU Cholesky workspace reports failures but explicitly lacks a condition estimate
or numerical fallback; its CPU auto solver has stronger diagnostics.

Normal equations square the condition number of the weighted design in exact
arithmetic when it has full column rank. Ridge may improve the system, but successful
Cholesky alone does not prove accurate coefficients. Use fixed feature scaling,
residual checks and a selected-node QR/SVD fallback or an explicit terminal failure.
Process exceptional nodes in a separate batch so one difficult node does not force
every node through the slow solver.

Keep float64 as the reference. Mixed-precision Gram formation or gates can be tested
later, with residual refinement and full-precision decision checks where necessary.
A tiny gate error can move a point across a hard boundary or alter whether a node
passes a tolerance. Precision changes are deliberate numerical changes, not invisible
kernel substitutions.

### 10.9 Recording a plane exposes a useful projection optimization

The current GPU `run_fused_split_projection` computes `M @ (x-c)` and then dots with
`v` for each point. That is `O(N d^2)` projection arithmetic. By computing
`a = M^T v` once per cluster, the projection can instead use `a^T(x-c)`, costing
`O(K d^2 + N d)`. The symmetric-metric derivation in the original design already
implies this possibility.

The new geometric mode needs that normal anyway. This is a worthwhile inexpensive
split baseline to measure before attributing all time differences to ART. It does
not remove the covariance/eigenproblem costs, and it changes floating-point
association. Preserve the legacy projection if bit-identical CPU behavior is required;
validate the new mode's stored predicate and membership under its own explicit
numerical convention.

## 11. MPI and work stealing

### 11.1 The existing ownership boundary is a good starting point

Keep stealing whole ready nodes between processing stages. An active optimizer owns
resident arrays, factor buffers and iteration state. Migrating it mid-iteration is
possible, but would greatly complicate the first implementation. Ready-node migration
preserves the useful existing rule that each job has one processing owner.

The existing [work-stealing implementation](src/engine/work_stealing.py) already handles
explicit grants/denials, crossed requests and retirement. Preserve those invariants.
Generic properties change the wire payload and its version, even if the request/grant
state machine remains the same. Calling that change "no protocol change" is too loose.

### 11.2 Point count is no longer an adequate load estimate

The current threshold is 1,000,000 queued points and the minimum grant target is
500,000 points. A rank with hundreds of expensive ART nodes can have less than that
point count and still have substantial work that an idle rank cannot obtain under
the current policy.

Two nodes with the same sample count can differ by polynomial degree, number of
constraints, expected optimizer iterations, retries, oracle cost and solver fallback.
Use a rough calibrated estimate of remaining seconds, such as

\[
\widehat t_k=
c_f n_{new,k}
+c_h C_k s_k m_k d
+c_g I_k n_k p_k^2
+c_s I_k p_k^3
+c_v n_{val,k},
\]

where constants come from recent phase timings and the oracle term is measured at
its actual batch sizes. This need not predict subtree cost perfectly to outperform
raw point count. Add estimated transfer bytes and required receiving memory.

A useful grant should save more waiting time than its packing, transfer and cold-cache
cost. Use conservative thresholds and hysteresis to avoid moving tiny jobs repeatedly.
Tail stealing can remain the first policy, but choose how much to give by estimated
work rather than only by rows.

### 11.3 Long optimizer calls reduce steal responsiveness

AFINNI serves requests between batches. An ART cohort can remain active much longer
than an AFINNI fit-and-split batch. An idle rank's request may wait while the victim
performs many iterations even though it owns ready work.

Add bounded progress points between groups of optimizer iterations or sampling
chunks, serving only host-ready jobs. Measure request-to-grant latency and the tail
of cohort runtimes. A communication thread is another option, but its MPI thread
support and buffer ownership need explicit handling; it should not be the default
assumption.

Do not announce the rank as empty while it has outstanding GPU work, oracle requests,
incoming payloads, pending sends, or children not yet published to the ready queue.
If asynchronous stages are added, expand the termination accounting accordingly.

### 11.4 A migratable job needs more than X, F and planes

The payload should include or reference immutable shared configuration for:

- Node key, true depth, domain/root identity and signed ancestor constraints.
- Samples, outputs, optional gradients, statistical weights and sample-role masks.
- Sample IDs or provenance needed for deterministic selection and reuse.
- RNG stream identity and counters for sampling, candidate directions and validation.
- Any retained chain positions, frozen direction transform and sampler progress.
- Budget state, retry count, stopping status and selected feature/model specification.
- A fallback model or enough local data to construct one if processing terminates.

Slacks can be recomputed from coordinates and constraints after migration rather than
sent, if that is cheaper. Shared root geometry and component settings can be broadcast
once. Verify a schema/configuration digest on every rank and reject inconsistent jobs.
Avoid transmitting Python closures or per-node Python object graphs in the hot path.

### 11.5 Reproducibility needs node-based random streams

The current augmentation seed is `seed + level + batch + rank*1000`. Changing a batch
cap, rank count or steal schedule changes the random points. The new design should
derive streams from a stable node key and purpose, with independent counters for
sampling, validation, candidate directions and retries.

This makes scheduling comparisons interpretable. It does not make floating-point
reductions bit-identical across CPU and GPU, or make a GPU atomic reduction
deterministic. Separate three contracts:

1. The legacy CPU mode remains bit-identical under storage-only refactors.
2. New algorithm modes have reproducible logical streams independent of rank ownership.
3. Backend comparisons use the repository's tight numerical tolerances and exact
   structural decisions where the same algorithm is expected to agree.

A deliberately different sampler or splitter will produce different trees from the
legacy baseline. Such a change needs accuracy/cost validation, not a false claim of
bit-identical compatibility.

### 11.6 Distributed oracle execution must match the scheduler

Collect new sample requests across several ready nodes so an expensive vectorized
oracle receives useful batches. Retain node/sample offsets to place outputs correctly.
Geometry-only pilots should never be counted as oracle evaluations.

An oracle invoked independently by asynchronous ranks must not unexpectedly require
all training ranks to enter a collective in the same order. If the oracle is itself
a distributed solver, give it an explicit worker-group/service design and schedule
calls through that design. AFINNI's independent-node scheduling and an arbitrary
collective oracle are not automatically compatible.

The current code also evaluates `f(c)` for every fitted node. A new affine-only
region mode may not need that value for its fit, predictor or splitter. Make centroid
oracle requirements explicit, just as gradient requirements are explicit, so the
integration does not retain unnecessary expensive calls merely to fill a legacy
shared column. Preserve the old behavior in legacy modes.

### 11.7 One GPU and many MPI ranks is not a multi-GPU scaling experiment

For WSL testing, start with one rank using the GPU. Test multi-rank migration and
termination on CPU. Several ranks sharing one GPU can validate some code paths, but
they create competing contexts, memory pools and kernels; they do not measure scaling
across several GPUs.

For actual multi-GPU tests, explicitly assign ranks to devices and record placement.
The initial default should be one training worker per GPU, with a deliberate design
for CPU oracle workers if needed. Verify device identity in each rank's logs rather
than relying on every process selecting an appropriate default device.

## 12. Inference after integration

### 12.1 Yes, tree routing should become a first-class predictor

The natural ART-style predictor selects one leaf by the stored split predicates,
then evaluates its local model. A root hierarchy adds its own path; Voronoi roots
require a root-center selection step. For an affine leaf,

\[
T_{query}=O(dh+dq),
\]

where `h` is the path length. A quadratic leaf adds `O(pq)` feature/evaluation work.
The `O(h)` shorthand in discussions of trees hides the `d`-component oblique dot
product at each split. It also hides a possible expensive root search.

With a balanced tree, `h` is approximately `log2(L)` for `L` leaves. An unbalanced
tree can have much larger depth, up to a linear path in the unconstrained worst
case. Report the query-weighted mean, p95 and maximum depth, not only leaf count.

### 12.2 The potential memory reduction is substantial

The present GPU Euclidean candidate search scans leaf centers in tiles. Ignoring
verification and selection constants, its work is approximately `O(BLd)` for `B`
queries and `L` leaves. Reranking `c` candidates adds `O(B c d^2)` work and, in the
current implementation, a gathered `Q` tensor with `B*c*d*d` float64 entries.

At `d=80`, one million dense leaf trust matrices require 51.2 GB in decimal units.
For query batch 5,000 and 10 coarse candidates, the gathered trust matrices alone
require 2.56 GB. These are direct size calculations from the stored shapes.

A full binary single-root tree with one million leaves has 999,999 internal planes.
Storing `d+1` float64 numbers per plane needs about 0.648 GB at `d=80`, before indices,
statuses and leaf coefficients. Scalar affine coefficients need another roughly
0.648 GB if stored as a raw-coordinate slope and intercept. These are not complete
model-size estimates, but they show the change from quadratic to linear dependence
on dimension in the selection geometry.

For a routed-only model, make `Q` optional. It can remain useful for diagnostics,
hybrid prediction or compatibility, but carrying it unconditionally forfeits much
of the memory benefit. Similarly, a routed affine predictor need not store a
redundant centroid value when its representation does not use it.

### 12.3 Hard routing does not make the old leaf models valid over their cells

Simply recording old splits and switching prediction can assign a query to a leaf
whose samples cover only a small interior hull. The query has a route, but may have
little supporting data. Thus the first routing experiment is a diagnostic of the
implicit partition, not proof that ART's coverage problem is solved.

Keep comparisons in two stages:

1. On one trained model with recorded valid topology, compare kNN/Q selection and
   routing to isolate predictor effects.
2. Then retrain with child-region sampling and validation to test a complete routed
   approximation algorithm.

The second experiment changes samples and fitted coefficients. Report it separately.

### 12.4 GPU traversal should have at least two implementations to compare

The simple correctness implementation keeps a node ID per query and advances active
queries one level at a time. It is easy to validate but can launch several kernels
per depth and materialize gathered planes repeatedly. Never index internal-plane
arrays using a leaf marker before applying the active mask; a high-level `where`
does not necessarily prevent evaluation of an invalid gather.

For an optimized version, compare:

| Traversal | Likely strength | Likely cost |
|---|---|---|
| Fused query traversal, one thread or cooperative warp per query | Few launches, no global query reshuffle, suitable for smaller batches | Divergent paths, irregular plane access, register pressure at large d |
| Grouped traversal, partition queries by upper subtree or node | Plane reuse, more regular work in large batches | Compaction/partition traffic, more scheduling and temporary storage |
| Shallow upper-level evaluation followed by fused subtrees | Amortizes common upper decisions and improves lower locality | Extra work if too many upper gates are evaluated unnecessarily |

For dense oblique planes, a warp cooperating across coordinates can read a plane
contiguously, whereas one thread per query may be better at small dimension. There
is no universal winner from asymptotics alone. Cache the frequently visited upper
nodes, and test depth-first, breadth-first and blocked layouts with real queries.

NVIDIA's [nvForest API](https://docs.nvidia.com/nvforest/26.10/python_api/)
also exposes multiple layouts and recommends testing realistic batch sizes. That
supports the need to measure layout choices; it does not mean nvForest is a drop-in
implementation for these dense oblique splits and affine leaves.

After routing, grouping queries by leaf can improve coefficient reuse for large
batches. For tiny batches, sorting may cost more than direct gathers. CPU traversal
is an important latency baseline for a single query or a sequential simulation.

### 12.5 Keep geometry, support and accuracy statuses separate

Expose, at least internally:

```text
outside_domain
terminal_reason      # accepted, max_depth, budget_exhausted, split_failed, ...
validation_status    # passed, failed, insufficient, not_performed
outside_sample_support_or_Q   # optional empirical diagnostic
leaf_id
```

An in-domain query routed to a depth-limited leaf is not outside the domain. A query
inside a fitted ellipsoid is not accuracy-certified. A fallback model can provide a
finite value without having met the requested tolerance.

Keep the old `(prediction, outside_mask)` return contract for the legacy predictor.
For the new predictor, define a structured status API or an explicit compatibility
mapping. Do not silently redefine the same boolean and compare its rate as though
it were the old `p_o`.

### 12.6 Hard trees may introduce jumps

Independent affine child models do not generally agree on their shared plane.
Even though the gate was optimized softly, ART's deployed rule is hard. This can
introduce value jumps and derivative jumps that matter in ODE integration,
optimization, sensitivity calculations and repeated surrogate composition.

AFINNI's current selection is not globally guaranteed smooth either. With `k_nn=1`
it selects a single model, and with multiple neighbors, candidate changes and trust
cutoffs can also affect regularity. Compare actual boundary behavior rather than
assuming the current method is continuous.

If continuity matters, test a boundary-aware alternative. Near a split plane,
traverse both children and blend predictions over a specified band. Bound the number
of alternative paths, because unrestricted branching can approach evaluating all
leaves. Train or validate both local predictions over the overlap band. A model
trained only on its own side may extrapolate poorly across the plane.

For nonnegative weights summing to one, if every participating model has error at
most `epsilon` at the query, the blend also has error at most `epsilon` by the
triangle inequality. That is the useful guarantee. Current trust ellipsoids do not
establish its premise.

Continuity-constrained affine fits are another research option. Across one shared
hyperplane, continuity constrains the difference between the two affine models.
Enforcing such constraints across a deep adaptive tree couples otherwise independent
fits and complicates MPI work ownership. It belongs in a separate algorithm.

### 12.7 A routed candidate predictor can preserve some of AFINNI's flexibility

Use the route to locate a small subtree or neighborhood of candidate leaves, then
apply the existing trust-distance reranking or blending locally. This may reduce
global search while retaining a support diagnostic.

However, routing alone cannot guarantee inclusion of the globally best `Q` leaf.
Ellipsoids can extend across several split boundaries, and the nearest sibling is
not necessarily the best candidate. Treat beam width and fallback behavior as
accuracy/performance parameters.

An exact alternative is branch-and-bound over subtrees with valid lower bounds on
trust distance. For positive-definite `Q_l`,

\[
(x-c_l)^TQ_l(x-c_l)\ge\lambda_{min}(Q_l)\|x-c_l\|^2.
\]

For a subtree, a lower bound is its minimum leaf eigenvalue times the squared
distance to a bounding box containing all its leaf centroids. This bound is valid
but can be very weak for elongated ellipsoids. It is a research option if exact
global trust selection matters, not the first implementation to build.

### 12.8 MPI inference can initially keep replication, then shard by subtree

For models that fit on every rank, replicate the compact routing tree and leaves,
scatter queries and predict locally as the current API does. A local prediction
entry point would also let applications with already-distributed queries avoid
gathering them to rank 0 first.

When replication becomes too expensive, replicate only the upper tree and route
queries to owners of lower subtrees. Batch exchanges, preserve original query IDs,
evaluate locally, and return outputs. This naturally matches hierarchical root
cells. Balance inference ownership by observed query frequency and model size,
which may differ from training work.

Requesting a remote node at each traversal step would create depth-dependent
communication latency. Route to a subtree owner in one batched exchange instead.

## 13. Accuracy, stopping and oracle budgets

### 13.1 Specify the scientific objective before choosing a split objective

For integrated squared error under a measure `nu`, the tree's total risk is

\[
\mathcal R=\sum_{\ell\in leaves}\nu(R_\ell)
\,\mathbb E[\|f-\widehat f_\ell\|^2\mid R_\ell].
\]

Sampling the same count in every leaf does not imply equal region probabilities.
An unweighted average over all per-leaf samples overweights small regions. Use
independent global test samples from `nu`, or properly estimated region weights.
Inherited sample fractions can be noisy and adaptively biased volume estimates.

For a pointwise tolerance target, a useful validation residual is

\[
e_i=\max_j\frac{|f_j(x_i)-\widehat f_j(x_i)|}
{\max(atol_j,rtol_j|f_j(x_i)|)}.
\]

Handle disabled tolerances and zero denominators explicitly. Count a violation when
`e_i > 1` using the chosen strict/non-strict convention. This is a proposed metric,
not a description of AFINNI's current cluster-normalized Linf rule.

For multi-output split selection, fixed tolerance-related output scales are a good
first proxy and preserve shared Gram solves. A sum of normalized squared errors can
still hide one difficult component. Keep per-component acceptance and report worst-
component performance. Worst-component split objectives or smooth maxima are later
alternatives when the proxy performs poorly.

### 13.2 Top up before acceptance, but do not equate sample count with information

An affine model in `d` dimensions has `d+1` coefficients per output. A child with
five points in ten dimensions can have a tiny training residual while being
underdetermined. A minimum count is necessary but not sufficient. Nearly collinear
samples, repeated points and correlated chains can still produce a weak design.

Use sample-count, rank/conditioning and validation requirements. Fresh child-region
sampling is particularly important after unbalanced cuts. Put a cap on failed
resampling attempts so a thin or invalid cell becomes a reported terminal failure
rather than an infinite loop.

### 13.3 A fixed node budget gives a useful oracle-cost identity

Suppose every processed node has exactly `n` samples, both children inherit all their
parent's samples without duplication, and each child is topped up to `n`. One split
then requires `2n-n=n` new oracle values in total, regardless of balance. For a full
binary tree with `L` terminal leaves, `I=L-1` splits, and one root,

\[
N_{oracle}=n+nI=nL.
\]

This excludes centroid evaluations, independent validation, failed sampling, changing
budgets, subsampling, and retries that request new values. It also assumes both
children are processed. It is a useful baseline, not an exact description of every
current ART configuration.

The consequence is that a splitter reducing leaf count can save oracle calls. But
an aggressive small-node budget can increase validation failures and deeper
refinement. Compare achieved accuracy, not just a lower reported evaluation count.

### 13.4 Independent validation has a calculable cost

Freeze the candidate predictor, then draw validation samples independently of its
fit and split selection. If `m` iid validation points have zero tolerance violations,
the one-sided upper confidence limit for violation probability at confidence
`1-delta` is

\[
p_{upper}=1-\delta^{1/m}.
\]

This follows from `Pr(no violations | p)=(1-p)^m`. With `m=100` and `delta=0.05`,
the bound is approximately 2.95%. Zero failures in 100 tests is not evidence for a
uniform error guarantee over the cell.

To target `p_upper <= p0`, one needs

\[
m\ge\frac{\log\delta}{\log(1-p_0)}.
\]

For multiple leaves or repeated adaptive tests, allocate the failure probability
across tests or use an appropriate sequential procedure. Do not reuse the same
validation set to select many candidates and then claim the fixed-model bound.
MCMC correlation also invalidates the iid formula as stated; replacing `m` by a
rough effective sample size is a diagnostic approximation, not a general theorem.

Maintain a global untouched test set for reporting the final deployed predictor.
For training decisions, independent chains or fresh geometry samples reduce leakage.
If a model is refit after passing validation, validate the final model again or make
the changed statistical contract explicit.

### 13.5 Partition completeness cannot repair poor validation

Record a model for every terminal region. At a depth or budget limit, keep the
current finite model with its failure status. If a split is invalid, retaining the
parent as a terminal leaf is often cleaner than committing an empty child and then
inventing a model for it.

Report the query-measure fraction routed to unresolved leaves. A total predictor can
have zero unroutable queries and still fail the requested tolerance almost everywhere.
That is why "ART has no outside points" is not an accuracy result.

### 13.6 The approximation-rate argument has assumptions

For sufficiently smooth functions, shape-regular cells of diameter `h` and stable
local affine approximation can give errors of order `h^2`. If `L` comparable cells
cover a `d`-dimensional region, `h` is of order `L^(-1/d)`, giving the familiar
`L^(-2/d)` scaling.

Arbitrary unbalanced oblique splits do not ensure shape regularity, bounded
conditioning, representative samples or stable approximation constants. A thin
cell may have small volume and a large diameter. ART's coverage property alone
does not establish the same rate or improve the curse of dimensionality.

Anisotropic geometry, low intrinsic dimension and reduced-feature models can improve
practical behavior. Treat those as structure to exploit and test, not an automatic
consequence of switching to trees.

## 14. The property-system proposal

The revised design moves the public property system to an appendix and chooses a
fixed internal schema for two supported algorithms. That reduction is sound.
The contracts below still apply internally; they do not require public registries
or a general stage scheduler. See sections 20.3 and 29 for the revised recommendation.

### 14.1 Generalize storage first, scientific scheduling second

The motivation is valid. `ClusterBatch`, queue slicing, selection, growth and MPI
transfer currently repeat the same fixed column list. Reusing `ParamSpec` and
`ColumnBuffer` ideas for point and cluster columns would reduce the chance of losing
a newly added field during migration.

The useful part of the [PySPH comparison](https://pysph.readthedocs.io/en/main/design/overview.html)
is separating stored properties from computations that use them. AFINNI still needs
its own lifecycle because top-up, acceptance, split commitment and terminal-model
validation have specific algorithmic meanings.

A first version should support:

- Typed numeric point columns, cluster columns and persistent node/leaf columns.
- One shared implementation of selection, permutation, slicing, growth and packing.
- Explicit required inputs and produced outputs for registered components.
- Inheritance rules for a small set of cluster metadata.
- Versioned manifests for saved models and migration messages.

Keep `sizes` and offsets structurally distinguished from ordinary columns. They
define the interpretation of every point column. Treating them as freely replaceable
user properties would make validation much harder.

### 14.2 Missing contracts in PropSpec

`scope`, `shape`, `stage` and `migrate` are not enough to define all behavior. Specify:

| Contract | Why it is needed |
|---|---|
| Initialization of new samples | Top-up creates new rows for every required point property, not only X/F/G |
| Invalidation | Adding or removing samples invalidates centroids, fits, statistics and possibly sampling metrics |
| Ownership and mutability | Components must know whether arrays are borrowed workspace views or durable job state |
| Ragged representation | Constraint counts vary by true depth and root geometry |
| Required migration state | A component must not depend on a field silently omitted by `migrate=False` |
| Persistence dependencies | A predictor needs all model transforms and columns required to reproduce evaluation |
| Host/device placement | Not all metadata belongs on the GPU; not all temporary arrays belong in the queue |
| Failure semantics | Invalid outputs must result in a defined fallback/status or an error before committing a split |

Do not serialize executable callbacks in model files. Store registered names,
validated settings and numeric arrays. Validate schemas before a run and before
receiving large payloads. Pack columns using a bounded, versioned message layout,
rather than allocating one new MPI tag per arbitrary property without a limit.

### 14.3 A lifecycle that can express both algorithms

One possible explicit lifecycle is:

```text
load ready job
    -> pre-fit budget and top-up, if this algorithm requires it
    -> compute required statistics / oracle quantities
    -> fit candidate model
    -> finalize the model that would be deployed
    -> validate / decide acceptance
         accepted -> commit terminal record
         rejected -> check depth, resource and oracle budgets
                      -> optional pre-split augmentation
                      -> propose and score split
                      -> validate stored predicate and child membership
                      -> commit split and create inherited child jobs
         cannot continue -> commit terminal model with failure reason
```

The legacy mode can retain its exact numerical order, including its existing
post-rejection augmentation and finalize/retest behavior. The new region mode uses
the pre-fit path. Preserving both deliberately is better than silently moving old
operations and then trying to recover bit-identical behavior afterward.

### 14.4 Small example corrections

The illustrative `LogVolume` computes `-0.5*logdet(Q)`. The actual ellipsoid log volume
in dimension `d` is `log(volume(unit_ball_d)) - 0.5*logdet(Q)`, assuming `Q` is
positive definite. The omitted constant is harmless for fixed-dimension ranking,
but the property should be named or documented accordingly. A positive determinant
alone does not prove positive definiteness.

Similarly, a shape depending on `"level"` needs a definition tied to tree depth,
not a rank-local loop counter. String dimensions such as `"d+1"` should use a small
validated dimension resolver, not unrestricted expression evaluation.

## 15. What can be decided on paper, and what needs measurement

### 15.1 Conclusions that do not require a GPU experiment

The following follow from representation sizes, code behavior or the derivations
above:

1. Hard routing removes the need to search all leaf centroids and can eliminate dense
   `Q` storage when that diagnostic/predictor is disabled.
2. A finite complete tree defines a leaf for each finite in-domain query under a
   consistent predicate, but it does not certify that leaf's accuracy.
3. Median-by-count membership with tied scores cannot always be reproduced by a plane.
4. The proposed packed node encoding collides at its depth boundary.
5. Dense quadratic ART features with a budget proportional to feature count become
   expensive rapidly in both memory and arithmetic.
6. Carrying ancestry avoids remote lookup at the price of path duplication; a compact
   authoritative node table avoids permanent duplication.
7. A point-count scheduler cannot represent the cost differences between different
   feature counts and optimizer iteration budgets.
8. A consistent exact inner minimization makes the frozen-coefficient loss a valid
   majorizer of the profiled loss.
9. A bounded split-search budget and an explicit terminal fallback bound work even
   when local optimization fails.
10. Fixed chain thinning is not a dimension-independent guarantee of representative
    samples or independent validation.

These are enough to improve the architecture before benchmarking.

### 15.2 Questions that need measurements on the target workload

The winning split objective, number of candidate directions, useful soft-iteration
budget, sample multiplier, chain count, warm-up length, cohort size, active-list
compaction frequency and inference layout depend on data and hardware.

In particular, a more expensive splitter can be worthwhile when oracle evaluations
are expensive or when the model will answer many queries. A useful accounting is

\[
T_{lifetime}=T_{training\ overhead}+N_f\,\overline t_f+N_q\,\overline t_q,
\]

with the understanding that oracle and prediction times depend on batching and
parallelism. Relative to a cheaper baseline, extra optimizer time is justified when

\[
\Delta T_{optimizer}
< \Delta N_f\,\overline t_f+N_q\,\Delta\overline t_q
\]

after accounting for any other added costs. This is a break-even model, not a
prediction of a specific GPU runtime. Report the individual terms so a user with a
different oracle or query volume can make their own decision.

For hardware analysis, use a phase-level roofline estimate

\[
T_{phase}\gtrsim\max(F/P_{effective},B/BW_{effective}),
\]

then add measured transfer, launch, synchronization and scheduling costs that do not
overlap. Peak specifications alone are insufficient. Small float64 solves and
irregular sampling rarely achieve the same fractions of peak as a large dense GEMM.

## 16. A staged WSL/GPU experiment plan

### 16.1 Separate four comparison questions

| Comparison | Hold fixed | What it answers |
|---|---|---|
| Predictor comparison | Same trained leaves and topology | Does routing improve cost and what accuracy changes come solely from selection? |
| Oracle efficiency | Same total training oracle budget, with validation calls identified | Which method learns more from expensive evaluations? |
| Time to accuracy | Same independent deployed-error target | Which complete algorithm reaches useful accuracy sooner? |
| Resource-constrained quality | Same time/RAM/VRAM limit | What is the best usable model on the actual machine? |

Use common initial samples and a common untouched global test set wherever the
algorithms permit it. Do not compare an affine AFINNI model with a quadratic ART
model and attribute the entire difference to splitting. Cross the model class and
splitter factors explicitly in selected experiments.

### 16.2 Candidate algorithms

Start with a small identifiable set rather than a full Cartesian product:

| Label | Training change | Predictor | Purpose |
|---|---|---|---|
| B0 | Current AFINNI | Current kNN/Q | Frozen reference |
| B1 | AFINNI with valid stored topology and explicit domain | Both kNN/Q and hard route | Isolate routing and expose coverage gaps |
| B2 | Child-region sampling, pre-fit budget, geometric AFINNI splits | Hard route | Test whether coverage and budgeting deliver most of the gain |
| B3 | B2 plus hard threshold/direction candidate search | Hard route | Test a bounded-compute residual splitter |
| B4 | B3 plus corrected soft refinement with a fixed budget | Hard route | Test whether nonlinear gate optimization pays for itself |
| R | Original ART at the pinned revision and documented settings | Original hard route | Reference behavior and integration comparison |

B1 may deliberately differ from B0 on tied or unrepresentable splits. Log those
events. A recording-only run can detect them without pretending it has produced a
valid routed tree. Keep the strict legacy regression test separate.

After these comparisons, add the hinge splitter and selected polynomial/reduced-
feature variants. Keep original ART and the corrected objective as distinct entries
when comparing optimization behavior.

### 16.3 Problem families and why each is needed

| Problem | Failure or advantage it tests |
|---|---|
| Constant and affine functions | Unnecessary refinement, duplicate data, exact-fit behavior, avoidable centroid calls |
| Rotated anisotropic positive quadratics | Curvature directions and elongated-region sampling |
| Indefinite quadratics and `x1*x2` | Cancellation in the low-residual curvature heuristic |
| Rosenbrock-type curved valleys | Thin cells, hard local geometry, sampler preconditioning |
| Smooth localized bump with low-volume support | Premature acceptance and whether sampling discovers difficult regions |
| A continuous oblique piecewise-affine function | A favorable case for residual/hinge splits and an exact boundary reference |
| Curved piecewise interface | Deliberately outside smooth AFINNI assumptions; gate saturation and tiny-side behavior |
| Vector outputs with different scales and dominant directions | Shared split quality and per-component tolerance handling |
| Real flow-map oracle | End-to-end oracle cost and accumulated prediction/derivative errors |

Begin at dimensions 2 and 10 for geometry inspection and debugging, then 40 and 80
with affine or explicitly bounded feature sets. Use full quadratic soft fits only
where the memory model admits them. Include translated and rescaled versions to
test coordinate conventions, near-tie behavior and conditioning.

For the localized bump, keep both global random tests and a targeted stress set that
actually visits the bump. Report them separately so the stress set does not pretend
to estimate uniform global risk.

### 16.4 Microbenchmarks before long trainings

1. **Weighted fits and gate optimization.** Sweep sample count, feature count,
   number of active nodes, imbalance and conditioning. Compare direct two-Gram and
   complement paths, dense BLAS and segmented kernels, masked and active-pointer
   execution, and fixed versus ragged-size buckets.
2. **Sampling.** Use boxes and clipped rotated thin polytopes with known low-dimensional
   reference moments. Sweep constraint count, chain count, warm-up and thinning.
   Measure feasibility, coordinate/projection moments, boundary exploration,
   autocorrelation and time per useful sample. Include shared-seed short chains.
3. **Lineage handling.** Measure path materialization, path inheritance, peak RSS and
   packed migration bytes at representative depths and frontier sizes. Test sharing
   within a transfer before building a distributed lookup scheme.
4. **Inference.** Keep one tree fixed. Sweep query batch sizes `1, 32, 1024, 65536`
   and larger only if memory permits. Compare CPU traversal, vectorized level routing,
   fused traversal, grouped traversal and the existing predictor.
5. **Scheduling.** Use nodes with deliberately skewed optimizer/oracle costs. Compare
   point-count and time-estimated stealing, and record request latency and rank idle time.

A fast sampler with biased moments is not a successful microbenchmark. A fast fit
that changes hard split decisions near tolerance thresholds needs numerical analysis
before being used as a replacement.

### 16.5 Required correctness tests for the implementation

The following are behavior tests to add when the corresponding features are built:

- Root and child routing reproduce the committed training membership.
- Duplicate points, equal projection scores, zero normals, translated coordinates,
  and near-plane points produce defined behavior on both backends.
- Node keys remain unique across roots, maximum depth, migrations and retries.
- Every committed split has two reachable children; every terminal has a model and
  status; root-only trees and all-failure trees can be saved and loaded.
- Constraint materialization equals explicit ancestor accumulation, including signs.
- Every generated sample is feasible within the documented numerical tolerance.
- Tiny/unbalanced children cannot pass solely because they were fitted on too few
  points; sampler failures and budget limits terminate with visible status.
- Weighted fits match a direct reference, including the intercept, output scaling,
  ridge, gate floors and extreme weights.
- Tangent finite differences agree with the selected soft objective; frozen and
  refitted accepted losses satisfy the expected inequality within solve tolerance.
- Complementary Gram calculations agree with direct calculations where enabled.
- A migrated job retains its constraints, statistical roles, RNG streams, budget and
  fallback state. Force steals below the current million-point threshold in tests.
- MPI request crossings, delayed replies, empty ranks, depth-limited jobs and failures
  leave no lost or duplicated jobs and complete without stranded transfers.
- Save/load preserves predictor selection, domain transforms, topology, terminal
  statuses and model columns. Malformed or incomplete topology is rejected.
- Training error, validation error, true outside-domain rate, unresolved query rate
  and optional trust-support rate remain separate metrics.

Do not require a new soft-split algorithm to produce the old AFINNI leaf counts.
Require CPU/GPU versions of the same intended algorithm to agree under the repository's
numerical contract, and investigate any structural disagreement rather than hiding
it with a loose tolerance.

### 16.6 Measurements to retain for every run

Record the exact revision, settings, seed streams, hardware/device assignment,
Python/CuPy/CUDA/MPI versions and thread counts. Separate first-use JIT/compilation
time from warm execution, while reporting both if cold start matters. Synchronize
the GPU at timing boundaries or use correctly scoped device events.

Retain:

- Total time and times for oracle evaluation, sampling/pilots, feature construction,
  fit, split optimization, validation, transfer, packing, assembly and inference.
- Oracle counts by purpose, generated and reused sample counts, and geometry-only
  chain steps.
- Peak host RSS per rank and aggregate on a host, peak VRAM, host/device bytes,
  MPI bytes, and spill bytes when enabled.
- Active cohort utilization, optimizer iterations, backtracks, retries, solve failures,
  sampler repairs and fallback rates.
- Leaf/internal-node counts, sample counts, depth distribution and child balance.
- Global test relative L2, normalized tolerance-violation rate, per-component error
  quantiles and a sampled maximum clearly labeled as sampled.
- Errors conditional on terminal status, region/support flags and boundary proximity.
- Inference throughput, single/small-batch latency and p95 latency, with model-loading
  and transfer costs included or excluded explicitly.

Use several independent seeds. Five is a reasonable starting allocation for stochastic
quality comparisons, with more runs if variability obscures conclusions. Benchmark
repeat count for timing should be chosen from observed timing noise.

### 16.7 WSL execution sequence

First verify the environment with explicit backends. The following commands are
existing repository checks for implementation work; they were not run as part of
this documentation review:

```bash
AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"
AFINNI_DEVICE=gpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"

AFINNI_DEVICE=cpu AFINNI_MPI=on AFINNI_MPI_NP=4 PATH=$PWD/.venv/bin:$PATH \
  .venv/bin/python -m pytest tests/mpi -o addopts='' -q
```

The MPI tests launch their own `mpiexec`; do not wrap pytest in another `mpiexec`.
For a storage-only refactor, save the differential reference from the current base
checkout, then compare the new checkout against it:

```bash
# In the unchanged base checkout, with its environment and PYTHONPATH:
AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. .venv/bin/python \
  scripts/regression_check.py --save /tmp/afinni_art_base

# In the checkout containing the refactor:
AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. .venv/bin/python \
  scripts/regression_check.py --save /tmp/afinni_art_new --compare /tmp/afinni_art_base
```

CPU refactors must be bit-identical. Use the repository's GPU comparison tolerance
with exact required decisions, and distinguish separate-run training variability
where the existing test helpers already do so.

After those checks, run the single-GPU microbenchmarks, then the small end-to-end
matrix, then resource-limited larger runs. Multi-rank CPU tests establish protocol
behavior; a real multi-GPU machine is still needed to judge multi-GPU speedup and
interconnect behavior. WSL on one GPU can settle many kernel and algorithm questions,
but cannot establish that scaling result.

## 17. Recommended implementation order

### Phase 1: preserve a trustworthy baseline and define contracts

Freeze the base reference. Define domain ownership, split tie rules, terminal statuses,
the accuracy metric and the model/predictor compatibility rules. Add only the storage
generalization needed to carry node identity and metadata. Keep the legacy numerical
path unchanged and run its differential check.

### Phase 2: build a correct explicit tree with inexpensive splits

Implement a single box and a shallow root hierarchy first. Add stored predicates,
unique keys, terminal records, final compact assembly and a CPU reference predictor.
Detect unrepresentable median cuts. Add an initial GPU routing implementation and
compare it with kNN/Q on the same fitted model where the topology is valid.

This phase answers whether the explicit representation and predictor are useful
before introducing iterative ART optimization.

### Phase 3: make the samples support the routed regions

Add pre-fit budgets, child-region top-up, independent validation and a bounded sampler.
Start with full carried constraints for correctness and self-contained migration.
Use low-dimensional reference sampling to validate the GPU sampler. Make `Q` optional
for routed-only models and avoid unnecessary centroid oracle evaluations.

This phase is the first complete region-based approximation algorithm.

### Phase 4: introduce bounded hard split optimization

Add threshold candidates and a small direction bank. Reuse segmented statistics and
batched solve machinery. Keep candidate memory bounded. Compare B2 and B3 at equal
oracle budget and equal deployed error.

### Phase 5: add corrected soft-oblique optimization

Implement the consistent penalized objective and finite-difference tests before
optimizing it. Borrow ART's fixed-slot workspaces and active-pointer execution.
Initialize from the hard candidate and retain it as a fallback. Use the majorization
argument to keep trial steps inexpensive, with refit-value checks for numerical safety.

### Phase 6: improve scheduling and storage where profiles justify it

Calibrate work estimates, add progress points, and adjust stealing granularity.
Then evaluate shared ancestry, bounded device-resident descendants, true host spilling
and distributed model assembly. Add polynomial/reduced-feature leaves only with a
working memory estimator and documented solver policy.

Each phase should produce a usable configuration and an attributable experiment.
Avoid combining a new sampler, objective, predictor, property framework and MPI
scheduler into one change whose accuracy or speed difference cannot be explained.

## 18. Direct answers to the original design's open questions

| Question | Initial choice and reasoning |
|---|---|
| Single box or Voronoi roots? | Add a shallow hierarchy as a third option. Use a box for mathematical reference, hierarchy for initial GPU/MPI concurrency, and Voronoi roots for compatibility experiments. |
| Sign or median cut? | Use a valid geometric predicate with a minimum-side rule. Search a few hard thresholds; exact count balance is subordinate to representable membership. |
| How should vector outputs combine? | Begin with fixed tolerance-related output scaling and summed squared residuals for split proposals, while accepting per component. Test worst-component failures explicitly. |
| Carry or gather constraints? | Carry for the first version. Measure `h/n`, shared-prefix duplication and migration bytes; then gather from a rank-local immutable ancestry store when warranted. |
| Who writes a stolen node? | Its current processing owner. Use stable logical keys and assemble compact indices later. |
| What does outside mean? | Outside the domain only. Report unresolved termination and empirical support separately. Preserve the legacy flag's meaning in legacy prediction. |
| Holdout or confidence-based acceptance? | Fresh validation of the deployed model first. Add a stated probabilistic criterion only when the sampling and adaptive-testing assumptions support it. |
| Is full ART soft optimization best for the GPU? | Not established. It is more expensive per node and may save nodes/oracle calls. Hard candidate search plus selective soft refinement is the preferred first challenger. |
| Must inference change? | It should become selectable. Hard routing is the natural ART predictor and likely the main asymptotic selection-memory improvement; kNN/Q remains a useful baseline and possible hybrid. |
| Can the best design be chosen entirely on paper? | Some designs and claims can be ruled out or corrected now. The best approximation/throughput tradeoff still requires controlled experiments on the intended oracles and hardware. |

## 19. Evidence and limits of this review

### 19.1 Source map

These are the main inspected sources. The relative ART links assume the sibling
checkout location that exists in this workspace. Pinned upstream links preserve
revision context when sharing the review.

| Topic | Source |
|---|---|
| AFINNI refinement, batch cap and RNG | [src/engine/adaptive_clustering.py](src/engine/adaptive_clustering.py), [pinned source](https://github.com/ut-padas/affini/blob/015adfd4a34b1b5ac5f8c67d9fa38a18a5c8bae7/src/engine/adaptive_clustering.py) |
| Queues and output columns | [src/core/clusters.py](src/core/clusters.py) |
| Splits, samples and model contracts | [src/components/splitting.py](src/components/splitting.py), [src/components/sampling.py](src/components/sampling.py), [src/components/local_models.py](src/components/local_models.py) |
| Numerical kernels and existing grid roots | [src/numerics.py](src/numerics.py), [src/engine/](src/engine/) |
| Acceptance and prediction | [src/components/acceptance.py](src/components/acceptance.py), [src/surrogate.py](src/surrogate.py), [src/neighbors/knn.py](src/neighbors/knn.py) |
| Initial cluster geometry | [src/neighbors/kmeans.py](src/neighbors/kmeans.py) |
| Migration and retirement | [src/engine/work_stealing.py](src/engine/work_stealing.py) |
| ART builder and stopping | [art/builder.py](../Adaptive-Regression-Trees-ART-/art/builder.py), [pinned source](https://github.com/PranksterComet/Adaptive-Regression-Trees-ART-/blob/854924af9ec8555a11db10de6ab75f429a31be2d/art/builder.py) |
| ART geometry and prediction | [art/domain.py](../Adaptive-Regression-Trees-ART-/art/domain.py), [art/tree.py](../Adaptive-Regression-Trees-ART-/art/tree.py) |
| ART objective and model solves | [art/objectives.py](../Adaptive-Regression-Trees-ART-/art/objectives.py), [art/models.py](../Adaptive-Regression-Trees-ART-/art/models.py) |
| ART sampling and pilots | [art/sampling.py](../Adaptive-Regression-Trees-ART-/art/sampling.py), the builder's `_sample_region` |
| ART GPU cohorts | [art/_cuda_batched_builder.py](../Adaptive-Regression-Trees-ART-/art/_cuda_batched_builder.py), [art/_cuda_batched_optimizer.py](../Adaptive-Regression-Trees-ART-/art/_cuda_batched_optimizer.py), [art/_cuda_batched_cholesky.py](../Adaptive-Regression-Trees-ART-/art/_cuda_batched_cholesky.py) |
| ART's own research caveats | [art/agents.md](../Adaptive-Regression-Trees-ART-/art/agents.md) |

The external primary references used for the specific systems/sampling points are
linked next to those claims. Their role is limited. The proposed scheduler, hybrid
splitters and implementation sequence are engineering recommendations, not
performance claims established by those references.

### 19.2 Checks actually performed

The small checks were run in the existing AFINNI virtual environment with
`AFINNI_DEVICE=cpu` and `AFINNI_MPI=off`, with the sibling ART checkout added to
`PYTHONPATH` for its imports. They checked:

1. Existing covariance median splitting returns two nonempty count-based children
   for four identical points.
2. The proposed packed ID expression produces the explicit collision described above.
3. The implemented Linf acceptance permits the cluster-normalized/pointwise discrepancy.
4. ART's actual CPU objective has the reported tangent finite-difference behavior.
5. ART's actual predictor returns a value outside its stored domain.
6. A separate NumPy implementation of a fixed-ridge objective agrees with its envelope
   gradient and illustrates the majorization inequality.
7. Feature, Gram, trust-matrix and plane sizes match the arithmetic in this review.

No full unit suite, multi-rank training or GPU benchmark was run for this
documentation-only change. There is no claim here that any proposed integrated
implementation has passed those tests.

### 19.3 Reproducing the most consequential objective check

From this repository, the following standalone command reproduces the actual ART
gradient comparison. It writes no files and requires the sibling checkout path used
in this review.

```bash
AFINNI_DEVICE=cpu AFINNI_MPI=off \
PYTHONPATH=.:../Adaptive-Regression-Trees-ART- .venv/bin/python - <<'PY'
import numpy as np
from art.models import AffineRidgeModel
from art.objectives import SoftObliqueRidgeObjective

rng = np.random.default_rng(13)
X = rng.normal(size=(100, 3))
y = X[:, 0]**2 + 0.5 * X[:, 1] * X[:, 2] + 0.2 * X[:, 0]
theta = np.array([0.8, 0.6, 0.0, 0.1])
tangent = np.array([-0.6, 0.8, 0.0, 0.0])
epsilon = 1e-6

for ridge in (0.0, 0.1):
    objective = SoftObliqueRidgeObjective(
        0.7, AffineRidgeModel(ridge=ridge), X, y
    )
    analytic = objective.grad(objective.evaluate(theta)) @ tangent
    finite_difference = (
        objective.value(theta + epsilon * tangent)
        - objective.value(theta - epsilon * tangent)
    ) / (2 * epsilon)
    print(ridge, analytic, finite_difference, abs(analytic - finite_difference))
PY
```

The perturbation is tangent to the unit sphere at the chosen normal. Its symmetric
off-sphere evaluations have the same first derivative as a retracted tangent curve
at zero. Use explicit retractions as well when adding the permanent optimizer tests.

The first implementation target should be B2 followed by B3: a complete, validated
region tree using AFINNI's execution engine, then a bounded hard split search. That
provides a meaningful baseline against which the corrected ART soft optimizer can
earn its additional compute cost.

## 20. Review of the revised design

### 20.1 Overall assessment

The revision is substantially more implementable. Approach 3 is sound, as are two named
algorithm presets, a fixed internal column schema, separate predictor semantics,
pre-fit region sampling, and bounded hard split search before soft refinement.
Deferring a public property framework is the right decision for this stage.

The revision also correctly incorporates the path-code fix, geometric split
validation, the corrected path-memory ratio, active-slot GPU execution, and the
majorization proof. Those are resolved design issues. They need tests when
implemented, but there is no reason to reopen their basic architectural choice.

The remaining problem is that Part V simplifies away some state and lifecycle
requirements that Part IV still promises. The first implementation can remain
small, but its schema and stage boundaries must describe the algorithm it actually
runs. In particular, do not implement the claim that this needs only two branch
points in the existing loop, or that the node key alone removes the need to describe
random-stream progress.

The revised position on the open questions is more specific than the first review:
use an independent global audit for a defensible first statistical claim; treat
local MCMC validation as diagnostic unless its mixing error is bounded; keep exact
Voronoi geometry through an optional certified chord calculation rather than insisting
on a globally minimal face list; and use residual-based inner-solve error budgets
for soft-optimizer descent checks.

### 20.2 Answers to the nine questions

| Question | Answer and detailed discussion |
|---|---|
| Q-A: dependent/adaptive validation | Use fresh fixed-predictor tests, explicit error-probability allocation, and an independent final audit. ESS alone does not turn MCMC samples into a confidence certificate. [Section 21](#21-q-a-validation-under-dependence-and-adaptive-testing). |
| Q-B: child-chain warm-up | No universal step count or finite diagnostic certifies mixing in an arbitrary learned cell. Use dispersed starts, geometry-aware pilots and bounded diagnostic escalation; keep certification claims separate. [Section 22](#22-q-b-warm-up-from-inherited-seeds). |
| Q-C: exact Voronoi faces | Low-dimensional topology construction is useful, but globally minimal face lists are unnecessary for exact chord sampling. A lazy endpoint-certified chord is another option. [Section 23](#23-q-c-exact-voronoi-geometry-without-enumerating-every-face). |
| Q-D: finite-precision majorization | Bound inner objective suboptimality from normal-equation residuals, account for loss-evaluation error, and require improvement larger than those uncertainties. Refitting every trial is not the first remedy. [Section 24](#24-q-d-majorization-with-floors-and-finite-precision). |
| Q-E: balance rules | Remove a universal minimum fraction if desired, but keep distinct support, geometry, optimization and resource safeguards. Count plus rank alone does not cure saturated gates. [Section 25](#25-q-e-minimum-side-count-versus-fraction). |
| Q-F: online stealing estimates | Start with coarse phase-specific throughput estimates; use uncertainty-aware makespan improvement and one-owner processing leases. Make low-volume ART work stealable before claiming MPI efficiency. [Section 26](#26-q-f-calibrating-steal-costs-and-preventing-bouncing). |
| Q-G: saddle curvature | Do not automatically buy gradient/Hessian evaluations. Exact absolute-Hessian geometry can itself be isotropic on `x1*x2`; use available derivative information as candidates and compare hard fits. [Section 27](#27-q-g-covariance-hessian-splitting-on-saddles). |
| Q-H: continuity | A bounded beam with blending does not by itself guarantee continuity. Choose the representation from the application's stability and derivative requirements. [Section 28](#28-q-h-hard-tree-discontinuities-and-dynamical-applications). |
| Q-I: two modes | Use two named, validated algorithm plans over one executor and shared numerical functions. Preserve legacy event order; do not expose arbitrary stage permutations. [Section 29](#29-q-i-two-algorithm-plans-over-one-executor). |

### 20.3 Remaining discrepancies that should be fixed before coding

**RNG identity and RNG position are different.** Design section 31.9 correctly
names purpose-specific streams. Section 36 then says the node key determines the
stream and no separate RNG state is needed. A node can request several top-ups,
warm-up extensions, validation attempts or split retries. Restarting each operation
from the same key can repeat samples. Use either explicit counters or fully indexed
draws including purpose, attempt, chain and step. If migration is allowed only before
any work on a node begins, counters can be reconstructed at that boundary, but that
restriction must be stated. It is incompatible with silently persisting partially
advanced chains across resumable stages without also tracking their progress.

**The proposed role enum mixes unrelated axes.** A row can be both inherited and
training, or both new and validation. `train / validation / inherited / new` cannot
be one exclusive four-valued role. Use a training/audit membership field and separate
provenance flags or IDs. Define what happens to a parent's validation samples after
its model fails. They can be released into future training, but they are then spent
as validation data and cannot certify a descendant trained or selected using them.

**The budget counts must exclude validation rows.** The existing fit functions and
segmented reductions consume every row in their supplied cluster. Adding a role
column does not make them respect it. Region mode needs a training-only packed view
or explicit weighting/masking in every relevant fit, statistic and split search.
Store separate training and validation counts. In particular, validation outputs
must not enter threshold selection, normalization based on outputs, or centroid
Taylor-residual ranking. Training and audit samples can share a transfer buffer, but
must not silently share a statistical role.

**Required centroid values are the union of component requirements.** Design
sections 25 and 37 imply that an affine routed model can omit `f(c)` based on the
leaf model alone. That is true only if the splitter and other stages also do not
need it. The current `CovarianceHessianSplit` reads `data.f_c` in both its exact-
centroid-gradient and local-LSQ-gradient paths. The latter centers its cross
covariance and Taylor residual at `f(c)`. B1/B2 reusing that metric still need
centroid values. Declare requirements on every consuming component; compute their
union. Replacing the metric's oracle anchor with a fitted intercept would be a new
algorithm and needs a separate comparison.

**Centered predicates need a complete representation.** Section 31.7 permits
`a^T(x-c_ref)-tau`, but the fixed schema persists only `(w,z)`, and the constraint
row has only `d+1` values. Converting to `z=tau+a^T c_ref` is equivalent in exact
arithmetic but can lose the numerical advantage of centering. Choose one canonical
coordinate system and serialized predicate format. A fixed run/root normalization
stored once is an economical option. If node-local anchors remain part of the
predicate, store those anchors and use the same representation in routing and
chord calculations. The sampler must describe the same cells the router owns,
subject to the documented boundary tolerance.

**The actual legacy ordering must remain explicit.** The implementation performs a
preliminary fit/acceptance, computes `Q` and finalizes only provisionally accepted
nodes, then retests when finalization changes the model. Section 37 puts `FINALIZE`
before `ACCEPT` while labeling it unchanged. Reordering that work can change oracle
counts, allocations and numerical behavior. Keep the precise legacy sequence as a
plan. Region mode should finalize the candidate it will deploy before its final
audit, with a defined preliminary training screen if desired.

**Region mode changes more than two branches.** It changes root initialization,
training/audit views, centroid requirements, acceptance and terminal handling,
optional `Q`, augmentation, split verification, child state, depth/budget termination,
batch admission, output assembly and persistence. Those can be small named methods
over one executor. Pretending they are only pre-STATS and post-SPLIT branches would
hide important behavior in side effects.

**The region memory estimator is required from B2.** Part IV correctly says to replace
the old capacity estimate, while section 37 says the VRAM cap is shared unchanged.
Even affine B2 adds top-up, audit rows, path constraints, chain state, slacks and
temporary packing. Use the same admission mechanism with a different footprint
function. Sharing the scheduler does not require sharing the old byte formula.

**The declared MPI tags do not cover the declared columns.** There can be ten array
buffers in section 36: X, F, G, role, sizes, depth, root, path_code, constraints and
retries. Tags `111..119` supply nine distinct tags. A packed buffer or repeated
ordered messages can use fewer tags, but the wire format must say so. The initial
scatter also calls the same transfer function with another base tag. Use relative
offsets from that base and a schema/version header. Persisted leaves, transient jobs
and node records can share column descriptors without being one identical payload.

**Several required fields disappear between sections 31.3 and 36.** Sample IDs,
statistical weights, chain state, transforms, budget counters and fallback models
appear in the work-item description but not the minimal schema. It is reasonable to
defer some of them. State a restricted contract instead: for example, unweighted
samples, no mid-node migration, chain state local to a non-resumable node task,
per-node budgets only, and fallback from the fitted parent before any split is
committed. If global oracle budgets are promised, account for in-flight allocations
across ranks; a shared config limit alone does not enforce them.

**B1 needs a documented construction policy distinct from B2.** The experiment table
describes B1 as legacy sampling with recorded valid topology, while the two-mode
definition makes every region run top up before fitting. Changing roots from
k-means to grid or box is another factor. For a predictor comparison, build one
valid topological model and run both predictors on that exact artifact. Keep `Q`
and its required columns for that experiment. Call the grid-root B1 experiment a
new construction baseline rather than claiming it is the unchanged B0 model.

**Gate flooring has a small but consequential formula typo.** For nonzero `delta`,
the stable left weight is

```text
w_left  = delta + (1 - 2*delta) * sigmoid(-s)
w_right = delta + (1 - 2*delta) * sigmoid( s)
```

It is not simply `sigmoid(-s)`, as the sentence in design section 33.3 currently
suggests. The logit derivative is
`(1-2*delta)*sigmoid(s)*sigmoid(-s)`. Do not use
`tilde_pi*(1-tilde_pi)` as its replacement.

**A tiny side by sample mass need not be tiny in every feature direction.** The
recommendation to form the smaller-mass Gram directly is a heuristic. A low-weight
sample with a large feature norm can dominate a Gram direction. Near singularity,
either complementary subtraction can be inaccurate. Use feature scaling, solve
residuals and the direct two-Gram fallback. The side mass alone is not a numerical
certificate.

**Hard-candidate SSE needs a direct check near convergence.** The sufficient-statistic
formula `C-2 tr(Beta^T B)+tr(Beta^T G Beta)` subtracts large nearly equal terms for
an excellent affine fit. Roundoff can give a negative SSE or change the order of
nearly tied candidates. Use it to screen, then recompute the leading candidates'
residual sums directly over rows before accepting a small gain. This matters most
near the requested tolerance, where a false ranking can change the whole tree.

**A few inherited statements remain too strong.** Part III still says the analogy
to a polytope tree is exact for every AFINNI tree, despite the tie and root-assignment
caveats now stated elsewhere. Section 5 should describe `|dx^T H dx| <= dx^T |H| dx`
as an upper-envelope relation for indefinite curvature, not equality to the Taylor
residual. Section 20 should say objective inconsistency and solve errors are possible
causes of loss increases; the majorization theorem does not establish the cause of
each historical increase without examining its trace. A finite hit-and-run run is
an approximate sampler with uniform stationary measure, not automatically a set of
uniform iid observations.

**Root identity and upper-hierarchy identity need one convention.** Box/grid roots
can use an explicit root table followed by per-root path codes. A learned shallow
binary hierarchy can instead use one root and full path prefixes through that
hierarchy. Do not reset every frontier cell to `path_code=1` and also expect ordinary
parent lookup to reconstruct its upper-tree ancestors. Keep the root selector
separate, or keep the full prefix. Depth limits and constraint lengths must use the
same choice. With a positive signed int64 sentinel code, depth is limited to 62;
using two root/path fields does not remove the path field's own limit.

## 21. Q-A: validation under dependence and adaptive testing

### 21.1 The statement to make first

For version one, make a precise global claim about a frozen deployed predictor using
fresh iid points from the declared query measure. For a box and uniform measure,
generate those points directly. Evaluate the final router and models, including
unresolved leaves, and count tolerance violations. Report local MCMC validation as
an empirical training diagnostic unless stronger sampling assumptions are justified.

This gives a useful result without pretending that a chain diagnostic is a proof.
It also tests the actual deployed selection rule, domain normalization and boundary
behavior. It does not certify every leaf individually or establish a uniform bound
over every point in the domain.

For zero failures at confidence 95%, a global violation probability below 1% requires
299 iid audit points, and below 0.1% requires 2,995. These numbers come directly from
`ceil(log(0.05)/log(1-p0))`. For observed failures, use an appropriate one-sided
binomial interval, not the zero-failure formula.

If a run is changed because of this audit, the next reported audit must be fresh or
part of an explicitly multiple-testing-valid procedure. A repeatedly inspected
"untouched" set is no longer untouched.

### 21.2 Adaptively chosen cells are compatible with fresh conditional tests

Let `H_t` be everything known before validation attempt `t`, including its selected
cell, fitted coefficients, normalization and predictor. Let `p_t` be the violation
probability of that frozen candidate under its declared cell measure. If fresh
validation gives a conditionally valid upper bound `U_t` satisfying

\[
\Pr(p_t>U_t\mid H_t)\le\delta_t,
\]

then a union bound controls all attempted statements when the allocated probabilities
sum to at most `delta_total`. The fact that the cell was chosen adaptively from
earlier training does not invalidate this argument. Reusing the data that chose the
candidate would invalidate the assumed conditional test.

One allocation is `delta_t = delta_total * 6/(pi^2 t^2)` for attempts numbered from
one. In MPI, unique node/attempt keys can avoid a centralized test counter. With `R`
fixed roots, depth `h`, and attempt index `j` starting at one, an example is

\[
\delta_{root,path,j}
=\frac{\delta_{total}}R\,2^{-(2h+1)}\,
\frac6{\pi^2j^2}.
\]

There are at most `2^h` paths per root at depth `h`, and the sum of their depth
weights is one over all depths. The sum over attempts is also one. This allocation
therefore remains valid for adaptively created finite trees. It can be conservative
for deep leaves; it is an example of a correct budget, not an optimized allocation.

Allocate the error budget across attempted tests, including failed or abandoned
candidate tests, not just the leaves that survive into the final tree. Replacing a
validated model or predictor creates a new candidate and a new test.

### 21.3 What a mixing-error bound would provide

Suppose the joint distribution of `m` validation points is within total variation
distance `eta` of `m` independent target draws, conditional on the frozen history.
For a true violation probability `p`,

\[
\Pr(\text{zero violations})\le(1-p)^m+\eta.
\]

Thus a zero-failure upper bound at failure probability `delta > eta` is

\[
U=1-(\delta-\eta)^{1/m}.
\]

This is a direct consequence of total variation controlling the probability of an
event. It exposes the missing quantity: a bound on joint sampling error. Marginal
stationarity of every point is insufficient when the points are dependent.

If independently run chains each supply one endpoint, and every conditional endpoint
law is within `epsilon_i` of the target, a coupling bound gives
`eta <= sum(epsilon_i)`. Those must be actual bounds for the chosen seeds, kernel and
run lengths. Observed autocorrelation is not such a bound.

There is another useful special case. If endpoints are iid from one fixed approximate
law `Q`, with `TV(Q,nu_R) <= epsilon`, then a binomial upper bound `U_Q` implies

\[
p_{\nu_R}\le\min(1,U_Q+\epsilon).
\]

The confidence level concerns the iid endpoint experiment; `epsilon` bounds its
distributional bias. This does not apply unchanged to several dependent outputs
from each persistent chain or to adaptively pooled chain lengths.

### 21.4 Why ESS is not a substitute for that argument

Integrated autocorrelation and ESS describe the uncertainty of selected observables
under assumptions about the chain. They do not universally describe tail events,
unvisited parts of a cell or nonstationary starting bias. A chain that never visits
the failure region produces an all-zero violation indicator, from which a useful
rare-event ESS cannot be inferred.

Finite-sample concentration results for Markov chains do exist, but their constants
depend on properties such as mixing time or a spectral gap. Those quantities cannot
be replaced by an unqualified empirical ESS estimate. See
[Paulin's Markov-chain concentration analysis](https://arxiv.org/abs/1212.2015).
Add such a bound only for a sampler/geometry class where its assumptions and
constants can be defended.

### 21.5 Sequential testing solves a different problem

If validation samples are iid and the predictor is fixed, a confidence sequence
allows checking a bound repeatedly and stopping when enough evidence has arrived.
For example, construct a Bernoulli confidence sequence and accept when its upper
endpoint is below the desired violation rate. It can save oracle calls for clear
successes or failures. [Howard et al.](https://arxiv.org/abs/1810.08240) develop
time-uniform confidence sequences under stated probabilistic assumptions.

That machinery does not by itself repair MCMC bias, arbitrary dependence or changing
the predictor while keeping the same test stream. Use it only after establishing
the sampling contract. For the first version, fixed-size fresh tests plus explicit
test-budget allocation are simpler to implement and audit.

## 22. Q-B: warm-up from inherited seeds

### 22.1 There is no dimension-only answer

No universal `20`, `50*d` or `100*d` step rule guarantees an adequate warm start in
an arbitrary learned polytope. The relevant variables include anisotropy, distance
to the boundary, bottlenecks in the chain's movement, the chosen direction
distribution, and the observable being estimated. A node can be easy for coordinate
means and difficult for the rare region where the surrogate fails.

Training and validation need different language. Biased but feasible training points
can still be useful, provided the approximation is evaluated separately. Calling a
validation endpoint representative enough for a probabilistic claim requires a
sampling argument beyond its usefulness for fitting.

### 22.2 A practical geometry-only warm-up policy

Use normalized coordinates and a bounded geometry pilot before querying new oracle
values. Start several chains from dispersed feasible points. Inherited points are
one source; coordinate-extreme or random-linear-objective feasible points followed
by movement toward an interior point can be another, when their setup cost is
justified. LP extrema themselves lie on boundaries and are poor direct chain starts.

Freeze the direction transform within each diagnostic/production segment. Monitor
normalized coordinates, several fixed random projections, projections along important
cut normals, squared radius, and normalized slacks to relevant faces. These probe
different possible failures. Merely monitoring raw coordinate means can miss a
rotated thin direction.

Run pilots in increasing blocks, such as doubling a configured block length, and
compare both between-chain and early/late behavior. Rank-normalized split-R-hat,
bulk/tail ESS, and projection/slack quantiles are useful diagnostics. The
[work of Vehtari et al.](https://arxiv.org/abs/1903.08008) explains improved R-hat and
localized efficiency diagnostics. Use published numerical thresholds as starting
diagnostics, not a theorem that this cell has mixed.

Require sufficient observations for those diagnostics to be meaningful. Four chains
with five retained states each cannot produce a persuasive convergence assessment.
If all chains start at the same point, agreement among them is especially weak
evidence that the rest of the cell was explored.

Stop escalating at a configured geometry-work budget. If diagnostics remain poor,
recondition, try another sampler, request better seeds, or finish with
`validation_status=insufficient_sampling`. Do not mark the node accepted merely
because its chains exhausted their budget without finding an error.

### 22.3 Diagnostic stopping and production sampling should be separated

Choose the production kernel, transform and run length from a pilot, then run a
separate production segment. Otherwise stopping exactly when recent diagnostics
look favorable selects the retained sample in an additional data-dependent way.
Separating segments does not magically make the production seed stationary; it
makes the policy explicit and its bias easier to analyze and test.

If a rigorous mixing bound is available, choose production run length from that
bound and the validation error budget in section 21. Without one, retain the
diagnostic status and rely on the independent final audit for the stated global
accuracy claim.

### 22.4 Independent short chains and persistent chains serve different purposes

Independent random streams from the same fixed seed point produce conditionally
independent endpoints when run for fixed lengths with a fixed kernel. They can still
all have the same starting bias. This is a useful clarification of the first review:
sharing a deterministic starting point does not itself create endpoint dependence;
it does not solve mixing either.

Persistent chains amortize warm-up and are attractive for training and geometry
pilots across many samples. One-endpoint-per-chain experiments are easier to reason
about for validation, if endpoint bias can be bounded. Keep both possibilities in
the sampler experiment, with oracle calls counted only for retained evaluated points.

No finite set of moment checks proves mixing in all learned high-dimensional cells.
Their purpose is to detect failures and allocate effort. Report diagnostic failures
and compare several kernel choices on difficult cells instead of hiding this limit
behind one warm-up constant.

## 23. Q-C: exact Voronoi geometry without enumerating every face

### 23.1 A global minimal face list is not required

For a root center `c_i`, its cell satisfies, for every other center `c_j`,

\[
(c_j-c_i)^Tx\le\tfrac12(\|c_j\|^2-\|c_i\|^2).
\]

For modest root counts, store the centers once and evaluate these inequalities in
tiles. That avoids persistent `R*(R-1)*(d+1)` plane storage. The sampler can reduce
chord bounds without materializing every root's faces. Its arithmetic still depends
on `R`; the gain is memory and implementation simplicity.

Low-dimensional full Voronoi construction is another option. SciPy's
[Voronoi interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html)
exposes `ridge_points`, the pairs of centers sharing ridges. The
[Delaunay interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)
provides related adjacency information and documents degeneracy/precision caveats.
This is an alternative to individually solving a redundancy LP for every pair.

Neither API makes globally sparse face lists cheap in every dimension and point
configuration. Full diagram construction can spend effort on vertices and simplices
that sampling never needs. Clipping by the domain can make additional faces redundant.
Degenerate, duplicate or perturbed centers also require explicit handling. Limit the first topology-based implementation to configurations where its construction
time and output size are measured to be manageable, not enforce an arbitrary hard
dimension cutoff as if dimension alone determined the cost.

### 23.2 Certified removal of a face is possible but can be expensive

To test whether `a_j^T x <= b_j` follows from a current retained region, maximize
`a_j^T x-b_j` over that region. A certified nonpositive upper bound proves redundancy.
Use conservative margins and retain uncertain constraints. A merely feasible primal
LP point is a lower bound on this maximum, so it cannot prove redundancy; a suitable
dual bound can. SciPy's [linear-programming interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
supports the necessary optimization form, though ordinary floating-point solver
success is not an exact-arithmetic certificate.

Do not independently delete every constraint that appears redundant relative to all
the others in a degenerate system. Duplicate constraints can each be redundant only
because the other exists. Remove them in a validated sequence or verify the final
retained system. A practical batched GPU path should prefer extra correct constraints
to numerically uncertain pruning.

### 23.3 A useful third option: certify only the chord that will be sampled

For a feasible point `x` and direction `u`, suppose box/path constraints and a cached
subset of root faces give an interval `I_S=[lo,hi]`. The exact root-cell chord
`I_full` is a subset of `I_S`.

Check both endpoints `x+lo*u` and `x+hi*u` against all centers, using an exact nearest-
center query under the selected floating-point convention. If `c_i` is no farther
than any other center at both endpoints, both satisfy all root inequalities.
Each inequality is affine along the line, so it holds throughout the segment. Hence

\[
I_S\subseteq I_{full}\quad\text{and}\quad I_{full}\subseteq I_S,
\]

which proves equality in exact arithmetic.

If an endpoint violates ownership, take a closer center as a separating constraint,
add its bisector, tighten the interval and repeat. Use a bounded number of rounds,
then fall back to the full tiled chord calculation. Draw the uniform chord position
only after certification is complete. The procedure then produces the same chord
as the full-constraint algorithm; it does not introduce a rejection-induced sampling
bias by shortening an uncertified chord heuristically.

Important details:

- Compare distances, not just the returned nearest-center index. A boundary point
  can tie with another center and still lie in the closed cell.
- Use finite bounds from the domain before checking endpoints.
- Keep uncertain floating-point endpoint tests conservative and use the full path
  as a reference/fallback. An approximate nearest-neighbor miss is not a certificate.
- Cache useful faces, but revalidate every new direction's chord. A face irrelevant
  for one chord can be essential for another.
- Root centers are fixed. Sampling does not update the centers or the ownership rule.

This derivation is an alternative worth testing, not a measured GPU speedup.
The exact nearest-neighbor index can make endpoint checks economical in favorable
low-dimensional cases. On the current high-dimensional GPU path, an exact query may
scan every center, so repeated endpoint checks can cost more than one full tiled
constraint pass. Measure and dispatch accordingly.

### 23.4 Recommendation

Keep grid/shallow-hierarchy roots as the first scalable baseline. Implement Voronoi
roots with all-center tiled chords as the correctness reference. Try cached-face
endpoint certification and low-dimensional adjacency construction only if root-face
work is a measured bottleneck. There is no need to solve global face minimization
before implementing correct Voronoi sampling.

The lazy chord construction was checked on 300 random box-clipped examples in dimensions
2, 5 and 10, with 32 centers and eight additional feasible oblique constraints. It
matched the full-constraint endpoints to a maximum absolute difference of
`2.22e-16`. Those small CPU cases required at most four verification rounds. This
checks the implementation of the argument on ordinary cases; it does not establish
worst-case complexity, a robust finite-precision certificate, or a GPU performance
advantage.

## 24. Q-D: majorization with floors and finite precision

### 24.1 Distinguish three numerical questions

The exact majorization inequality remains true for the consistent floored objective
in section 8.2: minimizing over coefficients cannot exceed the value at fixed
coefficients. Finite precision introduces three separate uncertainties:

1. The current fitted coefficients may not attain the current profiled minimum.
2. The reported frozen/refitted losses may be inaccurate.
3. The computed gradient may not be a sufficiently accurate descent direction.

Checking that the new refitted loss is close to or below the frozen loss addresses
only part of the second point and the quality of the new solve. It does not by itself
bound the error in the current profiled value. A fixed `1e-12` acceptance tolerance
cannot have the same meaning for all output scales, node sizes and conditioning.

### 24.2 An exact residual identity for the inner problem

For one side, write

\[
H=\Phi^TW\Phi+\lambda P,\qquad C=\Phi^TWY,
\qquad R=C-H\widetilde B,
\]

and assume `H` is positive definite. If `B_star=H^-1 C`, expansion of the quadratic
objective gives

\[
F_s(\widetilde B)-F_s(B_*)
=\frac1n\operatorname{tr}(R^TH^{-1}R).
\]

The identity includes multiple output columns and uses the same `1/n` normalization
as the proposed soft objective. Sum the two sides to get the total inner gap. If a
valid lower eigenvalue bound `H >= mu I` is available, then

\[
0\le\epsilon_{inner}
\le\frac{\|R_L\|_F^2}{n\mu_L}
+\frac{\|R_R\|_F^2}{n\mu_R}.
\]

With `P=I` and positive fixed `lambda`, `mu=lambda` is a simple exact-arithmetic
lower bound. It can be conservative. If the intercept is unpenalized or ridge is
zero, that bound is not available; use a justified lower bound from the weighted
design or retain a weaker numerical diagnostic. Do not label a condition estimate
an exact lower spectral bound.

For an inexpensive diagnostic, solve `H Z=R` using the existing factors and evaluate
`tr(R^T Z)/n`. It estimates the gap more tightly, but shares the factorization's
numerical limitations. For an actual certificate, account for residual/eigenvalue
bound error, rather than assuming those estimates are exact.

### 24.3 A sufficient decrease rule with explicit uncertainty

Let `F_old` be the objective at the old computed coefficients, and suppose

\[
J_{old}\ge F_{old}-\overline\epsilon_{inner,old}.
\]

Let `Q_new` be the new frozen-coefficient objective. By definition,
`J_new <= Q_new`. Let hats denote evaluated values, and let
`e_old`, `e_new` bound their absolute evaluation errors. Then the sufficient condition

\[
\widehat Q_{new}+e_{new}
\le\widehat F_{old}-e_{old}
-\overline\epsilon_{inner,old}-m,
\qquad m>0,
\]

implies `J_new <= J_old-m`. This is the finite-accuracy form of the majorization
argument. It makes the acceptance margin depend on solve quality and loss accuracy.
One can compare a refitted candidate value instead of `Q_new`; any coefficients give
an upper bound on the new profiled minimum, provided their loss is evaluated correctly.

In practice, use the normal frozen Armijo check to propose a step, then require
its measured improvement to dominate the estimated inner gap and numerical error.
The extra check should get stricter as the requested decrease approaches the numerical
resolution. If those terms are only engineering estimates, describe the result as
numerically monitored descent, not a formal certificate.

Once a step is accepted, verify that the new coefficient solve does not degrade the
frozen value beyond its numerical uncertainty. If it does, retain the old coefficients
as a valid candidate at the new gate while repairing the solve, or reject the step.
Do not install a known-worse refit simply because the exact solve would have improved it.

### 24.4 Computing the uncertainty terms

Evaluate the loss as weighted squared residuals plus penalty using stable reductions.
Avoid the subtractive sufficient-statistic SSE formula for the final acceptance
comparison. When practical, evaluate old and new frozen losses together from cached
residuals; their difference can avoid subtracting unrelated large totals.

The standard floating-point summation factor
`gamma_k = k*u/(1-k*u)` bounds a known sequence of `k` operations under the usual
model, where `u` is unit roundoff. A real bound must also include prediction-dot-product
errors, squaring, weights, reductions and regularization. A GPU reduction with atomics
or a library routine does not supply a proven bound merely because its dtype is
float64. Pairwise/fixed-order reductions can make both error and reproducibility
easier to characterize.

Recompute the solve residual from the feature data when checking assembly accuracy:

\[
R=\Phi^TW(Y-\Phi\widetilde B)-\lambda P\widetilde B.
\]

Using only the already rounded Gram can report a tiny residual for a system whose
formation lost important information. Track a normalized backward residual as an
engineering diagnostic, but remember that small backward error does not imply small
coefficient error for an ill-conditioned system.

Do not use the application's approximation tolerance as the solver tolerance. The
former measures surrogate error against the oracle; the latter controls whether
the proposed optimization step is numerically meaningful.

### 24.5 When to refit during line search

The fallback sequence is:

1. Reject nonfinite values and inconsistent gate/loss definitions immediately.
2. Recompute questionable losses with stable reductions and repair the inner solve
   using iterative refinement or a QR/SVD path as appropriate.
3. Recompute the current state and gradient after a repaired solve.
4. If frozen screening remains too conservative, try a bounded number of refitted
   candidate evaluations with the same fixed objective and uncertainty check.
5. If meaningful progress is below numerical resolution or the optimization budget
   is exhausted, stop polishing and return the best valid hard candidate.

Refitting every trial does not cure a mismatched objective, incorrect derivative,
poorly formed Gram or inaccurate old solve. Increasing ridge for a difficult node
changes the objective. If used as recovery, restart that node's optimization under
the new fixed ridge and recompute its reference values; do not compare losses across
two different ridge strengths as if they were one descent sequence.

The coherent floor helps keep inner systems supported, but it does not create a
nonzero derivative in a saturated sigmoid. That issue belongs to Q-E.

### 24.6 Numerical check

The residual identity was checked with 64 samples, eight features, three outputs,
positive weights and fixed `lambda=0.2`. For a deliberately perturbed coefficient
matrix, direct objective subtraction gave an inner gap of
`0.001864791470271854`; the residual identity gave
`0.001864791470271584`, differing by `2.70e-16`. The simple `mu=lambda` upper bound
was `0.33798649019511245`, valid but much looser. This illustrates why the easiest
rigorous bound may be too conservative for accepting small optimization steps.

## 25. Q-E: minimum side count versus fraction

### 25.1 Count plus rank is not enough to prevent saturation

Dropping a universal fractional-balance rule is reasonable when small-volume regions
can be sampled directly. But a minimum hard count and a rank check apply to the
eventual partition; they do not keep the soft optimizer away from saturated gates
while it is iterating.

For the coherent floor,

\[
\frac{d\widetilde\pi}{ds}
=(1-2\delta)\sigma(s)(1-\sigma(s))\longrightarrow0
\]

at either extreme. A floored soft model can be well-conditioned because every point
has positive weight, yet its hard child can have almost no samples. Ridge and floors
are not evidence that a hard child is statistically supported.

Drop the mandatory *constant fraction* in the first region mode, keep a
minimum inherited support count for trustworthy hard scoring, and retain the B3
candidate as the feasible incumbent throughout B4. A saturated gradient is then
`soft_optimizer_stalled`, not evidence of convergence to a useful partition or
acceptance of a leaf.

### 25.2 Separate four safeguards

| Safeguard | Purpose | First implementation |
|---|---|---|
| Geometric validity | Two represented regions with a reproducible predicate | Distinct-score gap, finite plane, feasible inherited seeds, boundary checks |
| Statistical support | Enough information to rank a hard candidate and fit its children | Minimum training-side count and rank/conditioning checks, followed by child top-up and validation |
| Optimization stability | Avoid wasting iterations near invalid saturated gates | Feasible hard incumbent, bounded steps/iterations/restarts, diagnostics on soft mass and derivative activity |
| Resource/progress control | Avoid arbitrarily deep sliver peeling | Depth, sample/oracle and split-attempt budgets, progress/gain criteria and explicit terminal statuses |

A point count is not a certified volume fraction when points are inherited or MCMC
sampled. Do not interpret the support rule as a geometric minimum-volume guarantee.

### 25.3 How permissive should the inherited count be?

For the first affine implementation, require enough inherited samples to make the
hard candidate score credible. A count at least the affine feature dimension, with
additional margin and conditioning checks, is a reasonable initial policy. The
margin is an experimental setting, not a theorem. Child top-up then supplies the
full training budget before deciding acceptance.

An even more permissive algorithm could allow a child with only one feasible seed,
top it up, and only then score or accept the split. That algorithm spends oracle
calls to evaluate candidate partitions and needs an explicit candidate-evaluation
budget. It should not be smuggled into a supposedly geometry-only hard search.
It also cannot treat a one-point hard training loss as a fair estimate of child risk.

Thus region top-up can eventually recover small, important caps that ART's minimum
fraction rejects, but it does not make their immediate inherited-data scores reliable.

### 25.4 Keep the optional balance control measurable

Soft mass barriers, offset restrictions or balance penalties can be useful options,
but they change the optimized objective. Include their gradients and losses
consistently if enabled. If an offset is restricted to an interval determined by
hard projection order statistics, treat that as constrained/nonsmooth optimization;
the unconstrained sphere-gradient argument no longer describes the whole update.

Log the hard-count distribution, soft derivative activity, fraction of optimizer
steps with invalid hard partitions, and cases where a fractional rule would reject
the best supported candidate. Evaluate localized features and curved-interface
benchmarks. This distinguishes a useful geometric restriction from a rule that
merely hides optimizer failure.

## 26. Q-F: calibrating steal costs and preventing bouncing

### 26.1 Start with costs that the timings can actually identify

Estimate phase time with nonnegative, scaled features, separately for relevant
hardware/model families. Useful groups are oracle time versus new points, sampling
time versus chain-steps and constraint dimension, and optimizer time versus feature
count, node size and active iterations. Include a fixed overhead term and measured
transfer/packing time.

Do not blindly fit every coefficient in the proposed formula at once. Under ART's
fixed `n=kappa*p` budget, `n*p^2` and `p^3` are proportional. Observations at that
budget cannot identify separate Gram and solve coefficients. Use separately timed
phases or combine the terms. The same warning applies to warm-up and output steps
when their ratio never changes.

For the initial version, per-bucket medians or trimmed exponentially weighted
throughput estimates are sufficient. A nonnegative ridge/robust regression can
follow when enough data span the features. Keep estimates of variability and a
minimum-observation threshold. Record cold compilation, allocation and exceptional
solver events separately so one such event does not become the ordinary node cost.

Use completed-node/phase observations to update predictions and report prediction
error. Avoid presenting an uncalibrated formula as a measured load. For heterogeneous
workers, predict the job's cost on the receiving worker as well as its source cost.

### 26.2 Model the current task, not an unknowable subtree

Predict the next bounded node-processing task first. Future child count and depth
are unknown; overpredicting an entire difficult-looking subtree can distort grants.
Optionally use measured split probability and expected child work as a bounded
secondary estimate once there is evidence it improves balance.

GPU cohort time is not perfectly additive across nodes. Use marginal cost estimates
appropriate to the cohort's occupancy, or a coarse cohort cost model, rather than
assuming that the sum of timings of isolated nodes predicts a batched solve.

### 26.3 Choose a transfer by predicted makespan improvement

Let `T_v` and `T_h` be the victim's and receiver's predicted time to finish their
currently owned work. For a candidate grant `S`, let `w_v(S)` be the removed source
work, `w_h(S)` the added receiver work, and `t_move(S)` include packing, transfer,
unpacking and any unavoidable cold setup. A simple conservative comparison is

\[
M_0=\max(T_v,T_h),
\qquad
M_1=\max(T_v-w_v(S),\ T_h+t_{move}(S)+w_h(S)).
\]

Grant only if `M0-M1` exceeds a margin for estimation uncertainty and scheduling
overhead, and the receiver has capacity. If the victim blocks while packing/sending,
add that delay to its remaining branch too. If transfers overlap useful work, the
model can later account for that measured overlap.

For identical workers, an idle receiver and an idealized fixed move cost, equalizing
the finish times gives `w=(T_v-t_move)/2`. That explains why half-load stealing is a
reasonable starting heuristic, and also why it is not correct to transfer exactly
half the points regardless of cost. Round to whole ready jobs and retain enough
local work to avoid making the victim immediately idle.

### 26.4 Hysteresis and ownership rules

Use a minimum useful-work duration, a transfer-benefit ratio and an uncertainty
margin. Their values should be tuned from measured latency, not hard-coded to the
old million-point threshold. Apply a short retry backoff after denials, and refresh
stale load information before repeatedly asking the same victim.

When a job is granted, its receiver should process it through at least one node task
before offering the same job onward, unless it cannot safely run it. Newly generated
children are new jobs and may be stealable. This ownership lease prevents traffic
from bouncing the same untouched job among idle ranks. Permit explicit exceptions
for capacity or worker failure rather than hiding those behind time-based heuristics.

Keep grants one-way and explicit. A numeric cost estimate does not replace the
existing request/grant/deny protocol or its retirement accounting. Time estimates
are unsuitable as the only liveness signal: a tiny task whose estimated cost rounds
to zero is still work. Track outstanding jobs separately from estimated seconds.

### 26.5 What should be implemented now?

The full calibrated model can be deferred. Making ART-size queues eligible for
stealing should not be deferred if MPI performance is part of the first claim.
Expose mode-specific thresholds and test forced transfers with hundreds or thousands
of points. Likewise, retain bounded polling points in long sampling/optimization
loops when B4 is enabled. Without them, the old scheduler can be logically correct
and almost useless for the new workload.

A minimal useful version is static per-model cost weights, configurable grant size,
measured transfer duration, no immediate re-donation, and request-latency metrics.
Use that baseline to decide whether online calibration pays for its complexity.

## 27. Q-G: covariance-Hessian splitting on saddles

### 27.1 An exact Hessian does not automatically solve the direction problem

For `f(x1,x2)=x1*x2`,

\[
H=\begin{pmatrix}0&1\\1&0\end{pmatrix},
\qquad |H|=I,\qquad |H|^{1/2}=I.
\]

On a centered isotropic square, `M C M` is therefore isotropic too. The exact-
Hessian metric has no distinguished leading direction. Returning a basis vector is
an eigensolver tie choice, not a discovery of the saddle's signed structure.

The low-absolute-residual sample set near the axes has the same reflection and
coordinate-exchange symmetries in the population. Its covariance is also proportional
to the identity. Its ellipsoid interpretation is misleading, but it does not follow
that replacing it with `|H|` will yield a better split on this example.

This sharpens the first review: the invalid level-set interpretation is a reason to
test the heuristic, not a proof that the heuristic performs worse on every saddle.

### 27.2 A concrete hard-fit comparison

Under uniform measure on `[-1,1]^2`, the best parent affine model for `x1*x2` is zero,
with MSE `1/9`. Splitting at `x1=0` and fitting each half gives slopes `+/-1/2` on
`x2` and total MSE `1/36`.

For a diagonal split at `x1+x2=0`, the positive triangle's best affine fit is

\[
\widehat f=\tfrac35x_1+\tfrac35x_2-\tfrac25.
\]

Its conditional MSE, and the total symmetric split MSE, is `7/225`, slightly larger
than `1/36`. These triangle moments were verified symbolically. Even choosing signed
Hessian eigenvectors is not universally the best one-step affine partition; the
domain and child regression objective matter too.

### 27.3 Recommended policy

Use the direction bank and hard fit scoring as the default protection against a weak
curvature proxy. Include ordinary covariance directions and reproducible random
directions; include signed/absolute Hessian-derived candidates when derivatives
are already available or affordable. Near repeated eigenvalues, consider a few
directions spanning the tied eigenspace instead of attaching meaning to one arbitrary
returned eigenvector.

Gradient-LSQ estimates can be valuable when point gradients are already required by
the leaf model. They bring sample and conditioning requirements of their own and
increase stored/transferred data by `n*q*d` numbers. An exact Hessian callback can
be an excellent benchmark when cheap analytically, but its cost for a physical oracle
may dominate the proposed savings. Do not request expensive derivatives solely
because the residual heuristic might encounter a saddle.

For vector outputs, derivative candidates should include relevant scaled components
or directions from an explicit combined criterion. The existing one-component
selection is itself a scientific choice, not a complete account of all output
curvatures.

If cross terms dominate, a small selected quadratic basis or low-rank quadratic
leaf may avoid extensive splitting. For `x1*x2`, adding the single interaction
feature represents the function exactly in ideal arithmetic. That is an argument
for testing model enrichment alongside splitting; it is not an argument for a dense
quadratic feature set at high dimension.

## 28. Q-H: hard-tree discontinuities and dynamical applications

### 28.1 A branching cap is a compute limit, not a continuity condition

Consider a smooth blend across a normalized signed distance `s`:

\[
\widehat f(x)=(1-\alpha(s))f_L(x)+\alpha(s)f_R(x).
\]

If `f_L`, `f_R` and `alpha` are continuous on the overlap, this expression is
continuous. But a query-dependent rule that keeps only the largest `B` branches can
drop a nonzero-weight contribution when the selected branch set changes. Renormalizing
the survivors does not in general restore continuity. Thus a bounded beam alone is
not enough.

Compact-support transition weights that vanish at the boundary where a contribution
is omitted can preserve continuity. Their maximum overlap multiplicity still needs
analysis or measurement; arbitrary intersecting oblique bands can create many active
branches. A global soft mixture over every leaf is continuous under suitable models
and gates, but may destroy the cheap routing advantage.

### 28.2 A narrow band can produce a large derivative

Differentiating the two-model blend gives

\[
D\widehat f=(1-\alpha)Df_L+\alpha Df_R
+(f_R-f_L)(\nabla\alpha)^T.
\]

For a transition width `b`, `||grad alpha||` is typically of order `1/b`. Even if both
models have value error at most `epsilon` throughout the band, their disagreement
can be `2*epsilon`, producing an extra derivative term of order `epsilon/b`.
Shrinking the band to save work can therefore worsen Jacobian accuracy and effective
stiffness. Validation must include the overlap, derivative behavior where relevant,
and the actual branching/pruning rule.

There is also a geometric issue: a query routed into both children can lie outside
one child's training region. That child subtree can select models in places it never
sampled. Training/validation must cover the intended extension or overlap, rather
than assuming that soft routing makes those extrapolations reliable.

### 28.3 Distinguish a learned flow map from a learned vector field

For a discrete flow map `Phi` with Lipschitz constant `L_Phi`, a uniform one-step
surrogate error `epsilon` along the relevant states gives

\[
E_{k+1}\le L_\Phi E_k+\epsilon.
\]

Iterating yields the geometric-sum accumulation bound. Hard-tree jumps do not
invalidate this inequality if its uniform one-step error premise holds. The problem
is that random pointwise validation may miss those jumps or their nearby large errors.
Chaotic or strongly expanding maps amplify any approximation error, continuous or not.

For a vector field used inside an ODE solver, discontinuity can additionally affect
existence/uniqueness assumptions, adaptive time-step control and sensitivities. For
well-defined trajectories with true field Lipschitz constant `L` and uniform field
error `epsilon`, the usual perturbation argument gives
`E(t) <= epsilon*(exp(L*t)-1)/L` from equal initial states, with limit `epsilon*t`
when `L=0`. This conditional error bound does not prove that an arbitrary discontinuous
surrogate field has a unique classical solution. Do not use it as that guarantee.

For applications that differentiate the surrogate, a value-accuracy test alone is
insufficient. Measure Jacobian errors, boundary jumps, trajectory errors and any
physical invariants relevant to the application.

### 28.4 What continuity-constrained affine fits require

Across a full-dimensional shared face `w^T x=z`, two vector affine functions agree
everywhere on that face if their coefficient differences have the form

\[
A_R-A_L=v w^T,\qquad b_R-b_L=-v z
\]

for some output vector `v`. This is a tractable sibling constraint. Both children
can be fitted jointly by the rank processing the parent, so initial enforcement
does not inherently require a global MPI solve.

The difficulty is preserving it after independent descendants are fitted. A refined
subtree must preserve the appropriate boundary trace on inherited interfaces, or
neighboring subtrees must coordinate changed traces. Simply making every immediate
sibling pair continuous at its creation is not enough if later fits overwrite their
interface behavior.

A hierarchical residual representation whose refinements vanish on protected
interfaces, or a continuous basis with constrained coefficients, can make those
invariants explicit. It changes approximation flexibility, fitting and inference.
A global sum of hinge functions is another continuous construction, but without
local support it can require evaluating many terms per query. None is a free
modification to the existing independent-leaf tree.

### 28.5 Decision for this project

Keep hard routing as the first reference, and add boundary/trajectory stress tests
from the beginning. For a flow-map approximation, decide from measured one-step and
multi-step error whether a continuous representation is needed. For a vector field
or a differentiable downstream optimization loop, treat continuity/Jacobian quality
as an explicit requirement and test a representation designed to provide it.

A boundary blend is worth a controlled prototype if it has defined overlap support,
smooth inclusion/exclusion and a measured derivative bound. Do not promise continuity
for an arbitrary capped beam. Continuity-constrained fits are worth their cost when
the application requires the invariant, rather than merely because ART uses a tree.

## 29. Q-I: two algorithm plans over one executor

### 29.1 Keep two named algorithms

`algorithm="afinni"` and `algorithm="region"` are sound as validated presets.
They have different sampling and acceptance semantics. Moving top-up before fitting
in the old mode changes the training data, random draws and potentially oracle calls
even for clusters that would already have passed. It cannot remain the bit-identical
legacy algorithm just because the same functions are called in another order.

Do not expose every stage as an independent boolean and then attempt to support all
combinations. Such an interface admits incoherent settings, including validation
before final model construction or routing without complete topology. Named plans
make the scientific contract and regression expectations visible.

### 29.2 Share execution and numerical ownership

One executor should own queue admission, host/device transfer, request progress,
timing, exception handling and job ownership. Existing numerical modules should
continue to own fitting, reductions, sampling, acceptance and split calculations.

A small internal plan object or explicit mode dispatch can choose:

```text
prepare_training_data
required_oracle_quantities
fit_and_validate_candidate
prepare_rejected_nodes
propose_and_verify_split
commit_terminal_or_children
estimate_peak_bytes
```

These are illustrative responsibilities, not a request to add a separate single-use
helper for each line. Reuse and extend the existing module boundaries. The important
point is that phase behavior and state transitions remain explicit.

The legacy plan should preserve its event order, batching, reductions and RNG calls.
The region plan can use pre-fit top-up and separate audit views. Both use the same
queue, transfer and node-ownership mechanisms. This avoids both a copied second
engine and an untestable collection of scattered conditionals.

### 29.3 Keep internal experiment variants narrow

B1 is an instrumented construction/prediction experiment; B2 adds region sampling;
B3 and B4 replace split selection. These can be internal named configurations under
the region implementation without becoming arbitrary public pipeline permutations.

Persist the resolved plan and predictor with the model. A model loaded with no `Q`
should default to routing and reject an explicitly requested incompatible kNN/Q
prediction. Conversely, the public legacy defaults should continue to produce the
same result and return shape. Do not make users accidentally invoke the old default
predictor on a routed-only artifact.

### 29.4 Acceptance and terminal decisions belong in a small state machine

Use states such as ready, sampling, fitted, audit-pending, split-ready and terminal
only where needed to define ownership and recovery. A work item at a steal boundary
must carry or reconstruct all state required for its allowed next transition.
If the first implementation steals only unstarted ready nodes, say so and keep
other state local to the owning task.

This is much smaller than a public dependency scheduler. It also makes the claims
about self-contained jobs, RNG reconstruction and retirement testable.

---

<a id="30-the-minimal-plan-to-implement-now"></a>

## 30. Review of the implemented region mode, November 11, 2026

This section reviews the implementation described in design section 42, including
the revised benchmark table. It supersedes the earlier introduction's statement
that the integration has not been implemented. Sections 1-29 remain the record of
the design reviews. This assessment is based on the working tree, source inspection,
fresh CPU and MPI runs, a differential regression against the base commit, and
targeted counterexamples. No implementation changes were made during this review.

The inspected AFINNI base is still
`015adfd4a34b1b5ac5f8c67d9fa38a18a5c8bae7`. ART is at
`854924af9ec8555a11db10de6ab75f429a31be2d`. The implemented changes are in the
AFINNI working tree. Relevant SHA-256 identifiers are:

| File | SHA-256 |
|---|---|
| `AFINNI_ART_DESIGN.md` | `558041a21a38d94f790203f5211dde3cd76cf77c21cc717854119485848c7843` |
| `src/region.py` | `8b13ee6c9f1a92c55b7e09aa15d2156c345e1ee78c712284bca4a602dde4f283` |
| `src/surrogate.py` | `09615c1e767fb2496ed8e53b729ae165ee1b7470f6a44b8568486cd6f44724c7` |
| `src/engine/adaptive_clustering.py` | `cce64acf03410edeb9a205db9342efdea7e9a81cab933477bfd32c7648eefdab` |
| `src/engine/work_stealing.py` | `42b0039360c13e795c582d6805715062bfc53a74a08ddf55e95cc7327127af63` |

### 30.1 Assessment

The architecture is suitable for continuing the integration. Explicit geometry,
complete terminal coverage, node-based random streams, shared deployment, and a
fixed queue schema are all present. Retaining validation points only inside a
processing task is a sound simplification while only unstarted tasks can migrate.
The CPU results support a real reduction in execution overhead relative to ART's
node-by-node implementation.

The current implementation still needs correctness fixes before its audit result
can be treated as an accuracy claim or its Voronoi sampler as a reference sampler.
The GPU execution strategy also needs measurement and further work. Batched matrix
products are a useful foundation, but the current search repeats substantial work,
the memory cap is approximate, and some execution paths synchronize frequently.

The immediate priorities are as follows. "Reproduced" means an executable
counterexample was run against the inspected source. "Source finding" identifies
a concrete implementation mismatch or an unsupported contract, without claiming
that a benchmark failure was observed.

| Priority | Finding | Evidence | Action |
|---|---|---|---|
| P1 | `audit()` can certify zero violations when every oracle result is non-finite | Reproduced for NaN and infinity | Reject invalid evaluations or count them as failures before forming the binomial result |
| P1 | Voronoi sampling retains a numerically nonzero self-bisector | Reproduced against the identical box problem; removing the self-constraint restores identical samples | Explicitly omit each row's owning center and test chord equivalence |
| P1 for sampling diagnostics | A completely stationary corner sample passes the drift diagnostic | Reproduced in 20 dimensions, 1,000 endpoints, no movement | Detect degeneracy and movement, use interior starts, and separate pilot decisions from production sampling |
| P2 | Sparse B1 input can crash threshold search instead of producing a fallback leaf | Reproduced through the public training API | Exclude undersupported nodes before indexing candidate cut positions |
| P2 | Accepted audit thresholds differ from pointwise training thresholds | Reproduced at equality; denominator mismatch also follows from source | Define one shared pointwise predicate and use resolved tolerances |
| P2 | Claimed depth limit exceeds the router's packed-key capacity | Complete depth-61 and depth-62 trees rejected after construction | Validate actual capacity early or use compact routing indices |
| P2 for scaling | Load publication and grant selection use different work units | Source finding | Pass the same remaining-work estimator through victim selection, grants and publication |
| P2 for numerical claims | The B4 gap and rounding margin do not certify the original data objective in floating point | Source finding | Narrow the guarantee and add objective-trajectory and ill-conditioned checks |
| P2 for GPU admission | The footprint omits configurable candidate-array growth and cannot enforce an upper bound for an oversized node | Source finding | Estimate bytes by stage and task, enforce capacity, and define oversized-task handling |

These findings do not invalidate the overall integration. They identify where a
working prototype currently makes stronger claims than its implementation supports.

### 30.2 The audit must fail closed on invalid evaluations

Relevant source: [`AFINNISurrogate.audit`](src/surrogate.py#L689).

The present computation is effectively:

```python
scale = maximum(atol, rtol * abs(F))
err = max(abs(F - F_hat) / scale, axis=1)
k = sum(err > 1)
```

For NaN residuals, `err > 1` is false. Infinity can also become NaN. For example,
with an absolute-only tolerance the relative contribution is still evaluated as
`0 * abs(F)`, and `0 * inf` is NaN. With a relative tolerance, `inf / inf` has the
same problem. The subsequent Clopper-Pearson calculation receives `k = 0`.

The counterexample trained an ordinary finite constant-valued region model, then
audited it against an evaluator returning only NaN, followed by one returning only
infinity. Each audit used 299 points and returned:

```text
n_violations = 0
upper_bound = 0.009969146792899269
confidence = 0.95
max_scaled_error = NaN
```

The statistical formula is not the source of the error. The Bernoulli event supplied
to it is wrong. This matters because the global audit is the design's principal
accuracy claim after explicitly declining to certify local MCMC validation.

The implementation should check shapes and finiteness of inputs, oracle outputs
and predictions before calculating a violation rate. Two contracts are reasonable:

1. Fail the audit with an explicit invalid-evaluation result and no certificate.
2. Count every non-finite row as a violation, report its count separately, and state
   that the event being bounded includes evaluation failures.

Silently deleting invalid rows is unsuitable. It changes the sampled measure by
conditioning on success. If the oracle is undefined on part of the declared domain,
the user must either change that domain or choose an explicit failure convention.

There are further consistency issues in the same path:

- `PointwiseAcceptance` uses strict `<` comparisons. The audit counts only scaled
  errors `> 1`, so equality fails training but passes the audit. A direct equality
  counterexample returned `pointwise_accepts=False` and `n_violations=0`.
- Pointwise training uses `max(abs(F), 1e-12)` in its relative denominator. The audit
  uses `abs(F)` without that floor. They define different events near zero. With
  zero tolerance and zero error, direct division also needs an explicit convention.
- An acceptance instance or a component specification can carry its own tolerances.
  Audit defaults currently read top-level `self.config['atol']` and `['rtol']`.
  Those are not necessarily the resolved acceptance settings. The split-scaling
  path already obtains tolerances from the resolved acceptance object.
- Validate `n_points` as a positive integer and `confidence` as a finite value
  strictly between zero and one. `int(n_points)` should not silently accept a
  fractional sample count.

Use one shared row-level pointwise predicate for acceptance and audit, with an
explicit rule for disabled tolerances, floors, equality and non-finite values.
Compute the scaled diagnostic separately if necessary. It should not determine
failure by a division that can hide invalid arithmetic.

Once the event is correct, the fixed-sample one-sided binomial upper bound is
appropriate for fresh iid domain points and a frozen predictor. A model selected
after repeatedly inspecting this same audit set no longer has that simple
interpretation. Record the audit seed, model identifier, resolved tolerances and
predictor. A fresh seed by itself also does not correct model selection across many
reported audits; either reserve a final holdout or allocate the reporting error
budget across selections. The exact binomial interval implementation is documented
in [SciPy's binomial test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

### 30.3 The Voronoi self-bisector is a reproducible sampling bug

Relevant source: [`HitAndRunSampler._steps`](src/region.py#L341) and
[`_chord`](src/region.py#L393).

For a point owned by center `r`, the Voronoi constraint associated with center `j`
is

\[
(c_j-c_r)^T u\leq \tfrac12(\|c_j\|^2-\|c_r\|^2).
\]

For `j = r`, this is exactly `0 <= 0` and must not restrict the chord. The
implementation includes that column and computes its directional coefficient as

```python
Ag = g @ C.T - sum(g * c_r, axis=1, keepdims=True)
```

The matrix product and elementwise reduction need not round identically. The
self-column can therefore have a tiny nonzero coefficient. Its slack is calculated
by another subtraction of separately evaluated dot products. `_chord` skips only
coefficients equal to zero, and clamps negative slacks to zero. A self-column with
zero clamped slack and positive coefficient incorrectly sets `t_max <= 0`; a negative
coefficient can incorrectly set `t_min >= 0`. The resulting restriction is an
order-one change in the available chord, despite the coefficient being at rounding
scale. Dividing a rounding-sized slack by a rounding-sized coefficient does not
produce a rounding-sized error in the endpoint.

A particularly strong reference case requires no polytope library. A Voronoi
diagram with one center has the entire box as its only cell. At positive depth
with no path constraints, the box sampler and one-center Voronoi sampler should
perform the same walk from the same seeds with the same random numbers.

The executed check used dimension 3, one center drawn uniformly in `[0.2, 0.8]^3`
with NumPy seed 5, three copies of that center as seeds, node RNG seed 7, and
5,000 independent chain starts run for 30 steps:

| Quantity | Observed result |
|---|---:|
| Nonzero self-direction coefficients across 30 calls | 54,792 |
| Maximum endpoint difference, current Voronoi path versus box path | 0.9665903467451289 |
| Maximum endpoint difference after omitting the self-constraint in memory | 0.0 |

In 7 dimensions the current paths also differed by about 0.994 in their largest
coordinate difference. Those large paired differences alone would not establish
distributional bias, since many different valid samplers can sample the same law.
Here they accompany a direct algebraic error: an identically zero inequality
incorrectly restricts the chord. The one-center experiment isolates that error, and
the exact paired agreement after removing it confirms the cause.

Explicitly mask the owning center for each row before reducing Voronoi bounds.
Do not replace the exact-zero test by a generic fixed small-coefficient cutoff.
A genuinely small coefficient can define a real, important constraint. Mask the
known redundant constraint by identity. Duplicate centers need a separate policy,
since the lowest-index ownership rule makes later identical centers empty in the
assigned partition even though their closed geometric inequalities coincide.

The clamp in `_chord` also needs a bounded-feasibility contract. Clamping a negative
slack is reasonable only after confirming that its violation is within a
scale-aware rounding tolerance. Otherwise it walks in a relaxed region that can
include invalid starts. Check seed feasibility, finite chord bounds, interval
ordering, and output feasibility. An appreciably infeasible seed should trigger a
repair or sampling failure rather than silent constraint relaxation.

Required tests include the one-center equivalence above; the owning column in a
many-center batch; almost coincident and duplicate centers; points near a bisector;
and direct full-half-space chord endpoints in dimensions 2, 3 and 7. Run them on
both backends because the dot-product discrepancy is backend and shape dependent.

### 30.4 The drift test can accept a completely unmixed sample

Relevant source: [`sample`](src/region.py#L253),
[`_starts`](src/region.py#L286), and [`_drifting`](src/region.py#L371).

The current check compares coordinate means and squared-radius means at halfway
and final endpoints, scaled by estimated endpoint variances. That is a useful
warning for visible drift. It cannot establish that either endpoint population
represents the region.

A concrete counterexample is a box corner. In a `d`-dimensional unit box at the
origin, an isotropic direction has a nonzero feasible chord only if all coordinate
signs agree. Mixed signs force both ends of the chord to zero. The probability of
an escaping direction is

\[
P(\text{all signs agree})=2^{1-d}.
\]

In 20 dimensions this is about `1.91e-6` per attempted step. Starting at a valid
corner is therefore a poor initialization even though it satisfies every region
constraint.

The executed case used four identical corner seeds, 1,000 proposed points, 20
initial steps, a maximum of 80 steps, and node RNG seed 2. Every endpoint remained
exactly at the origin. The sampler reported `insufficient=False`. The halfway and
final means are identical, so the diagnostic never requests the longer walk.

This case is deliberately adversarial. It does not show that all ordinary
inherited samples are bad. It shows that "no detected drift" cannot serve as the
sampler's success condition, and that the implementation's current fallback cannot
reliably flag lack of exploration.

The minimum repair should include:

- A check for repeated seeds, negligible centered sample variance, negligible
  displacement and predominantly zero-length chords. Report failures explicitly.
- An interior initialization strategy. For box roots this is cheap. For a general
  child, use a checked interior convex combination or a bounded interior-point
  procedure. A mean of full-dimensional dispersed feasible seeds can help, but a
  mean of identical corner seeds cannot.
- Diagnostics in coordinates scaled to the local region, including informative
  projections and constraint slacks. Raw squared radius around the global box
  center is weak in a thin, displaced descendant.
- A bounded geometry recovery path and an `insufficient_sampling` result when it
  fails. If the region is genuinely lower dimensional, declare that condition
  rather than pretending to sample its full-dimensional uniform measure.

The design calls this a pilot/production scheme, but the code returns the same
endpoints whose aggregate drift determined the stopping length. There is no separate
production segment after freezing the pilot's choice. Adaptively choosing a common
stopping length from all endpoints also couples the returned rows. Independent
chain random streams do not make outputs independent after a joint data-dependent
stopping decision.

A practical version is to run a separate geometry pilot, freeze the transition
count and preconditioner, then run production chains on separate streams. This
removes that particular stopping dependency. It does not prove stationarity or
uniformity, and must not be described as a local probabilistic certificate.

There is a further status issue. In `process_batch`, a node whose fit and validation
pass is still appended with `terminal_reason='accepted'` even when its sampler
reports `insufficient_sampling`. The status is retained, which is useful, but
`converged` counts only fallback leaves and can remain true. Decide whether
convergence means empirical fitting success or satisfactory region exploration.
If both meanings matter, expose both totals. A convergence flag must not silently
imply that all regions passed their sampling requirement.

### 30.5 Sparse B1 tasks need an early support check

Relevant source: [`RegionPlan._threshold_search`](src/region.py#L864).

B2-B4 training normally enforces `n_target >= 2 * min_side`, but B1 deliberately
does not top up every nonempty node to that budget. Its convex augmentation can
still leave too few rows for the affine split-scoring support rule.

`_threshold_search` clips candidate positions to at least `min_side`, then indexes
`ss[starts + pos]` before excluding nodes that cannot support any candidate. If a
node contains fewer than `min_side + 1` rows, this can index beyond its block or
beyond the array. In a multi-node batch it can also read another node's scores
before validity masks are applied.

This was reproduced through public training with two 3-D input points, a constant
local model, covariance splitting, `min_pts_per_cluster=1`, B1 convex sampling,
`split_search='bins'`, and a positive absolute tolerance. The result was:

```text
IndexError: index 4 is out of bounds for axis 0 with size 2
```

It should be an explicit `split_failed` terminal leaf or a documented top-up
decision. Form an eligible-node view before candidate indexing. If masked indexing
is retained, every index must be in range even for ineligible rows, and validity
must remain false for those rows throughout screening and hardening. Include
mixed batches with one undersupported node so a neighboring node cannot mask the
problem.

The implementation also does not yet perform the rank/conditioning checks promised
in design sections 37.2 and 37.3. `n >= p_eff` is only a count condition. Duplicated,
collinear or nearly collinear points can meet it while leaving the feature matrix
deficient. Ridge makes a linear system invertible; it does not establish geometric
coverage or data-supported coefficients. Add a rank/conditioning diagnostic and a
recovery policy, or narrow those design claims. This is especially relevant to the
corner-sampling case above.

### 30.6 The declared depth limit and packed routing keys disagree

Relevant source: [`MAX_DEPTH_LIMIT`](src/region.py#L59) and
[`RoutingTable.build`](src/region.py#L1075).

The task representation uses a sentinel path code, so depth 62 fits in a positive
signed 64-bit path. The router adds another packing scheme:

```text
max_bits = bit_length(max(path_code))
shift = max_bits + 1
packed_key = (root << shift) | path_code
```

It then reserves at least one additional root bit, even for the single-root case.
A complete depth-60 skinny tree builds. Complete depth-61 and depth-62 skinny trees
were rejected with `The tree is too deep for 63-bit node keys.` Each tree used only
O(depth) nodes, so the test did not require an exponentially large tree.

This is a contract mismatch: configuration accepts a depth that assembly can
reject after expensive training. Multiple roots reduce the available path capacity
further. The packing also reserves an extra path bit beyond the observed bit length.

For the immediate repair, derive an allowed depth from the actual packing and root
count and validate it before training. The better persistent representation is to
assemble compact node indices and retain `(root, path_code)` only as provenance.
Then inference does not need to pack logical identity into a signed integer at all.

Assembly should additionally reject nonpositive paths, out-of-range roots,
non-finite planes, and invalid plane shapes. `from_arrays` currently reconstructs
the table without running the build-time topology checks. A malformed saved table
can therefore bypass the stated validated-tree contract. Validate loaded arrays
against the model's dimension and leaf count, including sorted keys and leaf-index
bounds. The public route API should also reject non-finite query coordinates; NaN
currently fails neither `U < 0` nor `U > 1` and can be routed with `outside=False`.

### 30.7 B4 has a consistent objective, but its floating-point guarantee is too strong

Relevant source: [`_soft_refine`](src/region.py#L924) and
[`_soft_objective`](src/region.py#L1008).

Several important numerical decisions are correct:

- Both sides use the same fixed floored sigmoid convention.
- The node's ridge and temperature remain fixed during refinement.
- The objective includes the same ridge penalty used in the child solves.
- The gate gradient uses frozen optimal child models through the envelope theorem.
- Final hard SSE is recomputed from residuals and B4 retains B3 as its incumbent.

The existing finite-difference check supports the local gradient formula on its
test problem. The incumbent check supports the narrower guarantee that a returned
B4 plane has no larger measured hard split SSE than B3 on that node's scoring rows.
It does not establish better final test error, fewer leaves, or monotonic exact
minimized objectives at every soft step.

The remaining problem is the claim that `gap` is the inner solve's exact
suboptimality plus a sufficient rounding margin. For exact data sums and an SPD
normal matrix,

\[
H=\Phi^T W\Phi+\lambda I,\quad C=\Phi^T WY,\quad R=C-H\widetilde B,
\]

the identity is

\[
F(\widetilde B)-F(B_*)=\operatorname{tr}(R^T H^{-1}R)/n.
\]

The implementation forms `R = Bs - (H + lamI) @ beta` from the same rounded Gram
and right-hand side used to solve for `beta`. This can be an excellent residual
for the rounded normal equations while missing error in forming those equations
from the original rows. The computed quadratic form itself is rounded and is not
a certified upper bound. The additional `64 * eps * abs(J)` margin is independent
of feature count, reduction length, conditioning and cancellation. It is a useful
heuristic floor, but does not bound all those errors.

Recompute stationarity from the original data residual when evaluating suspicious
steps:

\[
R_{\mathrm{data}}=\Phi^T W(Y-\Phi\widetilde B)-\lambda\widetilde B.
\]

Even that expression needs an error analysis to become a certificate. For version
one, a more defensible claim is "numerically monitored descent with direct residual
objectives, a residual-based margin and a hard incumbent." Add diagnostic flags for
non-finite solves, negative computed gaps, rejected numerical steps, and inner
residuals that exceed a scale-aware threshold. CuPy documents that invalid solver
inputs may require explicitly enabling linear-algebra error reporting; exercise
that behavior on the installed version rather than assuming NumPy's failure mode.
See [CuPy's solve documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.solve.html).

`soft_converged` also overstates what the current stopping condition establishes.
It is incremented when no one of 12 trial steps qualifies and the mean gate
derivative exceeds the saturation threshold. A failed line search can mean a poor
trial scale, numerical error, or a badly conditioned direction. It does not imply
that the projected gradient is small. Separate `gradient_converged`,
`line_search_stalled`, `gate_saturated`, and `iteration_limit`, at least in internal
counters. Report a gradient norm with the termination reason.

Finally, `[u, 1]` uses coordinates normalized to the whole domain. A very small
deep node can have almost constant coordinate columns and a strongly correlated
intercept. Centering and scaling split features within the node can substantially
improve conditioning. That changes the meaning of isotropic ridge, so define the
new ridge in that feature basis and hold it fixed throughout the optimization.
Gate geometry can remain in the domain coordinates with an explicit conversion.
This is a separate numerical experiment and should not silently change legacy
AFINNI fits.

### 30.8 GPU execution: a sound first implementation with substantial remaining overhead

The replacement of per-point outer products by grouped batched matrix products is
the right change. It removes the `N * p * p` temporary and gives the GPU dense
matrix operations it can execute efficiently. Grouping by exact point count also
avoids padding and supported bitwise batch independence in the CPU checks.

The current implementation is nevertheless not the binned prefix-statistics
algorithm described in parts of the design. `_threshold_search` sorts each
direction, then loops over every threshold and calls `rows.gram(left)`. Each
threshold rereads all points and forms another weighted Gram. The right Gram is
obtained by subtraction for screening, and finalists are recomputed directly.

Let `D` be the number of directions, `B = n_bins - 1`, `N` the number of scoring
rows, `K` the number of nodes, `p = d + 1`, and `q` the output count. The screening
work has the approximate form

\[
O(D B N(p^2+p q))+O(D B K p^3),
\]

in addition to direction-wise sorting and finalist recomputation. With the default
`D=6` and `B=15`, the left sufficient statistics are formed 90 times. The improved
implementation reduces memory traffic per formation but still repeats the work.

A true binned alternative assigns each row to a score bin, accumulates each bin's
`Phi^T Phi`, `Phi^T Y` and `Y^T Y` once, then scans the small per-node bin table.
Its data-statistic work is approximately

\[
O(D N(p^2+p q)),
\]

with `O(K B(p^2+p q))` bin storage. This can avoid materializing all per-point
outer products by accumulating small feature tiles or using sorted-bin matrix
products. Quantile bins still need sorting or selection; prefix subtraction still
needs the direct-residual finalist safeguard. The improvement is a candidate to
benchmark, not an automatic 15-fold speedup, because small matrix throughput,
kernel launch costs, bin imbalance and atomic contention matter.

Three GPU concerns should be measured explicitly:

1. `_SplitRows.__post_init__` copies counts to the host and groups them with NumPy.
   Every distinct count creates separate gathered blocks and matrix-product calls.
   This is inexpensive when most nodes reach the same target count, but can fragment
   large inherited or B1 batches. Record group count and rows per group. Compare
   exact groups with a few masked capacity buckets if fragmentation appears.
2. B4 keeps fixed arrays, but its solves and most reductions still process inactive
   nodes. The `active` mask controls accepted updates, not the expensive linear
   algebra. Its backtracking loop evaluates all 12 trials even if every active node
   already took a step. Fixed slots can be efficient, but an early all-taken exit,
   periodic compaction, or smaller compatible cohorts may save substantial work.
   Measure the fraction of active nodes per iteration before choosing.
3. Python conversions such as `bool(xp.any(...))`, `int(xp.sum(...))`, and host
   gathers introduce synchronization. Sampling uses many small operations per walk
   step. B4 uses repeated sigmoid, residual, reduction and trial arrays. Profile
   launch count, synchronization time and allocation traffic, alongside FLOPs.

There is no reason to implement all optimizations before the first GPU run.
First obtain a correct GPU baseline. Then measure a B2/B3/B4 phase breakdown with
representative `K`, `d`, `q`, point counts and depths. A simple dense direction-bank
search may remain preferable at small dimension, while fused bin statistics can
matter at larger dimension or larger batches.

Timing must include completed device work. Use CUDA events for kernel sequences
and a synchronized wall-clock measurement for the complete operation. Warm up the
CUDA context, kernels and solver paths before steady-state timings. The project's
phase profiler synchronizes when timing is enabled, which is useful diagnostically
but can change overlap. Record both production wall time and instrumented phase
time. These practices are described in [CuPy's performance guide](https://docs.cupy.dev/en/stable/user_guide/performance.html).

### 30.9 The current admission rule is not a VRAM capacity guarantee

Relevant source: [`RegionPlan.bytes_per_point`](src/region.py#L559),
[`work_sizes`](src/region.py#L553), and the hardware cap in
[`_adaptive_clustering`](src/engine/adaptive_clustering.py#L908).

The new estimate accounts for several important per-point terms and is much better
than reusing the old AFINNI formula. It is still a heuristic. In particular:

- Candidate Gram storage grows with `K * B * p^2`, and right-hand sides grow with
  `K * B * p * q`. `bytes_per_point()` does not depend on `n_bins`. Increasing that
  supported setting can increase peak allocation without changing admission.
- Solver workspace, gathered blocks, B4 residuals, trial buffers and overlapping
  input/output lifetimes need stage-specific accounting. A generous coefficient
  on one term does not establish a bound for every configuration.
- The 1,000-point floor can exceed the calculated safe capacity. A lower bound on
  performance batch size is not a memory-safety condition.
- Batch endpoints include a whole cluster that crosses the target point count.
  An individual oversized node is still processed whole. Consequently the point
  cap does not bound actual admitted work, even with a perfect bytes-per-point
  estimate.
- The initial full input is converted to the device and evaluated before the
  refinement cap is computed. Box mode can retain all those points in one root.
  Host-backed queues do not make this initial step out of core.

At the default settings the estimate may be conservative in some stages. No GPU
out-of-memory result was observed here because CuPy is unavailable. The issue is
the absence of an enforced contract, not a claim that every default run will fail.

Use an estimator of the peak over stage lifetimes:

\[
M_{\mathrm{task}}=\max_s M_s(K,N,p,q,h,R,B,\text{model},\text{sampler}),
\]

including already resident buffers and a measured solver/runtime reserve. Admit
whole tasks only while their estimated peak fits. An oversized task needs a defined
response: process its sufficient statistics in chunks, split proposal work into
bounded chunks, reduce a scientifically optional setting with explicit reporting,
spill supported state, or fail with the estimated requirement. Arbitrarily
geometrically splitting the task is an algorithm change and should not be hidden
inside a memory workaround.

The present system is host-backed breadth-first training with bounded batches for
many workloads. It does not yet establish disk-backed training, streamed initial
evaluation, bounded memory for every single node, or a fully device-resident
descendant pipeline. State those limits when using "out of core" to describe it.
The host memory guard should eventually include cluster-scoped constraint storage
and grow-and-copy lifetimes as well as point columns.

### 30.10 MPI work accounting needs one shared unit

Relevant source: [`_run_local_refinement`](src/engine/adaptive_clustering.py#L194) and
[`WorkStealer.serve_requests`](src/engine/work_stealing.py#L193).

The executor estimates region load with

```python
sum(maximum(queue.sizes, n_target) + n_val)
```

for region sampling. That is a sensible first proxy for imminent work, especially
for empty roots or children that will be topped up. The victim still computes its
eligibility and grant size from `queue.sizes`, then publishes `queue.n_pts` after
a grant. Thus the load seen by thieves and the work used by victims are in different
units. A single victim can also switch its published unit after donating work.

For example, a grid of empty roots can have positive preparation work and zero
queued points. It is advertised as work by the executor but denied for stealing by
the victim's point-count test. More commonly, children carry about half their
parent's training rows and will top up before fitting, so inherited row count
systematically understates their work. Thresholds described as a few nodes' worth
can require substantially more nodes than that to pass the victim test.

Pass one region workload function into all of load publication, open-tail
eligibility and grant selection. Keep the actual point and constraint bytes as a
separate transfer estimate. Published remaining work should exclude already
processed nodes and identify the in-flight batch separately; current post-grant
publication includes the full retained queue's points. Keeping exact job ownership
separate from estimated work remains necessary for retirement.

Receiver leases are a reasonable initial anti-bounce policy. The four-rank test
passed with forced transfers, including B4 progress. That demonstrates protocol
functionality on the tested workloads. It does not establish useful scaling under
the defaults:

- The default point batch can admit an entire small or medium frontier. Once all
  those nodes are in flight, progress calls have no unstarted tail to donate.
- B4's progress is called after threshold search and periodically during soft
  refinement. Long sampling and hard-threshold work still have no internal progress
  points. A bounded iteration count is not a bound on service latency when each
  iteration becomes much larger at high dimension.
- The idle retry interval is 50 ms. Many low-dimensional batches are shorter than
  that. An idle rank can miss several opportunities before requesting useful work.
- A single box root initially exposes one task. Neither GPU batching across nodes
  nor MPI stealing creates early node-level concurrency before valid splits exist.

Before calibrated time prediction, use the shared workload proxy consistently,
measure request-to-answer latency, and limit cohort size when peers are idle. Add
progress between bounded sampler/search chunks. Grid roots can improve startup
parallelism, but are a different approximation partition and must be reported as
such. A shallow hierarchical startup remains a useful experiment when flat grids
or Voronoi roots are unattractive.

For GPU MPI, confirm explicit rank-to-device assignment and benchmark one rank per
GPU first. All ranks choosing the same default device can cause contention and
memory exhaustion despite passing CPU MPI tests. The current payload path stages
host arrays; pinned buffers, asynchronous transfer and overlap are future
performance experiments. They should follow a measured transfer bottleneck.

### 30.11 Inference is correctly routed, but compact node indices are the next useful step

ART's existing `RegressionTree.predict` in `art/tree.py` dispatches batches of
queries down stored split predicates and evaluates the selected leaf. It does not
need a nearest-neighbor reconstruction of the tree. The integrated region mode
now follows the same mathematical rule: assign a root, follow each stored plane,
then evaluate exactly one leaf. That is the correct reference predictor for this
training geometry. The old kNN/Q predictor remains an independently useful
comparison on an artifact that stores Q.

The router's current cost includes more than `O(d * depth)` arithmetic. At each
level, it performs `searchsorted` over the global split-key array, checks activity
on the host, gathers active rows, and reconstructs the next logical key. The
per-query lookup work is roughly

\[
O\!\left(h\log S+h d\right)
\]

for `S` split nodes and depth `h`, plus root assignment and leaf lookup. Voronoi
root assignment is an all-center search, `O(Rd)` per query, with tiled query storage.
For many roots that can dominate traversal even though no leaf kNN is used.

During assembly, logical keys can be mapped once to compact node indices. Store
`left_child`, `right_child`, split planes and a leaf-model index in contiguous
arrays, with a small root-index map. Traversal then follows child indices directly.
This also removes the packed-key depth restriction and most repeated searches.
It is a good next inference optimization because it changes representation while
preserving the exact predicates.

Benchmark a GPU one-thread-per-query traversal against level-wise active-query
processing. The former avoids Python round trips but can suffer warp divergence
and scattered plane loads. The latter can group work by node, at the cost of
sorting or compaction. Which wins depends on tree balance, dimension, batch size
and query coherence. A compact CPU router can also be competitive for small query
batches where a GPU launch and transfer dominate.

The API currently returns `outside_domain`, while validation and terminal status
are separate per-leaf arrays. This is internally consistent but is weaker than the
structured per-query status described in the plan. An in-domain prediction routed
to a `max_depth` or `split_failed` leaf has `outside=False`. Expose an optional
query result containing the leaf index, terminal reason and validation status, or
provide a documented combined diagnostic method so applications can distinguish
coverage from empirical accuracy.

Outside points are clipped for routing and evaluated at their original coordinates.
That is extrapolation by the boundary-selected leaf. It is a defensible policy when
explicitly documented, but the domain audit says nothing about its accuracy. The
existing test correctly checks this behavior. Add NaN/query-shape handling and
tests near root and child boundaries.

For dynamical systems the conclusions of section 28 still apply. Hard leaf fits can
jump at boundaries. Better routing speed does not remove that numerical issue.
Boundary jump, Jacobian, and multi-step trajectory checks should precede a claim
that routed inference is a drop-in replacement in an ODE or flow-map application.
No continuity mechanism needs to be added before an application demonstrates the
requirement.

### 30.12 What the benchmark table establishes

The revised table is more useful because it reports both error and fraction above
tolerance, includes ART's faster CPU sampler, and acknowledges the 3-D wave's
stream sensitivity. It supports the conclusion that batching can remove a large
amount of ART's CPU execution overhead on these problems. It also shows that the
advantage shrinks as dimension makes per-node arithmetic dominate.

It does not establish that B4 is generally the best algorithm, that the tolerance
has been reached, or that GPU/MPI performance will follow the same ratios.
Several observations matter:

- At 20-D, about 97% of test points exceed the requested tolerance and every method
  fills the depth-limited tree. This is a cost comparison at an unsuccessful
  accuracy target. It is useful, but cannot support an accuracy-success claim.
- On 3-D plane wave, B4's reported relative L2 varies from 0.023 through 0.084 to
  0.20 across different streams. The newest B4 result has fewer pointwise failures
  than ART but worse L2. Report both because a smaller failure fraction can coexist
  with a much worse tail.
- B3 and B2 are frequently faster, and sometimes more accurate. B4's per-node hard
  incumbent protects its split-scoring rows, not the entire adaptively sampled
  descendant tree. Different split geometry changes future data and optimization
  decisions. There is no theorem that final test error must improve.
- Equal depth limits or similar oracle counts do not guarantee equal achieved
  accuracy. The useful product comparison is time and memory to reach a specified
  independent accuracy target, with unsuccessful runs retained in the results.

Keep B3 as the reference configuration for performance work and B4 as an explicitly
measured alternative. Making B4 the public default can be a product choice, but the
current table does not yet justify a universal superiority claim. Consider a
bounded B4 policy that runs only where split uncertainty or a measured hard-SSE
improvement warrants its cost. Its thresholds need an experiment, not a new claim
of optimality.

The comparison script also needs guardrails. It accepts ART options such as
`--error-metric`, `--relative-error-floor`, leaf degree and bias, but constructs
the region model with affine leaves and `error_metric='pointwise'`, with its
hardcoded relative floor. Those defaults can be aligned, but arbitrary accepted
ART options need not describe the same region experiment. Either reject
unsupported combinations or print the full resolved configuration for both
methods and label the mismatch. The GPU region oracle adapter also converts every
target call to NumPy on the host and copies the result back. That is reasonable
for comparing identical targets, but it is a transfer-inclusive oracle path and
must be reported when interpreting GPU timings.

A robust comparison should retain per-seed results, with common independent test
points across methods and an additional reserved final audit. Use multiple
training seeds, paired method comparisons where possible, medians and dispersion
for time, and failure-rate/tail distributions for accuracy. Record source hashes,
BLAS thread counts, ART/region settings, warm-up policy, CPU/GPU model, maximum
resident memory, and complete terminal-status counts. The sampler stream change
should be evaluated against exact references, not only against an earlier sampler
that could share a bias. Similar means and a nonsignificant difference do not
establish equivalence of samplers.

### 30.13 Agreement with the plan and justified simplifications

The following implementation choices should be retained:

- The fixed `QUEUE_COLUMNS` table centralizes transport and queue slicing without
  creating a public property system.
- Validation arrays stay inside `process_batch`; only ready nodes migrate. This
  makes point-role queue columns unnecessary for the current lifecycle.
- The shared `deploy_leaves` preserves the legacy ordering and keeps finalization
  checks in one place.
- Complete terminal leaves make routing defined throughout the declared domain.
- The B3 incumbent and direct hard residual scoring are appropriate protection
  against a worse hardened soft split and cancellation in screening statistics.
- Distinct train, validation and direction RNG purposes make task placement
  independent of random-stream selection for B2-B4. This is a much better contract
  for MPI than rank-dependent seeding.

Other plan statements need to be changed or implemented:

| Plan statement | Current implementation | Required clarification |
|---|---|---|
| Pilot followed by separate production sampling | Endpoint comparison controls continuation of the same returned chains | Describe the weaker diagnostic or add separate production streams |
| Rank/conditioning checks | Count checks and ridge solves | Add diagnostics and failure/recovery policy |
| Binned prefix statistics | Repeated weighted matrix products for each cut | Correct the description; benchmark true bin accumulation separately |
| Exact suboptimality plus a finite-precision guarantee | Rounded normal-equation residual and heuristic margin | Use a monitored-descent claim unless a valid error bound is supplied |
| Structured routed statuses | Prediction/outside pair plus separate leaf metadata | Expose or document an efficient per-query status path |
| Region load based on expected preparation work | Executor uses that estimate; grants revert to inherited points | Share the estimator across the full protocol |
| Depth up to 62 | Path storage permits it; packed router can reject earlier | Validate capacity before work or use compact child indices |
| All phases implemented | Several listed resource contracts remain deferred or approximate | Separate implemented mechanisms, measured guarantees and deferred work |

The queue schema's wire format is versioned, which is appropriate. The array tags
for legacy mode remain the old offsets, but the header format has changed. Say
"same legacy array payloads and tags" rather than implying wire compatibility with
an old executable. Mixed versions must fail clearly, as the version check intends.

The legacy oversized-batch repair is justified. A run silently losing clusters
while reporting convergence is a correctness bug. It should be documented as an
intentional behavior correction with its own regression case, rather than claiming
unqualified bit identity with the old bug. The new regression result in section 31
supports exactly that distinction.

Relaxing the old serial-versus-MPI numerical equality test is also reasonable given
the preserved rank-dependent legacy RNG. The replacement's 15% leaf and 25% error
bands are smoke-test thresholds, not a proof of conservation or numerical
equivalence. Keep separate protocol tests for work conservation and message drain,
and compare each MPI configuration against a frozen reference for that configuration
when investigating legacy changes. Region mode's exact CPU rank/steal test remains
the stronger new contract.

GPU decision-count divergence deserves explicit treatment. A rounding-sized
perturbation can change a near-tied split and eventually produce a structurally
different tree. A tree is not "equal to rounding" merely because one or two leaf
counts differ. If that occurs, record the first divergent decision and its score
margin, determine whether it is consistent with numerical error, and choose a
documented reproducibility policy. Do not simply loosen test tolerances until
different decisions pass. Prediction tolerance and tree-decision equality are
separate contracts.

<a id="31-second-round-verification-and-remaining-uncertainty"></a>

## 31. Implementation verification and next experiments

### 31.1 Checks executed for this review

Every Python run used explicit backend and MPI environment settings. The CPU
suite, four-rank MPI suite, baseline regression, and focused probes were executed
against the source identified in section 30. No GPU training or GPU timing result
is claimed.

| Check | Result | Interpretation |
|---|---|---|
| CPU unit/integration suite, `-m 'not mpi'` | 476 passed, 35 skipped, 12 deselected; one backend subprocess failure | The failure was MPICH initialization under the sandbox, not a failed numerical assertion |
| That backend test outside the sandbox | 1 passed | Confirms the failing subprocess can initialize MPI in the unrestricted local environment |
| Four-rank CPU MPI suite | 12 passed in 33.20 s | Includes region rank/steal invariance, forced transfers, and failure/retirement tests |
| Regression on base `015adfd` | 45 cases ran, zero run errors | Fresh baseline extracted from the actual base commit |
| Regression on the implementation | 45 cases ran, zero run errors | Compared with `rtol=0` |
| Differential comparison | All differences confined to `extra_tiny_batches` | The other 44 cases are bit-identical on the tested CPU backend |
| Invalid global audit | NaN and infinity both produced zero violations and a 0.009969 upper bound | Reproduces section 30.2 |
| Equality threshold | Pointwise training rejected; audit counted no violations | Reproduces the strictness mismatch |
| One-center Voronoi versus identical box walk | Large paired endpoint differences; omitting self-column gave exact equality | Reproduces section 30.3 |
| Corner-seeded 20-D walk | 0 of 1,000 endpoints moved; insufficient flag false | Reproduces section 30.4 |
| Sparse B1 public training | `IndexError` in cut-position indexing | Reproduces section 30.5 |
| Complete skinny-tree assembly | Depth 60 built; 61 and 62 rejected | Reproduces section 30.6 without exponential storage |
| Explicit GPU backend import | Failed because CuPy is not installed | GPU correctness, peak memory and performance remain unverified here |

The regression exception changed leaf count from 1 to 7, evaluation count from
513 to 522, leaf arrays and predictions. That is consistent with repairing the
documented loss of work. It is a deliberate correction, not a rounding difference.

The initial CPU suite's subprocess failure was:

```text
MPIDI_UCX_init_worker: ucx function returned with failed status
```

Rerunning `tests/integration/test_backend.py::test_force_cpu` outside the sandbox
passed. The MPI suite was also run outside that restriction. The full CPU suite
was not rerun outside the sandbox, so the evidence is the suite result plus the
isolated successful rerun, rather than a claim of one completely clean full-suite
invocation.

Commands used for the main verification were:

```bash
AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m pytest \
  -q --no-cov -m 'not mpi'

AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m pytest \
  tests/integration/test_backend.py::test_force_cpu -q --no-cov

AFINNI_DEVICE=cpu AFINNI_MPI=on AFINNI_MPI_NP=4 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest tests/mpi -o addopts='' -q
```

A temporary extraction of `git archive HEAD` provided the baseline checkout,
without changing the repository's index or implementation. Both checkouts ran
`scripts/regression_check.py --save ...` with `AFINNI_DEVICE=cpu`,
`AFINNI_MPI=off` and `PYTHONPATH=.`. Their saved results were compared using the
script's `compare` function with zero tolerance. Review-time logs and focused
probes are in `/tmp/afinni_review_*` and
`/tmp/afinni_implementation_review_checks*`; these are temporary evidence, not
required project files.

### 31.2 Compact reproductions for the highest-priority findings

The following examples use only the current public API or the sampler's own
objects. Run from the repository root with `AFINNI_DEVICE=cpu AFINNI_MPI=off` and
`PYTHONPATH=.`. The test helper is used only to keep model construction short.

Invalid audit:

```python
import numpy as np
from src.config import RegionSettings
from src.core.evaluation import Evaluator
from tests._helpers import train_surrogate

X = np.random.default_rng(4).random((30, 2))
m = train_surrogate(
    X, f_fn=lambda x: np.ones((len(x), 1)), atol=1e-3,
    algorithm="region", domain=([0, 0], [1, 1]),
    local_model="affine_lsq", split_strategy="covariance",
    use_work_stealing=False, max_levels=2,
    region=RegionSettings(sample_multiplier=5, store_Q=False),
)
report = m.audit(
    Evaluator(f_fn=lambda x: np.full((len(x), 1), np.nan)),
    299, atol=1e-3,
)
print(report)
# Current result: n_violations=0 and upper_bound approximately 0.009969.
```

Unmoving chains that pass the drift check:

```python
import numpy as np
from src.region import (
    Domain, Roots, RegionPoints, HitAndRunSampler, node_keys, PURPOSE_TRAIN,
)
from src.components.sampling import ClusterPoints

d = 20
geo = RegionPoints(
    np.array([0]), np.array([1]),
    node_keys(2, np.array([0]), np.array([2])),
    np.zeros((1, 0, d + 1)), PURPOSE_TRAIN,
)
sampler = HitAndRunSampler(
    Domain(np.zeros(d), np.ones(d)), Roots.build("box", d, 1),
    walk_steps=20, max_walk_steps=80,
)
Y = sampler.sample(
    ClusterPoints(np.zeros((4, d)), np.array([4]), region=geo),
    np.array([1000]),
)
print(np.count_nonzero(np.any(Y != 0, axis=1)), geo.insufficient)
# Current result: 0, [False].
```

For the Voronoi check, use the same positive-depth geometry and RNG for a box walk
and a one-center Voronoi walk, both with no effective path restrictions. Set
`d=3`, `C=np.random.default_rng(5).uniform(.2, .8, (1, d))`, three copies of `C`
as inherited seeds, node seed 7, 5,000 requested points, and both step limits to
30. Their endpoints currently differ by up to 0.9665903467451289. Instrumenting
the Voronoi self-column shows 54,792 nonzero directional coefficients over the 30
steps. Temporarily ignoring only that known redundant column restores bitwise
agreement. A permanent test should compare the chord endpoints directly as well
as the final walks, so the error is localized before a long chain amplifies it.

### 31.3 Correctness tests to add before interpreting new performance results

The existing tests cover many ordinary cases. The next additions should target
contracts that currently lack adversarial coverage:

| Area | Test | Required outcome |
|---|---|---|
| Global audit | NaN/infinity in oracle, prediction or one component | Explicit invalid result or counted violation; no false zero-failure certificate |
| Pointwise event | Equality, zero outputs, zero/disabled tolerances, vector tolerances and custom acceptance settings | Training and audit use the documented same event when that is the selected audit contract |
| Audit arguments | Zero, negative and fractional counts; invalid confidence | Early descriptive error |
| Voronoi chords | One center, duplicate centers, nearly coincident centers, owning self-column | Redundant constraints do not change the chord; degeneracy policy is explicit |
| Polytope feasibility | Tiny and large negative slacks, zero directions, thin cells | Rounding repair stays bounded; significant failure is reported |
| Sampler exploration | Corners, repeated seeds, thin rotated boxes and deep slivers | Degenerate exploration is flagged or recovered; feasible output alone is insufficient |
| Sampler distribution | Exact rejection references in low dimension, multiple projections and streams | Quantified distributional discrepancy, not only coordinate means |
| Split support | Sparse B1 nodes alone and beside valid nodes; rank-deficient features | Safe fallback or documented resampling; no invalid index or cross-node read |
| B4 | Per-step minimized objective, projected gradient, solve residual and termination reason | Observed descent within a justified numerical tolerance; stalls are distinguished from convergence |
| B4 hard incumbent | Ties, saturated gates, invalid proposals and ill-conditioned child fits | A valid B3 incumbent is retained unless a valid direct hard score improves it |
| Topology/persistence | Deep paths, many roots, malformed planes, invalid saved keys and leaf indices | Early validation; no late ambiguous routing failure |
| Task placement | Several batch sizes and rank counts, with different node sizes and validation enabled | Logical RNG identity preserved; first divergent numerical decision diagnosed |
| Work accounting | Empty roots and heavily topped-up children, normal thresholds | Advertised work can be donated and the same units are retained after grants |
| Capacity | Large `n_bins`, high dimension/output count, and one oversized root | Admission respects the documented bound or fails with an explicit requirement |

For sampler tests, a single mean near the analytical value can miss boundary and
tail bias. Use analytic marginal CDFs where available, exact rejection sampling
for low-dimensional reference polytopes, and projections aligned with narrow
directions. Specify the tested sample sizes and diagnostic error thresholds before
comparing implementations. Large sample counts are appropriate for an offline
validation script; keep routine unit tests small and deterministic.

For B4, trace both the frozen trial objective and the refitted objective after an
accepted step. Check the data-residual stationarity estimate separately from the
formed normal-equation residual. Test large offsets, narrow regions, repeated
features, different output scales, and very small gate temperatures. Passing a
finite-difference check at one well-conditioned gate does not exercise these
failure modes.

### 31.4 A focused WSL/GPU experiment sequence

First fix the audit and Voronoi defects and add their regression tests. A fast
sampler that restricts the wrong chord cannot provide a valid performance baseline.
Then perform the following sequence on the actual CUDA environment:

1. Run the explicit GPU unit/integration suite and the GPU differential regression
   against `015adfd`, with the oversized-batch exception recorded separately. Check
   the installed NumPy/CuPy solver shapes, error-reporting behavior, uint64 hashing,
   reductions, route predicates and save/load round trip.
2. Compare CPU and GPU on frozen inputs for individual stages. Include sampler
   feasibility, exact root ownership, direct hard candidate scores, B4 gradients,
   residual norms and routed leaf indices. When a tree differs, locate the first
   differing predicate or candidate decision instead of comparing only final L2.
3. Measure B2, B3 and B4 kernels at dimensions 2, 5, 20 and a larger dimension
   appropriate to available memory. Use scalar and several-component outputs,
   shallow and deep constraints, a common point budget, and both large compatible
   batches and small frontiers. Record time, peak live/pool VRAM, launches, solver
   calls and active-node fractions.
4. Measure the current repeated-Gram B3 against a genuine binned-statistics
   prototype at equal direction banks, candidate positions and finalist rules.
   Keep the winning predicate fixed where scores agree. Report numerical decision
   differences separately from speed.
5. Measure sampling separately. Include exact box/grid roots, the corrected
   Voronoi path, isotropic descendants and thin rotated descendants. Record mixing
   diagnostics, constraint violations, step counts and total geometry work. Compare
   preconditioning only after the sampler reference is correct.
6. Measure complete training with a cheap GPU-native oracle and a representative
   expensive oracle. The host-only ART comparison adapter is another workload;
   record its transfers separately. Compare time to an independent accuracy target,
   oracle calls, leaves, failure fraction, tail error and fallback volume or sample
   share. Preserve failed target attempts in the report.
7. Measure inference on the same frozen model. Compare kNN/Q, current routing, and
   compact-index routing at query batches spanning latency and throughput use
   cases. Include Voronoi root cost, transfer-inclusive latency, model memory and
   optional per-query statuses. Boundary and trajectory checks belong alongside
   inference accuracy where the target application needs them.
8. Run MPI scaling with explicit device ownership. For multiple GPUs use one rank
   per GPU first. Report startup, sampling, fitting, search, transfer, idle time and
   request latency. Compare stealing off/on under both forced and production
   thresholds. CPU MPI success is protocol evidence; GPU MPI efficiency needs this
   separate measurement.

The first useful optimization result should answer a narrow question, such as
whether bin accumulation reduces B3 time at fixed candidate decisions, or whether
compact child indices reduce route latency on a frozen tree. That makes the cause
of an improvement inspectable. A complete algorithm comparison can then combine
the best validated components.

### 31.5 Recommended implementation order after this review

1. Repair the audit event, argument validation and resolved-tolerance handling.
   Add invalid-output and equality tests.
2. Remove the Voronoi self-bisector by identity, enforce bounded feasibility repair,
   and add direct chord reference tests.
3. Handle sparse split tasks safely; add sampler degeneracy and rank diagnostics;
   clarify acceptance versus insufficient-sampling convergence.
4. Fix early depth validation and loaded-routing validation. Assemble compact child
   indices if inference is an immediate target.
5. Unify remaining-work estimates and grant decisions. Measure service latency
   before adding calibrated time prediction.
6. Establish the first correct GPU baseline and stage memory measurements. Replace
   the point-cap heuristic with an enforced task-capacity contract for workloads
   that need it.
7. Compare B3 and B4 at matched independent accuracy, then optimize the measured
   bottleneck. True bin statistics, better feature conditioning, sampler
   preconditioning and compact routing are separate experiments with clear
   expected benefits.

The next decision does not require another general abstraction layer. The current
two-mode architecture is sufficient for these fixes and experiments. The strongest
next evidence would be a corrected sampler/audit pair, adversarial numerical
coverage, and measured GPU/MPI scaling with the same explicit scientific contracts.

---

## 32. Follow-up review of the fixes, November 11, 2026

This section reviews commit `9c17c1133f69743550272c63cdcc321cb61cdde4`
and the accompanying design update at
`5a47695c595a6a7dff74d388c6468b95b4bca552`. The working tree was clean when
inspection began. The comparison reference is
`48f59c3`, immediately before these fixes, which includes the earlier
oversized-batch correction. Earlier review sections remain historical; findings
listed below as resolved should no longer be treated as open bugs.

Inspected file hashes:

| File | SHA-256 |
|---|---|
| `AFINNI_ART_DESIGN.md` | `e14e9b3ee2ec6e1930f49190e1379b606270556f549bf4bd5bc6380edb523fd3` |
| `src/region.py` | `bb6454e7e7abe426225f8dd679ab8090fe395990360b9d17b3eb4b3d23005c04` |
| `src/surrogate.py` | `b5efb752bec1e970293d8142ba6dce101be2a8501ae812b609564e25eda61a18` |

### 32.1 Assessment and disposition of the previous findings

The fixes improve both correctness and the accuracy of the design's claims. The
shared pointwise predicate repairs the false audit result. Compact child indices
remove the packed-key depth limit and repeated routing searches. The work stealer
now uses the plan's workload units for donation and publication. The numerical
descent claim has been narrowed appropriately.

Fresh verification passed 499 CPU unit/integration tests and all 12 four-rank MPI
tests. All 45 legacy regression cases were bit-identical to a fresh reference from
`48f59c3`. The previous targeted counterexamples were also rerun.

The next changes should remain focused numerical and contract fixes. There is no
new evidence here requiring a third algorithm abstraction, a property registry,
or a replacement MPI protocol.

| Previous finding | Current result | Disposition |
|---|---|---|
| Invalid oracle values pass the global audit | NaN and infinity each produce 299 violations, 299 invalid rows and upper bound 1 on 299 points | Resolved for the reproduced failure |
| Training/audit equality mismatch | Both reject the reproduced equality case | Resolved through the shared predicate |
| Audit defaults ignore resolved acceptance tolerances | Defaults now read the acceptance component's settings | Resolved in the inspected path |
| Voronoi owning-center constraint changes the chord | One-center and box walks are bit-identical in the repeated 3-D and 7-D checks | Self-constraint defect resolved; a separate near-center instability remains below |
| Entire 20-D corner sample remains stationary without a warning | All 1,000 endpoints now move and the short-budget sample is flagged insufficient | Original counterexample resolved; partial immobility remains below |
| Sparse B1 scoring indexes beyond the node | The reproduced two-row scoring case returns without an exception | Resolved for supported queued tasks in that case |
| Depth-61/62 trees fail assembly | Both now build; an additional repository test covers depth 62 with 1,024 roots | Packed-key restriction resolved |
| Loaded topology and query coordinates bypass checks | Loading rebuilds topology; non-finite/misshaped queries are rejected | Substantially improved; domain/root geometry validation remains incomplete |
| Victim grants use inherited rows while advertised load uses top-up work | Work estimator is shared by grant selection and publication | Unit mismatch resolved; freshness/donatability of load is a separate performance issue |
| B4 residual margin is described as an exact numerical certificate | Residual comes from data, diagnostics distinguish termination reasons, and the claim is monitored descent | Appropriate correction; stable score variance remains necessary |
| Candidate memory does not depend on `n_bins` | Estimate now includes candidate-array growth | Improved heuristic; capacity remains unenforced |
| Benchmark accepts unsupported model/metric combinations | Degree and metric guards added | Partly resolved; bias and relative-floor options still differ |

### 32.2 P1: near-coincident Voronoi centers still produce the wrong region

Source: [`HitAndRunSampler._steps`](src/region.py#L383).

Masking the owning-center column fixes the redundant self-plane. The non-owning
bisectors still use differences of separately computed dot products and squared
norms:

\[
a_j^T g=g^Tc_j-g^Tc_r,
\qquad
\operatorname{slack}_j=
\tfrac12(\|c_j\|^2-\|c_r\|^2)-(u^Tc_j-u^Tc_r).
\]

When centers are close, both formulas subtract large nearly equal quantities. In
a thin cell, the actual slack can be much smaller than the rounding error in
those quantities. The error then enters a division by the already small
directional coefficient and becomes a large error in the geometric chord.

This is a different failure from the self-plane bug. Removing exact duplicate
centers does not address distinct centers with small separation.

An executed one-dimensional example uses the unit interval and three centers:

```python
base, separation = 0.695, 1e-12
C = np.array([[base - separation], [base], [base + separation]])
```

The middle center's true cell is approximately

```text
[0.6949999999995, 0.6950000000004999]
```

Starting at the middle center, with node seed 2, 8 initial steps, 32 maximum
steps, and 200 requested endpoints, the current sampler returned:

| Measurement | Result |
|---|---:|
| Smallest sampled coordinate | 0.6949645005327146 |
| Largest sampled coordinate | 0.6950408265998055 |
| Endpoints whose nearest center is not the assigned middle center | 200 of 200 |
| `insufficient_sampling` | False |

The sampled interval is tens of millions of times wider than the true cell. This
is not an inference from a mean test: every returned endpoint was checked with
the implementation's own direct-distance `root_of` assignment.

The initial chord already exposes the error. For direction `g=+1` at `u=base`,
the expanded formulas give approximately

```text
[-2.77561896e-5, 0]
```

for the displacement interval. A stable expression gives approximately

```text
[-4.99988939e-13, 4.99988939e-13]
```

The feasibility guard does not catch the failure. A residual of an unnormalized
bisector has units proportional to the center separation. The current tolerance
`1e-9 * (1 + abs(offset))` has a unit-sized floor, even when the normal has norm
about `1e-12`. A geometrically large violation can therefore have a residual
smaller than this threshold. "Within rounding" is not an adequate description of
that allowance.

Use the center difference explicitly. Let

\[
\Delta_j=c_j-c_r,\qquad v=u-c_r.
\]

Then the same bisector is

\[
\Delta_j^Tv\leq\tfrac12\|\Delta_j\|^2,
\]

and its directional coefficient and slack can be evaluated as

\[
\Delta_j^Tg,
\qquad
\tfrac12\|\Delta_j\|^2-\Delta_j^T(u-c_r).
\]

For nonzero separation, a useful geometric form divides by `||Delta_j||`:

\[
n_j^T(u-c_r)\leq\tfrac12\|\Delta_j\|,
\qquad n_j=\Delta_j/\|\Delta_j\|.
\]

The residual then has distance units, and a feasibility tolerance can be expressed
in the same units as the region coordinates. The algebraic identity alone does
not certify floating-point accuracy, but it avoids the specific large-term
subtractions responsible for this example.

Form and evaluate those quantities in root/center tiles. A root group can share
its center-difference matrix, avoiding an all-points by all-centers by dimension
temporary. No `R^2 d` permanent table is required. Preserve the explicit self-mask
and duplicate-center policy.

Required protections are:

1. Stable coefficient/slack evaluation for non-owning centers.
2. A scale-aware feasibility rule invariant under rescaling a half-space row.
   Multiplying a valid inequality by a positive scalar must not change whether
   the same point is considered appreciably infeasible.
3. A post-step or returned-endpoint check against the declared geometry, including
   the final step. At present feasibility is checked before a step; the last
   generated endpoint has no subsequent check in that segment.
4. Reference tests at distinct center separations such as `1e-6`, `1e-8`, `1e-10`
   and `1e-12`, with comparisons to stable 1-D intervals and low-dimensional
   direct half-space references.

The direct nearest-center check is useful as an independent ownership diagnostic.
The endpoint chord-certification idea in section 23 is also applicable: compare
the proposed endpoints against direct distances, handle ties explicitly, and
recompute uncertain chords with stable expressions. Merely classifying a sampled
point after the fact does not repair a biased chord, but it prevents an invalid
sample from silently entering the wrong node.

Do not repair this by deleting every pair of centers closer than a fixed
threshold without recording the change. Merging distinct roots changes the
partition. If a minimum root separation is an intentional supported-domain
restriction, validate and report it explicitly.

### 32.3 P2: cancellation corrupts B4 temperature and sampler drift statistics

Sources: [`RegionPlan._soft_refine`](src/region.py#L1019) and
[`HitAndRunSampler._drifting`](src/region.py#L437).

Both paths calculate variance by subtracting raw moments:

\[
\widehat{\operatorname{Var}}(s)
=\operatorname{mean}(s^2)-\operatorname{mean}(s)^2.
\]

That loses relative accuracy when the spread is small compared with the mean.
The problem appears around spreads of order `sqrt(eps) * abs(mean)`, well before
the points themselves become indistinguishable in float64. Deep regions in global
domain coordinates are a natural way to reach that regime.

The executed B4 case used 200 scores

```python
s = 0.9 + 1e-8 * np.linspace(-1, 1, 200)
```

and the default temperature multiplier `0.1`. The values are distinct and have a
well-resolved centered variance. Using the exact reduction path in `_SplitRows`
gave:

| Quantity | Result |
|---|---:|
| Raw-moment variance | -4.440892098500626e-16 |
| Reference temperature from centered variance | 5.802442736356279e-10 |
| Actual temperature passed into `_soft_objective` | 1.0000000000000001e-151 |
| One-iteration termination reason | `soft_gate_saturated` |

The `max(var, 1e-300)` clamp changes a negative rounding result into an effectively
zero temperature. It therefore converts a smooth, nonconstant gate problem into
an almost hard gate and then diagnoses the resulting saturation. The B3 incumbent
still protects the final measured hard score, so this example demonstrates a
disabled or distorted refinement rather than a proved degradation of the final
tree's test error.

Changing sample count or operation ordering can make the same raw-moment formula
return zero or a positive rounding-sized value instead. A test that checks only
for negative variance will miss the inflated-positive case.

The sampler has the same issue in each coordinate and in its squared-radius
feature. For 20 points around `0.9` with spread `1e-8`, shifting all halfway
endpoints by `1e-12` produced `drift=True`. The coordinate mean shift standardized
with centered endpoint variances is only about `0.000521`, far below the configured
threshold 4. A 200-point case also flagged a shift with standardized size about
`0.00172`. This can trigger unnecessary chain extensions or false insufficiency
statuses in otherwise narrow regions.

Use centered, segmented second moments:

\[
\mu_k=\frac{1}{n_k}\sum_{i\in k}s_i,
\qquad
v_k=\frac{1}{n_k}\sum_{i\in k}(s_i-\mu_k)^2.
\]

For tiled or streamed statistics, combine count, mean and centered second moment
with a stable merge formula. Reuse the existing segmented-reduction ownership;
there is no need for a separate numerical module for these two callers. NumPy's
documented variance calculation also uses deviations from the mean:
[NumPy variance documentation](https://numpy.org/doc/stable/reference/generated/numpy.var.html).

After calculating a stable variance, choose a temperature floor from an explicit
score-resolution or local-width policy. If the scores are numerically constant,
report that condition and retain the hard incumbent. A floor near `1e-151` has no
useful relationship to float64 resolution at a score near `0.9`.

Add translation and scale tests. Translating the same score cloud and its
threshold together should leave the intended score spread and gate memberships
unchanged up to numerical error. Include a mixed batch of narrow and broad nodes,
then compare one node alone with that node inside the batch. Run this on GPU as
well: different reduction orders can change the sign of the cancelled variance.

This repair is a prerequisite for interpreting B4's new termination counters on
small regions. Otherwise `gate_saturated` can describe an arithmetic artifact of
temperature selection rather than an optimization property.

### 32.4 P2: recovery detects a stuck majority but misses a large unmoved minority

Source: [`HitAndRunSampler._stuck`](src/region.py#L310).

The new check recovers a node only when fewer than half its walking chains moved.
That repairs the previous 20-D case, where none moved. A node can still return a
large singular component at its initial corner while reporting no insufficiency.

An executed 4-D unit-box example used three corner seeds, node seed 0, 20 requested
points, and `walk_steps=max_walk_steps=8`. The result was:

```text
unmoved endpoints = 8 of 20
insufficient_sampling = False
mean over all coordinates and endpoints = 0.21362382476731817
```

The true uniform distribution has no atom at that corner. The aggregate mean is
supporting context; the more direct finding is that 40% of the outputs remain
exactly at a zero-volume point. Since 60% moved, coordinate recovery is never
attempted. The aggregate drift check also passes in this case.

For an isotropic walk starting at a corner, the ideal probability of never taking
a nonzero step over `t` trials is

\[
\left(1-2^{1-d}\right)^t.
\]

At `d=4, t=8`, this is about 0.344. The observed unmoved fraction is therefore a
predictable finite-walk effect, not an implausible random-number accident.

The design now correctly labels the sampler as approximate and its drift result
as a diagnostic. This finding does not reinstate a demand for a mixing proof.
It identifies a large, cheap-to-detect exploration failure that the chosen
majority threshold currently ignores.

Record an unmoved-chain fraction per node. Recover the affected chains or the
whole node within a stated geometry budget, and flag a remaining fraction above
an explicit small diagnostic threshold. Avoid treating 49% unmoved as a success
condition merely because another 51% escaped. An interior-start recovery can be
more useful than coordinate directions for a rotated polytope whose feasible
corner cone contains no coordinate axis. Movement remains necessary evidence,
not evidence of mixing by itself.

If recovery depends on current chain outcomes, its outputs also retain adaptive
selection effects. A separate production stage after fixing geometry and walk
length remains the cleaner experiment from sections 22 and 30.4. The new source
documentation now states the weaker current behavior honestly; keep that
distinction when reporting validation results.

### 32.5 Capacity warnings do not yet describe actual batch admission

Source: the batch loop in
[`_run_local_refinement`](src/engine/adaptive_clustering.py#L303).

The candidate-array estimate now depends on `n_bins`, and warnings disclose the
unsafe floor and oversized-task cases. Those are useful improvements. The source
and warning still say an oversized cluster is processed alone, which is not
always what the cumulative-endpoint rule does.

A public-API probe created two grid roots containing 30 and 150 training rows,
set `batch_size=100`, and recorded calls to `RegionPlan.process_batch`. The
executor warned that the 150-point node would be processed alone, then called the
plan once with:

```text
sizes = [30, 150]
total admitted rows = 180
```

The first cumulative endpoint exceeding 100 is included, so the preceding node
and the oversized node enter the same batch. The same rule can exceed the target
with two individually admissible nodes: a whole crossing node is included rather
than deferred.

This does not show an actual GPU out-of-memory event. It shows that a disclosed
fallback policy is not implemented as stated, and that estimating one oversized
node's requirement is insufficient to predict the batch peak.

For region mode, choose the largest prefix of pending whole nodes that fits the
capacity estimate. If the first node alone exceeds the capacity, apply the
documented oversized-task policy to that node alone. Keep the legacy event order
and batching contract separate if changing legacy boundaries would change RNG
streams. Add a test where the oversized node follows a smaller one, not only the
case where it is the first node.

The warning-only approach can remain an explicitly documented prototype limit
while GPU profiles are collected. It cannot support a hard memory-bound or full
out-of-core claim. No disk spilling, streamed initial evaluation or single-node
streamed fitting was introduced by these fixes.

### 32.6 Loaded topology is checked; loaded geometry still needs validation

Source: [`RoutingTable.from_arrays`](src/region.py#L1280) and
[`AFINNISurrogate._install`](src/surrogate.py#L424).

The compact topology reconstruction and model leaf-count check are useful and
should stay. Domain and root objects are still constructed directly from arrays
without enforcing the geometry contract used during normal configuration.

A focused serialization probe built a valid one-leaf unit-interval table, replaced
its saved `domain_upper` with `[0]`, then loaded it. Loading succeeded. Routing
`[[0]]` returned leaf 0 with `outside=False` despite a zero domain width and an
invalid normalized coordinate.

This is a malformed-artifact case. It was not produced by ordinary training, and
it does not imply that a valid current save/load round trip fails. It demonstrates
that "loaded table validated" currently means topology validation, not complete
validation of the geometry used by routing and audit.

Validate at the common construction boundary:

- Lower and upper bounds are finite one-dimensional vectors of the same
  nonzero dimension, with every width strictly positive.
- Grid bins have the domain's dimension and are positive integers, with a root
  count that can be represented safely.
- Voronoi centers have matching dimension and finite coordinates. Handle loaded
  duplicate centers without silently renumbering existing logical root ids.
- The routing dimension matches the local model's input dimension, and the
  persisted leaf metadata has the same row count and key ordering as the routed
  model rows.

The last check is a consistency check across stores. Equal leaf counts alone do
not establish that the routing table and leaf arrays agree on what leaf row `i`
means. Use the stored `(root, path_code)` metadata to validate that association.
This does not require cryptographic integrity or a new persistence framework.

### 32.7 Benchmark option guards still leave two scientific mismatches

Source: [`scripts/art_comparison.py`](scripts/art_comparison.py#L98).

The new guard rejects non-affine degrees and unsupported error metrics. It still
accepts `--no-leaf-bias` and arbitrary `--relative-error-floor` values. ART uses
those settings, while region mode still constructs an affine model with an
intercept and pointwise training with a fixed `1e-12` floor.

The following small command completed successfully during this review:

```bash
AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. .venv/bin/python \
  scripts/art_comparison.py \
  --art-path ../Adaptive-Regression-Trees-ART- --skip-art \
  --benchmark quadratic --dimension 2 --max-depth 0 \
  --sample-multiplier 3 --n-test 20 \
  --no-leaf-bias --relative-error-floor 0.1
```

`--skip-art` makes this a cheap configuration probe; it is not a fresh timing
comparison. Its output accepts ART's `leaf_bias=False` and floor `0.1`, while the
region configuration uses `AffineLSQModel` with `b_fit` and the fixed floor. The
test-set evaluator uses the requested floor, so both methods would still be
scored with that floor even though their training stopping rules differ.

Two acceptable policies are:

1. Reject non-default bias/floor combinations in this equivalence-oriented
   comparison script until region mode supports them.
2. Support them explicitly in region fitting/acceptance and record any remaining
   intentional differences between methods.

Printing an accepted argument does not make it a resolved setting for both
implementations. The `CONFIG` line currently prints `split_strategy=None` when
the effective strategy is `covariance_hessian`; the later `COMPARE` line is more
accurate. Print the resolved local-model and acceptance specifications and their
settings, including intercept policy and relative floor. This matters most when
automated sweeps produce tables from combinations beyond the defaults.

The default comparison table is not invalidated by this option mismatch if its
bias and floor are aligned. Its single-seed accuracy and target-failure limits
remain as described in section 30.12. This review did not rerun the full ART
benchmark table.

### 32.8 Interpretation of the improved diagnostics and execution paths

The remaining issues below are mostly interpretation or measurement requirements.
They should not be presented as additional reproduced protocol failures.

The `rank_deficient` counter is calculated from the spatial covariance of a
rejected node during B3/B4 direction-bank construction. It is a useful diagnostic,
but it is not a check of every training fit, every candidate child or B2 gap mode.
A deficient node accepted before splitting never increments it. In exact
arithmetic spatial full rank relates to rank of affine features with an intercept;
in floating point a covariance eigenvalue ratio also squares the corresponding
singular-value ratio. Document what the `1e-12` eigenvalue threshold means. Keep
the plan's broader fit/child-support validation classified as incomplete until an
actual check and recovery policy exist.

The new B4 stopping counters are more informative. `soft_gradient_converged` means
the implementation's predicted first-order improvement at a gate-width scale is
below its estimated margin. It is not a certified bound on distance to a local
minimum. The revised docstring largely makes that distinction. Stable temperature
and variance are needed before those counters can diagnose narrow nodes reliably.

MPI work units now agree, which resolves the previous mismatch. A published
post-grant load still sums the full retained queue, including the processed prefix
and current in-flight batch. An idle thief can therefore select a rank with little
donatable work. Separate ready-tail work, in-flight work and transfer bytes when
measuring utilization. The existing ownership/retirement protocol need not be
replaced to make the scheduling estimate more useful.

Likewise, progress points serve requests only for work not already admitted to the
current batch. The default large batch, sampler/search service latency and initial
single-root concurrency limits remain performance questions. The four-rank pass
establishes the tested CPU protocol behavior; it does not substitute for GPU/MPI
scaling measurements under normal thresholds.

Compact child indices are an appropriate inference representation. The new router
still has Python level-wise active-query processing and dynamic `nonzero` calls.
On GPU those operations can impose synchronization and compaction costs. Compare
this reference with a device traversal kernel on a frozen tree before choosing
further changes. Use completed-device timing, warm-up and transfer-inclusive
latency where appropriate, as described in
[CuPy's performance guide](https://docs.cupy.dev/en/stable/user_guide/performance.html).
No GPU speed claim follows from this CPU review.

### 32.9 Verification record

All verification used explicit `AFINNI_DEVICE` and `AFINNI_MPI` settings. The CPU
and MPI suites ran outside the sandbox restriction that prevented MPICH
initialization during the previous review.

| Verification | Result |
|---|---|
| CPU unit/integration suite, `-m 'not mpi'` | 499 passed, 35 skipped, 12 deselected, 52 subtests passed; 53.15 s |
| Four-rank CPU MPI suite | 12 passed; 46.20 s |
| Fresh reference from `48f59c3` | All 45 regression cases ran without run errors |
| Current regression versus that reference, `rtol=0` | PASS; maximum relative array difference exactly 0 |
| Prior audit NaN/infinity/equality counterexamples | Repaired as reported in section 32.1 |
| Prior one-center Voronoi equivalence | Maximum paired endpoint difference 0 in both repeated dimensions |
| Prior 20-D corner case | 1,000 of 1,000 endpoints moved; short-budget sample flagged insufficient |
| Prior sparse scoring and depth probes | No previous indexing exception; depth 60, 61 and 62 all built |
| Near-coincident 1-D Voronoi cell | 200 of 200 returned samples assigned to another root; insufficiency flag false |
| Narrow-score B4 temperature | Approximately `1e-151` used instead of centered reference `5.80244e-10` |
| Narrow-cloud drift check | False visible-drift flags for tiny shifts with centered standardized size below 0.002 |
| Partly unmoved 4-D corner sample | 8 of 20 endpoints stayed at the corner; insufficiency flag false |
| Oversized node after a smaller node | Both processed together as `[30, 150]` with target batch size 100 |
| Loaded zero-width domain | Accepted; routed a query with `outside=False` |
| Unsupported bias/floor comparison options | Accepted by the new guard in the recorded configuration probe |
| GPU backend availability | Explicit GPU import failed: CuPy is not installed |

The main suite commands were the repository's documented CPU and four-rank MPI
commands. The reference was freshly extracted with `git archive 48f59c3` into
`/tmp/afinni_review_round4_base48f59c3`; no checkout or implementation file was
modified to create it. Both checkouts ran `scripts/regression_check.py --save ...`,
and their saved results were compared with zero tolerance.

The new focused probes are in
`/tmp/afinni_review_round4_new_probes.py`, with JSON results in the matching
`.jsonl` file. The prior probes were rerun from
`/tmp/afinni_implementation_review_checks.py`. Full suite and regression logs use
the `/tmp/afinni_review_round4_*` prefix. These are temporary review artifacts.
The central geometric reproduction can be retained independently as a small unit
test:

```python
base, separation = 0.695, 1e-12
centers = np.array([[base - separation], [base], [base + separation]])
roots = Roots.build("voronoi", 1, 3, centers)
geo = RegionPoints(
    np.array([1]), np.array([0]),
    node_keys(2, np.array([1]), np.array([1])),
    np.zeros((1, 0, 2)), PURPOSE_TRAIN,
)
sampler = HitAndRunSampler(
    Domain(np.zeros(1), np.ones(1)), roots,
    walk_steps=8, max_walk_steps=32,
)
points = sampler.sample(
    ClusterPoints(np.array([[base]]), np.array([1]), region=geo),
    np.array([200]),
)
print(np.count_nonzero(roots.root_of(points) != 1), geo.insufficient)
# Inspected implementation: 200, [False].
```

This is a sampler-level test using valid distinct centers and a feasible start.
It does not depend on persuading k-means to generate that particular geometry.

### 32.10 Recommended next steps

1. Fix non-owning Voronoi bisector evaluation and feasibility scaling. Add the
   near-coincident-center test and direct endpoint ownership/reference checks.
2. Replace raw-moment variance with centered segmented variance in B4 and the
   sampler diagnostic. Add narrow translated clouds to the CPU and GPU tests.
3. Expose and handle partial chain immobility, rather than triggering recovery
   only for a stuck majority. Keep local validation labeled as diagnostic.
4. Make the oversized-task batching policy match its warning, complete loaded
   geometry checks, and finish the comparison script's bias/floor guard.
5. Establish a correct GPU baseline using the experiment sequence in section
   31.4. Measure B3/B4, sampling, compact routing, peak memory and MPI utilization
   before selecting further optimization work.

The important change in priority is that the original audit and self-plane bugs
are fixed. The next correctness work is stable geometry and stable small-region
statistics. Broader execution optimizations can then be judged against a numerical
reference whose sampling and diagnostics survive these cases.

---

## 33. Review of the follow-up fixes, November 11, 2026

### 33.1 Scope and assessment

This review examines implementation commit `6a9a77a` and documentation commit
`90672cc563ef18b8e2ed149d4f342fc24739580a`, against the previous reviewed state
`5a47695`. It checks the changes against section 32, reruns its counterexamples,
and adds tests for the interactions between the revised sampler and inference.
Only this review document was changed. The numerical probes ran on the CPU.

The fixes are substantial and mostly correct. The previous one-dimensional
Voronoi failure, cancelled variance, fully immobile minority, oversized batch,
and zero-width loaded domain counterexamples no longer fail. The shared executor
still preserves the legacy AFINNI regression results exactly. There is no evidence
here that the integration needs a different overall architecture.

Two sampling issues still deserve attention before treating Voronoi region mode
as a reliable numerical reference. First, the stable sampler and nearest-center
inference disagree in thin multidimensional cells. Second, coordinate recovery
can move a chain off its original corner while leaving it on a zero-volume face,
without setting `insufficient_sampling`. The recovery diagnostic also includes
inactive rows in its statistics, which can conceal movement of the repaired
cohort. These are reproducible issues, rather than requests for a proof of mixing.

| Previous finding | Result in this revision |
|---|---|
| Expanded Voronoi bisectors cancel for close centers | Fixed in the sampler; inference still needs a compatible stable ownership predicate, section 33.2 |
| B4 raw-moment variance collapses its temperature | Fixed in the reproduced narrow-score case; actual temperature now agrees with the centered reference |
| Drift statistics cancel on narrow translated clouds | Fixed in the previous cases; small shifts no longer trigger false drift |
| A minority of chains remains completely unmoved | Recovery now reaches every such chain; detecting movement alone misses residual boundary support, section 33.3 |
| An oversized region node shares a batch with its predecessor | Fixed; the `[30, 150]` case at capacity 100 now executes as two separate batches |
| Loaded geometry and leaf identity are unchecked | Substantially improved; basic geometry and metadata checks are present, with one integer-overflow edge case remaining |
| Comparison accepts a different relative-error floor | Fixed; the script rejects floor `0.1` |
| Comparison accepts `--no-leaf-bias` | The previous review's interpretation was wrong; ART forces an intercept for degree 1, as explained below |

**Correction to section 32.7:** ART's
`examples/tree_benchmark_helpers.py:541` sets
`leaf_include_bias = True if args.leaf_degree == 1 else args.leaf_bias` before
constructing the model. Therefore, `--no-leaf-bias` does not remove the intercept
from the degree-1 ART benchmark. The earlier inference from the requested CLI
configuration to the fitted model was incorrect. The relative-floor mismatch was
real; the claimed degree-1 bias mismatch was not. The current decision to accept
that flag is sound.

### 33.2 P1: stable Voronoi sampling still disagrees with root assignment

Relevant code is `Roots.root_of` at `src/region.py:190`, the revised
`HitAndRunSampler._steps` at line 381, and `RoutingTable.route`.

The sampler now evaluates each bisector using

\[
\Delta_j=c_j-c_r,\qquad
n_j=\Delta_j/\|\Delta_j\|,\qquad
\mathrm{slack}_j=\tfrac12\|\Delta_j\|-n_j^T(u-c_r).
\]

This repairs the expanded-square cancellation. However, root assignment still
forms complete squared distances and takes their `argmin`:

\[
d_j^2=\sum_k(u_k-c_{j,k})^2.
\]

Direct differences avoid one source of cancellation, but summing squares can
still erase the component that distinguishes two nearby centers. A large common
tangential distance can dominate a much smaller difference along the separating
normal. The result is a false floating-point tie, which the lowest-index rule
then resolves as if it were a geometric tie.

#### A two-dimensional counterexample with a correctly assigned start

Use the unit-square domain and centers

```python
separation = 1e-8
centers = np.array([
    [0.695 - separation, 0.5],
    [0.695,              0.5],
    [0.695 + separation, 0.5],
])
seed = np.array([[0.695, 0.9]])
```

The middle cell is a thin vertical strip. The current `root_of(seed)` correctly
returns root 1 at this separation. Sample 200 endpoints in that root using seed 2,
path code 1, `PURPOSE_TRAIN`, `walk_steps=8`, and `max_walk_steps=32`.

All 200 endpoints belong to the middle cell according to an independent
80-decimal-digit distance calculation on the exact stored binary64 coordinates.
Nevertheless, `root_of` assigns 21 endpoints to another root. A routing table with
one leaf per root reproduces those 21 wrong leaf selections. The sampler's
insufficiency flag is false.

| Center separation | Current start owner | Endpoints truly in middle cell | Endpoints routed elsewhere |
|---|---:|---:|---:|
| `1e-6` | 1 | 200 | 0 |
| `1e-8` | 1 | 200 | 21 |
| `1e-10` | 0 | 200 | 200 |
| `1e-12` | 0 | 200 | 200 |

The `1e-8` case is the strongest reproduction because its start already satisfies
the implementation's own root assignment. The two smaller separations also
demonstrate incorrect assignment of the initial geometric query; they are not
needed to establish the sampler-to-inference disagreement.

For this construction, the shared tangential contribution is about `0.16`.
Differences between the squared normal distances are of order `1e-16` or less at
separation `1e-8`, and smaller near the cell boundaries. Some disappear when added
to `0.16`. In one dimension there is no large tangential contribution, which is
why the newly added one-dimensional regression test passes.

This affects the intended contract between training and prediction. Rows generated
for a node can later route to a different root and a different fitted model. It
does not imply that the default box-root benchmark is affected, and this review
has not measured the frequency of such center configurations in ordinary k-means
runs. It is still a correctness failure for valid distinct centers in a supported
root geometry.

#### Recommended repair

Keep the stable sampler formula. Repair ownership so that root selection and
sampling use compatible geometric comparisons. For a candidate owner `r`, the
distance difference can be evaluated without the shared tangential term:

\[
\|u-c_j\|^2-\|u-c_r\|^2
=\|\Delta_j\|^2-2\Delta_j^T(u-c_r).
\]

One practical approach is to retain the tiled distance calculation as a fast
candidate search, then certify the candidate against stable bisectors. Ambiguous
queries need a stable candidate comparison or a robust fallback, with an explicit
deterministic rule for true ties. A reference implementation can use higher
precision to check disputed cases. Production GPU code should isolate ambiguous
rows rather than transfer every query to the CPU.

Changing every squared-distance expression to the difference formula is not a
universal exact-arithmetic guarantee. A query almost exactly on a general
bisector can still require a rounding-error bound and fallback. The immediate
requirement is to remove the large common tangential term and make the ownership
contract consistent. Do not merge distinct nearby centers silently or increase
the feasibility tolerance to conceal disagreements.

Tests should include:

1. The two-dimensional case above, with exact-input high-precision ownership as
   the reference and an assertion on routed leaf IDs.
2. Rotations and translations inside normalized coordinates, several dimensions,
   and both center orderings. True ties should obey the declared index rule;
   false ties should not be introduced by unrelated coordinate components.
3. Sampler endpoint ownership, ordinary training assignment, and saved-model
   routing checked against the same predicate.
4. CPU/GPU ownership decisions and batch/tile independence. Prediction tolerance
   alone cannot detect selection of a different leaf with accidentally similar
   coefficients.

The design's claim that thin cells now stay exact should be narrowed until this
cross-component test passes. Stable chord geometry has been repaired; complete
training-to-inference consistency has not.

### 33.3 P2: corner recovery can leave a singular boundary population unflagged

Relevant code is `HitAndRunSampler.sample`, especially the new per-chain recovery
selection and the final `U != U0` check.

The change from majority-based recovery to recovering every unmoved chain fixes
the previous defect. The remaining issue is what counts as successful recovery.
A coordinate walk can change one coordinate while keeping the others exactly at
the corner values. Such a chain has moved, but it remains on a lower-dimensional
face of a full-dimensional cell.

The previous 4-D counterexample now returns:

```text
20 endpoints, walk_steps = max_walk_steps = 8
endpoints still at the original corner: 0
endpoints with at least one coordinate exactly zero: 7
insufficient_sampling: False
mean over all endpoint coordinates: 0.3873357947006743
```

This is not an artifact of the test omitting a real ancestor cut. With the
nonredundant constraint `u0 <= 0.5`, the same seed and node key again leave 7 of 20
endpoints on a coordinate face and return a false insufficiency flag. The four
coordinate means are approximately
`[0.2102801, 0.3865013, 0.4129816, 0.3089469]`, compared with uniform-cell means
`[0.25, 0.5, 0.5, 0.5]`.

The small-sample mean discrepancy alone would be weak evidence. The persistent
face population is the decisive observation. Under the continuous uniform law of
these positive-volume boxes, those faces have zero probability. Seven exact
boundary endpoints at this scale cannot reasonably be described as rounding of
ordinary interior samples.

For a coordinate walk in a `d`-dimensional box starting at a corner, a particular
coordinate remains untouched after `t` independent axis selections with
probability

\[
\Pr(\text{coordinate untouched})=(1-1/d)^t.
\]

At `d=4, t=8`, the probability that at least one coordinate remains untouched is

\[
4(3/4)^8-6(1/2)^8+4(1/4)^8\approx0.3771.
\]

Moving at least once therefore does not establish departure from the boundary.
This is a support problem before any discussion of mixing rates. The current
20-D tests have the same blind spot if they assert only that each point changed
and that coordinate means are close to one half. Means can look reasonable while
many endpoints still lie on faces.

#### Recommended repair

Track whether recovered chains leave the active boundary faces, using geometry
and scale-aware numerical resolution. Flag or continue chains that remain on an
inherited zero-volume face. A blanket `1e-9` interior margin would be inappropriate
for thin cells; this test must distinguish representational uncertainty from a
macroscopic unvisited coordinate.

For an axis-aligned box, a full coordinate sweep gives a simple bounded recovery:
each coordinate gets a conditional draw. In that special geometry, the
conditionals factorize, so one complete sweep from a corner produces independent
uniform coordinates. More generally, changing the recovery to systematic sweeps
at least prevents random axis selection from missing a coordinate altogether.
It does not make a general-polytope walk uniform after one sweep.

An oblique polytope needs a separate fallback policy. At some vertices, every
coordinate-axis chord is degenerate even though the polytope has positive
volume. A feasible interior-point computation, such as a bounded Chebyshev-center
recovery, can address that case at additional cost. Otherwise report insufficient
sampling. Do not claim that coordinate recovery can escape every convex corner.

Keep the distinction between obtaining an interior start and producing usable
samples. A repaired start should feed a defined production walk; adaptive recovery
alone does not establish stationarity. The documentation already acknowledges
that limitation, and that acknowledgment should remain.

### 33.4 P2: the recovery drift test averages over chains it did not walk

Relevant code is `HitAndRunSampler._drifting` at `src/region.py:450` and its call
from the recovery `run` closure.

The revised code passes the subset of unmoved chains as `rows` for recovery.
However, `_drifting` uses `walk` only to decide whether a node has any active rows.
Its means, centered variances, radius center, sample count, and minimum sample
size still use all `n_new` rows of the node. During recovery, most of those rows
may be held fixed.

Consequently, the diagnostic can be dominated by a large unchanged population.
A simple deterministic test isolates this without relying on a statistical
fluctuation:

```python
mid = np.r_[
    np.linspace(0.1, 0.100001, 4),
    np.linspace(0.1, 0.9, 96),
][:, None]
end = mid.copy()
end[:4] += 0.8
walk = np.arange(100) < 4
```

Calling the current `_drifting` on all 100 rows with that mask returns false.
Calling it on the four active rows, using their actual count, returns true. The
active population has moved almost the full domain width. The other 96 endpoints
did not participate in the recovery step.

This is a synthetic diagnostic test, not a claim that those exact midpoint and
endpoint arrays were produced by a random walk. It demonstrates the mask/count
mismatch directly. The real boundary example in section 33.3 separately shows
that the recovery path can already pass inadequate endpoints.

Compute cohort statistics using the rows actually being diagnosed: masked counts,
masked means, and masked centered moments. Apply the sample-count threshold to
that cohort too. If too few chains were repaired for a useful moment diagnostic,
state that limitation and use the recovery/support policy rather than pretending
the full node count supplies independent recovery observations.

An aggregate diagnostic over the whole returned cloud can also be useful. It
answers a different question and should not replace a diagnostic of the repaired
cohort. Add a test with heterogeneous recovery fractions and an unrelated fixed
population; adding fixed rows should not make the active cohort's diagnostic
silently pass.

### 33.5 Lower-priority loaded-grid hardening: positive bins can still overflow

The new loaded-geometry checks reject the previous malformed examples and check
leaf-to-routing identity. That is a real improvement. One requested invariant
from section 32.6 remains unchecked: the root-count product must be representable.

`Roots.n_roots` computes `np.prod(self.bins)` in a fixed-width integer type. With

```python
bins = np.array([2**62 + 1, 4], dtype=np.int64)
```

the mathematical product is `2**64 + 4`, but the implementation gets 4. A table
with four root leaves builds and survives `from_arrays`. Routing `[0.5, 0.5]`
then fails with:

```text
IndexError: index -9223372036854775806 is out of bounds for axis 0 with size 4
```

This concerns malformed or externally constructed artifacts, not ordinary grid
training with sensible parameters. It is lower priority than sections 33.2–33.4.
Use a Python-integer product for validation, reject counts beyond the supported
index range, and verify that the root count agrees with the supplied topology
before constructing large root arrays. Validate any cumulative strides before
casting them to device index types. A four-row artifact should not trigger an
enormous allocation merely because its bins claim billions of roots.

### 33.6 Consequences for batched GPU execution and MPI

The correctness fixes should remain. Performance work should preserve the new
center-difference geometry and centered variance, rather than restoring the
unstable expanded formulas to recover throughput.

#### Voronoi workspace and repeated geometry

The revised sampler explicitly materializes `D` and `normal` with shape
`(chains_in_chunk, number_of_roots, dimension)`. The old formulation mostly used
`chains_in_chunk × number_of_roots` products. The increased memory traffic buys
correctness, but duplicates the same center differences for every chain belonging
to a root. `_steps` rebuilds them when another walk segment is invoked.

For `n` chains, `R` centers and dimension `d`, the two persistent float64 tensors
alone require approximately `16 n R d` bytes. At `n=1024, R=256, d=16`, that is
64 MiB, before the other arrays. The implementation's new chunk-width estimate
reduces the chunk in such cases, so this is a cost illustration, not a claim that
this exact full allocation occurs. The current fixed `CHUNK_ELEMENTS` cap also
limits the estimate, not the complete device-memory peak or a user-specific
remaining budget. A minimum chunk of one cannot bound arbitrarily large root
geometry.

A useful next optimization is grouping chains by owning root and reusing
`Delta`, normalized normals, and half-distances within each group. For each root,
evaluate

\[
\mathrm{slack}=h-(U-c_r)N_r^T,\qquad
\mathrm{direction}=G N_r^T.
\]

This preserves subtraction of the owning center before projection. It permits
matrix products over many chains while storing normals once per root group,
rather than once per chain. A cache for only the roots present in a batch is an
option; an unconditional `R × R × d` global cache is not necessary. Root tiling
can bound both geometry storage and the chain-by-center products.

This is an optimization candidate, not a measured GPU speedup. Many tiny root
groups may replace useful batched work with launch overhead. Benchmark both large
groups and many small groups, including sorting/gathering cost. The repaired
ownership predicate from section 33.2 must also be included in the timing.

#### Centered statistics

The extra centered reduction is justified. Returning to `E[s²] - E[s]²` to save a
pass would restore a demonstrated numerical failure. If profiling identifies
these passes as expensive, use a stable segmented count/mean/M2 reduction or
reuse per-node centered coordinates. Verify its narrow-cloud behavior and its
batch-size sensitivity. A reduction that changes node decisions through a
different rounding order needs explicit numerical qualification.

The new B4 temperature matches the previous centered reference:
`5.80244273635628e-10`, versus `5.802442736356279e-10`. This supports the repair
of that defect. It does not establish B4 as the fastest split method or turn its
termination margin into an optimization certificate. Keep B3 as the hard
incumbent and compare accepted hard SSE, oracle calls, leaf counts, and elapsed
time. A low `soft_gate_saturated` counter alone is not an optimization benchmark.

#### Batch admission and work stealing

The new region prefix rule is appropriate: admit the largest whole-node prefix
that fits the point budget, or one oversized node. It repairs the immediate
admission defect while leaving legacy AFINNI boundaries unchanged. It is still
an admission estimate, not a guarantee of a fixed device-memory peak. Shared
sampler scratch, search arrays, resident model state and allocator behavior need
separate accounting.

The four-rank tests pass, including region prediction/key/status agreement and
actual steal grants. The current region MPI worker primarily exercises box roots
and ordinary interior sampling. It does not cover the thin Voronoi or recovery
cases above. Extend it with a bounded adversarial geometry case after fixing the
serial predicates; otherwise a shared numerical bug can agree perfectly across
all ranks.

No new MPI protocol defect was reproduced. The acknowledged stale load estimate
after a grant remains a scheduling concern. Measure useful donated work, denied
requests, queue wait time and rank idle time before changing the protocol. GPU
ranks also need progress measurements around long sampler segments, because
batch-size tuning alone does not establish how quickly a victim services steals.

#### Inference and experiment order

Compact child-index routing remains the right reference inference path for the
region model. Ordinary kNN cannot repair disagreement about which root owns a
point; it would define another predictor. Correct root assignment first, then
benchmark the existing level-wise traversal against a fused traversal on a fixed
tree. Separate root selection, traversal and leaf evaluation in the measurements.

On a GPU machine, start with frozen geometry and fixed node inputs so numerical
differences do not change the workload during a kernel comparison. Then run the
full adaptive builder. Include box/grid and Voronoi roots, coherent and scattered
queries, several dimensions and node sizes, peak allocated memory, transfer
volume, and achieved GPU occupancy or utilization. Follow with multi-rank runs.
These measurements can distinguish repeated geometry construction, small-kernel
launch cost, narrow-cell walk cost, and genuine load imbalance.

The ART comparison script now rejects unsupported floors and reports the resolved
AFINNI model and acceptance components. Two small reporting details remain: the
ART dictionary still contains requested `leaf_bias=False` even though the
effective affine model has a bias, and the generic registered AFINNI `sampler`
entry can say `convex_combination` while region mode actually runs hit-and-run.
The separate `region.sampling` entry makes the latter recoverable, but neither
dictionary should be described as entirely resolved runtime behavior. Print an
explicit effective sampling/model block when saving benchmark provenance.

### 33.7 Verification performed for this review

All test and probe commands explicitly selected CPU/GPU and MPI mode.

| Check | Result |
|---|---|
| CPU suite, `-m 'not mpi'`, unrestricted run | 517 passed, 35 skipped, 12 deselected, 52 subtests passed; 45.79 s |
| Four-rank CPU MPI suite | 12 passed; 33.45 s |
| Fresh legacy reference from `5a47695` | 45 regression cases, zero run errors |
| Current legacy regression against that reference, `rtol=0` | 45 cases compared; maximum relative array difference 0 |
| Earlier near-coincident 1-D sampling case | 0 of 200 ownership disagreements; insufficiency false |
| Earlier B4 and drift cancellation cases | Temperature repaired; all four reproduced benign shifts unflagged |
| Earlier oversized admission | Processed node sizes `[[30], [150]]` at capacity 100 |
| Earlier zero-width loaded domain | Rejected with `ValueError` |
| Two-dimensional Voronoi ownership at separation `1e-8` | 21 of 200 routed leaf IDs disagree with high-precision ownership |
| Corner recovery with and without a genuine ancestor cut | 7 of 20 endpoints remain on coordinate faces; insufficiency false |
| Recovery diagnostic with an active subset | Full population passes; active cohort fails |
| Loaded overflowing grid | Accepted, then query raises `IndexError` |
| Comparison floor `0.1` | Rejected before training |
| Comparison with `--no-leaf-bias`, degree 1, floor `1e-12` | Accepted appropriately; ART source confirms the effective intercept |
| Explicit GPU suite | Could not collect tests because CuPy is not installed; no GPU correctness or performance claim |

The first sandboxed CPU run had 516 passing tests and one failure in the backend
subprocess's MPICH initialization. The full rerun outside that restriction passed
all 517 tests. That initial failure is environmental, not an identified source
regression. The legacy comparison used a fresh extraction of the immediate
pre-fix state, rather than reusing the older section 32 reference. Deliberate
region sampling changes are not claimed to be bit-identical to the prior region
implementation.

The independent numerical probe is `/tmp/afinni_review_round5_probes.py`, and its
results are in `/tmp/afinni_review_round5_probes.jsonl`. The old section 32 probe
was rerun with results in `/tmp/afinni_review_round5_old_probes.jsonl`. Suite,
comparison and regression logs use the same `/tmp/afinni_review_round5_` prefix.
These are temporary artifacts; the inputs and numerical reasoning above are
included here so the findings can become maintained tests.

Recommended order of work:

1. Align Voronoi ownership with stable bisector geometry and add the 2-D routed
   leaf-ID regression. This is the highest-priority remaining correctness issue.
2. Make recovery detect residual boundary support and diagnose the active cohort.
   Preserve explicit insufficient-sampling outcomes when the budget is inadequate.
3. Add checked grid products and clearer effective benchmark configuration output.
4. Establish GPU timings and memory peaks for the corrected implementation before
   selecting root-grouped geometry, fused routing, or further MPI scheduling work.

The existing architectural direction remains suitable for these changes. The
next work is to make the same geometric region mean the same thing to sampling,
training assignment and inference, and to make recovery diagnostics describe the
chains they actually inspected.

---

## 34. Architectural reconsideration from first principles

### 34.1 Conclusion and scope

Approach 3 is a good integration baseline. It is not established as the best
algorithm for approximation, GPU throughput, oracle efficiency, or inference.
Those are different optimization problems, and the proposed node-storage schemes
answer only part of them.

The strongest direction is to retain Approach 3's ownership and persistence
contracts while relaxing three implementation choices: full carried paths for
every local descendant, a host round trip at every refinement round, and a fixed
sequence of uniform top-up followed by affine fitting and B4 splitting. These
choices make a reference implementation manageable. None is a mathematical
requirement of ART.

The most useful additional experiments are:

| Candidate | Main change | Expected benefit, subject to measurement |
|---|---|---|
| Bounded resident subtree execution | Process compatible descendants locally before spilling or donating them | Less host/device traffic and repeated ancestry copying |
| Shared-bin hard split search | Build sufficient statistics once per bin, then scan prefixes and suffixes | Remove repeated full-data Gram construction across thresholds |
| Geometry-dependent sampling | Exact box/rejection sampling where practical; rounded hit-and-run elsewhere | Better-supported samples and less wasted walk work |
| Local model enrichment before splitting | Try a small number of quadratic directions before creating children | Fewer leaves, oracle calls, constraints and routing steps on smooth curvature |
| Query-measure-aware refinement | Allocate work by deployed risk and expected use, with explicit exploration | Avoid spending the same effort on every small, rarely queried cell |

The first two primarily change execution. The next three change the approximation
algorithm and need separate accuracy experiments. Combining them immediately would
make it impossible to identify what caused an improvement or regression.

This section builds on the earlier architectural discussion in sections 5–16;
it does not replace the unresolved correctness findings in section 33. Sources
rechecked include the current `RegionPlan`, shared executor, queue and local-model
code, and the sibling ART checkout at
`854924af9ec8555a11db10de6ab75f429a31be2d`, particularly its builder, BFS dispatcher,
sampling, GPU cohort adapter and prediction code. The AFINNI implementation is
still the `90672cc` state reviewed in section 33. No new execution algorithm was
implemented here.

### 34.2 What should the combined method optimize?

Start with the deployed predictor and the intended distribution of queries,
denoted by `nu`. For a tolerance-normalized error `e(x)`, plausible requirements
include an integrated loss, a small probability `P_nu[e(x) >= 1]`, or a bound at
every point in the domain. They require different sampling and refinement
policies. A finite random audit supports a probability statement, not an
everywhere bound. Neither Morton ordering nor exhaustive tree coverage changes
that distinction.

A practical lifecycle objective is

\[
\min\quad T_{\mathrm{build}}+N_q\,\mathbb E_\nu[t_{\mathrm{predict}}(x)]
\quad\text{subject to a stated accuracy requirement and memory limits}.
\]

Model size, update cost and continuity can be further constraints. If queries
are effectively unlimited, a slower build that produces a smaller or shallower
model may be worthwhile. For a frequently rebuilt surrogate with few queries,
the same tradeoff can be wrong.

Break build work into quantities that the design actually changes:

\[
W= W_f+W_{\mathrm{sample}}+W_{\mathrm{fit}}+W_{\mathrm{split}}
   +W_{\mathrm{transfer}}+W_{\mathrm{control}}.
\]

These terms are accounting categories, not necessarily additive elapsed times:
oracle work, transfers and GPU computation can overlap. Even with ideal
scheduling, elapsed time is bounded below by the longest dependency chain and by
the work divided by the capacity of each required resource. A tree has genuine
dependencies: the child's region, inherited rows and useful sampling policy
depend on the parent's committed split.

This yields several conclusions without benchmarking:

- Work stealing cannot eliminate the serial dependency along a difficult branch.
- Fewer oracle calls do not automatically imply faster construction when sampling
  and split optimization are expensive.
- A representation that reduces the number of necessary leaves can save work in
  every stage, rather than just accelerate a particular kernel.
- Fast processing of a biased or poorly explored sample set is not progress
  toward a domain-level accuracy requirement.
- Storage layout, task scheduling, geometry, sampling and approximation order
  should be separable choices.

AFINNI already supplies efficient segmented data processing and distributed
ownership. ART supplies explicit geometric coverage, conditional sampling and
residual-based partition selection. The integrated method should preserve those
useful ideas without inheriting every scheduling and modeling choice of either
implementation.

### 34.3 Morton IDs: useful roles, and limits that matter here

#### A path code is an ancestry coordinate

For a node at depth `h` with sentinel path code `p`, write

\[
p=2^h+b,\qquad 0\le b<2^h.
\]

Its descendants occupy the abstract binary-path interval

\[
I(p)=[b\,2^{-h},(b+1)\,2^{-h}).
\]

This is useful for prefix tests, subtree grouping and ordered storage. It says
nothing about physical volume. An extremely unbalanced geometric split still
divides this abstract interval into equal halves. The interval's length is not
an estimate of query probability, sample cost or region volume.

For example, terminal paths `2, 6, 7` occupy `[0,1/2)`, `[1/2,3/4)`, and
`[3/4,1)`. They can represent very different physical volumes depending on the
stored planes. Sorting sentinel integers also groups depths, so it is not a
general subtree-contiguous traversal order. To obtain that order, use explicit
tree traversal or depth-normalized path prefixes with a stated ancestor ordering.

Keep three notions separate:

| Identifier/layout | Purpose | Should change when jobs move? |
|---|---|---|
| Logical `(root, path)` identity | Provenance, RNG streams, reproducibility | No |
| Local array index or handle | Fast addressing on one rank/device | Yes, if useful |
| Final inference ordering | Cache behavior and query traversal | Yes, during compilation |

The current separation is correct. Adding a globally coordinated integer ID
allocator would make distributed execution harder without improving the
mathematical tree.

#### Spatial Morton ordering is a different tool

A spatial Morton code interleaves quantized coordinate bits. It is useful for
building or ordering spatial indices; NVIDIA's tree-construction example uses
this distinction explicitly. It does not say that neighboring oblique path IDs
are close in Euclidean space. [NVIDIA, tree construction on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)

There is also a dimensional limit. A 64-bit spatial code with equal precision
per coordinate has at most `floor(64/d)` bits per dimension: one bit at `d=40`,
and no complete bit per coordinate at `d=80`. Longer codes or projected
coordinates are possible, but add storage/sorting cost or discard geometric
information. This is not a strong default mechanism for high-dimensional
surrogate routing.

Reasonable uses here are optional low-dimensional query ordering, coarse spatial
root construction, and locality experiments. Sorting leaf centroids by a Morton
code does not permit changing the tree's decision planes while keeping its
semantics. It also does not make distant memory accesses coalesced merely because
the integer IDs are consecutive.

The parallel construction of a hierarchy over a known set of objects should not
be confused with discovering ART's adaptive tree. In ART, the future leaves are
unknown until sampling, fitting and splitting have run. A fast final layout pass
can parallelize assembly; it cannot remove those scientific dependencies.

### 34.4 Approaches 1, 2 and 3 solve different storage lifetimes

Approach 1's compact parent/plane records are the best authoritative history.
Approach 2's ordered level buffers are one convenient way to organize that
history. Approach 3's complete work items are a good transfer contract. None
requires using the same representation in host queues, MPI packets, GPU scratch
and the final predictor.

The important optimization is amortization. For a node with `n` chains, `t` walk
steps and `h` ancestor constraints, materializing its path costs order `hd`
coefficient accesses once, whereas evaluating those constraints throughout the
walk costs order `nthd`. Ignoring latency and constants, the ratio is about
`1/(nt)`. Gathering a compact ancestry once can be cheap relative to using it
many times. Following parent pointers inside every walk step would lose that
advantage.

This explains why the concern about irregular parent access is valid but should
not decide the whole architecture. It is the placement and frequency of the
gather that matter. Approach 2's repeated `searchsorted` is not inherently better
than direct local parent handles; both should stop at a contiguous materialized
sampler input.

There is a larger saving when a transfer or resident group shares a prefix.
Consider `b=2^r` frontier nodes below a common depth-`h0` ancestor, with a complete
relative tree of depth `r`. Explicit paths store

\[
b(h_0+r)
\]

plane rows. The union of their ancestor records contains only

\[
h_0+(b-1)
\]

rows. With `h0=20`, `b=256`, and `r=8`, this is 7,168 versus 275 rows, a factor of
26.1 for the plane payload alone. Points, outputs, handles and serialization
overhead remain, so this is not a 26-fold total-memory or communication claim.

The preferred extension is therefore a bounded block of related nodes:

- Local unresolved jobs reference an immutable ancestry store.
- A batch materializes the constraints it needs, optionally reusing common
  prefix products across chains that share a root or ancestor.
- A migration packet contains the union of required ancestry plus the point
  blocks and logical keys. The receiver reconstructs local handles.
- Final inference arrays are built independently of those handles.

This combines all three approaches without remote ancestry fetches. Keep the
full-path implementation as a reference and retain it for small or unrelated
packets where compression would cost more than it saves. Exact deduplication is
safe; approximate removal of an apparently redundant face changes geometry
unless redundancy is proved.

### 34.5 Execution candidate: bounded resident subtrees over a global frontier

The current shared executor reads a host batch, processes it, writes children to
a host builder, and revisits them in a later round. That is robust when the
frontier exceeds VRAM. It forces transfer and allocation work even when several
generations of a small subtree could stay resident.

For a point payload of `b_point = 8(d+q+g*d*q)` bytes, each additional round that
exports and reloads a surviving row adds approximately `2*b_point` transfer
bytes. Summing that over its remaining inherited lifetime can dominate a cheap
oracle and small affine fits. The exact amount depends on top-up, termination,
validation and gradient policy; it should be measured from actual row lifetimes.

A better candidate scheduler has a bounded local resident frontier plus a
host-backed frontier of ready tasks:

```text
select a group of compatible ready nodes within byte and time budgets
load their point blocks and materialize their geometry
repeat while resident capacity and a bounded execution allowance remain:
    sample/evaluate/fit compatible nodes in batches
    finalize terminals and append immutable output records
    refine rejected nodes and retain selected children locally
    service communication and oracle-completion events
    expose surplus unstarted children for donation or spill
flush remaining ready work to the host frontier when necessary
```

This is a bounded blend of breadth-first and depth-first execution. Pure DFS
minimizes a live frontier but often starves batched linear algebra. Pure BFS
exposes parallelism but can maximize memory and transfer traffic. A bounded
frontier can retain enough sibling work for efficient batches without retaining
the entire tree level.

Compatible cohorts should reflect actual kernel requirements: feature count,
output count, sample-count range, geometry class, constraint-count range and
optimizer state. True depth remains metadata. Two nodes at different depths can
share a fit kernel even if their samplers need different geometry buckets.
ART's existing BFS dispatcher already permits cohorts crossing levels, while
its resident GPU adapter handles selected split work. The combined engine can
extend that principle across more stages instead of copying ART's object-level
builder.

There are necessary limits:

1. Resident execution must not hide all ready children from idle ranks. A packet
   or time allowance should bound how long one rank retains transferable work.
2. Stealing should remain at clean ownership boundaries. Started optimizer or
   chain state can stay local; unstarted descendants can be donated.
3. Byte admission must cover the maximum overlapping stage lifetimes, including
   pinned transfer buffers, geometry, fit/search workspaces and outputs.
4. Priority needs an age or fairness component so small awkward jobs do not wait
   indefinitely for a perfect cohort.
5. Stable node-based random streams must remain independent of placement. A
   scheduler change alone should not alter samples or accepted split decisions
   on the CPU reference path.
6. Termination must count resident work, pending oracle calls and in-flight
   transfers. A rank can have no host-queued or currently donatable nodes and
   still be busy. Estimated transferable seconds must not become the liveness
   signal.

This can evolve from the present executor. It does not require a new distributed
runtime, a general property framework, or migration of arbitrary Python objects.
Use the existing numerical modules, registries and column schemas.

#### Out-of-core requires separate device and host policies

A host-backed queue is out-of-core relative to VRAM, not relative to host RAM.
Bounded resident execution improves transfer reuse but does not by itself bound
the global frontier. Likewise, an initial full-array upload and final replicated
model can exceed device memory even when each refinement batch fits.

If the intended workload exceeds RAM, introduce an explicit spill layer for
ready point blocks and immutable records. Jobs should refer to logical block
handles; the executor packs only admitted rows into staging buffers. Splitting
must account for simultaneous parent, child and packing storage, and prefetch
must be bounded by credits. Memory mapping alone does not guarantee a bounded
working set or efficient access when repeated splits scatter rows across files.

Use subtree-related blocks to improve sequential access, but bound their size
so one skewed subtree cannot monopolize memory. Shared ancestry saves geometry
bytes; it does not remove point payloads. Replacing all rows by sufficient
statistics is also insufficient for the general algorithm because later split
directions and predicates need individual rows. A bounded reservoir plus
streamed statistics is a different approximation policy and must be tested as
such. Treat disk spill and sharded final-model assembly as explicit features,
rather than consequences of adding ART to the present queue.

#### Early parallelism has a price

For a fixed per-leaf sample target `n`, a complete refinement with `L` leaves and
no discarded evaluations has approximately `nL` sampled oracle values, excluding
centroid and validation calls. Forcing `R` roots therefore imposes at least `R`
terminal cells and roughly `nR` sampled values under those assumptions. For an
affine function that a single root can represent, forced subdivision is pure
overhead.

Fit the initial root before subdividing for concurrency. If it fails, grow a
shallow frontier with cheap useful cuts, stopping when available work is enough
for the devices or when cells already pass. Axis-aligned cuts give cheap exact
sampling and bounds. A full Cartesian grid grows exponentially if every
coordinate is divided, whereas a shallow binary partition can create a chosen
number of cells.

An expensive single node also has parallelism over samples, directions and
candidate thresholds. For exceptionally large nodes, distributing statistics
and reducing `p x p` and `p x q` arrays is an alternative to changing the
partition merely to occupy ranks. That introduces collectives inside a node and
should be a separate worker-group mode, not mixed casually with independent
node stealing. Small nodes are unlikely to amortize those collectives.

### 34.6 Sampling candidate: choose the sampler from the geometry

The current choice of one short isotropic chain per new point exposes parallelism,
but pays a warm-up cost repeatedly and can explore narrow directions poorly.
The tests in section 33 show why a fixed dimension-only step count is not a
reliable contract. Existing hit-and-run theory also depends on geometric and
starting conditions; a stationary uniform distribution does not establish that
these finite runs are uniform. [Lovasz and Vempala, Hit-and-Run from a Corner](https://faculty.cc.gatech.edu/~vempala/papers/start.pdf)

#### Candidate G1: exact conditional rejection before MCMC

For a frozen region `R`, find a known enclosing proposal region `B` with an exact
uniform sampler. Draw independently from `B`, retain points satisfying the
canonical inequalities of `R`, and evaluate the expensive oracle only at retained
points. In exact arithmetic,

\[
X\sim U(B),\quad R\subseteq B
\quad\Longrightarrow\quad X\mid X\in R\sim U(R).
\]

Acceptance probability is `rho = volume(R)/volume(B)`, and the expected number
of proposals per accepted point is `1/rho`. This eliminates a mixing question
when the envelope is good. It also separates cheap geometric work from expensive
oracle calls.

Useful envelopes include a retained axis-aligned parent box, a box tightened by
certified coordinate bounds, or a fixed oriented parallelepiped. Bounds derived
only from observed samples are not enclosing bounds and would silently truncate
the target. Optimized coordinate bounds require conservative numerical treatment.
A proposal chosen from a pilot can be frozen before collecting fresh production
draws; pilot acceptance statistics then choose a sampler without being mistaken
for a final validation set.

A 2-D example illustrates the tradeoff. Rotate a rectangle of length 1 and width
`eta` by 45 degrees. Rejection from its tight axis-aligned box has

\[
\rho=\frac{2\eta}{(1+\eta)^2}.
\]

| Width `eta` | Expected proposals per accepted point |
|---:|---:|
| 0.2 | 3.6 |
| 0.02 | 26.01 |
| 0.002 | 251.001 |

An oriented envelope matching the rectangle would accept every proposal. A real
polytope will not generally have such an exact envelope. The example establishes
that orientation matters and that rejection can be either excellent or poor;
it is not a reason to replace every walk with rejection.

Select rejection when its measured geometry cost per accepted point is lower
than the alternative's cost for an adequately explored point. Cap proposal work
and fall back explicitly. An empirical acceptance rate is useful for scheduling,
but is not a proof that a data-derived envelope contains the region.

Axis-aligned descendants of a box deserve a dedicated path: maintain lower and
upper bounds, sample directly, and fold repeated cuts into those bounds. Their
sampling geometry stays `O(d)` regardless of tree depth. This is a substantive
algorithmic advantage of allowing cheap axis splits even when oblique splitting
is available.

#### Candidate G2: frozen rounding plus several persistent chains

When rejection is poor, use a geometry pilot to estimate a regularized transform,
freeze that transform for a production segment, and run several chains per node.
ART already contains a pilot/covariance path and a multi-chain sampler with
incremental slacks. Those ideas are worth porting into batched array kernels;
the scalar ART builder is not the required interface.

If `m` points are requested, `b` warm-up steps are needed and `tau` is a chosen
production spacing, independent restarted chains require roughly `m*b` warm-up
steps. With `C` persistent chains, the analogous count is roughly `C*b + m*tau`.
This is a work model, not an equality of statistical quality. Persistent outputs
are correlated, and `tau` is not automatically a sufficient thinning interval.
Several chains also provide less raw GPU parallelism than one chain per output.

Batch across nodes to recover occupancy. Preserve multiple starts and test
geometry-sensitive observables, boundary support and actual failure-region
discovery. Do not select `C` solely to maximize kernel throughput. Keep the
pilot, production and statistical claims distinct.

For transformed directions, account for dense transform cost and storage. A
full `d x d` transform per node may be excessive at high dimension; diagonal or
low-rank corrections are useful candidates. A coordinate change does not turn
an arbitrary polytope into a box. All original constraints must still be enforced.

#### Candidate G3: global exploration plus local conditional sampling

For integrated error or violation probability under `nu`, draw a fresh global
exploration pool from `nu` and route it through a frozen current tree. This gives
direct information about which leaves queries actually visit and where deployed
errors occur. Local samplers can supply additional design points for difficult
cells.

This can reduce unnecessary MCMC work on broad cells and supports risk-weighted
refinement. It cannot supply every rare leaf efficiently: a cell of probability
`mu` receives only about `M*mu` points from `M` global draws. Use local conditional
sampling when quotas would otherwise require excessive global proposals.

Maintain separate roles for model-design samples, refinement observations and a
final audit. Points used to choose the next split or model are spent. If local
sampling changes the fitting distribution, importance weighting or a deliberately
different design objective must be stated. Do not treat a carefully selected
design set as an iid draw from the deployment measure.

This alternative is attractive when queries occupy a small portion of a large
box. It is inappropriate as a replacement for an everywhere-accuracy requirement
unless additional deterministic bounds are available.

### 34.7 Split-search candidate: reuse bin statistics before adding optimizer work

The current B3 implementation repeats a full weighted Gram construction for each
candidate threshold. For `D` directions, `B` bins, `n` points and `p` features,
the leading Gram work is approximately

\[
O\big(D(B-1)np^2\big),
\]

plus output cross products, candidate solves, sorting and direct finalist checks.
The zeros in a threshold mask do not make a dense matrix multiplication free.

At the defaults `D=6`, `B=16`, this corresponds to 90 full-data left-Gram passes
per node. Twenty B4 iterations require 40 child-fit Gram passes, although B4 has
substantial additional residual, gradient and line-search work. Thus the label
"cheap hard screen" does not itself establish that B3 is cheaper than B4.

For one direction, sort once and partition into disjoint bins. For bin `j`, form

\[
G_j=\Phi_j^T\Phi_j,\quad B_j=\Phi_j^T Y_j,\quad
C_j=\|Y_j\|_F^2,\quad n_j=|\Phi_j|.
\]

Every threshold candidate then uses prefix aggregates on its left and suffix
aggregates on its right. The leading data-product cost becomes

\[
O(Dnp^2),
\]

with additional `O(DBp^2)` aggregation and `O(DBp^3)` solves. Candidate solves
remain; the entire method does not become 15 times faster just because one term
loses that factor. The `p x q` terms and sample ordering also remain.

The same broad reuse principle appears in histogram tree algorithms, but here
each bin holds matrix sufficient statistics for local regression rather than a
few scalar gradient sums. That difference changes memory and numerical costs.
[LightGBM's official description of histogram construction](https://lightgbm.readthedocs.io/en/latest/Features.html)

Implementation options should be compared rather than assumed:

- Form each bin's products from contiguous sorted rows using batched GEMM, then
  scan the small matrix blocks. This avoids materializing an `n x p x p` tensor.
- Use a tiled segmented accumulation kernel when many small bins make GEMM
  inefficient. Its atomic/reduction behavior must satisfy the numerical contract.
- Keep the present direct-product path for small workloads or difficult cases.

Use direct suffix accumulation instead of `total - prefix` when the right child
is small. This avoids the worst complementary-subtraction cancellation. It does
not fix all instability from normal equations or from evaluating a small SSE as
a difference of large terms. Keep direct residual evaluation for finalists, and
retain extra candidates when the screening order is numerically uncertain.

For difficult fits, investigate merging QR factors of bin blocks. If
`Z_j = [Phi_j, Y_j]` has triangular factor `R_j`, a factor for a union of bins can
be obtained by factoring stacked `R_j` blocks. Prefix and suffix merges then
reuse row compression without forming a covariance by subtraction. This follows
the general communication-avoiding QR principle; the proposed threshold-search
integration still needs implementation and validation.
[Demmel et al., communication-optimal QR](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-89.html)

Ridge rows, rank deficiency, residual scoring and feature scaling must have
consistent definitions in that variant. A QR-based screen is not automatically
cheap when each bin has fewer rows than columns. Small well-conditioned affine
problems may favor the simpler Gram method.

#### Spend nonlinear refinement where it can matter

B4 should be a bounded optional improvement to a valid hard candidate. A fixed
20-iteration allowance for every rejected node is unlikely to be optimal across
all oracles and dimensions. Candidate policies can use measured hard-score
improvement, solver condition, previous improvement rate and an explicit split
time allowance to decide whether to continue polishing or to commit the hard
candidate.

The stopping policy must not claim an unseen reduction in future leaves as a
known quantity. Estimate that benefit from held-out experiments, and preserve a
conservative hard incumbent. Compaction of finished optimizer nodes can reduce
wasted GPU work independently of changing the scientific stopping rule.

Split scoring should also match the purpose of the tree. Tolerance-scaled SSE
is a useful smooth objective but can still favor average error over a rare
pointwise failure. For violation-rate targets, test a tail-sensitive or capped
tolerance-normalized loss on separate selection rows. For multi-output models,
the relative scales must agree with deployment tolerances. These are algorithm
variants, not numerically neutral B3 optimizations.

### 34.8 Approximation candidate: enrich a local model before subdividing space

The strongest potential improvement is to stop treating another affine split
as the only useful response to a failed affine fit.

For a smooth scalar function near `c`,

\[
f(c+u)=f(c)+g^T u+\tfrac12u^T H u+O(\|u\|^3).
\]

An affine model repeatedly subdivides to reduce the quadratic remainder. If
most curvature lies in a few directions, storing those directions can be much
cheaper than building enough affine cells to imitate them.

Consider the exact one-dimensional function `f(x)=x^2` on `[-1,1]`. The best
uniform affine approximation on an interval of width `ell` has maximum absolute
error `ell^2/8`. Any partition into `L` intervals has an interval of width at
least `2/L`; consequently,

\[
\|f-\hat f_{\mathrm{piecewise\ affine}}\|_\infty
\ge \frac{1}{2L^2}.
\]

Absolute tolerance `1e-4` therefore needs at least 71 affine leaves, even with
optimal cuts and coefficients. One quadratic model represents the function
exactly. This is an approximation-class difference that no Morton layout,
work-stealing policy or more accurate oblique optimizer can remove. The example
uses absolute error deliberately; pure relative error at a zero requires its
own floor or absolute tolerance.

A high-dimensional candidate model is

\[
\hat f_j(x)=b_j+a_j^T u+
\sum_{k=1}^{r}\gamma_{jk}(v_k^T u)^2,\qquad u=x-c.
\]

With fixed candidate directions `v_k`, it is linear in the fitted coefficients
and has `p=d+1+r` features. Shared directions cost `dr` stored values; evaluation
cost is approximately `O(dq+dr+rq)`. It does not require all `d(d+1)/2` quadratic
features. A rank-one quadratic ridge function is represented exactly if its
direction is supplied accurately. A general full-rank quadratic may not benefit
from a small `r`, and vector-output curvature may require different or additional
directions.

Directions could come from existing Hessian information, gradient differences,
or a bounded residual-based search. Direction estimation is real work and must
be counted. Computing and storing a full dense Hessian merely to discard most
of it may defeat the purpose at high dimension.

A useful decision sequence is:

```text
fit and test affine model
if it fails:
    try a small, bounded set of enriched features on training data
    compare richer model with a hard split using separate selection evidence
    top up if the richer feature space lacks support
    validate the chosen deployed candidate with fresh data
    keep the region, or commit the split
```

Under an ART-like budget `n=kappa*p`, adding `r` features requires approximately
`kappa*r` extra target rows. Creating two affine children from one target-sized
parent requires approximately `kappa*(d+1)` additional rows. This is a favorable
budget comparison when `r` is small, but it assumes adequate independent support,
reuse without duplication, and a valid enriched model. It excludes validation,
centroid and direction-estimation calls.

This is an `h/p` adaptation experiment: refine the region or increase the local
approximation space. A dense quadratic default is still a poor choice at high
dimension. Limit enrichment ranks, check conditioning, and favor splitting near
discontinuities or sharp interfaces where a smooth polynomial is inappropriate.

The present registries and leaf-column declarations can support new models.
However, `RegionPlan` currently sets `p_eff = d + 1`; registering an enriched
model alone would not update its sample target or support policy. The plan needs
the model's declared feature/support requirements, separately from the affine
features used by the split screen. Learned curvature directions also consume
statistical information: `p=d+1+r` counts linear coefficients with directions
fixed, not all the freedom used to select those directions. Independent
selection/validation evidence remains necessary.

Heterogeneous model classes within one tree need an additional representation:
a model-type tag and per-type parameter stores, followed by query grouping for
evaluation. That is a concrete extension, not something the current homogeneous
`LocalModel` artifact already supplies. First test a fixed small enriched class
for all leaves, then heterogeneous selection if the benefit warrants the added
storage and dispatch complexity.

Post-build subtree compression is another experiment: replace a completed
subtree by a richer model only if independent validation supports it. It can
reduce inference cost without changing how training discovers difficult regions,
but does not refund the oracle work already spent. The final compressed model
needs its own untouched audit.

### 34.9 Geometry-aware and query-aware split decisions

The least empirical child SSE is not necessarily the least total future cost.
A cut can improve the fitted residual while creating a thin child that is
expensive to sample, a deep high-probability branch, or a badly conditioned local
design. Conversely, a small difficult cap can deserve refinement even when it
contains few inherited points.

Keep the hard scientific constraints separate from economic preferences. A
candidate must satisfy the geometric predicate and support checks. Among valid
candidates, an experimental objective can trade estimated error against sampling,
oracle and prediction cost:

\[
\widehat{\mathcal R}_{\mathrm{children}}
+\lambda_t\widehat{\Delta T}_{\mathrm{build}}
+\lambda_q\widehat{\Delta T}_{\mathrm{queries}}
+\lambda_m\Delta B_{\mathrm{model}}.
\]

The coefficients have units and should encode a declared resource preference or
be calibrated on training workloads. An arbitrary sum of SSE, seconds and
condition numbers would be hard to interpret and easy to tune to a benchmark.

For a frozen tree, expected routing depth obeys a useful identity:

\[
\mathbb E_\nu[h(X)]
=\sum_{v\ \mathrm{internal}}\Pr_\nu[X\text{ visits }v].
\]

Splitting a terminal cell of query mass `mu_v` therefore adds `mu_v` expected
plane tests per query. This is a better inference-cost signal than its leaf
count alone. Node-key intervals and inherited sample counts are not substitutes
for `mu_v` under adaptive region top-up.

For violation probability, the contribution of a leaf is
`mu_v * P[e(X) >= 1 | X in R_v]`. A priority policy can refine the largest
estimated contributions per unit cost, with exploration and uncertainty terms
to avoid neglecting unseen failures. For a strict everywhere target, low mass
does not justify ignoring a cell. The policy must change with the objective.

A global best-first priority queue is not required. Coarse priority bands and
compatible GPU cohorts can approximate that policy while preserving parallelism.
If there is a global oracle budget, allocate explicit budget permits to ranks
and account for in-flight calls. With schedule-dependent permits, the resulting
tree may depend on rank timing; deterministic allocation rounds are needed if
rank-independent budgeted construction is a requirement.

### 34.10 Inference should be compiled separately from training

ART already predicts by traversing split predicates and evaluating a terminal
model. The current region mode correctly makes that a first-class predictor.
AFINNI's coarse kNN followed by `Q` reranking implements a different model-selection
rule. It should remain available for the legacy method and controlled same-model
comparisons, rather than being inserted automatically after tree routing.

The final model should be compiled from immutable node records into a layout
chosen for the expected query workload:

- For small query batches, fused per-query traversal avoids repeated host control
  and temporary active-index arrays.
- For large batches, route through a shallow prefix, group queries by subtree,
  and process compact subtree blocks. Restore original query order afterward.
- For heterogeneous leaf models, group reached leaves by model type before
  evaluating coefficients.
- For models too large to replicate, replicate a shallow router and shard lower
  subtrees. Account for query-routing communication and hot-subtree imbalance.

There is a useful tradeoff inside a small tree block. Speculatively evaluating all
planes in `b` binary levels costs up to `2^b-1` plane evaluations per query instead
of `b` along its path, but exposes regular arithmetic. It may pay for a very
shallow block on some hardware and fail badly for larger blocks or high dimension.
Compare it on frozen trees; do not assume fewer branches implies less work.

Spatial query sorting is optional. Subtree grouping uses the actual predicates
and remains meaningful in high dimension, unlike a coarse low-bit Morton code.
Packing a frequently visited subtree contiguously can help locality without
changing a single scientific decision.

Dropping `Q` for routed-only affine models removes `O(Ld^2)` coefficient storage.
That is a stronger asymptotic saving than changing a 64-bit node ID to a smaller
index. However, hard routing still permits jumps between leaf models. If a
downstream ODE solver or differentiable application requires continuity, neither
this layout nor a richer local polynomial supplies it automatically. Use the
separate continuity-constrained representation experiments in section 28.

### 34.11 Which changes are most likely to win?

| Workload | Leading candidate | Main reason | Main failure mode |
|---|---|---|---|
| Cheap oracle, many small affine nodes | Resident cohorts and shared-bin B3 | Transfer, allocation and repeated statistics can dominate | Too many tiny buckets or retained work can hurt occupancy/stealing |
| Expensive smooth oracle, low-rank curvature | Bounded local enrichment, then splitting | Avoids paying for many affine leaves | Wrong curvature directions, poor support or overfitting |
| Broad axis-aligned cells | Direct box or conditional rejection sampling | Exact independent draws with little geometric work | Oblique thin cells can make acceptance collapse |
| Thin rotated polytopes | Frozen rounding and multi-chain sampling; oriented envelopes where available | Addresses anisotropy rather than only increasing step count | Pilot misses geometry; transforms and constraints become expensive |
| Strongly nonuniform query measure | Global exploration plus risk-weighted local refinement | Targets deployed risk instead of equal effort per leaf | Rare failures are missed without exploration and honest uncertainty |
| Very large host frontier | Shared ancestry, bounded subtree blocks and explicit spill | Reduces duplicate geometry and resident queue growth | Disk latency or packet assembly dominates |
| Large inference workload | Fewer/richer leaves and compiled subtree routing | Reduces lifetime cost per query | More expensive leaf evaluation can erase the saving |

There is no evidence yet that one row dominates all others. The question should
be which combination meets a declared accuracy requirement at the lowest total
cost on the intended oracle and hardware.

Three complete configurations make useful end-to-end comparisons after the
individual experiments below:

1. **Reference region tree.** Keep the current affine model, repaired sampler,
   node budgets, full-path packets and host-backed executor. Run B3 and B4 as
   separate baselines. This establishes whether additional complexity is needed.
2. **Execution-focused region tree.** Preserve the reference node inputs and
   decisions, but use shared ancestry, bounded resident descendants and compiled
   inference. Add shared-bin B3 only after its numerical differences are
   characterized. This tests the value of better data movement and reuse.
3. **Adaptive model and geometry tree.** Fit the root before creating concurrency;
   use direct box/rejection sampling where valid, rounded chains otherwise; try
   bounded local enrichment on smooth residuals; use a hard split and optional
   soft refinement when enrichment fails. Keep immutable node records and the
   same ownership protocol. This tests whether fewer, better-supported regions
   beat a faster implementation of many affine regions.

The third is the most interesting research candidate, not a proposed immediate
default. Its selection budgets must be bounded so that unsuccessful enrichment,
envelope construction and soft refinement cannot consume unlimited work at one
node. Only combine components that show a benefit in their separate ablations.

### 34.12 Concrete experiments and decision criteria

Keep the repaired current region mode as the reference. Do not change sampling,
models, splitting and scheduling in one experiment.

| Experiment | Hold fixed | Vary | Evidence needed to keep it |
|---|---|---|---|
| E1: ancestry layout | Tree/jobs, point blocks, numerical kernels | Carried paths, local handles, prefix-deduplicated packets | Lower RSS/MPI bytes or elapsed time without changed node results |
| E2: resident scheduling | Per-node samples, model, split candidates and budgets | Host round trips versus bounded local descendant execution | Lower transferred bytes and makespan, with no starvation or lost steals |
| E3: B3 statistics | Directions, threshold positions, ridge and score definitions | Direct products, bin prefix/suffix, optional QR merges | Stable candidate ranking and useful end-to-end speedup, including packing |
| E4: sampling | Frozen cells and target measure | Exact rejection, current walks, rounded multi-chain walks | Feasibility, support and independent-reference distribution checks per second |
| E5: local enrichment | Domain, initial observations, error requirement | Affine-only versus bounded quadratic enrichment | Lower oracle/leaf/lifecycle cost at the same audited deployed accuracy |
| E6: refinement priority | Budget and target measure | Level order versus cost/risk priority bands | Better error-versus-time and error-versus-oracle curves on untouched tests |
| E7: inference compilation | Exact tree and leaf parameters | Level-wise, fused, grouped subtree traversal | Same leaf IDs/statuses and improved latency/throughput across batch sizes |

Use problem families that separate the hypotheses: affine functions, rank-one
quadratics, full-rank anisotropic and indefinite quadratics, rotated thin regions,
localized smooth features, oblique interfaces, vector outputs, and an actual
application oracle. Include a case where the root already passes; it exposes
parallelism schemes that manufacture unnecessary work. Include concentrated query
distributions separately from the uniform-domain experiment.

For E3, arithmetic reordering need not be bit-identical even when the mathematical
statistics are equivalent. Test ill-conditioned and near-tied candidates, retain
the existing tight numerical tolerances, and report any changed structural
decisions. E1/E2 should preserve the reference scientific algorithm; E4/E5/E6
deliberately change it and require accuracy/budget comparisons instead.

Record measured stage lifetimes, peak device and host bytes, transfer bytes,
oracle counts and batch sizes, sampler proposal/step counts, split iterations,
leaf count, query-weighted depth, terminal statuses and independent error metrics.
Report cold-start and steady-state times separately. A single fast kernel or one
favorable seed does not establish the winning algorithm.

One GPU under WSL can test kernels, transfers, geometry and inference. CPU MPI
tests can exercise ownership and termination. Multi-GPU speedup needs multiple
physical GPUs and appropriate oracle placement; several ranks contending for one
GPU cannot answer that question.

Suggested implementation order is E3 first for a bounded numerical experiment,
then E4 on frozen geometries, and E5 for the highest potential oracle reduction.
Profile before investing in E1/E2: they matter most when path/transfer overhead is
visible. E7 can proceed on frozen models independently. Add E6 only after the
target query measure and global-budget semantics are explicit.

### 34.13 Checks performed and limits of the recommendation

Small independent checks were run in
`/tmp/afinni_architecture_checks.py`, with results in the matching `.jsonl` file.
They support the algebra, not a claim about GPU performance:

- Per-bin prefix/suffix Gram and cross-product reconstruction was compared with
  direct products at dimensions 2, 10 and 40, with scalar and vector outputs.
  Relative Frobenius discrepancies were below `9.2e-16` on these well-conditioned
  random cases. This does not certify ranking stability near a singular fit.
- For the three rotated rectangles, 250,000 proposals each gave acceptance rates
  `0.2779`, `0.03832`, and `0.003852`, consistent with the analytic rates at the
  sampling precision of the experiment.
- Direct evaluation of the minimax affine approximation to `x^2` confirmed
  maximum errors `0.001953125`, `0.0001220703125`, and approximately
  `9.91867e-5` for 16, 64 and 71 equal intervals.
- The shared-prefix count and abstract path-interval examples were checked
  arithmetically.

No implementation files were changed, and the full suites were not repeated for
this document-only extension. Section 33 records the current code's CPU/MPI
verification and the unavailable GPU environment. External references above
support the named Morton, histogram, QR and sampling mechanisms; they do not
establish performance of the proposed combined system.

The architectural recommendation is to keep Approach 3 as the stable interface
between numerical work and distributed execution, use Approach 1's compact
history where repeated ancestry becomes expensive, and use Approach 2's ordering
where it actually improves a local batch. The larger research opportunity is to
reduce the work the tree needs to perform: cheaper supported sampling, reused
split statistics, and an adaptive choice between improving a local model and
creating more regions.
