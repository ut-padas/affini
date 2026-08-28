# Next steps for comparing iterative sampling methods on the Rosenbrock problem

**Status:** Proposed experimental protocol  
**Date:** 2026-08-28  
**Scope:** Outer-loop sampling for iterative reduction of the AFINNI uncovered fraction, $p_{\mathrm{out}}$, on bounded Rosenbrock problems

## Executive summary

The next study should answer one primary question:

> At the same number of new Rosenbrock evaluations, which sampling method reduces the uncovered fraction $p_{\mathrm{out}}$ most effectively while preserving acceptable predictive accuracy inside the reported trust regions?

The fair comparison is therefore not “five iterations versus five iterations.” Different methods can add very different numbers of training points in one iteration. The primary horizontal axis must be cumulative new true-function evaluations. Iteration number remains useful for diagnosing retraining dynamics, but it is not a fair measure of cost by itself.

The recommended study has these features:

1. Use one Rosenbrock definition, domain, tolerance, AFINNI configuration, and initial design within each dimension.
2. Compare samplers using identical per-iteration oracle budgets.
3. Reuse the same initial design and independent evaluation set across methods for each seed.
4. Separate acquisition, calibration, and final evaluation data.
5. Report coverage, calibration, prediction error, oracle efficiency, proposal efficiency, model size, time, and memory.
6. Repeat the experiment over independent seeds and use paired statistical comparisons.
7. Correct the current halo implementation before calling it a thin-shell sampler. The present code generates a scaled Gaussian cloud and clips it to the domain.
8. Reconstruct and specify the neighbor-chord method before including it in the confirmatory comparison. Its earlier implementation is not present in the repository.

The primary decision statistic should be the area under the $p_{\mathrm{out}}$-versus-oracle-budget curve, supplemented by the final-budget $p_{\mathrm{out}}$ and the oracle budget required to reach fixed coverage targets.

## 1. What is being compared

### 1.1 Rosenbrock problem

For dimension $d$, use the generalized Rosenbrock function already implemented in the repository:

\[
f(\mathbf{x})=
\sum_{j=1}^{d-1}
\left[
100\left(x_{j+1}-x_j^2\right)^2+(1-x_j)^2
\right],
\qquad \mathbf{x}\in\Omega=[-2,3]^d.
\]

The function has a curved, narrow valley and strong interactions between neighboring coordinates. This makes it a useful test of whether a sampling rule merely fills volume or learns where the surrogate's local trust regions fail to cover the domain.

For pointwise evaluation and calibration, use the convention already implemented by `compute_error_metrics`:

\[
r(\mathbf{x})=
\frac{|\hat f(\mathbf{x})-f(\mathbf{x})|}
{|f(\mathbf{x})|+\epsilon_f}.
\]

Use the current evaluation convention $\epsilon_f=10^{-12}$ and record it in the run configuration. A floor is necessary because pure relative error is unstable near the Rosenbrock minimum, where $f(\mathbf x)=0$.

The root experiment uses `ERROR_METRIC = "L2"` during model construction. For a candidate cluster $\mathcal C_k$, the acceptance quantities are

\[
E_{\mathrm{rel},k}=
\frac{\|\hat{\mathbf f}_k-\mathbf f_k\|_2}
{\|\mathbf f_k\|_2+\epsilon_f},
\qquad
E_{\mathrm{abs},k}=
\sqrt{\frac{1}{|\mathcal C_k|}
\|\hat{\mathbf f}_k-\mathbf f_k\|_2^2},
\]

and the cluster passes when

\[
E_{\mathrm{rel},k}<\tau
\quad\text{or}\quad
E_{\mathrm{abs},k}<\tau.
\]

This cluster-level fitting criterion is not identical to the pointwise relative-error test $r(\mathbf x)\leq\tau$ used to count good and bad evaluation points. Keep both definitions fixed and report them explicitly. The calibration analysis measures the practical consequence of this mismatch.

### 1.2 What “inside” and “outside” mean

For cluster $k$, let $\boldsymbol\mu_k$ be its center and $T_k\succ0$ its trust-region matrix. Define

\[
q_k(\mathbf{x})=
(\mathbf{x}-\boldsymbol\mu_k)^\mathsf{T}
T_k
(\mathbf{x}-\boldsymbol\mu_k).
\]

The cluster trust region is

\[
\mathcal G_k=\{\mathbf{x}:q_k(\mathbf{x})\leq1\},
\]

and the learned covered set is

\[
\mathcal G=\Omega\cap\bigcup_{k=1}^{N_c}\mathcal G_k.
\]

The empirical uncovered fraction is

\[
\hat p_{\mathrm{out}}=
\frac{1}{N_{\mathrm{test}}}
\sum_{i=1}^{N_{\mathrm{test}}}
\mathbf 1\!\left[\mathbf{x}_i\notin\mathcal G\right].
\]

“Inside” is a geometric classification produced by the fitted model. It should not be described as a guaranteed error bound unless an independent calibration experiment supports that claim.

The current code approximates membership by evaluating only the ten Euclidean-nearest cluster centers. Before the main study, quantify the error of this shortcut against a full-cluster membership calculation on manageable cases. If it is not exact enough, increase the shortlist size or implement an exact spatial index.

### 1.3 Primary scientific endpoint

The objective is constrained coverage optimization:

\[
\min_m\; p_{\mathrm{out},m}(B)
\quad\text{subject to}\quad
\Pr\!\left(r(\mathbf{x})>\tau
\mid \mathbf{x}\in\mathcal G_m\right)\leq\alpha,
\]

where $m$ denotes a sampling method, $B$ is the cumulative number of newly evaluated Rosenbrock points, $\tau$ is the error tolerance, and $\alpha$ is the maximum acceptable failure rate inside the reported trust region.

Recommended confirmatory values are

\[
\tau=0.05,
\qquad
\alpha=0.05.
\]

These values can be changed, but they must be fixed before the confirmatory runs. A secondary tolerance-sensitivity analysis can use $\tau\in\{0.01,0.05,0.10\}$ without mixing those results into the primary comparison.

## 2. Sampling methods

Use short, stable method identifiers in configurations and result files.

### 2.1 `uniform`: unconditional uniform augmentation

Draw $B_t$ points uniformly from $\Omega$, evaluate all of them, append them to the training set, and retrain.

This is the essential baseline. It measures the benefit of targeting the learned coverage boundary relative to simply adding more space-filling data.

For the primary study, use a shared maximin Latin hypercube initial design. Morris and Mitchell's computational-design work motivates designs that combine large interpoint distances with useful one-dimensional projections ([Morris and Mitchell, 1995](https://doi.org/10.1016/0378-3758(94)00035-T)). A scrambled Sobol initial design can be retained as a separate sensitivity study, not alternated between primary runs.

### 2.2 `uniform_gap`: global uniform gap rejection

Draw proposal points uniformly from $\Omega$, calculate their trust-region membership, retain proposals outside $\mathcal G$, and continue proposing until $B_t$ distinct survivors have been collected.

This is the older global gap-filling strategy in `research/p_out_iterative_experiment.py`. It directly targets uncovered volume, but its proposal acceptance rate should deteriorate as $p_{\mathrm{out}}$ becomes small.

Log the empirical proposal acceptance rate

\[
a_t=\frac{N_{\mathrm{survive},t}}{N_{\mathrm{proposed},t}}
\]

and the number of proposals required per new oracle call.

### 2.3 `shell`: normalized ellipsoidal halo

For a chosen source cluster $k$, calculate

\[
T_k=U_k\Lambda_kU_k^\mathsf{T},
\qquad
\mathbf u=\frac{\mathbf z}{\|\mathbf z\|_2},
\qquad
\mathbf z\sim\mathcal N(\mathbf0,I),
\]

and propose

\[
\mathbf x_{\mathrm{cand}}=
\boldsymbol\mu_k+\gamma
U_k\Lambda_k^{-1/2}\mathbf u,
\qquad \gamma>1.
\]

Before domain handling,

\[
q_k(\mathbf x_{\mathrm{cand}})=\gamma^2,
\]

so every proposal lies on a geometrically defined shell outside its source trust ellipsoid. Use $\gamma=1.5$ for historical consistency unless a pilot study selects another value. Any tuning must occur before the confirmatory seeds are evaluated.

Reject and redraw proposals outside $\Omega$. Do not clip coordinates in the primary comparison because clipping creates artificial concentrations on domain faces and destroys the shell geometry.

After generation, discard any point covered by any cluster, remove duplicates, and downselect $B_t$ survivors using the common diversity rule in Section 4.3.

### 2.4 `gaussian`: unnormalized Gaussian inflation

The current root script implements

\[
\mathbf x_{\mathrm{cand}}=
\boldsymbol\mu_k+\gamma
U_k\Lambda_k^{-1/2}\mathbf z,
\qquad
\mathbf z\sim\mathcal N(\mathbf0,I),
\]

without normalizing $\mathbf z$. Its source-cluster trust radius is therefore

\[
\sqrt{q_k(\mathbf x_{\mathrm{cand}})}=\gamma\|\mathbf z\|_2,
\]

not $\gamma$. This is a radially diffuse Gaussian cloud rather than a thin halo.

Use rejection at the domain boundary in the primary comparison so that `shell` and `gaussian` differ only in radial distribution. Retain a separate `gaussian_clip_historical` arm in the reproduction study, not in the main factorial comparison. That arm reproduces the current script's coordinate clipping and shows how much of the result came from boundary handling.

### 2.5 `neighbor_chord`: intercluster chord proposals

The historical results mention “Random Chord, $K=5$ nearest clusters, PPC = 500, noise = 0.3,” but the corresponding outer-loop implementation is not present in the repository. Do not include this label in a formal comparison until its definition has been recovered from an older branch, notebook, or commit, or has been explicitly reconstructed.

The reconstruction specification must answer all of the following:

1. Is the chord drawn between cluster centers, training points, ellipsoid boundary points, or another pair of anchors?
2. How is a partner chosen among the five nearest clusters?
3. What distribution is used for the chord interpolation coefficient?
4. What coordinate system and scale define “noise = 0.3”?
5. Is the proposal filtered against all trust regions?
6. How are out-of-domain points handled?
7. How are survivors deduplicated and downselected?

One acceptable new definition, if the original cannot be recovered, is

\[
\mathbf x_{\mathrm{base}}=
\rho\boldsymbol\mu_k+(1-\rho)\boldsymbol\mu_\ell,
\qquad \rho\sim\operatorname{Uniform}(0,1),
\]

followed by a precisely defined perturbation in standardized local coordinates. If this is adopted, label it as a newly specified method rather than as a reproduction of the old chord result.

### 2.6 Optional `hybrid`: boundary targeting plus model-error targeting

The principal study should remain focused. A hybrid method can be added as a preregistered extension after the four or five core samplers are stable.

For candidate $\mathbf x$, define an outside-boundary margin

\[
m(\mathbf x)=\min_k q_k(\mathbf x)-1
\]

and a local model-disagreement or cross-validation score $v(\mathbf x)$. Rank candidates using a fixed scalarization or Pareto rule that favors small positive $m$ and large $v$.

This separates space exploration from error exploitation. Comparable hybrid logic appears in LOLA-Voronoi, which combines a Voronoi coverage term with a local nonlinearity term ([Crombecq et al., 2011](https://epubs.siam.org/doi/10.1137/090761811)), and in expected-prediction-error adaptive Kriging ([Liu et al., 2017](https://www.sciencedirect.com/science/article/abs/pii/S009813541730234X)). These papers motivate the design principle, not a claim that their uncertainty measures transfer directly to AFINNI.

Active-learning reliability methods provide a second useful analogy. AK-MCS enriches a Kriging model near an estimated limit-state boundary to reduce costly true-function calls ([Echard, Gayton, and Lemaire, 2011](https://doi.org/10.1016/j.strusafe.2011.01.002)). AFINNI's boundary is the learned surface $q_{\min}(\mathbf x)=1$, but AFINNI does not supply the same posterior predictive variance. Import the boundary-targeting principle, not the Kriging-specific acquisition formula.

## 3. Experimental factors

### 3.1 Dimensions

Use

\[
d\in\{2,3,5,8,10\}.
\]

Dimensions 2 and 3 permit direct visualization and exact membership audits. Dimension 5 connects to the existing results. Dimensions 8 and 10 test whether the sampler remains effective as the volume and number of learned clusters grow.

If the full grid is too expensive, preregister $d\in\{2,3,5,8\}$ as the primary grid and $d=10$ as an extension. Do not decide whether to report $d=10$ after seeing its method ranking.

### 3.2 Shared dimension-dependent initialization

Within a dimension and seed, every sampler must start from the exact same initial points, function values, cluster configuration, and random state used to train the initial surrogate.

The academically clean default is to retain the repository's dimension-scaling heuristic from `src/heuristic_parameters.py`:

\[
N_0(d)=\operatorname{round}(5\times10^4\,1.3^d),
\]

\[
K_0(d)=\operatorname{round}(25\,1.3^d),
\qquad
M_{\min}(d)=\operatorname{round}(100\,1.3^d).
\]

Here $N_0$ is the initial true-function design size, $K_0$ is the requested initial cluster count, and $M_{\min}$ is the minimum cluster size parameter. Verify that these expressions still match the intended paper configuration and the current constructor semantics.

Also run one historical reproduction case at $d=5$, $N_0=5000$, and the current root-script settings. Keep this case separate from the confirmatory dimension study.

### 3.3 Replicates and common random numbers

Use at least ten independent top-level seeds for the final study. Five seeds are sufficient for a pilot, but usually too few for a stable method ranking.

For each pair $(d,s)$:

1. Generate one initial design and reuse it for all methods.
2. Generate one independent evaluation design and reuse it for all methods and iterations.
3. Use matched random streams for operations that are logically common across methods.
4. Key random streams by semantic identifiers such as `(seed, dimension, method, iteration, purpose)`, not by MPI rank.

The current rank-dependent seeding means a change in MPI process count changes the numerical experiment. Replace it with a counter-based or keyed random-number scheme before the main runs.

### 3.4 Iterations and budgets

Set the per-iteration true-evaluation budget to

\[
B_t=B(d)=\left\lceil0.1N_0(d)\right\rceil
\]

for ten acquisition iterations. The final cumulative added budget is then approximately $N_0(d)$. This produces a common normalized budget

\[
b_t=\frac{N_{\mathrm{added},1:t}}{N_0(d)}
\]

with checkpoints from 0 to 1.

If this is too expensive, reduce both the batch fraction and the total budget before the confirmatory run. Do not stop a method early merely because its $p_{\mathrm{out}}$ looks good. All methods must be evaluated at the same scheduled budget checkpoints.

Use coverage targets

\[
p_{\mathrm{out}}\in\{0.10,0.05,0.01\}
\]

for time-to-target summaries. A method that does not reach a target within the budget is right-censored and should be reported as “not reached,” not assigned the final budget as though it had succeeded.

## 4. Fairness controls

### 4.1 Primary experiment: fixed oracle budget

Every method adds exactly $B_t$ newly evaluated, unique points at iteration $t$. Candidate generation and filtering may require different numbers of cheap surrogate operations, but the number of expensive Rosenbrock calls is identical.

Generate proposals in deterministic chunks until the batch is full. Set a large finite proposal cap before the pilot, recommended initially as $C_t=1000B_t$. If a method cannot fill its batch before this cap, mark the acquisition as infeasible and do not silently top it up with another sampler. The pilot should establish whether the cap is sufficient for every confirmatory arm.

This experiment estimates data efficiency:

\[
\eta^{\mathrm{oracle}}_t=
\frac{p_{\mathrm{out},t-1}-p_{\mathrm{out},t}}
{B_t}.
\]

### 4.2 Secondary experiment: fixed proposal budget

Give each method the same number $M_t$ of raw proposals, recommended as

\[
M_t=20B_t,
\]

and measure how many useful, unique, uncovered points survive. This estimates search and filtering efficiency. It must not replace the oracle-controlled comparison.

For methods that naturally generate candidates per cluster, allocate a fixed total $M_t$, not a fixed points-per-cluster count. A fixed PPC makes the total proposal count grow with $N_c$, which favors models that happen to contain more clusters and obscures computational scaling.

### 4.3 Common downselection

When a sampler produces more than $B_t$ survivors, downselect with one shared algorithm:

1. Remove exact duplicates and near-duplicates using a fixed normalized distance threshold.
2. Seed the batch with the candidate having the smallest positive boundary margin, or use a seed selected by a fixed keyed random stream.
3. Add points greedily by maximum distance to the union of the current training set and already selected batch points.
4. Continue until $B_t$ points have been selected.

This produces a diverse batch and avoids spending many oracle calls on nearly identical proposals. Sequential maximin ideas are established in computer-experiment design, including batch augmentation based on interpoint distance ([Loeppky, Moore, and Williams, 2010](https://doi.org/10.1016/j.jspi.2009.12.004)) and fully sequential space-filling methods ([Shang and Apley, 2021](https://www.tandfonline.com/doi/abs/10.1080/00224065.2019.1705207)).

### 4.4 Domain-boundary handling

Use rejection and redraw as the primary rule for all local samplers. Log the rejection fraction. Coordinate clipping may be studied as a separate ablation because it changes the proposal distribution and can produce repeated points on faces, edges, and corners.

### 4.5 Internal model augmentation

The random-chord augmentation inside `src/adaptive_clustering.py` is part of surrogate construction, not one of the outer acquisition methods. Hold its settings fixed across all arms. If it is ablated, do so in a separate model-construction experiment.

### 4.6 Full retraining

Continue rebuilding the model from the complete accumulated training set at every iteration. Because clustering and trust ellipsoids are reconstructed, the covered set is not necessarily monotone. A point can move from inside to outside even after more data have been added. This behavior is a result to measure, not an implementation detail to hide.

## 5. Independent data roles

Use three disjoint data roles.

### 5.1 Acquisition data

These are candidate points proposed by a sampler. Only the selected $B_t$ points are evaluated with Rosenbrock and added to training.

### 5.2 Calibration data

Use a fixed holdout set to assess whether the geometric trust score is calibrated to the requested error tolerance. Do not add these points to training and do not use them to compare final methods.

Bin calibration points by a scalar trust score such as

\[
q_{\min}(\mathbf x)=\min_k q_k(\mathbf x)
\]

and report the empirical rate $\Pr(r>\tau)$ in each bin. If a scalar correction to the trust threshold is learned, fit it only on the calibration data and freeze it before evaluating the test set.

Distribution-free risk-control methods provide a useful model for separating calibration from evaluation and for stating finite-sample risk targets under exchangeability ([Bates et al., 2021](https://arxiv.org/abs/2101.02703); [Angelopoulos and Bates, 2021](https://arxiv.org/abs/2107.07511)). Direct conformal coverage is not automatic here because the fitted trust geometry and adaptive sampling process require careful exchangeability checks.

### 5.3 Evaluation data

Use an evaluation set that is independent of training, acquisition, and calibration. Reuse it across methods and iterations for a given seed so that method differences are paired point by point.

Recommended default:

\[
N_{\mathrm{test}}=2^{17}=131{,}072
\]

points from an independently scrambled Sobol design for each top-level seed. Scrambled digital nets preserve uniform marginals, provide collective equidistribution, and permit independent randomized replicates, which is useful for estimating volume-like quantities such as $p_{\mathrm{out}}$ ([Owen, 1997](https://epubs.siam.org/doi/10.1137/S0036142994277468)). Use independent scrambles across top-level seeds. If a simple binomial interpretation is required, use a large IID uniform test set instead and increase its size until Monte Carlo uncertainty is negligible relative to between-seed variation.

Never append evaluation points to the training set.

## 6. Metrics

### 6.1 Coverage metrics

Record at every checkpoint:

\[
\hat p_{\mathrm{out}}=\frac{N_{\mathrm{out}}}{N_{\mathrm{test}}},
\qquad
\hat p_{\mathrm{in}}=1-\hat p_{\mathrm{out}}.
\]

The primary summary across the normalized oracle budget $b\in[0,1]$ is

\[
\operatorname{CAUC}_{m,d,s}=
\int_0^1 \hat p_{\mathrm{out},m,d,s}(b)\,db,
\]

computed by the trapezoidal rule at scheduled checkpoints. Lower CAUC is better.

Also record:

- final-budget $p_{\mathrm{out}}$;
- budget to each coverage target;
- largest iteration-to-iteration increase in $p_{\mathrm{out}}$;
- fractions of fixed test points undergoing each transition: in-to-in, in-to-out, out-to-in, and out-to-out.

The out-to-in transition is the desired coverage gain. The in-to-out transition measures geometric instability caused by full retraining.

### 6.2 Accuracy and calibration metrics

Report all of the following:

\[
e_{L_2}=\frac{\|\hat{\mathbf f}-\mathbf f\|_2}{\|\mathbf f\|_2},
\]

\[
e_{\mathrm{in}}=
\frac{\|\hat{\mathbf f}_{\mathrm{in}}-\mathbf f_{\mathrm{in}}\|_2}
{\|\mathbf f_{\mathrm{in}}\|_2},
\qquad
e_{\mathrm{out}}=
\frac{\|\hat{\mathbf f}_{\mathrm{out}}-\mathbf f_{\mathrm{out}}\|_2}
{\|\mathbf f_{\mathrm{out}}\|_2},
\]

and

\[
\hat p_{\mathrm{bad}\mid\mathrm{in}}=
\frac{N_{\mathrm{in},\,r>\tau}}{N_{\mathrm{in}}}.
\]

Do not report $N_{\mathrm{in,bad}}/N_{\mathrm{test}}$ as though it were the conditional trust-region failure rate. Store both quantities with unambiguous names.

Because aggregate $L_2$ error can hide local failures, also report the median, 90th, 95th, 99th, and maximum pointwise relative error separately for inside and outside points.

### 6.3 Sampling-efficiency metrics

Record:

- raw proposals;
- domain rejections;
- proposals covered by another cluster;
- duplicate rejections;
- candidates surviving all filters;
- survivor fraction;
- proposals per selected point;
- boundary-margin distribution before and after downselection;
- distance-to-training-set distribution for selected points;
- unique selected points per source cluster.

For local samplers, store both the source-cluster trust radius $q_k$ and the global minimum trust radius $q_{\min}$. A point can be outside its source ellipsoid but inside another ellipsoid.

### 6.4 Computational metrics

Record wall time and peak memory separately for:

- candidate generation;
- trust-region filtering;
- diversity downselection;
- Rosenbrock evaluation;
- surrogate retraining;
- evaluation-set prediction and membership classification.

Also record $N_c$, training-set size before retraining, points added, training-set size after acquisition, MPI rank count, host and accelerator information, commit hash, and configuration hash.

## 7. Statistical analysis

### 7.1 Unit of replication

The top-level seed is the experimental replicate. The many test points within a seed provide precise conditional estimates, but they are not independent surrogate-model replicates.

### 7.2 Paired comparisons

For each dimension and pair of methods $A,B$, compute seed-level paired differences in:

- CAUC;
- final $p_{\mathrm{out}}$;
- final $p_{\mathrm{bad}\mid\mathrm{in}}$;
- budget to target, when reached;
- total wall time;
- peak memory.

Report the mean or median paired difference, a 95% paired bootstrap confidence interval over seeds, and the number of seeds favoring each method. Preserve direction in every table, for example

\[
\Delta\operatorname{CAUC}=\operatorname{CAUC}_A-\operatorname{CAUC}_B.
\]

Negative values then favor method $A$.

Use a Holm correction for the preregistered pairwise comparisons within each dimension. Distinguish confirmatory comparisons from exploratory plots and post hoc hypotheses.

For coverage targets, report both the fraction of seeds that reach the target and the restricted mean oracle budget up to the common maximum budget. Do not compare time-to-target using only successful runs because that excludes the hardest failures and biases the comparison.

### 7.3 Across-dimension model

Use a repeated-measures or mixed-effects analysis as a secondary summary:

\[
y_{m,d,s}=
\beta_0+\beta_m+g(d)+\beta_{m\times d}+u_s+\varepsilon_{m,d,s},
\]

where $y$ is CAUC or final $p_{\mathrm{out}}$, $g(d)$ is a prespecified dimension effect, and $u_s$ is a seed effect. Method-by-dimension interaction is scientifically important because a sampler that works in 2D may fail when trust regions become numerous and anisotropic.

Do not pool raw $p_{\mathrm{out}}$ values across dimensions without modeling dimension. The underlying difficulty and initialization size differ substantially.

### 7.4 Practical significance

Define practical thresholds before analysis. Suggested examples are:

- at least 1 percentage point lower final $p_{\mathrm{out}}$;
- at least 10% lower CAUC;
- at least 20% fewer oracle calls to reach $p_{\mathrm{out}}=0.05$;
- no increase above the calibration limit $\alpha=0.05$.

A tiny statistically detectable difference on a large test set should not be presented as a meaningful sampler improvement.

## 8. Required implementation changes

### 8.1 Introduce a sampler interface

Move each outer sampler behind one interface with no access to true Rosenbrock values during proposal generation:

```python
class Sampler(Protocol):
    def propose(
        self,
        model_state,
        domain,
        n_proposals: int,
        rng,
    ) -> ProposalBatch:
        ...
```

`ProposalBatch` should retain diagnostic arrays such as source cluster, pre-boundary coordinates, domain-rejection flag, source $q_k$, global $q_{\min}$, and filter outcome.

### 8.2 Separate the experiment stages

Implement explicit functions for:

1. `generate_proposals`;
2. `apply_domain_rule`;
3. `classify_global_membership`;
4. `deduplicate`;
5. `downselect_diverse_batch`;
6. `evaluate_oracle`;
7. `retrain_model`;
8. `evaluate_checkpoint`;
9. `write_checkpoint`.

This separation makes it possible to compare proposal distributions, filtering approximations, and downselection without rewriting the whole driver.

### 8.3 Correct row alignment

Each metrics row must state which model it describes. Use at least these fields:

```text
n_train_before
n_proposed
n_domain_rejected
n_covered_rejected
n_duplicate_rejected
n_survived
n_selected
n_added
n_train_after
model_fit_used_for_metrics
```

The current halo CSV places the post-acquisition training size beside pre-acquisition model metrics. That makes iteration-level interpretation ambiguous.

Recommended convention: checkpoint $t$ describes the model after exactly $t$ acquisitions and retrainings. The checkpoint-0 row describes the initial model. Proposal diagnostics for the transition $t\rightarrow t+1$ belong in a separate acquisition table keyed by `(run_id, iteration_from)`.

### 8.4 Make membership auditable

Provide two implementations:

- `membership_exact`, which evaluates all clusters in blocks;
- `membership_approx`, which uses the nearest-center shortlist.

For $d\in\{2,3,5\}$ and selected checkpoints, compare their classifications on the same evaluation set. Report false-inside and false-outside rates. Increase shortlist $K$ over a grid such as $10,25,50,100$, and choose the smallest value meeting a prespecified discrepancy tolerance.

### 8.5 Remove rank-dependent randomness

Use a reproducible seed derivation such as

```text
hash(global_seed, dimension, method, iteration, purpose, global_item_id)
```

so results do not change when the number of MPI ranks changes. Add a test that runs a small proposal problem with one and two ranks and compares the globally assembled candidates and summary metrics.

## 9. Suggested configuration and data layout

### 9.1 Configuration

Store every scientific choice in a versioned configuration file. For example:

```yaml
study: rosenbrock_sampling_v1
problem:
  name: rosenbrock
  domain: [-2.0, 3.0]
  relative_error_floor: 1.0e-12
initial_design:
  method: maximin_latin_hypercube
dimensions: [2, 3, 5, 8, 10]
tolerance: 0.05
calibration_failure_limit: 0.05
seeds: [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
samplers:
  - uniform
  - uniform_gap
  - shell
  - gaussian
iterations: 10
batch_fraction_of_initial: 0.10
proposal_multiplier: 20
halo_gamma: 1.5
domain_rule: reject
membership:
  mode: validated_approx
  nearest_clusters: 50
downselection:
  method: greedy_maximin
evaluation:
  design: scrambled_sobol
  n_points: 131072
calibration:
  design: independent_scrambled_sobol
  n_points: 32768
```

Do not include `neighbor_chord` until its specification is complete.

### 9.2 Output hierarchy

Use a self-describing layout:

```text
results/rosenbrock_sampling_v1/
  manifest.json
  configs/
  dimension_02/
    seed_1001/
      shared_initial_design.npz
      shared_calibration_design.npz
      shared_evaluation_design.npz
      uniform/
        checkpoints.parquet
        acquisitions.parquet
        transitions.parquet
        run.json
      shell/
        ...
  analysis/
    paired_effects.parquet
    figures/
    tables/
```

Prefer Parquet or another typed table format for detailed data. A compact CSV export can be generated for the paper.

### 9.3 Minimum run metadata

Every run must preserve:

- Git commit and dirty-worktree flag;
- complete resolved configuration;
- dependency versions;
- compiler, CUDA, MPI, CPU, and GPU information;
- start and end timestamps;
- top-level and derived seeds;
- MPI rank count;
- exception and incomplete-run status;
- hashes of shared initial, calibration, and evaluation designs.

## 10. Validation tests before expensive runs

### 10.1 Geometry tests

For synthetic positive-definite $T_k$:

- normalized shell proposals satisfy $q_k=\gamma^2$ to numerical tolerance before domain handling;
- Gaussian proposals have $q_k/\gamma^2\sim\chi_d^2$;
- every retained proposal satisfies $q_{\min}>1$ under exact membership;
- rejection handling never changes an accepted coordinate;
- clipping appears only in the explicitly named historical arm.

### 10.2 Accounting tests

At every acquisition:

\[
N_{\mathrm{proposed}}=
N_{\mathrm{domain\ rejected}}+
N_{\mathrm{membership\ checked}},
\]

and subsequent filter counts must reconcile exactly. Also require

\[
N_{\mathrm{train,after}}=
N_{\mathrm{train,before}}+N_{\mathrm{added}}.
\]

### 10.3 Leakage tests

Assert that no calibration or evaluation point identifier appears in the training set. Assert that true $f(\mathbf x)$ values are not available to `propose`, `filter`, or `downselect`.

### 10.4 Reproducibility tests

- identical configuration and seed produce identical selected points;
- changing only the MPI rank count does not change global results;
- restarting from a saved checkpoint reproduces an uninterrupted run;
- method order in the driver does not change any run.

### 10.5 Two-dimensional visual audit

For $d=2$, plot every stage on the same checkpoint:

1. current trust ellipsoids;
2. raw proposals;
3. domain rejections;
4. overlap rejections;
5. survivors;
6. selected diverse batch;
7. test-point coverage transitions after retraining.

This is the fastest way to detect incorrect filtering, boundary artifacts, duplicate concentrations, and a sampler that is targeting the wrong side of a trust boundary.

## 11. Staged execution plan

### Stage 0: freeze definitions

- Confirm the Rosenbrock formula, domain, relative-error floor, tolerance, and calibration limit.
- Decide whether the paper's dimension scaling is the final initialization rule.
- Recover or explicitly redefine the neighbor-chord sampler.
- Write a short preregistration listing primary endpoints and planned pairwise comparisons.

**Exit criterion:** every method has a mathematical definition and every configurable choice has one frozen value or a documented pilot-selection rule.

### Stage 1: refactor and test

- Introduce the sampler interface and separated experiment stages.
- Correct shell normalization and make clipping an explicit ablation.
- Implement exact membership and validate the approximate search.
- Add fixed-budget downselection and complete accounting.
- Replace rank-dependent random seeds.

**Exit criterion:** all geometry, accounting, leakage, and reproducibility tests pass.

### Stage 2: smoke study

Run $d=2$, two seeds, four samplers, two iterations, and small budgets.

Inspect the stage-by-stage plots manually. Confirm that all methods add the same number of points and that no evaluation-set data enter training.

**Exit criterion:** the complete pipeline produces valid tables and plots with reconciled counts.

### Stage 3: pilot study

Run $d\in\{2,5,8\}$ with five seeds and 30% of the intended final budget.

Use the pilot only to answer prespecified operational questions:

- Is $\gamma=1.5$ viable across dimensions?
- Is proposal multiplier 20 sufficient to obtain full batches?
- What nearest-cluster shortlist size matches exact membership?
- Are memory and wall-time limits feasible for the proposed full grid?
- Does any method routinely fail to fill its oracle batch?

Do not select a method because it happens to win the pilot. If a hyperparameter is selected, document the selection rule and then freeze it.

**Exit criterion:** final configuration and compute budget are fixed.

### Stage 4: historical reproduction

Reproduce the existing $d=5$, $N_0=5000$ halo case with:

- `gaussian_clip_historical`;
- `gaussian` with rejection;
- `shell` with rejection.

This isolates the effects of vector normalization and clipping. Compare the reconstructed metrics with the previously recorded table, allowing for identified seed and code-version differences.

**Exit criterion:** historical behavior is understood and discrepancies are documented.

### Stage 5: confirmatory study

Run all frozen methods, dimensions, and ten seeds. Schedule by `(dimension, seed, method)` and preserve shared designs before launching method jobs.

Monitor only for failures, numerical errors, and accounting violations. Do not change scientific parameters after results start accumulating. Failed jobs should resume from checkpoints under the same configuration.

**Exit criterion:** every planned run is complete or has a documented technical failure status.

### Stage 6: analysis and paper integration

Generate all figures and tables from one version-controlled analysis script. Produce machine-readable effect tables before writing conclusions.

The paper should distinguish:

- historical reproduction;
- confirmatory sampler comparison;
- optional hybrid or tolerance-sensitivity extensions;
- implementation ablations such as clipping and membership shortlist size.

## 12. Required figures and tables

### 12.1 Main figures

1. $p_{\mathrm{out}}$ versus normalized cumulative oracle budget, faceted by dimension, with one curve per method and uncertainty bands over seeds.
2. CAUC by method and dimension, shown as paired seed points plus confidence intervals.
3. Oracle budget to reach 10%, 5%, and 1% uncovered volume, including the fraction of runs that never reach each target.
4. Conditional inside failure rate $p_{\mathrm{bad}\mid\mathrm{in}}$ versus oracle budget with the $\alpha=0.05$ limit marked.
5. Pareto plot of final $p_{\mathrm{out}}$ against total wall time and peak memory.
6. Proposal survival and proposals-per-selected-point versus iteration.
7. Heat map of in-to-out and out-to-in transition rates.

### 12.2 Diagnostic figures

- 2D acquisition-stage geometry for each sampler;
- source $q_k$ and global $q_{\min}$ distributions;
- fraction of local proposals rejected at the domain boundary;
- selected-point coordinate marginals and boundary-face frequencies;
- exact-versus-approximate membership confusion matrices;
- pointwise relative-error quantiles inside and outside;
- calibration plot of empirical error-failure probability versus trust score;
- cluster count and retraining time versus training size.

### 12.3 Main tables

- frozen problem and algorithm parameters;
- final-budget metrics by method and dimension;
- paired CAUC effects with 95% intervals and Holm-adjusted $p$-values;
- oracle budgets to coverage targets;
- compute cost and peak memory;
- proposal-accounting summary;
- historical reproduction and ablation results.

## 13. Interpretation rules

Use these rules when drawing conclusions:

1. A reduction in $p_{\mathrm{out}}$ is useful only if the inside calibration constraint remains acceptable.
2. Iteration count is descriptive. Oracle evaluations are the primary scientific cost.
3. Candidate-generation time is not free. It is a secondary computational cost that can reverse a data-efficiency advantage.
4. A method that wins only in 2D should not be described as generally superior.
5. A method that creates more clusters may reduce $p_{\mathrm{out}}$ while making evaluation and storage much more expensive. Report both effects.
6. Nonmonotone coverage is expected under full reclustering. Report it and analyze transitions.
7. The term “guaranteed region” should be reserved for a region with demonstrated calibration or a proved bound. Otherwise use “accepted trust region” or “empirically covered region.”
8. Historical `gaussian_clip_historical` results do not establish the performance of normalized halo sampling.

## 14. Immediate action checklist

- [ ] Confirm the final Rosenbrock error definition and $\epsilon_f$.
- [ ] Confirm $\tau=0.05$ and $\alpha=0.05$, or replace them before the pilot.
- [ ] Recover or redefine the outer neighbor-chord algorithm.
- [ ] Refactor samplers behind a shared proposal interface.
- [ ] Implement true normalized-shell proposals.
- [ ] Replace clipping with rejection in primary local samplers.
- [ ] Add common deduplication and greedy-maximin batch selection.
- [ ] Add exact membership and audit the nearest-center shortcut.
- [ ] Separate acquisition, calibration, and evaluation sets.
- [ ] Correct pre-acquisition and post-acquisition metric alignment.
- [ ] Replace rank-dependent seeds with keyed random streams.
- [ ] Add checkpointing and complete run manifests.
- [ ] Run the 2D visual smoke study.
- [ ] Run the five-seed pilot and freeze the final configuration.
- [ ] Reproduce the historical 5D result as a separate ablation.
- [ ] Run the confirmatory paired study.
- [ ] Generate all paper figures and effect tables from version-controlled scripts.

## 15. Recommended final claim structure

If the evidence supports it, the strongest defensible conclusion will have this form:

> Under a fixed budget of new Rosenbrock evaluations, sampler $A$ achieved a lower uncovered-volume CAUC than sampler $B$ in dimensions $d=\ldots$, while maintaining an independently measured inside failure rate below $\alpha$. The advantage did or did not persist after accounting for proposal-generation time, model size, and dimension.

This formulation ties the claim to the actual experimental controls. It avoids equating iteration count with cost, coverage with accuracy, or empirical trust geometry with a proof of correctness.

## References

- Angelopoulos, A. N., and Bates, S. (2021). [A gentle introduction to conformal prediction and distribution-free uncertainty quantification](https://arxiv.org/abs/2107.07511).
- Bates, S., Angelopoulos, A. N., Lei, L., Malik, J., and Jordan, M. I. (2021). [Distribution-free, risk-controlling prediction sets](https://arxiv.org/abs/2101.02703).
- Crombecq, K., Laermans, E., and Dhaene, T. (2011). [A novel hybrid sequential design strategy for global surrogate modeling of computer experiments](https://epubs.siam.org/doi/10.1137/090761811). *SIAM Journal on Scientific Computing*, 33(4), 1948–1974.
- Echard, B., Gayton, N., and Lemaire, M. (2011). [AK-MCS: An active learning reliability method combining Kriging and Monte Carlo simulation](https://doi.org/10.1016/j.strusafe.2011.01.002). *Structural Safety*, 33(2), 145–154.
- Liu, H., Cai, J., and Ong, Y.-S. (2017). [An adaptive sampling approach for Kriging metamodeling by maximizing expected prediction error](https://www.sciencedirect.com/science/article/abs/pii/S009813541730234X). *Computers & Chemical Engineering*, 106, 171–182.
- Loeppky, J. L., Moore, L. M., and Williams, B. J. (2010). [Batch sequential designs for computer experiments](https://doi.org/10.1016/j.jspi.2009.12.004). *Journal of Statistical Planning and Inference*, 140(6), 1452–1464.
- Morris, M. D., and Mitchell, T. J. (1995). [Exploratory designs for computational experiments](https://doi.org/10.1016/0378-3758(94)00035-T). *Journal of Statistical Planning and Inference*, 43(3), 381–402.
- Owen, A. B. (1997). [Monte Carlo variance of scrambled net quadrature](https://epubs.siam.org/doi/10.1137/S0036142994277468). *SIAM Journal on Numerical Analysis*, 34(5), 1884–1910.
- Shang, B., and Apley, D. W. (2021). [Fully-sequential space-filling design algorithms for computer experiments](https://www.tandfonline.com/doi/abs/10.1080/00224065.2019.1705207). *Journal of Quality Technology*, 53(2), 173–196.
