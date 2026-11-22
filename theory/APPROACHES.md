# Approaches to surrogates of stiff step maps: polynomial, learned and other nonlinear bases

This document asks whether the project should explore nonlinear, learned bases (neural networks, latent dynamics
models, kernels, rational functions and others) beside its polynomial schemes, and how each family compares with what
the project has, on the criteria that matter for the target: surrogates of stiff ODE step maps whose **maximum
pointwise error over every state visited** stays within a tolerance, in rollouts and under operator splitting, ending
with astrochemical networks of 100+ species (KIDA's 578) coupled to MHD.

- **Scope.** What the panel asks of a new basis (section 1), the criteria (section 2) and one comparison table
  (section 3); the project's schemes (section 4) and every family surveyed (sections 5–7); the memo's four-factor
  decomposition (`theory/MEMO.md`, section E) applied to each family (section 8); hybrid designs (section 9); and a
  ranking with the cheapest decisive experiment for each (section 10).
- **Sources.** Every citation is an ID from `theory/LITERATURE.md` (`TH-*`, and the `NN-*` section added with this
  document) or `docs/LITERATURE.md` (`I-*` to `VIII-*`). The verification status of each source (read in full or in
  part, abstract only, search summary, from memory) is recorded there. In this document, "(memory)" marks a work or a
  claim cited without opening the source; numbers are quoted only from sources whose text or abstract was read.
- **Evidence from the project.** Panel results are those of `theory/panel/data/r4_generic.json`,
  `r5_data_scaling.json` and `p2_tradeoff.json`, as summarised in the commits `40fa52f` and `4b7ffaa`. No experiment
  was run for this document; every statement about how a family would do on the panel is a prediction, labelled so.
- **Search date.** 2026-11-21.

---

## 1. What the panel says a new basis must fix

The generic recipe (`theory/panel/generic.py`: a chart of validated dimension, polynomial fits of degree up to 8 in the
chart, refusal by Proposition 5.1's Lebesgue bound against `tol/(2 e_v)`, solver fallback) was run on all 37 step maps at
tolerance 0.01 (dex for chemistry).

1. **Where trajectories cover the points of use, polynomial chart fits work.** `allen_cahn_d2`, `bruss_d2`,
   `lorenz96_d4`, `chemakzo` and `hires` are served with a solver share below 50% and hold the tolerance on all 40
   test rollouts (2e-6 to 5e-4). Effective speed-ups (Corollary 5.3): 2,311x (`bruss_d2`), 876x (`allen_cahn_d2`),
   79x (`chemakzo`), 10x (`lorenz96_d4`), 3.1x (`hires`, 31% refused), 1.4x (`allen_cahn_d3`, 73% refused).
2. **Data-limited problems are fixed by data, not by the basis.** 400 fresh one-step starts from each problem's initial
   law make `lorenz96_d5`, `burgers_d10` and `burgers_d20` served (rollouts 5e-6 to 6e-5, 0–20% refused); the gain
   saturates by 400 starts (R5).
3. **Front-dominated, box-filling problems are approximation-limited.** On the Fisher–KPP and Allen–Cahn ladders the
   held-out error falls with data and then stalls (`fisher_kpp_d10` 0.09, `fisher_kpp_d3` 0.017) at degree at most 8,
   with 16 times more data (R5). This is factor 1 of section E: the space, not the samples.
4. **Mixing (operator-split inputs) defeats the gate.** On `hires` with mixing weight 0.01, accepted states err up to
   0.38; the chart-coordinate Lebesgue bound cannot see displacements normal to the visited set (G5 fails). At weight
   0.1 `chemakzo` refuses 91% (P2). This is factor 4.
5. **The gate itself is conservative.** Its bound $\sqrt{N h}$ is 1.5–2 times the exact Lebesgue function at the median, and
   its threshold assumes Proposition 5.1's worst case: `lorenz96_d10` at 1,600 extra starts (held-out error 1.5e-3)
   refuses 100% (62% with the exact function); POLLU's threshold 0.31 refuses everything (R5). Part of the
   "solver only" verdicts is calibration.
6. **KIDA** (`kida_dense`) failed in R4 on a registration error (fixed in `b0408bc`); its rerun is pending, so no panel
   result exists yet for the target class.

So a nonlinear basis is worth exploring for item 3 (approximation on fronts and steep layers) and possibly for the
ambient dimension of KIDA. Items 2 and 4 are about data and inputs, which the literature below shows no basis
removes by itself (section 8.4), and item 5 is about the gate, which should be fixed before any comparison (section 10,
step 0).

## 2. The criteria

| Criterion | Why it matters here | How it is measured on the panel |
|---|---|---|
| **Max error** | the target metric: the worst error over all states visited, not a mean or a percentile | `worst` over test inputs, rollouts and mixed chains |
| **Oracle and training cost** | solver calls to build; wall time to train | evaluations, training seconds |
| **Inference cost** | the surrogate must be much cheaper than a stiff solve, per cell, on CPU and GPU | `per_query_us`; effective speed-up with the gate's cost |
| **Stiff and multiscale maps** | fast directions, initial layers, abundances spanning tens of decades | the chemistry and lattice classes of the panel |
| **Rollout stability** | errors accumulate along chains (factor 3, $A_n$) | rollout worst against one-step worst |
| **Off the training trajectories** | operator splitting moves states between steps (factor 4) | mixing chains at weights 0.01, 0.1 |
| **Conservation and positivity** | element and charge balance; abundances > 0 (log space) | invariant residuals after `Projected` |
| **Dimension scaling** | intrinsic (chart) dimension against ambient (species count: 578 for KIDA) | cost and error against $k$ and $d$ |
| **Refusal** | the ability to detect inputs it cannot serve, and send them to the solver | refusal rate $\rho$, accepted error, $1/\rho$ cap |
| **Certifiability** | a bound that holds on a stated set (T4: audits; Theorem 8: rollouts) | which of the memo's statements apply |

## 3. Comparison table

Ratings are relative within this table: **good**, **fair**, **poor**, or **unknown** where no evidence was found.
"Max error" rates the *evidence* that the family reaches a small sup error, not its asymptotic rate. Families marked
with * are the project's.

| Family | Max error | Oracle / training cost | Inference (CPU / GPU) | Stiff, multiscale | Rollouts | Off-data (splitting) | Conservation, positivity | Dimension scaling | Refusal | Certifiable |
|---|---|---|---|---|---|---|---|---|---|---|
| ChebTT, -M, -A* | good on boxes (designed samples; Prop. 4.5 in principle) | poor at high $d$ and per output (U6) | good / good | fair (box sup norm wastes effort, T7) | fair (U2) | good inside the box | via projection | poor in ambient $d$; TT ranks help on lattices (T3) | box membership only | best in the project (Prop. 4.5, impractical sizes) |
| ChebTree, SparseCheb, hub* | good where accepted | fair–poor | good / good | fair; ChebTree splits layers | fair | good inside the box | via projection | poor in ambient $d$ | ChebTree refuses at its depth cap | Prop. 4.5 per box |
| AFINNI, ART* | good at loose tolerances; ART beats schemes in 21 of 402 function cases | fair; one sample serves all outputs | good / good (GPU engine) | fair (oblique leaves follow slow manifolds) | fair (Robertson: ART 0.0163 vs ChebTT 0.0505) | good inside the box | via projection | fair (oblique, but box-filling) | trust regions (AFINNI) | per leaf: Prop. 5.1 |
| Generic chart recipe* | good where trajectories cover use (5 problems, 2e-6 to 5e-4) | good (trajectories) | good / good | good on chemistry with low $k$ | good where served | **poor** (G5) | projection + lift guard | good in $k$; lift is the limit | **computable** (Prop. 5.1), conservative | Theorem 8 for factored maps |
| MetricKernel* | fair | fair; dense solves | fair / fair | unknown | unknown | fair (metric extrapolates smoothly) | via projection | fair with a learned metric | Lebesgue function computable | power function needs an RKHS-norm bound (II-27) |
| SlowTable (ISAT)* | fair (controls a quantile, III-13) | good (in situ) | fair (tree search) / poor | good: linear in small ellipsoids | good (refuses outside) | refuses | per record | good in intrinsic dim; memory grows | **built in** (ellipsoids) | estimates only |
| MLP / ResNet one-step | poor in the sources: astro 99th percentile 0.38–2.8 dex (NN-39) | fair (one fit; needs dense coverage) | good / **very good** (4,250x with GPU, NN-32) | fair with log/Box–Cox scaling (NN-28) | poor without care (NN-39, NN-38, NN-27) | poor (NN-28, NN-67) | soft or architectural (VI-17, VI-19) | good in ambient $d$ | ensembles, conformal (no bound) | Lipschitz and range bounds (NN-74–76); no sup bound from samples (NN-65) |
| Neural ODE (learned RHS) | fair (H2 autoignition, NN-26) | poor (stiff solves in training, VII-30) | poor (still integrates) | the solver handles stiffness | better than one-step (NN-27) | poor | soft loss (NN-27) | good | as MLP | reachability tools (NN-77), small nets |
| DeepONet, FNO, neural operators | fair: ~1e-2 (III-27), <0.6 dex mean (NN-38) | fair–poor | good / good | fair; PoU variants for stiffness (TH-61) | poor (NN-38) | poor | soft | designed for function spaces | as MLP | encoding/approximation/reconstruction bounds (NN-3), in $L^2$ |
| PINNs, PI operators | poor on stiff kinetics (NN-6, NN-7) | poor (optimisation) | good | **poor**: fail without QSSA (NN-6) | n/a (continuous time) | poor | soft | fair | none | none |
| AE + latent ODE / step map | poor in astro: 99th percentile 3–4.8 dex (NN-39); 26–65x (III-22, III-25) | fair | good / good | fair | **better** than one-step under iteration (NN-39) | poor | through decoder | **good**: ambient $d$ enters only the AE | weak (NN-39) | Theorem 8 form (NN-18, NN-17) |
| Koopman, DMD, EDMD | poor: linear latent 99th percentile 3.3–4.7 dex (NN-39); 4,000x fast (III-23) | good (linear algebra) | very good | poor (one linear map for multiple fixed points) | fair | poor | linear constraints exact | fair | dictionary residual | spectral, not sup |
| SINDy | unknown for step maps | good | good | poor (sparse RHS of a known network is the known RHS) | as the integrator | fair if exact terms found | possible | poor beyond tens of states | none | none |
| Transformers, reservoirs, foundation models | poor for sup error; good for short-horizon forecasts (NN-21, VII-16) | poor (large data) | poor (context) / fair | unknown | statistics, not states | poor | none | fair | none | none |
| Mixture of experts, partitioned nets | fair: 8 experts halve inference (NN-25) | fair | good / good | good (split by regime; NN-31, NN-25) | as MLP | poor | per expert | good | gate confidence | per expert as MLP |
| GP, kernels, RBF | fair; rigorous only with an RKHS bound (II-27) | fair; $O(N^3)$ | fair / fair | poor with stationary kernels on layers | unknown | posterior variance grows off data | via projection | poor beyond ~10 dims unless a metric is learned (V-24) | **posterior variance**, computable | conditional on RKHS norm |
| Random features + LS head | fair | good (linear solve) | good / good | as the features | unknown | poor (features extrapolate badly) | via projection | fair (NN-53: $O(\sqrt N\log N)$ features suffice in $L^2$) | **Lebesgue function computable** | Prop. 5.1 exactly |
| Rational (AAA, p-AAA, lightning) | **good** in 1-D near singularities (NN-42, NN-43) | good | good | **good** for layers and near-poles | unknown | poor (poles nearby) | via projection | poor beyond a few variables (NN-44, NN-45) | fixed-pole form: Lebesgue computable | fixed poles: Prop. 5.1 |
| Mapped / moving-mesh Chebyshev | good when the map fits the layer (NN-46) | good | good | good for one layer per axis | as ChebTT | good inside the box | via projection | as ChebTT | as ChebTT | as Prop. 4.5 after the map |
| TN with nonlinear features | unknown for sup error | fair | good | unknown | unknown | unknown | none | good if low rank (T3) | none | none |

## 4. The project's current schemes, in this frame

The project's methods are all **linear in their data once their nonlinear choices are frozen**: a tensor train's
pivots, a tree's partition, a chart and its degree, a kernel's metric and centres, a table's ellipsoids. That one
property is what makes Proposition 5.1 and its computable sampling factor available. Section 8 shows which learned
families keep it.

- **ChebTT, ChebTT-M, ChebTT-A, ChebTree, SparseCheb, the hub expansion** (`docs/SCHEMES.md`). Designed samples on a
  box give a logarithmic Lebesgue constant (regime (c) of T5), so the box's sup norm is controlled from samples, and
  Proposition 4.5 is the only certificate in the project (impractical at present radii). Their weakness for this target
  is the box itself: T7 and U4 show the effort goes to vertices no trajectory visits; the cost grows per output (U6) and
  with the ambient dimension. ChebTree's axis-aligned splitting handles one steep layer per axis; ChebTT-A adds one
  fitted interface. None represents a moving front in $d = 10$–80 lattice coordinates cheaply.
- **AFINNI and ART** (README, `docs/ARCHITECTURE.md`). Local polynomials on adaptive partitions: a *mixture of experts
  with a hard, geometric gate* and least-squares experts. ART's oblique leaves follow low-rank curvature; one sample
  serves every output. On the function suite they win only where fixed floors dominate (`docs/SCHEMES_VS_ART.md`). They
  still fill the box.
- **The generic chart recipe** (`theory/panel/generic.py`, memo section 8). Polynomials in learned *linear* (PCA-type)
  chart coordinates with a polynomial lift, trained on trajectories, gated by the Lebesgue function: the closest
  existing design to the hybrids of section 9, and the reference every learned family must beat.
- **MetricKernel.** Matérn-5/2 interpolation in a gradient-learned metric (V-29): a learned linear feature map with a
  kernel head. Linear in data; its Lebesgue function and power function are computable.
- **SlowTable** (ISAT, III-12 to III-14). Local linear records inside ellipsoids of accuracy, with retrieval checks and
  solver fallback: the only scheme whose refusal is built into the representation. Its limits are known from ISAT:
  ellipsoids are estimates, the error control is a quantile (III-13), and rare large errors come from non-convex
  regions of accuracy (III-14).

## 5. Learned surrogates of step maps and flow maps

### 5.1 MLPs and ResNets as one-step maps

*Mechanism.* A network $\Psi_\theta(x) \approx \Phi_\tau(x)$, usually as a residual $x + N_\theta(x)$ in scaled (log, Box–Cox)
coordinates, trained on pairs from trajectories or sampled states. Its error recurrence along a chain is the classical
one-step-method recurrence (TH-15, Lemma 4.1). Multiscale variants train one network per time scale and compose them
(NN-1).

*Strengths.* Inference is a few dense matrix products: Kelp et al. emulate CBM-Z (77 species, one-hour step) with a
17-layer residual network at 250x speed on CPU and a further 17x on GPU (NN-32); Chemulator is about 50,000x faster
than UCLCHEM (III-21). Cost per query does not grow with the number of outputs (unlike ChebTT, U6), and the parameter
count depends on the function's complexity, not on the ambient dimension.

*Weaknesses.*
- **Max error.** The only systematic astrochemistry benchmark that reports a tail metric (NN-39: four KROME datasets,
  up to 37 species and 287 reactions, hyperparameters optimised for accuracy and speed) finds the best architecture,
  a fully connected network, at median log-absolute errors of 0.035–0.26 dex and 99th percentiles of 0.38–2.8 dex
  (its Table 2). The authors chose the 99th percentile *instead of* the maximum on purpose, accepting rare failures
  to be caught by a fallback. Against this project's tolerance of 0.01 dex these are 30–280 times too large at the
  99th percentile, before the maximum.
- **Rollouts.** The same study finds the fully connected models accumulate error under iteration and conjectures
  "initial-condition drift: repeated surrogate evaluations expose the model to states outside its training domain"
  (NN-39). DeepONet emulators of Grackle meet "accumulated errors [that] can lead to numerical
  instability" (NN-38). In combustion, networks trained on source terms "diverge from the true solution when coupled
  with a CFD solver" (NN-26, NN-27).
- **Off-data inputs.** A network trained on flame-manifold data "cannot remain robust toward perturbation, which is
  inevitable for the DNN coupled with the flow field"; Monte Carlo samples cover the space but miss small-scale
  species; a multi-scale sampling with Box–Cox scaling fixes both on their cases (NN-28). ReLU MLPs provably become
  linear along rays away from the data (NN-67). Atmospheric chemistry found that offline-trained solvers give large
  errors in coupled runs, and online training was needed for stable one-year runs (NN-34, search summary).

*Fit to the project.* A strong candidate for the inference-cost criterion and for KIDA's ambient dimension; no source
shows a sup error near 0.01 dex on a chemistry network with 100+ species. Prediction: on the panel's served problems a
network matches the polynomial recipe only with more data; on the Fisher–KPP and Allen–Cahn ladders it may beat the
degree-8 stall (section 8.1), which is the decisive question (section 10).

### 5.2 Neural ODEs

*Mechanism.* A network right-hand side $\dot x = N_\theta(x)$, integrated by a solver during training and use (V-35).

*Evidence.* Plain neural ODEs fail on Robertson and POLLU without output and loss scaling (VII-30). ChemNODE trains
through the integrator on hydrogen–air autoignition and reproduces the full mechanism "at a fraction of the
computational cost" (NN-26); adding element-conservation penalties improves robustness in coupled CFD runs (NN-27).

*Fit.* The learned RHS of a known chemistry network competes with the known RHS: the cost of a chemistry step is the
stiff integration, not the RHS evaluation, and a neural ODE still integrates (more cheaply only if the learned RHS is
less stiff). It belongs in latent coordinates (section 5.5), not in the state space. Not recommended in state space.

### 5.3 DeepONet, FNO and other neural operators

*Mechanism.* Learn maps between function spaces: DeepONet as a branch-trunk inner product (V-32), FNO as Fourier-space
integral kernels (V-33), within the general neural-operator framework (NN-2). For a 0-D chemistry step the "function"
input is the state vector and the trunk input is the time, so DeepONet becomes a time-continuous flow-map emulator.

*Evidence.* Error bounds split into encoding, approximation and reconstruction errors (NN-3); FNO size grows
sub-(log)-linearly in the reciprocal error for Darcy and Navier–Stokes operators (NN-4). All are in $L^2$ or mean
norms. On stiff kinetics: DeepONet with autoencoder and partition-of-unity variants on ROBER, POLLU (25 species) and a
syngas mechanism (TH-61 = VII-31; no error metric in the abstract); ISM chemistry with nine species at about 1e-2
(III-27); Grackle emulation at mean errors below 0.6 dex, up to 6x faster, with iterative instability (NN-38);
atmospheric chemistry with an FNO-based emulator (NN-35, no sup-error numbers in the abstract).

*Fit.* For 0-D chemistry the operator view adds nothing over a one-step network except continuous time (useful under
adaptive hydrodynamic time steps, noted by NN-39). For the lattice problems (Fisher–KPP, Allen–Cahn, Burgers at $d$ =
10–80) translation-equivariant convolution or Fourier layers are the natural architecture, and are the network
counterpart of the light cone (T3). Worth a test only on the lattice ladders, after section 10's first experiment.

### 5.4 PINNs and physics-informed operator learning

*Mechanism.* Penalise the ODE residual at collocation points (NN-5), or train an operator network the same way (NN-8).

*Evidence.* "Stiffness could be the major reason for the failure of the regular PINN" on stiff kinetics; PINNs worked
only after quasi-steady-state reduction (NN-6). Soft residual penalties make the optimisation ill-conditioned for
reaction and convection terms (NN-7). Domain decomposition helps multiscale problems (NN-9).

*Fit.* The project has a cheap, exact oracle; the reason to use residuals instead of data (no data) does not apply,
and the residual loss is a mean, not a sup. **Not recommended.**

### 5.5 Latent dynamics: autoencoders with latent ODEs or step maps, Koopman and DMD, SINDy

*Mechanism.* An encoder to $k$ latent coordinates, a latent evolution (an ODE, a step map, or a linear map), a decoder
back. This is the memo's section 8 design with nonlinear charts and lifts: Theorem 8's $\hat\pi$, $\hat g$, $\hat G$.

*Evidence.*
- Astrochemistry: 29 species to 5 latent species, 65x (III-22); latent neural ODE 55x and a linear latent model up to
  4,000x (III-23); circumstellar envelopes, about 26x (III-25); photodissociation regions from a 3-D simulation
  with latent augmented neural ODEs, reproducing column density maps (NN-37). The systematic comparison (NN-39) finds
  latent models less accurate (99th percentiles 3.0–4.8 dex) but with less error growth under iteration.
- Dynamics: inertial-manifold coordinates for Kuramoto–Sivashinsky with network charts and latent evolution
  (NN-14); intrinsic coordinates through manifold atlases (NN-15); spectral-submanifold normal forms (NN-16);
  nonlinear-manifold Galerkin and LSPG with discrete-time a posteriori error bounds (NN-17); latent dynamics with
  stability and error estimates (NN-18). The memo already notes that Theorem 8 specialises NN-17's and NN-18's technique.
- Koopman: EDMD on a dictionary (NN-12), learned eigenfunctions (NN-13). A finite linear latent model cannot have
  several isolated fixed points or a stable limit cycle, which the panel's chemistry and oscillator classes have; the
  linear latent model's 99th percentile of 3.3–4.7 dex in NN-39 is consistent with that.
- SINDy (NN-10), and with an autoencoder (NN-11): sparse regression of a right-hand side. For a mass-action network
  the RHS is known and polynomial; SINDy would rediscover it, and the cost (stiff integration) remains.

*Fit.* The family that matches the target's dimension structure best: KIDA's 578 states enter only the encoder and
decoder; the dynamics live in $k$ coordinates whose lower count Proposition 6.5 bounds. The memo already found that
the *lift* is the limit (Demonstration 8.5, R4's primordial lift guard, G6), and that linear encoders cannot see
displacements normal to the visited set (G5). A nonlinear decoder and an encoder residual address exactly these two.
The latent *step map* can stay polynomial and keep the Lebesgue gate (section 9, H2). Koopman and SINDy: not
recommended.

### 5.6 Transformers and other sequence models

*Mechanism.* Attention over a history of states (NN-19), pretrained across many systems (NN-20), or foundation models
used zero-shot (NN-21); reservoir computers and nonlinear vector autoregression (VII-16, VII-17, NN-22).

*Evidence.* On 135 chaotic systems, zero-shot foundation models produce forecasts competitive with custom models
(NN-21). Metrics are valid prediction times and attractor statistics (VII-18 to VII-25), not sup errors over states.

*Fit.* A step map of an autonomous ODE is Markov: the history carries no information the state does not. Attention
costs grow with context, and nothing in these sources controls a pointwise error. **Not recommended.**

### 5.7 Mixture of experts and partition-based networks

*Mechanism.* A gate assigns inputs to experts (NN-24, memory; sparse gating at scale, NN-23). In combustion, a gating
network splits flamelet manifolds into regions with one network each, reducing the error at fixed size or the size at
fixed error, and halving inference time with 8 experts instead of 1 (NN-25). Several MLPs for states with small and
large composition changes, trained on hybrid flamelet and random data (NN-31). Partition-of-unity DeepONet (TH-61) and
finite-basis PINNs (NN-9) are the same idea.

*Fit.* AFINNI and ART are mixtures of experts with a geometric gate and polynomial experts; the literature's version
learns the gate. The project's evidence (ART's oblique leaves follow slow manifolds) and NN-25 agree that partitioning
by regime pays on stiff chemistry. A network gate gives curved boundaries that ART rejected in its quadric-split
experiment (I-2's note in `docs/LITERATURE.md`: leaves in several components, worst errors far above tolerance). Worth
considering only as a gate over the existing local experts.

## 6. Chemistry and astrochemistry emulators

| Work | System | Approach | Stiffness | Conservation, positivity | Rollouts | Coupling | Error reported |
|---|---|---|---|---|---|---|---|
| ISAT (III-12–14), SlowTable | combustion, up to 857 species with DAC (III-17) | local linear tables, ellipsoids, solver fallback | exact solves fill the table | exact on records | errors linear in the tolerance (III-14) | in situ in LES/PDF | quantile control (III-13); rare large errors |
| PRISM (III-15) | combustion | quadratic polynomials in hypercubes, on demand | as ISAT | — | — | in situ | — |
| Ihme et al. 2009 (NN-30) | methane/air, LES | optimal-architecture ANN tabulation | tabulated manifold | — | — | LES | search summary only |
| Ding et al. 2021 (NN-31) | turbulent flames | multiple MLPs, hybrid flamelet/random data | split by change magnitude | — | — | LES | search summary only |
| Zhang et al. 2022 (NN-28), DeepFlame (NN-29) | combustion kinetics | DNN step map, Box–Cox, multi-scale sampling | Box–Cox scaling | — | stable in their tests | OpenFOAM + Cantera | robust where manifold-trained nets were not |
| ChemNODE, PC-NODE (NN-26, NN-27) | H2–air autoignition | neural ODE, conservation penalty | integrated in training | soft mass/element penalty | improved | a posteriori CFD | — |
| Kelp et al. 2018 (NN-32) | CBM-Z, 77 species | 17-layer ResNet, 1 h step | — | — | — | offline | RMSE < 1.97 ppb, 250x CPU, 4,250x GPU |
| Keller, Evans 2019 (NN-33) | GEOS-Chem | random forests | predict changes for long-lived, values for short-lived species | — | O3 bias 4.2% at 5 days, 13% at 30 days | online | search summary |
| Kelp et al. 2022 (NN-34) | GEOS-Chem Super-Fast, 12 species | network + autoencoder, online training | — | — | stable 1-year runs | online | 5x; regional O3 and NOx biases |
| Sturm, Wexler (VI-17, VI-18) | photochemistry | learn reaction fluxes, multiply by stoichiometry | — | **exact** atom balance by construction | — | — | — |
| Chemulator (III-21) | UCLCHEM gas-grain | MLP step map, 1000 yr | — | — | stable over 1,000 iterations | for hydro | MSE 2e-4; 50,000x; large errors for some models |
| Grassi et al. (III-22) | 29 species | interpretable AE, 5 latent | — | — | — | — | 65x |
| Sulzer, Buck (III-23); CODES (III-24) | osu2008 | latent NODE, linear latent | — | — | — | — | 55x, 4,000x; mean rel. error < 4% |
| MACE (III-25) | AGB outflows | AE + latent ODE | — | — | — | — | about 26x |
| Branca, Pallottini (III-26, III-27) | ISM, 9 species | PINN, DeepONet | — | — | — | — | about 10%; about 1e-2 |
| Asensio Ramos et al. (NN-36) | Nautilus, 192 species | conditional neural field in time | — | — | none (direct in time) | — | < 0.2 dex; 1e4x |
| Ono, Sugimura (III-28) | primordial, 6 species | MLP | — | — | — | for star formation | < 10% in > 90% of cases |
| NeuralPDR (NN-37) | 3D-PDR | latent augmented NODE | — | — | — | for 3-D sims | column density maps reproduced |
| van de Bor et al. (NN-38) | Grackle | DeepONet variants | — | — | **unstable** when iterated | cosmological | mean < 0.6 dex; up to 6x |
| Janssen et al. 2026 (NN-39) | KROME, up to 37 species | MLP, latent NODE, linear latent, DeepONet-type | log space | — | latent more robust | — | 99th pct 0.38–4.8 dex; deep ensembles flag errors |
| Zhang et al. 2025 (NN-40) | nuclear, 3 and 13 species | DNN, evolutionary Monte Carlo sampling | — | — | stable | **temperature-gated** fallback, NN used > 75% | ≲ 1%; 2.6x CPU |

What the table shows, for this project:

1. **No emulator in the sources reports or controls a maximum error near 0.01 dex** on a network with 100+ species.
   Reported metrics are MSE, RMSE, means, and (once) a 99th percentile. The only representation with a local error
   control is ISAT-type tabulation, and its control is a quantile.
2. **Stability in coupled runs comes from data coverage, not from architecture**: multi-scale or random sampling
   (NN-28, NN-31, NN-40), online training (NN-34), noise in training (III-21 per NN-39). This is the memo's U8 and R5
   result, found independently in four communities.
3. **Fallback is the field's answer to rare large errors** (NN-39 argues for an uncertainty-triggered fallback; NN-40
   gates by temperature; ISAT always has one). The memo's Corollary 5.3 is the accounting this fallback needs and the
   field has not written down: the speed-up is capped at $1/\rho$.
4. **Conservation is solved by construction** (VI-17: fluxes times the stoichiometric matrix; VI-19, VI-20); the
   project's projection onto linear invariants does the same for any model.

## 7. Other non-polynomial bases

### 7.1 Kernels, Gaussian processes, RBFs, random features

*Mechanism.* Interpolation or regression in the span of kernel translates (V-27); GPs add a posterior variance; random
features approximate a kernel by a finite random basis with a least-squares head (NN-52, memory; NN-53); kernel flows
learn the kernel by cross-validation (NN-54), applied to learning dynamical systems (NN-55); random features for
operators between Banach spaces (NN-51).

*Strengths.* Linear in the data: the Lebesgue function (the sum of the absolute cardinal functions) is computable, so
Proposition 5.1 and Corollary 5.3 hold exactly. The GP posterior variance is computable before the value is known, the
kernel analogue of $\Lambda_X(t)$; with an RKHS-norm bound it gives a rigorous pointwise bound (II-27), the kernel
analogue of Proposition 4.5's analyticity hypothesis.

*Weaknesses.* Dense $O(N^3)$ solves (MetricKernel's limit); stationary kernels on steep layers need many centres;
in high dimension a learned metric is needed (V-24, V-29), which is what MetricKernel does. Random features with
$O(\sqrt N\log N)$ features reach the kernel's $L^2$ learning rate (NN-53); no sup-norm analogue was found.

*Fit.* Already represented by MetricKernel. Random features are the cheapest way to test a *nonlinear basis that keeps
Proposition 5.1 exactly* (section 10, experiment 1a).

### 7.2 Rational approximation: AAA, p-AAA, lightning

*Mechanism.* Barycentric rational interpolants with greedy support points (NN-42); multivariate parametric versions on
grids (NN-44) and scattered data (NN-45); fixed poles exponentially clustered near singularities, fitted by linear
least squares, with root-exponential convergence up to corner singularities (NN-43).

*Fit.* Rational functions resolve near-poles and steep layers that cost polynomials many degrees: the Bernstein-ellipse
rate (T1) is replaced by a rate set by the singularity's type. Fixed-pole forms (lightning) are linear in the data, so
Proposition 5.1 holds. The multivariate versions are tensor-grid or low-variable methods; at $d$ = 10–80 they would
enter as 1-D factors (a rational axis inside ChebTT, or a rational profile along a learned ridge). Candidate fit for
`fisher_kpp_d2` (needs degree > 8, G2) and Robertson's initial layer, not for the high-dimensional ladders.

### 7.3 Mapped Chebyshev, adaptive and moving meshes, transport-aware reduction

*Mechanism.* A coordinate map that stretches Chebyshev points into a layer (NN-47, memory) or adapts to the
singularities' location in the complex plane (NN-46: rational interpolation on transformed Chebyshev points, "far
outperforms" Chebyshev for singularities near $[-1, 1]$); moving meshes equidistribute a monitor function (NN-48,
memory); for transport, shifted POD separates moving fronts by time-dependent shifts (NN-49), because linear
reductions of transport problems converge slowly in the Kolmogorov width (NN-50).

*Fit.* NN-50's diagnosis is the panel's item 3 in model-reduction language: a front moving across the lattice has a
slowly decaying linear width, so a fixed polynomial space in lattice coordinates stalls. The cure is a nonlinear
coordinate (the front position or shift) followed by a smooth fit: this is either a hand-built transport chart (sPOD)
or a learned feature (section 9, H1).

### 7.4 Tensor networks with nonlinear features

Tensor trains over local feature maps learned by sweeps (NN-56); TT parametrisations of dynamical laws (TH-32, V-41).
They inherit ChebTT's rank structure (T3's light cone suggests ranks independent of $d$ on lattices) with learned 1-D
features. No sup-error evidence; the project's ChebTT with piecewise and rational axes is the same design with
designed samples. Not a priority.

### 7.5 Kolmogorov–Arnold networks

Spline activations on edges (V-34); scaling claims, no worst-case control. Not a priority.

## 8. What the theory says about each family

### 8.1 Approximation (factor 1)

- **Smooth functions.** Deep ReLU networks reach Sobolev rates with fewer parameters than shallow ones (NN-59); analytic
  functions are approximated at exponential rates in low dimension (NN-60) and, for maps holomorphic on Bernstein
  polyellipses, at $O(\exp(-bN^{1/(d+1)}))$ in $W^{1,\infty}$ in network size $N$ (NN-61, search summary): the
  constructions *re-approximate polynomials*, so for the analytic step maps of T1 networks match, but do not beat,
  polynomial rates (NN-64 compares the two from the viewpoint of rate distortion). Barron-space functions are
  approximated at a dimension-independent rate by two-layer networks in $L^2$ (NN-57, memory; NN-58), a class with no
  link yet to step maps.
- **Fronts and piecewise smooth functions.** Piecewise $C^\beta$ functions with $C^\beta$ interfaces need
  $O(\varepsilon^{-2(d-1)/\beta})$ weights in $L^2$, which is optimal, and a minimal depth (NN-63). More directly: a single
  $\tanh$ unit $\tanh(a\cdot x + b)$ *is* a planar front of the Allen–Cahn profile type, and a moving front is a
  smooth function of a few such units' parameters. Polynomials need degree growing with the front's steepness; this
  is where a learned basis can change factor 1 (section 1, item 3).
- **Practice against theory.** Trained networks fall short of their theoretical rates on smooth functions, where
  compressed-sensing polynomials do better (V-36); there is a provable gap from sampling-complexity bounds (V-37);
  networks are biased toward low frequencies and fit sharp local features last (NN-66).
- **Memo.** T1 (radii, degrees per decade) and section F's scaling laws are polynomial-specific. **T6's coordinate counts
  carry over unchanged:** Proposition 6.5's lower bound holds for every continuous encoder, so an autoencoder whose
  latent dimension is below the count cannot reach the tolerance, whatever its decoder.

### 8.2 Sampling (factor 2): what plays the role of $\Lambda_X(t)$ for a network

Proposition 5.1 needs an estimator that is **linear in the data and reproduces its space**. A trained network is
neither, and:

- **Uniform accuracy from samples is intractable for network classes.** For target classes containing ReLU networks of
  a fixed architecture, the number of samples any algorithm needs to guarantee a uniform accuracy grows exponentially
  in the depth and the input dimension (NN-65). Polynomial classes face the same worst case (II-26), but they have a
  computable *a posteriori* sampling factor; networks do not.
- **Frozen features restore it exactly.** If the network's last hidden layer $\varphi_\theta$ is frozen and only the
  linear head is fitted by least squares, $V = \operatorname{span}\varphi_\theta$ is a fixed linear space and Proposition 5.1,
  the bound $\Lambda_X(t) \le \sqrt{\varphi(t)^{\mathsf T}G^{-1}\varphi(t)}$ and Corollary 5.3 hold *verbatim*, with
  $\Lambda_X$ computable at any input. The inequality is algebraic, so it holds even if $\varphi_\theta$ was trained on the
  same data; what training on the same data biases is the held-out estimate of $e^*_V$, so features and head need
  separate folds. The same holds for random features, fixed-pole rationals and mapped Chebyshev.
- **Linearised networks.** In the infinite-width limit training follows a kernel, the NTK (NN-82): the kernel's
  Lebesgue function describes a linearised network, not a trained finite one.
- **Design of samples.** Generalised Christoffel functions extend optimal sampling to nonlinear model classes (unions
  of subspaces) with near-optimal sample complexity in some cases (NN-83); active, dynamics-guided data collection is
  what made networks work in high-dimensional evolution problems (NN-84).
- **Heuristics without an inequality.** Ensemble spread (NN-81), feature-space distance or density: they rank inputs by
  risk and can drive refusal, but no bound links them to the error.

### 8.3 Propagation (factor 3) and rollout stability

- **Carries over.** Propositions 2.1 and 2.2 and Theorem 8 hold for any map $\Psi$; they need the defect on a tube and a
  Lipschitz constant of the deployed map. For networks, Lipschitz upper bounds are computable (layer products; the SDP
  of NN-76), and can be enforced (spectral normalisation, NN-68; contracting recurrent models by construction, NN-69;
  a jointly learned Lyapunov function, NN-70). Contracting-by-construction models cannot represent the expanding
  directions of chaotic maps, so they suit the stiff contracting classes only.
- **Training for rollouts.** Training through several steps (pushforward and zero-stability, NN-71; integrating during
  training, NN-26) reduces drift. Climate parametrisations show that a network accurate offline can be unstable once
  coupled (NN-73) and can fail far outside its training manifold (NN-72): the coupling, not the one-step error, decides.

### 8.4 Off the data (factor 4) and operator splitting

The memo's Proposition 8.2 is model-independent; so is the conclusion of every chemistry study in section 6: coverage of
the inputs the coupled run produces decides the error (NN-28, NN-34, NN-39, NN-40), and ReLU networks extrapolate
linearly (NN-67). **No family removes factor 4 by its basis.** What a nonlinear decoder adds is a meaningful *normal
residual* $\lVert x - \hat G(\hat\pi(x))\rVert$, which sees the displacements G5 showed the chart-coordinate bound to miss.

### 8.5 Refusal and speed (Corollary 5.3)

Parts (c) and (d), the $1/\rho$ cap and refused steps adding no defect, hold for **any** gate and any surrogate. Parts
(a) and (b), a refusal rate computable from unlabelled inputs *and* an accepted error bounded by $(1 + \Lambda^*)e^*$,
need a gate tied to the error by an inequality: only linear-in-data heads have one. For networks, the gate is
heuristic (ensembles: NN-39 finds them to flag the worst 1% efficiently for fully connected models, poorly for latent
ones) or distributional: split conformal gives marginal coverage under exchangeability (V-38, V-39), weighted conformal
needs the likelihood ratio under covariate shift (V-40), adaptive conformal gives long-run coverage under drift
(NN-78), and there are multi-horizon versions (NN-79) and cell-wise calibration for surrogates (NN-80). None bounds the
maximum along one chain; and an ensemble of $M$ members costs $c_g \approx M c_\Psi$ in Corollary 5.3(c).

### 8.6 Certification

- **T4 audits carry over to every family** (Propositions 4.1, 4.2, Corollaries 4.1′, 4.3): they are statements about a
  frozen map and fresh samples. Proposition 4.2's step from measure to sup needs a Lipschitz constant of the error,
  available for networks through NN-76.
- **Enclosures of a network over a box** exist: interval bound propagation (NN-74), linear relaxations for general
  activations (NN-75), reachability of neural ODEs (NN-77). They bound the network's *range*, not its error against
  $\Phi$; combined with a verified enclosure of $\Phi$ on the same box they give a certificate in the form of
  Proposition 4.5, at a cost that grows with the box count.
- **Proposition 4.5 (analyticity plus grid residuals) is polynomial-specific.**

### 8.7 Summary: which memo results carry over

| Memo result | Polynomial schemes | Linear head on frozen features (random features, last layer, kernels, fixed-pole rationals, mapped Chebyshev) | Trained networks (one-step, operators, MoE) | Latent models (AE + latent map) |
|---|---|---|---|---|
| Prop. 5.1, computable $\Lambda_X(t)$ | yes | **yes, exactly** | no; heuristics or NTK approximation | yes for a linear latent head, in latent coordinates |
| Cor. 5.3 (a)–(b): refusal rate and accepted-error bound | yes | yes | no (conformal: marginal coverage only) | in latent coordinates; plus a normal residual |
| Cor. 5.3 (c)–(d): $1/\rho$ cap, refused steps add no defect | yes | yes | yes | yes |
| Props. 2.1, 2.2, Theorem 8 (propagation) | yes | yes | yes, with network Lipschitz bounds | **yes: Theorem 8 is this design** |
| Prop. 8.2 (off-data) | conditional | conditional | conditional | conditional; the nonlinear lift is what it asks for |
| T4 audits; Prop. 4.2 measure to sup | yes | yes | yes (Lipschitz via NN-76) | yes |
| Prop. 4.5 (certificate under analyticity) | yes | for polynomial heads | no; IBP/CROWN range bounds instead | no |
| T6 coordinate counts (Prop. 6.5) | yes | yes | yes | **yes: lower bound on the latent dimension** |
| T1 radii, degrees per decade; section F | yes | no | no | no |
| T3 light cone | yes (TT ranks) | yes (local features) | as architecture (convolutions) | no |

## 9. Hybrid designs worth considering

- **H1. Learned feature map, linear head, exact Lebesgue gate.** Train a small network (or draw random $\tanh$ ridge
  features) on fold A; freeze its last hidden layer $\varphi$; fit the head by least squares on fold B; refuse where
  $\Lambda_X(t)$ in feature space exceeds the threshold. Keeps every row of the middle column of 8.7. Targets item 3 of
  section 1: $\tanh$ ridge features represent planar fronts that polynomials of degree 8 do not.
- **H2. Nonlinear lift with a polynomial latent map and a two-part gate.** Keep the generic recipe's chart and polynomial
  latent step; replace the polynomial lift by a network decoder; refuse on the latent Lebesgue function *or* on the
  normal residual $\lVert x - \hat G(\hat\pi(x))\rVert$. Theorem 8 applies to the factored part; the normal-residual branch is
  the "augmented encoder" Theorem 8's scope asks for. Targets KIDA (ambient 578), POLLU and Nelson (lift-limited) and
  G5 (mixing).
- **H3. Network with solver fallback, gated by a calibrated ensemble.** A one-step MLP in log coordinates, a deep
  ensemble's spread as the gate, its threshold set by split conformal on held-out *rollout and mixed* inputs, solver
  fallback, scored by Corollary 5.3. The field's design (NN-39, NN-40), made measurable.
- **H4. ISAT with a learned base.** SlowTable over the residual of a network or polynomial base map: records only where the
  base errs, ellipsoids in the base's error metric. ISAT's refusal and in-situ growth, with fewer records. In
  combustion, networks replacing mature ISAT subtrees exist (IV-15).
- **H5. Rational or mapped axes for layers.** A fixed-pole rational or a mapped Chebyshev axis inside ChebTT or ChebTree
  where an axis's coefficients decay slowly (where ChebTT now searches for jumps). Keeps Proposition 5.1; low cost.
- **H6. Partitioned experts with a learned gate.** ART or AFINNI leaves as experts, a network gate trained to assign
  inputs to the leaf with the smallest error. Lower priority: ART's quadric-split experience warns about multi-component
  leaves.
- **Common to all:** conservation by projection (or VI-17's flux form for networks), log coordinates for positivity, and
  the memo's fields (`refused_percent`, `rollout_solved_by_step`, effective speed-up) in every record.

## 10. Recommendation

**Step 0 (prerequisite, cheap): fix the gate before comparing bases.** R5 found the refusal rule 1.5–2x conservative
and its threshold set for the worst case. Use the exact $\Lambda_X(t)$ (one solve per query with the factored Gram
matrix) and set $\Lambda^*$ from held-out errors as Corollary 5.3 describes, then rerun R5's twelve problems. Without
this, every learned basis below would be compared against a gate that refuses inputs the polynomial already serves.

Ranked, with the cheapest decisive experiment for each:

1. **H1: learned or random nonlinear features with a least-squares head and the exact Lebesgue gate.** It is the only
   family that can change factor 1 on the approximation-limited fronts while keeping Proposition 5.1 and Corollary 5.3
   exactly.
   *Experiment.* In `generic.py`, add a basis option beside the chart polynomials: (a) $m$ random $\tanh(w\cdot u + b)$
   features of the chart coordinates ($m$ = 50–800, no training); (b) the last hidden layer of a two-layer MLP trained
   on fold A, frozen. Run on R5's twelve problems at the same 400 and 1,600 extra starts, folds and tolerance.
   *Decisive reading.* If the held-out error on `fisher_kpp_d3` and `fisher_kpp_d10` keeps falling past the
   polynomial stall (0.017, 0.09) toward 0.01 with the refusal rate below 50%, the limit was the basis; if it stalls at
   the same level, the limit is data or the chart, and learned bases are not the fix. Minutes per problem on CPU.
2. **H2: a nonlinear lift (network decoder) with the polynomial latent map, and a normal-residual gate.** The one
   family suited to KIDA's dimension structure, and the direct answer to the lift-limited (G6, POLLU, Nelson) and
   mixing (G5) failures.
   *Experiment.* Decoder-only, before any rollout: on KIDA's dense trajectories (once the R4 rerun has produced them)
   and on `hires`, `pollu`, `nelson`, `primordial`, fit an MLP decoder from the chart coordinates at $k$ from
   Proposition 6.5's lower count to $k + 3$, and report the maximum reconstruction error on held-out states against the
   current affine-to-quartic decoders. Then rerun the `hires` mixing chain at weight 0.01 with the normal-residual
   refusal added. *Decisive reading.* If no decoder at any $k$ reconstructs held-out states within the tolerance,
   latent models cannot serve KIDA at 0.01, and nothing downstream will; if it does and the mixing worst falls from
   0.38 below 0.01 at an acceptable refusal rate, H2 is the design for the target.
3. **H3: an MLP one-step baseline with an ensemble gate calibrated by conformal prediction, and solver fallback.** The
   field's standard design, measured on the project's terms for the first time: max error, refusal rate and effective
   speed-up, beside the polynomial recipe on the same data.
   *Experiment.* One residual MLP per problem in the generic recipe's coordinates, a 5-member ensemble, threshold by
   split conformal on held-out rollout and mixed inputs; the full 37-problem panel and KIDA, scored with
   `p2_tradeoff.py`'s fields. GPU training is available; CODES' baselines (III-24, NN-39) are a reference
   implementation. *Decisive reading.* The comparison per class: if the network serves problems the polynomial refuses
   (beyond the fronts of experiment 1) at equal data and an effective speed-up above 10x, networks earn a place;
   if, as NN-39's 99th percentiles predict, the accepted worst error exceeds 0.01 wherever the refusal rate is below
   50%, the basis is not the bottleneck.
4. **H5: rational or mapped axes for steep layers in low dimension.** *Experiment.* On `fisher_kpp_d2` and Robertson's
   initial-layer region, compare the number of parameters for a sup error of 0.01 between Chebyshev, AAA (NN-42) and
   a fixed-pole (lightning-type, NN-43) basis on 1-D and 2-D slices. A day's work; decisive if the rational basis
   needs several times fewer parameters, in which case it enters ChebTT's piecewise-axis machinery.
5. **H4: ISAT with a learned base**, only after 1–3: it combines what they find.

**What not to pursue, and why.**

- **PINNs and physics-informed operators**: they fail on stiff kinetics without reduction (NN-6, NN-7), minimise a
  mean residual, and replace data the project gets cheaply from an exact oracle.
- **Neural ODEs in the full state space**: inference still integrates a stiff system, and the known RHS is cheaper than
  a learned one (VII-30, NN-26). In latent coordinates they are part of H2's comparison, not a separate line.
- **Koopman, DMD and linear latent models**: a finite linear model cannot represent several attractors; the best
  benchmark's 99th percentiles are 3.3–4.7 dex (NN-39).
- **SINDy**: it discovers right-hand sides; the project's are known, and the cost is the integration.
- **Transformers, reservoirs, foundation models**: the step map is Markov, the cost grows with context, and their
  successes are forecasts and attractor statistics (NN-21, VII-18), not sup errors.
- **Global neural operators (FNO, DeepONet) for 0-D chemistry**: no gain over a one-step network beyond continuous
  time, and their reported errors are ~1e-2 to 0.6 dex (III-27, NN-38). Revisit convolutional or Fourier layers only
  for the lattice ladders, if experiment 1 shows that fronts need learned features.
- **KANs and tensor networks with learned features**: no evidence on worst-case error; they duplicate designs the
  project already has with designed samples.

**Bottom line.** Learned bases are worth exploring narrowly: as feature maps and lifts inside the project's linear,
gated designs (H1, H2), where they can fix the two failures that are about the basis (fronts; the lift at high ambient
dimension) without losing the computable sampling factor, the refusal accounting and Theorem 8. As stand-alone
surrogates, the literature gives no evidence that they reach a maximum error of 0.01 dex on stiff chemistry, and it
shows the same data-coverage and off-data failures the panel found, in every community that coupled them to a flow
solver.
