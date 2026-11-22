# Synthesis: surrogates of stiff step maps — what was built, what was found, what went wrong, and what to try next

This document gathers one long stretch of work in one place, so that a reviewer can judge what to do next without reading the
history: the theory (`theory/MEMO.md`), every experiment and its result, the mistakes made and what each taught, the strengths and
weaknesses of every scheme tried, and what the literature says about newer surrogate families (`theory/APPROACHES.md`). It ends with
the open problems and the candidate directions, and with the questions put to the reviewers.

- **Evidence.** Every number comes from a committed script's output or record; the paths are given. Results are labelled
  *pre-registered* (predictions committed before the run; the commit is named) or *post hoc*. The memo's opening has the full
  provenance table; `git log --oneline -- theory/` shows each step.
- **Sources.** Literature IDs refer to `theory/LITERATURE.md` (`TH-*`, `NN-*`) and `docs/LITERATURE.md` (`I-*` to `VIII-*`), which
  record whether each source was read in full, in part, as an abstract, or is cited from memory. Published results are taken with
  caution: most report mean or percentile errors on the authors' chosen problems, not the maximum error over every state visited.
- **Units.** Errors are in dex (decades) for chemistry, in absolute units for the lattice problems; the tolerance is 0.01 unless
  stated. A surrogate *holds* the tolerance when its worst error over the scored outputs is within it.

---

## 1. The goal and the starting point

**The goal.** Surrogates of the step map $\Phi_\tau$ of stiff ODE systems, with the *maximum* pointwise error over every state a
simulation visits as the metric, in long rollouts and under operator splitting. The end target is astrochemistry: networks of
100+ species (KIDA uva 2024 has 578 species and 8,275 reactions, `libs/KIDApy`) coupled to MHD, where the chemistry is advanced cell
by cell between hydrodynamic steps.

**The library** ("adaptive surrogates", `README.md`, `docs/ARCHITECTURE.md`) has two adaptive local plans on one refinement engine
(AFINNI: adaptive clustering with trust regions; ART: adaptive regression trees with oblique splits) and global schemes
(`docs/SCHEMES.md`): ChebTT (Chebyshev tensor trains; ChebTT-M rotated, ChebTT-A with an interface term), ChebTree, SparseCheb,
the hub expansion, MetricKernel, SlowTable (ISAT-type tabulation). The benchmark (`docs/BENCHMARK.md`) runs every method on a
function suite and on 37 step maps: stiff chemistry (Robertson, HIRES, POLLU, CHEMAKZO, primordial), oscillators (Van der Pol, the
Oregonator), lattice ladders at $d$ = 2–80 (Brusselator, Allen–Cahn, Fisher–KPP, Burgers, Lorenz-96) and one astrochemical network
with a driven environment (Nelson, 14 species and 4 held inputs).

**Where the work began.** The benchmark's records showed ChebTT-type schemes holding 0.01 on small ladders (2e-3 to 4e-3 rollouts,
hundreds to tens of thousands of solves) and nothing finishing on HIRES, POLLU, Nelson, primordial or the large ladders. ART's
quadratic-leaf variant had a known vertex failure. The schemes had been improved by engineering one failure at a time. The user
asked for a foundation first: theorems with precise hypotheses, each checked numerically, before more schemes.

---

## 2. The theory as it stands (`theory/MEMO.md`, revision 5 plus later sections)

Ten review rounds shaped it (`theory/REVIEWS.md`: Claude Fable 5.2 rounds 1–3, 5–7 and 10; GPT Sol 6.2 rounds 4, 8, 9 and a recipe
review). The memo now opens with one decomposition and every section bounds one factor of it.

### 2.1 The organising decomposition (section E)

A surrogate $\Psi$ built in a space $V$ from data at states $X$ and used along a chain with inputs $t_0, t_1, \ldots$ errs, after $n$
steps, by about

$$A_n \max_j\,(1 + \Lambda_X(t_j))\,e^*_V(X\cup\{t_j\})$$

- **Approximation** $e^*_V$: how well the space can represent the step (T1: analyticity radii and degrees per decade; T6: reduction to
  fewer coordinates and its lower bound; T7: the box against the visited set).
- **Sampling** $1 + \Lambda_X(t)$: how the data sit relative to each point of use (T5, Proposition 5.1). For a fit linear in its data,
  $\Lambda_X(t) = \sum_i|\ell_i(t)|$ is computable at any input before the step there is known.
- **Propagation** $A_n$: how the dynamics carry one-step errors (T2, Theorem 8, T3).
- **Off the data**: inputs no data constrain (operator splitting, mixing), through factor 2 at those inputs and Proposition 8.2.
- **Refusal and speed** (Corollary 5.3): a refusal rate $\rho$ caps the speed-up at $1/\rho$ whatever the surrogate costs; refused steps
  add no defect in a rollout.

It is an estimate, not a bound: the chain's inputs are not known in advance, so a bound needs a tube argument (Theorem 8,
Proposition 2.1).

### 2.2 Results by status

| Result | Statement (short) | Status |
|---|---|---|
| T1, Theorem 1, Lemmas 1.1–1.2 | complexified log-norm tubes give Bernstein radii of $\Phi_\tau$ a priori | proved; valid wherever tried, informative only on small boxes; radii shrink with stiffness in practice |
| Proposition 1.5 | complex existence domains nest in $\tau$ | proved; the true nearest singularity need *not* approach monotonically (counterexample $w' = iw + w^2$, round 8) |
| T2, Propositions 2.1–2.4 | global error recurrence; first-order representation; neutral dynamics; normal stability | proved, with the hypotheses round 8 added (norms, defect budget on the chain) |
| T3, Lemma 3.1, Corollaries 3.2–3.3 | light cone for locally coupled systems; window freezing; TT ranks | proved; dimension-free only when $h^*$, $e^{\nu\tau}$, $\ell\Delta\tau$ are uniform in $d$ (rounds 8–9) |
| T4, Propositions 4.1–4.4, Corollary 4.3 | adaptive audits; measure to sup; sharpness | proved; 4.3 is a joint-event bound, not a conditional guarantee (round 8) |
| T5, Proposition 5.1 | error at a point of use $\le (1 + \Lambda_X(t))\,e^*$ for any linear fit | proved (exact least squares; ridge, rank deficiency, clipping and label noise are outside it, round 10) |
| Corollary 5.3 | refusal rate computable from unlabelled inputs; accepted error bound; speed-up $\le 1/\rho$; refused steps add no defect | proved; (b) hides $e^*_V(X\cup\{t\})$ off the data (round 10) |
| T6, Theorem 6, Corollaries 6.1–6.4 | factorisation through a slow manifold under Fenichel's hypothesis (S) | conditional; domains and paired constants fixed in rounds 8–9 |
| Proposition 6.5 | Borsuk–Ulam lower bound on the coordinate count | proved, citing Borsuk–Ulam (a specialisation of DeVore–Howard–Micchelli) |
| Theorem 6.6, Proposition 6.7, Example 6.8 | compound-matrix bound on singular values; staircase; local rank 1 with coordinate count 2 | proved; bounds a local rank, not a coordinate count |
| T7 | sup norm on a box against accuracy where states are (Hoeffding) | proved; an illustration |
| Theorem 8 | finite-horizon rollout bound for factored maps $\hat G\circ\hat\pi$ with computable constants | proved; applies only to maps constant on their encoder's fibres (round 9) |
| Proposition 8.2 | accuracy off the visited set under (S) | conditional; needs a paired Lipschitz constant on a domain holding both images (rounds 8–9) |

**Single-problem demonstrations** (memo Appendix A): Theorem 8 end to end on one Robertson leaf, bound 13–27 times the measured
rollout with half the constants rigorous (`theory/demos/robertson_leaf.py`); chart dimensions of the Fisher–KPP ladder mesh-independent
to $d$ = 80 (`reduced_dims.py`); a curved lift on HIRES 157 times better than an affine one (`curved_lift.py`).

**Prior art** (memo, `LITERATURE.md`). T1 is a combination of classical tools; T2's first-order representation and Theorem 8 are
discrete Lipschitz estimates of the kind of Lee–Carlberg (manifold LSPG) and Farenga et al. (latent dynamics); T4 is Wilks, adaptive
data analysis and optimal recovery (Sukharev, Novak); T5 is the classical least-squares Lebesgue inequality (Cohen, Davenport,
Leviatan, TH-40) applied at a point of use; T6 rests on Fenichel and DeVore–Howard–Micchelli; the first-order correction about a known
state with the step's Jacobian, an accuracy region and solver fallback is ISAT's (Pope 1997, TH-62). What would be new is a *verified*
end-to-end certificate for a stated class of stiff step maps, which no part of this work has yet produced.

---

## 3. The experiments, in order, with their results

### 3.1 Static functions first (memo section F; `theory/functions/f*.py`; post hoc)

Scaling laws on the benchmark's function suite. Degrees per decade from Bernstein radii within about 30% with the axes' order right;
kinks converge like $n^{-1}$; ChebTT with breakpoints holds aligned kinks at every $d$; an interface term (ChebTT-A) holds oblique
jumps; a rotation pays only for shared latent coordinates. Two predictions failed (`wave_pair`: a trigonometric ridge has TT rank 2;
`genz_triple`: shared parameters lower the local rank), both avoidable from the memo's own Lemma 7.2.

### 3.2 Reduced coordinates and the HIRES case (memo section 8, Appendix A; `theory/demos/`, `theory/gradients/`)

- **Charts and lifts** (Demonstrations 8.4–8.5): an affine chart needs 7 of 8 coordinates on HIRES and 16 of 20 on POLLU although the
  visited sets are about 3- and 7-dimensional (projecting a curved set onto a flat subspace moves the state in slow directions); a
  polynomial decoder fixes HIRES (lift error 2.45e-4 at $k = 3$).
- **A reduced ChebTT** through that decoder took 9,477 solves for a one-step error of 2.5e-3 and a rollout of 0.30 dex; a solve-free
  least-squares fit on the training pairs did better (degree 6: 6.4e-4 and 1.5e-2; degree 10: 1.5e-4 over all 40 test rollouts).
- **The gradient tier** (section G, `g1`–`g7`): Chebyshev–Hermite interpolation halves the points per decade (ratios 1.97–2.01);
  gradient-enhanced least squares needs several times fewer random points; on designed grids derivatives added nothing measurable; a
  forward-sensitivity solve cost 28–87 plain solves in Python (central differences cost 16).
- **The HIRES recipe** (chart, decoder, value-only fit, first-order normal correction $B(u)(z - D(u))$, refusal by the decoder residual):
  rollouts within the tolerance on the visited set; under a kick at every step (an operator-split proxy) no tested fit held the
  tolerance (best 2.3e-2). The correction is second order with exact pieces and first order with fitted ones.

### 3.3 The recipe across the panel fails (R2; `theory/recipe/r2_step_maps.py`, predictions `a42863b`)

The HIRES recipe, generalised (held inputs as chart coordinates, floor clip, $k$ by lift error, degree by the chained error on the
training trajectories), held the tolerance on **4 of 37** step maps: the three 2-D ladders and HIRES. On the others it passed its own
check (training-chain errors near 1e-6) and erred 0.03–5 on test. S1 and S3 refuted, S2 and S5 confirmed.

### 3.4 Diagnosis (`theory/reviews/recipe-round-1-sol.md`, `theory/panel/p1_descriptors.py`; post hoc)

- **In-sample selection** cannot see a fit that is wild between sparse training states.
- **No reduction on box-filling problems**: the step-0 states fill the initial law's box with one state per trajectory.
- **A chart clamp hid extrapolation**: CHEMAKZO's 0.22 came from 3 of 1,600 test inputs outside the chart box; refusing them gave 2.4e-6.
- **The Lebesgue function predicts it**: over the test inputs, $\Lambda_X$ stays O(10) as the degree rises on HIRES and the test error
  falls with the residual (to 3e-6 at degree 12); on `fisher_kpp_d3` it reaches 4.6e9 at degree 12 while the test error rises to O(1).
  This is Proposition 5.1's regime (a) against regime (b), measured, and the reason T5 was generalised.

### 3.5 The generic recipe (R4, p2, R5; `theory/panel/generic.py`, predictions `298dabf`, `fff5107`, `bd86a36`)

Fold-fitted transforms, three grouped held-out splits, no clamp, a refusal rule from Proposition 5.1 (refuse when the Lebesgue bound
exceeds $\mathrm{tol}/(2e_v) - 1$), a minimum-norm invariant projection, solver fallback.

- **R4** (37 step maps): served (solver share below 50%) on `allen_cahn_d2`, `bruss_d2`, `lorenz96_d4`, `chemakzo`, `hires`, all within
  the tolerance on every test rollout (2e-6 to 5e-4); every other problem refused every input. Under chained mixing HIRES erred 0.38.
- **p2** (refusal and speed): effective speed-ups 2,311x (`bruss_d2`), 876x, 79x (`chemakzo`), 10x (`lorenz96_d4`, the gate's own cost
  and 5.8% refused), 3.1x (`hires`, 31% refused), 1.4x (`allen_cahn_d3`, 73% refused). The refusal rate and the gate's cost dominate.
- **R5** (400 and 1,600 extra one-step samples at fresh starts): `lorenz96_d5`, `burgers_d10`, `burgers_d20` became served (rollouts
  5e-6 to 6e-5); the gain saturates by 400.

### 3.6 Round 10 corrects the reading (`theory/reviews/round-10-fable.md`; post hoc checks)

- **The refusal threshold was wrong.** It put the held-out error (which already contains $1 + \Lambda$) where Proposition 5.1 has the
  best approximation error, against a bound that is at least 1 and whose supremum is at least $\sqrt p$ for any design (Kiefer–Wolfowitz).
  With a held-out-calibrated threshold, 7 of 9 re-checked problems held the tolerance with at most 2.5% solved (post hoc).
- **"Approximation-limited" was not in the records.** On Fisher–KPP the worst held-out state is always a first-step state whose bound
  explodes (sampling); the in-sample worst keeps falling 2.5–3x per degree. Allen–Cahn's floor is the chart's own lift error. The claim
  was withdrawn.
- **"Served" was true in every record**; unserved problems reported degree 1.
- **The gate is blind across the chart's fibres**: on HIRES, the error of accepted mixed states tracks the quartic decoder's residual
  with rank correlation 0.99–1.00; refusing beyond the training residual cut the accepted worst from 1.2e-2 to 2.2e-3.

### 3.7 The second version (R4b; `theory/panel/generic2.py`, predictions `0f84031`)

A held-out-calibrated gate (scores $\mathrm{err}/(1 + b)$, the 99% quantile; the largest held-out bound as the validated range), the
residual term, "served" fixed before the run, degrees to 12 with early stopping, the per-factor diagnosis per degree.

**Results** (`theory/panel/data/records/r4b/`, `data/r4b_calibrated.json`; tolerance 0.01; "served": held-out chains within the
tolerance and a solver share below 50%, fixed before the run). For a not-served problem the row shows the solver alone (refused 100%) or
the best configuration's numbers.

| problem | d | k | q | served | rollout, 40 / 20 test | refused % | chained mixing 0.01 / 0.1 (solver %) | without residual 0.01 / 0.1 | speed-up eff. / nominal |
|---|---|---|---|---|---|---|---|---|---|
| allen_cahn_d2 | 2 | 2 | 10 | yes | 1.7e-05 / 1.1e-05 | 0.0 | 9.3e-05 / 1.6e-05 (0 / 89) | – (full chart) | 307.8 / 1336 |
| bruss_d2 | 2 | 2 | 11 | yes | 1.2e-08 / 1.2e-08 | 7.0 | 5.6e-08 / 4.3e-08 (35 / 68) | – (full chart) | 14.0 / 3908 |
| fisher_kpp_d2 | 2 | 2 | 11 | yes | 3.3e-04 / 3.3e-04 | 0.0 | 1.5e-02 / 1.3e-02 (0 / 0) | – (full chart) | 396.3 / 1781 |
| van_der_pol | 2 | 2 | 10 | no | 0.0e+00 / 0.0e+00 | 100.0 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 1.0 / 30571 |
| allen_cahn_d3 | 3 | 3 | 5 | yes | 8.7e-03 / 3.4e-03 | 0.0 | 5.3e-03 / 2.0e-03 (0 / 0) | – (full chart) | 287.1 / 1093 |
| fisher_kpp_d3 | 3 | 3 | 9 | yes | 2.5e-02 / 2.5e-02 | 0.6 | 2.4e-02 / 1.4e-02 (0 / 1) | – (full chart) | 52.3 / 530 |
| oregonator | 3 | 3 | 5 | no | 0.0e+00 / 0.0e+00 | 100.0 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 1.0 / 38731 |
| robertson | 3 | 3 | 3 | yes | 1.3e-03 / 5.0e-04 | 0.0 | 1.2e-03 / 5.3e-04 (0 / 0) | – (full chart) | 42.9 / 87 |
| bruss_d4 | 4 | 4 | 5 | yes | 6.3e-03 / 6.3e-03 | 0.4 | 6.2e-03 / 4.5e-03 (0 / 0) | – (full chart) | 151.4 / 1731 |
| lorenz96_d4 | 4 | 4 | 4 | yes | 2.1e-03 / 7.6e-04 | 0.0 | 1.7e-03 / 6.6e-04 (0 / 0) | – (full chart) | 118.4 / 453 |
| allen_cahn_d5 | 5 | 5 | 3 | no | 4.2e-02 / 4.0e-02 | 6.3 | 6.7e-02 / 2.8e-02 (6 / 4) | – (full chart) | 15.2 / 1362 |
| chemakzo | 5 | 5 | 3 | yes | 1.3e-02 / 5.2e-03 | 0.0 | 2.1e-02 / 1.4e-02 (0 / 10) | – (full chart) | 480.0 / 2036 |
| fisher_kpp_d5 | 5 | 5 | 5 | yes | 1.3e-02 / 5.2e-03 | 3.1 | 1.3e-02 / 8.8e-03 (3 / 3) | – (full chart) | 23.2 / 557 |
| lorenz96_d5 | 5 | 5 | 4 | yes | 3.6e-03 / 1.8e-03 | 0.0 | 2.9e-03 / 1.4e-03 (0 / 0) | – (full chart) | 66.9 / 320 |
| bruss_d6 | 6 | 6 | 4 | yes | 7.0e-03 / 7.0e-03 | 2.8 | 6.3e-03 / 3.7e-03 (3 / 2) | – (full chart) | 30.6 / 1217 |
| hires | 8 | 3 | 9 | yes | 1.2e-03 / 4.3e-04 | 0.0 | 2.2e-01 / 4.4e-03 (47 / 70) | 1.3e+00 / 4.3e-01 | 20.5 / 37 |
| allen_cahn_d10 | 10 | 6 | 1 | no | 0.0e+00 / 0.0e+00 | 100.0 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 1.0 / 3028 |
| bruss_d10 | 10 | 9 | 4 | yes | 8.1e-03 / 5.8e-03 | 2.6 | 7.7e-03 / 5.6e-03 (3 / 2) | 7.7e-03 / 5.0e-03 | 16.5 / 384 |
| burgers_d10 | 10 | 6 | 3 | yes | 2.0e-03 / 2.0e-03 | 0.9 | 1.9e-03 / 1.9e-03 (1 / 0) | 1.9e-03 / 1.7e-03 | 27.4 / 59 |
| fisher_kpp_d10 | 10 | 6 | 5 | yes | 4.9e-03 / 4.9e-03 | 6.9 | 2.8e-03 / 3.6e-03 (6 / 6) | 2.8e-03 / 2.8e-03 | 10.5 / 264 |
| lorenz96_d10 | 10 | 10 | 4 | yes | 9.7e-03 / 8.7e-03 | 2.0 | 9.2e-03 / 8.5e-03 (2 / 1) | – (full chart) | 3.8 / 43 |
| primordial | 10 | 6 | 2 | no | 0.0e+00 / 0.0e+00 | 100.0 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 1.0 / 721 |
| nelson | 18 | 12 | 1 | no | 0.0e+00 / 0.0e+00 | 100.0 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 1.0 / 8576 |
| allen_cahn_d20 | 20 | 6 | 1 | no | 0.0e+00 / 0.0e+00 | 100.0 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 1.0 / 1984 |
| bruss_d20 | 20 | 9 | 4 | yes | 7.1e-03 / 5.6e-03 | 2.6 | 6.4e-03 / 3.8e-03 (3 / 2) | 6.4e-03 / 4.8e-03 | 12.7 / 52 |
| burgers_d20 | 20 | 6 | 3 | yes | 2.6e-03 / 2.6e-03 | 0.8 | 2.5e-03 / 2.7e-03 (1 / 0) | 2.5e-03 / 2.9e-03 | 21.1 / 41 |
| fisher_kpp_d20 | 20 | 7 | 4 | yes | 3.2e-03 / 2.8e-03 | 6.8 | 4.4e-03 / 2.1e-03 (6 / 6) | 4.4e-03 / 1.8e-03 | 11.0 / 281 |
| lorenz96_d20 | 20 | 20 | 3 | yes | 1.6e-02 / 1.6e-02 | 7.2 | 1.4e-02 / 1.3e-02 (7 / 5) | – (full chart) | 1.2 / 12 |
| pollu | 20 | 20 | 2 | no | 1.2e-02 / 1.2e-02 | 10.2 | 0.0e+00 / 0.0e+00 (100 / 100) | – (full chart) | 5.8 / 57 |
| allen_cahn_d40 | 40 | 6 | 3 | no | 4.9e-02 / 3.2e-02 | 6.5 | 5.9e-02 / 2.7e-02 (6 / 5) | 5.9e-02 / 2.6e-02 | 10.7 / 59 |
| bruss_d40 | 40 | 9 | 4 | yes | 7.3e-03 / 5.3e-03 | 2.6 | 6.6e-03 / 3.4e-03 (3 / 3) | 6.6e-03 / 4.5e-03 | 15.8 / 70 |
| burgers_d40 | 40 | 6 | 3 | yes | 2.9e-03 / 2.9e-03 | 0.9 | 2.8e-03 / 3.3e-03 (1 / 0) | 2.8e-03 / 3.4e-03 | 41.4 / 133 |
| fisher_kpp_d40 | 40 | 7 | 4 | yes | 3.0e-03 / 2.8e-03 | 6.8 | 4.0e-03 / 2.3e-03 (6 / 6) | 4.0e-03 / 2.0e-03 | 9.0 / 50 |
| lorenz96_d40 | 40 | 40 | 2 | no | 2.0e-02 / 1.6e-02 | 31.5 | 1.5e-02 / 9.7e-03 (28 / 15) | – (full chart) | 2.5 / 114 |
| allen_cahn_d80 | 80 | 6 | 3 | no | 4.8e-02 / 3.2e-02 | 6.6 | 5.8e-02 / 2.7e-02 (6 / 5) | 5.8e-02 / 2.6e-02 | 6.0 / 19 |
| bruss_d80 | 80 | 9 | 5 | yes | 6.6e-03 / 5.0e-03 | 2.6 | 4.6e-03 / 2.4e-03 (3 / 3) | 4.6e-03 / 2.7e-03 | 3.9 / 11 |
| fisher_kpp_d80 | 80 | 7 | 4 | yes | 3.0e-03 / 2.7e-03 | 6.8 | 3.8e-03 / 2.1e-03 (6 / 6) | 3.8e-03 / 2.2e-03 | 2.8 / 7 |

**Verdicts against the committed predictions (`0f84031`).**
- C1 holds: 26 of 37 step maps served (R4: 5), all ten listed among them.
- C2 fails on 4 of 26: `fisher_kpp_d3` (2.5e-2), `chemakzo` (1.25e-2), `fisher_kpp_d5` (1.3e-2), `lorenz96_d20` (1.6e-2) pass the
  held-out chains and miss on the 40 test trajectories (two of them hold on the benchmark's first 20). A maximum over 20 held-out
  trajectories per split does not bound the maximum over 40 new ones: T4's point about what samples certify, now inside the selection.
- C3 holds on 11 of the 12 served problems with a reduced chart and fails on `hires` (0.22 at w = 0.01 with 47% solved); the residual
  term does what it was for (without it `hires` errs 1.34 and 0.43; with it 0.22 and 4.4e-3) but does not stop chained error growth
  through accepted steps.
- C4 holds for the step maps (Oregonator, Van der Pol, POLLU, Nelson, `allen_cahn_d10` not served); KIDA below.
- C5 holds: the median rollout solver share over the served problems is 1.4% and the median effective speed-up 22x.

**What changed against R4, and what is new.**
- The calibrated gate serves 26 problems where the worst-case threshold served 5; most held-out maxima were sampling, not failures of
  the fit (round 10's diagnosis, now pre-registered).
- The price: on some problems the gate accepts more and errs more (`chemakzo` 4.8e-5 at 0.4% refused in R4, 1.25e-2 at 0% here);
  calibration on a maximum over finite held-out chains is a distributional statement.
- **Mixing on full charts.** Five served problems with full charts (no residual term applies) fail chained mixing while refusing almost
  nothing: `fisher_kpp_d2`, `fisher_kpp_d3`, `fisher_kpp_d5`, `chemakzo`, `lorenz96_d20`. The gate is calibrated on undisturbed held-out
  states and its calibration does not transfer to mixed ones: the known limit of conformal-type calibration under covariate shift
  (V-40). Calibrating on the deployment's own inputs (mixed states) is the direct test.
- **The gate's cost sets the speed at high dimension.** Effective speed-ups fall to 2.8–4x at $d$ = 80 and on `lorenz96_d10`
  (1,001 terms), and to 1.2x on `lorenz96_d20`, with refusal rates of only 2–7%: the exact $O(p^2)$ bound costs about as much as the
  solve. A cheaper gate (a sketched bound, a box pre-test) is the lever there.
- **The existing schemes on the same problems** are tabulated in `theory/panel/data/scheme_table.md` (`p3_scheme_table.py`; the
  benchmark's 20-trajectory rollout). On the small ladders `chebtt_roll` holds 2e-3 to 7e-3 rollouts with 300–20,000 solves and
  speed-ups in the thousands; where both hold, the chart recipe needs no solves beyond its lift check but pays the gate's cost; on
  the $d \ge 20$ ladders and the chemistry problems most existing schemes have no finished record at 0.01.

### 3.8 KIDA (`theory/recipe/r3_kida.py`)

Two environment ranges set up like Nelson (a 66.7 kyr step, a driven environment, burn-in 1e4 yr, 24 steps): *diffuse* (Nelson's
ranges; 28–57 species above 1e-12) and *dense* (n_H 1e3–1e5, T 10–50 K, A_v 3–10; 120–128 species above 1e-12). A dense step costs
about 0.2 s and a burn-in 3.6–10 s per tracer on one core. The dense trajectories (200 training, 40 test) are complete; the first
recipe's lift check stalled on decoded states 3–7 dex off the data and was stopped; the diffuse trajectories are being generated
(some tracers take minutes: 3,000 K and strong UV make the chemistry far stiffer).

**Dense KIDA in R4b** (`theory/panel/data/records/r4b/kida_dense.json`; 200 training and 40 test tracers, 578 species and 4 held
environment inputs). The lift check found no reduction: at every k from 1 to 20 every decoder (affine, quadratic, quartic) put some
decoded species more than a decade from its input, so the guard scored the lift infinite without a solve, and the rule fell back to an
unvalidated k = 1. On that chart (1 coordinate plus the 4 environment inputs) the best held-out configuration erred 298
dex at its worst and 2.2 dex at the median (degree 3): with 578 outputs the fit is far from usable,
and the gate refused every input, so the solver does every step (not served, C4 holds for KIDA). The case took 1578 s on CPU
for selection. What it says: linear charts with polynomial lifts do not reduce KIDA at 0.01 dex; whether a nonlinear lift can (proposal
S4) is the open question for the target class. The diffuse range was stopped: its environments (T up to 3,000 K, strong UV) lie outside
the range KIDA's rate coefficients are fitted for, and single tracers took more than 15 minutes.

### 3.9 After the council (round 1): what was adopted

The three reviews (`theory/reviews/council-{fable,sol,astra}.md`) agreed on the order: repair the KIDA reference comparison first, put the
deployment test (a split replay with neighbours carrying surrogate error) before any model search, keep rollout-tuned ChebTT and the
chart as controls, run one bounded MLP against linear and local fits on equal data, compare designed data at equal cost, and stop
full-panel reruns, architecture surveys and unconditional AFINNI/ART sweeps. Adopted, in that order:

- **KIDA's projection** (Astra's finding: the deployed invariant projection, handed the solver's own answer, missed 0.01 on every state,
  by up to 293 dex). Cause: the numerical null-space search found 373 spurious laws on KIDA's sampled states. Fix in `src` (approved):
  a system can declare its conservation laws (`System.conserved`), `invariants_of` prefers them, and KIDA's 15 exact laws (13 elements,
  charge, grains) come from parsing its species names (`src/problems/ode/astro.py`, tests in `tests/unit/test_astro_conservation.py`).
  The projection now round-trips exact labels within 1.2e-9 dex.
- **KIDA's chemistry.** The dense range's median HCO+ of 1e-4 was a setup bug: KIDApy leaves electrons to the caller and the loader left
  them at zero, so every state carried a net charge the network conserves. The *dark* range sets electrons from charge neutrality and
  uses standard dark-cloud conditions; its states are physically plausible (`theory/KIDA_QOI.md`). Nelson starts the same way; it is
  flagged for the final rerun, since changing it changes benchmark data.
- **KIDA's key species** (the user's choice after a literature search): 13 species for cooling (H2, CO, C, C+, O, OH, H2O), ionisation
  (e-, H3+, HCO+, N2H+, S+) and a dense-gas tracer (HCN), at 0.01 dex where the abundance exceeds 1e-12; the strict all-species metric
  is reported beside it.
- **The third recipe** (`theory/panel/generic3.py`): the gate calibrated for the *maximum* on calibration states set apart from the
  selection's (each split's held-out fifth halved into calibration and selection trajectories), mixed inputs in the calibration, a
  propagation margin (served only if the worst held-out chain is within half the tolerance), and k up to the free dimension.

### 3.10 The third recipe on the discriminating cases (R4c; `theory/panel/r4c_rules.py`, predictions `305e0c4`)

Seventeen problems at tolerance 0.01, M = 0: the chemistry and oscillator class, R4b's four selection-test misses and full-chart mixing
failures, a propagation failure, and four controls. Timings are single measurements taken while other runs shared the
machine; read them as orders of magnitude.

| problem | d | k | degree | served | e_hat | refused (test) | rollout, 40 | chained mixing 0.01 / 0.1 (solver) | effective speed-up | predict / gate / solve (us) | train (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| allen_cahn_d5 | 5 | 5 | 3 | no | 3.3e-04 | 6.8% | 4.2e-02 | 6.0e-02 / 2.8e-02 (6 / 4%) | 14.3 | 1 / 2 / 1037 | 19 |
| bruss_d2 | 2 | 2 | 12 | yes | 3.0e-10 | 0.0% | 1.1e-09 | 4.6e-04 / 1.5e-03 (2 / 1%) | 753.5 | 1 / 3 / 2629 | 6 |
| bruss_d20 | 20 | 9 | 4 | yes | 2.2e-04 | 3.8% | 3.3e-03 | 2.5e-03 / 3.3e-03 (3 / 3%) | 13.0 | 21 / 43 / 1589 | 58 |
| burgers_d20 | 20 | 6 | 5 | yes | 2.6e-08 | 4.1% | 1.6e-03 | 2.3e-03 / 1.9e-03 (4 / 3%) | 14.1 | 12 / 23 / 1131 | 11 |
| chemakzo | 5 | 5 | 9 | yes | 3.6e-09 | 0.7% | 1.3e-04 | 2.0e-04 / 3.8e-05 (89 / 99%) | 0.7 | 10 / 1642 / 1186 | 855 |
| fisher_kpp_d2 | 2 | 2 | 12 | yes | 1.6e-06 | 0.0% | 1.3e-04 | 3.0e-03 / 5.5e-04 (0 / 0%) | 109.0 | 2 / 7 / 932 | 7 |
| fisher_kpp_d3 | 3 | 3 | 11 | yes | 1.5e-08 | 1.1% | 6.6e-03 | 6.5e-03 / 9.3e-03 (1 / 1%) | 32.2 | 2 / 14 / 804 | 18 |
| fisher_kpp_d5 | 5 | 5 | 8 | yes | 6.0e-06 | 20.1% | 7.8e-04 | 1.2e-03 / 3.4e-04 (19 / 18%) | 1.3 | 9 / 525 / 952 | 651 |
| hires | 8 | 3 | 12 | no | 3.3e-03 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 | 14 / 27 / 938 | 55 |
| lorenz96_d20 | 20 | 20 | 2 | no | 7.6e-04 | 62.9% | 2.4e-02 | 2.3e-02 / 8.0e-03 (45 / 20%) | 1.5 | 2 / 7 / 156 | 28 |
| lorenz96_d4 | 4 | 4 | 7 | yes | 6.3e-10 | 0.0% | 2.7e-05 | 2.5e-05 / 2.2e-05 (0 / 0%) | 14.0 | 5 / 14 / 268 | 16 |
| nelson | 18 | 12 | 1 | no | 1.2e-01 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 | 12 / 14 / 78048 | 31 |
| oregonator | 3 | 3 | 9 | no | 5.9e-01 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 | 1 / 5 / 7885 | 18 |
| pollu | 20 | 13 | 3 | no | 5.4e-06 | 11.9% | 1.3e-02 | 4.5e-03 / 0.0e+00 (99 / 100%) | 6.0 | 27 / 40 / 1373 | 144 |
| primordial | 10 | 6 | 2 | no | 3.3e-02 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 | 14 / 15 / 8351 | 42 |
| robertson | 3 | 3 | 6 | yes | 1.2e-09 | 0.0% | 1.3e-05 | 7.3e-06 / 2.3e-05 (0 / 0%) | 32.4 | 18 / 19 / 1218 | 28 |
| van_der_pol | 2 | 2 | 10 | no | 2.7e+02 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 | 0 / 1 / 6533 | 18 |

A rollout of 0 with 100% refused means the solver took every step.

Pre-registered readings: **V1 holds** (no served problem misses the tolerance on its 40 test rollouts; the worst is fisher_kpp_d3's
6.6e-3, which R4b missed at 2.5e-2); **V2 holds** (every served problem holds chained mixing at both weights; the worst is fisher_kpp_d3's
9.3e-3); **V3 fails** (9 of 17 served, the four controls among them, but not HIRES); **V4 holds** (POLLU, Nelson, the Oregonator, Van der
Pol and primordial are not served); **V5 fails** (the median effective speed-up over the served problems is 14, predicted at least 20).

- **HIRES is refused whole although its fit is accurate**: the ungated first-step worst is 1.1e-6, but e_hat (the largest calibration
  score) is 3.3e-3, set by mixed calibration states off its thin manifold, which the residual term refuses anyway; section 3.16 tests
  the fix.
- **CHEMAKZO is served at an effective speed-up of 0.7**, slower than the solver: the rule takes the smallest worst chain among served
  configurations (degree 9: 1.3e-3, a 24% solver share in the selection), not the fastest (degree 8: 2.8e-3, 5%). Its selection took
  1,287 s (degrees 8-10 cost 4-14 minutes each in solver fallbacks); a 1,800 s limit stopped the first two attempts.
- **The gate's cost sets the speed where the fit is cheap**: fisher_kpp_d5's gate costs 525 us against a 9 us prediction and a 952 us
  solve, so with 20% refused its effective speed-up is 1.3; CHEMAKZO's gate (1,642 us) costs more than its solve (1,186 us).
- Primordial and Nelson are coarse on plain states already (e_hat 3.3e-2 and 8.6e-2 at degrees 2 and 1, the largest the data
  allow); POLLU fails on a chain of 1.25e-2; the oscillators are refused whole (their approximation limit); lorenz96_d20 refuses 63%.

### 3.11 Rollout-tuned ChebTT under chained mixing (p4; `theory/panel/p4_chebtt_mixing.py`, prediction `59c582c`)

`chebtt_roll`, rebuilt exactly as the benchmark builds it on the 15 problems where its benchmark record holds the tolerance, holds the
tolerance under chained mixing at both weights on all 15, with solver shares within a point of zero (Fable's prediction: at least 13).
A convex mixture of in-box states stays in its box, so mixing is not off its data. This made it the control for the split replay.

### 3.12 The split replay: a scripted column (p6; `theory/panel/p6_column.py`, predictions `1ac8a5a`)

Chained mixing (sections 3.7, 3.11) mixes each cell with a *reference* partner and so pulls it toward the truth; no surrogate error feeds
back through the coupling (Sol's objection). The column removes that: the test trajectories are cells of a 1-D column (ordered by their
first principal coordinate), each step is conservative transport between neighbours, x_i <- (1 - 2D) x_i + D (x_{i-1} + x_{i+1}) in the
system's own variables, then chemistry; each cell's environment follows its own path, and the hybrid column (surrogate, solver on refused
cells) is compared with the solver-only column. D = 0.005 and 0.05 give the same exchange per step as the mixing weights 0.01 and 0.1.

| problem | surrogate | column D = 0.005 / 0.05 | chained mixing 0.01 / 0.1 | ratio at the larger exchange | column solver share | own-input defect | invariant drift (relative) |
|---|---|---|---|---|---|---|---|
| allen_cahn_d10 | chebtt_roll | 3.0e-03 / 5.0e-03 | 4.1e-03 / 1.7e-03 | 3.0 | 0.0 / 0.0% | 1.1e-04 | – |
| allen_cahn_d2 | chebtt_roll | 8.5e-03 / 3.4e-03 | 6.8e-03 / 9.0e-04 | 3.8 | 0.0 / 0.0% | 1.1e-04 | – |
| allen_cahn_d3 | chebtt_roll | 3.3e-03 / 3.1e-03 | 3.2e-03 / 2.3e-04 | 13.2 | 0.0 / 0.0% | 4.7e-05 | – |
| allen_cahn_d5 | chebtt_roll | 1.1e-03 / 1.7e-03 | 3.0e-03 / 4.8e-04 | 3.4 | 0.0 / 0.0% | 6.0e-05 | – |
| bruss_d20 | recipe | 2.6e-03 / 3.0e-03 | 2.5e-03 / 3.3e-03 | 0.9 | 3.5 / 3.1% | 3.0e-03 | 2.6e+02 |
| bruss_d2 | chebtt_roll | 2.7e-03 / 3.3e-03 | 2.9e-03 / 3.0e-03 | 1.1 | 0.0 / 0.0% | 7.5e-04 | – |
| bruss_d2 | recipe | 2.3e-04 / 2.2e-04 | 4.6e-04 / 1.5e-03 | 0.1 | 0.0 / 0.0% | 2.2e-04 | – |
| bruss_d4 | chebtt_roll | 2.5e-03 / 2.1e-03 | 2.3e-03 / 1.9e-03 | 1.1 | 0.0 / 0.0% | 5.5e-04 | – |
| bruss_d6 | chebtt_roll | 2.8e-03 / 2.6e-03 | 2.6e-03 / 1.9e-03 | 1.4 | 0.0 / 0.0% | 3.8e-04 | – |
| burgers_d20 | recipe | 1.5e-03 / 2.1e-03 | 2.3e-03 / 1.9e-03 | 1.1 | 4.1 / 3.5% | 2.1e-03 | 9.6e-13 |
| chemakzo | chebtt_roll | 5.0e-03 / 4.6e-03 | 3.7e-03 / 1.7e-03 | 2.8 | 0.0 / 0.0% | 2.5e-04 | – |
| fisher_kpp_d2 | chebtt_roll | 4.8e-03 / 4.8e-03 | 2.7e-03 / 1.8e-03 | 2.7 | 0.0 / 0.0% | 5.5e-04 | – |
| fisher_kpp_d2 | recipe | 6.7e-04 / 7.1e-04 | 3.0e-03 / 5.5e-04 | 1.3 | 0.0 / 0.1% | 7.1e-04 | – |
| fisher_kpp_d3 | chebtt_roll | 2.5e-03 / 2.3e-03 | 2.5e-03 / 1.9e-03 | 1.2 | 0.0 / 0.0% | 6.8e-04 | – |
| fisher_kpp_d3 | recipe | 6.5e-03 / 5.4e-03 | 6.5e-03 / 9.3e-03 | 0.6 | 1.0 / 0.9% | 6.5e-03 | – |
| fisher_kpp_d5 | chebtt_roll | 3.4e-03 / 1.6e-03 | 3.1e-03 / 1.4e-03 | 1.1 | 0.0 / 0.0% | 6.9e-04 | – |
| fisher_kpp_d5 | recipe | 7.6e-04 / 5.4e-04 | 1.2e-03 / 3.4e-04 | 1.6 | 19.1 / 18.1% | 7.2e-04 | – |
| lorenz96_d10 | chebtt_roll | 6.7e-03 / 6.6e-03 | 6.1e-03 / 1.8e-03 | 3.7 | 0.3 / 0.0% | 6.0e-04 | – |
| lorenz96_d4 | chebtt_roll | 3.9e-03 / 3.8e-03 | 3.6e-03 / 1.9e-03 | 2.0 | 0.8 / 0.5% | 8.1e-04 | – |
| lorenz96_d4 | recipe | 2.4e-05 / 8.4e-06 | 2.5e-05 / 2.2e-05 | 0.4 | 0.0 / 0.0% | 2.4e-05 | – |
| lorenz96_d5 | chebtt_roll | 2.5e-03 / 2.3e-03 | 2.5e-03 / 1.7e-03 | 1.3 | 0.2 / 0.1% | 1.0e-03 | – |
| robertson | chebtt_roll | 4.1e-03 / 4.0e-03 | 3.5e-03 / 1.1e-03 | 3.7 | 0.0 / 0.0% | 1.8e-04 | 5.8e-03 |
| robertson | recipe | 8.1e-06 / 1.0e-05 | 7.3e-06 / 2.3e-05 | 0.4 | 0.0 / 0.0% | 1.0e-05 | 2.3e-15 |

Pre-registered readings: **S1 holds in its first clause and fails in its second**: every audited surrogate holds 0.01 in the column at both
exchanges (the largest worst is 8.5e-3, allen_cahn_d2 with ChebTT at D = 0.005), but at the larger exchange the column's worst is
2.7-13x the chained-mixing worst for ChebTT on the Allen-Cahn ladders, CHEMAKZO, fisher_kpp_d2, Robertson and lorenz96_d10: mixing with a
reference partner pulls a cell's error back toward the truth, and the column does not. **S2 holds** (solver shares as under mixing).
**S3 fails once**: the recipe's own-input defect on fisher_kpp_d3 is 6.5e-3, above half the tolerance (its gate is calibrated on 20
trajectories' states and their mixtures, and the column's inputs are neither). **S4 holds where the laws are real** (the recipe's drift
3e-15 on Robertson, ChebTT's 5.8e-3); on bruss_d20 the recipe's relative drift is 38-265 because the law it projects onto is spurious
(section 9.7, item 7).


### 3.13 Data design at equal cost (r6; `theory/panel/r6_design.py`, predictions `5f0fa47`)

Sol's controlled comparison: the third recipe with 0, 400 random (three draws), 400 chosen by sequential maximum leverage (Proposition
5.1's bound, no solves) and all 1,600 fresh one-step pairs, everything else fixed.

| problem | arm | extra | served | k / degree | first step: ungated worst / median, refused | refused (test) | rollout, 40 | chained mixing 0.01 / 0.1 | effective speed-up | train (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| allen_cahn_d10 | all1600  | 1600 | yes | 6 / 7 | 3.7e+00 / 2.6e-03, 82% | 5.2% | 3.7e-03 | 3.0e-03 / 3.9e-03 | 4.1 | 60 |
| allen_cahn_d10 | leverage400  | 400 | yes | 6 / 7 | 3.8e+00 / 2.4e+00, 100% | 21.8% | 4.1e-03 | 5.9e-03 / 2.9e-03 | 0.5 | 201 |
| allen_cahn_d10 | m0  | 0 | no | 6 / 3 | 6.9e-02 / 3.6e-03, 100% | 7.8% | 4.9e-02 | 6.2e-02 / 2.7e-02 | 12.3 | 10 |
| allen_cahn_d10 | random400_s0  | 400 | yes | 6 / 5 | 3.9e-01 / 1.7e-03, 80% | 3.4% | 8.1e-03 | 7.6e-03 / 4.7e-03 | 15.6 | 217 |
| allen_cahn_d10 | random400_s1  | 400 | yes | 6 / 5 | 1.2e-01 / 2.1e-03, 68% | 2.5% | 7.3e-03 | 7.1e-03 / 5.8e-03 | 17.7 | 228 |
| allen_cahn_d10 | random400_s2  | 400 | no | 6 / 3 | 5.1e-03 / 1.6e-03, 85% | 3.5% | 5.6e-02 | 4.6e-02 / 2.7e-02 | 26.0 | 23 |
| burgers_d10 | all1600  | 1600 | yes | 6 / 6 | 5.2e-03 / 1.5e-04, 18% | 0.6% | 8.8e-04 | 8.2e-04 / 7.1e-04 | 20.3 | 50 |
| burgers_d10 | leverage400  | 400 | yes | 6 / 6 | 3.1e+00 / 3.3e-01, 100% | 5.1% | 3.9e-03 | 4.4e-03 / 3.5e-03 | 10.9 | 167 |
| burgers_d10 | m0  | 0 | yes | 6 / 5 | 3.1e+00 / 4.4e-01, 100% | 4.6% | 2.9e-03 | 2.7e-03 / 3.1e-03 | 14.2 | 8 |
| burgers_d10 | random400_s0  | 400 | yes | 6 / 6 | 3.1e+00 / 3.4e-01, 100% | 4.9% | 2.5e-03 | 2.3e-03 / 2.3e-03 | 10.5 | 17 |
| burgers_d10 | random400_s1  | 400 | yes | 6 / 6 | 3.1e+00 / 2.9e-01, 100% | 4.8% | 2.7e-03 | 2.5e-03 / 3.2e-03 | 10.7 | 188 |
| burgers_d10 | random400_s2  | 400 | yes | 6 / 6 | 3.1e+00 / 5.4e-01, 100% | 4.9% | 3.4e-03 | 2.9e-03 / 2.9e-03 | 10.5 | 18 |
| fisher_kpp_d10 | all1600  | 1600 | yes | 6 / 5 | 1.2e-01 / 7.1e-03, 100% | 15.0% | 1.4e-03 | 1.1e-03 / 1.3e-03 | 5.4 | 64 |
| fisher_kpp_d10 | leverage400  | 400 | yes | 6 / 7 | 4.9e+00 / 3.5e+00, 100% | 27.3% | 4.3e-04 | 1.5e-04 / 3.5e-05 | 0.5 | 240 |
| fisher_kpp_d10 | m0  | 0 | yes | 6 / 6 | 4.9e+00 / 3.4e+00, 100% | 17.4% | 1.2e-04 | 1.2e-04 / 8.2e-05 | 1.3 | 429 |
| fisher_kpp_d10 | random400_s0  | 400 | yes | 6 / 6 | 4.7e+00 / 2.9e-01, 100% | 5.1% | 4.2e-03 | 4.0e-03 / 2.3e-03 | 7.3 | 435 |
| fisher_kpp_d10 | random400_s1  | 400 | yes | 6 / 6 | 4.8e+00 / 3.0e-01, 100% | 5.2% | 2.5e-03 | 2.3e-03 / 2.7e-03 | 7.1 | 464 |
| fisher_kpp_d10 | random400_s2  | 400 | yes | 6 / 6 | 4.7e+00 / 4.4e-01, 100% | 5.3% | 2.0e-03 | 1.8e-03 / 1.8e-03 | 7.0 | 232 |
| fisher_kpp_d3 | all1600  | 1600 | yes | 3 / 11 | 5.0e-03 / 1.5e-04, 2% | 0.2% | 1.4e-03 | 2.1e-03 / 1.7e-03 | 41.1 | 16 |
| fisher_kpp_d3 | leverage400  | 400 | yes | 3 / 12 | 1.3e-02 / 1.5e-04, 5% | 0.1% | 3.8e-03 | 1.2e-03 / 2.8e-03 | 22.0 | 16 |
| fisher_kpp_d3 | m0 (R4c) | 0 | yes | 3 / 11 | 5.0e+00 / 4.2e-03, 45% | 1.1% | 6.6e-03 | 6.5e-03 / 9.3e-03 | 32.2 | 18 |
| fisher_kpp_d3 | random400_s0  | 400 | yes | 3 / 12 | 1.9e-01 / 5.6e-05, 10% | 0.2% | 1.2e-03 | 1.4e-03 / 1.4e-03 | 29.7 | 16 |
| fisher_kpp_d3 | random400_s1  | 400 | yes | 3 / 12 | 3.8e-01 / 7.9e-05, 8% | 0.2% | 1.7e-03 | 3.9e-03 / 3.6e-03 | 20.5 | 14 |
| fisher_kpp_d3 | random400_s2  | 400 | yes | 3 / 11 | 3.3e-01 / 1.0e-04, 10% | 0.3% | 1.0e-03 | 1.5e-03 / 1.5e-03 | 30.2 | 15 |
| hires | all1600  | 1600 | no | 3 / 11 | 1.8e-07 / 2.3e-08, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 65 |
| hires | leverage400  | 400 | no | 3 / 11 | 2.8e-07 / 6.3e-08, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 65 |
| hires | m0 (R4c) | 0 | no | 3 / 12 | 1.1e-06 / 5.5e-08, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 55 |
| hires | random400_s0  | 400 | no | 3 / 11 | 3.1e-07 / 7.6e-08, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 78 |
| nelson | all1600  | 1600 | no | 12 / 1 | 1.3e-01 / 5.4e-02, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 55 |
| nelson | leverage400  | 400 | no | 12 / 1 | 2.5e-01 / 1.1e-01, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 46 |
| nelson | m0 (R4c) | 0 | no | 12 / 1 | 5.2e-01 / 2.2e-01, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 31 |
| nelson | random400_s0  | 400 | no | 12 / 1 | 2.2e-01 / 8.3e-02, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 45 |
| pollu | all1600  | 1600 | no | 9 / 3 | 2.7e-03 / 8.3e-04, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 116 |
| pollu | leverage400  | 400 | no | 13 / 3 | 1.1e-03 / 6.4e-04, 2% | 53.9% | 7.3e-03 | 0.0e+00 / 0.0e+00 | 1.7 | 198 |
| pollu | m0 (R4c) | 0 | no | 13 / 3 | 1.1e-03 / 6.8e-04, 40% | 11.9% | 1.3e-02 | 4.5e-03 / 0.0e+00 | 6.0 | 144 |
| pollu | random400_s0  | 400 | no | 13 / 3 | 1.1e-03 / 6.4e-04, 12% | 49.6% | 6.7e-03 | 0.0e+00 / 0.0e+00 | 1.9 | 181 |
| primordial | all1600  | 1600 | no | 6 / 3 | 1.2e+00 / 1.7e-01, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 85 |
| primordial | leverage400  | 400 | no | 6 / 3 | 7.4e-01 / 2.0e-01, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 54 |
| primordial | m0 (R4c) | 0 | no | 6 / 2 | 1.1e+00 / 2.6e-01, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 42 |
| primordial | random400_s0  | 400 | no | 6 / 3 | 1.5e+00 / 1.4e-01, 100% | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 1.0 | 50 |

Pre-registered readings: **E1 fails**: leverage-chosen pairs beat the random draws' median first-step worst on 2 of 4 lattices (24x on
fisher_kpp_d3, 1.02x on burgers_d10); they are chosen in the fit space of the configuration selected without extra data, and on
fisher_kpp_d10 and allen_cahn_d10 the selection then moved to another degree, where they serve worse than random ones (refused 22-27%,
effective speed-up 0.5). **E2 holds in its first clause and fails in its second**: all 1,600 pairs serve every lattice within 0.01
(allen_cahn_d10 goes from not served to 3.7e-3; burgers_d10 2.9e-3 to 8.8e-4 with the speed-up from 14 to 20; fisher_kpp_d3 6.6e-3 to
1.4e-3, speed-up 32 to 41), but the first-step refused share stays at 68-100% on the three d = 10 lattices with 400 pairs. **E3 holds**
(HIRES, primordial, Nelson and POLLU are not served at any M; CHEMAKZO's arms were not run, since each would exceed the 1,800 s limit).
**E4 holds**.

- **The first step's data cost follows the initial law's dimension.** fisher_kpp_d10's first step does not move with any arm (ungated
  worst 4.7-4.9): its initial law fills all 10 dimensions, a test start lies 0.46 from its nearest pool state against 0.09 at d = 3, and
  the dynamics contract onto the k = 6 chart only after the first steps.


### 3.14 The KIDA screen (p5, p5b; `theory/panel/p5_kida_screen.py`, predictions `7885778`)

The council's first experiment on the target: five arms on equal data in the charge-neutral dark range (160 fit, 20 calibration, 20
selection tracers; 40 test tracers), scored on the 13 key species: persistence; a ridge fit linear in all 582 inputs; the chart's class
(degree 2 in 20 principal coordinates and the environment); a local ridge fit on the 64 nearest states; five MLPs (two hidden layers of
256, scikit-learn, early stopping). Each arm's gate is calibrated for the maximum on calibration states and their mixtures.

| arm | gate refuses (test) | ungated median, key species (dex) | ungated worst | test states above 0.01 | oracle coverage at 0.01 (p5b) | predict + gate (us/state) | train |
|---|---|---|---|---|---|---|---|
| persistence | 0% (never refuses) | 0.167 | 2.10 | 100% | 0% | 1 | 0 s |
| linear (582 inputs) | 100% | 0.034 | 1.90 | 96% | 3.6% | 29 + 28 | 1 s |
| chart, degree 2 | 100% | 0.028 | 1.73 | 93% | 7.1% | 60 + 40 | 1 s |
| local (64 nearest) | 100% | **0.022** | 1.99 | 83% | **17.4%** | 338 + 13 | 1 s |
| MLP ensemble | 100% | 0.081 | 2.09 | 99.9% | 0.1% | 103 + 99 | 18 s |

- **No arm serves anything** at 0.01 dex with 160 tracers: every calibrated gate refuses every input, and even a perfect gate (an
  oracle that accepts exactly the states within the tolerance) could serve at most 17% of the local arm's states (7% within half the
  tolerance). The solver reproduces the stored test trajectories to 2.5e-10 dex, so the reference is sound.
- **The MLP is the worst fitted arm** on this data (median 0.081 dex against the local fit's 0.022): 4,800 training pairs in 582 inputs
  do not train a network to 0.01 dex. This is the bounded MLP trial all three reviewers asked for, on CPU, at the data the project has.
- **Where the error sits** (p5b): the local arm's median is 1.2e-3 at the first step (the tracers share an initial composition after
  burn-in) and 1.9e-2 at the last, as the states spread with the driven environments; the misses are spread over S+, HCN, C and N2H+
  (41-57% of states each above 0.01; atomic carbon dominates the other arms), not one species. Coverage, not one hard output, is the limit.
- Pre-registered: K2 holds (no arm holds 0.01 with less than 50% refused); K4 holds vacuously (the gated arms accept nothing); K3 fails in
  its first clause (the MLP was predicted to have the smallest median and has the largest of the fitted arms) and holds in its second;
  K1 holds for persistence and is not measured directly for the linear fit.

### 3.15 Online local tables: a pilot that was not run (p7 smoke tests)

An ISAT-style local table (a local affine fit on the nearest stored pairs; frozen, growing, and cold arms; each query scored before its
answer is inserted) was written for the panel's hard cases and KIDA (`theory/panel/p7_online.py`, uncommitted). Smoke tests on HIRES
showed that its gate does not transfer from calibration states to chain states: three gate designs, each calibrated the same way (the
largest risk whose accepted calibration states err within half the tolerance), accepted chain states with one-step defects of 0.15-1.0
and chains of 0.5-4.9:

| gate (risk) | frozen: defect on own accepted inputs / chain worst / refused | online | cold |
|---|---|---|---|
| distance to the (m+1)-th neighbour | 0.15 / 2.4 / 2.3% | 0.21 / 2.1 / 1.5% | 0.20 / 3.0 / 11% |
| local Proposition 5.1: (1 + Lambda) x neighbour residual | 0.18 / 1.9 / 3.0% | 0.18 / 1.9 / 1.8% | 8.4e-3 / 0.52 / 17% |
| the same, and increment-fit against successor-fit disagreement | 0.25 / 2.4 / 5.5% | 0.57 / 2.3 / 1.3% | 1.0 / 4.9 / 7.0% |

The mechanism: a chain state drifts off its neighbours' span in a direction the step contracts; the increment fit carries the offset
forward, and none of the three risks sees it reliably. ISAT checks accuracy by solving as it grows; a table without that check is unsafe
here. The pilot was stopped and is a question for the reviewers (section 10).

### 3.16 Consistent calibration of the reduced-chart gate (r4d; `theory/panel/r4d_consistent.py`, predictions `d5e0124`)

A gate is a conjunction: a state the residual term refuses never needs the bound's term. With `consistent=True` (generic3), e_hat and
b_max come only from calibration states the residual term accepts. Post hoc on HIRES before the run: e_hat 4.7e-9 on plain calibration
states, 3.3e-3 on mixed ones, 1.9e-4 on the mixed states the residual term accepts (a third of them).

| problem | served | k / degree | e_hat | refused (test) | rollout, 40 | chained mixing 0.01 / 0.1 (solver) | effective speed-up |
|---|---|---|---|---|---|---|---|
| bruss_d20 | yes | 9 / 4 | 2.2e-04 | 3.8% | 3.3e-03 | 2.5e-03 / 3.3e-03 (3 / 3%) | 13.2 |
| burgers_d20 | yes | 6 / 5 | 2.6e-08 | 4.1% | 1.6e-03 | 2.3e-03 / 1.9e-03 (4 / 3%) | 12.1 |
| chemakzo | yes | 5 / 9 | 3.6e-09 | 0.7% | 1.3e-04 | 2.0e-04 / 3.8e-05 (89 / 99%) | 1.0 |
| fisher_kpp_d10 | yes | 6 / 6 | 1.6e-05 | 17.4% | 1.2e-04 | 1.2e-04 / 8.2e-05 (17 / 16%) | 2.0 |
| hires | yes | 3 / 12 | 1.9e-04 | 13.5% | 1.8e-06 | 1.6e-01 / 4.4e-03 (63 / 78%) | 5.7 |
| nelson | no | 12 / 1 | 8.6e-02 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 |
| pollu | no | 13 / 3 | 1.3e-06 | 0.0% | 4.4e-02 | 3.8e-02 / 3.4e-03 (82 / 100%) | 26.8 |
| primordial | no | 6 / 2 | 3.3e-02 | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 (100 / 100%) | 1.0 |

Pre-registered readings: **W1 fails**: HIRES is served (a rollout of 1.8e-6, 13.5% refused, effective speed-up 5.7, against R4c's
100% refused), but chained mixing at w = 0.01 errs 0.163. Post hoc, the accepted one-step defects on mixed inputs stay at most 2e-3,
within half the tolerance, and the chain still grows to 5e-2 within about 100 steps: small defects add up along directions the dynamics
do not contract. The selection's propagation margin, measured on plain held-out chains (1.8e-6 here), cannot see this. **W2, W3, W4
and W5 hold**: the other problems are unchanged (bruss_d20, burgers_d20, chemakzo and fisher_kpp_d10 served as before; primordial,
nelson and pollu not served; refused shares on the lattices unchanged).


### 3.17 After the council (round 2): what was adopted

The three reviews (`theory/reviews/council2-{fable,sol,astra}.md`; resolutions in `theory/REVIEWS.md`) corrected several readings of
section 9.7 and agreed on the order: repair the fit and the gate, test the tube with a control for data volume, redesign the coupled
test, choose configurations by speed, verify the conservation laws, audit KIDA's solver and measure its coverage before any more model
search on it, then the basis for larger charts. Corrections accepted:

- **9.7.1 withdrawn.** For rollout-tuned ChebTT the column is within 0.4-2.1x of the plain rollout; the 2.7-13x of section 3.12 was
  reference-partner mixing being an easier test (it damps each cell's error by w per step), and p6 also changed the donor geometry.
- **9.7.4 withdrawn.** fisher_kpp_d10's 1,600 arm moves the ungated first-step worst from 4.9 to 0.118 (median 3.4 to 7.1e-3); the
  4.7-5.0 values elsewhere are the output clip of an undetermined (rank-deficient) fit, not error magnitudes; the initial law has at most
  five shape parameters at d = 10, and the nearest-neighbour distance in the ambient metric is not a dimension.
- **9.7.3 refined.** CHEMAKZO's 0.7 was mostly an SVD solve repeated per gate call on a singular fit, not only the selection rule.
- **9.7.7 refined.** The spurious law changes bruss_d20's predictions by 3.5e-8; Burgers d10 and d20 and POLLU carry spurious laws too
  (a symmetry of the initial law, laws of the sampled states); in the benchmark's store only burgers_d10 is affected.
- **p5b's species labels** were in the wrong order (fixed): the local arm's misses are S+, HCN, C and N2H+.

### 3.18 The repaired recipe (R4e; `theory/panel/generic4.py`, `r4e_repaired.py`, predictions `e39273d`)

generic4: a truncated-SVD fit (the minimum-norm fit the bound describes), the bound and an out-of-span share from one precomputed factor,
one evaluation per batch for prediction and gate, a batched invariant projection, selection by estimated cost per step among
configurations within the margin, chains stopped once they cannot be served, conservation laws verified off the data, and the
accumulation budgets recorded. On R4c's 17 problems and fisher_kpp_d10:

| problem | k | degree (rank / terms) | laws | served | refused (test) | rollout, 40 | chained mixing 0.01 / 0.1 | evaluate (us) | effective speed-up (from step 10) [R4c] | refused under tol/(2 A_n) (A_n) |
|---|---|---|---|---|---|---|---|---|---|---|
| allen_cahn_d5 | 5 | 3 (56 / 56) | 0 | no | 6.8% | 4.2e-02 | 6.0e-02 / 2.8e-02 | 2 | 14.5 (664.4) [14.3] | 100% (123) |
| bruss_d2 | 2 | 6 (28 / 28) | 0 | yes | 0.0% | 6.5e-04 | 6.5e-03 / 2.9e-03 | 1 | 168.8 (3324.5) [753.5] | 100% (63) |
| bruss_d20 | 9 | 4 (310 / 715) | 0 | yes | 5.3% | 3.4e-03 | 1.4e-03 / 2.3e-03 | 28 | 14.1 (56.8) [13.0] | 18% (67) |
| burgers_d20 | 6 | 3 (84 / 84) | 1 | yes | 1.4% | 1.4e-03 | 1.5e-03 / 1.9e-03 | 4 | 54.9 (259.9) [14.1] | 32% (40) |
| chemakzo | 5 | 5 (252 / 252) | 0 | yes | 0.0% | 2.5e-03 | 4.8e-03 / 4.0e-03 | 22 | 53.7 (53.7) [0.7] | 0% (140) |
| fisher_kpp_d10 | 6 | 5 (462 / 462) | 0 | yes | 7.4% | 1.8e-03 | 2.2e-03 / 2.8e-03 | 25 | 9.6 (33.7) [0.0] | 14% (30) |
| fisher_kpp_d2 | 2 | 11 (78 / 78) | 0 | yes | 2.1% | 3.3e-04 | 1.5e-03 / 1.0e-03 | 2 | 43.4 (34.4) [109.0] | 13% (38) |
| fisher_kpp_d3 | 3 | 9 (220 / 220) | 0 | yes | 1.1% | 7.3e-03 | 6.6e-03 / 8.8e-03 | 4 | 55.4 (112.5) [32.2] | 5% (37) |
| fisher_kpp_d5 | 5 | 6 (461 / 462) | 0 | yes | 4.9% | 1.9e-03 | 1.7e-03 / 1.9e-03 | 20 | 14.3 (48.2) [1.3] | 9% (31) |
| hires | 3 | 12 (413 / 455) | 1 | no | 17.7% | 4.6e-06 | 1.6e-01 / 4.4e-03 | 25 | 4.8 (5.9) [1.0] | 100% (6669) |
| lorenz96_d20 | 20 | 2 (231 / 231) | 0 | no | 62.9% | 2.4e-02 | 2.3e-02 / 8.0e-03 | 6 | 1.5 (1.8) [1.5] | 100% (65) |
| lorenz96_d4 | 4 | 5 (126 / 126) | 0 | yes | 0.0% | 2.7e-04 | 2.5e-04 / 1.9e-04 | 3 | 77.2 (77.2) [14.0] | 1% (40) |
| nelson | 12 | 1 (17 / 17) | 4 | no | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 4 | 1.0 (1.0) [1.0] | 100% (82) |
| oregonator | 3 | 4 (35 / 35) | 0 | no | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 0 | 1.0 (1.0) [1.0] | 100% (38907) |
| pollu | 13 | 3 (560 / 560) | 3 | no | 0.1% | 4.7e-02 | 3.6e-02 / 3.4e-03 | 40 | 32.3 (32.5) [6.0] | 100% (234) |
| primordial | 6 | 2 (36 / 36) | 4 | no | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 3 | 1.0 (1.0) [1.0] | 100% (99) |
| robertson | 3 | 5 (50 / 56) | 1 | yes | 0.0% | 5.5e-05 | 6.7e-05 / 6.2e-05 | 1 | 778.4 (966.5) [32.4] | 0% (62) |
| van_der_pol | 2 | 4 (15 / 15) | 0 | no | 100.0% | 0.0e+00 | 0.0e+00 / 0.0e+00 | 0 | 1.0 (1.0) [1.0] | 100% (37432087) |

- **Speed**: the median effective speed-up over the 10 served problems is 54 (R4c: 14). The three formerly singular fits: CHEMAKZO
  0.7 -> 54 (degree 5, 46 s of training against 1,287 s), fisher_kpp_d5 1.3 -> 14, fisher_kpp_d10 2.0 -> 9.6; Robertson 32 -> 778 (the
  batched projection: its prediction and gate cost 37 us in R4c, 1 us now).
- **HIRES is not served**: its fit is truncated (rank 413 of 455) and the out-of-span term refuses 60-80% of the selection chains' early
  states, so every degree stops early; the plain rollout is 4.6e-6 with 18% refused, chained mixing 0.159.
- **The accumulation budget** tol / (2 A_n) would refuse 100% on HIRES (A_n 6,669), POLLU (234) and every unserved problem; 0-32% on the
  served lattices and 0% on CHEMAKZO (A_n 140, its accepted errors far below the budget anyway).
- Pre-registered: F1 fails (HIRES), F2 holds on speed (4.8-77x) and fails narrowly on cost (evaluation 2.2x R4c's prediction), F3
  holds, F4 holds on the budget and fails on "served", F5 fails (POLLU loses 3 laws, not 1), F6 fails (CHEMAKZO 0%, burgers_d20 32%).

### 3.19 Paired columns (p11; `theory/panel/p11_paired_columns.py`, predictions `bd8eda6`)

p6's column with a cell's neighbours from the hybrid column (closed) or the reference column (forced), cells sorted along their first
principal coordinate or in random order, at D = 0.005, 0.05 and 0.25, on the ten problems R4e serves; each step's worst error split into
the cell's own one-step defect on its own input and the inherited error.

- **Surrogate errors do not compound through the neighbours**: closed and forced columns agree within 1.37x in every setting (C1
  holds); the column's worst error tracks the cell's own-input defect.
- **Strong or unsorted exchange does not raise the own-input defect** in general (C2 fails): exchange averages states back toward the
  bundle.
- **Nine of ten hold 0.01 in all twelve settings**; fisher_kpp_d3 reaches 1.0-1.1e-2 in sorted columns at D = 0.05 and 0.25, from
  own-input defects near 1e-2 (C3 fails there).

### 3.20 The tube experiment (p10, p10b; `theory/panel/p10_tube.py`, predictions `dd12058`)

Can a surrogate serve the inputs a coupled run produces, slightly off the trajectories' bundle, by owning the first-order term of the
off-bundle defect (Fable's tube theory: Phi(x) - Psi(x) = [on the set] + [D Phi - D Psi] n + O(|n|^2) for an offset n across the bundle)?
The fourth recipe in five arms at R4e's chart dimension:
- **T0:** R4e as is.
- **T0+:** with 1,600 fresh plain pairs, the control for data volume.
- **T1m:** with 1,600 of the fit trajectories' own mixed pairs in the fit.
- **T1:** T1m with the off-chart coordinates as linear terms of the design. The model stays linear in its data, so Proposition 5.1
  applies to the enlarged design.
- **T2:** T1m with the chart from a reduced-rank regression of successors on inputs.

Gated scores (accepted defects on mixed inputs, the closed column):

| problem | arm | served | degree | rollout, 40 | mixed inputs 0.01 / 0.1: refused, accepted defect median / max | column 0.005 / 0.05 / 0.25 | effective speed-up |
|---|---|---|---|---|---|---|---|
| bruss_d20 | T0 | yes | 4 | 3.4e-03 | 5%, 1.4e-05 / 1.2e-03; 5%, 1.3e-05 / 1.2e-03 | 2.0e-03 / 1.4e-03 / 5.8e-04 | 15.1 |
| bruss_d20 | T0+ | yes | 5 | 3.7e-03 | 5%, 3.6e-06 / 1.8e-03; 5%, 3.4e-06 / 1.6e-03 | 3.4e-03 / 2.3e-03 / 3.0e-03 | 5.1 |
| bruss_d20 | T1m | yes | 4 | 2.6e-03 | 5%, 1.4e-05 / 6.3e-04; 5%, 1.4e-05 / 1.1e-03 | 1.5e-03 / 5.9e-04 / 5.8e-04 | 15.3 |
| bruss_d20 | T1 | yes | 4 | 2.6e-03 | 5%, 1.4e-05 / 6.3e-04; 5%, 1.4e-05 / 1.1e-03 | 1.5e-03 / 6.5e-04 / 6.3e-04 | 15.1 |
| bruss_d20 | T2 | yes | 4 | 1.9e-03 | 5%, 1.6e-05 / 6.8e-04; 4%, 1.6e-05 / 1.6e-03 | 2.3e-03 / 1.3e-03 / 1.7e-03 | 15.0 |
| burgers_d20 | T0 | yes | 3 | 1.4e-03 | 1%, 5.5e-06 / 1.4e-03; 1%, 5.4e-06 / 1.4e-03 | 1.6e-03 / 2.1e-03 / 1.6e-03 | 56.5 |
| burgers_d20 | T0+ | yes | 3 | 1.6e-03 | 0%, 6.7e-06 / 1.6e-03; 0%, 6.7e-06 / 1.2e-03 | 1.6e-03 / 1.4e-03 / 1.1e-03 | 320.7 |
| burgers_d20 | T1m | yes | 3 | 1.4e-03 | 1%, 4.7e-06 / 1.3e-03; 1%, 4.7e-06 / 1.4e-03 | 1.3e-03 / 1.5e-03 / 1.5e-03 | 58.7 |
| burgers_d20 | T1 | yes | 3 | 2.8e-03 | 0%, 7.0e-06 / 2.6e-03; 0%, 6.9e-06 / 2.2e-03 | 2.7e-03 / 2.1e-03 / 1.3e-03 | 2.9 |
| burgers_d20 | T2 | yes | 4 | 1.3e-03 | 8%, 1.5e-05 / 1.3e-03; 7%, 1.5e-05 / 1.7e-03 | 1.8e-03 / 2.0e-03 / 1.2e-03 | 12.2 |
| chemakzo | T0 | yes | 5 | 2.5e-03 | 1%, 9.2e-06 / 3.8e-03; 7%, 7.2e-05 / 4.7e-03 | 6.0e-03 / 4.3e-03 / 4.8e-03 | 202.0 |
| chemakzo | T0+ | yes | 5 | 2.5e-03 | 1%, 1.3e-05 / 4.2e-03; 9%, 1.0e-04 / 3.9e-03 | 9.2e-03 / 4.2e-03 / 4.4e-03 | 20.2 |
| chemakzo | T1m | yes | 4 | 1.0e-03 | 0%, 1.7e-05 / 2.5e-03; 1%, 2.7e-05 / 3.2e-03 | 9.5e-04 / 3.5e-03 / 2.7e-03 | 97.5 |
| chemakzo | T1 | yes | 4 | 1.0e-03 | 0%, 1.7e-05 / 2.5e-03; 1%, 2.7e-05 / 3.2e-03 | 9.5e-04 / 3.5e-03 / 2.7e-03 | 116.2 |
| chemakzo | T2 | yes | 5 | 3.3e-04 | 0%, 2.2e-06 / 2.0e-04; 0%, 4.0e-06 / 2.3e-03 | 3.3e-04 / 4.6e-04 / 5.2e-04 | 200.0 |
| fisher_kpp_d10 | T0 | yes | 5 | 1.8e-03 | 8%, 9.0e-06 / 1.8e-03; 7%, 8.5e-06 / 1.2e-03 | 2.4e-03 / 1.6e-03 / 9.3e-04 | 11.1 |
| fisher_kpp_d10 | T0+ | yes | 6 | 2.2e-03 | 5%, 2.7e-05 / 2.2e-03; 5%, 2.5e-05 / 1.4e-03 | 2.1e-03 / 2.0e-03 / 1.8e-03 | 9.7 |
| fisher_kpp_d10 | T1m | yes | 5 | 1.6e-03 | 5%, 1.3e-05 / 1.6e-03; 6%, 1.2e-05 / 2.4e-03 | 1.5e-03 / 1.7e-03 / 2.2e-03 | 12.1 |
| fisher_kpp_d10 | T1 | yes | 5 | 1.5e-03 | 5%, 1.2e-05 / 1.2e-03; 5%, 1.2e-05 / 2.1e-03 | 1.2e-03 / 2.1e-03 / 1.3e-03 | 14.0 |
| fisher_kpp_d10 | T2 | yes | 5 | 1.4e-03 | 8%, 1.6e-05 / 2.5e-03; 7%, 1.6e-05 / 1.2e-03 | 2.6e-03 / 1.8e-03 / 3.5e-03 | 10.0 |
| hires | T0 | no | 12 | 4.6e-06 | 66%, 2.4e-04 / 2.2e-03; 85%, 6.0e-04 / 2.2e-03 | 3.3e-01 / 7.1e-02 / 9.2e-02 | 4.8 |
| hires | T0+ | no | 11 | 1.8e-05 | 66%, 2.0e-04 / 1.2e-03; 88%, 3.3e-04 / 1.2e-03 | 2.9e-01 / 7.5e-02 / 9.7e-02 | 9.0 |
| hires | T1m | no | 2 | 0.0e+00 | 100%, –; 100%, – | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.0 |
| hires | T1 | no | 8 | 0.0e+00 | 100%, –; 100%, – | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.0 |
| hires | T2 | no | 2 | 0.0e+00 | 100%, –; 100%, – | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.0 |
| pollu | T0 | no | 3 | 4.7e-02 | 48%, 1.2e-03 / 7.1e-03; 95%, 1.9e-03 / 7.9e-03 | 7.7e-02 / 2.1e-02 / 2.0e-02 | 43.0 |
| pollu | T0+ | no | 3 | 2.0e-02 | 72%, 6.8e-04 / 4.1e-03; 99%, 7.6e-04 / 3.0e-03 | 1.7e-02 / 1.0e-02 / 4.9e-03 | 22.2 |
| pollu | T1m | no | 2 | 2.7e-10 | 100%, –; 100%, – | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.0 |
| pollu | T1 | no | 3 | 2.1e-02 | 40%, 3.0e-04 / 1.2e-03; 67%, 4.2e-04 / 2.6e-03 | 2.8e-02 / 9.8e-03 / 7.4e-03 | 1.5 |
| pollu | T2 | no | 2 | 2.7e-10 | 100%, –; 100%, – | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.0 |
| robertson | T0 | yes | 5 | 5.5e-05 | 0%, 3.0e-07 / 7.7e-05; 0%, 3.2e-07 / 7.2e-05 | 6.9e-05 / 1.0e-04 / 5.0e-06 | 392.7 |
| robertson | T0+ | yes | 6 | 3.7e-06 | 0%, 1.3e-07 / 1.3e-06; 1%, 1.5e-07 / 3.1e-06 | 3.2e-06 / 3.1e-06 / 1.9e-06 | 529.7 |
| robertson | T1m | yes | 3 | 5.2e-04 | 0%, 5.7e-06 / 1.1e-03; 0%, 5.7e-06 / 4.0e-04 | 1.3e-03 / 4.4e-04 / 1.0e-04 | 156.9 |
| robertson | T1 | yes | 3 | 5.2e-04 | 0%, 5.7e-06 / 1.1e-03; 0%, 5.7e-06 / 4.0e-04 | 1.3e-03 / 4.4e-04 / 1.0e-04 | 157.4 |
| robertson | T2 | yes | 3 | 5.2e-04 | 0%, 5.7e-06 / 1.1e-03; 0%, 5.7e-06 / 4.0e-04 | 1.3e-03 / 4.4e-04 / 1.0e-04 | 955.9 |

Ungated defects on mixed inputs (p10b, post hoc; median / maximum of the worst scored defect per state):

| problem | arm | plain | mixed w = 0.01 | mixed w = 0.1 |
|---|---|---|---|---|
| hires | T0 | 9.7e-9 / 2.5e-6 | 1.2e-3 / 0.35 | 1.1e-2 / 300 |
| hires | T1 | 1.4e-4 / 1.0e-2 | 2.2e-4 / 6.4e-2 | 4.3e-4 / 9.0e-2 |
| hires | T1m | 9.3e-3 / 0.14 | 1.3e-2 / 0.32 | 1.8e-2 / 0.65 |
| hires | T2 | 5.1e-3 / 7.3e-2 | 6.9e-3 / 0.28 | 9.1e-3 / 0.43 |
| pollu | T0 | 2.0e-4 / 9.9e-4 | 3.0e-3 / 4.5e-2 | 2.4e-2 / 0.29 |
| pollu | T1 | 4.2e-4 / 2.6e-3 | 4.3e-4 / 4.3e-3 | 8.6e-4 / 3.0e-2 |
| chemakzo | T0 | 1.1e-7 / 2.2e-5 | 9.3e-6 / 0.11 | 8.3e-5 / 0.71 |
| chemakzo | T2 | 1.9e-6 / 1.4e-4 | 2.2e-6 / 2.0e-4 | 4.1e-6 / 6.2e-2 |

- **The normal term owns the off-bundle defect on reduced charts.** On POLLU and HIRES it cuts the mixed inputs' median defect 5-28x and
  their maximum 5-3,300x. On POLLU this shows in the gated scores: accepted defects 4-6x lower, refusal under mixing from 48% to 40% and
  from 95% to 67%, and a column 2.7x better (2.8e-2 / 9.8e-3 / 7.4e-3), though POLLU is still not served.
- **Mixed data alone (T1m) and the sensitivity chart (T2) do not do it on reduced charts.** A chart with no normal term cannot tell the
  mixed states from the bundle's, so their successors conflict.
- **The normal term costs accuracy on the bundle.** HIRES's plain median rises from 1e-8 to 1.4e-4 and its worst to 1e-2, and the gate
  calibrated for the maximum then refuses everything. A composite (T0's fit on the bundle, T1's beyond a normal offset) is the next
  design.
- **On full charts the sensitivity chart appeared to help (withdrawn in 3.23: a timing tie picked another degree).** It improves CHEMAKZO 7-14x in every coupled score (column
  6.0e-3 to 3.3e-4, rollout 2.5e-3 to 3.3e-4, worst mixed defect at w = 0.01 from 0.11 to 2.0e-4).
- Pre-registered:
  - U1 and U2 fail as stated: T1 refuses all of HIRES, so the gated measures are undefined. The ungated medians fall 5.5x and 26x.
  - U3 fails: T2 is 31x above T1 on HIRES.
  - U4 fails on bruss_d20: plain pairs cut its median 3.9x.
  - U5 fails: CHEMAKZO's T2 and Robertson's T0+ and T1m move 7-15x.
  - U6 holds: every served arm holds 0.01 in the column at every exchange.
- Timings vary 2-3x between arms at the same degree (concurrent runs).

### 3.21 A basis for larger charts (p12; `theory/panel/p12_basis.py`, predictions `37b71f6`)

Total degree against the hyperbolic cross and a greedy downward-closed set (grown by correlation with the fit's residuals, at total
degree's size), at R4e's chart dimension:

| problem | best held-out worst: total / hyperbolic / adaptive | served |
|---|---|---|
| pollu | 3.0e-3 / 5.1e-3 / 1.2e-2 | none |
| nelson | 1.10 / 1.10 / 1.10 | none |
| primordial | 0.53 / 0.47 / 1.49 | none |
| lorenz96_d20 | 0.12 / 0.17 / 0.33 | none |
| bruss_d20 | 3.5e-2 / 4.0e-2 / 0.12 | all three |
| fisher_kpp_d10 | 0.32 / 0.30 / 0.42 | all three |
| burgers_d20 | 8.0e-3 / 2.1e-2 / 2.3e-2 | all three |

- **No index set helps**: the greedy set overfits (its in-sample ranking of candidates does not transfer to held-out states); the
  hyperbolic cross is comparable or worse. B1 fails, B2 and B3 hold.
- **Nelson and primordial are not basis-limited**: their ungated one-step errors on the test trajectories are 0.5-0.7 and 0.7-1.1 dex at
  every step (medians after step 5: 0.11 and 0.04 dex), with about 1,800 fitted pairs in 16 and 10 coordinates at degrees 1-2.
- **POLLU is accumulation-limited**: its ungated one-step errors after step 5 are at most 9.3e-4, its rollout 4.7e-2 (A_n 234).

### 3.22 KIDA: coverage against data, and the solver (p8, p9; `theory/panel/p8_kida_coverage.py`, `p9_kida_solver.py`; predictions committed with each script)

- **Coverage grows slowly with tracers** (p8): the local arm's oracle coverage at 0.01 dex is 1.8, 5.8, 13.6 and 17.4% at 20, 40, 80
  and 160 tracers (median 0.047 to 0.022 dex); the last doubling's exponent, 0.35, puts 50% at about 3,300 tracers. The degree-2 chart
  goes from 3.6 to 7.1% between 40 and 160.
- **The reference is sound and the solver's cost is its right-hand side** (p9, 120 dark states, one core): the stored labels are within
  1.8e-6 dex of a rtol 1e-9 reference on the key species (5.7e-6 strict). A step costs 0.48 s at the median with the dense Jacobian and
  0.61 s with a sparse one (35 nonzeros per row): about 670 right-hand-side evaluations, 3.5 Jacobians and 64 LU factorisations per
  step, so the right-hand side, not the linear algebra, is where a faster exact solver would come from.


### 3.23 After the council (round 3): what was adopted

The reviews (`theory/reviews/council3-{fable,sol,astra}.md`; resolutions in `theory/REVIEWS.md`) corrected four readings of 9.8. Fable
found that POLLU is data-limited, not accumulation-limited. HIRES is refused by the product form of the gate. The normal term's cost on
the bundle is a radius effect. CHEMAKZO's "rotation advantage" was a timing tie in the selection. KIDA's step costs its LU (42-46%)
more than its right-hand side (28-36%), and speed-ups are quoted against SciPy's per-call overhead. They asked, in order, for:
- learning curves in whole trajectories;
- the additive envelope gate with an error account per cell;
- the composite tube;
- a compiled baseline;
- KIDA's exact solver before any fit.
Tensor trains were deferred (the fitted coefficient tensors need 6-11x more coefficients as trains).

### 3.24 Learning curves in whole trajectories (p13; `theory/panel/p13_learning.py`, predictions `8f472cf`)

The fourth recipe at R4e's chart dimension with 220, 340 and 580 training trajectories, the extra ones from each problem's initial law
with new seeds; at most 57 solver seconds of extra data per problem.

| problem | 100 (R4e) | 220 | 340 | 580 |
|---|---|---|---|---|
| pollu | not served, rollout 4.7e-2 | not served, 5.5e-3 | **served**, 5.5e-3 | **served**, 1.5e-4 (degree 4, nothing refused) |
| allen_cahn_d5 | not served, 4.2e-2 | not served, 7.0e-2 | **served**, 7.1e-3 (speed-up 94) | **served**, 6.2e-3 |
| nelson (best held-out worst) | 1.10 | 0.43 (180 kept) | 0.64 (300) | 0.27 (540); never served |
| primordial | 0.53 | 0.54 | 0.62 | 0.85; never served |
| hires | not served | rollout 5.5e-6, mixing 0.16 | 5.1e-6, 0.16 | 4.0e-6, 0.17; never served |
| lorenz96_d20 | rollout 2.4e-2 | 7.5e-3 | 7.5e-3 | 4.3e-3; not served (the margin) |
| oregonator, van_der_pol | refused whole | refused whole at every size | | |

- **Data was the limit for POLLU and Allen-Cahn d5**, and it costs seconds of solves (Fable's reading; L1 and L5 hold).
- **Nelson improves 4x and is still not served**, so L2 holds in its first clause and fails in its second. Primordial does not improve
  with data (L3 holds).
- **HIRES's chain under weak mixing stays at 0.16-0.17 at every size** (L4 holds): it is not a data limit.

### 3.25 The envelope gate, the sketch and the error account (R4f; `theory/panel/r4f_envelope.py`, predictions `90e0902`)

The gate a0 + a1 b + a2 rho + a3 rho^2 is fitted by a linear program over the calibration states (plain and mixed), with its caps at
their ranges. The gate's products are sketched with 32 columns. Each chain is deployed with an error account per cell: a step is
refused once it would take the cell's sum of bounds past tol / 2.

- **Seven of 18 served, against R4e's ten** (G3 fails). Bruss_d20, fisher_kpp_d5 and fisher_kpp_d10 are lost. The account charges the
  envelope's floor a0 at every step, so over 40 steps it needs a0 < tol / 80, and their cells go to the solver for 97-100% of their
  steps.
- **The envelope is not conservative enough on burgers_d20** (G1 and G2 fail there). It is served at degree 5 with a rollout of 1.0e-2
  and chained mixing of 1.5e-2, and the account is exceeded on 1.8% of cells and steps. Elsewhere the account holds: fisher_kpp_d3
  0.94%, the rest 0%.
- **Where the account binds, coupled chains are honest**: HIRES 9.9e-3 / 4.2e-3 and POLLU 4.5e-3 / 3.4e-3 under chained mixing, at
  91-95% solver.
- **The sketched gate is 3-5x cheaper at equal size** (G5 mostly holds): HIRES 25 -> 6 us, POLLU 40 -> 8 us.
- G4 holds: the median effective speed-up over the seven served is 70.

### 3.26 The composite tube (smoke tests on HIRES and POLLU; generic4 `composite`)

The on-bundle fit is kept. An affine correction in the off-chart coordinates is fitted on the fit trajectories' mixed pairs between
the decoder's training residual and a radius, and added only beyond the residual. The gate is calibrated per region.
- **At radius 3e-3** HIRES's plain rollout is unchanged (4.6e-6, as designed) and its chained mixing at w = 0.01 stays at 0.18.
- **At 3e-2:** 0.16, with 37% refused.
- **Without a radius:** the correction errs 0.06 at the median on offsets up to 0.74, and the gate refuses everything.
- **On POLLU** the composite lowers the rollout from 4.7e-2 to 3.2e-2 at 100 trajectories.

HIRES's coupled failure comes from rare states whose defect is near 1e-2, summed over 170 steps: a first-order correction moves the
median, not those states. Not run as a pre-registered experiment.

### 3.27 A compiled exact baseline (p14; `theory/panel/p14_baseline.py`, prediction in the same commit)

SciPy's VODE, a Fortran BDF with the systems' tolerances and Python right-hand sides, on 40 test states at three steps per problem:
- **3.4-14x faster** per state and step than the benchmark's baseline on the lattices, Robertson and CHEMAKZO, agreeing with the stored
  states within 1.7e-7;
- **4.5x slower** on HIRES;
- **fails** on Nelson and primordial (convergence failures).

R4e's effective speed-ups against VODE:

| problem | against the baseline | against VODE |
|---|---|---|
| robertson | 778 | 82 |
| bruss_d2 | 169 | 109 |
| burgers_d20 | 55 | 30 |
| lorenz96_d4 | 77 | 23 |
| fisher_kpp_d2 | 43 | 19 |
| fisher_kpp_d3 | 55 | 14 |
| bruss_d20 | 14 | 8 |
| chemakzo | 54 | 4 |
| fisher_kpp_d5 | 14 | 4 |
| fisher_kpp_d10 | 10 | 3 |

The median is 16. A solver with a compiled right-hand side would lower these further, so the honest statement is that the recipe beats
a compiled BDF by 3-100x on the problems it serves, with the smallest margins on the higher-dimensional lattices and CHEMAKZO.


### 3.28 After the council (round 4): what was adopted

The reviews (`theory/reviews/council4-{fable,sol,astra}.md`; resolutions in `theory/REVIEWS.md`) corrected 9.9:
- **R4f's losses** come from uncapped calibration states, whole trajectories outside the fitted data with b up to 1e9, not from the
  floor a0 (the three lost lattices have a0 = 0).
- **Burgers d20's failure** is one accepted step whose bound was 3.3e-3 for a defect of 1.04e-2.
- **The account** assumes errors propagate with factor 1, so it is not a bound (HIRES reaches 1.27e-2 under weak mixing with no accepted
  defect above its bound).
- **Nelson and primordial are bias-limited.**
- **VODE is the wrong denominator on three problems.**

They asked for:
- the capped gate, with an account whose multiplier is calibrated on whole chains;
- the exact baselines re-measured, with LSODA;
- a data budget with a stopping rule;
- an end-to-end audit on fresh closed columns.

"Certifies coupled chains" and the composite tube are withdrawn.

### 3.29 The capped gate with a chain-calibrated account (R4g; `theory/panel/r4g_capped.py`, predictions `7163a03`)

The gate:
- **A label-free cap** b <= 3 sqrt(rank): the mean of b^2 over the fit is the rank, so at most a ninth of the fitted states exceed it.
- **The additive envelope**, fitted by a linear program inside the cap, as the gate.
- **An account charged with a typical-defect model**, accepting a step while lambda (S + c) <= tol / 2. Lambda is the smallest of 0.5-8
  for which at most 5% of whole calibration chains (plain, w = 0.01, w = 0.1) exceed tol / 2.

The deployed fit is calibrated on 100 fresh calibration trajectories. It is run at 100 and 340 training trajectories on R4e's 18
problems.

- **At 340 training trajectories 13 of 18 are served**, every one within tol / 2 in its 40 test rollouts and in chained mixing at both
  weights: R4e's ten, allen_cahn_d5 (rollout 3.6e-4, 10% solver), POLLU (3.7e-3, 2.9%) and HIRES (1.2e-5, but 85% solver with lambda 3).
  Lambda is 0.5 everywhere else (lorenz96_d20: 1.5).
- **At 100 the gate is not enough**: fisher_kpp_d3 is served with a rollout of 1.2e-2, and fisher_kpp_d10 is not served. H1 and H2
  hold at 340 and fail at 100.
- **H3 holds**: Burgers d20's accepted one-step worst is 1.1-1.3e-3.
- **H4 fails on Robertson and CHEMAKZO**: their envelopes sit near 1e-7, at the label noise, and are crossed at that level. It is
  harmless for the tolerance but breaks the scenario statement there.
- **H5** holds in lambda and fails in the solver share. **H6** holds at 100.
- **The cost is the cap under coupling**: CHEMAKZO's mixed inputs lie outside it (a third of its calibration states inside), so the
  solver takes 99-100% of its mixed steps, where the product gate took none.

### 3.30 The exact baselines re-measured (p15; `theory/panel/p15_denominators.py`, prediction in its commit)

Three exact solvers at each system's tolerances on 40 test states at three steps per problem, one core:
- the library's step;
- SciPy's VODE (BDF, Fortran);
- LSODA (Fortran, switching between Adams and BDF).

The fastest that agrees with the stored states within 1e-5 is the problem's denominator.
- **LSODA is the fastest on 9 of 15 problems, VODE on 4.** The library's own step is the only one that qualifies on primordial
  (VODE and LSODA fail some states) and on KIDA, where VODE takes minutes per state, LSODA fails some states, and the library takes
  0.38 s.
- **Nelson's library step is 18.7 ms on one core**, not the 78 ms of its baseline record.
- **R4e's served surrogates with the sketched gate (4-7x cheaper than the exact gate on the fits with 460-715 terms), against the
  fastest exact solver:** bruss_d2 128, robertson 120, lorenz96_d4 68, chemakzo 54, burgers_d20 46, fisher_kpp_d2 33, fisher_kpp_d3
  26, bruss_d20 15, fisher_kpp_d5 9.8, fisher_kpp_d10 8.5; median 39.
- The prediction fails in two clauses (LSODA does not qualify everywhere; the median is above the predicted 10-30) and holds in two
  (LSODA fastest on at least 8; the sketch at least 2x cheaper on the large fits).

### 3.31 The end-to-end audit on fresh closed columns (p17; `theory/panel/p17_audit.py`, predictions `76516a5`)

R4g's 13 served problems (at 340 trajectories) are deployed with their accounts in closed columns:
- **Cells:** 100 trajectories no fit, calibration or test had seen.
- **Settings:** sorted and unsorted orders at D = 0.005, 0.05 and 0.25.
- **Cost:** measured evaluation time plus the refused cells at the fastest exact solver, against the exact column.

Results:
- **Ten hold tol / 2 in all six columns.** Their worsts are 1e-7 to 2.2e-3: the lattices, Robertson, CHEMAKZO, lorenz96_d4,
  allen_cahn_d5.
- **HIRES fails the tolerance** (2.0e-2 at 93-98% solver), so its account's lambda = 3, chosen on calibration chains, does not transfer
  to fresh closed columns. fisher_kpp_d10 (5.3e-3) and POLLU (5.6e-3) exceed tol / 2. A1 fails.
- **The cap refuses coupled inputs on CHEMAKZO and POLLU**: their solver shares are 96-100% in the columns against 3-5% on plain
  rollouts. A2 fails.
- **End-to-end speed-ups against the fastest exact solver:**

  | problem | speed-up |
  |---|---|
  | bruss_d2 | 20 |
  | fisher_kpp_d2 | 9.4 |
  | burgers_d20 | 8.0 |
  | bruss_d20 | 7.9 |
  | lorenz96_d4 | 7.7 |
  | robertson | 7.1 |
  | fisher_kpp_d3 | 5.5 |
  | fisher_kpp_d5 | 2.3 |
  | fisher_kpp_d10 | 1.9 |
  | chemakzo, pollu, hires | below 1 |

  That is 8 of 12 problems at 2x or more and one at 10x or more (A3 fails). The surrogate is evaluated 100 cells per call on one core,
  so per-call overhead is in these numbers; a large GPU batch would remove most of it but not the refusals.

### 3.32 Refusal-driven data (p18; `theory/panel/p18_active.py`, predictions `5753026`)

R4g's deployed surrogate (100 training trajectories) is run, plain and under chained mixing, on a pool of 100 fresh trajectories. The
pairs it refuses, which deployment solves anyway, are added to the training set and the whole recipe is refit. Against them, the same
number of random pairs from fresh trajectories at the same solver cost.

| problem | extra pairs | R4g at 100: served, solver share | refused data: served, solver share, rollout | random data: served, solver share, rollout |
|---|---|---|---|---|
| allen_cahn_d5 | 7800 / 4800 | no, 97.5% | yes, 6.1%, 2.9e-03 | yes, 5.4%, 8.2e-03 |
| bruss_d2 | 452 / 452 | yes, 0.0% | yes, 0.0%, 6.8e-04 | yes, 0.0%, 6.2e-05 |
| bruss_d20 | 607 / 607 | yes, 7.6% | yes, 5.1%, 1.0e-03 | yes, 7.2%, 1.3e-03 |
| burgers_d20 | 240 / 240 | yes, 3.3% | yes, 2.0%, 1.3e-03 | yes, 3.2%, 1.1e-03 |
| chemakzo | 8782 / 8782 | yes, 3.6% | yes, 0.9%, 7.3e-06 | yes, 4.6%, 1.0e-05 |
| fisher_kpp_d10 | 8000 / 4800 | no, 100.0% | yes, 4.9%, 1.3e-03 | yes, 5.8%, 5.0e-04 |
| fisher_kpp_d2 | 158 / 158 | yes, 2.7% | yes, 0.5%, 1.4e-03 | yes, 2.6%, 2.4e-03 |
| fisher_kpp_d3 | 231 / 231 | yes, 3.7% | yes, 2.3%, 2.3e-03 | yes, 4.3%, 9.3e-04 |
| fisher_kpp_d5 | 533 / 533 | yes, 6.8% | yes, 4.5%, 9.9e-04 | yes, 6.1%, 1.4e-03 |
| hires | 31627 / 20400 | yes, 89.6% | no, 100.0%, 0.0e+00 | yes, 86.6%, 2.1e-05 |
| lorenz96_d20 | 7719 / 4800 | no, 98.6% | no, 95.6%, 4.9e-03 | no, 95.7%, 4.1e-03 |
| lorenz96_d4 | 355 / 355 | yes, 5.2% | yes, 2.9%, 4.2e-05 | yes, 4.4%, 2.8e-05 |
| nelson | 4800 / 2880 | no, 100.0% | no, 100.0%, 0.0e+00 | no, 100.0%, 0.0e+00 |
| oregonator | 8000 / 4800 | no, 100.0% | no, 100.0%, 0.0e+00 | no, 100.0%, 0.0e+00 |
| pollu | 11800 / 7080 | no, 100.0% | no, 94.4%, 6.2e-03 | no, 6.6%, 6.4e-03 |
| primordial | 6000 / 3600 | no, 100.0% | no, 100.0%, 0.0e+00 | no, 100.0%, 0.0e+00 |
| robertson | 389 / 389 | yes, 2.5% | yes, 1.4%, 2.2e-05 | yes, 2.4%, 2.3e-04 |
| van_der_pol | 8000 / 4800 | no, 100.0% | no, 100.0%, 0.0e+00 | no, 100.0%, 0.0e+00 |

- **Refused-state data beats random data at equal cost** on 6 of the 7 problems that refuse at least 3%:
  - CHEMAKZO: 3.6% -> 0.9%, against 4.6% with random data;
  - lorenz96_d4: 5.2% -> 2.9%, against 4.4%;
  - Burgers d20: 3.3% -> 2.0%, against 3.2%;
  - fisher_kpp_d3, fisher_kpp_d5 and bruss_d20: 1.5-1.6x lower;
  - fisher_kpp_d2, below the 3% threshold: 2.7% -> 0.5%, against 2.6%.

  Only CHEMAKZO falls 2x or more, so R1 holds in its second clause and fails in its first.
- **Refused-state data serves fisher_kpp_d10 and Allen-Cahn d5 at 100 trajectories** (R2 holds), though random data does too.
- **It holds tol / 2 in every served arm**, while random data's Allen-Cahn d5 (8.2e-3) and HIRES (mixing 6.6e-3) exceed it: R3 fails in
  the random arms only.
- **HIRES gets worse with refused data** (everything refused), and Nelson, primordial and the oscillators stay unserved (R4 holds).
- One round of refusal-driven data lowers rho by 1.5-5x on the served problems. Iterating it is the obvious next test.

### 3.33 Robustness of "served" across independent draws (p19; `theory/panel/p19_seeds.py`, predictions `fe1a62b`)

Three independent draws of 340 training and 100 calibration trajectories (seeds 20-31) on the five problems served near the margin:
- **S1 holds**: POLLU, Allen-Cahn d5 and fisher_kpp_d3, d5 and d10 are served in all three draws.
- **S3 holds**: the rollout solver shares are 2.8-4.6%, within 1.4x across draws.
- **S2 fails narrowly**: one Allen-Cahn d5 draw rolls out at 5.3e-3, above tol / 2 and within the tolerance.
- **POLLU is served only on plain rollouts**: under chained mixing every step goes to the solver (the cap), in every draw.


### 3.34 After the council (round 5): what was adopted

The reviews (`theory/reviews/council5-{fable,sol,astra}.md`; resolutions in `theory/REVIEWS.md`) found the following.
- **Refusals are a start-up transient.**
- **The account never decays**, so it refuses long chains for good.
- **p17's speed-ups were limited by the evaluation as timed**: the gate was computed twice, and 100 cells were evaluated per call.
- **CHEMAKZO's coupled refusals are the cap**, and refused-state data is leverage design.
- **On a GPU the fallback dominates above a refusal rate of about 1e-3**, and the comparator that matters is a batched stiff solver on the
  same GPU.

Adopted:
- the account reuses the gate's statistics, decays optionally and records refusal causes;
- long chains (p21);
- refusal-driven data in closed columns (p20);
- the GPU measurement plan (`theory/GPU_PLAN.md`), with no GPU code until it is measured.

### 3.35 Long chains with three account arms (p21; `theory/panel/p21_long.py`, predictions `a4fc656`)

Each problem's deployed surrogate (R4g at 340) runs on 64 new trajectories of 2,000 steps in three arms: the gate alone, the present
account, and a decaying account (gamma 0.5, chosen on 400-step chains). Speed-ups are against the fastest exact CPU solver, with the
evaluation timed at 1,000 cells per call.

| problem | gate alone: steady refusal (95% bound), worst over 2,000 steps, steady speed-up | present account: steady refusal |
|---|---|---|
| bruss_d2 | 0 (2e-5), 2.9e-3, 439 | 87% |
| burgers_d20 | 0 (2e-5), 9.6e-4, 119 | 68% |
| fisher_kpp_d2 | 0 (2e-5), 2.0e-3, 105 | 81% |
| lorenz96_d4 | 0 (2e-5), 2.8e-3, 94 | 29% |
| bruss_d20 | 0 (2e-5), 1.2e-3, 49 | 76% |
| fisher_kpp_d3 | 0 (2e-5), 1.6e-3, 36 | 86% |
| fisher_kpp_d5 | 0 (2e-5), 1.2e-3, 19 | 88% |
| fisher_kpp_d10 | 0 (2e-5), 1.8e-3, 18 | 94% |
| allen_cahn_d5 | 0.013%, 2.0e-3, - | 0.013% |
| robertson | 88%, 6.1e-7, about 1 | 88% |
| chemakzo | 96%, 1.8e-7, about 1 | 96% |

- **On 9 of 11 problems the refusals are only the start-up transient** (9-38% of steps 1-10). From step 11 to step 2,000 the gate
  refuses nothing in 1.3e5 cell-steps, and every chain holds tol / 2. The steady speed-up against the fastest CPU solver is 18-439x,
  median about 70.
- **The present account is what throttled long runs.** The decaying account matches the gate alone.
- **Robertson and CHEMAKZO refuse their long-time states**, which lie beyond the 40-step training trajectories (3.37).
- L1, L2 and L4 hold; L3 fails on those two.

### 3.36 Refusal-driven data in closed columns, iterated (p20; `theory/panel/p20_closed_active.py`, predictions `3406591`)

Three rounds on eight full-chart problems. Each round adds either the coupled inputs the deployed surrogate refused or the same number
of uniformly drawn coupled inputs, and is audited on independent closed columns.

| problem | p17 audit solver share | refused data, three rounds | uniform data, three rounds | refused arm's worst |
|---|---|---|---|---|
| chemakzo | 98.6% | 1.8% | 2.8% | 5.2e-6 |
| bruss_d2 | 1.6% | 0.2% | 1.3% | 1.3e-3 |
| fisher_kpp_d2 | 1.5% | 0.3% | 1.5% | 5.7e-4 |
| robertson | 4.7% | 0.6% | 2.9% | 2.7e-6 |
| allen_cahn_d5 | 7.2% | 1.0% | 2.0% | 5.4e-3 |
| fisher_kpp_d3 | 2.2% | 1.0% | 2.3% | 6.5e-4 |
| lorenz96_d4 | 1.5% | 1.0% | 1.7% | 3.5e-4 |
| fisher_kpp_d5 | 3.5% | 1.8% | 3.1% | 3.4e-3 |

- **Refused data beats uniform data at equal solves on all eight** (C2 holds). It cuts the coupled refusal rate 55x on CHEMAKZO (C1
  holds: 2.7% after one round, 1.8% after three) and 5-8x on bruss_d2, fisher_kpp_d2, Robertson and Allen-Cahn d5.
- **Every audit holds tol / 2 except Allen-Cahn d5** (5.4e-3, within the tolerance), so C3 fails there.
- **The audit speed-ups (100 cells per call) are noisy**, and C4 fails (1.2x, not 1.5x). The steady speed-ups of 3.35 are the better
  measure.

### 3.37 Coverage in time (p22; `theory/panel/p22_long_training.py`, prediction `38babf7`)

The one-step pairs of 40 training trajectories of 400 steps are added for Robertson and CHEMAKZO, with bruss_d20 as a control. Over the
2,000-step horizon:
- Robertson's steady refusal falls from 88% to 73%, CHEMAKZO's from 96% to 81% (the prediction, below 5%, fails);
- bruss_d20 stays at 0%.

Both systems keep reaching new states through the whole horizon, so the training trajectories have to span the deployment horizon.

### 3.38 Training over the whole horizon (p23; `theory/panel/p23_full_horizon.py`, prediction in its commit)

As p22, with the 40 long training trajectories spanning the whole 2,000-step horizon. Over the same 64 test chains:
- Robertson's steady refusal falls from 88% (40-step training) and 73% (400-step) to 0%, worst 2.1e-5, a steady speed-up of 42x;
- CHEMAKZO's falls from 96% and 81% to 45%, worst 3.1e-6: within tol / 2, but the prediction (below 5%) fails for it;
- bruss_d20 stays at 0% (worst 1.1e-3).

Training that spans the deployment horizon serves Robertson's long chains. CHEMAKZO keeps reaching states the 40 long trajectories do
not cover even then: its long-time states depend on the start more than Robertson's do, so it needs more long starts, not longer ones.

---

## 4. Mistakes, and what each taught

| Mistake | Where | What it taught |
|---|---|---|
| Building, tuning and recommending a recipe on one problem (HIRES) | sections 8, G, 10 before R2 | HIRES is a favourable special case (trajectories collapse onto a thin curved set the samples cover). Every claim now runs on the whole panel by class first. |
| Selecting the degree by the error on the training trajectories themselves | R2 | In-sample checks cannot see a fit that is wild between samples; selection must be held out, by whole trajectories, with every transform fitted inside the fold. |
| Clamping chart coordinates to the training box | R2 (CHEMAKZO) | A clamp silently extrapolates; inputs outside the validated range must be refused, not clipped. |
| A headline on 20 of 40 test trajectories, with a degree range that stopped one short of the good degrees | HIRES recipe, round 7 | Report every test trajectory; never let a search range end where the answer might start. |
| A refusal threshold that double-counted sampling and had a floor | R4, R5 | A theorem's worst-case constant is not a threshold; calibrate on held-out data and label it distributional. |
| Calling the fronts "approximation-limited" from a stalled held-out maximum | R5 write-up | A held-out maximum that falls then rises is sampling; diagnosing a factor needs the in-sample and held-out errors, the bound and the step index per degree. |
| A "served" flag that counted refuse-everything as served | R4 | Define success before the run, including what counts as the solver alone. |
| Claims stronger than the data (a corrected figure on a plain row; "falls 30-fold" for a 12.9-fold rise; unsourced overheads) | rounds 7, 8 | Every number from a committed output, quoted with its cohort. |
| Theorems with missing hypotheses (Proposition 8.2's domain, Corollary 6.3's supremum, Corollary 4.3 read as conditional, Proposition 1.5's monotonicity) | rounds 8–9 | Counterexamples on the simplest systems (a circle, $x' = -x$, a smoothstep) find these fast; ask for them. |
| Process slips: `pgrep -f` killing the session's own shell; a run started before its predictions were committed (stopped and restarted); killing a `nohup` wrapper instead of the process | throughout | Kill by checked PID only; commit predictions in a separate step from the launch; verify the process tree. |
| KIDA: a solver pool forked before the system was registered; a lift check stepping states far off the data | R3, R4 | Register before forking; never step a state the data do not support (now guarded). |
| Too many optimistic predictions | P/Q/G/D sets | Of the latest sets, roughly half the predictions failed. Pre-registration made that visible, which is its point. |

---

## 5. What was learned (generic, across problem classes)

1. **The error where a surrogate is used factors into approximation, sampling, propagation and off-data terms, and each factor has a
   computable descriptor.** The sampling factor is the one most often ignored and the one that explained most failures.
2. **For any fit linear in its data, the Lebesgue function at the point of use predicts generalisation before any value there is known.**
   Moderate $\Lambda$ (targets among the samples) lets the degree rise; exploding $\Lambda$ (targets between sparse samples) means a fit
   exact on its data is wrong elsewhere. Designed samples (Chebyshev grids) keep it logarithmic, which is why ChebTT works on boxes.
3. **Trajectory data are poorly placed for global fits.** A trajectory contributes one scattered state (its start) and then states the
   dynamics pull together; the transient layer is sampled once per trajectory per step index, and every held-out maximum examined sits
   there. More *independent starts* (or offsets into the transient) help; longer trajectories do not.
4. **Refusal sets the speed.** With a refusal rate $\rho$ the speed-up is at most $1/\rho$; the gate's own cost counts too
   ($O(p^2)$ for the exact bound). The benchmark's nominal speed-up ignores both.
5. **Off-data inputs (operator splitting) are a separate problem.** No basis removes it; refusal makes it safe and slow; data at
   the inputs the coupled run produces (mixed states) and a residual term for displacements across a reduced chart's fibres are the levers.
6. **Reduction pays only where trajectories collapse.** Chemistry with slow manifolds (HIRES, CHEMAKZO) reduces; box-filling ladders and
   chaotic systems do not, at the panel's horizons; POLLU and Nelson have no validated reduction with the current lifts.
7. **Pre-registration and adversarial reviews work.** Most errors in this work were caught by a committed prediction failing or by a
   reviewer's counterexample, not by the author.

---

## 6. The schemes: strengths and weaknesses

| Scheme | Strengths | Weaknesses | Where it fits |
|---|---|---|---|
| ChebTT, ChebTT-M, ChebTT-A | designed samples, logarithmic Lebesgue constant; holds 0.01 on small ladders (2e-3 to 4e-3); breakpoints for aligned kinks; an interface term for oblique jumps | the box: effort at vertices no trajectory visits (T7); cost per output (U6) and with the ambient dimension; fronts need very high degree; extrapolates badly outside its box | smooth, box-filling problems of moderate $d$ |
| ChebTree, SparseCheb, hub | local Chebyshev cells for layers; sparse grids for moderate $d$; the hub for shared latent coordinates | as ChebTT, cell counts grow with fronts in many coordinates | layers aligned with axes |
| AFINNI, ART | adaptive local polynomials; trust regions or oblique leaves; one sample serves every output; GPU engine | fill the box; ART's quadratic leaves fail at vertices (U5); global accuracy at tight tolerances is costly | loose tolerances, curved slow manifolds |
| MetricKernel | learned metric; computable Lebesgue and power functions | dense $O(N^3)$; stationary kernels struggle on layers | moderate $N$, smooth |
| SlowTable (ISAT) | refusal and in-situ growth built in; local linear records | ellipsoids are estimates; error control is a quantile; memory grows | in situ, near a slow manifold |
| Generic chart recipe (this work) | solve-free fits from trajectories; computable refusal; Theorem 8 for its factored form; 2e-6 to 5e-4 where served | needs trajectories that cover the use; blind across a reduced chart's fibres without the residual term; the transient layer is undersampled | thin visited sets (chemistry with slow manifolds), small $d$ |

---

## 7. Newer surrogate families (`theory/APPROACHES.md`; 84 new sources, NN-1 to NN-84)

What the literature says, read with caution (most results are means or percentiles on the authors' own problems):

- **No published learned chemistry surrogate reports or controls a maximum error near 0.01 dex.** The one astrochemistry benchmark
  with a tail metric (Janssen et al. 2026, NN-39; KROME networks up to 37 species) finds the best networks at 0.38–2.8 dex at the
  99th percentile, chosen *instead of* the maximum on purpose, with a fallback assumed for the rest.
- **Coupled runs drift in every community that tried them**, and the fixes were data coverage or online training, not architecture:
  combustion (NN-26, NN-28), atmospheric chemistry (NN-34), climate parametrisations (NN-72, NN-73), astrochemistry (NN-38, NN-39). This
  is the panel's U8 and off-data finding, found four times independently.
- **Fallback is the field's answer to rare large errors** (NN-39, NN-40, ISAT); Corollary 5.3 is the accounting it needs.
- **What carries over from the memo to any surrogate**: propagation (Propositions 2.1–2.2, Theorem 8), the $1/\rho$ cap and refused
  steps adding no defect, T4's audits, Proposition 8.2, and Proposition 6.5's lower bound on a latent dimension. **What does not**: the
  computable sampling factor survives only for models linear in their data once their nonlinear part is frozen (random features, a
  network's last layer with a least-squares head, kernels, fixed-pole rationals, mapped Chebyshev). For a fully trained network a
  uniform accuracy guarantee from samples needs exponentially many samples in depth and dimension (NN-65).
- **Families, briefly** (ratings and evidence in `APPROACHES.md`, section 3):
  - *one-step MLPs and ResNets*: fast inference (250x CPU and 4,250x GPU on a 77-species model, NN-32) and good in ambient dimension;
    poor tail errors, drift in rollouts and off-data;
  - *latent models* (autoencoder plus latent map; Theorem 8's design): suited to KIDA's dimension structure, more robust under
    iteration, less accurate (99th percentiles 3–4.8 dex in NN-39);
  - *neural operators* (DeepONet, FNO): nothing over a one-step network for 0-D chemistry; worth a test only on the lattices;
  - *PINNs*: fail on stiff kinetics without reduction (NN-6, NN-7); not recommended;
  - *Koopman, DMD, SINDy, transformers, foundation models, KANs*: not recommended for sup-error step maps (reasons in `APPROACHES.md`);
  - *rational bases* (AAA, lightning; NN-42, NN-43) and *mapped Chebyshev* (NN-46): strong for near-poles and layers in few variables;
    linear in data;
  - *kernels and random features*: linear in data; posterior variance or Lebesgue function computable;
  - *mixture of experts*: AFINNI and ART are mixtures with a geometric gate; partitioning by regime pays on stiff chemistry (NN-25).
- **The literature review's ranking** (`APPROACHES.md`, section 10): fix the gate first; then H1 (learned or random nonlinear features
  with a least-squares head and the exact Lebesgue gate, for fronts); H2 (a network decoder with a polynomial latent map and a
  normal-residual gate, for KIDA and mixing); H3 (an MLP one-step baseline with a conformal ensemble gate and fallback, the field's
  design measured on this project's terms); then rational or mapped axes for layers, and ISAT with a learned base. Its section 1 repeats
  the "approximation-limited fronts" reading that round 10 withdrew (it was written before round 10); its case for H1 must be re-read
  with section 3.6 above in mind.

---

## 8. Open problems

1. **The operator-split regime.** No scheme yet holds the tolerance under chained mixing at an acceptable refusal rate on a reduced
   chart (HIRES); full charts (CHEMAKZO) are accurate there but refuse 91%. This is the target's regime.
2. **The transient layer.** Held-out maxima sit at first steps; designed starts (greedy maximal-$\Lambda$ selection from a pool,
   zero solves to rank), offsets into the transient, and space-filling starts are untested.
3. **Lifts at high ambient dimension.** POLLU, Nelson and (so far) KIDA have no validated reduction with polynomial lifts.
4. **Fronts.** Whether Fisher–KPP's slow convergence is a moving front (poles that move across fibres) or a fixed layer decides between
   maps, rational axes, splitting, or local fits (the fibre test of round 10).
5. **The gate's calibration and cost.** A distributional threshold (held-out quantile) replaces a worst-case one; its coverage on mixed
   and rollout inputs, and a sketched $O(rp)$ bound, are untested.
6. **A verified certificate.** No number in this work is a verified bound; Theorem 8 with interval enclosures on one class is still open.

## 9. The proposal: what to experiment with, why, and what not to pursue

This section is the author's proposal and its reasoning, set out in full so that the reviewers can correct it, reject parts of it, or
add what it overlooks. It is deliberately narrow: the user asked that nothing be tried because it is fashionable, only what the
evidence says can help on stiff ODEs in general (chemistry included).

### 9.1 What the evidence says the bottlenecks are

1. **Where the data sit (factor 2).** In every problem examined in round 10 the worst held-out error sits at a first-step or early
   transient state, sampled once per trajectory per step index; the Lebesgue bound there is 10^2–10^5 while the in-sample error keeps
   falling. R5's fresh starts fixed three problems. This is a data-design problem, and no basis removes it.
2. **The gate (Corollary 5.3).** A wrongly calibrated threshold turned fits that hold the tolerance into "solver only" (round 10), and
   the refusal rate together with the gate's own cost sets the speed (p2: 10x on `lorenz96_d4` from 5.8% refused and a gate 6.6 times
   the fit's cost).
3. **Inputs from the coupled simulation (factor 4).** Under chained mixing, HIRES's accepted errors are 0.38 (R4) and, in R4b,
   `fisher_kpp_d2`'s are 1.5e-2 with nothing refused, a full chart. Every community that coupled a learned surrogate to a flow solver
   met the same drift (NN-26, NN-28, NN-34, NN-38, NN-39, NN-72, NN-73) and fixed it with coverage or online training. This is the
   target's regime (chemistry advanced cell by cell between MHD steps).
4. **Lifts at high ambient dimension (factor 1 through the chart).** POLLU (20), Nelson (18) and KIDA (582) have no validated reduction
   with linear charts and polynomial decoders; R3's KIDA lift errors were 3–7 dex at k = 5–7.
5. **Fronts (factor 1 or 2, unresolved).** Fisher–KPP and Allen–Cahn: round 10 showed sampling dominates the held-out maximum; whether a
   slowly converging approximation also limits them is open (Fisher–KPP's in-sample error falls only about 0.4 decades per degree).

6. **Calibration under a shift of inputs (from R4b).** The calibrated gate serves 26 of 37 problems, but on five full-chart problems it
   accepts mixed states that err beyond the tolerance while refusing almost nothing: calibration on undisturbed held-out states does not
   transfer to the inputs a coupled run produces.
7. **Selection against test (from R4b).** Four served problems pass the held-out chains and miss on the 40 test trajectories (by 1.25–2.5
   times the tolerance): a maximum over a few held-out trajectories is a weak estimate of the maximum over new ones.
8. **The gate's cost (from R4b).** At $d$ = 80 and wherever the fit has 1,000–2,000 terms the exact bound costs about as much as a solve,
   holding effective speed-ups to 1.2–4x at refusal rates of 2–7%.

### 9.2 The criterion that decides between families

The metric is the **maximum** error over every state visited, plus the speed that refusal leaves. That favours representations whose
error is *controllable where they are used*: a computable sampling factor (Proposition 5.1), a refusal rule with an accepted-error
statement (Corollary 5.3 (a)–(b)), and Theorem 8 for rollouts. These hold for any model that is **linear in its data once its nonlinear
choices are frozen**: polynomial charts and trees, Chebyshev schemes, kernels, random features, a network's last layer with a
least-squares head, fixed-pole rationals, local tables. For a fully trained network the computable sampling factor is lost, a uniform
guarantee from samples needs exponentially many samples (NN-65), refusal becomes heuristic or merely marginal (ensembles, conformal),
and the published tails on stiff chemistry are 30–280 times this project's tolerance at the 99th percentile (NN-39). So a learned
component earns a place only where it fixes a bottleneck that a linear-in-data design cannot, and only inside a gated linear design.

### 9.3 The shortlist, with the reasoning for each

**S1. The gated linear framework with designed data (a method, not a model).** *Addresses:* bottlenecks 1–3, for every scheme.
*Mechanism:* select training inputs where the sampling factor is worst: greedy maximal-$\Lambda$ selection from a pool drawn from the
initial law (pivoted QR, approximate Fekete points; ranking needs no solves), offsets into the transient ($\Phi_{\theta\,dt}(x_0)$ with
$\theta$ uniform on $[0, 3)$), and one-step samples at mixed states from the coupled regime; a held-out-calibrated gate, the decoder
residual term on reduced charts, and a propagation margin from held-out chains. *Why first:* it is the cheapest, it applies to every
scheme below, and the literature's own fix for coupled drift is coverage. *Cost:* minutes per problem; R5's 1,600 labelled extras allow
the greedy test at zero new solves. *Risk:* greedy selection picks the pool's edge, where the approximation factor is largest; mixed
states added to a *reduced* chart give inconsistent labels at the same chart value (round 10, counsel (b) 3).

**S2. In-situ local models in the ISAT style, gated by the Lebesgue function.** *Addresses:* bottleneck 3 directly, and 1 for the inputs
the coupled run actually produces. *Mechanism:* local linear (or quadratic) records around states the simulation visits, each with an
accuracy region; a query inside a record's region is answered from it, otherwise the solver steps it and the table grows (Pope 1997,
TH-62; the library's SlowTable). The project's theory adds what ISAT lacks: a computable sampling factor per local fit (Proposition 5.1),
the refusal accounting and speed cap (Corollary 5.3), and Theorem 8-type rollout statements per region. *Why:* it is the only design in
the literature built for inputs produced by a coupled run; it is proven at hundreds of species in combustion (with reduction, III-17); it
is generic for stiff systems with slow manifolds, not specific to chemistry; and the panel's off-data failure is exactly what in-situ
growth answers. *Cost:* the library has SlowTable; the experiment is to score it (and a Λ-gated variant) under chained mixing.
*Risk:* ISAT's error control is a quantile (III-13) and its rare large errors come from non-convex accuracy regions (III-14); memory grows
with the visited set; a high refusal rate early in a run.

**S3. AFINNI and ART under the same protocol.** *Addresses:* bottleneck 5 (fronts) and box-filling problems, where a global fit in
chart coordinates is the wrong space. *Mechanism:* local polynomials on adaptive partitions: a mixture of experts with a geometric gate;
AFINNI's trust regions are already a refusal rule. *Why:* local fits shrink each region, which shrinks the approximation factor where a
front sits, without a new basis; the library has them, and they have never been scored with the refusal rate, rollouts under mixing and
the effective speed-up beside the chart recipe. *Cost:* existing code; the benchmark's records give part of it. *Risk:* they fill the box
(T7), and ART's quadratic leaves fail at vertices (U5).

**S4. One conditional learned component: a network decoder as the chart's lift, for high ambient dimension.** *Addresses:* bottleneck
4 (POLLU, Nelson, KIDA). *Mechanism:* keep the chart's polynomial latent step map (it keeps Proposition 5.1 and Theorem 8 in latent
coordinates) and replace the polynomial decoder by a small network; the gate gets a residual term $\lVert x - D(\pi x)\rVert$ beside the latent
Lebesgue function. *Why:* the lift, not the latent map, is what failed at high dimension, and Proposition 6.5 bounds the latent dimension
from below whatever the decoder, so the question is well posed. It is the only learned piece whose failure is cheaply detectable before
any rollout. *Decisive first test (no rollouts):* the worst held-out reconstruction error of the decoder at $k$ from the lower count to
$k + 3$ on POLLU, Nelson, primordial and KIDA dense; if no decoder is within the tolerance, the idea is dropped. *Cost:* small networks
on CPU, minutes to an hour per problem. *Risk:* decoder errors in its tails (the literature's latent models: 99th percentiles 3–4.8 dex,
NN-39), and training cost on 578 outputs.

**One diagnostic, before any front-specific basis:** the fibre test of round 10 (Chebyshev tails and AAA poles along chart axes through
30 visited states; whether poles stay put across fibres). If they do, a rational or mapped axis (NN-42, NN-43, NN-46) inside ChebTT or
the chart fit is a candidate; if they move, local fits (S3) are the answer and no global basis will do.

### 9.4 What is not pursued, and why

| Family | Why not, on this project's evidence and the literature |
|---|---|
| End-to-end MLP or ResNet step maps | The published tails are 30–280x the tolerance at the 99th percentile (NN-39); rollouts drift and off-data inputs fail (NN-26, NN-28, NN-39); refusal is heuristic. Re-measuring this here would be the run "for its own sake" the user excluded. Reconsider only if S1–S4 all fail and a GPU is available. |
| Latent neural ODEs and linear latent (Koopman, DMD) models | Latent models are less accurate in tails (NN-39); a linear latent map cannot hold several attractors; a latent ODE still integrates. The useful part (a nonlinear lift) is S4. |
| PINNs, physics-informed operators | Fail on stiff kinetics without reduction (NN-6, NN-7); minimise a mean residual; replace data the project gets cheaply. |
| Neural operators (DeepONet, FNO) for 0-D chemistry | Nothing over a one-step network beyond continuous time; reported errors 1e-2 to 0.6 dex (III-27, NN-38). Possibly relevant for lattices only if S3 fails there. |
| Neural ODEs in the full state | Inference still integrates a stiff system; the known right-hand side is cheaper (VII-30, NN-26). |
| SINDy | The right-hand sides are known; the cost is the integration. |
| Transformers, reservoirs, foundation models | Step maps are Markov; successes are forecasts and statistics, not sup errors (NN-21, VII-18). |
| KANs, tensor networks with learned features | No evidence on worst-case error; they duplicate designs the project has with designed samples. |
| Random or learned features with a least-squares head (literature's H1) | Kept in reserve: round 10 found sampling, not the basis, behind the fronts' held-out maxima; only if the fibre test and S1 leave a genuine approximation limit. |
| Reduced ChebTT through a decoder on the chart box | Demonstration 8.5 cost 9,477 solves for a rollout of 0.30 dex on HIRES, worse than a solve-free fit; decoded box corners can be unphysical (KIDA). Reconsider for problems with a validated decoder and low $k$ if S1 leaves the transient undersampled. |

### 9.5 The plan

1. **Finish R4b** (running) and fill sections 3.7 and 3.8; state which problems the calibrated gate serves, by class.
2. **S1 data design**: greedy maximal-$\Lambda$ selection from R5's labelled extras (no solves); offsets into the transient against fresh
   starts at the same budget; mixed training states on full and reduced charts. On R5's twelve problems and every R4b-served problem.
3. **S2 in-situ models**: SlowTable and a Λ-gated variant under chained mixing at w = 0.01 and 0.1 on every problem with an R4b result,
   reporting the refusal rate over time, memory, and the accepted worst error.
4. **S3 local fits**: AFINNI and ART under the same protocol (rollouts, chained mixing, refusal per step, effective speed-up) on the
   panel, beside the chart recipe.
5. **S4 decoder test**: reconstruction only, on POLLU, Nelson, primordial, KIDA dense (and diffuse); continue to rollouts only where it passes.
6. **The fibre test** on the Fisher–KPP and Allen–Cahn ladders; a rational or mapped axis only if poles stay put.
7. **The decision rule across schemes** from the descriptors, scored against the benchmark's best method per problem with a class left out.

Every run: predictions committed before it; the whole panel by class (and quick probes on hard static functions and on ODEs of several
dimensions and tolerances where useful); maximum error on test inputs, rollouts and chained mixing; the refusal rate per step
(`refused_percent`, `refused_by_step`, `rollout_solved_by_step`); training and inference times, repeated, with the batch size; oracle
calls and data requirements; nominal and effective speed-ups; with and without gradients where the oracle gives them cheaply. Compute:
this laptop has no GPU; any model that cannot train in reasonable CPU time is trained as far as possible and marked "needs a GPU" for
TACC or an 8 GB RTX 2000.

### 9.6 Making ChebTT more versatile: candidates not yet posed

A tensor train already represents every interaction between inputs (a sum of products across all variables, the rank bounding how much
coupling it carries), and its inference is cheap (about d n r^2 operations per state, batched; under a millisecond at d = 578 for ranks
near 10). Neither interactions nor inference speed is what limits it in the records. What does:

1. **The box.** ChebTT is built by cross approximation on designed points of a full box; trajectories fill a thin region of it. Every
   ChebTT record at d >= 20 is a time limit or an evaluation cap, spent at vertices no trajectory visits (T7), not a failed fit.
2. **One train per output.** Fine for 3 species, prohibitive for 578 (U6).
3. **Rank growth with the variable order.** T3 keeps ranks small when coupling is local along the order (a lattice); a reaction network
   is a graph, not a chain.
4. **Jumps.** The relaxation oscillators' genuine approximation limit; no global smooth basis holds it.
5. **No exact sampling factor.** A TT cross divides by sampled pivot values (Astra), so Proposition 5.1 does not apply as it stands.

Candidates aimed at those limits, each keeping fast inference:

| Candidate | Limit it targets | Mechanism | What carries over | Risk |
|---|---|---|---|---|
| T-a. TT regression on trajectory data in chart coordinates | 1, 5 | cores fitted by alternating least squares on the visited states (no box, no cross); the box is the chart's occupied range | with all other cores frozen a core's fit is linear in the data, so a per-core Lebesgue bound (Proposition 5.1) and a calibrated gate exist; the gate of the final sweep is computable | ALS is non-convex; ranks must be chosen by held-out chains; data sparsity at the first step is the same as the chart's (S1 applies) |
| T-b. One train for all outputs | 2 | the output index as an extra core (a block TT), shared ranks across species | as T-a | output ranks may be large where species decouple; ordering of the output core |
| T-c. Order or tree from the coupling graph | 3 | variable order from the network's species graph (or the lattice), or a tree tensor network shaped like it | T3's light cone, with the tree's edges in place of the chain | trees need their own cross or ALS; gain only where coupling is local in the graph |
| T-d. Piecewise TT in leaves | 4 | a small TT per ART or ChebTree leaf, the split placed at the jump | T7's local version; ART's one-sample-serves-every-output | leaf boundaries carry inherited error; count of leaves near the jump |
| T-e. Rank and degree per core chosen by held-out chains | all | DMRG-style rank adaptation, selection by the propagation margin rather than in-sample error | the margin rule of generic3 | selection cost |

T-a and T-b address the two gaps that bind at the target's scale (the box at d >= 20, the cost per output at 578 species); T-c and T-d
are conditional on the records showing graph-local coupling or a jump the leaves can isolate.

### 9.7 Where things stood after sections 3.9-3.16 (the author's reading; round 2 corrected items 1, 3, 4 and 7, section 3.17)

1. **Coupling is the regime, and the closed column is the honest test of it.** Chained mixing pulls each cell toward a reference
   partner; the column, where neighbours carry surrogate error, gives 2.7-13x larger errors at the larger exchange for rollout-tuned
   ChebTT (section 3.12). Every surrogate that held chained mixing also holds the column at 0.01, but with less margin than mixing showed.
2. **Selection must see coupled chains.** HIRES under consistent calibration (section 3.16) is served with a plain rollout of 1.8e-6,
   accepted one-step defects on mixed inputs of at most 2e-3, and still a weak-mixing chain of 0.16: defects of 1e-3 per step add up
   over about 100 steps along directions the dynamics do not contract. The propagation margin is measured on plain held-out chains,
   which cannot see this; it has to be measured on mixed or column chains.
3. **Selection must see speed.** CHEMAKZO is served at an effective speed-up of 0.7 (slower than the solver), because the rule takes the
   smallest worst chain among served configurations; degree 8 served with chains of 2.8e-3 at a 5% solver share. Among configurations
   within the margin the rule should take the largest effective speed-up (Corollary 5.3), not the smallest error.
4. **Data at the first steps is the held-out problem, and its cost follows the initial law's dimension.** On fisher_kpp_d3, 400 fresh
   pairs chosen by leverage (no solves) cut the ungated first-step worst from 4.97 to 1.3e-2 (random 400: 0.19-0.38; all 1,600: 5.0e-3).
   On fisher_kpp_d10 no arm moves the first step (ungated worst 4.7-4.9, every first step refused): its initial law fills all 10
   dimensions, a test start lies 0.46 from its nearest pool state against 0.09 at d = 3, and the dynamics contract onto the k = 6 chart
   only after the first steps. A simulation's cells do not start from an i.i.d. box; they start near the states the training covers.
5. **The gate's calibration is the most fragile part of every scheme.** Three local-table gates calibrated on calibration states failed
   on chain states (section 3.15); the third recipe's maximum over mixed states refused all of HIRES until the calibration counted only
   states the residual term accepts (section 3.16).
6. **KIDA at 0.01 dex is out of reach with 200 tracers for every family tried**, the MLP included (section 3.14): coverage, not a hard
   species, limits it, and the best arm is local. The question for the target is the data (and the tube) more than the basis.
7. **A library bug of the same kind as KIDA's.** The numerical search for conservation laws (`invariants_of` without declared laws)
   returns spurious laws on every large lattice: Brusselator d20, d40, d80 (1, 16, 55 laws; the Brusselator conserves nothing), Allen-Cahn
   and Fisher-KPP d40 and d80 (11-55), Burgers d40 (partly). Each changes by about 100% of its scale along the test trajectories; the
   true laws (Robertson, HIRES, primordial, POLLU, Nelson, Burgers d10 and d20) hold to 1e-9 or better. The benchmark's `chebtt_roll`
   acceptance check and every method's projected scores enforce them on those problems. A fix (sampling the right-hand side off the
   data, since a linear law satisfies W F(y) = 0 for every y) needs the user's approval, as a change to `src`.

### 9.8 Where things stand after sections 3.17-3.22 (the author's reading, for the reviewers to correct)

1. **The repair was the largest gain of the whole study.** Removing the singular fits' per-call SVD and the per-state projection loop,
   and choosing configurations by cost, raised the median effective speed-up over the served problems from 14 to 54 with no loss of
   accuracy (3.18). Ten of 18 problems are served, all within 0.01 in rollouts, chained mixing and every paired-column setting except
   fisher_kpp_d3 at its margin (3.19).
2. **Coupling does not compound through neighbours** (closed and forced columns agree within 1.37x). Inside the bundle the coupled
   error is the cell's own one-step defect; the open question is the defect on off-bundle inputs, not feedback.
3. **The normal term is the mechanism off the bundle** (3.20). On POLLU and HIRES it cuts the off-bundle defect 5-28x in the median and
   up to 3,300x in the maximum. It needs mixed (tube) data in the fit, and it costs accuracy on the bundle. Neither mixed data alone nor a
   sensitivity chart replaces it on a reduced chart; on a full chart a sensitivity rotation alone gives 7-14x (CHEMAKZO).
4. **What still blocks the hard cases:**
   - HIRES: the gate's calibrated maximum. Its plain fit is excellent; the off-bundle fit is good but coarser on the bundle.
   - POLLU: accumulation. One-step errors after the transient are at most 1e-3, A_n = 234, and the rollout is 4.7e-2.
   - Nelson and primordial: fits too coarse everywhere (0.5-1.1 dex) with about 1,800 pairs in 10-16 coordinates; no index set helps.
   - lorenz96_d20: chaos with a full 20-coordinate chart.
   - The oscillators: the jump.
5. **The accumulation budget tol / (2 A_n) is as strict as Fable predicted.** It would refuse every unserved problem whole and 0-32% of
   the served lattices. A_n is a worst case over directions; the coherence-based budget is no looser on these records.
6. **KIDA:** the exact reference is sound (1.8e-6 dex), the solver's cost is its right-hand side (670 evaluations per step), and global
   coverage at 0.01 dex grows slowly with data (17% at 160 tracers, about 3,300 tracers for 50% by extrapolation).
7. **Not done:**
   - the trajectory-fitted tensor train (T-a) and the one-model-for-all-outputs screen;
   - the composite on/off-bundle model;
   - the checked-growth local table;
   - the KIDA rank-of-futures measurement;
   - the `src` fix for conservation laws (awaiting the user).

### 9.9 Where things stand after sections 3.23-3.27 (the author's reading, for the reviewers to correct)

1. **Data is cheap and decisive where the fit is close.** POLLU and Allen-Cahn d5 needed 240 more trajectories, seconds of solves.
   The data budget should be set per problem by a learning curve in solver seconds, not by a fixed 100 trajectories.
2. **The gate has two honest regimes, and neither is satisfactory alone.**
   - The product gate with plain selection serves 10 problems but does not certify coupled chains; HIRES fails weak mixing.
   - The envelope with the account certifies coupled chains wherever the envelope holds, but its floor is charged every step. It loses
     three lattices, and its linear-program envelope is not conservative enough on Burgers d20.
   - An account charged with a bound that tracks the actual defect (the floor removed by the account's own audits, or a two-level
     bound) is the open design.
3. **HIRES under coupling is limited by rare large defects.** Neither data nor a first-order correction moves them. The account handles
   them, at a high solver share.
4. **Against a compiled BDF the speed-ups are 3-100x**, not 3-800x. The surrogate pays most where the exact solver is expensive per
   step (Nelson 78 ms, primordial 8 ms, KIDA 0.5 s), and those are exactly the problems it does not serve yet.
5. Not done: the KIDA solver acceleration (fewer factorisations, a compiled right-hand side) and the conservation-law fix in `src`
   (awaiting the user); the measurement of how much solver time a local table could avoid on deployment-shaped KIDA inputs; the KIDA
   rank-of-futures measurement.

### 9.10 Where things stand after sections 3.28-3.33 (the author's reading, for the reviewers to correct)

1. **The capped gate, with data budgets of a few hundred trajectories, is the current recipe.** It serves 13 of 18 problems on plain
   rollouts and chained mixing, robustly across draws. The guarantee it carries is distributional, calibrated at the trajectory level.
2. **End to end, in fresh closed columns against the fastest CPU solver, it is worth 2-20x on nine problems** (median about 7). It is
   worth nothing on three: CHEMAKZO and POLLU, whose coupled inputs fall outside the cap, and HIRES, which also fails the tolerance
   there. The plain-rollout figure of 8.5-128x (median 39) overstates what coupling gives.
3. **Refusal is the lever.** The speed-up is at most 1/rho. Training on the refused states lowers rho more than random data at equal cost
   where the fit is close (3.32). Coupled inputs outside the cap are the largest remaining source of refusals.
4. Not done:
   - the KIDA solver acceleration;
   - the conservation-law fix in `src` (awaiting the user);
   - primordial's history-based predictor;
   - any GPU measurement (no GPU on this machine).

### 9.11 Where things stand after sections 3.34-3.37 (the author's reading)

1. **In steady state the recipe refuses nothing on most served problems.** On 9 of 11, refusals are a start-up transient. Against the
   fastest CPU solver the steady speed-up is 18-439x (median about 70), every chain within tol / 2 over 2,000 steps. The refusal rate
   that matters for a long simulation is the transient's, amortised over the run, plus whatever states the training did not cover.
2. **Refusal-driven data is the generic lever for coupled inputs.** It cuts the coupled refusal rate several-fold (55x on CHEMAKZO) and
   beats random data at equal cost on every problem tried.
3. **Coverage must match the deployment.** In time, the training trajectories must span the horizon (Robertson, CHEMAKZO). In coupling,
   the data must include coupled inputs, and refusal-driven data supplies them. The gate makes these gaps visible as refusals instead of
   errors.
4. **Still open:**
   - HIRES and POLLU under coupling (reduced charts; the chart must grow);
   - Nelson, primordial and the oscillators;
   - KIDA's solver and coverage;
   - every GPU measurement;
   - the conservation-law fix in `src`.

## 10. Questions for the reviewers (round 5)

Round 4's questions were answered in `theory/reviews/council4-*.md`; 3.28 lists what was adopted, 3.29-3.33 what was run.

The user's deployment picture, an example, not an existing setup: if the physics solver of a coupled simulation ran on a GPU, the stiff
chemistry beside it, on a much finer step, would be the bottleneck. The surrogate would then run on the GPU, batched over cells, and the
refusal rate would set the speed-up. Inference metrics matter most: GPU batching, the refusal rate, and anything else that affects
inference. Training is one large offline job, so heavy data and several GPUs for a while are acceptable, but it must not be absurdly
inefficient. Everything must stay generic: any ODE step map, scalar or vector-valued function, any scale, not a chemistry-specific
design.

1. **The plan from here.** Given 3.29-3.33 and 9.10, what next, in what order, with what decisive readings? Which readings in 9.10 do
   not follow?
2. **Cutting the refusal rate several-fold**, the user's first concern, since the speed-up is capped at 1/rho. Candidates:
   - refusal-driven data, iterated (3.32);
   - the burn-in rule (the solver takes only the first steps);
   - widening the cap where it is too strict for coupled inputs (CHEMAKZO and POLLU in 3.31);
   - local refits of refused regions;
   - better gates.

   For each: what bound survives, and the decisive test.
3. **GPU inference and batching.** Is the recipe's evaluation worth batching on a GPU, at which batch sizes and precision (float32 or
   float64, and the gate's decisions), against which comparators (one CPU core, a node, a batched GPU exact solver), with the 1/rho cap?
   Theory and the measurements to make; no code from the author yet.
4. **GPU training, as ART and AFINNI do it.** The library trains and predicts both on GPU with deterministic per-segment sums. Can the
   recipe's training (the selection's fits and SVDs, the calibration chains, the data generation) run on GPU the same way, so that
   training and inference both live there? Is it worth it, and what must stay deterministic?
5. **Scaling across several GPUs**, in training and in inference, the way neural networks scale with data-parallel training (DDP), a
   real advantage of deep learning even where its accuracy is not the best. How does this recipe compare:
   - the least squares as an all-reduce of Gram matrices or a tall-skinny QR;
   - the selection and the chains as independent work;
   - inference with no communication between cells?

   What would make it scale, and what would stop it?
6. **Other inference metrics** the user may not be aware of (latency against throughput, memory per cell, the cost of the account's
   state, fallback batching, precision), and how each should be measured.
