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

## 10. Questions for the reviewers

1. Is the proposal of section 9 right? What does it overlook, which of its choices or rejections are wrong, and what should be added or
   dropped? Should the next phase be narrower or broader? Which experiments, in what order, with what decisive readings?
2. Would the project gain more from improving its existing schemes (ChebTT and variants, ChebTree, AFINNI, ART, SlowTable, the chart
   recipe), for example making them more adaptive or giving them nonlinear functions (rational or mapped axes, learned features, local
   nonlinear terms), than from adding new model families? Use `theory/panel/data/scheme_table.md` (every scheme's records beside the
   recipe's) and the R4b results. The author's proposal (section 9) is one view to criticise, not the answer; drive the recommendation.
3. Which published claims in section 7 deserve the least trust, and which tests on this project's panel would check them?
4. What does the four-factor framework predict for each candidate family on each problem class, so that the predictions can be committed
   before the runs?
5. What is the minimum set of experiments that would tell the user whether to pursue learned surrogates at all for the astrochemistry
   target, given a CPU-only laptop now and a GPU later (TACC, or an 8 GB RTX 2000)?
