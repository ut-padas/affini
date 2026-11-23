#import "../companion/lib.typ": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#show: report.with(
  title: [Surrogates that hold a maximum error],
  subtitle: [A technical companion to the schemes: where a surrogate's error comes from, ChebTT on a box, the chart
    recipe on trajectories, and how both compare with AFINNI and ART],
  authors: ((name: [PADAS group], affiliation: [The University of Texas at Austin]),),
  date: [November 2026],
  abstract: [
    A surrogate stands in for an expensive function, or for one time step of a stiff ODE, and is useful only if
    its _maximum_ error over every input it is asked about stays within a stated tolerance, with the exact solver
    taking over wherever it cannot promise that. This companion starts from one decomposition of that error into
    four factors (how well the model's space can represent the function, how the data sit relative to the point
    of use, how the dynamics carry one step's error into the next, and what happens at inputs no data constrain),
    together with the rule that a refusal rate $rho$ caps the speed-up at $1 slash rho$. Each factor has a
    computable descriptor, and the two methods the companion explains are the two answers the decomposition
    suggests. ChebTT samples the function where Chebyshev interpolation wants it, along a few lines of a grid on
    a box, so its sampling factor stays logarithmic; it is the strongest scheme on every class of the function
    suite. The chart recipe fits the step map from trajectories alone, in coordinates of the states a simulation
    visits, and refuses every input where its own computable error bound exceeds the budget; it serves the stiff
    step maps whose boxes are mostly empty. One running example, a Brusselator cell at the tolerance 0.01, recurs
    in every picture. The last sections show both methods on every problem class and compare them, under one
    protocol, with AFINNI and ART at the tolerances 0.01 and 0.05.
  ],
)

#set math.equation(numbering: "(1)", supplement: [Eq.])
#show math.equation.where(block: true): it => {
  if it.has("label") or it.numbering == none { it } else {
    counter(math.equation).update(n => n - 1)
    math.equation(block: true, numbering: none, it.body)
  }
}

// ------------------------------------------------------------------- shared vocabulary
// The running example's numbers and colours (figures/make_figures.py) and the records' (figures/results.py), so that
// the text, the panels and the tables cannot disagree.
#let ex = json("figures/example.json")
#let rs = json("figures/results.json")
#let em = ex.methods
#let ch = em.chebtt_roll
#let af = em.afinni_roll
#let ar = em.art_roll
#let aq = em.art_quad_roll
#let rc = em.recipe_100
#let st = ex.settings
#let col = ex.colours
#let mcol(m) = rgb(col.method.at(m))
#let dark = rgb(col.dark)
#let red = rgb(col.red)
#let blue = rgb(col.blue)
#let grey = rgb(col.grey)
#let refused = rgb(col.refused)
#let panel(name, width: 100%) = image("figures/" + name + ".svg", width: width)
#show figure.where(kind: image): set block(breakable: false)
#let small(body) = text(8pt, body)
#let board(columns: 2, ..frames) = block(breakable: true, above: 1.1em, below: 1.1em, grid(
  columns: (1fr,) * columns,
  column-gutter: 1.0em,
  row-gutter: 0.9em,
  ..frames.pos().map(((picture, label)) => block(breakable: false, align(center)[
    #set par(justify: false)
    #picture #v(-0.3em) #small(label)
  ])),
))

// Numbers: thousands separators, two significant digits, scientific notation for small and large values.
#let num(x) = {
  if x == none { return [–] }
  let x = int(calc.round(x))
  if calc.abs(x) < 1000 { return str(x) }
  let s = str(x)
  let n = s.len()
  let out = ""
  for (i, c) in s.clusters().enumerate() {
    if i > 0 and calc.rem(n - i, 3) == 0 { out += "," }
    out += c
  }
  out
}
#let sci(x, digits: 2) = {
  if x == none { return [–] }
  if x == 0 { return [0] }
  let e = int(calc.floor(calc.log(calc.abs(x), base: 10)))
  let m = calc.round(x / calc.pow(10.0, e), digits: digits - 1)
  if calc.abs(m) >= 10 {
    m = m / 10
    e += 1
  }
  $#m dot 10^(#e)$
}
#let sig(x, n: 2) = {
  if x == none { return [–] }
  if x == 0 { return [0] }
  let e = int(calc.floor(calc.log(calc.abs(x), base: 10)))
  if e < -2 or e >= 5 { return sci(x, digits: n) }
  if e >= n - 1 { return num(x) }
  str(calc.round(x, digits: n - 1 - e))
}
#let pct(x, digits: 0) = if x == none [–] else [#calc.round(100 * x, digits: digits)%]
#let pctx(x) = if x == none [–] else if x == 0 [0%] else if x < 0.001 [#sig(100 * x)%] else if x < 0.1 [#calc.round(100 * x, digits: 1)%] else [#calc.round(100 * x, digits: 0)%]

// Notation: the AFINNI manuscript's where it has a symbol.
#let atol = $alpha_"tol"$
#let Lam = $Lambda_X$
#let lbl = rs.labels

// ---------------------------------------------------------------------- the algorithms
#let chebtt-alg = (
  require[the oracle $f$ on a box $B$, a tolerance $atol$, the safety factor $s = #st.chebtt.safety$],
  ensure[a tensor train of Chebyshev coefficients within $s atol$ at every check point],
  ln(0, [evaluate $f$ at $64 + 32 d$ Sobol points $X_c$], note: [the first check set]),
  ln(0, [$n_k <-$ the smallest degree at which #st.chebtt.n_fibre random fibres along axis $k$ have converged], note: [the degree pre-pass]),
  ln(0, [right index sets $J_k <-$ a few grid nodes], note: [rank $r_0 = #st.chebtt.rank0$]),
  ln(0, [*repeat*]),
  ln(1, [cores $<-$ #st.chebtt.sweeps alternating maxvol sweeps over fibres of the grid], note: [the cross]),
  ln(1, [errors $<-$ the train against $f$ at the check points, free grid points, look-ahead points]),
  ln(1, [*if* every error is within $s atol$: *return* the train, compressed]),
  ln(1, [*if* an axis has a kink or a jump: add a breakpoint there], note: [piecewise axes]),
  ln(1, [*else if* an axis's top coefficients are not yet small: double its degree]),
  ln(1, [*else*: raise the rank by #st.chebtt.rank_step, up to #st.chebtt.rank_max]),
)
#let chebtt-code = (label: <alg:chebtt>, lines: chebtt-alg)

#let roll-alg = (
  require[training trajectories, the step $Phi$ as oracle, the tolerance $atol$],
  ensure[a box surrogate whose chains along the training trajectories hold $atol slash 2$],
  ln(0, [$B <-$ the bounding box of the visited states, padded by 2% per axis]),
  ln(0, [$a <- atol$], note: [the one-step tolerance]),
  ln(0, [*repeat* at most three times]),
  ln(1, [one ChebTT per component of the increment $Phi(y) - y$ on $B$, at tolerance $a$], note: [@alg:chebtt; one cache of $Phi$]),
  ln(1, [$e <-$ the worst error of the surrogate chained along every training trajectory], note: [no solve: they are known]),
  ln(1, [*if* $e <= atol slash 2$: *return*]),
  ln(1, [$a <- 0.8 dot a dot max(atol slash (2 e), 1 slash 8)$]),
)
#let roll-code = (label: <alg:roll>, lines: roll-alg)

#let recipe-alg = (
  require[training, calibration trajectories; the tolerance $atol$; the step $Phi$ for calibration labels only],
  ensure[a fit $Psi$, its gate and its account],
  ln(0, [laws $W$: the declared conservation laws, else those that also hold off the data], note: [verified, not assumed]),
  ln(0, [chart $u = V_k^top (y - mu)$ on the free coordinates], note: [principal directions of the visited states]),
  ln(0, [*for* each degree $q = 1, 2, dots$], note: [selection]),
  ln(1, [*for* each of three splits by whole trajectory: fit on one third, calibrate on one, select on one]),
  ln(1, [cost $<-$ gate $+ (1 - rho)$ prediction $+ rho$ solve, on the held-out chains]),
  ln(0, [$q <-$ the cheapest degree whose chains hold $atol slash 2$ with $rho < 1 slash 2$]),
  ln(0, [fit: truncated SVD of the Chebyshev design $A = U S V^top$, kept above $#sci(st.rcond, digits: 1) dot s_1$], note: [minimum norm]),
  ln(0, [gate factor $M = V_r S_r^(-1)$, sketched to #st.sketch_columns columns], note: [@lem:bound]),
  ln(0, [envelope $a_0 + a_1 b$ over the calibration states with $b <= #st.beta sqrt(r)$], note: [a linear program]),
  ln(0, [account multiplier $lambda <-$ the smallest with at most #pct(st.alpha) of calibration chains above $atol slash 2$]),
)
#let recipe-code = (label: <alg:recipe>, lines: recipe-alg)

#let deploy-alg = (
  require[a batch of inputs $t$ (one per cell), each cell's account $S$],
  ensure[the next states, and which inputs went to the solver],
  ln(0, [$phi(t) <-$ the Chebyshev design of the chart coordinates], note: [one design for prediction and gate]),
  ln(0, [$hat(y) <- t + phi(t)^top c$, projected onto the laws $W$]),
  ln(0, [$b <- sqrt(N) norm(phi(t) M)$, #h(0.3em) $s <-$ the share of $phi(t)$ outside the fit's span]),
  ln(0, [refuse if $b > #st.beta sqrt(r)$, $s$ is beyond its calibration range, or $a_0 + a_1 b > atol slash 2$], note: [the gate]),
  ln(0, [$S <- gamma S$; #h(0.3em) $c <- m_0 + m_1 b$], note: [decay $gamma = #st.gamma$]),
  ln(0, [refuse if $lambda (S + c) > atol slash 2$; else $S <- S + c$], note: [the account]),
  ln(0, [the solver steps every refused input]),
)
#let deploy-code = (label: <alg:deploy>, lines: deploy-alg)

// ====================================================================== the text
= A step that costs a millisecond, needed a billion times <sec:intro>

#question[A simulation advances the chemistry of every cell by one stiff ODE step between two transport steps. The
  step is exact but costs tens to hundreds of microseconds, and is taken billions of times. What can stand in for it,
  how far can the stand-in be trusted, and what should happen where it cannot be?]

#scene(side: left, width: 40%, panel("states"))[
  The running example is one cell of the Brusselator, $u' = A + u^2 v - (B + 1) u$, $v' = B u - u^2 v$ with
  $A = #ex.bruss.A$ and $B = #ex.bruss.B$, in the coordinates $(log_10 u, log_10 v)$, stepped by
  $tau = #rs.problems.step_maps.bruss_d2.dt$. Its _step map_ $Phi$ takes a state to the state one step later; the exact
  solver computes it. The benchmark integrates #ex.train training trajectories (their #swatch(grey) states) and
  #ex.test #swatch(dark, shape: "line") test trajectories of #ex.steps steps from random starts (circles). Just below
  the Hopf point $B = 1 + A^2$, they spiral slowly into the steady state $(A, B slash A)$. The
  #swatch(dark, shape: "dash") dashed rectangle is the padded bounding box of the training states; only
  #pct(ex.visited_cells_share) of its $10 times 10$ cells contain a training state.
]

A surrogate $Psi$ of $Phi$ is a cheap function that agrees with it closely enough _wherever it will be asked_.
"Closely enough" here means the maximum pointwise error, over every state a simulation visits, within a tolerance
$atol$; the running example uses $atol = #ex.tol$ in $log_10$ units. Two facts shape everything that follows. First,
the inputs a surrogate is asked about are not test points drawn from a box: they are the states of chains that the
surrogate itself produces, mixed with the neighbouring cells' states by transport. Second, a surrogate need not
answer every input. It may _refuse_ one, and the solver steps it exactly. A refused input costs a solve but adds no
error, so the right question is not "is the surrogate accurate everywhere?" but "can it tell, before the step is
known, where it is accurate, and how often must it refuse?".

This companion explains two methods that answer that question in two settings, and the theory that chose them.

#block(breakable: false, above: 1em, below: 1em, table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[][ChebTT (@sec:chebtt)][The chart recipe (@sec:recipe)],
  [data], [the oracle, queried anywhere on a box, at points the method chooses], [trajectories the simulation would compute anyway, and calibration labels],
  [the model], [a tensor train of Chebyshev coefficients on the box], [a least-squares Chebyshev fit in the coordinates of the visited states],
  [where it answers], [anywhere in the box], [where its computable error bound is within the budget],
  [it refuses], [inputs outside the box], [inputs its gate or its error account rejects],
  [it is strongest on], [functions and step maps of moderate dimension that fill their box], [step maps whose states lie on a thin set: stiff chemistry],
))

The theory (@sec:theory) comes first, because it says why these two and not others: why designed samples on a box
work, why trajectory samples need a refusal rule, why a one-step tolerance does not hold over a rollout, and why the
refusal rate decides the speed-up. The problems and the protocol follow (@sec:problems, @sec:protocol), then the two
methods, and the results on every class of problem (@sec:results). The last section compares both with AFINNI and
ART (@sec:compare), the adaptive local methods of the other companion @companion, under one protocol.

== How to read this companion <sec:reading>

Each idea appears three times: as a *picture* of the running example, in plain *words*, and in *symbols*, in a tinted
box. The pictures are computed, not drawn: `figures/make_figures.py` builds every method on the running example
exactly as the comparison study does, and stops unless each one reproduces the study's recorded rollout error.
`figures/results.py` reads every other number from the committed records. Colours keep one meaning: each method has
its own colour in every panel (#swatch(mcol("chebtt_roll")) ChebTT, #swatch(mcol("recipe_100")) the chart recipe,
#swatch(mcol("afinni_roll")) AFINNI, #swatch(mcol("art_roll")) ART with affine leaves, #swatch(mcol("art_quad_roll"))
ART with quadratic leaves). In an error map, a point is pale #swatch(rgb(col.error.at(0))) to
#swatch(rgb(col.error.at(3))) blue within the tolerance, #swatch(rgb(col.error.at(4))) pink to #swatch(red) red
beyond it, and #swatch(refused) grey where the surrogate refuses.

#translate(
  [the step map, the surrogate, a state and its dimension], [$Phi$, $Psi$, $y$, $d$],
  [the tolerance (absolute, in the state's coordinates)], [$atol$ #h(0.3em) (`tol`)],
  [the data's inputs; a point of use], [$X = {x_1, dots, x_N}$, $t$],
  [a model space and its basis; its best error on a set $S$], [$V = "span"{phi_1, dots, phi_p}$, $e^*_V (S) = inf_(q in V) norm(Phi - q)_(infinity, S)$],
  [the Lebesgue function of the data at $t$], [$Lam (t) = sum_i |ell_i (t)|$],
  [the accumulation factor of a chain of $n$ steps], [$A_n$],
  [the refusal rate; the speed-up over the solver], [$rho$, $S$],
  [the costs of a solve, of a surrogate evaluation, of the refusal test], [$c_Phi$, $c_Psi$, $c_g$],
)

= Where a surrogate's error comes from <sec:theory>

#question[A surrogate was built from data at the states $X$. It is used at a state $t$ it was not built on, and again
  at the state its own answer leads to. What bounds its error there, and which parts of that bound can be computed
  before the true step at $t$ is known?]

#figure(diagram(
  spacing: (14mm, 3mm),
  node-stroke: 0.6pt,
  node-corner-radius: 3pt,
  node((0, 0), [the space $V$], name: <v>),
  node((1, 0), [the data $X$], name: <x>),
  node((2, 0), [a point of use $t_j$], name: <t>),
  node((3, 0), [the chain $t_0, t_1, dots$], name: <c>),
  node((0, 1), text(8pt)[approximation \ $e^*_V$], stroke: none),
  node((1, 1), text(8pt)[sampling \ $1 + Lam (t_j)$], stroke: none),
  node((2, 1), text(8pt)[off the data \ $t_j$ far from $X$], stroke: none),
  node((3, 1), text(8pt)[propagation \ $A_n$], stroke: none),
  edge(<v>, <x>, "->"), edge(<x>, <t>, "->"), edge(<t>, <c>, "->"),
), caption: [The four factors of a surrogate's error where it is used, in the order a build meets them.]) <fig:factors>

A surrogate errs where it is used for four reasons, and they multiply. The model's space may not contain anything
close to $Phi$ (_approximation_). The data may sit badly relative to the point of use, so that a fit exact on its data
is wrong between them (_sampling_). The point of use may lie where no data constrain the fit at all, because transport
moved a state off the set the trajectories visit (_off the data_). And along a chain, each step's error is carried and
possibly amplified by the dynamics into every later step (_propagation_).

#proposition(title: [the four factors])[
  Let $Psi$ be built in $V$ from data at $X$ and used along a chain with inputs $t_0, t_1, dots$. After $n$ steps its
  error is of order
  $ A_n max_(j < n) (1 + Lam (t_j)) e^*_V (X union {t_j}). $ <eq:factors>
  Each factor has a computable descriptor: $e^*_V$ is estimated by held-out residuals, $Lam$ is computed from the
  data's inputs alone, and $A_n$ from Jacobians along trajectories. A refusal rate $rho$ caps the speed-up at $1 slash
  rho$ (@cor:speed).
] <prop:factors>

@eq:factors is an estimate, not a bound: the chain's inputs depend on the surrogate's own answers, so a bound needs the
defect bounded on a whole tube around the chain. The sections below take the factors one at a time, each with the
example; the methods of @sec:chebtt and @sec:recipe are the two ways of keeping all four small.

== Sampling: the Lebesgue function at the point of use <sec:lebesgue>

#scene(side: right, width: 46%, panel("lebesgue-1d"))[
  Fit a polynomial of degree #ex.lebesgue_1d.degree on $[-1, 1]$. Its value at $t$ is a weighted sum of the data,
  $sum_i ell_i (t) f(x_i)$, and the weights' absolute sum is the amount by which an error in the data, or in the
  model's best fit, can be amplified at $t$. Interpolating at the #(ex.lebesgue_1d.degree + 1)
  #swatch(mcol("chebtt_roll")) Chebyshev points (the upper ticks), it stays below #sig(ex.lebesgue_1d.cheb_max)
  everywhere. Fitting by least squares to #ex.lebesgue_1d.samples #swatch(mcol("recipe_100")) points that cover only
  $[-1, #ex.lebesgue_1d.edge]$ (the lower ticks), as a trajectory's states cover only part of a box, it stays below
  #sig(ex.lebesgue_1d.part_inside_max) among them and reaches #sci(ex.lebesgue_1d.part_max) beyond them.
]

The weights' absolute sum is the _Lebesgue function_ of the data. It depends on the data's inputs and the space, not on
the function, so it can be computed at any input before the value there is known. That is the property everything
later builds on.

#translate(
  [the fit's weights on the data at $t$], [$ell (t) = A (A^top A)^(-1) phi(t)$, #h(0.3em) $A_(i j) = phi_j (x_i)$],
  [the fit at $t$], [$hat(f)(t) = sum_i ell_i (t) f(x_i)$],
  [the Lebesgue function], [$Lam (t) = sum_i |ell_i (t)| = norm(ell (t))_1$],
)

#proposition(title: [the error at a point of use @cohen2013])[
  For every $t$ and every $q in V$, the least-squares fit $hat(f)$ satisfies
  $|f(t) - hat(f)(t)| <= |f(t) - q(t)| + Lam (t) max_i |f(x_i) - q(x_i)|$, hence
  $ |f(t) - hat(f)(t)| <= (1 + Lam (t)) e^*_V (X union {t}). $ <eq:lebesgue>
] <prop:lebesgue>

#proof[The fit reproduces $V$, so $hat(f)(t) - q(t) = sum_i ell_i (t) (f(x_i) - q(x_i))$; add $f(t) - q(t)$ and take
  absolute values.]

#board(
  (panel("lebesgue-cheb"), [Interpolation on the box's Chebyshev grid of degree #ex.lebesgue.cheb_degree per axis:
    $Lam <= #sig(ex.lebesgue.cheb_box_max)$ everywhere.]),
  (panel("lebesgue-ls"), [Least squares of total degree #ex.lebesgue.ls_degree on the #num(ex.lebesgue.ls_samples)
    training states: $Lam <= #sig(ex.lebesgue.ls_test_max)$ on the test states, up to #sci(ex.lebesgue.ls_box_max) in the box.]),
  columns: 2,
)

In two dimensions the picture is the same. The designed grid keeps $Lam$ logarithmic in the degree everywhere in the
box. The trajectory states keep it moderate where the test trajectories go (median
#sig(ex.lebesgue.ls_test_median) on the test states) and let it explode in the box's empty corners: it exceeds 10 on
#pct(ex.lebesgue.ls_box_share_above_10) of the box. Three regimes follow, whatever the problem:
- *targets among the samples* (a thin visited set the samples cover): $Lam$ stays moderate as the degree rises, so the
  degree can rise until $e^*_V$ is small;
- *targets between sparse samples* (a box the samples do not fill, a state moved off the visited set): $Lam$ grows
  quickly with the degree, and a fit exact on its data can be arbitrarily wrong there, invisibly to any in-sample check;
- *designed samples* (a Chebyshev grid, the points an oracle-queried scheme chooses): $Lam$ is the grid's Lebesgue
  constant, logarithmic in the degree per axis, and the box is covered uniformly.

ChebTT lives in the third regime and the chart recipe in the first; the second is the one each must refuse.

#pitfall(title: [Trajectory data are poorly placed for global fits])[
  A trajectory contributes one scattered state, its start, and then states the dynamics pull together. The transient
  is sampled once per trajectory and step, and every held-out maximum examined in this work sat there. More
  independent starts help a fit from trajectories; longer trajectories of the same starts do not, until the deployment
  runs longer than the training did (@sec:time).
]

#takeaway[The error at a point of use is the space's best error amplified by $1 + Lam (t)$, and $Lam (t)$ is known
  before the step at $t$ is.]

== Approximation: how fast the coefficients fall <sec:approx>

#question[How many terms does the space need before $e^*_V$ is below the tolerance?]

For a smooth step map the answer is classical. A function analytic inside a Bernstein ellipse $E_rho$ around $[-1, 1]$
has Chebyshev coefficients that fall like $rho^(-n)$, so each decade of accuracy costs a fixed number of degrees,
about $1 slash log_10 rho$ per axis.

#theorem(title: [geometric convergence @trefethen2019])[
  If $f$ is analytic in the open Bernstein ellipse $E_rho$, $rho > 1$, and $|f| <= M$ there, its interpolant $p_n$ at
  $n + 1$ Chebyshev points satisfies $norm(f - p_n)_infinity <= 4 M rho^(-n) slash (rho - 1)$.
] <thm:geometric>

Two things make stiff step maps harder than this suggests. Stiffness shrinks the ellipse: a fast mode that relaxes in
time $epsilon$ puts the step map's nearest complex singularity at a distance of order $epsilon slash tau$ from the
states, so a box that reaches far from the slow set needs a high degree. And dimension multiplies the degrees: a dense
tensor grid costs $product_k (n_k + 1)$ values. ChebTT answers the second with a low-rank format (@sec:chebtt). The
chart recipe answers both by fitting only where the states are, in their own coordinates (@sec:recipe): near the slow
set the map is smooth, and the set has fewer dimensions than the box. Where trajectories collapse onto a slow manifold
@fenichel1979, the map factors through it; where they fill a box (lattices, chaotic systems) no such reduction exists
at these horizons.

#takeaway[Smoothness fixes the degrees per decade; stiffness and dimension decide whether a box or the visited set is
  the right domain.]

== Propagation: what a chain does to one step's error <sec:propagation>

#question[A surrogate within the tolerance on every one-step test can still fail a rollout. Why, and by how much?]

#figure(placement: auto, panel("rollout"), caption: [Left: the worst error over the #ex.test test trajectories after each step, for every
  method (solid), beside its worst _one-step_ error at the true states (dotted), with the #swatch(red, shape: "dash")
  budget $atol slash 2$. Right: the share of inputs each refused.]) <fig:rollout>

The rollout error is the one-step error carried forward: on @fig:rollout it is #sig(ch.amplification) times the worst
one-step error for #swatch(mcol("chebtt_roll")) ChebTT, #sig(af.amplification) times for #swatch(mcol("afinni_roll"))
AFINNI and #sig(ar.amplification) times for #swatch(mcol("art_roll")) ART.


The error of a chain obeys a simple recurrence. Write $y_k$ for the true states and $z_k$ for the chain's. Then
$z_(k+1) - y_(k+1) = (Phi(z_k) - Phi(y_k)) + delta_k$, with $delta_k = Psi(z_k) - Phi(z_k)$ the surrogate's one-step
defect at its own state: the dynamics carry the old error and the surrogate adds a new one.

#proposition(title: [the global error recurrence @hairer1996])[
  If $Phi$ is Lipschitz with constant $L_k$ on a set holding $y_k$ and $z_k$, then
  $ norm(z_n - y_n) <= sum_(k < n) (product_(k < j < n) L_j) norm(delta_k). $ <eq:recurrence>
  For a contracting map ($L_j < 1$) old defects fade; for a neutral one ($L_j approx 1$) they add up; for a chaotic one
  they grow geometrically. A step refused to the exact solver has $delta_k = 0$, and keeps propagating the earlier
  errors.
] <prop:recurrence>

The _accumulation factor_ $A_n$ is the measured ratio of the worst $n$-step error to the worst one-step error along
the training trajectories. Two consequences shape both methods. A surrogate built to a one-step tolerance $atol$ does
not hold $atol$ over a rollout; it needs a margin. ChebTT and the box methods get it by retraining at a tighter
one-step tolerance until the chains along the training trajectories hold $atol slash 2$ (@alg:roll). The chart recipe
gets it from its account: each cell carries the sum of its accepted steps' error charges and refuses a step that
would take the sum past the budget (@sec:account).

#takeaway[A rollout multiplies the one-step error by the dynamics' accumulation factor; a one-step tolerance needs a
  margin, or a per-cell account of the error spent.]

== Off the data, refusal and speed <sec:refusal>

#question[If the surrogate refuses whatever it cannot vouch for, how much speed is left?]

#scene(side: right, width: 44%, panel("speedup"))[
  Each curve is $c_Phi slash (c_Psi + rho c_Phi)$ for one method's measured evaluation cost, against the refusal
  rate $rho$, with $c_Phi = #sig(ex.denominator_us)$ µs, the fastest exact solver on the example. The
  #swatch(dark, shape: "dash") dashed line is $1 slash rho$. Every curve bends onto it: once a method refuses more than
  about $c_Psi slash c_Phi$ of its inputs, its own speed no longer matters. The dots are each method's steady state on
  2,000-step chains (@sec:protocol), drawn at $rho = 10^(-5)$ where no steady step was refused.
]

Inputs off the data are where @prop:lebesgue's factor $Lam (t)$ is large. Operator splitting produces them at every
step: transport mixes a cell's state with its neighbours', and the mixed state need not lie on the set any trajectory
visited. A surrogate can be made safe there by refusing, and the price of refusing is simple to state.

#proposition(title: [accuracy, refusal and speed])[
  Let the surrogate refuse $t$ when a computable bound of its error at $t$ exceeds the budget, the solver stepping what
  it refuses, and let $rho$ be the share refused. (a) $rho$ is computable from unlabelled inputs before deployment. (b)
  Every accepted input errs by at most the bound. (c) The expected cost per step is $c_g + (1 - rho) c_Psi + rho c_Phi$,
  so the speed-up over the solver is
  $ S = c_Phi / (c_g + (1 - rho) c_Psi + rho c_Phi) <= min(1 / rho, c_Phi / (c_g + (1 - rho) c_Psi)). $ <eq:speed>
  (d) Along a chain, refused steps add no defect (@prop:recurrence with $delta_k = 0$).
] <cor:speed>

#proof[(a) is the definition, (b) the bound's, (c) linearity of expectation, (d) the recurrence.]

This changes what "a good surrogate" means. A surrogate a thousand times cheaper than the solver that refuses
5% of its inputs is at most 20 times faster. A deployment that runs on a GPU, where the surrogate's evaluation is
nearly free, is limited by $rho$ alone. Lowering $rho$ at a fixed maximum error is therefore worth as much as making
the evaluation faster, and the results report it beside every error (@sec:results).

#pitfall(title: [A theorem's constant is not a threshold])[
  An early refusal rule took the worst-case constant of @prop:lebesgue as its threshold and added a floor; it
  double-counted the sampling factor and refused nearly everything. A threshold is calibrated on held-out data and
  labelled distributional (@sec:account); the theorem says which statistic to calibrate, not where to cut it.
]

#takeaway[Refusal makes off-data inputs safe and costs speed: $S <= 1 slash rho$, whatever the surrogate costs.]

= The problems <sec:problems>

#question[Which problems decide between methods, and what makes each class hard?]

The benchmark has two suites. Every claim in this companion is measured on whole classes, never tuned on one problem:
a recipe tuned on one stiff system (HIRES) failed on the others, because that system's trajectories collapse onto a
thin curved set that is unusually easy to cover.

== The function suite

Each function is evaluated on a box $[a, b]^d$ at the dimensions listed. A vector-valued function's components are
divided by their maximum magnitude on the test set, so one tolerance is the same relative accuracy for each.

#let fclass-desc = (
  "smooth": [analytic functions of several kinds of coupling: a curved valley, an anisotropic Gaussian, a plane wave, Genz's integrands],
  "many local features": [oscillating functions with many local extrema in every direction],
  "kinks, jumps, fronts": [a kink, jumps across a plane, an oblique plane and a sphere, a steep smooth front],
  "ODE flow maps": [the flow of an ODE over a fixed time, as a function of the initial state (one output)],
  "vector-valued": [several outputs at once: a gradient, a state of a flow map, outputs sharing a latent coordinate],
)
#block(breakable: false, table(
  columns: (auto, 1fr, auto),
  inset: (x: 5pt, y: 4pt),
  align: (left, left, left),
  table.header[class][the functions][dimensions],
  ..for (cls, fns) in rs.problems.functions {
    ([*#cls*], [#fclass-desc.at(cls) \ #small(raw(fns.keys().join(", ")))],
     [#fns.values().map(f => f.dims).flatten().dedup().sorted().map(str).join(", ")])
  },
))

== The step maps <sec:stepmaps>

Each step map advances an ODE system by a fixed time $tau$. The systems span the classes a chemistry or plasma code
meets:
- *stiff chemistry*: Robertson (rates over nine decades), HIRES, POLLU (air pollution), CHEMAKZO, primordial gas
  chemistry, and a reduced astrochemical network (Nelson and Langer) whose held inputs, the environment a simulation
  supplies (density, temperature, radiation, cosmic-ray rate), are not stepped;
- *oscillators*: Van der Pol and the Oregonator, with relaxation jumps;
- *reaction-diffusion lattices* at $d$ = 2 to 80: the Brusselator, Allen–Cahn (with an unstable state between two wells),
  Fisher–KPP fronts, viscous Burgers; these fill their boxes as the dimension grows;
- *non-stiff chaos*: Lorenz-96 at $d$ = 4 to 40.

Chemistry is modelled in $log_10$ abundances, because abundances span many decades and a maximum error in $log_10$ is a
relative error in the abundance; conservation laws are projected out exactly. The end target is a network of hundreds
of species (KIDA, 578 species and 8,275 reactions @wakelam2012) advanced cell by cell in a magnetohydrodynamics code.

#let sgroups = (
  ([stiff chemistry], ("robertson", "hires", "pollu", "chemakzo", "primordial", "nelson")),
  ([oscillators], ("van_der_pol", "oregonator")),
  ([lattices], ("bruss_d2", "bruss_d10", "bruss_d20", "allen_cahn_d2", "allen_cahn_d3", "allen_cahn_d5", "allen_cahn_d10", "allen_cahn_d20",
    "fisher_kpp_d2", "fisher_kpp_d3", "fisher_kpp_d5", "fisher_kpp_d10", "fisher_kpp_d20", "burgers_d10", "burgers_d20")),
  ([non-stiff chaos], ("lorenz96_d4", "lorenz96_d5", "lorenz96_d10", "lorenz96_d20")),
)
#figure(table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  inset: (x: 5pt, y: 2.5pt),
  align: (left, right, right, right, right, right, left),
  table.header[step map][$d$][held][$tau$][steps][train / test][coordinates],
  ..for (g, names) in sgroups {
    (table.cell(colspan: 7, emph(g)),)
    for n in names {
      let p = rs.problems.step_maps.at(n)
      (raw(n), str(p.d), str(p.held), sig(p.dt), str(p.steps), [#p.train / #p.test], if p.log [$log_10$] else [linear])
    }
  },
), caption: [The step maps used in this companion: dimension, held inputs, step, trajectory length and counts, and
  coordinates. The lattices also run at $d$ = 40 and 80 in the benchmark.]) <tab:stepmaps>

#takeaway[The function suite tests the space; the step maps test the data, the refusal and the chain.]

= How the data are built and how a method is judged <sec:protocol>

#question[What does each method get to see, and what counts as success?]

Fairness here means one thing: every method gets the same data budget rules and is judged by the same protocol with
the same definition of a refusal. The definitions were fixed before the runs, after early mistakes showed how easily a
protocol flatters a method (the pitfalls below).

*The function suite.* A method queries the oracle anywhere on the box, as often as its own error estimate asks. It
is judged on a fixed test set (a $#rs.problems.function_test.grid times #rs.problems.function_test.grid$ grid in 2-D,
#num(rs.problems.function_test.n_test) uniform points above) by its _worst ratio_, the worst error over the tolerance.
A case (a function, a dimension and a tolerance) is _held_ when the worst ratio is at most 1 on every one of
#rs.problems.function_test.seeds seeds, and _won_ by the held method with the fewest evaluations. The runs use 8 CPU ranks.

*The step maps.* The data are trajectories from the problem's initial law. The _box methods_ (ChebTT, AFINNI, ART) may
query the solver anywhere in the padded bounding box of the training states and refuse outside it; AFINNI also
refuses outside every trust region. The chart recipe sees only trajectories: the 100 training trajectories (or 340 for
its larger version), 100 more for calibration, whose solver calls are counted. Every method is then run through the
same protocol, in which a _refusal_ is a cell-step handed to the solver, whatever the reason:

#cards(columns: 3,
  ([Rollouts], [each of the 40 test trajectories chained from its start; the worst error, and the refusal rate over steps 1–10 (start-up) and 11–40 (steady)]),
  ([Chained mixing], [each step, a state moved a share $w = 0.01$ or $0.1$ of the way to another trajectory's state, as transport does]),
  ([Closed columns], [100 fresh trajectories as a column of cells, sorted or unsorted, with diffusion $D = 0.005, 0.05, 0.25$ between cells and the surrogate stepping each cell]),
  ([Long chains], [64 trajectories of 2,000 steps: the start-up and steady refusal rates, and the worst error over the whole horizon]),
  ([Speed-up], [the evaluation and the gate timed at 1,000 cells per call, against the fastest exact solver (LSODA or VODE); @eq:speed at the steady refusal rate]),
  ([Cost], [the training time and every solver call: oracle queries, trajectory steps, calibration labels]),
)

#figure(diagram(
  spacing: (9mm, 6mm),
  node-stroke: 0.6pt,
  node-corner-radius: 3pt,
  node((0, 0), [cells $y^((1)), dots, y^((m))$], name: <a>),
  node((1, 0), [transport \ #small[$y^((i)) <- (1 - 2D) y^((i)) + D (y^((i-1)) + y^((i+1)))$]], name: <b>),
  node((2, 0), [gate], name: <g>),
  node((3, -0.7), [surrogate $Psi$], name: <s>),
  node((3, 0.7), [solver $Phi$], name: <p>),
  node((4, 0), [next states], name: <n>),
  edge(<a>, <b>, "->"), edge(<b>, <g>, "->"),
  edge(<g>, <s>, "->", [accept]), edge(<g>, <p>, "->", [refuse], label-side: right),
  edge(<s>, <n>, "->"), edge(<p>, <n>, "->"),
  edge(<n>, (4, 1.5), (0, 1.5), <a>, "->", [the next step]),
), caption: [A closed column: transport, then the chemistry of every cell by the surrogate or, where it refuses, the
  solver, then the next transport. The inputs are mixtures no training trajectory visited.]) <fig:column>

#pitfall(title: [Mistakes the protocol now excludes])[
  - *Selecting by in-sample error.* A degree chosen by its error on the training trajectories themselves looked best
    exactly where it was wild between samples. Selection is now held out by whole trajectories, with every transform
    fitted inside the fold.
  - *Clamping to the box.* Clipping inputs to the training range silently extrapolated on CHEMAKZO; inputs outside the
    validated range are refused, not clipped.
  - *A partial cohort.* A headline once used 20 of 40 test trajectories, with a degree range that stopped one short of
    the good degrees. Every test trajectory is reported, and no search range ends where the answer might start.
  - *Counting refuse-everything as success.* A surrogate that refuses every input has zero error. "Served" now means
    within $atol slash 2$ on every chain _and_ refusing less than half of the inputs.
]

#takeaway[One protocol for everyone: the same trajectories, the same chains and columns, one definition of a refusal,
  and every solver call counted.]

= ChebTT: one model for the whole box <sec:chebtt>

#question[The oracle can be asked anywhere on a box, and the answer must hold everywhere in it. What is the cheapest
  model that does, as the dimension grows?]

== The candidates, and where they stumble

- *A dense Chebyshev grid.* Optimal sampling (@sec:lebesgue) and geometric convergence (@thm:geometric), but
  $product_k (n_k + 1)$ values: $20^(10) approx 10^(13)$ at $d = 10$.
- *Sparse grids.* Fewer points for functions of few interacting variables, but the Lebesgue constant grows faster and
  oblique features defeat them; the benchmark runs SparseCheb only up to $d = 3$.
- *Local pieces* (AFINNI, ART). Each piece needs its own data, and the pieces must tile the box to the tolerance. An
  affine piece's error is quadratic in its size, so halving the error halves the piece's width in every direction:
  their evaluations grow like $atol^(-d slash 2)$, where a global expansion's grow like $log(1 slash atol)$.

ChebTT keeps the grid and compresses it.

== A tensor train, sampled along fibres

#definition(title: [ChebTT @oseledets2011])[
  A _tensor train_ of Chebyshev coefficients represents
  $ f(x) approx sum_(alpha_0, dots, alpha_d) G_1 [alpha_0, j_1, alpha_1] dots.c G_d [alpha_(d-1), j_d, alpha_d] product_k T_(j_k) (s_k (x_k)), $
  with cores $G_k$ of shape $r_(k-1) times (n_k + 1) times r_k$, $r_0 = r_d = 1$, $T_j$ the Chebyshev polynomials and
  $s_k$ the affine map of axis $k$ onto $[-1, 1]$. It stores $sum_k r_(k-1) (n_k + 1) r_k$ numbers, linear in $d$ for
  bounded ranks, and evaluates in the same order of operations.
] <def:tt>

A product of univariate functions has rank 1; $cos(q dot x)$ has rank 2; a smooth function of a few oblique
coordinates has a low rank in practice. The train is built by _cross approximation_ @oseledets2010: instead of the full
grid, it evaluates $f$ on _fibres_, lines of the grid along one axis through a few chosen points of the others, and
chooses those points by the maxvol rule @goreinov2010, which picks the rows of a fibre matrix with the largest volume.

#algorithm(caption: [ChebTT: `chebtt.build` in `src/schemes/chebtt/train.py`.], pseudocode(chebtt-alg)) <alg:chebtt>

#story()
#step([Fibres, not a grid], code: chebtt-code, focus: (2, 3, 5), width: 40%, panel("samples-chebtt_roll"))[
  On the running example each component of the increment $Phi(y) - y$ is a train of degrees
  #ch.degrees.map(d => d.map(str).join(" and ")).join(", ") and ranks #ch.ranks.map(r => r.map(str).join(", ")).join(" and ").
  The #swatch(mcol("chebtt_roll")) dots are every state the build asked the solver about:
  #num(ch.points_drawn) solves on lines of the grid, the check points and the look-ahead points. They cover the box
  uniformly: #pct(ch.points_in_empty_cells) of them lie in cells no training state visits.
]
#step([Three checks decide degree or rank], code: chebtt-code, focus: (6, 7, 9, 10), width: 40%, panel("error-chebtt_roll"))[
  The train is accepted only when it is within $#st.chebtt.safety atol$ at three check sets: Sobol points, free points
  of the grid evaluated earlier, and look-ahead points where the next round would sample. When it fails, an axis whose
  top coefficients are not yet small gets double the degree, and otherwise the rank grows. The map shows the result:
  the worst one-step error on a $#ex.grid times #ex.grid$ grid over the box is #sig(ch.grid_worst_accepted /
  ex.tol) of the tolerance, pale everywhere.
]

#implementation(title: [Several outputs, one cache])[
  A step map has $d$ outputs. ChebTT trains one train per component, but all share one cache of the oracle, so a
  state the solver stepped for one component serves every other: the #num(ch.solver_calls) solves of the example are
  the union. The cost per output is the price: at $d = 20$ a separate train per component is the main reason ChebTT
  times out (@sec:compare).
]

== Piecewise axes, rotated coordinates and an interface term

Three extensions keep ChebTT strongest on the function classes where a single smooth train stumbles. Each is tried
only when the plain train fails, so it costs nothing where it is not needed.

- *Piecewise axes* (in ChebTT itself). Before raising the rank, the build scans fibres for an outlier second
  difference, bisects to a kink or a jump, and confirms it on far fibres. A confirmed edge aligned with an axis becomes
  a breakpoint, and each piece of that axis gets its own Chebyshev basis @trefethen2019; an edge found elsewhere on a
  far fibre is oblique and is left to ChebTT-A.
- *ChebTT-M: the coordinates the outputs share.* A function of an oblique coordinate $q dot x$ has a high rank on the
  axes and rank 1 in coordinates with $q$ as an axis.

#proposition(title: [a ridge in its own coordinates @zahm2020])[
  If $f(x) = g(q dot x)$ with $|q| = 1$, its _pullback metric_ $G = EE[nabla f nabla f^top] = EE[g'(q dot x)^2] q q^top$
  has rank 1 with eigenvector $q$, and in the coordinates $u = Q^top x$ of any orthogonal $Q$ with first column $q$,
  $f$ depends on $u_1$ alone: its tensor train has rank 1.
] <prop:ridge>

  ChebTT-M first trains every output on the axes with its rank capped at #st.chebtt.axes_rank. Only the outputs that
  reach the cap get a metric, estimated by finite differences at pilot points, and an output is turned when 99% of
  its metric's trace lies in at most half as many eigenvectors as it lies in axes. Outputs whose leading subspaces
  merge are trained together in their summed metric's eigenvectors. A separable output is never turned, since a
  rotation could only raise its rank.
- *ChebTT-A: a term for the jump.* A jump across an oblique plane or a quadric has singular values that barely fall,
  and a tree of boxes needs $(1 slash h)^(d - 1)$ leaves along it. ChebTT-A locates points on the interface (bisecting
  along segments from extreme check points), fits the simplest plane or diagonal quadric $phi$ through them, measures
  the one-sided jumps of $f$ and its derivatives along the normal, and builds a term $S$ with the same jump. Then
  $f - S$ is smooth across the interface and ChebTT handles it: $f approx "ChebTT"(f - S) + S$.

== The rollout margin for step maps

A step map's box is the padded bounding box of the training states, and the surrogate is a train per component of the
increment. On its own it meets the one-step tolerance; @prop:recurrence says a rollout needs more.

#algorithm(caption: [ChebTT with a rollout margin: `chebtt_roll` in `src/benchmark/step_maps.py`. The same loop wraps
  AFINNI and ART in the comparison (@sec:compare), so no method lacks the margin.], pseudocode(roll-alg)) <alg:roll>

On the example #if ch.one_step_tol < ex.tol [the first round's chains along the training trajectories exceed
$atol slash 2$, so it retrains at the one-step tolerance #sig(ch.one_step_tol), reusing every cached solve.] else [the
first round's chains along the training trajectories already hold $atol slash 2$.] The test rollouts' worst error is
#sci(ch.rollout_worst), #pct(ch.rollout_worst / ex.tol) of the tolerance (@fig:rollout).

== Where ChebTT stumbles

#pitfall(title: [The box is not the visited set])[
  On a step map ChebTT pays for the whole box. On the example #pct(ch.points_in_empty_cells) of its #num(ch.solver_calls) solves
  lie where no trajectory goes, and the effort grows with the box: in a stiff system the box reaches states far from the slow
  set, where the map is least smooth (@sec:approx). At $d >= 20$ the trains of every component, each on a box mostly
  empty, run out of time before they hold. Outside its box ChebTT can only refuse, so a transport step that leaves the
  box sends the cell to the solver.
]

The remedy is not a better box but a different domain: the states themselves.

#takeaway[ChebTT samples where Chebyshev interpolation wants to, along fibres, so its sampling factor stays
  logarithmic and its cost linear in the dimension; it holds on every class of the function suite and on step maps
  that fill their box.]

= The chart recipe: a model for the states a simulation visits <sec:recipe>

#question[The data are trajectories, and the box around them is mostly empty. Can a surrogate built from the
  trajectories alone know, at each input, whether it may answer?]

== The candidates, and where they stumble

- *ChebTT on the box* answers the question only at the box's boundary and pays for the empty corners
  (@sec:chebtt).
- *Local tables* (ISAT @pope1997, the library's SlowTable) refuse and grow in situ, but their ellipsoids of accuracy
  are estimates from a few points, their error control is a quantile, and their memory grows without bound. SlowTable
  holds #rs.steps.stiff.slowtable.held of the #rs.steps.stiff.slowtable.ran stiff step maps it ran at 0.01.
- *Neural networks* fit the trajectories well but give no computable bound at a point of use, so they cannot refuse
  by their own error; a learned error model inherits the same problem.

@prop:lebesgue suggests the answer: a fit _linear in its data_ has a computable Lebesgue function at every input.
Fit the step from the trajectories, in coordinates of the visited set, and refuse where the bound is too large.

== The recipe, step by step

#algorithm(caption: [Training the chart recipe: `build`, `select`, `final` and `calibrate_fresh` in
  `theory/panel/generic4.py`.], pseudocode(recipe-alg)) <alg:recipe>

#story()
#step([The data and the chart], code: recipe-code, focus: (1, 2), width: 40%, panel("samples-recipe_100"))[
  The recipe never chooses where to sample: its data are the #swatch(mcol("recipe_100")) training trajectories. Its
  coordinates are the visited states' principal directions after the conservation laws are removed. A law is used
  only if the system declares it or if it also holds at states drawn off the data: laws found on the data alone
  include symmetries of the initial law, which transport breaks. On larger systems the chart keeps the $k$ leading
  directions and a decoder for the rest; the running example keeps both coordinates.
]
#step([Selection by cost, held out by trajectory], code: recipe-code, focus: (3, 4, 5, 6), width: 40%, panel("selection"))[
  Each degree is fitted on a third of the trajectories, calibrated on a second and judged on the third's chains, for
  three splits. Among the degrees whose chains hold $atol slash 2$ while refusing less than half the inputs, the
  recipe takes the one with the smallest expected cost per step, gate plus prediction plus $rho$ solves, not the most
  accurate: the most accurate degree on CHEMAKZO was slower than the solver. The panel shows each degree's
  #swatch(mcol("recipe_100")) held-out chain error against the #swatch(red, shape: "dash") margin and its
  #swatch(dark) expected cost; on the example the recipe chooses degree #rc.degree.
]
#implementation(title: [A selection by measured cost depends on the machine's load])[
  The costs the selection compares are timed, and degrees whose costs lie within 10% of the cheapest are a tie, broken by
  the smaller chain error. On a loaded machine the timings move, and another degree can win#if rc.degree_selected_this_run != rc.degree [: rerunning the example while other studies ran chose degree #rc.degree_selected_this_run where the comparison's
  record has #rc.degree]. Every record therefore keeps the chosen configuration, and this companion rebuilds the
  recorded one.
]

#step([The fit and its bound], code: recipe-code, focus: (7, 8), width: 40%, panel("lebesgue-ls"))[
  The fit is the minimum-norm least-squares solution on the singular directions above a relative cut of
  #sci(st.rcond, digits: 1). The same factor $M$ gives the gate's bound: $b(t) = sqrt(N) norm(phi(t) M)$ is at least
  the fit's Lebesgue function at $t$ (@lem:bound), large in the box's empty corners and small along the trajectories.
  A second statistic, the share of $phi(t)$ outside the fit's span, catches the inputs where the data do not determine
  the fit at all and $b$ is blind.
]
#step([The envelope, calibrated], code: recipe-code, focus: (9, 10), width: 40%, panel("error-recipe_100"))[
  The gate refuses an input whose bound $b$ exceeds a cap of $#st.beta sqrt(r)$, and otherwise accepts it when an
  envelope $a_0 + a_1 b$ is within $atol slash 2$; the envelope is the tightest line, by a linear program, above the
  one-step errors of fresh calibration states, plain and mixed. The map shows the result: #pct(rc.grid_refused_share)
  of the box is #swatch(refused) refused, and the accepted part, around the spiral, errs by at most
  #sci(rc.grid_worst_accepted).
]

#lemma(title: [the gate's bound is computable and conservative])[
  For the truncated fit $hat(f)(t) = phi(t) M U_r^top f(X)$, with $A = U S V^top$ and $M = V_r S_r^(-1)$, the weights
  are $ell (t) = U_r M^top phi(t)^top$ and
  $ Lam (t) = norm(ell (t))_1 <= sqrt(N) norm(ell (t))_2 = sqrt(N) norm(phi(t) M) = b(t). $
  On average over the fitted states, $b^2$ equals the rank $r$, so at most $1 slash beta^2$ of them exceed the cap
  $beta sqrt(r)$.
] <lem:bound>

#proof[$U_r$ has orthonormal columns, so $norm(ell)_2 = norm(phi M)$; Cauchy–Schwarz gives $norm(ell)_1 <= sqrt(N)
  norm(ell)_2$. Over the fitted rows, $sum_i norm(phi(x_i) M)^2 = norm(A M)_F^2 = norm(U_r)_F^2 = r$, so the mean of
  $b^2 = N norm(phi M)^2$ is $r$, and Markov's inequality bounds the share above $beta sqrt(r)$.]

The sketch replaces the two products of the gate by Gaussian sketches with #st.sketch_columns columns @halko2011, which
preserve norms in expectation; the gate is calibrated on the sketched statistics, so the statistic deployed is the one
calibrated.

== The account: a budget per cell <sec:account>

A gate that judges each step alone cannot see a chain: #ex.steps accepted steps each within $atol slash 2$ can add up
to far more. The account carries, for each cell, the sum of a typical-defect charge $c = m_0 + m_1 b$ (fitted by
non-negative least squares to the calibration errors) over its accepted steps, and refuses a step that would take
$lambda (S + c)$ past $atol slash 2$.

#algorithm(caption: [Deploying the chart recipe on a batch of cells: `Surrogate4.evaluate` and `Account.refuses` in
  `theory/panel/generic4.py`.], pseudocode(deploy-alg)) <alg:deploy>

#proposition(title: [what the account guarantees])[
  For a non-expansive map ($L_j <= 1$ in @prop:recurrence), a chain whose accepted defects are at most their charges
  $lambda c_k$ errs by at most $lambda sum_k c_k <= atol slash 2$ at every step, whatever moved the state between steps.
  With decay $gamma < 1$ the account forgets old charges geometrically, which matches a contracting map's fading of
  old errors.
] <prop:account>

#proof[@eq:recurrence with $L_j <= 1$, $delta_k = 0$ at refused steps and $norm(delta_k) <= lambda c_k$ at accepted
  ones.]

The hypothesis "the defects are at most their charges" is distributional: the charge is a typical defect, not an
envelope, so the margin comes from $lambda$, calibrated on whole chains. The recipe picks the smallest $lambda$ among
#st.lambdas.map(str).join(", ") for which at most #pct(st.alpha) of fresh calibration chains (plain, and mixed at
$w = 0.01$ and $0.1$) exceed $atol slash 2$ anywhere, a chain-level form of conformal risk control @angelopoulos2024.

#let longrun = rs.recipe.problems.values().filter(p => p.long_steady != none)
// the account's own refusals: the never-forgetting account's steady rate beyond the gate's alone
#let by-account(p) = if p.account_steady == none or p.gate_steady == none { 0 } else { p.account_steady - p.gate_steady }
#let exhausted = longrun.filter(p => by-account(p) > 0.05).map(p => p.account_steady)
#let decayed = longrun.filter(p => by-account(p) > 0.05 and p.long_steady == 0).len()
#pitfall(title: [Three corrections the account needed])[
  - *An account that never forgets.* Summing the charges for ever exhausted the cells on 2,000-step chains: on
    #exhausted.len() of the #longrun.len() problems with long chains it alone refused #pctx(calc.min(..exhausted)) to
    #pctx(calc.max(..exhausted)) of the steady steps, although the errors stayed tiny. With the decay
    $gamma = #st.gamma$, #decayed of them refuse no steady step, and every chain stays within the budget.
  - *An envelope fitted on states the gate would refuse.* The first envelope gate fitted its line over every
    calibration state, including those far beyond the data, which pulled the slope up and refused good inputs. The
    cap $b <= #st.beta sqrt(r)$ comes first, and the envelope is fitted inside it.
  - *Charging the bound instead of the defect.* An account that charged the gate's bound, an envelope, at every step
    compounded its conservatism over a chain. Charging the typical defect and calibrating $lambda$ on whole chains
    puts the margin in one calibrated number.
]

== The data: refused states and the training horizon <sec:time>

Two findings decide what data the recipe should be given, and both are generic.

*Train where it refused.* A deployed surrogate's refused inputs are exactly where the data are missing, and the
solver stepped them anyway, so their labels are free. Adding the refused inputs of closed-column deployments and
refitting cut the refusal rate of fresh closed columns below the same number of uniformly drawn mixed inputs on every
one of the eight problems tried (@fig:active); on CHEMAKZO from
#pctx(rs.recipe.closed_active.chemakzo.start_refused) to #pctx(rs.recipe.closed_active.chemakzo.refused.audit_refused.last()).

*Train over the horizon of use.* On 2,000-step chains Robertson and CHEMAKZO refused nearly every steady step
(#pctx(rs.recipe.horizon.robertson.steady_short) and #pctx(rs.recipe.horizon.chemakzo.steady_short)): their late
states lie beyond the 40-step training trajectories. Adding 40 trajectories of 400 steps lowered the steady refusal
only to #pctx(rs.recipe.horizon.robertson.steady_400) and #pctx(rs.recipe.horizon.chemakzo.steady_400)\; 40
trajectories spanning the whole horizon lowered it to #pctx(rs.recipe.horizon.robertson.steady_full) on Robertson
(a steady speed-up of #sig(rs.recipe.horizon.robertson.speedup_full)) and #pctx(rs.recipe.horizon.chemakzo.steady_full)
on CHEMAKZO, every chain within $atol slash 2$. The training horizon must span the deployment's; CHEMAKZO needs more
long starts as well.

#figure(panel("recipe-active", width: 80%), caption: [The share of fresh closed-column steps refused, per problem:
  #swatch(grey, shape: "dot") the recipe at 340 trajectories, then after three rounds of #swatch(blue) uniformly drawn
  mixed inputs or #swatch(rgb("#d81b8a")) the inputs it refused, at equal numbers of labels.]) <fig:active>

== Where the recipe stumbles

#pitfall(title: [The recipe needs trajectories that cover the use])[
  Its gate makes it safe off the data, not accurate there: where the deployment wanders beyond the trajectories it
  refuses, and the speed-up falls to the solver's. Nelson's network and primordial chemistry stay bias-limited at 580
  trajectories; HIRES fails weak mixing; POLLU needs 340 trajectories to be served; and the box-filling lattices at
  $d >= 20$ and Lorenz-96 at $d = 20$ have no reduced chart at these horizons.
]

#takeaway[Fit from the trajectories in their own coordinates, refuse where the fit's own Lebesgue bound or its error
  account says so, and train on the inputs it refused: the recipe answers only where it can vouch for itself.]

= Results on every class <sec:results>

#question[Which scheme is strongest on each class of problem, and by how much?]

== Functions: ChebTT and its variants

#figure(panel("functions-held"), caption: [The share of function-suite cases each method holds, per class, at the
  tolerances 0.05 and 0.01 (three seeds each; a case is a function, a dimension and a tolerance).]) <fig:fheld>

#let mshown = ("chebtt", "chebtt_metric", "chebtt_adaptive", "sparsecheb", "chebtree", "hub", "afinni", "art", "art_quad")
#let ftable(tol) = table(
  columns: (auto,) + (1fr,) * 5,
  inset: (x: 4pt, y: 3pt),
  align: (left,) + (center,) * 5,
  table.header([], ..rs.problems.functions.keys().map(c => small[*#c*])),
  ..for m in mshown {
    (small(lbl.at(m)),) + rs.problems.functions.keys().map(c => {
      let v = rs.functions.classes.at(c + "/" + str(tol)).at(m, default: none)
      if v == none or v.ran == 0 { small[–] } else { small[#v.held / #v.ran #h(0.2em) (#v.won)] }
    })
  },
)
#figure(ftable(0.01), caption: [Tolerance 0.01: cases held of those run, and (in brackets) won with the fewest
  evaluations.]) <tab:f001>
#figure(ftable(0.05), caption: [Tolerance 0.05, as @tab:f001.]) <tab:f005>

The ChebTT family holds nearly every case it runs at both tolerances, on every class and at every dimension up to
20, and which variant wins follows the class: ChebTT-M on functions with many local features (shared oblique
coordinates), ChebTT-A on kinks, jumps and fronts (its interface term), plain ChebTT on vector-valued functions (one
cache for every output). SparseCheb and ChebTree win some smooth and flow-map cases in two and three dimensions with
fewer evaluations, and run nowhere above. The hub expansion holds many smooth cases, but rarely wins one.

== Step maps: the benchmark's rollouts

#let sshown = ("chebtt_box", "chebtt_adaptive", "chebtt_roll", "slowtable", "afinni", "art", "art_quad")
#figure(table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 5pt, y: 3pt),
  align: (left, center, center),
  table.header([], [stiff], [non-stiff]),
  ..for m in sshown {
    (lbl.at(m),) + ("stiff", "nonstiff").map(r => {
      let v = rs.steps.at(r).at(m, default: none)
      if v == none [–] else [#v.held / #v.ran #h(0.2em) (#v.won)]
    })
  },
), caption: [The benchmark's step maps at the tolerance 0.01: cases held of those run (the rollouts of the 40 test
  trajectories within the tolerance), and won. The benchmark stops a method at its first failure on a family, so
  _ran_ differs between methods.]) <tab:steps>

On the benchmark's step maps ChebTT holds the small lattices, Robertson, CHEMAKZO and Lorenz-96 at $d = 4$, and times
out on HIRES, POLLU, Nelson, primordial and the lattices from $d = 20$. The chart recipe serves several of those.

== Step maps: the chart recipe on the panel

#let rshown = ("bruss_d2", "fisher_kpp_d2", "fisher_kpp_d3", "lorenz96_d4", "robertson", "chemakzo", "allen_cahn_d5", "fisher_kpp_d5", "fisher_kpp_d10", "bruss_d20", "burgers_d20", "pollu", "hires")
#figure(table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  inset: (x: 4pt, y: 3pt),
  align: (left,) + (right,) * 7,
  table.header([], [$d$], [served], [rollout], [columns \ worst], [columns \ refused], [long chains \ steady refused], [steady \ speed-up]),
  ..for n in rshown {
    let p = rs.recipe.problems.at(n)
    (raw(n), str(p.d), if p.served == true [yes] else if p.served == false [no] else [–], sig(p.rollout_worst),
     sig(p.columns_worst), pctx(p.columns_refused), pctx(p.long_steady), sig(p.long_speedup))
  },
), caption: [The chart recipe at 340 training trajectories and the tolerance 0.01: whether it is served (within $atol
  slash 2$ in rollouts and mixing, refusing less than half), the worst rollout error, the worst error and the refused
  share over the six closed columns, and on 2,000-step chains the steady refusal rate and the speed-up over the fastest
  exact solver (@eq:speed).]) <tab:recipe>

#figure(panel("recipe-long", width: 80%), caption: [The recipe's steady speed-up on 2,000-step chains, against the
  fastest exact solver, at the steady refusal rate of @tab:recipe.]) <fig:long>

#let served-roll = rshown.map(n => rs.recipe.problems.at(n)).filter(p => p.served == true).map(p => p.rollout_worst)
Where it is served, the recipe's worst rollout error lies between #sig(calc.min(..served-roll)) and
#sig(calc.max(..served-roll)) and, on the long chains, it refuses no steady step on
#rshown.map(n => rs.recipe.problems.at(n)).filter(p => p.long_steady == 0).len() problems, for steady speed-ups of tens to hundreds over LSODA or VODE. Robertson and CHEMAKZO are the exceptions
@sec:time explained: their 40-step training does not reach their late states. The closed columns are evaluation-limited
in this protocol (100 cells per call), so their speed-ups are lower than the long chains'.

#takeaway[ChebTT and its variants are strongest wherever the oracle can be queried on a box; the chart recipe extends
  the served step maps to stiff systems whose boxes are mostly empty, at speed-ups of tens to hundreds.]

= ChebTT and the recipe against AFINNI and ART <sec:compare>

#question[On the same problems, with the same data rules and the same protocol, how do the adaptive local methods
  compare with the strongest schemes?]

== Functions

On the function suite every method queries the same oracle on the same box and is judged on the same test set. The
fairest comparison is on the cases that every compared method ran, since the benchmark stops a method at its first
failure on a family.

#let mtab(tol) = table(
  columns: (auto, auto) + (1fr,) * 6,
  inset: (x: 3pt, y: 3pt),
  align: (left, center) + (center,) * 6,
  table.header([], [cases], ..("chebtt", "chebtt_metric", "chebtt_adaptive", "afinni", "art", "art_quad").map(m => small(lbl.at(m)))),
  ..for c in rs.problems.functions.keys() {
    let v = rs.functions.matched.at(c + "/" + str(tol))
    (small[*#c*], small(str(v.cases))) + ("chebtt", "chebtt_metric", "chebtt_adaptive", "afinni", "art", "art_quad").map(m => {
      let x = v.methods.at(m)
      if v.cases == 0 { small[–] } else { small[#x.held held \ #if x.median_ratio == none [failed] else [ratio #sig(x.median_ratio)] \ #num(x.median_evals) ev.] }
    })
  },
)
#figure(mtab(0.01), caption: [The cases every compared method ran, at the tolerance 0.01: how many each held, its
  median worst ratio (worst error over the tolerance; above 1 fails) and its median evaluations.]) <tab:m001>
#figure(mtab(0.05), caption: [As @tab:m001, at the tolerance 0.05.]) <tab:m005>

#figure(panel("functions-matched"), caption: [Every matched case: evaluations against the worst ratio, at both
  tolerances. Above the #swatch(red, shape: "dash") line a case fails.]) <fig:matched>

On these cases the ChebTT variants hold every one, at a median worst ratio near a quarter of the tolerance (their
safety factor is #st.chebtt.safety), while AFINNI and both ARTs hold almost none: their median worst ratios are two to a
hundred times the tolerance, and they spend as many or more evaluations. The reason is the one @sec:chebtt gave: a
local affine or quadratic piece's error falls polynomially with its size, so a maximum-error tolerance over the whole
box needs pieces everywhere, while a global expansion's error falls geometrically with its degree.

== Step maps under one protocol

#let cshown = ("recipe_100", "recipe_340", "chebtt_roll", "afinni_roll", "art_roll", "art_quad_roll")
#let cproblems = ("bruss_d2", "robertson", "fisher_kpp_d3", "lorenz96_d4", "chemakzo", "allen_cahn_d5", "fisher_kpp_d5", "fisher_kpp_d10", "burgers_d20", "pollu")
// a case's share of rollout steps refused (steps 1-10 and 11-40 weighted by their counts)
#let crefused(r) = if r.rollout_fresh_rate == none { 0 } else { (r.rollout_fresh_rate + 3 * r.rollout_steady_rate) / 4 }
// served, as everywhere in this companion: within the budget on every rollout and refusing less than half of their steps
#let cserved(r, tol) = (r != none and r.status == "done" and r.rollout_worst != none and r.rollout_worst <= tol / 2
  and crefused(r) < 0.5)
#let crec(p, m, tol) = rs.compare.at(str(tol)).records.at(p + "/" + m, default: none)
#let ccell(r, tol) = {
  if r == none { return small[not run] }
  if r.status != "done" { return small[not within \ the budget] }
  let refused = crefused(r)
  let ok = cserved(r, tol)
  let body = small[#sig(r.rollout_worst)#if refused >= 0.5 [ (#pctx(refused) refused)] \ #pctx(r.long.steady_rate) \ #sig(r.long.speedup_steady)×#if r.over_budget [ \ \* #num(r.train_s) s]]
  let body = if ok { body } else { text(fill: red, body) }
  if r.over_budget { text(style: "italic", body) } else { body }
}
#let ctab(tol) = table(
  columns: (auto,) + (1fr,) * 6,
  inset: (x: 3pt, y: 3pt),
  align: (left,) + (center,) * 6,
  table.header([], ..cshown.map(m => small(lbl.at(m)))),
  ..for p in cproblems {
    (raw(p),) + cshown.map(m => ccell(rs.compare.at(str(tol)).records.at(p + "/" + m, default: none), tol))
  },
)
#figure(ctab(0.05), caption: [Every method under one protocol at the tolerance 0.05: the worst error over the 40 test
  rollouts, the steady refusal rate on 2,000-step chains, and the steady speed-up over the fastest exact solver. Red:
  not served, because the rollout exceeds $atol slash 2$ or the method refuses half of the rollouts' steps or more. Every case has the same compute budget, #(rs.compare.budget_s / 60) minutes on this
  laptop. A case that did not finish within it says so; a case that finished but took longer (it ran under the study's
  first limit, 40 minutes) keeps its values, in italics with an asterisk and its training time.]) <tab:c005>
#figure(ctab(0.01), caption: [As @tab:c005, at the tolerance 0.01.]) <tab:c001>

#figure(panel("compare-steps"), caption: [The comparison of @tab:c005 and @tab:c001 per problem: the rollout's worst
  error over the tolerance (above the dashed line it exceeds $atol slash 2$) and the steady speed-up on long chains; a
  cross marks a case that did not finish within the budget, a hollow marker one that finished beyond it, and a square one that refused half of the rollouts' steps or more.]) <fig:csteps>

#let holds(m) = m.rollout_worst <= ex.tol / 2
The running example shows the pattern in miniature (@fig:rollout). At $atol = #ex.tol$, AFINNI's rollouts
#if holds(af) [hold $atol slash 2$] else [exceed $atol slash 2$ (#sci(af.rollout_worst))] and both ARTs'
#if holds(ar) and holds(aq) [hold it] else [do not all hold it], but the box methods pay for it in solves: AFINNI
needed #num(af.solver_calls) and ART #num(ar.solver_calls), against ChebTT's #num(ch.solver_calls). AFINNI's trust
regions also refuse #pctx(af.record.long.steady_rate) of the steady steps, for a steady speed-up of
#sig(af.record.long.speedup_steady) against ChebTT's #sig(ch.record.long.speedup_steady). The box methods spend their solves throughout the box, the local ones most of all:

#board(
  (panel("samples-afinni_roll"), [#swatch(mcol("afinni_roll")) AFINNI: #num(af.solver_calls) solves]),
  (panel("samples-art_roll"), [#swatch(mcol("art_roll")) ART, affine leaves: #num(ar.solver_calls)]),
  (panel("samples-art_quad_roll"), [#swatch(mcol("art_quad_roll")) ART, quadratic leaves: #num(aq.solver_calls)]),
  (panel("error-afinni_roll"), [AFINNI's one-step error over the box; grey: outside every trust region]),
  (panel("error-art_roll"), [ART (affine)]),
  (panel("error-art_quad_roll"), [ART (quadratic)]),
  columns: 3,
)

#let median(xs) = { let s = xs.sorted(); if s.len() == 0 { none } else { s.at(calc.quo(s.len(), 2)) } }
#let tally(m, tol) = {
  // within the budget only: a case that finished beyond it (under the study's first limit) is counted as not finished
  let all = cproblems.map(p => crec(p, m, tol)).filter(r => r == none or not r.over_budget)
  let ok = cproblems.filter(p => cserved(crec(p, m, tol), tol) and not crec(p, m, tol).over_budget)
  (served: ok.len(), finished: all.filter(r => r != none and r.status == "done").len(),
   timed: all.filter(r => r != none and r.status != "done").len(),
   maxd: if ok.len() == 0 { none } else { calc.max(..ok.map(p => rs.problems.step_maps.at(p).d)) },
   calls: median(ok.map(p => crec(p, m, tol).solver_calls)),
   speed: median(ok.map(p => crec(p, m, tol).long.speedup_steady).filter(x => x != none)))
}
#figure(table(
  columns: (auto,) + (1fr,) * 6,
  inset: (x: 3pt, y: 3pt),
  align: (left,) + (center,) * 6,
  table.header([], table.cell(colspan: 3)[*tolerance 0.05*], table.cell(colspan: 3)[*tolerance 0.01*],
    [], small[served / finished], small[largest $d$ served], small[median solves; speed-up], small[served / finished],
    small[largest $d$ served], small[median solves; speed-up]),
  ..for m in cshown {
    (small(lbl.at(m)),) + (0.05, 0.01).map(tol => {
      let t = tally(m, tol)
      (small[#t.served / #t.finished], small[#if t.maxd == none [–] else [#t.maxd]],
       small[#if t.calls == none [–] else [#num(t.calls)\; #if t.speed == none [–] else [#sig(t.speed)×]]])
    }).flatten()
  },
), caption: [Each method over the #cproblems.len() step maps: the problems it serves (within $atol slash 2$ on every test
  rollout, refusing less than half of their steps) of those it finished within the budget (a case that finished beyond it counts as not finished), the largest dimension it
  serves, and over its served problems the median solver calls and median steady speed-up (the recipe's calls are
  its trajectories' steps and labels).]) <tab:tally>

#let both(tol) = cproblems.filter(p => cserved(crec(p, "chebtt_roll", tol), tol) and cserved(crec(p, "art_quad_roll", tol), tol))
#let ratios = (0.05, 0.01).map(tol => both(tol).map(p => crec(p, "art_quad_roll", tol).solver_calls / crec(p, "chebtt_roll", tol).solver_calls)).flatten()
#let t340 = (tally("recipe_340", 0.05), tally("recipe_340", 0.01))
#let tch = (tally("chebtt_roll", 0.05), tally("chebtt_roll", 0.01))
#let taf = (tally("afinni_roll", 0.05), tally("afinni_roll", 0.01))
#let taq = (tally("art_quad_roll", 0.05), tally("art_quad_roll", 0.01))
#let tar = (tally("art_roll", 0.05), tally("art_roll", 0.01))
#let t100 = (tally("recipe_100", 0.05), tally("recipe_100", 0.01))

@tab:tally condenses the comparison. Four findings stand out.

- *The local methods mostly stop at small dimension.* Within #(rs.compare.budget_s / 60) minutes AFINNI serves
  #taf.at(0).served and #taf.at(1).served of the #cproblems.len() problems at the two tolerances, ART with affine leaves
  #tar.at(0).served and #tar.at(1).served, ART with quadratic leaves #taq.at(0).served and #taq.at(1).served. Beyond
  $d = 4$ they mostly do not finish: each must tile the whole box with pieces accurate to the tolerance.
- *Where ChebTT and quadratic ART both serve a problem, ART pays more solves*: a median of #sig(median(ratios)) times
  ChebTT's over the #ratios.len() such cases.
- *ChebTT is the cheapest where it finishes, and stops at the box's size.* It serves #tch.at(0).served and
  #tch.at(1).served problems, with a median of #num(tch.at(0).calls) and #num(tch.at(1).calls) solves and median steady
  speed-ups of #sig(tch.at(0).speed)× and #sig(tch.at(1).speed)×. From $d = 10$ its trains, one per component on a mostly empty box, do not finish within the budget.
- *The chart recipe at 340 trajectories serves the most problems and the largest*: #t340.at(0).served and
  #t340.at(1).served, up to $d = #t340.at(1).maxd$, at errors often orders of magnitude below the tolerance. At 100
  trajectories it serves #t100.at(0).served and #t100.at(1).served: it fails where the data do not cover the use
  (allen_cahn_d5 at both tolerances), and on fisher_kpp_d10 at 0.01 its gate refuses every input.

Two problems limit every method. On Robertson and CHEMAKZO every method that finished refuses most steady steps of the
2,000-step chains, so the long-run speed-up is about 1: the late states lie beyond 40-step training, whatever the method
(@sec:time shows the remedy for the recipe). POLLU finished for no method within the budget; the recipe serves it in its
own studies (@tab:recipe), with a training longer than #(rs.compare.budget_s / 60) minutes.

#takeaway[Under one protocol AFINNI and ART serve fewer problems, at more solves, and stop at small dimension; ChebTT
  is the cheapest where it finishes; the chart recipe serves the most problems and the largest, and every method is
  limited where the training does not cover the deployment's horizon.]

= What remains open <sec:open>

- *Coverage in time and in mixing.* The recipe refuses where its trajectories did not go; the levers that worked
  (refused-state data, a training horizon that spans the deployment's) must become part of training, not studies.
- *Bias-limited networks.* Nelson's network and primordial chemistry are not served by a global fit in any chart
  tried; local fits in a chart, or a predictor of the environment's history, are candidates.
- *Large networks.* KIDA's 578 species are beyond every method here; the exact solver's own cost on KIDA comes first.
- *Devices.* No GPU was used for this companion. The recipe's evaluation is a few dense products per cell, so it suits
  batched GPU inference, but @cor:speed says the refusal rate, not the evaluation, decides the gain; measurements on a
  GPU against a batched stiff solver are the next step.
- *Certificates.* Every guarantee here is distributional (calibrated on fresh chains); a verified bound for a stated
  class of stiff step maps would need the defect bounded on a whole tube.

#show: appendix

= Settings <sec:settings>

#table(
  columns: (auto, 1fr),
  inset: (x: 5pt, y: 3pt),
  table.header[setting][value],
  [ChebTT safety factor; first rank; rank step; rank cap], [#st.chebtt.safety\; #st.chebtt.rank0\; #st.chebtt.rank_step\; #st.chebtt.rank_max],
  [ChebTT sweeps per round; fibres per axis in the pre-pass; largest degree], [#st.chebtt.sweeps\; #st.chebtt.n_fibre\; #st.chebtt.n_max],
  [ChebTT pieces per axis; ChebTT-M's rank cap on the axes], [#st.chebtt.max_pieces\; #st.chebtt.axes_rank],
  [recipe: singular-value cut; cap $beta$; sketch columns], [#sci(st.rcond, digits: 1)\; #st.beta\; #st.sketch_columns],
  [recipe: selection margin; served share], [$#st.margin atol$\; refusing less than #pct(st.served_share)],
  [recipe: account multipliers; exceedance share; decay], [#st.lambdas.map(str).join(", ")\; #pct(st.alpha)\; #st.gamma],
)

= From the companion's symbols to the code's names <sec:names>

#table(
  columns: (auto, 1fr),
  inset: (x: 5pt, y: 3pt),
  table.header[symbol][code],
  [$Lam$, $b$, $s$], [`Fit4.gate_terms` (bound, out-of-span share)],
  [$M = V_r S_r^(-1)$, the sketch], [`Fit4.M`, `Fit4.sketch`],
  [$a_0, a_1$; the cap $beta sqrt(r)$], [`Surrogate4.env`, `envelope`; `Surrogate4.b_cap`, `BETA`],
  [the account $S$, its charge $c$, $lambda$, $gamma$], [`Account.S`, `Surrogate4.charge`, `Surrogate4.lam`, `Account.decay`],
  [the rollout margin's one-step tolerance $a$], [`chebtt_roll`'s `one_step_tol`],
  [the box], [`state_box(visited(train))`],
)

= Reproducing the companion <sec:reproduce>

`make report-global-figures` runs `figures/make_figures.py` (the running example, a few minutes on one core) and
`figures/results.py` (the records), and `make report` compiles the PDF. The comparison's records are kept per case by
`theory/panel/p25_compare.py` under `theory/panel/data/records/p25/` (`THEORY_TOL=0.05` and `0.01`), the recipe's
studies under the other folders of `theory/panel/data/records/`, and the function suite and the benchmark's step maps
in the benchmark's store (`results/benchmark/runs/`).

#bibliography("references.yml", title: [References])
