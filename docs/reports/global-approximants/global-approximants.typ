#import "../companion/lib.typ": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#show: report.with(
  title: [Global approximants: one model for the whole box],
  subtitle: [A technical companion to the schemes: ChebTT, ChebTT-M, ChebTT-A and the hub expansion, and why
    they hold a max-norm tolerance where local pieces do not],
  authors: ((name: [PADAS group], affiliation: [The University of Texas at Austin]),),
  date: [November 2026],
  abstract: [
    A surrogate replaces an expensive function by a cheap one that agrees with it to a stated tolerance.
    The library's adaptive surrogate does this with many local affine pieces, and its companion explains
    how @companion. This companion explains the other family: _global approximants_, one model for the
    whole box, built from as few evaluations as the model's own error estimate asks for. All four start
    from Chebyshev interpolation. ChebTT stores it as a tensor train, sampled by cross approximation
    along a few lines of a Chebyshev grid, so its cost grows linearly with the dimension. ChebTT-M turns
    the coordinates so that features shared by several outputs line up with an axis. ChebTT-A adds one
    term aligned with a jump that no tensor train can follow. The hub expansion puts a dense model on
    the few directions a function is steep in and cheap low-order terms on all the others. Two running
    examples recur throughout: Rosenbrock's valley, as in the other companion, and a function built to
    be hard in three different ways. Every idea is shown as a picture, explained in words and stated
    formally; the pictures are computed from real runs, and the claims about the implementation are the
    ones its tests and the figure script check.
  ],
)

// Number only the equations the text refers to: those with a label.
#set math.equation(numbering: "(1)", supplement: [Eq.])
#show math.equation.where(block: true): it => {
  if it.has("label") or it.numbering == none { it } else {
    counter(math.equation).update(n => n - 1)
    math.equation(block: true, numbering: none, it.body)
  }
}

// ------------------------------------------------------------------- shared vocabulary
// Numbers and colours of the running examples, written by figures/make_figures.py with its panels, so that
// the text, the panels and the diagrams cannot disagree.
#let ex = json("figures/example.json")
#let col = ex.colours
#let dark = rgb(col.dark)
#let red = rgb(col.red)
#let blue = rgb(col.blue)
#let grey = rgb(col.grey)
#let green = rgb(col.green)
#let kink = rgb(col.feature.kink)
#let ridge = rgb(col.feature.ridge)
#let jumpc = rgb(col.feature.jump)
#let fib1 = rgb(col.fibre.at("0"))
#let fib2 = rgb(col.fibre.at("1"))
#let mcol(m) = rgb(col.method.at(m))
#let panel(name, width: 100%) = image("figures/" + name + ".svg", width: width)
// A panel of the AFINNI and ART companion, from its own recorded runs.
#let other(name, width: 100%) = image("../adaptive-surrogates/figures/" + name + ".svg", width: width)
#let error-key(width: 60%) = panel("error-key", width: width)
#show figure.where(kind: image): set block(breakable: false)
#let small(body) = text(8pt, body)
// Frames in rows, each (picture, label): a filmstrip that wraps.
#let board(columns: 2, ..frames) = block(breakable: true, above: 1.1em, below: 1.1em, grid(
  columns: (1fr,) * columns,
  column-gutter: 1.2em,
  row-gutter: 0.9em,
  ..frames.pos().map(((picture, label)) => block(breakable: false, align(center)[
    #set par(justify: false)
    #picture #v(-0.3em) #small(label)
  ])),
))
// Integers with thousands separators; a float in scientific notation.
#let num(x) = {
  if type(x) != int or calc.abs(x) < 1000 { return str(x) }
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
  let e = int(calc.floor(calc.log(calc.abs(x), base: 10)))
  let m = calc.round(x / calc.pow(10.0, e), digits: digits - 1)
  if calc.abs(m) >= 10 { // 9.96 rounds up to 10: one more power of ten
    m = m / 10
    e += 1
  }
  $#m dot 10^(#e)$
}
#let pct(x) = [#calc.round(100 * x, digits: 0)%]

// Notation: the AFINNI manuscript's where it has a symbol, and the companion's for the box.
#let fh = $hat(f)$
#let atol = $alpha_"tol"$
#let F = $cal(F)$
#let mdl = ex.chebtt.models
#let rnds = ex.chebtt.rounds

// ---------------------------------------------------------------------- the algorithms
#let build = (
  require[the oracle $f$ on the box, a tolerance $atol$, the safety factor $s$],
  ensure[a ChebTT within $s atol$ at every check point, compressed],
  ln(0, [evaluate $f$ at $64 + 32 d$ Sobol points $X_c$], note: [the first check set]),
  ln(0, [$n_k <-$ the smallest degree at which random fibres along axis $k$ have converged], note: [the degree pre-pass]),
  ln(0, [right index sets $<-$ the node nearest the largest $|f|$, the centre node], note: [rank $r_0$]),
  ln(0, [*repeat*]),
  ln(1, [cores $<-$ maxvol sweeps over the fibres of the grid], note: [@alg:sweeps]),
  ln(1, [errors $<-$ the model against $f$ at three check sets], note: [Sobol, free, look-ahead]),
  ln(1, [*if* every error is within $s atol$: *return* the model, compressed], note: [@alg:compress]),
  ln(1, [*if* edges are found on an axis that decays like a power: add breakpoints], note: [@alg:edges]),
  ln(1, [*else if* some axis's top degrees still change the model by more than its share: double their degrees]),
  ln(1, [*else*: the rank grows by $Delta r$, with random new right indices]),
)
#let build-code = (label: <alg:build>, lines: build)

#let sweeps = (
  require[the Chebyshev grid, right index sets $J_1, ..., J_(d-1)$],
  ensure[cores $G_1, ..., G_d$ of node values, left index sets $I_1, ..., I_(d-1)$],
  ln(0, [*for* $k = 1, ..., d$], note: [left to right]),
  ln(1, [evaluate $f$ on the fibres $I_(k-1) times {"every node of axis" k} times J_k$], note: [cached]),
  ln(1, [$Q R <-$ the fibre matrix; #h(0.3em) rows $<-$ maxvol$(Q)$]),
  ln(1, [$G_k <- Q Q_"rows"^(-1)$; #h(0.3em) $I_k <-$ the chosen rows]),
  ln(0, [*for* $k = d, ..., 2$: the same on the transposed fibres, choosing $J_(k-1)$], note: [right to left]),
  ln(0, [one more sweep left to right; its cores are the model]),
)
#let sweeps-code = (label: <alg:sweeps>, lines: sweeps)

#let edges = (
  require[an axis $k$, a base point, the axis's share $tau$ of the tolerance],
  ensure[breakpoints on axis $k$, each with a band],
  ln(0, [scan $f$ along the fibre through the base at #ex.chebtt.edges.scan equispaced points]),
  ln(0, [*for* each cell whose second difference is an outlier (at most 3)]),
  ln(1, [bisect on the value to $10^(-12)$ of the width; a jump if it passes the two-scale test], note: [`locate_edge`, `edge_at`]),
  ln(1, [*else* zoom into the largest second difference, to $0.01 tau slash |[f']|$; a kink if it passes]),
  ln(1, [*for* $2 d$ random far fibres: is the edge there, absent, or elsewhere?], note: [confirm]),
  ln(1, [elsewhere on some fibre: reject it, the edge is oblique]),
  ln(1, [otherwise keep it, with the largest displacement seen as its band]),
)
#let edges-code = (label: <alg:edges>, lines: edges)

#let compression = (
  require[the accepted train, probe points $Z$, a budget $beta$],
  ensure[a smaller train within $beta$ of the accepted one at every point of $Z$],
  ln(0, [bisect on $epsilon$: round the ranks by TT-SVD at $epsilon$; keep the largest $epsilon$ within $beta$], note: [`round_tt`]),
  ln(0, [*repeat* while some axis shrinks]),
  ln(1, [*for* each axis $k$: cut its top degrees while the model stays within $beta$ at $Z$]),
)
#let compress-code = (label: <alg:compress>, lines: compression)

= One model for the whole box <sec:intro>

#question[A function costs a second per evaluation, and a simulation will ask for it a billion times.
  How few evaluations buy a stand-in that is within a tolerance _everywhere_ in its box?]

An _oracle_ $f$ maps a point of a box $Omega = [ell_1, u_1] times dots times [ell_d, u_d]$ to one value
or to $m$ values. Evaluating it is the expensive part: an ODE solve, a chemistry step, a simulation.
A _surrogate_ $fh$ is built from a limited number of those evaluations and then answers in microseconds.
This companion asks for the strictest kind of agreement. The _worst_ error over the box must be within
a tolerance, $max_(x in Omega) |f(x) - fh(x)| <= atol$. A small average error is not enough. A
simulation that calls the surrogate a billion times will find the worst points.

Three costs decide between surrogates: the evaluations of $f$ that training spends, the time of one
answer and whether the worst error holds at all. The time of one answer matters because a surrogate is
trained once and asked millions of times. The library's benchmark judges every method by all three, on a fixed test set of each
problem.

== Two families

#board(
  (other("afinni-trust-all"), [*AFINNI* fits affine pieces to clusters of a given sample.]),
  (other("art-partition"), [*ART* cuts the box into polytopes, one affine piece each.]),
  (panel("rosenbrock-points"), [*ChebTT* samples a few lines of a grid and fits one model.]),
  columns: 3,
)

The library builds surrogates in two ways. The _adaptive surrogate_ divides the box into many small
regions and fits a local affine model to each. AFINNI groups a sample into clusters, and ART cuts the box
by planes. Where a piece misses the tolerance, it is cut again. Its companion follows both on
Rosenbrock's valley @companion.

The _schemes_ are the other family. They fit one model to the whole box, a _global approximant_. They
sample $f$ where their own error estimate asks, and they stop when the estimate and a set of check
points agree that the tolerance holds. They are NumPy code on the CPU, independent of the refinement
engine, and they spread only their oracle calls over MPI ranks.

#block(breakable: false, above: 1em, below: 1em, table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[][local pieces (AFINNI, ART)][one global model (the schemes)],
  [a model is], [affine, valid on its own region], [a polynomial of high degree, valid on the whole box],
  [its error], [falls with the square of a piece's width], [falls geometrically with the degree, when $f$ is smooth],
  [what is fitted], [least squares on scattered points], [interpolation at Chebyshev nodes],
  [what is refined], [the partition], [degrees, ranks, coordinates],
  [where it is weak], [the error between the training points], [features that are not smooth, or not aligned with the axes],
))

The table's second row is the whole argument in brief. A piece of width $h$ has an error of order
$h^2$. To halve the error, every piece must shrink by $sqrt(2)$, in every direction at once. In $d$
dimensions that multiplies the pieces by $2^(d slash 2)$. A Chebyshev interpolant of a smooth function
gains a fixed factor with every degree it adds (@sec:chebyshev). The rest of this companion is about
keeping that advantage when $f$ is not smooth, when its features are oblique, and when $d$ is large.

== The running examples <sec:examples>

#scene(side: left, width: 32%, panel("rosenbrock"))[
  The first example is the one the other companion follows: Rosenbrock's valley
  $f(x) = (1 - x_1)^2 + (x_2 - x_1^2)^2$ on
  $[#ex.rosenbrock.lower.at(0), #ex.rosenbrock.upper.at(0)] times [#ex.rosenbrock.lower.at(1), #ex.rosenbrock.upper.at(1)]$,
  with an absolute tolerance $atol = #ex.rosenbrock.alpha$. It is a polynomial of degree 4 in $x_1$ and
  2 in $x_2$, so a Chebyshev model can represent it exactly. It shows what a global model gains when
  its basis fits the function. AFINNI needed #num(ex.rosenbrock.afinni.evals) evaluations and ART
  #num(ex.rosenbrock.art.evals) for #ex.rosenbrock.afinni.leaves and #ex.rosenbrock.art.leaves affine
  pieces. Their worst errors over the box were #ex.rosenbrock.afinni.max_error and
  #ex.rosenbrock.art.max_error times the tolerance.
]

The second example, the _feature example_ $F$, is built to be hard in three different ways, and to be
hard for each scheme in a different way. Each part is a structure that real functions have, in its
simplest form, so that one picture can show what it does to a model. It lives on $[-1, 1]^2$ with the
tolerance $atol = #ex.feature.alpha$, and has three parts:
$ F(x) = underbrace(b(x), "base") + underbrace(r(x), "ridge") + underbrace(j(x), "jump"). $

#board(
  (panel("feature-b"), [$b$: a product, with a #swatch(kink, shape: "dash") kink]),
  (panel("feature-br"), [$b + r$: a #swatch(ridge, shape: "dash") ridge at #ex.feature.ridge.angle° added]),
  (panel("feature-brj"), [$F$: a #swatch(jumpc, shape: "line") jump at #ex.feature.jump.angle° added]),
  columns: 3,
)

- *The base* $b(x) = (0.3 |x_1 - #ex.feature.kink| + 0.5 cos 1.5 x_1)(1 + 0.3 sin 2 x_2)$ is a product
  of a function of $x_1$ and a function of $x_2$. That is the easiest structure for a tensor train. Its
  $x_1$ factor has a kink at $x_1 = #ex.feature.kink$, an edge that is aligned with an axis and slows
  every Chebyshev series that crosses it.
- *The ridge* $r(x) = #ex.feature.ridge.height tanh(#ex.feature.ridge.steep (q dot x + #ex.feature.ridge.offset))$
  is a smooth but steep step across a line, with $q$ at #ex.feature.ridge.angle° to the $x_1$ axis. It is
  smooth, but it is _oblique_. It depends on $x_1$ and $x_2$ only together. That is the structure that
  costs a tensor train rank (@sec:rank-pitfall).
- *The jump* $j(x) = #ex.feature.jump.height dot [p dot x > #ex.feature.jump.offset]$ adds
  #ex.feature.jump.height across a line with normal $p$ at #ex.feature.jump.angle°, cutting off the
  lower-right corner. It is both oblique and discontinuous, and no finite sum of products of
  univariate functions follows it.

Each scheme is introduced with the part of $F$ it is for. ChebTT handles the base, and finds its kink
by itself (@sec:edges). The ridge costs ChebTT rank; ChebTT-M and the hub expansion turn the
coordinates so that the ridge depends on one of them. The jump needs ChebTT-A's interface term.
@sec:compare runs every scheme on all of $F$, so one function gets four answers. For the hub expansion and
the dimension questions, the same ridge is lifted to ten dimensions.

== How to read this companion <sec:reading>

Each idea appears three times: as a *picture* of a running example, in plain *words*, and in
*symbols*, as a definition, lemma or theorem in a tinted box. The pictures are computed, not drawn.
`figures/make_figures.py` trains the schemes through `src.schemes` and records their stages. It
rebuilds what each picture shows from its definition, and stops unless that rebuild matches the
library. It also writes the numbers this text quotes.

Colours keep one meaning throughout. The feature example's parts are #swatch(kink) gold for the
kink, #swatch(ridge) cyan for the ridge and #swatch(jumpc) pink for the jump, in every panel and every
curve. Fibres of a tensor-train cross are #swatch(fib1) blue along $x_1$ and #swatch(fib2) orange along
$x_2$. In a map of a model's errors, a point is #swatch(rgb(col.error.at(1))) pale where the model is
within the tolerance, and #swatch(red) red beyond it, as in the other companion:

#align(center, error-key())

#translate(
  [the oracle, the surrogate, the input dimension, the number of outputs], [$f$, $fh$, $d$, $m$],
  [the box, its lower and upper corners], [$Omega$, $ell$, $u$ #h(0.3em) (`lo`, `hi`)],
  [the absolute tolerance on the worst error], [$atol$ #h(0.3em) (`atol`)],
  [the safety factor: a check passes within $s atol$], [$s = #ex.chebtt.defaults.safety$ #h(0.3em) (`safety`)],
  [the Chebyshev polynomial of degree $j$, and the coefficient of $f$ on it], [$T_j$, $a_j$],
  [the degree of axis $k$ and the number of its nodes], [$n_k$, $n_k + 1$],
  [the ranks of a tensor train, between axes $k$ and $k + 1$], [$r_k$, #h(0.3em) $r_0 = r_d = 1$],
  [a core of a tensor train: Chebyshev coefficients along axis $k$], [$C_k [alpha, j, beta]$, shape $r_(k-1) times (n_k + 1) times r_k$],
  [the share of the tolerance each axis's degree must meet], [$tau = s atol slash d$],
)

#takeaway[A global approximant is one model for the whole box, judged by its worst error, the
  evaluations it cost and the time of one answer.]

= Chebyshev interpolation, the shared tool <sec:chebyshev>

#question[A function of one variable on $[-1, 1]$, known at $n + 1$ points of our choice. Where should
  the points be, and how do we know when there are enough?]

== Where to put the points

The obvious choice is equally spaced points, and the polynomial through them. On the ridge's profile,
the feature example's $r$ along the line through the centre in the direction $q$, it fails in a
well-known way. With #(ex.chebyshev.runge.degree + 1) equispaced points the interpolant
#swatch(red, shape: "line") swings far outside the function near both ends. Its worst error is
#ex.chebyshev.runge.equispaced. Adding points makes it worse, not better, until rounding takes over.
At degree 64 the worst error is still #ex.chebyshev.runge.equispaced_64.

#figure(panel("runge"), caption: [The ridge's profile #swatch(ridge, shape: "line"), interpolated at
  #(ex.chebyshev.runge.degree + 1) points: equispaced #swatch(red, shape: "line"), and Chebyshev
  #swatch(dark, shape: "dash"). *Right.* The worst error against the degree, with the tolerance
  dotted. Chebyshev interpolation meets it at degree #ex.chebyshev.runge.chebyshev_within.]) <fig:runge>

#scene(side: left, width: 34%, panel("cheb-points"))[
  The answer is to cluster the points towards the ends. Take $n + 1$ equally spaced angles on a
  semicircle and drop them onto the diameter. The shadows are the _Chebyshev points_
  of the second kind, $x_j = cos(pi j slash n)$, $j = 0, ..., n$. The interpolant through them
  #swatch(dark, shape: "dash") is indistinguishable from the ridge at the same degree. Its worst
  error is #ex.chebyshev.runge.chebyshev.

  The same semicircle gives the basis. The _Chebyshev polynomial_ $T_j$ is defined by
  $T_j (cos theta) = cos j theta$, so it is a cosine seen through $x = cos theta$. So $|T_j (x)| <= 1$ on
  $[-1, 1]$, and the coefficients of the interpolant are a discrete cosine transform of its values.
]

#definition(title: [Chebyshev interpolant and coefficients])[
  The degree-$n$ _Chebyshev interpolant_ of $f$ on $[-1, 1]$ is the polynomial $p_n$ with
  $p_n (x_j) = f(x_j)$ at the $n + 1$ points $x_j = cos(pi j slash n)$. Written in the Chebyshev
  basis, $p_n (x) = sum_(j=0)^n c_j T_j (x)$, its coefficients $c_j$ come from one DCT-I of the values.
  The _Chebyshev coefficients_ of $f$ itself are the $a_j$ of $f = sum_(j >= 0) a_j T_j$; a box is mapped
  onto $[-1, 1]$ axis by axis.
] <def:interpolant>

#theorem(title: [interpolation at equispaced points is ill-conditioned, @platte2011])[
  Any method that approximates analytic functions from $n$ equispaced samples, and converges
  geometrically, has a condition number that grows exponentially in $n$.
] <thm:equispaced>

That is why every scheme in this companion samples at Chebyshev points, on an axis, a fibre, or a
rotated coordinate. The points cost nothing extra to choose, and the interpolant they give is within
a factor $2 + (2 slash pi) log(n + 1)$ of the best polynomial of its degree @trefethen2019.

== How fast the coefficients fall

Besides representing $p_n$, the coefficients carry information about $f$. The faster they fall, the
smoother $f$ is, and the size of the last few says how much the omitted ones can still matter. Three
slices of the feature example show the three cases every scheme meets: one through the ridge only, one
through the kink, one through the jump.

#figure(panel("decay"), caption: [*Top.* Three slices of $F$: #swatch(ridge, shape: "line") smooth
  (along $x_2$ at $x_1 = #ex.chebyshev.slices.smooth.value$, across the ridge),
  #swatch(kink, shape: "line") through the kink (along $x_1$ at $x_2 = #ex.chebyshev.slices.kink.value$),
  and #swatch(jumpc, shape: "line") through the jump (along $x_2$ at $x_1 = #ex.chebyshev.slices.jump.value$).
  *Bottom.* Their Chebyshev coefficients $|a_j|$ on log-log axes (dots), with the larger of each
  neighbouring pair (line). Dashed: the reference laws $rho^(-j)$, $j^(-2)$ and $j^(-1)$.]) <fig:decay>

The smooth slice's coefficients fall geometrically, by a factor $rho approx #ex.chebyshev.decay.smooth.rho$
per degree, until they reach rounding level at degree #ex.chebyshev.decay.smooth.floor_degree. On
log-log axes a geometric law falls ever more steeply. Through the kink they fall like a power, with a fitted exponent
of #ex.chebyshev.decay.kink.power, close to the 2 of the theory below. Through the jump the exponent
is #ex.chebyshev.decay.jump.power, close to 1. A jump's coefficients barely fall.

#theorem(title: [smoothness and decay, @trefethen2019[Thms. 7.1, 8.1]])[
  Let $f$ be analytic and bounded by $M$ inside the Bernstein ellipse with foci $plus.minus 1$ and
  semi-axis sum $rho > 1$. Then $|a_j| <= 2 M rho^(-j)$ and
  $ ||f - p_n||_infinity <= (4 M rho^(-n)) / (rho - 1). $
  Let instead $f, ..., f^((nu - 1))$ be absolutely continuous and $f^((nu))$ of bounded variation $V$.
  Then $|a_j| <= 2 V slash (pi j (j - 1) dots (j - nu))$ for $j > nu$, so $|a_j| = O(j^(-nu - 1))$.
] <thm:decay>

A kink is the case $nu = 1$, where $f$ is continuous and $f'$ jumps, so $|a_j| = O(j^(-2))$. A jump is
the case $nu = 0$, where $f$ itself has bounded variation and $|a_j| = O(j^(-1))$. The ridge is analytic, but its
steepness puts complex singularities of $tanh$ close to the real line, which makes $rho$ small. A
steeper ridge needs a higher degree, but its coefficients still fall geometrically.

== When to stop: the tail estimate <sec:tail>

A scheme never sees the error of its interpolant. It sees the coefficients it computed, and must
infer from them what the omitted ones can add.

#lemma(title: [aliasing, @trefethen2019[Thm. 4.2]])[
  $ ||f - p_n||_infinity <= 2 sum_(j > n) |a_j| . $
] <lem:alias>

#proof[
  Sampling at the $n + 1$ points folds each $T_j$ with $j > n$ onto a polynomial of degree at most $n$
  ("aliasing"), and every $T_j$ and its alias are bounded by 1. So the error is at most the omitted tail
  plus its folded copy.
]

The omitted tail is unknown, so the schemes estimate it from the top of what they have. A natural
estimate is the sum of the coefficients in the upper half of the degrees. It is far too cautious:

#proposition(title: [the top quarter, not the upper half])[
  Let $|a_j| = C rho^(-j)$. The error of $p_n$ is about $a_n$. The upper half of the degrees sums to
  about $a_(n slash 2) = sqrt(C a_n)$, and the top quarter to about $a_(3 n slash 4) = C^(1 slash 4) a_n^(3 slash 4)$
  (up to the factor $1 slash (1 - rho^(-1))$ of a geometric sum). So asking the upper half to be within
  a share $tau$ asks for $a_n approx tau^2 slash C$, about the _square_ of the accuracy, while the top
  quarter asks for $a_n approx tau^(4 slash 3) slash C^(1 slash 3)$.
] <prop:quarter>

#figure(panel("tail"), caption: [The tail estimate as the degree doubles, against the true worst error
  of the interpolant (coloured), with each axis's share $tau = #ex.chebyshev.tail.share$ of the tolerance
  dotted. Squares: twice the top quarter of the coefficients (`chebyshev_tail`). Triangles: twice the
  upper half. *Left*, the smooth slice; *right*, the kink.]) <fig:tail>

On the smooth slice the top-quarter estimate meets the share at degree
#ex.chebyshev.tail.smooth.quarter_degree, exactly where the true error does. The upper half meets it only at
degree #ex.chebyshev.tail.smooth.half_degree, with twice the points on that axis and no gain in accuracy. The figure
script computes the top quarter itself and stops unless it equals the library's `chebyshev_tail`. In the
experiments that chose it, at 5 and 10 dimensions, reading the top quarter instead of the upper half cut
ChebTT's evaluations on every case tried, with every case still within the tolerance.

#pitfall(title: [A tail is an estimate, and a kink fools it])[
  On the kink slice the top quarter says "converged" at degree #ex.chebyshev.tail.kink.quarter_degree,
  but the true error there is still above the share; it falls below it only at degree
  #ex.chebyshev.tail.kink.true_degree. A power law $j^(-2)$ has no steep fall for the top quarter to read, and
  the estimate assumes it has seen the worst of the tail. So no scheme here trusts the tail alone. ChebTT
  checks its model at points (@sec:checks), and searches an axis for kinks and jumps when its
  coefficients decay like a power (@sec:edges).
]

#implementation(title: [In the code])[
  `src/schemes/chebyshev.py` holds what the schemes share: `cheb_points`, `coefficients_along` (the DCT-I
  along one axis), `chebyshev_tail` (twice the top quarter) and `chebyshev_bases`, which evaluates
  $T_0, ..., T_n$ by the three-term recurrence $T_(j+1) = 2 x T_j - T_(j-1)$ rather than by $cos(j arccos x)$.
  The recurrence needs no transcendental function per query and axis, which matters at inference. ChebTree reads its tails from
  fitted decay laws (`axis_decay`, `tail_estimate`), which distinguish geometric from algebraic decay on
  the slab sums of a tensor box.
]

#takeaway[Chebyshev points make interpolation converge as fast as $f$ is smooth, and the size of the
  top quarter of the coefficients says when to stop, unless $f$ has a kink or a jump.]

= ChebTT: a tensor train sampled along fibres <sec:chebtt>

#question[A tensor grid of Chebyshev points has $(n + 1)^d$ nodes: $10^(10)$ at degree 9 in ten
  dimensions. How can a model use that grid without evaluating $f$ on it?]

== A function as a matrix, and its rank

Sample $f$ of two variables on a grid of Chebyshev nodes, and the values form a matrix $A_(i l) = f(x_(1,i), x_(2,l))$.
If $f = g(x_1) h(x_2)$, every row of $A$ is a multiple of every other, so $A$ has rank 1. A sum of $r$ such
products has rank $r$. Rosenbrock expands into exactly three products:
$(1 - x_1)^2 + x_1^4 + x_2^2 - 2 x_1^2 x_2$ is $g_1 (x_1) dot 1 + 1 dot x_2^2 + x_1^2 dot (-2 x_2)$,
with $g_1 = (1 - x_1)^2 + x_1^4$.

#figure(panel("matrix"), caption: [*Left.* Rosenbrock sampled on a $#ex.chebtt.matrix.grid times #ex.chebtt.matrix.grid$ grid of
  Chebyshev nodes, as a matrix. *Right.* The singular values of that matrix for
  #swatch(dark) Rosenbrock, #swatch(kink) $b$, #swatch(ridge) $b + r$ and #swatch(jumpc) $F$, relative to
  the largest. Rosenbrock has #ex.chebtt.matrix.rosenbrock.rank_exact nonzero singular values and $b$
  #ex.chebtt.matrix.b.rank_exact. Above $10^(-3)$, $b + r$ has #ex.chebtt.matrix.br.rank_1e3 and $F$
  #ex.chebtt.matrix.brj.rank_1e3.]) <fig:matrix>

The singular values say how many products a function needs. They fall to rounding level after
#ex.chebtt.matrix.rosenbrock.rank_exact for Rosenbrock and after #ex.chebtt.matrix.b.rank_exact for
the base. The ridge makes them fall slowly. The jump makes them fall hardly at all. The rank that a
tolerance of $10^(-3)$ needs is #ex.chebtt.matrix.brj.rank_1e3 for $F$, against
#ex.chebtt.matrix.br.rank_1e3 without the jump.

In $d$ dimensions the same idea becomes a chain of products, one matrix-valued factor per axis.

#definition(title: [tensor train, @oseledets2011])[
  A _tensor train_ (TT) of ranks $r_1, ..., r_(d-1)$ is
  $ fh(x) = G_1 (x_1) G_2 (x_2) dots G_d (x_d), $ <eq:tt>
  where $G_k (x_k)$ is an $r_(k-1) times r_k$ matrix ($r_0 = r_d = 1$) whose entries are polynomials in
  $x_k$. In ChebTT each entry is a Chebyshev series of degree $n_k$:
  $G_k (x_k)[alpha, beta] = sum_(j=0)^(n_k) C_k [alpha, j, beta] T_j (x_k)$. The array $C_k$ is the
  $k$-th _core_.
] <def:tt>

#proposition(title: [the ranks of simple structures])[
  A product $g_1 (x_1) dots g_d (x_d)$ has TT ranks 1, and a sum $g_1 (x_1) + dots + g_d (x_d)$ has
  ranks 2. For every direction $q$, $cos(q dot x)$ has ranks 2. A ridge $g(q dot x)$ with $q$ along an
  axis has ranks 1.
] <prop:ranks>

#proof[
  A product is @eq:tt with $1 times 1$ matrices. For a sum, take
  $G_1 = (g_1, 1)$, $G_k = mat(1, 0; g_k, 1)$, $G_d = (1, g_d)^top$. For the cosine,
  $cos(a + b) = cos a cos b - sin a sin b$ gives $2 times 2$ factors, one axis at a time.
]

== Sampling along fibres: the cross

A model of rank $r$ in two dimensions is determined by $r$ rows and $r$ columns of its matrix. That is
the idea that lets ChebTT avoid the grid.

#lemma(title: [skeleton decomposition])[
  Let $I$ be $r$ rows and $J$ be $r$ columns of $A$ with $A_(I J)$ invertible. Then
  $ tilde(A) = A_(: J) A_(I J)^(-1) A_(I :) $ <eq:skeleton>
  equals $A$ on the rows $I$ and on the columns $J$, and equals $A$ everywhere if $"rank" A = r$.
] <lem:skeleton>

#proof[
  On the columns $J$: $tilde(A)_(: J) = A_(: J) A_(I J)^(-1) A_(I J) = A_(: J)$, and the rows alike. If
  $"rank" A = r$, the columns $J$ span every column of $A$, and the coefficients that express a column
  in them are fixed by its $r$ entries in the rows $I$.
]

So the model needs $f$ only on a _cross_: $r$ lines along $x_1$ and $r$ lines along $x_2$. A line of
the grid along one axis, every node of that axis with the other coordinates fixed, is a _fibre_. Which
fibres should it use? The rows should be as far from linearly dependent as possible.

#theorem(title: [quasi-optimality of maximal volume, @goreinov2011])[
  If $A_(I J)$ has the largest $|det|$ among the $r times r$ submatrices of $A$, then
  $ max_(i, l) |A - tilde(A)|_(i l) <= (r + 1) sigma_(r+1)(A). $
] <thm:maxvol>

The bound is in the max norm, the norm this companion cares about. Finding the maximal volume is
hard, but `maxvol` @goreinov2010 finds a submatrix whose volume no single row swap can increase by
more than 5%, in a few iterations. In $d$ dimensions the cross becomes the TT-cross @oseledets2010. Between
axes $k$ and $k + 1$ it keeps $r_k$ left multi-indices $I_k$ (positions on axes $1, ..., k$) and
$r_k$ right ones $J_k$ (on axes $k + 1, ..., d$). It evaluates $f$ on the fibres
$I_(k-1) times {"every node of axis" k} times J_k$, and chooses the next left set by maxvol. Then it
sweeps back to choose the right sets.

#algorithm(caption: [The maxvol sweeps of the TT-cross: `_sweeps` in `src/schemes/chebtt/train.py`.],
  pseudocode(sweeps)) <alg:sweeps>

#algorithm(caption: [ChebTT: `chebtt.build` in `src/schemes/chebtt/train.py`. $Delta r$ is
  `rank_step` $= #ex.chebtt.defaults.rank_step$, $r_0$ is `rank0` $= #ex.chebtt.defaults.rank0$.],
  pseudocode(build)) <alg:build>

#story()
#step([Rosenbrock: one cross, exact], code: sweeps-code, focus: (1, 2, 3, 4), width: 36%, panel("cross-rosenbrock"))[
  On Rosenbrock the last sweep evaluates $f$ on #ex.chebtt.rosenbrock_cross.rank lines along $x_1$
  #swatch(fib1, shape: "line") and #ex.chebtt.rosenbrock_cross.rank along $x_2$
  #swatch(fib2, shape: "line"), each through #ex.chebtt.rosenbrock_cross.x1_nodes nodes. The dark
  points are $I times J$, the pivots maxvol chose. Since the rank of Rosenbrock is 3, the cross of rank
  #ex.chebtt.rosenbrock_cross.rank is exact by @lem:skeleton; compression later rounds the rank to
  #mdl.rosenbrock.ranks.at(0).
]

#step([$b + r$: the first round], code: build-code, focus: (1, 2, 3, 4, 5), window: 0, width: 36%, panel("cross-br-0"))[
  The feature example's $b + r$ starts at rank #rnds.at(0).rank, with degrees
  #rnds.at(0).degrees.at(0) and #rnds.at(0).degrees.at(1) from the pre-pass. Two lines each way see
  the base but not the ridge's full width. The worst check error is #rnds.at(0).worst times the
  tolerance. Grey dots are every point evaluated so far: the Sobol checks, the pre-pass fibres, the
  sweeps.
]

#step([The rank grows until the checks pass], code: build-code, focus: (6, 7, 10), width: 36%, panel("cross-br-6"))[
  Round by round the cross gains lines, and once an axis's degree doubles. After
  #(rnds.len() - 1) rounds it has #rnds.last().rank lines each way, the worst check error is
  #rnds.last().worst times the tolerance, and the model is accepted after #num(rnds.last().evals)
  evaluations. The figure script rebuilds the accepted model's node values from the recorded fibres as
  @eq:skeleton, and stops unless they match.
]

#filmstrip(
  (panel("cross-br-1"), [round 1: rank #rnds.at(1).rank, #rnds.at(1).worst]),
  (panel("cross-br-2"), [round 2: rank #rnds.at(2).rank, #rnds.at(2).worst]),
  (panel("cross-br-3"), [round 3: degree up, #rnds.at(3).worst]),
  (panel("cross-br-4"), [round 4: rank #rnds.at(4).rank, #rnds.at(4).worst]),
  (panel("cross-br-5"), [round 5: rank #rnds.at(5).rank, #rnds.at(5).worst]),
  caption: [The cross of ChebTT on $b + r$, round by round, with the worst check error in units of the
    tolerance. Each round adds #ex.chebtt.defaults.rank_step lines each way, or doubles a degree.],
)

#implementation(title: [The lattice store])[
  The degrees are $8 dot 2^m$, at most `n_max` $= #ex.chebtt.defaults.n_max$, and node $j$ of degree $n$
  is node $2 j$ of degree $2 n$. Values are stored by their position on the finest lattice, #num(ex.chebtt.defaults.lattice)
  intervals per axis (`LATTICE`), so a fibre seen again, or an old node of an axis whose degree doubled,
  costs no evaluation. The store (`KeyStore`) looks up a batch of keys by a 64-bit hash and checks every
  hit against the full key, so a collision raises instead of returning another point's value.
]

== Three check sets, and the choice between degree and rank <sec:checks>

#question[The cross interpolates $f$ on its fibres. How does ChebTT learn how wrong it is between them,
  and what should it refine?]

A model is exact on its own fibres (@lem:skeleton), so its errors are elsewhere. ChebTT measures them
at three sets of points after every round. No one set suffices.

#figure(panel("checks"), caption: [The three check sets of round #ex.chebtt.checks.round on $b + r$,
  each point coloured by the model's error there (colours as in @sec:reading). *Left.* The
  #ex.chebtt.checks.sobol Sobol points, fixed for the whole run. *Middle.* #num(ex.chebtt.checks.free)
  free checks: grid points evaluated in earlier rounds that the current cores do not interpolate. *Right.*
  The look-ahead: #ex.chebtt.checks.look points along every axis through the worst check point and
  through the largest $|f|$ seen (crosses), where the next nested grid would put its new nodes.]) <fig:checks>

- *Sobol points* cover the box evenly. Their worst error here is #ex.chebtt.checks.worst_sobol times
  the tolerance. They miss small features, because in high dimension a narrow peak fills a vanishing share of
  the box.
- *Free checks* are points ChebTT has already paid for. Fibres of earlier rounds, and the nodes of a
  degree since doubled, are no longer interpolated by the current cores. They cluster where the cross
  explored, which is where $f$ is interesting. Here their worst error is #ex.chebtt.checks.worst_free.
- *The look-ahead* evaluates the next nested grid's new nodes along every axis, through the worst
  check point and through the largest value seen. These are exactly the points a refinement would
  sample next. Its worst error, #ex.chebtt.checks.worst_look, is the round's worst. Look-ahead points
  join the check set for every later round.

If every error is within $s atol$, the model is accepted. Otherwise ChebTT must choose: more degree
or more rank. A missing degree cannot be made up by rank, and the reverse is also true. ChebTT tests,
for each axis, whether its top degrees still matter at the check points. It removes the top quarter of axis
$k$'s coefficients and measures how much the model changes there. That change is
@lem:alias's tail, measured on the model itself. An axis whose change exceeds its share $tau = s atol slash d$
doubles its degree. If no axis does, the rank grows.

#scene(side: left, width: 40%, panel("rounds"))[
  The rounds of $b + r$: the worst check error against the evaluations spent, marked by what came
  next. #swatch(fib1) The rank grows, #swatch(fib2, shape: "dot") a degree doubles, #swatch(green)
  the model is accepted. The dotted line is $s = #ex.chebtt.defaults.safety$, so a check must be within
  half the tolerance. After #num(rnds.at(0).evals) evaluations the error is #rnds.at(0).worst times
  the tolerance. Rank alone takes it to #rnds.at(2).worst, then axis 1 doubles to degree
  #rnds.at(3).degrees.at(0), and #rnds.slice(3).filter(r => r.action == "rank").len() more rank steps bring it to
  #rnds.last().worst.
]

#implementation(title: [One core's work per question])[
  Removing axis $k$'s top degrees changes only core $k$. The model with core $k$ replaced by $C'$ is
  $L_k (x) C'(x_k) R_(k+1)(x)$, where $L_k$ and $R_(k+1)$ are the products of the cores to the left and
  right of $k$ at each check point (`ChebTT.environments`, `through`). One set of environments per round
  answers every axis's question, and the look-ahead's lines too, since a line along axis $k$ shares
  every other coordinate with its base point.
]

#pitfall(title: [The checks are estimates, not bounds])[
  A feature that none of the three sets crosses is missed. ChebTT seeds its first right sets at the node
  nearest the largest $|f|$ seen and at the centre, because maxvol only moves among the fibres it is
  shown. Fibres that all miss a narrow peak converge to a model of the empty background. No scheme that
  samples a box can certify its worst error in high dimension @novak2008, and the benchmark judges each
  scheme on its test sets.
]

== Kinks and jumps aligned with an axis: piecewise axes <sec:edges>

#question[The base $b$ has a kink at $x_1 = #ex.feature.kink$. Its coefficients fall like $j^(-2)$, so
  degree alone converges slowly. Can ChebTT find the kink and put it where it does no harm?]

A tensor train is a sum of products of univariate functions. Such a sum can fail to be smooth only
along planes $x_k = c$, aligned with an axis. There it is enough to cut the axis at $c$ and give each
piece its own Chebyshev basis. On each piece $f$ is smooth again, and its coefficients fall
geometrically. An axis may therefore carry _breakpoints_. Its pieces share one degree, and its nodes
are the union of the pieces' Chebyshev points. ChebTT searches for breakpoints only where they can help,
on an axis whose coefficients decay like a power.

#algorithm(caption: [The edge search on one axis: `axis_edges` in `src/schemes/chebtt/train.py`.],
  pseudocode(edges)) <alg:edges>

#figure(panel("edges"), caption: [The edge search on the base $b$, read off the oracle calls it made.
  *Left.* The #(ex.chebtt.edges.scan)-point scan along the fibre $x_2 = #ex.chebtt.edges.base_x2$ and the size
  of its second differences (bars); the outlier is #swatch(kink) gold. *Middle.* Locating it: value
  bisection (grey) tries a jump first and is turned down after #ex.chebtt.edges.bisect_steps halvings, one
  evaluation each; the kink zoom (gold) then narrows the bracket in #ex.chebtt.edges.zoom_steps steps of 9
  points, to what the tolerance needs. *Right.* The
  #ex.chebtt.edges.confirm far fibres (dashed) that confirm the kink at the same $x_1$.]) <fig:edges>

The scan finds a cell whose second difference is an outlier. A kink's second difference is of order
$h |[f']|$ against $h^2 |f''|$ elsewhere, so it stands out at every resolution. A jump is tried first, by
bisection on the value, one evaluation per halving. The two-scale test turns it down. A jump's
one-sided difference keeps its size as the stencil shrinks, and a kink's does not. The zoom then
locates the kink only as finely as the tolerance needs, here to #sci(ex.chebtt.edges.zoom_width) of the
width. The far fibres decide whether the edge is aligned with the axis. If it is present at the same
$x_1$, or absent where its factor vanishes, the edge is kept. If it appears elsewhere, the edge is
oblique and no breakpoint represents it. @sec:adaptive treats that case.

The search cost #ex.chebtt.edges.evals evaluations and found the kink at
$x_1 = #ex.chebtt.edges.found$, #sci(ex.chebtt.edges.error) of the width from the true position. With it,
ChebTT models $b$ with #num(mdl.b.evals) evaluations, two pieces of degree #mdl.b.degrees.at(0) on
$x_1$, and a worst error of #mdl.b.worst times the tolerance. Without it (`edges=False`) the same
function costs #num(mdl.b_no_edges.evals) evaluations and degree #mdl.b_no_edges.degrees.at(0), and its
worst error is #mdl.b_no_edges.worst. The figure script rebuilds the zoom's last step from the recorded
values and stops unless it gives the model's breakpoint.

#implementation(title: [Bands, and why nodes stay off the edge])[
  A breakpoint is known only to within its location error and the largest displacement the confirming
  fibres saw. That margin is its _band_. The pieces' end nodes are inset by twice the band, so no node of
  one piece falls on the other side of a jump. The inset is at least $10^(-10)$ of the width and at most a
  quarter of either neighbouring piece. The model is not claimed within the band. Edge search draws from its own random stream, so a
  function without edges gets exactly the model it would get without the search; only its evaluation
  count differs. An axis has at most `max_pieces` $= #ex.chebtt.defaults.max_pieces$ pieces.
]

== Compression: a smaller train for the same tolerance <sec:compress>

#question[The cross grows ranks in steps of two and doubles degrees. The accepted model is larger than
  the tolerance needs. Inference pays for every coefficient. How much can be removed?]

The checks leave the accepted model at most $s atol$ from $f$ at its check points, and usually much less.
That leaves room, and a quarter of the tolerance, `chop_share` $= #ex.chebtt.defaults.chop_share$, goes
to compression.

#algorithm(caption: [Compression after acceptance: `compress` in `src/schemes/chebtt/model.py`. $beta$
  is `chop_share` $dot atol$; $Z$ is the check points and `n_probe` $= #num(ex.chebtt.defaults.n_probe)$
  uniform points.], pseudocode(compression)) <alg:compress>

#figure(panel("compress"), caption: [*Left.* The cores of $b + r$ as accepted (grey) and as compressed (coloured), drawn as
  (Chebyshev rows) $times$ (rank) blocks: #num(ex.chebtt.compress.entries_before) coefficients become
  #num(ex.chebtt.compress.entries_after). Rank rounding took the bond from
  #ex.chebtt.compress.shapes.at(0).before.at(2) to #(ex.chebtt.compress.shapes.at(0).after.at(2))\; degree
  cuts took axis 2 from #(ex.chebtt.compress.shapes.at(1).before.at(1) - 1) to
  #(ex.chebtt.compress.shapes.at(1).after.at(1) - 1). *Right.* How far the compressed model moved from
  the accepted one at the #num(ex.chebtt.compress.probe) probe points: at most
  #ex.chebtt.compress.change times the tolerance, within the budget (dotted). No evaluation of $f$ is
  involved; the comparison is between two models.])

#lemma(title: [what a coefficient can change])[
  On $[-1, 1]$, $|T_j| <= 1$. So removing coefficients from a Chebyshev series changes it by at most the
  sum of their magnitudes, everywhere. In a tensor train, removing degree $j$ of axis $k$ changes the
  model at $x$ by $T_j (x_k) L_k (x) C_k [:, j, :] R_(k+1)(x)$.
] <lem:chop>

ChebTree uses the first statement directly. It chops trailing slabs while their magnitudes sum to
within the budget (`chop`). ChebTT uses the second, measured at the probe points, because the
environments $L_k$, $R_(k+1)$ vary from point to point and a bound over them is loose. So ChebTT's
compression is guaranteed only at the probe, like its acceptance. The figure script repeats the
compression from the recorded accepted model, checks that it gives the returned one, and checks the
budget.

== Inference is linear in the dimension <sec:inference>

#theorem(title: [the cost of one answer])[
  Evaluating @eq:tt at one point costs, per axis, the $n_k + 1$ values $T_0 (x_k), ..., T_(n_k)(x_k)$ and
  $r_(k-1)(n_k + 1) r_k$ multiply-adds. At fixed ranks and degrees the cost is linear in $d$.
] <thm:inference>

#proof[
  Carry a row vector from left to right. Start with $v = 1$, and at axis $k$ form
  $v <- v G_k (x_k)$, where $G_k (x_k) = sum_j T_j (x_k) C_k [:, j, :]$ is an $r_(k-1) times r_k$
  matrix. The recurrence gives the $T_j$, and the sum and the product cost
  $r_(k-1)(n_k + 1) r_k$ multiply-adds.
]

The compressed Rosenbrock model holds #mdl.rosenbrock.madds coefficients, the base $b$
#mdl.b.madds, and $b + r$ #mdl.br.madds. A tensor grid of the same degrees would hold the products of
the node counts, and a sparse grid or a tree of boxes many more. The benchmark measures the time.
ChebTT's median inference over the smooth cases it holds is a fraction of a microsecond per query
(@sec:compare).

#implementation(title: [Prediction in the code])[
  `ChebTT.predict` works in chunks of queries, which bound the temporaries. Per core it multiplies the Chebyshev values by the
  core first, one matrix product $(Q, n + 1) times (n + 1, r_(k-1) r_k)$, then the running row vectors by
  that. This order avoids a strided $(Q, n + 1, r)$ temporary. Cores of degree 0 are constant matrices.
  They are multiplied into a neighbour once (`active`), so a function of few of its inputs pays only for
  those. On an axis with breakpoints a query uses its own piece's rows, zero in the others.
]

== Where ChebTT stumbles: oblique features <sec:rank-pitfall>

#scene(side: left, width: 34%, panel("angle"))[
  @prop:ranks says that a ridge aligned with an axis has rank 1, and that $cos(q dot x)$ has rank 2 at
  any angle. Steep monotone ridges are neither. Turn the feature example's ridge alone from the $x_1$
  axis towards the diagonal. ChebTT's rank climbs from #ex.chebtt.angle.at(0).rank at
  #ex.chebtt.angle.at(0).angle° to #ex.chebtt.angle.last().rank at #ex.chebtt.angle.last().angle°, and
  its evaluations from #num(ex.chebtt.angle.at(0).evals) to #num(ex.chebtt.angle.last().evals). Every
  case still holds the tolerance. The same function costs #calc.round(ex.chebtt.angle.last().evals / ex.chebtt.angle.at(0).evals) times as many evaluations only because it is turned.
]

#pitfall(title: [Oblique and curved features cost rank])[
  A feature across an oblique line or a curved surface
  has a high rank at any degree, and the cross pays for rank twice: more fibres, each through every node
  of its axis. In ten dimensions an oblique coupling between every pair of axes costs rank on every bond.
  An oblique _jump_ is worse. Its rank never converges. ChebTT on all of $F$ ran out of rank, at
  `rank_max` $= #ex.chebtt.defaults.rank_max$, after #num(mdl.brj.evals) evaluations. The next three
  sections are three answers: turn the coordinates (ChebTT-M), model the jump separately (ChebTT-A), or
  model the steep directions densely and the rest cheaply (the hub expansion).
]

#takeaway[ChebTT stores a Chebyshev interpolant as a tensor train, samples it along maxvol fibres, checks
  it at three sets of points, finds axis-aligned edges, and answers in time linear in $d$; oblique
  features are what it pays for.]

= ChebTT-M: the coordinates the outputs share <sec:metric>

#question[The ridge costs ChebTT rank only because it is turned. If we knew its direction $q$, we could
  turn the coordinates instead. How do we find $q$, and when is turning worth it?]

#let mt = ex.metric
#let lad = mt.ladder

A function of $q dot x$ alone is a function of one variable in the right coordinates. Take an
orthogonal $Q$ whose first column is $q$, and write $u = Q^top (x - c)$. In $u$ the ridge depends on
$u_1$ only, a product of one univariate function and constants, so it has rank 1 (@prop:ranks). The problem is
to find $Q$ from evaluations of $f$, and to know when it helps. Many of the library's problems have
several outputs, the species of a chemistry step or the states of a flow map, and outputs often share
such directions. So ChebTT-M works on vector-valued $f$, output by output.

== The first answer, and where it stumbles

The classical answer is the _active subspace_ @constantine2015: the eigenvectors of the average
gradient outer product. Applied to all outputs at once, it gives one rotation for everything, and that
is where it fails. The feature example's base is separable. On the axes ChebTT models it at rank
#mt.base_turned.axes_rank with #num(mt.base_turned.axes_evals) evaluations. In the ridge's coordinates
its kink is oblique, and ChebTT #if mt.base_turned.failed == none [needs rank #mt.base_turned.rank and
#num(mt.base_turned.evals) evaluations] else [fails. It reaches its rank limit after
#num(mt.base_turned.evals) evaluations]. A rotation helps an output only if the output is turned; for
one that is already aligned with the axes, it raises the rank.

== A metric per output

#definition(title: [pullback metric, @zahm2020])[
  The _pullback metric_ of output $c$ is $G_c = EE[nabla f_c nabla f_c^top]$, the average over the box
  of its gradient's outer product, here estimated at pilot points and divided by its trace.
] <def:metric>

#proposition(title: [a ridge in its own coordinates])[
  If $f_c (x) = g(q dot x)$ with $|q| = 1$, then $G_c = EE[g'(q dot x)^2] q q^top$ has rank 1 and its
  eigenvector is $q$. In the coordinates $u = Q^top (x - c)$ of any orthogonal $Q$ with first column
  $q$, $f_c$ is a function of $u_1$, and its tensor train has rank 1.
] <prop:ridge>

#proof[
  $nabla f_c = g'(q dot x) q$, so every outer product is a multiple of $q q^top$. And
  $q dot x = u_1 + q dot c$.
]

#figure(panel("metric"), caption: [The metrics of three outputs on $[-1, 1]^2$: the
  #swatch(ridge) ridge, a #swatch(mcol("chebtt_metric")) bump on the same direction, and the
  #swatch(kink) base. Each arrow is an eigenvector of the output's metric, its length the square root of
  the eigenvalue. The ridge's and the bump's metrics are lines along $q$. Their second eigenvalues are
  #sci(mt.outputs.at(0).eig.at(1)) and #sci(mt.outputs.at(1).eig.at(1)) of the first. The base's are
  #mt.outputs.at(2).eig.at(0) and #mt.outputs.at(2).eig.at(1).]) <fig:metric>

An output is turned when 99% of its metric's trace lies in at most
half as many eigenvectors as it lies in axes, the diagonal entries of $G_c$. The ridge's trace lies in
#mt.outputs.at(0).k_eig eigenvector against #mt.outputs.at(0).k_axes axes, so it is turned, and so is
the bump. The base needs #mt.outputs.at(2).k_eig of either. A separable output is never turned, since a
rotation could only raise its rank. Outputs that are turned are grouped when their leading subspaces
merge without growing, and each group is trained in the eigenvectors of its summed metric. The figure
script repeats the rule from these definitions and stops unless it gives `metric_groups`' groups.

#let metric-alg = (
  require[$f$ with $m$ outputs on the box, $atol$, the rank cap $r_"cap"$ (`axes_rank` $= #mt.axes_rank$)],
  ensure[for every output a ChebTT, on the axes or in its group's coordinates],
  ln(0, [*for* each output $c$: ChebTT on the axes, ranks capped at $r_"cap"$], note: [one cache of $f$]),
  ln(0, [the outputs within $atol$ keep that model], note: [nothing more paid]),
  ln(0, [for the others: $G_c$ from forward differences at $2^ceil(log_2 (4 + 2 d))$ Sobol points]),
  ln(0, [turn $c$ if 99% of its trace lies in at most half as many eigenvectors as axes], note: [`metric_groups`]),
  ln(0, [group the turned outputs whose leading subspaces merge without growing]),
  ln(0, [*for* each group: ChebTT in its eigenvectors, on the box that bounds the domain there]),
  ln(1, [unless $f$ is not finite, or more than 10 times its pilot values, at $2 d$ points of that box], note: [then the axes]),
)
#algorithm(caption: [ChebTT-M: `chebtt.build_vector(..., rotate=True)` in `src/schemes/chebtt/train.py`.],
  pseudocode(metric-alg)) <alg:metric>

#story()
#step([Try the axes first], code: (label: <alg:metric>, lines: metric-alg), focus: (1, 2), width: 38%, panel("metric-ladder"))[
  Deciding from the pilot first would pay the pilot on every problem, and turn outputs that are already
  cheap on the axes. So every output first trains on the axes with its rank capped at #mt.axes_rank.
  Only the outputs that reach the cap get a pilot. On the pair at $d = 2$ the ridge reaches it and
  is turned, and the bump is not. ChebTT-M costs #num(lad.at(0).metric.evals) evaluations against
  ChebTT's #num(lad.at(0).chebtt.evals), the price of the pilot and of a capped attempt that is thrown
  away. The gain grows with the dimension. ChebTT-M #swatch(mcol("chebtt_metric")) spends
  #num(lad.at(1).metric.evals) evaluations against the #num(lad.at(1).chebtt.evals) of ChebTT
  #swatch(mcol("chebtt")) at $d = #lad.at(1).d$, and #num(lad.at(2).metric.evals) against
  #num(lad.at(2).chebtt.evals) at $d = #lad.at(2).d$. At $d = #lad.at(3).d$ ChebTT exceeds its budget of #num(mt.max_evals) evaluations
  (cross), and ChebTT-M holds with #num(lad.at(3).metric.evals).
]

#step([The rotated box], code: (label: <alg:metric>, lines: metric-alg), focus: (6, 7), width: 38%, panel("rotated-box"))[
  In the group's coordinates the domain is a turned square. The model needs a box, so it uses the box
  that bounds the domain there (dark outline), #mt.box_area times the domain's area in two dimensions,
  and more in higher ones. The ridge's level sets are parallel to one of its sides (dashed). Training
  evaluates $f$ in the grey corners, outside the domain. An ODE can leave its positive orthant there,
  and a polynomial grow fast. So a group falls back to the axes when $f$, at $2 d$ random points of its
  box, is not finite or is more than ten times its largest pilot value.
]

On the pair the compressed models of ChebTT-M hold #lad.at(2).metric.coefficients coefficients at
$d = #lad.at(2).d$, against #num(lad.at(2).chebtt.coefficients) for ChebTT. Every worst error on the
test points is within the tolerance, at #lad.at(2).metric.worst and #lad.at(2).chebtt.worst of it.

#pitfall(title: [When not to turn, and what turning cannot do])[
  ChebTT-M on the axes _is_ ChebTT, at the same cost, for every output that meets the tolerance there. It
  pays more where a pilot buys nothing: at $d = 2$ on the pair, and wherever an output is turned that the
  axes would have handled at full rank. The rotation is global, and a curved latent coordinate has no one
  direction. The pilot sees nothing of a feature its points miss. The rotated box extends beyond the
  domain, so $f$ must be defined there.
]

#takeaway[ChebTT-M turns only the outputs whose pullback metric says they are ridges, groups the ones
  that share their directions, and leaves every other output on the axes, so it never costs much more
  than ChebTT and gains more as $d$ grows.]

= ChebTT-A: a term for the jump <sec:adaptive>

#question[The jump is oblique and discontinuous. No breakpoint represents it, and turning the coordinates
  for it would turn the ridge and the base away from theirs. How can a tensor train still be used?]

#let ad = ex.adaptive

== The candidates, and where they stumble

- *More rank.* ChebTT on $F$ climbed to its rank limit of #ex.chebtt.defaults.rank_max after
  #num(mdl.brj.evals) evaluations (@sec:rank-pitfall). A jump's singular values barely fall (@fig:matrix).
- *A finer partition.* A tree of boxes refined towards an interface of dimension $d - 1$ needs about
  $(1 slash h)^(d-1)$ leaves at resolution $h$, all of them along the interface. That is the cost of
  AFINNI and ART at a jump, and of ChebTree.
- *Aligned coordinates.* Turning the box so that the jump's normal is an axis makes the jump a
  breakpoint, but turns the base and the ridge away from their axes.

ChebTT-A takes the jump out instead. If $S$ is a term with the same jump as $f$ across the interface,
$f - S$ is smooth there, and ChebTT handles it:
$ f approx "ChebTT"(f - S) + S. $ <eq:adaptive>

#let adaptive-alg = (
  require[$f$ on the box, $atol$],
  ensure[a ChebTT of $f - S$, and the interface term $S$ ($S = 0$ when $f$ needs none)],
  ln(0, [ChebTT with piecewise axes on a first budget; *return* it if it holds], note: [stops at an oblique edge]),
  ln(0, [interface points $<-$ jumps on segments from the most extreme check point], note: [`segment_edges`]),
  ln(0, [$phi <-$ the simplest of a plane and a diagonal quadric that the points lie on], note: [`fit_interface`]),
  ln(0, [one-sided jumps of $f$, $f'$, $f''$ along the normal at interface points], note: [`interface_jets`]),
  ln(0, [$S <- h(phi)$ if the jump is constant, else jump fields of the foot point]),
  ln(0, [*return* ChebTT of $f - S$, and $S$]),
)
#let adaptive-code = (label: <alg:adaptive>, lines: adaptive-alg)
#algorithm(caption: [ChebTT-A: `adaptive.build` in `src/schemes/chebtt/adaptive.py`. A front, a steep
  smooth step with no jump, takes a branch of line 2. There $phi$ is fitted to crossings of the mid-level
  $(max f + min f) slash 2$, and $S = h(phi)$.], pseudocode(adaptive-alg)) <alg:adaptive>

#story()
#step([The first attempt meets the jump], code: adaptive-code, focus: (1, 2, 3), width: 40%, panel("adaptive-discovery"))[
  The first ChebTT's edge search finds the jump on a fibre, but on the far fibres it is somewhere else,
  so it is oblique. The build stops there (`ObliqueEdgeError`) after #num(ad.stages.at(0).evals) evaluations and
  hands over the #ad.interface.first_points points it located on the interface (diamonds). Segments from
  the most extreme check point (cross) to random points add more (dots). Each is scanned at 17 points and
  bisected on the value at its largest jump. A segment crosses a region of volume share $p$ with
  probability about $2 p (1 - p)$ in any dimension; an axis fibre does much worse as $d$ grows. A plane
  fits the points (dashed). Its normal is #ad.interface.angle_error° from $p$, and its offset
  #sci(ad.interface.offset_error) from the true one.
]

#step([How big is the jump?], code: adaptive-code, focus: (4,), width: 40%, panel("adaptive-jets"))[
  At #ad.jets.points interface points ChebTT-A walks along the normal on both sides. It fits a polynomial
  of degree 6 to 8 points on each side, within 0.2 of the point and inside the box, so no evaluation
  falls outside the domain. The difference of the two fits at the interface is the jump of $f$, and
  of their derivatives the jumps of $f'$ and $f''$. Here the jump, the side where $phi > 0$ minus the
  other, is #ad.jets.example.jump. Over all points its mean size is #ad.jets.jump_mean and its spread
  #sci(ad.jets.jump_spread). The figure script repeats the two fits and stops unless they give the
  library's jump.
]

#step([The remainder is smooth], code: adaptive-code, focus: (5, 6), width: 36%, panel("adaptive-remainder"))[
  In $F - S$ the jump is gone, and the base and the ridge remain. The jump's line is still drawn
  #swatch(jumpc, shape: "line"). The remainder's ChebTT has rank #ad.remainder.tt.ranks.at(0) and costs
  #num(ad.remainder.tt.evals) oracle calls, many of them answered by the cache the stages share. ChebTT-A
  spends #num(ad.evals) evaluations in all.
]

#figure(panel("adaptive-result"), caption: [*Left.* The error of ChebTT-A on $F$, within the tolerance
  everywhere, with a worst error of #ad.worst. *Right.* The coefficients along the jump slice of @fig:decay, of $F$
  #swatch(jumpc, shape: "line") and of $F - S$ #swatch(dark, shape: "line"). The largest in the upper
  half of the degrees falls from #ad.remainder.tail_f to #sci(ad.remainder.tail_remainder).]) <fig:adaptive-result>

#proposition(title: [what the interface term buys])[
  Let $f$ be smooth on each side of a smooth interface $phi = 0$, and let $S$ have the same jumps as $f$,
  $f'$, ..., $f^((k))$ along the normal at every point of it. Then $f - S$ is $k$ times continuously
  differentiable across the interface. Along any line that crosses it, its Chebyshev coefficients fall
  like $j^(-k-2)$, by @thm:decay with $nu = k + 1$. A residual jump of size $epsilon$ in $f - S$ adds a
  tail of order $epsilon slash j$.
] <prop:remainder>

#proof[
  On each side $f - S$ is smooth. At the interface the one-sided limits of $f - S$ and its first $k$
  normal derivatives agree by construction, and the tangential derivatives of a smooth jump field agree
  with them. So $(f - S)^((k))$ is continuous and $(f - S)^((k+1))$ has bounded variation along the line.
]

The term $S$ has two forms. A jump of constant size becomes a profile, $S = h(phi(x))$, with $h$ a 1-D
piecewise ChebTT sampled along the segment of the box from the smallest $phi$ to the largest. A jump
that varies along the interface becomes _jump fields_,
$ S(x) = [rho(x) s > 0] sum_(k=0)^2 J_k ("foot"(x)) rho(x)^k / k!, $
where $rho$ is the signed distance to the interface, the foot is the nearest point on it, $s$ chooses
the side, and $J_k$ are polynomials of the foot whose degree grows (0, 1, 2) until held-out interface
points agree to $0.1 atol$. They are defined on the side whose foot map is regular. The other side of a closed
quadric holds its centre, where the foot is undefined.

#pitfall(title: [One interface, and a band around it])[
  ChebTT-A models one interface, a level set of a plane or of a diagonal quadric. A query within the
  interface's location error may be on the wrong side, so the claim is the tolerance outside that band,
  and outside the bands of the breakpoints. Here the jump is constant, yet ChebTT-A built jump fields
  of degree #ad.interface.degree instead of the cheaper profile. Its test for a constant jump reads the
  jumps of $f''$ too, and the one-sided fits measure them noisily. Their spread over the interface
  points is #ad.jets.curvature_spread, against #sci(ad.jets.jump_spread) for $f$ itself. The first
  attempt is spent on every function with an interface. Here it took #num(ad.stages.at(0).evals) of the
  #num(ad.evals) evaluations.
]

#takeaway[ChebTT-A takes the one thing a tensor train cannot follow, an oblique or curved jump, out of
  $f$ as a term aligned with the interface, and leaves ChebTT a remainder that is smooth across it.]

= The hub expansion: dense where it is steep, cheap elsewhere <sec:hub>

#question[In ten dimensions even a turned tensor train pays for every axis. If a function is steep in
  one or two directions and nearly flat in all the others, can a model's cost follow that structure?]

#let hb = ex.hub

The library's `shared_latent` problem is built for that question. Its #hb.problem.outputs outputs are
nonlinear functions of two coordinates $g_1$, $g_2$, oblique to every axis, of a $d$-dimensional input,
and $g_1$ is curved. It stands in for a slow manifold, a set of outputs that share structure the axes do not
show. At $d = #hb.problem.d$ and the tolerance #hb.problem.alpha, the benchmark's ChebTT-M needed
#num(hb.store_metric.evals) evaluations#if hb.store_metric.stale [ (a record older than ChebTT-M's last
change to its results)]. The hub expansion needs #num(hb.values.evals).

== The gradient covariance tells the hub

#scene(side: left, width: 40%, panel("hub-spectrum"))[
  The eigenvalues of the gradient covariance, pooled over the outputs and divided by its trace: for
  #swatch(mcol("hub")) `shared_latent` at $d = #hb.problem.d$, and for #swatch(ridge) the feature
  example's ridge lifted to $d = #hb.problem.d$ with weak terms added. `shared_latent` puts all of its
  trace in two eigenvalues, #hb.spectra.shared_latent.top.at(0) and #hb.spectra.shared_latent.top.at(1).
  The lift puts #hb.spectra.lift.top.at(0) in one and spreads the rest thinly over the others.
]

#lemma(title: [ridges have a low-rank gradient covariance, @constantine2015])[
  If $f(x) = g(V_r^top x)$ with $V_r$ a $d times r$ matrix of orthonormal columns, then
  $C = EE[nabla f nabla f^top] = V_r EE[nabla g nabla g^top] V_r^top$ has rank at most $r$, and its range
  lies in the span of $V_r$.
] <lem:ridge-covariance>

So the hub dimension $r$ is read off the spectrum. It is the fewest eigenvectors holding 99% of the trace,
here #hb.spectra.shared_latent.r for `shared_latent` and #hb.spectra.lift.r for the lift, at most
`r_max` $= #hb.settings.r_max$. What the hub leaves out, #sci(hb.spectra.lift.share_left) of the lift's
trace, decides whether the other directions get terms at all. They do when it exceeds $10^(-4)$.

#definition(title: [hub expansion])[
  In rotated coordinates $(u, v) = V^top x$, with the $r$ hub directions first,
  $ fh(x) = P_0 (u) + sum_(j = r + 1)^d [P_(1 j)(u) T_1 (v_j) + P_(2 j)(u) T_2 (v_j)], $ <eq:hub>
  where every $P$ is a polynomial of total degree $p$ in the hub coordinates $u$, in the Chebyshev
  basis. It is a truncated ANOVA expansion @rabitz1999 around a dense low-dimensional core, in rotated
  coordinates @tipireddy2014, fitted by linear least squares.
] <def:hub>

#proposition(title: [the terms are linear in $d$])[
  With $N_h = binom(r + p, r)$ hub terms, @eq:hub has $N_h (1 + 2 (d - r))$ terms, or $N_h$ without weak
  terms. A full polynomial of total degree $p$ in $d$ variables has $binom(d + p, d)$.
] <prop:hub-terms>

#scene(side: right, width: 40%, panel("hub-terms"))[
  The hub's terms #swatch(mcol("hub")) against a full polynomial of the same degree #swatch(grey), as
  $d$ grows, for the lift's hub: $r = #hb.terms.r$ direction at degree #hb.terms.degree with weak terms.
  At $d = 10$ that is #num(hb.terms.d10) terms against #sci(hb.terms.total_d10) for the full polynomial;
  at $d = 100$ it is #num(hb.terms.d100). The figure script counts the terms from this definition and stops
  unless the count matches every fitted model's.
]

#let hub-alg = (
  require[$f$ on the box, $atol$, optionally its Jacobian],
  ensure[per group of outputs: a rotation $V$, a hub dimension $r$, a degree $p$, coefficients],
  ln(0, [gradients at #hb.settings.pilot pilot points: the Jacobian, or central differences], note: [$2 d$ evaluations each]),
  ln(0, [per output: $C$, and $r <-$ the fewest eigenvectors holding 99% of its trace]),
  ln(0, [outputs whose hubs agree with the pooled one share it; weak terms if more than $10^(-4)$ is left]),
  ln(0, [$n <- n_0$; *repeat*], note: [$n_0 = #num(hb.settings.n0)$]),
  ln(1, [design: half by pivoted QR of the hub basis on uniform, vertex-directed and blended points]),
  ln(1, [$p <-$ the candidate degree with the smallest worst error on a hold-out from the same design]),
  ln(1, [least squares at $p$; *stop* if the hold-out is within $atol slash 2$, or improved by less than 10%]),
  ln(1, [$n <- 2 n$ (within `max_evals`)]),
  ln(0, [refit on every sample, the hold-out included]),
)
#algorithm(caption: [The hub expansion: `hub.build` in `src/schemes/hub.py`.], pseudocode(hub-alg)) <alg:hub>

== Where to sample

#figure(panel("hub-design"), caption: [#num(hb.design.points) points of `shared_latent`'s box, seen in
  the two hub coordinates (scaled to $[-1, 1]$). *Left.* Uniform in the box. *Right.* The hub's design.
  Uniform points pile up at the centre. Their largest hub coordinate reaches #hb.design.uniform_reach,
  and #pct(hb.design.uniform_outer) of them lie beyond 0.5. The design reaches #hb.design.design_reach,
  with #pct(hb.design.design_outer) beyond 0.5.]) <fig:hub-design>

A sum of $d$ uniform coordinates concentrates near its mean, so a uniform sample of a 10-D box rarely
reaches the extremes of an oblique coordinate. A polynomial fitted there extrapolates at the edges of
its own variables, which is where its worst error lands. The design fixes this in two ways. It
draws _vertex-directed_ points, corners of the box chosen by the sign of a random combination of the
hub directions, and blends of them with uniform points. Then it picks a well-conditioned set from that
pool by a pivoted QR of the hub basis, and adds uniform points.

#scene(side: left, width: 38%, panel("hub-budget"))[
  The degree is chosen on a hold-out drawn from the same design. It is the candidate whose fit on part of
  the samples has the smallest worst error on the rest. Then the budget doubles until the hold-out is within
  half the tolerance (dotted), or stops improving by 10%. On `shared_latent` one round is enough, with
  values only #swatch(mcol("hub")) and with the Jacobian #swatch(dark). On the lifted ridge
  #swatch(ridge) the second round improves the hold-out from #hb.lift.history.at(0).at(1) to
  #hb.lift.history.at(1).at(1) times the tolerance, less than 10%, and the budget stops. More samples of
  the same design do not fix an error of the model class. A hold-out worst above $atol$
  (`holdout_worst`) reports a model that missed, rather than hiding it.
]

== With the oracle's Jacobian

When the oracle supplies gradients, from an adjoint, automatic differentiation or ODE sensitivities, they
replace the pilot's central differences, and every sample's gradient also enters the least squares as
$d$ more equations. On `shared_latent`:

#block(breakable: false, above: 1em, below: 1em, table(
  columns: (1fr, auto, auto, auto, auto),
  inset: (x: 6pt, y: 5pt),
  align: (left, right, right, right, right),
  table.header[`shared_latent`, $d = #hb.problem.d$, $atol = #hb.problem.alpha$][evaluations][Jacobians][degree][worst / $atol$],
  [values only], [#num(hb.values.evals)], [#hb.values.jacobians], [#hb.values.groups.at(0).degree], [#hb.values.worst],
  [with the Jacobian], [#num(hb.jacobian.evals)], [#num(hb.jacobian.jacobians)], [#hb.jacobian.groups.at(0).degree], [#hb.jacobian.worst],
))

The Jacobian halves the evaluations and lowers the worst error. It is worth it when a gradient costs
less than a few evaluations, as the discrete sensitivities of a stiff ODE step do. Here the Jacobian is
central differences of $f$, standing in for an exact one, as in the library's regression check.

== Where it stumbles

#let lf = hb.lift
On the feature example's ridge lifted to $d = #hb.problem.d$ with weak terms (the cyan spectrum above),
the hub expansion misses. It spends #num(lf.evals) evaluations and reaches a worst error of
#lf.worst times the tolerance $#ex.feature.alpha$. Its hold-out says so: #lf.holdout. With the exact
Jacobian it comes within #lf.jacobian.worst. ChebTT-M on the same function holds, with
#num(lf.metric.evals) evaluations and a worst error of #lf.metric.worst. On Rosenbrock, in two
dimensions, the hub spends #num(hb.rosenbrock.evals) evaluations, its first budget, where ChebTT spends
#num(mdl.rosenbrock.evals).

#pitfall(title: [Outside its class])[
  The hub expansion suits a few steep directions with weak, low-order dependence on the rest. A steep
  profile along a hub direction needs a high degree, and the rotated coordinate's range grows with $d$,
  since it spans the box's extent along an oblique direction. Yet the candidate degrees stop at
  #hb.settings.degrees.at("1").last() for one hub direction. Isotropic and radial features, many
  differently oriented ones, and jumps are outside its class. The hold-out is an estimate from the same
  design, not a bound. Each group is capped at #num(hb.settings.max_terms) terms, since its Gram matrix
  costs the square of that and its solve the cube.
]

#takeaway[The hub expansion reads the few steep directions off the gradient covariance, fits a dense
  Chebyshev model in them and degree-two terms in all the others, so its cost grows linearly with $d$;
  it is the tool for low-dimensional structure in high dimension, and it reports when a function lacks that
  structure.]

= Side by side, and where AFINNI and ART stumble <sec:compare>

#question[The same functions, every method. What does each one give, and where does a partition into
  local affine pieces fail a max-norm tolerance that one global model holds?]

#let cp = ex.compare
#let st = cp.store
#let cell(c) = if "failed" in c [fails at #num(c.evals)] else [#num(c.evals) / #c.worst]
#let mname = (chebtt: [ChebTT], chebtt_metric: [ChebTT-M], chebtt_adaptive: [ChebTT-A], hub: [hub expansion],
  afinni: [AFINNI], art: [ART, affine leaves], art_quad: [ART, quadratic leaves])

== Rosenbrock

#board(
  (other("error-afinni"), [AFINNI: #num(ex.rosenbrock.afinni.evals) evaluations, worst #ex.rosenbrock.afinni.max_error]),
  (other("error-art"), [ART: #num(ex.rosenbrock.art.evals) evaluations, worst #ex.rosenbrock.art.max_error]),
  (panel("error-rosenbrock"), [ChebTT: #num(mdl.rosenbrock.evals) evaluations, worst #sci(mdl.rosenbrock.grid_worst)]),
  columns: 3,
)

On Rosenbrock the contrast is as large as it can be. The polynomial is exactly representable in
ChebTT's basis, so its model is exact to rounding, while each affine piece of AFINNI and ART carries
the curvature it cannot represent. The other companion's maps are on its own grid and in the same
colours. The hub expansion is exact too, but it spends its first budget, #num(hb.rosenbrock.evals)
evaluations, on a function of two variables.

== One function, four answers

#block(breakable: false, above: 1em, below: 1em, table(
  columns: (1.3fr, 1fr, 1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: (left, right, right, right),
  table.header[evaluations / worst error over $atol$][$b$][$b + r$][$F = b + r + j$],
  ..("chebtt", "chebtt_metric", "chebtt_adaptive", "hub").map(m => (
    mname.at(m), cell(cp.layers.b.at(m)), cell(cp.layers.br.at(m)), cell(cp.layers.brj.at(m)),
  )).flatten(),
  ..("afinni", "art", "art_quad").map(m => (
    mname.at(m), [--], [--], [#num(cp.baselines.at(m).evals) / #cp.baselines.at(m).worst],
  )).flatten(),
))

The table summarises the companion's argument. On the base, every scheme holds, and the three ChebTTs need
a few hundred evaluations. ChebTT-M and ChebTT-A count each point once, through a shared cache, while
ChebTT counts every call of the oracle, and its look-ahead can ask for a point twice. With the ridge
added, ChebTT pays rank, and the others pay about as much. With the jump added, ChebTT and ChebTT-M run
out of rank, and the hub expansion misses by #cp.layers.brj.hub.worst times the tolerance. Only ChebTT-A
holds, with #num(cp.layers.brj.chebtt_adaptive.evals) evaluations.

#board(
  (panel("compare-afinni"), [AFINNI: #num(cp.baselines.afinni.evals) evaluations, #cp.baselines.afinni.leaves leaves]),
  (panel("compare-art"), [ART, affine: #num(cp.baselines.art.evals), #cp.baselines.art.leaves leaves]),
  (panel("compare-art_quad"), [ART, quadratic: #num(cp.baselines.art_quad.evals), #cp.baselines.art_quad.leaves leaves]),
  (panel("compare-hub"), [hub expansion: #num(cp.layers.brj.hub.evals)]),
  (panel("compare-chebtt_adaptive"), [ChebTT-A: #num(cp.layers.brj.chebtt_adaptive.evals)]),
  columns: 5,
)

AFINNI and ART refine towards the jump, and keep refining. Their trees reach their depth limits with
#num(cp.baselines.art_quad.leaves) to #num(cp.baselines.afinni.leaves) leaves and more than a hundred
thousand evaluations. They are within the tolerance on #cp.baselines.afinni.within% to
#cp.baselines.art.within% of the grid, but their worst errors, #cp.baselines.art_quad.worst to
#cp.baselines.afinni.worst times the tolerance, sit on the jump. A piece that straddles it has an
error of half the jump at least, and a piece too small to straddle it is never small enough. In this
figure only ChebTT-A's map is pale everywhere.

== The benchmark

#scene(side: left, width: 42%, panel("store-cases"))[
  Each dot is a case of the benchmark's function suite (@sec:problems defines and draws every function), a
  problem, dimension and tolerance that both families ran. It places the best worst error of AFINNI and ART against the best of the
  schemes, over every seed. Colours are classes: #swatch(rgb(st.class_colours.smooth)) smooth,
  #swatch(rgb(st.class_colours.at("many local features"))) many local features,
  #swatch(rgb(st.class_colours.at("kinks, jumps, fronts"))) kinks, jumps and fronts,
  #swatch(rgb(st.class_colours.at("ODE flow maps"))) ODE flow maps, and
  #swatch(rgb(st.class_colours.at("vector-valued"))) vector-valued problems. A hollow dot's best scheme has
  records older than its last change to its results. An exact model is drawn at the
  bottom edge, and a failed run, over its budget or the time limit, at the right or top edge. Of #st.total cases, a scheme holds and no baseline
  does in #st.scheme_only, #st.high_d_scheme_only of them at $d >= 10$; a baseline holds and no scheme
  does in #st.baseline_only\; both hold in #st.both, and neither in #st.neither.
]

The table below is the benchmark's summary, per class of problem, from its store of #num(st.records) runs.
The README shows the same table. A case is _held_ when the worst error is within the tolerance on every seed, and
_won_ when it is held with the fewest evaluations. A method marked $ast$ has records older than its last
change to its results.

#let fmt-us(v) = if v == none [--] else [#v]
#let fmt-ev(v) = if v == none [--] else [#num(v)]
#block(above: 1em, below: 1em, {
  set text(8pt)
  table(
    columns: (auto, 1fr, auto, auto, auto, auto, auto, auto),
    inset: (x: 4pt, y: 3pt),
    align: (left, left, right, right, right, right, right, right),
    table.header[class][method][ran][held][won][held, $d >= 10$][evaluations][inference µs],
    ..st.classes.map(c => c.rows.enumerate().map(((i, r)) => (
      if i == 0 [#c.class] else [], [#r.label#if r.stale [ $ast$]], [#r.ran], [#r.held], [#r.won], [#r.held_high],
      fmt-ev(r.evals), fmt-us(r.us),
    )).flatten()).flatten(),
  )
})

Least-squares pieces on random points control the mean error but not the worst. Their errors between the points are unchecked, and that gap
grows with the dimension. Interpolation at Chebyshev nodes, with checks at the points a refinement would
sample next, keeps its worst errors below the tolerance. ChebTree holds
#st.classes.find(c => c.class == "kinks, jumps, fronts").rows.find(r => r.method == "chebtree").held of the
#st.classes.find(c => c.class == "kinks, jumps, fronts").rows.find(r => r.method == "chebtree").ran kinks,
jumps and fronts it ran, but only up to $d = #st.max_d.chebtree$ (@sec:aside).

== Choosing among the four

#cards(
  columns: 2,
  ([ChebTT], [Smooth functions, and functions with kinks or jumps along the axes, of any dimension,
    with low-rank coupling. Oblique and curved features cost it rank. On the base it needs
    #num(cp.layers.b.chebtt.evals) evaluations.]),
  ([ChebTT-M], [Vector-valued functions whose outputs are ridges of shared oblique coordinates. It is
    ChebTT on every output the axes serve, and its gain grows with $d$. On the ridge pair it spends
    #num(lad.at(2).metric.evals) evaluations against #num(lad.at(2).chebtt.evals) at $d = #lad.at(2).d$.]),
  ([ChebTT-A], [One jump across an oblique plane or a diagonal quadric, or a sharp front. It is the only
    scheme that holds all of $F$, with #num(cp.layers.brj.chebtt_adaptive.evals) evaluations.]),
  ([Hub expansion], [A few steep directions and weak dependence on the rest, in high dimension, with the
    oracle's Jacobian when it has one. On `shared_latent` at $d = #hb.problem.d$ it spends
    #num(hb.jacobian.evals) evaluations and #num(hb.jacobian.jacobians) Jacobians. It says when a function is outside its class.]),
)

#takeaway[Where a function has the structure a scheme is built for, the scheme holds the worst error
  with a few thousand evaluations, where local affine pieces spend a hundred thousand and still miss it.]

= How the cost grows with the accuracy <sec:convergence>

#let cg = json("figures/convergence.json")
#let ca = json("../adaptive-surrogates/figures/convergence.json")
#let ok-rows(rows) = rows.filter(r => "failed" not in r)
#let first-ok(rows) = ok-rows(rows).first()
#let last-ok(rows) = ok-rows(rows).last()
#let failed-row(rows) = rows.find(r => "failed" in r)
// a tolerance: a power of ten as one, any other value as it is
// A tolerance as a power of ten, a power of two, or a power of ten halved h times (AFINNI's and ART's ladders).
#let tolstr(t) = {
  let exact(u, base) = {
    let k = calc.round(calc.log(u, base: base))
    if calc.abs(u / calc.pow(float(base), k) - 1) <= 1e-9 { int(k) } else { none }
  }
  let k = exact(t, 10)
  if k != none { return if k == 0 { "1" } else { $10^(#k)$ } }
  let b = exact(t, 2)
  if b != none { return if b >= -2 { str(t) } else { $2^(#b)$ } }
  for h in range(1, 31) {
    let j = exact(t * calc.pow(2.0, h), 10)
    if j != none { return if j == 0 { $2^(-#h)$ } else { $10^(#j) dot 2^(-#h)$ } }
  }
  str(t)
}
#let rr(d, m) = cg.ridge.at(str(d)).at(m)
#let ros(d, m) = cg.rosen.at(str(d)).at(m)
#let ca-rate(m, d) = ca.rates.at(m).at(str(d)).leaves

#question[Cut the tolerance by ten. How many more evaluations does each scheme need, does its error
  follow, and how does that compare with the $atol^(-d slash 2)$ leaves of AFINNI and ART?]

#figure(panel("convergence-rosen"), caption: [Evaluations against $1 slash atol$ on Rosenbrock's valley,
  the AFINNI and ART companion's running example, extended along a chain,
  $f(x) = sum_(i=1)^(d-1) (1 - x_i)^2 + (x_(i+1) - x_i^2)^2$, on its box with every further axis like the
  second. #swatch(mcol("chebtt")) ChebTT, #swatch(mcol("chebtt_metric")) ChebTT-M,
  #swatch(mcol("chebtt_adaptive")) ChebTT-A, #swatch(mcol("hub")) the hub expansion; a hollow mark misses
  the tolerance. #swatch(mcol("afinni")) AFINNI, whose sample grows with its leaves, and
  #swatch(mcol("art")) ART are the other companion's runs @companion, at tolerances that halve.]) <fig:convergence-rosen>

The picture shows two kinds of curve. AFINNI's and ART's rise on these log-log axes, more
steeply in higher dimensions. The schemes' curves are flat: ChebTT spends #num(first-ok(ros(2, "chebtt")).evals)
evaluations at $d = 2$ and #num(first-ok(ros(5, "chebtt")).evals) at $d = 5$, whether the tolerance is
#tolstr(first-ok(ros(5, "chebtt")).tol) or #tolstr(last-ok(ros(5, "chebtt")).tol).

The other companion shows why the affine pieces rise. An affine model misses $f$ by its curvature
times the square of the leaf's size, so a leaf must shrink like $atol^(1 slash 2)$, and $d$-dimensional leaves
multiply like $atol^(-d slash 2)$. That _leaf-count law_ holds for any partition into affine pieces. Its fitted
exponent there is #ca-rate("art", 2) for ART at $d = 2$ and #ca-rate("art", 5) at $d = 5$, against $1$ and $2.5$.
A Chebyshev scheme's cost is set instead by how fast its coefficients fall (@thm:decay). For a
polynomial they stop at its degree. Rosenbrock has degree 4 along every axis but the last, so its interpolant is
exact once the degrees reach 4 and 2, and a tighter tolerance costs nothing. The ranks are 3, as
@prop:ranks predicts for a sum of terms that couple neighbours.

#translate(
  [the degree along axis $k$ that the tolerance needs], [$n_k (atol)$],
  [the ellipse inside which $f$ is analytic along an axis, and the bound of $|f|$ there], [$E_rho$, $M$ (@thm:decay)],
  [the tensor train's ranks; the evaluations the train spends], [$r_k (atol)$, $N(atol)$],
  [the number of axes an unfolding separates from the rest], [$m_k = min(k, d - k)$],
)

#proposition(title: [the cost of a tolerance])[
  Let $f$ be analytic and bounded by $M$ in $E_rho$ along every axis. A ChebTT within $atol$ needs at most
  $ n_k (atol) <= (log(4 M slash ((rho - 1) atol))) / (log rho) $
  along each axis, and its cross evaluates $f$ at $s sum_k r_(k-1) (n_k + 1) r_k$ points in $s$ sweeps,
  plus a number of check points that does not depend on $atol$. If the ranks stay bounded, then
  $N(atol) = O(d log(1 slash atol))$. If $f$ is a polynomial of degree $p_k$ along axis $k$, then $n_k = p_k$ for
  every $atol$ above rounding, and $N$ does not depend on $atol$.
] <prop:cost>

#proof[
  The bound on $n_k$ is @thm:decay solved for $n$. The cross evaluates the fibres $I_(k-1) times
  {"every node of axis" k} times J_k$, $r_(k-1) (n_k + 1) r_k$ points per axis and sweep. A polynomial of
  degree $p$ is its own interpolant at any $p + 1$ distinct nodes, so every coefficient beyond $p$ is
  rounding, and the tail estimate stops at $p$.
]

#proposition(title: [how far the ranks can grow])[
  The unfolding of a polynomial of degree $n$ along every axis that separates the first $k$ axes from the
  others has rank at most $(n + 1)^(m_k)$. So when $f$ is analytic in $E_rho$ along every axis, ranks
  $r_k <= (1 + n(atol))^(m_k)$ suffice, with $n(atol)$ of @prop:cost, and
  $N(atol) = O(d (log 1 slash atol)^(1 + 2 m))$ with $m = max_k m_k$.
] <prop:rank-growth>

#proof[
  Such a polynomial is a sum of products $T_(j_1)(x_1) dots T_(j_d)(x_d)$ with every $j_i <= n$. Group the
  terms by the indices on the side of the unfolding with fewer axes. There are at most $(n + 1)^(m_k)$
  groups, and each is one separable product of a function of those axes and a function of the others.
]

The bound is the worst case. It is polylogarithmic in $1 slash atol$, which eventually grows more slowly than
any power. But its exponent grows with the dimension, so the curse of dimensionality returns. It is reached when a feature is
oblique to every axis. It is avoided when the coupling is weak, as on Rosenbrock, or when the scheme turns
the coordinates.

== A smooth, oblique ridge

#figure(panel("convergence-ridge"), caption: [Evaluations against $1 slash atol$ on the feature example's
  ridge $r(x) = #ex.feature.ridge.height tanh(#ex.feature.ridge.steep (q_d dot x + #ex.feature.ridge.offset))$,
  lifted to $[-1, 1]^d$ with $q_d prop (cos #ex.feature.ridge.angle°, sin #ex.feature.ridge.angle°, ..., sin #ex.feature.ridge.angle°)$, which is $q$ at $d = 2$ (@sec:examples). Colours as in @fig:convergence-rosen; a cross is a
  build that failed.]) <fig:convergence-ridge>

The ridge is analytic, but it is oblique to every axis. ChebTT's ranks grow as the tolerance tightens, as
@prop:rank-growth allows. At $d = 3$ they rise from #first-ok(rr(3, "chebtt")).ranks.at(0) at
$atol = #tolstr(first-ok(rr(3, "chebtt")).tol)$ to #last-ok(rr(3, "chebtt")).ranks.at(0) at
#tolstr(last-ok(rr(3, "chebtt")).tol), and its evaluations from #num(first-ok(rr(3, "chebtt")).evals) to
#num(last-ok(rr(3, "chebtt")).evals). At $d = 5$ the middle unfoldings separate two axes from three, and
the build fails at $atol = #tolstr(failed-row(rr(5, "chebtt")).tol)$: #failed-row(rr(5, "chebtt")).failed.

ChebTT-M turns the ridge into one coordinate (@prop:ridge). Its rank stays at
#first-ok(rr(5, "chebtt_metric")).ranks.at(0) to #last-ok(rr(5, "chebtt_metric")).ranks.at(0), and only the
degree along the ridge grows. At $d = 5$ it spends #num(first-ok(rr(5, "chebtt_metric")).evals) evaluations at
#tolstr(first-ok(rr(5, "chebtt_metric")).tol) and #num(last-ok(rr(5, "chebtt_metric")).evals) at
#tolstr(last-ok(rr(5, "chebtt_metric")).tol), so
#calc.round(calc.log(first-ok(rr(5, "chebtt_metric")).tol / last-ok(rr(5, "chebtt_metric")).tol, base: 10)) decades
of the tolerance cost a factor of
#calc.round(last-ok(rr(5, "chebtt_metric")).evals / first-ok(rr(5, "chebtt_metric")).evals, digits: 1) in evaluations.
Below $atol = #tolstr(first-ok(rr(2, "chebtt_adaptive").filter(r => "interface" in r)).tol)$ at $d = 2$,
ChebTT-A treats the steep ridge as a front and models it as a profile $h(phi)$ (@sec:adaptive); its
remainder is then constant, and its cost stays near #num(last-ok(rr(2, "chebtt_adaptive")).evals)
evaluations.

How fast the degree grows can be predicted. The ridge is $tanh$ of a linear function,
and $tanh$ has poles at $plus.minus i pi slash 2$. In ChebTT-M's rotated box, the nearest pole fixes the
ellipse $E_rho$ in @thm:decay, so the degree grows by $ln 10 slash ln rho$ per decade of the tolerance.

#scene(side: left, width: 44%, panel("convergence-degree"))[
  ChebTT-M's degree along the ridge, #swatch(mcol("chebtt_metric")) measured, against the slope
  $ln 10 slash ln rho$ #swatch(grey, shape: "dashed") predicted from the pole. The figure script
  computes $rho$ from the model's own rotation and box, and stops unless the rotation's first axis is the
  ridge's direction.

  #table(
    columns: 4,
    inset: (x: 5pt, y: 3pt),
    align: (center, right, right, right),
    table.header[$d$][$rho$][predicted][measured],
    ..cg.dims.map(d => {
      let r = cg.ridge_degree.at(str(d))
      ([#d], [#r.rho], [#r.predicted], [#r.measured])
    }).flatten(),
  )
  #small[Degrees per decade of $atol$; measured: the least-squares slope over the rungs.]
]

The box grows along the ridge with $d$, since $q_d$ is oblique to more axes. The pole then lies closer to
the interval in the box's coordinates, $rho$ falls, and each decade costs more degree. The prediction is a
lower bound on the decay rate. The coefficients fall at least like $rho^(-n)$, and the agreement shows that
they fall no faster.

AFINNI and ART need more evaluations at every tolerance on the ridge, but fewer than the leaf-count law
predicts for a function that curves in every direction. The ridge curves in one direction only, so a
partition into slabs across it needs $atol^(-1 slash 2)$ of them in any dimension. ART's oblique planes
find such slabs. Its fitted exponent is #cg.ridge_rates.at("2").art at $d = 2$,
#cg.ridge_rates.at("3").art at $d = 3$ and #{ let r = cg.ridge_rates.at("5").art; if r == none [--] else [#r] } at $d = 5$. AFINNI's is
#cg.ridge_rates.at("2").afinni, #cg.ridge_rates.at("3").afinni and #cg.ridge_rates.at("5").afinni.

== A jump: an algebraic rate

#figure(panel("convergence-feature"), caption: [The feature example $F$ on $[-1, 1]^2$. Left: evaluations;
  right: the worst error on the benchmark's grid, in units of the tolerance. Colours as in
  @fig:convergence-rosen.]) <fig:convergence-feature>

On $F$, ChebTT and ChebTT-M fail at the loosest tolerance, #tolstr(cg.F.chebtt.first().tol), as in
@sec:compare. ChebTT-A holds $F$ down to #tolstr(last-ok(cg.F.chebtt_adaptive).tol), with
#num(last-ok(cg.F.chebtt_adaptive).evals) evaluations there, and reaches the rank limit at
#tolstr(failed-row(cg.F.chebtt_adaptive).tol). Its rate is algebraic. From
#tolstr(ok-rows(cg.F.chebtt_adaptive).at(1).tol) to #tolstr(last-ok(cg.F.chebtt_adaptive).tol) its degree
along $x_1$ grows from #ok-rows(cg.F.chebtt_adaptive).at(1).degrees.at(0) to
#last-ok(cg.F.chebtt_adaptive).degrees.at(0). @prop:remainder gives the reason. What the jump fields leave of the
jump adds a tail that falls like a power of $j$, and a power needs a degree that grows like a power of
$1 slash atol$.

AFINNI and ART cannot hold a jump. An affine model on a leaf that straddles it misses by at least
half the jump's height, whatever the leaf's size. In practice the straddling leaf's model follows one side,
and their worst error stays near the jump's full height, #ex.feature.jump.height, while the tolerance
shrinks. In units of the tolerance it grows like $1 slash atol$. At #tolstr(cg.F.art.last().tol), ART's worst is
#cg.F.art.last().worst $atol$ and AFINNI's #cg.F.afinni.last().worst $atol$.

== A vector function, a flow map and a stiff step map

#let cg-vec = cg.at("vector", default: (:))
#let reach(name, m) = {
  if name not in cg-vec or m not in cg-vec.at(name) { return [--] }
  let ok = ok-rows(cg-vec.at(name).at(m)).filter(r => r.at("worst", default: 0) <= 1)
  if ok.len() == 0 [--] else [#tolstr(ok.last().tol)]
}
#let reach-evals(name, m) = {
  if name not in cg-vec or m not in cg-vec.at(name) { return [--] }
  let ok = ok-rows(cg-vec.at(name).at(m)).filter(r => r.at("worst", default: 0) <= 1)
  if ok.len() == 0 [--] else [#num(ok.last().evals)]
}

// The run of a ladder with the smallest worst error on the test set (in the problem's units, not atol's).
#let vbest(name, m) = {
  let ok = ok-rows(cg-vec.at(name).at(m))
  if ok.len() == 0 { none } else { ok.sorted(key: r => r.worst * r.tol).first() }
}
#let vabs(name, m) = sci(vbest(name, m).worst * vbest(name, m).tol)
#let vev(name, m) = num(vbest(name, m).evals)
#let vlast(name, m) = last-ok(cg-vec.at(name).at(m))
#let vfirst(name, m) = first-ok(cg-vec.at(name).at(m))
#let vfail(name, m) = cg-vec.at(name).at(m).find(r => "failed" in r)
// AFINNI's and ART's worst errors over every run on a problem, in units of atol.
#let vworst(name) = ("afinni", "art").map(m => cg-vec.at(name).at(m).map(r => r.worst)).flatten()
// How far the better of AFINNI and ART stays from ChebTT on a problem: decades of worst error, and the
// ratio of evaluations (the cheaper of the two against ChebTT's).
#let vgap(name) = {
  let abs(m) = vbest(name, m).worst * vbest(name, m).tol
  let decades = calc.round(calc.log(calc.min(abs("afinni"), abs("art")) / abs("chebtt"), base: 10))
  let times = calc.floor(calc.min(vbest(name, "afinni").evals, vbest(name, "art").evals) / vbest(name, "chebtt").evals)
  (decades: decades, times: times)
}
#let none-or(n, word, plural) = if n == 0 [no #word] else [#n #plural]

#figure(panel("convergence-vector"), caption: [Evaluations against $1 slash atol$ on three vector-valued
  problems: `shared_latent` and `lorenz96_state` at $d = 5$ (the benchmark's function suite, every output
  divided by its largest $|f_c|$), and Robertson's step map, the increment over one step in log abundances
  (dex), on the box of the states its training trajectories visit. Colours as in @fig:convergence-rosen. AFINNI
  and ART on the first two are the other companion's runs.]) <fig:convergence-vector>

Three vector-valued problems test the schemes where the library is meant to be used (@sec:problems defines
and draws them). `shared_latent` is the
vector form of the ridge, six functions of two oblique coordinates, at $d = 5$. `lorenz96_state` is the
Lorenz-96 flow over $t = 0.1$ in all five components, the manuscript's ODE. Robertson's step map is a stiff
chemistry step, the increment over one step in log abundances (dex), on the box of the states its training
trajectories visit (@sec:steps); every evaluation is a Radau solve, so there the evaluations are the cost.

A vector-valued $f$ is held when every output is. ChebTT and ChebTT-A build one train per output on one cache
of $f$, so a point any output asks for serves the others; ChebTT-M groups the outputs that share coordinates
and turns each group once (@sec:metric). AFINNI and ART accept a leaf when all outputs pass.

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  inset: (x: 6pt, y: 4pt),
  align: (left, right, right, right),
  table.header[Tightest tolerance held (evaluations there)][`shared_latent`][`lorenz96_state`][Robertson],
  ..("chebtt", "chebtt_metric", "chebtt_adaptive", "hub", "afinni", "art").map(m => (mname.at(m),
    ..("shared_latent", "lorenz96_state", "robertson").map(n => {
      let t = reach(n, m)
      if t == [--] [--] else [#t (#reach-evals(n, m))]
    }))).flatten(),
)
#v(-0.4em)
#small[The tightest tolerance of each ladder at which the worst error on the test set is within it. AFINNI's
  and ART's ladders halve and stop where a run would take about two minutes; the schemes' go down by decades
  and stop at a failed build. A dash means that no tolerance held.]

Holding the tolerance is a strict test. AFINNI and ART fail it on the first two problems at every rung, and their
worst errors there are #calc.round(calc.min(..vworst("shared_latent"), ..vworst("lorenz96_state")), digits: 1) to
#calc.round(calc.max(..vworst("shared_latent"), ..vworst("lorenz96_state")), digits: 1) times the tolerance,
on test points their leaves were never checked at. For a fixed budget the better measure is the smallest worst
error each method reached and what it cost:

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  inset: (x: 6pt, y: 4pt),
  align: (left, right, right, right),
  table.header[Smallest worst error \ (evaluations there)][`shared_latent`][`lorenz96_state`][Robertson (dex)],
  ..("chebtt", "chebtt_metric", "chebtt_adaptive", "hub", "afinni", "art").map(m => (mname.at(m),
    ..("shared_latent", "lorenz96_state", "robertson").map(n => if vbest(n, m) == none [--] else [#vabs(n, m) (#vev(n, m))]))).flatten(),
)
#v(-0.4em)
#small[The worst error on the test set, over every point and output, at the run of each ladder where it is
  smallest. For AFINNI it is taken over the points it does not flag as outside its trust regions.]

*The two flows.* On `lorenz96_state` the three ChebTT variants are one model. ChebTT-M turns
#none-or(vlast("lorenz96_state", "chebtt_metric").rotated_outputs, "outputs", "outputs") and ChebTT-A adds
#none-or(vlast("lorenz96_state", "chebtt_adaptive").interfaces, "interface", "interfaces"). They hold
#tolstr(vlast("lorenz96_state", "chebtt").tol) with #vev("lorenz96_state", "chebtt") evaluations; at the next
decade the build stops on its evaluation budget, after #num(vfail("lorenz96_state", "chebtt").evals). ART's
smallest worst error is #vabs("lorenz96_state", "art") after #vev("lorenz96_state", "art") evaluations and
AFINNI's #vabs("lorenz96_state", "afinni") after #vev("lorenz96_state", "afinni"). The better of the two is about
#vgap("lorenz96_state").decades decades less accurate for more than #vgap("lorenz96_state").times times the
evaluations. The flow over $t = 0.1$ is analytic in the initial state, so
a decade costs the schemes a constant number of degrees (@prop:cost), while by the other companion's leaf-count
law it multiplies the leaves by $10^(d slash 2)$. Robertson's step map widens the gap. The three variants
are again one model, with ranks at most #vlast("robertson", "chebtt").max_rank, and hold
#tolstr(vlast("robertson", "chebtt").tol) dex with #vev("robertson", "chebtt") evaluations. ART reaches
#vabs("robertson", "art") dex with #vev("robertson", "art") and AFINNI #vabs("robertson", "afinni") with
#vev("robertson", "afinni"). Each evaluation there is a Radau solve, so the evaluations are the time.

*The shared coordinates.* `shared_latent` is the problem where the rotation helps. ChebTT-M turns
#vlast("shared_latent", "chebtt_metric").rotated_outputs of the #cg-vec.shared_latent.outputs outputs and holds
every decade down to #tolstr(vlast("shared_latent", "chebtt_metric").tol), with
#vev("shared_latent", "chebtt_metric") evaluations there and largest rank
#vlast("shared_latent", "chebtt_metric").max_rank. On the original axes the two oblique coordinates couple every
pair. ChebTT's rank reaches #vlast("shared_latent", "chebtt").max_rank at
#tolstr(vlast("shared_latent", "chebtt").tol), after #vev("shared_latent", "chebtt") evaluations, and the next
decade would exceed the evaluation budget; ChebTT-A, with its smaller budget, stops after
#tolstr(vlast("shared_latent", "chebtt_adaptive").tol). The hub expansion is the cheapest at loose tolerances,
#vabs("shared_latent", "hub") with #vev("shared_latent", "hub") evaluations, and does not improve beyond it
(@prop:floor). ART reaches #vabs("shared_latent", "art") with #vev("shared_latent", "art") evaluations; AFINNI's
sample grown with its leaves would exceed the adaptive study's largest after its first
#cg-vec.shared_latent.afinni.len() tolerances.

*The hub expansion's floor* is the same on all three. Its worst error stays near #vabs("lorenz96_state", "hub") on
`lorenz96_state` and #vabs("robertson", "hub") dex on Robertson whatever the tolerance asked for. Neither problem
is a function of a few directions.

== The hub expansion's floor

#scene(side: right, width: 44%, panel("convergence-floor"))[
  The hub expansion's worst error, in absolute terms, against $1 slash atol$:
  #swatch(mcol("hub"), shape: "line") on Rosenbrock at $d = 3$ and (dashed) $d = 5$, and
  #swatch(ridge, shape: "line") on `shared_latent` and (dashed) the lifted ridge, both at $d = #ex.hub.problem.d$. The
  grey line is the tolerance itself. Above it the model misses.
]

The hub expansion fits a fixed class of functions by least squares (@def:hub). Its error falls with the
degree until the class itself is the limit, then stops.

#proposition(title: [the floor])[
  Let $cal(H)$ be the functions of the form @eq:hub, for the chosen rotation, hub dimension and every degree
  up to the cap. The hub's worst error is at least $op("dist")_infinity (f, cal(H))$, whatever its budget. It
  is zero only if $f$ lies in $cal(H)$.
] <prop:floor>

#proof[
  Every model the hub can fit lies in $cal(H)$, and no member of $cal(H)$ is nearer to $f$ in the max norm than
  $op("dist")_infinity (f, cal(H))$.
]

On Rosenbrock the hub is exact at $d = 2$, where its two hub coordinates are all of $x$. At $d = 3$ and $5$
its terms leave out couplings of the chain, and its worst error stops at
#sci(last-ok(cg.floor.at("rosen-3")).worst * last-ok(cg.floor.at("rosen-3")).tol) and
#sci(last-ok(cg.floor.at("rosen-5")).worst * last-ok(cg.floor.at("rosen-5")).tol). On the ridge the limit is
the degree cap. One hub coordinate at degree at most #ex.hub.settings.degrees.at("1").last() stops near
#sci(last-ok(rr(2, "hub")).worst * last-ok(rr(2, "hub")).tol) at $d = 2$, where ChebTT-M needs a higher degree.
The hub reports the miss on its hold-out set, so the caller sees a floor (@sec:hub).

#block(breakable: false, table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 6pt, y: 4pt),
  align: (left, left, left),
  table.header[Method][Predicted][Measured here],
  [AFINNI, ART], [$N_c tilde atol^(-d slash 2)$; a ridge: $atol^(-1 slash 2)$; a jump: no convergence], [exponents #ca-rate("art", 2)--#ca-rate("art", 5) (ART, $d = 2$--$5$); on the ridge #(cg.ridge_rates.at("2").art) to #{ let r = cg.ridge_rates.at("5").art; if r == none [--] else [#r] }],
  [ChebTT], [a polynomial: constant; analytic with bounded ranks: $d log(1 slash atol)$; ranks up to $(log 1 slash atol)^(m)$], [constant on Rosenbrock; on the ridge, ranks #first-ok(rr(3, "chebtt")).ranks.at(0) to #last-ok(rr(3, "chebtt")).ranks.at(0) at $d = 3$],
  [ChebTT-M], [a ridge: rank 1, $ln 10 slash ln rho$ degrees per decade], [#cg.ridge_degree.at("2").measured, #cg.ridge_degree.at("3").measured, #cg.ridge_degree.at("5").measured against #cg.ridge_degree.at("2").predicted, #cg.ridge_degree.at("3").predicted, #cg.ridge_degree.at("5").predicted],
  [ChebTT-A], [a jump: algebraic, by @prop:remainder], [$F$ to #tolstr(last-ok(cg.F.chebtt_adaptive).tol)],
  [Hub expansion], [geometric in its degree, down to the floor of @prop:floor], [floors on Rosenbrock ($d >= 3$) and the ridge],
))

#pitfall(title: [What ends a ladder])[
  Every scheme has caps: ChebTT's rank (#ex.appendix.settings.chebtt.find(s => s.name == "rank_max").default), degree and evaluation budget, the hub's
  degree. The predictions hold until a cap is reached; a failed build says which one. Rounding sets the last
  floor. Near $10^(-10)$ of $max |f|$ the tail estimate cannot tell coefficients from noise.
]

#takeaway[For smooth functions of low rank, a Chebyshev scheme's cost grows like $log(1 slash atol)$, and for
  a polynomial not at all, where affine pieces need $atol^(-d slash 2)$; the dimension and an oblique feature
  return as the exponent of the logarithm, the hub expansion stops at the distance of $f$ from its class, and a
  jump costs ChebTT-A an algebraic rate.]

= Stiff step maps: the chemistry the library is for <sec:steps>

#question[A stiff chemistry step is called by a simulation in every cell at every time step. Can a
  global approximant replace the solver, and what does chaining its answers do to its error?]

#let sp = ex.steps
#let rob = sp.robertson
#let stiff = sp.regimes.stiff

The library's end goal is surrogates of astrochemical reaction networks inside MHD simulations. A
surrogate is built from the states that tracers visit, and the simulation calls it instead of the stiff
solver. A _step map_ $Phi_(Delta t)$ takes a state $z$, the logarithms of the abundances, to the state
one step later. The surrogates learn the increment $Phi_(Delta t)(z) - z$, which is small and carries the
chemistry. They are trained on the states visited by training trajectories and judged on held-out ones.
ChebTT works on the box that bounds the visited states; SlowTable on the states themselves.

Two errors matter. The _one-step_ error is a surrogate's step from a true state against the solver's.
The _rollout_ error chains the surrogate from a trajectory's initial state, as a simulation would, and
compares the whole trajectory.

#lemma(title: [how errors add up along a rollout, @hairer1996])[
  Let a surrogate make a one-step error $e_k$ at step $k$, and let the true map's Jacobians be $J_k$.
  To first order the rollout error after $n$ steps is $sum_(k=1)^n J_(n) dots J_(k+1) e_k$. If the
  errors keep their sign and the dynamics neither damp nor grow them, it is about $n$ times the one-step
  error; errors of random sign grow like $sqrt(n)$.
] <lem:rollout>

#scene(side: left, width: 42%, panel("steps-robertson"))[
  Robertson's stiff network at the tolerance 0.01, rollouts of the test trajectories step by step.
  #swatch(mcol("chebtt"), shape: "line") ChebTT on the visited box: one-step error #rob.chebtt_box.one_step
  of the tolerance, but its errors keep their sign, and the rollout reaches #rob.chebtt_box.rollout.
  #swatch(red, shape: "line") ChebTT retrained until its rollouts along the training trajectories
  hold: one-step #rob.chebtt_roll.one_step, rollout #rob.chebtt_roll.rollout, for #num(rob.chebtt_roll.evals)
  solves against #num(rob.chebtt_box.evals). #swatch(mcol("slowtable"), shape: "line") SlowTable's one-step
  error is #rob.slowtable.one_step times the tolerance, in fast directions the next step damps, so its
  rollout is #rob.slowtable.rollout.
]

The retrained variant (`chebtt_roll`) chains the surrogate along the training trajectories, which are
known and so cost no solve, and trains again at a tighter one-step tolerance until those rollouts hold.
ChebTT's spectral convergence makes the tighter tolerance cheap, and the retraining reuses every cached
solve. Over the benchmark's stiff problems:

#block(breakable: false, above: 1em, below: 1em, {
  set text(8.5pt)
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    inset: (x: 5pt, y: 3.5pt),
    align: (left, right, right, right, right, right),
    table.header[stiff step maps][ran][one-step held][rollout held too][median solves][median speedup],
    ..stiff.map(r => ([#r.label#if r.stale [ $ast$]], [#r.ran], [#r.one_step], [#r.rollout], fmt-ev(r.evals),
      if r.speedup == none [--] else [#num(r.speedup)$times$])).flatten(),
  )
})

The speedup is the solver's time per state and step over the surrogate's inference time, at the cases
where both errors hold. The solver is SciPy's Radau, the integrator the system would be solved with. AFINNI and ART
hold no rollout. ChebTT on the box holds the one-step error on
#stiff.find(r => r.method == "chebtt_box").one_step of #stiff.find(r => r.method == "chebtt_box").ran cases, but its
rollouts hold on only #stiff.find(r => r.method == "chebtt_box").rollout. The retrained ChebTT holds both on
#stiff.find(r => r.method == "chebtt_roll").rollout of #stiff.find(r => r.method == "chebtt_roll").ran.

#scene(side: right, width: 40%, panel("steps-scatter"))[
  Every finished stiff record: one-step error against rollout error, both over the tolerance, for
  #swatch(mcol("chebtt")) ChebTT, #swatch(red) the retrained ChebTT, #swatch(mcol("slowtable")) SlowTable,
  #swatch(mcol("art_quad")) ART with quadratic leaves and #swatch(mcol("afinni")) AFINNI. The lower left
  quadrant holds both. A surrogate's rollout is usually several times its one-step error, which is why
  holding the one-step error is not enough.
]

== Towards the real networks

#let kd = sp.kida
The benchmark's systems have up to a few dozen states. The target is a network like KIDA @wakelam2012,
with #kd.species species and #num(kd.reactions) reactions. Measured with its own
tracer settings, one hydro step takes the solver about #kd.ms_per_step ms and #kd.lu_per_step sparse
factorisations. Over that step all but #kd.slow_low to #kd.slow_high of its modes relax completely. A
step map with a few dozen slow directions is the structure the hub expansion and ChebTT-M are built
for, steep in a few coordinates the data reveal and flat in the rest. On the 14-species Nelson network,
about #kd.nelson_slow of #kd.nelson_species species' directions stay slow.

#pitfall(title: [A box is not where a simulation goes])[
  The box that bounds the visited states is mostly empty in high dimension, and ChebTT samples all of it,
  including states that are not physical. On the #(kd.nelson_inputs)-input Nelson problem, ChebTT on the
  box did not train within the time cap. The surrogate must work in the few slow coordinates and be trained along the
  states the simulation visits. The rollout error needs one-step errors well below the tolerance unless
  errors cancel. The figures here are the one-step and rollout errors on held-out trajectories, not
  bounds.
]

#takeaway[On stiff step maps a ChebTT retrained until its rollouts hold keeps both errors within the
  tolerance on #stiff.find(r => r.method == "chebtt_roll").rollout of #stiff.find(r => r.method == "chebtt_roll").ran
  cases, at a median speedup of #num(stiff.find(r => r.method == "chebtt_roll").speedup)$times$ over the solver,
  where AFINNI and ART hold no rollout; the path to large networks is the few slow coordinates.]

= Tried and set aside <sec:aside>

#question[Four more schemes are in the library. What did each one teach, and why is each not among the
  four?]

#let asd = ex.aside
#let srow(cls, m) = st.classes.find(c => c.class == cls).rows.find(r => r.method == m)
#board(
  (panel("aside-chebtree"), [ChebTree: #asd.chebtree.boxes boxes]),
  (panel("aside-sparsecheb"), [SparseCheb: #num(asd.sparsecheb.evals) points]),
  (panel("aside-kernel"), [MetricKernel: #asd.kernel.centres centres]),
  (panel("aside-slowtable"), [SlowTable: #asd.slowtable.records records]),
  columns: 4,
)

*ChebTree* is an adaptive tree of tensor Chebyshev boxes. A box raises its degree where its
coefficients decay geometrically and splits where they do not. On $b + r$ it uses
#asd.chebtree.boxes boxes and #num(asd.chebtree.evals) evaluations, worst #asd.chebtree.worst. It holds
#srow("kinks, jumps, fronts", "chebtree").held of the
#srow("kinks, jumps, fronts", "chebtree").ran kink, jump and front cases it ran, at a median of #num(srow("kinks, jumps, fronts", "chebtree").evals)
evaluations. But a tensor box costs $product_k (n_k + 1)$ points, and the benchmark ran it to $d = #st.max_d.chebtree$
only.

*SparseCheb* is a dimension-adaptive sparse Chebyshev grid @gerstner2003. On $b + r$ it is the cheapest
scheme here, with #num(asd.sparsecheb.evals) evaluations over #asd.sparsecheb.indices indices and a worst
error of #asd.sparsecheb.worst. Its inference on the smooth cases it holds is #srow("smooth", "sparsecheb").us µs
per query, against #srow("smooth", "chebtt").us for ChebTT, and one global expansion pays for every kink
or jump. It was run to $d = #st.max_d.sparsecheb$.

*MetricKernel* interpolates with a Matérn kernel in a metric learned from the model's own gradients. On
$b + r$ its worst error is #asd.kernel.worst times the tolerance after #num(asd.kernel.evals) evaluations.
Random check batches miss the worst point, and a kernel interpolant is weakest at the boundary, where most
of a high-dimensional box's volume is. It holds #srow("smooth", "kernel").held of
#srow("smooth", "kernel").ran smooth cases, in two dimensions only.

*SlowTable* tabulates a step map over the states a simulation visits, after ISAT @pope1997. Each record
is a linear model inside an ellipsoid of accuracy, long along the directions the step forgets. On a
stiff two-state map at $atol = #asd.slowtable.alpha$ it keeps #asd.slowtable.records records for
#asd.slowtable.states visited states, and refuses #asd.slowtable.refused_fresh% of the states of new
trajectories, which the solver must answer. On the stiff benchmark it holds the one-step error on
#stiff.find(r => r.method == "slowtable").one_step of #stiff.find(r => r.method == "slowtable").ran
problems and the rollout on #stiff.find(r => r.method == "slowtable").rollout.

#takeaway[Tensor boxes stop at three dimensions, sparse grids at oblique features, kernels at the
  boundary and tables at what they have not seen; the tensor train and the hub are the global
  approximants that scale.]

= The engineering contract <sec:engineering>

#question[Four schemes, one library. What does every scheme promise its caller, and how is a run
  observed, distributed and saved?]

#implementation(title: [Settings and `build`])[
  Every scheme module has a frozen settings dataclass and a driver,
  `build(f, lo, hi, atol, settings=None, *, jacobian=None, stream_dir=None, **overrides)`. Overrides
  replace single settings (`chebtt.build(f, lo, hi, 1e-3, seed=1)`), and a model's settings are frozen
  with it. `f` takes host $(N, d)$ arrays and returns $(N,)$ values, or $(N, m)$ for a vector function.
  The model predicts on host arrays and counts its training evaluations in `evals`. @sec:settings
  lists every setting of the four.
]

#implementation(title: [The oracle and MPI])[
  `Oracle` (`src/schemes/oracle.py`) wraps $f$ and optionally its Jacobian, counts the points and the
  Jacobians it evaluated and the seconds spent in them, and does not deduplicate. A scheme that may ask
  for a point twice keeps its own store (ChebTT's lattice store, `PointCache` for vector trains). Under
  MPI, `with oracle.distributed(f) as o:` gives rank 0 an oracle that cuts every batch into one even
  chunk per rank, while the other ranks serve those chunks until rank 0 leaves the block. The values
  do not depend on the rank count, and a failure on any rank is raised again on rank 0, so no rank is
  left waiting in a collective. The schemes themselves run on rank 0, in NumPy.
]

#implementation(title: [Profiling: phases and rounds])[
  Every scheme times the same phase groups: `explore` (directions, edges, pilot gradients), `sample`,
  `fit`, `check`, `adapt` and `compress`, so schemes compare stage by stage. The oracle's time is counted
  separately. A _round_ is a scheme's unit of refinement: a cross sweep, a budget doubling, an output's
  train. Each round ends with one log line and one record in the model's `TrainingReport`. On $b + r$,
  ChebTT's report has #ex.engineering.chebtt.rounds rounds and uses
  #ex.engineering.chebtt.phases.len() phases. With `stream_dir` every phase and round is also streamed as
  JSON lines through `infra.profiling.Profiler`.
]

#implementation(title: [Saving and loading])[
  `base.save(model, path)` writes any model to one `.npz` with a JSON header (format version, model,
  report), and `base.load(path)` rebuilds it. The figure script saves and loads a ChebTT, a ChebTT-M, a
  ChebTT-A and a hub model, and stops unless each loaded model predicts bit for bit as trained.
]

#raw(block: true, lang: "python", ```
from src.schemes import base, chebtt, hub
from src.schemes.chebtt import adaptive

model = chebtt.build(f, lo, hi, atol, seed=1)                        # ChebTT
vector = chebtt.build_vector(g, lo, hi, atol, m, rotate=True)        # ChebTT-M, m outputs
jumpy = adaptive.build(f, lo, hi, atol)                              # ChebTT-A
steep = hub.build(g, lo, hi, atol, jacobian=jac)                     # the hub expansion
print(model.evals, model.report.summary())                           # evaluations, phases, rounds
base.save(model, "model.npz"); base.load("model.npz").predict(X)     # bit for bit as trained
```.text)

#takeaway[Every scheme is one call with frozen settings, trains on rank 0 while its oracle calls spread
  over the ranks, reports its phases and rounds, and saves to one file that predicts as trained.]

= How we know it is right <sec:verify>

#question[The schemes are global models with adaptive sampling, and their tolerances are judged at points.
  What stands between a change to a scheme and a wrong surrogate?]

#let ap = ex.appendix
#cards(
  columns: 3,
  ([Unit tests], [#ap.test_count tests in `tests/unit/test_schemes_*.py` check each scheme on small cases:
    ChebTT's edge search and pieces, ChebTT-M's groups, ChebTT-A's interfaces, the hub's groups and
    Jacobian, saving and loading, and the shared contract.]),
  ([Bit-identical regression], [`scripts/scheme_regression_check.py` trains #ap.regression_cases.len()
    small cases, one or two per scheme, and compares their evaluations and predictions bit for bit
    against a reference. A refactor must reproduce them; BLAS runs on one thread.]),
  ([Many ranks], [`tests/mpi` runs the distributed oracle under real `mpiexec` and checks that the values do
    not depend on the rank count and that a failure on one rank reaches rank 0.]),
  ([The figure script], [It records ChebTT's, ChebTT-A's and the hub's stages and checks that the
    recording changes nothing. It rebuilds the cross as a skeleton decomposition, the tail, the edge search's last
    step, the compression's budget, the metric groups, the one-sided jumps and the hub's terms, and stops
    unless each matches the library.]),
  ([The benchmark], [Every method is judged on a fixed test set by its worst error, and every record
    names the code that ran it, so a stale record is marked rather than mixed in.]),
  ([What is not checked], [The tolerance between test points, since the checks are estimates. No scheme here
    certifies its worst error, and in high dimension no box sampler can @novak2008.]),
)

#takeaway[Tests check each scheme on small cases, a regression check pins every scheme bit for bit, and
  the figures of this companion are rebuilt from definitions and compared with the library.]

#bibliography("references.yml", title: [References])

#show: appendix
#pagebreak(weak: true)

= The problems <sec:problems>

#let pj = json("figures/problems.json")
#let pfun(name) = pj.functions.find(r => r.name == name)
#let bound(x) = if calc.abs(x - calc.pi) < 1e-9 [$pi$] else [#x]
#let fbox(name) = [$[#bound(pfun(name).lo), #bound(pfun(name).hi)]^d$]
#let pex = pj.examples

This appendix defines every function the companion builds a model of. @sec:examples draws the two running
examples. The figures below draw the rest.

== The companion's own examples

#block(breakable: false, table(
  columns: (auto, 1fr, auto),
  inset: (x: 5pt, y: 4pt),
  align: left + horizon,
  table.header[Function][$f(x)$][Domain],
  [Rosenbrock's valley], [$(1 - x_1)^2 + (x_2 - x_1^2)^2$], [$[#pex.rosen_box.at(0).at(0), #pex.rosen_box.at(1).at(0)] times [#pex.rosen_box.at(0).at(1), #pex.rosen_box.at(1).at(1)]$],
  [The same along a chain (@sec:convergence)], [$sum_(i=1)^(d-1) (1 - x_i)^2 + (x_(i+1) - x_i^2)^2$], [$[#pex.rosen_box.at(0).at(0), #pex.rosen_box.at(1).at(0)] times [#pex.rosen_box.at(0).at(1), #pex.rosen_box.at(1).at(1)]^(d-1)$],
  [The feature example $F$], [$b + r + j$, the three parts of @sec:examples], [$[#pex.f_box.at(0).at(0), #pex.f_box.at(1).at(0)]^2$],
  [The ridge lifted to $d$], [$#ex.feature.ridge.height tanh(#ex.feature.ridge.steep (q_d dot x + #ex.feature.ridge.offset))$], [$[-1, 1]^d$],
  [The lifted pair (@sec:metric)], [the lifted ridge, and $#pex.second_output.at(0) exp(-#pex.second_output.at(1) (q_d dot x - #pex.second_output.at(2))^2)$], [$[-1, 1]^d$],
  [The lift with weak terms (@sec:hub)], [the lifted ridge $+ #pex.weak norm(x)^2 slash d$], [$[-1, 1]^d$],
))

The lifted direction $q_d$ is $(cos #ex.feature.ridge.angle°, sin #ex.feature.ridge.angle°, ..., sin #ex.feature.ridge.angle°)$
divided by its length, so $q_2 = q$. The lifted pair runs at $d = #pex.lift_dims.map(str).join(", ")$, and the
lift with weak terms at $d = #pex.hub_d$.

== The benchmark's functions

The benchmark runs every function of `src.problems` on the box $[l, u]^d$ at
$d = #pj.dims.map(str).join(", ")$, with seeds #pj.seeds.map(str).join(", "). A method that fails a function's
loosest tolerance at one $d$ is not run at larger $d$. Its tolerance is relative.
A model holds when its worst error on the test set is at most $"tol" dot max|f|$ there. The test set is a
$#pj.test.grid times #pj.test.grid$ grid at $d = 2$ and #num(pj.test.points) uniform points from one fixed seed
above. Each output of a vector function is divided by its largest $|f_c|$ on the test set, so one
tolerance is the same relative accuracy for every output. The tolerances depend on $d$:

#block(breakable: false, table(
  columns: (auto,) * pj.dims.len() + (auto,),
  inset: (x: 6pt, y: 3pt),
  align: center + horizon,
  table.header([$d$], ..pj.dims.map(d => [#d])),
  [tol], ..pj.dims.map(d => [#pj.tolerances.at(str(d)).map(str).join(", ")]),
))

#figure(panel("problems-functions"), caption: [Every benchmark function at $d = 2$ on its box, one row per
  class. A vector function shows its first output. The bands hold equal shares of the box, light where $f$
  is small, so functions of any range show their shape. Where one value covers more of the box than a band,
  as the zero outside `genz_discontinuous`'s corner does, it gets one band of its own.]) <fig:problems-functions>

The random parameters are drawn once per $d$ from a fixed seed, so they are part of the function. $Q$ is the
orthogonal factor of a $d times d$ standard normal matrix with its sign set to make $det Q = +1$, and $q$ a
standard normal vector divided by its length. The Genz functions draw $tilde(a)$ and then $w$ uniformly in
$[0, 1]^d$, and set $a = b tilde(a) slash sum_i tilde(a)_i$. The difficulty $b$ is
#pj.genz.pairs().map(((k, v)) => [#v for #raw(k)]).join(", "), and $sum_i a_i = 4.3$ for the discontinuous one.

#let ftable(cls, rows) = block(breakable: false, above: 0.8em, {
  set text(8.5pt)
  table(
    columns: (auto, 1fr, auto),
    inset: (x: 4pt, y: 3pt),
    align: left + horizon,
    table.header(table.cell(colspan: 3)[*#cls*]),
    ..rows.map(((name, f)) => (raw(name), f, fbox(name))).flatten(),
  )
})

#ftable([Smooth], (
  ("rosenbrock", [$sum_(i=1)^(d-1) (1 - x_i)^2 + (x_(i+1) - x_i^2)^2$]),
  ("gaussian", [$1 + exp(-1/2 sum_i i^3 (Q^top x)_i^2)$]),
  ("wave", [$cos(2 q^top x) + 1.5$]),
  ("genz_oscillatory", [$cos(2 pi w_1 + a^top x)$]),
  ("genz_product_peak", [$product_i (a_i^(-2) + (x_i - w_i)^2)^(-1)$]),
  ("genz_corner_peak", [$(1 + a^top x)^(-(d + 1))$]),
  ("genz_gaussian", [$exp(-sum_i a_i^2 (x_i - w_i)^2)$]),
))
#ftable([Many local features], (
  ("ackley", [$-20 exp(-0.2 sqrt(norm(x)^2 slash d)) - exp(1/d sum_i cos 2 pi x_i) + 20 + e$]),
  ("rastrigin", [$10 d + sum_i (x_i^2 - 10 cos 2 pi x_i)$]),
  ("griewank", [$1 + norm(x)^2 slash 4000 - product_i cos(x_i slash sqrt(i))$]),
  ("styblinski_tang", [$1/2 sum_i (x_i^4 - 16 x_i^2 + 5 x_i)$]),
  ("michalewicz", [$-sum_i sin x_i sin^(20)(i x_i^2 slash pi)$]),
))
#ftable([Kinks, jumps and fronts], (
  ("genz_continuous", [$exp(-sum_i a_i |x_i - w_i|)$]),
  ("genz_discontinuous", [$exp(a^top x)$ where $x_1 < w_1$ and $x_2 < w_2$, and 0 elsewhere]),
  ("oblique_jump", [$sin 2 s + [s > 0.3]$ with $s = q^top x$]),
  ("ball_jump", [$exp(-r^2) [r^2 < 0.64]$ with $r^2 = sum_i (x_i - 0.2)^2$]),
  ("front", [$tanh((norm(x) - 1) slash 0.05)$]),
))
#ftable([ODE flow maps], (
  ("lorenz96", [$x_1(0.1)$ of $dot(x)_i = (x_(i+1) - x_(i-2)) x_(i-1) - x_i + 1$, indices cyclic, $x(0) = x$]),
  ("lorenz96_chaotic", [$x_1(0.5)$ of the same flow with the forcing 8 in place of 1]),
  ("lotka_volterra", [$y_1(2)$ of $dot(y)_i = y_i (r_i - sum_j A_(i j) y_j)$, $y(0) = x$, with $r_i$ uniform in $[0.5, 1.5]$, $A_(i j)$ uniform in $[0, 0.8]$ and $A_(i i) = 1$]),
))
#ftable([Vector-valued], (
  ("rosenbrock_gradient", [the gradient of `rosenbrock`, $d$ outputs]),
  ("genz_triple", [`genz_oscillatory`, `genz_product_peak` and `genz_corner_peak`]),
  ("lotka_volterra_state", [the whole state $y(2)$ of `lotka_volterra`, $d$ outputs]),
  ("swirl", [each pair $(x_(2k-1), x_(2k))$ turned by $theta = 2 exp(-norm(x)^2)$; for odd $d$ the last output is $x_d cos theta$]),
  ("wave_pair", [$(cos 1.5 s, sin 1.5 s)$ with $s = q^top x$]),
  ("shared_latent", [six functions of $g_1 = 1.5 s_1 + 0.6 s_2^2$ and $g_2 = 1.5 s_2$, with $s_k = q_k^top x$
    and $q_1, q_2$ orthonormal: $tanh 2 g_1$, $e^(-(g_1 - 0.5)^2) cos 2 g_2$, $sin(g_1 + g_2)$, $g_1 g_2 slash 3$,
    $1 slash (1 + (g_1 + 1)^2 + g_2^2)$ and $cos(3 g_2 - g_1)$]),
  ("lorenz96_state", [the whole state $x(0.1)$ of `lorenz96`, $d$ outputs]),
))

The flow maps integrate each point on its own: Lorenz-96 with the library's Dormand-Prince 5(4) pair at
$r_"tol" = a_"tol" = 10^(-6)$, Lotka-Volterra with 200 fixed steps of the classical Runge-Kutta method. Three
cases on the grid are degenerate. Lorenz-96 needs four sites, and below four its indices coincide. At $d = 3$ the
site $i + 1$ is the site $i - 2$, and
the flow is the linear decay $dot(x)_i = 1 - x_i$. At $d = #pj.degenerate.d$ no test point falls in
`ball_jump`'s ball, so the function is #pj.degenerate.ball_jump on the whole test set, and so is every
tolerance. At $d = #pj.degenerate.d$ the largest value of `genz_product_peak` on the test set is
#sci(pj.degenerate.genz_product_peak).

== The step maps

A step map takes a state $z$ to the state one step $Delta t$ later, and a model learns the increment
$Phi_(Delta t)(z) - z$. In a system marked "dex" the state is the base-10 logarithm of every abundance, and
errors and tolerances are in dex. The others are in their own units. The oracle is one SciPy `solve_ivp` call
per state. Each system integrates training and test trajectories from random initial states. The box is the
bounding box of the states that start a step in the training trajectories, padded by 2% of each side. A
model is scored on the states of the test trajectories, by its worst error over the outputs, against the
tolerances #pj.step_tolerances.map(str).join(" and "). A family runs at several sizes, named by $d$.

#let sfam = pj.step_maps.map(r => r.family).dedup()
#block(breakable: false, {
  set text(8.5pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    inset: (x: 4pt, y: 3pt),
    align: (left, left, left, left, left, right, right, right),
    table.header[Family][$d$][regime][state][integrator][$Delta t$][steps][training, test],
    ..sfam.map(fam => {
      let rows = pj.step_maps.filter(r => r.family == fam)
      let r = rows.first()
      (raw(fam), [#rows.map(r => str(r.d)).join(", ")], [#r.regime], [#if r.log [dex] else [linear]],
       [#r.method], [#if r.dt >= 1e5 { sci(r.dt) } else { str(r.dt) }], [#r.steps], [#r.n_train, #r.n_test])
    }).flatten(),
  )
  v(-0.4em)
  small[$Delta t$ is in each system's own unit of time: seconds for `primordial` and `nelson`, minutes for
    `chemakzo` and `pollu`. "Steps" is the length of each trajectory.]
})

#block(breakable: false, table(
  columns: (auto, 1fr),
  inset: (x: 5pt, y: 4pt),
  align: left + horizon,
  table.header[System][Equations or source],
  [`robertson`], [$dot(y)_1 = -0.04 y_1 + 10^4 y_2 y_3$, $dot(y)_2 = 0.04 y_1 - 10^4 y_2 y_3 - 3 dot 10^7 y_2^2$,
    $dot(y)_3 = 3 dot 10^7 y_2^2$; $y_1$ uniform in $[0.5, 1]$, $y_2 = 10^u$ with $u$ uniform in $[-12, -6]$,
    $y_3 = 1 - y_1 - y_2$],
  [`van_der_pol`], [$dot(y)_1 = y_2$, $dot(y)_2 = ((1 - y_1^2) y_2 - y_1) slash epsilon$ with $epsilon = 10^(-3)$],
  [`oregonator`], [$dot(y)_1 = 77.27 (y_2 + y_1 (1 - 8.375 dot 10^(-6) y_1 - y_2))$,
    $dot(y)_2 = (y_3 - (1 + y_1) y_2) slash 77.27$, $dot(y)_3 = 0.161 (y_1 - y_3)$],
  [`hires`, `pollu`, `chemakzo`], [HIRES, POLLU and Chemical Akzo Nobel from the test set for IVP solvers
    @mazzia2008, the last reduced from its index-1 DAE to five equations],
  [`primordial`], [primordial hydrogen and helium chemistry: nine species and the temperature, 22 reactions
    with the rates of the Grackle library],
  [`nelson`], [the reduced carbon and oxygen network of Nelson and Langer (ApJ 524, 923, 1999) as the KIDApy
    package writes it, 14 species and four environment inputs held along a path],
  [`bruss`], [the Brusselator on $n = d slash 2$ cells, $dot(u) = 1 + u^2 v - 2.9 u + 0.5 L u$,
    $dot(v) = 1.9 u - u^2 v + 0.5 L v$, $L$ the Neumann Laplacian],
  [`allen_cahn`], [$dot(u) = L u + (u - u^3) slash 0.05$ on $d$ cells],
  [`fisher_kpp`], [$dot(u) = L u + 10 u (1 - u)$ on $d$ cells],
  [`lorenz96`], [the Lorenz-96 flow above, from initial states uniform in $[0, 2]^d$],
  [`burgers`], [viscous Burgers on $d$ periodic cells, $dot(u)_i = -(u_(i+1)^2 - u_(i-1)^2) slash (4 h) + 0.1 (u_(i+1) - 2 u_i + u_(i-1)) slash h^2$, $h = 1 slash d$],
))

#figure(panel("problems-robertson"), caption: [Robertson's step map: the increment over one step of each
  log abundance, over the plane of $log_10 y_2$ and $log_10 y_3$ with $log_10 y_1 = #pj.robertson.fixed_at$,
  the centre of the box. The box is the one the convergence study uses (@sec:convergence), and the grey dots
  are the #num(pj.robertson.visited) states its training trajectories visit, projected onto the plane.
  Bands as in @fig:problems-functions.]) <fig:problems-robertson>

The scattered dots are the initial states, with $log_10 y_2$ between $#pj.robertson.start.at(0)$ and
$#pj.robertson.start.at(1)$. After the first step every state has $log_10 y_2$ between
$#pj.robertson.later.at(0)$ and $#pj.robertson.later.at(1)$, and lies near one curve. Most of the box is far from
every state a trajectory visits.

#pagebreak(weak: true)

= The settings of the four schemes <sec:settings>

The figure script reads each table from the settings dataclass, with the comment beside each field
in the source. The defaults are the library's.

#let settings-table(rows, title) = block(above: 0.8em, below: 1em, {
  set text(8pt)
  table(
    columns: (auto, auto, 1fr),
    inset: (x: 4pt, y: 3pt),
    align: (left, right, left),
    table.header[#title][default][meaning],
    ..rows.map(r => (raw(r.name), raw(r.default), [#r.note])).flatten(),
  )
})

#settings-table(ap.settings.chebtt, [`ChebTTSettings`])
#settings-table(ap.settings.vector, [`VectorSettings`, besides ChebTT's])
#settings-table(ap.settings.adaptive, [`AdaptiveSettings`])
#settings-table(ap.settings.hub, [`HubSettings`])

= From the companion's symbols to the code's names <sec:names>

#block(above: 1em, below: 1em, {
  set text(8.5pt)
  table(
    columns: (auto, 1fr, 1fr),
    inset: (x: 5pt, y: 3.5pt),
    align: left,
    table.header[symbol][meaning][in the code],
    [$f$, $Omega$, $ell$, $u$], [the oracle and its box], [`f`, `lo`, `hi`],
    [$atol$, $s$], [the tolerance and the safety factor], [`atol`, `safety`],
    [$tau$], [an axis's share of the tolerance], [`share` $= s atol slash d$ in `build`],
    [$n_k$, $r_k$], [degree of axis $k$, TT rank], [`degrees`, `ranks`; `grid.n`],
    [$C_k$], [core $k$: coefficients $r_(k-1) times (n_k + 1) times r_k$], [`ChebTT.cores[k]`],
    [$I_k$, $J_k$], [left and right index sets of the cross], [`left`, `right` in `_sweeps`],
    [$L_k$, $R_(k+1)$], [the train's products left and right of core $k$], [`environments`, `through`],
    [$beta$, $Z$], [the compression's budget and probe], [`chop_share * atol`, `probe`],
    [$G_c$], [pullback metric of output $c$], [`metric_groups`],
    [$r_"cap"$], [ChebTT-M's rank cap on the axes], [`axes_rank`],
    [$phi$, $rho$, foot], [interface, signed distance, nearest point], [`Interface.phi`, `geometry`],
    [$S$, $h$, $J_k$], [interface term, profile, jump fields], [`Interface.term`, `profile`, `fields`],
    [$V$, $r$, $p$], [hub rotation, hub dimension, hub degree], [`HubGroup.V`, `r`, `degree`],
    [$C$], [gradient covariance], [`output_groups`, `covariance_rank`],
    [$P_0$, $P_(1 j)$, $P_(2 j)$], [hub block and weak terms], [`_columns`, `HubGroup.coef`],
  )
})

= The convergence runs <sec:convergence-runs>

This appendix lists every run of @sec:convergence, one table per problem. Each cell is the evaluations of $f$
and, after the slash, the worst error on the test set in units of $atol$. "fails" marks a build that failed, and
a dash marks a tolerance that was not run. A method whose build fails is not run at tighter tolerances. The
schemes' ladders go down by decades, AFINNI's and ART's by halvings, so most rows hold one family or the other.
AFINNI's runs are those from the sample grown with its leaves. On Rosenbrock, `shared_latent` and
`lorenz96_state`, AFINNI's and ART's runs are the other companion's.

#let conv-methods = ("chebtt", "chebtt_metric", "chebtt_adaptive", "hub", "afinni", "art")
#let conv-cell(rows, tol) = {
  if rows == none { return [--] }
  let r = rows.find(r => calc.abs(r.tol - tol) <= 1e-9 * tol)
  if r == none { [--] } else if "failed" in r { [fails] } else {
    let w = r.worst
    [#num(r.evals) / #if w > 0 and (w < 0.01 or w >= 1000) { sci(w) } else { str(w) }]
  }
}
#let conv-table(title, by-method) = {
  let tols = by-method.values().filter(rows => rows != none).map(rows => rows.map(r => r.tol)).flatten().dedup().sorted().rev()
  block(breakable: false, above: 1em)[
    #text(weight: "bold", title)
    #set text(7.5pt)
    #table(
      columns: (auto,) + (1fr,) * 6,
      inset: (x: 3pt, y: 2.5pt),
      align: (left,) + (right,) * 6,
      table.header([$atol$], ..conv-methods.map(m => mname.at(m))),
      ..tols.map(t => ([#tolstr(t)], ..conv-methods.map(m => conv-cell(by-method.at(m, default: none), t)))).flatten(),
    )
  ]
}
#let ca-probs = ca.at("problems", default: (:))

#for d in cg.dims.map(str) {
  let row = cg.rosen.at(d)
  conv-table([Rosenbrock's valley along a chain, $d = #d$],
    (..row, afinni: ca.runs.afinni_scaled.at(d, default: none), art: ca.runs.art.at(d, default: none)))
}
#for d in cg.dims.map(str) {
  conv-table([The ridge, $d = #d$], cg.ridge.at(d))
}
#conv-table([The feature example $F$, $d = 2$ (AFINNI and ART with the settings of @sec:compare)], cg.F)
#for name in ("shared_latent", "lorenz96_state", "robertson") {
  conv-table([#raw(name)], conv-methods.map(m => (m, cg-vec.at(name).at(m))).to-dict())
}

= Proof details <sec:proofs>

*The top quarter (@prop:quarter).* With $|a_j| = C rho^(-j)$, the sum over $j$ from $m$ to $n$ is
$C rho^(-m) (1 - rho^(-(n - m + 1))) slash (1 - rho^(-1)) approx |a_m| slash (1 - rho^(-1))$. For the upper
half $m = n slash 2$, and $|a_(n slash 2)| = C rho^(-n slash 2) = sqrt(C |a_n|)$. For the top quarter
$m = 3 n slash 4$, and $|a_(3 n slash 4)| = C^(1 slash 4) |a_n|^(3 slash 4)$. Setting either to the share
$tau$ and solving for $|a_n|$ gives the two accuracies of the proposition.

*The cross in $d$ dimensions (@lem:skeleton).* The TT-cross keeps nested index sets: every left set
$I_k$ extends some multi-indices of $I_(k-1)$ by one position on axis $k$, and every right set $J_k$
those of $J_(k+1)$. For nested sets the train built from the fibres
$I_(k-1) times {"axis" k} times J_k$ interpolates $f$ on every one of those fibres @oseledets2010, which
is @lem:skeleton applied to each unfolding of the tensor in turn. The figure script checks the
two-dimensional case on every recorded model.

*The interface term (@prop:remainder).* Write $f = f_+$ on the side $rho > 0$ and $f = f_-$ on the other,
both smooth. Jump fields of order $k$ give $S = [rho > 0] sum_(i <= k) J_i rho^i slash i!$ with
$J_i = partial_rho^i (f_+ - f_-)$ on the interface. Then on the side $rho > 0$,
$f - S = f_+ - sum_(i <= k) partial_rho^i (f_+ - f_-) rho^i slash i!$, whose first $k$ normal
derivatives at $rho = 0$ equal those of $f_-$. So $f - S$ and its first $k$ derivatives are continuous
across the interface.
