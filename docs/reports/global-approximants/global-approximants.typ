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
    examples follow every idea: Rosenbrock's valley, as in the other companion, and a function built to
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
This companion asks for the strictest kind of agreement: the _worst_ error over the box must be within
a tolerance, $max_(x in Omega) |f(x) - fh(x)| <= atol$. A small average error is not enough. A
simulation that calls the surrogate a billion times will find the worst points.

Three costs decide between surrogates. The evaluations of $f$ that training spends. The time of one
answer, since a surrogate is trained once and asked millions of times. And whether the worst error
holds at all. The library's benchmark judges every method by all three, on a fixed test set of each
problem.

== Two families

#board(
  (other("afinni-trust-all"), [*AFINNI* fits affine pieces to clusters of a given sample.]),
  (other("art-partition"), [*ART* cuts the box into polytopes, one affine piece each.]),
  (panel("rosenbrock-points"), [*ChebTT* samples a few lines of a grid and fits one model.]),
  columns: 3,
)

The library builds surrogates in two ways. The _adaptive surrogate_ divides the box into many small
regions and fits a local affine model to each. AFINNI groups a sample into clusters, ART cuts the box
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
  $x_1$ factor has a kink at $x_1 = #ex.feature.kink$: an edge that is aligned with an axis, and slows
  every Chebyshev series that crosses it.
- *The ridge* $r(x) = #ex.feature.ridge.height tanh(#ex.feature.ridge.steep (q dot x + #ex.feature.ridge.offset))$
  is a smooth but steep step across a line, with $q$ at #ex.feature.ridge.angle° to the $x_1$ axis. It is
  smooth, but it is _oblique_: it depends on $x_1$ and $x_2$ only together. That is the structure that
  costs a tensor train rank (@sec:rank-pitfall).
- *The jump* $j(x) = #ex.feature.jump.height dot [p dot x > #ex.feature.jump.offset]$ adds
  #ex.feature.jump.height across a line with normal $p$ at #ex.feature.jump.angle°, cutting off the
  lower-right corner. It is both oblique and discontinuous, and no finite sum of products of
  univariate functions follows it.

Each scheme is introduced with the part of $F$ it is for. ChebTT handles the base, and finds its kink
by itself (@sec:edges). The ridge costs ChebTT rank; ChebTT-M and the hub expansion turn the
coordinates so that the ridge depends on one of them. The jump needs ChebTT-A's interface term.
@sec:compare runs every scheme on all of $F$: one function, four answers. For the hub expansion and
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
the feature example's $r$ along the line through the centre in the direction $q$, it fails in a way
that is famous. With #(ex.chebyshev.runge.degree + 1) equispaced points the interpolant
#swatch(red, shape: "line") swings far outside the function near both ends: its worst error is
#ex.chebyshev.runge.equispaced. Adding points makes it worse, not better, until rounding takes over:
at degree 64 the worst error is still #ex.chebyshev.runge.equispaced_64.

#figure(panel("runge"), caption: [The ridge's profile #swatch(ridge, shape: "line"), interpolated at
  #(ex.chebyshev.runge.degree + 1) points: equispaced #swatch(red, shape: "line"), and Chebyshev
  #swatch(dark, shape: "dash"). *Right.* The worst error against the degree, with the tolerance
  dotted. Chebyshev interpolation meets it at degree #ex.chebyshev.runge.chebyshev_within.]) <fig:runge>

#scene(side: left, width: 34%, panel("cheb-points"))[
  The cure is to cluster the points towards the ends. Take $n + 1$ equally spaced angles on a
  semicircle and drop them onto the diameter. The shadows are the _Chebyshev points_
  $x_j = cos(pi j slash n)$, $j = 0, ..., n$ (of the second kind). The interpolant through them
  #swatch(dark, shape: "dash") is indistinguishable from the ridge at the same degree: its worst
  error is #ex.chebyshev.runge.chebyshev.

  The same semicircle gives the basis. The _Chebyshev polynomial_ $T_j$ is defined by
  $T_j (cos theta) = cos j theta$: a cosine, seen through $x = cos theta$. So $|T_j (x)| <= 1$ on
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

The coefficients do more than represent $p_n$: they report on $f$. The faster they fall, the
smoother $f$ is, and the size of the last few says how much the omitted ones can still matter. Three
slices of the feature example show the three cases every scheme meets: one through the ridge only, one
through the kink, one through the jump.

#figure(panel("decay"), caption: [*Top.* Three slices of $F$: #swatch(ridge, shape: "line") smooth
  (along $x_2$ at $x_1 = #ex.chebyshev.slices.smooth.value$, across the ridge),
  #swatch(kink, shape: "line") through the kink (along $x_1$ at $x_2 = #ex.chebyshev.slices.kink.value$),
  and #swatch(jumpc, shape: "line") through the jump (along $x_2$ at $x_1 = #ex.chebyshev.slices.jump.value$).
  *Bottom.* Their Chebyshev coefficients $|a_j|$ on log--log axes (dots), with the larger of each
  neighbouring pair (line). Dashed: the reference laws $rho^(-j)$, $j^(-2)$ and $j^(-1)$.]) <fig:decay>

The smooth slice's coefficients fall geometrically, by a factor $rho approx #ex.chebyshev.decay.smooth.rho$
per degree, until they reach rounding level at degree #ex.chebyshev.decay.smooth.floor_degree. On
log--log axes a geometric law plunges. Through the kink they fall like a power, with a fitted exponent
of #ex.chebyshev.decay.kink.power, close to the 2 of the theory below. Through the jump the exponent
is #ex.chebyshev.decay.jump.power, close to 1: a jump's coefficients barely fall at all.

#theorem(title: [smoothness and decay, @trefethen2019[Thms. 7.1, 8.1]])[
  Let $f$ be analytic and bounded by $M$ inside the Bernstein ellipse with foci $plus.minus 1$ and
  semi-axis sum $rho > 1$. Then $|a_j| <= 2 M rho^(-j)$ and
  $ ||f - p_n||_infinity <= (4 M rho^(-n)) / (rho - 1). $
  Let instead $f, ..., f^((nu - 1))$ be absolutely continuous and $f^((nu))$ of bounded variation $V$.
  Then $|a_j| <= 2 V slash (pi j (j - 1) dots (j - nu))$ for $j > nu$, so $|a_j| = O(j^(-nu - 1))$.
] <thm:decay>

A kink is the case $nu = 1$: $f$ is continuous and $f'$ jumps, so $|a_j| = O(j^(-2))$. A jump is
$nu = 0$: $f$ itself has bounded variation, and $|a_j| = O(j^(-1))$. The ridge is analytic, but its
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
#ex.chebyshev.tail.smooth.quarter_degree, exactly where the true error does. The upper half waits until
degree #ex.chebyshev.tail.smooth.half_degree, twice the points on that axis for nothing. The figure
script computes the top quarter itself and stops unless it equals the library's `chebyshev_tail`. In the
experiments that chose it, at 5 and 10 dimensions, reading the top quarter instead of the upper half cut
ChebTT's evaluations on every case tried, with every case still within the tolerance.

#pitfall(title: [A tail is an estimate, and a kink fools it])[
  On the kink slice the top quarter says "converged" at degree #ex.chebyshev.tail.kink.quarter_degree,
  but the true error there is still above the share; it falls below it only at degree
  #ex.chebyshev.tail.kink.true_degree. A power law $j^(-2)$ gives the top quarter no plunge to read, and
  the estimate assumes it has seen the worst of the tail. So no scheme here trusts the tail alone. ChebTT
  checks its model at points (@sec:checks), and searches an axis for kinks and jumps when its
  coefficients decay like a power (@sec:edges).
]

#implementation(title: [In the code])[
  `src/schemes/chebyshev.py` holds what the schemes share: `cheb_points`, `coefficients_along` (the DCT-I
  along one axis), `chebyshev_tail` (twice the top quarter) and `chebyshev_bases`, which evaluates
  $T_0, ..., T_n$ by the three-term recurrence $T_(j+1) = 2 x T_j - T_(j-1)$ rather than by $cos(j arccos x)$:
  no transcendental function per query and axis, which matters at inference. ChebTree reads its tails from
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
If $f = g(x_1) h(x_2)$, every row of $A$ is a multiple of every other: $A$ has rank 1. A sum of $r$ such
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
the base. The ridge makes them fall slowly. The jump makes them fall hardly at all: the rank that a
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
fibres? The rows should be as far from linearly dependent as possible.

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
  the base but not the ridge's full width: the worst check error is #rnds.at(0).worst times the
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
  the tolerance. They miss small features: in high dimension a narrow peak fills a vanishing share of
  the box.
- *Free checks* are points ChebTT has already paid for. Fibres of earlier rounds, and the nodes of a
  degree since doubled, are no longer interpolated by the current cores. They cluster where the cross
  explored, which is where $f$ is interesting. Here their worst error is #ex.chebtt.checks.worst_free.
- *The look-ahead* evaluates the next nested grid's new nodes along every axis, through the worst
  check point and through the largest value seen. These are exactly the points a refinement would
  sample next. Its worst error, #ex.chebtt.checks.worst_look, is the round's worst. Look-ahead points
  join the check set for every later round.

If every error is within $s atol$, the model is accepted. Otherwise ChebTT must choose: more degree
or more rank. A missing degree cannot be made up by rank, and the reverse is also true. ChebTT asks
each axis whether its top degrees still matter at the check points. It removes the top quarter of axis
$k$'s coefficients and measures how much the model changes there. That change is
@lem:alias's tail, measured on the model itself. An axis whose change exceeds its share $tau = s atol slash d$
doubles its degree. If no axis does, the rank grows.

#scene(side: left, width: 40%, panel("rounds"))[
  The rounds of $b + r$: the worst check error against the evaluations spent, marked by what came
  next. #swatch(fib1) The rank grows, #swatch(fib2, shape: "dot") a degree doubles, #swatch(green)
  the model is accepted. The dotted line is $s = #ex.chebtt.defaults.safety$: a check must be within
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
  shown: fibres that all miss a narrow peak converge to a model of the empty background. No scheme that
  samples a box can certify its worst error in high dimension @novak2008, and the benchmark's test sets
  are the judge.
]

== Kinks and jumps aligned with an axis: piecewise axes <sec:edges>

#question[The base $b$ has a kink at $x_1 = #ex.feature.kink$. Its coefficients fall like $j^(-2)$, so
  degree alone converges slowly. Can ChebTT find the kink and put it where it does no harm?]

A tensor train is a sum of products of univariate functions. Such a sum can fail to be smooth only
along planes $x_k = c$, aligned with an axis. There it is enough to cut the axis at $c$ and give each
piece its own Chebyshev basis: on each piece $f$ is smooth again, and its coefficients fall
geometrically. An axis may therefore carry _breakpoints_. Its pieces share one degree, and its nodes
are the union of the pieces' Chebyshev points. ChebTT searches for breakpoints only where it can help:
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

The scan finds a cell whose second difference is an outlier: a kink's second difference is of order
$h |[f']|$ against $h^2 |f''|$ elsewhere, so it wins at every resolution. A jump is tried first, by
bisection on the value, one evaluation per halving. The two-scale test turns it down: a jump's
one-sided difference keeps its size as the stencil shrinks, and a kink's does not. The zoom then
locates the kink only as finely as the tolerance needs, here to #sci(ex.chebtt.edges.zoom_width) of the
width. The far fibres decide whether the edge is aligned with the axis. If it is present at the same
$x_1$, or absent where its factor vanishes, the edge is kept. If it appears elsewhere, the edge is
oblique and no breakpoint represents it (@sec:adaptive takes it from there).

The search cost #ex.chebtt.edges.evals evaluations and found the kink at
$x_1 = #ex.chebtt.edges.found$, #sci(ex.chebtt.edges.error) of the width from the true position. With it,
ChebTT models $b$ with #num(mdl.b.evals) evaluations, two pieces of degree #mdl.b.degrees.at(0) on
$x_1$, and a worst error of #mdl.b.worst times the tolerance. Without it (`edges=False`) the same
function costs #num(mdl.b_no_edges.evals) evaluations and degree #mdl.b_no_edges.degrees.at(0), and its
worst error is #mdl.b_no_edges.worst. The figure script rebuilds the zoom's last step from the recorded
values and stops unless it gives the model's breakpoint.

#implementation(title: [Bands, and why nodes stay off the edge])[
  A breakpoint is known only to within its location error and the largest displacement the confirming
  fibres saw: its _band_. The pieces' end nodes are inset by twice the band (at least $10^(-10)$ of the
  width, at most a quarter of either neighbouring piece), so no node of one piece falls on the other side
  of a jump. The model is not claimed within the band. Edge search draws from its own random stream, so a
  function without edges gets exactly the model it would get without the search; only its evaluation
  count differs. An axis has at most `max_pieces` $= #ex.chebtt.defaults.max_pieces$ pieces.
]

== Compression: a smaller train for the same tolerance <sec:compress>

#question[The cross grows ranks in steps of two and doubles degrees. The accepted model is larger than
  the tolerance needs. Inference pays for every coefficient. How much can be removed?]

The checks leave the accepted model at most $s atol$ from $f$ at its check points, and usually much less.
That leaves room: a quarter of the tolerance, `chop_share` $= #ex.chebtt.defaults.chop_share$, is given
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

ChebTree uses the first statement directly: it chops trailing slabs while their magnitudes sum to
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
  Carry a row vector from left to right: start with $v = 1$, and at axis $k$ form
  $v <- v G_k (x_k)$, where $G_k (x_k) = sum_j T_j (x_k) C_k [:, j, :]$ is an $r_(k-1) times r_k$
  matrix. The recurrence gives the $T_j$, and the sum and the product cost
  $r_(k-1)(n_k + 1) r_k$ multiply-adds.
]

The compressed Rosenbrock model holds #mdl.rosenbrock.madds coefficients, the base $b$
#mdl.b.madds, and $b + r$ #mdl.br.madds. A tensor grid of the same degrees would hold the products of
the node counts, and a sparse grid or a tree of boxes many more. The benchmark measures the time:
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
  A tensor train sees the world through its axes. A feature across an oblique line or a curved surface
  has a high rank at any degree, and the cross pays for rank twice: more fibres, each through every node
  of its axis. In ten dimensions an oblique coupling between every pair of axes costs rank on every bond.
  An oblique _jump_ is worse: its rank never converges. ChebTT on all of $F$ ran out of rank, at
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
$u_1$ only, a product of one univariate function and constants: rank 1 (@prop:ranks). The problem is
to find $Q$ from evaluations of $f$, and to know when it helps. Many of the library's problems have
several outputs, the species of a chemistry step or the states of a flow map, and outputs often share
such directions. So ChebTT-M works on vector-valued $f$, output by output.

== The first answer, and where it stumbles

The classical answer is the _active subspace_ @constantine2015: the eigenvectors of the average
gradient outer product. Applied to all outputs at once, it gives one rotation for everything, and that
is the stumble. The feature example's base is separable: on the axes ChebTT models it at rank
#mt.base_turned.axes_rank with #num(mt.base_turned.axes_evals) evaluations. In the ridge's coordinates
its kink is oblique, and ChebTT #if mt.base_turned.failed == none [needs rank #mt.base_turned.rank and
#num(mt.base_turned.evals) evaluations] else [fails: it reaches its rank limit after
#num(mt.base_turned.evals) evaluations]. A rotation helps an output only if the output is turned; for
one that is aligned with the axes already, it does damage.

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
  the eigenvalue. The ridge's and the bump's metrics are lines along $q$: their second eigenvalues are
  #sci(mt.outputs.at(0).eig.at(1)) and #sci(mt.outputs.at(1).eig.at(1)) of the first. The base's are
  #mt.outputs.at(2).eig.at(0) and #mt.outputs.at(2).eig.at(1).]) <fig:metric>

The rule that decides is simple. An output is turned when 99% of its metric's trace lies in at most
half as many eigenvectors as it lies in axes, the diagonal entries of $G_c$. The ridge's trace lies in
#mt.outputs.at(0).k_eig eigenvector against #mt.outputs.at(0).k_axes axes, so it is turned, and so is
the bump. The base needs #mt.outputs.at(2).k_eig of either. A separable output is never turned, since a
rotation could only destroy its rank. Outputs that are turned are grouped when their leading subspaces
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
  is turned, and the bump is not: ChebTT-M costs #num(lad.at(0).metric.evals) evaluations against
  ChebTT's #num(lad.at(0).chebtt.evals), the price of the pilot and of a capped attempt that is thrown
  away. The gain grows with the dimension (ChebTT #swatch(mcol("chebtt")), ChebTT-M
  #swatch(mcol("chebtt_metric"))): #num(lad.at(1).metric.evals) against #num(lad.at(1).chebtt.evals) at
  $d = #lad.at(1).d$, #num(lad.at(2).metric.evals) against #num(lad.at(2).chebtt.evals) at
  $d = #lad.at(2).d$. At $d = #lad.at(3).d$ ChebTT exceeds its budget of #num(mt.max_evals) evaluations
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
test points is within the tolerance: #lad.at(2).metric.worst and #lad.at(2).chebtt.worst of it.

#pitfall(title: [When not to turn, and what turning cannot do])[
  ChebTT-M on the axes _is_ ChebTT, at the same cost, for every output that meets the tolerance there. It
  pays more where a pilot buys nothing: at $d = 2$ on the pair, and wherever an output is turned that the
  axes would have handled at full rank. The rotation is global: a curved latent coordinate has no one
  direction, and the pilot sees nothing of a feature its points miss. The rotated box extends beyond the
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
  smooth step with no jump, takes a branch of line 2: $phi$ is fitted to crossings of the mid-level
  $(max f + min f) slash 2$, and $S = h(phi)$.], pseudocode(adaptive-alg)) <alg:adaptive>

#story()
#step([The first attempt meets the jump], code: adaptive-code, focus: (1, 2, 3), width: 40%, panel("adaptive-discovery"))[
  The first ChebTT's edge search finds the jump on a fibre, but on the far fibres it is somewhere else:
  oblique. The build stops there (`ObliqueEdgeError`) after #num(ad.stages.at(0).evals) evaluations and
  hands over the #ad.interface.first_points points it located on the interface (diamonds). Segments from
  the most extreme check point (cross) to random points add more (dots): each is scanned at 17 points and
  bisected on the value at its largest jump. A segment crosses a region of volume share $p$ with
  probability about $2 p (1 - p)$ in any dimension; an axis fibre does much worse as $d$ grows. A plane
  fits the points (dashed): its normal is #ad.interface.angle_error° from $p$, and its offset
  #sci(ad.interface.offset_error) from the true one.
]

#step([How big is the jump?], code: adaptive-code, focus: (4,), width: 40%, panel("adaptive-jets"))[
  At #ad.jets.points interface points ChebTT-A walks along the normal on both sides. It fits a polynomial
  of degree 6 to 8 points on each side, within 0.2 of the point and inside the box, so no evaluation
  falls outside the domain. The difference of the two fits at the interface is the jump of $f$, and
  of their derivatives the jumps of $f'$ and $f''$. Here the jump is #ad.jets.example.jump (the side
  where $phi > 0$ minus the other). Over all points its mean size is #ad.jets.jump_mean and its spread
  #sci(ad.jets.jump_spread). The figure script repeats the two fits and stops unless they give the
  library's jump.
]

#step([The remainder is smooth], code: adaptive-code, focus: (5, 6), width: 36%, panel("adaptive-remainder"))[
  $F - S$: the jump is gone (its line is still drawn #swatch(jumpc, shape: "line")), and the base and the
  ridge remain. The remainder's ChebTT has rank #ad.remainder.tt.ranks.at(0) and costs
  #num(ad.remainder.tt.evals) oracle calls, many of them answered by the cache the stages share: ChebTT-A
  spends #num(ad.evals) evaluations in all.
]

#figure(panel("adaptive-result"), caption: [*Left.* The error of ChebTT-A on $F$, within the tolerance
  everywhere: its worst is #ad.worst. *Right.* The coefficients along the jump slice of @fig:decay, of $F$
  #swatch(jumpc, shape: "line") and of $F - S$ #swatch(dark, shape: "line"): the largest in the upper
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
points agree to $0.1 atol$. They live on the side whose foot map is regular: the other side of a closed
quadric holds its centre, where the foot is undefined.

#pitfall(title: [One interface, and a band around it])[
  ChebTT-A models one interface, a level set of a plane or of a diagonal quadric. A query within the
  interface's location error may be on the wrong side, so the claim is the tolerance outside that band,
  and outside the bands of the breakpoints. Here the jump is constant, yet ChebTT-A built jump fields
  of degree #ad.interface.degree instead of the cheaper profile. Its test for a constant jump reads the
  jumps of $f''$ too, and the one-sided fits measure them noisily: their spread over the interface
  points is #ad.jets.curvature_spread, against #sci(ad.jets.jump_spread) for $f$ itself. The first
  attempt is spent on every function with an interface (here #num(ad.stages.at(0).evals) of
  #num(ad.evals) evaluations).
]

#takeaway[ChebTT-A takes the one thing a tensor train cannot follow, an oblique or curved jump, out of
  $f$ as a term aligned with the interface, and leaves ChebTT a remainder that is smooth across it.]

= The hub expansion: dense where it is steep, cheap elsewhere <sec:hub>

#question[In ten dimensions even a turned tensor train pays for every axis. If a function is steep in
  one or two directions and nearly flat in all the others, can a model's cost follow that structure?]

#let hb = ex.hub

The library's `shared_latent` problem is built for that question. Its #hb.problem.outputs outputs are
nonlinear functions of two coordinates $g_1$, $g_2$, oblique to every axis, of a $d$-dimensional input,
and $g_1$ is curved. It stands in for a slow manifold: outputs that share structure the axes do not
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

So the hub dimension $r$ is read off the spectrum: the fewest eigenvectors holding 99% of the trace,
here #hb.spectra.shared_latent.r for `shared_latent` and #hb.spectra.lift.r for the lift, at most
`r_max` $= #hb.settings.r_max$. What the hub leaves out, #sci(hb.spectra.lift.share_left) of the lift's
trace, decides whether the other directions get terms at all: they do when it exceeds $10^(-4)$.

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
  Uniform points pile up at the centre: their largest hub coordinate reaches #hb.design.uniform_reach,
  and #pct(hb.design.uniform_outer) of them lie beyond 0.5. The design reaches #hb.design.design_reach,
  with #pct(hb.design.design_outer) beyond 0.5.]) <fig:hub-design>

A sum of $d$ uniform coordinates concentrates near its mean, so a uniform sample of a 10-D box rarely
reaches the extremes of an oblique coordinate. A polynomial fitted there extrapolates at the edges of
its own variables, which is where its worst error lands. The design fixes this in two ways. It
draws _vertex-directed_ points, corners of the box chosen by the sign of a random combination of the
hub directions, and blends of them with uniform points. Then it picks a well-conditioned set from that
pool by a pivoted QR of the hub basis, and adds uniform points.

#scene(side: left, width: 38%, panel("hub-budget"))[
  The degree is chosen on a hold-out drawn from the same design: the candidate whose fit on part of the
  samples has the smallest worst error on the rest. Then the budget doubles until the hold-out is within
  half the tolerance (dotted), or stops improving by 10%. On `shared_latent` one round is enough, with
  values only #swatch(mcol("hub")) and with the Jacobian #swatch(dark). On the lifted ridge
  #swatch(ridge) the second round improves the hold-out from #hb.lift.history.at(0).at(1) to
  #hb.lift.history.at(1).at(1) times the tolerance, less than 10%, and the budget stops: more samples of
  the same design do not fix an error of the model class. A hold-out worst above $atol$
  (`holdout_worst`) reports a model that missed, rather than hiding it.
]

== With the oracle's Jacobian

When the oracle supplies gradients (an adjoint, automatic differentiation, ODE sensitivities), they
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
the hub expansion misses: #num(lf.evals) evaluations and a worst error of
#lf.worst times the tolerance $#ex.feature.alpha$. Its hold-out says so: #lf.holdout. With the exact
Jacobian it comes within #lf.jacobian.worst. ChebTT-M on the same function holds, with
#num(lf.metric.evals) evaluations and a worst error of #lf.metric.worst. On Rosenbrock, in two
dimensions, the hub spends #num(hb.rosenbrock.evals) evaluations, its first budget, where ChebTT spends
#num(mdl.rosenbrock.evals).

#pitfall(title: [Outside its class])[
  The hub expansion suits a few steep directions with weak, low-order dependence on the rest. A steep
  profile along a hub direction needs a high degree, and the rotated coordinate's range grows with $d$
  (it spans the box's extent along an oblique direction), while the candidate degrees stop at
  #hb.settings.degrees.at("1").last() for one hub direction. Isotropic and radial features, many
  differently oriented ones, and jumps are outside its class. The hold-out is an estimate from the same
  design, not a bound. Each group is capped at #num(hb.settings.max_terms) terms, since its Gram matrix
  costs the square of that and its solve the cube.
]

#takeaway[The hub expansion reads the few steep directions off the gradient covariance, fits a dense
  Chebyshev model in them and degree-two terms in all the others, so its cost grows linearly with $d$;
  it is the tool for low-dimensional structure in high dimension, and it says when a function is not.]

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

The table is the companion in one picture. On the base, every scheme holds, and the three ChebTTs need
a few hundred evaluations. ChebTT-M and ChebTT-A count each point once, through a shared cache, while
ChebTT counts every call of the oracle, and its look-ahead can ask for a point twice. With the ridge
added, ChebTT pays rank, and the others pay about as much. With the jump added, ChebTT and ChebTT-M run
out of rank, and the hub expansion misses by #cp.layers.brj.hub.worst times the tolerance: only ChebTT-A
holds, with #num(cp.layers.brj.chebtt_adaptive.evals) evaluations.

#board(
  (panel("compare-afinni"), [AFINNI: #num(cp.baselines.afinni.evals) evaluations, #cp.baselines.afinni.leaves leaves]),
  (panel("compare-art"), [ART, affine: #num(cp.baselines.art.evals), #cp.baselines.art.leaves leaves]),
  (panel("compare-art_quad"), [ART, quadratic: #num(cp.baselines.art_quad.evals), #cp.baselines.art_quad.leaves leaves]),
  (panel("compare-hub"), [hub expansion: #num(cp.layers.brj.hub.evals)]),
  (panel("compare-chebtt_adaptive"), [ChebTT-A: #num(cp.layers.brj.chebtt_adaptive.evals)]),
  columns: 5,
)

AFINNI and ART refine towards the jump, and keep refining: their trees reach their depth limits with
#num(cp.baselines.art_quad.leaves) to #num(cp.baselines.afinni.leaves) leaves and more than a hundred
thousand evaluations. They are within the tolerance on #cp.baselines.afinni.within% to
#cp.baselines.art.within% of the grid, but their worst errors, #cp.baselines.art_quad.worst to
#cp.baselines.afinni.worst times the tolerance, sit on the jump. A piece that straddles it has an
error of half the jump at least, and a piece too small to straddle it is never small enough. In this
figure only ChebTT-A's map is pale everywhere.

== The benchmark

#scene(side: left, width: 42%, panel("store-cases"))[
  Each dot is a case of the benchmark's function suite, a problem, dimension and tolerance that both
  families ran: the best worst error of AFINNI and ART against the best of the schemes, over every
  seed. Colours are classes: #swatch(rgb(st.class_colours.smooth)) smooth,
  #swatch(rgb(st.class_colours.at("many local features"))) many local features,
  #swatch(rgb(st.class_colours.at("kinks, jumps, fronts"))) kinks, jumps and fronts,
  #swatch(rgb(st.class_colours.at("ODE flow maps"))) ODE flow maps, and
  #swatch(rgb(st.class_colours.at("vector-valued"))) vector-valued problems. A hollow dot's best scheme has
  records older than its last change to its results. An exact model is drawn at the
  bottom edge, and a failed run (over its budget or the time limit) at the right or top edge. Of #st.total cases, a scheme holds and no baseline
  does in #st.scheme_only (#st.high_d_scheme_only of them at $d >= 10$); a baseline holds and no scheme
  does in #st.baseline_only\; both hold in #st.both, and neither in #st.neither.
]

The benchmark's summary, per class of problem, from its store of #num(st.records) runs (the README
shows the same table). A case is _held_ when the worst error is within the tolerance on every seed, and
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

The pattern does not depend on the stale records' details. Least-squares pieces on random points
control the mean error but not the worst: their errors between the points are unchecked, and that gap
grows with the dimension. Interpolation at Chebyshev nodes, with checks at the points a refinement would
sample next, lands its worst errors below the tolerance. ChebTree holds the kinks, jumps and fronts it
ran, but only in two and three dimensions (@sec:aside).

== Choosing among the four

#cards(
  columns: 2,
  ([ChebTT], [Smooth functions, and functions with kinks or jumps along the axes, of any dimension,
    with low-rank coupling. Oblique and curved features cost it rank. On the base it needs
    #num(cp.layers.b.chebtt.evals) evaluations.]),
  ([ChebTT-M], [Vector-valued functions whose outputs are ridges of shared oblique coordinates. It is
    ChebTT on every output the axes serve, and its gain grows with $d$: on the ridge pair,
    #num(lad.at(2).metric.evals) evaluations against #num(lad.at(2).chebtt.evals) at $d = #lad.at(2).d$.]),
  ([ChebTT-A], [One jump across an oblique plane or a diagonal quadric, or a sharp front. The only scheme
    that holds all of $F$: #num(cp.layers.brj.chebtt_adaptive.evals) evaluations.]),
  ([Hub expansion], [A few steep directions and weak dependence on the rest, in high dimension, with the
    oracle's Jacobian when it has one: #num(hb.jacobian.evals) evaluations and #num(hb.jacobian.jacobians)
    Jacobians on `shared_latent` at $d = #hb.problem.d$. It says when a function is outside its class.]),
)

#takeaway[Where a function has the structure a scheme is built for, the scheme holds the worst error
  with a few thousand evaluations, where local affine pieces spend a hundred thousand and still miss it.]

= Stiff step maps: the chemistry the library is for <sec:steps>

#question[A stiff chemistry step is called by a simulation in every cell at every time step. Can a
  global approximant replace the solver, and what does chaining its answers do to its error?]

#let sp = ex.steps
#let rob = sp.robertson
#let stiff = sp.regimes.stiff

The library's end goal is surrogates of astrochemical reaction networks inside MHD simulations: a
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

The speedup is the solver's time per state and step (SciPy's Radau, the integrator the system would be
solved with) over the surrogate's inference time, at the cases where both errors hold. AFINNI and ART
hold no rollout. ChebTT holds the one-step error almost everywhere, and the retrained ChebTT holds both on
#stiff.at(0).rollout of #stiff.at(0).ran problems.

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
with #kd.species species and #num(kd.reactions) reactions. Measured in the iteration studies on its own
tracer settings, one hydro step takes the solver about #kd.ms_per_step ms and #kd.lu_per_step sparse
factorisations. Over that step all but #kd.slow_low to #kd.slow_high of its modes relax completely. A
step map with a few dozen slow directions is the structure the hub expansion and ChebTT-M are built
for: steep in a few coordinates the data reveal, flat in the rest. On the 14-species Nelson network,
about #kd.nelson_slow of #kd.nelson_species species' directions stay slow.

#pitfall(title: [A box is not where a simulation goes])[
  The box that bounds the visited states is mostly empty in high dimension, and ChebTT samples all of it,
  including states that are not physical: on the #(kd.nelson_inputs)-input Nelson problem a box ChebTT
  did not train within the time cap. The surrogate must live in the few slow coordinates and be trained along the
  states the simulation visits. The rollout error needs one-step errors well below the tolerance unless
  errors cancel. The figures here are the one-step and rollout errors on held-out trajectories, not
  bounds.
]

#takeaway[On stiff step maps a ChebTT retrained until its rollouts hold keeps both errors within the
  tolerance on almost every problem, at speedups of a thousand and more over the solver, where AFINNI and
  ART hold no rollout; the path to large networks is the few slow coordinates.]

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

*ChebTree* is an adaptive tree of tensor Chebyshev boxes: a box raises its degree where its
coefficients decay geometrically and splits where they do not. On $b + r$ it uses
#asd.chebtree.boxes boxes and #num(asd.chebtree.evals) evaluations, worst #asd.chebtree.worst. It holds every
kink, jump and front it ran, #srow("kinks, jumps, fronts", "chebtree").held of
#srow("kinks, jumps, fronts", "chebtree").ran, at a median of #num(srow("kinks, jumps, fronts", "chebtree").evals)
evaluations. But a tensor box costs $product_k (n_k + 1)$ points, and the benchmark ran it to $d = #st.max_d.chebtree$
only.

*SparseCheb* is a dimension-adaptive sparse Chebyshev grid @gerstner2003. On $b + r$ it is the cheapest
scheme here: #num(asd.sparsecheb.evals) evaluations over #asd.sparsecheb.indices indices, worst
#asd.sparsecheb.worst. Its inference on the smooth cases it holds is #srow("smooth", "sparsecheb").us µs
per query, against #srow("smooth", "chebtt").us for ChebTT, and one global expansion pays for every kink
or jump. It was run to $d = #st.max_d.sparsecheb$.

*MetricKernel* interpolates with a Matérn kernel in a metric learned from the model's own gradients. On
$b + r$ its worst error is #asd.kernel.worst times the tolerance after #num(asd.kernel.evals) evaluations:
random check batches miss the worst point, and a kernel interpolant is weakest at the boundary, where most
of a high-dimensional box's volume is. It holds #srow("smooth", "kernel").held of
#srow("smooth", "kernel").ran smooth cases, in two dimensions only.

*SlowTable* tabulates a step map over the states a simulation visits, after ISAT @pope1997: each record
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
  Jacobians it evaluated and the seconds spent in them, and does not deduplicate: a scheme that may ask
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

#question[Global models, adaptive sampling, tolerances judged at points. What stands between a change to
  a scheme and a wrong surrogate?]

#let ap = ex.appendix
#cards(
  columns: 3,
  ([Unit tests], [#ap.test_count tests in `tests/unit/test_schemes_*.py` check each scheme on small cases:
    ChebTT's edge search and pieces, ChebTT-M's groups, ChebTT-A's interfaces, the hub's groups and
    Jacobian, saving and loading, and the shared contract.]),
  ([Bit-identical regression], [`scripts/scheme_regression_check.py` trains #ap.regression_cases.len()
    small cases, one or two per scheme, and compares their evaluations and predictions bit for bit
    against a reference. A refactor must reproduce them; BLAS runs on one thread.]),
  ([Many ranks], [`tests/mpi` runs the distributed oracle under real `mpiexec`: the values do not depend
    on the rank count, and a failure on one rank reaches rank 0.]),
  ([The figure script], [It records ChebTT's, ChebTT-A's and the hub's stages and checks the recording
    changes nothing. It rebuilds the cross as a skeleton decomposition, the tail, the edge search's last
    step, the compression's budget, the metric groups, the one-sided jumps and the hub's terms, and stops
    unless each matches the library.]),
  ([The benchmark], [Every method is judged on a fixed test set by its worst error, and every record
    names the code that ran it, so a stale record is marked rather than mixed in.]),
  ([What is not checked], [The tolerance between test points: the checks are estimates. No scheme here
    certifies its worst error, and in high dimension no box sampler can @novak2008.]),
)

#takeaway[Tests check each scheme on small cases, a regression check pins every scheme bit for bit, and
  the figures of this companion are rebuilt from definitions and compared with the library.]

#bibliography("references.yml", title: [References])

#show: appendix
#pagebreak(weak: true)

= The settings of the four schemes <sec:settings>

Each table is read from the settings dataclass by the figure script, with the comment beside each field
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
