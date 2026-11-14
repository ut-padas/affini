#import "companion/lib.typ": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#show: report.with(
  title: [AFINNI and ART: adaptive surrogates on one refinement engine],
  subtitle: [A technical companion to the code: the two algorithms, the engine that runs them, and the
    contracts they keep],
  authors: ((name: [PADAS group], affiliation: [The University of Texas at Austin]),),
  date: [November 2026],
  abstract: [
    A surrogate replaces an expensive function by a cheap one that agrees with it to a stated
    tolerance. This code builds surrogates out of many small affine models, placed adaptively:
    where one model is not accurate enough, the region is cut in two and each half gets its
    own. It implements two ways of doing that. AFINNI @manuscript clusters a given sample,
    cuts clusters along the function's curvature and answers queries by a nearest-neighbour
    search among trust regions. ART @art cuts the domain itself into polytopes, samples each
    one afresh and answers queries by walking a tree. Both run on one refinement engine, on
    CPUs or GPUs and across MPI ranks. This companion follows one two-dimensional example
    through both algorithms and through the engine. Every idea is shown as a picture,
    explained in words and stated formally; the pictures are computed from real runs, and the
    claims about the implementation are the ones its tests check.
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
// Numbers and colours of the running example, written by figures/make_figures.py together
// with its panels, so that the text, the panels and the diagrams below cannot disagree.
#let ex = json("figures/example.json")
#let col = ex.colours
#let dark = rgb(col.dark)
#let red = rgb(col.red)
#let pink = rgb(col.pink)
#let cyan = rgb(col.cyan)
#let grey = rgb(col.grey)
#let panel(name, width: 100%) = image("figures/" + name + ".svg", width: width)
// A picture keeps its caption on its own page (the template lets figures break, for algorithms).
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

// Notation, kept from the AFINNI manuscript wherever it has a symbol.
#let fh = $hat(f)$
#let atol = $alpha_"tol"$
#let rtol = $r_"tol"$
#let Nmin = $N_"min"$
#let Ninit = $N_"init"$
#let absH = $abs(H)$
#let Cm = $C_(-1)$
#let Cp = $C_1$

// ---------------------------------------------------------------------- the algorithms
#let engine = (
  require[a plan (AFINNI or ART), the oracle $f$, tolerances $atol$, $rtol$, a depth limit],
  ensure[the leaves $cal(S)$, each with its local model],
  ln(0, [queue $<-$ the plan's initial clusters], note: [rank 0, then scattered (@sec:mpi)]),
  ln(0, [*while* this rank has clusters, or can steal some]),
  ln(1, [*for* each batch of whole clusters that fits the device], note: [@sec:batches]),
  ln(2, [answer steal requests from the tail of the queue], note: [@alg:steal]),
  ln(2, [leaves, children $<-$ the plan processes the batch], note: [@alg:afinni or @alg:art]),
  ln(2, [append the leaves to $cal(S)$ and the children to the next queue]),
  ln(1, [queue $<-$ the next queue, without children beyond the depth limit]),
  ln(0, [gather $cal(S)$ on rank 0], note: [ART also assembles its tree]),
)
#let engine-code = (label: <alg:engine>, lines: engine)

#let afinni = (
  require[a batch of clusters $C$ with points $X_C$ and values $f(X_C)$; $Nmin$, $atol$, $rtol$],
  ensure[leaves $(c, (w, b), T)$ and, for the next level, two children per rejected cluster],
  ln(0, [*parallel for* each cluster $C$]),
  ln(1, [$c <-$ mean of $X_C$; evaluate $f(c)$]),
  ln(1, [$(w, b) <- op("arg min") sum_(x_j in X_C) abs(f(x_j) - w^top (x_j - c) - b)^2$], note: [@eq:fit]),
  ln(1, [$e_a, e_r <-$ the cluster's errors], note: [@eq:errors]),
  ln(0, [*parallel for* each cluster with $e_r < rtol$ or $e_a < atol$], note: [accepted]),
  ln(1, [$T <-$ trust ellipsoid of $X_C$], note: [@alg:trust]),
  ln(1, [add $(c, (w, b), T)$ to the leaves]),
  ln(0, [*parallel for* each other cluster], note: [rejected]),
  ln(1, [*if* $\#C < 2 Nmin$: sample new points in $C$ and evaluate $f$ there], note: [@sec:augment]),
  ln(1, [$absH <-$ Hessian estimate], note: [@alg:split]),
  ln(1, [$Cm, Cp <-$ Hessian-informed split], note: [@alg:split]),
  ln(1, [append $Cm$ and $Cp$, next to each other, to the next level]),
)
#let afinni-code = (label: <alg:afinni>, lines: afinni)

#let splitting = (
  require[points $X_C$ of a rejected cluster, their values, the centroid $c$ and $f(c)$],
  ensure[children $Cm$, $Cp$],
  ln(0, [$hat(g) <-$ least-squares gradient at $c$ from the points nearest to $c$]),
  ln(0, [$e_i <- abs(f(x_i) - f(c) - hat(g)^top (x_i - c))$ for every point], note: [Taylor error]),
  ln(0, [$cal(I) <-$ the $N_"cov"$ points with the smallest $e_i$]),
  ln(0, [$Sigma <- sum_(i in cal(I)) (x_i - c)(x_i - c)^top$; #h(0.4em) $absH^(1 slash 2) <- Sigma^(-1 slash 2)$], note: [@thm:hessian]),
  ln(0, [$tilde(x)_i <- absH^(1 slash 2) (x_i - c)$], note: [stretch along the curvature]),
  ln(0, [$u_1 <-$ first eigenvector of $sum_i tilde(x)_i tilde(x)_i^top$]),
  ln(0, [$s_i <- u_1^top tilde(x)_i$; sort the points by $s_i$]),
  ln(0, [$Cm <-$ the lower half, $Cp <-$ the upper half], note: [median cut]),
)
#let split-code = (label: <alg:split>, lines: splitting)

#let trust = (
  require[points $X_C$ of an accepted cluster, centred at $c$; regularization $gamma$],
  ensure[$T$ positive definite, with $(x - c)^top T (x - c) <= 1$ for every $x in X_C$],
  ln(0, [$Sigma <- sum_i (x_i - c)(x_i - c)^top = U Lambda U^top$]),
  ln(0, [$Sigma <- U max(Lambda, gamma) U^top$], note: [no flat directions]),
  ln(0, [$alpha <- max_i (x_i - c)^top Sigma^(-1) (x_i - c)$], note: [the farthest point]),
  ln(0, [$T <- Sigma^(-1) slash alpha$]),
)
#let trust-code = (label: <alg:trust>, lines: trust)

#let inference = (
  require[the leaves $(c_i, (w_i, b_i), T_i)$, a query $x$, a number of candidates $kappa$],
  ensure[$fh(x)$, and whether $x$ is inside a trust region],
  ln(0, [$cal(C)_x <-$ the $kappa$ centres nearest to $x$], note: [Euclidean, exact]),
  ln(0, [$d_i <- (x - c_i)^top T_i (x - c_i)$ for each $c_i in cal(C)_x$]),
  ln(0, [$nu <- op("arg min")_i d_i$]),
  ln(0, [$fh(x) <- w_nu^top (x - c_nu) + b_nu$]),
  ln(0, [*return* $fh(x)$, and "inside" if $d_nu <= 1$, "outside" otherwise]),
)
#let inference-code = (label: <alg:inference>, lines: inference)

#let art = (
  require[a batch of nodes $C$: region $Omega_C$, inherited points, path code $pi$; budget $n$],
  ensure[leaves $(pi, (w, b))$; split records $(pi, a, z)$; two children per split node],
  ln(0, [*parallel for* each node $C$]),
  ln(1, [draw new points in $Omega_C$ until $C$ has $n$; evaluate $f$ at the new ones], note: [@alg:hitrun]),
  ln(1, [$(w, b) <-$ affine fit; $e_a, e_r <-$ its errors], note: [as AFINNI]),
  ln(0, [each node within tolerance becomes a leaf], note: [`accepted`]),
  ln(0, [each other node at the depth limit becomes a leaf anyway], note: [`max_depth`]),
  ln(0, [*parallel for* each other node $C$]),
  ln(1, [$(a, z) <-$ split search on the points of $C$], note: [@alg:search]),
  ln(1, [*if* a side would hold fewer than $p$ points: $C$ becomes a leaf], note: [`split_failed`]),
  ln(1, [$Cm <- {x in C : a^top u(x) < z}$, #h(0.3em) $Cp <- {x in C : a^top u(x) >= z}$]),
  ln(1, [their regions: $Omega_C$ with the row $a^top u <= z$, or $a^top u >= z$, added]),
  ln(1, [their path codes: $2 pi$ and $2 pi + 1$; record $(pi, a, z)$]),
)
#let art-code = (label: <alg:art>, lines: art)

#let hitrun = (
  require[the region $Omega_C = {u : A u <= bold(z)}$, a point $u in Omega_C$, a number of steps $tau$],
  ensure[a new point of $Omega_C$],
  ln(0, [*repeat* $tau$ times]),
  ln(1, [$g <-$ a direction, uniform on the unit sphere]),
  ln(1, [$[t_"min", t_"max"] <- {t : A(u + t g) <= bold(z)}$], note: [the chord through $u$]),
  ln(1, [$u <- u + t g$ with $t$ uniform in $[t_"min", t_"max"]$]),
  ln(0, [*return* $u$]),
)
#let hitrun-code = (label: <alg:hitrun>, lines: hitrun)

#let search = (
  require[the points of a node in normalized coordinates $u$, their values in units of the tolerance],
  ensure[a plane $(a, z)$ with $norm(a) = 1$],
  ln(0, [directions $<-$ the Hessian-informed one, further eigenvectors, the points' principal one, random ones], note: [$n_"dir"$ in all]),
  ln(0, [*for* each direction $a$ and each of $n_"bins" - 1$ cuts $z$ at quantiles of $a^top u$]),
  ln(1, [score $(a, z)$ by the squared error of two affine fits, one per side]),
  ln(0, [$(a, z) <-$ the best candidate], note: [the best few are refitted exactly]),
  ln(0, [*repeat* at most `soft_iters` times], note: [soft search]),
  ln(1, [step $(a, z)$ down the gradient of $J_h$, keeping $norm(a) = 1$], note: [@eq:soft]),
  ln(0, [move $z$ to the first point on the right; keep the result only if its two hard fits are better]),
)
#let search-code = (label: <alg:search>, lines: search)

#let steal = (
  require[rank $i$ of $P$; its work $w_i$ (points); threshold $w_*$; steal size $w_s$],
  ensure[every cluster is processed by exactly one rank],
  ln(0, [*while* not retired]),
  ln(1, [$L <-$ loads the other ranks have published], note: [gossip, never blocks]),
  ln(1, [*if* $w_i > 0$], note: [*victim*]),
  ln(2, [process one batch]),
  ln(2, [*for* each steal request from a rank $s$]),
  ln(3, [*if* more than $w_*$ of unstarted work is left: send $s$ clusters from the tail of the queue]),
  ln(3, [*else* send $s$ a denial], note: [always one reply]),
  ln(1, [*else*], note: [*thief*]),
  ln(2, [publish $w_i = 0$; deny every request]),
  ln(2, [*if* $sum_j L_j = 0$: join the barrier; retire once every rank has joined]),
  ln(2, [$r <- op("arg max")_j L_j$; *if* $L_r >= w_*$: ask $r$ and wait for its grant or denial]),
)
#let steal-code = (label: <alg:steal>, lines: steal)

= An expensive function, a cheap stand-in <sec:intro>

#question[A function takes a second to evaluate and is needed a billion times. What can
  stand in for it, and how much can the stand-in be trusted?]

#scene(side: left, width: 34%, panel("example"))[
  A _surrogate_ $fh$ is a cheap function built from a limited number of evaluations of the
  expensive $f$, so that $fh(x)$ is close to $f(x)$ wherever it will be asked. This companion
  builds surrogates from the simplest pieces that can work: _affine_ models,
  $fh(x) = w^top (x - c) + b$, each valid near its own centre $c$. A single affine model
  fits a curved function only over a small region, so the question is where to put the
  pieces: many where $f$ bends, few where it is nearly flat.

  The running example is the function on the left, the curved valley
  $f(x) = (1 - x_1)^2 + (x_2 - x_1^2)^2$ on the box
  $[#ex.domain.lower.at(0), #ex.domain.upper.at(0)] times [#ex.domain.lower.at(1), #ex.domain.upper.at(1)]$,
  drawn by its level sets: light on the valley floor, darker where $f$ is large. The
  surrogate must agree with it to an absolute tolerance $atol = #ex.alpha$.
]

Both algorithms of this code place the pieces _adaptively_. They fit one affine model to a
region, measure its error against the tolerance, and where the error is too large they cut
the region in two and repeat on each half. They differ in what a region is, where its
points come from and how the cut is chosen.

#board(
  (panel("afinni-level-0"), [*AFINNI* starts from a given sample, clustered into a few groups, ...]),
  (panel("afinni-level-2"), [... splits the groups whose model is not accurate enough, ...]),
  (panel("afinni-trust-all", width: 46%), [... and keeps a trust region around each accepted group.]),
  (panel("art-level-0"), [*ART* starts from the domain itself, ...]),
  (panel("art-level-3"), [... cuts it by planes, drawing fresh points in each part, ...]),
  (panel("art-partition", width: 46%), [... and ends with an exact partition into polytopes.]),
  columns: 3,
)

#block(breakable: false, above: 1em, below: 1em, table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[][AFINNI][ART],
  [a region is], [a _cluster_ of sample points], [a _polytope_ of the domain],
  [its points], [are given; a few are added inside small clusters], [are drawn by the node itself, uniformly in its polytope],
  [the cut], [follows the curvature of $f$, estimated from the points], [is searched for: the plane whose two fits are best],
  [what is kept], [only the leaves, each with a trust ellipsoid], [the whole tree, with every plane],
  [a query], [finds the nearest leaf in the leaves' own metrics, and may be flagged as outside every trust region], [walks down the tree to exactly one leaf],
))

AFINNI suits a sample that already exists, such as the states a simulation visited: it
never needs to know the domain. ART suits a known domain on which every point must get an
answer. In the code both are _plans_ of one _refinement engine_ (@sec:engine), chosen by
`TrainingConfig(algorithm="afinni")` or `"art"`.

== How to read this companion

Each idea appears three times: as a *picture* of the running example, in plain *words*, and
in *symbols*, as a definition, lemma or theorem in a tinted box. The pictures and the
symbols say the same thing, so either can be used to check the other. The pictures are
computed, not drawn: `figures/make_figures.py` trains both algorithms on the example,
records every level of the runs, and stops unless what it recorded reproduces the trained
models exactly. It also writes the numbers this text quotes.

Colours keep one meaning throughout. In a map of clusters each cluster has its own colour,
with a #swatch(dark, shape: "cross") dark cross at its centre if its model was accepted and
a #swatch(red, shape: "cross") red one if it was rejected. In a map of a model's _errors_ at
its points, a point is #swatch(rgb(col.residual_mid)) pale where the model is within the tolerance
and deep #swatch(rgb(col.blue)) blue or #swatch(red) red where it is too high or too low.
Highlighted pseudocode lines are the ones the picture beside them shows.

The notation follows the AFINNI manuscript @manuscript, and ART reuses it where the
objects are the same.

#translate(
  [the function, its surrogate, the input dimension], [$f$, $fh$, $d$],
  [the initial points and their number], [$X_0$, $N$ #h(0.3em) (`X_train`)],
  [a cluster, its points, its centre], [$C$, $X_C$, $c$],
  [the local model of a cluster], [$fh(x) = w^top (x - c) + b$ #h(0.3em) (`A_fit`, and `b_fit` $= b - w^top c$)],
  [the absolute and relative tolerances], [$atol$, $rtol$ #h(0.3em) (`atol`, `rtol`)],
  [the number of initial clusters; the minimum number of points in a cluster], [$Ninit$, $Nmin$ #h(0.3em) (`num_initial_clusters`, `min_pts_per_cluster`)],
  [the Hessian of $f$ and its absolute value], [$H$, #h(0.2em) $absH = U abs(D) U^top$ if $H = U D U^top$],
  [the trust region of a leaf], [$(x - c)^top T (x - c) <= 1$ #h(0.3em) ($T$ is `Q` in the code)],
  [the leaves and their number], [$cal(S)$, $N_c$],
  [ART: a node's region, in coordinates $u in [0, 1]^d$ of the domain], [$Omega_C = {u : A u <= bold(z)}$],
  [ART: a split plane; the side $a^top u >= z$ is the child $Cp$], [$(a, z)$, $norm(a) = 1$ #h(0.3em) (`normal`, `offset` in the code)],
  [ART: a node's path code, and its number of points], [$pi$, #h(0.2em) $n = ceil(m p)$ with $p = d + 1$],
  [the MPI ranks, and the work of rank $i$], [$P$, $w_i$],
)

#takeaway[Fit an affine model, test it against the tolerance, cut where it fails: AFINNI
  does this to clusters of a given sample, ART to polytopes of a known domain.]

= One engine, two plans <sec:engine>

#question[Two algorithms that fit, test and cut. What do they share, and what is each
  one's own?]

Everything except what happens to one batch of clusters. Both algorithms keep a _queue_ of
clusters waiting to be fitted, process it in batches, send accepted clusters to an output
buffer and the children of rejected ones to the next level's queue. The _refinement engine_
owns that loop, with the memory management, the transfers to the GPU, the MPI distribution
and the work stealing. An algorithm is a _plan_: it says what the initial queue is, where a
batch may end, how much memory a point needs, and what to do with one batch.

#algorithm(caption: [The refinement engine: `refine` in `src/engine/driver.py` and
  `refine_rank` in `src/engine/refinement.py`. Lines 5--6 are the plan's.],
  pseudocode(engine)) <alg:engine>

#scene(width: 54%, diagram(
  // The level loop is the cycle queue -> batch -> plan -> next queue -> queue; leaves leave it.
  spacing: (3.9em, 2.6em),
  node-stroke: 0.6pt,
  node-corner-radius: 2pt,
  node-inset: 5pt,
  node-shape: fletcher.shapes.rect,
  label-sep: 3pt,
  node((0, 0), small[queue \ of level $ell$], name: <q>, fill: luma(245), width: 5.4em),
  node((1, 0), small[one batch \ on the device], name: <b>, fill: luma(245), width: 5.9em),
  node((1, 1), small[*plan* \ fit, test, cut], name: <p>, fill: rgb("#fdf0c8"), width: 5.9em),
  node((0, 1), small[queue \ of level $ell + 1$], name: <n>, fill: luma(245), width: 5.4em),
  node((2, 1), small[leaves $cal(S)$], name: <s>, fill: luma(245)),
  edge(<q>, <b>, "-|>"),
  edge(<b>, <p>, "-|>"),
  edge(<p>, <s>, "-|>", label: small[accepted], label-side: left),
  edge(<p>, <n>, "-|>", label: small[children], label-side: right),
  edge(<n>, <q>, "-|>", label: small[next level], label-side: right),
))[
  A level is one pass over the queue. Clusters never move between levels except by being
  split, so in a single process the queue of level $ell$ holds exactly the clusters at
  depth $ell$ of the forest. The loop ends when the queue is empty: every cluster has
  either become a leaf or reached the depth limit.

  Processing whole _levels_, instead of following one branch down, is what makes the
  algorithm parallel: all clusters of a batch are fitted, tested and cut at once, as
  array operations over thousands of clusters (@sec:batches).
]

#figure(panel("queue-strips"), caption: [The queue of every AFINNI level of the running
  example, as it is stored: each cluster's points back to back, in one array. Accepted
  clusters (pale) leave for the leaves; each rejected cluster (outlined red) is replaced in
  the next level by its two children, next to each other. Some rows grow: a small rejected
  cluster is given new points before it is cut (@sec:augment).]) <fig:queue>

#translate(
  [a cluster is a run of consecutive rows], [rows `starts[k]` to `starts[k] + sizes[k]` of `X`, `F`],
  [a batch is a run of consecutive clusters], [`plan.batch_end(...)` chooses where it ends],
  [the children of a cluster are adjacent], [children $2 j$ and $2 j + 1$ of the next queue belong to the $j$-th cluster cut at this level],
)

#invariant(title: [queue order])[
  Batches are taken in queue order, rejected clusters are cut in queue order, and each
  appends its two children consecutively. Within one rank the queue of every level is
  therefore sorted by position in the forest. Nothing has to be sorted or looked up
  between levels, and a rank can give away the tail of its queue as one contiguous block.
] <inv:order>

The plans differ in the following, and in nothing else. `tests/unit/test_plans.py` checks
that both provide every member of the `Plan` protocol in `src/core/plan.py`.

#block(breakable: false, table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[][AFINNI (`AFINNIPlan`)][ART (`ARTPlan`)],
  [initial queue], [the $Ninit$ clusters of the initial sample], [the roots: the domain, or a grid or Voronoi cells of it],
  [points of a cluster], [its own, plus new ones if it is small and rejected], [topped up to $n$ inside its polytope before fitting],
  [extra columns], [none], [root, path code $pi$, the rows of $Omega_C$],
  [cut], [Hessian-informed, at the median], [searched plane $(a, z)$, recorded],
  [at the depth limit], [the cluster is dropped and reported], [the node becomes a leaf, flagged],
  [after the run], [nothing], [assembles and checks the tree],
))

#takeaway[The engine owns the queue, the batches, the memory and the ranks; a plan owns
  what a cluster is and what happens to one batch.]

= AFINNI: fit, test and what the errors say <sec:fit>

#question[A cluster of points with known values of $f$. Is one affine model enough for
  it?]

AFINNI begins by clustering the initial points $X_0$ into $Ninit$ groups with $k$-means:
the roots of a forest of binary trees. Then every level asks the same question of every
cluster. The answer is computed, not guessed: fit the best affine model to the cluster's
points, and look at how far it is from $f$ at those same points.

#definition(title: [local model and cluster errors])[
  For a cluster $C$ with points $X_C$ and centre $c$ (their mean), the _local model_ is
  $fh(x) = w^top (x - c) + b$ with
  $ (w, b) = op("arg min")_(w, b) sum_(x_j in X_C) abs(f(x_j) - w^top (x_j - c) - b)^2 . $ <eq:fit>
  Its _absolute_ and _relative errors_ are
  $ e_a = max_(x in X_C) abs(f(x) - fh(x)), quad e_r = e_a / (max_(x in X_C) abs(f(x))) . $ <eq:errors>
  The cluster is _accepted_ if $e_a < atol$ or $e_r < rtol$, and _rejected_ otherwise.
] <def:errors>

#algorithm(caption: [AFINNI on one batch of clusters (the manuscript's Algorithm 1, batched):
  `AFINNIPlan.process_batch` in `src/afinni/plan.py`.], pseudocode(afinni)) <alg:afinni>

#story()
#step([Level 0: no cluster passes], code: afinni-code, focus: (1, 2, 3, 4), window: 1, width: 36%, panel("afinni-error-0"))[
  The #ex.afinni.levels.at(0).clusters initial clusters, with every point coloured by the
  error of its cluster's model. Each cluster shows the same pattern: the model is too low
  at the rim and too high in the middle, or the reverse. That bowl is the part of $f$ an
  affine model cannot follow, its curvature. The worst error is
  #ex.afinni.levels.at(0).worst_error, far above $atol = #ex.alpha$, so all
  #ex.afinni.levels.at(0).clusters clusters are rejected (red crosses).
]

#step([The errors shrink with the clusters], code: afinni-code, focus: (5,), window: 1, width: 36%, panel("afinni-error-2"))[
  Two levels later there are #ex.afinni.levels.at(2).clusters clusters, a quarter of the
  size. The error of an affine model falls with the square of a cluster's width, so the
  colours have faded: #ex.afinni.levels.at(2).accepted clusters now pass (dark crosses) and
  retire as leaves. The others, where $f$ bends most, are cut again.
]

#filmstrip(
  (panel("afinni-error-0"), [level 0: #ex.afinni.levels.at(0).worst_error]),
  (panel("afinni-error-1"), [level 1: #ex.afinni.levels.at(1).worst_error]),
  (panel("afinni-error-2"), [level 2: #ex.afinni.levels.at(2).worst_error]),
  (panel("afinni-error-3"), [level 3: #ex.afinni.levels.at(3).worst_error]),
  (panel("afinni-error-4"), [level 4: #ex.afinni.levels.at(4).worst_error]),
  caption: [The clusters still in the queue at each level, coloured by the error of their
    models in units of the tolerance, with the worst error of each level. Accepted clusters
    leave the picture; the rest are cut and fitted again. After #ex.afinni.n_levels levels
    nothing is left.],
) <fig:errors>

The same run, seen as clusters and as the forest they form:

#board(
  (panel("afinni-level-0"), [level 0: #ex.afinni.levels.at(0).clusters clusters, #ex.afinni.levels.at(0).accepted accepted]),
  (panel("afinni-level-1"), [level 1: #ex.afinni.levels.at(1).clusters clusters, #ex.afinni.levels.at(1).accepted accepted]),
  (panel("afinni-level-2"), [level 2: #ex.afinni.levels.at(2).clusters clusters, #ex.afinni.levels.at(2).accepted accepted]),
  (panel("afinni-level-3"), [level 3: #ex.afinni.levels.at(3).clusters clusters, #ex.afinni.levels.at(3).accepted accepted]),
  (panel("afinni-level-4"), [level 4: #ex.afinni.levels.at(4).clusters clusters, #ex.afinni.levels.at(4).accepted accepted]),
  (panel("afinni-level-5"), [level 5: #ex.afinni.levels.at(5).clusters clusters, #ex.afinni.levels.at(5).accepted accepted]),
)

Each frame shows the clusters of one level in colour and the leaves of earlier levels
faded, beside the forest down to that level: a node ringed dark was accepted, one ringed
red was cut. The run ends with #ex.compare.afinni.leaves leaves after
#ex.compare.afinni.evals evaluations of $f$: the #ex.N0 initial points,
#ex.compare.afinni.generated new ones, and one at the centre of every cluster.

#lemma(title: [the fit is at least as good as a Taylor expansion])[
  For every cluster,
  $ sum_(x_j in X_C) abs(f(x_j) - fh(x_j))^2 <= sum_(x_j in X_C) abs(f(x_j) - f(c) - nabla f(c)^top (x_j - c))^2 . $
] <lem:taylor>

#proof[
  The Taylor expansion at $c$ is an affine function, and $fh$ minimizes the sum of squares
  over all affine functions.
]

So the error of a leaf can be reasoned about as a Taylor remainder, which is
$1/2 (x - c)^top H (x - c)$ to leading order: the bowl in the pictures. This is what the
next section uses to decide where to cut.

#implementation(title: [In the code])[
  `AffineLSQModel` (`src/components/local_models.py`) solves @eq:fit for every cluster of a
  batch at once, with a small ridge, $10^(-8) op("tr")(Sigma)$, so that a cluster lying in a
  subspace still has a solution. Vector-valued $f$ gets one row of $w$ per component, and a
  cluster passes only if every component passes; tolerances can be given per component.
  `error_metric="L2"` replaces the maxima in @eq:errors by root-mean-squares. Other local
  models (a Taylor model from oracle gradients, a constant, a quadratic) are registered
  under `SurrogateMode` and `LOCAL_MODELS`.
]

#pitfall(title: [What acceptance does not say])[
  The test is made on the points the model was fitted to, so it is optimistic, and $e_r$
  divides by the cluster's largest $abs(f)$, not by each point's own. An accepted leaf can
  therefore hold a point whose own relative error is large, where $f$ is near zero. The
  tolerance is a property of the training points; @sec:compare measures what it gives on
  the whole domain.
]

#takeaway[A cluster is accepted when its best affine model is within the tolerance on its
  own points; the pattern of the errors that remain is the curvature of $f$.]

= AFINNI: where to cut <sec:split>

#question[A cluster is rejected and must be cut in two. Along which direction?]

The obvious answer is the cluster's longest direction: that is $2$-means clustering, or a
cut across the first principal component @ding2004. It looks only at where the points
are. But a cut is useful only if the two halves are easier to fit than the whole, and that
depends on $f$.

#filmstrip(
  arrows: false,
  (panel("afinni-split-error"), [a rejected cluster: \ the error of its model]),
  (panel("afinni-split-plain"), [cut across the longest direction: \ children's worst error #ex.afinni.split.plain_error]),
  (panel("afinni-split-hessian"), [Hessian-informed cut: \ children's worst error #ex.afinni.split.hessian_error]),
  caption: [One rejected cluster of level #ex.afinni.split.level, with #ex.afinni.split.points
    points and worst error #ex.afinni.split.parent_error. Its error forms stripes: $f$ bends
    _across_ the cluster and hardly at all along it. A cut across the longest direction
    leaves both children with the full bend. A cut across the bend, #ex.afinni.split.angle
    degrees away, leaves children that are nearly flat.],
) <fig:split>

The stripes of the left frame are the level sets of the error, and the error is the
Taylor remainder $1/2 (x - c)^top H (x - c)$. So the cut should be made across the
direction in which that quadratic form grows fastest. The trick is to measure distances
with the curvature: stretch the cluster by $absH^(1 slash 2)$, so that a direction in which
$f$ bends strongly becomes long, and then cut the stretched cluster across _its_ longest
direction.

#theorem(title: [Hessian-informed two-means, @manuscript])[
  Let $f$ be a quadratic, possibly non-convex, with Hessian $H$, and let $hat(X)$ hold the
  centred points of a cluster as its columns. For the sum of the two children's errors
  $E = sum_(k = plus.minus 1) sum_(i in C_k) abs(f(x_i) - fh(x_i))$, with $fh$ the least-squares
  model of each child, the optimal continuous two-means indicator is the first eigenvector
  of $hat(X)^top absH hat(X)$.
] <thm:split>

The eigenvector of the $\#C times \#C$ matrix $hat(X)^top absH hat(X)$ is never formed. The
$d times d$ matrix $sum_i tilde(x)_i tilde(x)_i^top$ of the stretched points
$tilde(x)_i = absH^(1 slash 2)(x_i - c)$ has the same nonzero eigenvalues, and projecting
the stretched points on its first eigenvector $u_1$ gives the same indicator.

== Estimating the curvature from values alone

@thm:split needs $absH$, and the oracle gives only values of $f$. The estimate uses the
same pattern the pictures show: the points where the Taylor expansion is nearly exact lie
in a small sublevel set of the remainder around $c$, which is an ellipsoid with the shape
of $absH^(-1)$. Their scatter matrix is therefore proportional to $absH^(-1)$.

#story()
#step([A gradient from the nearest points], code: split-code, focus: (1,), window: 1, width: 36%, panel("afinni-hess-gradient"))[
  The Taylor expansion needs $nabla f(c)$. It is fitted by least squares to the
  #ex.afinni.split.n_grad points nearest to $c$ (green), where the expansion is most
  accurate, using their values and $f(c)$.
]

#step([The level set of the Taylor error], code: split-code, focus: (2, 3, 4), window: 1, width: 36%, panel("afinni-hess-levelset"))[
  The #ex.afinni.split.n_cov points with the smallest Taylor error (pink) trace the
  directions in which $f$ is nearly affine. The ellipse of their scatter matrix (pink) has
  the shape of $absH^(-1)$: long where $f$ is flat, short where it bends. The dark ellipse
  is the exact $absH^(-1)$ at $c$, scaled to the same area for comparison, which the
  algorithm never sees.
]

#step([Stretch, project, cut at the median], code: split-code, focus: (5, 6, 7, 8), window: 1, width: 36%, panel("afinni-split-hessian"))[
  In the metric $absH^(1 slash 2)$ the short axis of that ellipse becomes the long one. The
  arrow is the resulting direction $absH^(1 slash 2) u_1$; the points are sorted by their
  score $s_i$ along it and divided into a lower half $Cm$ (pink) and an upper half $Cp$
  (cyan).
]

#algorithm(caption: [Hessian estimate and Hessian-informed split of one rejected cluster
  (the manuscript's Algorithms 2 and 3): `CovarianceHessianSplit` in
  `src/components/splitting.py`.], pseudocode(splitting)) <alg:split>

#theorem(title: [the scatter of a level set estimates the Hessian, @manuscript])[
  Let $f$ be a convex, non-degenerate quadratic with Hessian $H$, and let
  $E_epsilon = {x : abs(f(x) - hat(f)_c (x)) <= epsilon}$, with $hat(f)_c$ the Taylor
  expansion at $c$. If $x_1, dots, x_n$ are independent and uniform on $E_epsilon$ and
  $hat(Sigma) = 1/n sum_i (x_i - c)(x_i - c)^top$, then for some constants $K$ and $kappa_0$, with
  probability $1 - e^(-v)$,
  $ norm(hat(Sigma)^(-1) - kappa_0 H)_2 lt.tilde K norm(H) sqrt((d + v) / n) . $
] <thm:hessian>

The factor $kappa_0$ does not matter: only the direction of the first eigenvector is used.
On the example the Hessian-informed cut left the children with a smaller worst error than
the cut across the longest direction in #ex.afinni.split.hessian_wins of the
#ex.afinni.split.rejected clusters that were split.

#implementation(title: [In the code])[
  - The gradient uses the $max(2 d, floor(Nmin slash 4))$ nearest points, the level set
    $N_"cov" = floor(Nmin slash 3)$ points.
  - The cut is at the _median_ of the scores by default, so the two children have equal
    size and the forest stays balanced. `cut="sign"` cuts at $s_i = 0$ as in the
    manuscript's Algorithm 2.
  - The assumptions of @thm:hessian rarely hold exactly: $f$ is not quadratic, not convex,
    and the points are not uniform on a level set. In practice the estimate recovers the
    dominant directions, which is what the cut needs. Where an exact gradient or Hessian
    is available, the modes `AFFINE_COV_HESS_EXACT_GRAD` and `AFFINE_EXACT_HESS` use it.
]

== Enough points to estimate anything <sec:augment>

#scene(side: left, width: 34%, panel("afinni-augment"))[
  A gradient, a Hessian and two children's models cannot be estimated from a handful of
  points. Before a rejected cluster with fewer than $2 Nmin$ points is cut, it is given
  new ones, up to $max(2 \#C, 2 Nmin)$, so that each child starts with at least $Nmin$.
  Here a cluster of #ex.afinni.augment.before points (dark) receives
  #ex.afinni.augment.new new ones (pink).

  A new point is a random convex combination of two of the cluster's points,
  $tilde(x) = lambda x_i + (1 - lambda) x_j$ with $lambda$ uniform in $[0, 1]$, and $f$ is
  evaluated there. This is one array operation on a GPU and needs no knowledge of the
  cluster's boundary.
]

#lemma(title: [new points stay inside])[
  Every new point lies in the convex hull of its cluster's points (dashed in the picture).
] <lem:hull>

#pitfall(title: [The hull is not the cell])[
  @lem:hull is the method's limit as well as its safety. New points never leave the hull,
  so the space between the hulls of neighbouring clusters is never sampled, and they
  concentrate towards the middle of the hull. When many points must be generated from few,
  the leaves cover less of the domain, and more queries fall outside every trust region
  (@sec:trust). The remedy within AFINNI is a larger initial sample. ART samples each
  region up to its boundary instead (@sec:regions).
]

#takeaway[Cut across the direction in which $f$ bends, not across the direction in which
  the points spread: stretch the cluster by $absH^(1 slash 2)$, estimated from the points
  where the Taylor expansion is nearly exact, and cut the stretched cluster in half.]

= AFINNI: where a leaf can be trusted <sec:trust>

#question[A query point arrives. Which leaf should answer it, and should the answer be
  believed?]

AFINNI keeps no tree and no boundaries, only the leaves. Each leaf knows where its model
was tested: at its own points. So each leaf carries a region that contains them.

#story()
#step([An ellipsoid around a leaf's points], code: trust-code, focus: (1, 2, 3, 4), width: 36%, panel("afinni-trust-one"))[
  The scatter matrix $Sigma$ of the leaf's #ex.afinni.trust_points points gives the shape:
  the ellipsoid ${x : (x - c)^top Sigma^(-1) (x - c) <= 1}$ is long where the points
  spread. Scaling it by the largest value $alpha$ that $(x_i - c)^top Sigma^(-1) (x_i - c)$
  takes puts the farthest point (ringed) exactly on the boundary and every other point
  inside.
]

#algorithm(caption: [The trust ellipsoid of an accepted cluster (the manuscript's Algorithm 5): `CovarianceEllipsoid` in
  `src/components/trust_regions.py`. $gamma$ is `q_reg`.], pseudocode(trust)) <alg:trust>

#lemma(title: [the trust region contains the leaf's points])[
  With $T$ from @alg:trust, $(x_i - c)^top T (x_i - c) <= 1$ for every point $x_i$ of the
  cluster, with equality for the farthest one.
] <lem:trust>

#proof[
  $(x_i - c)^top T (x_i - c) = (x_i - c)^top Sigma^(-1) (x_i - c) slash alpha$, and $alpha$ is the
  largest numerator.
]

A minimum-volume enclosing ellipsoid @todd2016 would be tighter, but needs an iterative
optimization per leaf; this one is a single eigendecomposition, batched over all accepted
clusters.

#scene(side: left, width: 40%, panel("afinni-inference"))[
  The ellipsoids do two jobs at inference. They are a _metric_: the distance from a query
  $x$ to leaf $i$ is $d_i = (x - c_i)^top T_i (x - c_i)$, which counts a step along a
  leaf's long axis as short. And they are a _flag_: if $d_i > 1$ for every candidate, no
  leaf has tested its model near $x$.

  *A*: the nearest centre in Euclidean distance (grey cross) is not the leaf that answers.
  The long leaf's centre is farther away, but $x$ lies well inside its ellipsoid
  ($d_nu = #ex.afinni.queries.A.d_nu$). \
  *B*: $x$ lies inside #ex.afinni.queries.B.inside ellipsoids; the smallest $d_i$ decides. \
  *C*: $x$ lies inside none ($d_nu = #ex.afinni.queries.C.d_nu$). The nearest leaf still
  answers, and the answer is returned _flagged_ (red).
]

#algorithm(caption: [Inference (the manuscript's Algorithm 6): `AFINNISurrogate.predict`
  with `predictor="knn"`.], pseudocode(inference)) <alg:inference>

Searching all $N_c$ leaves in their own metrics would cost $O(N_c d^2)$ per query. The
search is therefore done in two stages: an exact Euclidean search for $kappa$ candidates
(`src/neighbors/knn.py`), then the $kappa$ trust-region distances. $kappa$ should grow with
the dimension and with the anisotropy of the leaves; the manuscript uses $kappa = 2 d^2$.

#implementation(title: [In the code])[
  `predict(X, k_nn=1, coarse_factor=10)` searches $kappa =$ `k_nn * coarse_factor`
  candidates and returns the predictions with a mask of the flagged queries. With
  `k_nn > 1` the best `k_nn` leaves are blended with weights $1 slash d_i^2$, which smooths
  the jumps between leaves. On the example, queries A, B and C are answered with
  #ex.afinni.queries.A.pred, #ex.afinni.queries.B.pred and #ex.afinni.queries.C.pred, where
  $f$ is #ex.afinni.queries.A.f, #ex.afinni.queries.B.f and #ex.afinni.queries.C.f.
]

#pitfall(title: [A trust region is not a certificate])[
  Inside a trust region the model was tested at the leaf's points, not everywhere. The
  flag says "no leaf has data near here"; its absence does not bound the error. In high
  dimension the ellipsoids cover a vanishing share of a box, so the flag is informative
  only if the sample lies where the queries will. The manuscript's _iterative refinement_
  uses the flag as a sampling rule: draw new initial points, keep those outside every
  trust region, and train again.
]

#takeaway[A leaf answers for the ellipsoid that contains its points; a query is answered
  by the leaf nearest in the leaves' own metrics, and flagged if it is inside none.]

= ART: a region for every node <sec:regions>

#question[AFINNI never knows where a cluster ends. What changes if every node of the tree
  owns a precise piece of the domain?]

Three things: the node can be sampled up to its boundary, the leaves tile the domain with
no gaps, and a query can be sent to its leaf by a few comparisons. This is ART, the
adaptive regression tree @art. It needs what AFINNI does not, a domain, given as a box
(`domain=(lower, upper)`); all its geometry is done in the coordinates $u in [0, 1]^d$ of
that box.

#definition(title: [regions, planes and path codes])[
  The root's _region_ is the domain, $Omega = [0, 1]^d$ in $u$. A node $C$ that is split
  by the plane $(a, z)$, $norm(a) = 1$, has the children
  $ Omega_(Cm) = Omega_C inter {u : a^top u <= z}, quad Omega_(Cp) = Omega_C inter {u : a^top u >= z} , $
  and a point with $a^top u = z$ belongs to $Cp$. A region is therefore a convex polytope
  ${u : A u <= bold(z)}$: the box, and one row $plus.minus (a, z)$ of $(A, bold(z))$ for every
  ancestor. The _path code_ of
  the root is $pi = 1$, and the children of node $pi$ are $2 pi$ (for $Cm$) and $2 pi + 1$
  (for $Cp$): one binary digit per level.
] <def:region>

#board(
  (panel("art-level-0"), [depth 0: the domain, #ex.settings.art.node_points points]),
  (panel("art-level-1"), [depth 1: #ex.art.levels.at(1).nodes nodes, #ex.art.levels.at(1).new_points new points]),
  (panel("art-level-2"), [depth 2: #ex.art.levels.at(2).nodes nodes]),
  (panel("art-level-3"), [depth 3: #ex.art.levels.at(3).nodes nodes, #ex.art.levels.at(3).accepted accepted]),
  (panel("art-level-4"), [depth 4: #ex.art.levels.at(4).nodes nodes, #ex.art.levels.at(4).accepted accepted]),
  (panel("art-level-5"), [depth 5: #ex.art.levels.at(5).nodes nodes, #ex.art.levels.at(5).accepted accepted]),
  (panel("art-level-6"), [depth 6: #ex.art.levels.at(6).nodes nodes, #ex.art.levels.at(6).accepted accepted]),
  (panel("art-partition", width: 44%), [the #ex.compare.art.leaves leaves]),
)

Each frame shows the nodes of one depth: a region outlined dark was accepted, one outlined
red is cut in the next frame, and leaves of earlier depths are faded. The dots are the
node's points: dark ones were inherited from its parent, open ones are new. Compare with
AFINNI's frames in @sec:fit: the pieces now have edges, and they are refilled with points
at every depth instead of thinning out. The run ends with #ex.compare.art.leaves leaves at
depths up to #ex.art.depth, after #ex.compare.art.evals evaluations of $f$.

#algorithm(caption: [ART on one batch of nodes: `ARTPlan.process_batch` in
  `src/art/plan.py`. The notes in code type are the leaf's recorded `terminal_reason`.],
  pseudocode(art)) <alg:art>

== A fixed budget of points in every node

A node is fitted on $n = ceil(m p)$ points, where $p = d + 1$ is the number of
coefficients of an affine model and $m$ is the `sample_multiplier`: here $m = #ex.settings.art.sample_multiplier$ and
$n = #ex.settings.art.node_points$. It inherits the points of its parent that fall on its
side, which cost nothing, and draws the rest uniformly in its own region. A node deep in
the tree is thus fitted and tested on as many points as the root, concentrated in its
small region.

#story()
#step([A region is the box and its ancestors' planes], code: art-code, focus: ("in", 2), window: 0, width: 35%, panel("art-polytope"))[
  The node with path code #ex.art.sampling.path, at depth #ex.art.sampling.depth: its
  region (shaded) is bounded by the box and by the #ex.art.sampling.rows planes of its
  ancestors (lines), one row of $(A, bold(z))$ each. It inherited #ex.art.sampling.inherited
  points (dark) and needs #ex.art.sampling.new new ones (pink), anywhere in the region,
  including next to its boundary.
]

#step([Hit-and-run: a random walk that stays inside], code: hitrun-code, focus: (2, 3, 4), width: 35%, panel("art-hit-and-run"))[
  Drawing uniformly in a polytope is done by walking. From a point inside (dark), choose a
  random direction; the region cuts the line through the point in a chord (dotted); jump
  to a uniform point of the chord. Each step only needs the two ends of the chord, which
  come from one product $A g$. After a few steps the walker (pink) has forgotten where it
  started.
]

#algorithm(caption: [Hit-and-run in a polytope @smith1984 @lovasz2003:
  `HitAndRunSampler` in `src/art/sampling.py`. One chain per new point, started at a
  random inherited point.], pseudocode(hitrun)) <alg:hitrun>

#proposition(title: [hit-and-run samples uniformly, @smith1984])[
  The uniform distribution on a bounded convex region with interior is the unique
  stationary distribution of the hit-and-run walk, and the walk converges to it from any
  interior point.
] <prop:hitrun>

#pitfall(title: [Finitely many steps])[
  @prop:hitrun is a statement about the limit. A chain of $tau$ steps in a long thin
  region may not have crossed it yet, and its points then cluster near their starts. The
  code therefore _checks_: it compares the points of each node halfway through the walk
  and at its end, and if their mean has drifted by more than `mixing_z` standard errors,
  it continues those chains, doubling $tau$ from `walk_steps` up to `max_walk_steps`. A
  node that still drifts is fitted anyway and, if it becomes a leaf, is reported with the
  `validation_status` `insufficient_sampling`, so a model with unmixed regions cannot pass
  silently.
]

#invariant(title: [random numbers belong to nodes])[
  Every random number ART draws is a hash of the run's seed, the node's root and path
  code, the purpose of the draw and a counter (`src/art/rng.py`). No generator is
  advanced from batch to batch. The points a node draws, and so its fit and its plane, are
  therefore the same whatever batch the node is in, whichever rank processes it, and
  whenever it is stolen (@sec:mpi).
] <inv:rng>

#implementation(title: [In the code])[
  - `roots="grid"` or `"voronoi"` starts from several roots instead of one: a regular grid
    of the domain, or the Voronoi cells of the initial clustering's centres. More roots
    give the ranks independent work from the first level on.
  - A root box is sampled exactly, without a walk.
  - `validation_samples` draws a second, independent set of points in every node, on which
    an accepted model is checked again before it becomes a leaf.
  - On a GPU the whole walk of every chain of a batch runs in one kernel
    (`src/kernels/hit_and_run.cu`), one thread per chain.
]

#takeaway[An ART node owns a polytope, the box cut by its ancestors' planes, and is
  fitted on a fixed number of points drawn uniformly inside it by hit-and-run.]

= ART: choosing the plane <sec:plane>

#question[A node is rejected. Of all the planes that cut its region in two, which one?]

AFINNI computes its cut from the curvature in closed form. ART _searches_ for it: the
best plane is the one after which two affine models, one per side, fit the node's points
best, which is what the children's least-squares fits will minimize.

#definition(title: [the split objective])[
  For a node with points $u_i$ and values $y_i = f(x_i) slash atol$, the _hard_ objective of
  a plane $(a, z)$ is the squared error of the two children's least-squares fits,
  $ "SSE"(a, z) = min_(phi_(-1), phi_1 "affine") sum_i cases(
    (y_i - phi_(-1)(u_i))^2 & "if" a^top u_i < z,
    (y_i - phi_1(u_i))^2 & "if" a^top u_i >= z ) $ <eq:hard>
  among planes that leave at least $p$ points on each side.
] <def:objective>

#filmstrip(
  arrows: false,
  (panel("art-split-before"), [a rejected node: the error of its one model]),
  (panel("art-split-candidates"), [candidate directions; the best cut found (dashed), then refined (solid)]),
  (panel("art-split-after"), [the errors of the two children's models]),
  caption: [The split search on the node with path code #ex.art.split.path, with
    #ex.art.split.points points. One model leaves a mean squared error of
    #ex.art.split.sse_parent (in units of $atol^2$). The search tries cuts along
    #ex.art.split.directions directions (arrows; the red one is AFINNI's Hessian-informed
    direction) and finds a plane with error #ex.art.split.sse_bins; the soft search turns
    and shifts it to reach #ex.art.split.sse_final.],
) <fig:search>

== A coarse search, then a smooth one

The hard objective is a staircase: it changes only when the plane crosses a point, and
has no useful gradient. ART handles it in two stages.

#story()
#step([Many cuts along a few directions], code: search-code, focus: (1, 2, 3, 4), window: 0, width: 38%, panel("art-split-candidates"))[
  First a search by enumeration. Along each of $n_"dir" = #ex.art.split.directions$ directions
  the node's points are sorted, and $n_"bins" - 1 = #(ex.art.split.n_bins - 1)$ cuts are
  tried, at equal fractions of the points. Every candidate's two fits come from running
  sums over the bins, so a direction costs one pass over the points. The best candidates
  are refitted exactly; the winner is the dashed line.
]

#step([Soft gates make the objective smooth], code: search-code, focus: (5, 6), window: 0, width: 38%, panel("art-split-gates"))[
  Then a search by descent. Replace "left or right" by a _gate_: a point at signed
  distance $a^top u - z$ from the plane belongs to the right child with weight
  $sigma((a^top u - z) slash h)$, where $sigma$ is the logistic function and $h$ the gate
  width (the dotted band). Far from the plane the points are pink or cyan as before; in
  the band they are shared. The two models become weighted fits, and their error a smooth
  function of $(a, z)$.
]

#scene(width: 46%, panel("art-objective"))[
  Both objectives along the chosen normal, as functions of the offset $z$. The hard one
  (dark) is a staircase with a step at every point. The soft one (cyan) follows it and
  can be differentiated: an optimizer can slide down it, turning the normal as well as
  moving the offset, which the enumeration could not do. The red line is the plane that
  was chosen and the band is the gate width around it.
]

#definition(title: [the soft objective])[
  With weights $W_1 (s) = delta + (1 - 2 delta) sigma(s)$ and $W_(-1) (s) = delta + (1 - 2 delta) sigma(-s)$,
  $s_i = (a^top u_i - z) slash h$, a small floor $delta$ and a fixed ridge $lambda$,
  $ J_h (a, z) = min_(B_(-1), B_1) 1/n [ sum_i sum_(k = plus.minus 1) W_k (s_i) (y_i - B_k^top phi_i)^2 + lambda (norm(B_(-1))^2 + norm(B_1)^2) ] , $ <eq:soft>
  where $phi_i = (u_i, 1)$ and $B_k$ are the coefficients of the two models.
] <def:soft>

The minimization over the models is a pair of weighted least-squares problems with a
closed-form solution, so only the plane is optimized: this is _variable projection_
@golub1973. Its gradient needs no derivative of the models.

#lemma(title: [the gradient of the soft objective])[
  Let $r_(k, i) = (y_i - B_k^top phi_i)^2$ at the minimizing models. Then
  $ nabla_a J_h = 1/(n h) sum_i (r_(1, i) - r_(-1, i)) (1 - 2 delta) sigma'(s_i) u_i, quad
    partial_z J_h = -1/(n h) sum_i (r_(1, i) - r_(-1, i)) (1 - 2 delta) sigma'(s_i) . $
] <lem:gradient>

#proof[
  $J_h$ is a minimum over $B$ of a function that is differentiable in $(a, z)$, and the
  minimizer is unique because of the ridge. By the envelope theorem the derivative of the
  minimum is the partial derivative at the minimizer: only the weights depend on
  $(a, z)$.
]

The sum has a clear meaning: a point in the band pulls the plane towards the side whose
model fits it worse. Each step moves along the negative gradient, projected on the tangent
plane of the sphere $norm(a) = 1$ and followed by renormalizing $a$, with a backtracking line
search that keeps the models frozen.

#lemma(title: [frozen models still give descent])[
  Let $F(theta, B)$ be the function minimized over $B$ in @eq:soft, at plane $theta = (a, z)$
  and models $B$, and
  $B_t$ the minimizing models at the current plane $theta_t$. If a trial plane $theta'$
  satisfies $F(theta', B_t) < J_h (theta_t)$, then $J_h (theta') < J_h (theta_t)$.
] <lem:frozen>

#proof[
  $J_h (theta') = min_B F(theta', B) <= F(theta', B_t) < J_h (theta_t)$.
]

So a trial step is judged with the current models, which costs one pass over the points,
and the models are refitted only after a step is accepted. This holds because the same
weights, floor and ridge appear in the fit and in the objective.

#invariant(title: [the stored plane defines the children])[
  After the search, $z$ is moved to the score of the first point on the right, and the
  children are _defined_ by the stored predicate $a^top u >= z$. The code verifies that the
  predicate reproduces the cut, point by point, and raises an error otherwise. Training
  and routing use the same function `plane_scores` to evaluate it, so a point can never be
  on one side when it is fitted and on the other when it is routed.
] <inv:predicate>

#implementation(title: [In the code])[
  - The soft result is kept only if its two _hard_ fits are better than the enumeration's.
    The soft search can only improve a split: on the example it did for
    #ex.art.split.improved of the #ex.art.split.searched nodes that were split.
  - $h$ is `soft_temperature` $= #ex.art.split.temperature$ times the standard deviation
    of the scores $a^top u$, and at most `soft_iters` $= #ex.art.split.soft_iters$ steps
    are taken.
  - `split_search="bins"` stops after the enumeration, and `"gap"` cuts along the
    Hessian-informed direction near the median without any search.
  - The values are divided by the tolerance of their component first (the larger of
    $atol$ and $rtol$ times the node's largest $abs(f)$), so that for
    vector-valued $f$ a component is weighted by how far it is from _its_ tolerance.
]

#algorithm(caption: [ART's split search: `_bins_search` and `_soft_refine` in
  `src/art/plan.py`, on the primitives of `src/art/split_search.py`.],
  pseudocode(search)) <alg:search>

== From a tree to an answer

#scene(width: 62%, panel("art-route"))[
  A query is _routed_: start at the root, and at every split node evaluate
  $a^top u - z$ and go right if it is non-negative. The picture follows one query (black
  dot) through the nested regions it falls in, down to its leaf at depth
  #ex.art.route.depth; the path code of the leaf, #ex.art.route.path, is the record of the
  turns taken. The leaf's model gives the answer.
]

#figure(panel("art-tree"), caption: [The tree of the running example with its
  path codes in binary. A node's code is its parent's with one digit appended, 0 for the
  side $a^top u < z$ and 1 for the side $a^top u >= z$; the number of digits after the
  leading 1 is the depth.]) <fig:tree>

#theorem(title: [the leaves partition the domain])[
  Every point of the domain is routed to exactly one leaf, and it lies in that leaf's
  region. The cost is one inner product per level: $O(d dot "depth")$.
] <thm:partition>

#proof[
  At each split node the predicate $a^top u >= z$ is either true or false, so exactly one
  child is entered, and by @def:region the point lies in that child's region if it lay in
  the parent's. The root's region is the whole domain.
]

#implementation(title: [In the code])[
  The `RoutingTable` (`src/art/routing.py`) stores the tree as arrays: for every node its
  two children, its plane and, for a leaf, its row in the leaf arrays. It is assembled on
  rank 0 after training, from the split records $(pi, a, z)$ of all ranks, and _checked_:
  every split has both children, every node but a root has a parent, no code appears
  twice. A query outside the domain is routed from its nearest point of the domain and
  returned flagged, as AFINNI flags a query outside its trust regions.
]

#takeaway[ART picks the plane whose two children fit best: an enumeration over cuts along
  a few directions, then gradient descent on a smooth version of the same objective, kept
  only if it helps. The planes are stored, so the leaves tile the domain and a query walks
  to its leaf.]

= Side by side <sec:compare>

#question[The same function, the same tolerance, two algorithms. What does each one give?]

#scene(side: left, width: 46%, panel("slice"))[
  First along one line across the valley, $x_2 = #ex.compare.slice_x2$. The dark curve is
  $f$, the grey band is $f plus.minus atol$, and each coloured segment is one leaf's
  affine model. Both surrogates are piecewise affine and stay inside the band almost
  everywhere; both have small jumps where one leaf hands over to the next. The pieces are
  short where $f$ curves and long where it is nearly straight: that is the adaptivity.
]

#filmstrip(
  arrows: false,
  (panel("error-afinni", width: 68%), [AFINNI: #ex.compare.afinni.leaves leaves]),
  (panel("error-art", width: 68%), [ART: #ex.compare.art.leaves leaves]),
  caption: [The error $abs(f - fh) slash atol$ over the whole domain, on a
    #ex.compare.grid $times$ #ex.compare.grid grid: #swatch(rgb(col.error.at(0))) pale to
    #swatch(rgb(col.error.at(3))) blue within the tolerance, #swatch(rgb(col.error.at(4)))
    to #swatch(rgb(col.error.at(5))) red beyond it. Left, AFINNI's centres (crosses) and,
    hatched, the queries flagged as outside every trust region. Right, ART's leaves.],
) <fig:error>

#table(
  columns: (1fr, auto, auto),
  inset: (x: 8pt, y: 5pt),
  align: (left, right, right),
  table.header[On the running example, $atol = #ex.alpha$][AFINNI][ART],
  [leaves], [#ex.compare.afinni.leaves], [#ex.compare.art.leaves],
  [evaluations of $f$], [#ex.compare.afinni.evals], [#ex.compare.art.evals],
  [of which at new points, besides the centres],[#ex.compare.afinni.generated], [#ex.compare.art.generated],
  [share of the domain within the tolerance], [#ex.compare.afinni.within_percent%], [#ex.compare.art.within_percent%],
  [share of the domain flagged], [#ex.compare.afinni.outside_percent%], [#ex.compare.art.outside_percent%],
  [largest error on the grid, in units of $atol$], [#ex.compare.afinni.max_error], [#ex.compare.art.max_error],
)

Neither surrogate is within the tolerance everywhere, and neither claims to be: the
tolerance was enforced at training points, and between them a model can exceed it,
mostly near the edges of leaves. What differs is what each algorithm can say about it.

#cards(
  columns: 2,
  ([AFINNI], [Works from the sample it is given, wherever that lies, and evaluates $f$
    beyond it only inside the hulls of its clusters. Its flag marks the queries no leaf has data for; here they are
    at the rim of the box, beyond the outermost points. Inside the trust regions
    #ex.compare.afinni.within_inside_percent% of the grid is within the tolerance.]),
  ([ART], [Covers the whole domain by construction and flags nothing inside it. Because
    its leaves partition a known domain, an independent check is possible: `audit` draws
    fresh uniform points and bounds the share of the domain that violates the tolerance.]),
)

#proposition(title: [the audit bound, @clopper1934])[
  Let $k$ of $n$ independent uniform points of the domain violate the tolerance. With
  confidence $eta$, the share of the domain that violates it is at most the
  Clopper--Pearson bound, the $eta$-quantile of the $"Beta"(k + 1, n - k)$ distribution;
  with $k = 0$ this is $1 - (1 - eta)^(1 slash n)$.
] <prop:audit>

On the example, `audit` with #ex.art.audit.points fresh points finds
#ex.art.audit.violations violations (#ex.art.audit.rate_percent%), so with
#ex.art.audit.confidence_percent% confidence at most #ex.art.audit.bound_percent% of the
domain violates the tolerance. The bound holds for the trained model and the uniform
measure; it does not bound the error at any particular point.

#note(title: [Which one])[
  The two are not competitors so much as answers to different situations. Use AFINNI when
  the data come first: a large sample from a simulation, no natural box around it, and the
  need to know when a query leaves the region the data cover. Use ART when the domain
  comes first: a box that must be covered, an oracle that can be called anywhere in it,
  and the wish for a tree, exact routing and an audit. `examples/example05_afinni_vs_art.py`
  runs both on the same problems.
]

#takeaway[Both meet the tolerance on almost all of the domain with a few dozen affine
  pieces; AFINNI flags where it has no data, ART covers everything and can be audited.]

= Batches, memory and reproducible sums <sec:batches>

#question[The engine handles thousands of clusters at once on a GPU. How, without a loop
  over clusters, and without results that change from run to run?]

By keeping the layout of @fig:queue all the way down. A batch is three arrays, the points
`X`, the values `F`, and the cluster sizes, with every cluster a run of consecutive rows.
Every step of @alg:afinni and @alg:art is then one of two kinds of array operation.

#cards(
  columns: 2,
  ([Segmented reductions], [A sum, maximum or scatter matrix over the rows of each
    cluster: centroids, $Sigma$, the errors $e_a$. One pass over the rows produces one
    result per cluster.]),
  ([Small batched solves], [A $d times d$ or $p times p$ problem per cluster: the
    least-squares fit, an eigendecomposition for the cut or the trust ellipsoid. All
    clusters are solved together.]),
)

#block(breakable: false, table(
  columns: (1fr, 1.3fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[Constraint][What the engine does],
  [the points of a level can exceed the device's memory], [keeps the queue in host memory and moves one batch at a time to the device],
  [a batch holds as many whole clusters as fit], [caps `batch_size` (points) at 65% of the free device memory divided by `plan.bytes_per_point()`],
  [a batch's points must be released as early as possible], [hands them to the plan (`DeviceBatch.take`), which frees them before it augments and cuts],
  [the host can run out of memory too], [checks the available memory before the queue grows, and raises `MemoryError` with advice],
))

== The same bits every time

Floating-point addition is not associative. A sum computed by many GPU threads in
whatever order they finish differs from run to run in its last bits, and a last bit can
decide whether a cluster passes its tolerance, and so the shape of the whole tree.

#invariant(title: [deterministic sums])[
  Every sum on the training path is taken in a fixed order that depends only on the
  cluster it belongs to. On the GPU, a cluster's rows are summed sequentially in chunks
  counted from its own first row, and the chunks are then added in order
  (`segment_cross`, `reduce_sum_1d`, `det_matmul`, `spd_solve` in `src/numerics/`). No
  thread adds into a shared total, and no library routine chooses its algorithm by the
  size of the batch.
] <inv:sums>

#proposition(title: [what does not change the result])[
  + Repeated runs in one process on the same backend give bit-identical models.
  + An ART tree, as the set of its planes by path code, does not depend on the batch size,
    the number of ranks or the schedule of steals.
  + An AFINNI forest is fixed by the seed, the batch size and the number of ranks, when
    work stealing is off.
] <prop:determinism>

#proof[
  (1) follows from @inv:sums and the seeded random streams. (2) A node's points depend only on its key (@inv:rng), its
  sums only on its own rows (@inv:sums), and its plane only on those; by induction from
  the roots, so does every node. (3) AFINNI's new points are drawn from a generator
  seeded per batch, so they depend on which clusters share a batch; that is fixed by the
  batch size and the scatter over ranks, but a steal moves clusters between batches.
]

#pitfall(title: [CPU and GPU])[
  A CPU run and a GPU run agree to rounding, about $10^(-12)$, not bit for bit: their sums
  are taken in different orders. The tests compare them with a tolerance that tight, and
  require leaf counts and evaluation counts to match exactly.
]

#takeaway[Clusters are runs of rows, batches are runs of clusters, and every sum is
  taken in an order that only its own cluster determines: fast on a GPU, and the same on
  every run.]

= Many ranks <sec:mpi>

#question[Refinement is uneven: one region of the domain may need ten times the levels of
  another. How do many processes share work that nobody can predict?]

Rank 0 evaluates $f$ on the initial points, builds the initial queue and cuts it into $P$
contiguous runs with equal numbers of clusters, one per rank. From then on every rank
runs @alg:engine on its own queue. A static division cannot stay balanced, because the
work a cluster will generate is not known until it has been fitted: a rank whose
clusters are accepted early runs dry while another is many levels deep.

So idle ranks _steal_ @blumofe1999. The roles are a thief, with an empty queue, and a
victim, with work to spare.

#scene(width: 46%, diagram(
  // Labels sit close above their own arrow, and the rows are far enough apart that each
  // label is nearer its arrow than the arrow above it.
  spacing: (9em, 2.1em),
  label-sep: 2pt,
  node((0, 0), small[*thief* $s$], name: <t0>),
  node((1, 0), small[*victim* $r$], name: <v0>),
  node((0, 1), [], name: <t1>),
  node((1, 1), [], name: <v1>),
  node((0, 2), [], name: <t2>),
  node((1, 2), [], name: <v2>),
  node((0, 3), [], name: <t3>),
  node((1, 3), [], name: <v3>),
  node((0, 4), [], name: <t4>),
  node((1, 4), [], name: <v4>),
  edge(<t0>, <t4>, stroke: 0.5pt + luma(150)),
  edge(<v0>, <v4>, stroke: 0.5pt + luma(150)),
  edge(<v1>, <t1>, "-|>", label: small[load $L_r$ (gossip)], stroke: (dash: "dashed")),
  edge(<t2>, <v2>, "-|>", label: small[request]),
  edge(<v3>, <t3>, "-|>", label: small[grant + clusters, or denial]),
  edge(<t4>, <v4>, "-|>", label: small[new load (to all)], stroke: (dash: "dashed")),
))[
  Each rank publishes its load, the points waiting in its queue, to all others at every
  level boundary and after every steal: an 8-byte message, never waited for. A thief asks the rank it believes
  is the busiest. The victim looks at requests only _between batches_, so it is never
  interrupted in the middle of one, and answers every request with a grant or a denial.
  A grant is followed by the clusters themselves: whole clusters from the _tail_ of the
  victim's queue, the ones it would have reached last.
]

#algorithm(caption: [Decentralized work stealing, run by every rank $i$ (the manuscript's
  Algorithm 7): `WorkStealer` in `src/engine/work_stealing.py`.],
  pseudocode(steal)) <alg:steal>

#invariant(title: [the stealing protocol])[
  + *One reply.* Every request receives exactly one grant or denial. A thief waits for
    that explicit reply and for nothing else, so a stale view of the loads can delay it
    but not deadlock it.
  + *Only unstarted work.* A grant takes whole clusters from behind the current batch and
    always leaves the victim at least one. A cluster is processed by exactly one rank.
  + *A cluster travels complete.* A stolen cluster carries its points, values, depth and,
    for ART, its root, path code and the rows of its region. The thief needs nothing else from
    the victim, then or later.
  + *Nobody leaves early.* A rank retires only when every rank is idle: it joins a
    non-blocking barrier and keeps answering requests until all have joined.
] <inv:steal>

== One recorded run

Stealing depends on timing, so its picture is of one recorded run: ART on the running
example with a tighter tolerance, $atol = #ex.steal.atol$, on #ex.steal.ranks ranks. ART
starts from a single root, so rank 0 holds all the work at first and every other rank
must steal.

#figure(panel("steal-timeline"), caption: [Who was working when. A rank's bar is coloured
  while it has nodes and pale while it is idle; each arrow is a grant, from the victim
  to the thief. Rank 0 builds the first levels alone (including the one-time compilation
  of the CPU kernels); once its queue is long enough to share, the other ranks take work
  from it and, later, from each other: #ex.steal.steals grants in all.]) <fig:timeline>

#scene(side: left, width: 36%, panel("steal-owners"))[
  The same run as a map: each of the #ex.steal.leaves leaves, coloured by the rank that
  built it (#ex.steal.leaves_per_rank.map(str).join(", ") leaves per rank). A stolen
  node takes its whole subtree with it, so each rank's leaves form patches of the
  domain.

  The tree itself is the one a single process builds. By @prop:determinism every node's
  points and plane depend only on its key, so it does not matter which rank processed it;
  the recording script checks that every plane of this tree, by path code, is bit-identical
  to the tree built with stealing switched off.
]

#implementation(title: [In the code])[
  - The threshold $w_*$ and the steal size $w_s$ are counted in points: AFINNI grants
    when more than $10^6$ unstarted points remain, ART when a few nodes' worth do
    (`steal_min_load`, `steal_size`). A grant gives about $max(w_s, "half the unstarted work")$.
  - ART's split search can be long, so it also answers requests _inside_ a batch, through
    `batch.progress()`, still giving away only clusters behind the batch.
  - Under MPI an exception on one rank would leave the others waiting in a collective
    call. `run_collective` therefore logs the rank, phase and traceback and aborts every
    rank; in a single process the exception simply propagates.
  - `python -m src.infra.monitor` shows each rank's level, queue and steals live, from
    files the ranks write, without any MPI traffic.
]

#takeaway[Ranks start with equal shares and stay busy by stealing unstarted clusters from
  the tail of the busiest rank's queue; every request gets one answer, a cluster travels
  complete, and for ART the tree does not depend on who built it.]

= How we know it is right <sec:verify>

#question[Thousands of batched operations, two backends, many ranks. What stands between
  a change to the code and a wrong surrogate?]

#cards(
  columns: 3,
  ([Definitions], [Each component is tested against its formula on small cases: the fit
    recovers an affine function exactly, the errors match @eq:errors, the trust
    ellipsoid against @lem:trust, a plane against the cut it must reproduce.]),
  ([Both plans, one contract], [`test_plans.py` holds AFINNI and ART to the `Plan`
    protocol, checks their batch boundaries, and checks that an AFINNI batch's points are
    released before splitting.]),
  ([Differential regression], [`scripts/regression_check.py` trains 52 small cases
    covering every mode, the `L2` and `Linf` metrics and every split search, and stores every array of every model.
    A refactor must reproduce them bit for bit on a CPU, on one rank and on two.]),
  ([CPU against GPU], [The GPU suite runs the same tests and compares with a tolerance
    of $10^(-12)$ ($10^(-10)$ between two separate trainings); leaf counts and evaluation counts must match exactly.]),
  ([Many ranks], [`tests/mpi` runs real `mpiexec` jobs: scatter and gather, stealing
    with gradients, ART's tree against a single-rank run, and that a failure on one rank
    aborts all.]),
  ([Structure], [`test_layering.py` enforces the import layers; `test_kernel_sources.py`
    checks, on any machine, that every CUDA kernel has a caller and every call a kernel.]),
)

Three checks are built into the library itself, because they concern a particular model
more than the code. The routing table is validated when it is assembled and again when a
model is loaded (@sec:plane). ART reports every leaf's `terminal_reason` and
`validation_status`, so a tree with fallback leaves or unmixed samples says so. And
`audit` measures a trained ART model on fresh points (@prop:audit).

This companion's figures are a test of the same kind: the script that draws them rebuilds
AFINNI's forest from the recorded queue, ART's polytopes from the stored planes, the
affine fits, the Hessian estimate and the inference rule from their formulas, and stops
if any of them disagrees with the library.

#takeaway[Formulas are tested on small cases, whole models are compared bit for bit
  across refactors, backends and ranks, and a trained model carries its own diagnostics.]

= Using it <sec:using>

// The settings are filled in from `ex`, so the listing shows the running example's values.
#raw(block: true, lang: "python", ```
from src.config import ARTSettings, TrainingConfig
from src.core.evaluation import Evaluator
from src.surrogate import AFINNISurrogate

evaluator = Evaluator(f_fn=f)                    # f maps (N, d) points to (N, fd) values

afinni = AFINNISurrogate()                       # AFINNI: from a sample X0
afinni.train(X0, TrainingConfig(atol={atol}, num_initial_clusters={n_init}, min_pts_per_cluster={n_min}), evaluator)
values, outside = afinni.predict(queries, k_nn=1, coarse_factor={kappa})

art = AFINNISurrogate()                          # ART: on a domain
art.train(X0, TrainingConfig(atol={atol}, algorithm="art", domain=(lower, upper),
                             art=ARTSettings(sample_multiplier={m})), evaluator)
values, outside = art.predict(queries)           # routed down the tree
report = art.audit(evaluator, n_points={audit})     # share of the domain beyond the tolerance
```.text
  .replace("{atol}", repr(ex.settings.afinni.atol))
  .replace("{n_init}", str(ex.settings.afinni.num_initial_clusters))
  .replace("{n_min}", str(ex.settings.afinni.min_pts_per_cluster))
  .replace("{kappa}", str(ex.kappa))
  .replace("{m}", str(ex.settings.art.sample_multiplier))
  .replace("{audit}", str(ex.art.audit.points)))

These are the main settings of the running example (it also fixes `seed` and `max_levels`). The backend is chosen by the environment:
`AFINNI_DEVICE=cpu|gpu|auto` and `AFINNI_MPI=off|on|auto`; under `mpiexec`, `train` and
`predict` are collective. `docs/USAGE.md` lists every setting, `docs/ARCHITECTURE.md` maps
the stages above to the modules of `src/`, and `docs/AFINNI_ART_DESIGN.md` records the
design decisions behind them.

#bibliography("references.yml", title: [References])
