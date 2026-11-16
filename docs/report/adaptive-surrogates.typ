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
// The colour key of every map of a model's errors.
#let error-key(width: 100%) = panel("residual-key", width: width)
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
  require[a batch of nodes $C$: region $Omega_C$, inherited points, locational code $pi$; budget $n$],
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
  ln(1, [their locational codes: $2 pi$ and $2 pi + 1$; record $(pi, a, z)$]),
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

== How to read this companion <sec:reading>

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
  [a cluster, its points, its centre. The engine's unit of work in both algorithms; in ART every cluster is a _node_ of the tree, which adds a region and a code], [$C$, $X_C$, $c$],
  [a cluster's two children: the left one below the cut, the right one on or above it], [$Cm$, $Cp$ #h(0.3em) (ART's codes $2 pi$, $2 pi + 1$)],
  [the local model of a cluster], [$fh(x) = w^top (x - c) + b$ #h(0.3em) (`A_fit`, and `b_fit` $= b - w^top c$)],
  [the absolute and relative tolerances], [$atol$, $rtol$ #h(0.3em) (`atol`, `rtol`)],
  [the number of initial clusters; the minimum number of points in a cluster], [$Ninit$, $Nmin$ #h(0.3em) (`num_initial_clusters`, `min_pts_per_cluster`)],
  [the Hessian of $f$; its eigendecomposition, with the eigenvectors as the columns of $U$ and the eigenvalues on the diagonal of $D$; its absolute value],
  [$H$; #h(0.2em) $H = U D U^top$; #h(0.2em) $absH = U abs(D) U^top$],
  [the trust region of a leaf], [$(x - c)^top T (x - c) <= 1$ #h(0.3em) ($T$ is `trust_matrix` in the code)],
  [the leaves and their number], [$cal(S)$, $N_c$],
  [ART: a split plane, its unit normal and offset; the side $a^top u >= z$ is the child $Cp$], [$(a, z)$, $norm(a) = 1$ #h(0.3em) (`normal`, `offset` in the code)],
  [ART: a node's region, in coordinates $u in [0, 1]^d$ of the domain. Each ancestor's plane adds one row: $plus.minus a^top$ to $A$ and $plus.minus z$ to $bold(z)$, the sign choosing the node's side; the box's faces are rows too], [$Omega_C = {u : A u <= bold(z)}$],
  [ART: a node's locational code], [$pi$ #h(0.3em) (`locational_code`)],
  [ART: the points a node is fitted on: the multiplier $m$ times the number of coefficients $p$ of an affine model], [$n = ceil(m p)$, $p = d + 1$ #h(0.3em) ($m$ is `sample_multiplier`)],
  [the MPI ranks, and the work of rank $i$], [$P$, $w_i$],
)

A batch stores its clusters' points back to back, one point per row, and three arrays
describe the layout. Every section uses these names, and so does the code.

#translate(
  [the number of clusters in a batch], [$K$ #h(0.3em) (`K`)],
  [the points of each cluster, and the row where each cluster starts], [`counts`, `starts` #h(0.3em) ($K$ entries each)],
  [each row's cluster, numbered $0, ..., K - 1$ within the batch], [`labels` #h(0.3em) (one entry per row)],
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
  clusters (pale) leave for the leaves; each rejected cluster (outlined black) is replaced in
  the next level by its two children, next to each other. Some rows grow: a small rejected
  cluster is given new points before it is cut (@sec:augment).]) <fig:queue>

#figure(panel("queue-layout"), caption: [One level of @fig:queue up close, as it is stored:
  level #ex.afinni.layout.level cut into batches, and the next level its rejected clusters
  write, larger and so in more batches (@sec:augment explains the new points, @sec:mpi the
  tail).]) <fig:layout>

Note that $j$ counts the _rejected_ clusters only. Cluster #ex.afinni.layout.first_accepted
was accepted and has no children, so children $2 j$ and $2 j + 1$ of the next level belong to
the $j$-th rejected cluster, not to cluster $j$.

The queue also grows. Level #(ex.afinni.layout.level + 1) holds #ex.afinni.layout.next_points
points against #ex.afinni.layout.points: the children keep all of their parents' points, and
#ex.afinni.layout.new_points of them are new, added to rejected clusters too small to cut. The
engine therefore never sizes a level in advance. The next level's queue is a host buffer that
starts at twice the current level's points and doubles when it fills, after checking that
the machine has the memory, and the next level simply runs in more batches
(#ex.afinni.layout.next_batches instead of #ex.afinni.layout.batches here). Only one batch
is ever on the device (@sec:batches).

Three properties follow from this layout, and the engine relies on each of them. Every step
of a batch is an operation on consecutive rows, with no lookup. A batch can be cut anywhere
between clusters, and since the batches of a level run one after another, the clusters at
the end of the queue are the last to start: a block of them can be handed to an idle
process before this rank reaches them, which is how idle ranks take work from busy ones
(work stealing, @sec:mpi). And because
children are written in the order their parents were processed, the order is kept from
level to level:

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
  [extra columns], [none], [root, locational code $pi$, the rows of $Omega_C$],
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
#step([Level 0: no cluster passes], code: afinni-code, focus: (1, 2, 3, 4), window: 1, width: 36%,
  stack(spacing: 4pt, panel("afinni-error-0"), error-key()))[
  The #ex.afinni.levels.at(0).clusters initial clusters, with every point coloured by the
  error of its cluster's model (colours as in @sec:reading). Each cluster shows the same
  pattern: the model is too low at the rim and too high in the middle, or the reverse. That bowl is the part of $f$ an
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

#let error-strip = filmstrip(
  (panel("afinni-error-0"), [level 0: #ex.afinni.levels.at(0).worst_error]),
  (panel("afinni-error-1"), [level 1: #ex.afinni.levels.at(1).worst_error]),
  (panel("afinni-error-2"), [level 2: #ex.afinni.levels.at(2).worst_error]),
  (panel("afinni-error-3"), [level 3: #ex.afinni.levels.at(3).worst_error]),
  (panel("afinni-error-4"), [level 4: #ex.afinni.levels.at(4).worst_error]),
)
#figure(stack(spacing: 2pt, error-strip, align(center, error-key(width: 34%))), kind: image,
  caption: [The clusters still in the queue at each level, coloured by the error of their
    models, with the worst error of each level. Accepted clusters leave the
    picture; the rest are cut and fitted again. After #ex.afinni.n_levels levels nothing is
    left.]) <fig:errors>

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
  caption: [One rejected cluster of level #ex.afinni.split.level, with
    #ex.afinni.split.points points and worst error #ex.afinni.split.parent_error, cut two ways.
    Colours as in @sec:reading. *Left.* The error of the cluster's model forms stripes: $f$
    bends _across_ the cluster and hardly at all along it. *Middle.* A cut across the longest
    direction leaves both children with the full bend. *Right.* A cut across the bend,
    #ex.afinni.split.angle degrees away, leaves children that are nearly flat.],
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

#algorithm(caption: [Hessian estimate and Hessian-informed split of one rejected cluster
  (the manuscript's Algorithms 2 and 3): `CovarianceHessianSplit` in
  `src/components/splitting.py`.], pseudocode(splitting)) <alg:split>

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

#note(title: [How many points each fit uses, and why])[
  Both fits trade a well-conditioned solve against staying close to $c$. The lower bounds
  follow from the fits; the fractions of $Nmin$ are set empirically.
  - *Gradient: the $max(2 d, floor(Nmin slash 4))$ points nearest to $c$.* $hat(g)$ has $d$
    unknowns, so it needs at least $d$ points not lying in a subspace, and twice that keeps
    the solve well conditioned when some of them nearly do. More points cost accuracy: $f$
    is exact, so the fit's only error is the Taylor remainder $1/2 (x - c)^top H (x - c)$,
    which moves the fitted gradient by $O(norm(H) r)$ for points within distance $r$ of $c$.
    Hence the _nearest_ points, and few of them. $floor(Nmin slash 4)$ takes more of them
    when clusters are large, where they are still close to $c$.
  - *Level set: the $N_"cov" = floor(Nmin slash 3)$ points with the smallest Taylor error.*
    @thm:hessian bounds the error of $hat(Sigma)^(-1)$ by $sqrt((d + v) slash n)$: about $d$
    points at the least, and more help. But more points come from a wider set $E_epsilon$,
    whose shape the higher-order terms of $f$ and the error of $hat(g)$ distort. A third of
    $Nmin$ balances the two when $Nmin$ is well above $d$.
  - *Both stay local.* A cluster is cut only once it holds at least $2 Nmin$ points
    (@sec:augment), so the gradient uses at most an eighth of it and the level set at most a
    sixth.
]

#pitfall(title: [A level set smaller than the dimension])[
  $N_"cov" = floor(Nmin slash 3)$ has no floor in $d$: with the default $Nmin = 100$ it is
  33 points, so for $d > 33$ the level set cannot span every direction and $Sigma$ is
  singular. The code floors its eigenvalues at $10^(-12)$, which gives the directions the
  level set never reached the largest weight in $absH^(1 slash 2)$ instead of none. Raise
  `min_pts_per_cluster` with the dimension (`default_min_pts_per_cluster` gives
  $100 dot 1.3^d$) to stay clear of it.
]

#implementation(title: [In the code])[
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

#algorithm(caption: [The trust ellipsoid of an accepted cluster (the manuscript's Algorithm 5): `CovarianceEllipsoid` in
  `src/components/trust_regions.py`. $gamma$ is `q_reg`.], pseudocode(trust)) <alg:trust>

#story()
#step([An ellipsoid around a leaf's points], code: trust-code, focus: (1, 2, 3, 4), width: 36%, panel("afinni-trust-one"))[
  The scatter matrix $Sigma$ of the leaf's #ex.afinni.trust_points points gives the shape:
  the ellipsoid ${x : (x - c)^top Sigma^(-1) (x - c) <= 1}$ is long where the points
  spread. Scaling it by the largest value $alpha$ that $(x_i - c)^top Sigma^(-1) (x_i - c)$
  takes puts the farthest point (ringed) exactly on the boundary and every other point
  inside.
]

#lemma(title: [the trust region contains the leaf's points])[
  With $T$ from @alg:trust, $(x_i - c)^top T (x_i - c) <= 1$ for every point $x_i$ of the
  cluster, with equality for the farthest one.
] <lem:trust>

#proof[
  $(x_i - c)^top T (x_i - c) = (x_i - c)^top Sigma^(-1) (x_i - c) slash alpha$, and $alpha$ is the
  largest numerator.
]

#note(title: [Why the floor $gamma$])[
  $Sigma^(-1)$ exists only if the leaf's points spread in every direction. They need not:
  fewer than $d + 1$ points, or points that lie on a line or a plane (a thin slab left by
  repeated cuts, or augmented points on a segment between two others) give $Sigma$ a zero
  eigenvalue. The ellipsoid would then have zero width across that direction, every query
  off the points' subspace would have an infinite trust-region distance and be flagged,
  and the inversion itself would be unstable. Flooring the eigenvalues at $gamma$ gives
  the ellipsoid a minimum half-width $sqrt(gamma alpha)$ in every direction and caps how
  elongated it can be, while leaving well-spread leaves untouched. In the code $gamma$ is `q_reg`, by default
  $10^(#ex.defaults.q_reg_exponent)$. It floors the eigenvalues of the scatter $Sigma$, a sum
  over the points, in squared units of $x$: a small absolute length, not relative to the
  leaf's size.
]

A minimum-volume enclosing ellipsoid @todd2016 would be tighter, but needs an iterative
optimization per leaf; this one is a single eigendecomposition, batched over all accepted
clusters.

#algorithm(caption: [Inference (the manuscript's Algorithm 6): `AFINNISurrogate.predict`
  with `predictor="knn"`.], pseudocode(inference)) <alg:inference>

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

Searching all $N_c$ leaves in their own metrics would cost $O(N_c d^2)$ per query. The
search is therefore done in two stages: an exact Euclidean search for $kappa$ candidates
(`src/neighbors/knn.py`), then the $kappa$ trust-region distances. The first stage can miss
the right leaf: a long ellipsoid's centre can be far from a query its ellipsoid contains (query
A in the picture). So $kappa$ should grow with the dimension and with how elongated the
leaves are, and it is set empirically, for example $kappa = 2 d^2$; too small a $kappa$
shows up as queries flagged outside although a leaf contains them.

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
  only if the sample lies where the queries will. The flag can also guide sampling
  (_iterative refinement_): draw new initial points, keep those outside every trust
  region, and train again.
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

#definition(title: [regions, planes and locational codes])[
  The root's _region_ is the domain, $Omega = [0, 1]^d$ in $u$. A node $C$ that is split
  by the plane $(a, z)$, $norm(a) = 1$, has the children
  $ Omega_(Cm) = Omega_C inter {u : a^top u <= z}, quad Omega_(Cp) = Omega_C inter {u : a^top u >= z} , $
  and a point with $a^top u = z$ belongs to $Cp$. A region is therefore a convex polytope
  ${u : A u <= bold(z)}$: the box, and one row $plus.minus (a, z)$ of $(A, bold(z))$ for every
  ancestor. The _locational code_
  of the root is $pi = 1$, and the children of node $pi$ are $2 pi$ (for $Cm$) and $2 pi + 1$
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

== Naming a node: the locational code <sec:locational-code>

#question[A node is created on one rank, may be stolen by another, and is merged on a
  third. How should it be named, so that anyone can work out its name, its parent's and its
  children's, without asking anyone?]

The name has four jobs, and each rules something out.

#cards(
  columns: 4,
  ([Local], [Any rank computes a child's name from its parent's alone, with no message to
    anyone.]),
  ([Stable], [The name is the same whichever rank, batch or run builds the node. It seeds the
    node's random numbers (@inv:rng).]),
  ([Arithmetic], [Parent, children and depth follow from the name, without a lookup.]),
  ([Small], [One machine integer, carried with every queued node.]),
)

#cards(
  columns: 2,
  ([A global counter #text(fill: red)[✗]], [Numbering nodes $0, 1, 2, dots$ as they are created
    needs one counter shared by every rank, a message per node, and the numbers depend on the
    order of processing, so they are neither local nor stable.]),
  ([Coordinates #text(fill: red)[✗]], [A centre or a corner is a floating-point vector. It is
    not small, not exact, and it says nothing about the parent.]),
  ([A Morton code #text(fill: red)[✗]], [It is local, stable and arithmetic, but it names only
    boxes cut at their midpoints (@sec:morton).]),
  ([The locational code #text(fill: rgb("#2e7d32"))[✓]], [The route from the root, one bit per cut.
    It meets all four.]),
)

#figure(panel("locational-code-walk"), caption: [One point (black) on its way to its leaf. *Top.* The
  tree, with the point's route in red. Each step down adds one bit to the code, and the latest
  bit is red. *Bottom.* The same steps in the domain. Each frame shows the parent's region
  (grey), the plane that cut it, and the side the point is on (blue, the leaf in red).])
  <fig:walk>

The root's code is $1$. Every cut appends one bit to a node's code: $0$ for the side
$a^top u < z$ of the plane and $1$ for the side $a^top u >= z$. The point in @fig:walk
takes #ex.art.locational_code.depth turns, so its leaf's code is
#raw(ex.art.locational_code.leaf). Reading the code from left to right replays the turns.

#definition(title: [locational code])[
  The root has locational code $pi = 1$. A node with locational code $pi$ has children $2 pi$
  (the side $a^top u < z$) and $2 pi + 1$ (the side $a^top u >= z$). A node is identified by
  its root and its locational code, $(r, pi)$.
] <def:locational-code>

#scene(side: left, width: 34%, panel("art-forest"))[
  The running example has one root, so $r$ never changes there. With `roots="grid"` or
  `"voronoi"`, ART grows one tree per root cell instead, and each tree numbers its own
  nodes from $1$. The picture shows the same example with #ex.art.forest.roots grid roots.
  The code #raw(ex.art.forest.code) names a node in all #ex.art.forest.trees_with_code
  trees, and two of them are marked. Only the pair $(r, pi)$ tells them apart.

  The pair is needed in three places. The routing table stores each plane under $(r, pi)$
  and checks that no pair repeats. A stolen node carries both, so the thief knows which tree
  it extends. And $(r, pi)$ seeds the node's random numbers: with $pi$ alone, the nodes
  #raw(ex.art.forest.code) of different trees would draw the same numbers and place their
  samples in the same relative positions.
]

Every question about a node's place in the tree is a bit operation on its code:

#block(breakable: false, table(
  columns: (auto, auto, auto),
  inset: (x: 7pt, y: 5pt),
  align: (left, left, left),
  table.header[Question][Arithmetic][For #raw(ex.art.locational_code.leaf)],
  [its parent], [drop the last bit: $floor(pi slash 2)$, `pi >> 1`], [#raw(ex.art.locational_code.parent)],
  [which side of the parent's plane], [the last bit: $pi mod 2$], [#raw(ex.art.locational_code.leaf.slice(-1)) (#if ex.art.locational_code.leaf.ends-with("1") [right] else [left])],
  [its children], [append a bit: $2 pi$, $2 pi + 1$], [#raw(ex.art.locational_code.children.at(0)), #raw(ex.art.locational_code.children.at(1))],
  [its sibling], [flip the last bit: $pi xor 1$], [#raw(ex.art.locational_code.sibling)],
  [its depth], [bits after the place-holder bit: $floor(log_2 pi)$], [#ex.art.locational_code.depth],
  [is $pi'$ an ancestor of $pi$?], [$pi$'s bits begin with $pi'$'s], [#raw(ex.art.locational_code.codes.at(3)) is; #raw(ex.art.locational_code.sibling) is not],
))

The leading $1$ is the _place-holder bit_. Without it the codes $0$, $00$ and $000$ would be
the same integer. With it every route has its own number, and the numbers run
$1$; $2, 3$; $4, dots, 7$, one row per depth, as in a binary heap.

#note(title: [The standard terms])[
  A code that records a node's route from the root is a _locational code_. Gargantini
  introduced it for quadtrees as a base-4 code with one digit per subdivision
  @gargantini1982, and it is the standard term in the spatial-data-structure literature
  @samet2006. ART's locational code is binary, with one digit per cut. The leading $1$ is
  Warren and Salmon's _place-holder bit_, which they add "to distinguish the higher level
  internal nodes of the tree from the lowest level body nodes"; their root is also the key
  $1$ @warren1993. The code calls $pi$ `locational_code`.
]

=== Locational codes and Morton codes <sec:morton>

A _Morton code_ is the locational code of a quadtree, the tree that cuts every box into four
equal quarters. It also appends bits as it goes down, two per level in two dimensions, one
for each coordinate. Because every cut is at a known place, the bits also give the cell's
position.

#filmstrip(
  arrows: false,
  (panel("morton-code", width: 92%), [Morton code #raw("1001"): quarter #raw("10"), then its
    quarter #raw("01")]),
  (panel("morton-art", width: 92%), [ART code #raw(ex.art.locational_code.leaf): six turns;
    the shape needs the six planes]),
  (panel("code-capacity", width: 100%), [levels that fit in a 64-bit code]),
  caption: [What a code says about a cell. *Left.* The Morton code
    #raw("1001") picks the top-left quarter (#raw("10")) and then its bottom-right quarter
    (#raw("01")). The code alone gives the box. *Middle.* ART's leaf
    #raw(ex.art.locational_code.leaf) and the planes of its six ancestors, numbered by depth.
    Its code records which side of each plane the leaf is on, but not where the planes are.
    *Right.* How deep a tree a 64-bit code can name as the dimension grows.],
) <fig:morton>

The two codes share three properties. An ancestor's code is a prefix of its descendant's
code, the code fits in one integer, and a child's code follows from its parent's alone.

A Morton code has one more property that ART cannot use. Its cuts are always at the middle
of an axis, so the code determines the cell. ART cuts along oblique planes at offsets chosen
from the data (@sec:plane), and a string of side bits cannot give their position. ART
therefore stores the planes separately. The routing table holds each split node's plane
$(a, z)$ once, under that node's locational code, and the leaf's shape is the intersection
of its ancestors' half-spaces.

Dimension sets a second limit. A Morton code spends $d$ bits per level, so a 64-bit code
holds $floor(63 slash d)$ levels after the place-holder bit, and only
#ex.art.locational_code.morton_levels_d40 at $d = 40$. A locational code spends one bit per
cut in any dimension, which allows trees #ex.art.locational_code.max_depth levels deep.

Warren and Salmon's tree code uses the Morton code with a place-holder bit, so one key both
names an octree cell and locates it @warren1993. ART keeps only the naming part.

#implementation(title: [In the code])[
  - `root` and `locational_code` are two `int64` columns of every queued node
    (`src/core/clusters.py`), and `node_keys(seed, root, locational_code)` in
    `src/art/rng.py` hashes them into the node's random-number key. They stay separate on
    purpose. Packing them into one integer, `root << shift | code`, would take bits from the
    code and lower the depth limit as the number of roots grows. Two columns cost 8 bytes per
    queued node and keep the full depth for any number of roots.
  - A positive `int64` has 63 bits and the place-holder bit takes one, so routes fit up to depth
    #ex.art.locational_code.max_depth (`MAX_DEPTH_LIMIT`), the deepest ART tree allowed.
  - Codes are sparse, $2^"depth"$ wide but few in use, so they are keys, never array
    positions. When the tree is assembled, the routing table gives every node a compact
    index, and routing follows those, one array lookup per level (@sec:plane).
]

#takeaway[A node's locational code is its route from the root, one bit per cut. Any rank
  can compute it, and parent, children and depth are bit operations on it. A Morton code
  also gives a cell's position because its cuts are fixed; ART's cuts are chosen from the
  data, so ART stores each plane once next to the code.]


== A fixed budget of points in every node

A node is fitted on $n = ceil(m p)$ points, where $p = d + 1$ is the number of
coefficients of an affine model and $m$ is the `sample_multiplier`: here $m = #ex.settings.art.sample_multiplier$ and
$n = #ex.settings.art.node_points$. It inherits the points of its parent that fall on its
side, which cost nothing, and draws the rest uniformly in its own region. A node deep in
the tree is thus fitted and tested on as many points as the root, concentrated in its
small region.

#algorithm(caption: [Hit-and-run in a polytope @smith1984 @lovasz2003:
  `HitAndRunSampler` in `src/art/sampling.py`. One chain per new point, started at a
  random inherited point.], pseudocode(hitrun)) <alg:hitrun>

#story()
#step([A region is the box and its ancestors' planes], code: art-code, focus: ("in", 2), window: 0, width: 35%, panel("art-polytope"))[
  The node with locational code #ex.art.sampling.code is at depth #ex.art.sampling.depth.
  Its region (shaded) is bounded by the domain box (dark outline) and by the
  #ex.art.sampling.rows planes of its ancestors, numbered by depth. Each plane is one row of
  $(A, bold(z))$, and so is each face of the box; here the box's top face closes the region.
  The node inherited #ex.art.sampling.inherited points (dark) and needs
  #ex.art.sampling.new new ones (pink), anywhere in the region, including next to its
  boundary.
]

#step([Hit-and-run: a random walk that stays inside], code: hitrun-code, focus: (2, 3, 4), width: 35%, panel("art-hit-and-run"))[
  To draw uniformly in a polytope, ART walks. The walk starts at an inherited point (dark),
  and each numbered step does three things. It picks a random direction $g$. It finds the
  chord, the part of the line through the current point along $g$ that lies inside the
  region (dotted). It then jumps to a uniform point of that chord. The ends of the chord
  come from one product $A g$. After a few steps the walker (pink) no longer depends on
  where it started.
]

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
  Every random number ART draws is a hash of the run's seed, the node's root and
  locational code, the purpose of the draw and a counter (`src/art/rng.py`). No generator is
  advanced from batch to batch. The points a node draws, and so its fit and its plane, are
  therefore the same whatever batch the node is in, whichever rank processes it, and
  whenever it is stolen (@sec:mpi).
] <inv:rng>

#implementation(title: [In the code])[
  - `roots="grid"` or `"voronoi"` starts from several roots instead of one: a regular grid
    of the domain, or the Voronoi cells of the initial clustering's centres. More roots
    give the ranks independent work from the first level on.
  - A root box is sampled exactly, without a walk.
  - Every ART fit, of a node and in the plane search, is plain least squares with
    $lambda = 0$, as in the reference ART (`leaf_fit="lstsq"`): a Cholesky solve of the
    normal equations where their reciprocal condition number is at least $10^(-10)$, and QR
    on the node's points where it is not. AFINNI's ridge (`leaf_fit="tikhonov"`) would shrink
    the slope across a node of aspect ratio $r$ by about $10^(-8) r^2$, which deep, thin
    nodes reach.
  - `validation_samples` draws a second, independent set of points in every node, on which
    an accepted model is checked again before it becomes a leaf.
  - On a GPU every chain of a batch walks in the same kernel launch
    (`src/kernels/hit_and_run.cu`), one thread per chain: one launch per half of the walk,
    and one more for each doubling.
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
  (panel("art-split-after"), [the children's errors after ART's cut (solid); AFINNI's cut (dashed)]),
  caption: [The split search on the node with locational code
    #ex.art.split.code, which has #ex.art.split.points points. Error colours as in
    @sec:reading. *Left.* One model leaves a mean squared error of #ex.art.split.sse_parent,
    in units of $atol^2$. *Middle.* The search tries cuts along
    #ex.art.split.directions directions (arrows; the red one is AFINNI's Hessian-informed
    direction) and finds a plane with error #ex.art.split.sse_bins (dashed). The soft search
    turns and shifts it to reach #ex.art.split.sse_final (solid). *Right.* The errors of the
    two children's models after ART's cut (solid). The #swatch(red, shape: "dashed") red dashed
    line is where AFINNI would cut the same points: along its Hessian-informed direction, at
    the median. With one affine model on each side, ART's children have a largest error of
    #ex.art.split.art_worst $atol$ and a mean squared error of #ex.art.split.art_mse, against
    #ex.art.split.afinni_worst $atol$ and #ex.art.split.afinni_mse for AFINNI's. The node,
    its points and the fits are the same, so only the cut differs: AFINNI fixes the cut before
    fitting, and ART chooses it by how well the children fit. ART minimizes the squared error,
    so its smaller largest error here is a consequence, not a guarantee.],
) <fig:search>

== A coarse search, then a smooth one

The hard objective is a staircase: it changes only when the plane crosses a point, and
has no useful gradient. ART handles it in two stages.

#algorithm(caption: [ART's split search: `_bins_search` and `_soft_refine` in
  `src/art/plan.py`, on the primitives of `src/art/split_search.py`.],
  pseudocode(search)) <alg:search>

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
  $s_i = (a^top u_i - z) slash h$, a small floor $delta$ and a fixed ridge $lambda >= 0$,
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
  minimizer is unique: every weight is at least $delta > 0$, so each weighted Gram is
  positive definite once the node's $phi_i$ span $RR^p$ (and for any points if $lambda > 0$).
  By the envelope theorem the derivative of the
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

== From a tree to an answer

#scene(width: 62%, panel("art-route"))[
  A query is _routed_: start at the root, and at every split node evaluate
  $a^top u - z$ and go right if it is non-negative. The picture follows one query (black
  dot) through the nested regions it falls in, down to its leaf at depth
  #ex.art.route.depth\; the locational code of the leaf, #ex.art.route.code, is the record of the
  turns taken. The leaf's model gives the answer.
]

#figure(panel("art-tree"), caption: [The tree of the running example with its
  locational codes in binary. A node's code is its parent's with one digit appended, 0 for
  the side $a^top u < z$ and 1 for the side $a^top u >= z$; the number of digits after the
  place-holder bit, the leading 1, is the depth.]) <fig:tree>

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

#figure(
  stack(spacing: 6pt,
    filmstrip(
      arrows: false,
      (panel("error-afinni", width: 68%), [AFINNI: #ex.compare.afinni.leaves leaves]),
      (panel("error-art", width: 68%), [ART: #ex.compare.art.leaves leaves]),
    ),
    align(center, panel("error-map-key", width: 78%))),
  kind: image,
  caption: [The error of each surrogate over the whole domain, on a #ex.compare.grid
    $times$ #ex.compare.grid grid.],
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

= Batches, kernels and reproducible sums <sec:batches>

#question[The engine handles thousands of clusters at once on a GPU. How, without a loop
  over clusters, and without results that change from run to run?]

// Schematic rows of a batch: one cell per point, coloured by its cluster (three clusters of
// sizes 5, 3 and 6, the palette's 2nd to 4th colours), with chunks of two rows.
#let gpu-colours = ex.colours.palette.slice(1, 4).map(rgb)
#let gpu-sizes = (5, 3, 6)
#let cw = 1.45em
#let cell(c, body: none) = box(width: cw, height: cw, fill: c, stroke: 0.6pt + white,
  align(center + horizon, text(7.5pt, fill: white, weight: "bold", body)))
#let plain(body) = box(width: cw, height: 1em, align(center + horizon, text(7.5pt, body)))
#let span(n, body, stroke: none) = box(width: n * cw, inset: (x: 1.2pt), box(width: 100%,
  stroke: stroke, inset: (y: 2pt), align(center, text(7.5pt, body))))
#let row-label(body) = align(right + horizon, text(7.5pt, body))
#let cluster-cells(values: none, shade: none) = stack(dir: ltr, ..range(3).map(k => {
  let n = gpu-sizes.at(k)
  stack(dir: ltr, ..range(n).map(i => {
    let c = gpu-colours.at(k)
    if shade != none { c = if i < shade.at(k) { c.lighten(45%) } else { c } }
    cell(c, body: if values != none { str(values.at(k).at(i)) })
  }))
}))

#let batch-picture = grid(
  columns: (auto, auto),
  column-gutter: 0.5em,
  row-gutter: 0.35em,
  row-label[`X`, `F`], cluster-cells(),
  row-label[`labels`], stack(dir: ltr, ..range(3).map(k => stack(dir: ltr, ..range(gpu-sizes.at(k)).map(_ => plain(str(k)))))),
  row-label[`counts`, `starts`], stack(dir: ltr, ..range(3).map(k => span(gpu-sizes.at(k),
    [#gpu-sizes.at(k), #gpu-sizes.slice(0, k).sum(default: 0)]))),
  row-label[chunks], stack(dir: ltr,
    ..((2, 2, 1), (2, 1), (2, 2, 2)).enumerate().map(((k, cs)) => stack(dir: ltr, ..cs.map(n =>
      span(n, [], stroke: (left: 0.7pt + gpu-colours.at(k), right: 0.7pt + gpu-colours.at(k),
        bottom: 0.7pt + gpu-colours.at(k))))))),
  row-label[partials], stack(dir: ltr, ..((2, 2, 1, 2, 1, 2, 2, 2)).enumerate().map(((j, n)) =>
    span(n, $p_#j$))),
  row-label[sums], stack(dir: ltr,
    span(5, $p_0 + p_1 + p_2$), span(3, $p_3 + p_4$), span(6, $p_5 + p_6 + p_7$)),
)

#scene(side: left, width: 56%, batch-picture)[
  A batch is the points `X`, their values `F`, and one row per cluster, with every cluster
  a run of consecutive rows (@fig:queue). The points per cluster, `counts`, and their
  running sums, `starts`, say where each run begins, and `labels`, each row's cluster, comes from a
  cumulative sum of start markers, without a sort. To sum over clusters, every run is cut
  into chunks of #ex.defaults.segment_chunk rows (two in the picture), _counted from the
  run's own first row_. One thread adds up one entry over one chunk in row order, and a
  second pass adds each cluster's chunks in order. This is a _segmented reduction_
  (`segment_cross`), and one launch serves every cluster of the batch.
]

Almost every step of @alg:afinni and @alg:art is one of four kinds of array operation on
this layout.

#cards(
  columns: 2,
  ([Per row], [Projections, residuals, distances to a centre. One thread per row reads its
    cluster's matrix or plane in place, through `labels`, instead of a copy per row.]),
  ([Segmented reductions], [Sums, scatter matrices and maxima over each cluster's rows:
    centroids, $Sigma$, the normal equations, the largest error.]),
  ([Small batched solves], [A $p times p$ system or a $d times d$ product per cluster,
    one thread per system, or a batched eigendecomposition.]),
  ([Sorts within clusters], [A stable sort by value, then a stable sort by cluster: every
    cluster's rows end up in order, still in one run.]),
)

#block(breakable: false, table(
  columns: (1.25fr, 1fr, 1.45fr),
  inset: (x: 5pt, y: 4.5pt),
  align: left + horizon,
  table.header[Step][Kind][Kernel or library call],
  table.cell(colspan: 3, fill: luma(245))[*AFINNI* (`src/afinni/plan.py`, `src/components/`)],
  [centres, means, $Sigma$, the fit's normal equations], [segmented sums], [`segment_cross`; tiled through shared memory when the result has many entries],
  [the fit, the products with $absH^(1 slash 2)$], [small solves], [`batched_inplace_cholesky_solve`, `spd_solve`, `det_matmul`],
  [errors $e_a$, split scores, Mahalanobis distances], [per row], [`fused_split_projection`, `fused_mahalanobis_projection`, `taylor_residual_norm`],
  [largest error, trust radius], [segmented maxima], [`segmented_max_float64`, one thread per cluster],
  [cut direction, Hessian estimate, trust ellipsoid], [eigendecompositions], [CuPy's batched `eigh` (cuSOLVER)],
  [the median cut, the points nearest each centre], [sorts within clusters], [two stable `argsort`s over the whole batch],
  [new points (@sec:augment)], [per new coordinate], [one element-wise kernel],
  table.cell(colspan: 3, fill: luma(245))[*ART* (`src/art/`)],
  [hit-and-run in every rejected node], [per chain], [`hr_walk`, one thread per chain],
  [cut statistics for every bin], [segmented sums], [`segment_cross` over (node, bin) runs, then `bin_sides`, `spd_solve`],
  [soft search], [per row and per node, repeated], [`soft_gates`, `soft_residuals`, `soft_row_terms`, `soft_node_prep`, `soft_trial_terms`, `soft_node_select`],
  [children, their codes and regions], [per node], [index arithmetic and one gather],
))

The project's own kernels live in `src/kernels/*.cu`, one file per area, and are compiled
when first used. Each exists for one of three reasons: a library routine would sum in an
order that depends on the batch (@sec:same-bits), a chain of library calls would build a
temporary of one matrix per row, or many small launches would cost more than the work.

Batches exist because a level may not fit on the device.

#block(breakable: false, table(
  columns: (1fr, 1.3fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[Constraint][What the engine does],
  [the points of a level can exceed the device's memory], [keeps the queue in host memory and moves one batch at a time to the device],
  [a batch holds as many whole clusters as fit], [caps `batch_size` (points) at 65% of the free device memory divided by `plan.bytes_per_point()`; a larger cluster runs alone],
  [a batch's points must be released as early as possible], [hands them to the plan (`DeviceBatch.take`), which frees them before it augments and cuts],
  [the host can run out of memory too], [checks the available memory before the queue grows, and raises `MemoryError` with advice],
))

== What the layout makes easy

=== A median for the price of a sort

#scene(side: left, width: 52%, grid(
  columns: (auto, auto),
  column-gutter: 0.5em,
  row-gutter: 0.6em,
  row-label[scores], cluster-cells(values: ((7, 2, 9, 4, 5), (6, 1, 3), (8, 3, 5, 2, 9, 6))),
  row-label[sorted], cluster-cells(values: ((2, 4, 5, 7, 9), (1, 3, 6), (2, 3, 5, 6, 8, 9))),
  row-label[cut], cluster-cells(values: ((2, 4, 5, 7, 9), (1, 3, 6), (2, 3, 5, 6, 8, 9)), shade: (2, 1, 3)),
))[
  AFINNI cuts every rejected cluster at the median of its scores along the cut direction
  (@sec:split). Sorting the whole batch by score and then, stably, by cluster puts every
  cluster's rows in order of score while keeping each cluster in its own run (middle). The
  median then needs no search: the first $floor(n slash 2)$ rows of a run are $Cm$ (pale)
  and the rest are $Cp$. The children are already runs of consecutive rows, so one gather
  `X[order]` writes the next level in the layout this section began with.
]

Two passes of CuPy's stable `argsort` replace a lexicographic sort, which CuPy implements
as a slower merge sort. ART reuses the same sort for its children and then checks that
its stored plane puts every row on the side the position says, so that routing a training
point later reproduces the split exactly (`ARTPlan._split`).

=== A node's region travels with it

#let carry = ex.art.carry
// One carried row as three cells, a_1, a_2 | z, in its plane's colour.
#let carry-num(v, c) = box(width: 2.5em, height: 1.2em, fill: rgb(c).lighten(78%), stroke: 0.5pt + rgb(c),
  align(center + horizon, text(6.5pt, str(v))))
#let carry-row(r, c) = stack(dir: ltr, carry-num(r.at(0), c), carry-num(r.at(1), c), h(2pt), carry-num(r.at(2), c))
#let carry-block(depth) = if depth == 0 { box(height: 1.2em, align(horizon, text(7pt, fill: grey)[no rows])) } else {
  stack(spacing: 0pt, ..range(depth).map(j => carry-row(carry.rows.at(j), carry.plane_colours.at(j))))
}
#let qb = ex.art.queue_blocks.shapes

To sample a rejected ART node, hit-and-run needs its polytope $Omega_C$: the root's box cut
by every ancestor's plane (@sec:regions). The GPU never looks these planes up in a tree.
Every node _carries_ them, as a block of rows $[a^top | z]$, one per ancestor, each meaning
$a^top u <= z$.

#figure(panel("art-queue-blocks"), caption: [ART's queue in the running example, level by
  level, as the planes its nodes carry. Each node (labelled by its locational code) is a
  block with one row per ancestor plane, in the colour of the node that made the plane; a
  hatched row is negated, because the node lies on the plane's right side. A rejected node
  (outlined) writes two children whose blocks are its own plus one row. An accepted node
  (pale) leaves for the leaves, and its block is dropped with it. At the right, the shape of
  each level's array and how many of its nodes were accepted.]) <fig:queue-blocks>

Read down the figure: $h$ is the level, and $K$ at the next level is twice the number of
nodes rejected at this one, so $K$ grows only while nodes are rejected
(#qb.map(r => str(r.at(0))).join(", ") here). For a batch, the children's blocks are one
gather of the parents' blocks and one concatenation. The rest of a child's identity is index
arithmetic too: its code is $2 pi$ or $2 pi + 1$, its root is its parent's, and its random
numbers are keyed by $(r, pi)$ (@inv:rng). @fig:carry follows one node down, with the
numbers in its rows and the planes they are.

#figure(
  filmstrip(..range(carry.codes.len()).map(t => (panel("art-carry-" + str(t)), [
    $pi = #carry.codes.at(t)$ #v(2pt) #carry-block(t)]))),
  kind: image,
  caption: [Node #carry.codes.at(-1) of @fig:queue-blocks and its ancestors, from the root
    down. Each frame shades the node; each numbered plane is drawn in the colour of the node
    it split, and its arrow marks the side the node keeps. Under each frame, the node's
    block: one row $a_1, a_2 | z$ per plane, in the unit coordinates $u$, negated for a right
    child ($-a^top u <= -z$).],
) <fig:carry>

#let labels-picture = {
  let chains = (3, 2, 3)
  // A node's block: h = 2 rows of d + 1 = 3 numbers, the last (z) set apart.
  let num(k) = box(width: 0.9em, height: 0.9em, fill: gpu-colours.at(k).lighten(55%), stroke: 0.5pt + white)
  let block(k) = stack(spacing: 0pt, ..range(2).map(_ => stack(dir: ltr, num(k), num(k), h(2pt), num(k))))
  grid(
    columns: (auto, ..chains.map(_ => auto)),
    column-gutter: 0.9em,
    row-gutter: 0.4em,
    align: center + horizon,
    row-label[chains (threads)], ..range(3).map(k => stack(dir: ltr, ..range(chains.at(k)).map(_ => cell(gpu-colours.at(k))))),
    row-label[`labels`], ..range(3).map(k => stack(dir: ltr, ..range(chains.at(k)).map(_ => plain(str(k))))),
    [], ..range(3).map(_ => text(9pt)[↓]),
    row-label[blocks, $h times (d + 1)$], ..range(3).map(k => block(k)),
    row-label[node], ..range(3).map(k => text(7pt)[$#k$]),
  )
}

#scene(side: left, width: 52%, labels-picture)[
  On the device, a batch's blocks form one array of shape $(K, h, d + 1)$: $K$ nodes (three
  here), $h$ rows each (two here), and $d + 1$ numbers per row, the normal $a$ and the
  offset $z$. The `hr_walk` kernel runs one thread per chain, one chain per new point, and a
  chain's label is its node's row in that array, so each thread reads its node's block in
  place. The chains of one node are consecutive threads and read the same block, which the
  cache then serves to all of them, and every node has the same $h$, so the threads of a
  warp loop in step.
]

A queue holds one level, and even a stolen queue comes from one level of one rank, so the
nodes of a batch always share their depth. (The kernel would accept mixed depths, padding a
short block with rows $0 <= 0$, which cut no chord, but the engine never forms such a
batch.) Carrying the planes costs $h (d + 1)$ numbers per node, copied again by every
child. One shared table of planes, indexed by the ancestors' codes ($pi$, $floor(pi slash 2)$,
$floor(pi slash 4)$, ...), would store each plane once; its price is that a stolen node
would need the table too, and every chain step would gather planes from scattered places in
memory.

=== Every candidate cut in one pass

#let bins = ex.art.split.bins
The bins search (@sec:plane) scores $n_"bins" - 1$ cuts along each of
#ex.defaults.n_directions directions in every node. Refitting each candidate from its
points would read every point once per cut.

#figure(
  filmstrip(arrows: false,
    (panel("art-bins-points"), [#bins.n_bins bins along one direction: every boundary is a candidate]),
    (panel("art-bins-errors"), [the error each cut leaves; the chosen one in gold])),
  kind: image,
  caption: [The bins search on the node of @fig:search, along the direction it chose.
    *Left.* The node's #ex.art.split.points points, sorted by score and split into
    #bins.n_bins bins of about #bins.per_bin points. Every boundary between bins (grey) is a
    candidate cut. The chosen one (gold, dashed, as in @fig:search) leaves the pink points on
    the left and the cyan ones on the right. *Right.* The mean squared error of the two fits
    each cut leaves, in units of $atol^2$; cut #bins.chosen_cut leaves #bins.chosen_error.
    The dashed line is one model for the whole node.],
) <fig:bins>

#let slide-picture = {
  let bin(b, fill) = box(width: 3.2em, height: 1.5em, fill: fill, stroke: 0.6pt + white,
    align(center + horizon, text(7.5pt, $S_#b$)))
  let row(j) = stack(dir: ltr, ..range(1, 5).map(b => {
    let c = if b <= j { pink.lighten(55%) } else { cyan.lighten(60%) }
    if b == j + 1 { stack(dir: ltr, box(width: 1.6pt, height: 1.5em, fill: dark), bin(b, c)) } else { bin(b, c) }
  }))
  grid(
    columns: (auto, auto, auto),
    column-gutter: 0.7em,
    row-gutter: 0.5em,
    align: (right + horizon, left + horizon, left + horizon),
    text(7.5pt)[bins], stack(dir: ltr, ..range(1, 5).map(b => bin(b, luma(230)))), [],
    ..range(1, 4).map(j => (text(7.5pt)[cut #j], row(j),
      text(7.5pt)[$L_#j = #range(1, j + 1).map(b => $S_#b$).join($+$)$, #h(0.3em)
        $R_#j = #range(j + 1, 5).map(b => $S_#b$).join($+$)$])).flatten(),
  )
}

#scene(side: left, width: 56%, slide-picture)[
  After the sort, each bin's rows sit next to each other. One segmented reduction over the
  $K dot n_"bins"$ (node, bin) runs gives every bin's least-squares sums $S_b$, the products
  $Phi^top Phi$, $Phi^top y$ and $y^top y$ of its rows, in one pass over the points. A cut
  then only decides which whole bins go left: its two fits need the sums $L_j$ and $R_j$
  and one small $p times p$ solve each, whatever the number of points.
]

The kernel `bin_sides` builds every $L_j$ in one walk over the bins from left to right and
every $R_j$ in one walk from right to left, one thread per (node, entry of the sums). It sums
the right side directly, never as the total minus $L_j$, which would cancel digits when one
side is small.

=== A search that stays on the device

// One step of the soft search for five nodes: the halving m each node takes (schematic).
#let ls-takes = (0, 1, 0, 2, 5)
#let ls-cell(kind) = box(width: 1.7em, height: 1.3em, stroke: 0.5pt + if kind == "none" { luma(228) } else { white },
  fill: if kind == "taken" { rgb(col.gold) } else if kind == "tried" { luma(200) } else if kind == "extra" { luma(236) } else { white })
#let ls-grid(kind) = stack(spacing: 0pt, ..range(5).map(i => stack(dir: ltr, ..range(12).map(m => ls-cell(kind(i, m))))))
#let ls-mark(cols, body) = box(width: cols * 1.7em, stroke: (top: 0.6pt + dark, left: 0.6pt + dark, right: 0.6pt + dark),
  inset: (top: 2pt), align(center, text(6.5pt, body)))
#let ls-picture = {
  let last = calc.max(..ls-takes)
  let cpu(i, m) = if m > last { "none" } else if m < ls-takes.at(i) { "tried" } else if m == ls-takes.at(i) { "taken" } else { "extra" }
  let gpu(i, m) = {
    let t = ls-takes.at(i)
    if m < 4 { if m < t { "tried" } else if m == t { "taken" } else { "extra" } }
    else if t < 4 { "none" } else if m < t { "tried" } else if m == t { "taken" } else { "extra" }
  }
  grid(
    columns: (auto, auto),
    column-gutter: 0.6em,
    row-gutter: 0.35em,
    align: (right + horizon, left + horizon),
    [], stack(dir: ltr, ..range(12).map(m => box(width: 1.7em, align(center, text(6.5pt, $#m$))))),
    text(7pt)[CPU], ls-grid(cpu),
    text(7pt)[host checks], stack(dir: ltr, ..range(12).map(m => box(width: 1.7em, align(center,
      text(7pt, fill: rgb(col.red), if m <= calc.max(..ls-takes) [◆] else [])))) ),
    [], v(0.4em),
    [], stack(dir: ltr, ls-mark(4)[launch 1: all nodes], ls-mark(8)[launch 2: waiting nodes only]),
    text(7pt)[GPU], ls-grid(gpu),
  )
}

#figure(ls-picture, kind: image, caption: [One step of the soft search for five nodes (rows),
  schematic. Column $m$ is the trial step $alpha 2^(-m)$; #swatch(rgb(col.gold), shape: "dot")
  gold is the step a node takes, dark grey the trials before it, light grey trials evaluated
  but not needed. The CPU tries one halving at a time for all nodes and asks the host
  (#text(fill: rgb(col.red))[◆]) whether any node still waits. The GPU evaluates the first four
  halvings of every node in one launch, and the other eight only for nodes still waiting.])
  <fig:line-search>

The soft search (@sec:plane) runs at most #ex.art.split.soft_iters steps, and each step tries
the step sizes $alpha$, $alpha slash 2$, $alpha slash 4$, ... until one lowers the objective
enough. On the GPU, `soft_trial_terms` evaluates them in two launches (@fig:line-search), and
`soft_node_select` takes each node's first passing trial, the step the CPU's loop takes.
Nodes advance, or record why they stopped, on the device; the host reads one flag per step,
to end the loop when every node has stopped.

== Where it is still hard

#block(breakable: false, table(
  columns: (1fr, 1.5fr, 1.3fr),
  inset: (x: 5pt, y: 4.5pt),
  align: left + horizon,
  table.header[Problem][Why][What could help],
  [*Uneven clusters*], [`segmented_max_float64` gives a cluster one thread, however many rows
    it has. In `hr_walk` a thread's work grows with steps $times$ planes $times d$, so one
    deep node at high dimension holds up the whole launch.], [chunk large clusters, as
    `segment_cross` already does; one warp per chain at high $d$],
  [*Sequential steps*], [The soft search, the doubling of drifting hit-and-run chains
    (@alg:hitrun) and the walk down an ART tree are each a loop of launches, with one
    host read per pass to decide whether to go on.], [checking the stop flag every few
    steps instead of every step],
  [*The host in the loop*], [The queue lives in host memory, every batch is copied in and
    its children out (both through page-locked buffers), and nothing overlaps the copies
    with work. A batch has dozens of small host reads, mostly the sizes of masks.], [copying
    the next batch in while the current one runs],
  [*Carried planes*], [Charged per point when the batch size is chosen, so deep ART levels
    run in smaller batches. Beyond $d = #ex.defaults.walk_max_d$ dimensions `hr_walk`
    does not fit, and a slower path of array operations takes over.], [a shared plane
    table and per-node indices],
  [*AFINNI's random streams*], [One stream per batch, so new points depend on which
    clusters share a batch (@prop:determinism).], [keying streams by cluster, as ART keys
    them by node],
  [*AFINNI's trust test*], [`predict` gathers the trust matrices of all $kappa$ candidates
    of every query, $kappa d^2$ numbers per query.], [one kernel that evaluates the
    quadratic forms in place],
))

== Answering queries

Prediction runs on chunks of a few thousand queries, so that its temporaries fit the device
whatever the number of queries.

=== AFINNI: the same work for every query

// A chunk of six queries as it moves through AFINNI's prediction: one row per query (its own
// colour), every array as wide as one query's share of it.
#let q-colours = ex.colours.palette.slice(1, 9).map(c => rgb(c).lighten(50%))
#let q-array(cols, label, shape) = stack(spacing: 3pt,
  box(height: 1.6em, align(bottom + center, text(6.5pt, label))),
  stack(spacing: 0pt, ..range(6).map(i => stack(dir: ltr, ..range(cols).map(_ =>
    box(width: 0.72em, height: 0.72em, fill: q-colours.at(i), stroke: 0.4pt + white))))),
  text(6.5pt, shape))
#let q-step(body) = box(width: 4.4em, align(center, stack(spacing: 2pt, text(6pt, body), text(10pt)[→])))
#let afinni-pipeline = grid(columns: 11, column-gutter: 3pt, align: center + horizon,
  q-array(2, [queries], $Q times d$), q-step[$kappa$ nearest centres],
  q-array(4, [candidates], $Q times kappa$), q-step[gather their $T$],
  q-array(8, [trust matrices], $Q times kappa times d^2$), q-step[one product],
  q-array(4, [distances $d_i$], $Q times kappa$), q-step[sort, keep $k$],
  q-array(2, [nearest], $Q times k$), q-step[evaluate, weight],
  q-array(1, [$fh$], $Q times f_d$))

#figure(afinni-pipeline, kind: image, caption: [AFINNI's prediction for a chunk of $Q$
  queries (six here, one colour each). Every array has one row per query, and every step
  does the same work for every row, so the whole chunk moves through each step at once.])
  <fig:afinni-pipeline>

Nothing in @fig:afinni-pipeline depends on another query, and no query takes a branch the
others do not, so the threads of a warp stay in step. The nearest-centre search is a tiled
float32 kernel whose candidates are ranked again by exact float64 distances. It and the
$k$-means of the initial clustering are the project's own CuPy code: cuML ships for Linux
only, while the GPU backend also runs on Windows, and these kernels are exact and give the
same result on every run. The gathered trust matrices, $kappa d^2$ numbers per query, are
the largest array of the chain.

=== ART: one level per pass

#let passes = ex.art.passes
#let route-cell(code, leaf) = box(width: 3.2em, height: 1.25em, fill: if leaf { luma(232) } else { white },
  stroke: 0.5pt + if leaf { luma(200) } else { dark }, align(center + horizon,
  text(6.5pt, fill: if leaf { grey } else { black }, raw(code))))
#let lane(q) = box(width: 1.25em, height: 1.25em, fill: q-colours.at(q), stroke: 0.5pt + white,
  align(center + horizon, text(6.5pt)[#(q + 1)]))
#let n-passes = passes.codes.len()

#figure(
  grid(
    columns: (auto, ..passes.codes.at(0).map(_ => auto), 1.2em, auto),
    column-gutter: 2pt,
    row-gutter: 2pt,
    align: (x, y) => if x == passes.codes.at(0).len() + 2 { left + horizon } else { center + horizon },
    [], ..range(passes.codes.at(0).len()).map(i => box(width: 3.2em, fill: q-colours.at(i), inset: (y: 2pt),
      align(center, text(6.5pt)[query #(i + 1)]))), [], text(6.5pt)[threads of the pass],
    ..range(n-passes).map(t => (
      text(6.5pt)[#if t == 0 [start] else [pass #t]],
      ..passes.codes.at(t).zip(passes.leaf.at(t)).map(((c, l)) => route-cell(c, l)),
      [],
      if t == 0 [] else {
        stack(dir: ltr, ..passes.leaf.at(t - 1).enumerate().filter(((i, l)) => not l).map(((i, l)) => lane(i)))
      },
    )).flatten(),
  ),
  kind: image,
  caption: [Eight uniform queries of the running example, routed as the library routes
    them. Each row is one pass and shows every query's node by its locational code; a query
    at a leaf (grey) is done. *Right.* The threads of each pass: the queries not yet at a
    leaf, compacted into a dense list, one thread each.],
) <fig:passes>

Queries reach leaves at different depths. If each thread walked its query down to its leaf, a
warp would run as long as its deepest query, with the lanes of finished queries idle.
`RoutingTable.route` instead moves every unfinished query one level per pass. It first
compacts them into a dense list (@fig:passes, right); then one thread per listed query reads
its node's plane, takes one dot product and moves to a child. Every thread of a pass runs the
same instructions, and finished queries take no thread. A query costs $O(d)$ per level of its
path. Two costs remain:

- *Scattered reads.* The queries of a warp sit in different nodes, so the planes they read
  are not adjacent in memory. The table of planes is small, $d + 1$ numbers per node, and
  stays in the cache.
- *One pass per level.* Each pass costs a few launches and one host read, the length of the
  list. The last passes move few queries, and there the launches dominate. One kernel that
  walks each query to its leaf would remove them, at the price of the idle lanes; for trees
  of even depth that price is small, and it is an open option.

== The same bits every time <sec:same-bits>

Floating-point addition is not associative. A sum computed by many GPU threads in
whatever order they finish differs from run to run in its last bits, and a last bit can
decide whether a cluster passes its tolerance, and so the shape of the whole tree.

#invariant(title: [deterministic sums])[
  Every sum on the training path is taken in a fixed order that depends only on the
  cluster it belongs to. On the GPU, a cluster's rows are summed sequentially in chunks
  counted from its own first row, and the chunks are then added in order
  (`segment_cross`, `reduce_sum_1d`, `det_matmul`, `spd_solve` in `src/numerics/`). No
  thread adds into a shared floating-point total, and no batched library product or solve,
  whose algorithm can change with the size of the batch, is used.
] <inv:sums>

The one library routine left on the training path is the batched symmetric
eigendecomposition. The tests train ART with a batch size of 97 and with the default, on
both backends, and require the same number of leaves, the same number of evaluations and
the same predictions to within $10^(-10)$.


#proposition(title: [what does not change the result])[
  + Repeated runs in one process on the same backend give bit-identical models.
  + An ART tree with hit-and-run sampling, as the set of its planes by locational code, does
    not depend on the batch size, the number of ranks or the schedule of steals.
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

#algorithm(caption: [Decentralized work stealing, run by every rank $i$ (the manuscript's
  Algorithm 7): `WorkStealer` in `src/engine/work_stealing.py`.],
  pseudocode(steal)) <alg:steal>

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
  is the busiest. The victim looks at requests between batches, and ART also during its
  soft search. It never gives away the batch it is working on, and it answers every
  request with a grant or a denial.
  A grant is followed by the clusters themselves: whole clusters from the _tail_ of the
  victim's queue, the ones it would have reached last.
]

#invariant(title: [the stealing protocol])[
  + *One reply.* Every request receives exactly one grant or denial. A thief waits for
    that explicit reply and for nothing else, so a stale view of the loads can delay it
    but not deadlock it.
  + *Only unstarted work.* A grant takes whole clusters from behind the current batch and
    always leaves the victim at least one. A cluster is processed by exactly one rank.
  + *A cluster travels complete.* A stolen cluster carries its points, values, depth and,
    for ART, its root, locational code and the rows of its region. The thief needs nothing else from
    the victim, then or later.
  + *Nobody leaves early.* A rank retires only when every rank is idle: it joins a
    non-blocking barrier and keeps answering requests until all have joined.
] <inv:steal>

== Four recorded runs

#let sr = ex.steal.runs
#let run-names = (root_static: [one root, no stealing], root_steal: [one root, stealing],
  grid_static: [#ex.steal.ranks roots, no stealing], grid_steal: [#ex.steal.ranks roots, stealing])
Stealing depends on timing, so its pictures are of recorded runs: ART on the running example
with a tighter tolerance, $atol = #ex.steal.atol$, on #ex.steal.ranks ranks, started in two
ways and each run without and with stealing. With one root, rank 0 holds all the work at
first. With #ex.steal.ranks grid roots, one per rank, every rank starts with its own cell of
the domain, the usual way to share work without stealing. Each configuration ran
#ex.steal.repeats times, interleaved and after a warm-up run, each rank on a core of its own
of the same speed; the pictures show the run with the median wall time.

#figure(panel("steal-timeline"), caption: [Who was working when, in the four runs. The slow first
  run has its own time axis; the other three share one. A rank's bar is coloured while it has nodes and pale while it is idle; each
  arrow is a grant, from the victim to the thief. Above each run, its wall time and the share
  of the ranks' time spent idle.]) <fig:timeline>

#figure(
  board(columns: 4, ..run-names.pairs().map(((n, label)) =>
    (panel("steal-owners-" + n.replace("_", "-")), label))),
  kind: image,
  caption: [Which rank built each leaf, in the same four runs. A stolen node takes its whole
    subtree with it, so a rank's leaves form patches of the domain.],
) <fig:owners>

#block(breakable: false, text(9.5pt, table(
  columns: (auto, auto, auto, auto, auto, auto, 1fr, auto),
  inset: (x: 5pt, y: 4.5pt),
  align: (left + horizon, right + horizon, center + horizon, right + horizon, right + horizon, right + horizon, center + horizon, right + horizon),
  table.header[Run][Wall time][#ex.steal.repeats runs][Speedup][Idle][Leaves][Leaves per rank][Grants],
  ..run-names.pairs().map(((n, label)) => {
    let r = sr.at(n)
    (label, [#r.wall s], [#r.wall_range s], [#r.speedup×], [#r.idle_percent%], [#r.leaves_per_rank.sum()],
      [#r.leaves_per_rank.map(str).join(", ")], [#r.grants])
  }).flatten(),
)))

With one root and no stealing, ranks 1 to #(ex.steal.ranks - 1) never get work: rank 0
builds all #ex.steal.leaves leaves alone, and #sr.root_static.idle_percent% of the ranks'
time is idle. Stealing spreads the same tree over all ranks and makes the run
#sr.root_steal.speedup times faster. With one root per rank, every rank starts working at
once, but the cells are not equally hard: the ranks whose cells need fewer levels finish
early and wait, #sr.grid_static.idle_percent% of the ranks' time. Stealing hands them the
unstarted nodes of the busy ranks; idle time falls to #sr.grid_steal.idle_percent%, and the
same tree is built in #sr.grid_steal.wall s instead of #sr.grid_static.wall s.

How much stealing gains depends on how uneven the start is: on this function, whose valley
crosses every cell, a static split is only mildly uneven, and the gain is modest; when the
work starts in one place, it is several-fold. An adaptive tree cannot know in advance which
region will need more levels, and stealing makes the run time nearly independent of how the
work starts. In each pair the tree is the same: the recording script checks every plane, by
root and locational code, bit for bit (@prop:determinism).

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
