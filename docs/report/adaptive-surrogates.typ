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
  lists: ((image, [Figures]), (table, [Tables]), ("algorithm", [Algorithms])),
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
  ln(0, [each node within tolerance, on its validation points too if it has any, becomes a leaf], note: [`accepted`]),
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

  The running example is the curved valley
  $f(x) = (1 - x_1)^2 + (x_2 - x_1^2)^2$ on the box
  $[#ex.domain.lower.at(0), #ex.domain.upper.at(0)] times [#ex.domain.lower.at(1), #ex.domain.upper.at(1)]$,
  drawn on the left by its level sets, which are light on the valley floor and darker where
  $f$ is large. The surrogate must agree with $f$ to an absolute tolerance $atol = #ex.alpha$.
]

Both algorithms of the library place the pieces _adaptively_. Each fits one affine model to
a region and measures its error against the tolerance, and where the error is too large, it
cuts the region in two and repeats on each half. The two algorithms differ in what a region
is, where its points come from and how the cut is chosen.

#board(
  (panel("afinni-level-0"), [*AFINNI* starts from a given sample, clustered into a few groups, ...]),
  (panel("afinni-level-2"), [... splits the groups whose model is not accurate enough, ...]),
  (panel("afinni-trust-all", width: 46%), [... and keeps a trust region around each accepted group.]),
  (panel("art-level-0"), [*ART* starts from the domain itself, ...]),
  (panel("art-level-3"), [... cuts it by planes, drawing fresh points in each part, ...]),
  (panel("art-partition", width: 46%), [... and ends with an exact partition into polytopes.]),
  columns: 3,
)

#figure(kind: table, caption: brief([AFINNI and ART at a glance], [AFINNI and ART side by side: what a region is, where its points come from, how it is cut, what is kept and how a query is answered.]), table(
  columns: (auto, 1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[][AFINNI][ART],
  [a region is], [a _cluster_ of sample points], [a _polytope_ of the domain],
  [its points], [are given; a few are added inside small clusters], [are drawn by the node itself, uniformly in its polytope],
  [the cut], [follows the curvature of $f$, estimated from the points], [is the plane whose two fits are best, found by a search],
  [what is kept], [only the leaves, each with a trust ellipsoid], [the whole tree, with every plane],
  [a query], [finds the nearest leaf in the leaves' own metrics, and may be flagged as outside every trust region], [walks down the tree to exactly one leaf],
)) <tab:glance>

AFINNI suits a sample that already exists, such as the states a simulation visited, and it
never needs to know the domain. ART suits a known domain on which every point must get an
answer. In the code, both are _plans_ of one _refinement engine_ (@sec:engine). The
configuration `TrainingConfig(algorithm="afinni")` or `"art"` chooses the plan, and one
class, `AdaptiveSurrogate`, trains and queries both.

== How to read this companion <sec:reading>

Each idea appears three times: as a *picture* of the running example, in plain *words*, and
in *symbols*. The symbols state a definition, lemma or theorem in a tinted box. The
pictures and the symbols say the same thing, so the reader can check either against the other. The
pictures are computed from real runs. The script `figures/make_figures.py` trains both
algorithms on the example, records every level of the runs, and stops unless the recording
reproduces the trained models exactly; it also writes the numbers this text quotes.

Colours keep one meaning throughout. In a map of clusters, each cluster has its own colour.
A #swatch(dark, shape: "cross") dark cross at its centre marks an accepted model, and a
#swatch(red, shape: "cross") red cross a rejected one. In a map of a model's _errors_ at
its points, a point is #swatch(rgb(col.residual_mid)) pale where the model is within the tolerance
and deep #swatch(rgb(col.blue)) blue or #swatch(red) red where it is too high or too low.
Highlighted pseudocode lines are the ones the picture beside them shows.

The notation follows the AFINNI manuscript @manuscript, and ART reuses it where the
objects are the same.

#translate(caption: brief([Notation], [The notation of this companion: each object in words beside its symbol, with the name in the code where it differs.]),
  [the function, its surrogate, the input dimension], [$f$, $fh$, $d$],
  [the initial points and their number], [$X_0$, $N$ #h(0.3em) (`X_train`)],
  [a cluster, its points and its centre; the cluster is the engine's unit of work in both algorithms, and in ART every cluster is also a _node_ of the tree, which adds a region and a code], [$C$, $X_C$, $c$],
  [a cluster's two children: the left one below the cut, the right one on or above it], [$Cm$, $Cp$ #h(0.3em) (ART's codes $2 pi$, $2 pi + 1$)],
  [the local model of a cluster], [$fh(x) = w^top (x - c) + b$ #h(0.3em) (`A_fit`, and `b_fit` $= b - w^top c$)],
  [the absolute and relative tolerances], [$atol$, $rtol$ #h(0.3em) (`atol`, `rtol`)],
  [the number of initial clusters; the minimum number of points in a cluster], [$Ninit$, $Nmin$ #h(0.3em) (`num_initial_clusters`, `min_pts_per_cluster`)],
  [the Hessian of $f$; its eigendecomposition, with the eigenvectors as the columns of $U$ and the eigenvalues on the diagonal of $D$; its absolute value],
  [$H$; #h(0.2em) $H = U D U^top$; #h(0.2em) $absH = U abs(D) U^top$],
  [the trust region of a leaf], [$(x - c)^top T (x - c) <= 1$ #h(0.3em) ($T$ is `trust_matrix` in the code)],
  [the leaves and their number], [$cal(S)$, $N_c$],
  [ART: a split plane, its unit normal and offset; the side $a^top u >= z$ is the child $Cp$], [$(a, z)$, $norm(a) = 1$ #h(0.3em) (`normal`, `offset` in the code)],
  [ART: a node's region, in coordinates $u in [0, 1]^d$ of the domain. Each ancestor's plane adds one row, $plus.minus a^top$ to $A$ and $plus.minus z$ to $bold(z)$, the sign chooses the node's side, and the box's faces are rows too],[$Omega_C = {u : A u <= bold(z)}$],
  [ART: a node's locational code], [$pi$ #h(0.3em) (`locational_code`)],
  [ART: the number of points a node is fitted on, the multiplier $m$ times the number $p$ of coefficients of an affine model], [$n = ceil(m p)$, $p = d + 1$ #h(0.3em) ($m$ is `sample_multiplier`)],
  [the MPI ranks, and the work of rank $i$], [$P$, $w_i$],
)

A batch stores its clusters' points back to back, one point per row, and three arrays
describe the layout. Every section and the code use these names.

#translate(caption: brief([Notation of a batch], [How a batch stores its clusters: each array in words beside its symbol and its name in the code.]),
  [the number of clusters in a batch], [$K$ #h(0.3em) (`K`)],
  [the points of each cluster, and the row where each cluster starts], [`counts`, `starts` #h(0.3em) ($K$ entries each)],
  [each row's cluster, numbered $0, ..., K - 1$ within the batch], [`labels` #h(0.3em) (one entry per row)],
)

#takeaway[Both algorithms fit an affine model, test it against the tolerance and cut where
  it fails, AFINNI on clusters of a given sample and ART on polytopes of a known domain.]

= One engine, two plans <sec:engine>

#question[Two algorithms that fit, test and cut. What do they share, and what is each
  one's own?]

The two algorithms share everything except what happens to one batch of clusters. Both keep
a _queue_ of clusters waiting to be fitted and process it in batches, and both send accepted
clusters to an output buffer and the children of rejected clusters to the next level's
queue. The _refinement engine_ owns that loop, together with the memory management, the
transfers to the GPU, the MPI distribution and the work stealing. Each algorithm is a
_plan_, which says what the initial queue is, where a batch may end, how much memory a point
needs, and what to do with one batch.

#algorithm(caption: brief([The refinement engine], [The refinement engine: `refine` in `src/engine/driver.py` and
  `refine_rank` in `src/engine/refinement.py`. Lines 5--6 are the plan's.]),
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
  A level is one pass over the queue. Only a split moves a cluster to the next level, so in
  a single process the queue of level $ell$ holds exactly the clusters at depth $ell$ of
  the forest. The loop ends when the queue is empty, and by then every cluster has either
  become a leaf or reached the depth limit.

  The engine processes whole _levels_ instead of following one branch down, which makes
  the algorithm parallel. The plan fits, tests and cuts all clusters of a batch at once, as
  array operations over thousands of clusters (@sec:batches).
]

#figure(panel("queue-strips"), caption: brief([AFINNI's queue, level by level], [The queue of every AFINNI level of the running
  example, as it is stored: each cluster's points back to back, in one array. Accepted
  clusters (pale) become leaves, and in the next level the two children of each rejected
  cluster (outlined black) take its place, next to each other. Some rows grow, because a
  small rejected cluster receives new points before it is cut (@sec:augment).])) <fig:queue>

#figure(panel("queue-layout"), caption: brief([One level of the queue, cut into batches], [One level of @fig:queue up close, as it is stored.
  Level #ex.afinni.layout.level is cut into batches, and the next level, which its rejected
  clusters write, is larger and so takes more batches. @sec:augment explains the new points,
  and @sec:mpi the tail.])) <fig:layout>

The index $j$ counts only the _rejected_ clusters; cluster #ex.afinni.layout.first_accepted,
for instance, was accepted and has no children. Children $2 j$ and $2 j + 1$ of the next
level therefore belong to the $j$-th rejected cluster, not to cluster $j$.

The queue also grows. Level #(ex.afinni.layout.level + 1) holds #ex.afinni.layout.next_points
points against #ex.afinni.layout.points, because the children keep all of their parents'
points and #ex.afinni.layout.new_points of the points are new, added to rejected clusters
too small to cut. The engine therefore never sizes a level in advance. The next level's
queue is a host buffer that starts at twice the current level's points and doubles when it
fills, and every append first checks that the machine has the memory for it. The next level
then runs in more batches, #ex.afinni.layout.next_batches instead of
#ex.afinni.layout.batches here, and only one batch is ever on the device (@sec:batches).

Three properties follow from this layout, and the engine relies on each of them. First,
every step of a batch is an operation on consecutive rows, with no lookup. Second, the
engine can cut a batch anywhere between clusters. Because the batches of a level run one
after another, the clusters at the end of the queue are the last to start, and a rank can
hand a block of them to an idle rank before it reaches them. This hand-over, by which idle
ranks take work from busy ones, is _work stealing_ (@sec:mpi). Third, children are written in the
order in which their parents were processed, so the order carries over from level to level:

#invariant(title: [queue order])[
  The engine takes batches in queue order, the plan cuts rejected clusters in queue order,
  and each rejected cluster appends its two children consecutively. Therefore, within one
  rank, the queue of every level is sorted by position in the forest. Nothing has to be
  sorted or looked up between levels, and a rank can give away the tail of its queue as one
  contiguous block.
] <inv:order>

The plans differ only in the rows of the table below. The test `tests/unit/test_plans.py`
checks that both provide every member of the `Plan` protocol in `src/core/plan.py`.

#figure(kind: table, caption: brief([What each plan gives the engine], [What each plan provides to the engine: the initial queue, the points of a cluster, the extra columns, the cut, the leaves at the depth limit and the work after the run.]), table(
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
)) <tab:plans>

#takeaway[The engine owns the queue, the batches, the memory and the ranks; a plan owns
  what a cluster is and what happens to one batch.]

= AFINNI: fit, test and what the errors say <sec:fit>

#question[A cluster of points with known values of $f$. Is one affine model enough for
  it?]

AFINNI begins by clustering the initial points $X_0$ into $Ninit$ groups with $k$-means,
and these groups are the roots of a forest of binary trees. At every level, AFINNI then
asks the same question of every cluster. To answer it, AFINNI fits the best affine model to
the cluster's points and measures how far the model is from $f$ at those same points.

#definition(title: [local model and cluster errors])[
  For a cluster $C$ with points $X_C$ and centre $c$ (their mean), the _local model_ is
  $fh(x) = w^top (x - c) + b$ with
  $ (w, b) = op("arg min")_(w, b) sum_(x_j in X_C) abs(f(x_j) - w^top (x_j - c) - b)^2 . $ <eq:fit>
  Its _absolute_ and _relative errors_ are
  $ e_a = max_(x in X_C) abs(f(x) - fh(x)), quad e_r = e_a / (max_(x in X_C) abs(f(x))) . $ <eq:errors>
  The cluster is _accepted_ if $e_a < atol$ or $e_r < rtol$, and _rejected_ otherwise.
] <def:errors>

#algorithm(caption: brief([AFINNI on one batch of clusters], [AFINNI on one batch of clusters (the manuscript's Algorithm 1, batched):
  `AFINNIPlan.process_batch` in `src/afinni/plan.py`.]), pseudocode(afinni)) <alg:afinni>

#story()
#step([Level 0: no cluster passes], code: afinni-code, focus: (1, 2, 3, 4), window: 1, width: 36%,
  stack(spacing: 4pt, panel("afinni-error-0"), error-key()))[
  The panel shows the #ex.afinni.levels.at(0).clusters initial clusters, with every point
  coloured by the error of its cluster's model (colours as in @sec:reading). Each cluster shows the same
  pattern: the model is too low at the rim and too high in the middle, or the reverse. That
  bowl is the curvature of $f$, the part an affine model cannot represent. The worst error
  is #ex.afinni.levels.at(0).worst_error, far above $atol = #ex.alpha$, so AFINNI rejects
  all #ex.afinni.levels.at(0).clusters clusters (red crosses).
]

#step([The errors shrink with the clusters], code: afinni-code, focus: (5,), window: 1, width: 36%, panel("afinni-error-2"))[
  Two levels later there are #ex.afinni.levels.at(2).clusters clusters, a quarter of the
  size. Since the error of an affine model falls with the square of a cluster's width, the
  colours have faded, and #ex.afinni.levels.at(2).accepted clusters now pass (dark crosses)
  and become leaves. The others, where $f$ bends most, are cut again.
]

#let error-strip = filmstrip(
  (panel("afinni-error-0"), [level 0: #ex.afinni.levels.at(0).worst_error]),
  (panel("afinni-error-1"), [level 1: #ex.afinni.levels.at(1).worst_error]),
  (panel("afinni-error-2"), [level 2: #ex.afinni.levels.at(2).worst_error]),
  (panel("afinni-error-3"), [level 3: #ex.afinni.levels.at(3).worst_error]),
  (panel("afinni-error-4"), [level 4: #ex.afinni.levels.at(4).worst_error]),
)
#figure(stack(spacing: 2pt, error-strip, align(center, error-key(width: 34%))), kind: image,
  caption: brief([The queue's clusters, coloured by model error], [The clusters still in the queue at each level, coloured by the error of their
    models, with the worst error of each level. Accepted clusters leave the
    picture, and the rest are cut and fitted again. After #ex.afinni.n_levels levels nothing is
    left.])) <fig:errors>

The board below shows the same run as clusters and as the forest they form.

#board(
  (panel("afinni-level-0"), [level 0: #ex.afinni.levels.at(0).clusters clusters, #ex.afinni.levels.at(0).accepted accepted]),
  (panel("afinni-level-1"), [level 1: #ex.afinni.levels.at(1).clusters clusters, #ex.afinni.levels.at(1).accepted accepted]),
  (panel("afinni-level-2"), [level 2: #ex.afinni.levels.at(2).clusters clusters, #ex.afinni.levels.at(2).accepted accepted]),
  (panel("afinni-level-3"), [level 3: #ex.afinni.levels.at(3).clusters clusters, #ex.afinni.levels.at(3).accepted accepted]),
  (panel("afinni-level-4"), [level 4: #ex.afinni.levels.at(4).clusters clusters, #ex.afinni.levels.at(4).accepted accepted]),
  (panel("afinni-level-5"), [level 5: #ex.afinni.levels.at(5).clusters clusters, #ex.afinni.levels.at(5).accepted accepted]),
)

Each frame shows the clusters of one level in colour and the leaves of earlier levels
faded, and beside them the forest down to that level, in which a node ringed dark was
accepted and a node ringed red was cut. The run ends with #ex.compare.afinni.leaves leaves after
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

One can therefore reason about the error of a leaf as a Taylor remainder, which to leading
order is $1/2 (x - c)^top H (x - c)$, the bowl in the pictures. The next section uses this
remainder to decide where to cut.

#implementation(title: [In the code])[
  `AffineLSQModel` (`src/components/local_models.py`) solves @eq:fit for every cluster of a
  batch at once. It adds a small ridge, $10^(-8) op("tr")(Sigma)$, so that a cluster lying in
  a subspace still has a solution. For a vector-valued $f$, $w$ has one row per component, and
  a cluster passes only if every component passes; each component can have its own
  tolerances. `error_metric="L2"` replaces the maxima in @eq:errors by root-mean-squares.
  Other local models are registered under `SurrogateMode` and `LOCAL_MODELS`: a Taylor
  model from oracle gradients, a constant and a quadratic.
]

#pitfall(title: [What acceptance does not say])[
  AFINNI tests the model on the points it was fitted to, so the test is optimistic.
  Moreover, $e_r$ divides by the cluster's largest $abs(f)$ rather than by each point's own,
  so an accepted leaf can hold a point where $f$ is near zero and whose own relative error
  is large. The tolerance is a property of the training points, and @sec:compare measures
  what it gives on the whole domain.
]

#takeaway[AFINNI accepts a cluster when its best affine model is within the tolerance on its
  own points, and the errors that remain show the curvature of $f$.]

= AFINNI: where to cut <sec:split>

#question[A cluster is rejected and must be cut in two. Along which direction?]

The obvious answer is the cluster's longest direction; cutting across it amounts to
$2$-means clustering, or to a cut across the first principal component @ding2004. That
answer looks only at where the points are, but a cut is useful only if the two halves are
easier to fit than the whole, and that depends on $f$.

#filmstrip(
  arrows: false,
  (panel("afinni-split-error"), [a rejected cluster: \ the error of its model]),
  (panel("afinni-split-plain"), [cut across the longest direction: \ children's worst error #ex.afinni.split.plain_error]),
  (panel("afinni-split-hessian"), [Hessian-informed cut: \ children's worst error #ex.afinni.split.hessian_error]),
  caption: brief([A rejected cluster, cut two ways], [One rejected cluster of level #ex.afinni.split.level, cut two ways. The cluster has
    #ex.afinni.split.points points and worst error #ex.afinni.split.parent_error.
    Colours as in @sec:reading. *Left.* The error of the cluster's model forms stripes, because
    $f$ bends _across_ the cluster and hardly at all along it. *Middle.* A cut across the longest
    direction leaves both children with the full bend. *Right.* A cut across the bend,
    #ex.afinni.split.angle degrees away, leaves children that are nearly flat.]),
) <fig:split>

The stripes of the left frame are the level sets of the error, which is the Taylor
remainder $1/2 (x - c)^top H (x - c)$, so the cut should go across the direction in which
that quadratic form grows fastest. AFINNI finds that direction by measuring distances with
the curvature. It stretches the cluster by $absH^(1 slash 2)$, so that a direction in which
$f$ bends strongly becomes long, and then cuts the stretched cluster across _its_ longest
direction.

#theorem(title: [Hessian-informed two-means, @manuscript])[
  Let $f$ be a quadratic with Hessian $H$, possibly non-convex. Let $hat(X)$ hold the
  centred points of a cluster as its columns. Let
  $E = sum_(k = plus.minus 1) sum_(i in C_k) abs(f(x_i) - fh(x_i))$ be the sum of the two
  children's errors, with $fh$ the least-squares model of each child. Then the optimal
  continuous two-means indicator for $E$ is the first eigenvector of $hat(X)^top absH hat(X)$.
] <thm:split>

The code never forms the eigenvector of the $\#C times \#C$ matrix $hat(X)^top absH hat(X)$.
Let $tilde(x)_i = absH^(1 slash 2)(x_i - c)$ be the stretched points. Their $d times d$
matrix $sum_i tilde(x)_i tilde(x)_i^top$ has the same nonzero eigenvalues, and projecting the stretched points on its first eigenvector $u_1$ gives the same indicator.

== Estimating the curvature from values alone

@thm:split needs $absH$, but the oracle gives only values of $f$, so the estimate uses the
pattern the pictures show. The points where the Taylor expansion is nearly exact lie in a
small sublevel set of the remainder around $c$. That set is an ellipsoid with the shape of
$absH^(-1)$, so the scatter matrix of these points is proportional to $absH^(-1)$.

#algorithm(caption: brief([Hessian estimate and Hessian-informed split], [Hessian estimate and Hessian-informed split of one rejected cluster
  (the manuscript's Algorithms 2 and 3): `CovarianceHessianSplit` in
  `src/components/splitting.py`.]), pseudocode(splitting)) <alg:split>

#story()
#step([A gradient from the nearest points], code: split-code, focus: (1,), window: 1, width: 36%, panel("afinni-hess-gradient"))[
  The Taylor expansion needs $nabla f(c)$. AFINNI fits it by least squares to the values at
  the #ex.afinni.split.n_grad points nearest to $c$ (green) and to $f(c)$, because the
  expansion is most accurate near $c$.
]

#step([The level set of the Taylor error], code: split-code, focus: (2, 3, 4), window: 1, width: 36%, panel("afinni-hess-levelset"))[
  The #ex.afinni.split.n_cov points with the smallest Taylor error (pink) lie along the
  directions in which $f$ is nearly affine. The ellipse of their scatter matrix (pink) has
  the shape of $absH^(-1)$, long where $f$ is flat and short where it bends. The dark
  ellipse, which the algorithm never sees, is the exact $absH^(-1)$ at $c$, scaled to the
  same area for comparison.
]

#step([Stretch, project, cut at the median], code: split-code, focus: (5, 6, 7, 8), window: 1, width: 36%, panel("afinni-split-hessian"))[
  In the metric $absH^(1 slash 2)$, the short axis of that ellipse becomes the long one, and
  the arrow is the resulting direction $absH^(1 slash 2) u_1$. AFINNI sorts the points by their
  score $s_i$ along it and divides them into a lower half $Cm$ (pink) and an upper half
  $Cp$ (cyan).
]

#theorem(title: [the scatter of a level set estimates the Hessian, @manuscript])[
  Let $f$ be a convex, non-degenerate quadratic with Hessian $H$. Let $hat(f)_c$ be the
  Taylor expansion at $c$, and let $E_epsilon = {x : abs(f(x) - hat(f)_c (x)) <= epsilon}$.
  Let $x_1, dots, x_n$ be independent and uniform on $E_epsilon$, and let
  $hat(Sigma) = 1/n sum_i (x_i - c)(x_i - c)^top$. Then there are constants $K$ and
  $kappa_0$ such that, with probability $1 - e^(-v)$,
  $ norm(hat(Sigma)^(-1) - kappa_0 H)_2 lt.tilde K norm(H) sqrt((d + v) / n) . $
] <thm:hessian>

The factor $kappa_0$ does not matter, because the cut uses only the direction of the first
eigenvector. On the example, AFINNI split #ex.afinni.split.rejected clusters, and in
#ex.afinni.split.hessian_wins of them the Hessian-informed cut left the children with a
smaller worst error than the cut across the longest direction.

#note(title: [How many points each fit uses, and why])[
  Both fits trade a well-conditioned solve against staying close to $c$. The lower bounds
  follow from the fits, and the fractions of $Nmin$ are set empirically.
  - *Gradient: the $max(2 d, floor(Nmin slash 4))$ points nearest to $c$.* The gradient
    estimate $hat(g)$ has $d$ unknowns, so it needs at least $d$ points that do not lie in a
    subspace, and twice that number keeps the solve well conditioned when some of them
    nearly do. More points cost accuracy. The values of $f$ are exact, so the fit's only
    error is the Taylor remainder $1/2 (x - c)^top H (x - c)$, and for points within
    distance $r$ of $c$ the remainder moves the fitted gradient by $O(norm(H) r)$. Hence the
    fit uses the _nearest_ points, and few of them. The term $floor(Nmin slash 4)$ takes
    more of them when clusters are large, because in a large cluster they are still close
    to $c$.
  - *Level set: the $N_"cov" = floor(Nmin slash 3)$ points with the smallest Taylor error.*
    @thm:hessian bounds the error of $hat(Sigma)^(-1)$ by $sqrt((d + v) slash n)$, so the
    fit needs about $d$ points at the least, and more points help. But more points come
    from a wider set $E_epsilon$, whose shape the higher-order terms of $f$ and the error of
    $hat(g)$ distort. A third of $Nmin$ balances the two when $Nmin$ is well above $d$.
  - *Both stay local.* A cluster is cut only once it holds at least $2 Nmin$ points
    (@sec:augment), so the gradient uses at most an eighth of it and the level set at most a
    sixth.
]

#pitfall(title: [A level set smaller than the dimension])[
  $N_"cov" = floor(Nmin slash 3)$ has no lower bound that grows with $d$. With the default
  $Nmin = 100$ it is 33 points, so for $d > 33$ the level set cannot span every direction
  and $Sigma$ is singular. The code floors the eigenvalues of $Sigma$ at $10^(-12)$, so the
  directions that the level set never reached get the largest weight in
  $absH^(1 slash 2)$ instead of none. To avoid this, raise `min_pts_per_cluster` with the
  dimension. The default, `default_min_pts_per_cluster`, gives $100 dot 1.3^d$.
]

#implementation(title: [In the code])[
  - The cut is at the _median_ of the scores by default, so the two children have equal
    size and the forest stays balanced. `cut="sign"` cuts at $s_i = 0$ as in the
    manuscript's Algorithm 2.
  - The assumptions of @thm:hessian rarely hold exactly: $f$ is not quadratic, not convex,
    and the points are not uniform on a level set. Where $H$ is indefinite, the cut
    minimizes an envelope of the error, $abs(Delta x^top H Delta x) <= Delta x^top absH Delta x$.
    The envelope is tight only along the eigenvectors of one sign, so it is loose where $f$
    is saddle-shaped. In practice the estimate recovers the dominant directions, which is
    what the cut needs. Where an exact gradient or Hessian is available, the modes
    `AFFINE_COV_HESS_EXACT_GRAD` and `AFFINE_EXACT_HESS` use it.
  - Only $u_1$ is needed, so the code forms no eigendecomposition. Instead, `leading_eigvec`
    (`src/numerics/linalg.py`) squares the normalized matrix ten times, which amounts to a
    power iteration of $2^(10) = 1024$ steps. It then takes the column at the largest
    diagonal entry, applies the matrix itself three times, and fixes the sign so that the
    largest component is positive. Both backends run these steps with the same products
    (`det_matmul`), so a cluster's direction does not depend on its batch. The angle to the
    exact $u_1$ is about $(lambda_2 slash lambda_1)^(1024)$, below $3 dot 10^(-3)$ degrees
    when $lambda_2 slash lambda_1 <= 0.99$. A nearly tied pair of eigenvalues gives a unit
    vector in the plane of the two leading eigenvectors, along which the stretched points'
    scatter is within a factor $lambda_2 slash lambda_1$ of the largest, so that vector is
    as good a cut direction as $u_1$.
]

== Enough points to estimate anything <sec:augment>

#scene(side: left, width: 34%, panel("afinni-augment"))[
  A gradient, a Hessian and two children's models cannot be estimated from a handful of
  points. Before AFINNI cuts a rejected cluster with fewer than $2 Nmin$ points, it adds
  new points up to $max(2 \#C, 2 Nmin)$, so that each child starts with at least $Nmin$.
  In the picture, a cluster of #ex.afinni.augment.before points (dark) receives
  #ex.afinni.augment.new new ones (pink).

  A new point is a random convex combination of two of the cluster's points,
  $tilde(x) = lambda x_i + (1 - lambda) x_j$ with $lambda$ uniform in $[0, 1]$. AFINNI
  evaluates $f$ there. Generating the points is one array operation on a GPU and needs no
  knowledge of the cluster's boundary.
]

#lemma(title: [new points stay inside])[
  Every new point lies in the convex hull of its cluster's points (dashed in the picture).
] <lem:hull>

#pitfall(title: [The hull is not the cell])[
  @lem:hull is both a guarantee and a limit of the method. New points never leave the hull,
  so AFINNI never samples the space between the hulls of neighbouring clusters, and the new
  points also concentrate towards the middle of the hull. When AFINNI must generate many
  points from few, the leaves cover less of the domain, and more queries fall outside every
  trust region (@sec:trust). The remedy within AFINNI is a larger initial sample, whereas
  ART samples each region up to its boundary (@sec:regions).
]

#takeaway[AFINNI halves a cluster across the direction in which $f$ bends, not the one in
  which the points spread, by stretching it with an estimate of $absH^(1 slash 2)$.]

= AFINNI: where a leaf can be trusted <sec:trust>

#question[A query point arrives. Which leaf should answer it, and should the answer be
  believed?]

AFINNI keeps only the leaves, with no tree and no boundaries between them. Because a leaf's
model was tested only at the leaf's own points, each leaf carries a region that contains
those points.

#algorithm(caption: brief([The trust ellipsoid of an accepted cluster], [The trust ellipsoid of an accepted cluster (the manuscript's Algorithm 5): `CovarianceEllipsoid` in
  `src/components/trust_regions.py`. $gamma$ is `q_reg`.]), pseudocode(trust)) <alg:trust>

#story()
#step([An ellipsoid around a leaf's points], code: trust-code, focus: (1, 2, 3, 4), width: 36%, panel("afinni-trust-one"))[
  The scatter matrix $Sigma$ of the leaf's #ex.afinni.trust_points points gives the shape,
  and the ellipsoid ${x : (x - c)^top Sigma^(-1) (x - c) <= 1}$ is long in the directions
  where the points spread. Let $alpha$ be the largest value of
  $(x_i - c)^top Sigma^(-1) (x_i - c)$ over the points. Scaling the ellipsoid by $alpha$ puts
  the farthest point (ringed) exactly on the boundary and every other point inside.
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
  $Sigma^(-1)$ exists only if the leaf's points spread in every direction, which they need
  not do. $Sigma$ has a zero eigenvalue when the leaf has fewer than $d + 1$ points or when
  its points lie on a line or a plane. Points lie on a plane in a thin slab left by repeated
  cuts, and on a line when augmented points lie on a segment between two others. The ellipsoid would then have
  zero width across that direction, so every query off the points' subspace would have an
  infinite trust-region distance and be flagged, and the inversion itself would be
  unstable. Flooring the eigenvalues at $gamma$ gives the ellipsoid a minimum half-width
  $sqrt(gamma alpha)$ in every direction and caps how elongated the ellipsoid can be, while
  it leaves well-spread leaves untouched. In the code $gamma$ is `q_reg`, by default
  $10^(#ex.defaults.q_reg_exponent)$. It floors the eigenvalues of the scatter $Sigma$,
  which is a sum over the points and has the squared units of $x$, so $gamma$ sets a small
  absolute length rather than one relative to the leaf's size.
]

A minimum-volume enclosing ellipsoid @todd2016 would be tighter, but it needs an iterative
optimization per leaf, whereas the covariance ellipsoid needs a single eigendecomposition,
batched over all accepted clusters.

#algorithm(caption: brief([AFINNI's inference], [Inference (the manuscript's Algorithm 6): `AdaptiveSurrogate.predict`
  with `predictor="knn"`.]), pseudocode(inference)) <alg:inference>

#scene(side: left, width: 40%, panel("afinni-inference"))[
  At inference the ellipsoids have two uses. First, they give a _metric_, in which the
  distance from a query $x$ to leaf $i$ is $d_i = (x - c_i)^top T_i (x - c_i)$, so a step
  along a leaf's long axis counts as short. Second, they give a _flag_. If $d_i > 1$ for
  every candidate, no leaf's model was tested near $x$.

  *A*: the leaf whose centre is nearest in Euclidean distance (grey cross) is not the one
  that answers. The long leaf's centre is farther away, but $x$ lies well inside its
  ellipsoid ($d_nu = #ex.afinni.queries.A.d_nu$). \
  *B*: $x$ lies inside #ex.afinni.queries.B.inside ellipsoids, and the smallest $d_i$ decides. \
  *C*: $x$ lies inside no ellipsoid ($d_nu = #ex.afinni.queries.C.d_nu$). The nearest leaf
  still answers, and `predict` returns its answer _flagged_ (red).
]

Searching all $N_c$ leaves in their own metrics would cost $O(N_c d^2)$ per query, so
`predict` searches in two stages. It first runs an exact Euclidean search for $kappa$
candidates (`src/neighbors/knn.py`) and then computes the $kappa$ trust-region distances.
The first stage can miss the right leaf, because a long ellipsoid's centre can be far from a
query that the ellipsoid contains, as query A in the picture shows. Hence $kappa$ should
grow with the dimension and with how elongated the leaves are. It is set empirically, for
example to $kappa = 2 d^2$. When $kappa$ is too small, queries are flagged as outside
although a leaf contains them. @sec:exact-flag makes the flag
exact without searching every leaf.

#implementation(title: [In the code])[
  `predict(X, k_nn=1, coarse_factor=10)` searches $kappa =$ `k_nn * coarse_factor`
  candidates and returns the predictions with a mask of the flagged queries. With
  `k_nn > 1`, `predict` blends the best `k_nn` leaves with weights $1 slash d_i^2$, which
  smooths the jumps between leaves. On the running example, `predict` answers queries A, B
  and C with #ex.afinni.queries.A.pred, #ex.afinni.queries.B.pred and
  #ex.afinni.queries.C.pred, where the values of $f$ are #ex.afinni.queries.A.f,
  #ex.afinni.queries.B.f and #ex.afinni.queries.C.f.
]

#pitfall(title: [A trust region is not a certificate])[
  Inside a trust region the model was tested only at the leaf's points. The flag says that
  no leaf has data near the query, but its absence does not bound the error. In high
  dimension the ellipsoids cover a vanishing share of a box, so the flag is informative
  only if the sample lies where the queries will lie. The flag can also guide sampling, in
  a scheme called _iterative refinement_: draw new initial points, keep those outside every
  trust region, and train again.
]

#let ef = ex.afinni.exact_flag

== Is a flagged query really outside? <sec:exact-flag>

#question[The candidates are the $kappa$ centres nearest to the query. If none of their
  ellipsoids contains the query, could a leaf farther away still contain it? And can that be
  ruled out without measuring the query's distance to every leaf?]

#scene(side: left, width: 40%, panel("afinni-flag-miss"))[
  It can. In this picture $kappa = #ef.kappa$. The nearest centre (grey cross) lies
  right beside the query (black dot), but that leaf's ellipse does not contain the query, so
  the candidates flag it (#swatch(red, shape: "dot") red ring). A long ellipse does contain
  it, at $d = #ef.d_leaf$, but its centre is only the #ef.leaf_rank\th nearest. Over a grid of
  #ef.grid queries, one candidate flags #ef.flagged_percent% of the domain, while only
  #ef.exact_percent% lies outside every ellipse. In the running example
  $kappa = #ex.kappa$ finds every containing leaf, and #ex.compare.afinni.outside_percent%
  of the domain is flagged either way. No Euclidean $kappa$ guarantees this, however,
  because the longer the leaves are, the farther an ellipse reaches from its centre.
]

Measuring $d_i$ for every leaf costs $N_c d^2$ per flagged query. The search instead rules
out each leaf with the cheapest test that can rule it out, and no test ever rules out a
leaf that contains the query.

#scene(side: right, width: 40%, panel("afinni-flag-box"))[
  Every ellipse lies in an axis-aligned box, its _bounding box_, so a query outside the box
  is outside the ellipse. The box test takes $d$ comparisons and usually stops at the first. The ball of radius
  $1 slash sqrt(lambda_min)$ around a leaf also holds its ellipse, but a long leaf's ball is
  long in every direction. For the highlighted leaf the ellipse has area
  #ef.long_leaf.ellipse_area, its box #ef.long_leaf.box_area and its ball (dashed)
  #ef.long_leaf.ball_area. Averaged over the domain, a query lies in
  #ef.per_query.ellipses ellipses, #ef.per_query.boxes boxes and #ef.per_query.balls balls. A
  long thin ellipse in $d$ dimensions spans about $1 slash sqrt(d)$ of its length on each
  axis, so the box's advantage over the ball grows with $d$.
]

#lemma(title: [the bounding box of a trust region])[
  Every $x$ with $(x - c)^top T (x - c) <= 1$ satisfies $abs(x_k - c_k) <= sqrt((T^(-1))_(k k))$
  on every axis $k$, and each bound is reached at one point of the ellipsoid.
] <lem:box>

#proof[
  Write $x - c = T^(-1 slash 2) y$ with $norm(y) <= 1$. By Cauchy-Schwarz,
  $x_k - c_k = (T^(-1 slash 2) e_k)^top y <= norm(T^(-1 slash 2) e_k) = sqrt(e_k^top T^(-1) e_k)$,
  with equality for $y$ along $T^(-1 slash 2) e_k$.
]

A query inside a leaf's box need not be inside its ellipse, so before the search evaluates
the $d^2$ form, it applies a second test of $d$ operations, built from the ellipse's longest
and shortest axes.

#lemma(title: [a lower bound from the extreme axes])[
  Let $lambda_min$ and $lambda_max$ be the extreme eigenvalues of $T$, and let $u$ be the
  unit eigenvector of $lambda_max$. Then
  $ (x - c)^top T (x - c) >= lambda_min norm(x - c)^2 + (lambda_max - lambda_min) (u^top (x - c))^2 . $
] <lem:bound>

#proof[
  In $T$'s eigenbasis the form is $sum_j lambda_j y_j^2$, with $sum_j y_j^2 = norm(x - c)^2$.
  Every $lambda_j >= lambda_min$, and the coordinate along $u$ carries $lambda_max$.
]

#figure(filmstrip(
  ..ef.tree.levels.map(l => (panel("afinni-flag-tree-" + str(l.depth)),
    [depth #l.depth: in #l.entered of #l.tested tested])),
), kind: image, caption: brief([A query's walk down the tree of leaf boxes], [One query's walk down the tree of the leaves' boxes. The buckets
  hold #ef.tree.bucket leaves so that the tree shows four levels, whereas the code uses 16. At each
  depth the walk tests the query only against the boxes whose parent it entered. The query
  enters the #swatch(dark) dark boxes and falls outside the #swatch(red) red ones, whose
  subtrees the walk skips. The walk never tests the #swatch(rgb(col.light)) pale boxes.])) <fig:flag-tree>

The search gathers the boxes in a tree, in which each node holds the union of its leaves'
boxes. A query outside a node's box is outside the box of every leaf below it, so the walk
skips the whole subtree. In
@fig:flag-tree the query tests #ef.tree.tested of the #ef.tree.nodes nodes and reaches only
#ef.tree.tested_leaves of the #ef.tree.leaves leaves.

#translate(caption: brief([The bounding-box walk in symbols], [The walk of the exact search in words and in symbols: the boxes it tests, and when it tests a leaf.]),
  [a leaf's bounding box: its half-width on axis $k$], [$h_k = sqrt((T^(-1))_(k k))$],
  [a node's box: the union of its leaves' boxes], [$[min_i (c_i - h_i), max_i (c_i + h_i)]$],
  [the walk tests a leaf only if the query is in its box and in its node's], [$abs(x_k - c_(i,k)) <= h_(i,k)$ for all $k$],
)

#proposition(title: [the search is exact])[
  The walk returns exactly the leaves whose trust region contains the query, up to the
  rounding of the quadratic form itself. It keeps a (query, leaf) pair when both the
  search's evaluation of the form and `predict`'s evaluation give at most 1.
] <prop:exact-flag>

#proof[
  The walk skips a subtree only when the query lies outside the union of its leaves' boxes,
  and by @lem:box the query then lies outside their trust regions. It skips a leaf only
  when the query is outside its box or when the bound of @lem:bound exceeds 1, and in the
  second case the leaf's own form exceeds 1 too. The form decides every other leaf.
]

#implementation(title: [In the code])[
  - `predict(..., exact_outside=True)` (`src/neighbors/containment.py`) runs the search for
    the queries that the candidates flag, and the other queries keep their answers bit for
    bit. If a leaf contains a flagged query, `predict` takes the query's containing leaves,
    nearest first, as its candidates and answers from them.
  - The eigenvalues come from one batched `eigh` per model. The code lowers them by their
    rounding error before it forms the boxes and the bound, so rounding can only widen a box.
    The code then evaluates each pair that passes with `predict`'s own expression, so the
    search and the inference agree on a query that rounding places on a boundary. The code
    stores nothing with the model and builds the index at the first such call.
  - The index splits a node at the median of the axis whose two children have the smallest
    boxes relative to the node, because a query in the overlap of two children enters both.
    A query compares a node's sides starting from the narrowest, where the query is most
    likely to fall outside.
  - On the CPU the walk runs in parallel over the queries (Numba,
    `src/neighbors/containment_cpu.py`). On the GPU one kernel tests every (query, leaf) pair
    with @lem:bound before the form (`contained_mask`).
  - The launcher reports the exact share as `perct_out_exact`, beside the manuscript's
    $p_o$, which used the candidates only.
]

#pitfall(title: [The cost follows the flagged queries])[
  The search runs only for flagged queries, so it costs little where the queries lie near the
  training data. Far from the training data at high dimension, most queries are flagged and
  the boxes of long leaves overlap, so the walk visits a large part of the tree. Proving that
  a query lies outside every trust region is the expensive case.
]

#takeaway[The nearest leaf in the trust ellipsoids' metric answers a query, and a walk down
  a tree of bounding boxes finds exactly the queries outside every ellipsoid.]

= ART: a region for every node <sec:regions>

#question[AFINNI keeps no boundary for a cluster. What changes if every node of the tree
  owns a precise piece of the domain?]

Three things change. A node can be sampled up to its boundary, the leaves tile the domain
with no gaps, and a query can be sent to its leaf by a few comparisons. This is the design
of ART, the adaptive regression tree @art. ART needs a domain, and AFINNI does not. The domain is
a box (`domain=(lower, upper)`), and ART does all its geometry in the coordinates
$u in [0, 1]^d$ of that box.

#definition(title: [regions, planes and locational codes])[
  The root's _region_ is the domain, $Omega = [0, 1]^d$ in $u$. If the plane $(a, z)$, with
  $norm(a) = 1$, splits a node $C$, its children are
  $ Omega_(Cm) = Omega_C inter {u : a^top u <= z}, quad Omega_(Cp) = Omega_C inter {u : a^top u >= z} . $
  A point with $a^top u = z$ belongs to $Cp$. A region is therefore a convex polytope
  ${u : A u <= bold(z)}$, whose constraints are the box and one row $plus.minus (a, z)$ of
  $(A, bold(z))$ for every ancestor. The root has _locational code_ $pi = 1$, and the
  children of node $pi$ are $2 pi$ (for $Cm$) and $2 pi + 1$ (for $Cp$), so the code has one
  binary digit per level.
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

Each frame shows the nodes of one depth. A region outlined in dark was accepted, a region
outlined in red is cut in the next frame, and the leaves of earlier depths are faded. The
dots are the nodes' points. Dark dots were inherited from the parent, and open dots are new.
The pieces have edges, and ART refills them with points at every depth, so their points do
not thin out. AFINNI's frames in @sec:fit show the contrast. The run ends with #ex.compare.art.leaves leaves at
depths up to #ex.art.depth, after #ex.compare.art.evals evaluations of $f$.

#algorithm(caption: brief([ART on one batch of nodes], [ART on one batch of nodes: `ARTPlan.process_batch` in
  `src/art/plan.py`. The notes in code type are the leaf's recorded `terminal_reason`.]),
  pseudocode(art)) <alg:art>

== Naming a node: the locational code <sec:locational-code>

#question[A node is created on one rank, may be stolen by another, and is merged on a
  third. How should the node be named, so that any rank can work out its name, its parent's
  name and its children's names without asking another rank?]

The name must meet four requirements, and each rules out some naming schemes.

#cards(
  columns: 4,
  ([Local], [Any rank computes a child's name from its parent's name alone, without sending
    a message.]),
  ([Stable], [The name is the same whichever rank, batch or run builds the node, and it seeds
    the node's random numbers (@inv:rng).]),
  ([Arithmetic], [Parent, children and depth follow from the name, without a lookup.]),
  ([Small], [The name is one machine integer, carried with every queued node.]),
)

#cards(
  columns: 2,
  ([A global counter #text(fill: red)[✗]], [Numbering nodes $0, 1, 2, dots$ as they are created
    needs one counter shared by every rank and a message per node, and the numbers depend on
    the order of processing, so they are neither local nor stable.]),
  ([Coordinates #text(fill: red)[✗]], [A centre or a corner is a floating-point vector, which
    is neither small nor exact and says nothing about the parent.]),
  ([A Morton code #text(fill: red)[✗]], [It is local, stable and arithmetic, but it names only
    boxes cut at their midpoints (@sec:morton).]),
  ([The locational code #text(fill: rgb("#2e7d32"))[✓]], [It is the route from the root, one bit per
    cut, and it meets all four requirements.]),
)

#figure(panel("locational-code-walk"), caption: brief([A point's route to its leaf], [One point (black) on its way to its leaf. *Top.* The
  tree shows the point's route in red. Each step down adds one bit to the code, and the latest
  bit is red. *Bottom.* The same steps in the domain. Each frame shows the parent's region
  (grey), the plane that cut it, and the side the point is on (blue). The leaf is red.]))
  <fig:walk>

The root's code is $1$, and every cut appends one bit to a node's code, $0$ for the side
$a^top u < z$ of the plane and $1$ for the side $a^top u >= z$. The point in @fig:walk
takes #ex.art.locational_code.depth turns, so its leaf's code is
#raw(ex.art.locational_code.leaf), and reading the code from left to right replays the turns.

#definition(title: [locational code])[
  The root has locational code $pi = 1$. A node with locational code $pi$ has children $2 pi$
  (the side $a^top u < z$) and $2 pi + 1$ (the side $a^top u >= z$). The pair $(r, pi)$ of its
  root and its locational code identifies a node.
] <def:locational-code>

#scene(side: left, width: 34%, panel("art-forest"))[
  ART grows one tree per root cell, and each tree numbers its own nodes from $1$. The
  running example has one root, so $r$ is the same for every node there. With
  `roots="grid"` or `"voronoi"`, ART starts from several roots. The picture shows the same example with
  #ex.art.forest.roots grid roots. The code #raw(ex.art.forest.code) names a node in all
  #ex.art.forest.trees_with_code trees, two of which are marked, and only the pair
  $(r, pi)$ tells these nodes apart.

  The pair is needed in three places. The routing table stores each plane under $(r, pi)$
  and checks that no pair repeats. A stolen node carries both numbers, so the rank that
  steals it can tell which tree the node extends. The pair $(r, pi)$ also seeds the node's
  random numbers. With $pi$ alone, the nodes
  #raw(ex.art.forest.code) of different trees would draw the same numbers and place their
  samples at the same relative positions.
]

Every question about a node's place in the tree is a bit operation on its code:

#figure(kind: table, caption: brief([Bit operations on a locational code], [Questions about a node's place in the tree, each answered by arithmetic on its locational code, with the answer for one leaf of the running example.]), table(
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
)) <tab:code-ops>

The leading $1$ is the _place-holder bit_. Without it, the codes $0$, $00$ and $000$ would be
the same integer. With it, every route has its own number, and the numbers run
$1$; $2, 3$; $4, dots, 7$, one row per depth, which is the numbering of a binary heap.

#note(title: [The standard terms])[
  A code that records a node's route from the root is a _locational code_. Gargantini
  introduced it for quadtrees, where it has one base-4 digit per subdivision
  @gargantini1982. It is the standard term in the spatial-data-structure literature
  @samet2006. ART's locational code is binary, with one digit per cut. The leading $1$ is
  Warren and Salmon's _place-holder bit_, which they add "to distinguish the higher level
  internal nodes of the tree from the lowest level body nodes", and their root is also the
  key $1$ @warren1993. The code calls $pi$ `locational_code`.
]

=== Locational codes and Morton codes <sec:morton>

A _Morton code_ is the locational code of a quadtree, the tree that cuts every box into four
equal quarters. Both codes append bits level by level. A Morton code appends one bit per
coordinate at each level, so two per level in two dimensions. Because every cut is at a known place, the bits also
give the cell's position.

#filmstrip(
  arrows: false,
  (panel("morton-code", width: 92%), [Morton code #raw("1001"): quarter #raw("10"), then its
    quarter #raw("01")]),
  (panel("morton-art", width: 92%), [ART code #raw(ex.art.locational_code.leaf): six turns,
    and the shape needs the six planes]),
  (panel("code-capacity", width: 100%), [levels that fit in a 64-bit code]),
  caption: brief([What a locational code says about a cell], [What a code says about a cell. *Left.* The Morton code
    #raw("1001") picks the top-left quarter (#raw("10")) and then its bottom-right quarter
    (#raw("01")). The code alone gives the box. *Middle.* ART's leaf
    #raw(ex.art.locational_code.leaf) and the planes of its six ancestors, numbered by depth.
    Its code records which side of each plane the leaf is on, but not where the planes are.
    *Right.* How deep a tree a 64-bit code can name as the dimension grows.]),
) <fig:morton>

The two codes share three properties: an ancestor's code is a prefix of its descendant's
code, the code fits in one integer, and a child's code follows from its parent's alone.

A Morton code has one more property, which ART cannot use. Its cuts are always at the middle
of an axis, so the code determines the cell. ART cuts along oblique planes at offsets chosen
from the data (@sec:plane), and a string of side bits cannot give the position of such
planes. ART therefore stores the planes separately. The routing table holds each split
node's plane $(a, z)$ once, under that node's locational code, and a leaf's shape is the
intersection of its ancestors' half-spaces.

The dimension sets a second limit. A Morton code spends $d$ bits per level, so a 64-bit code
holds $floor(63 slash d)$ levels after the place-holder bit, and at $d = 40$
it holds only #ex.art.locational_code.morton_levels_d40. A locational code spends one bit per
cut in any dimension, which allows trees #ex.art.locational_code.max_depth levels deep.

Warren and Salmon's tree code uses the Morton code with a place-holder bit, so one key both
names an octree cell and locates it @warren1993. ART keeps only the naming part.

#implementation(title: [In the code])[
  - `root` and `locational_code` are two `int64` columns of every queued node
    (`src/core/clusters.py`). The function `node_keys(seed, root, locational_code)` in
    `src/art/rng.py` hashes them into the node's random-number key. They stay separate on
    purpose, because packing them into one integer, `root << shift | code`, would take bits
    from the code and lower the depth limit as the number of roots grows. Two columns cost
    8 bytes per queued node and keep the full depth for any number of roots.
  - A positive `int64` has 63 bits, and the place-holder bit takes one, so routes fit up to
    depth #ex.art.locational_code.max_depth (`MAX_DEPTH_LIMIT`), the deepest ART tree allowed.
  - Codes are sparse. At each depth they span $2^"depth"$ values, of which few are in use, so
    codes are keys and never array positions. When the library assembles the tree, the
    routing table gives every node a compact index, and routing follows these indices with
    one array lookup per level (@sec:plane).
]

#takeaway[A node's locational code is its route from the root, one bit per cut, so any
  rank can compute it and read off its parent, children and depth.]


== A fixed budget of points in every node

ART fits each node on at least $n = ceil(m p)$ points, where $p = d + 1$ is the number of
coefficients of an affine model and $m$ is the `sample_multiplier`. In the running example
$m = #ex.settings.art.sample_multiplier$ and $n = #ex.settings.art.node_points$. A node
inherits the points of its parent that fall on its side, which cost nothing, and draws the
rest uniformly in its own region. A node that inherits more than $n$ points keeps them all,
so a root, for example, holds every initial point that falls in it. With
`validation_samples`, a child also inherits its parent's validation points. ART therefore
fits and tests a node deep in the tree on as many points as the root, concentrated in the
node's small region.

#algorithm(caption: brief([Hit-and-run in a polytope], [Hit-and-run in a polytope @smith1984 @lovasz2003:
  `HitAndRunSampler` in `src/art/sampling.py`. Each new point has its own chain, started
  at a random inherited point.]), pseudocode(hitrun)) <alg:hitrun>

#story()
#step([A region is the box and its ancestors' planes], code: art-code, focus: ("in", 2), window: 0, width: 35%, panel("art-polytope"))[
  The node with locational code #ex.art.sampling.code is at depth #ex.art.sampling.depth.
  Its region (shaded) is bounded by the domain box (dark outline) and by the
  #ex.art.sampling.rows planes of its ancestors, numbered by depth, and each plane is one
  row of $(A, bold(z))$. The box bounds each coordinate directly, and here its top face
  closes the region. The node inherited #ex.art.sampling.inherited points (dark) and needs
  #ex.art.sampling.new new ones (pink) anywhere in the region, including next to its
  boundary.
]

#step([Hit-and-run: a random walk that stays inside], code: hitrun-code, focus: (2, 3, 4), width: 35%, panel("art-hit-and-run"))[
  ART draws uniformly in a polytope with a random walk. The walk starts at an inherited
  point (dark). Each numbered step picks a random direction $g$, finds the chord (dotted),
  which is the part of the line through the current point along $g$ that lies inside the
  region, and jumps to a uniform point of that chord. The ends of the chord come from one
  product $A g$. The panel draws #ex.art.sampling.walk_steps steps of an illustrative chain,
  while the library walks `walk_steps` steps per chain. A chain's end is independent of its
  start only in the limit (@prop:hitrun).
]

#proposition(title: [hit-and-run samples uniformly, @smith1984])[
  Let a region be bounded and convex, with an interior. Then the uniform distribution on
  the region is the unique stationary distribution of the hit-and-run walk, and the walk
  converges to it from any interior point.
] <prop:hitrun>

#pitfall(title: [Finitely many steps])[
  @prop:hitrun is a statement about the limit. A chain of $tau$ steps in a long thin
  region may not have crossed the region yet, and its points then cluster near their starts.
  The code therefore _checks_ each node's chains in three ways.
  - *Drift.* The code compares the chains' points halfway through the walk with those at its
    end. It compares them coordinate by coordinate and by the mean squared radius. If the points have drifted
    by more than `mixing_z` standard errors, the code continues those chains and doubles
    $tau$ from `walk_steps` up to `max_walk_steps`. The code skips this test for a node with
    fewer than four walking chains.
  - *Unmoved chains.* A step in a random direction changes every coordinate with
    probability one, so a chain that left a coordinate unchanged did not move. At a corner
    of the box, for example, a random line enters the region only with probability
    $2^(1 - d)$. The code walks such a chain again by sweeps along the coordinates, on a
    separate random stream.
  - *Feasibility.* A chain stops where it is if its start, a step or its end violates a face
    by more than rounding, or if its chord is not finite.
  The code still fits a node that fails any of the checks, but if that node becomes a leaf,
  its `validation_status` is `insufficient_sampling`, so a model with unmixed regions cannot
  pass unnoticed.
]

#invariant(title: [random numbers belong to nodes])[
  Every random number ART draws is a hash of the run's seed, the node's root and
  locational code, the purpose of the draw and a counter (`src/art/rng.py`), and ART advances
  no generator from batch to batch. The points a node draws are therefore the same whatever
  batch the node is in, whichever rank processes it and whenever it is stolen (@sec:mpi),
  and so are the node's fit and its plane.
] <inv:rng>

#implementation(title: [In the code])[
  - `roots="grid"` or `"voronoi"` starts from several roots instead of one: a regular grid
    of the domain, or the Voronoi cells of the initial clustering's centres. More roots
    give the ranks independent work from the first level on.
  - ART samples a root box exactly, without a walk.
  - The reference ART and this library's ART both fit every node by plain least squares
    with $lambda = 0$ (`leaf_fit="lstsq"`). A node's fit is a Cholesky solve of the normal equations when
    their reciprocal condition number is at least $10^(-10)$, and a QR solve on the node's
    points otherwise. The plane search's fits solve the normal equations directly and never
    switch to QR. They use Cholesky on a GPU and LU on a CPU. A solve fails on a GPU when
    the Gram matrix is not positive definite, and on a CPU when it is exactly singular. A
    candidate whose solve fails gets a NaN score, and the search never chooses it. AFINNI's ridge (`leaf_fit="tikhonov"`) would shrink
    the slope across a node of aspect ratio $r$ by a relative $10^(-8) r^2$, and deep, thin
    nodes reach aspect ratios at which that shrinkage matters.
  - `validation_samples` draws a second, independent set of points in every node, on which
    ART checks an accepted model again before the node becomes a leaf. ART splits a node
    that fails, and its validation points become training points of its children.
  - A node carries only its ancestors' planes as rows of $(A, bold(z))$, and the root cell is
    never stored as rows. The root cell enters the walk either as a box, which gives one
    interval per coordinate, or as a Voronoi cell, whose rows are the bisecting planes between
    its centre and the neighbouring centres.
  - On a GPU all chains of a batch share each kernel launch
    (`src/kernels/hit_and_run.cu`), with one thread per chain. The code launches the kernel
    once per half of the walk, once more for each doubling, and once for the sweeps of
    unmoved chains. Above $d = #ex.defaults.walk_max_d$ the GPU walk uses array operations
    instead, while the CPU walk always runs in Numba.
]

#takeaway[An ART node owns a polytope, the box cut by its ancestors' planes, and is
  fitted on a fixed number of points drawn uniformly inside it by hit-and-run.]

= ART: choosing the plane <sec:plane>

#question[A node is rejected. Of all the planes that cut its region in two, which one?]

AFINNI computes its cut in closed form from the curvature, whereas ART _searches_ for it. The
best plane is the one for which two affine models, one on each side, fit the node's points
best, measured by the same error that the children's least-squares fits will minimize.

#definition(title: [the split objective])[
  Consider a node with points $u_i$ and values $y_i = f(x_i) slash atol$. The _hard_ objective
  of a plane $(a, z)$ is the squared error of the two children's least-squares fits:
  $ "SSE"(a, z) = min_(phi_(-1), phi_1 "affine") sum_i cases(
    (y_i - phi_(-1)(u_i))^2 & "if" a^top u_i < z,
    (y_i - phi_1(u_i))^2 & "if" a^top u_i >= z ) $ <eq:hard>
  Only planes that leave at least $p$ points on each side are admitted.
] <def:objective>

#filmstrip(
  arrows: false,
  (panel("art-split-before"), [a rejected node: the error of its one model]),
  (panel("art-split-candidates"), [candidate directions; the best cut found (dashed), then refined (solid)]),
  (panel("art-split-after"), [the children's errors after ART's cut (solid); AFINNI's cut (dashed)]),
  caption: brief([The split search on one node], [The split search on the node with locational code
    #ex.art.split.code, which has #ex.art.split.points points; the error colours are those of
    @sec:reading. *Left.* One model leaves a mean squared error of #ex.art.split.sse_parent,
    in units of $atol^2$. *Middle.* The search tries cuts along
    #ex.art.split.directions directions, drawn as arrows. The red arrow is AFINNI's
    Hessian-informed direction. The best cut has error #ex.art.split.sse_bins (dashed), and the
    soft search turns and shifts it to reach #ex.art.split.sse_final (solid). *Right.* The
    errors of the two children's models after ART's cut (solid). The #swatch(red, shape:
    "dashed") red dashed line is where AFINNI would cut the same points, along its
    Hessian-informed direction at the median. Both cuts get one affine model on each side. ART's
    children then have a largest error of #ex.art.split.art_worst $atol$ and a mean squared error
    of #ex.art.split.art_mse. AFINNI's children have #ex.art.split.afinni_worst $atol$ and
    #ex.art.split.afinni_mse. Because the node, its points and the fits are the same, only the
    cut differs. AFINNI fixes the cut before fitting, while ART chooses the cut by how well the
    children fit. ART minimizes the squared error, so its smaller largest error here is a
    consequence, not a guarantee.]),
) <fig:search>

== A coarse search, then a smooth one

The hard objective changes only when the plane crosses a point, so it is a staircase with no
useful gradient. ART therefore minimizes it in two stages.

#algorithm(caption: brief([ART's split search], [ART's split search, implemented by `_bins_search` and `_soft_refine` in
  `src/art/plan.py` with the functions of `src/art/split_search.py`.]),
  pseudocode(search)) <alg:search>

#story()
#step([Many cuts along a few directions], code: search-code, focus: (1, 2, 3, 4), window: 0, width: 38%, panel("art-split-candidates"))[
  The first stage enumerates cuts. Along each of $n_"dir" = #ex.art.split.directions$
  directions, ART sorts the node's points and tries $n_"bins" - 1 = #(ex.art.split.n_bins - 1)$
  cuts at equal fractions of the points. The two fits of every candidate come from running
  sums over the bins, so one direction costs one pass over the points. ART then refits the
  best candidates exactly, and the winner is the dashed line.
]

#step([Soft gates make the objective smooth], code: search-code, focus: (5, 6), window: 0, width: 38%, panel("art-split-gates"))[
  The second stage descends a gradient, and for that it replaces the choice "left or right"
  by a _gate_. A point at signed distance $a^top u - z$ from the plane belongs to the right
  child with weight $sigma((a^top u - z) slash h)$. Here $sigma$ is the logistic function,
  and $h$ is the gate width, drawn as the dotted band. Far from the plane, each point is pink
  or cyan by its side, while in the band the two children share the points. The two models become
  weighted fits, and their error becomes a smooth function of $(a, z)$.
]

#scene(width: 46%, panel("art-objective"))[
  The picture plots both objectives against the offset $z$, for the chosen normal. The hard one (dark) is a staircase with a step at every point, and the soft one
  (cyan) follows it but is differentiable. A gradient method can therefore descend the soft
  objective and turn the normal as well as move the offset, which the enumeration cannot do.
  The red line is the chosen plane, and the band around it is the gate width.
]

#definition(title: [the soft objective])[
  Let $s_i = (a^top u_i - z) slash h$, let $delta$ be a small floor, and let $lambda >= 0$ be
  a fixed ridge. The weights are $W_1 (s) = delta + (1 - 2 delta) sigma(s)$ and
  $W_(-1) (s) = delta + (1 - 2 delta) sigma(-s)$. Let $phi_i = (u_i, 1)$, and let $B_k$ be the
  coefficients of the two models. The _soft_ objective is
  $ J_h (a, z) = min_(B_(-1), B_1) 1/n [ sum_i sum_(k = plus.minus 1) W_k (s_i) (y_i - B_k^top phi_i)^2 + lambda (norm(B_(-1))^2 + norm(B_1)^2) ] . $ <eq:soft>
] <def:soft>

The minimization over the models is a pair of weighted least-squares problems, each with a
closed-form solution, so ART optimizes only over the plane. This is _variable projection_
@golub1973, and the gradient of $J_h$ needs no derivative of the models.

#lemma(title: [the gradient of the soft objective])[
  Let $r_(k, i) = (y_i - B_k^top phi_i)^2$ at the minimizing models. Then
  $ nabla_a J_h = 1/(n h) sum_i (r_(1, i) - r_(-1, i)) (1 - 2 delta) sigma'(s_i) u_i, quad
    partial_z J_h = -1/(n h) sum_i (r_(1, i) - r_(-1, i)) (1 - 2 delta) sigma'(s_i) . $
] <lem:gradient>

#proof[
  $J_h$ is a minimum over $B$ of a function that is differentiable in $(a, z)$, and the
  minimizer is unique. Every weight is at least $delta > 0$, so each weighted Gram matrix is
  positive definite once the node's $phi_i$ span $RR^p$, and if $lambda > 0$ it is positive
  definite for any points. By the envelope theorem, the derivative of the minimum is the
  partial derivative at the minimizer, and in that partial derivative only the weights depend
  on $(a, z)$.
]

In the sum, each point in the band moves the plane towards the side whose model fits that
point worse. Each step follows the negative gradient, projected on the tangent plane of the
sphere $norm(a) = 1$, and after the step ART renormalizes $a$. A backtracking line search
chooses the step and keeps the models frozen while it does.

#lemma(title: [frozen models still give descent])[
  Let $F(theta, B)$ be the function that @eq:soft minimizes over $B$, at plane
  $theta = (a, z)$ and models $B$, and let $B_t$ be the minimizing models at the current
  plane $theta_t$. If a trial plane $theta'$ satisfies $F(theta', B_t) < J_h (theta_t)$,
  then $J_h (theta') < J_h (theta_t)$.
] <lem:frozen>

#proof[
  $J_h (theta') = min_B F(theta', B) <= F(theta', B_t) < J_h (theta_t)$.
]

So ART judges a trial step with the current models, which costs one pass over the points.
It refits the models only after it accepts a step. The lemma applies because the
fit and the objective use the same weights, floor and ridge.

#invariant(title: [the stored plane defines the children])[
  After the search, ART moves $z$ to the score of the first point on the right, so the stored
  predicate $a^top u >= z$ _defines_ the children. The code verifies point by point that the
  predicate reproduces the cut, and raises an error otherwise. Training and routing evaluate
  the predicate with the same function, `plane_scores`, so a point present at the split is
  routed to the same side on which it was fitted. A point drawn later into a child lies in
  the child's closed polytope only up to rounding, so if such a point lies within rounding of
  the plane, routing could still send it to the sibling.
] <inv:predicate>

#implementation(title: [In the code])[
  - ART keeps the soft result only if its two _hard_ fits are better than the
    enumeration's, so the soft search can only improve a split. On the example, it improved
    #ex.art.split.improved of the #ex.art.split.searched nodes that were split.
  - The gate width $h$ is `soft_temperature` $= #ex.art.split.temperature$ times the
    standard deviation of the scores $a^top u$, and the search takes at most
    `soft_iters` $= #ex.art.split.soft_iters$ steps.
  - From `affine_search_from` $= #ex.art.split.affine_search_from$ leaf coefficients on (quadratic
    leaves from $d = #range(1, 200).find(d => (d + 1) * (d + 2) / 2 >= ex.art.split.affine_search_from)$),
    both searches fit the two children with affine models $[u, 1]$. An affine model has $d + 1$
    coefficients instead of the leaf model's $p$, so a fit costs $(d + 1)^3$ instead of $p^3$. The bins search's
    best cuts and the soft result are still judged by the leaf model's hard fits, and the leaves are
    fitted with the leaf model. Below that size the saving is small, so the search ranks cuts by
    the model the leaf will use.
  - `split_search="bins"` stops after the enumeration, and `"gap"` cuts along the
    Hessian-informed direction near the median without any search.
  - ART first divides each component of the values by that component's tolerance, which is
    the larger of $atol$ and $rtol$ times the node's largest $abs(f)$. For vector-valued $f$,
    each component is therefore weighted by how far it is from _its_ tolerance.
]

== From a tree to an answer

#scene(width: 62%, panel("art-route"))[
  ART _routes_ a query down the tree, starting at the root. At every split node, routing
  evaluates $a^top u - z$ and goes right if the value is non-negative. The picture follows
  one query, the black dot, through the nested regions that contain it, down to its leaf at
  depth #ex.art.route.depth. The leaf's locational code, #ex.art.route.code, records the
  turns taken, and the leaf's model gives the answer.
]

#figure(panel("art-tree"), caption: brief([The tree of the running example], [The tree of the running example, with its locational
  codes in binary. A node's code is its parent's code with one digit appended. The digit is
  0 for the side $a^top u < z$ and 1 for the side $a^top u >= z$. Every code starts with a
  place-holder bit, the leading 1, and the number of digits after it is the depth.])) <fig:tree>

#theorem(title: [the leaves partition the domain])[
  Every point of the domain is routed to exactly one leaf and lies in that leaf's region.
  Routing costs one inner product per level, $O(d dot "depth")$ in total.
] <thm:partition>

#proof[
  At each split node, the predicate $a^top u >= z$ is either true or false, so routing
  enters exactly one child, and by @def:region a point in the parent's region lies in that
  child's region. The root's region is the whole domain.
]

#implementation(title: [In the code])[
  The `RoutingTable` (`src/art/routing.py`) stores the tree as arrays. For every node, it
  holds the two children, the plane and, for a leaf, the row in the leaf arrays. After training,
  rank 0 assembles the table from the split records $(pi, a, z)$ of all ranks and then
  _checks_ it: every split has both children, every node but a root has a
  parent, and no code appears twice. Both plans flag the answer to a query outside the region
  they cover. For AFINNI that region is the union of its trust regions. ART routes a query
  outside the domain from the nearest point of the domain.
]

#takeaway[ART enumerates cuts, refines the best one by descent on a smooth objective, and
  stores the plane, so every query reaches exactly one leaf.]
= Side by side <sec:compare>

#question[The same function, the same tolerance, two algorithms. What does each one give?]

#scene(side: left, width: 46%, panel("slice"))[
  The picture first shows one line across the valley, $x_2 = #ex.compare.slice_x2$. The
  dark curve is $f$, the grey band is $f plus.minus atol$, and each coloured segment is one
  leaf's affine model. Both surrogates are piecewise affine, stay inside the band almost
  everywhere, and have small jumps where one leaf's model ends and the next one begins. The
  pieces are short where $f$ curves and long where it is nearly straight, and this change of
  length is the adaptivity. #swatch(red) Red dots under AFINNI's curve mark where the line
  leaves every trust region, at #ex.compare.flagged_on_line of its
  #ex.compare.slice_points points, and there AFINNI flags its answer.
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
  caption: brief([Each surrogate's error over the domain], [The error of each surrogate over the whole domain, on a #ex.compare.grid
    $times$ #ex.compare.grid grid.]),
) <fig:error>

#figure(kind: table, caption: brief([AFINNI and ART on the running example], [AFINNI and ART on the running example at the same tolerance: the leaves, the evaluations of $f$, the share of the domain within the tolerance or flagged outside, and the largest error.]), table(
  columns: (1fr, auto, auto),
  inset: (x: 8pt, y: 5pt),
  align: (left, right, right),
  table.header[On the running example, $atol = #ex.alpha$][AFINNI][ART],
  [leaves], [#ex.compare.afinni.leaves], [#ex.compare.art.leaves],
  [evaluations of $f$], [#ex.compare.afinni.evals], [#ex.compare.art.evals],
  [of which at new points, besides the centres],[#ex.compare.afinni.generated], [#ex.compare.art.generated],
  [share of the domain within the tolerance], [#ex.compare.afinni.within_percent%], [#ex.compare.art.within_percent%],
  [share of the domain flagged], [#ex.compare.afinni.outside_percent%], [#ex.compare.art.outside_percent%],
  [the same, with the exact search (@sec:exact-flag)], [#ex.compare.afinni.outside_exact_percent%], [--],
  [largest error on the grid, in units of $atol$], [#ex.compare.afinni.max_error], [#ex.compare.art.max_error],
)) <tab:compare>

Neither surrogate is within the tolerance everywhere, and neither claims to be. Training
enforces the tolerance at the training points, and between them a model can exceed it,
mostly near the edges of leaves. The two algorithms differ in what each can say about that error.

#cards(
  columns: 2,
  ([AFINNI], [AFINNI works from the sample it is given, wherever that sample lies, and
    evaluates $f$ beyond the sample only inside the hulls of its clusters. Its flag marks the
    queries for which no leaf has data, which here lie at the rim of the box, beyond the
    outermost points. Inside the trust regions,
    #ex.compare.afinni.within_inside_percent% of the grid is within the tolerance.]),
  ([ART], [ART covers the whole domain by construction and flags nothing inside it. Because
    its leaves partition a known domain, an independent check is possible: `audit` draws
    fresh uniform points and bounds the share of the domain that violates the tolerance.]),
)

#proposition(title: [the audit bound, @clopper1934])[
  Draw $n$ independent uniform points of the domain, and let $k$ of them violate the
  tolerance. Then, with confidence $eta$, the share of the domain that violates the
  tolerance is at most the Clopper-Pearson bound, the $eta$-quantile of the
  $"Beta"(k + 1, n - k)$ distribution, which for $k = 0$ equals $1 - (1 - eta)^(1 slash n)$.
] <prop:audit>

On the example, `audit` draws #ex.art.audit.points fresh points and finds
#ex.art.audit.violations violations (#ex.art.audit.rate_percent%), so with
#ex.art.audit.confidence_percent% confidence at most #ex.art.audit.bound_percent% of the
domain violates the tolerance. The bound holds for the trained model and the uniform
measure; it does not bound the error at any particular point.

#note(title: [Which one])[
  The two algorithms suit different situations. Use AFINNI when
  the data come first: a large sample from a simulation, no natural box around it, and the
  need to know when a query leaves the region the data cover. Use ART when the domain
  comes first: a box that must be covered, an oracle that can be called anywhere in it,
  and the wish for a tree, exact routing and an audit. `examples/example05_afinni_vs_art.py`
  runs both on the same problems.
]

#takeaway[Both meet the tolerance on almost all of the domain with a few dozen affine
  pieces; AFINNI flags where it has no data, and ART covers the whole domain and can be
  audited.]

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
  A batch holds the points `X`, their values `F`, and two arrays with one entry per cluster.
  Every cluster is a run of consecutive rows (@fig:queue). `counts` holds the number of
  points in each cluster, and its running sums, `starts`, give the row where each run begins.
  `labels` gives each row's cluster and comes from a cumulative sum of start markers, without
  a sort. To sum over clusters, the kernel cuts every run into chunks of
  #ex.defaults.segment_chunk rows (two in the picture), _counted from the run's own first
  row_. One thread adds up one entry over one chunk, in row order, and a second pass adds each
  cluster's chunks in order. This is a _segmented reduction_ (`segment_cross`), and one launch
  covers every cluster of the batch.
]

Almost every step of @alg:afinni and @alg:art is one of four kinds of array operation on
this layout.

#cards(
  columns: 2,
  ([Per row], [Projections, residuals and distances to a centre are computed row by row, with
    one thread for each row. The thread finds its cluster's matrix or plane through `labels` and
    reads it in place, so no row needs its own copy.]),
  ([Segmented reductions], [Sums, scatter matrices and maxima are taken over each cluster's
    rows. They give the centroids, $Sigma$, the normal equations and the largest error.]),
  ([Small batched solves], [One thread solves each cluster's $p times p$ system or forms
    its $d times d$ product. Eigendecompositions run as one batched call.]),
  ([Sorts within clusters], [A stable sort by value, followed by a stable sort by cluster,
    puts every cluster's rows in order and keeps them in one run.]),
)

#figure(kind: table, caption: brief([Steps, kinds of operation and kernels], [Each step of the two plans, the kind of array operation it is, and the kernel or library call that runs it.]), table(
  columns: (1.25fr, 1fr, 1.45fr),
  inset: (x: 5pt, y: 4.5pt),
  align: left + horizon,
  table.header[Step][Kind][Kernel or library call],
  table.cell(colspan: 3, fill: luma(245))[*AFINNI* (`src/afinni/plan.py`, `src/components/`)],
  [centres, means, $Sigma$, the fit's normal equations], [segmented sums], [`segment_cross`; tiled through shared memory when the result has many entries],
  [the fit, the products with $absH^(1 slash 2)$], [small solves], [`batched_inplace_cholesky_solve`, `spd_solve`, `det_matmul`],
  [errors $e_a$, split scores, Mahalanobis distances], [per row], [`fused_split_projection`, `fused_mahalanobis_projection`, `taylor_residual_norm`],
  [largest error, trust radius], [segmented maxima], [`segmented_max_float64`, one thread per cluster],
  [cut direction], [repeated squaring], [`det_matmul` (`leading_eigvec`)],
  [Hessian estimate, trust ellipsoid], [eigendecompositions], [CuPy's batched `eigh` (cuSOLVER)],
  [exact outside flag (@sec:exact-flag)], [per (query, leaf) pair], [`contained_mask`, one thread per pair],
  [the median cut, the points nearest each centre], [sorts within clusters], [two stable `argsort`s over the whole batch],
  [new points (@sec:augment)], [per new coordinate], [one element-wise kernel],
  table.cell(colspan: 3, fill: luma(245))[*ART* (`src/art/`)],
  [hit-and-run in every rejected node], [per chain], [`hr_walk`, one thread per chain],
  [cut statistics for every bin], [segmented sums], [`segment_cross` over (node, bin) runs, then `bin_sides`, `spd_solve`],
  [soft search], [per row and per node, repeated], [`soft_gates`, `soft_residuals`, `soft_row_terms`, `soft_node_prep`, `soft_trial_terms`, `soft_node_select`],
  [children, their codes and regions], [per node], [index arithmetic and one gather],
)) <tab:kernels>

The project's own kernels live in `src/kernels/*.cu`, one file per area, and NVRTC compiles
each kernel when it is first used. Each kernel exists for one of three reasons: a library
routine would sum in an order that depends on the batch (@sec:same-bits), a chain of library
calls would build a temporary of one matrix per row, or many small launches would cost more
than the work.

The engine splits a level into batches because the level may not fit on the device.

#figure(kind: table, caption: brief([What a level's size asks of the engine], [The constraints that a level's size puts on the engine, and what the engine does about each one.]), table(
  columns: (1fr, 1.3fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[Constraint][What the engine does],
  [the points of a level can exceed the device's memory], [keeps the queue in host memory and moves one batch at a time to the device],
  [a batch must fit on the device], [caps `batch_size` (points) at 65% of the free device memory divided by `plan.bytes_per_point()`, and keeps it between `plan.batch_floor` and $2 dot 10^7$. An ART batch is the longest run of whole nodes that fits, and a node larger than a batch runs alone. An AFINNI batch ends at the first cluster that crosses the next multiple of `batch_size`, so it can overshoot by that cluster, however large. A cluster that starts beyond the next multiple runs alone, and the later multiples move with it.],
  [a batch's points must be released as early as possible], [hands them to the plan (`DeviceBatch.take`), which frees them before it augments and cuts],
  [the host can run out of memory too], [checks the available memory on every append to the queue, and raises `MemoryError` with advice],
)) <tab:constraints>

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
  (@sec:split). The engine sorts the whole batch by score and then, stably, by cluster, so
  each cluster stays in its own run with its rows in order of score (middle row). The median
  then needs no search, because the first $floor(n slash 2)$ rows of a run are $Cm$ (pale)
  and the rest are $Cp$. Since the children are already runs of consecutive rows, one gather
  `X[order]` writes the next level in the layout this section began with.
]

Both plans order a batch's children with this sort. It takes two passes of CuPy's stable
`argsort` instead of one lexicographic sort, because CuPy implements a lexicographic sort as
a slower merge sort. ART then also checks that its stored plane puts every row on the side
given by the row's position, so that routing a training point later reproduces the split
exactly (`ARTPlan._split`).

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

To sample a rejected ART node, hit-and-run needs the node's polytope $Omega_C$, which is the
root's box cut by every ancestor's plane (@sec:regions). On the GPU, every node _carries_
these planes as a block of rows $[a^top | z]$, one row per ancestor, so the planes are never
looked up in a tree. Each row is the constraint $a^top u <= z$.

#figure(panel("art-queue-blocks"), caption: brief([ART's queue, level by level], [ART's queue in the running example, level by
  level, drawn as the planes its nodes carry. Each node is a block labelled by its
  locational code, with one row per ancestor plane in the colour of the node that made the
  plane. A hatched row is negated, because the node lies on the plane's right side. A
  rejected node (outlined) writes two children, whose blocks are its own block plus one row.
  An accepted node (pale) moves to the leaves, and the queue drops its block. On the right
  are the shape of each level's array and the number of its nodes that were accepted.])) <fig:queue-blocks>

Read the figure from top to bottom, where $h$ is the level. The number of nodes $K$ at the
next level is twice the number rejected at this one, so $K$ grows only while nodes are rejected
(#qb.map(r => str(r.at(0))).join(", ") here). For a batch, the children's blocks take one
gather of the parents' blocks and one concatenation. The rest of a child's identity is index
arithmetic too: its code is $2 pi$ or $2 pi + 1$, its root is its parent's, and its random
numbers are keyed by $(r, pi)$ (@inv:rng). @fig:carry follows one node down the tree and
shows the numbers in the node's rows together with the plane each row describes.

#figure(
  filmstrip(..range(carry.codes.len()).map(t => (panel("art-carry-" + str(t)), [
    $pi = #carry.codes.at(t)$ #v(2pt) #carry-block(t)]))),
  kind: image,
  caption: brief([A node and the planes it carries], [Node #carry.codes.at(-1) of @fig:queue-blocks and its ancestors, from the root
    down. Each frame shades the node, each numbered plane has the colour of the node it
    split, and the plane's arrow marks the side the node keeps. Under each frame is the
    node's block, with one row $a_1, a_2 | z$ per plane in the unit coordinates $u$. A right
    child's row is negated ($-a^top u <= -z$).]),
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
  offset $z$. The `hr_walk` kernel runs one thread per chain and one chain per new point. A
  chain's label is its node's row in that array, so each thread reads its node's block in
  place. The chains of one node are consecutive threads that read the same block, and the
  cache serves it to all of them. Because every node has the same $h$, the threads of a
  warp loop in step.
]

A queue holds one level, and even a stolen queue comes from one level of one rank, so the
nodes of a batch always share their depth. The kernel would accept mixed depths if a short
block were padded with rows $0 <= 0$, which cut no chord, but the engine never forms such a
batch. Carrying the planes costs $h (d + 1)$ numbers per node, and every child copies them
again. One shared table of planes, indexed by the ancestors' codes ($pi$,
$floor(pi slash 2)$, $floor(pi slash 4)$, ...), would store each plane once. A stolen node
would then need the table too, however, and every chain step would gather planes from
scattered places in memory.

=== Every candidate cut in one pass

#let bins = ex.art.split.bins
The bins search (@sec:plane) scores $n_"bins" - 1$ cuts along each of
#ex.defaults.n_directions directions in every node. Refitting each candidate cut from its
points would read every point once per cut.

#figure(
  filmstrip(arrows: false,
    (panel("art-bins-points"), [#bins.n_bins bins along one direction, each boundary a candidate cut]),
    (panel("art-bins-errors"), [the error each cut leaves, with the chosen cut in gold])),
  kind: image,
  caption: brief([The bins search along the chosen direction], [The bins search on the node of @fig:search, along the direction it chose.
    *Left.* The node's #ex.art.split.points points, sorted by score and split into
    #bins.n_bins bins of about #bins.per_bin points. Every boundary between bins (grey) is a
    candidate cut. The chosen cut is drawn gold and dashed, the same mark it has in
    @fig:search. It leaves the pink points on the left and the cyan points on the right. *Right.* The mean squared error of the two
    fits that each cut leaves, in units of $atol^2$. Cut #bins.chosen_cut leaves an error of
    #bins.chosen_error.
    The dashed line is the error of one model fitted to the whole node.]),
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
  After the sort, each bin's rows are contiguous. One segmented reduction over the
  $K dot n_"bins"$ (node, bin) runs then makes one pass over the points and gives every
  bin's least-squares sums $S_b$, which are the products $Phi^top Phi$, $Phi^top y$ and
  $y^top y$ of the bin's rows. A cut only decides which whole bins go left, so its two fits
  need just the sums $L_j$ and $R_j$ and one small $p times p$ solve each, whatever the
  number of points.
]

The kernel `bin_sides` runs one thread per (node, entry of the sums). Each thread builds
every $L_j$ in one walk over the bins from left to right, and every $R_j$ in one walk from
right to left. It sums the right side directly rather than as the total minus $L_j$, because
that subtraction would cancel digits when one side is small.

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

#figure(ls-picture, kind: image, caption: brief([One step of the soft search], [A schematic of one step of the soft search for
  five nodes (rows). Column $m$ is the trial step $alpha 2^(-m)$. #swatch(rgb(col.gold), shape: "dot")
  Gold is the step a node takes, dark grey marks the trials before it, and light grey marks
  trials that were evaluated but not needed. The CPU tries one halving at a time for all
  nodes and asks the host (#text(fill: rgb(col.red))[◆]) whether any node is still waiting.
  The GPU evaluates the first four halvings of every node in one launch, and the other eight
  in a second launch only for the nodes still waiting.]))
  <fig:line-search>

The soft search (@sec:plane) runs at most #ex.art.split.soft_iters steps. Each step tries
the step sizes $alpha$, $alpha slash 2$, $alpha slash 4$, ... until one lowers the objective
enough, and each node takes its first passing trial. On the GPU, `soft_trial_terms`
evaluates these trials in two launches (@fig:line-search), and `soft_node_select` then
picks each node's first passing trial. Each node advances on the device or records there
why it stopped, and the host reads one flag per step and ends the loop when every node has
stopped.

== Where it is still hard

#figure(kind: table, caption: brief([Where the GPU path is still slow], [The parts of the GPU path that remain slow, why each one is slow, and what could help.]), table(
  columns: (1fr, 1.5fr, 1.3fr),
  inset: (x: 5pt, y: 4.5pt),
  align: left + horizon,
  table.header[Problem][Why][What could help],
  [*Uneven clusters*], [`segmented_max_float64` gives a cluster one thread, however many rows
    it has. In `hr_walk`, a thread's work grows with steps $times$ planes $times d$, so one
    deep node at high dimension delays the whole launch.], [chunking large clusters,
    which `segment_cross` already does, and one warp per chain at high $d$],
  [*Sequential steps*], [The soft search, the doubling of drifting hit-and-run chains
    (@alg:hitrun) and the walk down an ART tree are each a loop of launches, and each pass
    has one host read that decides whether to go on.], [checking the stop flag every few
    steps instead of every step],
  [*The host in the loop*], [The queue stays in host memory. The engine copies every batch
    in from pageable host memory and copies its children out through one reused page-locked
    buffer, and no work overlaps these copies. A batch also makes dozens of small host
    reads, mostly of the sizes of masks.], [copying
    the next batch in while the current one runs],
  [*Carried planes*], [The engine charges them per point when it chooses the batch size, so
    deep ART levels run in smaller batches. The `hr_walk` kernel keeps a chain's coordinates in a fixed per-thread
    array, so beyond $d = #ex.defaults.walk_max_d$ dimensions a slower path built from array operations takes over.], [a shared plane
    table and per-node indices],
  [*AFINNI's random streams*], [AFINNI uses one generator per batch, seeded by a hash of the
    seed, the level, the batch's index and the rank, so its new points depend on which
    clusters share a batch (@prop:determinism).], [keying each stream by its cluster.
    ART already keys its streams by node.],
  [*AFINNI's trust test*], [`predict` gathers the trust matrices of all $kappa$ candidates
    of every query, which is $kappa d^2$ numbers per query.], [one kernel that evaluates the
    quadratic forms in place],
)) <tab:hard>

== Answering queries

Prediction runs on chunks of a few thousand queries, so its temporaries fit on the device
however many queries there are.

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

#figure(afinni-pipeline, kind: image, caption: brief([AFINNI's prediction for a chunk of queries], [AFINNI's prediction for a chunk of $Q$
  queries (six here, one colour each). Every array has one row per query, and every step
  does the same work for every row, so the whole chunk moves through each step at once.]))
  <fig:afinni-pipeline>

No step in @fig:afinni-pipeline depends on another query, and every query takes the same
branches, so the threads of a warp stay in step. The nearest-centre search starts with a
tiled float32 kernel that proposes $k + max(8, k)$ candidates, which the search then ranks
again by float64 distances. A bound on the float32 rounding certifies that no omitted centre
is nearer, and the search repeats in float64 every row that the bound cannot certify, so the
search is exact, with ties going to the smaller index. This search and the $k$-means of the
initial clustering are the library's own CuPy code, because cuML ships for Linux only and
the GPU backend also runs on Windows. Both give the same result on every run. The $k$-means,
however, assigns points in float32 on a GPU and without the certificate, so near a tie its
assignment is deterministic but not exact. The gathered trust matrices, $kappa d^2$ numbers
per query, are the largest array of the prediction.

The exact outside flag (@sec:exact-flag) is the exception, because the work it needs for a
query depends on where the query lies, and on a GPU that makes the threads of a warp
diverge. The GPU therefore tests every (flagged query, leaf) pair in one kernel, which needs
no tree and keeps the warps in step, while the CPU walks the tree.

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
  caption: brief([Queries routed through ART's tree], [Eight uniform queries of the running example, routed by the library. Each row is one pass and shows every query's node by its locational code. A query
    at a leaf (grey) is done. *Right.* The threads of each pass, one for each query not yet
    at a leaf, compacted into a dense list.]),
) <fig:passes>

Queries reach leaves at different depths. If each thread walked its query down to its leaf, a
warp would run as long as its deepest query, and the lanes of finished queries would sit idle.
`RoutingTable.route` instead moves every unfinished query one level per pass. Each pass first
compacts the unfinished queries into a dense list (@fig:passes, right), and then one thread
per listed query reads its node's plane, takes one dot product and moves the query to a
child. Every thread of a pass runs the same instructions, and finished queries take no
thread. A query costs $O(d)$ per level of its path. Two costs remain:

- *Scattered reads.* The queries of a warp sit in different nodes, so the planes they read
  are not adjacent in memory. The table of planes is small, however, at $d + 1$ numbers per
  node, and it stays in the cache.
- *One pass per level.* Each pass costs a few launches and one host read, of the length of
  the list. The last passes move few queries, so there the launches dominate. One kernel that
  walks each query to its leaf would remove these launches but bring back the idle lanes.
  That price is small for trees of even depth, and such a kernel remains an open option.

== The same bits every time <sec:same-bits>

Floating-point addition is not associative, so when many GPU threads add into one sum in
whatever order they finish, the sum differs from run to run in its last bits. A last bit
can decide whether a cluster passes its tolerance, and so decide the shape of the whole tree.

#invariant(title: [deterministic sums])[
  Every sum on the training path is taken in a fixed order that depends only on the
  cluster it belongs to. On the GPU, the kernels sum a cluster's rows sequentially in
  chunks counted from the cluster's own first row, and then add the chunks in order
  (`segment_cross`, `reduce_sum_1d`, `det_matmul`, `spd_solve` in `src/numerics/`), so no
  thread adds into a shared floating-point total. The training path uses no batched library
  product or solve, because the algorithm of such a routine can change with the size of the
  batch.
] <inv:sums>

The only library routine left on the training path is the batched symmetric
eigendecomposition. The tests train ART on both backends, with a batch size of 97 and with
the default batch size, and require the same number of leaves, the same number of
evaluations and the same predictions to within $10^(-10)$.


#proposition(title: [what does not change the result])[
  + Repeated runs in one process on the same backend give bit-identical models.
  + Consider an ART tree grown with hit-and-run sampling. The set of its planes by
    locational code does not depend on the batch size, the number of ranks or the schedule
    of steals. On a GPU, this holds up to the batched eigendecomposition above.
  + With work stealing off, the backend, the seed, the effective batch size and the number
    of ranks fix an AFINNI forest.
] <prop:determinism>

#proof[
  (1) follows from @inv:sums and the seeded random streams. (2) A node's points depend only
  on its key (@inv:rng), its sums only on its own rows (@inv:sums), and its plane only on
  those points and sums, so by induction from the roots this holds for every node. The
  argument needs $f$ to evaluate each row independently of the other rows in its call, which
  the batched ODE integrator does by controlling each trajectory's step on its own. (3)
  AFINNI draws its new points from one generator per batch, seeded by a `SeedSequence` of
  the seed, the level, the batch's index and the rank, so the new points depend on which
  clusters share a batch. The batch size and the scatter over ranks fix that grouping, but a
  steal moves clusters between batches. On a GPU, the engine derives the effective batch size
  from the free device memory, so the batch size also depends on the card and on whatever
  else holds memory on it. The two backends draw from different generators (NumPy's
  PCG64, CuPy's XORWOW) and run different numbers of $k$-means iterations.
]

#pitfall(title: [CPU and GPU])[
  The two backends take their sums in different orders, so a kernel and its NumPy
  counterpart agree to rounding, about $10^(-12)$, but not bit for bit. Whole AFINNI
  trainings differ by more, because augmentation draws from different generators on the two
  backends and the initial $k$-means runs two Lloyd iterations on a CPU but up to 300 on a
  GPU. The two backends therefore grow different forests of comparable accuracy, and no test
  compares a CPU training with a GPU training.
]

#takeaway[Clusters are runs of rows, batches are runs of clusters, and each sum follows
  an order that only its own cluster fixes, so a GPU trains fast and gives the same bits on
  every run.]

= Many ranks <sec:mpi>

#question[Refinement is uneven: one region of the domain may need ten times the levels of
  another. How do many processes share work that nobody can predict?]

Rank 0 evaluates $f$ on the initial points, builds the initial queue and cuts it into $P$
contiguous runs with equal numbers of clusters, one per rank. From then on, every rank
runs @alg:engine on its own queue. A static division cannot stay balanced, because the work
a cluster will generate is unknown until a rank has fitted the cluster. A rank whose
clusters are accepted early runs out of work while another rank is many levels deep.

Idle ranks therefore _steal_ work @blumofe1999. A thief is a rank with an empty queue, and a
victim is a rank with work to spare.

#algorithm(caption: brief([Decentralized work stealing], [Decentralized work stealing, run by every rank $i$ (the manuscript's
  Algorithm 7), implemented by `WorkStealer` in `src/engine/work_stealing.py`.]),
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
  A rank's load is the number of points waiting in its queue. Each rank sends its load to
  all others at every level boundary and after every steal, in an 8-byte message that no
  rank waits for. A thief asks the rank with the highest published load. The victim checks
  for requests between batches and, in ART, also during its soft search. It never gives
  away the batch it is working on, and it answers every request with a grant or a denial.
  After a grant, the victim sends the clusters themselves, whole clusters from the _tail_
  of its queue, which it would have reached last.
]

#invariant(title: [the stealing protocol])[
  + *One reply.* Every request receives exactly one grant or denial. A thief waits for
    that explicit reply and for nothing else, so a stale view of the loads can delay it
    but not deadlock it.
  + *Only unstarted work.* A grant takes whole clusters from behind the current batch and
    always leaves the victim at least one. Exactly one rank processes each cluster.
  + *A stolen cluster is complete.* It carries its points, values and depth, and for ART
    also its root, its locational code and the rows of its region, so the thief needs
    nothing else from the victim, then or later.
  + *No rank retires early.* A rank retires only when every rank is idle. An idle rank
    joins a non-blocking barrier and keeps answering requests until all ranks have joined.
] <inv:steal>

== Four recorded runs

#let sr = ex.steal.runs
#let run-names = (root_static: [one root, no stealing], root_steal: [one root, stealing],
  grid_static: [#ex.steal.ranks roots, no stealing], grid_steal: [#ex.steal.ranks roots, stealing])
Stealing depends on timing, so the pictures show recorded runs. Each run trains ART on the
running example on #ex.steal.ranks ranks. The runs use the tighter tolerance
$atol = #ex.steal.atol$.
The runs start in two ways, and each start runs once without stealing and once with it.
With one root, rank 0 holds all the work at first. With #ex.steal.ranks grid roots, one per
rank, every rank starts with its own cell of the domain, which is the usual way to share
work without stealing. Each configuration ran #ex.steal.repeats times after a warm-up run,
with the configurations interleaved, and each rank had a core of its own, all cores having
the same speed. The pictures show the run with the median wall time.

#figure(panel("steal-timeline"), caption: brief([When each rank worked, in four runs], [When each rank was working, in the four runs. A rank's bar
  is coloured while the rank has nodes and pale while it is idle. Each arrow is a grant, from
  the victim to the thief. The slow first run has its own time axis, and the other three
  share one. Above each run are its wall time and the share of the ranks' time spent idle.])) <fig:timeline>

#figure(
  board(columns: 4, ..run-names.pairs().map(((n, label)) =>
    (panel("steal-owners-" + n.replace("_", "-")), label))),
  kind: image,
  caption: brief([Which rank built each leaf], [Which rank built each leaf, in the same four runs. A stolen node takes its whole
    subtree with it, so a rank's leaves form patches of the domain.]),
) <fig:owners>

#figure(kind: table, caption: brief([The four recorded work-stealing runs], [The four recorded runs: wall time, speedup, idle time, leaves, leaves per rank and steal grants.]), text(9.5pt, table(
  columns: (auto, auto, auto, auto, auto, auto, 1fr, auto),
  inset: (x: 5pt, y: 4.5pt),
  align: (left + horizon, right + horizon, center + horizon, right + horizon, right + horizon, right + horizon, center + horizon, right + horizon),
  table.header[Run][Wall time][#ex.steal.repeats runs][Speedup][Idle][Leaves][Leaves per rank][Grants],
  ..run-names.pairs().map(((n, label)) => {
    let r = sr.at(n)
    (label, [#r.wall s], [#r.wall_range s], [#r.speedup×], [#r.idle_percent%], [#r.leaves_per_rank.sum()],
      [#r.leaves_per_rank.map(str).join(", ")], [#r.grants])
  }).flatten(),
))) <tab:steal-runs>

With one root and no stealing, ranks 1 to #(ex.steal.ranks - 1) never get work, so rank 0
builds all #ex.steal.leaves leaves alone and #sr.root_static.idle_percent% of the ranks'
time is idle. Stealing spreads the same tree over all ranks and makes the run
#sr.root_steal.speedup times faster. With one root per rank, every rank starts working at
once, but the cells are not equally hard. The ranks whose cells need fewer levels finish
early and wait, so #sr.grid_static.idle_percent% of the ranks' time is idle. With stealing,
these ranks take the unstarted nodes of the busy ranks, idle time falls to
#sr.grid_steal.idle_percent%, and the run builds the same tree in #sr.grid_steal.wall s
instead of #sr.grid_static.wall s.

The gain from stealing depends on how uneven the start is. This function's valley crosses
every cell, so a static split is only mildly uneven and the gain is modest. When the work
starts in one place, the gain is several-fold. Which region of an adaptive tree will need
more levels cannot be known in advance, and stealing makes the run time nearly independent
of how the work starts. In each pair, both runs build the same tree, which the recording
script checks by comparing every plane bit for bit, matched by root and locational code
(@prop:determinism).

#implementation(title: [In the code])[
  - The threshold $w_*$ (`steal_min_load`) and the steal size $w_s$ (`steal_size`) are
    counted in points. AFINNI grants when more than $10^6$ unstarted points remain, and ART
    grants when a few nodes' worth of points remain. A grant gives about
    $max(w_s, "half the unstarted work")$.
  - ART's split search can take long, so ART also answers requests _inside_ a batch,
    through `batch.progress()`, though it still gives away only clusters behind the batch.
  - Under MPI, an exception on one rank would leave the others waiting in a collective
    call, so `run_collective` logs the rank, the phase and the traceback and aborts every
    rank. In a single process, the exception propagates.
  - `python -m src.infra.monitor` shows each rank's level, queue and steals during the
    run. It reads files that the ranks write, so it adds no MPI traffic.
]

#takeaway[Ranks start with equal shares and stay busy by stealing unstarted clusters from
  the busiest rank, and for ART the tree does not depend on which rank built it.]

= How we know it is right <sec:verify>

#question[The code runs thousands of batched operations on two backends and many ranks. What
  stands between a change to the code and a wrong surrogate?]

#cards(
  columns: 3,
  ([Definitions], [The unit tests check each component against its formula on small
    cases: the fit recovers an affine function exactly, the errors match @eq:errors, the
    trust ellipsoid matches @lem:trust, and a plane reproduces the cut it was made from.]),
  ([Both plans, one contract], [`test_plans.py` checks that AFINNI and ART follow the
    `Plan` protocol, checks their batch boundaries, and checks that AFINNI releases a
    batch's points before splitting.]),
  ([Differential regression], [`scripts/regression_check.py` trains small 2-D cases and two
    8-D cases, whose quadratic leaves reach the GPU's large-system kernels, and stores every
    array of every model. The cases cover every mode, the `L2` and `Linf` metrics and every
    split search. A refactor must reproduce them bit for bit on a CPU, on one rank and on two.]),
  ([On the GPU], [The GPU suite runs the whole test suite on CuPy. It compares a kernel with its
    NumPy reference to $10^(-12)$ and two GPU trainings with each other to $10^(-10)$, and
    leaf counts and evaluation counts must match exactly. No test compares a CPU training
    with a GPU training.]),
  ([Many ranks], [`tests/mpi` runs real `mpiexec` jobs that test scatter and gather,
    stealing with gradients, ART's tree against a single-rank run, and that a failure on
    one rank aborts all ranks.]),
  ([Structure], [`test_layering.py` enforces the import layers. On any machine,
    `test_kernel_sources.py` checks that every CUDA kernel has a caller and that every
    call has a kernel.]),
)

The library itself runs three checks, because they concern a particular model more than
the code. It validates the routing table when it assembles the table and again when it
loads a model (@sec:plane). ART reports every leaf's `terminal_reason` and
`validation_status`, so the report shows whether a tree has fallback leaves or unmixed
samples. Finally, `audit` measures a trained ART model on fresh points (@prop:audit).

This companion's figures are a test of the same kind. The script that draws them rebuilds
AFINNI's forest from the recorded queue and ART's polytopes from the stored planes. It also
rebuilds the affine fits, the Hessian estimate and the inference rule from their formulas,
and it stops if any of them disagrees with the library.

#takeaway[Tests check formulas on small cases and compare whole models bit for bit across
  refactors and ranks, and to rounding on a GPU, while a trained model reports its own
  diagnostics.]

= Using it <sec:using>

// The settings are filled in from `ex`, so the listing shows the running example's values.
#raw(block: true, lang: "python", ```
from src.config import ARTSettings, TrainingConfig
from src.core.evaluation import Evaluator
from src.surrogate import AdaptiveSurrogate

evaluator = Evaluator(f_fn=f)                    # f maps (N, d) points to (N, fd) values

afinni = AdaptiveSurrogate()                     # AFINNI: from a sample X0
afinni.train(X0, TrainingConfig(atol={atol}, num_initial_clusters={n_init}, min_pts_per_cluster={n_min}), evaluator)
values, outside = afinni.predict(queries, k_nn=1, coarse_factor={kappa})
values, outside = afinni.predict(queries, exact_outside=True)  # flag exact for any kappa

art = AdaptiveSurrogate()                        # ART: on a domain
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

These are the main settings of the running example, which also fixes `seed` and
`max_levels`. Two environment variables choose the backend: `SURROGATE_DEVICE=cpu|gpu|auto`
and `SURROGATE_MPI=off|on|auto`. Under `mpiexec`, `train` and `predict` are collective
calls. `docs/USAGE.md` lists every setting, `docs/ARCHITECTURE.md` maps
the stages above to the modules of `src/`, and `docs/AFINNI_ART_DESIGN.md` records the
design decisions behind them.

#bibliography("references.yml", title: [References])

#show: appendix
#pagebreak(weak: true)

// Written by figures/manuscript_tables.py: the manuscript's numbers from
// studies/sanity/paper/paper_values.py, the re-run's from committed records of
// studies/sanity/paper/run_manuscript.sh, and Appendix A's size rule from cases.py.
#let ms = json("figures/manuscript.json")

= Experimental setup <sec:study-config>

// studies/manuscript/spec.py is this appendix as code: a change to one is a change to the other.

This appendix specifies every experiment of the paper in full: the test functions, the accuracy
target, the algorithm's settings, the reported quantities and how they are aggregated. Each setting
follows one rule in every experiment, except in the experiments that vary it, and one command from
the repository produces everything here (@sec:study-reproduce).

== Test functions

#figure(kind: table, caption: brief([Test functions], [The test functions, their domains and the dimensions at which each one runs.]), table(
  columns: (auto, 1fr, auto, auto),
  inset: (x: 5pt, y: 4pt),
  align: left + horizon,
  table.header[Function][$f(x)$][Domain][$d$],
  [Quadratic], [$1 + x^top Q Lambda Q^top x$, $Lambda = "diag"(5^(-1), ..., 5^(-d))$], [$[-3, 3]^d$], [2, 5, 10],
  [Gaussian], [$1 + exp(-1/2 x^top Q D Q^top x)$, $D = "diag"(1^3, 2^3, ..., d^3)$], [$[-3, 3]^d$], [2, 5, 10],
  [Rosenbrock], [$sum_(i=1)^(d-1) 100 (x_(i+1) - x_i^2)^2 + (1 - x_i)^2$], [$[-2, 3]^d$], [2, 5, 10],
  [Wave], [$3/2 + cos(2 q^top x)$, $|q| = 1$], [$[-5, 5]^d$], [2, 5, 10],
  [Lorenz-96], [$x_1(t = 0.1)$ for $dot(x)_j = (x_(j+1) - x_(j-2)) x_(j-1) - x_j + 1$, indices cyclic, $x(0) = x$], [$[-5, 5]^d$], [5, 10, 20, 40, 80],
)) <tab:functions>

In the table, $Q$ is a random rotation, the orthogonal factor of a matrix of standard normal
entries, with determinant $+1$, and $q$ is a random unit vector. Both are drawn once from a fixed
seed, so they belong to the definition of the function, and an experiment's seed does not change
them.

For every function, the initial training sample and the test points are drawn uniformly from the
domain in the table. The points that AFINNI and ART add during training also lie in the domain, and
$f$ is evaluated at all of them in the same way.

For Lorenz-96, a point $x$ is an initial state of the ODE in the table, with forcing $1$, and $f(x)$
is the first component of the state that the ODE reaches from $x$ at time $t = 0.1$. To evaluate
$f(x)$, the code integrates the trajectory that starts at $x$ up to $t = 0.1$ in float64, with the
Dormand-Prince 5(4) pair (`integrate_ensemble` in `src/numerics/ode.py`). Each trajectory has its own
adaptive step size, controlled as in SciPy's `solve_ivp(method="RK45")`. Hairer's rule chooses the
first step, and a step is accepted when the RMS norm of its local error over the $d$ components is
within $r_"tol" = a_"tol" = 10^(-6)$. The last step ends exactly at $t = 0.1$, so a point's value is
the same in every batch, on the GPU and on the CPU. At this tolerance the integrator's error is far
below every accuracy target of the experiments.

== Accuracy target and reported quantities

A surrogate value $hat(f)(x)$ is within the tolerance $"tol"$ when every output component $k$
satisfies $|hat(f)_k - f_k| <= a_"tol"$ or $|hat(f)_k - f_k| <= r_"tol" |f_k|$. The absolute and
relative tolerances are equal, $a_"tol" = r_"tol" = "tol"$. For Lorenz-96 the test is absolute
only, and $a_"tol" = "tol" dot max|f|$, where the maximum is taken over the training sample. The
experiments use the tolerances $"tol" = 0.1, 0.05, 0.02, 0.01, 0.005$.

#figure(kind: table, caption: brief([Reported quantities], [The quantities that the result tables report, and how each one is measured.]), table(
  columns: (auto, 1fr),
  inset: (x: 5pt, y: 4pt),
  align: left + horizon,
  table.header[Quantity][Definition],
  [$N$], [the number of evaluations of $f$ during training, including those of the initial sample],
  [$N_c$], [the number of leaves of the trained surrogate],
  [$p_o$], [the percentage of test points flagged as outside the trust region of every leaf, reported for AFINNI only. The candidate search decides the flag, unless a table names the exact search (@sec:study-settings). An ART model routes every query in the domain to a leaf, so ART's cells are "–".],
  [$e_"in"$, $e$], [the relative $L^2$ error $norm(hat(f) - f) \/ norm(f)$, over the test points inside a trust region and over all test points, respectively. Only AFINNI reports $e_"in"$. For ART every test point is inside, so ART's cells are "–".],
  [$e_"nh"$], [the error $e$ of the same run without Hessian-informed splits],
  [$p_"in,good"$], [the percentage of the test points inside a trust region that are within the tolerance],
  [$t_"train"$], [the wall time of training in seconds, measured from the evaluated initial sample to the trained model. It includes every evaluation of $f$ at the points that training adds, but not the evaluation of the initial sample.],
  [$t_"inf"$], [the inference time per test point in microseconds. It is the wall time of one prediction call on all $10^5$ test points, divided by their number.],
)) <tab:quantities>

Besides the quantities above, each run records the same quantities under the exact search
(@sec:study-settings). It also records the $L^infinity$ errors and the distribution of each
point's error in units of that point's tolerance. Finally, it records its own times and memory
and, for each level, the number of clusters and points and the level's time and memory.

== Algorithm settings <sec:study-settings>

AFINNI and ART train on the same initial sample with the same seeds, and both are scored on the
same test points at the same tolerances. @tab:settings-shared gives the settings that the two
plans share, @tab:settings-afinni AFINNI's settings and @tab:settings-art ART's. Each setting is given with its name in the
code: a field of `TrainingConfig` or `ARTSettings`, an argument of `predict`, or a constant of
`studies/manuscript/spec.py`.

// The three settings tables share one layout: the setting, its value and its name in the code.
#let st-settings(head, caption, ..rows) = figure(kind: table, caption: caption, {
  show raw: set text(size: 0.9em)
  table(
    columns: (19%, 1fr, 28%),
    inset: (x: 5pt, y: 4pt),
    align: left + horizon,
    table.header(head, [Value], [In the code]),
    ..rows,
  )
})

#st-settings([Both plans], brief([Settings both plans share], [The settings that AFINNI and ART share.]),
  [Initial sample], [$N_0 = min(floor(5 dot 10^4 dot 1.3^d), 10^6)$ points drawn uniformly in the domain], [`spec.sizes(d)`],
  [Test points], [$10^5$ points drawn uniformly in the domain, one fixed set per function and dimension], [`spec.N_TEST` \ `spec.TEST_SEED`],
  [Seeds], [0, 1, 2, 3, 4. The seed changes the initial sample and the random choices of training, but not the function or the test points.], [`spec.SEEDS` \ `seed`],
  [Acceptance], [the test of the accuracy target above, on the largest error over a cluster's points], [`error_metric="Linf"` \ `atol`, `rtol`],
  [Batch size], [at most $10^7$ points moved to the GPU per batch, lowered when the GPU's free memory requires it. AFINNI's trees depend on where a batch ends (@sec:apx-batches).], [`batch_size=10_000_000`],
) <tab:settings-shared>

#st-settings([AFINNI], brief([AFINNI's settings], [AFINNI's own settings.]),
  [Initial clusters, cluster size], [$M = min(floor(100 dot 1.3^d), floor(N_0 \/ 32))$ and $K_0 = min(floor(25 dot 1.3^d), floor(N_0 \/ (2 M)))$. Before a split, each initial cluster is topped up to $2M$ points, and these sizes make the $K_0$ topped-up clusters fit in the sample. The initial clusters come from $k$-means.], [`num_initial_clusters` $= K_0$ \ `min_pts_per_cluster` $= M$ \ `initial_clustering_method=` `"kmeans"`],
  [Splitting], [Hessian-informed splits, with the Hessian estimated from function values. A rejected cluster is augmented by convex combinations of its points. Training stops at 1000 levels.], [`mode="affine_cov_hess"` \ `sampling_method=` `"convex_combination"` \ `max_levels=1000`],
  [Trust regions], [built from each leaf's point scatter. The scatter's eigenvalues are floored at $10^(-6)$.], [`q_reg=1e-6`],
  [Inference], [AFINNI's candidate search (@sec:trust), which the manuscript also uses. An exact Euclidean search finds the $kappa = 50$ centres nearest the query, and the same $kappa$ holds in every dimension. Among these candidates, the one leaf ($k_"nn" = 1$) nearest the query in its own trust metric answers the query. If none of the candidates' trust regions contains the query, the query is flagged as outside, and that leaf still answers it. Each run also completes the same predictions with the exact search (@sec:exact-flag), and the tables of @sec:study-search compare the two searches.], [`predict(X, k_nn=1,` `coarse_factor=50,` `strict_trust_region=False)` \ exact search: `exact_outside=True`],
) <tab:settings-afinni>

#figure(kind: table, caption: brief([Sample and cluster sizes by dimension], [The sample and cluster sizes by dimension, from the rules of the AFINNI settings (`studies/manuscript/spec.py`, `sizes`).]), {
  set text(9pt)
  table(
    columns: 4,
    inset: (x: 8pt, y: 3pt),
    align: center + horizon,
    table.header[$d$][$N_0$][$K_0$][$M$],
    ..ms.sizes.map(r => (r.d, r.N_0, r.K_0, r.M).map(v => {
      let t = str(v)
      // thousands separators, as the tables print counts
      let n = t.len()
      range(n).map(i => (if i > 0 and calc.rem(n - i, 3) == 0 { "," } else { "" }) + t.at(i)).join()
    })).flatten(),
  )
}) <tab:sizes>

#st-settings([ART], brief([ART's settings], [Every setting of ART.]),
  [Root], [one box over the domain], [`roots="box"` \ `domain`],
  [Leaf model], [affine, with $p = d + 1$ coefficients, or quadratic, with $p = (d+1)(d+2)\/2$ coefficients], [`local_model=` \ `"affine_lsq"` or `"quadratic_lsq"`],
  [Leaf fit], [least squares without a ridge. The fit is a Cholesky solve of the normal equations when their reciprocal condition number is at least $10^(-10)$, and a QR solve on the node's points otherwise.], [`leaf_fit="lstsq"`],
  [Points per node], [$n = ceil(m p)$ with $m = 50$], [`sample_multiplier=50`],
  [New points], [uniform in the root box. In a deeper node, each new point is the endpoint of a hit-and-run chain of $tau = 20$ steps that starts at a random inherited point (@alg:hitrun).], [`walk_steps=20`],
  [Mixing check], [A chain whose endpoint still drifts between its halfway step and its last step, by a standardized mean difference above 4, continues with its steps doubled, up to $4 tau = 80$ steps.], [`mixing_z=4` \ `max_walk_steps`],
  [Validation points], [none], [`validation_samples=0`],
  [Split directions], [$n_"dir" = 6$ directions per node: the Hessian-informed one, from the Hessian estimated from function values, further eigenvectors of that Hessian, the points' principal direction and random directions], [`n_directions=6`],
  [Split search], [the soft search (@alg:search). The bins search scores $n_"bins" - 1 = 15$ cuts along each direction from running sums, and refits the 3 best candidates directly, each side on its own points. The soft search then takes at most 200 steps from the best candidate, and ART keeps its plane only if the plane's two hard fits are better.], [`split_search="soft"` \ `n_bins=16` \ `n_screened=3` \ `soft_iters=200`],
  [Soft gate], [width $h$ equal to $0.1$ times the standard deviation of the node's scores, and weight floor $delta = 10^(-8)$], [`soft_temperature=0.1` \ `soft_weight_floor=1e-8`],
  [Affine search fits], [From $p = 100$ leaf coefficients, the search ranks its cuts with affine fits, and the leaf model judges the screened candidates and the soft result. In this study, that applies to quadratic leaves on Lorenz-96 at $d >= 20$.], [`affine_search_from=100`],
  [Split margin], [Each side of a split keeps at least $p$ points.], [`min_side_margin=0`],
  [Depth], [at most 62 levels, the deepest that the locational codes allow], [`max_levels=62`],
  [Prediction], [the model of the leaf that the query is routed to], [`predict(X)`],
) <tab:settings-art>


== Experiments

#figure(kind: table, caption: brief([Experiments], [The experiments of the study and the runs that each one makes.]), table(
  columns: (auto, 1fr),
  inset: (x: 5pt, y: 4pt),
  align: left + horizon,
  table.header[Experiment][Runs],
  [Convergence], [the Quadratic, Gaussian, Rosenbrock and Wave functions at $d = 2, 5, 10$ and Lorenz-96 at $d = 5$. Each case runs at every tolerance, once with and once without Hessian-informed splits ($e_"nh"$).],
  [Dimension], [Lorenz-96 at $d = 5, 10, 20, 40, 80$ and $"tol" = 0.1$],
  [Absolute tolerance], [Rosenbrock at $d = 5$ under the absolute test only. For $k = 0, 1, 2, 3$, the tolerance is $a_"tol" = 579.8 dot 2^(-k)$ and the initial sample has $N_0 = 5 dot 10^4 dot 2^(5k\/2)$ points, so that $N_0 prop a_"tol"^(-5\/2)$.],
  [Gradient], [Wave at $d = 5$ at every tolerance. The exact gradient replaces the estimated one.],
  [Run progression], [Rosenbrock at $d = 10$ and $"tol" = 0.05$, recorded level by level],
  [Distributed], [Rosenbrock at $d = 5$ with $N_0 = 10^5$, at $"tol" = 0.05$ and $0.005$. The runs use $P = 1, 2, 4, ..., 128$ CPU ranks of one node and seeds 0, 1, 2. Each runs once with work stealing and once without.],
  [Iterative refinement], [Gaussian, Quadratic, Rosenbrock and Wave at $d = 2$ and $"tol" = 0.05$. The refinement has ten rounds, and each round adds $0.4 N_0$ points drawn uniformly outside every trust region. The candidate points are drawn in batches whose size follows the share of candidates found outside so far.],
  [ART], [the Convergence, Dimension and Absolute tolerance experiments repeated with ART, once with affine and once with quadratic leaves. Their tables give one row per method for each setting.],
)) <tab:experiments>

The tables report the median over the seeds, and the data files give the values of every seed. A
case that does not fit in the machine's memory or time is reported as not run, and its cells are
empty.

== Reproducing the experiments <sec:study-reproduce>

The repository's locked environment (`uv sync --locked`) installs every dependency, and
`studies/manuscript` runs the experiments on any machine:

#raw(block: true, lang: "bash", "M=studies/manuscript/machines/workstation.toml   # or a cluster's profile
python -m studies.manuscript.run run --smoke --machine $M --seeds 0   # checks the machine in minutes
python -m studies.manuscript.run plan --machine $M                    # every case, with projected cost
python -m studies.manuscript.run run --machine $M                     # resumable
python -m studies.manuscript.collect                                  # data files and tables")

Each case runs in its own process. Before a case runs, the runner projects its time and memory
from the cases already measured. The runner skips a case, and records the reason, when the
projection exceeds the machine's limits, when the case's series already ran out of memory or time
at a looser tolerance, or when the same case did so at another seed. A case that fails is recorded
with the reason, and the other cases continue. The collection step writes one row per case. Each
row holds every quantity above, the case's settings and its provenance (commit, machine, library
versions). The collection step then builds the tables from those rows. Each table cell is the
median over the seeds, with the minimum, the maximum and the number of seeds beside it.

== Results: AFINNI and ART <sec:study-results>

// Written by figures/study_results.py (tables, figure data and the study-*.svg panels) from the study's collected data (studies/manuscript/collect.py).
#let st = json("figures/study.json")

The tables below give the results of the convergence, dimension and absolute-tolerance
experiments, with one row per method for each setting: AFINNI, ART with affine leaves and ART with
quadratic leaves. All three methods train on the same initial sample and are scored on the same
test points (@sec:study-settings). Each cell is the median over the finished seeds, and the last
column counts those seeds. A row whose cases have not finished gives the reason in place of
numbers. AFINNI predicts with its candidate search ($kappa = 50$, @sec:study-settings), which the
manuscript also uses. The tables of @sec:study-search compare it with the exact search. ART's
$p_o$, $e_"in"$ and $e_"nh"$ cells are "–". ART routes every query in the domain to a leaf, so no
query lies outside, and it has no Hessian-informed splits to remove. An AFINNI $e_"nh"$ cell is "–"
where the run without those splits ran out of memory or has not finished. The times are measured
in the runs themselves, which ran on the GPU named below. Each run had its node to itself.

Two groups of rows are exact by construction. The Quadratic is a quadratic polynomial, so ART with
quadratic leaves fits it in a single leaf, exactly up to rounding error, and accepts its root
without a single new evaluation. The Gaussian at $d = 10$ is the other group, and the callout below
explains it together with the Gaussian at $d = 5$.

#let st-gauss = st.figures.gaussian
#let st-g(d) = st-gauss.dims.find(r => r.d == d)
#pitfall(title: [The Gaussian's bump narrows with the dimension])[
  The Gaussian is $f = 1 + exp(-1/2 x^top Q D Q^top x)$ with $D_(i i) = i^3$, so each added
  dimension curves more steeply than the last, and the bump shrinks fast as $d$ grows
  (@fig:study-gaussian). In float64, $f = 1$ exactly wherever the exponential falls below half an
  ulp of 1.
  - At $d = 2$ the bump is broad: $f - 1$ exceeds the tightest tolerance,
    #st-gauss.tol, on #st-g(2).share of the domain, and every method has to resolve it.
  - At $d = 5$, $f = 1$ exactly on #st-g(5).exact of the domain, and $f - 1$ exceeds
    #st-gauss.tol on only #st-g(5).share of it. About #st-g(5).test of the $10^5$ test points and
    #st-g(5).train of the training points lie in the bump, so a few points decide a run's error.
    That is why the seeds spread so widely there, and why the error changes unevenly from one
    tolerance to the next.
  - At $d = 10$, $f = 1$ exactly at every sampled point: no training point and no test point
    falls in the bump. Every method then fits a constant, which is exact, so these rows say
    nothing about the methods.
]

#figure(panel("study-gaussian"), caption: brief([The Gaussian's bump against the dimension],
  [The study's Gaussian at uniform points of its domain, $[-3, 3]^d$. Left: the share of the
  domain where $f - 1$ exceeds a threshold $t$, against $t$, one line per dimension; the dashed
  line is the tightest tolerance, #st-gauss.tol. Right: the expected number of the $10^5$ test
  points where $f - 1$ exceeds #st-gauss.tol, on a log scale.]))
  <fig:study-gaussian>

#if st.progress.pending > 0 {
  pitfall(title: [Partial results])[
    These tables are a snapshot of a study that is still running: #st.progress.done of its
    #st.progress.cases cases have finished, #st.progress.failed failed because they ran out of host
    or GPU memory or time, #st.progress.skipped were skipped by the runner's rules, and
    #st.progress.pending have not run yet. The runner runs seed 0 of every case first, so many cells
    hold a single seed. The results come from commits #st.commits.map(raw).join(", ") on
    #st.gpus.join(", ").
  ]
} else [
  Of the study's #st.progress.cases cases, #st.progress.done finished, #st.progress.failed ran out
  of host or GPU memory or time, and #st.progress.skipped were skipped by the runner's rules. A row
  has fewer than five seeds when one of its seeds ran out of memory or time, since the runner then
  skips the seeds after it and the case's tighter tolerances. The runs come from commits
  #st.commits.map(raw).join(", ") on #st.gpus.join(", ").
]

#pitfall(title: [Inference times are upper bounds])[
  In every run, the first prediction call includes one-time set-up on the GPU, such as loading the
  compiled kernels and making the first allocations. That first call is AFINNI's candidate search
  or ART's prediction, so their $t_"inf"$ include the set-up. For a small model the set-up takes
  most of the call's time. The exact search, which AFINNI calls next, can then take less time
  although it does more work. Each $t_"inf"$ is therefore an upper bound, tightest for the large
  models. Each run times one call, and the tables give the median over the seeds.
]

// One study table: group rows span the table; a row without numbers gives its reason across the
// quantity columns; a thin rule separates the settings, a fainter one the methods of one setting.
// A table with "top" has two header rows: quantities over (cand., exact) pairs.
#let st-table(t) = {
  let n = t.columns
  figure(kind: table, caption: brief(eval(t.short, mode: "markup"), eval(t.caption, mode: "markup")), align(center, {
    set text(8pt)
    table(
      columns: n,
      inset: (x: 4pt, y: 2.4pt),
      align: (x, y) => if x < t.keys { left + horizon } else { center + horizon },
      stroke: none,
      table.header(
        ..t.top.map(c => table.cell(colspan: c.at("colspan", default: 1), rowspan: c.at("rowspan", default: 1),
          align: if c.at("colspan", default: 1) > 1 { center + bottom } else { auto }, eval(c.text, mode: "markup"))),
        ..t.header.map(h => eval(h, mode: "markup")),
        table.hline(stroke: 0.6pt),
      ),
      ..t.rows.map(r => if "group" in r {
        (table.cell(colspan: n, inset: (top: 5pt, bottom: 2.4pt, x: 4pt), eval(r.group, mode: "markup")),)
      } else {
        let lead = if r.first { (table.hline(stroke: 0.3pt + luma(190)),) } else { () }
        let body = if "status" in r {
          (..r.cells, table.cell(colspan: n - t.keys, text(fill: luma(110), emph(r.status))))
        } else { r.cells }
        lead + body
      }).flatten(),
      table.hline(stroke: 0.6pt),
    )
  }))
}

The two figures show the results that differ most between the methods. @fig:study-dimension plots
the cost of Lorenz-96 against the dimension, and @fig:study-convergence plots the error against
the evaluations for every problem of the convergence experiment. The tables after them give
every quantity of every setting.

#let st-fig = st.figures
#figure(panel("study-dimension"), caption: brief([Lorenz-96: cost against dimension], [Lorenz-96 at $"tol" = 0.1$: the evaluations (left)
  and the leaves (right) against the dimension. Each line is one method's median over the seeds,
  and its band spans the smallest to the largest seed. From $d = #st-fig.dims.at(0)$ to $d = #st-fig.dims.at(1)$, AFINNI's
  evaluations grow by a factor of #st-fig.afinni_growth, while ART's stay close to its initial
  sample. ART with quadratic leaves has no point at $d = #st-fig.dims.at(1)$, where it ran out of
  GPU memory. The three methods' errors change little with the dimension, and
  @tab:study-dimension gives them.])) <fig:study-dimension>

#figure(placement: auto, panel("study-convergence"), caption: brief([Error against evaluations, all problems], [Error against evaluations for every function and
  dimension of the convergence experiment. Each line is one method's median over the seeds, and its
  band spans the smallest to the largest seed. Each marker is one tolerance, so a line runs from
  the loosest tolerance to the tightest, and a point lower and further left is a smaller error for
  fewer evaluations of $f$. The shaded panels,
  #eval(st-fig.focus.join(", ", last: " and "), mode: "markup"), have the largest ratio of
  evaluations or of error between AFINNI and either ART variant. A log axis cannot show an error
  that is exact to rounding, so ART with quadratic leaves, which fits the Quadratic exactly, has no
  line in the Quadratic's panels.])) <fig:study-convergence>

#for t in st.tables.filter(t => not t.key.ends-with("-search")) [#st-table(t) #label("tab:study-" + t.key)]

== AFINNI: the candidate search against the exact search <sec:study-search>

The tables above take AFINNI's predictions from its candidate search. The candidate search has one
parameter, $kappa = 50$. It takes the $kappa$ centres nearest the query in the Euclidean metric and
looks among their leaves for one whose trust region holds the query. When the leaf that contains a
query is not among these candidates, the search flags the query as outside, although the query is
inside a trust region. It then answers the query with a candidate leaf whose model was never
checked against the function near that query. The exact search (@sec:exact-flag) has no parameter.
It runs only on the queries that the candidate search flags, and it finds every leaf whose trust
region contains the query by walking a tree of the leaves' bounding boxes (@prop:exact-flag). A
flagged query that some leaf contains is then answered by those leaves, the nearest in their trust
metric first. A query that no leaf contains stays flagged and keeps its candidate answer. The
queries that the candidate search did not flag keep their answers bit for bit. Each run predicts
the test points both ways with the same trained model. The tables below give AFINNI's $p_o$, errors
and inference time under each search. The columns "cand." give the candidate search alone, which
are the predictions of the tables above. The columns "exact" give the same predictions completed by
the exact search. ART needs no such table, because it routes each query to one leaf.

#figure(panel("study-search"), caption: brief([AFINNI: the candidate against the exact search], [AFINNI with the exact search (filled) and with the
  candidate search (open) on the #st-fig.search_settings of its #st-fig.search_total convergence
  settings where the candidate search flags at least #st-fig.search_gap times as many test points
  outside. The rows are grouped by problem, each problem's tolerances run from the loosest to the
  tightest, and a line separates the problems. The left panel gives the share flagged
  outside, $p_o$, and the right panel the error $e$. In the other settings, the two searches flag
  shares within a factor of #st-fig.search_gap of each other.])) <fig:study-search>

#for t in st.tables.filter(t => t.key.ends-with("-search")) [#st-table(t) #label("tab:study-" + t.key)]

#pagebreak(weak: true)

= Sanity check: the overhauled code against the manuscript <sec:manuscript-tables>

#pitfall(title: [Internal])[To be removed once verified.]
#let ms-cell(r, c) = if r.at("strong", default: false) { strong(c) } else { c }
// A re-run cell beyond the record's noise band from the manuscript's: green where it is better,
// red where it is worse; the list under the table names the same cells.
#let ms-better = rgb("#1b7a35")
#let ms-worse = rgb("#b42318")
#let ms-fill = (better: rgb("#e3f1e5"), worse: rgb("#fbe4e2"))
#let ms-cells(r) = r.cells.enumerate().map(((k, c)) => {
  let mark = r.at("marks", default: ()).at(k, default: none)
  if mark == none { ms-cell(r, c) } else { table.cell(fill: ms-fill.at(mark), ms-cell(r, c)) }
})
// The re-run side names the records its values come from (a supplementary record can replace the
// main one's values for a few groups).
#let ms-side-head(t) = (
  [manuscript],
  if t.sources.len() == 0 [re-run (none recorded)] else [re-run (#t.sources.map(i => raw(ms.records.at(i).commit)).join(", "))],
)
// One table per manuscript table: its columns once for the manuscript and once for the re-run,
// side by side in one grid, so that a row's two versions stay on one line and the table breaks
// across pages as a whole; both sides share the column widths. A table too wide for that is
// shown twice, the re-run below.
#let ms-table(t) = {
  let n = t.header.len()
  let head = t.header.map(h => eval(h, mode: "markup"))
  let groups = t.at("groups", default: ())
  let line(r) = if "group" in r {
    (table.cell(colspan: n, align: left, eval(r.group, mode: "markup")),)
  } else {
    ms-cells(r)
  }
  figure(kind: table, caption: brief([Manuscript #t.title beside the re-run],
    [The manuscript's #t.title (#raw(t.label)) beside the re-run. #eval(t.caption, mode: "markup")]), {
  if "note" in t {
    block(sticky: true, implementation(title: [How the re-run differs], text(9pt, eval(t.note, mode: "markup"))))
  }
  layout(size => {
    // Widths at 8pt; a table slightly too wide for both sides gets a smaller type, down to 6.8pt.
    let rows = (t.paper + t.rerun).filter(r => "cells" in r)
    let natural = range(n).map(i => calc.max(
      measure(text(8pt, head.at(i))).width,
      ..rows.map(r => measure(text(8pt, ms-cell(r, r.cells.at(i)))).width),
    ))
    let pad = 6pt
    let gap = 14pt
    let scale = calc.min(1.0, (size.width - gap - 2 * n * pad) / (2 * natural.sum()))
    let side-by-side = scale >= 0.85
    let s = if side-by-side { scale } else { 1.0 }
    let widths = natural.map(w => w * s + pad)
    let grid-of(sides) = {
      let k = sides.len()
      let cols = if k == 2 { widths + (gap,) + widths } else { widths }
      let joined(parts) = if k == 2 { parts.at(0) + ([],) + parts.at(1) } else { parts.at(0) }
      set text(8pt * s)
      table(
        columns: cols,
        inset: (x: 3pt, y: 2.6pt),
        align: center + horizon,
        stroke: (x, y) => if k == 2 and x == n { none } else if y <= 1 + int(groups.len() > 0) {
          (bottom: 0.6pt)
        } else { (bottom: 0.3pt + luma(205)) },
        table.header(
          ..joined(sides.map(side => (table.cell(colspan: n, text(9pt, ms-side-head(t).at(if side == "paper" { 0 } else { 1 }))),))),
          ..if groups.len() > 0 { joined(sides.map(_ => groups.map(((g, span)) => table.cell(colspan: span, g)))) },
          ..joined(sides.map(_ => head)),
        ),
        ..range(t.paper.len()).map(i => joined(sides.map(side => line(t.at(side).at(i))))).flatten(),
      )
    }
    if side-by-side { grid-of(("paper", "rerun")) } else {
      grid-of(("paper",))
      v(0.4em)
      grid-of(("rerun",))
    }
  })
  })
  if t.sources.len() > 0 {
    set text(8.5pt)
    block(above: 0.4em, below: 1.2em, if t.differences.len() == 0 [
      - No major differences: every re-run number is within the noise band of the manuscript's.
    ] else {
      list(..t.differences.map(d => [
        #text(fill: if d.mark == "better" { ms-better } else { ms-worse })[
          #if d.mark == "better" [*Better:*] else [*Worse:*] #eval(d.text, mode: "markup")]
        #block(above: 0.25em, below: 0.35em, text(fill: luma(70))[_Cause:_ #eval(d.at("cause", default: "Not analysed."), mode: "markup")])
      ]))
    })
  } else { v(0.8em) }
}

The code behind the manuscript has since been rewritten: one refinement engine for both plans,
new GPU kernels with deterministic sums, a float64 ODE integrator in place of the JAX float32 flow
map, new trust-region regularization and new data structures throughout. This appendix checks that
the rewritten code still computes what the manuscript reports. It is a sanity check rather than a
reproduction to the digit, because two implementations that draw random numbers differently and
round differently cannot agree exactly. A gross error, such as a wrong formula, a broken stage or a
lost leaf, still shows as a difference far beyond the run-to-run noise.

Every table of the manuscript appears twice. The left side shows the table as the manuscript prints
it, and the right side shows it as this code measures it. The re-run repeats the same experiments
with `studies/sanity/paper/run_manuscript.sh`, which `studies/sanity/README.md` describes. Where the re-run's settings had to differ from the manuscript's text to reproduce a
table, a note above the table says so. Both sides use the same format in each column, so a cell can
be compared digit by digit with the one beside it. Each re-run value comes from the seed-0 run. A
"–" marks a number that the re-run does not have, because the experiment lies outside the re-run
matrix, the case has not run or did not finish, or the case does not record that quantity.

A re-run cell is shaded where it differs from the manuscript's cell by more than the record's noise
band. The band is 25% for counts, errors and times, and for percentages it is the larger of 1 point
and 30%. The shade is #text(fill: ms-better)[green] where the re-run is better and
#text(fill: ms-worse)[red] where it is worse. Lower is better for every quantity except
$p_"in,good"$. The list under each table names the shaded cells, regressions first. It gives each
one as the ratio of the re-run's number to the manuscript's, followed by its cause as far as it has
been traced, or "Not analysed." The labels D1-D7 refer to `studies/sanity/README.md`.
#if ms.records.len() == 0 [
  No re-run has been recorded yet, so every re-run cell is "–".
] else [
  The re-run numbers come from the records #ms.records.map(r => [#raw("studies/sanity/results/" + r.name + "/")
  (commit #raw(r.commit), recorded on #r.recorded on a #r.machine, #r.gpu_runs runs on the GPU)]).join([; ]).
  A later record replaces an earlier one's values for the cases it holds. Each table's re-run side
  names the commits of the records behind it, and `studies/sanity/results/README.md` describes the
  records.
]

#for t in ms.tables [#ms-table(t) #label("tab:ms-" + t.key)]

*Verdict.* No table shows a gross error. Where the re-run differs from the manuscript beyond the
noise band, the difference comes from one of these causes:
- a setting that the re-run matches, namely Table 2's cluster sizes and Figure 8's candidate count;
- a function instance drawn differently, namely the Gaussian's rotation;
- seed-to-seed noise larger than the band, for the Gaussian at $d = 5$;
- the known difference of the runs without Hessian splits, which this code makes more accurate
  than the manuscript's (`studies/sanity/README.md`, D3).

At the two loosest tolerances, the Quadratic at $d = 10$ spends more evaluations than in the
manuscript, for the same or a smaller error.

The manuscript's values appear exactly as the manuscript prints them, typos included. The
inconsistencies found between its tables are the following:

#text(9pt)[#for n in ms.notes [- #n \ ]]


// Draft sections, for review before they move into the body; the panels and numbers of each
// come from figures/appendix/<topic>.py.
#pagebreak()
= Additions under review <sec:additions>

#pitfall(title: [Internal])[
  To be moved into the body once reviewed. Each section below describes something the code does
  that the sections above do not yet say: the AFINNI inference (@sec:apx-knn, @sec:apx-cells), the ART
  search and sampling (@sec:apx-soft, @sec:apx-hitrun), and the engine (@sec:apx-batches,
  @sec:apx-memory, @sec:apx-sums, @sec:apx-ode, @sec:apx-steal).
]

// ---- appendix/knn.typ
== An exact nearest-centre search from float32 candidates <sec:apx-knn>
#let apx-knn = json("figures/apx-knn.json")
// "1.5e-05" as 1.5 × 10^-5.
#let apx-knn-sci(v) = {
  let (m, e) = v.split("e")
  $#m times 10^#str(int(e))$
}

#question[Line 1 of @alg:inference needs exactly the $kappa$ centres nearest to the query. A
  float32 search is fast, above all on a GPU, but it rounds every distance, and measuring its
  answer again in float64 cannot bring back a centre that the search left out. How can a float32
  search return exactly the centres that a float64 scan of every centre returns?]

#scene(side: left, width: 42%, panel("apx-knn-scene"))[
  The black dot is one query of the running example, and around it lie the #apx-knn.leaves leaf
  centres, each in its own colour. Here $k = kappa = #apx-knn.k$. The float32 search proposes
  $k' = k + max(#apx-knn.slack, k) = #apx-knn.kk$ candidates, and the dashed circle passes
  through the last of them. The search then measures their distances again in float64. The solid
  circle passes through the $k$-th nearest, and the bold crosses are the $k$ nearest. The
  #apx-knn.scene.omitted faded centres beyond the dashed circle are never measured in float64.

  The #swatch(grey) grey band shows how far float32 rounding can have moved any squared distance.
  It is drawn $#apx-knn.scene.scale_factor times 10^#apx-knn.scene.scale_exp$ times too wide,
  and at true scale its width is #apx-knn-sci(apx-knn.scene.beta) in squared distance. In
  float64, a centre beyond the dashed circle lies at least as far as the band's inner edge, and
  the solid circle lies inside that edge, so no omitted centre can be nearer than the $k$-th.
  Below the map, the same leaves sit on the line of squared distances. The search checks the
  _margin_ between the $k$-th and the band, and this check is the search's _certificate_.
]

Measuring the candidates again in float64 is not enough by itself. Take the points 1 and 0, in
that order, and #apx-knn.merge.far points near $10^#apx-knn.far_exp$. The query is
#apx-knn.merge.query. Centred on the mean of the points, 0 and 1 lie where float32 numbers are
#apx-knn.merge.spacing apart, so both round to the same float32 number and the float32 search
ranks them equal. Ties go to the smaller index, and point 1 comes first, so the search returns
#apx-knn.merge.float32_top.at(0) as the nearest. Measuring that one candidate again in float64
changes nothing, because the nearest point, #apx-knn.merge.exact.at(0), was never proposed. With
$k' = #apx-knn.merge.kk$ candidates, the search proposes both points, the float64 distances put
#apx-knn.merge.exact.at(0) first, and the certificate holds. The extra candidates leave room for
the float32 order to be wrong, and the certificate proves that the float32 order was wrong by no
more than that room.

#translate(caption: brief([The certified nearest-centre search in symbols], [The quantities of the exact nearest-centre search from float32 candidates, in words and in symbols.]),
  [the squared distance from the query to point $p$, computed in float32], [$tilde(d)^2 (p)$],
  [the same squared distance in float64, from direct differences], [$d^2 (p)$],
  [a bound on their difference, one per query], [$beta$ (`_error_bound`)],
  [the $k$-th smallest $d^2$ among the candidates], [$d^2_((k))$],
  [the largest $tilde(d)^2$ among the candidates, which is the $k'$-th], [$tilde(d)^2_((k'))$],
  [the certificate], [$tilde(d)^2_((k')) - beta > d^2_((k))$],
)

#lemma(title: [the certificate])[
  Assume three things: $abs(tilde(d)^2 (p) - d^2 (p)) <= beta$ for every point $p$; the $k'$
  candidates are the $k'$ points of smallest $tilde(d)^2$; and
  $tilde(d)^2_((k')) - beta > d^2_((k))$. Then the $k$ nearest points in float64 are all among
  the candidates, and ranking the candidates by $d^2$ returns them.
] <lem:apx-certificate>

#proof[
  A point $p$ that is not a candidate has $tilde(d)^2 (p) >= tilde(d)^2_((k'))$. So
  $d^2 (p) >= tilde(d)^2 (p) - beta >= tilde(d)^2_((k')) - beta > d^2_((k))$, and $k$ candidates
  are strictly nearer than $p$. The inequality must be strict, because otherwise an omitted point
  at exactly $d^2_((k))$ with a smaller index would have to be returned instead of the $k$-th.
]

On the #apx-knn.grid queries of the grid of @fig:error, #apx-knn.fallback_rows rows fail the
certificate. The row with the smallest margin clears it by #apx-knn-sci(apx-knn.min_margin)
times its $beta$, and the float32 error is never more than #apx-knn.max_ratio_percent% of $beta$.
The search does not approximate a row that fails, but scans it again in float64, over every
centre. The certificate does fail on other data, such as the points
$0, 1, ..., #(apx-knn.dense.count - 1)$ and one point at $10^#apx-knn.far_exp$, with
$k = #apx-knn.dense.k$ and the query #apx-knn.dense.query. Centred on their mean, these points
lie where float32 numbers are #apx-knn.dense.spacing apart, so $beta$ is about
#apx-knn-sci(apx-knn.dense.beta), which exceeds every gap. The search therefore scans the row
(#apx-knn.dense.fallback_rows row), although the float32 list happened to be right.

#implementation(title: [In the code])[
  - `_verified` (`src/neighbors/knn.py`) runs the float32 search `_search32`, re-scores the
    candidates and applies @lem:apx-certificate row by row. `_search32` uses `knn_tiled.cu` on
    the GPU and Numba or BLAS tiles on the CPU. Let $hat(q)$ and $hat(x)_p$ be the query and the
    centres, centred on the centres' mean $macron(x)$ and rounded to float32. Let
    $u_32 = 2^(-24)$ and $u_64 = 2^(-53)$. Then
    $ beta = 4 (d + 8) u_32 (norm(hat(q))^2 + max_p norm(hat(x)_p)^2) + 16 u_64 (norm(q - macron(x))^2 + 2 max_p norm(hat(x)_p)^2) . $
    The docstring of `_error_bound` lists what the bound covers: rounding the coordinates, the
    expanded form $norm(hat(q))^2 - 2 hat(q)^top hat(x) + norm(hat(x))^2$, the float64 centring,
    and the square root and square. The bound also includes a safety factor of two. Because it
    grows only with the centred norms, translating the centres and the queries by the same vector
    does not loosen it.
  - Two kinds of row go to `_fallback`, a float64 scan of every centre: a row that fails the
    certificate, and a row whose float32 result is invalid, because an index is out of range or
    repeated or a distance is not finite. For a GPU index the scan runs on the device. If
    $k' >= N_c$, the search runs the scan at once.
  - Every float64 squared distance is one strict sequential sum, in which each subtraction,
    product and addition is rounded on its own. On the GPU, `candidate_d2` (`knn_select.cu`)
    keeps this rounding with `__dsub_rn`, `__dmul_rn` and `__dadd_rn`, intrinsics that forbid
    fused multiply-adds.
  - On every path, a tie in float64 distance goes to the smaller leaf index. The float32 kernels
    insert only a strictly smaller value, `_verified` orders the candidates by index before a
    stable sort by distance, and the scans keep the first of equal values. So the candidates
    $cal(C)_x$ depend on $x$ and the leaves alone, and so does the answering leaf
    (@sec:apx-cells), which makes them the same on either backend, in any chunk and on any rank.
    The trust-region distances that the GPU computes from them can still change in their last
    bits with the chunk size, which `KNN_GATHER_BYTES` in `src/surrogate/model.py` bounds.
  - On the CPU with $d <= #apx-knn.tree_max_d$ (`CPU_TREE_MAX_D`), the search has no float32
    stage. A SciPy cKDTree on the float64 centres returns $k + 1$ neighbours, which the search
    re-scores with the strict sequential float64 sum and orders by $(d^2, "index")$. A row goes
    to the scan when its $(k + 1)$-th distance lies within a relative $10^(-12)$ of its $k$-th. `predict` takes this
    path on the running example and scans #apx-knn.tree_fallback_rows rows of the grid, while the
    panel calls the float32 path directly on the same index. On every grid query, both paths
    return the neighbours that a scan of every leaf returns.
]

#pitfall(title: [What exactness costs, and where the code does not pay it])[
  The certificate doubles the candidate lists, with $k' = 2k$ once $k >= #apx-knn.slack$, and a
  row that the certificate rejects costs a scan of all $N_c$ centres, $O(N_c d)$. Data whose gaps
  are small beside their spread, like the #apx-knn.dense.count points above, fail the certificate
  everywhere, and the search becomes a float64 brute-force scan, which is exact but no longer
  fast. The initial $k$-means (`src/neighbors/kmeans.py`) does not pay for exactness. On the GPU
  it assigns points with the float32 candidates as found (`kneighbors_unverified`), and on the
  CPU it assigns them in float64 without a certificate. Its sums run in a fixed order, so a run
  repeats bit for bit, but a point within rounding of two centres may go to either centre. That
  choice fixes the initial clusters, and so the forest.
  $k$-means tolerates the choice, because any partition is a valid start for refinement.
]

#takeaway[A float32 search proposes more neighbours than needed, and float64 ranks them. A bound
  on the float32 rounding proves that no point left out could be nearer, and a row without that
  proof is scanned in float64. So the search is exact, and its ties always go to the smaller index.]

// ---- appendix/cells.typ
== What the inference rule really computes: answer cells <sec:apx-cells>
#let apx-cells = json("figures/apx-cells.json")
#let apx-share(k) = apx-cells.share.at(str(k))

#question[@alg:inference answers every query with one leaf, $nu(x)$. Where in the domain does
  each leaf answer, and how much of that does the number $kappa$ of Euclidean candidates decide,
  rather than the leaves?]

#figure(board(columns: 3,
  (panel("apx-cells-all"), [every leaf a candidate, $kappa = N_c = #apx-cells.leaves$]),
  (panel("apx-cells-one"), [$kappa = #apx-cells.k_low$: the nearest centre]),
  (panel("apx-cells-exact"), [$kappa = #apx-cells.k_low$, `exact_outside=True`]),
), kind: image, caption: brief([The answer cells of the inference rule], [Each query of the running example's grid has the colour of the leaf
  that answers it. The panels also show the leaves' trust ellipses, and white lines separate the
  cells. Cross-hatching marks the queries whose answering leaf differs from the left
  panel's.])) <fig:apx-cells>

Colour every query by the leaf that answers it. In the left panel every leaf is a candidate, so
leaf $i$ answers where its trust-region distance $d_i$ is the smallest. That region is the
leaf's _answer cell_, and its edges are curves on which two leaves' distances are equal. A long
leaf's cell reaches far along the leaf's axis. A cell need not be in one piece, and here
#apx-cells.split_cells cells are split, into up to #apx-cells.pieces_max pieces within the
domain. In the middle panel there is one candidate, so the answer is the nearest centre and the
cells are the ordinary Euclidean Voronoi diagram, with straight edges. There a different leaf
than in the left panel answers #apx-share(1)% of the grid. More candidates bring the cells closer
to the left panel: the share is #apx-share(2)% for $kappa = 2$, #apx-share(3)% for $kappa = 3$
and #apx-share(5)% for $kappa = 5$. From $kappa = #apx-cells.first_zero$ on, every query of this
grid has the same answer as in the left panel, and this range includes the companion's
$kappa = #apx-cells.kappa$.

The cells are not the clusters. The leaves were fitted on #apx-cells.points points, and a leaf
other than their own answers #apx-cells.foreign_points of them (#apx-cells.foreign_percent%).
A leaf's point can lie deeper inside a neighbour's ellipse than inside its own. With one
neighbour, $fh$ equals one leaf's affine model on each cell, so the cells are the surrogate's
pieces, and their edges are where the surrogate jumps.

#translate(caption: brief([The inference rule in symbols], [The inference rule in words and in symbols: the trust-region distance, the candidates and the leaf that answers.]),
  [the trust-region distance of leaf $i$], [$d_i (x) = (x - c_i)^top T_i (x - c_i)$],
  [the $kappa$ centres nearest to $x$, with ties going to the smaller index (@sec:apx-knn)], [$cal(C)_kappa (x)$],
  [the leaf that answers], [$nu_kappa (x) = op("arg min")_(i in cal(C)_kappa (x)) d_i (x)$],
  [the answer cell of leaf $i$], [$A_i^kappa = {x : nu_kappa (x) = i}$],
  [the smallest eigenvalue of any trust matrix], [$lambda_* = min_i lambda_min (T_i)$],
  [the $kappa$-th smallest Euclidean distance from $x$ to a centre], [$rho_kappa (x)$],
)

#proposition(title: [the answer cells])[
  (1) Let $kappa = N_c$. Then the cells form the leaves' Voronoi diagram, each leaf in its own
  metric, and the edge between the cells of $i$ and $j$ lies on the quadric
  $(x - c_i)^top T_i (x - c_i) = (x - c_j)^top T_j (x - c_j)$, which is a hyperplane when
  $T_i = T_j$. If $kappa = 1$, the cells are the Euclidean Voronoi cells of the centres. \
  (2) If $nu_(N_c) (x) in cal(C)_kappa (x)$, then $nu_kappa (x) = nu_(N_c) (x)$. In particular,
  this holds wherever $d_(nu_kappa (x)) (x) < lambda_* thin rho_kappa (x)^2$. \
  (3) If $k_"nn" = 1$, then $fh(x) = w_i^top (x - c_i) + b_i$ on $A_i^kappa$, so the surrogate
  is affine on every cell and can jump only across the cells' edges.
] <prop:apx-cells>

#proof[
  (1) On the edge, $d_i = d_j$, and the difference $d_i - d_j$ is a quadratic polynomial in $x$
  whose quadratic part is $x^top (T_i - T_j) x$. With one candidate, the smallest $d_i$ is taken
  over the nearest centre alone. (2) If the overall winner is a candidate, the smallest $d_i$
  among the candidates is the overall smallest. A leaf $j$ outside the candidates has
  $norm(x - c_j) >= rho_kappa (x)$, so
  $d_j (x) >= lambda_min (T_j) norm(x - c_j)^2 >= lambda_* rho_kappa (x)^2 > d_nu (x)$, and no
  leaf left out can win. (3) restates line 4 of @alg:inference.
]

On the running example, $lambda_* = #apx-cells.lambda_min$, and for $kappa = #apx-cells.kappa$
the sufficient condition of (2) holds at #apx-cells.certified% of the grid. The check would be
cheap, because $rho_kappa$ is the search's own $kappa$-th distance, but the code does not make it.

#implementation(title: [In the code])[
  - `predict_local(..., return_cluster_assignments=True)` (`src/surrogate/model.py`) returns the
    `k_nn` leaves that answer each query, and the panels are drawn from its output. The figure
    script checks that output against $nu_kappa$ restated from the definition, and the
    predictions against the leaves' affine models.
  - `exact_outside=True` changes only the rows that the candidates flag. When some leaf contains a
    flagged row, the leaves that contain it answer it, nearest first (`_with_containing_leaves`),
    so there $nu$ is the overall winner. Two kinds of row keep their candidate answer: a flagged
    row that no leaf contains, and every row that the candidates do not flag. So with this option
    the flag does not depend on $kappa$, but the leaf still does. At $kappa = #apx-cells.k_low$
    the option repairs the answer of #apx-cells.flag_fixed% of the grid (@fig:apx-cells, right),
    and a different leaf than with every leaf a candidate still answers #apx-cells.exact_share% of
    the grid. These queries lie inside a trust region, and their nearest centre answers them,
    although another leaf is nearer in its own metric.
  - With `k_nn > 1`, the search takes $kappa = $ `k_nn * coarse_factor` candidates, and the code
    blends the `k_nn` candidates with the smallest $d_i$:
    $fh(x) = sum_i omega_i fh_i (x) slash sum_i omega_i$ with
    $omega_i = 1 slash (d_i (x)^2 + 10^(-12))$. The code rejects `k_nn > coarse_factor`. The
    blend removes the jumps between the blended leaves, but not the jumps where the set of the
    best `k_nn` changes. There a model of non-zero weight is replaced by another, so $fh$ still
    jumps, and these jumps lie on the edges of the cells of order `k_nn`.
]

#pitfall(title: [`strict_trust_region=True` answers 0 outside])[
  This option drops from the blend every leaf with $d_i > 1$. If none of the best `k_nn`
  contains the query, every weight is 0. The code raises the total weight to $10^(-12)$ to avoid
  a division by zero, and the prediction is exactly 0, neither NaN nor the nearest leaf's
  answer. Only the flag tells these queries apart. On the running
  example with $kappa = #apx-cells.kappa$, the #apx-cells.strict_zero flagged queries of the grid
  (#apx-cells.strict_zero_percent%) are answered 0. In strict mode, read the flag, not the value.
]

#takeaway[With one neighbour, AFINNI's surrogate is affine on each leaf's answer cell. The answer
  cells form a Voronoi diagram in the leaves' own metrics, cut further by the $kappa$ Euclidean
  candidates, and these cells, not the clusters, are the surrogate's pieces. The exact search
  (@sec:exact-flag) makes the flag independent of $kappa$, but not the leaf.]

// ---- appendix/soft.typ
== The soft search is a monitored descent <sec:apx-soft>
#let apx-soft = json("figures/apx-soft.json")
// "3.7e-06" as 3.7 × 10^-6.
#let apx-soft-sci(v) = {
  let (m, e) = v.split("e")
  $#m times 10^#str(int(e))$
}
#let apx-soft-pow(v) = $10^#str(int(calc.round(calc.log(v))))$

#question[@lem:frozen accepts a trial plane when the plane lowers the objective with the models
  frozen. But the computed models are floating-point results, not exact minimizers, and near the
  end of a search the decreases are not much larger than rounding. When does a step count as
  descent, and when does the search give up?]

#filmstrip(
  arrows: false,
  (panel("apx-soft-run", width: 96%), [the whole search: $J_t$ above the trials each line
    search evaluated]),
  (panel("apx-soft-stop", width: 96%), [the trials of the last two iterations, against the bar
    they must clear]),
  caption: brief([The soft search as a monitored descent], [The soft search on the node of @fig:search with locational code #raw(apx-soft.code),
    which holds #apx-soft.points points. *Left.* The top plot shows the soft objective at each
    iterate, and the bottom plot shows how many trials each line search evaluated. #swatch(dark)
    Dark marks an iteration where a trial was accepted, and #swatch(red) red one where no trial
    was. The dashed line is the hard error of the bins plane, and the dotted line is the hard error
    of the final, hardened plane. *Right.* The plot shows each trial's change of the frozen
    objective, on a scale that is logarithmic away from zero. A trial is accepted only if it lies
    below both the #swatch(grey, shape: "dash") dashed Armijo line and the dotted margin. The
    #swatch(dark) dark dot is the accepted trial, and the #swatch(red) red circles are rejected
    trials.]),
) <fig:apx-soft>

The search starts from the bins plane at $J = #apx-soft.J_first$ and takes #apx-soft.steps steps
in #apx-soft.iterations iterations, down to $J = #apx-soft.J_last$. The first steps are long, and
the search took #apx-soft.first_trial_steps of its steps at the first trial. Later the first
trial overshoots. The line search then halves the step more and more often before a trial
passes, and some iterations need all #apx-soft.halvings trials. In the last iteration no trial passes.
Every trial _raises_ the frozen objective, by at least #apx-soft-sci(apx-soft.last_best), and
the search stops with `soft_line_search_stalled`. Hardening the plane then takes the node's hard
error from #apx-soft.hard_bins to #apx-soft.hard_final, and that decrease is why the result is
kept. Every step had to pass a test that does not trust the computed models, and such a search
is a _monitored descent_.

#translate(caption: brief([The monitored soft search in symbols], [The quantities of one step of the soft search, in words and in symbols.]),
  [the models computed at the plane $theta_t = (a, z)$, which are not the exact minimizers $B^*$, and the objective that the code evaluates there], [$tilde(B)_t$ (`beta`), #h(0.3em) $J_t = F(theta_t, tilde(B)_t) >= J_h (theta_t)$ (`J`)],
  [each child's weighted Gram with the ridge, and its residual at the computed models], [$M_k = Phi^top W_k Phi + lambda I$, #h(0.3em) $R_k = Phi^top W_k (Y - Phi tilde(B)_k) - lambda tilde(B)_k$],
  [the _gap_, which measures how far the computed models are from the best ones. The _margin_ that a decrease must beat is the gap plus an allowance for rounding], [$gamma_t = 1/n sum_(k = plus.minus 1) tr(R_k^top M_k^(-1) R_k)$ (`gap`), \ $mu_t = gamma_t + #apx-soft.rounding epsilon abs(J_t)$ (`margin`)],
  [the gradient of @lem:gradient, tangent to $norm(a) = 1$. Trial $m$ steps $alpha_m$ along it and renormalizes $a$. The first trial moves the gate by about one width $h$], [$g_t$, #h(0.3em) $alpha_m = alpha_0 2^(-m)$, $m < #apx-soft.halvings$, #h(0.3em) $alpha_0 = h slash norm(g_t)$],
  [accept the first trial that passes both tests], [$F(theta_m, tilde(B)_t) <= J_t - #apx-soft-pow(apx-soft.armijo) alpha_m norm(g_t)^2$, \ $J_t - F(theta_m, tilde(B)_t) > mu_t$],
)

#lemma(title: [frozen models with an inexact inner solve])[
  For any models $B$, $F(theta_t, B) - J_h (theta_t) = 1/n sum_k tr((B_k - B_k^*)^top M_k (B_k - B_k^*))$.
  At the computed models this excess is exactly the gap: $J_t - J_h (theta_t) = gamma_t$.
  Hence, if a trial plane $theta'$ has $F(theta', tilde(B)_t) < J_t - gamma_t$, then
  $J_h (theta') < J_h (theta_t)$.
] <lem:apx-gap>

#proof[
  At a fixed plane, $n F = sum_k [tr(Y^top W_k Y) - 2 tr(B_k^top b_k) + tr(B_k^top M_k B_k)]$
  with $b_k = Phi^top W_k Y$. This is a quadratic in each $B_k$, minimized by
  $B_k^* = M_k^(-1) b_k$, and completing the square gives the first identity. Because
  $R_k = b_k - M_k tilde(B)_k = M_k (B_k^* - tilde(B)_k)$, the excess at $tilde(B)_t$ is
  $1/n sum_k tr(R_k^top M_k^(-1) M_k M_k^(-1) R_k) = gamma_t$. Then
  $J_h (theta') <= F(theta', tilde(B)_t) < J_t - gamma_t = J_h (theta_t)$. With exact models
  $gamma_t = 0$, and the lemma reduces to @lem:frozen.
]

The figure script checks the identity on the recorded node, with the models perturbed far from
$tilde(B)_t$ so that both sides are large. The script also replays the search with the
library's own functions and checks that the replay reaches the library's objective values and
final plane to the bit.

The margin test is the lemma's premise, with an allowance added for the rounding of $J_t$ and of
the trial sums. The Armijo test is the usual sufficient-decrease condition, which asks for a
decrease in proportion to the step and the squared gradient. On the running example the gap
never exceeds #apx-soft-sci(apx-soft.gap_max), while the rounding allowance is at least
#apx-soft-sci(apx-soft.margin_min), so here the margin is the rounding term. In the right panel
of @fig:apx-soft, the Armijo line is the stricter of the two tests.

The code computes the gap from $R_k$ as written, from the residual of the data. The same matrix
equals $b_k - M_k tilde(B)_k$, but near a good fit the two terms of that difference agree in
nearly every digit, and what is left is then the rounding of $b_k$, not $R_k$. The figure script
computed the true gap in 60-digit arithmetic for the #apx-soft.fits child fits at the bins
planes of the example's #apx-soft.nodes split nodes, and measured both forms against it. These
true gaps range
from #apx-soft-sci(apx-soft.exact_gap_range.at(0)) to
#apx-soft-sci(apx-soft.exact_gap_range.at(1)). The residual form is within
#apx-soft.residual_form_error_percent% of the true gap for every fit, while the difference of
sums is off by more than a factor of two for #apx-soft.sum_form_off of them.

#implementation(title: [In the code])[
  - `ARTPlan._soft_refine` (`src/art/plan.py`) runs the loop on the CPU. On a GPU, `soft_step`
    evaluates the trials with the kernels `soft_node_prep`, `soft_trial_terms` and
    `soft_node_select`, in two groups: the first four halvings, and then the rest, for the nodes
    still waiting. On either device, a node takes the first trial that passes, so both devices
    take the same step.
  - A node stops for one of five reasons, and `profile["art"]` counts each node's reason once. In
    order of precedence, the reasons are:
    - `soft_nonfinite`: $J_t$, $gamma_t$ or $norm(g_t)$ is not finite.
    - `soft_gate_saturated`: the mean of $sigma(s_i) sigma(-s_i)$ is below
      #apx-soft-pow(apx-soft.saturation). The gates are then hard, and the gradient vanishes.
    - `soft_gradient_converged`: $h norm(g_t) <= mu_t$, so even the first trial's first-order
      decrease $alpha_0 norm(g_t)^2 = h norm(g_t)$ cannot beat the margin.
    - `soft_line_search_stalled`: no trial passed, for any other reason.
    - `soft_iteration_limit`: the search is still stepping after `soft_iters`
      $= #apx-soft.soft_iters$ iterations.
    On the example, #apx-soft.counters.soft_line_search_stalled of the #apx-soft.nodes searches
    stalled and #apx-soft.counters.soft_iteration_limit reached the limit. #apx-soft.improved of
    them were kept.
  - Each iteration evaluates the objective once. That evaluation forms two weighted Grams and
    right-hand sides, solves twice for the models, makes one pass for the residuals and $R_k$,
    and solves twice more for the gap. The solves use LU on the CPU and Cholesky on the GPU. For
    $f_d$ outputs the cost is $O(n p (p + f_d) + p^3)$ per node. Each trial costs one pass of the
    gates and one sum, $O(n d)$, and forms no Gram. The example's searches took
    #apx-soft.iterations_all iterations and #apx-soft.trials_all trials. While any node of a
    batch is active, the CPU loop evaluates every node of the batch, whereas the GPU skips the
    trial terms of the nodes that cannot use them.
]

#pitfall(title: [Stalled is not converged])[
  When the recorded node stalled, its first-order decrease $h norm(g_t)$ was
  #apx-soft-sci(apx-soft.last_step) and its margin #apx-soft-sci(apx-soft.last_margin), so the
  decrease was far above the margin, and the node was not at a stationary point. Its first trial
  moves the gate by a whole width, and after #apx-soft.halvings halvings the step was still too
  long for the frozen objective to fall. So the soft search returns a plane that is better than
  its starting plane, but not an optimum, and the hard refit decides whether the plane is kept.
]

#takeaway[The soft search accepts a step only if the frozen objective passes two tests. First, it
  must fall by more than the inexact inner solve could account for, which by @lem:apx-gap
  guarantees a true decrease. Second, it must fall by an Armijo amount. The search stops as soon
  as no halving passes, and it records why it stopped.]

// ---- appendix/hitrun.typ
== Why hit-and-run from inherited points gives uniform points <sec:apx-hitrun>
#let apx-hitrun = json("figures/apx-hitrun.json")
#let apx-hr = apx-hitrun.child

#question[@prop:hitrun reaches the uniform law only in the limit, yet @alg:hitrun stops after
  $tau$ steps at every depth. Why can a fixed number of steps be enough deep in the tree, and
  why not draw uniform points by rejection, which is exact?]

#scene(side: left, width: 50%, panel("apx-hitrun-area"))[
  Rejection would propose points uniformly in the domain and keep those with $A u <= bold(z)$.
  The number of proposals per kept point is geometric, with mean
  $abs(Omega) slash abs(Omega_C)$, and the picture plots this mean for every node. The
  #apx-hitrun.nodes nodes of the running example are drawn in their own colours. The
  #apx-hitrun.deep.nodes nodes of the same function at $atol = #apx-hitrun.deep.atol$ are drawn
  in #swatch(grey) grey, and their tree is #apx-hitrun.deep.depth levels deep. Each cut divides the area,
  so the mean grows with depth, and for cuts that halve the area it grows as $2^"depth"$, which
  the dashed line shows. The #swatch(pink, shape: "line") pink line is the cost of hit-and-run,
  $tau = #apx-hitrun.tau$ steps per point at any depth.

  In the running example, most nodes would need fewer proposals than a walk takes steps. Only
  #apx-hitrun.above_tau of the #apx-hitrun.nodes nodes need more than $tau$, and the largest mean
  is #apx-hitrun.max_ratio, at depth #apx-hitrun.max_ratio_depth. At
  $atol = #apx-hitrun.deep.atol$, #apx-hitrun.deep.above_tau_percent% of the nodes need more
  than $tau$, and at depth #apx-hitrun.deep.depth the median is #apx-hitrun.deep.median_deepest
  and the largest mean #apx-hitrun.deep.max_ratio. In that tree the walks took
  #apx-hitrun.deep.steps_per_point steps per new point, including the doublings, while in the
  running example each of the #apx-hitrun.new_points new points took exactly $tau$ steps.
]

#translate(caption: brief([Rejection and hit-and-run in symbols], [The two ways of drawing a point in a node, and the law that each step keeps, in words and in symbols.]),
  [one rejection proposal draws $u$ and tests it against the $h$ ancestor rows], [$O(d + h d)$, #h(0.3em) $abs(Omega) slash abs(Omega_C)$ times per point],
  [one walk step draws a direction, finds the chord from the box and from $A g$, and moves], [$O(d + h d)$, #h(0.3em) $tau$ times per point (`walk_steps`)],
  [the uniform law on a node's region; one move along a fixed direction $g$ to a uniform point of the chord through $u$], [$pi_C = "Unif"(Omega_C)$; #h(0.3em) $K_g (u, dot)$],
  [one step of @alg:hitrun, which first draws $g$ from a law $nu$ that does not depend on $u$], [$K = integral K_g thin nu(d g)$],
)

#lemma(title: [every step keeps the uniform law])[
  Let $Omega$ be a bounded convex region with nonempty interior, and let
  $pi = "Unif"(Omega)$. Then for every fixed direction $g$, $K_g$ is reversible with respect
  to $pi$, and hence a step whose direction is drawn independently of the current point is
  reversible with respect to $pi$ too.
  @alg:hitrun draws that direction isotropically. A sequence
  of steps along prescribed directions leaves $pi$ invariant, and the code's coordinate sweeps
  are such sequences.
] <lem:apx-reversible>

#proof[
  Write $u = y + t g$, with $y$ in the hyperplane orthogonal to $g$, so that
  $pi(d u) = d y thin d t slash abs(Omega)$. The line through $u$ along $g$ is the line through
  $y$ along $g$, and its chord is an interval $I_y$. So
  $pi(d u) K_g (u, d u') = d y thin d t thin d t' slash (abs(Omega) abs(I_y))$ for
  $t, t' in I_y$, which is symmetric in $t$ and $t'$. A mixture of kernels that are reversible
  with respect to $pi$ is reversible, and a composition of kernels that keep $pi$ keeps it.
]

Two consequences apply the lemma at every depth of the tree. First, let $X tilde pi_C$ and
fix the plane $(a, z)$. Then $X$ conditioned on $X in Omega_(Cp)$ has the density
$1 slash abs(Omega_(Cp))$ on $Omega_(Cp)$, so $X$ is uniform in the child $Cp$, and likewise
in $Cm$. Second, if a Markov kernel keeps $pi$, the endpoint's distance to $pi$ never exceeds
the start's:
$ norm(mu K^tau - pi)_"TV" <= norm(mu - pi)_"TV". $
The right-hand side is zero when $mu = pi$. Now assume that the root's points are uniform.
Then the points a child inherits are uniform in the child, a start picked among them is
uniform, and so is every endpoint, whatever $tau$. The walk starts _warm_, and its steps only
have to remove what makes the inherited points not exactly uniform.

#filmstrip(
  arrows: false,
  (panel("apx-hitrun-child", width: 92%), [node #raw(apx-hr.code), with #apx-hr.inherited inherited and
    #apx-hr.new new points]),
  (panel("apx-hitrun-walk-1", width: 92%), [#apx-hr.chains chains from the same starts after
    one step]),
  (panel("apx-hitrun-walk-" + str(apx-hitrun.tau), width: 92%), [the same chains after
    $tau = #apx-hitrun.tau$ steps]),
  caption: brief([Hit-and-run points in a depth-1 node], [The depth-1 node #raw(apx-hr.code) covers #apx-hr.area_percent% of the domain, and
    the library's own sampler drew its points. *Left.* The node's #swatch(dark) inherited points
    and its #swatch(pink) new points. *Middle and right.* The plan's sampler runs again for the
    same node with #apx-hr.chains chains, and their first #apx-hr.new endpoints after $tau$ steps
    are the node's new points, bit for bit. After one step, every endpoint lies on a chord
    through one of the #apx-hr.inherited starts, and after #apx-hitrun.tau steps the starts can
    no longer be seen.]),
) <fig:apx-hitrun>

#scene(width: 50%, panel("apx-hitrun-ks"))[
  The picture measures how far the endpoints are from uniform. The measure is the
  Kolmogorov-Smirnov distance between the endpoints' coordinates and the exact marginals of
  $"Unif"(Omega_C)$, and the plotted value is the larger of the two coordinates' distances. The solid curve starts from the node's
  #apx-hr.inherited inherited points, and its distance falls from #apx-hr.D_starts for the
  starts themselves to #apx-hr.D_first after one step and to #apx-hr.D_tau after
  #apx-hitrun.tau steps. The shaded band is the 5% level for #apx-hr.chains independent uniform
  points, #apx-hr.critical, and the first walk length of the experiment inside that band is
  $tau = #apx-hr.first_tau_within$. The dashed curve starts from #apx-hr.chains uniform starts.
  It stays between #apx-hr.D_uniform_min and #apx-hr.D_uniform_max at every $tau$, close to the
  #apx-hr.D_reference of independent uniform points, and shows no trend, which is the
  invariance of @lem:apx-reversible. The experiment turns the drift check off, so every chain
  walks exactly $tau$ steps, and it tests the marginals, not the joint law.
]

#pitfall(title: [Approximately uniform, and why])[
  The docstring of `HitAndRunSampler` calls the new points approximately uniform, not exactly
  uniform. The argument above needs three things the code does not have.
  - *Finitely many starts.* `_starts` picks each start uniformly, with replacement, among the
    node's inherited points, so a start's law is the empirical law of those points, which is
    uniform only on average over them. Given the starts, the endpoints are independent and each
    one uses its own stream, but they share the starts' bias until the steps have removed it.
    The solid curve above shows this bias.
  - *A plane chosen from the points.* The split search chooses the plane from the node's points
    (@sec:plane) instead of fixing it in advance, and the `"bins"` and `"soft"` searches put the
    offset at one of those points (@inv:predicate). So every right child inherits a point that
    lies exactly on its boundary.
  - *The root's points are the user's.* A box root draws exactly uniform points, but only the
    points it lacks. The points it holds are $X_0$, which need not be uniform. In the running
    example, $X_0$ is uniform and fills the root, so the root draws nothing.
  A node's own sample is too small to show uniformity. The #apx-hr.new new points of node
  #raw(apx-hr.code) have the distance #apx-hr.D_node_new, which is above
  #apx-hr.critical_node_new, the 5% level for that many independent points. The endpoints of
  the #apx-hr.chains chains form #apx-hr.blocks blocks of #apx-hr.new, and
  #apx-hr.blocks_above of these blocks are above that level too, whereas of #apx-hr.blocks
  blocks of independent points, #apx-hr.reference_blocks_above are above it. The sampler's
  checks, which the pitfall after @prop:hitrun describes, look for drift, unmoved chains and
  violated faces, not for uniformity.
]

#implementation(title: [In the code])[
  `HitAndRunSampler._starts` (`src/art/sampling.py`) picks a start as
  $floor(r dot "count")$, with $r$ taken from the draw's own stream. A draw's random numbers
  are laid out by its index, with room for `max_walk_steps` $= #apx-hitrun.max_walk_steps$
  steps, so a longer walk continues the same chain. That is why the experiment's chains extend
  the node's own chains.
]

#takeaway[Every hit-and-run step keeps the uniform law, and a uniform parent's points are
  uniform in its children. So the walk starts warm, and its $tau$ steps only remove the bias of
  finitely many starts chosen from the data. Their cost does not grow with depth, while the
  cost of rejection grows with every cut.]

// ---- appendix/batches.typ
== Where a batch ends: AFINNI against ART <sec:apx-batches>

#let apx-batches = json("figures/apx-batches.json")
#let bx = apx-batches.example
#let bt = apx-batches.test

#question[The engine cuts every level into batches that fit the device, and the plan decides
  where each batch ends (`batch_end`). Why do the two plans cut the same queue differently, and
  what does each rule guarantee about a batch?]

#figure(
  board(columns: 1,
    (panel("apx-batches-example"), [Level #apx-batches.level of the running example (the queue
      of @fig:layout), with $B = #apx-batches.batch_size$. AFINNI makes #bx.afinni.len()
      batches, and ART makes #bx.art.len().]),
    (panel("apx-batches-anchor"), [The queue of `tests/unit/test_plans.py`
      (#bt.counts.map(str).join(", ")), with $B = #bt.batch_size$. A cluster larger than a
      batch moves AFINNI's grid.]),
  ),
  kind: image,
  caption: brief([Where each plan ends a batch], [Each plan's own `batch_end` cuts each queue. Each block is one cluster, drawn in
    its colour with its points. The AFINNI batches and their points are drawn above the
    clusters, and the ART batches below them. A #swatch(rgb(col.blue), shape: "dash") blue
    mark is a line at which an AFINNI batch was cut, and a faint mark is a multiple of $B$ at
    which no batch was cut.]),
) <fig:apx-batches>

Both rules read the same running sum of the queue's points, yet they give different answers.
AFINNI draws a line at every multiple of $B$ points, and these lines form its _grid_. It ends
each batch at the first cluster that reaches the batch's line. In the first frame, batch 0
takes the #bx.afinni.at(0) clusters up to the one that reaches #bx.lines.at(0), which is
#bx.afinni_sizes.at(0) points in all, more than $B$. ART packs instead. It takes the longest
run of whole clusters that holds at most $B$ points, so its first batch stops one cluster
earlier, at #bx.art_sizes.at(0) points. Every later boundary moves with it, and the level
takes #bx.art.len() batches instead of #bx.afinni.len(). In the second frame, the cluster of
#bt.counts.at(2) points takes a batch of its own under both rules. Under AFINNI's rule it also
crosses the next lines, so the line of the following batch already lies behind it. The next
cluster then forms a batch alone, and AFINNI moves the grid to start from that cluster's end.
This move is the _re-anchoring_ of the grid. The grid's offset, its anchor, returns to zero at
the first batch of every level.

#translate(caption: brief([Batch boundaries in symbols], [Where a batch ends for AFINNI and for ART, in words and in symbols.]),
  [the points of the first $k$ clusters of the level's queue],
  [$S_k = n_1 + dots + n_k$, #h(0.3em) $S_0 = 0$ (`cumsum`). For ART, $n_k$ is a node's work, defined below.],
  [batch $b$ holds clusters $e_(b-1) + 1, dots, e_b$],
  [$e_(-1) = 0$; `batch_end` returns $e_b$],
  [the batch size and AFINNI's anchor],
  [$B$ (`batch_size`), #h(0.3em) $o$ (`_grid_offset`), $o = 0$ at $b = 0$],
)

#definition(title: [the two batch rules])[
  Let the queue hold $K$ clusters. AFINNI's rule (`AFINNIPlan.batch_end` in
  `src/afinni/plan.py`) is
  $ e_b = min(K, min{k : S_k >= (b + 1) B + o}). $
  If that gives $e_b <= e_(b-1)$, the rule re-anchors the grid with
  $o <- S_(e_(b-1) + 1) - (b + 1) B$ and sets $e_b = e_(b-1) + 1$. ART's rule
  (`ARTPlan.batch_end` in `src/art/plan.py`) is
  $ e_b = min(K, max(e_(b-1) + 1, max{k : S_k <= S_(e_(b-1)) + B})). $
] <def:apx-batch-rules>

#lemma(title: [what a batch holds])[
  Under both rules, the batches cover the queue once and in order, and no batch is empty. Take
  a batch of more than one cluster. Under AFINNI's rule it holds fewer than $B + n_(e_b)$
  points, so it exceeds $B$ by less than its last cluster, and under ART's rule it holds at most
  $B$ points. A batch of one cluster may hold any number of points. Each AFINNI boundary is
  fixed by its own line, $(b + 1) B + o$, whatever the earlier batches did, except a boundary
  where the grid was re-anchored. Each ART boundary depends on the boundary before it.
] <lem:apx-batches>

#proof[
  Both rules give $e_(b-1) < e_b <= K$, so the batches follow each other, and the loop ends at
  $K$. *AFINNI's rule.* The line of batch $b$ lies at most $B$ points past $S_(e_(b-1))$,
  because batch $b - 1$ ended at the first cluster that reached the previous line, which lies
  $B$ points lower. After a re-anchoring, batch $b - 1$ ended exactly at that line. The
  clusters before the last one have not reached the line, so they hold fewer than $B$ points,
  and the batch therefore holds fewer than $B + n_(e_b)$ points. When a cluster reaches several
  lines at once, the next line lies behind it, and the cluster after it then forms a batch
  alone. The new anchor puts the next line $B$ points past that cluster. *ART's rule.*
  $S_(e_b) <= S_(e_(b-1)) + B$, unless the outer maximum is $e_(b-1) + 1$, a batch of one
  cluster. The last claim follows from the formulas. AFINNI's $e_b$ involves $e_(b-1)$ only
  through the re-anchoring test, whereas ART's $e_b$ involves it through $S_(e_(b-1))$.
]

The script that draws @fig:apx-batches restates the two formulas and checks them against the
library's `batch_end` on these queues and on #apx-batches.random_checked random ones. It also
checks the bounds of @lem:apx-batches on every batch. Under AFINNI's rule, a large cluster does
not always form a batch alone, because a batch that has not yet reached its line takes the
cluster in. For example, the queue #apx-batches.joined.counts.map(str).join(", ") with
$B = #bt.batch_size$ is one batch of #apx-batches.joined.afinni_sizes.at(0) points.

Why two rules? AFINNI draws the new points of all rejected clusters of a batch from one
generator, seeded by `SeedSequence([seed, level, batch, rank])`, and one `sampler.sample` call
per batch consumes it in queue order. So where a batch ends decides which stream a cluster's new
points come from, and that stream decides the cluster's children and its subtree. The batch
boundaries are therefore part of what fixes AFINNI's forests (@prop:determinism), and the
library keeps the grid rule because changing it would change the forests. ART draws a node's
points from a stream keyed by the node's root and locational code (@inv:rng), and it sums only
over the node's own rows (@inv:sums). So the ART tree does not depend on where a batch ends,
and ART can use the rule that keeps a batch within $B$.

#implementation(title: [The batch size the rules receive])[
  - $B$ is the user's `batch_size`, lowered once per run to what the device holds
    (`_cap_batch_size` in `src/engine/driver.py`). The cap is $floor(0.65 F slash beta)$, where
    $beta$ is the plan's `bytes_per_point()`. On a GPU, $F$ is the free device memory, read
    after the memory pool has released its cached blocks. On a CPU, the engine takes $F$ as
    32 GiB and leaves out the factor 0.65. The cap is clipped to lie between `plan.batch_floor`
    and #apx-batches.text.cap_max, and the floor is #apx-batches.text.floor_afinni for AFINNI
    and #apx-batches.text.floor_art for ART. For the running
    example's AFINNI plan, $beta = #apx-batches.afinni_bytes_per_point$ bytes, and the default `batch_size` of
    #apx-batches.text.default stays #apx-batches.text.cpu.
  - The floor bounds only the device's cap. A smaller `batch_size`, such as the
    #ex.steal.batch_size points of the runs of @sec:mpi, stays as it is.
  - AFINNI's running sum counts the points a cluster holds. ART's running sum counts work, which
    is what a node will occupy on the device. A node's work (`ARTPlan.work_sizes`) is its number
    of points after its top-up to $n$, plus its validation points.
  - A steal in the middle of a level only shortens the queue from its tail. So the boundaries
    of the batches that remain do not move.
]

#pitfall(title: [Over the budget, and a mode of ART that depends on boundaries])[
  $B$ comes from the memory budget, so an AFINNI batch can exceed the budget by up to one
  cluster, and the engine warns only when a single cluster is larger than $B$. With
  `sampling="convex_combination"`, ART also augments the nodes it splits from one generator per
  batch, and the generator is keyed by the batch's first such node. So in that mode the ART tree
  depends on the batch boundaries too.
]

#takeaway[AFINNI cuts a level where its running sum crosses fixed lines, because the cuts
  decide its random streams and so its forests. The ART tree does not depend on the cuts, so
  ART packs the longest run of whole nodes that fits the batch size.]

// ---- appendix/memory.typ
== How fast the queue grows, and the memory guard <sec:apx-memory>

#let apx-memory = json("figures/apx-memory.json")
#let mx = apx-memory.example
#let mt = apx-memory.tight

#question[A level's queue is stored in host memory, and the engine writes the next level's
  queue while it still reads the current one. How large can the next level be, and what stops a
  run before it uses up the host's memory?]

#figure(panel("apx-memory-growth"), caption: brief([AFINNI's point count against its bounds], [The point count of every AFINNI level, against
  the two bounds of @lem:apx-growth, taken from the levels' recorded statistics. Both runs use
  the running example's function and sample, and $Nmin = #mx.n_min$. Left: the running
  example's tolerance. Right: a tolerance #(int(1 / apx-memory.tight_factor)) times tighter.
  The next level always lies in the grey band, and from level #mt.tight_from on the right-hand
  run lies on the band's upper edge.])) <fig:apx-memory>

In the next level, each rejected cluster is replaced by its two children, which keep all of the
cluster's points. A cluster with fewer than $2 Nmin$ points first receives new points
(@sec:augment), and its children share those too. The next level therefore holds the
augmentation target of every rejected cluster. A rejected cluster of $n$ points has the target
$n$ if $n >= 2 Nmin$, and $max(2 n, 2 Nmin)$ otherwise. In the first levels of the running
example the clusters are large, and their children only share their parents' points. Once
every rejected cluster has between $Nmin$ and $2 Nmin$ points, each one is doubled. Each child
then has as many points as its parent had, so a rejected child is doubled in turn, and the next
level holds exactly twice the rejected points. This happens on #mt.tight_levels of the
#mt.levels.len() levels of the tighter run. While nearly every cluster is rejected, the queue
doubles from level to level; it peaks at #apx-memory.text.peak_level points and shrinks as
clusters are accepted. This is the _growth law_ of the queue.

#translate(caption: brief([Queue growth in symbols], [The quantities of the bound on the queue's growth, in words and in symbols.]),
  [the points and clusters of level $ell$; the rejected clusters and their points],
  [$P_ell$, $K_ell$; #h(0.3em) $R_ell$, $P_ell^"rej" = sum_(C "rejected") n_C$],
  [the number of points a rejected cluster of $n$ points grows to (`augmentation_targets`)],
  [$t(n) = n$ if $n >= 2 Nmin$, #h(0.3em) $t(n) = max(2 n, 2 Nmin)$ otherwise],
)

#lemma(title: [growth law])[
  Assume that no steal changes the queue of level $ell$, and that every child is within the
  depth limit. Then $P_(ell + 1) = sum_(C "rejected") t(n_C)$, and hence
  $ 2 Nmin R_ell <= P_(ell + 1), quad P_ell^"rej" <= P_(ell + 1). $
  If every rejected cluster has at least $Nmin$ points, then also
  $ P_(ell + 1) <= 2 P_ell^"rej" <= 2 P_ell. $
  The first of these inequalities is an equality when all rejected clusters have fewer than
  $2 Nmin$ points. With the median cut, every cluster of a level $ell >= 1$ has at least
  $Nmin$ points, so the upper bound holds at every level but the first, and at the first level
  too when each initial cluster has at least $Nmin$ points.
] <lem:apx-growth>

#proof[
  Accepted clusters leave the queue. The plan grows a rejected cluster to $t(n)$ points and
  cuts it into two children whose counts add up to $t(n)$, which `_check_split` enforces. Both
  children are appended to the next level, and an empty child adds no points. For every $n$,
  $t(n) >= 2 Nmin$ and $t(n) >= n$. Also $t(n) <= 2 n$ exactly when $n >= Nmin$, and
  $t(n) = 2 n$ for $Nmin <= n < 2 Nmin$. The median cut gives the children
  $floor(t slash 2)$ and $ceil(t slash 2)$ points, and both counts are at least $Nmin$ because
  $t >= 2 Nmin$.
]

The script that draws @fig:apx-memory wraps `augmentation_targets`, so it records the points
of every rejected cluster as the plan calls it. It checks that the next level holds exactly
their targets, and it checks both bounds at every level. The initial clusters here have at
least #mx.initial_smallest points, more than $Nmin$, so the upper bound holds from level 0.

#implementation(title: [What the engine does with it])[
  - *Twice the level.* The next level's buffer starts with room for $2 P_ell$ points and
    $2 K_ell$ clusters (`_refine_level`). By @lem:apx-growth, the buffer has to grow only if
    an initial cluster has fewer than $Nmin$ points, or if a cut other than the median makes
    smaller children. Growing would copy the buffer, and in both recorded runs it never grew.
    Room that is never written costs address space, not memory. For ART, the room is twice
    the level's work (`ARTPlan.work_sizes`), where a node's work is the number of points it
    will occupy on the device. With hit-and-run sampling, a node's children hold exactly its
    work, which is its training points after the top-up plus its validation points, so the ART
    buffer does not grow either.
  - *Two levels at once.* The engine reads the current level while it writes the next, so the
    host holds $P_ell + P_(ell + 1) <= 3 P_ell$ points besides the leaves. Each point takes
    #apx-memory.queue_row_bytes bytes here, for $x$ and $f$. In the tighter run, the two
    levels hold at most #apx-memory.text.peak_two_levels points, or
    #apx-memory.peak_two_levels_kib KiB.
  - *The guard runs on every append.* The engine calls `check_host_memory` once per batch,
    before it writes to the next level's queue. The check requires that the bytes about to be
    written fit in the available memory, less a reserve of
    $max(#apx-memory.reserve_share_percent% "of RAM", #apx-memory.reserve_min_gib "GiB")$.
    If the buffer must grow, the check counts instead all the rows that its new copy will
    hold, and if they do not fit, it raises `MemoryError` with advice. It checks every append,
    including appends that need no growth, because rows written into room reserved earlier
    become resident too. The tighter run made #mt.checks checks for #mt.appends appends.
  - *The leaves' buffer.* This buffer (`ColumnBuffer`) runs the check when it grows, for the
    rows that the copy touches. Its first reservation is capped at
    #apx-memory.initial_reserve_gib GiB, because Linux's default overcommit refuses a single
    allocation larger than RAM plus swap, even one that is never touched. The default hint is
    #apx-memory.max_clusters_est rows, shared among the ranks, and this hint reaches the cap
    above #apx-memory.cap_row_bytes bytes per row. An AFINNI leaf takes
    #apx-memory.leaf_row_bytes_d2 bytes at $d = 2$, and it exceeds that row size from
    $d = #apx-memory.leaf_cap_d$.
]

#pitfall(title: [Ranks that share a node])[
  Each rank checks the free memory of its node independently, so several ranks on one node can
  each pass the check against the same free memory and together exhaust it. The check bounds
  what one rank adds. With $k$ ranks per node, plan for $k$ times the memory that
  @lem:apx-growth allows one rank.
]

#takeaway[The children of a rejected cluster hold at least $2 Nmin$ points, and if the cluster
  has at least $Nmin$ points, they hold at most twice its points, so the next level is at most
  twice the current one. The engine reserves exactly that room, and it checks host memory on
  every append because written rows use memory and reserved rows do not.]

// ---- appendix/sums.typ
== Why the chunked sums keep their bits <sec:apx-sums>
#let apx-sums = json("figures/apx-sums.json")
// "1.8e-11" as 1.8 x 10^-11 (the JSON keeps the script's rounding as text).
#let apx-sci(s) = {
  let s = str(s)
  if not s.contains("e") { return s }
  let (m, e) = s.split("e")
  let e = int(e)
  let p = if e < 0 { $10^(-#str(-e))$ } else { $10^#str(e)$ }
  if m == "1" { p } else { $#m times #p$ }
}

#question[@inv:sums promises that a cluster's sums depend only on its own rows. Why does the
  two-pass chunked sum of @sec:same-bits keep that promise, and why do the obvious GPU
  alternatives, which are atomic adds, CuPy's `add.reduceat` and cuBLAS's batched products,
  break it?]

// The cluster of the panel, in its own colour, after m rows of other clusters (grey), with
// chunks of two rows as in the batch picture of the batches section.
#let apx-gold = rgb(apx-sums.cluster.colour)
#let apx-strip = {
  let before = 4
  let mine = 5
  grid(
    columns: (auto, auto),
    column-gutter: 0.5em,
    row-gutter: 0.4em,
    align: (right + horizon, left + horizon),
    row-label[`F`], stack(dir: ltr, ..range(before).map(_ => cell(luma(205))), plain[⋯], ..range(mine).map(_ => cell(apx-gold))),
    row-label[rows], stack(dir: ltr, span(before + 1, [$m$ rows of other clusters]), span(mine, [$s, ..., s + n - 1$])),
    row-label[chunked], stack(dir: ltr, box(width: (before + 1) * cw),
      ..(2, 2, 1).map(c => span(c, [], stroke: (left: 0.7pt + apx-gold, right: 0.7pt + apx-gold, bottom: 0.7pt + apx-gold)))),
    row-label[prefix], stack(dir: ltr, span(before + 1 + mine, $S_(s+n)$, stroke: (bottom: 0.7pt + grey, left: 0.7pt + grey, right: 0.7pt + grey))),
    [], stack(dir: ltr, span(before + 1, $S_s$, stroke: (bottom: 0.7pt + grey, left: 0.7pt + grey, right: 0.7pt + grey))),
  )
}

#scene(side: left, width: 50%, apx-strip)[
  One cluster's rows sit somewhere in a batch, after the rows of $m$ other clusters. The
  chunked sum (gold brackets) cuts the cluster into chunks, counts them from the cluster's own
  first row $s$, and reads no other row. A _prefix-difference sum_ (grey brackets) takes the
  running sum $S$ of the whole batch and returns $S_(s+n) - S_s$. Each of the two prefixes is
  as large as everything before the cluster, so its rounding error is that large too, and the
  subtraction cannot remove this error.
]

#figure(panel("apx-sums-prefix", width: 78%), caption: brief([The rounding of a chunked sum], [The sum of $f$ over the running
  example's largest cluster of level 0, relative to its exact value. The cluster has
  #apx-sums.cluster.points points, and #swatch(apx-gold) is its colour. Its rows come after
  $m$ rows of $f$ at uniform points of the domain. The script restates the chunked sum of
  `segment_cross` in NumPy and checks it against the library. This chunked sum has the same
  bits for every $m$, and its error is the cluster's own rounding. A prefix difference loses
  digits with every row before the cluster. It does so with NumPy's row-order scan (filled),
  and even with correctly rounded prefixes (hollow). Correctly rounded prefixes are the best
  any scan can do, and their error stays under the dashed bound of @lem:apx-prefix. All sums
  are computed on the CPU.]))
  <fig:apx-sums>

At $m = 2^#apx-sums.m_max_log2$, the row-order prefix difference is off by
#apx-sci(apx-sums.scan_rel_max) of the cluster's sum and the correctly rounded one by
#apx-sci(apx-sums.rounded_rel_max), while the chunked sum is off by
#apx-sci(apx-sums.chunked_rel) at every $m$. An error $delta$ in the sum shifts the cluster's
mean by about $delta slash n$, and so shifts its affine fit and every residual by about as
much. When a cluster's largest error lies that close to $atol$, the rounding decides whether
the cluster is accepted or cut, and with it the subtree below.

#translate(caption: brief([Chunked sums in symbols], [The chunked segmented sum, in words and in symbols.]),
  [the rows of a batch, one entry of the sum per row], [$a_0, a_1, ..., a_(N-1)$],
  [segment $k$: its first row and its number of rows], [$s_k$, $n_k$ #h(0.3em) (`starts`, `counts`)],
  [the chunk length], [$c = #apx-sums.chunk$ #h(0.3em) (`SEGMENT_CHUNK`)],
  [the $j$-th chunk of segment $k$], [$"ch"_j (k) = [s_k + c j, min(s_k + c(j + 1), s_k + n_k))$],
  [a floating-point sum taken left to right, starting from 0], [$plus.o.big_(i in I) a_i$],
  [the unit roundoff of float64, and machine epsilon], [$u = 2^(-53)$, $epsilon = 2 u$],
)

#lemma(title: [a chunked sum is local])[
  For every segment $k$ and every output entry, `segment_cross` computes
  $ "out"_k = plus.o.big_(j = 0)^(ceil(n_k slash c) - 1) (plus.o.big_(i in "ch"_j (k)) a_i). $
  So $"out"_k$ is a function of the sequence $a_(s_k), ..., a_(s_k + n_k - 1)$ alone, and it
  has the same bits on every run and at any position in any batch.
] <lem:apx-chunked>

#proof[
  `segment_cross_partial` gives each (chunk, entry) pair one thread. The thread finds its
  segment $k$ by a binary search of the cumulative chunk counts; these counts depend on the
  other segments, but the thread uses them only to number the chunks. Its rows are
  $"ch"_(j')(k)$, where $j'$ is its chunk's index within segment $k$, a function of $s_k$ and
  $n_k$ only, and it adds them in row order in one register. `segment_cross_reduce` gives each
  (segment, entry) pair one thread, which adds the segment's $ceil(n_k slash c)$ partials in
  chunk order. No other thread writes either value, and floating-point evaluation in a fixed
  order is deterministic.
]

#lemma(title: [a prefix difference is not])[
  Let $S_m = sum_(i < m) a_i$, and let $hat(S)_m$ be $S_m$ rounded to the nearest double. Let
  $sigma = sum_(s <= i < s + n) a_i$ be the exact segment sum. Then the segment sum
  $hat(S)_(s+n) minus.o hat(S)_s$ differs from $sigma$ by at most
  $ epsilon sum_(i < s + n) abs(a_i) + u abs(sigma). $
  A computed scan adds its own error to each prefix, up to
  $(m - 1) u sum_(i < m) abs(a_i)$ per prefix for a row-order scan. Both bounds grow with
  every row before the segment.
] <lem:apx-prefix>

#proof[
  For both prefixes, $abs(hat(S)_m - S_m) <= u abs(S_m) <= u sum_(i < m) abs(a_i)$, and the
  subtraction rounds once more, by at most $u$ of its result. The scan's own error is the
  classical bound for recursive summation. In @fig:apx-sums the correctly rounded difference
  follows the bound, so the bound is not pessimistic.
]

#cards(
  columns: 3,
  ([`atomicAdd`], [Every thread adds its row into one total, in the order in which the
    threads finish. That order changes between runs, and so do the bits.]),
  ([CuPy's `add.reduceat`], [CuPy computes it as `cumsum`, then `acc[end] - acc[start]`
    (`_add_reduceat` in its source). This is the prefix difference of @lem:apx-prefix, whose
    error depends on the whole batch before the segment.]),
  ([cuBLAS batched products], [cuBLAS picks its algorithm from the problem shape, which
    includes the batch count, and the algorithm fixes the order of each inner sum. So one
    matrix's bits depend on how many matrices share the call, as the code records in
    `det_matmul` and `QuadraticLSQModel.fit`.]),
)

#implementation(title: [What else keeps the bits])[
  - *Offsets per row.* With offsets $a$ and $b$, the kernels sum $(A_i - a)(B_i - b)$ row by
    row instead of using $sum A_i B_i - n a b$, because for a tight cluster far from the origin
    the two terms of that form nearly cancel.
  - *The tiled kernel.* From #apx-sums.tiled_min_e output entries on, `segment_cross_tiled`
    loads each chunk into shared memory once. The untiled kernel fixes each entry's rows, their
    order and the expression $((A - a) w)(B - b)$. The tiled kernel keeps all three and only
    reads the operands from shared memory, so its bits match the untiled kernel's. From
    #apx-sums.tiled2d_min rows and columns of output on, `segment_cross_tiled2d` gives each
    thread a $4 times 4$ tile of entries, so one shared-memory read feeds four products, and it
    also keeps each entry's rows and their order. The Gram of a quadratic leaf in high
    dimension is such an output. No test compares these paths directly.
  - *No host read.* The launch covers $floor(N slash c) + K >= sum_k ceil(n_k slash c)$
    chunks, an upper bound that is known without reading the chunk counts back to the host.
    The threads past the last real chunk return at once.
  - *Small products and solves.* `det_matmul` gives each output entry one thread and a
    sequential inner sum, and `spd_solve` gives each system one thread. From
    $p = #apx-sums.coop_min_p$ unknowns on, one block factors each system, with
    `spd_factor_shared` in shared memory while the packed triangle fits and `spd_factor_coop`
    otherwise. The one-thread Cholesky fixes which subtractions each entry receives and their
    order. The block factorizations keep both, so the factor has the same bits.
    `spd_solve_factored` gives each pair of a system and a right-hand side one thread, and that
    thread does the one-thread solve's arithmetic.
  - *One relaxed order.* The one-thread solve's back substitution subtracts in increasing
    order along a serial chain. From the same number of unknowns on, the split search's solves
    run in one block per system (`spd_solve_shared`), whose back substitution subtracts in
    decreasing order, in parallel. The order is fixed for each system, so runs stay
    reproducible and a node's bits do not depend on its
    batch, but they differ from the one-thread kernel's at rounding level. Leaf fits keep the
    one-thread order (`exact=True`), so a leaf's coefficients do not depend on this. The unit
    tests compare these kernels with NumPy, and across stack sizes, only below that size.
  - *The CPU.* NumPy callers do not use `segment_cross`, whose NumPy branch is a reference.
    `segment_sum` multiplies by a sparse 0/1 matrix. This product adds a cluster's rows in
    input order, starting from zero, and so gives the bits of `add.at`. `reduce_sum_1d` is NumPy's
    `add.reduceat`. On NumPy #apx-sums.numpy, it starts from a segment's first row and adds
    the rest by pairwise summation, #if apx-sums.reduceat_sequential [in row order] else [not
    in row order], and it reads only the segment. `count_groups` and `grouped_cross` gather the
    blocks of each size into one dense array and run one BLAS product per block. The script
    checks that the CPU's `reduce_sum_1d` keeps its bits for every $m$ of @fig:apx-sums, and
    that `grouped_cross` gives a block the same bits whichever blocks share its call.
]

#pitfall(title: [What the invariant does not reach])[
  The GPU training path also uses CuPy's axis reductions, each of which sums the entries of one
  row or of one node. They are `xp.sum(..., axis=1)` over $d$ or $f_d$ (in `trust_regions`,
  `local_models`, `split_search` and `ARTPlan`), `xp.trace` of $p times p$ and
  $f_d times f_d$ matrices, and the column sums in `lstsq_auto`. By CuPy's source, the order
  of one output's additions depends on the reduction's length and layout, and not on the
  number of outputs. The generic kernel derives its stride from the length, and the CUB path
  gives each output one block sized by the length. CuPy uses the CUB path by default on CUDA
  from #apx-sums.cub_min_length contiguous entries. So these reductions are batch-independent
  as long as CuPy's autotuner (`cupyx.optimizing`) stays off, and the library never enables
  it. This conclusion comes from reading CuPy's code and was not run on a GPU here. The
  batched eigendecomposition (@sec:same-bits) and the oracle $f$ itself are outside the
  invariant. For the study's ODE oracle, @sec:apx-ode shows how the library keeps $f$ row by
  row.
]

#takeaway[A chunked sum reads only its own cluster's rows, in an order that the cluster's
  start and size fix, while every alternative lets the rest of the batch, or the order in which
  threads finish, into the result.]

// ---- appendix/ode.typ
== The batched ODE integrator <sec:apx-ode>
#let apx-ode = json("figures/apx-ode.json")
// "1.8e-11" as 1.8 x 10^-11 (the JSON keeps the script's rounding as text).
#let apx-sci(s) = {
  let s = str(s)
  if not s.contains("e") { return s }
  let (m, e) = s.split("e")
  let e = int(e)
  let p = if e < 0 { $10^(-#str(-e))$ } else { $10^#str(e)$ }
  if m == "1" { p } else { $#m times #p$ }
}

#question[This section leaves the running example for the library's test functions. The
  manuscript's Lorenz-96 study (@sec:study-config) evaluates $f(x)$ by integrating an ODE from
  $x$ to time $t_1$, and @prop:determinism (2) needs each row of an oracle call to get the
  value it would get alone. One call integrates thousands of trajectories, each with its own
  adaptive step. How can the call still give every row that value?]

#let apx-ode-colours = apx-ode.shown_colours.map(rgb)
#figure(
  filmstrip(arrows: false,
    (panel("apx-ode-steps"), [step sizes $h(t)$ of four rows of one batch]),
    (panel("apx-ode-error"), [each row's gap to RK45 on that row alone])),
  kind: image,
  caption: brief([One batch of the Lorenz-96 integrator], [One batch of #apx-ode.batch rows of the Lorenz-96 study, integrated on the CPU.
    The study has $d = #apx-ode.d$, forcing #apx-ode.forcing, $t_1 = #apx-ode.tf$ and
    $atol = rtol = #apx-sci(apx-ode.tol_text)$, and the batch is the first rows of its test
    set. *Left.* The steps of four rows
    (#apx-ode-colours.map(c => swatch(c, shape: "line")).join[ ]). Lines show the steps that
    `integrate_ensemble` takes, and hollow dots show the steps that SciPy's
    `solve_ivp(method="RK45")` takes on each row alone, placed at each step's start. The dark
    dashed line is the one step size that a single `solve_ivp` over the stacked batch shares
    among all rows. *Right.* The largest difference of every row at $t_1$ from RK45 on that row
    alone. Grey marks the stacked solve, and dark marks `integrate_ensemble`. Rows are ordered
    by their own number of steps.]),
) <fig:apx-ode>

The rows of one batch need between #apx-ode.steps_min and #apx-ode.steps_max steps, of
different lengths, and each row ends with a short step that lands on $t_1$. A single solver
over the stacked batch takes #apx-ode.stacked_steps steps for all of them. One error norm over
all $#apx-ode.batch times #apx-ode.d$ components chooses these steps, so the norm can average a
row's own error away. The stacked solve misses the per-row result by up to
#apx-sci(apx-ode.stacked_gap_max), which is more than the tolerance, and the value of the first
row moves by #apx-sci(apx-ode.row0_spread) when the batch around it has
#apx-ode.stacks.map(str).join(", ", last: " or ") rows. `integrate_ensemble` takes each row's
own steps and agrees with RK45 on that row to #apx-sci(apx-ode.library_gap_max), which is
rounding. `integrate_ensemble` is the _batched ensemble integrator_.

// Lane-to-component maps: one thread per trajectory (d = 5), and one warp per trajectory at
// d = warp-d (lanes 0..31, components l + 32 m; past d, idle).
#let warp-d = 80
#let lane-cell(body, fill: white) = box(width: 1.6em, height: 1.25em, fill: fill, stroke: 0.5pt + luma(170),
  align(center + horizon, text(6pt, body)))
#let thread-map = grid(columns: 6, column-gutter: 0pt, row-gutter: 0pt, align: center + horizon,
  text(6pt)[component #h(2pt)], ..range(5).map(j => text(6pt)[$#j$]),
  ..range(4).map(t => (text(6pt)[thread #t #h(2pt)], ..range(5).map(j => lane-cell([#t], fill: luma(235))))).flatten())
#let warp-lanes = (0, 1, 2, "⋯", 15, 16, "⋯", 31)
#let warp-map = grid(columns: 9, column-gutter: 0pt, row-gutter: 0pt, align: center + horizon,
  text(6pt)[lane #h(2pt)], ..warp-lanes.map(l => text(6pt)[#l]),
  ..range(3).map(m => (text(6pt)[$m = #m$ #h(2pt)], ..warp-lanes.map(l => if type(l) == str { lane-cell(l) } else {
    let j = l + 32 * m
    if j < warp-d { lane-cell($#j$, fill: luma(235)) } else { lane-cell([], fill: luma(250)) }
  }))).flatten())
// The butterfly of group_sum for L = 8 lanes: stage s adds the value of lane l xor off.
#let butterfly = diagram(
  spacing: (1.1em, 1.5em),
  node-stroke: 0.5pt + luma(150),
  node-shape: fletcher.shapes.rect,
  ..range(4).map(s => range(8).map(l => node((l, s), text(5.5pt, if s == 0 { $v_#l$ } else if s == 3 { $Sigma$ } else { [] }),
    width: 1.4em, height: 1.1em, name: label("bf-" + str(s) + "-" + str(l))))).flatten(),
  ..range(3).map(s => {
    let off = calc.pow(2, 2 - s)
    range(8).map(l => (edge(label("bf-" + str(s) + "-" + str(l)), label("bf-" + str(s + 1) + "-" + str(l)), stroke: 0.4pt + luma(120)),
      edge(label("bf-" + str(s) + "-" + str(l.bit-xor(off))), label("bf-" + str(s + 1) + "-" + str(l)), stroke: 0.4pt + rgb(col.gold)))).flatten()
  }).flatten(),
)

#scene(side: left, width: 50%, stack(spacing: 0.8em,
  [#small[*one thread per trajectory* ($d <= #apx-ode.thread_max_d$; here $d = #apx-ode.d$)] #v(-0.4em) #thread-map],
  [#small[*one warp per trajectory* ($#apx-ode.thread_max_d < d <= #apx-ode.warp_max_d$; here $d = #warp-d$)] #v(-0.4em) #warp-map],
))[
  The kernel `ensemble_rk` gives each trajectory a group of $L$ consecutive threads. Up to
  $d = #apx-ode.thread_max_d$, the group has $L = 1$ thread. The top grid shows this case, and
  each cell gives the thread that owns that component. Above $d = #apx-ode.thread_max_d$, the
  group has $L = #apx-ode.warp$ threads, which is one warp. The bottom grid shows this case,
  and each cell gives the component that lane $l$ holds in its $m$-th slot. Lane $l$ owns
  components
  $l, l + L, l + 2L, ...$, and pale cells are idle. A component's derivative reads its
  neighbours, so stage inputs go through $d$ doubles of the group's shared memory.
]

#translate(caption: brief([The batched integrator in symbols], [The batched Dormand-Prince integrator, in words and in symbols.]),
  [row $i$ of the batch, its state, its end time], [$y^((i)) in RR^d$, $t_1$ #h(0.3em) (`X0[i]`, `t1`)],
  [the #apx-ode.stages stages of Dormand and Prince's 5(4) pair, and the error weights], [$k_0, ..., k_(#(apx-ode.stages - 1))$; $e = b - hat(b)$ #h(0.3em) (`DOPRI5`)],
  [the scaled root-mean-square norm], [$norm(v) = sqrt(1/d sum_j (v_j slash (atol + rtol abs(y_j)))^2)$],
  [the error of a step of size $h$], [$"err" = norm(h sum_q e_q k_q)$, with $abs(y_j)$ the larger of old and new],
)

#definition(title: [RK45's step control, per trajectory])[
  + *Initial step* (Hairer). Set $d_0 = norm(y)$, $d_1 = norm(k_0)$ and
    $h_0 = 0.01 d_0 slash d_1$. If either $d_0$ or $d_1$ is below $10^(-5)$, set
    $h_0 = 10^(-6)$ instead. Cap $h_0$ at $t_1$. Then set
    $d_2 = norm(f(y + h_0 k_0) - k_0) slash h_0$ and
    $h = min(100 h_0, (0.01 slash max(d_1, d_2))^(1 slash q), t_1)$, with $q = #apx-ode.order$.
  + *Accept* the step when $"err" < 1$, and multiply $h$ by $min(10, 0.9 "err"^(-1 slash q))$,
    or by 10 if $"err" = 0$. Right after a rejection, multiply it by at most 1.
  + *Reject* the step otherwise, and multiply $h$ by $max(0.2, 0.9 "err"^(-1 slash q))$.
  + *Minimum step.* Raise $h$ to ten units in the last place of $t$, and fail if $h$ falls
    below that again after a rejection.
  + *Last step.* When $t + h >= t_1$, take $h = t_1 - t$ and set $t = t_1$ exactly.
  + *Failure.* The row comes back NaN after a NaN error, a step that is too small, or more
    than $10^6$ steps.
] <def:apx-rk45>

#scene(side: right, width: 52%, butterfly)[
  The step control needs sums over all $d$ components, for the scaled norms of the initial
  step and for the error of every step. Each lane adds its own components, and then
  `group_sum` combines the lanes by a fixed butterfly. The kernel uses #apx-ode.warp lanes in
  #int(calc.round(calc.log(apx-ode.warp, base: 2))) stages, and the picture shows eight lanes
  in three stages. At each stage, every lane adds the value of the lane whose index differs
  from its own in one bit (gold), so after $log_2 L$ stages every lane holds the sum.
]

#lemma(title: [every lane holds the same bits])[
  After the stages $"off" = L slash 2, ..., 1$ of $v <- v + v_(l xor "off")$, all $L$ lanes
  hold the same double.
] <lem:apx-butterfly>

#proof[
  Before the stage with offset $o$, lane $l$ holds a value that depends only on $l mod 2o$,
  which is true at the start, where $2o = L$. Lanes $l$ and $l xor o$ then add the same two
  values in opposite orders, and IEEE addition is commutative, so both lanes get the same bits.
  Their new value depends only on $l mod o$, which at $o = 1$ leaves one value.
]

#proposition(title: [a row does not depend on its batch])[
  For every row $i$, `integrate_ensemble(X0)[i]` is a function of `X0[i]` alone, bit for bit
  on one backend and one code path. Up to rounding, the value is RK45 with the control of
  @def:apx-rk45 applied to `X0[i]` alone.
] <prop:apx-ode>

#proof[
  A group reads only its row, and it writes only its row and its own $d$ doubles of shared
  memory; the tableau and the forcing are read-only. Every decision of @def:apx-rk45 is taken
  from `group_sum` values, which are the same in every lane (@lem:apx-butterfly), so a group
  follows one control path, its trajectory's own. The value of $d$ chooses the code path,
  thread or warp, and the batch never does. On the CPU, a Numba version of the kernel gives
  each row its own iteration of a parallel loop. Each rule of @def:apx-rk45 is SciPy's
  (`select_initial_step` and `RungeKutta._step_impl`), so the code differs from SciPy only in
  the order of the stage sums and of the norm's sum.
]

The script checks this on the batch of @fig:apx-ode. A Python restatement of the CPU version
ends on the library's bits for every row. Every row takes as many steps as SciPy, its step
times agree with SciPy's to within #apx-sci(apx-ode.t_gap_max) of $t_1$, and it is
bit-identical in batches of #apx-ode.subsets.map(str).join(", ", last: " or ") rows.
`tests/unit/test_ode.py` checks these properties against SciPy. It covers both sides of the
thread-to-warp switch and several regimes of state size, forcing, horizon and tolerance. The
GPU suite compares the kernel with the CPU version to rounding.

#implementation[
  - The code writes the error weights $e$ from their own fractions instead of computing
    $b - hat(b)$, so the error estimate has SciPy's bits; SciPy's weights are $-e$, and the
    norm squares them. The estimate still cancels, because the $e_q$ sum to zero, which is why
    step sizes agree with SciPy's to about #apx-sci(apx-ode.h_rel_max) relative, and not to
    the unit roundoff.
  - A lane keeps the #apx-ode.stages stages of its components in a fixed array,
    #raw("k[" + str(apx-ode.stages) + "][" + str(apx-ode.thread_max_d) + "]"), which caps a
    lane at #apx-ode.thread_max_d components and so caps $d$ at #apx-ode.warp_max_d on the
    GPU. Just above $d = #apx-ode.thread_max_d$, the last pass over a lane's components leaves
    most lanes of the warp idle.
  - The GPU's thread and warp paths sum the norm in different orders, and the GPU fuses
    multiply-adds, so the two backends agree to rounding, not bit for bit.
  - The CPU version has one cached Numba entry point per right-hand side, because Numba caches
    neither closures nor functions that receive another compiled function as an argument.
]

#pitfall(title: [Only Lorenz-96's flow map])[
  `lorenz96_flowmap` is the only oracle that uses the integrator. `vanderpol_flowmap` still
  calls one `solve_ivp` over a stacked chunk, and so do the flow maps' Jacobians,
  `lorenz96_flowmap_grad` and `vanderpol_flowmap_grad`, through `sensitivities_solver`. A
  stacked solve chooses one step size for all rows from one error norm, so their rows depend
  on the batch, as @fig:apx-ode shows. The manuscript study calls none of them, and its
  Lorenz-96 oracle has no gradient. In a chaotic regime, such as strong forcing over a long
  horizon, rounding errors grow by orders of magnitude, and the integrator then matches RK45's
  error but not its digits.
]

#takeaway[The integrator gives each trajectory one group of threads and takes every step
  decision from sums that all its lanes agree on, so each row gets SciPy's RK45 on that row
  alone, whatever batch it is in.]

// ---- appendix/steal.typ
== Why work stealing neither deadlocks nor retires early <sec:apx-steal>

#let apx-steal = json("figures/apx-steal.json")
#let sa = apx-steal.first
#let sz = apx-steal.last

#question[In @alg:steal, a thief waits for its reply, a victim waits until its clusters are
  received, and retirement waits for every rank. Why can no group of ranks end up waiting for
  each other forever, and why can no rank retire while some work is still unprocessed?]

The pictures are message-sequence charts of one recorded run of the steal example of
@sec:mpi. The run is ART at $atol = #apx-steal.run.atol$ on #apx-steal.run.ranks ranks with
one root, so rank 0 holds all the work at first, and the threshold is
$w_* = #apx-steal.run.min_load$ points. Each rank is a vertical lifeline, and time runs
downward. Every event of the protocol has its own row, whose time the left column gives. A
lifeline is drawn in its rank's colour while the rank holds work, and thin while it is idle.
Grey arrows are requests, a #swatch(dark, shape: "line") dark arrow is a grant labelled with
the nodes it carried, and a #swatch(red, shape: "line") red arrow is a denial labelled with its
reason. A diamond is a load published to every rank. Steal timing differs between runs, so the
charts show this one recording, and after the recording the recording script checked that the
tree equals the one built without stealing.

#figure(panel("apx-steal-grant"), caption: brief([Work stealing: a victim grants], [The first round of the run in which a victim
  grants. #sa.thieves idle ranks ask rank #sa.victim, the only rank with work, and rank
  #sa.victim answers all of them between two batches. It grants #sa.grants.at(0).nodes and
  then #sa.grants.at(1).nodes nodes from the tail of its queue, and it publishes its smaller
  load after each grant. It denies the other #sa.denied, because only #sa.left points are left,
  which is no more than $w_*$.])) <fig:apx-steal-grant>

Every request in @fig:apx-steal-grant gets exactly one reply. The victim answers the requests
in one pass over its pending requests, between two of its batches, and meanwhile each thief
only waits for the rank it asked.

#figure(panel("apx-steal-retire"), caption: brief([Work stealing: the ranks retire], [The end of the same run, from the first request
  after the last reply that a working rank sent. Rank #sz.emptied empties its queue. Rank
  #sz.stale_thief has not yet read rank #sz.emptied's new load, so it asks rank #sz.emptied,
  and while it waits, it denies the requests of ranks
  #sz.waiting_denied.map(str).join(" and ") with the reason "idle". Rank #sz.first_join reads
  every load as zero and joins the barrier #sz.early_ms ms before rank #sz.last_join. In the
  barrier, a rank only answers requests. The barrier completes once the last rank has joined,
  and then every rank retires at once.])) <fig:apx-steal-retire>

@fig:apx-steal-retire shows two facts that the argument below rests on. First, a rank that
waits for its own reply still answers the requests sent to it, so its waiting never stops
other ranks. Second, the barrier decides retirement, and the loads do not: rank
#sz.first_join joined on loads that said nobody had work, while ranks
#sz.out_at_join.map(str).join(" and ") still had requests out, and that cost nothing but rank
#sz.first_join's help. Over the whole run, the script counts #apx-steal.requests requests and
as many replies, which are #apx-steal.replies.grant grants,
#(apx-steal.replies.mid_level_load) denials for too little work left,
#(apx-steal.replies.mid_level_no_clusters) for a single node left,
#apx-steal.replies.leased for leased work, and #apx-steal.replies.idle from idle ranks. ART
does not pass on work it received until it has processed it, so it denies requests for that
leased work.

Every MPI call that the protocol makes while ranks work is one of the following, and only the
column sends of a grant can be large.

#translate(caption: brief([The messages of work stealing], [The messages of work stealing and the MPI calls that carry them.]),
  [a thief asks; a victim or an idle rank denies; a rank publishes its load],
  [`send` of a small object, or `Send` of 8 bytes, each of which returns at once if the message is delivered eagerly],
  [a victim grants: a header, then each column of the clusters],
  [`send` of the header, then `Send` of each column, which may wait for the matching `Recv`],
  [a thief waits for its reply],
  [`Iprobe` for a grant or denial from its victim, in a polling loop in which the thief denies other requests],
  [a thief receives a grant],
  [`recv_clusters`: `recv`, then one `Recv` per column, from its victim only],
  [an idle rank with $sum_j L_j = 0$ retires],
  [`Ibarrier`, then `Test` in a polling loop in which the rank denies every request],
)

#lemma(title: [no deadlock])[
  Assume that every message of at most a few hundred bytes is delivered eagerly, which means
  that its blocking send returns before the receiver has posted the receive. Then the following
  hold.
  - A rank is never a victim that grants and a thief that waits at the same time.
  - A thief has at most one request out.
  - Every request is answered once.
  - The only rank that can block on a send is a victim that sends a grant, whose receiver is
    the thief that asked it, and that thief waits for this victim alone.
  So no set of ranks waits in a cycle.
] <lem:apx-no-deadlock>

#proof[
  _Roles._ `serve_requests` is the only place that grants, and it runs with a queue that
  still has unstarted clusters. `idle_step` is the only place that asks, and it runs only with
  an empty queue, where a rank answers every request with a denial (`_deny_idle_requests`),
  inside its wait loop and in the barrier too.

  _One request._ A thief sends one request and then loops until a grant or a denial arrives
  from that rank. A published load, even a zero, is not a reply.

  _One reply._ A victim answers every request it probes. It probes between batches, at every
  level boundary and, for ART, inside its split search, and a batch is finite, so the answer
  comes. If two idle ranks ask each other, each denies the other inside its wait loop.

  _Sends._ Requests, denials and loads are eager by assumption. The column `Send`s of a grant
  may wait for their receive; their sender is a victim, and their receiver is the thief that
  asked it. The thief's wait loop consists of probes and eager sends, so the thief reaches the
  probe that finds the header, and it then posts each `Recv` in the order in which the victim
  sends the columns. The victim waits only for that thief and the thief only for that victim,
  so their sends and receives match. A cycle would need a rank that waits as a thief while
  another rank waits for it as a victim, and the roles exclude that.
]

#proposition(title: [retirement])[
  When the barrier completes, no rank holds a cluster and no grant is in transit, so every
  cluster has been processed, by exactly one rank (@inv:steal). Also, once every queue is
  empty, the barrier completes.
] <prop:apx-retirement>

#proof[
  A rank joins the barrier only from `idle_step`, with an empty queue and no request out. Once
  in the barrier, it only reads loads and denies requests, so it never asks again and never
  receives a cluster. Clusters move only in a grant, a grant goes only to the rank that asked,
  and that rank does not join the barrier until it has received the grant. The barrier
  completes only when every rank has joined, and at that moment every queue is empty and no
  request is outstanding, so no grant is outstanding either. Conversely, a rank's last
  published load before it goes idle is the load of the empty queue that its last level left,
  which is zero. MPI keeps the order of the messages from one sender, and every idle rank
  keeps reading them, so each idle rank eventually sees $sum_j L_j = 0$ and joins.
]

#implementation(title: [What the code relies on])[
  - *The start is blocking.* `exchange_initial_loads` sends every rank's load to all other
    ranks and receives one load from each before the loop starts. Here ranks 1 to
    #(apx-steal.run.ranks - 1) start empty. Without the blocking start, such a rank would read
    its initial zeros as "nobody has work", join the barrier at once and never help. By
    @prop:apx-retirement that is not wrong, but the run would be serial.
  - *The load messages are drained exactly.* When the barrier completes, every request, reply
    and grant has been received, but loads are sent without waiting, so some may still be
    buffered. Each rank learns how many loads every other rank sent (`allgather`) and then
    receives exactly the ones it has not yet read. A probe that finds nothing proves nothing,
    because a message may still be on its way. Messages left behind would be pending at
    `MPI_Finalize`, or the next `refine` in the same process would read them as its initial
    loads, because it uses the same communicator and the same tag.
  - `tests/mpi/test_mpi.py` runs on several ranks. It checks that requests, replies and their
    receipts are equal in number, and that no message of any tag is pending after the run. It
    also checks that the work is conserved in two cases: when grants are larger than the eager
    limit, and when grants arrive after the victim has already published a zero load.
]

#pitfall(title: [Loads are hints])[
  A thief chooses its victim from loads that may be stale, and an idle rank decides to join
  the barrier from the same loads. MPI orders the messages of one sender, but not those of
  different senders, so a rank can read a zero from the victim of a grant before it reads the
  new load of the thief that received the grant. The rank then concludes that nobody has work
  and joins the barrier early. By @prop:apx-retirement this is safe, but the rank is lost for
  the rest of the run. Stale loads also lead thieves to ask ranks that have just gone idle, as
  rank #sz.stale_thief asks in @fig:apx-steal-retire. Last, the published load counts the
  whole queue of the current level, processed clusters included, so it overstates what a
  victim could still give.
]

#pitfall(title: [Eager delivery])[
  @lem:apx-no-deadlock assumes that the small messages are sent eagerly, but the MPI standard
  allows a blocking send to wait for its receive. In `exchange_initial_loads`, every rank first
  sends to all others and only then receives, so without eager delivery this exchange
  deadlocks at once. A request or a load sent to a rank in the middle of a long batch would
  also hold its sender until that batch ends. MPI libraries, the bundled MPICH among them, send
  messages this small eagerly by default, and a configuration that turns eager delivery off
  breaks the protocol.
]

#takeaway[A rank is either a victim that answers or a thief that waits for one answer, never
  both, so no waits form a cycle. A barrier decides retirement, and a rank joins it only with
  nothing to do and nothing on its way, so stale loads can cost parallelism but never lose
  work.]
