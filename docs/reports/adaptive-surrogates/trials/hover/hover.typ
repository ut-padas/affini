// Trial: symbols that explain themselves. One source, two outputs:
//   typst compile --root ../.. hover.typ hover.pdf                                   (PDF: click a symbol)
//   typst compile --root ../.. --features html --format html hover.typ hover.html    (HTML: hover a symbol)
// tooltips.py then adds hover tooltips to the PDF for viewers that show them.
//
// The glossary is data: every symbol is defined once, with its words, its formal
// definition, its name in the code and the section that introduces it. A show rule finds
// the symbols in equations by themselves, so the prose is written as usual.

#let html-out() = target() == "html"

// ---------------------------------------------------------------------------- glossary
#let glossary = (
  "T": (
    symbol: $T$,
    name: [trust-region matrix],
    plain: "T: the trust-region matrix of a leaf. Its trust region is (x − c)ᵀ T (x − c) ≤ 1, the ellipsoid through the leaf's farthest point. In the code: trust_matrix.",
    formal: [$T = Sigma^(-1) slash alpha$, positive definite],
    code: "trust_matrix",
    where: <sec-trust>,
  ),
  "Σ": (
    symbol: $Sigma$,
    name: [scatter matrix],
    plain: "Σ: the scatter matrix of a leaf's points about its centre, Σ = Σᵢ (xᵢ − c)(xᵢ − c)ᵀ, with eigenvalues floored at γ.",
    formal: [$Sigma = sum_i (x_i - c)(x_i - c)^top$],
    code: "Cov (unnormalized covariance)",
    where: <sec-trust>,
  ),
  "α": (
    symbol: $alpha$,
    name: [farthest scaled distance],
    plain: "α: the largest (xᵢ − c)ᵀ Σ⁻¹ (xᵢ − c) over the leaf's points; dividing by it puts the farthest point on the ellipsoid.",
    formal: [$alpha = max_i (x_i - c)^top Sigma^(-1) (x_i - c)$],
    code: "max_dist",
    where: <sec-trust>,
  ),
  "c": (
    symbol: $c$,
    name: [centre of a cluster],
    plain: "c: the centre of a cluster, the mean of its points. In the code: centroids.",
    formal: [$c = 1 / abs(X_C) sum_(x in X_C) x$],
    code: "centroids",
    where: <sec-fit>,
  ),
  "κ": (
    symbol: $kappa$,
    name: [number of candidates],
    plain: "κ: how many centres the Euclidean search returns before the trust-region distances decide. In the code: k_nn · coarse_factor.",
    formal: [a positive integer; the manuscript uses $kappa = 2 d^2$],
    code: "k_nn * coarse_factor",
    where: <sec-inference>,
  ),
  "ν": (
    symbol: $nu$,
    name: [answering leaf],
    plain: "ν: the leaf that answers a query, the candidate with the smallest trust-region distance.",
    formal: [$nu = op("arg min")_(i in cal(C)_x) (x - c_i)^top T_i (x - c_i)$],
    code: "nn_idx[:, 0]",
    where: <sec-inference>,
  ),
)

// ---------------------------------------------------------------------------- marking
// The symbols of the glossary are found in equations by a show rule, so the text is
// written as usual. A letter means one thing only where its section says so: `scope`
// lists the symbols the rule may mark from here on.
#let scope = state("glossary-scope", ())
#let use-symbols(..keys) = scope.update(keys.pos())

#let mark(key, body) = context {
  if html-out() {
    // An <mi> keeps the symbol a MathML identifier, italic and spaced as one.
    html.elem("mi", attrs: (class: "sym", "data-sym": key, tabindex: "0"), body.text)
  } else {
    // In the PDF a symbol that explains itself is tinted, as a link is.
    link(label("gloss-" + key), text(fill: rgb("#1f4e79"), body))
  }
}

#let pattern = regex("^(" + glossary.keys().join("|") + ")$")
#show math.equation: it => {
  show pattern: s => context if s.text in scope.get() { mark(s.text, s) } else { s }
  it
}

// A symbol mentioned in running text, outside an equation.
#let sym(key) = context {
  let e = glossary.at(key)
  if html-out() {
    html.elem("span", attrs: (class: "sym", "data-sym": key, tabindex: "0"), e.symbol)
  } else {
    link(label("gloss-" + key), text(fill: rgb("#1f4e79"), e.symbol))
  }
}

// ---------------------------------------------------------------------------- page
#set document(title: [Symbols that explain themselves (trial)])
#set text(font: "Libertinus Serif", size: 10.5pt)
#show math.equation: set text(font: "New Computer Modern Math")
#set heading(numbering: "1.1")

= A trial: symbols that explain themselves

*How to try it.* In the HTML version, rest the pointer on a symbol in an equation, or tab
to it: a card shows what it is, its formula, its name in the code and the section that
defines it, and every other occurrence of the symbol lights up. In the PDF, click a symbol
to jump to its entry in the glossary at the end; the versions with tooltips also show a
short definition on hover, in viewers that support it.

= The fit <sec-fit>
#use-symbols("c")
A cluster's centre is $c$, the mean of its points, and its model is
$hat(f)(x) = w^top (x - c) + b$.

= Where a leaf can be trusted <sec-trust>
#use-symbols("c", "T", "Σ", "α")

// A picture beside its words: a grid in the PDF, a flex row in HTML.
#let scene(words, picture) = context if html-out() {
  html.elem("div", attrs: (class: "scene"))[#html.elem("div")[#words]#html.elem("div")[#picture]]
} else {
  grid(columns: (1fr, 38%), column-gutter: 1.5em, align: horizon, words, picture)
}

#scene(
  [
    The scatter matrix $Sigma = sum_i (x_i - c)(x_i - c)^top$ of a leaf's points gives the
    shape: the ellipsoid ${x : (x - c)^top Sigma^(-1) (x - c) <= 1}$ is long where the points
    spread. Scaling it by the largest value $alpha$ that $(x_i - c)^top Sigma^(-1) (x_i - c)$
    takes puts the farthest point exactly on the boundary:
    $ T = Sigma^(-1) / alpha, quad (x - c)^top T (x - c) <= 1 . $
  ],
  image("../../figures/afinni-trust-one.svg", width: 100%),
)

= Which leaf answers <sec-inference>
#use-symbols("c", "T", "κ", "ν")
A query $x$ is answered by the leaf
$ nu = op("arg min")_(i in cal(C)_x) (x - c_i)^top T_i (x - c_i) $
among the $kappa$ centres nearest to it, and flagged if
$(x - c_nu)^top T_nu (x - c_nu) > 1$. The same rule in running text: #sym("ν") is chosen
among #sym("κ") candidates.

// ---------------------------------------------------------------------------- glossary
= Glossary <glossary>
#use-symbols()
#for (key, e) in glossary [
  // The link target is the entry's first line, so a viewer's link preview starts there.
  #block(below: 0.9em)[
    #metadata(e.plain) #label("gloss-" + key)
    #box(width: 2.2em)[#e.symbol] *#e.name.* #e.formal. In the code: #raw(e.code).
    Defined in #ref(e.where).
  ]
]

// ---------------------------------------------------------------------------- HTML cards
// One hidden card per symbol, rendered by Typst (so its formula is MathML), and a small
// script that shows the card beside the symbol under the pointer or the keyboard focus.
#context if html-out() {
  for (key, e) in glossary {
    html.elem("template", attrs: (id: "card-" + key))[
      #html.elem("div", attrs: (class: "card-head"))[#e.symbol  #e.name]
      #html.elem("div", attrs: (class: "card-formal"))[#e.formal]
      #html.elem("div", attrs: (class: "card-code"))[in the code: #raw(e.code)]
    ]
  }
  html.elem("style", read("cards.css"))
  html.elem("script", read("cards.js"))
}
