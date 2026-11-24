// Lists of figures, tables and algorithms after the contents, with a short title per entry.

#import "theme.typ": theme

// True while the entries of a list are laid out, so that `brief` shows its short form there.
#let in-list = state("companion-in-list", false)

// A caption with a short form for the lists: `caption: brief([Short title], [The full caption.])`.
// The figure itself shows the full caption; a list shows the short title.
#let brief(short, long) = context if in-list.get() { short } else { long }

// One entry of the contents or of a list: the number in the accent colour in a column of its own,
// the title in the body colour, a faint leader and the page in the muted colour.
#let list-entry(it) = context {
  let t = theme()
  let el = it.element
  let number = if el.func() == heading { it.prefix() } else { numbering(el.numbering, ..el.counter.at(el.location())) }
  show link: set text(fill: black)  // the report tints every link; here only the number is tinted
  link(el.location(), grid(
    columns: (1.9em, 1fr),
    text(fill: t.accent, weight: "semibold", number),
    [#it.body() #box(width: 1fr, it.fill) #text(fill: t.muted, it.page())],
  ))
}

// The list of every captioned figure of `kind`, under `title`.
#let figure-list(kind, title) = {
  in-list.update(true)
  show outline.entry: list-entry
  outline(title: title, target: figure.where(kind: kind), indent: 0pt)
  in-list.update(false)
}
