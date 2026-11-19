"""Add hover tooltips to the symbol links of hover.pdf, written to hover-tooltips.pdf.

Usage (from this directory): uv run --no-project --with pikepdf python tooltips.py

Typst writes each symbol in an equation as a link to its glossary entry. This script asks
Typst where each entry is and what its one-line definition says (the entry's metadata),
finds the links that point there, and lays a transparent push button over each one: its
tooltip (/TU) is the definition, and clicking it still jumps to the entry. Whether a viewer
shows the tooltip is up to the viewer; the click works everywhere.
"""

import json
import subprocess
from pathlib import Path

import pikepdf

HERE = Path(__file__).resolve().parent


def glossary_entries() -> list[tuple[str, int, float, float]]:
    """(one-line definition, page index, x, y from the top) of every glossary entry, in pt."""
    expr = 'query(metadata).filter(m => str(m.at("label", default: "")).starts-with("gloss-")).map(m => (m.value, m.location().position()))'
    out = subprocess.run(["typst", "eval", "--root", "../..", expr, "--in", "hover.typ"], cwd=HERE, capture_output=True,
                         text=True, check=True).stdout  # fmt: skip
    return [(text, pos["page"] - 1, float(pos["x"][:-2]), float(pos["y"][:-2])) for text, pos in json.loads(out)]


def main() -> None:
    pdf = pikepdf.open(HERE / "hover.pdf")
    entries = glossary_entries()
    fields = pikepdf.Array()
    added = 0
    names = pdf.Root.get("/Names")
    named = pikepdf.NameTree(names.Dests) if names is not None and "/Dests" in names else {}
    for page in pdf.pages:
        annots = page.obj.get("/Annots", pikepdf.Array())
        new = pikepdf.Array()
        for annot in annots:
            new.append(annot)
            dest = annot.get("/Dest")
            if annot.get("/Subtype") != "/Link" or dest is None:
                continue
            if not isinstance(dest, pikepdf.Array):  # a named destination: look it up
                dest = named[str(dest)]
                dest = dest.D if isinstance(dest, pikepdf.Dictionary) else dest
            target, left, top = dest[0], float(dest[2]), float(dest[3])
            height = float(target.MediaBox[3])
            # The link targets the top of the entry's line, the query gives its baseline: take
            # the nearest entry on the same page, a line's height at most.
            near = sorted((abs(height - y - top), text) for text, p, x, y in entries
                          if pdf.pages[p].obj.objgen == target.objgen and abs(x - left) < 1)  # fmt: skip
            if not near or near[0][0] > 14:
                continue
            match = [near[0][1]]
            # An empty appearance keeps the button invisible; the link underneath still draws nothing.
            blank = pdf.make_stream(b"", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form,
                                    BBox=[0, 0, 1, 1])  # fmt: skip
            button = pdf.make_indirect(pikepdf.Dictionary(
                Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Widget, FT=pikepdf.Name.Btn,
                Ff=1 << 16,  # push button
                T=pikepdf.String(f"sym{added}"), TU=pikepdf.String(match[0]), Rect=annot.Rect, F=4,
                A=pikepdf.Dictionary(S=pikepdf.Name.GoTo, D=dest), AP=pikepdf.Dictionary(N=blank),
                P=page.obj,
            ))  # fmt: skip
            new.append(button)
            fields.append(button)
            added += 1
        page.obj.Annots = new
    pdf.Root.AcroForm = pikepdf.Dictionary(Fields=fields, NeedAppearances=False)
    pdf.save(HERE / "hover-tooltips.pdf", deterministic_id=True)
    print(f"{added} symbol tooltips added -> hover-tooltips.pdf")


if __name__ == "__main__":
    main()
