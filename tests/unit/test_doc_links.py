"""Every relative link in the current-state documentation points at a file that exists."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# The technical review describes an earlier snapshot of the code, line numbers included.
SNAPSHOTS = {"docs/AFINNI_ART_TECHNICAL_REVIEW.md"}
DOCS = sorted(
    str(p.relative_to(ROOT)) for p in [ROOT / "README.md", ROOT / "AGENTS.md", *(ROOT / "docs").glob("*.md")]
    if str(p.relative_to(ROOT)) not in SNAPSHOTS
)  # fmt: skip


@pytest.mark.parametrize("doc", DOCS)
def test_relative_links_resolve(doc):
    text = (ROOT / doc).read_text()
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        path = target.partition("#")[0]
        if not path or re.match(r"^[a-z]+:", target):
            continue
        if not ((ROOT / doc).parent / path).exists():
            broken.append(target)
    assert not broken, f"{doc}: {broken}"
