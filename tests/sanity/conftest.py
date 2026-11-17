"""
Tests of studies/sanity: the comparison of this code with the earlier draft's numbers and with the
reference ART. They import that folder's scripts by name, as its drivers do, and go with it.
"""

import sys
from pathlib import Path

SANITY = Path(__file__).resolve().parents[2] / "studies" / "sanity"
for folder in (SANITY, SANITY / "paper"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
