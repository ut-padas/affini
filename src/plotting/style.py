"""Shared matplotlib style and figure helpers, so every AFINNI plot looks the same."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "image.cmap": "viridis",
}


@contextmanager
def afinni_style():
    """Temporarily apply ``STYLE`` (restores the caller's rcParams afterwards)."""
    with mpl.rc_context(STYLE):
        yield


def finish(fig: plt.Figure, path: str | Path | None = None, close: bool = False) -> plt.Figure:
    """Save ``fig`` to ``path`` (creating parent folders) and optionally close it; returns ``fig``."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    if close:
        plt.close(fig)
    return fig
