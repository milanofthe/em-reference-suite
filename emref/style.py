"""The one place for colours, sizes and the deterministic SVG setup.

Every figure in the suite draws from these tokens. Categorical slots are taken in
fixed order and follow the entity (a layer, an S-parameter), never its rank.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Validated categorical order (light surface). Lines use adjacent pairs only.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

FONT = "sans-serif"
FONT_SIZE = 8
TITLE_SIZE = 9
LINE_WIDTH = 1.5
FIG_SIZE = (4.8, 4.2)    # inches; render and plot share it so the README rows align


def setup():
    """Matplotlib in a byte-reproducible configuration: text stays text, ids are
    salted with a constant, and no timestamp is written."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.fonttype": "none",
        "svg.hashsalt": "emref",
        "font.family": FONT,
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": LINE_WIDTH,
        "path.simplify": False,
    })


def save(fig, path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
