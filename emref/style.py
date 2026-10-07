"""The one place for colours, sizes and the deterministic SVG setup.

Every figure in the suite draws from these tokens. Categorical slots are taken in
fixed order and follow the entity (a layer, an S-parameter), never its rank.
"""
from __future__ import annotations

# Figures are transparent and must read on GitHub's light and dark theme alike:
# one mid gray carries all text and chrome, and the series use the mid-lightness
# steps of the palette, validated against both surfaces.
INK = "#898781"
GRID = "#8987814d"      # INK at 30 percent

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500",
          "#d55181", "#008300", "#9085e9", "#e66767"]

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
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.transparent": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.labelcolor": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": LINE_WIDTH,
        "path.simplify": False,
    })


def save(fig, path, tight: bool = False) -> None:
    """tight crops the figure to its content: a layout fills its column instead of
    floating in the fixed frame its aspect ratio leaves."""
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None},
                bbox_inches="tight" if tight else None, pad_inches=0.05)
