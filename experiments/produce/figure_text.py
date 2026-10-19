"""Wrap a figure's or table's description to the width of what it describes.

Text is measured as matplotlib renders it, so a note breaks exactly where it
would run past the plot area or table instead of widening the saved image
(every figure is saved with bbox_inches="tight", which grows the canvas to
fit an over-long line).
"""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.text import Text

LINE_SPACING = 1.2  # matplotlib's default for multi-line text


def text_width_in(fig: Figure, text: str, fontsize: float, **font) -> float:
    """Rendered width of one line of `text`, in inches."""
    probe = Text(0, 0, text, fontsize=fontsize, **font)
    probe.set_figure(fig)
    return probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def wrap_to_width(fig: Figure, text: str, width_in: float, fontsize: float, **font) -> str:
    """Break `text` at spaces so no line is wider than `width_in` inches.
    Existing line breaks are kept; a single word wider than the limit stays
    on its own line."""
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}" if line else word
            if line and text_width_in(fig, candidate, fontsize, **font) > width_in:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def line_height_in(fontsize: float) -> float:
    """Height one extra line of `fontsize` text adds, in inches."""
    return fontsize * LINE_SPACING / 72


def axes_right_in(fig: Figure, axes) -> float:
    """Right edge of the rightmost of `axes`, in inches from the figure's
    left edge (call after the layout is set)."""
    return max(ax.get_position().x1 for ax in axes) * fig.get_figwidth()
