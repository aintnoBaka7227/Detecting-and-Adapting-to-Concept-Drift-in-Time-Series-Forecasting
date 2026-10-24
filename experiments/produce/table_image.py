"""Render a results DataFrame as a styled table PNG, alongside its CSV twin.

Shared by every produce_table_*.py script so all report tables get the same
look without duplicating matplotlib table styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.produce.figure_text import line_height_in, wrap_to_width

HEADER_COLOR = "#0D2C54"
HEADER_TEXT_COLOR = "white"
ROW_COLORS = ("white", "#FAFBFC")
BORDER_COLOR = "#3E5C80"
TITLE_SIZE = 12
SUBTITLE_SIZE = 8.5
CAPTION_SIZE = 10


def _cell_text(value: object) -> str:
    """Render one value as the table image shows it; missing data as an em dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return str(value)


def save_table_image(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> Path:
    """Draw `df` as a styled table and save it as a PNG at `out_path`.

    `subtitle`, if given, is a smaller note printed under the title -- e.g.
    to record a processing step or caveat that isn't obvious from the
    column names alone. Title and subtitle are wrapped to the table's
    rendered width, so they never make the image wider than the table.
    """
    return save_tables_image([(None, df)], out_path, title=title, subtitle=subtitle)


def _draw_table(ax, df: pd.DataFrame):
    ax.axis("off")
    table = ax.table(
        cellText=[[_cell_text(v) for v in row] for row in df.itertuples(index=False)],
        colLabels=[str(c) for c in df.columns],
        cellLoc="center",
        loc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER_COLOR)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.get_text().set_color(HEADER_TEXT_COLOR)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(ROW_COLORS[row % 2])
    return table


def _table_chars(df: pd.DataFrame) -> int:
    return sum(
        max(len(str(col)), df[col].map(_cell_text).str.len().max()) for col in df.columns
    )


def save_tables_image(
    tables: list[tuple[str | None, pd.DataFrame]],
    out_path: Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> Path:
    """Draw several tables stacked top to bottom in one PNG, each under an
    optional bold caption, all spanning the same width. Title, subtitle and
    captions are wrapped to that width."""
    fig_width = max(6.0, max(_table_chars(df) for _, df in tables) * 0.13)
    row_height = 0.34
    gap = 0.12  # inches above, between and below the header lines
    heights = [row_height * (len(df) + 1) for _, df in tables]

    fig = plt.figure(figsize=(fig_width, sum(heights)))
    axes = [fig.add_axes((0.0, 0.0, 1.0, 1.0)) for _ in tables]
    drawn = [_draw_table(ax, df) for ax, (_, df) in zip(axes, tables, strict=True)]

    # The auto-sized columns decide the tables' real width; measure the
    # widest, then wrap the text to it.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [t.get_window_extent(renderer) for t in drawn]
    left_in = min(e.x0 for e in extents) / fig.dpi
    width_in = max(e.width for e in extents) / fig.dpi

    def block_height(text: str, size: float) -> float:
        return (text.count("\n") + 1) * line_height_in(size) + gap

    header = []  # (text, fontsize, extra kwargs)
    if title:
        header.append((wrap_to_width(fig, title, width_in, TITLE_SIZE, fontweight="bold"),
                       TITLE_SIZE, {"fontweight": "bold"}))
    if subtitle:
        header.append((wrap_to_width(fig, subtitle, width_in, SUBTITLE_SIZE),
                       SUBTITLE_SIZE, {"color": "#555555"}))
    captions = [
        None if caption is None
        else wrap_to_width(fig, caption, width_in, CAPTION_SIZE, fontweight="bold")
        for caption, _ in tables
    ]

    header_height = gap + sum(block_height(text, size) for text, size, _ in header)
    fig_height = header_height + sum(
        h + (0 if c is None else block_height(c, CAPTION_SIZE) + gap)
        for h, c in zip(heights, captions, strict=True)
    ) + gap * (len(tables) - 1)
    fig.set_size_inches(fig_width, fig_height)

    top = gap
    for text, size, kwargs in header:
        fig.text(left_in / fig_width, 1 - top / fig_height, text, fontsize=size, va="top", **kwargs)
        top += block_height(text, size)
    top = header_height
    for i, (ax, height, caption) in enumerate(zip(axes, heights, captions, strict=True)):
        if i:
            top += gap
        if caption is not None:
            top += gap
            fig.text(left_in / fig_width, 1 - top / fig_height, caption, fontsize=CAPTION_SIZE,
                     va="top", fontweight="bold", color=HEADER_COLOR)
            top += block_height(caption, CAPTION_SIZE)
        # Each table keeps exactly its computed height, so bbox_inches="tight"
        # leaves no blank space around it.
        ax.set_position((0.0, 1 - (top + height) / fig_height, 1.0, height / fig_height))
        top += height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path
