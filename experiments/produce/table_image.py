"""Render a results DataFrame as a styled table PNG, alongside its CSV twin.

Shared by every produce_table_t*.py script so all report tables get the same
look without duplicating matplotlib table styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.produce.figure_text import line_height_in, wrap_to_width

HEADER_COLOR = "#1464A5"
HEADER_TEXT_COLOR = "white"
ROW_COLORS = ("white", "#FAFBFC")
BORDER_COLOR = "#AAB2BD"
TITLE_SIZE = 12
SUBTITLE_SIZE = 8.5


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
    n_rows = len(df)
    col_labels = [str(c) for c in df.columns]
    cell_text = [[_cell_text(v) for v in row] for row in df.itertuples(index=False)]

    col_chars = [
        max(len(col_labels[i]), df[col].map(_cell_text).str.len().max())
        for i, col in enumerate(df.columns)
    ]
    fig_width = max(6.0, sum(col_chars) * 0.13)
    row_height = 0.34
    table_height = row_height * (n_rows + 1)

    fig = plt.figure(figsize=(fig_width, table_height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER_COLOR)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.get_text().set_color(HEADER_TEXT_COLOR)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(ROW_COLORS[row % 2])

    # The auto-sized columns decide the table's real width; measure it, then
    # wrap the title and subtitle to it.
    fig.canvas.draw()
    extent = table.get_window_extent(fig.canvas.get_renderer())
    left_in, width_in = extent.x0 / fig.dpi, extent.width / fig.dpi

    header = []  # (text, fontsize, extra kwargs)
    if title:
        header.append((wrap_to_width(fig, title, width_in, TITLE_SIZE, fontweight="bold"),
                       TITLE_SIZE, {"fontweight": "bold"}))
    if subtitle:
        header.append((wrap_to_width(fig, subtitle, width_in, SUBTITLE_SIZE),
                       SUBTITLE_SIZE, {"color": "#555555"}))

    gap = 0.12  # inches above, between and below the header lines
    title_height = gap + sum(
        (text.count("\n") + 1) * line_height_in(size) + gap for text, size, _ in header
    )
    fig_height = table_height + title_height
    fig.set_size_inches(fig_width, fig_height)
    # The table keeps exactly its computed height under the header strip, so
    # bbox_inches="tight" leaves no blank space above or below it.
    ax.set_position((0.0, 0.0, 1.0, table_height / fig_height))

    top = gap
    for text, size, kwargs in header:
        fig.text(left_in / fig_width, 1 - top / fig_height, text, fontsize=size, va="top", **kwargs)
        top += (text.count("\n") + 1) * line_height_in(size) + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path
