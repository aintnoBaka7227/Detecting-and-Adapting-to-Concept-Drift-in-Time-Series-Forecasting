"""Render a results DataFrame as a styled table PNG, alongside its CSV twin.

Shared by every produce_table_t1*.py and produce_table_t2*.py script so all
report tables get the same look without duplicating matplotlib table styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HEADER_COLOR = "#1464A5"
HEADER_TEXT_COLOR = "white"
ROW_COLORS = ("white", "#FAFBFC")
BORDER_COLOR = "#AAB2BD"


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
    column names alone.
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
    title_height = 0.1
    if title:
        title_height += 0.32
    if subtitle:
        title_height += 0.34
    table_height = row_height * (n_rows + 1)
    fig_height = table_height + title_height

    fig = plt.figure(figsize=(fig_width, fig_height))

    if title:
        fig.text(0.01, 1 - (0.28 / fig_height), title, fontsize=12, fontweight="bold", va="top")
    if subtitle:
        subtitle_y = 1 - ((0.64 if title else 0.28) / fig_height)
        fig.text(0.01, subtitle_y, subtitle, fontsize=8.5, color="#555555", va="top")

    # Axes fills the whole figure except the reserved title strip; giving the
    # table an explicit bbox=[0, 0, 1, 1] within it means the figure height
    # computed above is exactly what gets rendered -- no auto-centred table
    # leaving blank space above/below once bbox_inches="tight" crops the rest.
    ax = fig.add_axes((0.0, 0.0, 1.0, table_height / fig_height))
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path
