"""Rendering-oriented helpers and display constants for diversity reports."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .model_registry import effort_sort_key
from .schemas.sample import SampleRecord

EFFORT_COLORS = {
    "none": "#0072B2",
    "low": "#CC79A7",
    "medium": "#E69F00",
    "high": "#009E73",
}

EFFORT_MARKERS = {
    "none": "circle",
    "low": "square",
    "medium": "triangle",
    "high": "diamond",
}


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def slugify(text: str) -> str:
    out = []
    for char in text.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_"}:
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "keyword"


def save_svg_scatter(
    *,
    path: Path,
    records: list[SampleRecord],
    coords: np.ndarray,
    explained_variance: list[float],
    title: str,
) -> None:
    colors = EFFORT_COLORS
    width = 960
    height = 720
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = coords[:, 0]
    y_values = coords[:, 1]
    min_x = float(np.min(x_values))
    max_x = float(np.max(x_values))
    min_y = float(np.min(y_values))
    max_y = float(np.max(y_values))

    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0

    def scale_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + (1.0 - (value - min_y) / (max_y - min_y)) * plot_height

    legend_efforts = []
    y_cursor = margin_top
    for effort in sorted({record.effort for record in records}, key=effort_sort_key):
        legend_efforts.append(
            f'<rect x="{width - 180}" y="{y_cursor - 10}" width="14" height="14" fill="{colors.get(effort, "#666")}"/>'
        )
        legend_efforts.append(
            f'<text x="{width - 160}" y="{y_cursor + 2}" font-size="14" font-family="monospace">{effort}</text>'
        )
        y_cursor += 24

    circles = []
    for index, record in enumerate(records):
        circles.append(
            (
                '<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.2" fill="{fill}" '
                'fill-opacity="0.68" stroke="white" stroke-width="0.8">'
                "<title>{title}</title></circle>"
            ).format(
                cx=scale_x(float(coords[index, 0])),
                cy=scale_y(float(coords[index, 1])),
                fill=colors.get(record.effort, "#666666"),
                title=html_escape(
                    f"effort={record.effort} sample={record.sample_index} idea={record.idea[:180]}"
                ),
            )
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_left}" y="36" font-size="24" font-family="monospace">Effort Diversity Pilot: {html_escape(title)}</text>
<text x="{margin_left}" y="58" font-size="14" font-family="monospace">PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}</text>
<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.2"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.2"/>
<text x="{width / 2:.1f}" y="{height - 24}" font-size="14" text-anchor="middle" font-family="monospace">PC1</text>
<text x="26" y="{height / 2:.1f}" font-size="14" text-anchor="middle" transform="rotate(-90 26 {height / 2:.1f})" font-family="monospace">PC2</text>
{''.join(legend_efforts)}
{''.join(circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
