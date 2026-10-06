#!/usr/bin/env python3
"""Render a category x effort cosine-similarity figure from summary_by_category_effort.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.render_effort_diversity_report import load_csv
from src.model_registry import EFFORT_ORDER
from src.visualizer import EFFORT_COLORS, html_escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a category-level grouped figure with cosine similarity on the y-axis "
            "from summary_by_category_effort.csv."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains summary_by_category_effort.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to <run-dir>/category_similarity_by_effort_fixed_0_1.svg.",
    )
    parser.add_argument(
        "--title",
        default="LiveIdeaBench Category x Effort Pairwise Cosine Similarity",
        help="Figure title.",
    )
    parser.add_argument(
        "--subtitle",
        default=(
            "Metric: 1 - mean_pairwise_cosine_distance aggregated over keywords/category"
            " | higher = more similar / more duplicated | y-axis fixed to [0, 1]"
        ),
        help="Figure subtitle.",
    )
    return parser.parse_args()


def render_category_similarity_svg(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    categories = sorted({row["category"] for row in rows})
    rows_by_key = {(row["category"], row["effort"]): row for row in rows}

    bars_per_group = len(EFFORT_ORDER)
    bar_width = 16
    intra_gap = 4
    group_gap = 18
    group_width = bars_per_group * bar_width + (bars_per_group - 1) * intra_gap

    width = max(2200, 140 + len(categories) * (group_width + group_gap) + 120)
    height = 860
    margin_left = 90
    margin_right = 30
    margin_top = 78
    margin_bottom = 190
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_y(value: float) -> float:
        clipped = max(0.0, min(1.0, value))
        return margin_top + (1.0 - clipped) * plot_height

    group_spacing = plot_width / max(1, len(categories))
    category_starts = {
        category: margin_left + group_spacing * index + (group_spacing - group_width) / 2
        for index, category in enumerate(categories)
    }

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="36" font-size="24" font-family="monospace">{html_escape(title)}</text>',
        f'<text x="{margin_left}" y="58" font-size="14" font-family="monospace">{html_escape(subtitle)}</text>',
    ]

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = scale_y(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" font-size="13" text-anchor="end" font-family="monospace">{tick:.1f}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.4"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="28" y="{height / 2:.1f}" font-size="14" text-anchor="middle" transform="rotate(-90 28 {height / 2:.1f})" font-family="monospace">Pairwise Cosine Similarity</text>'
    )

    legend_x = width - 250
    legend_y = 98
    for index, effort in enumerate(EFFORT_ORDER):
        y = legend_y + index * 22
        color = EFFORT_COLORS[effort]
        parts.append(
            f'<rect x="{legend_x}" y="{y - 6}" width="14" height="14" fill="{color}" fill-opacity="0.85" stroke="{color}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{legend_x + 22}" y="{y + 4}" font-size="13" font-family="monospace">{html_escape(effort)}</text>'
        )

    baseline = height - margin_bottom
    for category in categories:
        start_x = category_starts[category]
        end_x = start_x + group_width
        center_x = (start_x + end_x) / 2
        parts.append(
            f'<line x1="{center_x:.2f}" y1="{margin_top}" x2="{center_x:.2f}" y2="{baseline}" stroke="#f3f3f3" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{start_x - 8:.2f}" y1="{baseline + 4}" x2="{end_x + 8:.2f}" y2="{baseline + 4}" stroke="#aaaaaa" stroke-width="1"/>'
        )

        for effort_index, effort in enumerate(EFFORT_ORDER):
            row = rows_by_key.get((category, effort))
            if row is None:
                continue
            x = start_x + effort_index * (bar_width + intra_gap)
            mean_distance = float(row["mean_pairwise_cosine_distance"])
            std_distance = float(row["std_pairwise_cosine_distance"])
            similarity = 1.0 - mean_distance
            std_similarity = std_distance
            ymin = max(0.0, similarity - std_similarity)
            ymax = min(1.0, similarity + std_similarity)
            y = scale_y(similarity)
            color = EFFORT_COLORS[effort]
            bar_height = baseline - y

            parts.append(
                (
                    '<rect x="{x:.2f}" y="{y:.2f}" width="{w}" height="{h:.2f}" '
                    'fill="{fill}" fill-opacity="0.82" stroke="{fill}" stroke-width="1">'
                    "<title>{title}</title></rect>"
                ).format(
                    x=x,
                    y=y,
                    w=bar_width,
                    h=bar_height,
                    fill=color,
                    title=html_escape(
                        f"{category} | {effort} | mean_similarity={similarity:.4f} | std={std_similarity:.4f} | n={row['n_keywords']}"
                    ),
                )
            )
            center_bar_x = x + bar_width / 2
            parts.append(
                f'<line x1="{center_bar_x:.2f}" y1="{scale_y(ymin):.2f}" x2="{center_bar_x:.2f}" y2="{scale_y(ymax):.2f}" stroke="#222" stroke-width="1.6"/>'
            )
            parts.append(
                f'<line x1="{center_bar_x - 4:.2f}" y1="{scale_y(ymin):.2f}" x2="{center_bar_x + 4:.2f}" y2="{scale_y(ymin):.2f}" stroke="#222" stroke-width="1.6"/>'
            )
            parts.append(
                f'<line x1="{center_bar_x - 4:.2f}" y1="{scale_y(ymax):.2f}" x2="{center_bar_x + 4:.2f}" y2="{scale_y(ymax):.2f}" stroke="#222" stroke-width="1.6"/>'
            )
            parts.append(
                (
                    '<text x="{x:.2f}" y="{y:.2f}" font-size="10" text-anchor="end" '
                    'transform="rotate(-90 {x:.2f} {y:.2f})" font-family="monospace">{label}</text>'
                ).format(
                    x=center_bar_x + 4,
                    y=baseline + 46,
                    label=html_escape(effort),
                )
            )

        parts.append(
            (
                '<text x="{x:.2f}" y="{y:.2f}" font-size="11" text-anchor="end" '
                'transform="rotate(-55 {x:.2f} {y:.2f})" font-family="monospace">{label}</text>'
            ).format(
                x=center_x + 10,
                y=baseline + 112,
                label=html_escape(category),
            )
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = load_csv(run_dir / "summary_by_category_effort.csv")
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else run_dir / "category_similarity_by_effort_fixed_0_1.svg"
    )
    render_category_similarity_svg(
        rows,
        output_path,
        title=args.title,
        subtitle=args.subtitle,
    )
    print(output_path)


if __name__ == "__main__":
    main()
