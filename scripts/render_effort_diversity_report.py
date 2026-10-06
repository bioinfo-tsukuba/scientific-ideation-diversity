#!/usr/bin/env python3
"""Render markdown and fixed-scale SVG artifacts for an effort-diversity run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.metrics import build_category_summary_rows
from src.model_registry import EFFORT_ORDER
from src.visualizer import EFFORT_COLORS, html_escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render markdown and within-set diversity SVG from a run directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory produced by scripts/run_diversity_experiment.py",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
def render_diversity_svg(
    summary_rows: list[dict[str, str]],
    output_path: Path,
    *,
    title: str = "Within-Set Diversity by Effort",
    subtitle: str = "Metric: mean_pairwise_cosine_distance | y-axis fixed to [0, 1]",
) -> None:
    values_by_effort: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in summary_rows:
        values_by_effort[row["effort"]].append(
            (row["keyword"], float(row["mean_pairwise_cosine_distance"]))
        )

    width = 980
    height = 680
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_height

    group_spacing = plot_width / max(1, len(EFFORT_ORDER))
    centers = {
        effort: margin_left + group_spacing * (index + 0.5)
        for index, effort in enumerate(EFFORT_ORDER)
    }
    box_width = 68

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="90" y="34" font-size="24" font-family="monospace">{html_escape(title)}</text>',
        f'<text x="90" y="56" font-size="14" font-family="monospace">{html_escape(subtitle)}</text>',
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
        f'<text x="26" y="{height / 2:.1f}" font-size="14" text-anchor="middle" transform="rotate(-90 26 {height / 2:.1f})" font-family="monospace">Pairwise Cosine Distance</text>'
    )

    for effort in EFFORT_ORDER:
        values = [value for _, value in values_by_effort[effort]]
        if not values:
            continue

        center_x = centers[effort]
        q1 = quantile(values, 0.25)
        q2 = quantile(values, 0.50)
        q3 = quantile(values, 0.75)
        vmin = min(values)
        vmax = max(values)
        mean_value = sum(values) / len(values)
        color = EFFORT_COLORS[effort]

        left = center_x - box_width / 2
        right = center_x + box_width / 2

        parts.append(
            f'<line x1="{center_x:.2f}" y1="{scale_y(vmin):.2f}" x2="{center_x:.2f}" y2="{scale_y(vmax):.2f}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{left + 12:.2f}" y1="{scale_y(vmin):.2f}" x2="{right - 12:.2f}" y2="{scale_y(vmin):.2f}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{left + 12:.2f}" y1="{scale_y(vmax):.2f}" x2="{right - 12:.2f}" y2="{scale_y(vmax):.2f}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<rect x="{left:.2f}" y="{scale_y(q3):.2f}" width="{box_width:.2f}" height="{scale_y(q1) - scale_y(q3):.2f}" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{left:.2f}" y1="{scale_y(q2):.2f}" x2="{right:.2f}" y2="{scale_y(q2):.2f}" stroke="{color}" stroke-width="2.4"/>'
        )
        parts.append(
            f'<circle cx="{center_x:.2f}" cy="{scale_y(mean_value):.2f}" r="5.5" fill="{color}" fill-opacity="0.85" stroke="white" stroke-width="1.2"/>'
        )

        effort_values = values_by_effort[effort]
        for index, (keyword, value) in enumerate(effort_values):
            offset = ((index % 7) - 3) * 7 + (index // 7 % 2) * 2
            cx = center_x + offset
            cy = scale_y(value)
            parts.append(
                (
                    '<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.8" fill="{fill}" fill-opacity="0.55" stroke="white" stroke-width="0.8">'
                    "<title>{title}</title></circle>"
                ).format(
                    cx=cx,
                    cy=cy,
                    fill=color,
                    title=html_escape(f"{effort} | {keyword} | {value:.4f}"),
                )
            )

        parts.append(
            f'<text x="{center_x:.2f}" y="{height - margin_bottom + 26}" font-size="15" text-anchor="middle" font-family="monospace">{effort}</text>'
        )
        parts.append(
            f'<text x="{center_x:.2f}" y="{height - margin_bottom + 46}" font-size="12" text-anchor="middle" font-family="monospace">mean={mean_value:.3f}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_category_diversity_svg(
    category_summary_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Category-Level Within-Set Diversity",
    subtitle: str = "Metric: mean_pairwise_cosine_distance aggregated over 5 keywords/category | y-axis fixed to [0, 1]",
) -> None:
    categories = sorted({row["category"] for row in category_summary_rows})
    rows_by_key = {
        (row["category"], row["effort"]): row
        for row in category_summary_rows
    }

    width = max(1500, 180 + len(categories) * 56)
    height = 860
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 260
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_y(value: float) -> float:
        clipped = max(0.0, min(1.0, value))
        return margin_top + (1.0 - clipped) * plot_height

    group_spacing = plot_width / max(1, len(categories))
    category_centers = {
        category: margin_left + group_spacing * (index + 0.5)
        for index, category in enumerate(categories)
    }
    effort_offsets = {
        "none": -16,
        "low": -5,
        "medium": 5,
        "high": 16,
    }

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="90" y="34" font-size="24" font-family="monospace">{html_escape(title)}</text>',
        f'<text x="90" y="56" font-size="14" font-family="monospace">{html_escape(subtitle)}</text>',
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
        f'<text x="26" y="{height / 2:.1f}" font-size="14" text-anchor="middle" transform="rotate(-90 26 {height / 2:.1f})" font-family="monospace">Pairwise Cosine Distance</text>'
    )

    legend_x = width - 260
    legend_y = 96
    for index, effort in enumerate(EFFORT_ORDER):
        y = legend_y + index * 22
        color = EFFORT_COLORS[effort]
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}" fill-opacity="0.85"/>')
        parts.append(
            f'<text x="{legend_x + 14}" y="{y + 4}" font-size="13" font-family="monospace">{effort}</text>'
        )

    for category in categories:
        center_x = category_centers[category]
        parts.append(
            f'<line x1="{center_x:.2f}" y1="{margin_top}" x2="{center_x:.2f}" y2="{height - margin_bottom}" stroke="#f3f3f3" stroke-width="1"/>'
        )
        for effort in EFFORT_ORDER:
            row = rows_by_key.get((category, effort))
            if row is None:
                continue
            color = EFFORT_COLORS[effort]
            x = center_x + effort_offsets[effort]
            mean_value = float(row["mean_pairwise_cosine_distance"])
            std_value = float(row["std_pairwise_cosine_distance"])
            ymin = max(0.0, mean_value - std_value)
            ymax = min(1.0, mean_value + std_value)
            parts.append(
                f'<line x1="{x:.2f}" y1="{scale_y(ymin):.2f}" x2="{x:.2f}" y2="{scale_y(ymax):.2f}" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{x - 5:.2f}" y1="{scale_y(ymin):.2f}" x2="{x + 5:.2f}" y2="{scale_y(ymin):.2f}" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{x - 5:.2f}" y1="{scale_y(ymax):.2f}" x2="{x + 5:.2f}" y2="{scale_y(ymax):.2f}" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(
                (
                    '<circle cx="{x:.2f}" cy="{y:.2f}" r="5.2" fill="{fill}" fill-opacity="0.88" stroke="white" stroke-width="1.0">'
                    "<title>{title}</title></circle>"
                ).format(
                    x=x,
                    y=scale_y(mean_value),
                    fill=color,
                    title=html_escape(
                        f"{category} | {effort} | mean={mean_value:.4f} | std={std_value:.4f} | n={row['n_keywords']}"
                    ),
                )
            )

        parts.append(
            (
                '<text x="{x:.2f}" y="{y:.2f}" font-size="11" text-anchor="end" '
                'transform="rotate(-55 {x:.2f} {y:.2f})" font-family="monospace">{label}</text>'
            ).format(
                x=center_x + 10,
                y=height - margin_bottom + 88,
                label=html_escape(category),
            )
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_markdown(
    *,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, str]],
    keyword_summary_rows: list[dict[str, str]],
    samples: list[dict[str, Any]],
    output_path: Path,
    svg_name: str,
    category_svg_name: str,
) -> None:
    summary_by_effort = {row["effort"]: row for row in summary_rows}
    grouped_samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    category_by_keyword: dict[str, str | None] = {}
    for sample in samples:
        grouped_samples[sample["effort"]][sample["keyword"]].append(sample)
        category_by_keyword[sample["keyword"]] = sample.get("category")

    metric_rows = []
    for effort in EFFORT_ORDER:
        row = summary_by_effort.get(effort)
        if not row:
            continue
        metric_rows.append(
            "| {effort} | {mean_output_tokens:.1f} | {mean_pairwise:.3f} | {mean_centroid:.3f} | {reasoning_rate:.1f} |".format(
                effort=effort,
                mean_output_tokens=float(row["mean_output_tokens"]),
                mean_pairwise=float(row["mean_pairwise_cosine_distance"]),
                mean_centroid=float(row["mean_distance_to_centroid"]),
                reasoning_rate=float(row["mean_reasoning_block_rate"]),
            )
        )

    keyword_effect_counts: dict[str, dict[str, int]] = {}
    metric_names = [
        "mean_pairwise_cosine_distance",
        "mean_distance_to_centroid",
    ]
    grouped_keyword_rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in keyword_summary_rows:
        grouped_keyword_rows[row["keyword"]][row["effort"]] = row
    for metric_name in metric_names:
        medium_gt_none = 0
        high_gt_none = 0
        low_lt_none = 0
        total = 0
        for keyword, effort_rows in grouped_keyword_rows.items():
            if not all(effort in effort_rows for effort in EFFORT_ORDER):
                continue
            total += 1
            if float(effort_rows["medium"][metric_name]) > float(effort_rows["none"][metric_name]):
                medium_gt_none += 1
            if float(effort_rows["high"][metric_name]) > float(effort_rows["none"][metric_name]):
                high_gt_none += 1
            if float(effort_rows["low"][metric_name]) < float(effort_rows["none"][metric_name]):
                low_lt_none += 1
        keyword_effect_counts[metric_name] = {
            "medium_gt_none": medium_gt_none,
            "high_gt_none": high_gt_none,
            "low_lt_none": low_lt_none,
            "total": total,
        }

    lines: list[str] = [
        "# Effort Diversity Pilot Report",
        "",
        "## Run",
        "",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Keywords: `{manifest['n_keywords']}`",
        f"- Samples per effort: `{manifest['samples_per_effort']}`",
        f"- Efforts: `{', '.join(manifest['efforts'])}`",
        f"- Generation model: `{manifest['idea_model']}`",
        f"- Embedding model: `{manifest['embedding_model_id']}`",
        "",
        "## Summary",
        "",
        "| Effort | Mean Output Tokens | Mean Pairwise Distance | Mean Distance to Centroid | Reasoning Block Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        *metric_rows,
        "",
        "## Within-Set Diversity",
        "",
        "Graph metric: `mean_pairwise_cosine_distance` computed within each `keyword x effort` set.",
        "The y-axis is fixed to `[0, 1]`.",
        "",
        f"![Within-set diversity]({svg_name})",
        "",
        "## Category-Level Diversity",
        "",
        "Category-level plot uses the mean and standard deviation of `mean_pairwise_cosine_distance` across the 5 keywords sampled within each category.",
        "The y-axis is fixed to `[0, 1]`.",
        "",
        f"![Category-level diversity]({category_svg_name})",
        "",
        "## Keyword-Level Checks",
        "",
        (
            f"- `mean_pairwise_cosine_distance`: `medium > none` in "
            f"`{keyword_effect_counts['mean_pairwise_cosine_distance']['medium_gt_none']}/"
            f"{keyword_effect_counts['mean_pairwise_cosine_distance']['total']}` keywords, "
            f"`high > none` in `{keyword_effect_counts['mean_pairwise_cosine_distance']['high_gt_none']}/"
            f"{keyword_effect_counts['mean_pairwise_cosine_distance']['total']}`, "
            f"`low < none` in `{keyword_effect_counts['mean_pairwise_cosine_distance']['low_lt_none']}/"
            f"{keyword_effect_counts['mean_pairwise_cosine_distance']['total']}`."
        ),
        (
            f"- `mean_distance_to_centroid`: `medium > none` in "
            f"`{keyword_effect_counts['mean_distance_to_centroid']['medium_gt_none']}/"
            f"{keyword_effect_counts['mean_distance_to_centroid']['total']}` keywords, "
            f"`high > none` in `{keyword_effect_counts['mean_distance_to_centroid']['high_gt_none']}/"
            f"{keyword_effect_counts['mean_distance_to_centroid']['total']}`, "
            f"`low < none` in `{keyword_effect_counts['mean_distance_to_centroid']['low_lt_none']}/"
            f"{keyword_effect_counts['mean_distance_to_centroid']['total']}`."
        ),
        "",
        "## Final Outputs",
        "",
        "Only the final model output is listed below. `reasoningContent` is intentionally excluded.",
        "",
    ]

    keywords = sorted(category_by_keyword, key=lambda keyword: (category_by_keyword[keyword] or "", keyword))
    for effort in EFFORT_ORDER:
        lines.extend([f"## {effort}", ""])
        for keyword in keywords:
            samples_for_keyword = sorted(
                grouped_samples[effort][keyword],
                key=lambda sample: sample["sample_index"],
            )
            category = category_by_keyword.get(keyword)
            lines.extend(
                [
                    f"### {category or 'NA'} / {keyword}",
                    "",
                ]
            )
            for sample in samples_for_keyword:
                lines.extend(
                    [
                        f"#### Sample {sample['sample_index']}",
                        "",
                        "```text",
                        sample["full_response"].rstrip(),
                        "```",
                        "",
                    ]
                )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_keyword_first_markdown(
    *,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, str]],
    keyword_summary_rows: list[dict[str, str]],
    samples: list[dict[str, Any]],
    output_path: Path,
    svg_name: str,
    category_svg_name: str,
) -> None:
    summary_by_effort = {row["effort"]: row for row in summary_rows}
    grouped_samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    grouped_keyword_rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    category_by_keyword: dict[str, str | None] = {}

    for sample in samples:
        grouped_samples[sample["keyword"]][sample["effort"]].append(sample)
        category_by_keyword[sample["keyword"]] = sample.get("category")

    for row in keyword_summary_rows:
        grouped_keyword_rows[row["keyword"]][row["effort"]] = row

    lines: list[str] = [
        "# Effort Diversity Pilot Report",
        "",
        "## How To Read",
        "",
        "- Each keyword was generated `10` times for each effort condition.",
        "- For one keyword, you therefore have `none=10`, `low=10`, `medium=10`, `high=10`.",
        "- `Sample 0` to `Sample 9` are independent generations under the same `keyword x effort` condition.",
        "- This report is grouped by keyword first so you can compare efforts side by side.",
        "",
        "## Overall Summary",
        "",
        "| Effort | Mean Output Tokens | Mean Pairwise Distance | Mean Distance to Centroid |",
        "| --- | ---: | ---: | ---: |",
    ]

    for effort in EFFORT_ORDER:
        row = summary_by_effort.get(effort)
        if not row:
            continue
        lines.append(
            "| {effort} | {mean_output_tokens:.1f} | {mean_pairwise:.3f} | {mean_centroid:.3f} |".format(
                effort=effort,
                mean_output_tokens=float(row["mean_output_tokens"]),
                mean_pairwise=float(row["mean_pairwise_cosine_distance"]),
                mean_centroid=float(row["mean_distance_to_centroid"]),
            )
        )

    lines.extend(
        [
            "",
            f"![Within-set diversity]({svg_name})",
            "",
            f"![Category-level diversity]({category_svg_name})",
            "",
        ]
    )

    keywords = sorted(category_by_keyword, key=lambda keyword: (category_by_keyword[keyword] or "", keyword))
    for keyword in keywords:
        category = category_by_keyword.get(keyword)
        lines.extend([f"## {category or 'NA'} / {keyword}", ""])
        if keyword in grouped_keyword_rows:
            lines.extend(
                [
                    "| Effort | Mean Output Tokens | Mean Pairwise Distance | Mean Distance to Centroid |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for effort in EFFORT_ORDER:
                row = grouped_keyword_rows[keyword].get(effort)
                if not row:
                    continue
                lines.append(
                    "| {effort} | {mean_output_tokens:.1f} | {mean_pairwise:.3f} | {mean_centroid:.3f} |".format(
                        effort=effort,
                        mean_output_tokens=float(row["mean_output_tokens"]),
                        mean_pairwise=float(row["mean_pairwise_cosine_distance"]),
                        mean_centroid=float(row["mean_distance_to_centroid"]),
                    )
                )
            lines.append("")

        for effort in EFFORT_ORDER:
            effort_samples = sorted(
                grouped_samples[keyword][effort],
                key=lambda sample: sample["sample_index"],
            )
            lines.extend([f"### {effort}", ""])
            for sample in effort_samples:
                lines.extend(
                    [
                        f"#### Sample {sample['sample_index']}",
                        "",
                        "```text",
                        sample["full_response"].rstrip(),
                        "```",
                        "",
                    ]
                )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_rows = load_csv(run_dir / "summary.csv")
    keyword_summary_rows = load_csv(run_dir / "summary_by_keyword_effort.csv")
    samples = load_jsonl(run_dir / "samples.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    category_summary_rows = build_category_summary_rows(keyword_summary_rows)

    svg_path = run_dir / "within_set_diversity_pairwise_fixed_0_1.svg"
    category_svg_path = run_dir / "category_diversity_pairwise_fixed_0_1.svg"
    category_summary_csv_path = run_dir / "summary_by_category_effort.csv"
    markdown_path = run_dir / "effort_outputs_report.md"
    keyword_first_markdown_path = run_dir / "keyword_outputs_report.md"

    render_diversity_svg(keyword_summary_rows, svg_path)
    write_csv(category_summary_csv_path, category_summary_rows)
    render_category_diversity_svg(category_summary_rows, category_svg_path)
    render_markdown(
        manifest=manifest,
        summary_rows=summary_rows,
        keyword_summary_rows=keyword_summary_rows,
        samples=samples,
        output_path=markdown_path,
        svg_name=svg_path.name,
        category_svg_name=category_svg_path.name,
    )
    render_keyword_first_markdown(
        manifest=manifest,
        summary_rows=summary_rows,
        keyword_summary_rows=keyword_summary_rows,
        samples=samples,
        output_path=keyword_first_markdown_path,
        svg_name=svg_path.name,
        category_svg_name=category_svg_path.name,
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "markdown": str(markdown_path),
                "keyword_first_markdown": str(keyword_first_markdown_path),
                "svg": str(svg_path),
                "category_svg": str(category_svg_path),
                "category_summary_csv": str(category_summary_csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
