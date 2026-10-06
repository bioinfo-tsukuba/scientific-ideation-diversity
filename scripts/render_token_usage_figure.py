#!/usr/bin/env python3
"""Render category/subject x effort token usage figures from samples.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.model_registry import EFFORT_ORDER
from src.visualizer import EFFORT_COLORS, html_escape

TOKEN_METRICS = {
    "input_tokens": "Input Tokens",
    "output_tokens": "Output Tokens",
    "total_tokens": "Total Tokens",
}

BENCHMARK_LABELS = {
    "gpqa_diamond": "GPQA Diamond",
    "liveideabench": "LiveIdeaBench",
    "math_500": "MATH-500",
}


class RunningStat(BaseModel):
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    values: list[float] = Field(default_factory=list)

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.total_sq += value * value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)
        self.values.append(value)

    @property
    def mean(self) -> float:
        if self.n == 0:
            return 0.0
        return self.total / self.n

    @property
    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        variance = self.total_sq / self.n - self.mean * self.mean
        return math.sqrt(max(0.0, variance))

    def min_or_zero(self) -> float:
        return 0.0 if self.min_value is None else self.min_value

    def max_or_zero(self) -> float:
        return 0.0 if self.max_value is None else self.max_value

    def quantile(self, q: float) -> float:
        if not self.values:
            return 0.0
        if len(self.values) == 1:
            return self.values[0]
        sorted_values = sorted(self.values)
        position = (len(sorted_values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        fraction = position - lower
        return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate samples.jsonl token usage by category/subject x effort and render "
            "a grouped bar SVG with quantile or standard-deviation error bars."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing samples.jsonl.",
    )
    parser.add_argument(
        "--group-field",
        choices=["category", "subject"],
        default=None,
        help="Field used as the category axis. Defaults to category if present, otherwise subject.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TOKEN_METRICS),
        default="total_tokens",
        help="Token metric to render on the y-axis.",
    )
    parser.add_argument(
        "--benchmark-label",
        default=None,
        help="Optional benchmark label for the figure title.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to <run-dir>/summary_by_category_effort_token_usage.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional SVG output path. Defaults to <run-dir>/category_<metric>_usage_by_effort.svg.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom figure title.",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Optional custom figure subtitle.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed y-axis maximum. Defaults to an automatic nice upper bound.",
    )
    parser.add_argument(
        "--center-stat",
        choices=["mean", "median"],
        default="median",
        help="Bar height statistic.",
    )
    parser.add_argument(
        "--interval",
        choices=["iqr", "p10-p90", "std"],
        default="iqr",
        help="Error-bar interval. iqr uses p25-p75; p10-p90 uses p10-p90; std uses mean +/- std.",
    )
    return parser.parse_args()


def new_metric_stats() -> dict[str, RunningStat]:
    return {metric: RunningStat() for metric in TOKEN_METRICS}


def as_float(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def token_values(record: dict[str, Any]) -> dict[str, float | None]:
    input_tokens = as_float(record.get("input_tokens"))
    output_tokens = as_float(record.get("output_tokens"))
    total_tokens = as_float(record.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def infer_group_field(record: dict[str, Any]) -> str:
    if record.get("category") not in (None, ""):
        return "category"
    if record.get("subject") not in (None, ""):
        return "subject"
    raise ValueError("Could not infer group field: expected a non-empty category or subject field")


def benchmark_id(record: dict[str, Any], default_benchmark: str) -> str:
    value = record.get("benchmark")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_benchmark


def effort_sort_key(effort: str) -> tuple[int, str]:
    if effort in EFFORT_ORDER:
        return (EFFORT_ORDER.index(effort), effort)
    return (len(EFFORT_ORDER), effort)


def load_summary_rows(
    samples_path: Path,
    *,
    group_field: str | None,
    default_benchmark: str,
) -> tuple[list[dict[str, Any]], str]:
    grouped: dict[tuple[str, str, str], dict[str, RunningStat]] = {}
    resolved_group_field = group_field

    with samples_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {samples_path} line {line_number}: {exc}") from exc

            if resolved_group_field is None:
                resolved_group_field = infer_group_field(record)

            group_value = record.get(resolved_group_field)
            effort = record.get("effort")
            if group_value in (None, "") or effort in (None, ""):
                continue

            key = (benchmark_id(record, default_benchmark), str(group_value), str(effort))
            metric_stats = grouped.setdefault(key, new_metric_stats())
            for metric, value in token_values(record).items():
                if value is not None:
                    metric_stats[metric].add(value)

    if resolved_group_field is None:
        raise RuntimeError(f"No usable rows found in {samples_path}")

    rows: list[dict[str, Any]] = []
    for benchmark, category, effort in sorted(
        grouped,
        key=lambda item: (item[0], item[1], effort_sort_key(item[2])),
    ):
        stats = grouped[(benchmark, category, effort)]
        sample_count = max((stat.n for stat in stats.values()), default=0)
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "category": category,
            "source_group_field": resolved_group_field,
            "effort": effort,
            "sample_count": sample_count,
        }
        for metric in TOKEN_METRICS:
            stat = stats[metric]
            prefix = metric.removesuffix("_tokens")
            row[f"{prefix}_token_sample_count"] = stat.n
            row[f"mean_{metric}"] = stat.mean
            row[f"std_{metric}"] = stat.std
            row[f"min_{metric}"] = stat.min_or_zero()
            row[f"p10_{metric}"] = stat.quantile(0.10)
            row[f"p25_{metric}"] = stat.quantile(0.25)
            row[f"median_{metric}"] = stat.quantile(0.50)
            row[f"p75_{metric}"] = stat.quantile(0.75)
            row[f"p90_{metric}"] = stat.quantile(0.90)
            row[f"max_{metric}"] = stat.max_or_zero()
        rows.append(row)
    return rows, resolved_group_field


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nice_upper_bound(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    magnitude = 10**exponent
    normalized = value / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * magnitude)


def format_tick(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def default_benchmark_label(rows: list[dict[str, Any]], fallback: str) -> str:
    benchmarks = sorted({str(row["benchmark"]) for row in rows})
    if len(benchmarks) == 1:
        return BENCHMARK_LABELS.get(benchmarks[0], benchmarks[0])
    return fallback


def render_token_usage_svg(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    metric: str,
    group_field: str,
    title: str,
    subtitle: str,
    y_max: float | None = None,
    center_stat: str = "median",
    interval: str = "iqr",
) -> None:
    categories = sorted({str(row["category"]) for row in rows})
    rows_by_key = {(str(row["category"]), str(row["effort"])): row for row in rows}

    bars_per_group = len(EFFORT_ORDER)
    bar_width = 16
    intra_gap = 4
    group_gap = 18
    group_width = bars_per_group * bar_width + (bars_per_group - 1) * intra_gap

    width = max(1600, 160 + len(categories) * (group_width + group_gap) + 180)
    height = 860
    margin_left = 100
    margin_right = 220
    margin_top = 78
    margin_bottom = 190
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    center_column = f"{center_stat}_{metric}"
    mean_column = f"mean_{metric}"
    std_column = f"std_{metric}"
    count_column = f"{metric.removesuffix('_tokens')}_token_sample_count"

    def interval_bounds(row: dict[str, Any]) -> tuple[float, float]:
        if interval == "std":
            mean_value = float(row[mean_column])
            std_value = float(row[std_column])
            return max(0.0, mean_value - std_value), mean_value + std_value
        if interval == "p10-p90":
            return float(row[f"p10_{metric}"]), float(row[f"p90_{metric}"])
        return float(row[f"p25_{metric}"]), float(row[f"p75_{metric}"])

    upper_value = max(
        (
            max(float(row[center_column]), interval_bounds(row)[1])
            for row in rows
            if int(row[count_column]) > 0
        ),
        default=1.0,
    )
    y_max = nice_upper_bound(upper_value) if y_max is None else y_max
    if y_max <= 0:
        raise ValueError("--y-max must be positive")

    def scale_y(value: float) -> float:
        clipped = max(0.0, min(y_max, value))
        return margin_top + (1.0 - clipped / y_max) * plot_height

    group_spacing = plot_width / max(1, len(categories))
    category_starts = {
        category: margin_left + group_spacing * index + (group_spacing - group_width) / 2
        for index, category in enumerate(categories)
    }
    baseline = height - margin_bottom

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="36" font-size="24" font-family="monospace">{html_escape(title)}</text>',
        f'<text x="{margin_left}" y="58" font-size="14" font-family="monospace">{html_escape(subtitle)}</text>',
    ]

    for index in range(6):
        tick = y_max * index / 5
        y = scale_y(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" font-size="13" text-anchor="end" font-family="monospace">{format_tick(tick)}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{baseline}" stroke="#333" stroke-width="1.4"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{baseline}" x2="{width - margin_right}" y2="{baseline}" stroke="#333" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="30" y="{height / 2:.1f}" font-size="14" text-anchor="middle" transform="rotate(-90 30 {height / 2:.1f})" font-family="monospace">{html_escape(TOKEN_METRICS[metric])}</text>'
    )

    legend_x = width - 160
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
            if row is None or int(row[count_column]) == 0:
                continue
            x = start_x + effort_index * (bar_width + intra_gap)
            mean_value = float(row[mean_column])
            std_value = float(row[std_column])
            center_value = float(row[center_column])
            ymin, ymax = interval_bounds(row)
            y = scale_y(center_value)
            color = EFFORT_COLORS[effort]
            bar_height = baseline - y
            sample_count = int(row[count_column])

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
                        f"{category} | {effort} | {center_stat}={center_value:.1f} | "
                        f"{interval}=({ymin:.1f}, {ymax:.1f}) | mean={mean_value:.1f} | "
                        f"std={std_value:.1f} | n={sample_count}"
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

    axis_label = "Category" if group_field == "category" else "Subject"
    parts.append(
        f'<text x="{(margin_left + width - margin_right) / 2:.2f}" y="{height - 18}" font-size="14" text-anchor="middle" font-family="monospace">{axis_label} x Effort</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    default_benchmark = (
        (args.benchmark_label or "liveideabench").lower().replace("-", "_").replace(" ", "_")
    )
    rows, group_field = load_summary_rows(
        samples_path,
        group_field=args.group_field,
        default_benchmark=default_benchmark,
    )
    if not rows:
        raise RuntimeError(f"No token usage rows found in {samples_path}")

    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else run_dir / "summary_by_category_effort_token_usage.csv"
    )
    metric_slug = args.metric.removesuffix("_tokens")
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else run_dir / f"category_{metric_slug}_token_usage_by_effort.svg"
    )

    benchmark_label = args.benchmark_label or default_benchmark_label(rows, run_dir.name)
    axis_label = "Category" if group_field == "category" else "Subject"
    title = args.title or f"{benchmark_label} {axis_label} x Effort {TOKEN_METRICS[args.metric]}"
    subtitle = args.subtitle or (
        f"Bars: {args.center_stat} {args.metric} per sample | error bars: {args.interval} | source: samples.jsonl"
    )

    write_csv(summary_csv, rows)
    render_token_usage_svg(
        rows,
        output_path,
        metric=args.metric,
        group_field=group_field,
        title=title,
        subtitle=subtitle,
        y_max=args.y_max,
        center_stat=args.center_stat,
        interval=args.interval,
    )
    print(json.dumps({"summary_csv": str(summary_csv), "svg": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
