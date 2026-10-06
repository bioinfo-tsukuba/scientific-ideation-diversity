#!/usr/bin/env python3
"""Compute per-field semantic diversity from runner-generated embeddings.

Reads embeddings_{field}.npy and samples.jsonl from runner output
directories. No embedding API calls are made.

Usage:
    uv run python scripts/compute_facet_diversity.py \
        --input-dir results/effort_diversity/my_run \
        --output-dir results/effort_diversity/my_run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from src.artifacts import load_sample_records
from src.model_registry import EffortName
from src.schemas.diversity import DiversitySummaryRow, PerKeywordDiversityRow
from src.schemas.sample import SampleRecord

EFFORT_ORDER = [e.value for e in EffortName]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-field diversity from runner-generated embeddings.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Runner output directory containing samples.jsonl and embeddings_{field}.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV files. Defaults to --input-dir.",
    )
    return parser.parse_args()


def build_index(records: list[SampleRecord]) -> tuple[list[EffortName], list[str], list[str]]:
    """Extract efforts, keywords, and field names from records."""
    efforts = sorted(
        set(r.effort for r in records),
        key=lambda e: EFFORT_ORDER.index(e.value) if e.value in EFFORT_ORDER else len(EFFORT_ORDER),
    )
    keywords = sorted(set(r.keyword for r in records))
    field_names = list(type(records[0].structured_output).model_fields.keys())
    return efforts, keywords, field_names


def compute_diversity_table(
    embeddings: np.ndarray,
    records: list[SampleRecord],
    field: str,
    efforts: list[EffortName],
    keywords: list[str],
) -> list[DiversitySummaryRow]:
    rows: list[DiversitySummaryRow] = []
    for effort in efforts:
        kw_divs: list[float] = []
        for kw in keywords:
            indices = [
                i for i, r in enumerate(records)
                if r.effort == effort and r.keyword == kw
            ]
            kw_divs.append(float(np.mean(pdist(embeddings[indices], "cosine"))))
        rows.append(DiversitySummaryRow(
            effort=effort,
            field=field,
            mean_pairwise_cosine_distance=float(np.mean(kw_divs)),
            std=float(np.std(kw_divs)),
        ))
    return rows


def compute_per_keyword_table(
    embeddings: np.ndarray,
    records: list[SampleRecord],
    field: str,
    efforts: list[EffortName],
    keywords: list[str],
) -> list[PerKeywordDiversityRow]:
    rows: list[PerKeywordDiversityRow] = []
    for kw in keywords:
        effort_values: dict[str, float] = {}
        for effort in efforts:
            indices = [
                i for i, r in enumerate(records)
                if r.effort == effort and r.keyword == kw
            ]
            effort_values[effort.value] = float(np.mean(pdist(embeddings[indices], "cosine")))
        rows.append(PerKeywordDiversityRow(keyword=kw, field=field, **effort_values))
    return rows


def write_csv_rows(path: Path, rows: list[DiversitySummaryRow | PerKeywordDiversityRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [row.model_dump() for row in rows]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    samples_path = input_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path}")

    records = load_sample_records(samples_path)
    efforts, keywords, field_names = build_index(records)

    print(f"records={len(records)} keywords={len(keywords)} efforts={efforts} fields={field_names}")

    # Validate: every keyword x effort must have the same sample count
    expected_per_group = None
    for effort in efforts:
        for kw in keywords:
            count = sum(1 for r in records if r.effort == effort and r.keyword == kw)
            if expected_per_group is None:
                expected_per_group = count
            if count != expected_per_group:
                raise ValueError(
                    f"Inconsistent sample count: keyword={kw!r} effort={effort.value} "
                    f"has {count} samples, expected {expected_per_group}"
                )

    all_summary_rows: list[DiversitySummaryRow] = []
    all_per_kw_rows: list[PerKeywordDiversityRow] = []

    for field in field_names:
        embeddings_path = input_dir / f"embeddings_{field}.npy"
        if not embeddings_path.exists():
            print(f"[{field}] embeddings not found at {embeddings_path}, skipping")
            continue

        embeddings = np.load(embeddings_path, mmap_mode="r")
        print(f"[{field}] loaded {embeddings_path} shape={embeddings.shape}")

        summary_rows = compute_diversity_table(embeddings, records, field, efforts, keywords)
        per_kw_rows = compute_per_keyword_table(embeddings, records, field, efforts, keywords)

        all_summary_rows.extend(summary_rows)
        all_per_kw_rows.extend(per_kw_rows)

        for row in summary_rows:
            print(f"  {row.effort.value:7s} {row.field:12s} {row.mean_pairwise_cosine_distance:.4f}±{row.std:.4f}")

    summary_path = output_dir / "facet_diversity.csv"
    per_kw_path = output_dir / "facet_diversity_per_keyword.csv"
    write_csv_rows(summary_path, all_summary_rows)
    write_csv_rows(per_kw_path, all_per_kw_rows)

    print(f"\n-> {summary_path}")
    print(f"-> {per_kw_path}")


if __name__ == "__main__":
    main()
