#!/usr/bin/env python3
"""Build a stratified keyword subset from LiveIdeaBench classifications."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified keyword subset with a fixed count per category.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV path (e.g. external/liveideabench/csvs/keyword_classifications.csv).",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        required=True,
        help="Number of keywords to sample from each category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8", newline="")))
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    rng = random.Random(args.seed)
    sampled_rows: list[dict[str, str]] = []

    for category in sorted(by_category):
        candidates = by_category[category]
        if len(candidates) < args.per_category:
            raise ValueError(
                f"Category '{category}' has only {len(candidates)} rows; "
                f"cannot sample {args.per_category}."
            )
        chosen = rng.sample(candidates, args.per_category)
        sampled_rows.extend(
            sorted(
                chosen,
                key=lambda row: (row["keyword"], -float(row["similarity_score"])),
            )
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["keyword", "category", "similarity_score"],
        )
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(output)
    print(f"n_categories={len(by_category)}")
    print(f"n_keywords={len(sampled_rows)}")


if __name__ == "__main__":
    main()
