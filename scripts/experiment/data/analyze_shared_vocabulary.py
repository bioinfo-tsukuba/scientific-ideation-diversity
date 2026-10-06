#!/usr/bin/env python3
"""Identify vocabulary shared across keywords per facet.

For each facet, we want to know:
 1. Which words appear in MOST keywords (universal scaffolding)?
 2. How much of each sample text is made of these shared words?
 3. How often do these universal words appear per sample?

This complements Jaccard (set-level) and IDF (frequency-weighted) by looking
at which specific words create the cross-keyword similarity picked up by
embedding models.

Usage:
    uv run python scripts/experiment/data/analyze_shared_vocabulary.py \
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FACETS = ["purpose", "mechanism", "evaluation"]

STOPWORDS = set(
    """
a an the and or of to for in on at by with as is are was were be been being
it its from this that these those such which who whom where when how why what
not no nor so also while whether than then there here both either neither
will can could would should may might must does did do has have had having
it's don't couldn't should've would've wouldn't doesn't hasn't isn't aren't
""".split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]*[A-Za-z]", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.run_dir / "samples.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    keywords = sorted(set(r["keyword"] for r in records))
    n_kw = len(keywords)
    print(f"Loaded {len(records)} records, {n_kw} keywords")

    # For each facet: {word: set of keywords it appears in}
    word_keywords: dict[str, dict[str, set]] = {f: defaultdict(set) for f in FACETS}
    # For each facet: list of (keyword, tokens) pairs
    sample_tokens: dict[str, list[tuple[str, list[str]]]] = {f: [] for f in FACETS}

    for r in records:
        for facet in FACETS:
            toks = tokenize(r["structured_output"][facet])
            sample_tokens[facet].append((r["keyword"], toks))
            for t in set(toks):
                word_keywords[facet][t].add(r["keyword"])

    # 1. Universal words: words appearing in >= 90% of keywords
    print("\n" + "=" * 72)
    print("  TOP UNIVERSAL WORDS per facet (appear in >= 90% of keywords)")
    print("=" * 72)

    threshold = int(0.9 * n_kw)
    csv_rows: list[dict] = []
    for facet in FACETS:
        universal = [(w, len(kws)) for w, kws in word_keywords[facet].items() if len(kws) >= threshold]
        # Sort by total frequency
        freq = Counter()
        for kw, toks in sample_tokens[facet]:
            for t in toks:
                freq[t] += 1
        universal.sort(key=lambda x: freq[x[0]], reverse=True)
        print(f"\n  [{facet}]  {len(universal)} words appear in >= {threshold}/{n_kw} keywords")
        for w, nkw in universal[: args.top_n]:
            print(f"    {w:<25s}  appears in {nkw:>2d}/{n_kw} keywords, total freq {freq[w]}")
            csv_rows.append({"facet": facet, "word": w, "n_keywords": nkw, "total_freq": freq[w]})

    # 2. What fraction of each sample's tokens are "universal" (appear in >=90% of keywords)?
    print("\n" + "=" * 72)
    print("  FRACTION OF SAMPLE TOKENS THAT ARE UNIVERSAL (>=90% of keywords)")
    print("=" * 72)
    for facet in FACETS:
        universal_set = {w for w, kws in word_keywords[facet].items() if len(kws) >= threshold}
        fractions: list[float] = []
        for kw, toks in sample_tokens[facet]:
            if not toks:
                continue
            n_universal = sum(1 for t in toks if t in universal_set)
            fractions.append(n_universal / len(toks))
        mean_frac = sum(fractions) / len(fractions) if fractions else 0.0
        print(f"  {facet:<12s} mean universal-word fraction per sample = {mean_frac:.2%}")

    # 3. Document frequency distribution
    print("\n" + "=" * 72)
    print("  DOCUMENT FREQUENCY DISTRIBUTION (how many keywords each word appears in)")
    print("=" * 72)
    for facet in FACETS:
        df_counts = Counter(len(kws) for kws in word_keywords[facet].values())
        total = sum(df_counts.values())
        print(f"\n  [{facet}] (n_unique_words = {total})")
        print(f"    appears in only 1 keyword:        {df_counts[1]:>5d}  ({df_counts[1] / total:.1%})")
        print(
            f"    appears in 2-5 keywords:          {sum(df_counts[i] for i in range(2, 6)):>5d}  "
            f"({sum(df_counts[i] for i in range(2, 6)) / total:.1%})"
        )
        print(
            f"    appears in 6-15 keywords:         {sum(df_counts[i] for i in range(6, 16)):>5d}  "
            f"({sum(df_counts[i] for i in range(6, 16)) / total:.1%})"
        )
        print(
            f"    appears in >=16 keywords:         {sum(df_counts[i] for i in range(16, n_kw + 1)):>5d}  "
            f"({sum(df_counts[i] for i in range(16, n_kw + 1)) / total:.1%})"
        )

    # Save
    csv_path = output_dir / "shared_vocabulary_universal.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["facet", "word", "n_keywords", "total_freq"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
