#!/usr/bin/env python3
"""Analyze vocabulary specificity across facets.

Hypothesis: mechanism/evaluation texts use more generic (keyword-agnostic) vocabulary
than purpose, which hurts embedding-based diversity metrics for these facets.

Metrics computed per (model, facet):
  1. Keyword-phrase coverage: fraction of samples whose facet text contains the keyword
  2. Cross-keyword vocabulary Jaccard: mean pairwise Jaccard of unique vocabulary
     (lower = more keyword-specific vocabulary)
  3. Mean IDF: average IDF of content tokens (lower = tokens appear across many keywords)
  4. Top TF-IDF terms per keyword (qualitative inspection)
  5. Side-by-side qualitative examples

Usage:
    uv run python scripts/experiment/data/analyze_facet_vocabulary.py \
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FACETS = ["purpose", "mechanism", "evaluation"]

# Simple English stopwords + common scientific filler words
STOPWORDS = set(
    """
a an the and or of to for in on at by with as is are was were be been being this that these those
it its from our we propose proposes proposed proposing approach approaches method methods
using used use uses via through will can could would should may might
has have had having which whose who whom when where how why what
not no nor so such also while whether than then there here
paper framework work study research
""".split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="top-K TF-IDF terms per keyword to show")
    parser.add_argument(
        "--sample-keywords",
        type=str,
        nargs="*",
        default=["natural selection", "tornadoes", "computational social science"],
        help="Keywords to include in qualitative examples",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]*[A-Za-z]|[A-Za-z]", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def load_records(run_dir: Path) -> list[dict]:
    records = []
    with open(run_dir / "samples.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_keyword_coverage(records: list[dict]) -> dict[str, float]:
    """Fraction of samples where the keyword phrase appears literally in the facet text."""
    result = {}
    for facet in FACETS:
        hits = 0
        for r in records:
            kw = r["keyword"].lower()
            text = r["structured_output"][facet].lower()
            if kw in text:
                hits += 1
        result[facet] = hits / len(records) if records else 0.0
    return result


def compute_jaccard_per_facet(records: list[dict]) -> dict[str, float]:
    """Cross-keyword vocabulary Jaccard, mean over all keyword pairs."""
    vocab: dict[str, dict[str, set]] = {f: defaultdict(set) for f in FACETS}
    for r in records:
        kw = r["keyword"]
        for facet in FACETS:
            tokens = set(tokenize(r["structured_output"][facet]))
            vocab[facet][kw].update(tokens)

    results = {}
    for facet in FACETS:
        keywords = sorted(vocab[facet].keys())
        jaccards = []
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                a, b = vocab[facet][keywords[i]], vocab[facet][keywords[j]]
                union = a | b
                if not union:
                    continue
                jaccards.append(len(a & b) / len(union))
        results[facet] = float(np.mean(jaccards)) if jaccards else 0.0
    return results


def compute_idf_and_top_tfidf(
    records: list[dict], top_k: int = 10
) -> tuple[dict[str, float], dict[str, dict[str, list[tuple[str, float]]]]]:
    """
    Build a per-facet corpus where each document = all concatenated text for one keyword.
    Compute IDF across keywords (so low IDF = token appears in many keywords).

    Returns:
      mean_idf_per_facet: average IDF of content tokens across all sample texts
      top_tfidf: {facet: {keyword: [(term, tfidf), ...]}}
    """
    keyword_docs: dict[str, dict[str, list[str]]] = {f: defaultdict(list) for f in FACETS}
    for r in records:
        for facet in FACETS:
            keyword_docs[facet][r["keyword"]].extend(tokenize(r["structured_output"][facet]))

    mean_idf_per_facet: dict[str, float] = {}
    top_tfidf: dict[str, dict[str, list[tuple[str, float]]]] = {f: {} for f in FACETS}

    for facet in FACETS:
        keywords = sorted(keyword_docs[facet].keys())
        n_docs = len(keywords)

        # document frequency: number of keywords in which each token appears
        df: Counter = Counter()
        for kw in keywords:
            for token in set(keyword_docs[facet][kw]):
                df[token] += 1

        idf: dict[str, float] = {token: math.log(n_docs / count) for token, count in df.items()}

        # mean IDF weighted by token frequency across all samples
        total_tokens = 0
        idf_sum = 0.0
        for kw in keywords:
            for token in keyword_docs[facet][kw]:
                idf_sum += idf.get(token, 0.0)
                total_tokens += 1
        mean_idf_per_facet[facet] = idf_sum / total_tokens if total_tokens else 0.0

        # top TF-IDF per keyword
        for kw in keywords:
            tf = Counter(keyword_docs[facet][kw])
            total = sum(tf.values())
            if total == 0:
                top_tfidf[facet][kw] = []
                continue
            scores = [(term, (count / total) * idf.get(term, 0.0)) for term, count in tf.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            top_tfidf[facet][kw] = scores[:top_k]

    return mean_idf_per_facet, top_tfidf


def write_qualitative_examples(
    records: list[dict], keywords: list[str], effort: str, path: Path, n_samples: int = 2
) -> None:
    lines = []
    for kw in keywords:
        matches = [r for r in records if r["keyword"] == kw and r["effort"] == effort][:n_samples]
        lines.append(f"\n{'=' * 80}")
        lines.append(f"keyword: {kw}   (effort={effort})")
        lines.append(f"{'=' * 80}")
        for i, r in enumerate(matches):
            lines.append(f"\n[sample {i}]")
            for facet in FACETS:
                text = r["structured_output"][facet]
                lines.append(f"  [{facet}]\n    {text}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(run_dir)
    print(f"Loaded {len(records)} records from {run_dir}")

    # 1. Keyword coverage
    coverage = compute_keyword_coverage(records)
    print("\n=== Keyword-phrase coverage (keyword string appears in facet text) ===")
    for facet in FACETS:
        print(f"  {facet:<12s} {coverage[facet]:.2%}")

    # 2. Jaccard
    jaccards = compute_jaccard_per_facet(records)
    print("\n=== Cross-keyword vocabulary Jaccard (lower = more keyword-specific) ===")
    for facet in FACETS:
        print(f"  {facet:<12s} {jaccards[facet]:.4f}")

    # 3. Mean IDF
    mean_idf, top_tfidf = compute_idf_and_top_tfidf(records, top_k=args.top_k)
    print("\n=== Mean IDF of content tokens (lower = more generic, keyword-agnostic) ===")
    for facet in FACETS:
        print(f"  {facet:<12s} {mean_idf[facet]:.4f}")

    # 4. Top TF-IDF per keyword per facet (for sample keywords)
    sample_kws = [kw for kw in args.sample_keywords if kw in top_tfidf["purpose"]]
    if not sample_kws:
        sample_kws = sorted(top_tfidf["purpose"].keys())[:3]

    print(f"\n=== Top-{args.top_k} TF-IDF terms per keyword per facet ===")
    for kw in sample_kws:
        print(f"\n  [{kw}]")
        for facet in FACETS:
            terms = top_tfidf[facet][kw]
            term_str = ", ".join(f"{t}({s:.3f})" for t, s in terms)
            print(f"    {facet:<12s} {term_str}")

    # 5. Qualitative examples
    qual_path = output_dir / "facet_qualitative_examples.txt"
    write_qualitative_examples(records, sample_kws, effort="medium", path=qual_path)
    print(f"\n-> Qualitative examples saved to {qual_path}")

    # 6. Save summary CSV
    csv_path = output_dir / "facet_vocabulary_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["facet", "keyword_coverage", "vocab_jaccard", "mean_idf"])
        for facet in FACETS:
            writer.writerow([facet, coverage[facet], jaccards[facet], mean_idf[facet]])
    print(f"-> Summary CSV saved to {csv_path}")

    # 7. Save top-TF-IDF CSV
    tfidf_path = output_dir / "facet_top_tfidf_terms.csv"
    with open(tfidf_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["facet", "keyword", "rank", "term", "tfidf_score"])
        for facet in FACETS:
            for kw, terms in top_tfidf[facet].items():
                for rank, (term, score) in enumerate(terms, 1):
                    writer.writerow([facet, kw, rank, term, f"{score:.6f}"])
    print(f"-> Top-TF-IDF CSV saved to {tfidf_path}")


if __name__ == "__main__":
    main()
