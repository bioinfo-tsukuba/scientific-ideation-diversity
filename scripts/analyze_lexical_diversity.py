#!/usr/bin/env python3
"""Compute lexical diversity metrics (Distinct-N, Self-BLEU) per (keyword, effort, facet).

Two n-gram-based metrics to cross-check embedding-based diversity:
  - Distinct-N = |unique n-grams in corpus| / |total n-grams|
                 (corpus-level vocabulary richness, length-normalized)
  - Self-BLEU-n = average of sentence-level BLEU scores where each sample is the
                  candidate and the remaining samples are the references, using
                  the standard BLEU formulation (geometric mean of clipped
                  precisions p_1..p_n with uniform weights, brevity penalty,
                  Chen & Cherry 2014 method-1 smoothing).  Matches Zhu et al.
                  2018 (Texygen) and subsequent diversity-benchmark papers.

Both computed for n = 1..6 to expose saturation behavior.

Usage:
    uv run python scripts/analyze_lexical_diversity.py \
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

FACETS = ["purpose", "mechanism", "evaluation"]
COMBINED_FACET = "combined"
# OUTPUT_FACETS controls which facet rows appear in the aggregate / per-keyword
# CSVs.  ``combined`` concatenates the three facet strings (in FACETS order,
# joined by single spaces) before tokenisation, mirroring the combined-text
# input fed to the embedding models.
OUTPUT_FACETS = FACETS + [COMBINED_FACET]
NGRAM_SIZES = [1, 2, 3, 4, 5, 6]
EFFORT_ORDER = ["none", "low", "medium", "high"]
EFFORT_COLORS = {"none": "#90CAF9", "low": "#42A5F5", "medium": "#1E88E5", "high": "#0D47A1"}

# Mathematical lower bound: Self-BLEU's leave-one-out (candidate = one sample,
# references = the remaining N-1 samples) is undefined for N < 2.  This
# constant is an internal invariant — the only value at which the algorithm
# itself is defined — and should not change.  Experiment-design thresholds
# (e.g. "Phase B expects 30 samples per bucket") belong in the
# ``--min-samples-per-bucket`` CLI argument instead.
MIN_SAMPLES_PER_BUCKET: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-samples-per-bucket",
        type=int,
        default=MIN_SAMPLES_PER_BUCKET,
        help=(
            "Abort if any (keyword, effort, facet) bucket has fewer than this "
            f"many samples.  Defaults to {MIN_SAMPLES_PER_BUCKET} (the mathematical "
            "minimum for Self-BLEU's leave-one-out).  Pass the expected design "
            "size (e.g. 30 for Phase B, 50 for Phase A) to additionally catch "
            "generation runs that did not reach their target sample count."
        ),
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word boundaries, keep all non-empty tokens."""
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(sample_tokens_list: list[list[str]], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for tokens in sample_tokens_list:
        all_ngrams.extend(get_ngrams(tokens, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# Chen & Cherry (2014) method-1 smoothing (NLTK implementation).  When a
# specific n-gram precision p_n has 0 clipped matches (common at high n on
# short candidates) it is replaced with ``epsilon / denominator`` (NLTK uses
# ``epsilon = 0.1``) so the geometric mean can still be computed; non-zero
# p_n values are left untouched, so well-populated cases behave exactly
# like unsmoothed BLEU.
_BLEU_SMOOTHING = SmoothingFunction().method1


def self_bleu_n(sample_tokens_list: list[list[str]], n: int) -> float:
    """Standard Self-BLEU-n (Zhu et al. 2018): mean over leave-one-out BLEU.

    For each sample, computes sentence-level BLEU against the remaining samples
    as references using uniform weights over orders 1..n (so Self-BLEU-4 =
    geometric mean of p_1..p_4 with brevity penalty, matching Texygen).  Higher
    = more overlap = less lexical diversity.

    Caller must ensure ``len(sample_tokens_list) >= MIN_SAMPLES_PER_BUCKET``;
    :func:`collect_samples_by_group` enforces this up front, so an undersized
    bucket reaching this function is treated as a contract violation and
    surfaced rather than silently returning 0.0.
    """
    total = len(sample_tokens_list)
    if total < MIN_SAMPLES_PER_BUCKET:
        raise RuntimeError(
            f"self_bleu_n requires at least {MIN_SAMPLES_PER_BUCKET} samples "
            f"(got {total}); caller should have validated bucket size upstream."
        )
    weights = tuple([1.0 / n] * n)
    scores = [
        sentence_bleu(
            [sample_tokens_list[j] for j in range(total) if j != i],
            sample_tokens_list[i],
            weights=weights,
            smoothing_function=_BLEU_SMOOTHING,
        )
        for i in range(total)
    ]
    return float(np.mean(scores))


def _compute_bucket_metrics(
    key: tuple[str, str, str],
    sample_tokens_list: list[list[str]],
) -> dict:
    """Worker: compute Distinct-N and Self-BLEU-n for one bucket.

    Runs in a :class:`ProcessPoolExecutor` worker.  Returned dict becomes one
    row of ``lexical_diversity_per_keyword.csv``.
    """
    kw, eff, facet = key
    row = {"keyword": kw, "effort": eff, "facet": facet, "n_samples": len(sample_tokens_list)}
    for n in NGRAM_SIZES:
        row[f"distinct_{n}"] = distinct_n(sample_tokens_list, n)
        row[f"self_bleu_{n}"] = self_bleu_n(sample_tokens_list, n)
    return row


def collect_samples_by_group(records: list[dict], *, min_samples: int) -> dict:
    """Return {(keyword, effort, facet): [tokens_per_sample]}.

    Aborts if any bucket has fewer than ``min_samples`` samples.  The effective
    floor is ``max(MIN_SAMPLES_PER_BUCKET, min_samples)``: the former is the
    mathematical invariant (Self-BLEU's leave-one-out requires at least one
    reference per candidate), the latter is the experiment-design expectation
    supplied by the caller (e.g. Phase B's 30-sample target).
    """
    if min_samples < MIN_SAMPLES_PER_BUCKET:
        raise ValueError(
            f"min_samples={min_samples} is below the mathematical minimum "
            f"({MIN_SAMPLES_PER_BUCKET}); Self-BLEU is undefined in that regime."
        )

    groups: dict = defaultdict(list)
    for r in records:
        kw, eff = r["keyword"], r["effort"]
        for facet in FACETS:
            tokens = tokenize(r["structured_output"][facet])
            groups[(kw, eff, facet)].append(tokens)
        combined_text = " ".join(r["structured_output"][f] for f in FACETS)
        groups[(kw, eff, COMBINED_FACET)].append(tokenize(combined_text))

    undersized = [(key, len(vals)) for key, vals in groups.items()
                  if len(vals) < min_samples]
    if undersized:
        lines = "\n".join(f"  {k}: n={n}" for k, n in sorted(undersized))
        raise RuntimeError(
            f"{len(undersized)} (keyword, effort, facet) bucket(s) have "
            f"fewer than {min_samples} samples:\n{lines}\n"
            "Re-run generation, lower --min-samples-per-bucket, or filter "
            "these records before analysis."
        )
    return groups


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in open(args.run_dir / "samples.jsonl")]
    print(f"Loaded {len(records)} records from {args.run_dir}")

    groups = collect_samples_by_group(records, min_samples=args.min_samples_per_bucket)
    keywords = sorted({kw for kw, _, _ in groups.keys()})
    efforts = [e for e in EFFORT_ORDER if any(e == eff for _, eff, _ in groups.keys())]
    print(f"  {len(keywords)} keywords, efforts={efforts}, facets={OUTPUT_FACETS}")

    # Per (keyword, effort, facet) metrics — parallelise across buckets.
    # Self-BLEU is O(N^2) per bucket and dominates runtime, but each bucket is
    # independent so a process pool scales nearly linearly with core count.
    # Empirically: 1180-keyword Phase B.1 runs drop from ~30+ min single-thread
    # to ~2 min with ~10 workers on an M-series Mac.
    bucket_items = list(groups.items())
    workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"  computing metrics for {len(bucket_items)} buckets with {workers} workers...")
    per_kw_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_compute_bucket_metrics, key, tokens)
                   for key, tokens in bucket_items]
        completed = 0
        for fut in as_completed(futures):
            per_kw_rows.append(fut.result())
            completed += 1
            if completed % 500 == 0 or completed == len(bucket_items):
                print(f"  bucket {completed}/{len(bucket_items)}", flush=True)

    # Aggregate: mean over keywords per (effort, facet)
    print("\n" + "=" * 90)
    print("  AGGREGATE per (effort, facet) — mean over keywords")
    print("=" * 90)
    agg_rows: list[dict] = []
    for facet in OUTPUT_FACETS:
        for eff in efforts:
            facet_eff_rows = [r for r in per_kw_rows if r["effort"] == eff and r["facet"] == facet]
            agg = {"effort": eff, "facet": facet, "n_keywords": len(facet_eff_rows)}
            for n in NGRAM_SIZES:
                agg[f"distinct_{n}_mean"] = float(np.mean([r[f"distinct_{n}"] for r in facet_eff_rows]))
                agg[f"distinct_{n}_std"] = float(np.std([r[f"distinct_{n}"] for r in facet_eff_rows]))
                agg[f"self_bleu_{n}_mean"] = float(np.mean([r[f"self_bleu_{n}"] for r in facet_eff_rows]))
                agg[f"self_bleu_{n}_std"] = float(np.std([r[f"self_bleu_{n}"] for r in facet_eff_rows]))
            agg_rows.append(agg)

    # Print compact summary for n=2 and n=4
    for metric_prefix in ["distinct", "self_bleu"]:
        print(f"\n  {metric_prefix.upper()} (n=2 and n=4, aggregate)")
        print(f"  {'facet':<12s} {'effort':<8s} {'n=2':>10s} {'n=4':>10s}")
        for row in agg_rows:
            print(
                f"  {row['facet']:<12s} {row['effort']:<8s} "
                f"{row[f'{metric_prefix}_2_mean']:>10.4f} {row[f'{metric_prefix}_4_mean']:>10.4f}"
            )

    # Save CSVs
    per_kw_path = output_dir / "lexical_diversity_per_keyword.csv"
    with open(per_kw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_kw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_kw_rows)
    agg_path = output_dir / "lexical_diversity_aggregate.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\n-> {per_kw_path}")
    print(f"-> {agg_path}")

    # Plots
    # Plot A: metric vs n-gram size, one line per effort, one subplot per facet
    for metric_prefix, ylabel, higher_is_more_diverse in [
        ("distinct", "Distinct-N", True),
        ("self_bleu", "Self-BLEU (n-gram precision)", False),
    ]:
        fig, axes = plt.subplots(1, len(FACETS), figsize=(5 * len(FACETS), 4), sharey=True)
        for ax_i, facet in enumerate(FACETS):
            ax = axes[ax_i]
            for eff in efforts:
                ys = []
                for n in NGRAM_SIZES:
                    row = next(r for r in agg_rows if r["effort"] == eff and r["facet"] == facet)
                    ys.append(row[f"{metric_prefix}_{n}_mean"])
                ax.plot(NGRAM_SIZES, ys, marker="o", color=EFFORT_COLORS[eff], label=eff, linewidth=2)
            ax.set_title(f"{facet}")
            ax.set_xlabel("n-gram size")
            if ax_i == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if ax_i == 0:
                ax.legend(title="effort")
        direction = "↑ = more diverse" if higher_is_more_diverse else "↓ = more diverse"
        fig.suptitle(f"{ylabel} by n-gram size  ({direction})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fname = f"lexical_diversity_{metric_prefix}_vs_n.png"
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"-> {output_dir / fname}")

    # Plot B: effort trend for each n, one subplot per facet per metric
    for metric_prefix, ylabel, higher_is_more_diverse in [
        ("distinct", "Distinct-N", True),
        ("self_bleu", "Self-BLEU", False),
    ]:
        fig, axes = plt.subplots(1, len(FACETS), figsize=(5 * len(FACETS), 4), sharey=True)
        n_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(NGRAM_SIZES)))
        for ax_i, facet in enumerate(FACETS):
            ax = axes[ax_i]
            for n, color in zip(NGRAM_SIZES, n_colors):
                ys = []
                for eff in efforts:
                    row = next(r for r in agg_rows if r["effort"] == eff and r["facet"] == facet)
                    ys.append(row[f"{metric_prefix}_{n}_mean"])
                ax.plot(range(len(efforts)), ys, marker="o", color=color, label=f"n={n}", linewidth=2)
            ax.set_xticks(range(len(efforts)))
            ax.set_xticklabels(efforts)
            ax.set_title(f"{facet}")
            ax.set_xlabel("effort")
            if ax_i == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if ax_i == 0:
                ax.legend(title="n-gram", fontsize=8)
        direction = "↑ = more diverse" if higher_is_more_diverse else "↓ = more diverse"
        fig.suptitle(f"{ylabel} vs effort  ({direction})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fname = f"lexical_diversity_{metric_prefix}_vs_effort.png"
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"-> {output_dir / fname}")


if __name__ == "__main__":
    main()
