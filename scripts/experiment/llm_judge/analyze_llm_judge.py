#!/usr/bin/env python3
"""Analyze LLM-judge pairwise similarity results.

Reads ``llm_judge_{judge}/pairwise.jsonl`` produced by
``scripts/experiment/llm_judge/llm_judge_pairwise.py`` and computes:

  1. ABCD distribution per (stratum, effort)
  2. Rank-level saturation (how ABCD varies with rank within a stratum)
  3. Position-bias analysis: AB vs BA consistency + quadratic-weighted
     Cohen's kappa
  4. Effort effect size (Cohen's d) and significance (Mann-Whitney U) on
     "fraction of A pairs" in the top stratum per keyword
  5. Per-keyword heterogeneity in the effort effect
  6. Embedding-distance vs ABCD label correspondence

Usage:
    uv run python scripts/experiment/analyze_llm_judge.py \\
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42 \\
        --judge-model gpt-4.1
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

from src.model_registry import EffortName, JudgeModelName
from src.schemas.llm_judge import JudgeAnswer, JudgeOrder, JudgeResponse, PairStratum

LABELS = [member.value for member in JudgeAnswer]  # ['A', 'B', 'C', 'D']
EFFORT_ORDER = [member.value for member in EffortName]
STRATUM_ORDER = [member.value for member in PairStratum]
EFFORT_COLORS = {"none": "#90CAF9", "low": "#42A5F5", "medium": "#1E88E5", "high": "#0D47A1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.GPT_4_1,
        choices=list(JudgeModelName),
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --run-dir")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def load_responses(path: Path) -> list[JudgeResponse]:
    rows: list[JudgeResponse] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(JudgeResponse.model_validate_json(line))
    return rows


# --- Section 1: ABCD distribution per (stratum, effort) --------------------


def compute_stratum_effort_distribution(rows: list[JudgeResponse]) -> list[dict]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        key = (row.request.pair.stratum.value, row.request.pair.effort.value)
        buckets[key].append(row.answer.value)

    summary: list[dict] = []
    for stratum in STRATUM_ORDER:
        for effort in EFFORT_ORDER:
            labels = buckets.get((stratum, effort), [])
            if not labels:
                continue
            counts = Counter(labels)
            n = len(labels)
            row: dict = {"stratum": stratum, "effort": effort, "n": n}
            for label in LABELS:
                row[f"count_{label}"] = counts.get(label, 0)
                row[f"frac_{label}"] = counts.get(label, 0) / n if n else 0.0
            row["frac_A_or_B"] = (counts.get("A", 0) + counts.get("B", 0)) / n if n else 0.0
            row["frac_C_or_D"] = (counts.get("C", 0) + counts.get("D", 0)) / n if n else 0.0
            summary.append(row)
    return summary


def plot_stratum_effort_distribution(summary: list[dict], path: Path) -> None:
    strata = sorted({row["stratum"] for row in summary}, key=STRATUM_ORDER.index)
    efforts = sorted({row["effort"] for row in summary}, key=EFFORT_ORDER.index)

    fig, axes = plt.subplots(1, len(strata), figsize=(6 * len(strata), 4), sharey=True)
    if len(strata) == 1:
        axes = [axes]

    bar_width = 0.8 / len(LABELS)
    label_colors = {"A": "#C62828", "B": "#EF6C00", "C": "#558B2F", "D": "#1565C0"}

    for ax, stratum in zip(axes, strata):
        rows = [next((r for r in summary if r["stratum"] == stratum and r["effort"] == e), None) for e in efforts]
        rows = [r for r in rows if r is not None]
        efforts_present = [r["effort"] for r in rows]
        x = np.arange(len(efforts_present))
        for li, label in enumerate(LABELS):
            ax.bar(
                x + li * bar_width,
                [r[f"frac_{label}"] for r in rows],
                bar_width,
                color=label_colors[label],
                label=label,
            )
        ax.set_xticks(x + bar_width * (len(LABELS) - 1) / 2)
        ax.set_xticklabels(efforts_present)
        ax.set_title(f"stratum = {stratum}")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1)
        ax.legend(title="answer", fontsize=8)
    fig.suptitle("ABCD distribution per (stratum, effort)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Section 2: Rank-level saturation --------------------------------------


def compute_rank_saturation(rows: list[JudgeResponse]) -> list[dict]:
    buckets: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for row in rows:
        key = (row.request.pair.stratum.value, row.request.pair.effort.value, row.request.pair.rank)
        buckets[key].append(row.answer.value)

    summary: list[dict] = []
    for (stratum, effort, rank), labels in sorted(buckets.items()):
        counts = Counter(labels)
        n = len(labels)
        row: dict = {
            "stratum": stratum,
            "effort": effort,
            "rank": rank,
            "n": n,
        }
        for label in LABELS:
            row[f"frac_{label}"] = counts.get(label, 0) / n if n else 0.0
        summary.append(row)
    return summary


def plot_rank_saturation(rank_rows: list[dict], path: Path) -> None:
    efforts_present = sorted(
        {r["effort"] for r in rank_rows},
        key=lambda e: EFFORT_ORDER.index(e) if e in EFFORT_ORDER else len(EFFORT_ORDER),
    )
    strata_present = sorted(
        {r["stratum"] for r in rank_rows},
        key=lambda s: STRATUM_ORDER.index(s) if s in STRATUM_ORDER else len(STRATUM_ORDER),
    )
    n_rows = max(1, len(strata_present))
    n_cols = max(1, len(efforts_present))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), sharey=True, squeeze=False)

    for stratum_idx, stratum in enumerate(strata_present):
        for effort_idx, effort in enumerate(efforts_present):
            ax = axes[stratum_idx, effort_idx]
            subset = sorted(
                [r for r in rank_rows if r["stratum"] == stratum and r["effort"] == effort],
                key=lambda r: r["rank"],
            )
            if not subset:
                ax.axis("off")
                continue
            ranks = [r["rank"] for r in subset]
            for label, color in [("A", "#C62828"), ("B", "#EF6C00"), ("C", "#558B2F"), ("D", "#1565C0")]:
                ax.plot(ranks, [r[f"frac_{label}"] for r in subset], marker="o", color=color, label=label)
            ax.set_title(f"{stratum} / {effort}", fontsize=10)
            if stratum_idx == n_rows - 1:
                ax.set_xlabel("rank (1 = most extreme within stratum)")
            if effort_idx == 0:
                ax.set_ylabel("fraction")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if stratum_idx == 0 and effort_idx == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Rank-level saturation of ABCD distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Section 3: Position bias --------------------------------------------


def compute_position_bias(rows: list[JudgeResponse]) -> dict:
    """AB vs BA agreement + weighted Cohen's kappa."""
    ab: dict[tuple, str] = {}
    ba: dict[tuple, str] = {}
    for row in rows:
        pair_key = (
            row.request.pair.keyword,
            row.request.pair.effort.value,
            row.request.pair.sample_i_index,
            row.request.pair.sample_j_index,
        )
        if row.request.order == JudgeOrder.AB:
            ab[pair_key] = row.answer.value
        else:
            ba[pair_key] = row.answer.value

    shared = sorted(set(ab) & set(ba))
    ab_list = [ab[k] for k in shared]
    ba_list = [ba[k] for k in shared]

    same = sum(1 for a, b in zip(ab_list, ba_list) if a == b)
    adjacent = sum(1 for a, b in zip(ab_list, ba_list) if a != b and abs(LABELS.index(a) - LABELS.index(b)) == 1)
    far = len(shared) - same - adjacent

    weighted_kappa = cohen_kappa_score(ab_list, ba_list, labels=LABELS, weights="quadratic") if shared else float("nan")
    unweighted_kappa = cohen_kappa_score(ab_list, ba_list, labels=LABELS) if shared else float("nan")

    # Per (stratum, effort) consistency
    by_bucket: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        pair_key = (
            row.request.pair.keyword,
            row.request.pair.effort.value,
            row.request.pair.sample_i_index,
            row.request.pair.sample_j_index,
        )
        if row.request.order != JudgeOrder.AB or pair_key not in ba:
            continue
        bucket_key = (row.request.pair.stratum.value, row.request.pair.effort.value)
        by_bucket[bucket_key].append(ab[pair_key] == ba[pair_key])
    bucket_summary = [
        {"stratum": k[0], "effort": k[1], "n": len(v), "consistency": float(np.mean(v)) if v else 0.0}
        for k, v in sorted(
            by_bucket.items(), key=lambda kv: (STRATUM_ORDER.index(kv[0][0]), EFFORT_ORDER.index(kv[0][1]))
        )
    ]

    return {
        "n_pairs": len(shared),
        "n_consistent": same,
        "n_adjacent_disagreement": adjacent,
        "n_far_disagreement": far,
        "consistency_rate": same / len(shared) if shared else 0.0,
        "weighted_kappa": float(weighted_kappa),
        "unweighted_kappa": float(unweighted_kappa),
        "per_bucket": bucket_summary,
    }


# --- Section 4: Effort effect size -----------------------------------------


def compute_effort_effect_on_frac_a(rows: list[JudgeResponse]) -> dict:
    """For top-stratum pairs, fraction of A judgments per keyword per effort.

    Uses both orders (majority / either counts as A if the more-different
    label agrees).  Effect size Cohen's d for none vs high effort, paired
    Mann-Whitney U for significance.
    """
    by_kw_effort: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row.request.pair.stratum != PairStratum.TOP:
            continue
        key = (row.request.pair.keyword, row.request.pair.effort.value)
        by_kw_effort[key].append(row.answer.value)

    frac_a: dict[str, dict[str, float]] = defaultdict(dict)
    for (kw, effort), labels in by_kw_effort.items():
        frac_a[kw][effort] = sum(1 for x in labels if x == "A") / len(labels)

    # Effect size on paired per-keyword arrays (same keyword measured at
    # both effort levels, so xs[i] and ys[i] share a baseline).  We use the
    # paired design (Cohen's d_z + Wilcoxon signed-rank) so the between-
    # keyword variance — which is nuisance here — doesn't inflate the
    # denominator of the effect size or dilute the test's power; an
    # independent-samples Cohen's d / Mann-Whitney U would treat each
    # (kw, effort_a, frac_A) and (kw, effort_b, frac_A) pair as an
    # independent subject and systematically under-report the effort effect
    # whenever keywords start from different frac_A baselines.
    effect_size_rows = []
    keywords = sorted(frac_a.keys())
    for effort_a, effort_b in [
        ("none", "low"),
        ("none", "medium"),
        ("none", "high"),
        ("low", "high"),
        ("medium", "high"),
    ]:
        xs = [frac_a[kw].get(effort_a) for kw in keywords if effort_a in frac_a[kw] and effort_b in frac_a[kw]]
        ys = [frac_a[kw].get(effort_b) for kw in keywords if effort_a in frac_a[kw] and effort_b in frac_a[kw]]
        if len(xs) < 2:
            continue
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        diffs = ys - xs
        diff_std = float(np.std(diffs, ddof=1))
        # Cohen's d_z = mean(diff) / std(diff); standardizes the within-
        # keyword change relative to the keyword-to-keyword scatter of that
        # change.
        d_z = float(diffs.mean() / diff_std) if diff_std > 0 else float("nan")
        # Wilcoxon signed-rank on ys vs xs: tests whether the median of
        # diffs is > 0.  ``zero_method="wilcox"`` drops zero-diff pairs
        # (pairs where frac_A is identical at both effort levels) before
        # ranking, matching the original Wilcoxon (1945) definition.
        w_stat, p_value = wilcoxon(ys, xs, alternative="greater", zero_method="wilcox")
        effect_size_rows.append(
            {
                "comparison": f"{effort_a}_vs_{effort_b}",
                "n_keywords": len(xs),
                "mean_baseline": float(xs.mean()),
                "mean_treatment": float(ys.mean()),
                "mean_diff": float(diffs.mean()),
                "cohens_dz": d_z,
                "wilcoxon_w": float(w_stat),
                "wilcoxon_p_greater": float(p_value),
            }
        )
    return {"per_keyword_frac_a": frac_a, "effect_size": effect_size_rows}


# --- Section 5: Embedding distance vs ABCD ----------------------------------


def compute_distance_label_correspondence(rows: list[JudgeResponse]) -> list[dict]:
    """Mean embedding distance per (stratum, effort, label)."""
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row.request.pair.stratum.value, row.request.pair.effort.value, row.answer.value)
        buckets[key].append(row.request.pair.embedding_distance)

    summary: list[dict] = []
    for stratum in STRATUM_ORDER:
        for effort in EFFORT_ORDER:
            for label in LABELS:
                arr = buckets.get((stratum, effort, label), [])
                if not arr:
                    continue
                summary.append(
                    {
                        "stratum": stratum,
                        "effort": effort,
                        "label": label,
                        "n": len(arr),
                        "mean_distance": float(np.mean(arr)),
                        "median_distance": float(np.median(arr)),
                        "std_distance": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    }
                )
    return summary


# --- Main ------------------------------------------------------------------


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("analyze_llm_judge")

    slug = _slugify(args.judge_model.value)
    judge_dir = args.run_dir / f"llm_judge_{slug}"
    input_path = judge_dir / "pairwise.jsonl"
    output_dir = args.output_dir or judge_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_responses(input_path)
    log.info("Loaded %d successful judgments from %s", len(rows), input_path)

    # 1. Stratum/effort distribution
    dist_rows = compute_stratum_effort_distribution(rows)
    write_csv(dist_rows, output_dir / "abcd_by_stratum_effort.csv")
    plot_stratum_effort_distribution(dist_rows, output_dir / "abcd_by_stratum_effort.png")

    print("\n=== ABCD distribution per (stratum, effort) ===")
    print(f"{'stratum':<8s} {'effort':<8s} {'n':>5s} {'%A':>6s} {'%B':>6s} {'%C':>6s} {'%D':>6s}")
    for row in dist_rows:
        print(
            f"{row['stratum']:<8s} {row['effort']:<8s} {row['n']:>5d} "
            f"{row['frac_A'] * 100:>5.1f}% {row['frac_B'] * 100:>5.1f}% "
            f"{row['frac_C'] * 100:>5.1f}% {row['frac_D'] * 100:>5.1f}%"
        )

    # 2. Rank saturation
    rank_rows = compute_rank_saturation(rows)
    write_csv(rank_rows, output_dir / "rank_saturation.csv")
    plot_rank_saturation(rank_rows, output_dir / "rank_saturation.png")

    # 3. Position bias
    bias = compute_position_bias(rows)
    print("\n=== Position bias (AB vs BA) ===")
    print(f"  pairs evaluated in both orders: {bias['n_pairs']}")
    print(f"  consistency rate: {bias['consistency_rate']:.2%}")
    print(f"    - exact agreement:        {bias['n_consistent']:>5d}")
    print(f"    - adjacent disagreement:  {bias['n_adjacent_disagreement']:>5d} (A<->B, B<->C, C<->D)")
    print(f"    - non-adjacent disagree:  {bias['n_far_disagreement']:>5d} (A<->C/D, B<->D)")
    print(f"  quadratic weighted kappa: {bias['weighted_kappa']:.4f}")
    print(f"  unweighted kappa:         {bias['unweighted_kappa']:.4f}")
    write_csv(bias["per_bucket"], output_dir / "position_bias.csv")

    # 4. Effort effect
    effect = compute_effort_effect_on_frac_a(rows)
    print("\n=== Effort effect on fraction-of-A (TOP stratum, paired by keyword) ===")
    print(f"{'comparison':<20s} {'n':>5s} {'mean_base':>10s} {'mean_trt':>10s} {'diff':>8s} {'d_z':>8s} {'p':>10s}")
    for row in effect["effect_size"]:
        print(
            f"{row['comparison']:<20s} {row['n_keywords']:>5d} "
            f"{row['mean_baseline']:>10.4f} {row['mean_treatment']:>10.4f} "
            f"{row['mean_diff']:>+8.4f} {row['cohens_dz']:>8.3f} {row['wilcoxon_p_greater']:>10.2e}"
        )
    write_csv(effect["effect_size"], output_dir / "effort_effect.csv")

    # per-keyword frac A table
    per_kw_rows = []
    for kw, efforts in sorted(effect["per_keyword_frac_a"].items()):
        row = {"keyword": kw}
        row.update({f"frac_A_{e}": v for e, v in efforts.items()})
        per_kw_rows.append(row)
    write_csv(per_kw_rows, output_dir / "per_keyword_frac_a.csv")

    # 5. Distance-label correspondence
    dist_label = compute_distance_label_correspondence(rows)
    write_csv(dist_label, output_dir / "distance_by_label.csv")

    log.info("Analysis artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
