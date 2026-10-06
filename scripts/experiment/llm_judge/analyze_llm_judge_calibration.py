#!/usr/bin/env python3
"""Distance-conditional calibration of LLM-judge labels.

Answers the question "does the judge agree with embedding distance
consistently, regardless of effort?" by binning pairs by absolute embedding
distance and plotting, for each effort, the fraction of each judge label in
the bin.  If the per-effort curves overlap, the judge is distance-consistent
and any rank-stratum effort shift is simply a consequence of the effort
shifting the distance distribution (not a judge bias).

Pairs are deduplicated across the two AB/BA orderings: an AB/BA pair is
counted once, using the most-different of the two labels when they disagree
(mirrors the adjacent-disagreement handling in analyze_llm_judge.py).

Usage:
    uv run python scripts/experiment/llm_judge/analyze_llm_judge_calibration.py \\
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.model_registry import EffortName, JudgeModelName
from src.schemas.llm_judge import JudgeAnswer, JudgeResponse

LABELS = [m.value for m in JudgeAnswer]
EFFORT_ORDER = [m.value for m in EffortName]
EFFORT_COLORS = {"none": "#90CAF9", "low": "#42A5F5", "medium": "#1E88E5", "high": "#0D47A1"}
DEFAULT_BINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.GPT_4_1,
        choices=list(JudgeModelName),
    )
    parser.add_argument(
        "--bins",
        type=float,
        nargs="+",
        default=DEFAULT_BINS,
        help="Distance bin edges (ascending).  Default gives 12 bins over [0, 0.7].",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def collapse_pair_orders(responses: list[JudgeResponse]) -> dict[tuple, dict]:
    """Return {pair_key: {"label": final_label, "distance": float, "effort": str}}."""
    by_pair: dict[tuple, list[JudgeResponse]] = defaultdict(list)
    for resp in responses:
        pair_key = (
            resp.request.pair.keyword,
            resp.request.pair.effort.value,
            resp.request.pair.sample_i_index,
            resp.request.pair.sample_j_index,
        )
        by_pair[pair_key].append(resp)

    collapsed: dict[tuple, dict] = {}
    for pair_key, rs in by_pair.items():
        labels = [r.answer.value for r in rs]
        # Most-different (smallest index in ABCD) wins on disagreement.
        final_label = min(labels, key=lambda x: LABELS.index(x))
        collapsed[pair_key] = {
            "label": final_label,
            "distance": rs[0].request.pair.embedding_distance,
            "effort": pair_key[1],
            "stratum": rs[0].request.pair.stratum.value,
        }
    return collapsed


def compute_calibration_rows(
    pair_summary: dict[tuple, dict],
    bin_edges: tuple[float, ...],
) -> list[dict]:
    """For each (effort, bin), fraction of each label and total n."""
    rows: list[dict] = []
    efforts = sorted(
        {p["effort"] for p in pair_summary.values()},
        key=lambda e: EFFORT_ORDER.index(e) if e in EFFORT_ORDER else len(EFFORT_ORDER),
    )
    for effort in efforts:
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            pairs_in_bin = [p for p in pair_summary.values() if p["effort"] == effort and lo <= p["distance"] < hi]
            if not pairs_in_bin:
                continue
            counts = Counter(p["label"] for p in pairs_in_bin)
            n = len(pairs_in_bin)
            row: dict = {
                "effort": effort,
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_center": (lo + hi) / 2,
                "n": n,
                "mean_distance": float(np.mean([p["distance"] for p in pairs_in_bin])),
            }
            for label in LABELS:
                row[f"count_{label}"] = counts.get(label, 0)
                row[f"frac_{label}"] = counts.get(label, 0) / n if n else 0.0
            row["frac_A_or_B"] = (counts.get("A", 0) + counts.get("B", 0)) / n if n else 0.0
            rows.append(row)
    return rows


def plot_calibration(rows: list[dict], output_path: Path, title_suffix: str = "") -> None:
    """Line plots of P(label) vs distance, per effort.  3 metrics side by side."""
    efforts = sorted(
        {r["effort"] for r in rows}, key=lambda e: EFFORT_ORDER.index(e) if e in EFFORT_ORDER else len(EFFORT_ORDER)
    )

    metrics: list[tuple[str, str]] = [
        ("frac_A", "P(label = A)"),
        ("frac_A_or_B", "P(label ∈ {A, B})"),
        ("frac_D", "P(label = D)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharex=True)
    for mi, (key, ylabel) in enumerate(metrics):
        ax = axes[mi]
        for effort in efforts:
            rs = sorted([r for r in rows if r["effort"] == effort], key=lambda r: r["bin_center"])
            if not rs:
                continue
            # Use bin_center as x, but filter to bins with n >= some threshold for robustness
            xs = [r["mean_distance"] for r in rs if r["n"] >= 10]
            ys = [r[key] for r in rs if r["n"] >= 10]
            ns = [r["n"] for r in rs if r["n"] >= 10]
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=[max(4, min(12, n / 20)) for n in ns][0] if ys else 6,
                color=EFFORT_COLORS.get(effort, "#666"),
                label=effort,
                linewidth=2,
            )
        ax.set_xlabel("embedding distance")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend(title="effort")
    title = "Distance-conditional judge calibration"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("calibration")

    slug = _slugify(args.judge_model.value)
    judge_dir = args.run_dir / f"llm_judge_{slug}"
    output_dir = args.output_dir or judge_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    responses = [
        JudgeResponse.model_validate_json(line)
        for line in open(judge_dir / "pairwise.jsonl", encoding="utf-8")
        if line.strip()
    ]
    log.info("Loaded %d responses", len(responses))

    pair_summary = collapse_pair_orders(responses)
    log.info("Collapsed to %d unique pairs", len(pair_summary))

    rows = compute_calibration_rows(pair_summary, tuple(args.bins))
    csv_path = output_dir / "calibration.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", csv_path)

    plot_path = output_dir / "calibration.png"
    plot_calibration(rows, plot_path, title_suffix=args.run_dir.name.split("_")[1])
    log.info("Wrote %s", plot_path)

    # Print concise summary: compute curve separation per effort pair
    efforts = sorted(
        {r["effort"] for r in rows}, key=lambda e: EFFORT_ORDER.index(e) if e in EFFORT_ORDER else len(EFFORT_ORDER)
    )
    print("\n=== Calibration summary: P(label=A) per (effort, distance-bin) ===")
    # Row per bin, col per effort
    bins = sorted({(r["bin_lo"], r["bin_hi"]) for r in rows})
    header = (
        f"{'bin':<14s} | "
        + " | ".join(f"{e:>10s}" for e in efforts)
        + " | "
        + " | ".join(f"n({e:<4s})" for e in efforts)
    )
    print(header)
    for lo, hi in bins:
        cells = []
        ns = []
        for eff in efforts:
            match = next((r for r in rows if r["effort"] == eff and r["bin_lo"] == lo and r["bin_hi"] == hi), None)
            if match is None:
                cells.append(f"{'—':>10s}")
                ns.append("—")
            else:
                cells.append(f"{match['frac_A'] * 100:>9.1f}%")
                ns.append(str(match["n"]))
        print(f"[{lo:.2f},{hi:.2f})   | " + " | ".join(cells) + " | " + " | ".join(f"{n:>6s}" for n in ns))


if __name__ == "__main__":
    main()
