#!/usr/bin/env python3
"""Facet attribution for LLM-judge labels.

Role in the paper (see ``docs/20260419_results_structure_combined_vs_facet.md``):
  Substitutes for running a per-facet LLM-judge.  For each pair that was
  judged holistically (combined idea text), we compute its cosine distance in
  each of the three per-facet embedding spaces (purpose / mechanism /
  evaluation) and look at how the ABCD label distribution correlates with
  each facet distance.  This tells us which facet(s) drive the judge's
  "different" (A) or "identical" (D) calls — i.e., facet-level *attribution*
  of the holistic judgment, at zero additional API cost.

Usage:
    uv run python scripts/experiment/llm_judge/analyze_llm_judge_facet_attribution.py \\
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42 \\
        --judge-model gpt-4.1

Outputs (written next to the judge JSONL):
  - ``llm_judge_facet_attribution_{judge}.csv``  — per (label, facet, effort, stratum) distance stats
  - ``llm_judge_facet_attribution_{judge}.png``  — faceted boxplots
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import load_sample_records
from src.metrics import normalize_rows
from src.model_registry import EffortName, EmbeddingModelName, JudgeModelName
from src.schemas.llm_judge import JudgeAnswer, JudgeResponse, PairStratum
from src.text import slugify

FACETS = ["purpose", "mechanism", "evaluation"]
EMBEDDING_MODEL = EmbeddingModelName.TEXT_EMBEDDING_3_LARGE
LABELS = [m.value for m in JudgeAnswer]
STRATA = [m.value for m in PairStratum]
# Effort ordering mirrors :class:`EffortName` (none < low < medium < high)
# rather than the alphabetical sort that ``sorted()`` produces, which
# would reorder the x-axis to (high, low, medium, none) — visually
# meaningless for a dose-response curve.
EFFORTS = [m.value for m in EffortName]
LABEL_COLORS = {"A": "#C62828", "B": "#EF6C00", "C": "#558B2F", "D": "#1565C0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.GPT_4_1,
        choices=list(JudgeModelName),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def load_per_facet_embeddings(run_dir: Path, n_records: int) -> dict[str, np.ndarray]:
    """Return ``{facet: (n_records, D) normalized array}`` from per-facet npy.

    Reads ``<run_dir>/<slugify(model)>/embeddings_<facet>.npy`` for each of
    the three facets.  The per-facet arrays are already aligned to
    :func:`load_sample_records` order (see ``build_combined_openai_embedding``
    and ``convert_facet_npz_to_per_field_npy``), so no reordering is needed.
    """
    subdir = run_dir / slugify(EMBEDDING_MODEL.value)
    out: dict[str, np.ndarray] = {}
    for facet in FACETS:
        path = subdir / f"embeddings_{facet}.npy"
        arr = np.load(path)
        if arr.shape[0] != n_records:
            raise ValueError(f"facet={facet!r}: {path} has {arr.shape[0]} rows != {n_records} records")
        out[facet] = normalize_rows(arr.astype(np.float64))
    return out


def _pair_distance(
    facet_matrix: np.ndarray,
    i: int,
    j: int,
) -> float:
    return float(1.0 - np.dot(facet_matrix[i], facet_matrix[j]))


def compute_attribution_rows(
    responses: list[JudgeResponse],
    facet_matrices: dict[str, np.ndarray],
) -> list[dict]:
    """For each (facet, label, stratum, effort), mean/median/std distance."""
    # Collect per-facet distances for each response
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for resp in responses:
        i = resp.request.pair.sample_i_index
        j = resp.request.pair.sample_j_index
        for facet in FACETS:
            dist = _pair_distance(facet_matrices[facet], i, j)
            key = (
                facet,
                resp.answer.value,
                resp.request.pair.stratum.value,
                resp.request.pair.effort.value,
            )
            buckets[key].append(dist)

    rows: list[dict] = []
    for (facet, label, stratum, effort), dists in sorted(buckets.items()):
        arr = np.asarray(dists, dtype=np.float64)
        rows.append(
            {
                "facet": facet,
                "label": label,
                "stratum": stratum,
                "effort": effort,
                "n": int(arr.size),
                "mean_distance": float(arr.mean()),
                "median_distance": float(np.median(arr)),
                "std_distance": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "p10_distance": float(np.percentile(arr, 10)),
                "p90_distance": float(np.percentile(arr, 90)),
            }
        )
    return rows


def plot_attribution(rows: list[dict], output_path: Path) -> None:
    """Grid: rows = facet, cols = stratum, x = effort, y = mean distance by label."""
    strata = sorted({r["stratum"] for r in rows}, key=STRATA.index)
    efforts = sorted({r["effort"] for r in rows}, key=EFFORTS.index)

    fig, axes = plt.subplots(
        len(FACETS),
        len(strata),
        figsize=(5 * len(strata), 3.5 * len(FACETS)),
        sharey="row",
    )
    if len(strata) == 1:
        axes = axes.reshape(-1, 1)

    for fi, facet in enumerate(FACETS):
        for si, stratum in enumerate(strata):
            ax = axes[fi, si]
            bar_width = 0.8 / max(1, len(LABELS))
            x = np.arange(len(efforts))
            any_bar = False
            for li, label in enumerate(LABELS):
                ys = []
                for effort in efforts:
                    match = next(
                        (
                            r
                            for r in rows
                            if r["facet"] == facet
                            and r["label"] == label
                            and r["stratum"] == stratum
                            and r["effort"] == effort
                        ),
                        None,
                    )
                    ys.append(match["mean_distance"] if match else np.nan)
                if any(not np.isnan(y) for y in ys):
                    ax.bar(
                        x + li * bar_width,
                        ys,
                        bar_width,
                        color=LABEL_COLORS[label],
                        label=label,
                        alpha=0.85,
                    )
                    any_bar = True
            ax.set_title(f"facet={facet}, stratum={stratum}", fontsize=10)
            ax.set_xticks(x + bar_width * (len(LABELS) - 1) / 2)
            ax.set_xticklabels(efforts)
            if si == 0:
                ax.set_ylabel(f"mean dist ({facet})")
            if fi == len(FACETS) - 1:
                ax.set_xlabel("effort")
            ax.grid(True, axis="y", alpha=0.3)
            if fi == 0 and si == 0 and any_bar:
                ax.legend(title="judge label", fontsize=8)
    fig.suptitle(
        "Facet attribution: mean per-facet distance by (label, stratum, effort)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("facet_attribution")

    slug = _slugify(args.judge_model.value)
    judge_dir = args.run_dir / f"llm_judge_{slug}"
    output_dir = args.output_dir or judge_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_path = judge_dir / "pairwise.jsonl"

    records = load_sample_records(args.run_dir / "samples.jsonl")
    log.info("Loaded %d records", len(records))

    facet_matrices = load_per_facet_embeddings(args.run_dir, n_records=len(records))
    for facet, arr in facet_matrices.items():
        log.info("  facet=%s shape=%s", facet, arr.shape)

    responses = [JudgeResponse.model_validate_json(line) for line in open(judge_path) if line.strip()]
    log.info("Loaded %d judge responses from %s", len(responses), judge_path)

    rows = compute_attribution_rows(responses, facet_matrices)

    # Write CSV
    csv_path = output_dir / "facet_attribution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", csv_path)

    plot_path = output_dir / "facet_attribution.png"
    plot_attribution(rows, plot_path)
    log.info("Wrote %s", plot_path)

    # Human-readable summary: per (stratum, label) facet-distance ranking
    print("\n=== Per-facet mean distance by (stratum, label), effort-aggregated ===")
    print(f"{'stratum':<8s} {'label':<6s} {'purpose':>10s} {'mechanism':>11s} {'evaluation':>11s} {'n':>6s}")
    agg: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["stratum"], r["label"], r["facet"])].append((r["mean_distance"], r["n"]))
    printed = set()
    for stratum in STRATA:
        for label in LABELS:
            vals = []
            n_total = 0
            for facet in FACETS:
                entries = agg.get((stratum, label, facet), [])
                if not entries:
                    vals.append(None)
                else:
                    weighted_mean = sum(m * n for m, n in entries) / sum(n for _, n in entries)
                    vals.append(weighted_mean)
                    n_total = max(n_total, sum(n for _, n in entries))
            if all(v is None for v in vals):
                continue
            printed.add((stratum, label))
            p, m, e = vals

            def _fmt(x):
                return f"{x:>10.4f}" if x is not None else f"{'-':>10s}"

            print(f"{stratum:<8s} {label:<6s} {_fmt(p):>10s} {_fmt(m):>10s} {_fmt(e):>10s} {n_total:>6d}")


if __name__ == "__main__":
    main()
