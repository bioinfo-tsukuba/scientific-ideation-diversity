#!/usr/bin/env python3
"""TTS curves: diversity metrics vs actual generation compute.

Three versions emitted (one figure each) so the best Y-axis can be picked:
  (a) intra_dist  — intra-keyword mean cosine distance (combined embedding)
  (b) frac_A_top  — fraction of top-stratum pairs judged A
  (c) frac_A_all  — fraction of all judged pairs (top+middle+bottom) judged A

Each figure is a single panel with:
  - thin per-keyword line connecting that keyword's efforts (22 keywords / model)
  - thick median curve per model
  - log-scale x (mean gen_tokens per bucket)

Compute axis: ``total_tokens - input_tokens`` per sample, averaged over the
50 samples of each (keyword, effort) bucket.  This reconciles APIs where
reasoning tokens fold into ``output_tokens`` (Claude / GPT-5.4) and APIs
that expose them separately (Gemini).

Usage:
    uv run python scripts/experiment/llm_judge/analyze_llm_judge_tts_curve.py \\
        --run-dirs \\
            results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42 \\
            results/effort_diversity/phase1_gpt54_22kw_50x4_facet_seed42 \\
            results/effort_diversity/phase1_gemini31pro_22kw_50x3_facet_seed42 \\
        --judge-model gpt-4.1 \\
        --output-dir results/effort_diversity
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts import load_sample_records
from src.metrics import normalize_rows
from src.model_registry import EffortName, EmbeddingModelName, JudgeModelName
from src.schemas.llm_judge import JudgeAnswer, JudgeResponse
from src.text import slugify

# Tolerance for the cosine-distance range check (matches PR #51 convention).
_DISTANCE_EPS: float = 1e-9

LABELS = [m.value for m in JudgeAnswer]
EFFORT_ORDER = [m.value for m in EffortName]
MODEL_COLORS = {
    "claude": "#D32F2F",
    "gpt54": "#1976D2",
    "gemini31pro": "#2E7D32",
}

EMBEDDING_MODEL = EmbeddingModelName.TEXT_EMBEDDING_3_LARGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.GPT_4_1,
        choices=list(JudgeModelName),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def _model_key(run_dir: Path) -> str:
    parts = run_dir.name.split("_")
    return parts[1] if len(parts) > 1 else run_dir.name


def compute_intra_distance_per_bucket(run_dir: Path) -> dict[tuple[str, str], float]:
    """Per (keyword, effort) mean pairwise cosine distance over all C(50, 2) pairs."""
    records = load_sample_records(run_dir / "samples.jsonl")
    emb = np.load(run_dir / slugify(EMBEDDING_MODEL.value) / "embeddings.npy")
    normalized = normalize_rows(np.asarray(emb, dtype=np.float64))

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r.keyword, r.effort.value)].append(i)

    result: dict[tuple[str, str], float] = {}
    for key, idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        mat = normalized[idx]
        sim = mat @ mat.T
        tri_i, tri_j = np.triu_indices(len(idx), k=1)
        raw = 1.0 - sim[tri_i, tri_j]
        # Tolerance guard matching PR #51: drift beyond a few ULPs signals a
        # bug (un-normalized inputs / NaN embeddings), not float noise.
        if np.any((raw < -_DISTANCE_EPS) | (raw > 2.0 + _DISTANCE_EPS)):
            raise RuntimeError(
                f"Cosine distance out of expected [0, 2] range for {key!r}: "
                f"min={float(raw.min())!r}, max={float(raw.max())!r}. "
                "Embeddings are likely not row-normalized or contain NaN."
            )
        dists = np.clip(raw, 0.0, 2.0)
        result[key] = float(dists.mean())
    return result


def compute_bucket_rows(run_dir: Path, judge_model: str) -> list[dict]:
    """Per (keyword, effort): mean gen_tokens, intra_dist, judge-based fractions."""
    samples_path = run_dir / "samples.jsonl"
    judge_path = run_dir / f"llm_judge_{_slugify(judge_model)}" / "pairwise.jsonl"

    gen_tokens: dict[tuple[str, str], list[int]] = defaultdict(list)
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gen_tokens[(rec["keyword"], rec["effort"])].append(rec["total_tokens"] - rec["input_tokens"])

    intra_dist = compute_intra_distance_per_bucket(run_dir)

    # Raw aggregation: count each judgment (including both AB and BA orders).
    # This matches the per-judgment fraction used in INSIGHT 009 effect-size
    # calculations (Cohen's d), and avoids the inflation of "at least one A"
    # that pair-collapse introduces.
    top_A: dict[tuple[str, str], int] = defaultdict(int)
    top_n: dict[tuple[str, str], int] = defaultdict(int)
    all_A: dict[tuple[str, str], int] = defaultdict(int)
    all_n: dict[tuple[str, str], int] = defaultdict(int)
    with open(judge_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            resp = JudgeResponse.model_validate_json(line)
            kw = resp.request.pair.keyword
            effort = resp.request.pair.effort.value
            stratum = resp.request.pair.stratum.value
            bk = (kw, effort)
            is_A = resp.answer.value == "A"
            all_n[bk] += 1
            if is_A:
                all_A[bk] += 1
            if stratum == "top":
                top_n[bk] += 1
                if is_A:
                    top_A[bk] += 1

    rows: list[dict] = []
    for key in sorted(gen_tokens.keys()):
        kw, effort = key
        tok = np.asarray(gen_tokens[key], dtype=np.float64)
        row = {
            "keyword": kw,
            "effort": effort,
            "n_samples": int(tok.size),
            "mean_gen_tokens": float(tok.mean()),
            "median_gen_tokens": float(np.median(tok)),
            "intra_dist": intra_dist.get(key, float("nan")),
            "frac_A_top": top_A[key] / top_n[key] if top_n[key] else float("nan"),
            "frac_A_all": all_A[key] / all_n[key] if all_n[key] else float("nan"),
            "n_top_pairs": top_n[key],
            "n_all_pairs": all_n[key],
        }
        rows.append(row)
    return rows


def _baseline_effort(efforts: set[str]) -> str:
    """Lowest-compute effort available for this model.

    Claude/GPT expose ``none``; Gemini has no ``none`` and starts at ``low``.
    """
    for eff in EFFORT_ORDER:
        if eff in efforts:
            return eff
    raise ValueError("no effort found")


def plot_single_metric(
    data: dict[str, list[dict]],
    metric_key: str,
    metric_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Ratio-to-baseline plot with per-keyword thin lines + per-model median thick line.

    Each model's own minimum-effort bucket is the baseline (ratio 1.0).  The
    baseline's median absolute value is annotated in the legend to keep the
    absolute scale information without visually distorting small changes.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, color="#999", linestyle="--", linewidth=1.0, zorder=0)

    legend_entries: list[tuple[str, str, float]] = []  # (model, color, baseline_abs)
    for model, rows in data.items():
        color = MODEL_COLORS.get(model, "#666")
        efforts_present = {r["effort"] for r in rows}
        baseline_eff = _baseline_effort(efforts_present)

        # Per-keyword baseline and ratios
        by_kw: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_kw[r["keyword"]].append(r)
        kw_ratios: list[list[tuple[float, float, str]]] = []  # per-kw: list of (x, ratio, effort)
        baseline_abs_values: list[float] = []
        for kw, rs in by_kw.items():
            rs_sorted = sorted(rs, key=lambda r: EFFORT_ORDER.index(r["effort"]))
            baseline_row = next((r for r in rs_sorted if r["effort"] == baseline_eff), None)
            if baseline_row is None:
                continue
            baseline_val = baseline_row.get(metric_key, float("nan"))
            if np.isnan(baseline_val) or baseline_val == 0.0:
                continue
            baseline_abs_values.append(float(baseline_val))
            points: list[tuple[float, float, str]] = []
            for r in rs_sorted:
                v = r.get(metric_key, float("nan"))
                if np.isnan(v):
                    continue
                points.append((r["mean_gen_tokens"], v / baseline_val, r["effort"]))
            kw_ratios.append(points)

        # Thin per-keyword lines
        for points in kw_ratios:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color=color, alpha=0.10, linewidth=0.8, zorder=1)

        # Per-model median curve
        by_eff_ratios: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for points in kw_ratios:
            for x, ratio, eff in points:
                by_eff_ratios[eff].append((x, ratio))
        eff_xs: list[float] = []
        eff_ys: list[float] = []
        effort_labels: list[str] = []
        for eff in EFFORT_ORDER:
            entries = by_eff_ratios.get(eff, [])
            if not entries:
                continue
            xs_arr = np.asarray([e[0] for e in entries])
            ys_arr = np.asarray([e[1] for e in entries])
            eff_xs.append(float(np.median(xs_arr)))
            eff_ys.append(float(np.median(ys_arr)))
            effort_labels.append(eff)
        if eff_xs:
            ax.plot(
                eff_xs,
                eff_ys,
                color=color,
                alpha=1.0,
                linewidth=3.0,
                marker="o",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=5,
                label=model,
            )
            for x, y, lbl in zip(eff_xs, eff_ys, effort_labels):
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color=color, zorder=6)

        baseline_abs = float(np.median(baseline_abs_values)) if baseline_abs_values else float("nan")
        legend_entries.append((model, color, baseline_abs))

    ax.set_xscale("log")
    ax.set_xlabel("mean gen_tokens per sample  (total − input, log scale)")
    ax.set_ylabel(f"{metric_label}  /  baseline")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    # Legend with baseline absolute values
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for model, color, base_abs in legend_entries:
        handles.append(Line2D([0], [0], color=color, linewidth=3))
        labels.append(f"{model}  (baseline median = {base_abs:.3f})")
    handles.append(Line2D([0], [0], color="#999", linestyle="--", linewidth=1))
    labels.append("baseline = 1.0 (each model's lowest-effort bucket)")
    ax.legend(handles, labels, title="ratio to per-model baseline", loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("tts_curve")

    slug = _slugify(args.judge_model.value)
    all_rows: list[dict] = []
    data: dict[str, list[dict]] = {}
    for run_dir in args.run_dirs:
        model_key = _model_key(run_dir)
        rows = compute_bucket_rows(run_dir, args.judge_model.value)
        for r in rows:
            r["model"] = model_key
        data[model_key] = rows
        all_rows.extend(rows)
        log.info("%s: %d buckets", model_key, len(rows))

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"llm_judge_tts_curve_{slug}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Wrote %s", csv_path)

    variants = [
        ("intra_dist", "intra-kw mean cosine distance", "(a) intra-kw embedding distance vs compute"),
        ("frac_A_top", "P(label = A | top stratum)", "(b) top-stratum judge fraction-A vs compute"),
        ("frac_A_all", "P(label = A | all sampled pairs)", "(c) all-pairs judge fraction-A vs compute"),
    ]
    for key, ylabel, title in variants:
        path = output_dir / f"llm_judge_tts_curve_{slug}_{key}.png"
        plot_single_metric(data, key, ylabel, title, path)
        log.info("Wrote %s", path)

    # Summary
    print(f"\n{'model':<12s} {'effort':<8s} {'med_gen':>9s} {'med_intra':>10s} {'med_A_top':>10s} {'med_A_all':>10s}")
    for model, rows in data.items():
        per_eff = defaultdict(list)
        for r in rows:
            per_eff[r["effort"]].append(r)
        for eff in EFFORT_ORDER:
            rs = per_eff.get(eff, [])
            if not rs:
                continue
            med_gen = np.median([r["mean_gen_tokens"] for r in rs])
            med_intra = np.median([r["intra_dist"] for r in rs])
            med_A_top = np.median([r.get("frac_A_top", np.nan) for r in rs])
            med_A_all = np.median([r.get("frac_A_all", np.nan) for r in rs])
            print(f"{model:<12s} {eff:<8s} {med_gen:>9.1f} {med_intra:>10.4f} {med_A_top:>10.3f} {med_A_all:>10.3f}")


if __name__ == "__main__":
    main()
