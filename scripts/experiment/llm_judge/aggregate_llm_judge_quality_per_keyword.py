#!/usr/bin/env python3
"""Aggregate LLM-judge quality scores to per (keyword, effort) means.

Reads ``run_dir/llm_judge_quality_<slug>/quality.jsonl`` produced by
``scripts/experiment/llm_judge/llm_judge_quality.py`` and emits two CSVs:

  * ``per_keyword_effort.csv`` — wide format, one row per keyword,
    columns ``{axis}_{effort}`` and ``{axis}_delta_hn`` (high - none).
    Consumed by the §6 correlation script
    (``analyze_diversity_quality_correlation.py``); the global
    ``quality_per_keyword_effort.csv`` is the concatenation of this
    file across both generator run dirs, one judge at a time.

  * ``by_effort.csv`` — long format, one row per effort, mean +
    bootstrap 95% CI on each of originality / feasibility / clarity.
    Primary summary for the paper's per-effort quality table.

Usage:
    uv run python scripts/experiment/llm_judge/aggregate_llm_judge_quality_per_keyword.py \\
        --run-dir results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42 \\
        --judge-model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_registry import EffortName, JudgeModelName

_EFFORT_ORDER: tuple[str, ...] = tuple(e.value for e in EffortName)
# CSV-column suffixes line up with the legacy hand-rolled
# ``quality_per_keyword_effort.csv`` (``orig_none``, ``orig_low``,
# ``orig_med``, ``orig_high``) so the §6 correlation script keeps
# working without schema changes.
_EFFORT_COLUMN_SUFFIX: dict[str, str] = {
    "none": "none",
    "low": "low",
    "medium": "med",
    "high": "high",
}
_AXES: tuple[str, ...] = ("orig", "feas", "clar")
# JSONL field names -> short axis tag.
_SCORE_FIELD: dict[str, str] = {
    "originality": "orig",
    "feasibility": "feas",
    "clarity": "clar",
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def load_scores(quality_jsonl: Path) -> pd.DataFrame:
    """Load quality.jsonl into a long-format DataFrame.

    One row per judgment with columns (keyword, effort, orig, feas,
    clar).  Errored samples (present in ``quality.errors.jsonl`` but
    not ``quality.jsonl``) are simply absent; keeping the union
    explicit in downstream aggregates would require a placeholder
    that analysis scripts would have to filter anyway.
    """
    rows: list[dict] = []
    with open(quality_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            scores = payload["scores"]
            target = payload["request"]["target"]
            rows.append(
                {
                    "keyword": target["keyword"],
                    "effort": target["effort"],
                    "orig": float(scores["originality"]),
                    "feas": float(scores["feasibility"]),
                    "clar": float(scores["clarity"]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_per_keyword_effort(df: pd.DataFrame) -> pd.DataFrame:
    """Wide format: one row per keyword, columns ``{axis}_{suffix}``.

    Requires every (keyword, effort) cell that exists in the data to
    have >=1 sample; keywords with any missing effort bucket (among the
    tiers actually present in the dataframe) are dropped so Δ(high - lowest)
    is always defined. The Δ column ``{axis}_delta_hn`` uses ``high`` minus
    the lowest effort tier present (``none`` if available, else ``low``).
    Emits a warning log line with the drop count.
    """
    mean = df.groupby(["keyword", "effort"])[list(_AXES)].mean().reset_index()
    # Determine effort tiers actually present in this run.
    present_efforts = sorted(
        set(mean["effort"].unique()) & set(_EFFORT_COLUMN_SUFFIX.keys()),
        key=lambda e: list(_EFFORT_COLUMN_SUFFIX.keys()).index(e),
    )
    if "high" not in present_efforts:
        raise ValueError(f"no `high` effort tier in data; got {present_efforts}")
    lowest_effort = present_efforts[0]
    # pivot: keyword x (axis, effort)
    wide = mean.pivot(index="keyword", columns="effort", values=list(_AXES))
    # Flatten MultiIndex columns to ``{axis}_{suffix}``.
    wide.columns = [f"{axis}_{_EFFORT_COLUMN_SUFFIX[effort]}" for axis, effort in wide.columns]
    wide = wide.reset_index()
    # Drop keywords that miss any effort bucket among the tiers present
    # in this run.
    needed = [f"{axis}_{_EFFORT_COLUMN_SUFFIX[effort]}" for axis in _AXES for effort in present_efforts]
    before = len(wide)
    wide = wide.dropna(subset=needed)
    dropped = before - len(wide)
    if dropped:
        logging.warning(
            "dropped %d keywords with missing effort buckets (kept %d)",
            dropped,
            len(wide),
        )
    lowest_suffix = _EFFORT_COLUMN_SUFFIX[lowest_effort]
    for axis in _AXES:
        wide[f"{axis}_delta_hn"] = wide[f"{axis}_high"] - wide[f"{axis}_{lowest_suffix}"]
    ordered_cols = (
        ["keyword"]
        + [f"{axis}_{_EFFORT_COLUMN_SUFFIX[effort]}" for axis in _AXES for effort in present_efforts]
        + [f"{axis}_delta_hn" for axis in _AXES]
    )
    return wide[ordered_cols]


def aggregate_by_effort(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Long format: one row per effort, mean + bootstrap 95% CI per axis.

    Bootstrap over keywords (not individual judgments) because the
    natural unit of variability in this corpus is the keyword — within
    a keyword, the 30 samples share a prompt and are not independent
    draws for the effort-level estimate.  Using keyword-level bootstrap
    keeps the CI honest about how many *independent* observations back
    each effort mean.
    """
    # First collapse to per-(keyword, effort) means so the bootstrap
    # sample unit is the keyword.
    per_kw = df.groupby(["keyword", "effort"])[list(_AXES)].mean().reset_index()
    rows: list[dict] = []
    rng = np.random.default_rng(seed=42)
    for effort in _EFFORT_ORDER:
        sub = per_kw[per_kw["effort"] == effort]
        row: dict = {"effort": effort, "n_keywords": int(len(sub))}
        for axis in _AXES:
            vals = sub[axis].to_numpy()
            row[f"{axis}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            if len(vals) >= 2:
                boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
                row[f"{axis}_ci_lo"] = float(np.percentile(boots, 2.5))
                row[f"{axis}_ci_hi"] = float(np.percentile(boots, 97.5))
            else:
                row[f"{axis}_ci_lo"] = float("nan")
                row[f"{axis}_ci_hi"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.CLAUDE_HAIKU_4_5,
        choices=list(JudgeModelName),
        metavar=f"{{{','.join(m.value for m in JudgeModelName)}}}",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("aggregate_llm_judge_quality_per_keyword")

    slug = _slugify(args.judge_model.value)
    quality_dir = args.run_dir / f"llm_judge_quality_{slug}"
    quality_jsonl = quality_dir / "quality.jsonl"
    if not quality_jsonl.exists():
        raise SystemExit(f"quality.jsonl not found: {quality_jsonl}")

    df = load_scores(quality_jsonl)
    log.info("loaded %d quality judgments from %s", len(df), quality_jsonl)

    per_keyword = aggregate_per_keyword_effort(df)
    per_keyword_path = quality_dir / "per_keyword_effort.csv"
    per_keyword.to_csv(per_keyword_path, index=False)
    log.info("wrote %d keywords -> %s", len(per_keyword), per_keyword_path)

    by_effort = aggregate_by_effort(df)
    by_effort_path = quality_dir / "by_effort.csv"
    by_effort.to_csv(by_effort_path, index=False)
    log.info("wrote %d effort rows -> %s", len(by_effort), by_effort_path)


if __name__ == "__main__":
    main()
