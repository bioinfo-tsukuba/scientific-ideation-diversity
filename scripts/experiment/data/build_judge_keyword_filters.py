#!/usr/bin/env python3
"""Generate per-model Q_1 ∪ Q_4 keyword filter CSVs for the strong-judge ablation.

Splits ``q1_q4_all_models.csv`` (the prompt-sensitivity quartile pools, with
``generation_model`` column) into three per-model files that the judge CLI
(``llm_judge_quality.py --filter-keywords-csv``) can consume.  The
resulting CSV files share the input's column structure so existing
analysis scripts that read ``q1_q4_all_models.csv`` still work; the
judge CLI itself only reads the ``keyword`` column.

Usage::

    uv run python scripts/experiment/data/build_judge_keyword_filters.py

Writes to::

    results/effort_diversity/prompt_sensitivity/judge_keyword_filters/
      q1q4_claude.csv         (Claude Sonnet 4.6: 100 rows)
      q1q4_gpt54.csv          (GPT-5.4:           99 rows)
      q1q4_gemini31pro.csv    (Gemini 3.1 Pro:   100 rows)
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger("build_judge_keyword_filters")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "effort_diversity"
    / "prompt_sensitivity"
    / "q1_q4_all_models.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "effort_diversity"
    / "prompt_sensitivity"
    / "judge_keyword_filters"
)

# ``generation_model`` value in q1_q4_all_models.csv → output filename slug.
# Slugs mirror IDEA_MODEL_SHORT_KEY in src/model_registry.py.
MODEL_TO_SLUG: dict[str, str] = {
    "Claude Sonnet 4.6": "claude",
    "GPT-5.4": "gpt54",
    "Gemini 3.1 Pro": "gemini31pro",
}

EXPECTED_ROW_COUNTS: dict[str, int] = {
    "Claude Sonnet 4.6": 100,
    "GPT-5.4": 99,
    "Gemini 3.1 Pro": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "generation_model" not in fieldnames or "keyword" not in fieldnames:
            raise SystemExit(
                f"Input {input_path} missing required columns "
                f"(found: {fieldnames!r}); expected at least "
                f"'keyword' and 'generation_model'."
            )
        rows = list(reader)

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, slug in MODEL_TO_SLUG.items():
        model_rows = [r for r in rows if r["generation_model"] == model_name]
        expected = EXPECTED_ROW_COUNTS[model_name]
        if len(model_rows) != expected:
            raise SystemExit(
                f"Model {model_name!r} has {len(model_rows)} rows in input; "
                f"expected {expected} (Q_1 + Q_4 = 50 + 50, GPT-5.4 has 49 + 50 "
                f"due to bioterrorism safety refusal under SSoT — see paper §3.6)."
            )

        out_path = output_dir / f"q1q4_{slug}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(model_rows)
        logger.info(
            "wrote %s (%d rows; %s)", out_path.relative_to(REPO_ROOT), len(model_rows), model_name
        )

    logger.info("done.  Pass these CSVs to llm_judge_quality.py via --filter-keywords-csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
