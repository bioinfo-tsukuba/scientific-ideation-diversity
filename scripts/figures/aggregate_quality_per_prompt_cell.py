"""Task 10 (issue #185): Per-(prompt, effort) quality avg table for paper §4.5 / appendix.

Computes per-cell mean ± bootstrap 95% CI for originality, feasibility, clarity
across the 6 (prompt × effort) cells per model = 18 rows total.

All cells use the same 100-keyword Q1/Q4 subset defined by q1_q4_<model>.csv so
means are comparable across prompt conditions.

Source:
  - Q1/Q4 keyword lists:       results/effort_diversity/prompt_sensitivity/q1_q4_<model>.csv
  - Default (phase2) quality:  results/effort_diversity/<phase2_dir>/
                                  cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/
                                  per_keyword_effort_gpt_4_1.csv
  - VS/SSoT quality:           results/effort_diversity/prompt_sensitivity/<cell>/
                                  cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/
                                  per_keyword_effort_gpt_4_1.csv

Output: results/effort_diversity/prompt_sensitivity/paper_table_quality_per_prompt_cell.csv
        (18 rows × 9 quality columns)

Usage:
    uv run python paper/scripts/aggregate_quality_per_prompt_cell.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import (  # noqa: E402
    IDEA_MODEL_LABEL,
    IDEA_MODEL_PHASE2_DIR,
    IDEA_MODEL_SHORT_KEY,
    IdeaModelName,
)

EFFORT_DIR = REPO_ROOT / "results" / "effort_diversity"
PS_DIR = EFFORT_DIR / "prompt_sensitivity"
CROSS_JUDGE = "cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001"
AXES = ("orig", "feas", "clar")
N_BOOT = 10_000


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    boots = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for idea_model in IdeaModelName:
        model = IDEA_MODEL_SHORT_KEY[idea_model]
        kw_df = pd.read_csv(PS_DIR / f"q1_q4_{model}.csv")
        kw_100 = set(kw_df["keyword"])

        # Default (phase2) — filter to the 100-kw subset
        phase2_kw = pd.read_csv(
            EFFORT_DIR / IDEA_MODEL_PHASE2_DIR[idea_model] / CROSS_JUDGE / "per_keyword_effort_gpt_4_1.csv"
        )
        phase2_kw = phase2_kw[phase2_kw["keyword"].isin(kw_100)]
        for effort in ("low", "high"):
            row: dict = {
                "model": model,
                "model_label": IDEA_MODEL_LABEL[idea_model],
                "prompt": "default",
                "effort": effort,
                "n_keywords": len(phase2_kw),
            }
            for ax in AXES:
                col = f"{ax}_{effort}"
                vals = phase2_kw[col].dropna().to_numpy()
                mean, ci_lo, ci_hi = bootstrap_mean_ci(vals, rng)
                row[f"{ax}_mean"] = mean
                row[f"{ax}_ci_lo"] = ci_lo
                row[f"{ax}_ci_hi"] = ci_hi
            rows.append(row)

        # VS and SSoT cells
        for prompt in ("vs", "ssot"):
            for effort in ("low", "high"):
                cell = f"{model}_{prompt}_{effort}"
                cell_kw = pd.read_csv(PS_DIR / cell / CROSS_JUDGE / "per_keyword_effort_gpt_4_1.csv")
                row = {
                    "model": model,
                    "model_label": IDEA_MODEL_LABEL[idea_model],
                    "prompt": "VS" if prompt == "vs" else "SSoT",
                    "effort": effort,
                    "n_keywords": len(cell_kw),
                }
                for ax in AXES:
                    col = f"{ax}_{effort}"
                    vals = cell_kw[col].dropna().to_numpy()
                    mean, ci_lo, ci_hi = bootstrap_mean_ci(vals, rng)
                    row[f"{ax}_mean"] = mean
                    row[f"{ax}_ci_lo"] = ci_lo
                    row[f"{ax}_ci_hi"] = ci_hi
                rows.append(row)

    out = pd.DataFrame(rows)
    out_path = PS_DIR / "paper_table_quality_per_prompt_cell.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nSaved → {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
