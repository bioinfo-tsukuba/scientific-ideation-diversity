"""Task 3b (issue #185): Aggregate cross-judge kappa across all 12 prompt-sensitivity cells.

Reads pairwise_agreement_summary.csv from each cell's
cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/ directory and emits a
12-row summary table with (model, prompt, effort, kappa, exact_agreement_rate, n_aligned).

Output: results/effort_diversity/prompt_sensitivity/cross_judge_kappa_summary.csv

Usage:
    uv run python paper/scripts/aggregate_cross_judge_kappa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import IDEA_MODEL_LABEL, IDEA_MODEL_SHORT_KEY, IdeaModelName  # noqa: E402

PS_DIR = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"
CROSS_JUDGE_SUBDIR = "cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001"
PROMPTS = ("vs", "ssot")
PROMPT_LABEL = {"vs": "VS", "ssot": "SSoT"}
EFFORTS = ("low", "high")


def main() -> None:
    rows: list[dict] = []
    for idea_model in IdeaModelName:
        model = IDEA_MODEL_SHORT_KEY[idea_model]
        for prompt in PROMPTS:
            for effort in EFFORTS:
                cell = f"{model}_{prompt}_{effort}"
                csv = PS_DIR / cell / CROSS_JUDGE_SUBDIR / "pairwise_agreement_summary.csv"
                df = pd.read_csv(csv)
                assert len(df) == 1
                r = df.iloc[0]
                rows.append(
                    {
                        "model": model,
                        "model_label": IDEA_MODEL_LABEL[idea_model],
                        "prompt": PROMPT_LABEL[prompt],
                        "effort": effort,
                        "n_aligned": int(r["n_aligned"]),
                        "quadratic_kappa": r["quadratic_kappa"],
                        "exact_agreement_rate": r["exact_agreement_rate"],
                    }
                )

    out = pd.DataFrame(rows)
    out_path = PS_DIR / "cross_judge_kappa_summary.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
