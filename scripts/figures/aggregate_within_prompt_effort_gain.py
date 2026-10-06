"""Task 1 (issue #185): Within-prompt effort scaling gain table for paper §4.4 Para 1.

For each (model, prompt kind, diversity stratum), computes the low→high relative gain:
    (diversity_high - diversity_low) / diversity_low × 100  [percent]

Source: results/effort_diversity/prompt_sensitivity/<model>_token_pareto.csv
Output: results/effort_diversity/prompt_sensitivity/within_prompt_effort_gain.csv
        (~27 rows: 3 model × 3 kind × 3 stratum)

Usage:
    uv run python paper/scripts/aggregate_within_prompt_effort_gain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import IDEA_MODEL_LABEL, IDEA_MODEL_SHORT_KEY, IdeaModelName  # noqa: E402

CSV_DIR = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"
KIND_LABEL = {"phase2": "default", "vs": "VS", "ssot": "SSoT"}


def _relative_gain_pct(high: float, low: float) -> float:
    return (high - low) / low * 100.0


def main() -> None:
    rows: list[dict] = []
    for idea_model in IdeaModelName:
        model = IDEA_MODEL_SHORT_KEY[idea_model]
        df = pd.read_csv(CSV_DIR / f"{model}_token_pareto.csv")
        for kind in ("phase2", "vs", "ssot"):
            sub = df[df["kind"] == kind].set_index("effort")
            if "low" not in sub.index or "high" not in sub.index:
                continue
            lo, hi = sub.loc["low"], sub.loc["high"]
            for stratum, col in [
                ("overall", "diversity_mean"),
                ("Q1", "diversity_q1_mean"),
                ("Q4", "diversity_q4_mean"),
            ]:
                rows.append(
                    {
                        "model": model,
                        "model_label": IDEA_MODEL_LABEL[idea_model],
                        "prompt": KIND_LABEL[kind],
                        "stratum": stratum,
                        "diversity_low": lo[col],
                        "diversity_high": hi[col],
                        "gain_pct": _relative_gain_pct(hi[col], lo[col]),
                    }
                )

    out = pd.DataFrame(rows)
    out_path = CSV_DIR / "within_prompt_effort_gain.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
