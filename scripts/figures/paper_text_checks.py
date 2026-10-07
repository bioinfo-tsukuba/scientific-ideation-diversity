"""Recompute two statistics quoted in the paper text whose definitions the paper leaves implicit.

1. Cross-judge agreement on the effort axis (Sect. 5, "Pairwise judge and facet
   localization"). Quadratic-weighted Cohen's kappa between GPT-4.1 and Claude
   Haiku 4.5 on aligned pairwise judgments, reported both over all effort tiers
   present in the run (Claude Sonnet 4.6 and GPT-5.4 runs also contain the
   `none` tier) and over the three analysed tiers low / medium / high.
2. Prompt-axis quality change (Sect. 5, quality paragraph). For each
   (model, judge, effort in {low, high}, quality axis), the absolute difference
   between the Q1 u Q4 keyword mean under VS or SSoT and under the default
   prompt; mean and max over all combinations.

Run from the repository root after Stage 0 (REPRODUCING.md):

    uv run python scripts/figures/paper_text_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiment" / "llm_judge"))
from analyze_cross_judge_agreement import _confusion_and_kappa, _load_pairwise  # noqa: E402

ROOT = REPO_ROOT / "results" / "effort_diversity"
PS = ROOT / "prompt_sensitivity"
PHASE2 = {
    "claude": "phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54": "phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini31pro": "phase2_gemini31pro_1180kw_30x3_facet_seed42",
}
JUDGES = ["gpt_4_1", "claude_haiku_4_5_20251001"]
XJUDGE = "cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001"
AXES = ["orig", "feas", "clar"]
KEY = ["keyword", "effort", "sample_i", "sample_j", "order"]


def kappa_table() -> pd.DataFrame:
    rows = []
    for model, run in PHASE2.items():
        a = _load_pairwise(ROOT / run / "llm_judge_gpt_4_1" / "pairwise.jsonl")
        b = _load_pairwise(ROOT / run / "llm_judge_claude_haiku_4_5_20251001" / "pairwise.jsonl")
        merged = a.merge(b, on=KEY, suffixes=("_a", "_b"), how="inner", validate="one_to_one")
        for tiers, sub in [("all tiers", merged), ("low/medium/high", merged[merged.effort.isin(["low", "medium", "high"])])]:
            _, kappa, _ = _confusion_and_kappa(sub["answer_a"].to_numpy(), sub["answer_b"].to_numpy())
            rows.append({"model": model, "tiers": tiers, "n_aligned": len(sub), "quadratic_kappa": round(kappa, 4)})
    return pd.DataFrame(rows)


def prompt_quality_change() -> pd.DataFrame:
    rows = []
    for model, run in PHASE2.items():
        keywords = pd.read_csv(PS / f"q1_q4_{model}.csv").keyword
        for judge in JUDGES:
            default = pd.read_csv(ROOT / run / XJUDGE / f"per_keyword_effort_{judge}.csv")
            for effort in ["low", "high"]:
                base = default[default.keyword.isin(keywords)]
                for prompt in ["vs", "ssot"]:
                    cell = pd.read_csv(PS / f"{model}_{prompt}_{effort}" / XJUDGE / f"per_keyword_effort_{judge}.csv")
                    cell = cell[cell.keyword.isin(keywords)]
                    for axis in AXES:
                        col = f"{axis}_{effort}"
                        rows.append(
                            {
                                "model": model,
                                "judge": judge,
                                "effort": effort,
                                "prompt": prompt,
                                "axis": axis,
                                "abs_change_vs_default": abs(cell[col].mean() - base[col].mean()),
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    kappa = kappa_table()
    print("Cross-judge quadratic kappa (effort axis)")
    print(kappa.to_string(index=False))
    change = prompt_quality_change()
    print("\nPrompt-axis quality change |prompt - default|, both judges, Q1 u Q4")
    print(f"mean {change.abs_change_vs_default.mean():.3f} pt, max {change.abs_change_vs_default.max():.3f} pt, n={len(change)}")
    out = REPO_ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    kappa.to_csv(out / "cross_judge_kappa_effort_axis.csv", index=False)
    change.to_csv(out / "prompt_axis_quality_change.csv", index=False)


if __name__ == "__main__":
    main()
