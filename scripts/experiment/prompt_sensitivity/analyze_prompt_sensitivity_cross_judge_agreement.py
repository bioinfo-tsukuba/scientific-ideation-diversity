"""Cross-judge agreement (#162): GPT-4.1 vs Haiku 4.5 across 12 prompt-
sensitivity cells (3 model × {VS,SSoT} × {low,high}).

Mirrors paper App I's phase2 4-cell cross-judge analysis on the prompt
axis introduced by paper §4.6: for each cell, align the two judges on
shared (keyword, effort, sample_i, sample_j, order) pairs, then report
quadratic-weighted Cohen's kappa and exact-agreement rate, both overall
and per pair-stratum (top / middle / bottom). The pair-stratum split
is what shows whether judge agreement is driven by the easy
extremes or holds in the middle where judges genuinely disagree.

Reuses the merge / confusion helpers from
``analyze_cross_judge_agreement.py`` so the per-cell numbers are exactly
comparable to the phase2 ones.

Output: a single long-format CSV at
``results/effort_diversity/prompt_sensitivity/cross_judge_agreement.csv``
with one row per (model, prompt, effort, pair_stratum) where
``pair_stratum=all`` aggregates over the cell.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.llm_judge.analyze_cross_judge_agreement import (  # noqa: E402
    _confusion_and_kappa,
    _load_pairwise,
    _slugify,
)
from src.model_registry import JudgeModelName  # noqa: E402
from src.schemas.llm_judge import PairStratum  # noqa: E402

MODELS: tuple[str, ...] = ("claude", "gpt54", "gemini31pro")
PROMPT_EFFORTS: tuple[tuple[str, str], ...] = (
    ("vs", "low"),
    ("vs", "high"),
    ("ssot", "low"),
    ("ssot", "high"),
)
JUDGE_A = JudgeModelName.GPT_4_1
JUDGE_B = JudgeModelName.CLAUDE_HAIKU_4_5
STRATA: tuple[str, ...] = tuple(m.value for m in PairStratum)


def _agreement_for_cell(run_dir: Path, log: logging.Logger) -> list[dict] | None:
    """Return one row per pair_stratum (incl. ``all``) or None if missing."""
    path_a = run_dir / f"llm_judge_{_slugify(JUDGE_A.value)}" / "pairwise.jsonl"
    path_b = run_dir / f"llm_judge_{_slugify(JUDGE_B.value)}" / "pairwise.jsonl"
    if not path_a.exists() or not path_b.exists():
        log.warning("missing pairwise.jsonl in %s, skipping", run_dir)
        return None

    df_a = _load_pairwise(path_a)
    df_b = _load_pairwise(path_b)
    key = ["keyword", "effort", "sample_i", "sample_j", "order"]
    merged = df_a.merge(
        df_b,
        on=key,
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    log.info(
        "%s: aligned=%d (judge_a only=%d, judge_b only=%d)",
        run_dir.name,
        len(merged),
        len(df_a) - len(merged),
        len(df_b) - len(merged),
    )

    rows: list[dict] = []
    _, kappa_all, agree_all = _confusion_and_kappa(merged["answer_a"].to_numpy(), merged["answer_b"].to_numpy())
    rows.append(
        {
            "pair_stratum": "all",
            "n": int(len(merged)),
            "quadratic_kappa": kappa_all,
            "exact_agreement_rate": agree_all,
        }
    )
    for stratum in STRATA:
        sub = merged[merged["stratum_a"] == stratum]
        if sub.empty:
            rows.append(
                {
                    "pair_stratum": stratum,
                    "n": 0,
                    "quadratic_kappa": float("nan"),
                    "exact_agreement_rate": float("nan"),
                }
            )
            continue
        _, k, a = _confusion_and_kappa(sub["answer_a"].to_numpy(), sub["answer_b"].to_numpy())
        rows.append(
            {
                "pair_stratum": stratum,
                "n": int(len(sub)),
                "quadratic_kappa": k,
                "exact_agreement_rate": a,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/effort_diversity",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=("Override CSV destination. Defaults to <results-root>/prompt_sensitivity/cross_judge_agreement.csv."),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("analyze_prompt_sensitivity_cross_judge_agreement")

    rows: list[dict] = []
    for model in MODELS:
        for prompt, effort in PROMPT_EFFORTS:
            cell = f"{model}_{prompt}_{effort}"
            run_dir = args.results_root / "prompt_sensitivity" / cell
            cell_rows = _agreement_for_cell(run_dir, log)
            if cell_rows is None:
                continue
            for r in cell_rows:
                r["model"] = model
                r["prompt"] = prompt
                r["effort"] = effort
                r["judge_a"] = JUDGE_A.value
                r["judge_b"] = JUDGE_B.value
            rows.extend(cell_rows)

    if not rows:
        sys.exit("no cells aggregated; check pairwise.jsonl paths")

    df = (
        pd.DataFrame(rows)[
            [
                "model",
                "prompt",
                "effort",
                "pair_stratum",
                "judge_a",
                "judge_b",
                "n",
                "quadratic_kappa",
                "exact_agreement_rate",
            ]
        ]
        .sort_values(["model", "prompt", "effort", "pair_stratum"])
        .reset_index(drop=True)
    )

    print()
    print("=== Cross-judge agreement (GPT-4.1 vs Haiku 4.5), all-stratum ===")
    print(f"  {'model':<12s} {'prompt':<5s} {'effort':<5s} {'n':>6s} {'kappa':>6s} {'agree%':>7s}")
    for _, r in df[df["pair_stratum"] == "all"].iterrows():
        print(
            f"  {r['model']:<12s} {r['prompt']:<5s} {r['effort']:<5s} "
            f"{int(r['n']):>6d} {r['quadratic_kappa']:>6.3f} "
            f"{r['exact_agreement_rate'] * 100:>6.2f}%"
        )

    out_csv = args.output_csv or (args.results_root / "prompt_sensitivity" / "cross_judge_agreement.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
