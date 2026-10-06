"""Q-A pairwise judge classification (#158): per-cell × Q1/Q4 × pair-stratum
ABCD distribution, mirroring paper §4.2 fig:abcd-stratum on the prompt axis.

For each (model, prompt, effort, q_stratum, pair_stratum, judge) bucket,
counts the ABCD answers in ``llm_judge_<judge_slug>/pairwise.jsonl`` and
reports frac_A (= "completely different ideas" rate). The headline comparison
is the **top stratum** A-rate, identical to paper §4.2's main panel:

  - default-low → default-high reproduces the §4.2 effort-axis number
  - VS-low / VS-high / SSoT-low / SSoT-high give the prompt-axis cells

Pair-stratum {top, middle, bottom} and q_stratum {Q1, Q4, all} are emitted
together so cross-stratum sanity checks (e.g., bottom-stratum A-rate stays
near zero everywhere) can run off the same CSV.

Outputs: stdout summary + long-format CSV under
``results/effort_diversity/prompt_sensitivity/<model>_q1q4_pairwise_abcd.csv``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import JudgeModelName  # noqa: E402
from src.schemas.llm_judge import JudgeAnswer, JudgeResponse, PairStratum  # noqa: E402

MODELS: dict[str, dict[str, str]] = {
    "claude": {
        "phase2_dir": "phase2_claude_1180kw_30x4_facet_seed42",
        "ps_prefix": "claude",
    },
    "gpt54": {
        "phase2_dir": "phase2_gpt54_1180kw_30x4_facet_seed42",
        "ps_prefix": "gpt54",
    },
    "gemini31pro": {
        "phase2_dir": "phase2_gemini31pro_1180kw_30x3_facet_seed42",
        "ps_prefix": "gemini31pro",
    },
}

JUDGES: tuple[JudgeModelName, ...] = (
    JudgeModelName.GPT_4_1,
    JudgeModelName.CLAUDE_HAIKU_4_5,
)

LABELS: tuple[str, ...] = tuple(member.value for member in JudgeAnswer)
STRATA: tuple[str, ...] = tuple(member.value for member in PairStratum)
Q_STRATA: tuple[str, ...] = ("Q1", "Q4", "all")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


def load_responses(path: Path) -> list[JudgeResponse]:
    rows: list[JudgeResponse] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(JudgeResponse.model_validate_json(line))
    return rows


def aggregate(
    rows: list[JudgeResponse],
    *,
    q1_set: frozenset[str],
    q4_set: frozenset[str],
    excluded: frozenset[str],
    cell_prompt: str,
    cell_effort: str | None,
    judge_value: str,
) -> list[dict]:
    """Return one row per (effort, q_stratum, pair_stratum) bucket.

    ``cell_effort=None`` keeps the response's own effort (used for the
    phase2 default cell which spans low/medium/high in a single dir);
    a non-None ``cell_effort`` enforces it (used for the prompt-sensitivity
    cells which are each effort-pinned by construction).
    """
    buckets: dict[tuple[str, str, str], list[str]] = {}
    for r in rows:
        kw = r.request.pair.keyword
        if kw in excluded:
            continue
        in_q1 = kw in q1_set
        in_q4 = kw in q4_set
        if not (in_q1 or in_q4):
            continue
        effort = r.request.pair.effort.value
        if cell_effort is not None and effort != cell_effort:
            # Defensive: prompt-sensitivity cells are effort-pinned, but skip
            # rather than assume.
            continue
        pair_stratum = r.request.pair.stratum.value
        answer = r.answer.value
        # Each kw contributes to its own q-stratum (Q1 xor Q4 by construction)
        # and to the q_stratum="all" aggregation over Q1∪Q4.
        q_strata = ["Q1"] if in_q1 else ["Q4"]
        q_strata.append("all")
        for q_stratum in q_strata:
            buckets.setdefault((effort, q_stratum, pair_stratum), []).append(answer)

    out: list[dict] = []
    for (effort, q_stratum, pair_stratum), labels in buckets.items():
        n = len(labels)
        counts = Counter(labels)
        row = {
            "prompt": cell_prompt,
            "effort": effort,
            "q_stratum": q_stratum,
            "pair_stratum": pair_stratum,
            "judge": judge_value,
            "n": n,
        }
        for label in LABELS:
            row[f"count_{label}"] = counts.get(label, 0)
            row[f"frac_{label}"] = counts.get(label, 0) / n if n else 0.0
        row["frac_A_or_B"] = (counts.get("A", 0) + counts.get("B", 0)) / n if n else 0.0
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude", choices=list(MODELS.keys()))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/effort_diversity",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV destination. Defaults to <results-root>/prompt_sensitivity/<model>_q1q4_pairwise_abcd.csv."
        ),
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all judgments for KW from every cell before aggregation. "
            "Mirrors the same flag on analyze_prompt_sensitivity_q1q4_*.py "
            "(e.g., bioterrorism on gpt54 SSoT cells via OpenAI safety "
            "filter)."
        ),
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    excluded: frozenset[str] = frozenset(args.exclude_keyword)

    q1q4 = pd.read_csv(args.results_root / "prompt_sensitivity" / f"q1_q4_{args.model}.csv")
    q1_set = frozenset(q1q4.loc[q1q4["stratum"] == "Q1", "keyword"])
    q4_set = frozenset(q1q4.loc[q1q4["stratum"] == "Q4", "keyword"])
    if excluded:
        q1_set = q1_set - excluded
        q4_set = q4_set - excluded
        print(
            f"Excluded {len(excluded)} keyword(s) from Q1/Q4 set ({sorted(excluded)}); "
            f"kept Q1={len(q1_set)} Q4={len(q4_set)}"
        )

    # (cell_prompt, cell_effort_or_none, run_dir)
    # cell_effort=None for phase2 (default) cell which spans all efforts;
    # the prompt-sensitivity cells are each effort-pinned.
    cells: list[tuple[str, str | None, Path]] = [
        ("default", None, args.results_root / cfg["phase2_dir"]),
    ]
    for prompt_kind in ("vs", "ssot"):
        for eff in ("low", "high"):
            cells.append(
                (
                    prompt_kind,
                    eff,
                    args.results_root / "prompt_sensitivity" / f"{cfg['ps_prefix']}_{prompt_kind}_{eff}",
                )
            )

    rows: list[dict] = []
    for cell_prompt, cell_effort, run_dir in cells:
        for judge in JUDGES:
            slug = _slugify(judge.value)
            jsonl = run_dir / f"llm_judge_{slug}" / "pairwise.jsonl"
            if not jsonl.exists():
                print(f"  [skip] missing {jsonl.relative_to(args.results_root.parent)}")
                continue
            responses = load_responses(jsonl)
            cell_rows = aggregate(
                responses,
                q1_set=q1_set,
                q4_set=q4_set,
                excluded=excluded,
                cell_prompt=cell_prompt,
                cell_effort=cell_effort,
                judge_value=judge.value,
            )
            for r in cell_rows:
                r["model"] = args.model
            rows.extend(cell_rows)
            print(
                f"  {cell_prompt:>7s}/{cell_effort or '*':<6s} "
                f"judge={judge.value:<32s} loaded={len(responses):>5d} "
                f"emitted_rows={len(cell_rows):>3d}"
            )

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no rows aggregated; check pairwise.jsonl paths")

    col_order = [
        "model",
        "prompt",
        "effort",
        "q_stratum",
        "pair_stratum",
        "judge",
        "n",
        *(f"count_{lbl}" for lbl in LABELS),
        *(f"frac_{lbl}" for lbl in LABELS),
        "frac_A_or_B",
    ]
    df = df[col_order].sort_values(["judge", "prompt", "effort", "q_stratum", "pair_stratum"]).reset_index(drop=True)

    print()
    print(f"=== Top-stratum A-rate by cell × Q-stratum × judge ({args.model}) ===")
    top = df[df["pair_stratum"] == "top"]
    for judge_val in sorted(top["judge"].unique()):
        print(f"\njudge={judge_val}")
        print(f"  {'prompt':<8s} {'effort':<7s} {'q_strat':<8s} {'n':>6s} {'%A':>6s}")
        sub = top[top["judge"] == judge_val]
        for _, r in sub.iterrows():
            print(
                f"  {r['prompt']:<8s} {r['effort']:<7s} {r['q_stratum']:<8s} "
                f"{int(r['n']):>6d} {r['frac_A'] * 100:>5.1f}%"
            )

    if args.output_csv is not None:
        out_csv = args.output_csv
    else:
        out_csv = args.results_root / "prompt_sensitivity" / f"{args.model}_q1q4_pairwise_abcd.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
