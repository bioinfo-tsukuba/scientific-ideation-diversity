"""LiveIdeaBench full fluency-score aggregation.

LiveIdeaBench's pairwise rubric labels each idea pair with one of four
ordered labels A / B / C / D (A = completely different ideas,
D = academically identical) and aggregates them into a single fluency
score by mapping A -> 10, B -> 7, C -> 4, D -> 1 and averaging:

    fluency = (10 * count_A + 7 * count_B + 4 * count_C + 1 * count_D) / n

Higher fluency = more diverse pairs on average.

The body §4.1 currently reports only the top-distance-bin A-rate slice
(fraction of pairs labeled A in the top distance bin); the full
LiveIdeaBench fluency aggregation pools all three distance bins
(top, middle, bottom) and uses the four-weight scoring above. This
script computes the full aggregation per (idea-generation model,
judge model, reasoning-effort level) for the K_full / default-prompt
phase2 runs, alongside the corresponding low -> high lift.

Reads:
- results/effort_diversity/phase2_{model}_*/llm_judge_{judge}/abcd_by_stratum_effort.csv

Output:
- stdout: a flat CSV with columns
  (model, judge, effort, n_pairs, fluency_score, frac_A, frac_B, frac_C, frac_D)
  followed by per-(model, judge) low -> high lift summary lines.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results/effort_diversity"

PHASE2_RUNS = {
    "Claude Sonnet 4.6": "phase2_claude_1180kw_30x4_facet_seed42",
    "GPT-5.4": "phase2_gpt54_1180kw_30x4_facet_seed42",
    "Gemini 3.1 Pro": "phase2_gemini31pro_1180kw_30x3_facet_seed42",
}

JUDGE_DIRS = {
    "GPT-4.1": "llm_judge_gpt_4_1",
    "Claude Haiku 4.5": "llm_judge_claude_haiku_4_5_20251001",
}

# LiveIdeaBench fluency-score weights (A is "most different", D is "most identical").
WEIGHTS = {"A": 10, "B": 7, "C": 4, "D": 1}

EFFORTS = ("low", "medium", "high")


def aggregate(run_dir: Path) -> dict[str, dict]:
    """Return {effort: {n, count_A, count_B, count_C, count_D, fluency, frac_*}}."""
    csv_path = run_dir / "abcd_by_stratum_effort.csv"
    out: dict[str, dict] = {e: {"n": 0, **{f"count_{label}": 0 for label in WEIGHTS}} for e in EFFORTS}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            effort = row["effort"]
            if effort not in EFFORTS:
                continue
            out[effort]["n"] += int(row["n"])
            for label in WEIGHTS:
                out[effort][f"count_{label}"] += int(row[f"count_{label}"])
    for effort, d in out.items():
        n = d["n"]
        d["fluency"] = sum(WEIGHTS[label] * d[f"count_{label}"] for label in WEIGHTS) / n if n else float("nan")
        for label in WEIGHTS:
            d[f"frac_{label}"] = d[f"count_{label}"] / n if n else float("nan")
    return out


def main() -> None:
    print("model,judge,effort,n_pairs,fluency,frac_A,frac_B,frac_C,frac_D")
    summaries: list[tuple[str, str, float, float, float]] = []
    for model_label, run_slug in PHASE2_RUNS.items():
        for judge_label, judge_slug in JUDGE_DIRS.items():
            run_dir = RESULTS_DIR / run_slug / judge_slug
            agg = aggregate(run_dir)
            for effort in EFFORTS:
                d = agg[effort]
                print(
                    f"{model_label},{judge_label},{effort},{d['n']},"
                    f"{d['fluency']:.3f},{d['frac_A']:.4f},{d['frac_B']:.4f},"
                    f"{d['frac_C']:.4f},{d['frac_D']:.4f}"
                )
            f_low = agg["low"]["fluency"]
            f_high = agg["high"]["fluency"]
            summaries.append((model_label, judge_label, f_low, f_high, f_high - f_low))

    print("\n# low -> high fluency lift summary (LiveIdeaBench A->10/B->7/C->4/D->1)")
    print("model,judge,fluency_low,fluency_high,lift")
    for model_label, judge_label, lo, hi, lift in summaries:
        print(f"{model_label},{judge_label},{lo:.3f},{hi:.3f},{lift:+.3f}")


if __name__ == "__main__":
    main()
