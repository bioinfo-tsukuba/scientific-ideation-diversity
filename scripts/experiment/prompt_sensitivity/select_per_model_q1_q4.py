"""Per-model Q1/Q4 quartile-random keyword selection for the prompt-sensitivity test (#133).

For each generation model, the existing phase2 baseline pair distance
(default prompt × low effort × text-embedding-3-large combined) defines
4 quartiles of the keyword distribution. We then sample, with a fixed
seed, ``n`` keywords uniformly at random from the Q1 (lowest 25%) and
Q4 (highest 25%) quartiles.

This matches paper §4.2's stratification axis exactly: the same Q1/Q4
populations whose mean low→high effort gain ratio is 4-8× across the
three models. The prompt-sensitivity experiment then measures the
default→VS / default→SSoT gain on the same Q1/Q4 populations, so
"prompt gain Q1/Q4 ratio" can be compared apples-to-apples with the
"effort gain Q1/Q4 ratio" reported in §4.2.

Estimator note: random sampling without replacement of ``n`` keywords
from each quartile yields an **unbiased estimator** of the quartile's
population mean (under finite-population SRS theory). With ``n = 50``
and quartile size ~290, paired Wilcoxon power for d=1.5 is >0.99 even
after Bonferroni correction (~25 tests, α=0.002).

Outputs one CSV per model with columns:
  category, keyword, baseline_pair_distance, stratum, generation_model
where stratum ∈ {Q1, Q4}.

Usage:
    uv run python scripts/experiment/prompt_sensitivity/select_per_model_q1_q4.py \\
        --phase2-root results/effort_diversity \\
        --n-per-stratum 50 \\
        --seed 42 \\
        --output-dir results/effort_diversity/prompt_sensitivity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PHASE2_DIRS: dict[str, str] = {
    "Claude Sonnet 4.6": "phase2_claude_1180kw_30x4_facet_seed42",
    "GPT-5.4": "phase2_gpt54_1180kw_30x4_facet_seed42",
    "Gemini 3.1 Pro": "phase2_gemini31pro_1180kw_30x3_facet_seed42",
}

# Stratification axis is fixed by design to match paper §4.2:
#   - baseline effort: low (the same axis on which §4.2 reports Q1/Q4 quartile gains)
#   - embedding: text-embedding-3-large combined (the §4.2 primary axis)
# These are not parametrized; changing them would break the apples-to-apples
# comparison with §4.2's effort gain Q1/Q4 ratio.
BASELINE_EFFORT = "low"
EMBEDDING_SUBDIR = "text-embedding-3-large"


def select_for_model(
    *,
    run_dir: Path,
    n_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """Sample n_per_stratum keywords each from Q1 and Q4 quartiles.

    The keyword pool is partitioned by ``mean_pairwise_cosine_distance`` at
    ``BASELINE_EFFORT`` (= ``low``) under ``EMBEDDING_SUBDIR``
    (= ``text-embedding-3-large`` combined); Q1 = lowest 25%, Q4 = highest 25%.
    Within each pool we sample uniformly at random without replacement.
    """
    summary_csv = run_dir / EMBEDDING_SUBDIR / "summary_by_keyword_effort.csv"
    df = pd.read_csv(summary_csv)
    baseline = df[df["effort"] == BASELINE_EFFORT].copy()
    if baseline.empty:
        raise ValueError(f"no baseline rows in {summary_csv} for effort={BASELINE_EFFORT}")

    # Sort ascending by pair distance so head/tail correspond to Q1/Q4.
    baseline = baseline.sort_values("mean_pairwise_cosine_distance", ascending=True)
    n = len(baseline)
    q25 = n // 4

    q1_pool = baseline.head(q25).copy()
    q4_pool = baseline.tail(q25).copy()

    if n_per_stratum > len(q1_pool) or n_per_stratum > len(q4_pool):
        raise ValueError(
            f"n_per_stratum={n_per_stratum} exceeds quartile pool size (Q1: {len(q1_pool)}, Q4: {len(q4_pool)})"
        )

    # Use ``numpy.random.default_rng`` (modern Generator API). Two
    # ``rng.choice`` calls advance the same Generator state, so the Q1 and
    # Q4 samples are independent draws from one stream — equivalent to
    # using two separate seeds derived from ``seed``, but without going
    # through ``pandas.DataFrame.sample``'s integer-seed legacy interface.
    rng = np.random.default_rng(seed)
    q1_idx = rng.choice(len(q1_pool), size=n_per_stratum, replace=False)
    q4_idx = rng.choice(len(q4_pool), size=n_per_stratum, replace=False)
    q1 = q1_pool.iloc[q1_idx].copy()
    q4 = q4_pool.iloc[q4_idx].copy()
    q1["stratum"] = "Q1"
    q4["stratum"] = "Q4"

    out = pd.concat([q1, q4], ignore_index=True)
    out = out[["category", "keyword", "mean_pairwise_cosine_distance", "stratum"]]
    out = out.rename(columns={"mean_pairwise_cosine_distance": "baseline_pair_distance"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase2-root", type=Path, default=Path("results/effort_diversity"))
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=50,
        help="Number of keywords to sample from each quartile (Q1 and Q4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/effort_diversity/prompt_sensitivity"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for model_name, run_subdir in PHASE2_DIRS.items():
        run_dir = args.phase2_root / run_subdir
        df = select_for_model(
            run_dir=run_dir,
            n_per_stratum=args.n_per_stratum,
            seed=args.seed,
        )
        df["generation_model"] = model_name
        per_model_path = args.output_dir / f"q1_q4_{run_subdir.split('_')[1]}.csv"
        df.to_csv(per_model_path, index=False)
        logger.info(
            "[%s] wrote %d kw (Q1=%d, Q4=%d) to %s; Q1 mean baseline=%.3f, Q4 mean baseline=%.3f",
            model_name,
            len(df),
            (df.stratum == "Q1").sum(),
            (df.stratum == "Q4").sum(),
            per_model_path,
            df.loc[df.stratum == "Q1", "baseline_pair_distance"].mean(),
            df.loc[df.stratum == "Q4", "baseline_pair_distance"].mean(),
        )
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = args.output_dir / "q1_q4_all_models.csv"
    combined.to_csv(combined_path, index=False)
    logger.info("wrote combined %d rows to %s", len(combined), combined_path)

    # Cross-model overlap (post-hoc inspection): Q1/Q4 are model-specific so
    # overlap is informational only.
    print()
    print("=== Per-stratum overlap (kw selected by N models) ===")
    for stratum in ["Q1", "Q4"]:
        sub = combined[combined.stratum == stratum]
        kw_counts = sub.groupby("keyword")["generation_model"].nunique()
        print(
            f"  {stratum}: "
            f"{(kw_counts == 1).sum()} kw in 1 model, "
            f"{(kw_counts == 2).sum()} in 2, "
            f"{(kw_counts == 3).sum()} in all 3"
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
