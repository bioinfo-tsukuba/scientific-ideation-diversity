"""Q-C analysis (#133): F3 quality regression-to-mean generality on the
per-model bimodal Q1/Q4 keyword subsets, mirroring paper §4.2's F3 finding.

For each (prompt, effort) cell in the prompt-sensitivity experiment, compute
per-keyword quality delta (intervention - baseline) for each of originality
/ feasibility / clarity (and the simple mean), then report per-stratum
(Q1, Q4) mean Δ ± bootstrap 95% CI, paired Wilcoxon p (within stratum,
H0: Δ = 0), and Mann-Whitney U p (between strata, H0: same Δ distribution).

Two strata definitions are supported (``--stratify-by``):

* ``baseline-quality`` (default — the F3 test): within the 100 kw subset,
  rank kw by baseline (default-low) average quality from this judge, then
  Q1 = bottom 50, Q4 = top 50. Asks: do low-baseline-quality kw improve
  (Δ > 0) and high-baseline-quality kw degrade (Δ < 0)? The strata are
  judge-dependent because the ranking variable is the judge's own scale.
* ``pair-distance`` (parallel to Q-A): use ``q1_q4_<model>.csv`` from
  PR #138 (Q1 = mode-collapse-prone, Q4 = already-diverse, by baseline
  pair distance). Asks the related but distinct question: does the
  pair-distance Q1/Q4 split also predict Δ quality?

Quality scores per sample come from ``quality.jsonl`` (1-10 scale, integers,
produced by ``scripts/experiment/llm_judge/llm_judge_quality.py``). Per-cell per-kw
quality is the mean across the 30 samples; phase2 default cells use
``llm_judge_quality_<slug>/per_keyword_effort.csv`` columns
``{axis}_{effort_suffix}`` directly.

Outputs: stdout table + CSV at
``results/effort_diversity/prompt_sensitivity/<model>_q1q4_quality__<judge_slug>__<stratify>.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import JudgeModelName  # noqa: E402


def _judge_dir_slug(judge_model_value: str) -> str:
    """Slugify matching `_slugify` in scripts/experiment/llm_judge/llm_judge_quality.py
    so this analyzer resolves the same `llm_judge_quality_<slug>/` subdir
    that the quality judge wrote (e.g., `gpt-4.1` -> `gpt_4_1`,
    `claude-haiku-4-5-20251001` -> `claude_haiku_4_5_20251001`).

    `src.text.slugify` uses hyphens for non-alnum runs and would yield
    `gpt-41` here, which diverges from the on-disk layout.
    """
    return re.sub(r"[^a-z0-9]+", "_", judge_model_value.lower()).strip("_") or "judge"


# Phase2 dir + prompt_sensitivity prefix per generation model. Mirrors Q-A.
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

# Effort tier -> CSV-column suffix used by aggregate_llm_judge_quality_per_keyword.py.
_EFFORT_COLUMN_SUFFIX: dict[str, str] = {
    "none": "none",
    "low": "low",
    "medium": "med",
    "high": "high",
}

# Long axis name (in quality.jsonl) -> short tag (in per_keyword_effort.csv).
_AXES_LONG: tuple[str, ...] = ("originality", "feasibility", "clarity")
_AXIS_SHORT: dict[str, str] = {"originality": "orig", "feasibility": "feas", "clarity": "clar"}

# (label, lhs_kind, lhs_effort, rhs_kind, rhs_effort) — same shape as Q-A.
GAIN_DEFINITIONS: list[tuple[str, str, str, str, str]] = [
    ("effort low->high (reference, paper §4.2)", "phase2", "high", "phase2", "low"),
    ("VS-low - default-low", "vs", "low", "phase2", "low"),
    ("VS-high - default-high", "vs", "high", "phase2", "high"),
    ("VS-high - default-low (combined)", "vs", "high", "phase2", "low"),
    ("SSoT-low - default-low", "ssot", "low", "phase2", "low"),
    ("SSoT-high - default-high", "ssot", "high", "phase2", "high"),
    ("SSoT-high - default-low (combined)", "ssot", "high", "phase2", "low"),
]

JUDGE_DIR_PREFIX = "llm_judge_quality_"


def load_phase2_quality(phase2_dir: Path, judge_slug: str) -> pd.DataFrame:
    """Wide per-kw quality from the phase2 default-prompt cell.

    ``per_keyword_effort.csv`` has one row per kw with ``{axis}_{effort_suffix}``
    columns produced by ``aggregate_llm_judge_quality_per_keyword.py``. Returns a
    long-format DataFrame with columns ``keyword, effort, orig, feas, clar``.
    """
    csv_path = phase2_dir / f"{JUDGE_DIR_PREFIX}{judge_slug}" / "per_keyword_effort.csv"
    if not csv_path.exists():
        raise SystemExit(f"phase2 per_keyword_effort.csv not found for judge_slug={judge_slug!r}: {csv_path}")
    wide = pd.read_csv(csv_path)
    rows: list[dict] = []
    for effort, suffix in _EFFORT_COLUMN_SUFFIX.items():
        col_orig, col_feas, col_clar = f"orig_{suffix}", f"feas_{suffix}", f"clar_{suffix}"
        if not all(c in wide.columns for c in (col_orig, col_feas, col_clar)):
            continue
        for _, row in wide.iterrows():
            rows.append(
                {
                    "keyword": row["keyword"],
                    "effort": effort,
                    "orig": float(row[col_orig]),
                    "feas": float(row[col_feas]),
                    "clar": float(row[col_clar]),
                }
            )
    return pd.DataFrame(rows)


def load_cell_quality(quality_jsonl: Path) -> pd.DataFrame:
    """Per-(kw, effort, sample) quality from a single prompt-sensitivity cell.

    Aggregated to per-(kw, effort) mean before being merged with the phase2
    long-format frame. Each line of quality.jsonl has the per-sample scores
    keyed by axis name (originality / feasibility / clarity, 1-10).
    """
    rows: list[dict] = []
    with open(quality_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            target = payload["request"]["target"]
            scores = payload["scores"]
            rows.append(
                {
                    "keyword": target["keyword"],
                    "effort": target["effort"],
                    "orig": float(scores["originality"]),
                    "feas": float(scores["feasibility"]),
                    "clar": float(scores["clarity"]),
                }
            )
    df = pd.DataFrame(rows)
    return df.groupby(["keyword", "effort"])[["orig", "feas", "clar"]].mean().reset_index()


def bootstrap_mean_ci(
    values: np.ndarray, *, rng: np.random.Generator, n_boot: int = 10_000
) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude", choices=list(MODELS.keys()))
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.GPT_4_1,
        choices=[
            JudgeModelName.GPT_4_1,
            JudgeModelName.CLAUDE_HAIKU_4_5,
        ],
        metavar="{gpt-4.1,claude-haiku-4-5-20251001}",
        help=f"Judge model whose quality.jsonl to read. Default: {JudgeModelName.GPT_4_1.value}",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/effort_diversity",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument(
        "--stratify-by",
        choices=("baseline-quality", "pair-distance"),
        default="baseline-quality",
        help=(
            "Q1/Q4 stratification. `baseline-quality` (default, F3 test): "
            "rank the 100 kw by avg baseline quality from this judge, "
            "Q1=bottom 50, Q4=top 50. `pair-distance`: use q1_q4_<model>.csv "
            "(PR #138, mode-collapse-prone vs already-diverse)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV destination. Defaults to "
            "<results-root>/prompt_sensitivity/"
            "<model>_q1q4_quality__<judge_slug>__<stratify>.csv"
        ),
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all rows for KW from the Q1/Q4 set before computing deltas. "
            "Use for keywords whose generation yield is below the per-cell "
            "target (e.g., bioterrorism on gpt54 SSoT cells via OpenAI safety "
            "filter) so analyses across cells stay on the same keyword support."
        ),
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    rng = np.random.default_rng(args.seed)
    judge_slug = _judge_dir_slug(args.judge_model.value)
    excluded_keywords: tuple[str, ...] = tuple(sorted(set(args.exclude_keyword)))

    # Per-cell long-format quality DataFrames (columns: keyword, effort, orig, feas, clar).
    # Phase2 first because the baseline-quality stratification needs it.
    cells: dict[tuple[str, str], pd.DataFrame | None] = {}
    phase2_long = load_phase2_quality(args.results_root / cfg["phase2_dir"], judge_slug)
    for eff in ("low", "high"):
        cells[("phase2", eff)] = phase2_long[phase2_long["effort"] == eff].copy()

    # Q1/Q4 strata definition.
    kw_subset = pd.read_csv(args.results_root / "prompt_sensitivity" / f"q1_q4_{args.model}.csv")
    if excluded_keywords:
        n_before = len(kw_subset)
        kw_subset = kw_subset[~kw_subset["keyword"].isin(excluded_keywords)].reset_index(drop=True)
        print(
            f"Excluded {n_before - len(kw_subset)} keywords from kw subset "
            f"({list(excluded_keywords)}); kept {len(kw_subset)}"
        )

    if args.stratify_by == "pair-distance":
        # PR #138's Q1/Q4 split (mode-collapse-prone vs already-diverse).
        q1q4 = kw_subset[["keyword", "stratum"]].copy()
    else:
        # F3 test: rank by baseline (default-low) avg quality from this judge.
        baseline_low = phase2_long[phase2_long["effort"] == "low"][["keyword", "orig", "feas", "clar"]].copy()
        baseline_low["baseline_quality_avg"] = baseline_low[["orig", "feas", "clar"]].mean(axis=1)
        merged = kw_subset.merge(baseline_low[["keyword", "baseline_quality_avg"]], on="keyword", how="left")
        n_missing = int(merged["baseline_quality_avg"].isna().sum())
        if n_missing:
            print(f"WARN: {n_missing} kw lack baseline quality (judge={judge_slug}); dropping")
            merged = merged.dropna(subset=["baseline_quality_avg"]).reset_index(drop=True)
        # Median split within the kw subset; bottom = Q1, top = Q4. Equal halves
        # avoid the median-tie ambiguity by sorting + assigning by row index.
        merged = merged.sort_values("baseline_quality_avg", kind="mergesort").reset_index(drop=True)
        half = len(merged) // 2
        merged["stratum"] = ["Q1"] * half + ["Q4"] * (len(merged) - half)
        q1q4 = merged[["keyword", "stratum", "baseline_quality_avg"]]
        print(
            f"baseline-quality strata: "
            f"Q1 mean baseline quality = {merged.loc[merged.stratum == 'Q1', 'baseline_quality_avg'].mean():.3f}, "
            f"Q4 mean baseline quality = {merged.loc[merged.stratum == 'Q4', 'baseline_quality_avg'].mean():.3f}"
        )

    for cell_kind in ("vs", "ssot"):
        for eff in ("low", "high"):
            cell_dir = args.results_root / "prompt_sensitivity" / f"{cfg['ps_prefix']}_{cell_kind}_{eff}"
            quality_path = cell_dir / f"{JUDGE_DIR_PREFIX}{judge_slug}" / "quality.jsonl"
            cells[(cell_kind, eff)] = load_cell_quality(quality_path) if quality_path.exists() else None

    rows: list[dict] = []
    for label, lhs_kind, lhs_eff, rhs_kind, rhs_eff in GAIN_DEFINITIONS:
        lhs = cells.get((lhs_kind, lhs_eff))
        rhs = cells.get((rhs_kind, rhs_eff))
        if lhs is None or rhs is None:
            rows.append({"label": label, "skipped": "missing cell"})
            continue
        merged = q1q4.merge(
            lhs[["keyword", "orig", "feas", "clar"]].rename(
                columns={"orig": "lhs_orig", "feas": "lhs_feas", "clar": "lhs_clar"}
            ),
            on="keyword",
            how="left",
        ).merge(
            rhs[["keyword", "orig", "feas", "clar"]].rename(
                columns={"orig": "rhs_orig", "feas": "rhs_feas", "clar": "rhs_clar"}
            ),
            on="keyword",
            how="left",
        )
        # Per-axis delta + simple unweighted mean of the three axes.
        for short in ("orig", "feas", "clar"):
            merged[f"d_{short}"] = merged[f"lhs_{short}"] - merged[f"rhs_{short}"]
        merged["d_avg"] = merged[["d_orig", "d_feas", "d_clar"]].mean(axis=1)

        for axis in ("orig", "feas", "clar", "avg"):
            d_col = f"d_{axis}"
            stats: dict[str, float | str] = {
                "label": label,
                "axis": axis,
                "n_q1": int(merged[merged["stratum"] == "Q1"][d_col].notna().sum()),
                "n_q4": int(merged[merged["stratum"] == "Q4"][d_col].notna().sum()),
            }
            for stratum in ("q1", "q4"):
                d = merged.loc[merged["stratum"] == stratum.upper(), d_col].dropna().to_numpy()
                m, lo, hi = bootstrap_mean_ci(d, rng=rng, n_boot=args.n_boot)
                stats[f"{stratum}_mean"] = m
                stats[f"{stratum}_ci_low"] = lo
                stats[f"{stratum}_ci_high"] = hi
                # Paired Wilcoxon: H0 Δ = 0 within this stratum. Skip if all zeros / too few.
                stats[f"wilcoxon_{stratum}_p"] = (
                    float(wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue)
                    if len(d) >= 2 and np.any(d != 0)
                    else float("nan")
                )
            d_q1 = merged.loc[merged["stratum"] == "Q1", d_col].dropna().to_numpy()
            d_q4 = merged.loc[merged["stratum"] == "Q4", d_col].dropna().to_numpy()
            stats["mannwhitney_q1_vs_q4_p"] = (
                float(mannwhitneyu(d_q1, d_q4, alternative="two-sided").pvalue)
                if len(d_q1) >= 1 and len(d_q4) >= 1
                else float("nan")
            )
            rows.append(stats)

    # BH-FDR correction across all tests in this model run (family = one --model call).
    _P_COLS = ("wilcoxon_q1_p", "wilcoxon_q4_p", "mannwhitney_q1_vs_q4_p")
    valid_rows = [r for r in rows if "skipped" not in r]
    idx_pval: list[tuple[int, str, float]] = [
        (i, col, r[col]) for i, r in enumerate(valid_rows) for col in _P_COLS if not np.isnan(r[col])
    ]
    if idx_pval:
        _, pvals_fdr, _, _ = multipletests([v for _, _, v in idx_pval], method="fdr_bh")
        fdr_map = {(i, col): fdr for (i, col, _), fdr in zip(idx_pval, pvals_fdr)}
    else:
        fdr_map = {}
    for i, r in enumerate(valid_rows):
        for col in _P_COLS:
            r[col.replace("_p", "_p_fdr")] = fdr_map.get((i, col), float("nan"))

    df = pd.DataFrame(rows)

    print()
    print(f"=== Q-C quality regression-to-mean on {args.model} (judge={args.judge_model.value}) ===")
    print("   Δ = lhs cell quality - rhs cell quality, per kw, on the {Q1, Q4} pair-distance strata.")
    print(f"   Bootstrap 95% CI on per-kw distributions (n_boot={args.n_boot}, seed={args.seed}).")
    print("   F3 reference (paper §4.2): Q1 improves (>0), Q4 degrades (<0) under high effort.")
    print()
    cur_label = None
    for r in rows:
        if r.get("skipped"):
            print(f"{r['label']}  [skip: {r['skipped']}]")
            cur_label = None
            continue
        if r["label"] != cur_label:
            print(f"\n{r['label']}  (n={r['n_q1']}/{r['n_q4']})")
            cur_label = r["label"]
        if r["axis"] == "avg":
            print(
                f"  {r['axis']}:  "
                f"Q1 Δ {r['q1_mean']:+.3f} [{r['q1_ci_low']:+.3f},{r['q1_ci_high']:+.3f}] | "
                f"Q4 Δ {r['q4_mean']:+.3f} [{r['q4_ci_low']:+.3f},{r['q4_ci_high']:+.3f}] | "
                f"W p Q1={r['wilcoxon_q1_p']:.1e}(FDR={r['wilcoxon_q1_p_fdr']:.1e})"
                f" Q4={r['wilcoxon_q4_p']:.1e}(FDR={r['wilcoxon_q4_p_fdr']:.1e}) | "
                f"MW p={r['mannwhitney_q1_vs_q4_p']:.1e}(FDR={r['mannwhitney_q1_vs_q4_p_fdr']:.1e})"
            )

    if args.output_csv is not None:
        out_csv = args.output_csv
    else:
        stratify_tag = args.stratify_by.replace("-", "_")
        out_csv = (
            args.results_root / "prompt_sensitivity" / f"{args.model}_q1q4_quality__{judge_slug}__{stratify_tag}.csv"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
