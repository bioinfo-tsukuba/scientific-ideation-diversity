"""Q-A primary analysis (#133): prompt-gain Q1/Q4 ratio on the per-model
bimodal Q1/Q4 keyword subsets, mirroring paper §4.2's effort-gain Q1/Q4 ratio.

For each (prompt, effort) cell in the prompt-sensitivity experiment, computes
per-keyword pair distance gain (Δ vs the appropriate baseline cell), then
reports per-stratum (Q1, Q4) mean ± bootstrap 95% CI, paired Wilcoxon p
(within stratum), Mann-Whitney U p (between strata), and the Q1/Q4 ratio
with bootstrap CI.

The headline comparison is the row labelled ``effort_low_to_high (reference)``
— the same gain paper §4.2 reports — versus the prompt-swap rows. If VS / SSoT
prompt-swap rows show Q1/Q4 ratios in the 4–8× ballpark of the effort row,
F2 (heterogeneity) is a property of diversity interventions in general; if
flat (~1×), F2 has a reasoning-effort-specific signature.

Outputs: stdout table + optional CSV under
``results/effort_diversity/prompt_sensitivity/<model>_q1q4_pair_distance.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import EmbeddingModelName  # noqa: E402
from src.text import slugify  # noqa: E402

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

# Mirrors `_SUPPORTED_EMBEDDING_MODELS` in scripts/experiment/llm_judge/llm_judge_pairwise.py:
# the three embedding-model dirs the gen pipeline writes per cell, and the
# only ones whose summary_by_keyword_effort.csv we know how to read here.
_SUPPORTED_EMBEDDING_MODELS: tuple[EmbeddingModelName, ...] = (
    EmbeddingModelName.AMAZON_TITAN_EMBED_TEXT_V2_0,
    EmbeddingModelName.SPECTER2,
    EmbeddingModelName.SPECTER2_ADHOC_QUERY,
    EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
)
PAIR_COL = "mean_pairwise_cosine_distance"

# (label, lhs_kind, lhs_effort, rhs_kind, rhs_effort)
# The "reference" effort-only row is what paper §4.2 reports — the rest are
# the new prompt-swap rows. ``phase2`` means the default-prompt cell from the
# existing phase2 run; ``vs`` / ``ssot`` are the new prompt-sensitivity cells.
GAIN_DEFINITIONS: list[tuple[str, str, str, str, str]] = [
    ("effort low->high (reference, paper §4.2)", "phase2", "high", "phase2", "low"),
    ("(VS, low) vs (default, low)", "vs", "low", "phase2", "low"),
    ("(VS, low) vs (default, high) [orthogonality]", "vs", "low", "phase2", "high"),
    ("(VS, high) vs (default, high)", "vs", "high", "phase2", "high"),
    ("(VS, high) vs (default, low) [combined]", "vs", "high", "phase2", "low"),
    ("(SSoT, low) vs (default, low)", "ssot", "low", "phase2", "low"),
    ("(SSoT, high) vs (default, high)", "ssot", "high", "phase2", "high"),
    ("(SSoT, high) vs (default, low) [combined]", "ssot", "high", "phase2", "low"),
]


def load_pair_distances_long(run_dir: Path, emb_subdir: str) -> pd.DataFrame:
    csv_path = run_dir / emb_subdir / "summary_by_keyword_effort.csv"
    df = pd.read_csv(csv_path)
    return df[["keyword", "effort", PAIR_COL]].rename(columns={PAIR_COL: "pd"})


def bootstrap_mean_ci(
    values: np.ndarray, *, rng: np.random.Generator, n_boot: int = 10_000
) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_ratio_ci(
    num: np.ndarray, den: np.ndarray, *, rng: np.random.Generator, n_boot: int = 10_000
) -> tuple[float, float, float]:
    """Q1/Q4 ratio with bootstrap CI under independent-strata resampling."""
    if len(num) == 0 or len(den) == 0:
        return float("nan"), float("nan"), float("nan")
    boot_num = rng.choice(num, size=(n_boot, len(num)), replace=True).mean(axis=1)
    boot_den = rng.choice(den, size=(n_boot, len(den)), replace=True).mean(axis=1)
    eps = 1e-9
    boot_den_safe = np.where(np.abs(boot_den) < eps, np.nan, boot_den)
    boot_ratio = boot_num / boot_den_safe
    den_mean = den.mean()
    point = num.mean() / den_mean if abs(den_mean) >= eps else float("nan")
    return float(point), float(np.nanpercentile(boot_ratio, 2.5)), float(np.nanpercentile(boot_ratio, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude", choices=list(MODELS.keys()))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/effort_diversity",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument(
        "--embedding-model",
        type=EmbeddingModelName,
        default=EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
        choices=list(_SUPPORTED_EMBEDDING_MODELS),
        metavar=f"{{{','.join(m.value for m in _SUPPORTED_EMBEDDING_MODELS)}}}",
        help=(
            "Embedding model whose summary_by_keyword_effort.csv to read. "
            f"Default: {EmbeddingModelName.TEXT_EMBEDDING_3_LARGE.value}."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV destination. Defaults to "
            "<results-root>/prompt_sensitivity/<model>_q1q4_pair_distance.csv "
            "for text-embedding-3-large (backwards-compat), or "
            "<...>/<model>_q1q4_pair_distance__<emb_slug>.csv otherwise."
        ),
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all rows for KW from the Q1/Q4 set before computing gains. "
            "Can be repeated. Use for keywords whose generation yield is below "
            "the per-cell target (e.g., bioterrorism on gpt54 SSoT cells via "
            "OpenAI safety filter) so analyses across cells stay on the same "
            "keyword support."
        ),
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    rng = np.random.default_rng(args.seed)
    excluded_keywords: tuple[str, ...] = tuple(sorted(set(args.exclude_keyword)))

    q1q4 = pd.read_csv(args.results_root / "prompt_sensitivity" / f"q1_q4_{args.model}.csv")
    if excluded_keywords:
        n_before = len(q1q4)
        q1q4 = q1q4[~q1q4["keyword"].isin(excluded_keywords)].reset_index(drop=True)
        print(f"Excluded {n_before - len(q1q4)} keywords from Q1/Q4 set ({list(excluded_keywords)}); kept {len(q1q4)}")

    emb_subdir = slugify(args.embedding_model.value)
    cells: dict[tuple[str, str], pd.DataFrame | None] = {}
    phase2_long = load_pair_distances_long(args.results_root / cfg["phase2_dir"], emb_subdir)
    for eff in ("low", "high"):
        cells[("phase2", eff)] = phase2_long[phase2_long["effort"] == eff][["keyword", "pd"]].copy()

    for cell_kind in ("vs", "ssot"):
        for eff in ("low", "high"):
            cell_dir = args.results_root / "prompt_sensitivity" / f"{cfg['ps_prefix']}_{cell_kind}_{eff}"
            csv_path = cell_dir / emb_subdir / "summary_by_keyword_effort.csv"
            cells[(cell_kind, eff)] = (
                load_pair_distances_long(cell_dir, emb_subdir)[["keyword", "pd"]] if csv_path.exists() else None
            )

    rows: list[dict] = []
    for label, lhs_kind, lhs_eff, rhs_kind, rhs_eff in GAIN_DEFINITIONS:
        lhs = cells.get((lhs_kind, lhs_eff))
        rhs = cells.get((rhs_kind, rhs_eff))
        if lhs is None or rhs is None:
            rows.append({"label": label, "skipped": "missing cell"})
            continue
        merged = q1q4.merge(lhs.rename(columns={"pd": "lhs_pd"}), on="keyword", how="left").merge(
            rhs.rename(columns={"pd": "rhs_pd"}), on="keyword", how="left"
        )
        merged["abs_gain"] = merged["lhs_pd"] - merged["rhs_pd"]
        # Per-keyword relative gain = (lhs - rhs) / rhs.  Q1/Q4 baselines
        # (the rhs cell) are well above 0.12 in this experiment, so dividing
        # is stable; we'd need to revisit if any future cell pushed kws
        # toward rhs ~ 0 (then absolute Δ stays meaningful but relative blows up).
        merged["rel_gain"] = merged["abs_gain"] / merged["rhs_pd"]
        merged = merged.dropna(subset=["abs_gain", "rel_gain"])

        q1_abs = merged.loc[merged["stratum"] == "Q1", "abs_gain"].to_numpy()
        q4_abs = merged.loc[merged["stratum"] == "Q4", "abs_gain"].to_numpy()
        q1_rel = merged.loc[merged["stratum"] == "Q1", "rel_gain"].to_numpy()
        q4_rel = merged.loc[merged["stratum"] == "Q4", "rel_gain"].to_numpy()

        abs_q1_m, abs_q1_lo, abs_q1_hi = bootstrap_mean_ci(q1_abs, rng=rng, n_boot=args.n_boot)
        abs_q4_m, abs_q4_lo, abs_q4_hi = bootstrap_mean_ci(q4_abs, rng=rng, n_boot=args.n_boot)
        abs_r_m, abs_r_lo, abs_r_hi = bootstrap_ratio_ci(q1_abs, q4_abs, rng=rng, n_boot=args.n_boot)
        rel_q1_m, rel_q1_lo, rel_q1_hi = bootstrap_mean_ci(q1_rel, rng=rng, n_boot=args.n_boot)
        rel_q4_m, rel_q4_lo, rel_q4_hi = bootstrap_mean_ci(q4_rel, rng=rng, n_boot=args.n_boot)
        rel_r_m, rel_r_lo, rel_r_hi = bootstrap_ratio_ci(q1_rel, q4_rel, rng=rng, n_boot=args.n_boot)

        # Wilcoxon and Mann-Whitney p-values are computed on absolute gain only:
        # both tests are non-parametric / rank-based, so dividing by rhs_pd
        # (a positive constant per kw) only rescales each rank but does NOT
        # change the within-stratum sign-rank or between-stratum rank-sum, so
        # the p-values are identical on rel_gain.
        wq1_p = wilcoxon(q1_abs).pvalue if len(q1_abs) >= 6 else float("nan")
        wq4_p = wilcoxon(q4_abs).pvalue if len(q4_abs) >= 6 else float("nan")
        mw_p = (
            mannwhitneyu(q1_abs, q4_abs, alternative="two-sided").pvalue
            if len(q1_abs) >= 1 and len(q4_abs) >= 1
            else float("nan")
        )

        rows.append(
            {
                "label": label,
                "n_q1": len(q1_abs),
                "n_q4": len(q4_abs),
                "abs_q1_mean": abs_q1_m,
                "abs_q1_ci_low": abs_q1_lo,
                "abs_q1_ci_high": abs_q1_hi,
                "abs_q4_mean": abs_q4_m,
                "abs_q4_ci_low": abs_q4_lo,
                "abs_q4_ci_high": abs_q4_hi,
                "abs_ratio_q1_over_q4": abs_r_m,
                "abs_ratio_ci_low": abs_r_lo,
                "abs_ratio_ci_high": abs_r_hi,
                "rel_q1_mean": rel_q1_m,
                "rel_q1_ci_low": rel_q1_lo,
                "rel_q1_ci_high": rel_q1_hi,
                "rel_q4_mean": rel_q4_m,
                "rel_q4_ci_low": rel_q4_lo,
                "rel_q4_ci_high": rel_q4_hi,
                "rel_ratio_q1_over_q4": rel_r_m,
                "rel_ratio_ci_low": rel_r_lo,
                "rel_ratio_ci_high": rel_r_hi,
                "wilcoxon_q1_p": wq1_p,
                "wilcoxon_q4_p": wq4_p,
                "mannwhitney_q1_vs_q4_p": mw_p,
            }
        )

    # BH-FDR correction across all tests in this model run (family = one --model call).
    # Rows marked "skipped" occur only when a gain definition references a missing
    # prompt cell. They have no p-values, so they are excluded from the FDR family.
    # NaN p-values can still arise from insufficient sample sizes.
    p_cols = ("wilcoxon_q1_p", "wilcoxon_q4_p", "mannwhitney_q1_vs_q4_p")
    skipped_rows = [r for r in rows if "skipped" in r]
    valid_rows = [r for r in rows if "skipped" not in r]
    idx_pval: list[tuple[int, str, float]] = [
        (i, col, r[col])
        for i, r in enumerate(valid_rows)
        for col in p_cols
        if not np.isnan(r[col])
    ]
    if idx_pval:
        _, pvals_fdr, _, _ = multipletests([v for _, _, v in idx_pval], method="fdr_bh")
        fdr_map = {(i, col): fdr for (i, col, _), fdr in zip(idx_pval, pvals_fdr)}
    else:
        fdr_map = {}
    for i, r in enumerate(valid_rows):
        for col in p_cols:
            r[col.replace("_p", "_p_fdr")] = fdr_map.get((i, col), float("nan"))

    df = pd.DataFrame(rows)

    print()
    print(f"=== Q-A primary: prompt-gain Q1/Q4 on {args.model} (pair distance) ===")
    print(
        f"   Bootstrap 95% CI (n_boot={args.n_boot}, seed={args.seed}). "
        "abs Δ = lhs - rhs (cosine units); rel Δ = abs Δ / rhs_baseline (% units, per-kw avg). "
        "Wilcoxon / MW p reported once: rel Δ scaling is rank-preserving so p-values are identical."
    )
    print(
        f"   BH-FDR family: {len(idx_pval)} p-values across {len(valid_rows)} condition-contrast rows; "
        f"skipped condition-contrast rows={len(skipped_rows)}."
    )
    print()
    abs_fmt = (
        "  abs: Q1 {:+.3f} [{:+.3f},{:+.3f}] | Q4 {:+.3f} [{:+.3f},{:+.3f}] | ratio {:>+6.2f}x [{:>+6.2f},{:>+6.2f}]"
    )
    rel_fmt = (
        "  rel: Q1 {:+.0%} [{:+.0%},{:+.0%}] | Q4 {:+.0%} [{:+.0%},{:+.0%}] | ratio {:>+6.2f}x [{:>+6.2f},{:>+6.2f}]"
    )
    for r in rows:
        if r.get("skipped"):
            print(f"{r['label']} [skip: {r['skipped']}]")
            continue
        print(
            f"{r['label']}  (n={r['n_q1']}/{r['n_q4']})  "
            f"W p Q1={r['wilcoxon_q1_p']:.1e}(FDR={r['wilcoxon_q1_p_fdr']:.1e}) "
            f"Q4={r['wilcoxon_q4_p']:.1e}(FDR={r['wilcoxon_q4_p_fdr']:.1e})  "
            f"MW p={r['mannwhitney_q1_vs_q4_p']:.1e}(FDR={r['mannwhitney_q1_vs_q4_p_fdr']:.1e})"
        )
        print(
            abs_fmt.format(
                r["abs_q1_mean"],
                r["abs_q1_ci_low"],
                r["abs_q1_ci_high"],
                r["abs_q4_mean"],
                r["abs_q4_ci_low"],
                r["abs_q4_ci_high"],
                r["abs_ratio_q1_over_q4"],
                r["abs_ratio_ci_low"],
                r["abs_ratio_ci_high"],
            )
        )
        print(
            rel_fmt.format(
                r["rel_q1_mean"],
                r["rel_q1_ci_low"],
                r["rel_q1_ci_high"],
                r["rel_q4_mean"],
                r["rel_q4_ci_low"],
                r["rel_q4_ci_high"],
                r["rel_ratio_q1_over_q4"],
                r["rel_ratio_ci_low"],
                r["rel_ratio_ci_high"],
            )
        )
        print()

    if args.output_csv is not None:
        out_csv = args.output_csv
    else:
        # Backwards-compatible default for text-embedding-3-large; tag with the
        # embedding slug otherwise so multiple embedding-model runs co-exist
        # in the same directory.
        if args.embedding_model is EmbeddingModelName.TEXT_EMBEDDING_3_LARGE:
            out_name = f"{args.model}_q1q4_pair_distance.csv"
        else:
            out_name = f"{args.model}_q1q4_pair_distance__{emb_subdir}.csv"
        out_csv = args.results_root / "prompt_sensitivity" / out_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
