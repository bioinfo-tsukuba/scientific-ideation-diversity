"""Correlate per-keyword Δdiversity vs Δquality for the effort-scaling paper.

Produces the appendix figure and table for the paper's central claim that
reasoning effort increases diversity without sacrificing per-idea quality.
The figure visualises whether Δdiversity (high − low, per keyword) co-varies
with Δquality on any of three LiveIdeaBench criteria (originality / feasibility
/ clarity) for all three generator models.

**Interpretation convention.**  We report practical effect size rather than
statistical significance.  At n ≈ 1 160–1 180 the zero-correlation test has
enormous power and would flag even practically-null cells (|r| ~ 0.1) as
"significant", which misleads readers about effect size.  The paper reads the
table off ``|r|`` and ``r^2`` directly.

Input artefacts (all under ``results/effort_diversity/``):
    phase2_{model}_1180kw_30{x4,x3}_facet_seed42/text-embedding-3-large/
        summary_by_keyword_effort.csv   # mean_pairwise_cosine_distance per (kw, effort)
    phase2_{model}_1180kw_30{x4,x3}_facet_seed42/
        cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001/
        per_keyword_effort_gpt_4_1.csv  # per-keyword quality per effort

Outputs:
    results/effort_diversity/diversity_quality_correlation.csv
    paper/figures/fig_diversity_quality_per_keyword.{pdf,png}

Usage:
    uv run python scripts/experiment/diversity/analyze_diversity_quality_correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paper.scripts._paper_style import MODEL_COLOR, MODEL_LABEL, MODEL_ORDER, apply_style  # noqa: E402

apply_style()

EFFORT_DIR = REPO_ROOT / "results" / "effort_diversity"
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_SUBDIR = "text-embedding-3-large"
CROSS_JUDGE = "cross_judge_gpt_4_1_vs_claude_haiku_4_5_20251001"
DIVERSITY_METRIC = "mean_pairwise_cosine_distance"
QUALITY_DIMENSIONS = ("orig", "feas", "clar")
QUALITY_LABELS = {"orig": "Originality", "feas": "Feasibility", "clar": "Clarity"}

_RUN_DIR: dict[str, str] = {
    "claude": "phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54": "phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini31pro": "phase2_gemini31pro_1180kw_30x3_facet_seed42",
}

N_BOOT = 1_000


def load_diversity_delta(model: str) -> pd.DataFrame:
    """Load per-keyword Δdiversity = high − low (text-embedding-3-large)."""
    path = EFFORT_DIR / _RUN_DIR[model] / EMBEDDING_SUBDIR / "summary_by_keyword_effort.csv"
    df = pd.read_csv(path)[["keyword", "effort", DIVERSITY_METRIC]]
    pivot = df.pivot_table(index="keyword", columns="effort", values=DIVERSITY_METRIC, aggfunc="first")
    return pd.DataFrame({"keyword": pivot.index, "delta_div": (pivot["high"] - pivot["low"]).to_numpy()})


def load_quality_delta(model: str) -> pd.DataFrame:
    """Load per-keyword Δquality = high − low for orig/feas/clar."""
    path = EFFORT_DIR / _RUN_DIR[model] / CROSS_JUDGE / "per_keyword_effort_gpt_4_1.csv"
    df = pd.read_csv(path).set_index("keyword")
    out = pd.DataFrame({"keyword": df.index})
    for dim in QUALITY_DIMENSIONS:
        out[f"{dim}_delta"] = (df[f"{dim}_high"] - df[f"{dim}_low"]).to_numpy()
    return out.reset_index(drop=True)


def bootstrap_pearson_ci(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    point_r = float(stats.pearsonr(x, y).statistic)
    n = len(x)
    boot = np.array([
        stats.pearsonr(x[idx := rng.integers(0, n, n)], y[idx]).statistic
        for _ in range(N_BOOT)
    ])
    return point_r, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    rng = np.random.default_rng(42)

    # Load and join data for all 3 models
    joined: dict[str, pd.DataFrame] = {}
    for model in MODEL_ORDER:
        div = load_diversity_delta(model)
        qual = load_quality_delta(model)
        merged = div.merge(qual, on="keyword", how="inner")
        joined[model] = merged
        print(f"[{model}] n={len(merged)}")

    # Compute correlations
    stat_rows: list[dict] = []
    for model in MODEL_ORDER:
        df = joined[model]
        x_div = df["delta_div"].to_numpy()
        for dim in QUALITY_DIMENSIONS:
            y_qual = df[f"{dim}_delta"].to_numpy()
            mask = np.isfinite(x_div) & np.isfinite(y_qual)
            r, ci_lo, ci_hi = bootstrap_pearson_ci(x_div, y_qual, rng)
            rho = float(stats.spearmanr(x_div[mask], y_qual[mask]).statistic)
            stat_rows.append({
                "model": model,
                "quality_dim": dim,
                "n": int(mask.sum()),
                "pearson_r": r,
                "pearson_ci_lo": ci_lo,
                "pearson_ci_hi": ci_hi,
                "spearman_rho": rho,
            })

    corr = pd.DataFrame(stat_rows)
    out_csv = EFFORT_DIR / "diversity_quality_correlation.csv"
    corr.to_csv(out_csv, index=False)
    print(f"\nCorrelations → {out_csv}")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(corr.to_string(index=False))

    # 3×3 figure: rows = quality axes, cols = models
    fig, axes = plt.subplots(
        len(QUALITY_DIMENSIONS),
        len(MODEL_ORDER),
        figsize=(10, 9),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    for col_idx, model in enumerate(MODEL_ORDER):
        df = joined[model]
        x_div = df["delta_div"].to_numpy()
        color = MODEL_COLOR[model]
        for row_idx, (dim, dim_label) in enumerate(QUALITY_LABELS.items()):
            ax = axes[row_idx][col_idx]
            y_qual = df[f"{dim}_delta"].to_numpy()
            mask = np.isfinite(x_div) & np.isfinite(y_qual)
            ax.scatter(x_div[mask], y_qual[mask], s=8, alpha=0.4, color=color, linewidths=0)
            if mask.sum() >= 3:
                m, b = np.polyfit(x_div[mask], y_qual[mask], 1)
                xr = np.array([x_div[mask].min(), x_div[mask].max()])
                ax.plot(xr, m * xr + b, color=color, lw=1.2, alpha=0.9)
            ax.axhline(0, color="#999", lw=0.6, ls="--")
            ax.axvline(0, color="#999", lw=0.6, ls="--")

            row = corr[(corr.model == model) & (corr.quality_dim == dim)].iloc[0]
            ax.text(
                0.03, 0.97,
                f"r = {row['pearson_r']:+.2f}  (r² = {row['pearson_r']**2 * 100:.1f}%)",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, lw=0),
            )
            if row_idx == 0:
                ax.set_title(MODEL_LABEL[model], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{dim_label}\nΔ quality (high−low)", fontsize=7)
            if row_idx == len(QUALITY_DIMENSIONS) - 1:
                ax.set_xlabel("Δ pair dist. (high−low)", fontsize=7)
            ax.tick_params(labelsize=6)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_diversity_quality_per_keyword.{ext}", dpi=150)
    plt.close(fig)
    print(f"\nFigure → {FIG_DIR}/fig_diversity_quality_per_keyword.{{pdf,png}}")


if __name__ == "__main__":
    main()
