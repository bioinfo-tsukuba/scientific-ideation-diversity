"""Per-idea quality stays flat under both judges.

The diversity-rises message is already established by Fig 1 and
the prompt-axis figures, so this figure focuses on the quality
side: under both the primary GPT-4.1 judge and the secondary
Claude Haiku 4.5 judge, the per-idea rating (mean of originality,
feasibility, clarity on the 1-10 scale) is approximately flat
across reasoning-effort tiers.

Single panel, 1-column width: 3 models x 2 judges = 6 lines on
the same x = (low, medium, high) grid. Color encodes the
generation model, linestyle encodes the judge model (GPT-4.1
solid, Claude Haiku 4.5 dashed). All points share the same marker
since the x-axis labels already identify each effort tier
unambiguously. The y-axis spans the upper half of the 1-10 scale
so the small absolute variation across efforts reads as flat
against the fraction of the scale the data actually inhabits,
and so the two judges remain on the same axis even though Haiku
assigns systematically lower mean ratings than GPT-4.1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _paper_style import (
    MODEL_COLOR,
    MODEL_LABEL,
    MODEL_ORDER,
    REASONING_EFFORTS,
    apply_style,
    soft_grid,
)
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "claude": REPO_ROOT / "results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54": REPO_ROOT / "results/effort_diversity/phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini31pro": REPO_ROOT / "results/effort_diversity/phase2_gemini31pro_1180kw_30x3_facet_seed42",
}


JUDGE_DIR = {
    "gpt_4_1": "llm_judge_quality_gpt_4_1",
    "claude_haiku_4_5": "llm_judge_quality_claude_haiku_4_5_20251001",
}


def load_quality_mean(run_dir: Path, judge: str) -> dict[str, float]:
    out: dict[str, float] = {}
    csv_path = run_dir / JUDGE_DIR[judge] / "by_effort.csv"
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["effort"] not in REASONING_EFFORTS:
                continue
            if not row["orig_mean"]:
                continue
            out[row["effort"]] = (float(row["orig_mean"]) + float(row["feas_mean"]) + float(row["clar_mean"])) / 3.0
    return out


JUDGE_STYLE = {
    "gpt_4_1": {"linestyle": "-", "label": "GPT-4.1 judge"},
    "claude_haiku_4_5": {"linestyle": "--", "label": "Claude Haiku 4.5 judge"},
}


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(REASONING_EFFORTS))

    quality = {
        judge: {m: [load_quality_mean(RUNS[m], judge)[e] for e in REASONING_EFFORTS] for m in MODEL_ORDER}
        for judge in JUDGE_STYLE
    }

    for judge, style in JUDGE_STYLE.items():
        for model in MODEL_ORDER:
            ys = quality[judge][model]
            color = MODEL_COLOR[model]
            ax.plot(
                x,
                ys,
                linestyle=style["linestyle"],
                color=color,
                linewidth=1.6,
                alpha=0.95,
                zorder=2,
            )
            ax.scatter(
                x,
                ys,
                s=36,
                marker="o",
                c=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(REASONING_EFFORTS))
    ax.set_xlim(-0.4, len(REASONING_EFFORTS) - 0.6)
    ax.set_ylim(5.0, 10.0)
    ax.set_yticks([5, 6, 7, 8, 9, 10])
    ax.set_xlabel("Reasoning effort", labelpad=6)
    ax.set_ylabel("Per-idea quality (1-10)", labelpad=6)
    ax.tick_params(axis="both", which="major", length=3.5, labelsize=8.5)
    soft_grid(ax, x=False, y=True)

    # Three legends, stacked compactly below the panel: model (color),
    # judge (linestyle), and effort tier (marker shape).
    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR[m],
            linewidth=1.8,
            marker="o",
            markersize=5,
            markerfacecolor=MODEL_COLOR[m],
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=MODEL_LABEL[m],
        )
        for m in MODEL_ORDER
    ]
    judge_handles = [
        Line2D(
            [0],
            [0],
            color="#444",
            linewidth=1.6,
            linestyle=style["linestyle"],
            label=style["label"],
        )
        for style in JUDGE_STYLE.values()
    ]
    # Two rows of legends below the panel: model (color) and
    # judge (linestyle). Effort tier is read off the x-axis labels
    # directly, so encoding it in marker shape would be redundant.
    legend_kwargs = dict(
        loc="lower center",
        fontsize=7.4,
        handletextpad=0.5,
        columnspacing=1.2,
        frameon=False,
    )
    leg_models = fig.legend(
        handles=model_handles,
        ncols=len(model_handles),
        bbox_to_anchor=(0.5, 0.06),
        **legend_kwargs,
    )
    leg_judges = fig.legend(
        handles=judge_handles,
        ncols=len(judge_handles),
        bbox_to_anchor=(0.5, 0.00),
        **legend_kwargs,
    )
    fig.add_artist(leg_models)
    fig.add_artist(leg_judges)

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out = FIG_DIR / "quality_two_judges.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
