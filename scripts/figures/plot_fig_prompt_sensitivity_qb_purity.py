"""Q-B kNN purity bar chart: full prompt x effort 2x2 + cross-diagonal.

For each of the five natural pairwise contrasts in the prompt x effort 2x2
(plus the Q-B cross-diagonal), shows the 1-NN purity across 3 embedders.
Purity = 0.5 means the two conditions are indistinguishable in embedding space; 1.0 means perfectly separable.

Layout: 2 rows (Q1 / Q4) x 3 cols (Claude / GPT-5.4 / Gemini 3.1 Pro).
Within each panel: 3 embedder groups, 5 bars per group (one per contrast).

Source: results/effort_diversity/prompt_sensitivity/cross_embedder_purity_qb.csv
        (built by paper/scripts/aggregate_cross_embedder_purity.py).

Usage:
    uv run python paper/scripts/plot_fig_prompt_sensitivity_qb_purity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _paper_style import MODEL_LABEL, MODEL_ORDER, apply_style, soft_grid  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures"
RESULTS = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"
CSV_PATH = RESULTS / "cross_embedder_purity_qb.csv"

EMBEDDER_ORDER = ("text-emb-3-large", "titan-v2", "specter2")
EMBEDDER_LABEL = {
    "text-emb-3-large": "text-embedding-3-large",
    "titan-v2": "Titan v2",
    "specter2": "SPECTER2",
}

# Twelve contrasts in the 3-prompt × 2-effort design, in display order.
# Color encodes the type of axis being crossed: greys = effort axis
# (held-prompt), blues = prompt axis (held-effort, with light/dark = low/high
# effort), navy = Q-B cross-diagonal (both axes flipped).
CONTRAST_ORDER = (
    "effort @ default",
    "effort @ VS",
    "effort @ SSoT",
    "prompt @ low (default-VS)",
    "prompt @ low (default-SSoT)",
    "prompt @ low (VS-SSoT)",
    "prompt @ high (default-VS)",
    "prompt @ high (default-SSoT)",
    "prompt @ high (VS-SSoT)",
    "Q-B (default-VS)",
    "Q-B (default-SSoT)",
    "Q-B (VS-SSoT)",
)
CONTRAST_LABEL = {
    "effort @ default": "(default, low) vs (default, high)",
    "effort @ VS": "(VS, low) vs (VS, high)",
    "effort @ SSoT": "(SSoT, low) vs (SSoT, high)",
    "prompt @ low (default-VS)": "(default, low) vs (VS, low)",
    "prompt @ low (default-SSoT)": "(default, low) vs (SSoT, low)",
    "prompt @ low (VS-SSoT)": "(VS, low) vs (SSoT, low)",
    "prompt @ high (default-VS)": "(default, high) vs (VS, high)",
    "prompt @ high (default-SSoT)": "(default, high) vs (SSoT, high)",
    "prompt @ high (VS-SSoT)": "(VS, high) vs (SSoT, high)",
    "Q-B (default-VS)": "(default, high) vs (VS, low)",
    "Q-B (default-SSoT)": "(default, high) vs (SSoT, low)",
    "Q-B (VS-SSoT)": "(VS, high) vs (SSoT, low)",
}

# Group the 12 comparisons by which axes differ between the two
# conditions: effort only, prompt only at fixed effort, or both (the
# "cross-diagonal"). Used as legend group titles so the ad-hoc "Q-B" /
# "@"-prefixed labels are no longer needed in the rendered figure.
LEGEND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Effort axis (held prompt)",
     ("effort @ default", "effort @ VS", "effort @ SSoT")),
    ("Prompt axis at low effort",
     ("prompt @ low (default-VS)", "prompt @ low (default-SSoT)", "prompt @ low (VS-SSoT)")),
    ("Prompt axis at high effort",
     ("prompt @ high (default-VS)", "prompt @ high (default-SSoT)", "prompt @ high (VS-SSoT)")),
    ("Both axes (cross-diagonal)",
     ("Q-B (default-VS)", "Q-B (default-SSoT)", "Q-B (VS-SSoT)")),
)
CONTRAST_COLOR = {
    "effort @ default": "#d9d9d9",
    "effort @ VS": "#969696",
    "effort @ SSoT": "#525252",
    "prompt @ low (default-VS)": "#deebf7",
    "prompt @ low (default-SSoT)": "#c6dbef",
    "prompt @ low (VS-SSoT)": "#9ecae1",
    "prompt @ high (default-VS)": "#6baed6",
    "prompt @ high (default-SSoT)": "#4292c6",
    "prompt @ high (VS-SSoT)": "#2171b5",
    "Q-B (default-VS)": "#08519c",
    "Q-B (default-SSoT)": "#08306b",
    "Q-B (VS-SSoT)": "#031432",
}

STRATUM_LABEL = {"Q1": r"$Q_1$", "Q4": r"$Q_4$"}


def main() -> None:
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIG_DIR / "prompt_sensitivity_qb_purity.png")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.5, 7.2),
        sharey=True,
        gridspec_kw={"hspace": 0.32, "wspace": 0.10},
    )

    n_contrasts = len(CONTRAST_ORDER)
    bar_w = 0.075
    offsets = (np.arange(n_contrasts) - (n_contrasts - 1) / 2) * bar_w
    x = np.arange(len(EMBEDDER_ORDER))

    for row_idx, stratum in enumerate(("Q1", "Q4")):
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            sub = df[(df["model"] == model) & (df["stratum"] == stratum)]

            for off, contrast in zip(offsets, CONTRAST_ORDER):
                heights, ci_lo, ci_hi = [], [], []
                for emb in EMBEDDER_ORDER:
                    row = sub[(sub["embedder"] == emb) & (sub["contrast"] == contrast)]
                    if len(row) != 1:
                        heights.append(np.nan)
                        ci_lo.append(0)
                        ci_hi.append(0)
                        continue
                    r = row.iloc[0]
                    heights.append(float(r["purity_mean"]))
                    ci_lo.append(float(r["purity_mean"]) - float(r["purity_ci_low"]))
                    ci_hi.append(float(r["purity_ci_high"]) - float(r["purity_mean"]))
                ax.bar(
                    x + off,
                    heights,
                    bar_w,
                    color=CONTRAST_COLOR[contrast],
                    edgecolor="none",
                    yerr=[ci_lo, ci_hi],
                    error_kw=dict(ecolor="#444", elinewidth=0.6, capsize=1.6),
                )

            ax.set_ylim(0.5, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [EMBEDDER_LABEL[e] for e in EMBEDDER_ORDER],
                fontsize=8.5,
            )
            if row_idx == 0:
                ax.set_title(MODEL_LABEL[model], fontsize=10.5, color="#222", pad=6)
            soft_grid(ax, x=False, y=True)

        # Y-axis label puts the metric first and the Q-quartile scope
        # second ("1-NN purity in Q_n"), so the axis reads as the
        # metric with Q_n as a scope qualifier rather than a row tag.
        axes[row_idx, 0].set_ylabel(
            f"1-NN purity in {STRATUM_LABEL[stratum]}",
            fontsize=9.5,
        )

    # Four bottom-anchored sub-legends, one per axis-difference group.
    # Group titles carry the categorical structure (effort axis / prompt
    # axis at fixed effort / cross-diagonal), so each entry only needs
    # the bare condition-pair tuple.
    # Position the four sub-legends close to figure center so the
    # whole legend block reads as one unit; the default "1 / 4 of
    # width" spacing left a large gap between adjacent groups.
    n_groups = len(LEGEND_GROUPS)
    x_positions = [0.28, 0.43, 0.57, 0.72]
    assert len(x_positions) == n_groups
    for i, (group_title, contrasts) in enumerate(LEGEND_GROUPS):
        group_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=CONTRAST_COLOR[c],
                markeredgecolor="none",
                markersize=10,
                label=CONTRAST_LABEL[c],
            )
            for c in contrasts
        ]
        leg = fig.legend(
            handles=group_handles,
            loc="lower center",
            bbox_to_anchor=(x_positions[i], -0.06),
            ncol=1,
            title=group_title,
            title_fontsize=9.0,
            fontsize=8.5,
            handletextpad=0.4,
            borderpad=0.2,
            frameon=False,
        )
        leg._legend_box.align = "left"
        if i < n_groups - 1:
            fig.add_artist(leg)

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.97])
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")
    pdf_out = args.output.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"wrote {pdf_out}")


if __name__ == "__main__":
    main()
