"""Fig (paper §4.6): diversity vs reasoning tokens per generated idea,
faceted by generation model. Plain (token, diversity) scatter — no
frontier line, no dominated-region shading, no "Pareto" framing in
visuals or legend (per-vendor reasoning-token reporting is vendor-
specific, see Methods §3.1; we therefore avoid Pareto-frontier
language and let the marker positions read directly).

Cost axis is reasoning tokens only (per-vendor: Claude ``reasoning_tokens``
from ``count_claude_reasoning_tokens.py``; GPT-5.4 ``reasoning_tokens``
subset; Gemini 3.1 Pro ``thoughts_token_count``), divided by
``n_calls × k_ideas_per_call`` to amortize VS's k=3 across ideas. Reasoning
tokens isolate the variable being scaled along the effort axis; visible
answer tokens vary with prompt-output structure (e.g., VS emits 3 ideas in
one answer) and would dilute the (cost, diversity) reading. The legacy
``generation_tokens`` (= visible + reasoning) column is still produced by
the aggregator for cross-checking but is not the headline axis here.

Visual encoding:
- panels = generation model (the kw subset is per-model, so cross-model
  comparison of diversity values is not statistically meaningful — see
  issue #133. Each panel reads on its own kw support; cross-panel
  comparisons stay qualitative). The panel title carries the model
  identity; using the per-model color from `_paper_style.MODEL_COLOR`
  inside the panel would be redundant with the title and crowds out the
  prompt encoding, so this figure uses a model-agnostic palette inside
  the panels.
- shape = reasoning-effort tier (`_paper_style.EFFORT_MARKER`: low = ○,
  high = ◆) — Fig 1 / Fig 5 convention preserved.
- color shade = prompt method (light = default, mid = VS, dark = SSoT),
  drawn from a 3-tone blue palette unused elsewhere in the paper. Same
  shades on every panel, so the prompt key reads once and applies
  globally; the reader doesn't have to mentally re-tint a generic grey
  swatch into each panel's model hue.

The §4.6 reading is per-model: the VS markers (mid-tinted, both ○ and
◆) sit at lower x than the (default, high) marker (light-tinted ◆) on
every model while reaching the same or higher diversity. SSoT markers
(dark-tinted) sit below the (default, low) / (default, high) pair at
similar or higher x — doubly inefficient.

Source: results/effort_diversity/prompt_sensitivity/<model>_token_pareto.csv
(PR #170 + PR #171; gpt54 file is the bioterrorism-excluded variant).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from _paper_style import (
    EFFORT_MARKER,
    MODEL_LABEL,
    MODEL_ORDER,
    PROMPT_COLOR,
    PROMPT_LABEL,
    apply_style,
    soft_grid,
)
from matplotlib.ticker import FixedLocator, SymmetricalLogLocator

apply_style()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPO_ROOT / "results" / "effort_diversity" / "prompt_sensitivity"

# Upstream CSV uses ``phase2`` as the default-prompt key; legend ordering
# follows display order (default / VS / SSoT).
PROMPT_ORDER: tuple[str, ...] = ("phase2", "vs", "ssot")
EFFORT_ORDER: tuple[str, ...] = ("low", "high")


def load_cells(model: str) -> list[dict]:
    csv_path = CSV_DIR / f"{model}_token_pareto.csv"
    rows = list(csv.DictReader(csv_path.open()))
    out: list[dict] = []
    for r in rows:
        if r.get("skipped"):
            continue
        out.append(
            {
                "kind": r["kind"],
                "effort": r["effort"],
                "diversity": float(r["diversity_mean"]),
                "reasoning_tokens_per_idea": float(r["reasoning_tokens_per_idea"]),
            }
        )
    return out


def main() -> None:
    fig, axes = plt.subplots(
        1,
        len(MODEL_ORDER),
        figsize=(8.0, 3.5),
        sharex=True,
        sharey=True,
    )

    all_cells = {m: load_cells(m) for m in MODEL_ORDER}

    # Global ranges (shared axes — same x and y across panels for visual
    # comparability of the per-model spreads). The reasoning-token axis
    # spans 0 (Claude VS-low / SSoT-low produce no reasoning block on any
    # of the 3,000 samples) up to ~1900 (GPT-5.4 SSoT-high), so we use a
    # symlog scale with a small linear threshold (linthresh=1) so the 0-
    # valued cells get an axis-break-style stub at the left edge while the
    # rest of the axis reads as a clean 10^1–10^3 log scale.
    all_x = [c["reasoning_tokens_per_idea"] for cells in all_cells.values() for c in cells]
    all_y = [c["diversity"] for cells in all_cells.values() for c in cells]
    x_lo = -0.3
    x_hi = max(all_x) * 1.5
    y_lo = min(all_y) - 0.020
    y_hi = max(all_y) + 0.020

    for ax, model in zip(axes, MODEL_ORDER):
        soft_grid(ax)
        cells = all_cells[model]

        for c in cells:
            # Borderless markers — modern ML-paper aesthetic.
            ax.scatter(
                c["reasoning_tokens_per_idea"],
                c["diversity"],
                marker=EFFORT_MARKER[c["effort"]],
                color=PROMPT_COLOR[c["kind"]],
                s=120 if EFFORT_MARKER[c["effort"]] != "D" else 110,
                linewidth=0,
                zorder=4,
            )
        ax.set_title(MODEL_LABEL[model], fontsize=10.0, color="#222", pad=6)

    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_ylim(y_lo, y_hi)
    # Symlog ticks: 0 at the linear stub, then decade ticks across the log
    # region (1, 10, 100, 1000). With linthresh=1 the stub is just wide
    # enough to host the 0-marker without intruding on the log reading.
    # Sub-decade minor ticks (2..9 × 10^k) match Panel A's log-axis convention.
    # `sharex=True` propagates limits/scale but not locators, so set per axis.
    axes[0].xaxis.set_major_locator(FixedLocator([0, 1, 1e1, 1e2, 1e3]))
    for ax in axes:
        ax.xaxis.set_minor_locator(
            SymmetricalLogLocator(base=10, linthresh=1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        )
        ax.tick_params(axis="x", which="major", labelsize=7.5, length=3.5)
        ax.tick_params(axis="x", which="minor", length=2.0, width=0.7)
        ax.tick_params(axis="y", labelsize=7.5)
    for ax in axes:
        ax.set_xlabel("")  # supxlabel below

    # Layout: panels take ~70 % of the figure height; bottom band carries
    # the x-label + legend. `subplots_adjust` directly because tight_layout
    # over-pads when supxlabel is multi-line.
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.22, wspace=0.07)
    fig.supxlabel(
        "Reasoning tokens / idea",
        fontsize=9.0,
        y=0.13,
        color="#222",
    )
    # Embedder choice and 100-kw subset are described in the caption of
    # fig:fig1-tts; keep the in-figure label minimal.
    axes[0].set_ylabel(
        "Within-keyword pair distance",
        fontsize=9.0,
        color="#222",
    )

    # Legend: prompt fills (PROMPT_COLOR Blues family) + effort shape key.
    # Mid-grey for the shape-only handles reads independently of any
    # prompt color (which are all in the Blues hue family).
    grey_base = "#666"
    prompt_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PROMPT_COLOR[k],
            markeredgecolor="none",
            markersize=9,
            label=PROMPT_LABEL[k],
        )
        for k in PROMPT_ORDER
    ]
    effort_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=EFFORT_MARKER[eff],
            linestyle="none",
            markerfacecolor=grey_base,
            markeredgecolor="none",
            markersize=9,
            label=f"{eff} effort",
        )
        for eff in EFFORT_ORDER
    ]
    fig.legend(
        handles=prompt_handles + effort_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.005),
        fontsize=8.5,
        ncol=5,
        handletextpad=0.4,
        columnspacing=1.4,
        frameon=False,
    )

    axes[0].annotate(
        "(b)",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-0.18, 1.08),
        textcoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    out = FIG_DIR / "prompt_sensitivity_token_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
