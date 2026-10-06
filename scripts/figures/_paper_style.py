"""Shared paper-figure style: rcParams + per-model colors + effort markers.

Importing this module sets matplotlib rcParams to a consistent
"modern paper" look (Helvetica fallback, no top/right spines, soft
grid, frameless legends). All paper figures should import and call
``apply_style()`` at script start so the look matches Fig 1.
"""

from __future__ import annotations

import matplotlib as mpl

# Three-model palette (consistent across the paper).
MODEL_COLOR = {
    "claude": "#d95f02",  # orange
    "gpt54": "#1b9e77",  # teal-green
    "gemini31pro": "#7570b3",  # slate-purple
}
MODEL_LABEL = {
    "claude": "Claude Sonnet 4.6",
    "gpt54": "GPT-5.4",
    "gemini31pro": "Gemini 3.1 Pro",
}
MODEL_ORDER = ("claude", "gpt54", "gemini31pro")

# Effort-tier marker shapes (used by Fig 1 and Fig 5).
EFFORT_MARKER = {"low": "o", "medium": "^", "high": "D"}
REASONING_EFFORTS = ("low", "medium", "high")

# Prompt-method palette (sequential Blues, used by Fig 1b / Fig 2 / UMAP
# appendix). Same hue family across the three prompts; the three shades
# span the full Blues range (very light / mid / very dark) so that
# default / VS / SSoT remain discriminable at small marker sizes.
# Hues are chosen from ColorBrewer "Blues" to avoid clashing with
# ``MODEL_COLOR`` (Dark2 #1–#3 = orange / teal-green / slate-purple).
PROMPT_COLOR = {
    "default": "#c6dbef",  # very light blue (baseline)
    "vs": "#3182bd",  # mid blue
    "ssot": "#08306b",  # very dark blue / navy
}
# Some scripts use the upstream CSV's prompt key (``phase2``) for the default
# prompt rather than the display label (``default``). Provide the alias so
# either key resolves to the same color.
PROMPT_COLOR["phase2"] = PROMPT_COLOR["default"]
PROMPT_LABEL = {"default": "default", "phase2": "default", "vs": "VS", "ssot": "SSoT"}
PROMPT_ORDER = ("default", "vs", "ssot")

# Joint (prompt, effort) palette used by figures that show raw conditions
# rather than intervention lifts. Convention: hue family = prompt
# (default → grey, VS → blue, SSoT → purple); lightness = effort
# (low → light, high → dark). Three distinct hue families so adjacent
# prompts (default | VS | SSoT) do not visually collide.
# Used by figures that visualize (prompt, effort) cells side by side:
#   - plot_fig_prompt_sensitivity_abcd_a_rate.py
#   - scripts/experiment/rebuttal/plot_strong_judge_path_decision.py
# Figures that visualize *intervention lifts* (q1q4_ratio, ksweep) keep
# the prompt-axis convention via PROMPT_COLOR; they are not the target
# of this palette.
CONDITION_COLOR = {
    ("default", "low"): "#cccccc",
    ("default", "high"): "#555555",
    ("vs", "low"): "#9ecae1",
    ("vs", "high"): "#2171b5",
    ("ssot", "low"): "#bcbddc",
    ("ssot", "high"): "#54278f",
}


def apply_style() -> None:
    """Apply the shared rcParams. Call once at the start of each script."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#444",
            "axes.labelcolor": "#222",
            "xtick.color": "#444",
            "ytick.color": "#444",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.frameon": False,
            "axes.grid": False,
        }
    )


def soft_grid(ax, *, x: bool = True, y: bool = True) -> None:
    """Apply the standard soft major-only grid used across paper figures."""
    if y:
        ax.yaxis.grid(True, which="major", linewidth=0.6, color="#dddddd")
    if x:
        ax.xaxis.grid(True, which="major", linewidth=0.7, color="#cccccc")
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
