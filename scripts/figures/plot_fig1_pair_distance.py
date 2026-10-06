"""Fig 1: within-keyword pair distance vs median reasoning tokens.

Visual encoding:
- Color = model (Claude orange / GPT-5.4 green / Gemini purple),
  consistent across the paper.
- Marker shape = effort tier (low = circle, medium = triangle,
  high = diamond). Encoded by shape rather than color shade so that
  the "color = model" rule stays clean.
- Per-marker text labels are dropped; the upper-left legend carries
  the model identity (and relative low->high shift), the lower-right
  legend carries the effort-tier shape key.
- The ``none`` control tier sits at zero reasoning tokens by
  construction and is reported separately in App I, not on this axis.

Strict reasoning-only token axis:
- Claude Sonnet 4.6 uses the per-sample CSV produced by
  ``scripts/experiment/count_claude_reasoning_tokens.py``.
- GPT-5.4 uses ``raw_api_response.usage.output_tokens_details.reasoning_tokens``.
- Gemini 3.1 Pro uses ``raw_api_response.usage_metadata.thoughts_token_count``.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _paper_style import apply_style, soft_grid
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

apply_style()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "claude": REPO_ROOT / "results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54": REPO_ROOT / "results/effort_diversity/phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini31pro": REPO_ROOT / "results/effort_diversity/phase2_gemini31pro_1180kw_30x3_facet_seed42",
}
MODEL_LABEL = {
    "claude": "Claude Sonnet 4.6",
    "gpt54": "GPT-5.4",
    "gemini31pro": "Gemini 3.1 Pro",
}
LINE_COLORS = {"claude": "#d95f02", "gpt54": "#1b9e77", "gemini31pro": "#7570b3"}
EFFORT_MARKERS = {"low": "o", "medium": "^", "high": "D"}
EFFORTS = ("low", "medium", "high")
MODEL_ORDER = ("claude", "gpt54", "gemini31pro")


def _claude_reasoning_tokens(run_dir: Path) -> dict:
    tokens = defaultdict(list)
    with (run_dir / "reasoning_token_counts.csv").open() as f:
        for row in csv.DictReader(f):
            tokens[(row["keyword"], row["effort"])].append(int(row["reasoning_tokens"]))
    return tokens


def _gpt_reasoning_tokens(samples_path: Path) -> dict:
    tokens = defaultdict(list)
    with samples_path.open() as f:
        for line in f:
            r = json.loads(line)
            raw = json.loads(r["raw_api_response"]) if isinstance(r["raw_api_response"], str) else r["raw_api_response"]
            tokens[(r["keyword"], r["effort"])].append(int(raw["usage"]["output_tokens_details"]["reasoning_tokens"]))
    return tokens


def _gemini_reasoning_tokens(samples_path: Path) -> dict:
    tokens = defaultdict(list)
    with samples_path.open() as f:
        for line in f:
            r = json.loads(line)
            raw = json.loads(r["raw_api_response"]) if isinstance(r["raw_api_response"], str) else r["raw_api_response"]
            tokens[(r["keyword"], r["effort"])].append(int(raw["usage_metadata"].get("thoughts_token_count", 0)))
    return tokens


def _load_dist(run_dir: Path) -> dict:
    out = {}
    with (run_dir / "text-embedding-3-large/summary_by_keyword_effort.csv").open() as f:
        for row in csv.DictReader(f):
            out[(row["keyword"], row["effort"])] = float(row["mean_pairwise_cosine_distance"])
    return out


cells: dict = defaultdict(list)
for model, run_dir in RUNS.items():
    if model == "claude":
        per_sample = _claude_reasoning_tokens(run_dir)
    elif model == "gpt54":
        per_sample = _gpt_reasoning_tokens(run_dir / "samples.jsonl")
    elif model == "gemini31pro":
        per_sample = _gemini_reasoning_tokens(run_dir / "samples.jsonl")
    else:
        raise ValueError(model)
    dist = _load_dist(run_dir)
    for (kw, eff), toks in per_sample.items():
        if (kw, eff) in dist:
            cells[(model, eff)].append((float(np.median(toks)), dist[(kw, eff)]))


fig, ax = plt.subplots(figsize=(5.2, 3.4))

model_rel_gain: dict[str, float] = {}
for model in MODEL_ORDER:
    color = LINE_COLORS[model]
    xs, ys = [], []
    for eff in EFFORTS:
        pts = cells[(model, eff)]
        tokens = np.asarray([p[0] for p in pts])
        dists = np.asarray([p[1] for p in pts])
        xs.append(float(np.median(tokens)))
        ys.append(float(np.mean(dists)))

    ax.plot(xs, ys, "-", color=color, linewidth=2.7, alpha=0.95, zorder=2)
    for eff, x, y in zip(EFFORTS, xs, ys):
        ax.scatter(
            x,
            y,
            s=100 if EFFORT_MARKERS[eff] != "D" else 90,
            marker=EFFORT_MARKERS[eff],
            c=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
    model_rel_gain[model] = (ys[-1] / ys[0] - 1.0) * 100.0

ax.set_xscale("log")
ax.set_xlim(10, 2200)
ax.set_ylim(0.20, 0.42)
ax.set_xlabel("Reasoning tokens / idea", labelpad=6)
ax.set_ylabel("Within-keyword pair distance", labelpad=6)
ax.tick_params(axis="both", which="major", length=3.5, labelsize=8.5)
ax.tick_params(axis="both", which="minor", length=2.0)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
# Soft major-only grid: horizontal y-grid for value reading, plus
# vertical lines at the powers of 10 on the log x-axis.
soft_grid(ax, x=True, y=True)

# Two compact legends, both anchored in the corners, mirroring the
# layout used by inference-scaling and open-ended-LLM evaluation papers
# (e.g. Brown et al. 2024, Hoffmann et al. 2022): a model legend (color)
# and an effort legend (shape), rendered as separate boxes so each
# encoding is read independently.
model_handles = [
    Line2D(
        [0],
        [0],
        color=LINE_COLORS[model],
        linewidth=2.0,
        marker="o",
        markersize=6,
        markerfacecolor=LINE_COLORS[model],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=f"{MODEL_LABEL[model]} (+{model_rel_gain[model]:.0f}%)",
    )
    for model in MODEL_ORDER
]
effort_handles = [
    Line2D(
        [0],
        [0],
        marker=EFFORT_MARKERS[eff],
        color="w",
        markerfacecolor="#666",
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=7 if EFFORT_MARKERS[eff] != "D" else 6.4,
        label=eff,
    )
    for eff in EFFORTS
]
leg_model = ax.legend(
    handles=model_handles,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    fontsize=8.2,
    title=r"Model ($\Delta$ low $\to$ high)",
    title_fontsize=8.2,
    handletextpad=0.6,
    borderpad=0.0,
    labelspacing=0.45,
)
leg_model._legend_box.align = "left"
ax.add_artist(leg_model)

leg_effort = ax.legend(
    handles=effort_handles,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.0),
    fontsize=8.2,
    title="Reasoning effort",
    title_fontsize=8.2,
    handletextpad=0.4,
    borderpad=0.0,
    ncols=3,
    columnspacing=1.1,
)
leg_effort._legend_box.align = "left"

ax.annotate(
    "(a)",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(-0.10, 1.02),
    textcoords="axes fraction",
    fontsize=11,
    fontweight="bold",
    va="bottom",
    ha="left",
)

fig.tight_layout()
out = FIG_DIR / "pair_distance_vs_reasoning_tokens.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.close(fig)
print(f"saved {out}")
