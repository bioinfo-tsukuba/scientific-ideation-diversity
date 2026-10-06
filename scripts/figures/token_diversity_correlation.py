"""O10: token × diversity correlation per (kw, effort) — strict reasoning-only.

For each (kw, effort) cell, compute:
- median reasoning-only tokens per sample (strict, vendor-subset based).
  * Claude Sonnet 4.6 uses the per-sample CSV produced by
    ``scripts/experiment/count_claude_reasoning_tokens.py``
    (reasoning_tokens = output_tokens − count_tokens(full_response) +
    system_overhead, i.e. billed output_tokens minus the visible answer,
    with the per-call count_tokens overhead added back).
  * GPT-5.4 reads ``raw_api_response.usage.output_tokens_details.reasoning_tokens``.
  * Gemini 3.1 Pro reads ``raw_api_response.usage_metadata.thoughts_token_count``.
- mean within-kw pair distance (combined OpenAI embedding).

Then compute Pearson r across all per-cell points per model.
Also report per-effort breakdown and within-effort r.

Output: data/token_vs_diversity_correlation.csv + Markdown table to stdout.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import CACHE_DIR, DATA_DIR, REPO_ROOT

RUNS = {
    "claude": REPO_ROOT / "results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42",
    "gpt54": REPO_ROOT / "results/effort_diversity/phase2_gpt54_1180kw_30x4_facet_seed42",
    "gemini31pro": REPO_ROOT / "results/effort_diversity/phase2_gemini31pro_1180kw_30x3_facet_seed42",
}
EFFORTS = ("none", "low", "medium", "high")
DISPLAY_NAME = {
    "claude": "Claude Sonnet 4.6",
    "gpt54": "GPT-5.4",
    "gemini31pro": "Gemini 3.1 Pro",
}


def _claude_reasoning_tokens(run_dir: Path) -> dict:
    """Load per-sample reasoning tokens from the pre-computed CSV."""
    csv_path = run_dir / "reasoning_token_counts.csv"
    tokens = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            tokens[(row["keyword"], row["effort"])].append(int(row["reasoning_tokens"]))
    return tokens


def _gpt_reasoning_tokens(samples_path: Path) -> dict:
    """Read reasoning_tokens subset field from the OpenAI Responses usage block."""
    tokens = defaultdict(list)
    with samples_path.open() as f:
        for line in f:
            r = json.loads(line)
            raw = json.loads(r["raw_api_response"]) if isinstance(r["raw_api_response"], str) else r["raw_api_response"]
            rt = raw["usage"]["output_tokens_details"]["reasoning_tokens"]
            tokens[(r["keyword"], r["effort"])].append(int(rt))
    return tokens


def _gemini_reasoning_tokens(samples_path: Path) -> dict:
    """Read thoughts_token_count from the Vertex AI usage_metadata block."""
    tokens = defaultdict(list)
    with samples_path.open() as f:
        for line in f:
            r = json.loads(line)
            raw = json.loads(r["raw_api_response"]) if isinstance(r["raw_api_response"], str) else r["raw_api_response"]
            thoughts = raw["usage_metadata"].get("thoughts_token_count", 0)
            tokens[(r["keyword"], r["effort"])].append(int(thoughts))
    return tokens


def load_per_cell_tokens(model: str, run_dir: Path) -> dict:
    if model == "claude":
        per_sample = _claude_reasoning_tokens(run_dir)
    elif model == "gpt54":
        per_sample = _gpt_reasoning_tokens(run_dir / "samples.jsonl")
    elif model == "gemini31pro":
        per_sample = _gemini_reasoning_tokens(run_dir / "samples.jsonl")
    else:
        raise ValueError(f"unknown model: {model!r}")
    return {k: float(np.median(v)) for k, v in per_sample.items()}


def load_per_cell_diversity(run_dir: Path) -> dict:
    """Read summary_by_keyword_effort.csv (combined OpenAI embedding) for mean pairwise distance."""
    out = {}
    csv_path = run_dir / "text-embedding-3-large/summary_by_keyword_effort.csv"
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[(row["keyword"], row["effort"])] = float(row["mean_pairwise_cosine_distance"])
    return out


def pearson(xs: list, ys: list) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    # Constant cells (e.g., effort=none with strict reasoning-only tokens
    # where every sample holds at 0 / ~3 tokens) have zero variance, so
    # the Pearson denominator is ill-defined. Surface as NaN rather than
    # an accidental 0.0 or RuntimeWarning-cast value.
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


print("Loading per-cell metrics (strict reasoning-only)...")
results = {}
for model, run_dir in RUNS.items():
    tokens = load_per_cell_tokens(model, run_dir)
    diversity = load_per_cell_diversity(run_dir)
    common = set(tokens) & set(diversity)
    print(f"  {model}: {len(common)} (kw, effort) cells")
    cells = sorted(common)
    xs = [tokens[c] for c in cells]
    ys = [diversity[c] for c in cells]
    r_all = pearson(xs, ys)
    print(f"    overall Pearson r (reasoning-token × diversity) across all cells: {r_all:.3f}")
    per_effort = {}
    for eff in EFFORTS:
        cells_eff = [c for c in cells if c[1] == eff]
        if not cells_eff:
            continue
        xs_eff = [tokens[c] for c in cells_eff]
        ys_eff = [diversity[c] for c in cells_eff]
        r_eff = pearson(xs_eff, ys_eff)
        per_effort[eff] = (r_eff, len(cells_eff))
        r_str = "n/a" if np.isnan(r_eff) else f"{r_eff:+.3f}"
        print(f"    {eff:6s}: r={r_str}  n={per_effort[eff][1]}")
    results[model] = {"all_r": r_all, "per_effort": per_effort}

# Persist
out_csv = DATA_DIR / "token_vs_diversity_correlation.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "scope", "n", "pearson_r"])
    for model, res in results.items():
        n_total = sum(v[1] for v in res["per_effort"].values())
        w.writerow([model, "all_effort_pooled", n_total, res["all_r"]])
        for eff, (r, n) in res["per_effort"].items():
            w.writerow([model, f"effort={eff}", n, r])
print(f"\nsaved {out_csv}")

# Markdown for paper appendix
print("\n=== Markdown table for Appendix ===\n")
print("| Model | Scope | n cells | Pearson r |")
print("| --- | --- | ---: | ---: |")
def _fmt_r(r: float) -> str:
    return "n/a" if np.isnan(r) else f"{r:.3f}"


for model, res in results.items():
    label = DISPLAY_NAME[model]
    n_total = sum(v[1] for v in res["per_effort"].values())
    efforts_present = "/".join(res["per_effort"].keys())
    print(f"| {label} | all ({efforts_present} pooled) | {n_total} | {_fmt_r(res['all_r'])} |")
    for eff, (r, n) in res["per_effort"].items():
        print(f"| {label} | within effort = {eff} | {n} | {_fmt_r(r)} |")
