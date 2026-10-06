"""Q-D analysis (#133): reasoning-token Pareto across (effort, prompt) cells.

For each cell in the {default, VS, SSoT} × {low, high} × per-model design,
computes the per-keyword pair-distance diversity and the per-keyword
reasoning-token usage (deduplicated by api call so VS's k=3 ideas-per-call
don't triple-count), then reports cell-level summaries that the §4.6
\"reasoning Pareto\" discussion needs.

Headline cost axis: **reasoning tokens per idea** (per-vendor reasoning
subset only, excluding visible answer tokens):

  Claude Sonnet 4.6: ``reasoning_tokens`` from
    ``count_claude_reasoning_tokens.py`` (``output_tokens − count_tokens(answer)
    + system_overhead``; clamped to 0 when ``has_reasoning_block=False``).
  GPT-5.4: ``raw_api_response.usage.output_tokens_details.reasoning_tokens``.
  Gemini 3.1 Pro: ``raw_api_response.usage_metadata.thoughts_token_count``.

  Reasoning tokens isolate the variable being scaled along the effort axis.
  Visible-answer length varies with prompt-output structure (e.g., VS emits
  3 ideas in one answer) and would dilute the (cost, diversity) reading if
  pooled in. The legacy ``generation_tokens`` (= visible + reasoning) is
  still computed and emitted in the CSV for cross-checking but is not the
  headline axis.

The cost is divided by ``k_ideas_per_call`` (3 for VS, 1 for default /
SSoT) to amortize VS's call across its 3 ideas — the apples-to-apples
per-generated-idea cost for the §4.6 \"reasoning per idea\" question.

Cell-level outputs:

* mean diversity across the 100-kw subset (= the existing pair-distance
  number used by Q-A);
* reasoning tokens per idea (= ``reasoning_total / n_calls / k``);
* generation tokens per idea (= ``(visible + reasoning) / n_calls / k``;
  legacy axis, retained for cross-check);
* diversity per 1 000 reasoning tokens (per idea), for the headline
  Pareto ratio.

The point of the analysis is not the absolute reasoning-token numbers (the
API providers price tokens differently and the kw distributions per model
are bimodal, not representative) but the *ranking*: at the same model, does
VS or SSoT deliver more pair-distance per reasoning token than the default
prompt at the same effort tier? And does the effort knob trade more
diversity per reasoning token than the prompt knob within VS?

Implementation notes
--------------------
* VS samples store the per-call tokens duplicated across the k=3 ideas of
  that call. Deduplication is keyed by (keyword, batch_index) where
  batch_index = sample_index // PROMPT_METHOD_K[prompt_method]. SSoT and
  default cells use k=1 so each sample is its own call and the dedupe is
  a no-op.
* Phase2 default cells store per-sample tokens, k=1; same dedupe key works.

Outputs: stdout table + CSV at
``results/effort_diversity/prompt_sensitivity/<model>_token_pareto.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import IdeaModelName  # noqa: E402

# CLI accepts the short paper-side slugs (`claude` / `gpt54` / `gemini31pro`)
# matching `paper/scripts/_paper_style.py::MODEL_ORDER` and result-file
# naming (`<slug>_token_pareto.csv`). Internal branching on vendor uses the
# `IdeaModelName` Enum so per-vendor reasoning-token extraction stays type-
# safe — mirrors `run_prompt_sensitivity.py:141`'s slug→IdeaModelName map.
MODELS: dict[str, dict] = {
    "claude": {
        "idea_model": IdeaModelName.CLAUDE_SONNET_4_6,
        "phase2_dir": "phase2_claude_1180kw_30x4_facet_seed42",
        "ps_prefix": "claude",
    },
    "gpt54": {
        "idea_model": IdeaModelName.GPT_5_4,
        "phase2_dir": "phase2_gpt54_1180kw_30x4_facet_seed42",
        "ps_prefix": "gpt54",
    },
    "gemini31pro": {
        "idea_model": IdeaModelName.GEMINI_3_1_PRO,
        "phase2_dir": "phase2_gemini31pro_1180kw_30x3_facet_seed42",
        "ps_prefix": "gemini31pro",
    },
}

# Mirrors run_prompt_sensitivity.py PROMPT_METHOD_K. Default-prompt cells
# (phase2) record one sample per call too, so k=1 is correct there.
_K_PER_CALL: dict[str, int] = {"phase2": 1, "vs": 3, "ssot": 1}

# Embedding subdir whose summary_by_keyword_effort.csv yields pair distance.
# Q-D is run_dir-only — the diversity numbers come from the same embedding
# Q-A's headline uses (text-embedding-3-large), so they line up panel-by-panel
# in the §4.6 figure. Cross-embedding triangulation lives in Q-A's output.
_EMB_SUBDIR = "text-embedding-3-large"
_PAIR_COL = "mean_pairwise_cosine_distance"


def _load_claude_reasoning_index(run_dir: Path) -> dict[tuple[str, str, int], int]:
    """Map (keyword, effort, sample_index) → claude reasoning_tokens.

    Built from `reasoning_token_counts.csv` produced by
    `scripts/experiment/data/count_claude_reasoning_tokens.py`. Raises if the
    CSV is absent so the caller doesn't silently fall back to garbage.
    """
    import csv as _csv  # local import: avoid global name shadowing

    csv_path = run_dir / "reasoning_token_counts.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"reasoning_token_counts.csv not found in {run_dir}; run "
            f"scripts/experiment/data/count_claude_reasoning_tokens.py {run_dir} first."
        )
    out: dict[tuple[str, str, int], int] = {}
    with csv_path.open() as f:
        for row in _csv.DictReader(f):
            out[(row["keyword"], row["effort"], int(row["sample_index"]))] = int(row["reasoning_tokens"])
    return out


def _per_sample_reasoning_tokens(
    sample: dict, *, model: IdeaModelName, claude_index: dict[tuple[str, str, int], int]
) -> int:
    """Extract per-vendor reasoning ("thinking") tokens for one sample.

    Mirrors `paper/scripts/plot_fig4_tokens_by_effort.py::load_reasoning_tokens`
    so the §4.6 numbers stay consistent with paper §4.2 / Fig 4. Per-vendor
    field locations:

    - Claude (Bedrock Converse): `output_tokens` already includes reasoning,
      but isn't separable at the API. We use the post-hoc extracted value
      from `reasoning_token_counts.csv` (computed per-sample by
      `count_claude_reasoning_tokens.py` via `count_tokens(visible_answer)`
      subtraction). The CSV is keyed by (keyword, effort, sample_index).
    - GPT-5.4 (OpenAI Responses): `usage.output_tokens_details.reasoning_tokens`
      inside `raw_api_response`. Reasoning is *included in* `output_tokens`,
      i.e. `output_tokens = visible + reasoning`.
    - Gemini 3.1 Pro (Vertex AI): `usage_metadata.thoughts_token_count`
      inside `raw_api_response`. Reasoning is *not in* `output_tokens`,
      i.e. `total_tokens = input + output (visible) + thoughts`.
    """
    keyword = sample["keyword"]
    effort = sample["effort"]
    if isinstance(effort, str) and effort.startswith("EffortName."):
        effort = effort.split(".", 1)[1].lower()

    if model is IdeaModelName.CLAUDE_SONNET_4_6:
        return int(claude_index[(keyword, effort, int(sample["sample_index"]))])

    raw = sample["raw_api_response"]
    raw = json.loads(raw) if isinstance(raw, str) else raw
    if model is IdeaModelName.GPT_5_4:
        return int(raw["usage"]["output_tokens_details"]["reasoning_tokens"])
    if model is IdeaModelName.GEMINI_3_1_PRO:
        return int(raw.get("usage_metadata", {}).get("thoughts_token_count", 0))
    raise ValueError(f"unknown model: {model!r}")


def _per_kw_tokens(samples_jsonl: Path, prompt_kind: str, *, model: IdeaModelName) -> pd.DataFrame:
    """Per-(keyword, effort) token cost, deduplicated by api call.

    Returns columns: keyword, effort, n_calls, input_tokens, output_tokens,
    total_tokens, reasoning_tokens, visible_tokens. n_calls is the number
    of distinct batches the cell ran for that keyword, e.g., 10 calls ×
    k=3 = 30 samples for VS, vs 30 calls × 1 for SSoT / default.

    Tokens are summed across batches (deduplicated by sample_index // k so
    VS's call-level fields aren't triple-counted). `visible_tokens` is the
    answer-only side: `total - input - reasoning`. `reasoning_tokens` is
    the model-side reasoning / thinking field, extracted per-vendor (see
    `_per_sample_reasoning_tokens`).
    """
    k = _K_PER_CALL[prompt_kind]
    claude_index: dict[tuple[str, str, int], int] = {}
    if model is IdeaModelName.CLAUDE_SONNET_4_6:
        claude_index = _load_claude_reasoning_index(samples_jsonl.parent)
    seen: set[tuple[str, str, int]] = set()
    rows: list[dict] = []
    with open(samples_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            keyword = d["keyword"]
            effort = d["effort"]
            if effort.startswith("EffortName."):  # tolerate enum-repr-leakage
                effort = effort.split(".", 1)[1].lower()
            batch_index = int(d["sample_index"]) // k
            key = (keyword, effort, batch_index)
            if key in seen:
                continue
            seen.add(key)
            reasoning = _per_sample_reasoning_tokens(d, model=model, claude_index=claude_index)
            input_tokens = int(d["input_tokens"])
            total_tokens = int(d["total_tokens"])
            # `visible_tokens` is the answer-only side (excludes reasoning),
            # derived consistently across providers as total - input - reasoning.
            # For Claude / GPT, output_tokens = visible + reasoning, so this
            # matches output_tokens - reasoning. For Gemini, output_tokens is
            # already visible-only, so this matches output_tokens.
            visible_tokens = total_tokens - input_tokens - reasoning
            rows.append(
                {
                    "keyword": keyword,
                    "effort": effort,
                    "input_tokens": input_tokens,
                    "output_tokens": int(d["output_tokens"]),
                    "total_tokens": total_tokens,
                    "reasoning_tokens": reasoning,
                    "visible_tokens": visible_tokens,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "keyword",
                "effort",
                "n_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "visible_tokens",
            ]
        )
    df = pd.DataFrame(rows)
    return df.groupby(["keyword", "effort"], as_index=False).agg(
        n_calls=("input_tokens", "size"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        total_tokens=("total_tokens", "sum"),
        reasoning_tokens=("reasoning_tokens", "sum"),
        visible_tokens=("visible_tokens", "sum"),
    )


def _per_kw_diversity(run_dir: Path) -> pd.DataFrame:
    """Per-(keyword, effort) pair-distance diversity from summary CSV."""
    csv_path = run_dir / _EMB_SUBDIR / "summary_by_keyword_effort.csv"
    df = pd.read_csv(csv_path)
    return df[["keyword", "effort", _PAIR_COL]].rename(columns={_PAIR_COL: "diversity"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/effort_diversity",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=("Optional CSV destination. Defaults to <results-root>/prompt_sensitivity/<model>_token_pareto.csv"),
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all rows for KW from the per-cell aggregates. Use for "
            "keywords whose generation yield is below the per-cell target."
        ),
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    idea_model: IdeaModelName = cfg["idea_model"]
    excluded_keywords: tuple[str, ...] = tuple(sorted(set(args.exclude_keyword)))

    # Cell layout: (kind, effort) → (samples.jsonl, run_dir for diversity)
    # Phase2 cell is the source for default-prompt at low/high effort. The
    # 100-kw subset filter is applied at aggregation time so cross-cell
    # comparisons stay on the same kw support.
    phase2_dir = args.results_root / cfg["phase2_dir"]
    cell_specs: list[tuple[str, str, Path]] = []
    for eff in ("low", "high"):
        cell_specs.append(("phase2", eff, phase2_dir))
    for kind in ("vs", "ssot"):
        for eff in ("low", "high"):
            cell_specs.append(
                (
                    kind,
                    eff,
                    args.results_root / "prompt_sensitivity" / f"{cfg['ps_prefix']}_{kind}_{eff}",
                )
            )

    # The 100-kw subset Q-A / Q-C analyses use; tokens-vs-diversity is read on
    # the same kw support so the §4.6 numbers come from one bimodal subset.
    kw_subset_path = args.results_root / "prompt_sensitivity" / f"q1_q4_{args.model}.csv"
    kw_subset = pd.read_csv(kw_subset_path)
    if excluded_keywords:
        kw_subset = kw_subset[~kw_subset["keyword"].isin(excluded_keywords)].reset_index(drop=True)
        print(
            f"Excluded {len(excluded_keywords)} keywords from kw subset "
            f"({list(excluded_keywords)}); kept {len(kw_subset)}"
        )
    keep_kws = set(kw_subset["keyword"])

    rows: list[dict] = []
    for kind, effort, run_dir in cell_specs:
        # Diversity (pair distance) per kw — only the 100-kw subset.
        div_path = run_dir / _EMB_SUBDIR / "summary_by_keyword_effort.csv"
        if not div_path.exists():
            rows.append({"kind": kind, "effort": effort, "skipped": "missing diversity CSV"})
            continue
        div = _per_kw_diversity(run_dir)
        div = div[(div["keyword"].isin(keep_kws)) & (div["effort"] == effort)]

        samples_path = run_dir / "samples.jsonl"
        if not samples_path.exists():
            rows.append({"kind": kind, "effort": effort, "skipped": "missing samples.jsonl"})
            continue
        tok = _per_kw_tokens(samples_path, kind, model=idea_model)
        tok = tok[(tok["keyword"].isin(keep_kws)) & (tok["effort"] == effort)]

        merged = div.merge(tok, on=["keyword", "effort"], how="inner")
        if merged.empty:
            rows.append({"kind": kind, "effort": effort, "skipped": "no kw overlap"})
            continue

        # Per-idea cost components, all divided by k=_K_PER_CALL[kind] to
        # amortize a VS call across its k=3 ideas (k=1 for default/SSoT).
        # The headline Pareto axis is `reasoning_tokens_per_idea`, which
        # isolates the variable being scaled along the effort axis. The
        # legacy `generation_tokens` (= visible + reasoning) is retained
        # in the CSV for cross-checking but is not the headline axis.
        k = _K_PER_CALL[kind]
        merged["per_idea_reasoning_tokens"] = merged["reasoning_tokens"] / merged["n_calls"] / k
        merged["generation_tokens"] = merged["visible_tokens"] + merged["reasoning_tokens"]
        merged["per_idea_generation_tokens"] = merged["generation_tokens"] / merged["n_calls"] / k
        # Diversity per kilo-reasoning-token (headline efficiency metric).
        # Guarded against div-by-zero when a cell has no reasoning block at
        # all (e.g., Claude VS-low / SSoT-low collapse to 0); those cells
        # surface as inf rather than crashing the aggregate.
        merged["diversity_per_kilo_reasoning_token"] = merged["diversity"] / (
            merged["per_idea_reasoning_tokens"].replace(0, pd.NA) / 1000.0
        )
        merged["diversity_per_kilo_generation_token"] = merged["diversity"] / (
            merged["per_idea_generation_tokens"] / 1000.0
        )

        # Cell-level summary across kws.
        n_calls_total = int(merged["n_calls"].sum())
        in_total = int(merged["input_tokens"].sum())
        out_total = int(merged["output_tokens"].sum())
        tot_total = int(merged["total_tokens"].sum())
        reason_total = int(merged["reasoning_tokens"].sum())
        visible_total = int(merged["visible_tokens"].sum())
        gen_total = int(merged["generation_tokens"].sum())
        rows.append(
            {
                "kind": kind,
                "effort": effort,
                "k_ideas_per_call": k,
                "n_kws": int(len(merged)),
                "diversity_mean": float(merged["diversity"].mean()),
                "diversity_q1_mean": float(
                    merged.loc[
                        merged["keyword"].isin(set(kw_subset.loc[kw_subset["stratum"] == "Q1", "keyword"])), "diversity"
                    ].mean()
                ),
                "diversity_q4_mean": float(
                    merged.loc[
                        merged["keyword"].isin(set(kw_subset.loc[kw_subset["stratum"] == "Q4", "keyword"])), "diversity"
                    ].mean()
                ),
                "n_calls_total": n_calls_total,
                "n_ideas_total": n_calls_total * k,
                # Cell totals (raw fields + the derived reasoning / visible split).
                "input_tokens_total": in_total,
                "output_tokens_total": out_total,
                "total_tokens_total": tot_total,
                "reasoning_tokens_total": reason_total,
                "visible_tokens_total": visible_total,
                "generation_tokens_total": gen_total,
                # Per-call (input is per-prompt; reasoning / visible / generation
                # are the model-side breakdown).
                "input_tokens_per_call": float(in_total / n_calls_total),
                "output_tokens_per_call": float(out_total / n_calls_total),
                "reasoning_tokens_per_call": float(reason_total / n_calls_total),
                "visible_tokens_per_call": float(visible_total / n_calls_total),
                "generation_tokens_per_call": float(gen_total / n_calls_total),
                # Per-idea (further amortized by k for VS).
                "input_tokens_per_idea": float(in_total / n_calls_total / k),
                "output_tokens_per_idea": float(out_total / n_calls_total / k),
                "reasoning_tokens_per_idea": float(reason_total / n_calls_total / k),
                "visible_tokens_per_idea": float(visible_total / n_calls_total / k),
                "generation_tokens_per_idea": float(gen_total / n_calls_total / k),
                "diversity_per_kilo_reasoning_token_mean": float(
                    merged["diversity_per_kilo_reasoning_token"].dropna().mean()
                ) if merged["diversity_per_kilo_reasoning_token"].notna().any() else float("nan"),
                "diversity_per_kilo_generation_token_mean": float(merged["diversity_per_kilo_generation_token"].mean()),
            }
        )

    df = pd.DataFrame(rows)

    print()
    print(f"=== Q-D reasoning-token Pareto on {args.model} ===")
    print("   diversity = mean pairwise cosine distance over the 100-kw subset (text-emb-3-l).")
    print("   reasoning tokens extracted per-vendor (claude reasoning_tokens.csv /")
    print("     gpt54 output_tokens_details.reasoning_tokens / gemini31pro thoughts_token_count).")
    print("   per_idea_* = cell totals / n_calls / k (k=3 for VS, 1 otherwise).")
    print("   gen_tokens (legacy: visible + reasoning) shown for cross-check only.")
    print()
    print(
        f"{'cell':<14s} | {'k':>2s} | {'div mean':>9s} | {'Q1 div':>7s} | {'Q4 div':>7s} | "
        f"{'reason/idea':>11s} | {'visible/idea':>12s} | {'gen/idea':>9s} | {'div/kRTok':>10s}"
    )
    print("-" * 110)
    for r in rows:
        if r.get("skipped"):
            print(f"{r['kind']:<6s} {r['effort']:<7s} | [skip: {r['skipped']}]")
            continue
        cell_label = f"{r['kind']:<6s} {r['effort']:<7s}"
        div_per_krtok = r["diversity_per_kilo_reasoning_token_mean"]
        div_per_krtok_str = f"{div_per_krtok:>10.3f}" if div_per_krtok == div_per_krtok else f"{'n/a':>10s}"
        print(
            f"{cell_label} | {r['k_ideas_per_call']:>2d} | {r['diversity_mean']:>9.3f} | "
            f"{r['diversity_q1_mean']:>7.3f} | {r['diversity_q4_mean']:>7.3f} | "
            f"{r['reasoning_tokens_per_idea']:>11.0f} | "
            f"{r['visible_tokens_per_idea']:>12.0f} | "
            f"{r['generation_tokens_per_idea']:>9.0f} | "
            f"{div_per_krtok_str}"
        )

    out_csv = args.output_csv or (args.results_root / "prompt_sensitivity" / f"{args.model}_token_pareto.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
