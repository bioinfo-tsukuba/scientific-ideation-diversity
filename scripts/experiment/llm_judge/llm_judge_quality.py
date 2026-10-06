#!/usr/bin/env python3
"""Run LLM-as-judge quality critique on an effort-diversity run.

Uses the LiveIdeaBench v4 ``critic_prompt`` (verbatim, via
``external/liveideabench/utils/prompts.json``) with structured output
so the three-axis scores (originality / feasibility / clarity, 1-10)
come back without regex retries that could bias per-sample attempt
counts.  Single-idea evaluation; no pair selection and no
position-bias control (the pairwise sibling script lives at
``llm_judge_pairwise.py``).

Supports both non-reasoning judges (e.g. ``gpt-4.1``) and
reasoning-capable judges (e.g. ``gpt-5.4``) via the same entry point;
``--judge-effort`` is required for reasoning judges and unused for
non-reasoning ones.

Usage:
    # Non-reasoning judge
    uv run python scripts/experiment/llm_judge/llm_judge_quality.py \\
        --run-dir results/effort_diversity/phase2_claude_1180kw_30x4_facet_seed42 \\
        --judge-model gpt-4.1 \\
        --max-concurrency 32

    # Reasoning judge (strong-judge ablation; see rebuttal/strong_judge_design.md)
    uv run python scripts/experiment/llm_judge/llm_judge_quality.py \\
        --run-dir results/effort_diversity/prompt_sensitivity/claude_vs_low \\
        --judge-model gpt-5.4 \\
        --judge-effort medium \\
        --max-concurrency 32

Output layout (mirrors the pairwise judge):
    run_dir/
      llm_judge_quality_<slug>/                 # non-reasoning
      llm_judge_quality_<slug>__<effort>/       # reasoning
        quality.jsonl
        quality.errors.jsonl
        manifest.json

See docs/20260421_paper_issue_tree.md §3.5 for the design rationale
and docs/20260418_llm_judge_plan.md for the pairwise equivalent.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.artifacts import load_run_sample_records
from src.model_registry import (
    JUDGE_MODEL_ALLOWED_PROVIDERS,
    JUDGE_MODEL_PROVIDER,
    REASONING_JUDGE_DEFAULT_EFFORT,
    EffortName,
    GoogleAIStudioConfig,
    JudgeModelName,
    ProviderName,
    is_reasoning_judge,
)
from src.providers.llm_judge.anthropic_quality_backend import AnthropicQualityJudgeBackend
from src.providers.llm_judge.anthropic_reasoning_quality_backend import (
    AnthropicReasoningQualityJudgeBackend,
)
from src.providers.llm_judge.base import QualityJudgeBackend
from src.providers.llm_judge.gemini_quality_backend import GeminiQualityJudgeBackend
from src.providers.llm_judge.gemini_reasoning_quality_backend import (
    GeminiReasoningQualityJudgeBackend,
)
from src.providers.llm_judge.openai_quality_backend import OpenAIQualityJudgeBackend
from src.providers.llm_judge.openai_reasoning_quality_backend import (
    OpenAIReasoningQualityJudgeBackend,
)
from src.providers.tracing import flush_langfuse
from src.schemas.llm_judge_quality import QualityJudgeRunManifest
from src.schemas.sample import SampleRecord
from src.services.llm_judge_quality_service import (
    PROMPT_TEMPLATE_VERSION,
    build_quality_requests,
    errors_path_for,
    run_quality_judgments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Effort-diversity run directory containing samples.jsonl",
    )
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        required=True,
        choices=list(JudgeModelName),
        metavar=f"{{{','.join(m.value for m in JudgeModelName)}}}",
        help=(
            "Judge model.  Both non-reasoning (e.g. gpt-4.1) and "
            "reasoning-capable (e.g. gpt-5.4) judges are accepted; "
            "the latter require --judge-effort."
        ),
    )
    parser.add_argument(
        "--judge-effort",
        type=EffortName,
        default=None,
        choices=[EffortName.LOW, EffortName.MEDIUM, EffortName.HIGH],
        metavar="{low,medium,high}",
        help=(
            "Reasoning effort tier passed to the vendor's reasoning / "
            "thinking parameter.  Required when --judge-model is a "
            "reasoning judge; rejected for non-reasoning judges.  "
            "Defaults to REASONING_JUDGE_DEFAULT_EFFORT[--judge-model] "
            "(vendor default) when omitted with a reasoning judge."
        ),
    )
    parser.add_argument(
        "--provider",
        type=ProviderName,
        default=None,
        choices=[
            ProviderName.OPENAI,
            ProviderName.VERTEX,
            ProviderName.ANTHROPIC,
            ProviderName.BEDROCK,
            ProviderName.GOOGLE_AI_STUDIO,
        ],
        metavar="{openai,vertex,anthropic,bedrock,google_ai_studio}",
        help=(
            "Provider that serves --judge-model. If omitted, the default "
            "provider from JUDGE_MODEL_PROVIDER is used. Models that accept "
            "multiple providers (e.g. gemini-2.5-flash on Vertex or "
            "Google AI Studio) need an explicit choice only when the "
            "non-default path is wanted."
        ),
    )
    parser.add_argument(
        "--filter-keywords-csv",
        type=Path,
        default=None,
        help=(
            "CSV file whose ``keyword`` column lists the keywords to keep. "
            "Records whose ``keyword`` is not in this set are dropped before "
            "the uniformity check.  The CSV path is recorded in the manifest "
            "for reproducibility; the same CSV must be passed on resume."
        ),
    )
    parser.add_argument(
        "--filter-effort",
        type=EffortName,
        default=None,
        choices=[
            EffortName.NONE,
            EffortName.LOW,
            EffortName.MEDIUM,
            EffortName.HIGH,
        ],
        metavar="{none,low,medium,high}",
        help=(
            "If set, drop records whose generation effort != this value.  "
            "Used to scope a phase2 run_dir (which holds all four efforts in "
            "one samples.jsonl) to a single condition; not needed for the "
            "per-condition prompt-sensitivity dirs that are already scoped."
        ),
    )
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --run-dir/llm_judge_quality_<slug>",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all samples for KW before the uniformity check and judging. "
            "Can be repeated. Use for keywords whose generation yield is below "
            "30 (e.g., safety-filter blocks) so the remaining keywords still "
            "satisfy the uniform-count invariant."
        ),
    )
    return parser.parse_args()


def _require_uniform_sample_count_per_keyword(records: list[SampleRecord]) -> None:
    """Abort if sample counts differ across keywords.

    Mirrors the pairwise judge's invariant: a balanced
    (n_effort * n_sample_per_bucket) per keyword design is a
    prerequisite for per-(keyword, effort) aggregates to be comparable.
    An imbalance signals partial generation, dropped keywords that
    weren't finalised away, or accidentally merged run dirs, and would
    bias downstream aggregates.
    """
    counts: dict[str, int] = {}
    for r in records:
        counts[r.keyword] = counts.get(r.keyword, 0) + 1
    if not counts:
        raise SystemExit("samples.jsonl is empty: no records to judge")
    distinct_counts = sorted(set(counts.values()))
    if len(distinct_counts) > 1:
        # Show up to 3 examples per observed count so the operator can
        # eyeball which keywords are under-/over-filled.
        examples_by_count: dict[int, list[str]] = {}
        for kw, c in counts.items():
            examples_by_count.setdefault(c, []).append(kw)
        breakdown = "\n".join(
            f"  {c} samples: {len(examples_by_count[c])} kw (e.g. {sorted(examples_by_count[c])[:3]!r})"
            for c in distinct_counts
        )
        raise SystemExit("Sample count is not uniform across keywords. Counts:\n" + breakdown)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "judge"


# Resume-compat fields: the JSONL dedup key is
# (keyword, effort, sample_row_index, judge_model), so judge_model +
# prompt_template_version is the full set of fields whose change would
# produce rows that analysis scripts cannot distinguish from rows
# produced under a prior config.  Unlike the pairwise judge there is no
# stratum / top_m / bottom_m / both_orders to check here.  ``judge_effort``
# is included so a re-run with a different reasoning tier on the same
# (run_dir, judge_model) writes to a fresh manifest (the output dir
# slug already carries effort for reasoning judges; this check is the
# safety net if a user passes --output-dir explicitly).
_MANIFEST_RESUME_COMPAT_FIELDS: tuple[str, ...] = (
    "prompt_template_version",
    "judge_model",
    "judge_effort",
    "excluded_keywords",
    # Sample-scoping flags must match on resume so a half-finished run
    # is not silently re-extended over a different subset.
    "filter_keywords_csv",
    "filter_effort",
)


def _require_manifest_compat(manifest_path: Path, candidate: QualityJudgeRunManifest) -> None:
    """Refuse to resume if the on-disk manifest disagrees on judge config."""
    if not manifest_path.exists():
        return
    prior = QualityJudgeRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for field in _MANIFEST_RESUME_COMPAT_FIELDS:
        prior_val = getattr(prior, field)
        new_val = getattr(candidate, field)
        if prior_val != new_val:
            mismatches.append(f"  {field}: prior={prior_val!r} new={new_val!r}")
    if mismatches:
        raise SystemExit(
            f"Refusing to resume {manifest_path}: judge config changed.\n"
            + "\n".join(mismatches)
            + "\nUse --output-dir to write to a fresh location, or delete the "
            "existing manifest + JSONL if the prior run should be discarded."
        )


def _load_keyword_filter_csv(csv_path: Path) -> frozenset[str]:
    """Read a ``keyword`` column from a CSV into a frozenset.

    Used to scope a run to a known keyword set (e.g., Q_1 ∪ Q_4) without
    embedding the list in argv; the CSV path is recorded in the manifest
    so a re-run with the same CSV reproduces the same scope.
    """
    if not csv_path.exists():
        raise SystemExit(f"--filter-keywords-csv not found: {csv_path}")
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "keyword" not in reader.fieldnames:
            raise SystemExit(
                f"--filter-keywords-csv {csv_path} must have a 'keyword' column "
                f"(found columns: {reader.fieldnames!r})"
            )
        keywords = frozenset(row["keyword"] for row in reader if row["keyword"])
    if not keywords:
        raise SystemExit(
            f"--filter-keywords-csv {csv_path} yielded an empty keyword set"
        )
    return keywords


def _resolve_judge_effort(
    *, judge_model: JudgeModelName, judge_effort: EffortName | None
) -> EffortName | None:
    """Validate and normalise ``--judge-effort`` against ``--judge-model``.

    Reasoning judges require an effort tier (defaults to the vendor
    default in :data:`REASONING_JUDGE_DEFAULT_EFFORT` when omitted);
    non-reasoning judges reject any ``--judge-effort`` value so an
    operator misconfiguration fails loudly rather than silently
    ignoring it.
    """
    if is_reasoning_judge(judge_model):
        return judge_effort or REASONING_JUDGE_DEFAULT_EFFORT[judge_model]
    if judge_effort is not None:
        raise SystemExit(
            f"--judge-effort is only valid for reasoning judges; "
            f"--judge-model={judge_model.value!r} is non-reasoning."
        )
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("llm_judge_quality")

    run_dir: Path = args.run_dir
    judge_effort = _resolve_judge_effort(
        judge_model=args.judge_model, judge_effort=args.judge_effort
    )
    # All quality-judge artefacts (quality.jsonl, errors, manifest,
    # downstream analysis outputs) live under a single per-judge subdir
    # so that multi-judge quality runs never step on each other and
    # run_dir stays easy to browse.  Non-reasoning judges use
    # ``llm_judge_quality_<slug>/`` (parallel to the pairwise
    # ``llm_judge_<slug>/`` layout); reasoning judges append the judge
    # effort tier (``__<effort>``) so the same judge at different
    # effort levels does not collide; ``--filter-effort`` (which scopes
    # to a single *generation* effort within a phase2 run_dir) further
    # appends ``__gen_<effort>`` so default-low vs default-high judge
    # runs on the same phase2 dir do not collide either.
    slug = _slugify(args.judge_model.value)
    if judge_effort is not None:
        slug = f"{slug}__{judge_effort.value}"
    if args.filter_effort is not None:
        slug = f"{slug}__gen_{args.filter_effort.value}"
    judge_dir: Path = args.output_dir or (run_dir / f"llm_judge_quality_{slug}")
    judge_dir.mkdir(parents=True, exist_ok=True)

    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        raise SystemExit(f"samples.jsonl not found: {samples_path}")

    records = load_run_sample_records(run_dir)
    log.info(
        "Loaded %d sample records from %s (+ backfill_successes.jsonl if present)",
        len(records),
        samples_path,
    )

    excluded_keywords: tuple[str, ...] = tuple(sorted(set(args.exclude_keyword)))
    if excluded_keywords:
        n_before = len(records)
        records = [r for r in records if r.keyword not in excluded_keywords]
        log.info(
            "Excluded %d records across %d keywords (%s); kept %d records",
            n_before - len(records),
            len(excluded_keywords),
            list(excluded_keywords),
            len(records),
        )

    # Sample scoping: keyword CSV first (smaller-domain), then effort.
    if args.filter_keywords_csv is not None:
        keep_keywords = _load_keyword_filter_csv(args.filter_keywords_csv)
        n_before = len(records)
        records = [r for r in records if r.keyword in keep_keywords]
        log.info(
            "Filter --filter-keywords-csv=%s: kept %d / %d records (CSV has %d unique keywords)",
            args.filter_keywords_csv,
            len(records),
            n_before,
            len(keep_keywords),
        )

    if args.filter_effort is not None:
        n_before = len(records)
        records = [r for r in records if r.effort == args.filter_effort]
        log.info(
            "Filter --filter-effort=%s: kept %d / %d records",
            args.filter_effort.value,
            len(records),
            n_before,
        )

    if not records:
        raise SystemExit(
            "No records remain after filtering.  Check "
            "--filter-keywords-csv / --filter-effort against this run_dir's "
            "samples.jsonl contents."
        )

    _require_uniform_sample_count_per_keyword(records)

    requests = build_quality_requests(
        records=records,
        judge_model=args.judge_model,
        judge_effort=judge_effort,
    )
    log.info("Built %d quality-judge requests (1 per sample)", len(requests))

    output_path = judge_dir / "quality.jsonl"
    manifest_path = judge_dir / "manifest.json"

    manifest = QualityJudgeRunManifest(
        run_dir=run_dir,
        samples_path=samples_path,
        judge_model=args.judge_model,
        judge_effort=judge_effort,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        total_judgments_expected=len(requests),
        started_at=datetime.now(timezone.utc),
        excluded_keywords=excluded_keywords,
        filter_keywords_csv=args.filter_keywords_csv,
        filter_effort=args.filter_effort,
    )
    _require_manifest_compat(manifest_path, manifest)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info("Manifest written: %s", manifest_path)

    provider = _resolve_provider(args.judge_model, args.provider)
    backend = _create_backend(
        model=args.judge_model,
        provider=provider,
        samples_jsonl_path=samples_path,
        effort=judge_effort,
    )
    n_ok, n_err = run_quality_judgments(
        requests=requests,
        backend=backend,
        output_path=output_path,
        max_concurrency=args.max_concurrency,
    )

    # Re-count from disk so resumed rows are reflected in the manifest totals.
    errors_path = errors_path_for(output_path)
    manifest.completed_at = datetime.now(timezone.utc)
    manifest.completed_judgments = _count_nonempty_lines(output_path)
    manifest.error_judgments = _count_nonempty_lines(errors_path)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    log.info("Done. Successes: %s", output_path)
    log.info("Errors: %s", errors_path)
    log.info(
        "Run totals: ok=%d err=%d (this run added +%d ok, +%d err)",
        manifest.completed_judgments,
        manifest.error_judgments,
        n_ok,
        n_err,
    )


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _resolve_provider(model: JudgeModelName, provider: ProviderName | None) -> ProviderName:
    """Validate ``--provider`` against the model's allowed set.

    Mirrors the pairwise sibling.  Defaults to the canonical
    :data:`JUDGE_MODEL_PROVIDER` entry when ``--provider`` is omitted.
    """
    allowed = JUDGE_MODEL_ALLOWED_PROVIDERS[model]
    if provider is None:
        return JUDGE_MODEL_PROVIDER[model]
    if provider not in allowed:
        allowed_values = sorted(p.value for p in allowed)
        raise ValueError(
            f"--provider={provider.value!r} is not supported for "
            f"--judge-model={model.value!r}. Allowed providers: {allowed_values}. "
            f"See JUDGE_MODEL_ALLOWED_PROVIDERS in src/model_registry.py."
        )
    if provider == ProviderName.GOOGLE_AI_STUDIO:
        import os

        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError(
                "--provider=google_ai_studio requires the GEMINI_API_KEY "
                "environment variable to be set (read by the google-genai SDK)."
            )
    return provider


def _create_backend(
    *,
    model: JudgeModelName,
    provider: ProviderName,
    samples_jsonl_path: Path,
    effort: EffortName | None,
) -> QualityJudgeBackend:
    """Return the quality backend that serves ``(model, provider)``.

    Dispatch is centralised here rather than via a provider lookup on
    :data:`~src.model_registry.JUDGE_MODEL_PROVIDER` because each
    backend class has its own import surface (OpenAI / Vertex / Anthropic
    / AI Studio / Bedrock SDK); a ``match`` keeps the wiring explicit so
    a missing case fails at call time rather than defaulting silently.

    ``effort`` is required (already resolved by
    :func:`_resolve_judge_effort`) for reasoning judges and ``None`` for
    non-reasoning judges; passing it the wrong way for a given branch is
    an internal-logic error, not a user-facing one.
    """
    match (model, provider):
        # Non-reasoning judges.
        case (JudgeModelName.GPT_4_1, ProviderName.OPENAI):
            return OpenAIQualityJudgeBackend(model=model, samples_jsonl_path=samples_jsonl_path)
        case (JudgeModelName.GEMINI_2_5_FLASH, ProviderName.VERTEX):
            return GeminiQualityJudgeBackend(model=model, samples_jsonl_path=samples_jsonl_path)
        case (JudgeModelName.GEMINI_2_5_FLASH, ProviderName.GOOGLE_AI_STUDIO):
            return GeminiQualityJudgeBackend(
                model=model,
                samples_jsonl_path=samples_jsonl_path,
                config=GoogleAIStudioConfig(),
            )
        case (JudgeModelName.CLAUDE_HAIKU_4_5, ProviderName.ANTHROPIC):
            return AnthropicQualityJudgeBackend(model=model, samples_jsonl_path=samples_jsonl_path)
        # Reasoning judges (strong-judge ablation; see
        # rebuttal/strong_judge_design.md and PR #255).
        case (JudgeModelName.GPT_5_4, ProviderName.OPENAI):
            assert effort is not None  # see _resolve_judge_effort
            return OpenAIReasoningQualityJudgeBackend(
                model=model, effort=effort, samples_jsonl_path=samples_jsonl_path
            )
        case (JudgeModelName.GEMINI_3_1_PRO, ProviderName.VERTEX):
            assert effort is not None
            return GeminiReasoningQualityJudgeBackend(
                model=model, effort=effort, samples_jsonl_path=samples_jsonl_path
            )
        case (JudgeModelName.GEMINI_3_1_PRO, ProviderName.GOOGLE_AI_STUDIO):
            assert effort is not None
            return GeminiReasoningQualityJudgeBackend(
                model=model,
                effort=effort,
                samples_jsonl_path=samples_jsonl_path,
                config=GoogleAIStudioConfig(),
            )
        case (JudgeModelName.CLAUDE_SONNET_4_6, ProviderName.BEDROCK):
            assert effort is not None
            return AnthropicReasoningQualityJudgeBackend(
                model=model, effort=effort, samples_jsonl_path=samples_jsonl_path
            )
        case _:
            raise ValueError(
                f"No quality judge backend registered for (model={model.value!r}, provider={provider.value!r})"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_langfuse()
