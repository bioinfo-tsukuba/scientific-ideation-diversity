#!/usr/bin/env python3
"""Run LLM-as-judge pairwise similarity evaluation on an effort-diversity run.

Uses the LiveIdeaBench v4 fluency-critic prompt (verbatim) with structured
output so that the ABCD answer comes back without any regex retry that could
bias per-pair attempt counts.  Evaluates both orderings (position-bias
control) and stratifies pairs by embedding distance (top-m + bottom-m per
(keyword, effort) bucket).

Usage:
    uv run python scripts/experiment/llm_judge/llm_judge_pairwise.py \\
        --run-dir results/effort_diversity/phase1_claude_22kw_50x4_facet_seed42 \\
        --judge-model gpt-4.1 \\
        --embedding-model text-embedding-3-large \\
        --top-m 10 --bottom-m 10

See docs/20260418_llm_judge_plan.md for the full design rationale.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.artifacts import load_run_sample_records
from src.model_registry import (
    JUDGE_MODEL_ALLOWED_PROVIDERS,
    JUDGE_MODEL_PROVIDER,
    EmbeddingModelName,
    GoogleAIStudioConfig,
    JudgeModelName,
    ProviderName,
)
from src.providers.llm_judge.anthropic_backend import AnthropicJudgeBackend
from src.providers.llm_judge.base import LLMJudgeBackend
from src.providers.llm_judge.gemini_backend import GeminiJudgeBackend
from src.providers.llm_judge.openai_backend import OpenAIJudgeBackend
from src.providers.tracing import flush_langfuse
from src.schemas.llm_judge import JudgeRunManifest
from src.schemas.sample import SampleRecord
from src.services.llm_judge_service import (
    PROMPT_TEMPLATE_VERSION,
    build_judge_requests,
    errors_path_for,
    run_judgments,
    select_stratified_pairs,
)
from src.text import slugify

# Embedding models whose ``embeddings.npy`` can rank pairs for the judge.
# The gen pipeline writes each model's output to
# ``run_dir/<slugify(model_id)>/embeddings.npy`` via
# ``build_embedding_output_dir``, so the judge resolves the same path.
_SUPPORTED_EMBEDDING_MODELS: tuple[EmbeddingModelName, ...] = (
    EmbeddingModelName.AMAZON_TITAN_EMBED_TEXT_V2_0,
    EmbeddingModelName.SPECTER2,
    EmbeddingModelName.SPECTER2_ADHOC_QUERY,
    EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
)


def resolve_embeddings_path(run_dir: Path, embedding_model: EmbeddingModelName) -> Path:
    """Return the ``embeddings.npy`` path for this run and model.

    Mirrors the gen pipeline's ``build_embedding_output_dir`` convention
    (``run_dir / slugify(model_id) / embeddings.npy``).  Raises
    ``SystemExit`` with the expected path if the file is absent.
    """
    path = run_dir / slugify(embedding_model.value) / "embeddings.npy"
    if not path.exists():
        raise SystemExit(f"embeddings.npy not found for embedding_model={embedding_model.value!r}: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Effort-diversity run directory containing samples.jsonl",
    )
    parser.add_argument(
        "--embedding-model",
        type=EmbeddingModelName,
        default=EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
        choices=list(_SUPPORTED_EMBEDDING_MODELS),
        metavar=f"{{{','.join(m.value for m in _SUPPORTED_EMBEDDING_MODELS)}}}",
        help=(
            "Embedding model whose combined distances rank pairs. "
            f"Default: {EmbeddingModelName.TEXT_EMBEDDING_3_LARGE.value}"
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=JudgeModelName,
        default=JudgeModelName.GPT_4_1,
        choices=list(JudgeModelName),
        metavar=f"{{{','.join(m.value for m in JudgeModelName)}}}",
        help=f"Judge model (default: {JudgeModelName.GPT_4_1.value})",
    )
    parser.add_argument(
        "--provider",
        type=ProviderName,
        default=None,
        choices=[ProviderName.OPENAI, ProviderName.VERTEX, ProviderName.ANTHROPIC, ProviderName.GOOGLE_AI_STUDIO],
        metavar="{openai,vertex,anthropic,google_ai_studio}",
        help=(
            "Provider that serves --judge-model. If omitted, the default "
            "provider from JUDGE_MODEL_PROVIDER is used. Models that accept "
            "multiple providers (e.g. gemini-2.5-flash on Vertex or "
            "Google AI Studio) need an explicit choice only when the "
            "non-default path is wanted."
        ),
    )
    parser.add_argument("--top-m", type=int, default=10, help="Pairs to take from the top (most distant)")
    parser.add_argument("--bottom-m", type=int, default=10, help="Pairs to take from the bottom (most similar)")
    parser.add_argument(
        "--middle-m",
        type=int,
        default=0,
        help="Pairs to take around the median rank (0 = disabled, used for calibration curve)",
    )
    parser.add_argument(
        "--no-both-orders",
        dest="both_orders",
        action="store_false",
        default=True,
        help="Disable position-bias control (default: both orders evaluated)",
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --run-dir")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=[],
        metavar="KW",
        help=(
            "Drop all samples for KW before the uniformity check, ranking, and "
            "judging. Can be repeated. Use for keywords whose generation yield "
            "is below 30 (e.g., safety-filter blocks) so the remaining keywords "
            "still satisfy the uniform-count invariant."
        ),
    )
    return parser.parse_args()


def _require_uniform_sample_count_per_keyword(records: list[SampleRecord]) -> None:
    """Abort if sample counts differ across keywords.

    A balanced design (n_effort * n_sample_per_bucket per keyword) is a
    prerequisite for fair cross-keyword comparison; an imbalance signals
    partial generation, dropped keywords that weren't finalised away, or
    accidentally merged run dirs, and would bias downstream aggregates.
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


# Fields that must match between a prior manifest and the current args for
# resume to be safe.  The dedup key in the JSONL is
# (keyword, effort, sample_i, sample_j, order, judge_model), so changing any
# of these sampling-critical knobs (ranking basis, stratum widths, prompt
# text) would produce rows that share a dedup key with rows from an
# incompatible configuration and silently contaminate downstream analyses.
_MANIFEST_RESUME_COMPAT_FIELDS: tuple[str, ...] = (
    "embedding_model_for_sort",
    "prompt_template_version",
    "top_m",
    "middle_m",
    "bottom_m",
    "both_orders",
    "judge_model",
    "excluded_keywords",
)


def _require_manifest_compat(manifest_path: Path, candidate: JudgeRunManifest) -> None:
    """Refuse to resume if the on-disk manifest disagrees on sampling config.

    The JSONL dedup key includes judge_model but nothing about the embedding
    ranking, stratum widths, or prompt text, so without this check a second
    invocation with (say) a different ``--top-m`` would append rows that
    analysis scripts cannot distinguish from rows produced under the prior
    config.  We error out early and tell the caller to use a different
    ``--output-dir`` or delete the stale artefact.
    """
    if not manifest_path.exists():
        return
    prior = JudgeRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for field in _MANIFEST_RESUME_COMPAT_FIELDS:
        prior_val = getattr(prior, field)
        new_val = getattr(candidate, field)
        if prior_val != new_val:
            mismatches.append(f"  {field}: prior={prior_val!r} new={new_val!r}")
    if mismatches:
        raise SystemExit(
            f"Refusing to resume {manifest_path}: sampling config changed.\n"
            + "\n".join(mismatches)
            + "\nUse --output-dir to write to a fresh location, or delete the "
            "existing manifest + JSONL if the prior run should be discarded."
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("llm_judge_pairwise")

    run_dir: Path = args.run_dir
    slug = _slugify(args.judge_model.value)
    # All judge artefacts (pairwise.jsonl, errors, manifest, downstream
    # analysis outputs) live under a single ``llm_judge_<slug>/`` subdir per
    # judge model so that multi-judge runs (e.g. Phase B.3's Opus + GPT-4.1)
    # never step on each other and run_dir stays easy to browse.
    judge_dir: Path = args.output_dir or (run_dir / f"llm_judge_{slug}")
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

    emb_path = resolve_embeddings_path(run_dir, args.embedding_model)
    embeddings = np.load(emb_path)
    log.info(
        "Loaded embeddings %s shape=%s (%s)",
        emb_path,
        embeddings.shape,
        args.embedding_model.value,
    )

    excluded_keywords: tuple[str, ...] = tuple(sorted(set(args.exclude_keyword)))
    if excluded_keywords:
        keep_mask = np.array([r.keyword not in excluded_keywords for r in records], dtype=bool)
        n_dropped = int((~keep_mask).sum())
        records = [r for r, m in zip(records, keep_mask) if m]
        embeddings = embeddings[keep_mask]
        log.info(
            "Excluded %d records across %d keywords (%s); kept %d records, embeddings shape=%s",
            n_dropped,
            len(excluded_keywords),
            list(excluded_keywords),
            len(records),
            embeddings.shape,
        )

    pairs = select_stratified_pairs(
        records=records,
        embeddings=embeddings,
        embedding_model=args.embedding_model,
        top_m=args.top_m,
        bottom_m=args.bottom_m,
        middle_m=args.middle_m,
    )
    expected = len(pairs) * (2 if args.both_orders else 1)
    log.info(
        "Selected %d pairs -> %d expected judgments (both_orders=%s)",
        len(pairs),
        expected,
        args.both_orders,
    )

    requests = build_judge_requests(
        pairs=pairs,
        records=records,
        judge_model=args.judge_model,
        both_orders=args.both_orders,
    )

    output_path = judge_dir / "pairwise.jsonl"
    manifest_path = judge_dir / "manifest.json"

    manifest = JudgeRunManifest(
        run_dir=run_dir,
        samples_path=samples_path,
        embeddings_path=emb_path,
        embedding_model_for_sort=args.embedding_model,
        judge_model=args.judge_model,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        top_m=args.top_m,
        middle_m=args.middle_m,
        bottom_m=args.bottom_m,
        both_orders=args.both_orders,
        total_judgments_expected=expected,
        started_at=datetime.now(timezone.utc),
        excluded_keywords=excluded_keywords,
    )
    _require_manifest_compat(manifest_path, manifest)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info("Manifest written: %s", manifest_path)

    _require_uniform_sample_count_per_keyword(records)
    provider = _resolve_provider(args.judge_model, args.provider)
    backend = _create_backend(
        model=args.judge_model,
        provider=provider,
        samples_jsonl_path=samples_path,
    )
    n_ok, n_err = run_judgments(
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

    If ``provider`` is None the canonical :data:`JUDGE_MODEL_PROVIDER`
    entry is used.  Otherwise the choice must appear in
    :data:`JUDGE_MODEL_ALLOWED_PROVIDERS` for the model.  Mirrors
    :func:`scripts.run_diversity_experiment.validate_idea_model_and_provider`.
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
) -> LLMJudgeBackend:
    """Return the backend that serves ``(model, provider)``.

    Dispatch is centralised here rather than via a provider lookup on
    :data:`~src.model_registry.JUDGE_MODEL_PROVIDER` because each
    backend class has its own import surface (OpenAI / Vertex / Anthropic
    / AI Studio SDK); a ``match`` keeps the wiring explicit so a missing
    case fails at call time rather than defaulting silently.
    """
    match (model, provider):
        case (JudgeModelName.GPT_4_1, ProviderName.OPENAI):
            return OpenAIJudgeBackend(model=model, samples_jsonl_path=samples_jsonl_path)
        case (JudgeModelName.GEMINI_2_5_FLASH, ProviderName.VERTEX):
            return GeminiJudgeBackend(model=model, samples_jsonl_path=samples_jsonl_path)
        case (JudgeModelName.GEMINI_2_5_FLASH, ProviderName.GOOGLE_AI_STUDIO):
            return GeminiJudgeBackend(
                model=model,
                samples_jsonl_path=samples_jsonl_path,
                config=GoogleAIStudioConfig(),
            )
        case (JudgeModelName.CLAUDE_HAIKU_4_5, ProviderName.ANTHROPIC):
            return AnthropicJudgeBackend(model=model, samples_jsonl_path=samples_jsonl_path)
        case _:
            raise ValueError(
                f"No pairwise judge backend registered for (model={model.value!r}, provider={provider.value!r})"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_langfuse()
