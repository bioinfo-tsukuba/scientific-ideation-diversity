#!/usr/bin/env python3
"""Generate judge-free idea samples by effort and analyze embedding diversity.

This pilot script is intentionally simple:

1. Reuse the LiveIdeaBench idea prompt for one or more keywords.
2. Generate ``n`` samples for each requested keyword/effort pair.
3. Embed the generated idea texts with a selected embedding backend.
4. Optionally fan out the same generations to Titan, OpenAI, and SPECTER2 in one run.
5. Save raw samples and lightweight diversity artifacts:
   - per-keyword/per-effort summary metrics
   - per-effort aggregate summary metrics
   - centroid distance matrix
   - PCA coordinates
   - an SVG scatter plot
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from scripts.render_effort_diversity_report import (
    render_category_diversity_svg,
    render_diversity_svg,
)
from src.artifacts import (
    EmbeddingArtifactWriter,
    allocate_output_dir,
    append_jsonl,
    clone_records_for_embedding,
    copy_cached_embedding_artifacts_to_writer,
    load_sample_records,
    materialize_generation_artifacts,
    save_pca_coordinates,
    sort_sample_records,
    write_csv,
    write_matrix_csv,
    write_sample_records,
)
from src.metrics import (
    build_category_summary_rows,
    build_centroid_distance_rows,
    build_summary_by_effort_rows,
    build_summary_by_keyword_effort_rows,
    pca_2d,
)
from src.model_registry import (
    IDEA_MODEL_ALLOWED_PROVIDERS,
    AwsRegion,
    BedrockConfig,
    EffortName,
    EmbeddingModelName,
    GenerationProviderName,
    GoogleAIStudioConfig,
    IdeaModelName,
    PromptStyleName,
    ProviderName,
    VertexAIConfig,
)
from src.providers.embedding.base import EmbeddingBackend
from src.providers.tracing import flush_langfuse
from src.schemas.embedding import EmbeddingRunSpec
from src.schemas.generation import StructuredBgIdeaOutput, StructuredFacetOutput, StructuredOutput
from src.schemas.sample import SampleRecord
from src.schemas.validation import PydanticValidationStatus
from src.services.embedding_service import create_embedding_backend, resolve_embed_provider
from src.services.generation_service import generate_idea
from src.text import (
    clean_text,
    extract_record_texts,
    slugify,
)
from src.visualizer import save_svg_scatter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFORTS = ["none", "low", "medium", "high"]
DEFAULT_EMBED_MODEL = EmbeddingModelName.AMAZON_TITAN_EMBED_TEXT_V2_0
DEFAULT_EMBED_PROVIDER = "auto"
DEFAULT_OPENAI_EMBED_MODEL = EmbeddingModelName.TEXT_EMBEDDING_3_LARGE
DEFAULT_SPECTER2_ADAPTER_MODEL = EmbeddingModelName.SPECTER2_ADHOC_QUERY
DEFAULT_SPECTER2_BASE_MODEL_ID = EmbeddingModelName.SPECTER2_BASE.value
DEFAULT_SPECTER2_BATCH_SIZE = 16
DEFAULT_LIVEIDEABENCH_ROOT = ROOT / "external" / "liveideabench"
DEFAULT_MULTI_EMBED_MODELS: list[EmbeddingModelName] = [
    DEFAULT_EMBED_MODEL,
    DEFAULT_OPENAI_EMBED_MODEL,
    DEFAULT_SPECTER2_ADAPTER_MODEL,
]
GENERATION_ARTIFACT_FILENAMES = ["errors.jsonl", "backfill_successes.jsonl"]
TEXT_COMPONENTS = ["combined", "background", "idea"]
class SampleGenerationError(Exception):
    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload["error_message"])
        self.payload = payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot: generate ideas by effort and analyze embedding diversity.",
    )
    parser.add_argument(
        "--keyword",
        nargs="+",
        default=None,
        help="One or more keywords to probe, e.g. 'relativity' 'dark matter'.",
    )
    parser.add_argument(
        "--keyword-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV/TXT file containing keywords. CSV should contain a 'keyword' column "
            "and can include 'category'. For pinned LiveIdeaBench metadata, point this to "
            "external/liveideabench/csvs/keyword_classifications.csv."
        ),
    )
    parser.add_argument(
        "--liveideabench-root",
        type=Path,
        default=DEFAULT_LIVEIDEABENCH_ROOT,
        help=(
            "Pinned LiveIdeaBench source tree, typically the official submodule at "
            "external/liveideabench. Used for prompt defaults and provenance recording."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Optional prompt JSON override. Defaults to <liveideabench-root>/utils/prompts.json "
            "so the prompt source can be pinned and hashed."
        ),
    )
    parser.add_argument(
        "--idea-model",
        required=True,
        choices=[m.value for m in IdeaModelName],
        help="Model ID for idea generation (e.g. claude-sonnet-4-6, gpt-5.4, gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=[p.value for p in GenerationProviderName],
        help=(
            "Generation provider. Must match the canonical provider for --idea-model "
            "(see IDEA_MODEL_PROVIDER in src/model_registry.py). A mismatch is a hard error."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        default=PromptStyleName.STRUCTURED_BG_IDEA.value,
        choices=[e.value for e in PromptStyleName],
        help="Output schema style. 'structured_bg_idea' uses {background, idea}; 'structured_facet' uses {purpose, mechanism, evaluation}.",
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=DEFAULT_EFFORTS,
        choices=DEFAULT_EFFORTS,
        help="Effort conditions. 'none' omits adaptive thinking entirely.",
    )
    parser.add_argument(
        "--samples-per-effort",
        type=int,
        default=50,
        help="Number of generations to collect for each effort condition.",
    )
    parser.add_argument(
        "--specter2-batch-size",
        type=int,
        default=DEFAULT_SPECTER2_BATCH_SIZE,
        help="Batch size for local SPECTER2 embedding inference.",
    )
    _selectable_embed_models = [
        m for m in EmbeddingModelName if m != EmbeddingModelName.SPECTER2_BASE
    ]
    parser.add_argument(
        "--embed-models",
        nargs="+",
        type=EmbeddingModelName,
        default=DEFAULT_MULTI_EMBED_MODELS,
        choices=_selectable_embed_models,
        metavar="EMBED_MODEL",
        help=(
            "Embedding models to run (space-separated values from "
            f"{{{', '.join(m.value for m in _selectable_embed_models)}}}). "
            "Defaults to Titan + OpenAI + SPECTER2 adhoc_query. Pass a subset "
            "(e.g. 'allenai/specter2_adhoc_query') to re-embed only one model; "
            "existing embeddings for the others are left untouched on disk."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum number of concurrent generation or embedding requests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/effort_diversity/<timestamp>.",
    )
    parser.add_argument(
        "--resume-output-dir",
        type=Path,
        default=None,
        help="Resume from an existing run directory containing samples.jsonl. Generation is skipped and embedding/aggregation is rerun.",
    )
    parser.add_argument(
        "--resume-generation-output-dir",
        type=Path,
        default=None,
        help=(
            "Resume generation into an existing run directory by backfilling only missing "
            "keyword/category/effort/sample_index tasks, then continue to embedding."
        ),
    )
    parser.add_argument(
        "--embed-max-concurrency",
        type=int,
        default=None,
        help="Optional concurrency override for embedding only. Defaults to --max-concurrency.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Stop after generation or resume-generation backfill without running embedding/aggregation.",
    )
    return parser.parse_args()


def validate_idea_model_and_provider(idea_model: str, provider: str) -> None:
    """Reject runs where --provider is not allowed for the chosen model.

    The argparse layer already constrains both args to their respective enum
    values.  This function enforces the relational constraint (model ↔ provider)
    defined in :data:`IDEA_MODEL_ALLOWED_PROVIDERS` and fails loudly on mismatch
    so the user picks a coherent pair explicitly (no silent fallback).

    Some models accept more than one provider (e.g., Gemini 3.1 Pro can run
    on Vertex AI or Google AI Studio); any provider in the allowed set is
    accepted, so long as the model is registered.
    """
    model = IdeaModelName(idea_model)
    allowed = IDEA_MODEL_ALLOWED_PROVIDERS[model]
    if GenerationProviderName(provider) not in allowed:
        allowed_values = sorted(p.value for p in allowed)
        raise ValueError(
            f"--provider={provider!r} is not supported for "
            f"--idea-model={idea_model!r}. Allowed providers: {allowed_values}. "
            f"See IDEA_MODEL_ALLOWED_PROVIDERS in src/model_registry.py."
        )

    if provider == GenerationProviderName.GOOGLE_AI_STUDIO.value:
        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError(
                "--provider=google_ai_studio requires the GEMINI_API_KEY "
                "environment variable to be set (read by the google-genai SDK). "
                "Export it in your shell or load it via direnv before running."
            )


def resolve_keyword_rows(args: argparse.Namespace) -> list[dict[str, Optional[str]]]:
    if args.keyword and args.keyword_file:
        raise ValueError("Specify either --keyword or --keyword-file, not both.")

    if args.keyword_file:
        rows = load_keyword_rows_from_file(args.keyword_file)
        if not rows:
            raise ValueError(f"No keywords found in {args.keyword_file}")
        return rows

    if args.keyword:
        return [{"keyword": keyword, "category": None} for keyword in args.keyword]

    raise ValueError("You must provide --keyword or --keyword-file.")


def resolve_liveideabench_root(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_prompt_file(args: argparse.Namespace) -> Path:
    prompt_path = (
        args.prompt_file
        if args.prompt_file is not None
        else args.liveideabench_root / "utils" / "prompts.json"
    )
    prompt_path = prompt_path.expanduser().resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path


def load_keyword_rows_from_file(path: Path) -> list[dict[str, Optional[str]]]:
    if path.suffix.lower() == ".txt":
        rows: list[dict[str, Optional[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            keyword = line.strip()
            if keyword:
                rows.append({"keyword": keyword, "category": None})
        return rows

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {path}")

        keyword_field = None
        category_field = None
        for field in reader.fieldnames:
            lowered = field.lower()
            if lowered == "keyword":
                keyword_field = field
            elif lowered == "category":
                category_field = field

        if keyword_field is None:
            raise ValueError(f"CSV file must contain a 'keyword' column: {path}")

        rows = []
        for row in reader:
            keyword = (row.get(keyword_field) or "").strip()
            if not keyword:
                continue
            category = (row.get(category_field) or "").strip() if category_field else None
            rows.append({"keyword": keyword, "category": category or None})
        return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: list[str], *, cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def describe_git_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    repo_root_text = run_git(["rev-parse", "--show-toplevel"], cwd=resolved.parent)
    repo_root = Path(repo_root_text).resolve() if repo_root_text else None
    repo_commit = run_git(["rev-parse", "HEAD"], cwd=resolved.parent) if repo_root else None
    repo_relative_path = None
    if repo_root is not None:
        try:
            repo_relative_path = str(resolved.relative_to(repo_root))
        except ValueError:
            repo_relative_path = None

    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "repo_root": str(repo_root) if repo_root is not None else None,
        "repo_commit": repo_commit,
        "repo_relative_path": repo_relative_path,
    }


def build_benchmark_provenance(args: argparse.Namespace, *, prompt_file: Path) -> dict[str, Any]:
    liveideabench_root = args.liveideabench_root
    liveideabench_repo_root_text = (
        run_git(["rev-parse", "--show-toplevel"], cwd=liveideabench_root)
        if liveideabench_root.exists()
        else None
    )
    liveideabench_repo_root = (
        Path(liveideabench_repo_root_text).resolve()
        if liveideabench_repo_root_text
        else None
    )
    liveideabench_commit = (
        run_git(["rev-parse", "HEAD"], cwd=liveideabench_root)
        if liveideabench_root.exists()
        else None
    )
    keyword_source = None
    if args.keyword_file is not None:
        keyword_source = describe_git_file(args.keyword_file)

    return {
        "liveideabench_root": str(liveideabench_root),
        "liveideabench_repo_root": (
            str(liveideabench_repo_root) if liveideabench_repo_root is not None else None
        ),
        "liveideabench_commit": liveideabench_commit,
        "prompt_source": describe_git_file(prompt_file),
        "keyword_source": keyword_source,
        "keyword_input_mode": "file" if args.keyword_file is not None else "cli_keywords",
        "notes": (
            "category values from keyword metadata are auxiliary benchmark metadata, not gold labels"
        ),
    }


def infer_keyword_rows_from_records(records: list[SampleRecord]) -> list[dict[str, Optional[str]]]:
    rows = sorted(
        {(record.keyword, record.category) for record in records},
        key=lambda item: (item[1] or "", item[0]),
    )
    return [{"keyword": keyword, "category": category} for keyword, category in rows]


def infer_attempted_samples_from_records(records: list[SampleRecord]) -> int:
    group_to_max_index: dict[tuple[Optional[str], str, str], int] = {}
    for record in records:
        key = (record.category, record.keyword, record.effort)
        group_to_max_index[key] = max(group_to_max_index.get(key, -1), record.sample_index)
    return sum(max_index + 1 for max_index in group_to_max_index.values())


def build_generation_tasks(
    *,
    keyword_rows: list[dict[str, Optional[str]]],
    efforts: list[str],
    samples_per_effort: int,
) -> list[tuple[str, Optional[str], str, int]]:
    return [
        (keyword_row["keyword"], keyword_row.get("category"), effort_name, sample_index)
        for sample_index in range(samples_per_effort)
        for keyword_row in keyword_rows
        for effort_name in efforts
    ]


def sample_task_key(
    *,
    keyword: str,
    category: Optional[str],
    effort: str,
    sample_index: int,
) -> tuple[Optional[str], str, str, int]:
    return (category, keyword, effort, sample_index)


def record_task_key(record: SampleRecord) -> tuple[Optional[str], str, str, int]:
    return sample_task_key(
        keyword=record.keyword,
        category=record.category,
        effort=record.effort,
        sample_index=record.sample_index,
    )


def infer_missing_generation_tasks(
    *,
    existing_records: list[SampleRecord],
    keyword_rows: list[dict[str, Optional[str]]],
    efforts: list[str],
    samples_per_effort: int,
) -> list[tuple[str, Optional[str], str, int]]:
    existing_keys = {record_task_key(record) for record in existing_records}
    all_tasks = build_generation_tasks(
        keyword_rows=keyword_rows,
        efforts=efforts,
        samples_per_effort=samples_per_effort,
    )
    return [
        (keyword, category, effort, sample_index)
        for keyword, category, effort, sample_index in all_tasks
        if sample_task_key(
            keyword=keyword,
            category=category,
            effort=effort,
            sample_index=sample_index,
        )
        not in existing_keys
    ]


def load_prompts(prompt_file: Path) -> dict[str, dict[str, str]]:
    with open(prompt_file, encoding="utf-8") as f:
        return json.load(f)


def annotate_api_calls(
    api_calls: list[dict[str, Any]],
    *,
    call_stage: str,
) -> list[dict[str, Any]]:
    annotated = []
    for call in api_calls:
        row = dict(call)
        row["call_stage"] = call_stage
        annotated.append(row)
    return annotated


def resolve_prompt_template(
    *,
    prompt_file: Path,
    prompt_style: PromptStyleName,
) -> tuple[str, type[StructuredOutput]]:
    """Load the idea prompt template and select the matching output schema.

    For ``structured_bg_idea``, uses {background, idea}.
    For ``structured_facet``, uses {purpose, mechanism, evaluation}.
    """
    prompts = load_prompts(prompt_file)
    schema: type[StructuredOutput]
    if prompt_style == PromptStyleName.STRUCTURED_FACET:
        schema = StructuredFacetOutput
    else:
        schema = StructuredBgIdeaOutput
    return prompts["idea_prompt"]["description"], schema


def dedupe_preserve_order(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return unique_values


def build_embedding_output_dir(base_output_dir: Path, model_id: str) -> Path:
    return base_output_dir / slugify(model_id)


def build_embedding_run_specs(
    *,
    base_output_dir: Path,
    embed_models: list[EmbeddingModelName],
) -> list[EmbeddingRunSpec]:
    specs: list[EmbeddingRunSpec] = []
    seen: set[EmbeddingModelName] = set()
    for model in embed_models:
        if model in seen:
            continue
        seen.add(model)
        provider = resolve_embed_provider("auto", model)
        output_dir = build_embedding_output_dir(base_output_dir, model.value)
        specs.append(
            EmbeddingRunSpec(
                embedding_model=model,
                provider=ProviderName(provider),
                output_dir=output_dir,
            )
        )
    return specs


def generate_one_sample(
    *,
    keyword: str,
    category: Optional[str],
    idea_model: str,
    provider: str,
    effort_name: str,
    sample_index: int,
    idea_prompt_template: str,
    structured_output_schema: type[StructuredOutput],
    embedding_model: str,
    prompt_style: str,
) -> SampleRecord:
    idea_prompt = idea_prompt_template.replace("{{keywords}}", keyword)
    error_base = {
        "keyword": keyword,
        "category": category,
        "prompt_style": prompt_style,
        "effort": effort_name,
        "sample_index": sample_index,
        "idea_model": idea_model,
        "embedding_model": embedding_model,
    }
    bedrock_region_str = (
        os.getenv("BEDROCK_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or AwsRegion.AP_NORTHEAST_1.value
    )
    bedrock_cfg = BedrockConfig(region=AwsRegion(bedrock_region_str))
    vertex_ai_project = os.getenv("VERTEX_AI_PROJECT", "")
    vertex_ai_location = os.getenv("VERTEX_AI_LOCATION", "global")
    vertex_ai_cfg = VertexAIConfig(project=vertex_ai_project, location=vertex_ai_location)
    google_ai_studio_cfg = GoogleAIStudioConfig()
    try:
        generation = generate_idea(
            idea_prompt,
            idea_model=IdeaModelName(idea_model),
            provider=GenerationProviderName(provider),
            effort=EffortName(effort_name),
            structured_output_schema=structured_output_schema,
            bedrock_config=bedrock_cfg,
            vertex_ai_config=vertex_ai_cfg,
            google_ai_studio_config=google_ai_studio_cfg,
        )
    except Exception as exc:
        raise SampleGenerationError(
            {
                **error_base,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
        ) from exc
    try:
        so = generation.structured_output
        idea = clean_text(so.combined_text())
        return SampleRecord(
            keyword=keyword,
            category=category,
            prompt_style=prompt_style,
            effort=effort_name,
            sample_index=sample_index,
            idea_model=idea_model,
            embedding_model=embedding_model,
            idea=idea,
            full_response=generation.text,
            structured_output=so,
            pydantic_validation_status=PydanticValidationStatus.PASSED,
            pydantic_validation_error=None,
            response_reasoning=clean_text(generation.response_reasoning)
            if generation.response_reasoning
            else None,
            has_reasoning_block=generation.has_reasoning_block,
            stop_reason=generation.stop_reason,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            total_tokens=generation.total_tokens,
            raw_api_response=generation.raw_api_response,
            api_calls=generation.api_calls,
            embedding_input_tokens=0,
            word_count=len(idea.split()),
            char_count=len(idea),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        raise SampleGenerationError(
            {
                **error_base,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stop_reason": generation.stop_reason,
                "usage": {
                    "inputTokens": generation.input_tokens,
                    "outputTokens": generation.output_tokens,
                    "totalTokens": generation.total_tokens,
                },
                "raw_api_response": generation.raw_api_response,
                "api_calls": generation.api_calls,
                "timestamp": datetime.now().isoformat(),
            }
        ) from exc


def generate_samples(
    *,
    keyword_rows: list[dict[str, Optional[str]]],
    idea_model: str,
    provider: str,
    efforts: list[str],
    samples_per_effort: int,
    output_dir: Path,
    embedding_model: str,
    max_concurrency: int,
    prompt_file: Path,
    prompt_style: PromptStyleName,
    tasks: Optional[list[tuple[str, Optional[str], str, int]]] = None,
    success_log_filename: Optional[str] = None,
) -> list[SampleRecord]:
    idea_prompt_template, structured_output_schema = resolve_prompt_template(
        prompt_file=prompt_file,
        prompt_style=prompt_style,
    )

    samples_jsonl = output_dir / "samples.jsonl"
    errors_jsonl = output_dir / "errors.jsonl"
    success_log_path = (
        output_dir / success_log_filename if success_log_filename is not None else None
    )
    records: list[SampleRecord] = []
    append_lock = threading.Lock()
    if tasks is None:
        tasks = build_generation_tasks(
            keyword_rows=keyword_rows,
            efforts=efforts,
            samples_per_effort=samples_per_effort,
        )
    if not tasks:
        return []

    max_workers = max(1, min(max_concurrency, len(tasks)))

    def submit_task(
        executor: concurrent.futures.ThreadPoolExecutor,
        task: tuple[str, Optional[str], str, int],
    ) -> concurrent.futures.Future[SampleRecord]:
        keyword, category, effort_name, sample_index = task
        return executor.submit(
            generate_one_sample,
            keyword=keyword,
            category=category,
            idea_model=idea_model,
            provider=provider,
            effort_name=effort_name,
            sample_index=sample_index,
            idea_prompt_template=idea_prompt_template,
            structured_output_schema=structured_output_schema,
            embedding_model=embedding_model,
            prompt_style=prompt_style.value,
        )

    task_iter = iter(tasks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task: dict[concurrent.futures.Future[SampleRecord], tuple[str, Optional[str], str, int]] = {}
        for _ in range(max_workers):
            try:
                task = next(task_iter)
            except StopIteration:
                break
            future = submit_task(executor, task)
            future_to_task[future] = task

        completed_index = 0
        while future_to_task:
            done, _ = concurrent.futures.wait(
                future_to_task,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                task = future_to_task.pop(future)
                completed_index += 1
                try:
                    record = future.result()
                    records.append(record)
                    with append_lock:
                        append_jsonl(samples_jsonl, record.model_dump(mode="json"))
                        if success_log_path is not None:
                            append_jsonl(success_log_path, record.model_dump(mode="json"))
                except SampleGenerationError as exc:
                    with append_lock:
                        append_jsonl(errors_jsonl, exc.payload)
                if completed_index % 10 == 0 or completed_index == len(tasks):
                    print(
                        f"[generate] completed {completed_index}/{len(tasks)}",
                        flush=True,
                    )
                try:
                    next_task = next(task_iter)
                except StopIteration:
                    continue
                next_future = submit_task(executor, next_task)
                future_to_task[next_future] = next_task

    return sort_sample_records(records)


def _create_embed_backend(spec: EmbeddingRunSpec, specter2_batch_size: int) -> EmbeddingBackend:
    return create_embedding_backend(
        embedding_model=spec.embedding_model,
        provider=spec.provider,
        region=(
            os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or AwsRegion.AP_NORTHEAST_1.value
        ),
        specter2_batch_size=specter2_batch_size,
    )


def embed_records(
    *,
    records: list[SampleRecord],
    spec: EmbeddingRunSpec,
    embed_max_concurrency: int,
    backend: EmbeddingBackend,
) -> np.ndarray:

    writer = EmbeddingArtifactWriter(
        embeddings_path=spec.output_dir / "embeddings.npy",
        responses_path=spec.output_dir / "embedding_responses.jsonl",
        records=records,
        embedding_model=spec.embedding_model,
    )
    try:
        reused_indices = copy_cached_embedding_artifacts_to_writer(
            output_dir=spec.output_dir,
            embedding_model=spec.embedding_model,
            records=records,
            writer=writer,
        )
        # Add slots already populated by writer's partial-resume.
        reused_indices.update(writer.written_indices())
        missing_indices = [index for index in range(len(records)) if index not in reused_indices]
        missing_records = [records[index] for index in missing_indices]

        print(
            f"[embed-cache] model={spec.embedding_model} reused={len(reused_indices)} missing={len(missing_records)}",
            flush=True,
        )

        if missing_records:
            backend.embed_records_streaming(
                missing_records,
                writer=writer,
                result_indices=missing_indices,
                max_concurrency=embed_max_concurrency,
            )

        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def embed_field(
    *,
    texts: list[str],
    records: list[SampleRecord],
    backend: EmbeddingBackend,
    spec: EmbeddingRunSpec,
    component: str,
    embed_max_concurrency: int,
) -> np.ndarray:
    """Embed a single text component and save to embeddings_{component}.npy.

    Reuses partial results from a previous interrupted run if available.
    Skips entirely if the final output file already exists.
    """
    embeddings_filename = f"embeddings_{component}.npy"
    responses_filename = f"embedding_responses_{component}.jsonl"
    embeddings_path = spec.output_dir / embeddings_filename
    responses_path = spec.output_dir / responses_filename
    if embeddings_path.exists():
        existing = np.load(embeddings_path, mmap_mode="r")
        if existing.shape[0] == len(records):
            print(
                f"[embed-field] component={component} model={spec.embedding_model} cached, skipping",
                flush=True,
            )
            return existing
        # Shape mismatch between the cached array and the current records
        # list: samples.jsonl has grown (retry added rows) or shrunk
        # (dataset filtered).  Fall through to the writer + copy_cached
        # path below — it key-matches each cached row against the current
        # records, remaps surviving embeddings to their new positions, and
        # recomputes only the delta.  Announced explicitly so the rebuild
        # never looks like a silent cache-skip.
        print(
            f"[embed-field] component={component} model={spec.embedding_model} "
            f"cache shape {existing.shape} vs records={len(records)} — "
            f"rebuilding via copy_cached (key-based remap + delta compute)",
            flush=True,
        )
        del existing  # release the mmap before writer recreates the target file
    writer = EmbeddingArtifactWriter(
        embeddings_path=embeddings_path,
        responses_path=responses_path,
        records=records,
        embedding_model=spec.embedding_model,
    )
    try:
        reused_indices = copy_cached_embedding_artifacts_to_writer(
            output_dir=spec.output_dir,
            embedding_model=spec.embedding_model,
            records=records,
            writer=writer,
            embeddings_filename=embeddings_filename,
            responses_filename=responses_filename,
        )
        # Add slots already populated by writer's partial-resume.
        reused_indices.update(writer.written_indices())
        missing_indices = [i for i in range(len(texts)) if i not in reused_indices]
        missing_texts = [texts[i] for i in missing_indices]

        print(
            f"[embed-field] component={component} model={spec.embedding_model} reused={len(reused_indices)} missing={len(missing_texts)}",
            flush=True,
        )

        if missing_texts:
            backend.embed_texts_streaming(
                missing_texts,
                writer=writer,
                result_indices=missing_indices,
                max_concurrency=embed_max_concurrency,
            )
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def build_run_manifest(
    *,
    keyword_rows: list[dict[str, Optional[str]]],
    output_dir: Path,
    attempted_samples: int,
    failed_samples: int,
    args: argparse.Namespace,
    embed_max_concurrency: int,
    spec: EmbeddingRunSpec,
    all_specs: list[EmbeddingRunSpec],
    source_run_dir: Optional[Path],
    prompt_file: Path,
) -> dict[str, Any]:
    return {
        "keywords": [row["keyword"] for row in keyword_rows],
        "n_keywords": len(keyword_rows),
        "idea_model": args.idea_model,
        "prompt_style": args.prompt_style,
        "provider": args.provider,
        "efforts": args.efforts,
        "samples_per_effort": args.samples_per_effort,
        "max_concurrency": args.max_concurrency,
        "embed_max_concurrency": embed_max_concurrency,
        "embed_provider": spec.provider,
        "specter2_base_model_id": DEFAULT_SPECTER2_BASE_MODEL_ID,
        "specter2_batch_size": args.specter2_batch_size,
        "embedding_model_id": spec.embedding_model,
        "output_dir": str(output_dir),
        "attempted_samples": attempted_samples,
        "generated_samples": attempted_samples - failed_samples,
        "failed_samples": failed_samples,
        "success_rate": (
            ((attempted_samples - failed_samples) / attempted_samples)
            if attempted_samples
            else 0.0
        ),
        "generated_at": datetime.now().isoformat(),
        "benchmark_provenance": build_benchmark_provenance(args, prompt_file=prompt_file),
        "all_embed_models": [
            {
                "embedding_model_id": embed_spec.embedding_model,
                "embed_provider": embed_spec.provider,
                "output_dir": str(embed_spec.output_dir),
            }
            for embed_spec in all_specs
        ],
        "source_run_dir": str(source_run_dir) if source_run_dir is not None else None,
    }


def run_embedding_analysis(
    *,
    records: list[SampleRecord],
    keyword_rows: list[dict[str, Optional[str]]],
    base_output_dir: Path,
    attempted_samples: int,
    failed_samples: int,
    args: argparse.Namespace,
    embed_max_concurrency: int,
    spec: EmbeddingRunSpec,
    all_specs: list[EmbeddingRunSpec],
    prompt_file: Path,
) -> dict[str, Any]:
    print(
        f"[embed-run] model={spec.embedding_model} provider={spec.provider} output_dir={spec.output_dir}",
        flush=True,
    )
    records_for_run = clone_records_for_embedding(records, embedding_model=spec.embedding_model)
    materialize_generation_artifacts(
        source_dir=base_output_dir,
        target_dir=spec.output_dir,
        records=records_for_run,
    )
    write_sample_records(spec.output_dir / "samples.jsonl", records_for_run)

    backend = _create_embed_backend(spec, args.specter2_batch_size)

    embeddings = embed_records(
        records=records_for_run,
        spec=spec,
        embed_max_concurrency=embed_max_concurrency,
        backend=backend,
    )
    write_sample_records(spec.output_dir / "samples.jsonl", records_for_run)

    # Per-field embedding: derive field names from the schema class
    _, structured_output_schema = resolve_prompt_template(
        prompt_file=prompt_file, prompt_style=PromptStyleName(args.prompt_style),
    )
    for component in structured_output_schema.model_fields:
        texts = extract_record_texts(records_for_run, component)
        embed_field(
            texts=texts,
            records=records_for_run,
            backend=backend,
            spec=spec,
            component=component,
            embed_max_concurrency=embed_max_concurrency,
        )

    keyword_effort_rows = build_summary_by_keyword_effort_rows(records_for_run, embeddings)
    write_csv(spec.output_dir / "summary_by_keyword_effort.csv", keyword_effort_rows)

    effort_summary_rows = build_summary_by_effort_rows(keyword_effort_rows)
    write_csv(spec.output_dir / "summary.csv", effort_summary_rows)

    category_summary_rows = build_category_summary_rows(keyword_effort_rows)
    write_csv(spec.output_dir / "summary_by_category_effort.csv", category_summary_rows)

    centroid_header, centroid_rows = build_centroid_distance_rows(records_for_run, embeddings)
    write_matrix_csv(spec.output_dir / "centroid_distances.csv", centroid_header, centroid_rows)

    coords, explained = pca_2d(embeddings)
    save_pca_coordinates(path=spec.output_dir / "pca_coordinates.csv", records=records_for_run, coords=coords)
    save_svg_scatter(
        path=spec.output_dir / "pca_scatter.svg",
        records=records_for_run,
        coords=coords,
        explained_variance=explained,
        title=(
            keyword_rows[0]["keyword"]
            if len(keyword_rows) == 1
            else f"{len(keyword_rows)} keywords"
        ),
    )
    render_diversity_svg(
        keyword_effort_rows,
        spec.output_dir / "within_set_diversity_pairwise_fixed_0_1.svg",
        subtitle=(
            "Metric: mean_pairwise_cosine_distance"
            f" | model={spec.embedding_model} | y-axis fixed to [0, 1]"
        ),
    )
    render_category_diversity_svg(
        category_summary_rows,
        spec.output_dir / "category_diversity_pairwise_fixed_0_1.svg",
        subtitle=(
            "Metric: mean_pairwise_cosine_distance aggregated over keywords/category"
            f" | model={spec.embedding_model} | y-axis fixed to [0, 1]"
        ),
    )

    manifest = build_run_manifest(
        keyword_rows=keyword_rows,
        output_dir=spec.output_dir,
        attempted_samples=attempted_samples,
        failed_samples=failed_samples,
        args=args,
        embed_max_concurrency=embed_max_concurrency,
        spec=spec,
        all_specs=all_specs,
        source_run_dir=base_output_dir,
        prompt_file=prompt_file,
    )
    with open(spec.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def build_embedding_failure_payload(
    *,
    spec: EmbeddingRunSpec,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "embedding_model_id": spec.embedding_model,
        "embed_provider": spec.provider,
        "output_dir": str(spec.output_dir),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": datetime.now().isoformat(),
    }


def persist_embedding_failure(
    *,
    spec: EmbeddingRunSpec,
    payload: dict[str, Any],
) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    with open(spec.output_dir / "embedding_failure.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clear_embedding_failure(spec: EmbeddingRunSpec) -> None:
    failure_path = spec.output_dir / "embedding_failure.json"
    if failure_path.exists():
        failure_path.unlink()


def run_embedding_analysis_safe(
    *,
    records: list[SampleRecord],
    keyword_rows: list[dict[str, Optional[str]]],
    base_output_dir: Path,
    attempted_samples: int,
    failed_samples: int,
    args: argparse.Namespace,
    embed_max_concurrency: int,
    spec: EmbeddingRunSpec,
    all_specs: list[EmbeddingRunSpec],
    prompt_file: Path,
) -> dict[str, Any]:
    try:
        manifest = run_embedding_analysis(
            records=records,
            keyword_rows=keyword_rows,
            base_output_dir=base_output_dir,
            attempted_samples=attempted_samples,
            failed_samples=failed_samples,
            args=args,
            embed_max_concurrency=embed_max_concurrency,
            spec=spec,
            all_specs=all_specs,
            prompt_file=prompt_file,
        )
        clear_embedding_failure(spec)
        return {
            "status": "succeeded",
            "embedding_model_id": spec.embedding_model,
            "embed_provider": spec.provider,
            "output_dir": str(spec.output_dir),
            "manifest": manifest,
        }
    except Exception as exc:
        failure_payload = build_embedding_failure_payload(spec=spec, exc=exc)
        persist_embedding_failure(spec=spec, payload=failure_payload)
        print(
            f"[embed-run] failed model={spec.embedding_model} provider={spec.provider} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "status": "failed",
            "embedding_model_id": spec.embedding_model,
            "embed_provider": spec.provider,
            "output_dir": str(spec.output_dir),
            "failure": failure_payload,
        }


def run_embedding_analyses(
    *,
    records: list[SampleRecord],
    keyword_rows: list[dict[str, Optional[str]]],
    base_output_dir: Path,
    attempted_samples: int,
    failed_samples: int,
    args: argparse.Namespace,
    embed_max_concurrency: int,
    embed_specs: list[EmbeddingRunSpec],
    prompt_file: Path,
) -> list[dict[str, Any]]:
    results_by_model: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(embed_specs)) as executor:
        future_to_spec = {
            executor.submit(
                run_embedding_analysis_safe,
                records=records,
                keyword_rows=keyword_rows,
                base_output_dir=base_output_dir,
                attempted_samples=attempted_samples,
                failed_samples=failed_samples,
                args=args,
                embed_max_concurrency=embed_max_concurrency,
                spec=spec,
                all_specs=embed_specs,
                prompt_file=prompt_file,
            ): spec
            for spec in embed_specs
        }
        for future in concurrent.futures.as_completed(future_to_spec):
            spec = future_to_spec[future]
            result = future.result()
            results_by_model[spec.embedding_model] = result
            print(
                f"[embed-run] finished model={spec.embedding_model} status={result['status']}",
                flush=True,
            )

    return [results_by_model[spec.embedding_model] for spec in embed_specs]


def main() -> None:
    args = parse_args()
    validate_idea_model_and_provider(args.idea_model, args.provider)
    args.liveideabench_root = resolve_liveideabench_root(args.liveideabench_root)
    if args.keyword_file is not None:
        args.keyword_file = args.keyword_file.expanduser().resolve()
    if args.prompt_file is not None:
        args.prompt_file = args.prompt_file.expanduser().resolve()
    prompt_file = resolve_prompt_file(args)
    prompt_style = PromptStyleName(args.prompt_style)
    if args.resume_output_dir is not None and args.resume_generation_output_dir is not None:
        raise ValueError(
            "Specify at most one of --resume-output-dir and --resume-generation-output-dir."
        )
    embed_max_concurrency = args.embed_max_concurrency or args.max_concurrency

    if args.resume_generation_output_dir is not None:
        if not args.keyword and not args.keyword_file:
            raise ValueError(
                "--resume-generation-output-dir requires --keyword or --keyword-file so the full task space can be reconstructed."
            )
        output_dir = args.resume_generation_output_dir
        samples_jsonl = output_dir / "samples.jsonl"
        if not samples_jsonl.exists():
            raise FileNotFoundError(f"Missing samples.jsonl in resume directory: {output_dir}")
        keyword_rows = resolve_keyword_rows(args)
        attempted_samples = len(keyword_rows) * len(args.efforts) * args.samples_per_effort
        existing_records = load_sample_records(samples_jsonl)
        missing_tasks = infer_missing_generation_tasks(
            existing_records=existing_records,
            keyword_rows=keyword_rows,
            efforts=args.efforts,
            samples_per_effort=args.samples_per_effort,
        )
        print(
            f"[resume-generate] existing_successes={len(existing_records)} missing_tasks={len(missing_tasks)}",
            flush=True,
        )
        backfill_records = generate_samples(
            keyword_rows=keyword_rows,
            idea_model=args.idea_model,
            provider=args.provider,
            efforts=args.efforts,
            samples_per_effort=args.samples_per_effort,
            output_dir=output_dir,
            embedding_model=DEFAULT_EMBED_MODEL.value,
            max_concurrency=args.max_concurrency,
            prompt_file=prompt_file,
            prompt_style=prompt_style,
            tasks=missing_tasks,
            success_log_filename="backfill_successes.jsonl",
        )
        records = sort_sample_records(existing_records + backfill_records)
        failed_samples = attempted_samples - len(records)
    elif args.resume_output_dir is not None:
        output_dir = args.resume_output_dir
        samples_jsonl = output_dir / "samples.jsonl"
        if not samples_jsonl.exists():
            raise FileNotFoundError(f"Missing samples.jsonl in resume directory: {output_dir}")
        records = load_sample_records(samples_jsonl)
        if args.keyword or args.keyword_file:
            keyword_rows = resolve_keyword_rows(args)
            attempted_samples = len(keyword_rows) * len(args.efforts) * args.samples_per_effort
        else:
            keyword_rows = infer_keyword_rows_from_records(records)
            attempted_samples = infer_attempted_samples_from_records(records)
        failed_samples = attempted_samples - len(records)
    else:
        keyword_rows = resolve_keyword_rows(args)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_output_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path("results")
            / "effort_diversity"
            / (
                f"{timestamp}_{slugify(keyword_rows[0]['keyword'])}"
                if len(keyword_rows) == 1
                else f"{timestamp}_{len(keyword_rows)}keywords"
            )
        )
        output_dir = allocate_output_dir(requested_output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

        records = generate_samples(
            keyword_rows=keyword_rows,
            idea_model=args.idea_model,
            provider=args.provider,
            efforts=args.efforts,
            samples_per_effort=args.samples_per_effort,
            output_dir=output_dir,
            embedding_model=DEFAULT_EMBED_MODEL.value,
            max_concurrency=args.max_concurrency,
            prompt_file=prompt_file,
            prompt_style=prompt_style,
        )
        attempted_samples = len(keyword_rows) * len(args.efforts) * args.samples_per_effort
        failed_samples = attempted_samples - len(records)
    if not records:
        raise RuntimeError(
            f"No successful generations were produced. attempted={attempted_samples} failed={failed_samples}. "
            f"See {output_dir / 'errors.jsonl'} for API and validation failures."
        )

    if args.skip_embedding:
        payload = {
            "output_dir": str(output_dir),
            "attempted_samples": attempted_samples,
            "generated_samples": len(records),
            "failed_samples": failed_samples,
            "success_rate": len(records) / attempted_samples if attempted_samples else 0.0,
            "skip_embedding": True,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    embed_specs = build_embedding_run_specs(
        base_output_dir=output_dir,
        embed_models=args.embed_models,
    )
    results = run_embedding_analyses(
        records=records,
        keyword_rows=keyword_rows,
        base_output_dir=output_dir,
        attempted_samples=attempted_samples,
        failed_samples=failed_samples,
        args=args,
        embed_max_concurrency=embed_max_concurrency,
        embed_specs=embed_specs,
        prompt_file=prompt_file,
    )
    succeeded = [result["manifest"] for result in results if result["status"] == "succeeded"]
    failed = [result["failure"] for result in results if result["status"] == "failed"]

    if len(results) == 1 and succeeded:
        print(json.dumps(succeeded[0], ensure_ascii=False, indent=2))
        return

    summary_payload = {
        "base_output_dir": str(output_dir),
        "succeeded_embed_models": succeeded,
        "failed_embed_models": failed,
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    if failed:
        failed_model_ids = ", ".join(item["embedding_model_id"] for item in failed)
        raise RuntimeError(
            f"Embedding failed for {len(failed)}/{len(results)} model(s): {failed_model_ids}. "
            "Successful model outputs were preserved on disk."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        flush_langfuse()
