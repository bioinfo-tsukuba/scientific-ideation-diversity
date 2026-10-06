"""Generation service: idea generation via LLM backends."""

from __future__ import annotations

import json
from typing import Any

from src.model_registry import (
    BedrockConfig,
    EffortName,
    GenerationProviderName,
    GoogleAIStudioConfig,
    IdeaModelName,
    VertexAIConfig,
)
from src.providers.llm.base import GenerationBackend
from src.providers.llm.bedrock_converse import BedrockConverseBackend
from src.providers.llm.gemini_backend import GeminiGenerationBackend
from src.providers.llm.openai_backend import OpenAIGenerationBackend
from src.schemas.generation import (
    GenerationResult,
    IdeaGenerationServiceResult,
    StructuredOutput,
)


def _create_backend(
    idea_model: IdeaModelName,
    *,
    provider: GenerationProviderName,
    bedrock_config: BedrockConfig,
    vertex_ai_config: VertexAIConfig,
    google_ai_studio_config: GoogleAIStudioConfig,
) -> GenerationBackend:
    """Instantiate the right backend for *idea_model*.

    Each provider uses its official SDK directly:
    - Claude → Bedrock Converse API (boto3)
    - GPT → OpenAI SDK
    - Gemini → Vertex AI (google-genai SDK) OR Google AI Studio
      (google-genai SDK, API key auth); switched via ``provider``.
    """
    if idea_model == IdeaModelName.CLAUDE_SONNET_4_6:
        return BedrockConverseBackend(idea_model, config=bedrock_config)
    if idea_model == IdeaModelName.GPT_5_4:
        return OpenAIGenerationBackend(idea_model)
    if idea_model == IdeaModelName.GEMINI_3_1_PRO:
        if provider == GenerationProviderName.GOOGLE_AI_STUDIO:
            return GeminiGenerationBackend(idea_model, config=google_ai_studio_config)
        if provider == GenerationProviderName.VERTEX:
            return GeminiGenerationBackend(idea_model, config=vertex_ai_config)
        raise ValueError(
            f"Provider {provider.value!r} is not supported for model "
            f"{idea_model.value!r}; expected one of 'vertex' or 'google_ai_studio'."
        )
    raise ValueError(f"No backend registered for model {idea_model.value!r}")


def _result_to_call_record(result: GenerationResult, *, call_stage: str) -> dict[str, Any]:
    return {
        "call_stage": call_stage,
        "stop_reason": result.stop_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "has_reasoning_block": result.has_reasoning_block,
    }


def _serialise_response(result: GenerationResult) -> str:
    return json.dumps(result.raw_api_response, ensure_ascii=False, default=str)


def generate_idea(
    prompt: str,
    *,
    idea_model: IdeaModelName,
    provider: GenerationProviderName,
    effort: EffortName,
    structured_output_schema: type[StructuredOutput],
    bedrock_config: BedrockConfig,
    vertex_ai_config: VertexAIConfig,
    google_ai_studio_config: GoogleAIStudioConfig,
) -> IdeaGenerationServiceResult:
    """Generate a scientific idea.

    Each model uses its official SDK. Only ``effort`` (reasoning effort) is
    controlled as the experimental variable; other generation parameters
    use each provider's API default.

    For models that require ``max_tokens`` (e.g. Claude), the backend
    looks up the value from :data:`IDEA_MODEL_MAX_TOKENS` internally.

    Args:
        prompt: Idea generation prompt.
        idea_model: Which model to use.
        provider: Which provider backend to route the request through.
            For Gemini 3.1 Pro this selects Vertex vs Google AI Studio;
            for other models it must match the model's canonical provider.
        effort: Reasoning effort level.
        structured_output_schema: Pydantic model class for output constraint.
        bedrock_config: Frozen Bedrock configuration (region, timeouts).
        vertex_ai_config: Frozen Vertex AI configuration (project, location).
        google_ai_studio_config: Frozen Google AI Studio configuration
            (auth via ``GEMINI_API_KEY`` env var, read by the SDK).
    """
    backend = _create_backend(
        idea_model,
        provider=provider,
        bedrock_config=bedrock_config,
        vertex_ai_config=vertex_ai_config,
        google_ai_studio_config=google_ai_studio_config,
    )

    result = backend.complete(
        prompt,
        effort=effort,
        structured_output_schema=structured_output_schema,
    )

    return IdeaGenerationServiceResult(
        text=result.text,
        full_response=result.text,
        structured_output=result.structured_output,
        response_reasoning=result.response_reasoning,
        has_reasoning_block=result.has_reasoning_block,
        stop_reason=result.stop_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        raw_api_response=_serialise_response(result),
        api_calls=[_result_to_call_record(result, call_stage="generate")],
    )
