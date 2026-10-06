"""Gemini backend via Vertex AI or Google AI Studio (google-genai SDK)."""

from __future__ import annotations

from typing import Optional

from google import genai
from google.genai import types

from src.model_registry import (
    IDEA_MODEL_SUPPORTED_EFFORTS,
    EffortName,
    GoogleAIStudioConfig,
    IdeaModelName,
    VertexAIConfig,
)
from src.providers.retry import call_with_retry
from src.providers.tracing import get_langfuse
from src.schemas.generation import StructuredOutput
from src.schemas.llm_judge_quality import QualityScores

from .base import GenerationBackend, GenerationResult


class GeminiGenerationBackend(GenerationBackend):
    """Generation backend for Gemini via the ``google-genai`` SDK.

    Accepts either :class:`~src.model_registry.VertexAIConfig` (auth via
    Application Default Credentials; throughput bounded by Vertex AI's
    Dynamic Shared Quota) or :class:`~src.model_registry.GoogleAIStudioConfig`
    (auth via ``GEMINI_API_KEY``; throughput bounded by AI Studio's published
    tier limits, e.g. Tier-2 gives 1k RPM / 7M TPM / 50k RPD).  All other
    behavior — structured output schema, thinking level mapped from effort,
    retry policy, logprobs extraction — is identical across the two modes.

    Only ``thinking_level`` (mapped from effort) is passed as an
    experimental variable.
    """

    def __init__(
        self,
        idea_model: IdeaModelName,
        config: VertexAIConfig | GoogleAIStudioConfig,
    ) -> None:
        self.idea_model = idea_model
        if isinstance(config, GoogleAIStudioConfig):
            # The SDK picks up GEMINI_API_KEY from the environment when
            # ``vertexai`` is not set; we deliberately do NOT pass the key
            # through config so it never leaks into logs / manifests.
            self._client = genai.Client()
            self._provider_name = "google_ai_studio"
        elif isinstance(config, VertexAIConfig):
            self._client = genai.Client(
                vertexai=True,
                project=config.project,
                location=config.location,
            )
            self._provider_name = "vertex_ai"
        else:
            # Defensive: a new config type added to the union must also
            # land an ``elif`` here, otherwise we want to fail loudly at
            # construction rather than silently defaulting to Vertex.
            raise TypeError(
                f"Unsupported config type for GeminiGenerationBackend: "
                f"{type(config).__name__!r}"
            )
        self._langfuse = get_langfuse()

    def complete(
        self,
        prompt: str,
        *,
        effort: EffortName,
        structured_output_schema: type[StructuredOutput | QualityScores],
        system: str | None = None,
    ) -> GenerationResult:
        supported = IDEA_MODEL_SUPPORTED_EFFORTS[self.idea_model]
        if effort not in supported:
            raise ValueError(
                f"Model {self.idea_model.value!r} does not support effort={effort.value!r}. "
                f"Supported: {sorted(e.value for e in supported)}"
            )

        # Gemini's response_json_schema requires a dict, not a Pydantic class.
        schema_dict = structured_output_schema.model_json_schema()

        # ``system_instruction=None`` is accepted by the SDK as "no system
        # message" (verified) so the constructor stays a single literal
        # call; the reasoning judge passes the LiveIdeaBench critic_prompt
        # here (paper §3.4), generation passes ``None``.
        config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(
                thinking_level=effort.value,
                include_thoughts=True,
            ),
            response_mime_type="application/json",
            response_json_schema=schema_dict,
        )

        # Branch on call shape explicitly for the Langfuse trace input:
        # generation = single user message; reasoning judge = system + user.
        observation_input: list[dict[str, str]]
        if system is None:
            observation_input = [
                {"role": "user", "content": prompt},
            ]
        else:
            observation_input = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

        generation = self._langfuse.start_observation(
            name="gemini_generate_content",
            as_type="generation",
            model=self.idea_model.value,
            input=observation_input,
            metadata={"effort": effort.value, "provider": self._provider_name},
        )

        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self.idea_model.value,
                contents=prompt,
                config=config,
            )
        )

        if not response.text:
            raise RuntimeError(
                f"Gemini response has no text for model={self.idea_model.value!r}"
            )
        text: str = response.text

        # Extract thinking content if present.
        response_reasoning: Optional[str] = None
        has_reasoning_block = False
        if response.candidates:
            thinking_parts = [
                part.text
                for part in (response.candidates[0].content.parts or [])
                if getattr(part, "thought", None)
            ]
            if thinking_parts:
                has_reasoning_block = True
                response_reasoning = "\n".join(thinking_parts).strip() or None

        usage = response.usage_metadata
        if usage is None:
            raise RuntimeError(
                f"Gemini response missing usage_metadata for model={self.idea_model.value!r}"
            )

        # Gemini finish_reason
        stop_reason: Optional[str] = None
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr is not None:
                stop_reason = str(fr)

        structured_output = structured_output_schema.model_validate_json(text)

        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count

        generation.update(
            output=text,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
            },
            metadata={
                "effort": effort.value,
                "provider": self._provider_name,
                "stop_reason": stop_reason,
                "has_reasoning_block": has_reasoning_block,
            },
        )
        generation.end()

        return GenerationResult(
            text=text,
            structured_output=structured_output,
            response_reasoning=response_reasoning,
            has_reasoning_block=has_reasoning_block,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_api_response=response.to_json_dict(),
        )
