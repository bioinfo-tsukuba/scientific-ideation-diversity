"""OpenAI official SDK backend for GPT models (Responses API)."""

from __future__ import annotations

import json

import openai

from src.model_registry import IDEA_MODEL_SUPPORTED_EFFORTS, EffortName, IdeaModelName
from src.providers.retry import call_with_retry
from src.providers.tracing import get_langfuse
from src.schemas.generation import StructuredOutput
from src.schemas.llm_judge_quality import QualityScores

from .base import GenerationBackend, GenerationResult


class OpenAIGenerationBackend(GenerationBackend):
    """Generation backend using the OpenAI Responses API.

    Uses ``reasoning.effort`` and ``reasoning.summary`` to control and
    capture the model's chain-of-thought reasoning.
    """

    def __init__(self, idea_model: IdeaModelName) -> None:
        self.idea_model = idea_model
        self._client = openai.OpenAI()
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

        schema_dict = structured_output_schema.model_json_schema()

        # The two callers of this method use different message shapes:
        #   * Generation (system=None): single user message — the idea-
        #     generation prompt itself.  Original v0 behavior.
        #   * Reasoning judge (system≠None): system + user — LiveIdeaBench
        #     critic_prompt as system, idea body as user (paper §3.4).
        # Branching explicitly makes the shape difference visible at the
        # call site instead of relying on append-when-truthy.
        input_messages: list[dict[str, str]]
        if system is None:
            input_messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            input_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

        generation = self._langfuse.start_observation(
            name="openai_responses",
            as_type="generation",
            model=self.idea_model.value,
            input=input_messages,
            metadata={"effort": effort.value, "provider": "openai"},
        )

        response = call_with_retry(
            lambda: self._client.responses.create(
                model=self.idea_model.value,
                input=input_messages,
                reasoning={"effort": effort.value, "summary": "auto"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": structured_output_schema.__name__,
                        "schema": schema_dict,
                        "strict": True,
                    }
                },
            )
        )

        # Extract reasoning summary and text from output items.
        response_reasoning = None
        text = None
        stop_reason = None

        for item in response.output:
            if item.type == "reasoning" and item.summary:
                response_reasoning = "\n".join(
                    entry.text for entry in item.summary if hasattr(entry, "text") and entry.text
                )
            elif item.type == "message":
                stop_reason = item.status
                for content in item.content:
                    if content.type == "output_text":
                        text = content.text

        if not text:
            raise RuntimeError(
                f"OpenAI response has no text content for model={self.idea_model.value!r}"
            )

        structured_output = structured_output_schema.model_validate_json(text)
        has_reasoning_block = response_reasoning is not None

        usage = response.usage
        if usage is None:
            raise RuntimeError(
                f"OpenAI response missing usage for model={self.idea_model.value!r}"
            )
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        generation.update(
            output=text,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": usage.total_tokens,
            },
            metadata={
                "effort": effort.value,
                "provider": "openai",
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
            total_tokens=usage.total_tokens,
            raw_api_response=json.loads(response.model_dump_json()),
        )
