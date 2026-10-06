"""Reasoning-capable Vertex Gemini quality-critic backend.

Single-idea analogue of the non-reasoning judges
(:class:`~src.providers.llm_judge.gemini_quality_backend.GeminiQualityJudgeBackend`):
calls the ``google-genai`` SDK directly in :meth:`_invoke` rather than
going through :class:`~src.providers.llm.gemini_backend.GeminiGenerationBackend`.
Mirroring the non-reasoning judges' pattern keeps the Langfuse trace
hierarchy clean — :class:`~src.providers.llm_judge.base.StructuredQualityJudgeBase`
emits a single ``gemini_reasoning_judge_critic_prompt`` observation; no
sibling ``gemini_generate_content`` span from a wrapped generation
backend.

Call shape mirrors LiveIdeaBench's ``CriticLLM.critique_idea``:
  * ``system`` message is passed through Gemini's dedicated
    ``system_instruction`` slot on :class:`types.GenerateContentConfig`.
  * ``user`` message is the sole ``contents`` payload.

Structured output via ``response_json_schema`` derived from
:class:`~src.schemas.llm_judge_quality.QualityScores`; thinking enabled
via ``ThinkingConfig(thinking_level=..., include_thoughts=True)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from src.model_registry import (
    SSOT_IDEA_MODEL_FOR_JUDGE,
    EffortName,
    GoogleAIStudioConfig,
    JudgeModelName,
    ProviderName,
    VertexAIConfig,
)
from src.providers.llm_judge.base import StructuredQualityJudgeBase
from src.providers.llm_judge.gemini_backend import _load_vertex_config_from_env
from src.providers.retry import call_with_retry
from src.schemas.llm_judge_quality import (
    QualityScores,
    ReasoningQualityInvocation,
)


class GeminiReasoningQualityJudgeBackend(StructuredQualityJudgeBase):
    """Quality-critique backend using Vertex AI with thinking enabled.

    Args:
        model: Reasoning judge identifier; narrowed to a single
            :class:`JudgeModelName` member (currently
            :attr:`JudgeModelName.GEMINI_3_1_PRO`).  Expand the type
            hint when another Vertex reasoning SKU is added.  Translated
            to the corresponding :class:`IdeaModelName` via
            :data:`SSOT_IDEA_MODEL_FOR_JUDGE` to pick the API model id.
        effort: Reasoning effort tier (LOW / MEDIUM / HIGH) passed
            through to ``ThinkingConfig.thinking_level``.  Typical:
            ``EffortName.HIGH`` (Gemini 3 Pro default).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the idea being judged.  Recorded on every Langfuse trace.
        config: Vertex AI / Google AI Studio connection config; defaults
            to :func:`_load_vertex_config_from_env` so both judges share
            a single env-var contract.
    """

    def __init__(
        self,
        *,
        model: JudgeModelName,
        effort: EffortName,
        samples_jsonl_path: Path,
        config: VertexAIConfig | GoogleAIStudioConfig | None = None,
    ) -> None:
        super().__init__(model=model, samples_jsonl_path=samples_jsonl_path)
        self._effort = effort
        self._api_model_id = SSOT_IDEA_MODEL_FOR_JUDGE[model].value
        self._config = config or _load_vertex_config_from_env()
        if isinstance(self._config, GoogleAIStudioConfig):
            self._client = genai.Client()
            self._provider = ProviderName.GOOGLE_AI_STUDIO
        elif isinstance(self._config, VertexAIConfig):
            self._client = genai.Client(
                vertexai=True,
                project=self._config.project,
                location=self._config.location,
            )
            self._provider = ProviderName.VERTEX
        else:
            raise TypeError(
                f"Unsupported config type for GeminiReasoningQualityJudgeBackend: "
                f"{type(self._config).__name__!r}"
            )

    def _provider_tag(self) -> ProviderName:
        return self._provider

    def _observation_name(self) -> str:
        return "gemini_reasoning_judge_critic_prompt"

    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> ReasoningQualityInvocation:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt_text,
            thinking_config=types.ThinkingConfig(
                thinking_level=self._effort.value,
                include_thoughts=True,
            ),
            response_mime_type="application/json",
            response_json_schema=QualityScores.model_json_schema(),
        )

        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self._api_model_id,
                contents=user_prompt_text,
                config=config,
            )
        )

        if not response.text:
            raise RuntimeError(
                f"Gemini reasoning judge response has no text for "
                f"model={self._api_model_id!r}"
            )
        text = response.text
        scores = QualityScores.model_validate_json(text)

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
                f"Gemini reasoning judge response missing usage_metadata for "
                f"model={self._api_model_id!r}"
            )

        stop_reason: str = ""
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr is not None:
                stop_reason = str(fr)

        # Vertex Gemini exposes thinking-token count as
        # ``usage_metadata.thoughts_token_count`` (paper §B.1).
        reasoning_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)

        return ReasoningQualityInvocation(
            scores=scores,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
            stop_reason=stop_reason,
            raw_api_response=response.to_json_dict(),
            response_reasoning=response_reasoning,
            has_reasoning_block=has_reasoning_block,
            reasoning_tokens=reasoning_tokens,
        )
