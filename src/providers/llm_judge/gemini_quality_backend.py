"""Vertex AI (``google-genai``) backend for the LiveIdeaBench quality-critic protocol.

Single-idea analogue of :mod:`src.providers.llm_judge.gemini_backend`.
Inherits from :class:`StructuredQualityJudgeBase` which owns the shared
Langfuse tracing, retry wrapping, and :class:`QualityJudgeResponse`
construction.  This subclass only implements the Vertex-specific
``_invoke`` plus two metadata hooks.

Call shape mirrors LiveIdeaBench's ``CriticLLM.critique_idea``:
  * ``system`` message is passed through Gemini's dedicated
    ``system_instruction`` slot on :class:`types.GenerateContentConfig`.
  * ``user`` message is the sole ``contents`` payload.

Structured output via ``response_json_schema`` derived from
:class:`~src.schemas.llm_judge_quality.QualityScores`.

Non-reasoning only: ``thinking_config=ThinkingConfig(thinking_budget=0)``
and ``temperature=0`` are hardcoded.
"""

from __future__ import annotations

from pathlib import Path

from google import genai
from google.genai import types

from src.model_registry import (
    GoogleAIStudioConfig,
    JudgeModelName,
    ProviderName,
    VertexAIConfig,
)
from src.providers.llm_judge.base import StructuredQualityJudgeBase
from src.providers.llm_judge.gemini_backend import _load_vertex_config_from_env
from src.schemas.llm_judge_quality import QualityInvocation, QualityScores


class GeminiQualityJudgeBackend(StructuredQualityJudgeBase):
    """Quality-critique backend using Vertex AI via the ``google-genai`` SDK.

    Args:
        model: Judge model enum member (currently
            :attr:`JudgeModelName.GEMINI_2_5_FLASH`).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the idea being judged.  Recorded on every Langfuse trace
            so a score can be round-tripped back to the exact idea text.
        config: Vertex AI connection config; defaults to
            :func:`_load_vertex_config_from_env` from the pairwise
            backend module so both judges share a single env-var contract.

    Note:
        Temperature ``0`` and ``thinking_budget=0`` are hardcoded at the
        API call site.  Matches the pairwise backend; see
        :class:`~src.providers.llm_judge.gemini_backend.GeminiJudgeBackend`
        for the rationale (temperature=0 is deterministic judging in
        non-thinking mode; Gemini 3 reasoning models would need a
        separate backend to honour the default-temperature=1.0 recommendation).
    """

    def __init__(
        self,
        *,
        model: JudgeModelName,
        samples_jsonl_path: Path,
        config: VertexAIConfig | GoogleAIStudioConfig | None = None,
    ) -> None:
        super().__init__(model=model, samples_jsonl_path=samples_jsonl_path)
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
                f"Unsupported config type for GeminiQualityJudgeBackend: "
                f"{type(self._config).__name__!r}"
            )

    def _provider_tag(self) -> ProviderName:
        return self._provider

    def _observation_name(self) -> str:
        return "gemini_judge_critic_prompt"

    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> QualityInvocation:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt_text,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_json_schema=QualityScores.model_json_schema(),
        )

        response = self._client.models.generate_content(
            model=self._model.value,
            contents=user_prompt_text,
            config=config,
        )

        if not response.text:
            raise RuntimeError(
                f"Gemini quality judge response has no text for "
                f"model={self._model.value!r}"
            )
        parsed = QualityScores.model_validate_json(response.text)

        usage = response.usage_metadata
        if usage is None:
            raise RuntimeError(
                f"Gemini quality judge response missing usage_metadata for "
                f"model={self._model.value!r}"
            )

        # Guard against silent regression of thinking-disabled mode.
        thoughts = getattr(usage, "thoughts_token_count", None)
        if thoughts is not None and thoughts > 0:
            raise RuntimeError(
                f"Gemini quality judge emitted thoughts_token_count={thoughts} "
                f"despite thinking_budget=0 for model={self._model.value!r}"
            )

        stop_reason: str = ""
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr is not None:
                stop_reason = str(fr)

        return QualityInvocation(
            scores=parsed,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
            stop_reason=stop_reason,
            raw_api_response=response.to_json_dict(),
        )
