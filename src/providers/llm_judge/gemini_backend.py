"""Vertex AI (``google-genai``) backend for the LLM-judge pairwise protocol.

Inherits from :class:`StructuredJudgeBase` which owns the shared Langfuse
tracing, retry wrapping, and :class:`JudgeResponse` construction.  This
subclass only implements the Vertex-specific ``_invoke`` (client call +
``response_json_schema`` parse) plus two metadata hooks.

Non-reasoning only.  ``thinking_config=ThinkingConfig(thinking_budget=0)``
is passed explicitly so the response carries no chain-of-thought; if a
reasoning-capable Gemini judge is added later, introduce a separate
backend so the non-reasoning path stays free of branches that could
mask "no thinking content because the model produced none" versus
"thinking was disabled".

We deliberately pin model id ``gemini-2.5-flash`` (GA) rather than a
``-preview`` SKU to avoid Dynamic Shared Quota throttling; see the
:class:`~src.model_registry.JudgeModelName` docstring.
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
from src.providers.llm_judge.base import StructuredJudgeBase
from src.schemas.llm_judge import FluencyAnswer, PairwiseInvocation


class GeminiJudgeBackend(StructuredJudgeBase):
    """LLM-judge backend using Vertex AI via the ``google-genai`` SDK.

    Args:
        model: Judge model enum member (currently
            :attr:`JudgeModelName.GEMINI_2_5_FLASH`).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the ideas being judged.  Recorded on every Langfuse trace
            so the idea text behind a judgment can be recovered via
            ``samples_jsonl_path`` + ``sample_i_index`` / ``sample_j_index``.
        config: Vertex AI connection config (project + ``location="global"``).
            Defaults to values read from ``VERTEX_AI_PROJECT`` /
            ``VERTEX_AI_LOCATION`` environment variables so callers that
            construct the backend via the simple ``__init__(*, model,
            samples_jsonl_path)`` signature do not need to know about
            Vertex wiring.

    Note:
        Temperature is hardcoded to ``0`` at the API call site and
        ``thinking_budget=0`` forces non-reasoning behaviour; neither is
        exposed as a parameter.  This keeps the judge deterministic and
        makes it obvious where the values are set.

        ``temperature=0`` is safe here because we hold the model to
        non-thinking mode (``thinking_budget=0``).  The "leave temperature
        at the default 1.0" advice in the Gemini 3 prompting guide
        (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide)
        applies to Gemini 3 *reasoning* models, where lowering temperature
        can trigger looping or degrade reasoning behaviour.  A future
        reasoning-capable Gemini judge should land in a separate backend
        (per the module-level docstring) and would revisit this setting.
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
            # SDK reads GEMINI_API_KEY from env; no project/location.
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
            # A new config type added to the union must also land a
            # matching ``elif`` here; fail loudly rather than silently
            # defaulting to Vertex.
            raise TypeError(
                f"Unsupported config type for GeminiJudgeBackend: "
                f"{type(self._config).__name__!r}"
            )

    def _provider_tag(self) -> ProviderName:
        return self._provider

    def _observation_name(self) -> str:
        return "gemini_judge_fluency_critic"

    def _invoke(self, prompt_text: str) -> PairwiseInvocation:
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_json_schema=FluencyAnswer.model_json_schema(),
        )

        response = self._client.models.generate_content(
            model=self._model.value,
            contents=prompt_text,
            config=config,
        )

        if not response.text:
            raise RuntimeError(
                f"Gemini judge response has no text for model={self._model.value!r}"
            )
        parsed = FluencyAnswer.model_validate_json(response.text)

        usage = response.usage_metadata
        if usage is None:
            raise RuntimeError(
                f"Gemini judge response missing usage_metadata for "
                f"model={self._model.value!r}"
            )

        # ``thoughts_token_count`` should be None/0 since thinking is
        # disabled; guard against a silent regression where the SDK
        # starts emitting a thinking block despite ``thinking_budget=0``.
        thoughts = getattr(usage, "thoughts_token_count", None)
        if thoughts is not None and thoughts > 0:
            raise RuntimeError(
                f"Gemini judge emitted thoughts_token_count={thoughts} "
                f"despite thinking_budget=0 for model={self._model.value!r}"
            )

        stop_reason: str = ""
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr is not None:
                stop_reason = str(fr)

        return PairwiseInvocation(
            answer=parsed.answer,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            total_tokens=usage.total_token_count,
            stop_reason=stop_reason,
            raw_api_response=response.to_json_dict(),
        )


def _load_vertex_config_from_env() -> VertexAIConfig:
    """Build a :class:`VertexAIConfig` from ``VERTEX_AI_*`` env vars.

    An empty / missing project id fails loudly at construction time
    (``VertexAIConfig`` only requires the field to be a ``str`` but
    ``genai.Client`` rejects an empty project).
    """
    import os

    project = os.getenv("VERTEX_AI_PROJECT", "")
    location = os.getenv("VERTEX_AI_LOCATION", "global")
    if not project:
        raise RuntimeError(
            "VERTEX_AI_PROJECT is unset; required to construct GeminiJudgeBackend"
        )
    return VertexAIConfig(project=project, location=location)
