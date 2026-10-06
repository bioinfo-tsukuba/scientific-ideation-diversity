"""Reasoning-capable OpenAI quality-critic backend.

Single-idea analogue of the non-reasoning judges
(:class:`~src.providers.llm_judge.openai_quality_backend.OpenAIQualityJudgeBackend`
et al.): calls the OpenAI Responses API directly in :meth:`_invoke` rather
than going through :class:`~src.providers.llm.openai_backend.OpenAIGenerationBackend`.
Mirroring the non-reasoning judges' pattern keeps the Langfuse trace
hierarchy clean — :class:`~src.providers.llm_judge.base.StructuredQualityJudgeBase`
emits a single ``openai_reasoning_judge_critic_prompt`` observation; no
sibling ``openai_responses`` span from a wrapped generation backend.

Call shape mirrors LiveIdeaBench's ``CriticLLM.critique_idea``:
  * ``system`` message is passed as a ``role=system`` entry in the
    Responses-API ``input`` array.
  * ``user`` message is the sole user entry.

Structured output via the Responses API ``text.format`` with
``type="json_schema"`` and ``strict=True``, schema derived from
:class:`~src.schemas.llm_judge_quality.QualityScores`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import openai

from src.model_registry import (
    SSOT_IDEA_MODEL_FOR_JUDGE,
    EffortName,
    JudgeModelName,
    ProviderName,
)
from src.providers.llm_judge.base import StructuredQualityJudgeBase
from src.providers.retry import call_with_retry
from src.schemas.llm_judge_quality import (
    QualityScores,
    ReasoningQualityInvocation,
)


class OpenAIReasoningQualityJudgeBackend(StructuredQualityJudgeBase):
    """Quality-critique backend using the OpenAI Responses API with reasoning.

    Args:
        model: Reasoning judge identifier; narrowed to
            :attr:`JudgeModelName.GPT_5_4` (the only OpenAI reasoning
            judge currently registered).  Expand the ``Literal`` when
            another OpenAI reasoning SKU is added.  Internally
            translated to the corresponding :class:`IdeaModelName` via
            :data:`SSOT_IDEA_MODEL_FOR_JUDGE` to pick the API model id.
        effort: Reasoning effort tier passed to the Responses API
            ``reasoning.effort`` field.  Typical: ``EffortName.MEDIUM``
            (OpenAI default for GPT-5.4).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the idea being judged.  Recorded on every Langfuse trace.
    """

    def __init__(
        self,
        *,
        model: Literal[JudgeModelName.GPT_5_4],
        effort: EffortName,
        samples_jsonl_path: Path,
    ) -> None:
        super().__init__(model=model, samples_jsonl_path=samples_jsonl_path)
        self._effort = effort
        self._api_model_id = SSOT_IDEA_MODEL_FOR_JUDGE[model].value
        self._client = openai.OpenAI()

    def _provider_tag(self) -> ProviderName:
        return ProviderName.OPENAI

    def _observation_name(self) -> str:
        return "openai_reasoning_judge_critic_prompt"

    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> ReasoningQualityInvocation:
        schema_dict = QualityScores.model_json_schema()
        input_messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text},
        ]

        response = call_with_retry(
            lambda: self._client.responses.create(
                model=self._api_model_id,
                input=input_messages,
                reasoning={"effort": self._effort.value, "summary": "auto"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": QualityScores.__name__,
                        "schema": schema_dict,
                        "strict": True,
                    }
                },
            )
        )

        # Walk output items: a reasoning summary (optional) followed by
        # the JSON message body.  Mirrors OpenAIGenerationBackend.
        response_reasoning: str | None = None
        text: str | None = None
        stop_reason: str | None = None
        for item in response.output:
            if item.type == "reasoning" and item.summary:
                response_reasoning = "\n".join(
                    entry.text
                    for entry in item.summary
                    if hasattr(entry, "text") and entry.text
                )
            elif item.type == "message":
                stop_reason = item.status
                for content in item.content:
                    if content.type == "output_text":
                        text = content.text

        if not text:
            raise RuntimeError(
                f"OpenAI reasoning judge response has no text content for "
                f"model={self._api_model_id!r}"
            )

        scores = QualityScores.model_validate_json(text)
        has_reasoning_block = response_reasoning is not None

        usage = response.usage
        if usage is None:
            raise RuntimeError(
                f"OpenAI reasoning judge response missing usage for "
                f"model={self._api_model_id!r}"
            )

        # OpenAI Responses exposes the reasoning subset of output_tokens
        # under ``output_tokens_details.reasoning_tokens`` (paper §B.1).
        # Read from the dumped raw response to keep the same access path
        # the analysis scripts use.
        raw_api_response = json.loads(response.model_dump_json())
        details = (
            raw_api_response.get("usage", {}).get("output_tokens_details") or {}
        )
        reasoning_tokens = int(details.get("reasoning_tokens", 0))

        return ReasoningQualityInvocation(
            scores=scores,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            stop_reason=stop_reason or "",
            raw_api_response=raw_api_response,
            response_reasoning=response_reasoning,
            has_reasoning_block=has_reasoning_block,
            reasoning_tokens=reasoning_tokens,
        )
