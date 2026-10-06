"""OpenAI Responses-API backend for the LiveIdeaBench quality-critic protocol.

Single-idea analogue of :mod:`src.providers.llm_judge.openai_backend`.
Inherits from :class:`StructuredQualityJudgeBase` which owns the shared
Langfuse tracing, retry, and :class:`QualityJudgeResponse` construction.
This subclass only implements the OpenAI-specific ``_invoke`` plus two
metadata hooks.

The call shape follows LiveIdeaBench's ``CriticLLM.critique_idea``
(``external/liveideabench/utils/LLM.py``) verbatim:

  * ``system`` message = ``critic_prompt.description``
    (from ``external/liveideabench/utils/prompts.json``, rendered via
    :func:`src.services.llm_judge_quality_service.render_system_prompt`)
  * ``user`` message = ``"Please evaluate the following scientific idea:\\n\\n" + idea``

Structured output is constrained by a strict JSON schema derived from
:class:`~src.schemas.llm_judge_quality.QualityScores`, mirroring the
pairwise backend.

Non-reasoning (GPT-4.1) only.  ``reasoning.*`` is intentionally not
forwarded; if a reasoning-capable judge is added later, introduce a
separate backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openai

from src.model_registry import JudgeModelName, ProviderName
from src.providers.llm_judge.base import StructuredQualityJudgeBase
from src.schemas.llm_judge_quality import QualityInvocation, QualityScores


class OpenAIQualityJudgeBackend(StructuredQualityJudgeBase):
    """Quality-critique backend using the OpenAI Responses API.

    Args:
        model: Judge model enum member (currently :attr:`JudgeModelName.GPT_4_1`).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced the
            idea being judged.  Recorded on every Langfuse trace so a
            score can be round-tripped back to the exact idea text via
            ``samples_jsonl_path`` + ``sample_row_index`` without
            leaking the idea text itself into Langfuse.

    Note:
        Temperature is hardcoded to ``0`` at the API call site rather than
        exposed as a parameter.  This keeps the judge deterministic and
        makes it obvious where the value is set.  Matches the pairwise
        backend.
    """

    def __init__(
        self,
        *,
        model: JudgeModelName,
        samples_jsonl_path: Path,
    ) -> None:
        super().__init__(model=model, samples_jsonl_path=samples_jsonl_path)
        self._client = openai.OpenAI()

    def _provider_tag(self) -> ProviderName:
        return ProviderName.OPENAI

    def _observation_name(self) -> str:
        return "openai_judge_critic_prompt"

    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> QualityInvocation:
        input_messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text},
        ]
        create_kwargs: dict[str, Any] = {
            "model": self._model.value,
            "input": input_messages,
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": QualityScores.__name__,
                    "schema": QualityScores.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response = self._client.responses.create(**create_kwargs)

        # Responses API with strict json_schema guarantees exactly one
        # ``message`` item with one ``output_text`` content on success;
        # any other shape is an SDK contract violation.
        [message] = [item for item in response.output if item.type == "message"]
        [text_content] = [c for c in message.content if c.type == "output_text"]

        parsed = QualityScores.model_validate_json(text_content.text)
        usage = response.usage

        return QualityInvocation(
            scores=parsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            stop_reason=message.status,
            raw_api_response=json.loads(response.model_dump_json()),
        )
