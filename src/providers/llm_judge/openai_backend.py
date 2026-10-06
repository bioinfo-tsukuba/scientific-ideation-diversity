"""OpenAI Responses-API backend for the LLM-judge pairwise protocol.

Inherits from :class:`StructuredJudgeBase` which owns the shared Langfuse
tracing, retry, and :class:`JudgeResponse` construction.  This subclass
only implements the OpenAI-specific ``_invoke`` (client call + strict
JSON-schema parse) plus the two tiny metadata hooks.

Non-reasoning (GPT-4.1) only.  ``reasoning.*`` is intentionally not
forwarded; if a reasoning-capable OpenAI judge is added later,
introduce a separate backend so the non-reasoning path stays free of
branches that could mask "no reasoning trace because the model never
produced one" versus "no reasoning trace because the model has no
reasoning block".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openai

from src.model_registry import JudgeModelName, ProviderName
from src.providers.llm_judge.base import StructuredJudgeBase
from src.schemas.llm_judge import FluencyAnswer, PairwiseInvocation


class OpenAIJudgeBackend(StructuredJudgeBase):
    """LLM-judge backend using the OpenAI Responses API.

    Args:
        model: Judge model enum member (currently :attr:`JudgeModelName.GPT_4_1`).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced the
            ideas being judged.  Recorded on every Langfuse trace so the
            idea text behind a judgment can be recovered via
            ``samples_jsonl_path`` + ``sample_i_index`` / ``sample_j_index``.

    Note:
        Temperature is hardcoded to ``0`` at the API call site rather than
        exposed as a parameter.  This keeps the judge deterministic and
        makes it obvious where the value is set.
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
        return "openai_judge_fluency_critic"

    def _invoke(self, prompt_text: str) -> PairwiseInvocation:
        create_kwargs: dict[str, Any] = {
            "model": self._model.value,
            "input": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": FluencyAnswer.__name__,
                    "schema": FluencyAnswer.model_json_schema(),
                    "strict": True,
                }
            },
        }
        response = self._client.responses.create(**create_kwargs)

        # Responses API with strict json_schema guarantees exactly one
        # ``message`` item with one ``output_text`` content on success; any
        # other shape is an SDK contract violation and surfaces as a
        # destructuring ValueError rather than being silently recovered.
        [message] = [item for item in response.output if item.type == "message"]
        [text_content] = [c for c in message.content if c.type == "output_text"]

        parsed = FluencyAnswer.model_validate_json(text_content.text)
        usage = response.usage

        return PairwiseInvocation(
            answer=parsed.answer,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            stop_reason=message.status,
            raw_api_response=json.loads(response.model_dump_json()),
        )
