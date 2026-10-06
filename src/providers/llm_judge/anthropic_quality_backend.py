"""Anthropic SDK backend for the LiveIdeaBench quality-critic protocol.

Single-idea analogue of :mod:`src.providers.llm_judge.anthropic_backend`.
Inherits from :class:`StructuredQualityJudgeBase` which owns the shared
Langfuse tracing, retry wrapping, and :class:`QualityJudgeResponse`
construction.  This subclass only implements the Anthropic-specific
``_invoke`` plus two metadata hooks.

Call shape mirrors LiveIdeaBench's ``CriticLLM.critique_idea``:
  * ``system`` message is passed through Anthropic's top-level
    ``system`` parameter on ``messages.create``.
  * ``user`` message is the sole ``messages`` content.

Structured output uses Anthropic's native **JSON output** mode via
``output_config.format = {"type": "json_schema", "schema": ...}``
(GA on Claude Haiku 4.5; see the pairwise backend docstring for the
full rationale).  Strict schema enforcement is preserved via
constrained decoding plus a pydantic ``model_validate_json`` on parse.

Non-reasoning only — see pairwise backend for rationale.
"""

from __future__ import annotations

from pathlib import Path

from anthropic import Anthropic

from src.model_registry import JudgeModelName, ProviderName
from src.providers.llm_judge.base import StructuredQualityJudgeBase
from src.schemas.llm_judge_quality import QualityInvocation, QualityScores


class AnthropicQualityJudgeBackend(StructuredQualityJudgeBase):
    """Quality-critique backend using the official Anthropic Python SDK.

    Args:
        model: Judge model enum member (currently
            :attr:`JudgeModelName.CLAUDE_HAIKU_4_5`).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the idea being judged.

    Note:
        ``temperature=0`` and ``max_tokens=4096`` are hardcoded at the
        call site.  ``max_tokens`` is 4x the pairwise backend's 1024
        because the LiveIdeaBench ``critic_prompt`` asks for an
        ``analysis`` field whose length is unbounded; on the Sonnet 4.6
        / GPT-5.4 overnight run, ~0.17% of responses hit the old 1024
        cap and were truncated mid-string, producing unparseable JSON
        that surfaced as ``pydantic.ValidationError`` with no clean
        retry path.  4096 covers the observed long tail (truncations
        occurred around columns 3k-6.5k of the JSON payload, ~800-1.6k
        output tokens once you account for analysis prose plus the
        three scalar scores) with headroom.  Pairwise remains at 1024
        because its payload is a single-letter answer.

        ``temperature=0`` is safe in the non-thinking default; see the
        pairwise backend's ``Note`` for the rationale.  Reasoning-capable
        Claude judges require ``temperature`` unset and should land in
        a separate backend.
    """

    def __init__(
        self,
        *,
        model: JudgeModelName,
        samples_jsonl_path: Path,
    ) -> None:
        super().__init__(model=model, samples_jsonl_path=samples_jsonl_path)
        self._client = Anthropic()

    def _provider_tag(self) -> ProviderName:
        return ProviderName.ANTHROPIC

    def _observation_name(self) -> str:
        return "anthropic_judge_critic_prompt"

    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> QualityInvocation:
        response = self._client.messages.create(
            model=self._model.value,
            max_tokens=4096,
            temperature=0,
            system=system_prompt_text,
            messages=[{"role": "user", "content": user_prompt_text}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": QualityScores.model_json_schema(),
                }
            },
        )

        # JSON output mode guarantees exactly one ``text`` content block
        # whose body is schema-conformant JSON; any other shape is an
        # SDK contract violation and surfaces as a destructuring
        # ValueError rather than being silently recovered.
        [text_block] = [b for b in response.content if b.type == "text"]
        if any(b.type == "thinking" for b in response.content):
            raise RuntimeError(
                f"Anthropic quality judge emitted a thinking block despite "
                f"non-thinking mode for model={self._model.value!r}"
            )

        parsed = QualityScores.model_validate_json(text_block.text)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return QualityInvocation(
            scores=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            stop_reason=response.stop_reason or "",
            raw_api_response=response.model_dump(),
        )
