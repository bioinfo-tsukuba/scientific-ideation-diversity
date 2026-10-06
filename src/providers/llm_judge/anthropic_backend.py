"""Anthropic SDK backend for the LLM-judge pairwise protocol.

Inherits from :class:`StructuredJudgeBase` which owns the shared Langfuse
tracing, retry wrapping, and :class:`JudgeResponse` construction.  This
subclass only implements the Anthropic-specific ``_invoke`` (native JSON
output parse) plus two metadata hooks.

We use the **direct Anthropic API**, not AWS Bedrock Converse, because
the account is on Anthropic Tier 4 which gives much higher throughput
than the Bedrock path.

Structured output uses Anthropic's native **JSON output** mode via
``output_config.format = {"type": "json_schema", "schema": ...}``
(GA on Claude Haiku 4.5 and the other Claude 4.x family; see
https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
The response is a single text content block containing schema-conformant
JSON, enforced at decode time by constrained decoding.  This is strictly
cheaper than the forced-tool-use workaround:

  * no tool schema is prepended to the model's input context
    (~400 tok saved per call; tool schemas are too small to benefit
    from Anthropic's 1024-token minimum for prompt caching, so JSON
    mode is the only path to recovering that overhead),
  * no ``tool_use`` structural wrapping in the output (the raw text
    is just the JSON payload, so "pairwise answer" fits in 6-9
    tokens instead of 33),
  * strict schema enforcement is preserved via constrained decoding
    plus a pydantic ``model_validate_json`` on parse.

The OpenAI backend already uses Anthropic's equivalent (strict
``text.format=json_schema``) and the Gemini backend uses
``response_json_schema``; this change brings Anthropic onto the same
"native strict-schema" primitive so all three backends have the same
reporting shape.

Non-reasoning / non-thinking only.  Claude 4.x has thinking disabled
by default when the ``thinking`` parameter is NOT passed; we rely on
that default and guard against silent regression by rejecting any
``thinking`` content block in the response.
"""

from __future__ import annotations

from pathlib import Path

from anthropic import Anthropic

from src.model_registry import JudgeModelName, ProviderName
from src.providers.llm_judge.base import StructuredJudgeBase
from src.schemas.llm_judge import FluencyAnswer, PairwiseInvocation


class AnthropicJudgeBackend(StructuredJudgeBase):
    """LLM-judge backend using the official Anthropic Python SDK.

    Args:
        model: Judge model enum member (currently
            :attr:`JudgeModelName.CLAUDE_HAIKU_4_5`).
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the ideas being judged.  Recorded on every Langfuse trace
            so the idea text behind a judgment can be recovered via
            ``samples_jsonl_path`` + ``sample_i_index`` /
            ``sample_j_index``.

    Note:
        ``temperature=0`` and ``max_tokens=1024`` are hardcoded at the
        call site rather than exposed as parameters.  ``max_tokens``
        only needs to cover a single JSON payload (``{"answer": "X"}``
        is a handful of tokens), so 1024 is comfortably sufficient and
        keeps cost deterministic.

        ``temperature=0`` is safe here because Claude 4.x has thinking
        disabled by default when the ``thinking`` parameter is not
        passed, and we additionally reject any ``thinking`` content
        block in ``_invoke`` to catch silent SDK-default regressions.
        Anthropic's extended-thinking docs require ``temperature`` to
        be unset when ``thinking`` is enabled; a future reasoning-capable
        Claude judge should therefore land in a separate backend so the
        non-reasoning path stays free of branches.  See
        https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
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
        return "anthropic_judge_fluency_critic"

    def _invoke(self, prompt_text: str) -> PairwiseInvocation:
        response = self._client.messages.create(
            model=self._model.value,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt_text}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": FluencyAnswer.model_json_schema(),
                }
            },
        )

        # JSON output mode guarantees exactly one ``text`` content block
        # whose body is schema-conformant JSON; any other shape is an
        # SDK contract violation and surfaces as a destructuring
        # ValueError rather than being silently recovered.
        [text_block] = [b for b in response.content if b.type == "text"]
        # Guard against silent SDK default-behaviour change: if thinking
        # started being emitted by default on Claude 4.x, fail loudly.
        if any(b.type == "thinking" for b in response.content):
            raise RuntimeError(
                f"Anthropic judge emitted a thinking block despite "
                f"non-thinking mode for model={self._model.value!r}"
            )

        parsed = FluencyAnswer.model_validate_json(text_block.text)

        # Anthropic returns input/output tokens separately; sum
        # ourselves because the SDK has no ``total_tokens`` field.
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return PairwiseInvocation(
            answer=parsed.answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            stop_reason=response.stop_reason or "",
            raw_api_response=response.model_dump(),
        )
