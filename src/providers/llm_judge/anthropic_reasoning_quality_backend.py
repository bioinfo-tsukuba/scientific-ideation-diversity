"""Reasoning-capable Bedrock Converse quality-critic backend for Claude.

Single-idea analogue of the non-reasoning judges
(:class:`~src.providers.llm_judge.anthropic_quality_backend.AnthropicQualityJudgeBackend`
calls the Anthropic SDK directly; this backend calls AWS Bedrock
Converse directly via boto3).  Mirroring the non-reasoning judges'
pattern keeps the Langfuse trace hierarchy clean —
:class:`~src.providers.llm_judge.base.StructuredQualityJudgeBase` emits
a single ``anthropic_reasoning_judge_critic_prompt`` observation; no
sibling ``bedrock_converse`` span from a wrapped generation backend.

We use Bedrock (not the Anthropic SDK) for Sonnet 4.6 with thinking
because the experiment runs through an AP-Northeast Inference Profile
and Bedrock's ``additionalModelRequestFields.output_config.effort``
exposes the effort tier in a way that survives the Converse contract.

Bedrock's ``usage`` does not break out a reasoning-token subset, so we
derive it post-hoc via Anthropic's native ``count_tokens`` endpoint
following paper §B.1 / ``tab:token-fields-app``; the helper lives in
:mod:`src.providers.llm_judge.anthropic_count_tokens`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import boto3
from botocore.config import Config as BotocoreConfig

from src.model_registry import (
    SSOT_IDEA_MODEL_FOR_JUDGE,
    AwsRegion,
    BedrockConfig,
    EffortName,
    JudgeModelName,
    ProviderName,
)

# Reasoning judge needs more max_tokens than idea generation: 32k thinking
# tokens + JSON analysis field + scores easily exceeds the 4096 idea-side
# default and triggers ``no text content blocks`` errors when reasoning
# consumes the whole token budget.  16384 leaves comfortable headroom for
# the visible answer after deep reasoning.
_JUDGE_MAX_TOKENS = 16384
from src.providers.llm_judge.anthropic_count_tokens import derive_reasoning_tokens
from src.providers.llm_judge.base import StructuredQualityJudgeBase
from src.providers.retry import call_with_retry
from src.schemas.llm_judge_quality import (
    QualityScores,
    ReasoningQualityInvocation,
)


class AnthropicReasoningQualityJudgeBackend(StructuredQualityJudgeBase):
    """Quality-critique backend for Claude Sonnet 4.6 via Bedrock with thinking.

    Args:
        model: Reasoning judge identifier; narrowed to
            :attr:`JudgeModelName.CLAUDE_SONNET_4_6` (the only Bedrock
            reasoning judge currently registered).  Expand the
            ``Literal`` when another Bedrock reasoning SKU is added.
            Translated to the corresponding :class:`IdeaModelName` via
            :data:`SSOT_IDEA_MODEL_FOR_JUDGE` to pick the API model id.
        effort: Reasoning effort tier mapped to Bedrock
            ``output_config.effort`` (LOW / MEDIUM / HIGH / MAX).  Uses
            HIGH as Anthropic's vendor default for the strong-judge
            ablation.
        samples_jsonl_path: Path to the ``samples.jsonl`` that sourced
            the idea being judged.
        bedrock_config: Bedrock connection config; defaults to
            ``BedrockConfig(region=AwsRegion.AP_NORTHEAST_1)`` (same
            default as the generation backend).
    """

    def __init__(
        self,
        *,
        model: Literal[JudgeModelName.CLAUDE_SONNET_4_6],
        effort: EffortName,
        samples_jsonl_path: Path,
        bedrock_config: BedrockConfig | None = None,
    ) -> None:
        super().__init__(model=model, samples_jsonl_path=samples_jsonl_path)
        self._effort = effort
        self._idea_model = SSOT_IDEA_MODEL_FOR_JUDGE[model]
        self._api_model_id = self._idea_model.value
        self._max_tokens = _JUDGE_MAX_TOKENS
        cfg = bedrock_config or BedrockConfig(region=AwsRegion.AP_NORTHEAST_1)
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=cfg.region.value,
            config=BotocoreConfig(
                connect_timeout=cfg.connect_timeout,
                read_timeout=cfg.read_timeout,
                max_pool_connections=cfg.max_pool_connections,
            ),
        )

    def _provider_tag(self) -> ProviderName:
        return ProviderName.BEDROCK

    def _observation_name(self) -> str:
        return "anthropic_reasoning_judge_critic_prompt"

    def _build_request(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> dict[str, Any]:
        schema_dict = QualityScores.model_json_schema()
        additional_fields: dict[str, Any] = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort.value},
        }
        return {
            "modelId": self._api_model_id,
            "messages": [{"role": "user", "content": [{"text": user_prompt_text}]}],
            "system": [{"text": system_prompt_text}],
            "inferenceConfig": {"maxTokens": self._max_tokens},
            "additionalModelRequestFields": additional_fields,
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(schema_dict, ensure_ascii=False),
                            "name": "structured_output",
                            "description": "Structured JSON output",
                        }
                    },
                }
            },
        }

    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> ReasoningQualityInvocation:
        request = self._build_request(
            system_prompt_text=system_prompt_text,
            user_prompt_text=user_prompt_text,
        )

        response = call_with_retry(lambda: self._client.converse(**request))

        stop_reason = response.get("stopReason") or ""
        usage = response.get("usage", {})
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        reasoning_texts: list[str] = []
        for block in content_blocks:
            rt = block.get("reasoningContent", {}).get("reasoningText", {}).get("text")
            if rt:
                reasoning_texts.append(rt)
        response_reasoning = "\n".join(reasoning_texts).strip() or None
        has_reasoning_block = any("reasoningContent" in b for b in content_blocks)

        texts = [b["text"] for b in content_blocks if "text" in b and b["text"]]
        if not texts:
            raise RuntimeError(
                f"Bedrock Converse response has no text content blocks for "
                f"model={self._api_model_id!r}"
            )
        text = "\n".join(texts).strip()
        scores = QualityScores.model_validate_json(text)

        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        total_tokens = usage["totalTokens"]

        # Bedrock's ``usage`` does not break out a reasoning subset; the
        # cross-vendor billable-output count (paper §B.1) is
        # ``output_tokens`` minus the visible-answer tokens counted via
        # Anthropic's count_tokens endpoint.  Keyed by IdeaModelName
        # (which looks up the Anthropic-native model id).
        reasoning_tokens = derive_reasoning_tokens(
            model=self._idea_model,
            visible_text=text,
            output_tokens=output_tokens,
        )

        raw_api_response = json.loads(
            json.dumps(response, ensure_ascii=False, default=str)
        )

        return ReasoningQualityInvocation(
            scores=scores,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            stop_reason=stop_reason,
            raw_api_response=raw_api_response,
            response_reasoning=response_reasoning,
            has_reasoning_block=has_reasoning_block,
            reasoning_tokens=reasoning_tokens,
        )
