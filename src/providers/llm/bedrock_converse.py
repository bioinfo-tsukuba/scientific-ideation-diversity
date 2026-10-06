"""Bedrock Converse API backend for Claude models.

This backend calls the Bedrock Converse API directly so that
``output_config.effort`` is preserved exactly as the experiment requires.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.config import Config as BotocoreConfig

from src.model_registry import IDEA_MODEL_MAX_TOKENS, BedrockConfig, EffortName, IdeaModelName
from src.providers.retry import call_with_retry
from src.providers.tracing import get_langfuse
from src.schemas.generation import StructuredOutput
from src.schemas.llm_judge_quality import QualityScores

from .base import GenerationBackend, GenerationResult

_clients: dict[BedrockConfig, BaseClient] = {}
_clients_lock = threading.Lock()


def _get_client(config: BedrockConfig) -> BaseClient:
    with _clients_lock:
        client = _clients.get(config)
        if client is None:
            client = boto3.client(
                "bedrock-runtime",
                region_name=config.region.value,
                config=BotocoreConfig(
                    connect_timeout=config.connect_timeout,
                    read_timeout=config.read_timeout,
                    max_pool_connections=config.max_pool_connections,
                ),
            )
            _clients[config] = client
    return client


class BedrockConverseBackend(GenerationBackend):
    """Generation backend that calls the AWS Bedrock Converse API directly."""

    def __init__(self, idea_model: IdeaModelName, config: BedrockConfig) -> None:
        self.idea_model = idea_model
        self.config = config
        self._langfuse = get_langfuse()

    def _build_request(
        self,
        prompt: str,
        *,
        effort: EffortName,
        structured_output_schema: type[StructuredOutput | QualityScores],
        max_tokens: int,
        system: str | None = None,
    ) -> dict[str, Any]:
        schema_dict = structured_output_schema.model_json_schema()

        # effort=none → disable thinking entirely.
        # Bedrock only accepts low/medium/high/max for output_config.effort.
        if effort == EffortName.NONE:
            additional_fields: dict[str, Any] = {
                "thinking": {"type": "disabled"},
            }
        else:
            additional_fields = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort.value},
            }

        request: dict[str, Any] = {
            "modelId": self.idea_model.value,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
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
        if system is not None:
            request["system"] = [{"text": system}]
        return request

    @staticmethod
    def _parse_response(
        response: dict[str, Any],
        structured_output_schema: type[StructuredOutput | QualityScores],
    ) -> GenerationResult:
        stop_reason = response.get("stopReason")
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
            raise RuntimeError("Bedrock Converse response has no text content blocks")
        text = "\n".join(texts).strip()

        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        total_tokens = usage["totalTokens"]

        structured_output = structured_output_schema.model_validate_json(text)

        raw_api_response = json.loads(
            json.dumps(response, ensure_ascii=False, default=str)
        )

        return GenerationResult(
            text=text,
            structured_output=structured_output,
            response_reasoning=response_reasoning,
            has_reasoning_block=has_reasoning_block,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_api_response=raw_api_response,
        )

    def complete(
        self,
        prompt: str,
        *,
        effort: EffortName,
        structured_output_schema: type[StructuredOutput | QualityScores],
        system: str | None = None,
    ) -> GenerationResult:
        max_tokens = IDEA_MODEL_MAX_TOKENS[self.idea_model]
        client = _get_client(self.config)
        request = self._build_request(
            prompt,
            effort=effort,
            structured_output_schema=structured_output_schema,
            max_tokens=max_tokens,
            system=system,
        )

        # Branch on call shape explicitly: generation uses prompt-only;
        # reasoning judge passes the LiveIdeaBench critic_prompt as the
        # Bedrock Converse ``system`` field and the idea body as the
        # user message (paper §3.4).
        if system is None:
            observation_input: list[dict[str, str]] = [
                {"role": "user", "content": prompt},
            ]
        else:
            observation_input = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

        generation = self._langfuse.start_observation(
            name="bedrock_converse",
            as_type="generation",
            model=self.idea_model.value,
            input=observation_input,
            metadata={"effort": effort.value, "provider": "bedrock"},
        )

        response = call_with_retry(lambda: client.converse(**request))
        result = self._parse_response(response, structured_output_schema)

        generation.update(
            output=result.text,
            usage_details={
                "input": result.input_tokens,
                "output": result.output_tokens,
                "total": result.total_tokens,
            },
            metadata={
                "effort": effort.value,
                "provider": "bedrock",
                "stop_reason": result.stop_reason,
                "has_reasoning_block": result.has_reasoning_block,
            },
        )
        generation.end()

        return result
