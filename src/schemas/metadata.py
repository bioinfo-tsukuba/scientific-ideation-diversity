"""Response metadata, token usage, and API trace schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

# Shared type for provider-specific API response dicts (Bedrock, OpenAI, etc.).
# Use this alias wherever a raw API response is stored.
RawApiResponse = dict[str, Any]

# JSON-serialised form of RawApiResponse, stored in records and artifacts.
SerialisedApiResponse = str


class ResponseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_reasoning: Optional[str]
    has_reasoning_block: bool
    stop_reason: Optional[str]


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ApiTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_api_response: SerialisedApiResponse
    api_calls: list[dict[str, Any]]


