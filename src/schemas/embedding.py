"""Embedding-related schemas: run specs, artifact rows, payloads, and results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from src.model_registry import (
    CategoryName,
    EffortName,
    EmbeddingModelName,
    IdeaModelName,
    PromptStyleName,
    ProviderName,
)
from src.schemas.metadata import RawApiResponse
from src.schemas.sample import SampleKey


class GenerationRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    idea_model: IdeaModelName
    provider: ProviderName
    prompt_style: PromptStyleName
    output_dir: Path


class EmbeddingRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_model: EmbeddingModelName
    provider: ProviderName
    output_dir: Path


class EmbeddingArtifactRow(SampleKey, BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: EmbeddingModelName
    embedding_row_index: int
    input_text_token_count: int
    raw_api_response: RawApiResponse


class EmbeddingPayload(BaseModel):
    """Per-text payload returned by embedding workers and cached artifact loaders."""

    model_config = ConfigDict(extra="forbid")

    raw_api_response: RawApiResponse
    input_text_token_count: int


class PcaCoordinateRow(BaseModel):
    """One row of PCA coordinate output."""

    model_config = ConfigDict(extra="forbid")

    keyword: str
    category: CategoryName
    prompt_style: PromptStyleName
    effort: EffortName
    sample_index: int
    x: float
    y: float
    word_count: int
    output_tokens: int
    stop_reason: Optional[str]
    has_reasoning_block: int
    idea: str


class EmbeddingResult(BaseModel):
    """Structured return value from each per-text embedding worker callable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: np.ndarray
    input_text_token_count: int
    raw_api_response: RawApiResponse

    @field_validator("vector")
    @classmethod
    def _check_vector_1d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            raise ValueError(f"Expected 1-D vector, got shape {v.shape}")
        if v.shape[0] == 0:
            raise ValueError("Vector must have at least one element")
        return v
