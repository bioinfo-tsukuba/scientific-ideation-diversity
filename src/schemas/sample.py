"""Sample key, model config, statistics, and record schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.model_registry import (
    CategoryName,
    EffortName,
    EmbeddingModelName,
    IdeaModelName,
    PromptStyleName,
)
from src.schemas.generation import StructuredIdeaContent
from src.schemas.metadata import (
    ApiTrace,
    ResponseMetadata,
    TokenUsage,
)
from src.schemas.validation import ValidationMetadata


class SampleKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str
    category: CategoryName
    prompt_style: PromptStyleName
    effort: EffortName
    sample_index: int


class SampleModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea_model: IdeaModelName
    embedding_model: EmbeddingModelName


class SampleStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_input_tokens: int
    word_count: int
    char_count: int
    timestamp: str


class SampleRecord(
    SampleKey,
    SampleModelConfig,
    StructuredIdeaContent,
    ValidationMetadata,
    ResponseMetadata,
    TokenUsage,
    ApiTrace,
    SampleStatistics,
):

    def for_embedding_model(self, embedding_model: EmbeddingModelName) -> "SampleRecord":
        return self.model_copy(
            update={
                "embedding_model": embedding_model,
                "embedding_input_tokens": 0,
            }
        )
