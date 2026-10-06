"""Generation-related schemas: structured output, request context, and results."""

from __future__ import annotations

from typing import Annotated, Any, Union

from pydantic import AfterValidator, BaseModel, ConfigDict

from src.model_registry import (
    EffortName,
    IdeaModelName,
    PromptStyleName,
    ProviderName,
)
from src.schemas.llm_judge_quality import QualityScores
from src.schemas.metadata import (
    ApiTrace,
    RawApiResponse,
    ResponseMetadata,
    SerialisedApiResponse,
    TokenUsage,
)
from src.schemas.validation import ValidationMetadata


def _normalize(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split())


def _reject_empty_after_strip(value: str) -> str:
    """Reject strings that are empty or whitespace-only.

    Used as an ``AfterValidator`` inside ``NonEmptyText``; runs Python-side
    after the base ``str`` parse, so the generated JSON schema stays a plain
    ``"type": "string"`` — OpenAI strict structured-output mode forbids
    ``minLength`` / ``maxLength`` / ``pattern`` in the request schema, and
    Bedrock / Vertex simply ignore them — but the pydantic parse of the API
    response catches degenerate outputs before they leak into ``samples.jsonl``
    and blow up downstream (see ``docs/20260420_phase_b1_execution.md`` §4.1).
    """
    if not value.strip():
        raise ValueError("must be non-empty after whitespace strip")
    return value


NonEmptyText = Annotated[str, AfterValidator(_reject_empty_after_strip)]


class StructuredBgIdeaOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: NonEmptyText
    idea: NonEmptyText

    def combined_text(self) -> str:
        return f"{self.background} {self.idea}"

    def text_for_component(self, component: str) -> str:
        if component == "background":
            return self.background
        if component == "idea":
            return self.idea
        raise ValueError(f"Unsupported component {component!r} for StructuredBgIdeaOutput")

    def specter2_parts(self) -> tuple[str, str]:
        return _normalize(self.background), _normalize(self.idea)


class StructuredFacetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: NonEmptyText
    mechanism: NonEmptyText
    evaluation: NonEmptyText

    def combined_text(self) -> str:
        return f"{self.purpose} {self.mechanism} {self.evaluation}"

    def text_for_component(self, component: str) -> str:
        if component == "purpose":
            return self.purpose
        if component == "mechanism":
            return self.mechanism
        if component == "evaluation":
            return self.evaluation
        raise ValueError(f"Unsupported component {component!r} for StructuredFacetOutput")

    def specter2_parts(self) -> tuple[str, str]:
        return _normalize(self.purpose), _normalize(self.mechanism)


class FacetIdeaWithProbability(BaseModel):
    """One facet idea + a verbalized probability — the inner element of VS responses."""

    model_config = ConfigDict(extra="forbid")

    purpose: NonEmptyText
    mechanism: NonEmptyText
    evaluation: NonEmptyText
    probability: float

    def combined_text(self) -> str:
        return f"{self.purpose} {self.mechanism} {self.evaluation}"

    def text_for_component(self, component: str) -> str:
        if component == "purpose":
            return self.purpose
        if component == "mechanism":
            return self.mechanism
        if component == "evaluation":
            return self.evaluation
        raise ValueError(
            f"Unsupported component {component!r} for FacetIdeaWithProbability"
        )

    def specter2_parts(self) -> tuple[str, str]:
        return _normalize(self.purpose), _normalize(self.mechanism)


class StructuredVSOutput(BaseModel):
    """Verbalized Sampling output: a list of k=3 facet ideas with verbalized probabilities."""

    model_config = ConfigDict(extra="forbid")

    responses: list[FacetIdeaWithProbability]


class StructuredSSoTOutput(BaseModel):
    """SSoT output: a model-generated random_string seed + one facet idea."""

    model_config = ConfigDict(extra="forbid")

    random_string: NonEmptyText
    purpose: NonEmptyText
    mechanism: NonEmptyText
    evaluation: NonEmptyText

    def combined_text(self) -> str:
        return f"{self.purpose} {self.mechanism} {self.evaluation}"

    def text_for_component(self, component: str) -> str:
        if component == "purpose":
            return self.purpose
        if component == "mechanism":
            return self.mechanism
        if component == "evaluation":
            return self.evaluation
        raise ValueError(
            f"Unsupported component {component!r} for StructuredSSoTOutput"
        )

    def specter2_parts(self) -> tuple[str, str]:
        return _normalize(self.purpose), _normalize(self.mechanism)


StructuredOutput = Union[
    StructuredBgIdeaOutput,
    StructuredFacetOutput,
    FacetIdeaWithProbability,
    StructuredVSOutput,
    StructuredSSoTOutput,
]
"""Union of all structured-output shapes that ``generate_idea`` may return.
The first two (``StructuredBgIdeaOutput``, ``StructuredFacetOutput``) are the
default phase2-pipeline shapes (one idea per call). ``StructuredVSOutput``
holds k=3 verbalized-sampling responses per call; ``StructuredSSoTOutput``
holds a single seed-derived idea per call. The prompt-sensitivity pilot
runner handles the list-vs-single shape difference when it expands a VS call
into 3 sample records."""


class StructuredIdeaContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: str
    full_response: str
    structured_output: StructuredOutput


class GenerationRequestContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea_model: IdeaModelName
    provider: ProviderName
    effort_name: EffortName
    prompt_style: PromptStyleName


class PromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    structured_output_schema: type[StructuredOutput]


class IdeaGenerationRequest(GenerationRequestContext, PromptPayload):
    pass


class IdeaGenerationResult(
    StructuredIdeaContent,
    ValidationMetadata,
    ResponseMetadata,
    TokenUsage,
    ApiTrace,
):
    pass


class GenerationResult(ResponseMetadata, TokenUsage):
    """Return value from a single LLM completion call.

    Inherits :class:`~src.schemas.metadata.ResponseMetadata` (reasoning trace
    + stop reason, all required) and :class:`~src.schemas.metadata.TokenUsage`
    so the field set stays aligned with ``IdeaGenerationServiceResult`` and
    ``JudgeResponse``.  Defaulting ``response_reasoning`` / ``stop_reason``
    here would make "None because the model has no reasoning block" and
    "None because the backend forgot to populate it" indistinguishable, so
    every backend must pass the fields explicitly.

    Reasoning-token *counts* (as opposed to the textual trace in
    ``response_reasoning``) are not first-class on this result: existing
    analysis code reads them on demand from ``raw_api_response`` at
    analyze-time
    (``scripts/experiment/prompt_sensitivity/analyze_prompt_sensitivity_token_pareto.py``,
    paper §B.1 / ``tab:token-fields-app``).  The reasoning-judge
    backend does the same vendor-specific extraction inside its own
    ``_invoke`` so the generation-side schema stays unchanged.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    # Either a generation-side :data:`StructuredOutput` shape or a
    # reasoning-judge :class:`~src.schemas.llm_judge_quality.QualityScores`,
    # matching the widened ``structured_output_schema`` type on
    # :meth:`~src.providers.llm.base.GenerationBackend.complete` (the
    # reasoning-judge backends construct generation backends with
    # ``QualityScores`` as the schema).
    structured_output: StructuredOutput | QualityScores

    raw_api_response: RawApiResponse


class IdeaGenerationServiceResult(ResponseMetadata, TokenUsage):
    """Return value from :func:`generate_idea`.

    Inherits reasoning metadata from :class:`ResponseMetadata`
    and token counts from :class:`TokenUsage` so that the
    schema structure stays consistent across the codebase (SSOT).
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    full_response: str
    structured_output: StructuredOutput
    raw_api_response: SerialisedApiResponse
    api_calls: list[dict[str, Any]] = []
