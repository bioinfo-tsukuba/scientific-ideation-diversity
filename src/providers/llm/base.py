"""Abstract base for LLM generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.model_registry import EffortName
from src.schemas.generation import GenerationResult, StructuredOutput
from src.schemas.llm_judge_quality import QualityScores


class GenerationBackend(ABC):
    """Base class for all LLM generation provider backends.

    Concrete backends are reused by both the generation pipeline (idea
    generation with a single-user-message prompt and a generation-side
    structured-output schema) and the reasoning-judge pipeline
    (LiveIdeaBench critic_prompt as ``system`` + idea as ``user``, with
    :class:`~src.schemas.llm_judge_quality.QualityScores` as the schema).
    Backends therefore accept either schema family and an optional
    system prompt; concrete backends decide how to encode the system
    message on the vendor's API (top-level ``system`` parameter on
    Bedrock / Anthropic, ``system_instruction`` on Vertex, role=``system``
    message on OpenAI).
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        effort: EffortName,
        structured_output_schema: type[StructuredOutput | QualityScores],
        system: str | None = None,
    ) -> GenerationResult:
        """Fire one LLM request and return a structured result.

        Args:
            prompt: User prompt text.
            effort: Reasoning effort level.
            structured_output_schema: Pydantic model class whose JSON Schema
                constrains the output format.  Union of generation-side
                :data:`StructuredOutput` shapes and reasoning-judge-side
                :class:`QualityScores`; both families share this single
                entry point (PR #253 review).
            system: Optional system prompt.  ``None`` keeps the original
                generation-side behaviour (single user message).  When
                provided, the backend routes it to its vendor-specific
                system slot.
        """
