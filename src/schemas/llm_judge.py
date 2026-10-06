"""Schemas for LLM-as-judge pairwise similarity evaluation.

Implements the LiveIdeaBench v4 fluency-critic protocol (ABCD ordinal labels)
with structured output, stratified pair sampling, and position-bias control.
See docs/20260418_llm_judge_plan.md for the full design.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.model_registry import EffortName, EmbeddingModelName, JudgeModelName
from src.schemas.metadata import RawApiResponse, ResponseMetadata, TokenUsage


class JudgeAnswer(str, Enum):
    """LiveIdeaBench v4 fluency-critic ABCD ordinal labels.

    Ordering by conceptual distance (A = most different, D = most similar).
    """

    A = "A"  # Completely different ideas addressing different problems
    B = "B"  # Different ideas but addressing similar problems
    C = "C"  # Similar ideas addressing similar or identical problems
    D = "D"  # Academically identical ideas with same core approach and problem statement


class FluencyAnswer(BaseModel):
    """Structured output wrapper for the fluency-critic judgment.

    Fed to the judge backend so the JSON schema is derived from pydantic
    (``.model_json_schema()``), mirroring the generation pipeline's pattern
    of using pydantic models as the source of truth for structured output.
    ``extra="forbid"`` makes the emitted schema ``additionalProperties: false``
    which is required for OpenAI's strict structured-output mode.
    """

    model_config = ConfigDict(extra="forbid")

    answer: JudgeAnswer


class PairwiseInvocation(TokenUsage):
    """Internal result bundle returned by a pairwise judge backend's ``_invoke``.

    Carries the minimum a :class:`~src.providers.llm_judge.base.StructuredJudgeBase`
    needs to assemble a :class:`JudgeResponse`.  This is a transport type
    between a provider backend and the shared Template Method in the base
    class; it is not persisted to disk on its own, and the request / timestamp
    / reasoning-block fields that distinguish a full :class:`JudgeResponse`
    are added by the base class after ``_invoke`` returns.

    Inherits :class:`~src.schemas.metadata.TokenUsage` to share the
    ``input_tokens`` / ``output_tokens`` / ``total_tokens`` contract with
    :class:`JudgeResponse` (which also inherits ``TokenUsage``) so the base
    class can forward fields by name when assembling the response.
    """

    answer: JudgeAnswer
    stop_reason: str
    raw_api_response: RawApiResponse


class PairStratum(str, Enum):
    """Which stratum of the embedding-distance ranking a pair was drawn from.

    ``TOP``/``BOTTOM`` pick the extremes so the judge can validate that
    embedding-distant pairs are semantically distinct and embedding-close
    pairs are semantically identical.  ``MIDDLE`` picks pairs whose rank is
    around the median of the distance distribution; this anchors the middle
    of the calibration curve (R2 in the Results structure; see
    ``docs/20260419_results_structure_combined_vs_facet.md``).
    """

    TOP = "top"        # most distant in embedding space
    MIDDLE = "middle"  # around median embedding distance
    BOTTOM = "bottom"  # most similar in embedding space


class JudgeOrder(str, Enum):
    """Order in which the two ideas are presented to the judge.

    Used to control for position bias (MT-Bench / AlpacaEval 2.0 style).
    """

    AB = "ab"  # position A = sample_i (canonical), position B = sample_j
    BA = "ba"  # position A = sample_j, position B = sample_i


class PairSpec(BaseModel):
    """Canonical identification of a sample pair selected for judgment.

    ``sample_i_index`` and ``sample_j_index`` are row indices into the sorted
    ``samples.jsonl`` (the ordering produced by
    :func:`src.artifacts.load_sample_records`).  Enforces
    ``sample_i_index < sample_j_index`` so the same pair has exactly one canonical
    representation regardless of selection order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keyword: str
    effort: EffortName
    sample_i_index: int = Field(..., ge=0)
    sample_j_index: int = Field(..., ge=0)
    embedding_distance: float = Field(..., ge=0.0, le=2.0)
    embedding_model: EmbeddingModelName
    stratum: PairStratum
    rank: int = Field(..., ge=1)


class JudgeRequest(BaseModel):
    """A single LLM-judge call specification.

    ``idea_text_a`` and ``idea_text_b`` are already assigned to their positions
    according to :attr:`order`, so the backend just has to render the prompt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair: PairSpec
    order: JudgeOrder
    judge_model: JudgeModelName
    prompt_template_version: str
    idea_text_a: str
    idea_text_b: str


JudgmentDedupKey = tuple[str, str, int, int, str, str]


def _judgment_dedup_key(request: "JudgeRequest") -> JudgmentDedupKey:
    """Identity tuple used for idempotent resume.

    Shared by :class:`JudgeResponse` (successes) and :class:`JudgeErrorRecord`
    (failures) so both files can be consulted when deciding whether to re-issue
    a given pair/order/judge combination.  Enum values are unwrapped to strings
    so the key remains hashable across processes.
    """
    return (
        request.pair.keyword,
        request.pair.effort.value,
        request.pair.sample_i_index,
        request.pair.sample_j_index,
        request.order.value,
        request.judge_model.value,
    )


class JudgeResponse(ResponseMetadata, TokenUsage):
    """Successful result of a single LLM-judge call with full audit trail.

    Inherits :class:`~src.schemas.metadata.ResponseMetadata` (reasoning
    trace + stop reason, all required) and
    :class:`~src.schemas.metadata.TokenUsage` so the field set stays
    aligned with the generation pipeline's SSOT.  Errors are surfaced via
    :class:`JudgeErrorRecord` rather than via Optional fields here.
    """

    model_config = ConfigDict(extra="forbid")

    request: JudgeRequest
    answer: JudgeAnswer
    prompt_text: str
    raw_api_response: RawApiResponse
    timestamp: datetime

    def dedup_key(self) -> JudgmentDedupKey:
        return _judgment_dedup_key(self.request)


class JudgeErrorRecord(BaseModel):
    """Persistent record of a failed LLM-judge attempt.

    Written to a companion JSONL file (``*.errors.jsonl``).  On resume the
    service reads this file and retries every failed request, so
    :class:`JudgeErrorRecord` has no "retry count" field; a given
    ``dedup_key`` simply reappears in the successes file once it finally
    succeeds.
    """

    model_config = ConfigDict(extra="forbid")

    request: JudgeRequest
    prompt_text: str
    timestamp: datetime
    error_type: str
    error_message: str
    raw_api_response: RawApiResponse = Field(default_factory=dict)

    def dedup_key(self) -> JudgmentDedupKey:
        return _judgment_dedup_key(self.request)


class JudgeRunManifest(BaseModel):
    """Metadata about a whole LLM-judge run (one judge model over one run_dir)."""

    model_config = ConfigDict(extra="forbid")

    run_dir: Path
    samples_path: Path
    embeddings_path: Path
    embedding_model_for_sort: EmbeddingModelName
    judge_model: JudgeModelName
    prompt_template_version: str
    top_m: int = Field(..., ge=0)
    middle_m: int = Field(default=0, ge=0)
    bottom_m: int = Field(..., ge=0)
    both_orders: bool
    total_judgments_expected: int = Field(..., ge=0)
    started_at: datetime
    completed_at: datetime | None = None
    completed_judgments: int = Field(default=0, ge=0)
    error_judgments: int = Field(default=0, ge=0)
    excluded_keywords: tuple[str, ...] = ()
