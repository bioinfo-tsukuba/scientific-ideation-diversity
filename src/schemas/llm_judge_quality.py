"""Schemas for LLM-as-judge quality critique (single-idea evaluation).

Implements the LiveIdeaBench v4 ``critic_prompt`` protocol (Nature /
Science reviewer persona; 3-axis 1-10 scoring of originality /
feasibility / clarity) with structured output.  The schema shape
mirrors the pairwise fluency judge (:mod:`src.schemas.llm_judge`) so
downstream analysis code can follow the same pattern and so idea text
can be recovered from ``samples.jsonl`` + ``sample_row_index`` in the
same way as the pairwise judge recovers the two ideas behind a pair.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from src.model_registry import EffortName, JudgeModelName
from src.schemas.metadata import RawApiResponse, ResponseMetadata, TokenUsage


def _score_1_to_10(value: int) -> int:
    """Enforce ``1 <= value <= 10`` after parsing.

    Mirrors :func:`src.schemas.generation._reject_empty_after_strip`:
    running the range check as an ``AfterValidator`` keeps the emitted
    JSON schema as a plain ``"type": "integer"`` so OpenAI strict
    structured-output mode accepts it — that mode forbids ``minimum`` /
    ``maximum`` / ``multipleOf`` in the request schema — while still
    validating the response value Python-side.
    """
    if not 1 <= value <= 10:
        raise ValueError(f"score must be in 1..10 (got {value})")
    return value


Score1To10 = Annotated[int, AfterValidator(_score_1_to_10)]


class QualityScores(BaseModel):
    """Structured output from the LiveIdeaBench ``critic_prompt``.

    Fed to the judge backend so the JSON schema is derived from pydantic
    (``.model_json_schema()``), mirroring the generation pipeline and
    the pairwise judge.  ``extra="forbid"`` makes the emitted schema
    ``additionalProperties: false``, required by OpenAI's strict
    structured-output mode.

    The upstream prompt instructs the model to emit "a text analysis
    followed by a JSON score block"; we fold both parts into the single
    JSON object so strict structured output can parse the response
    without a regex retry.  The information content is preserved.
    """

    model_config = ConfigDict(extra="forbid")

    analysis: str
    originality: Score1To10
    feasibility: Score1To10
    clarity: Score1To10


class QualityInvocation(TokenUsage):
    """Internal bundle returned by a non-reasoning quality judge backend's ``_invoke``.

    Carries the minimum that
    :class:`~src.providers.llm_judge.base.StructuredQualityJudgeBase` needs
    to assemble a :class:`QualityJudgeResponse`.  Transport type between a
    provider backend and the shared Template Method base; not persisted
    on its own.

    Inherits :class:`~src.schemas.metadata.TokenUsage` to share the
    ``input_tokens`` / ``output_tokens`` / ``total_tokens`` contract with
    :class:`QualityJudgeResponse`, mirroring
    :class:`~src.schemas.llm_judge.PairwiseInvocation`.

    Reasoning-capable backends return :class:`ReasoningQualityInvocation`
    instead, so this class deliberately omits the reasoning fields:
    splitting the type lets the base class dispatch on ``isinstance`` and
    keeps the contract "this is a non-reasoning result" unambiguous.
    """

    scores: QualityScores
    stop_reason: str
    raw_api_response: RawApiResponse


class ReasoningQualityInvocation(QualityInvocation):
    """Internal bundle returned by a reasoning-capable quality judge backend.

    All three reasoning-judge fields are required (not Optional) on this
    subclass: the existence of the result *is* the proof that reasoning
    was enabled.  ``response_reasoning`` may still be ``None`` (the
    visible answer carries no reasoning trace under e.g. OpenAI's
    Responses API, which only surfaces token counts), but
    ``has_reasoning_block`` and ``reasoning_tokens`` must be populated
    explicitly by the concrete backend.

    Reasoning-token bookkeeping follows the cross-vendor convention in
    paper §B.1 (``tab:token-fields-app``): OpenAI from
    ``output_tokens_details.reasoning_tokens``; Vertex Gemini from
    ``usage_metadata.thoughts_token_count``; Bedrock Claude derived via
    Anthropic ``count_tokens`` on the visible answer.
    """

    response_reasoning: Optional[str]
    has_reasoning_block: bool
    reasoning_tokens: int


class QualityTarget(BaseModel):
    """Identifies which sample is being critiqued.

    ``sample_row_index`` is the row index into the sorted ``samples.jsonl``
    produced by :func:`src.artifacts.load_sample_records`, matching the
    ``sample_i_index`` / ``sample_j_index`` convention used by the
    pairwise judge (:class:`src.schemas.llm_judge.PairSpec`).  Round-
    tripping a score back to the original idea text is
    ``samples_jsonl_path`` (from :class:`QualityJudgeRunManifest`) +
    ``sample_row_index`` -> the :class:`~src.schemas.sample.SampleRecord`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keyword: str
    effort: EffortName
    sample_row_index: int = Field(..., ge=0)


class QualityJudgeRequest(BaseModel):
    """A single quality-critique call specification.

    ``idea_text`` is extracted from the sample ahead of time so the
    backend only renders the two messages and issues the API call.

    ``judge_model`` is a :class:`JudgeModelName` covering both
    non-reasoning judges (paper main pipeline) and reasoning judges
    (strong-judge ablation; see ``rebuttal/strong_judge_design.md``).
    Branch on
    :func:`src.model_registry.is_reasoning_judge` to distinguish.
    ``judge_effort`` is populated only when ``judge_model`` is a
    reasoning judge; ``None`` for non-reasoning judges.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: QualityTarget
    judge_model: JudgeModelName
    prompt_template_version: str
    idea_text: str
    judge_effort: Optional[EffortName] = None


QualityJudgmentDedupKey = tuple[str, str, int, str]


def _quality_dedup_key(request: "QualityJudgeRequest") -> QualityJudgmentDedupKey:
    """Identity tuple used for idempotent resume.

    Shared by :class:`QualityJudgeResponse` (successes) and
    :class:`QualityJudgeErrorRecord` (failures) so both files can be
    consulted when deciding whether to re-issue a given sample/judge
    combination.  Enum values are unwrapped to strings so the key
    remains hashable across processes.
    """
    return (
        request.target.keyword,
        request.target.effort.value,
        request.target.sample_row_index,
        request.judge_model.value,
    )


class QualityJudgeResponse(ResponseMetadata, TokenUsage):
    """Successful result of a single quality-critique call with full audit trail.

    Inherits :class:`~src.schemas.metadata.ResponseMetadata` (reasoning
    trace + stop reason, all required) and
    :class:`~src.schemas.metadata.TokenUsage` so the field set stays
    aligned with the generation pipeline and the pairwise judge.
    Errors are surfaced via :class:`QualityJudgeErrorRecord` rather than
    Optional fields here.

    ``reasoning_tokens`` is the per-call count of reasoning / thinking
    tokens consumed by reasoning-capable judges (None for non-reasoning
    judges).  For OpenAI it is read from
    ``output_tokens_details.reasoning_tokens``; for Vertex Gemini from
    ``usage_metadata.thoughts_token_count``; for Bedrock Claude it is
    derived via Anthropic's ``count_tokens`` endpoint following the
    paper §B.1 convention (tab:token-fields-app).
    """

    model_config = ConfigDict(extra="forbid")

    request: QualityJudgeRequest
    scores: QualityScores
    system_prompt_text: str
    user_prompt_text: str
    raw_api_response: RawApiResponse
    timestamp: datetime
    reasoning_tokens: Optional[int] = None

    def dedup_key(self) -> QualityJudgmentDedupKey:
        return _quality_dedup_key(self.request)


class QualityJudgeErrorRecord(BaseModel):
    """Persistent record of a failed quality-critique attempt.

    Written to a companion ``*.errors.jsonl`` file.  On resume the
    service reads this file and retries every failed request, so this
    schema has no "retry count": a given ``dedup_key`` simply reappears
    in the successes file once it eventually succeeds.
    """

    model_config = ConfigDict(extra="forbid")

    request: QualityJudgeRequest
    system_prompt_text: str
    user_prompt_text: str
    timestamp: datetime
    error_type: str
    error_message: str
    raw_api_response: RawApiResponse = Field(default_factory=dict)

    def dedup_key(self) -> QualityJudgmentDedupKey:
        return _quality_dedup_key(self.request)


class QualityJudgeRunManifest(BaseModel):
    """Metadata for a whole quality-judge run (one judge model, one run_dir)."""

    model_config = ConfigDict(extra="forbid")

    run_dir: Path
    samples_path: Path
    judge_model: JudgeModelName
    prompt_template_version: str
    total_judgments_expected: int = Field(..., ge=0)
    started_at: datetime
    completed_at: datetime | None = None
    completed_judgments: int = Field(default=0, ge=0)
    error_judgments: int = Field(default=0, ge=0)
    excluded_keywords: tuple[str, ...] = ()
    judge_effort: Optional[EffortName] = None
    # Sample scoping recorded for reproducibility + resume-compat.
    # Added with the strong-judge ablation CLI flags
    # (``--filter-keywords-csv`` / ``--filter-effort``).  ``None`` on
    # runs that did not scope (e.g., paper-pipeline non-reasoning runs).
    filter_keywords_csv: Path | None = None
    filter_effort: Optional[EffortName] = None
