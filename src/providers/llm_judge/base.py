"""Abstract bases for LLM-judge backends.

Two parallel ABCs cover the two judge shapes in this project:

  * :class:`LLMJudgeBackend` — pairwise distinctness (single prompt, ABCD label).
  * :class:`QualityJudgeBackend` — single-idea LiveIdeaBench critic (system +
    user prompt, 3-axis rubric).

A single base with a generic request would obscure the call-site shape
difference (pairwise callers would have to pass an unused
``system_prompt_text`` through), so both stay.

Error contract:
    The service layer catches exceptions and records them as error
    records.  Backends MUST raise on transport errors, missing fields,
    or parse failures rather than returning partially populated
    responses.  This matches the generation-backend pattern.

``StructuredJudgeBase`` / ``StructuredQualityJudgeBase`` provide a
Template Method default implementation of :meth:`judge` that centralises
Langfuse tracing, retry wrapping, and response assembly.  Provider
backends only implement ``_invoke`` (the actual client call + parse)
plus two small metadata hooks (``_provider_tag`` and
``_observation_name``).  This keeps the six concrete backends
(OpenAI / Anthropic / Gemini × pairwise / quality) free of duplicated
ritual code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from src.model_registry import JudgeModelName, ProviderName
from src.providers.retry import call_with_retry
from src.providers.tracing import get_langfuse
from src.schemas.llm_judge import (
    JudgeRequest,
    JudgeResponse,
    PairwiseInvocation,
)
from src.schemas.llm_judge_quality import (
    QualityInvocation,
    QualityJudgeRequest,
    QualityJudgeResponse,
    ReasoningQualityInvocation,
)


class LLMJudgeBackend(ABC):
    """Backend that dispatches a single pairwise judgment call to an LLM provider."""

    @property
    @abstractmethod
    def model_name(self) -> JudgeModelName:
        """Judge model enum member (matches the provider API model string)."""

    @abstractmethod
    def judge(self, request: JudgeRequest, prompt_text: str) -> JudgeResponse:
        """Issue one judge call and return a populated :class:`JudgeResponse`.

        Raises on any error (API transport failure after retries, missing
        output_text, missing usage, structured-output parse failure).
        """


class QualityJudgeBackend(ABC):
    """Backend that dispatches a single quality-critique call to an LLM provider.

    Split from :class:`LLMJudgeBackend` because the call shape is different
    (system + user prompt instead of a single prompt string); widening
    :class:`LLMJudgeBackend` to cover both would force every pairwise caller
    to pass through an unused ``system_prompt_text`` parameter.
    """

    @property
    @abstractmethod
    def model_name(self) -> JudgeModelName:
        """Judge model enum member (matches the provider API model string)."""

    @abstractmethod
    def judge(
        self,
        request: QualityJudgeRequest,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> QualityJudgeResponse:
        """Issue one quality-critique call and return a populated response.

        Raises on any error (transport failure, missing fields, parse
        failure).
        """


class StructuredJudgeBase(LLMJudgeBackend):
    """Template Method base for non-reasoning structured-output pairwise judges.

    Subclasses implement:
      * :meth:`_invoke` — make the client call, validate structured output,
        return a :class:`PairwiseInvocation`.
      * :meth:`_provider_tag` — :class:`~src.model_registry.ProviderName`
        enum member; the base writes its ``.value`` into Langfuse metadata.
      * :meth:`_observation_name` — Langfuse observation name
        (e.g. ``"openai_judge_fluency_critic"``).

    The shared :meth:`judge` implementation handles:
      * Langfuse ``start_observation`` / ``update`` / ``end``
      * :func:`call_with_retry` wrapping of ``_invoke``
      * :class:`JudgeResponse` construction (identical across providers)
    """

    def __init__(self, *, model: JudgeModelName, samples_jsonl_path: Path) -> None:
        self._model = model
        self._samples_jsonl_path = samples_jsonl_path
        self._langfuse = get_langfuse()

    @property
    def model_name(self) -> JudgeModelName:
        return self._model

    def judge(self, request: JudgeRequest, prompt_text: str) -> JudgeResponse:
        started_at = datetime.now(timezone.utc)

        # ``samples_jsonl_path`` + ``sample_*_index`` let a trace round-trip to
        # the exact idea text judged without leaking the text into Langfuse.
        generation = self._langfuse.start_observation(
            name=self._observation_name(),
            as_type="generation",
            model=self._model.value,
            input=[{"role": "user", "content": prompt_text}],
            metadata={
                "provider": self._provider_tag().value,
                "temperature": 0,
                "prompt_template_version": request.prompt_template_version,
                "keyword": request.pair.keyword,
                "effort": request.pair.effort.value,
                "stratum": request.pair.stratum.value,
                "pair_rank": request.pair.rank,
                "order": request.order.value,
                "samples_jsonl_path": str(self._samples_jsonl_path),
                "sample_i_index": request.pair.sample_i_index,
                "sample_j_index": request.pair.sample_j_index,
                "embedding_distance": request.pair.embedding_distance,
                "embedding_model_for_sort": request.pair.embedding_model.value,
            },
        )

        invocation = call_with_retry(lambda: self._invoke(prompt_text))

        generation.update(
            output=invocation.answer.value,
            usage_details={
                "input": invocation.input_tokens,
                "output": invocation.output_tokens,
                "total": invocation.total_tokens,
            },
            metadata={
                "stop_reason": invocation.stop_reason,
                "answer": invocation.answer.value,
            },
        )
        generation.end()

        return JudgeResponse(
            request=request,
            answer=invocation.answer,
            prompt_text=prompt_text,
            raw_api_response=invocation.raw_api_response,
            timestamp=started_at,
            response_reasoning=None,
            has_reasoning_block=False,
            stop_reason=invocation.stop_reason,
            input_tokens=invocation.input_tokens,
            output_tokens=invocation.output_tokens,
            total_tokens=invocation.total_tokens,
        )

    @abstractmethod
    def _invoke(self, prompt_text: str) -> PairwiseInvocation:
        """Dispatch the provider call, parse structured output, return the bundle.

        Raises on any error.  The shared :meth:`judge` wraps this in
        :func:`call_with_retry`.
        """

    @abstractmethod
    def _provider_tag(self) -> ProviderName:
        """:class:`~src.model_registry.ProviderName` enum member for this backend."""

    @abstractmethod
    def _observation_name(self) -> str:
        """Langfuse observation name."""


class StructuredQualityJudgeBase(QualityJudgeBackend):
    """Template Method base for structured-output quality judges.

    Same pattern as :class:`StructuredJudgeBase` but for the single-idea
    critic call shape (system + user prompt, 4-axis rubric output).
    Supports both non-reasoning and reasoning judges: the shared
    :meth:`judge` reads reasoning fields from the :class:`QualityInvocation`
    returned by ``_invoke``, defaulting to ``None`` / ``False`` /
    ``None`` for non-reasoning concrete backends.
    """

    def __init__(
        self,
        *,
        model: JudgeModelName,
        samples_jsonl_path: Path,
    ) -> None:
        self._model = model
        self._samples_jsonl_path = samples_jsonl_path
        self._langfuse = get_langfuse()

    @property
    def model_name(self) -> JudgeModelName:
        return self._model

    def judge(
        self,
        request: QualityJudgeRequest,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> QualityJudgeResponse:
        started_at = datetime.now(timezone.utc)

        input_messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text},
        ]
        generation = self._langfuse.start_observation(
            name=self._observation_name(),
            as_type="generation",
            model=self._model.value,
            input=input_messages,
            metadata={
                "provider": self._provider_tag().value,
                "temperature": 0,
                "prompt_template_version": request.prompt_template_version,
                "keyword": request.target.keyword,
                "effort": request.target.effort.value,
                "samples_jsonl_path": str(self._samples_jsonl_path),
                "sample_row_index": request.target.sample_row_index,
            },
        )

        invocation = call_with_retry(
            lambda: self._invoke(
                system_prompt_text=system_prompt_text,
                user_prompt_text=user_prompt_text,
            )
        )

        # Discriminate by invocation type so the non-reasoning path
        # never accesses absent reasoning fields and the reasoning path
        # is type-checked to have populated them.
        if isinstance(invocation, ReasoningQualityInvocation):
            response_reasoning = invocation.response_reasoning
            has_reasoning_block = invocation.has_reasoning_block
            reasoning_tokens: int | None = invocation.reasoning_tokens
        else:
            response_reasoning = None
            has_reasoning_block = False
            reasoning_tokens = None

        generation.update(
            output={
                "originality": invocation.scores.originality,
                "feasibility": invocation.scores.feasibility,
                "clarity": invocation.scores.clarity,
            },
            usage_details={
                "input": invocation.input_tokens,
                "output": invocation.output_tokens,
                "total": invocation.total_tokens,
            },
            metadata={
                "stop_reason": invocation.stop_reason,
                "originality": invocation.scores.originality,
                "feasibility": invocation.scores.feasibility,
                "clarity": invocation.scores.clarity,
                "reasoning_tokens": reasoning_tokens,
                "has_reasoning_block": has_reasoning_block,
            },
        )
        generation.end()

        return QualityJudgeResponse(
            request=request,
            scores=invocation.scores,
            system_prompt_text=system_prompt_text,
            user_prompt_text=user_prompt_text,
            raw_api_response=invocation.raw_api_response,
            timestamp=started_at,
            response_reasoning=response_reasoning,
            has_reasoning_block=has_reasoning_block,
            stop_reason=invocation.stop_reason,
            input_tokens=invocation.input_tokens,
            output_tokens=invocation.output_tokens,
            total_tokens=invocation.total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    @abstractmethod
    def _invoke(
        self,
        *,
        system_prompt_text: str,
        user_prompt_text: str,
    ) -> QualityInvocation:
        """Dispatch the provider call, parse structured output, return the bundle.

        Raises on any error.  The shared :meth:`judge` wraps this in
        :func:`call_with_retry`.
        """

    @abstractmethod
    def _provider_tag(self) -> ProviderName:
        """:class:`~src.model_registry.ProviderName` enum member for this backend."""

    @abstractmethod
    def _observation_name(self) -> str:
        """Langfuse observation name."""
