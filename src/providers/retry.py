"""Shared retry utility for provider backends, powered by tenacity."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_TOKENS = frozenset(
    {
        "throttl",
        "too many requests",
        "timeout",
        "temporar",
        "service unavailable",
        "modelnotready",
        "internalserver",
        # Vertex / Gemini 429 quota signal: ``RESOURCE_EXHAUSTED`` /
        # ``Resource has been exhausted``.
        "exhausted",
    }
)


def _is_retryable(exc: BaseException) -> bool:
    error_str = str(exc).lower()
    return any(tok in error_str for tok in _RETRYABLE_TOKENS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    max_wait: float = 30.0,
) -> T:
    """Call *fn* with exponential backoff on transient errors via tenacity.

    Used by backends that do not have built-in retry (e.g. Bedrock Converse).
    LiteLLM has its own retry mechanism.

    Args:
        fn: Zero-argument callable that performs the API request.
        max_retries: Maximum number of retry attempts.
        max_wait: Maximum wait time per retry in seconds.
    """
    for attempt in Retrying(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential_jitter(initial=1, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return fn()
    raise RuntimeError("Unreachable: tenacity retry loop exited without result")
