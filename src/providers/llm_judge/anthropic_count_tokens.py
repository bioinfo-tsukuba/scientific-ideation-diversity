"""Shared helpers for Claude reasoning-token derivation via Anthropic count_tokens.

The Bedrock Converse ``usage`` block bills thinking tokens as part of
``outputTokens`` without exposing a subset field (issue #110 audited
the alternative routes), so any caller that needs the per-call
reasoning-token count derives it post-hoc by tokenising the visible
answer with Anthropic's native SDK and subtracting from
``outputTokens``.  Anthropic's count_tokens endpoint includes a small
constant per-call system overhead which must be added back; the
arithmetic is documented in paper §B.1 / ``tab:token-fields-app`` and
the script header of
``scripts/experiment/data/count_claude_reasoning_tokens.py``.

This module is the single source of truth for those helpers, used by:

* ``scripts/experiment/data/count_claude_reasoning_tokens.py`` —
  offline batch backfill over an existing generation run, populates
  ``reasoning_token_counts.csv`` for analysis scripts and the paper.
* :mod:`src.providers.llm_judge.anthropic_reasoning_quality_backend` —
  online per-call derivation inside the reasoning quality judge.

Adding it to ``src/`` (rather than keeping the helpers inside the
script) lets both callers import the same implementation; the constant
boilerplate offset is cached per process so repeated derivation does
not re-issue the probe call.
"""

from __future__ import annotations

import threading

from anthropic import Anthropic

from src.model_registry import IDEA_MODEL_ANTHROPIC_NATIVE_ID, IdeaModelName

_anthropic_client_lock = threading.Lock()
_anthropic_client_cache: Anthropic | None = None
_anthropic_overhead_cache: dict[str, int] = {}


def get_anthropic_client() -> Anthropic:
    """Thread-safe Anthropic SDK client cache."""
    global _anthropic_client_cache
    with _anthropic_client_lock:
        if _anthropic_client_cache is None:
            _anthropic_client_cache = Anthropic()
        return _anthropic_client_cache


def native_model_id(model: IdeaModelName) -> str:
    """Return the Anthropic-native model identifier for ``model``.

    Raises ``KeyError`` for non-Anthropic models so misuse fails loudly
    rather than silently falling back to the Bedrock inference-profile
    id (which the Anthropic SDK would reject).
    """
    return IDEA_MODEL_ANTHROPIC_NATIVE_ID[model]


def count_visible_answer_tokens(client: Anthropic, full_response: str, *, model: IdeaModelName) -> int:
    """Count tokens for the visible answer via Anthropic ``count_tokens``.

    Lower-level helper compatible with the existing batch script's call
    shape (explicit client passed in).  The reasoning judge typically
    calls :func:`derive_reasoning_tokens` instead.
    """
    response = client.messages.count_tokens(
        model=native_model_id(model),
        messages=[{"role": "user", "content": full_response}],
    )
    return int(response.input_tokens)


def measure_system_overhead(client: Anthropic, *, model: IdeaModelName) -> int:
    """Calibrate the per-call system overhead added by count_tokens.

    ``count_tokens(".") - 1`` isolates the constant boilerplate because
    a single period tokenises to exactly one BPE token on the Claude
    tokenizer.  Returned as a non-negative int for addition back into
    derived reasoning counts.  Mirrors the original helper inlined in
    ``scripts/experiment/data/count_claude_reasoning_tokens.py``.
    """
    probe = count_visible_answer_tokens(client, ".", model=model)
    overhead = probe - 1
    if overhead < 0:
        raise RuntimeError(
            f"Unexpected count_tokens probe: count_tokens('.') = {probe} "
            f"(expected >= 1); tokenizer contract has changed."
        )
    return overhead


def derive_reasoning_tokens(
    *,
    model: IdeaModelName,
    visible_text: str,
    output_tokens: int,
) -> int:
    """High-level wrapper used by the reasoning judge.

    Uses a cached client and a per-process overhead cache so repeated
    calls on the same model do not re-issue the probe.  Returns
    ``output_tokens - (count_tokens(visible_text) - system_overhead)``,
    clamped to ``>= 0``.
    """
    client = get_anthropic_client()
    native = native_model_id(model)
    overhead = _anthropic_overhead_cache.get(native)
    if overhead is None:
        overhead = measure_system_overhead(client, model=model)
        _anthropic_overhead_cache[native] = overhead
    visible_raw = count_visible_answer_tokens(client, visible_text, model=model)
    visible = max(visible_raw - overhead, 0)
    return max(output_tokens - visible, 0)
