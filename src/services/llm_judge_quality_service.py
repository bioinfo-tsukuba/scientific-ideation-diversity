"""Orchestration service for LLM-as-judge single-idea quality critique.

Responsibilities:
  * prompt loading from the LiveIdeaBench ``critic_prompt`` template
    (source of truth: the ``external/liveideabench`` submodule)
  * system / user message rendering, verbatim from
    ``external/liveideabench/utils/LLM.py`` ``CriticLLM.critique_idea``
  * request construction (one call per sample)
  * idempotent, concurrent judgment execution with JSONL audit trail

Data-flow (one run_dir, one judge model):

    samples.jsonl ──► build_quality_requests ──► run_quality_judgments ──► *.jsonl

Mirrors :mod:`src.services.llm_judge_service`; see that module for the
pairwise analogue.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.model_registry import EffortName, JudgeModelName
from src.providers.llm_judge.base import QualityJudgeBackend
from src.schemas.llm_judge_quality import (
    QualityJudgeErrorRecord,
    QualityJudgeRequest,
    QualityJudgeResponse,
    QualityJudgmentDedupKey,
    QualityTarget,
)
from src.schemas.sample import SampleRecord

logger = logging.getLogger(__name__)


# Source of truth for the critic_prompt: the LiveIdeaBench submodule.
# Loaded at import time so any upstream update is picked up as soon as
# the submodule is refreshed; ``PROMPT_TEMPLATE_HASH`` then reflects
# the change and the manifest records exactly which version was used.
# Mirrors the pairwise service loader.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVEIDEABENCH_PROMPTS_PATH = (
    _REPO_ROOT / "external" / "liveideabench" / "utils" / "prompts.json"
)

# Vestigial placeholder preserved from LiveIdeaBench's own caller
# (``external/liveideabench/run.py:143`` does
# ``critic_prompt.description.replace('{{keywords}}', str(keyword))``).
# The current upstream ``critic_prompt.description`` contains no
# ``{{keywords}}`` token so the replace is a no-op, but keeping it
# means we automatically follow upstream if the placeholder is
# reintroduced rather than silently shipping a stale template.
_PLACEHOLDER_KEYWORDS = "{{keywords}}"

# User message template, verbatim from
# ``external/liveideabench/utils/LLM.py`` ``CriticLLM.critique_idea``
# (``prompt = "Please evaluate the following scientific idea:\n\n" + idea``).
_USER_MESSAGE_PREFIX = "Please evaluate the following scientific idea:\n\n"


def _load_critic_prompt_template() -> str:
    """Read ``critic_prompt.description`` from the submodule."""
    with open(_LIVEIDEABENCH_PROMPTS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["critic_prompt"]["description"]


CRITIC_PROMPT_TEMPLATE: str = _load_critic_prompt_template()

# Hash of the raw upstream template (keyword-independent).  If upstream
# reintroduces ``{{keywords}}``, the rendered text will vary per
# keyword but the hash here still identifies the template form, which
# is what manifests need to detect version drift.
PROMPT_TEMPLATE_HASH: str = hashlib.sha256(
    CRITIC_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()

# ``PROMPT_TEMPLATE_VERSION`` tags the prompt both by its LiveIdeaBench
# lineage and by a short hash, so analysis scripts can detect when a
# run was collected under a different version without needing the
# submodule checked out.
PROMPT_TEMPLATE_VERSION: str = (
    f"liveideabench_critic_prompt@{PROMPT_TEMPLATE_HASH[:12]}"
)


def render_system_prompt(*, keyword: str) -> str:
    """Render the critic system prompt for one target.

    Uses :meth:`str.replace` instead of :meth:`str.format` because the
    LiveIdeaBench template is Jinja-style (``{{var}}``).  Keeping the
    vestigial ``{{keywords}}`` replace preserves parity with
    ``external/liveideabench/run.py:143`` even though the current
    upstream template has no placeholder.
    """
    return CRITIC_PROMPT_TEMPLATE.replace(_PLACEHOLDER_KEYWORDS, keyword)


def render_user_prompt(*, idea: str) -> str:
    """Render the user message, verbatim from LiveIdeaBench ``LLM.py`` line 389."""
    return _USER_MESSAGE_PREFIX + idea


def build_quality_requests(
    *,
    records: list[SampleRecord],
    judge_model: JudgeModelName,
    judge_effort: EffortName | None = None,
) -> list[QualityJudgeRequest]:
    """Build one :class:`QualityJudgeRequest` per sample record.

    ``records`` MUST be the canonical sort order produced by
    :func:`src.artifacts.load_sample_records`; the row index carries
    back to ``samples.jsonl`` only when this ordering is stable.

    ``judge_model`` is a :class:`JudgeModelName` covering both
    non-reasoning judges and reasoning judges; pass ``judge_effort``
    to record the effort tier on each reasoning-judge request
    (metadata only; the backend instance already knows its effort
    and is responsible for the actual API parameter).
    """
    requests: list[QualityJudgeRequest] = []
    for row_index, record in enumerate(records):
        target = QualityTarget(
            keyword=record.keyword,
            effort=record.effort,
            sample_row_index=row_index,
        )
        requests.append(
            QualityJudgeRequest(
                target=target,
                judge_model=judge_model,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                idea_text=record.idea,
                judge_effort=judge_effort,
            )
        )
    return requests


def errors_path_for(success_path: Path) -> Path:
    """Companion ``*.errors.jsonl`` path next to the main successes file."""
    return success_path.with_suffix(".errors.jsonl")


def _load_success_keys(path: Path) -> set[QualityJudgmentDedupKey]:
    """Return dedup keys of successful judgments already persisted."""
    done: set[QualityJudgmentDedupKey] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                response = QualityJudgeResponse.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_no, path, exc)
                continue
            done.add(response.dedup_key())
    return done


def run_quality_judgments(
    *,
    requests: list[QualityJudgeRequest],
    backend: QualityJudgeBackend,
    output_path: Path,
    max_concurrency: int,
    progress_every: int = 500,
) -> tuple[int, int]:
    """Execute quality judgments with idempotent resume.

    Successful responses are appended to ``output_path`` (JSONL).  Failed
    attempts are recorded as :class:`QualityJudgeErrorRecord` in a
    companion ``*.errors.jsonl`` file and are retried on each subsequent
    run until they succeed.  Mirrors
    :func:`src.services.llm_judge_service.run_judgments`.

    Returns ``(n_succeeded, n_failed)`` for the attempts made in this run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = errors_path_for(output_path)

    success_keys = _load_success_keys(output_path)
    pending = [r for r in requests if _request_key(r) not in success_keys]

    logger.info(
        "Quality-judge run: %d requests total, %d already done, %d pending",
        len(requests), len(requests) - len(pending), len(pending),
    )
    if not pending:
        return (0, 0)

    n_succeeded = 0
    n_failed = 0

    with open(output_path, "a", encoding="utf-8") as successes_f, \
         open(errors_path, "a", encoding="utf-8") as errors_f:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(_run_single, backend, req): req for req in pending}
            for i, future in enumerate(as_completed(futures), start=1):
                request = futures[future]
                try:
                    response = future.result()
                except Exception as exc:  # noqa: BLE001
                    error_record = _build_error_record(request, exc)
                    errors_f.write(error_record.model_dump_json() + "\n")
                    errors_f.flush()
                    n_failed += 1
                else:
                    successes_f.write(response.model_dump_json() + "\n")
                    successes_f.flush()
                    n_succeeded += 1

                if i % progress_every == 0 or i == len(pending):
                    logger.info(
                        "  progress: %d/%d (ok=%d, err=%d)",
                        i, len(pending), n_succeeded, n_failed,
                    )

    return n_succeeded, n_failed


def _run_single(
    backend: QualityJudgeBackend,
    request: QualityJudgeRequest,
) -> QualityJudgeResponse:
    system_text = render_system_prompt(keyword=request.target.keyword)
    user_text = render_user_prompt(idea=request.idea_text)
    return backend.judge(
        request,
        system_prompt_text=system_text,
        user_prompt_text=user_text,
    )


def _build_error_record(
    request: QualityJudgeRequest, exc: BaseException
) -> QualityJudgeErrorRecord:
    system_text = render_system_prompt(keyword=request.target.keyword)
    user_text = render_user_prompt(idea=request.idea_text)
    return QualityJudgeErrorRecord(
        request=request,
        system_prompt_text=system_text,
        user_prompt_text=user_text,
        timestamp=datetime.now(timezone.utc),
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _request_key(request: QualityJudgeRequest) -> QualityJudgmentDedupKey:
    """Dedup key mirroring :meth:`QualityJudgeResponse.dedup_key`."""
    return (
        request.target.keyword,
        request.target.effort.value,
        request.target.sample_row_index,
        request.judge_model.value,
    )
