"""Orchestration service for LLM-as-judge pairwise similarity evaluation.

Responsibilities:
  * stratified pair sampling by embedding distance per (keyword, effort)
  * prompt rendering from the LiveIdeaBench v4 fluency-critic template
  * both-orders (position-bias) request construction
  * idempotent, concurrent judgment execution with JSONL audit trail

Data-flow (one run_dir, one judge model):

    samples.jsonl ─┐
                   ├──► select_stratified_pairs ──► build_judge_requests
    embeddings.npy ┘                                       │
                                                           ▼
                                                    run_judgments ──► *.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from src.metrics import normalize_rows
from src.model_registry import EffortName, EmbeddingModelName, JudgeModelName
from src.providers.llm_judge.base import LLMJudgeBackend
from src.schemas.llm_judge import (
    JudgeErrorRecord,
    JudgeOrder,
    JudgeRequest,
    JudgeResponse,
    JudgmentDedupKey,
    PairSpec,
    PairStratum,
)
from src.schemas.sample import SampleRecord

logger = logging.getLogger(__name__)


# Source of truth for the fluency-critic prompt: the LiveIdeaBench submodule.
# We load the template at import time so any upstream update is picked up as
# soon as the submodule is refreshed; ``PROMPT_TEMPLATE_HASH`` then reflects
# the change and the manifest records exactly which version was used.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVEIDEABENCH_PROMPTS_PATH = (
    _REPO_ROOT / "external" / "liveideabench" / "utils" / "prompts.json"
)

_PLACEHOLDER_KEYWORD = "{{keyword}}"
_PLACEHOLDER_A = "{{A}}"
_PLACEHOLDER_B = "{{B}}"


def _load_fluency_critic_template() -> str:
    """Read ``fluency_critic_prompt.description`` from the submodule."""
    with open(_LIVEIDEABENCH_PROMPTS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)
    template = prompts["fluency_critic_prompt"]["description"]
    for placeholder in (_PLACEHOLDER_KEYWORD, _PLACEHOLDER_A, _PLACEHOLDER_B):
        if placeholder not in template:
            raise RuntimeError(
                f"fluency_critic_prompt from {_LIVEIDEABENCH_PROMPTS_PATH} is missing "
                f"expected placeholder {placeholder!r}; aborting so that the schema "
                "cannot silently drift from the LiveIdeaBench protocol."
            )
    return template


FLUENCY_CRITIC_TEMPLATE: str = _load_fluency_critic_template()

PROMPT_TEMPLATE_HASH: str = hashlib.sha256(
    FLUENCY_CRITIC_TEMPLATE.encode("utf-8")
).hexdigest()

# ``PROMPT_TEMPLATE_VERSION`` tags the prompt both by its LiveIdeaBench lineage
# and by a short hash of the current text, so analysis scripts can detect when
# a run was collected under a different version without needing the submodule.
PROMPT_TEMPLATE_VERSION: str = f"liveideabench_fluency_critic@{PROMPT_TEMPLATE_HASH[:12]}"


# Tolerance for the cosine-distance range check in :func:`select_stratified_pairs`.
# Cosine similarity on row-normalized float64 vectors drifts by at most a few
# ULPs, so 1e-9 leaves several orders of magnitude of headroom while still
# catching e.g. an un-normalized input whose deviation is 0.01+.
_DISTANCE_EPS: float = 1e-9


def render_prompt(*, keyword: str, idea_a: str, idea_b: str) -> str:
    """Render the fluency-critic prompt.  Inputs are inserted verbatim.

    Uses :meth:`str.replace` instead of :meth:`str.format` because the
    LiveIdeaBench template is Jinja-style (``{{var}}``) rather than Python
    format-style.  Keeping the template verbatim avoids any divergence from
    the upstream protocol.
    """
    return (
        FLUENCY_CRITIC_TEMPLATE
        .replace(_PLACEHOLDER_KEYWORD, keyword)
        .replace(_PLACEHOLDER_A, idea_a)
        .replace(_PLACEHOLDER_B, idea_b)
    )


def select_stratified_pairs(
    *,
    records: list[SampleRecord],
    embeddings: np.ndarray,
    embedding_model: EmbeddingModelName,
    top_m: int,
    bottom_m: int,
    middle_m: int = 0,
) -> list[PairSpec]:
    """Pick top-m most distant and bottom-m most similar pairs per (keyword, effort).

    Uses cosine distance on row-normalized ``embeddings``.  ``records`` and
    ``embeddings`` MUST be aligned (row i of ``embeddings`` corresponds to
    ``records[i]``); :func:`src.artifacts.load_sample_records` and the embedding
    pipeline guarantee this because both use the same canonical sort order.

    Returns a flat list of :class:`PairSpec` with canonical index ordering
    (``sample_i_index < sample_j_index``).  A pair selected in both strata is
    emitted once per stratum.
    """
    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"records/embeddings length mismatch: {len(records)} records vs {embeddings.shape[0]} rows"
        )
    if top_m < 0 or bottom_m < 0 or middle_m < 0:
        raise ValueError(
            f"top_m/middle_m/bottom_m must be non-negative (got {top_m}, {middle_m}, {bottom_m})"
        )

    normalized = normalize_rows(np.asarray(embeddings, dtype=np.float64))

    groups: dict[tuple[str, EffortName], list[int]] = {}
    for idx, record in enumerate(records):
        groups.setdefault((record.keyword, record.effort), []).append(idx)

    pairs: list[PairSpec] = []
    for (keyword, effort), member_indices in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1].value)):
        if len(member_indices) < 2:
            logger.warning(
                "Skipping (keyword=%r, effort=%r): only %d samples (need >=2)",
                keyword, effort.value, len(member_indices),
            )
            continue

        idx_arr = np.asarray(member_indices, dtype=np.int64)
        mat = normalized[idx_arr]
        similarity = mat @ mat.T
        tri_i, tri_j = np.triu_indices(len(idx_arr), k=1)
        raw_distances = 1.0 - similarity[tri_i, tri_j]
        # Normalized cosine similarity can exceed ±1 by O(1e-16), pushing the
        # distance O(1e-16) outside [0, 2].  Anything beyond _DISTANCE_EPS is
        # not float noise and indicates a bug (non-normalized inputs, NaN
        # embeddings, etc.); raise so the bad value never reaches the schema.
        if np.any((raw_distances < -_DISTANCE_EPS) | (raw_distances > 2.0 + _DISTANCE_EPS)):
            raise RuntimeError(
                "Cosine distance out of expected [0, 2] range for "
                f"(keyword={keyword!r}, effort={effort.value!r}): "
                f"min={float(raw_distances.min())!r}, max={float(raw_distances.max())!r}. "
                "This usually means the embeddings were not row-normalized or contain NaN."
            )
        distances = np.clip(raw_distances, 0.0, 2.0)

        order_desc = np.argsort(-distances)  # most distant first
        order_asc = np.argsort(distances)    # most similar first

        pairs.extend(
            _emit_pairs_for_ranks(
                ranks=order_desc[:top_m],
                tri_i=tri_i,
                tri_j=tri_j,
                distances=distances,
                idx_arr=idx_arr,
                keyword=keyword,
                effort=effort,
                embedding_model=embedding_model,
                stratum=PairStratum.TOP,
            )
        )
        if middle_m > 0:
            # Middle stratum: middle_m pairs centered on the median rank of
            # the sorted-ascending distance array.  ``rank`` in the emitted
            # PairSpec reflects offset from the lowest selected position so
            # rank=1 is the smallest-distance member of the middle window.
            n_pairs = len(order_asc)
            if middle_m > n_pairs:
                raise ValueError(
                    f"middle_m={middle_m} exceeds number of pairs {n_pairs} for "
                    f"(keyword={keyword!r}, effort={effort.value!r})"
                )
            start = max(0, (n_pairs - middle_m) // 2)
            middle_ranks = order_asc[start : start + middle_m]
            pairs.extend(
                _emit_pairs_for_ranks(
                    ranks=middle_ranks,
                    tri_i=tri_i,
                    tri_j=tri_j,
                    distances=distances,
                    idx_arr=idx_arr,
                    keyword=keyword,
                    effort=effort,
                    embedding_model=embedding_model,
                    stratum=PairStratum.MIDDLE,
                )
            )
        pairs.extend(
            _emit_pairs_for_ranks(
                ranks=order_asc[:bottom_m],
                tri_i=tri_i,
                tri_j=tri_j,
                distances=distances,
                idx_arr=idx_arr,
                keyword=keyword,
                effort=effort,
                embedding_model=embedding_model,
                stratum=PairStratum.BOTTOM,
            )
        )
    return pairs


def _emit_pairs_for_ranks(
    *,
    ranks: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    distances: np.ndarray,
    idx_arr: np.ndarray,
    keyword: str,
    effort: EffortName,
    embedding_model: EmbeddingModelName,
    stratum: PairStratum,
) -> list[PairSpec]:
    emitted: list[PairSpec] = []
    for rank_idx, pair_pos in enumerate(ranks, start=1):
        local_i = int(tri_i[pair_pos])
        local_j = int(tri_j[pair_pos])
        global_i = int(idx_arr[local_i])
        global_j = int(idx_arr[local_j])
        if global_i > global_j:
            global_i, global_j = global_j, global_i
        emitted.append(
            PairSpec(
                keyword=keyword,
                effort=effort,
                sample_i_index=global_i,
                sample_j_index=global_j,
                embedding_distance=float(distances[pair_pos]),
                embedding_model=embedding_model,
                stratum=stratum,
                rank=rank_idx,
            )
        )
    return emitted


def build_judge_requests(
    *,
    pairs: Iterable[PairSpec],
    records: list[SampleRecord],
    judge_model: JudgeModelName,
    both_orders: bool,
) -> list[JudgeRequest]:
    """Expand each :class:`PairSpec` into one or two :class:`JudgeRequest`."""
    requests: list[JudgeRequest] = []
    for pair in pairs:
        idea_i = records[pair.sample_i_index].idea
        idea_j = records[pair.sample_j_index].idea
        requests.append(
            JudgeRequest(
                pair=pair,
                order=JudgeOrder.AB,
                judge_model=judge_model,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                idea_text_a=idea_i,
                idea_text_b=idea_j,
            )
        )
        if both_orders:
            requests.append(
                JudgeRequest(
                    pair=pair,
                    order=JudgeOrder.BA,
                    judge_model=judge_model,
                    prompt_template_version=PROMPT_TEMPLATE_VERSION,
                    idea_text_a=idea_j,
                    idea_text_b=idea_i,
                )
            )
    return requests


def errors_path_for(success_path: Path) -> Path:
    """Companion ``*.errors.jsonl`` path next to the main successes file."""
    return success_path.with_suffix(".errors.jsonl")


def _load_success_keys(path: Path) -> set[JudgmentDedupKey]:
    """Return dedup keys of successful judgments already persisted."""
    done: set[JudgmentDedupKey] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                response = JudgeResponse.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_no, path, exc)
                continue
            done.add(response.dedup_key())
    return done


def run_judgments(
    *,
    requests: list[JudgeRequest],
    backend: LLMJudgeBackend,
    output_path: Path,
    max_concurrency: int,
    progress_every: int = 50,
) -> tuple[int, int]:
    """Execute judgments with idempotent resume.

    Successful responses are appended to ``output_path`` (JSONL).  Failed
    attempts are recorded as :class:`JudgeErrorRecord` in a companion
    ``*.errors.jsonl`` file and are retried on each subsequent run until they
    succeed.

    Returns ``(n_succeeded, n_failed)`` for the attempts made in this run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = errors_path_for(output_path)

    success_keys = _load_success_keys(output_path)
    pending = [r for r in requests if _request_key(r, backend.model_name) not in success_keys]

    logger.info(
        "LLM-judge run: %d requests total, %d already done, %d pending",
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


def _run_single(backend: LLMJudgeBackend, request: JudgeRequest) -> JudgeResponse:
    prompt_text = render_prompt(
        keyword=request.pair.keyword,
        idea_a=request.idea_text_a,
        idea_b=request.idea_text_b,
    )
    return backend.judge(request, prompt_text)


def _build_error_record(request: JudgeRequest, exc: BaseException) -> JudgeErrorRecord:
    prompt_text = render_prompt(
        keyword=request.pair.keyword,
        idea_a=request.idea_text_a,
        idea_b=request.idea_text_b,
    )
    return JudgeErrorRecord(
        request=request,
        prompt_text=prompt_text,
        timestamp=datetime.now(timezone.utc),
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _request_key(request: JudgeRequest, judge_model: JudgeModelName) -> JudgmentDedupKey:
    """Dedup key mirroring :meth:`JudgeResponse.dedup_key`."""
    return (
        request.pair.keyword,
        request.pair.effort.value,
        request.pair.sample_i_index,
        request.pair.sample_j_index,
        request.order.value,
        judge_model.value,
    )
