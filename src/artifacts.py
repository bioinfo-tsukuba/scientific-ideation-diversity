"""Artifact I/O helpers for effort-diversity runs."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import ValidationError

from .model_registry import EffortName, EmbeddingModelName, effort_sort_key
from .schemas.embedding import (
    EmbeddingArtifactRow,
    EmbeddingPayload,
    PcaCoordinateRow,
)
from .schemas.sample import SampleRecord

GENERATION_ARTIFACT_FILENAMES = ["errors.jsonl", "backfill_successes.jsonl"]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def allocate_output_dir(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def sort_sample_records(records: list[SampleRecord]) -> list[SampleRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.category or "",
            record.keyword,
            record.prompt_style,
            effort_sort_key(record.effort),
            record.sample_index,
        ),
    )


def load_sample_records(path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            records.append(SampleRecord.model_validate(payload))
    return sort_sample_records(records)


def load_run_sample_records(run_dir: Path) -> list[SampleRecord]:
    """Load the canonical merged record list for a run directory.

    Folds ``samples.jsonl`` ∪ ``backfill_successes.jsonl`` and dedupes by
    sample-task key. This matches what ``run_prompt_sensitivity.py`` (and
    the embedding pipeline downstream of it) sees, so the returned indices
    align 1:1 with ``<embedding_subdir>/embeddings.npy``. For runs without
    a backfill file (e.g. phase2), this is equivalent to
    :func:`load_sample_records` on the top-level ``samples.jsonl``.
    """
    samples_path = run_dir / "samples.jsonl"
    records = load_sample_records(samples_path)
    backfill_path = run_dir / "backfill_successes.jsonl"
    if not backfill_path.exists():
        return records
    merged = sort_sample_records(records + load_sample_records(backfill_path))
    seen: set[tuple] = set()
    deduped: list[SampleRecord] = []
    for r in merged:
        key = (r.keyword, r.prompt_style, r.effort, r.sample_index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def write_sample_records(path: Path, records: list[SampleRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clone_records_for_embedding(
    records: list[SampleRecord],
    *,
    embedding_model: EmbeddingModelName,
) -> list[SampleRecord]:
    return [record.for_embedding_model(embedding_model) for record in records]


def materialize_generation_artifacts(
    *,
    source_dir: Path,
    target_dir: Path,
    records: list[SampleRecord],
) -> None:
    if source_dir == target_dir:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    write_sample_records(target_dir / "samples.jsonl", records)
    for filename in GENERATION_ARTIFACT_FILENAMES:
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_dir / filename)


def build_embedding_response_row(
    *,
    record: SampleRecord,
    embedding_model: EmbeddingModelName,
    embedding_row_index: int,
    payload: EmbeddingPayload,
) -> dict[str, Any]:
    return EmbeddingArtifactRow(
        keyword=record.keyword,
        category=record.category,
        prompt_style=record.prompt_style,
        effort=record.effort,
        sample_index=record.sample_index,
        embedding_model=embedding_model,
        embedding_row_index=embedding_row_index,
        input_text_token_count=payload.input_text_token_count,
        raw_api_response=payload.raw_api_response,
    ).model_dump(mode="json")


def _dedup_responses_partial(path: Path) -> int:
    """Rewrite the partial responses jsonl keeping the last row per index.

    When ``_resume_from_partials`` reopens an existing partial jsonl in
    append mode and the crashed writer had already recorded a row for an
    index that the current writer will also fill, the finalised file ends
    up with duplicates.  ``finalize`` calls this helper so the post-
    condition "one row per index, in index order" is restored before the
    atomic ``os.replace`` to the final path.

    Unparseable rows raise — by the time finalize runs, ``_resume_from_partials``
    has already validated every pre-existing row, and every append the
    current writer emits comes straight from ``build_embedding_response_row``,
    so seeing a malformed line here is a real bug rather than the benign
    crash-truncation case.

    Returns the number of duplicate rows dropped.
    """
    rows_by_index: dict[int, str] = {}
    total_read = 0
    with path.open(encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_read += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"unparseable row at line {lineno} in {path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            idx = row.get("embedding_row_index")
            if not isinstance(idx, int):
                raise RuntimeError(
                    f"row at line {lineno} in {path} is missing a valid "
                    f"embedding_row_index field; got {idx!r}"
                )
            # Last occurrence wins; this matches the memmap value at that index
            # (the writer's ``_embeddings[index] = vector_array`` assignment is
            # similarly last-write-wins).
            rows_by_index[idx] = line
    if total_read == len(rows_by_index):
        return 0
    tmp = path.with_suffix(path.suffix + ".dedup-tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for idx in sorted(rows_by_index):
            f.write(rows_by_index[idx] + "\n")
    os.replace(tmp, path)
    return total_read - len(rows_by_index)


class EmbeddingArtifactWriter:
    def __init__(
        self,
        *,
        embeddings_path: Path,
        responses_path: Path,
        records: list[SampleRecord],
        embedding_model: EmbeddingModelName,
    ) -> None:
        self.embeddings_path = embeddings_path
        self.responses_path = responses_path
        self.temp_embeddings_path = embeddings_path.with_name(f"{embeddings_path.name}.partial")
        self.temp_responses_path = responses_path.with_name(f"{responses_path.name}.partial")
        self.records = records
        self.embedding_model = embedding_model
        # The embedding dimension is unknown until the first vector is written.
        self._embeddings: Optional[np.memmap] = None
        self._written = [False] * len(records)
        self._count = 0

        # Resume from a prior crashed run if its .partial files are consistent
        # with the current records.  ``resumed`` toggles the responses-file
        # open mode: "a" appends to the preserved partial, "w" starts fresh.
        resumed = self._resume_from_partials()
        responses_open_mode = "a" if resumed else "w"
        self._responses_file = open(
            self.temp_responses_path, responses_open_mode, encoding="utf-8"
        )

    def _resume_from_partials(self) -> bool:
        """Re-seed writer state from .partial files preserved by a prior run.

        A run may terminate part-way through a component (e.g. Bedrock
        ModelErrorException after tens of thousands of successful calls).
        ``abort`` keeps the .partial files on disk, and this method rebuilds
        ``_written`` / ``_count`` / ``_embeddings`` so the next call to
        ``embed_field`` can re-run only the records that are still missing.
        """
        if not (self.temp_embeddings_path.exists() and self.temp_responses_path.exists()):
            return False
        try:
            existing = np.load(self.temp_embeddings_path, mmap_mode="r+")
        except (OSError, ValueError):
            return False
        if existing.shape[0] != len(self.records):
            return False

        index_by_key = {_record_task_key(r): i for i, r in enumerate(self.records)}
        with open(self.temp_responses_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                # Any unparseable line in .partial.jsonl is treated as a hard
                # error rather than silently skipped.  A truncated last line
                # from a SIGKILL mid-``writer.write`` is the benign case, but
                # the same signature could also be a schema drift (older
                # EmbeddingArtifactRow shape) or disk corruption.  Swallowing
                # those would let wrong data propagate to the embeddings
                # matrix, so surface them here with enough context for the
                # operator to decide (repair vs. drop the partial and rerun).
                try:
                    row = EmbeddingArtifactRow.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise RuntimeError(
                        f"unparseable row at line {lineno} in "
                        f"{self.temp_responses_path}: {type(exc).__name__}: {exc}.  "
                        f"If this is a truncated trailing line from a crashed "
                        f"writer, trim that line manually or delete the partial "
                        f"pair (.jsonl.partial + .npy.partial) to force a fresh "
                        f"recompute."
                    ) from exc
                if row.embedding_model != self.embedding_model:
                    continue
                key = _sample_task_key(
                    keyword=row.keyword,
                    category=row.category,
                    effort=row.effort,
                    sample_index=row.sample_index,
                )
                idx = index_by_key.get(key)
                if idx is None or self._written[idx]:
                    continue
                self._written[idx] = True
                self._count += 1
        self._embeddings = existing
        return True

    def written_indices(self) -> set[int]:
        return {i for i, done in enumerate(self._written) if done}

    def ensure_matrix(self, *, dim: int) -> None:
        if self._embeddings is None:
            self._embeddings = np.lib.format.open_memmap(
                self.temp_embeddings_path,
                mode="w+",
                dtype=np.float64,
                shape=(len(self.records), dim),
            )

    def write(
        self,
        *,
        index: int,
        vector: np.ndarray,
        payload: EmbeddingPayload,
    ) -> None:
        vector_array = np.asarray(vector, dtype=np.float64)
        if vector_array.ndim != 1:
            raise ValueError(f"Expected 1-D embedding vector at index={index}, got shape={vector_array.shape}")
        self.ensure_matrix(dim=int(vector_array.shape[0]))
        if self._embeddings is None:
            raise RuntimeError("Embedding matrix was not initialized")
        self._embeddings[index] = vector_array
        self.records[index].embedding_input_tokens = payload.input_text_token_count
        row = build_embedding_response_row(
            record=self.records[index],
            embedding_model=self.embedding_model,
            embedding_row_index=index,
            payload=payload,
        )
        self._responses_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._written[index] = True
        self._count += 1

    def finalize(self) -> np.ndarray:
        if self._count != len(self.records):
            missing = len(self.records) - self._count
            raise RuntimeError(
                f"Embedding artifact writer incomplete for model={self.embedding_model}: missing={missing}"
            )
        if not all(self._written):
            raise RuntimeError(f"Embedding artifact writer has unwritten slots for model={self.embedding_model}")
        if self._embeddings is None:
            raise RuntimeError(f"No embeddings were written for model={self.embedding_model}")
        self._embeddings.flush()
        self._responses_file.flush()
        self._responses_file.close()
        del self._embeddings
        # Resume-in-append-mode (_resume_from_partials) can leave duplicate rows
        # in the partial responses file when a prior crashed run had written
        # rows whose ``embedding_row_index`` the current run also needs to fill.
        # The memmap value at each index is authoritative (last write wins), so
        # collapse the jsonl to one row per index, keeping the last occurrence,
        # to preserve the (N rows in jsonl) == (N vectors in npy) invariant
        # that ``copy_cached_embedding_artifacts_to_writer`` enforces downstream.
        _dedup_responses_partial(self.temp_responses_path)
        os.replace(self.temp_embeddings_path, self.embeddings_path)
        os.replace(self.temp_responses_path, self.responses_path)
        return np.load(self.embeddings_path, mmap_mode="r")

    def abort(self) -> None:
        """Close handles without deleting .partial files.

        Preserving the .partial files lets the next ``EmbeddingArtifactWriter``
        for the same component resume from them via ``_resume_from_partials``.
        Previously, a single transient provider error (e.g. Bedrock
        ``ModelErrorException``) mid-component discarded every successful
        embedding accumulated in the current pass.
        """
        try:
            self._responses_file.close()
        except Exception:
            pass


def _sample_task_key(
    *,
    keyword: str,
    category: str,
    effort: EffortName,
    sample_index: int,
) -> tuple[str, str, EffortName, int]:
    return (category, keyword, effort, sample_index)


def _record_task_key(record: SampleRecord) -> tuple[str, str, EffortName, int]:
    return _sample_task_key(
        keyword=record.keyword,
        category=record.category,
        effort=record.effort,
        sample_index=record.sample_index,
    )


def copy_cached_embedding_artifacts_to_writer(
    *,
    output_dir: Path,
    embedding_model: EmbeddingModelName,
    records: list[SampleRecord],
    writer: EmbeddingArtifactWriter,
    embeddings_filename: str = "embeddings.npy",
    responses_filename: str = "embedding_responses.jsonl",
) -> set[int]:
    embeddings_path = output_dir / embeddings_filename
    responses_path = output_dir / responses_filename
    if not embeddings_path.exists() or not responses_path.exists():
        return set()

    embeddings = np.load(embeddings_path, mmap_mode="r")
    index_by_key = {_record_task_key(record): index for index, record in enumerate(records)}
    copied_indices: set[int] = set()
    response_row_count = 0
    with open(responses_path, encoding="utf-8") as f:
        for response_row_count, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = EmbeddingArtifactRow.model_validate(json.loads(line))
            if row.embedding_model != embedding_model:
                continue
            key = _sample_task_key(
                keyword=row.keyword,
                category=row.category,
                effort=row.effort,
                sample_index=row.sample_index,
            )
            record_index = index_by_key.get(key)
            if record_index is None or record_index in copied_indices:
                continue
            writer.write(
                index=record_index,
                vector=embeddings[row.embedding_row_index],
                payload=EmbeddingPayload(
                    raw_api_response=row.raw_api_response,
                    input_text_token_count=row.input_text_token_count,
                ),
            )
            copied_indices.add(record_index)

    if response_row_count != len(embeddings):
        raise ValueError(
            f"Cached embedding artifact mismatch in {output_dir}: "
            f"{response_row_count} response rows vs {len(embeddings)} vectors"
        )

    return copied_indices


def write_matrix_csv(path: Path, header: list[str], rows: list[list[str | float]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_embedding_responses(
    *,
    path: Path,
    records: list[SampleRecord],
    embedding_payloads: list[EmbeddingPayload],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for embedding_row_index, (record, payload) in enumerate(
            zip(records, embedding_payloads, strict=True)
        ):
            row = build_embedding_response_row(
                record=record,
                embedding_model=record.embedding_model,
                embedding_row_index=embedding_row_index,
                payload=payload,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hydrate_cached_embedding_artifacts(
    *,
    output_dir: Path,
    embedding_model: EmbeddingModelName,
    records: list[SampleRecord],
    vectors: list[Optional[np.ndarray]],
    payloads: list[Optional[EmbeddingPayload]],
) -> int:
    embeddings_path = output_dir / "embeddings.npy"
    responses_path = output_dir / "embedding_responses.jsonl"
    if not embeddings_path.exists() or not responses_path.exists():
        return 0

    embeddings = np.load(embeddings_path, mmap_mode="r")
    index_by_key = {_record_task_key(record): index for index, record in enumerate(records)}
    reused = 0
    response_row_count = 0
    with open(responses_path, encoding="utf-8") as f:
        for response_row_count, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("embedding_model") != embedding_model:
                continue
            key = _sample_task_key(
                keyword=row["keyword"],
                category=row["category"],
                effort=EffortName(row["effort"]),
                sample_index=row["sample_index"],
            )
            record_index = index_by_key.get(key)
            if record_index is None:
                continue
            embedding_row_index = row.get("embedding_row_index", response_row_count - 1)
            vectors[record_index] = np.asarray(embeddings[embedding_row_index], dtype=float)
            payloads[record_index] = EmbeddingPayload(
                raw_api_response=row["raw_api_response"],
                input_text_token_count=row["input_text_token_count"],
            )
            reused += 1

    if response_row_count != len(embeddings):
        raise ValueError(
            f"Cached embedding artifact mismatch in {output_dir}: "
            f"{response_row_count} response rows vs {len(embeddings)} vectors"
        )

    return reused


def save_pca_coordinates(
    *,
    path: Path,
    records: list[SampleRecord],
    coords: np.ndarray,
) -> None:
    rows = [
        PcaCoordinateRow(
            keyword=record.keyword,
            category=record.category,
            prompt_style=record.prompt_style,
            effort=record.effort,
            sample_index=record.sample_index,
            x=float(coords[index, 0]),
            y=float(coords[index, 1]),
            word_count=record.word_count,
            output_tokens=record.output_tokens,
            stop_reason=record.stop_reason,
            has_reasoning_block=int(record.has_reasoning_block),
            idea=record.idea,
        ).model_dump(mode="json")
        for index, record in enumerate(records)
    ]
    write_csv(path, rows)
