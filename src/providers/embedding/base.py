"""Abstract base for embedding backends and shared threadpool helper."""

from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

from src.artifacts import EmbeddingArtifactWriter
from src.schemas.embedding import EmbeddingPayload, EmbeddingResult
from src.schemas.sample import SampleRecord


class EmbeddingBackend(ABC):
    """Base class for all embedding provider backends.

    Concrete backends implement two streaming methods:

    * ``embed_records_streaming`` — high-level entry point used by the
      service layer.  Extracts texts from sample records (provider-specific
      logic) and embeds them.
    * ``embed_texts_streaming`` — low-level entry point that embeds
      pre-extracted texts.  Used by analysis scripts that apply their own
      text extraction (e.g. component-level reports).
    """

    @abstractmethod
    def embed_records_streaming(
        self,
        records: list[SampleRecord],
        *,
        writer: EmbeddingArtifactWriter,
        result_indices: Optional[list[int]] = None,
        max_concurrency: int,
    ) -> None:
        """Extract texts from *records*, embed them, and stream to *writer*.

        Each backend handles its own text assembly (e.g. SPECTER2 combines
        background + sep_token + idea, OpenAI normalises whitespace, etc.)
        and delegates to ``embed_texts_streaming``.
        """

    @abstractmethod
    def embed_texts_streaming(
        self,
        texts: list[str],
        *,
        writer: EmbeddingArtifactWriter,
        result_indices: Optional[list[int]] = None,
        max_concurrency: int,
    ) -> None:
        """Embed pre-extracted *texts* and stream results to *writer*."""

def stream_embed_texts_with_threadpool(
    texts: list[str],
    *,
    worker: Callable[[str], EmbeddingResult],
    writer: EmbeddingArtifactWriter,
    result_indices: Optional[list[int]] = None,
    max_concurrency: int,
) -> None:
    """Embed texts in parallel via a thread pool, streaming results to *writer*.

    Args:
        texts: Texts to embed.
        worker: Callable that takes a single text string and returns an
            :class:`EmbeddingResult`.
        writer: Artifact writer that receives each vector and payload.
        result_indices: Global row indices for the writer.  Defaults to
            ``range(len(texts))``.
        max_concurrency: Maximum number of concurrent worker threads.
    """
    if not texts:
        return

    if result_indices is None:
        result_indices = list(range(len(texts)))
    if len(result_indices) != len(texts):
        raise ValueError("result_indices must have the same length as texts")

    max_workers = max(1, min(max_concurrency, len(texts)))

    def _submit(
        executor: concurrent.futures.ThreadPoolExecutor,
        local_index: int,
    ) -> concurrent.futures.Future[EmbeddingResult]:
        return executor.submit(worker, texts[local_index])

    next_local_index = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_local: dict[concurrent.futures.Future[EmbeddingResult], int] = {}
        for _ in range(max_workers):
            future_to_local[_submit(executor, next_local_index)] = next_local_index
            next_local_index += 1

        completed = 0
        while future_to_local:
            done, _ = concurrent.futures.wait(
                future_to_local,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                local_index = future_to_local.pop(future)
                result = future.result()
                writer.write(
                    index=result_indices[local_index],
                    vector=result.vector,
                    payload=EmbeddingPayload(
                        raw_api_response=result.raw_api_response,
                        input_text_token_count=result.input_text_token_count,
                    ),
                )
                completed += 1
                if completed % 25 == 0 or completed == len(texts):
                    print(f"[embed] completed {completed}/{len(texts)}", flush=True)
                if next_local_index >= len(texts):
                    continue
                future_to_local[_submit(executor, next_local_index)] = next_local_index
                next_local_index += 1
