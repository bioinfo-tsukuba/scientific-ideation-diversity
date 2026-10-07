"""OpenAI embedding backend using the official SDK."""

from __future__ import annotations

import concurrent.futures
from typing import Optional

import numpy as np
import openai
import tiktoken

from src.artifacts import EmbeddingArtifactWriter
from src.model_registry import EmbeddingModelName
from src.schemas.embedding import EmbeddingPayload, EmbeddingResult
from src.schemas.sample import SampleRecord
from src.text import normalize_embedding_text

from .base import EmbeddingBackend

# OpenAI embeddings accept up to 2048 inputs / 300k tokens per request.
# 128 is a conservative batch size that keeps per-request token counts well
# under the TPM ceiling while amortising HTTP/TLS overhead effectively.
_DEFAULT_BATCH_SIZE = 128


def _embed_batch(
    texts: list[str],
    indices: list[int],
    *,
    embedding_model: EmbeddingModelName,
    client: openai.OpenAI,
    encoder: tiktoken.Encoding,
) -> list[tuple[int, EmbeddingResult]]:
    """Embed a list of texts in one API call and return per-text results.

    ``tiktoken`` tokenises each input locally so ``input_text_token_count``
    stays per-text (the API's ``usage.prompt_tokens`` aggregates the batch).
    """
    response = client.embeddings.create(
        model=embedding_model.value,
        input=texts,
        encoding_format="float",
    )
    assert len(response.data) == len(texts)
    per_text_tokens = [len(encoder.encode(t)) for t in texts]
    usage = response.usage

    results: list[tuple[int, EmbeddingResult]] = []
    for j, data in enumerate(response.data):
        vector = np.asarray(data.embedding, dtype=np.float64)
        results.append((
            indices[j],
            EmbeddingResult(
                vector=vector,
                raw_api_response={
                    "object": data.object,
                    "model": response.model,
                    "batch_index": j,
                    "batch_size": len(texts),
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "embedding_dimensions": int(vector.shape[0]),
                },
                input_text_token_count=per_text_tokens[j],
            ),
        ))
    return results


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend using the official ``openai`` Python SDK.

    Sends up to :data:`_DEFAULT_BATCH_SIZE` texts per request — the API is
    specified to return per-input embeddings independently, so the resulting
    vectors are identical to sending one text at a time, but RPM usage drops
    by the batch factor and effective throughput scales accordingly.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModelName,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        # Created on first use, so resume runs that only reuse cached /
        # published embeddings never need OPENAI_API_KEY.
        self._client: Optional[openai.OpenAI] = None
        try:
            self._encoder = tiktoken.encoding_for_model(embedding_model.value)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(max_retries=5)
        return self._client

    def embed_records_streaming(
        self,
        records: list[SampleRecord],
        *,
        writer: EmbeddingArtifactWriter,
        result_indices: Optional[list[int]] = None,
        max_concurrency: int,
    ) -> None:
        texts = [normalize_embedding_text(record.idea) for record in records]
        self.embed_texts_streaming(
            texts, writer=writer, result_indices=result_indices, max_concurrency=max_concurrency,
        )

    def embed_texts_streaming(
        self,
        texts: list[str],
        *,
        writer: EmbeddingArtifactWriter,
        result_indices: Optional[list[int]] = None,
        max_concurrency: int,
    ) -> None:
        if not texts:
            return
        if result_indices is None:
            result_indices = list(range(len(texts)))
        if len(result_indices) != len(texts):
            raise ValueError("result_indices must have the same length as texts")

        batches: list[tuple[list[str], list[int]]] = []
        for start in range(0, len(texts), self.batch_size):
            batches.append((
                texts[start : start + self.batch_size],
                result_indices[start : start + self.batch_size],
            ))

        max_workers = max(1, min(max_concurrency, len(batches)))
        completed = 0
        total = len(texts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(
                    _embed_batch,
                    batch_texts,
                    batch_indices,
                    embedding_model=self.embedding_model,
                    client=self.client,
                    encoder=self._encoder,
                ): (batch_texts, batch_indices)
                for batch_texts, batch_indices in batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                for global_index, result in future.result():
                    writer.write(
                        index=global_index,
                        vector=result.vector,
                        payload=EmbeddingPayload(
                            raw_api_response=result.raw_api_response,
                            input_text_token_count=result.input_text_token_count,
                        ),
                    )
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"[embed] completed {completed}/{total}", flush=True)
