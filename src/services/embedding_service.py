"""Embedding service: backend factory and cached record-embedding orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.artifacts import (
    EmbeddingArtifactWriter,
    copy_cached_embedding_artifacts_to_writer,
    load_published_embedding_matrix,
)
from src.model_registry import (
    DEFAULT_SPECTER2_BATCH_SIZE,
    EMBEDDING_MODEL_PROVIDER,
    AwsRegion,
    EmbeddingModelName,
    ProviderName,
)
from src.providers.embedding.base import EmbeddingBackend
from src.providers.embedding.bedrock import BedrockEmbeddingBackend
from src.providers.embedding.openai import OpenAIEmbeddingBackend
from src.providers.embedding.specter2 import Specter2EmbeddingBackend
from src.schemas.sample import SampleRecord


def resolve_embed_provider(explicit_provider: str, model: EmbeddingModelName) -> ProviderName:
    """Return the embedding provider for *model*.

    If ``explicit_provider`` is not ``'auto'`` it is coerced to
    :class:`ProviderName` directly.  Otherwise the provider is looked up from
    the ``EMBEDDING_MODEL_PROVIDER`` mapping — an exact match is required.
    """
    if explicit_provider != "auto":
        return ProviderName(explicit_provider)
    try:
        return EMBEDDING_MODEL_PROVIDER[model]
    except KeyError:
        raise ValueError(
            f"No provider registered for embedding model {model.value!r}. "
            f"Add an entry to EMBEDDING_MODEL_PROVIDER in model_registry.py "
            f"or pass --embed-provider explicitly."
        )


def create_embedding_backend(
    *,
    embedding_model: EmbeddingModelName,
    provider: ProviderName,
    region: Optional[str] = None,
    specter2_batch_size: int = DEFAULT_SPECTER2_BATCH_SIZE,
) -> EmbeddingBackend:
    """Construct the appropriate :class:`EmbeddingBackend` for *embedding_model* / *provider*.

    For Bedrock, *region* is resolved from the argument first, then from the
    ``BEDROCK_REGION``, ``AWS_REGION``, and ``AWS_DEFAULT_REGION`` environment
    variables.  A missing or unrecognised region raises :exc:`EnvironmentError` /
    :exc:`ValueError` respectively.
    """
    if provider == ProviderName.BEDROCK:
        raw_region = (
            region
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )
        if not raw_region:
            raise EnvironmentError(
                "A region is required for Bedrock embeddings. "
                "Set BEDROCK_REGION, AWS_REGION, or AWS_DEFAULT_REGION, "
                "or pass region= explicitly."
            )
        try:
            resolved_region = AwsRegion(raw_region)
        except ValueError:
            raise ValueError(
                f"Unknown AWS region: {raw_region!r}. "
                "Add it to AwsRegion in diversity/model_registry.py."
            )
        return BedrockEmbeddingBackend(embedding_model=embedding_model, region=resolved_region)

    if provider == ProviderName.OPENAI:
        return OpenAIEmbeddingBackend(embedding_model=embedding_model)

    if provider == ProviderName.SPECTER2:
        return Specter2EmbeddingBackend(
            adapter_model_id=embedding_model,
            base_model_id=EmbeddingModelName.SPECTER2_BASE,
            batch_size=specter2_batch_size,
        )

    raise ValueError(f"Unsupported embed provider: {provider!r}")


def embed_records(
    *,
    records: list[SampleRecord],
    embedding_model: EmbeddingModelName,
    provider: ProviderName,
    output_dir: Path,
    embed_max_concurrency: int,
    region: Optional[str] = None,
    specter2_batch_size: int = DEFAULT_SPECTER2_BATCH_SIZE,
) -> np.ndarray:
    """Embed *records* with cache support, writing artifacts to *output_dir*.

    Existing ``embeddings.npy`` / ``embedding_responses.jsonl`` artifacts in
    *output_dir* are reused if they match *embedding_model*; only the
    missing rows are forwarded to the backend.

    A published ``embeddings.npy`` without ``embedding_responses.jsonl`` (the
    Hugging Face dataset layout) is reused as-is when its rows line up with
    *records*; see :func:`src.artifacts.load_published_embedding_matrix`.

    Returns the final ``(n_records, dim)`` embedding matrix loaded from disk.
    """
    published = load_published_embedding_matrix(output_dir=output_dir, records=records)
    if published is not None:
        return published
    backend = create_embedding_backend(
        embedding_model=embedding_model,
        provider=provider,
        region=region,
        specter2_batch_size=specter2_batch_size,
    )
    writer = EmbeddingArtifactWriter(
        embeddings_path=output_dir / "embeddings.npy",
        responses_path=output_dir / "embedding_responses.jsonl",
        records=records,
        embedding_model=embedding_model,
    )
    try:
        reused_indices = copy_cached_embedding_artifacts_to_writer(
            output_dir=output_dir,
            embedding_model=embedding_model,
            records=records,
            writer=writer,
        )
        # Add slots already populated by writer's partial-resume.
        reused_indices.update(writer.written_indices())
        missing_indices = [i for i in range(len(records)) if i not in reused_indices]
        missing_records = [records[i] for i in missing_indices]

        print(
            f"[embed-cache] model={embedding_model.value} "
            f"reused={len(reused_indices)} missing={len(missing_records)}",
            flush=True,
        )

        if missing_records:
            texts = backend.extract_texts(missing_records)
            backend.embed_texts_streaming(
                texts,
                writer=writer,
                result_indices=missing_indices,
                max_concurrency=embed_max_concurrency,
            )

        return writer.finalize()
    except Exception:
        writer.abort()
        raise
