"""AWS Bedrock embedding backend."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Optional, TypedDict

import boto3
import numpy as np
from botocore.client import BaseClient
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import BotoCoreError, ClientError

from src.artifacts import EmbeddingArtifactWriter
from src.model_registry import AwsRegion, EmbeddingModelName
from src.schemas.embedding import EmbeddingResult
from src.schemas.sample import SampleRecord

from .base import EmbeddingBackend, stream_embed_texts_with_threadpool
from .errors import BEDROCK_RETRYABLE_CLIENT_ERROR_CODES, BOTOCORE_RETRYABLE_ERROR_NAMES

_DEFAULT_RETRY_ATTEMPTS = 12
_MAX_POOL_CONNECTIONS = int(os.getenv("BEDROCK_MAX_POOL_CONNECTIONS", "256"))

_clients: dict[AwsRegion, BaseClient] = {}
_clients_lock = threading.Lock()


class BedrockEmbeddingPayload(TypedDict, total=False):
    """Shape of the JSON body returned by the Bedrock invoke-model API."""

    embedding: list[float]
    embeddingsByType: dict[str, list[float] | list[list[float]]]
    inputTextTokenCount: int


def _get_client(region: AwsRegion) -> BaseClient:
    with _clients_lock:
        client = _clients.get(region)
        if client is None:
            client = boto3.client(
                "bedrock-runtime",
                region_name=region.value,
                config=BotocoreConfig(max_pool_connections=_MAX_POOL_CONNECTIONS),
            )
            _clients[region] = client
    return client


def _extract_embedding_vector(payload: BedrockEmbeddingPayload) -> np.ndarray:
    if isinstance(payload.get("embedding"), list):
        return np.asarray(payload["embedding"], dtype=np.float64)

    embeddings_by_type = payload.get("embeddingsByType")
    if isinstance(embeddings_by_type, dict):
        float_values = embeddings_by_type.get("float")
        if isinstance(float_values, list):
            if float_values and isinstance(float_values[0], list):
                return np.asarray(float_values[0], dtype=np.float64)
            return np.asarray(float_values, dtype=np.float64)

    raise ValueError(f"Unexpected Bedrock embedding payload shape: {payload}")


def _embed_one_text(
    text: str,
    *,
    embedding_model: EmbeddingModelName,
    region: AwsRegion,
    retry_attempts: int,
) -> EmbeddingResult:
    client = _get_client(region)
    body = json.dumps({"inputText": text})
    last_retryable_error: Optional[Exception] = None

    for attempt in range(retry_attempts):
        try:
            response = client.invoke_model(
                modelId=embedding_model.value,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            raw_body = response["body"].read().decode("utf-8")
            payload: BedrockEmbeddingPayload = json.loads(raw_body)
            vector = _extract_embedding_vector(payload)
            token_count = payload.get("inputTextTokenCount")
            if token_count is None:
                raise RuntimeError("Bedrock embedding response missing inputTextTokenCount")
            return EmbeddingResult(
                vector=vector,
                raw_api_response={
                    "ResponseMetadata": response.get("ResponseMetadata"),
                    "contentType": response.get("contentType"),
                    "inputTextTokenCount": token_count,
                    "embedding_dimensions": int(vector.shape[0]),
                },
                input_text_token_count=token_count,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in BEDROCK_RETRYABLE_CLIENT_ERROR_CODES or attempt == retry_attempts - 1:
                raise
            last_retryable_error = exc
            time.sleep(min(30.0, (2**attempt) * 0.5) + random.uniform(0.0, 0.25))
        except BotoCoreError as exc:
            if (
                type(exc).__name__ not in BOTOCORE_RETRYABLE_ERROR_NAMES
                or attempt == retry_attempts - 1
            ):
                raise
            last_retryable_error = exc
            time.sleep(min(30.0, (2**attempt) * 0.5) + random.uniform(0.0, 0.25))

    if last_retryable_error is not None:
        raise last_retryable_error
    raise RuntimeError("Bedrock embedding retry loop exited without a response")


class BedrockEmbeddingBackend(EmbeddingBackend):
    """Embedding backend that calls the AWS Bedrock invoke-model API."""

    def __init__(
        self,
        embedding_model: EmbeddingModelName,
        region: AwsRegion,
        retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS,
    ) -> None:
        self.embedding_model = embedding_model
        self.region = region
        self.retry_attempts = retry_attempts

    def embed_records_streaming(
        self,
        records: list[SampleRecord],
        *,
        writer: EmbeddingArtifactWriter,
        result_indices: Optional[list[int]] = None,
        max_concurrency: int,
    ) -> None:
        texts = [record.idea for record in records]
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
        stream_embed_texts_with_threadpool(
            texts,
            worker=lambda text: _embed_one_text(
                text,
                embedding_model=self.embedding_model,
                region=self.region,
                retry_attempts=self.retry_attempts,
            ),
            writer=writer,
            result_indices=result_indices,
            max_concurrency=max_concurrency,
        )
