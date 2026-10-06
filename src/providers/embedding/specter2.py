"""SPECTER2 local adapter embedding backend."""

from __future__ import annotations

from typing import Optional

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from src.artifacts import EmbeddingArtifactWriter
from src.model_registry import DEFAULT_SPECTER2_BATCH_SIZE, EmbeddingModelName
from src.schemas.embedding import EmbeddingPayload
from src.schemas.sample import SampleRecord
from src.text import assemble_specter2_texts

from .base import EmbeddingBackend

_DEFAULT_MAX_SEQ_LENGTH = 512


class Specter2EmbeddingBackend(EmbeddingBackend):
    """Embedding backend that runs a SPECTER2 HuggingFace adapter model locally.

    ``embed_records_streaming`` assembles background + sep_token + idea
    formatting and delegates to ``embed_texts_streaming`` which batches
    the texts through the model.  ``max_concurrency`` is ignored because
    inference is sequential on the local device.
    """

    def __init__(
        self,
        adapter_model_id: EmbeddingModelName,
        base_model_id: EmbeddingModelName = EmbeddingModelName.SPECTER2_BASE,
        batch_size: int = DEFAULT_SPECTER2_BATCH_SIZE,
        max_seq_length: int = _DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        self.adapter_model_id = adapter_model_id
        self.base_model_id = base_model_id
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoAdapterModel] = None
        self._device: Optional[str] = None

    @property
    def is_adhoc(self) -> bool:
        return self.adapter_model_id.endswith("adhoc_query")

    def get_sep_token(self) -> str:
        """Return the tokenizer sep_token, loading the tokenizer if needed."""
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer.sep_token

    def _load_model(self) -> None:
        if self._model is not None:
            return

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id.value)
        model = AutoAdapterModel.from_pretrained(self.base_model_id.value)
        model.load_adapter(self.adapter_model_id.value, source="hf", load_as="task", set_active=True)
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def embed_records_streaming(
        self,
        records: list[SampleRecord],
        *,
        writer: EmbeddingArtifactWriter,
        result_indices: Optional[list[int]] = None,
        max_concurrency: int,
    ) -> None:
        self._load_model()
        texts = assemble_specter2_texts(
            records, sep_token=self.get_sep_token(), is_adhoc=self.is_adhoc
        )
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
        self._load_model()
        assert self._model is not None
        assert self._tokenizer is not None
        if result_indices is None:
            result_indices = list(range(len(texts)))

        fmt = "adhoc_query_concat" if self.is_adhoc else "background_sep_idea"

        for start in range(0, len(texts), self.batch_size):
            text_batch = texts[start : start + self.batch_size]
            index_batch = result_indices[start : start + self.batch_size]

            encoded = self._tokenizer(
                text_batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            seq_len: int = encoded["input_ids"].shape[1]
            if seq_len > self.max_seq_length:
                attention_mask = encoded.get("attention_mask")
                token_counts = (
                    attention_mask.sum(dim=1).tolist()
                    if attention_mask is not None
                    else [seq_len] * len(text_batch)
                )
                over = [
                    (start + i, int(n))
                    for i, n in enumerate(token_counts)
                    if int(n) > self.max_seq_length
                ]
                raise ValueError(
                    f"Input sequence length {seq_len} exceeds "
                    f"max_seq_length={self.max_seq_length}. "
                    f"Offending (global_index, token_count): {over}. "
                    "Shorten the texts or increase max_seq_length."
                )
            attention_mask = encoded.get("attention_mask")
            input_counts = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask is not None
                else [None] * len(text_batch)
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            with torch.no_grad():
                output = self._model(**encoded)
                batch_vectors = output.last_hidden_state[:, 0, :].detach().cpu().numpy()

            for row_idx, (global_index, vector) in enumerate(
                zip(index_batch, batch_vectors, strict=True)
            ):
                writer.write(
                    index=global_index,
                    vector=vector.astype(float),
                    payload=EmbeddingPayload(
                        raw_api_response={
                            "provider": "specter2",
                            "base_model_id": self.base_model_id.value,
                            "adapter_model_id": self.adapter_model_id.value,
                            "device": self._device,
                            "batch_start": start,
                            "format": fmt,
                        },
                        input_text_token_count=int(input_counts[row_idx]),
                    ),
                )

            completed = min(start + len(text_batch), len(texts))
            if completed % 25 == 0 or completed == len(texts):
                print(f"[embed] completed {completed}/{len(texts)}", flush=True)
