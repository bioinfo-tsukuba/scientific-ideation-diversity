"""Schema for per-facet embedding panels used by UMAP/PCA/kNN analyses.

Each entry is one generated sample's embedding for one schema field
(e.g. purpose/mechanism/evaluation), labelled with the keyword it was
generated for, the embedding model it was produced under, and the
generation (LLM) model / effort level that produced the sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.model_registry import EffortName, IdeaModelName


class FacetEmbeddingEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    keyword: str
    embed_subdir: Path
    field: str
    vector: np.ndarray
    effort: EffortName
    idea_model: IdeaModelName


class FacetEmbeddingData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entries: list[FacetEmbeddingEntry]
    keywords: list[str]
    efforts: list[EffortName]
    fields: list[str]
    embed_subdirs: list[Path]
    embed_labels: list[str]
    idea_models: list[IdeaModelName]

    def index_by_panel(self) -> dict[tuple[str, Path, str], list[FacetEmbeddingEntry]]:
        """Group entries by (keyword, embed_subdir, field) for O(1) panel lookup."""
        index: dict[tuple[str, Path, str], list[FacetEmbeddingEntry]] = {}
        for entry in self.entries:
            index.setdefault((entry.keyword, entry.embed_subdir, entry.field), []).append(entry)
        return index
