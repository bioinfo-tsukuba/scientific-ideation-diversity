"""CSV-row schemas for compute_knn_purity.py output."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.model_registry import IdeaModelName


class KnnPurityGroupRow(BaseModel):
    """One row of per_group_metrics.csv (per keyword × embedding × facet × LLM)."""

    model_config = ConfigDict(extra="forbid")

    keyword: str
    embed_subdir: Path
    embed_label: str
    field: str
    idea_model: IdeaModelName
    n: int
    knn_purity: float
    within_dist: float
    between_dist: float
    within_between_ratio: float


class KnnPurityOverallRow(BaseModel):
    """One row of overall_metrics.csv (per keyword × embedding × facet)."""

    model_config = ConfigDict(extra="forbid")

    keyword: str
    embed_subdir: Path
    embed_label: str
    field: str
    n_total: int
    actual_k: int
    knn_purity_overall: float
    silhouette: float
