"""Diversity metric schemas for facet-level analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.model_registry import EffortName


class DiversitySummaryRow(BaseModel):
    """Aggregate diversity for one effort x field combination."""

    model_config = ConfigDict(extra="forbid")

    effort: EffortName
    field: str
    mean_pairwise_cosine_distance: float
    std: float


class PerKeywordDiversityRow(BaseModel):
    """Per-keyword diversity with one column per effort level."""

    model_config = ConfigDict(extra="allow")

    keyword: str
    field: str
